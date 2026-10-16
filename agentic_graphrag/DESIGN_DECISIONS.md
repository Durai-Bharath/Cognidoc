# Design decisions (Stage 1)

This records how ambiguities in *Agentic Document Extraction Framework* (the PDF spec)
were resolved, plus the engineering choices a reviewer is likely to question. Items 1–4
were decided in the task brief. Items 5 onward came up during implementation.

## Decided in the brief

1. **Alg. 15 forwards to an undefined "Relationship Extraction Agent".** The entity
   repository goes to Metadata Fusion (Alg. 20) instead. Relationship extraction belongs
   to Stage 2.
2. **"Figure Analysis Agent" (Alg. 9) = "Figure Understanding Agent" (Alg. 11).** One
   class, `FigureUnderstandingAgent`, also exported as the alias `FigureAnalysisAgent`.
3. **Alg. 1's OCR if/else.** Read as: *if the document requires OCR*, run OCR Quality,
   and run Image Enhancement only when quality < τ. Quality ≥ τ skips enhancement.
   Native-text PDFs and DOCX skip the OCR branch entirely.
4. **Confidence Scoring omits Classification and Abbreviation scores.** Both are accepted
   as components with default weight `0` in `confidence.weights`. Set a weight > 0 to
   include them.

## Found during implementation

5. **Per-section scope of Alg. 13–19.** In Alg. 1, lines 21–27 sit inside
   `foreach section S_i`, but Alg. 13–17 also loop over sections internally. Each agent
   therefore runs once per top-level section on a `SectionTask`, which carries `S_i`
   *and* the full hierarchy `S`, so citations can resolve targets anywhere in the
   document. Document-wide checks accumulate across iterations:
   - Cross-reference *orphan objects* are evaluated only on the last section, against
     all references collected so far. Earlier sections can't know whether a later
     section cites an object.
   - *Circular references* are evaluated on the accumulated section→section graph.
   - Abbreviations are one document-wide repository, threaded through the loop.
   - Document-level confidence is the token-weighted mean of all section scores so far.
     Fusion recomputes it over every section.
6. **Who runs OCR text recognition?** The spec never says. Router's "Invoke OCR
   Pipeline" loads or rasterises page images. Alg. 4 runs recognition, since OCR
   confidence is one of its metrics, and Alg. 5 re-runs it on enhanced images and
   replaces the text when ΔC > 0.
7. **"Requires OCR" is decided per page.** Images always need it. A PDF page with fewer
   than `router.min_native_text_chars` native characters is rasterised at
   `ocr.render_dpi` and OCR'd, so mixed native/scanned PDFs work.
8. **Classification runs before OCR (Alg. 1 line 3).** An image has no text at that
   point. The agent returns `Unknown` with `based_on_text=false` and does **not** flag
   it. After the OCR branch the graph runs the same agent again (node `reclassify`).
   This is the only addition to the 1:1 node mapping.
9. **Error vs warning checks.** Some listed checks hold for perfectly valid documents,
   and making them hard failures would flag nearly every document. These are
   *warnings*: they are reported in the outputs but never fail an outcome or send it to
   review on their own.
   - Cross-ref orphan objects, circular references and duplicate references.
   - Undefined, unused and duplicate abbreviations.
   - Table and figure captions.
   - Empty sections and orphan sections.
   - Keyword mismatch in Global Validation.

   Everything else is an error.
10. **Provider defaults.** The brief asks for Anthropic and sentence-transformers as
    defaults, and those are the defaults in code (`common/config.py`):
    - `claude-opus-5-5` via the official SDK, with structured outputs
      (`output_config.format` + JSON schema) and `effort: medium`.
    - Server-side refusal fallbacks (`fallbacks: "default"`), which can be turned off
      with `llm.anthropic_fallbacks`.

    The repository's `config.yaml` selects **Azure OpenAI** chat and **OpenRouter**
    embeddings instead, because those are the providers this project is configured for.
    Switching back takes two lines of YAML.
11. **Retries.** `LLMClient` retries transient provider errors and invalid or truncated
    JSON, feeding the validation error back to the model, with exponential backoff.
    SDK-level retries are disabled so there is one retry policy. Embedders also retry
    with backoff: a live run showed OpenRouter's free embedding tier returning 429s
    after 20 requests per minute.
12. **Layout model.** The default `HeuristicLayoutModel` uses parser/OCR geometry, font
    statistics and text patterns. It is deterministic, needs no downloads, and keeps
    `make test` offline. It sits behind the `LayoutModel` interface. `DoclingLayoutModel`
    (`layout.backend: docling`) is an adapter for IBM docling; it is not covered by the
    offline tests because it downloads models. OCR pages get table detection from
    column-gap alignment across consecutive lines (`ocr.words_to_blocks`).
13. **Image-quality metrics (Alg. 4).** Sharpness (Laplacian variance) and blur
    (high-frequency energy) are measured on a lightly denoised copy, otherwise pixel
    noise reads as "sharp". Contrast is the paper-minus-ink mean grey level from an Otsu
    split, which suits mostly-white pages better than global standard deviation.
    Normalisation references are calibrated so that a clean 150-dpi scan scores ≈0.95
    and a blurred, noisy, washed-out one ≈0.55 (τ_quality = 0.6).
14. **Super-resolution is pluggable.** `SuperResolution` interface: bicubic by default,
    OpenCV `dnn_superres` with a model file, or any object in
    `ctx.services["super_resolution"]`.
15. **Hallucination similarity.** `Similarity(g_i, D_i)` is the maximum of the cosine to
    the whole source and to its best-matching paragraph, so short generated texts aren't
    penalised against long sections. Regeneration uses the LLM with the detected issues,
    up to `hallucination.max_regenerations` attempts, keeping the best-scoring text. If
    no attempt reaches HScore ≤ τ, the item is `Flagged` and goes to the review queue.
    Verified regenerated texts replace the originals before fusion. Checked texts:
    section summaries and abstracts, and table, figure, equation and entity descriptions.
16. **Missing confidence components.** A component that doesn't apply (e.g. `table` for a
    section with no tables) is excluded and the remaining weights are renormalised.
    Missing *required* components (`confidence.required_components`) fail validation.
    Invalid values (NaN or outside [0, 1]) trigger the "recompute inconsistent scores"
    branch with sanitised inputs, and stay flagged.
17. **Export always writes M and J (Alg. 20 lines 33–38).** When schema validation fails,
    `export_approved=false`, the failure is flagged, and Global Validation reports it.
    The files are still written so a human or Stage 2 can inspect them. Approval lives
    in `validation_report.json`.
18. **Fatal vs non-fatal agent failures.** Router, OCR Quality, Layout, Reading Order,
    Section Identification and Fusion produce data everything downstream needs. If one
    fails, the run ends with a failure `validation_report.json`. Any other agent failure
    is recorded in `agent_errors`, the pipeline continues, and Global Validation refuses
    approval and recommends re-running that agent.
19. **Token counting** uses an offline approximation (word pieces + punctuation) instead
    of a provider tokenizer, so `context_limit` behaves the same offline and with any
    provider.
20. **Section granularity.** The loop iterates top-level sections. Subsections are
    processed with their parent and appear nested in J. Content before the first heading
    (title, unlabelled abstract) becomes the `sec-0 Front Matter` section, so there is
    never orphan content.
21. **Object numbering** ("Figure 3", "Table 2") comes from caption association in
    Alg. 7 and is what citations (Alg. 13) resolve against.
22. **Stage 2 contract.** `schemas/*.schema.json` are generated from
    `extraction/output_models.py` (`make schemas`) and versioned (`schema_version
    1.0.0`). A test fails if the published files drift from the models.
23. **Test doubles.** Tests run fully offline with `MockLLM` (prompt-aware, deterministic
    answers from `extraction.testing.offline_llm`), `FakeEmbedder` (hashed
    bag-of-words) and `MockOCREngine`. The mock OCR places known text lines onto the
    ink bands of a synthetic scan, and its confidence tracks image sharpness and
    contrast, so enhancement changes it in a measurable way.
24. **Tesseract** is the default real OCR engine. It is an external binary: install it
    with `sudo apt install tesseract-ocr` and `pip install -e ".[ocr]"`. Without it,
    image documents fail at Alg. 4 with a clear error.
