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

---

# Design decisions (Stage 2: graph construction)

How ambiguities in *Construction.pdf* (Algorithms 1–16, agents A1–A15) were resolved.
Items 1–3 were decided in the task brief. Items 4 onward came up during implementation.
Code lives in `src/construction/`, tests in `tests/construction/`.

## Decided in the brief

1. **Where A11–A15 run relative to the chunk loop.** Alg. 1 indents lines 24–26
   (Provenance, Embedding, Vector Index) inside the loop and lines 27–28 (Validation,
   Audit) after it. A11–A13 therefore run once per **accepted** chunk, on the nodes and
   edges that chunk touched. A14–A15 run once after all chunks, on the whole KG and VDB.
   Skipped chunks (lines 8–10) produce nothing: no Chunk node, no vectors.
2. **Conflict collapses to Reject (Alg. 10 line 11), but is never silently discarded.**
   For every Conflict *and* Reject, the Decision Agent writes a `ConflictRecord`. It holds
   the incoming triple with its provenance, the existing edges it clashes with and their
   provenance, and the reason. Records go to the review queue (`agent: decision`) and
   into `construction_report.json` under `conflicts`. A10 discards the triple.
3. **Community summaries for Stage 3.** `CommunityBuilder` (`src/construction/community.py`)
   runs after A15. It applies Louvain (networkx, fixed seed) to the confidence-weighted
   undirected entity graph, or Leiden with `community.algorithm: leiden` when `igraph` and
   `leidenalg` are installed (otherwise it falls back to Louvain). Each community of at
   least `min_size` members gets a `Community` node with an LLM summary (extractive
   fallback), `MEMBER_OF` edges from its members, and an embedding in the `communities`
   collection. *Only changed communities are rebuilt.* A community's id is a hash of its
   sorted members, and its signature is a hash of the member names and internal edges.
   An unchanged id and signature means no LLM call and no re-embedding. Communities that
   disappear are deleted along with their vectors. A14 and A15 run before this step, so
   on the *next* run they also validate Community nodes and vectors.

## Found during implementation

4. **Merge vs Update.** The PDF names both but never separates them. The rule:
   - The edge `(subject, predicate, object)` already exists *and* has a provenance record
     for the same `(doc_id, chunk_id)` → **Update**. Only the confidence and provenance
     timestamp change (Alg. 1 line 21).
   - The edge exists and the source is new → **Merge**. A provenance record is appended
     and the endpoints' aliases and provenance are unioned.

   Edge confidence is the max over its sources, and `support` is the number of distinct
   sources. This is what makes re-ingestion idempotent: the same document yields only
   Updates, with no new nodes, edges, vectors or provenance records
   (`test_reingesting_the_same_document_is_idempotent`).
5. **What counts as a Conflict.** A new edge conflicts when:
   - its predicate is `functional: true` in the ontology (`has_interval`, `has_value`,
     `has_status`) and the subject already has a *different* object for it; or
   - an `exclusive` predicate pair (`requires`/`prohibits`, `causes`/`prevents`) links the
     same subject and object.

   Both rules check the KG and earlier triples in the same chunk. **Reject** covers: no
   ontology predicate (A7), a self-loop after entity resolution, or a domain/range
   violation.
6. **A6 proposes, A10 writes.** Entity Resolution returns an `EntityRef` per mention:
   either an existing node id, or a deterministic new id (a hash of the normalised name).
   Only Graph CRUD creates nodes, and only on Create/Merge. Rejected or conflicting
   triples therefore leave the entity graph exactly as it was
   (`test_contradictory_triples_across_documents_are_logged_and_graph_unchanged`).
   The incoming document's Document node and accepted Chunk nodes are still registered,
   because A1 and A11 always run.
7. **Entity resolution order.** First an exact canonical-id match (similarity 1.0). Then
   `GraphStore.similar_nodes` over Entity embeddings, where cosine ≥ δ (0.85) with
   compatible types links. Then earlier *new* mentions in the same chunk, so one chunk
   never proposes two near-duplicates. Entities are embedded by **name only** (A12), the
   same text A6 compares, so lookups stay consistent.
8. **Type `Concept` is the fallback type and a wildcard.** An LLM type that maps to no
   ontology type or synonym becomes `Concept`. `Concept` satisfies every domain/range
   constraint and links with any type in A6. A linked `Concept` node takes the more
   specific type of a later mention.
9. **A5 "Evaluate confidence".** The evaluated confidence is the LLM confidence,
   multiplied by `ungrounded_penalty` (0.6) when the head or tail cannot be found in the
   chunk text or its overlap context. Grounding matches the name, the original mention
   or an abbreviation, verbatim or token by token, with numbers kept and unit letters
   ignored ("500 h" is grounded by "500"). Triples with evaluated confidence ≥ τ (0.5)
   survive. The Alg. 1 line 8 test "Confidence Score < τ" is read at chunk level as *no
   surviving triple*.
10. **A10 writes provenance; A11 completes and verifies it.** Merge and Update are defined
    in terms of provenance, so the CRUD operation must store the triple's provenance
    record. A11 then:
    - registers the Chunk node (text, section path, pages) with `IN_DOCUMENT` and
      `NEXT_CHUNK` edges;
    - adds `MENTIONED_IN` edges from every touched entity;
    - fills a missing page, source reference or timestamp from the chunk;
    - fails validation (and flags) when a touched element still lacks a complete record
      for this source.
11. **Document ID.** `doc-` + the first 16 hex characters of the sha256 of the canonical
    JSON of J, excluding the volatile `processing_timestamp` and `source_file`. Re-running
    Stage 1 on the same file therefore yields the same id. The Stage 1 id is kept as the
    Document property `source_document_id`. A1 also requires `M.document_id ==
    J.document_id` (`ingestion.require_metadata_match`).
12. **Chunking reuses Stage 1.** Stage 1's Context Memory agent already makes section-aware
    semantic chunks. A2 reuses them and adds:
    - document-scoped ids (`<doc_id>:<stage1 chunk id>`);
    - overlap (the tail of the previous chunk in the same section, up to
      `overlap_tokens`), passed as `context` and used only for coreference and grounding,
      so facts are never extracted twice;
    - `section_path`, including the first subsection whose title appears in the chunk;
    - sentence-window splits for oversized chunks and for sections with no chunks;
    - one linearised chunk per table (`Header: value; …` per row), because tables carry
      dense facts that pipe-separated text hides.
13. **Coreference is rule-based and deterministic.** It resolves:
    - pronouns to the last explicit subject in the chunk, else the previous chunk's
      subject, else the triple is dropped;
    - "this report" and similar to the document title;
    - abbreviations through Stage 1's repository in J (whole names, embedded tokens,
      plurals, and the "Expansion (ABBR)" form);
    - duplicate mentions (case, articles, plurals) to the most frequent form, with ties
      going to the shortest, i.e. the singular.

    An LLM coreference pass would be a drop-in replacement behind the same agent.
14. **Relation normalisation order.** The relation is tried as an exact predicate name,
    then a synonym, then an inverse synonym (which swaps head and tail, e.g. "contains" →
    `part_of`), then a stemmed match. Only after that does the LLM fallback run, choosing
    a listed predicate or `NONE`; its answers are cached per phrase for the run. An answer
    not in the ontology counts as `NONE`. The default ontology deliberately has no generic
    `related_to`, which would absorb every unmatched relation and make Reject
    unreachable. The ontology is validated at startup (snake_case predicates, known types
    in domain/range, no synonym mapping to two predicates, known exclusive pairs, the
    `Concept` fallback present); an invalid file raises `OntologyError` before any
    document is read.
15. **Graph model.** Node labels: `Document`, `Chunk`, `Entity`, `Community`. Structural
    predicates: `IN_DOCUMENT`, `NEXT_CHUNK`, `MENTIONED_IN`, `MEMBER_OF`. All other edges
    are ontology predicates between Entities. Edge id = hash(source, predicate, target),
    so every write is an idempotent upsert. A14 checks that structural edges connect the
    right labels.
16. **Failure isolation.** Every per-chunk node (A3–A13) is wrapped. Two kinds of failure
    are caught: a typed `AgentError` (e.g. the LLM failing after retries) and an unexpected
    exception inside the node. Either one:
    - marks the chunk `failed`, with its stage and error;
    - flags it to the review queue;
    - routes back to `next_chunk`.

    Graph CRUD isolates failures per *operation*: one failing triple is logged in
    `agent_errors`, and the chunk's other triples still apply. Only A1 and A2 are fatal.
    They end the run with a `rejected` construction report and no KG export.
    Limitation: if a chunk fails *after* CRUD (in A11–A13), its edges stay in the graph
    without full provenance or vectors. A14 reports exactly that, rather than rolling
    back, because the in-memory and Neo4j backends have no shared transaction.
17. **Stale relationships (A15).** An edge is stale when none of its provenance records
    points at a Document and Chunk still in the KG, or when its newest source timestamp
    is older than `audit.stale_after_days` (365). Duplicates are entity pairs with the
    same mention key, or with embedding cosine ≥ `audit.duplicate_similarity` (0.95).
    `health_score` is the mean of (1 − orphan ratio, 1 − duplicate ratio, 1 − stale ratio,
    1 − low-confidence ratio, largest-component ratio, ontology-compliance ratio).
18. **Several disconnected entity components only warn** (`single_entity_component`).
    A KG built from several unrelated documents is naturally disconnected. Hard
    connectivity requirements are that every Chunk is `IN_DOCUMENT` and every Entity is
    `MENTIONED_IN` some chunk.
19. **Vector verification.** A13 reads every vector back. It must be present, map to the
    right node (`payload.node_id`), and be equal or have cosine ≥ 0.999 (backends may
    re-normalise). A text with no content words embeds to a zero vector, which has no
    direction, so equality is accepted for it. A verification failure is a typed
    `VectorIndexError`.
20. **Backends.** `GraphStore` has `InMemoryGraphStore` (networkx `MultiDiGraph`, JSON
    persistence) and `Neo4jGraphStore`, which stores nested data as JSON strings and uses
    native vector indexes per label (Neo4j's cosine score (1+cos)/2 is converted back).
    `VectorStore` has `InMemoryVectorStore` and `QdrantVectorStore` (string ids mapped to
    uuid5, original id in the payload). Chroma was not implemented; the brief allowed
    either. The CLI persists in-memory state to `paths.state_dir`, so successive
    `construct` runs are incremental against the same KG/VDB.
21. **Offline test doubles.** `construction.testing.offline_construction_llm()` is a
    prompt-aware `MockLLM`. Triples come from simple verb patterns and linearised table
    rows in the TEXT block. Relation fallback strips modal verbs and retries the
    ontology. Community summaries are built from the listed facts. It is deterministic,
    not good, and exists to drive every branch. `make_document()` builds schema-valid
    J/M pairs for tests.
