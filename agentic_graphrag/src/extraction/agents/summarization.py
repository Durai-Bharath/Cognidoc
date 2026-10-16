"""Algorithm 17 - Text Summarization Agent."""

from __future__ import annotations

import itertools
import re

import numpy as np
from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.embedder import cosine
from common.llm import LLMError
from common.text import keywords, numbers_in, split_sentences
from common.validation import CheckResult, ValidationOutcome
from extraction.agents.context_memory import entities_in
from extraction.schemas import ElementType, SectionSummary, SummarizationInput

_DOMAIN_TERM = re.compile(r"\b(?:[A-Za-z]+[-_/][A-Za-z0-9]+|[A-Za-z]*\d+[A-Za-z]+|[A-Z]{2,}\w*)\b")


class SummaryLLM(BaseModel):
    summary: str = ""
    abstract: str = ""
    key_concepts: list[str] = []
    important_entities: list[str] = []


class TextSummarizationAgent(BaseAgent[SummarizationInput, SectionSummary]):
    name = "TextSummarizationAgent"
    algorithm = 17

    def _execute(self, data: SummarizationInput) -> SectionSummary:
        cfg = self.ctx.config.summarization
        section = data.task.section
        text = section.text()

        # Lines 4-6: clean, de-duplicate and rank sentences.
        body = "\n\n".join(
            e.text
            for e in section.all_elements()
            if e.type in (ElementType.PARAGRAPH, ElementType.LIST_ITEM)
        )
        sentences = self._clean_sentences(split_sentences(body))
        key_sentences = self._key_sentences(sentences, cfg.key_sentences)
        kw = keywords(text, 10)
        concepts = self._key_concepts(text)
        entities = entities_in(text)
        domain = list(dict.fromkeys(m.group(0) for m in _DOMAIN_TERM.finditer(text)))[:15]

        if not sentences:
            # Nothing to abstract over (e.g. a title-only front matter): keep the text itself.
            brief = self._truncate(" ".join(text.split()) or section.title, cfg.max_summary_words)
            out = SummaryLLM(summary=brief, abstract=brief)
        else:
            try:
                out = self.llm_json(
                    task="summarization",
                    system="You write faithful summaries of document sections. Use only facts "
                    "stated in the section; keep numbers exactly as written.",
                    prompt=(
                        f"Section title: {section.title}\nKeywords: {', '.join(kw)}\n"
                        f"Key sentences:\n- "
                        + "\n- ".join(key_sentences)
                        + f"\n\nFull section text:\n{text[:8000]}\n\n"
                        f"Write a summary of at most {cfg.max_summary_words} words, a one-"
                        "sentence abstract, key concepts and important entities."
                    ),
                    schema=SummaryLLM,
                )
            except LLMError as exc:
                self.flag(f"summary LLM call failed: {exc}", item_ref=section.section_id)
                out = SummaryLLM()

        summary = self._build(
            section.section_id,
            out,
            text,
            key_sentences,
            kw,
            concepts,
            entities,
            domain,
            used_fallback=False,
        )
        if not summary.validation.passed:
            # Line 34: regenerate with the extractive fallback strategy.
            fallback = SummaryLLM(
                summary=self._truncate(
                    " ".join(key_sentences) or section.title, cfg.max_summary_words
                ),
                abstract=key_sentences[0] if key_sentences else section.title,
                key_concepts=concepts,
                important_entities=entities[:10],
            )
            summary = self._build(
                section.section_id,
                fallback,
                text,
                key_sentences,
                kw,
                concepts,
                entities,
                domain,
                used_fallback=True,
            )
            if not summary.validation.passed:
                self.flag(
                    "summary failed validation after fallback",
                    item_ref=section.section_id,
                    details={"failures": summary.validation.failure_reasons()},
                )
        return summary

    def confidence(self, output: SectionSummary) -> float | None:
        return output.confidence

    # ------------------------------------------------------------------ helpers
    def _clean_sentences(self, sentences: list[str]) -> list[str]:
        cfg = self.ctx.config.summarization
        noisy = [
            s
            for s in sentences
            if len(s.split()) >= 4 and sum(c.isalpha() for c in s) / max(1, len(s)) >= 0.5
        ]
        if not noisy:
            return []
        vecs = self.ctx.embedder.embed(noisy)
        kept: list[int] = []
        for i in range(len(noisy)):
            if all(cosine(vecs[i], vecs[j]) < cfg.redundancy_similarity for j in kept):
                kept.append(i)
        return [noisy[i] for i in kept]

    def _key_sentences(self, sentences: list[str], k: int) -> list[str]:
        if len(sentences) <= k:
            return sentences
        vecs = self.ctx.embedder.embed(sentences)
        centroid = vecs.mean(axis=0)
        scores = [cosine(v, centroid) for v in vecs]
        top = sorted(np.argsort(scores)[::-1][:k])
        return [sentences[i] for i in top]

    @staticmethod
    def _key_concepts(text: str) -> list[str]:
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)]
        from common.text import STOPWORDS

        bigrams: dict[str, int] = {}
        for a, b in itertools.pairwise(words):
            if a not in STOPWORDS and b not in STOPWORDS:
                bigrams[f"{a} {b}"] = bigrams.get(f"{a} {b}", 0) + 1
        ranked = sorted(bigrams, key=lambda x: -bigrams[x])
        return [b for b in ranked if bigrams[b] > 1][:8] or keywords(text, 5)

    @staticmethod
    def _truncate(text: str, max_words: int) -> str:
        words = text.split()
        return text if len(words) <= max_words else " ".join(words[:max_words]) + " …"

    def _build(
        self,
        section_id: str,
        out: SummaryLLM,
        text: str,
        key_sentences: list[str],
        kw: list[str],
        concepts: list[str],
        entities: list[str],
        domain: list[str],
        *,
        used_fallback: bool,
    ) -> SectionSummary:
        cfg = self.ctx.config.summarization
        summary_text = out.summary.strip()
        sim = self.ctx.embedder.similarity(summary_text, text) if summary_text and text else 0.0
        top = kw[:5]
        coverage = (sum(k in summary_text.lower() for k in top) / len(top)) if top else 1.0
        n_words = len(summary_text.split())
        length_ok = 1.0 if 0 < n_words <= cfg.max_summary_words else 0.0
        quality = round(max(0.0, 0.5 * sim + 0.25 * coverage + 0.25 * length_ok), 4)

        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures(
                "summary_completeness",
                (
                    ["summary is empty"]
                    if not summary_text
                    else [f"keyword coverage {coverage:.2f} < {cfg.min_keyword_coverage}"]
                    if coverage < cfg.min_keyword_coverage
                    else []
                ),
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "context_preservation",
                (
                    [f"similarity to section {sim:.2f} < {cfg.min_context_similarity}"]
                    if text and sim < cfg.min_context_similarity
                    else []
                ),
            )
        )
        unsupported = sorted(numbers_in(summary_text) - numbers_in(text))
        outcome.add(
            CheckResult.from_failures(
                "factual_consistency",
                ([f"numbers not in source: {unsupported}"] if unsupported else []),
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "length_constraints",
                ([f"{n_words} words (limit {cfg.max_summary_words})"] if not length_ok else []),
            )
        )
        return SectionSummary(
            section_id=section_id,
            summary=summary_text,
            abstract=out.abstract.strip() or (key_sentences[0] if key_sentences else summary_text),
            keywords=kw,
            key_concepts=out.key_concepts or concepts,
            important_entities=out.important_entities or entities[:10],
            domain_terms=domain,
            key_sentences=key_sentences,
            summary_length=n_words,
            quality_score=quality,
            confidence=quality,
            used_fallback=used_fallback,
            validation=outcome,
        )
