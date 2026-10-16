"""Algorithm 18 - Hallucination Detection Agent."""

from __future__ import annotations

import re

from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.embedder import cosine
from common.llm import LLMError
from common.text import DATE_RE, keywords, numbers_in, split_sentences
from extraction.schemas import (
    GeneratedItem,
    HallucinationInput,
    HallucinationItem,
    HallucinationReport,
)

_PROPER = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:[-\s][A-Z][a-zA-Z0-9]+)*\b")
_SENTENCE_START = {
    "The",
    "This",
    "These",
    "It",
    "A",
    "An",
    "In",
    "Section",
    "Table",
    "Figure",
    "Equation",
    "Summary",
    "Overall",
    "Each",
    "All",
    "We",
    "They",
}


class RegeneratedText(BaseModel):
    text: str = ""


class HallucinationDetectionAgent(BaseAgent[HallucinationInput, HallucinationReport]):
    name = "HallucinationDetectionAgent"
    algorithm = 18

    def similarity(self, generated: str, source: str) -> float:
        """Similarity(g_i, D_i): best of whole-source and best-paragraph cosine."""
        windows = [p for p in source.split("\n\n") if p.strip()] or [source]
        vecs = self.ctx.embedder.embed([generated, source, *windows[:50]])
        g = vecs[0]
        return max(cosine(g, v) for v in vecs[1:])

    def issues(self, item: GeneratedItem, text: str) -> list[str]:
        """Lines 6-14: factual consistency, unsupported statements, omissions."""
        cfg = self.ctx.config.hallucination
        src = item.source
        found: list[str] = []
        extra_numbers = sorted(numbers_in(text) - numbers_in(src))
        if extra_numbers:
            found.append(f"numbers not in source: {extra_numbers}")
        extra_dates = sorted(set(DATE_RE.findall(text)) - set(DATE_RE.findall(src)))
        if extra_dates:
            found.append(f"dates not in source: {extra_dates}")
        src_lower = src.lower()
        unknown = sorted(
            {
                m.group(0)
                for m in _PROPER.finditer(text)
                if m.group(0) not in _SENTENCE_START and m.group(0).lower() not in src_lower
            }
        )
        if unknown:
            found.append(f"entities not in source: {unknown[:10]}")
        windows = [p for p in src.split("\n\n") if p.strip()] or [src]
        if len(windows) > 50:
            windows = windows[:50]
        sentences = split_sentences(text)
        if sentences and windows:
            vecs = self.ctx.embedder.embed([*sentences, *windows])
            s_vecs, w_vecs = vecs[: len(sentences)], vecs[len(sentences) :]
            for sent, sv in zip(sentences, s_vecs, strict=True):
                if max(cosine(sv, wv) for wv in w_vecs) < cfg.unsupported_sentence_similarity:
                    found.append(f"unsupported statement: '{sent[:80]}'")
        if item.kind in ("summary", "abstract"):
            critical = keywords(src, 3)
            if critical and not any(k in text.lower() for k in critical):
                found.append(f"omits critical terms {critical}")
        return found

    def _regenerate(self, item: GeneratedItem, text: str, issues: list[str]) -> str | None:
        try:
            out = self.llm_json(
                task="hallucination_regeneration",
                system="You rewrite generated text so that every statement is supported by the "
                "SOURCE. Remove anything not in the source. Keep it concise.",
                prompt=(
                    f"Kind: {item.kind}\nIssues found: {issues}\n\nGENERATED:\n{text}\n\n"
                    f"SOURCE:\n{item.source[:8000]}"
                ),
                schema=RegeneratedText,
            )
        except LLMError as exc:
            self.log.warning(
                "regeneration_failed", extra={"item": item.content_id, "error": str(exc)}
            )
            return None
        return out.text.strip() or None

    def _execute(self, data: HallucinationInput) -> HallucinationReport:
        cfg = self.ctx.config.hallucination
        report = HallucinationReport(section_id=data.section_id)
        for item in data.items:
            if not item.text.strip():
                continue
            text = item.text
            sim = self.similarity(text, item.source)
            hscore = round(1 - sim, 4)  # HScore = 1 − Similarity(g_i, D_i)
            found = self.issues(item, text)
            if hscore <= cfg.tau:
                report.items.append(
                    HallucinationItem(
                        content_id=item.content_id,
                        kind=item.kind,
                        status="Verified",
                        similarity=round(sim, 4),
                        hscore=hscore,
                        issues=found,
                        final_text=text,
                    )
                )
                continue
            # Lines 20-21: flag and request regeneration (bounded retries).
            best = (hscore, text, sim, found)
            attempts = 0
            for attempt in range(1, cfg.max_regenerations + 1):
                attempts = attempt
                new = self._regenerate(item, best[1], best[3])
                if new is None:
                    break
                new_sim = self.similarity(new, item.source)
                new_h = round(1 - new_sim, 4)
                if new_h < best[0]:
                    best = (new_h, new, new_sim, self.issues(item, new))
                if best[0] <= cfg.tau:
                    break
            status = "Regenerated" if best[0] <= cfg.tau else "Flagged"
            if status == "Flagged":
                self.flag(
                    f"potential hallucination in {item.kind} (HScore {best[0]:.3f} > τ={cfg.tau})",
                    item_ref=item.content_id,
                    details={"issues": best[3], "regenerations": attempts},
                )
            report.items.append(
                HallucinationItem(
                    content_id=item.content_id,
                    kind=item.kind,
                    status=status,
                    similarity=round(best[2], 4),
                    hscore=best[0],
                    issues=best[3],
                    regenerations=attempts,
                    final_text=best[1],
                )
            )
        return report

    def confidence(self, output: HallucinationReport) -> float | None:
        if not output.items:
            return None
        return round(sum(1 - min(1.0, i.hscore) for i in output.items) / len(output.items), 4)
