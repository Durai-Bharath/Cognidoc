"""Algorithm 3 - Document Classification Agent."""

from __future__ import annotations

import re
from statistics import mean

from common.base_agent import BaseAgent
from common.embedder import cosine
from common.text import keywords
from extraction.heuristics import caption_match, heading_match
from extraction.schemas import (
    ClassificationInput,
    ClassificationResult,
    DocumentFeatures,
    ParsedDocument,
)

UNKNOWN = "Unknown"
_TOC_LINE = re.compile(r"^(?P<title>.+?)\s*(?:\.{3,}|\s{3,}|…+)\s*(?P<page>\d{1,4})$")


class DocumentClassificationAgent(BaseAgent[ClassificationInput, ClassificationResult]):
    name = "DocumentClassificationAgent"
    algorithm = 3

    def _execute(self, data: ClassificationInput) -> ClassificationResult:
        cfg = self.ctx.config.classification
        parsed = data.parsed

        # Lines 2-10: document features.
        features = extract_features(parsed)
        feature_text = self._feature_text(features, parsed.raw_text, cfg.max_feature_chars)
        has_text = bool(parsed.raw_text.strip())

        # Lines 11-12: semantic embedding + feature vector F.
        feature_vector = [
            float(features.table_count),
            float(features.figure_count),
            float(len(features.headings)),
            *(features.layout_stats[k] for k in sorted(features.layout_stats)),
        ]
        if not has_text:
            return ClassificationResult(
                category=UNKNOWN,
                predicted_category=UNKNOWN,
                confidence=0.0,
                scores={c: 0.0 for c in cfg.class_descriptions},
                features=features,
                feature_vector=feature_vector,
                flagged=False,
                based_on_text=False,
            )
        names = list(cfg.class_descriptions)
        vectors = self.ctx.embedder.embed([feature_text, *cfg.class_descriptions.values()])
        doc_vec, class_vecs = vectors[0], vectors[1:]

        # Lines 13-29: Conf(C_i) = Similarity(F, C_i); C = argmax.
        scores = {
            name: round(cosine(doc_vec, vec), 6)
            for name, vec in zip(names, class_vecs, strict=True)
        }
        predicted = max(scores, key=lambda n: scores[n])
        conf = scores[predicted]

        # Lines 30-34: threshold.
        if conf < cfg.tau:
            self.flag(
                f"classification confidence {conf:.3f} below τ={cfg.tau}; category Unknown",
                item_ref=parsed.document_id,
                details={"predicted": predicted, "scores": scores},
            )
            category, flagged = UNKNOWN, True
        else:
            category, flagged = predicted, False

        # Lines 35-37: stored in metadata by the pipeline state / Metadata Fusion.
        return ClassificationResult(
            category=category,
            predicted_category=predicted,
            confidence=max(0.0, conf),
            scores=scores,
            features=features,
            feature_vector=feature_vector,
            flagged=flagged,
        )

    def confidence(self, output: ClassificationResult) -> float | None:
        return round(max(0.0, min(1.0, output.confidence)), 4)

    @staticmethod
    def _feature_text(features: DocumentFeatures, raw_text: str, budget: int) -> str:
        parts = [
            f"title: {features.title or ''}",
            "headings: " + "; ".join(features.headings[:40]),
            "keywords: " + ", ".join(features.keywords),
            raw_text[: budget // 2],
        ]
        return "\n".join(parts)[:budget]


def extract_features(parsed: ParsedDocument) -> DocumentFeatures:
    blocks = [b for p in parsed.pages for b in p.blocks]
    text_blocks = [b for b in blocks if b.kind == "text" and b.text]
    sizes = [b.font_size for b in text_blocks if b.font_size]
    body = sorted(sizes)[len(sizes) // 2] if sizes else None

    headings: list[str] = []
    toc: list[str] = []
    for b in text_blocks:
        first_line = b.text.split("\n")[0]
        style = (b.style or "").lower()
        is_heading = style.startswith("heading") or heading_match(b.text) is not None
        if not is_heading and body and b.font_size and b.font_size >= body * 1.15:
            is_heading = len(first_line.split()) <= 14 and "\n" not in b.text
        if is_heading:
            headings.append(first_line.strip())
        for line in b.text.split("\n"):
            m = _TOC_LINE.match(line.strip())
            if m:
                toc.append(m.group("title").strip())

    captions = [caption_match(b.text) for b in text_blocks]
    figure_caps = sum(1 for c in captions if c and c[0] == "figure")
    table_caps = sum(1 for c in captions if c and c[0] == "table")
    tables = sum(1 for b in blocks if b.kind == "table")
    figures = sum(1 for b in blocks if b.kind == "image")
    pages = max(1, len(parsed.pages))
    chars = [sum(len(b.text) for b in p.blocks) for p in parsed.pages] or [0]
    narrow = [b for b in text_blocks if b.bbox.w < 0.45 * _page_width(parsed, b.page)]
    props = parsed.properties
    return DocumentFeatures(
        title=props.title,
        table_of_contents=toc[:50],
        sections=[h for h in headings if not re.match(r"^\d+\.\d+", h)][:50],
        headings=headings[:100],
        keywords=keywords(parsed.raw_text, 15),
        metadata={
            k: str(v)
            for k, v in {
                "author": props.author,
                "language": props.language,
                "extension": props.extension,
                "pages": props.page_count,
                "created": props.creation_date,
            }.items()
            if v
        },
        table_count=max(tables, table_caps),
        figure_count=max(figures, figure_caps),
        layout_stats={
            "pages": float(pages),
            "blocks_per_page": len(blocks) / pages,
            "chars_per_page": float(mean(chars)),
            "narrow_block_ratio": len(narrow) / max(1, len(text_blocks)),
        },
    )


def _page_width(parsed: ParsedDocument, page: int) -> float:
    return next((p.width for p in parsed.pages if p.page_number == page), 612.0)
