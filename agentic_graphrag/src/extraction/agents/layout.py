"""Algorithm 6 - Layout Detection Agent."""

from __future__ import annotations

from itertools import combinations

from common.base_agent import BaseAgent
from common.config import LayoutConfig
from common.validation import CheckResult, ValidationOutcome
from extraction.layout_models import LayoutModel, build_layout_model
from extraction.schemas import (
    FURNITURE,
    ElementType,
    LayoutElement,
    LayoutInput,
    LayoutMap,
    PageLayout,
)

_CAPTION_TARGET = {
    "figure": ElementType.FIGURE,
    "table": ElementType.TABLE,
    "algorithm": ElementType.ALGORITHM,
    "listing": ElementType.ALGORITHM,
}


def validate_page_layout(page: PageLayout, cfg: LayoutConfig) -> ValidationOutcome:
    """Alg. 6 lines 43-48."""
    content = [e for e in page.elements if e.type not in FURNITURE]
    outcome = ValidationOutcome()

    overlaps = [
        f"{a.element_id} overlaps {b.element_id} (IoU {a.bbox.iou(b.bbox):.2f})"
        for a, b in combinations(content, 2)
        if a.bbox.iou(b.bbox) > cfg.overlap_iou_threshold
    ]
    outcome.add(CheckResult.from_failures("no_overlapping_regions", overlaps))

    tol = 2.0
    invalid = [
        f"{e.element_id} bbox outside page or degenerate"
        for e in page.elements
        if e.bbox.w <= 0
        or e.bbox.h <= 0
        or e.bbox.x < -tol
        or e.bbox.y < -tol
        or e.bbox.x1 > page.width + tol
        or e.bbox.y1 > page.height + tol
    ] + [
        f"{e.element_id} text region is empty"
        for e in page.elements
        if e.type not in (ElementType.FIGURE, ElementType.TABLE) and not e.text.strip()
    ]
    outcome.add(CheckResult.from_failures("valid_reading_regions", invalid))

    max_gap = cfg.caption_max_gap_ratio * page.height
    caption_failures: list[str] = []
    for cap in (e for e in page.elements if e.type == ElementType.CAPTION):
        target = _CAPTION_TARGET.get(str(cap.attrs.get("label_kind", "")))
        if target is None:
            continue
        candidates = [
            e
            for e in page.elements
            if e.type == target
            and e.bbox.h_overlap(cap.bbox) > 0
            and min(abs(cap.bbox.y - e.bbox.y1), abs(e.bbox.y - cap.bbox.y1)) <= max_gap
        ]
        if not candidates:
            caption_failures.append(f"{cap.element_id} ('{cap.text[:40]}') has no aligned {target}")
    outcome.add(CheckResult.from_failures("caption_alignment", caption_failures))

    def object_alignment(kind: ElementType) -> list[str]:
        failures: list[str] = []
        multi = len(page.columns) >= 2
        for e in (x for x in page.elements if x.type == kind):
            if e.bbox.x < -tol or e.bbox.x1 > page.width + tol:
                failures.append(f"{e.element_id} extends beyond the page")
            if multi and e.column is not None:
                lo, hi = page.columns[e.column]
                if e.bbox.x < lo - 0.1 * page.width or e.bbox.x1 > hi + 0.1 * page.width:
                    failures.append(f"{e.element_id} straddles column {e.column}")
            if kind == ElementType.TABLE:
                cells = e.attrs.get("cells") or []
                if not cells or not any(cells):
                    failures.append(f"{e.element_id} has no cells")
        return failures

    outcome.add(CheckResult.from_failures("table_alignment", object_alignment(ElementType.TABLE)))
    outcome.add(CheckResult.from_failures("figure_alignment", object_alignment(ElementType.FIGURE)))
    return outcome


class LayoutDetectionAgent(BaseAgent[LayoutInput, LayoutMap]):
    name = "LayoutDetectionAgent"
    algorithm = 6

    def _model(self) -> LayoutModel:
        model = self.ctx.services.get("layout_model")
        if isinstance(model, LayoutModel):
            return model
        model = build_layout_model(self.ctx.config.layout)
        self.ctx.services["layout_model"] = model
        return model

    def _execute(self, data: LayoutInput) -> LayoutMap:
        cfg = self.ctx.config.layout
        pages, body = self._model().detect(data.parsed)
        for page in pages:
            for order, el in enumerate(page.elements):
                el.attrs.setdefault("detection_order", order)
            page.validation = validate_page_layout(page, cfg)
            page.approved = page.validation.passed
            if not page.approved:
                self.flag(
                    f"layout validation failed on page {page.page}",
                    item_ref=f"page:{page.page}",
                    details={"failures": page.validation.failure_reasons()},
                )
        return LayoutMap(pages=pages, body_font_size=body)

    def confidence(self, output: LayoutMap) -> float | None:
        return round(output.confidence, 4)


def element_index(layout: LayoutMap) -> dict[str, LayoutElement]:
    return {e.element_id: e for e in layout.all_elements()}
