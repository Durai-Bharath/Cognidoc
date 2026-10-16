"""Layout models behind one interface.

``HeuristicLayoutModel`` (default) works on parser/OCR blocks using geometry, font
statistics and text patterns, so it runs offline with no model download.
``DoclingLayoutModel`` adapts IBM docling's layout model when it is installed.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import ClassVar

from common.config import LayoutConfig
from extraction.heuristics import (
    EQUATION_NUMBER,
    LIST_ITEM,
    NAMED_SECTIONS,
    PAGE_NUMBER,
    REFERENCE_HEADINGS,
    caption_match,
    dehyphenate_lines,
    heading_depth,
    heading_match,
    looks_like_equation,
)
from extraction.schemas import (
    FURNITURE,
    BBox,
    ElementType,
    LayoutElement,
    PageData,
    PageLayout,
    ParsedDocument,
)


class LayoutModel(ABC):
    name: str

    @abstractmethod
    def detect(self, parsed: ParsedDocument) -> tuple[list[PageLayout], float]:
        """Return per-page layouts (elements, columns, layout type) and the body font size."""


# =========================================================================== geometry helpers


def body_font_size(parsed: ParsedDocument) -> float:
    weighted: Counter[float] = Counter()
    for page in parsed.pages:
        for b in page.blocks:
            if b.kind == "text" and b.font_size:
                weighted[round(b.font_size, 1)] += len(b.text)
    if not weighted:
        return 11.0
    return float(weighted.most_common(1)[0][0])


def detect_columns(
    elements: list[LayoutElement], width: float, full_width_ratio: float
) -> tuple[str, list[tuple[float, float]]]:
    """Cluster narrow body elements by x-centre → single / double / multi column."""
    body_types = {
        ElementType.PARAGRAPH,
        ElementType.LIST_ITEM,
        ElementType.SECTION_HEADING,
        ElementType.SUBSECTION_HEADING,
        ElementType.EQUATION,
        ElementType.REFERENCE,
        ElementType.CAPTION,
        ElementType.TABLE,
        ElementType.FIGURE,
    }
    body = [e for e in elements if e.type in body_types]
    if not body:
        return "single", [(0.0, width)]
    narrow = [e for e in body if e.bbox.w < full_width_ratio * width]
    if len(narrow) < 2 or len(narrow) / len(body) < 0.4:
        lo = min(e.bbox.x for e in body)
        hi = max(e.bbox.x1 for e in body)
        return "single", [(lo, hi)]
    narrow.sort(key=lambda e: e.bbox.cx)
    clusters: list[list[LayoutElement]] = [[narrow[0]]]
    for e in narrow[1:]:
        if e.bbox.cx - clusters[-1][-1].bbox.cx > 0.15 * width:
            clusters.append([e])
        else:
            clusters[-1].append(e)
    intervals = [(min(e.bbox.x for e in c), max(e.bbox.x1 for e in c)) for c in clusters]
    if len(intervals) == 1:
        return "single", intervals
    return ("double" if len(intervals) == 2 else "multi"), intervals


def assign_columns(
    elements: list[LayoutElement],
    columns: list[tuple[float, float]],
    width: float,
    full_width_ratio: float,
) -> None:
    for e in elements:
        if e.type in FURNITURE or e.bbox.w >= full_width_ratio * width or len(columns) < 2:
            e.column = None if len(columns) >= 2 else 0
            continue
        overlaps = [max(0.0, min(e.bbox.x1, hi) - max(e.bbox.x, lo)) for lo, hi in columns]
        e.column = int(max(range(len(columns)), key=lambda i: overlaps[i]))


def margins_and_whitespace(
    elements: list[LayoutElement], width: float, height: float, min_gap_ratio: float
) -> tuple[dict[str, float], list[BBox]]:
    content = [e for e in elements if e.type not in FURNITURE]
    if not content:
        return {"left": width / 2, "right": width / 2, "top": height / 2, "bottom": height / 2}, []
    margins = {
        "left": round(min(e.bbox.x for e in content), 2),
        "right": round(width - max(e.bbox.x1 for e in content), 2),
        "top": round(min(e.bbox.y for e in content), 2),
        "bottom": round(height - max(e.bbox.y1 for e in content), 2),
    }
    whitespace: list[BBox] = []
    ordered = sorted(content, key=lambda e: e.bbox.y)
    lowest = ordered[0].bbox.y1
    for e in ordered[1:]:
        if e.bbox.y - lowest > min_gap_ratio * height:
            whitespace.append(BBox(x=0, y=lowest, w=width, h=e.bbox.y - lowest))
        lowest = max(lowest, e.bbox.y1)
    return margins, whitespace


# =========================================================================== heuristic model


def _furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


class HeuristicLayoutModel(LayoutModel):
    name = "heuristic"

    def __init__(self, config: LayoutConfig) -> None:
        self.cfg = config

    def detect(self, parsed: ParsedDocument) -> tuple[list[PageLayout], float]:
        body = body_font_size(parsed)
        repeated = self._repeated_furniture(parsed.pages)
        state = {"title_found": False, "in_references": False}
        layouts: list[PageLayout] = []
        for page in parsed.pages:
            elements: list[LayoutElement] = []
            blocks = sorted(page.blocks, key=lambda b: (round(b.bbox.y, 0), b.bbox.x))
            for idx, block in enumerate(blocks):
                el = self._classify(
                    block,
                    page,
                    idx,
                    body,
                    repeated,
                    state,
                    first_page=page.page_number == parsed.pages[0].page_number,
                )
                if el is not None:
                    elements.append(el)
            layout_type, columns = detect_columns(elements, page.width, self.cfg.full_width_ratio)
            assign_columns(elements, columns, page.width, self.cfg.full_width_ratio)
            margins, whitespace = margins_and_whitespace(
                elements, page.width, page.height, self.cfg.whitespace_min_gap_ratio
            )
            layouts.append(
                PageLayout(
                    page=page.page_number,
                    width=page.width,
                    height=page.height,
                    orientation="landscape" if page.width > page.height else "portrait",
                    layout_type=layout_type,
                    columns=[(round(a, 2), round(b, 2)) for a, b in columns],
                    margins=margins,
                    whitespace=whitespace,
                    elements=elements,
                )
            )
        return layouts, body

    def _repeated_furniture(self, pages: list[PageData]) -> set[str]:
        if len(pages) < 2:
            return set()
        counts: Counter[str] = Counter()
        for page in pages:
            keys = set()
            for b in page.blocks:
                in_band = (
                    b.bbox.y1 <= self.cfg.header_band * page.height * 1.5
                    or b.bbox.y >= (1 - self.cfg.footer_band * 1.5) * page.height
                )
                if b.kind == "text" and b.text and in_band:
                    keys.add(_furniture_key(b.text))
            counts.update(keys)
        return {k for k, n in counts.items() if n >= max(2, len(pages) // 2)}

    def _classify(
        self,
        block: object,
        page: PageData,
        idx: int,
        body: float,
        repeated: set[str],
        state: dict[str, bool],
        *,
        first_page: bool,
    ) -> LayoutElement | None:
        from extraction.schemas import RawBlock

        assert isinstance(block, RawBlock)
        eid = f"p{page.page_number}-e{idx}"
        base = {
            "element_id": eid,
            "page": page.page_number,
            "bbox": block.bbox,
            "font_size": block.font_size,
        }
        attrs: dict[str, object] = {"source_block": block.block_id, "source": block.source}
        if block.confidence is not None:
            attrs["ocr_confidence"] = block.confidence

        if block.kind == "table":
            attrs["cells"] = block.table_cells or []
            return LayoutElement(type=ElementType.TABLE, text=block.text, attrs=attrs, **base)
        if block.kind == "image":
            attrs["image_path"] = block.image_path
            return LayoutElement(type=ElementType.FIGURE, text=block.text, attrs=attrs, **base)

        raw = block.text.strip()
        if not raw:
            return None
        flat = dehyphenate_lines(raw)
        size = block.font_size or body
        style = (block.style or "").lower()
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        h = page.height
        in_header = block.bbox.y1 <= self.cfg.header_band * h
        in_footer = block.bbox.y >= (1 - self.cfg.footer_band) * h

        def make(t: ElementType, text: str = flat) -> LayoutElement:
            return LayoutElement(type=t, text=text, attrs=attrs, **base)

        # DOCX styles are authoritative when present.
        if style:
            if style == "title":
                state["title_found"] = True
                return make(ElementType.TITLE)
            m = re.match(r"heading\s*(\d)", style)
            if m:
                hm = heading_match(flat)
                attrs["number"] = hm[0] if hm else None
                attrs["heading_title"] = hm[1] if hm and hm[0] else flat
                state["in_references"] = flat.lower() in REFERENCE_HEADINGS
                level = int(m.group(1))
                return make(
                    ElementType.SECTION_HEADING if level == 1 else ElementType.SUBSECTION_HEADING
                )
            if style == "caption":
                cm = caption_match(flat)
                if cm:
                    attrs["label_kind"], attrs["number"] = cm
                return make(ElementType.CAPTION)
            if style == "equation":
                attrs["number"] = None
                return make(ElementType.EQUATION, raw)
            if style.startswith("list"):
                return make(ElementType.LIST_ITEM)

        if (in_header or in_footer) and PAGE_NUMBER.match(flat):
            return make(ElementType.PAGE_NUMBER)
        if in_header or in_footer:
            small = size < body * 0.95 and len(flat.split()) <= 12
            if _furniture_key(raw) in repeated or small:
                return make(ElementType.HEADER if in_header else ElementType.FOOTER)

        cm = caption_match(flat)
        if cm and len(flat) < 600:
            kind, number = cm
            attrs["label_kind"], attrs["number"] = kind, number
            if kind == "algorithm" and len(lines) > 2:
                return make(ElementType.ALGORITHM, raw)
            if kind in {"figure", "table", "algorithm", "listing"}:
                return make(ElementType.CAPTION)

        hm = heading_match(raw)
        is_large = size >= body * self.cfg.heading_size_ratio
        if (
            not state["title_found"]
            and first_page
            and len(lines) <= 3
            and size >= body * self.cfg.title_size_ratio
            and block.bbox.y < 0.4 * h
            and not hm
        ):
            state["title_found"] = True
            return make(ElementType.TITLE)
        if hm and (is_large or block.bold or hm[0] is not None or block.source == "ocr"):
            hnum, title = hm
            attrs["number"] = hnum
            attrs["heading_title"] = title if hnum else flat
            state["in_references"] = title.lower().rstrip(":") in REFERENCE_HEADINGS
            depth = heading_depth(hnum) if hnum and not hnum.isalpha() else 1
            if not hnum and title.lower().rstrip(":") not in NAMED_SECTIONS and not is_large:
                depth = 2
            return make(
                ElementType.SECTION_HEADING if depth == 1 else ElementType.SUBSECTION_HEADING
            )
        if is_large and len(lines) == 1 and len(flat.split()) <= 14 and not flat.endswith("."):
            attrs["number"] = None
            attrs["heading_title"] = flat
            state["in_references"] = flat.lower() in REFERENCE_HEADINGS
            return make(
                ElementType.SECTION_HEADING
                if size >= body * 1.3
                else ElementType.SUBSECTION_HEADING
            )

        if looks_like_equation(raw):
            m = EQUATION_NUMBER.search(" ".join(raw.split()))
            attrs["number"] = m.group("num") if m else None
            return make(ElementType.EQUATION, raw)
        if state["in_references"]:
            return make(ElementType.REFERENCE)
        if size < body * 0.85 and block.bbox.y > 0.75 * h and re.match(r"^[\d\*†‡§]", flat):
            return make(ElementType.FOOTNOTE)
        if LIST_ITEM.match(lines[0]):
            return make(ElementType.LIST_ITEM)
        return make(ElementType.PARAGRAPH)


# =========================================================================== docling adapter


class DoclingLayoutModel(LayoutModel):  # pragma: no cover - optional heavy dependency
    """Adapter for docling (``pip install agentic-graphrag[layout]``)."""

    name = "docling"
    _LABELS: ClassVar[dict[str, ElementType]] = {
        "title": ElementType.TITLE,
        "section_header": ElementType.SECTION_HEADING,
        "page_header": ElementType.HEADER,
        "page_footer": ElementType.FOOTER,
        "text": ElementType.PARAGRAPH,
        "paragraph": ElementType.PARAGRAPH,
        "list_item": ElementType.LIST_ITEM,
        "table": ElementType.TABLE,
        "picture": ElementType.FIGURE,
        "caption": ElementType.CAPTION,
        "formula": ElementType.EQUATION,
        "footnote": ElementType.FOOTNOTE,
        "code": ElementType.ALGORITHM,
        "reference": ElementType.REFERENCE,
    }

    def __init__(self, config: LayoutConfig) -> None:
        from docling.document_converter import DocumentConverter

        self.cfg = config
        self._converter = DocumentConverter()

    def detect(self, parsed: ParsedDocument) -> tuple[list[PageLayout], float]:
        doc = self._converter.convert(parsed.source_path).document
        by_page: dict[int, list[LayoutElement]] = {p.page_number: [] for p in parsed.pages}
        for idx, (item, level) in enumerate(doc.iterate_items()):
            label = str(getattr(item, "label", "text")).split(".")[-1].lower()
            etype = self._LABELS.get(label, ElementType.PARAGRAPH)
            if not getattr(item, "prov", None):
                continue
            prov = item.prov[0]
            page_no = int(prov.page_no)
            page_h = float(doc.pages[page_no].size.height)
            bb = prov.bbox.to_top_left_origin(page_height=page_h)
            if etype == ElementType.SECTION_HEADING and level > 1:
                etype = ElementType.SUBSECTION_HEADING
            attrs: dict[str, object] = {}
            text = str(getattr(item, "text", "") or "")
            if etype == ElementType.TABLE:
                df = item.export_to_dataframe()
                attrs["cells"] = [list(map(str, df.columns)), *df.astype(str).values.tolist()]
            if etype in (ElementType.SECTION_HEADING, ElementType.SUBSECTION_HEADING):
                hm = heading_match(text)
                attrs["number"] = hm[0] if hm else None
                attrs["heading_title"] = hm[1] if hm and hm[0] else text
            by_page.setdefault(page_no, []).append(
                LayoutElement(
                    element_id=f"p{page_no}-e{idx}",
                    type=etype,
                    page=page_no,
                    bbox=BBox.from_corners(bb.l, bb.t, bb.r, bb.b),
                    text=text,
                    attrs=attrs,
                )
            )
        layouts: list[PageLayout] = []
        for page in parsed.pages:
            elements = by_page.get(page.page_number, [])
            layout_type, columns = detect_columns(elements, page.width, self.cfg.full_width_ratio)
            assign_columns(elements, columns, page.width, self.cfg.full_width_ratio)
            margins, whitespace = margins_and_whitespace(
                elements, page.width, page.height, self.cfg.whitespace_min_gap_ratio
            )
            layouts.append(
                PageLayout(
                    page=page.page_number,
                    width=page.width,
                    height=page.height,
                    orientation="landscape" if page.width > page.height else "portrait",
                    layout_type=layout_type,
                    columns=columns,
                    margins=margins,
                    whitespace=whitespace,
                    elements=elements,
                )
            )
        return layouts, body_font_size(parsed)


def build_layout_model(config: LayoutConfig) -> LayoutModel:
    if config.backend == "docling":
        return DoclingLayoutModel(config)
    return HeuristicLayoutModel(config)
