"""Algorithm 6 - LayoutDetectionAgent (and the LayoutModel interface)."""

from __future__ import annotations

from pathlib import Path

from common.base_agent import AgentContext
from extraction.agents.layout import LayoutDetectionAgent, validate_page_layout
from extraction.agents.router import RouterAgent
from extraction.layout_models import LayoutModel, detect_columns
from extraction.schemas import (
    BBox,
    DocumentProperties,
    DocumentType,
    ElementType,
    LayoutInput,
    LayoutMap,
    PageData,
    PageLayout,
    ParsedDocument,
    RawBlock,
    RouterInput,
)
from tests.extraction.builders import el


def _block(
    i: int,
    text: str,
    x: float,
    y: float,
    w: float = 450,
    h: float = 14,
    size: float = 10,
    bold: bool = False,
    page: int = 1,
    **kw: object,
) -> RawBlock:
    return RawBlock(
        block_id=f"b{page}-{i}",
        page=page,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        font_size=size,
        bold=bold,
        **kw,
    )  # type: ignore[arg-type]


def _doc(pages: list[list[RawBlock]]) -> ParsedDocument:
    return ParsedDocument(
        document_id="d",
        source_path="x",
        doc_type=DocumentType.PDF,
        properties=DocumentProperties(page_count=len(pages)),
        pages=[
            PageData(page_number=i + 1, width=600, height=800, blocks=b)
            for i, b in enumerate(pages)
        ],
        workdir=".",
    )


def _detect(ctx: AgentContext, parsed: ParsedDocument) -> LayoutMap:
    return LayoutDetectionAgent(ctx).run(LayoutInput(parsed=parsed)).unwrap()


BODY = "Body text that is long enough to be a paragraph and ends with a period."


def test_single_column_element_types(ctx: AgentContext) -> None:
    page = [
        _block(0, "Running header", 72, 20, h=10, size=7),
        _block(1, "A Very Important Document", 72, 80, size=22, h=26),
        _block(2, "1 Introduction", 72, 130, size=14, bold=True, w=150),
        _block(3, BODY, 72, 160),
        _block(4, "• first bullet item", 72, 190),
        _block(5, "E = m c^2        (1)", 72, 220, w=200),
        _block(6, "Figure 2: A caption.", 72, 480, w=200, size=9),
        _block(7, "", 72, 300, w=300, h=170, kind="image", image_path="x.png"),
        _block(8, "References", 72, 520, size=14, bold=True, w=150),
        _block(9, "[1] A. Author, Some paper, 2020.", 72, 550),
        _block(10, "3", 290, 780, w=12, h=10, size=7),
    ]
    layout = _detect(ctx, _doc([page]))
    types = [e.type for e in layout.pages[0].elements]
    assert types == [
        ElementType.HEADER,
        ElementType.TITLE,
        ElementType.SECTION_HEADING,
        ElementType.PARAGRAPH,
        ElementType.LIST_ITEM,
        ElementType.EQUATION,
        ElementType.FIGURE,
        ElementType.CAPTION,
        ElementType.SECTION_HEADING,
        ElementType.REFERENCE,
        ElementType.PAGE_NUMBER,
    ]
    p = layout.pages[0]
    assert p.layout_type == "single" and p.orientation == "portrait" and p.approved
    eq = p.elements[5]
    assert eq.attrs["number"] == "1"
    assert p.margins["left"] == 72 and p.whitespace  # gap between equation and figure


def test_double_and_multi_column_detection(ctx: AgentContext) -> None:
    two = [_block(i, BODY, 50 if i % 2 == 0 else 320, 100 + 40 * (i // 2), w=230) for i in range(8)]
    three = [_block(i, BODY, [40, 220, 400][i % 3], 100 + 40 * (i // 3), w=160) for i in range(9)]
    layout = _detect(ctx, _doc([two, three]))
    assert layout.pages[0].layout_type == "double" and len(layout.pages[0].columns) == 2
    assert {e.column for e in layout.pages[0].elements} == {0, 1}
    assert layout.pages[1].layout_type == "multi" and len(layout.pages[1].columns) == 3


def test_repeated_header_and_footer_detected(ctx: AgentContext) -> None:
    pages = [
        [
            _block(0, "ACME Corp Manual", 72, 20, h=10, size=10, page=p),
            _block(1, BODY, 72, 100, page=p),
            _block(2, f"Page {p} of 2", 72, 775, h=10, page=p),
        ]
        for p in (1, 2)
    ]
    layout = _detect(ctx, _doc(pages))
    for page in layout.pages:
        assert [e.type for e in page.elements] == [
            ElementType.HEADER,
            ElementType.PARAGRAPH,
            ElementType.PAGE_NUMBER,
        ]


def test_docx_styles_are_authoritative(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    parsed = (
        RouterAgent(ctx)
        .run(RouterInput(path=str(samples["docx"]), workdir=str(tmp_path)))
        .unwrap()
        .parsed
    )
    types = [e.type for e in _detect(ctx, parsed).all_elements()]
    assert types[:3] == [ElementType.TITLE, ElementType.SECTION_HEADING, ElementType.PARAGRAPH]
    assert ElementType.SUBSECTION_HEADING in types and ElementType.CAPTION in types


def test_validation_failures_flag_page(ctx: AgentContext) -> None:
    page = [
        _block(0, BODY, 72, 100),
        _block(1, BODY, 74, 102),  # overlaps the first block
        _block(2, "Table 4: Orphan caption.", 72, 400, w=200),
    ]
    layout = _detect(ctx, _doc([page]))
    v = layout.pages[0].validation
    assert not layout.pages[0].approved
    assert not v.get("no_overlapping_regions").passed  # type: ignore[union-attr]
    assert not v.get("caption_alignment").passed  # type: ignore[union-attr]
    assert ctx.review_queue.items[-1].item_ref == "page:1"


def test_invalid_regions_and_object_alignment() -> None:
    from common.config import LayoutConfig

    page = PageLayout(
        page=1,
        width=600,
        height=800,
        orientation="portrait",
        layout_type="double",
        columns=[(0, 290), (310, 600)],
        elements=[
            el("t", ElementType.TABLE, box=(-50, 10, 100, 50), column=0, cells=[]),
            el("f", ElementType.FIGURE, box=(320, 10, 0, 50), column=1),
            el("p", ElementType.PARAGRAPH, "", box=(10, 300, 100, 10)),
        ],
    )
    v = validate_page_layout(page, LayoutConfig())
    assert not v.get("valid_reading_regions").passed  # type: ignore[union-attr]
    assert not v.get("table_alignment").passed  # type: ignore[union-attr]
    assert v.get("figure_alignment").passed  # type: ignore[union-attr]


def test_layout_model_is_pluggable(ctx: AgentContext) -> None:
    class Fixed(LayoutModel):
        name = "fixed"

        def detect(self, parsed: ParsedDocument) -> tuple[list[PageLayout], float]:
            return [
                PageLayout(
                    page=1,
                    width=600,
                    height=800,
                    orientation="portrait",
                    layout_type="single",
                    elements=[el("x", ElementType.PARAGRAPH, "hi")],
                )
            ], 11.0

    ctx.services["layout_model"] = Fixed()
    layout = _detect(ctx, _doc([[]]))
    assert layout.body_font_size == 11.0 and layout.pages[0].elements[0].element_id == "x"


def test_detect_columns_empty_page() -> None:
    assert detect_columns([], 600, 0.6) == ("single", [(0.0, 600)])
