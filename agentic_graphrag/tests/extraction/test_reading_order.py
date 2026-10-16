"""Algorithm 7 - ReadingOrderAgent."""

from __future__ import annotations

from common.base_agent import AgentContext
from extraction.agents.reading_order import ReadingOrderAgent, graph_reading_order
from extraction.schemas import ElementType as T
from extraction.schemas import LayoutMap, OrderedDocument, ReadingOrderInput
from tests.extraction.builders import el, layout_of


def _order(ctx: AgentContext, layout: LayoutMap) -> OrderedDocument:
    return ReadingOrderAgent(ctx).run(ReadingOrderInput(layout=layout)).unwrap()


def _ids(doc: OrderedDocument) -> list[str]:
    return [e.element_id for e in doc.elements]


def test_single_column_top_to_bottom_with_furniture(ctx: AgentContext) -> None:
    layout = layout_of(
        el("foot", T.PAGE_NUMBER, "1", box=(300, 780, 10, 10), column=None),
        el("b", T.PARAGRAPH, "second.", box=(72, 200, 400, 20)),
        el("a", T.PARAGRAPH, "first.", box=(72, 100, 400, 20)),
        el("head", T.HEADER, "Header", box=(72, 10, 400, 10), column=None),
        el("fn", T.FOOTNOTE, "1 A note.", box=(72, 700, 400, 10)),
    )
    doc = _order(ctx, layout)
    assert _ids(doc) == ["head", "a", "b", "fn", "foot"] and doc.approved
    priorities = [e.reading_priority for e in doc.elements]
    assert priorities == sorted(priorities)  # type: ignore[type-var]


def test_double_column_left_then_right_with_full_width_band(ctx: AgentContext) -> None:
    layout = layout_of(
        el("title", T.TITLE, "Title", box=(50, 40, 500, 30), column=None),
        el("r1", T.PARAGRAPH, "r1.", box=(320, 100, 250, 40), column=1),
        el("l1", T.PARAGRAPH, "l1.", box=(50, 100, 250, 40), column=0),
        el("l2", T.PARAGRAPH, "l2.", box=(50, 160, 250, 40), column=0),
        el("r2", T.PARAGRAPH, "r2.", box=(320, 160, 250, 40), column=1),
        el("wide", T.FIGURE, "", box=(50, 300, 500, 100), column=None),
        el("l3", T.PARAGRAPH, "l3.", box=(50, 450, 250, 40), column=0),
        el("r3", T.PARAGRAPH, "r3.", box=(320, 450, 250, 40), column=1),
        layout_type="double",
    )
    doc = _order(ctx, layout)
    assert _ids(doc) == ["title", "l1", "l2", "r1", "r2", "wide", "l3", "r3"]
    assert doc.validation.get("correct_column_transitions").passed  # type: ignore[union-attr]


def test_multi_column_uses_graph_order(ctx: AgentContext) -> None:
    layout = layout_of(
        el("c3", T.PARAGRAPH, "c3.", box=(420, 100, 150, 40), column=2),
        el("c1a", T.PARAGRAPH, "c1a.", box=(20, 100, 150, 40), column=0),
        el("c2", T.PARAGRAPH, "c2.", box=(220, 100, 150, 40), column=1),
        el("c1b", T.PARAGRAPH, "c1b.", box=(20, 160, 150, 40), column=0),
        layout_type="multi",
    )
    assert _ids(_order(ctx, layout)) == ["c1a", "c1b", "c2", "c3"]


def test_graph_order_cycle_falls_back_to_position() -> None:
    a = el("a", T.PARAGRAPH, "a", box=(0, 100, 101, 10), column=0)
    b = el("b", T.PARAGRAPH, "b", box=(100, 50, 100, 51), column=1)  # above a and right of it
    order, cycle = graph_reading_order([a, b])
    assert cycle and [e.element_id for e in order] == ["a", "b"]


def test_associations_and_cross_page_continuity(ctx: AgentContext) -> None:
    layout = layout_of(
        el("p1", T.PARAGRAPH, "The pump pressure1 is computed as", box=(72, 100, 400, 20)),
        el("eq", T.EQUATION, "P = F / A (1)", box=(72, 130, 300, 20)),
        el("fig", T.FIGURE, "", box=(72, 200, 400, 200), image_path="f.png"),
        el(
            "cap",
            T.CAPTION,
            "Figure 1: Flow.",
            box=(72, 410, 300, 15),
            label_kind="figure",
            number="1",
        ),
        el("tail", T.PARAGRAPH, "and the text continues onto the", box=(72, 600, 400, 20)),
        el("fn", T.FOOTNOTE, "1 Measured in bar.", box=(72, 760, 400, 10)),
        el("next", T.PARAGRAPH, "next page without a break.", page=2, box=(72, 80, 400, 20)),
    )
    doc = _order(ctx, layout)
    assert doc.caption_links == {"cap": "fig"}
    assert doc.equation_context == {"eq": "p1"}
    assert doc.footnote_links == {"fn": "p1"}
    assert [(c.from_id, c.to_id, c.kind) for c in doc.continuations] == [
        ("tail", "next", "paragraph")
    ]
    fig = next(e for e in doc.elements if e.element_id == "fig")
    assert fig.attrs["caption"] == "Figure 1: Flow." and fig.attrs["number"] == "1"


def test_multi_page_table_continuation(ctx: AgentContext) -> None:
    cells = [["a", "b"], ["1", "2"]]
    layout = layout_of(
        el("t1", T.TABLE, "", box=(72, 600, 400, 150), cells=cells),
        el("t2", T.TABLE, "", page=2, box=(72, 80, 400, 100), cells=cells),
    )
    assert [(c.from_id, c.kind) for c in _order(ctx, layout).continuations] == [("t1", "table")]


def test_validation_failures_are_flagged(ctx: AgentContext) -> None:
    layout = layout_of(
        el("h2", T.SECTION_HEADING, "2 Later", box=(72, 50, 200, 20), number="2"),
        el("h1", T.SECTION_HEADING, "1 Earlier", box=(72, 100, 200, 20), number="1"),
        el("cap", T.CAPTION, "Table 9: none.", box=(72, 300, 200, 15), label_kind="table"),
    )
    doc = _order(ctx, layout)
    v = doc.validation
    assert not doc.approved
    assert not v.get("preserve_document_hierarchy").passed  # type: ignore[union-attr]
    assert not v.get("valid_caption_associations").passed  # type: ignore[union-attr]
    assert ctx.review_queue.items[-1].agent == "ReadingOrderAgent"


def test_validate_detects_missing_and_duplicate_elements(ctx: AgentContext) -> None:
    layout = layout_of(el("a", T.PARAGRAPH, "a."), el("b", T.PARAGRAPH, "b."))
    doc = _order(ctx, layout)
    doc.elements = [doc.elements[0], doc.elements[0]]
    v = ReadingOrderAgent.validate(doc, layout)
    assert not v.get("no_duplicated_elements").passed  # type: ignore[union-attr]
    assert not v.get("no_missing_elements").passed  # type: ignore[union-attr]


def test_column_transition_violation_detected(ctx: AgentContext) -> None:
    layout = layout_of(
        el("r", T.PARAGRAPH, "r.", column=1),
        el("l", T.PARAGRAPH, "l.", column=0),
        layout_type="double",
    )
    doc = _order(ctx, layout)
    doc.elements = sorted(doc.elements, key=lambda e: -(e.column or 0))
    v = ReadingOrderAgent.validate(doc, layout)
    assert not v.get("correct_column_transitions").passed  # type: ignore[union-attr]
