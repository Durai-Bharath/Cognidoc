"""Algorithm 8 - SectionIdentificationAgent."""

from __future__ import annotations

from common.base_agent import AgentContext
from extraction.agents.section_identification import (
    FRONT_MATTER_ID,
    SectionIdentificationAgent,
)
from extraction.schemas import ElementType as T
from extraction.schemas import (
    LayoutElement,
    OrderedDocument,
    SectionHierarchy,
    SectionIdentificationInput,
)
from tests.extraction.builders import el


def _sections(
    ctx: AgentContext, *elements: LayoutElement, captions: dict[str, str] | None = None
) -> SectionHierarchy:
    ordered = OrderedDocument(elements=list(elements), caption_links=captions or {})
    return (
        SectionIdentificationAgent(ctx)
        .run(SectionIdentificationInput(ordered=ordered, page_count=3))
        .unwrap()
    )


def _h(eid: str, number: str, title: str, sub: bool = False, page: int = 1) -> LayoutElement:
    return el(
        eid,
        T.SUBSECTION_HEADING if sub else T.SECTION_HEADING,
        f"{number} {title}",
        page=page,
        number=number,
        heading_title=title,
    )


def test_builds_hierarchy_with_front_matter_and_objects(ctx: AgentContext) -> None:
    h = _sections(
        ctx,
        el("hdr", T.HEADER, "Header"),
        el("title", T.TITLE, "Doc Title"),
        el("abs", T.PARAGRAPH, "Abstract text."),
        _h("s1", "1", "Intro"),
        el("p1", T.PARAGRAPH, "Intro text."),
        _h("s11", "1.1", "Background", sub=True),
        el("tbl", T.TABLE, "", cells=[["a"]]),
        el("cap", T.CAPTION, "Table 1: x", label_kind="table"),
        _h("s111", "1.1.1", "Detail", sub=True, page=2),
        el("p2", T.PARAGRAPH, "Detail text.", page=2),
        _h("s2", "2", "Method", page=3),
        el("eq", T.EQUATION, "a = b", page=3),
        captions={"cap": "tbl"},
    )
    assert h.document_title == "Doc Title" and [e.element_id for e in h.furniture] == ["hdr"]
    assert [s.section_id for s in h.sections] == [FRONT_MATTER_ID, "sec-1", "sec-2"]
    intro = h.sections[1]
    assert intro.subsections[0].section_id == "sec-1.1"
    assert intro.subsections[0].subsections[0].section_id == "sec-1.1.1"
    assert (intro.page_start, intro.page_end) == (1, 2)
    assert intro.object_ids["tables"] == ["tbl"] and intro.stats["subsections"] == 1
    assert h.sections[2].object_ids["equations"] == ["eq"]
    assert h.approved, h.validation.failure_reasons()


def test_subsection_without_parent_creates_implicit_section(ctx: AgentContext) -> None:
    h = _sections(ctx, _h("s", "3.1", "Orphan", sub=True), el("p", T.PARAGRAPH, "text."))
    assert h.sections[0].title.endswith("(implicit)") and h.sections[0].number == "3"
    assert not h.validation.get("hierarchy_consistency").passed  # type: ignore[union-attr]
    assert not h.approved and ctx.review_queue.items


def test_numbering_jump_breaks_continuity(ctx: AgentContext) -> None:
    h = _sections(
        ctx,
        _h("a", "1", "A"),
        el("p", T.PARAGRAPH, "x."),
        _h("b", "3", "C"),
        el("q", T.PARAGRAPH, "y."),
    )
    check = h.validation.get("section_continuity")
    assert check is not None and not check.passed and "from 1 to 3" in check.reasons[0]


def test_subsection_number_must_match_parent(ctx: AgentContext) -> None:
    h = _sections(
        ctx, _h("a", "1", "A"), _h("b", "2.1", "Wrong", sub=True), el("p", T.PARAGRAPH, "x.")
    )
    assert not h.validation.get("hierarchy_consistency").passed  # type: ignore[union-attr]


def test_caption_in_other_section_breaks_object_association(ctx: AgentContext) -> None:
    h = _sections(
        ctx,
        _h("a", "1", "A"),
        el("fig", T.FIGURE, ""),
        _h("b", "2", "B"),
        el("cap", T.CAPTION, "Figure 1: x", label_kind="figure"),
        captions={"cap": "fig"},
    )
    assert not h.validation.get("object_association").passed  # type: ignore[union-attr]


def test_empty_section_is_a_warning_only(ctx: AgentContext) -> None:
    h = _sections(ctx, _h("a", "1", "A"), _h("b", "2", "B"), el("p", T.PARAGRAPH, "x."))
    check = h.validation.get("non_empty_sections")
    assert check is not None and not check.passed and h.approved


def test_no_orphan_content_detects_unassigned(ctx: AgentContext) -> None:
    h = _sections(ctx, _h("a", "1", "A"), el("p", T.PARAGRAPH, "x."))
    ordered = OrderedDocument(elements=[*h.sections[0].elements, el("ghost", T.PARAGRAPH, "?")])
    v = SectionIdentificationAgent.validate(h, ordered, [])
    assert not v.get("no_orphan_content").passed  # type: ignore[union-attr]
