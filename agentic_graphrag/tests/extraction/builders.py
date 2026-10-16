"""Small builders for hand-crafted agent inputs."""

from __future__ import annotations

from typing import Any

from common.validation import ValidationOutcome
from extraction.schemas import (
    BBox,
    ElementType,
    LayoutElement,
    LayoutMap,
    OrderedDocument,
    PageLayout,
    Section,
    SectionHierarchy,
    SectionTask,
)

_Y = {"value": 0.0}


def el(
    eid: str,
    etype: ElementType,
    text: str = "",
    *,
    page: int = 1,
    box: tuple[float, float, float, float] | None = None,
    column: int | None = 0,
    **attrs: Any,
) -> LayoutElement:
    if box is None:
        _Y["value"] += 20
        box = (72, _Y["value"], 400, 14)
    x, y, w, h = box
    return LayoutElement(
        element_id=eid,
        type=etype,
        page=page,
        bbox=BBox(x=x, y=y, w=w, h=h),
        text=text,
        column=column,
        attrs=attrs,
    )


def section(
    sid: str,
    title: str,
    elements: list[LayoutElement],
    *,
    number: str | None = None,
    subsections: list[Section] | None = None,
    level: int = 1,
) -> Section:
    pages = [e.page for e in elements] or [1]
    return Section(
        section_id=sid,
        number=number,
        title=title,
        level=level,
        heading_element_id=elements[0].element_id if elements else None,
        elements=elements,
        subsections=subsections or [],
        page_start=min(pages),
        page_end=max(pages),
    )


def hierarchy(*sections: Section) -> SectionHierarchy:
    return SectionHierarchy(sections=list(sections), validation=ValidationOutcome())


def task(
    sec: Section,
    hier: SectionHierarchy | None = None,
    *,
    page_count: int = 2,
    last: bool = True,
) -> SectionTask:
    hier = hier or hierarchy(sec)
    ordered = OrderedDocument(elements=[e for s in hier.sections for e in s.all_elements()])
    return SectionTask(
        section=sec, hierarchy=hier, ordered=ordered, page_count=page_count, is_last_section=last
    )


def layout_of(
    *elements: LayoutElement, layout_type: str = "single", width: float = 600, height: float = 800
) -> LayoutMap:
    pages: dict[int, list[LayoutElement]] = {}
    for e in elements:
        pages.setdefault(e.page, []).append(e)
    return LayoutMap(
        pages=[
            PageLayout(
                page=p,
                width=width,
                height=height,
                orientation="portrait",
                layout_type=layout_type,  # type: ignore[arg-type]
                columns=[(0.0, 290.0), (310.0, 600.0)] if layout_type != "single" else [],
                elements=els,
            )
            for p, els in sorted(pages.items())
        ]
    )
