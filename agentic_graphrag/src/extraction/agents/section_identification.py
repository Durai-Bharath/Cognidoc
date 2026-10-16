"""Algorithm 8 - Section Identification Agent."""

from __future__ import annotations

import itertools
import re
from collections import Counter

from common.base_agent import BaseAgent
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.schemas import (
    FURNITURE,
    ElementType,
    LayoutElement,
    OrderedDocument,
    Section,
    SectionHierarchy,
    SectionIdentificationInput,
)

_OBJECT_KEYS = {
    ElementType.TABLE: "tables",
    ElementType.FIGURE: "figures",
    ElementType.EQUATION: "equations",
    ElementType.ALGORITHM: "algorithms",
    ElementType.REFERENCE: "references",
    ElementType.CAPTION: "captions",
}
FRONT_MATTER_ID = "sec-0"


class SectionIdentificationAgent(BaseAgent[SectionIdentificationInput, SectionHierarchy]):
    name = "SectionIdentificationAgent"
    algorithm = 8

    def _execute(self, data: SectionIdentificationInput) -> SectionHierarchy:
        ordered = data.ordered
        sections: list[Section] = []
        furniture: list[LayoutElement] = []
        title: str | None = None
        issues: list[str] = []
        current: Section | None = None
        current_sub: Section | None = None
        front: Section | None = None

        def next_id() -> str:
            return f"sec-{sum(s.section_id != FRONT_MATTER_ID for s in sections) + 1}"

        def target() -> Section:
            nonlocal front
            if current_sub is not None:
                return current_sub
            if current is not None:
                return current
            if front is None:
                front = Section(section_id=FRONT_MATTER_ID, title="Front Matter", level=1)
                sections.insert(0, front)
            return front

        for el in ordered.elements:
            if el.type in FURNITURE:
                furniture.append(el)
                continue
            if el.type == ElementType.TITLE and title is None:
                title = el.text
                target().elements.append(el)
                continue
            if el.type == ElementType.SECTION_HEADING:
                current = Section(
                    section_id=next_id(),
                    number=el.attrs.get("number"),
                    title=str(el.attrs.get("heading_title") or el.text),
                    level=1,
                    heading_element_id=el.element_id,
                    elements=[el],
                )
                sections.append(current)
                current_sub = None
                continue
            if el.type == ElementType.SUBSECTION_HEADING:
                number = el.attrs.get("number")
                if current is None:
                    parent_no = str(number).split(".")[0] if number else None
                    current = Section(
                        section_id=next_id(),
                        number=parent_no,
                        title=f"Section {parent_no or '?'} (implicit)",
                        level=1,
                    )
                    sections.append(current)
                    issues.append(
                        f"subsection '{el.text[:40]}' had no parent section; "
                        "implicit parent created"
                    )
                depth = str(number).count(".") + 1 if number else 2
                parent = current
                if depth >= 3 and current_sub is not None:
                    parent = current_sub
                sub = Section(
                    section_id=f"{parent.section_id}.{len(parent.subsections) + 1}",
                    number=number,
                    title=str(el.attrs.get("heading_title") or el.text),
                    level=parent.level + 1,
                    heading_element_id=el.element_id,
                    elements=[el],
                )
                parent.subsections.append(sub)
                if depth < 3 or current_sub is None:
                    current_sub = sub
                continue
            target().elements.append(el)

        for sec in sections:
            for node in sec.iter_sections():
                self._update_metadata(node)
        hierarchy = SectionHierarchy(document_title=title, sections=sections, furniture=furniture)
        hierarchy.validation = self.validate(hierarchy, ordered, issues)
        hierarchy.approved = hierarchy.validation.passed
        if not hierarchy.approved:
            self.flag(
                "section hierarchy inconsistencies",
                details={"failures": hierarchy.validation.failure_reasons()},
            )
        return hierarchy

    @staticmethod
    def _update_metadata(section: Section) -> None:
        elements = section.all_elements()
        pages = [e.page for e in elements] or [1]
        section.page_start, section.page_end = min(pages), max(pages)
        counts = Counter(e.type for e in elements)
        section.stats = {
            "elements": len(elements),
            "words": sum(len(e.text.split()) for e in elements),
            "paragraphs": counts[ElementType.PARAGRAPH] + counts[ElementType.LIST_ITEM],
            "tables": counts[ElementType.TABLE],
            "figures": counts[ElementType.FIGURE],
            "equations": counts[ElementType.EQUATION],
            "references": counts[ElementType.REFERENCE],
            "subsections": len(section.subsections),
        }
        section.object_ids = {}
        for el in elements:
            key = _OBJECT_KEYS.get(el.type)
            if key:
                section.object_ids.setdefault(key, []).append(el.element_id)

    @staticmethod
    def validate(
        hierarchy: SectionHierarchy, ordered: OrderedDocument, issues: list[str]
    ) -> ValidationOutcome:
        outcome = ValidationOutcome()

        # Section continuity: numbered siblings are consecutive.
        continuity: list[str] = []

        def check_siblings(siblings: list[Section], parent: str) -> None:
            nums = [s.number for s in siblings if s.number]
            last_parts = [n.split(".")[-1] for n in nums]
            ints = [int(p) for p in last_parts if p.isdigit()]
            if len(ints) == len(last_parts):
                for a, b in itertools.pairwise(ints):
                    if b != a + 1:
                        continuity.append(f"{parent}: numbering jumps from {a} to {b}")
            for s in siblings:
                check_siblings(s.subsections, s.section_id)

        check_siblings([s for s in hierarchy.sections if s.section_id != FRONT_MATTER_ID], "root")
        outcome.add(CheckResult.from_failures("section_continuity", continuity))

        empty = [
            f"{s.section_id} '{s.title}' has no content"
            for s in hierarchy.iter_all()
            if len(s.all_elements()) <= (1 if s.heading_element_id else 0)
        ]
        outcome.add(CheckResult.from_failures("non_empty_sections", empty, Severity.WARNING))

        consistency = list(issues)
        for top in hierarchy.sections:
            for sec in top.iter_sections():
                for sub in sec.subsections:
                    if (
                        sub.number
                        and sec.number
                        and not re.match(rf"^{re.escape(sec.number)}\.", sub.number)
                    ):
                        consistency.append(f"subsection {sub.number} is under section {sec.number}")
        outcome.add(CheckResult.from_failures("hierarchy_consistency", consistency))

        owner = hierarchy.element_section()
        association: list[str] = []
        for el in ordered.elements:
            if (
                el.type in _OBJECT_KEYS
                and el.type != ElementType.CAPTION
                and el.element_id not in owner
            ):
                association.append(f"{el.element_id} ({el.type}) has no section")
        for cap_id, target_id in ordered.caption_links.items():
            if cap_id in owner and target_id in owner and owner[cap_id] != owner[target_id]:
                association.append(
                    f"caption {cap_id} is in {owner[cap_id]} but its object "
                    f"{target_id} is in {owner[target_id]}"
                )
        outcome.add(CheckResult.from_failures("object_association", association))

        assigned = [e.element_id for s in hierarchy.sections for e in s.all_elements()]
        assigned += [e.element_id for e in hierarchy.furniture]
        counts = Counter(assigned)
        orphan = [
            f"{e.element_id} not assigned to any section"
            for e in ordered.elements
            if e.element_id not in counts
        ]
        orphan += [f"{i} assigned {n} times" for i, n in counts.items() if n > 1]
        outcome.add(CheckResult.from_failures("no_orphan_content", orphan))
        return outcome

    def confidence(self, output: SectionHierarchy) -> float | None:
        return round(output.validation.pass_ratio, 4)
