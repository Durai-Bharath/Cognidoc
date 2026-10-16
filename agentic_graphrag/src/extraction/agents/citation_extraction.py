"""Algorithm 13 - Internal Citation Extraction Agent."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from common.base_agent import BaseAgent
from common.validation import CheckResult, ValidationOutcome
from extraction.heuristics import APPENDIX_HEADING
from extraction.schemas import (
    Citation,
    CitationRepository,
    CitationType,
    ElementType,
    SectionHierarchy,
    SectionTask,
)

PATTERNS: list[tuple[CitationType, re.Pattern[str]]] = [
    (CitationType.SUBSECTION, re.compile(r"\bSubsection\s+(\d+(?:\.\d+)+)", re.I)),
    (CitationType.SECTION, re.compile(r"(?:\bSections?|\bSec\.|§)\s*(\d+(?:\.\d+)*)", re.I)),
    (CitationType.FIGURE, re.compile(r"\b(?:Figures?|Figs?\.)\s*(\d+[a-z]?(?:\.\d+)?)", re.I)),
    (CitationType.TABLE, re.compile(r"\bTables?\s+(\d+(?:\.\d+)?)", re.I)),
    (
        CitationType.EQUATION,
        re.compile(r"\b(?:Equations?|Eqs?\.|Eqn\.)\s*\(?(\d+(?:\.\d+)?)\)?", re.I),
    ),
    (CitationType.ALGORITHM, re.compile(r"\b(?:Algorithms?|Alg\.)\s*(\d+)", re.I)),
    (CitationType.APPENDIX, re.compile(r"\bAppendix\s+([A-Z]|\d+)\b")),
    (CitationType.PAGE, re.compile(r"\b(?:page|p\.)\s*(\d+)\b", re.I)),
]
_VALID_LABEL = re.compile(r"^(?:[1-9]\d*(?:\.\d+)*[a-z]?|[A-Z])$")
_SKIP = {
    ElementType.CAPTION,
    ElementType.TABLE,
    ElementType.FIGURE,
    ElementType.TITLE,
    ElementType.SECTION_HEADING,
    ElementType.SUBSECTION_HEADING,
    ElementType.ALGORITHM,
}


@dataclass
class ObjectRegistry:
    """Every referencable object in S, keyed by (type, label)."""

    targets: dict[tuple[CitationType, str], list[str]] = field(default_factory=dict)
    section_of: dict[str, str] = field(default_factory=dict)
    page_of: dict[str, int] = field(default_factory=dict)
    level_of: dict[str, int] = field(default_factory=dict)
    page_count: int = 0

    def add(self, kind: CitationType, label: str, target: str) -> None:
        self.targets.setdefault((kind, label), []).append(target)

    def lookup(self, kind: CitationType, label: str) -> list[str]:
        if kind == CitationType.PAGE:
            n = int(label) if label.isdigit() else 0
            return [f"page:{n}"] if 1 <= n <= self.page_count else []
        if kind in (CitationType.SECTION, CitationType.SUBSECTION):
            return self.targets.get((CitationType.SECTION, label), [])
        return self.targets.get((kind, label), [])

    def labels(self, kind: CitationType) -> list[str]:
        if kind == CitationType.SUBSECTION:
            kind = CitationType.SECTION
        return sorted({lbl for (k, lbl) in self.targets if k == kind})

    def exists(self, target: str) -> bool:
        if target.startswith("page:"):
            return 1 <= int(target.split(":")[1]) <= self.page_count
        return target in self.section_of


def build_registry(hierarchy: SectionHierarchy, page_count: int) -> ObjectRegistry:
    reg = ObjectRegistry(page_count=page_count)
    kinds = {
        ElementType.FIGURE: CitationType.FIGURE,
        ElementType.TABLE: CitationType.TABLE,
        ElementType.EQUATION: CitationType.EQUATION,
        ElementType.ALGORITHM: CitationType.ALGORITHM,
    }
    for top in hierarchy.sections:
        for sec in top.iter_sections():
            reg.section_of[sec.section_id] = top.section_id
            reg.page_of[sec.section_id] = sec.page_start
            reg.level_of[sec.section_id] = sec.level
            if sec.number:
                reg.add(CitationType.SECTION, sec.number, sec.section_id)
            m = APPENDIX_HEADING.match(sec.title)
            if m:
                reg.add(CitationType.APPENDIX, m.group("num"), sec.section_id)
        for el in top.all_elements():
            reg.section_of[el.element_id] = top.section_id
            reg.page_of[el.element_id] = el.page
            kind = kinds.get(el.type)
            if kind and el.attrs.get("number"):
                reg.add(kind, str(el.attrs["number"]), el.element_id)
    return reg


class InternalCitationExtractionAgent(BaseAgent[SectionTask, CitationRepository]):
    name = "InternalCitationExtractionAgent"
    algorithm = 13

    def _execute(self, data: SectionTask) -> CitationRepository:
        section = data.section
        registry = build_registry(data.hierarchy, data.page_count)
        repo = CitationRepository(section_id=section.section_id)
        for el in section.all_elements():
            if el.type in _SKIP or not el.text:
                continue
            taken: list[tuple[int, int]] = []
            for kind, pattern in PATTERNS:
                for m in pattern.finditer(el.text):
                    if any(a <= m.start() < b for a, b in taken):
                        continue  # e.g. "Subsection 2.1" already matched as subsection
                    taken.append((m.start(), m.end()))
                    label = m.group(1)
                    if kind == CitationType.SECTION and "." in label:
                        kind_eff = CitationType.SUBSECTION
                    else:
                        kind_eff = kind
                    targets = registry.lookup(kind_eff, label)
                    target = targets[0] if targets else None
                    repo.citations.append(
                        Citation(
                            citation_id=f"{section.section_id}-cit{len(repo.citations) + 1}",
                            citation_type=kind_eff,
                            label=label,
                            text=m.group(0),
                            source_section=section.section_id,
                            source_element=el.element_id,
                            offset=m.start(),
                            target_object=target,
                            target_page=(
                                int(label)
                                if kind_eff == CitationType.PAGE and label.isdigit()
                                else registry.page_of.get(target or "")
                            ),
                            resolved=target is not None,
                            valid_numbering=bool(_VALID_LABEL.match(label)),
                        )
                    )
        for c in repo.citations:
            if c.resolved and c.target_object:
                repo.referenced.setdefault(c.citation_type.value, []).append(c.target_object)
        repo.validation = self.validate(repo, registry)
        repo.approved = repo.validation.passed
        if not repo.approved:
            self.flag(
                "unresolved or invalid internal citations",
                item_ref=section.section_id,
                details={"failures": repo.validation.failure_reasons()},
            )
        return repo

    @staticmethod
    def validate(repo: CitationRepository, registry: ObjectRegistry) -> ValidationOutcome:
        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures(
                "broken_references",
                [
                    f"{c.citation_id} '{c.text}' does not resolve"
                    for c in repo.citations
                    if not c.resolved and c.valid_numbering
                ],
            )
        )
        keys = Counter((c.source_element, c.offset) for c in repo.citations)
        outcome.add(
            CheckResult.from_failures(
                "duplicate_citations",
                [f"{k[0]}@{k[1]} extracted {n} times" for k, n in keys.items() if n > 1],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "invalid_numbering",
                [
                    f"{c.citation_id} '{c.text}' has invalid number '{c.label}'"
                    for c in repo.citations
                    if not c.valid_numbering
                ],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "missing_target_objects",
                [
                    f"{c.citation_id} target {c.target_object} not in document"
                    for c in repo.citations
                    if c.resolved and c.target_object and not registry.exists(c.target_object)
                ],
            )
        )
        return outcome
