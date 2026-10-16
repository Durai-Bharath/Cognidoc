"""Algorithm 14 - Cross-Reference Validation Agent."""

from __future__ import annotations

import difflib
from collections import Counter

from common.base_agent import BaseAgent
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.agents.citation_extraction import ObjectRegistry, build_registry
from extraction.schemas import (
    CitationType,
    CrossReference,
    CrossRefInput,
    CrossRefRepository,
    ElementType,
)

_ORPHAN_TYPES = {
    ElementType.TABLE: "table",
    ElementType.FIGURE: "figure",
    ElementType.EQUATION: "equation",
}


def find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one node (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, set()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                result.append(sorted(comp))

    for node in list(edges):
        if node not in index:
            visit(node)
    return result


class CrossReferenceValidationAgent(BaseAgent[CrossRefInput, CrossRefRepository]):
    name = "CrossReferenceValidationAgent"
    algorithm = 14

    def _execute(self, data: CrossRefInput) -> CrossRefRepository:
        task = data.task
        registry = build_registry(task.hierarchy, task.page_count)
        repo = CrossRefRepository(section_id=task.section.section_id)
        for cit in data.citations.citations:
            ref = CrossReference(
                reference_id=f"xref-{cit.citation_id}",
                citation_id=cit.citation_id,
                citation_type=cit.citation_type,
                source_section=cit.source_section,
                source_element=cit.source_element,
                target_object=None,
                status="unresolved",
            )
            candidates = registry.lookup(cit.citation_type, cit.label)
            if candidates:
                target = candidates[0]
                ref.target_object = target
                ref.target_section = registry.section_of.get(target)
                ref.errors = self._validate_reference(
                    cit.citation_type, cit.label, candidates, registry
                )
                ref.status = "resolved"
                repo.forward.setdefault(cit.source_element, []).append(target)
                repo.backward.setdefault(target, []).append(cit.source_element)
            else:
                ref.errors = [f"'{cit.text}' has no matching object"]
                ref.suggestion = self._nearest(cit.citation_type, cit.label, registry)
                self.log.warning(
                    "unresolved_reference",
                    extra={
                        "citation": cit.citation_id,
                        "text": cit.text,
                        "suggestion": ref.suggestion,
                    },
                )
            repo.references.append(ref)

        all_refs = [*data.prior_references, *repo.references]
        if task.is_last_section:
            referenced = {r.target_object for r in all_refs if r.target_object}
            for top in task.hierarchy.sections:
                for el in top.all_elements():
                    if (
                        el.type in _ORPHAN_TYPES
                        and el.attrs.get("number")
                        and el.element_id not in referenced
                    ):
                        repo.orphan_objects.append(el.element_id)
        section_edges: dict[str, set[str]] = {}
        for r in all_refs:
            if r.target_section and r.target_section != r.source_section:
                section_edges.setdefault(r.source_section, set()).add(r.target_section)
        repo.circular_references = find_cycles(section_edges)
        repo.report = {
            "citations": len(repo.references),
            "resolved": sum(r.status == "resolved" for r in repo.references),
            "unresolved": sum(r.status == "unresolved" for r in repo.references),
            "orphan_objects": len(repo.orphan_objects),
            "circular_groups": len(repo.circular_references),
        }
        repo.validation = self.validate(repo, data)
        repo.approved = repo.validation.passed
        if not repo.approved:
            self.flag(
                "cross-reference validation failed; manual review",
                item_ref=task.section.section_id,
                details={"failures": repo.validation.failure_reasons()},
            )
        return repo

    @staticmethod
    def _validate_reference(
        kind: CitationType, label: str, candidates: list[str], registry: ObjectRegistry
    ) -> list[str]:
        """Lines 14-18: number, hierarchy, page location, uniqueness."""
        errors: list[str] = []
        target = candidates[0]
        if kind != CitationType.PAGE and not any(
            (k == kind or {k, kind} <= {CitationType.SECTION, CitationType.SUBSECTION})
            and lbl == label
            and target in ids
            for (k, lbl), ids in registry.targets.items()
        ):
            errors.append(f"reference number {label} does not match target {target}")
        if kind == CitationType.SUBSECTION and registry.level_of.get(target, 2) < 2:
            errors.append(f"subsection reference {label} points to a top-level section")
        if (
            kind == CitationType.SECTION
            and registry.level_of.get(target, 1) > 1
            and "." not in label
        ):
            errors.append(f"section reference {label} points to a subsection")
        page = registry.page_of.get(target)
        if kind != CitationType.PAGE and page is not None and not 1 <= page <= registry.page_count:
            errors.append(f"target {target} on invalid page {page}")
        if len(candidates) > 1:
            errors.append(f"{kind.value} {label} is not unique ({len(candidates)} objects)")
        return errors

    @staticmethod
    def _nearest(kind: CitationType, label: str, registry: ObjectRegistry) -> str | None:
        labels = registry.labels(kind)
        if kind == CitationType.PAGE and registry.page_count:
            return f"page {min(max(1, int(label) if label.isdigit() else 1), registry.page_count)}"
        if not labels:
            return None
        close = difflib.get_close_matches(label, labels, n=1, cutoff=0.0)
        if not close and all(lbl.replace(".", "").isdigit() for lbl in labels) and label.isdigit():
            close = [min(labels, key=lambda lbl: abs(float(lbl.split(".")[0]) - int(label)))]
        return f"{kind.value} {close[0]}" if close else None

    @staticmethod
    def validate(repo: CrossRefRepository, data: CrossRefInput) -> ValidationOutcome:
        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures(
                "broken_references",
                [
                    f"{r.citation_id}: {'; '.join(r.errors)}"
                    for r in repo.references
                    if r.status == "unresolved"
                ],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "object_uniqueness",
                [
                    f"{r.citation_id}: {e}"
                    for r in repo.references
                    for e in r.errors
                    if "not unique" in e
                ],
            )
        )
        pairs = Counter(
            (r.source_element, r.target_object) for r in repo.references if r.target_object
        )
        outcome.add(
            CheckResult.from_failures(
                "duplicate_references",
                [f"{s} → {t} cited {n} times" for (s, t), n in pairs.items() if n > 1],
                Severity.WARNING,
            )
        )
        invalid = {c.citation_id for c in data.citations.citations if not c.valid_numbering}
        outcome.add(
            CheckResult.from_failures(
                "invalid_numbering", [f"{cid} has invalid numbering" for cid in sorted(invalid)]
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "reference_targets",
                [
                    f"{r.citation_id}: {e}"
                    for r in repo.references
                    for e in r.errors
                    if r.status == "resolved" and "not unique" not in e
                ],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "orphan_objects",
                [f"{o} is never referenced" for o in repo.orphan_objects],
                Severity.WARNING,
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "circular_references",
                [
                    f"sections {' ↔ '.join(c)} reference each other"
                    for c in repo.circular_references
                ],
                Severity.WARNING,
            )
        )
        return outcome
