"""Algorithm 7 - Reading Order Agent."""

from __future__ import annotations

import heapq
import itertools
import re
from collections import Counter

from common.base_agent import BaseAgent
from common.validation import CheckResult, ValidationOutcome
from extraction.schemas import (
    FURNITURE,
    Continuation,
    ElementType,
    LayoutElement,
    LayoutMap,
    OrderedDocument,
    PageLayout,
    ReadingOrderInput,
)

_BODY_EXCLUDE = FURNITURE | {ElementType.FOOTNOTE}
_CAPTION_TARGET = {
    "figure": ElementType.FIGURE,
    "table": ElementType.TABLE,
    "algorithm": ElementType.ALGORITHM,
    "listing": ElementType.ALGORITHM,
}


def _bands(elements: list[LayoutElement]) -> list[list[LayoutElement]]:
    """Split a page into horizontal bands separated by full-width (column=None) elements."""
    bands: list[list[LayoutElement]] = [[]]
    for el in sorted(elements, key=lambda e: (e.bbox.y, e.bbox.x)):
        if el.column is None:
            if bands[-1]:
                bands.append([])
            bands[-1].append(el)
            bands.append([])
        else:
            bands[-1].append(el)
    return [b for b in bands if b]


def graph_reading_order(elements: list[LayoutElement]) -> tuple[list[LayoutElement], bool]:
    """Topological order over 'above' and 'left-of' precedence edges (Kahn + priority).

    Returns the order and whether a cycle forced a positional fallback.
    """
    n = len(elements)
    tol = 2.0
    succ: dict[int, set[int]] = {i: set() for i in range(n)}
    indeg = [0] * n
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i == j:
                continue
            above = a.bbox.h_overlap(b.bbox) > 0 and a.bbox.y1 <= b.bbox.y + tol
            left = (
                a.bbox.x1 <= b.bbox.x + tol
                and a.bbox.y < b.bbox.y1
                and (a.column or 0) < (b.column or 0)
            )
            if (above or left) and j not in succ[i]:
                succ[i].add(j)
                indeg[j] += 1
    heap = [
        ((elements[i].column or 0), elements[i].bbox.y, elements[i].bbox.x, i)
        for i in range(n)
        if indeg[i] == 0
    ]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        *_, i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(
                    heap, ((elements[j].column or 0), elements[j].bbox.y, elements[j].bbox.x, j)
                )
    cycle = len(order) < n
    if cycle:
        rest = sorted(
            (i for i in range(n) if i not in set(order)),
            key=lambda i: ((elements[i].column or 0), elements[i].bbox.y),
        )
        order.extend(rest)
    return [elements[i] for i in order], cycle


def order_page(page: PageLayout) -> tuple[list[LayoutElement], list[str]]:
    """Lines 17-37 for one page: sorted elements and any warnings."""
    warnings: list[str] = []
    headers = sorted(
        (e for e in page.elements if e.type == ElementType.HEADER),
        key=lambda e: (e.bbox.y, e.bbox.x),
    )
    footers = sorted(
        (e for e in page.elements if e.type in (ElementType.FOOTER, ElementType.PAGE_NUMBER)),
        key=lambda e: (e.bbox.y, e.bbox.x),
    )
    footnotes = sorted(
        (e for e in page.elements if e.type == ElementType.FOOTNOTE),
        key=lambda e: (e.bbox.y, e.bbox.x),
    )
    body = [e for e in page.elements if e.type not in _BODY_EXCLUDE]

    if page.layout_type == "single":
        # Priority(e) = f(y, x, single) = y + x / W
        ordered = sorted(body, key=lambda e: e.bbox.y + e.bbox.x / max(page.width, 1.0) * 1e-3)
    else:
        ordered = []
        for band in _bands(body):
            if len(band) == 1 or all(e.column is None for e in band):
                ordered.extend(sorted(band, key=lambda e: (e.bbox.y, e.bbox.x)))
            elif page.layout_type == "double":
                for col in sorted({e.column for e in band if e.column is not None}):
                    ordered.extend(
                        sorted((e for e in band if e.column == col), key=lambda e: e.bbox.y)
                    )
            else:
                band_order, cycle = graph_reading_order(band)
                if cycle:
                    warnings.append(f"page {page.page}: cyclic precedence; positional fallback")
                ordered.extend(band_order)
    return headers + ordered + footnotes + footers, warnings


def _numeric_key(number: str) -> tuple[int, ...] | None:
    if not re.fullmatch(r"\d+(\.\d+)*", number):
        return None
    return tuple(int(p) for p in number.split("."))


class ReadingOrderAgent(BaseAgent[ReadingOrderInput, OrderedDocument]):
    name = "ReadingOrderAgent"
    algorithm = 7

    def _execute(self, data: ReadingOrderInput) -> OrderedDocument:
        layout = data.layout
        sequence: list[LayoutElement] = []
        layout_types: dict[int, str] = {}
        warnings: list[str] = []
        for page in sorted(layout.pages, key=lambda p: p.page):
            ordered, page_warnings = order_page(page)
            warnings.extend(page_warnings)
            layout_types[page.page] = page.layout_type
            for rank, el in enumerate(ordered):
                el = el.model_copy(deep=True)
                el.reading_priority = float(page.page * 10_000 + rank)
                sequence.append(el)

        caption_links = self._associate_captions(sequence)
        equation_context = self._associate_equations(sequence)
        footnote_links = self._associate_footnotes(sequence)
        continuations = self._continuity(sequence)
        sequence.sort(key=lambda e: e.reading_priority or 0.0)

        result = OrderedDocument(
            elements=sequence,
            caption_links=caption_links,
            equation_context=equation_context,
            footnote_links=footnote_links,
            continuations=continuations,
            page_layout_types=layout_types,
        )
        result.validation = self.validate(result, layout)
        result.approved = result.validation.passed
        if warnings:
            self.log.warning("reading_order_warnings", extra={"warnings": warnings})
        if not result.approved:
            self.flag(
                "reading order validation failed",
                details={"failures": result.validation.failure_reasons()},
            )
        return result

    # ------------------------------------------------------------------ associations
    @staticmethod
    def _associate_captions(seq: list[LayoutElement]) -> dict[str, str]:
        links: dict[str, str] = {}
        used: set[str] = set()
        for cap in (e for e in seq if e.type == ElementType.CAPTION):
            target_type = _CAPTION_TARGET.get(str(cap.attrs.get("label_kind", "")))
            if target_type is None:
                continue
            candidates = [
                e
                for e in seq
                if e.type == target_type and e.page == cap.page and e.element_id not in used
            ]
            if not candidates:
                continue

            def distance(e: LayoutElement, cap: LayoutElement = cap) -> float:
                gap = min(abs(cap.bbox.y - e.bbox.y1), abs(e.bbox.y - cap.bbox.y1))
                return gap - (0.5 * e.bbox.h_overlap(cap.bbox))

            best = min(candidates, key=distance)
            links[cap.element_id] = best.element_id
            used.add(best.element_id)
            best.attrs["caption"] = cap.text
            best.attrs["caption_id"] = cap.element_id
            if cap.attrs.get("number"):
                best.attrs.setdefault("number", cap.attrs["number"])
        for alg in (e for e in seq if e.type == ElementType.ALGORITHM):
            alg.attrs.setdefault("caption", alg.text.split("\n")[0])
        return links

    @staticmethod
    def _associate_equations(seq: list[LayoutElement]) -> dict[str, str]:
        links: dict[str, str] = {}
        for i, eq in enumerate(seq):
            if eq.type != ElementType.EQUATION:
                continue
            before = next((e for e in reversed(seq[:i]) if e.type == ElementType.PARAGRAPH), None)
            after = next((e for e in seq[i + 1 :] if e.type == ElementType.PARAGRAPH), None)
            target = before or after
            if target is not None:
                links[eq.element_id] = target.element_id
                eq.attrs["context"] = target.text[-400:]
        return links

    @staticmethod
    def _associate_footnotes(seq: list[LayoutElement]) -> dict[str, str]:
        links: dict[str, str] = {}
        for fn in (e for e in seq if e.type == ElementType.FOOTNOTE):
            m = re.match(r"^([\d]+|[\*†‡§]+)", fn.text)
            marker = m.group(1) if m else ""
            same_page = [
                e
                for e in seq
                if e.page == fn.page
                and e.type in (ElementType.PARAGRAPH, ElementType.LIST_ITEM, ElementType.TABLE)
            ]
            pattern = re.compile(rf"[A-Za-z\)\].,]{re.escape(marker)}(?!\d)") if marker else None
            target = next((e for e in same_page if pattern and pattern.search(e.text)), None)
            if target is None and same_page:
                target = same_page[-1]
            if target is not None:
                links[fn.element_id] = target.element_id
        return links

    @staticmethod
    def _continuity(seq: list[LayoutElement]) -> list[Continuation]:
        out: list[Continuation] = []
        pages = sorted({e.page for e in seq})
        body_types = {
            ElementType.PARAGRAPH,
            ElementType.TABLE,
            ElementType.FIGURE,
            ElementType.LIST_ITEM,
        }
        for p, nxt in itertools.pairwise(pages):
            last = next(
                (e for e in reversed(seq) if e.page == p and e.type not in _BODY_EXCLUDE), None
            )
            first = next((e for e in seq if e.page == nxt and e.type not in _BODY_EXCLUDE), None)
            if last is None or first is None or last.type not in body_types:
                continue
            kind = None
            if (
                last.type == ElementType.PARAGRAPH
                and first.type == ElementType.PARAGRAPH
                and not re.search(r"[.!?:;\"”')\]]$", last.text.strip())
                and (first.text[:1].islower() or last.text.rstrip().endswith("-"))
            ):
                kind = "paragraph"
            elif last.type == first.type == ElementType.TABLE:
                a, b = last.attrs.get("cells") or [[]], first.attrs.get("cells") or [[]]
                if a and b and len(a[0]) == len(b[0]) and "caption_id" not in first.attrs:
                    kind = "table"
            elif last.type == first.type == ElementType.FIGURE and "caption_id" not in last.attrs:
                kind = "figure"
            if kind:
                last.attrs["continues_to"] = first.element_id
                first.attrs["continued_from"] = last.element_id
                out.append(Continuation(from_id=last.element_id, to_id=first.element_id, kind=kind))
        return out

    # ------------------------------------------------------------------ validation
    @staticmethod
    def validate(result: OrderedDocument, layout: LayoutMap) -> ValidationOutcome:
        outcome = ValidationOutcome()
        ids = [e.element_id for e in result.elements]
        dupes = [f"{i} appears {n} times" for i, n in Counter(ids).items() if n > 1]
        outcome.add(CheckResult.from_failures("no_duplicated_elements", dupes))

        expected = {e.element_id for e in layout.all_elements()}
        missing = [f"{i} missing from reading sequence" for i in sorted(expected - set(ids))]
        extra = [f"{i} not in layout" for i in sorted(set(ids) - expected)]
        outcome.add(CheckResult.from_failures("no_missing_elements", missing + extra))

        transitions: list[str] = []
        for page_no, kind in result.page_layout_types.items():
            if kind == "single":
                continue
            page_seq = [
                e for e in result.elements if e.page == page_no and e.type not in _BODY_EXCLUDE
            ]
            for band in _split_on_full_width(page_seq):
                cols = [e.column for e in band if e.column is not None]
                if any(b < a for a, b in itertools.pairwise(cols)):
                    transitions.append(f"page {page_no}: column order {cols}")
        outcome.add(CheckResult.from_failures("correct_column_transitions", transitions))

        caption_issues: list[str] = []
        for cap in (e for e in result.elements if e.type == ElementType.CAPTION):
            if (
                cap.attrs.get("label_kind") in _CAPTION_TARGET
                and cap.element_id not in result.caption_links
            ):
                caption_issues.append(f"{cap.element_id} ('{cap.text[:40]}') not associated")
        targets = Counter(result.caption_links.values())
        caption_issues += [f"{t} has {n} captions" for t, n in targets.items() if n > 1]
        outcome.add(CheckResult.from_failures("valid_caption_associations", caption_issues))

        hierarchy: list[str] = []
        title_pos = next(
            (i for i, e in enumerate(result.elements) if e.type == ElementType.TITLE), None
        )
        first_heading = next(
            (
                i
                for i, e in enumerate(result.elements)
                if e.type in (ElementType.SECTION_HEADING, ElementType.SUBSECTION_HEADING)
            ),
            None,
        )
        if title_pos is not None and first_heading is not None and title_pos > first_heading:
            hierarchy.append("document title appears after the first heading")
        prev: tuple[int, ...] | None = None
        for e in result.elements:
            if e.type not in (ElementType.SECTION_HEADING, ElementType.SUBSECTION_HEADING):
                continue
            key = _numeric_key(str(e.attrs.get("number") or ""))
            if key is None:
                continue
            if prev is not None and key < prev:
                hierarchy.append(
                    f"heading {'.'.join(map(str, key))} follows {'.'.join(map(str, prev))}"
                )
            prev = key
        outcome.add(CheckResult.from_failures("preserve_document_hierarchy", hierarchy))
        return outcome

    def confidence(self, output: OrderedDocument) -> float | None:
        return round(output.validation.pass_ratio, 4)


def _split_on_full_width(seq: list[LayoutElement]) -> list[list[LayoutElement]]:
    bands: list[list[LayoutElement]] = [[]]
    for e in seq:
        if e.column is None:
            bands.append([])
        else:
            bands[-1].append(e)
    return [b for b in bands if b]
