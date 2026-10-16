"""Algorithm 10 - Table Analysis Agent."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.llm import LLMError
from common.text import keywords
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.schemas import (
    ElementType,
    LayoutElement,
    MergedCell,
    SectionTask,
    StructuredTable,
    TableRepository,
)

_UNIT_IN_HEADER = re.compile(r"[\(\[]\s*([^\)\]]{1,12})\s*[\)\]]\s*$")
_UNIT_SUFFIX = re.compile(
    r"^-?\d+(?:[.,]\d+)?\s*(%|mm|cm|m|km|kg|g|mg|s|ms|min|h|°C|°F|V|A|W|kW|Hz|"
    r"MHz|GHz|MB|GB|TB|KB)$"
)
_MISSING = {"", "-", "—", "n/a", "na", "none", "null"}


class TableLLM(BaseModel):
    description: str = ""
    summary: str = ""
    keywords: list[str] = []
    key_attributes: list[str] = []
    relationships: list[str] = []


def normalize_value(value: str | None) -> str | float | None:
    if value is None:
        return None
    text = " ".join(value.split())
    if text.lower() in _MISSING:
        return None
    m = _UNIT_SUFFIX.match(text)
    candidate = m.group(0)[: -len(m.group(1))].strip() if m else text
    candidate = (
        candidate.replace(",", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", candidate)
        else candidate
    )
    try:
        return float(candidate.rstrip("%"))
    except ValueError:
        return text


def structure_cells(
    cells: list[list[str | None]],
) -> tuple[list[str], list[list[str]], list[MergedCell], list[list[str | None]]]:
    """Headers, rows, merged-cell spans and the grid with merged cells filled from origin."""
    n_cols = max((len(r) for r in cells), default=0)
    grid = [list(r) + [None] * (n_cols - len(r)) for r in cells]
    filled: list[list[str | None]] = [list(r) for r in grid]
    spans: dict[tuple[int, int], MergedCell] = {}

    def merge_left(r: int, c: int) -> bool:
        origin = next(((r, k) for k in range(c - 1, -1, -1) if grid[r][k] is not None), None)
        if origin is None:
            return False
        span = spans.setdefault(origin, MergedCell(row=origin[0], col=origin[1]))
        span.col_span = max(span.col_span, c - origin[1] + 1)
        filled[r][c] = grid[origin[0]][origin[1]]
        return True

    def merge_up(r: int, c: int) -> bool:
        origin = next(((k, c) for k in range(r - 1, -1, -1) if grid[k][c] is not None), None)
        if origin is None:
            return False
        span = spans.setdefault(origin, MergedCell(row=origin[0], col=origin[1]))
        span.row_span = max(span.row_span, r - origin[0] + 1)
        filled[r][c] = grid[origin[0]][origin[1]]
        return True

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value is not None:
                continue
            # Header rows usually span columns; body rows usually span rows (grouping).
            if r == 0 or (
                grid[r - 1][c] is None and not any(grid[k][c] is not None for k in range(r))
            ):
                merge_left(r, c) or merge_up(r, c)
            else:
                merge_up(r, c) or merge_left(r, c)
    headers = [str(h or "").strip() for h in (filled[0] if filled else [])]
    rows = [[str(v or "").strip() for v in row] for row in filled[1:]]
    return headers, rows, list(spans.values()), grid


def detect_units(headers: list[str], rows: list[list[str]]) -> dict[str, str]:
    units: dict[str, str] = {}
    for c, header in enumerate(headers):
        m = _UNIT_IN_HEADER.search(header)
        if m:
            units[header] = m.group(1).strip()
            continue
        suffixes = Counter(
            mm.group(1) for row in rows if c < len(row) and (mm := _UNIT_SUFFIX.match(row[c]))
        )
        if suffixes:
            unit, n = suffixes.most_common(1)[0]
            if n >= max(1, len(rows) // 2):
                units[header] = unit
    return units


def validate_table(
    table: StructuredTable, raw_rows: list[list[str | None]], max_missing: float
) -> ValidationOutcome:
    outcome = ValidationOutcome()
    header_issues: list[str] = []
    if not any(table.headers):
        header_issues.append("no header labels")
    dupes = [h for h, n in Counter(h for h in table.headers if h).items() if n > 1]
    if dupes:
        header_issues.append(f"duplicate headers {dupes}")
    widths = {len(r) for r in raw_rows}
    if len(widths) > 1:
        header_issues.append(f"rows have differing widths {sorted(widths)}")
    outcome.add(CheckResult.from_failures("header_consistency", header_issues))

    merged = {
        (m.row + dr, m.col + dc)
        for m in table.merged_cells
        for dr in range(m.row_span)
        for dc in range(m.col_span)
    }
    total = empty = 0
    for r, row in enumerate(raw_rows[1:], start=1):
        for c, value in enumerate(row):
            if (r, c) in merged:
                continue
            total += 1
            if value is None or value.strip().lower() in _MISSING:
                empty += 1
    ratio = empty / total if total else 0.0
    outcome.add(
        CheckResult.from_failures(
            "missing_cells",
            [f"{ratio:.0%} of body cells are empty (limit {max_missing:.0%})"]
            if ratio > max_missing
            else [],
        )
    )
    outcome.add(
        CheckResult.from_failures(
            "caption_association",
            [] if table.caption else ["table has no associated caption"],
            Severity.WARNING,
        )
    )
    integrity: list[str] = []
    if table.n_rows < 1 or table.n_cols < 1:
        integrity.append(f"degenerate table {table.n_rows}x{table.n_cols}")
    if any(len(r) != table.n_cols for r in table.rows):
        integrity.append("rows are not rectangular")
    for m in table.merged_cells:
        if m.row + m.row_span > table.n_rows + 1 or m.col + m.col_span > table.n_cols:
            integrity.append(f"merged cell at ({m.row},{m.col}) exceeds the grid")
    outcome.add(CheckResult.from_failures("structural_integrity", integrity))
    return outcome


def to_markdown(headers: list[str], rows: list[list[str]], limit: int = 60) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... ({len(rows) - limit} more rows)")
    return "\n".join(lines)


class TableAnalysisAgent(BaseAgent[SectionTask, TableRepository]):
    name = "TableAnalysisAgent"
    algorithm = 10

    def _execute(self, data: SectionTask) -> TableRepository:
        section = data.section
        repo = TableRepository(section_id=section.section_id)
        for idx, el in enumerate(section.elements_of(ElementType.TABLE)):
            repo.tables.append(self._analyse(el, section.section_id, idx))
        for table in repo.tables:
            for check in table.validation.checks:
                repo.validation.add(
                    check.model_copy(update={"name": f"{table.table_id}:{check.name}"})
                )
        return repo

    def _analyse(self, el: LayoutElement, section_id: str, idx: int) -> StructuredTable:
        cfg = self.ctx.config.table
        raw: list[list[str | None]] = el.attrs.get("cells") or [[]]
        headers, rows, merged, grid = structure_cells(raw)
        units = detect_units(headers, rows)
        normalized = [[normalize_value(v) for v in row] for row in rows]
        key_attrs = self._key_attributes(headers, normalized)
        number = el.attrs.get("number")
        caption = el.attrs.get("caption")
        table = StructuredTable(
            table_id=f"{section_id}-tbl{idx + 1}",
            element_id=el.element_id,
            number=str(number) if number else None,
            caption=str(caption) if caption else None,
            page=el.page,
            bbox=el.bbox,
            section_id=section_id,
            headers=headers,
            rows=rows,
            normalized_rows=normalized,
            n_rows=len(rows),
            n_cols=len(headers),
            merged_cells=merged,
            units=units,
            key_attributes=key_attrs,
        )
        llm_out = self._describe(table)
        table.description = llm_out.description
        table.summary = llm_out.summary
        table.keywords = llm_out.keywords
        table.relationships = llm_out.relationships
        table.key_attributes = list(
            dict.fromkeys(key_attrs + [k for k in llm_out.key_attributes if k in headers])
        )
        table.validation = validate_table(table, grid, cfg.max_missing_cell_ratio)
        table.approved = table.validation.passed
        if not table.approved:
            self.flag(
                "table failed validation; reprocessing recommended",
                item_ref=table.table_id,
                details={"failures": table.validation.failure_reasons()},
            )
        return table

    @staticmethod
    def _key_attributes(headers: list[str], rows: list[list[str | float | None]]) -> list[str]:
        keys: list[str] = []
        for c, h in enumerate(headers):
            col = [r[c] for r in rows if c < len(r)]
            if not h or not col:
                continue
            values = [v for v in col if v is not None]
            if (len(values) == len(col) and len(set(map(str, values))) == len(values)) or (
                values and all(isinstance(v, float) for v in values)
            ):
                keys.append(h)
        return keys

    def _describe(self, table: StructuredTable) -> TableLLM:
        md = to_markdown(table.headers, table.rows)
        try:
            out = self.llm_json(
                task="table_analysis",
                system="You analyse tables extracted from documents. Describe only what the "
                "table contains; never invent values.",
                prompt=(
                    f"Caption: {table.caption or '(none)'}\nUnits: {table.units}\n\n{md}\n\n"
                    "Return a semantic description, a 1-2 sentence summary, up to 8 "
                    "keywords, the key attribute column names and notable relationships "
                    "between columns."
                ),
                schema=TableLLM,
            )
        except LLMError as exc:
            self.flag(f"table LLM analysis failed: {exc}", item_ref=table.table_id)
            out = TableLLM()
        if not out.description.strip():
            out.description = (
                f"Table with {table.n_rows} rows and {table.n_cols} columns "
                f"({', '.join(h for h in table.headers if h)})."
            )
        if not out.summary.strip():
            out.summary = table.caption or out.description
        if not out.keywords:
            out.keywords = keywords(" ".join(table.headers) + " " + (table.caption or ""), 8)
        return out
