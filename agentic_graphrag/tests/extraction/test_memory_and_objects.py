"""Algorithms 9-12 - ContextMemoryAgent, TableAnalysisAgent, FigureUnderstandingAgent,
EquationAnalysisAgent."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.base_agent import AgentContext
from common.llm import LLMError, MockLLM
from extraction.agents.context_memory import ContextMemoryAgent, search
from extraction.agents.equation_analysis import (
    EquationAnalysisAgent,
    text_to_latex,
    validate_equation,
)
from extraction.agents.figure_understanding import FigureUnderstandingAgent
from extraction.agents.table_analysis import (
    TableAnalysisAgent,
    detect_units,
    normalize_value,
    structure_cells,
)
from extraction.schemas import BBox, EquationVariable, StructuredEquation
from extraction.schemas import ElementType as T
from tests.extraction.builders import el, section, task

PUMP = (
    "The hydraulic pump moves fluid through the system. Pump pressure is regulated by a "
    "valve. The valve opens when pressure rises above the limit."
)
GUITAR = (
    "Guitar strings vibrate to produce music. Concert halls amplify the sound of the "
    "orchestra. Musicians tune instruments before each concert performance."
)


# ----------------------------------------------------------------- Alg. 9


def _memory_task(paragraphs: list[str]):  # type: ignore[no-untyped-def]
    els = [el("h", T.SECTION_HEADING, "1 Pumps")] + [
        el(f"p{i}", T.PARAGRAPH, text, page=1 + i // 3) for i, text in enumerate(paragraphs)
    ]
    return task(section("sec-1", "Pumps", els, number="1"))


def test_short_section_stored_whole(ctx: AgentContext) -> None:
    mem = ContextMemoryAgent(ctx).run(_memory_task([PUMP])).unwrap()
    assert mem.stored_whole and len(mem.chunks) == 1 and mem.approved
    assert mem.chunks[0].previous_chunk is None and mem.index.ids == ["sec-1-c0"]
    assert mem.embedding and len(mem.chunks[0].embedding) == ctx.embedder.dim


def test_context_limit_boundary_is_whole(ctx: AgentContext) -> None:
    agent = ContextMemoryAgent(ctx)
    tokens = agent.run(_memory_task([PUMP, GUITAR])).unwrap().token_count
    ctx.config.context_memory.context_limit = tokens  # L ≤ ContextLimit
    assert agent.run(_memory_task([PUMP, GUITAR])).unwrap().stored_whole
    ctx.config.context_memory.context_limit = tokens - 1
    assert not agent.run(_memory_task([PUMP, GUITAR])).unwrap().stored_whole


def test_long_section_semantic_chunks_graph_and_index(ctx: AgentContext) -> None:
    cfg = ctx.config.context_memory
    cfg.context_limit, cfg.max_chunk_tokens, cfg.min_chunk_tokens = 40, 60, 20
    cfg.link_threshold = 0.3
    paras = [PUMP, PUMP.replace("The", "A"), GUITAR, GUITAR.replace("The", "A"), PUMP]
    mem = ContextMemoryAgent(ctx).run(_memory_task(paras)).unwrap()
    assert not mem.stored_whole and len(mem.chunks) >= 3 and mem.approved
    for prev, nxt in zip(mem.chunks, mem.chunks[1:], strict=False):
        assert prev.next_chunk == nxt.chunk_id and nxt.previous_chunk == prev.chunk_id
    kinds = {e.kind for e in mem.edges}
    assert kinds == {"next", "semantic"}  # first and last chunks are both about pumps
    assert mem.chunks[-1].page_end == 2 and all(c.summary for c in mem.chunks)
    hits = search(mem.index, ctx.embedder.embed_one("guitar concert music"), k=1)
    assert "guitar" in next(c.text for c in mem.chunks if c.chunk_id == hits[0][0]).lower()


def test_validation_detects_and_reconstruction_repairs(ctx: AgentContext) -> None:
    ctx.config.context_memory.context_limit = 20
    ctx.config.context_memory.max_chunk_tokens = 40
    agent = ContextMemoryAgent(ctx)
    t = _memory_task([PUMP, GUITAR, PUMP])
    mem = agent.run(t).unwrap()
    broken = mem.model_copy(deep=True)
    broken.chunks[0].next_chunk = None
    broken.chunks[1].token_count = 999
    broken.chunks[-1].unit_ids = []
    v = ContextMemoryAgent.validate(broken, expected_units=4)
    assert not v.passed and {c.name for c in v.errors} >= {
        "no_missing_chunks",
        "correct_chunk_sequence",
        "consistent_metadata",
    }
    fixed = agent.reconstruct(broken, agent._units(t))
    assert ContextMemoryAgent.validate(fixed, expected_units=4).passed


def test_llm_chunk_summaries(ctx: AgentContext) -> None:
    ctx.config.context_memory.llm_chunk_summaries = True
    ctx.llm = MockLLM({"chunk_summary": {"summary": "Pumps move fluid."}})
    mem = ContextMemoryAgent(ctx).run(_memory_task([PUMP])).unwrap()
    assert mem.chunks[0].summary == "Pumps move fluid."


# ----------------------------------------------------------------- Alg. 10

CELLS: list[list[str | None]] = [
    ["Component", "Interval (h)", "Cost", None],
    ["Seal", "500", "1,200", "12%"],
    ["Valve", None, "90", "5%"],
]


def test_structure_cells_merged_units_and_values() -> None:
    headers, rows, merged, _grid = structure_cells(CELLS)
    assert headers == ["Component", "Interval (h)", "Cost", "Cost"]
    assert any(m.row == 0 and m.col == 2 and m.col_span == 2 for m in merged)
    assert any(m.row == 1 and m.col == 1 and m.row_span == 2 for m in merged)
    assert rows[1][1] == "500"  # vertical merge filled from origin
    assert detect_units(["Interval (h)", "Load"], [["1", "5 kg"], ["2", "7 kg"]]) == {
        "Interval (h)": "h",
        "Load": "kg",
    }
    assert normalize_value("1,200") == 1200.0 and normalize_value("12%") == 12.0
    assert normalize_value("n/a") is None and normalize_value("OK") == "OK"


def _table_task(cells: list[list[str | None]], caption: str | None = "Table 1: Parts."):  # type: ignore[no-untyped-def]
    attrs = {"cells": cells, "number": "1", "caption": caption} if caption else {"cells": cells}
    return task(
        section(
            "sec-2",
            "Parts",
            [el("h", T.SECTION_HEADING, "2 Parts"), el("tbl", T.TABLE, "", **attrs)],
        )
    )


def test_table_analysis_success(ctx: AgentContext) -> None:
    repo = (
        TableAnalysisAgent(ctx)
        .run(_table_task([["Part", "Qty"], ["Seal", "2"], ["Valve", "1"]]))
        .unwrap()
    )
    t = repo.tables[0]
    assert t.table_id == "sec-2-tbl1" and t.element_id == "tbl" and t.approved
    assert t.n_rows == 2 and t.n_cols == 2 and t.number == "1"
    assert "Part" in t.key_attributes and t.description and t.keywords
    assert repo.validation.passed


def test_table_validation_failures(ctx: AgentContext) -> None:
    repo = (
        TableAnalysisAgent(ctx)
        .run(_table_task([["A", "A", "C"], ["", "", "x"], ["", "", ""]], caption=None))
        .unwrap()
    )
    t = repo.tables[0]
    assert not t.approved
    assert not t.validation.get("header_consistency").passed  # type: ignore[union-attr]
    assert not t.validation.get("missing_cells").passed  # type: ignore[union-attr]
    caption = t.validation.get("caption_association")
    assert caption is not None and not caption.passed and caption.severity == "warning"
    assert ctx.review_queue.items[-1].item_ref == "sec-2-tbl1"


def test_missing_cell_ratio_boundary(ctx: AgentContext) -> None:
    ctx.config.table.max_missing_cell_ratio = 0.5
    ok = TableAnalysisAgent(ctx).run(_table_task([["A", "B"], ["1", ""]])).unwrap()
    assert ok.tables[0].validation.get("missing_cells").passed  # type: ignore[union-attr]


def test_table_llm_failure_falls_back(ctx: AgentContext) -> None:
    ctx.llm = MockLLM({"table_analysis": [LLMError("down")]})
    t = (
        TableAnalysisAgent(ctx)
        .run(_table_task([["Part", "Qty"], ["Seal", "2"]]))
        .unwrap()
        .tables[0]
    )
    assert t.description.startswith("Table with 1 rows") and t.summary == "Table 1: Parts."
    assert any("LLM analysis failed" in i.reason for i in ctx.review_queue.items)


# ----------------------------------------------------------------- Alg. 11


def _figure_task(image_path: str | None, **attrs: object):  # type: ignore[no-untyped-def]
    fig = el(
        "fig", T.FIGURE, "", image_path=image_path, number="1", caption="Figure 1: Flow.", **attrs
    )
    return task(
        section(
            "sec-3",
            "Flow",
            [
                el("h", T.SECTION_HEADING, "3 Flow"),
                el("p", T.PARAGRAPH, "As shown in Figure 1, it flows."),
                fig,
            ],
        )
    )


def test_figure_with_image_uses_vision(ctx: AgentContext, samples: dict[str, Path]) -> None:
    repo = FigureUnderstandingAgent(ctx).run(_figure_task(str(samples["clean_scan"]))).unwrap()
    fig = repo.figures[0]
    assert ctx.llm.calls[-1]["images"] == 1  # type: ignore[attr-defined]
    assert "As shown in Figure 1" in ctx.llm.calls[-1]["prompt"]  # type: ignore[attr-defined]
    assert fig.figure_type.value == "flowchart" and fig.workflow == ["Inspect", "Clean", "Replace"]
    assert fig.approved and repo.validation.passed


def test_figure_without_image_is_text_only(ctx: AgentContext) -> None:
    FigureUnderstandingAgent(ctx).run(_figure_task(None)).unwrap()
    assert ctx.llm.calls[-1]["images"] == 0  # type: ignore[attr-defined]


def test_figure_validation_failures(ctx: AgentContext) -> None:
    ctx.llm = MockLLM(
        {
            "figure_understanding": {
                "figure_type": "flowchart",
                "objects": [{"id": "a", "label": "A", "kind": ""}],
                "connectors": [{"source": "a", "target": "zzz", "label": ""}],
                "legends": ["x", "x"],
                "workflow": [],
                "description": "A diagram.",
            }
        }
    )
    fig = FigureUnderstandingAgent(ctx).run(_figure_task(None)).unwrap().figures[0]
    failed = {c.name for c in fig.validation.errors}
    assert failed == {"legend_consistency", "object_completeness", "workflow_integrity"}
    assert not fig.approved


def test_figure_llm_failure_and_continuation_skip(ctx: AgentContext) -> None:
    ctx.llm = MockLLM({"figure_understanding": [LLMError("down")]})
    repo = FigureUnderstandingAgent(ctx).run(_figure_task(None)).unwrap()
    assert repo.figures[0].description == "Figure 1: Flow."
    skipped = FigureUnderstandingAgent(ctx).run(_figure_task(None, continued_from="x")).unwrap()
    assert skipped.figures == []


# ----------------------------------------------------------------- Alg. 12


def _eq_task(text: str, number: str | None = "1", dup: bool = False):  # type: ignore[no-untyped-def]
    els = [
        el("h", T.SECTION_HEADING, "2 Method"),
        el(
            "eq",
            T.EQUATION,
            text,
            number=number,
            context="where P is the pressure and F is the force.",
        ),
    ]
    if dup:
        els.append(el("eq2", T.EQUATION, "x = y (1)", number=number))
    return task(section("sec-2", "Method", els))


def test_equation_analysis_success(ctx: AgentContext) -> None:
    repo = EquationAnalysisAgent(ctx).run(_eq_task("P = F / A (1)")).unwrap()
    eq = repo.equations[0]
    assert eq.latex == "P = F / A" and eq.number == "1" and eq.approved
    assert [v.symbol for v in eq.variables] == ["P", "F"]


def test_equation_fallback_latex_and_duplicate_numbering(ctx: AgentContext) -> None:
    ctx.llm = MockLLM({"equation_analysis": [LLMError("down"), LLMError("down")]})
    repo = EquationAnalysisAgent(ctx).run(_eq_task("Σ x ≤ α (1)", dup=True)).unwrap()
    eq = repo.equations[0]
    assert eq.latex == r"\Sigma x \leq \alpha"
    assert not eq.validation.get("equation_reference_mapping").passed  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("latex", "variables", "failed"),
    [
        (r"\frac{a}{b", [], "syntax_correctness"),
        (r"\left( a \right", [], "syntax_correctness"),
        ("a = b", [EquationVariable(symbol="z")], "variable_consistency"),
        ("$a = b$", [], "latex_integrity"),
        ("a ≤ b", [], "latex_integrity"),
        ("a = 50% b", [], "latex_integrity"),
    ],
)
def test_equation_validation_checks(
    latex: str, variables: list[EquationVariable], failed: str
) -> None:
    eq = StructuredEquation(
        equation_id="e",
        latex=latex,
        page=1,
        bbox=BBox(x=0, y=0, w=1, h=1),
        section_id="s",
        variables=variables,
    )
    v = validate_equation(eq, {})
    assert {c.name for c in v.errors} == {failed}


def test_text_to_latex_escapes_specials() -> None:
    assert text_to_latex("a ∈ B, 5% #1 (2)") == r"a \in B, 5\% \#1"
