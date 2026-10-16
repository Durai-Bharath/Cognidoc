"""Algorithms 17-19 - TextSummarizationAgent, HallucinationDetectionAgent,
ConfidenceScoringAgent."""

from __future__ import annotations

import math

import pytest

from common.base_agent import AgentContext
from common.llm import LLMError, MockLLM
from extraction.agents.confidence import ConfidenceScoringAgent, level_for, weighted_score
from extraction.agents.hallucination import HallucinationDetectionAgent
from extraction.agents.summarization import TextSummarizationAgent
from extraction.schemas import (
    ConfidenceInput,
    ConfidenceLevel,
    GeneratedItem,
    HallucinationInput,
    ObjectConfidence,
    SummarizationInput,
)
from extraction.schemas import ElementType as T
from tests.extraction.builders import el, section, task

SECTION = (
    "The hydraulic pump is inspected every 500 hours. Technicians replace the seal when wear "
    "exceeds the limit. The valve controls pump pressure across the hydraulic system. "
    "Inspection records are stored for each pump. The pump pressure is checked after every "
    "seal replacement. The hydraulic system must be depressurised before inspection."
)


def _sum_task(text: str = SECTION) -> SummarizationInput:
    return SummarizationInput(
        task=task(
            section(
                "sec-1",
                "Pump Inspection",
                [el("h", T.SECTION_HEADING, "1 Pump Inspection"), el("p", T.PARAGRAPH, text)],
            )
        )
    )


# ----------------------------------------------------------------- Alg. 17


def test_llm_summary_accepted(ctx: AgentContext) -> None:
    s = TextSummarizationAgent(ctx).run(_sum_task()).unwrap()
    assert s.summary and not s.used_fallback and s.validation.passed
    assert 0 < s.summary_length <= ctx.config.summarization.max_summary_words
    assert "pump" in s.keywords and len(s.key_sentences) == ctx.config.summarization.key_sentences
    assert s.confidence == s.quality_score > 0


@pytest.mark.parametrize(
    ("answer", "failed_check"),
    [
        (
            {"summary": "The pump is inspected every 900 hours by the hydraulic pump team."},
            "factual_consistency",
        ),
        ({"summary": " ".join(["pump"] * 400)}, "length_constraints"),
        ({"summary": ""}, "summary_completeness"),
        (
            {"summary": "Guitars and concerts are fun for musicians in halls."},
            "context_preservation",
        ),
    ],
)
def test_invalid_summary_triggers_fallback(
    ctx: AgentContext, answer: dict[str, str], failed_check: str
) -> None:
    ctx.llm = MockLLM({"summarization": answer})
    agent = TextSummarizationAgent(ctx)
    s = agent.run(_sum_task()).unwrap()
    assert s.used_fallback and s.validation.passed
    assert s.summary.startswith("The hydraulic pump is inspected every 500 hours")


def test_llm_error_uses_fallback(ctx: AgentContext) -> None:
    ctx.llm = MockLLM({"summarization": [LLMError("down")]})
    s = TextSummarizationAgent(ctx).run(_sum_task()).unwrap()
    assert s.used_fallback and any("summary LLM" in i.reason for i in ctx.review_queue.items)


def test_redundant_and_noisy_sentences_removed(ctx: AgentContext) -> None:
    text = SECTION + " " + SECTION.split(". ")[0] + ". ### 12 ## !! ok."
    s = TextSummarizationAgent(ctx).run(_sum_task(text)).unwrap()
    assert len(set(s.key_sentences)) == len(s.key_sentences)
    assert not any("###" in k for k in s.key_sentences)


def test_section_without_sentences(ctx: AgentContext) -> None:
    s = TextSummarizationAgent(ctx).run(_sum_task("Short.")).unwrap()
    assert s.summary == "1 Pump Inspection Short." and s.validation.passed


# ----------------------------------------------------------------- Alg. 18


def _hal(ctx: AgentContext, text: str, source: str = SECTION, kind: str = "summary"):  # type: ignore[no-untyped-def]
    item = GeneratedItem(
        content_id=f"sec-1:{kind}",
        kind=kind,
        text=text,  # type: ignore[arg-type]
        source=source,
    )
    return (
        HallucinationDetectionAgent(ctx)
        .run(HallucinationInput(section_id="sec-1", items=[item]))
        .unwrap()
    )


def test_grounded_text_is_verified(ctx: AgentContext) -> None:
    report = _hal(ctx, "The hydraulic pump is inspected every 500 hours.")
    item = report.items[0]
    assert item.status == "Verified" and item.hscore == pytest.approx(1 - item.similarity, abs=1e-4)
    assert report.replacements == {}


def test_hscore_equal_to_tau_is_verified(ctx: AgentContext) -> None:
    text = "Guitar music concert pump."
    hscore = round(1 - HallucinationDetectionAgent(ctx).similarity(text, SECTION), 4)
    ctx.config.hallucination.tau = hscore  # HScore ≤ τ → Verified
    item = _hal(ctx, text).items[0]
    assert item.status == "Verified" and item.hscore == hscore


def test_hallucination_regenerated_within_bound(ctx: AgentContext) -> None:
    report = _hal(ctx, "Guitar strings vibrate in a Vienna concert hall in 1999.")
    item = report.items[0]
    assert item.status == "Regenerated" and item.regenerations == 1
    assert item.final_text.startswith("The hydraulic pump")
    assert report.replacements == {"sec-1:summary": item.final_text}


def test_hallucination_flagged_after_bounded_retries(ctx: AgentContext) -> None:
    ctx.llm = MockLLM({"hallucination_regeneration": {"text": "Still about guitars and music."}})
    ctx.config.hallucination.max_regenerations = 2
    item = _hal(ctx, "Guitar strings vibrate in a Vienna concert hall in 1999.").items[0]
    assert item.status == "Flagged" and item.regenerations == 2
    assert len(ctx.llm.calls) == 2  # type: ignore[attr-defined]
    assert any(i for i in item.issues if "numbers not in source" in i)
    assert any(i for i in item.issues if "entities not in source" in i)
    assert ctx.review_queue.items[-1].agent == "HallucinationDetectionAgent"


def test_no_regeneration_when_bound_is_zero(ctx: AgentContext) -> None:
    ctx.config.hallucination.max_regenerations = 0
    item = _hal(ctx, "Guitar strings vibrate.").items[0]
    assert item.status == "Flagged" and item.regenerations == 0


def test_empty_items_are_skipped(ctx: AgentContext) -> None:
    assert _hal(ctx, "  ").items == []


# ----------------------------------------------------------------- Alg. 19


def test_weighted_score_and_levels(ctx: AgentContext) -> None:
    cfg = ctx.config.confidence
    assert (
        weighted_score({"layout": 1.0, "summarization": 0.0}, {"layout": 1, "summarization": 1})
        == 0.5
    )
    assert weighted_score({"classification": 0.2, "layout": 0.8}, cfg.weights) == pytest.approx(0.8)
    assert level_for(0.8, cfg) == ConfidenceLevel.HIGH
    assert level_for(0.5, cfg) == ConfidenceLevel.MEDIUM
    assert level_for(0.4999, cfg) == ConfidenceLevel.LOW


def _conf(
    ctx: AgentContext,
    comps: dict[str, float | None],
    prior: list[ObjectConfidence] | None = None,
    weight: float = 1.0,
):  # type: ignore[no-untyped-def]
    return (
        ConfidenceScoringAgent(ctx)
        .run(
            ConfidenceInput(
                section_id="sec-2",
                component_scores={"sec-2": comps, "sec-2-tbl1": {"table": 0.5}},
                object_kinds={"sec-2": "section", "sec-2-tbl1": "table"},
                section_weight=weight,
                prior_sections=prior or [],
            )
        )
        .unwrap()
    )


GOOD = {
    "layout": 1.0,
    "section_identification": 1.0,
    "summarization": 0.9,
    "hallucination": 0.8,
    "table": None,
    "classification": 0.1,
}


def test_confidence_scores_and_document_level(ctx: AgentContext) -> None:
    prior = [
        ObjectConfidence(
            object_id="sec-1",
            kind="section",
            score=0.4,
            level=ConfidenceLevel.LOW,
            components={},
            weight=3.0,
        )
    ]
    repo = _conf(ctx, GOOD, prior=prior, weight=1.0)
    sec = next(o for o in repo.objects if o.object_id == "sec-2")
    assert "table" not in sec.components  # None = not applicable
    expected = (0.1 * 1 + 0.1 * 1 + 0.15 * 0.9 + 0.15 * 0.8) / 0.5
    assert sec.score == pytest.approx(expected, abs=1e-4) and sec.level == ConfidenceLevel.HIGH
    assert repo.document_score == pytest.approx((0.4 * 3 + sec.score) / 4, abs=1e-4)
    assert repo.validation.passed and not repo.recomputed


@pytest.mark.parametrize("bad", [1.7, math.nan, -0.2])
def test_invalid_values_are_recomputed(ctx: AgentContext, bad: float) -> None:
    repo = _conf(ctx, {**GOOD, "summarization": bad})
    assert repo.recomputed and repo.validation.get("score_consistency").passed  # type: ignore[union-attr]
    sec = next(o for o in repo.objects if o.object_id == "sec-2")
    assert 0 <= sec.score <= 1
    assert not repo.validation.get("invalid_values").passed  # type: ignore[union-attr]
    assert ctx.review_queue.items  # inputs remain invalid → flagged


def test_missing_required_score_fails(ctx: AgentContext) -> None:
    repo = _conf(ctx, {"layout": 1.0})
    check = repo.validation.get("missing_scores")
    assert check is not None and not check.passed
    assert "sec-2 missing summarization" in check.reasons
