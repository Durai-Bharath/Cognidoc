"""A3 Triple Extraction, A4 Coreference Resolution, A5 Confidence Evaluation."""

from __future__ import annotations

from common.llm import LLMError, MockLLM
from construction.agents.confidence_evaluation import (
    ConfidenceEvaluationAgent,
    ConfidenceInput,
    grounded,
)
from construction.agents.coreference import (
    CoreferenceInput,
    CoreferenceResolutionAgent,
    mention_key,
)
from construction.agents.triple_extraction import (
    TripleExtractionAgent,
    TripleExtractionInput,
    TripleSet,
)
from construction.config import ConstructionConfig
from construction.context import ConstructionContext
from tests.construction.builders import chunk, make_ctx, triple


def _extract(ctx: ConstructionContext, text: str = "The pump uses oil."):  # type: ignore[no-untyped-def]
    return TripleExtractionAgent(ctx).run(
        TripleExtractionInput(chunk=chunk(text), document_title="Doc")
    )


def _raw(head: str, rel: str, tail: str, conf: float = 0.9, **kw: str) -> dict[str, object]:
    return {
        "head": head,
        "relation": rel,
        "tail": tail,
        "confidence": conf,
        "head_type": "",
        "tail_type": "",
        **kw,
    }


# ----------------------------------------------------------------------------- A3


def test_triples_are_extracted_with_provenance_and_types(cconfig: ConstructionConfig) -> None:
    llm = MockLLM(
        {"triple_extraction": {"triples": [_raw("The Pump", "uses", "oil", 0.8, head_type="part")]}}
    )
    ctx = make_ctx(cconfig, llm)
    out: TripleSet = _extract(ctx).unwrap()
    (t,) = out.triples
    assert (t.head, t.relation, t.tail, t.confidence) == ("Pump", "uses", "oil", 0.8)
    assert t.head_type == "Component" and t.tail_type == "Concept"
    assert t.provenance.doc_id == "doc-test" and t.provenance.chunk_id == "doc-test:c0"
    assert t.provenance.page == 1 and t.provenance.source_ref == "sec-1/doc-test:c0"
    prompt = llm.calls[0]["prompt"]
    assert "TEXT:\n<<<\nThe pump uses oil.\n>>>" in prompt and "has_interval" in prompt


def test_triples_are_normalised_deduplicated_and_limited(cconfig: ConstructionConfig) -> None:
    raws = [
        _raw("Pump", "uses", "oil", 0.6),
        _raw("pump", "uses", "Oil", 0.9),  # duplicate, higher confidence wins
        _raw("", "uses", "oil"),  # incomplete
        _raw("Pump", "is", "the pump"),  # self-referential
        _raw("Valve", "uses", "water"),
        _raw("Seal", "uses", "grease"),
    ]
    cconfig.triples.max_triples_per_chunk = 2
    ctx = make_ctx(cconfig, MockLLM({"triple_extraction": {"triples": raws}}))
    out = _extract(ctx).unwrap()
    assert out.raw_count == 6
    assert [(t.head, t.tail, t.confidence) for t in out.triples] == [
        ("pump", "Oil", 0.9),
        ("Valve", "water", 0.9),
    ]
    assert any("incomplete" in d for d in out.dropped) and any(
        "self-referential" in d for d in out.dropped
    )
    assert any("over limit" in d for d in out.dropped) and any(
        "duplicate" in d for d in out.dropped
    )


def test_malformed_llm_output_is_retried(cconfig: ConstructionConfig) -> None:
    good = {"triples": [_raw("Pump", "uses", "oil")]}
    llm = MockLLM(
        {"triple_extraction": ["not json", {"triples": [_raw("a", "b", "c", 1.7)]}, good]}
    )
    ctx = make_ctx(cconfig, llm)
    out = _extract(ctx).unwrap()
    assert len(out.triples) == 1 and len(llm.calls) == 3
    assert "previous answer was not valid" in llm.calls[2]["prompt"]


def test_llm_failure_is_a_typed_recoverable_error(cconfig: ConstructionConfig) -> None:
    ctx = make_ctx(cconfig, MockLLM({"triple_extraction": LLMError("boom")}))
    result = _extract(ctx)
    assert not result.ok and result.error is not None
    assert result.error.error_type == "TripleExtractionError" and result.error.recoverable
    exhausted = make_ctx(cconfig, MockLLM({"triple_extraction": "garbage"}, max_retries=1))
    assert _extract(exhausted).error.error_type == "TripleExtractionError"  # type: ignore[union-attr]


# ----------------------------------------------------------------------------- A4


def _coref(ctx: ConstructionContext, triples, **kw):  # type: ignore[no-untyped-def]
    return (
        CoreferenceResolutionAgent(ctx)
        .run(CoreferenceInput(triples=TripleSet(chunk_id="c", triples=triples), **kw))
        .unwrap()
    )


def test_pronouns_and_document_references_are_resolved(cctx: ConstructionContext) -> None:
    out = _coref(
        cctx,
        [
            triple("Pump", "uses", "oil"),
            triple("It", "requires", "maintenance"),
            triple("This report", "describes", "pump"),
        ],
        document_title="Pump Manual",
    )
    heads = [(t.head, t.tail) for t in out.triples.triples]
    assert ("Pump", "maintenance") in heads and ("Pump Manual", "Pump") in heads
    assert {"role": "head", "from": "It", "to": "Pump"} in [
        {k: r[k] for k in ("role", "from", "to")} for r in out.replacements
    ]
    assert out.last_subject == "Pump Manual"


def test_pronoun_uses_previous_chunk_subject_or_is_dropped(cctx: ConstructionContext) -> None:
    out = _coref(cctx, [triple("It", "uses", "oil")], previous_subject="Valve")
    assert [t.head for t in out.triples.triples] == ["Valve"]
    out = _coref(cctx, [triple("It", "uses", "oil"), triple("Pump", "uses", "they")])
    assert out.triples.triples == [] and len(out.unresolved) == 2


def test_abbreviations_resolve_to_canonical_names(cctx: ConstructionContext) -> None:
    abbr = {"HCU": "Hydraulic Control Unit", "P.L.C.": "Programmable Logic Controller"}
    out = _coref(
        cctx,
        [
            triple("HCU", "regulates", "pressure"),
            triple("Hydraulic Control Unit (HCU)", "uses", "oil"),
            triple("HCU valve", "part_of", "PLC"),
            triple("HCUs", "requires", "power"),
        ],
        abbreviations=abbr,
    )
    names = {(t.head, t.tail) for t in out.triples.triples}
    assert ("Hydraulic Control Unit", "pressure") in names
    assert ("Hydraulic Control Unit", "oil") in names
    assert ("Hydraulic Control Unit valve", "Programmable Logic Controller") in names
    assert ("Hydraulic Control Unit", "power") in names


def test_duplicate_mentions_are_merged(cctx: ConstructionContext) -> None:
    out = _coref(
        cctx,
        [
            triple("Pumps", "uses", "oil", 0.6),
            triple("pump", "uses", "Oil", 0.8),
            triple("pump", "requires", "seal"),
            triple("the pump", "requires", "seals", 0.5),
        ],
    )
    pairs = sorted((t.head, t.relation, t.tail, t.confidence) for t in out.triples.triples)
    assert pairs == [("pump", "requires", "seal", 0.9), ("pump", "uses", "oil", 0.8)]
    assert (
        mention_key("Inspections") == mention_key("inspection")
        and mention_key("status") == "status"
    )


def test_resolution_creating_self_loop_is_dropped(cctx: ConstructionContext) -> None:
    out = _coref(
        cctx,
        [triple("HCU", "part_of", "Hydraulic Control Unit")],
        abbreviations={"HCU": "Hydraulic Control Unit"},
    )
    assert out.triples.triples == [] and "self-referential" in out.unresolved[0]


# ----------------------------------------------------------------------------- A5


def _evaluate(ctx: ConstructionContext, triples, text: str = "The pump uses oil and grease."):  # type: ignore[no-untyped-def]
    return (
        ConfidenceEvaluationAgent(ctx)
        .run(ConfidenceInput(triples=TripleSet(chunk_id="c", triples=triples), chunk=chunk(text)))
        .unwrap()
    )


def test_confidence_threshold_accepts_and_discards(cctx: ConstructionContext) -> None:
    tau = cctx.settings.confidence.tau
    out = _evaluate(
        cctx,
        [
            triple("pump", "uses", "oil", 0.9),
            triple("pump", "uses", "grease", tau),
            triple("pump", "uses", "oil seal", tau - 0.01),
        ],
    )
    assert [t.tail for t in out.accepted] == ["oil", "grease"]  # boundary τ is accepted
    assert out.discarded[0]["tail"] == "oil seal" and "< τ" in str(out.discarded[0]["reason"])
    assert out.chunk_passes and out.accepted[0].provenance.confidence == 0.9


def test_ungrounded_triples_are_penalised(cctx: ConstructionContext) -> None:
    out = _evaluate(
        cctx,
        [triple("pump", "uses", "hydrogen", 0.9), triple("pump", "uses", "500 h", 0.9)],
        "Pump uses oil. Interval 500.",
    )
    by_tail = {t.tail: t for t in out.accepted}
    assert "500 h" in by_tail and by_tail["500 h"].confidence == 0.9  # units are ignored
    assert (
        by_tail["hydrogen"].confidence == round(0.9 * 0.6, 4)
        and "ungrounded" in by_tail["hydrogen"].notes
    )


def test_no_surviving_triple_fails_the_chunk(cctx: ConstructionContext) -> None:
    out = _evaluate(cctx, [triple("pump", "uses", "oil", 0.2)])
    assert not out.chunk_passes and out.mean_confidence == 0.0
    assert not _evaluate(cctx, []).chunk_passes


def test_grounding_uses_aliases_and_context() -> None:
    assert grounded("Hydraulic Control Unit", "The HCU is fine.", ["HCU"])
    assert grounded("control unit", "the hydraulic control unit", [])
    assert not grounded("valve", "the pump", [""])
