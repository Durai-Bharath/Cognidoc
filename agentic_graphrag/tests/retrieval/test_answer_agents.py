"""A9 Answer Generation, A10 Citation Mapping, A11 Faithfulness Validation, A12 Self-Critique."""

from __future__ import annotations

from common.llm import LLMError, MockLLM
from construction.models import entity_node_id
from retrieval.agents.answer_generation import AnswerGenerationAgent, AnswerInput
from retrieval.agents.citation_mapping import CitationInput, CitationMappingAgent, split_statements
from retrieval.agents.faithfulness_validation import (
    FaithfulnessValidationAgent,
    ValidationInput,
)
from retrieval.agents.self_critique import CritiqueInput, SelfCritiqueAgent
from retrieval.config import RetrievalConfig
from retrieval.context import RetrievalAgentContext
from retrieval.models import (
    ContextBlock,
    GeneratedAnswer,
    GroundedAnswer,
    QueryType,
    RetrievalContext,
    StatementCitation,
    ValidationStatus,
)
from tests.retrieval.conftest import Stores, make_rctx

QUESTION = "What is the status and interval of the Valve?"


def valve_context(ctx: RetrievalAgentContext) -> RetrievalContext:
    """E1/E3: real Valve facts, E2: the table chunk asserting them."""
    valve = entity_node_id("Valve")
    status = ctx.graph.find_edges(source=valve, predicate="has_status")[0]
    interval = ctx.graph.find_edges(source=valve, predicate="has_interval")[0]
    chunk_id = status.provenance[0].chunk_id
    chunk = ctx.graph.get_node(chunk_id)
    assert chunk is not None
    doc = chunk.properties["doc_id"]
    return RetrievalContext(
        query=QUESTION,
        query_type=QueryType.LOCAL,
        blocks=[
            ContextBlock(
                citation_id="E1",
                evidence_id=f"fact:{status.id}",
                kind="fact",
                text="Valve has status Replaced.",
                chunk_ids=[chunk_id],
                node_ids=[valve, status.target],
                edge_ids=[status.id],
                page=chunk.properties["page_start"],
                doc_id=doc,
            ),
            ContextBlock(
                citation_id="E2",
                evidence_id=f"chunk:{chunk_id}",
                kind="chunk",
                text=chunk.properties["text"],
                chunk_ids=[chunk_id],
                node_ids=[chunk_id],
                page=chunk.properties["page_start"],
                doc_id=doc,
            ),
            ContextBlock(
                citation_id="E3",
                evidence_id=f"fact:{interval.id}",
                kind="fact",
                text="Valve has interval 1000 h.",
                chunk_ids=[chunk_id],
                node_ids=[valve, interval.target],
                edge_ids=[interval.id],
                page=chunk.properties["page_start"],
                doc_id=doc,
            ),
        ],
    )


def ground(ctx: RetrievalAgentContext, answer: str, insufficient: bool = False) -> GroundedAnswer:
    return (
        CitationMappingAgent(ctx)
        .run(
            CitationInput(
                answer=GeneratedAnswer(answer=answer, insufficient_evidence=insufficient),
                context=valve_context(ctx),
            )
        )
        .unwrap()
    )


def validate(
    ctx: RetrievalAgentContext, answer: str, insufficient: bool = False
) -> tuple[GroundedAnswer, ValidationStatus]:
    grounded = ground(ctx, answer, insufficient)
    status = (
        FaithfulnessValidationAgent(ctx)
        .run(ValidationInput(question=QUESTION, grounded=grounded, context=valve_context(ctx)))
        .unwrap()
    )
    return grounded, status


GOOD = "Valve has status Replaced [E1]. Valve has interval 1000 h [E3]."
BAD = "Valve has status Replaced [E1]. Valve has interval 9999 h [E3]."

# =========================================================================== A9


def test_empty_context_is_insufficient_without_llm_call(rctx: RetrievalAgentContext) -> None:
    out = AnswerGenerationAgent(rctx).run(
        AnswerInput(question="q", context=RetrievalContext(query="q", query_type=QueryType.LOCAL))
    )
    assert out.unwrap().insufficient_evidence and "Insufficient evidence" in out.unwrap().answer
    assert rctx.llm.calls == []  # type: ignore[attr-defined]


def test_answer_is_grounded_with_markers(rctx: RetrievalAgentContext) -> None:
    out = AnswerGenerationAgent(rctx).run(
        AnswerInput(question=QUESTION, context=valve_context(rctx))
    )
    ans = out.unwrap()
    assert not ans.insufficient_evidence and "[E1]" in ans.answer and "[E3]" in ans.answer
    prompt = rctx.llm.calls[0]["prompt"]  # type: ignore[attr-defined]
    assert (
        "[E2] (type=chunk" in prompt
        and "insufficient evidence" in rctx.llm.calls[0]["system"].lower()
    )  # type: ignore[attr-defined]


def test_llm_insufficient_or_empty_answer_is_flagged(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    for spec in (
        {"answer": "I think so.", "insufficient_evidence": True},
        {"answer": "There is insufficient evidence in the context."},
        {"answer": ""},
    ):
        ctx = make_rctx(rconfig, stores, MockLLM({"answer_generation": spec}))
        ans = (
            AnswerGenerationAgent(ctx)
            .run(AnswerInput(question=QUESTION, context=valve_context(ctx)))
            .unwrap()
        )
        assert ans.insufficient_evidence


def test_intermediate_mode_returns_short_answer(rctx: RetrievalAgentContext) -> None:
    ans = (
        AnswerGenerationAgent(rctx)
        .run(
            AnswerInput(
                question="Which component has status Replaced?",
                context=valve_context(rctx),
                mode="intermediate",
            )
        )
        .unwrap()
    )
    assert ans.short_answer == "Valve"
    assert "MODE: intermediate" in rctx.llm.calls[0]["prompt"]  # type: ignore[attr-defined]


def test_llm_failure_is_a_typed_error(rconfig: RetrievalConfig, stores: Stores) -> None:
    ctx = make_rctx(
        rconfig, stores, MockLLM({"answer_generation": LLMError("down")}, max_retries=0)
    )
    out = AnswerGenerationAgent(ctx).run(AnswerInput(question=QUESTION, context=valve_context(ctx)))
    assert not out.ok and out.error is not None and out.error.error_type == "LLMError"


# =========================================================================== A10


def test_markers_map_statements_to_chunks_nodes_edges_and_pages(
    rctx: RetrievalAgentContext,
) -> None:
    g = ground(rctx, GOOD)
    s1, _s2 = g.statements
    assert (s1.text, s1.citation_ids) == ("Valve has status Replaced.", ["E1"])
    src = s1.sources[0]
    assert src.edge_ids and src.chunk_ids and src.page and src.doc_id.startswith("doc-")
    assert entity_node_id("Valve") in s1.entity_ids
    assert [c.citation_id for c in g.citations] == ["E1", "E3"] and not g.invalid_markers


def test_unknown_markers_and_uncited_statements(rctx: RetrievalAgentContext) -> None:
    g = ground(
        rctx, "Valve has status Replaced [E9]. Valve has interval 1000 h. The moon is cheese."
    )
    s1, s2, s3 = g.statements
    assert s1.invalid_markers == ["E9"] and s1.sources == [] and g.invalid_markers == ["E9"]
    assert s2.inferred and s2.citation_ids in (["E2"], ["E3"])  # best support, lowest id
    assert not s3.inferred and s3.sources == []


def test_split_statements_handles_marker_placement() -> None:
    assert split_statements("A is b. [E1] c is d [E2]. e is f [E3].") == [
        "A is b [E1].",
        "c is d [E2].",
        "e is f [E3].",
    ]


# =========================================================================== A11


def test_supported_answer_is_valid_with_integrity_proof(rctx: RetrievalAgentContext) -> None:
    _, vs = validate(rctx, GOOD)
    assert vs.valid and vs.grounding == 1.0 and vs.score >= vs.tau
    assert vs.integrity.passed and vs.integrity.checks
    assert {c.kind for c in vs.integrity.checks} == {"chunk", "node", "edge"}
    assert all(c.resolved and c.store == "graph:memory" for c in vs.integrity.checks)


def test_hallucinated_number_is_unsupported_and_invalid(rctx: RetrievalAgentContext) -> None:
    _, vs = validate(rctx, BAD)
    assert not vs.valid and vs.unsupported == ["S2"] and vs.grounding == 0.5
    claim = vs.claims[1]
    assert claim.support == 0.0 and "9999" in claim.missing_terms
    assert vs.validation.get("claims_supported").severity == "warning"  # type: ignore[union-attr]


def test_unresolvable_citation_fails_integrity(rctx: RetrievalAgentContext) -> None:
    _, marker = validate(rctx, "Valve has status Replaced [E1][E7].")
    assert not marker.valid and marker.integrity.invalid_markers == ["E7"]
    grounded, context = ground(rctx, GOOD), valve_context(rctx)
    rctx.graph.delete_edge(grounded.statements[1].sources[0].edge_ids[0])
    vs = (
        FaithfulnessValidationAgent(rctx)
        .run(ValidationInput(question=QUESTION, grounded=grounded, context=context))
        .unwrap()
    )
    assert not vs.valid and not vs.integrity.passed and len(vs.integrity.unresolved) == 1
    assert any("citation_integrity" in r for r in vs.reasons)


def test_insufficient_answer_is_invalid(rctx: RetrievalAgentContext) -> None:
    _, vs = validate(rctx, "Insufficient evidence to answer this question.", insufficient=True)
    assert not vs.valid and vs.insufficient_evidence and vs.completeness == 0.0
    assert any("has_claims" in r for r in vs.reasons)


def test_score_at_tau_is_valid(rconfig: RetrievalConfig, stores: Stores) -> None:
    ctx = make_rctx(rconfig, stores)
    _, vs = validate(ctx, BAD)
    rconfig.validation.tau = vs.score
    _, at_tau = validate(ctx, BAD)
    assert at_tau.score == at_tau.tau and at_tau.valid


def test_llm_judge_can_only_make_validation_stricter(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    rconfig.validation.use_llm = True
    judge = {
        "verdicts": [
            {"statement_id": "S1", "supported": False, "reason": "no"},
            {"statement_id": "S2", "supported": True},
        ]
    }
    ctx = make_rctx(rconfig, stores, MockLLM({"faithfulness_judge": judge}))
    _, vs = validate(ctx, BAD)
    assert [c.supported for c in vs.claims] == [False, False]
    assert vs.claims[0].reason == "LLM judge: no"


# =========================================================================== A12


def critique(ctx: RetrievalAgentContext, answer: str):  # type: ignore[no-untyped-def]
    grounded, vs = validate(ctx, answer)
    return (
        SelfCritiqueAgent(ctx)
        .run(
            CritiqueInput(
                question=QUESTION, grounded=grounded, status=vs, context=valve_context(ctx)
            )
        )
        .unwrap()
    )


def test_unsupported_claims_become_major_issues_with_implicated_knowledge(
    rctx: RetrievalAgentContext,
) -> None:
    va = critique(rctx, BAD)
    assert va.verdict == "needs_revision"
    assert [(i.statement_id, i.issue_type, i.severity) for i in va.issues] == [
        ("S2", "unsupported", "major")
    ]
    interval_edge = valve_context(rctx).blocks[2].edge_ids[0]
    assert va.implicated_edge_ids == [interval_edge]
    assert entity_node_id("Valve") in va.implicated_node_ids
    assert va.corrected_answer == "Valve has status Replaced [E1]."
    assert va.supported_edge_ids == valve_context(rctx).blocks[0].edge_ids


def test_contradiction_is_detected(rctx: RetrievalAgentContext) -> None:
    va = critique(rctx, "Valve has interval 1000 h [E3]. Valve has interval 500 h [E3].")
    kinds = {i.issue_type for i in va.issues}
    assert "contradiction" in kinds and va.verdict == "needs_revision"


def test_clean_answer_is_verified(rctx: RetrievalAgentContext) -> None:
    va = critique(rctx, GOOD)
    assert va.verdict == "verified" and va.issues == [] and va.used_llm


def test_llm_issues_must_name_known_statements(rconfig: RetrievalConfig, stores: Stores) -> None:
    llm = MockLLM(
        {
            "self_critique": {
                "verdict": "needs_revision",
                "issues": [
                    {
                        "statement_id": "S1",
                        "issue_type": "logical",
                        "severity": "major",
                        "description": "x",
                    },
                    {"statement_id": "S42", "issue_type": "logical", "severity": "major"},
                ],
                "corrected_answer": "Valve has interval 1000 h [E3].",
            }
        }
    )
    va = critique(make_rctx(rconfig, stores, llm), GOOD)
    assert [i.statement_id for i in va.issues] == ["S1"]
    assert va.corrected_answer == "Valve has interval 1000 h [E3]."
    failing = MockLLM({"self_critique": LLMError("down")}, max_retries=0)
    assert critique(make_rctx(rconfig, stores, failing), GOOD).used_llm is False


def test_statement_citation_defaults() -> None:
    assert StatementCitation(statement_id="S1", text="x").sources == []
