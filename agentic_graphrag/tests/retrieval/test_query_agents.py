"""A1 Query Classification, A2 Query Rewriting, A3 Query Decomposition."""

from __future__ import annotations

from common.llm import LLMError, MockLLM
from retrieval.agents.query_classification import (
    ClassificationInput,
    QueryClassificationAgent,
    decide,
)
from retrieval.agents.query_decomposition import (
    DecompositionInput,
    QueryDecompositionAgent,
    validate_subqueries,
)
from retrieval.agents.query_rewriting import QueryRewritingAgent, RewritingInput
from retrieval.context import RetrievalAgentContext
from retrieval.models import Decomposition, QueryType, Subquery, Turn
from tests.retrieval.conftest import Stores, make_rctx

# =========================================================================== A1


def classify(ctx: RetrievalAgentContext, query: str) -> QueryType:
    return QueryClassificationAgent(ctx).run(ClassificationInput(query=query)).unwrap().query_type


def test_decide_evaluates_hybrid_first() -> None:
    assert decide(True, True) is QueryType.HYBRID
    assert decide(True, False) is QueryType.LOCAL
    assert decide(False, True) is QueryType.GLOBAL
    assert decide(False, False) is QueryType.SIMPLE_LOOKUP


def test_local_when_query_names_an_entity(rctx: RetrievalAgentContext) -> None:
    out = QueryClassificationAgent(rctx).run(
        ClassificationInput(query="What is the interval of the Valve?")
    )
    c = out.unwrap()
    assert c.query_type is QueryType.LOCAL
    assert "Valve" in c.entities and c.entity_specific and not c.thematic
    assert 0 <= c.confidence <= 1 and out.trace.confidence == c.confidence


def test_global_on_thematic_cues_without_entities(rctx: RetrievalAgentContext) -> None:
    c = (
        QueryClassificationAgent(rctx)
        .run(ClassificationInput(query="What are the main themes across the documents?"))
        .unwrap()
    )
    assert c.query_type is QueryType.GLOBAL
    assert {"main themes", "across"} <= set(c.cues)


def test_hybrid_is_reachable(rctx: RetrievalAgentContext) -> None:
    assert (
        classify(rctx, "What is the overall role of the Valve across the report?")
        is QueryType.HYBRID
    )


def test_simple_lookup_otherwise(rctx: RetrievalAgentContext) -> None:
    assert classify(rctx, "When was it written?") is QueryType.SIMPLE_LOOKUP


def test_acronyms_are_expanded_before_entity_matching(rctx: RetrievalAgentContext) -> None:
    c = (
        QueryClassificationAgent(rctx)
        .run(ClassificationInput(query="What does the HCU regulate?"))
        .unwrap()
    )
    assert c.query_type is QueryType.LOCAL
    assert "Hydraulic Control Unit" in c.entities


def test_llm_judgement_overrides_heuristic(rconfig, stores: Stores) -> None:  # type: ignore[no-untyped-def]
    llm = MockLLM(
        {
            "query_classification": {
                "entity_specific": False,
                "thematic": True,
                "confidence": 0.9,
                "rationale": "llm",
            }
        }
    )
    c = (
        QueryClassificationAgent(make_rctx(rconfig, stores, llm))
        .run(ClassificationInput(query="What is the interval of the Valve?"))
        .unwrap()
    )
    assert c.query_type is QueryType.GLOBAL and c.method == "llm" and c.confidence == 0.9
    assert "HINTS: entity_specific=true thematic=false" in llm.calls[0]["prompt"]


def test_llm_failure_or_empty_answer_keeps_heuristic(rconfig, stores: Stores) -> None:  # type: ignore[no-untyped-def]
    for spec in (LLMError("down"), {}):
        llm = MockLLM({"query_classification": spec}, max_retries=0)
        c = (
            QueryClassificationAgent(make_rctx(rconfig, stores, llm))
            .run(ClassificationInput(query="What is the interval of the Valve?"))
            .unwrap()
        )
        assert c.query_type is QueryType.LOCAL and c.method == "heuristic"


# =========================================================================== A2


def rewrite(ctx: RetrievalAgentContext, query: str, history: list[Turn] | None = None):  # type: ignore[no-untyped-def]
    return (
        QueryRewritingAgent(ctx)
        .run(RewritingInput(query=query, query_type=QueryType.LOCAL, history=history or []))
        .unwrap()
    )


def test_conversational_references_resolve_to_last_entity(rctx: RetrievalAgentContext) -> None:
    history = [
        Turn(role="user", content="Tell me about the Seal."),
        Turn(role="assistant", content="The Valve was replaced."),
    ]
    out = rewrite(rctx, "What is its inspection interval?", history)
    assert out.rewritten == "What is Valve's inspection interval?"
    assert out.resolved_references == {"its": "Valve's"}


def test_pronouns_stay_without_history(rctx: RetrievalAgentContext) -> None:
    out = rewrite(rctx, "What is its inspection interval?")
    assert out.rewritten == "What is its inspection interval?" and not out.resolved_references


def test_acronyms_are_expanded(rconfig, stores: Stores) -> None:  # type: ignore[no-untyped-def]
    rconfig.rewriting.acronyms = {"ISO": "International Organization for Standardization"}
    out = rewrite(make_rctx(rconfig, stores), "Does the HCU follow ISO rules?")
    assert "Hydraulic Control Unit (HCU)" in out.rewritten
    assert "International Organization for Standardization (ISO)" in out.rewritten
    assert out.expansions["HCU"] == "Hydraulic Control Unit"


def test_spelling_is_corrected_to_corpus_vocabulary(rctx: RetrievalAgentContext) -> None:
    out = rewrite(rctx, "What is the intervall of the Fiter?")
    assert out.rewritten == "What is the interval of the Filter?"
    assert out.corrections == {"intervall": "interval", "Fiter": "filter"}


def test_known_word_forms_are_left_alone(rctx: RetrievalAgentContext) -> None:
    out = rewrite(rctx, "Which inspections replaced the Valve?")
    assert out.rewritten == "Which inspections replaced the Valve?" and not out.corrections


def test_enrichment_lists_entity_types(rctx: RetrievalAgentContext) -> None:
    assert rewrite(rctx, "What is the status of the Valve?").enrichment


def test_llm_rewrite_accepted_only_if_entities_kept(rconfig, stores: Stores) -> None:  # type: ignore[no-untyped-def]
    good = MockLLM(
        {"query_rewriting": {"rewritten": "What inspection interval does the Valve have?"}}
    )
    out = rewrite(make_rctx(rconfig, stores, good), "What is the interval of the Valve?")
    assert out.rewritten == "What inspection interval does the Valve have?" and out.method == "llm"

    bad = MockLLM({"query_rewriting": {"rewritten": "What is the interval?"}})
    out = rewrite(make_rctx(rconfig, stores, bad), "What is the interval of the Valve?")
    assert out.rewritten == "What is the interval of the Valve?" and out.method == "heuristic"

    failing = MockLLM({"query_rewriting": LLMError("down")}, max_retries=0)
    out = rewrite(make_rctx(rconfig, stores, failing), "What is the interval of the Valve?")
    assert out.rewritten == "What is the interval of the Valve?"


# =========================================================================== A3


def decompose(ctx: RetrievalAgentContext, query: str) -> Decomposition:
    return (
        QueryDecompositionAgent(ctx)
        .run(DecompositionInput(query=query, query_type=QueryType.LOCAL))
        .unwrap()
    )


def test_simple_query_is_one_subquery(rctx: RetrievalAgentContext) -> None:
    d = decompose(rctx, "What is the interval of the Seal and the Valve?")
    assert [s.text for s in d.subqueries] == ["What is the interval of the Seal and the Valve?"]
    assert not d.complex and d.subqueries[0].depends_on == []


def test_dependency_chain_uses_placeholders(rctx: RetrievalAgentContext) -> None:
    d = decompose(rctx, "Which component has status Replaced, and what is its inspection interval?")
    q1, q2 = d.subqueries
    assert (q1.id, q1.text, q1.depends_on) == ("q1", "Which component has status Replaced?", [])
    assert q2.text == "What is {q1}'s inspection interval?" and q2.depends_on == ["q1"]
    assert d.complex and d.waves() == [["q1"], ["q2"]]
    assert q2.resolve({"q1": "Valve"}) == "What is Valve's inspection interval?"


def test_independent_subqueries_share_a_wave(rctx: RetrievalAgentContext) -> None:
    d = decompose(
        rctx,
        "What does the HCU regulate? What is the interval of the Seal? What is its status?",
    )
    assert [s.depends_on for s in d.subqueries] == [[], [], ["q2"]]
    assert d.waves() == [["q1", "q2"], ["q3"]]


def test_validate_subqueries_detects_every_problem() -> None:
    ok = [Subquery(id="q1", text="a?"), Subquery(id="q2", text="{q1} b?", depends_on=["q1"])]
    assert validate_subqueries(ok, 5) == []
    assert validate_subqueries([], 5) == ["no subqueries"]
    bad = [
        Subquery(id="q2", text="a?", depends_on=["q3"]),
        Subquery(id="q2", text="{q1}?"),
    ]
    problems = " ".join(validate_subqueries(bad, 1))
    for needle in ("> max 1", "expected q1", "not an earlier subquery", "placeholder {q1}"):
        assert needle in problems


def test_llm_decomposition_accepted_or_rejected(rconfig, stores: Stores) -> None:  # type: ignore[no-untyped-def]
    proposal = {
        "complex": True,
        "subqueries": [
            {"id": "q1", "text": "Which component was replaced?", "depends_on": []},
            {"id": "q2", "text": "What is the interval of {q1}?", "depends_on": ["q1"]},
        ],
    }
    ctx = make_rctx(rconfig, stores, MockLLM({"query_decomposition": proposal}))
    d = decompose(ctx, "Interval of the replaced component?")
    assert d.method == "llm" and d.subqueries[1].depends_on == ["q1"]

    invalid = {"subqueries": [{"id": "q1", "text": "x {q2}?", "depends_on": ["q2"]}]}
    ctx = make_rctx(rconfig, stores, MockLLM({"query_decomposition": invalid}))
    d = decompose(ctx, "Interval of the replaced component?")
    assert d.method == "heuristic" and len(d.subqueries) == 1
    assert ctx.review_queue.items[-1].agent == "query_decomposition"
