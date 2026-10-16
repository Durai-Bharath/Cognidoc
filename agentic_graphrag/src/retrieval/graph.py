"""Algorithm 1 - Overall Agentic Graph Retrieval Framework, as a LangGraph state machine.

Graph nodes map 1:1 to Algorithm 1 lines:

    cache_lookup(A15, before line 2) → [hit] → generate_report(22) → END
                                     → [miss] → query_classification(2) → query_rewriting(3)
        → query_decomposition(4) → next_subquery(5)
    next_subquery → [wave of ready subqueries] → vector_retrieval(6) → graph_traversal(7)
        → community_retrieval(8) → evidence_aggregation(9) → subquery_answers → next_subquery
    next_subquery → [all done] → context_construction(10) → answer_generation(11)
        → citation_mapping(12) → faithfulness_validation(13)
    faithfulness_validation → [Valid](14-15)            → feedback → cache_manager
                            → [Invalid or always_critique] → self_critique(17) → feedback(18)
    feedback → [Valid]   → cache_manager(21)
             → [Invalid] → retry_controller(19) → [retry](20) → next_subquery (restart at A4)
                                                → [limit reached] → cache_manager(21)
    cache_manager → generate_report(22) → END(23)

Subqueries run in dependency *waves*. Independent subqueries share a wave and run in
parallel inside each retrieval node. A subquery that later ones depend on gets an
intermediate answer (``subquery_answers``, A8 + A9) that fills their ``{qN}`` placeholders.
Every agent failure is recorded in ``agent_errors`` and degrades gracefully. Termination is
structural: the attempt counter only grows and is capped by ``retry.max_retries``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, TypeVar

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.base_agent import AgentResult, TraceRecord, TraceSink
from common.embedder import Embedder, build_embedder
from common.llm import LLMClient, build_llm
from common.logging import configure_logging, get_logger
from common.review_queue import ReviewQueue
from construction.stores import GraphStore, VectorStore, build_graph_store, build_vector_store
from retrieval.agents import (
    AnswerGenerationAgent,
    CacheManagerAgent,
    CitationMappingAgent,
    CommunityRetrievalAgent,
    ContextConstructionAgent,
    EvidenceAggregationAgent,
    FaithfulnessValidationAgent,
    FeedbackAgent,
    GraphTraversalAgent,
    QueryClassificationAgent,
    QueryDecompositionAgent,
    QueryRewritingAgent,
    RetryControllerAgent,
    SelfCritiqueAgent,
    VectorRetrievalAgent,
)
from retrieval.agents.answer_generation import AnswerInput
from retrieval.agents.cache_manager import CacheRequest, CacheResult, SemanticCache
from retrieval.agents.citation_mapping import CitationInput
from retrieval.agents.community_retrieval import CommunityRetrievalInput
from retrieval.agents.context_construction import ContextInput
from retrieval.agents.evidence_aggregation import AggregationInput
from retrieval.agents.faithfulness_validation import ValidationInput
from retrieval.agents.feedback import FeedbackInput
from retrieval.agents.graph_traversal import GraphTraversalInput
from retrieval.agents.query_classification import ClassificationInput
from retrieval.agents.query_decomposition import DecompositionInput
from retrieval.agents.query_rewriting import RewritingInput
from retrieval.agents.retry_controller import RetryInput
from retrieval.agents.self_critique import CritiqueInput
from retrieval.agents.vector_retrieval import VectorRetrievalInput
from retrieval.config import RetrievalConfig, load_retrieval_config
from retrieval.context import RetrievalAgent, RetrievalAgentContext
from retrieval.models import (
    AggregatedEvidence,
    AttemptRecord,
    Classification,
    CommunityEvidence,
    Decomposition,
    FeedbackReport,
    GeneratedAnswer,
    GraphEvidence,
    GroundedAnswer,
    QueryType,
    RetrievalContext,
    RetrievalParams,
    RetryDecision,
    RewrittenQuery,
    StatementCitation,
    Subquery,
    SubqueryResult,
    Turn,
    ValidationStatus,
    VerifiedAnswer,
    history_hash,
)
from retrieval.text import strip_markers

_log = get_logger(__name__)
REPORT_SCHEMA_VERSION = "1.0.0"
A = TypeVar("A", bound=RetrievalAgent[Any, Any])
OutT = TypeVar("OutT", bound=BaseModel)


class RetrievalState(TypedDict, total=False):
    report_id: str
    query: str
    history: list[Turn]
    history_hash: str
    started: float
    cache_lookup: CacheResult
    classification: Classification
    rewritten: RewrittenQuery
    decomposition: Decomposition
    params: RetrievalParams
    attempt: int
    waves: list[list[str]]
    wave_index: int
    current_wave: list[str]
    results: dict[str, SubqueryResult]
    sub_answers: dict[str, str]
    context: RetrievalContext
    generated: GeneratedAnswer
    grounded: GroundedAnswer
    status: ValidationStatus
    critique: VerifiedAnswer | None
    feedback: FeedbackReport | None
    retry: RetryDecision | None
    retries: list[RetryDecision]
    attempts: list[AttemptRecord]
    invalidated: list[str]
    cache_store: CacheResult | None
    agent_errors: list[str]
    report: dict[str, Any]


class RetrievalResult(BaseModel):
    """(A, C, R): grounded answer, citations, validation report."""

    report_id: str
    query: str
    answer: str
    verified: bool
    low_confidence: bool
    cached: bool = False
    citations: list[StatementCitation] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None
    traces: list[TraceRecord] = Field(default_factory=list)


def default_params(config: RetrievalConfig, query_type: QueryType) -> RetrievalParams:
    t = query_type.value
    return RetrievalParams(
        query_type=query_type,
        top_k=config.vector.top_k,
        entity_top_k=config.vector.entity_top_k,
        hops=int(config.traversal.hops[t]),
        community_weight=config.community.weights[t],
        community_top_n=config.community.top_n,
    )


class AgenticGraphRetrievalFramework:
    """Algorithm 1. Builds the shared context, the 15 agents and the state machine."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        review_queue: ReviewQueue | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        cc = self.config.as_construction_config()
        self.ctx = RetrievalAgentContext(
            config=self.config.as_extraction_config(),
            llm=llm or build_llm(self.config.llm),
            embedder=embedder or build_embedder(self.config.embedder),
            review_queue=review_queue or ReviewQueue(self.config.paths.review_queue),
            settings=self.config,
            graph=graph_store or build_graph_store(cc),
            vectors=vector_store or build_vector_store(cc),
            cache=cache
            or SemanticCache(
                self.config.paths.cache_file, max_entries=self.config.cache.max_entries
            ),
        )
        c = self.ctx
        self.classification = QueryClassificationAgent(c)
        self.rewriting = QueryRewritingAgent(c)
        self.decomposition = QueryDecompositionAgent(c)
        self.context_construction = ContextConstructionAgent(c)
        self.answer_generation = AnswerGenerationAgent(c)
        self.citation_mapping = CitationMappingAgent(c)
        self.validation = FaithfulnessValidationAgent(c)
        self.critique = SelfCritiqueAgent(c)
        self.feedback = FeedbackAgent(c)
        self.retry = RetryControllerAgent(c)
        self.cache = CacheManagerAgent(c)
        self.graph = self._build_graph()
        self.last_state: RetrievalState = {}

    @property
    def kg(self) -> GraphStore:
        return self.ctx.graph

    @property
    def vdb(self) -> VectorStore:
        return self.ctx.vectors

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())

    # ================================================================== public API
    def query(
        self,
        query: str,
        history: list[Turn] | list[dict[str, str]] | None = None,
        *,
        report_dir: str | Path | None = None,
    ) -> RetrievalResult:
        self.ctx.traces = TraceSink()
        self._flags_before = len(self.ctx.review_queue.items)
        turns = [t if isinstance(t, Turn) else Turn.model_validate(t) for t in history or []]
        report_id = "rpt-" + uuid.uuid4().hex[:16]
        self.ctx.document_id = report_id
        self._report_dir = Path(report_dir or Path(self.config.paths.output_dir) / "reports")
        state: RetrievalState = {
            "report_id": report_id,
            "query": query,
            "history": turns,
            "history_hash": history_hash(turns),
            "started": datetime.now(UTC).timestamp(),
            "attempt": 1,
            "results": {},
            "sub_answers": {},
            "attempts": [],
            "retries": [],
            "invalidated": [],
            "agent_errors": [],
            "critique": None,
            "feedback": None,
            "retry": None,
            "cache_store": None,
        }
        limit = (
            200
            + 60 * (self.config.retry.max_retries + 1) * self.config.decomposition.max_subqueries
        )
        final: RetrievalState = self.graph.invoke(state, config={"recursion_limit": limit})
        self.last_state = final
        report = final.get("report", {})
        return RetrievalResult(
            report_id=report_id,
            query=query,
            answer=report.get("answer", ""),
            verified=bool(report.get("verified")),
            low_confidence=bool(report.get("low_confidence")),
            cached=bool(report.get("cached")),
            citations=[StatementCitation.model_validate(s) for s in report.get("citations", [])],
            report=report,
            report_path=report.get("report_path"),
            traces=self.ctx.traces.records,
        )

    def invalidate_cache(self, element_ids: list[str] | None = None) -> list[str]:
        """Drop cache entries citing ``element_ids`` (or every stale entry when None)."""
        req = (
            CacheRequest(op="purge_stale")
            if element_ids is None
            else CacheRequest(op="invalidate", element_ids=element_ids)
        )
        result = self.cache.run(req)
        return result.output.invalidated if result.output else []

    # ================================================================== graph
    def _build_graph(self) -> Any:
        g: StateGraph[RetrievalState] = StateGraph(RetrievalState)
        nodes: dict[str, Callable[[RetrievalState], dict[str, Any]]] = {
            "cache_lookup": self._n_cache_lookup,
            "query_classification": self._n_classification,
            "query_rewriting": self._n_rewriting,
            "query_decomposition": self._n_decomposition,
            "next_subquery": self._n_next_subquery,
            "vector_retrieval": self._n_vector,
            "graph_traversal": self._n_traversal,
            "community_retrieval": self._n_communities,
            "evidence_aggregation": self._n_aggregation,
            "subquery_answers": self._n_subquery_answers,
            "context_construction": self._n_context,
            "answer_generation": self._n_answer,
            "citation_mapping": self._n_citations,
            "faithfulness_validation": self._n_validation,
            "self_critique": self._n_critique,
            "feedback": self._n_feedback,
            "retry_controller": self._n_retry,
            "cache_manager": self._n_cache_store,
            "generate_report": self._n_report,
        }
        for name, fn in nodes.items():
            g.add_node(name, fn)  # type: ignore[call-overload]
        g.add_edge(START, "cache_lookup")
        g.add_conditional_edges(
            "cache_lookup",
            lambda s: "generate_report" if s["cache_lookup"].hit else "query_classification",
            {"generate_report": "generate_report", "query_classification": "query_classification"},
        )
        for src, dst in [
            ("query_classification", "query_rewriting"),
            ("query_rewriting", "query_decomposition"),
            ("query_decomposition", "next_subquery"),
            ("vector_retrieval", "graph_traversal"),
            ("graph_traversal", "community_retrieval"),
            ("community_retrieval", "evidence_aggregation"),
            ("evidence_aggregation", "subquery_answers"),
            ("subquery_answers", "next_subquery"),
            ("context_construction", "answer_generation"),
            ("answer_generation", "citation_mapping"),
            ("citation_mapping", "faithfulness_validation"),
            ("self_critique", "feedback"),
            ("cache_manager", "generate_report"),
        ]:
            g.add_edge(src, dst)
        # Line 5: foreach subquery (in dependency waves).
        g.add_conditional_edges(
            "next_subquery",
            lambda s: "vector_retrieval" if s.get("current_wave") else "context_construction",
            {
                "vector_retrieval": "vector_retrieval",
                "context_construction": "context_construction",
            },
        )
        # Lines 14-17: the validation branch.
        g.add_conditional_edges(
            "faithfulness_validation",
            self._route_validation,
            {"feedback": "feedback", "self_critique": "self_critique"},
        )
        g.add_conditional_edges(
            "feedback",
            lambda s: "cache_manager" if s["status"].valid else "retry_controller",
            {"cache_manager": "cache_manager", "retry_controller": "retry_controller"},
        )
        # Lines 19-20: retry restarts at A4, or ends at the limit.
        g.add_conditional_edges(
            "retry_controller",
            self._route_retry,
            {"next_subquery": "next_subquery", "cache_manager": "cache_manager"},
        )
        g.add_edge("generate_report", END)
        return g.compile()

    # ------------------------------------------------------------------ routing
    def _route_validation(self, state: RetrievalState) -> str:
        if state["status"].valid and not self.config.critique.always_critique:
            return "feedback"
        return "self_critique"

    def _route_retry(self, state: RetrievalState) -> str:
        decision = state.get("retry")
        hard_cap = self.config.retry.max_retries + 1
        if decision is not None and decision.action == "retry" and state["attempt"] <= hard_cap:
            return "next_subquery"
        return "cache_manager"

    # ------------------------------------------------------------------ helpers
    def _tc(self, state: RetrievalState, **extra: str) -> dict[str, str]:
        return {"attempt": str(state.get("attempt", 1)), "report": state["report_id"], **extra}

    def _record(
        self, state: RetrievalState, result: AgentResult[Any], where: str = ""
    ) -> list[str]:
        errors = list(state.get("agent_errors", []))
        if result.error:
            loc = f" [{where}]" if where else ""
            errors.append(f"{result.error.agent}{loc}: {result.error.message}")
        return errors

    def _per_subquery(
        self,
        state: RetrievalState,
        agent_cls: type[RetrievalAgent[Any, OutT]],
        make_input: Callable[[SubqueryResult], BaseModel | None],
    ) -> tuple[dict[str, OutT | None], list[str]]:
        """Run one agent for every subquery of the current wave (in parallel if allowed)."""
        results = state["results"]
        wave = [sid for sid in state.get("current_wave", []) if results[sid].error is None]

        def one(sid: str) -> tuple[str, AgentResult[OutT] | None]:
            data = make_input(results[sid])
            if data is None:
                return sid, None
            agent = agent_cls(self.ctx)  # one instance per task: agents keep per-run usage
            return sid, agent.run(data, trace_context=self._tc(state, subquery=sid))

        if self.config.parallel_subqueries and len(wave) > 1:
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(wave))) as pool:
                pairs = list(pool.map(one, wave))
        else:
            pairs = [one(sid) for sid in wave]
        out: dict[str, OutT | None] = {}
        errors = list(state.get("agent_errors", []))
        for sid, res in pairs:
            if res is None:
                out[sid] = None
                continue
            if res.error:
                errors.append(f"{res.error.agent} [{sid}]: {res.error.message}")
            out[sid] = res.output
        return out, errors

    # ------------------------------------------------------------------ cache lookup
    def _n_cache_lookup(self, state: RetrievalState) -> dict[str, Any]:
        result = self.cache.run(
            CacheRequest(op="lookup", query=state["query"], history_hash=state["history_hash"]),
            trace_context=self._tc(state),
        )
        lookup = result.output or CacheResult(op="lookup", reason="cache lookup failed")
        return {
            "cache_lookup": lookup,
            "invalidated": [*state.get("invalidated", []), *lookup.invalidated],
            "agent_errors": self._record(state, result),
        }

    # ------------------------------------------------------------------ lines 2-4
    def _n_classification(self, state: RetrievalState) -> dict[str, Any]:
        result = self.classification.run(
            ClassificationInput(query=state["query"]), trace_context=self._tc(state)
        )
        # Fallback: an unclassifiable query gets the broadest strategy.
        classification = result.output or Classification(
            query_type=QueryType.HYBRID, confidence=0.0, rationale="classification failed"
        )
        return {"classification": classification, "agent_errors": self._record(state, result)}

    def _n_rewriting(self, state: RetrievalState) -> dict[str, Any]:
        qtype = state["classification"].query_type
        result = self.rewriting.run(
            RewritingInput(
                query=state["query"], query_type=qtype, history=state.get("history", [])
            ),
            trace_context=self._tc(state),
        )
        rewritten = result.output or RewrittenQuery(
            original=state["query"], rewritten=state["query"], query_type=qtype
        )
        return {"rewritten": rewritten, "agent_errors": self._record(state, result)}

    def _n_decomposition(self, state: RetrievalState) -> dict[str, Any]:
        rq = state["rewritten"]
        result = self.decomposition.run(
            DecompositionInput(query=rq.rewritten, query_type=rq.query_type),
            trace_context=self._tc(state),
        )
        decomposition = result.output or Decomposition(
            subqueries=[Subquery(id="q1", text=rq.rewritten)]
        )
        return {
            "decomposition": decomposition,
            "params": default_params(self.config, rq.query_type),
            "attempt": 1,
            **self._fresh_loop(decomposition),
            "agent_errors": self._record(state, result),
        }

    @staticmethod
    def _fresh_loop(decomposition: Decomposition) -> dict[str, Any]:
        return {
            "waves": decomposition.waves(),
            "wave_index": -1,
            "current_wave": [],
            "results": {},
            "sub_answers": {},
        }

    # ------------------------------------------------------------------ line 5: subquery loop
    def _n_next_subquery(self, state: RetrievalState) -> dict[str, Any]:
        i = state.get("wave_index", -1) + 1
        waves = state.get("waves", [])
        if i >= len(waves):
            return {"wave_index": i, "current_wave": []}
        subs = {s.id: s for s in state["decomposition"].subqueries}
        results = dict(state.get("results", {}))
        for sid in waves[i]:
            sq = subs[sid]
            results[sid] = SubqueryResult(
                subquery=sq, resolved_text=sq.resolve(state.get("sub_answers", {}))
            )
        return {"wave_index": i, "current_wave": list(waves[i]), "results": results}

    def _update_results(
        self, state: RetrievalState, field: str, outputs: dict[str, Any]
    ) -> dict[str, SubqueryResult]:
        results = dict(state["results"])
        for sid, value in outputs.items():
            r = results[sid]
            if value is None:
                results[sid] = r.model_copy(update={"error": f"{field} failed"})
            else:
                results[sid] = r.model_copy(update={field: value})
        return results

    def _n_vector(self, state: RetrievalState) -> dict[str, Any]:
        params = state["params"]
        outs, errors = self._per_subquery(
            state,
            VectorRetrievalAgent,
            lambda r: VectorRetrievalInput(
                subquery_id=r.subquery.id, text=r.resolved_text, params=params
            ),
        )
        return {"results": self._update_results(state, "vector", outs), "agent_errors": errors}

    def _n_traversal(self, state: RetrievalState) -> dict[str, Any]:
        params = state["params"]
        outs, errors = self._per_subquery(
            state,
            GraphTraversalAgent,
            lambda r: GraphTraversalInput(vector=r.vector, params=params) if r.vector else None,
        )
        return {"results": self._update_results(state, "graph", outs), "agent_errors": errors}

    def _n_communities(self, state: RetrievalState) -> dict[str, Any]:
        params = state["params"]
        outs, errors = self._per_subquery(
            state,
            CommunityRetrievalAgent,
            lambda r: (
                CommunityRetrievalInput(vector=r.vector, graph=r.graph, params=params)
                if r.vector and r.graph
                else None
            ),
        )
        return {"results": self._update_results(state, "communities", outs), "agent_errors": errors}

    def _n_aggregation(self, state: RetrievalState) -> dict[str, Any]:
        def make(r: SubqueryResult) -> AggregationInput | None:
            if r.vector is None:
                return None
            return AggregationInput(
                vector=r.vector,
                graph=r.graph or GraphEvidence(subquery_id=r.subquery.id),
                communities=r.communities or CommunityEvidence(subquery_id=r.subquery.id),
            )

        outs, errors = self._per_subquery(state, EvidenceAggregationAgent, make)
        return {"results": self._update_results(state, "aggregated", outs), "agent_errors": errors}

    def _n_subquery_answers(self, state: RetrievalState) -> dict[str, Any]:
        """Dependency resolution: answer the subqueries that later subqueries refer to."""
        needed = {d for s in state["decomposition"].subqueries for d in s.depends_on}
        results = dict(state["results"])
        sub_answers = dict(state.get("sub_answers", {}))
        errors = list(state.get("agent_errors", []))
        for sid in state.get("current_wave", []):
            if sid not in needed:
                continue
            r = results[sid]
            ctx_res = self.context_construction.run(
                ContextInput(
                    query=r.resolved_text,
                    query_type=state["params"].query_type,
                    evidence=[r.aggregated] if r.aggregated else [],
                ),
                trace_context=self._tc(state, subquery=sid, mode="intermediate"),
            )
            errors = self._record({**state, "agent_errors": errors}, ctx_res, sid)
            answer = None
            if ctx_res.output is not None:
                ans_res = self.answer_generation.run(
                    AnswerInput(
                        question=r.resolved_text, context=ctx_res.output, mode="intermediate"
                    ),
                    trace_context=self._tc(state, subquery=sid, mode="intermediate"),
                )
                errors = self._record({**state, "agent_errors": errors}, ans_res, sid)
                if ans_res.output is not None and not ans_res.output.insufficient_evidence:
                    answer = ans_res.output.short_answer or strip_markers(ans_res.output.answer)
            # No answer: dependants still retrieve around the unresolved question.
            sub_answers[sid] = answer or r.resolved_text.rstrip("?")
            results[sid] = r.model_copy(update={"answer": sub_answers[sid]})
        return {"results": results, "sub_answers": sub_answers, "agent_errors": errors}

    # ------------------------------------------------------------------ lines 10-13
    def _question(self, state: RetrievalState) -> str:
        return state["rewritten"].rewritten

    def _n_context(self, state: RetrievalState) -> dict[str, Any]:
        evidence: list[AggregatedEvidence] = [
            r.aggregated for r in state["results"].values() if r.aggregated is not None
        ]
        result = self.context_construction.run(
            ContextInput(
                query=self._question(state),
                query_type=state["params"].query_type,
                evidence=evidence,
                sub_answers=state.get("sub_answers", {}),
            ),
            trace_context=self._tc(state),
        )
        context = result.output or RetrievalContext(
            query=self._question(state), query_type=state["params"].query_type
        )
        return {"context": context, "agent_errors": self._record(state, result)}

    def _n_answer(self, state: RetrievalState) -> dict[str, Any]:
        subs = state["decomposition"].subqueries
        result = self.answer_generation.run(
            AnswerInput(
                question=self._question(state),
                context=state["context"],
                subquestions=[
                    state["results"][s.id].resolved_text for s in subs if s.id in state["results"]
                ],
            ),
            trace_context=self._tc(state),
        )
        generated = result.output or GeneratedAnswer(
            answer="Insufficient evidence to answer this question.", insufficient_evidence=True
        )
        return {"generated": generated, "agent_errors": self._record(state, result)}

    def _n_citations(self, state: RetrievalState) -> dict[str, Any]:
        result = self.citation_mapping.run(
            CitationInput(answer=state["generated"], context=state["context"]),
            trace_context=self._tc(state),
        )
        grounded = result.output or GroundedAnswer(
            answer=state["generated"].answer,
            insufficient_evidence=state["generated"].insufficient_evidence,
        )
        return {"grounded": grounded, "agent_errors": self._record(state, result)}

    def _n_validation(self, state: RetrievalState) -> dict[str, Any]:
        subs = state["decomposition"].subqueries
        targets = [state["results"][s.id].resolved_text for s in subs if s.id in state["results"]]
        result = self.validation.run(
            ValidationInput(
                question=self._question(state),
                grounded=state["grounded"],
                context=state["context"],
                targets=targets if len(targets) > 1 else [],
            ),
            trace_context=self._tc(state),
        )
        status = result.output or ValidationStatus(
            valid=False,
            score=0.0,
            grounding=0.0,
            completeness=0.0,
            tau=self.config.validation.tau,
            reasons=["faithfulness validation failed to run"],
        )
        record = AttemptRecord(
            attempt=state["attempt"],
            params=state["params"],
            escalation=(r.escalation if (r := state.get("retry")) is not None else None),
            answer=state["grounded"].answer,
            score=status.score,
            valid=status.valid,
            unsupported=status.unsupported,
            grounded=state["grounded"],
            validation=status,
        )
        return {
            "status": status,
            "critique": None,
            "attempts": [*state.get("attempts", []), record],
            "agent_errors": self._record(state, result),
        }

    # ------------------------------------------------------------------ lines 17-20
    def _n_critique(self, state: RetrievalState) -> dict[str, Any]:
        result = self.critique.run(
            CritiqueInput(
                question=self._question(state),
                grounded=state["grounded"],
                status=state["status"],
                context=state["context"],
            ),
            trace_context=self._tc(state),
        )
        update: dict[str, Any] = {
            "critique": result.output,
            "agent_errors": self._record(state, result),
        }
        status = state["status"]
        critique = result.output
        if (
            critique is not None
            and status.valid
            and critique.major_issues
            and self.config.critique.can_invalidate
        ):
            status = status.model_copy(
                update={
                    "valid": False,
                    "reasons": [
                        *status.reasons,
                        f"self-critique: {len(critique.major_issues)} major issue(s)",
                    ],
                }
            )
            update["status"] = status
        attempts = list(state["attempts"])
        attempts[-1] = attempts[-1].model_copy(
            update={
                "critique": critique.verdict if critique else None,
                "valid": status.valid,
                "validation": status,
            }
        )
        update["attempts"] = attempts
        return update

    def _n_feedback(self, state: RetrievalState) -> dict[str, Any]:
        result = self.feedback.run(
            FeedbackInput(
                query=state["query"],
                attempt=state["attempt"],
                grounded=state["grounded"],
                status=state["status"],
                critique=state.get("critique"),
            ),
            trace_context=self._tc(state),
        )
        update: dict[str, Any] = {
            "feedback": result.output,
            "agent_errors": self._record(state, result),
        }
        fb = result.output
        if fb is not None:
            attempts = list(state["attempts"])
            attempts[-1] = attempts[-1].model_copy(
                update={
                    "feedback": {
                        "mode": fb.mode,
                        "updates": len(fb.updates),
                        "flagged": fb.flagged,
                    }
                }
            )
            update["attempts"] = attempts
            if fb.changed_ids:  # penalised knowledge: drop cached answers that cite it
                inv = self.cache.run(
                    CacheRequest(op="invalidate", element_ids=fb.changed_ids),
                    trace_context=self._tc(state),
                )
                if inv.output is not None:
                    update["invalidated"] = [*state.get("invalidated", []), *inv.output.invalidated]
        return update

    def _n_retry(self, state: RetrievalState) -> dict[str, Any]:
        result = self.retry.run(
            RetryInput(status=state["status"], attempt=state["attempt"], params=state["params"]),
            trace_context=self._tc(state),
        )
        decision = result.output or RetryDecision(
            action="exhausted",
            attempt=state["attempt"],
            params=state["params"],
            reason="retry controller failed",
        )
        update: dict[str, Any] = {
            "retry": decision,
            "retries": [*state.get("retries", []), decision],
            "agent_errors": self._record(state, result),
        }
        if decision.action == "retry":  # line 6: restart from the Vector Retrieval Agent
            update.update(self._fresh_loop(state["decomposition"]))
            update["params"] = decision.params
            update["attempt"] = decision.attempt
        return update

    # ------------------------------------------------------------------ lines 21-23
    def _best_attempt(self, state: RetrievalState) -> AttemptRecord | None:
        attempts = state.get("attempts", [])
        if not attempts:
            return None
        valid = [a for a in attempts if a.valid]
        if valid:
            return valid[-1]
        return max(attempts, key=lambda a: (a.score, -a.attempt))

    def _n_cache_store(self, state: RetrievalState) -> dict[str, Any]:
        status = state.get("status")
        verified = bool(status and status.valid)
        grounded = state.get("grounded")
        paths = [
            p.text
            for r in state.get("results", {}).values()
            if r.graph is not None
            for p in r.graph.paths[:5]
        ]
        context = state.get("context")
        result = self.cache.run(
            CacheRequest(
                op="store",
                query=state["query"],
                history_hash=state["history_hash"],
                verified=verified,
                grounded=grounded,
                score=status.score if status else 0.0,
                report_id=state["report_id"],
                reasoning_paths=paths,
                evidence_ids=[b.evidence_id for b in context.blocks] if context else [],
            ),
            trace_context=self._tc(state),
        )
        return {"cache_store": result.output, "agent_errors": self._record(state, result)}

    def _n_report(self, state: RetrievalState) -> dict[str, Any]:
        report = self._build_report(state)
        self._report_dir.mkdir(parents=True, exist_ok=True)
        path = self._report_dir / f"{state['report_id']}.json"
        report["report_path"] = str(path)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return {"report": report}

    def _trace_summary(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        records = self.ctx.traces.records
        per_call = [
            {
                "agent": t.agent,
                "algorithm": t.algorithm,
                "latency_ms": t.latency_ms,
                "tokens": t.token_usage.total,
                "score": t.confidence,
                "status": t.status,
                "error": t.error,
                **t.context,
            }
            for t in records
        ]
        summary: dict[str, dict[str, Any]] = {}
        for t in records:
            row = summary.setdefault(
                t.agent,
                {
                    "algorithm": t.algorithm,
                    "calls": 0,
                    "errors": 0,
                    "latency_ms": 0.0,
                    "tokens": 0,
                    "scores": [],
                },
            )
            row["calls"] += 1
            row["errors"] += t.status == "error"
            row["latency_ms"] = round(row["latency_ms"] + t.latency_ms, 3)
            row["tokens"] += t.token_usage.total
            if t.confidence is not None:
                row["scores"].append(t.confidence)
        for row in summary.values():
            scores = row.pop("scores")
            row["mean_score"] = round(sum(scores) / len(scores), 4) if scores else None
        return per_call, summary

    def _build_report(self, state: RetrievalState) -> dict[str, Any]:
        lookup = state.get("cache_lookup")
        cached = bool(lookup and lookup.hit and lookup.entry)
        per_call, summary = self._trace_summary()
        base: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "report_id": state["report_id"],
            "generated_at": datetime.now(UTC).isoformat(),
            "query": state["query"],
            "history_turns": len(state.get("history", [])),
            "cached": cached,
            "backends": {"graph": self.kg.backend, "vectors": self.vdb.backend},
        }
        if cached:
            assert lookup is not None and lookup.entry is not None
            entry = lookup.entry
            grounded = GroundedAnswer(answer=entry.answer, statements=entry.statements)
            integrity = self.validation.check_integrity(grounded)
            base.update(
                {
                    "answer": entry.answer,
                    "verified": integrity.passed,
                    "low_confidence": not integrity.passed,
                    "citations": [s.model_dump() for s in entry.statements],
                    "citation_integrity": _integrity_dict(integrity),
                    "cache": {
                        "lookup": lookup.model_dump(exclude={"entry"}),
                        "source_report_id": entry.report_id,
                        "score": entry.score,
                        "reasoning_paths": entry.reasoning_paths,
                    },
                    "attempts": [],
                }
            )
        else:
            best = self._best_attempt(state)
            status = best.validation if best else None
            grounded_best = best.grounded if best else None
            verified = bool(best and best.valid)
            integrity_opt = status.integrity if status else None
            context = state.get("context")
            store = state.get("cache_store")
            base.update(
                {
                    "answer": best.answer if best else "",
                    "verified": verified,
                    "low_confidence": not verified,
                    "best_attempt": best.attempt if best else None,
                    "classification": _dump(state.get("classification")),
                    "rewritten_query": _dump(state.get("rewritten")),
                    "decomposition": {
                        **(_dump(state.get("decomposition")) or {}),
                        "waves": state.get("waves", []),
                        "intermediate_answers": state.get("sub_answers", {}),
                    },
                    "final_params": _dump(state.get("params")),
                    "citations": [s.model_dump() for s in grounded_best.statements]
                    if grounded_best
                    else [],
                    "citation_integrity": _integrity_dict(integrity_opt) if integrity_opt else None,
                    "validation": {
                        "valid": status.valid,
                        "score": status.score,
                        "tau": status.tau,
                        "grounding": status.grounding,
                        "completeness": status.completeness,
                        "insufficient_evidence": status.insufficient_evidence,
                        "claims": [c.model_dump() for c in status.claims],
                        "checks": [c.model_dump() for c in status.validation.checks],
                        "reasons": status.reasons,
                    }
                    if status
                    else None,
                    "attempts": [
                        {
                            "attempt": a.attempt,
                            "escalation": a.escalation,
                            "params": a.params.model_dump(),
                            "score": a.score,
                            "valid": a.valid,
                            "unsupported": a.unsupported,
                            "critique": a.critique,
                            "feedback": a.feedback,
                            "answer": a.answer,
                        }
                        for a in state.get("attempts", [])
                    ],
                    "retry_history": [d.model_dump() for d in state.get("retries", [])],
                    "critique": _dump(state.get("critique")),
                    "feedback": _dump(state.get("feedback")),
                    "context": {
                        "blocks": [
                            b.model_dump(exclude={"text"}) | {"preview": b.text[:160]}
                            for b in context.blocks
                        ],
                        "token_count": context.token_count,
                        "token_budget": context.token_budget,
                        "dropped": context.dropped,
                    }
                    if context
                    else None,
                    "evidence": {
                        sid: {
                            "resolved_text": r.resolved_text,
                            "chunks": len(r.vector.chunks) if r.vector else 0,
                            "entities": len(r.vector.entities) if r.vector else 0,
                            "facts": len(r.graph.facts) if r.graph else 0,
                            "paths": [p.text for p in r.graph.paths[:5]] if r.graph else [],
                            "communities": [c.community_id for c in r.communities.items]
                            if r.communities
                            else [],
                            "aggregated": len(r.aggregated.items) if r.aggregated else 0,
                            "error": r.error,
                        }
                        for sid, r in state.get("results", {}).items()
                    },
                    "cache": {
                        "lookup": lookup.model_dump(exclude={"entry"}) if lookup else None,
                        "store": store.model_dump(exclude={"entry"}) if store else None,
                    },
                }
            )
        base["cache"] = {**base.get("cache", {}), "invalidated": state.get("invalidated", [])}
        base.update(
            {
                "review_items": len(self.ctx.review_queue.items) - self._flags_before,
                "agent_errors": state.get("agent_errors", []),
                "trace": per_call,
                "agents": summary,
                "total_latency_ms": round(
                    (datetime.now(UTC).timestamp() - state["started"]) * 1000, 3
                ),
                "total_tokens": sum(row["tokens"] for row in summary.values()),
            }
        )
        return base


def _dump(value: BaseModel | None) -> dict[str, Any] | None:
    return value.model_dump(mode="json") if value is not None else None


def _integrity_dict(integrity: Any) -> dict[str, Any]:
    return {
        "passed": integrity.passed,
        "resolved": sum(c.resolved for c in integrity.checks),
        "total": len(integrity.checks),
        "unresolved": integrity.unresolved,
        "invalid_markers": integrity.invalid_markers,
        "checks": [c.model_dump() for c in integrity.checks],
    }


def build_retrieval_pipeline(
    config_path: str | Path | None = None, **kwargs: Any
) -> AgenticGraphRetrievalFramework:
    config = load_retrieval_config(config_path)
    configure_logging(config.log_level)
    return AgenticGraphRetrievalFramework(config, **kwargs)


def load_report(report_id: str, config: RetrievalConfig) -> dict[str, Any] | None:
    """Read a persisted validation report (used by ``GET /report/{id}``)."""
    if not report_id.startswith("rpt-") or not report_id[4:].isalnum():
        return None
    path = Path(config.paths.output_dir) / "reports" / f"{report_id}.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
