"""Algorithm 1 - Overall Agentic Graph Construction Framework, as a LangGraph state machine.

Graph nodes map 1:1 to Algorithm 1 lines:

    ingestion(2) → semantic_chunking(3) → next_chunk(4)
    next_chunk → [chunk left?] → triple_extraction(5) → coreference_resolution(6)
        → confidence_evaluation(7) → [score < τ?](8)
            yes → discard_chunk(9-10) → next_chunk
            no  → entity_resolution(12) → relation_normalization(13)
                → conflict_detection(14) → decision(15) → next_operation
    next_operation → [Decision?](16-23)
        New → crud_create(17) | Merge → crud_merge(19) | Update → crud_update(21)
        | else → reject(23)   → next_operation
        (no operations left)  → provenance(24) → embedding(25) → vector_index(26) → next_chunk
    next_chunk → [no chunks left] → construction_validation(27) → graph_health_audit(28)
        → community_builder (post-step, DESIGN_DECISIONS #3) → export(29-32)

Per-chunk failure isolation: any failing per-chunk node records the error on the chunk's
outcome and routes back to ``next_chunk``. Ingestion/chunking failures terminate (Alg. 2).
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from collections.abc import Callable, Hashable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.base_agent import AgentError, AgentResult, TraceRecord, TraceSink
from common.embedder import Embedder, build_embedder
from common.llm import LLMClient, build_llm
from common.logging import configure_logging, get_logger
from common.review_queue import ReviewQueue
from construction.agents import (
    ConfidenceEvaluationAgent,
    ConflictDetectionAgent,
    ConstructionValidationAgent,
    CoreferenceResolutionAgent,
    DecisionAgent,
    DocumentIngestionAgent,
    EmbeddingAgent,
    EntityResolutionAgent,
    GraphCRUDAgent,
    GraphHealthAuditAgent,
    ProvenanceAgent,
    RelationNormalizationAgent,
    SemanticChunkingAgent,
    TripleExtractionAgent,
    VectorIndexAgent,
)
from construction.agents.chunking import ChunkSet
from construction.agents.confidence_evaluation import ConfidenceInput, ValidatedTriples
from construction.agents.conflict_detection import ConflictInput, ConflictReport
from construction.agents.construction_validation import (
    ConstructionValidationReport,
    ValidationRequest,
)
from construction.agents.coreference import CoreferenceInput, CoreferenceOutput
from construction.agents.decision import Decisions
from construction.agents.embedding import EmbeddingBatch, EmbeddingInput
from construction.agents.entity_resolution import EntityResolutionInput, ResolvedTriples
from construction.agents.graph_crud import CRUDInput
from construction.agents.health_audit import AuditReport
from construction.agents.ingestion import IngestionInput, ValidatedDocument
from construction.agents.provenance import ProvenanceInput
from construction.agents.relation_normalization import NormalizationInput, NormalizedTriples
from construction.agents.triple_extraction import TripleExtractionInput, TripleSet
from construction.community import CommunityBuilder, CommunityInput, CommunityReport
from construction.config import ConstructionConfig, load_construction_config
from construction.context import ConstructionContext
from construction.models import ConflictRecord, GraphOperation, Operation, SemanticChunk, utc_now
from construction.ontology import Ontology, load_ontology
from construction.stores import (
    COLLECTIONS,
    GraphStore,
    VectorStore,
    build_graph_store,
    build_vector_store,
    save_state,
)

_log = get_logger(__name__)
REPORT_SCHEMA_VERSION = "1.0.0"


class ChunkOutcome(BaseModel):
    chunk_id: str
    status: Literal["accepted", "skipped_low_confidence", "failed"] = "accepted"
    stage: str | None = None
    error: str | None = None
    triples_extracted: int = 0
    triples_after_coreference: int = 0
    triples_accepted: int = 0
    triples_discarded: int = 0
    statuses: dict[str, int] = Field(default_factory=dict)
    operations: dict[str, int] = Field(default_factory=dict)


class ConstructionState(TypedDict, total=False):
    document: dict[str, Any]
    metadata: dict[str, Any]
    output_dir: str
    validated: ValidatedDocument
    chunks: list[SemanticChunk]
    chunk_index: int
    current: SemanticChunk | None
    outcome: ChunkOutcome | None
    outcomes: list[ChunkOutcome]
    last_subject: str | None
    triples: TripleSet
    coref: CoreferenceOutput
    validated_triples: ValidatedTriples
    resolved: ResolvedTriples
    normalized: NormalizedTriples
    statuses: ConflictReport
    decisions: Decisions
    op_index: int
    touched_nodes: list[str]
    touched_edges: list[str]
    embeddings: EmbeddingBatch
    validation: ConstructionValidationReport
    audit: AuditReport
    communities: CommunityReport
    agent_errors: list[str]
    fatal: AgentError | None
    report: dict[str, Any]
    output_paths: dict[str, str]


class ConstructionResult(BaseModel):
    document_id: str | None
    ok: bool
    report: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: AgentError | None = None
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    traces: list[TraceRecord] = Field(default_factory=list)


_OP_NODES: dict[Operation, str] = {
    Operation.CREATE: "crud_create",
    Operation.MERGE: "crud_merge",
    Operation.UPDATE: "crud_update",
}


def _as_dict(value: dict[str, Any] | BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _chunk_node(
    stage: str,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Wrap a per-chunk node so an unexpected exception fails only the current chunk."""

    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(
            self: AgenticGraphConstructionFramework, state: ConstructionState
        ) -> dict[str, Any]:
            try:
                return fn(self, state)
            except Exception as exc:
                _log.exception("chunk_node_crashed", extra={"stage": stage})
                return self._fail_chunk(state, stage, f"{type(exc).__name__}: {exc}")

        return wrapper

    return deco


class AgenticGraphConstructionFramework:
    """Algorithm 1. Builds the shared context, the 15 agents + CommunityBuilder and the graph."""

    def __init__(
        self,
        config: ConstructionConfig | None = None,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        ontology: Ontology | None = None,
        review_queue: ReviewQueue | None = None,
    ) -> None:
        self.config = config or ConstructionConfig()
        # The ontology is validated at startup: an invalid one raises OntologyError here.
        onto = ontology or load_ontology(self.config.paths.ontology)
        self.ctx = ConstructionContext(
            config=self.config.as_extraction_config(),
            llm=llm or build_llm(self.config.llm),
            embedder=embedder or build_embedder(self.config.embedder),
            review_queue=review_queue or ReviewQueue(self.config.paths.review_queue),
            settings=self.config,
            graph=graph_store or build_graph_store(self.config),
            vectors=vector_store or build_vector_store(self.config),
            ontology=onto,
        )
        c = self.ctx
        self.ingestion = DocumentIngestionAgent(c)
        self.chunking = SemanticChunkingAgent(c)
        self.triple_extraction = TripleExtractionAgent(c)
        self.coreference = CoreferenceResolutionAgent(c)
        self.confidence = ConfidenceEvaluationAgent(c)
        self.entity_resolution = EntityResolutionAgent(c)
        self.relation_normalization = RelationNormalizationAgent(c)
        self.conflict_detection = ConflictDetectionAgent(c)
        self.decision = DecisionAgent(c)
        self.crud = GraphCRUDAgent(c)
        self.provenance = ProvenanceAgent(c)
        self.embedding = EmbeddingAgent(c)
        self.vector_index = VectorIndexAgent(c)
        self.validation = ConstructionValidationAgent(c)
        self.audit = GraphHealthAuditAgent(c)
        self.communities = CommunityBuilder(c)
        self.graph = self._build_graph()
        self.last_state: ConstructionState = {}

    @property
    def kg(self) -> GraphStore:
        return self.ctx.graph

    @property
    def vdb(self) -> VectorStore:
        return self.ctx.vectors

    # ================================================================== public API
    def run(
        self,
        document: dict[str, Any] | BaseModel,
        metadata: dict[str, Any] | BaseModel,
        output_dir: str | Path | None = None,
        *,
        persist_state: bool = False,
    ) -> ConstructionResult:
        self.ctx.traces = TraceSink()
        self.ctx.document_id = None
        self.ctx.session_id = None
        self.ctx.conflicts = []
        self._flags_before = len(self.ctx.review_queue.items)
        self._persist = persist_state
        state: ConstructionState = {
            "document": _as_dict(document),
            "metadata": _as_dict(metadata),
            "output_dir": str(output_dir or self.config.paths.output_dir),
            "chunks": [],
            "chunk_index": -1,
            "current": None,
            "outcome": None,
            "outcomes": [],
            "last_subject": None,
            "agent_errors": [],
            "fatal": None,
            "output_paths": {},
        }
        final: ConstructionState = self.graph.invoke(state, config={"recursion_limit": 1_000_000})
        self.last_state = final
        return ConstructionResult(
            document_id=self.ctx.document_id,
            ok=final.get("fatal") is None,
            report=final.get("report", {}),
            output_paths=final.get("output_paths", {}),
            error=final.get("fatal"),
            conflicts=list(self.ctx.conflicts),
            traces=self.ctx.traces.records,
        )

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())

    # ================================================================== graph
    def _build_graph(self) -> Any:
        g: StateGraph[ConstructionState] = StateGraph(ConstructionState)
        nodes = {
            "ingestion": self._n_ingestion,
            "semantic_chunking": self._n_chunking,
            "next_chunk": self._n_next_chunk,
            "triple_extraction": self._n_triples,
            "coreference_resolution": self._n_coreference,
            "confidence_evaluation": self._n_confidence,
            "discard_chunk": self._n_discard,
            "entity_resolution": self._n_entity_resolution,
            "relation_normalization": self._n_relation_normalization,
            "conflict_detection": self._n_conflict_detection,
            "decision": self._n_decision,
            "next_operation": self._n_next_operation,
            "crud_create": self._n_crud,
            "crud_merge": self._n_crud,
            "crud_update": self._n_crud,
            "reject": self._n_reject,
            "provenance": self._n_provenance,
            "embedding": self._n_embedding,
            "vector_index": self._n_vector_index,
            "construction_validation": self._n_validation,
            "graph_health_audit": self._n_audit,
            "community_builder": self._n_communities,
            "export": self._n_export,
        }
        for name, fn in nodes.items():
            g.add_node(name, fn)

        g.add_edge(START, "ingestion")
        g.add_conditional_edges(
            "ingestion",
            self._fatal_or("semantic_chunking"),
            {"semantic_chunking": "semantic_chunking", "export": "export"},
        )
        g.add_conditional_edges(
            "semantic_chunking",
            self._fatal_or("next_chunk"),
            {"next_chunk": "next_chunk", "export": "export"},
        )
        # Line 4: foreach chunk.
        g.add_conditional_edges(
            "next_chunk",
            self._route_has_chunk,
            {
                "triple_extraction": "triple_extraction",
                "construction_validation": "construction_validation",
            },
        )
        for src, dst in [
            ("triple_extraction", "coreference_resolution"),
            ("coreference_resolution", "confidence_evaluation"),
            ("entity_resolution", "relation_normalization"),
            ("relation_normalization", "conflict_detection"),
            ("conflict_detection", "decision"),
            ("decision", "next_operation"),
            ("provenance", "embedding"),
            ("embedding", "vector_index"),
        ]:
            g.add_conditional_edges(
                src, self._chunk_ok_or(dst), {dst: dst, "next_chunk": "next_chunk"}
            )
        # Lines 8-11: the confidence branch.
        g.add_conditional_edges(
            "confidence_evaluation",
            self._route_confidence,
            {
                "entity_resolution": "entity_resolution",
                "discard_chunk": "discard_chunk",
                "next_chunk": "next_chunk",
            },
        )
        g.add_edge("discard_chunk", "next_chunk")
        # Lines 16-23: the decision branches, once per triple.
        targets: dict[Hashable, str] = {
            n: n for n in ["crud_create", "crud_merge", "crud_update", "reject", "provenance"]
        }
        g.add_conditional_edges("next_operation", self._route_operation, targets)
        for n in ("crud_create", "crud_merge", "crud_update", "reject"):
            g.add_edge(n, "next_operation")
        # Lines 24-26.
        g.add_edge("vector_index", "next_chunk")
        # Lines 27-32.
        g.add_edge("construction_validation", "graph_health_audit")
        g.add_edge("graph_health_audit", "community_builder")
        g.add_edge("community_builder", "export")
        g.add_edge("export", END)
        return g.compile()

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _fatal_or(target: str) -> Callable[[ConstructionState], str]:
        def route(state: ConstructionState) -> str:
            return "export" if state.get("fatal") else target

        return route

    @staticmethod
    def _chunk_ok_or(target: str) -> Callable[[ConstructionState], str]:
        def route(state: ConstructionState) -> str:
            outcome = state.get("outcome")
            return "next_chunk" if outcome is not None and outcome.status == "failed" else target

        return route

    @staticmethod
    def _route_has_chunk(state: ConstructionState) -> str:
        return (
            "triple_extraction" if state.get("current") is not None else "construction_validation"
        )

    @staticmethod
    def _route_confidence(state: ConstructionState) -> str:
        outcome = state.get("outcome")
        if outcome is not None and outcome.status == "failed":
            return "next_chunk"
        return "entity_resolution" if state["validated_triples"].chunk_passes else "discard_chunk"

    @staticmethod
    def _current_op(state: ConstructionState) -> GraphOperation | None:
        ops = state["decisions"].operations
        i = state.get("op_index", 0)
        return ops[i] if 0 <= i < len(ops) else None

    def _route_operation(self, state: ConstructionState) -> str:
        op = self._current_op(state)
        if op is None:
            return "provenance"
        return _OP_NODES.get(op.op, "reject")

    # ------------------------------------------------------------------ helpers
    def _record(self, state: ConstructionState, result: AgentResult[Any]) -> list[str]:
        errors = list(state.get("agent_errors", []))
        if result.error:
            ref = state.get("current")
            where = f" [{ref.chunk_id}]" if ref is not None else ""
            errors.append(f"{result.error.agent}{where}: {result.error.message}")
        return errors

    def _fail_chunk(self, state: ConstructionState, stage: str, message: str) -> dict[str, Any]:
        outcome = state.get("outcome")
        chunk = state.get("current")
        if outcome is None and chunk is not None:
            outcome = ChunkOutcome(chunk_id=chunk.chunk_id)
        update: dict[str, Any] = {}
        if outcome is not None:
            update["outcome"] = outcome.model_copy(
                update={"status": "failed", "stage": stage, "error": message}
            )
        if chunk is not None:
            self.ctx.review_queue.flag(
                agent=stage,
                reason="chunk failed and was skipped",
                document_id=self.ctx.document_id,
                item_ref=chunk.chunk_id,
                details={"error": message},
            )
        errors = list(state.get("agent_errors", []))
        errors.append(f"{stage} [{chunk.chunk_id if chunk else '-'}]: {message}")
        update["agent_errors"] = errors
        return update

    def _chunk_result(
        self, state: ConstructionState, stage: str, result: AgentResult[Any]
    ) -> dict[str, Any] | None:
        """Failure update for a per-chunk agent result, or None when it succeeded."""
        if result.ok:
            return None
        update = self._fail_chunk(
            state, stage, result.error.message if result.error else "no output"
        )
        update["agent_errors"] = self._record(state, result)
        return update

    # ------------------------------------------------------------------ nodes: lines 1-3
    def _n_ingestion(self, state: ConstructionState) -> dict[str, Any]:
        result = self.ingestion.run(
            IngestionInput(document=state["document"], metadata=state["metadata"])
        )
        if not result.ok:
            return {"fatal": result.error, "agent_errors": self._record(state, result)}
        return {"validated": result.unwrap()}

    def _n_chunking(self, state: ConstructionState) -> dict[str, Any]:
        result = self.chunking.run(state["validated"])
        if not result.ok:
            return {"fatal": result.error, "agent_errors": self._record(state, result)}
        chunks: ChunkSet = result.unwrap()
        return {"chunks": chunks.chunks, "chunk_index": -1}

    # ------------------------------------------------------------------ line 4: chunk loop
    def _n_next_chunk(self, state: ConstructionState) -> dict[str, Any]:
        outcomes = list(state.get("outcomes", []))
        if state.get("outcome") is not None:
            outcomes.append(state["outcome"])  # type: ignore[arg-type]
        i = state.get("chunk_index", -1) + 1
        chunks = state.get("chunks", [])
        current = chunks[i] if i < len(chunks) else None
        return {
            "outcomes": outcomes,
            "chunk_index": i,
            "current": current,
            "outcome": ChunkOutcome(chunk_id=current.chunk_id) if current else None,
            "touched_nodes": [],
            "touched_edges": [],
            "op_index": 0,
        }

    def _ctx_trace(self, state: ConstructionState) -> dict[str, str]:
        c = state.get("current")
        return {"chunk": c.chunk_id} if c else {}

    @_chunk_node("triple_extraction")
    def _n_triples(self, state: ConstructionState) -> dict[str, Any]:
        chunk = state["current"]
        assert chunk is not None
        result = self.triple_extraction.run(
            TripleExtractionInput(chunk=chunk, document_title=state["validated"].title),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "triple_extraction", result)) is not None:
            return fail
        triples = result.unwrap()
        outcome = state["outcome"]
        assert outcome is not None
        return {
            "triples": triples,
            "outcome": outcome.model_copy(update={"triples_extracted": len(triples.triples)}),
        }

    @_chunk_node("coreference_resolution")
    def _n_coreference(self, state: ConstructionState) -> dict[str, Any]:
        v = state["validated"]
        result = self.coreference.run(
            CoreferenceInput(
                triples=state["triples"],
                abbreviations=v.abbreviations,
                document_title=v.title,
                previous_subject=state.get("last_subject"),
            ),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "coreference_resolution", result)) is not None:
            return fail
        coref = result.unwrap()
        outcome = state["outcome"]
        assert outcome is not None
        return {
            "coref": coref,
            "last_subject": coref.last_subject,
            "outcome": outcome.model_copy(
                update={"triples_after_coreference": len(coref.triples.triples)}
            ),
        }

    @_chunk_node("confidence_evaluation")
    def _n_confidence(self, state: ConstructionState) -> dict[str, Any]:
        chunk = state["current"]
        assert chunk is not None
        result = self.confidence.run(
            ConfidenceInput(
                triples=state["coref"].triples,
                chunk=chunk,
                abbreviations=state["validated"].abbreviations,
            ),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "confidence_evaluation", result)) is not None:
            return fail
        vt = result.unwrap()
        outcome = state["outcome"]
        assert outcome is not None
        return {
            "validated_triples": vt,
            "outcome": outcome.model_copy(
                update={
                    "triples_accepted": len(vt.accepted),
                    "triples_discarded": len(vt.discarded),
                }
            ),
        }

    def _n_discard(self, state: ConstructionState) -> dict[str, Any]:
        """Lines 8-10: discard the extracted triples and continue with the next chunk."""
        outcome = state["outcome"]
        assert outcome is not None
        return {"outcome": outcome.model_copy(update={"status": "skipped_low_confidence"})}

    # ------------------------------------------------------------------ lines 12-15
    @_chunk_node("entity_resolution")
    def _n_entity_resolution(self, state: ConstructionState) -> dict[str, Any]:
        result = self.entity_resolution.run(
            EntityResolutionInput(triples=state["validated_triples"].accepted),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "entity_resolution", result)) is not None:
            return fail
        return {"resolved": result.unwrap()}

    @_chunk_node("relation_normalization")
    def _n_relation_normalization(self, state: ConstructionState) -> dict[str, Any]:
        result = self.relation_normalization.run(
            NormalizationInput(items=state["resolved"].items), trace_context=self._ctx_trace(state)
        )
        if (fail := self._chunk_result(state, "relation_normalization", result)) is not None:
            return fail
        return {"normalized": result.unwrap()}

    @_chunk_node("conflict_detection")
    def _n_conflict_detection(self, state: ConstructionState) -> dict[str, Any]:
        result = self.conflict_detection.run(
            ConflictInput(items=state["normalized"].items), trace_context=self._ctx_trace(state)
        )
        if (fail := self._chunk_result(state, "conflict_detection", result)) is not None:
            return fail
        report = result.unwrap()
        outcome = state["outcome"]
        assert outcome is not None
        counts = Counter(i.status.value for i in report.items)
        return {
            "statuses": report,
            "outcome": outcome.model_copy(update={"statuses": dict(counts)}),
        }

    @_chunk_node("decision")
    def _n_decision(self, state: ConstructionState) -> dict[str, Any]:
        result = self.decision.run(state["statuses"], trace_context=self._ctx_trace(state))
        if (fail := self._chunk_result(state, "decision", result)) is not None:
            return fail
        return {"decisions": result.unwrap(), "op_index": -1}

    # ------------------------------------------------------------------ lines 16-23
    def _n_next_operation(self, state: ConstructionState) -> dict[str, Any]:
        return {"op_index": state.get("op_index", -1) + 1}

    def _bump(self, state: ConstructionState, op: Operation) -> ChunkOutcome:
        outcome = state["outcome"]
        assert outcome is not None
        ops = dict(outcome.operations)
        ops[op.value] = ops.get(op.value, 0) + 1
        return outcome.model_copy(update={"operations": ops})

    def _n_crud(self, state: ConstructionState) -> dict[str, Any]:
        """Lines 17 / 19 / 21: Create, Merge or Update through the Graph CRUD Agent."""
        op = self._current_op(state)
        assert op is not None
        result = self.crud.run(CRUDInput(operations=[op]), trace_context=self._ctx_trace(state))
        errors = self._record(state, result)
        if not result.ok:
            return {"agent_errors": errors}
        crud = result.unwrap()
        if crud.errors:  # operation-level isolation: log, skip this triple, keep going
            current = state.get("current")
            where = current.chunk_id if current is not None else "-"
            errors += [f"graph_crud [{where}]: {e}" for e in crud.errors]
            return {"agent_errors": errors}
        return {
            "touched_nodes": list(
                dict.fromkeys([*state.get("touched_nodes", []), *crud.touched_nodes])
            ),
            "touched_edges": list(
                dict.fromkeys([*state.get("touched_edges", []), *crud.touched_edges])
            ),
            "outcome": self._bump(state, op.op),
            "agent_errors": errors,
        }

    def _n_reject(self, state: ConstructionState) -> dict[str, Any]:
        """Line 23: reject conflicting knowledge (the ConflictRecord is already logged by A9)."""
        return {"outcome": self._bump(state, Operation.REJECT)}

    # ------------------------------------------------------------------ lines 24-26
    @_chunk_node("provenance")
    def _n_provenance(self, state: ConstructionState) -> dict[str, Any]:
        chunk = state["current"]
        assert chunk is not None
        result = self.provenance.run(
            ProvenanceInput(
                chunk=chunk,
                touched_nodes=state.get("touched_nodes", []),
                touched_edges=state.get("touched_edges", []),
            ),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "provenance", result)) is not None:
            return fail
        rep = result.unwrap()
        return {"touched_nodes": list(dict.fromkeys([rep.chunk_node_id, *rep.entity_ids]))}

    @_chunk_node("embedding")
    def _n_embedding(self, state: ConstructionState) -> dict[str, Any]:
        result = self.embedding.run(
            EmbeddingInput(node_ids=state.get("touched_nodes", [])),
            trace_context=self._ctx_trace(state),
        )
        if (fail := self._chunk_result(state, "embedding", result)) is not None:
            return fail
        return {"embeddings": result.unwrap()}

    @_chunk_node("vector_index")
    def _n_vector_index(self, state: ConstructionState) -> dict[str, Any]:
        result = self.vector_index.run(state["embeddings"], trace_context=self._ctx_trace(state))
        if (fail := self._chunk_result(state, "vector_index", result)) is not None:
            return fail
        return {}

    # ------------------------------------------------------------------ lines 27-28 + post
    def _n_validation(self, state: ConstructionState) -> dict[str, Any]:
        outcomes = list(state.get("outcomes", []))
        result = self.validation.run(ValidationRequest(document_id=self.ctx.document_id))
        update: dict[str, Any] = {"outcomes": outcomes, "agent_errors": self._record(state, result)}
        if result.ok:
            update["validation"] = result.unwrap()
        return update

    def _n_audit(self, state: ConstructionState) -> dict[str, Any]:
        validation = state.get("validation") or ConstructionValidationReport()
        result = self.audit.run(validation)
        update: dict[str, Any] = {"agent_errors": self._record(state, result)}
        if result.ok:
            update["audit"] = result.unwrap()
        return update

    def _n_communities(self, state: ConstructionState) -> dict[str, Any]:
        if not self.config.community.enabled:
            return {}
        result = self.communities.run(CommunityInput(document_id=self.ctx.document_id))
        update: dict[str, Any] = {"agent_errors": self._record(state, result)}
        if result.ok:
            update["communities"] = result.unwrap()
        return update

    # ------------------------------------------------------------------ lines 29-32
    def _n_export(self, state: ConstructionState) -> dict[str, Any]:
        out = Path(state.get("output_dir") or self.config.paths.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        if state.get("fatal") is None:
            paths["knowledge_graph"] = self.kg.export(out / "knowledge_graph.json")
            paths["vector_database"] = self.vdb.export(out / "vector_database.json")
            if self._persist:
                paths.update(save_state(self.kg, self.vdb, self.config.paths.state_dir))
        report = self._build_report(state)
        report_path = out / "construction_report.json"
        paths["construction_report"] = str(report_path)
        report["outputs"] = paths
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return {"report": report, "output_paths": paths}

    def _build_report(self, state: ConstructionState) -> dict[str, Any]:
        outcomes: list[ChunkOutcome] = list(state.get("outcomes", []))
        fatal = state.get("fatal")
        v = state.get("validated")
        statuses: Counter[str] = Counter()
        ops: Counter[str] = Counter()
        for o in outcomes:
            statuses.update(o.statuses)
            ops.update(o.operations)
        by_status = Counter(o.status for o in outcomes)
        traces: dict[str, dict[str, Any]] = {}
        for t in self.ctx.traces.records:
            row = traces.setdefault(
                t.agent, {"calls": 0, "errors": 0, "latency_ms": 0.0, "tokens": 0}
            )
            row["calls"] += 1
            row["errors"] += t.status == "error"
            row["latency_ms"] = round(row["latency_ms"] + t.latency_ms, 3)
            row["tokens"] += t.token_usage.total
        validation = state.get("validation")
        audit = state.get("audit")
        communities = state.get("communities")
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "rejected" if fatal else "completed",
            "error": fatal.model_dump() if fatal else None,
            "document_id": self.ctx.document_id,
            "source_document_id": v.source_document_id if v else None,
            "title": v.title if v else None,
            "session_id": self.ctx.session_id,
            "reingested": v.reingested if v else False,
            "backends": {"graph": self.kg.backend, "vectors": self.vdb.backend},
            "chunks": {
                "total": len(state.get("chunks", [])),
                "accepted": by_status.get("accepted", 0),
                "skipped_low_confidence": by_status.get("skipped_low_confidence", 0),
                "failed": by_status.get("failed", 0),
                "outcomes": [o.model_dump() for o in outcomes],
            },
            "triples": {
                "extracted": sum(o.triples_extracted for o in outcomes),
                "after_coreference": sum(o.triples_after_coreference for o in outcomes),
                "accepted": sum(o.triples_accepted for o in outcomes),
                "discarded_low_confidence": sum(o.triples_discarded for o in outcomes),
            },
            "conflict_statuses": dict(statuses),
            "operations": {
                op.value: ops.get(op.value, 0) for op in Operation if op is not Operation.DELETE
            },
            "conflicts": [c.model_dump() for c in self.ctx.conflicts],
            "validation": {
                "passed": validation.passed,
                "checks": [c.model_dump() for c in validation.validation.checks],
                "invalid_entities": validation.invalid_entities,
                "invalid_relations": validation.invalid_relations,
                "stats": validation.stats,
            }
            if validation
            else None,
            "audit": audit.model_dump() if audit else None,
            "communities": communities.model_dump() if communities else None,
            "graph": {"nodes": len(self.kg.nodes()), "edges": len(self.kg.edges())},
            "vectors": {c: self.vdb.count(c) for c in COLLECTIONS},
            "review_items": len(self.ctx.review_queue.items) - self._flags_before,
            "agent_errors": state.get("agent_errors", []),
            "agents": traces,
            "finished_at": utc_now(),
        }


def build_construction_pipeline(
    config_path: str | Path | None = None, **kwargs: Any
) -> AgenticGraphConstructionFramework:
    config = load_construction_config(config_path)
    configure_logging(config.log_level)
    return AgenticGraphConstructionFramework(config, **kwargs)
