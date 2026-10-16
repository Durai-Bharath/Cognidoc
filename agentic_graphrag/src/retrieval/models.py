"""Stage 3 data model: queries, subqueries, evidence, context, answers, citations, reports."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.validation import ValidationOutcome
from construction.models import GraphEdge, GraphNode, stable_hash


class QueryType(StrEnum):
    LOCAL = "Local"
    GLOBAL = "Global"
    HYBRID = "Hybrid"
    SIMPLE_LOOKUP = "Simple Lookup"


class Turn(BaseModel):
    """One conversation turn (history for A2's reference resolution)."""

    role: Literal["user", "assistant"] = "user"
    content: str


def history_hash(history: list[Turn]) -> str:
    """Stable key for the conversation history ("" when there is none)."""
    if not history:
        return ""
    return stable_hash(*[f"{t.role}:{t.content}" for t in history])


# =========================================================================== A1-A3


class Classification(BaseModel):
    query_type: QueryType
    confidence: float = Field(ge=0, le=1)
    entity_specific: bool = False
    thematic: bool = False
    entities: list[str] = Field(default_factory=list, description="KG entity names in Q.")
    entity_ids: list[str] = Field(default_factory=list)
    cues: list[str] = Field(default_factory=list, description="Thematic cues found in Q.")
    method: Literal["llm", "heuristic"] = "heuristic"
    rationale: str = ""


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
    query_type: QueryType
    resolved_references: dict[str, str] = Field(default_factory=dict)
    expansions: dict[str, str] = Field(default_factory=dict)
    corrections: dict[str, str] = Field(default_factory=dict)
    enrichment: list[str] = Field(default_factory=list)
    method: Literal["llm", "heuristic"] = "heuristic"


_PLACEHOLDER = re.compile(r"\{(q\d+)\}")


class Subquery(BaseModel):
    id: str
    text: str
    depends_on: list[str] = Field(default_factory=list)

    def placeholders(self) -> list[str]:
        return _PLACEHOLDER.findall(self.text)

    def resolve(self, answers: dict[str, str]) -> str:
        """Substitute ``{qN}`` placeholders with earlier subquery answers."""
        return _PLACEHOLDER.sub(lambda m: answers.get(m.group(1), m.group(0)), self.text)


class Decomposition(BaseModel):
    subqueries: list[Subquery]
    complex: bool = False
    method: Literal["llm", "heuristic"] = "heuristic"

    def waves(self) -> list[list[str]]:
        """Topological layers: every subquery's dependencies sit in earlier waves."""
        done: set[str] = set()
        remaining = [s for s in self.subqueries]
        out: list[list[str]] = []
        while remaining:
            ready = [s.id for s in remaining if set(s.depends_on) <= done]
            if not ready:  # cannot happen after validation; never loop forever
                ready = [remaining[0].id]
            out.append(ready)
            done.update(ready)
            remaining = [s for s in remaining if s.id not in done]
        return out


# =========================================================================== parameters


class RetrievalParams(BaseModel):
    """What A14 modifies between attempts."""

    query_type: QueryType
    top_k: int
    entity_top_k: int
    hops: int
    community_weight: float
    community_top_n: int


# =========================================================================== A4-A7 evidence


class ChunkHit(BaseModel):
    chunk_id: str
    doc_id: str = ""
    page: int | None = None
    section: str = ""
    text: str = ""
    score: float = 0.0
    source: Literal["vector", "graph"] = "vector"


class EntityHit(BaseModel):
    node_id: str
    name: str
    type: str = ""
    score: float = 0.0


class VectorEvidence(BaseModel):
    subquery_id: str
    query_text: str
    chunks: list[ChunkHit] = Field(default_factory=list)
    entities: list[EntityHit] = Field(default_factory=list)
    stale_hits: list[str] = Field(default_factory=list, description="Vector ids with no node.")


class FactEvidence(BaseModel):
    """One relationship reached by traversal, with the chunks that assert it."""

    edge_id: str
    source_id: str
    source: str
    predicate: str
    target_id: str
    target: str
    confidence: float
    hop: int = 1
    score: float = Field(0.0, description="Best reasoning-path score through this edge.")
    chunk_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)

    @property
    def sentence(self) -> str:
        return f"{self.source} {self.predicate.replace('_', ' ')} {self.target}."


class ReasoningPath(BaseModel):
    seed_id: str
    edge_ids: list[str]
    node_ids: list[str]
    text: str
    score: float


class GraphEvidence(BaseModel):
    subquery_id: str
    seeds: list[str] = Field(default_factory=list)
    visited: list[str] = Field(default_factory=list)
    facts: list[FactEvidence] = Field(default_factory=list)
    paths: list[ReasoningPath] = Field(default_factory=list)
    hops: int = 0


class CommunityHit(BaseModel):
    community_id: str
    title: str
    summary: str
    member_ids: list[str] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    relevance: float = 0.0
    overlap: float = 0.0
    weight: float = 0.0
    score: float = 0.0


class CommunityEvidence(BaseModel):
    subquery_id: str
    weight: float = 0.0
    items: list[CommunityHit] = Field(default_factory=list)


EvidenceKind = Literal["chunk", "fact", "path", "community"]


class EvidenceItem(BaseModel):
    evidence_id: str
    kind: EvidenceKind
    text: str
    relevance: float = 0.0
    confidence: float = 1.0
    score: float = 0.0
    chunk_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    page: int | None = None
    doc_id: str = ""
    sources: list[str] = Field(default_factory=list, description="vector / graph / community")
    subquery_ids: list[str] = Field(default_factory=list)


class AggregatedEvidence(BaseModel):
    subquery_id: str
    items: list[EvidenceItem] = Field(default_factory=list)
    duplicates_removed: int = 0


class SubqueryResult(BaseModel):
    subquery: Subquery
    resolved_text: str
    vector: VectorEvidence | None = None
    graph: GraphEvidence | None = None
    communities: CommunityEvidence | None = None
    aggregated: AggregatedEvidence | None = None
    answer: str | None = Field(None, description="Intermediate answer (only if depended on).")
    error: str | None = None


# =========================================================================== A8 context


class ContextBlock(BaseModel):
    citation_id: str
    evidence_id: str
    kind: EvidenceKind
    text: str
    chunk_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    page: int | None = None
    doc_id: str = ""
    doc_title: str = ""
    score: float = 0.0
    tokens: int = 0
    truncated: bool = False

    def render(self) -> str:
        where = [f"type={self.kind}"]
        if self.doc_title or self.doc_id:
            where.append(f"doc={self.doc_title or self.doc_id}")
        if self.page:
            where.append(f"page={self.page}")
        return f"[{self.citation_id}] ({', '.join(where)}) {self.text}"


class RetrievalContext(BaseModel):
    query: str
    query_type: QueryType
    blocks: list[ContextBlock] = Field(default_factory=list)
    sub_answers: dict[str, str] = Field(default_factory=dict)
    token_count: int = 0
    token_budget: int = 0
    dropped: int = 0

    def block(self, citation_id: str) -> ContextBlock | None:
        return next((b for b in self.blocks if b.citation_id == citation_id), None)

    def render(self) -> str:
        return "\n".join(b.render() for b in self.blocks)


# =========================================================================== A9-A11


class GeneratedAnswer(BaseModel):
    answer: str
    insufficient_evidence: bool = False
    short_answer: str = ""


class SourceRef(BaseModel):
    citation_id: str
    kind: EvidenceKind
    chunk_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    page: int | None = None
    doc_id: str = ""
    doc_title: str = ""


class StatementCitation(BaseModel):
    statement_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list, description="Graph nodes named here.")
    inferred: bool = Field(False, description="No marker; best supporting block attached.")
    invalid_markers: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    answer: str
    statements: list[StatementCitation] = Field(default_factory=list)
    citations: list[SourceRef] = Field(default_factory=list)
    invalid_markers: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    def referenced_ids(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {"chunk": set(), "node": set(), "edge": set()}
        for s in self.statements:
            for src in s.sources:
                out["chunk"].update(src.chunk_ids)
                out["node"].update(src.node_ids)
                out["edge"].update(src.edge_ids)
            out["node"].update(s.entity_ids)
        return out


class ClaimCheck(BaseModel):
    statement_id: str
    text: str
    support: float
    supported: bool
    missing_terms: list[str] = Field(default_factory=list)
    reason: str = ""


class IntegrityCheck(BaseModel):
    ref_id: str
    kind: Literal["chunk", "node", "edge"]
    store: str
    resolved: bool


class CitationIntegrity(BaseModel):
    checks: list[IntegrityCheck] = Field(default_factory=list)
    invalid_markers: list[str] = Field(default_factory=list)

    @property
    def unresolved(self) -> list[str]:
        return [c.ref_id for c in self.checks if not c.resolved]

    @property
    def passed(self) -> bool:
        return not self.unresolved and not self.invalid_markers


class ValidationStatus(BaseModel):
    valid: bool
    score: float
    grounding: float
    completeness: float
    tau: float
    claims: list[ClaimCheck] = Field(default_factory=list)
    integrity: CitationIntegrity = Field(default_factory=CitationIntegrity)
    insufficient_evidence: bool = False
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    reasons: list[str] = Field(default_factory=list)

    @property
    def unsupported(self) -> list[str]:
        return [c.statement_id for c in self.claims if not c.supported]


# =========================================================================== A12-A14


class CritiqueIssue(BaseModel):
    statement_id: str | None = None
    issue_type: Literal["unsupported", "contradiction", "logical", "citation"] = "unsupported"
    severity: Literal["major", "minor"] = "major"
    description: str = ""
    correction: str = ""


class VerifiedAnswer(BaseModel):
    verdict: Literal["verified", "needs_revision"]
    issues: list[CritiqueIssue] = Field(default_factory=list)
    corrected_answer: str | None = None
    implicated_node_ids: list[str] = Field(default_factory=list)
    implicated_edge_ids: list[str] = Field(default_factory=list)
    supported_node_ids: list[str] = Field(default_factory=list)
    supported_edge_ids: list[str] = Field(default_factory=list)
    used_llm: bool = False

    @property
    def major_issues(self) -> list[CritiqueIssue]:
        return [i for i in self.issues if i.severity == "major"]


class FeedbackUpdate(BaseModel):
    element_id: str
    kind: Literal["node", "edge"]
    direction: Literal["penalize", "reinforce"]
    before: float
    after: float


class FeedbackReport(BaseModel):
    mode: Literal["penalize", "reinforce", "none"] = "none"
    updates: list[FeedbackUpdate] = Field(default_factory=list)
    flagged: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    changed_ids: list[str] = Field(default_factory=list, description="Penalised/flagged ids.")
    logged: bool = False


class RetryDecision(BaseModel):
    action: Literal["accept", "retry", "exhausted"]
    attempt: int
    params: RetrievalParams
    escalation: str | None = None
    reason: str = ""


class AttemptRecord(BaseModel):
    attempt: int
    params: RetrievalParams
    escalation: str | None = None
    answer: str = ""
    score: float = 0.0
    valid: bool = False
    unsupported: list[str] = Field(default_factory=list)
    critique: str | None = None
    feedback: dict[str, Any] = Field(default_factory=dict)
    grounded: GroundedAnswer | None = None
    validation: ValidationStatus | None = None


# =========================================================================== A15 cache


class CacheEntry(BaseModel):
    key: str
    query: str
    history_hash: str = ""
    embedding: list[float]
    answer: str
    statements: list[StatementCitation] = Field(default_factory=list)
    reasoning_paths: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    fingerprints: dict[str, str] = Field(default_factory=dict)
    report_id: str = ""
    score: float = 0.0
    created_at: str = ""
    hits: int = 0


def element_fingerprint(element: GraphNode | GraphEdge) -> str:
    """Content signature that Stage 2 changes but retrieval feedback does not.

    Feedback only touches ``edge.confidence`` and node/edge ``properties`` feedback fields,
    so reinforcing an element never invalidates a cache entry; penalties invalidate
    explicitly through A15.
    """
    prov = sorted((p.doc_id, p.chunk_id, round(p.confidence, 4)) for p in element.provenance)
    if isinstance(element, GraphEdge):
        payload: dict[str, Any] = {
            "s": element.source,
            "t": element.target,
            "p": element.predicate,
            "prov": prov,
        }
    else:
        payload = {
            "label": element.label,
            "name": element.name,
            "type": element.type,
            "aliases": sorted(element.aliases),
            "text": element.properties.get("text"),
            "summary": element.properties.get("summary"),
            "prov": prov,
        }
    return stable_hash(json.dumps(payload, sort_keys=True, default=str))
