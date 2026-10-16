"""Stage 2 data model: chunks, triples, graph elements, provenance, operations, conflicts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

NodeLabel = Literal["Entity", "Document", "Chunk", "Community"]
NODE_LABELS: tuple[str, ...] = ("Entity", "Document", "Chunk", "Community")

# Structural (non-ontology) predicates written by the pipeline itself.
IN_DOCUMENT = "IN_DOCUMENT"  # Chunk → Document
NEXT_CHUNK = "NEXT_CHUNK"  # Chunk → Chunk
MENTIONED_IN = "MENTIONED_IN"  # Entity → Chunk
MEMBER_OF = "MEMBER_OF"  # Entity → Community
STRUCTURAL_PREDICATES: frozenset[str] = frozenset(
    {IN_DOCUMENT, NEXT_CHUNK, MENTIONED_IN, MEMBER_OF}
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_hash(*parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


_WS = re.compile(r"\s+")
_LEADING_DET = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Canonical comparison key for an entity surface form."""
    text = unicodedata.normalize("NFKC", name).strip().strip(" .,;:'\"")
    text = _LEADING_DET.sub("", text)
    return _WS.sub(" ", text).lower()


def clean_surface(name: str) -> str:
    """Display form: trimmed, single-spaced, leading article removed."""
    text = unicodedata.normalize("NFKC", name).strip().strip(" .,;:'\"")
    return _WS.sub(" ", _LEADING_DET.sub("", text))


def entity_node_id(name: str) -> str:
    return "ent-" + stable_hash(normalize_name(name))


def edge_id(source: str, predicate: str, target: str) -> str:
    return "rel-" + stable_hash(source, predicate, target)


# =========================================================================== provenance


class Provenance(BaseModel):
    """Where a node or relationship came from (Alg. 12)."""

    doc_id: str
    chunk_id: str
    page: int | None = None
    timestamp: str = Field(default_factory=utc_now)
    confidence: float = Field(1.0, ge=0, le=1)
    source_ref: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.chunk_id)

    def missing_fields(self) -> list[str]:
        missing = [
            f for f in ("doc_id", "chunk_id", "timestamp", "source_ref") if not getattr(self, f)
        ]
        if self.page is None:
            missing.append("page")
        return missing


def merge_provenance(existing: list[Provenance], incoming: Provenance) -> list[Provenance]:
    """Replace the record for the same (doc, chunk) source, else append."""
    out = [p for p in existing if p.key != incoming.key]
    out.append(incoming)
    return out


# =========================================================================== graph elements


class GraphNode(BaseModel):
    id: str
    label: NodeLabel
    name: str
    type: str = ""
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)
    vector_id: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def confidence(self) -> float:
        return max((p.confidence for p in self.provenance), default=1.0)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    predicate: str
    confidence: float = Field(1.0, ge=0, le=1)
    provenance: list[Provenance] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def structural(self) -> bool:
        return self.predicate in STRUCTURAL_PREDICATES

    @property
    def support(self) -> int:
        """Number of distinct (document, chunk) sources asserting this edge."""
        return len({p.key for p in self.provenance})

    def recompute_confidence(self) -> None:
        """Edge confidence = max over its sources' confidences."""
        if self.provenance:
            self.confidence = max(p.confidence for p in self.provenance)


# =========================================================================== chunks


class SemanticChunk(BaseModel):
    chunk_id: str
    doc_id: str
    order: int
    kind: Literal["text", "table"] = "text"
    text: str
    context: str = Field("", description="Overlap: tail of the previous chunk in the section.")
    section_id: str
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    level: int = 1
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0
    source_chunk_id: str | None = Field(None, description="Stage 1 chunk/table id reused.")
    previous_chunk: str | None = None
    next_chunk: str | None = None

    @property
    def source_ref(self) -> str:
        return f"{self.section_id}/{self.source_chunk_id or self.chunk_id}"


# =========================================================================== triples


class RawTriple(BaseModel):
    """One triple as the LLM returns it (Alg. 4)."""

    head: str = ""
    head_type: str = ""
    relation: str = ""
    tail: str = ""
    tail_type: str = ""
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return v


class TripleExtractionOutput(BaseModel):
    triples: list[RawTriple] = Field(default_factory=list)


class Triple(BaseModel):
    triple_id: str
    head: str
    head_type: str = "Concept"
    relation: str
    tail: str
    tail_type: str = "Concept"
    confidence: float = Field(ge=0, le=1)
    llm_confidence: float = Field(1.0, ge=0, le=1)
    provenance: Provenance
    mentions: dict[str, str] = Field(
        default_factory=dict, description="Original head/tail/relation surface forms."
    )
    notes: list[str] = Field(default_factory=list)

    def key(self) -> tuple[str, str, str]:
        return (normalize_name(self.head), self.relation.lower().strip(), normalize_name(self.tail))


class EntityRef(BaseModel):
    """A mention resolved against the KG (Alg. 7)."""

    mention: str
    name: str
    type: str
    node_id: str
    is_new: bool
    similarity: float = 0.0
    matched_name: str | None = None


class ResolvedTriple(BaseModel):
    triple: Triple
    head: EntityRef
    tail: EntityRef


class NormalizedTriple(BaseModel):
    resolved: ResolvedTriple
    predicate: str | None = None
    method: Literal["exact", "synonym", "stem", "llm", "none"] = "none"
    inverted: bool = False
    rejected_reason: str | None = None

    @property
    def triple(self) -> Triple:
        return self.resolved.triple

    @property
    def subject(self) -> EntityRef:
        return self.resolved.tail if self.inverted else self.resolved.head

    @property
    def object(self) -> EntityRef:
        return self.resolved.head if self.inverted else self.resolved.tail

    @property
    def edge_id(self) -> str:
        return edge_id(self.subject.node_id, self.predicate or "", self.object.node_id)


class ConflictStatus(StrEnum):
    NEW = "New"
    MERGE = "Merge"
    UPDATE = "Update"
    CONFLICT = "Conflict"
    REJECT = "Reject"


class StatusedTriple(BaseModel):
    triple: NormalizedTriple
    status: ConflictStatus
    reason: str = ""
    existing_edge_id: str | None = None
    conflicting_edge_ids: list[str] = Field(default_factory=list)


class Operation(StrEnum):
    CREATE = "Create"
    MERGE = "Merge"
    UPDATE = "Update"
    REJECT = "Reject"
    DELETE = "Delete"


class GraphOperation(BaseModel):
    op: Operation
    item: StatusedTriple | None = None
    reason: str = ""
    delete_edge_id: str | None = Field(None, description="Target of a Delete operation.")
    delete_source: tuple[str, str] | None = Field(
        None, description="(doc_id, chunk_id) to retract; None deletes the whole edge."
    )


class ConflictRecord(BaseModel):
    """Logged for every Conflict/Reject so no knowledge is silently discarded."""

    conflict_id: str
    status: ConflictStatus
    reason: str
    incoming: dict[str, Any]
    existing: list[dict[str, Any]] = Field(default_factory=list)
    detected_at: str = Field(default_factory=utc_now)


def triple_summary(item: NormalizedTriple) -> dict[str, Any]:
    t = item.triple
    return {
        "head": item.subject.name,
        "relation": item.predicate or t.relation,
        "raw_relation": t.relation,
        "tail": item.object.name,
        "confidence": t.confidence,
        "provenance": t.provenance.model_dump(),
    }


def edge_summary(edge: GraphEdge, names: dict[str, str]) -> dict[str, Any]:
    return {
        "edge_id": edge.id,
        "head": names.get(edge.source, edge.source),
        "relation": edge.predicate,
        "tail": names.get(edge.target, edge.target),
        "confidence": edge.confidence,
        "provenance": [p.model_dump() for p in edge.provenance],
    }
