"""A11 / Algorithm 12 - Provenance Agent (runs once per accepted chunk).

Graph CRUD already wrote each triple's provenance record, because Merge and Update are
defined in terms of it. This agent then:

* registers the ``Chunk`` node (source text, section hierarchy, pages) and links it
  ``IN_DOCUMENT`` to the Document and ``NEXT_CHUNK`` from the previous stored chunk;
* links every entity this chunk touched to it with ``MENTIONED_IN``;
* checks that every touched node and edge carries a complete record for this source:
  document id, chunk id, page, timestamp, confidence and source reference. Missing
  page/source/timestamp fields are filled in from the chunk, and anything still
  incomplete fails validation and is flagged.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.validation import CheckResult, ValidationOutcome
from construction.context import ConstructionAgent
from construction.models import (
    IN_DOCUMENT,
    MENTIONED_IN,
    NEXT_CHUNK,
    GraphEdge,
    GraphNode,
    Provenance,
    SemanticChunk,
    edge_id,
    utc_now,
)


class ProvenanceInput(BaseModel):
    chunk: SemanticChunk
    touched_nodes: list[str] = Field(default_factory=list)
    touched_edges: list[str] = Field(default_factory=list)


class ProvenanceReport(BaseModel):
    chunk_node_id: str
    entity_ids: list[str] = Field(default_factory=list)
    structural_edges: list[str] = Field(default_factory=list)
    checked_nodes: int = 0
    checked_edges: int = 0
    repaired: list[str] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


class ProvenanceAgent(ConstructionAgent[ProvenanceInput, ProvenanceReport]):
    name: ClassVar[str] = "provenance"
    algorithm: ClassVar[int] = 12
    agent_id: ClassVar[str] = "A11"

    def _chunk_provenance(self, chunk: SemanticChunk, confidence: float = 1.0) -> Provenance:
        return Provenance(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            page=chunk.page_start,
            timestamp=utc_now(),
            confidence=confidence,
            source_ref=chunk.source_ref,
        )

    def _complete(
        self, records: list[Provenance], chunk: SemanticChunk
    ) -> tuple[list[Provenance], bool]:
        changed = False
        out: list[Provenance] = []
        for p in records:
            if p.key == (chunk.doc_id, chunk.chunk_id):
                fill: dict[str, object] = {}
                if p.page is None:
                    fill["page"] = chunk.page_start
                if not p.source_ref:
                    fill["source_ref"] = chunk.source_ref
                if not p.timestamp:
                    fill["timestamp"] = utc_now()
                if fill:
                    p = p.model_copy(update=fill)
                    changed = True
            out.append(p)
        return out, changed

    def _execute(self, data: ProvenanceInput) -> ProvenanceReport:
        graph = self.ctx.graph
        chunk = data.chunk
        prov = self._chunk_provenance(chunk)
        existing = graph.get_node(chunk.chunk_id)
        graph.upsert_node(
            GraphNode(
                id=chunk.chunk_id,
                label="Chunk",
                name=f"{chunk.section_title} #{chunk.order}",
                type=chunk.kind,
                properties={
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    "section_id": chunk.section_id,
                    "section_path": chunk.section_path,
                    "level": chunk.level,
                    "order": chunk.order,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "token_count": chunk.token_count,
                    "source_chunk_id": chunk.source_chunk_id,
                },
                provenance=[prov],
                vector_id=existing.vector_id if existing else None,
                created_at=existing.created_at if existing else prov.timestamp,
            )
        )
        report = ProvenanceReport(chunk_node_id=chunk.chunk_id)

        def link(source: str, predicate: str, target: str, confidence: float = 1.0) -> None:
            eid = edge_id(source, predicate, target)
            old = graph.get_edge(eid)
            graph.upsert_edge(
                GraphEdge(
                    id=eid,
                    source=source,
                    target=target,
                    predicate=predicate,
                    confidence=confidence,
                    provenance=[self._chunk_provenance(chunk, confidence)],
                    created_at=old.created_at if old else utc_now(),
                )
            )
            report.structural_edges.append(eid)

        link(chunk.chunk_id, IN_DOCUMENT, chunk.doc_id)
        if chunk.previous_chunk and graph.has_node(chunk.previous_chunk):
            link(chunk.previous_chunk, NEXT_CHUNK, chunk.chunk_id)

        failures: list[str] = []
        for nid in data.touched_nodes:
            node = graph.get_node(nid)
            if node is None:
                failures.append(f"touched node {nid} missing")
                continue
            node.provenance, changed = self._complete(node.provenance, chunk)
            if changed:
                graph.upsert_node(node)
                report.repaired.append(nid)
            report.checked_nodes += 1
            failures += self._incomplete(nid, node.provenance, chunk)
            if node.label == "Entity":
                report.entity_ids.append(nid)
                link(nid, MENTIONED_IN, chunk.chunk_id, node.confidence)
        for eid in data.touched_edges:
            edge = graph.get_edge(eid)
            if edge is None:
                failures.append(f"touched edge {eid} missing")
                continue
            edge.provenance, changed = self._complete(edge.provenance, chunk)
            if changed:
                graph.upsert_edge(edge)
                report.repaired.append(eid)
            report.checked_edges += 1
            failures += self._incomplete(eid, edge.provenance, chunk)
        report.validation.add(CheckResult.from_failures("provenance_complete", failures))
        self.flag_if_failed(report.validation, "incomplete provenance", item_ref=chunk.chunk_id)
        return report

    @staticmethod
    def _incomplete(ref: str, records: list[Provenance], chunk: SemanticChunk) -> list[str]:
        mine = [p for p in records if p.key == (chunk.doc_id, chunk.chunk_id)]
        if not mine:
            return [f"{ref}: no provenance record for {chunk.chunk_id}"]
        return [f"{ref}: missing {f}" for f in mine[0].missing_fields()]
