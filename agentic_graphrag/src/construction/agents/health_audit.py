"""A15 / Algorithm 16 - Graph Health Audit Agent.

Detects:

* orphan nodes (degree 0);
* duplicate entities: the same mention key, or embedding cosine ≥ ``audit.duplicate_similarity``;
* stale relationships: provenance that points at a Document or Chunk no longer in the
  KG, or no source re-observed within ``audit.stale_after_days``;
* disconnected entity components (count, sizes, largest-component ratio);
* low-confidence edges (< ``audit.low_confidence``).

From these it derives graph quality metrics and an overall ``health_score``, the mean of
(1 − orphan ratio, 1 − duplicate ratio, 1 − stale ratio, 1 − low-confidence ratio,
largest-component ratio, ontology-compliance ratio).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from construction.agents.construction_validation import (
    ConstructionValidationReport,
    entity_components,
)
from construction.agents.coreference import mention_key
from construction.context import ConstructionAgent
from construction.models import GraphEdge
from construction.stores.vector_store import COLLECTIONS


class DuplicatePair(BaseModel):
    a: str
    b: str
    a_name: str
    b_name: str
    similarity: float
    reason: str


class AuditReport(BaseModel):
    orphans: list[str] = Field(default_factory=list)
    duplicates: list[DuplicatePair] = Field(default_factory=list)
    stale_edges: list[str] = Field(default_factory=list)
    low_confidence_edges: list[str] = Field(default_factory=list)
    components: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    health_score: float = 1.0


def _parse(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _ratio(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


class GraphHealthAuditAgent(ConstructionAgent[ConstructionValidationReport, AuditReport]):
    name: ClassVar[str] = "graph_health_audit"
    algorithm: ClassVar[int] = 16
    agent_id: ClassVar[str] = "A15"

    def _execute(self, data: ConstructionValidationReport) -> AuditReport:
        cfg = self.settings.audit
        graph = self.ctx.graph
        nodes = graph.nodes()
        edges = graph.edges()
        by_id = {n.id: n for n in nodes}
        entities = [n for n in nodes if n.label == "Entity"]
        semantic = [e for e in edges if not e.structural]
        out = AuditReport()

        # Orphans
        degree: Counter[str] = Counter()
        for e in edges:
            degree[e.source] += 1
            degree[e.target] += 1
        out.orphans = sorted(n.id for n in nodes if degree[n.id] == 0)

        # Duplicates
        seen_pairs: set[tuple[str, str]] = set()
        by_key: dict[str, list[str]] = {}
        for n in entities:
            by_key.setdefault(mention_key(n.name), []).append(n.id)
        for ids in by_key.values():
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    seen_pairs.add((a, b))
                    out.duplicates.append(
                        DuplicatePair(
                            a=a,
                            b=b,
                            a_name=by_id[a].name,
                            b_name=by_id[b].name,
                            similarity=1.0,
                            reason="same mention key",
                        )
                    )
        embedded = [(n.id, v) for n in entities if (v := graph.get_embedding(n.id)) is not None]
        if len(embedded) > 1:
            mat = np.stack([v for _, v in embedded]).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            sims = mat @ mat.T
            ii, jj = np.where(np.triu(sims, k=1) >= cfg.duplicate_similarity)
            for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
                a, b = embedded[i][0], embedded[j][0]
                if (a, b) in seen_pairs or (b, a) in seen_pairs:
                    continue
                out.duplicates.append(
                    DuplicatePair(
                        a=a,
                        b=b,
                        a_name=by_id[a].name,
                        b_name=by_id[b].name,
                        similarity=round(float(sims[i, j]), 4),
                        reason="embedding similarity",
                    )
                )

        # Stale relationships
        horizon = datetime.now(UTC) - timedelta(days=cfg.stale_after_days)
        docs = {n.id for n in nodes if n.label == "Document"}
        chunks = {n.id for n in nodes if n.label == "Chunk"}

        def stale(e: GraphEdge) -> bool:
            live = [p for p in e.provenance if p.doc_id in docs and p.chunk_id in chunks]
            if not live:
                return True
            seen = [d for p in live if (d := _parse(p.timestamp)) is not None]
            return bool(seen) and max(seen) < horizon

        out.stale_edges = sorted(e.id for e in semantic if stale(e))
        out.low_confidence_edges = sorted(
            e.id for e in semantic if e.confidence < cfg.low_confidence
        )

        # Components
        comps = sorted(entity_components(nodes, edges), key=len, reverse=True)
        sizes = [len(c) for c in comps]
        largest = _ratio(sizes[0], len(entities)) if sizes else 1.0
        out.components = {
            "count": len(comps),
            "sizes": sizes[:20],
            "largest_component_ratio": largest,
            "isolated_entities": sum(1 for s in sizes if s == 1),
        }

        # Metrics
        labels = Counter(n.label for n in nodes)
        predicates = Counter(e.predicate for e in edges)
        n_ent, n_sem = len(entities), len(semantic)
        records = [n.provenance for n in nodes] + [e.provenance for e in edges]
        complete = sum(1 for r in records if any(not p.missing_fields() for p in r))
        embeddable = [n for n in nodes if n.label in ("Entity", "Chunk", "Community")]
        orphan_r = _ratio(len(out.orphans), len(nodes))
        dup_r = _ratio(len({d.b for d in out.duplicates}), n_ent)
        stale_r = _ratio(len(out.stale_edges), n_sem)
        low_r = _ratio(len(out.low_confidence_edges), n_sem)
        compliance = float(data.stats.get("ontology_compliance_ratio", 1.0))
        out.metrics = {
            "nodes": len(nodes),
            "edges": len(edges),
            "nodes_by_label": dict(sorted(labels.items())),
            "edges_by_predicate": dict(sorted(predicates.items())),
            "entities": n_ent,
            "semantic_edges": n_sem,
            "documents": labels.get("Document", 0),
            "chunks": labels.get("Chunk", 0),
            "avg_edge_confidence": round(float(np.mean([e.confidence for e in semantic])), 4)
            if semantic
            else 0.0,
            "avg_edge_support": round(float(np.mean([e.support for e in semantic])), 4)
            if semantic
            else 0.0,
            "avg_entity_degree": round(2 * n_sem / n_ent, 4) if n_ent else 0.0,
            "entity_graph_density": round(n_sem / (n_ent * (n_ent - 1)), 6) if n_ent > 1 else 0.0,
            "provenance_coverage": _ratio(complete, len(nodes) + len(edges)),
            "vector_coverage": _ratio(sum(1 for n in embeddable if n.vector_id), len(embeddable)),
            "vectors": {c: self.ctx.vectors.count(c) for c in COLLECTIONS},
            "orphan_ratio": orphan_r,
            "duplicate_ratio": dup_r,
            "stale_ratio": stale_r,
            "low_confidence_ratio": low_r,
            "ontology_compliance_ratio": compliance,
            "validation_passed": data.passed,
        }
        parts = [1 - orphan_r, 1 - dup_r, 1 - stale_r, 1 - low_r, largest, compliance]
        out.health_score = round(sum(parts) / len(parts), 4)
        out.metrics["health_score"] = out.health_score

        issues = {
            "orphans": len(out.orphans),
            "duplicates": len(out.duplicates),
            "stale_edges": len(out.stale_edges),
        }
        if any(issues.values()):
            self.flag("graph health issues", details={**issues, "health_score": out.health_score})
        return out

    def confidence(self, output: AuditReport) -> float | None:
        return output.health_score
