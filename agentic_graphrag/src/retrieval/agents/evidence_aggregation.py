"""A7 / Algorithm 8 - Evidence Aggregation Agent.

Merges vector evidence V, graph evidence E and community evidence CE (lines 1-2) into one
list of :class:`EvidenceItem` of four kinds:

* ``chunk``: a retrieved chunk. Relevance is its vector score; graph confidence is
  ``aggregation.default_confidence``.
* ``fact``: a relationship. Relevance = ½·cos(query, fact sentence) + ½·best path score.
  Confidence = edge confidence × the lower ``retrieval_confidence`` of its endpoints
  (A13 feedback; 1.0 when never adjusted).
* ``path``: a multi-hop reasoning path (≥ 2 edges), scored like a fact, with confidence =
  the weakest edge on it.
* ``community``: a summary. Relevance from A6; confidence = the Community node's
  confidence, multiplied by the query-type community weight.

Duplicates (same chunk, edge, edge sequence or community) are merged, keeping the highest
relevance and the union of sources (line 3). Items are ranked by score = relevance ×
confidence (line 4) and capped at ``aggregation.max_items`` (line 5).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from common.embedder import cosine
from construction.models import MENTIONED_IN
from retrieval.context import RetrievalAgent
from retrieval.models import (
    AggregatedEvidence,
    CommunityEvidence,
    EvidenceItem,
    GraphEvidence,
    VectorEvidence,
)


class AggregationInput(BaseModel):
    vector: VectorEvidence
    graph: GraphEvidence
    communities: CommunityEvidence


def merge_items(items: list[EvidenceItem]) -> tuple[list[EvidenceItem], int]:
    """De-duplicate by evidence id (max relevance/score, union of sources). Returns removed."""
    merged: dict[str, EvidenceItem] = {}
    removed = 0
    for item in items:
        old = merged.get(item.evidence_id)
        if old is None:
            merged[item.evidence_id] = item.model_copy(deep=True)
            continue
        removed += 1
        if item.score > old.score:
            old.relevance, old.score, old.confidence = item.relevance, item.score, item.confidence
        old.sources = list(dict.fromkeys([*old.sources, *item.sources]))
        old.subquery_ids = list(dict.fromkeys([*old.subquery_ids, *item.subquery_ids]))
        old.chunk_ids = list(dict.fromkeys([*old.chunk_ids, *item.chunk_ids]))
    ranked = sorted(merged.values(), key=lambda i: (-i.score, i.evidence_id))
    return ranked, removed


class EvidenceAggregationAgent(RetrievalAgent[AggregationInput, AggregatedEvidence]):
    name: ClassVar[str] = "evidence_aggregation"
    algorithm: ClassVar[int] = 8
    agent_id: ClassVar[str] = "A7"

    def _retrieval_confidence(self, node_id: str, cache: dict[str, float]) -> float:
        if node_id not in cache:
            node = self.ctx.graph.get_node(node_id)
            value = node.properties.get("retrieval_confidence") if node else None
            cache[node_id] = float(value) if isinstance(value, int | float) else 1.0
        return cache[node_id]

    def _member_chunks(self, member_ids: list[str], limit: int = 3) -> list[str]:
        out: list[str] = []
        for mid in member_ids:
            for edge in self.ctx.graph.find_edges(source=mid, predicate=MENTIONED_IN):
                if edge.target not in out:
                    out.append(edge.target)
                if len(out) >= limit:
                    return out
        return out

    def _execute(self, data: AggregationInput) -> AggregatedEvidence:
        sq = data.vector.subquery_id
        default_conf = self.settings.aggregation.default_confidence
        items: list[EvidenceItem] = []

        for c in data.vector.chunks:
            rel = max(0.0, c.score)
            items.append(
                EvidenceItem(
                    evidence_id=f"chunk:{c.chunk_id}",
                    kind="chunk",
                    text=c.text,
                    relevance=round(rel, 4),
                    confidence=default_conf,
                    score=round(rel * default_conf, 4),
                    chunk_ids=[c.chunk_id],
                    node_ids=[c.chunk_id],
                    page=c.page,
                    doc_id=c.doc_id,
                    sources=["vector"],
                    subquery_ids=[sq],
                )
            )

        facts = {f.edge_id: f for f in data.graph.facts}
        paths = [p for p in data.graph.paths if len(p.edge_ids) >= 2]
        texts = [f.sentence for f in facts.values()] + [p.text for p in paths]
        if texts:
            qvec = self.ctx.embedder.embed_one(data.vector.query_text)
            vecs = self.ctx.embedder.embed(texts)
            sims = [max(0.0, cosine(qvec, v)) for v in vecs]
        else:
            sims = []
        rc: dict[str, float] = {}
        for f, sim in zip(facts.values(), sims[: len(facts)], strict=True):
            rel = 0.5 * sim + 0.5 * f.score
            conf = f.confidence * min(
                self._retrieval_confidence(f.source_id, rc),
                self._retrieval_confidence(f.target_id, rc),
            )
            items.append(
                EvidenceItem(
                    evidence_id=f"fact:{f.edge_id}",
                    kind="fact",
                    text=f.sentence,
                    relevance=round(rel, 4),
                    confidence=round(conf, 4),
                    score=round(rel * conf, 4),
                    chunk_ids=f.chunk_ids,
                    node_ids=[f.source_id, f.target_id],
                    edge_ids=[f.edge_id],
                    page=f.pages[0] if f.pages else None,
                    doc_id=f.doc_ids[0] if f.doc_ids else "",
                    sources=["graph"],
                    subquery_ids=[sq],
                )
            )
        for p, sim in zip(paths, sims[len(facts) :], strict=True):
            path_facts = [facts[e] for e in p.edge_ids if e in facts]
            conf = min((f.confidence for f in path_facts), default=1.0)
            rel = 0.5 * sim + 0.5 * p.score
            chunk_ids = list(dict.fromkeys(c for f in path_facts for c in f.chunk_ids))
            pages = [pg for f in path_facts for pg in f.pages]
            docs = [d for f in path_facts for d in f.doc_ids]
            items.append(
                EvidenceItem(
                    evidence_id="path:" + ">".join(p.edge_ids),
                    kind="path",
                    text=p.text,
                    relevance=round(rel, 4),
                    confidence=round(conf, 4),
                    score=round(rel * conf, 4),
                    chunk_ids=chunk_ids,
                    node_ids=p.node_ids,
                    edge_ids=p.edge_ids,
                    page=pages[0] if pages else None,
                    doc_id=docs[0] if docs else "",
                    sources=["graph"],
                    subquery_ids=[sq],
                )
            )
        for h in data.communities.items:
            node = self.ctx.graph.get_node(h.community_id)
            conf = node.confidence if node else default_conf
            members = ", ".join(h.members)
            items.append(
                EvidenceItem(
                    evidence_id=f"community:{h.community_id}",
                    kind="community",
                    text=f"{h.title}: {h.summary}" + (f" (members: {members})" if members else ""),
                    relevance=h.relevance,
                    confidence=round(conf, 4),
                    score=round(h.relevance * conf * h.weight, 4),
                    chunk_ids=self._member_chunks(h.member_ids),
                    node_ids=[h.community_id],
                    doc_id=h.doc_ids[0] if h.doc_ids else "",
                    sources=["community"],
                    subquery_ids=[sq],
                )
            )
        ranked, removed = merge_items(items)
        return AggregatedEvidence(
            subquery_id=sq,
            items=ranked[: self.settings.aggregation.max_items],
            duplicates_removed=removed,
        )

    def confidence(self, output: AggregatedEvidence) -> float | None:
        return output.items[0].score if output.items else 0.0
