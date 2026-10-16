"""A6 / Algorithm 7 - Community Retrieval Agent.

Identifies the communities of the retrieved entities through their ``MEMBER_OF`` edges
(line 2). With ``community.semantic_search`` it also searches the ``communities``
collection with the subquery embedding, so a Global query with no entity in it still
reaches the summaries. It then fetches the Community summary nodes (line 3) and their
members (line 4), and ranks them (line 5):

    relevance = (1 − m) · max(0, cos(query, summary)) + m · member overlap
    score     = relevance × weight(query type)

``m`` is ``community.membership_weight``. The weight comes from ``params.community_weight``:
1.0 for Global, 0.6 Hybrid, 0.2 Local, 0 for Simple Lookup, which skips the agent. A14 can
raise it and ``community_top_n`` when it widens community retrieval.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from common.embedder import cosine
from construction.models import MEMBER_OF
from construction.stores.vector_store import COMMUNITIES
from retrieval.context import RetrievalAgent
from retrieval.models import (
    CommunityEvidence,
    CommunityHit,
    GraphEvidence,
    RetrievalParams,
    VectorEvidence,
)


class CommunityRetrievalInput(BaseModel):
    vector: VectorEvidence
    graph: GraphEvidence
    params: RetrievalParams


class CommunityRetrievalAgent(RetrievalAgent[CommunityRetrievalInput, CommunityEvidence]):
    name: ClassVar[str] = "community_retrieval"
    algorithm: ClassVar[int] = 7
    agent_id: ClassVar[str] = "A6"

    def _execute(self, data: CommunityRetrievalInput) -> CommunityEvidence:
        cfg = self.settings.community
        weight = data.params.community_weight
        out = CommunityEvidence(subquery_id=data.vector.subquery_id, weight=weight)
        if weight <= 0:
            return out
        graph = self.ctx.graph
        qvec = self.ctx.embedder.embed_one(data.vector.query_text)
        entities = set(data.graph.visited) | set(data.graph.seeds)
        entities |= {e.node_id for e in data.vector.entities}

        # line 2: communities of the retrieved entities (+ semantic matches)
        candidates: dict[str, float | None] = {}
        for eid in sorted(entities):
            for edge in graph.find_edges(source=eid, predicate=MEMBER_OF):
                candidates.setdefault(edge.target, None)
        if cfg.semantic_search:
            for hit in self.ctx.vectors.search(
                COMMUNITIES, qvec, top_k=max(2 * data.params.community_top_n, 4)
            ):
                cid = str(hit.payload.get("node_id", hit.id))
                candidates[cid] = max(hit.score, candidates.get(cid) or -1.0)

        # lines 3-5
        hits: list[CommunityHit] = []
        for cid, sim in candidates.items():
            node = graph.get_node(cid)
            if node is None or node.label != "Community":
                continue
            if sim is None:
                emb = graph.get_embedding(cid)
                sim = cosine(qvec, emb) if emb is not None else 0.0
            member_ids = [str(m) for m in node.properties.get("members") or []]
            overlap = len(entities & set(member_ids)) / len(member_ids) if member_ids else 0.0
            relevance = (1 - cfg.membership_weight) * max(
                0.0, sim
            ) + cfg.membership_weight * overlap
            names = [
                n.name for m in member_ids[: cfg.max_members_listed] if (n := graph.get_node(m))
            ]
            hits.append(
                CommunityHit(
                    community_id=cid,
                    title=node.name,
                    summary=str(node.properties.get("summary", "")),
                    member_ids=member_ids,
                    members=names,
                    doc_ids=[str(d) for d in node.properties.get("doc_ids") or []],
                    relevance=round(relevance, 4),
                    overlap=round(overlap, 4),
                    weight=weight,
                    score=round(relevance * weight, 4),
                )
            )
        hits.sort(key=lambda h: (-h.score, h.community_id))
        out.items = hits[: data.params.community_top_n]
        return out

    def confidence(self, output: CommunityEvidence) -> float | None:
        return output.items[0].score if output.items else 0.0
