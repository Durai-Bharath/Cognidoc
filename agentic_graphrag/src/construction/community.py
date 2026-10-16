"""CommunityBuilder: a post-construction step not in *Construction.pdf* (DESIGN_DECISIONS #3).

Stage 3 retrieval needs community summary nodes. After the health audit this step:

1. runs Louvain (or Leiden, if ``igraph`` and ``leidenalg`` are installed) on the
   undirected entity graph, weighted by edge confidence, with a fixed seed;
2. gives each community of ≥ ``min_size`` members a stable id (a hash of its sorted
   members) and a *signature* (a hash of its members and internal edges);
3. leaves communities whose id and signature are unchanged alone: no LLM call, no
   re-embedding. That makes re-runs incremental;
4. summarises changed or new communities with the LLM (extractive fallback), upserts a
   ``Community`` node, links members ``MEMBER_OF`` and embeds and indexes the summary;
5. deletes Community nodes (and their vectors) that no longer exist.
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

import networkx as nx
from pydantic import BaseModel, Field

from common.llm import LLMError
from construction.agents.embedding import embedding_text, vector_payload
from construction.context import ConstructionAgent
from construction.models import (
    MEMBER_OF,
    GraphEdge,
    GraphNode,
    Provenance,
    edge_id,
    stable_hash,
    utc_now,
)
from construction.stores.vector_store import COMMUNITIES, VectorRecord

SYSTEM = (
    "You summarise one community of a knowledge graph for retrieval. Write a short title "
    "and a factual summary (2-5 sentences) of what the entities have in common and how they "
    "relate, using only the facts given."
)


class CommunitySummary(BaseModel):
    title: str = ""
    summary: str = ""
    key_entities: list[str] = Field(default_factory=list)


class CommunityInput(BaseModel):
    document_id: str | None = None


class CommunityInfo(BaseModel):
    community_id: str
    title: str
    size: int
    status: str


class CommunityReport(BaseModel):
    algorithm: str = "louvain"
    total: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    modularity: float | None = None
    llm_summaries: int = 0
    communities: list[CommunityInfo] = Field(default_factory=list)


class CommunityBuilder(ConstructionAgent[CommunityInput, CommunityReport]):
    name: ClassVar[str] = "community_builder"
    algorithm: ClassVar[int] = 0  # not part of the PDF
    agent_id: ClassVar[str] = "CB"

    # ------------------------------------------------------------------ detection
    def _partition(self, g: nx.Graph[str]) -> tuple[list[set[str]], str]:
        cfg = self.settings.community
        if cfg.algorithm == "leiden":
            try:
                import igraph as ig
                import leidenalg

                ig_graph = ig.Graph.from_networkx(g)
                part = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=cfg.resolution,
                    seed=cfg.seed,
                )
                names = ig_graph.vs["_nx_name"]
                return [{names[i] for i in comm} for comm in part], "leiden"
            except ImportError:
                self.log.warning("leiden_unavailable_falling_back_to_louvain")
        comms = nx.community.louvain_communities(
            g, weight="weight", resolution=cfg.resolution, seed=cfg.seed
        )
        return [set(c) for c in comms], "louvain"

    # ------------------------------------------------------------------ summaries
    def _facts(self, members: set[str], edges: list[GraphEdge], names: dict[str, str]) -> list[str]:
        internal = [e for e in edges if e.source in members and e.target in members]
        internal.sort(key=lambda e: (-e.confidence, e.id))
        return [
            f"{names[e.source]} {e.predicate.replace('_', ' ')} {names[e.target]}."
            for e in internal[: self.settings.community.max_edges_in_prompt]
        ]

    def _summarise(
        self, ranked: list[str], facts: list[str], names: dict[str, str]
    ) -> tuple[CommunitySummary, bool]:
        cfg = self.settings.community
        member_names = [names[m] for m in ranked[: cfg.max_members_in_prompt]]
        if cfg.llm_summaries:
            prompt = "ENTITIES: " + "; ".join(member_names) + "\nFACTS:\n" + "\n".join(facts)
            try:
                s = self.llm_json(
                    task="community_summary", system=SYSTEM, prompt=prompt, schema=CommunitySummary
                )
                if s.title.strip() and s.summary.strip():
                    return s, True
            except LLMError as exc:
                self.log.warning("community_summary_failed", extra={"error": str(exc)[:200]})
        title = ", ".join(member_names[:3]) + (
            " and related entities" if len(member_names) > 3 else ""
        )
        return CommunitySummary(
            title=title, summary=" ".join(facts[:5]), key_entities=member_names[:5]
        ), False

    # ------------------------------------------------------------------ main
    def _execute(self, data: CommunityInput) -> CommunityReport:
        cfg = self.settings.community
        graph, vectors = self.ctx.graph, self.ctx.vectors
        nodes = graph.nodes()
        names = {n.id: n.name for n in nodes}
        entity_ids = {n.id for n in nodes if n.label == "Entity"}
        edges = [e for e in graph.edges() if not e.structural]

        g: nx.Graph[str] = nx.Graph()
        g.add_nodes_from(entity_ids)
        for e in edges:
            if e.source in entity_ids and e.target in entity_ids:
                w = g.get_edge_data(e.source, e.target, {}).get("weight", 0.0)
                g.add_edge(e.source, e.target, weight=w + e.confidence)

        report = CommunityReport()
        partition: list[set[str]] = []
        if g.number_of_edges():
            partition, report.algorithm = self._partition(g)
            report.modularity = round(
                float(nx.community.modularity(g, partition, weight="weight")), 4
            )
        keep = [c for c in partition if len(c) >= cfg.min_size]

        existing = {n.id: n for n in nodes if n.label == "Community"}
        wanted: set[str] = set()
        degree = dict(g.degree(weight="weight"))
        for members in keep:
            cid = "com-" + stable_hash(*sorted(members))
            wanted.add(cid)
            internal = sorted(
                f"{e.id}:{e.predicate}"
                for e in edges
                if e.source in members and e.target in members
            )
            signature = stable_hash(*sorted(names[m] for m in members), *internal)
            old = existing.get(cid)
            if old is not None and old.properties.get("signature") == signature and old.vector_id:
                report.unchanged += 1
                report.communities.append(
                    CommunityInfo(
                        community_id=cid, title=old.name, size=len(members), status="unchanged"
                    )
                )
                continue
            ranked = sorted(members, key=lambda m: (-degree.get(m, 0.0), names[m]))
            facts = self._facts(members, edges, names)
            summary, used_llm = self._summarise(ranked, facts, names)
            report.llm_summaries += used_llm
            member_docs = Counter(
                p.doc_id for m in members if (n := graph.get_node(m)) for p in n.provenance
            )
            internal_conf = [
                e.confidence for e in edges if e.source in members and e.target in members
            ]
            conf = round(sum(internal_conf) / len(internal_conf), 4) if internal_conf else 1.0
            prov = Provenance(
                doc_id=data.document_id
                or (member_docs.most_common(1)[0][0] if member_docs else "derived"),
                chunk_id=f"community:{cid}",
                page=0,
                confidence=conf,
                source_ref="CommunityBuilder",
            )
            now = utc_now()
            node = GraphNode(
                id=cid,
                label="Community",
                name=summary.title.strip(),
                type="community",
                properties={
                    "summary": summary.summary.strip(),
                    "key_entities": summary.key_entities,
                    "members": sorted(members),
                    "size": len(members),
                    "signature": signature,
                    "level": 0,
                    "algorithm": report.algorithm,
                    "doc_ids": sorted(member_docs),
                },
                provenance=[prov],
                created_at=old.created_at if old else now,
                updated_at=now,
            )
            graph.upsert_node(node)
            # Replace membership edges.
            if old is not None:
                for e in graph.find_edges(target=cid, predicate=MEMBER_OF):
                    graph.delete_edge(e.id)
            for m in sorted(members):
                graph.upsert_edge(
                    GraphEdge(
                        id=edge_id(m, MEMBER_OF, cid),
                        source=m,
                        target=cid,
                        predicate=MEMBER_OF,
                        confidence=conf,
                        provenance=[prov],
                    )
                )
            vec = self.ctx.embedder.embed_one(embedding_text(node))
            graph.set_embedding(cid, vec)
            vectors.ensure_collection(COMMUNITIES, int(vec.shape[0]))
            vectors.upsert(
                COMMUNITIES,
                [
                    VectorRecord(
                        id=cid, vector=[float(x) for x in vec], payload=vector_payload(node)
                    )
                ],
            )
            node.vector_id = cid
            graph.upsert_node(node)
            status = "updated" if old is not None else "created"
            report.updated += old is not None
            report.created += old is None
            report.communities.append(
                CommunityInfo(community_id=cid, title=node.name, size=len(members), status=status)
            )

        stale = sorted(set(existing) - wanted)
        for cid in stale:
            graph.delete_node(cid)
        if stale:
            vectors.delete(COMMUNITIES, stale)
        report.removed = len(stale)
        report.total = len(keep)
        return report

    def confidence(self, output: CommunityReport) -> float | None:
        return output.modularity
