"""A5 / Algorithm 6 - Graph Traversal Agent.

Locates graph nodes for the retrieved evidence (line 2). Seeds are:

* entities the subquery names (exact name/alias match), seed score 1.0;
* entity hits from A4 with score ≥ ``traversal.seed_min_score``;
* entities ``MENTIONED_IN`` a retrieved chunk, seed score = 0.8 × the chunk's score.

From the seeds it runs a breadth-first, multi-hop traversal up to ``params.hops`` (line 3)
over ontology relationships in both directions. Structural edges (``MENTIONED_IN``,
``MEMBER_OF`` …) are not reasoning steps. It collects the neighbouring entities and
relationships (line 4) and builds reasoning paths (line 5):

    path score = seed score × Π edge confidence × decay^(hops − 1)

Each relationship keeps the chunks that assert it (from its provenance), so a fact can be
cited back to a page. ``hops = 0`` (Simple Lookup) returns the seeds only.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from construction.models import MENTIONED_IN, GraphEdge
from retrieval.context import RetrievalAgent
from retrieval.models import (
    FactEvidence,
    GraphEvidence,
    ReasoningPath,
    RetrievalParams,
    VectorEvidence,
)
from retrieval.text import find_entity_mentions


class GraphTraversalInput(BaseModel):
    vector: VectorEvidence
    params: RetrievalParams


class GraphTraversalAgent(RetrievalAgent[GraphTraversalInput, GraphEvidence]):
    name: ClassVar[str] = "graph_traversal"
    algorithm: ClassVar[int] = 6
    agent_id: ClassVar[str] = "A5"

    def seeds(self, data: GraphTraversalInput) -> dict[str, float]:
        graph = self.ctx.graph
        cfg = self.settings.traversal
        seeds: dict[str, float] = {}

        def add(node_id: str, score: float) -> None:
            seeds[node_id] = max(seeds.get(node_id, 0.0), round(score, 4))

        for m in find_entity_mentions(data.vector.query_text, graph):
            add(m.node.id, 1.0)
        for e in data.vector.entities:
            if e.score >= cfg.seed_min_score:
                add(e.node_id, e.score)
        for c in data.vector.chunks:
            for edge in graph.find_edges(target=c.chunk_id, predicate=MENTIONED_IN):
                add(edge.source, 0.8 * max(c.score, 0.0))
        return seeds

    def _fact(self, edge: GraphEdge, names: dict[str, str], hop: int) -> FactEvidence:
        graph = self.ctx.graph
        chunk_ids, pages, docs = [], [], []
        for p in edge.provenance:
            node = graph.get_node(p.chunk_id)
            if node is not None and node.label == "Chunk":
                chunk_ids.append(p.chunk_id)
                if p.page is not None:
                    pages.append(p.page)
                docs.append(p.doc_id)
        return FactEvidence(
            edge_id=edge.id,
            source_id=edge.source,
            source=names.get(edge.source, edge.source),
            predicate=edge.predicate,
            target_id=edge.target,
            target=names.get(edge.target, edge.target),
            confidence=round(edge.confidence, 4),
            hop=hop,
            chunk_ids=list(dict.fromkeys(chunk_ids)),
            pages=sorted(set(pages)),
            doc_ids=list(dict.fromkeys(docs)),
        )

    def _execute(self, data: GraphTraversalInput) -> GraphEvidence:
        graph = self.ctx.graph
        cfg = self.settings.traversal
        seeds = self.seeds(data)
        out = GraphEvidence(
            subquery_id=data.vector.subquery_id, seeds=sorted(seeds), hops=data.params.hops
        )
        names: dict[str, str] = {}

        def name_of(node_id: str) -> str | None:
            if node_id not in names:
                node = graph.get_node(node_id)
                if node is None or node.label != "Entity":
                    return None
                names[node_id] = node.name
            return names[node_id]

        for sid in seeds:
            name_of(sid)
        visited: set[str] = {s for s in seeds if s in names}
        facts: dict[str, FactEvidence] = {}
        paths: dict[tuple[str, ...], ReasoningPath] = {}
        # frontier: node → (seed, edge ids, node ids, text, score)
        frontier: list[tuple[str, str, list[str], list[str], str, float]] = [
            (s, s, [], [s], names[s], seeds[s]) for s in sorted(visited)
        ]
        for depth in range(1, data.params.hops + 1):
            nxt: list[tuple[str, str, list[str], list[str], str, float]] = []
            for node_id, seed, eids, nids, text, score in frontier:
                edges = graph.find_edges(source=node_id) + graph.find_edges(target=node_id)
                for edge in sorted(edges, key=lambda e: e.id):
                    if edge.structural or edge.id in eids:
                        continue
                    other = edge.target if edge.source == node_id else edge.source
                    other_name = name_of(other)
                    if other_name is None or other in nids:
                        continue
                    decay = cfg.path_decay if depth > 1 else 1.0
                    new_score = round(score * edge.confidence * decay, 4)
                    arrow = (
                        f" -[{edge.predicate}]-> "
                        if edge.source == node_id
                        else f" <-[{edge.predicate}]- "
                    )
                    new_text = text + arrow + other_name
                    fact = facts.get(edge.id)
                    if fact is None:
                        fact = facts[edge.id] = self._fact(edge, names, depth)
                    fact.score = max(fact.score, new_score)
                    fact.hop = min(fact.hop, depth)
                    key = (*eids, edge.id)
                    if key not in paths or paths[key].score < new_score:
                        paths[key] = ReasoningPath(
                            seed_id=seed,
                            edge_ids=list(key),
                            node_ids=[*nids, other],
                            text=new_text,
                            score=new_score,
                        )
                    if other not in visited and len(visited) < cfg.max_nodes:
                        visited.add(other)
                        nxt.append(
                            (other, seed, [*eids, edge.id], [*nids, other], new_text, new_score)
                        )
            frontier = nxt
            if not frontier:
                break
        out.visited = sorted(visited)
        out.facts = sorted(facts.values(), key=lambda f: (-f.score, f.edge_id))
        out.paths = sorted(paths.values(), key=lambda p: (-p.score, p.text))[: cfg.max_paths]
        return out

    def confidence(self, output: GraphEvidence) -> float | None:
        return output.paths[0].score if output.paths else 0.0
