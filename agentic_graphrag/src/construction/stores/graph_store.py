"""``GraphStore`` interface plus the networkx in-memory backend used by tests and the CLI.

Nodes and edges are :class:`GraphNode` / :class:`GraphEdge` models keyed by stable ids,
so every write is an idempotent upsert. Embeddings live next to the node (not in the
model) and back ``similar_nodes``, the lookup Entity Resolution (A6) needs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np

from common.embedder import Vector
from construction.models import GraphEdge, GraphNode

Direction = Literal["out", "in", "both"]


class GraphStoreError(RuntimeError):
    pass


class GraphStore(ABC):
    backend: str = "abstract"

    # ------------------------------------------------------------------ nodes
    @abstractmethod
    def upsert_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None: ...

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and every edge touching it (no-op when absent)."""

    @abstractmethod
    def nodes(self, label: str | None = None) -> list[GraphNode]: ...

    # ------------------------------------------------------------------ edges
    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None:
        """Insert or replace an edge; both endpoints must exist."""

    @abstractmethod
    def get_edge(self, edge_id: str) -> GraphEdge | None: ...

    @abstractmethod
    def delete_edge(self, edge_id: str) -> None: ...

    @abstractmethod
    def find_edges(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        predicate: str | None = None,
    ) -> list[GraphEdge]: ...

    @abstractmethod
    def edges(self, predicate: str | None = None) -> list[GraphEdge]: ...

    @abstractmethod
    def neighbors(
        self, node_id: str, *, direction: Direction = "both", predicate: str | None = None
    ) -> list[GraphNode]: ...

    # ------------------------------------------------------------------ embeddings
    @abstractmethod
    def set_embedding(self, node_id: str, vector: Vector) -> None: ...

    @abstractmethod
    def get_embedding(self, node_id: str) -> Vector | None: ...

    @abstractmethod
    def similar_nodes(
        self, vector: Vector, *, label: str = "Entity", top_k: int = 5
    ) -> list[tuple[GraphNode, float]]:
        """Nodes of ``label`` ranked by cosine similarity to ``vector`` (highest first)."""

    # ------------------------------------------------------------------ misc
    def has_node(self, node_id: str) -> bool:
        return self.get_node(node_id) is not None

    def degree(self, node_id: str) -> int:
        return len(self.find_edges(source=node_id)) + len(self.find_edges(target=node_id))

    def export(self, path: str | Path) -> str:
        """Write KG′ as JSON (nodes + edges) and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "nodes": [n.model_dump() for n in self.nodes()],
            "edges": [e.model_dump() for e in self.edges()],
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(out)

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections (no-op for in-memory)."""


def _cosine_rank(
    query: Vector, items: list[tuple[str, Vector]], top_k: int
) -> list[tuple[str, float]]:
    if not items:
        return []
    q = np.asarray(query, dtype=np.float32)
    qn = float(np.linalg.norm(q)) or 1.0
    mat = np.stack([v for _, v in items]).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    scores = (mat @ q) / (norms * qn)
    order = np.argsort(-scores)[:top_k]
    return [(items[i][0], float(scores[i])) for i in order]


class InMemoryGraphStore(GraphStore):
    """networkx ``MultiDiGraph`` keyed by edge id; optional JSON persistence."""

    backend = "memory"

    def __init__(self) -> None:
        self.g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self._edges: dict[str, tuple[str, str]] = {}
        self._vectors: dict[str, Vector] = {}

    # nodes
    def upsert_node(self, node: GraphNode) -> None:
        self.g.add_node(node.id, data=node.model_copy(deep=True))

    def get_node(self, node_id: str) -> GraphNode | None:
        if node_id not in self.g:
            return None
        node: GraphNode = self.g.nodes[node_id]["data"]
        return node.model_copy(deep=True)

    def delete_node(self, node_id: str) -> None:
        if node_id not in self.g:
            return
        for eid in [e for e, (s, t) in self._edges.items() if node_id in (s, t)]:
            del self._edges[eid]
        self.g.remove_node(node_id)
        self._vectors.pop(node_id, None)

    def nodes(self, label: str | None = None) -> list[GraphNode]:
        out = [d["data"].model_copy(deep=True) for _, d in self.g.nodes(data=True)]
        return [n for n in out if label is None or n.label == label]

    # edges
    def upsert_edge(self, edge: GraphEdge) -> None:
        for end in (edge.source, edge.target):
            if end not in self.g:
                raise GraphStoreError(f"edge {edge.id}: endpoint {end} does not exist")
        if edge.id in self._edges:
            s, t = self._edges[edge.id]
            self.g.remove_edge(s, t, key=edge.id)
        self.g.add_edge(edge.source, edge.target, key=edge.id, data=edge.model_copy(deep=True))
        self._edges[edge.id] = (edge.source, edge.target)

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        ends = self._edges.get(edge_id)
        if ends is None:
            return None
        edge: GraphEdge = self.g.edges[ends[0], ends[1], edge_id]["data"]
        return edge.model_copy(deep=True)

    def delete_edge(self, edge_id: str) -> None:
        ends = self._edges.pop(edge_id, None)
        if ends is not None:
            self.g.remove_edge(ends[0], ends[1], key=edge_id)

    def find_edges(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        predicate: str | None = None,
    ) -> list[GraphEdge]:
        if source is not None and source not in self.g:
            return []
        if target is not None and target not in self.g:
            return []
        if source is not None:
            raw = self.g.out_edges(source, keys=True, data=True)
        elif target is not None:
            raw = self.g.in_edges(target, keys=True, data=True)
        else:
            raw = self.g.edges(keys=True, data=True)
        out: list[GraphEdge] = []
        for s, t, _k, d in raw:
            e: GraphEdge = d["data"]
            ends_ok = (target is None or t == target) and (source is None or s == source)
            if ends_ok and (predicate is None or e.predicate == predicate):
                out.append(e.model_copy(deep=True))
        return out

    def edges(self, predicate: str | None = None) -> list[GraphEdge]:
        return self.find_edges(predicate=predicate)

    def neighbors(
        self, node_id: str, *, direction: Direction = "both", predicate: str | None = None
    ) -> list[GraphNode]:
        ids: list[str] = []
        if direction in ("out", "both"):
            ids += [e.target for e in self.find_edges(source=node_id, predicate=predicate)]
        if direction in ("in", "both"):
            ids += [e.source for e in self.find_edges(target=node_id, predicate=predicate)]
        return [n for i in dict.fromkeys(ids) if (n := self.get_node(i)) is not None]

    def degree(self, node_id: str) -> int:
        return int(self.g.degree(node_id)) if node_id in self.g else 0

    # embeddings
    def set_embedding(self, node_id: str, vector: Vector) -> None:
        if node_id not in self.g:
            raise GraphStoreError(f"cannot embed missing node {node_id}")
        self._vectors[node_id] = np.asarray(vector, dtype=np.float32)

    def get_embedding(self, node_id: str) -> Vector | None:
        return self._vectors.get(node_id)

    def similar_nodes(
        self, vector: Vector, *, label: str = "Entity", top_k: int = 5
    ) -> list[tuple[GraphNode, float]]:
        items = [
            (nid, v)
            for nid, v in self._vectors.items()
            if nid in self.g and self.g.nodes[nid]["data"].label == label
        ]
        ranked = _cosine_rank(vector, items, top_k)
        return [(n, s) for nid, s in ranked if (n := self.get_node(nid)) is not None]

    # persistence
    def save(self, path: str | Path) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "nodes": [n.model_dump() for n in self.nodes()],
            "edges": [e.model_dump() for e in self.edges()],
            "embeddings": {k: v.tolist() for k, v in self._vectors.items()},
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> InMemoryGraphStore:
        store = cls()
        p = Path(path)
        if not p.exists():
            return store
        payload = json.loads(p.read_text(encoding="utf-8"))
        for n in payload.get("nodes", []):
            store.upsert_node(GraphNode.model_validate(n))
        for e in payload.get("edges", []):
            store.upsert_edge(GraphEdge.model_validate(e))
        for nid, vec in payload.get("embeddings", {}).items():
            if nid in store.g:
                store._vectors[nid] = np.asarray(vec, dtype=np.float32)
        return store
