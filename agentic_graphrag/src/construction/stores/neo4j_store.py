"""Neo4j ``GraphStore`` (production). Requires the optional ``neo4j`` driver.

Every node carries the shared label ``:KGNode`` (unique ``id`` constraint) plus its own
label (Entity/Document/Chunk/Community). Relationship types are the predicates.
Nested data (properties, provenance) is stored as JSON strings because Neo4j
properties must be primitives or arrays of primitives. Similarity lookup uses one native
vector index per label, created on the first ``set_embedding``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from common.embedder import Vector
from common.logging import get_logger
from construction.models import NODE_LABELS, GraphEdge, GraphNode, Provenance
from construction.stores.graph_store import Direction, GraphStore, GraphStoreError

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_log = get_logger(__name__)


def _ident(value: str, kind: str) -> str:
    """Labels and relationship types cannot be query parameters: whitelist them."""
    if not _IDENT.match(value):
        raise GraphStoreError(f"invalid {kind} {value!r}")
    return value


def _label(label: str) -> str:
    if label not in NODE_LABELS:
        raise GraphStoreError(f"unknown node label {label!r}")
    return label


class Neo4jGraphStore(GraphStore):
    backend = "neo4j"

    def __init__(self, uri: str, user: str, password: str, *, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._vector_indexes: dict[str, int] = {}
        self._run("CREATE CONSTRAINT kgnode_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self._driver.session(database=self._db) as session:
            return list(session.run(query, **params))

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------ mapping
    @staticmethod
    def _node_props(node: GraphNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "label": node.label,
            "name": node.name,
            "type": node.type,
            "aliases": node.aliases,
            "props_json": json.dumps(node.properties, default=str),
            "provenance_json": json.dumps([p.model_dump() for p in node.provenance]),
            "vector_id": node.vector_id,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
        }

    @staticmethod
    def _to_node(props: Any) -> GraphNode:
        p = dict(props)
        return GraphNode(
            id=p["id"],
            label=p["label"],
            name=p.get("name", ""),
            type=p.get("type", ""),
            aliases=list(p.get("aliases") or []),
            properties=json.loads(p.get("props_json") or "{}"),
            provenance=[
                Provenance.model_validate(x) for x in json.loads(p.get("provenance_json") or "[]")
            ],
            vector_id=p.get("vector_id"),
            created_at=p.get("created_at", ""),
            updated_at=p.get("updated_at", ""),
        )

    @staticmethod
    def _to_edge(source: str, target: str, props: Any) -> GraphEdge:
        p = dict(props)
        return GraphEdge(
            id=p["id"],
            source=source,
            target=target,
            predicate=p["predicate"],
            confidence=float(p.get("confidence", 1.0)),
            provenance=[
                Provenance.model_validate(x) for x in json.loads(p.get("provenance_json") or "[]")
            ],
            properties=json.loads(p.get("props_json") or "{}"),
            created_at=p.get("created_at", ""),
            updated_at=p.get("updated_at", ""),
        )

    # ------------------------------------------------------------------ nodes
    def upsert_node(self, node: GraphNode) -> None:
        label = _label(node.label)
        self._run(
            f"MERGE (n:KGNode {{id: $id}}) SET n:{label} SET n += $props",
            id=node.id,
            props=self._node_props(node),
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        rows = self._run("MATCH (n:KGNode {id: $id}) RETURN properties(n) AS p", id=node_id)
        return self._to_node(rows[0]["p"]) if rows else None

    def delete_node(self, node_id: str) -> None:
        self._run("MATCH (n:KGNode {id: $id}) DETACH DELETE n", id=node_id)

    def nodes(self, label: str | None = None) -> list[GraphNode]:
        match = f"MATCH (n:{_label(label)})" if label else "MATCH (n:KGNode)"
        return [self._to_node(r["p"]) for r in self._run(f"{match} RETURN properties(n) AS p")]

    # ------------------------------------------------------------------ edges
    def upsert_edge(self, edge: GraphEdge) -> None:
        rel = _ident(edge.predicate, "predicate")
        self.delete_edge(edge.id)
        props = {
            "id": edge.id,
            "predicate": edge.predicate,
            "confidence": edge.confidence,
            "provenance_json": json.dumps([p.model_dump() for p in edge.provenance]),
            "props_json": json.dumps(edge.properties, default=str),
            "created_at": edge.created_at,
            "updated_at": edge.updated_at,
        }
        rows = self._run(
            f"MATCH (s:KGNode {{id: $s}}), (t:KGNode {{id: $t}}) "
            f"CREATE (s)-[r:{rel}]->(t) SET r = $props RETURN r.id AS id",
            s=edge.source,
            t=edge.target,
            props=props,
        )
        if not rows:
            raise GraphStoreError(f"edge {edge.id}: endpoint missing")

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        rows = self._run(
            "MATCH (s:KGNode)-[r {id: $id}]->(t:KGNode) "
            "RETURN s.id AS s, t.id AS t, properties(r) AS p",
            id=edge_id,
        )
        return self._to_edge(rows[0]["s"], rows[0]["t"], rows[0]["p"]) if rows else None

    def delete_edge(self, edge_id: str) -> None:
        self._run("MATCH ()-[r {id: $id}]->() DELETE r", id=edge_id)

    def find_edges(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        predicate: str | None = None,
    ) -> list[GraphEdge]:
        rel = f":{_ident(predicate, 'predicate')}" if predicate else ""
        rows = self._run(
            f"MATCH (s:KGNode)-[r{rel}]->(t:KGNode) "
            "WHERE ($s IS NULL OR s.id = $s) AND ($t IS NULL OR t.id = $t) AND r.id IS NOT NULL "
            "RETURN s.id AS s, t.id AS t, properties(r) AS p",
            s=source,
            t=target,
        )
        return [self._to_edge(r["s"], r["t"], r["p"]) for r in rows]

    def edges(self, predicate: str | None = None) -> list[GraphEdge]:
        return self.find_edges(predicate=predicate)

    def neighbors(
        self, node_id: str, *, direction: Direction = "both", predicate: str | None = None
    ) -> list[GraphNode]:
        rel = f":{_ident(predicate, 'predicate')}" if predicate else ""
        pattern = {"out": f"-[{rel}]->", "in": f"<-[{rel}]-", "both": f"-[{rel}]-"}[direction]
        rows = self._run(
            f"MATCH (n:KGNode {{id: $id}}){pattern}(m:KGNode) RETURN DISTINCT properties(m) AS p",
            id=node_id,
        )
        return [self._to_node(r["p"]) for r in rows]

    def degree(self, node_id: str) -> int:
        rows = self._run("MATCH (n:KGNode {id: $id})-[r]-() RETURN count(r) AS c", id=node_id)
        return int(rows[0]["c"]) if rows else 0

    # ------------------------------------------------------------------ embeddings
    def _ensure_vector_index(self, label: str, dim: int) -> str:
        """Create the label's vector index, or recreate it when the dimension changed.

        A vector index keeps the dimension it was created with. After an embedder change,
        old vectors of the wrong size are dropped so the index never mixes models.
        """
        name = f"{label.lower()}_embedding"
        if self._vector_indexes.get(name) == dim:
            return name
        rows = self._run(
            "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options", name=name
        )
        if rows:
            existing = int(rows[0]["options"]["indexConfig"]["vector.dimensions"])
            if existing != dim:
                _log.warning(
                    "neo4j_vector_index_dimension_changed",
                    extra={"index": name, "old_dim": existing, "new_dim": dim},
                )
                self._run(f"DROP INDEX {name} IF EXISTS")
                self._run(
                    f"MATCH (n:{_label(label)}) WHERE n.embedding IS NOT NULL "
                    "AND size(n.embedding) <> $dim REMOVE n.embedding",
                    dim=dim,
                )
        self._run(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{_label(label)}) ON (n.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'}}",
            dim=dim,
        )
        self._run("CALL db.awaitIndexes(300)")
        self._vector_indexes[name] = dim
        return name

    def set_embedding(self, node_id: str, vector: Vector) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise GraphStoreError(f"cannot embed missing node {node_id}")
        self._ensure_vector_index(node.label, len(vector))
        self._run(
            "MATCH (n:KGNode {id: $id}) SET n.embedding = $vec",
            id=node_id,
            vec=[float(x) for x in vector],
        )

    def get_embedding(self, node_id: str) -> Vector | None:
        rows = self._run("MATCH (n:KGNode {id: $id}) RETURN n.embedding AS e", id=node_id)
        if not rows or rows[0]["e"] is None:
            return None
        return np.asarray(rows[0]["e"], dtype=np.float32)

    def similar_nodes(
        self, vector: Vector, *, label: str = "Entity", top_k: int = 5
    ) -> list[tuple[GraphNode, float]]:
        name = self._ensure_vector_index(label, len(vector))
        rows = self._run(
            "CALL db.index.vector.queryNodes($name, $k, $vec) YIELD node, score "
            "RETURN properties(node) AS p, score",
            name=name,
            k=top_k,
            vec=[float(x) for x in vector],
        )
        # Neo4j reports cosine as (1 + cos) / 2; convert back to cosine.
        return [(self._to_node(r["p"]), 2.0 * float(r["score"]) - 1.0) for r in rows]
