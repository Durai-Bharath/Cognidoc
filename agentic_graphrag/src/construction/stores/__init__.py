"""Graph and vector stores behind interfaces, each with an in-memory test backend."""

from __future__ import annotations

import os
from pathlib import Path

from construction.config import ConstructionConfig
from construction.stores.graph_store import GraphStore, GraphStoreError, InMemoryGraphStore
from construction.stores.vector_store import (
    CHUNKS,
    COLLECTIONS,
    COMMUNITIES,
    ENTITIES,
    InMemoryVectorStore,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)

GRAPH_STATE_FILE = "graph.json"
VECTOR_STATE_FILE = "vectors.json"

__all__ = [
    "CHUNKS",
    "COLLECTIONS",
    "COMMUNITIES",
    "ENTITIES",
    "GraphStore",
    "GraphStoreError",
    "InMemoryGraphStore",
    "InMemoryVectorStore",
    "SearchHit",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "build_graph_store",
    "build_vector_store",
    "save_state",
]


def build_graph_store(config: ConstructionConfig, *, load_state: bool = True) -> GraphStore:
    s = config.stores
    if s.graph_backend == "neo4j":
        from construction.stores.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            os.environ.get("NEO4J_URI", s.neo4j_uri),
            os.environ.get("NEO4J_USER", s.neo4j_user),
            os.environ.get("NEO4J_PASSWORD", ""),
            database=s.neo4j_database,
        )
    if load_state:
        return InMemoryGraphStore.load(Path(config.paths.state_dir) / GRAPH_STATE_FILE)
    return InMemoryGraphStore()


def build_vector_store(config: ConstructionConfig, *, load_state: bool = True) -> VectorStore:
    s = config.stores
    if s.vector_backend == "qdrant":
        from construction.stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            os.environ.get("QDRANT_URL", s.qdrant_url),
            prefix=s.collection_prefix,
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
    if load_state:
        return InMemoryVectorStore.load(Path(config.paths.state_dir) / VECTOR_STATE_FILE)
    return InMemoryVectorStore()


def save_state(graph: GraphStore, vectors: VectorStore, state_dir: str | Path) -> dict[str, str]:
    """Persist in-memory backends so the next run is incremental (servers persist themselves)."""
    out: dict[str, str] = {}
    if isinstance(graph, InMemoryGraphStore):
        out["graph_state"] = graph.save(Path(state_dir) / GRAPH_STATE_FILE)
    if isinstance(vectors, InMemoryVectorStore):
        out["vector_state"] = vectors.save(Path(state_dir) / VECTOR_STATE_FILE)
    return out
