"""A12 / Algorithm 13 - Embedding Agent.

Embeds the entities a chunk touched (by canonical name, the same text Entity Resolution
compares against, so A6 lookups stay consistent) and the chunk itself (section path +
text). Each vector is attached to its graph node via ``GraphStore.set_embedding``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from construction.context import ConstructionAgent
from construction.models import GraphNode
from construction.stores.vector_store import CHUNKS, COMMUNITIES, ENTITIES


class EmbeddingInput(BaseModel):
    node_ids: list[str]


class EmbeddingItem(BaseModel):
    node_id: str
    collection: str
    text: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class EmbeddingBatch(BaseModel):
    items: list[EmbeddingItem] = Field(default_factory=list)
    dim: int = 0


def embedding_text(node: GraphNode) -> str:
    if node.label == "Chunk":
        path = " > ".join(node.properties.get("section_path") or [])
        return f"{path}\n{node.properties.get('text', '')}".strip()
    if node.label == "Community":
        return f"{node.name}\n{node.properties.get('summary', '')}".strip()
    return node.name


def collection_for(node: GraphNode) -> str:
    return {"Chunk": CHUNKS, "Community": COMMUNITIES}.get(node.label, ENTITIES)


def vector_payload(node: GraphNode) -> dict[str, Any]:
    payload: dict[str, Any] = {"node_id": node.id, "label": node.label, "name": node.name}
    if node.label == "Entity":
        payload["type"] = node.type
        payload["doc_ids"] = sorted({p.doc_id for p in node.provenance})
    elif node.label == "Chunk":
        payload["doc_id"] = node.properties.get("doc_id")
        payload["section_id"] = node.properties.get("section_id")
        payload["page_start"] = node.properties.get("page_start")
    elif node.label == "Community":
        payload["size"] = node.properties.get("size")
    return payload


class EmbeddingAgent(ConstructionAgent[EmbeddingInput, EmbeddingBatch]):
    name: ClassVar[str] = "embedding"
    algorithm: ClassVar[int] = 13
    agent_id: ClassVar[str] = "A12"

    def _execute(self, data: EmbeddingInput) -> EmbeddingBatch:
        graph = self.ctx.graph
        nodes = [n for nid in dict.fromkeys(data.node_ids) if (n := graph.get_node(nid))]
        texts = [embedding_text(n) for n in nodes]
        vectors = self.ctx.embedder.embed(texts)
        out = EmbeddingBatch(dim=int(vectors.shape[1]) if len(nodes) else self.ctx.embedder.dim)
        for node, text, vec in zip(nodes, texts, vectors, strict=True):
            graph.set_embedding(node.id, vec)
            out.items.append(
                EmbeddingItem(
                    node_id=node.id,
                    collection=collection_for(node),
                    text=text,
                    vector=[float(x) for x in vec],
                    payload=vector_payload(node),
                )
            )
        return out
