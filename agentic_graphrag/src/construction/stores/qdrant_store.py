"""Qdrant ``VectorStore`` (production). Requires the optional ``qdrant-client`` package.

Qdrant point ids must be integers or UUIDs, so string ids map to ``uuid5`` and the
original id is kept in the payload under ``_id``.
"""

from __future__ import annotations

import uuid
from typing import Any

from common.embedder import Vector
from construction.stores.vector_store import SearchHit, VectorRecord, VectorStore

_NS = uuid.UUID("6f1c2a52-3d4e-4b8f-9a10-5c7d2e8f0b31")


def point_id(record_id: str) -> str:
    return str(uuid.uuid5(_NS, record_id))


class QdrantVectorStore(VectorStore):
    backend = "qdrant"

    def __init__(self, url: str, *, prefix: str = "graphrag_", api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self.client = QdrantClient(url=url, api_key=api_key)
        self.prefix = prefix

    def _name(self, collection: str) -> str:
        return f"{self.prefix}{collection}"

    def ensure_collection(self, name: str, dim: int) -> None:
        m = self._models
        if not self.client.collection_exists(self._name(name)):
            self.client.create_collection(
                self._name(name),
                vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE),
            )

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        points = [
            self._models.PointStruct(
                id=point_id(r.id), vector=r.vector, payload={**r.payload, "_id": r.id}
            )
            for r in records
        ]
        self.client.upsert(self._name(collection), points=points, wait=True)

    def get(self, collection: str, ids: list[str]) -> list[VectorRecord]:
        if not ids or not self.client.collection_exists(self._name(collection)):
            return []
        points = self.client.retrieve(
            self._name(collection), ids=[point_id(i) for i in ids], with_vectors=True
        )
        out: list[VectorRecord] = []
        for p in points:
            payload = dict(p.payload or {})
            rid = str(payload.pop("_id", p.id))
            vec: Any = p.vector
            out.append(VectorRecord(id=rid, vector=[float(x) for x in vec], payload=payload))
        return out

    def delete(self, collection: str, ids: list[str]) -> None:
        if ids and self.client.collection_exists(self._name(collection)):
            self.client.delete(
                self._name(collection),
                points_selector=self._models.PointIdsList(points=[point_id(i) for i in ids]),
                wait=True,
            )

    def search(
        self,
        collection: str,
        vector: Vector,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        m = self._models
        if not self.client.collection_exists(self._name(collection)):
            return []
        flt = (
            m.Filter(
                must=[
                    m.FieldCondition(key=k, match=m.MatchValue(value=v)) for k, v in where.items()
                ]
            )
            if where
            else None
        )
        res = self.client.query_points(
            self._name(collection),
            query=[float(x) for x in vector],
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for p in res.points:
            payload = dict(p.payload or {})
            hits.append(SearchHit(id=str(payload.pop("_id", p.id)), score=p.score, payload=payload))
        return hits

    def ids(self, collection: str) -> list[str]:
        if not self.client.collection_exists(self._name(collection)):
            return []
        out: list[str] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self._name(collection), limit=256, offset=offset, with_payload=["_id"]
            )
            out.extend(str((p.payload or {}).get("_id", p.id)) for p in points)
            if offset is None:
                return out
