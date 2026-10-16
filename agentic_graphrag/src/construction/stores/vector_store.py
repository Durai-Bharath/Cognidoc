"""``VectorStore`` interface plus the in-memory backend used by tests and the CLI.

Collections hold :class:`VectorRecord` rows. ``payload["node_id"]`` is the entity↔vector
mapping Vector Index (A13) maintains and Construction Validation (A14) checks.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from common.embedder import Vector

ENTITIES = "entities"
CHUNKS = "chunks"
COMMUNITIES = "communities"
COLLECTIONS: tuple[str, ...] = (ENTITIES, CHUNKS, COMMUNITIES)


class VectorStoreError(RuntimeError):
    pass


class VectorRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    vector: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    backend: str = "abstract"

    @abstractmethod
    def ensure_collection(self, name: str, dim: int) -> None: ...

    @abstractmethod
    def upsert(self, collection: str, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def get(self, collection: str, ids: list[str]) -> list[VectorRecord]: ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> None: ...

    @abstractmethod
    def search(
        self,
        collection: str,
        vector: Vector,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    def ids(self, collection: str) -> list[str]: ...

    def count(self, collection: str) -> int:
        return len(self.ids(collection))

    def export(self, path: str | Path) -> str:
        """Write a VDB′ manifest (collections and counts) and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"backend": self.backend, "collections": {c: self.count(c) for c in COLLECTIONS}}
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(out)

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release connections (no-op for in-memory)."""


class InMemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict[str, VectorRecord]] = {}
        self._dims: dict[str, int] = {}

    def ensure_collection(self, name: str, dim: int) -> None:
        self._data.setdefault(name, {})
        known = self._dims.setdefault(name, dim)
        if known != dim:
            raise VectorStoreError(f"collection {name} has dim {known}, got {dim}")

    def _coll(self, name: str) -> dict[str, VectorRecord]:
        if name not in self._data:
            raise VectorStoreError(f"unknown collection {name}")
        return self._data[name]

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        coll = self._coll(collection)
        dim = self._dims[collection]
        for r in records:
            if len(r.vector) != dim:
                raise VectorStoreError(f"{r.id}: vector dim {len(r.vector)} != {dim}")
            coll[r.id] = r.model_copy(deep=True)

    def get(self, collection: str, ids: list[str]) -> list[VectorRecord]:
        coll = self._data.get(collection, {})
        return [coll[i].model_copy(deep=True) for i in ids if i in coll]

    def delete(self, collection: str, ids: list[str]) -> None:
        coll = self._data.get(collection, {})
        for i in ids:
            coll.pop(i, None)

    def search(
        self,
        collection: str,
        vector: Vector,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        rows = [
            r
            for r in self._data.get(collection, {}).values()
            if not where or all(r.payload.get(k) == v for k, v in where.items())
        ]
        if not rows:
            return []
        q = np.asarray(vector, dtype=np.float32)
        mat = np.asarray([r.vector for r in rows], dtype=np.float32)
        denom = np.linalg.norm(mat, axis=1) * (float(np.linalg.norm(q)) or 1.0)
        denom[denom == 0] = 1.0
        scores = (mat @ q) / denom
        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(id=rows[i].id, score=float(scores[i]), payload=dict(rows[i].payload))
            for i in order
        ]

    def ids(self, collection: str) -> list[str]:
        return list(self._data.get(collection, {}))

    # persistence
    def save(self, path: str | Path) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dims": self._dims,
            "collections": {
                c: [r.model_dump() for r in rows.values()] for c, rows in self._data.items()
            },
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        store = cls()
        p = Path(path)
        if not p.exists():
            return store
        payload = json.loads(p.read_text(encoding="utf-8"))
        for c, dim in payload.get("dims", {}).items():
            store.ensure_collection(c, int(dim))
        for c, rows in payload.get("collections", {}).items():
            store.upsert(c, [VectorRecord.model_validate(r) for r in rows])
        return store
