"""A15 / Algorithm 16 - Cache Manager Agent, plus the :class:`SemanticCache` it manages.

The cache is semantic: entries are found by query-embedding similarity, not exact text.
One agent class serves four operations:

* ``lookup`` (pipeline start): embed Q, take the most similar entry *with the same
  conversation-history hash*, and hit if cosine ≥ ``cache.similarity_threshold``. Before
  a hit is served, every cited graph element is re-fingerprinted. An element that changed
  or disappeared (e.g. Stage 2 re-ran) invalidates the entry, and the lookup misses.
* ``store`` (Alg. 1 line 21): only **verified** answers are written (lines 1-6). An entry
  holds the key (a hash of the query text and history), the query embedding, the
  reasoning paths and evidence ids, the answer with citations, and fingerprints of every
  cited chunk, node and edge.
* ``invalidate``: drop every entry citing one of the given element ids (called after A13
  penalises or flags elements).
* ``purge_stale``: re-fingerprint every entry and drop the stale ones (called after Stage 2
  construction, e.g. by ``run-all``).

Fingerprints cover what Stage 2 changes (names, text, summaries, provenance and its
confidences), not what feedback reinforcement changes, so reinforcing an element never
invalidates an entry. Penalties invalidate explicitly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from common.embedder import cosine
from common.logging import get_logger
from construction.models import normalize_name, stable_hash, utc_now
from construction.stores.graph_store import GraphStore
from retrieval.context import RetrievalAgent
from retrieval.models import CacheEntry, GroundedAnswer, element_fingerprint

_log = get_logger(__name__)


class SemanticCache:
    """Thread-safe entry store with optional JSON persistence."""

    def __init__(self, path: str | Path | None = None, *, max_entries: int = 500) -> None:
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, CacheEntry] = {}
        if self.path is not None and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for item in raw.get("entries", []):
                    entry = CacheEntry.model_validate(item)
                    self._entries[entry.key] = entry
            except (OSError, ValueError):
                _log.exception("cache_load_failed", extra={"path": str(self.path)})

    @property
    def entries(self) -> list[CacheEntry]:
        with self._lock:
            return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            return self._entries.get(key)

    def best_match(self, vector: np.ndarray, history: str) -> tuple[CacheEntry | None, float]:
        best: CacheEntry | None = None
        best_sim = -1.0
        for entry in self.entries:
            if entry.history_hash != history:
                continue
            sim = cosine(vector, np.asarray(entry.embedding, dtype=np.float32))
            if sim > best_sim:
                best, best_sim = entry, sim
        return best, round(best_sim, 4)

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._entries[entry.key] = entry
            while len(self._entries) > self.max_entries:
                oldest = min(self._entries.values(), key=lambda e: e.created_at)
                del self._entries[oldest.key]
        self.save()

    def touch(self, key: str) -> None:
        with self._lock:
            if key in self._entries:
                self._entries[key].hits += 1
        self.save()

    def remove(self, keys: list[str]) -> None:
        if not keys:
            return
        with self._lock:
            for k in keys:
                self._entries.pop(k, None)
        self.save()

    def citing(self, element_ids: set[str]) -> list[str]:
        return [e.key for e in self.entries if element_ids & set(e.fingerprints)]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            payload = {"entries": [e.model_dump() for e in self._entries.values()]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            _log.exception("cache_save_failed", extra={"path": str(self.path)})


def fingerprints(graph: GraphStore, ids: set[str]) -> dict[str, str]:
    """element id → fingerprint ("" when the element no longer exists)."""
    out: dict[str, str] = {}
    for i in sorted(ids):
        element = graph.get_node(i) or graph.get_edge(i)
        out[i] = element_fingerprint(element) if element is not None else ""
    return out


def stale_ids(graph: GraphStore, entry: CacheEntry) -> list[str]:
    current = fingerprints(graph, set(entry.fingerprints))
    return [i for i, fp in entry.fingerprints.items() if not current[i] or current[i] != fp]


def cache_key(query: str, history: str) -> str:
    return "qc-" + stable_hash(normalize_name(query), history)


class CacheRequest(BaseModel):
    op: Literal["lookup", "store", "invalidate", "purge_stale"]
    query: str = ""
    history_hash: str = ""
    verified: bool = False
    grounded: GroundedAnswer | None = None
    score: float = 0.0
    report_id: str = ""
    reasoning_paths: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)


class CacheResult(BaseModel):
    op: str
    hit: bool = False
    entry: CacheEntry | None = None
    similarity: float | None = None
    stored: bool = False
    key: str | None = None
    invalidated: list[str] = Field(default_factory=list)
    reason: str = ""


class CacheManagerAgent(RetrievalAgent[CacheRequest, CacheResult]):
    name: ClassVar[str] = "cache_manager"
    algorithm: ClassVar[int] = 16
    agent_id: ClassVar[str] = "A15"

    @property
    def cache(self) -> SemanticCache:
        return self.ctx.cache

    def _execute(self, data: CacheRequest) -> CacheResult:
        if data.op == "invalidate":
            keys = self.cache.citing(set(data.element_ids))
            self.cache.remove(keys)
            return CacheResult(
                op=data.op, invalidated=keys, reason=f"{len(keys)} entries cite changed elements"
            )
        if data.op == "purge_stale":
            keys = [e.key for e in self.cache.entries if stale_ids(self.ctx.graph, e)]
            self.cache.remove(keys)
            return CacheResult(op=data.op, invalidated=keys, reason=f"{len(keys)} stale entries")
        if not self.settings.cache.enabled:
            return CacheResult(op=data.op, reason="cache disabled")
        if data.op == "lookup":
            return self._lookup(data)
        return self._store(data)

    def _lookup(self, data: CacheRequest) -> CacheResult:
        vec = self.ctx.embedder.embed_one(data.query)
        entry, sim = self.cache.best_match(vec, data.history_hash)
        threshold = self.settings.cache.similarity_threshold
        if entry is None or sim < threshold:
            return CacheResult(op="lookup", similarity=sim if entry else None, reason="miss")
        stale = stale_ids(self.ctx.graph, entry)
        if stale:
            self.cache.remove([entry.key])
            return CacheResult(
                op="lookup",
                similarity=sim,
                invalidated=[entry.key],
                reason=f"stale: {len(stale)} cited elements changed ({', '.join(stale[:3])})",
            )
        self.cache.touch(entry.key)
        return CacheResult(
            op="lookup",
            hit=True,
            entry=self.cache.get(entry.key),
            similarity=sim,
            key=entry.key,
            reason="hit",
        )

    def _store(self, data: CacheRequest) -> CacheResult:
        if not data.verified or data.grounded is None:
            return CacheResult(op="store", reason="only verified answers are cached")
        ids = data.grounded.referenced_ids()
        all_ids = ids["chunk"] | ids["node"] | ids["edge"]
        key = cache_key(data.query, data.history_hash)
        entry = CacheEntry(
            key=key,
            query=data.query,
            history_hash=data.history_hash,
            embedding=[float(x) for x in self.ctx.embedder.embed_one(data.query)],
            answer=data.grounded.answer,
            statements=data.grounded.statements,
            reasoning_paths=data.reasoning_paths,
            evidence_ids=data.evidence_ids,
            fingerprints=fingerprints(self.ctx.graph, all_ids),
            report_id=data.report_id,
            score=data.score,
            created_at=utc_now(),
        )
        self.cache.put(entry)
        return CacheResult(op="store", stored=True, key=key, reason="stored verified answer")
