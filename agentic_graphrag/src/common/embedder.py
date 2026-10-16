"""Text embedding behind one interface. All vectors are L2-normalised float32 rows."""

from __future__ import annotations

import hashlib
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from common.config import EmbedderConfig
from common.logging import get_logger
from common.text import STOPWORDS

_log = get_logger(__name__)

Vector = npt.NDArray[np.float32]


def _normalise(mat: npt.NDArray[Any]) -> Vector:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vector = (mat / norms).astype(np.float32)
    return out


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two (already normalised) vectors; 0.0 if either is zero."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TransientEmbeddingError(RuntimeError):
    """Retryable embedding failure (rate limit, timeout, 5xx)."""


class Embedder(ABC):
    """Base class: caching plus retries with exponential backoff around ``_embed_batch``."""

    def __init__(
        self,
        *,
        max_retries: int = 0,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache: dict[str, Vector] = {}
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._sleep = sleep

    def _embed_with_retry(self, texts: Sequence[str]) -> Vector:
        for attempt in range(self.max_retries + 1):
            try:
                return self._embed_batch(texts)
            except TransientEmbeddingError as exc:
                if attempt == self.max_retries:
                    raise
                delay = min(self.backoff_base_s * (2**attempt), self.backoff_max_s)
                _log.warning(
                    "embedding_retry",
                    extra={"attempt": attempt + 1, "delay_s": delay, "error": str(exc)[:200]},
                )
                self._sleep(delay)
        raise AssertionError("unreachable")

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str]) -> Vector: ...

    def embed(self, texts: Sequence[str]) -> Vector:
        """Embed ``texts`` → array of shape ``(len(texts), dim)`` (cached per text)."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = _normalise(self._embed_with_retry(missing))
            for text, vec in zip(missing, vectors, strict=True):
                self._cache[text] = vec
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        stacked: Vector = np.stack([self._cache[t] for t in texts])
        return stacked

    def embed_one(self, text: str) -> Vector:
        row: Vector = self.embed([text])[0]
        return row

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return cosine(va, vb)


class FakeEmbedder(Embedder):
    """Deterministic, offline bag-of-words hashing embedder for tests.

    Texts sharing content words get high cosine similarity, so threshold logic behaves
    like a real embedder without any model download.
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_batch(self, texts: Sequence[str]) -> Vector:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
            for word in words:
                digest = hashlib.sha1(word.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, idx] += sign
        return out


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, batch_size: int = 32) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)
        self._batch = batch_size
        self._dim = int(self._model.get_sentence_embedding_dimension() or 384)

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_batch(self, texts: Sequence[str]) -> Vector:
        vecs = self._model.encode(list(texts), batch_size=self._batch, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class OpenAICompatibleEmbedder(Embedder):
    """OpenRouter / OpenAI ``/embeddings`` endpoint."""

    def __init__(
        self,
        provider: str,
        model: str,
        batch_size: int = 32,
        *,
        max_retries: int = 6,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
    ) -> None:
        super().__init__(
            max_retries=max_retries, backoff_base_s=backoff_base_s, backoff_max_s=backoff_max_s
        )
        import openai

        self._openai = openai

        if provider == "openrouter":
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            )
        else:
            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._batch = batch_size
        self._dim = 0

    @property
    def dim(self) -> int:
        if not self._dim:
            self._dim = int(self.embed_one("dimension probe").shape[0])
        return self._dim

    def _embed_batch(self, texts: Sequence[str]) -> Vector:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = [t if t.strip() else " " for t in texts[start : start + self._batch]]
            try:
                resp = self._client.embeddings.create(
                    model=self._model, input=batch, encoding_format="float"
                )
            except (
                self._openai.RateLimitError,
                self._openai.APIConnectionError,
                self._openai.InternalServerError,
            ) as exc:
                raise TransientEmbeddingError(str(exc)) from exc
            rows.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        arr = np.asarray(rows, dtype=np.float32)
        self._dim = self._dim or int(arr.shape[1])
        return arr


def build_embedder(config: EmbedderConfig) -> Embedder:
    if config.provider == "fake":
        return FakeEmbedder(config.fake_dim)
    if config.provider == "sentence_transformers":
        return SentenceTransformerEmbedder(config.model, config.batch_size)
    return OpenAICompatibleEmbedder(
        config.provider,
        config.model,
        config.batch_size,
        max_retries=config.max_retries,
        backoff_base_s=config.backoff_base_s,
        backoff_max_s=config.backoff_max_s,
    )
