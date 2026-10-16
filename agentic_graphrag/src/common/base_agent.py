"""``BaseAgent[InT, OutT]``: timing, structured logging, typed errors and trace records."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from common.logging import get_logger
from common.review_queue import ReviewQueue
from common.validation import ValidationOutcome

if TYPE_CHECKING:
    from common.config import ExtractionConfig
    from common.embedder import Embedder
    from common.llm import LLMClient

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class TraceRecord(BaseModel):
    agent: str
    algorithm: int
    started_at: datetime
    latency_ms: float
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    context: dict[str, str] = Field(default_factory=dict)


class AgentError(BaseModel):
    agent: str
    error_type: str
    message: str
    recoverable: bool = False


class AgentFailure(RuntimeError):
    """Raised by :meth:`AgentResult.unwrap` and by agents for expected, typed failures."""

    def __init__(
        self, message: str, *, error_type: str = "AgentFailure", recoverable: bool = False
    ):
        super().__init__(message)
        self.error_type = error_type
        self.recoverable = recoverable


class AgentResult(BaseModel, Generic[OutT]):
    output: OutT | None = None
    error: AgentError | None = None
    trace: TraceRecord

    @property
    def ok(self) -> bool:
        return self.error is None and self.output is not None

    def unwrap(self) -> OutT:
        if self.output is None:
            err = self.error
            raise AgentFailure(
                err.message if err else "no output",
                error_type=err.error_type if err else "NoOutput",
            )
        return self.output


class TraceSink:
    """Thread-safe collector of every agent invocation's trace record."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[TraceRecord] = []

    def add(self, record: TraceRecord) -> None:
        with self._lock:
            self._records.append(record)

    @property
    def records(self) -> list[TraceRecord]:
        with self._lock:
            return list(self._records)


@dataclass
class AgentContext:
    """Dependencies shared by all agents of a pipeline run."""

    config: ExtractionConfig
    llm: LLMClient
    embedder: Embedder
    review_queue: ReviewQueue
    traces: TraceSink = field(default_factory=TraceSink)
    document_id: str | None = None
    services: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC, Generic[InT, OutT]):
    """Template for every agent. Subclasses implement :meth:`_execute`.

    ``run`` never raises: failures come back as an :class:`AgentResult` carrying a typed
    :class:`AgentError`, and every invocation appends a :class:`TraceRecord`.
    """

    name: ClassVar[str]
    algorithm: ClassVar[int]

    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.log = get_logger(f"agent.{self.name}")
        self._usage = TokenUsage()

    # ------------------------------------------------------------------ public API
    def run(self, data: InT, *, trace_context: dict[str, str] | None = None) -> AgentResult[OutT]:
        self._usage = TokenUsage()
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        output: OutT | None = None
        error: AgentError | None = None
        confidence: float | None = None
        try:
            output = self._execute(data)
            confidence = self.confidence(output)
        except AgentFailure as exc:
            error = AgentError(
                agent=self.name,
                error_type=exc.error_type,
                message=str(exc),
                recoverable=exc.recoverable,
            )
        except Exception as exc:
            self.log.exception("agent_crashed", extra={"agent": self.name})
            error = AgentError(agent=self.name, error_type=type(exc).__name__, message=str(exc))
        latency_ms = (time.perf_counter() - t0) * 1000
        trace = TraceRecord(
            agent=self.name,
            algorithm=self.algorithm,
            started_at=started,
            latency_ms=round(latency_ms, 3),
            token_usage=self._usage.model_copy(),
            confidence=confidence,
            status="error" if error else "ok",
            error=error.message if error else None,
            context=trace_context or {},
        )
        self.ctx.traces.add(trace)
        self.log.info(
            "agent_finished",
            extra={
                "agent": self.name,
                "algorithm": self.algorithm,
                "status": trace.status,
                "latency_ms": trace.latency_ms,
                "tokens": self._usage.total,
                "confidence": confidence,
                "doc": self.ctx.document_id,
                **(trace_context or {}),
            },
        )
        return AgentResult[OutT](output=output, error=error, trace=trace)

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _execute(self, data: InT) -> OutT: ...

    def confidence(self, output: OutT) -> float | None:
        """Agent-level confidence in ``[0, 1]`` recorded on the trace (default: none)."""
        validation = getattr(output, "validation", None)
        if isinstance(validation, ValidationOutcome):
            return round(validation.pass_ratio, 4)
        return None

    # ------------------------------------------------------------------ helpers
    def llm_json(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        images: list[bytes] | None = None,
    ) -> SchemaT:
        result = self.ctx.llm.complete_json(
            task=task, system=system, prompt=prompt, schema=schema, images=images or []
        )
        self._usage.add(result.usage)
        return result.value

    def flag(
        self, reason: str, *, item_ref: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        self.ctx.review_queue.flag(
            agent=self.name,
            reason=reason,
            document_id=self.ctx.document_id,
            item_ref=item_ref,
            details=details,
        )

    def flag_if_failed(
        self, outcome: ValidationOutcome, reason: str, *, item_ref: str | None = None
    ) -> bool:
        """Write failed validation to the review queue. Returns True when it passed."""
        if outcome.passed:
            return True
        self.flag(reason, item_ref=item_ref, details={"failures": outcome.failure_reasons()})
        return False
