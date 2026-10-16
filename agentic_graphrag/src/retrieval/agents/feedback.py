"""A13 / Algorithm 14 - Feedback Agent.

Receives the (critiqued) answer (line 1) and identifies the graph knowledge behind it
(line 2):

* **failed answer** → the entities, communities and relationships cited by unsupported or
  contradictory statements (``VerifiedAnswer.implicated_*``) are *penalised*;
* **verified answer** (``feedback.reinforce_verified``) → the elements cited by supported
  statements are *reinforced*. This branch is not in Alg. 1, which calls A13 only on
  failure (DESIGN_DECISIONS Stage 3 #2).

Line 3, bounded updates: one step per answer (``penalty_step`` / ``reward_step``), clamped
to [``floor``, ``ceiling``]. A value already outside that band is never moved further by
the step. Edges change ``confidence`` directly. Entity and Community nodes derive their
confidence from provenance, so feedback writes ``properties.retrieval_confidence``, which
A7 multiplies into a fact's ranking. Chunk and Document nodes are source text and are
never adjusted.

Line 4: a penalised element that falls below ``feedback.flag_below``, or has been
penalised more than once, is flagged to the review queue as inconsistent.

Line 5: every update is appended to ``paths.feedback_log`` (JSONL). Penalised and flagged
ids are returned as ``changed_ids``, so the framework can invalidate cache entries that
cite them (A15).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from construction.models import utc_now
from construction.stores import save_state
from retrieval.context import RetrievalAgent
from retrieval.models import (
    FeedbackReport,
    FeedbackUpdate,
    GroundedAnswer,
    ValidationStatus,
    VerifiedAnswer,
)

_LOG_LOCK = threading.Lock()


class FeedbackInput(BaseModel):
    query: str
    attempt: int = 1
    grounded: GroundedAnswer
    status: ValidationStatus
    critique: VerifiedAnswer | None = None


def bounded_step(value: float, step: float, floor: float, ceiling: float) -> float:
    """Move ``value`` by ``step`` (signed) without crossing [floor, ceiling]."""
    if step < 0:
        return value if value <= floor else round(max(floor, value + step), 4)
    return value if value >= ceiling else round(min(ceiling, value + step), 4)


class FeedbackAgent(RetrievalAgent[FeedbackInput, FeedbackReport]):
    name: ClassVar[str] = "feedback"
    algorithm: ClassVar[int] = 14
    agent_id: ClassVar[str] = "A13"

    def _targets(self, data: FeedbackInput) -> tuple[str, list[str], list[str]]:
        cfg = self.settings.feedback
        if not data.status.valid:
            c = data.critique
            return (
                ("penalize", c.implicated_node_ids, c.implicated_edge_ids)
                if c
                else ("none", [], [])
            )
        if not cfg.reinforce_verified:
            return "none", [], []
        if data.critique is not None:
            return "reinforce", data.critique.supported_node_ids, data.critique.supported_edge_ids
        supported = {c.statement_id for c in data.status.claims if c.supported}
        nodes: list[str] = []
        edges: list[str] = []
        for s in data.grounded.statements:
            if s.statement_id in supported:
                nodes += s.entity_ids
                for src in s.sources:
                    if src.kind != "chunk":
                        nodes += src.node_ids
                    edges += src.edge_ids
        return "reinforce", list(dict.fromkeys(nodes)), list(dict.fromkeys(edges))

    def _execute(self, data: FeedbackInput) -> FeedbackReport:
        cfg = self.settings.feedback
        graph = self.ctx.graph
        mode, node_ids, edge_ids = self._targets(data)
        out = FeedbackReport(mode="none" if mode == "none" else mode)
        if mode == "none":
            return out
        direction: Literal["penalize", "reinforce"] = (
            "penalize" if mode == "penalize" else "reinforce"
        )
        step = -cfg.penalty_step if direction == "penalize" else cfg.reward_step
        counter = "penalties" if direction == "penalize" else "rewards"
        now = utc_now()

        for eid in edge_ids:
            edge = graph.get_edge(eid)
            if edge is None:
                out.missing.append(eid)
                continue
            before = edge.confidence
            edge.confidence = bounded_step(before, step, cfg.floor, cfg.ceiling)
            fb: dict[str, Any] = dict(edge.properties.get("feedback") or {})
            fb[counter] = int(fb.get(counter, 0)) + 1
            fb["last_feedback_at"] = now
            edge.properties["feedback"] = fb
            graph.upsert_edge(edge)
            out.updates.append(
                FeedbackUpdate(
                    element_id=eid,
                    kind="edge",
                    direction=direction,
                    before=before,
                    after=edge.confidence,
                )
            )
            if direction == "penalize":
                self._maybe_flag(out, eid, edge.confidence, int(fb["penalties"]), data)

        for nid in node_ids:
            node = graph.get_node(nid)
            if node is None:
                out.missing.append(nid)
                continue
            if node.label not in ("Entity", "Community"):
                continue
            raw = node.properties.get("retrieval_confidence")
            before = float(raw) if isinstance(raw, int | float) else node.confidence
            after = bounded_step(before, step, cfg.floor, cfg.ceiling)
            fb = dict(node.properties.get("feedback") or {})
            fb[counter] = int(fb.get(counter, 0)) + 1
            fb["last_feedback_at"] = now
            node.properties["retrieval_confidence"] = after
            node.properties["feedback"] = fb
            graph.upsert_node(node)
            out.updates.append(
                FeedbackUpdate(
                    element_id=nid, kind="node", direction=direction, before=before, after=after
                )
            )
            if direction == "penalize":
                self._maybe_flag(out, nid, after, int(fb["penalties"]), data)

        if direction == "penalize":
            out.changed_ids = list(dict.fromkeys([u.element_id for u in out.updates] + out.flagged))
        out.logged = self._log(data, out)
        if cfg.persist_state and out.updates:
            save_state(graph, self.ctx.vectors, self.settings.paths.state_dir)
        return out

    def _maybe_flag(
        self,
        out: FeedbackReport,
        element_id: str,
        after: float,
        penalties: int,
        data: FeedbackInput,
    ) -> None:
        if after < self.settings.feedback.flag_below or penalties > 1:
            out.flagged.append(element_id)
            self.flag(
                "inconsistent graph knowledge implicated in a failed answer",
                item_ref=element_id,
                details={
                    "query": data.query,
                    "confidence": after,
                    "penalties": penalties,
                    "unsupported": data.status.unsupported,
                },
            )

    def _log(self, data: FeedbackInput, out: FeedbackReport) -> bool:
        path = self.settings.paths.feedback_log
        if not path:
            return False
        record = {
            "at": utc_now(),
            "query": data.query,
            "attempt": data.attempt,
            "mode": out.mode,
            "score": data.status.score,
            "valid": data.status.valid,
            "updates": [u.model_dump() for u in out.updates],
            "flagged": out.flagged,
        }
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK, p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            self.log.exception("feedback_log_write_failed", extra={"path": path})
            return False
        return True
