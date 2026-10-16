"""A14 / Algorithm 15 - Retry Controller Agent.

Receives the validation status (line 1).

* Valid → ``accept``: return the validated answer (lines 2-3).
* Invalid and fewer than ``retry.max_retries`` retries used → ``retry`` with modified
  parameters (line 5); the framework restarts the workflow at the Vector Retrieval Agent
  (line 6). Retry *i* applies ``retry.escalation[i − 1]`` (cycled):

  - ``increase_k``: top_k and entity_top_k += ``k_step`` (≤ ``max_k``)
  - ``increase_hops``: hops += ``hop_step`` (≤ ``max_hops``)
  - ``switch_to_hybrid``: query type → Hybrid, with at least Hybrid's hops and community
    weight
  - ``widen_communities``: community_top_n += ``community_step`` (≤ ``max_communities``),
    community weight ≥ ``min_community_weight``

  An escalation that would change nothing (already at its cap) is skipped for the next
  one in the list. When none changes anything, retrying cannot help, so the run ends.
* Otherwise → ``exhausted`` (line 7). The framework returns the best-scoring attempt with
  ``verified=false`` and a low-confidence flag.

Termination: the attempt number strictly increases and ``max_retries`` is bounded (≤ 10),
so at most ``max_retries + 1`` attempts ever run.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from retrieval.context import RetrievalAgent
from retrieval.models import QueryType, RetrievalParams, RetryDecision, ValidationStatus


class RetryInput(BaseModel):
    status: ValidationStatus
    attempt: int
    params: RetrievalParams


class RetryControllerAgent(RetrievalAgent[RetryInput, RetryDecision]):
    name: ClassVar[str] = "retry_controller"
    algorithm: ClassVar[int] = 15
    agent_id: ClassVar[str] = "A14"

    def escalate(self, params: RetrievalParams, step: str) -> RetrievalParams:
        cfg = self.settings.retry
        p = params.model_copy()
        if step == "increase_k":
            p.top_k = min(cfg.max_k, p.top_k + cfg.k_step)
            p.entity_top_k = min(cfg.max_k, p.entity_top_k + cfg.k_step)
        elif step == "increase_hops":
            p.hops = min(cfg.max_hops, p.hops + cfg.hop_step)
        elif step == "switch_to_hybrid":
            p.query_type = QueryType.HYBRID
            p.hops = max(p.hops, min(cfg.max_hops, int(self.settings.traversal.hops["Hybrid"])))
            p.community_weight = max(p.community_weight, self.settings.community.weights["Hybrid"])
        elif step == "widen_communities":
            p.community_top_n = min(cfg.max_communities, p.community_top_n + cfg.community_step)
            p.community_weight = max(p.community_weight, cfg.min_community_weight)
        return p

    def _execute(self, data: RetryInput) -> RetryDecision:
        cfg = self.settings.retry
        if data.status.valid:
            return RetryDecision(
                action="accept", attempt=data.attempt, params=data.params, reason="valid"
            )
        retries_used = data.attempt - 1
        if retries_used >= cfg.max_retries:
            return RetryDecision(
                action="exhausted",
                attempt=data.attempt,
                params=data.params,
                reason=f"retry limit {cfg.max_retries} reached",
            )
        n = len(cfg.escalation)
        for offset in range(n):
            step = cfg.escalation[(retries_used + offset) % n]
            new = self.escalate(data.params, step)
            if new != data.params:
                return RetryDecision(
                    action="retry",
                    attempt=data.attempt + 1,
                    params=new,
                    escalation=step,
                    reason=f"score {data.status.score} < τ {data.status.tau}"
                    if data.status.score < data.status.tau
                    else "; ".join(data.status.reasons[:3]) or "invalid",
                )
        return RetryDecision(
            action="exhausted",
            attempt=data.attempt,
            params=data.params,
            reason="every escalation is at its limit",
        )
