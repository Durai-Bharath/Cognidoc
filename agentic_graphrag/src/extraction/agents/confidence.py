"""Algorithm 19 - Confidence Scoring Agent."""

from __future__ import annotations

import math

from common.base_agent import BaseAgent
from common.config import ConfidenceConfig
from common.validation import CheckResult, ValidationOutcome
from extraction.schemas import (
    ConfidenceInput,
    ConfidenceLevel,
    ConfidenceRepository,
    ObjectConfidence,
)


def level_for(score: float, cfg: ConfidenceConfig) -> ConfidenceLevel:
    if score >= cfg.high_threshold:
        return ConfidenceLevel.HIGH
    if score >= cfg.medium_threshold:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    """C_i = Σ w_k c_k, renormalised over the components that are present (weight > 0)."""
    used = {k: v for k, v in components.items() if weights.get(k, 0.0) > 0}
    total = sum(weights[k] for k in used)
    if total == 0:
        present = list(components.values())
        return sum(present) / len(present) if present else 0.0
    return sum(weights[k] * v for k, v in used.items()) / total


class ConfidenceScoringAgent(BaseAgent[ConfidenceInput, ConfidenceRepository]):
    name = "ConfidenceScoringAgent"
    algorithm = 19

    def _execute(self, data: ConfidenceInput) -> ConfidenceRepository:
        repo = ConfidenceRepository(section_id=data.section_id)
        repo.objects = self._score(data, sanitize=False)
        self._document_level(repo, data)
        repo.validation = self.validate(repo, data)
        if not repo.validation.passed:
            # Line 34: recompute inconsistent scores from sanitised inputs.
            repo.objects = self._score(data, sanitize=True)
            self._document_level(repo, data)
            repo.recomputed = True
            repo.validation = self.validate(repo, data)
            if not repo.validation.passed:
                self.flag(
                    "confidence scores still inconsistent after recomputation",
                    item_ref=data.section_id,
                    details={"failures": repo.validation.failure_reasons()},
                )
        return repo

    def _score(self, data: ConfidenceInput, *, sanitize: bool) -> list[ObjectConfidence]:
        cfg = self.ctx.config.confidence
        objects: list[ObjectConfidence] = []
        for object_id, raw in data.component_scores.items():
            comps: dict[str, float] = {}
            for name, value in raw.items():
                if value is None:
                    continue
                if sanitize:
                    if math.isnan(value):
                        continue
                    value = min(1.0, max(0.0, value))
                comps[name] = round(value, 4)
            score = weighted_score(comps, cfg.weights)
            score = round(score, 4) if not math.isnan(score) else score
            objects.append(
                ObjectConfidence(
                    object_id=object_id,
                    kind=data.object_kinds.get(object_id, "section"),
                    score=score,
                    level=level_for(score, cfg) if not math.isnan(score) else ConfidenceLevel.LOW,
                    components=comps,
                    weight=data.section_weight
                    if data.object_kinds.get(object_id) == "section"
                    else 1.0,
                )
            )
        return objects

    def _document_level(self, repo: ConfidenceRepository, data: ConfidenceInput) -> None:
        sections = [o for o in [*data.prior_sections, *repo.objects] if o.kind == "section"]
        total = sum(o.weight for o in sections)
        if total > 0:
            score = sum(o.score * o.weight for o in sections) / total
        else:
            score = 0.0
        repo.document_score = round(score, 4) if not math.isnan(score) else score
        repo.document_level = level_for(
            score if not math.isnan(score) else 0.0, self.ctx.config.confidence
        )

    def validate(self, repo: ConfidenceRepository, data: ConfidenceInput) -> ValidationOutcome:
        cfg = self.ctx.config.confidence
        outcome = ValidationOutcome()
        missing: list[str] = []
        for obj in repo.objects:
            if obj.kind == "section":
                missing += [
                    f"{obj.object_id} missing {req}"
                    for req in cfg.required_components
                    if req not in obj.components
                ]
        outcome.add(CheckResult.from_failures("missing_scores", missing))
        invalid = [
            f"{oid}.{name}={value}"
            for oid, comps in data.component_scores.items()
            for name, value in comps.items()
            if value is not None and (math.isnan(value) or not 0.0 <= value <= 1.0)
        ]
        invalid += [
            f"{o.object_id} score {o.score}"
            for o in repo.objects
            if math.isnan(o.score) or not 0.0 <= o.score <= 1.0
        ]
        outcome.add(CheckResult.from_failures("invalid_values", invalid))
        inconsistent: list[str] = []
        for o in repo.objects:
            expected = weighted_score(o.components, cfg.weights)
            if (
                math.isnan(o.score)
                or abs(expected - o.score) > 1e-3
                or level_for(o.score, cfg) != o.level
                or not 0.0 <= o.score <= 1.0
            ):
                inconsistent.append(f"{o.object_id} score/level inconsistent")
        if math.isnan(repo.document_score) or not 0.0 <= repo.document_score <= 1.0:
            inconsistent.append(f"document score {repo.document_score}")
        outcome.add(CheckResult.from_failures("score_consistency", inconsistent))
        return outcome

    def confidence(self, output: ConfidenceRepository) -> float | None:
        return output.document_score if not math.isnan(output.document_score) else None
