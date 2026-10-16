"""A5 / Algorithm 6 - Confidence Evaluation Agent.

The evaluated confidence is the LLM confidence, multiplied by ``ungrounded_penalty`` when
the head or tail cannot be found in the chunk text (or its overlap context). Triples
with an evaluated confidence ≥ τ are accepted; the rest are discarded with a reason.
When nothing survives, the framework skips the chunk (Alg. 1 lines 8-10).
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from common.text import STOPWORDS
from construction.agents.triple_extraction import TripleSet
from construction.context import ConstructionAgent
from construction.models import SemanticChunk, Triple, normalize_name


class ConfidenceInput(BaseModel):
    triples: TripleSet
    chunk: SemanticChunk
    abbreviations: dict[str, str] = Field(default_factory=dict)


class ValidatedTriples(BaseModel):
    chunk_id: str
    accepted: list[Triple] = Field(default_factory=list)
    discarded: list[dict[str, str | float]] = Field(default_factory=list)
    tau: float = 0.5

    @property
    def chunk_passes(self) -> bool:
        return bool(self.accepted)

    @property
    def mean_confidence(self) -> float:
        return (
            sum(t.confidence for t in self.accepted) / len(self.accepted) if self.accepted else 0.0
        )


_TOKEN = re.compile(r"[a-z0-9]+")


def significant_tokens(text: str) -> list[str]:
    """Numbers and content words longer than two characters (units like "h" are ignored)."""
    return [
        t
        for t in _TOKEN.findall(text.lower())
        if t.isdigit() or (len(t) > 2 and t not in STOPWORDS)
    ]


def grounded(name: str, text: str, aliases: list[str]) -> bool:
    """True when the name (or an alias) occurs in the text, verbatim or token by token."""
    hay = normalize_name(text)
    hay_tokens = set(_TOKEN.findall(hay))
    for candidate in [name, *aliases]:
        cand = normalize_name(candidate)
        if not cand:
            continue
        if cand in hay:
            return True
        tokens = significant_tokens(candidate)
        if tokens and all(t in hay_tokens for t in tokens):
            return True
    return False


class ConfidenceEvaluationAgent(ConstructionAgent[ConfidenceInput, ValidatedTriples]):
    name: ClassVar[str] = "confidence_evaluation"
    algorithm: ClassVar[int] = 6
    agent_id: ClassVar[str] = "A5"

    def _execute(self, data: ConfidenceInput) -> ValidatedTriples:
        cfg = self.settings.confidence
        text = f"{data.chunk.context}\n{data.chunk.text}"
        reverse_abbr: dict[str, list[str]] = {}
        for short, long in data.abbreviations.items():
            reverse_abbr.setdefault(normalize_name(long), []).append(short)
        out = ValidatedTriples(chunk_id=data.chunk.chunk_id, tau=cfg.tau)
        for t in data.triples.triples:
            aliases_h = [t.mentions.get("head", ""), *reverse_abbr.get(normalize_name(t.head), [])]
            aliases_t = [t.mentions.get("tail", ""), *reverse_abbr.get(normalize_name(t.tail), [])]
            is_grounded = grounded(t.head, text, aliases_h) and grounded(t.tail, text, aliases_t)
            score = t.llm_confidence * (1.0 if is_grounded else cfg.ungrounded_penalty)
            score = round(min(1.0, max(0.0, score)), 4)
            notes = [*t.notes] + ([] if is_grounded else ["ungrounded"])
            evaluated = t.model_copy(
                update={
                    "confidence": score,
                    "notes": notes,
                    "provenance": t.provenance.model_copy(update={"confidence": score}),
                }
            )
            if score >= cfg.tau:
                out.accepted.append(evaluated)
            else:
                out.discarded.append(
                    {
                        "triple_id": t.triple_id,
                        "head": t.head,
                        "relation": t.relation,
                        "tail": t.tail,
                        "confidence": score,
                        "reason": f"confidence {score} < τ={cfg.tau}"
                        + ("" if is_grounded else " (ungrounded)"),
                    }
                )
        return out

    def confidence(self, output: ValidatedTriples) -> float | None:
        return round(output.mean_confidence, 4)
