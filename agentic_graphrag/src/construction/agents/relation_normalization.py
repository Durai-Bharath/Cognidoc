"""A7 / Algorithm 8 - Relation Normalization Agent.

Each extracted relation is mapped onto an ontology predicate by: an exact name match,
then a synonym, then a stemmed synonym ("inverse" synonyms swap head and tail). When
nothing matches and ``relations.llm_fallback`` is on, the LLM picks a predicate from the
ontology or answers ``NONE``. A relation that still has no predicate is marked rejected,
and Conflict Detection (A8) turns it into a Reject. LLM answers are cached per relation
phrase for the run.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.llm import LLMError
from construction.context import ConstructionAgent, ConstructionContext
from construction.models import NormalizedTriple, ResolvedTriple
from construction.ontology import PredicateMatch

SYSTEM = (
    "Map a relation phrase from a knowledge-graph triple onto one predicate of a fixed "
    "ontology. Answer with the predicate name exactly as listed, or NONE when no predicate "
    "means the same thing. Set inverted=true when the predicate holds with head and tail "
    "swapped (e.g. 'is controlled by' → regulates, inverted)."
)


class RelationChoice(BaseModel):
    predicate: str = "NONE"
    inverted: bool = False
    reason: str = ""


class NormalizationInput(BaseModel):
    items: list[ResolvedTriple]


class NormalizedTriples(BaseModel):
    items: list[NormalizedTriple] = Field(default_factory=list)

    @property
    def rejected(self) -> list[NormalizedTriple]:
        return [i for i in self.items if i.predicate is None]


class RelationNormalizationAgent(ConstructionAgent[NormalizationInput, NormalizedTriples]):
    name: ClassVar[str] = "relation_normalization"
    algorithm: ClassVar[int] = 8
    agent_id: ClassVar[str] = "A7"

    def __init__(self, ctx: ConstructionContext) -> None:
        super().__init__(ctx)
        self._llm_cache: dict[str, PredicateMatch | None] = {}

    def _llm_match(self, relation: str, example: ResolvedTriple) -> PredicateMatch | None:
        key = relation.lower().strip()
        if key in self._llm_cache:
            return self._llm_cache[key]
        onto = self.ctx.ontology
        prompt = (
            f"PREDICATES:\n{onto.describe_predicates()}\n\n"
            f"TRIPLE: ({example.head.name}) -[{relation}]-> ({example.tail.name})\n"
            f"RELATION: {relation}"
        )
        match: PredicateMatch | None = None
        try:
            choice = self.llm_json(
                task="relation_normalization", system=SYSTEM, prompt=prompt, schema=RelationChoice
            )
            pred = choice.predicate.strip()
            if pred in onto.predicates:
                match = PredicateMatch(predicate=pred, inverted=choice.inverted, method="llm")
        except LLMError as exc:
            self.log.warning("relation_llm_failed", extra={"relation": relation, "error": str(exc)})
        self._llm_cache[key] = match
        return match

    def _execute(self, data: NormalizationInput) -> NormalizedTriples:
        onto = self.ctx.ontology
        out = NormalizedTriples()
        for item in data.items:
            relation = item.triple.relation
            match = onto.match_relation(relation)
            if match is None and self.settings.relations.llm_fallback:
                match = self._llm_match(relation, item)
            if match is None:
                out.items.append(
                    NormalizedTriple(
                        resolved=item,
                        rejected_reason=f"relation {relation!r} has no ontology predicate",
                    )
                )
                continue
            out.items.append(
                NormalizedTriple(
                    resolved=item,
                    predicate=match.predicate,
                    method=match.method,
                    inverted=match.inverted,
                )
            )
        return out
