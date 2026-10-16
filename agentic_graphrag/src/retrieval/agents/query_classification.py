"""A1 / Algorithm 2 - Query Classification Agent.

Two features drive the decision (Alg. 2 line 2, "analyse intent and semantic features"):

* **entity_specific** - Q names a KG entity. Detected by matching entity names/aliases in
  the query text (acronyms expanded first, so "HCU" finds "Hydraulic Control Unit"), then
  by embedding query n-grams against the entity index (cosine ≥
  ``classification.entity_min_similarity``).
* **thematic** - Q needs community-level reasoning. Detected from thematic cues
  ("overall", "main themes", "across" …).

With ``classification.use_llm`` the LLM sees the heuristic features as hints and makes the
final call on both booleans. An LLM failure or an empty answer keeps the heuristic.

The PDF's if/else tests Local before Hybrid, which makes Hybrid unreachable (a query that
needs both entity and thematic reasoning is also entity-specific). The rule is evaluated
in this order instead (DESIGN_DECISIONS Stage 3 #1):

    entity_specific and thematic → Hybrid
    entity_specific              → Local
    thematic                     → Global
    otherwise                    → Simple Lookup
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel

from common.llm import LLMError
from retrieval.context import RetrievalAgent
from retrieval.models import Classification, QueryType
from retrieval.text import ACRONYM, acronym_table, find_entity_mentions

SYSTEM = (
    "You classify a user question for a knowledge-graph retrieval system. Decide two "
    "booleans: entity_specific (the question asks about one or more specific named "
    "entities, components, standards or values) and thematic (it needs themes, summaries "
    "or reasoning across many parts of the corpus). HINTS come from exact matching against "
    "the knowledge graph; overrule them only when clearly wrong. Give a confidence in [0, 1]."
)

_LOOKUP_CUES = re.compile(r"^\s*(?:what|when|where|who|which|how many|how much|define)\b", re.I)


class ClassificationInput(BaseModel):
    query: str


class LLMClassification(BaseModel):
    entity_specific: bool = False
    thematic: bool = False
    confidence: float = 0.0
    rationale: str = ""


def decide(entity_specific: bool, thematic: bool) -> QueryType:
    """Algorithm 2 lines 3-11 with Hybrid evaluated first."""
    if entity_specific and thematic:
        return QueryType.HYBRID
    if entity_specific:
        return QueryType.LOCAL
    if thematic:
        return QueryType.GLOBAL
    return QueryType.SIMPLE_LOOKUP


def _ngrams(text: str, n_max: int = 3, limit: int = 40) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][\w\-]*", text)
    out: list[str] = []
    for n in range(n_max, 0, -1):
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i : i + n])
            if len(gram) > 2 and gram.lower() not in out:
                out.append(gram.lower())
    return out[:limit]


class QueryClassificationAgent(RetrievalAgent[ClassificationInput, Classification]):
    name: ClassVar[str] = "query_classification"
    algorithm: ClassVar[int] = 2
    agent_id: ClassVar[str] = "A1"

    def thematic_cues(self, query: str) -> list[str]:
        low = f" {query.lower()} "
        return [c for c in self.settings.classification.thematic_cues if f" {c} " in low]

    def entity_matches(self, query: str) -> dict[str, tuple[str, float]]:
        """node_id → (name, score) for KG entities named in the query (acronyms expanded)."""
        table = acronym_table(self.ctx.graph) if ACRONYM.search(query) else {}
        expansions = [table[a] for a in ACRONYM.findall(query) if a in table]
        text = " ; ".join([query, *expansions])
        out = {
            m.node.id: (m.node.name, m.score) for m in find_entity_mentions(text, self.ctx.graph)
        }
        threshold = self.settings.classification.entity_min_similarity
        grams = _ngrams(query)
        if grams:
            vectors = self.ctx.embedder.embed(grams)
            for vec in vectors:
                for node, score in self.ctx.graph.similar_nodes(vec, label="Entity", top_k=1):
                    if score >= threshold and score > out.get(node.id, ("", 0.0))[1]:
                        out[node.id] = (node.name, round(score, 4))
        return out

    def _execute(self, data: ClassificationInput) -> Classification:
        query = data.query.strip()
        matches = self.entity_matches(query)
        cues = self.thematic_cues(query)
        entity_specific, thematic = bool(matches), bool(cues)
        best = max((s for _, s in matches.values()), default=0.0)
        method = "heuristic"
        rationale = ""
        confidence = self._heuristic_confidence(query, entity_specific, thematic, best, len(cues))
        if self.settings.classification.use_llm:
            hints = (
                f"HINTS: entity_specific={str(entity_specific).lower()} "
                f"thematic={str(thematic).lower()}\n"
                f"KG_ENTITIES_IN_QUERY: {'; '.join(n for n, _ in matches.values()) or '-'}\n"
                f"THEMATIC_CUES: {'; '.join(cues) or '-'}"
            )
            try:
                out = self.llm_json(
                    task="query_classification",
                    system=SYSTEM,
                    prompt=f"QUESTION: {query}\n{hints}",
                    schema=LLMClassification,
                )
                if out.confidence > 0:  # an empty/default answer keeps the heuristic
                    entity_specific, thematic = out.entity_specific, out.thematic
                    confidence = min(1.0, max(0.0, out.confidence))
                    method, rationale = "llm", out.rationale
            except LLMError as exc:
                self.log.warning("classification_llm_failed", extra={"error": str(exc)[:200]})
        return Classification(
            query_type=decide(entity_specific, thematic),
            confidence=round(confidence, 4),
            entity_specific=entity_specific,
            thematic=thematic,
            entities=[n for n, _ in matches.values()],
            entity_ids=list(matches),
            cues=cues,
            method="llm" if method == "llm" else "heuristic",
            rationale=rationale or self._rationale(entity_specific, thematic, matches, cues),
        )

    @staticmethod
    def _heuristic_confidence(
        query: str, entity_specific: bool, thematic: bool, best: float, n_cues: int
    ) -> float:
        entity_conf = 0.6 + 0.3 * best
        theme_conf = 0.6 + 0.1 * min(3, n_cues)
        if entity_specific and thematic:
            return (entity_conf + theme_conf) / 2
        if entity_specific:
            return entity_conf
        if thematic:
            return theme_conf
        return 0.7 if _LOOKUP_CUES.search(query) else 0.55

    @staticmethod
    def _rationale(
        entity_specific: bool,
        thematic: bool,
        matches: dict[str, tuple[str, float]],
        cues: list[str],
    ) -> str:
        parts = []
        if entity_specific:
            parts.append("names KG entities: " + ", ".join(n for n, _ in matches.values()))
        if thematic:
            parts.append("thematic cues: " + ", ".join(cues))
        return "; ".join(parts) or "no KG entity and no thematic cue"

    def confidence(self, output: Classification) -> float | None:
        return output.confidence


__all__ = ["ClassificationInput", "LLMClassification", "QueryClassificationAgent", "decide"]
