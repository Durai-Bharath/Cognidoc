"""A2 / Algorithm 3 - Query Rewriting Agent.

Deterministic steps, each recorded in the output:

1. **Conversational references** (line 2). The pronouns it/its/they/them/their resolve to
   the KG entity mentioned last in the history (most recent turn first). Value entities
   (Quantity, Status, Date) are skipped unless a turn mentions nothing else.
2. **Acronyms** (line 3). An acronym is expanded as "Expansion (ACR)". Sources, in
   priority order: ``rewriting.acronyms``, KG aliases written in capitals, and
   "Long Form (LF)" definitions found in the stored chunk texts (Stage 1's
   Schwartz-Hearst matcher).
3. **Spelling and terminology** (line 4). A word that is not corpus vocabulary (entity
   names, aliases and chunk text) and shares no stem with it is replaced by its closest
   corpus word when the difflib ratio is ≥ ``rewriting.fuzzy_cutoff``. A non-canonical
   entity alias is normalised to the entity's name.
4. **Enrichment** (line 5). The types of the named entities are recorded as enrichment
   terms for retrieval.

With ``rewriting.use_llm`` the LLM may polish the draft. Its version is accepted only if it
is non-empty and keeps every entity name the draft resolved; otherwise the draft stands.
"""

from __future__ import annotations

import difflib
import re
from typing import ClassVar

from pydantic import BaseModel, Field

from common.llm import LLMError
from common.text import STOPWORDS
from retrieval.context import RetrievalAgent
from retrieval.models import QueryType, RewrittenQuery, Turn
from retrieval.text import ACRONYM, QUESTION_WORDS, acronym_table, find_entity_mentions, stem

SYSTEM = (
    "You rewrite a user question for retrieval over a knowledge graph. Keep its meaning. "
    "Keep every entity name that appears in DRAFT exactly. Resolve references using "
    "HISTORY, expand acronyms, fix spelling. Return one self-contained question."
)

VALUE_TYPES = frozenset({"Quantity", "Status", "Date"})
_PRONOUN = re.compile(r"\b(its|it|they|them|their)\b", re.I)


class RewritingInput(BaseModel):
    query: str
    query_type: QueryType
    history: list[Turn] = Field(default_factory=list)


class LLMRewrite(BaseModel):
    rewritten: str = ""


class QueryRewritingAgent(RetrievalAgent[RewritingInput, RewrittenQuery]):
    name: ClassVar[str] = "query_rewriting"
    algorithm: ClassVar[int] = 3
    agent_id: ClassVar[str] = "A2"

    # ------------------------------------------------------------------ line 2
    def _last_entity(self, history: list[Turn]) -> str | None:
        for turn in reversed(history[-self.settings.rewriting.max_history_turns :]):
            mentions = find_entity_mentions(turn.content, self.ctx.graph)
            # Values (statuses, quantities, dates) are rarely what "it" refers to.
            subjects = [m for m in mentions if m.node.type not in VALUE_TYPES] or mentions
            if subjects:
                low = turn.content.lower()
                subjects.sort(key=lambda m: low.rfind(m.surface))
                return subjects[-1].node.name
        return None

    def resolve_references(self, query: str, history: list[Turn]) -> tuple[str, dict[str, str]]:
        if not history or not _PRONOUN.search(query):
            return query, {}
        entity = self._last_entity(history)
        if entity is None:
            return query, {}
        resolved: dict[str, str] = {}

        def sub(m: re.Match[str]) -> str:
            word = m.group(1)
            repl = f"{entity}'s" if word.lower() in ("its", "their") else entity
            resolved[word.lower()] = repl
            return repl

        return _PRONOUN.sub(sub, query), resolved

    # ------------------------------------------------------------------ line 3
    def expand_acronyms(self, query: str) -> tuple[str, dict[str, str]]:
        table = acronym_table(self.ctx.graph, self.settings.rewriting.acronyms)
        expanded: dict[str, str] = {}

        def sub(m: re.Match[str]) -> str:
            acr = m.group(0)
            key = acr if acr in table else acr.rstrip("s")
            full = table.get(key)
            if full is None or full.lower() in query.lower():
                return acr
            expanded[key] = full
            return f"{full} ({acr})"

        return ACRONYM.sub(sub, query), expanded

    # ------------------------------------------------------------------ line 4
    def vocabulary(self) -> set[str]:
        """Corpus words: entity names, aliases and chunk text."""
        words = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
        known: set[str] = set()
        for node in self.ctx.graph.nodes("Entity"):
            for form in (node.name, *node.aliases):
                known.update(w.lower() for w in words.findall(form))
        for chunk in self.ctx.graph.nodes("Chunk"):
            known.update(w.lower() for w in words.findall(str(chunk.properties.get("text", ""))))
        return known

    def correct_spelling(self, query: str) -> tuple[str, dict[str, str]]:
        known = self.vocabulary()
        targets = known
        if not targets:
            return query, {}
        stems = {stem(w) for w in known}
        cutoff = self.settings.rewriting.fuzzy_cutoff
        fixed: dict[str, str] = {}

        def sub(m: re.Match[str]) -> str:
            word = m.group(0)
            low = word.lower()
            if len(low) < 4 or low in known or low in STOPWORDS or low in QUESTION_WORDS:
                return word
            if word.isupper() or stem(low) in stems:
                return word
            close = difflib.get_close_matches(low, sorted(targets), n=1, cutoff=cutoff)
            if not close:
                return word
            fixed[word] = close[0]
            return close[0].capitalize() if word[0].isupper() else close[0]

        return re.sub(r"[A-Za-z][A-Za-z\-]+", sub, query), fixed

    def normalise_aliases(self, query: str) -> tuple[str, dict[str, str]]:
        changed: dict[str, str] = {}
        for m in find_entity_mentions(query, self.ctx.graph):
            name = m.node.name
            alias = next((a for a in m.node.aliases if a.lower() == m.surface), "")
            if m.surface != name.lower() and not alias.isupper():
                pattern = re.compile(rf"\b{re.escape(m.surface)}\b", re.I)
                if pattern.search(query) and name.lower() not in query.lower():
                    query = pattern.sub(name, query, count=1)
                    changed[m.surface] = name
        return query, changed

    # ------------------------------------------------------------------ main
    def _execute(self, data: RewritingInput) -> RewrittenQuery:
        text = re.sub(r"\s+", " ", data.query).strip()
        text, resolved = self.resolve_references(text, data.history)
        text, expansions = self.expand_acronyms(text)
        text, corrections = self.correct_spelling(text)
        text, normalised = self.normalise_aliases(text)
        corrections.update(normalised)
        mentions = find_entity_mentions(text, self.ctx.graph)
        enrichment = sorted({m.node.type for m in mentions if m.node.type})
        method = "heuristic"
        if self.settings.rewriting.use_llm:
            history = "\n".join(f"{t.role}: {t.content}" for t in data.history[-6:]) or "-"
            try:
                out = self.llm_json(
                    task="query_rewriting",
                    system=SYSTEM,
                    prompt=f"ORIGINAL: {data.query}\nHISTORY:\n{history}\nDRAFT: {text}",
                    schema=LLMRewrite,
                )
                candidate = out.rewritten.strip()
                keeps = all(m.node.name.lower() in candidate.lower() for m in mentions)
                if candidate and keeps:
                    method = "llm" if candidate != text else method
                    text = candidate
            except LLMError as exc:
                self.log.warning("rewriting_llm_failed", extra={"error": str(exc)[:200]})
        return RewrittenQuery(
            original=data.query,
            rewritten=text,
            query_type=data.query_type,
            resolved_references=resolved,
            expansions=expansions,
            corrections=corrections,
            enrichment=enrichment,
            method="llm" if method == "llm" else "heuristic",
        )
