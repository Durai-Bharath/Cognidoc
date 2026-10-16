"""A3 / Algorithm 4 - Query Decomposition Agent.

Complexity analysis (line 2) splits the rewritten query into question clauses: several
question marks, ";", ", and then", or " and " followed by a new question
("…, and what is its interval?"). One clause → the whole query is the single subquery
``q1`` (line 6). Several → ordered subqueries (line 4).

Dependencies (line 7) are explicit: a later clause that refers back with a pronoun
(it/its/they/them/their) gets the placeholder ``{qN}`` for the clause it refers to and
``depends_on=[qN]``. The framework fills the placeholder with ``qN``'s intermediate
answer before retrieving for the dependent subquery.

With ``decomposition.use_llm`` the LLM may propose its own decomposition. It is accepted
only when it validates: ids ``q1..qn`` in order, dependencies on earlier ids only, every
placeholder backed by a dependency, and at most ``decomposition.max_subqueries``.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar

from pydantic import BaseModel, Field

from common.llm import LLMError
from retrieval.context import RetrievalAgent
from retrieval.models import Decomposition, QueryType, Subquery

SYSTEM = (
    "You split a complex question into the smallest ordered list of simple subquestions. "
    "Use ids q1, q2, ... in order. When a subquestion needs the answer of an earlier one, "
    "write the placeholder {qN} where that answer goes and list qN in depends_on. A simple "
    "question stays one subquestion. DRAFT is a rule-based proposal you may keep."
)

_QWORD = r"(?:what|which|who|whom|whose|when|where|why|how|is|are|does|do|did|can|list|describe)"
_SPLIT = re.compile(
    rf"\?\s+(?=\S)|;\s*|,?\s+and then\s+|,?\s+and\s+(?={_QWORD}\b)|,\s+(?={_QWORD}\b)",
    re.I,
)
_BACKREF = re.compile(r"\b(its|it|they|them|their)\b", re.I)


class DecompositionInput(BaseModel):
    query: str
    query_type: QueryType


class LLMSubquery(BaseModel):
    id: str = ""
    text: str = ""
    depends_on: list[str] = Field(default_factory=list)


class LLMDecomposition(BaseModel):
    complex: bool = False
    subqueries: list[LLMSubquery] = Field(default_factory=list)


def validate_subqueries(subqueries: list[Subquery], max_n: int) -> list[str]:
    """Problems with a decomposition (empty list = valid)."""
    problems: list[str] = []
    if not subqueries:
        return ["no subqueries"]
    if len(subqueries) > max_n:
        problems.append(f"{len(subqueries)} subqueries > max {max_n}")
    seen: list[str] = []
    for i, sq in enumerate(subqueries, start=1):
        if sq.id != f"q{i}":
            problems.append(f"subquery {i} has id {sq.id!r}, expected q{i}")
        if not sq.text.strip():
            problems.append(f"{sq.id}: empty text")
        for dep in sq.depends_on:
            if dep not in seen:
                problems.append(f"{sq.id}: depends on {dep}, which is not an earlier subquery")
        for ph in sq.placeholders():
            if ph not in sq.depends_on:
                problems.append(f"{sq.id}: placeholder {{{ph}}} without a dependency")
        seen.append(sq.id)
    return problems


class QueryDecompositionAgent(RetrievalAgent[DecompositionInput, Decomposition]):
    name: ClassVar[str] = "query_decomposition"
    algorithm: ClassVar[int] = 4
    agent_id: ClassVar[str] = "A3"

    def heuristic(self, query: str) -> Decomposition:
        clauses = [c.strip(" ,?") for c in _SPLIT.split(query.strip()) if c and c.strip(" ,?")]
        if len(clauses) <= 1:
            return Decomposition(subqueries=[Subquery(id="q1", text=query.strip())])
        subs: list[Subquery] = []
        for i, clause in enumerate(clauses[: self.settings.decomposition.max_subqueries], 1):
            deps: list[str] = []
            text = clause[0].upper() + clause[1:]
            if i > 1 and _BACKREF.search(text):
                prev = f"q{i - 1}"
                deps = [prev]
                text = _BACKREF.sub(
                    lambda m, prev=prev: (
                        f"{{{prev}}}'s" if m.group(1).lower() in ("its", "their") else f"{{{prev}}}"
                    ),
                    text,
                )
            subs.append(Subquery(id=f"q{i}", text=text + "?", depends_on=deps))
        return Decomposition(subqueries=subs, complex=True)

    def _execute(self, data: DecompositionInput) -> Decomposition:
        draft = self.heuristic(data.query)
        max_n = self.settings.decomposition.max_subqueries
        if not self.settings.decomposition.use_llm:
            return draft
        try:
            out = self.llm_json(
                task="query_decomposition",
                system=SYSTEM,
                prompt=(
                    f"QUESTION: {data.query}\nQUERY TYPE: {data.query_type.value}\n"
                    f"DRAFT: {json.dumps([s.model_dump() for s in draft.subqueries])}"
                ),
                schema=LLMDecomposition,
            )
        except LLMError as exc:
            self.log.warning("decomposition_llm_failed", extra={"error": str(exc)[:200]})
            return draft
        if not out.subqueries:
            return draft
        proposal = [
            Subquery(
                id=s.id.strip(), text=s.text.strip(), depends_on=list(dict.fromkeys(s.depends_on))
            )
            for s in out.subqueries
        ]
        problems = validate_subqueries(proposal, max_n)
        if problems:
            self.flag(
                "LLM decomposition rejected; using rule-based subqueries",
                details={"problems": problems},
            )
            return draft
        return Decomposition(subqueries=proposal, complex=len(proposal) > 1, method="llm")
