"""A9 / Algorithm 10 - Answer Generation Agent.

The PDF omits this algorithm's header; it is implemented from its body and Table 1:
receive the retrieval context (line 1), prompt the LLM (line 2), generate a grounded
answer (line 3) consistent with the evidence (line 4).

The system prompt requires an inline citation marker ``[E#]`` after every sentence, using
only the citation ids in the context, and the exact phrase "insufficient evidence" when
the context cannot answer the question. Nothing is guessed:

* an empty context never reaches the LLM; the answer is "Insufficient evidence …";
* an empty LLM answer, or one that says "insufficient evidence", sets
  ``insufficient_evidence``.

``mode="intermediate"`` answers one subquery that others depend on. It also asks for a
``short_answer`` (the bare entity or value) that fills ``{qN}`` placeholders.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from retrieval.context import RetrievalAgent
from retrieval.models import GeneratedAnswer, RetrievalContext
from retrieval.text import INSUFFICIENT, is_insufficient

SYSTEM = (
    "You answer questions strictly from the numbered CONTEXT blocks of a knowledge graph "
    "and its source documents. Rules: (1) Use only facts stated in the CONTEXT; never use "
    "outside knowledge. (2) After every sentence put the citation marker(s) of the block(s) "
    "that support it, e.g. [E2] or [E1][E3]; use only ids that appear in the CONTEXT. "
    "(3) If the CONTEXT does not contain the answer, reply exactly 'Insufficient evidence "
    "to answer this question.' and set insufficient_evidence to true. Do not guess. "
    "(4) Be concise: at most {max_sentences} sentences. When SUBQUESTIONS are listed, "
    "answer each of them. INTERMEDIATE NOTES are earlier sub-answers; they are not citable."
)
INSUFFICIENT_ANSWER = "Insufficient evidence to answer this question."


class AnswerInput(BaseModel):
    question: str
    context: RetrievalContext
    mode: Literal["final", "intermediate"] = "final"
    subquestions: list[str] = Field(default_factory=list, description="Resolved subqueries.")


class LLMAnswer(BaseModel):
    answer: str = ""
    insufficient_evidence: bool = False
    short_answer: str = ""


def build_prompt(data: AnswerInput) -> str:
    ctx = data.context
    notes = "".join(f"{k}: {v}\n" for k, v in ctx.sub_answers.items()) or "-\n"
    parts = " | ".join(data.subquestions) if len(data.subquestions) > 1 else "-"
    return (
        f"QUESTION: {data.question}\n"
        f"SUBQUESTIONS: {parts}\n"
        f"QUERY TYPE: {ctx.query_type.value}\n"
        f"MODE: {data.mode}\n"
        f"INTERMEDIATE NOTES:\n{notes}"
        f"CONTEXT:\n{ctx.render()}\n"
        "Answer the QUESTION with citation markers."
        + (
            " Also give short_answer: only the entity or value."
            if data.mode == "intermediate"
            else ""
        )
    )


class AnswerGenerationAgent(RetrievalAgent[AnswerInput, GeneratedAnswer]):
    name: ClassVar[str] = "answer_generation"
    algorithm: ClassVar[int] = 10
    agent_id: ClassVar[str] = "A9"

    def _execute(self, data: AnswerInput) -> GeneratedAnswer:
        if not data.context.blocks:
            return GeneratedAnswer(answer=INSUFFICIENT_ANSWER, insufficient_evidence=True)
        out = self.llm_json(
            task="answer_generation",
            system=SYSTEM.format(max_sentences=self.settings.answer.max_answer_sentences),
            prompt=build_prompt(data),
            schema=LLMAnswer,
        )
        answer = out.answer.strip()
        if not answer or out.insufficient_evidence or is_insufficient(answer):
            return GeneratedAnswer(answer=answer or INSUFFICIENT_ANSWER, insufficient_evidence=True)
        return GeneratedAnswer(answer=answer, short_answer=out.short_answer.strip())


__all__ = [
    "INSUFFICIENT",
    "INSUFFICIENT_ANSWER",
    "AnswerGenerationAgent",
    "AnswerInput",
    "LLMAnswer",
]
