"""A12 / Algorithm 13 - Self-Critique Agent.

An independent reasoning pass over the grounded answer and its validation status
(lines 1-2). It runs when validation fails, and on every answer with
``critique.always_critique``.

Deterministic checks:

* every claim A11 found unsupported is a *major* ``unsupported`` issue (line 4);
* every unknown citation marker is a *major* ``citation`` issue;
* two statements about the same entity that share a relation term but state different
  numbers are a *major* ``contradiction`` (line 3, logical inconsistency).

With ``critique.use_llm`` the LLM re-reads the claims and their evidence and may add
issues and a corrected answer. LLM issues must name an existing statement (or none).

Corrections (line 5): the LLM's corrected answer if it gives one; otherwise the answer
without its unsupported statements, provided a supported statement remains. The graph
elements behind unsupported or contradictory statements become ``implicated`` ids, and
those behind supported ones ``supported`` ids, for the Feedback Agent (line 6).
"""

from __future__ import annotations

from itertools import combinations
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from common.llm import LLMError
from common.text import numbers_in
from retrieval.context import RetrievalAgent
from retrieval.models import (
    CritiqueIssue,
    GroundedAnswer,
    RetrievalContext,
    StatementCitation,
    ValidationStatus,
    VerifiedAnswer,
)
from retrieval.text import terms

SYSTEM = (
    "You are an independent reviewer of a question-answering system. Re-read each "
    "statement and the evidence it cites. Report logical inconsistencies, contradictions "
    "and claims the evidence does not support, with a correction for each. Give "
    "verdict 'verified' only if every statement is supported and consistent. If needed, "
    "write corrected_answer using only supported facts and the same [E#] markers."
)


class CritiqueInput(BaseModel):
    question: str
    grounded: GroundedAnswer
    status: ValidationStatus
    context: RetrievalContext


class LLMIssue(BaseModel):
    statement_id: str = ""
    issue_type: Literal["unsupported", "contradiction", "logical", "citation"] = "logical"
    severity: Literal["major", "minor"] = "minor"
    description: str = ""
    correction: str = ""


class LLMCritique(BaseModel):
    verdict: str = ""
    issues: list[LLMIssue] = Field(default_factory=list)
    corrected_answer: str = ""


def _refs(stmt: StatementCitation) -> tuple[set[str], set[str]]:
    nodes: set[str] = set(stmt.entity_ids)
    edges: set[str] = set()
    for src in stmt.sources:
        if src.kind != "chunk":  # chunks are source text, not graph knowledge
            nodes.update(src.node_ids)
        edges.update(src.edge_ids)
    return nodes, edges


class SelfCritiqueAgent(RetrievalAgent[CritiqueInput, VerifiedAnswer]):
    name: ClassVar[str] = "self_critique"
    algorithm: ClassVar[int] = 13
    agent_id: ClassVar[str] = "A12"

    @staticmethod
    def contradictions(statements: list[StatementCitation]) -> list[CritiqueIssue]:
        issues: list[CritiqueIssue] = []
        for a, b in combinations(statements, 2):
            shared = set(a.entity_ids) & set(b.entity_ids)
            na, nb = numbers_in(a.text), numbers_in(b.text)
            if not shared or not na or not nb or na == nb:
                continue
            words_a = {t for t in terms(a.text) if not t[0].isdigit()}
            words_b = {t for t in terms(b.text) if not t[0].isdigit()}
            if len(words_a & words_b) >= 2:  # same entity + a shared relation term
                issues.append(
                    CritiqueIssue(
                        statement_id=b.statement_id,
                        issue_type="contradiction",
                        severity="major",
                        description=(
                            f"{a.statement_id} and {b.statement_id} give different values "
                            f"({sorted(na)} vs {sorted(nb)}) for the same entity"
                        ),
                        correction="keep only the value the evidence supports",
                    )
                )
        return issues

    def _execute(self, data: CritiqueInput) -> VerifiedAnswer:
        grounded, status = data.grounded, data.status
        stmts = {s.statement_id: s for s in grounded.statements}
        issues: list[CritiqueIssue] = []
        for claim in status.claims:
            if not claim.supported:
                issues.append(
                    CritiqueIssue(
                        statement_id=claim.statement_id,
                        issue_type="unsupported",
                        severity="major",
                        description=claim.reason or "claim not supported by its evidence",
                        correction="remove the claim or retrieve supporting evidence",
                    )
                )
        for marker in dict.fromkeys(grounded.invalid_markers):
            issues.append(
                CritiqueIssue(
                    issue_type="citation",
                    severity="major",
                    description=f"citation marker [{marker}] names no context block",
                    correction="cite an existing block",
                )
            )
        issues += self.contradictions(grounded.statements)

        corrected: str | None = None
        used_llm = False
        if self.settings.critique.use_llm and grounded.statements:
            corrected, extra = self._llm_pass(data)
            used_llm = extra is not None
            issues += extra or []

        bad = {i.statement_id for i in issues if i.severity == "major" and i.statement_id}
        out = VerifiedAnswer(
            verdict="needs_revision" if any(i.severity == "major" for i in issues) else "verified",
            issues=issues,
            used_llm=used_llm,
        )
        for sid, stmt in stmts.items():
            nodes, edges = _refs(stmt)
            if sid in bad:
                out.implicated_node_ids += sorted(nodes)
                out.implicated_edge_ids += sorted(edges)
            else:
                out.supported_node_ids += sorted(nodes)
                out.supported_edge_ids += sorted(edges)
        for field in (
            "implicated_node_ids",
            "implicated_edge_ids",
            "supported_node_ids",
            "supported_edge_ids",
        ):
            setattr(out, field, list(dict.fromkeys(getattr(out, field))))
        if corrected is None and bad:
            keep = [s for sid, s in stmts.items() if sid not in bad]
            if keep:
                corrected = " ".join(
                    s.text.rstrip(".") + "".join(f" [{c}]" for c in s.citation_ids) + "."
                    for s in keep
                )
        out.corrected_answer = corrected
        return out

    def _llm_pass(self, data: CritiqueInput) -> tuple[str | None, list[CritiqueIssue] | None]:
        ctx = data.context
        lines = []
        for s in data.grounded.statements:
            ev = " | ".join(f"[{c}] {b.text}" for c in s.citation_ids if (b := ctx.block(c)))
            lines.append(f"{s.statement_id}: {s.text}\n    EVIDENCE: {ev or '-'}")
        prompt = (
            f"QUESTION: {data.question}\nANSWER: {data.grounded.answer}\n"
            f"VALIDATION: valid={data.status.valid} score={data.status.score} "
            f"unsupported={data.status.unsupported}\nSTATEMENTS:\n" + "\n".join(lines)
        )
        try:
            out = self.llm_json(
                task="self_critique", system=SYSTEM, prompt=prompt, schema=LLMCritique
            )
        except LLMError as exc:
            self.log.warning("self_critique_llm_failed", extra={"error": str(exc)[:200]})
            return None, None
        known = {s.statement_id for s in data.grounded.statements}
        issues = [
            CritiqueIssue(
                statement_id=i.statement_id or None,
                issue_type=i.issue_type,
                severity=i.severity,
                description=i.description,
                correction=i.correction,
            )
            for i in out.issues
            if not i.statement_id or i.statement_id in known
        ]
        return (out.corrected_answer.strip() or None), issues

    def confidence(self, output: VerifiedAnswer) -> float | None:
        return 1.0 if output.verdict == "verified" else 0.0
