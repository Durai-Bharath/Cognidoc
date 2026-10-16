"""A11 / Algorithm 12 - Faithfulness Validation Agent.

Compares every statement with the evidence it cites (line 2):

    support(s) = max over cited blocks of the share of s's content terms found in the block

A number in the statement that is missing from the evidence sets support to 0. A claim is
supported when support ≥ ``validation.claim_support_threshold``. With
``validation.use_llm`` an LLM judge must agree as well (logical AND), so a judge can only
make validation stricter.

Line 3 measures factual grounding and completeness:

    grounding    = supported claims / claims
    completeness = mean over the question (or each subquery) of the share of its terms
                   the answer addresses
    score        = w · grounding + (1 − w) · completeness      (w = grounding_weight)

Line 4 lists unsupported claims. Citation integrity is checked against the stores: every
cited chunk, node and edge id must resolve in the graph, and every marker must name a
context block. Each lookup is recorded, which is the proof the report carries.

Lines 5-9: Valid ⇔ score ≥ τ, citation integrity holds, the answer is not "insufficient
evidence", and it makes at least one claim. Individual unsupported claims are a *warning*
check: they lower the score, and the score decides.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.llm import LLMError
from common.validation import CheckResult, Severity, ValidationOutcome
from retrieval.context import RetrievalAgent
from retrieval.models import (
    CitationIntegrity,
    ClaimCheck,
    GroundedAnswer,
    IntegrityCheck,
    RetrievalContext,
    ValidationStatus,
)
from retrieval.text import coverage, is_insufficient, strip_markers, support_score

SYSTEM = (
    "You check whether each CLAIM is fully supported by the EVIDENCE quoted with it. "
    "Answer supported=false if any part of the claim is missing from or contradicted by "
    "its evidence."
)


class ValidationInput(BaseModel):
    question: str
    grounded: GroundedAnswer
    context: RetrievalContext
    targets: list[str] = Field(default_factory=list, description="Subquery texts (completeness).")


class LLMVerdict(BaseModel):
    statement_id: str = ""
    supported: bool = True
    reason: str = ""


class LLMJudgement(BaseModel):
    verdicts: list[LLMVerdict] = Field(default_factory=list)


class FaithfulnessValidationAgent(RetrievalAgent[ValidationInput, ValidationStatus]):
    name: ClassVar[str] = "faithfulness_validation"
    algorithm: ClassVar[int] = 12
    agent_id: ClassVar[str] = "A11"

    # ------------------------------------------------------------------ integrity
    def check_integrity(self, grounded: GroundedAnswer) -> CitationIntegrity:
        graph = self.ctx.graph
        out = CitationIntegrity(invalid_markers=list(dict.fromkeys(grounded.invalid_markers)))
        ids = grounded.referenced_ids()
        for cid in sorted(ids["chunk"]):
            node = graph.get_node(cid)
            out.checks.append(
                IntegrityCheck(
                    ref_id=cid,
                    kind="chunk",
                    store=f"graph:{graph.backend}",
                    resolved=node is not None and node.label == "Chunk",
                )
            )
        for nid in sorted(ids["node"] - ids["chunk"]):
            out.checks.append(
                IntegrityCheck(
                    ref_id=nid,
                    kind="node",
                    store=f"graph:{graph.backend}",
                    resolved=graph.get_node(nid) is not None,
                )
            )
        for eid in sorted(ids["edge"]):
            out.checks.append(
                IntegrityCheck(
                    ref_id=eid,
                    kind="edge",
                    store=f"graph:{graph.backend}",
                    resolved=graph.get_edge(eid) is not None,
                )
            )
        return out

    # ------------------------------------------------------------------ claims
    def check_claims(self, grounded: GroundedAnswer, ctx: RetrievalContext) -> list[ClaimCheck]:
        threshold = self.settings.validation.claim_support_threshold
        claims: list[ClaimCheck] = []
        for stmt in grounded.statements:
            if is_insufficient(stmt.text):
                continue
            texts = [b.text for cid in stmt.citation_ids if (b := ctx.block(cid)) is not None]
            if not texts:
                claims.append(
                    ClaimCheck(
                        statement_id=stmt.statement_id,
                        text=stmt.text,
                        support=0.0,
                        supported=False,
                        reason="no resolvable citation",
                    )
                )
                continue
            best, missing = max((support_score(stmt.text, t) for t in texts), key=lambda s: s[0])
            claims.append(
                ClaimCheck(
                    statement_id=stmt.statement_id,
                    text=stmt.text,
                    support=best,
                    supported=best >= threshold,
                    missing_terms=missing,
                    reason="" if best >= threshold else f"support {best} < {threshold}",
                )
            )
        if self.settings.validation.use_llm and claims:
            self._llm_judge(claims, grounded, ctx)
        return claims

    def _llm_judge(
        self, claims: list[ClaimCheck], grounded: GroundedAnswer, ctx: RetrievalContext
    ) -> None:
        by_id = {s.statement_id: s for s in grounded.statements}
        lines = []
        for c in claims:
            ev = " | ".join(
                b.text for cid in by_id[c.statement_id].citation_ids if (b := ctx.block(cid))
            )
            lines.append(f"{c.statement_id}: CLAIM: {c.text}\n    EVIDENCE: {ev or '-'}")
        try:
            out = self.llm_json(
                task="faithfulness_judge",
                system=SYSTEM,
                prompt="\n".join(lines),
                schema=LLMJudgement,
            )
        except LLMError as exc:
            self.log.warning("faithfulness_judge_failed", extra={"error": str(exc)[:200]})
            return
        verdicts = {v.statement_id: v for v in out.verdicts}
        for c in claims:
            v = verdicts.get(c.statement_id)
            if v is not None and not v.supported and c.supported:
                c.supported = False
                c.reason = f"LLM judge: {v.reason or 'unsupported'}"

    # ------------------------------------------------------------------ main
    def _execute(self, data: ValidationInput) -> ValidationStatus:
        cfg = self.settings.validation
        grounded = data.grounded
        claims = self.check_claims(grounded, data.context)
        integrity = self.check_integrity(grounded)
        insufficient = grounded.insufficient_evidence or is_insufficient(grounded.answer)
        grounding = sum(c.supported for c in claims) / len(claims) if claims else 0.0
        answer_text = strip_markers(grounded.answer)
        targets = data.targets or [data.question]
        completeness = sum(coverage(t, answer_text) for t in targets) / len(targets)
        if insufficient:
            completeness = 0.0
        score = round(
            cfg.grounding_weight * grounding + (1 - cfg.grounding_weight) * completeness, 4
        )

        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures("has_claims", [] if claims else ["answer makes no claim"])
        )
        outcome.add(
            CheckResult.from_failures(
                "claims_supported",
                [f"{c.statement_id}: {c.reason}" for c in claims if not c.supported],
                Severity.WARNING,
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "citation_integrity",
                [f"unresolved {r}" for r in integrity.unresolved]
                + [f"unknown marker {m}" for m in integrity.invalid_markers],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "sufficient_evidence",
                ["answer reports insufficient evidence"] if insufficient else [],
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "faithfulness_score", [] if score >= cfg.tau else [f"score {score} < τ {cfg.tau}"]
            )
        )
        valid = outcome.passed  # score ≥ τ, integrity, sufficient evidence, ≥ 1 claim
        return ValidationStatus(
            valid=valid,
            score=score,
            grounding=round(grounding, 4),
            completeness=round(completeness, 4),
            tau=cfg.tau,
            claims=claims,
            integrity=integrity,
            insufficient_evidence=insufficient,
            validation=outcome,
            reasons=outcome.failure_reasons(),
        )

    def confidence(self, output: ValidationStatus) -> float | None:
        return output.score
