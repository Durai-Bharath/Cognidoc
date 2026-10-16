"""Pass/fail validation checks shared by all agents.

Every "Validate ..." block in the pseudocode becomes a list of :class:`CheckResult`
objects. A check carries a severity: ``error`` checks decide whether validation
succeeds, ``warning`` checks are reported but never fail the outcome.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class CheckResult(BaseModel):
    name: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    severity: Severity = Severity.ERROR

    @classmethod
    def from_failures(
        cls, name: str, failures: Iterable[str], severity: Severity = Severity.ERROR
    ) -> CheckResult:
        reasons = list(failures)
        return cls(name=name, passed=not reasons, reasons=reasons, severity=severity)


class ValidationOutcome(BaseModel):
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == Severity.ERROR)

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == Severity.WARNING]

    @property
    def pass_ratio(self) -> float:
        """Fraction of checks that passed (warnings count); 1.0 when there are no checks."""
        if not self.checks:
            return 1.0
        return sum(c.passed for c in self.checks) / len(self.checks)

    def failure_reasons(self) -> list[str]:
        return [f"{c.name}: {r}" for c in self.errors for r in (c.reasons or ["failed"])]

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def get(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)
