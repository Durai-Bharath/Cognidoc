"""Algorithm 12 - Equation Analysis Agent."""

from __future__ import annotations

import re

from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.llm import LLMError
from common.validation import CheckResult, ValidationOutcome
from extraction.schemas import (
    ElementType,
    EquationRepository,
    EquationType,
    EquationVariable,
    LayoutElement,
    SectionTask,
    StructuredEquation,
)

UNICODE_TO_LATEX = {
    "∑": r"\sum ",
    "∏": r"\prod ",
    "∫": r"\int ",
    "√": r"\sqrt ",
    "∞": r"\infty ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≠": r"\neq ",
    "≈": r"\approx ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∀": r"\forall ",
    "∃": r"\exists ",
    "∂": r"\partial ",
    "∇": r"\nabla ",
    "×": r"\times ",
    "·": r"\cdot ",
    "±": r"\pm ",
    "→": r"\rightarrow ",
    "←": r"\leftarrow ",
    "↔": r"\leftrightarrow ",
    "−": "-",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\epsilon ",
    "θ": r"\theta ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "π": r"\pi ",
    "σ": r"\sigma ",
    "τ": r"\tau ",
    "φ": r"\phi ",
    "ω": r"\omega ",
    "Σ": r"\Sigma ",
    "Δ": r"\Delta ",
    "Ω": r"\Omega ",
    "Φ": r"\Phi ",
    "Ψ": r"\Psi ",
}
_NON_LATEX = set(UNICODE_TO_LATEX)


class EquationLLM(BaseModel):
    latex: str = ""
    equation_type: EquationType = EquationType.MISC
    variables: list[EquationVariable] = []
    constants: list[str] = []
    operators: list[str] = []
    functions: list[str] = []
    constraints: list[str] = []
    description: str = ""
    summary: str = ""


def text_to_latex(text: str) -> str:
    """Deterministic fallback conversion of linear math text to LaTeX."""
    flat = " ".join(text.split())
    flat = re.sub(r"\(\d+(?:\.\d+)?\)\s*$", "", flat).strip()
    out = "".join(UNICODE_TO_LATEX.get(ch, ch) for ch in flat)
    out = re.sub(r"([%#&$])", r"\\\1", out)
    return re.sub(r"\s+", " ", out).strip()


def _balanced(latex: str) -> list[str]:
    problems: list[str] = []
    stripped = re.sub(r"\\[{}()\[\]]", "", latex)
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    for ch in stripped:
        if ch in pairs:
            stack.append(ch)
        elif ch in pairs.values() and (not stack or pairs[stack.pop()] != ch):
            problems.append(f"unbalanced '{ch}'")
            return problems
    if stack:
        problems.append(f"unclosed '{stack[-1]}'")
    if latex.count(r"\left") != latex.count(r"\right"):
        problems.append(r"\left/\right mismatch")
    begins = re.findall(r"\\begin\{(\w+\*?)\}", latex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", latex)
    if begins != ends[::-1] and sorted(begins) != sorted(ends):
        problems.append("\\begin/\\end mismatch")
    return problems


def validate_equation(eq: StructuredEquation, doc_numbers: dict[str, int]) -> ValidationOutcome:
    outcome = ValidationOutcome()
    outcome.add(CheckResult.from_failures("syntax_correctness", _balanced(eq.latex)))
    plain = eq.latex.replace("\\", "")
    var_issues = [
        f"variable '{v.symbol}' not in LaTeX"
        for v in eq.variables
        if v.symbol and v.symbol.replace("\\", "").strip("$ ") not in plain
    ]
    outcome.add(CheckResult.from_failures("variable_consistency", var_issues))
    ref_issues: list[str] = []
    if eq.number and doc_numbers.get(eq.number, 0) > 1:
        ref_issues.append(f"equation number ({eq.number}) used {doc_numbers[eq.number]} times")
    outcome.add(CheckResult.from_failures("equation_reference_mapping", ref_issues))
    integrity: list[str] = []
    if not eq.latex.strip():
        integrity.append("empty LaTeX")
    if "$" in re.sub(r"\\\$", "", eq.latex):
        integrity.append("contains math-mode delimiters")
    leftover = sorted({ch for ch in eq.latex if ch in _NON_LATEX})
    if leftover:
        integrity.append(f"unconverted unicode symbols {leftover}")
    if re.search(r"(?<!\\)[%#]", eq.latex):
        integrity.append("unescaped % or #")
    outcome.add(CheckResult.from_failures("latex_integrity", integrity))
    return outcome


class EquationAnalysisAgent(BaseAgent[SectionTask, EquationRepository]):
    name = "EquationAnalysisAgent"
    algorithm = 12

    def _execute(self, data: SectionTask) -> EquationRepository:
        section = data.section
        repo = EquationRepository(section_id=section.section_id)
        numbers: dict[str, int] = {}
        for top in data.hierarchy.sections:
            for el in top.elements_of(ElementType.EQUATION):
                if el.attrs.get("number"):
                    numbers[str(el.attrs["number"])] = numbers.get(str(el.attrs["number"]), 0) + 1
        for idx, el in enumerate(section.elements_of(ElementType.EQUATION)):
            repo.equations.append(self._analyse(el, section.section_id, idx, numbers))
        for eq in repo.equations:
            for check in eq.validation.checks:
                repo.validation.add(
                    check.model_copy(update={"name": f"{eq.equation_id}:{check.name}"})
                )
        return repo

    def _analyse(
        self, el: LayoutElement, section_id: str, idx: int, numbers: dict[str, int]
    ) -> StructuredEquation:
        number = el.attrs.get("number")
        context = str(el.attrs.get("context", ""))
        try:
            out = self.llm_json(
                task="equation_analysis",
                system="You convert extracted equations to valid LaTeX (no $ delimiters, no "
                "equation number) and analyse them. Take variable definitions only from "
                "the surrounding text.",
                prompt=(
                    f"Equation text: {el.text}\nSurrounding text: {context or '(none)'}\n"
                    "Return LaTeX, equation type, variables with definitions, constants, "
                    "operators, functions, constraints, a semantic description and summary."
                ),
                schema=EquationLLM,
            )
        except LLMError as exc:
            self.flag(f"equation LLM analysis failed: {exc}", item_ref=f"{section_id}-eq{idx + 1}")
            out = EquationLLM()
        latex = out.latex.strip().strip("$").strip() or text_to_latex(el.text)
        eq = StructuredEquation(
            equation_id=f"{section_id}-eq{idx + 1}",
            element_id=el.element_id,
            number=str(number) if number else None,
            latex=latex,
            source_text=el.text,
            page=el.page,
            bbox=el.bbox,
            section_id=section_id,
            equation_type=out.equation_type,
            variables=out.variables,
            constants=out.constants,
            operators=out.operators,
            functions=out.functions,
            constraints=out.constraints,
            description=out.description.strip() or f"Equation {number or idx + 1}: {el.text}",
            summary=out.summary.strip() or out.description.strip(),
        )
        eq.validation = validate_equation(eq, numbers)
        eq.approved = eq.validation.passed
        if not eq.approved:
            self.flag(
                "equation failed validation; reprocessing recommended",
                item_ref=eq.equation_id,
                details={"failures": eq.validation.failure_reasons()},
            )
        return eq
