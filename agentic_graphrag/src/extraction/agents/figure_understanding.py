"""Algorithm 11 - Figure Understanding Agent (a.k.a. "Figure Analysis Agent" in Alg. 9)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.llm import LLMError
from common.text import keywords
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.schemas import (
    ElementType,
    FigureConnector,
    FigureObject,
    FigureRepository,
    FigureType,
    LayoutElement,
    SectionTask,
    StructuredFigure,
)

_WORKFLOW_TYPES = {FigureType.FLOWCHART, FigureType.PIPELINE}


class FigureLLM(BaseModel):
    figure_type: FigureType = FigureType.MISC
    objects: list[FigureObject] = []
    labels: list[str] = []
    legends: list[str] = []
    annotations: list[str] = []
    connectors: list[FigureConnector] = []
    relationships: list[str] = []
    workflow: list[str] = []
    description: str = ""
    summary: str = ""
    keywords: list[str] = []


def validate_figure(fig: StructuredFigure) -> ValidationOutcome:
    outcome = ValidationOutcome()
    outcome.add(
        CheckResult.from_failures(
            "caption_association",
            [] if fig.caption else ["figure has no associated caption"],
            Severity.WARNING,
        )
    )
    legend_issues = [f"duplicate legend '{lg}'" for lg, n in Counter(fig.legends).items() if n > 1]
    legend_issues += ["empty legend entry"] if any(not lg.strip() for lg in fig.legends) else []
    outcome.add(CheckResult.from_failures("legend_consistency", legend_issues))

    completeness: list[str] = []
    ids = [o.id or o.label for o in fig.objects]
    if any(not (o.label or o.id).strip() for o in fig.objects):
        completeness.append("object without a label")
    dup_ids = [i for i, n in Counter(ids).items() if n > 1 and i]
    if dup_ids:
        completeness.append(f"duplicate object ids {dup_ids}")
    known = {o.id for o in fig.objects} | {o.label for o in fig.objects}
    for conn in fig.connectors:
        for end in (conn.source, conn.target):
            if end and end not in known:
                completeness.append(f"connector endpoint '{end}' is not a detected object")
    if not fig.description.strip():
        completeness.append("no figure description")
    outcome.add(CheckResult.from_failures("object_completeness", completeness))

    workflow: list[str] = []
    if fig.figure_type in _WORKFLOW_TYPES and not fig.workflow:
        workflow.append(f"{fig.figure_type.value} has no inferred workflow")
    if any(not step.strip() for step in fig.workflow):
        workflow.append("empty workflow step")
    if any(a == b for a, b in zip(fig.workflow, fig.workflow[1:], strict=False)):
        workflow.append("workflow repeats a step consecutively")
    outcome.add(CheckResult.from_failures("workflow_integrity", workflow))
    return outcome


class FigureUnderstandingAgent(BaseAgent[SectionTask, FigureRepository]):
    name = "FigureUnderstandingAgent"
    algorithm = 11

    def _execute(self, data: SectionTask) -> FigureRepository:
        section = data.section
        repo = FigureRepository(section_id=section.section_id)
        text = section.text()
        for idx, el in enumerate(section.elements_of(ElementType.FIGURE)):
            if el.attrs.get("continued_from"):
                continue  # second half of a multi-page figure: analysed with its first part
            repo.figures.append(self._analyse(el, section.section_id, idx, text))
        for fig in repo.figures:
            for check in fig.validation.checks:
                repo.validation.add(
                    check.model_copy(update={"name": f"{fig.figure_id}:{check.name}"})
                )
        return repo

    def _analyse(
        self, el: LayoutElement, section_id: str, idx: int, section_text: str
    ) -> StructuredFigure:
        number = el.attrs.get("number")
        caption = el.attrs.get("caption")
        image_path = el.attrs.get("image_path")
        fig = StructuredFigure(
            figure_id=f"{section_id}-fig{idx + 1}",
            element_id=el.element_id,
            number=str(number) if number else None,
            caption=str(caption) if caption else None,
            page=el.page,
            bbox=el.bbox,
            section_id=section_id,
            image_path=str(image_path) if image_path else None,
        )
        mentions = [
            p
            for p in section_text.split("\n\n")
            if number and (f"Figure {number}" in p or f"Fig. {number}" in p)
        ]
        images: list[bytes] = []
        if self.ctx.config.figure.use_vision and fig.image_path and Path(fig.image_path).exists():
            images = [Path(fig.image_path).read_bytes()]
        try:
            out = self.llm_json(
                task="figure_understanding",
                system="You analyse figures from technical documents. Identify the figure type, "
                "its components (objects, labels, legends, annotations, connectors), the "
                "relationships between components and any step-by-step workflow. Only "
                "report what is visible or stated in the caption/context.",
                prompt=(
                    f"Caption: {fig.caption or '(none)'}\n"
                    f"Context mentioning the figure:\n{chr(10).join(mentions)[:3000] or '(none)'}\n"
                    + (
                        ""
                        if images
                        else "(No image available: base the analysis on the "
                        "caption and context only.)"
                    )
                ),
                schema=FigureLLM,
                images=images,
            )
        except LLMError as exc:
            self.flag(f"figure analysis failed: {exc}", item_ref=fig.figure_id)
            out = FigureLLM()
        fig.figure_type = out.figure_type
        fig.objects = out.objects
        fig.labels = out.labels
        fig.legends = out.legends
        fig.annotations = out.annotations
        fig.connectors = out.connectors
        fig.relationships = out.relationships
        fig.workflow = out.workflow
        fig.description = out.description.strip() or (fig.caption or "")
        fig.summary = out.summary.strip() or fig.description
        fig.keywords = out.keywords or keywords(f"{fig.caption or ''} {fig.description}", 8)
        fig.validation = validate_figure(fig)
        fig.approved = fig.validation.passed
        if not fig.approved:
            self.flag(
                "figure failed validation; reprocessing recommended",
                item_ref=fig.figure_id,
                details={"failures": fig.validation.failure_reasons()},
            )
        return fig
