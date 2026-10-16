"""Algorithm 1 - Overall Agentic Document Extraction Framework, as a LangGraph state machine.

Graph nodes map 1:1 to Algorithm 1 lines:

    router(2) → classification(3) → [requires OCR?] → ocr_quality(4-5)
        → [Q < τ?] → image_enhancement(6-7) → (reclassify) → layout(10)
    layout(10) → reading_order(11) → section_identification(12)
    → next_section(13) → context_memory(14) → [table?](15-16) → [figure?](17-18)
    → [equation?](19-20) → citation(21) → cross_reference(22) → entity(23)
    → abbreviation(24) → summarization(25) → hallucination(26) → confidence(27)
    → [more sections?] → next_section | metadata_fusion(28) → global_validation(29)
    → export(30-34)
"""

from __future__ import annotations

import json
from collections.abc import Hashable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.base_agent import AgentContext, AgentError, AgentResult, TraceRecord, TraceSink
from common.config import ExtractionConfig, load_config
from common.embedder import Embedder, build_embedder
from common.llm import LLMClient, build_llm
from common.logging import configure_logging, get_logger
from common.review_queue import ReviewQueue
from extraction.agents import (
    AbbreviationResolutionAgent,
    ConfidenceScoringAgent,
    ContextMemoryAgent,
    CrossReferenceValidationAgent,
    DocumentClassificationAgent,
    EntityExtractionAgent,
    EquationAnalysisAgent,
    FigureUnderstandingAgent,
    GlobalValidationAgent,
    HallucinationDetectionAgent,
    ImageEnhancementAgent,
    InternalCitationExtractionAgent,
    LayoutDetectionAgent,
    MetadataFusionAgent,
    OCRQualityAssessmentAgent,
    ReadingOrderAgent,
    RouterAgent,
    SectionIdentificationAgent,
    TableAnalysisAgent,
    TextSummarizationAgent,
)
from extraction.agents.global_validation import GlobalValidationInput
from extraction.agents.metadata_fusion import FusionOutput
from extraction.agents.table_analysis import to_markdown
from extraction.ocr import OCREngine
from extraction.output_models import (
    DocumentJSON,
    MetadataFile,
    ValidationReport,
    ValidationStatistics,
)
from extraction.schemas import (
    AbbreviationInput,
    AbbreviationRepository,
    CitationRepository,
    ClassificationInput,
    ClassificationResult,
    ConfidenceInput,
    CrossReference,
    CrossRefInput,
    DocumentFeatures,
    ElementType,
    EnhancementInput,
    EnhancementReport,
    FusionInput,
    GeneratedItem,
    HallucinationInput,
    HallucinationReport,
    LayoutInput,
    LayoutMap,
    OCRQualityInput,
    OCRQualityReport,
    OrderedDocument,
    ParsedDocument,
    ReadingOrderInput,
    RouterInput,
    SectionHierarchy,
    SectionIdentificationInput,
    SectionResults,
    SectionTask,
    SummarizationInput,
)

_log = get_logger(__name__)


class ExtractionState(TypedDict, total=False):
    input_path: str
    output_dir: str
    parsed: ParsedDocument
    classification: ClassificationResult
    quality: OCRQualityReport | None
    enhancement: EnhancementReport | None
    layout: LayoutMap
    ordered: OrderedDocument
    hierarchy: SectionHierarchy
    section_ids: list[str]
    section_index: int
    results: dict[str, SectionResults]
    abbreviations: AbbreviationRepository | None
    prior_references: list[CrossReference]
    fusion: FusionOutput
    report: ValidationReport
    agent_errors: list[str]
    fatal: AgentError | None
    output_paths: dict[str, str]


class PipelineResult(BaseModel):
    document_id: str | None
    metadata: MetadataFile | None = None
    document: DocumentJSON | None = None
    validation_report: ValidationReport | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    error: AgentError | None = None
    agent_errors: list[str] = Field(default_factory=list)
    traces: list[TraceRecord] = Field(default_factory=list)
    review_items: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.metadata is not None


class AgenticDocumentExtractionFramework:
    """Algorithm 1. Builds the shared context, the 20 agents and the LangGraph graph."""

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        ocr: OCREngine | None = None,
        review_queue: ReviewQueue | None = None,
        services: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self.ctx = AgentContext(
            config=self.config,
            llm=llm or build_llm(self.config.llm),
            embedder=embedder or build_embedder(self.config.embedder),
            review_queue=review_queue or ReviewQueue(self.config.paths.review_queue),
            services=dict(services or {}),
        )
        if ocr is not None:
            self.ctx.services["ocr"] = ocr
        c = self.ctx
        self.router = RouterAgent(c)
        self.classification = DocumentClassificationAgent(c)
        self.ocr_quality = OCRQualityAssessmentAgent(c)
        self.enhancement = ImageEnhancementAgent(c)
        self.layout = LayoutDetectionAgent(c)
        self.reading_order = ReadingOrderAgent(c)
        self.sections = SectionIdentificationAgent(c)
        self.memory = ContextMemoryAgent(c)
        self.tables = TableAnalysisAgent(c)
        self.figures = FigureUnderstandingAgent(c)
        self.equations = EquationAnalysisAgent(c)
        self.citations = InternalCitationExtractionAgent(c)
        self.cross_refs = CrossReferenceValidationAgent(c)
        self.entities = EntityExtractionAgent(c)
        self.abbreviations = AbbreviationResolutionAgent(c)
        self.summarizer = TextSummarizationAgent(c)
        self.hallucination = HallucinationDetectionAgent(c)
        self.confidence = ConfidenceScoringAgent(c)
        self.fusion = MetadataFusionAgent(c)
        self.global_validation = GlobalValidationAgent(c)
        self.graph = self._build_graph()
        self.last_state: ExtractionState = {}

    # ================================================================== public API
    def run(self, path: str | Path, output_dir: str | Path | None = None) -> PipelineResult:
        self.ctx.traces = TraceSink()
        self.ctx.document_id = None
        flags_before = len(self.ctx.review_queue.items)
        out = (
            Path(output_dir) if output_dir else Path(self.config.paths.output_dir) / Path(path).stem
        )
        state: ExtractionState = {
            "input_path": str(path),
            "output_dir": str(out),
            "results": {},
            "agent_errors": [],
            "prior_references": [],
            "abbreviations": None,
            "quality": None,
            "enhancement": None,
            "fatal": None,
            "output_paths": {},
        }
        final: ExtractionState = self.graph.invoke(state, config={"recursion_limit": 10_000})
        self.last_state = final
        fusion = final.get("fusion")
        return PipelineResult(
            document_id=self.ctx.document_id,
            metadata=fusion.metadata if fusion else None,
            document=fusion.document if fusion else None,
            validation_report=final.get("report"),
            output_paths=final.get("output_paths", {}),
            error=final.get("fatal"),
            agent_errors=final.get("agent_errors", []),
            traces=self.ctx.traces.records,
            review_items=len(self.ctx.review_queue.items) - flags_before,
        )

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())

    # ================================================================== graph
    def _build_graph(self) -> Any:
        g: StateGraph[ExtractionState] = StateGraph(ExtractionState)
        nodes = {
            "router": self._n_router,
            "classification": self._n_classification,
            "ocr_quality": self._n_ocr_quality,
            "image_enhancement": self._n_enhancement,
            "reclassify": self._n_classification,
            "layout_detection": self._n_layout,
            "reading_order": self._n_reading_order,
            "section_identification": self._n_sections,
            "next_section": self._n_next_section,
            "context_memory": self._n_memory,
            "table_analysis": self._n_tables,
            "figure_understanding": self._n_figures,
            "equation_analysis": self._n_equations,
            "citation_extraction": self._n_citations,
            "cross_reference_validation": self._n_cross_refs,
            "entity_extraction": self._n_entities,
            "abbreviation_resolution": self._n_abbreviations,
            "text_summarization": self._n_summarization,
            "hallucination_detection": self._n_hallucination,
            "confidence_scoring": self._n_confidence,
            "metadata_fusion": self._n_fusion,
            "global_validation": self._n_global_validation,
            "export": self._n_export,
        }
        for name, fn in nodes.items():
            g.add_node(name, fn)

        g.add_edge(START, "router")
        g.add_conditional_edges(
            "router",
            self._ok_or_fail("classification"),
            {"classification": "classification", "export": "export"},
        )
        # Alg. 1 lines 4-9: OCR branch.
        g.add_conditional_edges(
            "classification",
            self._route_requires_ocr,
            {"ocr_quality": "ocr_quality", "layout_detection": "layout_detection"},
        )
        g.add_conditional_edges(
            "ocr_quality",
            self._route_quality,
            {
                "image_enhancement": "image_enhancement",
                "reclassify": "reclassify",
                "layout_detection": "layout_detection",
                "export": "export",
            },
        )
        g.add_conditional_edges(
            "image_enhancement",
            self._route_after_enhancement,
            {"reclassify": "reclassify", "layout_detection": "layout_detection"},
        )
        g.add_edge("reclassify", "layout_detection")
        # Lines 10-12.
        g.add_conditional_edges(
            "layout_detection",
            self._ok_or_fail("reading_order"),
            {"reading_order": "reading_order", "export": "export"},
        )
        g.add_conditional_edges(
            "reading_order",
            self._ok_or_fail("section_identification"),
            {"section_identification": "section_identification", "export": "export"},
        )
        g.add_conditional_edges(
            "section_identification",
            self._ok_or_fail("next_section"),
            {"next_section": "next_section", "export": "export"},
        )
        # Lines 13-27: per-section loop with conditional table/figure/equation agents.
        g.add_conditional_edges(
            "next_section",
            self._route_has_section,
            {"context_memory": "context_memory", "metadata_fusion": "metadata_fusion"},
        )
        chain = [
            "context_memory",
            "table_analysis",
            "figure_understanding",
            "equation_analysis",
            "citation_extraction",
        ]
        for i, node in enumerate(chain[:3]):
            targets: dict[Hashable, str] = {n: n for n in chain[i + 1 :]}
            g.add_conditional_edges(node, self._route_objects(node), targets)
        g.add_edge("equation_analysis", "citation_extraction")
        g.add_edge("citation_extraction", "cross_reference_validation")
        g.add_edge("cross_reference_validation", "entity_extraction")
        g.add_edge("entity_extraction", "abbreviation_resolution")
        g.add_edge("abbreviation_resolution", "text_summarization")
        g.add_edge("text_summarization", "hallucination_detection")
        g.add_edge("hallucination_detection", "confidence_scoring")
        g.add_edge("confidence_scoring", "next_section")
        # Lines 28-34.
        g.add_conditional_edges(
            "metadata_fusion",
            self._ok_or_fail("global_validation"),
            {"global_validation": "global_validation", "export": "export"},
        )
        g.add_edge("global_validation", "export")
        g.add_edge("export", END)
        return g.compile()

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _ok_or_fail(next_node: str) -> Any:
        def route(state: ExtractionState) -> str:
            return "export" if state.get("fatal") else next_node

        return route

    @staticmethod
    def _route_requires_ocr(state: ExtractionState) -> Literal["ocr_quality", "layout_detection"]:
        return "ocr_quality" if state["parsed"].requires_ocr else "layout_detection"

    @staticmethod
    def _route_quality(state: ExtractionState) -> str:
        if state.get("fatal"):
            return "export"
        quality = state.get("quality")
        if quality is not None and quality.low_quality_pages:  # OCR quality < τ
            return "image_enhancement"
        if not state["classification"].based_on_text:
            return "reclassify"
        return "layout_detection"

    @staticmethod
    def _route_after_enhancement(state: ExtractionState) -> str:
        return "layout_detection" if state["classification"].based_on_text else "reclassify"

    @staticmethod
    def _route_has_section(state: ExtractionState) -> str:
        if state["section_index"] < len(state["section_ids"]):
            return "context_memory"
        return "metadata_fusion"

    def _route_objects(self, after: str) -> Any:
        order = ["context_memory", "table_analysis", "figure_understanding", "equation_analysis"]
        kinds = {
            "table_analysis": ElementType.TABLE,
            "figure_understanding": ElementType.FIGURE,
            "equation_analysis": ElementType.EQUATION,
        }

        def route(state: ExtractionState) -> str:
            section = self._current_task(state).section
            for node in order[order.index(after) + 1 :]:
                if section.elements_of(kinds[node]):
                    return node
            return "citation_extraction"

        return route

    # ------------------------------------------------------------------ helpers
    def _current_task(self, state: ExtractionState) -> SectionTask:
        hierarchy = state["hierarchy"]
        idx = state["section_index"]
        sid = state["section_ids"][idx]
        section = next(s for s in hierarchy.sections if s.section_id == sid)
        return SectionTask(
            section=section,
            hierarchy=hierarchy,
            ordered=state["ordered"],
            page_count=state["parsed"].properties.page_count,
            is_last_section=idx == len(state["section_ids"]) - 1,
        )

    def _results(self, state: ExtractionState) -> tuple[dict[str, SectionResults], SectionResults]:
        sid = state["section_ids"][state["section_index"]]
        results = dict(state["results"])
        current = results.get(sid) or SectionResults(section_id=sid)
        results[sid] = current
        return results, current

    @staticmethod
    def _fatal(result: AgentResult[Any]) -> dict[str, Any]:
        return {"fatal": result.error}

    @staticmethod
    def _error_note(state: ExtractionState, result: AgentResult[Any]) -> list[str]:
        assert result.error is not None
        return [
            *state.get("agent_errors", []),
            f"{result.error.agent}: {result.error.error_type}: {result.error.message}",
        ]

    def _section_ctx(self, state: ExtractionState) -> dict[str, str]:
        return {"section": state["section_ids"][state["section_index"]]}

    # ------------------------------------------------------------------ nodes: lines 2-12
    def _n_router(self, state: ExtractionState) -> dict[str, Any]:
        workdir = str(Path(state["output_dir"]) / "work")
        result = self.router.run(RouterInput(path=state["input_path"], workdir=workdir))
        if not result.ok:
            return self._fatal(result)
        return {"parsed": result.unwrap().parsed}

    def _n_classification(self, state: ExtractionState) -> dict[str, Any]:
        result = self.classification.run(ClassificationInput(parsed=state["parsed"]))
        if not result.ok:
            fallback = ClassificationResult(
                category="Unknown",
                predicted_category="Unknown",
                confidence=0.0,
                scores={},
                features=DocumentFeatures(),
                flagged=True,
            )
            return {"classification": fallback, "agent_errors": self._error_note(state, result)}
        return {"classification": result.unwrap()}

    def _n_ocr_quality(self, state: ExtractionState) -> dict[str, Any]:
        result = self.ocr_quality.run(OCRQualityInput(parsed=state["parsed"]))
        if not result.ok:
            return self._fatal(result)
        report = result.unwrap()
        return {"quality": report, "parsed": report.parsed}

    def _n_enhancement(self, state: ExtractionState) -> dict[str, Any]:
        quality = state["quality"]
        assert quality is not None
        result = self.enhancement.run(EnhancementInput(parsed=state["parsed"], quality=quality))
        if not result.ok:
            return {"agent_errors": self._error_note(state, result)}
        report = result.unwrap()
        return {"enhancement": report, "parsed": report.parsed}

    def _n_layout(self, state: ExtractionState) -> dict[str, Any]:
        result = self.layout.run(LayoutInput(parsed=state["parsed"]))
        return self._fatal(result) if not result.ok else {"layout": result.unwrap()}

    def _n_reading_order(self, state: ExtractionState) -> dict[str, Any]:
        result = self.reading_order.run(ReadingOrderInput(layout=state["layout"]))
        return self._fatal(result) if not result.ok else {"ordered": result.unwrap()}

    def _n_sections(self, state: ExtractionState) -> dict[str, Any]:
        result = self.sections.run(
            SectionIdentificationInput(
                ordered=state["ordered"], page_count=state["parsed"].properties.page_count
            )
        )
        if not result.ok:
            return self._fatal(result)
        hierarchy = result.unwrap()
        return {
            "hierarchy": hierarchy,
            "section_ids": [s.section_id for s in hierarchy.sections],
            "section_index": -1,
        }

    # ------------------------------------------------------------------ nodes: loop 13-27
    def _n_next_section(self, state: ExtractionState) -> dict[str, Any]:
        return {"section_index": state["section_index"] + 1}

    def _run_section_agent(
        self, state: ExtractionState, agent: Any, payload: BaseModel, field: str
    ) -> dict[str, Any]:
        result = agent.run(payload, trace_context=self._section_ctx(state))
        results, current = self._results(state)
        if not result.ok:
            return {"results": results, "agent_errors": self._error_note(state, result)}
        results[current.section_id] = current.model_copy(update={field: result.unwrap()})
        return {"results": results}

    def _n_memory(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(state, self.memory, self._current_task(state), "memory")

    def _n_tables(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(state, self.tables, self._current_task(state), "tables")

    def _n_figures(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(state, self.figures, self._current_task(state), "figures")

    def _n_equations(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(
            state, self.equations, self._current_task(state), "equations"
        )

    def _n_citations(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(
            state, self.citations, self._current_task(state), "citations"
        )

    def _n_cross_refs(self, state: ExtractionState) -> dict[str, Any]:
        _, current = self._results(state)
        task = self._current_task(state)
        citations = current.citations or CitationRepository(section_id=task.section.section_id)
        update = self._run_section_agent(
            state,
            self.cross_refs,
            CrossRefInput(
                task=task, citations=citations, prior_references=state["prior_references"]
            ),
            "cross_references",
        )
        repo = update["results"][task.section.section_id].cross_references
        if repo is not None:
            update["prior_references"] = [*state["prior_references"], *repo.references]
        return update

    def _n_entities(self, state: ExtractionState) -> dict[str, Any]:
        return self._run_section_agent(state, self.entities, self._current_task(state), "entities")

    def _n_abbreviations(self, state: ExtractionState) -> dict[str, Any]:
        abbr_repo = state.get("abbreviations")
        existing = abbr_repo.abbreviations if abbr_repo else {}
        result = self.abbreviations.run(
            AbbreviationInput(task=self._current_task(state), existing=existing),
            trace_context=self._section_ctx(state),
        )
        if not result.ok:
            return {"agent_errors": self._error_note(state, result)}
        return {"abbreviations": result.unwrap()}

    def _n_summarization(self, state: ExtractionState) -> dict[str, Any]:
        _, current = self._results(state)
        return self._run_section_agent(
            state,
            self.summarizer,
            SummarizationInput(task=self._current_task(state), memory=current.memory),
            "summary",
        )

    def _generated_items(self, state: ExtractionState) -> list[GeneratedItem]:
        task = self._current_task(state)
        _, res = self._results(state)
        sid = task.section.section_id
        text = task.section.text()
        items: list[GeneratedItem] = []
        if res.summary:
            items.append(
                GeneratedItem(
                    content_id=f"{sid}:summary",
                    kind="summary",
                    text=res.summary.summary,
                    source=text,
                )
            )
            items.append(
                GeneratedItem(
                    content_id=f"{sid}:abstract",
                    kind="abstract",
                    text=res.summary.abstract,
                    source=text,
                )
            )
        for t in res.tables.tables if res.tables else []:
            src = f"{t.caption or ''}\n\n{to_markdown(t.headers, t.rows)}"
            items.append(
                GeneratedItem(
                    content_id=f"{t.table_id}:description",
                    kind="table",
                    text=t.description,
                    source=src,
                )
            )
        for f in res.figures.figures if res.figures else []:
            items.append(
                GeneratedItem(
                    content_id=f"{f.figure_id}:description",
                    kind="figure",
                    text=f.description,
                    source=f"{f.caption or ''}\n\n{text}",
                )
            )
        for e in res.equations.equations if res.equations else []:
            items.append(
                GeneratedItem(
                    content_id=f"{e.equation_id}:description",
                    kind="equation",
                    text=e.description,
                    source=f"{e.source_text}\n\n{text}",
                )
            )
        for ent in res.entities.entities if res.entities else []:
            items.append(
                GeneratedItem(
                    content_id=f"{ent.entity_id}:description",
                    kind="entity",
                    text=ent.description,
                    source=ent.context or text,
                )
            )
        return items

    def _n_hallucination(self, state: ExtractionState) -> dict[str, Any]:
        sid = state["section_ids"][state["section_index"]]
        update = self._run_section_agent(
            state,
            self.hallucination,
            HallucinationInput(section_id=sid, items=self._generated_items(state)),
            "hallucination",
        )
        res: SectionResults = update["results"][sid]
        if res.hallucination is not None:
            update["results"][sid] = apply_replacements(res, res.hallucination)
        return update

    def _n_confidence(self, state: ExtractionState) -> dict[str, Any]:
        task = self._current_task(state)
        _, res = self._results(state)
        prior = [
            o
            for r in state["results"].values()
            if r.confidence and r.section_id != task.section.section_id
            for o in r.confidence.objects
            if o.kind == "section"
        ]
        payload = component_scores(state, task, res)
        payload.prior_sections = prior
        return self._run_section_agent(state, self.confidence, payload, "confidence")

    # ------------------------------------------------------------------ nodes: 28-34
    def _n_fusion(self, state: ExtractionState) -> dict[str, Any]:
        result = self.fusion.run(
            FusionInput(
                parsed=state["parsed"],
                classification=state["classification"],
                quality=state.get("quality"),
                enhancement=state.get("enhancement"),
                layout=state["layout"],
                ordered=state["ordered"],
                hierarchy=state["hierarchy"],
                results=state["results"],
                abbreviations=state.get("abbreviations"),
                review_flags=len(self.ctx.review_queue.items),
            )
        )
        return self._fatal(result) if not result.ok else {"fusion": result.unwrap()}

    def _n_global_validation(self, state: ExtractionState) -> dict[str, Any]:
        fusion = state["fusion"]
        result = self.global_validation.run(
            GlobalValidationInput(
                metadata=fusion.metadata,
                document=fusion.document,
                agent_errors=state.get("agent_errors", []),
            )
        )
        if not result.ok:
            return {"agent_errors": self._error_note(state, result)}
        return {"report": result.unwrap()}

    def _n_export(self, state: ExtractionState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        fusion = state.get("fusion")
        report = state.get("report")
        if report is None:
            fatal = state.get("fatal")
            errors = list(state.get("agent_errors", []))
            if fatal:
                errors.insert(0, f"{fatal.agent}: {fatal.error_type}: {fatal.message}")
            report = ValidationReport(
                document_id=self.ctx.document_id or "unknown",
                generated_at=datetime.now(UTC).isoformat(),
                approved=False,
                quality_score=0.0,
                categories=[],
                statistics=ValidationStatistics(
                    validation_accuracy=0.0,
                    total_errors=len(errors),
                    warnings=0,
                    passed_checks=0,
                    total_checks=0,
                ),
                recommendations=["Fix the fatal error and re-run the pipeline"],
                agent_errors=errors,
            )
        if fusion is not None:
            paths["metadata"] = _write(
                out / "metadata.json", fusion.metadata.model_dump(mode="json")
            )
            paths["document"] = _write(
                out / "document.json", fusion.document.model_dump(mode="json")
            )
        paths["validation_report"] = _write(
            out / "validation_report.json", report.model_dump(mode="json")
        )
        return {"report": report, "output_paths": paths}


# ====================================================================== module helpers


def _write(path: Path, data: dict[str, Any]) -> str:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def apply_replacements(res: SectionResults, report: HallucinationReport) -> SectionResults:
    """Swap regenerated (verified) text back into the agent outputs."""
    repl = report.replacements
    if not repl:
        return res
    res = res.model_copy(deep=True)
    sid = res.section_id
    if res.summary:
        res.summary.summary = repl.get(f"{sid}:summary", res.summary.summary)
        res.summary.abstract = repl.get(f"{sid}:abstract", res.summary.abstract)
    for t in res.tables.tables if res.tables else []:
        t.description = repl.get(f"{t.table_id}:description", t.description)
    for f in res.figures.figures if res.figures else []:
        f.description = repl.get(f"{f.figure_id}:description", f.description)
    for e in res.equations.equations if res.equations else []:
        e.description = repl.get(f"{e.equation_id}:description", e.description)
    for ent in res.entities.entities if res.entities else []:
        ent.description = repl.get(f"{ent.entity_id}:description", ent.description)
    return res


def component_scores(
    state: ExtractionState, task: SectionTask, res: SectionResults
) -> ConfidenceInput:
    """Collect c_k from every agent (Alg. 19 lines 4-13) for the section and its objects."""
    section = task.section
    quality = state.get("quality")
    ocr = quality.document_score if quality else None
    pages = [p for p in state["layout"].pages if section.page_start <= p.page <= section.page_end]
    layout = (sum(p.validation.pass_ratio for p in pages) / len(pages)) if pages else None
    hall = res.hallucination.items if res.hallucination else []

    def hall_for(prefix: str) -> float | None:
        vals = [1 - min(1.0, i.hscore) for i in hall if i.content_id.startswith(prefix)]
        return sum(vals) / len(vals) if vals else None

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    tables = res.tables.tables if res.tables else []
    figures = res.figures.figures if res.figures else []
    equations = res.equations.equations if res.equations else []
    abbr = state.get("abbreviations")
    sid = section.section_id
    scores: dict[str, dict[str, float | None]] = {
        sid: {
            "ocr_quality": ocr,
            "layout": layout,
            "section_identification": task.hierarchy.validation.pass_ratio,
            "table": mean([t.validation.pass_ratio for t in tables]),
            "figure": mean([f.validation.pass_ratio for f in figures]),
            "equation": mean([e.validation.pass_ratio for e in equations]),
            "entity": res.entities.validation.pass_ratio if res.entities else None,
            "summarization": res.summary.confidence if res.summary else None,
            "hallucination": hall_for(f"{sid}:"),
            "classification": state["classification"].confidence,
            "abbreviation": abbr.validation.pass_ratio if abbr else None,
        }
    }
    kinds: dict[str, Literal["section", "table", "figure", "equation"]] = {sid: "section"}
    for kind, objs, key in (
        ("table", tables, "table_id"),
        ("figure", figures, "figure_id"),
        ("equation", equations, "equation_id"),
    ):
        for obj in objs:
            oid = getattr(obj, key)
            scores[oid] = {
                "ocr_quality": ocr,
                "layout": layout,
                kind: obj.validation.pass_ratio,
                "hallucination": hall_for(f"{oid}:"),
            }
            kinds[oid] = kind  # type: ignore[assignment]
    return ConfidenceInput(
        section_id=sid,
        component_scores=scores,
        object_kinds=kinds,
        section_weight=float(res.memory.token_count if res.memory else 1) or 1.0,
    )


def build_pipeline(
    config_path: str | Path | None = None, **kwargs: Any
) -> AgenticDocumentExtractionFramework:
    config = load_config(config_path)
    configure_logging(config.log_level)
    return AgenticDocumentExtractionFramework(config, **kwargs)


ExtractionPipeline = AgenticDocumentExtractionFramework
