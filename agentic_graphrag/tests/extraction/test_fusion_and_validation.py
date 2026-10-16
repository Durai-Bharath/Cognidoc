"""Algorithms 20-21 - MetadataFusionAgent and GlobalValidationAgent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.base_agent import AgentContext
from common.config import ExtractionConfig
from common.embedder import FakeEmbedder
from extraction.agents.global_validation import GlobalValidationAgent, GlobalValidationInput
from extraction.agents.metadata_fusion import FusionOutput, MetadataFusionAgent
from extraction.graph import AgenticDocumentExtractionFramework
from extraction.ocr import MockOCREngine
from extraction.schemas import FusionInput
from extraction.testing import offline_llm


@pytest.fixture
def fusion_input(config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path) -> FusionInput:
    fw = AgenticDocumentExtractionFramework(
        config, llm=offline_llm(), embedder=FakeEmbedder(), ocr=MockOCREngine()
    )
    fw.run(samples["pdf"], tmp_path / "run")
    s = fw.last_state
    return FusionInput(
        parsed=s["parsed"],
        classification=s["classification"],
        layout=s["layout"],
        ordered=s["ordered"],
        hierarchy=s["hierarchy"],
        results=s["results"],
        abbreviations=s["abbreviations"],
    )


def _fuse(ctx: AgentContext, data: FusionInput) -> FusionOutput:
    return MetadataFusionAgent(ctx).run(data).unwrap()


# ----------------------------------------------------------------- Alg. 20


def test_fusion_builds_schema_valid_m_and_j(ctx: AgentContext, fusion_input: FusionInput) -> None:
    out = _fuse(ctx, fusion_input)
    assert out.export_approved and out.validation.passed
    m, j = out.metadata, out.document
    assert m.document.document_type == "PDF" and m.document.total_pages == 2
    assert [s.section_id for s in m.sections] == ["sec-1", "sec-2", "sec-3"]
    method = next(s for s in j.sections if s.section_id == "sec-2")
    assert method.tables[0].headers == ["Component", "Interval (h)", "Status"]
    assert method.figures[0].figure_type == "flowchart" and method.equations[0].latex
    assert m.statistics == j.statistics and m.statistics.total_tables == 1
    types = {r.type for r in j.relationships}
    assert {"contains", "references", "defines"} <= types
    assert any(r.source == "abbr:HCU" for r in j.relationships)
    assert m.quality.document_confidence > 0 and not m.quality.ocr_applied
    assert method.text.count("2 Method") == 1  # headings are not duplicated


def test_fusion_applies_verified_texts_and_flags_schema_errors(
    ctx: AgentContext, fusion_input: FusionInput, tmp_path: Path
) -> None:
    strict_dir = tmp_path / "schemas"
    strict_dir.mkdir()
    for name in ("metadata.schema.json", "document.schema.json"):
        (strict_dir / name).write_text(
            json.dumps({"type": "object", "required": ["field_that_does_not_exist"]})
        )
    ctx.config.paths.schemas_dir = str(strict_dir)
    out = _fuse(ctx, fusion_input)
    assert not out.export_approved
    assert {c.name for c in out.validation.errors} == {"metadata_schema", "json_schema"}
    assert ctx.review_queue.items[-1].agent == "MetadataFusionAgent"


# ----------------------------------------------------------------- Alg. 21


def _validate(ctx: AgentContext, out: FusionOutput, errors: list[str] | None = None):  # type: ignore[no-untyped-def]
    return (
        GlobalValidationAgent(ctx)
        .run(
            GlobalValidationInput(
                metadata=out.metadata, document=out.document, agent_errors=errors or []
            )
        )
        .unwrap()
    )


def test_clean_document_is_approved(ctx: AgentContext, fusion_input: FusionInput) -> None:
    report = _validate(ctx, _fuse(ctx, fusion_input))
    assert report.approved and report.quality_score == 1.0
    assert [c.name for c in report.categories] == [
        "document_integrity",
        "extracted_objects",
        "metadata_consistency",
        "json_schema",
        "cross_document_consistency",
    ]
    assert report.statistics.total_errors == 0 and report.statistics.validation_accuracy == 1.0


def test_inconsistencies_are_reported_with_recommendations(
    ctx: AgentContext, fusion_input: FusionInput
) -> None:
    out = _fuse(ctx, fusion_input)
    j = out.document
    method = next(s for s in j.sections if s.section_id == "sec-2")
    method.tables[0].rows.append(["only one cell"])  # table width
    method.tables = []  # and stats mismatch
    j.sections[0].cross_references[0].status = "unresolved"
    j.sections[0].chunks.append(j.sections[0].chunks[0])  # duplicate id
    out.metadata.sections[0].confidence_score = 0.1  # level mismatch
    report = _validate(ctx, out)
    failed = {
        ch.name
        for cat in report.categories
        for ch in cat.checks
        if not ch.passed and ch.severity == "error"
    }
    assert {
        "document_statistics",
        "broken_references",
        "duplicate_entries",
        "confidence_scores",
        "object_associations",
    } <= failed
    assert not report.approved and report.quality_score < 1.0
    assert any("CrossReferenceValidationAgent" in r for r in report.recommendations)
    assert ctx.review_queue.items[-1].agent == "GlobalValidationAgent"


def test_agent_errors_block_approval(ctx: AgentContext, fusion_input: FusionInput) -> None:
    report = _validate(ctx, _fuse(ctx, fusion_input), errors=["TableAnalysisAgent: boom"])
    assert not report.approved and report.agent_errors == ["TableAnalysisAgent: boom"]
    assert report.recommendations[-1].startswith("Re-run failed agents")


def test_quality_threshold_boundary(ctx: AgentContext, fusion_input: FusionInput) -> None:
    out = _fuse(ctx, fusion_input)
    out.document.sections[0].keywords = ["changed"]  # warning-only check fails
    report = _validate(ctx, out)
    ctx.config.global_validation.approve_threshold = report.quality_score
    assert _validate(ctx, out).approved  # Q ≥ threshold and no errors
    ctx.config.global_validation.approve_threshold = min(1.0, report.quality_score + 1e-3)
    assert not _validate(ctx, out).approved
