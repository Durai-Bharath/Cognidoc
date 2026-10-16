"""Algorithm 1 - the LangGraph orchestration (routing, loop, conditional agents)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from common.config import ExtractionConfig
from common.embedder import FakeEmbedder
from extraction.graph import AgenticDocumentExtractionFramework, PipelineResult
from extraction.ocr import MockOCREngine
from extraction.testing import offline_llm, scanned_page_lines


def _run(config: ExtractionConfig, path: Path, out: Path) -> PipelineResult:
    fw = AgenticDocumentExtractionFramework(
        config, llm=offline_llm(), embedder=FakeEmbedder(), ocr=MockOCREngine(scanned_page_lines())
    )
    return fw.run(path, out)


def _algorithms(result: PipelineResult) -> list[int]:
    return [t.algorithm for t in result.traces]


def test_native_pdf_skips_ocr_branch(
    config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path
) -> None:
    result = _run(config, samples["pdf"], tmp_path)
    algs = _algorithms(result)
    assert algs[:5] == [2, 3, 6, 7, 8] and 4 not in algs and 5 not in algs
    assert algs[-2:] == [20, 21]
    # per-section loop: 3 sections; table/figure/equation only where present (sec-2)
    counts = Counter(algs)
    assert counts[9] == counts[13] == counts[19] == 3
    assert counts[10] == counts[11] == counts[12] == 1
    sec2 = [t.algorithm for t in result.traces if t.context.get("section") == "sec-2"]
    assert sec2 == [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def test_degraded_image_runs_quality_enhancement_and_reclassification(
    config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path
) -> None:
    algs = _algorithms(_run(config, samples["scan"], tmp_path))
    assert algs[:6] == [2, 3, 4, 5, 3, 6]


def test_clean_image_skips_enhancement(
    config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path
) -> None:
    algs = _algorithms(_run(config, samples["clean_scan"], tmp_path))
    assert algs[:5] == [2, 3, 4, 3, 6] and 5 not in algs


def test_router_failure_ends_with_failure_report(config: ExtractionConfig, tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("plain text")
    result = _run(config, bad, tmp_path / "out")
    assert not result.ok and result.error is not None
    assert result.error.error_type == "UnsupportedDocument"
    report = result.validation_report
    assert report is not None and not report.approved and report.agent_errors
    assert set(result.output_paths) == {"validation_report"}
    assert Path(result.output_paths["validation_report"]).exists()


def test_non_critical_agent_failure_is_recorded_and_pipeline_continues(
    config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path
) -> None:
    fw = AgenticDocumentExtractionFramework(
        config, llm=offline_llm(), embedder=FakeEmbedder(), ocr=MockOCREngine()
    )

    def boom(_data: object) -> object:
        raise RuntimeError("table agent exploded")

    fw.tables._execute = boom  # type: ignore[method-assign,assignment]
    result = fw.run(samples["pdf"], tmp_path)
    assert result.ok and any("table agent exploded" in e for e in result.agent_errors)
    assert result.validation_report is not None and not result.validation_report.approved


def test_mermaid_lists_every_node(config: ExtractionConfig) -> None:
    fw = AgenticDocumentExtractionFramework(config, llm=offline_llm(), embedder=FakeEmbedder())
    diagram = fw.mermaid()
    for node in (
        "router",
        "ocr_quality",
        "image_enhancement",
        "next_section",
        "table_analysis",
        "figure_understanding",
        "equation_analysis",
        "confidence_scoring",
        "metadata_fusion",
        "global_validation",
        "export",
    ):
        assert node in diagram
