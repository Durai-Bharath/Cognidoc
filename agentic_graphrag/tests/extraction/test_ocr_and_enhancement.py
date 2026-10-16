"""Algorithms 4 and 5 - OCRQualityAssessmentAgent and ImageEnhancementAgent."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from common.base_agent import AgentContext
from extraction import image_ops as ops
from extraction.agents.image_enhancement import ImageEnhancementAgent
from extraction.agents.ocr_quality import (
    OCRQualityAssessmentAgent,
    normalize_metrics,
    quality_score,
)
from extraction.agents.router import RouterAgent
from extraction.ocr import MockOCREngine, OCREngine, words_to_blocks
from extraction.schemas import (
    BBox,
    EnhancementInput,
    OCRPageResult,
    OCRQualityInput,
    OCRQualityReport,
    OCRWord,
    PageData,
    ParsedDocument,
    RouterInput,
)


def _parsed(ctx: AgentContext, path: Path, tmp_path: Path) -> ParsedDocument:
    return RouterAgent(ctx).run(RouterInput(path=str(path), workdir=str(tmp_path))).unwrap().parsed


def _quality(ctx: AgentContext, parsed: ParsedDocument) -> OCRQualityReport:
    return OCRQualityAssessmentAgent(ctx).run(OCRQualityInput(parsed=parsed)).unwrap()


# ----------------------------------------------------------------- Alg. 4


def test_clean_page_is_valid_and_ocr_text_extracted(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    report = _quality(ctx, _parsed(ctx, samples["clean_scan"], tmp_path))
    page = report.pages[0]
    assert page.status == "VALID" and report.low_quality_pages == []
    assert set(page.normalized) == set(ctx.config.ocr_quality.weights)
    assert all(0 <= v <= 1 for v in page.normalized.values())
    assert set(page.defects) == {
        "missing_regions",
        "broken_characters",
        "shadow_regions",
        "watermarks",
        "low_illumination",
        "background_noise",
    }
    assert "1 Introduction" in report.parsed.raw_text
    assert any(b.kind == "table" for b in report.parsed.pages[0].blocks)


def test_degraded_page_is_low_quality(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    report = _quality(ctx, _parsed(ctx, samples["scan"], tmp_path))
    assert report.pages[0].status == "LOW_QUALITY" and report.low_quality_pages == [1]
    assert report.pages[0].score < ctx.config.ocr_quality.tau_quality


def test_quality_threshold_boundary_is_valid(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    parsed = _parsed(ctx, samples["scan"], tmp_path)
    score = _quality(ctx, parsed).pages[0].score
    ctx.config.ocr_quality.tau_quality = score  # Q_i ≥ τ → VALID
    assert _quality(ctx, parsed).pages[0].status == "VALID"


def test_quality_score_is_weighted_sum() -> None:
    m = {"a": 1.0, "b": 0.0}
    assert quality_score(m, {"a": 3.0, "b": 1.0}) == 0.75  # renormalised weights


def test_rotation_and_skew_normalisation(ctx: AgentContext) -> None:
    raw = {
        "resolution": 2000,
        "brightness": 200,
        "contrast": 120,
        "sharpness": 50,
        "noise": 0,
        "blur": 1,
        "skew": 5.0,
        "rotation": 90.0,
        "compression": 0,
        "ocr_confidence": 0.9,
    }
    m = normalize_metrics(raw, ctx.config.ocr_quality)
    assert m["rotation"] == 0.0 and m["skew"] == 0.5 and m["resolution"] == 1.0


def test_skew_estimation_detects_rotation() -> None:
    img = np.full((600, 800), 255, np.uint8)
    for y in range(80, 560, 40):
        cv2.putText(
            img, "skewed text line for testing " * 2, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2
        )
    rotated = ops.rotate_image(img, 4.0)
    assert abs(ops.estimate_skew(rotated) - 4.0) <= 1.0
    assert abs(ops.estimate_skew(img)) <= 0.5


def test_missing_page_image_is_a_typed_error(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    parsed = _parsed(ctx, samples["scan"], tmp_path)
    parsed.pages[0].image_path = None
    result = OCRQualityAssessmentAgent(ctx).run(OCRQualityInput(parsed=parsed))
    assert result.error is not None and result.error.error_type == "MissingPageImage"


def test_words_to_blocks_detects_tables_and_paragraphs() -> None:
    def word(text: str, x: float, y: float, line: int) -> OCRWord:
        return OCRWord(
            text=text, bbox=BBox(x=x, y=y, w=10 * len(text), h=12), confidence=0.9, line=line
        )

    words = [
        word("Heading", 10, 10, 0),
        word("A", 10, 40, 1),
        word("B", 200, 40, 1),
        word("1", 10, 60, 2),
        word("2", 200, 60, 2),
    ]
    blocks = words_to_blocks(
        PageData(page_number=1, width=400, height=400),
        OCRPageResult(words=words, mean_confidence=0.9),
    )
    assert [b.kind for b in blocks] == ["text", "table"]
    assert blocks[1].table_cells == [["A", "B"], ["1", "2"]]


# ----------------------------------------------------------------- Alg. 5


def test_enhancement_accepted_when_confidence_improves(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    parsed = _parsed(ctx, samples["scan"], tmp_path)
    quality = _quality(ctx, parsed)
    report = (
        ImageEnhancementAgent(ctx)
        .run(EnhancementInput(parsed=quality.parsed, quality=quality))
        .unwrap()
    )
    page = report.pages[0]
    assert page.accepted and page.delta > 0
    assert page.confidence_after == page.confidence_before + page.delta
    assert "noise" in page.degradations and "denoise" in page.operations
    assert page.image_path.endswith("_enhanced.png") and Path(page.image_path).exists()
    assert report.parsed.pages[0].image_path == page.image_path


class _WorseOCR(OCREngine):
    name = "worse"

    def __init__(self, before: float, after: float) -> None:
        self.calls = 0
        self.values = [before, after]

    def recognize(self, image: ops.Image) -> OCRPageResult:
        value = self.values[min(self.calls, 1)]
        self.calls += 1
        return OCRPageResult(
            words=[OCRWord(text="x", bbox=BBox(x=1, y=1, w=5, h=5), confidence=value)],
            mean_confidence=value,
        )


@pytest.mark.parametrize("after", [0.40, 0.50], ids=["worse", "equal-boundary"])
def test_enhancement_rejected_restores_original_and_flags(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path, after: float
) -> None:
    ctx.services["ocr"] = _WorseOCR(before=0.50, after=after)
    ctx.config.ocr_quality.tau_quality = 0.99
    parsed = _parsed(ctx, samples["scan"], tmp_path)
    quality = _quality(ctx, parsed)
    original = quality.parsed.pages[0].image_path
    report = (
        ImageEnhancementAgent(ctx)
        .run(EnhancementInput(parsed=quality.parsed, quality=quality))
        .unwrap()
    )
    page = report.pages[0]
    assert not page.accepted and page.delta <= 0  # ΔC > 0 required
    assert report.parsed.pages[0].image_path == original
    assert ctx.review_queue.items[-1].agent == "ImageEnhancementAgent"


def test_rotation_skew_and_super_resolution_paths(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    class Doubler(ops.SuperResolution):
        used = False

        def upscale(self, img: ops.Image) -> ops.Image:
            Doubler.used = True
            return ops.BicubicSuperResolution(2).upscale(img)

    ctx.services["super_resolution"] = Doubler()
    ctx.config.enhancement.min_side_px = 5000
    parsed = _parsed(ctx, samples["scan"], tmp_path)
    quality = _quality(ctx, parsed)
    q = quality.pages[0]
    q.metrics["skew"], q.metrics["rotation"] = 2.0, 90.0
    agent = ImageEnhancementAgent(ctx)
    degradations = agent.estimate_degradations(q)
    assert {"skew", "rotation", "low_resolution"} <= set(degradations)
    img = ops.read_image(parsed.pages[0].image_path or "")
    enhanced, ops_applied = agent.enhance(img, q, degradations)
    assert Doubler.used and "super_resolution" in ops_applied
    assert any(o.startswith("deskew") for o in ops_applied)
    assert enhanced.shape[0] == 2 * img.shape[1]  # rotated 90° then upscaled x2


def test_mock_ocr_confidence_tracks_image_quality(samples: dict[str, Path]) -> None:
    engine = MockOCREngine()
    clean = engine.confidence_for(ops.to_gray(ops.read_image(samples["clean_scan"])))
    degraded = engine.confidence_for(ops.to_gray(ops.read_image(samples["scan"])))
    assert clean > degraded
