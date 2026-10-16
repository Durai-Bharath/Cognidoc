"""Algorithm 4 - OCR Quality Assessment Agent."""

from __future__ import annotations

from common.base_agent import AgentContext, AgentFailure, BaseAgent
from common.config import OCRQualityConfig
from extraction import image_ops as ops
from extraction.ocr import OCREngine, build_ocr_engine, words_to_blocks
from extraction.schemas import OCRPageResult, OCRQualityInput, OCRQualityReport, PageQuality


def get_ocr_engine(ctx: AgentContext) -> OCREngine:
    engine = ctx.services.get("ocr")
    if engine is None:
        engine = build_ocr_engine(ctx.config.ocr.engine, ctx.config.ocr.language)
        ctx.services["ocr"] = engine
    assert isinstance(engine, OCREngine)
    return engine


def compute_metrics(img: ops.Image, ocr: OCRPageResult) -> dict[str, float]:
    """Raw image-quality metrics of Alg. 4 lines 4-14."""
    gray = ops.to_gray(img)
    return {
        "resolution": float(min(gray.shape[:2])),
        "brightness": ops.brightness(gray),
        "contrast": ops.contrast(gray),
        "sharpness": ops.sharpness(gray),
        "noise": ops.noise_sigma(gray),
        "blur": ops.high_frequency_ratio(gray),
        "skew": ops.estimate_skew(gray),
        "rotation": float(ops.estimate_rotation(gray)),
        "compression": ops.blockiness(gray),
        "ocr_confidence": ocr.mean_confidence,
    }


def normalize_metrics(raw: dict[str, float], cfg: OCRQualityConfig) -> dict[str, float]:
    """m_k ∈ [0, 1] for every metric (1 = ideal)."""

    def clip(v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    return {
        "resolution": clip(raw["resolution"] / cfg.target_min_side_px),
        "brightness": clip(
            1 - abs(raw["brightness"] - cfg.target_brightness) / cfg.target_brightness
        ),
        "contrast": clip(raw["contrast"] / cfg.contrast_ref),
        "sharpness": clip(raw["sharpness"] / cfg.sharpness_ref),
        "noise": clip(1 - raw["noise"] / cfg.noise_ref),
        "blur": clip(raw["blur"] / cfg.blur_ref),
        "skew": clip(1 - abs(raw["skew"]) / cfg.max_skew_deg),
        "rotation": 1.0 if raw["rotation"] == 0 else 0.0,
        "compression": clip(1 - raw["compression"] / cfg.compression_ref),
        "ocr_confidence": clip(raw["ocr_confidence"]),
    }


def quality_score(normalized: dict[str, float], weights: dict[str, float]) -> float:
    """Q_i = Σ w_k m_k with weights renormalised to sum to 1."""
    total = sum(weights.values())
    return round(sum(weights[k] * normalized.get(k, 0.0) for k in weights) / total, 4)


class OCRQualityAssessmentAgent(BaseAgent[OCRQualityInput, OCRQualityReport]):
    name = "OCRQualityAssessmentAgent"
    algorithm = 4

    def _execute(self, data: OCRQualityInput) -> OCRQualityReport:
        cfg = self.ctx.config.ocr_quality
        engine = get_ocr_engine(self.ctx)
        parsed = data.parsed.model_copy(deep=True)
        report: list[PageQuality] = []

        for page_no in parsed.ocr_pages:
            page = parsed.page(page_no)
            if not page.image_path:
                raise AgentFailure(
                    f"page {page_no} needs OCR but has no image", error_type="MissingPageImage"
                )
            img = ops.read_image(page.image_path)
            ocr = engine.recognize(img)
            raw = compute_metrics(img, ocr)
            normalized = normalize_metrics(raw, cfg)
            defects, details = ops.detect_defects(ops.to_gray(img), cfg.low_illumination_mean)
            score = quality_score(normalized, cfg.weights)
            status = "VALID" if score >= cfg.tau_quality else "LOW_QUALITY"
            page.blocks = words_to_blocks(page, ocr)
            report.append(
                PageQuality(
                    page=page_no,
                    metrics={k: round(v, 4) for k, v in raw.items()},
                    normalized=normalized,
                    defects=defects,
                    defect_details={k: round(v, 4) for k, v in details.items()},
                    score=score,
                    status=status,
                    ocr_confidence=round(ocr.mean_confidence, 4),
                )
            )

        parsed.raw_text = "\n\n".join(b.text for p in parsed.pages for b in p.blocks if b.text)
        doc_score = round(sum(p.score for p in report) / len(report), 4) if report else 1.0
        return OCRQualityReport(
            pages=report,
            document_score=doc_score,
            low_quality_pages=[p.page for p in report if p.status == "LOW_QUALITY"],
            parsed=parsed,
        )

    def confidence(self, output: OCRQualityReport) -> float | None:
        return output.document_score
