"""Algorithm 5 - Image Enhancement Agent."""

from __future__ import annotations

from pathlib import Path

import cv2

from common.base_agent import BaseAgent
from extraction import image_ops as ops
from extraction.agents.ocr_quality import get_ocr_engine
from extraction.ocr import words_to_blocks
from extraction.schemas import EnhancementInput, EnhancementReport, PageEnhancement, PageQuality


class ImageEnhancementAgent(BaseAgent[EnhancementInput, EnhancementReport]):
    name = "ImageEnhancementAgent"
    algorithm = 5

    def _super_resolution(self) -> ops.SuperResolution:
        cfg = self.ctx.config.enhancement
        backend = self.ctx.services.get("super_resolution")
        if isinstance(backend, ops.SuperResolution):
            return backend
        if cfg.super_resolution == "opencv_dnn" and cfg.super_resolution_model:
            return ops.OpenCVDnnSuperResolution(
                cfg.super_resolution_model, cfg.super_resolution_scale
            )
        return ops.BicubicSuperResolution(cfg.super_resolution_scale)

    def estimate_degradations(self, quality: PageQuality) -> list[str]:
        """Lines 3-11: degradation types from the page's quality metrics and defects."""
        cfg = self.ctx.config.enhancement
        m, d = quality.normalized, quality.defects
        found: list[str] = []
        if m["noise"] < cfg.noise_threshold:
            found.append("noise")
        if min(m["sharpness"], m["blur"]) < cfg.blur_threshold:
            found.append("blur")
        if m["contrast"] < cfg.contrast_threshold:
            found.append("low_contrast")
        if d.get("low_illumination") or d.get("shadow_regions"):
            found.append("low_illumination")
        if abs(quality.metrics["skew"]) > cfg.skew_threshold_deg:
            found.append("skew")
        if quality.metrics["rotation"] != 0:
            found.append("rotation")
        if quality.metrics["resolution"] < cfg.min_side_px:
            found.append("low_resolution")
        if d.get("background_noise") or d.get("shadow_regions") or d.get("watermarks"):
            found.append("background_artifacts")
        return found

    def enhance(
        self, img: ops.Image, quality: PageQuality, degradations: list[str]
    ) -> tuple[ops.Image, list[str]]:
        """Lines 12-35: conditional preprocessing followed by normalisation."""
        gray = ops.to_gray(img)
        applied: list[str] = []
        if "noise" in degradations:
            gray = ops.denoise(gray)
            applied.append("denoise")
        if "blur" in degradations:
            gray = ops.sharpen(gray)
            applied.append("sharpen")
        if "low_contrast" in degradations:
            gray = ops.enhance_contrast(gray)
            applied.append("contrast_enhancement")
        if "low_illumination" in degradations:
            gray = ops.correct_brightness(gray, self.ctx.config.ocr_quality.target_brightness)
            applied.append("brightness_correction")
        if "skew" in degradations:
            gray = ops.rotate_image(gray, -quality.metrics["skew"])
            applied.append(f"deskew({quality.metrics['skew']:.2f}deg)")
        if "rotation" in degradations:
            gray = ops.as_u8(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE))
            applied.append("orientation_correction")
        if "low_resolution" in degradations:
            gray = ops.to_gray(self._super_resolution().upscale(gray))
            applied.append("super_resolution")
        if "background_artifacts" in degradations:
            gray = ops.remove_background_noise(gray)
            if quality.defects.get("shadow_regions"):
                gray = ops.suppress_shadows(gray)
                applied.append("shadow_suppression")
            applied.append("background_noise_removal")
        gray = ops.normalize_intensity(gray)
        applied.append("intensity_normalization")
        applied.append("color_normalization(grayscale)")
        return gray, applied

    def _execute(self, data: EnhancementInput) -> EnhancementReport:
        engine = get_ocr_engine(self.ctx)
        parsed = data.parsed.model_copy(deep=True)
        by_page = {q.page: q for q in data.quality.pages}
        results: list[PageEnhancement] = []

        for page_no in data.quality.low_quality_pages:
            quality = by_page[page_no]
            page = parsed.page(page_no)
            assert page.image_path is not None
            original = ops.read_image(page.image_path)
            degradations = self.estimate_degradations(quality)
            enhanced, applied = self.enhance(original, quality, degradations)

            # Lines 36-38: OCR on the enhanced image, ΔC = C_after − C_before.
            ocr_after = engine.recognize(enhanced)
            before, after = quality.ocr_confidence, ocr_after.mean_confidence
            delta = round(after - before, 4)
            out_path = Path(page.image_path).with_name(f"page_{page_no:03d}_enhanced.png")
            if delta > 0:
                ops.write_image(out_path, enhanced)
                page.image_path = str(out_path)
                # OCR coordinates are in enhanced-image pixels.
                page.width, page.height = float(enhanced.shape[1]), float(enhanced.shape[0])
                page.blocks = words_to_blocks(page, ocr_after)
                accepted = True
            else:
                # Lines 41-43: restore the original image and flag for inspection.
                accepted = False
                self.flag(
                    f"enhancement did not improve OCR confidence (ΔC={delta:+.4f}); "
                    "original image restored",
                    item_ref=f"page:{page_no}",
                    details={"degradations": degradations, "operations": applied},
                )
            results.append(
                PageEnhancement(
                    page=page_no,
                    degradations=degradations,
                    operations=applied,
                    confidence_before=round(before, 4),
                    confidence_after=round(after, 4),
                    delta=delta,
                    accepted=accepted,
                    image_path=page.image_path,
                )
            )
        parsed.raw_text = "\n\n".join(b.text for p in parsed.pages for b in p.blocks if b.text)
        return EnhancementReport(pages=results, parsed=parsed)

    def confidence(self, output: EnhancementReport) -> float | None:
        if not output.pages:
            return 1.0
        return round(sum(p.accepted for p in output.pages) / len(output.pages), 4)
