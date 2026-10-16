"""OCR engines behind one interface (Tesseract for real runs, a deterministic mock for tests)."""

from __future__ import annotations

import itertools
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from extraction.image_ops import Image, as_u8, contrast, sharpness, to_gray
from extraction.schemas import BBox, OCRPageResult, OCRWord, PageData, RawBlock


class OCREngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image: Image) -> OCRPageResult: ...


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def __init__(self, language: str = "eng") -> None:
        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract binary not found. Install it (e.g. `sudo apt install tesseract-ocr`) "
                "or set ocr.engine in config.yaml."
            )
        import pytesseract

        self._tess = pytesseract
        self.language = language

    def recognize(self, image: Image) -> OCRPageResult:
        data = self._tess.image_to_data(
            image, lang=self.language, output_type=self._tess.Output.DICT
        )
        words: list[OCRWord] = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0 or not str(text).strip():
                continue
            words.append(
                OCRWord(
                    text=str(text),
                    bbox=BBox(
                        x=data["left"][i], y=data["top"][i], w=data["width"][i], h=data["height"][i]
                    ),
                    confidence=conf / 100.0,
                    block=int(data["block_num"][i]) * 1000 + int(data["par_num"][i]),
                    line=int(data["line_num"][i]),
                )
            )
        mean = float(np.mean([w.confidence for w in words])) if words else 0.0
        return OCRPageResult(words=words, mean_confidence=mean)


class MockOCREngine(OCREngine):
    """Deterministic OCR stand-in.

    Text lines come from ``lines`` and are laid onto the ink bands found by a horizontal
    projection of the image. Confidence grows with sharpness and contrast, so image
    enhancement measurably changes it (ΔC).
    """

    name = "mock"

    def __init__(
        self,
        lines: Sequence[str] | None = None,
        *,
        sharpness_ref: float = 50.0,
        contrast_ref: float = 120.0,
    ) -> None:
        self.lines = list(lines or [])
        self.sharpness_ref = sharpness_ref
        self.contrast_ref = contrast_ref

    def confidence_for(self, gray: Image) -> float:
        s = min(1.0, sharpness(gray) / self.sharpness_ref)
        c = min(1.0, contrast(gray) / self.contrast_ref)
        return round(0.3 + 0.7 * (0.6 * s + 0.4 * c), 4)

    def recognize(self, image: Image) -> OCRPageResult:
        gray = to_gray(image)
        conf = self.confidence_for(gray)
        smooth = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (0, 0), 1.5)
        _, raw_mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = as_u8(raw_mask)
        rows = mask.sum(axis=1) > 0.004 * 255 * mask.shape[1]
        bands: list[tuple[int, int]] = []
        start: int | None = None
        for y, on in enumerate([*rows.tolist(), False]):
            if on and start is None:
                start = y
            elif not on and start is not None:
                if y - start >= 3:
                    bands.append((start, y))
                start = None
        words: list[OCRWord] = []
        height, width = gray.shape[:2]
        for line_no, text in enumerate(self.lines):
            if line_no < len(bands):
                y0, y1 = bands[line_no]
                cols = np.where(mask[y0:y1].sum(axis=0) > 0)[0]
                x0 = int(cols[0]) if len(cols) else int(width * 0.1)
                x1 = int(cols[-1]) if len(cols) else int(width * 0.9)
            else:  # more text lines than ink bands: stack them below, inside the page
                y0 = min(
                    height - 14, (bands[-1][1] if bands else 0) + 14 * (line_no - len(bands) + 1)
                )
                y1 = min(height - 1, y0 + 12)
                x0, x1 = int(width * 0.1), int(width * 0.9)
            segments = [seg for seg in re.split(r"\s{3,}", text) if seg.strip()]
            spans = [(x0, x1)]
            if len(segments) > 1 and line_no < len(bands):
                clusters = self._ink_clusters(
                    mask[bands[line_no][0] : bands[line_no][1]], min_gap=int(3 * (y1 - y0))
                )
                if len(clusters) == len(segments):
                    spans = clusters
            if len(spans) != len(segments):
                segments, spans = [text], [(x0, x1)]
            for seg, (sx0, sx1) in zip(segments, spans, strict=True):
                n = max(1, len(seg))
                for m in re.finditer(r"\S+", seg):  # words at their character offsets
                    wx0 = sx0 + (sx1 - sx0) * m.start() / n
                    wx1 = sx0 + (sx1 - sx0) * m.end() / n
                    words.append(
                        OCRWord(
                            text=m.group(0),
                            bbox=BBox(x=wx0, y=y0, w=max(1.0, wx1 - wx0), h=y1 - y0),
                            confidence=conf,
                            block=0,
                            line=line_no,
                        )
                    )
        return OCRPageResult(words=words, mean_confidence=conf)

    @staticmethod
    def _ink_clusters(band: Image, min_gap: int) -> list[tuple[int, int]]:
        """Horizontal ink extents in a text band, merging gaps narrower than ``min_gap``."""
        cols = np.where(band.sum(axis=0) > 0)[0]
        if not len(cols):
            return []
        clusters = [[int(cols[0]), int(cols[0])]]
        for c in cols[1:]:
            if c - clusters[-1][1] > min_gap:
                clusters.append([int(c), int(c)])
            else:
                clusters[-1][1] = int(c)
        return [(a, b) for a, b in clusters]


def build_ocr_engine(engine: str, language: str) -> OCREngine:
    if engine == "tesseract":
        return TesseractOCREngine(language)
    return MockOCREngine()


def _split_cells(words: list[OCRWord], line_h: float) -> list[list[OCRWord]]:
    """Split a line's words into cells at gaps wider than 1.5x the line height."""
    cells: list[list[OCRWord]] = [[words[0]]]
    for prev, w in itertools.pairwise(words):
        if w.bbox.x - prev.bbox.x1 > 1.5 * line_h:
            cells.append([w])
        else:
            cells[-1].append(w)
    return cells


def words_to_blocks(page: PageData, result: OCRPageResult) -> list[RawBlock]:
    """Group OCR words into blocks: aligned multi-cell line runs become tables, other lines
    are grouped into paragraphs (split on vertical gaps, size changes and engine blocks)."""
    if not result.words:
        return []
    grouped: dict[tuple[int, int], list[OCRWord]] = {}
    for w in result.words:
        grouped.setdefault((w.block, w.line), []).append(w)
    lines: list[dict[str, Any]] = []
    for (block, _line), ws in grouped.items():
        ws.sort(key=lambda w: w.bbox.x)
        box = ws[0].bbox
        for w in ws[1:]:
            box = box.union(w.bbox)
        lines.append(
            {
                "box": box,
                "words": ws,
                "block": block,
                "text": " ".join(w.text for w in ws),
                "conf": float(np.mean([w.confidence for w in ws])),
            }
        )
    lines.sort(key=lambda ln: (ln["box"].y, ln["box"].x))
    heights = sorted(ln["box"].h for ln in lines)
    median_h = heights[len(heights) // 2] or 10.0
    for ln in lines:
        ln["cells"] = _split_cells(ln["words"], median_h)

    # Runs of >= 2 consecutive lines with the same number (>= 2) of aligned cells → table.
    in_table: dict[int, int] = {}
    runs: list[list[int]] = []
    i = 0
    while i < len(lines):
        n = len(lines[i]["cells"])
        j = i + 1
        while (
            n >= 2
            and j < len(lines)
            and len(lines[j]["cells"]) == n
            and all(
                abs(a[0].bbox.x - b[0].bbox.x) < 2.5 * median_h
                for a, b in zip(lines[j - 1]["cells"], lines[j]["cells"], strict=True)
            )
        ):
            j += 1
        if n >= 2 and j - i >= 2:
            runs.append(list(range(i, j)))
            for k in range(i, j):
                in_table[k] = len(runs) - 1
            i = j
        else:
            i += 1

    blocks: list[RawBlock] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        box = current[0]["box"]
        for item in current[1:]:
            box = box.union(item["box"])
        sizes = sorted(item["box"].h for item in current)
        blocks.append(
            RawBlock(
                block_id=f"p{page.page_number}-o{len(blocks)}",
                page=page.page_number,
                bbox=box,
                text="\n".join(item["text"] for item in current),
                font_size=round(sizes[len(sizes) // 2], 2),
                source="ocr",
                confidence=round(float(np.mean([item["conf"] for item in current])), 4),
            )
        )
        current.clear()

    emitted: set[int] = set()
    for idx, ln in enumerate(lines):
        if idx in in_table:
            flush()
            run_id = in_table[idx]
            if run_id in emitted:
                continue
            emitted.add(run_id)
            rows = [lines[k] for k in runs[run_id]]
            box = rows[0]["box"]
            for r in rows[1:]:
                box = box.union(r["box"])
            cells: list[list[str | None]] = [
                [" ".join(w.text for w in cell) for cell in r["cells"]] for r in rows
            ]
            blocks.append(
                RawBlock(
                    block_id=f"p{page.page_number}-o{len(blocks)}",
                    page=page.page_number,
                    bbox=box,
                    kind="table",
                    table_cells=cells,
                    source="ocr",
                    text="\n".join(" | ".join(c or "" for c in row) for row in cells),
                    confidence=round(float(np.mean([r["conf"] for r in rows])), 4),
                )
            )
            continue
        if current:
            prev = current[-1]
            gap = ln["box"].y - prev["box"].y1
            size_change = abs(ln["box"].h - prev["box"].h) > 0.35 * median_h
            if gap > 0.9 * median_h or ln["block"] != prev["block"] or size_change:
                flush()
        current.append(ln)
    flush()
    return blocks
