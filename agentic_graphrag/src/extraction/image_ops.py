"""OpenCV image-quality metrics, defect detectors and enhancement operations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


def as_u8(arr: Any) -> Image:
    """Coerce an OpenCV result to a typed uint8 array."""
    return np.asarray(arr, dtype=np.uint8)


def read_image(path: str | Path) -> Image:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img.astype(np.uint8)


def write_image(path: str | Path, img: Image) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def to_gray(img: Image) -> Image:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def _downscale(gray: Image, max_side: int = 1000) -> Image:
    h, w = gray.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return gray
    return as_u8(cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA))


# =========================================================================== metrics


def _presmooth(gray: Image) -> Image:
    """Suppress pixel noise so it is not mistaken for edge detail (noise has its own metric)."""
    return cv2.GaussianBlur(cv2.medianBlur(gray, 3), (0, 0), 1.0).astype(np.uint8)


def brightness(gray: Image) -> float:
    return float(gray.mean())


def contrast(gray: Image) -> float:
    """Ink/paper separation: mean grey of the paper class minus the ink class (Otsu split)."""
    smooth = _presmooth(gray)
    t, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper, ink = smooth[smooth > t], smooth[smooth <= t]
    if paper.size == 0 or ink.size == 0:
        return 0.0
    return float(paper.mean() - ink.mean())


def sharpness(gray: Image) -> float:
    """Variance of the Laplacian of the pre-smoothed image (higher = sharper)."""
    return float(cv2.Laplacian(_presmooth(gray), cv2.CV_64F).var())


def noise_sigma(gray: Image) -> float:
    """Immerkær fast noise-variance estimate (sigma in grey levels)."""
    g = gray.astype(np.float64)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, kernel)
    return float(math.sqrt(math.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))


def high_frequency_ratio(gray: Image) -> float:
    """Share of spectral energy outside the low-frequency disc (low = blurry)."""
    small = _downscale(_presmooth(gray), 512).astype(np.float64)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small - small.mean())))
    h, w = spectrum.shape
    yy, xx = np.ogrid[:h, :w]
    radius = min(h, w) / 8
    low = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= radius**2
    total = spectrum.sum()
    return float(spectrum[~low].sum() / total) if total > 0 else 0.0


def _ink_mask(gray: Image) -> Image:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def rotate_image(img: Image, angle_deg: float, border: int = 255) -> Image:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    value = (border, border, border) if img.ndim == 3 else border
    out = cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=value
    )
    return out.astype(np.uint8)


def estimate_skew(gray: Image, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Skew angle (degrees, counter-clockwise positive) maximising the row-profile variance."""
    mask = _ink_mask(_downscale(gray, 800))
    if mask.sum() == 0:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        rotated = rotate_image(mask, float(angle), border=0)
        score = float(np.var(rotated.sum(axis=1, dtype=np.float64)))
        if score > best_score + 1e-6:
            best_angle, best_score = float(angle), score
    return -best_angle if abs(best_angle) > 1e-9 else 0.0


def estimate_rotation(gray: Image) -> int:
    """0 or 90: text lines produce stronger row than column profile variance."""
    mask = _ink_mask(_downscale(gray, 800))
    if mask.sum() == 0:
        return 0
    rows = float(np.var(mask.sum(axis=1, dtype=np.float64)))
    cols = float(np.var(mask.sum(axis=0, dtype=np.float64)))
    return 90 if cols > 1.5 * rows else 0


def blockiness(gray: Image) -> float:
    """JPEG 8x8 blocking: mean jump across block boundaries minus elsewhere."""
    g = gray.astype(np.float64)
    if g.shape[1] < 17:
        return 0.0
    diffs = np.abs(np.diff(g, axis=1))
    cols = np.arange(diffs.shape[1])
    boundary = diffs[:, (cols % 8) == 7].mean()
    inner = diffs[:, (cols % 8) != 7].mean()
    return float(max(0.0, boundary - inner))


# =========================================================================== defects


def detect_defects(
    gray: Image, low_illumination_mean: float
) -> tuple[dict[str, bool], dict[str, float]]:
    small = _downscale(gray, 1000)
    h, w = small.shape
    details: dict[str, float] = {}

    tiles = small[: h - h % 32, : w - w % 32].reshape(h // 32, 32, w // 32, 32).mean(axis=(1, 3))
    details["black_tile_ratio"] = float((tiles < 25).mean()) if tiles.size else 0.0

    mask = _ink_mask(small)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    details["tiny_component_ratio"] = float((areas < 4).mean()) if areas.size else 0.0
    details["specks_per_mpx"] = float((areas <= 2).sum() / (h * w / 1e6)) if areas.size else 0.0

    background = cv2.medianBlur(cv2.dilate(small, np.ones((15, 15), np.uint8)), 21)
    details["background_range"] = float(
        np.percentile(background, 95) - np.percentile(background, 5)
    )
    details["midgrey_ratio"] = float(((small > 110) & (small < 200)).mean())
    details["mean_brightness"] = float(small.mean())

    defects = {
        "missing_regions": details["black_tile_ratio"] > 0.02,
        "broken_characters": details["tiny_component_ratio"] > 0.5,
        "shadow_regions": details["background_range"] > 60,
        "watermarks": details["midgrey_ratio"] > 0.25,
        "low_illumination": details["mean_brightness"] < low_illumination_mean,
        "background_noise": details["specks_per_mpx"] > 2000,
    }
    return defects, details


# =========================================================================== enhancement ops


def denoise(gray: Image) -> Image:
    return cv2.fastNlMeansDenoising(
        gray, None, h=12, templateWindowSize=7, searchWindowSize=21
    ).astype(np.uint8)


def sharpen(gray: Image) -> Image:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.8, blur, -0.8, 0).astype(np.uint8)


def enhance_contrast(gray: Image) -> Image:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray).astype(np.uint8)


def correct_brightness(gray: Image, target: float = 200.0) -> Image:
    mean = max(1.0, float(gray.mean()))
    gamma = math.log(target / 255.0) / math.log(mean / 255.0) if mean < 254 else 1.0
    gamma = min(3.0, max(0.33, gamma))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut).astype(np.uint8)


def remove_background_noise(gray: Image) -> Image:
    return cv2.medianBlur(gray, 3).astype(np.uint8)


def suppress_shadows(gray: Image) -> Image:
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 21)
    diff = 255 - cv2.absdiff(gray, background)
    return as_u8(cv2.normalize(diff, np.zeros_like(diff), 0, 255, cv2.NORM_MINMAX))


def normalize_intensity(gray: Image) -> Image:
    return as_u8(cv2.normalize(gray, np.zeros_like(gray), 0, 255, cv2.NORM_MINMAX))


class SuperResolution(ABC):
    """Pluggable super-resolution backend."""

    @abstractmethod
    def upscale(self, img: Image) -> Image: ...


class BicubicSuperResolution(SuperResolution):
    def __init__(self, scale: int = 2) -> None:
        self.scale = scale

    def upscale(self, img: Image) -> Image:
        h, w = img.shape[:2]
        return cv2.resize(
            img, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC
        ).astype(np.uint8)


class OpenCVDnnSuperResolution(SuperResolution):  # pragma: no cover - needs contrib + model
    def __init__(self, model_path: str, scale: int = 2) -> None:
        from cv2 import dnn_superres  # type: ignore[attr-defined]

        name = Path(model_path).stem.split("_")[0].lower()
        self._sr = dnn_superres.DnnSuperResImpl_create()
        self._sr.readModel(model_path)
        self._sr.setModel(name, scale)

    def upscale(self, img: Image) -> Image:
        return as_u8(self._sr.upsample(img))
