"""Typed configuration. Every threshold (τ), weight (w_k), ContextLimit and retry limit
used by the algorithms lives here with a documented default; ``config.yaml`` overrides.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_weights(weights: dict[str, float], name: str) -> None:
    if any(w < 0 or math.isnan(w) for w in weights.values()):
        raise ValueError(f"{name}: weights must be non-negative numbers")
    if sum(weights.values()) <= 0:
        raise ValueError(f"{name}: at least one weight must be positive")


# --------------------------------------------------------------------------- providers


class LLMConfig(_Strict):
    provider: Literal["anthropic", "azure_openai", "openrouter", "mock"] = Field(
        "anthropic", description="Chat/vision LLM provider."
    )
    model: str = Field(
        "claude-opus-5-5",
        description="Model id (Anthropic/OpenRouter) or deployment name (Azure). "
        "For azure_openai an empty value falls back to $AZURE_OPENAI_DEPLOYMENT.",
    )
    max_tokens: int = Field(4096, ge=64, description="Max output tokens per call.")
    effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        "medium", description="Anthropic output_config.effort for extraction calls."
    )
    anthropic_fallbacks: bool = Field(
        True, description="Send server-side refusal fallbacks ('default') on Anthropic calls."
    )
    temperature: float | None = Field(
        0.0, description="Sampling temperature for OpenAI-compatible providers (None = omit)."
    )
    max_retries: int = Field(3, ge=0, description="Retries on transient/invalid-JSON errors.")
    backoff_base_s: float = Field(1.0, ge=0, description="Exponential backoff base (s).")
    backoff_max_s: float = Field(30.0, ge=0, description="Backoff cap (s).")
    timeout_s: float = Field(120.0, gt=0, description="Per-request timeout (s).")


class EmbedderConfig(_Strict):
    provider: Literal["sentence_transformers", "openrouter", "openai", "fake"] = Field(
        "sentence_transformers", description="Embedding backend."
    )
    model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Embedding model.")
    batch_size: int = Field(32, ge=1)
    fake_dim: int = Field(256, ge=8, description="Dimension of the deterministic FakeEmbedder.")
    max_retries: int = Field(6, ge=0, description="Retries on rate limits / transient errors.")
    backoff_base_s: float = Field(2.0, ge=0, description="Exponential backoff base (s).")
    backoff_max_s: float = Field(
        60.0, ge=0, description="Backoff cap (s); free tiers reset per minute."
    )


class OCRConfig(_Strict):
    engine: Literal["tesseract", "mock"] = Field("tesseract", description="OCR engine.")
    language: str = Field("eng", description="Tesseract language code(s).")
    render_dpi: int = Field(300, ge=72, description="DPI used to rasterise scanned PDF pages.")


# --------------------------------------------------------------------------- agents


class RouterConfig(_Strict):
    max_file_size_mb: float = Field(200.0, gt=0, description="Larger files are inaccessible.")
    min_native_text_chars: int = Field(
        25, ge=0, description="PDF pages with fewer native characters are treated as scanned."
    )


DEFAULT_CLASS_DESCRIPTIONS: dict[str, str] = {
    "Standard Operating Procedure (SOP)": "standard operating procedure sop purpose scope "
    "responsibilities procedure steps shall must approval revision compliance work instruction",
    "User Manual": "user manual guide how to use features getting started controls interface "
    "operation instructions troubleshooting faq user",
    "Technical Report": "technical report analysis results evaluation methodology system "
    "design architecture performance findings conclusion engineering",
    "Installation Guide": "installation guide install setup prerequisites requirements "
    "configure deploy download package dependencies version step",
    "Maintenance Manual": "maintenance manual inspection service schedule replace lubricate "
    "repair parts preventive maintenance interval torque",
    "Machine Log / System Log": "log timestamp error warning info debug event system machine "
    "process status code trace message",
    "Incident Report": "incident report occurred root cause impact timeline corrective action "
    "investigation severity outage injury",
    "Test Report": "test report test case pass fail result expected actual verification "
    "validation coverage defect test plan",
    "Research Document": "abstract introduction related work method experiments results "
    "discussion conclusion references paper research study hypothesis",
    "Policy or Compliance Document": "policy compliance regulation must shall governance "
    "requirements audit controls obligations legal privacy security",
    "Invoice / Form": "invoice form bill amount total tax date due payment quantity price "
    "customer signature field",
    "Miscellaneous": "document general information notes miscellaneous content",
}


class ClassificationConfig(_Strict):
    tau: float = Field(
        0.20,
        ge=-1,
        le=1,
        description="τ: minimum cosine Conf(C) to accept the argmax class; below → Unknown.",
    )
    class_descriptions: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_CLASS_DESCRIPTIONS),
        description="Prototype text embedded for each of the 12 document classes.",
    )
    max_feature_chars: int = Field(4000, ge=200, description="Text budget for the feature text.")


DEFAULT_OCR_QUALITY_WEIGHTS: dict[str, float] = {
    "resolution": 0.10,
    "brightness": 0.08,
    "contrast": 0.12,
    "sharpness": 0.15,
    "noise": 0.10,
    "blur": 0.10,
    "skew": 0.07,
    "rotation": 0.05,
    "compression": 0.03,
    "ocr_confidence": 0.20,
}


class OCRQualityConfig(_Strict):
    tau_quality: float = Field(0.60, ge=0, le=1, description="τ_quality: Q_i ≥ τ → VALID.")
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_OCR_QUALITY_WEIGHTS),
        description="w_k for each normalised metric m_k (renormalised to sum 1).",
    )
    target_min_side_px: int = Field(1600, description="Shorter side at which resolution m=1.")
    sharpness_ref: float = Field(
        50.0, gt=0, description="Laplacian variance (of the pre-smoothed page) at which m=1."
    )
    contrast_ref: float = Field(
        120.0, gt=0, description="Paper-minus-ink mean grey level at which m=1."
    )
    target_brightness: float = Field(200.0, description="Ideal mean grey level of a page.")
    noise_ref: float = Field(12.0, gt=0, description="Residual noise sigma at which m=0.")
    blur_ref: float = Field(0.12, gt=0, description="High-frequency energy ratio at which m=1.")
    max_skew_deg: float = Field(10.0, gt=0, description="Skew at which m=0.")
    compression_ref: float = Field(8.0, gt=0, description="8x8 blockiness at which m=0.")
    low_illumination_mean: float = Field(90.0, description="Mean grey below → low illumination.")

    @model_validator(mode="after")
    def _weights_ok(self) -> Self:
        _check_weights(self.weights, "ocr_quality.weights")
        unknown = set(self.weights) - set(DEFAULT_OCR_QUALITY_WEIGHTS)
        if unknown:
            raise ValueError(f"ocr_quality.weights: unknown metrics {sorted(unknown)}")
        return self


class EnhancementConfig(_Strict):
    noise_threshold: float = Field(0.6, description="Denoise when noise metric m < this.")
    blur_threshold: float = Field(0.5, description="Sharpen when sharpness/blur m < this.")
    contrast_threshold: float = Field(0.5, description="Enhance contrast when m < this.")
    skew_threshold_deg: float = Field(0.5, ge=0, description="Deskew when |skew| > this.")
    min_side_px: int = Field(1000, description="Super-resolve when the shorter side < this.")
    super_resolution: Literal["bicubic", "opencv_dnn"] = Field(
        "bicubic", description="Pluggable super-resolution backend."
    )
    super_resolution_model: str | None = Field(
        None, description="Path to an OpenCV dnn_superres model (opencv_dnn backend)."
    )
    super_resolution_scale: int = Field(2, ge=2, le=4)


class LayoutConfig(_Strict):
    backend: Literal["heuristic", "docling"] = Field(
        "heuristic", description="Layout model behind the LayoutModel interface."
    )
    header_band: float = Field(0.07, gt=0, lt=0.5, description="Top fraction = header zone.")
    footer_band: float = Field(0.07, gt=0, lt=0.5, description="Bottom fraction = footer zone.")
    heading_size_ratio: float = Field(1.12, gt=1, description="Font ratio vs body → heading.")
    title_size_ratio: float = Field(1.45, gt=1, description="Font ratio vs body → title.")
    full_width_ratio: float = Field(0.6, gt=0, le=1, description="Width ratio → full-width.")
    overlap_iou_threshold: float = Field(0.3, gt=0, le=1, description="IoU → overlapping.")
    caption_max_gap_ratio: float = Field(
        0.12, gt=0, description="Max caption↔object vertical gap as fraction of page height."
    )
    whitespace_min_gap_ratio: float = Field(0.03, gt=0, description="Min gap = whitespace.")


class ContextMemoryConfig(_Strict):
    context_limit: int = Field(1500, ge=32, description="ContextLimit in tokens (Alg. 9).")
    max_chunk_tokens: int = Field(400, ge=16, description="Hard upper bound per chunk.")
    min_chunk_tokens: int = Field(60, ge=1, description="No semantic break before this size.")
    semantic_break_threshold: float = Field(
        0.15, description="Start a new chunk when sim(unit, chunk) < this."
    )
    link_threshold: float = Field(0.45, description="Add a semantic edge when cos ≥ this.")
    llm_chunk_summaries: bool = Field(False, description="Use the LLM for chunk summaries.")


class TableConfig(_Strict):
    max_missing_cell_ratio: float = Field(0.3, ge=0, le=1, description="Empty-cell tolerance.")


class FigureConfig(_Strict):
    use_vision: bool = Field(True, description="Send the figure crop to a vision LLM.")


class EntityConfig(_Strict):
    use_llm: bool = Field(True, description="Classify/describe NER candidates with the LLM.")
    max_candidates: int = Field(60, ge=1, description="Candidates sent to the LLM per section.")


class AbbreviationConfig(_Strict):
    max_abbr_len: int = Field(10, ge=2)


class SummarizationConfig(_Strict):
    max_summary_words: int = Field(160, ge=10, description="Length constraint.")
    key_sentences: int = Field(5, ge=1, description="Sentences kept by semantic ranking.")
    min_context_similarity: float = Field(0.35, description="sim(summary, section) floor.")
    min_keyword_coverage: float = Field(0.2, ge=0, le=1, description="Completeness floor.")
    redundancy_similarity: float = Field(0.95, description="Drop near-duplicate sentences.")


class HallucinationConfig(_Strict):
    tau: float = Field(0.65, ge=0, le=2, description="τ: HScore ≤ τ → Verified.")
    max_regenerations: int = Field(2, ge=0, description="Bounded regeneration retries.")
    unsupported_sentence_similarity: float = Field(
        0.2, description="A generated sentence below this max-sim to the source is unsupported."
    )


DEFAULT_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "ocr_quality": 0.10,
    "layout": 0.10,
    "section_identification": 0.10,
    "table": 0.10,
    "figure": 0.10,
    "equation": 0.10,
    "entity": 0.10,
    "summarization": 0.15,
    "hallucination": 0.15,
    "classification": 0.0,
    "abbreviation": 0.0,
}


class ConfidenceConfig(_Strict):
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_WEIGHTS),
        description="w_k per agent; renormalised over the scores present for each object.",
    )
    high_threshold: float = Field(0.8, ge=0, le=1, description="C ≥ this → High.")
    medium_threshold: float = Field(0.5, ge=0, le=1, description="C ≥ this → Medium.")
    required_components: list[str] = Field(
        default_factory=lambda: ["layout", "section_identification", "summarization"],
        description="Scores that must be present for a section (missing → validation fails).",
    )

    @model_validator(mode="after")
    def _ok(self) -> Self:
        _check_weights(self.weights, "confidence.weights")
        if self.medium_threshold > self.high_threshold:
            raise ValueError("confidence.medium_threshold must be ≤ high_threshold")
        return self


DEFAULT_GLOBAL_WEIGHTS: dict[str, float] = {
    "document_integrity": 0.25,
    "extracted_objects": 0.25,
    "metadata_consistency": 0.20,
    "json_schema": 0.15,
    "cross_document_consistency": 0.15,
}


class GlobalValidationConfig(_Strict):
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_GLOBAL_WEIGHTS),
        description="w_i for each validation category q_i in Q = Σ w_i q_i.",
    )
    approve_threshold: float = Field(0.8, ge=0, le=1, description="Q ≥ this (and no errors).")

    @model_validator(mode="after")
    def _ok(self) -> Self:
        _check_weights(self.weights, "global_validation.weights")
        return self


class PathsConfig(_Strict):
    output_dir: str = Field("output", description="Where M, J and the report are written.")
    review_queue: str = Field("output/review_queue.jsonl", description="ReviewQueue JSONL.")
    schemas_dir: str = Field(str(PROJECT_ROOT / "schemas"), description="Published schemas.")


class ExtractionConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    ocr_quality: OCRQualityConfig = Field(default_factory=OCRQualityConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    context_memory: ContextMemoryConfig = Field(default_factory=ContextMemoryConfig)
    table: TableConfig = Field(default_factory=TableConfig)
    figure: FigureConfig = Field(default_factory=FigureConfig)
    entity: EntityConfig = Field(default_factory=EntityConfig)
    abbreviation: AbbreviationConfig = Field(default_factory=AbbreviationConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    global_validation: GlobalValidationConfig = Field(default_factory=GlobalValidationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    log_level: str = Field("INFO")


def load_config(path: str | Path | None = None) -> ExtractionConfig:
    """Load and validate ``config.yaml``; ``None`` returns the documented defaults."""
    if path is None:
        return ExtractionConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    config = ExtractionConfig.model_validate(data)
    schemas = Path(config.paths.schemas_dir)
    if not schemas.is_absolute():  # published schemas live next to the config file
        config.paths.schemas_dir = str((Path(path).resolve().parent / schemas).resolve())
    return config
