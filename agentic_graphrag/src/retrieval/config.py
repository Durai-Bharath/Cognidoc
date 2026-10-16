"""Typed Stage 3 configuration. Every threshold (τ), budget, step size and retry limit used
by the retrieval agents lives here with a documented default; ``retrieval.yaml`` overrides.

The LLM, embedder and store sections reuse the Stage 1/2 config classes unchanged, so one
set of provider and backend settings drives all three stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.config import PROJECT_ROOT, EmbedderConfig, ExtractionConfig, LLMConfig, PathsConfig
from construction.config import ConstructionConfig, StoresConfig

QUERY_TYPES: tuple[str, ...] = ("Local", "Global", "Hybrid", "Simple Lookup")
Escalation = Literal["increase_k", "increase_hops", "switch_to_hybrid", "widen_communities"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _per_type(values: dict[str, float], name: str) -> None:
    missing = set(QUERY_TYPES) - set(values)
    if missing:
        raise ValueError(f"{name}: missing query types {sorted(missing)}")
    if any(v < 0 for v in values.values()):
        raise ValueError(f"{name}: values must be non-negative")


DEFAULT_THEMATIC_CUES: list[str] = [
    "overall",
    "overview",
    "main themes",
    "themes",
    "theme",
    "in general",
    "across",
    "summarize",
    "summarise",
    "summary of",
    "big picture",
    "key topics",
    "what topics",
    "trends",
    "high-level",
    "high level",
    "broadly",
    "landscape",
    "all of the",
]


class ClassificationConfig(_Strict):
    use_llm: bool = Field(True, description="Ask the LLM to judge the A1 features.")
    entity_min_similarity: float = Field(
        0.75, ge=-1, le=1, description="Embedding cosine at which a query n-gram names a KG entity."
    )
    thematic_cues: list[str] = Field(
        default_factory=lambda: list(DEFAULT_THEMATIC_CUES),
        description="Phrases that signal thematic / community-level reasoning.",
    )


class RewritingConfig(_Strict):
    use_llm: bool = Field(True, description="Let the LLM polish the deterministic rewrite.")
    acronyms: dict[str, str] = Field(
        default_factory=dict,
        description="Extra acronym → expansion pairs (KG aliases are used too).",
    )
    fuzzy_cutoff: float = Field(
        0.85,
        ge=0,
        le=1,
        description="difflib ratio at which a misspelt word maps to KG vocabulary.",
    )
    max_history_turns: int = Field(6, ge=0, description="History turns used for references.")


class DecompositionConfig(_Strict):
    use_llm: bool = Field(True, description="Ask the LLM to decompose complex queries.")
    max_subqueries: int = Field(5, ge=1, description="Hard cap on subqueries.")


class VectorRetrievalConfig(_Strict):
    top_k: int = Field(5, ge=1, description="k: chunks returned per subquery.")
    entity_top_k: int = Field(5, ge=0, description="Entities returned per subquery.")
    min_score: float = Field(0.0, ge=-1, le=1, description="Hits below this cosine are dropped.")


class TraversalConfig(_Strict):
    hops: dict[str, float] = Field(
        default_factory=lambda: {"Local": 2.0, "Global": 1.0, "Hybrid": 2.0, "Simple Lookup": 0.0},
        description="Multi-hop depth per query type.",
    )
    seed_min_score: float = Field(0.2, description="Entity hits below this do not seed traversal.")
    max_paths: int = Field(20, ge=1, description="Reasoning paths kept (best first).")
    max_nodes: int = Field(60, ge=1, description="Traversal stops after visiting this many nodes.")
    path_decay: float = Field(0.85, gt=0, le=1, description="Score multiplier per extra hop.")

    @model_validator(mode="after")
    def _ok(self) -> Self:
        _per_type(self.hops, "traversal.hops")
        return self


class CommunityRetrievalConfig(_Strict):
    weights: dict[str, float] = Field(
        default_factory=lambda: {"Global": 1.0, "Hybrid": 0.6, "Local": 0.2, "Simple Lookup": 0.0},
        description="Weight of community evidence per query type (0 disables A6).",
    )
    top_n: int = Field(3, ge=1, description="Communities kept after ranking.")
    semantic_search: bool = Field(
        True, description="Also search the communities collection with the query embedding."
    )
    membership_weight: float = Field(
        0.3, ge=0, le=1, description="Share of the relevance taken from member overlap."
    )
    max_members_listed: int = Field(12, ge=1)

    @model_validator(mode="after")
    def _ok(self) -> Self:
        _per_type(self.weights, "community.weights")
        return self


class AggregationConfig(_Strict):
    max_items: int = Field(30, ge=1, description="Evidence items kept per subquery.")
    default_confidence: float = Field(
        1.0, ge=0, le=1, description="Graph confidence of a chunk with no scored mentions."
    )


class ContextConfig(_Strict):
    token_budget: int = Field(1800, ge=64, description="Token budget for the LLM context.")
    min_block_tokens: int = Field(24, ge=1, description="Smaller remainders are not truncated in.")


class AnswerConfig(_Strict):
    max_answer_sentences: int = Field(8, ge=1, description="Guidance given to the LLM.")


class CitationConfig(_Strict):
    inferred_min_support: float = Field(
        0.5, ge=0, le=1, description="Uncited statement → best block if support ≥ this."
    )


class ValidationConfig(_Strict):
    tau: float = Field(0.7, ge=0, le=1, description="τ: faithfulness score ≥ τ → Valid.")
    claim_support_threshold: float = Field(
        0.6, ge=0, le=1, description="A claim is supported when its support score ≥ this."
    )
    grounding_weight: float = Field(
        0.8, ge=0, le=1, description="Score = w·grounding + (1−w)·completeness."
    )
    use_llm: bool = Field(False, description="Add an LLM judge verdict per claim.")


class CritiqueConfig(_Strict):
    always_critique: bool = Field(False, description="Run A12 even when validation passed.")
    use_llm: bool = Field(True, description="Use the LLM for the independent reasoning pass.")
    can_invalidate: bool = Field(
        True, description="A major critique issue downgrades a Valid answer to Invalid."
    )


class FeedbackConfig(_Strict):
    penalty_step: float = Field(0.1, gt=0, le=1, description="Confidence decrease per failure.")
    reward_step: float = Field(
        0.02, ge=0, le=1, description="Confidence increase per verification."
    )
    floor: float = Field(0.05, ge=0, le=1, description="Confidence never drops below this.")
    ceiling: float = Field(0.99, ge=0, le=1, description="Confidence never rises above this.")
    flag_below: float = Field(0.4, ge=0, le=1, description="Flag elements penalised below this.")
    reinforce_verified: bool = Field(True, description="Reward elements behind verified answers.")
    persist_state: bool = Field(
        False, description="Save in-memory KG state after feedback (CLI sets this)."
    )

    @model_validator(mode="after")
    def _ok(self) -> Self:
        if self.floor > self.ceiling:
            raise ValueError("feedback.floor must be ≤ feedback.ceiling")
        return self


class RetryConfig(_Strict):
    max_retries: int = Field(3, ge=0, le=10, description="Hard limit on retries after attempt 1.")
    escalation: list[Escalation] = Field(
        default_factory=lambda: [  # type: ignore[arg-type]
            "increase_k",
            "increase_hops",
            "switch_to_hybrid",
            "widen_communities",
        ],
        description="Parameter change applied on retry i (cycled when exhausted).",
    )
    k_step: int = Field(5, ge=1)
    max_k: int = Field(40, ge=1)
    hop_step: int = Field(1, ge=1)
    max_hops: int = Field(4, ge=0)
    community_step: int = Field(2, ge=1)
    max_communities: int = Field(10, ge=1)
    min_community_weight: float = Field(
        0.5, ge=0, le=1, description="Floor applied to the community weight when widening."
    )

    @model_validator(mode="after")
    def _ok(self) -> Self:
        if not self.escalation:
            raise ValueError("retry.escalation must list at least one step")
        return self


class CacheConfig(_Strict):
    enabled: bool = Field(True, description="Use the semantic cache (A15).")
    similarity_threshold: float = Field(
        0.95, ge=-1, le=1, description="Query-embedding cosine at which a cached answer is reused."
    )
    max_entries: int = Field(500, ge=1, description="Oldest entries are evicted beyond this.")


class RetrievalPaths(_Strict):
    output_dir: str = Field("output/retrieval", description="Reports land in <output_dir>/reports.")
    state_dir: str = Field("output/kg_state", description="Stage 2 in-memory KG/VDB state.")
    review_queue: str = Field("output/review_queue.jsonl", description="Shared ReviewQueue.")
    cache_file: str | None = Field(
        "output/retrieval/cache.json", description="Semantic cache file (None = in memory)."
    )
    feedback_log: str | None = Field(
        "output/retrieval/feedback.jsonl", description="Retrieval feedback records (A13)."
    )
    schemas_dir: str = Field(str(PROJECT_ROOT / "schemas"))


class RetrievalConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    stores: StoresConfig = Field(default_factory=StoresConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    rewriting: RewritingConfig = Field(default_factory=RewritingConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    vector: VectorRetrievalConfig = Field(default_factory=VectorRetrievalConfig)
    traversal: TraversalConfig = Field(default_factory=TraversalConfig)
    community: CommunityRetrievalConfig = Field(default_factory=CommunityRetrievalConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    citation: CitationConfig = Field(default_factory=CitationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    critique: CritiqueConfig = Field(default_factory=CritiqueConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    parallel_subqueries: bool = Field(True, description="Run independent subqueries in parallel.")
    max_workers: int = Field(4, ge=1)
    paths: RetrievalPaths = Field(default_factory=RetrievalPaths)
    log_level: str = Field("INFO")

    def as_extraction_config(self) -> ExtractionConfig:
        """The Stage 1 config view that ``AgentContext.config`` expects."""
        return ExtractionConfig(
            llm=self.llm,
            embedder=self.embedder,
            paths=PathsConfig(
                output_dir=self.paths.output_dir,
                review_queue=self.paths.review_queue,
                schemas_dir=self.paths.schemas_dir,
            ),
            log_level=self.log_level,
        )

    def as_construction_config(self) -> ConstructionConfig:
        """The Stage 2 view used to open the same KG/VDB backends."""
        cfg = ConstructionConfig(llm=self.llm, embedder=self.embedder, stores=self.stores)
        cfg.paths.state_dir = self.paths.state_dir
        cfg.paths.review_queue = self.paths.review_queue
        return cfg


def load_retrieval_config(path: str | Path | None = None) -> RetrievalConfig:
    """Load and validate ``retrieval.yaml``; ``None`` returns the documented defaults."""
    if path is None:
        return RetrievalConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return RetrievalConfig.model_validate(data)
