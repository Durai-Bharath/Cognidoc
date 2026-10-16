"""Typed Stage 2 configuration. Every threshold (τ, δ), limit and backend choice lives here
with a documented default; ``construction.yaml`` overrides.

The LLM and embedder sections reuse the Stage 1 config classes unchanged, so one set of
provider settings drives both stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.config import PROJECT_ROOT, EmbedderConfig, ExtractionConfig, LLMConfig, PathsConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestionConfig(_Strict):
    require_metadata_match: bool = Field(
        True, description="M.document.document_id must equal J.document.document_id."
    )


class ChunkingConfig(_Strict):
    max_tokens: int = Field(400, ge=16, description="Re-split any chunk larger than this.")
    overlap_tokens: int = Field(
        60, ge=0, description="Tail of the previous chunk (same section) given as context."
    )
    include_tables: bool = Field(True, description="Emit one linearised chunk per table.")
    min_tokens: int = Field(3, ge=0, description="Drop chunks shorter than this.")


class TripleExtractionConfig(_Strict):
    max_triples_per_chunk: int = Field(40, ge=1, description="Truncate LLM output beyond this.")


class ConfidenceEvaluationConfig(_Strict):
    tau: float = Field(0.5, ge=0, le=1, description="τ: triples with confidence < τ are discarded.")
    ungrounded_penalty: float = Field(
        0.6,
        ge=0,
        le=1,
        description="Multiplier when the head or tail cannot be found in the chunk text.",
    )


class EntityResolutionConfig(_Strict):
    delta: float = Field(
        0.85, ge=-1, le=1, description="δ: cosine ≥ δ links a mention to an existing node."
    )
    top_k: int = Field(5, ge=1, description="Candidates fetched from the graph per mention.")
    require_type_compatibility: bool = Field(
        True, description="A typed mention only links to a node of the same (or Concept) type."
    )


class RelationNormalizationConfig(_Strict):
    llm_fallback: bool = Field(True, description="Ask the LLM when no synonym matches.")


class ValidationConfig(_Strict):
    max_entity_name_chars: int = Field(120, ge=8, description="Longer names are invalid.")


class AuditConfig(_Strict):
    low_confidence: float = Field(0.6, ge=0, le=1, description="Edges below this are flagged.")
    duplicate_similarity: float = Field(
        0.95, ge=-1, le=1, description="Two entity nodes this similar are likely duplicates."
    )
    stale_after_days: float = Field(
        365.0, gt=0, description="Edges not re-observed for this long are stale."
    )


class CommunityConfig(_Strict):
    enabled: bool = Field(True, description="Run CommunityBuilder after the health audit.")
    algorithm: Literal["louvain", "leiden"] = Field(
        "louvain", description="leiden requires the optional igraph + leidenalg packages."
    )
    resolution: float = Field(1.0, gt=0, description="Modularity resolution γ.")
    seed: int = Field(42, description="Random seed so partitions are reproducible.")
    min_size: int = Field(2, ge=1, description="Smaller communities get no summary node.")
    llm_summaries: bool = Field(True, description="Summarise with the LLM (else extractive).")
    max_members_in_prompt: int = Field(40, ge=1)
    max_edges_in_prompt: int = Field(60, ge=1)


class StoresConfig(_Strict):
    graph_backend: Literal["memory", "neo4j"] = Field("memory", description="GraphStore.")
    vector_backend: Literal["memory", "qdrant"] = Field("memory", description="VectorStore.")
    neo4j_uri: str = Field("bolt://localhost:7687", description="Overridden by $NEO4J_URI.")
    neo4j_user: str = Field("neo4j", description="Overridden by $NEO4J_USER.")
    neo4j_database: str = Field("neo4j")
    qdrant_url: str = Field("http://localhost:6333", description="Overridden by $QDRANT_URL.")
    collection_prefix: str = Field("graphrag_", description="Prefix for vector collections.")


class ConstructionPaths(_Strict):
    output_dir: str = Field("output/construction", description="Report and exports go here.")
    state_dir: str = Field(
        "output/kg_state", description="In-memory backends load/save KG and VDB here."
    )
    review_queue: str = Field("output/review_queue.jsonl", description="Shared ReviewQueue.")
    schemas_dir: str = Field(str(PROJECT_ROOT / "schemas"), description="Stage 1 schemas.")
    ontology: str = Field(str(PROJECT_ROOT / "ontology.yaml"), description="Ontology O.")


class ConstructionConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    triples: TripleExtractionConfig = Field(default_factory=TripleExtractionConfig)
    confidence: ConfidenceEvaluationConfig = Field(default_factory=ConfidenceEvaluationConfig)
    entity_resolution: EntityResolutionConfig = Field(default_factory=EntityResolutionConfig)
    relations: RelationNormalizationConfig = Field(default_factory=RelationNormalizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    stores: StoresConfig = Field(default_factory=StoresConfig)
    paths: ConstructionPaths = Field(default_factory=ConstructionPaths)
    log_level: str = Field("INFO")

    @model_validator(mode="after")
    def _ok(self) -> Self:
        if self.chunking.overlap_tokens >= self.chunking.max_tokens:
            raise ValueError("chunking.overlap_tokens must be < chunking.max_tokens")
        return self

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


def load_construction_config(path: str | Path | None = None) -> ConstructionConfig:
    """Load and validate ``construction.yaml``; ``None`` returns the documented defaults.

    Relative ``schemas_dir`` and ``ontology`` paths resolve against the config file.
    """
    if path is None:
        return ConstructionConfig()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    config = ConstructionConfig.model_validate(data)
    base = Path(path).resolve().parent
    for attr in ("schemas_dir", "ontology"):
        value = Path(getattr(config.paths, attr))
        if not value.is_absolute():
            setattr(config.paths, attr, str((base / value).resolve()))
    return config
