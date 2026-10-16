"""Stage 1 output contracts: metadata file M, structured JSON J, validation report V.

The JSON Schemas in ``schemas/`` are generated from these models
(``make schemas``) and are what Stage 2 validates against.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


ConfidenceLevelName = Literal["High", "Medium", "Low"]

# =========================================================================== shared


class DocumentInfo(_Out):
    document_id: str
    document_type: Literal["PDF", "DOCX", "IMAGE"]
    document_category: str
    category_confidence: float
    title: str | None
    author: str | None
    creation_date: str | None
    language: str
    total_pages: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    mime_type: str
    source_file: str
    processing_timestamp: str


class DocumentStatistics(_Out):
    total_sections: int = Field(ge=0)
    total_subsections: int = Field(ge=0)
    total_tables: int = Field(ge=0)
    total_figures: int = Field(ge=0)
    total_equations: int = Field(ge=0)
    total_entities: int = Field(ge=0)
    total_abbreviations: int = Field(ge=0)
    total_cross_references: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_words: int = Field(ge=0)


# =========================================================================== M


class SectionMetadata(_Out):
    section_id: str
    number: str | None
    title: str
    level: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    summary: str
    keywords: list[str]
    confidence_score: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevelName
    subsection_ids: list[str]
    table_ids: list[str]
    figure_ids: list[str]
    equation_ids: list[str]
    entity_ids: list[str]
    abbreviations: list[str]
    cross_reference_ids: list[str]


class QualityInfo(_Out):
    ocr_applied: bool
    ocr_quality_score: float | None
    enhanced_pages: list[int]
    layout_confidence: float = Field(ge=0, le=1)
    document_confidence: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevelName


class MetadataFile(_Out):
    """M: document-level metadata."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    document: DocumentInfo
    sections: list[SectionMetadata]
    statistics: DocumentStatistics
    quality: QualityInfo
    review_flags: int = Field(ge=0)


# =========================================================================== J


class BBoxJSON(_Out):
    x: float
    y: float
    w: float
    h: float


class TableJSON(_Out):
    table_id: str
    number: str | None
    caption: str | None
    page: int
    bbox: BBoxJSON
    headers: list[str]
    rows: list[list[str]]
    units: dict[str, str]
    key_attributes: list[str]
    description: str
    summary: str
    keywords: list[str]
    approved: bool


class FigureObjectJSON(_Out):
    id: str
    label: str
    kind: str


class FigureConnectorJSON(_Out):
    source: str
    target: str
    label: str


class FigureJSON(_Out):
    figure_id: str
    number: str | None
    caption: str | None
    page: int
    bbox: BBoxJSON
    figure_type: str
    objects: list[FigureObjectJSON]
    connectors: list[FigureConnectorJSON]
    relationships: list[str]
    workflow: list[str]
    legends: list[str]
    description: str
    summary: str
    keywords: list[str]
    image_path: str | None
    approved: bool


class EquationVariableJSON(_Out):
    symbol: str
    definition: str


class EquationJSON(_Out):
    equation_id: str
    number: str | None
    latex: str
    page: int
    bbox: BBoxJSON
    equation_type: str
    variables: list[EquationVariableJSON]
    description: str
    approved: bool


class EntityJSON(_Out):
    entity_id: str
    type: str
    name: str
    description: str
    page: int


class AbbreviationJSON(_Out):
    abbreviation: str
    expansion: str
    occurrences: int


class CrossReferenceJSON(_Out):
    reference_id: str
    citation_type: str
    source_element: str
    target_object: str | None
    target_section: str | None
    status: Literal["resolved", "unresolved"]
    suggestion: str | None


class ChunkJSON(_Out):
    chunk_id: str
    order: int
    text: str
    page_start: int
    page_end: int
    token_count: int
    previous_chunk: str | None
    next_chunk: str | None
    summary: str


class SummaryJSON(_Out):
    summary: str
    abstract: str
    key_concepts: list[str]
    important_entities: list[str]
    confidence: float
    verification_status: Literal["Verified", "Regenerated", "Flagged", "Unchecked"]


class ConfidenceJSON(_Out):
    score: float = Field(ge=0, le=1)
    level: ConfidenceLevelName
    components: dict[str, float]


class SubsectionJSON(_Out):
    section_id: str
    number: str | None
    title: str
    level: int
    page_start: int
    page_end: int
    text: str


class SectionJSON(_Out):
    section_id: str
    number: str | None
    title: str
    level: int
    page_start: int
    page_end: int
    text: str
    keywords: list[str]
    summary: SummaryJSON
    confidence: ConfidenceJSON
    subsections: list[SubsectionJSON]
    tables: list[TableJSON]
    figures: list[FigureJSON]
    equations: list[EquationJSON]
    entities: list[EntityJSON]
    abbreviations: list[AbbreviationJSON]
    cross_references: list[CrossReferenceJSON]
    chunks: list[ChunkJSON]


class RelationshipJSON(_Out):
    source: str
    target: str
    type: Literal["contains", "references", "continues", "chunk_next", "chunk_similar", "defines"]


class DocumentJSON(_Out):
    """J: structured JSON document."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    document: DocumentInfo
    sections: list[SectionJSON]
    abbreviations: list[AbbreviationJSON]
    relationships: list[RelationshipJSON]
    statistics: DocumentStatistics


# =========================================================================== V


class CheckJSON(_Out):
    name: str
    passed: bool
    severity: Literal["error", "warning"]
    reasons: list[str]


class ValidationCategory(_Out):
    name: str
    weight: float = Field(ge=0)
    score: float = Field(ge=0, le=1)
    checks: list[CheckJSON]


class ValidationStatistics(_Out):
    validation_accuracy: float = Field(ge=0, le=1)
    total_errors: int = Field(ge=0)
    warnings: int = Field(ge=0)
    passed_checks: int = Field(ge=0)
    total_checks: int = Field(ge=0)


class ValidationReport(_Out):
    """V: global validation report."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    document_id: str
    generated_at: str
    approved: bool
    quality_score: float = Field(ge=0, le=1)
    categories: list[ValidationCategory]
    statistics: ValidationStatistics
    recommendations: list[str]
    agent_errors: list[str]
