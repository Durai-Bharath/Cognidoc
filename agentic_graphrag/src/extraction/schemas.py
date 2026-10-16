"""Pydantic v2 payloads exchanged between the Stage 1 agents.

Names follow the pseudocode symbols: P (parsed stream), T (type), C (category),
Q (OCR quality report), E (enhanced images), L (layout), R (reading order),
S (section hierarchy), M_s (section memory), and so on.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.validation import ValidationOutcome

# =========================================================================== geometry


class BBox(BaseModel):
    """B = (x, y, w, h) in page points (PDF) or pixels (images); origin top-left."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    def intersection(self, other: BBox) -> float:
        iw = min(self.x1, other.x1) - max(self.x, other.x)
        ih = min(self.y1, other.y1) - max(self.y, other.y)
        return max(0.0, iw) * max(0.0, ih)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def h_overlap(self, other: BBox) -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x, other.x))

    def v_overlap(self, other: BBox) -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y, other.y))

    def union(self, other: BBox) -> BBox:
        x, y = min(self.x, other.x), min(self.y, other.y)
        return BBox(x=x, y=y, w=max(self.x1, other.x1) - x, h=max(self.y1, other.y1) - y)

    @classmethod
    def from_corners(cls, x0: float, y0: float, x1: float, y1: float) -> BBox:
        return cls(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


# =========================================================================== Alg. 2 Router


class DocumentType(StrEnum):
    PDF = "PDF"
    DOCX = "DOCX"
    IMAGE = "IMAGE"


class RawBlock(BaseModel):
    """A content block produced by a parser or by OCR, before layout analysis."""

    block_id: str
    page: int
    bbox: BBox
    kind: Literal["text", "image", "table"] = "text"
    text: str = ""
    font_size: float | None = None
    bold: bool = False
    style: str | None = Field(None, description="DOCX paragraph style name, if any.")
    table_cells: list[list[str | None]] | None = None
    image_path: str | None = None
    source: Literal["native", "ocr", "docx"] = "native"
    confidence: float | None = Field(None, description="OCR confidence of the block (0-1).")


class PageData(BaseModel):
    page_number: int
    width: float
    height: float
    rotation: int = 0
    blocks: list[RawBlock] = Field(default_factory=list)
    has_text_layer: bool = True
    image_path: str | None = Field(None, description="Rasterised page, for OCR pages.")


class DocumentProperties(BaseModel):
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    page_count: int = 0
    language: str = "und"
    file_size_bytes: int = 0
    mime_type: str = ""
    extension: str = ""


class ParsedDocument(BaseModel):
    """P: parsed document stream."""

    document_id: str
    source_path: str
    doc_type: DocumentType
    properties: DocumentProperties
    pages: list[PageData]
    raw_text: str = ""
    removed_objects: list[str] = Field(default_factory=list)
    ocr_pages: list[int] = Field(default_factory=list, description="Pages requiring OCR.")
    workdir: str

    @property
    def requires_ocr(self) -> bool:
        return bool(self.ocr_pages)

    def page(self, number: int) -> PageData:
        return next(p for p in self.pages if p.page_number == number)


class RouterInput(BaseModel):
    path: str
    workdir: str


class RouterOutput(BaseModel):
    parsed: ParsedDocument
    doc_type: DocumentType


# =========================================================================== Alg. 3


class DocumentFeatures(BaseModel):
    title: str | None = None
    table_of_contents: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    table_count: int = 0
    figure_count: int = 0
    layout_stats: dict[str, float] = Field(default_factory=dict)


class ClassificationInput(BaseModel):
    parsed: ParsedDocument


class ClassificationResult(BaseModel):
    """C: document category plus the evidence behind it."""

    category: str
    predicted_category: str
    confidence: float
    scores: dict[str, float]
    features: DocumentFeatures
    feature_vector: list[float] = Field(default_factory=list)
    flagged: bool = False
    based_on_text: bool = True


# =========================================================================== Alg. 4 / 5


class OCRWord(BaseModel):
    text: str
    bbox: BBox
    confidence: float
    block: int = 0
    line: int = 0


class OCRPageResult(BaseModel):
    words: list[OCRWord] = Field(default_factory=list)
    mean_confidence: float = 0.0


class PageQuality(BaseModel):
    page: int
    metrics: dict[str, float] = Field(description="Raw metric values.")
    normalized: dict[str, float] = Field(description="m_k in [0, 1].")
    defects: dict[str, bool]
    defect_details: dict[str, float] = Field(default_factory=dict)
    score: float = Field(description="Q_i = Σ w_k m_k")
    status: Literal["VALID", "LOW_QUALITY"]
    ocr_confidence: float


class OCRQualityInput(BaseModel):
    parsed: ParsedDocument


class OCRQualityReport(BaseModel):
    """Q plus I': the validated document (OCR text blocks filled in)."""

    pages: list[PageQuality]
    document_score: float
    low_quality_pages: list[int]
    parsed: ParsedDocument


class PageEnhancement(BaseModel):
    page: int
    degradations: list[str]
    operations: list[str]
    confidence_before: float
    confidence_after: float
    delta: float
    accepted: bool
    image_path: str


class EnhancementInput(BaseModel):
    parsed: ParsedDocument
    quality: OCRQualityReport


class EnhancementReport(BaseModel):
    """E: enhanced image set (accepted pages swapped into ``parsed``)."""

    pages: list[PageEnhancement]
    parsed: ParsedDocument


# =========================================================================== Alg. 6-8


class ElementType(StrEnum):
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SECTION_HEADING = "section_heading"
    SUBSECTION_HEADING = "subsection_heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    EQUATION = "equation"
    ALGORITHM = "algorithm"
    REFERENCE = "reference"
    FOOTNOTE = "footnote"


FURNITURE = {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER}
OBJECT_TYPES = {ElementType.TABLE, ElementType.FIGURE, ElementType.EQUATION, ElementType.ALGORITHM}


class LayoutElement(BaseModel):
    element_id: str
    type: ElementType
    page: int
    bbox: BBox
    text: str = ""
    font_size: float | None = None
    column: int | None = None
    reading_priority: float | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class PageLayout(BaseModel):
    page: int
    width: float
    height: float
    orientation: Literal["portrait", "landscape"]
    layout_type: Literal["single", "double", "multi"]
    columns: list[tuple[float, float]] = Field(default_factory=list)
    margins: dict[str, float] = Field(default_factory=dict)
    whitespace: list[BBox] = Field(default_factory=list)
    elements: list[LayoutElement] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class LayoutInput(BaseModel):
    parsed: ParsedDocument


class LayoutMap(BaseModel):
    """L: document layout representation."""

    pages: list[PageLayout]
    body_font_size: float = 10.0

    def all_elements(self) -> list[LayoutElement]:
        return [e for p in self.pages for e in p.elements]

    @property
    def confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.validation.pass_ratio for p in self.pages) / len(self.pages)


class Continuation(BaseModel):
    from_id: str
    to_id: str
    kind: Literal["paragraph", "table", "figure"]


class ReadingOrderInput(BaseModel):
    layout: LayoutMap


class OrderedDocument(BaseModel):
    """R: elements in logical reading order plus their associations."""

    elements: list[LayoutElement]
    caption_links: dict[str, str] = Field(default_factory=dict, description="caption → object")
    equation_context: dict[str, str] = Field(default_factory=dict, description="eq → paragraph")
    footnote_links: dict[str, str] = Field(default_factory=dict, description="footnote → elem")
    continuations: list[Continuation] = Field(default_factory=list)
    page_layout_types: dict[int, str] = Field(default_factory=dict)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class Section(BaseModel):
    section_id: str
    number: str | None = None
    title: str
    level: int = 1
    heading_element_id: str | None = None
    elements: list[LayoutElement] = Field(default_factory=list)
    subsections: list[Section] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    stats: dict[str, int] = Field(default_factory=dict)
    object_ids: dict[str, list[str]] = Field(default_factory=dict)

    def iter_sections(self) -> Iterator[Section]:
        yield self
        for sub in self.subsections:
            yield from sub.iter_sections()

    def all_elements(self) -> list[LayoutElement]:
        out = list(self.elements)
        for sub in self.subsections:
            out.extend(sub.all_elements())
        return out

    def elements_of(self, *types: ElementType) -> list[LayoutElement]:
        return [e for e in self.all_elements() if e.type in types]

    def text(self) -> str:
        """Section text (headings are elements too), joining cross-page continuations."""
        parts: list[str] = []
        for sec in self.iter_sections():
            for el in sec.elements:
                if not el.text:
                    continue
                if el.attrs.get("continued_from") and parts:
                    parts[-1] = (
                        parts[-1].rstrip("-") + ("" if parts[-1].endswith("-") else " ") + el.text
                    )
                else:
                    parts.append(el.text)
        return "\n\n".join(parts)


class SectionIdentificationInput(BaseModel):
    ordered: OrderedDocument
    page_count: int


class SectionHierarchy(BaseModel):
    """S: hierarchical section representation."""

    document_title: str | None = None
    sections: list[Section]
    furniture: list[LayoutElement] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True

    def iter_all(self) -> Iterator[Section]:
        for s in self.sections:
            yield from s.iter_sections()

    def find(self, section_id: str) -> Section | None:
        return next((s for s in self.iter_all() if s.section_id == section_id), None)

    def element_section(self) -> dict[str, str]:
        """element_id → id of the top-level section that contains it."""
        out: dict[str, str] = {}
        for top in self.sections:
            for el in top.all_elements():
                out[el.element_id] = top.section_id
        return out


# =========================================================================== per-section input


class SectionTask(BaseModel):
    """Input to the per-section agents (Alg. 9-19): S_i plus the whole hierarchy S."""

    section: Section
    hierarchy: SectionHierarchy
    ordered: OrderedDocument
    page_count: int
    is_last_section: bool = False


# =========================================================================== Alg. 9


class Chunk(BaseModel):
    chunk_id: str
    parent_section: str
    order: int
    text: str
    unit_ids: list[int] = Field(default_factory=list)
    previous_chunk: str | None = None
    next_chunk: str | None = None
    page_start: int
    page_end: int
    token_count: int
    summary: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class ChunkEdge(BaseModel):
    source: str
    target: str
    kind: Literal["next", "semantic"]
    weight: float


class RetrievalIndex(BaseModel):
    ids: list[str] = Field(default_factory=list)
    vectors: list[list[float]] = Field(default_factory=list)


class SectionMemory(BaseModel):
    """M_s: section memory representation."""

    section_id: str
    token_count: int
    stored_whole: bool
    chunks: list[Chunk]
    edges: list[ChunkEdge] = Field(default_factory=list)
    summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    index: RetrievalIndex = Field(default_factory=RetrievalIndex)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    reconstructed: bool = False
    approved: bool = True


# =========================================================================== Alg. 10-12


class MergedCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


class StructuredTable(BaseModel):
    table_id: str
    element_id: str = ""
    number: str | None = None
    caption: str | None = None
    page: int
    bbox: BBox
    section_id: str
    headers: list[str]
    rows: list[list[str]]
    normalized_rows: list[list[str | float | None]] = Field(default_factory=list)
    n_rows: int
    n_cols: int
    merged_cells: list[MergedCell] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    key_attributes: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    description: str = ""
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class TableRepository(BaseModel):
    """T: structured tables of one section."""

    section_id: str
    tables: list[StructuredTable] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


class FigureType(StrEnum):
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture_diagram"
    BLOCK = "block_diagram"
    CHART = "chart_graph"
    IMAGE = "image"
    PIPELINE = "pipeline"
    NETWORK = "network_topology"
    MISC = "miscellaneous"


class FigureObject(BaseModel):
    id: str = ""
    label: str = ""
    kind: str = ""


class FigureConnector(BaseModel):
    source: str = ""
    target: str = ""
    label: str = ""


class StructuredFigure(BaseModel):
    figure_id: str
    element_id: str = ""
    number: str | None = None
    caption: str | None = None
    page: int
    bbox: BBox
    section_id: str
    image_path: str | None = None
    figure_type: FigureType = FigureType.MISC
    objects: list[FigureObject] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    legends: list[str] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    connectors: list[FigureConnector] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    workflow: list[str] = Field(default_factory=list)
    description: str = ""
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class FigureRepository(BaseModel):
    section_id: str
    figures: list[StructuredFigure] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


class EquationType(StrEnum):
    ALGEBRAIC = "algebraic"
    STATISTICAL = "statistical"
    OPTIMIZATION = "optimization"
    DIFFERENTIAL = "differential"
    LOGICAL = "logical"
    MATRIX = "matrix"
    MISC = "miscellaneous"


class EquationVariable(BaseModel):
    symbol: str = ""
    definition: str = ""


class StructuredEquation(BaseModel):
    equation_id: str
    element_id: str = ""
    number: str | None = None
    latex: str
    source_text: str = ""
    page: int
    bbox: BBox
    section_id: str
    equation_type: EquationType = EquationType.MISC
    variables: list[EquationVariable] = Field(default_factory=list)
    constants: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    description: str = ""
    summary: str = ""
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class EquationRepository(BaseModel):
    section_id: str
    equations: list[StructuredEquation] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


# =========================================================================== Alg. 13-14


class CitationType(StrEnum):
    SECTION = "section"
    SUBSECTION = "subsection"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    ALGORITHM = "algorithm"
    APPENDIX = "appendix"
    PAGE = "page"


class Citation(BaseModel):
    citation_id: str
    citation_type: CitationType
    label: str = Field(description="The identifier as written, e.g. '3' in 'Figure 3'.")
    text: str = Field(description="Matched citation text.")
    source_section: str
    source_element: str
    offset: int = Field(0, description="Character offset of the match in the source element.")
    target_object: str | None = None
    target_page: int | None = None
    resolved: bool = False
    valid_numbering: bool = True


class CitationRepository(BaseModel):
    """C: internal citation repository (for one section)."""

    section_id: str
    citations: list[Citation] = Field(default_factory=list)
    referenced: dict[str, list[str]] = Field(default_factory=dict)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class CrossRefInput(BaseModel):
    task: SectionTask
    citations: CitationRepository
    prior_references: list[CrossReference] = Field(
        default_factory=list, description="Validated references from earlier sections."
    )


class CrossReference(BaseModel):
    reference_id: str
    citation_id: str
    citation_type: CitationType
    source_section: str
    source_element: str
    target_object: str | None
    target_section: str | None = None
    status: Literal["resolved", "unresolved"]
    errors: list[str] = Field(default_factory=list)
    suggestion: str | None = None


class CrossRefRepository(BaseModel):
    """R (Alg. 14): validated cross-references."""

    section_id: str
    references: list[CrossReference] = Field(default_factory=list)
    forward: dict[str, list[str]] = Field(default_factory=dict, description="source → targets")
    backward: dict[str, list[str]] = Field(default_factory=dict, description="target → sources")
    orphan_objects: list[str] = Field(default_factory=list)
    circular_references: list[list[str]] = Field(default_factory=list)
    report: dict[str, int] = Field(default_factory=dict)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


# =========================================================================== Alg. 15-17


class EntityType(StrEnum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    SOFTWARE = "Software/Framework"
    HARDWARE = "Hardware"
    DATASET = "Dataset"
    MODEL = "Model"
    API_TOOL = "API/Tool"
    STANDARD = "Standard/Protocol"
    DOMAIN_TERM = "Domain-specific term"


class Entity(BaseModel):
    entity_id: str
    type: EntityType
    name: str
    normalized_name: str
    description: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    context: str = ""
    section_id: str
    page: int


class EntityRepository(BaseModel):
    section_id: str
    entities: list[Entity] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class Abbreviation(BaseModel):
    abbreviation: str
    expansion: str
    section_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    occurrences: int = 0
    description: str = ""


class AbbreviationInput(BaseModel):
    task: SectionTask
    existing: dict[str, Abbreviation] = Field(default_factory=dict)


class AbbreviationRepository(BaseModel):
    """A: document-wide abbreviation repository after processing this section."""

    section_id: str
    abbreviations: dict[str, Abbreviation] = Field(default_factory=dict)
    conflicts: dict[str, list[str]] = Field(default_factory=dict)
    undefined: list[str] = Field(default_factory=list)
    unused: list[str] = Field(default_factory=list)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    approved: bool = True


class SectionSummary(BaseModel):
    section_id: str
    summary: str
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    important_entities: list[str] = Field(default_factory=list)
    domain_terms: list[str] = Field(default_factory=list)
    key_sentences: list[str] = Field(default_factory=list)
    summary_length: int = 0
    quality_score: float = 0.0
    confidence: float = 0.0
    used_fallback: bool = False
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


class SummarizationInput(BaseModel):
    task: SectionTask
    memory: SectionMemory | None = None


# =========================================================================== Alg. 18


class GeneratedItem(BaseModel):
    content_id: str
    kind: Literal["summary", "abstract", "table", "figure", "equation", "entity"]
    text: str
    source: str


class HallucinationInput(BaseModel):
    section_id: str
    items: list[GeneratedItem]


class HallucinationItem(BaseModel):
    content_id: str
    kind: str
    status: Literal["Verified", "Regenerated", "Flagged"]
    similarity: float
    hscore: float
    issues: list[str] = Field(default_factory=list)
    regenerations: int = 0
    final_text: str


class HallucinationReport(BaseModel):
    """H: hallucination validation report."""

    section_id: str
    items: list[HallucinationItem] = Field(default_factory=list)

    @property
    def replacements(self) -> dict[str, str]:
        return {i.content_id: i.final_text for i in self.items if i.status == "Regenerated"}


# =========================================================================== Alg. 19


class ConfidenceLevel(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ObjectConfidence(BaseModel):
    object_id: str
    kind: Literal["section", "table", "figure", "equation"]
    score: float
    level: ConfidenceLevel
    components: dict[str, float]
    weight: float = 1.0


class ConfidenceInput(BaseModel):
    section_id: str
    component_scores: dict[str, dict[str, float | None]] = Field(
        description="object_id → {agent name → c_k or None if not applicable}"
    )
    object_kinds: dict[str, Literal["section", "table", "figure", "equation"]]
    section_weight: float = 1.0
    prior_sections: list[ObjectConfidence] = Field(default_factory=list)


class ConfidenceRepository(BaseModel):
    section_id: str
    objects: list[ObjectConfidence] = Field(default_factory=list)
    document_score: float = 0.0
    document_level: ConfidenceLevel = ConfidenceLevel.LOW
    recomputed: bool = False
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


# =========================================================================== Alg. 20-21


class SectionResults(BaseModel):
    """Everything the per-section loop produced for one top-level section."""

    section_id: str
    memory: SectionMemory | None = None
    tables: TableRepository | None = None
    figures: FigureRepository | None = None
    equations: EquationRepository | None = None
    citations: CitationRepository | None = None
    cross_references: CrossRefRepository | None = None
    entities: EntityRepository | None = None
    summary: SectionSummary | None = None
    hallucination: HallucinationReport | None = None
    confidence: ConfidenceRepository | None = None


class FusionInput(BaseModel):
    parsed: ParsedDocument
    classification: ClassificationResult
    quality: OCRQualityReport | None = None
    enhancement: EnhancementReport | None = None
    layout: LayoutMap
    ordered: OrderedDocument
    hierarchy: SectionHierarchy
    results: dict[str, SectionResults]
    abbreviations: AbbreviationRepository | None = None
    review_flags: int = 0


Section.model_rebuild()
CrossRefInput.model_rebuild()
