"""The 21 Stage 1 agents, one per algorithm in the specification."""

from extraction.agents.abbreviation import AbbreviationResolutionAgent
from extraction.agents.citation_extraction import InternalCitationExtractionAgent
from extraction.agents.classification import DocumentClassificationAgent
from extraction.agents.confidence import ConfidenceScoringAgent
from extraction.agents.context_memory import ContextMemoryAgent
from extraction.agents.cross_reference import CrossReferenceValidationAgent
from extraction.agents.entity_extraction import EntityExtractionAgent
from extraction.agents.equation_analysis import EquationAnalysisAgent
from extraction.agents.figure_understanding import FigureUnderstandingAgent
from extraction.agents.global_validation import GlobalValidationAgent
from extraction.agents.hallucination import HallucinationDetectionAgent
from extraction.agents.image_enhancement import ImageEnhancementAgent
from extraction.agents.layout import LayoutDetectionAgent
from extraction.agents.metadata_fusion import MetadataFusionAgent
from extraction.agents.ocr_quality import OCRQualityAssessmentAgent
from extraction.agents.reading_order import ReadingOrderAgent
from extraction.agents.router import RouterAgent
from extraction.agents.section_identification import SectionIdentificationAgent
from extraction.agents.summarization import TextSummarizationAgent
from extraction.agents.table_analysis import TableAnalysisAgent

# Alg. 11 is named "Figure Understanding Agent"; Alg. 9 calls it "Figure Analysis Agent".
FigureAnalysisAgent = FigureUnderstandingAgent

ALL_AGENTS = [
    RouterAgent,  # 2
    DocumentClassificationAgent,  # 3
    OCRQualityAssessmentAgent,  # 4
    ImageEnhancementAgent,  # 5
    LayoutDetectionAgent,  # 6
    ReadingOrderAgent,  # 7
    SectionIdentificationAgent,  # 8
    ContextMemoryAgent,  # 9
    TableAnalysisAgent,  # 10
    FigureUnderstandingAgent,  # 11
    EquationAnalysisAgent,  # 12
    InternalCitationExtractionAgent,  # 13
    CrossReferenceValidationAgent,  # 14
    EntityExtractionAgent,  # 15
    AbbreviationResolutionAgent,  # 16
    TextSummarizationAgent,  # 17
    HallucinationDetectionAgent,  # 18
    ConfidenceScoringAgent,  # 19
    MetadataFusionAgent,  # 20
    GlobalValidationAgent,  # 21
]

__all__ = [
    "ALL_AGENTS",
    "AbbreviationResolutionAgent",
    "ConfidenceScoringAgent",
    "ContextMemoryAgent",
    "CrossReferenceValidationAgent",
    "DocumentClassificationAgent",
    "EntityExtractionAgent",
    "EquationAnalysisAgent",
    "FigureAnalysisAgent",
    "FigureUnderstandingAgent",
    "GlobalValidationAgent",
    "HallucinationDetectionAgent",
    "ImageEnhancementAgent",
    "InternalCitationExtractionAgent",
    "LayoutDetectionAgent",
    "MetadataFusionAgent",
    "OCRQualityAssessmentAgent",
    "ReadingOrderAgent",
    "RouterAgent",
    "SectionIdentificationAgent",
    "TableAnalysisAgent",
    "TextSummarizationAgent",
]
