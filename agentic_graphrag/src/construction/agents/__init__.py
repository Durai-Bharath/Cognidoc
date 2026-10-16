"""The 15 Stage 2 agents (A1-A15 in Table 1 of *Construction.pdf*)."""

from construction.agents.chunking import SemanticChunkingAgent
from construction.agents.confidence_evaluation import ConfidenceEvaluationAgent
from construction.agents.conflict_detection import ConflictDetectionAgent
from construction.agents.construction_validation import ConstructionValidationAgent
from construction.agents.coreference import CoreferenceResolutionAgent
from construction.agents.decision import DecisionAgent
from construction.agents.embedding import EmbeddingAgent
from construction.agents.entity_resolution import EntityResolutionAgent
from construction.agents.graph_crud import GraphCRUDAgent
from construction.agents.health_audit import GraphHealthAuditAgent
from construction.agents.ingestion import DocumentIngestionAgent
from construction.agents.provenance import ProvenanceAgent
from construction.agents.relation_normalization import RelationNormalizationAgent
from construction.agents.triple_extraction import TripleExtractionAgent
from construction.agents.vector_index import VectorIndexAgent

ALL_AGENTS = [
    DocumentIngestionAgent,  # A1 / Alg. 2
    SemanticChunkingAgent,  # A2 / Alg. 3
    TripleExtractionAgent,  # A3 / Alg. 4
    CoreferenceResolutionAgent,  # A4 / Alg. 5
    ConfidenceEvaluationAgent,  # A5 / Alg. 6
    EntityResolutionAgent,  # A6 / Alg. 7
    RelationNormalizationAgent,  # A7 / Alg. 8
    ConflictDetectionAgent,  # A8 / Alg. 9
    DecisionAgent,  # A9 / Alg. 10
    GraphCRUDAgent,  # A10 / Alg. 11
    ProvenanceAgent,  # A11 / Alg. 12
    EmbeddingAgent,  # A12 / Alg. 13
    VectorIndexAgent,  # A13 / Alg. 14
    ConstructionValidationAgent,  # A14 / Alg. 15
    GraphHealthAuditAgent,  # A15 / Alg. 16
]

__all__ = [
    "ALL_AGENTS",
    "ConfidenceEvaluationAgent",
    "ConflictDetectionAgent",
    "ConstructionValidationAgent",
    "CoreferenceResolutionAgent",
    "DecisionAgent",
    "DocumentIngestionAgent",
    "EmbeddingAgent",
    "EntityResolutionAgent",
    "GraphCRUDAgent",
    "GraphHealthAuditAgent",
    "ProvenanceAgent",
    "RelationNormalizationAgent",
    "SemanticChunkingAgent",
    "TripleExtractionAgent",
    "VectorIndexAgent",
]
