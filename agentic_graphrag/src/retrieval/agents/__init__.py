"""The 15 Stage 3 agents (A1-A15 in Table 1 of *Retrieval.pdf*)."""

from retrieval.agents.answer_generation import AnswerGenerationAgent
from retrieval.agents.cache_manager import CacheManagerAgent
from retrieval.agents.citation_mapping import CitationMappingAgent
from retrieval.agents.community_retrieval import CommunityRetrievalAgent
from retrieval.agents.context_construction import ContextConstructionAgent
from retrieval.agents.evidence_aggregation import EvidenceAggregationAgent
from retrieval.agents.faithfulness_validation import FaithfulnessValidationAgent
from retrieval.agents.feedback import FeedbackAgent
from retrieval.agents.graph_traversal import GraphTraversalAgent
from retrieval.agents.query_classification import QueryClassificationAgent
from retrieval.agents.query_decomposition import QueryDecompositionAgent
from retrieval.agents.query_rewriting import QueryRewritingAgent
from retrieval.agents.retry_controller import RetryControllerAgent
from retrieval.agents.self_critique import SelfCritiqueAgent
from retrieval.agents.vector_retrieval import VectorRetrievalAgent

ALL_AGENTS = [
    QueryClassificationAgent,  # A1 / Alg. 2
    QueryRewritingAgent,  # A2 / Alg. 3
    QueryDecompositionAgent,  # A3 / Alg. 4
    VectorRetrievalAgent,  # A4 / Alg. 5
    GraphTraversalAgent,  # A5 / Alg. 6
    CommunityRetrievalAgent,  # A6 / Alg. 7
    EvidenceAggregationAgent,  # A7 / Alg. 8
    ContextConstructionAgent,  # A8 / Alg. 9
    AnswerGenerationAgent,  # A9 / Alg. 10
    CitationMappingAgent,  # A10 / Alg. 11
    FaithfulnessValidationAgent,  # A11 / Alg. 12
    SelfCritiqueAgent,  # A12 / Alg. 13
    FeedbackAgent,  # A13 / Alg. 14
    RetryControllerAgent,  # A14 / Alg. 15
    CacheManagerAgent,  # A15 / Alg. 16
]

__all__ = [
    "ALL_AGENTS",
    "AnswerGenerationAgent",
    "CacheManagerAgent",
    "CitationMappingAgent",
    "CommunityRetrievalAgent",
    "ContextConstructionAgent",
    "EvidenceAggregationAgent",
    "FaithfulnessValidationAgent",
    "FeedbackAgent",
    "GraphTraversalAgent",
    "QueryClassificationAgent",
    "QueryDecompositionAgent",
    "QueryRewritingAgent",
    "RetryControllerAgent",
    "SelfCritiqueAgent",
    "VectorRetrievalAgent",
]
