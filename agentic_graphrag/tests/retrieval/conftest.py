"""Offline Stage 3 fixtures: a KG/VDB built once by Stage 2, copied per test."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from common.embedder import FakeEmbedder
from common.llm import MockLLM
from common.review_queue import ReviewQueue
from construction.config import ConstructionConfig, ConstructionPaths
from construction.graph import AgenticGraphConstructionFramework
from construction.stores import InMemoryGraphStore, InMemoryVectorStore
from construction.testing import make_document, offline_construction_llm
from retrieval.agents.cache_manager import SemanticCache
from retrieval.config import RetrievalConfig, RetrievalPaths
from retrieval.context import RetrievalAgentContext
from retrieval.graph import AgenticGraphRetrievalFramework
from retrieval.testing import offline_retrieval_llm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SECTIONS = [
    (
        "Introduction",
        "This report describes the maintenance procedure for the hydraulic pump. The "
        "Hydraulic Control Unit (HCU) regulates pressure across the system. Inspection "
        "results are summarised in Table 1.",
    ),
    (
        "Method",
        "Technicians follow ISO 9001 guidelines when servicing the pump. The HCU is "
        "inspected every 500 hours. Pressure is computed with Equation 1.",
    ),
    ("Conclusion", "The HCU performed within limits during all inspections."),
]
TABLES = {
    1: [
        (
            "Table 1: Inspection intervals",
            ["Component", "Interval (h)", "Status"],
            [["Seal", "500", "OK"], ["Valve", "1000", "Replaced"], ["Filter", "250", "OK"]],
        )
    ]
}

Stores = tuple[InMemoryGraphStore, InMemoryVectorStore]


@pytest.fixture(scope="session")
def built_stores(tmp_path_factory: pytest.TempPathFactory) -> Stores:
    root = tmp_path_factory.mktemp("kg")
    j, m = make_document(
        SECTIONS,
        title="Hydraulic Pump Maintenance Report",
        abbreviations={"HCU": "Hydraulic Control Unit"},
        tables=TABLES,
    )
    cfg = ConstructionConfig(
        paths=ConstructionPaths(
            output_dir=str(root / "out"),
            state_dir=str(root / "state"),
            review_queue=str(root / "rq.jsonl"),
            schemas_dir=str(PROJECT_ROOT / "schemas"),
            ontology=str(PROJECT_ROOT / "ontology.yaml"),
        )
    )
    graph, vectors = InMemoryGraphStore(), InMemoryVectorStore()
    result = AgenticGraphConstructionFramework(
        cfg,
        llm=offline_construction_llm(),
        embedder=FakeEmbedder(),
        graph_store=graph,
        vector_store=vectors,
    ).run(j, m, root / "out")
    assert result.ok, result.error
    return graph, vectors


@pytest.fixture
def stores(built_stores: Stores) -> Stores:
    """A private copy: feedback writes never leak between tests."""
    return copy.deepcopy(built_stores)


@pytest.fixture
def rconfig(tmp_path: Path) -> RetrievalConfig:
    return RetrievalConfig(
        paths=RetrievalPaths(
            output_dir=str(tmp_path / "retrieval"),
            state_dir=str(tmp_path / "state"),
            review_queue=str(tmp_path / "rq.jsonl"),
            cache_file=str(tmp_path / "retrieval" / "cache.json"),
            feedback_log=str(tmp_path / "retrieval" / "feedback.jsonl"),
            schemas_dir=str(PROJECT_ROOT / "schemas"),
        )
    )


def make_rctx(
    config: RetrievalConfig, stores: Stores, llm: MockLLM | None = None
) -> RetrievalAgentContext:
    return RetrievalAgentContext(
        config=config.as_extraction_config(),
        llm=llm or offline_retrieval_llm(),
        embedder=FakeEmbedder(),
        review_queue=ReviewQueue(config.paths.review_queue),
        settings=config,
        graph=stores[0],
        vectors=stores[1],
        cache=SemanticCache(config.paths.cache_file, max_entries=config.cache.max_entries),
    )


@pytest.fixture
def rctx(rconfig: RetrievalConfig, stores: Stores) -> RetrievalAgentContext:
    return make_rctx(rconfig, stores)


def make_framework(
    config: RetrievalConfig, stores: Stores, llm: MockLLM | None = None, **kw: Any
) -> AgenticGraphRetrievalFramework:
    return AgenticGraphRetrievalFramework(
        config,
        llm=llm or offline_retrieval_llm(),
        embedder=FakeEmbedder(),
        graph_store=stores[0],
        vector_store=stores[1],
        **kw,
    )
