"""Shared offline fixtures: config, agent context, sample documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.base_agent import AgentContext
from common.config import ExtractionConfig, PathsConfig
from common.embedder import FakeEmbedder
from common.review_queue import ReviewQueue
from extraction.ocr import MockOCREngine
from extraction.testing import (
    make_docx,
    make_native_pdf,
    make_scanned_image,
    offline_llm,
    scanned_page_lines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path: Path) -> ExtractionConfig:
    return ExtractionConfig(
        paths=PathsConfig(
            output_dir=str(tmp_path / "out"),
            review_queue=str(tmp_path / "review_queue.jsonl"),
            schemas_dir=str(PROJECT_ROOT / "schemas"),
        )
    )


@pytest.fixture
def ctx(config: ExtractionConfig) -> AgentContext:
    return AgentContext(
        config=config,
        llm=offline_llm(),
        embedder=FakeEmbedder(),
        review_queue=ReviewQueue(config.paths.review_queue),
        services={"ocr": MockOCREngine(scanned_page_lines())},
        document_id="doc-test",
    )


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("samples")
    return {
        "pdf": make_native_pdf(root / "sample.pdf"),
        "docx": make_docx(root / "sample.docx"),
        "scan": make_scanned_image(root / "scan.png"),
        "clean_scan": make_scanned_image(root / "clean.png", degrade=False),
    }
