"""Offline Stage 2 fixtures: config and a context with in-memory stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from construction.config import ConstructionConfig, ConstructionPaths
from construction.context import ConstructionContext
from tests.construction.builders import make_ctx

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cconfig(tmp_path: Path) -> ConstructionConfig:
    return ConstructionConfig(
        paths=ConstructionPaths(
            output_dir=str(tmp_path / "out"),
            state_dir=str(tmp_path / "state"),
            review_queue=str(tmp_path / "review_queue.jsonl"),
            schemas_dir=str(PROJECT_ROOT / "schemas"),
            ontology=str(PROJECT_ROOT / "ontology.yaml"),
        )
    )


@pytest.fixture
def cctx(cconfig: ConstructionConfig) -> ConstructionContext:
    return make_ctx(cconfig)
