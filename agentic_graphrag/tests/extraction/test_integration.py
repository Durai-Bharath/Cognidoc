"""End-to-end (offline): native PDF and scanned image yield schema-valid J and M."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from common.config import ExtractionConfig
from common.embedder import FakeEmbedder
from extraction.cli import main as cli_main
from extraction.graph import AgenticDocumentExtractionFramework
from extraction.ocr import MockOCREngine
from extraction.schema_export import SCHEMA_MODELS, generate_schema
from extraction.testing import offline_llm, scanned_page_lines

SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"
pytestmark = pytest.mark.integration


def _load(name: str) -> dict[str, object]:
    return json.loads((SCHEMAS / name).read_text())  # type: ignore[no-any-return]


@pytest.mark.parametrize("kind", ["pdf", "scan", "docx"])
def test_outputs_validate_against_published_schemas(
    config: ExtractionConfig, samples: dict[str, Path], tmp_path: Path, kind: str
) -> None:
    fw = AgenticDocumentExtractionFramework(
        config, llm=offline_llm(), embedder=FakeEmbedder(), ocr=MockOCREngine(scanned_page_lines())
    )
    result = fw.run(samples[kind], tmp_path / kind)
    assert result.ok, result.error
    files = {k: json.loads(Path(v).read_text()) for k, v in result.output_paths.items()}
    jsonschema.validate(files["metadata"], _load("metadata.schema.json"))
    jsonschema.validate(files["document"], _load("document.schema.json"))
    jsonschema.validate(files["validation_report"], _load("validation_report.schema.json"))
    report = result.validation_report
    assert report is not None and report.approved, [
        (c.name, ch.name, ch.reasons) for c in report.categories for ch in c.checks if not ch.passed
    ]
    m = files["metadata"]
    assert m["statistics"]["total_sections"] >= 2 and m["statistics"]["total_tables"] == 1
    if kind == "scan":
        assert m["quality"]["ocr_applied"] and m["quality"]["enhanced_pages"] == [1]
    assert Path(config.paths.review_queue).exists() or result.review_items == 0


def test_published_schemas_match_models() -> None:
    for name in SCHEMA_MODELS:
        assert _load(name) == generate_schema(name), f"{name} is stale: run `make schemas`"


def test_cli_offline(
    samples: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli_main([str(samples["docx"]), "--offline", "--out", str(tmp_path / "cli")])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["ok"] and summary["approved"]
    assert set(summary["outputs"]) == {"metadata", "document", "validation_report"}
