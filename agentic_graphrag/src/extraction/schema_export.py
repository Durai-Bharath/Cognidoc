"""Publish and load the Stage 1 JSON Schemas (``schemas/``) that Stage 2 validates against."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import BaseModel

from extraction.output_models import SCHEMA_VERSION, DocumentJSON, MetadataFile, ValidationReport

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "document.schema.json": DocumentJSON,
    "metadata.schema.json": MetadataFile,
    "validation_report.schema.json": ValidationReport,
}


def generate_schema(filename: str) -> dict[str, Any]:
    model = SCHEMA_MODELS[filename]
    schema = model.model_json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://agentic-graphrag/schemas/stage1/{SCHEMA_VERSION}/{filename}",
        **schema,
    }


def write_schemas(directory: str | Path) -> list[Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in SCHEMA_MODELS:
        path = out / name
        path.write_text(json.dumps(generate_schema(name), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def load_schema(directory: str | Path, filename: str) -> dict[str, Any]:
    """Load a published schema, falling back to the in-code model if the file is missing."""
    path = Path(directory) / filename
    if path.exists():
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    return generate_schema(filename)


def schema_errors(instance: Any, schema: dict[str, Any], limit: int = 20) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    return [
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors[:limit]
    ]
