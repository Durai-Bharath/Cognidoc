"""Regenerate the published JSON Schemas in ``schemas/`` from the output models."""

from pathlib import Path

from extraction.schema_export import write_schemas

if __name__ == "__main__":
    for path in write_schemas(Path(__file__).resolve().parents[1] / "schemas"):
        print(f"wrote {path}")
