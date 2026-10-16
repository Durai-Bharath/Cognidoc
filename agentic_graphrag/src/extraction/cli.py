"""Command line entry point: ``agentic-extract <document> [--config config.yaml]``."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import find_dotenv, load_dotenv

from common.embedder import FakeEmbedder
from common.llm import MockLLM
from extraction.graph import build_pipeline
from extraction.ocr import MockOCREngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 agentic document extraction")
    parser.add_argument("document", help="PDF, DOCX or image (PNG/JPG/JPEG/TIFF/BMP)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Output directory for M, J and the report")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use MockLLM, FakeEmbedder and MockOCREngine (no network)",
    )
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    kwargs = {}
    if args.offline:
        kwargs = {"llm": MockLLM(), "embedder": FakeEmbedder(), "ocr": MockOCREngine()}
    pipeline = build_pipeline(args.config, **kwargs)
    result = pipeline.run(args.document, args.out)
    summary = {
        "document_id": result.document_id,
        "ok": result.ok,
        "approved": result.validation_report.approved if result.validation_report else False,
        "quality_score": result.validation_report.quality_score
        if result.validation_report
        else 0.0,
        "outputs": result.output_paths,
        "review_items": result.review_items,
        "error": result.error.model_dump() if result.error else None,
        "agent_errors": result.agent_errors,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
