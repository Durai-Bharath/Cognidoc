"""Run Stage 1 on generated sample documents.

    python examples/run_example.py            # offline: MockLLM, FakeEmbedder, MockOCREngine
    python examples/run_example.py --live     # providers from config.yaml + keys from .env

Outputs land in examples/output/<sample>/{metadata,document,validation_report}.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from common.embedder import FakeEmbedder
from common.review_queue import ReviewQueue
from extraction.graph import build_pipeline
from extraction.ocr import MockOCREngine
from extraction.testing import (
    make_docx,
    make_native_pdf,
    make_scanned_image,
    offline_llm,
    scanned_page_lines,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use the providers in config.yaml")
    args = parser.parse_args()
    load_dotenv(find_dotenv(usecwd=True))

    inputs = HERE / "output" / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    docs = {
        "native_pdf": make_native_pdf(inputs / "sample.pdf"),
        "scanned_image": make_scanned_image(inputs / "scan.png"),
        "docx": make_docx(inputs / "sample.docx"),
    }
    kwargs: dict[str, Any] = {}
    if not args.live:
        kwargs = {
            "llm": offline_llm(),
            "embedder": FakeEmbedder(),
            "ocr": MockOCREngine(scanned_page_lines()),
        }
    kwargs["review_queue"] = ReviewQueue(HERE / "output" / "review_queue.jsonl")
    pipeline = build_pipeline(HERE.parent / "config.yaml", **kwargs)

    for name, path in docs.items():
        result = pipeline.run(path, HERE / "output" / name)
        report = result.validation_report
        print(
            json.dumps(
                {
                    "sample": name,
                    "ok": result.ok,
                    "approved": report.approved if report else False,
                    "quality_score": report.quality_score if report else 0.0,
                    "sections": [s.title for s in result.metadata.sections]
                    if result.metadata
                    else [],
                    "outputs": result.output_paths,
                    "llm_tokens": sum(t.token_usage.total for t in result.traces),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
