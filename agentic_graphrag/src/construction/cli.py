"""Command line entry point: ``construct --json document.json --metadata metadata.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from common.embedder import FakeEmbedder
from common.logging import configure_logging
from construction.config import load_construction_config
from construction.graph import AgenticGraphConstructionFramework
from construction.testing import offline_construction_llm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 agentic knowledge-graph construction")
    parser.add_argument("--json", required=True, help="Stage 1 structured JSON J (document.json)")
    parser.add_argument("--metadata", required=True, help="Stage 1 metadata M (metadata.json)")
    parser.add_argument("--config", default=None, help="Path to construction.yaml")
    parser.add_argument("--out", default=None, help="Output directory for the report and exports")
    parser.add_argument("--graph-backend", choices=["memory", "neo4j"], default=None)
    parser.add_argument("--vector-backend", choices=["memory", "qdrant"], default=None)
    parser.add_argument("--state-dir", default=None, help="Where in-memory KG/VDB state persists")
    parser.add_argument(
        "--no-persist", action="store_true", help="Do not save in-memory KG/VDB state after the run"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the offline MockLLM and FakeEmbedder (no network)",
    )
    args = parser.parse_args(argv)

    load_dotenv(find_dotenv(usecwd=True))
    config = load_construction_config(args.config)
    if args.graph_backend:
        config.stores.graph_backend = args.graph_backend
    if args.vector_backend:
        config.stores.vector_backend = args.vector_backend
    if args.state_dir:
        config.paths.state_dir = args.state_dir
    configure_logging(config.log_level)
    if args.offline:
        pipeline = AgenticGraphConstructionFramework(
            config, llm=offline_construction_llm(), embedder=FakeEmbedder()
        )
    else:
        pipeline = AgenticGraphConstructionFramework(config)
    document = json.loads(Path(args.json).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    try:
        result = pipeline.run(document, metadata, args.out, persist_state=not args.no_persist)
    finally:
        pipeline.kg.close()
        pipeline.vdb.close()
    report = result.report
    summary = {
        "document_id": result.document_id,
        "ok": result.ok,
        "status": report.get("status"),
        "chunks": {k: v for k, v in report.get("chunks", {}).items() if k != "outcomes"},
        "operations": report.get("operations"),
        "conflicts": len(result.conflicts),
        "validation_passed": (report.get("validation") or {}).get("passed"),
        "health_score": (report.get("audit") or {}).get("health_score"),
        "communities": (report.get("communities") or {}).get("total"),
        "outputs": result.output_paths,
        "error": result.error.model_dump() if result.error else None,
    }
    print(json.dumps(summary, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
