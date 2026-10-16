"""Command line entry point for Stage 3 (and the three-stage ``run-all``).

    graphrag query "What is the inspection interval of the HCU?" [--history h.json]
    graphrag report rpt-0123456789abcdef
    graphrag serve [--host 127.0.0.1 --port 8000]
    graphrag run-all [--input doc.pdf] [--query "..."]... [--out examples/output/run_all]

``--offline`` swaps in the MockLLM and FakeEmbedder (no network). It must match how the
KG was built: a KG made with ``construct --offline`` is queried with ``--offline``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from common.config import PROJECT_ROOT
from common.embedder import FakeEmbedder
from common.logging import configure_logging
from retrieval.config import RetrievalConfig, load_retrieval_config
from retrieval.graph import AgenticGraphRetrievalFramework, load_report
from retrieval.run_all import DEFAULT_QUERIES, run_all, summarise
from retrieval.testing import offline_retrieval_llm


def _config(args: argparse.Namespace) -> RetrievalConfig:
    path = args.config
    if path is None and not args.offline and (PROJECT_ROOT / "retrieval.yaml").exists():
        path = PROJECT_ROOT / "retrieval.yaml"
    config = load_retrieval_config(path)
    if args.state_dir:
        config.paths.state_dir = args.state_dir
    if args.out:
        config.paths.output_dir = args.out
        config.paths.cache_file = str(Path(args.out) / "cache.json")
        config.paths.feedback_log = str(Path(args.out) / "feedback.jsonl")
    if args.graph_backend:
        config.stores.graph_backend = args.graph_backend
    if args.vector_backend:
        config.stores.vector_backend = args.vector_backend
    if getattr(args, "max_retries", None) is not None:
        config.retry.max_retries = args.max_retries
    config.feedback.persist_state = not getattr(args, "no_persist", False)
    return config


def _framework(args: argparse.Namespace) -> AgenticGraphRetrievalFramework:
    config = _config(args)
    configure_logging(config.log_level)
    kwargs: dict[str, Any] = {}
    if args.offline:
        kwargs = {"llm": offline_retrieval_llm(), "embedder": FakeEmbedder()}
    return AgenticGraphRetrievalFramework(config, **kwargs)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to retrieval.yaml")
    p.add_argument("--offline", action="store_true", help="MockLLM + FakeEmbedder (no network)")
    p.add_argument("--state-dir", default=None, help="Stage 2 in-memory KG/VDB state directory")
    p.add_argument("--out", default=None, help="Output directory for reports and the cache")
    p.add_argument("--graph-backend", choices=["memory", "neo4j"], default=None)
    p.add_argument("--vector-backend", choices=["memory", "qdrant"], default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphrag", description="Agentic GraphRAG Stage 3")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Answer one question over the KG/VDB")
    q.add_argument("question")
    q.add_argument("--history", default=None, help="JSON file: [{role, content}, ...]")
    q.add_argument("--max-retries", type=int, default=None)
    q.add_argument("--no-persist", action="store_true", help="Do not save feedback to the KG state")
    q.add_argument("--full", action="store_true", help="Print the whole validation report")
    _common(q)

    r = sub.add_parser("report", help="Print a stored validation report")
    r.add_argument("report_id")
    _common(r)

    s = sub.add_parser("serve", help="Run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    _common(s)

    a = sub.add_parser("run-all", help="Stage 1 → 2 → 3 on an example document")
    a.add_argument("--input", default=None, help="Document (default: generated sample PDF)")
    a.add_argument("--query", action="append", default=None, help="Repeatable")
    a.add_argument("--out", default="examples/output/run_all")
    a.add_argument("--offline", action="store_true")
    a.add_argument("--config-dir", default=str(PROJECT_ROOT))

    args = parser.parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))

    if args.command == "run-all":
        configure_logging("WARNING")
        with contextlib.redirect_stdout(sys.stderr):  # keep stdout pure JSON (PyMuPDF prints)
            summary = run_all(
                input_path=args.input,
                queries=args.query or DEFAULT_QUERIES,
                out_dir=args.out,
                offline=args.offline,
                config_dir=args.config_dir,
            )
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1

    if args.command == "report":
        report = load_report(args.report_id, _config(args))
        if report is None:
            print(f"report {args.report_id} not found", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2))
        return 0

    framework = _framework(args)
    if args.command == "serve":
        import uvicorn

        from retrieval.api import create_app

        uvicorn.run(create_app(framework), host=args.host, port=args.port)
        return 0

    history = json.loads(Path(args.history).read_text(encoding="utf-8")) if args.history else []
    try:
        result = framework.query(args.question, history)
    finally:
        framework.kg.close()
        framework.vdb.close()
    print(json.dumps(result.report if args.full else summarise(result), indent=2, default=str))
    return 0 if result.verified else 2


if __name__ == "__main__":
    sys.exit(main())
