"""``run-all``: Stage 1 → Stage 2 → Stage 3 on one document.

Extraction writes J and M; construction builds the KG/VDB and persists it to the state
directory; stale cache entries are purged (A15); then each query runs through Algorithm 1.
Offline mode uses the MockLLMs, FakeEmbedder and MockOCREngine, so it needs no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.config import load_config
from common.embedder import Embedder, FakeEmbedder, build_embedder
from common.llm import LLMClient, build_llm
from common.review_queue import ReviewQueue
from construction.config import load_construction_config
from construction.graph import AgenticGraphConstructionFramework
from construction.stores import build_graph_store, build_vector_store
from extraction.graph import AgenticDocumentExtractionFramework
from extraction.ocr import MockOCREngine
from extraction.testing import make_native_pdf, offline_llm, scanned_page_lines
from retrieval.agents.cache_manager import SemanticCache
from retrieval.config import load_retrieval_config
from retrieval.graph import AgenticGraphRetrievalFramework, RetrievalResult
from retrieval.testing import offline_all_stages_llm

DEFAULT_QUERIES = [
    "What is the inspection interval of the HCU?",
    "What are the main themes across the maintenance report?",
    "Which component has status Replaced, and what is its inspection interval?",
]


def run_all(
    *,
    input_path: str | Path | None,
    queries: list[str],
    out_dir: str | Path,
    offline: bool,
    config_dir: str | Path,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dir = Path(config_dir)
    review_queue = ReviewQueue(out / "review_queue.jsonl")
    (out / "inputs").mkdir(exist_ok=True)
    source = Path(input_path) if input_path else make_native_pdf(out / "inputs" / "sample.pdf")

    # Stage 1 -------------------------------------------------------------------------
    s1_cfg = load_config(cfg_dir / "config.yaml") if not offline else load_config(None)
    s1_cfg.paths.review_queue = str(out / "review_queue.jsonl")
    s1_kwargs: dict[str, Any] = {"review_queue": review_queue}
    embedder: Embedder
    llm: LLMClient | None
    if offline:
        embedder, llm = FakeEmbedder(), offline_all_stages_llm()
        s1_kwargs.update(
            llm=offline_llm(), embedder=embedder, ocr=MockOCREngine(scanned_page_lines())
        )
    else:
        embedder, llm = build_embedder(s1_cfg.embedder), None
        s1_kwargs["embedder"] = embedder
    stage1 = AgenticDocumentExtractionFramework(s1_cfg, **s1_kwargs)
    extracted = stage1.run(source, out / "stage1")
    if not extracted.ok or "document" not in extracted.output_paths:
        raise RuntimeError(f"Stage 1 failed on {source}")
    j = json.loads(Path(extracted.output_paths["document"]).read_text(encoding="utf-8"))
    m = json.loads(Path(extracted.output_paths["metadata"]).read_text(encoding="utf-8"))

    # Stage 2 -------------------------------------------------------------------------
    s2_cfg = load_construction_config(None if offline else cfg_dir / "construction.yaml")
    s2_cfg.paths.state_dir = str(out / "kg_state")
    s2_cfg.paths.review_queue = str(out / "review_queue.jsonl")
    graph, vectors = build_graph_store(s2_cfg), build_vector_store(s2_cfg)
    stage2 = AgenticGraphConstructionFramework(
        s2_cfg,
        llm=llm or build_llm(s2_cfg.llm),
        embedder=embedder,
        graph_store=graph,
        vector_store=vectors,
        review_queue=review_queue,
    )
    built = stage2.run(j, m, out / "stage2", persist_state=True)
    if not built.ok:
        raise RuntimeError(f"Stage 2 failed: {built.error}")

    # Stage 3 -------------------------------------------------------------------------
    s3_cfg = load_retrieval_config(None if offline else cfg_dir / "retrieval.yaml")
    s3_cfg.stores = s2_cfg.stores
    s3_cfg.paths.state_dir = s2_cfg.paths.state_dir
    s3_cfg.paths.output_dir = str(out / "stage3")
    s3_cfg.paths.cache_file = str(out / "stage3" / "cache.json")
    s3_cfg.paths.feedback_log = str(out / "stage3" / "feedback.jsonl")
    s3_cfg.paths.review_queue = str(out / "review_queue.jsonl")
    stage3 = AgenticGraphRetrievalFramework(
        s3_cfg,
        llm=llm or build_llm(s3_cfg.llm),
        embedder=embedder,
        graph_store=graph,
        vector_store=vectors,
        review_queue=review_queue,
        cache=SemanticCache(s3_cfg.paths.cache_file, max_entries=s3_cfg.cache.max_entries),
    )
    purged = stage3.invalidate_cache()  # construction may have changed cited elements
    results: list[RetrievalResult] = [stage3.query(q) for q in queries]
    return {
        "input": str(source),
        "stage1": {"document_id": extracted.document_id, "outputs": extracted.output_paths},
        "stage2": {
            "document_id": built.document_id,
            "nodes": built.report.get("graph", {}).get("nodes"),
            "edges": built.report.get("graph", {}).get("edges"),
            "communities": (built.report.get("communities") or {}).get("total"),
            "report": built.output_paths.get("construction_report"),
        },
        "stage3": {
            "cache_purged": purged,
            "queries": [summarise(r) for r in results],
        },
        "ok": bool(results) and all(r.verified for r in results),
    }


def summarise(r: RetrievalResult) -> dict[str, Any]:
    integrity = r.report.get("citation_integrity") or {}
    return {
        "query": r.query,
        "answer": r.answer,
        "verified": r.verified,
        "low_confidence": r.low_confidence,
        "cached": r.cached,
        "query_type": (r.report.get("classification") or {}).get("query_type"),
        "attempts": len(r.report.get("attempts", [])),
        "citations": [
            {
                "statement": s.text,
                "sources": [
                    {
                        "id": src.citation_id,
                        "kind": src.kind,
                        "chunks": src.chunk_ids,
                        "nodes": src.node_ids,
                        "edges": src.edge_ids,
                        "page": src.page,
                        "document": src.doc_title or src.doc_id,
                    }
                    for src in s.sources
                ],
            }
            for s in r.citations
        ],
        "citations_resolved": f"{integrity.get('resolved', 0)}/{integrity.get('total', 0)}",
        "citation_integrity": integrity.get("passed"),
        "report": r.report_path,
    }
