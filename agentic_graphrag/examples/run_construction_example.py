"""Stage 1 → Stage 2 on a generated sample PDF, fully offline.

    python examples/run_construction_example.py [--out examples/output/construction]

Runs Stage 1 extraction on the sample maintenance report, then builds the knowledge graph
twice. The second run shows idempotent re-ingestion (Updates only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import ExtractionConfig
from common.embedder import FakeEmbedder
from construction.config import ConstructionConfig
from construction.graph import AgenticGraphConstructionFramework
from construction.stores import InMemoryGraphStore, InMemoryVectorStore
from construction.testing import offline_construction_llm
from extraction.graph import AgenticDocumentExtractionFramework
from extraction.ocr import MockOCREngine
from extraction.testing import make_native_pdf, offline_llm, scanned_page_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/output/construction")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    stage1_cfg = ExtractionConfig()
    stage1_cfg.paths.review_queue = str(out / "review_queue.jsonl")
    stage1 = AgenticDocumentExtractionFramework(
        stage1_cfg,
        llm=offline_llm(),
        embedder=FakeEmbedder(),
        ocr=MockOCREngine(scanned_page_lines()),
    )
    extracted = stage1.run(make_native_pdf(out / "sample.pdf"), out / "stage1")
    j = json.loads(Path(extracted.output_paths["document"]).read_text())
    m = json.loads(Path(extracted.output_paths["metadata"]).read_text())

    cfg = ConstructionConfig()
    cfg.paths.review_queue = str(out / "review_queue.jsonl")
    fw = AgenticGraphConstructionFramework(
        cfg,
        llm=offline_construction_llm(),
        embedder=FakeEmbedder(),
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
    )
    for run in ("first", "second"):
        r = fw.run(j, m, out / f"stage2_{run}").report
        print(
            f"{run:>6} run: ops={r['operations']} chunks={r['chunks']['accepted']}/"
            f"{r['chunks']['total']} validation={r['validation']['passed']} "
            f"health={r['audit']['health_score']} communities={r['communities']['total']}"
        )
    names = {n.id: n.name for n in fw.kg.nodes()}
    print("\nfacts:")
    for e in sorted(fw.kg.edges(), key=lambda e: e.id):
        if not e.structural:
            print(f"  {names[e.source]} -[{e.predicate}]-> {names[e.target]}  ({e.confidence})")
    print(f"\nreport: {out / 'stage2_second' / 'construction_report.json'}")


if __name__ == "__main__":
    main()
