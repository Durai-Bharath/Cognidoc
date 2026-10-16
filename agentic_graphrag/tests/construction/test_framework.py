"""Algorithm 1 end to end (offline): integration, idempotency, conflicts, isolation, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from common.config import ExtractionConfig
from common.embedder import FakeEmbedder
from common.llm import LLMError, MockLLM
from construction.cli import main as cli_main
from construction.config import ConstructionConfig
from construction.graph import AgenticGraphConstructionFramework
from construction.models import MENTIONED_IN, entity_node_id
from construction.ontology import OntologyError
from construction.stores import (
    CHUNKS,
    COMMUNITIES,
    ENTITIES,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from construction.testing import make_document, offline_construction_llm, offline_triples
from extraction.graph import AgenticDocumentExtractionFramework
from extraction.ocr import MockOCREngine
from extraction.testing import offline_llm, scanned_page_lines

pytestmark = pytest.mark.integration

TABLE = (
    "Table 1: Intervals",
    ["Component", "Interval (h)", "Status"],
    [["Seal", "500", "OK"], ["Valve", "1000", "Replaced"]],
)


def framework(
    config: ConstructionConfig, llm: MockLLM | None = None, **kw: Any
) -> AgenticGraphConstructionFramework:
    return AgenticGraphConstructionFramework(
        config,
        llm=llm or offline_construction_llm(),
        embedder=FakeEmbedder(),
        graph_store=kw.pop("graph_store", InMemoryGraphStore()),
        vector_store=kw.pop("vector_store", InMemoryVectorStore()),
        **kw,
    )


@pytest.fixture(scope="module")
def stage1_output(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Stage 1 offline on the sample PDF and return its (J, M)."""
    from extraction.testing import make_native_pdf

    root = tmp_path_factory.mktemp("stage1")
    pdf = make_native_pdf(root / "sample.pdf")
    cfg = ExtractionConfig()
    cfg.paths.review_queue = str(root / "rq.jsonl")
    fw = AgenticDocumentExtractionFramework(
        cfg, llm=offline_llm(), embedder=FakeEmbedder(), ocr=MockOCREngine(scanned_page_lines())
    )
    result = fw.run(pdf, root / "out")
    assert result.ok
    j = json.loads(Path(result.output_paths["document"]).read_text())
    m = json.loads(Path(result.output_paths["metadata"]).read_text())
    return j, m


def _semantic(fw: AgenticGraphConstructionFramework) -> set[tuple[str, str, str]]:
    names = {n.id: n.name for n in fw.kg.nodes()}
    return {
        (names[e.source], e.predicate, names[e.target]) for e in fw.kg.edges() if not e.structural
    }


def _entity_graph(fw: AgenticGraphConstructionFramework) -> tuple[set[str], set[str]]:
    return {n.id for n in fw.kg.nodes("Entity")}, {e.id for e in fw.kg.edges() if not e.structural}


# ----------------------------------------------------------------------------- integration


def test_stage1_output_builds_a_validated_graph(
    cconfig: ConstructionConfig,
    stage1_output: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    j, m = stage1_output
    fw = framework(cconfig)
    result = fw.run(j, m, tmp_path / "out")
    assert result.ok, result.error
    r = result.report
    assert (
        r["status"] == "completed" and r["chunks"]["failed"] == 0 and r["chunks"]["accepted"] >= 3
    )
    assert r["operations"]["Create"] > 5 and r["triples"]["accepted"] > 5
    facts = _semantic(fw)
    assert (
        "Hydraulic Control Unit",
        "regulates",
        "pressure",
    ) in facts  # abbreviation resolved (A4)
    assert ("Seal", "has_interval", "500 h") in facts  # table chunk (A2) + synonym (A7)
    assert ("Technicians", "complies_with", "ISO 9001 guidelines") in facts
    assert any(p == "has_interval" and h.startswith("method") for h, p, _ in facts)  # LLM fallback
    # Provenance on every node and edge, vectors for every entity and chunk.
    for x in [*fw.kg.nodes(), *fw.kg.edges()]:
        assert x.provenance and not x.provenance[0].missing_fields(), x.id
    entities, chunks = fw.kg.nodes("Entity"), fw.kg.nodes("Chunk")
    assert (
        fw.vdb.count(ENTITIES) == len(entities)
        and fw.vdb.count(CHUNKS) == len(chunks) == r["chunks"]["accepted"]
    )
    assert all(fw.kg.find_edges(source=e.id, predicate=MENTIONED_IN) for e in entities)
    # Consistency and ontology checks pass; audit metrics and communities are reported.
    checks = {c["name"]: c["passed"] for c in r["validation"]["checks"]}
    for name in (
        "ontology_compliance",
        "node_edge_consistency",
        "provenance_coverage",
        "graph_connectivity",
        "vector_mapping_integrity",
    ):
        assert checks[name], name
    assert r["validation"]["passed"]
    audit = r["audit"]
    for key in (
        "orphan_ratio",
        "duplicate_ratio",
        "stale_ratio",
        "low_confidence_ratio",
        "provenance_coverage",
        "vector_coverage",
        "health_score",
    ):
        assert key in audit["metrics"]
    assert (
        audit["metrics"]["provenance_coverage"] == 1.0
        and audit["metrics"]["vector_coverage"] == 1.0
    )
    assert {"orphans", "duplicates", "stale_edges", "components", "low_confidence_edges"} <= set(
        audit
    )
    assert r["communities"]["total"] >= 1 and fw.vdb.count(COMMUNITIES) == r["communities"]["total"]
    # Exports (lines 29-31).
    on_disk = json.loads(Path(result.output_paths["construction_report"]).read_text())
    assert on_disk["document_id"] == result.document_id
    kg = json.loads(Path(result.output_paths["knowledge_graph"]).read_text())
    assert len(kg["nodes"]) == len(fw.kg.nodes()) and len(kg["edges"]) == len(fw.kg.edges())
    assert json.loads(Path(result.output_paths["vector_database"]).read_text())["collections"][
        ENTITIES
    ] == len(entities)


def test_reingesting_the_same_document_is_idempotent(
    cconfig: ConstructionConfig,
    stage1_output: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    j, m = stage1_output
    fw = framework(cconfig)
    first = fw.run(j, m, tmp_path / "1")
    nodes1, edges1 = {n.id for n in fw.kg.nodes()}, {e.id for e in fw.kg.edges()}
    vectors1 = {c: sorted(fw.vdb.ids(c)) for c in (ENTITIES, CHUNKS, COMMUNITIES)}
    second = fw.run(j, m, tmp_path / "2")
    assert second.document_id == first.document_id and second.report["reingested"]
    ops = second.report["operations"]
    assert (
        ops["Create"] == 0
        and ops["Update"] == first.report["operations"]["Create"]
        and ops["Reject"] == 0
    )
    assert {n.id for n in fw.kg.nodes()} == nodes1 and {e.id for e in fw.kg.edges()} == edges1
    assert {c: sorted(fw.vdb.ids(c)) for c in vectors1} == vectors1
    assert all(e.support == 1 for e in fw.kg.edges())  # no duplicated provenance either
    assert second.report["communities"]["unchanged"] == second.report["communities"]["total"]
    assert second.report["audit"]["duplicates"] == []


def test_persisted_state_makes_cli_runs_incremental(
    cconfig: ConstructionConfig,
    stage1_output: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    j, m = stage1_output
    framework_a = AgenticGraphConstructionFramework(
        cconfig, llm=offline_construction_llm(), embedder=FakeEmbedder()
    )
    framework_a.run(j, m, tmp_path / "a", persist_state=True)
    framework_b = AgenticGraphConstructionFramework(
        cconfig, llm=offline_construction_llm(), embedder=FakeEmbedder()
    )
    result = framework_b.run(j, m, tmp_path / "b")
    assert result.report["operations"]["Create"] == 0 and result.report["reingested"]


# ----------------------------------------------------------------------------- conflicts


def test_contradictory_triples_across_documents_are_logged_and_graph_unchanged(
    cconfig: ConstructionConfig, tmp_path: Path
) -> None:
    fw = framework(cconfig)
    j1, m1 = make_document(
        [("Maintenance", "Seals are inspected regularly.")], title="Manual A", tables={0: [TABLE]}
    )
    first = fw.run(j1, m1, tmp_path / "a")
    assert first.ok and first.report["conflicts"] == []
    before = _entity_graph(fw)
    edges_before = {e.id: e.model_dump() for e in fw.kg.edges() if not e.structural}

    contradiction = (
        "Table 3: Revised intervals",
        ["Component", "Interval (h)"],
        [["Seal", "2000"]],
    )
    j2, m2 = make_document(
        [("Revision", "Revised schedule.")], title="Manual B", tables={0: [contradiction]}
    )
    second = fw.run(j2, m2, tmp_path / "b")
    assert second.ok
    assert _entity_graph(fw) == before  # no node or edge for "2000 h"
    assert {e.id: e.model_dump() for e in fw.kg.edges() if not e.structural} == edges_before
    assert fw.kg.get_node(entity_node_id("2000 h")) is None
    (record,) = second.conflicts
    assert record.status.value == "Conflict" and "functional" in record.reason
    assert record.incoming["head"] == "Seal" and record.incoming["tail"] == "2000 h"
    assert record.incoming["provenance"]["doc_id"] == second.document_id
    assert record.existing[0]["tail"] == "500 h"
    assert record.existing[0]["provenance"][0]["doc_id"] == first.document_id
    assert second.report["conflicts"][0]["conflict_id"] == record.conflict_id
    assert (
        second.report["operations"]["Reject"] == 1
        and second.report["conflict_statuses"]["Conflict"] == 1
    )
    queued = [
        i
        for i in fw.ctx.review_queue.read_all()
        if i.details.get("conflict_id") == record.conflict_id
    ]
    assert len(queued) == 1 and queued[0].agent == "decision"


def test_corroborating_document_merges(cconfig: ConstructionConfig, tmp_path: Path) -> None:
    fw = framework(cconfig)
    j1, m1 = make_document([("A", "The pump uses oil.")], title="Doc A")
    j2, m2 = make_document([("B", "The pump uses oil, as noted.")], title="Doc B")
    first = fw.run(j1, m1, tmp_path / "a")
    result = fw.run(j2, m2, tmp_path / "b")
    assert result.report["operations"]["Merge"] == 1 and result.report["operations"]["Create"] == 0
    (edge,) = [e for e in fw.kg.edges() if not e.structural]
    assert edge.support == 2
    assert {p.doc_id for p in edge.provenance} == {first.document_id, result.document_id}


# ----------------------------------------------------------------------------- branches + isolation


def test_low_confidence_chunk_is_skipped(cconfig: ConstructionConfig, tmp_path: Path) -> None:
    fw = framework(cconfig)
    j, m = make_document([("A", "The pump uses oil."), ("B", "The valve may require grease.")])
    j["sections"][1]["chunks"][0]["text"] = "The valve requires grease possibly."
    result = fw.run(j, m, tmp_path)
    outcomes = {o["chunk_id"].split(":")[1]: o for o in result.report["chunks"]["outcomes"]}
    assert outcomes["sec-1-c0"]["status"] == "accepted"
    skipped = outcomes["sec-2-c0"]
    assert skipped["status"] == "skipped_low_confidence" and skipped["triples_discarded"] == 1
    assert skipped["operations"] == {} and result.report["chunks"]["skipped_low_confidence"] == 1
    assert fw.kg.get_node(f"{result.document_id}:sec-2-c0") is None  # discarded: no chunk node
    assert fw.kg.get_node(entity_node_id("valve")) is None


def test_one_failing_chunk_is_logged_and_skipped(
    cconfig: ConstructionConfig, tmp_path: Path
) -> None:
    def flaky(prompt: str, system: str, images: Any) -> dict[str, Any]:
        if "EXPLODE" in prompt:
            raise LLMError("provider down")
        return offline_triples(prompt)

    fw = framework(cconfig, offline_construction_llm(triple_extraction=flaky))
    j, m = make_document(
        [("A", "The pump uses oil."), ("B", "EXPLODE here."), ("C", "The valve uses water.")]
    )
    result = fw.run(j, m, tmp_path)
    assert result.ok
    r = result.report
    assert r["chunks"]["failed"] == 1 and r["chunks"]["accepted"] == 2
    failed = next(o for o in r["chunks"]["outcomes"] if o["status"] == "failed")
    assert failed["stage"] == "triple_extraction" and "provider down" in failed["error"]
    assert any("triple_extraction" in e for e in r["agent_errors"])
    assert {("pump", "uses", "oil"), ("valve", "uses", "water")} <= _semantic(fw)
    assert any(i.reason == "chunk failed and was skipped" for i in fw.ctx.review_queue.items)
    assert r["validation"]["passed"]


def test_unexpected_crash_inside_a_chunk_is_isolated(
    cconfig: ConstructionConfig, tmp_path: Path
) -> None:
    fw = framework(cconfig)
    original = fw._n_conflict_detection.__wrapped__  # type: ignore[attr-defined]

    def boom(self: Any, state: Any) -> Any:
        if "valve" in state["current"].text:
            raise RuntimeError("bug")
        return original(self, state)

    import construction.graph as g

    fw._n_conflict_detection = g._chunk_node("conflict_detection")(boom).__get__(fw)  # type: ignore[method-assign]
    fw.graph = fw._build_graph()
    j, m = make_document([("A", "The pump uses oil."), ("B", "The valve uses water.")])
    r = fw.run(j, m, tmp_path).report
    assert r["chunks"]["failed"] == 1 and r["chunks"]["accepted"] == 1
    assert (
        "RuntimeError: bug"
        in next(o for o in r["chunks"]["outcomes"] if o["status"] == "failed")["error"]
    )


def test_rejects_follow_the_reject_branch(cconfig: ConstructionConfig, tmp_path: Path) -> None:
    fw = framework(cconfig)
    j, m = make_document([("A", "The pump teleports cargo. The pump uses oil.")])
    r = fw.run(j, m, tmp_path).report
    assert r["operations"] == {"Create": 1, "Merge": 0, "Update": 0, "Reject": 1}
    assert (
        r["conflicts"][0]["status"] == "Reject"
        and "no ontology predicate" in r["conflicts"][0]["reason"]
    )


def test_invalid_document_terminates_with_report(
    cconfig: ConstructionConfig, tmp_path: Path
) -> None:
    fw = framework(cconfig)
    j, m = make_document([("A", "x")])
    j["schema_version"] = "9.9.9"
    result = fw.run(j, m, tmp_path)
    assert not result.ok and result.error and result.error.error_type == "DocumentValidationError"
    r = json.loads(Path(result.output_paths["construction_report"]).read_text())
    assert r["status"] == "rejected" and "schema_version" in r["error"]["message"]
    assert "knowledge_graph" not in result.output_paths and fw.kg.nodes() == []
    assert [t.agent for t in result.traces] == ["document_ingestion"]


def test_invalid_ontology_fails_at_startup(cconfig: ConstructionConfig, tmp_path: Path) -> None:
    bad = tmp_path / "o.yaml"
    bad.write_text("entity_types: {}\npredicates: {}\n")
    cconfig.paths.ontology = str(bad)
    with pytest.raises(OntologyError):
        framework(cconfig)


def test_graph_maps_algorithm_1_nodes() -> None:
    fw = AgenticGraphConstructionFramework(
        ConstructionConfig(),
        llm=MockLLM(),
        embedder=FakeEmbedder(),
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
    )
    mermaid = fw.mermaid()
    for node in (
        "ingestion",
        "semantic_chunking",
        "next_chunk",
        "triple_extraction",
        "coreference_resolution",
        "confidence_evaluation",
        "discard_chunk",
        "entity_resolution",
        "relation_normalization",
        "conflict_detection",
        "decision",
        "next_operation",
        "crud_create",
        "crud_merge",
        "crud_update",
        "reject",
        "provenance",
        "embedding",
        "vector_index",
        "construction_validation",
        "graph_health_audit",
        "community_builder",
        "export",
    ):
        assert node in mermaid, node


def test_cli_offline(
    stage1_output: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    j, m = stage1_output
    (tmp_path / "document.json").write_text(json.dumps(j))
    (tmp_path / "metadata.json").write_text(json.dumps(m))
    cfg = tmp_path / "construction.yaml"
    cfg.write_text(
        f"paths:\n  review_queue: {tmp_path / 'rq.jsonl'}\n  state_dir: {tmp_path / 'state'}\n"
        f"  schemas_dir: {Path(__file__).resolve().parents[2] / 'schemas'}\n"
        f"  ontology: {Path(__file__).resolve().parents[2] / 'ontology.yaml'}\nlog_level: WARNING\n"
    )
    args = [
        "--json",
        str(tmp_path / "document.json"),
        "--metadata",
        str(tmp_path / "metadata.json"),
        "--config",
        str(cfg),
        "--out",
        str(tmp_path / "out"),
        "--offline",
    ]
    assert cli_main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] and summary["validation_passed"] and summary["operations"]["Create"] > 0
    assert (
        Path(summary["outputs"]["construction_report"]).exists()
        and "graph_state" in summary["outputs"]
    )
    assert cli_main(args) == 0  # second run reuses persisted state
    assert json.loads(capsys.readouterr().out)["operations"]["Create"] == 0
    (tmp_path / "bad.json").write_text("{}")
    assert cli_main([*args[:1], str(tmp_path / "bad.json"), *args[2:], "--no-persist"]) == 1
