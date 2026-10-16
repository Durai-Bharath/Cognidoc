"""Algorithm 1 end to end (offline): Local, Global, multi-hop, forced failure, retry recovery,
cache, critique, failure isolation, API, CLI and run-all."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from common.llm import LLMError
from construction.stores import save_state
from retrieval.cli import main as cli_main
from retrieval.config import RetrievalConfig
from retrieval.graph import AgenticGraphRetrievalFramework, load_report
from retrieval.testing import offline_answer, offline_retrieval_llm
from tests.retrieval.conftest import Stores, make_framework

pytestmark = pytest.mark.integration

LOCAL = "What is the inspection interval of the HCU?"
GLOBAL = "What are the main themes across the report?"
MULTI = "Which component has status Replaced, and what is its inspection interval?"
HALLUCINATION = {
    "answer": "The pump was built on Mars in 1850 [E1].",
    "insufficient_evidence": False,
}


def resolves(fw: AgenticGraphRetrievalFramework, report: dict[str, Any]) -> None:
    """Every citation id in the report resolves to a real chunk/node/edge in the stores."""
    integrity = report["citation_integrity"]
    assert integrity["passed"] and integrity["resolved"] == integrity["total"] > 0
    for stmt in report["citations"]:
        assert stmt["sources"], stmt
        for src in stmt["sources"]:
            for cid in src["chunk_ids"]:
                node = fw.kg.get_node(cid)
                assert node is not None and node.label == "Chunk"
            for nid in src["node_ids"]:
                assert fw.kg.get_node(nid) is not None
            for eid in src["edge_ids"]:
                assert fw.kg.get_edge(eid) is not None


def test_local_query_cites_real_chunks(rconfig: RetrievalConfig, stores: Stores) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query(LOCAL)
    assert r.verified and not r.low_confidence and not r.cached
    assert r.report["classification"]["query_type"] == "Local"
    assert "500 hours" in r.answer
    resolves(fw, r.report)
    pages = [s["page"] for st in r.report["citations"] for s in st["sources"]]
    assert all(isinstance(p, int) for p in pages)
    chunks = {c for st in r.report["citations"] for s in st["sources"] for c in s["chunk_ids"]}
    assert chunks and all(fw.vdb.get("chunks", [c]) for c in chunks)
    assert r.report["validation"]["valid"] and r.report["attempts"][0]["escalation"] is None
    assert Path(r.report_path or "").exists()


def test_global_query_uses_community_summaries(rconfig: RetrievalConfig, stores: Stores) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query(GLOBAL)
    assert r.verified and r.report["classification"]["query_type"] == "Global"
    kinds = [b["kind"] for b in r.report["context"]["blocks"]]
    assert kinds[0] == "community"
    cited = [s for st in r.report["citations"] for s in st["sources"]]
    community = [s for s in cited if s["kind"] == "community"]
    assert community and fw.kg.get_node(community[0]["node_ids"][0]).label == "Community"  # type: ignore[union-attr]
    assert r.report["final_params"]["community_weight"] == 1.0
    resolves(fw, r.report)


def test_multi_hop_dependent_subqueries(rconfig: RetrievalConfig, stores: Stores) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query(MULTI)
    d = r.report["decomposition"]
    assert d["waves"] == [["q1"], ["q2"]] and d["intermediate_answers"] == {"q1": "Valve"}
    assert r.report["evidence"]["q2"]["resolved_text"] == "What is Valve's inspection interval?"
    assert r.verified and "Replaced" in r.answer and "1000 h" in r.answer
    resolves(fw, r.report)


def test_independent_subqueries_run_in_one_wave(rconfig: RetrievalConfig, stores: Stores) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query("What does the HCU regulate? What is the status of the Seal?")
    assert r.report["decomposition"]["waves"] == [["q1", "q2"]]
    traced = {t.context.get("subquery") for t in r.traces if t.agent == "vector_retrieval"}
    assert traced == {"q1", "q2"}
    assert r.verified


def test_forced_failure_exhausts_retries_and_flags(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    fw = make_framework(rconfig, stores, offline_retrieval_llm(answer_generation=HALLUCINATION))
    r = fw.query(LOCAL)
    rep = r.report
    assert not r.verified and r.low_confidence
    assert len(rep["attempts"]) == rconfig.retry.max_retries + 1
    assert [a["escalation"] for a in rep["attempts"]] == [
        None,
        "increase_k",
        "increase_hops",
        "switch_to_hybrid",
    ]
    assert [d["action"] for d in rep["retry_history"]] == ["retry", "retry", "retry", "exhausted"]
    assert rep["attempts"][-1]["params"]["query_type"] == "Hybrid"
    assert all(a["critique"] == "needs_revision" for a in rep["attempts"])
    assert all(a["feedback"]["mode"] == "penalize" for a in rep["attempts"])
    assert rep["cache"]["store"]["stored"] is False and len(fw.ctx.cache) == 0
    assert rep["review_items"] > 0
    assert rep["agents"]["retry_controller"]["calls"] == 4


def test_retry_recovers_on_a_later_attempt(rconfig: RetrievalConfig, stores: Stores) -> None:
    llm = offline_retrieval_llm(answer_generation=[HALLUCINATION, offline_answer])
    fw = make_framework(rconfig, stores, llm)
    r = fw.query(LOCAL)
    assert r.verified and r.report["best_attempt"] == 2
    assert [a["valid"] for a in r.report["attempts"]] == [False, True]
    assert r.report["attempts"][1]["escalation"] == "increase_k"
    assert r.report["final_params"]["top_k"] == rconfig.vector.top_k + rconfig.retry.k_step


def test_cache_hit_skips_retrieval_and_proves_citations(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    fw = make_framework(rconfig, stores)
    first = fw.query(LOCAL)
    assert first.report["cache"]["store"]["stored"]
    calls = len(fw.ctx.llm.calls)  # type: ignore[attr-defined]
    second = fw.query(LOCAL)
    assert second.cached and second.verified and second.answer == first.answer
    assert len(fw.ctx.llm.calls) == calls  # type: ignore[attr-defined]
    assert {t.agent for t in second.traces} == {"cache_manager"}
    assert second.report["citation_integrity"]["passed"]
    assert second.report["cache"]["source_report_id"] == first.report_id


def test_penalised_knowledge_invalidates_cached_answers(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    fw = make_framework(rconfig, stores)
    good = fw.query(LOCAL)
    assert len(fw.ctx.cache) == 1
    fw.ctx.llm.responses["answer_generation"] = lambda p, s, i: {  # type: ignore[attr-defined]
        "answer": offline_answer(p)["answer"].replace("500", "900"),
    }
    bad = fw.query("What is the inspection interval of the Hydraulic Control Unit (HCU)?")
    assert not bad.verified
    assert good.report["cache"]["store"]["key"] in bad.report["cache"]["invalidated"]
    assert len(fw.ctx.cache) == 0


def test_conversation_history_is_resolved_and_keyed(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query("What is its status?", [{"role": "user", "content": "Tell me about the Valve."}])
    assert r.report["rewritten_query"]["rewritten"] == "What is Valve's status?"
    assert r.verified and "Replaced" in r.answer
    assert not fw.query("What is its status?").cached  # different history → no cache hit


def test_always_critique_can_invalidate_a_valid_answer(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    rconfig.critique.always_critique = True
    rconfig.retry.max_retries = 0
    issue = {"statement_id": "S1", "issue_type": "logical", "severity": "major", "description": "x"}
    llm = offline_retrieval_llm(self_critique={"verdict": "needs_revision", "issues": [issue]})
    r = make_framework(rconfig, stores, llm).query(LOCAL)
    assert r.report["attempts"][0]["critique"] == "needs_revision" and not r.verified
    assert any("self-critique" in x for x in r.report["validation"]["reasons"])
    ok = make_framework(rconfig, stores).query(LOCAL)
    assert ok.verified and ok.report["critique"]["verdict"] == "verified"


def test_agent_failures_are_isolated_and_the_run_terminates(
    rconfig: RetrievalConfig, stores: Stores
) -> None:
    rconfig.retry.max_retries = 1
    llm = offline_retrieval_llm(
        answer_generation=LLMError("provider down"), query_classification=LLMError("down")
    )
    llm.max_retries = 0
    r = make_framework(rconfig, stores, llm).query(LOCAL)
    assert not r.verified and len(r.report["attempts"]) == 2
    assert any(e.startswith("answer_generation") for e in r.report["agent_errors"])
    assert r.report["classification"]["query_type"] == "Local"  # heuristic fallback


def test_report_has_traces_and_persists(rconfig: RetrievalConfig, stores: Stores) -> None:
    fw = make_framework(rconfig, stores)
    r = fw.query(LOCAL)
    agents = r.report["agents"]
    for name in (
        "cache_manager", "query_classification", "query_rewriting", "query_decomposition",
        "vector_retrieval", "graph_traversal", "community_retrieval", "evidence_aggregation",
        "context_construction", "answer_generation", "citation_mapping",
        "faithfulness_validation", "feedback",
    ):  # fmt: skip
        assert agents[name]["calls"] >= 1 and agents[name]["latency_ms"] >= 0
    assert agents["answer_generation"]["tokens"] > 0
    assert agents["faithfulness_validation"]["mean_score"] == r.report["validation"]["score"]
    assert all("latency_ms" in t and "attempt" in t for t in r.report["trace"])
    assert load_report(r.report_id, rconfig) == json.loads(Path(r.report_path or "").read_text())
    assert load_report("rpt-missing", rconfig) is None and load_report("../etc", rconfig) is None


def test_mermaid_lists_algorithm_1_nodes(rconfig: RetrievalConfig, stores: Stores) -> None:
    text = make_framework(rconfig, stores).mermaid()
    for node in (
        "cache_lookup", "query_classification", "query_rewriting", "query_decomposition",
        "next_subquery", "vector_retrieval", "graph_traversal", "community_retrieval",
        "evidence_aggregation", "subquery_answers", "context_construction", "answer_generation",
        "citation_mapping", "faithfulness_validation", "self_critique", "feedback",
        "retry_controller", "cache_manager", "generate_report",
    ):  # fmt: skip
        assert node in text


# =========================================================================== interfaces


def test_api_query_and_report(rconfig: RetrievalConfig, stores: Stores) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retrieval.api import create_app

    client = TestClient(create_app(make_framework(rconfig, stores)))
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/query", json={"query": LOCAL})
    assert resp.status_code == 200
    body = resp.json()
    assert (
        body["verified"]
        and body["citations"]
        and body["report_url"] == f"/report/{body['report_id']}"
    )
    report = client.get(body["report_url"]).json()
    assert report["report_id"] == body["report_id"] and report["citation_integrity"]["passed"]
    assert client.get("/report/rpt-0000000000000000").status_code == 404
    assert client.post("/query", json={"query": ""}).status_code == 422
    follow = client.post(
        "/query",
        json={
            "query": "What is its status?",
            "history": [{"role": "user", "content": "The Valve."}],
        },
    )
    assert "Replaced" in follow.json()["answer"]


def test_cli_query_and_report(
    tmp_path: Path, stores: Stores, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state"
    save_state(stores[0], stores[1], state)
    out = tmp_path / "out"
    code = cli_main(["query", LOCAL, "--offline", "--state-dir", str(state), "--out", str(out)])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["verified"] and summary["citation_integrity"]
    report_id = Path(summary["report"]).stem
    assert cli_main(["report", report_id, "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["report_id"] == report_id
    assert cli_main(["report", "rpt-0000000000000000", "--out", str(out)]) == 1


def test_run_all_chains_three_stages(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["run-all", "--offline", "--out", str(tmp_path / "run_all")])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["ok"]
    assert summary["stage2"]["nodes"] and summary["stage2"]["communities"]
    queries = summary["stage3"]["queries"]
    assert [q["query_type"] for q in queries] == ["Local", "Global", "Local"]
    assert all(q["verified"] and q["citation_integrity"] for q in queries)
