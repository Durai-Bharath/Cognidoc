"""TRACEABILITY_STAGE2.md must cover A1-A15, the framework and CommunityBuilder."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import construction.agents as agents
import construction.community as community
import construction.graph as graph

ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[list[str]]:
    text = (ROOT / "TRACEABILITY_STAGE2.md").read_text()
    table = text.split("| Agent | Alg |", 1)[1].split("\n\n", 1)[0]
    rows = [line.strip().strip("|").split(" | ") for line in table.splitlines()[2:]]
    return [[c.strip() for c in r] for r in rows if len(r) == 6]


def _defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_agent_maps_to_class_file_and_tests() -> None:
    rows = _rows()
    assert [r[0] for r in rows] == ["—", *[f"A{i}" for i in range(1, 16)], "CB"]
    for agent_id, alg, _name, cls, file, tests in rows:
        cls_name = cls.strip("`")
        source = ROOT / file.strip("`")
        assert source.exists(), f"{agent_id}: {source} missing"
        assert re.search(rf"^class {cls_name}\b", source.read_text(), re.M), (
            f"{agent_id}: {cls_name} not defined in {source}"
        )
        if agent_id.startswith("A"):
            obj = getattr(agents, cls_name)
            assert obj.agent_id == agent_id and obj.algorithm == int(alg)
            assert obj in agents.ALL_AGENTS
        else:
            owner = graph if agent_id == "—" else community
            assert getattr(owner, cls_name)
        ids = re.findall(r"`([\w.]+)::(\w+)`", tests)
        assert ids, f"{agent_id}: no tests listed"
        for test_file, test_name in ids:
            assert test_name in _defined_tests(ROOT / "tests" / "construction" / test_file), (
                f"{agent_id}: {test_file}::{test_name} does not exist"
            )


def test_all_agents_are_traced_in_pdf_order() -> None:
    traced = [r[3].strip("`") for r in _rows() if r[0].startswith("A")]
    assert traced == [a.__name__ for a in agents.ALL_AGENTS]
    assert [a.agent_id for a in agents.ALL_AGENTS] == [f"A{i}" for i in range(1, 16)]
