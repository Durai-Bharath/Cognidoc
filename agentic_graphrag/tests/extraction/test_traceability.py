"""TRACEABILITY_STAGE1.md must cover all 21 algorithms with real classes, files and tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import extraction.agents as agents
import extraction.graph as graph

ROOT = Path(__file__).resolve().parents[2]


def _rows() -> list[list[str]]:
    text = (ROOT / "TRACEABILITY_STAGE1.md").read_text()
    table = text.split("| Alg | Algorithm (PDF) |", 1)[1].split("\n\n", 1)[0]
    rows = [line.strip().strip("|").split(" | ") for line in table.splitlines()[2:]]
    return [[c.strip() for c in r] for r in rows if r and r[0].strip().isdigit()]


def _defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_algorithm_maps_to_class_file_and_tests() -> None:
    rows = _rows()
    assert [int(r[0]) for r in rows] == list(range(1, 22))
    for alg, _name, cls, file, tests in rows:
        cls_name = cls.strip("`")
        source = ROOT / file.strip("`")
        assert source.exists(), f"Alg {alg}: {source} missing"
        assert re.search(rf"^class {cls_name}\b", source.read_text(), re.M), (
            f"Alg {alg}: {cls_name} not defined in {source}"
        )
        owner = graph if alg == "1" else agents
        obj = getattr(owner, cls_name)
        if alg != "1":
            assert obj.algorithm == int(alg) and obj in agents.ALL_AGENTS
        ids = re.findall(r"`([\w.]+)::(\w+)`", tests)
        assert ids, f"Alg {alg}: no tests listed"
        for test_file, test_name in ids:
            assert test_name in _defined_tests(ROOT / "tests" / "extraction" / test_file), (
                f"Alg {alg}: {test_file}::{test_name} does not exist"
            )


def test_all_agents_are_traced() -> None:
    traced = {r[2].strip("`") for r in _rows()}
    assert {a.__name__ for a in agents.ALL_AGENTS} <= traced
