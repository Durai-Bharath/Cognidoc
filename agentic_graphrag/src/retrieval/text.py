"""Text helpers shared by the retrieval agents: terms, support, entity mentions, markers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from common.text import STOPWORDS, numbers_in
from construction.models import GraphNode, normalize_name
from construction.stores.graph_store import GraphStore
from extraction.agents.abbreviation import find_pairs

MARKER = re.compile(r"\[(E\d+)\]")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,9}s?\b")
_MARKER_GROUP = re.compile(r"\s*\[(?:E\d+)(?:\s*,\s*E\d+)*\]")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:[.,]\d+)?")
INSUFFICIENT = "insufficient evidence"

# Question words are not content: "what", "which" … never need evidential support.
QUESTION_WORDS = frozenset(
    "what which who whom whose when where why how many much does do did is are was were "
    "describe explain list tell give show find name".split()
)


def stem(word: str) -> str:
    """Light, deterministic suffix stripping so "inspections" ~ "inspection" ~ "inspected"."""
    w = word.lower()
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            base = w[: -len(suffix)]
            return base + "y" if suffix in ("ies", "ied") else base
    return w


def terms(text: str) -> set[str]:
    """Stemmed content words plus numbers (markers removed)."""
    text = MARKER.sub(" ", text)
    out: set[str] = set()
    for tok in _WORD.findall(text):
        low = tok.lower()
        if low in STOPWORDS or low in QUESTION_WORDS:
            continue
        if low[0].isdigit():
            out.add(low.replace(",", "."))
        elif len(low) > 1:
            out.add(stem(low))
    return out


def strip_markers(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", _MARKER_GROUP.sub("", text)).strip()


def is_insufficient(text: str) -> bool:
    return INSUFFICIENT in text.lower()


def support_score(statement: str, evidence: str) -> tuple[float, list[str]]:
    """Share of a statement's terms found in the evidence, and the terms that are missing.

    Numbers are hard requirements: a number absent from the evidence caps support at 0.
    """
    claim = terms(statement)
    if not claim:
        return 1.0, []
    have = terms(evidence)
    missing = sorted(t for t in claim if t not in have)
    nums = {n.rstrip("%") for n in numbers_in(strip_markers(statement))}
    if any(n not in have for n in nums):
        return 0.0, missing
    return round(1 - len(missing) / len(claim), 4), missing


def coverage(query: str, answer: str) -> float:
    """Share of the query's terms the answer addresses (completeness)."""
    q = terms(query)
    if not q:
        return 1.0
    return round(len(q & terms(answer)) / len(q), 4)


@dataclass
class EntityMention:
    node: GraphNode
    surface: str
    score: float


def _surface_forms(node: GraphNode) -> Iterable[str]:
    yield node.name
    yield from node.aliases


def find_entity_mentions(
    text: str, graph: GraphStore, *, min_chars: int = 2
) -> list[EntityMention]:
    """KG entities whose name or alias occurs in ``text`` (word-bounded, longest first)."""
    norm = " " + normalize_name(re.sub(r"[^\w\s\-]", " ", text)) + " "
    found: dict[str, EntityMention] = {}
    taken: list[tuple[int, int]] = []
    forms: list[tuple[str, GraphNode]] = []
    for node in graph.nodes("Entity"):
        for form in _surface_forms(node):
            key = normalize_name(re.sub(r"[^\w\s\-]", " ", form))
            if len(key) >= min_chars:
                forms.append((key, node))
    forms.sort(key=lambda f: -len(f[0]))
    for key, node in forms:
        for m in re.finditer(rf"(?<=\s){re.escape(key)}(?=\s)", norm):
            span = (m.start(), m.end())
            if any(span[0] < e and s < span[1] for s, e in taken):
                continue
            taken.append(span)
            if node.id not in found:
                found[node.id] = EntityMention(node=node, surface=key, score=1.0)
    return list(found.values())


def acronym_table(graph: GraphStore, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Acronym → expansion: ``extra`` first, then capitalised KG aliases, then
    "Long Form (LF)" definitions in stored chunk texts (Stage 1's Schwartz-Hearst matcher)."""
    table = dict(extra or {})
    for node in graph.nodes("Entity"):
        for alias in node.aliases:
            if ACRONYM.fullmatch(alias) and alias != node.name:
                table.setdefault(alias, node.name)
    for chunk in graph.nodes("Chunk"):
        for short, long in find_pairs(str(chunk.properties.get("text", "")), 10):
            table.setdefault(short, long)
    return table
