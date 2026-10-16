"""Offline Stage 3 helpers: a prompt-aware MockLLM for every retrieval task.

Shared by the test suite, ``query --offline`` and ``run-all --offline``. Each handler is
deterministic and reads only the prompt:

* classification, rewriting and decomposition echo the agent's own rule-based hints or
  draft, so the LLM code path runs end to end;
* answer generation is *extractive*: it copies the context sentences that best overlap
  the question and cites their blocks, or reports insufficient evidence;
* the critique and the faithfulness judge find no extra issues.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from common.llm import MockLLM
from common.text import split_sentences
from retrieval.text import terms

_BLOCK = re.compile(r"^\[(E\d+)\] \(type=(\w+)[^)]*\) (.*)$", re.M)
_CAPS = re.compile(r"[A-Z0-9][\w\-]*(?:\s+[A-Z0-9][\w\-]*)*")


def _line(prompt: str, key: str) -> str:
    m = re.search(rf"^{key}: (.*)$", prompt, re.M)
    return m.group(1).strip() if m else ""


def offline_classification(prompt: str, _s: str = "", _i: Sequence[bytes] = ()) -> dict[str, Any]:
    hints = _line(prompt, "HINTS")
    return {
        "entity_specific": "entity_specific=true" in hints,
        "thematic": "thematic=true" in hints,
        "confidence": 0.85,
        "rationale": "offline: accepted the knowledge-graph hints",
    }


def offline_rewriting(prompt: str, _s: str = "", _i: Sequence[bytes] = ()) -> dict[str, Any]:
    return {"rewritten": _line(prompt, "DRAFT")}


def offline_decomposition(prompt: str, _s: str = "", _i: Sequence[bytes] = ()) -> dict[str, Any]:
    draft = json.loads(_line(prompt, "DRAFT") or "[]")
    return {"complex": len(draft) > 1, "subqueries": draft}


def _short_answer(sentence: str, question_terms: set[str]) -> str:
    for run in _CAPS.findall(sentence):
        words = [w for w in run.split() if not (terms(w) & question_terms)]
        if words and not (terms(run) <= question_terms):
            return " ".join(words)
    return ""


def _candidates(prompt: str, q: set[str], qtype: str) -> list[tuple[float, str, str]]:
    out: list[tuple[float, str, str]] = []
    for cid, kind, text in _BLOCK.findall(prompt):
        if kind == "path":
            continue
        if kind == "community":
            text = text.split(": ", 1)[-1].split(" (members:")[0]
        # Prefer summaries for Global questions and atomic facts otherwise.
        bonus = 0.5 if kind == "community" and qtype == "Global" else 0.0
        bonus += 0.05 if kind == "fact" and qtype != "Global" else 0.0
        for sent in split_sentences(text):
            overlap = len(terms(sent) & q) / len(q) if q else 0.0
            if overlap > 0:
                out.append((overlap + bonus, cid, sent.strip().rstrip(".")))
    out.sort(key=lambda c: (-c[0], int(c[1][1:])))
    return out


def offline_answer(prompt: str, _s: str = "", _i: Sequence[bytes] = ()) -> dict[str, Any]:
    question = _line(prompt, "QUESTION")
    qtype = _line(prompt, "QUERY TYPE")
    mode = _line(prompt, "MODE")
    subs = [s for s in _line(prompt, "SUBQUESTIONS").split(" | ") if s and s != "-"]
    chosen: list[tuple[float, str, str]] = []
    # One best sentence per subquestion; a simple question may take a second sentence
    # that covers question terms the first one missed.
    for target in subs or [question]:
        q = terms(target)
        cands = [c for c in _candidates(prompt, q, qtype) if c[2] not in {x[2] for x in chosen}]
        if not cands:
            continue
        chosen.append(cands[0])
        if subs:
            continue
        covered = terms(cands[0][2]) & q
        limit = 3 if qtype == "Global" else 2
        for cand in cands[1:]:
            if len(chosen) >= limit:
                break
            new = (terms(cand[2]) & q) - covered
            if new:
                chosen.append(cand)
                covered |= new
    if not chosen:
        return {
            "answer": "Insufficient evidence to answer this question.",
            "insufficient_evidence": True,
            "short_answer": "",
        }
    answer = " ".join(f"{s} [{cid}]." for _, cid, s in chosen)
    return {
        "answer": answer,
        "insufficient_evidence": False,
        "short_answer": _short_answer(chosen[0][2], terms(question))
        if mode == "intermediate"
        else "",
    }


def offline_retrieval_llm(**overrides: Any) -> MockLLM:
    responses: dict[str, Any] = {
        "query_classification": offline_classification,
        "query_rewriting": offline_rewriting,
        "query_decomposition": offline_decomposition,
        "answer_generation": offline_answer,
        "self_critique": {"verdict": "verified", "issues": [], "corrected_answer": ""},
        "faithfulness_judge": {"verdicts": []},
    }
    responses.update(overrides)
    return MockLLM(responses)


def offline_all_stages_llm(**overrides: Any) -> MockLLM:
    """One MockLLM answering Stage 2 and Stage 3 tasks (used by ``run-all --offline``)."""
    from construction.testing import offline_construction_llm

    llm = offline_construction_llm()
    llm.responses.update(offline_retrieval_llm().responses)
    llm.responses.update(overrides)
    return llm
