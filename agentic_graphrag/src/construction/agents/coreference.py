"""A4 / Algorithm 5 - Coreference Resolution Agent.

Rule-based and deterministic:

* pronouns ("it", "they", …) → the most recent explicit subject in this chunk, else the
  last subject carried over from the previous chunk; unresolvable triples are dropped;
* document self-references ("this report", "the document") → the document title;
* abbreviations → expansions, from Stage 1's abbreviation repository in J (whole names,
  embedded tokens, and the "Expansion (ABBR)" form);
* duplicate mentions (case, articles, plural forms) → one canonical surface form.

Triples that become identical after canonicalisation are merged; the highest confidence wins.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import ClassVar

from pydantic import BaseModel, Field

from construction.agents.triple_extraction import TripleSet
from construction.context import ConstructionAgent
from construction.models import Triple, clean_surface, normalize_name
from extraction.agents.abbreviation import normalize_abbr

PRONOUNS = frozenset(
    "it its itself they them their theirs themselves he him his she her hers this that "
    "these those which who whom one".split()
)
DOCUMENT_REFERENCES = frozenset(
    {
        "this report",
        "this document",
        "the report",
        "the document",
        "this manual",
        "this paper",
        "this guide",
        "this study",
        "report",
        "document",
    }
)
_TRAILING_ABBR = re.compile(r"^(?P<long>.+?)\s*\((?P<abbr>[^()]{1,12})\)$")


class CoreferenceInput(BaseModel):
    triples: TripleSet
    abbreviations: dict[str, str] = Field(default_factory=dict)
    document_title: str = ""
    previous_subject: str | None = None


class CoreferenceOutput(BaseModel):
    triples: TripleSet
    replacements: list[dict[str, str]] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    last_subject: str | None = None


def mention_key(name: str) -> str:
    """Duplicate-mention key: normalised, with simple plural endings removed."""
    words = []
    for w in normalize_name(name).split():
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


class CoreferenceResolutionAgent(ConstructionAgent[CoreferenceInput, CoreferenceOutput]):
    name: ClassVar[str] = "coreference_resolution"
    algorithm: ClassVar[int] = 5
    agent_id: ClassVar[str] = "A4"

    def _execute(self, data: CoreferenceInput) -> CoreferenceOutput:
        abbr = {normalize_abbr(k): v for k, v in data.abbreviations.items() if k and v}
        out = CoreferenceOutput(triples=TripleSet(chunk_id=data.triples.chunk_id))
        out.triples.raw_count = data.triples.raw_count
        out.triples.dropped = list(data.triples.dropped)
        last_subject = data.previous_subject

        def canonical(name: str, role: str) -> str | None:
            text = clean_surface(name)
            low = text.lower()
            if low in DOCUMENT_REFERENCES and data.document_title:
                return data.document_title
            if low in PRONOUNS:
                return last_subject if role == "head" else None
            m = _TRAILING_ABBR.match(text)
            if m and normalize_abbr(m.group("abbr")) in abbr:
                text = m.group("long").strip()
            stripped = normalize_abbr(text)
            singular = stripped[:-1] if stripped.endswith("s") else stripped
            if stripped in abbr:
                return abbr[stripped]
            if singular in abbr and singular.isupper():
                return abbr[singular]
            tokens = [abbr.get(normalize_abbr(t), t) if t.isupper() else t for t in text.split()]
            return " ".join(tokens)

        resolved: list[Triple] = []
        for t in data.triples.triples:
            head = canonical(t.head, "head")
            tail = canonical(t.tail, "tail")
            if head is None or tail is None:
                missing = t.head if head is None else t.tail
                out.unresolved.append(f"{t.triple_id}: cannot resolve {missing!r}")
                continue
            for role, before, after in (("head", t.head, head), ("tail", t.tail, tail)):
                if before != after:
                    out.replacements.append(
                        {"triple": t.triple_id, "role": role, "from": before, "to": after}
                    )
            if t.head.lower() not in PRONOUNS:
                last_subject = head
            resolved.append(t.model_copy(update={"head": head, "tail": tail}))

        # Merge duplicate mentions → most frequent surface form (ties: shortest, i.e. the
        # singular, then first seen).
        forms: dict[str, Counter[str]] = {}
        order: dict[str, int] = {}
        for t in resolved:
            for name in (t.head, t.tail):
                key = mention_key(name)
                forms.setdefault(key, Counter())[name] += 1
                order.setdefault(name, len(order))
        chosen = {
            key: sorted(c, key=lambda n: (-c[n], len(n), order[n]))[0] for key, c in forms.items()
        }
        merged: dict[tuple[str, str, str], Triple] = {}
        for t in resolved:
            head, tail = chosen[mention_key(t.head)], chosen[mention_key(t.tail)]
            if normalize_name(head) == normalize_name(tail):
                out.unresolved.append(f"{t.triple_id}: became self-referential after resolution")
                continue
            t2 = t.model_copy(update={"head": head, "tail": tail})
            tkey = t2.key()
            if tkey not in merged or merged[tkey].confidence < t2.confidence:
                merged[tkey] = t2
        out.triples.triples = list(merged.values())
        out.last_subject = last_subject
        return out
