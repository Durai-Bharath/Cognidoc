"""Offline Stage 2 helpers: a prompt-aware MockLLM and schema-valid J/M builders.

Shared by the test suite and ``construct --offline`` so both exercise the same behaviour.
The offline triple extractor is a small pattern matcher over the chunk TEXT block of the
prompt. It is not meant to be good, only deterministic and realistic enough to drive
every branch of the pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from common.llm import MockLLM
from common.text import split_sentences
from construction.models import stable_hash
from construction.ontology import load_ontology, relation_forms
from extraction.output_models import (
    AbbreviationJSON,
    BBoxJSON,
    ChunkJSON,
    ConfidenceJSON,
    DocumentInfo,
    DocumentJSON,
    DocumentStatistics,
    MetadataFile,
    QualityInfo,
    SectionJSON,
    SectionMetadata,
    SummaryJSON,
    TableJSON,
)

_TEXT_BLOCK = re.compile(r"^TEXT:\n<<<\n(?P<text>.*?)\n>>>", re.S | re.M)
_VERBS = [
    "should be repeated every",
    "is inspected every",
    "are inspected every",
    "is computed with",
    "is calculated with",
    "are summarised in",
    "is summarised in",
    "is controlled by",
    "is part of",
    "is located in",
    "is made by",
    "regulates",
    "controls",
    "requires",
    "prohibits",
    "produces",
    "measures",
    "monitors",
    "contains",
    "describes",
    "follow",
    "follows",
    "causes",
    "prevents",
    "uses",
    "lists",
    "shows",
    "teleports",
]
_SENTENCE = re.compile(
    r"^(?P<h>[A-Z][\w\s()\-/.]*?)\s+(?P<r>"
    + "|".join(re.escape(v) for v in _VERBS)
    + r")\s+(?P<t>.+)$"
)
_TAIL_CUT = re.compile(r"\s+(?:when|across|during|for|and|which|while|because|to)\s+|[,;]")
_HEDGE = re.compile(r"\b(?:may|might|possibly|perhaps|could)\b", re.I)


def guess_type(name: str) -> str:
    low = name.lower()
    if re.search(r"\b(?:iso|iec|standard|guideline|regulation)\b", low):
        return "Standard"
    if re.match(r"^\d", low):
        return "Quantity"
    if re.match(r"^(?:table|figure|section|equation|report)\b", low):
        return "Document"
    return ""


def offline_triples(
    prompt: str, _system: str = "", _images: Sequence[bytes] = ()
) -> dict[str, Any]:
    m = _TEXT_BLOCK.search(prompt)
    text = m.group("text") if m else ""
    triples: list[dict[str, Any]] = []
    lines = text.splitlines()
    # Linearised table rows: "Header: value; Header: value; ..."
    for line in lines:
        cells = [c.split(":", 1) for c in line.split("; ") if ":" in c]
        if len(cells) >= 2 and all(len(c) == 2 for c in cells):
            subject = cells[0][1].strip()
            for header, value in cells[1:]:
                header, value = header.strip(), value.strip()
                unit = re.search(r"\(([^)]+)\)", header)
                tail = f"{value} {unit.group(1)}" if unit and re.match(r"^\d", value) else value
                ttype = (
                    "Quantity"
                    if re.match(r"^\d", value)
                    else ("Status" if header.lower().startswith("status") else "")
                )
                triples.append(
                    {
                        "head": subject,
                        "head_type": "Component",
                        "relation": header,
                        "tail": tail,
                        "tail_type": ttype,
                        "confidence": 0.95,
                    }
                )
    prose = "\n".join(line for line in lines if line.count(": ") < 2)
    for sentence in split_sentences(prose):
        s = sentence.strip().rstrip(".")
        sm = _SENTENCE.match(s)
        if not sm:
            continue
        head = sm.group("h").strip()
        tail = _TAIL_CUT.split(sm.group("t"), maxsplit=1)[0].strip()
        if not tail:
            continue
        triples.append(
            {
                "head": head,
                "head_type": guess_type(head),
                "relation": sm.group("r"),
                "tail": tail,
                "tail_type": guess_type(tail),
                "confidence": 0.35 if _HEDGE.search(s) else 0.9,
            }
        )
    return {"triples": triples}


_ONTOLOGY = None


def offline_relation_choice(
    prompt: str, _system: str = "", _images: Sequence[bytes] = ()
) -> dict[str, Any]:
    """Strip modal/auxiliary words and retry the ontology lookup; else NONE."""
    global _ONTOLOGY
    if _ONTOLOGY is None:
        from common.config import PROJECT_ROOT

        _ONTOLOGY = load_ontology(PROJECT_ROOT / "ontology.yaml")
    m = re.search(r"^RELATION: (.+)$", prompt, re.M)
    relation = m.group(1) if m else ""
    words = [
        w
        for w in relation_forms(relation)[0].split()
        if w not in {"should", "must", "can", "be", "will"}
    ]
    hit = _ONTOLOGY.match_relation(" ".join(words)) if words else None
    if hit is None:
        return {"predicate": "NONE", "inverted": False, "reason": "no equivalent"}
    return {"predicate": hit.predicate, "inverted": hit.inverted, "reason": "offline"}


def offline_community_summary(
    prompt: str, _system: str = "", _images: Sequence[bytes] = ()
) -> dict[str, Any]:
    m = re.search(r"^ENTITIES: (.+)$", prompt, re.M)
    names = [n.strip() for n in (m.group(1) if m else "").split(";") if n.strip()]
    facts = prompt.split("FACTS:\n", 1)[1].strip() if "FACTS:\n" in prompt else ""
    return {
        "title": f"{names[0]} cluster" if names else "Community",
        "summary": " ".join(facts.splitlines()[:3]) or "Related entities.",
        "key_entities": names[:5],
    }


def offline_construction_llm(**overrides: Any) -> MockLLM:
    responses: dict[str, Any] = {
        "triple_extraction": offline_triples,
        "relation_normalization": offline_relation_choice,
        "community_summary": offline_community_summary,
    }
    responses.update(overrides)
    return MockLLM(responses)


# =========================================================================== J / M builders


def make_document(
    sections: Sequence[tuple[str, str]],
    *,
    title: str = "Test Document",
    source_id: str | None = None,
    abbreviations: dict[str, str] | None = None,
    tables: dict[int, list[tuple[str, list[str], list[list[str]]]]] | None = None,
    chunks_per_section: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a schema-valid ``(J, M)`` pair.

    ``sections`` is a list of ``(title, text)``; each section becomes one Stage 1 chunk
    (unless ``chunks_per_section`` is false). ``tables[i]`` adds ``(caption, headers, rows)``
    tables to section ``i``.
    """
    abbreviations = abbreviations or {}
    tables = tables or {}
    doc_source = source_id or stable_hash(title, *[t for s in sections for t in s])
    info = DocumentInfo(
        document_id=doc_source,
        document_type="PDF",
        document_category="Technical Report",
        category_confidence=0.9,
        title=title,
        author=None,
        creation_date=None,
        language="en",
        total_pages=max(1, len(sections)),
        file_size_bytes=1000,
        mime_type="application/pdf",
        source_file=f"{title}.pdf",
        processing_timestamp="2026-01-01T00:00:00+00:00",
    )
    abbrs = [
        AbbreviationJSON(abbreviation=k, expansion=v, occurrences=1)
        for k, v in abbreviations.items()
    ]
    secs: list[SectionJSON] = []
    metas: list[SectionMetadata] = []
    n_tables = 0
    for i, (stitle, text) in enumerate(sections):
        sid = f"sec-{i + 1}"
        page = i + 1
        chunks = (
            [
                ChunkJSON(
                    chunk_id=f"{sid}-c0",
                    order=0,
                    text=text,
                    page_start=page,
                    page_end=page,
                    token_count=len(text.split()),
                    previous_chunk=None,
                    next_chunk=None,
                    summary=stitle,
                )
            ]
            if chunks_per_section
            else []
        )
        sec_tables = [
            TableJSON(
                table_id=f"{sid}-tbl{j + 1}",
                number=str(j + 1),
                caption=cap,
                page=page,
                bbox=BBoxJSON(x=0, y=0, w=1, h=1),
                headers=headers,
                rows=rows,
                units={},
                key_attributes=[],
                description=cap,
                summary=cap,
                keywords=[],
                approved=True,
            )
            for j, (cap, headers, rows) in enumerate(tables.get(i, []))
        ]
        n_tables += len(sec_tables)
        secs.append(
            SectionJSON(
                section_id=sid,
                number=str(i + 1),
                title=stitle,
                level=1,
                page_start=page,
                page_end=page,
                text=text,
                keywords=[],
                summary=SummaryJSON(
                    summary=text[:100],
                    abstract="",
                    key_concepts=[],
                    important_entities=[],
                    confidence=0.9,
                    verification_status="Verified",
                ),
                confidence=ConfidenceJSON(score=0.9, level="High", components={}),
                subsections=[],
                tables=sec_tables,
                figures=[],
                equations=[],
                entities=[],
                abbreviations=[],
                cross_references=[],
                chunks=chunks,
            )
        )
        metas.append(
            SectionMetadata(
                section_id=sid,
                number=str(i + 1),
                title=stitle,
                level=1,
                page_start=page,
                page_end=page,
                summary=text[:100],
                keywords=[],
                confidence_score=0.9,
                confidence_level="High",
                subsection_ids=[],
                table_ids=[t.table_id for t in sec_tables],
                figure_ids=[],
                equation_ids=[],
                entity_ids=[],
                abbreviations=list(abbreviations),
                cross_reference_ids=[],
            )
        )
    stats = DocumentStatistics(
        total_sections=len(secs),
        total_subsections=0,
        total_tables=n_tables,
        total_figures=0,
        total_equations=0,
        total_entities=0,
        total_abbreviations=len(abbrs),
        total_cross_references=0,
        total_chunks=sum(len(s.chunks) for s in secs),
        total_words=sum(len(t.split()) for _, t in sections),
    )
    j = DocumentJSON(
        document=info, sections=secs, abbreviations=abbrs, relationships=[], statistics=stats
    )
    m = MetadataFile(
        document=info,
        sections=metas,
        statistics=stats,
        quality=QualityInfo(
            ocr_applied=False,
            ocr_quality_score=None,
            enhanced_pages=[],
            layout_confidence=0.9,
            document_confidence=0.9,
            confidence_level="High",
        ),
        review_flags=0,
    )
    return j.model_dump(mode="json"), m.model_dump(mode="json")
