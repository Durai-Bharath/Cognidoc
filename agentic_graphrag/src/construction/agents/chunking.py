"""A2 / Algorithm 3 - Semantic Chunking Agent.

Stage 1 already produced section-aware semantic chunks (Context Memory, Alg. 9), so this
agent reuses them rather than re-chunking. It adds:

* stable, document-scoped Chunk IDs (``<doc_id>:<stage1 chunk id>``),
* overlap: the tail of the previous chunk in the same section, carried in ``context``
  so triples are extracted from ``text`` only, without double-counting,
* the section hierarchy (``section_path``), including the subsection a chunk starts in,
* a sentence-window split for sections with no Stage 1 chunks or oversized chunks,
* one linearised chunk per table (``Header: value; …`` per row).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.text import count_tokens, split_sentences
from common.validation import CheckResult, ValidationOutcome
from construction.agents.ingestion import ValidatedDocument
from construction.context import ConstructionAgent
from construction.models import SemanticChunk
from extraction.output_models import SectionJSON, TableJSON


class ChunkSet(BaseModel):
    doc_id: str
    chunks: list[SemanticChunk] = Field(default_factory=list)
    reused_stage1: int = 0
    resplit: int = 0
    table_chunks: int = 0
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


def window_sentences(text: str, max_tokens: int) -> list[str]:
    """Greedy sentence windows of at most ``max_tokens`` (a long sentence stands alone)."""
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        n = count_tokens(sentence)
        if current and size + n > max_tokens:
            windows.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        windows.append(" ".join(current))
    return windows


def tail_tokens(text: str, budget: int) -> str:
    """Trailing sentences of ``text`` fitting in ``budget`` tokens (the overlap)."""
    if budget <= 0:
        return ""
    picked: list[str] = []
    size = 0
    for sentence in reversed(split_sentences(text)):
        n = count_tokens(sentence)
        if picked and size + n > budget:
            break
        picked.insert(0, sentence)
        size += n
        if size >= budget:
            break
    return " ".join(picked)


def linearise_table(table: TableJSON) -> str:
    lines = [table.caption or f"Table {table.number or table.table_id}"]
    headers = table.headers
    for row in table.rows:
        cells = [
            f"{headers[i] if i < len(headers) and headers[i] else f'Column {i + 1}'}: {v}"
            for i, v in enumerate(row)
            if v.strip()
        ]
        if cells:
            lines.append("; ".join(cells))
    return "\n".join(lines)


def _subsection_path(section: SectionJSON, text: str) -> list[str]:
    path = [section.title]
    for sub in section.subsections:
        probe = sub.title.strip()
        if probe and probe in text:
            path.append(sub.title)
            break
    return path


class SemanticChunkingAgent(ConstructionAgent[ValidatedDocument, ChunkSet]):
    name: ClassVar[str] = "semantic_chunking"
    algorithm: ClassVar[int] = 3
    agent_id: ClassVar[str] = "A2"

    def _execute(self, data: ValidatedDocument) -> ChunkSet:
        cfg = self.settings.chunking
        doc_id = data.doc_id
        out = ChunkSet(doc_id=doc_id)
        chunks: list[SemanticChunk] = []

        def add(
            section: SectionJSON,
            text: str,
            *,
            local_id: str,
            source_id: str | None,
            page_start: int,
            page_end: int,
            kind: str = "text",
            context: str = "",
        ) -> None:
            tokens = count_tokens(text)
            if tokens < cfg.min_tokens:
                return
            chunks.append(
                SemanticChunk(
                    chunk_id=f"{doc_id}:{local_id}",
                    doc_id=doc_id,
                    order=len(chunks),
                    kind="table" if kind == "table" else "text",
                    text=text,
                    context=context,
                    section_id=section.section_id,
                    section_title=section.title,
                    section_path=_subsection_path(section, text),
                    level=section.level,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=tokens,
                    source_chunk_id=source_id,
                )
            )

        for section in data.document.sections:
            # Lines 2-3: section-aware split, reusing Stage 1 chunks.
            pieces: list[tuple[str, str, str | None, int, int]] = []
            if section.chunks:
                for c in sorted(section.chunks, key=lambda c: c.order):
                    if c.token_count <= cfg.max_tokens and count_tokens(c.text) <= cfg.max_tokens:
                        pieces.append((c.chunk_id, c.text, c.chunk_id, c.page_start, c.page_end))
                        out.reused_stage1 += 1
                    else:
                        for j, w in enumerate(window_sentences(c.text, cfg.max_tokens)):
                            pieces.append(
                                (f"{c.chunk_id}-w{j}", w, c.chunk_id, c.page_start, c.page_end)
                            )
                            out.resplit += 1
            elif section.text.strip():
                for j, w in enumerate(window_sentences(section.text, cfg.max_tokens)):
                    pieces.append(
                        (
                            f"{section.section_id}-w{j}",
                            w,
                            None,
                            section.page_start,
                            section.page_end,
                        )
                    )
                    out.resplit += 1
            # Line 4: overlapping context within the section.
            previous = ""
            for local_id, text, source_id, p0, p1 in pieces:
                add(
                    section,
                    text,
                    local_id=local_id,
                    source_id=source_id,
                    page_start=p0,
                    page_end=p1,
                    context=tail_tokens(previous, cfg.overlap_tokens),
                )
                previous = text
            if cfg.include_tables:
                for table in section.tables:
                    add(
                        section,
                        linearise_table(table),
                        local_id=table.table_id,
                        source_id=table.table_id,
                        page_start=table.page,
                        page_end=table.page,
                        kind="table",
                    )
                    out.table_chunks += 1

        # Line 5: unique ids and adjacency.
        for i, chunk in enumerate(chunks):
            chunk.previous_chunk = chunks[i - 1].chunk_id if i > 0 else None
            chunk.next_chunk = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
        out.chunks = chunks

        ids = [c.chunk_id for c in chunks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        known_sections = {s.section_id for s in data.document.sections}
        out.validation.add(CheckResult.from_failures("unique_chunk_ids", dupes))
        out.validation.add(
            CheckResult.from_failures("non_empty", [] if chunks else ["no chunks produced"])
        )
        out.validation.add(
            CheckResult.from_failures(
                "hierarchy_preserved",
                [
                    c.chunk_id
                    for c in chunks
                    if c.section_id not in known_sections or not c.section_path
                ],
            )
        )
        self.flag_if_failed(out.validation, "semantic chunking validation failed")
        return out
