"""Algorithm 9 - Context Memory Agent."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.embedder import Vector, cosine
from common.text import count_tokens, keywords, split_sentences
from common.validation import CheckResult, ValidationOutcome
from extraction.schemas import (
    Chunk,
    ChunkEdge,
    ElementType,
    LayoutElement,
    RetrievalIndex,
    SectionMemory,
    SectionTask,
)

_ENTITY = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,}[a-z]*\d*|[A-Z][a-z]+[A-Z]\w+)\b"
)


class ChunkSummaryLLM(BaseModel):
    summary: str = ""


@dataclass
class Unit:
    idx: int
    element: LayoutElement
    text: str
    tokens: int


def element_text(el: LayoutElement) -> str:
    if el.type == ElementType.TABLE:
        cells = el.attrs.get("cells") or []
        header = " | ".join(c or "" for c in cells[0]) if cells else ""
        return f"{el.attrs.get('caption', 'Table')}: {header}".strip()
    if el.type == ElementType.FIGURE:
        return str(el.attrs.get("caption") or "[figure]")
    return el.text


def entities_in(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in _ENTITY.finditer(text)))[:20]


class ContextMemoryAgent(BaseAgent[SectionTask, SectionMemory]):
    name = "ContextMemoryAgent"
    algorithm = 9

    def _units(self, task: SectionTask) -> list[Unit]:
        cap = self.ctx.config.context_memory.max_chunk_tokens
        units: list[Unit] = []
        for el in task.section.all_elements():
            text = element_text(el).strip()
            if not text:
                continue
            pieces = [text]
            if count_tokens(text) > cap:
                pieces, buf = [], ""
                for sent in split_sentences(text) or [text]:
                    if buf and count_tokens(buf + " " + sent) > cap:
                        pieces.append(buf)
                        buf = sent
                    else:
                        buf = f"{buf} {sent}".strip()
                if buf:
                    pieces.append(buf)
            for piece in pieces:
                units.append(Unit(len(units), el, piece, count_tokens(piece)))
        return units

    def _execute(self, data: SectionTask) -> SectionMemory:
        cfg = self.ctx.config.context_memory
        section = data.section
        units = self._units(data)
        total = sum(u.tokens for u in units)  # L = TokenCount(S)
        embedder = self.ctx.embedder

        if total <= cfg.context_limit:
            groups = [units]
        else:
            groups = self._semantic_chunks(units)

        chunks: list[Chunk] = []
        vectors: list[Vector] = []
        for order, group in enumerate(groups):
            text = "\n\n".join(u.text for u in group) if group else ""
            vec = embedder.embed_one(text) if text else np.zeros(embedder.dim, dtype=np.float32)
            vectors.append(vec)

            def by_type(t: ElementType, group: list[Unit] = group) -> list[str]:
                return list(
                    dict.fromkeys(u.element.element_id for u in group if u.element.type == t)
                )

            chunks.append(
                Chunk(
                    chunk_id=f"{section.section_id}-c{order}",
                    parent_section=section.section_id,
                    order=order,
                    text=text,
                    unit_ids=[u.idx for u in group],
                    page_start=min((u.element.page for u in group), default=section.page_start),
                    page_end=max((u.element.page for u in group), default=section.page_end),
                    token_count=count_tokens(text),
                    summary=self._chunk_summary(text),
                    key_concepts=keywords(text, 6),
                    entities=entities_in(text),
                    tables=by_type(ElementType.TABLE),
                    figures=by_type(ElementType.FIGURE),
                    equations=by_type(ElementType.EQUATION),
                    references=by_type(ElementType.REFERENCE),
                    embedding=[round(float(x), 6) for x in vec],
                )
            )
        self._link(chunks)
        edges = self._edges(chunks, vectors)
        section_vec = (
            np.mean(np.stack(vectors), axis=0)
            if vectors
            else np.zeros(embedder.dim, dtype=np.float32)
        )
        norm = float(np.linalg.norm(section_vec)) or 1.0
        memory = SectionMemory(
            section_id=section.section_id,
            token_count=total,
            stored_whole=total <= cfg.context_limit,
            chunks=chunks,
            edges=edges,
            summary=" ".join(c.summary for c in chunks[:3])[:600],
            embedding=[round(float(x) / norm, 6) for x in section_vec],
            index=RetrievalIndex(
                ids=[c.chunk_id for c in chunks], vectors=[c.embedding for c in chunks]
            ),
        )
        expected_units = len(units)
        memory.validation = self.validate(memory, expected_units)
        if not memory.validation.passed:
            # Lines 47-48: reconstruct invalid entries and revalidate.
            memory = self.reconstruct(memory, units)
            memory.validation = self.validate(memory, expected_units)
            memory.reconstructed = True
        memory.approved = memory.validation.passed
        if not memory.approved:
            self.flag(
                "context memory inconsistent after reconstruction",
                item_ref=section.section_id,
                details={"failures": memory.validation.failure_reasons()},
            )
        return memory

    # ------------------------------------------------------------------ chunking
    def _semantic_chunks(self, units: list[Unit]) -> list[list[Unit]]:
        """S = {C_1..C_n}: greedy chunks that break on token budget or a semantic shift."""
        cfg = self.ctx.config.context_memory
        if not units:
            return [[]]
        vecs = self.ctx.embedder.embed([u.text for u in units])
        groups: list[list[Unit]] = [[units[0]]]
        centroid = vecs[0].copy()
        tokens = units[0].tokens
        for unit, vec in zip(units[1:], vecs[1:], strict=True):
            sim = cosine(centroid, vec)
            over_budget = tokens + unit.tokens > cfg.max_chunk_tokens
            shift = tokens >= cfg.min_chunk_tokens and sim < cfg.semantic_break_threshold
            if over_budget or shift:
                groups.append([unit])
                centroid = vec.copy()
                tokens = unit.tokens
            else:
                groups[-1].append(unit)
                centroid = centroid + vec
                tokens += unit.tokens
        return groups

    def _chunk_summary(self, text: str) -> str:
        if not text:
            return ""
        if self.ctx.config.context_memory.llm_chunk_summaries:
            out = self.llm_json(
                task="chunk_summary",
                system="You summarise document passages in one or two factual sentences "
                "using only information present in the passage.",
                prompt=f"Passage:\n{text[:6000]}",
                schema=ChunkSummaryLLM,
            )
            if out.summary.strip():
                return out.summary.strip()
        sentences = split_sentences(text)
        first = sentences[0] if sentences else text
        return " ".join(first.split()[:40])

    @staticmethod
    def _link(chunks: list[Chunk]) -> None:
        for i, c in enumerate(chunks):
            c.order = i
            c.previous_chunk = chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None

    def _edges(self, chunks: list[Chunk], vectors: list[Vector]) -> list[ChunkEdge]:
        """G = (V, E): sequential edges plus semantic edges between non-adjacent chunks."""
        threshold = self.ctx.config.context_memory.link_threshold
        edges = [
            ChunkEdge(source=a.chunk_id, target=b.chunk_id, kind="next", weight=1.0)
            for a, b in itertools.pairwise(chunks)
        ]
        for i in range(len(chunks)):
            for j in range(i + 2, len(chunks)):
                sim = cosine(vectors[i], vectors[j])
                if sim >= threshold:
                    edges.append(
                        ChunkEdge(
                            source=chunks[i].chunk_id,
                            target=chunks[j].chunk_id,
                            kind="semantic",
                            weight=round(sim, 4),
                        )
                    )
        return edges

    # ------------------------------------------------------------------ validation
    @staticmethod
    def validate(memory: SectionMemory, expected_units: int) -> ValidationOutcome:
        outcome = ValidationOutcome()
        seen = [i for c in memory.chunks for i in c.unit_ids]
        missing = sorted(set(range(expected_units)) - set(seen))
        problems = [f"units {missing} not in any chunk"] if missing else []
        dupes = sorted({i for i in seen if seen.count(i) > 1})
        if dupes:
            problems.append(f"units {dupes} in several chunks")
        if expected_units and any(not c.text.strip() for c in memory.chunks):
            problems.append("empty chunk")
        outcome.add(CheckResult.from_failures("no_missing_chunks", problems))

        sequence: list[str] = []
        for i, c in enumerate(memory.chunks):
            if c.order != i:
                sequence.append(f"{c.chunk_id} has order {c.order}, expected {i}")
            prev = memory.chunks[i - 1].chunk_id if i > 0 else None
            nxt = memory.chunks[i + 1].chunk_id if i + 1 < len(memory.chunks) else None
            if c.previous_chunk != prev or c.next_chunk != nxt:
                sequence.append(f"{c.chunk_id} has broken prev/next pointers")
        outcome.add(CheckResult.from_failures("correct_chunk_sequence", sequence))

        ids = {c.chunk_id for c in memory.chunks}
        links = [
            f"edge {e.source}->{e.target} invalid"
            for e in memory.edges
            if e.source not in ids
            or e.target not in ids
            or e.source == e.target
            or not -1.0 <= e.weight <= 1.0
        ]
        outcome.add(CheckResult.from_failures("valid_semantic_links", links))

        meta: list[str] = []
        dims = {len(c.embedding) for c in memory.chunks}
        if len(dims) > 1:
            meta.append(f"inconsistent embedding dims {sorted(dims)}")
        for c in memory.chunks:
            if c.parent_section != memory.section_id:
                meta.append(f"{c.chunk_id} parent {c.parent_section} != {memory.section_id}")
            if c.token_count != count_tokens(c.text):
                meta.append(f"{c.chunk_id} token_count mismatch")
            if c.page_start > c.page_end:
                meta.append(f"{c.chunk_id} page range reversed")
        if memory.index.ids != [c.chunk_id for c in memory.chunks]:
            meta.append("retrieval index out of sync with chunks")
        outcome.add(CheckResult.from_failures("consistent_metadata", meta))
        return outcome

    def reconstruct(self, memory: SectionMemory, units: list[Unit]) -> SectionMemory:
        memory = memory.model_copy(deep=True)
        by_idx = {u.idx: u for u in units}
        seen: set[int] = set()
        for c in memory.chunks:
            c.unit_ids = [i for i in c.unit_ids if i in by_idx and i not in seen]
            seen.update(c.unit_ids)
        missing = [u for u in units if u.idx not in seen]
        memory.chunks = [c for c in memory.chunks if c.unit_ids]
        if missing:
            vec = self.ctx.embedder.embed_one("\n\n".join(u.text for u in missing))
            memory.chunks.append(
                Chunk(
                    chunk_id=f"{memory.section_id}-c{len(memory.chunks)}r",
                    parent_section=memory.section_id,
                    order=len(memory.chunks),
                    text="",
                    unit_ids=[u.idx for u in missing],
                    page_start=1,
                    page_end=1,
                    token_count=0,
                    embedding=[round(float(x), 6) for x in vec],
                )
            )
        for c in memory.chunks:
            group = [by_idx[i] for i in c.unit_ids]
            c.parent_section = memory.section_id
            c.text = "\n\n".join(u.text for u in group)
            c.token_count = count_tokens(c.text)
            c.page_start = min(u.element.page for u in group)
            c.page_end = max(u.element.page for u in group)
            c.summary = c.summary or self._chunk_summary(c.text)
        self._link(memory.chunks)
        ids = {c.chunk_id for c in memory.chunks}
        memory.edges = [
            e
            for e in memory.edges
            if e.source in ids
            and e.target in ids
            and e.source != e.target
            and -1.0 <= e.weight <= 1.0
        ]
        memory.index = RetrievalIndex(
            ids=[c.chunk_id for c in memory.chunks], vectors=[c.embedding for c in memory.chunks]
        )
        return memory


def search(index: RetrievalIndex, query: Vector, k: int = 5) -> list[tuple[str, float]]:
    """Top-k cosine search over a retrieval index."""
    if not index.ids:
        return []
    mat = np.asarray(index.vectors, dtype=np.float32)
    scores = mat @ query
    top = np.argsort(-scores)[:k]
    return [(index.ids[i], float(scores[i])) for i in top]
