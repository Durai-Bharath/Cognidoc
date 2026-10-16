"""A10 / Algorithm 11 - Citation Mapping Agent.

Splits the generated answer into statements (sentences) and, for each one:

* resolves its ``[E#]`` markers to context blocks: the blocks' chunks, graph nodes and
  edges, document and page (lines 3-4). A marker that names no context block is recorded
  as invalid; A11 treats any invalid marker as a citation-integrity failure;
* attaches the best-supporting block when a statement has no marker at all and that
  block's support is ≥ ``citation.inferred_min_support``, marked ``inferred``;
* identifies the supporting graph nodes (line 2): KG entities named in the statement
  whose name also appears in a cited block.

The output keeps the answer text and adds per-statement citations plus the de-duplicated
list of sources (line 5).
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel

from common.text import split_sentences
from retrieval.context import RetrievalAgent
from retrieval.models import (
    ContextBlock,
    GeneratedAnswer,
    GroundedAnswer,
    RetrievalContext,
    SourceRef,
    StatementCitation,
)
from retrieval.text import (
    MARKER,
    find_entity_mentions,
    is_insufficient,
    strip_markers,
    support_score,
)


class CitationInput(BaseModel):
    answer: GeneratedAnswer
    context: RetrievalContext


def source_ref(block: ContextBlock) -> SourceRef:
    return SourceRef(
        citation_id=block.citation_id,
        kind=block.kind,
        chunk_ids=block.chunk_ids,
        node_ids=block.node_ids,
        edge_ids=block.edge_ids,
        page=block.page,
        doc_id=block.doc_id,
        doc_title=block.doc_title,
    )


_TRAILING_MARKERS = re.compile(r"([.!?])((?:\s*\[E\d+\])+)")
_CITED_END = re.compile(r"(?<=\][.!?])\s+")


def split_statements(answer: str) -> list[str]:
    """Sentences with their markers. Markers placed after the full stop ("X. [E1]") are
    moved before it, and a cited sentence ends at its marker even when the next sentence
    starts in lower case."""
    text = _TRAILING_MARKERS.sub(lambda m: m.group(2) + m.group(1), answer)
    out: list[str] = []
    for part in _CITED_END.split(text):
        out.extend(s for s in split_sentences(part) if strip_markers(s))
    return out


class CitationMappingAgent(RetrievalAgent[CitationInput, GroundedAnswer]):
    name: ClassVar[str] = "citation_mapping"
    algorithm: ClassVar[int] = 11
    agent_id: ClassVar[str] = "A10"

    def _execute(self, data: CitationInput) -> GroundedAnswer:
        ctx = data.context
        threshold = self.settings.citation.inferred_min_support
        out = GroundedAnswer(
            answer=data.answer.answer, insufficient_evidence=data.answer.insufficient_evidence
        )
        seen: dict[str, SourceRef] = {}
        for i, sentence in enumerate(split_statements(data.answer.answer), start=1):
            text = strip_markers(sentence)
            if not text:
                continue
            stmt = StatementCitation(statement_id=f"S{i}", text=text)
            blocks: list[ContextBlock] = []
            for marker in dict.fromkeys(MARKER.findall(sentence)):
                block = ctx.block(marker)
                if block is None:
                    stmt.invalid_markers.append(marker)
                else:
                    blocks.append(block)
            if not blocks and not stmt.invalid_markers and not is_insufficient(text):
                scored = [(support_score(text, b.text)[0], b) for b in ctx.blocks]
                scored.sort(key=lambda s: (-s[0], s[1].citation_id))
                if scored and scored[0][0] >= threshold:
                    blocks = [scored[0][1]]
                    stmt.inferred = True
            stmt.citation_ids = [b.citation_id for b in blocks]
            stmt.sources = [source_ref(b) for b in blocks]
            cited_text = " ".join(b.text for b in blocks).lower()
            stmt.entity_ids = [
                m.node.id
                for m in find_entity_mentions(text, self.ctx.graph)
                if m.node.name.lower() in cited_text
            ]
            for ref in stmt.sources:
                seen.setdefault(ref.citation_id, ref)
            out.invalid_markers += stmt.invalid_markers
            out.statements.append(stmt)
        out.citations = sorted(seen.values(), key=lambda r: int(r.citation_id[1:]))
        return out

    def confidence(self, output: GroundedAnswer) -> float | None:
        if not output.statements:
            return 0.0
        return round(sum(bool(s.sources) for s in output.statements) / len(output.statements), 4)
