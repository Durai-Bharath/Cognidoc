"""A8 / Algorithm 9 - Context Construction Agent.

Receives the aggregated evidence of every subquery (line 1), merges it again across
subqueries (an item found by two subqueries appears once), and fills the token budget
``context.token_budget`` greedily by score. A chunk that does not fit whole is truncated
when at least ``context.min_block_tokens`` remain; anything else that does not fit is
dropped and counted.

The selected items are rendered as citable blocks: communities, then reasoning paths and
facts (line 2), then document chunks (line 3). Each block gets a citation id ``E1…En``
and carries its provenance (chunk, node and edge ids, document id and title, page;
line 4). The rendered blocks form the LLM context (line 5). Intermediate subquery answers
travel alongside as non-citable notes.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.text import count_tokens
from retrieval.agents.evidence_aggregation import merge_items
from retrieval.context import RetrievalAgent
from retrieval.models import (
    AggregatedEvidence,
    ContextBlock,
    EvidenceItem,
    QueryType,
    RetrievalContext,
)

_KIND_ORDER = {"community": 0, "path": 1, "fact": 2, "chunk": 3}


class ContextInput(BaseModel):
    query: str
    query_type: QueryType
    evidence: list[AggregatedEvidence]
    sub_answers: dict[str, str] = Field(default_factory=dict)
    token_budget: int | None = None


class ContextConstructionAgent(RetrievalAgent[ContextInput, RetrievalContext]):
    name: ClassVar[str] = "context_construction"
    algorithm: ClassVar[int] = 9
    agent_id: ClassVar[str] = "A8"

    def _doc_title(self, doc_id: str, cache: dict[str, str]) -> str:
        if doc_id and doc_id not in cache:
            node = self.ctx.graph.get_node(doc_id)
            cache[doc_id] = node.name if node is not None and node.label == "Document" else ""
        return cache.get(doc_id, "")

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        words = text.split()
        out: list[str] = []
        for w in words:
            if count_tokens(" ".join([*out, w])) > max_tokens - 1:
                break
            out.append(w)
        return " ".join(out) + " …"

    def _execute(self, data: ContextInput) -> RetrievalContext:
        cfg = self.settings.context
        budget = data.token_budget or cfg.token_budget
        items, _ = merge_items([i for ae in data.evidence for i in ae.items])
        notes = "".join(f"{k}: {v}\n" for k, v in data.sub_answers.items())
        used = count_tokens(notes)
        chosen: list[tuple[EvidenceItem, str, bool]] = []
        dropped = 0
        for item in items:
            text = item.text.strip()
            if not text:
                dropped += 1
                continue
            cost = count_tokens(text) + 12  # citation id + provenance header
            remaining = budget - used
            if cost <= remaining:
                chosen.append((item, text, False))
                used += cost
            elif item.kind == "chunk" and remaining - 12 >= cfg.min_block_tokens:
                short = self._truncate(text, remaining - 12)
                chosen.append((item, short, True))
                used += count_tokens(short) + 12
            else:
                dropped += 1
        chosen.sort(key=lambda c: (_KIND_ORDER[c[0].kind], -c[0].score, c[0].evidence_id))
        titles: dict[str, str] = {}
        blocks: list[ContextBlock] = []
        for n, (item, text, truncated) in enumerate(chosen, start=1):
            blocks.append(
                ContextBlock(
                    citation_id=f"E{n}",
                    evidence_id=item.evidence_id,
                    kind=item.kind,
                    text=text,
                    chunk_ids=item.chunk_ids,
                    node_ids=item.node_ids,
                    edge_ids=item.edge_ids,
                    page=item.page,
                    doc_id=item.doc_id,
                    doc_title=self._doc_title(item.doc_id, titles),
                    score=item.score,
                    tokens=count_tokens(text),
                    truncated=truncated,
                )
            )
        return RetrievalContext(
            query=data.query,
            query_type=data.query_type,
            blocks=blocks,
            sub_answers=dict(data.sub_answers),
            token_count=used,
            token_budget=budget,
            dropped=dropped,
        )

    def confidence(self, output: RetrievalContext) -> float | None:
        return (
            round(min(1.0, output.token_count / output.token_budget), 4)
            if output.token_budget
            else None
        )
