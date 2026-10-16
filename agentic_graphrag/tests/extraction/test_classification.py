"""Algorithm 3 - DocumentClassificationAgent."""

from __future__ import annotations

from pathlib import Path

from common.base_agent import AgentContext
from extraction.agents.classification import UNKNOWN, DocumentClassificationAgent
from extraction.agents.router import RouterAgent
from extraction.schemas import (
    BBox,
    ClassificationInput,
    DocumentProperties,
    DocumentType,
    PageData,
    ParsedDocument,
    RawBlock,
    RouterInput,
)


def _parsed(text: str, title: str = "Doc") -> ParsedDocument:
    blocks = [
        RawBlock(
            block_id="b0", page=1, bbox=BBox(x=72, y=50, w=300, h=20), text=title, font_size=20
        ),
        RawBlock(
            block_id="b1",
            page=1,
            bbox=BBox(x=72, y=90, w=200, h=14),
            text="1 Purpose",
            font_size=14,
            bold=True,
        ),
        RawBlock(
            block_id="b2", page=1, bbox=BBox(x=72, y=120, w=450, h=60), text=text, font_size=10
        ),
        RawBlock(
            block_id="b3",
            page=1,
            bbox=BBox(x=72, y=200, w=450, h=60),
            kind="table",
            table_cells=[["a", "b"]],
        ),
        RawBlock(
            block_id="b4",
            page=1,
            bbox=BBox(x=72, y=300, w=450, h=12),
            text="Introduction ........ 3",
            font_size=10,
        ),
    ]
    return ParsedDocument(
        document_id="d",
        source_path="x",
        doc_type=DocumentType.PDF,
        properties=DocumentProperties(title=title, page_count=1, language="en"),
        pages=[PageData(page_number=1, width=600, height=800, blocks=blocks)],
        raw_text=f"{title}\n{text}",
        workdir=".",
    )


SOP = (
    "This standard operating procedure (SOP) defines the purpose, scope, responsibilities "
    "and procedure steps. Operators shall follow each step and obtain approval for every "
    "revision to remain in compliance with the work instruction."
)


def test_accepts_argmax_class_above_tau(ctx: AgentContext) -> None:
    out = DocumentClassificationAgent(ctx).run(ClassificationInput(parsed=_parsed(SOP))).unwrap()
    assert out.category == "Standard Operating Procedure (SOP)" == out.predicted_category
    assert out.confidence == max(out.scores.values()) and not out.flagged
    assert len(out.scores) == 12
    f = out.features
    assert (
        "1 Purpose" in f.headings and f.table_count == 1 and f.table_of_contents == ["Introduction"]
    )


def test_below_tau_is_unknown_and_flagged(ctx: AgentContext) -> None:
    ctx.config.classification.tau = 0.99
    out = DocumentClassificationAgent(ctx).run(ClassificationInput(parsed=_parsed(SOP))).unwrap()
    assert out.category == UNKNOWN and out.flagged
    assert out.predicted_category == "Standard Operating Procedure (SOP)"
    assert ctx.review_queue.items[-1].agent == "DocumentClassificationAgent"


def test_tau_boundary_is_accepted(ctx: AgentContext) -> None:
    agent = DocumentClassificationAgent(ctx)
    first = agent.run(ClassificationInput(parsed=_parsed(SOP))).unwrap()
    ctx.config.classification.tau = first.confidence  # Conf(C) == τ → not "< τ"
    again = agent.run(ClassificationInput(parsed=_parsed(SOP))).unwrap()
    assert again.category == first.predicted_category and not again.flagged


def test_document_without_text_defers_classification(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    parsed = (
        RouterAgent(ctx)
        .run(RouterInput(path=str(samples["scan"]), workdir=str(tmp_path)))
        .unwrap()
        .parsed
    )
    out = DocumentClassificationAgent(ctx).run(ClassificationInput(parsed=parsed)).unwrap()
    assert out.category == UNKNOWN and not out.based_on_text and not out.flagged
    assert not ctx.review_queue.items
