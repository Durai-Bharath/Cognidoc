"""A1 Document Ingestion and A2 Semantic Chunking."""

from __future__ import annotations

import copy

from construction.agents.chunking import (
    ChunkSet,
    SemanticChunkingAgent,
    linearise_table,
    tail_tokens,
    window_sentences,
)
from construction.agents.ingestion import (
    DocumentIngestionAgent,
    IngestionInput,
    ValidatedDocument,
    document_id_for,
)
from construction.context import ConstructionContext
from construction.testing import make_document
from extraction.output_models import BBoxJSON, TableJSON

LONG = " ".join(f"Sentence number {i} talks about the pump seal." for i in range(60))


def _ingest(ctx: ConstructionContext, j: dict, m: dict):  # type: ignore[no-untyped-def,type-arg]
    return DocumentIngestionAgent(ctx).run(IngestionInput(document=j, metadata=m))


def _validated(ctx: ConstructionContext, **kw) -> ValidatedDocument:  # type: ignore[no-untyped-def]
    j, m = make_document(**kw)
    return _ingest(ctx, j, m).unwrap()


def test_valid_document_is_registered_with_session(cctx: ConstructionContext) -> None:
    j, m = make_document(
        [("Intro", "The pump uses oil.")], abbreviations={"HCU": "Hydraulic Control Unit"}
    )
    d = _ingest(cctx, j, m).unwrap()
    assert d.doc_id == document_id_for(j) and d.doc_id.startswith("doc-")
    assert d.validation.passed and not d.reingested and d.session_id
    assert d.abbreviations == {"HCU": "Hydraulic Control Unit"}
    node = cctx.graph.get_node(d.doc_id)
    assert node is not None and node.label == "Document" and node.properties["ingest_count"] == 1
    assert cctx.document_id == d.doc_id and cctx.session_id == d.session_id


def test_document_id_is_content_hash_and_reingestion_is_detected(cctx: ConstructionContext) -> None:
    j, m = make_document([("Intro", "The pump uses oil.")])
    first = _ingest(cctx, j, m).unwrap()
    j2 = copy.deepcopy(j)
    j2["document"]["processing_timestamp"] = "2030-01-01T00:00:00+00:00"  # volatile field
    second = _ingest(cctx, j2, m).unwrap()
    assert second.doc_id == first.doc_id and second.reingested
    assert second.session_id != first.session_id
    assert cctx.graph.get_node(first.doc_id).properties["ingest_count"] == 2  # type: ignore[union-attr]
    j3, m3 = make_document([("Intro", "The pump uses water.")])
    assert _ingest(cctx, j3, m3).unwrap().doc_id != first.doc_id


def test_invalid_schema_terminates_with_error_report(cctx: ConstructionContext) -> None:
    j, m = make_document([("Intro", "text")])
    del j["sections"][0]["chunks"]
    result = _ingest(cctx, j, m)
    assert not result.ok and result.error is not None
    assert result.error.error_type == "DocumentValidationError"
    assert "document_schema" in result.error.message
    flagged = cctx.review_queue.items[-1]
    assert flagged.agent == "document_ingestion" and flagged.details["errors"]
    assert cctx.graph.nodes("Document") == []


def test_metadata_mismatch_and_empty_document_are_rejected(cctx: ConstructionContext) -> None:
    j, _ = make_document([("Intro", "a b c")], source_id="one")
    _, m = make_document([("Intro", "a b c")], source_id="two")
    result = _ingest(cctx, j, m)
    assert not result.ok and "metadata_matches_document" in result.error.message  # type: ignore[union-attr]
    cctx.settings.ingestion.require_metadata_match = False
    assert _ingest(cctx, j, m).ok
    j, m = make_document([("Intro", "")], chunks_per_section=False)
    result = _ingest(cctx, j, m)
    assert not result.ok and "has_content" in result.error.message  # type: ignore[union-attr]


def test_chunking_reuses_stage1_chunks_with_overlap_and_hierarchy(
    cctx: ConstructionContext,
) -> None:
    d = _validated(
        cctx,
        sections=[
            ("Intro", "The pump uses oil. The seal is checked."),
            ("Method", "Valves are replaced."),
        ],
        tables={1: [("Table 1: Intervals", ["Component", "Interval (h)"], [["Seal", "500"]])]},
    )
    out: ChunkSet = SemanticChunkingAgent(cctx).run(d).unwrap()
    ids = [c.chunk_id for c in out.chunks]
    assert ids == [f"{d.doc_id}:sec-1-c0", f"{d.doc_id}:sec-2-c0", f"{d.doc_id}:sec-2-tbl1"]
    assert out.reused_stage1 == 2 and out.table_chunks == 1 and out.validation.passed
    assert [c.order for c in out.chunks] == [0, 1, 2]
    assert out.chunks[0].next_chunk == ids[1] and out.chunks[1].previous_chunk == ids[0]
    assert out.chunks[0].section_path == ["Intro"] and out.chunks[2].kind == "table"
    assert "Component: Seal; Interval (h): 500" in out.chunks[2].text
    # Stable ids: re-chunking the same document yields the same ids.
    assert [c.chunk_id for c in SemanticChunkingAgent(cctx).run(d).unwrap().chunks] == ids


def test_chunking_splits_long_and_unchunked_sections_with_overlap(
    cctx: ConstructionContext,
) -> None:
    cctx.settings.chunking.max_tokens = 60
    cctx.settings.chunking.overlap_tokens = 12
    d = _validated(cctx, sections=[("Long", LONG)], chunks_per_section=False)
    out = SemanticChunkingAgent(cctx).run(d).unwrap()
    assert len(out.chunks) > 3 and out.resplit == len(out.chunks) and out.reused_stage1 == 0
    assert out.chunks[0].context == ""
    for prev, cur in zip(out.chunks, out.chunks[1:], strict=False):
        assert cur.context and cur.context in prev.text  # overlap is the previous tail
    # An oversized Stage 1 chunk is re-split too.
    d2 = _validated(cctx, sections=[("Long", LONG)])
    out2 = SemanticChunkingAgent(cctx).run(d2).unwrap()
    assert out2.resplit > 1 and all(c.source_chunk_id == "sec-1-c0" for c in out2.chunks)


def test_chunking_tables_can_be_disabled_and_tiny_chunks_dropped(cctx: ConstructionContext) -> None:
    cctx.settings.chunking.include_tables = False
    cctx.settings.chunking.min_tokens = 4
    d = _validated(
        cctx,
        sections=[("A", "Pump uses oil daily here."), ("B", "Ok.")],
        tables={0: [("T", ["a", "b"], [["1", "2"]])]},
    )
    out = SemanticChunkingAgent(cctx).run(d).unwrap()
    assert [c.section_id for c in out.chunks] == ["sec-1"]


def test_chunk_helpers() -> None:
    assert window_sentences("A b. C d. E f.", 6) == ["A b. C d.", "E f."]
    assert tail_tokens("One two. Three four.", 3) == "Three four."
    assert tail_tokens("One two.", 0) == ""
    table = TableJSON(
        table_id="t",
        number=None,
        caption=None,
        page=1,
        bbox=BBoxJSON(x=0, y=0, w=1, h=1),
        headers=["A", ""],
        rows=[["x", "y"], ["", ""]],
        units={},
        key_attributes=[],
        description="",
        summary="",
        keywords=[],
        approved=True,
    )
    assert linearise_table(table) == "Table t\nA: x; Column 2: y"
