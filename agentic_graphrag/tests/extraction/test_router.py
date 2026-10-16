"""Algorithm 2 - RouterAgent."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from common.base_agent import AgentContext
from extraction.agents.router import RouterAgent, sniff_mime
from extraction.schemas import DocumentType, RouterInput


def _route(ctx: AgentContext, path: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    return RouterAgent(ctx).run(RouterInput(path=str(path), workdir=str(tmp_path / "work")))


def test_pdf_is_parsed_with_properties(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    out = _route(ctx, samples["pdf"], tmp_path).unwrap()
    p = out.parsed
    assert out.doc_type == DocumentType.PDF and not p.requires_ocr
    assert p.properties.title == "Hydraulic Pump Maintenance Report"
    assert p.properties.author == "Test Author"
    assert p.properties.creation_date == "2024-01-15T12:00:00"
    assert p.properties.page_count == 2 and p.properties.language == "en"
    assert p.properties.file_size_bytes == samples["pdf"].stat().st_size
    assert ctx.document_id == p.document_id and len(p.document_id) == 16
    kinds = {b.kind for page in p.pages for b in page.blocks}
    assert kinds == {"text", "table", "image"}


def test_docx_is_parsed(ctx: AgentContext, samples: dict[str, Path], tmp_path: Path) -> None:
    out = _route(ctx, samples["docx"], tmp_path).unwrap()
    assert out.doc_type == DocumentType.DOCX
    assert out.parsed.properties.author == "Docs Team"
    styles = {b.style for b in out.parsed.pages[0].blocks}
    assert {"Title", "Heading 1", "Heading 2", "Caption"} <= styles
    assert any(b.kind == "table" for b in out.parsed.pages[0].blocks)


def test_image_goes_to_ocr_pipeline(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    out = _route(ctx, samples["scan"], tmp_path).unwrap()
    assert out.doc_type == DocumentType.IMAGE and out.parsed.ocr_pages == [1]
    assert Path(out.parsed.pages[0].image_path or "").exists()


def test_scanned_pdf_pages_require_ocr(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_image(page.rect, filename=str(samples["clean_scan"]))
    path = tmp_path / "scanned.pdf"
    pdf.save(str(path))
    out = _route(ctx, path, tmp_path).unwrap()
    assert out.parsed.ocr_pages == [1] and not out.parsed.pages[0].has_text_layer


def _bad_input(kind: str, d: Path) -> Path:
    files = {
        "empty": ("empty.pdf", b""),
        "unsupported-ext": ("notes.txt", b"hi"),
        "mime-mismatch": ("fake.pdf", b"\x89PNG\r\n\x1a\n0000"),
    }
    if kind == "missing":
        return d / "missing.pdf"
    if kind == "directory":
        return d
    name, data = files[kind]
    (d / name).write_bytes(data)
    return d / name


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        ("missing", "DocumentInaccessible"),
        ("empty", "DocumentInaccessible"),
        ("unsupported-ext", "UnsupportedDocument"),
        ("mime-mismatch", "UnsupportedDocument"),
        ("directory", "DocumentInaccessible"),
    ],
)
def test_rejections(ctx: AgentContext, tmp_path: Path, kind: str, error_type: str) -> None:
    result = _route(ctx, _bad_input(kind, tmp_path), tmp_path)
    assert result.error is not None and result.error.error_type == error_type


def test_encrypted_pdf_is_inaccessible(ctx: AgentContext, tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    result = _route(ctx, path, tmp_path)
    assert result.error is not None and "password" in result.error.message


def test_oversized_file_is_inaccessible(
    ctx: AgentContext, samples: dict[str, Path], tmp_path: Path
) -> None:
    ctx.config.router.max_file_size_mb = 0.0001
    result = _route(ctx, samples["pdf"], tmp_path)
    assert result.error is not None and "exceeds" in result.error.message


def test_unsupported_embedded_objects_are_removed(ctx: AgentContext, tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A page with an attachment and enough text to parse.")
    doc.embfile_add("payload.bin", b"\x00\x01")
    path = tmp_path / "attach.pdf"
    doc.save(str(path))
    out = _route(ctx, path, tmp_path).unwrap()
    assert out.parsed.removed_objects == ["embedded file: payload.bin"]


def test_text_is_normalised_to_utf8(ctx: AgentContext, tmp_path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_paragraph("e\ufb03cient   work \ufb02ow\u00ad is ﬁne")
    path = tmp_path / "lig.docx"
    d.save(str(path))
    text = _route(ctx, path, tmp_path).unwrap().parsed.raw_text
    assert text == "efficient work flow is fine"


def test_sniff_mime(samples: dict[str, Path]) -> None:
    assert sniff_mime(samples["pdf"]) == "application/pdf"
    assert sniff_mime(samples["scan"]) == "image/png"
    assert sniff_mime(samples["docx"]).endswith("wordprocessingml.document")  # type: ignore[union-attr]
