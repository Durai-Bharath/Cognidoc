"""Algorithm 2 - Router Agent."""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

from common.base_agent import AgentFailure, BaseAgent
from common.text import detect_language, normalize_text
from extraction.parsers import (
    DocumentInaccessible,
    ParseResult,
    load_image_pages,
    parse_docx,
    parse_pdf,
)
from extraction.schemas import (
    DocumentProperties,
    DocumentType,
    ParsedDocument,
    RouterInput,
    RouterOutput,
)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp"}
_MIME_BY_EXT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "tif": "image/tiff",
    "bmp": "image/bmp",
}


def sniff_mime(path: Path) -> str | None:
    """Content-based MIME detection from magic bytes (no libmagic dependency)."""
    with path.open("rb") as fh:
        head = fh.read(16)
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                if "word/document.xml" in zf.namelist():
                    return _MIME_BY_EXT["docx"]
        except zipfile.BadZipFile:
            return None
        return "application/zip"
    return None


class RouterAgent(BaseAgent[RouterInput, RouterOutput]):
    name = "RouterAgent"
    algorithm = 2

    def _execute(self, data: RouterInput) -> RouterOutput:
        path = Path(data.path)
        cfg = self.ctx.config.router

        # Lines 2-6: validate document accessibility.
        problem = self._accessibility_problem(path, cfg.max_file_size_mb)
        if problem:
            raise AgentFailure(problem, error_type="DocumentInaccessible")

        # Lines 7-21: extension + MIME routing.
        ext = path.suffix.lower().lstrip(".")
        sniffed = sniff_mime(path)
        if ext == "pdf":
            doc_type = DocumentType.PDF
        elif ext == "docx":
            doc_type = DocumentType.DOCX
        elif ext in IMAGE_EXTENSIONS:
            doc_type = DocumentType.IMAGE
        else:
            raise AgentFailure(
                f"unsupported document type '.{ext}'", error_type="UnsupportedDocument"
            )
        expected = _MIME_BY_EXT[ext]
        if sniffed != expected and not (
            doc_type == DocumentType.IMAGE and sniffed and sniffed.startswith("image/")
        ):
            raise AgentFailure(
                f"extension '.{ext}' does not match content type {sniffed or 'unknown'}",
                error_type="UnsupportedDocument",
            )

        workdir = Path(data.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            parsed = self._parse(doc_type, path, workdir)
        except DocumentInaccessible as exc:
            raise AgentFailure(str(exc), error_type="DocumentInaccessible") from exc

        # Lines 22-24: raw content, UTF-8 normalisation, unsupported objects.
        for page in parsed.pages:
            for block in page.blocks:
                block.text = normalize_text(block.text)
                if block.table_cells:
                    block.table_cells = [
                        [normalize_text(c) if c is not None else None for c in row]
                        for row in block.table_cells
                    ]
        raw_text = "\n\n".join(b.text for p in parsed.pages for b in p.blocks if b.text)

        # Lines 25-31: document properties.
        title = parsed.title or self._title_from_blocks(parsed) or path.stem
        properties = DocumentProperties(
            title=normalize_text(title),
            author=normalize_text(parsed.author) if parsed.author else None,
            creation_date=parsed.creation_date,
            page_count=len(parsed.pages),
            language=detect_language(raw_text),
            file_size_bytes=path.stat().st_size,
            mime_type=sniffed or expected,
            extension=ext,
        )
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.ctx.document_id = document_id
        if parsed.removed_objects:
            self.log.info("removed_unsupported_objects", extra={"objects": parsed.removed_objects})

        # Lines 32-34: P and T.
        stream = ParsedDocument(
            document_id=document_id,
            source_path=str(path.resolve()),
            doc_type=doc_type,
            properties=properties,
            pages=parsed.pages,
            raw_text=raw_text,
            removed_objects=parsed.removed_objects,
            ocr_pages=parsed.ocr_pages,
            workdir=str(workdir),
        )
        return RouterOutput(parsed=stream, doc_type=doc_type)

    def confidence(self, output: RouterOutput) -> float | None:
        return 1.0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _accessibility_problem(path: Path, max_mb: float) -> str | None:
        if not path.exists():
            return f"document not found: {path}"
        if not path.is_file():
            return f"not a regular file: {path}"
        if not os.access(path, os.R_OK):
            return f"document is not readable: {path}"
        size = path.stat().st_size
        if size == 0:
            return "document is empty"
        if size > max_mb * 1024 * 1024:
            return f"document exceeds {max_mb} MB"
        return None

    def _parse(self, doc_type: DocumentType, path: Path, workdir: Path) -> ParseResult:
        cfg = self.ctx.config
        if doc_type == DocumentType.PDF:
            return parse_pdf(
                path,
                workdir,
                min_native_chars=cfg.router.min_native_text_chars,
                ocr_dpi=cfg.ocr.render_dpi,
            )
        if doc_type == DocumentType.DOCX:
            return parse_docx(path, workdir)
        return load_image_pages(path, workdir)

    @staticmethod
    def _title_from_blocks(parsed: ParseResult) -> str | None:
        if not parsed.pages:
            return None
        blocks = [b for b in parsed.pages[0].blocks if b.kind == "text" and b.text]
        styled = [b for b in blocks if (b.style or "").lower() == "title"]
        if styled:
            return styled[0].text
        sized = [b for b in blocks if b.font_size]
        if not sized:
            return None
        best = max(sized, key=lambda b: (b.font_size or 0, -b.bbox.y))
        return best.text.split("\n")[0][:200]
