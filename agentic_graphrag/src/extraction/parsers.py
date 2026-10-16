"""Format-specific parsers invoked by the Router Agent (PDF, DOCX, image)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image, ImageSequence

from common.text import normalize_text
from extraction.schemas import BBox, PageData, RawBlock


@dataclass
class ParseResult:
    pages: list[PageData]
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    removed_objects: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)


class DocumentInaccessible(Exception):
    """The document exists but cannot be opened/read (encrypted, corrupt, ...)."""


# =========================================================================== PDF

_UNSUPPORTED_ANNOTS = {"FileAttachment", "Sound", "Movie", "RichMedia", "Screen", "3D", "Widget"}


def _pdf_date(value: str | None) -> str | None:
    if not value:
        return None
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", value)
    if not m:
        return value
    parts = [int(p) if p else d for p, d in zip(m.groups(), (0, 1, 1, 0, 0, 0), strict=True)]
    try:
        return datetime(*parts).isoformat()  # type: ignore[arg-type]
    except ValueError:
        return value


def _is_bold(span: dict[str, Any]) -> bool:
    return bool(span.get("flags", 0) & 16) or "bold" in str(span.get("font", "")).lower()


def _inside_any(bbox: BBox, regions: list[BBox], frac: float = 0.6) -> bool:
    return any(bbox.area > 0 and bbox.intersection(r) / bbox.area >= frac for r in regions)


def _render_clip(page: pymupdf.Page, rect: BBox, path: Path, dpi: int = 150) -> str:
    clip = pymupdf.Rect(rect.x, rect.y, rect.x1, rect.y1)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(path))
    return str(path)


def _cluster_drawings(rects: list[BBox], gap: float) -> list[BBox]:
    clusters: list[BBox] = []
    for r in sorted(rects, key=lambda b: (b.y, b.x)):
        grown = BBox(x=r.x - gap, y=r.y - gap, w=r.w + 2 * gap, h=r.h + 2 * gap)
        for i, c in enumerate(clusters):
            if grown.intersection(c) > 0:
                clusters[i] = c.union(r)
                break
        else:
            clusters.append(r)
    # merge transitively overlapping clusters
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i].intersection(clusters[j]) > 0:
                    clusters[i] = clusters[i].union(clusters.pop(j))
                    merged = True
                    break
            if merged:
                break
    return clusters


def parse_pdf(path: Path, workdir: Path, *, min_native_chars: int, ocr_dpi: int) -> ParseResult:
    try:
        doc: Any = pymupdf.open(str(path))
    except Exception as exc:
        raise DocumentInaccessible(f"cannot open PDF: {exc}") from exc
    if doc.needs_pass:
        raise DocumentInaccessible("PDF is password protected")
    meta = doc.metadata or {}
    result = ParseResult(
        pages=[],
        title=(meta.get("title") or "").strip() or None,
        author=(meta.get("author") or "").strip() or None,
        creation_date=_pdf_date(meta.get("creationDate")),
    )
    for i in range(doc.embfile_count()):
        result.removed_objects.append(f"embedded file: {doc.embfile_names()[i]}")

    for page in doc:
        pno = page.number + 1
        width, height = page.rect.width, page.rect.height
        for annot in page.annots() or []:
            kind = annot.type[1]
            if kind in _UNSUPPORTED_ANNOTS:
                result.removed_objects.append(f"page {pno}: {kind} annotation")
        for widget in page.widgets() or []:
            result.removed_objects.append(f"page {pno}: form field {widget.field_name!r}")

        blocks: list[RawBlock] = []
        native_text = page.get_text("text")
        has_text = len(native_text.strip()) >= min_native_chars
        page_data = PageData(
            page_number=pno,
            width=width,
            height=height,
            rotation=page.rotation,
            has_text_layer=has_text,
        )
        if not has_text:
            img_path = workdir / "pages" / f"page_{pno:03d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(dpi=ocr_dpi).save(str(img_path))
            page_data.image_path = str(img_path)
            # OCR coordinates are pixels; express the page in pixels too.
            scale = ocr_dpi / 72.0
            page_data.width, page_data.height = width * scale, height * scale
            result.ocr_pages.append(pno)
            result.pages.append(page_data)
            continue

        # Tables first so their text is not duplicated as paragraphs.
        table_boxes: list[BBox] = []
        try:
            found = page.find_tables()
            tables = list(found.tables)
        except Exception:
            tables = []
        for t_idx, table in enumerate(tables):
            bbox = BBox.from_corners(*table.bbox)
            cells = [
                [normalize_text(c) if c is not None else None for c in row]
                for row in table.extract()
            ]
            if not cells or not cells[0]:
                continue
            table_boxes.append(bbox)
            blocks.append(
                RawBlock(
                    block_id=f"p{pno}-t{t_idx}",
                    page=pno,
                    bbox=bbox,
                    kind="table",
                    table_cells=cells,
                    text="\n".join(" | ".join(c or "" for c in r) for r in cells),
                )
            )

        data = page.get_text("dict")
        image_boxes: list[BBox] = []
        b_idx = 0
        for block in data.get("blocks", []):
            bbox = BBox.from_corners(*block["bbox"])
            if block.get("type") == 1:
                if bbox.area < 0.005 * width * height or _inside_any(bbox, table_boxes):
                    continue
                img_path = workdir / "figures" / f"p{pno}_img{b_idx}.png"
                _render_clip(page, bbox, img_path)
                image_boxes.append(bbox)
                blocks.append(
                    RawBlock(
                        block_id=f"p{pno}-b{b_idx}",
                        page=pno,
                        bbox=bbox,
                        kind="image",
                        image_path=str(img_path),
                    )
                )
                b_idx += 1
                continue
            # Split text blocks where the font changes (heading glued to a paragraph).
            current: list[dict[str, Any]] = []
            groups: list[list[dict[str, Any]]] = []
            prev_key: tuple[float, bool] | None = None
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                size = max(s["size"] for s in spans)
                bold = all(_is_bold(s) for s in spans)
                key = (round(size, 1), bold)
                if prev_key is not None and (
                    abs(key[0] - prev_key[0]) > 0.6 or key[1] != prev_key[1]
                ):
                    groups.append(current)
                    current = []
                current.append(
                    {
                        "line": line,
                        "size": size,
                        "bold": bold,
                        "text": "".join(s["text"] for s in spans),
                    }
                )
                prev_key = key
            if current:
                groups.append(current)
            for group in groups:
                gbox = BBox.from_corners(*group[0]["line"]["bbox"])
                for item in group[1:]:
                    gbox = gbox.union(BBox.from_corners(*item["line"]["bbox"]))
                if _inside_any(gbox, table_boxes) or _inside_any(gbox, image_boxes, 0.8):
                    continue
                text = normalize_text("\n".join(item["text"] for item in group))
                if not text:
                    continue
                sizes = [item["size"] for item in group]
                blocks.append(
                    RawBlock(
                        block_id=f"p{pno}-b{b_idx}",
                        page=pno,
                        bbox=gbox,
                        kind="text",
                        text=text,
                        font_size=round(sorted(sizes)[len(sizes) // 2], 2),
                        bold=all(item["bold"] for item in group),
                    )
                )
                b_idx += 1

        # Vector graphics (diagrams drawn with paths) become figure regions.
        rects: list[BBox] = []
        for drawing in page.get_drawings():
            r = drawing.get("rect")
            if r is None:
                continue
            box = BBox.from_corners(r.x0, r.y0, r.x1, r.y1)
            if box.w * box.h > 0.9 * width * height or _inside_any(box, table_boxes, 0.5):
                continue
            rects.append(box)
        text_boxes = [b.bbox for b in blocks if b.kind == "text"]
        for c_idx, cluster in enumerate(_cluster_drawings(rects, gap=8.0)):
            if cluster.area < 0.02 * width * height or cluster.h < 30 or cluster.w < 30:
                continue
            if _inside_any(cluster, image_boxes, 0.5):
                continue
            covered = sum(cluster.intersection(t) for t in text_boxes)
            if covered > 0.85 * cluster.area:
                continue  # boxes drawn around text, not a figure
            img_path = workdir / "figures" / f"p{pno}_vec{c_idx}.png"
            _render_clip(page, cluster, img_path)
            blocks = [
                b
                for b in blocks
                if not (
                    b.kind == "text" and _inside_any(b.bbox, [cluster], 0.9) and len(b.text) < 60
                )
            ]
            blocks.append(
                RawBlock(
                    block_id=f"p{pno}-v{c_idx}",
                    page=pno,
                    bbox=cluster,
                    kind="image",
                    image_path=str(img_path),
                    text="",
                )
            )
        page_data.blocks = blocks
        result.pages.append(page_data)
    doc.close()
    return result


# =========================================================================== DOCX

_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
}
_W = "{{{}}}".format(_NS["w"])

DOCX_PAGE_W, DOCX_PAGE_H, DOCX_MARGIN, DOCX_LINE_H = 612.0, 792.0, 72.0, 14.0


def parse_docx(path: Path, workdir: Path) -> ParseResult:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise DocumentInaccessible(f"cannot open DOCX: {exc}") from exc
    props = document.core_properties
    result = ParseResult(
        pages=[],
        title=(props.title or "").strip() or None,
        author=(props.author or "").strip() or None,
        creation_date=props.created.isoformat() if props.created else None,
    )
    page_no, y = 1, DOCX_MARGIN
    content_w = DOCX_PAGE_W - 2 * DOCX_MARGIN
    pages: dict[int, list[RawBlock]] = {1: []}
    counter = 0

    def place(lines: int) -> BBox:
        nonlocal page_no, y
        h = max(1, lines) * DOCX_LINE_H
        if y + h > DOCX_PAGE_H - DOCX_MARGIN and y > DOCX_MARGIN:
            page_no += 1
            pages.setdefault(page_no, [])
            y = DOCX_MARGIN
        box = BBox(x=DOCX_MARGIN, y=y, w=content_w, h=h)
        y += h + 6
        return box

    def new_page() -> None:
        nonlocal page_no, y
        page_no += 1
        pages.setdefault(page_no, [])
        y = DOCX_MARGIN

    def add(block: RawBlock) -> None:
        pages.setdefault(block.page, []).append(block)

    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, document)
            breaks_before = (
                any(br.get(f"{_W}type") == "page" for br in child.iter(f"{_W}br"))
                and not para.text.strip()
            )
            if breaks_before:
                new_page()
                continue
            for _obj in child.iter(f"{{{_NS['w']}}}object"):
                result.removed_objects.append(f"OLE object near paragraph {counter}")
            # images
            for blip in child.iter(f"{{{_NS['a']}}}blip"):
                rid = blip.get(f"{{{_NS['r']}}}embed")
                part = document.part.related_parts.get(rid) if rid else None
                if part is None:
                    continue
                img_path = workdir / "figures" / f"docx_img{counter}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(part.blob)
                box = place(12)
                add(
                    RawBlock(
                        block_id=f"d{counter}",
                        page=page_no,
                        bbox=box,
                        kind="image",
                        image_path=str(img_path),
                        source="docx",
                    )
                )
                counter += 1
            # equations (OMML)
            maths = list(child.iter(f"{{{_NS['m']}}}oMath"))
            for math_el in maths:
                eq_text = "".join(t.text or "" for t in math_el.iter(f"{{{_NS['m']}}}t"))
                if eq_text.strip():
                    add(
                        RawBlock(
                            block_id=f"d{counter}",
                            page=page_no,
                            bbox=place(2),
                            text=normalize_text(eq_text),
                            style="Equation",
                            source="docx",
                        )
                    )
                    counter += 1
            text = normalize_text(para.text)
            if not text:
                continue
            style = para.style.name if para.style is not None else None
            sizes = [r.font.size.pt for r in para.runs if r.font.size is not None]
            if not sizes and para.style is not None and para.style.font.size is not None:
                sizes = [para.style.font.size.pt]
            runs = [r for r in para.runs if r.text.strip()]
            bold = bool(runs) and all(bool(r.bold) for r in runs)
            lines = max(1, math.ceil(len(text) / 95))
            add(
                RawBlock(
                    block_id=f"d{counter}",
                    page=page_no,
                    bbox=place(lines),
                    text=text,
                    style=style,
                    font_size=max(sizes) if sizes else None,
                    bold=bold,
                    source="docx",
                )
            )
            counter += 1
            if any(br.get(f"{_W}type") == "page" for br in child.iter(f"{_W}br")):
                new_page()
        elif tag == "tbl":
            table = Table(child, document)
            seen: set[int] = set()
            cells: list[list[str | None]] = []
            for row in table.rows:
                row_cells: list[str | None] = []
                for cell in row.cells:
                    key = id(cell._tc)
                    row_cells.append(None if key in seen else normalize_text(cell.text))
                    seen.add(key)
                cells.append(row_cells)
            if not cells:
                continue
            add(
                RawBlock(
                    block_id=f"d{counter}",
                    page=page_no,
                    bbox=place(len(cells) + 1),
                    kind="table",
                    table_cells=cells,
                    source="docx",
                    text="\n".join(" | ".join(c or "" for c in r) for r in cells),
                )
            )
            counter += 1
    result.pages = [
        PageData(page_number=n, width=DOCX_PAGE_W, height=DOCX_PAGE_H, blocks=blocks)
        for n, blocks in sorted(pages.items())
    ]
    return result


# =========================================================================== images


def load_image_pages(path: Path, workdir: Path) -> ParseResult:
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise DocumentInaccessible(f"cannot open image: {exc}") from exc
    result = ParseResult(pages=[], title=path.stem)
    exif = img.getexif()
    if exif.get(306):
        result.creation_date = str(exif.get(306))
    if exif.get(315):
        result.author = str(exif.get(315))
    out_dir = workdir / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(ImageSequence.Iterator(img), start=1):
        rgb = frame.convert("RGB")
        page_path = out_dir / f"page_{idx:03d}.png"
        rgb.save(page_path)
        result.pages.append(
            PageData(
                page_number=idx,
                width=float(rgb.width),
                height=float(rgb.height),
                has_text_layer=False,
                image_path=str(page_path),
            )
        )
        result.ocr_pages.append(idx)
    return result
