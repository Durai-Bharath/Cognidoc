"""Shared text patterns for headings, captions, equations, lists and references."""

from __future__ import annotations

import re

NUMBERED_HEADING = re.compile(
    r"^(?P<num>\d+(?:\.\d+){0,3}|[A-Z](?:\.\d+){1,2}|(?:[IVXLC]+|[A-Z])(?=\.))\.?\s+"
    r"(?P<title>[A-Z][^\n]{0,120})$"
)
APPENDIX_HEADING = re.compile(r"^Appendix\s+(?P<num>[A-Z0-9]+)\b[.:]?\s*(?P<title>.*)$", re.I)
NAMED_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "summary",
    "scope",
    "purpose",
    "overview",
    "procedure",
    "responsibilities",
    "definitions",
    "requirements",
    "installation",
    "troubleshooting",
    "maintenance",
    "safety",
    "glossary",
}
REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "literature"}

CAPTION = re.compile(
    r"^(?P<kind>Figure|Fig\.|Table|Algorithm|Listing|Equation|Eq\.)\s*(?P<num>[A-Z]?\d+(?:\.\d+)*[a-z]?)\s*[:.\-–—]?\s*(?P<text>.*)",
    re.I | re.S,
)
LIST_ITEM = re.compile(r"^(?:[•●▪‣\-\*•·]|\(?\d{1,2}[\).]|\(?[a-z][\).])\s+\S")
REFERENCE_ENTRY = re.compile(r"^\[\d+\]\s+\S|^\d+\.\s+[A-Z][a-z]+,\s+[A-Z]")
EQUATION_NUMBER = re.compile(r"\((?P<num>\d+(?:\.\d+)?)\)\s*$")
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.I)
_MATH_CHARS = set("=+−-*/^_∑∫∏√∞≤≥≠≈∈∉⊂⊆∪∩∀∃∂∇×·±→←↔λμσπθαβγδεωΩΣΔΦΨ|<>{}[]()")


def math_density(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c in _MATH_CHARS for c in chars) / len(chars)


def looks_like_equation(text: str) -> bool:
    flat = " ".join(text.split())
    if not flat or len(flat) > 300:
        return False
    words = re.findall(r"[A-Za-z]{4,}", flat)
    has_relation = any(op in flat for op in ("=", "≤", "≥", "≈", "∝", "→"))
    density = math_density(flat)
    if EQUATION_NUMBER.search(flat) and has_relation and len(words) <= 8:
        return True
    return has_relation and density >= 0.18 and len(words) <= 4


def heading_match(text: str) -> tuple[str | None, str] | None:
    """Return ``(number, title)`` when a single line looks like a heading."""
    line = " ".join(text.split())
    if not line or len(line) > 140 or "\n" in text.strip():
        return None
    if line.endswith((".", ",", ";")) and not re.match(r"^\d+(\.\d+)*\.\s", line):
        return None
    m = APPENDIX_HEADING.match(line)
    if m:
        return (m.group("num"), line)
    m = NUMBERED_HEADING.match(line)
    if m and len(m.group("title").split()) <= 14 and not m.group("title").endswith("."):
        num = m.group("num")
        roman_or_letter = re.fullmatch(r"[IVXLC]+|[A-Z]", num)
        if roman_or_letter and len(m.group("title").split()) > 8:
            return None
        return (num, m.group("title").strip())
    if line.lower().rstrip(":") in NAMED_SECTIONS:
        return (None, line.rstrip(":"))
    return None


def heading_depth(number: str | None) -> int:
    if not number:
        return 1
    return number.count(".") + 1


def caption_match(text: str) -> tuple[str, str] | None:
    """Return ``(kind, number)`` for 'Figure 3: ...', 'Table 2.1 ...' etc."""
    m = CAPTION.match(text.strip())
    if not m:
        return None
    kind = m.group("kind").lower().rstrip(".")
    kind = {"fig": "figure", "eq": "equation"}.get(kind, kind)
    return kind, m.group("num")


def dehyphenate_lines(text: str) -> str:
    """Join PDF lines into flowing text, removing end-of-line hyphenation."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out += " " + ln
    return out
