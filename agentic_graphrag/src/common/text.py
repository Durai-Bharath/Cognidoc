"""Text utilities: UTF-8 normalisation, tokenisation, sentences, keywords, language."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{1,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\sa-z])")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his
    how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves using used use may must shall within
    without per via e.g i.e etc""".split()
)

_LANG_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and of to in is that for with as on are this be by".split()),
    "de": frozenset("der die und das ist nicht mit den von zu ein eine auf für".split()),
    "fr": frozenset("le la les et des est une pour dans que qui sur pas du".split()),
    "es": frozenset("el la los las y de que en es por para con una un del".split()),
    "it": frozenset("il la di che e per con non una sono del della gli".split()),
    "pt": frozenset("o a os as e de que em para com uma um do da não".split()),
    "nl": frozenset("de het een en van is dat op te in voor met niet".split()),
}


def normalize_text(text: str) -> str:
    """Normalise to clean UTF-8: NFKC (expands ligatures), drop control chars and lone
    surrogates, unify line endings and collapse runs of spaces."""
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Approximate LLM token count (word pieces + punctuation), deterministic and offline."""
    count = 0
    for tok in _TOKEN.findall(text):
        count += max(1, (len(tok) + 3) // 4) if len(tok) > 8 else 1
    return count


def split_sentences(text: str) -> list[str]:
    """Sentences, never crossing a blank-line (paragraph/heading) boundary."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        out.extend(s.strip() for s in _SENT_SPLIT.split(block) if s.strip())
    return out


def content_words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def keywords(text: str, k: int = 10) -> list[str]:
    """Top-k content words by frequency (ties broken by first occurrence)."""
    words = content_words(text)
    counts = Counter(words)
    first: dict[str, int] = {}
    for i, w in enumerate(words):
        first.setdefault(w, i)
    ranked = sorted(counts, key=lambda w: (-counts[w], first[w]))
    return [w for w in ranked if len(w) > 2][:k]


def detect_language(text: str) -> str:
    """Stopword-overlap language guess; returns an ISO-639-1 code or ``"und"``."""
    words = [w.lower() for w in re.findall(r"\w+", text[:20000])]
    if len(words) < 5:
        return "und"
    scores = {lang: sum(w in sw for w in words) for lang, sw in _LANG_STOPWORDS.items()}
    lang, best = max(scores.items(), key=lambda kv: kv[1])
    return lang if best >= max(2, len(words) // 50) else "und"


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?%?")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4})\b"
)


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", ".").rstrip("%") for n in NUMBER_RE.findall(text)}
