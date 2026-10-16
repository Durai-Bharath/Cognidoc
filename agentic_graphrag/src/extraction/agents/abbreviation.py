"""Algorithm 16 - Abbreviation Resolution Agent (Schwartz-Hearst pair detection)."""

from __future__ import annotations

import re

from common.base_agent import BaseAgent
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.schemas import Abbreviation, AbbreviationInput, AbbreviationRepository

_PAREN = re.compile(r"\(([^()]{1,100})\)")
_ABBR_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9.\-&/]*[A-Z0-9][A-Za-z0-9.\-&/]*s?$")
_ALLCAPS_USE = re.compile(r"\b[A-Z][A-Z0-9&]{1,7}s?\b")
_IGNORE = {
    "I",
    "II",
    "III",
    "IV",
    "VI",
    "VII",
    "VIII",
    "IX",
    "XI",
    "XII",
    "OK",
    "AM",
    "PM",
    "USA",
    "UK",
    "EU",
    "ID",
    "PDF",
    "JSON",
    "DOCX",
    "PNG",
    "JPEG",
    "TIFF",
    "BMP",
    "URL",
}


def is_abbreviation(token: str, max_len: int) -> bool:
    token = token.strip()
    if not 2 <= len(token.replace(".", "")) <= max_len or " " in token:
        return False
    if not _ABBR_TOKEN.match(token):
        return False
    return sum(c.isupper() for c in token) >= max(2, len(token.replace(".", "")) // 2)


def normalize_abbr(abbr: str) -> str:
    return abbr.replace(".", "").strip()


def best_long_form(short: str, long: str) -> str | None:
    """Schwartz & Hearst (2003): shortest suffix of ``long`` whose letters cover ``short``."""
    s = [c.lower() for c in short if c.isalnum()]
    lf = long.strip()
    if not s:
        return None
    s_idx, l_idx = len(s) - 1, len(lf) - 1
    while s_idx >= 0:
        c = s[s_idx]
        while l_idx >= 0 and (
            lf[l_idx].lower() != c or (s_idx == 0 and l_idx > 0 and lf[l_idx - 1].isalnum())
        ):
            l_idx -= 1
        if l_idx < 0:
            return None
        l_idx -= 1
        s_idx -= 1
    start = lf.rfind(" ", 0, l_idx + 1) + 1
    candidate = lf[start:].strip(" ,;:")
    words = candidate.split()
    if not words or len(words) > min(len(s) + 5, len(s) * 2) or candidate.lower() == short.lower():
        return None
    return candidate


def find_pairs(text: str, max_len: int) -> list[tuple[str, str]]:
    """'Full Term (ABBR)' and 'ABBR (Full Term)' definitions in ``text``."""
    pairs: list[tuple[str, str]] = []
    for m in _PAREN.finditer(text):
        inside = m.group(1).strip()
        before = text[: m.start()].rstrip()
        if is_abbreviation(inside, max_len):
            window_words = before.split()[-(len(inside) * 2 + 2) :]
            # stop at sentence/clause boundaries
            window = " ".join(window_words)
            window = re.split(r"[.;:!?]\s", window)[-1]
            long = best_long_form(inside, window)
            if long:
                pairs.append((normalize_abbr(inside), long))
            continue
        prev = before.split()[-1] if before.split() else ""
        if is_abbreviation(prev, max_len) and len(inside.split()) >= 2:
            long = best_long_form(prev, inside)
            if long and long == inside.strip(" ,;:"):
                pairs.append((normalize_abbr(prev), long))
    return pairs


def _norm_exp(expansion: str) -> str:
    return " ".join(expansion.lower().split())


class AbbreviationResolutionAgent(BaseAgent[AbbreviationInput, AbbreviationRepository]):
    name = "AbbreviationResolutionAgent"
    algorithm = 16

    def _execute(self, data: AbbreviationInput) -> AbbreviationRepository:
        max_len = self.ctx.config.abbreviation.max_abbr_len
        section = data.task.section
        repo = AbbreviationRepository(
            section_id=section.section_id,
            abbreviations={k: v.model_copy(deep=True) for k, v in data.existing.items()},
        )
        conflicts: dict[str, set[str]] = {}
        for el in section.all_elements():
            for abbr, expansion in find_pairs(el.text, max_len):
                existing = repo.abbreviations.get(abbr)
                if existing is None:
                    repo.abbreviations[abbr] = Abbreviation(
                        abbreviation=abbr,
                        expansion=expansion,
                        description=f"{abbr} stands for {expansion}.",
                    )
                    existing = repo.abbreviations[abbr]
                elif _norm_exp(existing.expansion) != _norm_exp(expansion):
                    conflicts.setdefault(abbr, {existing.expansion}).add(expansion)
                if section.section_id not in existing.section_ids:
                    existing.section_ids.append(section.section_id)
                if el.page not in existing.pages:
                    existing.pages.append(el.page)

        text = section.text()
        for abbr, entry in repo.abbreviations.items():
            uses = len(re.findall(rf"(?<![\w-]){re.escape(abbr)}s?(?![\w-])", text))
            if uses and section.section_id not in entry.section_ids:
                entry.section_ids.append(section.section_id)
            entry.occurrences += uses
        repo.conflicts = {k: sorted(v) for k, v in conflicts.items()}
        known = set(repo.abbreviations) | {a + "s" for a in repo.abbreviations}
        repo.undefined = sorted(
            {
                m.group(0)
                for m in _ALLCAPS_USE.finditer(text)
                if m.group(0) not in known and m.group(0) not in _IGNORE and len(m.group(0)) >= 2
            }
        )
        # an abbreviation whose only occurrence is its own definition is unused
        repo.unused = sorted(a for a, e in repo.abbreviations.items() if e.occurrences <= 1)
        repo.validation = self.validate(repo)
        repo.approved = repo.validation.passed
        if not repo.approved:
            self.flag(
                "abbreviation inconsistencies",
                item_ref=section.section_id,
                details={"failures": repo.validation.failure_reasons()},
            )
        return repo

    @staticmethod
    def validate(repo: AbbreviationRepository) -> ValidationOutcome:
        outcome = ValidationOutcome()
        by_expansion: dict[str, list[str]] = {}
        for abbr, entry in repo.abbreviations.items():
            by_expansion.setdefault(_norm_exp(entry.expansion), []).append(abbr)
        outcome.add(
            CheckResult.from_failures(
                "duplicate_abbreviations",
                [
                    f"'{exp}' defined as {abbrs}"
                    for exp, abbrs in by_expansion.items()
                    if len(abbrs) > 1
                ],
                Severity.WARNING,
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "conflicting_definitions", [f"{a}: {defs}" for a, defs in repo.conflicts.items()]
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "undefined_abbreviations",
                [f"{a} used without definition" for a in repo.undefined],
                Severity.WARNING,
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "unused_abbreviations",
                [f"{a} defined but never used" for a in repo.unused],
                Severity.WARNING,
            )
        )
        return outcome
