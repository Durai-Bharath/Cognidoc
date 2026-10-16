"""Algorithm 15 - Entity Extraction Agent.

Per DESIGN_DECISIONS.md #1 the repository is forwarded to Metadata Fusion (relationship
extraction belongs to Stage 2).
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from common.base_agent import BaseAgent
from common.llm import LLMError
from common.text import split_sentences
from common.validation import CheckResult, ValidationOutcome
from extraction.schemas import Entity, EntityRepository, EntityType, SectionTask

_STANDARD = re.compile(r"\b(?:ISO|IEC|IEEE|RFC|ASTM|ANSI|DIN|EN|NIST|IETF|W3C) ?[\d.:\-/]+\b")
_VERSIONED = re.compile(r"\b[A-Z][A-Za-z]*(?:[- ]?\d+(?:\.\d+)*)[A-Za-z]?\b")
_CAMEL = re.compile(r"\b[A-Z]?[a-z]+[A-Z][A-Za-z]+\b|\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")
_TITLE_SEQ = re.compile(
    r"\b[A-Z][a-z]+(?:[ \t]+(?:of|for|and|de|van|von)?[ \t]?[A-Z][a-z]+){1,3}\b"
)
_ACRONYM = re.compile(r"\b[A-Z]{2,6}s?\b")
_COMMON_CAPS = {
    "The",
    "This",
    "These",
    "That",
    "In",
    "For",
    "We",
    "It",
    "A",
    "An",
    "Table",
    "Figure",
    "Section",
    "Equation",
    "Algorithm",
    "Appendix",
    "Page",
    "If",
    "When",
}
_TYPE_BY_VALUE = {t.value.lower(): t for t in EntityType} | {t.name.lower(): t for t in EntityType}


class _KV(BaseModel):
    key: str = ""
    value: str = ""


class _EntityItem(BaseModel):
    name: str = ""
    type: str = ""
    description: str = ""
    attributes: list[_KV] = []


class EntityLLM(BaseModel):
    entities: list[_EntityItem] = []


def normalize_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"(?:'s|’s)$", "", name)
    return name.strip(" .,;:()[]\"'")


def ner_candidates(text: str) -> list[str]:
    """Rule-based NER candidates (standards, versioned names, CamelCase, title-case, acronyms)."""
    found: list[str] = []
    for pattern in (_STANDARD, _VERSIONED, _CAMEL, _TITLE_SEQ, _ACRONYM):
        for m in pattern.finditer(text):
            words = normalize_name(m.group(0)).split()
            while words and words[0] in _COMMON_CAPS:  # "The Hydraulic Unit" → "Hydraulic Unit"
                words = words[1:]
            cand = " ".join(words)
            if len(cand) < 2 or cand.isdigit() or (len(words) == 1 and pattern is _TITLE_SEQ):
                continue
            found.append(cand)
    counts = Counter(found)
    return sorted(counts, key=lambda c: (-counts[c], found.index(c)))


def fallback_type(name: str) -> EntityType:
    if _STANDARD.fullmatch(name):
        return EntityType.STANDARD
    if _CAMEL.fullmatch(name) or _VERSIONED.fullmatch(name):
        return EntityType.SOFTWARE
    return EntityType.DOMAIN_TERM


class EntityExtractionAgent(BaseAgent[SectionTask, EntityRepository]):
    name = "EntityExtractionAgent"
    algorithm = 15

    def _execute(self, data: SectionTask) -> EntityRepository:
        cfg = self.ctx.config.entity
        section = data.section
        text = section.text()
        sentences = split_sentences(text)
        candidates = ner_candidates(text)[: cfg.max_candidates]
        repo = EntityRepository(section_id=section.section_id)
        invalid: list[str] = []

        items: list[_EntityItem] = []
        if cfg.use_llm and text.strip():
            try:
                items = self.llm_json(
                    task="entity_extraction",
                    system="You extract named entities from technical documents. Allowed types: "
                    + ", ".join(t.value for t in EntityType)
                    + ". Use only entities literally present in the text.",
                    prompt=(
                        f"NER candidates: {candidates}\n\nSection text:\n{text[:8000]}\n\n"
                        "Return the entities (confirm, correct or add) with type, a one-"
                        "sentence description grounded in the text, and attributes."
                    ),
                    schema=EntityLLM,
                ).entities
            except LLMError as exc:
                self.flag(f"entity LLM classification failed: {exc}", item_ref=section.section_id)
        if not items:
            items = [_EntityItem(name=c, type=fallback_type(c).value) for c in candidates]

        seen: set[str] = set()
        for item in items:
            name = normalize_name(item.name)
            if not name:
                continue
            etype = _TYPE_BY_VALUE.get(item.type.strip().lower())
            if etype is None:
                invalid.append(f"'{name}' has invalid type '{item.type}'")
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            context = next((s for s in sentences if name.casefold() in s.casefold()), "")
            page = next(
                (e.page for e in section.all_elements() if name.casefold() in e.text.casefold()),
                section.page_start,
            )
            repo.entities.append(
                Entity(
                    entity_id=f"{section.section_id}-ent{len(repo.entities) + 1}",
                    type=etype,
                    name=item.name.strip(),
                    normalized_name=name,
                    description=item.description.strip()
                    or (f"{etype.value} mentioned in '{section.title}': {context[:200]}"),
                    attributes={kv.key: kv.value for kv in item.attributes if kv.key},
                    context=context[:400],
                    section_id=section.section_id,
                    page=page,
                )
            )
        repo.validation = self.validate(repo, text, invalid)
        repo.approved = repo.validation.passed
        if not repo.approved:
            self.flag(
                "entities need re-extraction",
                item_ref=section.section_id,
                details={"failures": repo.validation.failure_reasons()},
            )
        return repo

    @staticmethod
    def validate(repo: EntityRepository, text: str, invalid: list[str]) -> ValidationOutcome:
        outcome = ValidationOutcome()
        names = Counter(e.normalized_name.casefold() for e in repo.entities)
        outcome.add(
            CheckResult.from_failures(
                "duplicate_entities", [f"'{n}' x{c}" for n, c in names.items() if c > 1]
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "missing_descriptions",
                [f"{e.entity_id} '{e.name}'" for e in repo.entities if not e.description.strip()],
            )
        )
        outcome.add(CheckResult.from_failures("invalid_classifications", invalid))
        lowered = text.casefold()
        outcome.add(
            CheckResult.from_failures(
                "context_consistency",
                [
                    f"'{e.name}' not found in section text"
                    for e in repo.entities
                    if e.normalized_name.casefold() not in lowered
                ],
            )
        )
        return outcome
