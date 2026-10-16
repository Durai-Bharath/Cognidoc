"""Ontology O: entity types, predicates, synonyms and domain/range constraints.

Loaded from ``ontology.yaml`` and validated when the pipeline starts; an invalid
ontology raises :class:`OntologyError` before any document is touched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from construction.models import STRUCTURAL_PREDICATES

ANY = "Any"
FALLBACK_TYPE = "Concept"
_PRED_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_AUX = ("has been ", "have been ", "is being ", "is ", "are ", "was ", "were ", "be ", "been ")
_NON_WORD = re.compile(r"[^a-z0-9]+")


class OntologyError(ValueError):
    pass


def relation_forms(raw: str) -> list[str]:
    """Normalised lookup keys for a relation phrase: as-is, then without auxiliaries."""
    text = _NON_WORD.sub(" ", raw.lower().replace("_", " ")).strip()
    forms = [text]
    stripped = text
    changed = True
    while changed:
        changed = False
        for aux in _AUX:
            if stripped.startswith(aux) and len(stripped) > len(aux):
                stripped = stripped[len(aux) :]
                changed = True
    if stripped != text:
        forms.append(stripped)
    return [f for f in forms if f]


def _stem_word(word: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def stem_phrase(phrase: str) -> str:
    return " ".join(_stem_word(w) for w in phrase.split())


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityTypeDef(_Strict):
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)


class PredicateDef(_Strict):
    description: str = ""
    domain: list[str] = Field(default_factory=lambda: [ANY])
    range: list[str] = Field(default_factory=lambda: [ANY])
    synonyms: list[str] = Field(default_factory=list)
    inverse_synonyms: list[str] = Field(default_factory=list)
    functional: bool = False


class PredicateMatch(BaseModel):
    predicate: str
    inverted: bool = False
    method: str = "exact"


class Ontology(_Strict):
    version: int = 1
    entity_types: dict[str, EntityTypeDef]
    predicates: dict[str, PredicateDef]
    exclusive: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        errors: list[str] = []
        if FALLBACK_TYPE not in self.entity_types:
            errors.append(f"entity type {FALLBACK_TYPE!r} (the fallback) must be defined")
        types = set(self.entity_types)
        seen_type_syn: dict[str, str] = {}
        for tname, tdef in self.entity_types.items():
            for syn in [tname, *tdef.synonyms]:
                key = syn.lower().strip()
                if seen_type_syn.get(key, tname) != tname:
                    errors.append(
                        f"type synonym {syn!r} maps to both {seen_type_syn[key]} and {tname}"
                    )
                seen_type_syn[key] = tname
        seen_forms: dict[str, str] = {}
        for pname, pdef in self.predicates.items():
            if not _PRED_NAME.match(pname):
                errors.append(f"predicate {pname!r} must be snake_case")
            if pname.upper() in STRUCTURAL_PREDICATES:
                errors.append(f"predicate {pname!r} collides with a structural predicate")
            for side, values in (("domain", pdef.domain), ("range", pdef.range)):
                if not values:
                    errors.append(f"{pname}.{side} must not be empty")
                for t in values:
                    if t != ANY and t not in types:
                        errors.append(f"{pname}.{side}: unknown entity type {t!r}")
            forms = [(f, "") for f in [pname, *pdef.synonyms]] + [
                (f, "~") for f in pdef.inverse_synonyms
            ]
            for phrase, marker in forms:
                for form in relation_forms(phrase):
                    owner = f"{marker}{pname}"
                    if seen_forms.get(form, owner) != owner:
                        errors.append(
                            f"relation form {form!r} maps to both {seen_forms[form]} and {owner}"
                        )
                    seen_forms.setdefault(form, owner)
        for a, b in self.exclusive:
            for p in (a, b):
                if p not in self.predicates:
                    errors.append(f"exclusive pair references unknown predicate {p!r}")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    # ------------------------------------------------------------------ types
    def canonical_type(self, raw: str | None) -> str:
        if not raw:
            return FALLBACK_TYPE
        key = raw.lower().strip()
        for tname, tdef in self.entity_types.items():
            if key == tname.lower() or key in (s.lower() for s in tdef.synonyms):
                return tname
        return FALLBACK_TYPE

    # ------------------------------------------------------------------ predicates
    def _index(self) -> tuple[dict[str, PredicateMatch], dict[str, PredicateMatch]]:
        cached = self.__dict__.get("_index_cache")
        if cached is not None:
            return cached  # type: ignore[no-any-return]
        exact: dict[str, PredicateMatch] = {}
        stemmed: dict[str, PredicateMatch] = {}
        for pname, pdef in self.predicates.items():
            entries = [(pname, False, "exact")]
            entries += [(s, False, "synonym") for s in pdef.synonyms]
            entries += [(s, True, "synonym") for s in pdef.inverse_synonyms]
            for phrase, inverted, method in entries:
                for form in relation_forms(phrase):
                    exact.setdefault(
                        form, PredicateMatch(predicate=pname, inverted=inverted, method=method)
                    )
                    stemmed.setdefault(
                        stem_phrase(form),
                        PredicateMatch(predicate=pname, inverted=inverted, method="stem"),
                    )
        self.__dict__["_index_cache"] = (exact, stemmed)
        return exact, stemmed

    def match_relation(self, raw: str) -> PredicateMatch | None:
        """Exact name / synonym match first, then a stemmed match; ``None`` if nothing fits."""
        exact, stemmed = self._index()
        forms = relation_forms(raw)
        for form in forms:
            if form in exact:
                return exact[form]
        for form in forms:
            hit = stemmed.get(stem_phrase(form))
            if hit is not None:
                return hit
        return None

    def is_functional(self, predicate: str) -> bool:
        pdef = self.predicates.get(predicate)
        return bool(pdef and pdef.functional)

    def exclusive_with(self, predicate: str) -> set[str]:
        out: set[str] = set()
        for a, b in self.exclusive:
            if a == predicate:
                out.add(b)
            elif b == predicate:
                out.add(a)
        return out

    def type_allowed(self, allowed: list[str], entity_type: str) -> bool:
        return ANY in allowed or entity_type == FALLBACK_TYPE or entity_type in allowed

    def check_domain_range(self, predicate: str, head_type: str, tail_type: str) -> str | None:
        """Return a violation message, or ``None`` when the triple satisfies O."""
        pdef = self.predicates.get(predicate)
        if pdef is None:
            return f"predicate {predicate!r} is not in the ontology"
        if not self.type_allowed(pdef.domain, head_type):
            return f"{predicate}: head type {head_type} not in domain {pdef.domain}"
        if not self.type_allowed(pdef.range, tail_type):
            return f"{predicate}: tail type {tail_type} not in range {pdef.range}"
        return None

    def describe_predicates(self) -> str:
        return "\n".join(f"- {n}: {d.description}" for n, d in self.predicates.items())


def load_ontology(path: str | Path) -> Ontology:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OntologyError(f"cannot read ontology {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise OntologyError(f"{path}: top level must be a mapping")
    try:
        return Ontology.model_validate(data)
    except ValidationError as exc:
        raise OntologyError(f"invalid ontology {path}: {exc}") from exc
