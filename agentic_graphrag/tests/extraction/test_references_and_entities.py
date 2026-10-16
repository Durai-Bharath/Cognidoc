"""Algorithms 13-16 - citations, cross-references, entities, abbreviations."""

from __future__ import annotations

from common.base_agent import AgentContext
from common.llm import LLMError, MockLLM
from common.validation import ValidationOutcome
from extraction.agents.abbreviation import (
    AbbreviationResolutionAgent,
    best_long_form,
    find_pairs,
)
from extraction.agents.citation_extraction import InternalCitationExtractionAgent
from extraction.agents.cross_reference import CrossReferenceValidationAgent, find_cycles
from extraction.agents.entity_extraction import (
    EntityExtractionAgent,
    fallback_type,
    ner_candidates,
)
from extraction.schemas import (
    AbbreviationInput,
    CitationType,
    CrossRefInput,
    Entity,
    EntityRepository,
    EntityType,
    SectionHierarchy,
    SectionTask,
)
from extraction.schemas import ElementType as T
from tests.extraction.builders import el, hierarchy, section, task


def _doc(text1: str, text2: str = "Nothing here.") -> SectionHierarchy:
    s1 = section(
        "sec-1",
        "Intro",
        [
            el("h1", T.SECTION_HEADING, "1 Intro"),
            el("p1", T.PARAGRAPH, text1),
            el("fig", T.FIGURE, "", number="1"),
            el("tbl", T.TABLE, "", number="1", cells=[["a"]]),
            el("eq", T.EQUATION, "a = b", number="1"),
            el("alg", T.ALGORITHM, "Algorithm 1: Router", number="1"),
        ],
        number="1",
        subsections=[
            section(
                "sec-1.1",
                "Sub",
                [el("h11", T.SUBSECTION_HEADING, "1.1 Sub"), el("p11", T.PARAGRAPH, "Sub text.")],
                number="1.1",
                level=2,
            )
        ],
    )
    s2 = section(
        "sec-2",
        "Appendix A Proofs",
        [
            el("h2", T.SECTION_HEADING, "Appendix A Proofs", page=2),
            el("p2", T.PARAGRAPH, text2, page=2),
        ],
        number="2",
    )
    return hierarchy(s1, s2)


def _task(h: SectionHierarchy, idx: int = 0, last: bool = True) -> SectionTask:
    return task(h.sections[idx], h, page_count=2, last=last)


# ----------------------------------------------------------------- Alg. 13

TEXT = (
    "See Figure 1, Table 1, Eq. (1), Algorithm 1, Section 2, Subsection 1.1, Appendix A and page 2."
)


def test_citations_resolve_all_types(ctx: AgentContext) -> None:
    repo = InternalCitationExtractionAgent(ctx).run(_task(_doc(TEXT))).unwrap()
    got = {(c.citation_type, c.target_object) for c in repo.citations}
    assert got == {
        (CitationType.FIGURE, "fig"),
        (CitationType.TABLE, "tbl"),
        (CitationType.EQUATION, "eq"),
        (CitationType.ALGORITHM, "alg"),
        (CitationType.SECTION, "sec-2"),
        (CitationType.SUBSECTION, "sec-1.1"),
        (CitationType.APPENDIX, "sec-2"),
        (CitationType.PAGE, "page:2"),
    }
    assert repo.approved and repo.referenced["figure"] == ["fig"]
    assert all(c.source_element == "p1" for c in repo.citations)


def test_broken_and_invalid_citations_are_flagged(ctx: AgentContext) -> None:
    repo = (
        InternalCitationExtractionAgent(ctx)
        .run(_task(_doc("See Figure 9 and Table 0 and page 7.")))
        .unwrap()
    )
    failed = {c.name for c in repo.validation.errors}
    assert failed == {"broken_references", "invalid_numbering"}
    assert not repo.approved and ctx.review_queue.items


# ----------------------------------------------------------------- Alg. 14


def _xref(
    ctx: AgentContext,
    h: SectionHierarchy,
    idx: int = 0,
    last: bool = True,
    prior: list | None = None,
):  # type: ignore[no-untyped-def,type-arg]
    t = _task(h, idx, last)
    cits = InternalCitationExtractionAgent(ctx).run(t).unwrap()
    return (
        CrossReferenceValidationAgent(ctx)
        .run(CrossRefInput(task=t, citations=cits, prior_references=prior or []))
        .unwrap()
    )


def test_bidirectional_mapping_and_orphans_on_last_section(ctx: AgentContext) -> None:
    repo = _xref(ctx, _doc("As Figure 1 shows."))
    assert repo.forward == {"p1": ["fig"]} and repo.backward == {"fig": ["p1"]}
    assert set(repo.orphan_objects) == {"tbl", "eq"}  # algorithms are not orphan-checked
    orphan = repo.validation.get("orphan_objects")
    assert orphan is not None and not orphan.passed and repo.approved  # warning only


def test_orphans_not_checked_before_last_section(ctx: AgentContext) -> None:
    assert _xref(ctx, _doc("As Figure 1 shows."), last=False).orphan_objects == []


def test_unresolved_reference_gets_nearest_suggestion(ctx: AgentContext) -> None:
    repo = _xref(ctx, _doc("See Figure 2 and Section 3."))
    unresolved = [r for r in repo.references if r.status == "unresolved"]
    assert {r.suggestion for r in unresolved} == {"figure 1", "section 2"}
    assert not repo.approved and ctx.review_queue.items[-1].agent.startswith("CrossReference")


def test_non_unique_target_and_wrong_level(ctx: AgentContext) -> None:
    h = _doc("See Figure 1 and Subsection 2.")
    h.sections[1].elements.append(el("fig2", T.FIGURE, "", number="1", page=2))
    repo = _xref(ctx, h)
    failed = {c.name for c in repo.validation.errors}
    assert "object_uniqueness" in failed


def test_circular_section_references_warn(ctx: AgentContext) -> None:
    h = _doc("See Section 2.", "Back to Section 1.")
    first = _xref(ctx, h, 0, last=False)
    second = _xref(ctx, h, 1, prior=first.references)
    assert second.circular_references == [["sec-1", "sec-2"]] and second.approved
    assert find_cycles({"a": {"b"}, "b": {"c"}, "c": set()}) == []


# ----------------------------------------------------------------- Alg. 15

ENT_TEXT = (
    "The Model X200 pump uses PyTorch and follows ISO 9001. The Hydraulic Control Unit "
    "is built by Acme Industries in Berlin."
)


def _ent_task() -> SectionTask:
    return task(
        section(
            "sec-1",
            "Intro",
            [el("h", T.SECTION_HEADING, "1 Intro"), el("p", T.PARAGRAPH, ENT_TEXT)],
        )
    )


def test_entity_candidates_and_fallback_types() -> None:
    cands = ner_candidates(ENT_TEXT)
    assert {"ISO 9001", "PyTorch", "Acme Industries", "Hydraulic Control Unit"} <= set(cands)
    assert "The" not in cands
    assert fallback_type("ISO 9001") == EntityType.STANDARD
    assert fallback_type("PyTorch") == EntityType.SOFTWARE
    assert fallback_type("Berlin") == EntityType.DOMAIN_TERM


def test_llm_entities_are_validated(ctx: AgentContext) -> None:
    ctx.llm = MockLLM(
        {
            "entity_extraction": {
                "entities": [
                    {
                        "name": "PyTorch",
                        "type": "Software/Framework",
                        "description": "A framework.",
                        "attributes": [{"key": "kind", "value": "library"}],
                    },
                    {"name": "pytorch", "type": "Software/Framework", "description": "dup"},
                    {"name": "Berlin", "type": "Location", "description": "A city."},
                    {"name": "Mars Base", "type": "Location", "description": "Not in text."},
                    {"name": "Acme", "type": "Planet", "description": "bad type"},
                ]
            }
        }
    )
    repo = EntityExtractionAgent(ctx).run(_ent_task()).unwrap()
    names = [e.name for e in repo.entities]
    assert names == ["PyTorch", "Berlin", "Mars Base"]  # duplicate removed, invalid dropped
    assert repo.entities[0].attributes == {"kind": "library"} and repo.entities[0].context
    failed = {c.name for c in repo.validation.errors}
    assert failed == {"invalid_classifications", "context_consistency"}
    assert not repo.approved


def test_entity_rule_based_when_llm_disabled_or_failing(ctx: AgentContext) -> None:
    ctx.config.entity.use_llm = False
    repo = EntityExtractionAgent(ctx).run(_ent_task()).unwrap()
    types = {e.normalized_name: e.type for e in repo.entities}
    assert types["ISO 9001"] == EntityType.STANDARD and repo.approved
    ctx.config.entity.use_llm = True
    ctx.llm = MockLLM({"entity_extraction": [LLMError("down")]})
    assert EntityExtractionAgent(ctx).run(_ent_task()).unwrap().entities
    assert any("entity LLM" in i.reason for i in ctx.review_queue.items)


def test_entity_validation_detects_missing_description_and_duplicates() -> None:
    e = Entity(
        entity_id="e1",
        type=EntityType.MODEL,
        name="X",
        normalized_name="X",
        description="",
        section_id="s",
        page=1,
    )
    repo = EntityRepository(section_id="s", entities=[e, e.model_copy(update={"entity_id": "e2"})])
    v = EntityExtractionAgent.validate(repo, "X", [])
    assert {c.name for c in v.errors} == {"duplicate_entities", "missing_descriptions"}


# ----------------------------------------------------------------- Alg. 16


def test_schwartz_hearst_pairs() -> None:
    assert best_long_form("HCU", "the Hydraulic Control Unit") == "Hydraulic Control Unit"
    assert best_long_form("XYZ", "nothing matching here") is None
    pairs = find_pairs(
        "The Hydraulic Control Unit (HCU) and the API (Application Programming "
        "Interface) and U.S.A. (United States of America).",
        10,
    )
    assert ("HCU", "Hydraulic Control Unit") in pairs
    assert ("API", "Application Programming Interface") in pairs
    assert ("USA", "United States of America") in pairs


def _abbr_task(text: str, sid: str = "sec-1") -> SectionTask:
    return task(
        section(
            sid, "S", [el(f"{sid}-h", T.SECTION_HEADING, "1 S"), el(f"{sid}-p", T.PARAGRAPH, text)]
        )
    )


def test_abbreviations_repository_and_warnings(ctx: AgentContext) -> None:
    agent = AbbreviationResolutionAgent(ctx)
    first = agent.run(
        AbbreviationInput(
            task=_abbr_task(
                "The Hydraulic Control Unit (HCU) is checked. The HCU works. NASA is mentioned."
            )
        )
    ).unwrap()
    hcu = first.abbreviations["HCU"]
    assert hcu.expansion == "Hydraulic Control Unit" and hcu.occurrences == 2
    assert first.undefined == ["NASA"] and first.approved  # undefined is a warning
    second = agent.run(
        AbbreviationInput(task=_abbr_task("HCU again.", "sec-2"), existing=first.abbreviations)
    ).unwrap()
    assert second.abbreviations["HCU"].occurrences == 3
    assert second.abbreviations["HCU"].section_ids == ["sec-1", "sec-2"]


def test_conflicting_definitions_fail(ctx: AgentContext) -> None:
    repo = (
        AbbreviationResolutionAgent(ctx)
        .run(
            AbbreviationInput(
                task=_abbr_task("A Primary Control Unit (PCU) and later a Pump Cooling Unit (PCU).")
            )
        )
        .unwrap()
    )
    assert repo.conflicts == {"PCU": ["Primary Control Unit", "Pump Cooling Unit"]}
    assert not repo.approved and ctx.review_queue.items[-1].agent.startswith("Abbreviation")


def test_unused_and_duplicate_abbreviations_warn(ctx: AgentContext) -> None:
    repo = (
        AbbreviationResolutionAgent(ctx)
        .run(
            AbbreviationInput(
                task=_abbr_task("Machine Learning (ML) and Machine Learning (MLG) defined once.")
            )
        )
        .unwrap()
    )
    assert set(repo.unused) == {"ML", "MLG"}
    dup = repo.validation.get("duplicate_abbreviations")
    assert dup is not None and not dup.passed and repo.approved
    assert isinstance(repo.validation, ValidationOutcome)
