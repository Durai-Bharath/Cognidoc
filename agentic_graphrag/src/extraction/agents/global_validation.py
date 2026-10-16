"""Algorithm 21 - Global Validation Agent."""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from common.base_agent import BaseAgent
from common.validation import CheckResult, Severity, ValidationOutcome
from extraction.agents.confidence import level_for
from extraction.output_models import (
    CheckJSON,
    DocumentJSON,
    DocumentStatistics,
    MetadataFile,
    ValidationCategory,
    ValidationReport,
    ValidationStatistics,
)
from extraction.schema_export import load_schema, schema_errors
from extraction.schemas import EntityType


class GlobalValidationInput(BaseModel):
    metadata: MetadataFile
    document: DocumentJSON
    agent_errors: list[str] = Field(default_factory=list)


REPROCESS = {
    "document_information": "RouterAgent",
    "section_hierarchy": "SectionIdentificationAgent",
    "page_numbering": "LayoutDetectionAgent",
    "document_statistics": "MetadataFusionAgent",
    "tables": "TableAnalysisAgent",
    "figures": "FigureUnderstandingAgent",
    "equations": "EquationAnalysisAgent",
    "entities": "EntityExtractionAgent",
    "abbreviations": "AbbreviationResolutionAgent",
    "cross_references": "CrossReferenceValidationAgent",
    "section_summaries": "TextSummarizationAgent",
    "keywords": "TextSummarizationAgent",
    "confidence_scores": "ConfidenceScoringAgent",
    "object_associations": "MetadataFusionAgent",
    "required_fields": "MetadataFusionAgent",
    "object_hierarchy": "MetadataFusionAgent",
    "data_types": "MetadataFusionAgent",
    "duplicate_entries": "MetadataFusionAgent",
    "broken_references": "CrossReferenceValidationAgent",
    "missing_objects": "MetadataFusionAgent",
    "orphan_sections": "SectionIdentificationAgent",
    "invalid_relationships": "MetadataFusionAgent",
}


def recompute_statistics(doc: DocumentJSON) -> DocumentStatistics:
    s = doc.sections
    return DocumentStatistics(
        total_sections=len(s),
        total_subsections=sum(len(x.subsections) for x in s),
        total_tables=sum(len(x.tables) for x in s),
        total_figures=sum(len(x.figures) for x in s),
        total_equations=sum(len(x.equations) for x in s),
        total_entities=sum(len(x.entities) for x in s),
        total_abbreviations=len(doc.abbreviations),
        total_cross_references=sum(len(x.cross_references) for x in s),
        total_chunks=sum(len(x.chunks) for x in s),
        total_words=sum(len(x.text.split()) for x in s),
    )


def _all_ids(doc: DocumentJSON) -> list[str]:
    ids: list[str] = []
    for s in doc.sections:
        ids.append(s.section_id)
        ids += [x.section_id for x in s.subsections]
        ids += [x.table_id for x in s.tables] + [x.figure_id for x in s.figures]
        ids += [x.equation_id for x in s.equations] + [x.entity_id for x in s.entities]
        ids += [x.chunk_id for x in s.chunks] + [x.reference_id for x in s.cross_references]
    return ids


def _has_nan(value: Any) -> bool:
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, dict):
        return any(_has_nan(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_nan(v) for v in value)
    return False


class GlobalValidationAgent(BaseAgent[GlobalValidationInput, ValidationReport]):
    name = "GlobalValidationAgent"
    algorithm = 21

    def _execute(self, data: GlobalValidationInput) -> ValidationReport:
        cfg = self.ctx.config.global_validation
        m, j = data.metadata, data.document
        categories = {
            "document_integrity": self._document_integrity(m, j),
            "extracted_objects": self._extracted_objects(j),
            "metadata_consistency": self._metadata_consistency(m, j),
            "json_schema": self._json_schema(m, j),
            "cross_document_consistency": self._cross_document(j),
        }
        weights = cfg.weights
        total_w = sum(weights.get(k, 0.0) for k in categories) or 1.0
        # Q = Σ w_i q_i
        quality = sum(weights.get(k, 0.0) * o.pass_ratio for k, o in categories.items()) / total_w
        checks = [c for o in categories.values() for c in o.checks]
        errors = [c for c in checks if not c.passed and c.severity == Severity.ERROR]
        warnings = [c for c in checks if not c.passed and c.severity == Severity.WARNING]
        passed = [c for c in checks if c.passed]
        approved = not errors and not data.agent_errors and quality >= cfg.approve_threshold
        recommendations = sorted(
            {
                f"Reprocess with {REPROCESS.get(c.name, 'MetadataFusionAgent')} ({c.name})"
                for c in errors
            }
        )
        if data.agent_errors:
            recommendations.append("Re-run failed agents: " + "; ".join(data.agent_errors))
        report = ValidationReport(
            document_id=m.document.document_id,
            generated_at=datetime.now(UTC).isoformat(),
            approved=approved,
            quality_score=round(min(1.0, max(0.0, quality)), 4),
            categories=[
                ValidationCategory(
                    name=name,
                    weight=weights.get(name, 0.0),
                    score=round(o.pass_ratio, 4),
                    checks=[
                        CheckJSON(
                            name=c.name,
                            passed=c.passed,
                            severity=c.severity.value,
                            reasons=c.reasons,
                        )
                        for c in o.checks
                    ],
                )
                for name, o in categories.items()
            ],
            statistics=ValidationStatistics(
                validation_accuracy=round(len(passed) / len(checks), 4) if checks else 1.0,
                total_errors=len(errors),
                warnings=len(warnings),
                passed_checks=len(passed),
                total_checks=len(checks),
            ),
            recommendations=recommendations,
            agent_errors=data.agent_errors,
        )
        if not approved:
            self.flag(
                "document failed global validation",
                item_ref=m.document.document_id,
                details={
                    "errors": [f"{c.name}: {c.reasons[:3]}" for c in errors],
                    "quality_score": report.quality_score,
                },
            )
        return report

    def confidence(self, output: ValidationReport) -> float | None:
        return output.quality_score

    # ------------------------------------------------------------------ categories
    @staticmethod
    def _document_integrity(m: MetadataFile, j: DocumentJSON) -> ValidationOutcome:
        o = ValidationOutcome()
        d = m.document
        info: list[str] = []
        if not d.document_id:
            info.append("missing document_id")
        if not d.title:
            info.append("missing title")
        if d.total_pages < 1:
            info.append("document has no pages")
        if d.document_id != j.document.document_id:
            info.append("M and J describe different documents")
        o.add(CheckResult.from_failures("document_information", info))

        hierarchy: list[str] = []
        for s in j.sections:
            if s.level != 1:
                hierarchy.append(f"{s.section_id} top-level section has level {s.level}")
            for sub in s.subsections:
                if not sub.section_id.startswith(s.section_id + "."):
                    hierarchy.append(f"{sub.section_id} not nested under {s.section_id}")
                if sub.level < 2:
                    hierarchy.append(f"{sub.section_id} subsection has level {sub.level}")
        o.add(CheckResult.from_failures("section_hierarchy", hierarchy))

        pages: list[str] = []
        n = d.total_pages
        for s in j.sections:
            if not 1 <= s.page_start <= s.page_end <= max(n, 1):
                pages.append(f"{s.section_id} pages {s.page_start}-{s.page_end} of {n}")
            for obj_page, oid in (
                [(t.page, t.table_id) for t in s.tables]
                + [(f.page, f.figure_id) for f in s.figures]
                + [(e.page, e.equation_id) for e in s.equations]
            ):
                if not 1 <= obj_page <= max(n, 1):
                    pages.append(f"{oid} on page {obj_page} of {n}")
        o.add(CheckResult.from_failures("page_numbering", pages))

        recomputed = recompute_statistics(j)
        stats = [
            f"{k}: M={getattr(m.statistics, k)} J={getattr(j.statistics, k)} "
            f"actual={getattr(recomputed, k)}"
            for k in DocumentStatistics.model_fields
            if not getattr(m.statistics, k) == getattr(j.statistics, k) == getattr(recomputed, k)
        ]
        o.add(CheckResult.from_failures("document_statistics", stats))
        return o

    @staticmethod
    def _extracted_objects(j: DocumentJSON) -> ValidationOutcome:
        o = ValidationOutcome()
        tables, figures, equations, entities, xrefs = [], [], [], [], []
        valid_types = {t.value for t in EntityType}
        for s in j.sections:
            for t in s.tables:
                if any(len(r) != len(t.headers) for r in t.rows):
                    tables.append(f"{t.table_id} rows do not match header width")
                if not t.headers:
                    tables.append(f"{t.table_id} has no headers")
            for f in s.figures:
                if not f.description.strip():
                    figures.append(f"{f.figure_id} has no description")
            for eq in s.equations:
                if not eq.latex.strip():
                    equations.append(f"{eq.equation_id} has empty LaTeX")
            for ent in s.entities:
                if not ent.name.strip() or ent.type not in valid_types:
                    entities.append(f"{ent.entity_id} invalid ({ent.type!r}, {ent.name!r})")
            for x in s.cross_references:
                if x.status == "resolved" and not x.target_object:
                    xrefs.append(f"{x.reference_id} resolved without target")
        abbrs = [a.abbreviation for a in j.abbreviations]
        abbr_issues = [f"{a} defined {n} times" for a, n in Counter(abbrs).items() if n > 1]
        abbr_issues += [
            f"{a.abbreviation} has no expansion" for a in j.abbreviations if not a.expansion.strip()
        ]
        for name, failures in (
            ("tables", tables),
            ("figures", figures),
            ("equations", equations),
            ("entities", entities),
            ("abbreviations", abbr_issues),
            ("cross_references", xrefs),
        ):
            o.add(CheckResult.from_failures(name, failures))
        return o

    def _metadata_consistency(self, m: MetadataFile, j: DocumentJSON) -> ValidationOutcome:
        o = ValidationOutcome()
        conf_cfg = self.ctx.config.confidence
        by_id = {s.section_id: s for s in j.sections}
        summaries, kws, confs, assoc = [], [], [], []
        for ms in m.sections:
            js = by_id.get(ms.section_id)
            if js is None:
                assoc.append(f"{ms.section_id} in M but not in J")
                continue
            if js.text.strip() and not ms.summary.strip():
                summaries.append(f"{ms.section_id} has no summary")
            if ms.summary != js.summary.summary:
                summaries.append(f"{ms.section_id} summary differs between M and J")
            if ms.keywords != js.keywords:
                kws.append(f"{ms.section_id} keywords differ between M and J")
            elif js.text.strip() and len(js.text.split()) > 20 and not ms.keywords:
                kws.append(f"{ms.section_id} has text but no keywords")
            if (
                not 0 <= ms.confidence_score <= 1
                or level_for(ms.confidence_score, conf_cfg).value != ms.confidence_level
                or ms.confidence_score != js.confidence.score
            ):
                confs.append(
                    f"{ms.section_id} confidence {ms.confidence_score} "
                    f"({ms.confidence_level}) inconsistent"
                )
            pairs = (
                ("table_ids", [t.table_id for t in js.tables]),
                ("figure_ids", [f.figure_id for f in js.figures]),
                ("equation_ids", [e.equation_id for e in js.equations]),
                ("entity_ids", [e.entity_id for e in js.entities]),
                ("cross_reference_ids", [x.reference_id for x in js.cross_references]),
                ("subsection_ids", [x.section_id for x in js.subsections]),
            )
            for field, ids in pairs:
                if getattr(ms, field) != ids:
                    assoc.append(f"{ms.section_id}.{field} differs between M and J")
        if len(m.sections) != len(j.sections):
            assoc.append(f"M has {len(m.sections)} sections, J has {len(j.sections)}")
        o.add(CheckResult.from_failures("section_summaries", summaries))
        o.add(CheckResult.from_failures("keywords", kws, Severity.WARNING))
        o.add(CheckResult.from_failures("confidence_scores", confs))
        o.add(CheckResult.from_failures("object_associations", assoc))
        return o

    def _json_schema(self, m: MetadataFile, j: DocumentJSON) -> ValidationOutcome:
        o = ValidationOutcome()
        schemas_dir = self.ctx.config.paths.schemas_dir
        m_dump, j_dump = m.model_dump(mode="json"), j.model_dump(mode="json")
        required = [
            f"M {e}"
            for e in schema_errors(m_dump, load_schema(schemas_dir, "metadata.schema.json"))
        ]
        required += [
            f"J {e}"
            for e in schema_errors(j_dump, load_schema(schemas_dir, "document.schema.json"))
        ]
        o.add(CheckResult.from_failures("required_fields", required))
        hierarchy = []
        for s in j.sections:
            for oid in (
                [t.table_id for t in s.tables]
                + [f.figure_id for f in s.figures]
                + [e.equation_id for e in s.equations]
                + [e.entity_id for e in s.entities]
                + [c.chunk_id for c in s.chunks]
            ):
                if not oid.startswith(s.section_id + "-") and not oid.startswith(
                    s.section_id + "."
                ):
                    hierarchy.append(f"{oid} is not owned by {s.section_id}")
        o.add(CheckResult.from_failures("object_hierarchy", hierarchy))
        o.add(
            CheckResult.from_failures(
                "data_types",
                (["non-finite numbers in M"] if _has_nan(m_dump) else [])
                + (["non-finite numbers in J"] if _has_nan(j_dump) else []),
            )
        )
        dupes = [f"{i} x{n}" for i, n in Counter(_all_ids(j)).items() if n > 1]
        o.add(CheckResult.from_failures("duplicate_entries", dupes))
        return o

    @staticmethod
    def _cross_document(j: DocumentJSON) -> ValidationOutcome:
        o = ValidationOutcome()
        broken = [
            f"{x.reference_id} ({x.citation_type}) unresolved"
            + (f"; suggestion: {x.suggestion}" if x.suggestion else "")
            for s in j.sections
            for x in s.cross_references
            if x.status == "unresolved"
        ]
        o.add(CheckResult.from_failures("broken_references", broken))
        known = set(_all_ids(j)) | {f"abbr:{a.abbreviation}" for a in j.abbreviations}
        missing = sorted(
            {
                f"{r.type}: {end}"
                for r in j.relationships
                for end in (r.source, r.target)
                if end not in known
            }
        )
        o.add(CheckResult.from_failures("missing_objects", missing))
        orphans = [
            f"{s.section_id} '{s.title}' has no content"
            for s in j.sections
            if not s.subsections
            and not s.tables
            and not s.figures
            and not s.equations
            and len(s.text.split())
            <= len(s.title.split()) + (len(s.number.split()) if s.number else 0)
        ]
        o.add(CheckResult.from_failures("orphan_sections", orphans, Severity.WARNING))
        section_ids = {s.section_id for s in j.sections} | {
            x.section_id for s in j.sections for x in s.subsections
        }
        invalid = [
            f"{r.source} -{r.type}-> {r.target}"
            for r in j.relationships
            if r.source == r.target
            or (r.type == "contains" and r.source not in section_ids)
            or (r.type == "defines" and not r.source.startswith("abbr:"))
        ]
        o.add(CheckResult.from_failures("invalid_relationships", invalid))
        return o
