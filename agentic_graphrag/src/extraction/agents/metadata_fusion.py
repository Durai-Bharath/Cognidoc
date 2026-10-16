"""Algorithm 20 - Metadata Fusion & JSON Appending Agent."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from common.base_agent import BaseAgent
from common.validation import CheckResult, ValidationOutcome
from extraction.agents.confidence import level_for
from extraction.output_models import (
    AbbreviationJSON,
    BBoxJSON,
    ChunkJSON,
    ConfidenceJSON,
    CrossReferenceJSON,
    DocumentInfo,
    DocumentJSON,
    DocumentStatistics,
    EntityJSON,
    EquationJSON,
    EquationVariableJSON,
    FigureConnectorJSON,
    FigureJSON,
    FigureObjectJSON,
    MetadataFile,
    QualityInfo,
    RelationshipJSON,
    SectionJSON,
    SectionMetadata,
    SubsectionJSON,
    SummaryJSON,
    TableJSON,
)
from extraction.schema_export import load_schema, schema_errors
from extraction.schemas import (
    Abbreviation,
    BBox,
    FusionInput,
    Section,
    SectionResults,
)


class FusionOutput(BaseModel):
    metadata: MetadataFile
    document: DocumentJSON
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    export_approved: bool = True


def _bbox(b: BBox) -> BBoxJSON:
    return BBoxJSON(x=round(b.x, 2), y=round(b.y, 2), w=round(b.w, 2), h=round(b.h, 2))


def _clamp(v: float) -> float:
    return 0.0 if v != v else round(min(1.0, max(0.0, v)), 4)  # NaN → 0


class MetadataFusionAgent(BaseAgent[FusionInput, FusionOutput]):
    name = "MetadataFusionAgent"
    algorithm = 20

    def _execute(self, data: FusionInput) -> FusionOutput:
        parsed = data.parsed
        props = parsed.properties
        # Lines 4-9: document-level metadata.
        info = DocumentInfo(
            document_id=parsed.document_id,
            document_type=parsed.doc_type.value,
            document_category=data.classification.category,
            category_confidence=_clamp(data.classification.confidence),
            title=data.hierarchy.document_title or props.title,
            author=props.author,
            creation_date=props.creation_date,
            language=props.language,
            total_pages=props.page_count,
            file_size_bytes=props.file_size_bytes,
            mime_type=props.mime_type,
            source_file=parsed.source_path,
            processing_timestamp=datetime.now(UTC).isoformat(),
        )
        abbreviations = data.abbreviations.abbreviations if data.abbreviations else {}
        el_to_obj: dict[str, str] = {}
        for res in data.results.values():
            for t in res.tables.tables if res.tables else []:
                el_to_obj[t.element_id] = t.table_id
            for f in res.figures.figures if res.figures else []:
                el_to_obj[f.element_id] = f.figure_id
            for e in res.equations.equations if res.equations else []:
                el_to_obj[e.element_id] = e.equation_id

        sections_meta: list[SectionMetadata] = []
        sections_json: list[SectionJSON] = []
        relationships: list[RelationshipJSON] = []
        # Lines 10-26: per section metadata, objects, text, relationships.
        for section in data.hierarchy.sections:
            res = data.results.get(
                section.section_id, SectionResults(section_id=section.section_id)
            )
            meta, js, rels = self._section(section, res, abbreviations, el_to_obj)
            sections_meta.append(meta)
            sections_json.append(js)
            relationships.extend(rels)
        for abbr, entry in sorted(abbreviations.items()):
            for sid in entry.section_ids:
                relationships.append(
                    RelationshipJSON(source=f"abbr:{abbr}", target=sid, type="defines")
                )
        for cont in data.ordered.continuations:
            a, b = el_to_obj.get(cont.from_id), el_to_obj.get(cont.to_id)
            if a and b:
                relationships.append(RelationshipJSON(source=a, target=b, type="continues"))
        relationships = list({(r.source, r.target, r.type): r for r in relationships}.values())

        # Lines 27-32: document statistics.
        stats = DocumentStatistics(
            total_sections=len(sections_json),
            total_subsections=sum(len(s.subsections) for s in sections_json),
            total_tables=sum(len(s.tables) for s in sections_json),
            total_figures=sum(len(s.figures) for s in sections_json),
            total_equations=sum(len(s.equations) for s in sections_json),
            total_entities=sum(len(s.entities) for s in sections_json),
            total_abbreviations=len(abbreviations),
            total_cross_references=sum(len(s.cross_references) for s in sections_json),
            total_chunks=sum(len(s.chunks) for s in sections_json),
            total_words=sum(len(s.text.split()) for s in sections_json),
        )
        doc_conf = self._document_confidence(data)
        quality = QualityInfo(
            ocr_applied=parsed.requires_ocr,
            ocr_quality_score=data.quality.document_score if data.quality else None,
            enhanced_pages=[p.page for p in data.enhancement.pages if p.accepted]
            if data.enhancement
            else [],
            layout_confidence=_clamp(data.layout.confidence),
            document_confidence=doc_conf[0],
            confidence_level=doc_conf[1],
        )
        metadata = MetadataFile(
            document=info,
            sections=sections_meta,
            statistics=stats,
            quality=quality,
            review_flags=data.review_flags,
        )
        document = DocumentJSON(
            document=info,
            sections=sections_json,
            abbreviations=[
                AbbreviationJSON(abbreviation=a, expansion=e.expansion, occurrences=e.occurrences)
                for a, e in sorted(abbreviations.items())
            ],
            relationships=relationships,
            statistics=stats,
        )

        # Lines 33-38: validate against the published schemas.
        schemas_dir = self.ctx.config.paths.schemas_dir
        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures(
                "metadata_schema",
                schema_errors(
                    metadata.model_dump(mode="json"),
                    load_schema(schemas_dir, "metadata.schema.json"),
                ),
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "json_schema",
                schema_errors(
                    document.model_dump(mode="json"),
                    load_schema(schemas_dir, "document.schema.json"),
                ),
            )
        )
        out = FusionOutput(
            metadata=metadata, document=document, validation=outcome, export_approved=outcome.passed
        )
        if not outcome.passed:
            self.flag(
                "metadata/JSON schema validation failed",
                item_ref=parsed.document_id,
                details={"failures": outcome.failure_reasons()},
            )
        return out

    # ------------------------------------------------------------------ helpers
    def _document_confidence(self, data: FusionInput) -> tuple[float, str]:
        """Token-weighted mean of section confidences (Alg. 19 line 26, over all sections)."""
        sections = [
            o
            for r in data.results.values()
            if r.confidence
            for o in r.confidence.objects
            if o.kind == "section"
        ]
        total = sum(o.weight for o in sections)
        score = _clamp(sum(o.score * o.weight for o in sections) / total if total else 0.0)
        return score, level_for(score, self.ctx.config.confidence).value

    def _section(
        self,
        section: Section,
        res: SectionResults,
        abbreviations: dict[str, Abbreviation],
        el_to_obj: dict[str, str],
    ) -> tuple[SectionMetadata, SectionJSON, list[RelationshipJSON]]:
        sid = section.section_id
        conf = next(
            (o for o in (res.confidence.objects if res.confidence else []) if o.object_id == sid),
            None,
        )
        summary = res.summary
        status = "Unchecked"
        if res.hallucination:
            item = next(
                (i for i in res.hallucination.items if i.content_id == f"{sid}:summary"), None
            )
            status = item.status if item else "Unchecked"
        tables = res.tables.tables if res.tables else []
        figures = res.figures.figures if res.figures else []
        equations = res.equations.equations if res.equations else []
        entities = res.entities.entities if res.entities else []
        xrefs = res.cross_references.references if res.cross_references else []
        chunks = res.memory.chunks if res.memory else []
        section_abbrs = sorted(a for a, e in abbreviations.items() if sid in e.section_ids)
        subsections = [s for s in section.iter_sections() if s.section_id != sid]

        conf_json = ConfidenceJSON(
            score=_clamp(conf.score) if conf else 0.0,
            level=conf.level.value if conf else "Low",
            components={k: _clamp(v) for k, v in (conf.components if conf else {}).items()},
        )
        meta = SectionMetadata(
            section_id=sid,
            number=section.number,
            title=section.title,
            level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
            summary=summary.summary if summary else "",
            keywords=summary.keywords if summary else [],
            confidence_score=conf_json.score,
            confidence_level=conf_json.level,
            subsection_ids=[s.section_id for s in subsections],
            table_ids=[t.table_id for t in tables],
            figure_ids=[f.figure_id for f in figures],
            equation_ids=[e.equation_id for e in equations],
            entity_ids=[e.entity_id for e in entities],
            abbreviations=section_abbrs,
            cross_reference_ids=[x.reference_id for x in xrefs],
        )
        js = SectionJSON(
            section_id=sid,
            number=section.number,
            title=section.title,
            level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
            text=section.text(),
            keywords=meta.keywords,
            summary=SummaryJSON(
                summary=meta.summary,
                abstract=summary.abstract if summary else "",
                key_concepts=summary.key_concepts if summary else [],
                important_entities=summary.important_entities if summary else [],
                confidence=_clamp(summary.confidence) if summary else 0.0,
                verification_status=status,
            ),
            confidence=conf_json,
            subsections=[
                SubsectionJSON(
                    section_id=s.section_id,
                    number=s.number,
                    title=s.title,
                    level=s.level,
                    page_start=s.page_start,
                    page_end=s.page_end,
                    text="\n\n".join(e.text for e in s.elements if e.text),
                )
                for s in subsections
            ],
            tables=[
                TableJSON(
                    table_id=t.table_id,
                    number=t.number,
                    caption=t.caption,
                    page=t.page,
                    bbox=_bbox(t.bbox),
                    headers=t.headers,
                    rows=t.rows,
                    units=t.units,
                    key_attributes=t.key_attributes,
                    description=t.description,
                    summary=t.summary,
                    keywords=t.keywords,
                    approved=t.approved,
                )
                for t in tables
            ],
            figures=[
                FigureJSON(
                    figure_id=f.figure_id,
                    number=f.number,
                    caption=f.caption,
                    page=f.page,
                    bbox=_bbox(f.bbox),
                    figure_type=f.figure_type.value,
                    objects=[FigureObjectJSON(**o.model_dump()) for o in f.objects],
                    connectors=[FigureConnectorJSON(**c.model_dump()) for c in f.connectors],
                    relationships=f.relationships,
                    workflow=f.workflow,
                    legends=f.legends,
                    description=f.description,
                    summary=f.summary,
                    keywords=f.keywords,
                    image_path=f.image_path,
                    approved=f.approved,
                )
                for f in figures
            ],
            equations=[
                EquationJSON(
                    equation_id=e.equation_id,
                    number=e.number,
                    latex=e.latex,
                    page=e.page,
                    bbox=_bbox(e.bbox),
                    equation_type=e.equation_type.value,
                    variables=[EquationVariableJSON(**v.model_dump()) for v in e.variables],
                    description=e.description,
                    approved=e.approved,
                )
                for e in equations
            ],
            entities=[
                EntityJSON(
                    entity_id=e.entity_id,
                    type=e.type.value,
                    name=e.name,
                    description=e.description,
                    page=e.page,
                )
                for e in entities
            ],
            abbreviations=[
                AbbreviationJSON(
                    abbreviation=a,
                    expansion=abbreviations[a].expansion,
                    occurrences=abbreviations[a].occurrences,
                )
                for a in section_abbrs
            ],
            cross_references=[
                CrossReferenceJSON(
                    reference_id=x.reference_id,
                    citation_type=x.citation_type.value,
                    source_element=x.source_element,
                    target_object=el_to_obj.get(x.target_object or "", x.target_object),
                    target_section=x.target_section,
                    status=x.status,
                    suggestion=x.suggestion,
                )
                for x in xrefs
            ],
            chunks=[
                ChunkJSON(
                    chunk_id=c.chunk_id,
                    order=c.order,
                    text=c.text,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    token_count=c.token_count,
                    previous_chunk=c.previous_chunk,
                    next_chunk=c.next_chunk,
                    summary=c.summary,
                )
                for c in chunks
            ],
        )
        rels: list[RelationshipJSON] = []
        for child in [
            *(s.section_id for s in section.subsections),
            *meta.table_ids,
            *meta.figure_ids,
            *meta.equation_ids,
            *meta.entity_ids,
            *(c.chunk_id for c in chunks),
        ]:
            rels.append(RelationshipJSON(source=sid, target=child, type="contains"))
        for sub in subsections:
            rels += [
                RelationshipJSON(source=sub.section_id, target=s.section_id, type="contains")
                for s in sub.subsections
            ]
        for x in xrefs:
            if x.status == "resolved" and x.target_object:
                target = el_to_obj.get(x.target_object, x.target_object)
                if target.startswith("page:") or target == sid:
                    continue
                if target.startswith("sec-") or target in el_to_obj.values():
                    rels.append(RelationshipJSON(source=sid, target=target, type="references"))
        if res.memory:
            for edge in res.memory.edges:
                rels.append(
                    RelationshipJSON(
                        source=edge.source,
                        target=edge.target,
                        type="chunk_next" if edge.kind == "next" else "chunk_similar",
                    )
                )
        return meta, js, rels
