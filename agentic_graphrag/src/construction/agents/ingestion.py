"""A1 / Algorithm 2 - Document Ingestion Agent.

Validates J and M against the Stage 1 JSON Schemas, derives a deterministic Document ID
from J's content (so re-ingesting the same document is idempotent), registers a
``Document`` node and starts a construction session. An invalid document raises a typed
failure carrying the error report, and the framework terminates the run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import jsonschema
from pydantic import BaseModel, Field, ValidationError

from common.base_agent import AgentFailure
from common.validation import CheckResult, ValidationOutcome
from construction.context import ConstructionAgent
from construction.models import GraphNode, Provenance, utc_now
from extraction.output_models import DocumentJSON, MetadataFile

# Fields that change between two extractions of the same file and are excluded from the
# content hash.
VOLATILE_FIELDS = ("processing_timestamp", "source_file")


class IngestionInput(BaseModel):
    document: dict[str, Any]
    metadata: dict[str, Any]


class ValidatedDocument(BaseModel):
    """D: the validated document object (Alg. 2 line 8)."""

    doc_id: str
    session_id: str
    content_hash: str
    source_document_id: str
    title: str
    category: str
    language: str
    total_pages: int
    reingested: bool = False
    abbreviations: dict[str, str] = Field(default_factory=dict)
    document: DocumentJSON
    metadata: MetadataFile
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)


def content_hash(document: dict[str, Any]) -> str:
    doc = copy.deepcopy(document)
    info = doc.get("document")
    if isinstance(info, dict):
        for key in VOLATILE_FIELDS:
            info.pop(key, None)
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def document_id_for(document: dict[str, Any]) -> str:
    return "doc-" + content_hash(document)[:16]


@cache
def _validator(path: str) -> jsonschema.protocols.Validator:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    validator: jsonschema.protocols.Validator = cls(schema)
    return validator


def schema_errors(instance: Any, schema_path: str | Path, limit: int = 20) -> list[str]:
    errors = sorted(_validator(str(schema_path)).iter_errors(instance), key=lambda e: list(e.path))
    out = []
    for err in errors[:limit]:
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(f"{where}: {err.message[:200]}")
    return out


class DocumentIngestionAgent(ConstructionAgent[IngestionInput, ValidatedDocument]):
    name: ClassVar[str] = "document_ingestion"
    algorithm: ClassVar[int] = 2
    agent_id: ClassVar[str] = "A1"

    def _execute(self, data: IngestionInput) -> ValidatedDocument:
        schemas = Path(self.settings.paths.schemas_dir)
        outcome = ValidationOutcome()
        outcome.add(
            CheckResult.from_failures(
                "document_schema", schema_errors(data.document, schemas / "document.schema.json")
            )
        )
        outcome.add(
            CheckResult.from_failures(
                "metadata_schema", schema_errors(data.metadata, schemas / "metadata.schema.json")
            )
        )
        doc_model: DocumentJSON | None = None
        meta_model: MetadataFile | None = None
        if outcome.passed:
            try:
                doc_model = DocumentJSON.model_validate(data.document)
                meta_model = MetadataFile.model_validate(data.metadata)
            except ValidationError as exc:
                outcome.add(CheckResult(name="model_parse", passed=False, reasons=[str(exc)[:500]]))
        if doc_model is not None and meta_model is not None:
            mismatch = []
            if (
                self.settings.ingestion.require_metadata_match
                and doc_model.document.document_id != meta_model.document.document_id
            ):
                mismatch.append(
                    f"J document_id {doc_model.document.document_id} != "
                    f"M document_id {meta_model.document.document_id}"
                )
            outcome.add(CheckResult.from_failures("metadata_matches_document", mismatch))
            has_text = any(s.text.strip() or s.chunks or s.tables for s in doc_model.sections)
            outcome.add(
                CheckResult.from_failures(
                    "has_content", [] if has_text else ["document has no section text"]
                )
            )

        if not outcome.passed or doc_model is None or meta_model is None:
            # Alg. 2 lines 11-13: error report, reject, terminate.
            reasons = outcome.failure_reasons()
            self.flag("document rejected at ingestion", details={"errors": reasons})
            raise AgentFailure(
                "invalid document: " + "; ".join(reasons[:10]),
                error_type="DocumentValidationError",
            )

        # Lines 4-8: metadata, Document ID, registration, session.
        digest = content_hash(data.document)
        doc_id = "doc-" + digest[:16]
        info = doc_model.document
        title = info.title or Path(info.source_file).stem or doc_id
        existing = self.ctx.graph.get_node(doc_id)
        now = utc_now()
        node = GraphNode(
            id=doc_id,
            label="Document",
            name=title,
            type=info.document_category,
            properties={
                "source_document_id": info.document_id,
                "content_hash": digest,
                "document_type": info.document_type,
                "category": info.document_category,
                "language": info.language,
                "total_pages": info.total_pages,
                "source_file": info.source_file,
                "author": info.author,
                "document_confidence": meta_model.quality.document_confidence,
                "ingest_count": int(existing.properties.get("ingest_count", 0)) + 1
                if existing
                else 1,
                "last_ingested_at": now,
            },
            provenance=[
                Provenance(
                    doc_id=doc_id,
                    chunk_id="document",
                    page=1,
                    timestamp=now,
                    confidence=1.0,
                    source_ref=info.source_file or info.document_id,
                )
            ],
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self.ctx.graph.upsert_node(node)
        session_id = uuid.uuid4().hex
        self.ctx.document_id = doc_id
        self.ctx.session_id = session_id

        abbreviations = {a.abbreviation: a.expansion for a in doc_model.abbreviations}
        for section in doc_model.sections:
            for a in section.abbreviations:
                abbreviations.setdefault(a.abbreviation, a.expansion)
        return ValidatedDocument(
            doc_id=doc_id,
            session_id=session_id,
            content_hash=digest,
            source_document_id=info.document_id,
            title=title,
            category=info.document_category,
            language=info.language,
            total_pages=info.total_pages,
            reingested=existing is not None,
            abbreviations=abbreviations,
            document=doc_model,
            metadata=meta_model,
            validation=outcome,
        )
