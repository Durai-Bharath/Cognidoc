"""A3 / Algorithm 4 - Triple Extraction Agent.

The LLM returns ``(Head, Relation, Tail, Confidence)`` as schema-validated JSON. Malformed
output (bad JSON, a confidence outside [0, 1]) is retried by ``LLMClient.complete_json``,
which feeds the validation error back to the model. Triples are then normalised: names
trimmed, empty and self-referential triples dropped, duplicates collapsed (highest
confidence wins). Each triple is stamped with its chunk-level provenance.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from common.base_agent import AgentFailure
from common.llm import LLMError
from construction.context import ConstructionAgent
from construction.models import (
    Provenance,
    SemanticChunk,
    Triple,
    TripleExtractionOutput,
    clean_surface,
    normalize_name,
    stable_hash,
)

SYSTEM = (
    "You extract a knowledge graph from technical documents. Return factual "
    "(head, relation, tail) triples stated in the TEXT, each with a confidence in [0, 1] "
    "reflecting how explicitly the text states it. Use short canonical entity names, "
    "never whole clauses. Prefer the listed predicates when one fits; otherwise use a "
    "short verb phrase. CONTEXT is the end of the previous chunk: use it only to resolve "
    "references, never as a source of triples."
)


class TripleExtractionInput(BaseModel):
    chunk: SemanticChunk
    document_title: str = ""


class TripleSet(BaseModel):
    chunk_id: str
    triples: list[Triple] = Field(default_factory=list)
    raw_count: int = 0
    dropped: list[str] = Field(default_factory=list)


def build_prompt(data: TripleExtractionInput, predicates: str, types: list[str]) -> str:
    c = data.chunk
    return (
        f"DOCUMENT: {data.document_title}\n"
        f"SECTION: {' > '.join(c.section_path) or c.section_title}\n"
        f"ENTITY TYPES: {', '.join(types)}\n"
        f"PREDICATES:\n{predicates}\n\n"
        f"CONTEXT:\n<<<\n{c.context}\n>>>\n\n"
        f"TEXT:\n<<<\n{c.text}\n>>>"
    )


class TripleExtractionAgent(ConstructionAgent[TripleExtractionInput, TripleSet]):
    name: ClassVar[str] = "triple_extraction"
    algorithm: ClassVar[int] = 4
    agent_id: ClassVar[str] = "A3"

    def _execute(self, data: TripleExtractionInput) -> TripleSet:
        chunk = data.chunk
        onto = self.ctx.ontology
        prompt = build_prompt(data, onto.describe_predicates(), list(onto.entity_types))
        try:
            raw = self.llm_json(
                task="triple_extraction",
                system=SYSTEM,
                prompt=prompt,
                schema=TripleExtractionOutput,
            )
        except LLMError as exc:
            raise AgentFailure(
                f"triple extraction failed for {chunk.chunk_id}: {exc}",
                error_type="TripleExtractionError",
                recoverable=True,
            ) from exc

        out = TripleSet(chunk_id=chunk.chunk_id, raw_count=len(raw.triples))
        best: dict[tuple[str, str, str], Triple] = {}
        for rt in raw.triples:
            head, tail = clean_surface(rt.head), clean_surface(rt.tail)
            relation = " ".join(rt.relation.split())
            if not head or not tail or not relation:
                out.dropped.append(f"incomplete triple {rt.model_dump()}")
                continue
            if normalize_name(head) == normalize_name(tail):
                out.dropped.append(f"self-referential triple {head!r}")
                continue
            triple = Triple(
                triple_id="tr-" + stable_hash(chunk.chunk_id, head, relation, tail),
                head=head,
                head_type=onto.canonical_type(rt.head_type),
                relation=relation,
                tail=tail,
                tail_type=onto.canonical_type(rt.tail_type),
                confidence=rt.confidence,
                llm_confidence=rt.confidence,
                provenance=Provenance(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page=chunk.page_start,
                    confidence=rt.confidence,
                    source_ref=chunk.source_ref,
                ),
                mentions={"head": rt.head, "relation": rt.relation, "tail": rt.tail},
            )
            key = triple.key()
            if key not in best or best[key].confidence < triple.confidence:
                if key in best:
                    out.dropped.append(f"duplicate triple {key}")
                best[key] = triple
            else:
                out.dropped.append(f"duplicate triple {key}")
        limit = self.settings.triples.max_triples_per_chunk
        triples = list(best.values())
        if len(triples) > limit:
            out.dropped += [f"over limit: {t.key()}" for t in triples[limit:]]
            triples = triples[:limit]
        out.triples = triples
        return out
