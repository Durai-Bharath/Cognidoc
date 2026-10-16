"""A6 / Algorithm 7 - Entity Resolution Agent.

For every distinct mention:

1. an existing node with the same canonical id (same normalised name) is an exact link;
2. otherwise the mention's embedding is compared with existing ``Entity`` nodes through
   ``GraphStore.similar_nodes``. Cosine ≥ δ (with compatible types) links to that node;
3. otherwise a *new* entity is proposed, with a deterministic id. Earlier new mentions in
   the same batch are matched too, so one chunk never proposes two near-identical nodes.

Nothing is written here. New nodes are only created by Graph CRUD (A10) on a Create
decision, so rejected or conflicting triples leave the entity graph untouched.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from common.embedder import Vector, cosine
from construction.context import ConstructionAgent
from construction.models import EntityRef, ResolvedTriple, Triple, entity_node_id, normalize_name
from construction.ontology import FALLBACK_TYPE


class EntityResolutionInput(BaseModel):
    triples: list[Triple]


class ResolvedTriples(BaseModel):
    items: list[ResolvedTriple] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)

    @property
    def linked(self) -> int:
        return sum(not e.is_new for e in self.entities)

    @property
    def new(self) -> int:
        return sum(e.is_new for e in self.entities)


def types_compatible(a: str, b: str) -> bool:
    return a == b or FALLBACK_TYPE in (a, b)


class EntityResolutionAgent(ConstructionAgent[EntityResolutionInput, ResolvedTriples]):
    name: ClassVar[str] = "entity_resolution"
    algorithm: ClassVar[int] = 7
    agent_id: ClassVar[str] = "A6"

    def _execute(self, data: EntityResolutionInput) -> ResolvedTriples:
        cfg = self.settings.entity_resolution
        graph = self.ctx.graph
        refs: dict[tuple[str, str], EntityRef] = {}
        batch_new: list[tuple[EntityRef, Vector]] = []

        mentions: list[tuple[str, str]] = []
        for t in data.triples:
            for name, etype in ((t.head, t.head_type), (t.tail, t.tail_type)):
                if (normalize_name(name), etype) not in {
                    (normalize_name(n), e) for n, e in mentions
                }:
                    mentions.append((name, etype))
        vectors = (
            self.ctx.embedder.embed([n for n, _ in mentions]) if mentions else np.zeros((0, 1))
        )

        for (name, etype), vec in zip(mentions, vectors, strict=True):
            key = (normalize_name(name), etype)
            # 1. exact canonical id
            node = graph.get_node(entity_node_id(name))
            if (
                node is not None
                and node.label == "Entity"
                and (not cfg.require_type_compatibility or types_compatible(node.type, etype))
            ):
                refs[key] = EntityRef(
                    mention=name,
                    name=node.name,
                    type=etype if node.type == FALLBACK_TYPE else node.type,
                    node_id=node.id,
                    is_new=False,
                    similarity=1.0,
                    matched_name=node.name,
                )
                continue
            # 2. embedding similarity against the KG
            best: EntityRef | None = None
            for cand, score in graph.similar_nodes(vec, label="Entity", top_k=cfg.top_k):
                if score < cfg.delta:
                    break
                if cfg.require_type_compatibility and not types_compatible(cand.type, etype):
                    continue
                best = EntityRef(
                    mention=name,
                    name=cand.name,
                    type=etype if cand.type == FALLBACK_TYPE else cand.type,
                    node_id=cand.id,
                    is_new=False,
                    similarity=round(score, 4),
                    matched_name=cand.name,
                )
                break
            # 2b. earlier new mentions in this batch
            if best is None:
                for other, ovec in batch_new:
                    score = cosine(vec, ovec)
                    if score >= cfg.delta and (
                        not cfg.require_type_compatibility or types_compatible(other.type, etype)
                    ):
                        best = other.model_copy(
                            update={"mention": name, "similarity": round(score, 4)}
                        )
                        break
            # 3. new entity
            if best is None:
                best = EntityRef(
                    mention=name, name=name, type=etype, node_id=entity_node_id(name), is_new=True
                )
                batch_new.append((best, vec))
            refs[key] = best

        out = ResolvedTriples()
        for t in data.triples:
            out.items.append(
                ResolvedTriple(
                    triple=t,
                    head=refs[(normalize_name(t.head), t.head_type)],
                    tail=refs[(normalize_name(t.tail), t.tail_type)],
                )
            )
        unique: dict[str, EntityRef] = {}
        for ref in refs.values():
            unique.setdefault(f"{ref.node_id}|{ref.mention}", ref)
        out.entities = list(unique.values())
        return out
