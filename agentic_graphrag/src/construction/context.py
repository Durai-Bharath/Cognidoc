"""Shared dependencies for Stage 2 agents: Stage 1's ``AgentContext`` plus KG, VDB and O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

from common.base_agent import AgentContext, BaseAgent
from construction.config import ConstructionConfig
from construction.models import ConflictRecord
from construction.ontology import Ontology
from construction.stores.graph_store import GraphStore
from construction.stores.vector_store import VectorStore

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


@dataclass(kw_only=True)
class ConstructionContext(AgentContext):
    settings: ConstructionConfig
    graph: GraphStore
    vectors: VectorStore
    ontology: Ontology
    session_id: str | None = None
    conflicts: list[ConflictRecord] = field(default_factory=list)


class ConstructionAgent(BaseAgent[InT, OutT], Generic[InT, OutT]):
    """``BaseAgent`` bound to a :class:`ConstructionContext`.

    ``algorithm`` is the algorithm number in *Construction.pdf* (A1 = Algorithm 2 …
    A15 = Algorithm 16); ``agent_id`` is the agent's id in Table 1.
    """

    agent_id: ClassVar[str]

    def __init__(self, ctx: ConstructionContext) -> None:
        super().__init__(ctx)
        self.ctx: ConstructionContext = ctx

    @property
    def settings(self) -> ConstructionConfig:
        return self.ctx.settings
