"""Shared dependencies for Stage 3 agents: Stage 1's ``AgentContext`` plus KG, VDB and cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from common.base_agent import AgentContext, BaseAgent
from construction.stores.graph_store import GraphStore
from construction.stores.vector_store import VectorStore
from retrieval.config import RetrievalConfig

if TYPE_CHECKING:
    from retrieval.agents.cache_manager import SemanticCache

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


@dataclass(kw_only=True)
class RetrievalAgentContext(AgentContext):
    settings: RetrievalConfig
    graph: GraphStore
    vectors: VectorStore
    cache: SemanticCache


class RetrievalAgent(BaseAgent[InT, OutT], Generic[InT, OutT]):
    """``BaseAgent`` bound to a :class:`RetrievalAgentContext`.

    ``algorithm`` is the algorithm number in *Retrieval.pdf* (A1 = Algorithm 2 …
    A15 = Algorithm 16); ``agent_id`` is the agent's id in Table 1.
    """

    agent_id: ClassVar[str]

    def __init__(self, ctx: RetrievalAgentContext) -> None:
        super().__init__(ctx)
        self.ctx: RetrievalAgentContext = ctx

    @property
    def settings(self) -> RetrievalConfig:
        return self.ctx.settings
