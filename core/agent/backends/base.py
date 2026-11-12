"""Backend protocols for planning vs biological annotation inference.

Roles may be split across objects (OpenAI planner + MetaboLLM inference) or
satisfied by one object in offline tests (MockBackend).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from core.agent.schemas import AnnotationPlan, ModelInventory, PlannerDecision, ToolObservation


class PlannerBackend(Protocol):
    """Orchestration only: emit validated PlannerDecision action groups."""

    name: str

    def identity(self) -> dict[str, Any]:
        """Exact planner model/resource identity for provenance."""

    def plan(
        self,
        *,
        request: str,
        inventory: ModelInventory,
        prior_plan: Optional[AnnotationPlan] = None,
        observations: Sequence[ToolObservation] = (),
        revision_reason: str = "",
    ) -> PlannerDecision:
        ...

    def unload(self) -> None:
        ...


class InferenceBackend(Protocol):
    """Biological annotation inference (local MetaboLLM / mock transport)."""

    name: str

    def identity(self) -> dict[str, Any]:
        """Exact inference model/resource identity for provenance."""

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Local grounded selector transport. Must not fall back to hosted providers."""
        ...

    def unload(self) -> None:
        ...


class AgentBackend(PlannerBackend, Protocol):
    """Compat protocol: planner + optional grounded transport (legacy single backend)."""

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Optional local grounded selector transport. Must not fall back to hosted providers."""
        ...
