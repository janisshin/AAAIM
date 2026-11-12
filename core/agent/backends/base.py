"""Backend protocol for planning and optional grounded transport."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

from core.agent.schemas import AnnotationPlan, ModelInventory, PlannerDecision, ToolObservation


class AgentBackend(Protocol):
    name: str

    def identity(self) -> dict[str, Any]:
        """Exact model/resource identity for provenance."""

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

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Optional local grounded selector transport. Must not fall back to hosted providers."""
        ...

    def unload(self) -> None:
        ...
