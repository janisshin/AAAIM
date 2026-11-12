"""LLM planning helpers with validated structured actions."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from core.agent.backends.mock import MockBackend
from core.agent.schemas import (
    AnnotationPlan,
    AnnotationScope,
    ExecutionLimits,
    ModelInventory,
    PlannerDecision,
    ToolObservation,
)
from core.agent.validate import PlanValidationError, validate_plan


def build_plan_from_decision(
    decision: PlannerDecision,
    inventory: ModelInventory,
    *,
    scope: AnnotationScope,
    limits: ExecutionLimits,
    prior: Optional[AnnotationPlan] = None,
    settings: Optional[dict[str, Any]] = None,
    revision_reason: str = "",
) -> AnnotationPlan:
    version = 1 if prior is None else prior.plan_version + 1
    plan = AnnotationPlan(
        plan_version=version,
        scope=scope,
        tasks=list(decision.tasks) if decision.tasks else (list(prior.tasks) if prior else []),
        actions=list(decision.actions),
        revision_reason=revision_reason or decision.reason,
        settings=settings or {},
    )
    return validate_plan(plan, inventory, limits)


def request_plan(
    backend: Any,
    *,
    request: str,
    inventory: ModelInventory,
    scope: AnnotationScope,
    limits: ExecutionLimits,
    prior_plan: Optional[AnnotationPlan] = None,
    observations: Sequence[ToolObservation] = (),
    revision_reason: str = "",
    settings: Optional[dict[str, Any]] = None,
    allow_structural_fallback: bool = True,
) -> tuple[AnnotationPlan, PlannerDecision, dict[str, Any]]:
    """Ask the backend for a plan; validate before returning.

    Malformed LLM actions never execute. If the backend raises and
    ``allow_structural_fallback`` is true, a deterministic MockBackend structural
    plan is used and recorded as a technical fallback in provenance.
    """
    meta: dict[str, Any] = {"raw_decision": None, "fallback": False, "error": None}
    try:
        decision = backend.plan(
            request=request,
            inventory=inventory,
            prior_plan=prior_plan,
            observations=observations,
            revision_reason=revision_reason,
        )
        meta["raw_decision"] = decision.model_dump(mode="json")
        plan = build_plan_from_decision(
            decision,
            inventory,
            scope=scope,
            limits=limits,
            prior=prior_plan,
            settings=settings,
            revision_reason=revision_reason,
        )
        return plan, decision, meta
    except (RuntimeError, PlanValidationError, ValueError) as exc:
        meta["error"] = str(exc)
        if not allow_structural_fallback:
            raise
        fallback = MockBackend()._default_plan(inventory)
        meta["fallback"] = True
        meta["raw_decision"] = fallback.model_dump(mode="json")
        plan = build_plan_from_decision(
            fallback,
            inventory,
            scope=scope,
            limits=limits,
            prior=prior_plan,
            settings={**(settings or {}), "structural_fallback": True, "fallback_error": str(exc)},
            revision_reason=f"structural fallback after planner error: {exc}",
        )
        return plan, fallback, meta
