"""LLM planning helpers with validated structured actions."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from core.agent.backends.mock import MockBackend
from core.agent.inspect_model import entities_needing_annotation
from core.agent.schemas import (
    AnnotationPlan,
    AnnotationScope,
    AnnotationStatus,
    EntityRuntimeState,
    ExecutionLimits,
    ModelInventory,
    PlannerDecision,
    PlannerDecisionName,
    TaskState,
    ToolObservation,
)
from core.agent.validate import PlanValidationError, validate_plan

_LEGACY_DECISION_MAP: dict[str, PlannerDecisionName] = {
    "plan": "continue",
    "revise": "continue",
    "continue": "continue",
    "stop": "finish",
    "finish": "finish",
    "ask_user": "needs_clarification",
    "needs_clarification": "needs_clarification",
}

_TERMINAL = {
    TaskState.COMPLETED,
    TaskState.SKIPPED,
    TaskState.FAILED,
    TaskState.UNRESOLVED,
}


class PlannerRequestError(RuntimeError):
    """Planner failed after optional generation; carries provenance meta."""

    def __init__(self, message: str, meta: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.meta = meta or {}


def _sanitize_action_dict(action: dict[str, Any]) -> dict[str, Any]:
    """Drop LLM-invented argument keys; keep allowlisted tool-arg fields only."""
    from core.agent.schemas import TOOL_ARG_MODELS

    out = dict(action)
    tool = out.get("tool")
    args = out.get("arguments")
    if isinstance(args, dict) and tool in TOOL_ARG_MODELS:
        model = TOOL_ARG_MODELS[tool]
        allowed = set(getattr(model, "model_fields", {}) or {})
        cleaned = {k: v for k, v in args.items() if k in allowed}
        # Common mistake: put tool/target identifiers inside arguments.
        out["arguments"] = cleaned
    if not out.get("task_ids") and out.get("target_ids"):
        # Synthesize task ids so compact plans can omit them.
        targets = list(out.get("target_ids") or [])
        out["task_ids"] = [f"{tool}::{tid}" for tid in targets]
    if not out.get("action_id"):
        out["action_id"] = f"act::{tool}::{'-'.join(out.get('target_ids') or ['x'])}"
    return out


def normalize_planner_decision(raw: PlannerDecision | dict[str, Any]) -> PlannerDecision:
    """Map legacy decision names onto continue/finish/needs_clarification."""
    if isinstance(raw, PlannerDecision):
        data = raw.model_dump(mode="python")
    else:
        data = dict(raw)
    legacy = str(data.get("decision") or "continue")
    mapped = _LEGACY_DECISION_MAP.get(legacy)
    if mapped is None:
        raise ValueError(f"unsupported planner decision: {legacy!r}")
    data["decision"] = mapped
    if not data.get("clarification_prompt") and data.get("ask_user_prompt"):
        data["clarification_prompt"] = data["ask_user_prompt"]
    if isinstance(data.get("actions"), list):
        data["actions"] = [
            _sanitize_action_dict(a) if isinstance(a, dict) else a for a in data["actions"]
        ]
    # Ensure tasks exist for synthesized task_ids when the LLM omitted tasks.
    if data.get("decision") == "continue" and data.get("actions"):
        if not data.get("tasks"):
            tasks = []
            retrieve_ids: dict[str, str] = {}
            for action in data["actions"]:
                if not isinstance(action, dict):
                    continue
                tool = action.get("tool")
                if tool not in {
                    "annotate_species_candidates",
                    "assess_identity_support",
                    "predict_qualifiers",
                    "retrieve_reaction_rrf",
                    "select_reaction_grounded",
                    "write_species_annotations",
                    "write_reaction_annotations",
                }:
                    continue
                for tid, task_id in zip(action.get("target_ids") or [], action.get("task_ids") or []):
                    if tool == "retrieve_reaction_rrf":
                        retrieve_ids[str(tid)] = str(task_id)
                    deps = []
                    if tool == "select_reaction_grounded" and str(tid) in retrieve_ids:
                        deps = [retrieve_ids[str(tid)]]
                    tasks.append(
                        {
                            "task_id": task_id,
                            "target_id": tid,
                            "representation": (
                                "sbml_reaction"
                                if tool
                                in {
                                    "retrieve_reaction_rrf",
                                    "select_reaction_grounded",
                                    "write_reaction_annotations",
                                }
                                else "sbml_species"
                            ),
                            "biological_type": (
                                "reaction"
                                if tool
                                in {
                                    "retrieve_reaction_rrf",
                                    "select_reaction_grounded",
                                    "write_reaction_annotations",
                                }
                                else "unknown"
                            ),
                            "tool": tool,
                            "depends_on": deps,
                            "arguments": action.get("arguments") or {},
                        }
                    )
            data["tasks"] = tasks
        else:
            # Repair missing select→retrieve dependencies when both tasks are present.
            by_id = {t.get("task_id"): t for t in data["tasks"] if isinstance(t, dict)}
            retrieve_by_target = {
                t.get("target_id"): t.get("task_id")
                for t in data["tasks"]
                if isinstance(t, dict) and t.get("tool") == "retrieve_reaction_rrf"
            }
            for t in data["tasks"]:
                if not isinstance(t, dict):
                    continue
                if t.get("tool") == "select_reaction_grounded":
                    rid = retrieve_by_target.get(t.get("target_id"))
                    deps = list(t.get("depends_on") or [])
                    if rid and rid not in deps:
                        deps.append(rid)
                        t["depends_on"] = deps
    # Prefer local metabolm model id in select arguments.
    for action in data.get("actions") or []:
        if isinstance(action, dict) and action.get("tool") == "select_reaction_grounded":
            args = dict(action.get("arguments") or {})
            if str(args.get("llm_model", "")).startswith("gpt"):
                args["llm_model"] = "metabolm-local"
            action["arguments"] = args
    return PlannerDecision.model_validate(data)


def compact_inventory_summary(inventory: ModelInventory) -> list[dict[str, Any]]:
    """Compact entity rows for planner prompts (fits small output budgets)."""
    rows: list[dict[str, Any]] = []
    for ent in inventory.entities:
        rows.append(
            {
                "id": ent.target_id,
                "representation": ent.representation.value,
                "biological_type": ent.biological_type.value,
                "status": ent.annotation_status.value,
                "name": ent.display_name,
                "participants": list(ent.participants)[:12],
                "participant_of": list(ent.participant_of)[:12],
            }
        )
    return rows


def unfinished_scope_targets(
    inventory: ModelInventory,
    entity_states: dict[str, EntityRuntimeState],
    scope: AnnotationScope,
    plan: Optional[AnnotationPlan] = None,
) -> list[str]:
    """Return in-scope target ids that are not yet terminal in the ledger."""
    unfinished: list[str] = []
    needed_ids = {
        e.target_id
        for e in entities_needing_annotation(
            inventory, scope_add_missing=scope == AnnotationScope.ADD_MISSING
        )
    }
    for ent in inventory.entities:
        if ent.annotation_status == AnnotationStatus.UNSUPPORTED:
            # Intentionally skipped / unsupported — terminal if marked skipped/failed/unresolved
            est = entity_states.get(ent.target_id)
            if est is None or est.state not in _TERMINAL:
                unfinished.append(ent.target_id)
            continue
        if scope == AnnotationScope.ADD_MISSING and ent.annotation_status in {
            AnnotationStatus.COMPLETE,
            AnnotationStatus.CONFLICTING,
        }:
            # Out of additive scope unless a replace task exists
            continue
        if ent.target_id not in needed_ids and scope == AnnotationScope.ADD_MISSING:
            # Not needing annotation under add_missing
            est = entity_states.get(ent.target_id)
            if est is not None and est.state in {TaskState.PENDING, TaskState.RUNNING}:
                unfinished.append(ent.target_id)
            continue
        est = entity_states.get(ent.target_id)
        if est is None or est.state not in _TERMINAL:
            unfinished.append(ent.target_id)
    if plan is not None:
        for task in plan.tasks:
            if task.state in {TaskState.PENDING, TaskState.RUNNING}:
                if task.target_id not in unfinished:
                    unfinished.append(task.target_id)
    return unfinished


def validate_finish(
    inventory: ModelInventory,
    entity_states: dict[str, EntityRuntimeState],
    scope: AnnotationScope,
    plan: Optional[AnnotationPlan],
) -> list[str]:
    """Reject finish when any in-scope target remains non-terminal."""
    leftover = unfinished_scope_targets(inventory, entity_states, scope, plan)
    return [f"finish rejected; non-terminal target: {tid}" for tid in leftover]


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
    decision = normalize_planner_decision(decision)
    if decision.decision == "needs_clarification":
        # No new executable actions on clarification.
        version = 1 if prior is None else prior.plan_version + 1
        plan = AnnotationPlan(
            plan_version=version,
            scope=scope,
            tasks=list(prior.tasks) if prior else list(decision.tasks),
            actions=[],
            revision_reason=revision_reason or decision.reason,
            settings=settings or {},
        )
        return plan

    if decision.decision == "finish":
        version = 1 if prior is None else prior.plan_version + 1
        plan = AnnotationPlan(
            plan_version=version,
            scope=scope,
            tasks=list(decision.tasks) if decision.tasks else (list(prior.tasks) if prior else []),
            actions=[],
            revision_reason=revision_reason or decision.reason,
            settings=settings or {},
        )
        # Finish may carry no new actions; validate task structures if present.
        if plan.tasks:
            return validate_plan(plan, inventory, limits)
        plan.limits = {
            "max_actions": limits.max_actions,
            "max_planning_turns": limits.max_planning_turns,
            "max_failure_revisions": limits.max_failure_revisions,
            "max_retries": limits.max_retries,
            "max_runtime_sec": limits.max_runtime_sec,
            "allow_names_only_reactions": limits.allow_names_only_reactions,
            "max_actions_per_turn": limits.max_actions_per_turn,
            "planner_max_new_tokens": limits.planner_max_new_tokens,
        }
        return plan

    # continue: bounded action group
    if len(decision.actions) > limits.max_actions_per_turn:
        raise PlanValidationError(
            f"continue action group size {len(decision.actions)} exceeds "
            f"max_actions_per_turn={limits.max_actions_per_turn}"
        )
    # Drop export_reports from mid-run groups; export is controller-owned after finish.
    actions = [a for a in decision.actions if a.tool != "export_reports"]
    if len(actions) > limits.max_actions_per_turn:
        raise PlanValidationError(
            f"continue action group size {len(actions)} exceeds "
            f"max_actions_per_turn={limits.max_actions_per_turn}"
        )

    version = 1 if prior is None else prior.plan_version + 1
    if prior is None:
        tasks = list(decision.tasks)
    else:
        by_id = {t.task_id: t for t in prior.tasks}
        for task in decision.tasks:
            prev = by_id.get(task.task_id)
            if prev is not None and prev.state == TaskState.COMPLETED:
                task.state = TaskState.COMPLETED
            by_id[task.task_id] = task
        tasks = list(by_id.values())

    plan = AnnotationPlan(
        plan_version=version,
        scope=scope,
        tasks=tasks,
        actions=actions,
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
    allow_structural_fallback: bool = False,
    entity_states: Optional[dict[str, EntityRuntimeState]] = None,
) -> tuple[AnnotationPlan, PlannerDecision, dict[str, Any]]:
    """Ask the backend for a plan; validate before returning.

    Malformed LLM actions never execute. Structural MockBackend fallback runs
    only when ``allow_structural_fallback`` is explicitly true and is labeled
    as fallback workflow orchestration in meta.
    """
    meta: dict[str, Any] = {
        "raw_decision": None,
        "raw_text": None,
        "fallback": False,
        "error": None,
        "usage": None,
        "truncated": False,
        "terminal_status": None,
        "execution_mode": None,
        "llm_planning_succeeded": None,
    }
    try:
        plan_kwargs: dict[str, Any] = {
            "request": request,
            "inventory": inventory,
            "prior_plan": prior_plan,
            "observations": observations,
            "revision_reason": revision_reason,
        }
        # Optional compact/limits awareness for backends that accept it.
        if hasattr(backend, "plan"):
            import inspect as _inspect

            sig = _inspect.signature(backend.plan)
            if "limits" in sig.parameters:
                plan_kwargs["limits"] = limits
            if "entity_states" in sig.parameters:
                plan_kwargs["entity_states"] = entity_states
        raw = backend.plan(**plan_kwargs)
        # Backends may return (decision, generate_meta) or just decision.
        generate_meta: dict[str, Any] = {}
        if isinstance(raw, tuple) and len(raw) == 2:
            decision_raw, generate_meta = raw
        else:
            decision_raw = raw
            generate_meta = dict(getattr(backend, "last_generate_meta", {}) or {})
        if isinstance(generate_meta, dict):
            meta["raw_text"] = generate_meta.get("raw_text")
            meta["usage"] = generate_meta.get("usage")
            meta["truncated"] = bool(generate_meta.get("truncated"))
            meta["terminal_status"] = generate_meta.get("terminal_status")
            if generate_meta.get("error"):
                meta["error"] = generate_meta.get("error")

        decision = normalize_planner_decision(decision_raw)
        meta["raw_decision"] = decision.model_dump(mode="json")
        meta["llm_planning_succeeded"] = True
        meta["execution_mode"] = "llm_planning"
        # Finish ledger checks are enforced by the controller (may re-plan on premature finish).

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
        meta["llm_planning_succeeded"] = False
        # Preserve any raw generation already captured / left on the backend.
        if meta.get("raw_text") is None and hasattr(backend, "last_generate_meta"):
            gm = getattr(backend, "last_generate_meta") or {}
            meta["raw_text"] = gm.get("raw_text")
            meta["usage"] = meta.get("usage") or gm.get("usage")
            meta["truncated"] = bool(gm.get("truncated"))
            meta["terminal_status"] = gm.get("terminal_status")
        if not allow_structural_fallback:
            raise PlannerRequestError(str(exc), meta) from exc
        fallback = MockBackend()._default_plan(inventory)
        fallback = normalize_planner_decision(fallback)
        # Bound fallback to one action group so it obeys max_actions_per_turn.
        if fallback.decision == "continue" and len(fallback.actions) > limits.max_actions_per_turn:
            fallback = PlannerDecision(
                decision="continue",
                tasks=list(fallback.tasks),
                actions=list(fallback.actions)[: max(1, limits.max_actions_per_turn)],
                reason=fallback.reason + " (bounded fallback action group)",
            )
        meta["fallback"] = True
        meta["execution_mode"] = "fallback_workflow_orchestration"
        meta["raw_decision"] = fallback.model_dump(mode="json")
        plan = build_plan_from_decision(
            fallback,
            inventory,
            scope=scope,
            limits=limits,
            prior=prior_plan,
            settings={
                **(settings or {}),
                "structural_fallback": True,
                "fallback_error": str(exc),
                "execution_mode": "fallback_workflow_orchestration",
            },
            revision_reason=f"structural fallback after planner error: {exc}",
        )
        return plan, fallback, meta
