"""Inspect → plan → execute → check → revise → export controller."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.agent.export import export_artifacts
from core.agent.inspect_model import entities_needing_annotation, inspect_sbml_model
from core.agent.io import atomic_write_json
from core.agent.planner import (
    PlannerRequestError,
    request_plan,
    unfinished_scope_targets,
    validate_finish,
)
from core.agent.schemas import (
    AgentAction,
    AgentCheckpoint,
    AnnotationPlan,
    AnnotationScope,
    AnnotationStatus,
    AutoAnnotateResult,
    CheckpointCompatibilityError,
    CheckpointProvenance,
    EntityRuntimeState,
    ExecutionLimits,
    FailureKind,
    InFlightAction,
    TaskState,
    TOOL_SCHEMA_VERSION,
    ToolObservation,
)
from core.agent.resource_identity import (
    assert_resource_identity_compatible,
    build_resource_identity,
)
from core.agent.tools import AgentResources, ToolRegistry
from core.agent.validate import PlanValidationError, validate_action

_TERMINAL_TASK_STATES = {
    TaskState.COMPLETED,
    TaskState.SKIPPED,
    TaskState.FAILED,
    TaskState.UNRESOLVED,
}


def _action_fingerprint(action: AgentAction) -> str:
    payload = {
        "tool": action.tool,
        "target_ids": list(action.target_ids),
        "task_ids": list(action.task_ids),
        "arguments": action.arguments,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_identity(
    model_sha: str,
    backend_id: dict[str, Any],
    settings: dict[str, Any],
    *,
    tool_schema_version: str,
    resource_identity: dict[str, Any],
) -> str:
    blob = json.dumps(
        {
            "model_sha256": model_sha,
            "backend": backend_id,
            "settings": settings,
            "tool_schema_version": tool_schema_version,
            "resource_identity": resource_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resource_identity(
    *,
    reaction_retriever: Any,
    reaction_resources: Any,
    species_annotator_provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Content-based resource identity plus absolute paths for provenance only."""
    return build_resource_identity(
        reaction_retriever=reaction_retriever,
        reaction_resources=reaction_resources,
        species_annotator_provenance=species_annotator_provenance,
    )


def _resolve_roles(
    *,
    backend: Any,
    planner: Any,
    inference: Any,
) -> tuple[Any, Any]:
    """Resolve planner/inference backends with single-backend compat."""
    if planner is None and inference is None:
        if backend is None:
            from core.agent.backends.mock import MockBackend

            backend = MockBackend()
        return backend, backend
    if planner is not None and inference is not None:
        return planner, inference
    if planner is not None and inference is None:
        return planner, backend if backend is not None else planner
    return backend if backend is not None else inference, inference


class AnnotationAgentController:
    def __init__(
        self,
        *,
        model_file: str,
        request: str,
        backend: Any = None,
        planner: Any = None,
        inference: Any = None,
        output_dir: str | Path,
        resources: Any = None,
        reaction_retriever: Any = None,
        scope: AnnotationScope = AnnotationScope.ADD_MISSING,
        limits: Optional[ExecutionLimits] = None,
        resume_from: Optional[str | Path] = None,
        species_annotator: Any = None,
        allow_structural_fallback: bool = True,
        interrupt_after_actions: Optional[int] = None,
    ) -> None:
        self.model_file = str(model_file)
        self.request = request
        self.planner, self.inference = _resolve_roles(
            backend=backend, planner=planner, inference=inference
        )
        self.backend = self.planner  # compat alias for planning/unload
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scope = scope
        self.limits = limits or ExecutionLimits()
        self.resume_from = Path(resume_from) if resume_from else None
        self.allow_structural_fallback = allow_structural_fallback
        self.interrupt_after_actions = interrupt_after_actions
        self.inventory = inspect_sbml_model(
            self.model_file, scope_add_missing=scope == AnnotationScope.ADD_MISSING
        )
        from core.agent.species import resolve_species_annotator

        resolved_annotator, species_prov = resolve_species_annotator(
            self.inference, species_annotator
        )
        grounded = getattr(self.inference, "grounded_transport", None)
        if getattr(self.inference, "name", "") == "openai_planner":
            grounded = None
        species_backend = self.inference if hasattr(self.inference, "generate") else None
        if getattr(species_backend, "name", "") == "openai_planner":
            species_backend = None
        self.budget_path = self.output_dir / "planner_budget.json"
        if hasattr(self.planner, "configure_budget"):
            self.planner.budget_path = self.budget_path
            self.planner.configure_budget(
                spend_authorized=bool(self.limits.planner_spend_authorized),
                hard_cap_usd=float(self.limits.planner_spend_cap_usd),
                max_attempts=int(self.limits.max_planning_turns),
            )
        self.agent_resources = AgentResources(
            model_file=self.model_file,
            output_dir=self.output_dir,
            inventory=self.inventory,
            reaction_resources=resources,
            reaction_retriever=reaction_retriever,
            grounded_transport=grounded,
            species_annotator=resolved_annotator,
            species_backend=species_backend,
            species_annotator_provenance=species_prov,
            authorized_annotation_models=list(
                self.limits.authorized_annotation_models or ["metabolm-local"]
            ),
            allow_names_only_reactions=self.limits.allow_names_only_reactions,
            checkpoint_saver=self._save_checkpoint,
        )
        self.tools = ToolRegistry(self.agent_resources)
        self.observations: list[ToolObservation] = []
        self.entity_states: dict[str, EntityRuntimeState] = {}
        self.completed_fingerprints: set[str] = set()
        self.action_counts: dict[str, int] = {}
        self.initial_plan: Optional[AnnotationPlan] = None
        self.executed_plan: Optional[AnnotationPlan] = None
        self.revision_count = 0  # failure-driven (compat)
        self.failure_revision_count = 0
        self.planning_turn_count = 0
        self.last_planner_decision: Optional[str] = None
        self.clarification_prompt = ""
        self.actions_executed = 0
        self.in_flight_action: Optional[InFlightAction] = None
        self._restored_from_checkpoint = False
        self._awaiting_planning_turn = True
        self.started = time.time()
        self.resource_identity, resource_paths = _resource_identity(
            reaction_retriever=reaction_retriever,
            reaction_resources=resources,
            species_annotator_provenance=species_prov,
        )
        planner_identity = (
            self.planner.identity()
            if hasattr(self.planner, "identity")
            else {"backend": type(self.planner).__name__, "role": "planner"}
        )
        inference_identity = (
            self.inference.identity()
            if hasattr(self.inference, "identity")
            else {"backend": type(self.inference).__name__, "role": "inference"}
        )
        composite_backend = {"planner": planner_identity, "inference": inference_identity}
        self.provenance: dict[str, Any] = {
            "backend": composite_backend,
            "planner": planner_identity,
            "inference": inference_identity,
            "settings": {
                "scope": scope.value,
                "limits": self.limits.model_dump(mode="json"),
                "request": request,
            },
            "species_annotator": species_prov,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "resource_identity": self.resource_identity,
            "planner_meta": [],
            "revisions": [],
            "inference_stages": [],
            "planner_usage": {},
            "execution_semantics": "at_least_once",
        }
        # Absolute resource paths are audit-only; resume identity uses content fingerprints.
        self.provenance["resource_paths"] = dict(resource_paths)
        if hasattr(self.inference, "resource_paths"):
            self.provenance["backend_resource_paths"] = dict(
                getattr(self.inference, "resource_paths") or {}
            )
        if hasattr(self.inference, "load_info") and getattr(self.inference, "load_info"):
            self.provenance["backend_load_info"] = dict(getattr(self.inference, "load_info") or {})
        if hasattr(self.planner, "budget"):
            self.provenance["planner_usage"] = self.planner.budget.summary()
        self.provenance["cache_identity"] = _cache_identity(
            self.inventory.sha256,
            self.provenance["backend"],
            self.provenance["settings"],
            tool_schema_version=TOOL_SCHEMA_VERSION,
            resource_identity=self.resource_identity,
        )
        if self.resume_from:
            self._restore_checkpoint(self.resume_from)

    def _checkpoint_path(self, path: Path) -> Path:
        return path / "checkpoint.json" if path.is_dir() else path

    def _build_checkpoint(self) -> AgentCheckpoint:
        planner_budget = {}
        if hasattr(self.planner, "budget"):
            planner_budget = self.planner.budget.summary()
            self.provenance["planner_usage"] = planner_budget
        return AgentCheckpoint(
            schema_version=3,
            cache_identity=self.provenance["cache_identity"],
            model_sha256=self.inventory.sha256,
            backend=dict(self.provenance.get("backend") or {}),
            planner=dict(self.provenance.get("planner") or {}),
            inference=dict(self.provenance.get("inference") or {}),
            settings=dict(self.provenance.get("settings") or {}),
            tool_schema_version=TOOL_SCHEMA_VERSION,
            resource_identity=dict(self.resource_identity),
            planner_budget=planner_budget,
            initial_plan=self.initial_plan,
            executed_plan=self.executed_plan,
            entity_states=dict(self.entity_states),
            observations=list(self.observations),
            completed_fingerprints=sorted(self.completed_fingerprints),
            action_counts=dict(self.action_counts),
            actions_executed=self.actions_executed,
            revision_count=self.failure_revision_count,
            planning_turn_count=self.planning_turn_count,
            failure_revision_count=self.failure_revision_count,
            last_planner_decision=self.last_planner_decision,  # type: ignore[arg-type]
            species_recommendations=None
            if self.agent_resources.species_recommendations is None
            else self.agent_resources.species_recommendations.fillna("").to_dict("records"),
            reaction_recommendations=None
            if self.agent_resources.reaction_recommendations is None
            else self.agent_resources.reaction_recommendations.fillna("").to_dict("records"),
            reaction_evidence=list(self.agent_resources.reaction_evidence),
            entity_identities=dict(self.agent_resources.entity_identities),
            paths=dict(self.agent_resources.checkpoint),
            provenance=CheckpointProvenance(
                planner_meta=list(self.provenance.get("planner_meta") or []),
                revisions=list(self.provenance.get("revisions") or []),
                stop_reason=self.provenance.get("stop_reason"),
                inference_stages=list(self.provenance.get("inference_stages") or []),
                species_annotator=dict(self.provenance.get("species_annotator") or {}),
                last_planner_decision=self.last_planner_decision,
                clarification_prompt=self.clarification_prompt,
                execution_mode=self.provenance.get("execution_mode"),
                llm_planning_succeeded=self.provenance.get("llm_planning_succeeded"),
                planner_prompt_version=(self.provenance.get("planner") or {}).get(
                    "prompt_version"
                ),
                planner_schema_version=(self.provenance.get("planner") or {}).get(
                    "schema_version"
                ),
            ),
            in_flight_action=self.in_flight_action,
        )

    def _validate_evidence_dependencies(self, ckpt: AgentCheckpoint) -> None:
        retrieve_done = any(o.tool == "retrieve_reaction_rrf" and o.ok for o in ckpt.observations)
        select_ok = any(o.tool == "select_reaction_grounded" and o.ok for o in ckpt.observations)
        if ckpt.executed_plan is not None:
            for task in ckpt.executed_plan.tasks:
                if task.tool == "retrieve_reaction_rrf" and task.state == TaskState.COMPLETED:
                    retrieve_done = True
                if task.tool == "select_reaction_grounded" and task.state in {
                    TaskState.COMPLETED,
                    TaskState.UNRESOLVED,
                }:
                    select_ok = True
        if (retrieve_done or select_ok) and not ckpt.reaction_evidence:
            raise CheckpointCompatibilityError(
                "checkpoint missing reaction_evidence required by completed retrieval/selection"
            )
        if select_ok and ckpt.reaction_recommendations is None:
            raise CheckpointCompatibilityError(
                "checkpoint missing reaction_recommendations required by completed selection"
            )

    def _restore_checkpoint(self, path: Path) -> None:
        ckpt_path = self._checkpoint_path(path)
        if not ckpt_path.is_file():
            raise CheckpointCompatibilityError(
                f"checkpoint not found at {ckpt_path}; cannot resume (pre-v1 or missing)"
            )
        try:
            raw = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointCompatibilityError(f"corrupt checkpoint JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise CheckpointCompatibilityError("checkpoint root must be a JSON object")
        if "schema_version" not in raw:
            raise CheckpointCompatibilityError(
                "unsupported checkpoint: missing schema_version "
                "(pre-v3 checkpoints cannot be resumed; restart without resume_from)"
            )
        if raw.get("schema_version") != 3:
            raise CheckpointCompatibilityError(
                f"unsupported checkpoint schema_version={raw.get('schema_version')!r}; "
                "only schema_version=3 is supported (requires dual planner/inference "
                "identities and planner budget ledger; restart without resume_from)"
            )
        try:
            ckpt = AgentCheckpoint.model_validate(raw)
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"incomplete or invalid checkpoint schema: {exc}"
            ) from exc
        if not ckpt.planner or not ckpt.inference:
            raise CheckpointCompatibilityError(
                "checkpoint missing planner/inference identities required by schema_version=3"
            )

        mismatches: list[str] = []
        if ckpt.model_sha256 != self.inventory.sha256:
            mismatches.append("model_sha256")
        if ckpt.backend != self.provenance["backend"]:
            mismatches.append("backend")
        if ckpt.planner != self.provenance.get("planner"):
            mismatches.append("planner")
        if ckpt.inference != self.provenance.get("inference"):
            mismatches.append("inference")
        if ckpt.settings != self.provenance["settings"]:
            mismatches.append("settings")
        if ckpt.tool_schema_version != TOOL_SCHEMA_VERSION:
            mismatches.append("tool_schema_version")
        # Incomplete/legacy resource identities raise with an explicit restart message.
        assert_resource_identity_compatible(ckpt.resource_identity, self.resource_identity)
        if ckpt.cache_identity != self.provenance["cache_identity"]:
            mismatches.append("cache_identity")
        if mismatches:
            raise CheckpointCompatibilityError(
                "checkpoint incompatible with current run: " + ", ".join(mismatches)
            )

        self._validate_evidence_dependencies(ckpt)

        self.initial_plan = ckpt.initial_plan
        self.executed_plan = ckpt.executed_plan
        self.entity_states = dict(ckpt.entity_states)
        self.observations = list(ckpt.observations)
        self.completed_fingerprints = set(ckpt.completed_fingerprints)
        self.action_counts = {k: int(v) for k, v in ckpt.action_counts.items()}
        self.actions_executed = int(ckpt.actions_executed)
        self.failure_revision_count = int(
            getattr(ckpt, "failure_revision_count", None) or ckpt.revision_count or 0
        )
        self.revision_count = self.failure_revision_count
        self.planning_turn_count = int(getattr(ckpt, "planning_turn_count", 0) or 0)
        self.last_planner_decision = getattr(ckpt, "last_planner_decision", None)
        self.clarification_prompt = str(
            getattr(ckpt.provenance, "clarification_prompt", "") or ""
        )
        self.agent_resources.entity_identities = dict(ckpt.entity_identities)
        self.agent_resources.checkpoint = dict(ckpt.paths)
        self.agent_resources.reaction_evidence = list(ckpt.reaction_evidence)
        if ckpt.species_recommendations is not None:
            self.agent_resources.species_recommendations = pd.DataFrame(ckpt.species_recommendations)
        if ckpt.reaction_recommendations is not None:
            self.agent_resources.reaction_recommendations = pd.DataFrame(ckpt.reaction_recommendations)

        if ckpt.planner_budget and hasattr(self.planner, "budget"):
            from core.agent.planner_budget import PlannerBudgetLedger

            self.planner.budget = PlannerBudgetLedger.from_dict(dict(ckpt.planner_budget))
            self.planner.budget_path = self.budget_path
            self.provenance["planner_usage"] = self.planner.budget.summary()
            if self.budget_path is not None:
                self.planner.budget.persist(self.budget_path)

        prov = ckpt.provenance
        self.provenance["planner_meta"] = list(prov.planner_meta)
        self.provenance["revisions"] = list(prov.revisions)
        if prov.stop_reason:
            # Clear interrupt stop so a resumed run can continue unless limits hit again.
            if prov.stop_reason != "interrupt_after_actions":
                self.provenance["stop_reason"] = prov.stop_reason
        self.provenance["inference_stages"] = list(prov.inference_stages)
        if prov.species_annotator:
            self.provenance["species_annotator"] = dict(prov.species_annotator)

        # At-least-once: incomplete tool result before durable completion → re-execute.
        if ckpt.in_flight_action is not None:
            fp = ckpt.in_flight_action.fingerprint
            if fp not in self.completed_fingerprints:
                self.in_flight_action = None
                self.provenance.setdefault("resume_notes", []).append(
                    "cleared in_flight_action for at-least-once re-execution"
                )
                # Roll task/entity states back so the scheduler can retry the action.
                if self.executed_plan is not None:
                    inflight_task_ids = set(ckpt.in_flight_action.action.task_ids)
                    for task in self.executed_plan.tasks:
                        if task.task_id not in inflight_task_ids:
                            continue
                        if task.state in {TaskState.RUNNING, TaskState.COMPLETED}:
                            task.state = TaskState.PENDING
                            task.reason = "re-queued after incomplete durable record"
                        est = self.entity_states.get(task.target_id)
                        if est is not None and est.state in {
                            TaskState.RUNNING,
                            TaskState.COMPLETED,
                        }:
                            est.state = TaskState.PENDING
            else:
                self.in_flight_action = None

        # Reset RUNNING tasks left by an interrupted save so they can run again.
        if self.executed_plan is not None:
            for task in self.executed_plan.tasks:
                if task.state == TaskState.RUNNING:
                    task.state = TaskState.PENDING

        self._restored_from_checkpoint = self.executed_plan is not None

    def _save_checkpoint(self) -> None:
        ckpt = self._build_checkpoint()
        atomic_write_json(ckpt.model_dump(mode="json"), self.output_dir / "checkpoint.json")

    def _init_entity_states(self) -> None:
        needed = entities_needing_annotation(
            self.inventory, scope_add_missing=self.scope == AnnotationScope.ADD_MISSING
        )
        for ent in self.inventory.entities:
            if ent.target_id in self.entity_states:
                continue
            if ent.annotation_status == AnnotationStatus.UNSUPPORTED:
                self.entity_states[ent.target_id] = EntityRuntimeState(
                    target_id=ent.target_id,
                    state=TaskState.SKIPPED,
                    failure_kind=FailureKind.UNSUPPORTED_IDENTITY,
                    notes=list(ent.support_notes) or ["unsupported representation"],
                )
            elif ent.annotation_status in {
                AnnotationStatus.COMPLETE,
                AnnotationStatus.CONFLICTING,
            } and self.scope == AnnotationScope.ADD_MISSING:
                self.entity_states[ent.target_id] = EntityRuntimeState(
                    target_id=ent.target_id,
                    state=TaskState.SKIPPED,
                    notes=["existing annotation preserved under add_missing scope"],
                )
            elif ent in needed or ent.target_id in {e.target_id for e in needed}:
                self.entity_states[ent.target_id] = EntityRuntimeState(
                    target_id=ent.target_id, state=TaskState.PENDING
                )

    def _task_ready(self, plan: AnnotationPlan, task_id: str) -> bool:
        by_id = {t.task_id: t for t in plan.tasks}
        task = by_id.get(task_id)
        if task is None:
            return False
        if task.state in _TERMINAL_TASK_STATES:
            return False
        for dep in task.depends_on:
            dep_task = by_id.get(dep)
            if dep_task is None:
                return False
            # Unresolved species deps do not block names-only reaction retrieval
            if dep_task.tool == "annotate_species_candidates" and dep_task.state == TaskState.UNRESOLVED:
                if self.limits.allow_names_only_reactions and task.tool in {
                    "retrieve_reaction_rrf",
                    "select_reaction_grounded",
                }:
                    continue
            if dep_task.state != TaskState.COMPLETED:
                return False
        return True

    def _mark_task(self, plan: AnnotationPlan, task_ids: list[str], state: TaskState, reason: str = "", kind: FailureKind = FailureKind.NONE) -> None:
        for task in plan.tasks:
            if task.task_id in task_ids:
                task.state = state
                task.reason = reason
                task.failure_kind = kind
                est = self.entity_states.setdefault(
                    task.target_id, EntityRuntimeState(target_id=task.target_id)
                )
                est.state = state
                est.failure_kind = kind
                if reason:
                    est.notes.append(reason)

    def _invalidate_downstream(self, plan: AnnotationPlan, species_id: str) -> None:
        """If an upstream identity changes, invalidate dependent reaction tasks."""
        affected_reactions = {
            e.target_id
            for e in self.inventory.entities
            if species_id in e.participants
        }
        for task in plan.tasks:
            if task.target_id in affected_reactions and task.tool in {
                "retrieve_reaction_rrf",
                "select_reaction_grounded",
            }:
                if task.state == TaskState.COMPLETED:
                    task.state = TaskState.PENDING
                    task.reason = f"invalidated after upstream identity change: {species_id}"
                    # Remove fingerprint so recompute is allowed
                    for action in plan.actions:
                        if task.task_id in action.task_ids:
                            self.completed_fingerprints.discard(_action_fingerprint(action))

    def _apply_observation(self, plan: AnnotationPlan, action: AgentAction, obs: ToolObservation) -> Optional[str]:
        """Update states; return revision reason if a plan revision is warranted."""
        if action.tool == "annotate_species_candidates":
            if not obs.ok:
                self._mark_task(
                    plan,
                    action.task_ids,
                    TaskState.FAILED,
                    obs.error or "species annotation failed",
                    FailureKind.TECHNICAL_FAILURE,
                )
                return f"species annotation failed: {obs.error}"
            unresolved = obs.result.get("unresolved_ids") or []
            done = [t for t in action.task_ids if True]
            if unresolved:
                unresolved_tasks = [
                    t.task_id
                    for t in plan.tasks
                    if t.task_id in action.task_ids and t.target_id in unresolved
                ]
                self._mark_task(
                    plan,
                    unresolved_tasks,
                    TaskState.UNRESOLVED,
                    "unresolved or ambiguous species identity",
                    FailureKind.UNRESOLVED_IDENTITY
                    if obs.failure_kind == FailureKind.NONE
                    else obs.failure_kind,
                )
                completed_tasks = [t for t in action.task_ids if t not in unresolved_tasks]
                self._mark_task(plan, completed_tasks, TaskState.COMPLETED)
                for tid in unresolved:
                    # Names-only reactions may continue
                    pass
                for tid in obs.target_ids:
                    if tid not in unresolved:
                        self._invalidate_downstream(plan, tid)
                return None
            self._mark_task(plan, done, TaskState.COMPLETED)
            for tid in obs.target_ids:
                self._invalidate_downstream(plan, tid)
            return None

        if action.tool == "retrieve_reaction_rrf":
            if not obs.ok:
                self._mark_task(
                    plan, action.task_ids, TaskState.FAILED, obs.error or "retrieve failed", FailureKind.TECHNICAL_FAILURE
                )
                return f"retrieval failed: {obs.error}"
            self._mark_task(plan, action.task_ids, TaskState.COMPLETED)
            return None

        if action.tool == "select_reaction_grounded":
            if not obs.ok:
                self._mark_task(
                    plan, action.task_ids, TaskState.FAILED, obs.error or "select failed", FailureKind.TECHNICAL_FAILURE
                )
                return f"grounded selection failed: {obs.error}"
            failed = set(obs.result.get("failed_ids") or [])
            abstained = set(obs.result.get("abstained_ids") or [])
            selected = set(obs.result.get("selected_ids") or [])
            for task in plan.tasks:
                if task.task_id not in action.task_ids:
                    continue
                if task.target_id in selected:
                    task.state = TaskState.COMPLETED
                    self.entity_states[task.target_id].state = TaskState.COMPLETED
                elif task.target_id in abstained:
                    task.state = TaskState.UNRESOLVED
                    task.failure_kind = FailureKind.BIOLOGICAL_ABSTENTION
                    self.entity_states[task.target_id].state = TaskState.UNRESOLVED
                    self.entity_states[task.target_id].failure_kind = FailureKind.BIOLOGICAL_ABSTENTION
                    self.entity_states[task.target_id].notes.append("biological abstention")
                elif task.target_id in failed:
                    task.state = TaskState.FAILED
                    task.failure_kind = FailureKind.TECHNICAL_FAILURE
                    self.entity_states[task.target_id].state = TaskState.FAILED
                    self.entity_states[task.target_id].failure_kind = FailureKind.TECHNICAL_FAILURE
            return None

        if action.tool in {"write_species_annotations", "write_reaction_annotations"}:
            if not obs.ok:
                return f"write failed: {obs.error}"
            self._mark_task(plan, action.task_ids, TaskState.COMPLETED)
            return None

        if action.tool == "export_reports":
            return None

        if not obs.ok:
            self._mark_task(
                plan,
                action.task_ids,
                TaskState.FAILED,
                obs.error or "tool failed",
                FailureKind.TECHNICAL_FAILURE,
            )
            return f"tool failure: {action.tool}: {obs.error}"
        self._mark_task(plan, action.task_ids, TaskState.COMPLETED)
        return None

    def _pending_actions(self, plan: AnnotationPlan) -> list[AgentAction]:
        pending = []
        for action in plan.actions:
            fp = _action_fingerprint(action)
            if fp in self.completed_fingerprints:
                continue
            if action.tool == "export_reports":
                pending.append(action)
                continue
            if action.task_ids and not all(self._task_ready(plan, tid) for tid in action.task_ids):
                continue
            pending.append(action)
        return pending

    def _sync_terminal_tasks_to_entities(self, plan: AnnotationPlan) -> None:
        """Mirror already-terminal plan tasks (e.g. ambiguous unresolved) into the ledger."""
        for task in plan.tasks:
            if task.state not in _TERMINAL_TASK_STATES:
                continue
            est = self.entity_states.setdefault(
                task.target_id, EntityRuntimeState(target_id=task.target_id)
            )
            if est.state in {TaskState.PENDING, TaskState.RUNNING}:
                est.state = task.state
                est.failure_kind = task.failure_kind
                if task.reason:
                    est.notes.append(task.reason)

    def _reconcile_terminal_pending(self, *, interrupted: bool) -> None:
        """Promote leftover PENDING only on final (non-interrupted) shutdown.

        Never demote entities/tasks that are already terminal.
        """
        if self.executed_plan is None:
            return
        if interrupted:
            for task in self.executed_plan.tasks:
                if task.state == TaskState.RUNNING:
                    task.state = TaskState.PENDING
            return
        for task in self.executed_plan.tasks:
            if task.state not in {TaskState.PENDING, TaskState.RUNNING}:
                continue
            est = self.entity_states.get(task.target_id)
            if est is not None and est.state in _TERMINAL_TASK_STATES:
                # Prefer existing entity terminal state; align task without demoting entity.
                task.state = est.state
                task.reason = task.reason or (est.notes[-1] if est.notes else "aligned to entity state")
                continue
            task.state = TaskState.UNRESOLVED
            task.reason = task.reason or self.provenance.get("stop_reason") or "not executed"
            est = self.entity_states.setdefault(
                task.target_id, EntityRuntimeState(target_id=task.target_id)
            )
            if est.state in {TaskState.PENDING, TaskState.RUNNING}:
                est.state = TaskState.UNRESOLVED
                est.notes.append(task.reason)

    def run(self) -> AutoAnnotateResult:
        if not self.inventory.parse_ok:
            report = export_artifacts(
                output_dir=self.output_dir,
                model_file=self.model_file,
                inventory=self.inventory,
                initial_plan=None,
                executed_plan=None,
                entity_states={},
                observations=[],
                species_df=None,
                reaction_df=None,
                provenance=self.provenance,
            )
            report.status = "failed"
            report.technical_errors.append(self.inventory.parse_error or "parse failed")
            report.partial = True
            return AutoAnnotateResult(
                report=report,
                inventory=self.inventory,
                output_dir=str(self.output_dir),
                provenance=self.provenance,
            )

        self._init_entity_states()
        if self._restored_from_checkpoint and self.executed_plan is not None:
            if self.initial_plan is None:
                self.initial_plan = self.executed_plan.model_copy(deep=True)
            pending = self._pending_actions(self.executed_plan)
            self._awaiting_planning_turn = not any(a.tool != "export_reports" for a in pending)
        else:
            self._awaiting_planning_turn = True
            self.executed_plan = AnnotationPlan(scope=self.scope, tasks=[], actions=[])
            self.initial_plan = None

        interrupted = False
        accepted_finish = False

        while True:
            if time.time() - self.started > self.limits.max_runtime_sec:
                interrupted = True
                self.provenance["stop_reason"] = "max_runtime_sec"
                break
            if self.actions_executed >= self.limits.max_actions:
                interrupted = True
                self.provenance["stop_reason"] = "max_actions"
                break
            if (
                self.interrupt_after_actions is not None
                and self.actions_executed >= self.interrupt_after_actions
            ):
                interrupted = True
                self.provenance["stop_reason"] = "interrupt_after_actions"
                break

            assert self.executed_plan is not None
            pending = self._pending_actions(self.executed_plan)
            pending_non_export = [a for a in pending if a.tool != "export_reports"]

            # Exhausting an action group triggers another planning turn unless the
            # scope ledger is already fully terminal (no LLM finish call needed).
            if not pending_non_export:
                leftover = unfinished_scope_targets(
                    self.inventory, self.entity_states, self.scope, self.executed_plan
                )
                if not leftover:
                    accepted_finish = True
                    self.last_planner_decision = "finish"
                    self.provenance["stop_reason"] = self.provenance.get("stop_reason") or "ledger_complete"
                    if self.initial_plan is None:
                        self.initial_plan = self.executed_plan.model_copy(deep=True)
                    self._save_checkpoint()
                    break
                self._awaiting_planning_turn = True

            if self._awaiting_planning_turn:
                if self.planning_turn_count >= self.limits.max_planning_turns:
                    interrupted = True
                    self.provenance["stop_reason"] = "max_planning_turns"
                    break
                try:
                    plan, decision, meta = request_plan(
                        self.backend,
                        request=self.request,
                        inventory=self.inventory,
                        scope=self.scope,
                        limits=self.limits,
                        prior_plan=(
                            self.executed_plan
                            if (
                                self.planning_turn_count
                                or self._restored_from_checkpoint
                                or (self.executed_plan is not None and self.executed_plan.tasks)
                            )
                            else None
                        ),
                        observations=self.observations,
                        revision_reason="",
                        settings=self.provenance["settings"],
                        allow_structural_fallback=self.allow_structural_fallback,
                        entity_states=self.entity_states,
                    )
                except (PlannerRequestError, RuntimeError, PlanValidationError, ValueError) as exc:
                    self.provenance["stop_reason"] = "planner_failure"
                    self.provenance.setdefault("technical_errors", []).append(str(exc))
                    meta_fail = getattr(exc, "meta", None) or {
                        "error": str(exc),
                        "fallback": False,
                        "llm_planning_succeeded": False,
                        "execution_mode": None,
                    }
                    meta_fail.setdefault("llm_planning_succeeded", False)
                    self.provenance["planner_meta"].append(meta_fail)
                    if meta_fail.get("raw_decision"):
                        # Model produced output that failed validation — still LLM attempt.
                        self.provenance["llm_planning_succeeded"] = False
                        self.provenance["execution_mode"] = "llm_planning_invalid"
                    break

                self.planning_turn_count += 1
                self._awaiting_planning_turn = False
                self.last_planner_decision = decision.decision
                self.provenance["planner_meta"].append(meta)
                if meta.get("execution_mode"):
                    self.provenance["execution_mode"] = meta["execution_mode"]
                if meta.get("llm_planning_succeeded") is not None:
                    self.provenance["llm_planning_succeeded"] = meta["llm_planning_succeeded"]
                if meta.get("fallback"):
                    self.provenance["execution_mode"] = "fallback_workflow_orchestration"
                    self.provenance["llm_planning_succeeded"] = False

                if decision.decision == "needs_clarification":
                    self.clarification_prompt = (
                        decision.clarification_prompt or decision.ask_user_prompt or decision.reason
                    )
                    self.provenance["stop_reason"] = "needs_clarification"
                    self.provenance["clarification_prompt"] = self.clarification_prompt
                    self._save_checkpoint()
                    break

                if decision.decision == "finish":
                    problems = validate_finish(
                        self.inventory,
                        self.entity_states,
                        self.scope,
                        self.executed_plan,
                    )
                    if problems:
                        self.provenance.setdefault("technical_errors", []).extend(problems)
                        self.provenance["stop_reason"] = "premature_finish"
                        self._awaiting_planning_turn = True
                        continue
                    accepted_finish = True
                    if self.initial_plan is None:
                        self.initial_plan = self.executed_plan.model_copy(deep=True)
                    self._save_checkpoint()
                    break

                if self.initial_plan is None:
                    self.initial_plan = plan.model_copy(deep=True)
                self.executed_plan = plan
                self._sync_terminal_tasks_to_entities(plan)
                self._save_checkpoint()
                continue

            action = pending_non_export[0]
            fp = _action_fingerprint(action)
            self.action_counts[fp] = self.action_counts.get(fp, 0) + 1
            if self.action_counts[fp] > self.limits.max_retries + 1:
                self.provenance["stop_reason"] = "repeated_equivalent_action"
                self._mark_task(
                    self.executed_plan,
                    action.task_ids,
                    TaskState.FAILED,
                    "repeated equivalent action without new evidence",
                    FailureKind.TECHNICAL_FAILURE,
                )
                self.completed_fingerprints.add(fp)
                self.in_flight_action = None
                self._awaiting_planning_turn = True
                self._save_checkpoint()
                continue

            try:
                validate_action(
                    action,
                    self.inventory,
                    known_task_ids=[t.task_id for t in self.executed_plan.tasks],
                    authorized_annotation_models=self.limits.authorized_annotation_models,
                )
            except PlanValidationError as exc:
                obs = ToolObservation(
                    action_id=action.action_id,
                    tool=action.tool,
                    ok=False,
                    target_ids=action.target_ids,
                    task_ids=action.task_ids,
                    error=str(exc),
                    failure_kind=FailureKind.TECHNICAL_FAILURE,
                )
                self.observations.append(obs)
                self.completed_fingerprints.add(fp)
                self.actions_executed += 1
                self.in_flight_action = None
                self._save_checkpoint()
                continue

            for tid in action.task_ids:
                for task in self.executed_plan.tasks:
                    if task.task_id == tid:
                        task.state = TaskState.RUNNING

            self.in_flight_action = InFlightAction(action=action, fingerprint=fp)
            self._save_checkpoint()
            obs = self.tools.execute(action)
            self.observations.append(obs)
            self.actions_executed += 1
            revision_reason = self._apply_observation(self.executed_plan, action, obs)
            if obs.ok:
                self.completed_fingerprints.add(fp)
            self.in_flight_action = None
            self._save_checkpoint()

            if revision_reason:
                if self.failure_revision_count >= self.limits.max_failure_revisions:
                    self.provenance["stop_reason"] = "max_failure_revisions"
                    continue
                if self.planning_turn_count >= self.limits.max_planning_turns:
                    self.provenance["stop_reason"] = "max_planning_turns"
                    interrupted = True
                    break
                self.failure_revision_count += 1
                self.revision_count = self.failure_revision_count
                try:
                    new_plan, new_decision, new_meta = request_plan(
                        self.backend,
                        request=self.request,
                        inventory=self.inventory,
                        scope=self.scope,
                        limits=self.limits,
                        prior_plan=self.executed_plan,
                        observations=self.observations,
                        revision_reason=revision_reason,
                        settings=self.provenance["settings"],
                        allow_structural_fallback=self.allow_structural_fallback,
                        entity_states=self.entity_states,
                    )
                except (RuntimeError, PlanValidationError, ValueError) as exc:
                    self.provenance["stop_reason"] = "planner_failure"
                    self.provenance.setdefault("technical_errors", []).append(str(exc))
                    break
                self.planning_turn_count += 1
                self.last_planner_decision = new_decision.decision
                self.provenance["planner_meta"].append(new_meta)
                self.provenance["revisions"].append(
                    {
                        "reason": revision_reason,
                        "plan_version": new_plan.plan_version,
                        "kind": "failure_revision",
                    }
                )
                if new_meta.get("execution_mode"):
                    self.provenance["execution_mode"] = new_meta["execution_mode"]
                if new_decision.decision == "needs_clarification":
                    self.clarification_prompt = (
                        new_decision.clarification_prompt
                        or new_decision.ask_user_prompt
                        or new_decision.reason
                    )
                    self.provenance["stop_reason"] = "needs_clarification"
                    self.provenance["clarification_prompt"] = self.clarification_prompt
                    break
                if new_decision.decision == "finish":
                    problems = validate_finish(
                        self.inventory, self.entity_states, self.scope, self.executed_plan
                    )
                    if problems:
                        self.provenance.setdefault("technical_errors", []).extend(problems)
                        self.provenance["stop_reason"] = "premature_finish"
                        self._awaiting_planning_turn = True
                    else:
                        accepted_finish = True
                        break
                else:
                    self.executed_plan = new_plan
                    self._awaiting_planning_turn = False
                self._save_checkpoint()

        self._reconcile_terminal_pending(interrupted=interrupted and not accepted_finish)
        self._save_checkpoint()
        for role_backend in (self.planner, self.inference):
            if role_backend is not None and hasattr(role_backend, "unload"):
                try:
                    role_backend.unload()
                except Exception:
                    pass
        if hasattr(self.planner, "budget"):
            self.provenance["planner_usage"] = self.planner.budget.summary()

        report = export_artifacts(
            output_dir=self.output_dir,
            model_file=self.model_file,
            inventory=self.inventory,
            initial_plan=self.initial_plan,
            executed_plan=self.executed_plan,
            entity_states=self.entity_states,
            observations=self.observations,
            species_df=self.agent_resources.species_recommendations,
            reaction_df=self.agent_resources.reaction_recommendations,
            provenance=self.provenance,
            interrupt=interrupted,
        )
        if self.provenance.get("execution_mode") == "fallback_workflow_orchestration":
            report.recommendations.append(
                "execution_mode=fallback_workflow_orchestration: deterministic "
                "structural plan ran after planner failure; not LLM planning success"
            )
            report.partial = True
        if self.provenance.get("stop_reason") == "needs_clarification":
            report.status = "partial"
            report.partial = True
            prompt = self.clarification_prompt or "user input required"
            report.recommendations.append(f"needs_clarification: {prompt}")
        if self.provenance.get("stop_reason") == "premature_finish":
            report.status = "partial"
            report.partial = True
        for err in self.provenance.get("technical_errors") or []:
            if err not in report.technical_errors:
                report.technical_errors.append(err)
        return AutoAnnotateResult(
            report=report,
            inventory=self.inventory,
            initial_plan=self.initial_plan,
            executed_plan=self.executed_plan,
            entity_states=self.entity_states,
            observations=self.observations,
            output_dir=str(self.output_dir),
            provenance=self.provenance,
        )



def auto_annotate_model(
    model_file: str,
    request: str = "Annotate missing entities in this model",
    backend: Any = None,
    resources: Any = None,
    output_dir: str | Path = "auto_annotate_output",
    *,
    planner: Any = None,
    inference: Any = None,
    reaction_retriever: Any = None,
    scope: str | AnnotationScope = AnnotationScope.ADD_MISSING,
    limits: Optional[ExecutionLimits] = None,
    resume_from: Optional[str | Path] = None,
    species_annotator: Any = None,
    allow_structural_fallback: Optional[bool] = None,
    interrupt_after_actions: Optional[int] = None,
) -> AutoAnnotateResult:
    """Public entry point for the automatic model-annotation agent.

    Prefer ``planner=`` / ``inference=`` for split orchestration vs biological
    annotation. Legacy ``backend=`` alone remains supported for offline MockBackend
    tests (both roles). Conflicting simultaneous legacy/new wiring should be
    resolved by callers before entry (CLI rejects conflicts).
    """
    if backend is None and planner is None and inference is None:
        from core.agent.backends.mock import MockBackend

        backend = MockBackend()
    if isinstance(scope, str):
        scope = AnnotationScope(scope)
    # Real/openai/metabolm planners default to no silent structural fallback.
    if allow_structural_fallback is None:
        role_objs = [o for o in (planner, inference, backend) if o is not None]
        names = [getattr(o, "name", type(o).__name__).lower() for o in role_objs]
        allow_structural_fallback = bool(names) and all(n == "mock" for n in names)
    controller = AnnotationAgentController(
        model_file=model_file,
        request=request,
        backend=backend,
        planner=planner,
        inference=inference,
        output_dir=output_dir,
        resources=resources,
        reaction_retriever=reaction_retriever,
        scope=scope,
        limits=limits,
        resume_from=resume_from,
        species_annotator=species_annotator,
        allow_structural_fallback=allow_structural_fallback,
        interrupt_after_actions=interrupt_after_actions,
    )
    return controller.run()
