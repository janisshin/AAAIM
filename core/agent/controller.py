"""Inspect → plan → execute → check → revise → export controller."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from core.agent.export import export_artifacts
from core.agent.inspect_model import entities_needing_annotation, inspect_sbml_model
from core.agent.planner import request_plan
from core.agent.schemas import (
    AgentAction,
    AnnotationPlan,
    AnnotationScope,
    AnnotationStatus,
    AutoAnnotateResult,
    EntityRuntimeState,
    ExecutionLimits,
    FailureKind,
    TaskState,
    ToolObservation,
)
from core.agent.tools import AgentResources, ToolRegistry
from core.agent.validate import PlanValidationError, validate_action


def _action_fingerprint(action: AgentAction) -> str:
    payload = {
        "tool": action.tool,
        "target_ids": list(action.target_ids),
        "task_ids": list(action.task_ids),
        "arguments": action.arguments,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_identity(model_sha: str, backend_id: dict[str, Any], settings: dict[str, Any]) -> str:
    blob = json.dumps(
        {"model_sha256": model_sha, "backend": backend_id, "settings": settings},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class AnnotationAgentController:
    def __init__(
        self,
        *,
        model_file: str,
        request: str,
        backend: Any,
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
        self.backend = backend
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
        self.agent_resources = AgentResources(
            model_file=self.model_file,
            output_dir=self.output_dir,
            inventory=self.inventory,
            reaction_resources=resources,
            reaction_retriever=reaction_retriever,
            grounded_transport=getattr(backend, "grounded_transport", None),
            species_annotator=species_annotator,
            allow_names_only_reactions=self.limits.allow_names_only_reactions,
        )
        self.tools = ToolRegistry(self.agent_resources)
        self.observations: list[ToolObservation] = []
        self.entity_states: dict[str, EntityRuntimeState] = {}
        self.completed_fingerprints: set[str] = set()
        self.action_counts: dict[str, int] = {}
        self.initial_plan: Optional[AnnotationPlan] = None
        self.executed_plan: Optional[AnnotationPlan] = None
        self.revision_count = 0
        self.actions_executed = 0
        self.started = time.time()
        self.provenance: dict[str, Any] = {
            "backend": backend.identity() if hasattr(backend, "identity") else {"backend": type(backend).__name__},
            "settings": {
                "scope": scope.value,
                "limits": self.limits.model_dump(mode="json"),
                "request": request,
            },
            "planner_meta": [],
            "revisions": [],
        }
        self.provenance["cache_identity"] = _cache_identity(
            self.inventory.sha256,
            self.provenance["backend"],
            self.provenance["settings"],
        )
        if self.resume_from:
            self._restore_checkpoint(self.resume_from)

    def _restore_checkpoint(self, path: Path) -> None:
        data = _load_checkpoint(path / "checkpoint.json" if path.is_dir() else path)
        if not data:
            return
        saved_identity = data.get("cache_identity")
        if saved_identity and saved_identity != self.provenance["cache_identity"]:
            raise RuntimeError(
                "checkpoint cache identity incompatible with current settings/backend/model"
            )
        self.agent_resources.entity_identities = data.get("entity_identities") or {}
        self.agent_resources.checkpoint = data.get("checkpoint") or {}
        self.completed_fingerprints = set(data.get("completed_fingerprints") or [])
        self.actions_executed = int(data.get("actions_executed") or 0)
        if data.get("species_recommendations"):
            import pandas as pd

            self.agent_resources.species_recommendations = pd.DataFrame(
                data["species_recommendations"]
            )
        if data.get("reaction_recommendations"):
            import pandas as pd

            self.agent_resources.reaction_recommendations = pd.DataFrame(
                data["reaction_recommendations"]
            )
        for tid, payload in (data.get("entity_states") or {}).items():
            self.entity_states[tid] = EntityRuntimeState.model_validate(payload)

    def _save_checkpoint(self) -> None:
        payload = {
            "cache_identity": self.provenance["cache_identity"],
            "entity_identities": self.agent_resources.entity_identities,
            "checkpoint": self.agent_resources.checkpoint,
            "completed_fingerprints": sorted(self.completed_fingerprints),
            "actions_executed": self.actions_executed,
            "entity_states": {k: v.model_dump(mode="json") for k, v in self.entity_states.items()},
            "species_recommendations": None
            if self.agent_resources.species_recommendations is None
            else self.agent_resources.species_recommendations.fillna("").to_dict("records"),
            "reaction_recommendations": None
            if self.agent_resources.reaction_recommendations is None
            else self.agent_resources.reaction_recommendations.fillna("").to_dict("records"),
        }
        (self.output_dir / "checkpoint.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

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
        if task.state in {TaskState.COMPLETED, TaskState.SKIPPED, TaskState.FAILED, TaskState.UNRESOLVED}:
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
            # If all associated entities are already terminal without this action, skip
            pending.append(action)
        return pending

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
        plan, decision, meta = request_plan(
            self.backend,
            request=self.request,
            inventory=self.inventory,
            scope=self.scope,
            limits=self.limits,
            settings=self.provenance["settings"],
            allow_structural_fallback=self.allow_structural_fallback,
        )
        self.provenance["planner_meta"].append(meta)
        self.initial_plan = plan.model_copy(deep=True)
        self.executed_plan = plan
        interrupted = False

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

            pending = self._pending_actions(self.executed_plan)
            # Prefer non-export actions first
            pending_non_export = [a for a in pending if a.tool != "export_reports"]
            if not pending_non_export:
                # All work done or blocked → export
                export_action = next((a for a in pending if a.tool == "export_reports"), None)
                if export_action is not None:
                    fp = _action_fingerprint(export_action)
                    if fp not in self.completed_fingerprints:
                        obs = self.tools.execute(export_action)
                        self.observations.append(obs)
                        self.completed_fingerprints.add(fp)
                        self.actions_executed += 1
                break

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
                # Drop this action from the plan to avoid infinite loop
                self.completed_fingerprints.add(fp)
                continue

            try:
                validate_action(
                    action,
                    self.inventory,
                    known_task_ids=[t.task_id for t in self.executed_plan.tasks],
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
                continue

            for tid in action.task_ids:
                for task in self.executed_plan.tasks:
                    if task.task_id == tid:
                        task.state = TaskState.RUNNING
            obs = self.tools.execute(action)
            self.observations.append(obs)
            self.actions_executed += 1
            revision_reason = self._apply_observation(self.executed_plan, action, obs)
            if obs.ok:
                self.completed_fingerprints.add(fp)
            self._save_checkpoint()

            if revision_reason and self.revision_count < self.limits.max_revisions:
                self.revision_count += 1
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
                )
                self.provenance["planner_meta"].append(new_meta)
                self.provenance["revisions"].append(
                    {"reason": revision_reason, "plan_version": new_plan.plan_version}
                )
                # Preserve completed task states where task ids match
                old_states = {t.task_id: t for t in self.executed_plan.tasks}
                for task in new_plan.tasks:
                    if task.task_id in old_states and old_states[task.task_id].state == TaskState.COMPLETED:
                        task.state = TaskState.COMPLETED
                self.executed_plan = new_plan
            elif revision_reason and self.revision_count >= self.limits.max_revisions:
                self.provenance["stop_reason"] = "max_revisions"

        # Mark leftover pending reaction/species tasks that never ran
        for task in self.executed_plan.tasks:
            if task.state == TaskState.PENDING:
                task.state = TaskState.UNRESOLVED
                task.reason = task.reason or self.provenance.get("stop_reason") or "not executed"
                est = self.entity_states.setdefault(
                    task.target_id, EntityRuntimeState(target_id=task.target_id)
                )
                if est.state == TaskState.PENDING:
                    est.state = TaskState.UNRESOLVED
                    est.notes.append(task.reason)

        self._save_checkpoint()
        # Unload GPU backend before any potential encoder use elsewhere
        if hasattr(self.backend, "unload"):
            try:
                self.backend.unload()
            except Exception:
                pass

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
    reaction_retriever: Any = None,
    scope: str | AnnotationScope = AnnotationScope.ADD_MISSING,
    limits: Optional[ExecutionLimits] = None,
    resume_from: Optional[str | Path] = None,
    species_annotator: Any = None,
    allow_structural_fallback: bool = True,
    interrupt_after_actions: Optional[int] = None,
) -> AutoAnnotateResult:
    """Public entry point for the automatic model-annotation agent."""
    if backend is None:
        from core.agent.backends.mock import MockBackend

        backend = MockBackend()
    if isinstance(scope, str):
        scope = AnnotationScope(scope)
    controller = AnnotationAgentController(
        model_file=model_file,
        request=request,
        backend=backend,
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
