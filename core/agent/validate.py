"""Plan, action, and observation validation for the automatic annotation agent.

Layering:
1. **Pydantic schema validation** — structural well-formedness of plans, tool
   arguments, and tool results.
2. **Semantic checks** — target existence, representation/tool compatibility,
   dependency acyclicity, and evidence consistency (including reuse of
   ``core.reaction.grounded.validate_evidence`` / ``GroundedAnnotation``).

Neither layer establishes biological correctness of an identity or selection.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from pydantic import ValidationError

from core.agent.schemas import (
    ALLOWED_TOOLS,
    TOOL_RESULT_MODELS,
    AgentAction,
    AnnotationPlan,
    AnnotationTask,
    EntityRepresentation,
    ExecutionLimits,
    FailureKind,
    ModelInventory,
    RetrieveReactionEvidence,
    SemanticCheckReport,
    SelectReactionResult,
    ToolObservation,
    parse_tool_arguments,
)
from core.reaction import grounded as grounded_mod


_REACTION_TOOLS = {"retrieve_reaction_rrf", "select_reaction_grounded", "write_reaction_annotations"}
_SPECIES_TOOLS = {
    "annotate_species_candidates",
    "assess_identity_support",
    "predict_qualifiers",
    "write_species_annotations",
}
_GLOBAL_TOOLS = {
    "inspect_model",
    "persist_checkpoint",
    "export_reports",
    "classify_or_route_entities",
}


class PlanValidationError(ValueError):
    pass


class ObservationValidationError(ValueError):
    pass


def _topo_problems(tasks: list[AnnotationTask]) -> list[str]:
    problems: list[str] = []
    ids = {t.task_id for t in tasks}
    deps = {t.task_id: list(t.depends_on) for t in tasks}
    for tid, depends in deps.items():
        for d in depends:
            if d not in ids:
                problems.append(f"dependency {d!r} for task {tid!r} does not exist")
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> bool:
        if node in done:
            return True
        if node in visiting:
            return False
        visiting.add(node)
        for d in deps.get(node, []):
            if d in ids and not visit(d):
                return False
        visiting.remove(node)
        done.add(node)
        return True

    for task in tasks:
        if not visit(task.task_id):
            problems.append(f"cyclic dependency involving task {task.task_id!r}")
            break
    return problems


def check_target_existence(
    target_ids: Iterable[str],
    inventory: ModelInventory,
    *,
    tool: str,
) -> list[str]:
    if tool in _GLOBAL_TOOLS and not target_ids:
        return []
    entity_ids = {e.target_id for e in inventory.entities}
    problems = []
    for tid in target_ids:
        if tid not in entity_ids:
            problems.append(f"target_id {tid!r} does not exist in model inventory")
    return problems


def check_representation_support(
    target_ids: Iterable[str],
    inventory: ModelInventory,
    *,
    tool: str,
) -> list[str]:
    by_id = {e.target_id: e for e in inventory.entities}
    problems = []
    for tid in target_ids:
        ent = by_id.get(tid)
        if ent is None:
            continue
        if ent.representation == EntityRepresentation.SBML_QUAL_TRANSITION:
            problems.append(
                f"target {tid!r}: SBML-qual transitions are unsupported for annotation tools"
            )
            continue
        if tool in _REACTION_TOOLS and ent.representation != EntityRepresentation.SBML_REACTION:
            problems.append(
                f"tool {tool} requires sbml_reaction; {tid!r} is {ent.representation.value}"
            )
        if tool in _SPECIES_TOOLS and ent.representation in {
            EntityRepresentation.SBML_REACTION,
            EntityRepresentation.SBML_QUAL_TRANSITION,
        }:
            problems.append(
                f"tool {tool} does not support representation {ent.representation.value} for {tid!r}"
            )
    return problems


def check_dependency_validity(plan: AnnotationPlan) -> list[str]:
    problems = _topo_problems(plan.tasks)
    task_by_id = {t.task_id: t for t in plan.tasks}
    # Reaction retrieve/select should not depend on unrelated non-participant species tasks
    # beyond declared depends_on; enforce that depends_on entries reference known tasks only
    # (already in topo) and that species deps target species tools.
    for task in plan.tasks:
        for dep in task.depends_on:
            parent = task_by_id.get(dep)
            if parent is None:
                continue
            if task.tool in {"retrieve_reaction_rrf", "select_reaction_grounded"}:
                if parent.tool not in {
                    "annotate_species_candidates",
                    "assess_identity_support",
                    "predict_qualifiers",
                    "retrieve_reaction_rrf",
                }:
                    problems.append(
                        f"task {task.task_id!r} has invalid dependency tool {parent.tool!r}"
                    )
        if task.tool == "select_reaction_grounded":
            retrieve_deps = [
                task_by_id[d]
                for d in task.depends_on
                if d in task_by_id and task_by_id[d].tool == "retrieve_reaction_rrf"
            ]
            if not any(d.target_id == task.target_id for d in retrieve_deps):
                problems.append(
                    f"select task {task.task_id!r} must depend on retrieve_reaction_rrf "
                    f"for the same reaction target"
                )
    return problems


def check_evidence_consistency(
    *,
    tool: str,
    target_ids: list[str],
    result: dict[str, Any],
    evidence: dict[str, Any],
    grounded_answer: Any = None,
) -> list[str]:
    """Structural/evidence alignment checks. Not biological correctness."""
    problems: list[str] = []
    if tool == "retrieve_reaction_rrf":
        try:
            ev = RetrieveReactionEvidence.model_validate(evidence)
        except ValidationError as exc:
            return [f"retrieval evidence schema invalid: {exc}"]
        returned = set(result.get("reaction_ids") or [])
        artifact_ids = {a.reaction_id for a in ev.artifacts}
        if returned != artifact_ids:
            problems.append(
                f"retrieval result reaction_ids {sorted(returned)} != evidence artifacts {sorted(artifact_ids)}"
            )
        for tid in target_ids:
            if tid not in artifact_ids:
                problems.append(f"requested target {tid!r} missing from retrieval evidence")
        for art in ev.artifacts:
            try:
                grounded_mod.validate_evidence(art.evidence)
            except ValueError as exc:
                problems.append(f"evidence for {art.reaction_id!r} failed grounded validate_evidence: {exc}")
            if not art.query.strip():
                problems.append(f"empty retrieval query for {art.reaction_id!r}")

    elif tool == "select_reaction_grounded":
        try:
            parsed = SelectReactionResult.model_validate(result)
        except ValidationError as exc:
            return [f"selection result schema invalid: {exc}"]
        row_ids = {r.id for r in parsed.recommendations}
        for tid in target_ids:
            if tid not in row_ids:
                problems.append(f"selection missing recommendation row for target {tid!r}")
        disjoint = (
            set(parsed.selected_ids) & set(parsed.abstained_ids),
            set(parsed.selected_ids) & set(parsed.failed_ids),
            set(parsed.abstained_ids) & set(parsed.failed_ids),
        )
        if any(disjoint):
            problems.append("selection status sets are not disjoint")
        for row in parsed.recommendations:
            if row.status == "selected":
                if row.biological_abstain is True:
                    problems.append(f"{row.id}: selected row cannot be biological_abstain=True")
                if not str(row.annotation).startswith("KEGG:R"):
                    problems.append(f"{row.id}: selected annotation must be KEGG:R#####")
                if row.update_annotation != "add":
                    problems.append(f"{row.id}: selected row should have update_annotation='add'")
            if row.status == "abstained":
                if row.biological_abstain is False:
                    problems.append(f"{row.id}: abstained row must not set biological_abstain=False")
                if row.annotation:
                    problems.append(f"{row.id}: abstained row must not carry an annotation")
            if row.status in {
                "provider_error",
                "schema_invalid",
                "incomplete",
                "empty_response",
                "evidence_noncompliant",
            }:
                if row.biological_abstain is True:
                    problems.append(
                        f"{row.id}: technical/schema failure must not be promoted to biological abstention"
                    )
        if grounded_answer is not None:
            # Reuse GroundedAnnotation schema; semantic evidence rules already applied upstream.
            try:
                grounded_mod.GroundedAnnotation.model_validate(
                    grounded_answer.model_dump() if hasattr(grounded_answer, "model_dump") else grounded_answer
                )
            except ValidationError as exc:
                problems.append(f"grounded_answer failed GroundedAnnotation schema: {exc}")

    elif tool == "annotate_species_candidates":
        recs = result.get("recommendations") or []
        rec_ids = {str(r.get("id")) for r in recs if isinstance(r, dict)}
        for tid in target_ids:
            if tid not in rec_ids and tid not in set(result.get("unresolved_ids") or []):
                # Allow unresolved-only rows that still appear in recommendations
                if tid not in rec_ids:
                    problems.append(f"species result missing row for target {tid!r}")
        for rid in result.get("unresolved_ids") or []:
            if rid not in rec_ids:
                problems.append(f"unresolved_id {rid!r} missing from recommendations")

    return problems


def validate_action(
    action: AgentAction,
    inventory: ModelInventory,
    *,
    known_task_ids: Iterable[str] | None = None,
) -> SemanticCheckReport:
    checks = ["allowlist", "argument_schema", "target_existence", "representation_support"]
    problems: list[str] = []
    if action.tool not in ALLOWED_TOOLS:
        problems.append(f"tool not allowlisted: {action.tool}")
    try:
        parse_tool_arguments(action.tool, action.arguments)
    except (ValidationError, ValueError) as exc:
        problems.append(f"argument schema invalid for {action.tool}: {exc}")
    problems.extend(check_target_existence(action.target_ids, inventory, tool=action.tool))
    problems.extend(check_representation_support(action.target_ids, inventory, tool=action.tool))
    if known_task_ids is not None:
        checks.append("task_id_existence")
        known = set(known_task_ids)
        for task_id in action.task_ids:
            if task_id not in known:
                problems.append(f"unknown task_id: {task_id}")
    report = SemanticCheckReport(ok=not problems, problems=problems, checks_performed=checks)
    if problems:
        raise PlanValidationError("; ".join(problems))
    return report


def validate_plan(
    plan: AnnotationPlan,
    inventory: ModelInventory,
    limits: ExecutionLimits | None = None,
) -> AnnotationPlan:
    limits = limits or ExecutionLimits()
    if not inventory.parse_ok:
        raise PlanValidationError(f"cannot plan for unreadable model: {inventory.parse_error}")
    checks = [
        "inventory_parse_ok",
        "unique_task_ids",
        "target_existence",
        "argument_schema",
        "representation_support",
        "dependency_validity",
        "action_semantic_checks",
    ]
    problems: list[str] = []
    task_ids = [t.task_id for t in plan.tasks]
    if len(task_ids) != len(set(task_ids)):
        problems.append("duplicate task_id")
    entity_ids = {e.target_id for e in inventory.entities}
    for task in plan.tasks:
        if task.target_id not in entity_ids:
            problems.append(f"task target missing: {task.target_id}")
        if task.tool not in ALLOWED_TOOLS:
            problems.append(f"task tool not allowlisted: {task.tool}")
        try:
            parse_tool_arguments(task.tool, task.arguments)
        except (ValidationError, ValueError) as exc:
            problems.append(f"task {task.task_id} argument schema invalid: {exc}")
        problems.extend(
            check_representation_support([task.target_id], inventory, tool=task.tool)
        )
    problems.extend(check_dependency_validity(plan))
    for action in plan.actions:
        try:
            validate_action(action, inventory, known_task_ids=task_ids)
        except PlanValidationError as exc:
            problems.append(str(exc))
    semantic = SemanticCheckReport(
        ok=not problems,
        problems=problems,
        checks_performed=checks,
    )
    plan.semantic_checks = semantic
    if problems:
        raise PlanValidationError("; ".join(problems))
    plan.limits = {
        "max_actions": limits.max_actions,
        "max_revisions": limits.max_revisions,
        "max_retries": limits.max_retries,
        "max_runtime_sec": limits.max_runtime_sec,
        "allow_names_only_reactions": limits.allow_names_only_reactions,
    }
    return plan


def validate_observation(
    observation: ToolObservation,
    inventory: ModelInventory,
    *,
    expected_target_ids: Optional[list[str]] = None,
) -> ToolObservation:
    """Validate tool observation schema + semantic alignment.

    A passing report never sets ``biological_correctness_certified``.
    """
    checks = ["observation_schema", "result_schema", "target_alignment", "evidence_consistency"]
    problems: list[str] = []
    # Result schema (tool-specific). inspect_model historically dumps inventory fields flat.
    result_model = TOOL_RESULT_MODELS.get(observation.tool)
    if result_model is not None and observation.ok:
        try:
            if observation.tool == "inspect_model":
                ModelInventory.model_validate(observation.result)
            else:
                result_model.model_validate(observation.result)
        except ValidationError as exc:
            problems.append(f"result schema invalid: {exc}")

    targets = expected_target_ids if expected_target_ids is not None else list(observation.target_ids)
    if set(observation.target_ids) != set(targets) and observation.tool not in _GLOBAL_TOOLS:
        problems.append(
            f"observation targets {observation.target_ids} != expected {targets}"
        )
    problems.extend(check_target_existence(observation.target_ids, inventory, tool=observation.tool))
    if observation.ok:
        problems.extend(
            check_evidence_consistency(
                tool=observation.tool,
                target_ids=list(observation.target_ids),
                result=observation.result,
                evidence=observation.evidence,
                grounded_answer=observation.grounded_answer,
            )
        )
    # Never allow technical failures to be labeled biological abstention at observation level
    if (
        not observation.ok
        and observation.failure_kind == FailureKind.BIOLOGICAL_ABSTENTION
    ):
        problems.append("failed observation cannot use failure_kind=biological_abstention")

    report = SemanticCheckReport(ok=not problems, problems=problems, checks_performed=checks)
    observation.semantic_checks = report
    if problems:
        raise ObservationValidationError("; ".join(problems))
    return observation
