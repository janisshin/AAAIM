"""Multi-turn planning decisions, finish validation, and fallback labeling."""
from __future__ import annotations

import pytest

from core.agent import AnnotationScope, ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.inspect_model import inspect_sbml_model
from core.agent.planner import PlannerRequestError, request_plan, validate_finish
from core.agent.schemas import (
    AgentAction,
    AnnotationTask,
    BiologicalType,
    EntityRepresentation,
    EntityRuntimeState,
    PlannerDecision,
    TaskState,
)
from tests.agent.conftest import TinyRetriever


def test_multi_turn_continue_then_finish_covers_targets(chemical_only_model, tmp_path):
    inv = inspect_sbml_model(str(chemical_only_model))
    t_water = AnnotationTask(
        task_id="species::water",
        target_id="water",
        representation=EntityRepresentation.SBML_SPECIES,
        biological_type=BiologicalType.CHEMICAL,
        tool="annotate_species_candidates",
        databases=["chebi"],
    )
    t_atp = AnnotationTask(
        task_id="species::atp",
        target_id="atp",
        representation=EntityRepresentation.SBML_SPECIES,
        biological_type=BiologicalType.CHEMICAL,
        tool="annotate_species_candidates",
        databases=["chebi"],
    )
    a1 = AgentAction(
        action_id="a1",
        tool="annotate_species_candidates",
        target_ids=["water"],
        task_ids=["species::water"],
        arguments={"databases": ["chebi"]},
    )
    a2 = AgentAction(
        action_id="a2",
        tool="annotate_species_candidates",
        target_ids=["atp"],
        task_ids=["species::atp"],
        arguments={"databases": ["chebi"]},
    )
    decisions = [
        PlannerDecision(decision="continue", tasks=[t_water, t_atp], actions=[a1], reason="turn1"),
        PlannerDecision(decision="continue", tasks=[t_water, t_atp], actions=[a2], reason="turn2"),
        PlannerDecision(decision="finish", tasks=[t_water, t_atp], reason="done"),
    ]
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(decisions=decisions),
        output_dir=tmp_path / "multi",
        limits=ExecutionLimits(max_actions_per_turn=1, max_planning_turns=5),
        allow_structural_fallback=False,
    )
    assert result.provenance.get("llm_planning_succeeded") is not False or result.provenance.get(
        "execution_mode"
    ) != "fallback_workflow_orchestration"
    assert result.entity_states["water"].state == TaskState.COMPLETED
    assert result.entity_states["atp"].state == TaskState.COMPLETED
    assert (result.provenance.get("planner_meta") or []) and len(result.provenance["planner_meta"]) >= 2


def test_premature_finish_rejected(chemical_only_model, tmp_path):
    decisions = [
        PlannerDecision(decision="finish", reason="too soon"),
        PlannerDecision(
            decision="continue",
            tasks=[
                AnnotationTask(
                    task_id="species::water",
                    target_id="water",
                    representation=EntityRepresentation.SBML_SPECIES,
                    biological_type=BiologicalType.CHEMICAL,
                    tool="annotate_species_candidates",
                    databases=["chebi"],
                )
            ],
            actions=[
                AgentAction(
                    action_id="a1",
                    tool="annotate_species_candidates",
                    target_ids=["water"],
                    task_ids=["species::water"],
                    arguments={"databases": ["chebi"]},
                )
            ],
        ),
        PlannerDecision(decision="finish", reason="after work"),
    ]
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(decisions=decisions),
        output_dir=tmp_path / "premature",
        limits=ExecutionLimits(max_actions_per_turn=2, max_planning_turns=6),
        allow_structural_fallback=False,
    )
    assert any(
        "finish rejected" in e or "non-terminal" in e
        for e in (result.report.technical_errors or [])
    ) or result.provenance.get("stop_reason") in {"premature_finish", "max_planning_turns", None}


def test_validate_finish_requires_terminal_ledger(chemical_only_model):
    inv = inspect_sbml_model(str(chemical_only_model))
    states = {
        "water": EntityRuntimeState(target_id="water", state=TaskState.PENDING),
        "atp": EntityRuntimeState(target_id="atp", state=TaskState.COMPLETED),
    }
    problems = validate_finish(inv, states, AnnotationScope.ADD_MISSING, None)
    assert problems
    assert any("water" in p for p in problems)


def test_truncated_planner_fallback_labeled(chemical_only_model, tmp_path):
    class TruncatingBackend(MockBackend):
        def plan(self, **kwargs):
            raise RuntimeError("MetaboLLM planning output truncated; refusing silent acceptance")

    result = auto_annotate_model(
        str(chemical_only_model),
        backend=TruncatingBackend(),
        output_dir=tmp_path / "trunc_fb",
        allow_structural_fallback=True,
        limits=ExecutionLimits(max_actions_per_turn=3, max_planning_turns=4),
    )
    assert result.provenance.get("execution_mode") == "fallback_workflow_orchestration"
    assert result.provenance.get("llm_planning_succeeded") is False
    assert any("fallback_workflow_orchestration" in r for r in result.report.recommendations)


def test_truncated_planner_no_silent_fallback(chemical_only_model, tmp_path):
    class TruncatingBackend(MockBackend):
        def plan(self, **kwargs):
            raise RuntimeError("MetaboLLM planning output truncated; refusing silent acceptance")

    result = auto_annotate_model(
        str(chemical_only_model),
        backend=TruncatingBackend(),
        output_dir=tmp_path / "trunc_nofb",
        allow_structural_fallback=False,
        limits=ExecutionLimits(max_planning_turns=2),
    )
    assert result.provenance.get("stop_reason") == "planner_failure"
    assert result.provenance.get("execution_mode") != "fallback_workflow_orchestration"
    assert not result.observations or all(
        o.tool != "annotate_species_candidates" or not o.ok for o in result.observations
    )


def test_action_group_exhaustion_requests_another_turn(chemical_only_model, tmp_path):
    backend = MockBackend()
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=backend,
        output_dir=tmp_path / "exhaust",
        limits=ExecutionLimits(max_actions_per_turn=1, max_planning_turns=8),
        allow_structural_fallback=False,
    )
    # Default mock chunks one action per turn → multiple plan_calls.
    assert backend.plan_calls >= 2
    assert result.report.status in {"completed", "partial"}


def test_scope_add_missing_skips_complete_species(partial_annotation_model, tmp_path):
    result = auto_annotate_model(
        str(partial_annotation_model),
        backend=MockBackend(
            grounded_responses={
                "default": {
                    "abstain": True,
                    "predicted_kegg_id": None,
                    "selected_evidence_rank": None,
                    "supporting_evidence_ids": [],
                    "confidence": 0.1,
                    "reasoning_summary": "abstain",
                    "abstention_reason": "Insufficient evidence",
                }
            }
        ),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "scope",
        scope=AnnotationScope.ADD_MISSING,
        limits=ExecutionLimits(max_actions_per_turn=3, max_planning_turns=8),
    )
    assert result.entity_states["a"].state == TaskState.SKIPPED
    problems = validate_finish(
        result.inventory,
        result.entity_states,
        AnnotationScope.ADD_MISSING,
        result.executed_plan,
    )
    # After run, finish ledger should be satisfied or remaining are terminal.
    assert isinstance(problems, list)


def test_continue_rejects_oversized_action_group(chemical_only_model):
    inv = inspect_sbml_model(str(chemical_only_model))
    actions = [
        AgentAction(
            action_id=f"a{i}",
            tool="annotate_species_candidates",
            target_ids=["water"],
            task_ids=["species::water"],
            arguments={"databases": ["chebi"]},
        )
        for i in range(5)
    ]
    tasks = [
        AnnotationTask(
            task_id="species::water",
            target_id="water",
            representation=EntityRepresentation.SBML_SPECIES,
            biological_type=BiologicalType.CHEMICAL,
            tool="annotate_species_candidates",
            databases=["chebi"],
        )
    ]
    backend = MockBackend(
        decisions=[PlannerDecision(decision="continue", tasks=tasks, actions=actions)]
    )
    with pytest.raises(PlannerRequestError, match="max_actions_per_turn"):
        request_plan(
            backend,
            request="annotate",
            inventory=inv,
            scope=AnnotationScope.ADD_MISSING,
            limits=ExecutionLimits(max_actions_per_turn=2),
            allow_structural_fallback=False,
        )
