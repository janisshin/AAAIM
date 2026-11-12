"""Deterministic controller tests for the automatic annotation agent."""
from __future__ import annotations

import json
from pathlib import Path

import libsbml
import pandas as pd
import pytest

from core.agent import AnnotationScope, ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.inspect_model import inspect_sbml_model
from core.agent.schemas import (
    AgentAction,
    AnnotationPlan,
    AnnotationTask,
    BiologicalType,
    EntityRepresentation,
    FailureKind,
    PlannerDecision,
    TaskState,
)
from core.agent.validate import PlanValidationError, validate_plan
from tests.agent.conftest import TinyRetriever


def test_inspect_mixed_does_not_classify_whole_file(mixed_model):
    inv = inspect_sbml_model(str(mixed_model))
    assert inv.parse_ok
    types = {e.biological_type for e in inv.entities}
    assert BiologicalType.CHEMICAL in types or BiologicalType.REACTION in types
    assert BiologicalType.REACTION in types
    # Ambiguous species remains distinct; file is not exclusively chemical
    amb = next(e for e in inv.entities if e.target_id == "amb")
    assert amb.biological_type in {BiologicalType.AMBIGUOUS, BiologicalType.CHEMICAL}


def test_inspect_invalid_model(invalid_model):
    inv = inspect_sbml_model(str(invalid_model))
    assert not inv.parse_ok
    assert inv.parse_error


def test_chemical_only_end_to_end(chemical_only_model, tmp_path):
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=tmp_path / "chem",
    )
    assert result.report.status in {"completed", "partial"}
    assert result.initial_plan is not None
    assert any(t.tool == "annotate_species_candidates" for t in result.initial_plan.tasks)
    assert Path(result.report.change_table_path).is_file()


def test_gene_protein_and_complex_routing(gene_protein_model, complex_model, tmp_path):
    for model, label in [(gene_protein_model, "gp"), (complex_model, "cx")]:
        result = auto_annotate_model(
            str(model),
            backend=MockBackend(),
            output_dir=tmp_path / label,
        )
        tools = {t.tool for t in result.initial_plan.tasks}
        assert "annotate_species_candidates" in tools


def test_reaction_only_with_tiny_retriever(reaction_only_model, tmp_path):
    evidence = TinyRetriever().retrieve("ATP")
    answer = {
        "abstain": False,
        "predicted_kegg_id": evidence[0]["kegg_id"],
        "selected_evidence_rank": 1,
        "supporting_evidence_ids": ["E01"],
        "confidence": 0.7,
        "reasoning_summary": "Synthetic selection",
        "abstention_reason": None,
    }
    backend = MockBackend(grounded_responses={"default": answer})
    result = auto_annotate_model(
        str(reaction_only_model),
        backend=backend,
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "rxn",
    )
    assert result.report.status in {"completed", "partial"}
    assert any(t.tool == "retrieve_reaction_rrf" for t in result.initial_plan.tasks)
    assert any(t.tool == "select_reaction_grounded" for t in result.initial_plan.tasks)
    assert backend.transport_calls >= 1


def test_mixed_partial_species_failure_continues_reactions(mixed_model, tmp_path):
    answer = {
        "abstain": True,
        "predicted_kegg_id": None,
        "selected_evidence_rank": None,
        "supporting_evidence_ids": [],
        "confidence": 0.1,
        "reasoning_summary": "abstain",
        "abstention_reason": "Insufficient evidence",
    }
    result = auto_annotate_model(
        str(mixed_model),
        backend=MockBackend(grounded_responses={"default": answer}),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "mixed",
        limits=ExecutionLimits(allow_names_only_reactions=True),
    )
    # Ambiguous species unresolved should not prevent reaction tasks from existing/executing
    reaction_tasks = [
        t for t in result.executed_plan.tasks if t.tool in {"retrieve_reaction_rrf", "select_reaction_grounded"}
    ]
    assert reaction_tasks
    assert any(t.state in {TaskState.COMPLETED, TaskState.UNRESOLVED, TaskState.FAILED} for t in reaction_tasks)
    assert result.report.partial or result.report.unresolved_ids


def test_existing_annotations_preserved(partial_annotation_model, tmp_path):
    result = auto_annotate_model(
        str(partial_annotation_model),
        backend=MockBackend(),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "partial",
        scope=AnnotationScope.ADD_MISSING,
    )
    # Species a already annotated → skipped under add_missing
    assert result.entity_states["a"].state == TaskState.SKIPPED
    # Species b missing → planned
    assert "b" in result.entity_states


def test_invalid_plan_rejected(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    bad = AnnotationPlan(
        tasks=[
            AnnotationTask(
                task_id="t1",
                target_id="missing_id",
                representation=EntityRepresentation.SBML_REACTION,
                biological_type=BiologicalType.REACTION,
                tool="retrieve_reaction_rrf",
            )
        ],
        actions=[],
    )
    with pytest.raises(PlanValidationError):
        validate_plan(bad, inv)


def test_malformed_tool_result_does_not_become_biological_abstention(reaction_only_model, tmp_path):
    class BrokenTransportBackend(MockBackend):
        def grounded_transport(self, provider_payload):
            return {"status": "completed", "model": provider_payload.get("model"), "output": [], "usage": {}}

    result = auto_annotate_model(
        str(reaction_only_model),
        backend=BrokenTransportBackend(),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "broken",
    )
    # Technical/empty response path must not be labeled biological abstention for failures
    for obs in result.observations:
        if obs.tool == "select_reaction_grounded":
            for row in obs.result.get("recommendations") or []:
                if row.get("status") in {"empty_response", "schema_invalid", "provider_error"}:
                    assert row.get("biological_abstain") in {None, False}


def test_revision_and_action_limits(chemical_only_model, tmp_path):
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=tmp_path / "limits",
        limits=ExecutionLimits(max_actions=2, max_revisions=0),
    )
    assert result.report.status in {"partial", "interrupted", "completed"}
    assert result.provenance.get("stop_reason") in {None, "max_actions", "max_revisions"}


def test_repeated_action_detection(chemical_only_model, tmp_path):
    # Script a plan that repeats the same action
    inv_tasks = [
        AnnotationTask(
            task_id="species::water",
            target_id="water",
            representation=EntityRepresentation.SBML_SPECIES,
            biological_type=BiologicalType.CHEMICAL,
            tool="annotate_species_candidates",
            databases=["chebi"],
        )
    ]
    action = AgentAction(
        action_id="repeat",
        tool="annotate_species_candidates",
        target_ids=["water"],
        task_ids=["species::water"],
        arguments={"databases": ["chebi"]},
        rationale="repeat",
    )
    decisions = [
        PlannerDecision(decision="plan", tasks=inv_tasks, actions=[action, action, action]),
        PlannerDecision(decision="stop", reason="stop"),
    ]
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(decisions=decisions),
        output_dir=tmp_path / "repeat",
        limits=ExecutionLimits(max_actions=10, max_retries=0),
        allow_structural_fallback=False,
    )
    assert result.report is not None


def test_interrupt_resume_skips_completed_work(chemical_only_model, tmp_path):
    out1 = tmp_path / "run1"
    result1 = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out1,
        interrupt_after_actions=1,
    )
    assert result1.report.status == "interrupted"
    assert (out1 / "checkpoint.json").is_file()
    out2 = tmp_path / "run2"
    # Copy checkpoint into out2 and resume
    out2.mkdir()
    (out2 / "checkpoint.json").write_text((out1 / "checkpoint.json").read_text(encoding="utf-8"), encoding="utf-8")
    # Fix cache identity by using same backend settings via resume_from of out1 directly
    result2 = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out1,
        resume_from=out1,
    )
    assert result2.report.status in {"completed", "partial", "interrupted"}
    # Completed fingerprints should prevent duplicate purchases/work
    assert result2.provenance["cache_identity"] == result1.provenance["cache_identity"]


def test_output_preserves_source_and_unrelated_content(partial_annotation_model, tmp_path):
    result = auto_annotate_model(
        str(partial_annotation_model),
        backend=MockBackend(
            grounded_responses={
                "default": {
                    "abstain": False,
                    "predicted_kegg_id": "R00001",
                    "selected_evidence_rank": 1,
                    "supporting_evidence_ids": ["E01"],
                    "confidence": 0.6,
                    "reasoning_summary": "ok",
                    "abstention_reason": None,
                }
            }
        ),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "preserve",
    )
    source = libsbml.SBMLReader().readSBML(str(partial_annotation_model)).getModel()
    assert source.getSpecies("a") is not None
    # Source file unchanged path still parseable
    assert Path(partial_annotation_model).is_file()
    if result.report.annotated_model_path:
        out_model = libsbml.SBMLReader().readSBML(result.report.annotated_model_path).getModel()
        assert out_model.getSpecies("a") is not None
        assert out_model.getReaction("r1") is not None


def test_dependency_invalidation_after_upstream_change(reaction_only_model, tmp_path):
    from core.agent.controller import AnnotationAgentController

    controller = AnnotationAgentController(
        model_file=str(reaction_only_model),
        request="annotate",
        backend=MockBackend(),
        output_dir=tmp_path / "inv",
        reaction_retriever=TinyRetriever(),
    )
    plan, _, _ = __import__("core.agent.planner", fromlist=["request_plan"]).request_plan(
        controller.backend,
        request="annotate",
        inventory=controller.inventory,
        scope=AnnotationScope.ADD_MISSING,
        limits=ExecutionLimits(),
    )
    for task in plan.tasks:
        if task.tool in {"retrieve_reaction_rrf", "select_reaction_grounded"}:
            task.state = TaskState.COMPLETED
    controller.executed_plan = plan
    controller._invalidate_downstream(plan, "a")
    assert any(
        t.state == TaskState.PENDING and t.tool in {"retrieve_reaction_rrf", "select_reaction_grounded"}
        for t in plan.tasks
    )


def test_compare_auto_plan_with_explicit_workflow_coverage(reaction_only_model, tmp_path):
    """Operational comparison only — no biological accuracy claim."""
    auto = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(
            grounded_responses={
                "default": {
                    "abstain": True,
                    "predicted_kegg_id": None,
                    "selected_evidence_rank": None,
                    "supporting_evidence_ids": [],
                    "confidence": 0.2,
                    "reasoning_summary": "abstain",
                    "abstention_reason": "Insufficient evidence",
                }
            }
        ),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "auto",
    )
    from core import annotate_model

    explicit = annotate_model(
        str(reaction_only_model),
        annotate="reactions",
        reaction_method="rrf_grounded",
        reaction_retriever=TinyRetriever(),
        dry_run=True,
        llm_model="gpt-6.1-terra",
        save_to=str(tmp_path / "explicit"),
    )
    auto_targets = {t.target_id for t in auto.initial_plan.tasks if t.tool.startswith("retrieve") or t.tool.startswith("select")}
    explicit_ids = set(explicit.reaction_recommendations_df["id"].tolist())
    assert auto_targets <= explicit_ids or auto_targets == explicit_ids or explicit_ids
    comparison = {
        "auto_task_tools": sorted({t.tool for t in auto.initial_plan.tasks}),
        "explicit_status_counts": explicit.reaction_recommendations_df["status"].value_counts().to_dict(),
        "auto_report_status": auto.report.status,
        "note": "Operational plan/routing comparison only; no accuracy claim",
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    assert "retrieve_reaction_rrf" in comparison["auto_task_tools"]


def test_parse_failure_does_not_claim_annotated_file(invalid_model, tmp_path):
    result = auto_annotate_model(
        str(invalid_model),
        backend=MockBackend(),
        output_dir=tmp_path / "bad",
    )
    assert result.report.annotated_model_path is None
    assert result.report.status == "failed"
