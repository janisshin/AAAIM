"""Semantic checks are distinct from Pydantic schema validation and biology."""
from __future__ import annotations

import pytest

from core.agent.inspect_model import inspect_sbml_model
from core.agent.schemas import (
    AgentAction,
    AnnotationPlan,
    AnnotationTask,
    BiologicalType,
    EntityRepresentation,
    FailureKind,
    RetrieveReactionArgs,
    SemanticCheckReport,
    SelectReactionArgs,
    ToolObservation,
    parse_tool_arguments,
)
from core.agent.validate import (
    ObservationValidationError,
    PlanValidationError,
    check_evidence_consistency,
    check_target_existence,
    validate_action,
    validate_observation,
    validate_plan,
)
from core.reaction.grounded import GroundedAnnotation


def test_parse_tool_arguments_rejects_unknown_fields():
    with pytest.raises(Exception):
        parse_tool_arguments("retrieve_reaction_rrf", {"allow_names_only": True, "extra": 1})


def test_parse_tool_arguments_reuses_typed_defaults():
    args = parse_tool_arguments("select_reaction_grounded", {})
    assert isinstance(args, SelectReactionArgs)
    assert args.max_output_tokens == 512
    args2 = parse_tool_arguments("retrieve_reaction_rrf", {"allow_names_only": False})
    assert isinstance(args2, RetrieveReactionArgs)
    assert args2.allow_names_only is False


def test_target_existence_semantic_check(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    problems = check_target_existence(["missing"], inv, tool="retrieve_reaction_rrf")
    assert problems
    assert not check_target_existence(["r1"], inv, tool="retrieve_reaction_rrf")


def test_validate_action_rejects_species_target_for_reaction_tool(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    action = AgentAction(
        action_id="bad",
        tool="retrieve_reaction_rrf",
        target_ids=["a"],
        task_ids=[],
        arguments={"allow_names_only": True},
    )
    with pytest.raises(PlanValidationError, match="sbml_reaction"):
        validate_action(action, inv)


def test_dependency_validity_requires_retrieve_before_select(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    plan = AnnotationPlan(
        tasks=[
            AnnotationTask(
                task_id="select::r1",
                target_id="r1",
                representation=EntityRepresentation.SBML_REACTION,
                biological_type=BiologicalType.REACTION,
                tool="select_reaction_grounded",
                depends_on=[],
                arguments={},
            )
        ],
        actions=[],
    )
    with pytest.raises(PlanValidationError, match="retrieve_reaction_rrf"):
        validate_plan(plan, inv)


def test_evidence_consistency_rejects_mismatched_retrieval_ids(reaction_only_model):
    problems = check_evidence_consistency(
        tool="retrieve_reaction_rrf",
        target_ids=["r1"],
        result={"n_retrieved": 1, "reaction_ids": ["r1"]},
        evidence={"artifacts": []},
    )
    assert problems


def test_evidence_consistency_blocks_technical_as_biological_abstention():
    problems = check_evidence_consistency(
        tool="select_reaction_grounded",
        target_ids=["r1"],
        result={
            "selected_ids": [],
            "abstained_ids": [],
            "failed_ids": ["r1"],
            "recommendations": [
                {
                    "id": "r1",
                    "status": "schema_invalid",
                    "biological_abstain": True,
                    "annotation": "",
                    "update_annotation": "ignore",
                }
            ],
        },
        evidence={},
    )
    assert any("biological abstention" in p for p in problems)


def test_validate_observation_attaches_semantic_report(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    obs = ToolObservation(
        action_id="a1",
        tool="assess_identity_support",
        ok=True,
        target_ids=["a"],
        result={"assessments": [{"id": "a", "identity_status": "supported"}]},
    )
    out = validate_observation(obs, inv, expected_target_ids=["a"])
    assert isinstance(out.semantic_checks, SemanticCheckReport)
    assert out.semantic_checks.ok
    assert out.semantic_checks.biological_correctness_certified is False


def test_validate_observation_fails_closed_on_target_mismatch(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    obs = ToolObservation(
        action_id="a1",
        tool="assess_identity_support",
        ok=True,
        target_ids=["a"],
        result={"assessments": [{"id": "a", "identity_status": "supported"}]},
    )
    with pytest.raises(ObservationValidationError):
        validate_observation(obs, inv, expected_target_ids=["b"])


def test_grounded_annotation_reuse_and_nonclaim():
    answer = GroundedAnnotation(
        abstain=True,
        predicted_kegg_id=None,
        selected_evidence_rank=None,
        supporting_evidence_ids=[],
        confidence=0.1,
        reasoning_summary="insufficient",
        abstention_reason="Insufficient evidence",
    )
    # Schema-valid grounded answers still do not certify biology.
    assert answer.abstain is True
    report = SemanticCheckReport(ok=True, problems=[], checks_performed=["schema"])
    assert report.biological_correctness_certified is False


def test_failed_observation_cannot_use_biological_abstention_kind(reaction_only_model):
    inv = inspect_sbml_model(str(reaction_only_model))
    obs = ToolObservation(
        action_id="a1",
        tool="select_reaction_grounded",
        ok=False,
        target_ids=["r1"],
        error="provider down",
        failure_kind=FailureKind.BIOLOGICAL_ABSTENTION,
        result={},
    )
    with pytest.raises(ObservationValidationError, match="biological_abstention"):
        validate_observation(obs, inv)
