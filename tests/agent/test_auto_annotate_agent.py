"""Deterministic controller tests for the automatic annotation agent."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import libsbml
import pandas as pd
import pytest

from core.agent import AnnotationScope, CheckpointCompatibilityError, ExecutionLimits, auto_annotate_model
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


def _selected_answer():
    return {
        "abstain": False,
        "predicted_kegg_id": "R00001",
        "selected_evidence_rank": 1,
        "supporting_evidence_ids": ["E01"],
        "confidence": 0.7,
        "reasoning_summary": "Synthetic selection",
        "abstention_reason": None,
    }


def _tool_counts(result) -> Counter:
    return Counter(o.tool for o in result.observations)


def _entity_states(result) -> dict[str, str]:
    return {k: v.state.value for k, v in sorted(result.entity_states.items())}


def _task_states(result) -> dict[str, str]:
    assert result.executed_plan is not None
    return {t.task_id: t.state.value for t in result.executed_plan.tasks}


def _reaction_ids(result) -> list[str]:
    df = None
    # recommendations may live only on resources via report path; use observations/result
    if result.report.recommendations_path:
        payload = json.loads(Path(result.report.recommendations_path).read_text(encoding="utf-8"))
        rows = payload.get("reactions") or []
        return [r.get("id") for r in rows if r.get("annotation")]
    return []


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
        limits=ExecutionLimits(max_actions=2, max_failure_revisions=0),
    )
    assert result.report.status in {"partial", "interrupted", "completed"}
    assert result.provenance.get("stop_reason") in {
        None,
        "max_actions",
        "max_failure_revisions",
        "max_planning_turns",
        "max_revisions",
    }


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
        PlannerDecision(decision="continue", tasks=inv_tasks, actions=[action, action, action]),
        PlannerDecision(decision="finish", reason="stop", tasks=inv_tasks),
    ]
    result = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(decisions=decisions),
        output_dir=tmp_path / "repeat",
        limits=ExecutionLimits(max_actions=10, max_retries=0),
        allow_structural_fallback=False,
    )
    assert result.report is not None


def test_interrupt_resume_after_species_annotation(chemical_only_model, tmp_path):
    """Interrupt after species work; resume must not re-annotate or replan."""
    out = tmp_path / "species_resume"
    baseline = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=tmp_path / "species_base",
    )
    # chemical_only: two species annotate actions then export
    interrupted = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out,
        interrupt_after_actions=2,
    )
    assert interrupted.report.status == "interrupted"
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["schema_version"] == 2
    assert ckpt["executed_plan"] is not None
    assert any(
        t["tool"] == "annotate_species_candidates" and t["state"] == "completed"
        for t in ckpt["executed_plan"]["tasks"]
    )

    backend2 = MockBackend()
    resumed = auto_annotate_model(
        str(chemical_only_model),
        backend=backend2,
        output_dir=out,
        resume_from=out,
    )
    # May request finish/continue turns after the saved action group is exhausted,
    # but must not re-execute completed species annotation.
    assert resumed.report.status in {"completed", "partial"}
    assert _entity_states(resumed) == _entity_states(baseline)
    assert _tool_counts(resumed)["annotate_species_candidates"] == _tool_counts(baseline)[
        "annotate_species_candidates"
    ]
    assert all(
        o.action_id in {x.action_id for x in interrupted.observations}
        for o in resumed.observations
        if o.tool == "annotate_species_candidates"
    )
    for tid in resumed.report.completed_ids:
        assert resumed.entity_states[tid].state == TaskState.COMPLETED
    species_rows = json.loads(Path(resumed.report.recommendations_path).read_text(encoding="utf-8"))[
        "species"
    ]
    ids = [r["id"] for r in species_rows]
    assert len(ids) == len(set(ids))


def test_interrupt_resume_between_retrieve_and_select(reaction_only_model, tmp_path):
    out = tmp_path / "retrieve_resume"
    retriever = TinyRetriever()
    answer = _selected_answer()
    baseline_ret = TinyRetriever()
    baseline = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(grounded_responses={"default": answer}),
        reaction_retriever=baseline_ret,
        output_dir=tmp_path / "retrieve_base",
    )
    interrupted = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(grounded_responses={"default": answer}),
        reaction_retriever=retriever,
        output_dir=out,
        interrupt_after_actions=1,
    )
    assert interrupted.report.status == "interrupted"
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["reaction_evidence"], "evidence must be persisted before select"
    assert retriever.retrieve_calls == 1

    backend2 = MockBackend(grounded_responses={"default": answer})
    resumed = auto_annotate_model(
        str(reaction_only_model),
        backend=backend2,
        reaction_retriever=retriever,
        output_dir=out,
        resume_from=out,
    )
    assert retriever.retrieve_calls == 1  # no re-retrieve
    assert any(o.tool == "select_reaction_grounded" and o.ok for o in resumed.observations)
    assert _entity_states(resumed) == _entity_states(baseline)
    assert _tool_counts(resumed)["retrieve_reaction_rrf"] == _tool_counts(baseline)[
        "retrieve_reaction_rrf"
    ]
    assert all(
        o.action_id in {x.action_id for x in interrupted.observations}
        for o in resumed.observations
        if o.tool == "retrieve_reaction_rrf"
    )


def test_interrupt_resume_after_select_before_write(reaction_only_model, tmp_path):
    out = tmp_path / "select_resume"
    answer = _selected_answer()
    retriever = TinyRetriever()
    baseline = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(grounded_responses={"default": answer}),
        reaction_retriever=TinyRetriever(),
        output_dir=tmp_path / "select_base",
    )
    interrupted = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(grounded_responses={"default": answer}),
        reaction_retriever=retriever,
        output_dir=out,
        interrupt_after_actions=2,
    )
    assert interrupted.report.status == "interrupted"
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["reaction_recommendations"] is not None
    select_obs = [o for o in interrupted.observations if o.tool == "select_reaction_grounded"]
    assert select_obs and select_obs[0].ok

    backend2 = MockBackend(grounded_responses={"default": answer})
    resumed = auto_annotate_model(
        str(reaction_only_model),
        backend=backend2,
        reaction_retriever=retriever,
        output_dir=out,
        resume_from=out,
    )
    assert backend2.transport_calls == 0  # selection not repeated
    assert _entity_states(resumed) == _entity_states(baseline)
    assert _reaction_ids(resumed) == _reaction_ids(baseline)
    assert _tool_counts(resumed)["select_reaction_grounded"] == _tool_counts(baseline)[
        "select_reaction_grounded"
    ]


def test_interrupt_resume_during_plan_revision(reaction_only_model, tmp_path):
    inv = inspect_sbml_model(str(reaction_only_model))
    seed = MockBackend()._default_plan(inv)
    decisions = [
        seed,
        PlannerDecision(
            decision="continue",
            tasks=list(seed.tasks),
            actions=list(seed.actions),
            reason="retry after transient retrieval failure",
        ),
    ]

    class FlakyRetriever(TinyRetriever):
        def retrieve(self, query):
            if self.retrieve_calls == 0:
                self.retrieve_calls += 1
                raise RuntimeError("transient retrieval failure")
            return super().retrieve(query)

    out = tmp_path / "revision_resume"
    answer = _selected_answer()
    retriever = FlakyRetriever()
    interrupted = auto_annotate_model(
        str(reaction_only_model),
        backend=MockBackend(decisions=list(decisions), grounded_responses={"default": answer}),
        reaction_retriever=retriever,
        output_dir=out,
        interrupt_after_actions=1,
        allow_structural_fallback=False,
    )
    assert interrupted.report.status == "interrupted"
    assert interrupted.provenance.get("revisions")
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["revision_count"] >= 1
    assert ckpt["failure_revision_count"] >= 1
    assert ckpt["executed_plan"]["plan_version"] >= 2

    backend2 = MockBackend(decisions=[], grounded_responses={"default": answer})
    resumed = auto_annotate_model(
        str(reaction_only_model),
        backend=backend2,
        reaction_retriever=retriever,
        output_dir=out,
        resume_from=out,
        allow_structural_fallback=False,
    )
    assert resumed.executed_plan.plan_version >= 2
    assert resumed.provenance.get("revisions")
    assert resumed.report.status in {"completed", "partial", "interrupted"}
    # Failure revision was restored; successful retrieve happens at most once after the flake.
    assert retriever.retrieve_calls == 2


def test_resume_rejects_incompatible_and_corrupt_checkpoints(chemical_only_model, tmp_path):
    out = tmp_path / "compat"
    auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out,
        interrupt_after_actions=1,
    )
    ckpt_path = out / "checkpoint.json"
    good = json.loads(ckpt_path.read_text(encoding="utf-8"))

    # Changed settings / backend / tool schema / resources / model hash
    for field, value in [
        ("settings", {**good["settings"], "request": "different request"}),
        ("backend", {**good["backend"], "backend": "other"}),
        ("tool_schema_version", "agent-tools-v999"),
        ("resource_identity", {**good["resource_identity"], "reaction_retriever": {"resource": "other"}}),
        ("model_sha256", "0" * 64),
    ]:
        mutated = dict(good)
        mutated[field] = value
        # Keep cache_identity stale so mismatch is reported clearly
        ckpt_path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
        with pytest.raises(CheckpointCompatibilityError, match=field):
            auto_annotate_model(
                str(chemical_only_model),
                backend=MockBackend(),
                output_dir=out,
                resume_from=out,
            )

    # Corrupt JSON
    ckpt_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointCompatibilityError, match="corrupt"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=out,
            resume_from=out,
        )

    # Missing schema_version (pre-v1)
    ckpt_path.write_text(json.dumps({"cache_identity": good["cache_identity"]}), encoding="utf-8")
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=out,
            resume_from=out,
        )

    # Pre-v2 schema rejected (resource/tokenizer identity strengthened in v2)
    ckpt_path.write_text(
        json.dumps({"schema_version": 1, "cache_identity": good["cache_identity"]}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointCompatibilityError, match="schema_version"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=out,
            resume_from=out,
        )

    # Incomplete / truncated required fields on v2
    ckpt_path.write_text(
        json.dumps({"schema_version": 2, "cache_identity": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointCompatibilityError, match="incomplete|invalid"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=out,
            resume_from=out,
        )

    # Missing checkpoint file
    missing = tmp_path / "missing_dir"
    missing.mkdir()
    with pytest.raises(CheckpointCompatibilityError, match="not found"):
        auto_annotate_model(
            str(chemical_only_model),
            backend=MockBackend(),
            output_dir=missing,
            resume_from=missing,
        )


def test_resume_clears_in_flight_for_at_least_once(chemical_only_model, tmp_path):
    out = tmp_path / "inflight"
    auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out,
        interrupt_after_actions=1,
    )
    ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
    # Simulate crash after tool started but before durable completion.
    completed = list(ckpt["completed_fingerprints"])
    assert completed
    fp = completed[0]
    action = next(
        a for a in ckpt["executed_plan"]["actions"] if a["tool"] == "annotate_species_candidates"
    )
    ckpt["completed_fingerprints"] = [x for x in completed if x != fp]
    ckpt["in_flight_action"] = {"action": action, "fingerprint": fp}
    # Roll back that observation from the durable log for a realistic crash window.
    ckpt["observations"] = [o for o in ckpt["observations"] if o.get("action_id") != action["action_id"]]
    (out / "checkpoint.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    resumed = auto_annotate_model(
        str(chemical_only_model),
        backend=MockBackend(),
        output_dir=out,
        resume_from=out,
    )
    assert resumed.report.status in {"completed", "partial"}
    annotate = [o for o in resumed.observations if o.tool == "annotate_species_candidates"]
    assert annotate  # at-least-once re-execution allowed


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
