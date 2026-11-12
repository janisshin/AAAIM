"""Deterministic mocked backend for controller tests."""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from core.agent.schemas import (
    AgentAction,
    AnnotationPlan,
    AnnotationStatus,
    AnnotationTask,
    BiologicalType,
    EntityRepresentation,
    FailureKind,
    ModelInventory,
    PlannerDecision,
    TaskState,
    ToolObservation,
)


class MockBackend:
    """Scriptable planner/transport used by unit tests."""

    name = "mock"

    def __init__(
        self,
        *,
        decisions: Optional[list[PlannerDecision]] = None,
        grounded_responses: Optional[dict[str, dict[str, Any]]] = None,
        fail_plan_times: int = 0,
    ) -> None:
        self._decisions = list(decisions or [])
        self._grounded = dict(grounded_responses or {})
        self._call_i = 0
        self._fail_plan_times = fail_plan_times
        self.plan_calls = 0
        self.transport_calls = 0

    def identity(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "schema_version": "agent-planner-v1",
            "quantization": None,
            "decoding": None,
        }

    def unload(self) -> None:
        return None

    def _default_plan(self, inventory: ModelInventory) -> PlannerDecision:
        tasks: list[AnnotationTask] = []
        actions: list[AgentAction] = []
        species_task_ids: dict[str, str] = {}

        for ent in inventory.entities:
            if ent.representation in {
                EntityRepresentation.SBML_SPECIES,
                EntityRepresentation.FBC_GENE_PRODUCT,
                EntityRepresentation.SBML_QUAL_SPECIES,
            }:
                if ent.annotation_status in {
                    AnnotationStatus.COMPLETE,
                    AnnotationStatus.CONFLICTING,
                    AnnotationStatus.UNSUPPORTED,
                }:
                    continue
                if ent.biological_type == BiologicalType.AMBIGUOUS:
                    tid = f"species_unresolved::{ent.target_id}"
                    tasks.append(
                        AnnotationTask(
                            task_id=tid,
                            target_id=ent.target_id,
                            representation=ent.representation,
                            biological_type=ent.biological_type,
                            tool="annotate_species_candidates",
                            databases=[],
                            state=TaskState.UNRESOLVED,
                            failure_kind=FailureKind.UNCERTAIN_TYPE,
                            reason="ambiguous biological type",
                        )
                    )
                    continue
                tid = f"species::{ent.target_id}"
                species_task_ids[ent.target_id] = tid
                db = {
                    BiologicalType.CHEMICAL: ["chebi"],
                    BiologicalType.GENE: ["ncbigene"],
                    BiologicalType.PROTEIN: ["uniprot"],
                    BiologicalType.COMPLEX: ["chebi", "uniprot"],
                }.get(ent.biological_type, ["chebi"])
                tasks.append(
                    AnnotationTask(
                        task_id=tid,
                        target_id=ent.target_id,
                        representation=ent.representation,
                        biological_type=ent.biological_type,
                        tool="annotate_species_candidates",
                        databases=db,
                    )
                )
                actions.append(
                    AgentAction(
                        action_id=f"act::{tid}",
                        tool="annotate_species_candidates",
                        target_ids=[ent.target_id],
                        task_ids=[tid],
                        arguments={"databases": db},
                        rationale="annotate missing species identity",
                    )
                )

        for ent in inventory.entities:
            if ent.representation != EntityRepresentation.SBML_REACTION:
                continue
            if ent.annotation_status in {
                AnnotationStatus.COMPLETE,
                AnnotationStatus.CONFLICTING,
                AnnotationStatus.UNSUPPORTED,
            }:
                continue
            deps = [
                species_task_ids[p]
                for p in ent.participants
                if p in species_task_ids
            ]
            rid = f"retrieve::{ent.target_id}"
            sid = f"select::{ent.target_id}"
            tasks.append(
                AnnotationTask(
                    task_id=rid,
                    target_id=ent.target_id,
                    representation=ent.representation,
                    biological_type=BiologicalType.REACTION,
                    tool="retrieve_reaction_rrf",
                    databases=["kegg"],
                    depends_on=deps,
                    arguments={"allow_names_only": True},
                )
            )
            tasks.append(
                AnnotationTask(
                    task_id=sid,
                    target_id=ent.target_id,
                    representation=ent.representation,
                    biological_type=BiologicalType.REACTION,
                    tool="select_reaction_grounded",
                    databases=["kegg"],
                    depends_on=[rid],
                )
            )
            actions.append(
                AgentAction(
                    action_id=f"act::{rid}",
                    tool="retrieve_reaction_rrf",
                    target_ids=[ent.target_id],
                    task_ids=[rid],
                    arguments={"allow_names_only": True},
                    rationale="RRF retrieval for reaction",
                )
            )
            actions.append(
                AgentAction(
                    action_id=f"act::{sid}",
                    tool="select_reaction_grounded",
                    target_ids=[ent.target_id],
                    task_ids=[sid],
                    arguments={},
                    rationale="grounded selection for reaction",
                )
            )

        actions.append(
            AgentAction(
                action_id="act::export",
                tool="export_reports",
                target_ids=[],
                task_ids=[],
                arguments={},
                rationale="export annotated copy and reports",
            )
        )
        return PlannerDecision(
            decision="plan",
            tasks=tasks,
            actions=actions,
            reason="mock default mixed-entity plan",
        )

    def plan(
        self,
        *,
        request: str,
        inventory: ModelInventory,
        prior_plan: Optional[AnnotationPlan] = None,
        observations: Sequence[ToolObservation] = (),
        revision_reason: str = "",
    ) -> PlannerDecision:
        self.plan_calls += 1
        if self._fail_plan_times > 0:
            self._fail_plan_times -= 1
            raise RuntimeError("mock planner technical failure")
        if self._decisions:
            if self._call_i < len(self._decisions):
                decision = self._decisions[self._call_i]
                self._call_i += 1
                return decision
            return PlannerDecision(decision="stop", reason="no further scripted decisions")
        if prior_plan is None:
            return self._default_plan(inventory)
        # After observations: continue until export, or stop if asked.
        if revision_reason:
            return PlannerDecision(
                decision="revise",
                tasks=prior_plan.tasks,
                actions=[a for a in prior_plan.actions if a.tool == "export_reports"],
                reason=revision_reason or "mock revision after observation",
            )
        return PlannerDecision(decision="continue", reason="mock continue")

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        self.transport_calls += 1
        key = json.dumps(provider_payload.get("input", ""), sort_keys=True)[:80]
        answer = self._grounded.get(key) or self._grounded.get("default")
        if answer is None:
            answer = {
                "abstain": True,
                "predicted_kegg_id": None,
                "selected_evidence_rank": None,
                "supporting_evidence_ids": [],
                "confidence": 0.1,
                "reasoning_summary": "Mock abstention",
                "abstention_reason": "Insufficient evidence in mock backend",
            }
        model = provider_payload.get("model", "mock-grounded")
        return {
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(answer)}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
