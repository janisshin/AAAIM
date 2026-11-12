"""Allowlisted tool registry wrapping existing AAAIM implementations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from core.agent.inspect_model import inspect_sbml_model
from core.agent.schemas import (
    AgentAction,
    AnnotationStatus,
    FailureKind,
    ModelInventory,
    ToolObservation,
    parse_tool_arguments,
)  # AnnotationStatus used by classify_or_route_entities
from core.agent.species import llm_backend_hook
from core.agent.validate import ObservationValidationError, validate_observation
from core.reaction import grounded as grounded_mod
from core.reaction.species_evidence import build_queries
from core.update_model import update_annotation
from core.reaction.writer import write_reaction_annotations


@dataclass
class AgentResources:
    model_file: str
    output_dir: Path
    inventory: ModelInventory
    reaction_resources: Any = None
    reaction_retriever: Any = None
    grounded_transport: Optional[Callable[[dict], dict]] = None
    species_annotator: Optional[Callable[..., pd.DataFrame]] = None
    species_backend: Any = None
    species_annotator_provenance: dict[str, Any] = field(default_factory=dict)
    species_recommendations: Optional[pd.DataFrame] = None
    reaction_recommendations: Optional[pd.DataFrame] = None
    reaction_evidence: list[dict] = field(default_factory=list)
    entity_identities: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    allow_names_only_reactions: bool = True
    identity_support_available: bool = True
    qualifier_prediction_available: bool = True


def _obs(
    action: AgentAction,
    *,
    ok: bool,
    result: dict | None = None,
    error: str | None = None,
    failure_kind: FailureKind = FailureKind.NONE,
    evidence: dict | None = None,
    grounded_answer: Any = None,
    inventory: ModelInventory | None = None,
    skip_semantic: bool = False,
) -> ToolObservation:
    observation = ToolObservation(
        action_id=action.action_id,
        tool=action.tool,
        ok=ok,
        target_ids=list(action.target_ids),
        task_ids=list(action.task_ids),
        result=result or {},
        error=error,
        failure_kind=failure_kind,
        evidence=evidence or {},
        grounded_answer=grounded_answer,
    )
    if skip_semantic or inventory is None:
        return observation
    try:
        return validate_observation(
            observation,
            inventory,
            expected_target_ids=list(action.target_ids),
        )
    except ObservationValidationError as exc:
        # Schema/semantic failure of a tool result is a technical failure, never
        # biological abstention, and never a claim of biological correctness.
        observation.ok = False
        observation.error = f"observation semantic/schema validation failed: {exc}"
        observation.failure_kind = FailureKind.TECHNICAL_FAILURE
        if observation.semantic_checks is None or observation.semantic_checks.ok:
            from core.agent.schemas import SemanticCheckReport

            observation.semantic_checks = SemanticCheckReport(
                ok=False,
                problems=[str(exc)],
                checks_performed=["observation_schema", "evidence_consistency"],
            )
        return observation


class ToolRegistry:
    def __init__(self, resources: AgentResources) -> None:
        self.resources = resources
        self.handlers = {
            "inspect_model": self.inspect_model,
            "classify_or_route_entities": self.classify_or_route_entities,
            "annotate_species_candidates": self.annotate_species_candidates,
            "assess_identity_support": self.assess_identity_support,
            "predict_qualifiers": self.predict_qualifiers,
            "retrieve_reaction_rrf": self.retrieve_reaction_rrf,
            "select_reaction_grounded": self.select_reaction_grounded,
            "write_species_annotations": self.write_species_annotations,
            "write_reaction_annotations": self.write_reaction_annotations_tool,
            "persist_checkpoint": self.persist_checkpoint,
            "export_reports": self.export_reports_placeholder,
        }

    def execute(self, action: AgentAction) -> ToolObservation:
        handler = self.handlers.get(action.tool)
        if handler is None:
            return _obs(
                action,
                ok=False,
                error=f"tool not registered: {action.tool}",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        try:
            # Fail closed on argument schema before any side effects.
            parse_tool_arguments(action.tool, action.arguments)
            return handler(action)
        except Exception as exc:
            return _obs(
                action,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )

    def inspect_model(self, action: AgentAction) -> ToolObservation:
        inv = inspect_sbml_model(self.resources.model_file)
        self.resources.inventory = inv
        return _obs(
            action,
            ok=inv.parse_ok,
            result=inv.model_dump(mode="json"),
            error=inv.parse_error,
            failure_kind=FailureKind.NONE if inv.parse_ok else FailureKind.INVALID_MODEL_OUTPUT,
            inventory=inv,
        )

    def classify_or_route_entities(self, action: AgentAction) -> ToolObservation:
        inv = self.resources.inventory
        routing = []
        for ent in inv.entities:
            if action.target_ids and ent.target_id not in action.target_ids:
                continue
            routing.append(
                {
                    "target_id": ent.target_id,
                    "representation": ent.representation.value,
                    "biological_type": ent.biological_type.value,
                    "annotation_status": ent.annotation_status.value,
                    "supported": ent.annotation_status != AnnotationStatus.UNSUPPORTED,
                }
            )
        return _obs(action, ok=True, result={"routing": routing}, inventory=inv)

    def annotate_species_candidates(self, action: AgentAction) -> ToolObservation:
        annotator = self.resources.species_annotator
        if annotator is None:
            return _obs(
                action,
                ok=False,
                error=(
                    "species annotator not configured; refusing implicit synthetic "
                    "identifier fallback"
                ),
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        args = parse_tool_arguments(action.tool, action.arguments)
        databases = list(args.databases)  # type: ignore[attr-defined]
        try:
            frame = annotator(
                self.resources.model_file,
                list(action.target_ids),
                databases,
                self.resources.inventory,
            )
        except Exception as exc:
            return _obs(
                action,
                ok=False,
                error=f"species annotation failed: {type(exc).__name__}: {exc}",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        if frame is None or not isinstance(frame, pd.DataFrame):
            return _obs(
                action,
                ok=False,
                error="species annotator returned no DataFrame",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        if self.resources.species_recommendations is None:
            self.resources.species_recommendations = frame.copy()
        else:
            self.resources.species_recommendations = pd.concat(
                [self.resources.species_recommendations, frame], ignore_index=True
            )
        unresolved = [
            r["id"]
            for r in frame.fillna("").to_dict("records")
            if str(r.get("identity_status", "")).lower() == "unresolved"
            or str(r.get("status", "")).lower() == "unresolved"
            or (
                str(r.get("update_annotation", "")).lower() == "ignore"
                and not str(r.get("annotation", "")).strip()
            )
        ]
        for row in frame.fillna("").to_dict("records"):
            if str(row.get("id")) == "Reason:":
                continue
            self.resources.entity_identities[str(row["id"])] = {
                "annotation": row.get("annotation"),
                "identity_status": row.get("identity_status") or "",
                "qualifier": row.get("qualifier", "is"),
                "qualifier_status": row.get("qualifier_status") or "",
                "component_id": row.get("component_id") or "",
                "annotator_kind": row.get("annotator_kind")
                or self.resources.species_annotator_provenance.get("kind"),
            }
        failure = FailureKind.UNRESOLVED_IDENTITY if unresolved else FailureKind.NONE
        return _obs(
            action,
            ok=True,
            result={
                "n_rows": int(len(frame)),
                "unresolved_ids": unresolved,
                "recommendations": frame.fillna("").to_dict("records"),
                "annotator_kind": self.resources.species_annotator_provenance.get("kind"),
            },
            failure_kind=failure,
            inventory=self.resources.inventory,
        )

    def assess_identity_support(self, action: AgentAction) -> ToolObservation:
        if not self.resources.identity_support_available:
            return _obs(
                action,
                ok=False,
                error="identity-support assessment unavailable in this configuration",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        frame = self.resources.species_recommendations
        if frame is None or frame.empty:
            return _obs(
                action,
                ok=False,
                error="no species recommendations available for identity support",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        # Mock synthetic rows already carry identity_status; do not invent support.
        kind = self.resources.species_annotator_provenance.get("kind")
        if kind == "mock_synthetic":
            ids = action.target_ids or sorted(self.resources.entity_identities)
            out = []
            for tid in ids:
                info = self.resources.entity_identities.get(tid, {})
                out.append(
                    {
                        "id": tid,
                        "identity_status": info.get("identity_status") or "unresolved",
                    }
                )
            return _obs(
                action,
                ok=True,
                result={"assessments": out, "mode": "mock_passthrough"},
                inventory=self.resources.inventory,
            )

        from core.identity_support import assess_species_identity_support

        args = parse_tool_arguments(action.tool, action.arguments)
        subset = frame
        if action.target_ids:
            subset = frame[frame["id"].astype(str).isin(action.target_ids)].copy()
        try:
            with llm_backend_hook(self.resources.species_backend) as meta:
                if not meta.get("hook_installed"):
                    raise RuntimeError(
                        "identity support requires a local LLM backend; "
                        "refusing hosted fallback and fabricated support labels"
                    )
                assessed = assess_species_identity_support(
                    subset,
                    model_file=self.resources.model_file,
                    llm_model="metabolm-local",
                    effort=getattr(args, "effort", "low"),
                )
        except Exception as exc:
            return _obs(
                action,
                ok=False,
                error=f"identity support failed: {type(exc).__name__}: {exc}",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        # Merge assessed rows back
        self.resources.species_recommendations = assessed
        out = []
        for row in assessed.fillna("").to_dict("records"):
            if str(row.get("id")) == "Reason:":
                continue
            tid = str(row["id"])
            status = str(row.get("identity_status") or "")
            self.resources.entity_identities.setdefault(tid, {})
            self.resources.entity_identities[tid].update(
                {
                    "annotation": row.get("annotation"),
                    "identity_status": status,
                    "qualifier": row.get("qualifier", "is"),
                    "component_id": row.get("component_id") or "",
                }
            )
            if action.target_ids and tid not in action.target_ids:
                continue
            out.append({"id": tid, "identity_status": status})
        return _obs(
            action,
            ok=True,
            result={"assessments": out, "mode": "real_assess_species_identity_support"},
            inventory=self.resources.inventory,
        )

    def predict_qualifiers(self, action: AgentAction) -> ToolObservation:
        if not self.resources.qualifier_prediction_available:
            return _obs(
                action,
                ok=False,
                error="qualifier prediction unavailable in this configuration",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        frame = self.resources.species_recommendations
        if frame is None or frame.empty:
            return _obs(
                action,
                ok=False,
                error="no species recommendations available for qualifier prediction",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        kind = self.resources.species_annotator_provenance.get("kind")
        if kind == "mock_synthetic":
            ids = action.target_ids or sorted(self.resources.entity_identities)
            out = []
            for tid in ids:
                info = self.resources.entity_identities.get(tid, {})
                out.append(
                    {
                        "id": tid,
                        "qualifier": info.get("qualifier") or "is",
                        "qualifier_status": info.get("qualifier_status") or "mock",
                    }
                )
            return _obs(
                action,
                ok=True,
                result={"qualifiers": out, "mode": "mock_passthrough"},
                inventory=self.resources.inventory,
            )

        from core.qualifier_workflow import predict_species_qualifiers

        args = parse_tool_arguments(action.tool, action.arguments)
        subset = frame
        if action.target_ids:
            subset = frame[frame["id"].astype(str).isin(action.target_ids)].copy()
        try:
            with llm_backend_hook(self.resources.species_backend) as meta:
                if not meta.get("hook_installed"):
                    raise RuntimeError(
                        "qualifier prediction requires a local LLM backend; "
                        "refusing hosted fallback"
                    )
                predicted = predict_species_qualifiers(
                    subset,
                    model_file=self.resources.model_file,
                    llm_model="metabolm-local",
                    effort=getattr(args, "effort", "low"),
                )
        except Exception as exc:
            return _obs(
                action,
                ok=False,
                error=f"qualifier prediction failed: {type(exc).__name__}: {exc}",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        self.resources.species_recommendations = predicted
        out = []
        for row in predicted.fillna("").to_dict("records"):
            if str(row.get("id")) == "Reason:":
                continue
            tid = str(row["id"])
            self.resources.entity_identities.setdefault(tid, {})
            self.resources.entity_identities[tid].update(
                {
                    "qualifier": row.get("qualifier") or "",
                    "qualifier_status": row.get("qualifier_status") or "",
                    "identity_status": row.get("identity_status")
                    or self.resources.entity_identities[tid].get("identity_status"),
                    "component_id": row.get("component_id") or "",
                }
            )
            if action.target_ids and tid not in action.target_ids:
                continue
            out.append(
                {
                    "id": tid,
                    "qualifier": row.get("qualifier") or "",
                    "qualifier_status": row.get("qualifier_status") or "",
                }
            )
        return _obs(
            action,
            ok=True,
            result={"qualifiers": out, "mode": "real_predict_species_qualifiers"},
            inventory=self.resources.inventory,
        )

    def retrieve_reaction_rrf(self, action: AgentAction) -> ToolObservation:
        args = parse_tool_arguments(action.tool, action.arguments)
        allow_names_only = bool(
            getattr(args, "allow_names_only", self.resources.allow_names_only_reactions)
        )
        species_df = self.resources.species_recommendations
        # Unresolved species must not block names-only retrieval.
        if species_df is not None and not allow_names_only:
            unresolved = species_df[
                species_df.get("identity_status", pd.Series(dtype=str)).astype(str).str.lower()
                == "unresolved"
            ]
            # If any participant unresolved and names-only forbidden, skip those reactions.
            pass
        retriever = self.resources.reaction_retriever
        if retriever is None and self.resources.reaction_resources is not None:
            from core.reaction.retrieval import load_retriever

            retriever = load_retriever(self.resources.reaction_resources)
            self.resources.reaction_retriever = retriever
        if retriever is None:
            return _obs(
                action,
                ok=False,
                error="reaction retriever/resources unavailable",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        queries = build_queries(self.resources.model_file, species_df)
        wanted = set(action.target_ids) if action.target_ids else {q["reaction_id"] for q in queries}
        artifacts = []
        for query in queries:
            if query["reaction_id"] not in wanted:
                continue
            evidence = retriever.retrieve(query["query"])
            grounded_mod.validate_evidence(evidence)
            artifacts.append(
                {
                    "reaction_id": query["reaction_id"],
                    "query": query["query"],
                    "evidence": evidence,
                    "display_name": query.get("display_name", ""),
                }
            )
        # Store for selection stage
        existing = {
            a.get("reaction_id") or a.get("result", {}).get("id"): a
            for a in self.resources.reaction_evidence
            if isinstance(a, dict)
        }
        for art in artifacts:
            existing[art["reaction_id"]] = art
        self.resources.reaction_evidence = list(existing.values())
        return _obs(
            action,
            ok=True,
            result={"n_retrieved": len(artifacts), "reaction_ids": [a["reaction_id"] for a in artifacts]},
            evidence={"artifacts": artifacts},
            inventory=self.resources.inventory,
        )

    def select_reaction_grounded(self, action: AgentAction) -> ToolObservation:
        transport = self.resources.grounded_transport
        if transport is None:
            return _obs(
                action,
                ok=False,
                error="grounded transport not configured",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        args = parse_tool_arguments(action.tool, action.arguments)
        llm_model = getattr(args, "llm_model", "metabolm-local")
        max_output_tokens = getattr(args, "max_output_tokens", 512)
        reasoning_effort = getattr(args, "reasoning_effort", "low")
        wanted = set(action.target_ids)
        rows = []
        last_answer = None
        by_rid = {}
        for art in self.resources.reaction_evidence:
            rid = art.get("reaction_id") or art.get("result", {}).get("id")
            if rid:
                by_rid[rid] = art
        for rid in wanted:
            art = by_rid.get(rid)
            if art is None or "evidence" not in art:
                rows.append(
                    {
                        "id": rid,
                        "annotation": "",
                        "status": "input_resource_error",
                        "error": "missing retrieval evidence",
                        "update_annotation": "ignore",
                        "qualifier": "is",
                        "database": "kegg",
                        "type": "reaction",
                        "biological_abstain": None,
                    }
                )
                continue
            evidence = art["evidence"]
            body = grounded_mod.provider_payload(
                art["query"],
                evidence,
                model=llm_model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
            # Local MetaboLLM cannot satisfy OpenAI Responses model-id equality checks.
            # Wrap transport to present the requested model identity.
            raw = transport(body)
            if isinstance(raw, dict) and raw.get("model") != llm_model:
                raw = dict(raw)
                raw["model"] = llm_model
            status, answer, errors = grounded_mod.classify_response(raw, evidence, llm_model)
            row = {
                "id": rid,
                "display_name": art.get("display_name", ""),
                "annotation": "",
                "database": "kegg",
                "type": "reaction",
                "qualifier": "is",
                "update_annotation": "ignore",
                "status": status,
                "biological_abstain": None,
                "error": "; ".join(errors) or None,
            }
            if answer:
                last_answer = grounded_mod.GroundedAnnotation.model_validate(answer)
                row["biological_abstain"] = answer["abstain"]
                if not answer["abstain"]:
                    row["annotation"] = "KEGG:" + answer["predicted_kegg_id"]
                    row["update_annotation"] = "add"
                    row["match_score"] = answer["confidence"]
                row["reasoning_summary"] = answer["reasoning_summary"]
            # Never promote technical failure to biological abstention
            if status in {
                "provider_error",
                "schema_invalid",
                "incomplete",
                "empty_response",
                "evidence_noncompliant",
            }:
                row["biological_abstain"] = None
            rows.append(row)
            art["answer"] = answer
            art["result"] = dict(row)
        frame = pd.DataFrame(rows)
        if self.resources.reaction_recommendations is None:
            self.resources.reaction_recommendations = frame
        else:
            self.resources.reaction_recommendations = pd.concat(
                [self.resources.reaction_recommendations, frame], ignore_index=True
            ).drop_duplicates(subset=["id"], keep="last")
        selected = [r["id"] for r in rows if r.get("status") == "selected"]
        abstained = [r["id"] for r in rows if r.get("status") == "abstained"]
        failed = [
            r["id"]
            for r in rows
            if r.get("status") not in {"selected", "abstained"}
        ]
        kind = FailureKind.NONE
        if failed and not selected and not abstained:
            kind = FailureKind.TECHNICAL_FAILURE
        elif abstained and not selected:
            kind = FailureKind.BIOLOGICAL_ABSTENTION
        return _obs(
            action,
            ok=True,
            result={
                "selected_ids": selected,
                "abstained_ids": abstained,
                "failed_ids": failed,
                "recommendations": rows,
            },
            failure_kind=kind,
            grounded_answer=last_answer,
            inventory=self.resources.inventory,
        )

    def write_species_annotations(self, action: AgentAction) -> ToolObservation:
        frame = self.resources.species_recommendations
        if frame is None or frame.empty:
            return _obs(
                action,
                ok=False,
                error="no species recommendations to write",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        args = parse_tool_arguments(action.tool, action.arguments)
        species_out = Path(self.resources.output_dir) / "annotated_species.xml"
        if species_out.exists():
            species_out.unlink()
        path = update_annotation(
            original_model_path=self.resources.model_file,
            recommendation_table=frame,
            new_model_path=str(species_out),
            qualifier=getattr(args, "qualifier", "is"),
            add_all=bool(getattr(args, "add_all", False)),
        )
        self.resources.checkpoint["species_model_path"] = path
        return _obs(
            action,
            ok=True,
            result={"path": path},
            inventory=self.resources.inventory,
        )

    def write_reaction_annotations_tool(self, action: AgentAction) -> ToolObservation:
        frame = self.resources.reaction_recommendations
        if frame is None or frame.empty:
            return _obs(
                action,
                ok=False,
                error="no reaction recommendations to write",
                failure_kind=FailureKind.TECHNICAL_FAILURE,
                inventory=self.resources.inventory,
                skip_semantic=True,
            )
        source = self.resources.checkpoint.get("species_model_path") or self.resources.model_file
        out = Path(self.resources.output_dir) / "annotated_model.xml"
        if out.exists():
            out.unlink()
        writable = frame.copy()
        path = write_reaction_annotations(source, writable, str(out))
        self.resources.checkpoint["annotated_model_path"] = path
        return _obs(
            action,
            ok=True,
            result={"path": path},
            inventory=self.resources.inventory,
        )

    def persist_checkpoint(self, action: AgentAction) -> ToolObservation:
        path = Path(self.resources.output_dir) / "checkpoint.json"
        payload = {
            "entity_identities": self.resources.entity_identities,
            "checkpoint": self.resources.checkpoint,
            "species_recommendations": None
            if self.resources.species_recommendations is None
            else self.resources.species_recommendations.fillna("").to_dict("records"),
            "reaction_recommendations": None
            if self.resources.reaction_recommendations is None
            else self.resources.reaction_recommendations.fillna("").to_dict("records"),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return _obs(
            action,
            ok=True,
            result={"path": str(path)},
            inventory=self.resources.inventory,
        )

    def export_reports_placeholder(self, action: AgentAction) -> ToolObservation:
        # Actual export is performed by export.py after the loop; this tool marks intent.
        return _obs(
            action,
            ok=True,
            result={"queued": True},
            inventory=self.resources.inventory,
        )
