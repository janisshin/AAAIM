"""Annotated-copy export and report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import libsbml
import pandas as pd

from core.agent.io import atomic_write_json
from core.agent.schemas import (
    AnnotationPlan,
    AutoAnnotateReport,
    EntityRuntimeState,
    ModelInventory,
    TaskState,
    ToolObservation,
)
from core.reaction.writer import write_reaction_annotations
from core.update_model import update_annotation


def _write_json(path: Path, payload: Any) -> str:
    return atomic_write_json(payload, path)


def validate_sbml_file(path: str) -> tuple[bool, str]:
    if not Path(path).is_file():
        return False, "missing file"
    doc = libsbml.SBMLReader().readSBML(path)
    if doc.getModel() is None:
        return False, "no model"
    return True, "ok"


def build_change_table(
    species_df: Optional[pd.DataFrame],
    reaction_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    if species_df is not None:
        for row in species_df.fillna("").to_dict("records"):
            rows.append(
                {
                    "target_id": row.get("id"),
                    "entity_kind": "species",
                    "annotation": row.get("annotation"),
                    "qualifier": row.get("qualifier", "is"),
                    "action": row.get("update_annotation", "ignore"),
                    "status": row.get("status") or row.get("identity_status"),
                }
            )
    if reaction_df is not None:
        for row in reaction_df.fillna("").to_dict("records"):
            rows.append(
                {
                    "target_id": row.get("id"),
                    "entity_kind": "reaction",
                    "annotation": row.get("annotation"),
                    "qualifier": row.get("qualifier", "is"),
                    "action": row.get("update_annotation", "ignore"),
                    "status": row.get("status"),
                }
            )
    return pd.DataFrame(rows)


def export_artifacts(
    *,
    output_dir: str | Path,
    model_file: str,
    inventory: ModelInventory,
    initial_plan: Optional[AnnotationPlan],
    executed_plan: Optional[AnnotationPlan],
    entity_states: dict[str, EntityRuntimeState],
    observations: list[ToolObservation],
    species_df: Optional[pd.DataFrame],
    reaction_df: Optional[pd.DataFrame],
    provenance: dict[str, Any],
    interrupt: bool = False,
) -> AutoAnnotateReport:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    initial_plan_path = _write_json(
        out / "initial_plan.json",
        initial_plan.model_dump(mode="json") if initial_plan else {},
    )
    executed_plan_path = _write_json(
        out / "executed_plan.json",
        executed_plan.model_dump(mode="json") if executed_plan else {},
    )
    provenance_path = _write_json(out / "provenance.json", provenance)
    _write_json(out / "inventory.json", inventory.model_dump(mode="json"))
    _write_json(
        out / "observations.jsonl",
        # store as list for simplicity; also write jsonl
        [o.model_dump(mode="json") for o in observations],
    )
    with (out / "observations.jsonl").open("w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o.model_dump(mode="json"), sort_keys=True) + "\n")

    change_df = build_change_table(species_df, reaction_df)
    change_table_path = str(out / "annotation_changes.csv")
    change_df.to_csv(change_table_path, index=False)

    recommendations_path = str(out / "recommendations.json")
    recommendations_payload = {
        "species": None if species_df is None else species_df.fillna("").to_dict("records"),
        "reactions": None if reaction_df is None else reaction_df.fillna("").to_dict("records"),
    }
    _write_json(Path(recommendations_path), recommendations_payload)
    if species_df is not None:
        species_df.to_csv(out / "species_recommendations.csv", index=False)
    if reaction_df is not None:
        reaction_df.to_csv(out / "reaction_recommendations.csv", index=False)

    completed, unresolved, skipped, failed = [], [], [], []
    technical_errors, biological_reservations, recommendations = [], [], []
    # Prefer entity_states; if a plan task is terminal and entity is still PENDING,
    # align the report with the non-PENDING plan state.
    plan_states: dict[str, TaskState] = {}
    if executed_plan is not None:
        for task in executed_plan.tasks:
            prev = plan_states.get(task.target_id)
            if prev is None or (
                prev in {TaskState.PENDING, TaskState.RUNNING}
                and task.state not in {TaskState.PENDING, TaskState.RUNNING}
            ):
                plan_states[task.target_id] = task.state

    report_ids = set(entity_states) | set(plan_states)
    for tid in report_ids:
        state = entity_states.get(tid)
        effective = state.state if state is not None else TaskState.PENDING
        plan_state = plan_states.get(tid)
        if (
            plan_state is not None
            and effective in {TaskState.PENDING, TaskState.RUNNING}
            and plan_state not in {TaskState.PENDING, TaskState.RUNNING}
        ):
            effective = plan_state
        if effective == TaskState.COMPLETED:
            completed.append(tid)
        elif effective == TaskState.UNRESOLVED:
            unresolved.append(tid)
            notes = state.notes if state is not None else []
            kind = state.failure_kind.value if state is not None else ""
            biological_reservations.append(f"{tid}: {kind} {'; '.join(notes)}".strip())
        elif effective == TaskState.SKIPPED:
            skipped.append(tid)
        elif effective == TaskState.FAILED:
            failed.append(tid)
            notes = state.notes if state is not None else []
            technical_errors.append(f"{tid}: {' '.join(notes)}".strip())
        elif effective == TaskState.PENDING:
            unresolved.append(tid)

    for obs in observations:
        if not obs.ok and obs.error:
            technical_errors.append(f"{obs.tool}/{obs.action_id}: {obs.error}")
        if obs.failure_kind.value == "biological_abstention":
            biological_reservations.append(f"{obs.action_id}: biological abstention")

    annotated_model_path = None
    write_errors: list[str] = []
    try:
        source = model_file
        species_path = out / "_tmp_species.xml"
        final_path = out / "annotated_model.xml"
        wrote_any = False
        if species_df is not None and not species_df.empty:
            if species_path.exists():
                species_path.unlink()
            update_annotation(
                original_model_path=source,
                recommendation_table=species_df,
                new_model_path=str(species_path),
                add_all=False,
            )
            source = str(species_path)
            wrote_any = True
        if reaction_df is not None and not reaction_df.empty:
            writable = reaction_df.copy()
            # Only rows marked add+selected are written by the reaction writer.
            has_add = any(
                str(r.get("update_annotation")) == "add" and str(r.get("status")) == "selected"
                for r in writable.fillna("").to_dict("records")
            )
            if has_add:
                if final_path.exists():
                    final_path.unlink()
                write_reaction_annotations(source, writable, str(final_path))
                annotated_model_path = str(final_path)
                wrote_any = True
            elif wrote_any:
                # Species-only annotated copy
                if final_path.exists():
                    final_path.unlink()
                Path(source).replace(final_path) if Path(source) != final_path else None
                if not final_path.exists() and Path(source).exists():
                    import shutil

                    shutil.copy2(source, final_path)
                annotated_model_path = str(final_path)
        elif wrote_any:
            import shutil

            if final_path.exists():
                final_path.unlink()
            shutil.copy2(source, final_path)
            annotated_model_path = str(final_path)

        if annotated_model_path:
            ok, msg = validate_sbml_file(annotated_model_path)
            if not ok:
                write_errors.append(f"output SBML invalid: {msg}")
                annotated_model_path = None
                recommendations.append("Writing failed validation; recommendations retained without annotated file claim")
    except Exception as exc:
        write_errors.append(f"{type(exc).__name__}: {exc}")
        annotated_model_path = None
        recommendations.append(
            "Annotation writing failed; returning recommendations/report without claiming an annotated file"
        )

    technical_errors.extend(write_errors)
    partial = bool(unresolved or failed or skipped or write_errors or interrupt)
    if interrupt:
        status = "interrupted"
    elif failed and not completed:
        status = "failed"
    elif partial:
        status = "partial"
    else:
        status = "completed"

    if partial:
        recommendations.append(
            "Result is partial: not all requested entities reached a successful annotated state"
        )
    recommendations.append(
        "Default scope preserves existing annotations and only adds missing ones within requested tasks"
    )
    recommendations.append(
        "Pydantic schema validation and semantic checks (target existence, dependencies, "
        "evidence consistency) do not establish biological correctness of annotations"
    )

    report = AutoAnnotateReport(
        status=status,
        annotated_model_path=annotated_model_path,
        change_table_path=change_table_path,
        initial_plan_path=initial_plan_path,
        executed_plan_path=executed_plan_path,
        recommendations_path=recommendations_path,
        provenance_path=provenance_path,
        completed_ids=sorted(set(completed)),
        unresolved_ids=sorted(set(unresolved)),
        skipped_ids=sorted(set(skipped)),
        failed_ids=sorted(set(failed)),
        technical_errors=technical_errors,
        biological_reservations=biological_reservations,
        recommendations=recommendations,
        partial=partial,
        resume_info={
            "output_dir": str(out),
            "checkpoint_path": str(out / "checkpoint.json"),
            "compatible_cache_identity": provenance.get("cache_identity"),
        },
        schema_validated=True,
        biological_correctness_certified=False,
    )
    _write_json(out / "report.json", report.model_dump(mode="json"))
    return report
