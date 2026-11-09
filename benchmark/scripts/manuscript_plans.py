"""No-call species frontier and matched grounded plans with existing budget guards."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd
import libsbml

from benchmark.scripts.manuscript_propagation import (read_json, read_jsonl, write_json, write_rows,
    write_csv, canonical, digest, sha256, load_model, network_identity)
from core.reaction.grounded import provider_payload


def load_luna_runner(luna_root):
    """Explicitly reuse the preserved runner, without importing any live adapter."""
    import benchmark.scripts
    for name in ["luna_model250728_analysis", "luna_output_budget"]:
        full_name = "benchmark.scripts." + name
        path = Path(luna_root) / "benchmark/scripts" / (name + ".py")
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        setattr(benchmark.scripts, name, module)
        spec.loader.exec_module(module)
    return sys.modules["benchmark.scripts.luna_output_budget"]


def write_runner_plan(directory, rows, runner):
    directory.mkdir(parents=True, exist_ok=False)
    requests, manifest = [], []
    for row in rows:
        body, unit = row["provider_payload"], row["unit"]
        key = runner.original.unit_cache_key(unit, body["model"], runner.original.digest(body))
        requests.append({"unit": unit, "provider_payload": body, "unit_cache_key": key})
        manifest.append({"unit": unit, "unit_cache_key": key, "provider_payload_sha256": runner.original.digest(body),
                         "reservation": runner.reservation(body)})
    write_rows(directory / "requests.jsonl", requests)
    write_rows(directory / "request_manifest.jsonl", manifest)
    total = sum(row["reservation"]["reservation_nusd"] for row in manifest)
    cap = math.ceil(total / 1e9) * 1_000_000_000
    plan = {"schema": "species-propagation-grounded-arm-v1", "request_count": len(requests),
        "maximum_attempts": len(requests),
        "model": "gpt-6.1-terra", "reasoning_effort": "low", "max_output_tokens": 2048,
        "reservation_nusd": total, "hard_cap_nusd": cap, "authorized_spend_nusd": 0,
        "requests_sha256": sha256(directory / "requests.jsonl"),
        "request_manifest_sha256": sha256(directory / "request_manifest.jsonl"),
        "execution_status": "not authorized; require exact arm/cohort/plan approval", "automatic_retries": 0}
    runner.verify_plan(plan, requests, manifest)
    write_json(directory / "plan.json", plan)
    identity = runner.original.digest(plan)
    write_json(directory / "plan_identity.json", {"plan_digest": identity})
    # Constructor verifies complete frozen request/manifest identities. No reserve/run call.
    state = runner.RunState(directory, directory / "_dry_run_state")
    coverage = state.coverage()
    if len(coverage) != len(requests) or any(row["status"] != "not_attempted" for row in coverage):
        raise ValueError("dry-run runner coverage mutated attempts")
    return {**plan, "plan_digest": identity, "all_retained_reservations_fit": total <= cap,
            "existing_runner_compatible": True, "provider_calls": 0}


def prepare_species_frontier(study, output):
    from core.annotation_workflow import _normalization_response_format
    from core.llm_interface import get_system_prompt
    from core.model_info import format_prompt
    from utils.constants import EntityType
    output.mkdir(parents=True, exist_ok=False)
    (output / "sanitized_sbml").mkdir()
    queries = read_jsonl(Path(study) / "queries.jsonl")
    targets = {}
    for row in queries:
        if row["condition"] == "reference":
            targets.setdefault(row["model_id"], set()).update(row["species_evidence"])
    alignment = read_json(Path(study) / "alignment.json")
    requests, units, source_records = [], [], []
    for info in alignment["validation_models"]:
        document = load_model(info.get("query_source_path", info["path"]))
        model = document.getModel()
        before = network_identity(model)
        model.unsetAnnotation()
        model.unsetNotes()
        for entity in list(model.getListOfSpecies()) + list(model.getListOfReactions()):
            entity.unsetAnnotation()
            entity.unsetCVTerms()
            entity.unsetNotes()
        if network_identity(model) != before:
            raise ValueError("sanitization changed names/structure")
        path = output / "sanitized_sbml" / (info["model_id"] + ".xml")
        if not libsbml.writeSBMLToFile(document, str(path)):
            raise OSError(path)
        ids = sorted(targets[info["model_id"]])
        inverse = {alias: sid for sid, alias in info.get("species_aliases", {}).items()}
        source_records.append({"model_id": info["model_id"], "source_sha256": info["sbml_sha256"],
            "sanitized_sha256": sha256(path), "participant_species": len(ids), "all_source_species": info["n_species"]})
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            body = {"model": "gpt-5-mini-2025-08-07", "messages": [
                {"role": "system", "content": get_system_prompt(EntityType.CHEMICAL)},
                {"role": "user", "content": format_prompt(str(path), chunk, EntityType.CHEMICAL, context=False, structured_output=True)}],
                "max_completion_tokens": 10000, "response_format": _normalization_response_format(EntityType.CHEMICAL)}
            unit = {"model_id": info["model_id"], "stage": "normalization", "chunk": i // 50,
                    "species_ids": chunk}
            requests.append({"unit": unit, "provider_payload": body, "provider_payload_sha256": digest(body),
                             "cache_key": digest({"endpoint": "/v1/chat/completions", "body": body, "unit": unit})})
        units.extend({"model_id": info["model_id"], "species_id": sid,
                      "original_source_species_id": inverse.get(sid, sid)} for sid in ids)
    write_rows(output / "normalization_requests.jsonl", requests)
    write_csv(output / "target_species.csv", units)
    write_json(output / "sanitization_manifest.json", source_records)
    # Actual complete frontier bodies; downstream prompts depend on unknown responses.
    reservations = []
    for row in requests:
        bound = len(canonical(row["provider_payload"])) + 4096
        nanos = bound * 250 + 10000 * 2000
        reservations.append({"unit": row["unit"], "input_byte_token_bound": bound,
                             "max_total_output_tokens": 10000, "reservation_nusd": nanos, "reservation_usd": nanos / 1e9})
    write_rows(output / "normalization_reservations.jsonl", reservations)
    result = {"stage": "missing actual species predictions", "model": "gpt-5-mini-2025-08-07",
        "basis": "same species model as recorded Luna machine runs; no model variant comparison",
        "target_species": len(units), "models": len(source_records), "frontier_calls": len(requests),
        "frontier_reservation_nusd": sum(row["reservation_nusd"] for row in reservations),
        "frontier_reservation_usd": sum(row["reservation_nusd"] for row in reservations) / 1e9,
        "frontier_proposed_cap_usd": math.ceil(sum(row["reservation_nusd"] for row in reservations) / 1e7) / 100,
        "configuration": {"method": "direct", "entity_type": "chemical", "database": "chebi", "top_k": 3,
            "n_return": 1, "chunk_size": 50, "validation": True, "source_fidelity": True,
            "predict_qualifiers": True, "effort": "low", "context": False, "max_completion_tokens": 10000},
        "differences_from_legacy_runs": "reference annotations/notes removed and context disabled to prevent contamination; current separate support/qualifier stage enabled",
        "uniform_species_id_aliasing": read_json(Path(study) / "protocol.json").get("uniform_species_id_aliases", False),
        "alignment_policy": "map normalization output aliases back to original source IDs using the frozen bijection before joining actual predicted evidence; preserve names and stoichiometry",
        "conditional_stages": ["database candidates depend on normalization responses", "ranking request bodies depend on retrieved candidates",
            "identity-support/qualifier requests depend on selected identities; no invented supported statuses"],
        "full_species_cost_status": "frontier exact; later bodies/count/cost unfreezable until responses. No full-generation cost guarantee.",
        "executor_status": "existing public species path has retries/semantic repair; must adapt persistent guards and disable all automatic repurchases before authorization",
        "authorized_calls": 0, "authorized_spend_usd": 0,
        "normalization_requests_sha256": sha256(output / "normalization_requests.jsonl"), "balance": "unverified; no old $20 credit assumption"}
    write_json(output / "generation_plan.json", result)
    write_json(output / "plan_identity.json", {"plan_digest": digest(result)})
    return result


def build(study, luna_root, pricing_path, out):
    study, out = Path(study), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    prices = read_json(pricing_path)
    if prices["per_million_tokens"]["gpt-6.1-terra"]["implicit_30m_cache_write"] != 1.25:
        raise ValueError("runner pricing needs revalidation")
    runner = load_luna_runner(luna_root)
    rows = read_jsonl(study / "queries.jsonl")
    evidence = {(r["condition"], r["model_id"], r["reaction_id"]): r["records"] for r in read_jsonl(study / "evidence.jsonl")}
    prepared = {}
    pending = []
    for row in rows:
        condition = row["condition"]
        body = provider_payload(row["query"], evidence[(condition, row["model_id"], row["reaction_id"])],
                                model="gpt-6.1-terra", reasoning_effort="low", max_output_tokens=2048)
        unit = {"model_id": row["model_id"], "reaction_id": row["reaction_id"], "condition": condition}
        if len(canonical(body)) > prices["request_byte_cap"]:
            pending.append({"unit": unit, "status": "oversize_input_requires_protocol_decision", "bytes": len(canonical(body))})
        else:
            prepared.setdefault(condition, []).append({"unit": unit, "provider_payload": body})
    arms = {condition: write_runner_plan(out / condition, values, runner) for condition, values in prepared.items()}
    cohort = read_jsonl(study / "cohort.jsonl")
    if "predicted" not in arms:
        pending.extend({"unit": {"model_id": row["model_id"], "reaction_id": row["reaction_id"], "condition": "predicted"},
                        "status": "pending_species_evidence", "provider_payload": None} for row in cohort)
    write_rows(out / "pending_units.jsonl", pending)
    species = prepare_species_frontier(study, out / "species_generation")
    # Upper bound for as-yet-unknown B bodies under a predeclared 32-KiB cap.
    max_b_usd = ((32768 + 4096) * 1250 + 2048 * 5000) / 1e9
    current = sum(arm["reservation_nusd"] for arm in arms.values()) / 1e9
    identities = []
    for condition in prepared:
        for row in prepared[condition]:
            identities.append({"unit": row["unit"], "payload_sha256": digest(row["provider_payload"])})
    by_reaction = {}
    for row in identities:
        unit = row["unit"]
        by_reaction.setdefault((unit["model_id"], unit["reaction_id"]), {})[unit["condition"]] = row["payload_sha256"]
    identical = sum(len(set(values.values())) == 1 and len(values) > 1 for values in by_reaction.values())
    proposal = {"arms": arms, "matched_ABC_final_annotation_results": "pending; no response generated",
        "ready_requests": sum(arm["request_count"] for arm in arms.values()), "pending_units": len(pending),
        "ready_arm_reservation_usd": current, "ready_arm_proposed_caps_usd": sum(a["hard_cap_nusd"] for a in arms.values()) / 1e9,
        "predicted_arm_upper_bound_usd_if_every_body_fits_32768_bytes": max_b_usd * len(cohort),
        "three_arm_reservation_envelope_usd": current + max_b_usd * len(cohort),
        "identical_complete_payloads_within_reaction_across_ready_arms": identical,
        "cross_arm_response_reuse": "not automatic; only exact body and provider/cohort compatibility; identical cases are not independent observations",
        "historical_gpt56_cache_hits": 0, "existing_gpt61_response_hits": 0,
        "species_frontier_separate_cost": species, "pricing_snapshot": prices,
        "balance_verified": False, "authorized_calls": 0, "authorized_spend_usd": 0,
        "runner": {"path": str(Path(luna_root) / "benchmark/scripts/luna_output_budget.py"),
            "sha256": sha256(Path(luna_root) / "benchmark/scripts/luna_output_budget.py"),
            "compatibility": "existing RunState and verifier accept each single-condition plan with actual model/reaction IDs; separate state per arm avoids duplicate-unit collision",
            "remaining_work": "ABC interleaving/provider cohort controller and B plan after species generation; species frontier needs a Chat Completions guard adapter"}}
    write_json(out / "execution_proposal.json", proposal)
    write_json(out / "proposal_identity.json", {"digest": digest(proposal)})
    return {key: value for key, value in proposal.items() if key not in {"species_frontier_separate_cost", "pricing_snapshot", "arms"}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "luna-root", "pricing", "out"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.study, args.luna_root, args.pricing, args.out), indent=2))
