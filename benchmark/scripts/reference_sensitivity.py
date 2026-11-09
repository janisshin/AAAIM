"""Offline, private, single-reviewer reference sensitivity; never alter a source.

Only explicit reference-proposal fields authorize set changes. Output-assessment
fields and taxonomy labels never supply replacement IDs. Conflicting statements
can suspend a proposal through a checksum-bound, quote-verified hold record.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
import lzma
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from benchmark.scripts.manuscript_propagation import (
    read_json, read_jsonl, write_json, write_rows, write_csv, sha256, digest)

VERSION = "single-reviewer-review-informed-v1-20261109"
POLICY = {
    "name": "explicit-pass1-proposals-with-conflict-holds-v1",
    "replacement": "label_incorrect + explicit proposed_kegg_ids + rationale; reject the original set only under that explicit verdict",
    "addition": "label_correct_but_incomplete or multiple_defensible_annotations + explicit proposed_kegg_ids + rationale; union with originals",
    "pass2": "prediction assessments, defensible_prediction_ids, abstention assessments and taxonomy never authorize reference changes; unstructured label-action text remains a proposal needing explicit set specification",
    "holds": "quote-verified contradictory reference statements suspend changes; no agent chooses a biological interpretation",
    "non_mappable": "flag and retain original set in primary fixed-denominator analysis; do not manufacture empty gold sets",
    "coverage": "all 163 pilot reactions; reviewed 60, unreviewed 103 retain unverified originals; no extrapolation to 969",
    "review_scope": "one reviewer/two stages; not independent adjudication or a fully corrected gold standard",
}


def key(row):
    return row["model_id"], row["reaction_id"]


def unique(rows, description):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {description} reaction keys")
    return result


def parse_proposal(text):
    """Accept only an explicit ID list, never scrape IDs from rationale prose."""
    parts = [p for p in re.split(r"[;,\s]+", text.strip()) if p]
    if any(not re.fullmatch(r"R\d{5}", part) for part in parts):
        raise ValueError("reference proposal is not an explicit KEGG ID list")
    return sorted(set(parts))


def derive_change(pass1, pass2, original, hold=None):
    original = sorted(set(original))
    if not original:
        raise ValueError("original accepted set must be nonempty")
    proposal = parse_proposal(pass1["proposed_kegg_ids"])
    verdict = pass1["verdict"]
    operation, revised, reason = "none", original, "no explicit actionable reference revision"
    if proposal and pass1["biological_rationale"].strip():
        if verdict == "label_incorrect":
            operation, revised, reason = "replace", proposal, "explicit Pass 1 rejection and proposed replacement set"
        elif verdict in {"label_correct_but_incomplete", "multiple_defensible_annotations"}:
            operation, revised, reason = "add", sorted(set(original) | set(proposal)), "explicit Pass 1 additional accepted alternatives; original retained"
    if hold:
        for evidence in hold["evidence"]:
            review = pass1 if evidence["pass"] == 1 else pass2
            if review[evidence["field"]] != evidence["recorded_text"]:
                raise ValueError("hold quotation no longer matches authoritative export")
        revised, operation, reason = original, "hold", hold["reason"]
    elif revised == original:
        operation = "none"
    if verdict == "not_kegg_mappable":
        change_type = "non_mappability"
    elif verdict == "multiple_defensible_annotations":
        change_type = "ambiguity"
    elif verdict == "label_correct_but_incomplete":
        change_type = "incomplete_reference"
    elif verdict == "label_incorrect":
        change_type = "incorrect_reference"
    elif verdict in {"insufficient_model_information", "catalog_snapshot_issue", "unresolved"}:
        change_type = "insufficient_information"
    else:
        change_type = "no_reference_issue"
    actionable = revised != original
    return {"audit_id": pass1["audit_id"], "original_reference_ids": original,
        "explicit_pass1_proposed_ids": proposal, "revised_reference_ids": revised,
        "operation": operation, "change_type": change_type, "actionable": actionable,
        "source_pass": "Pass 1" if actionable else None, "decision_reason": reason,
        "reference_issue_unresolved": change_type != "no_reference_issue" and not actionable,
        "pass2_reference_action_text": pass2.get("recommended_label_action", ""),
        "pass2_exact_matching_flag": pass2.get("exact_matching_too_strict", ""),
        "pass2_system_information_flag": pass2.get("prediction_assessment") == "insufficient_information",
        "hold": hold}


VALIDATOR = r'''
import json,sys
from pathlib import Path
from benchmark.scripts import phase3_error_analysis as a,phase3_error_reviewer as r
root=Path(sys.argv[1]); candidates=json.loads(sys.argv[2]); results=[]
bundles=[a.load(r.OUT/f'formal_pass{i}_data.json') for i in (1,2)]
for path in candidates:
 p=a.load(Path(path)); phase=1 if p.get('review_kind')=='formal_pass1' else 2
 results.append({'path':path,'phase':phase,'errors':a.validate(p,bundles[phase-1],phase)})
from benchmark.scripts import phase3_error_inventory as inv,phase3_error_sampling as sam
verifiers={'inventory':inv.verify_manifest(),'sampling':sam.verify_manifest(),'reviewer':r.verify_manifest()}
assert not any(verifiers.values()),verifiers
assert all(r._digest({k:v for k,v in b.items() if k!='dataset_digest'})==b['dataset_digest'] for b in bundles)
print(json.dumps({'candidates':results,'manifest_verifiers':verifiers,'dataset_digests':[b['dataset_digest'] for b in bundles]}))
'''


def select_exports(source_root):
    audit = Path(source_root) / "benchmark/phase3/error_audit"
    paths = sorted({p.resolve() for parent in [audit / "review_work", audit / "reviewer"]
                    for phase in [1, 2] for p in parent.rglob(f"pass{phase}_review_*.json")})
    result = subprocess.run([sys.executable, "-c", VALIDATOR, str(source_root), json.dumps(list(map(str, paths)))],
        cwd=source_root, check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    valid = [Path(r["path"]) for r in report["candidates"] if not r["errors"]]
    pairs = []
    for p2 in valid:
        b = read_json(p2)
        if b.get("review_kind") != "formal_pass2":
            continue
        for p1 in valid:
            a = read_json(p1)
            if a.get("review_kind") == "formal_pass1" and b.get("locked_pass1_export") == a:
                pairs.append((b["export_timestamp"], sha256(p1), sha256(p2), p1, p2))
    if not pairs:
        raise ValueError("no validated complete matching export pair")
    latest = max(r[0] for r in pairs)
    newest = [r for r in pairs if r[0] == latest]
    if len({r[1:3] for r in newest}) != 1:
        raise ValueError("ambiguous distinct authoritative pairs with identical latest timestamp")
    chosen = min(newest, key=lambda r: ("review_work" not in r[3].parts, str(r[3]), str(r[4])))
    report.update(selected_pass1=str(chosen[3]), selected_pass2=str(chosen[4]),
                  selected_timestamp=latest, embedded_pass1_identical=True)
    return chosen[3], chosen[4], report


def apply_layer(original, changes):
    view = {k: sorted(set(v)) for k, v in original.items()}
    seen = set()
    for row in changes:
        k = key(row)
        if k in seen or k not in view:
            raise ValueError("duplicate or unknown correction key")
        seen.add(k)
        if row["original_reference_ids"] != view[k]:
            raise ValueError("correction original-set mismatch")
        proposed = row["revised_reference_ids"]
        if not proposed:
            raise ValueError("fixed-denominator layer cannot empty an accepted set")
        if not row["actionable"] and proposed != view[k]:
            raise ValueError("nonactionable correction changed a set")
        if row["actionable"]:
            explicit = row["explicit_pass1_proposed_ids"]
            allowed = (explicit if row["operation"] == "replace" else
                       sorted(set(view[k]) | set(explicit)) if row["operation"] == "add" else None)
            if proposed != allowed:
                raise ValueError("revision does not follow its explicit replacement/addition operation")
        view[k] = proposed
    return view


def score_assertion(prediction, reference):
    # The response state is immutable across reference versions.
    return {**prediction, "correct": bool(prediction["answered"] and prediction["predicted_id"] in reference)}


def final_summary(rows):
    n, answered = len(rows), sum(r["answered"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    models = sorted({r["model_id"] for r in rows})
    return {"n": n, "n_models": len(models), "n_clusters": len({r["cluster_id"] for r in rows}),
        "correct": correct, "answered": answered, "agreement": correct / n,
        "coverage": answered / n, "correct_over_answered": correct / answered if answered else None,
        "biological_abstentions": sum(r["status"] == "abstained" for r in rows),
        "no_candidate": sum(r["status"] == "no_candidate" for r in rows),
        "technical_failures": sum(r["status"] not in {"selected", "abstained", "no_candidate"} for r in rows),
        "incomplete_outputs": sum(r["status"] == "incomplete" for r in rows),
        "model_macro_agreement": float(np.mean([np.mean([r["correct"] for r in rows if r["model_id"] == m]) for m in models]))}


def retrieval_summary(rows):
    """Stable accepted-set aggregates across row order and Python hash seeds."""
    ordered = sorted(unique(rows, "retrieval summary").values(), key=key)
    if not ordered:
        raise ValueError("empty retrieval summary")
    result = {}
    for col in ["recall_at_1_exact", "recall_at_3_exact", "recall_at_5_exact", "recall_at_10_exact", "mrr_at_10_exact"]:
        result[col] = math.fsum(r[col] for r in ordered) / len(ordered)
        if col != "mrr_at_10_exact":
            result[col + "_count"] = sum(bool(r[col]) for r in ordered)
    return result


def paired_comparison(a, b, *, replicates=2000, seed=20261109):
    aa, bb = unique(a, "paired A"), unique(b, "paired B")
    if aa.keys() != bb.keys() or any(aa[k]["cluster_id"] != bb[k]["cluster_id"] for k in aa):
        raise ValueError("paired population or cluster mismatch")
    clusters = sorted({r["cluster_id"] for r in a})
    totals = np.array([[sum(int(bb[k]["correct"]) - int(aa[k]["correct"]) for k in aa if aa[k]["cluster_id"] == c),
                        sum(aa[k]["cluster_id"] == c for k in aa)] for c in clusters])
    draw = np.random.default_rng(seed).integers(0, len(clusters), size=(replicates, len(clusters)))
    sample = totals[draw].sum(axis=1)
    return {"n": len(a), "n_clusters": len(clusters), "direction": "B minus A", "delta_count": int(totals[:, 0].sum()),
        "delta": float(totals[:, 0].sum() / len(a)), "ci95": list(map(float, np.quantile(sample[:, 0] / sample[:, 1], [.025, .975]))),
        "seed": seed, "replicates": replicates, "unit": "paired frozen cluster", "transitions": transitions(a, b)}


def transitions(a, b):
    aa, bb = unique(a, "transition A"), unique(b, "transition B")
    if aa.keys() != bb.keys():
        raise ValueError("transition populations differ")
    counts = {"correct_correct": 0, "correct_incorrect": 0, "incorrect_correct": 0, "incorrect_incorrect": 0}
    for k in aa:
        counts[("correct" if aa[k]["correct"] else "incorrect") + "_" + ("correct" if bb[k]["correct"] else "incorrect")] += 1
    return counts


def build(source_root, holds_path, out):
    source_root, holds_path, out = map(lambda p: Path(p).resolve(), [source_root, holds_path, out])
    if out.exists():
        raise FileExistsError("immutable version output exists; choose a fresh destination")
    local = Path(__file__).resolve().parents[2]
    if subprocess.run(["git", "check-ignore", "--quiet", str(out)], cwd=local).returncode:
        raise ValueError("personal correction and score reports must be ignored by Git")
    inputs = {}
    def track(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha256(path)
        return path
    audit, pilot = source_root / "benchmark/phase3/error_audit", source_root / "benchmark/phase3/phase3c_validation"
    for name in ["benchmark/phase3/error_audit/artifact_manifest.json",
                 "benchmark/phase3/error_audit/reviewer/artifact_manifest.json",
                 "benchmark/phase3/retrieval_baselines/artifact_manifest.json",
                 "benchmark/phase3/phase3b_fusion/artifact_manifest.json",
                 "benchmark/phase3/phase3c_smoke/artifact_manifest.json",
                 "benchmark/phase3/phase3c_validation/artifact_manifest.json"]:
        track(source_root / name)
    p1path, p2path, validation = select_exports(source_root)
    p1, p2 = [read_json(track(path)) for path in [p1path, p2path]]
    # Compare old answers mechanically. No case-specific target answer is imposed.
    prior = audit / "review_work/analysis_20261105/joined_cases.csv"
    prior_changes = []
    if prior.is_file():
        old = list(csv.DictReader(track(prior).open(encoding="utf-8-sig")))
        old_map = {r["audit_id"]: json.loads(r["pass2_review"]) for r in old}
        if len(old_map) != 60 or set(old_map) != {r["audit_id"] for r in p2["reviews"]}:
            raise ValueError("prior-analysis case identities differ")
        for row in p2["reviews"]:
            for field in sorted(set(row) | set(old_map[row["audit_id"]])):
                before = old_map[row["audit_id"]].get(field)
                if before != row.get(field):
                    prior_changes.append({"audit_id": row["audit_id"], "field": field, "before": before, "after": row.get(field)})
    validation["prior_analysis_pass2_field_differences"] = prior_changes
    validation["prior_analysis_available"] = prior.is_file()
    holds = read_json(track(holds_path))
    if holds["source_pass1_sha256"] != sha256(p1path) or holds["source_pass2_sha256"] != sha256(p2path):
        raise ValueError("holds refer to a different review session")
    hold_map = {r["audit_id"]: r for r in holds["holds"]}
    if len(hold_map) != len(holds["holds"]):
        raise ValueError("duplicate hold")
    register = list(csv.DictReader(track(audit / "formal_audit_crosswalk.csv").open(encoding="utf-8-sig")))
    cross = {r["audit_id"]: r for r in register}
    if len(register) != 60 or len(cross) != 60:
        raise ValueError("invalid reviewed crosswalk")
    if len(cross) != 60 or set(cross) != {r["audit_id"] for r in p1["reviews"]} or not set(hold_map) <= set(cross):
        raise ValueError("formal sample alignment mismatch")
    validation["remaining_abstention_vs_prediction_assessment_inconsistencies"] = [
        r["audit_id"] for r in p2["reviews"] if cross[r["audit_id"]]["phase3c_abstain"] == "true"
        and r["prediction_assessment"].startswith("prediction_")]
    validation["reference_statement_conflict_holds"] = sorted(hold_map)
    bundles = [read_json(track(audit / f"reviewer/formal_pass{i}_data.json")) for i in [1, 2]]
    bundle = {r["audit_id"]: r for r in bundles[0]["cases"]}
    p2by = {r["audit_id"]: r for r in p2["reviews"]}
    answers = read_jsonl(track(pilot / "answer_key_join.jsonl"))
    population = unique(answers, "pilot")
    if len(population) != 163:
        raise ValueError("expected frozen 163-case pilot")
    originals = {k: sorted(row["ground_truth_ids"]) for k, row in population.items()}
    catalog = set(read_json(track(source_root / "benchmark/phase3/kegg_catalog_ids.json"))["ids"])
    changes = []
    for i, review in enumerate(p1["reviews"]):
        aid, cw = review["audit_id"], cross[review["audit_id"]]
        k = key(cw)
        if k not in originals or bundle[aid]["sample_id"] != cw["sample_id"] or key(bundle[aid]) != k:
            raise ValueError("review/model/reaction/sample alignment mismatch")
        if sorted(r["kegg_id"] for r in bundle[aid]["existing_labels"]) != originals[k]:
            raise ValueError("reviewed original references differ from pilot")
        row = derive_change(review, p2by[aid], originals[k], hold_map.get(aid))
        if not set(row["explicit_pass1_proposed_ids"]) <= catalog:
            raise ValueError("explicit replacement absent from frozen catalog; requires unresolved handling")
        row.update({field: cw[field] for field in ["model_id", "reaction_id", "sample_id", "cluster_id", "primary_category", "corrected_phase2_stratum", "retrieval_state"]})
        row.update(pass1_reference_judgment=review, pass2_reference_context={name: p2by[aid].get(name) for name in [
            "recommended_label_action", "exact_matching_too_strict", "equivalence_relationship", "biological_rationale", "supporting_sources", "reviewer_notes"]},
            reference_evidence={"model_reaction": bundle[aid]["model_reaction"], "existing_label_provenance": bundle[aid]["existing_label_provenance"]},
            evidence_pointers=[str(p1path) + f"#/reviews/{i}", str(p2path) + f"#/reviews/{next(j for j,r in enumerate(p2['reviews']) if r['audit_id']==aid)}",
                str(audit / 'reviewer/formal_pass1_data.json') + f"#/cases/{next(j for j,r in enumerate(bundles[0]['cases']) if r['audit_id']==aid)}"],
            reviewer={"pass1_id": p1.get("reviewer_id"), "pass2_id": p2.get("reviewer_id"), "status": "IDs blank; user-attributed same reviewer, no independent review"})
        changes.append(row)
    revised = apply_layer(originals, changes)
    out.mkdir(parents=True)
    write_json(out / "POLICY.json", POLICY)
    write_json(out / "conflict_holds.json", holds)
    write_json(out / "source_validation.json", validation)
    write_rows(out / "reviewed_case_register_private.jsonl", changes)
    issues = [r for r in changes if r["change_type"] != "no_reference_issue" or r["pass2_exact_matching_flag"] == "yes" or r["pass2_system_information_flag"]]
    write_rows(out / "reference_changes_private.jsonl", issues)
    write_csv(out / "reference_changes_private.csv", issues)
    reviewed_keys = {key(r) for r in changes}
    issue_by_key = {key(r): r for r in changes}
    view = [{"model_id": k[0], "reaction_id": k[1], "sample_id": population[k]["sample_id"], "cluster_id": population[k]["cluster_id"],
        "original_reference_ids": originals[k], "reference_ids": revised[k], "reviewed": k in reviewed_keys,
        "unverified_original_retained": k not in reviewed_keys, "audit_id": issue_by_key.get(k, {}).get("audit_id"),
        "actionable_revision_applied": revised[k] != originals[k]} for k in sorted(population)]
    write_rows(out / "reference_view_163_private.jsonl", view)
    freeze = {"version": VERSION, "description": "single-reviewer, review-informed reference version; not independently adjudicated gold",
        "policy_sha256": sha256(out / "POLICY.json"), "correction_file_sha256": sha256(out / "reference_changes_private.jsonl"),
        "view_sha256": sha256(out / "reference_view_163_private.jsonl"), "holds_sha256": sha256(holds_path),
        "original_benchmark_and_export_sha256": dict(inputs), "reviewed": 60, "unreviewed_unverified": 103,
        "fixed_denominator": 163, "frozen_before_rescoring": True}
    freeze["reference_version_digest"] = digest(freeze)
    write_json(out / "REFERENCE_VERSION.json", freeze)
    # No metric or prediction correctness has been consulted to derive the layer.
    run_metrics(source_root, out, population, originals, revised, issue_by_key, track)
    track(Path(__file__))
    for path in [source_root / "goals.md", audit / "sampling_config.json", audit / "reviewer/review_schema.json",
                 local / "benchmark/scripts/manuscript_metrics.py", local / "benchmark/scripts/phase3_retrieval.py"]:
        track(path)
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("original input changed during analysis")
    from benchmark.scripts import phase3_retrieval as retrieval, phase3b_release as release, phase3c_validation as validation_module, phase3c_grounded as grounded
    verifiers = {"retrieval": retrieval.verify_manifest(), "fusion": release.verify_fusion_manifest(),
        "grounded_validation": validation_module.verify_manifest(), "grounded_smoke": grounded.verify_artifacts()}
    if any(verifiers.values()):
        raise ValueError(verifiers)
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("read-only verification changed an original artifact")
    write_json(out / "INPUT_CHECKSUMS.json", inputs)
    write_json(out / "VERIFICATION.json", {"original_inputs_unchanged": True, "files": len(inputs), "manifests": verifiers,
        "all_scoring_changes_attributable": True, "answer_and_evidence_states_unchanged": True,
        "paid_calls": 0, "heldout_content_read": False, "retrieval_or_inference_rerun": False})
    report(out, changes, freeze)
    write_json(out / "MANIFEST.json", {"version": VERSION, "files": [
        {"path": p.relative_to(out).as_posix(), "sha256": sha256(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    return {"out": str(out), "version": VERSION, "reviewed": 60,
        "actionable_revisions": sum(r["actionable"] for r in changes), "unresolved_reference_issues": sum(r["reference_issue_unresolved"] for r in changes)}


def run_metrics(source_root, out, population, originals, revised, issue_by_key, track):
    from benchmark.scripts import phase3_retrieval as evaluator
    base = source_root / "benchmark/phase3"
    sources = {"bm25": base / "retrieval_baselines/rankings_bm25.jsonl",
        "pretrained_dense": base / "retrieval_baselines/rankings_bge_m3_dense.jsonl",
        "bm25_pretrained_rrf": base / "retrieval_baselines/rankings_bm25_bge_m3_rrf.jsonl",
        "historical_rule_based": base / "retrieval_baselines/rankings_phase2_rule_based.jsonl",
        "trained_epoch1": base / "phase3b_full/rankings_epoch_1.jsonl",
        "selected_rrf": base / "phase3b_fusion/rankings_bm25_trained_epoch1_rrf.jsonl.xz"}
    ranks = {}
    for method, path in sources.items():
        track(path)
        if path.suffix == ".xz":
            with lzma.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = read_jsonl(path)
        index = unique(rows, method)
        if not set(population) <= set(index):
            raise ValueError("ranking lacks paired pilot cases")
        ranks[method] = [{**index[k], "method": method} for k in sorted(population)]
        if any(len(r["ranked_ids"]) != len(set(r["ranked_ids"])) for r in ranks[method]):
            raise ValueError("duplicate ranked candidate")
    frozen_predictions = {}
    for method, rows in ranks.items():
        frozen_predictions[method] = [{"model_id": r["model_id"], "reaction_id": r["reaction_id"],
            "cluster_id": population[key(r)]["cluster_id"], "predicted_id": r["ranked_ids"][0] if r["ranked_ids"] else None,
            "answered": bool(r["ranked_ids"]), "status": "selected" if r["ranked_ids"] else "no_candidate",
            "evidence_compliant": None, "kind": "frozen retrieval Top-1 comparator"} for r in rows]
    responses = unique(read_jsonl(track(base / "phase3c_validation/frozen_responses.jsonl")), "grounded response")
    if responses.keys() != population.keys():
        raise ValueError("grounded population differs")
    frozen_predictions["grounded_gpt56"] = []
    for k in sorted(population):
        r, annotation = responses[k], responses[k].get("annotation") or {}
        status = ("incomplete" if r.get("terminal_status") == "incomplete" else
            "evidence_noncompliant" if r.get("compliance_problems") else
            "abstained" if annotation.get("abstain") else "selected" if annotation.get("predicted_kegg_id") else "technical_failure")
        answered = bool(population[k]["grounded_answered"])
        if answered != (status == "selected"):
            raise ValueError("grounded answer-state crosswalk changed")
        frozen_predictions["grounded_gpt56"].append({"model_id": k[0], "reaction_id": k[1], "cluster_id": population[k]["cluster_id"],
            "predicted_id": annotation.get("predicted_kegg_id") if answered else None, "answered": answered,
            "evidence_compliant": population[k]["grounded_evidence_compliant"], "status": status, "kind": "historical actual GPT-5.6 Terra inference"})
    direct = unique([r for r in read_json(track(base / "validation/scored_rows.json")) if r["variant"] == "target_only"], "direct response")
    if direct.keys() != population.keys():
        raise ValueError("direct population differs")
    frozen_predictions["direct_gpt56"] = [{"model_id": k[0], "reaction_id": k[1], "cluster_id": population[k]["cluster_id"],
        "predicted_id": direct[k]["top1_kegg_id"], "answered": bool(direct[k]["answered"]),
        "status": "selected" if direct[k]["answered"] else "abstained" if direct[k]["abstain"] else direct[k]["terminal_status"],
        "evidence_compliant": direct[k]["evidence_backed"], "kind": "historical actual GPT-5.6 Terra direct inference; no retrieved evidence"} for k in sorted(population)]
    write_json(out / "frozen_prediction_identity.json", {"digest": digest(frozen_predictions), "model": "gpt-5.6-terra", "responses_not_regenerated": True})
    groups = {"all_163": set(population), "reviewed_60": set(issue_by_key), "unreviewed_103": set(population) - set(issue_by_key)}
    for dimension in ["primary_category", "corrected_phase2_stratum", "retrieval_state"]:
        for value in sorted({r[dimension] for r in issue_by_key.values()}):
            groups[f"reviewed/{dimension}/{value}"] = {k for k, r in issue_by_key.items() if r[dimension] == value}
    summaries, scored, retrieval_scored = [], {}, {}
    for version, refs in [("original", originals), ("review_informed", revised)]:
        scored[version], retrieval_scored[version] = {}, {}
        truth = pd.DataFrame([{**population[k], "truth": refs[k], "stratum": population[k]["corrected_stratum"],
                               "seen_in_train": population[k]["target_seen_in_train"]} for k in sorted(population)])
        for method, pred in frozen_predictions.items():
            rows = [score_assertion(r, refs[key(r)]) for r in pred]
            scored[version][method] = rows
            if method in ranks:
                frame = evaluator.score_rankings(ranks[method], truth)
                retrieval_scored[version][method] = unique(frame.to_dict("records"), "evaluated ranks")
            for group, keys in groups.items():
                selected = [r for r in rows if key(r) in keys]
                summary = {"reference_version": version, "method": method, "group": group, **final_summary(selected)}
                if method in ranks:
                    rr = [retrieval_scored[version][method][k] for k in keys]
                    summary.update(retrieval_summary(rr))
                summaries.append(summary)
    # Verify exact reproduction of every available historical final comparator.
    overall = read_json(track(base / "phase3c_validation/overall_metrics.json"))
    names = {"grounded_gpt56": ("grounded", "exact_top1_accuracy"), "selected_rrf": ("fusion_top1", "exact"),
        "direct_gpt56": ("phase3a_target_only", "exact"), "trained_epoch1": ("trained_biencoder_top1", "exact"),
        "historical_rule_based": ("phase2_heuristic_top1", "exact")}
    for method, (name, metric) in names.items():
        if sum(r["correct"] for r in scored["original"][method]) != overall[name][metric]["count"]:
            raise ValueError("original historical metric failed to reproduce")
    diffs, changes = [], []
    for method in frozen_predictions:
        a, b = unique(scored["original"][method], "old"), unique(scored["review_informed"][method], "new")
        for k in sorted(a):
            if {f: v for f, v in a[k].items() if f != "correct"} != {f: v for f, v in b[k].items() if f != "correct"}:
                raise ValueError("reference changed immutable output state")
            old_r, new_r = (retrieval_scored[v].get(method, {}).get(k, {}) for v in ["original", "review_informed"])
            fields = ["recall_at_1_exact", "recall_at_3_exact", "recall_at_5_exact", "recall_at_10_exact", "mrr_at_10_exact"]
            rr = {f: [old_r.get(f), new_r.get(f)] for f in fields if f in old_r}
            changed = a[k]["correct"] != b[k]["correct"] or any(x != y for x, y in rr.values())
            if changed and originals[k] == revised[k]:
                raise ValueError("scoring change without listed reference revision")
            diffs.append({"method": method, "model_id": k[0], "reaction_id": k[1], "audit_id": issue_by_key.get(k, {}).get("audit_id"),
                "reviewed": k in issue_by_key, "original_reference_ids": originals[k], "revised_reference_ids": revised[k],
                "old_correct": a[k]["correct"], "new_correct": b[k]["correct"], "outcome_changed": changed,
                "predicted_id": a[k]["predicted_id"], "answered_unchanged": a[k]["answered"], "status_unchanged": a[k]["status"],
                "evidence_compliance_unchanged": a[k]["evidence_compliant"], "retrieval_metric_diff": rr,
                "reason": issue_by_key.get(k, {}).get("decision_reason") if originals[k] != revised[k] else "reference unchanged"})
        changes.append({"method": method, "n": len(a), **transitions(list(a.values()), list(b.values()))})
    write_csv(out / "metric_summary.csv", summaries)
    write_rows(out / "metric_summary.jsonl", summaries)
    write_rows(out / "case_scoring_diff_private.jsonl", diffs)
    write_csv(out / "changed_outcomes_private.csv", [r for r in diffs if r["outcome_changed"]])
    write_json(out / "reference_correctness_transitions.json", changes)
    paired = []
    for version in scored:
        for group in ["all_163", "reviewed_60"]:
            for a, b in itertools.combinations(sorted(frozen_predictions), 2):
                result = paired_comparison([r for r in scored[version][a] if key(r) in groups[group]],
                                          [r for r in scored[version][b] if key(r) in groups[group]])
                paired.append({"reference_version": version, "group": group, "method_A": a, "method_B": b, **result})
    write_rows(out / "paired_method_differences.jsonl", paired)


def report(out, changes, freeze):
    metrics = read_jsonl(out / "metric_summary.jsonl")
    full = {(r["reference_version"], r["method"]): r for r in metrics if r["group"] == "all_163"}
    counts = {"reviewed": len(changes), "explicit_replacement_proposals": sum(r["pass1_reference_judgment"]["verdict"] == "label_incorrect" and bool(r["explicit_pass1_proposed_ids"]) for r in changes),
        "actionable_revisions": sum(r["actionable"] for r in changes),
        "replacements": sum(r["actionable"] and r["operation"] == "replace" for r in changes),
        "additions": sum(r["actionable"] and r["operation"] == "add" for r in changes),
        "conflict_holds": sum(bool(r["hold"]) for r in changes),
        "unresolved_reference_issues": sum(r["reference_issue_unresolved"] for r in changes),
        "pass1_reference_issue_types": dict(Counter(r["change_type"] for r in changes if r["change_type"] != "no_reference_issue")),
        "pass2_exact_matching_flags": sum(r["pass2_exact_matching_flag"] == "yes" for r in changes),
        "pass2_system_information_flags": sum(r["pass2_system_information_flag"] for r in changes),
        "external_supporting_source_fields_populated_pass1": sum(bool(r["pass1_reference_judgment"]["supporting_sources"].strip()) for r in changes)}
    write_json(out / "SUMMARY.json", counts)
    table = "| Method | Original agreement | Review-informed agreement | Answered (unchanged) | Correct/answered original → revised |\n|---|---:|---:|---:|---:|\n"
    for method in sorted({r["method"] for r in metrics}):
        a, b = full['original', method], full['review_informed', method]
        table += f"| {method} | {a['correct']}/163 | {b['correct']}/163 | {a['answered']}/163 | {a['correct']}/{a['answered']} → {b['correct']}/{b['answered']} |\n"
    ga, gb = full['original', 'grounded_gpt56'], full['review_informed', 'grounded_gpt56']
    fa, fb = full['original', 'selected_rrf'], full['review_informed', 'selected_rrf']
    paragraph = (f"With the same frozen historical GPT-5.6 Terra outputs, accepted-reference agreement changed from "
        f"{ga['correct']}/163 to {gb['correct']}/163 for grounded selection and from {fa['correct']}/163 to {fb['correct']}/163 "
        f"for selected fusion Top-1. The paired grounded-minus-fusion difference remained {gb['correct']-fb['correct']}/163. "
        f"Grounded coverage remained {gb['answered']}/163 and correct/answered changed from {ga['correct']}/{ga['answered']} "
        f"to {gb['correct']}/{gb['answered']}. These are reference-sensitivity findings, not new inference or independently verified biological improvements.\n\n")
    (out / "RESULTS.md").write_text(
        f"# Single-reviewer reference sensitivity\n\n{len(changes)}/163 pilot reactions were reviewed. "
        f"{counts['explicit_replacement_proposals']} explicit Pass 1 replacement proposals were recorded; {counts['conflict_holds']} are suspended because later reference statements conflict. "
        f"The primary {VERSION} view applies {counts['replacements']} replacements and {counts['additions']} additions, with {counts['unresolved_reference_issues']} reference issues left unresolved. "
        "All 103 unreviewed reactions retain unverified original labels. No correction rate is extrapolated to 969.\n\n" + paragraph + table +
        "\nThese paired scores use identical frozen outputs and a fixed denominator. They quantify sensitivity to documented review-informed references, "
        "not independently confirmed biological accuracy or a fully corrected gold standard. Original metrics remain primary historical results. "
        "Retrieval R@1/3/5/10 and MRR@10, strata, paired method differences and complete scoring changes are supplied separately. "
        "No evaluable-only exclusion analysis is used: ambiguity/non-mappability remain flagged without silently dropping cases.\n", encoding="utf-8")
    (out / "METHODS_AND_LIMITATIONS.md").write_text(
        "# Methods and limitations\n\nWe verified the latest complete, matching JSON pair using the existing reviewer schema, canonical digests, dataset digests, "
        "60 unique frozen IDs, embedded Pass 1 equality and sample/model/reaction crosswalk. The immutable layer was frozen before rescoring. "
        "Only explicit Pass 1 proposed IDs accompanying a reference verdict and rationale authorize changes. Incorrect references are replaced; incomplete/ambiguous "
        "references accept an explicit additional list by union, preserving originals. Prediction-correctness, taxonomy and inappropriate abstention never generate corrections. "
        "Pass 2 free-text reference observations are recorded separately; conflicting statements suspend the proposal, without resolving biology. "
        "All source/support pointers and verbatim judgments remain private. Supporting-source fields are blank in these exports; recorded rationale and local model/catalog "
        "snapshots are available, but no external biological verification or independent adjudication was performed.\n\n"
        "For unchanged frozen rankings we call the established phase3_retrieval.score_rankings evaluator. R@K is any accepted reference in Top-K, MRR@10 "
        "uses the first accepted reference within ten. Top-K lists are not asserted annotations. Final Top-1 agreement tests membership in the accepted set, "
        "not equality to the whole multi-reference set. Deterministic retrieval Top-1 comparators are labeled separately from actual historical GPT-5.6 inference. "
        "Coverage, biological abstention, empty deterministic candidate sets, technical failures and incomplete outputs retain their original states. "
        "All 163 cases remain in accuracy/coverage; correct/answered has an explicit answer denominator. Grounded evidence compliance is reference-independent, "
        "and direct-mode evidence support has a different opportunity, so neither a changed reference nor a support-rate difference establishes reduced hallucination.\n\n"
        "The 60-case audit is stratified/error-enriched and contains one reviewer in two stages. Stratum denominators are in metric_summary.csv; no unweighted "
        "subset rate is a population error rate. Sampling strata, including retrieval-state strata, are fixed at their original definitions rather than reassigned after reference changes. "
        "Pass 1 hid system alternatives but exposed source labels/context; the developer/reviewer could have prior familiarity. "
        "Pass 2 exposed system outputs, creating anchoring risk; it is not an independent review. Paired differences bootstrap whole frozen clusters (2,000 draws, "
        "seed 20261109); only 12 pilot clusters and fewer reviewed clusters limit interval stability. Intervals describe this development sample, not uncertainty "
        "about which references are biologically true. The optional question packet is not a completion gate or reopened mandatory review. "
        "Frozen labels, rankings, metrics, original exports and earlier manuscript packages remain unchanged. Phase 5 stays paused; held-out data is sealed.\n", encoding="utf-8")
    questions = [{"audit_id": r["audit_id"], "model_id": r["model_id"], "reaction_id": r["reaction_id"],
        "change_type": r["change_type"], "original_reference_ids": r["original_reference_ids"],
        "proposed_ids_not_automatically_accepted": r["explicit_pass1_proposed_ids"], "recorded_rationale": r["pass1_reference_judgment"]["biological_rationale"],
        "pass2_reference_context": r["pass2_reference_context"], "hold": r["hold"],
        "question": "Optionally specify a consistent explicit accepted reference set, or retain uncertainty; no answer is supplied by the agent.",
        "required_for_this_report": False} for r in changes if r["reference_issue_unresolved"] or r["pass2_exact_matching_flag"] == "yes"]
    write_csv(out / "optional_human_questions_private.csv", questions)
    retrieval_text = "# Original versus review-informed retrieval\n\nSame frozen rankings; established any-accepted-reference evaluator.\n\n"
    reviewed_text = "# Reviewed subset: descriptive, stratified sample of 60\n\n| Method | Original correct | Review-informed correct | Answered (unchanged) |\n|---|---:|---:|---:|\n"
    for group in ["all_163", "reviewed_60"]:
        rows = {(r['reference_version'], r['method']): r for r in metrics if r['group'] == group}
        retrieval_text += f"## {group}\n\n| Method | R@1 original → revised | R@3 | R@5 | R@10 | MRR@10 |\n|---|---:|---:|---:|---:|---:|\n"
        for method in sorted({r['method'] for r in metrics}):
            a, b = rows['original', method], rows['review_informed', method]
            if group == "reviewed_60":
                reviewed_text += f"| {method} | {a['correct']}/60 | {b['correct']}/60 | {a['answered']}/60 |\n"
            if 'recall_at_1_exact' not in a:
                continue
            cells = [f"{a[f'recall_at_{k}_exact_count']}/{a['n']} → {b[f'recall_at_{k}_exact_count']}/{b['n']}" for k in [1, 3, 5, 10]]
            retrieval_text += f"| {method} | " + " | ".join(cells) + f" | {a['mrr_at_10_exact']:.6f} → {b['mrr_at_10_exact']:.6f} |\n"
        retrieval_text += "\n"
    reviewed_text += "\nPrimary sampling strata (not redefined after rescoring):\n\n| Stratum | n | Grounded original → revised | Fusion original → revised |\n|---|---:|---:|---:|\n"
    for group in sorted({r['group'] for r in metrics if r['group'].startswith('reviewed/primary_category/')}):
        rows = {(r['reference_version'], r['method']): r for r in metrics if r['group'] == group}
        a, b = rows['original', 'grounded_gpt56'], rows['review_informed', 'grounded_gpt56']
        c, d = rows['original', 'selected_rrf'], rows['review_informed', 'selected_rrf']
        reviewed_text += f"| {group.rsplit('/', 1)[-1]} | {a['n']} | {a['correct']} → {b['correct']} | {c['correct']} → {d['correct']} |\n"
    (out / "RETRIEVAL_TABLES.md").write_text(retrieval_text, encoding="utf-8")
    (out / "REVIEWED_SUBSET.md").write_text(reviewed_text, encoding="utf-8")
    (out / "CHANGELOG_PRIVATE.md").write_text("# Traceable reference change log\n\n" + "\n".join(
        f"- {r['audit_id']} ({r['model_id']}/{r['reaction_id']}): {r['operation']}; {r['original_reference_ids']} → {r['revised_reference_ids']}. "
        f"{r['decision_reason']} Recorded Pass 1 rationale: {r['pass1_reference_judgment']['biological_rationale']}"
        for r in changes if r["change_type"] != "no_reference_issue"), encoding="utf-8")
    (out / "REPRODUCE.md").write_text(
        "# Reproduction\n\nFrom the isolated reference_sensitivity worktree, into a NEW ignored directory:\n\n```powershell\n"
        "python -m benchmark.scripts.reference_sensitivity --source-root ../../.. --holds _manuscript_results_work/reference_conflict_holds_v1.json "
        "--out _manuscript_results_work/reference_review_v1_NEW\n```\n\n"
        "The holds file is checksum-bound to the exports and each quotation is verified against the recorded field. No biological decision is encoded in reusable code. "
        "Existing destinations are refused. Compare all MANIFEST.json entries across two fresh builds for deterministic reproduction. "
        "REFERENCE_VERSION.json binds the original references/exports, corrections, policy and derived view; INPUT_CHECKSUMS.json binds predictions, rankings and code. "
        "Do not hand-edit a frozen view or manifest; a change requires a fresh version/directory.\n", encoding="utf-8")


def verify_output(out):
    """Read-only verification of immutable files, inputs and applied set changes."""
    out = Path(out)
    manifest = read_json(out / "MANIFEST.json")
    listed = [r["path"] for r in manifest["files"]]
    if len(listed) != len(set(listed)) or any(sha256(out / r["path"]) != r["sha256"] for r in manifest["files"]):
        raise ValueError("output manifest mismatch")
    inputs = read_json(out / "INPUT_CHECKSUMS.json")
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise ValueError("original input or analysis code checksum changed")
    freeze = read_json(out / "REFERENCE_VERSION.json")
    if digest({k: v for k, v in freeze.items() if k != "reference_version_digest"}) != freeze["reference_version_digest"]:
        raise ValueError("reference version digest mismatch")
    for name, field in [("POLICY.json", "policy_sha256"), ("reference_changes_private.jsonl", "correction_file_sha256"),
                        ("reference_view_163_private.jsonl", "view_sha256")]:
        if sha256(out / name) != freeze[field]:
            raise ValueError("reference layer checksum mismatch")
    view = read_jsonl(out / "reference_view_163_private.jsonl")
    changes = read_jsonl(out / "reviewed_case_register_private.jsonl")
    originals = {key(r): r["original_reference_ids"] for r in view}
    revised = apply_layer(originals, changes)
    if len(originals) != 163 or len(changes) != 60 or sum(r["reviewed"] for r in view) != 60:
        raise ValueError("reference coverage mismatch")
    if any(revised[key(r)] != r["reference_ids"] for r in view):
        raise ValueError("derived reference view differs from correction application")
    for row in read_jsonl(out / "case_scoring_diff_private.jsonl"):
        if row["outcome_changed"] and originals[key(row)] == revised[key(row)]:
            raise ValueError("outcome change without a reference revision")
    return {"status": "passed", "files": len(listed), "input_files": len(inputs), "n": 163, "reviewed": 60,
            "actionable_revisions": sum(r["actionable"] for r in changes), "reference_version_digest": freeze["reference_version_digest"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "holds", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_output(args.out) if args.verify_only else build(args.source_root, args.holds, args.out), indent=2))
