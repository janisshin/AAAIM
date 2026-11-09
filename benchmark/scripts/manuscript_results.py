"""Assemble verified development results; never run provider or held-out evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

import pandas as pd

from benchmark.scripts.manuscript_propagation import read_json, read_jsonl, write_json, write_csv, write_rows, sha256, digest
from benchmark.scripts.manuscript_metrics import final_annotation_metrics


def portable_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_checksum_map(root, manifest):
    payload = read_json(manifest)
    files = payload.get("files", payload)
    if not isinstance(files, dict):
        raise ValueError("expected checksum map")
    problems, checked = [], []
    for relative, expected in files.items():
        path = Path(root) / relative
        actual = sha256(path) if path.is_file() else None
        portable = portable_sha(path) if path.is_file() else None
        if expected not in {actual, portable}:
            problems.append(relative)
        checked.append({"path": str(path), "expected": expected, "actual_bytes": actual,
                        "portable_lf": portable, "matches": expected in {actual, portable}})
    return {"manifest": str(manifest), "checked": len(checked), "problems": problems, "records": checked}


AUDIT_VALIDATOR = '''import json,sys
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root))
from benchmark.scripts import phase3_error_analysis as a,phase3_error_reviewer as r
p=root/'benchmark/phase3/error_audit/review_work/inputs_revision_20261105'
exports=[a.load(p/f'pass{i}_review_reviewer.json') for i in (1,2)]
bundles=[a.load(r.OUT/f'formal_pass{i}_data.json') for i in (1,2)]
errors={f'pass{i}':a.validate(x,b,i) for i,x,b in zip((1,2),exports,bundles)}
errors['bundle']=['dataset content digest mismatch' for b in bundles if r._digest({k:v for k,v in b.items() if k!='dataset_digest'})!=b['dataset_digest']]
embedded=exports[1].get('locked_pass1_export')==exports[0]
cross=a.csv_rows(a.AUDIT/'formal_audit_crosswalk.csv')
ids={row['audit_id'] for row in cross}
assert len(cross)==len(ids)==60 and ids=={row['audit_id'] for row in exports[1]['reviews']}
assert embedded and not any(errors.values()), errors
from benchmark.scripts import phase3_error_inventory as inv,phase3_error_sampling as sam
verifiers={'inventory':inv.verify_manifest(),'sampling':sam.verify_manifest(),'reviewer':r.verify_manifest()}
assert not any(verifiers.values()),verifiers
print(json.dumps({'errors':errors,'embedded_pass1_identical':embedded,'frozen_case_ids':sorted(ids),'manifest_verifiers':verifiers,
 'dataset_digests':[b['dataset_digest'] for b in bundles],'bundle_fields':[sorted(bundles[0]['cases'][0]),sorted(bundles[1]['cases'][0])]}))
'''


def build(source_root, phase4_root, phase5_root, prediction_root, out):
    source_root, phase4_root, phase5_root, prediction_root, out = map(lambda x: Path(x).resolve(),
        [source_root, phase4_root, phase5_root, prediction_root, out])
    out.mkdir(parents=True, exist_ok=False)
    for name in ["tables", "figures", "audit", "protocols", "case_study", "sources"]:
        (out / name).mkdir()
    sources = {}
    def track(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return path
    # Verify historical checksums in place. No frozen report is rewritten.
    verifiers = {}
    for name, root, relative in [("phase4", phase4_root, "benchmark/phase4/artifact_checksums.json"),
                                 ("phase5", phase5_root, "benchmark/phase5/artifact_checksums.json")]:
        manifest = track(root / relative)
        result = verify_checksum_map(root, manifest)
        verifiers[name] = result
        if result["problems"]:
            planning_only = {"benchmark/PHASE5.md", "benchmark/phase5/README.md", "benchmark/phase5/freeze/evaluation_freeze.json",
                "benchmark/phase5/validation/COMPLETION_CHECKLIST.md", "benchmark/phase5/validation/REPORT.md",
                "benchmark/phase5/validation/REPRODUCE.md", "benchmark/phase5/validation/llm_dry_run.json",
                "benchmark/phase5/validation/llm_request_manifest.jsonl", "benchmark/scripts/phase5_llm_plan.py",
                "benchmark/scripts/phase5_reporting.py", "goals.md", "tests/test_phase5_llm_plan.py"}
            if name != "phase5" or set(result["problems"]) - planning_only:
                raise ValueError(f"{name} numerical artifact checksum failures: {result['problems']}")
            result["interpretation"] = "Historical global ledger is stale for later planning/docs changes; all numerical retrieval entries match. No ledger repaired."
    audit_verification = subprocess.run([sys.executable, "-c", AUDIT_VALIDATOR, str(source_root)],
        cwd=source_root, check=True, capture_output=True, text=True)
    verifiers["audit"] = json.loads(audit_verification.stdout)
    # All imported modules below read manifests only, not held-out queries/labels.
    from benchmark.scripts import phase3_retrieval as r, phase3b_release as b, phase3c_grounded as g, phase3c_validation as v
    verifiers["phase3"] = {"retrieval": r.verify_manifest(), "fusion": b.verify_fusion_manifest(),
        "grounded_smoke": g.verify_artifacts(), "grounded_validation": v.verify_manifest()}
    if any(verifiers["phase3"].values()):
        raise ValueError(verifiers["phase3"])
    write_json(out / "sources/verification.json", verifiers)

    fusion_dir = source_root / "benchmark/phase3/phase3b_fusion"
    methods = read_json(track(fusion_dir / "metrics_by_method.json"))["methods"]
    retrieval = []
    identities = {"bm25": "BM25 lexical", "bge_m3_dense": "BAAI/bge-m3 pretrained dense",
        "bm25_bge_m3_rrf": "historical BM25 + pretrained BGE-M3 RRF",
        "trained_biencoder_epoch1": "selected trained BAAI/bge-small-en-v1.5 epoch 1",
        "bm25_trained_epoch1_rrf": "selected BM25 + trained epoch-1 RRF",
        "phase2_rule_based": "historical ontology-relaxed Phase 2 rule-based"}
    for name, values in methods.items():
        for metric in ["recall_at_1_exact", "recall_at_3_exact", "recall_at_5_exact", "recall_at_10_exact", "mrr_at_10_exact"]:
            value = values[metric]
            if value["n_reactions"] != 969:
                raise ValueError("population mismatch in full validation metric")
            count = round(value["reaction_micro"] * 969) if metric.startswith("recall") else None
            retrieval.append({"method": name, "identity": identities[name], "metric": metric,
                "count": count, "denominator": 969, "reaction_micro": value["reaction_micro"],
                "model_macro": value["model_macro"], "cluster_macro": value["cluster_macro"],
                "n_models": 18, "n_clusters": 12, "experiment_type": "executed full-catalog validation retrieval",
                "production_scope": name in {"bm25", "trained_biencoder_epoch1", "bm25_trained_epoch1_rrf"},
                "source": str(fusion_dir / "metrics_by_method.json") + f"#/methods/{name}/{metric}"})
    write_csv(out / "tables/retrieval_969.csv", retrieval)
    write_json(out / "tables/retrieval_969_uncertainty.json", read_json(track(fusion_dir / "bootstrap_comparisons.json")))

    pilot_dir = source_root / "benchmark/phase3/phase3c_validation"
    pilot_config = read_json(track(pilot_dir / "config.json"))
    overall = read_json(track(pilot_dir / "overall_metrics.json"))
    pilot = []
    for name, metrics in overall.items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if isinstance(value, dict) and {"count", "denominator", "rate"} <= set(value):
                pilot.append({"method": name, "metric": metric, **value, "population": "stratified 163-reaction validation pilot",
                    "model": pilot_config.get("model") if name == "grounded" else "historical comparator; see source",
                    "experiment_type": "frozen historical actual inference" if name == "grounded" else "historical frozen comparator",
                    "source": str(pilot_dir / "overall_metrics.json") + f"#/{name}/{metric}"})
    write_csv(out / "tables/grounded_pilot_163.csv", pilot)
    write_json(out / "tables/grounded_pilot_uncertainty.json", read_json(track(pilot_dir / "bootstrap_comparisons.json")))
    responses = read_jsonl(track(pilot_dir / "frozen_responses.jsonl"))
    truth = read_jsonl(track(pilot_dir / "answer_key_join.jsonl"))
    truth_map = {(row["model_id"], row["reaction_id"]): row for row in truth}
    final_rows = []
    for row in responses:
        ref = truth_map[(row["model_id"], row["reaction_id"])]
        answer = row["annotation"]
        status = ("evidence_noncompliant" if row["compliance_problems"] else
                  "abstained" if answer["abstain"] else "selected")
        final_rows.append({"model_id": row["model_id"], "reaction_id": row["reaction_id"],
            "reference_ids": ref["ground_truth_ids"], "status": status,
            "prediction_ids": [answer["predicted_kegg_id"]] if status == "selected" else []})
    final_metrics = final_annotation_metrics(final_rows)
    if final_metrics["correct"] != 102 or final_metrics["answered_evaluable"] != 124:
        raise ValueError("historical accuracy changed in metric crosswalk")
    write_json(out / "tables/pilot_asserted_precision_recall.json", final_metrics)
    write_rows(out / "tables/pilot_asserted_sets.jsonl", final_rows)

    phase4 = pd.read_csv(track(phase4_root / "benchmark/phase4/validation/comparison.csv")).to_dict("records")
    for row in phase4:
        row.update(population="same stratified 163 validation reactions", experiment_type="cached replay; zero new calls",
                   response_model=pilot_config["model"], actual_new_calls=0,
                   source=str(phase4_root / "benchmark/phase4/validation/comparison.csv"))
    write_csv(out / "tables/phase4_cached_routing.csv", phase4)
    write_json(out / "tables/phase4_uncertainty.json", read_json(track(phase4_root / "benchmark/phase4/validation/metrics.json")))
    phase5 = pd.read_csv(track(phase5_root / "benchmark/phase5/validation/comparison_by_condition.csv")).to_dict("records")
    for row in phase5:
        row.update(population="163 validation reactions x synthetic query conditions", n_reactions=163,
            experiment_type="executed offline synthetic perturbation retrieval", actual_new_calls=0,
            source=str(phase5_root / "benchmark/phase5/validation/comparison_by_condition.csv"))
    write_csv(out / "tables/phase5_retrieval_by_condition.csv", phase5)
    family_source = track(phase5_root / "benchmark/phase5/validation/paired_family_summary.csv")
    family = pd.read_csv(family_source).to_dict("records")
    write_csv(out / "tables/phase5_retrieval_families.csv", [{**row, "n_unique_reactions": 163, "source": str(family_source)} for row in family])

    audit = source_root / "benchmark/phase3/error_audit"
    final_audit = audit / "review_work/analysis_20261105_single_reviewer_final"
    exports = [read_json(track(audit / f"review_work/inputs_revision_20261105/pass{i}_review_reviewer.json")) for i in (1, 2)]
    original_validation = read_json(track(final_audit / "validation_report.json"))
    for i in (1, 2):
        suffix = f"inputs_revision_20261105/pass{i}_review_reviewer.json"
        expected = next(value for key, value in original_validation["input_sha256_bytes"].items() if key.replace("\\", "/").endswith(suffix))
        if sha256(audit / "review_work" / suffix) != expected:
            raise ValueError("final audit export differs from accepted final checksum")
        track(audit / f"reviewer/formal_pass{i}_data.json")
    track(audit / "reviewer/review_schema.json")
    crosswalk = pd.read_csv(track(audit / "formal_audit_crosswalk.csv"), dtype=str).fillna("")
    pass1, pass2 = [{row["audit_id"]: row for row in payload["reviews"]} for payload in exports]
    # Export checksums above preserve every accepted judgment. Keep the actual
    # case-level answers in ignored output, not hard-coded into reusable code.
    write_json(out / "audit/clarified_case_verification_private.json", {
        aid: {"source_export_unchanged": True, "pass1": pass1[aid], "pass2": pass2[aid]}
        for aid in ["P3EA0066", "P3EA0140", "P3EA0090"]})
    joined = pd.read_csv(track(final_audit / "joined_cases.csv"), dtype=str).fillna("")
    for row in joined.to_dict("records"):
        aid = row["audit_id"]
        if json.loads(row["pass1_review"]) != pass1[aid] or json.loads(row["pass2_review"]) != pass2[aid]:
            raise ValueError("case-level joined judgments differ from original export")
        frozen = crosswalk.loc[crosswalk.audit_id.eq(aid)].iloc[0]
        if any(row[key] != frozen[key] for key in ["model_id", "reaction_id", "sample_id", "cluster_id"]):
            raise ValueError("audit case identity changed")
    summary = read_json(track(final_audit / "aggregate_summary.json"))
    for column, expected in [("verdict", summary["all_formal"]["verdict"]),
                             ("prediction_assessment", summary["all_formal"]["prediction_assessment"])]:
        if dict(sorted(Counter(joined[column]).items())) != expected:
            raise ValueError("audit summary does not reproduce")
    taxonomy = Counter(tag for row in pass2.values() for tag in set(row["failure_taxonomy"]))
    if dict(sorted(taxonomy.items())) != summary["all_formal"]["failure_taxonomy_multilabel"]:
        raise ValueError("audit multilabel taxonomy does not reproduce")
    write_csv(out / "audit/case_index_private.csv", joined.to_dict("records"))
    write_json(out / "audit/final_summary.json", summary)
    audit_table = []
    for dimension in ["all_formal", "primary_category", "corrected_phase2_stratum", "retrieval_state"]:
        groups = {"all": summary[dimension]} if dimension == "all_formal" else summary[dimension]
        for group, values in groups.items():
            for outcome in ["verdict", "prediction_assessment", "failure_taxonomy_multilabel"]:
                for label, count in values[outcome].items():
                    audit_table.append({"dimension": dimension, "stratum": group, "outcome": outcome,
                        "label": label, "count": count, "denominator": values["n"],
                        "interpretation": "descriptive single-reviewer stratified sample; taxonomy overlaps",
                        "source": str(final_audit / "aggregate_summary.json") + f"#/{dimension}/{group}/{outcome}/{label}"})
    write_csv(out / "tables/audit_counts_by_stratum.csv", audit_table)
    examples = {"P3EA0066", "P3EA0140", "P3EA0090"}
    for label in sorted(taxonomy):
        examples.add(min(aid for aid, row in pass2.items() if label in row["failure_taxonomy"]))
    for label in sorted(summary["all_formal"]["verdict"]):
        examples.add(min(aid for aid, row in pass1.items() if row["verdict"] == label))
    example_rows = []
    for aid in sorted(examples):
        cross = joined.loc[joined.audit_id.eq(aid)].iloc[0].to_dict()
        example_rows.append({"audit_id": aid, "model_id": cross["model_id"], "reaction_id": cross["reaction_id"],
            "primary_category": cross["primary_category"], "sampling_stratum": cross["corrected_phase2_stratum"],
            "pass1_judgment": pass1[aid], "pass2_judgment": pass2[aid],
            "selection_rule": "mandatory clarified/uncertain IDs plus lexicographic first existing verdict/taxonomy example; no new assessment"})
    write_rows(out / "audit/representative_examples_private.jsonl", example_rows)
    scope = track(audit / "review_work/audit_scope_single_reviewer_20261105.md")
    (out / "audit/AUDIT_SCOPE_AND_LIMITATIONS.md").write_text(scope.read_text(encoding="utf-8") +
        "\nPass 1 hid system alternatives/predictions, not the source label being assessed or necessarily recognizable model/reaction identity. "
        "The same developer/reviewer may have prior familiarity. Pass 2 revealed system evidence; anchoring is possible. "
        "No random-assignment blinding test or independent agreement is available. All reported examples quote only existing judgments. "
        "Information-limited judgments remain unresolved; no new biological decision is supplied.\n", encoding="utf-8")
    plotting_deps = phase4_root / "benchmark/phase4/_deps"
    if plotting_deps.is_dir():
        sys.path.append(str(plotting_deps))
    figures(out, retrieval, phase4, phase5, summary)
    write_json(out / "sources/input_checksums.json", sources)
    write_json(out / "sources/package_inventory.json", {"retrieval_n": 969, "pilot_n": 163, "audit_n": 60,
        "historical_grounded_model": pilot_config["model"], "phase4_new_calls": 0, "phase5_new_calls": 0,
        "human_exports_validate": True, "sources_unchanged": all(sha256(p) == value for p, value in sources.items())})
    return {"package": str(out), "historical_model": pilot_config["model"], "pilot_metrics": {k: v for k, v in final_metrics.items() if k not in {"per_reaction", "model_rows"}},
            "audit": summary["all_formal"], "sources": len(sources)}


def figures(out, retrieval, phase4, phase5, audit):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "aaaim-manuscript-results-v1"})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "svg", "pdf"]:
            metadata = ({"Date": None} if suffix == "svg" else
                        {"CreationDate": None, "ModDate": None} if suffix == "pdf" else
                        {"Software": "AAAIM manuscript results v1"})
            fig.savefig(out / "figures" / (name + "." + suffix), dpi=200, bbox_inches="tight", metadata=metadata)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = list(dict.fromkeys(row["method"] for row in retrieval))
    for pos, metric in enumerate(["recall_at_1_exact", "recall_at_10_exact"]):
        values = [next(row["reaction_micro"] for row in retrieval if row["method"] == method and row["metric"] == metric) for method in methods]
        ax.bar(np.arange(len(methods)) + (pos - .5) * .36, values, width=.36, label=metric.replace("_exact", ""))
    display = {"bge_m3_dense": "Pretrained\nBGE-M3", "bm25": "BM25", "bm25_bge_m3_rrf": "BM25 + BGE-M3\nRRF",
        "bm25_trained_epoch1_rrf": "BM25 + trained\nRRF (selected)", "trained_biencoder_epoch1": "Trained\nepoch 1",
        "phase2_rule_based": "Rule-based\n(historical)"}
    ax.set_xticks(range(len(methods)), [display[m] for m in methods], fontsize=8)
    ax.set(ylabel="Accepted-reference hit rate", ylim=(0, 1), title="Executed retrieval: 969 validation reactions, 18 models / 12 clusters")
    ax.legend()
    save(fig, "retrieval_969")
    fig, ax = plt.subplots(figsize=(8, 4))
    for row in phase4:
        ax.scatter(row["simulated_llm_calls"], row["accuracy"])
        ax.annotate(row["method"], (row["simulated_llm_calls"], row["accuracy"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="Simulated grounded calls / 163 (actual new calls = 0)", ylabel="Exact reference agreement / 163",
           title="Cached routing replay on the stratified 163-case development pilot", ylim=(0, 1))
    save(fig, "cached_routing_163")
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ["bm25", "trained", "fusion"]:
        clean = next(r["recall_at_10"] for r in phase5 if r["method"] == method and r["condition"] == "clean")
        points = [clean]
        for dose in [.1, .25, .5, .75]:
            values = [r["recall_at_10"] for r in phase5 if r["method"] == method and r["condition"].startswith(f"remove_{dose}_")]
            points.append(float(np.mean(values)))
        ax.plot([0, .1, .25, .5, .75], points, marker="o", label=method)
    ax.set(xlabel="Synthetic identifier-removal fraction (3 seeds per nonzero dose)", ylabel="Accepted-reference hit rate @10",
           title="Phase 5 executed retrieval controls: 163 reactions, not actual species predictions", ylim=(0, 1))
    ax.legend()
    save(fig, "synthetic_robustness_163")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    display = {"label_correct": "Correct source label", "label_incorrect": "Incorrect source label",
        "multiple_defensible_annotations": "Multiple defensible annotations", "not_kegg_mappable": "Not KEGG-mappable",
        "prediction_correct": "Correct selection", "prediction_defensible_alternative": "Defensible alternative",
        "prediction_related_but_incorrect": "Related, incorrect selection", "abstention_appropriate": "Appropriate abstention",
        "abstention_inappropriate": "Inappropriate abstention", "insufficient_information": "Insufficient information"}
    for ax, field, title in zip(axes, ["verdict", "prediction_assessment"], ["Pass 1 source-label assessment", "Pass 2 system-informed assessment"]):
        values = audit["all_formal"][field]
        bars = ax.barh([display[k] for k in values], list(values.values()))
        ax.bar_label(bars, padding=3)
        ax.set_xlim(0, max(values.values()) * 1.12)
        ax.set(title=title, xlabel="Cases / stratified sample of 60 (one reviewer)")
    save(fig, "human_audit_counts_60")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "phase4-root", "phase5-root", "prediction-root", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_root, args.phase4_root, args.phase5_root, args.prediction_root, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
