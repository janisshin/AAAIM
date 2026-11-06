"""Render Phase 5 retrieval reports/plots and a new explicitly draft freeze."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from benchmark.scripts.phase5_robustness import ROOT, load, write


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot(out, plot_deps):
    if plot_deps:
        sys.path.append(str(plot_deps))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "benchmark/phase5/_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = csv_rows(out / "paired_family_summary.csv")
    clean = {r["method"]: r for r in csv_rows(out / "comparison_by_condition.csv") if r["condition"] == "clean"}
    methods = ["rule_strict", "bm25", "trained", "fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for method in methods:
        for ax, metric in zip(axes, ["recall_at_1", "recall_at_10"]):
            values = [next(r for r in table if r["method"] == method and r["family"] == f"remove_{x:g}") for x in [.1, .25, .5, .75]]
            ax.plot([0, .1, .25, .5, .75], [float(clean[method][metric])] + [float(r[metric]) for r in values], marker="o", label=method)
            ax.set(xlabel="Nominal removed identifier-bundle fraction", ylabel=metric.replace("recall_at_", "Recall@"), ylim=(0, 1), title="163 validation reactions; names/equations retained")
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(out / "identifier_removal.svg")
    fig.savefig(out / "identifier_removal.png", dpi=160)
    plt.close(fig)
    families = ["remove_0.1", "remove_0.25", "remove_0.5", "remove_0.75", "reactants_1", "products_1", "ancestor_1", "wrong_0.25"]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for offset, method in enumerate(methods):
        values = [next(r for r in table if r["method"] == method and r["family"] == f) for f in families]
        y = [float(v["delta_recall_at_10"]) for v in values]
        low, high = [[float(v[f"recall_at_10_delta_ci_{bound}"]) for v in values] for bound in ("low", "high")]
        # Some discrete bootstrap intervals can lie just outside a Monte Carlo
        # point estimate: plot interval endpoints directly, never invent errors.
        x = [i+(offset-1.5)*.15 for i in range(len(families))]
        ax.vlines(x, low, high, alpha=.65, colors=f"C{offset}")
        ax.scatter(x, y, color=f"C{offset}", label=method, s=25)
    ax.set_xticks(range(len(families)), families, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(ylabel="Paired change in Recall@10", title="95% paired cluster + perturbation-replicate bootstrap intervals")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.savefig(out / "robustness_deltas.svg")
    fig.savefig(out / "robustness_deltas.png", dpi=160)
    plt.close(fig)


def report(out, source):
    family = csv_rows(out / "paired_family_summary.csv")
    clean = {r["method"]: r for r in csv_rows(out / "comparison_by_condition.csv") if r["condition"] == "clean"}
    realized = csv_rows(out / "realized_corruptions.csv")
    llm = load(out / "llm_dry_run.json")
    lines = ["# Phase 5 robustness results and experiment coverage", "",
             "Status: local retrieval study complete; Phase 5 overall remains blocked on the prespecified degraded grounded-LLM validation comparison. Selected method is unchanged: BM25 + epoch-1 trained bi-encoder RRF Top-10 followed by grounded selection/abstention for every reaction. No retraining, adaptation or policy retuning occurred.", "",
             "## Population and executed work", "",
             "Exactly 163 frozen validation reactions, 19 aligned query conditions (3,097 total): clean, four random removal fractions with three seeds each, reactant-side removal, product-side removal, one-hop ancestor replacement, and three sibling-corruption seeds. This is not all 969 validation reactions or a held-out evaluation. The 609 annotated participant bundles are query-level occurrences, not 609 unique model species; there are 1,306 distinct identifier links across these occurrences.",
             "All variants were actually rebuilt and reranked for strict rules, BM25, the trained encoder and the selected fusion. Only byte-identical query/mapping inputs were memoized; 1,573 unique query texts were encoded. One fixed catalog encoding was shared because the catalog did not change. No clean rankings were assigned to changed inputs.", "",
             "| Method | Clean R@1 | Clean R@10 | Clean empty rate |",
             "| --- | ---: | ---: | ---: |"]
    for method, row in clean.items():
        lines.append(f"| {method} | {float(row['recall_at_1']):.4f} | {float(row['recall_at_10']):.4f} | {float(row['empty']):.4f} |")
    lines += ["", "The strict-rule control uses existing core matching/scoring with direct mappings and no iterative ontology relaxation. It is not the historical ontology-relaxed Phase 2 configuration. Its clean baseline is explicitly recomputed here and all its deltas are paired against that baseline.",
              "BM25 reproduces all 163 historical Top-100 lists. The encoder reproduces all historical Top-1 and Top-10 lists; one Top-100 tail differs due to the disclosed numerical inference path. Fusion Top-10 historical agreement is recorded in historical_clean_comparison.json. Comparisons below use recomputed clean results throughout, so no historical population or precision mix is hidden.", "",
              "## Paired robustness findings", "",
              "| Method | Perturbation family | Replicates | R@10 | Change vs clean | 95% interval for change |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in family:
        lines.append(f"| {row['method']} | {row['family']} | {row['perturbation_replicates']} | {float(row['recall_at_10']):.4f} | {float(row['delta_recall_at_10']):+.4f} | [{float(row['recall_at_10_delta_ci_low']):+.4f}, {float(row['recall_at_10_delta_ci_high']):+.4f}] |")
    lines += ["", "The trained encoder is descriptively most stable under identifier removal (R@10 near its clean 0.7423 across all four levels). BM25 loses substantial R@10 at high removal. Fusion starts from higher clean R@10 and degrades less than BM25 in point estimates, but its intervals are wide and often include zero. paired_method_robustness.csv provides paired differences in changes between methods; these distinguish stability from absolute retrieval performance. Do not infer universal superiority from one curve or overlapping intervals. These are reused-development-pilot results, not an independent final test.",
              "Removing constraints can improve strict-rule candidate recall by admitting more reactions; this is not proof that less biological evidence improves annotation. Candidate counts, emptiness, R@1/3/5/10/100 and MRR@10/100 must be read together. Full-catalog candidate recall is query-independent catalog coverage, not success at a useful rank.",
              "Names, equation text, stoichiometry and other participant evidence remain present. Neural stability here does not establish robustness to loss of all biological information. Ancestor replacement retains direct KEGG IDs and many changes are consequently masked. Sibling corruptions are synthetic graph-constrained perturbations, not independently verified biological misannotations.", "",
              "## Realized doses and unchanged cases", "",
              "| Condition | Changed bundles / eligible | Removed identifier links / original | Unchanged queries / 163 | Selected but unchanged bundles |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in realized:
        lines.append(f"| {row['condition']} | {row['changed_species']}/{row['eligible_species']} | {row['links_removed']}/{row['links_before']} | {row['unchanged_query']}/163 | {row['unchanged_selected_species']} |")
    lines += ["", "Random removal uses per-query half-up rounding, so nominal 10/25/50/75% corresponds to 8.05/27.59/56.65/78.33% of annotated bundles in this pilot. Each seed selects the same count but can remove different numbers of identifier links. One-sided removal has 8 reactant-side and 2 product-side unchanged/ineligible queries, retained in denominators. No shared-both-sides participant was selected in this pilot; the implementation/test handles that case. Ancestor replacement changes 324/609 bundles with 68 unchanged queries; 285 changed bundles retain direct KEGG evidence. Sibling replacement changes 78/84/82 bundles of 609 (12.81/13.79/13.46%), with 90/84/86 selected attempts unable to change; no outcome-driven refill is performed.",
              "The full event-level perturbation_manifest.jsonl records old/new IDs and one-hop distances or no-replacement outcomes. No species names, stoichiometry, sides or equations were altered. SBML and ontology/mapping source bytes remain unchanged.", "",
              "## Uncertainty and limitations", "",
              "Two thousand bootstrap draws, seed 20261106, pair clean and degraded query outcomes while resampling model clusters and stochastic perturbation replicate indices. The same 163 query denominator is used across conditions; three seeds are not 489 independent reactions. Deterministic side/ancestor conditions have only cluster variation. There are 12 clusters, three stochastic seeds and a development-stratified pilot; intervals are descriptive, not population-wide error bars or multiplicity-corrected confirmatory evidence. Individual-seed minima/maxima and stratum/cluster denominators are exported. Corruption generation never reads reference membership, audit flags or known correctness.", "",
              "## Missing end-to-end comparison and concrete resource decision", "",
              f"The written primary end-to-end condition is {llm['condition']} on the same 163 cases. The dry plan contains {llm['new_calls_needed']} new requests and {llm['compatible_clean_cache_hits']} canonically identical clean-response reuses. No paid calls were made and no LLM predictions were fabricated for changed requests. Retrieval recall is not grounded accuracy or abstention robustness.",
              f"At current public Standard short-context prices for the unchanged model, the input-estimate/historical-mean-output scenario is ${llm['estimated_cost_usd_at_historical_mean_output']:.2f}; the scenario allowing all 2,048 output tokens is ${llm['estimated_cost_usd_at_max_output_tokens']:.2f}. Those estimates are not guarantees. A 32,768-input-unit pessimistic reservation is ${llm['conservative_32768_input_reservation_total_usd']:.2f} for this plan; proposed hard cap $15, one attempt per missing request, zero currently authorized. No cache discount or regional modifier is assumed. [Official OpenAI API pricing](https://developers.openai.com/api/docs/pricing) was fetched on 2026-11-06; rates are recorded separately without changing old snapshots.",
              "Expected value of the proposed run: measure paired grounded correctness, coverage, selective accuracy and reference-based abstention under 50% identifier removal. The decision required is authorization of this exact validation-only request plan and $15 maximum, with the selected model/settings unchanged. The CLI is dry-run-only; execute remains disabled. A persistent approved provider/cache/budget adapter must be integrated before live work; no new held-out executor is enabled.",
              "Iterative ontology relaxation, direct LLM perturbation runs, other grounded perturbation cells, all-969 expansion and retraining are deferred optional extensions. Required comparisons are not silently replaced by clean cached outputs.", "",
              "## Completion and next step", "",
              "See COMPLETION_CHECKLIST.md and freeze/evaluation_freeze.json. Phase 5 retrieval work is complete, overall Phase 5 is not complete while the required degraded grounded comparison is blocked. Final method selection remains the Phase 4 grounded-fusion method; no adaptation is justified by this retrieval-only study. Resolve the bounded validation run before finalizing the method/evaluation freeze, then integrate/authorize a separate later one-time held-out evaluation. The held-out test stays sealed.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "COMPLETION_CHECKLIST.md").write_text(CHECKLIST, encoding="utf-8")
    freeze_dir = ROOT / "benchmark/phase5/freeze"
    freeze_dir.mkdir(exist_ok=True)
    prior = load(ROOT / "benchmark/phase4/freeze/evaluation_freeze.json")
    updated = {**prior, "schema": "phase5-evaluation-freeze-draft-v1", "status": "draft",
               "phase5_status": "retrieval study complete; primary grounded robustness comparison blocked",
               "method_selection": "unchanged grounded_llm_on_every_reaction over BM25 + trained epoch-1 RRF Top-10",
               "sequence": "Phase 5 development -> final method/evaluation freeze -> later separately authorized held-out evaluation",
               "phase5_protocol": "benchmark/PHASE5.md", "phase5_prediction_freeze": load(out / "prediction_freeze.json"),
               "phase5_grounded_run_plan": llm,
               "blockers": ["Required 50%-identifier-removal grounded validation outputs missing; budget and live-run authorization not supplied.",
                            "Finish persistent validation runner integration using the frozen degraded requests; dry-run provider/schema/budget/cache behavior is testable without paid calls.",
                            "Later held-out executor/label-sealing integration and held-out budget remain unfinished; held-out execution disabled.",
                            "Lock hosted model identity/availability and drift policy before any final test freeze."],
               "preserved_history": "Phase 4 draft retained unchanged under benchmark/phase4/freeze; new planning package only",
               "phase5_reference_policy": "unchanged benchmark labels, no human-audit corrected labels",
               "adaptation_or_retraining": False}
    write(freeze_dir / "evaluation_freeze.json", updated)
    write(freeze_dir / "selected_method.json", load(ROOT / "benchmark/phase4/freeze/selected_method.json"))
    (out / "REPRODUCE.md").write_text(
        "# Local reproduction\n\nRun from the isolated Phase 5 worktree, using a NEW output directory for preparation. SOURCE_ROOT is the preserved original workspace (no mixed-split loaders are called).\n\n```powershell\n"
        "python -m benchmark.scripts.phase5_robustness prepare --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced\n"
        "SOURCE_ROOT/benchmark/phase3/_phase3b_env/Scripts/python.exe -m benchmark.scripts.phase5_robustness dense --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced\n"
        "python -m benchmark.scripts.phase5_robustness analyze --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced\n"
        "python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced\n"
        "python -m benchmark.scripts.phase5_reporting --source-root SOURCE_ROOT --out benchmark/phase5/_reproduced --plot-deps PHASE4_WORKTREE/benchmark/phase4/_deps\n"
        "```\n\nSOURCE_ROOT and PHASE4_WORKTREE are recorded in the local handoff provenance. Plotting uses existing local Matplotlib dependencies; no downloads are needed. The existing GPU environment supplies pinned Torch/Transformers, while the normal benchmark environment supplies SBML/core matching dependencies. All trained weights/catalog/settings stay fixed. The reporting stage writes a new Phase 5 draft freeze and does not edit historical Phase 4 files. Paid execution is disabled.\n", encoding="utf-8")


CHECKLIST = """# Phase 5 completion checklist

| Item | Status | Evidence / boundary |
| --- | --- | --- |
| Separate Phase 5 branch/worktree and verified Phase 4 handoff | Complete | original worktrees preserved; original goals snapshot retained in ignored _provenance; Phase 4 uncommitted implementation copied by checksum |
| Pre-experiment protocol and fixed grid | Complete | benchmark/PHASE5.md; study_config.json; 19 conditions, three fixed stochastic seeds, 163 validation reactions |
| Random 10/25/50/75% bundle removal | Complete locally | derived_inputs.jsonl; perturbation_manifest.jsonl; realized_corruptions.csv; names/stoichiometry retained |
| Reactant/product removal, reversibility and eligibility | Complete locally | same manifests; record SBML sides; retain unchanged cases in all denominators |
| ChEBI ancestor replacement | Complete locally | frozen local is_a graph, deterministic immediate parent, distance=1; direct KEGG retained and masking disclosed |
| Synthetic incorrect mappings | Complete locally | shared-parent sibling with disjoint original compound mapping; inability to replace counted; no biological plausibility claim beyond constraints |
| Rule-based retrieval comparison | Complete for explicit strict control | existing strict filter/scorer recomputed on direct mappings; does not certify historical iterative Phase 2 workflow |
| BM25, trained encoder and selected fusion | Complete locally | frozen_rankings.jsonl; fixed catalog/checkpoint, clean checks, no clean-ranking substitution |
| Paired metrics, subgroups and cluster/replicate uncertainty | Complete | comparison_by_condition.csv, paired_family_summary.csv, paired_method_robustness.csv, subgroups.csv, plots |
| Grounded selected-method robustness at 50% removal | Blocked, required | llm_dry_run.json and ignored _requests/primary_grounded_plan.jsonl: new responses require separate authorization/budget; retrieval-only is not full Phase 5 |
| Other grounded perturbation cells and direct LLM recovery | Deferred optional | no compatible changed-input outputs; not every method-condition combination is required |
| Iterative ontology relaxation / EM comparison | Deferred optional | existing expensive configuration not part of bounded strict-control study; no claim of execution |
| All-969 validation, adaptation or retraining | Deferred optional | not part of current fixed 163-query protocol |
| Human labels / audit reopening | Not required, excluded | completed scoped audit preserved; no private case judgments in Phase 5 artifacts |
| Selected method decision | Complete, unchanged | keep Phase 4 grounded-fusion selection; retrieval-only results do not select a new end-to-end method |
| Final method/evaluation freeze | Draft, pending | ../freeze/evaluation_freeze.json; required Phase 5 outputs and later operational test gates remain missing |
| Held-out evaluation | Sealed, not executed | no test reaction/label/prediction/diagnostic access; later separate authorization only |

Overall: retrieval robustness study complete; Phase 5 not fully complete. Do not close the final freeze until the required bounded grounded comparison and operational gates are resolved or an explicit later scope decision changes those requirements.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-deps", type=Path)
    args = parser.parse_args(argv)
    plot(args.out, args.plot_deps)
    report(args.out, args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
