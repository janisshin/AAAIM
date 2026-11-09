"""Finalize private manuscript text, reporting crosswalks and readiness records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

import pandas as pd

from benchmark.scripts.manuscript_propagation import (read_json, read_jsonl, write_json, write_rows, write_csv,
    sha256, digest, load_model, network_identity)


def table_markdown(rows, columns):
    header = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    return header + "".join("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n" for row in rows)


def finalize(package, study, execution, source_root, prediction_root, phase5_root):
    package, study, execution, source_root, prediction_root, phase5_root = map(lambda p: Path(p).resolve(),
        [package, study, execution, source_root, prediction_root, phase5_root])
    if (package / "README.md").exists():
        raise FileExistsError("final package already exists")
    def text(name, value):
        path = package / name
        if path.exists():
            raise FileExistsError(path)
        path.write_text(value.strip() + "\n", encoding="utf-8")
    frozen = read_json(study / "ranking_freeze.json")
    if sha256(study / "rankings.jsonl") != frozen["rankings_sha256"]:
        raise ValueError("ranking freeze changed")
    protocol = read_json(study / "protocol.json")
    alignment = read_json(study / "alignment.json")
    controls = pd.read_csv(study / "summary.csv").set_index("condition")
    paired = read_json(study / "paired_reference_names_only.json")
    proposal = read_json(execution / "execution_proposal.json")
    audit = read_json(package / "audit/final_summary.json")["all_formal"]
    historical = pd.read_csv(package / "tables/retrieval_969.csv")
    precision = read_json(package / "tables/pilot_asserted_precision_recall.json")
    pilot_ci = read_json(package / "tables/grounded_pilot_uncertainty.json")["grounded_minus_fusion_exact_accuracy"]
    # Copy small report tables/protocols, link large ranking/request payload artifacts.
    for name in ["summary.csv", "per_model.csv", "paired_reference_names_only.json", "predefined_missingness_strata.json",
                 "identifier_availability.csv", "analysis_validation.json"]:
        shutil.copyfile(study / name, package / "tables" / ("propagation_control_" + name))
    for name in ["protocol.json", "protocol_freeze.json", "query_freeze.json", "ranking_freeze.json", "alignment.json", "input_checksums.json"]:
        shutil.copyfile(study / name, package / "protocols" / ("propagation_" + name))
    for name in ["evaluation_freeze.json", "selected_method.json"]:
        path = phase5_root / "benchmark/phase5/freeze" / name
        if path.exists():
            shutil.copyfile(path, package / "sources" / ("historical_draft_" + name))
    source_queries = {(r["model_id"], r["reaction_id"]): r["query"] for r in read_jsonl(source_root / "benchmark/phase3/retrieval_baselines/queries.jsonl")}
    reference_queries = [row for row in read_jsonl(study / "queries.jsonl") if row["condition"] == "reference"]
    strip_ids = lambda query: re.sub(r"; (?:ChEBI|KEGG-compound)=[^;\]]*", "", query)
    changes = []
    aliases_by_model = {r["model_id"]: r.get("species_aliases", {}) for r in alignment["validation_models"]}
    for row in reference_queries:
        old = source_queries[(row["model_id"], row["reaction_id"])]
        if old != row["query"]:
            dealiased = row["query"]
            for original_id, alias in aliases_by_model[row["model_id"]].items():
                dealiased = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", lambda m, value=original_id: value, dealiased)
            changes.append({"model_id": row["model_id"], "reaction_id": row["reaction_id"], "old_query": old,
                "current_reference_query": row["query"], "dealiased_query_for_diagnostics_only": dealiased,
                "identifier_fields_changed_after_dealias": old != dealiased,
                "equation_identical": old.split("\nParticipants:")[0] == dealiased.split("\nParticipants:")[0],
                "non_identifier_text_identical": strip_ids(old) == strip_ids(dealiased),
                "exclusion_reasons": sorted({record["exclusion_reason"] for species in row["species_evidence"].values()
                    for record in species["records"] if record["exclusion_reason"]})})
    write_rows(package / "protocols/historical_reference_query_differences_private.jsonl", changes)
    if any(not r["equation_identical"] or not r["non_identifier_text_identical"] for r in changes):
        raise ValueError("historical reference control has unaccounted context drift")
    contract_changes = sum(r["identifier_fields_changed_after_dealias"] for r in changes)
    context_diagnostic(package, study, source_root)
    # Additional legacy case-study naming convention, inventoried without altering frozen cohort.
    additional = []
    for directory in sorted((prediction_root / "results").iterdir()):
        candidate = directory / "sbml_only_source_fidelity_species.csv"
        meta = directory / "sbml_only_source_fidelity_metrics.json"
        model_path = prediction_root / (directory.name + ".xml")
        if not model_path.is_file():
            model_path = directory / (directory.name + ".xml")
        if not candidate.is_file() or not meta.is_file() or not model_path.is_file():
            continue
        model = load_model(model_path).getModel().clone()
        signature = network_identity(model)
        frame = pd.read_csv(candidate, dtype=str).fillna("")
        additional.append({"model_id": directory.name, "path": str(candidate), "prediction_sha256": sha256(candidate),
            "source_sha256": sha256(model_path), "configuration": read_json(meta)["configuration"],
            "rows": len(frame), "source_species": model.getNumSpecies(), "source_reactions": model.getNumReactions(),
            "unknown_species_ids": sorted(set(frame.id) - {s.getId() for s in model.getListOfSpecies()} - {"", "Reason:"}),
            "potential_validation_network_aliases": [r["model_id"] for r in alignment["validation_models"] if r["network_identity"] == signature],
            "status": "additional case-study naming convention; not substituted into frozen propagation cohort"})
    write_json(package / "protocols/additional_prediction_inventory.json", additional)

    wide = []
    for method in historical.method.unique():
        values = historical[historical.method.eq(method)].set_index("metric")
        row = {"Method": method}
        for k in [1, 3, 5, 10]:
            metric = values.loc[f"recall_at_{k}_exact"]
            row[f"Hit R@{k}"] = f"{int(metric['count'])}/969"
        row["MRR@10"] = f"{values.loc['mrr_at_10_exact','reaction_micro']:.4f}"
        row["Model-macro R@1"] = f"{values.loc['recall_at_1_exact','model_macro']:.4f}"
        row["Model-macro R@10"] = f"{values.loc['recall_at_10_exact','model_macro']:.4f}"
        wide.append(row)
    text("tables/TABLE1_RETRIEVAL_969.md", "# Full validation retrieval: 969 reactions, 18 models, 12 clusters\n\n" +
        table_markdown(wide, list(wide[0])) + "\nR@K is recovery of any accepted reference at K, not final annotation precision. "
        "Historical rule-based and pretrained variants are preserved comparisons, not required components of the selected production pipeline. "
        "Paired bootstrap uncertainty: retrieval_969_uncertainty.json. These are validation-selected development findings.\n")
    control_rows = []
    for condition in ["reference", "names_only"]:
        row = controls.loc[condition]
        control_rows.append({"Condition": condition, **{f"Hit R@{k}": f"{int(row[f'hit_at_{k}_count'])}/969" for k in [1, 3, 5, 10]},
                            "MRR@10": f"{row.mrr_at_10:.4f}", "Model-macro R@1": f"{row.hit_at_1_model_macro:.4f}",
                            "Model-macro R@10": f"{row.hit_at_10_model_macro:.4f}"})
    text("tables/TABLE_PROPAGATION_CONTROLS.md", "# Predeclared reference/names-only controls; B actual-prediction arm pending\n\n" +
        table_markdown(control_rows, list(control_rows[0])) + f"\nAll 969 reactions / 18 models / 12 clusters are paired. "
        f"{len(changes)} raw reference queries differ from historical inputs, including uniform structural aliases; "
        f"after reversing aliases only for diagnostics, {contract_changes} differ in identifier evidence under the public qualifier contract. "
        "These new control results do not replace historical rankings or constitute actual-prediction propagation results. "
        "Paired cluster intervals/transitions are in propagation_control_paired_reference_names_only.json.\n")

    a, c = controls.loc["reference"], controls.loc["names_only"]
    delta10 = paired["metrics"]["hit_at_10"]
    text("RESULTS.md", f'''# Reaction annotation: development results for the manuscript

The full-catalog retrieval comparison comprised 969 validation reactions from 18 models in 12 frozen clusters. Selected BM25 + trained epoch-1 RRF recovered an accepted reference at rank 1 for 849/969 reactions (87.6%) and within rank 10 for 923/969 (95.3%). The trained encoder recovered 814/969 and 893/969, and BM25 recovered 756/969 and 891/969, respectively. Model-macro scores and paired cluster-bootstrap intervals are reported separately in Table 1 and its uncertainty file. These are development results from validation-selected methods, not held-out performance. Historical rule-based and pretrained comparisons remain in the table without becoming required production components.

On the separate stratified 163-reaction pilot (17 models, 12 clusters), historical **GPT-5.6 Terra** grounded selection agreed with an accepted reference for 102/163 reactions, compared with 93/163 for fusion Top-1. The paired difference was 9/163 (5.52 percentage points; frozen 95% cluster-bootstrap interval {pilot_ci['ci_95_percentile'][0]:.4f} to {pilot_ci['ci_95_percentile'][1]:.4f}). There were 124/163 answered cases and 102/124 correct answered cases. The historical 39 abstaining outputs comprise 37 valid biological/evidence abstentions and two evidence-noncompliant technical outputs under the strict reporting crosswalk; they remain in the 163-case denominator. No selected grounded identifier was outside its supplied evidence, but that is evidence compliance, not proof of zero biological errors. Direct-mode answers without retrieval and grounded out-of-evidence selections have different support opportunities; their support-rate bootstrap is not a hallucination-reduction estimate. The 163-case pilot is not a representative substitute for the 969-query set.

Phase 4 replayed cached outputs on the same 163 pilot reactions, with zero new calls. The routed 0.05 policy reproduced 102/163 agreement and 124/163 coverage while selecting 142/163 historical grounded responses rather than 163/163. The prespecified selection nevertheless retained full grounded annotation. Replayed cost is historical accounting, not new spending, refunds or an established end-to-end speedup. Validation threshold selection and only 12 clusters limit generalization.

Phase 5 completed offline retrieval on 163 reactions under 19 synthetic conditions (3,097 condition units). At 75% identifier removal, selected fusion R@10 averaged 0.7301 over three seeds versus 0.8160 clean; its paired change interval included zero. BM25 fell from 0.7362 to 0.5521, while trained-encoder R@10 stayed near its clean 0.7423. Names and equations remained available, so this does not establish robustness to losing all biological information. Seed/condition repetitions are not independent reactions. Historical strict-rule controls remain in the source tables. The corrected matched grounded robustness comparison has not executed; Phase 5 remains paused/incomplete.

The completed 60-case stratified, single-reviewer audit classified {audit['verdict']['label_correct']} source labels as correct, {audit['verdict']['label_incorrect']} as incorrect, {audit['verdict']['multiple_defensible_annotations']} as allowing multiple defensible annotations and {audit['verdict']['not_kegg_mappable']} as not KEGG-mappable. Of 40 returned predictions, 38 were judged correct, one a defensible alternative and one related but incorrect. Of 20 abstaining cases, 11 were judged appropriate, eight inappropriate and one insufficiently informative. Overlapping failure labels and stratum-specific denominators are retained. These are descriptive reviewer judgments, not population error rates, independent adjudication, or corrected benchmark accuracy. Pass 1 hid system alternatives but exposed source evidence/labels; the same system developer reviewed Pass 2 after system evidence was shown, so prior familiarity and anchoring remain limitations.

The predeclared propagation alignment found no compatible machine-prediction exports for the 18 validation models. Existing case-study predictions are outside this paired cohort; none were relabeled as benchmark predictions. A separately labeled offline control comparison therefore recomputed reference-evidence and names-only retrieval on all 969 reactions using the same frozen selected model/catalog. Reference evidence recovered {int(a.hit_at_1_count)}/969 at rank 1 and {int(a.hit_at_10_count)}/969 at rank 10; names-only input recovered {int(c.hit_at_1_count)}/969 and {int(c.hit_at_10_count)}/969. The names-only minus reference R@10 change was {delta10['micro_delta']:.4f} (paired 95% cluster interval {delta10['micro_ci95'][0]:.4f} to {delta10['micro_ci95'][1]:.4f}). Model-macro R@1 was {a.hit_at_1_model_macro:.4f} versus {c.hit_at_1_model_macro:.4f}. Species IDs use a fixed, nonsemantic bijection identically in every arm, after an earlier preserved control exposed identifier-looking source IDs; biological names and equation structure are retained. No changed-input query ranking was replaced by a historical clean ranking. Reversing aliases only for a diagnostic comparison reveals {contract_changes} reference-query identifier changes due to the public qualifier contract. The new baseline is reported separately from historical Phase 3B. Actual predicted-species propagation and all matched GPT-6.1 Terra final-annotation outcomes remain pending.

MODEL2507280001 remains a separate practical case study with Top-10 candidate evidence for all 541 reactions. Candidate coverage and historical rubric review are not benchmark annotation accuracy. No actual GPT-6.1 grounded response cache was found in the prepared case-study directories; inference and biological review remain pending. Its judgments are not pooled with benchmark accuracy.
''')

    text("METRIC_DEFINITIONS_AND_LUNA_CROSSWALK.md", f'''# Metrics and actual evaluator crosswalk

Let G_i be the frozen set of accepted reference reaction IDs and P_i the set of validated asserted reaction IDs. Top-K candidates are **not** P_i. For nonempty P_i and G_i, precision_i = |P_i intersect G_i| / |P_i| and recall_i = |P_i intersect G_i| / |G_i|. Pooled micro precision sums intersections / sums asserted memberships; pooled micro recall sums intersections / sums reference memberships. Set-exact equality is a separate metric from agreement with any accepted alternative.

For single-ID outputs, accepted-reference agreement = 1(P_i intersects G_i). Exact agreement / all evaluable reactions is overall task accuracy; answered / all reactions is coverage; correct / answered is selective accuracy. Single-ID pooled precision equals selective accuracy, but recall need not equal accuracy when G_i has multiple accepted IDs. These reference members can be alternative acceptable annotations rather than simultaneous biochemical requirements; do not turn fractional set recovery into a biological error claim.

The historical pilot illustrates this distinction: {precision['pooled_intersection_count']}/{precision['pooled_assertion_count']} pooled precision = {precision['pooled_micro_precision']:.6f}; {precision['pooled_intersection_count']}/{precision['pooled_reference_membership_count']} pooled recall = {precision['pooled_micro_recall']:.6f}, while overall accepted-reference agreement is 102/163 = {precision['reference_agreement']:.6f}. Mean per-reaction recall is {precision['mean_reaction_recall']:.6f}. Empty predictions have undefined precision (39 pilot cases), are excluded only from mean defined precision, and contribute zero recall when G_i is known. Technical failures remain in overall/recall and failure-rate denominators. All 163 pilot references are evaluable. Missing G_i is non-evaluable for reference metrics but remains in all-input coverage/failure accounting with explicit counts. Planned/unexecuted rows are not scored as failures.

Model-macro accuracy/recall averages per-reaction scores within each model, then weights models equally. Model-macro precision averages defined per-reaction precision within a model, then averages models with defined precision; the pilot has {precision['model_macro_precision_defined_models']} such models of {precision['n_models']}. Models with no asserted output have undefined precision and stay in coverage/overall denominators. Cluster bootstrap resamples whole frozen clusters, retaining their models/reactions and paired conditions. Conditions, seeds and shared species do not create independent reactions.

For retrieval, R@K means the fraction with any accepted reference in Top-K; MRR@10 uses the first accepted reference within 10. Label-recovery recall@K is additionally reported as |TopK intersect G_i|/|G_i|. Neither is candidate-list annotation precision. No final-annotation precision is calculated from ranked candidates.

## Definitions established by reading Luna's actual evaluator

Source: `tests/useCase/evaluate_biomodel_usecases.py` in the preserved Luna worktree; a checksum-bound source snapshot is included under sources. It was **read, not executed**: no publication supplement or external biological resource was queried.

| Evaluator/report | Actual unit and definition | Comparability |
| --- | --- | --- |
| `evaluate_model2506050001`: source_exact_entrez_precision/recall | Gene-ID sets per reference-bearing model node; sum exact intersections / predicted memberships or deposited reference memberships; nodes without reference gene mappings excluded from those ratios | Set formulas align, but units, reference coverage and identity expansion differ from reaction IDs; not directly poolable |
| Same function: reviewed_supported_precision | Adds separately accepted ortholog predictions to exact deposited-source hits | Biological review-augmented numerator, not strict frozen-reference reaction precision |
| `evaluate_model2507280001`: exact_top1_rate | First ChEBI prediction per species; level-0 ChEBI-to-KEGG mapping hit / locally scorable supplement species | Species mapping hit rate with restricted denominator, not reaction precision; tied alternatives beyond the first are not treated as asserted sets |
| Same function: two_hop_top1_rate | Mapping with up to two ontology hops / same scorable species | Relaxed mapping metric; not exact reaction agreement |
| Same function: reviewed_supported_top1_rate | Union of relaxed hits, existing identity/charge/synonym/manual-supported pairs / scorable species | Review-augmented descriptive metric; cannot replace frozen benchmark labels |
| `tests/evaluate_reaction_annotation.py` | Ground-truth-bearing reactions; Top-1/3/5 recovery with optional BRITE collapsing | Retrieval hit/rank evaluation, not full asserted-annotation precision/recall |

The scripts contain special source-specific exclusions and review pair lists. Their column names do not establish common denominators. No direct comparability is claimed beyond the explicit formula crosswalk. New reaction results use their own visible G/P sets and status/denominator policy.
''')

    arms = proposal["arms"]
    species = proposal["species_frontier_separate_cost"]
    arm_lines = "\n".join(f"- {name}: {arm['request_count']} exact frozen requests; full reservation ${arm['reservation_nusd']/1e9:.9f}; proposed cap ${arm['hard_cap_nusd']/1e9:.2f}; plan `{arm['plan_digest']}`."
                          for name, arm in sorted(arms.items()))
    text("EXECUTION_PROPOSAL.md", f'''# Unapproved propagation execution proposal

**Authorized spend: USD 0. Provider calls made: 0. Balance: unverified.** No previously reported ~$20 credit is assumed. Phase 5 has no allocation and stays paused.

## Missing species evidence

The fixed target is 1,063 reaction-participating species across 18 validation models (1,938 species exist in those whole models). No compatible actual predicted table currently covers the 969 reactions. `species_generation/target_species.csv` lists the exact IDs. Sanitized SBML removes annotations/notes, preserves names/stoichiometry/FBC chemistry, and disables model-context prompts. The single proposed species model is Luna's recorded `gpt-5-mini-2025-08-07`, direct chemical/ChEBI retrieval, top_k=3, n_return=1, validation/source_fidelity and separate identity/qualifier assessment. This sanitized, current-stage plan is explicit and is not represented as already observed legacy predictions.

There are **{species['frontier_calls']} exact normalization frontier requests**, max completion output 10,000 including reasoning, each frozen with schema/messages/settings and a model-specific complete-body identity. Their conservative reservation totals **${species['frontier_reservation_usd']:.9f}**, proposed frontier cap **${species['frontier_proposed_cap_usd']:.2f}**. This is NOT a full species-generation price: later database/ranking/support/qualifier prompts depend on missing responses. They are declared dependencies and must be frozen/quoted before any corresponding authorization. No fabricated normalization responses/candidate lists are supplied. Existing public species retry/repair behavior needs a persistent guarded adapter before paid use; the downstream Terra RunState cannot directly execute this Chat Completions frontier.

## Grounded reaction conditions

{arm_lines}

These 1,938 ready A/C requests use GPT-6.1 Terra, low reasoning, strict GroundedAnnotation, 2,048 total output tokens, default tier, store=false and no tools. The B manifest contains 969 explicit `pending_species_evidence` units with no provider payload, not fake names-only predictions. B cannot be fully serialized until actual predictions and fresh B rankings exist. Its conditional upper reservation is ${proposal['predicted_arm_upper_bound_usd_if_every_body_fits_32768_bytes']:.8f} if all 969 bodies fit a predeclared 32,768-byte cap; oversize inputs require explicit failure/protocol handling, never silent truncation or cohort dropping. The conditional three-arm reservation envelope is ${proposal['three_arm_reservation_envelope_usd']:.9f}, excluding unfinished species stages. This envelope is not an authorizable complete B plan.

Reservation = (complete canonical provider-body UTF-8 bytes + 4,096 wrapper allowance) x applicable pessimistic input rate + maximum billable output x output rate, rounded up to nanodollars. This uses request size, not full model context capacity. A byte-to-token upper estimate and the 4,096 wrapper allowance are conservative accounting assumptions, not a provider token-count guarantee. All ready bodies fit the short-context band; no cache-hit discount is assumed. Rates verified on 2026-11-09 from [official pricing](https://developers.openai.com/api/docs/pricing) and [Terra model billing](https://developers.openai.com/api/docs/models/gpt-6.1-terra): input $1/M, 30-minute cache-write $1.25/M, output $5/M; mini input $0.25/M and output $2/M. No Fast/Batch/Flex/regional surcharge or discount is selected. Taxes, currency conversion and later price/provider changes are excluded; recheck before authorization. If retention/tier changes, regenerate every plan/reservation.

The existing persistent RunState and full-plan verifier accept each separate 969-unit A/C plan unchanged, with distinct condition-qualified units and separate state directories. Dry-run states have no reserved attempts. Same model/reaction IDs cannot coexist as multiple arms in one legacy verifier, hence per-arm plans; a controller must interleave arms and enforce the aggregate cap/provider cohort. Responses from GPT-5.6 are incompatible, and no actual GPT-6.1 cache hits are available. Eight A/C pairs have identical complete bodies; any future response reuse requires exact compatibility and linked provenance, and does not create independent observations. No budget savings are assumed from such reuse.

Failed, invalid, interrupted or uncertain attempts retain their pessimistic reservation. Surplus may be released only after successful verified billing settlement under the existing runner policy; never automatically on an unverified usage estimate. No retries or silent repurchases. Incompleteness, empty output, refusal, schema/evidence violations and biological abstention remain separate statuses and every reaction stays in coverage. Equal alias/body identity cannot prove an unchanged hosted backend; log requested/returned identities, endpoint/SDK, request/response IDs, timestamp and usage, interleave conditions, and stop on identity drift.

On Windows, use a short per-arm state directory (for example under TEMP/aaaim_propagation) or explicitly verified long-path support. Deep worktree/cache/digest paths exceed the platform filename limit; the mocked interruption/resume tests use a short temporary directory. The dry-run state inside each plan is only a zero-attempt validation artifact, not a prescribed live state path.

Plan directory: `{execution}`. Both ready arm caps cover all requests even when every reservation is retained. Full ABC inference and biological review are pending, not manuscript results.
''')

    text("HELD_OUT_READINESS.md", f'''# Proposed later held-out evaluation: not final, not executed

Source drafts are copied under sources/historical_draft_evaluation_freeze.json and selected_method.json from `{phase5_root}/benchmark/phase5/freeze`. They are inspected as protocol metadata only; no held-out query/label is opened.

Required before freezing:

1. Obtain provenance-verified predicted species evidence for the fixed validation cohort; finish the actual B retrieval arm and matched A/B/C grounded comparison. Do not use the completed synthetic Phase 5 curves as a substitute. Decide the production species-input policy from these development findings, not held-out outcomes.
2. Validate the GPT-6.1 model/settings change with actual authorized validation responses; historical GPT-5.6 results do not establish GPT-6.1 behavior. Freeze the selected single pipeline, source/modern/legacy evidence policy and any biological review scope. Independent review remains deferred for the completed audit, not a reopened gate.
3. Decide explicitly whether the deferred Phase 5 matched grounded experiment remains a final-freeze requirement or is deferred from scope. It is paused/incomplete; this package neither unpauses nor marks it complete.
4. Pin catalog, encoder/tokenizer/manifest, query/evidence serialization, prompts/schema, metric definitions, reference alternatives/evaluability rules, device/precision/environment, hashes and code state. Keep validation-selected architecture/threshold optimism disclosed.
5. Replace stale failure-policy language in the historical draft before final freeze: oversize/schema/compliance errors are technical failures, not biological abstentions. Preserve the original draft unchanged. Final policy includes no automatic retries, all-reaction coverage, explicit abstention and error subtypes, pessimistic reservation before attempt and verified billing settlement.
6. Implement and synthetically verify the sealed-query executor/controller, label-sealing integration, interleaving/provider identity gate, persistent resumability, aggregate budget guard and crash behavior. The public API alone lacks those safeguards. The existing draft's fail-closed --execute placeholder is not a finished evaluator.
7. Obtain explicit permission for the later sealed-query/request-plan preparation stage, keep labels sealed, quote complete frozen request sizes/settings under then-current pricing and verify available credit. Earlier rough 1,350-call/old-context estimates in the draft are metadata proposals, not a current spend authorization or held-out result.
8. Obtain exact-plan spending authorization; run the one-time locked protocol with no selective repurchases, freeze predictions/statuses and coverage hashes, then let the separate evaluator join sealed labels exactly once. Report denominators, uncertainty and failures without retuning on held-out outcomes.

Open now: B species inputs, B retrieval, all GPT-6.1 final propagation outcomes, final species policy, explicit scope decisions, executor/controller/label gate, provider/budget approval. The held-out freeze remains **draft**. No held-out data was accessed for this checklist.
''')

    case_root = prediction_root / "results/MODEL2507280001"
    case_caches = [str(p) for p in case_root.rglob("cache") if p.is_dir() and list(p.glob("*.json"))]
    write_json(package / "case_study/status.json", {"model_id": "MODEL2507280001", "reaction_count": 541,
        "top10_candidate_evidence_coverage": "541/541, previously verified frozen artifacts", "actual_response_cache_directories_found": case_caches,
        "grounded_annotation": "pending unless separately verified actual outputs provided", "biological_review": "pending",
        "not_pooled_with_benchmark": True, "candidate_coverage_is_not_accuracy": True})
    write_csv(package / "case_study/prediction_review_template.csv", [], ["model_id", "reaction_id", "request_sha256", "status",
        "predicted_kegg_id", "biological_abstain", "technical_failure_type", "reviewer_judgment", "reviewer_rationale",
        "reviewer_sources", "representative_example_reason", "independent_reference_if_available"])
    text("case_study/REPORTING_TEMPLATE.md", '''# Separate practical case-study report

MODEL2507280001 has frozen Top-10 retrieval evidence for all 541 reactions. Report 541/all-source-reaction candidate coverage separately from actual annotation outcomes. Preserve all 541 in a status table: valid selections, biological abstentions, input/resource/provider failures, incomplete/refusal/empty/schema/evidence failures and not-attempted units. Candidate presence does not establish accuracy.

After separately authorized actual inference, record exact plan/model/settings/body hashes, usage/billing/provider cohort and source species evidence. Review biological examples only from actual outputs. Existing historical rubric judgments remain descriptive and cannot be reused as independent benchmark truth. If no independently justified reaction reference set exists, do not manufacture an accuracy score; report reviewer counts and limitations instead. Use the CSV template for eventual reviewer findings and examples. Current grounded inference and biological review remain pending.
''')
    evaluator = prediction_root / "evaluate_biomodel_usecases.py"
    shutil.copyfile(evaluator, package / "sources/luna_evaluator_snapshot.py")
    write_json(package / "sources/luna_evaluator_provenance.json", {"path": str(evaluator), "sha256": sha256(evaluator),
        "executed": False, "external_biological_resources_queried": False})
    write_csv(package / "audit/new_assessment_packet_template.csv", [], ["model_id", "reaction_id", "condition", "actual_prediction",
        "available_reference", "uncertainty_reason", "human_question", "reviewer_response", "supporting_source"])
    text("COMPLETION_CHECKLIST.md", '''# Completion and remaining work

- Complete: verified historical retrieval, historical GPT-5.6 pilot, cached Phase 4 and executed Phase 5 retrieval tables/figures; accepted audit revalidated without new judgments; source pointers/checksums and metric crosswalk.
- Complete: versioned paired target protocol/alignment/input freezes, source/names-only offline controls, paired transitions/model macro/cluster uncertainty, safe no-fallback construction and exact A/C grounded request plans.
- Complete: exact missing-species normalization frontier, separately costed; conditional later dependencies explicitly pending; existing downstream budget runner accepts separate arm plans without live execution.
- Pending inputs: compatible actual predicted species tables for validation models (not the separate case-study predictions). Target 1,063 participating species. No B results fabricated.
- Pending: B rankings and full three-arm final annotation; later species prompts/costs, guarded species adapter and aggregate interleaving/provider controller; exact paid authorization and verified balance.
- Pending: final held-out protocol decisions/executor and later one-time authorized evaluation. Existing freeze is draft.
- Separate case study: MODEL2507280001 inference/biological review pending. Phase 5 paid work stays paused.
- Not required to finish the accepted audit: an independent second review or adjudication. Existing uncertainty/flags remain documented limitations. No corrected benchmark accuracy is computed.
- No paid calls, held-out access, frozen-artifact edits, external messages, commits, pushes or merges occurred in this task.
''')
    text("README.md", f'''# AAAIM reaction manuscript results package (2026-11-09)

This package contains development findings only. Start with RESULTS.md, tables/TABLE1_RETRIEVAL_969.md, tables/TABLE_PROPAGATION_CONTROLS.md and METRIC_DEFINITIONS_AND_LUNA_CROSSWALK.md. Publication figures are standalone PNG/SVG/PDF under figures. Each CSV/JSON points to the original source artifact and denominator; bootstrap files retain their original resampling contracts. Audit case index/examples are private and remain ignored by Git.

Populations: full validation 969 reactions / 18 models / 12 clusters; historical stratified pilot 163 / 17 models / 12 clusters; formal audit 60 stratified cases by one reviewer in two stages. Do not pool these populations or reinterpret unweighted audit/pilot proportions as population error rates. Historical inference is GPT-5.6 Terra; proposed new final annotation is GPT-6.1 Terra and has no results.

The actual-prediction propagation cohort currently has zero compatible observed B reactions. Available machine predictions concern separate case-study models; additional legacy naming is inventoried separately. New A/C control rankings were recomputed on all fixed validation keys. All original frozen labels, rankings, responses and accepted audit judgments remain unchanged. The prior Phase 5 global checksum ledger is stale for twelve later planning/doc files; numerical retrieval entries match, and sources/verification.json lists every mismatch rather than repairing history.

Large reproducibility artifacts are retained outside this report folder:

- Protocol/cohort/queries/rankings/evidence/cache: `{study}`
- Exact no-call requests, per-arm plans and pending B manifest: `{execution}`
- Source/audit originals: `{source_root}` (pointers/checksums under sources)

Only new query-ranking CPU work and local descriptive analysis were executed. No provider inference or held-out evaluation was executed. See EXECUTION_PROPOSAL.md, HELD_OUT_READINESS.md, case_study and COMPLETION_CHECKLIST.md for precise pending work.
''')
    text("REPRODUCE.md", '''# Reproduce into fresh ignored output directories

From this isolated manuscript_results worktree, use the preserved source-root and phase worktrees. Do not overwrite any output directory; choose NEW suffixes. No command below enables paid inference or held-out evaluation.

```powershell
python -m benchmark.scripts.manuscript_propagation prepare --source-root ../../.. --prediction-root ../luna_public/tests/useCase --out _manuscript_results_work/propagation_NEW --alias-source-ids
python -m benchmark.scripts.manuscript_propagation rank --source-root ../../.. --prediction-root ../luna_public/tests/useCase --out _manuscript_results_work/propagation_NEW --encoder ../rrf_grounded_api/_rrf_grounded_work/encoder --catalog-cache _manuscript_results_work/propagation_v2
python -m benchmark.scripts.manuscript_propagation analyze --source-root ../../.. --prediction-root ../luna_public/tests/useCase --out _manuscript_results_work/propagation_NEW
python -m benchmark.scripts.manuscript_results --source-root ../../.. --phase4-root ../../../benchmark/phase3/_phase4_worktree --phase5-root ../../../benchmark/phase3/_phase5_worktree --prediction-root ../luna_public/tests/useCase --out _manuscript_results_work/package_NEW
python -m benchmark.scripts.manuscript_plans --study _manuscript_results_work/propagation_NEW --luna-root ../luna_public --pricing _manuscript_results_work/pricing_snapshot_20261109.json --out _manuscript_results_work/execution_NEW
python -m benchmark.scripts.manuscript_handoff --package _manuscript_results_work/package_NEW --study _manuscript_results_work/propagation_NEW --execution _manuscript_results_work/execution_NEW --source-root ../../.. --prediction-root ../luna_public/tests/useCase --phase5-root ../../../benchmark/phase3/_phase5_worktree
```

The expensive ranking command builds one verified full-catalog CPU index and memoizes only exact query texts; the stored matrix has resource/environment/hash provenance. It does not train/tune. Protocol and query freezes precede ranking; label extraction from validation-whitelisted source SBML follows ranking freeze. The same extractor used by frozen benchmark construction reproduces all 969 old BM25 score checks and all 163 frozen pilot reference sets. No mixed-split label CSV or held-out SBML is read. Existing scripts/outputs remain intact.

The complete manuscript tables/figures are reproducible from small frozen artifacts; rerunning the expensive retrieval command is unnecessary to reproduce historical tables or tests. Keep private reports ignored. Source code is uncommitted and is pinned by package SHA-256 manifests rather than an invented commit ID.
''')
    import sys
    plotting_deps = source_root / "benchmark/phase3/_phase4_worktree/benchmark/phase4/_deps"
    if plotting_deps.is_dir():
        sys.path.append(str(plotting_deps))
    control_figure(package, paired)
    pilot_figure(package, precision)
    text("FIGURE_CAPTIONS.md", '''# Figure captions

1. **retrieval_969**: Executed full-catalog validation retrieval on 969 reactions (18 models, 12 clusters). R@K is any accepted-reference hit, not asserted annotation precision. The selected trained encoder is epoch-1 BGE-small; RRF uses equal Top-100 components and k=60. Historical methods outside current production scope remain visible. Source: tables/retrieval_969.csv; uncertainty in retrieval_969_uncertainty.json.
2. **grounded_pilot_163**: Separate stratified 163-reaction historical GPT-5.6 Terra pilot (17 models, 12 clusters). Counts sum to 163 for both methods. Two of 39 historically abstaining grounded outputs carry evidence-compliance violations and are shown as technical failures under the explicit crosswalk, with the historical report unchanged. Source: grounded_pilot_163.csv and pilot_asserted_precision_recall.json. These are not GPT-6.1 results.
3. **cached_routing_163**: Reuse of historical outputs, not new inference. The x axis is simulated calls selected for replay; actual new calls and spending were zero. Original policy selection retained grounded annotation. Source: phase4_cached_routing.csv and phase4_uncertainty.json. Grounded confidence and retrieval margins are not calibrated probabilities.
4. **synthetic_robustness_163**: Executed Phase 5 retrieval perturbations, mean across three seeds at each nonzero identifier-removal fraction. Unique reactions remain 163; 489 reaction-seed observations are not independent reactions. Names and equations remain available. Source: phase5_retrieval_by_condition.csv and phase5_retrieval_families.csv, including paired cluster/seed delta intervals. This is not actual species-prediction propagation, and the paid Phase 5 comparison remains paused.
5. **human_audit_counts_60**: Counts from the accepted one-reviewer, two-stage stratified audit. Pass 1 assessed source evidence/labels with system alternatives hidden; Pass 2 revealed system evidence. The graph is descriptive and unweighted, not a population error-rate estimate or inter-reviewer agreement. Stratum denominators and overlapping taxonomy labels are in audit_counts_by_stratum.csv; private case index/examples remain separate.
6. **reference_names_control_differences**: Newly executed, predeclared available-reference versus names-only retrieval controls on the same 969 validation reactions, 18 models and 12 clusters. Error bars are 2,000-replicate paired cluster percentile intervals for names-only minus reference. No actual predicted-species arm exists yet. Public qualifier filtering changes 49 reference query inputs relative to historical Phase 3B; this baseline does not replace the original results. Source: propagation_control_summary.csv and paired_reference_names_only.json.

All figures have standalone PNG, SVG and PDF versions. All findings are development findings; the held-out test remained sealed.
''')
    paths = [p for p in package.rglob("*") if p.is_file() and "_mplconfig" not in p.parts]
    write_json(package / "MANIFEST.json", {"schema": "manuscript-results-private-v1",
        "files": [{"path": p.relative_to(package).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(paths)],
        "study": str(study), "execution": str(execution), "protocol_sha256": sha256(study / "protocol.json"),
        "rankings_sha256": frozen["rankings_sha256"], "paid_calls": 0, "heldout_content_read": False})
    return {"package": str(package), "files": len(paths), "historical_reference_queries_changed_by_public_contract": len(changes),
            "additional_prediction_models_checked": len(additional), "pending_B_reactions": 969, "paid_calls": 0}


def context_diagnostic(package, study, source_root):
    reference = [r for r in read_jsonl(Path(study) / "queries.jsonl") if r["condition"] == "reference"]
    pattern = re.compile(r"CHEBI[:_]\d+|(?<![A-Za-z0-9])C\d{5}(?!\d)", re.I)
    species = {(r["model_id"], sid): info["name"] for r in reference for sid, info in r["species_evidence"].items()}
    sid_matches = {key for key in species if pattern.search(key[1])}
    name_matches = {key for key, name in species.items() if pattern.search(name)}
    affected = sid_matches | name_matches
    alignment = read_json(Path(study) / "alignment.json")
    inverse = {r["model_id"]: {alias: sid for sid, alias in r.get("species_aliases", {}).items()} for r in alignment["validation_models"]}
    raw_identifier_ids = {key for key in species if pattern.search(inverse[key[0]].get(key[1], key[1]))}
    rows = [{"model_id": mid, "species_id": sid, "name": species[(mid, sid)],
             "identifier_like_species_id": (mid, sid) in sid_matches,
             "identifier_like_name": (mid, sid) in name_matches} for mid, sid in sorted(affected)]
    write_csv(Path(package) / "protocols/identifier_like_source_context_private.csv", rows)
    prior = pd.read_csv(Path(source_root) / "benchmark/phase3/retrieval_baselines/scored_bm25.csv")
    summary = {"participating_species": len(species), "identifier_like_ids": len(sid_matches),
        "identifier_like_names": len(name_matches), "identifier_like_species_union": len(affected),
        "raw_source_identifier_like_ids": len(raw_identifier_ids),
        "uniform_species_id_aliases": read_json(Path(study) / "protocol.json").get("uniform_species_id_aliases", False),
        "reactions_with_identifier_like_source_context": sum(any((r["model_id"], sid) in affected for sid in r["species_evidence"]) for r in reference),
        "validation_reactions_with_reference_target_seen_in_training": int(prior.seen_in_train.sum()),
        "validation_reactions_with_reference_target_unseen_in_training": int((~prior.seen_in_train).sum()),
        "model_cluster_partition_overlap": 0,
        "interpretation": "source names and equation structure identical in every arm; supplied annotation fields absent in C; verify current alias/name matches below",
        "source_id_decision": "uniform nonsemantic aliases frozen in revised protocol before its outcomes; prior source-ID controls preserved separately; do not mutate either freeze",
        "hosted_model_pretraining_overlap": "unknown"}
    write_json(Path(package) / "protocols/context_and_training_diagnostic.json", summary)
    (Path(package) / "CONTEXT_AND_TRAINING_LIMITATIONS.md").write_text(
        f"# Source context and training overlap\n\nThe C condition removes supplied annotation identifier fields and suppresses SBML fallback. "
        f"It retains the same source names, species IDs in equations/participant records, and stoichiometry in every arm. "
        f"{len(raw_identifier_ids)}/{len(species)} original source species IDs were identifier-like. "
        f"With the protocol's uniform alias policy, {len(affected)}/{len(species)} model-visible participating IDs/names are identifier-looking; "
        f"{summary['reactions_with_identifier_like_source_context']}/969 reactions contain that context. "
        "The initial source-context controls are preserved under propagation_v2, including their intrinsic-ID limitation. "
        "The revised aliased protocol was frozen before its new comparative outcomes; all changed query rankings were recomputed. "
        "Biological names remain available by design; no claim of zero biological information or zero pretraining knowledge is made. "
        "Original IDs are kept only in alignment sidecars, including the inverse map for actual species-prediction generation.\n\n"
        f"Frozen training metadata records {summary['validation_reactions_with_reference_target_seen_in_training']}/969 validation reactions with a reference target seen in training "
        f"and {summary['validation_reactions_with_reference_target_unseen_in_training']}/969 without one. "
        "These are reaction counts, not unique target counts. Validation model and cluster allocations are disjoint from training; "
        "that does not imply novel reaction targets or absence of hosted-model pretraining memorization. The encoder and thresholds were validation-selected. "
        "All findings remain development findings. Existing prediction cases are outside the frozen validation cohort; prediction disagreements are not verified biological errors.\n",
        encoding="utf-8")


def control_figure(package, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "aaaim-manuscript-results-v1"})
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr_at_10"]
    values = [paired["metrics"][m]["micro_delta"] for m in metrics]
    low = [v - paired["metrics"][m]["micro_ci95"][0] for m, v in zip(metrics, values)]
    high = [paired["metrics"][m]["micro_ci95"][1] - v for m, v in zip(metrics, values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(np.arange(len(metrics)), values, yerr=[low, high], fmt="o", capsize=4)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(range(len(metrics)), ["Hit R@1", "Hit R@3", "Hit R@5", "Hit R@10", "MRR@10"])
    ax.set(ylabel="Names-only minus reference (paired cluster 95% interval)",
        title="Available-evidence controls: 969 reactions / 18 models / 12 clusters\nActual-prediction arm remains pending")
    fig.tight_layout()
    for suffix in ["png", "svg", "pdf"]:
        metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Software": "AAAIM manuscript results v1"}
        fig.savefig(package / "figures" / ("reference_names_control_differences." + suffix), dpi=200, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def pilot_figure(package, precision):
    import matplotlib.pyplot as plt
    import numpy as np
    values = [("Correct selection", [93, precision["correct"]], "#2878A2"),
              ("Incorrect selection", [70, precision["answered_evaluable"] - precision["correct"]], "#D89044"),
              ("Valid abstention", [0, precision["biological_abstentions"]], "#A8B0B7"),
              ("Technical failure", [0, precision["technical_failures"]], "#AE4654")]
    fig, ax = plt.subplots(figsize=(7, 4))
    left = np.zeros(2)
    for label, counts, color in values:
        bars = ax.barh(["Fusion Top-1", "Grounded GPT-5.6 Terra"], counts, left=left, label=label, color=color)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in counts], label_type="center", fontsize=9)
        left += counts
    ax.set(xlim=(0, 163), xlabel="Reactions / separate stratified pilot of 163",
           title="Historical actual grounded inference: 17 models / 12 clusters")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ["png", "svg", "pdf"]:
        metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Software": "AAAIM manuscript results v1"}
        fig.savefig(package / "figures" / ("grounded_pilot_163." + suffix), dpi=200, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["package", "study", "execution", "source-root", "prediction-root", "phase5-root"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.package, args.study, args.execution, args.source_root, args.prediction_root, args.phase5_root), indent=2))
