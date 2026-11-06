# AAAIM research project: goals and roadmap

**Status updated:** 2026-11-06
**Current priority:** Phase 5 robustness development on validation data, followed by final method/evaluation freeze, then a separately authorized held-out evaluation. Phase 4's bounded study is complete; manuscript preparation may proceed in parallel.

## Current status and agreed sequence

This update preserves historical benchmark results and revises the roadmap using repository evidence and the user's explicit scope decisions. Current repository reports remain authoritative for exact configurations and metrics. The Phase 4 handoff, Phase 5 local retrieval artifacts and listed read-only verifiers were checked for this update; completion of other workstreams is not inferred.

| Workstream | Status |
|---|---|
| Phase 1 frozen benchmark | Complete |
| Phase 2 rule-based retrieval baseline | Complete and frozen |
| Phase 3A direct/open-set LLM pilot | Direct validation experiments completed; the broader original multi-mode proposal is not automatically fulfilled |
| Phase 3B learned full-catalog retrieval | Bi-encoder trained; lexical, dense, and fusion comparisons performed; verify exact experiment coverage in repository reports |
| Phase 3C grounded-fusion validation pilot | Complete under its implemented scope |
| Phase 3C human audit | Complete as a scoped single-reviewer, two-pass descriptive audit |
| Phase 4 routing, uncertainty, and abstention | Bounded cached study complete and verified; routing failed the prespecified improvement requirement; grounded-fusion method retained |
| Final method and evaluation freeze | Draft; after Phase 5 development and required outputs, not before |
| Held-out test evaluation | Remains sealed; later one-time evaluation after the final freeze and run authorization |
| Phase 5 robustness experiments | Local retrieval study complete; corrected matched clean/degraded prompt design prepared, requiring 326 unit-level outputs/authorization; not fully complete |
| Phase 6 packaging and manuscript | Remaining work; may proceed alongside Phase 5 |

**Agreed order:** preserve completed Phase 3C baseline and Phase 4 selection → Phase 5 development → final method/evaluation freeze → later separately authorized held-out evaluation → finalize research reporting.

Phase 4 was originally stretch work and is now complete under its bounded protocol. The user has explicitly chosen Phase 5 before the final freeze. This does not make every older proposed experiment a requirement. Earlier calendar targets below are historical, not current deadlines.

### Other workstreams requiring a repository status check

Upstream integration and public API tests, documentation fixes, EM ablation/default selection, species-side retrieval comparisons, and uncurated-model case studies remain in this roadmap. Their completion is not established by closing reaction-side Phase 3C. Review evidence and assign ownership before treating them as complete or as blockers for the manuscript.

## Project objective

AAAIM is already a biological annotation tool. This project is therefore not simply to build an annotation tool, but to turn AAAIM into a reproducible research system for studying metabolic reaction annotation when metabolite evidence and database retrieval are incomplete.

The central research question is:

> Can learned retrieval, compact-context LLM inference, and tool-assisted recovery improve metabolic reaction annotation while recognizing when available evidence is insufficient?

A current working title is:

> **Metabolic Reaction Annotation Under Incomplete Biological Evidence: Benchmarking Rule-Based Retrieval, Learned Models, and Tool-Augmented LLMs**

The desired final artifact is:

- A frozen, reproducible benchmark with leakage-resistant model clusters
- Credible rule-based, lexical, embedding, learned, and LLM baselines
- At least one small neural model trained for full-database retrieval or ranking
- A rigorous separation of retrieval failure, reranking failure, and open-set recovery
- Calibration, abstention, and selective-accuracy evaluation
- Robustness experiments under degraded metabolite annotations
- A public experimental pipeline and research-oriented README
- A preprint or technical report

## What the repository investigation established

Substantial evaluation work existed on the unmerged upstream branch `sys-bio/AAAIM:test/no-rule-evals`, but it was stranded and not reproducible from `main`. It contained a 75-model manifest, ground-truth extraction, ChEBI-to-KEGG candidate generation, rule-based and LLM reranking, Top-k evaluation, and several ablations.

The historical claim of 68 models and 4,379 reactions could not be reproduced. The historical script referenced an uncommitted input list, and the relevant outputs were never committed. The new benchmark therefore records the observed corpus transparently instead of changing exclusions to match the historical numbers.

The reconstruction also uncovered several silent-corruption bugs, including URI-pattern mismatches, reaction-ID misalignment after filtering, empty-set candidate explosions, and model-wide loss after a single reaction error. These are now fixed and pinned by tests.

## Phase 1: Frozen benchmark — complete

Phase 1 recovered and verified all 75 BioModels accessions, pinned source files by checksum, documented exclusions, preserved multiple valid KEGG identifiers, detected related-model clusters, and produced deterministic benchmark artifacts.

Observed benchmark:

| Quantity | Result |
|---|---:|
| Manifest models | 75 |
| Included models | 74 |
| Scientifically excluded models | 1 |
| Ground-truth reactions | 5,838 |
| Evaluable reactions | 5,816 |
| Exchange/SSX exclusions | 19 |
| Malformed-ID exclusions | 3 |
| Reactions with multiple valid KEGG IDs | 91 |
| Leakage-control model clusters | 54 |

The Phase 1 snapshot is tagged `benchmark-phase1-v1`.

## Phase 2: Candidate retrieval and baseline evaluation — complete and frozen

Phase 2 froze the rule-based candidate-generation outputs and separated candidate retrieval from candidate reranking. The complete run assembled all 74 models with zero pipeline failures.

Observed candidate-generation results:

| Status | Reactions |
|---|---:|
| Nonempty candidate set (`ok`) | 2,359 |
| Empty constrained search (`no_candidates`) | 2,646 |
| No mapped participants (`unconstrained_candidate_set`) | 811 |
| Total evaluable | 5,816 |
| Candidate rows | 91,802 |

Headline retrieval results:

| Metric | Exact | BRITE/orthology-aware |
|---|---:|---:|
| Recall at any rank, reaction micro | 34.7% | 35.1% |
| Recall@1, reaction micro | 33.3% | 34.3% |
| Recall at any rank, model macro | 43.8% | 44.9% |
| Recall@1, model macro | 35.5% | 40.0% |
| Recall at any rank, cluster macro | 42.4% | 43.5% |
| Recall@1, cluster macro | 35.9% | 38.9% |

The zero-candidate rate is 59.4%. Overall exact retrieval failure is therefore approximately 65.3%, including both zero-candidate reactions and reactions whose nonempty set lacks the correct answer.

Conditional on the 2,359 reactions with nonempty candidate sets:

| Ranker | Exact Top-1 |
|---|---:|
| Existing heuristic | 82.0% |
| Lexical | 81.6% |
| Random | 78.1% |
| Oracle | 85.6% |

The key conclusion is that **retrieval, not reranking, is the dominant bottleneck**. The existing heuristic is only 3.6 percentage points below the oracle among reactions with candidates. Across the complete benchmark, the gap between exact candidate ceiling and existing Top-1 performance is only about 1.4 percentage points.

The Phase 2 reporting and reproducibility issues have been resolved. Overall retrieval failure is 65.29% of all 5,816 evaluable reactions; conditional retrieval failure is 14.41% among the 2,359 reactions with nonempty candidate sets. `BIOMD0000001063` was confirmed to contain genuine, extremely large candidate sets caused mainly by reactions with only one mapped participant, not duplicate rows or a pipeline defect. Per-reaction metrics weight each reaction once.

All Phase 2 invariants pass, aggregate artifacts and per-model caches are restorable, and verification is read-only. The snapshot is tagged `benchmark-phase2-v1`.

## Upstream integration work — completion status to verify

Upstream `main` now integrates species annotation and reaction annotation into one public workflow. Commit `1611464db16a2f55c0fc02a983275cdaa21cab26` introduced the documented `annotate="species"`, `annotate="reactions"`, and `annotate="both"` modes; subsequent upstream commits adjusted ranking and messages.

This integration is strategically important because it turns the benchmarked reaction code into an end-user workflow. Its validation remains a separate product-readiness workstream. Reconcile its current status with repository evidence; do not infer completion from the reaction benchmark experiments.

### Required integration validation

- Reconcile current upstream `main` with the benchmark branches without losing the Phase 1–3 fixes. Do this deliberately; do not assume a clean merge because the reaction code evolved on both lines.
- Build an end-to-end test matrix for:
  - species-only annotation;
  - reaction-only annotation from ChEBI annotations already in SBML;
  - reaction-only annotation from a species recommendation DataFrame;
  - reaction-only annotation from a species recommendation CSV;
  - combined species-then-reaction annotation;
  - empty or unusable species results;
  - direct and RAG species retrieval;
  - EM enabled and disabled;
  - mocked LLM failures and malformed responses.
- Assert output schemas, species-to-reaction handoff, reaction/species IDs, saved files, metrics, and `AnnotationResult` attributes—not merely that the call does not raise.
- Add at least one complete SBML integration test with mocked LLM responses. Existing tests around commit `1611464` mainly cover helper behavior.
- Resolve the return-type inconsistency in error paths: public documentation says annotation calls return `AnnotationResult`, while some reaction failures still return a raw `(DataFrame, metrics)` tuple.
- Confirm whether direct KEGG-compound species annotations are supported through the public combined/reaction wrapper. The documentation currently says “ChEBI (or KEGG compound),” but `_load_species_recommendations()` filters inputs to ChEBI rows. Either implement and test KEGG-compound inputs or narrow the documentation.

### Documentation review for commit `1611464`

The new documentation is a useful foundation and correctly exposes the combined workflow. Before considering it final:

- Verify every example against the current public API and run examples in CI where practical.
- Document `top_k` and `n_return` separately and make clear that reaction retrieval does not use `top_k` in the same way as species retrieval.
- Put the EM runtime/default behavior in the main reaction-annotation section rather than only in advanced parameters.
- Document failure behavior, required reference files, output filenames, and the exact provenance accepted for species annotations.
- Correct small source/documentation hygiene issues found during review, including duplicated argument/assignment lines, while avoiding unrelated refactors.

## Retrieval modernization across annotation types

The current method named `rag` is a dense vector-retrieval pipeline built with ChromaDB. ChEBI, NCBI Gene, and UniProt index entries are mostly individual names/synonyms embedded with `all-MiniLM-L6-v2`; an older optional path uses `text-embedding-ada-002`. This is functional, but it should not be treated as the only modern retrieval baseline. Retrieval modernization applies to species annotations across ChEBI, NCBI Gene, and UniProt and to full-catalog KEGG reaction retrieval, although each database must be evaluated separately.

LunaStarr confirmed that she meant **BM25**, a lexical ranking method, and that her concern applies to all supported annotation types. BM25 is especially plausible for ontology entity linking because exact names, abbreviations, identifiers, and rare tokens can be more informative than generic semantic similarity. The particular paper or implementation she had in mind can be collected later; it is not a prerequisite for implementing the standard baseline.

### Feasible retrieval experiment

Do not replace the existing dense RAG path immediately. Implement a common retriever interface and compare:

1. Current direct dictionary/synonym matching.
2. BM25 lexical retrieval over canonical names plus synonyms.
3. Current dense embedding retrieval.
4. A simple hybrid of BM25 and dense retrieval using reciprocal rank fusion (RRF), with candidate union retained as an auditable intermediate.

Evaluate candidate Recall@1/3/10, latency, index size, and downstream LLM-selection accuracy separately for ChEBI, NCBI Gene, UniProt, and KEGG reactions wherever labeled evaluation data are available. Use the same frozen inputs and ground truth for all methods. Only promote a new default after the comparison. Updating the dense encoder is a later option, not a prerequisite for the BM25 baseline.

BM25 remains a strong, interpretable baseline rather than an obsolete method. More elaborate retrievers such as SPLADE or ColBERT are out of the immediate scope unless BM25, dense retrieval, and their hybrid expose a clear failure that justifies the additional complexity.

## Expectation-maximization decision

The EM-style participant/reaction update loop remains active in the reaction pipeline. It defaults to five iterations but can already be disabled with `em_max_iterations=0`. Because it is expensive, the immediate goal is to determine whether it earns its runtime—not to delete it first.

### EM ablation and decision rule

- Benchmark `em_max_iterations=0`, `1`, `2`, and `5` on a small but representative set spanning model sizes and Phase 2 retrieval strata.
- Measure wall-clock time, candidate recall, Top-1 accuracy, number of recovered reactions, and any newly introduced false candidates.
- If zero or one iteration preserves essentially all useful accuracy while materially reducing runtime, make the faster setting the default and retain the iterative method as an explicitly experimental option.
- If EM provides no reproducible gain, deprecate it in the public path before considering code removal.
- Run a full-corpus EM comparison only if the small ablation shows a meaningful benefit; do not commit to another multi-day run without that evidence.

## Uncurated BioModels case studies

The new uncurated BioModels examples are external-use cases, not a replacement for the curated benchmark. Because they lack reaction-level ground truth, do not report their output as accuracy.

For each selected model:

- Record the accession/revision, selection rationale, model size, existing annotation coverage, and whether it was used during development.
- Run the frozen workflow and capture runtime, token cost, candidate coverage, abstentions, and proposed species/reaction annotations.
- Have a domain expert manually review a predefined sample, with evidence links and an explicit rubric.
- Report these as qualitative case studies or prospective annotation demonstrations.
- Keep test models chosen after method decisions separate from examples used to debug the workflow.

## Phase 3: Recover answers missing from rule-based retrieval

Phase 3 should address the observed bottleneck directly. It should not begin as a large reranking project confined to the existing candidate sets.

### Phase 3A: Direct/open-set LLM validation — implemented experiments completed

The cluster-separated split contains 3,497 train, 969 validation, and 1,350 test reactions. The exploratory validation pilot contains 163 reactions across five retrieval strata. Historical scaffolding considered three bounded context variants (489 prompts); do not infer that all scaffolded conditions were executed.

Direct LLM identification was evaluated as a research probe without retrieved candidate lists. Grounded retrieval and selection were subsequently evaluated in Phase 3C. Direct, tool-assisted, and closed-set conditions must be distinguished in the paper; the original proposal does not establish that every mode or context variant was completed.

KEGG-shaped reaction-ID leakage was detected and addressed during development. Preserve the implemented leakage checks, cached responses, runner accounting, and exact experiment provenance. Refer to frozen repository reports for executed model IDs, budgets, configurations, and results rather than the older planned starting model or credit allocation.

The held-out test has not been used for method selection and must remain sealed.

### Phase 3B: Learned full-catalog retrieval — implemented study completed

A small pretrained bi-encoder was fine-tuned for full-catalog KEGG reaction retrieval. The implemented comparisons include BM25 lexical retrieval, off-the-shelf dense retrieval, learned retrieval, and reciprocal-rank fusion. The selected BM25–fine-tuned-bi-encoder fusion supplies the grounded Phase 3C candidate evidence.

Preserve model-cluster separation, query leakage guards, frozen catalog records, training/checkpoint provenance, negative-sampling details, and retrieval outputs. Use repository reports for exact encoder/checkpoint IDs, Recall@k, MRR, latency, and uncertainty estimates.

Do not assume that every proposed context ablation, rule-based candidate union, calibration experiment, or species-side retrieval comparison was performed. Identify these separately as implemented, deferred, or unverified.

### Phase 3C: Grounded-fusion validation study — complete under implemented scope

The implemented study used retrieved KEGG candidates and reaction-local evidence for constrained LLM selection or abstention on the 163-reaction validation pilot. Preserve its exact method, outputs, mechanical metrics, and reports as the Phase 4 baseline.

**Scope decision:** accept the implemented grounded-fusion validation pilot and scoped single-reviewer audit as completed Phase 3C work. The original broader comparison of heuristic, lexical, embedding, trained cross-encoder, and LLM rerankers is **not declared fulfilled**. Omitted comparisons remain documented scope limitations or future work; they are not automatically reopened.

**Support-interpretation clarification:** read the [Phase 3C support addendum](benchmark/PHASE3C_SUPPORT_ADDENDUM.md) in the Phase 5 worktree/current branch. The historical 130 in-catalog direct answers without retrieved evidence include 49 correct reference matches; they are not 130 biological hallucinations. Grounded zero out-of-evidence IDs measures candidate-evidence compliance under different support opportunities, not zero biological errors. A bootstrap of those differently defined indicators does not establish reduced biological hallucination. The independently measured grounded 102/163 versus fusion 93/163 accuracy result and all frozen reports/definitions remain unchanged.

#### Completed single-reviewer audit

The reviewer completed 60 stratified cases in each of two sequential passes:

- Pass 1: blinded assessment of source labels, without system outputs.
- Pass 2: evidence-informed assessment of system predictions and abstentions, retaining the original Pass 1 judgments.

The revised exports validated with the expected frozen IDs, digests, required fields, and identical embedded Pass 1 judgments. Two assessment inconsistencies were clarified; no other answers changed. Original exports and previous analyses were preserved.

**Single-reviewer decision:** independent second review and inter-reviewer adjudication are deferred. This explicitly supersedes the original two-reviewer plan for the present scoped audit. The two passes are not independent reviews. Do not claim inter-reviewer agreement or independently adjudicated labels.

The final analysis is recorded locally under:

`benchmark/phase3/error_audit/review_work/analysis_20261105_single_reviewer_final/`

Consult `AUDIT_SCOPE.md`, `FINAL_ASSESSMENT.md`, and `PHASE3C_CHECKLIST.md` for the decision and evidence. The reported 22 flagged cases remain findings or limitations, rather than automatic completion blockers; an insufficient-information case remains unresolved.

Audit counts describe a stratified sample. Do not generalize unweighted sample rates to the full validation set or benchmark. Do not convert reviewer judgments into corrected reference labels or corrected benchmark accuracy automatically. Disclose single-reviewer limitations, potential Pass 2 anchoring, and reviewer involvement in development where applicable.

Personal review exports and private case-level analyses remain ignored by Git. Frozen labels, predictions, and original benchmark accuracy remain unchanged.

## Phase 4: Routed hybrid system, uncertainty, and abstention — bounded study complete

The verified cached study used the same 163 validation reactions and fixed agreement/margin gates. The broadest gate retained 102 exact correct and 124 answered cases while reducing simulated grounded calls from 163 to 142 (12.9%), below the prespecified 20% minimum. No routed candidate was retained. The fixed confidence filter lost one correct answer and saved no calls. The selected method remains BM25 + trained epoch-1 bi-encoder RRF Top-10 followed by grounded LLM selection or abstention for every reaction. These are cached replay findings, not a new end-to-end execution or held-out result.

The protocol, comparisons, plots, uncertainty, source checksums and method-selection rationale are under `benchmark/phase4/` in the completed Phase 4 worktree and the Phase 5 checksum-verified handoff. The original development objectives below remain for historical scope; their presence does not imply every proposed subsystem was implemented. No calibration fit or broader routing claim is established.

### Objective and protocol

Determine whether a simple routing policy can improve the accuracy–coverage–resource tradeoff over the frozen Phase 3C method. Write the protocol before running experiments, including hypotheses, eligible data, candidate policies, confidence signals, threshold selection, baselines, metrics, resource limits, and stopping rules.

Start with the smallest useful, interpretable routing study. Potential routes are:

- Inexpensive deterministic or learned selection for high-confidence inputs.
- Grounded LLM selection for ambiguous inputs.
- Full-catalog retrieval when rule-based evidence is inadequate.
- Abstention when available evidence is insufficient.

Only include routes supported by the implementation and available evidence. Routing may use prediction-time features; it must never use ground-truth membership, known correctness, audit flags, or labels unavailable at deployment.

### Data and development restrictions

- Keep the held-out test sealed, including test-level diagnostics.
- Fit models/calibration using training data and select methods using validation data.
- Disclose that the validation pilot and human audit have already influenced development.
- Do not use the human-reviewed sample as corrected training labels or tune specifically to its judgments.
- Reuse cached outputs where scientifically valid. Label cached policy simulations distinctly from newly executed end-to-end evaluations.
- Missing route outputs cannot be replaced by fabricated counterfactual predictions.
- Begin with local implementation, tests, cached analysis, and bounded local experiments. Paid API calls or substantial new training/inference require an explicit run scope and budget decision; prepare runners in dry-run mode first.

### Baselines and outcomes

Compare the strongest existing retriever's Top-1 output, frozen Phase 3C grounded selection, a simple confidence-threshold policy, and the proposed routing policy on aligned eligible inputs.

Evaluate where supported:

- Overall accuracy, explicitly accounting for abstentions.
- Coverage, selective accuracy, and risk–coverage curves.
- Area under the risk–coverage curve, with its definition.
- Correctness detection and reference-answer-absence detection.
- Calibration when data and sample size support it.
- Performance by retrieval stratum and model cluster.
- LLM-call frequency, token usage, latency, and cost where measured.

Keep reference-based mechanical evaluation separate from biological human judgments. Raw retrieval scores, margins, and LLM confidence are not calibrated probabilities by default.

Controlled answer-absent cases may be constructed by removing **all** accepted reference IDs from candidate evidence. Mark them synthetic and report them separately. Absence of a reference answer is not proof that no biologically defensible candidate exists.

Use cluster-aware uncertainty estimates where feasible. Limit threshold search and report validation-selection optimism. A negative result is acceptable: retain the stronger prior method if Phase 4 does not improve the stated tradeoff.

### Completion criteria

Phase 4 is complete when the scoped implementation, reproducible validation comparisons, uncertainty/resource analysis, limitations, and method-selection rationale are documented and verified. A more complex router is not required merely to claim a new component.

### Final method and evaluation freeze — after Phase 5 development

Phase 4 validation-based selection is complete. Freeze the selected method only after the required Phase 5 robustness comparison is resolved or explicitly rescoped, and operational evaluation gates are complete. Preserve Phase 3C as an unchanged comparison baseline throughout development. The updated package remains explicitly draft under `benchmark/phase5/freeze/` in the Phase 5 worktree; the earlier Phase 4 draft remains historical and unchanged.

The freeze package must specify:

- Exact selected methods and baselines.
- Model/checkpoint identities, prompts, catalog/retrieval settings, routing rules, and thresholds.
- Code revision, configurations, seeds, environments, and artifact digests.
- Preprocessing, leakage protections, and split definitions.
- Primary/secondary metrics and cluster-aware comparison/uncertainty procedures.
- Failure, retry, caching, abstention, and exclusion rules.
- Hard budget controls and cost/token accounting.
- The held-out command and dry-run checks.
- A prohibition on test-driven method changes and selective reruns.

If required outputs or decisions are missing, label the package a **draft freeze** and identify the blockers. Do not claim a final freeze prematurely.

### Later held-out evaluation

Execute the planned held-out evaluation only after the final freeze and separate authorization for the concrete run and any cost. Apply the frozen protocol without tuning on test outcomes. Document operational failures and deviations transparently; rerun only under predeclared failure/retry rules.

## Phase 5: Robustness to degraded biological information — active; retrieval study complete, end-to-end comparison blocked

The prespecified bounded protocol (`benchmark/PHASE5.md` in the separate Phase 5 worktree) evaluates 163 validation reactions, not all 969. Nineteen aligned conditions (3,097 query conditions) cover clean inputs, nominal 10/25/50/75% identifier-bundle removal with three seeds, each reaction side, one-hop frozen ChEBI ancestors, and three locally constrained synthetic sibling-corruption seeds. Names, equation text and stoichiometry remain present; realized doses, masking, unchanged/ineligible cases and query-local shared-species handling are reported explicitly.

Strict rule-based matching, BM25, the fixed selected encoder and the selected fusion were actually recomputed on degraded inputs. The strict-rule control is not presented as the historical iterative ontology-relaxation pipeline. Paired cluster/replicate intervals, Recall@1/3/5/10/100, truncated MRR, candidate counts/emptiness, stratum/cluster results and plots are under `benchmark/phase5/validation/`. The trained encoder is descriptively more stable than BM25 under identifier removal, with names retained; uncertainty versus fusion and limited development-sample generalizability remain explicit. No retraining or end-to-end method change is justified from these retrieval-only results.

**Not fully complete:** the required grounded comparison is matched clean versus nominal 50% removal, seed 20261106, on the same 163 cases. Review of ea241ea found that the original builder changes all historical clean prompts through visible provenance; P3P0096 also has component-rank/score drift. Its one-arm plan is preserved as superseded. The corrected `phase5-grounded-no-provenance-v2` keeps accurate lineage in sidecars and serializes both arms identically with the same provider settings/schema. It requires 163 new clean plus 163 degraded case/arm outputs (326 attempts under the existing experimental-unit cache policy; 317 distinct bodies), not historical clean responses as sole matched controls. The proposed USD 30 maximum is unapproved, and execution remains disabled. Interleave pairs and disclose provider/temporal drift; even identical bodies and returned aliases do not guarantee unchanged hosted behavior. Retrieval results, architecture and the validated historical baseline remain unchanged; the proposed v2 prompt is not yet validated or finally frozen. Optional broader comparisons remain deferred, the completed single-reviewer audit is not reopened, and the test stays sealed.

The original proposal below is retained as historical scope, not a declaration that every comparison has been run:

Systematically test the dependence on metabolite annotations:

- Randomly remove 10%, 25%, 50%, and 75% of species annotations.
- Remove annotations from one side of a reaction.
- Replace specific ChEBI terms with ontology ancestors.
- Introduce plausible but incorrect metabolite mappings.
- Compare rule-based retrieval, ontology relaxation, learned retrieval, direct LLM recovery, and tool-assisted recovery.

## Phase 6: Package the research artifact

After the experimental design stabilizes:

- Move remaining benchmark logic out of ad hoc notebooks.
- Add fixed configurations, seeds, environment documentation, and CI.
- Cache candidate sets and LLM responses without committing secrets.
- Generate manuscript figures and tables directly from frozen outputs.
- Publish a research-oriented README, reproducible CLI, and documented release.
- Write the preprint or technical report.

## Token-efficiency principle

Do not send the entire model reaction chain separately for every reaction. That repeats model context approximately quadratically as models grow.

Use compact reaction-local prompts and route expensive calls selectively:

- Include the target reaction and normalized participants.
- Add only a small, fixed amount of model or neighboring-reaction context when an ablation shows that it helps.
- Rerank only genuinely ambiguous candidate sets.
- Use open-set or tool-assisted calls only for sampled retrieval failures initially.
- Cache every response and track input/output tokens and cost.

This preserves the ability to measure an LLM's open-set biological knowledge without making the complete evaluation unnecessarily expensive.

## Historical manuscript scope — September plan, status to reconcile

The following priorities preserve the earlier plan for traceability. They are not evidence of current completion or current deadlines. The first manuscript needs a coherent minimum story; reconcile these items with repository reports and collaborator ownership while Phase 4 proceeds.

### Must-have before or during early October

- Verified species→reaction combined workflow with real end-to-end tests.
- Reviewed and corrected public documentation.
- Frozen Phase 1 and Phase 2 benchmark results.
- EM timing/accuracy ablation and a documented default decision.
- At least one fair species-retrieval comparison including BM25.
- OpenAI Phase 3 validation pilot, if funding and the live runner are ready.
- A fixed protocol and initial outputs for the uncurated-model case studies.
- Manuscript outline, figure/table list, methods skeleton, and ownership assignments.

### Valuable but allowed to continue during writing

- Cross-provider LLM replication.
- Learned full-catalog bi-encoder.
- Tool-assisted recovery.
- Full held-out test evaluation after all method choices are frozen.
- Expanded robustness experiments.

### Explicitly out of scope unless early results demand it

- Rebuilding every subsystem simultaneously.
- Running all model/context/tool combinations over all 5,816 reactions.
- A full-corpus EM run before the subset ablation demonstrates value.
- Treating uncurated models as quantitative ground truth.

## Historical schedule — superseded

Assuming focused work begins now and LunaStarr can review asynchronously:

| Window | Deliverable |
|---|---|
| Sep 3–9 | Reconcile branches; review commit `1611464`; build and run combined-workflow integration tests; identify documentation corrections |
| Sep 10–16 | Run EM subset ablation; choose default; implement BM25 and RRF behind a common retriever interface |
| Sep 17–23 | Compare direct/BM25/dense/RRF retrieval; run the capped OpenAI operational smoke test and validation pilot once credits arrive |
| Sep 24–30 | Run initial uncurated-model case studies; finalize the KEGG bi-encoder protocol; freeze September results; generate core tables/figures; agree on manuscript outline |
| October | Begin writing immediately; train/evaluate the KEGG bi-encoder and run selected final evaluations in parallel where they strengthen the agreed paper story |

This is feasible for an October **writing start**, not necessarily an October submission. A realistic first complete draft is late October to November if collaboration and compute are available. A polished submission is more plausibly November to December. The schedule should be revisited after the integration tests and EM subset ablation, because those are the largest near-term uncertainty reducers.

## Immediate next actions — current

1. Preserve the completed Phase 3C baseline and single-reviewer audit; keep private review exports ignored.
2. Preserve the verified Phase 4 worktree, protocol and negative routing result; retain the selected grounded-fusion method.
3. Use the separate Phase 5 worktree and completed local retrieval robustness artifacts; keep fixed catalog/weights and unchanged reference labels.
4. Resolve the corrected matched v2 clean/degraded validation plan (163 pairs, 326 unit-level calls, proposed USD 30 cap) and its explicit run/budget decision; historical clean responses cannot be the sole matched controls for this changed prompt.
5. Complete Phase 5 analysis after the required new outputs, or document an explicit later scope decision; do not mark retrieval-only work as full end-to-end robustness.
6. Finalize the method/evaluation freeze after Phase 5 development; keep the held-out test sealed until the final freeze and separate run authorization.
7. Reconcile the status and ownership of integration tests, EM ablation, species-side retrieval, external case studies, and packaging without assuming they were completed by the reaction study.
8. Continue manuscript methods, figure/table planning, and scope reconciliation in parallel. Phase 5 has been explicitly added before the final freeze; broader extensions remain optional.

## Revision record

- 2026-09-03: Original roadmap status date; historical experiments and calendar targets retained where useful.
- 2026-11-05: Updated implemented Phase 3 progress; accepted scoped Phase 3C completion; documented the single-reviewer audit decision; selected Phase 4 as next work; placed final method/evaluation freeze after Phase 4 and before the later held-out run.
- 2026-11-06: Verified bounded Phase 4 completion and retained grounded-fusion selection; completed the local Phase 5 retrieval grid, documented the blocked primary grounded robustness run, and placed Phase 5 development before the final freeze and later authorized held-out evaluation.
- 2026-11-06 review correction: preserved valid retrieval results, superseded the confounded one-arm request plan with matched v2 clean/degraded controls and sidecar provenance, and added the Phase 3C support-interpretation clarification without rewriting frozen metrics or reports.
