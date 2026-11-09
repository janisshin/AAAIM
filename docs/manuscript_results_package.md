# Reaction manuscript results package

The scripts `benchmark/scripts/manuscript_{results,propagation,metrics,plans,handoff}.py`
produce private manuscript artifacts from preserved research worktrees. They do
not call a provider, train a model, or read held-out queries or labels. Current
outputs are ignored under `_manuscript_results_work/` in the isolated
`manuscript_results` worktree. Source artifacts and original exports remain intact.

The assembled report is
`_manuscript_results_work/reaction_package_20261109_v2/README.md`.
It includes Results text, CSV/JSON/Markdown tables, PNG/SVG/PDF figures, audit
stratum denominators and private examples, provenance/checksums, metric crosswalk,
execution proposal, completion checklist, and held-out readiness. Its
`REPRODUCE.md` gives commands for fresh ignored destinations. The audit is one
reviewer in two stages; case judgments and exports must stay ignored.

`propagation_v3_aliased` is the current protocol/cohort/query/ranking freeze.
It applies the public identity/qualifier contract, complete tables to suppress
SBML fallback, and the selected frozen BM25 + trained epoch-1 Top-100 RRF
(`k=60`, grounded Top-10). Uniform nonsemantic species-ID aliases preserve names
and structure across all conditions while removing identifier-looking source
IDs. Earlier source-ID controls remain preserved under `propagation_v2` and
are superseded for the manuscript's primary names-only comparison. Only a
verified unchanged catalog embedding matrix is reused; all new query texts
are encoded and ranked. Reference annotations are an available baseline, not
independently verified biological truth.

The target is 969 validation reactions / 18 models / 12 clusters. Existing
case-study prediction exports match none of these models. Reference/names-only
controls are executed; actual predicted-species retrieval and all final grounded
propagation outcomes are pending. `propagation_execution_v4_aliased` preserves
exact ready A/C request bodies and hashes, pending B units, per-arm persistent
runner plans, and a separately costed missing-species normalization frontier.
Later species stages depend on actual responses and are not falsely frozen or
priced as complete. These plans authorize zero spending. Do not reuse historical
GPT-5.6 responses as GPT-6.1 results.

Report reaction-micro and model-macro metrics separately, with all denominators.
Bootstrap whole frozen clusters, preserving pairing. Top-K reference recovery is
retrieval performance, not asserted annotation precision. Biological abstentions,
technical failures, pending units, and incomplete responses have distinct states.
The stratified pilot, stratified audit, synthetic robustness, cached replay,
actual inference, and new offline controls remain separate populations.

The Phase 5 paid comparison stays paused; its historical numerical retrieval
results are preserved. Its global checksum ledger has stale later planning/doc
entries, explicitly recorded in the private verification report. MODEL2507280001
remains a distinct 541-reaction practical case study awaiting inference/review.
The held-out checklist is preparatory and does not declare the freeze final.

Run focused tests with `python -m pytest tests/test_manuscript_results.py -q`
and the existing reaction integration/qualifier compatibility tests. On Windows,
choose a fresh short `--basetemp` under the user's temporary directory if path
length limits affect persistent runner fixtures. Never overwrite a frozen output
directory. Resource paths and historical worktrees are explicit local inputs;
their absence is an error or an explicitly skipped optional runner test, not
permission to fabricate results.

## Reference sensitivity

The isolated `analysis/review-informed-reference-v1` branch adds
`benchmark/scripts/reference_sensitivity.py`. Its ignored output is
`_manuscript_results_work/reference_review_v1_release2/` in the
`reference_sensitivity` worktree. The original results package remains in the
sibling `manuscript_results` worktree, unchanged.

This is a single-reviewer, review-informed reference version, not a fully
corrected or independently adjudicated gold standard. Four explicit Pass 1
replacements are actionable; no additional IDs are inferred from prediction
assessments, taxonomy labels or inappropriate abstentions. Two contradictory
cross-pass reference proposals are held. Ambiguity/non-mappability remain flagged
without emptying reference sets or dropping cases. The 103 unreviewed pilot
cases retain unverified originals. No 969-case correction claim is made.

Start with `RESULTS.md`, `RETRIEVAL_TABLES.md`, `REVIEWED_SUBSET.md` and
`METHODS_AND_LIMITATIONS.md` in the private output. `REFERENCE_VERSION.json`
binds the policy, correction layer, derived 163-case view and original input
checksums. The reference version has a content digest; generation refuses an
existing destination. `MANIFEST.json` binds all output files.
`case_scoring_diff_private.jsonl` traces every method/case, while
`changed_outcomes_private.csv` contains the changed scores only.
`source_validation.json` records actual differences from the earlier export,
remaining discrepancies and authoritative export selection. Personal review
records, the quote-bound conflict holds, and the optional question packet remain
ignored by Git.

The original grounded/fusion metrics (102/163 and 93/163) remain frozen results.
The new view yields 104/163 and 95/163 on identical predictions, with unchanged
coverage and a nine-case paired difference. Tables separate full-pilot and
stratified reviewed-subset results; fixed sampling strata are not reassigned
after rescoring. Historical GPT-5.6 inference remains distinct from pending
GPT-6.1 work. Sensitivity does not independently establish better biological
accuracy, finish propagation, unpause Phase 5 or finalize held-out evaluation.

Run `python -m pytest tests/test_reference_sensitivity.py -q` for synthetic
policy/denominator/alignment tests. `REPRODUCE.md` gives the exact no-call command
and fresh-output policy; the same command plus `--verify-only` checks output
and source checksums without rescoring. Two fresh builds must match every
manifest entry. Existing frozen artifact verifiers are also run during a build.
