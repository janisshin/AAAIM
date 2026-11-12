# Automatic model-annotation agent

Bounded annotation agent that inspects an SBML model, builds a structured plan,
calls allowlisted AAAIM tools, and returns an annotated copy plus unresolved
cases, reservations, and errors.

This module does **not** replace the explicit APIs (`annotate_model`,
`assess_species_identity_support`, `predict_species_qualifiers`,
`write_reaction_annotations`, `update_annotation`). It orchestrates them.

## Entry point

Prefer dual roles: **planner** (orchestration) vs **inference** (biological
annotation). Legacy `backend=` alone still works for offline MockBackend tests
(same object for both roles). Do not combine `--backend` with
`--planner`/`--inference`.

```python
from core.agent import auto_annotate_model, ExecutionLimits
from core.agent.backends.mock import MockBackend
from core.agent.backends.openai_planner import OpenAIPlannerBackend
from core.agent.backends.metabolm import MetaboLLMBackend

# Offline control-loop (both roles = mock)
result = auto_annotate_model(
    model_file="model.xml",
    request="Annotate missing entities in this model",
    backend=MockBackend(),
    resources=None,                 # or RetrievalResources(...) for real RRF
    reaction_retriever=None,        # injectable for tests
    output_dir="_auto_annotate_out",
    limits=ExecutionLimits(),       # planner_spend_* default to zero paid calls
)
print(result.report.status, result.report.annotated_model_path)

# Split roles (OpenAI orchestration + local MetaboLLM biology)
# result = auto_annotate_model(
#     "model.xml",
#     planner=OpenAIPlannerBackend(),
#     inference=MetaboLLMBackend(hf_cache=...),
#     resources=RetrievalResources(...),
#     limits=ExecutionLimits(
#         planner_spend_authorized=True,   # explicit; default False
#         planner_spend_cap_usd=0.05,      # hard cap; default 0.0
#         authorized_annotation_models=["metabolm-local"],
#     ),
# )
```

### Synthetic demo (mock; no model cache)

```powershell
python -m core.agent.cli model.xml --out _auto_annotate_out --backend mock
python examples/auto_annotate_example.py --out _auto_annotate_demo_mock
```

### Real local demo (MetaboLLM planner + inference; catalog + encoder)

```powershell
# Prefer the existing MetaboLLM env (parent AAAIM checkout when using the nested worktree)
$repo = (Resolve-Path ..\..\..).Path   # adjust if already at AAAIM root
$py = Join-Path $repo "benchmark\phase3\_metabolm_env\Scripts\python.exe"
$hf = Join-Path $repo "benchmark\phase3\metabolm_grounded_comparison\_hf_cache"
$enc = Join-Path $repo "benchmark\integration_worktrees\rrf_grounded_api\_rrf_grounded_work\encoder"

& $py -m core.agent.cli model.xml --out _auto_annotate_out --backend metabolm `
  --hf-cache $hf `
  --catalog data/kegg/kegg_reaction_features.lzma `
  --encoder $enc --device cpu

& $py examples/auto_annotate_real_local_example.py --out _auto_annotate_demo_real_local `
  --hf-cache $hf --catalog data/kegg/kegg_reaction_features.lzma --encoder $enc
```

### Dual-role CLI (OpenAI planner + MetaboLLM inference)

Paid OpenAI planner dispatches default to **off** (`--planner-spend-cap-usd 0`
and no `--planner-spend-authorized`). Do not combine with `--backend`.

```powershell
& $py -m core.agent.cli model.xml --out _auto_annotate_out `
  --planner openai --inference metabolm `
  --hf-cache $hf --catalog data/kegg/kegg_reaction_features.lzma `
  --encoder $enc --device cpu `
  --planner-spend-authorized --planner-spend-cap-usd 0.05
```

## Control loop

**Inspect → planning turn → execute bounded action group → validate →
(continue | finish | needs_clarification) → export after accepted finish.**

Planner decisions:

- `continue` — next action group (`max_actions_per_turn`). Exhausting a group
  requests another planning turn; it does **not** auto-complete the run.
- `finish` — validated against the requested scope and task/entity ledger;
  every in-scope target must be completed, unresolved, failed, or intentionally
  skipped. Premature finish is rejected.
- `needs_clarification` — stop partial with an explicit prompt.

Ordinary `max_planning_turns` are separate from failure-driven
`max_failure_revisions`. Malformed actions never execute. Real/metabolm mode
defaults `allow_structural_fallback=False`; if explicitly enabled, provenance
records `execution_mode=fallback_workflow_orchestration` (not LLM planning
success).

## Schemas and semantic checks

Plans, actions, per-tool arguments/results, observations, and reports are
Pydantic models in `core/agent/schemas.py` (`extra="forbid"` where strict).
Grounded reaction answers reuse `core.reaction.grounded.GroundedAnnotation`.

Validation is layered:

1. **Schema validation** — structural well-formedness via Pydantic.
2. **Semantic checks** (`core/agent/validate.py`) — target existence in the
   model inventory, tool/representation compatibility, acyclic/valid
   dependencies, and evidence consistency (including
   `validate_evidence` for RRF Top-10 records and status/annotation alignment).

`SemanticCheckReport.biological_correctness_certified` is always `False`.
Passing schema or semantic checks never certifies biological identity.

## Supported representations

| Representation | Species annotation | Reaction annotation | Notes |
|---|---|---|---|
| SBML species (chemical/gene/protein/complex) | Yes | n/a | Type guessed from names/annotations; ambiguous types unresolved |
| FBC gene products | Yes (via `update_annotation`) | n/a | Supported for identity writing |
| SBML-qual qualitative species | Yes (via `update_annotation`) | n/a | Supported for identity writing |
| SBML reactions | n/a | Yes (`rrf_grounded` only) | No rule-based reaction matching in this module |
| SBML-qual transitions | n/a | **Unsupported** | Reported explicitly |

Default scope is **add missing** annotations while preserving existing ones.
Replacement requires `scope="replace_existing"` (curator policy).

## Local MetaboLLM backend

- Baseline only: Hub `MetaboLLM/MetaboLLM-Qwen3-4B` @ `13deefb…` on
  `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75…`
- Experimental SFT checkpoints are **not** used
- 4-bit NF4, unmerged adapter, greedy decoding; configurable
  `planner_max_new_tokens` (default 768; **512 is not a mandatory ceiling**),
  justified by measured prompt/output fit; `prefill_chunk_size=512`
- Compact inventory + bounded action-group prompts (`agent-metabolm-planner-v2-compact`)
- Prompted JSON + Pydantic validation (no grammar-constrained decoding in-repo)
- Truncation/malformed plans refused; raw text + usage recorded in `planner_meta`
- No hosted-provider fallback; no silent structural fallback in real mode
- No process-wide `HF_HOME` mutation (explicit `cache_dir` / local paths only)
- Serial GPU use on RTX 3070 8 GB: bi-encoder defaults to CPU; unload MetaboLLM
  between heavy stages

### HF cache configuration precedence

1. Explicit API `hf_cache=` / CLI `--hf-cache` (**authoritative**; invalid → fail, no fallback)
2. Environment variable `AAAIM_METABOLM_HF_CACHE` (**authoritative**; invalid → fail, no fallback)
3. Discovered documented default: walk ancestors for
   `benchmark/phase3/metabolm_grounded_comparison/_hf_cache` (ordinary checkouts
   and nested worktrees; no fixed folder depth). If no AAAIM-like root and no
   existing cache are found, fail and require explicit/env configuration.

Validation checks pinned snapshot **files** (weight shards, configs, broken
symlinks) and records stable content fingerprints. Resume may relocate the
cache directory only when fingerprints match; changed bytes invalidate
compatibility even if revision strings are unchanged. Absolute paths are audit
fields (`backend_resource_paths`); transient `load_info` (e.g. free VRAM) is
excluded from resume identity. Catalog (`--catalog`) and encoder (`--encoder`)
remain separately configurable.

Requires the working env under `benchmark/phase3/_metabolm_env` for real runs.

## Outputs

Under `output_dir`:

- `annotated_model.xml` (only if writing succeeds; never overwrites the source)
- `annotation_changes.csv`
- `initial_plan.json`, `executed_plan.json`
- `recommendations.json` (+ species/reaction CSVs)
- `provenance.json`, `inventory.json`, `observations.jsonl`
- `checkpoint.json` (resume)
- `report.json`

Partial results are labeled `status="partial"` or `"interrupted"` and must not
be described as fully annotated.

## Resume

```python
result = auto_annotate_model(
    "model.xml",
    backend=backend,
    output_dir="_auto_annotate_out",
    resume_from="_auto_annotate_out",
)
```

```powershell
# Inspect reports from a prior run, then resume when compatible
Get-Content _auto_annotate_out\report.json
Get-Content _auto_annotate_out\provenance.json
& $py -m core.agent.cli model.xml --out _auto_annotate_out --backend metabolm `
  --hf-cache $hf --catalog data/kegg/kegg_reaction_features.lzma --encoder $enc `
  --resume-from _auto_annotate_out
```

`checkpoint.json` uses **schema_version=3** (`AgentCheckpoint` in
`core/agent/schemas.py`) and is written atomically (temp file + replace).
v3 requires dual `planner` / `inference` identities plus a durable
`planner_budget` ledger. Pre-v3 checkpoints cannot be resumed.

Resume restores the saved plan, task/entity states, observations, completed
action fingerprints, attempt/revision counts, species and reaction
recommendations, reaction retrieval evidence, provenance, planner-only spend
accounting, and execution-limit accounting, then **continues that plan**
instead of generating an unrelated fresh plan.

Compatibility is checked against source model SHA-256, composite
planner/inference identity (including MetaboLLM weight **and
tokenizer/chat-template** content fingerprints when used for inference),
settings/limits/request, tool schema version, **content-based** resource
identities, and the summary `cache_identity`.

Retrieval resource identity (see `core/agent/resource_identity.py`) records
catalog file SHA-256, selected-encoder manifest/file digests via
`verify_model_dir`, and output-affecting settings (`device`, dtype policy,
`batch_size`, retrieval `CONFIG`). Absolute catalog/encoder/HF-cache paths are
audit-only (`resource_paths` / `backend_resource_paths`) and do not participate
in the resume hash, so relocating identical bytes with the same settings is
allowed. Changed catalog/encoder bytes or device/batch/config settings reject
resume before further annotation. Injected retrievers must expose a complete
provenance identity (or wrap `RetrievalResources` / `ReactionRetriever`);
class-name equality alone is rejected.

MetaboLLM offline cache validation fingerprints adapter tokenizer layouts
(`tokenizer_config.json` plus `tokenizer.json` or alternate vocab/model files,
special-token maps, `chat_template.jinja` / `chat_template.json` /
`additional_chat_templates/`). Tokenizer or chat-template edits invalidate
resume even when revision strings and weight shards are unchanged.

Evidence dependencies are validated (completed retrieval/selection requires
persisted `reaction_evidence`, and completed selection requires
`reaction_recommendations`).

Unsupported or incompatible checkpoints are rejected with
`CheckpointCompatibilityError` (missing file, corrupt JSON, missing /
unsupported `schema_version`, incomplete fields, or envelope mismatch).
**Pre-v3 checkpoints cannot be resumed** (v2 lacked dual planner/inference
identities and planner budget ledger; v1 also lacked content-based
resource/tokenizer identity); restart without `resume_from`.

## Role separation and planner budget

- **Planner** (`OpenAIPlannerBackend`, model `gpt-6-luna`): Responses API +
  **strict** Structured Outputs (`strict: true`). The submitted schema is
  generated from typed `TOOL_ARG_MODELS` as per-tool `anyOf` variants: every
  object has `additionalProperties: false`, every property is `required`, and
  free-form `arguments` bags are not used. `max_retries=0`; refuses
  `generate()` / `grounded_transport()` biological paths. Schema version
  `agent-planner-decision-v2`.
- **Inference** (local MetaboLLM / MockBackend): species annotation and
  grounded reaction selection. `select_reaction_grounded.llm_model` must be in
  `ExecutionLimits.authorized_annotation_models` (default `metabolm-local`);
  unauthorized ids are rejected without rewrite.
- **Budget** (planner-only, **ledger schema v2**): default
  `planner_spend_authorized=False` and `planner_spend_cap_usd=0.0` (zero paid
  dispatches). `planner_budget.json` is the **authoritative** attempt store
  (single writer via atomic replace); checkpoints embed a summary for
  cross-check. Before each OpenAI call the ledger computes a hard UTF-8-byte
  input-token bound, reserves conservative Standard pricing (`$0.10` /
  `$0.50` per 1M in/out for gpt-6-luna, checked 2026-11-12), **persists the
  reservation**, journals a controller checkpoint, then dispatches.
  Settlement requires **verified response identity** before releasing any
  reservation: raw Responses API `model` and `service_tier` must be present
  (never filled from request defaults). Compatibility policy (docs checked
  2026-11-12): reserved Standard-rate planner expects request model
  `gpt-6-luna` and request tier `default` (OpenAI’s name for Standard
  pricing); accepted response identities are exactly `gpt-6-luna` +
  `default`. Documented Fast-mode request `fast`/`priority` → response
  `priority`, plus `flex` / `ultrafast` / `scale` / `auto`, are **not**
  accepted for Standard-rate settlement. Validated token usage alone cannot
  settle an identity-uncertain attempt. Usage must also be strict
  nonnegative integers within reserved bounds; token-derived USD is an
  **estimate**, not invoice verification. Missing/blank/unknown/incompatible
  identity or usage outside bounds → retain conservative exposure, record
  requested vs actual identity metadata (no credentials), record any
  `potential_additional_exposure_usd` when reported liability exceeds the
  reservation, set `accounting_status=blocked`, and refuse further paid
  dispatches. **Post-response validation cannot undo charges already
  incurred.** Uncertain/failed attempts retain exposure against the cap.
  Biological/local MetaboLLM work is not billed here.
- **Resume:** do not overwrite an existing ledger during init. Reconcile
  `planner_budget.json` with checkpoint `planner_budget` by `attempt_id`
  (prefer newer terminal status; fail closed on conflicting charges/bounds).
  Open `reserved` attempts become retained exposure. Pre-v2 ledgers and
  unbound foreign ledgers in an existing `output_dir` are rejected (use
  `resume_from` or a fresh directory). Resume applies current spend
  authorization/limits without erasing prior charged/exposure.

### Live integrated demo proposal (not executed)

**Goal:** one mixed-task SBML run with real OpenAI planner + real local
MetaboLLM inference (species + RRF retrieve/select), no held-out access.

**Preconditions (all required before any paid call):**

1. Explicit spend authorization and hard USD cap (CLI flags / `ExecutionLimits`).
2. Verified `OPENAI_API_KEY` and offline MetaboLLM HF cache + catalog + encoder.
3. Mocked component tests green; no SDK retries; budget ledger persist path writable.
4. Separate operator approval of the numeric cap for that session.

**Not authorized by this document.** Defaults remain zero paid dispatches.

Execution is **at-least-once**, not exactly-once: an `in_flight_action` is
journaled before tool execution; if interruption occurs before the result is
durably recorded, resume may re-execute that action. Successful completed
fingerprints are not repeated.

## Tests

```bash
python -m pytest tests/agent -q --basetemp=_pytest_agent
python -m pytest tests/test_public_rrf_grounded.py -q
```

Controller tests use `MockBackend` only. Bounded local MetaboLLM demos are
authorized separately and are not a biological accuracy claim.

On Windows, MetaboLLM HF-cache fixture paths can hit legacy `MAX_PATH` limits
when nested under long pytest basetemp directories; use a short `--basetemp`
(for example `C:\t\aaaim`) for those tests. That is an environment/path
limitation, not a product resume failure.

## Species annotation path

- **MockBackend / tests:** explicit `mock_species_annotator` may emit synthetic
  IDs (`CHEBI:15377`, etc.). Provenance records
  `species_annotator.kind = "mock_synthetic"`.
- **Real backends (MetaboLLM):** `make_real_species_annotator` wraps
  `annotate_single_model` with `method="direct"` (local ChEBI/NCBIGene/UniProt
  pickles), local LLM hooks for normalization/ranking/identity/qualifiers, and
  **no** hosted-provider fallback. Missing DB files or LLM failures are
  **technical failures** for that task; other tasks may continue.
  ChromaDB / `grpc._cython.cygrpc` are **not** required for this direct path;
  `core.database_search` loads them lazily only for RAG (`method="rag"`).
- There is **no** implicit synthetic-ID default for real backends. A missing
  species annotator fails closed.
- `update_annotation` still refuses URIs for `identity_status=unresolved` and
  `qualifier_status=abstain`. Schema-valid IDs alone never imply `supported`.

## Real vs synthetic demos

| Example | Backend | Retrieval | Species | Claims |
|---------|---------|-----------|---------|--------|
| `examples/auto_annotate_example.py` | MockBackend | `SyntheticTinyRetriever` (not from `tests/`) | `mock_synthetic` | Control-loop only |
| `examples/auto_annotate_real_local_example.py` | MetaboLLM (both roles) | verified catalog + selected encoder (CPU) | `make_real_species_annotator` | Real LLM tool choice + retrieval; **not** biological accuracy |
| CLI `--planner openai --inference metabolm` | OpenAI planner + MetaboLLM inference | same as real local | real species path | Integrated demo **proposed only** until spend authorized |

Measured planner fit on the bounded real local demo (development model): ~749 input /
169 output tokens with `planner_max_new_tokens=768` (not truncated).

## Known limitations

- Real species annotation requires local reference databases under `data/` and a
  local LLM backend with `generate()` (MetaboLLM baseline).
- Planning uses prompted JSON; grammar-constrained decoding is unavailable.
  Compact prompts + argument sanitization help, but invalid LLM plans are still
  possible and are rejected (or labeled fallback if explicitly enabled).
- Biological accuracy of annotations is out of scope for the operational demo.
- Phase 5 paid runs and held-out evaluation remain unauthorized.
