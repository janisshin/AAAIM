# Automatic model-annotation agent

Bounded annotation agent that inspects an SBML model, builds a structured plan,
calls allowlisted AAAIM tools, and returns an annotated copy plus unresolved
cases, reservations, and errors.

This module does **not** replace the explicit APIs (`annotate_model`,
`assess_species_identity_support`, `predict_species_qualifiers`,
`write_reaction_annotations`, `update_annotation`). It orchestrates them.

## Entry point

```python
from core.agent import auto_annotate_model, ExecutionLimits
from core.agent.backends.mock import MockBackend

result = auto_annotate_model(
    model_file="model.xml",
    request="Annotate missing entities in this model",
    backend=MockBackend(),
    resources=None,                 # or RetrievalResources(...) for real RRF
    reaction_retriever=None,        # injectable for tests
    output_dir="_auto_annotate_out",
    limits=ExecutionLimits(),
)
print(result.report.status, result.report.annotated_model_path)
```

CLI:

```bash
python -m core.agent.cli model.xml --out _auto_annotate_out --backend mock
python examples/auto_annotate_example.py
python examples/auto_annotate_example.py --backend metabolm
```

## Control loop

**Inspect → plan → execute one allowlisted action → validate observation →
continue or revise → export.**

Deterministic code owns SBML parsing, inventories, dependency scheduling,
schema checks, checkpoints, and writing. The LLM chooses tools/configurations
and justified revisions. Malformed actions never execute.

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
- 4-bit NF4, unmerged adapter, greedy decoding, `max_new_tokens=512`,
  `prefill_chunk_size=512`
- Prompted JSON + Pydantic validation (no grammar-constrained decoding in-repo)
- No silent truncation; no hosted-provider fallback
- Serial GPU use on RTX 3070 8 GB: bi-encoder defaults to CPU; unload MetaboLLM
  between heavy stages

Requires the working env under `benchmark/phase3/_metabolm_env` and the HF cache
at `benchmark/phase3/metabolm_grounded_comparison/_hf_cache`.

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

Resume refuses incompatible cache identities (model hash + backend + settings).
Completed action fingerprints are not repeated.

## Tests

```bash
python -m pytest tests/agent -q --basetemp=_pytest_agent
python -m pytest tests/test_public_rrf_grounded.py -q
```

Controller tests use `MockBackend` only. Bounded local MetaboLLM demos are
authorized separately and are not a biological accuracy claim.

## Species annotation path

- **MockBackend / tests:** explicit `mock_species_annotator` may emit synthetic
  IDs (`CHEBI:15377`, etc.). Provenance records
  `species_annotator.kind = "mock_synthetic"`.
- **Real backends (MetaboLLM):** `make_real_species_annotator` wraps
  `annotate_single_model` with `method="direct"` (local ChEBI/NCBIGene/UniProt
  pickles), local LLM hooks for normalization/ranking/identity/qualifiers, and
  **no** hosted-provider fallback. Missing DB files or LLM failures are
  **technical failures** for that task; other tasks may continue.
- There is **no** implicit synthetic-ID default for real backends. A missing
  species annotator fails closed.
- `update_annotation` still refuses URIs for `identity_status=unresolved` and
  `qualifier_status=abstain`. Schema-valid IDs alone never imply `supported`.

## Known limitations

- Real species annotation requires local reference databases under `data/` and a
  local LLM backend with `generate()` (MetaboLLM baseline).
- Planning uses prompted JSON; grammar-constrained decoding is unavailable.
- Biological accuracy of annotations is out of scope for the operational demo.
- Phase 5 paid runs and held-out evaluation remain unauthorized.
