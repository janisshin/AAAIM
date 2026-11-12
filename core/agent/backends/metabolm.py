"""Local MetaboLLM baseline backend for planning and grounded selection.

Uses the installed Hub baseline adapter (not experimental SFT checkpoints).
No hosted-provider fallback. No silent truncation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from core.agent.schemas import (
    AnnotationPlan,
    ModelInventory,
    PlannerDecision,
    ToolObservation,
)
from core.reaction.grounded import GroundedAnnotation

ADAPTER_ID = "MetaboLLM/MetaboLLM-Qwen3-4B"
BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_REVISION = "13deefb4762ac259b35bd61a5701c7abd50b02b0"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PLANNER_SCHEMA_VERSION = "agent-planner-v2"
PROMPT_VERSION = "agent-metabolm-planner-v2-compact"

HF_CACHE_ENV = "AAAIM_METABOLM_HF_CACHE"
DEFAULT_HF_CACHE_REL = Path("benchmark") / "phase3" / "metabolm_grounded_comparison" / "_hf_cache"

REQUIRED_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    (ADAPTER_ID, ADAPTER_REVISION),
    (BASE_ID, BASE_REVISION),
)

QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
GENERATION = {
    "do_sample": False,
    # Default budget; overridable per call via planner_max_new_tokens. Not a hard ceiling of 512.
    "max_new_tokens": 768,
    "prefill_chunk_size": 512,
}

COMPACT_PLANNER_SCHEMA = {
    "type": "object",
    "required": ["decision", "actions", "reason"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["continue", "finish", "needs_clarification"],
        },
        "reason": {"type": "string"},
        "clarification_prompt": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action_id", "tool", "target_ids"],
                "properties": {
                    "action_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "task_ids": {"type": "array", "items": {"type": "string"}},
                    "arguments": {"type": "object"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "tasks": {"type": "array"},
    },
}

JSON_TRANSPORT_SUFFIX = (
    "\n\nLOCAL_TRANSPORT_JSON_SCHEMA\n"
    "Return exactly one JSON object matching the schema and no other text:\n"
)


class MetaboLLMCacheError(RuntimeError):
    """Local MetaboLLM HF cache missing, incomplete, or misconfigured."""


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _is_aaaim_root(candidate: Path) -> bool:
    return (candidate / "core" / "agent").is_dir() and (candidate / "benchmark").is_dir()


def _hub_repo_dir(hf_cache: Path, repo_id: str) -> Path:
    return hf_cache / "hub" / f"models--{repo_id.replace('/', '--')}"


def discover_default_hf_cache(*, start: Optional[Path] = None) -> Path:
    """Locate the documented cache without fixed-depth parents.

    Prefers an ancestor that already contains the documented ``_hf_cache``
    directory (so nested worktrees reuse the parent checkout cache). Otherwise
    returns the documented relative path under the nearest AAAIM-like root.
    """
    origin = (start or Path(__file__)).resolve()
    ancestors = [origin, *origin.parents]
    for candidate in ancestors:
        cache = candidate / DEFAULT_HF_CACHE_REL
        if cache.is_dir():
            return cache.resolve()
    for candidate in ancestors:
        if _is_aaaim_root(candidate):
            return (candidate / DEFAULT_HF_CACHE_REL).resolve()
    raise MetaboLLMCacheError(
        "Could not locate an AAAIM repository root or the documented MetaboLLM "
        f"HF cache ({DEFAULT_HF_CACHE_REL.as_posix()}). Pass hf_cache= / "
        f"--hf-cache or set {HF_CACHE_ENV} to a validated local cache. "
        "Refusing hosted download or a guessed home/cwd cache."
    )


def default_hf_cache() -> Path:
    """Documented default cache path via ancestor discovery (no fixed depth)."""
    return discover_default_hf_cache()


def resolve_hf_cache(
    hf_cache: Optional[str | Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    start: Optional[Path] = None,
) -> Path:
    """Resolve HF cache path.

    Precedence (authoritative; no silent fallback after a chosen source):
      1. explicit ``hf_cache`` argument / ``--hf-cache``
      2. dedicated env ``AAAIM_METABOLM_HF_CACHE``
      3. discovered documented default

    Generic ``HF_HOME`` is not consulted. Invalid explicit/env paths are not
    replaced by discovery; callers must validate the returned path.
    """
    env = environ if environ is not None else os.environ
    if hf_cache is not None and str(hf_cache).strip() != "":
        return _expand_path(hf_cache)
    env_value = (env.get(HF_CACHE_ENV) or "").strip()
    if env_value:
        return _expand_path(env_value)
    return discover_default_hf_cache(start=start)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_readable_file(path: Path, *, cache: Path, repo_id: str, revision: str) -> Path:
    label = f"{repo_id}@{revision}"
    if path.is_symlink():
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise MetaboLLMCacheError(
                f"Broken symlink for required snapshot file {path.name} under "
                f"{label} in HF cache ({cache}): {exc}. Refusing hosted-provider fallback."
            ) from exc
        if not resolved.is_file():
            raise MetaboLLMCacheError(
                f"Symlink {path} for {label} in HF cache ({cache}) does not "
                "resolve to a regular file. Refusing hosted-provider fallback."
            )
        return resolved
    if not path.is_file():
        raise MetaboLLMCacheError(
            f"Missing required snapshot file {path.name} for {label} in HF cache "
            f"({cache}). Refusing hosted-provider fallback."
        )
    return path


def resolve_snapshot_dir(hf_cache: Path, repo_id: str, revision: str) -> Path:
    """Resolve a local Hub snapshot directory for a pinned revision (offline)."""
    repo_dir = _hub_repo_dir(hf_cache, repo_id)
    snapshots = repo_dir / "snapshots"
    direct = snapshots / revision
    if direct.is_dir():
        return direct.resolve()
    ref = repo_dir / "refs" / revision
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        via_ref = snapshots / commit
        if via_ref.is_dir():
            return via_ref.resolve()
    raise MetaboLLMCacheError(
        f"Pinned snapshot {repo_id}@{revision} not found under local HF cache "
        f"({hf_cache}); checked {direct}. Refusing hosted-provider fallback."
    )


def _read_weight_index(path: Path, *, hf_cache: Path, repo_id: str, revision: str) -> list[str]:
    resolved = _require_readable_file(path, cache=hf_cache, repo_id=repo_id, revision=revision)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    shards = sorted(set((payload.get("weight_map") or {}).values()))
    if not shards:
        raise MetaboLLMCacheError(
            f"Weight index {path.name} for {repo_id}@{revision} in HF cache "
            f"({hf_cache}) has an empty weight_map. Refusing hosted-provider fallback."
        )
    return shards


def _weight_relative_paths(
    snapshot_dir: Path,
    *,
    hf_cache: Path,
    repo_id: str,
    revision: str,
) -> list[str]:
    adapter_index = snapshot_dir / "adapter_model.safetensors.index.json"
    model_index = snapshot_dir / "model.safetensors.index.json"
    if adapter_index.exists() or adapter_index.is_symlink():
        return _read_weight_index(
            adapter_index, hf_cache=hf_cache, repo_id=repo_id, revision=revision
        )
    if model_index.exists() or model_index.is_symlink():
        return _read_weight_index(
            model_index, hf_cache=hf_cache, repo_id=repo_id, revision=revision
        )
    for name in (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        candidate = snapshot_dir / name
        if candidate.exists() or candidate.is_symlink():
            return [name]
    return []


def _exists_in_snapshot(snapshot_dir: Path, rel: str) -> bool:
    path = snapshot_dir / rel
    return path.exists() or path.is_symlink()


def _tokenizer_vocab_layouts() -> tuple[tuple[str, ...], ...]:
    """Accepted alternative vocab/model layouts for AutoTokenizer."""
    return (
        ("tokenizer.json",),
        ("vocab.json", "merges.txt"),
        ("vocab.txt",),
        ("tokenizer.model",),
        ("spiece.model",),
    )


def discover_tokenizer_relative_paths(snapshot_dir: Path, *, required: bool) -> list[str]:
    """Discover tokenizer / chat-template files that affect AutoTokenizer loading.

    Supports common alternative layouts. When ``required`` is true (adapter
    snapshot, where ``AutoTokenizer.from_pretrained(adapter_dir)`` runs), a
    usable tokenizer layout must be present.
    """
    found: set[str] = set()
    config_name = "tokenizer_config.json"
    has_config = _exists_in_snapshot(snapshot_dir, config_name)
    if has_config:
        found.add(config_name)

    layout_hit: tuple[str, ...] | None = None
    for layout in _tokenizer_vocab_layouts():
        if all(_exists_in_snapshot(snapshot_dir, name) for name in layout):
            layout_hit = layout
            found.update(layout)
            break

    for optional in (
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
    ):
        if _exists_in_snapshot(snapshot_dir, optional):
            found.add(optional)

    additional = snapshot_dir / "additional_chat_templates"
    if additional.is_dir():
        for path in sorted(additional.rglob("*")):
            if path.is_file() or path.is_symlink():
                found.add(path.relative_to(snapshot_dir).as_posix())

    # tokenizer_config may reference a chat template file by relative name.
    if has_config:
        try:
            payload = json.loads((snapshot_dir / config_name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            for key in ("chat_template", "chat_template_file"):
                value = payload.get(key)
                if isinstance(value, str) and "/" not in value and "\\" not in value:
                    # Embedded template strings are already covered by hashing tokenizer_config.
                    # Only treat simple relative filenames as sidecar dependencies.
                    if value.endswith((".jinja", ".json", ".txt")) and _exists_in_snapshot(
                        snapshot_dir, value
                    ):
                        found.add(value)

    if required and (not has_config or layout_hit is None):
        raise MetaboLLMCacheError(
            f"Adapter snapshot {snapshot_dir} is missing tokenizer files required "
            "for offline AutoTokenizer / chat-template loading "
            "(need tokenizer_config.json plus one of tokenizer.json, "
            "vocab.json+merges.txt, vocab.txt, tokenizer.model, or spiece.model). "
            "Refusing hosted-provider fallback."
        )
    return sorted(found)


def required_snapshot_relative_paths(
    snapshot_dir: Path,
    *,
    hf_cache: Path,
    repo_id: str,
    revision: str,
) -> list[str]:
    """Ordered relative paths that must exist and contribute to the fingerprint."""
    weights = _weight_relative_paths(
        snapshot_dir, hf_cache=hf_cache, repo_id=repo_id, revision=revision
    )
    if not weights:
        raise MetaboLLMCacheError(
            f"No weight shards found in snapshot directory {snapshot_dir} for "
            f"{repo_id}@{revision} in HF cache ({hf_cache}); "
            "refusing hosted-provider fallback."
        )
    required = set(weights)
    if repo_id == ADAPTER_ID:
        required.add("adapter_config.json")
        if (snapshot_dir / "adapter_model.safetensors.index.json").exists() or (
            snapshot_dir / "adapter_model.safetensors.index.json"
        ).is_symlink():
            required.add("adapter_model.safetensors.index.json")
        # AutoTokenizer is loaded from the adapter snapshot.
        required.update(discover_tokenizer_relative_paths(snapshot_dir, required=True))
    else:
        required.add("config.json")
        if (snapshot_dir / "model.safetensors.index.json").exists() or (
            snapshot_dir / "model.safetensors.index.json"
        ).is_symlink():
            required.add("model.safetensors.index.json")
        # Base may carry unused tokenizer sidecars; fingerprint those that exist.
        required.update(discover_tokenizer_relative_paths(snapshot_dir, required=False))
    return sorted(required)


def fingerprint_snapshot(
    snapshot_dir: Path,
    *,
    hf_cache: Path,
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    """Validate required files and return a stable content fingerprint."""
    rel_paths = required_snapshot_relative_paths(
        snapshot_dir, hf_cache=hf_cache, repo_id=repo_id, revision=revision
    )
    file_digests: dict[str, str] = {}
    for rel in rel_paths:
        path = snapshot_dir / rel
        resolved = _require_readable_file(path, cache=hf_cache, repo_id=repo_id, revision=revision)
        file_digests[rel] = _sha256_file(resolved)
    blob = json.dumps(file_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_dir": str(snapshot_dir),
        "files": file_digests,
        "content_sha256": hashlib.sha256(blob).hexdigest(),
    }


def validate_metabolm_hf_cache(hf_cache: str | Path) -> dict[str, Any]:
    """Validate pinned baseline snapshots under ``hf_cache`` (offline, no download)."""
    cache = _expand_path(hf_cache)
    if not cache.exists():
        raise MetaboLLMCacheError(
            f"MetaboLLM HF cache path does not exist ({cache}). "
            "Refusing hosted-provider fallback."
        )
    if not cache.is_dir():
        raise MetaboLLMCacheError(
            f"MetaboLLM HF cache path is not a directory ({cache}). "
            "Refusing hosted-provider fallback."
        )
    snapshots: dict[str, Any] = {}
    for repo_id, revision in REQUIRED_SNAPSHOTS:
        snap_dir = resolve_snapshot_dir(cache, repo_id, revision)
        snapshots[f"{repo_id}@{revision}"] = fingerprint_snapshot(
            snap_dir, hf_cache=cache, repo_id=repo_id, revision=revision
        )
    aggregate = json.dumps(
        {key: value["content_sha256"] for key, value in snapshots.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "hf_cache": str(cache),
        "snapshots": snapshots,
        "content_sha256": hashlib.sha256(aggregate).hexdigest(),
        "required": [f"{repo}@{rev}" for repo, rev in REQUIRED_SNAPSHOTS],
    }


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON payload is not an object")
    return value


class MetaboLLMBackend:
    """Pinned baseline MetaboLLM for planning/action JSON and grounded transport."""

    name = "metabolm_baseline"

    def __init__(
        self,
        *,
        hf_cache: Optional[str | Path] = None,
        lazy: bool = True,
        planner_max_new_tokens: Optional[int] = None,
        validate: bool = True,
    ) -> None:
        self.hf_cache = resolve_hf_cache(hf_cache)
        self.tokenizer = None
        self.model = None
        self.load_info: dict[str, Any] = {}
        self.cache_validation: dict[str, Any] = {}
        self.weight_fingerprints: dict[str, Any] = {}
        self.resource_paths: dict[str, str] = {"hf_cache": str(self.hf_cache)}
        self._lazy = lazy
        self.planner_max_new_tokens = (
            int(planner_max_new_tokens)
            if planner_max_new_tokens is not None
            else int(GENERATION["max_new_tokens"])
        )
        self.last_generate_meta: dict[str, Any] = {}
        if validate:
            self._refresh_cache_validation()
        if not lazy:
            self.load()

    def _refresh_cache_validation(self) -> dict[str, Any]:
        try:
            self.cache_validation = validate_metabolm_hf_cache(self.hf_cache)
        except MetaboLLMCacheError:
            raise
        except Exception as exc:
            raise MetaboLLMCacheError(
                f"MetaboLLM baseline weights unavailable in local HF cache "
                f"({self.hf_cache}); refusing hosted-provider fallback: {exc}"
            ) from exc
        self.weight_fingerprints = {
            "content_sha256": self.cache_validation["content_sha256"],
            "snapshots": {
                key: {
                    "revision": value["revision"],
                    "content_sha256": value["content_sha256"],
                    "files": value["files"],
                }
                for key, value in self.cache_validation["snapshots"].items()
            },
        }
        self.resource_paths = {
            "hf_cache": str(self.hf_cache),
            **{
                key: value["snapshot_dir"]
                for key, value in self.cache_validation["snapshots"].items()
            },
        }
        return self.cache_validation

    def identity(self) -> dict[str, Any]:
        """Stable resume/provenance identity (no paths, no transient load_info)."""
        return {
            "backend": self.name,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "adapter_id": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION,
            "quantization": QUANTIZATION,
            "decoding": {
                **GENERATION,
                "max_new_tokens": self.planner_max_new_tokens,
            },
            "prompt_version": PROMPT_VERSION,
            "schema_version": PLANNER_SCHEMA_VERSION,
            "grammar_constrained_decoding": False,
            "weight_fingerprints": dict(self.weight_fingerprints),
            "note": "Installed Hub baseline; experimental SFT checkpoints are not used",
        }

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.load_info = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load(self) -> dict[str, Any]:
        if self.model is not None:
            return self.load_info
        # Do not mutate process-wide HF_HOME / hub cache env vars; pass paths explicitly.
        if not self.cache_validation:
            self._refresh_cache_validation()

        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for local MetaboLLM; refusing CPU substitution or hosted fallback"
            )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        hub_cache = str(self.hf_cache / "hub")
        try:
            adapter_dir = snapshot_download(
                repo_id=ADAPTER_ID,
                revision=ADAPTER_REVISION,
                cache_dir=hub_cache,
                local_files_only=True,
            )
            base_dir = snapshot_download(
                repo_id=BASE_ID,
                revision=BASE_REVISION,
                cache_dir=hub_cache,
                local_files_only=True,
            )
        except Exception as exc:
            raise MetaboLLMCacheError(
                "MetaboLLM baseline weights unavailable in local HF cache "
                f"({self.hf_cache}); refusing hosted-provider fallback: {exc}"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir, trust_remote_code=False, local_files_only=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_dir,
            quantization_config=bnb,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        self.model.eval()
        adapters = sorted(getattr(self.model, "peft_config", {}).keys())
        if not adapters:
            raise RuntimeError("PEFT adapter did not load; refusing base-only inference")
        free, total = torch.cuda.mem_get_info()
        # Transient runtime telemetry only — never part of identity()/resume hash.
        self.load_info = {
            "adapter_loaded": True,
            "peft_adapters": adapters,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "adapter_id": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION,
            "quantization": QUANTIZATION,
            "hf_cache": str(self.hf_cache),
            "weight_content_sha256": self.weight_fingerprints.get("content_sha256"),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "vram_free_mb": round(free / 1024**2, 1),
            "vram_total_mb": round(total / 1024**2, 1),
            "merged_adapter_into_quantized_weights": False,
        }
        return self.load_info

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        import torch

        self.load()
        assert self.tokenizer is not None and self.model is not None
        budget = int(max_new_tokens or self.planner_max_new_tokens)
        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        n_input = int(encoded["input_ids"].shape[-1])
        max_ctx = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if max_ctx and n_input + budget > max_ctx:
            meta = {
                "terminal_status": "context_overflow",
                "raw_text": None,
                "truncated": False,
                "error": "prompt exceeds feasible context window; refused silent truncation",
                "usage": {"input_tokens": n_input, "output_tokens": 0, "max_new_tokens": budget},
            }
            self.last_generate_meta = meta
            return meta
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        torch.cuda.empty_cache()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=budget,
                    do_sample=False,
                    use_cache=True,
                    prefill_chunk_size=GENERATION["prefill_chunk_size"],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][encoded["input_ids"].shape[-1] :]
            raw_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            truncated = int(generated.shape[-1]) >= budget
            meta = {
                "terminal_status": "generated",
                "raw_text": raw_text,
                "truncated": truncated,
                "error": "output reached max_new_tokens" if truncated else None,
                "usage": {
                    "input_tokens": n_input,
                    "output_tokens": int(generated.shape[-1]),
                    "max_new_tokens": budget,
                },
            }
            self.last_generate_meta = meta
            return meta
        finally:
            del encoded
            torch.cuda.empty_cache()

    def _fallback_plan(self, inventory: ModelInventory) -> PlannerDecision:
        # Deterministic structural plan used when the local model returns malformed JSON.
        # Callers treat malformed LLM actions as non-executable; this fallback is only
        # used when generate itself fails with a clear technical error path in plan().
        from core.agent.backends.mock import MockBackend

        return MockBackend()._default_plan(inventory)

    def plan(
        self,
        *,
        request: str,
        inventory: ModelInventory,
        prior_plan: Optional[AnnotationPlan] = None,
        observations: Sequence[ToolObservation] = (),
        revision_reason: str = "",
        limits: Any = None,
        entity_states: Any = None,
    ) -> tuple[PlannerDecision, dict[str, Any]]:
        from core.agent.planner import compact_inventory_summary
        from core.agent.schemas import ExecutionLimits

        lim = limits if isinstance(limits, ExecutionLimits) else ExecutionLimits()
        budget = int(lim.planner_max_new_tokens or self.planner_max_new_tokens)
        self.planner_max_new_tokens = budget
        system = (
            "You are the AAAIM automatic annotation planner. "
            "Return one JSON object only. Decisions: continue (next action group), "
            "finish (all in-scope targets terminal), or needs_clarification. "
            f"Emit at most {lim.max_actions_per_turn} actions per continue turn. "
            "Prefer add_missing. Use retrieve_reaction_rrf / select_reaction_grounded "
            "for SBML reactions only. Do not invent unsupported identities. "
            "Do not include export_reports; the controller exports after finish."
            + JSON_TRANSPORT_SUFFIX
            + json.dumps(COMPACT_PLANNER_SCHEMA, sort_keys=True)
        )
        open_tasks = []
        if prior_plan is not None:
            for t in prior_plan.tasks:
                open_tasks.append(
                    {
                        "task_id": t.task_id,
                        "target_id": t.target_id,
                        "tool": t.tool,
                        "state": t.state.value,
                    }
                )
        ledger = {}
        if isinstance(entity_states, dict):
            for tid, est in list(entity_states.items())[:40]:
                ledger[tid] = getattr(est, "state", est).value if hasattr(est, "state") else str(est)
        payload = {
            "request": request,
            "revision_reason": revision_reason,
            "inventory_summary": compact_inventory_summary(inventory),
            "entity_ledger": ledger,
            "open_tasks": open_tasks[-40:],
            "recent_observations": [
                {
                    "action_id": o.action_id,
                    "tool": o.tool,
                    "ok": o.ok,
                    "target_ids": o.target_ids,
                    "error": o.error,
                }
                for o in list(observations)[-6:]
            ],
            "limits": {
                "max_actions_per_turn": lim.max_actions_per_turn,
                "planner_max_new_tokens": budget,
            },
            "allowed_tools": [
                "annotate_species_candidates",
                "assess_identity_support",
                "predict_qualifiers",
                "retrieve_reaction_rrf",
                "select_reaction_grounded",
                "write_species_annotations",
                "write_reaction_annotations",
            ],
            "tool_argument_keys": {
                "annotate_species_candidates": ["databases", "method", "predict_qualifiers"],
                "assess_identity_support": ["effort"],
                "predict_qualifiers": ["effort"],
                "retrieve_reaction_rrf": ["allow_names_only", "top_k"],
                "select_reaction_grounded": ["llm_model", "reasoning_effort", "max_output_tokens"],
                "write_species_annotations": ["add_all", "qualifier"],
                "write_reaction_annotations": ["require_selected"],
            },
            "action_shape_example": {
                "action_id": "act::retrieve::r1",
                "tool": "retrieve_reaction_rrf",
                "target_ids": ["r1"],
                "task_ids": ["retrieve::r1"],
                "arguments": {"allow_names_only": True},
                "rationale": "full-catalog RRF retrieval",
            },
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
        generated = self.generate(messages, max_new_tokens=budget)
        if generated.get("terminal_status") != "generated" or not generated.get("raw_text"):
            raise RuntimeError(
                f"MetaboLLM planning failed: {generated.get('error') or generated.get('terminal_status')}"
            )
        if generated.get("truncated"):
            raise RuntimeError("MetaboLLM planning output truncated; refusing silent acceptance")
        try:
            obj = extract_json_object(generated["raw_text"])
            from core.agent.planner import normalize_planner_decision

            decision = normalize_planner_decision(obj)
            return decision, generated
        except (ValueError, ValidationError) as exc:
            raise RuntimeError(f"MetaboLLM planner returned malformed actions: {exc}") from exc

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        instructions = provider_payload.get("instructions") or ""
        user_input = provider_payload.get("input") or ""
        schema = GroundedAnnotation.model_json_schema()
        system = instructions + JSON_TRANSPORT_SUFFIX + json.dumps(schema, sort_keys=True)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        generated = self.generate(messages)
        model = provider_payload.get("model", "metabolm-local")
        if generated.get("terminal_status") != "generated" or not generated.get("raw_text"):
            return {
                "status": "incomplete",
                "model": model,
                "error": generated.get("error") or generated.get("terminal_status"),
                "incomplete_details": generated.get("error"),
                "output": [],
                "usage": generated.get("usage") or {},
            }
        if generated.get("truncated"):
            return {
                "status": "incomplete",
                "model": model,
                "error": "truncated local output",
                "incomplete_details": "max_new_tokens reached",
                "output": [],
                "usage": generated.get("usage") or {},
            }
        # Local models do not emit OpenAI Responses envelopes; wrap text for classify_response.
        return {
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": generated["raw_text"]}],
                }
            ],
            "usage": generated.get("usage") or {},
        }


def ensure_gpu_idle_enough(*, max_used_mb: float = 3500.0) -> dict[str, Any]:
    """Refuse concurrent heavy GPU loads before MetaboLLM inference."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        used = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        name = torch.cuda.get_device_name(0)
    except Exception as exc:
        raise RuntimeError(f"GPU check failed: {exc}") from exc
    info = {"device": name, "allocated_mb": round(used, 1), "reserved_mb": round(reserved, 1)}
    if used > max_used_mb:
        raise RuntimeError(
            f"GPU already heavily occupied ({used:.0f} MB allocated); "
            "refusing concurrent MetaboLLM load"
        )
    return info
