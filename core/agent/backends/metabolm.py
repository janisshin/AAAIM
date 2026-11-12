"""Local MetaboLLM baseline backend for planning and grounded selection.

Uses the installed Hub baseline adapter (not experimental SFT checkpoints).
No hosted-provider fallback. No silent truncation.
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from core.agent.schemas import (
    AgentAction,
    AnnotationPlan,
    AnnotationStatus,
    AnnotationTask,
    BiologicalType,
    EntityRepresentation,
    FailureKind,
    ModelInventory,
    PlannerDecision,
    TaskState,
    ToolObservation,
)
from core.reaction.grounded import GroundedAnnotation, classify_response

ADAPTER_ID = "MetaboLLM/MetaboLLM-Qwen3-4B"
BASE_ID = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_REVISION = "13deefb4762ac259b35bd61a5701c7abd50b02b0"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
PLANNER_SCHEMA_VERSION = "agent-planner-v1"
PROMPT_VERSION = "agent-metabolm-planner-v1"

QUANTIZATION = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
GENERATION = {
    "do_sample": False,
    "max_new_tokens": 512,
    "prefill_chunk_size": 512,
}

JSON_TRANSPORT_SUFFIX = (
    "\n\nLOCAL_TRANSPORT_JSON_SCHEMA\n"
    "Return exactly one JSON object matching the schema and no other text:\n"
)


def default_hf_cache() -> Path:
    # Prefer the verified comparison cache on the main workspace.
    root = Path(__file__).resolve()
    # .../auto_annotate_agent/core/agent/backends/metabolm.py
    # parents: backends, agent, core, auto_annotate_agent, integration_worktrees, benchmark, repo
    repo = root.parents[6]
    return repo / "benchmark" / "phase3" / "metabolm_grounded_comparison" / "_hf_cache"


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
    ) -> None:
        self.hf_cache = Path(hf_cache) if hf_cache else default_hf_cache()
        self.tokenizer = None
        self.model = None
        self.load_info: dict[str, Any] = {}
        self._lazy = lazy
        if not lazy:
            self.load()

    def identity(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "adapter_id": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION,
            "quantization": QUANTIZATION,
            "decoding": GENERATION,
            "prompt_version": PROMPT_VERSION,
            "schema_version": PLANNER_SCHEMA_VERSION,
            "grammar_constrained_decoding": False,
            "hf_cache": str(self.hf_cache),
            "load_info": dict(self.load_info),
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
        os.environ.setdefault("HF_HOME", str(self.hf_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.hf_cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_cache / "transformers"))
        self.hf_cache.mkdir(parents=True, exist_ok=True)

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
        # Prefer offline cache hits from the verified comparison HF cache.
        # Refuse hosted LLM providers; Hub snapshot access is only for local weights.
        try:
            adapter_dir = snapshot_download(
                repo_id=ADAPTER_ID,
                revision=ADAPTER_REVISION,
                cache_dir=str(self.hf_cache / "hub"),
                local_files_only=True,
            )
            base_dir = snapshot_download(
                repo_id=BASE_ID,
                revision=BASE_REVISION,
                cache_dir=str(self.hf_cache / "hub"),
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
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
        self.load_info = {
            "adapter_loaded": True,
            "peft_adapters": adapters,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "adapter_id": ADAPTER_ID,
            "adapter_revision": ADAPTER_REVISION,
            "quantization": QUANTIZATION,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "vram_free_mb": round(free / 1024**2, 1),
            "vram_total_mb": round(total / 1024**2, 1),
            "merged_adapter_into_quantized_weights": False,
        }
        return self.load_info

    def generate(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        import torch

        self.load()
        assert self.tokenizer is not None and self.model is not None
        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        n_input = int(encoded["input_ids"].shape[-1])
        max_ctx = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if max_ctx and n_input + GENERATION["max_new_tokens"] > max_ctx:
            return {
                "terminal_status": "context_overflow",
                "raw_text": None,
                "truncated": False,
                "error": "prompt exceeds feasible context window; refused silent truncation",
                "usage": {"input_tokens": n_input, "output_tokens": 0},
            }
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        torch.cuda.empty_cache()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=GENERATION["max_new_tokens"],
                    do_sample=False,
                    use_cache=True,
                    prefill_chunk_size=GENERATION["prefill_chunk_size"],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][encoded["input_ids"].shape[-1] :]
            raw_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            truncated = int(generated.shape[-1]) >= GENERATION["max_new_tokens"]
            return {
                "terminal_status": "generated",
                "raw_text": raw_text,
                "truncated": truncated,
                "error": "output reached max_new_tokens" if truncated else None,
                "usage": {
                    "input_tokens": n_input,
                    "output_tokens": int(generated.shape[-1]),
                },
            }
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
    ) -> PlannerDecision:
        schema = PlannerDecision.model_json_schema()
        system = (
            "You are the AAAIM automatic annotation planner. "
            "Choose allowlisted tools only. Return one JSON object. "
            "Prefer add_missing scope. Use reaction_method rrf_grounded tools only "
            "for SBML reactions. Do not invent unsupported identities."
            + JSON_TRANSPORT_SUFFIX
            + json.dumps(schema, sort_keys=True)
        )
        payload = {
            "request": request,
            "revision_reason": revision_reason,
            "inventory": inventory.model_dump(mode="json"),
            "prior_plan": prior_plan.model_dump(mode="json") if prior_plan else None,
            "observations": [o.model_dump(mode="json") for o in observations[-8:]],
            "allowed_tools": [
                "annotate_species_candidates",
                "assess_identity_support",
                "predict_qualifiers",
                "retrieve_reaction_rrf",
                "select_reaction_grounded",
                "write_species_annotations",
                "write_reaction_annotations",
                "export_reports",
            ],
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
        generated = self.generate(messages)
        if generated.get("terminal_status") != "generated" or not generated.get("raw_text"):
            raise RuntimeError(
                f"MetaboLLM planning failed: {generated.get('error') or generated.get('terminal_status')}"
            )
        if generated.get("truncated"):
            raise RuntimeError("MetaboLLM planning output truncated; refusing silent acceptance")
        try:
            obj = extract_json_object(generated["raw_text"])
            return PlannerDecision.model_validate(obj)
        except (ValueError, ValidationError) as exc:
            # Malformed actions must not execute. Raise so the controller records technical failure
            # and can fall back to a deterministic structural plan only via explicit controller policy.
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
