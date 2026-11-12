"""OpenAI gpt-6-luna planner backend (orchestration only).

Biological annotation inference must use the local MetaboLLM (or mock) path.
This backend does not implement generate()/grounded_transport() for biology.
SDK automatic retries are disabled (max_retries=0).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import ValidationError

from core.agent.planner_budget import (
    GPT6_LUNA_STANDARD,
    PlannerBudgetError,
    PlannerBudgetLedger,
    hard_input_token_bound,
)
from core.agent.schemas import (
    AnnotationPlan,
    ExecutionLimits,
    ModelInventory,
    PlannerDecision,
    ToolObservation,
)

PLANNER_MODEL = "gpt-6-luna"
PROMPT_VERSION = "agent-openai-planner-v1"
SCHEMA_VERSION = "agent-planner-decision-v1"

# Strict JSON Schema for Responses API structured outputs (PlannerDecision-shaped).
PLANNER_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "actions", "tasks", "reason", "clarification_prompt"],
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
                "additionalProperties": False,
                "required": [
                    "action_id",
                    "tool",
                    "target_ids",
                    "task_ids",
                    "arguments",
                    "rationale",
                ],
                "properties": {
                    "action_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "task_ids": {"type": "array", "items": {"type": "string"}},
                    "arguments": {"type": "object", "additionalProperties": True},
                    "rationale": {"type": "string"},
                },
            },
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "task_id",
                    "target_id",
                    "representation",
                    "biological_type",
                    "tool",
                    "depends_on",
                    "arguments",
                ],
                "properties": {
                    "task_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "representation": {"type": "string"},
                    "biological_type": {"type": "string"},
                    "tool": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "arguments": {"type": "object", "additionalProperties": True},
                    "rationale": {"type": "string"},
                    "completion_condition": {"type": "string"},
                },
            },
        },
    },
}


def build_planner_messages(
    *,
    request: str,
    inventory: ModelInventory,
    prior_plan: Optional[AnnotationPlan],
    observations: Sequence[ToolObservation],
    revision_reason: str,
    limits: ExecutionLimits,
    entity_states: Any = None,
    authorized_annotation_models: Sequence[str] = ("metabolm-local",),
) -> list[dict[str, str]]:
    from core.agent.planner import compact_inventory_summary

    system = (
        "You are the AAAIM automatic annotation planner (orchestration only). "
        "Return one JSON object matching the provided schema. "
        "Decisions: continue (next action group), finish (all in-scope targets "
        "terminal), or needs_clarification. "
        f"Emit at most {limits.max_actions_per_turn} actions per continue turn. "
        "Prefer add_missing. Use retrieve_reaction_rrf / select_reaction_grounded "
        "for SBML reactions only. "
        "For select_reaction_grounded.arguments.llm_model you MUST request an "
        f"authorized local annotation model only: {list(authorized_annotation_models)}. "
        "Do not request OpenAI or other hosted models for biological annotation. "
        "Do not invent unsupported identities. "
        "Do not include export_reports; the controller exports after finish."
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
    ledger: dict[str, Any] = {}
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
            "max_actions_per_turn": limits.max_actions_per_turn,
            "planner_max_output_tokens": limits.planner_max_new_tokens,
        },
        "authorized_annotation_models": list(authorized_annotation_models),
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
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def serialize_responses_request_for_bound(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    json_schema: dict[str, Any],
    max_output_tokens: int,
) -> str:
    """Exact request body text used to establish the hard input-token bound."""
    body = {
        "model": model,
        "input": [
            {
                "role": m["role"],
                "content": [{"type": "input_text", "text": m["content"]}],
            }
            for m in messages
        ],
        "max_output_tokens": int(max_output_tokens),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "aaaim_planner_decision",
                "strict": True,
                "schema": json_schema,
            }
        },
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


class OpenAIPlannerBackend:
    """Hosted orchestration planner. Not an annotation inference backend."""

    name = "openai_planner"

    def __init__(
        self,
        *,
        model: str = PLANNER_MODEL,
        api_key: Optional[str] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        budget_ledger: Optional[PlannerBudgetLedger] = None,
        budget_path: Optional[str | Path] = None,
        authorized_annotation_models: Optional[Sequence[str]] = None,
    ) -> None:
        if model != PLANNER_MODEL:
            raise ValueError(
                f"OpenAIPlannerBackend refuses model substitution: requested {model!r}, "
                f"required {PLANNER_MODEL!r}"
            )
        self.model = PLANNER_MODEL
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self._client_factory = client_factory
        self.budget_path = Path(budget_path) if budget_path else None
        self.budget = budget_ledger or PlannerBudgetLedger(
            spend_authorized=False,
            hard_cap_usd=0.0,
            pricing=dict(GPT6_LUNA_STANDARD),
            model=PLANNER_MODEL,
        )
        self.authorized_annotation_models = list(
            authorized_annotation_models or ("metabolm-local",)
        )
        self.last_generate_meta: dict[str, Any] = {}
        self._client = None

    def identity(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "role": "planner",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "structured_outputs": True,
            "api": "responses",
            "max_retries": 0,
            "authorized_annotation_models": list(self.authorized_annotation_models),
            "pricing": dict(self.budget.pricing),
            "pricing_checked": self.budget.pricing_checked,
            "note": "Orchestration only; biological inference must use local MetaboLLM",
        }

    def unload(self) -> None:
        self._client = None

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "OpenAIPlannerBackend refuses biological generate(); "
            "use local MetaboLLM inference for species/reaction annotation"
        )

    def grounded_transport(self, provider_payload: Mapping[str, Any]) -> dict[str, Any]:
        raise RuntimeError(
            "OpenAIPlannerBackend refuses grounded_transport(); "
            "use local MetaboLLM inference for reaction selection"
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing; OpenAI planner cannot dispatch "
                "(default budget also permits zero paid calls)"
            )
        from openai import OpenAI

        # Disable SDK automatic retries — budget ledger owns attempt authorization.
        self._client = OpenAI(api_key=self.api_key, max_retries=0)
        return self._client

    def configure_budget(
        self,
        *,
        spend_authorized: bool,
        hard_cap_usd: float,
        max_attempts: Optional[int] = None,
    ) -> None:
        self.budget.spend_authorized = bool(spend_authorized)
        self.budget.hard_cap_usd = float(hard_cap_usd)
        if max_attempts is not None:
            self.budget.max_attempts = int(max_attempts)
        if self.budget_path is not None:
            self.budget.persist(self.budget_path)

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
        lim = limits if isinstance(limits, ExecutionLimits) else ExecutionLimits()
        max_out = int(lim.planner_max_new_tokens or 1024)
        messages = build_planner_messages(
            request=request,
            inventory=inventory,
            prior_plan=prior_plan,
            observations=observations,
            revision_reason=revision_reason,
            limits=lim,
            entity_states=entity_states,
            authorized_annotation_models=self.authorized_annotation_models,
        )
        serialized = serialize_responses_request_for_bound(
            model=self.model,
            messages=messages,
            json_schema=PLANNER_DECISION_JSON_SCHEMA,
            max_output_tokens=max_out,
        )
        try:
            input_bound = hard_input_token_bound(serialized)
        except PlannerBudgetError as exc:
            meta = {
                "terminal_status": "blocked",
                "raw_text": None,
                "error": str(exc),
                "usage": None,
                "truncated": False,
                "model": self.model,
            }
            self.last_generate_meta = meta
            raise RuntimeError(str(exc)) from exc

        try:
            attempt = self.budget.reserve_attempt(
                input_token_bound=input_bound,
                max_output_tokens=max_out,
                persist_path=self.budget_path,
            )
        except PlannerBudgetError as exc:
            meta = {
                "terminal_status": "blocked",
                "raw_text": None,
                "error": str(exc),
                "usage": None,
                "truncated": False,
                "model": self.model,
                "input_token_bound": input_bound,
            }
            self.last_generate_meta = meta
            raise RuntimeError(str(exc)) from exc

        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": m["role"],
                        "content": [{"type": "input_text", "text": m["content"]}],
                    }
                    for m in messages
                ],
                max_output_tokens=max_out,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "aaaim_planner_decision",
                        "strict": True,
                        "schema": PLANNER_DECISION_JSON_SCHEMA,
                    }
                },
            )
        except Exception as exc:
            self.budget.retain_exposure(
                attempt.attempt_id,
                error=f"request failed: {type(exc).__name__}: {exc}",
                persist_path=self.budget_path,
            )
            meta = {
                "terminal_status": "error",
                "raw_text": None,
                "error": f"{type(exc).__name__}: {exc}",
                "usage": None,
                "truncated": False,
                "model": self.model,
                "attempt_id": attempt.attempt_id,
                "input_token_bound": input_bound,
            }
            self.last_generate_meta = meta
            raise RuntimeError(f"OpenAI planner request failed: {exc}") from exc

        status = getattr(response, "status", None) or (
            response.get("status") if isinstance(response, dict) else None
        )
        response_model = getattr(response, "model", None) or (
            response.get("model") if isinstance(response, dict) else self.model
        )
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None and isinstance(response, dict):
            usage_obj = response.get("usage")
        usage = _usage_to_dict(usage_obj)

        if status in {"incomplete", "failed"} or status is None and _is_refusal(response):
            self.budget.retain_exposure(
                attempt.attempt_id,
                error=f"incomplete_or_refused status={status!r}",
                persist_path=self.budget_path,
            )
            meta = {
                "terminal_status": "incomplete",
                "raw_text": None,
                "error": f"planner response status={status!r} or refusal",
                "usage": usage,
                "truncated": status == "incomplete",
                "model": response_model,
                "attempt_id": attempt.attempt_id,
                "input_token_bound": input_bound,
            }
            self.last_generate_meta = meta
            raise RuntimeError(meta["error"])

        raw_text = _extract_output_text(response)
        if not raw_text:
            self.budget.retain_exposure(
                attempt.attempt_id,
                error="empty structured output",
                persist_path=self.budget_path,
            )
            meta = {
                "terminal_status": "incomplete",
                "raw_text": None,
                "error": "empty structured output",
                "usage": usage,
                "truncated": False,
                "model": response_model,
                "attempt_id": attempt.attempt_id,
            }
            self.last_generate_meta = meta
            raise RuntimeError("OpenAI planner returned empty structured output")

        try:
            self.budget.settle_validated(
                attempt.attempt_id, usage or {}, persist_path=self.budget_path
            )
        except PlannerBudgetError as exc:
            # Exposure already retained inside settle_validated when usage invalid.
            meta = {
                "terminal_status": "generated_unsettled",
                "raw_text": raw_text,
                "error": str(exc),
                "usage": usage,
                "truncated": False,
                "model": response_model,
                "attempt_id": attempt.attempt_id,
            }
            self.last_generate_meta = meta
            # Still try to parse for caller diagnostics, but fail closed on budget.
            raise RuntimeError(str(exc)) from exc

        try:
            obj = json.loads(raw_text)
            from core.agent.planner import normalize_planner_decision

            decision = normalize_planner_decision(obj)
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            meta = {
                "terminal_status": "generated",
                "raw_text": raw_text,
                "error": f"malformed planner JSON: {exc}",
                "usage": usage,
                "truncated": False,
                "model": response_model,
                "attempt_id": attempt.attempt_id,
            }
            self.last_generate_meta = meta
            raise RuntimeError(f"OpenAI planner returned malformed actions: {exc}") from exc

        meta = {
            "terminal_status": "generated",
            "raw_text": raw_text,
            "error": None,
            "usage": usage,
            "truncated": False,
            "model": response_model,
            "attempt_id": attempt.attempt_id,
            "input_token_bound": input_bound,
            "budget": self.budget.summary(),
        }
        self.last_generate_meta = meta
        return decision, meta


def _usage_to_dict(usage: Any) -> Optional[dict[str, Any]]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    ):
        if hasattr(usage, key):
            out[key] = getattr(usage, key)
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        if isinstance(details, dict):
            out["input_tokens_details"] = dict(details)
        else:
            out["input_tokens_details"] = {
                "cached_tokens": getattr(details, "cached_tokens", None)
            }
    return out or None


def _extract_output_text(response: Any) -> str:
    if isinstance(response, dict):
        # dict-shaped mock
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in {
                    "output_text",
                    "text",
                }:
                    text = part.get("text")
                    if text:
                        return str(text)
        return str(response.get("output_text") or "")
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            ptype = getattr(part, "type", None)
            if ptype in {"output_text", "text"}:
                chunks.append(str(getattr(part, "text", "") or ""))
    return "".join(chunks)


def _is_refusal(response: Any) -> bool:
    if isinstance(response, dict):
        for item in response.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "refusal":
                return True
            for part in (item.get("content") if isinstance(item, dict) else None) or []:
                if isinstance(part, dict) and part.get("type") == "refusal":
                    return True
        return False
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "refusal":
            return True
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return True
    return False
