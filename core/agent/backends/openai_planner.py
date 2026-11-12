"""OpenAI gpt-6-luna planner backend (orchestration only).

Biological annotation inference must use the local MetaboLLM (or mock) path.
This backend does not implement generate()/grounded_transport() for biology.
SDK automatic retries are disabled (max_retries=0).

Structured Outputs use strict=True with a generated schema: every object has
additionalProperties=false and every property is required (nullable unions for
optional semantics). Tool arguments are closed per-tool anyOf variants from
TOOL_ARG_MODELS — no free-form arguments bags.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, get_args, get_origin

from pydantic import ValidationError
from pydantic.fields import FieldInfo

from core.agent.planner_budget import (
    GPT6_LUNA_STANDARD,
    PLANNER_REQUESTED_SERVICE_TIER,
    PlannerBudgetError,
    PlannerBudgetLedger,
    hard_input_token_bound,
)
from core.agent.schemas import (
    ALLOWED_TOOLS,
    TOOL_ARG_MODELS,
    AnnotationPlan,
    BiologicalType,
    EntityRepresentation,
    ExecutionLimits,
    ModelInventory,
    PlannerDecision,
    TaskToolName,
    ToolObservation,
)

PLANNER_MODEL = "gpt-6-luna"
PROMPT_VERSION = "agent-openai-planner-v1"
SCHEMA_VERSION = "agent-planner-decision-v2"

TASK_TOOLS: tuple[str, ...] = get_args(TaskToolName)


def _literal_values(annotation: Any) -> Optional[list[Any]]:
    origin = get_origin(annotation)
    if origin is None:
        return None
    # typing.Literal
    if getattr(origin, "__name__", "") == "Literal" or str(origin) == "typing.Literal":
        return list(get_args(annotation))
    # Python 3.10+ Literal from typing_extensions / typing
    from typing import Literal as _Literal

    if origin is _Literal:
        return list(get_args(annotation))
    return None


def _json_type_for_annotation(annotation: Any) -> dict[str, Any]:
    lit = _literal_values(annotation)
    if lit is not None:
        return {"type": "string", "enum": [str(v) for v in lit]}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is bool or annotation is int or annotation is float or annotation is str:
        mapping = {bool: "boolean", int: "integer", float: "number", str: "string"}
        return {"type": mapping[annotation]}
    if origin is list:
        item = args[0] if args else str
        return {"type": "array", "items": _json_type_for_annotation(item)}
    # Fallback string
    return {"type": "string"}


def _arguments_schema_for_tool(tool: str) -> dict[str, Any]:
    model_cls = TOOL_ARG_MODELS[tool]
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, field in model_cls.model_fields.items():
        assert isinstance(field, FieldInfo)
        schema = _json_type_for_annotation(field.annotation)
        # Preserve integer bounds when present on Field
        if schema.get("type") == "integer":
            if field.metadata:
                for m in field.metadata:
                    ge = getattr(m, "ge", None)
                    le = getattr(m, "le", None)
                    if ge is not None:
                        schema["minimum"] = int(ge)
                    if le is not None:
                        schema["maximum"] = int(le)
        props[name] = schema
        required.append(name)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": required,
    }


def _default_arguments_for_tool(tool: str) -> dict[str, Any]:
    return TOOL_ARG_MODELS[tool]().model_dump(mode="json")


def _action_variant(tool: str) -> dict[str, Any]:
    props = {
        "action_id": {"type": "string"},
        "tool": {"type": "string", "const": tool},
        "target_ids": {"type": "array", "items": {"type": "string"}},
        "task_ids": {"type": "array", "items": {"type": "string"}},
        "arguments": _arguments_schema_for_tool(tool),
        "rationale": {"type": "string"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": list(props.keys()),
    }


def _task_variant(tool: str) -> dict[str, Any]:
    props = {
        "task_id": {"type": "string"},
        "target_id": {"type": "string"},
        "representation": {
            "type": "string",
            "enum": [e.value for e in EntityRepresentation],
        },
        "biological_type": {
            "type": "string",
            "enum": [e.value for e in BiologicalType],
        },
        "tool": {"type": "string", "const": tool},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "arguments": _arguments_schema_for_tool(tool),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": list(props.keys()),
    }


def build_planner_decision_json_schema() -> dict[str, Any]:
    """Strict Structured Outputs schema for PlannerDecision-shaped responses."""
    action_tools = list(ALLOWED_TOOLS)
    task_tools = list(TASK_TOOLS)
    return {
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
                "items": {"anyOf": [_action_variant(t) for t in action_tools]},
            },
            "tasks": {
                "type": "array",
                "items": {"anyOf": [_task_variant(t) for t in task_tools]},
            },
        },
    }


PLANNER_DECISION_JSON_SCHEMA: dict[str, Any] = build_planner_decision_json_schema()


def tool_argument_keys() -> dict[str, list[str]]:
    """Prompt-aligned argument keys derived from TOOL_ARG_MODELS."""
    return {
        tool: list(model.model_fields.keys())
        for tool, model in TOOL_ARG_MODELS.items()
        if tool in TASK_TOOLS
        or tool
        in {
            "annotate_species_candidates",
            "assess_identity_support",
            "predict_qualifiers",
            "retrieve_reaction_rrf",
            "select_reaction_grounded",
            "write_species_annotations",
            "write_reaction_annotations",
        }
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
        "allowed_tools": list(TASK_TOOLS),
        "tool_argument_keys": tool_argument_keys(),
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
        on_budget_journal: Optional[Callable[[], None]] = None,
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
        self.on_budget_journal = on_budget_journal
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
            "strict_structured_outputs": True,
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

    def _journal(self) -> None:
        if self.on_budget_journal is not None:
            self.on_budget_journal()

    def configure_budget(
        self,
        *,
        spend_authorized: bool,
        hard_cap_usd: float,
        max_attempts: Optional[int] = None,
        persist: bool = False,
    ) -> None:
        """Apply spend limits. Does not overwrite an existing ledger by default."""
        self.budget.apply_spend_limits(
            spend_authorized=spend_authorized,
            hard_cap_usd=hard_cap_usd,
            max_attempts=max_attempts,
        )
        if persist and self.budget_path is not None:
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
        schema = PLANNER_DECISION_JSON_SCHEMA
        serialized = serialize_responses_request_for_bound(
            model=self.model,
            messages=messages,
            json_schema=schema,
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
                expected_model=self.model,
                expected_service_tier=str(
                    self.budget.pricing.get("service_tier")
                    or PLANNER_REQUESTED_SERVICE_TIER
                ),
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

        # Durable reservation before dispatch; journal controller checkpoint.
        self._journal()

        requested_tier = str(
            self.budget.pricing.get("service_tier") or PLANNER_REQUESTED_SERVICE_TIER
        )
        client = self._get_client()
        try:
            # Explicit Standard-rate request tier so the API returns the processing
            # mode actually used (Responses docs). Do not invent response identity.
            response = client.responses.create(
                model=self.model,
                service_tier=requested_tier,
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
                        "schema": schema,
                    }
                },
            )
        except Exception as exc:
            self.budget.retain_exposure(
                attempt.attempt_id,
                error=f"request failed: {type(exc).__name__}: {exc}",
                persist_path=self.budget_path,
            )
            self._journal()
            meta = {
                "terminal_status": "error",
                "raw_text": None,
                "error": f"{type(exc).__name__}: {exc}",
                "usage": None,
                "truncated": False,
                "model": self.model,
                "requested_service_tier": requested_tier,
                "attempt_id": attempt.attempt_id,
                "input_token_bound": input_bound,
            }
            self.last_generate_meta = meta
            raise RuntimeError(f"OpenAI planner request failed: {exc}") from exc

        status = getattr(response, "status", None) or (
            response.get("status") if isinstance(response, dict) else None
        )
        response_model, response_tier, identity_meta = _extract_response_identity(
            response
        )
        identity_meta["requested_model"] = self.model
        identity_meta["requested_service_tier"] = requested_tier
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
            self._journal()
            meta = {
                "terminal_status": "incomplete",
                "raw_text": None,
                "error": f"planner response status={status!r} or refusal",
                "usage": usage,
                "truncated": status == "incomplete",
                "model": response_model,
                "response_service_tier": response_tier,
                "requested_service_tier": requested_tier,
                "response_identity": identity_meta,
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
            self._journal()
            meta = {
                "terminal_status": "incomplete",
                "raw_text": None,
                "error": "empty structured output",
                "usage": usage,
                "truncated": False,
                "model": response_model,
                "response_service_tier": response_tier,
                "requested_service_tier": requested_tier,
                "response_identity": identity_meta,
                "attempt_id": attempt.attempt_id,
            }
            self.last_generate_meta = meta
            raise RuntimeError("OpenAI planner returned empty structured output")

        try:
            # Pass through raw response identity only — never substitute request defaults.
            self.budget.settle_validated(
                attempt.attempt_id,
                usage or {},
                persist_path=self.budget_path,
                response_model=response_model,
                service_tier=response_tier,
                response_identity_meta=identity_meta,
            )
        except PlannerBudgetError as exc:
            self._journal()
            meta = {
                "terminal_status": "generated_unsettled",
                "raw_text": raw_text,
                "error": str(exc),
                "usage": usage,
                "truncated": False,
                "model": response_model,
                "response_service_tier": response_tier,
                "requested_service_tier": requested_tier,
                "response_identity": identity_meta,
                "attempt_id": attempt.attempt_id,
            }
            self.last_generate_meta = meta
            raise RuntimeError(str(exc)) from exc

        self._journal()

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
                "response_service_tier": response_tier,
                "requested_service_tier": requested_tier,
                "response_identity": identity_meta,
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
            "response_service_tier": response_tier,
            "requested_model": self.model,
            "requested_service_tier": requested_tier,
            "response_identity": identity_meta,
            "attempt_id": attempt.attempt_id,
            "input_token_bound": input_bound,
            "budget": self.budget.summary(),
            "schema_version": SCHEMA_VERSION,
        }
        self.last_generate_meta = meta
        return decision, meta


def _extract_response_identity(
    response: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Extract model/service_tier from a Responses API object without defaults.

    Missing fields remain None/absent — callers must not substitute the request
    model or requested tier.
    """
    meta: dict[str, Any] = {"response_type": type(response).__name__}
    if isinstance(response, dict):
        response_model = response.get("model", None)
        response_tier = response.get("service_tier", None)
        if "id" in response:
            meta["response_id"] = response.get("id")
        if "object" in response:
            meta["object"] = response.get("object")
        if "status" in response:
            meta["status"] = response.get("status")
        return response_model, response_tier, meta

    response_model = getattr(response, "model", None)
    response_tier = getattr(response, "service_tier", None)
    for attr in ("id", "object", "status"):
        if hasattr(response, attr):
            meta[attr] = getattr(response, attr)
    return response_model, response_tier, meta


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
    out_details = getattr(usage, "output_tokens_details", None)
    if out_details is not None:
        if isinstance(out_details, dict):
            out["output_tokens_details"] = dict(out_details)
        else:
            out["output_tokens_details"] = {
                "reasoning_tokens": getattr(out_details, "reasoning_tokens", None)
            }
    return out or None


def _extract_output_text(response: Any) -> str:
    if isinstance(response, dict):
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


# Re-export for tests that build example arguments
__all__ = [
    "PLANNER_MODEL",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "PLANNER_DECISION_JSON_SCHEMA",
    "OpenAIPlannerBackend",
    "build_planner_decision_json_schema",
    "build_planner_messages",
    "serialize_responses_request_for_bound",
    "tool_argument_keys",
    "_default_arguments_for_tool",
    "TASK_TOOLS",
]
