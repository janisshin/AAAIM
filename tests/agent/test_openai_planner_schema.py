"""Offline strict Structured Outputs schema checks for the OpenAI planner."""
from __future__ import annotations

import json
from typing import Any

import pytest

from core.agent.backends.openai_planner import (
    PLANNER_DECISION_JSON_SCHEMA,
    PLANNER_MODEL,
    TASK_TOOLS,
    OpenAIPlannerBackend,
    _default_arguments_for_tool,
    build_planner_decision_json_schema,
    build_planner_messages,
    serialize_responses_request_for_bound,
)
from core.agent.planner import normalize_planner_decision
from core.agent.schemas import (
    ALLOWED_TOOLS as SCHEMA_ALLOWED_TOOLS,
    ExecutionLimits,
    PlannerDecision,
    parse_tool_arguments,
)


def _walk_objects(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            found.append((path, node))
        for key, val in node.items():
            found.extend(_walk_objects(val, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, val in enumerate(node):
            found.extend(_walk_objects(val, f"{path}[{i}]"))
    return found


def assert_strict_schema_compatible(schema: dict[str, Any]) -> None:
    """OpenAI strict Structured Outputs constraints (documented)."""
    blob = json.dumps(schema)
    assert "additionalProperties\": true" not in blob.lower().replace(" ", "")
    assert '"additionalProperties": true' not in blob
    for path, obj in _walk_objects(schema):
        assert obj.get("additionalProperties") is False, f"{path} missing additionalProperties:false"
        props = obj.get("properties") or {}
        required = obj.get("required")
        assert isinstance(required, list), f"{path} missing required array"
        assert set(required) == set(props.keys()), (
            f"{path} required={sorted(required)} != properties={sorted(props)}"
        )


def test_module_schema_is_strict_compatible():
    schema = build_planner_decision_json_schema()
    assert schema == PLANNER_DECISION_JSON_SCHEMA
    assert_strict_schema_compatible(schema)
    # No orphan task keys
    for variant in schema["properties"]["tasks"]["items"]["anyOf"]:
        props = variant["properties"]
        assert "rationale" not in props
        assert "completion_condition" not in props


def test_submitted_request_embeds_identical_strict_schema(chemical_only_model):
    from core.agent.inspect_model import inspect_sbml_model

    captured: dict[str, Any] = {}

    class _Resp:
        status = "completed"
        model = PLANNER_MODEL
        service_tier = "default"
        output_text = json.dumps(
            {
                "decision": "finish",
                "reason": "done",
                "clarification_prompt": "",
                "tasks": [],
                "actions": [],
            }
        )
        usage = {"input_tokens": 1, "output_tokens": 1}

        class _U:
            input_tokens = 1
            output_tokens = 1
            input_tokens_details = {"cached_tokens": 0}

        usage = _U()

    class _API:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _Resp()

    class _Client:
        responses = _API()

    inv = inspect_sbml_model(str(chemical_only_model))
    planner = OpenAIPlannerBackend(client_factory=lambda: _Client())
    planner.configure_budget(spend_authorized=True, hard_cap_usd=1.0, max_attempts=3)
    messages = build_planner_messages(
        request="annotate",
        inventory=inv,
        prior_plan=None,
        observations=[],
        revision_reason="",
        limits=ExecutionLimits(planner_spend_authorized=True, planner_spend_cap_usd=1.0),
    )
    bound_body = json.loads(
        serialize_responses_request_for_bound(
            model=PLANNER_MODEL,
            messages=messages,
            json_schema=PLANNER_DECISION_JSON_SCHEMA,
            max_output_tokens=64,
        )
    )
    assert bound_body["text"]["format"]["strict"] is True
    assert_strict_schema_compatible(bound_body["text"]["format"]["schema"])

    planner.plan(
        request="annotate",
        inventory=inv,
        limits=ExecutionLimits(planner_spend_authorized=True, planner_spend_cap_usd=1.0),
    )
    fmt = captured["kwargs"]["text"]["format"]
    assert fmt["strict"] is True
    assert fmt["schema"] == PLANNER_DECISION_JSON_SCHEMA
    assert fmt["schema"] == bound_body["text"]["format"]["schema"]


@pytest.mark.parametrize("tool", list(SCHEMA_ALLOWED_TOOLS))
def test_each_action_tool_variant_parses(tool: str):
    args = _default_arguments_for_tool(tool)
    decision_obj = {
        "decision": "continue",
        "reason": "test",
        "clarification_prompt": "",
        "tasks": [],
        "actions": [
            {
                "action_id": f"a::{tool}",
                "tool": tool,
                "target_ids": ["x"],
                "task_ids": [f"{tool}::x"],
                "arguments": args,
                "rationale": "unit",
            }
        ],
    }
    decision = normalize_planner_decision(decision_obj)
    assert isinstance(decision, PlannerDecision)
    assert decision.actions[0].tool == tool
    parsed = parse_tool_arguments(tool, decision.actions[0].arguments)
    assert parsed.model_dump(mode="json") == args
    # Schema branch exists
    consts = [
        v["properties"]["tool"]["const"]
        for v in PLANNER_DECISION_JSON_SCHEMA["properties"]["actions"]["items"]["anyOf"]
    ]
    assert tool in consts


@pytest.mark.parametrize("tool", list(TASK_TOOLS))
def test_each_task_tool_variant_parses(tool: str):
    args = _default_arguments_for_tool(tool)
    decision_obj = {
        "decision": "continue",
        "reason": "test",
        "clarification_prompt": "",
        "actions": [],
        "tasks": [
            {
                "task_id": f"{tool}::x",
                "target_id": "x",
                "representation": "sbml_species",
                "biological_type": "chemical",
                "tool": tool,
                "depends_on": [],
                "arguments": args,
            }
        ],
    }
    decision = normalize_planner_decision(decision_obj)
    assert decision.tasks[0].tool == tool
    parse_tool_arguments(tool, decision.tasks[0].arguments)
    consts = [
        v["properties"]["tool"]["const"]
        for v in PLANNER_DECISION_JSON_SCHEMA["properties"]["tasks"]["items"]["anyOf"]
    ]
    assert tool in consts


def test_schema_rejects_free_form_arguments_shape():
    for variant in PLANNER_DECISION_JSON_SCHEMA["properties"]["actions"]["items"]["anyOf"]:
        args = variant["properties"]["arguments"]
        assert args.get("additionalProperties") is False
        assert "additionalProperties" in args
