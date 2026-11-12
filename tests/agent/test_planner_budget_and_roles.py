"""Mocked component tests for planner budget, role separation, and resume accounting."""
from __future__ import annotations

import json
from typing import Any

import pytest

from core.agent import ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.backends.openai_planner import PLANNER_MODEL, OpenAIPlannerBackend
from core.agent.planner_budget import (
    PlannerBudgetError,
    PlannerBudgetLedger,
    conservative_attempt_cost_usd,
    hard_input_token_bound,
)
from core.agent.schemas import AgentAction, ModelInventory
from core.agent.validate import PlanValidationError, check_authorized_annotation_model, validate_action


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = {"cached_tokens": 0}


class _FakeResponse:
    def __init__(self, text: str, *, status: str = "completed") -> None:
        self.status = status
        self.model = PLANNER_MODEL
        self.service_tier = "standard"
        self.usage = _FakeUsage(12, 8)
        self.output_text = text
        self.output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ]


class _FakeResponsesAPI:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls += 1
        assert kwargs.get("model") == PLANNER_MODEL
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.responses = _FakeResponsesAPI(text)


def _finish_decision_json() -> str:
    return json.dumps(
        {
            "decision": "finish",
            "reason": "mocked structured finish",
            "tasks": [],
            "actions": [],
            "clarification_prompt": "",
        }
    )


class BudgetMock(MockBackend):
    """Mock planner that carries a durable budget ledger for resume tests."""

    def __init__(self) -> None:
        super().__init__()
        self.budget = PlannerBudgetLedger()
        self.budget_path = None

    def identity(self) -> dict[str, Any]:
        return MockBackend().identity()

    def configure_budget(self, **kwargs: Any) -> None:
        self.budget.apply_spend_limits(
            spend_authorized=bool(kwargs.get("spend_authorized")),
            hard_cap_usd=float(kwargs.get("hard_cap_usd") or 0.0),
            max_attempts=kwargs.get("max_attempts"),
        )


def test_hard_input_bound_uses_utf8_bytes():
    payload = '{"a": "ü"}'
    assert hard_input_token_bound(payload) == len(payload.encode("utf-8"))
    with pytest.raises(PlannerBudgetError):
        hard_input_token_bound("")


def test_budget_defaults_block_paid_dispatch(tmp_path):
    ledger = PlannerBudgetLedger()
    assert ledger.spend_authorized is False
    assert ledger.hard_cap_usd == 0.0
    with pytest.raises(PlannerBudgetError, match="zero paid"):
        ledger.reserve_attempt(
            input_token_bound=100,
            max_output_tokens=50,
            persist_path=tmp_path / "b.json",
        )


def test_reserve_settle_and_retain_exposure(tmp_path):
    path = tmp_path / "planner_budget.json"
    ledger = PlannerBudgetLedger(spend_authorized=True, hard_cap_usd=1.0)
    ledger.bind_identity(run_id="r", cache_identity="c", model=PLANNER_MODEL)
    attempt = ledger.reserve_attempt(
        input_token_bound=1000,
        max_output_tokens=100,
        persist_path=path,
        expected_model=PLANNER_MODEL,
    )
    assert attempt.status == "reserved"
    assert path.is_file()
    restored = PlannerBudgetLedger.load(path)
    assert restored.reserved_usd == ledger.reserved_usd

    ledger.settle_validated(
        attempt.attempt_id,
        {"input_tokens": 100, "output_tokens": 20},
        persist_path=path,
        response_model=PLANNER_MODEL,
        service_tier="standard",
    )
    assert ledger.charged_usd > 0
    assert ledger.reserved_usd == 0.0

    attempt2 = ledger.reserve_attempt(
        input_token_bound=500, max_output_tokens=50, persist_path=path
    )
    ledger.retain_exposure(attempt2.attempt_id, error="network", persist_path=path)
    assert ledger.retained_exposure_usd > 0
    ledger.hard_cap_usd = ledger.charged_usd + ledger.retained_exposure_usd
    with pytest.raises(PlannerBudgetError, match="exceeds remaining"):
        ledger.reserve_attempt(input_token_bound=10_000, max_output_tokens=1000)


def test_openai_planner_blocks_when_unauthorized(chemical_only_model):
    from core.agent.inspect_model import inspect_sbml_model

    client = _FakeClient(_finish_decision_json())
    planner = OpenAIPlannerBackend(client_factory=lambda: client)
    inv = inspect_sbml_model(str(chemical_only_model))
    with pytest.raises(RuntimeError, match="zero paid|not authorized"):
        planner.plan(request="annotate", inventory=inv, limits=ExecutionLimits())
    assert client.responses.calls == 0


def test_openai_planner_settles_mocked_dispatch(tmp_path, chemical_only_model):
    from core.agent.inspect_model import inspect_sbml_model

    client = _FakeClient(_finish_decision_json())
    planner = OpenAIPlannerBackend(
        client_factory=lambda: client,
        budget_path=tmp_path / "planner_budget.json",
    )
    planner.configure_budget(spend_authorized=True, hard_cap_usd=1.0, max_attempts=3)
    inv = inspect_sbml_model(str(chemical_only_model))
    decision, meta = planner.plan(
        request="annotate missing",
        inventory=inv,
        limits=ExecutionLimits(planner_spend_authorized=True, planner_spend_cap_usd=1.0),
    )
    assert decision.decision == "finish"
    assert client.responses.calls == 1
    assert meta.get("usage")
    assert planner.budget.charged_usd > 0
    assert (tmp_path / "planner_budget.json").is_file()


def test_openai_planner_refuses_biology_paths():
    planner = OpenAIPlannerBackend(client_factory=lambda: _FakeClient("{}"))
    with pytest.raises(RuntimeError, match="refuses biological generate"):
        planner.generate("prompt")
    with pytest.raises(RuntimeError, match="refuses grounded_transport"):
        planner.grounded_transport({"model": "metabolm-local", "input": []})


def test_unauthorized_annotation_model_rejected():
    problems = check_authorized_annotation_model(
        "select_reaction_grounded",
        {"llm_model": "gpt-6-luna"},
        ["metabolm-local"],
    )
    assert problems and "unauthorized" in problems[0]

    action = AgentAction(
        action_id="a1",
        tool="select_reaction_grounded",
        target_ids=["r1"],
        task_ids=["t1"],
        arguments={
            "llm_model": "gpt-6-luna",
            "reasoning_effort": "low",
            "max_output_tokens": 256,
        },
        rationale="bad",
    )
    with pytest.raises(PlanValidationError, match="unauthorized"):
        validate_action(
            action,
            ModelInventory(model_file="x", sha256="0" * 64, entities=[]),
            known_task_ids=["t1"],
            authorized_annotation_models=["metabolm-local"],
        )


def test_dual_roles_wire_planner_not_inference(chemical_only_model, tmp_path):
    """Planner receives plan(); inference must not."""

    class PlannerOnly(MockBackend):
        def grounded_transport(self, provider_payload):
            raise AssertionError("planner must not receive grounded_transport")

    class InferenceOnly(MockBackend):
        def plan(self, **kwargs):
            raise AssertionError("inference must not receive plan()")

    planner = PlannerOnly()
    inference = InferenceOnly()
    result = auto_annotate_model(
        str(chemical_only_model),
        planner=planner,
        inference=inference,
        output_dir=tmp_path / "roles",
        allow_structural_fallback=True,
    )
    assert planner.plan_calls >= 1
    assert result.provenance["planner"]["backend"] == "mock"
    assert result.provenance["inference"]["backend"] == "mock"
    ckpt = json.loads((tmp_path / "roles" / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["schema_version"] == 3
    assert ckpt.get("planner") and ckpt.get("inference")


def test_resume_restores_planner_budget_accounting(tmp_path, chemical_only_model):
    out = tmp_path / "budget_resume"
    spend_limits = ExecutionLimits(
        planner_spend_authorized=True,
        planner_spend_cap_usd=1.0,
        max_planning_turns=12,
    )
    interrupted = auto_annotate_model(
        str(chemical_only_model),
        backend=BudgetMock(),
        output_dir=out,
        interrupt_after_actions=1,
        limits=spend_limits,
    )
    assert interrupted.report.status == "interrupted"
    ckpt_path = out / "checkpoint.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert ckpt["schema_version"] == 3

    # Authoritative file ahead of checkpoint: settled + retained exposure.
    budget_path = out / "planner_budget.json"
    file_ledger = PlannerBudgetLedger.load(budget_path)
    file_ledger.apply_spend_limits(spend_authorized=True, hard_cap_usd=1.0)
    a1 = file_ledger.reserve_attempt(
        input_token_bound=1000,
        max_output_tokens=100,
        persist_path=budget_path,
        expected_model=PLANNER_MODEL,
    )
    file_ledger.settle_validated(
        a1.attempt_id,
        {"input_tokens": 100, "output_tokens": 20},
        persist_path=budget_path,
        response_model=PLANNER_MODEL,
        service_tier="standard",
    )
    a2 = file_ledger.reserve_attempt(
        input_token_bound=500, max_output_tokens=50, persist_path=budget_path
    )
    file_ledger.retain_exposure(a2.attempt_id, error="uncertain", persist_path=budget_path)
    charged = file_ledger.charged_usd
    retained = file_ledger.retained_exposure_usd
    assert charged > 0 and retained > 0

    resumed_backend = BudgetMock()
    resumed = auto_annotate_model(
        str(chemical_only_model),
        backend=resumed_backend,
        output_dir=out,
        resume_from=out,
        limits=spend_limits,
    )
    assert resumed.report.status in {"completed", "partial"}
    assert resumed_backend.budget.charged_usd == pytest.approx(charged)
    assert resumed_backend.budget.retained_exposure_usd == pytest.approx(retained)
    assert resumed.provenance.get("planner_usage", {}).get("charged_usd") == pytest.approx(
        charged
    )


def test_cli_rejects_conflicting_backend_and_roles():
    from core.agent.cli import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "model.xml",
                "--backend",
                "mock",
                "--planner",
                "openai",
                "--inference",
                "metabolm",
            ]
        )
    assert "conflicting" in str(exc.value).lower()


def test_conservative_cost_never_uses_cache_discount():
    cheap = conservative_attempt_cost_usd(
        input_token_bound=100_000, max_output_tokens=1000, assume_uncached=True
    )
    also = conservative_attempt_cost_usd(
        input_token_bound=100_000, max_output_tokens=1000, assume_uncached=False
    )
    assert cheap == also
    assert cheap > 0
