"""Settlement requires verified response model + service_tier identity."""
from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from core.agent.backends.openai_planner import (
    PLANNER_MODEL,
    OpenAIPlannerBackend,
    _extract_response_identity,
)
from core.agent.planner_budget import (
    PLANNER_REQUESTED_SERVICE_TIER,
    PlannerBudgetError,
    PlannerBudgetLedger,
    reconcile_ledgers,
    verify_settlement_response_identity,
)
from core.agent.schemas import ExecutionLimits


def _ledger() -> PlannerBudgetLedger:
    ledger = PlannerBudgetLedger(spend_authorized=True, hard_cap_usd=1.0)
    ledger.bind_identity(run_id="r", cache_identity="c", model=PLANNER_MODEL)
    return ledger


def _reserve(ledger: PlannerBudgetLedger, path=None):
    return ledger.reserve_attempt(
        input_token_bound=100,
        max_output_tokens=20,
        persist_path=path,
        expected_model=PLANNER_MODEL,
        expected_service_tier=PLANNER_REQUESTED_SERVICE_TIER,
    )


VALID_USAGE = {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.parametrize(
    "response_model,response_tier,match",
    [
        (None, "default", "model"),
        (PLANNER_MODEL, None, "service_tier"),
        (None, None, "missing"),
        ("", "default", "model"),
        (PLANNER_MODEL, "", "service_tier"),
        ("   ", "default", "model"),
        (PLANNER_MODEL, "  ", "service_tier"),
    ],
)
def test_missing_or_blank_identity_cannot_settle(tmp_path, response_model, response_tier, match):
    path = tmp_path / "b.json"
    ledger = _ledger()
    attempt = _reserve(ledger, path)
    with pytest.raises(PlannerBudgetError, match=match):
        ledger.settle_validated(
            attempt.attempt_id,
            VALID_USAGE,
            persist_path=path,
            response_model=response_model,
            service_tier=response_tier,
        )
    restored = PlannerBudgetLedger.load(path)
    assert restored.accounting_status == "blocked"
    assert restored.reserved_usd == 0.0
    assert restored.retained_exposure_usd > 0
    assert restored.charged_usd == 0.0
    with pytest.raises(PlannerBudgetError, match="blocked|accounting"):
        restored.ensure_can_dispatch()


def test_valid_model_tier_settles_and_preserves_requested_vs_actual(tmp_path):
    path = tmp_path / "ok.json"
    ledger = _ledger()
    attempt = _reserve(ledger, path)
    settled = ledger.settle_validated(
        attempt.attempt_id,
        VALID_USAGE,
        persist_path=path,
        response_model=PLANNER_MODEL,
        service_tier="default",
        response_identity_meta={"response_id": "resp_test"},
    )
    assert settled.status == "settled"
    assert settled.expected_model == PLANNER_MODEL
    assert settled.expected_service_tier == PLANNER_REQUESTED_SERVICE_TIER
    assert settled.actual_response_model == PLANNER_MODEL
    assert settled.actual_service_tier == "default"
    assert settled.response_identity_meta.get("response_id") == "resp_test"
    assert ledger.reserved_usd == 0.0
    assert ledger.charged_usd > 0


@pytest.mark.parametrize(
    "response_model,response_tier,match",
    [
        ("other-model", "default", "incompatible|model"),
        (PLANNER_MODEL, "priority", "incompatible|service_tier"),
        (PLANNER_MODEL, "fast", "incompatible|service_tier"),
        (PLANNER_MODEL, "flex", "incompatible|service_tier"),
        (PLANNER_MODEL, "auto", "incompatible|service_tier"),
        (PLANNER_MODEL, "not-a-tier", "unrecognized"),
    ],
)
def test_unknown_or_mismatched_identity_blocks(tmp_path, response_model, response_tier, match):
    path = tmp_path / "bad.json"
    ledger = _ledger()
    attempt = _reserve(ledger, path)
    with pytest.raises(PlannerBudgetError, match=match):
        ledger.settle_validated(
            attempt.attempt_id,
            VALID_USAGE,
            persist_path=path,
            response_model=response_model,
            service_tier=response_tier,
        )
    assert ledger.accounting_status == "blocked"
    assert ledger.retained_exposure_usd > 0


def test_valid_usage_with_missing_identity_does_not_release_reserve(tmp_path):
    path = tmp_path / "usage_ok_id_missing.json"
    ledger = _ledger()
    attempt = _reserve(ledger, path)
    reserved_before = attempt.reserved_usd
    with pytest.raises(PlannerBudgetError, match="identity"):
        ledger.settle_validated(
            attempt.attempt_id,
            {"input_tokens": 1, "output_tokens": 1},
            persist_path=path,
            response_model=None,
            service_tier=None,
        )
    assert ledger.charged_usd == 0.0
    # Reserve released into retained exposure, not settled.
    assert ledger.reserved_usd == 0.0
    assert ledger.retained_exposure_usd == pytest.approx(reserved_before)
    loaded = PlannerBudgetLedger.load(path)
    row = next(a for a in loaded.attempts if a.attempt_id == attempt.attempt_id)
    assert row.status == "retained_exposure"
    assert row.actual_response_model is None
    assert row.actual_service_tier is None
    assert row.expected_model == PLANNER_MODEL


def test_identity_anomaly_survives_restart_and_stale_checkpoint_reconcile(tmp_path):
    path = tmp_path / "planner_budget.json"
    ledger = _ledger()
    attempt = _reserve(ledger, path)
    with pytest.raises(PlannerBudgetError):
        ledger.settle_validated(
            attempt.attempt_id,
            VALID_USAGE,
            persist_path=path,
            response_model=None,
            service_tier="default",
        )
    restored = PlannerBudgetLedger.load(path)
    assert restored.accounting_status == "blocked"
    stale = PlannerBudgetLedger(
        spend_authorized=True,
        hard_cap_usd=1.0,
        run_id=restored.run_id,
        cache_identity=restored.cache_identity,
        model=PLANNER_MODEL,
    )
    merged = reconcile_ledgers(restored, stale)
    assert merged.accounting_status == "blocked"
    assert merged.retained_exposure_usd > 0
    with pytest.raises(PlannerBudgetError, match="blocked"):
        merged.ensure_can_dispatch()


def test_compatibility_policy_helpers():
    assert verify_settlement_response_identity(
        expected_model=PLANNER_MODEL,
        expected_service_tier="default",
        response_model=PLANNER_MODEL,
        response_service_tier="default",
    ) == (PLANNER_MODEL, "default")
    with pytest.raises(PlannerBudgetError, match="missing"):
        verify_settlement_response_identity(
            expected_model=PLANNER_MODEL,
            expected_service_tier="default",
            response_model=None,
            response_service_tier=None,
        )


def test_extract_response_identity_does_not_substitute_defaults():
    model, tier, meta = _extract_response_identity({"status": "completed", "id": "resp_1"})
    assert model is None
    assert tier is None
    assert meta["response_id"] == "resp_1"

    class Obj:
        status = "completed"
        # deliberately no model / service_tier attributes

    model2, tier2, _ = _extract_response_identity(Obj())
    assert model2 is None
    assert tier2 is None


def test_adapter_does_not_replace_absent_response_identity(tmp_path, chemical_only_model):
    from core.agent.inspect_model import inspect_sbml_model

    class _Usage:
        input_tokens = 5
        output_tokens = 2
        input_tokens_details = {"cached_tokens": 0}

    class _Resp:
        status = "completed"
        # Intentionally omit model and service_tier attributes.
        usage = _Usage()
        output_text = json.dumps(
            {
                "decision": "finish",
                "reason": "x",
                "clarification_prompt": "",
                "tasks": [],
                "actions": [],
            }
        )

    class _API:
        def __init__(self) -> None:
            self.kwargs: Optional[dict[str, Any]] = None

        def create(self, **kwargs: Any) -> _Resp:
            self.kwargs = kwargs
            return _Resp()

    api = _API()

    class _Client:
        responses = api

    planner = OpenAIPlannerBackend(
        client_factory=lambda: _Client(),
        budget_path=tmp_path / "planner_budget.json",
    )
    planner.configure_budget(spend_authorized=True, hard_cap_usd=1.0, max_attempts=3)
    inv = inspect_sbml_model(str(chemical_only_model))
    with pytest.raises(RuntimeError, match="identity|anomaly"):
        planner.plan(
            request="annotate",
            inventory=inv,
            limits=ExecutionLimits(planner_spend_authorized=True, planner_spend_cap_usd=1.0),
        )
    # Request asked for default tier; response fields were absent — not filled.
    assert api.kwargs is not None
    assert api.kwargs.get("service_tier") == PLANNER_REQUESTED_SERVICE_TIER
    assert api.kwargs.get("model") == PLANNER_MODEL
    meta = planner.last_generate_meta
    assert meta.get("model") is None
    assert meta.get("response_service_tier") is None
    assert meta.get("requested_service_tier") == PLANNER_REQUESTED_SERVICE_TIER
    assert planner.budget.accounting_status == "blocked"
    assert planner.budget.retained_exposure_usd > 0
    assert planner.budget.charged_usd == 0.0
