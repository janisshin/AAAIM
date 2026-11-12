"""Crash-window resume and settlement-bound regression tests for planner ledger v2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agent import ExecutionLimits, auto_annotate_model
from core.agent.backends.mock import MockBackend
from core.agent.backends.openai_planner import PLANNER_MODEL
from core.agent.planner_budget import (
    LEDGER_SCHEMA_VERSION,
    PlannerBudgetError,
    PlannerBudgetLedger,
    reconcile_ledgers,
)


class BudgetMock(MockBackend):
    def __init__(self) -> None:
        super().__init__()
        self.budget = PlannerBudgetLedger()
        self.budget_path = None

    def identity(self):
        return MockBackend().identity()

    def configure_budget(self, **kwargs):
        self.budget.apply_spend_limits(
            spend_authorized=bool(kwargs.get("spend_authorized")),
            hard_cap_usd=float(kwargs.get("hard_cap_usd") or 0.0),
            max_attempts=kwargs.get("max_attempts"),
        )


def _bound_ledger(**kwargs) -> PlannerBudgetLedger:
    kwargs.setdefault("spend_authorized", True)
    kwargs.setdefault("hard_cap_usd", 1.0)
    ledger = PlannerBudgetLedger(**kwargs)
    ledger.bind_identity(
        run_id="run-test",
        cache_identity="cache-test",
        model=PLANNER_MODEL,
    )
    return ledger


_SPEND_LIMITS = ExecutionLimits(
    planner_spend_authorized=True,
    planner_spend_cap_usd=1.0,
    max_planning_turns=12,
)


def test_exact_boundary_usage_settles(tmp_path):
    path = tmp_path / "planner_budget.json"
    ledger = _bound_ledger()
    attempt = ledger.reserve_attempt(
        input_token_bound=100,
        max_output_tokens=50,
        persist_path=path,
        expected_model=PLANNER_MODEL,
    )
    settled = ledger.settle_validated(
        attempt.attempt_id,
        {"input_tokens": 100, "output_tokens": 50},
        persist_path=path,
        response_model=PLANNER_MODEL,
        service_tier="standard",
    )
    assert settled.status == "settled"
    assert ledger.reserved_usd == 0.0
    assert ledger.charged_usd > 0
    assert ledger.accounting_status == "ok"


def test_usage_overflow_blocks_further_dispatch(tmp_path):
    path = tmp_path / "planner_budget.json"
    ledger = PlannerBudgetLedger(spend_authorized=True, hard_cap_usd=0.001)
    ledger.bind_identity(run_id="r", cache_identity="c", model=PLANNER_MODEL)
    attempt = ledger.reserve_attempt(
        input_token_bound=1,
        max_output_tokens=1,
        persist_path=path,
        expected_model=PLANNER_MODEL,
    )
    with pytest.raises(PlannerBudgetError, match="anomaly|exceeds"):
        ledger.settle_validated(
            attempt.attempt_id,
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            persist_path=path,
            response_model=PLANNER_MODEL,
        )
    assert ledger.accounting_status == "blocked"
    assert ledger.retained_exposure_usd > 0
    assert ledger.potential_additional_exposure_usd > 0
    with pytest.raises(PlannerBudgetError, match="accounting_status|blocked"):
        ledger.reserve_attempt(input_token_bound=1, max_output_tokens=1)


@pytest.mark.parametrize(
    "usage,match",
    [
        ({}, "missing"),
        ({"input_tokens": -1, "output_tokens": 1}, "nonnegative"),
        ({"input_tokens": 1.5, "output_tokens": 1}, "integer"),
        ({"input_tokens": "1", "output_tokens": 1}, "integer"),
        (
            {
                "input_tokens": 10,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 11},
            },
            "cached_tokens",
        ),
        (
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 6},
            },
            "reasoning_tokens",
        ),
    ],
)
def test_malformed_usage_retains_exposure(tmp_path, usage, match):
    path = tmp_path / "b.json"
    ledger = _bound_ledger()
    attempt = ledger.reserve_attempt(
        input_token_bound=10, max_output_tokens=5, persist_path=path
    )
    with pytest.raises(PlannerBudgetError, match=match):
        ledger.settle_validated(attempt.attempt_id, usage, persist_path=path)
    assert ledger.accounting_status == "blocked"
    assert any(a.status == "retained_exposure" for a in ledger.attempts)


def test_model_tier_mismatch(tmp_path):
    path = tmp_path / "b.json"
    ledger = _bound_ledger()
    attempt = ledger.reserve_attempt(
        input_token_bound=10,
        max_output_tokens=5,
        persist_path=path,
        expected_model=PLANNER_MODEL,
        expected_service_tier="standard",
    )
    with pytest.raises(PlannerBudgetError, match="model"):
        ledger.settle_validated(
            attempt.attempt_id,
            {"input_tokens": 1, "output_tokens": 1},
            persist_path=path,
            response_model="other-model",
        )
    ledger2 = _bound_ledger()
    attempt2 = ledger2.reserve_attempt(
        input_token_bound=10,
        max_output_tokens=5,
        expected_model=PLANNER_MODEL,
        expected_service_tier="standard",
    )
    with pytest.raises(PlannerBudgetError, match="service_tier"):
        ledger2.settle_validated(
            attempt2.attempt_id,
            {"input_tokens": 1, "output_tokens": 1},
            response_model=PLANNER_MODEL,
            service_tier="priority",
        )


def test_reconciliation_required_survives_restart(tmp_path):
    path = tmp_path / "planner_budget.json"
    ledger = PlannerBudgetLedger(spend_authorized=True, hard_cap_usd=0.001)
    ledger.bind_identity(run_id="r1", cache_identity="c1", model=PLANNER_MODEL)
    attempt = ledger.reserve_attempt(input_token_bound=1, max_output_tokens=1, persist_path=path)
    with pytest.raises(PlannerBudgetError):
        ledger.settle_validated(
            attempt.attempt_id,
            {"input_tokens": 1000, "output_tokens": 1000},
            persist_path=path,
            response_model=PLANNER_MODEL,
        )
    restored = PlannerBudgetLedger.load(path)
    assert restored.ledger_schema_version == LEDGER_SCHEMA_VERSION
    assert restored.accounting_status == "blocked"
    with pytest.raises(PlannerBudgetError, match="blocked|accounting"):
        restored.ensure_can_dispatch()


def test_stale_checkpoint_cannot_restore_capacity(tmp_path):
    """File has settlement; stale checkpoint has empty budget — file wins."""
    file_ledger = _bound_ledger(hard_cap_usd=1.0)
    a = file_ledger.reserve_attempt(input_token_bound=100, max_output_tokens=10)
    file_ledger.settle_validated(
        a.attempt_id,
        {"input_tokens": 50, "output_tokens": 5},
        response_model=PLANNER_MODEL,
        service_tier="standard",
    )
    charged = file_ledger.charged_usd
    stale = PlannerBudgetLedger(
        spend_authorized=True,
        hard_cap_usd=1.0,
        run_id=file_ledger.run_id,
        cache_identity=file_ledger.cache_identity,
        model=PLANNER_MODEL,
    )
    reconciled = reconcile_ledgers(file_ledger, stale)
    assert reconciled.charged_usd == pytest.approx(charged)
    assert reconciled.remaining_cap_usd < 1.0


def test_reserve_before_dispatch_crash_retains_on_resume(tmp_path, chemical_only_model):
    out = tmp_path / "crash_reserve"
    # Seed a run so cache_identity / checkpoint exist (same limits on resume).
    interrupted = auto_annotate_model(
        str(chemical_only_model),
        backend=BudgetMock(),
        output_dir=out,
        interrupt_after_actions=1,
        limits=_SPEND_LIMITS,
    )
    assert interrupted.report.status == "interrupted"
    budget_path = out / "planner_budget.json"
    file_ledger = PlannerBudgetLedger.load(budget_path)
    # Simulate reservation persisted after last checkpoint (file ahead of ckpt).
    file_ledger.apply_spend_limits(spend_authorized=True, hard_cap_usd=1.0, max_attempts=12)
    attempt = file_ledger.reserve_attempt(
        input_token_bound=100, max_output_tokens=10, persist_path=budget_path
    )
    assert attempt.status == "reserved"

    resumed_backend = BudgetMock()
    resumed = auto_annotate_model(
        str(chemical_only_model),
        backend=resumed_backend,
        output_dir=out,
        resume_from=out,
        limits=_SPEND_LIMITS,
    )
    assert resumed.report.status in {"completed", "partial"}
    # Orphan reservation retained as exposure — not freed as capacity.
    assert resumed_backend.budget.retained_exposure_usd > 0
    assert any(
        a.attempt_id == attempt.attempt_id and a.status == "retained_exposure"
        for a in resumed_backend.budget.attempts
    )


def test_settle_before_checkpoint_crash_keeps_charge(tmp_path, chemical_only_model):
    out = tmp_path / "crash_settle"
    auto_annotate_model(
        str(chemical_only_model),
        backend=BudgetMock(),
        output_dir=out,
        interrupt_after_actions=1,
        limits=_SPEND_LIMITS,
    )
    budget_path = out / "planner_budget.json"
    file_ledger = PlannerBudgetLedger.load(budget_path)
    file_ledger.apply_spend_limits(spend_authorized=True, hard_cap_usd=1.0)
    attempt = file_ledger.reserve_attempt(
        input_token_bound=100, max_output_tokens=10, persist_path=budget_path
    )
    file_ledger.settle_validated(
        attempt.attempt_id,
        {"input_tokens": 40, "output_tokens": 4},
        persist_path=budget_path,
        response_model=PLANNER_MODEL,
        service_tier="standard",
    )
    charged = file_ledger.charged_usd
    # Checkpoint still stale (no charged attempt).
    resumed_backend = BudgetMock()
    auto_annotate_model(
        str(chemical_only_model),
        backend=resumed_backend,
        output_dir=out,
        resume_from=out,
        limits=_SPEND_LIMITS,
    )
    assert resumed_backend.budget.charged_usd == pytest.approx(charged)


def test_configure_budget_does_not_wipe_file_during_resume_init(
    tmp_path, chemical_only_model, monkeypatch
):
    out = tmp_path / "no_wipe"
    auto_annotate_model(
        str(chemical_only_model),
        backend=BudgetMock(),
        output_dir=out,
        interrupt_after_actions=1,
        limits=_SPEND_LIMITS,
    )
    budget_path = out / "planner_budget.json"
    ledger = PlannerBudgetLedger.load(budget_path)
    ledger.apply_spend_limits(spend_authorized=True, hard_cap_usd=1.0)
    attempt = ledger.reserve_attempt(
        input_token_bound=80, max_output_tokens=8, persist_path=budget_path
    )
    writes: list[dict] = []
    original_persist = PlannerBudgetLedger.persist

    def tracking_persist(self, path):
        writes.append(self.to_dict())
        return original_persist(self, path)

    monkeypatch.setattr(PlannerBudgetLedger, "persist", tracking_persist)
    resumed_backend = BudgetMock()
    auto_annotate_model(
        str(chemical_only_model),
        backend=resumed_backend,
        output_dir=out,
        resume_from=out,
        limits=_SPEND_LIMITS,
    )
    # First persist during resume must not be an empty wipe (attempt retained).
    assert any(
        any(a.get("attempt_id") == attempt.attempt_id for a in (w.get("attempts") or []))
        for w in writes
    )
    assert any(
        a.attempt_id == attempt.attempt_id and a.status == "retained_exposure"
        for a in resumed_backend.budget.attempts
    )


def test_identity_mismatch_fail_closed(tmp_path):
    a = _bound_ledger()
    b = _bound_ledger()
    b.cache_identity = "other"
    b.run_id = a.run_id
    with pytest.raises(PlannerBudgetError, match="cache_identity"):
        reconcile_ledgers(a, b)


def test_default_zero_paid_dispatch():
    ledger = PlannerBudgetLedger()
    with pytest.raises(PlannerBudgetError, match="zero paid"):
        ledger.reserve_attempt(input_token_bound=10, max_output_tokens=1)


def test_prev2_ledger_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "spend_authorized": True,
                "hard_cap_usd": 1.0,
                "charged_usd": 0.1,
                "attempts": [
                    {
                        "attempt_id": "x",
                        "status": "settled",
                        "input_token_bound": 1,
                        "max_output_tokens": 1,
                        "reserved_usd": 0.1,
                        "charged_usd": 0.1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PlannerBudgetError, match="ledger_schema_version"):
        PlannerBudgetLedger.load(path)
