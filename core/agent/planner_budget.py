"""Persistent planner-only spend ledger for OpenAI orchestration.

Biological/local MetaboLLM inference is not billed through this ledger.
Default authorization permits zero paid dispatches.

``planner_budget.json`` is the authoritative attempt store (single writer:
ledger ``persist()`` via atomic replace). Checkpoint embeds a summary for
cross-check. Post-response validation cannot undo provider charges already
incurred; anomalies retain conservative exposure and block further paid
dispatches until reconciled.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from core.agent.io import atomic_write_json

# Official gpt-6-luna Standard short-context rates (USD / 1M tokens).
# Checked 2026-11-12 from https://developers.openai.com/api/docs/models/gpt-6-luna
# Service-tier naming: OpenAI Responses API uses ``default`` for Standard pricing
# (see Responses create docs / openai.types.responses.Response.service_tier).
# Pricing page labels this "Standard"; API response identity is ``default``.
PRICING_CHECKED = "2026-11-12"
PRICING_IDENTITY_CHECKED = "2026-11-12"
PLANNER_SETTLEMENT_MODEL = "gpt-6-luna"
# Requested / reserved billable tier for Standard-rate reservations.
PLANNER_REQUESTED_SERVICE_TIER = "default"
GPT6_LUNA_STANDARD = {
    "model": PLANNER_SETTLEMENT_MODEL,
    "service_tier": PLANNER_REQUESTED_SERVICE_TIER,
    "input_per_mtok": 0.10,
    "cached_input_per_mtok": 0.01,
    "cache_write_per_mtok": 0.125,
    "output_per_mtok": 0.50,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "long_context_input_threshold": 272_000,
    "include_cache_write_premium": False,
}

# Explicit settlement identity policy for Standard-priced gpt-6-luna planner.
# Response model must match the reserved model exactly. Documented snapshot
# alias list for Luna is only ``gpt-6-luna`` (model page, 2026-11-12).
ACCEPTED_RESPONSE_MODELS_FOR_REQUESTED: dict[str, frozenset[str]] = {
    PLANNER_SETTLEMENT_MODEL: frozenset({PLANNER_SETTLEMENT_MODEL}),
}
# For Standard pricing we accept only response service_tier ``default``.
# Documented OpenAI request→response equivalences that are NOT accepted here
# (different rates): request ``fast``/``priority`` → response ``priority``
# (Fast, 2x); ``flex`` / ``ultrafast`` / ``scale`` are other tiers.
# ``auto`` is a request default, not an accepted settlement response identity
# for Standard-rate reservations.
ACCEPTED_RESPONSE_TIERS_FOR_REQUESTED: dict[str, frozenset[str]] = {
    PLANNER_REQUESTED_SERVICE_TIER: frozenset({PLANNER_REQUESTED_SERVICE_TIER}),
}
KNOWN_RESPONSE_SERVICE_TIERS = frozenset(
    {"auto", "default", "flex", "scale", "priority", "fast", "ultrafast"}
)

LEDGER_SCHEMA_VERSION = 2
AccountingStatus = Literal["ok", "reconciliation_required", "blocked"]

_STATUS_RANK = {
    "blocked": 0,
    "reserved": 1,
    "retained_exposure": 2,
    "settled": 3,
}


class PlannerBudgetError(RuntimeError):
    """Spend authorization, reservation, settlement, or reconcile failure."""


def _normalize_identity_token(value: Any) -> Optional[str]:
    """Return stripped identity string, or None if missing/blank."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    return text if text else None


def verify_settlement_response_identity(
    *,
    expected_model: str,
    expected_service_tier: str,
    response_model: Any,
    response_service_tier: Any,
) -> tuple[str, str]:
    """Require verified response model + billing tier before releasing a reservation.

    Preserves requested vs actual separately: callers must pass the raw response
    fields without substituting request defaults for missing values.

    Returns normalized (response_model, response_service_tier) on success.
    """
    actual_model = _normalize_identity_token(response_model)
    actual_tier = _normalize_identity_token(response_service_tier)
    expected_model_n = _normalize_identity_token(expected_model)
    expected_tier_n = _normalize_identity_token(expected_service_tier)
    if not expected_model_n or not expected_tier_n:
        raise PlannerBudgetError(
            "reservation missing expected_model/expected_service_tier; cannot settle"
        )
    if actual_model is None and actual_tier is None:
        raise PlannerBudgetError(
            "settlement identity missing: response model and service_tier are absent; "
            "validated usage alone cannot settle an identity-uncertain attempt"
        )
    if actual_model is None:
        raise PlannerBudgetError(
            "settlement identity missing: response model is absent/blank; "
            "validated usage alone cannot settle an identity-uncertain attempt"
        )
    if actual_tier is None:
        raise PlannerBudgetError(
            "settlement identity missing: response service_tier is absent/blank; "
            "validated usage alone cannot settle an identity-uncertain attempt"
        )

    accepted_models = ACCEPTED_RESPONSE_MODELS_FOR_REQUESTED.get(expected_model_n)
    if accepted_models is None:
        raise PlannerBudgetError(
            f"unrecognized reserved expected_model={expected_model_n!r}; "
            "no settlement compatibility policy"
        )
    if actual_model not in accepted_models:
        raise PlannerBudgetError(
            f"response model={actual_model!r} incompatible with reserved "
            f"expected_model={expected_model_n!r} "
            f"(accepted={sorted(accepted_models)})"
        )

    accepted_tiers = ACCEPTED_RESPONSE_TIERS_FOR_REQUESTED.get(expected_tier_n)
    if accepted_tiers is None:
        raise PlannerBudgetError(
            f"unrecognized reserved expected_service_tier={expected_tier_n!r}; "
            "no settlement compatibility policy"
        )
    if actual_tier not in KNOWN_RESPONSE_SERVICE_TIERS:
        raise PlannerBudgetError(
            f"unrecognized response service_tier={actual_tier!r}; "
            f"known={sorted(KNOWN_RESPONSE_SERVICE_TIERS)}"
        )
    if actual_tier not in accepted_tiers:
        raise PlannerBudgetError(
            f"response service_tier={actual_tier!r} incompatible with reserved "
            f"expected_service_tier={expected_tier_n!r} "
            f"(accepted={sorted(accepted_tiers)}; "
            "fast/priority/flex/ultrafast are not Standard-rate equivalents)"
        )
    return actual_model, actual_tier


def hard_input_token_bound(serialized_request: str) -> int:
    """Complete per-request input-token upper bound before dispatch.

    Uses UTF-8 byte length as a documented hard upper bound for BPE tokenizers
    on JSON/text planner payloads (bytes >= typical token count for these
    payloads). Soft estimates alone must not authorize spend.
    """
    if not serialized_request:
        raise PlannerBudgetError("cannot establish input bound: empty request payload")
    return max(1, len(serialized_request.encode("utf-8")))


def _ceil_usd(value: float) -> float:
    return math.ceil(float(value) * 1_000_000.0) / 1_000_000.0


def conservative_attempt_cost_usd(
    *,
    input_token_bound: int,
    max_output_tokens: int,
    pricing: Optional[dict[str, Any]] = None,
    assume_uncached: bool = True,
) -> float:
    """Worst-case USD for one attempt under conservative Standard pricing."""
    p = dict(pricing or GPT6_LUNA_STANDARD)
    in_rate = float(p["input_per_mtok"])
    out_rate = float(p["output_per_mtok"])
    # Never use cache-read discount for reservations.
    _ = assume_uncached  # documented: reservations always uncached
    threshold = int(p.get("long_context_input_threshold") or 272_000)
    if input_token_bound > threshold:
        in_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        out_rate *= float(p.get("long_context_output_multiplier") or 1.5)
    write_extra = 0.0
    if p.get("include_cache_write_premium"):
        write_rate = float(p.get("cache_write_per_mtok") or 0.0)
        if input_token_bound > threshold:
            write_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        write_extra = (input_token_bound / 1_000_000.0) * write_rate
    cost = (input_token_bound / 1_000_000.0) * in_rate
    cost += (max(0, int(max_output_tokens)) / 1_000_000.0) * out_rate
    cost += write_extra
    return float(cost)


def _require_nonneg_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerBudgetError(
            f"usage field {field_name} must be a nonnegative integer (got {value!r})"
        )
    if value < 0:
        raise PlannerBudgetError(f"usage field {field_name} must be nonnegative")
    return value


def parse_strict_usage(usage: Any) -> dict[str, Any]:
    """Parse provider usage without silent coercion or clamping.

    Returns a normalized dict with integer input_tokens, output_tokens,
    optional cached_tokens / reasoning_tokens, and raw evidence preserved.
    """
    if not isinstance(usage, dict) or not usage:
        raise PlannerBudgetError("usage missing or not an object")
    raw = dict(usage)
    input_tokens = raw.get("input_tokens")
    if input_tokens is None:
        input_tokens = raw.get("prompt_tokens")
    output_tokens = raw.get("output_tokens")
    if output_tokens is None:
        output_tokens = raw.get("completion_tokens")
    if input_tokens is None or output_tokens is None:
        raise PlannerBudgetError("usage missing input_tokens/output_tokens")
    input_tokens = _require_nonneg_int(input_tokens, "input_tokens")
    output_tokens = _require_nonneg_int(output_tokens, "output_tokens")

    cached_tokens = 0
    details = raw.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise PlannerBudgetError("input_tokens_details must be an object")
        if "cached_tokens" in details and details["cached_tokens"] is not None:
            cached_tokens = _require_nonneg_int(details["cached_tokens"], "cached_tokens")
            if cached_tokens > input_tokens:
                raise PlannerBudgetError(
                    f"cached_tokens={cached_tokens} exceeds input_tokens={input_tokens}"
                )

    reasoning_tokens = 0
    out_details = raw.get("output_tokens_details")
    if out_details is not None:
        if not isinstance(out_details, dict):
            raise PlannerBudgetError("output_tokens_details must be an object")
        if "reasoning_tokens" in out_details and out_details["reasoning_tokens"] is not None:
            reasoning_tokens = _require_nonneg_int(
                out_details["reasoning_tokens"], "reasoning_tokens"
            )
            if reasoning_tokens > output_tokens:
                raise PlannerBudgetError(
                    f"reasoning_tokens={reasoning_tokens} exceeds output_tokens={output_tokens}"
                )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "raw": raw,
    }


def validated_usage_cost_usd(
    usage: dict[str, Any],
    *,
    pricing: Optional[dict[str, Any]] = None,
) -> float:
    """Token-derived USD estimate from already-strict-parsed usage fields.

    Distinct from invoice verification — estimates only.
    Accepts either parse_strict_usage output or raw usage (re-parsed).
    """
    if "raw" in usage and "input_tokens" in usage:
        parsed = usage
    else:
        parsed = parse_strict_usage(usage)
    p = dict(pricing or GPT6_LUNA_STANDARD)
    input_tokens = int(parsed["input_tokens"])
    output_tokens = int(parsed["output_tokens"])
    cached_tokens = int(parsed.get("cached_tokens") or 0)
    uncached = input_tokens - cached_tokens
    in_rate = float(p["input_per_mtok"])
    cached_rate = float(p["cached_input_per_mtok"])
    out_rate = float(p["output_per_mtok"])
    threshold = int(p.get("long_context_input_threshold") or 272_000)
    if input_tokens > threshold:
        in_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        cached_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        out_rate *= float(p.get("long_context_output_multiplier") or 1.5)
    cost = (uncached / 1_000_000.0) * in_rate
    cost += (cached_tokens / 1_000_000.0) * cached_rate
    cost += (output_tokens / 1_000_000.0) * out_rate
    if p.get("include_cache_write_premium"):
        write_rate = float(p.get("cache_write_per_mtok") or 0.0)
        if input_tokens > threshold:
            write_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        # Settlement: only charge write premium on uncached tokens when enabled.
        cost += (uncached / 1_000_000.0) * write_rate
    return float(cost)


@dataclass
class PlannerAttemptRecord:
    attempt_id: str
    status: str  # reserved | settled | retained_exposure | blocked
    input_token_bound: int
    max_output_tokens: int
    reserved_usd: float
    charged_usd: float = 0.0
    usage: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    settled_at: Optional[float] = None
    potential_additional_exposure_usd: float = 0.0
    # Requested / reserved identity (never overwritten with response values).
    expected_model: str = PLANNER_SETTLEMENT_MODEL
    expected_service_tier: str = PLANNER_REQUESTED_SERVICE_TIER
    # Actual response identity as returned (may remain None if absent).
    actual_response_model: Optional[str] = None
    actual_service_tier: Optional[str] = None
    # Safe diagnostic metadata (no credentials).
    response_identity_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerBudgetLedger:
    """Durable planner-only spend state (ledger schema v2)."""

    spend_authorized: bool = False
    hard_cap_usd: float = 0.0
    charged_usd: float = 0.0
    reserved_usd: float = 0.0
    retained_exposure_usd: float = 0.0
    potential_additional_exposure_usd: float = 0.0
    attempts: list[PlannerAttemptRecord] = field(default_factory=list)
    pricing: dict[str, Any] = field(default_factory=lambda: dict(GPT6_LUNA_STANDARD))
    pricing_checked: str = PRICING_CHECKED
    model: str = "gpt-6-luna"
    max_attempts: int = 12
    ledger_schema_version: int = LEDGER_SCHEMA_VERSION
    run_id: str = ""
    cache_identity: str = ""
    accounting_status: AccountingStatus = "ok"
    anomaly_notes: list[str] = field(default_factory=list)
    estimate_not_invoice: bool = True

    @property
    def remaining_cap_usd(self) -> float:
        exposure = (
            self.reserved_usd
            + self.retained_exposure_usd
            + self.potential_additional_exposure_usd
        )
        return float(self.hard_cap_usd) - float(self.charged_usd) - float(exposure)

    def summary(self) -> dict[str, Any]:
        return {
            "ledger_schema_version": self.ledger_schema_version,
            "run_id": self.run_id,
            "cache_identity": self.cache_identity,
            "accounting_status": self.accounting_status,
            "anomaly_notes": list(self.anomaly_notes),
            "estimate_not_invoice": self.estimate_not_invoice,
            "spend_authorized": self.spend_authorized,
            "hard_cap_usd": self.hard_cap_usd,
            "charged_usd": self.charged_usd,
            "reserved_usd": self.reserved_usd,
            "retained_exposure_usd": self.retained_exposure_usd,
            "potential_additional_exposure_usd": self.potential_additional_exposure_usd,
            "remaining_cap_usd": self.remaining_cap_usd,
            "attempt_count": len(self.attempts),
            "model": self.model,
            "pricing_checked": self.pricing_checked,
            "pricing": dict(self.pricing),
            "max_attempts": self.max_attempts,
            "attempts": [asdict(a) for a in self.attempts],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannerBudgetLedger":
        if not isinstance(data, dict):
            raise PlannerBudgetError("planner budget ledger root must be an object")
        version = data.get("ledger_schema_version")
        if version is None and not data.get("attempts") and not data.get("charged_usd"):
            # Empty legacy/empty checkpoint budget → upgrade to v2 empty ledger.
            version = LEDGER_SCHEMA_VERSION
        if version is None:
            raise PlannerBudgetError(
                "unsupported planner budget ledger: missing ledger_schema_version "
                "(pre-v2 ledgers cannot be resumed; restart without resume_from or "
                "with a fresh output_dir)"
            )
        if int(version) != LEDGER_SCHEMA_VERSION:
            raise PlannerBudgetError(
                f"unsupported planner budget ledger_schema_version={version!r}; "
                f"only {LEDGER_SCHEMA_VERSION} is supported"
            )
        attempts = []
        for raw in data.get("attempts") or []:
            attempts.append(
                PlannerAttemptRecord(
                    attempt_id=str(raw["attempt_id"]),
                    status=str(raw["status"]),
                    input_token_bound=int(raw["input_token_bound"]),
                    max_output_tokens=int(raw["max_output_tokens"]),
                    reserved_usd=float(raw["reserved_usd"]),
                    charged_usd=float(raw.get("charged_usd") or 0.0),
                    usage=raw.get("usage"),
                    error=raw.get("error"),
                    created_at=float(raw.get("created_at") or time.time()),
                    settled_at=raw.get("settled_at"),
                    potential_additional_exposure_usd=float(
                        raw.get("potential_additional_exposure_usd") or 0.0
                    ),
                    expected_model=str(
                        raw.get("expected_model")
                        or data.get("model")
                        or PLANNER_SETTLEMENT_MODEL
                    ),
                    expected_service_tier=str(
                        raw.get("expected_service_tier")
                        or (data.get("pricing") or {}).get("service_tier")
                        or PLANNER_REQUESTED_SERVICE_TIER
                    ),
                    actual_response_model=raw.get("actual_response_model"),
                    actual_service_tier=raw.get("actual_service_tier"),
                    response_identity_meta=dict(raw.get("response_identity_meta") or {}),
                )
            )
        ledger = cls(
            spend_authorized=bool(data.get("spend_authorized", False)),
            hard_cap_usd=float(data.get("hard_cap_usd") or 0.0),
            charged_usd=float(data.get("charged_usd") or 0.0),
            reserved_usd=float(data.get("reserved_usd") or 0.0),
            retained_exposure_usd=float(data.get("retained_exposure_usd") or 0.0),
            potential_additional_exposure_usd=float(
                data.get("potential_additional_exposure_usd") or 0.0
            ),
            attempts=attempts,
            pricing=dict(data.get("pricing") or GPT6_LUNA_STANDARD),
            pricing_checked=str(data.get("pricing_checked") or PRICING_CHECKED),
            model=str(data.get("model") or "gpt-6-luna"),
            max_attempts=int(data.get("max_attempts") or 12),
            ledger_schema_version=LEDGER_SCHEMA_VERSION,
            run_id=str(data.get("run_id") or ""),
            cache_identity=str(data.get("cache_identity") or ""),
            accounting_status=str(data.get("accounting_status") or "ok"),  # type: ignore[arg-type]
            anomaly_notes=list(data.get("anomaly_notes") or []),
            estimate_not_invoice=bool(data.get("estimate_not_invoice", True)),
        )
        if ledger.accounting_status not in {"ok", "reconciliation_required", "blocked"}:
            raise PlannerBudgetError(
                f"invalid accounting_status={ledger.accounting_status!r}"
            )
        ledger.recompute_totals()
        return ledger

    def recompute_totals(self) -> None:
        """Recompute aggregates from attempts (never double-count)."""
        charged = 0.0
        reserved = 0.0
        retained = 0.0
        extra = 0.0
        for a in self.attempts:
            if a.status == "settled":
                charged += float(a.charged_usd)
                extra += float(a.potential_additional_exposure_usd or 0.0)
            elif a.status == "reserved":
                reserved += float(a.reserved_usd)
            elif a.status == "retained_exposure":
                retained += float(a.reserved_usd)
                extra += float(a.potential_additional_exposure_usd or 0.0)
            elif a.status == "blocked":
                extra += float(a.potential_additional_exposure_usd or 0.0)
        self.charged_usd = float(charged)
        self.reserved_usd = float(reserved)
        self.retained_exposure_usd = float(retained)
        self.potential_additional_exposure_usd = float(extra)

    def bind_identity(
        self,
        *,
        run_id: str,
        cache_identity: str,
        model: str,
        pricing: Optional[dict[str, Any]] = None,
        pricing_checked: Optional[str] = None,
    ) -> None:
        self.run_id = str(run_id)
        self.cache_identity = str(cache_identity)
        self.model = str(model)
        if pricing is not None:
            self.pricing = dict(pricing)
        if pricing_checked is not None:
            self.pricing_checked = str(pricing_checked)
        self.ledger_schema_version = LEDGER_SCHEMA_VERSION

    def validate_identity(
        self,
        *,
        cache_identity: str,
        model: str,
        pricing_checked: Optional[str] = None,
        require_bound: bool = True,
    ) -> None:
        if require_bound and (not self.cache_identity or not self.run_id):
            raise PlannerBudgetError(
                "planner budget ledger missing run_id/cache_identity; "
                "cannot accept unbound ledger for this run"
            )
        if self.cache_identity and self.cache_identity != cache_identity:
            raise PlannerBudgetError(
                "planner budget ledger cache_identity mismatch with current run"
            )
        if self.model and self.model != model:
            raise PlannerBudgetError(
                f"planner budget ledger model={self.model!r} mismatches {model!r}"
            )
        if pricing_checked is not None and self.pricing_checked != pricing_checked:
            raise PlannerBudgetError(
                "planner budget ledger pricing_checked mismatch with planner identity"
            )

    def apply_spend_limits(
        self,
        *,
        spend_authorized: bool,
        hard_cap_usd: float,
        max_attempts: Optional[int] = None,
    ) -> None:
        """Apply current operator limits without erasing prior exposure/charges."""
        self.spend_authorized = bool(spend_authorized)
        self.hard_cap_usd = float(hard_cap_usd)
        if max_attempts is not None:
            self.max_attempts = int(max_attempts)

    def mark_anomaly(self, note: str, *, block: bool = True) -> None:
        self.anomaly_notes.append(note)
        self.accounting_status = "blocked" if block else "reconciliation_required"

    def ensure_can_dispatch(self) -> None:
        if self.accounting_status in {"reconciliation_required", "blocked"}:
            raise PlannerBudgetError(
                f"planner accounting_status={self.accounting_status}; "
                "further paid dispatches blocked until reconciled "
                "(post-response validation cannot undo charges already incurred)"
            )
        if not self.spend_authorized or float(self.hard_cap_usd) <= 0.0:
            raise PlannerBudgetError(
                "planner spend not authorized or hard cap is $0; zero paid dispatches permitted"
            )
        if len(self.attempts) >= int(self.max_attempts):
            raise PlannerBudgetError(
                f"planner max_attempts={self.max_attempts} exhausted"
            )

    def reserve_attempt(
        self,
        *,
        input_token_bound: int,
        max_output_tokens: int,
        persist_path: Optional[Path] = None,
        expected_model: Optional[str] = None,
        expected_service_tier: Optional[str] = None,
    ) -> PlannerAttemptRecord:
        self.ensure_can_dispatch()
        if input_token_bound <= 0:
            raise PlannerBudgetError("input_token_bound must be positive")
        reserved = conservative_attempt_cost_usd(
            input_token_bound=input_token_bound,
            max_output_tokens=max_output_tokens,
            pricing=self.pricing,
            assume_uncached=True,
        )
        reserved = _ceil_usd(reserved)
        if reserved > self.remaining_cap_usd + 1e-12:
            raise PlannerBudgetError(
                f"reservation ${reserved:.6f} exceeds remaining cap "
                f"${self.remaining_cap_usd:.6f}"
            )
        tier = expected_service_tier or str(
            self.pricing.get("service_tier") or PLANNER_REQUESTED_SERVICE_TIER
        )
        attempt = PlannerAttemptRecord(
            attempt_id=str(uuid.uuid4()),
            status="reserved",
            input_token_bound=int(input_token_bound),
            max_output_tokens=int(max_output_tokens),
            reserved_usd=float(reserved),
            expected_model=str(expected_model or self.model or PLANNER_SETTLEMENT_MODEL),
            expected_service_tier=str(tier),
        )
        self.attempts.append(attempt)
        self.recompute_totals()
        if persist_path is not None:
            self.persist(persist_path)
        return attempt

    def settle_validated(
        self,
        attempt_id: str,
        usage: dict[str, Any],
        *,
        persist_path: Optional[Path] = None,
        response_model: Any = None,
        service_tier: Any = None,
        response_identity_meta: Optional[dict[str, Any]] = None,
    ) -> PlannerAttemptRecord:
        attempt = self._get(attempt_id)
        if attempt.status not in {"reserved", "retained_exposure"}:
            raise PlannerBudgetError(f"attempt {attempt_id} not open for settlement")

        # Record raw response identity as provided — never fill from request defaults.
        attempt.actual_response_model = _normalize_identity_token(response_model)
        attempt.actual_service_tier = _normalize_identity_token(service_tier)
        meta = dict(response_identity_meta or {})
        meta.setdefault("raw_response_model", response_model)
        meta.setdefault("raw_response_service_tier", service_tier)
        meta.setdefault("expected_model", attempt.expected_model)
        meta.setdefault("expected_service_tier", attempt.expected_service_tier)
        # Strip anything that looks credential-like if a caller passed extras.
        for bad in ("api_key", "authorization", "Authorization", "token", "secret"):
            meta.pop(bad, None)
        attempt.response_identity_meta = meta

        try:
            verified_model, verified_tier = verify_settlement_response_identity(
                expected_model=attempt.expected_model,
                expected_service_tier=attempt.expected_service_tier,
                response_model=response_model,
                response_service_tier=service_tier,
            )
            attempt.actual_response_model = verified_model
            attempt.actual_service_tier = verified_tier
            parsed = parse_strict_usage(usage)
            if parsed["input_tokens"] > attempt.input_token_bound:
                raise PlannerBudgetError(
                    f"input_tokens={parsed['input_tokens']} exceeds reserved "
                    f"input_token_bound={attempt.input_token_bound}"
                )
            billable_output = int(parsed["output_tokens"])
            if billable_output > attempt.max_output_tokens:
                raise PlannerBudgetError(
                    f"output_tokens={billable_output} exceeds reserved "
                    f"max_output_tokens={attempt.max_output_tokens}"
                )
            charged = validated_usage_cost_usd(parsed, pricing=self.pricing)
            charged = _ceil_usd(charged)
            if charged > float(attempt.reserved_usd) + 1e-12:
                raise PlannerBudgetError(
                    f"token-derived liability ${charged:.6f} exceeds reservation "
                    f"${attempt.reserved_usd:.6f}"
                )
        except PlannerBudgetError as exc:
            # Preserve evidence; retain conservative exposure; record any overage.
            attempt.usage = dict(usage) if isinstance(usage, dict) else {"raw": usage}
            overage = 0.0
            try:
                if isinstance(usage, dict) and usage:
                    maybe = validated_usage_cost_usd(usage, pricing=self.pricing)
                    maybe = _ceil_usd(maybe)
                    if maybe > float(attempt.reserved_usd):
                        overage = maybe - float(attempt.reserved_usd)
            except PlannerBudgetError:
                overage = 0.0
            attempt.potential_additional_exposure_usd = float(
                attempt.potential_additional_exposure_usd
            ) + float(overage)
            note = (
                f"settlement anomaly attempt_id={attempt_id}: {exc}; "
                "post-response validation cannot undo charges already incurred"
            )
            if attempt.status == "reserved":
                self.retain_exposure(attempt_id, error=str(exc), persist_path=None)
            else:
                attempt.error = str(exc)
            self.mark_anomaly(note, block=True)
            self.recompute_totals()
            if persist_path is not None:
                self.persist(persist_path)
            raise PlannerBudgetError(note) from exc

        # Success: release unused reserve only after identity + usage validation.
        attempt.usage = dict(parsed["raw"])
        attempt.charged_usd = float(charged)
        attempt.status = "settled"
        attempt.settled_at = time.time()
        attempt.error = None
        self.recompute_totals()
        if persist_path is not None:
            self.persist(persist_path)
        return attempt

    def retain_exposure(
        self,
        attempt_id: str,
        *,
        error: Optional[str] = None,
        persist_path: Optional[Path] = None,
    ) -> PlannerAttemptRecord:
        attempt = self._get(attempt_id)
        if attempt.status == "retained_exposure":
            if error:
                attempt.error = error
            if persist_path is not None:
                self.persist(persist_path)
            return attempt
        if attempt.status != "reserved":
            raise PlannerBudgetError(
                f"attempt {attempt_id} cannot retain exposure from {attempt.status}"
            )
        attempt.status = "retained_exposure"
        attempt.error = error
        attempt.settled_at = time.time()
        self.recompute_totals()
        if persist_path is not None:
            self.persist(persist_path)
        return attempt

    def retain_open_reservations(
        self,
        *,
        error: str = "orphan reservation after crash/resume",
        persist_path: Optional[Path] = None,
    ) -> list[PlannerAttemptRecord]:
        retained: list[PlannerAttemptRecord] = []
        for attempt in list(self.attempts):
            if attempt.status == "reserved":
                retained.append(
                    self.retain_exposure(attempt.attempt_id, error=error, persist_path=None)
                )
        if retained and persist_path is not None:
            self.persist(persist_path)
        return retained

    def _get(self, attempt_id: str) -> PlannerAttemptRecord:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise PlannerBudgetError(f"unknown planner attempt_id={attempt_id}")

    def persist(self, path: Path) -> None:
        """Atomic single-writer persist of the authoritative ledger file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "PlannerBudgetLedger":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PlannerBudgetError("planner budget ledger root must be an object")
        return cls.from_dict(raw)


def _attempt_conflict(a: PlannerAttemptRecord, b: PlannerAttemptRecord) -> Optional[str]:
    if a.input_token_bound != b.input_token_bound or a.max_output_tokens != b.max_output_tokens:
        return "bound mismatch"
    if abs(float(a.reserved_usd) - float(b.reserved_usd)) > 1e-9:
        return "reserved_usd mismatch"
    rank_a = _STATUS_RANK.get(a.status, -1)
    rank_b = _STATUS_RANK.get(b.status, -1)
    if rank_a == rank_b and a.status == "settled":
        if abs(float(a.charged_usd) - float(b.charged_usd)) > 1e-9:
            return "settled charged_usd conflict"
    if rank_a == rank_b and a.status == b.status and a.status != "reserved":
        # Same terminal-ish status with conflicting extra exposure
        if abs(
            float(a.potential_additional_exposure_usd)
            - float(b.potential_additional_exposure_usd)
        ) > 1e-9:
            return "potential_additional_exposure conflict"
    return None


def _prefer_attempt(
    a: PlannerAttemptRecord, b: PlannerAttemptRecord
) -> PlannerAttemptRecord:
    conflict = _attempt_conflict(a, b)
    if conflict:
        raise PlannerBudgetError(
            f"irreconcilable planner attempt {a.attempt_id}: {conflict}"
        )
    rank_a = _STATUS_RANK.get(a.status, -1)
    rank_b = _STATUS_RANK.get(b.status, -1)
    if rank_a > rank_b:
        return a
    if rank_b > rank_a:
        return b
    # Same status: prefer newer settled_at / created_at
    ta = float(a.settled_at or a.created_at or 0.0)
    tb = float(b.settled_at or b.created_at or 0.0)
    return a if ta >= tb else b


def reconcile_ledgers(
    file_ledger: Optional[PlannerBudgetLedger],
    checkpoint_ledger: Optional[PlannerBudgetLedger],
) -> PlannerBudgetLedger:
    """Union attempts by id; prefer newest valid terminal state; fail closed on conflict."""
    if file_ledger is None and checkpoint_ledger is None:
        return PlannerBudgetLedger()
    if file_ledger is None:
        out = PlannerBudgetLedger.from_dict(checkpoint_ledger.to_dict())  # type: ignore[union-attr]
        out.recompute_totals()
        return out
    if checkpoint_ledger is None:
        out = PlannerBudgetLedger.from_dict(file_ledger.to_dict())
        out.recompute_totals()
        return out

    # Identity must agree when both bound (even if one side has no attempts yet).
    if (
        file_ledger.cache_identity
        and checkpoint_ledger.cache_identity
        and file_ledger.cache_identity != checkpoint_ledger.cache_identity
    ):
        raise PlannerBudgetError(
            "ledger/checkpoint cache_identity mismatch during reconcile"
        )
    if (
        file_ledger.run_id
        and checkpoint_ledger.run_id
        and file_ledger.run_id != checkpoint_ledger.run_id
    ):
        raise PlannerBudgetError("ledger/checkpoint run_id mismatch during reconcile")
    if (
        file_ledger.model
        and checkpoint_ledger.model
        and file_ledger.model != checkpoint_ledger.model
    ):
        raise PlannerBudgetError("ledger/checkpoint model mismatch during reconcile")

    if (
        not checkpoint_ledger.attempts
        and not checkpoint_ledger.charged_usd
        and not checkpoint_ledger.retained_exposure_usd
        and not checkpoint_ledger.potential_additional_exposure_usd
    ):
        # Stale/empty checkpoint summary: authoritative file wins after identity check.
        out = PlannerBudgetLedger.from_dict(file_ledger.to_dict())
        out.recompute_totals()
        return out

    by_id: dict[str, PlannerAttemptRecord] = {}
    for src in (checkpoint_ledger.attempts, file_ledger.attempts):
        for attempt in src:
            prev = by_id.get(attempt.attempt_id)
            if prev is None:
                by_id[attempt.attempt_id] = PlannerAttemptRecord(**asdict(attempt))
            else:
                by_id[attempt.attempt_id] = _prefer_attempt(prev, attempt)

    # Prefer file metadata for identity/status when file is present (newer writer).
    base = file_ledger
    out = PlannerBudgetLedger(
        spend_authorized=base.spend_authorized,
        hard_cap_usd=base.hard_cap_usd,
        attempts=list(by_id.values()),
        pricing=dict(base.pricing),
        pricing_checked=base.pricing_checked,
        model=base.model,
        max_attempts=max(base.max_attempts, checkpoint_ledger.max_attempts),
        ledger_schema_version=LEDGER_SCHEMA_VERSION,
        run_id=base.run_id or checkpoint_ledger.run_id,
        cache_identity=base.cache_identity or checkpoint_ledger.cache_identity,
        accounting_status=base.accounting_status
        if base.accounting_status != "ok"
        else checkpoint_ledger.accounting_status,
        anomaly_notes=list(
            dict.fromkeys(list(checkpoint_ledger.anomaly_notes) + list(base.anomaly_notes))
        ),
        estimate_not_invoice=True,
    )
    # Prefer blocked/reconciliation from either side.
    if "blocked" in {base.accounting_status, checkpoint_ledger.accounting_status}:
        out.accounting_status = "blocked"
    elif "reconciliation_required" in {
        base.accounting_status,
        checkpoint_ledger.accounting_status,
    }:
        out.accounting_status = "reconciliation_required"
    out.recompute_totals()
    return out
