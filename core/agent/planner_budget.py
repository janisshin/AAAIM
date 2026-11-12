"""Persistent planner-only spend ledger for OpenAI orchestration.

Biological/local MetaboLLM inference is not billed through this ledger.
Default authorization permits zero paid dispatches.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.agent.io import atomic_write_json

# Official gpt-6-luna Standard short-context rates (USD / 1M tokens).
# Checked 2026-11-12 from https://developers.openai.com/api/docs/models/gpt-6-luna
PRICING_CHECKED = "2026-11-12"
GPT6_LUNA_STANDARD = {
    "model": "gpt-6-luna",
    "input_per_mtok": 0.10,
    "cached_input_per_mtok": 0.01,
    "cache_write_per_mtok": 0.125,
    "output_per_mtok": 0.50,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "long_context_input_threshold": 272_000,
}


class PlannerBudgetError(RuntimeError):
    """Spend authorization, reservation, or settlement failure."""


def hard_input_token_bound(serialized_request: str) -> int:
    """Complete per-request input-token upper bound before dispatch.

    Uses UTF-8 byte length as a documented hard upper bound for BPE tokenizers
    on JSON/text planner payloads (bytes >= typical token count for these
    payloads). Soft estimates alone must not authorize spend.
    """
    if not serialized_request:
        raise PlannerBudgetError("cannot establish input bound: empty request payload")
    return max(1, len(serialized_request.encode("utf-8")))


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
    if not assume_uncached:
        in_rate = float(p["input_per_mtok"])
    threshold = int(p.get("long_context_input_threshold") or 272_000)
    if input_token_bound > threshold:
        in_rate *= float(p.get("long_context_input_multiplier") or 2.0)
        out_rate *= float(p.get("long_context_output_multiplier") or 1.5)
    # Optional cache-write premium when caching is enabled: still charge full
    # uncached input; add write premium on the same bound for conservatism.
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


def validated_usage_cost_usd(
    usage: dict[str, Any],
    *,
    pricing: Optional[dict[str, Any]] = None,
) -> Optional[float]:
    """Return settled USD from validated usage, or None if usage is incomplete."""
    if not isinstance(usage, dict) or not usage:
        return None
    # Accept OpenAI-style and local meta keys.
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    try:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    p = dict(pricing or GPT6_LUNA_STANDARD)
    cached = usage.get("input_tokens_details", {})
    cached_tokens = 0
    if isinstance(cached, dict):
        try:
            cached_tokens = int(cached.get("cached_tokens") or 0)
        except (TypeError, ValueError):
            cached_tokens = 0
    cached_tokens = max(0, min(cached_tokens, input_tokens))
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


@dataclass
class PlannerBudgetLedger:
    """Durable planner-only spend state."""

    spend_authorized: bool = False
    hard_cap_usd: float = 0.0
    charged_usd: float = 0.0
    reserved_usd: float = 0.0
    retained_exposure_usd: float = 0.0
    attempts: list[PlannerAttemptRecord] = field(default_factory=list)
    pricing: dict[str, Any] = field(default_factory=lambda: dict(GPT6_LUNA_STANDARD))
    pricing_checked: str = PRICING_CHECKED
    model: str = "gpt-6-luna"
    max_attempts: int = 12

    @property
    def remaining_cap_usd(self) -> float:
        exposure = self.reserved_usd + self.retained_exposure_usd
        return float(self.hard_cap_usd) - float(self.charged_usd) - float(exposure)

    def summary(self) -> dict[str, Any]:
        return {
            "spend_authorized": self.spend_authorized,
            "hard_cap_usd": self.hard_cap_usd,
            "charged_usd": self.charged_usd,
            "reserved_usd": self.reserved_usd,
            "retained_exposure_usd": self.retained_exposure_usd,
            "remaining_cap_usd": self.remaining_cap_usd,
            "attempt_count": len(self.attempts),
            "model": self.model,
            "pricing_checked": self.pricing_checked,
            "pricing": dict(self.pricing),
            "attempts": [asdict(a) for a in self.attempts],
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannerBudgetLedger":
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
                )
            )
        return cls(
            spend_authorized=bool(data.get("spend_authorized", False)),
            hard_cap_usd=float(data.get("hard_cap_usd") or 0.0),
            charged_usd=float(data.get("charged_usd") or 0.0),
            reserved_usd=float(data.get("reserved_usd") or 0.0),
            retained_exposure_usd=float(data.get("retained_exposure_usd") or 0.0),
            attempts=attempts,
            pricing=dict(data.get("pricing") or GPT6_LUNA_STANDARD),
            pricing_checked=str(data.get("pricing_checked") or PRICING_CHECKED),
            model=str(data.get("model") or "gpt-6-luna"),
            max_attempts=int(data.get("max_attempts") or 12),
        )

    def ensure_can_dispatch(self) -> None:
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
        # Round up to micros for stable comparisons.
        reserved = math.ceil(reserved * 1_000_000.0) / 1_000_000.0
        if reserved > self.remaining_cap_usd + 1e-12:
            raise PlannerBudgetError(
                f"reservation ${reserved:.6f} exceeds remaining cap "
                f"${self.remaining_cap_usd:.6f}"
            )
        attempt = PlannerAttemptRecord(
            attempt_id=str(uuid.uuid4()),
            status="reserved",
            input_token_bound=int(input_token_bound),
            max_output_tokens=int(max_output_tokens),
            reserved_usd=float(reserved),
        )
        self.attempts.append(attempt)
        self.reserved_usd = float(self.reserved_usd) + float(reserved)
        if persist_path is not None:
            self.persist(persist_path)
        return attempt

    def settle_validated(
        self,
        attempt_id: str,
        usage: dict[str, Any],
        *,
        persist_path: Optional[Path] = None,
    ) -> PlannerAttemptRecord:
        attempt = self._get(attempt_id)
        if attempt.status not in {"reserved", "retained_exposure"}:
            raise PlannerBudgetError(f"attempt {attempt_id} not open for settlement")
        charged = validated_usage_cost_usd(usage, pricing=self.pricing)
        if charged is None:
            # Missing/uncertain usage retains exposure.
            self.retain_exposure(attempt_id, error="usage missing or invalid", persist_path=persist_path)
            raise PlannerBudgetError(
                "cannot settle without validated usage; reservation retained as exposure"
            )
        charged = math.ceil(float(charged) * 1_000_000.0) / 1_000_000.0
        if attempt.status == "reserved":
            self.reserved_usd = max(0.0, float(self.reserved_usd) - float(attempt.reserved_usd))
        elif attempt.status == "retained_exposure":
            self.retained_exposure_usd = max(
                0.0, float(self.retained_exposure_usd) - float(attempt.reserved_usd)
            )
        self.charged_usd = float(self.charged_usd) + float(charged)
        attempt.charged_usd = float(charged)
        attempt.usage = dict(usage)
        attempt.status = "settled"
        attempt.settled_at = time.time()
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
            raise PlannerBudgetError(f"attempt {attempt_id} cannot retain exposure from {attempt.status}")
        self.reserved_usd = max(0.0, float(self.reserved_usd) - float(attempt.reserved_usd))
        self.retained_exposure_usd = float(self.retained_exposure_usd) + float(attempt.reserved_usd)
        attempt.status = "retained_exposure"
        attempt.error = error
        attempt.settled_at = time.time()
        if persist_path is not None:
            self.persist(persist_path)
        return attempt

    def _get(self, attempt_id: str) -> PlannerAttemptRecord:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise PlannerBudgetError(f"unknown planner attempt_id={attempt_id}")

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "PlannerBudgetLedger":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PlannerBudgetError("planner budget ledger root must be an object")
        return cls.from_dict(raw)
