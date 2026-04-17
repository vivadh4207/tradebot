"""Pydantic schema for `config/settings.yaml`.

Validates types + reasonable bounds at startup. A typo like
`kelly_fraction_cap: 2.5` (should be 0.25) would run fine without this
and potentially destroy capital. Here it fails fast with a clear error.

Usage:
    schema = validate_settings(settings_dict)
    # raises pydantic.ValidationError if anything is out of range

Optional dependency: if pydantic isn't installed we fall through to a
manual bounds check so the bot still starts.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    BaseModel = object       # type: ignore
    ValidationError = Exception  # type: ignore


if _HAS_PYDANTIC:

    class AccountSettings(BaseModel):
        paper_starting_equity: float = Field(gt=0)
        max_risk_per_trade_pct: float = Field(ge=0.0, le=0.10)
        max_daily_loss_pct: float = Field(ge=0.0, le=0.20)
        max_open_positions: int = Field(ge=1, le=100)

    class SessionSettings(BaseModel):
        market_open: str
        market_close: str
        no_new_entries_after: str
        eod_force_close: str

        @field_validator("market_open", "market_close",
                          "no_new_entries_after", "eod_force_close")
        @classmethod
        def _time_format(cls, v: str) -> str:
            parts = v.split(":")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError(f"time must be HH:MM, got {v!r}")
            h, m = int(parts[0]), int(parts[1])
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(f"bad time: {v!r}")
            return v

    class VixSettings(BaseModel):
        halt_above: float = Field(gt=0)
        no_short_premium_above: float = Field(gt=0)
        no_0dte_long_below: float = Field(gt=0)

    class IVRankSettings(BaseModel):
        block_sell_below: float = Field(ge=0.0, le=1.0)
        block_buy_above: float = Field(ge=0.0, le=1.0)

    class ExecutionSettings(BaseModel):
        min_volume_confirmation: float = Field(ge=0.0, le=10.0)
        max_spread_pct_etf: float = Field(gt=0.0, le=0.5)
        max_spread_pct_stock: float = Field(gt=0.0, le=0.5)
        min_open_interest: int = Field(ge=0)
        min_today_option_volume: int = Field(ge=0)
        max_0dte_per_day: int = Field(ge=0, le=1000)
        order_ttl_seconds: int = Field(ge=1, le=60)

    class SizingSettings(BaseModel):
        kelly_fraction_cap: float = Field(gt=0.0, le=1.0)
        kelly_hard_cap_pct: float = Field(gt=0.0, le=0.25)
        max_contracts_0dte: int = Field(ge=1, le=1000)
        max_contracts_multiday: int = Field(ge=1, le=1000)

    class ExitSettings(BaseModel):
        profit_target_short_dte_pct: float = Field(gt=0.0, le=5.0)
        profit_target_multi_dte_pct: float = Field(gt=0.0, le=5.0)
        stop_loss_short_dte_pct: float = Field(gt=0.0, le=1.0)
        stop_loss_multi_dte_pct: float = Field(gt=0.0, le=1.0)
        hard_profit_cap_pct: float = Field(gt=0.0, le=10.0)

    class TradebotSettings(BaseModel):
        account: AccountSettings
        universe: List[str] = Field(min_length=1, max_length=500)
        session: SessionSettings
        vix: VixSettings
        iv_rank: IVRankSettings
        execution: ExecutionSettings
        sizing: SizingSettings
        exits: ExitSettings
        live_trading: bool = False


def validate_settings(raw: Dict[str, Any]) -> None:
    """Raises pydantic.ValidationError (or ValueError if no pydantic) if
    the settings dict is malformed. Ignores unknown keys — we don't want
    to break on new knobs that haven't been schema'd yet.
    """
    if not _HAS_PYDANTIC:
        _log.info("pydantic_not_installed_falling_back_to_manual_check")
        _manual_check(raw)
        return
    TradebotSettings(**{k: raw[k] for k in
                         ("account", "universe", "session", "vix",
                          "iv_rank", "execution", "sizing", "exits",
                          "live_trading") if k in raw})


def _manual_check(raw: Dict[str, Any]) -> None:
    """Last-resort bounds check when pydantic is unavailable."""
    def bound(path, lo, hi):
        cur = raw
        for k in path.split("."):
            if k not in cur:
                return
            cur = cur[k]
        if isinstance(cur, (int, float)) and not (lo <= cur <= hi):
            raise ValueError(f"config out of range: {path}={cur}, expected [{lo}, {hi}]")

    bound("sizing.kelly_fraction_cap", 0.01, 1.0)
    bound("sizing.kelly_hard_cap_pct", 0.01, 0.25)
    bound("account.max_daily_loss_pct", 0.0, 0.20)
    bound("account.max_open_positions", 1, 100)
    bound("vix.halt_above", 10, 100)
    bound("exits.profit_target_short_dte_pct", 0.01, 5.0)
    bound("exits.stop_loss_short_dte_pct", 0.01, 1.0)
