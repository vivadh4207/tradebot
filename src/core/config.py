"""Settings loader. Reads config/settings.yaml + .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def universe(self) -> List[str]:
        return self.raw.get("universe", [])

    @property
    def paper_equity(self) -> float:
        return float(self.raw["account"]["paper_starting_equity"])

    @property
    def max_risk_per_trade_pct(self) -> float:
        return float(self.raw["account"]["max_risk_per_trade_pct"])

    @property
    def max_daily_loss_pct(self) -> float:
        return float(self.raw["account"]["max_daily_loss_pct"])

    @property
    def max_open_positions(self) -> int:
        return int(self.raw["account"]["max_open_positions"])

    @property
    def live_trading(self) -> bool:
        env = os.getenv("LIVE_TRADING", "").lower() == "true"
        cfg = bool(self.raw.get("live_trading", False))
        return env and cfg

    def get(self, path: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def load_settings(path: str | Path = "config/settings.yaml",
                   validate: bool = True) -> Settings:
    p = Path(path)
    if not p.exists():
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "config" / "settings.yaml"
            if candidate.exists():
                p = candidate
                break
    with open(p, "r") as f:
        raw = yaml.safe_load(f)
    raw = raw or {}
    if validate:
        try:
            from .config_schema import validate_settings
            validate_settings(raw)
        except ValueError:
            # Manual bounds violations ARE fatal — kill the bot rather than
            # let invalid config destroy capital.
            raise
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "settings_schema_validation_failed: %s", e
            )
    return Settings(raw=raw)
