"""Run the bot in paper-trading mode (simulated fills against live OR synthetic
market data). Safe default: synthetic data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_settings
from src.core.logger import configure_logging
from src.main import TradeBot


def main() -> int:
    s = load_settings(ROOT / "config" / "settings.yaml")
    if os.getenv("LIVE_TRADING", "").lower() == "true" and s.raw.get("broker", {}).get("name") != "paper":
        print("LIVE_TRADING is set AND broker is not 'paper'. "
              "This script refuses to route real orders. "
              "Use a dedicated live-entry script after 30+ days of paper validation.")
        return 2
    configure_logging("INFO")
    bot = TradeBot(s)
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
