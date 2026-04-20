"""Run the bot in paper-trading mode (simulated fills against live OR synthetic
market data). Safe default: synthetic data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.config import load_settings
from src.core.logger import configure_logging
from src.main import TradeBot
from src.notify.issue_reporter import alert_on_crash, install_excepthooks


@alert_on_crash("run_paper")
def main() -> int:
    # Install excepthooks BEFORE anything else — if load_settings or
    # TradeBot.__init__ blows up, the traceback should still reach
    # Discord alerts rather than dying silently to stderr.
    install_excepthooks(scope_prefix="tradebot")

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
