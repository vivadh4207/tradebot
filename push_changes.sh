#!/usr/bin/env bash
# Push: Groq LLM + pre/post scans + runtime overrides + smart exits.
#
# What this commits:
#   - Groq client (cloud 70B) for research/audit/macro/catalyst
#   - Pre-market + post-market scanners (separate Discord channels)
#   - Runtime-overrides infra (Discord buttons adjust live knobs)
#   - 0DTE cap=20, expensive-contract filter, tighter bid pricing
#   - Profit-lock trailing, ratcheting profit floor, support-break exit
#   - Discord AutotradePanel: 0DTE +/-/=20, Smarter Bids / Mid / Chase
#
# Run from the tradebot root on your Mac:
#   bash push_changes.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

rm -f .git/index.lock .git/HEAD.lock 2>/dev/null || true
rm -f .git/objects/*/tmp_obj_* 2>/dev/null || true

branch="$(git branch --show-current)"
if [ "$branch" != "main" ]; then
  echo "  [!] current branch is not 'main'. Aborting."
  exit 1
fi
if git status --short | grep -qE '^(M|A|\?\?) \.env$'; then
  echo "  [!] .env appears in git status. Refusing."
  exit 1
fi

echo "== committing: Groq LLM + pre/post scan + runtime overrides + smart exits =="

git add \
  .gitignore \
  config/settings.yaml \
  push_changes.sh \
  scripts/discord_terminal.py \
  scripts/run_morning_scan.py \
  scripts/run_strategy_report.py \
  src/core/runtime_overrides.py \
  src/data/market_data.py \
  src/exits/fast_exit.py \
  src/intelligence/groq_client.py \
  src/intelligence/ollama_client.py \
  src/intelligence/options_research.py \
  src/intelligence/position_advisor.py \
  src/intelligence/symbol_scanner.py \
  src/main.py \
  src/notify/base.py \
  src/risk/execution_chain.py \
  deploy/launchd/com.tradebot.scan_premarket.plist \
  deploy/launchd/com.tradebot.scan_postmarket.plist

MSG_FILE="$(mktemp "${TMPDIR:-/tmp}/tradebot_commit_msg.XXXXXX")"
{
  echo "Groq LLM + pre/post scans + runtime overrides + smart profit exits"
  echo ""
  echo "1. Groq cloud LLM backend (src/intelligence/groq_client.py)"
  echo "   OpenAI-compatible client for Groq's hosted Llama 3.3 70B."
  echo "   Free tier = 14,400 req/day, 500+ tok/sec. Used for research,"
  echo "   audit, macro, catalyst, chat_70b roles. Per-trade brain stays"
  echo "   on local Ollama 8B for speed + privacy."
  echo ""
  echo "   Role-based factory build_llm_client_for(role):"
  echo "     research/audit/macro/catalyst/chat_70b -> Groq if"
  echo "     GROQ_API_KEY set, else local Ollama 70B tag."
  echo "     brain/chat -> always local Ollama 8B."
  echo ""
  echo "   Ollama-style model tags (llama3.1:70b) auto-map to Groq tags"
  echo "   (llama-3.3-70b-versatile) so consumer scripts don't branch."
  echo ""
  echo "   options_research.py rewired to use this factory — its 70B"
  echo "   calls now run on Groq cloud when configured."
  echo ""
  echo "2. Pre-market + post-market scanners (scripts/run_morning_scan.py)"
  echo "   Session-aware ticker screener with Discord integration."
  echo ""
  echo "     --session pre  -> '🌅 Pre-Market Brief' at 08:30 ET"
  echo "     --session post -> '🌆 Post-Market Recap' at 16:30 ET"
  echo "     --session auto -> detects from ET clock"
  echo ""
  echo "   Each session routes to its own Discord channel via notifier"
  echo "   titles 'scan_premarket' / 'scan_postmarket' (wired into"
  echo "   src/notify/base.py _TITLE_TO_CHANNEL). 70B desk-note"
  echo "   commentary with bull/bear parity is generated per session"
  echo "   via the Groq factory."
  echo ""
  echo "   Two launchd plists installed for daily fire:"
  echo "     deploy/launchd/com.tradebot.scan_premarket.plist"
  echo "     deploy/launchd/com.tradebot.scan_postmarket.plist"
  echo ""
  echo "3. Runtime-overrides infra (src/core/runtime_overrides.py)"
  echo "   File-backed JSON state at data/runtime_overrides.json. Discord"
  echo "   buttons write here; filter chain + order pricing read live."
  echo "   No restart needed to tune knobs."
  echo ""
  echo "   Overrides wired:"
  echo "     max_0dte_per_day       (f13_0dte_cap)"
  echo "     max_premium_per_contract_usd (f11 expensive-contract gate)"
  echo "     entry_spread_pct       (main.py order pricing)"
  echo ""
  echo "4. Expensive-contract discipline (config/settings.yaml)"
  echo "   Operator: 'bid/ask-cross on wide spreads bleeds us. Need to"
  echo "   be selective, not just go for asks.'"
  echo ""
  echo "     max_spread_pct_etf:   0.05 -> 0.03"
  echo "     max_spread_pct_stock: 0.10 -> 0.06"
  echo "     entry_spread_pct:     0.30 -> 0.15 (sit near bid)"
  echo "     max_0dte_per_day:     5    -> 20"
  echo "     max_premium_per_contract_usd: NEW -> 3.00"
  echo ""
  echo "   New filter (f11 extension): blocks any contract with"
  echo "   ask > max_premium cap. Forces cheaper OTM strikes ->"
  echo "   diversification across more positions within same $-budget."
  echo ""
  echo "5. Intelligent profit protection (src/exits/fast_exit.py)"
  echo "   Operator: '+75% went back to +1% — algo needs to monitor"
  echo "   charts, detect trend reversal, close and take profit.'"
  echo ""
  echo "   Five new exit layers run every fast-loop tick (1s cadence)"
  echo "   with 1-min bars of the underlying. All fire ONLY when in"
  echo "   profit — losers still ride the stop-loss."
  echo ""
  echo "   a. Profit-lock trailing with ADAPTIVE give-back"
  echo "      Arms at +10% peak pnl (was +15%). Give-back tier-scaled:"
  echo "        peak >=+100% -> give back 15% -> floor +85%"
  echo "        peak >= +50% -> give back 20% -> floor +60% of peak"
  echo "        peak >= +25% -> give back 25% -> floor +26% min"
  echo "        peak >= +10% -> give back 35% -> floor +7%"
  echo "      Works on single-contract positions (no scale-out needed)."
  echo ""
  echo "   b. Ratcheting profit floor (tier-based hard min)"
  echo "        peak>+25%  -> never exit below +10%"
  echo "        peak>+50%  -> never exit below +25%"
  echo "        peak>+100% -> never exit below +50%"
  echo ""
  echo "   c. Support-break / resistance-break exit"
  echo "      Long call: close breaks 5-bar low AND pnl>+8% -> exit."
  echo "      Long put:  close breaks 5-bar high AND pnl>+8% -> exit."
  echo ""
  echo "   d. Lower-high / higher-low chart pattern"
  echo "      Long call: 2 consecutive lower highs AND pnl>+5% -> exit."
  echo "      Long put:  2 consecutive higher lows AND pnl>+5% -> exit."
  echo "      Classic trend-change signature — fires BEFORE the reversal"
  echo "      bleeds gains back."
  echo ""
  echo "   e. VWAP-break exit"
  echo "      Long call: underlying closes below rolling VWAP AND pnl>+5%"
  echo "         -> exit. Institutional value line lost."
  echo "      Long put:  mirror (reclaim VWAP)."
  echo ""
  echo "   Human-readable exit reasons wired in main.py for Discord:"
  echo "     profit_lock, profit_floor, support_break, chart_lower_highs,"
  echo "     chart_higher_lows, vwap_break, vwap_break_up -- each"
  echo "     explained in the close notification."
  echo ""
  echo "6. Discord AutotradePanel: 2 new button rows (10 buttons)"
  echo "   Row 3 (0DTE cap): 🎯 +1 / +5 / -1 / =20 / 🔄 Reset"
  echo "   Row 4 (bid pricing): 💰 Smarter Bids / ⚖ Mid / 🚀 Chase /"
  echo "                        💵 Cheap Only"
  echo ""
  echo "   All writes go through runtime_overrides. Live bot reads"
  echo "   next filter pass; no restart needed."
  echo ""
  echo "7. Strategy-bucket experiment (config/settings.yaml + main.py)"
  echo "   Operator: 'like the long strategy but also experiment with"
  echo "   0DTE, swing so we have data.'"
  echo ""
  echo "   Each entry is assigned to a bucket by weighted random pick:"
  echo "     0dte:  20% — DTE 0 or 1  (same-day / overnight lottery)"
  echo "     short: 30% — DTE 2, 5, 7 (directional bets)"
  echo "     swing: 50% — DTE 14, 21, 30 (thesis has time to play out)"
  echo ""
  echo "   Entry tag gets |strategy=<bucket> appended so the journal"
  echo "   can group closed trades by bucket."
  echo ""
  echo "   Runtime override key 'strategy_bucket_weights' lets Discord"
  echo "   buttons (future) tilt the mix without a restart."
  echo ""
  echo "8. Strategy-bucket P&L report (scripts/run_strategy_report.py)"
  echo "   Reads closed trades from SqliteJournal, groups by bucket,"
  echo "   reports per-bucket count / win rate / avg pnl% / total $,"
  echo "   median hold, top exit reasons, best + worst trade."
  echo ""
  echo "   Falls back to DTE-based classification for trades recorded"
  echo "   before the bucket tag existed — historical data still reports."
  echo ""
  echo "   Posts to Discord title='strategy_report' (new route in"
  echo "   notify/base.py _TITLE_TO_CHANNEL). Env var to route to its"
  echo "   own channel: DISCORD_WEBHOOK_URL_STRATEGY_REPORT=..."
  echo ""
  echo "   Run:  .venv/bin/python scripts/run_strategy_report.py"
  echo ""
  echo "9. Scanner ticker blocklist hardened (src/intelligence/symbol_scanner.py)"
  echo "   Operator saw noise in the post-market scan: US, EU, UK, FX,"
  echo "   ZEW, PM, IT, UT, GLP, S, P, U, I being treated as tickers."
  echo ""
  echo "   Extended _BLOCKED_SYMBOLS with:"
  echo "     - Geopolitical: US, EU, UK, UN, NATO, WHO, G7, G20, OPEC"
  echo "     - Economic:    GDP, CPI, PPI, FOMC, ECB, ZEW, IFO, NFP, ADP"
  echo "     - Policy:      SEC, FTC, EPA, IRS, DOJ, FDA, FAA, FBI, CIA"
  echo "     - Single-letter noise: S, P, U, E, V, M, N, O, B, W, Y, Z"
  echo "     - State codes: UT, NY, CA, TX, FL, ...  (30+ entries)"
  echo "     - Words: BUY, SELL, HOLD, TOP, NEW, OPEN, CLOSE, ..."
  echo ""
  echo "   run_morning_scan.py rewired to use the canonical"
  echo "   _extract_tickers_from_text instead of inline regex. Defense-"
  echo "   in-depth: _screen_ticker rejects bad symbols before hitting"
  echo "   the network. yfinance 'possibly delisted' stderr silenced."
  echo ""
  echo "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
} > "$MSG_FILE"

git commit -F "$MSG_FILE"
rm -f "$MSG_FILE"

echo ""
echo "== pushing to origin/main =="
git push origin main

echo ""
echo "done."
echo ""
echo "On the Mac (in ~/tradebot):"
echo "  1. git pull  (if you ran the push from elsewhere)"
echo "  2. Add to .env:"
echo "       GROQ_API_KEY=gsk_...      (get from https://console.groq.com/keys)"
echo "       DISCORD_WEBHOOK_URL_SCAN_PREMARKET=https://discord.com/api/webhooks/..."
echo "       DISCORD_WEBHOOK_URL_SCAN_POSTMARKET=https://discord.com/api/webhooks/..."
echo "       # Leave webhook URLs empty to land in the default channel."
echo ""
echo "  3. Install the two new launchd plists:"
echo "       bash scripts/tradebotctl.sh install  # re-render all plists"
echo "       # OR manually copy + load each:"
echo "       cp deploy/launchd/com.tradebot.scan_premarket.plist  ~/Library/LaunchAgents/"
echo "       cp deploy/launchd/com.tradebot.scan_postmarket.plist ~/Library/LaunchAgents/"
echo "       # Replace __TRADEBOT_ROOT__ + __TRADEBOT_PY__ placeholders in the copies."
echo "       launchctl load ~/Library/LaunchAgents/com.tradebot.scan_premarket.plist"
echo "       launchctl load ~/Library/LaunchAgents/com.tradebot.scan_postmarket.plist"
echo ""
echo "  4. Restart the paper bot + Discord bot so smart-exit, filter,"
echo "     and button changes take effect:"
echo "       bash scripts/tradebotctl.sh restart"
echo "       # the Discord terminal service picks up the new buttons"
echo "       # on next restart; !autopanel to re-post in a channel."
echo ""
echo "  5. Sanity-check manually:"
echo "       .venv/bin/python scripts/run_morning_scan.py --session pre"
echo "       .venv/bin/python scripts/run_morning_scan.py --session post"
echo ""
echo "  6. In Discord: !autopanel — should now show 4 rows of buttons."
