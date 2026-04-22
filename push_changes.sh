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
  src/core/runtime_overrides.py \
  src/data/market_data.py \
  src/exits/fast_exit.py \
  src/intelligence/groq_client.py \
  src/intelligence/ollama_client.py \
  src/intelligence/options_research.py \
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
  echo "   Operator: 'even if we're in profit we can lose when it moves"
  echo "   the other way and we can't act on quick upside. Need smart"
  echo "   detection of support loss + profit-take.'"
  echo ""
  echo "   Three new exit layers, all run every fast-loop tick:"
  echo ""
  echo "   a. Profit-lock trailing (works on single-contract positions)"
  echo "      Arms at +15% peak pnl, tracks peak_pnl_pct per position,"
  echo "      closes when pnl retraces 30% from peak."
  echo "      peak=+40% -> floor=+28%  peak=+100% -> floor=+70%"
  echo ""
  echo "   b. Ratcheting profit floor"
  echo "      Tier-based minimum pnl once peaks cross thresholds:"
  echo "        peak>+25%  -> never exit below +10%"
  echo "        peak>+50%  -> never exit below +25%"
  echo "        peak>+100% -> never exit below +50%"
  echo ""
  echo "   c. Support-break exit"
  echo "      Long call: if close < 5-bar low AND pnl>+8% -> take profit."
  echo "      Long put:  if close > 5-bar high AND pnl>+8% -> take profit."
  echo "      Thesis is broken; don't wait for PT."
  echo ""
  echo "6. Discord AutotradePanel: 2 new button rows (10 buttons)"
  echo "   Row 3 (0DTE cap): 🎯 +1 / +5 / -1 / =20 / 🔄 Reset"
  echo "   Row 4 (bid pricing): 💰 Smarter Bids / ⚖ Mid / 🚀 Chase /"
  echo "                        💵 Cheap Only"
  echo ""
  echo "   All writes go through runtime_overrides. Live bot reads"
  echo "   next filter pass; no restart needed."
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
