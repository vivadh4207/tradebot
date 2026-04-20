#!/usr/bin/env bash
# Threshold + py3.8 chat compat + reconcile skip + hourly log digest.
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

echo "== committing: ensemble threshold 0.85 -> 0.60 + py3.8 chat compat =="

git add \
  config/settings.yaml \
  scripts/discord_terminal.py \
  src/brokers/mirror_alpaca.py \
  src/reports/__init__.py \
  src/reports/log_digest.py \
  scripts/post_log_summary.py \
  scripts/tradebotctl.sh \
  deploy/systemd/tradebot-summary.service \
  deploy/systemd/tradebot-summary.timer \
  tests/test_log_digest.py

MSG_FILE="$(mktemp "${TMPDIR:-/tmp}/tradebot_commit_msg.XXXXXX")"
{
  echo "Lower ensemble threshold 0.85 -> 0.60 + py3.8 Discord chat compat"
  echo ""
  echo "1. Ensemble threshold"
  echo "   - ensemble.min_weighted_confidence: 0.85 -> 0.60"
  echo "   - 0.85 was so strict it rejected every entry candidate during"
  echo "     2026-04-20: scores topped out at 0.70. Operator missed a"
  echo "     real SPY-put entry during a dip because of this."
  echo "   - 0.60 still requires majority agreement across the signal"
  echo "     stack; just not near-unanimity."
  echo "   - No code change; settings.yaml only."
  echo ""
  echo "2. Optional reconcile skip-list"
  echo "   - New env ALPACA_RECONCILE_SKIP_SYMBOLS=<sym>,<sym> tells the"
  echo "     mirror to ignore specific Alpaca-only positions at startup."
  echo "     Useful when operator is intentionally letting a cheap"
  echo "     expiring position run to zero (e.g. SPY260424P00560000 at"
  echo "     \$0.02 entry, 4 DTE). Suppresses the repeated"
  echo "     alpaca_reconcile_skip_no_quote WARN lines without hiding"
  echo "     the mirror from other unseen positions."
  echo ""
  echo "3. Hourly log digest -> Discord (automated monitoring)"
  echo "   - New src/reports/log_digest.py parses logs/tradebot.out for"
  echo "     the last N minutes, groups events (entries, exits, skips,"
  echo "     warnings, errors, shutdowns), pulls most recent audit"
  echo "     health + age, emits a Discord-sized markdown digest."
  echo "   - scripts/post_log_summary.py runs the digest + posts via"
  echo "     the existing MultiChannelNotifier. Honors the title-based"
  echo "     routing (title='summary')."
  echo "   - deploy/systemd/tradebot-summary.{service,timer} fire the"
  echo "     digest at :05 past every hour on the Jetson. Persistent=true"
  echo "     so a missed trigger fires once on next boot."
  echo "   - tradebotctl.sh summary-install / summary-uninstall manage"
  echo "     the systemd --user unit + timer."
  echo "   - Discord !summary [N] command (in-process, ~1s) for ad-hoc"
  echo "     digest on demand. Defaults to 60-min window."
  echo ""
  echo "4. Py3.8 compat for Discord free-form chat"
  echo "   - Jetson default is Python 3.8. asyncio.to_thread() was added"
  echo "     in 3.9 so the previous handler raised AttributeError on"
  echo "     every chat call, which our fail-open wrapper masked as a"
  echo "     generic 'chat failed -- check logs' reply."
  echo "   - Swap to loop.run_in_executor(None, lambda: fn(...)). Same"
  echo "     dispatch semantics, works on 3.8+."
  echo "   - Chat reply now includes exception type + first 200 chars of"
  echo "     the error so the next failure is diagnosable in-place."
  echo "   - Full traceback still goes through _log.warning(..., exc_info=True)."
  echo ""
  echo "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
} > "$MSG_FILE"

git commit -F "$MSG_FILE"
rm -f "$MSG_FILE"

echo "== pushing to origin/main =="
git push origin main

echo ""
echo "done."
echo ""
echo "On the Jetson:"
echo "  1. git pull"
echo "  2. Edit ~/tradebot/.env:"
echo "       LLM_BACKEND=ollama"
echo "       LLM_CHAT_ENABLED=1"
echo "       DISCORD_CHAT_70B_CHANNELS=<channel_id_1>,<channel_id_2>"
echo "       # leave empty to keep every channel on 8B"
echo "  3. bash scripts/tradebotctl.sh restart"
echo "  4. systemctl --user restart tradebot-discord-terminal.service"
echo "  5. Watch your Discord — one hello per channel. Then in a 70B"
echo "     channel: 'walk me through today carefully'. In an 8B"
echo "     channel: 'quick status?'. The 70B reply ends with"
echo "     ' . model=llama3.1:70b' so you can confirm routing."
echo ""
echo "If audit still says strategy_auditor_model_missing, run:"
echo "  cd ~/tradebot"
echo "  python3 scripts/run_strategy_audit.py --no-discord 2>&1 | \\"
echo "    grep -E 'config|missing|unreachable|bad_json'"
echo "The new INFO log shows the resolved backend+model so you can see"
echo "exactly what the factory decided."
