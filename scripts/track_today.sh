#!/usr/bin/env bash
# Real-time trade tracker — Tradier + local snapshot side-by-side.
# Usage:
#   bash scripts/track_today.sh           # one-shot
#   bash scripts/track_today.sh watch     # auto-refresh every 30s
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# load env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

if [ -z "${TRADIER_TOKEN:-}" ] || [ -z "${TRADIER_ACCOUNT:-}" ]; then
  echo "[!] TRADIER_TOKEN / TRADIER_ACCOUNT not set. source .env first."
  exit 1
fi

PYTHON_BIN="${TRADEBOT_PY:-$ROOT/.venv/bin/python}"

run_once() {
  clear
  echo "================================================================"
  echo "  TRADE TRACKER — $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S ET')"
  echo "================================================================"

  # Tradier balance
  echo ""
  echo "─── Tradier Balance (truth) ─────────────────────────────────"
  curl -s -H "Authorization: Bearer $TRADIER_TOKEN" -H "Accept: application/json" \
    "https://sandbox.tradier.com/v1/accounts/$TRADIER_ACCOUNT/balances" \
    | "$PYTHON_BIN" -c "
import sys, json
raw = sys.stdin.read().strip()
if not raw: print('  (Tradier balance empty — retry shortly)'); sys.exit(0)
try: data = json.loads(raw)
except Exception as e: print(f'  (parse err: {str(e)[:60]})'); sys.exit(0)
b = data.get('balances', {})
eq = float(b.get('total_equity') or 0)
cash = float(b.get('total_cash') or 0)
op = float(b.get('open_pl') or 0)
cp = float(b.get('close_pl') or 0)
print(f'  equity:    \${eq:>10,.2f}')
print(f'  cash:      \${cash:>10,.2f}')
print(f'  open_pl:   \${op:>+10,.2f}')
print(f'  close_pl:  \${cp:>+10,.2f}')
print(f'  net_day:   \${op+cp:>+10,.2f}')
"

  # Tradier positions
  echo ""
  echo "─── Tradier Open Positions ──────────────────────────────────"
  curl -s -H "Authorization: Bearer $TRADIER_TOKEN" -H "Accept: application/json" \
    "https://sandbox.tradier.com/v1/accounts/$TRADIER_ACCOUNT/positions" \
    | "$PYTHON_BIN" -c "
import sys, json
raw = sys.stdin.read().strip()
if not raw: print('  (positions empty — retry shortly)'); sys.exit(0)
try: data = json.loads(raw)
except Exception as e: print(f'  (parse err: {str(e)[:60]})'); sys.exit(0)
pos_obj = data.get('positions')
# Tradier returns 'null' (string) when account is flat
if not pos_obj or not isinstance(pos_obj, dict):
    print('  (none — flat)')
    sys.exit(0)
ps = pos_obj.get('position', []) or []
if isinstance(ps, dict): ps = [ps]
if not ps: print('  (none — flat)'); sys.exit(0)
total_cost = 0
for p in ps:
    sym = p.get('symbol','')
    qty = int(float(p.get('quantity', 0)))
    cost = float(p.get('cost_basis') or 0)
    total_cost += cost
    print(f'  {sym:25s} qty={qty:>3}  cost=\${cost:>8,.2f}')
print(f'  TOTAL EXPOSURE:                \${total_cost:>10,.2f}')
"

  # Today's order activity
  echo ""
  echo "─── Today's Tradier Activity ────────────────────────────────"
  curl -s -H "Authorization: Bearer $TRADIER_TOKEN" -H "Accept: application/json" \
    "https://sandbox.tradier.com/v1/accounts/$TRADIER_ACCOUNT/orders" \
    | "$PYTHON_BIN" -c "
import sys, json, datetime
raw = sys.stdin.read().strip()
if not raw:
    print('  (Tradier returned empty — possibly rate-limited)'); sys.exit(0)
try:
    data = json.loads(raw)
except Exception as e:
    print(f'  (json parse failed: {str(e)[:80]})'); sys.exit(0)
ord_obj = data.get('orders')
if not ord_obj or not isinstance(ord_obj, dict):
    print('  (no orders today)'); sys.exit(0)
orders = ord_obj.get('order', []) or []
if isinstance(orders, dict): orders = [orders]
today = datetime.date.today().isoformat()
today_orders = [o for o in orders if o.get('transaction_date','')[:10] == today]
filled = [o for o in today_orders if o.get('status') == 'filled']
rejected = [o for o in today_orders if o.get('status') == 'rejected']
pending = [o for o in today_orders if o.get('status') in ('open','pending')]
print(f'  Filled:    {len(filled):>3}')
print(f'  Rejected:  {len(rejected):>3}')
print(f'  Pending:   {len(pending):>3}')
print()
filled.sort(key=lambda x: x.get('transaction_date',''), reverse=True)
print('  Latest 5 fills:')
for o in filled[:5]:
    sym = (o.get('option_symbol') or o.get('symbol',''))[:30]
    side = (o.get('side','')).split('_')[0]
    qty = int(float(o.get('exec_quantity') or 0))
    px = float(o.get('avg_fill_price') or 0)
    cash = qty * px * 100 * (-1 if 'buy' in o.get('side','') else 1)
    t = o.get('transaction_date','')[11:16]
    print(f'    {t:>5s} {sym:30s} {side:5s} qty={qty:>2}  \${px:>5.2f}  \${cash:>+8.2f}')
print()
if rejected:
    print('  Latest 3 rejects:')
    for o in rejected[-3:]:
        sym = (o.get('option_symbol') or o.get('symbol',''))[:30]
        reason = (o.get('reason_description','') or '')[:60]
        print(f'    {sym:30s} {reason}')
"

  # Local broker state
  echo ""
  echo "─── Local PaperBroker (bot's view) ──────────────────────────"
  if [ -f logs/broker_state.json ]; then
    "$PYTHON_BIN" -c "
import json
with open('logs/broker_state.json') as f: s = json.load(f)
print(f\"  cash:       \${s.get('cash', 0):>10,.2f}\")
print(f\"  day_pnl:    \${s.get('day_pnl', 0):>+10,.2f}\")
print(f\"  positions:  {len(s.get('positions', []))}\")
"
  else
    echo "  (no broker_state.json found)"
  fi

  # Recent log activity
  echo ""
  echo "─── Last 5 Bot Events ───────────────────────────────────────"
  if [ -f logs/tradebot.out ]; then
    grep -E "ensemble_emit|exec_chain_pass|tradier_filled|tradier_terminal_reject|fast_exit|entry_skip" logs/tradebot.out 2>/dev/null \
      | tail -5 \
      | sed 's/^/  /' \
      | cut -c1-120
  fi

  echo ""
  echo "================================================================"
}

if [ "${1:-}" = "watch" ]; then
  while true; do
    run_once
    echo "Refreshing in 30s... (Ctrl-C to stop)"
    sleep 30
  done
else
  run_once
fi
