#!/usr/bin/env bash
# Build script — produces dashboard.html in this directory, ready for
# Vercel static deploy. Copies the live template and injects a
# credentials loader that reads from localStorage (set by index.html).
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/../.."
TEMPLATE="$ROOT/src/dashboard/templates/index.html"

if [ ! -f "$TEMPLATE" ]; then
  echo "[!] template not found: $TEMPLATE" >&2
  exit 1
fi

# Header injected at the top of <script> — reads API base + token
# from localStorage set by the login page (./index.html). If missing,
# redirects back to login.
read -r -d '' INJECT << 'EOF' || true
// ===== Remote-host bootstrap (Vercel deploy) =====
const TRADEBOT_API_BASE  = localStorage.getItem('tradebot_api_base')  || '';
const TRADEBOT_API_TOKEN = localStorage.getItem('tradebot_api_token') || '';
if (!TRADEBOT_API_BASE || !TRADEBOT_API_TOKEN) {
  window.location.href = './';
}
// Logout helper
function tradebotLogout() {
  localStorage.removeItem('tradebot_api_base');
  localStorage.removeItem('tradebot_api_token');
  window.location.href = './';
}
window.tradebotLogout = tradebotLogout;
EOF

# Build dashboard.html = template with credential bootstrap prepended
# to the first <script> tag.
python3 - "$TEMPLATE" "$HERE/dashboard.html" "$INJECT" << 'PY'
import sys
src, dst, inject = sys.argv[1], sys.argv[2], sys.argv[3]
html = open(src).read()
# Prepend inject after the first "<script>" that isn't external.
marker = '<script>\n'
idx = html.find(marker)
if idx < 0:
    print('[!] could not find <script> block to inject into', file=sys.stderr)
    sys.exit(1)
new_html = html[:idx + len(marker)] + inject + '\n' + html[idx + len(marker):]
# Inject a logout link in the top bar brand section
new_html = new_html.replace(
    '<span class="pill mut" id="mode-pill">paper</span>',
    '<span class="pill mut" id="mode-pill">paper</span>\n'
    '    <button onclick="tradebotLogout()" '
    'style="background:transparent;border:1px solid var(--border);'
    'color:var(--fg-muted);font-size:10px;padding:2px 8px;border-radius:6px;'
    'cursor:pointer;margin-left:8px">logout</button>'
)
open(dst, 'w').write(new_html)
print(f'[ok] built {dst} ({len(new_html):,} chars)')
PY

echo
echo "Vercel files ready:"
ls -la "$HERE"/*.html "$HERE"/vercel.json 2>/dev/null | awk '{print "  " $NF}'
echo
echo "Deploy:"
echo "  cd $HERE"
echo "  vercel --prod"
echo
echo "Or just drag this folder to vercel.com/new."
