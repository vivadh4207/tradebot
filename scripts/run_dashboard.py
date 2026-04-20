"""Launch the read-only dashboard.

Usage:
  python scripts/run_dashboard.py                     # 127.0.0.1:8000
  python scripts/run_dashboard.py --port 8080
  python scripts/run_dashboard.py --host 0.0.0.0      # DANGEROUS: only behind auth

Bind to localhost by default. If you want to view from another machine,
SSH-tunnel:
  ssh -L 8000:localhost:8000 user@your-vps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. pip install 'uvicorn[standard]' fastapi")
        return 2
    if args.host != "127.0.0.1":
        print(f"[WARN] Binding to {args.host} — dashboard is unauthenticated. "
              "Only do this behind a reverse-proxy with auth, or an SSH tunnel.")
    uvicorn.run("src.dashboard.app:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
