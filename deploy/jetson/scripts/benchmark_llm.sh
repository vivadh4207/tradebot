#!/usr/bin/env bash
# Benchmark the local LLM classifier on a canned payload.
# Confirms CUDA offload is working and measures tokens/sec.
set -euo pipefail

REPO="${1:-$(pwd)}"
VENV_PY="$REPO/.venv/bin/python"

"$VENV_PY" - <<'PY'
import os, time
from datetime import datetime, timezone

from src.intelligence.news import NewsItem
from src.intelligence.news_classifier_local import LocalLLMNewsClassifier

headlines = [
    "Company misses Q2 earnings estimates; guidance cut",
    "FDA issues complete response letter; stock tumbles",
    "Shares downgraded to Sell at Morgan Stanley",
    "Lawsuit filed over accounting irregularities",
    "Activist investor pushes for board changes",
]
items = [NewsItem(symbol="TEST", headline=h, source="bench",
                   published_at=datetime.now(tz=timezone.utc)) for h in headlines]

path = os.getenv("LLM_MODEL_PATH", "")
if not path:
    raise SystemExit("LLM_MODEL_PATH not set. Run download_model.sh first.")

print(f"[*] Loading model: {path}")
c = LocalLLMNewsClassifier()
if c._llm is None:
    raise SystemExit("Failed to load model. Check log above.")

print("[*] Warmup (1 call) …")
c.score(items)

N = 3
print(f"[*] Benchmark: {N} classifications")
t0 = time.time()
for _ in range(N):
    score, rationale = c.score(items)
dt = time.time() - t0
print(f"[=] {N} calls in {dt:.2f}s — avg {dt/N:.2f}s per call")
print(f"[=] Last result: score={score:+.3f} rationale={rationale!r}")
PY
