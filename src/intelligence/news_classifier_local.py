"""LocalLLMNewsClassifier — GPU-accelerated news sentiment via llama-cpp-python.

Designed for NVIDIA Jetson AGX Orin but runs anywhere llama-cpp-python
installs with CUDA or Metal support. Uses a quantized GGUF model so the
Jetson can run Qwen2.5-7B / Llama-3.1-8B at real-time speeds entirely
on-device, with no API cost and no data leaving the box.

Configuration via env (all optional — safe fallback to keyword classifier):
  LLM_MODEL_PATH     absolute path to the .gguf model file
  LLM_N_GPU_LAYERS   how many transformer layers on GPU (-1 = all). Default: -1.
  LLM_N_CTX          context size. Default: 4096.
  LLM_N_THREADS      CPU threads if no GPU. Default: number of cores.
  LLM_TEMPERATURE    sampling temperature. Default: 0.1 (near-deterministic).
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from .news import NewsItem
from .news_classifier import NewsClassifier, KeywordClassifier


class LocalLLMNewsClassifier(NewsClassifier):
    """News sentiment via a local llama.cpp model.

    Model file format: GGUF (quantized Q4_K_M or Q5_K_M recommended).
    Memory use on Jetson AGX Orin 64GB: ~6GB for a 7B-8B Q4 model, leaving
    plenty of headroom for the rest of the bot.
    """

    def __init__(self,
                 model_path: str | None = None,
                 n_gpu_layers: int | None = None,
                 n_ctx: int = 4096,
                 n_threads: int | None = None,
                 temperature: float = 0.1):
        self._fallback = KeywordClassifier()
        self._model_path = model_path or os.getenv("LLM_MODEL_PATH", "").strip()
        if n_gpu_layers is None:
            n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
        if n_threads is None:
            n_threads = int(os.getenv("LLM_N_THREADS", os.cpu_count() or 4))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self._llm = None
        if not self._model_path or not os.path.exists(self._model_path):
            return
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self._model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
                logits_all=False,
                embedding=False,
            )
        except Exception:
            self._llm = None

    def score(self, items: List[NewsItem]) -> Tuple[float, str]:
        if not items:
            return 0.0, "no_items"
        if self._llm is None:
            return self._fallback.score(items)
        heads = [it.headline for it in items[:20]]
        prompt = self._build_prompt(heads)
        try:
            out = self._llm.create_completion(
                prompt=prompt,
                max_tokens=120,
                temperature=self.temperature,
                top_p=0.9,
                stop=["\n\n", "```"],
            )
            text = (out["choices"][0]["text"] or "").strip()
            # find the first {..} JSON object in the output
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return self._fallback.score(items)
            data = json.loads(text[start: end + 1])
            score = float(data.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            rationale = str(data.get("rationale", ""))[:120]
            return score, f"local_llm: {rationale}"
        except Exception:
            return self._fallback.score(items)

    def _build_prompt(self, headlines: List[str]) -> str:
        joined = "\n".join(f"- {h}" for h in headlines)
        return (
            "You are a financial news sentiment classifier. Read the headlines "
            "for one ticker and return a JSON object with keys:\n"
            '  "score" (number in [-1,1]; -1=very negative, +1=very positive)\n'
            '  "rationale" (<=120 chars)\n'
            "Strongly negative: downgrades, misses, FDA rejections, probes, "
            "lawsuits, guidance cuts.\n"
            "Strongly positive: beats, upgrades, approvals, buybacks, raises.\n"
            "Neutral: routine / off-topic.\n\n"
            f"Headlines:\n{joined}\n\n"
            "Respond with JSON only. No prose, no code fences.\nJSON: "
        )

    def close(self) -> None:
        """Release model weights (useful in tests)."""
        self._llm = None
