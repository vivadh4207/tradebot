# Running an AI agent on the Jetson

Goal: have an agent-capable setup on the Orin that can do the kind of
work you're doing with Claude today — chat, code edits, research — with
as much running locally (free, private, offline) as possible, and a
clean fallback to hosted Claude when you need frontier quality.

---

## Honest framing (read first)

| Task | Best fit | Why |
|---|---|---|
| Bot's news classifier (already set up) | Local 7B GGUF via llama-cpp | Runs in the bot process, no API cost |
| "Add a test / refactor this function" on your codebase | **Local coder model via Ollama + Aider** | Private, free, 8–15 tok/s is fine for this |
| "Design the next version of my strategy" | **Claude via claude.ai or Claude Code → Anthropic API** | Local 32B can't match Claude Opus reasoning |
| Open-ended chat / research / explain-this-paper | Local model for quick stuff, Claude for hard stuff | Your call per query |
| Fully autonomous multi-step agent | Don't. Not yet at this hardware tier. | Small models compound errors on long loops |

Short story: **the Jetson is a great way to avoid API bills on routine
work and keep sensitive data local, but it's not a Claude replacement
for the hard problems.** Use both, switch per task.

---

## Three-tier setup (~40 minutes)

Run the layers you want. Tier 1 gives you a local chat agent. Tier 2
gives you a coding agent. Tier 3 gives you Claude-quality when needed.

### Tier 1 — Ollama + Open WebUI (local chat, 15 min)

**[Ollama](https://ollama.com)** is the easiest way to manage local LLMs.
It has first-class ARM64/CUDA support on JetPack 6.

```bash
# On the Jetson, via SSH
curl -fsSL https://ollama.com/install.sh | sh

# Pull a couple of models. Q4 quants. Pick based on what you'll do:
ollama pull qwen2.5:7b-instruct       # fast general chat (~40 tok/s)
ollama pull qwen2.5-coder:14b         # coding (~20 tok/s, fits comfortably)
ollama pull llama3.1:8b               # alternative for general chat

# Test:
ollama run qwen2.5:7b-instruct "Summarize options theta decay in 3 lines."
```

Want a ChatGPT-like web UI? Add **Open WebUI**:

```bash
# Docker is the path of least resistance:
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER && newgrp docker

docker run -d --name open-webui \
  --add-host=host.docker.internal:host-gateway \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# From your Mac, tunnel the port:
# ssh -L 3000:127.0.0.1:3000 orin@<jetson-ip>
# Then open http://127.0.0.1:3000 on your Mac.
```

You now have a private chat UI that routes to the Jetson's GPU. Create
an account on first load; it's local-only.

**When to use this**: brainstorming, explaining docs, rubber-ducking,
asking "how do I do X" questions where you don't need perfection.

### Tier 2 — Aider (coding agent on the tradebot repo, 5 min)

[Aider](https://aider.chat) is the cleanest local-LLM coding agent: a
CLI that reads/writes files in a git repo with your guidance. It speaks
to Ollama directly.

```bash
# On the Jetson:
cd ~/tradebot
source .venv/bin/activate
pip install aider-chat

# Point it at the local coder model. Aider auto-discovers Ollama on
# localhost.
export OLLAMA_API_BASE=http://127.0.0.1:11434

# Start a session, passing the files you want it to focus on:
aider --model ollama/qwen2.5-coder:14b \
      src/signals/momentum.py tests/test_momentum.py
```

Once it's running, talk to it like you'd talk to me — "add a
`min_volume_confirmation` parameter to the momentum signal with a
default of 1.2", "write a test for the edge case where volume is
zero", "explain why this function returns None on line 47". It diffs,
shows you the change, and commits if you say yes.

What it does well:

- Targeted edits to files you name
- Writing tests for existing functions
- Explaining code
- Translating between languages

What it does badly (versus Claude):

- Open-ended design ("architect me a new strategy")
- Multi-file refactors spanning the whole repo
- Subtle reasoning about probability, statistics, or trading specifics
- Anything where being subtly wrong matters

**Rule of thumb**: if the task is "here's a file, do X to it", local
coder is fine. If the task is "figure out what to do and do it across
the repo", use Claude.

### Tier 3 — Claude Code on the Jetson (Claude-quality when needed)

When you hit a task that really needs Claude, run Claude Code from the
Jetson itself. The Jetson is just a different shell; Claude Code is a
client that talks to Anthropic's API.

```bash
# Install (requires Node.js):
sudo apt-get install -y nodejs npm
npm install -g @anthropic-ai/claude-code

# Set your API key:
export ANTHROPIC_API_KEY=sk-ant-...
# Add that line to ~/.bashrc so it persists.

# Run it from the repo:
cd ~/tradebot
claude
```

Same UX as the Cowork you're using now, just rooted in the Jetson's
filesystem. Costs Anthropic API tokens per turn — budget accordingly.

**When to use this**: the hard problems you'd have picked Claude for
anyway. Don't spend API tokens on "rename this variable"; use Tier 2
for that.

---

## Wiring the tiers together (what I'd actually do)

One good pattern: keep a terminal pane per tier, switch based on the
task.

```
┌─────────────────────────┬─────────────────────────┐
│ Pane 1: Open WebUI      │ Pane 2: Aider           │
│ (chat / research)       │ (coding on tradebot)    │
├─────────────────────────┼─────────────────────────┤
│ Pane 3: Claude Code     │ Pane 4: tradebot logs   │
│ (hard problems)         │ (tail -f tradebot.out)  │
└─────────────────────────┴─────────────────────────┘
```

`tmux` is worth learning if you haven't — `sudo apt install tmux`, then
`tmux` to launch, Ctrl-b `%` for vertical split, Ctrl-b `"` for
horizontal. Attach from anywhere with `tmux attach`.

---

## The tradebot's own LLM (already set up)

The bot uses a local LLM for news classification — replacing the
Anthropic API call that would otherwise fire every 5 minutes per
symbol. That's already installed by `deploy/jetson/setup.sh` (Qwen2.5
7B Q4 via llama-cpp-python). It lives in its own Python process inside
the bot and has nothing to do with Ollama — if you kill Ollama, the
bot keeps working.

If you want to point the bot at an Ollama model instead of the
llama-cpp one, that's a code change in
`src/classifiers/build_classifier.py`. Not urgent; the llama-cpp path
is efficient and already works.

---

## Model picks, ranked for this hardware

The Orin 64GB can fit these comfortably. Pick by task, not by hype.

**Chat / general:**

- `qwen2.5:7b-instruct` — best all-rounder at this size
- `llama3.1:8b` — close second, slightly better at reasoning
- `qwen2.5:14b` — noticeably smarter if you can tolerate ~20 tok/s
- `qwen2.5:32b` — best local-model experience; ~8 tok/s (needs Q4_K_M)

**Coding (Aider):**

- `qwen2.5-coder:14b` — best at this size; default pick
- `qwen2.5-coder:32b` — better quality, slower; worth it for big refactors
- `deepseek-coder-v2:16b` — strong alternative, fatter MoE

**Embedding (if you build a RAG layer later):**

- `nomic-embed-text` — fast, decent quality
- `bge-m3` — better quality, slower

Sample pulls:

```bash
ollama pull qwen2.5-coder:32b       # big coder (~20 GB)
ollama pull qwen2.5:14b             # mid chat (~9 GB)
ollama pull nomic-embed-text        # embeddings (~275 MB)
```

Default pick if you want one of each: `qwen2.5:7b-instruct` +
`qwen2.5-coder:14b`. Total disk ~15 GB, fits alongside the tradebot
fine.

---

## What you do NOT want

- **"Let the local model autonomously edit 20 files in a loop."** Error
  compounding eats you alive at this quality tier. Keep the human (you)
  in the loop for every diff. That's why Aider's approval flow is the
  right shape.
- **Running the local model against live-trading decisions.** The
  bot's current news classifier is fine because it's a structured
  3-class output and the bot has other signals to corroborate. Don't
  add a "GPT decides whether to buy" layer with a local model; the
  error distribution is wrong for that.
- **Exposing Ollama or Open WebUI on the public internet.** They have
  minimal auth. Keep them behind SSH tunnels or a VPN.
- **Running a 70B model because bigger is better.** On the Orin a 70B
  Q4 runs at maybe 4 tok/s. Painful. The quality bump rarely justifies
  the speed hit at this hardware tier. 32B is the sweet spot.

---

## Quick command reference

```bash
# List models Ollama has
ollama list

# Chat with a specific model once
ollama run qwen2.5:7b-instruct

# Run Aider on a subset of files
aider --model ollama/qwen2.5-coder:14b src/main.py

# Claude Code from anywhere in the repo
claude

# Open WebUI (if docker container is up)
# (SSH tunnel from your Mac, then http://127.0.0.1:3000)

# Check GPU usage while a model runs
sudo jtop   # from deploy/jetson/requirements-jetson.txt

# Watch VRAM
watch -n 1 'nvidia-smi || tegrastats | head -1'
```

---

## Honest cost comparison (your ~monthly bill)

| Setup | What you run | Typical monthly cost |
|---|---|---|
| Mac only | Claude API for everything | $30–150+ depending on usage |
| Jetson, routine tasks local | Ollama for ~80% of requests, Claude for the hard 20% | Probably $5–30 |
| Jetson, all local | Ollama only | $0 (plus ~$15/month electricity for 24/7 Orin) |

The Jetson pays for itself quickly if you're a heavy Claude user,
provided you're OK routing the routine work to smaller models. If you
route *everything* to Claude, the Jetson is dead weight.

---

## What this does NOT give you

Just so we're clear:

- It does **not** give you GPT-4/Claude Opus-quality output locally.
  Nothing in a $2K dev kit does, and this is a physics/hardware fact
  rather than a software-tuning gap.
- It does **not** replace my role in Cowork. I can still act on your
  Mac with tools and memory. The Jetson is a separate, independent
  agent host — not a remote brain for Cowork.
- It does **not** auto-update. When Ollama/Aider/Claude-Code release
  new versions, `curl ... | sh`, `pip install -U aider-chat`,
  `npm update -g @anthropic-ai/claude-code` respectively.

---

## Starter prompt I'd run first

After Tier 2 is up, try this in Aider:

```
/add src/signals/momentum.py tests/test_pattern_filters.py
> Explain how the momentum signal works in this repo, file by file,
  and identify the two weakest assumptions in its current logic.
  Do not write code yet — just analysis.
```

Good litmus test. If the coder model gives you a coherent answer,
you're ready. If it spits garbage, drop back to the 32B model
(`--model ollama/qwen2.5-coder:32b`) or just use Claude for this one.
