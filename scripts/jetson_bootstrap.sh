#!/usr/bin/env bash
# One-command Jetson AGX Orin bring-up. Run from the repo root:
#
#   bash scripts/jetson_bootstrap.sh
#
# What it does, in order:
#   1. Sanity-check we're actually on a Jetson.
#   2. Prompt for the SD-card device + mount point (defaults given).
#   3. Mount the SD card and add it to /etc/fstab so it survives reboots.
#   4. Create tradebot-data subdirs (logs, data_cache, checkpoints, models).
#   5. Run the existing deploy/jetson/setup.sh (installs venv, CUDA
#      llama-cpp, Jetson PyTorch, downloads the 7B news-classifier model).
#   6. Download Llama-3.1-8B Q4 GGUF  → $SD/models/  (review brain)
#      Download Llama-3.1-70B Q4 GGUF → $SD/models/ (strategy auditor)
#      BOTH URLS CAN BE OVERRIDDEN via env so you can use a different
#      quantization or a mirror close to your region.
#   7. Write/update .env with TRADEBOT_DATA_ROOT + LLM_* paths.
#   8. Install systemd --user units for watchdog + dashboard.
#   9. Run scripts/doctor.sh for a green/red readiness check.
#
# Safe to re-run. Every step is idempotent — already-mounted SD, already-
# downloaded models, already-installed systemd units are detected and
# skipped.
set -u   # (no -e: we want the doctor check at the end to run even if some earlier step warned)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Model download URLs. Override with:
#   LLM_8B_URL=<your-mirror> LLM_70B_URL=<your-mirror> bash scripts/jetson_bootstrap.sh
: "${LLM_8B_URL:=https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
: "${LLM_70B_URL:=https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf}"

SD_DEFAULT_DEV="/dev/mmcblk1p1"
SD_DEFAULT_MNT="/media/$USER/tradebot-data"

step() { printf '\n\033[1;32m[bootstrap %d/%d]\033[0m %s\n' "$1" "$2" "$3"; }

# ------------------------------------------------------------------ step 1
step 1 9 "Jetson sanity check"
if [ ! -f /etc/nv_tegra_release ]; then
  echo "  [!] /etc/nv_tegra_release missing — this does not look like a Jetson."
  echo "      If you want to continue anyway, set FORCE_NOT_JETSON=1 and re-run."
  if [ "${FORCE_NOT_JETSON:-0}" != "1" ]; then exit 2; fi
else
  head -1 /etc/nv_tegra_release
fi

# ------------------------------------------------------------------ step 2
step 2 9 "SD card mount"
read -r -p "SD-card partition device [${SD_DEFAULT_DEV}]: " SD_DEV
SD_DEV="${SD_DEV:-$SD_DEFAULT_DEV}"
read -r -p "Mount point [${SD_DEFAULT_MNT}]: " SD_MNT
SD_MNT="${SD_MNT:-$SD_DEFAULT_MNT}"
if ! lsblk "$SD_DEV" >/dev/null 2>&1; then
  echo "  [!] $SD_DEV not found. Insert the SD card and re-run."
  exit 2
fi
sudo mkdir -p "$SD_MNT"
if mountpoint -q "$SD_MNT"; then
  echo "  already mounted at $SD_MNT"
else
  # Detect existing filesystem. If none, bail out and ask the user to
  # format themselves — we won't format a device without explicit consent.
  FSTYPE="$(lsblk -no FSTYPE "$SD_DEV" 2>/dev/null)"
  if [ -z "$FSTYPE" ]; then
    echo "  [!] $SD_DEV has no filesystem. Format manually first, e.g.:"
    echo "      sudo mkfs.ext4 -L tradebot-data $SD_DEV"
    exit 3
  fi
  echo "  filesystem: $FSTYPE"
  sudo mount "$SD_DEV" "$SD_MNT"
fi
sudo chown -R "$USER:$USER" "$SD_MNT"

# ------------------------------------------------------------------ step 3
step 3 9 "fstab entry (survives reboot)"
UUID="$(lsblk -no UUID "$SD_DEV" 2>/dev/null)"
if [ -n "$UUID" ]; then
  if grep -q "$UUID" /etc/fstab; then
    echo "  fstab entry for UUID=$UUID already present"
  else
    echo "UUID=$UUID  $SD_MNT  auto  defaults,nofail,x-systemd.automount  0  2" \
      | sudo tee -a /etc/fstab
    echo "  appended fstab entry (nofail + x-systemd.automount so a missing SD doesn't block boot)"
  fi
else
  echo "  [!] could not read UUID; skipping fstab (SD will need to be re-mounted manually after reboot)"
fi

# ------------------------------------------------------------------ step 4
step 4 9 "Create subdirs on SD"
for sub in logs data_cache checkpoints models; do
  mkdir -p "$SD_MNT/$sub"
done
ls -la "$SD_MNT"

# ------------------------------------------------------------------ step 5
step 5 9 "Run deploy/jetson/setup.sh"
if [ -f "$REPO/deploy/jetson/setup.sh" ]; then
  bash "$REPO/deploy/jetson/setup.sh"
else
  echo "  [!] deploy/jetson/setup.sh missing; skipping base bootstrap."
fi

# ------------------------------------------------------------------ step 6
step 6 9 "Check / download Llama 8B + 70B GGUFs"
# Rule: NEVER download a model that's already on disk > 1 MB.
# SKIP_LLM_DL=1 forces skip even if files are absent (useful if you're
# manually copying models in parallel).
_has_model() {
  # $1 = filesystem path
  [ -f "$1" ] && [ "$(stat -c%s "$1" 2>/dev/null || echo 0)" -gt 1000000 ]
}

# Accept common alternate filenames (e.g. Q4_K_M, Instruct-Q4_K_M)
# and symlink to the canonical names the bot expects. Saves the user
# from having to rename a model they already downloaded.
_find_and_link() {
  # $1 = dir, $2 = glob pattern (match anything *8b*q4*.gguf), $3 = canonical target name
  local dir="$1" pattern="$2" canonical="$3"
  if _has_model "$dir/$canonical"; then
    return 0
  fi
  # shellcheck disable=SC2086
  for candidate in $dir/$pattern; do
    [ -e "$candidate" ] || continue
    if _has_model "$candidate"; then
      echo "  found existing model: $(basename "$candidate") — linking as $canonical"
      ln -sf "$(basename "$candidate")" "$dir/$canonical"
      return 0
    fi
  done
  return 1
}

MODEL_DIR="$SD_MNT/models"
mkdir -p "$MODEL_DIR"

# Auto-discover existing models with fuzzy names before deciding to download.
_find_and_link "$MODEL_DIR" "*[8B]*[qQ]4*.gguf" "llama-3.1-8b-q4.gguf"   || true
_find_and_link "$MODEL_DIR" "*[8B]*[qQ]5*.gguf" "llama-3.1-8b-q4.gguf"   || true   # Q5 works too
_find_and_link "$MODEL_DIR" "*[7]0[bB]*[qQ]4*.gguf" "llama-3.1-70b-q4.gguf" || true

if [ "${SKIP_LLM_DL:-0}" = "1" ]; then
  echo "  SKIP_LLM_DL=1 — skipping downloads."
else
  if _has_model "$MODEL_DIR/llama-3.1-8b-q4.gguf"; then
    echo "  [ok] 8B model present: $(ls -lh "$MODEL_DIR/llama-3.1-8b-q4.gguf" | awk '{print $5}')"
  else
    echo "  downloading 8B GGUF ($(basename "$LLM_8B_URL"))..."
    wget --continue -O "$MODEL_DIR/llama-3.1-8b-q4.gguf" "$LLM_8B_URL" || \
      echo "  [!] 8B download failed — bot still runs with LLM brain disabled."
  fi
  if _has_model "$MODEL_DIR/llama-3.1-70b-q4.gguf"; then
    echo "  [ok] 70B model present: $(ls -lh "$MODEL_DIR/llama-3.1-70b-q4.gguf" | awk '{print $5}')"
  else
    echo "  downloading 70B GGUF ($(basename "$LLM_70B_URL"))..."
    wget --continue -O "$MODEL_DIR/llama-3.1-70b-q4.gguf" "$LLM_70B_URL" || \
      echo "  [!] 70B download failed — strategy auditor will be unavailable."
  fi
fi

# ------------------------------------------------------------------ step 7
step 7 9 "Wire TRADEBOT_DATA_ROOT + LLM paths into .env"
ENV_PATH="$REPO/.env"
if [ ! -f "$ENV_PATH" ]; then
  cp "$REPO/.env.example" "$ENV_PATH"
  echo "  created .env from .env.example"
fi
_upsert() {
  # _upsert <KEY> <VALUE>  — replaces existing or appends
  local key="$1" val="$2"
  if grep -qE "^${key}=" "$ENV_PATH"; then
    sed -i -E "s|^${key}=.*|${key}=${val}|" "$ENV_PATH"
  else
    echo "${key}=${val}" >> "$ENV_PATH"
  fi
}
_upsert "TRADEBOT_DATA_ROOT" "$SD_MNT"
_upsert "LLM_MODEL_PATH"          "$SD_MNT/models/llama-3.1-8b-q4.gguf"
_upsert "LLM_AUDITOR_MODEL_PATH"  "$SD_MNT/models/llama-3.1-70b-q4.gguf"
_upsert "LLM_BRAIN_ENABLED"       "1"
echo "  .env updated (TRADEBOT_DATA_ROOT, LLM_*)"

# ------------------------------------------------------------------ step 8
step 8 9 "Install systemd --user units (watchdog + dashboard)"
if command -v loginctl >/dev/null 2>&1; then
  sudo loginctl enable-linger "$USER" || true
  echo "  linger enabled so services survive logout"
fi
bash "$REPO/scripts/tradebotctl.sh" watchdog-install  || true
bash "$REPO/scripts/tradebotctl.sh" dashboard-install || true

# ------------------------------------------------------------------ step 9
step 9 9 "Doctor"
bash "$REPO/scripts/doctor.sh"

cat <<EOF

============================================================
Jetson bootstrap done. Next steps:

  1. Edit .env to fill in ALPACA + DISCORD credentials if you
     haven't already. TRADEBOT_DATA_ROOT and LLM_* paths are
     already set for you.
  2. Verify the bot is ticking:
       systemctl --user status tradebot-watchdog.service
       journalctl --user -u tradebot-watchdog.service -f
  3. Open the dashboard (from your Mac, through an SSH tunnel):
       ssh -L 8000:127.0.0.1:8000 $USER@<jetson-ip>
       open http://127.0.0.1:8000
  4. Trigger a 70B strategy audit on demand:
       bash scripts/tradebotctl.sh strategy-audit
     Or click the "run 70B audit" button in the dashboard.

Mental model:
  - rules (momentum, ORB, breadth, RSI, S/R, credit-spreads) run every
    tick and propose decisions.
  - 8B brain reviews each proposed decision (~500ms). Can scale
    confidence up or down. Veto only in hard-gate mode.
  - 70B auditor runs nightly + on demand. Reviews the whole config
    against recent trade performance and market state, flags issues,
    scores the setup 0-100.
  - All three are additive; the rules still work with both LLMs off.
============================================================
EOF
