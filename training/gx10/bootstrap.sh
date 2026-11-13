#!/usr/bin/env bash
# One-time setup of the GX10 (aarch64, Blackwell GPU). Run it ON the GX10, from the repo root:
#
#   export MTAH_DATA_DIR=/srv/mtah2027        # the unpacked bundle, OUTSIDE this repo
#   training/gx10/bootstrap.sh
#
# What it does, in order:
#   1. checks the machine (arch, GPU) and that MTAH_DATA_DIR is outside the repo
#   2. adds deny rules for MTAH_DATA_DIR to .claude/settings.local.json (never committed), so a
#      coding agent on this machine cannot read the match data; ./data/** is already denied
#   3. creates .venv with uv (Python 3.12), installs requirements + torch for CUDA on aarch64
#   4. writes the exact versions to training/gx10/requirements-gx10.lock (commit that file)
#   5. runs training/gx10/check_env.py (aggregates only)
#
# Override the torch wheel index with TORCH_INDEX if the default does not support the GPU; the
# fallback is NVIDIA's NGC PyTorch container (see training/README.md).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REPO="$PWD"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
PY_VERSION="${PY_VERSION:-3.12}"

: "${MTAH_DATA_DIR:?set MTAH_DATA_DIR to the unpacked data bundle (outside this repo)}"
DATA_DIR="$(realpath "$MTAH_DATA_DIR")"
case "$DATA_DIR/" in
  "$REPO/"*) echo "MTAH_DATA_DIR ($DATA_DIR) is inside the repo; put the bundle outside it" >&2; exit 1 ;;
esac
[ -d "$DATA_DIR/tactical" ] || { echo "no tactical/ folder under $DATA_DIR (see DATA.md)" >&2; exit 1; }

echo "== machine"
echo "arch: $(uname -m)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "nvidia-smi not found: no GPU visible" >&2
fi

echo "== agent deny rules for the data bundle"
python3 training/gx10/deny_data.py "$DATA_DIR"
EXCLUDE="$(git rev-parse --git-path info/exclude)"
grep -qxF '.claude/settings.local.json' "$EXCLUDE" 2>/dev/null || echo '.claude/settings.local.json' >> "$EXCLUDE"

echo "== python environment"
command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install it (https://docs.astral.sh/uv/getting-started/installation/) and re-run." >&2
  exit 1
}
uv venv --python "$PY_VERSION" .venv
uv pip install --python .venv/bin/python -r requirements.txt -r training/gx10/requirements-gx10.in
uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" torch
{
  echo "# Exact environment on the GX10, written by training/gx10/bootstrap.sh. Commit this file."
  echo "# torch index: $TORCH_INDEX"
  echo "# python: $(.venv/bin/python -c 'import sys; print(sys.version.split()[0])')"
  uv pip freeze --python .venv/bin/python
} > training/gx10/requirements-gx10.lock

echo "== environment check"
MTAH_DATA_DIR="$DATA_DIR" .venv/bin/python training/gx10/check_env.py

cat <<EOF

Done. Add this to ~/.bashrc on the GX10 so every shell (and tmux) sees the bundle:
  export MTAH_DATA_DIR=$DATA_DIR
Then commit training/gx10/requirements-gx10.lock.
EOF
