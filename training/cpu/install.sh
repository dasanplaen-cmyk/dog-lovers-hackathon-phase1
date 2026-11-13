#!/usr/bin/env bash
# CPU-only install for any x86_64 or aarch64 Linux machine (backup machine, clean reproduction).
# Same Python and package versions as the GX10 reference (training/gx10/requirements-gx10.lock),
# torch from the CPU wheel index. Inference for the submission runs on CPU anyway.
#
#   export MTAH_DATA_DIR=/path/to/mtah2027      # the data bundle, outside this repo
#   training/cpu/install.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install it (https://docs.astral.sh/uv/getting-started/installation/) and re-run." >&2
  exit 1
}
if [ -n "${MTAH_DATA_DIR:-}" ]; then
  python3 training/gx10/deny_data.py "$(realpath "$MTAH_DATA_DIR")"   # in case a coding agent runs here
  EXCLUDE="$(git rev-parse --git-path info/exclude)"
  grep -qxF '.claude/settings.local.json' "$EXCLUDE" 2>/dev/null || echo '.claude/settings.local.json' >> "$EXCLUDE"
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r training/cpu/requirements-cpu.txt
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"
{
  echo "# Environment written by training/cpu/install.sh on $(uname -m)."
  echo "# python: $(.venv/bin/python -c 'import sys; print(sys.version.split()[0])')"
  uv pip freeze --python .venv/bin/python
} > "training/cpu/requirements-cpu-$(uname -m).lock"
MTAH_CPU_ONLY=1 .venv/bin/python training/gx10/check_env.py
