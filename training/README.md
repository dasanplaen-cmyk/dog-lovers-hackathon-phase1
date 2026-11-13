# Training and inference on the GX10

Owner: Chanti. Everything that trains or runs a model happens on the GX10 (aarch64, Blackwell GPU),
which is the only machine that holds the match data. Training and tuning use **external games
only** (`external_data/SOURCES.md`); BAR-ATH is only ever *input* to inference.

## Two places, one git history

| | cloud session (Claude Code on the web / app) | GX10 |
|---|---|---|
| what it does | writes code, reviews, board, issues, PRs | runs setup, training, inference |
| match data | **never** (it has no copy, and must not get one) | `MTAH_DATA_DIR`, outside the repo |
| how to reach it | any device | SSH + `tmux`, or `claude remote-control` in the repo folder (then it shows up in the Claude app on any device) |

The loop: code is pushed from either side to the working branch → `git pull` on the GX10 → run
inside `tmux` → the run leaves `training/runs/<id>/` (config, commit, seed, aggregate metrics) →
commit that folder and push → the other side reads the numbers from git. Only aggregates travel.

When both sessions post on the board, sign them apart: `[Chanti's Claude (cloud)]` and
`[Chanti's Claude (GX10)]`.

## First time on the GX10

```bash
git clone https://github.com/dasanplaen-cmyk/dog-lovers-hackathon.git && cd dog-lovers-hackathon
rsync -a <laptop>:/path/to/mtah2027/ /srv/mtah2027/          # bundle over the LAN, never via cloud
export MTAH_DATA_DIR=/srv/mtah2027
training/gx10/bootstrap.sh                                     # deny rules, .venv, lock file, check
git add training/gx10/requirements-gx10.lock && git commit -m "GX10 environment lock"
.venv/bin/python starter/baseline.py                           # rewrites evaluation/results/baseline_*.csv
git status --short evaluation/results/                         # empty = byte-identical to the committed ones
.venv/bin/python evaluation/validate_submission.py evaluation/results/baseline_raw.csv
```

The baseline CSVs are committed upstream, so an empty `git status` proves data, paths, frame
alignment and versions are right on the GX10 without anyone looking at a single row.

`bootstrap.sh` installs torch from the CUDA 13 wheel index (`TORCH_INDEX` overrides it). If
`check_env.py` reports that torch cannot run a kernel on the GPU, use NVIDIA's NGC PyTorch
container instead, pinned by tag, and record the tag in the lock file.

## Every run

```python
from training.runlog import Run, set_seed
set_seed(cfg["seed"])
run = Run("seq_v1", config=cfg, seed=cfg["seed"], val_games=[...], train_games=[...])
run.log(epoch=e, val_spellF1=..., val_macroF1=...)   # scalars only
run.finish(val_spellF1=..., val_macroF1=...)
```

- Launch inside `tmux new -s train` so an SSH drop does not kill it.
- Commit before a run you intend to keep; `Run` warns on a dirty tree and refuses BAR-ATH as a
  training or validation game.
- Weights go to `run.weights_dir` (gitignored work dir). Final weights: GitHub Release + sha256 in
  the repo + a download command, so the organisers can rerun everything.
- Inference output for any game follows `inference/README.md`.
