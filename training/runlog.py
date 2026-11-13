"""Every training run leaves a trace: config, commit, seed, validation games and aggregate metrics.

    from training.runlog import Run, set_seed

    set_seed(cfg["seed"])
    run = Run("seq_v1", config=cfg, seed=cfg["seed"], val_games=["DFL-J03WMX", "DFL-J03WN1"])
    ...
    run.log(epoch=3, val_spellF1=0.612, val_macroF1=0.701)
    torch.save(model.state_dict(), run.weights_dir / "model.pt")
    run.finish(val_spellF1=0.615, val_macroF1=0.704)

training/runs/<UTC time>_<commit>_<name>/ holds run.json, metrics.jsonl and metrics.json. That
folder is small and goes to git. Weights go to run.weights_dir, under the gitignored work dir
(MTAH_WORK_DIR, default data/work/); publish the final ones as a GitHub Release with their sha256.

Metrics must be scalars. Arrays, frames or per-frame values are refused on purpose: nothing
frame-level about any game belongs in git, and nothing at all about BAR-ATH belongs in training.
"""

from __future__ import annotations

import datetime as _dt
import json
import numbers
import os
import platform
import random
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("MTAH_RUNS_DIR") or REPO / "training" / "runs")  # env: smoke tests


def work_dir() -> Path:
    """Where large or match-derived outputs go: gitignored, and denied to coding agents."""
    return Path(os.environ.get("MTAH_WORK_DIR") or REPO / "data" / "work")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch; ask torch for deterministic kernels where it has them."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _scalars(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or v is None or isinstance(v, str):
            out[k] = v
        elif isinstance(v, numbers.Number):
            out[k] = float(v) if not isinstance(v, numbers.Integral) else int(v)
        elif hasattr(v, "item") and getattr(v, "ndim", 1) == 0:  # numpy / torch 0-d scalars
            out[k] = v.item()
        else:
            raise TypeError("metric {!r} is not a scalar ({}); only aggregates are logged".format(
                k, type(v).__name__))
    return out


class Run:
    def __init__(self, name: str, config: dict, seed: int, val_games: list[str],
                 train_games: list[str] | None = None):
        if any("BAR-ATH" in g for g in list(val_games) + list(train_games or [])):
            raise ValueError("BAR-ATH is the test game: never train, tune or validate on it")
        self.git = git_state()
        now = _dt.datetime.now(_dt.timezone.utc)
        self.id = "{}_{}_{}".format(now.strftime("%Y%m%dT%H%M%SZ"), self.git["commit"][:7], name)
        self.dir = RUNS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.weights_dir = work_dir() / "weights" / self.id
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self._write("run.json", {
            "id": self.id, "name": name, "seed": seed, "config": config,
            "train_games": train_games, "val_games": val_games, "git": self.git,
            "host": platform.node(), "machine": platform.machine(),
            "started_utc": now.isoformat(timespec="seconds"),
        })
        if self.git["dirty"]:
            print("runlog: WARNING working tree has uncommitted changes; commit before a run you keep")

    @classmethod
    def reopen(cls, run_id: str) -> "Run":
        """Continue an interrupted run in its own folder; records when and from which commit."""
        self = cls.__new__(cls)
        self.id = run_id
        self.dir = RUNS_DIR / run_id
        if not (self.dir / "run.json").is_file():
            raise FileNotFoundError("no run {} in {}".format(run_id, RUNS_DIR))
        self.weights_dir = work_dir() / "weights" / run_id
        with open(self.dir / "run.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.config, self.git = meta["config"], meta["git"]
        meta.setdefault("resumed", []).append({
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "git": git_state()})
        self._write("run.json", meta)
        return self

    def _write(self, fname: str, obj: dict) -> None:
        with open(self.dir / fname, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=str)
            f.write("\n")

    def log(self, **metrics) -> None:
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(_scalars(metrics), sort_keys=True) + "\n")

    def finish(self, **final_metrics) -> None:
        final = _scalars(final_metrics)
        final["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self._write("metrics.json", final)
