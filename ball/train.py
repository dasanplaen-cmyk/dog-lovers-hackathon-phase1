"""Train the ball-from-players model on external games only.

    python -m ball.train ball/configs/v1.yaml
    python -m ball.train ball/configs/v1.yaml --set steps=200 eval_every=100   # quick override
    python -m ball.train --resume latest        # after a crash or reboot: continue where it stopped

Every run is logged by training/runlog.py (config, commit, seed, games, scalar metrics). The
checkpoint with the lowest mean error on hidden frames of the validation games is kept in the
run's weights dir. Test games are scored once at the end, for the record, never for selection.
The model must beat linear interpolation on the validation games to be worth using; both are
reported side by side.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from ball.batching import GameArrays, interpolate_baseline, make_batch  # noqa: E402
from ball.game import Game  # noqa: E402
from ball.model import BallFromPlayers, confidence, gaussian_nll  # noqa: E402
from ball.sources import external_dir  # noqa: E402
from training.runlog import Run, set_seed  # noqa: E402


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def load_games(names: list[str], max_players: int) -> list[GameArrays]:
    games = []
    for name in names:
        if "BAR-ATH" in name:
            raise SystemExit("BAR-ATH is the test game: never train, tune or validate on it")
        games.append(GameArrays(Game.load(os.path.join(external_dir(), name + ".npz")), max_players))
    return games


def to_torch(batch, device):
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return (t(batch.players), t(batch.pmask, torch.bool), t(batch.ball_in),
            t(batch.fmask, torch.bool), t(batch.target), t(batch.hidden, torch.bool))


def model_from_config(cfg: dict) -> BallFromPlayers:
    return BallFromPlayers(d_model=cfg["d_model"], heads=cfg["heads"], set_layers=cfg["set_layers"],
                           time_layers=cfg["time_layers"], window=cfg["window"],
                           dropout=cfg["dropout"])


def fixed_eval_set(games: list[GameArrays], cfg: dict, seed: int, max_windows: int | None):
    """Non-overlapping windows over every segment, gaps drawn once from a fixed seed."""
    starts = [(gi, s) for gi, g in enumerate(games)
              for s in g.window_starts(cfg["window"], cfg["window"])]
    rng = np.random.default_rng(seed)
    if max_windows and len(starts) > max_windows:
        keep = np.sort(rng.choice(len(starts), max_windows, replace=False))
        starts = [starts[i] for i in keep]
    batches = []
    for i in range(0, len(starts), cfg["batch_size"]):
        batches.append(make_batch(games, rng, cfg, train=False,
                                  starts=starts[i:i + cfg["batch_size"]]))
    return batches


def _gap_lengths(hidden: np.ndarray) -> np.ndarray:
    """Length of the hidden run each frame belongs to (0 where visible)."""
    out = np.zeros(hidden.shape, np.int32)
    for k in range(hidden.shape[0]):
        h = hidden[k]
        edges = np.flatnonzero(np.diff(np.r_[0, h.astype(np.int8), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            out[k, a:b] = b - a
    return out


@torch.no_grad()
def evaluate(model, batches, device, prefix: str, radius: float) -> dict:
    model.eval()
    errs, base, confs, gaps, nll_sum, nll_n = [], [], [], [], 0.0, 0
    for b in batches:
        players, pmask, ball_in, fmask, target, hidden = to_torch(b, device)
        mu, logvar = model(players, pmask, ball_in, fmask)
        sel = (hidden & ~torch.isnan(target[..., 0])).cpu().numpy()
        if not sel.any():
            continue
        nll = gaussian_nll(mu.float(), logvar.float(), target, hidden.float())
        nll_sum += float(nll) * int(sel.sum())
        nll_n += int(sel.sum())
        e = torch.linalg.norm(mu.float() - torch.nan_to_num(target), dim=-1).cpu().numpy()
        interp = interpolate_baseline(b)
        eb = np.linalg.norm(interp - b.target, axis=-1)
        c = confidence(logvar.float(), radius).cpu().numpy()
        g = _gap_lengths(b.hidden)
        errs.append(e[sel])
        base.append(eb[sel])
        confs.append(c[sel])
        gaps.append(g[sel])
    model.train()
    if not errs:
        return {}
    e, eb, c, g = (np.concatenate(x) for x in (errs, base, confs, gaps))
    eb_known = ~np.isnan(eb)
    m = {"frames": int(e.size), "nll": nll_sum / max(nll_n, 1),
         "err_mean_m": float(e.mean()), "err_median_m": float(np.median(e)),
         "err_p90_m": float(np.percentile(e, 90)),
         "within_2m": float((e < 2).mean()), "within_5m": float((e < 5).mean()),
         # Same frames as the interpolation (some ball visible in the window): the fair comparison.
         "err_mean_m_where_interp": float(e[eb_known].mean()) if eb_known.any() else float("nan"),
         "interp_err_mean_m": float(eb[eb_known].mean()) if eb_known.any() else float("nan"),
         "interp_within_2m": float((eb[eb_known] < 2).mean()) if eb_known.any() else float("nan"),
         "conf_mean": float(c.mean()), "calib_gap": float(c.mean() - (e < radius).mean())}
    for name, lo, hi in (("short", 1, 10), ("mid", 11, 50), ("long", 51, 10**9)):
        k = (g >= lo) & (g <= hi)
        if k.any():
            m["err_mean_m_" + name] = float(e[k].mean())
            kb = k & eb_known
            if kb.any():
                m["interp_err_mean_m_" + name] = float(eb[kb].mean())
    return {prefix + k: v for k, v in m.items()}


_WORKER_GAMES = None  # set before the worker pool forks, inherited by the workers


def _train_batch(args):
    seed, step, cfg = args
    # One generator per step: the batch for step s is the same whoever builds it, so a resumed
    # run sees exactly the batches the original would have seen.
    return make_batch(_WORKER_GAMES, np.random.default_rng([seed, step]), cfg, train=True)


def train_batches(games, cfg, first_step, steps):
    """Batches first_step..steps, built ahead of time by worker processes (num_workers > 0)."""
    global _WORKER_GAMES
    _WORKER_GAMES = games
    workers = int(cfg.get("num_workers", 0))
    if workers <= 0:
        for step in range(first_step, steps + 1):
            yield step, _train_batch((cfg["seed"], step, cfg))
        return
    import collections
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ahead = collections.deque()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        nxt = first_step
        for step in range(first_step, steps + 1):
            while nxt <= steps and len(ahead) < 2 * workers:
                ahead.append((nxt, pool.submit(_train_batch, (cfg["seed"], nxt, cfg))))
                nxt += 1
            s, fut = ahead.popleft()
            yield s, fut.result()


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)   # a crash mid-write never leaves a broken checkpoint


def latest_resumable() -> str:
    from training.runlog import RUNS_DIR, work_dir
    runs = sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    for rid in runs:
        if "ball" in rid and (work_dir() / "weights" / rid / "last.pt").is_file() and \
                not (RUNS_DIR / rid / "metrics.json").is_file():
            return rid
    raise SystemExit("no unfinished run with a last.pt checkpoint to resume")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?", help="YAML config (not needed with --resume)")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run, or 'latest'")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.resume:
        run = Run.reopen(latest_resumable() if args.resume == "latest" else args.resume)
        cfg = run.config
        state = torch.load(run.weights_dir / "last.pt", map_location=device, weights_only=False)
    else:
        if not args.config:
            ap.error("a config is required unless --resume is given")
        cfg = load_config(args.config, args.set)
        state = None

    set_seed(cfg["seed"])
    train_games = load_games(cfg["train_games"], cfg["max_players"])
    val_games = load_games(cfg["val_games"], cfg["max_players"])
    if state is None:
        run = Run(cfg["name"], config=cfg, seed=cfg["seed"], val_games=cfg["val_games"],
                  train_games=cfg["train_games"])
    print("run {} on {} | train {} | val {}".format(
        run.id, device, [g.game.name for g in train_games], [g.game.name for g in val_games]))

    model = model_from_config(cfg).to(device)
    print("parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm, steps = cfg["warmup_steps"], cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / steps))))
    best, best_step, first = float("inf"), -1, 1
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, best_step, first = state["best"], state["best_step"], state["step"] + 1
        print("resumed at step {} (best {:.3f} at step {})".format(state["step"], best, best_step))
    val_batches = fixed_eval_set(val_games, cfg, cfg["seed"] + 1, cfg.get("eval_max_windows"))
    use_amp = device.type == "cuda" and cfg.get("bf16", True)
    ckpt_every = cfg.get("ckpt_every", 500)

    t0 = time.time()
    for step, b in train_batches(train_games, cfg, first, steps):
        players, pmask, ball_in, fmask, target, hidden = to_torch(b, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            mu, logvar = model(players, pmask, ball_in, fmask)
        weight = hidden.float() + cfg["visible_weight"] * (~hidden).float()
        loss = gaussian_nll(mu.float(), logvar.float(), target, weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        if step % cfg["log_every"] == 0:
            print("step {:>6} loss {:.4f} lr {:.2e} {:.1f} steps/s".format(
                step, loss.item(), sched.get_last_lr()[0], (step - first + 1) / (time.time() - t0)))
        if step % cfg["eval_every"] == 0 or step == steps:
            m = evaluate(model, val_batches, device, "val_", cfg["conf_radius_m"])
            run.log(step=step, train_loss=loss.item(), **m)
            print("eval  {:>6} ".format(step) + " ".join(
                "{}={:.3f}".format(k[4:], v) for k, v in m.items()
                if k in ("val_err_mean_m", "val_err_mean_m_where_interp", "val_interp_err_mean_m",
                         "val_within_2m", "val_calib_gap", "val_err_mean_m_long")))
            if m and m["val_err_mean_m"] < best:
                best, best_step = m["val_err_mean_m"], step
                save_atomic({"model": model.state_dict(), "config": cfg, "step": step,
                             "run": run.id}, run.weights_dir / "best.pt")
        if step % ckpt_every == 0 or step % cfg["eval_every"] == 0:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "step": step, "best": best,
                         "best_step": best_step, "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                        run.weights_dir / "last.pt")

    final = {"best_step": best_step, "best_val_err_mean_m": best}
    ckpt = torch.load(run.weights_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final.update(evaluate(model, val_batches, device, "val_", cfg["conf_radius_m"]))
    if cfg.get("test_games"):
        test = load_games(cfg["test_games"], cfg["max_players"])
        final.update(evaluate(model, fixed_eval_set(test, cfg, cfg["seed"] + 2,
                                                    cfg.get("eval_max_windows")),
                              device, "test_", cfg["conf_radius_m"]))
    run.finish(**final)
    print("done: {} (best step {}), weights {}".format(run.dir, best_step, run.weights_dir / "best.pt"))


if __name__ == "__main__":
    main()
