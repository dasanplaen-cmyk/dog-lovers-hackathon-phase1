"""Train the possession model on external games with per-frame H/A/D labels (DFL).

    python -m models.possession.train models/possession/configs/v1.yaml
    python -m models.possession.train --resume latest

Validation runs the model over each whole validation game with the ball hidden on a fixed set of
simulated gaps, then scores it with the official evaluation/possession_metrics.py (spellF1,
macroF1) after two fixed decoders: argmax, and a 3-state Viterbi with one switch penalty. The
Viterbi here is only a stable yardstick for choosing checkpoints; the real decoder is Ying's
(decode/). The checkpoint with the best validation spellF1 (Viterbi) is kept.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))

import torch  # noqa: E402

import possession_metrics as pm  # noqa: E402
from ball.game import Game  # noqa: E402
from ball.sources import external_dir  # noqa: E402
from ball.train import load_config, save_atomic  # noqa: E402
from models.possession.data import (PossessionGame, direction_report, make_batch,  # noqa: E402
                                   masked_copy, window)
from models.possession.model import PossessionNet, loss_fn  # noqa: E402
from training.runlog import RUNS_DIR, Run, set_seed, work_dir  # noqa: E402


def load_games(names, max_players, require_labels=True, scoring=False):
    """scoring=True (evaluate.py) also accepts COMPARE_ONLY games: they may be scored, never
    used for training or selection. Held-out games are gated by evaluate.py's --final."""
    out = []
    for name in names:
        if "BAR-ATH" in name:
            raise SystemExit("BAR-ATH is the test game: never train, tune or validate on it")
        import eval.external as split
        n = name.upper()
        if getattr(split, "DROPPED", "METRICA") in n or (not scoring and (
                n in split.HELD_OUT or n in getattr(split, "COMPARE_ONLY", set()))):
            raise SystemExit("refused: {} may not be used for training or selection".format(name))
        g = Game.load(os.path.join(external_dir(), name + ".npz"))
        if require_labels and not (g.label >= 0).any():
            raise SystemExit("{} has no labels".format(name))
        out.append(g)
    return out


def model_from_config(cfg):
    return PossessionNet(d_model=cfg["d_model"], heads=cfg["heads"], set_layers=cfg["set_layers"],
                         time_layers=cfg["time_layers"], window=cfg["window"], dropout=cfg["dropout"],
                         ball_est=bool(cfg.get("ball_weights")))


def ball_model_for(cfg, device):
    """(ball model, its config) when the config feeds the ball model's estimate, else None."""
    if not cfg.get("ball_weights"):
        return None
    from models.possession.ballest import load_ball_model
    path = cfg["ball_weights"]
    if path.startswith("release:"):      # pinned: manifest entry, sha256 checked on every load
        from inference.weights import local_path
        path = local_path(path.split(":", 1)[1])
    elif not os.path.isabs(path):
        path = os.path.join(REPO, path)
    return load_ball_model(path, device)


def forward(model, ball, players, pmask, ball_in, fmask):
    est = None
    if ball is not None:
        from models.possession.ballest import ball_estimates
        est = ball_estimates(ball[0], ball[1], players, pmask, ball_in, fmask)
    return model(players, pmask, ball_in, fmask, est)


def tt(a, device, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


@torch.no_grad()
def predict_game(model, pg: PossessionGame, cfg: dict, device, stride=None, ball=None):
    """Per-frame state probabilities [T,3] and change probability [T] for a whole game, from
    overlapping windows weighted toward their centre. Uses the ball exactly as the game has it."""
    model.eval()
    W = cfg["window"]
    stride = stride or W // 2
    T = pg.game.n_frames
    acc_w, acc_p, acc_c = np.zeros(T), np.zeros((T, 3)), np.zeros(T)
    tri = 1.0 - np.abs(np.arange(W) - (W - 1) / 2) / ((W + 1) / 2)
    starts = pg.window_starts(W, stride)
    bs = cfg["batch_size"]
    dt = next(model.parameters()).dtype   # float64 at inference on CPU (cross-machine reproducibility)
    for i in range(0, len(starts), bs):
        chunk = starts[i:i + bs]
        parts = [window(pg, s, cfg, None, train=False, hide=False) for s in chunk]
        players, pmask, ball_in, fmask = (np.stack([p[j] for p in parts]) for j in range(4))
        logits, ch = forward(model, ball, tt(players, device, dt), tt(pmask, device, torch.bool),
                             tt(ball_in, device, dt), tt(fmask, device, torch.bool))
        prob = torch.softmax(logits.double(), -1).cpu().numpy()
        pc = torch.sigmoid(ch.double()).cpu().numpy()
        for k, s in enumerate(chunk):
            n = int((~fmask[k]).sum())
            w = tri[:n]
            acc_w[s:s + n] += w
            acc_p[s:s + n] += w[:, None] * prob[k, :n]
            acc_c[s:s + n] += w * pc[k, :n]
    model.train()
    p = acc_p / np.maximum(acc_w, 1e-9)[:, None]
    c = acc_c / np.maximum(acc_w, 1e-9)
    p /= p.sum(axis=1, keepdims=True)
    c[0] = 0.0
    return p, c


def viterbi(p: np.ndarray, penalty: float) -> np.ndarray:
    """Most likely H/A/D path with a constant cost per state switch."""
    logp = np.log(np.clip(p, 1e-9, 1.0))
    T = len(logp)
    back = np.zeros((T, 3), np.int8)
    score = logp[0].copy()
    for t in range(1, T):
        best_prev = int(score.argmax())
        stay = score
        switch = score[best_prev] - penalty
        choose_switch = switch > stay
        back[t] = np.where(choose_switch, best_prev, np.arange(3))
        score = np.where(choose_switch, switch, stay) + logp[t]
    path = np.zeros(T, np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def target_labels(game: Game) -> tuple[np.ndarray, bool]:
    """Ground truth for scoring an external game, as eval/external.py builds it: X at each period
    start (no possession across half-time) and, when available, align_control (a change of team
    in live play needs about 2 s on the ball, the challenge's written proxy). Returns (codes,
    aligned) so the run records which target it was scored on."""
    gt = np.where(game.label < 0, pm.X, game.label).astype(np.int64)
    gt[np.flatnonzero(np.diff(game.period)) + 1] = pm.X
    try:
        from eval.external import align_control
    except ImportError:          # before Ying's #16 is on this branch
        return gt, False
    return align_control(gt, game.fps), True


def align_training_labels(game: Game) -> int:
    """Apply eval/external.py's align_control to a training game's labels, period by period:
    a change of team in live play that keeps the ball under about 2 s is not a possession
    change. The same rule builds the validation target, so the model learns what it is scored
    on. Returns the number of frames whose label changed."""
    from eval.external import align_control
    codes = np.where(game.label < 0, pm.X, game.label).astype(np.int64)
    out = codes.copy()
    for p in np.unique(game.period):
        rows = np.flatnonzero(game.period == p)
        out[rows] = align_control(codes[rows], game.fps)
    new = np.where(out == pm.X, -1, out).astype(game.label.dtype)
    changed = int((new != game.label).sum())
    game.label = new
    return changed


def score(game: Game, p: np.ndarray, penalty: float) -> dict:
    gt, aligned = target_labels(game)
    out = {"labels_aligned": float(aligned)}
    for name, pred in (("argmax", p.argmax(1)), ("viterbi", viterbi(p, penalty))):
        r = pm.evaluate(gt, pred.astype(np.int64), bootstrap_n=0)
        out["spellF1_" + name] = float(r["spell_f1_mean"])
        out["macroF1_" + name] = float(r["macro_f1"])
    return out


def evaluate(model, val, cfg, device, prefix, ball=None):
    if not val:
        return {}
    rows = [score(pg.game, predict_game(model, pg, cfg, device, ball=ball)[0], cfg["viterbi_penalty"])
            for pg in val]
    return {prefix + k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def selection_score(m: dict) -> float:
    """Checkpoint choice: mean Viterbi spellF1 over the simulated-gap validation games and, when
    configured, the real-gap validation games (external, with labels)."""
    keys = [k for k in ("val_spellF1_viterbi", "realval_spellF1_viterbi") if k in m]
    return float(np.mean([m[k] for k in keys]))


_WORKER_GAMES = None


def _batch(args):
    seed, step, cfg = args
    return make_batch(_WORKER_GAMES, np.random.default_rng([seed, step]), cfg)


def batches(games, cfg, first, steps):
    global _WORKER_GAMES
    _WORKER_GAMES = games
    workers = int(cfg.get("num_workers", 0))
    if workers <= 0:
        for s in range(first, steps + 1):
            yield s, _batch((cfg["seed"], s, cfg))
        return
    import collections
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ahead = collections.deque()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        nxt = first
        for _ in range(first, steps + 1):
            while nxt <= steps and len(ahead) < 2 * workers:
                ahead.append((nxt, pool.submit(_batch, (cfg["seed"], nxt, cfg))))
                nxt += 1
            s, f = ahead.popleft()
            yield s, f.result()


def latest_resumable(name_part="possession"):
    for rid in sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True):
        if name_part in rid and (work_dir() / "weights" / rid / "last.pt").is_file() \
                and not (RUNS_DIR / rid / "metrics.json").is_file():
            return rid
    raise SystemExit("no unfinished possession run with a last.pt to resume")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", metavar="RUN_ID")
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state = None
    if args.resume:
        run = Run.reopen(latest_resumable() if args.resume == "latest" else args.resume)
        cfg = run.config
        state = torch.load(run.weights_dir / "last.pt", map_location=device, weights_only=False)
    else:
        if not args.config:
            ap.error("a config is required unless --resume is given")
        cfg = load_config(args.config, args.set)

    set_seed(cfg["seed"])
    train_games = load_games(cfg["train_games"], cfg["max_players"])
    if cfg.get("align_labels"):     # v3: learn the challenge's controlled possession, not any touch
        n_changed = [align_training_labels(g) for g in train_games]
        print("aligned training labels (control >= 2 s): {} frames changed".format(sum(n_changed)))
    train = [PossessionGame(g, cfg["max_players"]) for g in train_games]
    val = [PossessionGame(masked_copy(g, cfg, cfg["seed"] + 1 + i), cfg["max_players"])
           for i, g in enumerate(load_games(cfg["val_games"], cfg["max_players"]))]
    real_val = [PossessionGame(g, cfg["max_players"])
                for g in load_games(cfg.get("real_val_games", []), cfg["max_players"])]
    if state is None:
        run = Run(cfg["name"], config=cfg, seed=cfg["seed"], val_games=cfg["val_games"],
                  train_games=cfg["train_games"])
    print("run {} on {}".format(run.id, device))
    for pg in train + val + real_val:
        print("  {} home attack sign per half {} | goalkeepers {}".format(
            pg.game.name, pg.half_signs, direction_report(pg.game)))

    ball = ball_model_for(cfg, device)
    model = model_from_config(cfg).to(device)
    print("parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warm, steps = cfg["warmup_steps"], cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / steps))))
    best, best_step, first = -1.0, -1, 1
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        best, best_step, first = state["best"], state["best_step"], state["step"] + 1
        print("resumed at step {} (best {:.4f} at step {})".format(state["step"], best, best_step))
    use_amp = device.type == "cuda" and cfg.get("bf16", True)

    t0 = time.time()
    for step, b in batches(train, cfg, first, steps):
        players, pmask, ball_in, fmask, label, change = b
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, ch = forward(model, ball, tt(players, device), tt(pmask, device, torch.bool),
                                 tt(ball_in, device), tt(fmask, device, torch.bool))
        loss, ce, bce = loss_fn(logits, ch, tt(label, device, torch.long),
                                tt(change, device, torch.long), cfg["change_weight"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            print("step {:>6} loss {:.4f} (ce {:.4f} change {:.4f}) {:.1f} steps/s".format(
                step, loss.item(), ce.item(), bce.item(), (step - first + 1) / (time.time() - t0)))
        if step % cfg["eval_every"] == 0 or step == steps:
            m = evaluate(model, val, cfg, device, "val_", ball)
            m.update(evaluate(model, real_val, cfg, device, "realval_", ball))
            m["select"] = selection_score(m)
            run.log(step=step, train_loss=loss.item(), **m)
            print("eval  {:>6} ".format(step) + " ".join("{}={:.4f}".format(k, v) for k, v in m.items()))
            if m["select"] > best:
                best, best_step = m["select"], step
                save_atomic({"model": model.state_dict(), "config": cfg, "step": step, "run": run.id},
                            run.weights_dir / "best.pt")
        if step % cfg.get("ckpt_every", 500) == 0 or step % cfg["eval_every"] == 0:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "step": step, "best": best,
                         "best_step": best_step, "torch_rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                        run.weights_dir / "last.pt")

    ckpt = torch.load(run.weights_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    final = {"best_step": best_step}
    final.update(evaluate(model, val, cfg, device, "val_", ball))
    final.update(evaluate(model, real_val, cfg, device, "realval_", ball))
    final["select"] = selection_score(final)
    run.finish(**final)
    print("done: {} (best step {}), weights {}".format(run.dir, best_step, run.weights_dir / "best.pt"))


if __name__ == "__main__":
    main()
