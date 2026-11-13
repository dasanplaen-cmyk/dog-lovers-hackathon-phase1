"""Rule-based solution: a fallback that needs no trained model, and a reference for the model.

    python -m baseline.rules profile SC-1874553                     # writes baseline/gap_profile.json
    python -m baseline.rules fit DFL-J03WQQ DFL-J03WR9 SC-1874553   # writes baseline/params.json
    python -m baseline.rules probs DFL-J03WQQ DFL-J03WR9 SC-1874553 # <work>/probs_rules/<game>_probs.csv

Who has the ball: the team of the nearest player while a player is close enough to control it,
carried unchanged otherwise (CHALLENGE.md: a loose ball stays with the last team that had it).

Dead ball: a logistic score over a few cues: how fast the players move, the ball off the pitch, the
ball standing still with nobody near it, and how long the ball has been missing. Only these weights
are fitted, on external games. The output is the agreed probabilities format, so the same decoder
turns it into labels (eval/external.py --probs-dir probs_rules, tuned separately).

DFL never loses the ball, so its ball is hidden with gaps drawn from a real tracker's profile, which
do not depend on the labels: the visible and missing runs of SkillCorner's detected ball
(baseline/gap_profile.json, tracking only). Prints aggregates of external games only; for the provided game, path,
hash and pass/fail, and no fitting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from eval.external import COMPARE_ONLY, DROPPED, HELD_OUT, work_dir  # noqa: E402
from inference.probs_format import write_probs  # noqa: E402
from inference.quiet import is_test_game, quiet_report  # noqa: E402
from models.possession.gaps import build_profile, load_profile, renewal_mask  # noqa: E402

PARAMS_PATH = os.path.join(REPO, "baseline", "params.json")
PROFILE_PATH = os.path.join(REPO, "baseline", "gap_profile.json")
CUES = ["bias", "player_speed", "ball_off_pitch", "ball_still", "log_gap_s", "ball_missing"]
INTERP_S = 1.0        # gaps up to this long are interpolated
CONTROL_M = 1.5       # a player this close to the ball controls it
CONF_CONTROL = 0.9    # P(team) while a player of that team controls the ball
CONF_CARRY = 0.7      # P(team) while possession is only carried over
MASK_SEED = 1


def game_path(name: str) -> str:
    folder = "test" if is_test_game(name) else "external"
    return os.path.join(work_dir(), "games", folder, name + ".npz")


def load(name: str) -> dict:
    z = np.load(game_path(name), allow_pickle=False)
    g = {k: z[k] for k in ("pos", "team", "ball", "pitch", "period")}
    g["label"] = z["label"]
    g["fps"] = float(z["fps"])
    g["name"] = name
    if bool(np.isfinite(g["ball"][:, 0]).all()):     # a source that never loses the ball
        hidden = renewal_mask(len(g["ball"]), np.random.default_rng(MASK_SEED), load_profile(PROFILE_PATH))
        g["ball"] = np.where(hidden[:, None], np.nan, g["ball"])
    return g


def forward_fill(a: np.ndarray, start: float) -> np.ndarray:
    last = np.maximum.accumulate(np.where(np.isnan(a), -1, np.arange(len(a))))
    return np.where(last < 0, start, a[np.clip(last, 0, None)])


def interpolate_short_gaps(ball: np.ndarray, seen: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(len(seen))
    tv = t[seen]
    if len(tv) == 0:
        return ball, seen
    prev = forward_fill(np.where(seen, t.astype(float), np.nan), -1e9)
    nxt = len(seen) - 1 - forward_fill(np.where(seen[::-1], t.astype(float), np.nan), -1e9)[::-1]
    known = seen | ((nxt - prev) <= limit)
    out = np.stack([np.interp(t, tv, ball[seen, i]) for i in (0, 1)], axis=1)
    return np.where(known[:, None], out, np.nan), known


def gap_seconds(seen: np.ndarray, fps: float) -> np.ndarray:
    """For every frame, the length of the ball gap it belongs to (0 where the ball is seen)."""
    edges = np.flatnonzero(np.diff(np.r_[1, seen.astype(np.int8), 1]))
    out = np.zeros(len(seen))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = (b - a) / fps
    return out


def mean_speed(pos: np.ndarray, fps: float) -> np.ndarray:
    """Mean player speed (m/s) over one second, smoothed over about one second more."""
    step = max(1, int(fps))
    moved = np.hypot(*np.moveaxis(pos[step:] - pos[:-step], -1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmean(moved, axis=1)
    speed = np.r_[speed, np.full(step, speed[-1])]
    speed = np.nan_to_num(speed, nan=0.0)
    return np.convolve(speed, np.ones(step) / step, mode="same")


def cues(g: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dead-ball cues [T, len(CUES)] and P(home | live) [T]."""
    fps, length, width = g["fps"], float(g["pitch"][0]), float(g["pitch"][1])
    step = max(1, int(fps))
    seen = np.isfinite(g["ball"][:, 0])
    ball, known = interpolate_short_gaps(g["ball"][:, :2], seen, int(INTERP_S * fps))
    d = np.hypot(g["pos"][..., 0] - ball[:, None, 0], g["pos"][..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    home = d[:, g["team"] == 0].min(axis=1)
    away = d[:, g["team"] == 1].min(axis=1)
    nearest = np.minimum(home, away)

    control = np.where(nearest < CONTROL_M, (home < away).astype(float), np.nan)
    carried = forward_fill(control, 0.5)
    certain = np.where(np.isnan(control), CONF_CARRY, CONF_CONTROL)
    q = np.where(carried == 0.5, 0.5, np.where(carried == 1.0, certain, 1.0 - certain))

    off = known & ((np.abs(ball[:, 0]) > length / 2 + 1) | (np.abs(ball[:, 1]) > width / 2 + 1))
    moved = np.hypot(*np.moveaxis(ball[step:] - ball[:-step], -1, 0))
    moved = np.r_[moved, np.full(step, np.nan)]
    still = (moved < 1.0) & (nearest > 2.0)
    gap = gap_seconds(seen, fps)
    x = np.stack([np.ones(len(seen)), mean_speed(g["pos"], fps), off.astype(float), still.astype(float),
                  np.log1p(gap), (~seen).astype(float)], axis=1)
    return x, q


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit(names: list[str], out: str) -> None:
    refuse(names)
    xs, ys = [], []
    for n in names:
        g = load(n)
        x, _ = cues(g)
        keep = np.flatnonzero(g["label"] >= 0)[::5]
        xs.append(x[keep])
        ys.append((g["label"][keep] == 2).astype(float))
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = np.zeros(x.shape[1])
    for _ in range(25):                               # Newton steps with a small ridge
        p = sigmoid(x @ w)
        h = (x * (p * (1 - p) + 1e-6)[:, None]).T @ x + 1e-3 * np.eye(len(w))
        w = w + np.linalg.solve(h, x.T @ (y - p) - 1e-3 * w)
    acc = float(((sigmoid(x @ w) > 0.5) == (y > 0.5)).mean())
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weights": dict(zip(CUES, [round(float(v), 4) for v in w])), "fitted_on": names,
                   "dead_accuracy_on_fit_games": round(acc, 4),
                   "note": "fitted on external games only (organisers' clarification, 2026-11-03)"},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    print("weights {}  dead accuracy {:.3f} -> {}".format(json.dumps(dict(zip(CUES, np.round(w, 3).tolist()))), acc,
                                                          os.path.relpath(out, REPO)))


def write_rule_probs(name: str, params_path: str = PARAMS_PATH) -> str:
    with open(params_path, encoding="utf-8") as f:
        weights = json.load(f)["weights"]
    w = np.array([weights[c] for c in CUES])
    g = load(name)
    x, q = cues(g)
    dead = sigmoid(x @ w)
    n = len(dead)
    df = {"frame": np.arange(n), "p_H": (1 - dead) * q, "p_A": (1 - dead) * (1 - q), "p_D": dead,
          "p_change": np.zeros(n), "ball_x": np.full(n, np.nan), "ball_y": np.full(n, np.nan),
          "ball_conf": np.zeros(n), "ball_src": np.zeros(n, np.int64)}
    import pandas as pd
    path = os.path.join(work_dir(), "probs_rules", name + "_probs.csv")
    write_probs(pd.DataFrame(df), path, name, n_frames=n)
    return path


def refuse(names: list[str]) -> None:
    for n in names:
        if is_test_game(n) or n.upper() in HELD_OUT | COMPARE_ONLY or n.upper().startswith(DROPPED):
            raise SystemExit("refused: never fit on {}".format(n))


def profile(names: list[str], out: str) -> None:
    """Ball-visibility profile from external games with a real tracker (tracking only, no labels)."""
    refuse(names)
    p = build_profile([n.upper() for n in names])
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(p, f)
        f.write("\n")
    print("visible share {}, {} visible and {} missing runs -> {}".format(
        p["visible_share"], len(p["visible_runs"]), len(p["missing_runs"]), os.path.relpath(out, REPO)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["profile", "fit", "probs"])
    ap.add_argument("games", nargs="+")
    ap.add_argument("--out", default=None, help="default baseline/params.json (fit) or "
                    "baseline/gap_profile.json (profile)")
    args = ap.parse_args(argv)
    if args.command == "profile":
        profile(args.games, args.out or PROFILE_PATH)
    elif args.command == "fit":
        fit(args.games, args.out or PARAMS_PATH)
    else:
        for n in args.games:
            path = write_rule_probs(n)
            print(quiet_report(path) if is_test_game(n) else path)


if __name__ == "__main__":
    main()
