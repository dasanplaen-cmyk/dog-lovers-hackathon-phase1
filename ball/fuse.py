"""Fuse the observed ball, linear interpolation and the ball model into one ball estimate.

Rule: on a gap bounded by observed balls on both sides and at most `max_interp_gap` frames long,
use linear interpolation (ball_src 4, confidence from an empirical table by gap length); on
longer or open-ended gaps use the model (ball_src 2, confidence from its variance). Observed
frames keep the observation (ball_src 1).

The threshold and the table are fitted on the validation games named in the checkpoint's config
(external games only) and written to a small JSON that goes to git:

    python -m ball.fuse fit --weights data/work/weights/<run>/best.pt --out ball/configs/fusion_v1.json

ball/infer.py applies it with --fusion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CANDIDATE_GAPS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]
MIN_COUNT = 200   # frames per gap length before its own rate is trusted; otherwise pooled upwards


def gap_runs(vis: np.ndarray, valid: np.ndarray):
    """For each frame: length of the run of missing frames it is in (0 if visible) and whether
    that run has a visible frame immediately before and after it."""
    n = len(vis)
    miss = ~vis & valid
    length = np.zeros(n, np.int64)
    interior = np.zeros(n, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        length[a:b] = b - a
        interior[a:b] = a > 0 and b < n and vis[a - 1] and vis[b] and valid[a - 1] and valid[b]
    return length, interior


def interp_conf_lookup(fusion: dict, length: np.ndarray) -> np.ndarray:
    table = np.asarray(fusion["interp_conf"], float)   # index = gap length, 0 unused
    return table[np.clip(length, 0, len(table) - 1)]


def apply_fusion(game, model_xy: np.ndarray, model_conf: np.ndarray, fusion: dict):
    """Per-frame fused xy (metres), confidence and ball_src for a whole game."""
    obs = ~np.isnan(game.ball[:, 0])
    covered = ~np.isnan(model_xy[:, 0])
    xy = np.where(obs[:, None], game.ball[:, :2], model_xy)
    conf = np.where(covered | obs, model_conf, 0.0)
    src = np.where(obs, 1, np.where(covered, 2, 0))
    gmax = int(fusion["max_interp_gap"])
    for a, b in game.segments():
        vis = obs[a:b]
        length, interior = gap_runs(vis, np.ones(b - a, bool))
        use = interior & (length <= gmax) & (length > 0)
        if not use.any():
            continue
        t = np.arange(b - a)
        bx = game.ball[a:b, 0]
        by = game.ball[a:b, 1]
        ix = np.interp(t, t[vis], bx[vis])
        iy = np.interp(t, t[vis], by[vis])
        rows = a + np.flatnonzero(use)
        xy[rows, 0] = ix[use]
        xy[rows, 1] = iy[use]
        conf[rows] = interp_conf_lookup(fusion, length[use])
        src[rows] = 4
    return xy, conf, src


def fit(weights: str, out: str, device_name: str | None) -> dict:
    import torch
    from ball.batching import interpolate_baseline
    from ball.model import confidence
    from ball.train import fixed_eval_set, load_games, model_from_config, to_torch
    from inference.probs_format import sha256_file

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    radius = cfg["conf_radius_m"]

    def collect(names, seed):
        games = load_games(names, cfg["max_players"])
        em, ei, cm, ln, inter = [], [], [], [], []
        with torch.no_grad():
            for b in fixed_eval_set(games, cfg, seed, cfg.get("eval_max_windows")):
                players, pmask, ball_in, fmask, target, hidden = to_torch(b, device)
                mu, logvar = model(players, pmask, ball_in, fmask)
                e_m = np.linalg.norm(mu.float().cpu().numpy() - b.target, axis=-1)
                e_i = np.linalg.norm(interpolate_baseline(b) - b.target, axis=-1)
                c_m = confidence(logvar.float(), radius).cpu().numpy()
                for k in range(len(b.target)):
                    vis = b.ball_in[k, :, 2] > 0
                    L, I = gap_runs(vis, ~b.fmask[k])
                    sel = b.hidden[k] & ~np.isnan(b.target[k, :, 0])
                    em.append(e_m[k, sel]); ei.append(e_i[k, sel]); cm.append(c_m[k, sel])
                    ln.append(L[sel]); inter.append(I[sel])
        return [np.concatenate(x) for x in (em, ei, cm, ln, inter)]

    em, ei, cm, ln, inter = collect(cfg["val_games"], cfg["seed"] + 1)

    def fused_err(g, em, ei, ln, inter):
        use = inter & (ln <= g)
        return np.where(use, ei, em)

    scores = {g: float(fused_err(g, em, ei, ln, inter).mean()) for g in CANDIDATE_GAPS}
    best = min(CANDIDATE_GAPS, key=lambda g: (round(scores[g], 4), g))

    # P(interpolation error < radius) by gap length, pooled upwards until MIN_COUNT frames, then
    # made non-increasing in the gap length.
    table = [1.0]
    for L in range(1, best + 1):
        lo = L
        while True:
            k = inter & (ln >= L) & (ln <= lo)
            if k.sum() >= MIN_COUNT or lo >= best:
                break
            lo += 1
        rate = float((ei[k] < radius).mean()) if k.any() else table[-1]
        table.append(min(rate, table[-1]))

    fusion = {"max_interp_gap": int(best), "interp_conf": [round(v, 4) for v in table],
              "radius_m": radius, "weights_sha256": sha256_file(weights), "run": ckpt.get("run"),
              "fitted_on": cfg["val_games"], "val_mean_err_by_threshold_m":
                  {str(g): round(v, 4) for g, v in scores.items()}}

    def report(prefix, em, ei, cm, ln, inter):
        use = inter & (ln <= best)
        e = np.where(use, ei, em)
        conf = np.where(use, interp_conf_lookup(fusion, ln), cm)
        return {prefix + "err_mean_m_model": round(float(em.mean()), 4),
                prefix + "err_mean_m_fused": round(float(e.mean()), 4),
                prefix + "within_2m_fused": round(float((e < radius).mean()), 4),
                prefix + "calib_gap_fused": round(float(conf.mean() - (e < radius).mean()), 4),
                prefix + "share_interp": round(float(use.mean()), 4)}

    metrics = report("val_", em, ei, cm, ln, inter)
    if cfg.get("test_games"):   # reported once, never used to choose anything
        metrics.update(report("test_", *collect(cfg["test_games"], cfg["seed"] + 2)))
    fusion["metrics"] = metrics
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(fusion, f, indent=2, sort_keys=True)
        f.write("\n")
    return fusion


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="choose the threshold and confidence table on validation games")
    f.add_argument("--weights", required=True)
    f.add_argument("--out", default=os.path.join(REPO, "ball", "configs", "fusion_v1.json"))
    f.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    fusion = fit(args.weights, args.out, args.device)
    print("max_interp_gap {} frames -> {}".format(fusion["max_interp_gap"], args.out))
    print(json.dumps(fusion["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
