"""Run a trained ball model over a whole game and write its ball file. Prints aggregates only.

    python -m ball.infer --weights data/work/weights/<run>/best.pt --game DFL-J03WQQ
    python -m ball.infer --weights data/work/weights/<run>/best.pt --game BAR-ATH --device cpu

Output: <work>/ball/<game>_ball.csv with frame, ball_x, ball_y, ball_conf, ball_src (the ball
columns of inference/README.md: normalised 0-1 like the ATD, 4 decimals), and a .meta.json
sidecar with the weights' sha256. Where the source observed the ball, its position is kept
(ball_src 1); elsewhere the model's estimate is used (ball_src 2), or with --fusion linear
interpolation on short bounded gaps (ball_src 4, see ball/fuse.py). ball_conf is
P(error < conf_radius_m).

`frame` is the row of the game's records; for BAR-ATH that is the video frame (checked when the
ATD was converted by ball/sources.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from ball.batching import GameArrays, make_batch  # noqa: E402
from ball.fuse import apply_fusion  # noqa: E402
from ball.game import Game  # noqa: E402
from ball.model import confidence  # noqa: E402
from ball.sources import external_dir, test_dir  # noqa: E402
from ball.train import model_from_config, to_torch  # noqa: E402
from inference.probs_format import BALL_RANGE, DECIMALS, sha256_file  # noqa: E402
from inference.quiet import is_test_game, quiet_report  # noqa: E402
from training.runlog import set_seed, work_dir  # noqa: E402


def game_path(name: str) -> str:
    folder = test_dir() if name == "BAR-ATH" else external_dir()
    return os.path.join(folder, name + ".npz")


@torch.no_grad()
def predict(model, ga: GameArrays, cfg: dict, device, stride: int | None = None):
    """Per-frame mean [T,2] (m) and log-variance [T] from overlapping windows, weighted toward
    each window's centre."""
    W = cfg["window"]
    stride = stride or W // 2
    T = ga.game.n_frames
    acc_w = np.zeros(T)
    acc_mu = np.zeros((T, 2))
    acc_var = np.zeros(T)
    tri = 1.0 - np.abs(np.arange(W) - (W - 1) / 2) / ((W + 1) / 2)
    starts = [(0, s) for s in ga.window_starts(W, stride)]
    rng = np.random.default_rng(0)  # unused when nothing is hidden; kept for the signature
    for i in range(0, len(starts), cfg["batch_size"]):
        b = make_batch([ga], rng, cfg, train=False, starts=starts[i:i + cfg["batch_size"]],
                       simulate_hidden=False)
        players, pmask, ball_in, fmask, _, _ = to_torch(b, device)
        dt = next(model.parameters()).dtype           # float64 for cross-machine reproducibility
        players, ball_in = players.to(dt), ball_in.to(dt)
        mu, logvar = model(players, pmask, ball_in, fmask)
        mu, var = mu.double().cpu().numpy(), torch.exp(logvar.double()).cpu().numpy()
        for k, s in enumerate(b.start):
            n = int((~b.fmask[k]).sum())
            w = tri[:n]
            acc_w[s:s + n] += w
            acc_mu[s:s + n] += w[:, None] * mu[k, :n]
            acc_var[s:s + n] += w * var[k, :n]
    covered = acc_w > 0
    mu = np.full((T, 2), np.nan)
    logvar = np.full(T, np.nan)
    mu[covered] = acc_mu[covered] / acc_w[covered, None]
    logvar[covered] = np.log(acc_var[covered] / acc_w[covered])
    return mu, logvar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--game", required=True, help="BAR-ATH or an external game name")
    ap.add_argument("--device", default=None, help="cpu for the byte-for-byte final run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fusion", default=None, help="fusion JSON from ball.fuse fit")
    ap.add_argument("--precision", choices=["float64", "float32"], default=None,
                    help="default float64 on CPU: results then agree across machines to far below the 4 written decimals")
    args = ap.parse_args(argv)

    set_seed(0)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.precision = args.precision or ("float64" if device.type == "cpu" else "float32")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if args.precision == "float64":
        model.double()

    game = Game.load(game_path(args.game))
    ga = GameArrays(game, cfg["max_players"])
    mu, logvar = predict(model, ga, cfg, device)

    obs = ~np.isnan(game.ball[:, 0])
    covered = ~np.isnan(mu[:, 0])
    model_conf = np.where(covered, confidence(torch.as_tensor(np.nan_to_num(logvar)),
                                              cfg["conf_radius_m"]).numpy(), 0.0)
    fusion = None
    if args.fusion:
        with open(args.fusion, encoding="utf-8") as f:
            fusion = json.load(f)
        if fusion["weights_sha256"] != sha256_file(args.weights):
            raise SystemExit("{} was fitted for other weights".format(args.fusion))
        xy, conf, src = apply_fusion(game, mu, model_conf, fusion)
    else:
        xy = np.where(obs[:, None], game.ball[:, :2], mu)
        conf = np.where(covered | obs, model_conf, 0.0)
        src = np.where(obs, 1, np.where(covered, 2, 0))
    have = src > 0
    norm = np.clip(game.to_normalised(xy), *BALL_RANGE)
    df = pd.DataFrame({"frame": np.arange(game.n_frames),
                       "ball_x": np.where(have, norm[:, 0], np.nan),
                       "ball_y": np.where(have, norm[:, 1], np.nan),
                       "ball_conf": np.round(conf, DECIMALS) + 0.0, "ball_src": src})
    df[["ball_x", "ball_y"]] = df[["ball_x", "ball_y"]].round(DECIMALS) + 0.0

    out = args.out or os.path.join(str(work_dir()), "ball", "{}_ball.csv".format(game.name))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False, float_format="%.{}f".format(DECIMALS), lineterminator="\n",
              na_rep="")
    meta = {"game": game.name, "rows": int(len(df)), "csv_sha256": sha256_file(out),
            "weights": os.path.basename(args.weights), "weights_sha256": sha256_file(args.weights),
            "run": ckpt.get("run"), "step": ckpt.get("step"), "device": device.type,
            "precision": args.precision,
            "fusion": os.path.basename(args.fusion) if args.fusion else None,
            "fusion_sha256": sha256_file(args.fusion) if args.fusion else None,
            "torch": torch.__version__}
    with open(out + ".meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    model_frames = src == 2
    summary = {
        "game": game.name, "rows": int(len(df)), "out": out, "csv_sha256": meta["csv_sha256"][:16],
        "src_counts": {"none": int((src == 0).sum()), "observed": int((src == 1).sum()),
                       "model": int(model_frames.sum()), "interpolated": int((src == 4).sum())},
        "conf_mean_model_frames": round(float(conf[model_frames].mean()), 4) if model_frames.any() else None,
        "conf_deciles_model_frames": [round(float(q), 3) for q in np.quantile(
            conf[model_frames], np.linspace(0, 1, 11))] if model_frames.any() else None,
    }
    if is_test_game(game.name):     # stop rule: path, hash and pass/fail only
        print(quiet_report(out))
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
