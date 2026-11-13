"""Write a game's probabilities file (inference/README.md) from the possession model and the ball
file. Prints aggregates only.

    python -m models.possession.infer --weights data/work/weights/<run>/best.pt --game BAR-ATH --device cpu
    python -m models.possession.infer --weights ... --game DFL-J03WQQ --mask-seed 1   # external,
                                                       # ball hidden on simulated gaps like the ATD

The ball columns come from <work>/ball/<game>_ball.csv when it exists (ball/infer.py); otherwise
from the observed ball only. Output: <work>/probs/<game>_probs.csv plus its .meta.json sidecar.
The attack direction found for each half is printed with the goalkeepers' mean x it comes from:
a check on the input alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from ball.game import Game  # noqa: E402
from ball.infer import game_path  # noqa: E402
from inference.probs_format import BALL_RANGE, sha256_file, validate_probs, write_probs  # noqa: E402
from models.possession.data import PossessionGame, direction_report, masked_copy  # noqa: E402
from models.possession.train import ball_model_for, model_from_config, predict_game  # noqa: E402
from inference.quiet import is_test_game, quiet_report  # noqa: E402
from training.runlog import set_seed, work_dir  # noqa: E402


def ball_columns(game: Game, use_file: bool = True) -> tuple[pd.DataFrame, str | None]:
    path = os.path.join(str(work_dir()), "ball", "{}_ball.csv".format(game.name))
    if use_file and os.path.isfile(path):
        df = pd.read_csv(path)
        if len(df) != game.n_frames:
            raise SystemExit("{} has {} rows, game has {}".format(path, len(df), game.n_frames))
        return df[["ball_x", "ball_y", "ball_conf", "ball_src"]], sha256_file(path)
    obs = ~np.isnan(game.ball[:, 0])
    xy = np.clip(game.to_normalised(game.ball[:, :2]), *BALL_RANGE)
    return pd.DataFrame({"ball_x": np.where(obs, xy[:, 0], np.nan),
                         "ball_y": np.where(obs, xy[:, 1], np.nan),
                         "ball_conf": obs.astype(float), "ball_src": obs.astype(int)}), None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--device", default=None, help="cpu for the byte-for-byte final run")
    ap.add_argument("--mask-seed", type=int, default=None,
                    help="external games only: hide the ball on simulated gaps first")
    ap.add_argument("--out", default=None)
    ap.add_argument("--precision", choices=["float64", "float32"], default=None,
                    help="default float64 on CPU: results then agree across machines")
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
    if args.mask_seed is not None:
        if game.name == "BAR-ATH":
            raise SystemExit("--mask-seed is for external games; BAR-ATH's gaps are real")
        game = masked_copy(game, cfg, args.mask_seed)
    pg = PossessionGame(game, cfg["max_players"])
    if is_test_game(game.name) and not all(h["opposite"] for h in direction_report(game)):
        raise SystemExit("FAILED: goalkeeper direction check is ambiguous in some half; not writing "
                         "a file whose H/A could be swapped")
    ball = ball_model_for(cfg, device)
    if ball is not None and args.precision == "float64":
        ball[0].double()
    p, c = predict_game(model, pg, cfg, device, ball=ball)

    ball, ball_sha = ball_columns(game, use_file=args.mask_seed is None)
    df = pd.DataFrame({"frame": np.arange(game.n_frames), "p_H": p[:, 0], "p_A": p[:, 1],
                       "p_D": p[:, 2], "p_change": c})
    df = pd.concat([df, ball.reset_index(drop=True)], axis=1)
    out = args.out or os.path.join(str(work_dir()), "probs", "{}_probs.csv".format(game.name))
    meta = write_probs(df, out, game=game.name, n_frames=game.n_frames, meta={
        "model": cfg["name"], "run": ckpt.get("run"), "step": ckpt.get("step"),
        "weights_sha256": sha256_file(args.weights), "ball_csv_sha256": ball_sha,
        "ball_weights": cfg.get("ball_weights"),
        "mask_seed": args.mask_seed, "device": device.type, "precision": args.precision, "torch": torch.__version__})
    stats = validate_probs(pd.read_csv(out, keep_default_na=False, na_values=[""]),
                           n_frames=game.n_frames)
    if is_test_game(game.name):     # stop rule: path, hash and pass/fail only
        print(quiet_report(out))
        return
    print(json.dumps({"game": game.name, "out": out, "csv_sha256": meta["csv_sha256"][:16],
                      "home_attack_sign_per_half": pg.half_signs,
                      "goalkeepers_per_half": direction_report(game), **stats}, indent=2))


if __name__ == "__main__":
    main()
