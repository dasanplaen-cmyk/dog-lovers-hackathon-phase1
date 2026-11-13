"""Compare a game's ball-gap profile and ball-file confidence with another game. Aggregates only.

    python -m ball.diagnose SC-1874553

Refused for BAR-ATH while the team's stop rule is on (see inference/quiet.py).

An automatic check on the inputs and our own outputs: nothing is compared with the video or with
any ground truth, and nothing here is used to choose or tune anything. For BAR-ATH only the
in-play blocks of games.py (kickoff to final whistle of each half) are counted; external games
use every frame.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ball.fuse import gap_runs  # noqa: E402
from ball.game import Game  # noqa: E402
from ball.infer import game_path  # noqa: E402
from training.runlog import work_dir  # noqa: E402

BUCKETS = [(1, 10), (11, 30), (31, 75), (76, 250), (251, 10**9)]


def in_play(game: Game) -> np.ndarray:
    keep = np.zeros(game.n_frames, bool)
    if game.name == "BAR-ATH":
        import games
        for _, a, b in games.half_video_blocks("BAR-ATH"):
            keep[a:b + 1] = True
    else:
        keep[:] = True
    return keep


def profile(name: str) -> dict:
    game = Game.load(game_path(name))
    keep = in_play(game)
    ball_csv = os.path.join(str(work_dir()), "ball", "{}_ball.csv".format(name))
    conf = src = None
    if os.path.isfile(ball_csv):
        df = pd.read_csv(ball_csv, usecols=["ball_conf", "ball_src"])
        conf, src = df["ball_conf"].to_numpy(), df["ball_src"].to_numpy()
    obs = ~np.isnan(game.ball[:, 0])
    length, interior = gap_runs(obs, keep)
    miss = keep & ~obs
    tracked = (~np.isnan(game.pos[..., 0])).sum(axis=1)
    out = {"game": name, "frames_in_play": int(keep.sum()),
           "ball_observed_in_play": round(float(obs[keep].mean()), 4),
           "mean_tracked_players_in_play": round(float(tracked[keep].mean()), 2),
           "missing_frames_bounded_share": round(float(interior[miss].mean()), 4) if miss.any() else None,
           "missing_frames_by_gap_length": {}}
    for lo, hi in BUCKETS:
        k = miss & (length >= lo) & (length <= hi)
        row = {"share": round(float(k.sum() / max(miss.sum(), 1)), 4)}
        if conf is not None and k.any():
            row["conf_mean"] = round(float(conf[k].mean()), 4)
        out["missing_frames_by_gap_length"]["{}-{}".format(lo, hi if hi < 10**9 else "inf")] = row
    if conf is not None:
        m = miss & (src == 2)
        out["conf_mean_model_frames_in_play"] = round(float(conf[m].mean()), 4) if m.any() else None
    return out


def main(argv=None):
    from inference.quiet import is_test_game
    names = (argv or sys.argv[1:]) or ["SC-1874553"]
    if any(is_test_game(n) for n in names):
        raise SystemExit("refused: no aggregates of the provided game while the team's stop rule is on")
    for name in names:
        print(json.dumps(profile(name), indent=2))


if __name__ == "__main__":
    main()
