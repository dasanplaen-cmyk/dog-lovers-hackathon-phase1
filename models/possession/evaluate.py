"""Score a possession checkpoint on external games with labels. Prints aggregates only.

    python -m models.possession.evaluate --weights data/work/weights/release/possession_v1.pt --games SC-1886347
    python -m models.possession.evaluate --weights ... --games DFL-J03WQQ DFL-J03WR9 --mask-seed 1

Real-gap games (SkillCorner) are scored as they are; --mask-seed hides the ball of games that never
lose it (DFL) with the checkpoint's own gap simulation. Same fixed Viterbi yardstick as training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from models.possession.data import PossessionGame, masked_copy  # noqa: E402
from models.possession.train import (ball_model_for, load_games, model_from_config,  # noqa: E402
                                     predict_game, score)
import eval.external as split  # noqa: E402  (one list of held-out, compare-only and dropped games)
from training.runlog import set_seed  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--mask-seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--final", action="store_true", help="allow scoring the held-out games")
    args = ap.parse_args(argv)
    games = [g.upper() for g in args.games]
    dropped = getattr(split, "DROPPED", "METRICA")
    if any(dropped in g for g in games):
        raise SystemExit("refused: {} games are dropped from the project".format(dropped))
    held = sorted((split.HELD_OUT - getattr(split, "COMPARE_ONLY", set())) & set(games))
    if held and not args.final:
        raise SystemExit("refused: {} are held out until the end (--final)".format(held))
    set_seed(0)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ball = ball_model_for(cfg, device)
    out = {}
    for i, g in enumerate(load_games(args.games, cfg["max_players"], scoring=True)):
        if args.mask_seed is not None:
            g = masked_copy(g, cfg, args.mask_seed + i)
        p, _ = predict_game(model, PossessionGame(g, cfg["max_players"]), cfg, device, ball=ball)
        out[g.name] = {k: round(v, 4) for k, v in score(g, p, cfg["viterbi_penalty"]).items()}
    print(json.dumps({"model": cfg["name"], "run": ckpt.get("run"), "scores": out}, indent=2))


if __name__ == "__main__":
    main()
