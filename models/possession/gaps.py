"""Ball-visibility profile of real tracking, and masks drawn from it.

The profile is the list of lengths of every visible run and every missing run of the ball in some
external games with real tracker gaps (SkillCorner's detected ball; MIT licence). It is
read from the tracking alone, no labels. Training then hides the ball of games that never lose it
(DFL) with an alternating renewal process over those empirical lengths, so the model sees gaps
like a real tracker's rather than synthetic ones.

    python -m models.possession.gaps SC-1874553 --out <file.json>

The profile used by v2_sc and v3 is baseline/gap_profile.json, written by `baseline.rules profile`
with the same build_profile.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

PROFILE = os.path.join(REPO, "baseline", "gap_profile.json")   # the profile v2_sc and v3 use


def runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of the maximal runs of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return edges[1::2] - edges[::2]


def build_profile(names: list[str]) -> dict:
    from ball.game import Game
    from ball.sources import external_dir
    vis, miss, seen, total = [], [], 0, 0
    for name in names:
        if "BAR-ATH" in name:
            raise SystemExit("profile from external games only")
        g = Game.load(os.path.join(external_dir(), name + ".npz"))
        for a, b in g.segments():
            obs = ~np.isnan(g.ball[a:b, 0])
            vis += runs(obs).tolist()
            miss += runs(~obs).tolist()
            seen += int(obs.sum())
            total += b - a
    return {"games": names, "visible_share": round(seen / total, 4),
            "visible_runs": sorted(int(v) for v in vis), "missing_runs": sorted(int(v) for v in miss)}


def renewal_mask(n: int, rng: np.random.Generator, profile: dict) -> np.ndarray:
    """Boolean [n], True = hidden: alternating visible / missing runs drawn from the profile,
    entered at a random point of a random run (length-biased, as a window lands in a real game)."""
    vis = np.asarray(profile["visible_runs"])
    miss = np.asarray(profile["missing_runs"])
    # Probability that a random frame is missing, and length-biased choice of the run it is in.
    hidden = rng.random() > profile["visible_share"]
    pool = miss if hidden else vis
    first = int(rng.choice(pool, p=pool / pool.sum()))
    t = -int(rng.integers(0, first))
    out = np.zeros(n, bool)
    length = first
    while t < n:
        if hidden:
            out[max(t, 0):max(0, min(n, t + length))] = True
        t += length
        hidden = not hidden
        length = int(rng.choice(miss if hidden else vis))
    return out


def load_profile(path: str = PROFILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games", nargs="+")
    ap.add_argument("--out", required=True, help="where to write the profile JSON")
    args = ap.parse_args(argv)
    p = build_profile(args.games)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(p, f)
        f.write("\n")
    for k in ("visible_runs", "missing_runs"):
        a = np.asarray(p[k])
        print("{}: {} runs, median {} frames, p90 {}, frames in runs > 250: {:.1%}".format(
            k, len(a), int(np.median(a)), int(np.percentile(a, 90)), a[a > 250].sum() / a.sum()))
    print("visible share {} -> {}".format(p["visible_share"], args.out))


if __name__ == "__main__":
    main()
