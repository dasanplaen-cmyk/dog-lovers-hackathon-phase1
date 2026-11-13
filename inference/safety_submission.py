"""Interim safety-net submission from a probabilities file. Prints aggregates only.

    python -m inference.safety_submission                      # data/work/probs/BAR-ATH_probs.csv
    python evaluation/validate_submission.py data/work/submission/Dog_Lovers_safety.csv

Decoding: inside each in-play block of games.py (kick-off to final whistle of each half), the
3-state Viterbi of models/possession/train.py with its fixed switch penalty, the same yardstick
the checkpoint was chosen with on external games. Outside those blocks: X. Nothing here is tuned
on BAR-ATH. Ying's decoder (decode/) replaces this for the real submission.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import games  # noqa: E402
from inference.probs_format import read_probs  # noqa: E402
from training.runlog import work_dir  # noqa: E402

PENALTY = 8.0   # = viterbi_penalty in models/possession/configs/v1.yaml
LETTERS = np.array(["H", "A", "D"])


def main(argv=None):
    from models.possession.train import viterbi

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probs", default=os.path.join(str(work_dir()), "probs", "BAR-ATH_probs.csv"))
    ap.add_argument("--out", default=os.path.join(str(work_dir()), "submission", "Dog_Lovers_safety.csv"))
    args = ap.parse_args(argv)

    df = read_probs(args.probs, game="BAR-ATH")
    p = df[["p_H", "p_A", "p_D"]].to_numpy()
    labels = np.full(len(df), "X", dtype=object)
    for _, a, b in games.half_video_blocks("BAR-ATH"):
        block = p[a:b + 1]
        if np.isnan(block).any():
            raise SystemExit("probabilities missing inside an in-play block")
        labels[a:b + 1] = LETTERS[viterbi(block, PENALTY)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    from inference.quiet import quiet_report
    print(quiet_report(args.out))     # stop rule: path, hash and pass/fail only


if __name__ == "__main__":
    main()
