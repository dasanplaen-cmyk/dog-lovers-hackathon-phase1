"""Tune and score the decoder on external games only. Prints aggregates only.

    python -m eval.external tune DFL-J03WQQ DFL-J03WR9 SC-1874553 --model possession_v3 --probs-dir probs_v3  # decode/params.json
    python -m eval.external score DFL-J03WQQ DFL-J03WR9 SC-1874553 [--bootstrap 1000]
    python -m eval.external score SC-1886347               # model comparison, once per candidate
    python -m eval.external score SC-<the other 18> --final --bootstrap 1000   # once, at the end

Split agreed on the board (2026-11-09): DFL val games + SC-1874553 for tuning and model selection,
SC-1886347 for comparing candidate models (never tuned on), the other 18 SkillCorner games held out.
Tuning on held-out games is refused and scoring them needs --final. Metrica is not used (no
licence); its games are refused.

Labels are aligned with the challenge's written definition before scoring (align_control):
provider labels change team on any touch, the challenge only on controlled possession. The length
is fixed from the written proxy, never tuned, and never compared with the public ground truth.

Reads <work>/games/external/<game>.npz (labels, periods; see external_data/export.py) and
<work>/probs/<game>_probs.csv (inference/README.md). <work> is MTAH_WORK_DIR or data/work.
Scores with the organisers' own evaluation/possession_metrics.py.

The provided game is refused by name: it has no labels we may touch, and organisers' rules forbid
tuning on it (external data only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))

import possession_metrics as pm  # noqa: E402
from decode.decoder import DEFAULTS, LETTERS, PARAMS_PATH, decode, load_params  # noqa: E402
from external_data.download import SKILLCORNER_MATCHES  # noqa: E402
from inference.probs_format import read_probs  # noqa: E402

# SkillCorner 1874553 was scored once while checking the source and is used for tuning, so it
# cannot be a fair test. 1886347 compares candidate models: each is scored on it once and nothing is
# tuned on it. The other 18 are held out until the final score.
COMPARE_ONLY = {"SC-1886347"}
HELD_OUT = {"SC-" + m for m in SKILLCORNER_MATCHES if m not in ("1874553", "1886347")}
DROPPED = "METRICA"     # no licence: dropped by the team on 2026-11-09, refused everywhere
# CHALLENGE.md, "Controlled" possession: "a player keeps the ball for a few seconds".
CONTROL_MIN_S = 2.0

GRID = {
    "pen_team": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 28, 40, 64, 100, 160, 256, 400, 640],
    "pen_to_dead": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 28, 40, 64, 100, 160, 256, 400, 640],
    "pen_from_dead": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 28, 40, 64, 100, 160, 256, 400, 640],
    "change_weight": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    "min_spell_s": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    "min_dead_s": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
}


def work_dir() -> str:
    return os.environ.get("MTAH_WORK_DIR") or os.path.join(REPO, "data", "work")


def align_control(gt: np.ndarray, fps: float, min_s: float = CONTROL_MIN_S) -> np.ndarray:
    """Labels with uncontrolled touches removed: a team that takes over from the other team in
    live play but keeps the ball under min_s never had possession, so it stays with the team that
    had it. Possessions that start after a stoppage are kept: a restart has a clear possessor."""
    out = gt.copy()
    edges = np.flatnonzero(np.diff(gt)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(gt)]
    holder = pm.X
    for a, b in zip(starts.tolist(), stops.tolist()):
        state = int(gt[a])
        if state not in (pm.H, pm.A):
            holder = pm.X
        elif holder != pm.X and state != holder and b - a < min_s * fps:
            out[a:b] = holder
        else:
            holder = state
    return out


def load_game(name: str, final: bool = False, probs_dir: str = "probs") -> dict:
    name = name.upper()
    if "BAR-ATH" in name:
        raise SystemExit("refused: {} is the provided game; tuning and scoring here are for "
                         "external games only".format(name))
    if name.startswith(DROPPED):
        raise SystemExit("refused: {} is Metrica data, dropped for lack of a licence".format(name))
    if name in HELD_OUT and not final:
        raise SystemExit("refused: {} is held out until the end (score --final)".format(name))
    z = np.load(os.path.join(work_dir(), "games", "external", name + ".npz"), allow_pickle=False)
    label, period = z["label"], z["period"]
    gt = np.where(label < 0, pm.X, label).astype(np.int64)
    # One graded span per period, so no possession is counted across half-time (as in BAR-ATH).
    starts = np.flatnonzero(np.diff(period)) + 1
    gt[starts] = pm.X
    if not (gt != pm.X).any():
        raise SystemExit("{} has no labels: export it with external_data.export".format(name))
    edges = np.r_[0, starts, len(period)]
    blocks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    df = read_probs(os.path.join(work_dir(), probs_dir, name + "_probs.csv"), n_frames=len(label))
    fps = float(z["fps"])
    return {"name": name, "gt": align_control(gt, fps), "gt_raw": gt, "blocks": blocks, "fps": fps,
            "p": df[["p_H", "p_A", "p_D"]].to_numpy(dtype=np.float64),
            "pc": df["p_change"].to_numpy(dtype=np.float64)}


def official_spell_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    """spellF1 as pm.evaluate computes it: abstentions filled inside the graded runs first."""
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"]


def spell_f1(g: dict, params: dict) -> float:
    return official_spell_f1(g["gt"], decode(g["p"], g["pc"], params, g["fps"], g["blocks"]).astype(np.int64))


def tune(names: list[str], out: str, sweeps: int = 6, probs_dir: str = "probs", model: str | None = None) -> None:
    for n in names:
        if n.upper() in COMPARE_ONLY:
            raise SystemExit("refused: {} is for comparing models only, never for tuning".format(n.upper()))
    games = [load_game(n, probs_dir=probs_dir) for n in names]
    params = dict(DEFAULTS)
    best = float(np.mean([spell_f1(g, params) for g in games]))
    print("start {} mean spellF1 {:.4f}".format(json.dumps(params), best), flush=True)
    for sweep in range(sweeps):
        t0, changed = time.time(), False
        for key, values in GRID.items():
            for v in values:
                if v == params[key]:
                    continue
                trial = {**params, key: float(v)}
                score = float(np.mean([spell_f1(g, trial) for g in games]))
                if score > best + 1e-9:
                    best, params, changed = score, trial, True
            print("sweep {} {} -> {} (spellF1 {:.4f})".format(sweep, key, params[key], best), flush=True)
        print("sweep {} done in {:.0f} s".format(sweep, time.time() - t0), flush=True)
        if not changed:
            break
    else:
        print("warning: still improving after {} sweeps".format(sweeps))
    edge = [k for k, v in GRID.items() if params[k] == max(v) or (params[k] == min(v) and min(v) != 0)]
    if edge:
        print("warning: on the edge of the grid: {}".format(", ".join(edge)))
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"params": params, "tuned_on": [g["name"] for g in games], "spell_f1_mean": round(best, 4),
                   "control_min_s": CONTROL_MIN_S, "model": model,
                   "note": "tuned on external games only (organisers' clarification, 2026-11-03)"},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {}".format(os.path.relpath(out, REPO)))


def score(names: list[str], params: dict, bootstrap: int, final: bool = False, probs_dir: str = "probs") -> None:
    print("params {}".format(json.dumps(params, sort_keys=True)))
    rows = []
    for n in names:
        g = load_game(n, final=final, probs_dir=probs_dir)
        pred = decode(g["p"], g["pc"], params, g["fps"], g["blocks"]).astype(np.int64)
        argmax = np.where(np.isnan(g["p"]).any(1), pm.X, np.nan_to_num(g["p"]).argmax(1)).astype(np.int64)
        r = pm.evaluate(g["gt"], pred, bootstrap_n=bootstrap)
        row = {"game": g["name"], "spellF1": round(r["spell_f1_mean"], 4), "macroF1": round(r["macro_f1"], 4),
               "spellF1_argmax": round(official_spell_f1(g["gt"], argmax), 4),
               "spellF1_raw_labels": round(official_spell_f1(g["gt_raw"], pred), 4),
               "spell_count_error": int(r["spell_count_error"]),
               "labels": {str(LETTERS[c]): int((pred == c).sum()) for c in range(4)}}
        if bootstrap:
            row["spellF1_5_95"] = [round(r["spell_bootstrap_low"], 4), round(r["spell_bootstrap_high"], 4)]
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print("mean spellF1 {:.4f}  macroF1 {:.4f}".format(np.mean([r["spellF1"] for r in rows]),
                                                       np.mean([r["macroF1"] for r in rows])))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["tune", "score"])
    ap.add_argument("games", nargs="+")
    ap.add_argument("--params", default=None, help="params JSON (default decode/params.json, or defaults)")
    ap.add_argument("--out", default=PARAMS_PATH)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--final", action="store_true", help="allow scoring the held-out games")
    ap.add_argument("--probs-dir", default="probs", help="folder under <work>, e.g. probs_rules")
    ap.add_argument("--model", default=None, help="checkpoint behind the probabilities, recorded by tune")
    args = ap.parse_args(argv)
    if args.probs_dir != "probs" and ((args.command == "tune" and args.out == PARAMS_PATH) or
                                      (args.command == "score" and not args.params)):
        ap.error("with --probs-dir, give --out or --params: decode/params.json belongs to the model")
    if args.command == "tune":
        if args.out == PARAMS_PATH and not args.model:
            ap.error("tuning decode/params.json needs --model, the name of the checkpoint")
        tune(args.games, args.out, probs_dir=args.probs_dir, model=args.model)
    else:
        if args.params and not os.path.exists(args.params):
            raise SystemExit("no such params file: {}".format(args.params))
        path = args.params or PARAMS_PATH
        score(args.games, load_params(path) if os.path.exists(path) else dict(DEFAULTS), args.bootstrap,
              final=args.final, probs_dir=args.probs_dir)


if __name__ == "__main__":
    main()
