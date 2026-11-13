"""Decoder: per-frame probabilities (inference/README.md format) -> one label per frame.

Most likely H/A/D path over a block of play (Viterbi), with a cost for every change of state, then
removal of runs too short to be real. spellF1, the ranking metric, punishes spurious possessions,
which is why the raw argmax of the model scores far below a smoothed path.

Parameters (decode/params.json, tuned on external games only, see eval/external.py):

    pen_team        cost of a change H <-> A
    pen_to_dead     cost of a change H/A -> D
    pen_from_dead   cost of a change D -> H/A
    change_weight   0..1, how much the model's p_change lowers the cost at that frame:
                    cost(t) = pen * (1 - change_weight * p_change(t))
    min_spell_s     possessions shorter than this are absorbed by a neighbour
    min_dead_s      dead-ball runs shorter than this are absorbed by a neighbour

Codes match evaluation/possession_metrics.py: H=0, A=1, D=2, X=3. Rows with no prediction, and rows
outside `blocks`, are X. Deterministic across machines: no randomness, and log-probabilities come
from a table rounded to 6 decimals, so a last-bit difference in a maths library cannot flip a choice.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
DEFAULTS = {"pen_team": 8.0, "pen_to_dead": 8.0, "pen_from_dead": 8.0, "change_weight": 0.0,
            "min_spell_s": 0.0, "min_dead_s": 0.0}
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
# log(p) for p = i / 10000, the precision of the probabilities format. p = 0 costs as much as 1e-6.
LOG_TABLE = np.array([round(math.log(max(i / 10000, 1e-6)), 6) for i in range(10001)])


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return {**DEFAULTS, **saved["params"]}


def viterbi(p: np.ndarray, p_change: np.ndarray, params: dict) -> np.ndarray:
    """Best state path for one block. p [T,3], p_change [T]. Plain Python floats: fast enough
    (under a second for a half) and free of platform-dependent vectorised rounding."""
    logp = LOG_TABLE[np.rint(np.clip(p, 0.0, 1.0) * 10000).astype(np.int64)].tolist()
    w = float(params["change_weight"])
    scale = (1.0 - w * np.nan_to_num(p_change, nan=0.0)).tolist()
    team, to_dead, from_dead = (float(params[k]) for k in ("pen_team", "pen_to_dead", "pen_from_dead"))
    T = len(logp)
    back = np.zeros((T, 3), np.int8)
    h, a, d = logp[0]
    for t in range(1, T):
        s = scale[t]
        c_team, c_to, c_from = team * s, to_dead * s, from_dead * s
        lh, la, ld = logp[t]
        # H
        best, arg = h, 0
        v = a - c_team
        if v > best:
            best, arg = v, 1
        v = d - c_from
        if v > best:
            best, arg = v, 2
        nh, bh = best + lh, arg
        # A
        best, arg = a, 1
        v = h - c_team
        if v > best:
            best, arg = v, 0
        v = d - c_from
        if v > best:
            best, arg = v, 2
        na, ba = best + la, arg
        # D
        best, arg = d, 2
        v = h - c_to
        if v > best:
            best, arg = v, 0
        v = a - c_to
        if v > best:
            best, arg = v, 1
        nd, bd = best + ld, arg
        h, a, d = nh, na, nd
        back[t, 0], back[t, 1], back[t, 2] = bh, ba, bd
    path = np.zeros(T, np.int8)
    last = [h, a, d]
    path[-1] = last.index(max(last))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def absorb_short_runs(path: np.ndarray, min_spell: int, min_dead: int) -> np.ndarray:
    """Repeatedly give the shortest too-short run to a neighbour: the one matching both sides if
    they agree, otherwise the longer one (the earlier on a tie)."""
    path = path.copy()
    limit = np.array([min_spell, min_spell, min_dead])
    while True:
        edges = np.flatnonzero(np.diff(path)) + 1
        starts = np.r_[0, edges]
        stops = np.r_[edges, len(path)]
        if len(starts) == 1:
            return path
        lengths = stops - starts
        states = path[starts]
        short = np.flatnonzero(lengths < limit[states])
        if len(short) == 0:
            return path
        i = short[np.argmin(lengths[short])]
        if i == 0:
            fill = states[1]
        elif i == len(starts) - 1:
            fill = states[i - 1]
        elif states[i - 1] == states[i + 1] or lengths[i - 1] >= lengths[i + 1]:
            fill = states[i - 1]
        else:
            fill = states[i + 1]
        path[starts[i]:stops[i]] = fill


def _prediction_blocks(has_pred: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, has_pred.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def decode(p: np.ndarray, p_change: np.ndarray, params: dict, fps: float = 25.0,
           blocks: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Codes [T]. `blocks`: half-open [start, stop) spans decoded independently (e.g. the halves);
    default, every contiguous span with a prediction. Everything else is X."""
    params = {**DEFAULTS, **params}
    has_pred = ~np.isnan(p).any(axis=1)
    out = np.full(len(p), X, dtype=np.int8)
    min_spell = int(round(params["min_spell_s"] * fps))
    min_dead = int(round(params["min_dead_s"] * fps))
    spans = []
    for a, b in (blocks if blocks is not None else [(0, len(p))]):
        spans += [(a + s, a + e) for s, e in _prediction_blocks(has_pred[a:b])]
    for a, b in spans:
        path = viterbi(p[a:b], p_change[a:b], params)
        out[a:b] = absorb_short_runs(path, min_spell, min_dead)
    return out


def selftest() -> None:
    rng = np.random.default_rng(0)
    truth = np.repeat(rng.integers(0, 3, 40), rng.integers(100, 400, 40))
    p = np.full((len(truth), 3), 0.2)
    p[np.arange(len(truth)), truth] = 0.6
    p = p + rng.normal(0, 0.25, p.shape)                # noisy frame-level scores
    p = np.clip(p, 0.01, None)
    p = p / p.sum(axis=1, keepdims=True)
    pc = np.zeros(len(truth))
    argmax = p.argmax(1)
    out = decode(p, pc, {"min_spell_s": 1.0, "min_dead_s": 1.0})
    assert (out == truth).mean() > (argmax == truth).mean(), "decoder should beat argmax"
    assert (np.diff(out) != 0).sum() < (np.diff(argmax) != 0).sum() / 10, "decoder should smooth"
    assert (out == decode(p, pc, {"min_spell_s": 1.0, "min_dead_s": 1.0})).all(), "deterministic"
    q = p.copy()
    q[100:200] = np.nan
    out2 = decode(q, pc, {})
    assert (out2[100:200] == X).all() and (out2[:100] != X).all(), "no prediction -> X"
    assert (absorb_short_runs(np.array([0, 0, 0, 1, 0, 0, 0]), 2, 2) == 0).all(), "absorb"
    print("decoder selftest OK: frame accuracy {:.3f} vs argmax {:.3f}, {} vs {} changes".format(
        (out == truth).mean(), (argmax == truth).mean(), int((np.diff(out) != 0).sum()),
        int((np.diff(argmax) != 0).sum())))


if __name__ == "__main__":
    selftest()
