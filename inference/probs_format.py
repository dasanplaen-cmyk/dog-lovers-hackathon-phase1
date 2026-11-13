"""Per-frame probabilities file: the contract between the models (Chanti) and the decoder (Ying).

One CSV per game, one row per frame of that game's reference video (for BAR-ATH: the tactical
video, frames 0..199318). The models write it with `write_probs`; the decoder reads it with
`read_probs`. Both go through `validate_probs`, so a file that one side accepts the other side
accepts too. The full description is in inference/README.md.

    frame      int    0..n-1, contiguous, ascending
    p_H        float  P(home in possession, ball alive)       } sum to 1 (4-decimal rounding
    p_A        float  P(away in possession, ball alive)       }  tolerated); all three empty
    p_D        float  P(dead ball)                            }  together = no prediction
    p_change   float  P(state at frame t != state at t-1), states H/A/D; 0 on frame 0
    ball_x     float  ball position, normalised 0-1 like the ATD (x along the pitch length); empty if unknown
    ball_y     float  same, across the pitch
    ball_conf  float  0-1 confidence of (ball_x, ball_y); 0 when unknown
    ball_src   int    0 none, 1 ATD observed, 2 ball-from-players model, 3 video detector, 4 fused

Determinism: floats are clipped and written with exactly DECIMALS decimals, "\\n" line endings and
no index, so the same values give the same bytes on any machine. A JSON sidecar
(<file>.meta.json) records the schema version, game, model, commit, weights hash, seed and the
CSV's own sha256. Neither file may be committed or posted for BAR-ATH: it is frame-level match
data. Write it under data/ (gitignored, and denied to the coding agents).

CLI (prints aggregates only, never rows):

    python -m inference.probs_format validate data/work/BAR-ATH_probs.csv --game BAR-ATH
    python -m inference.probs_format selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
DECIMALS = 4
STATE_COLS = ["p_H", "p_A", "p_D"]
FLOAT_COLS = STATE_COLS + ["p_change", "ball_x", "ball_y", "ball_conf"]
COLUMNS = ["frame"] + FLOAT_COLS + ["ball_src"]
BALL_SOURCES = {0: "none", 1: "atd", 2: "players_model", 3: "video_detector", 4: "fused"}

SUM_TOL = 1e-3          # |p_H + p_A + p_D - 1| after rounding to DECIMALS
BALL_RANGE = (-0.1, 1.1)  # normalised; a ball just off the pitch is legitimate

# Reference-video length per game. Only BAR-ATH is fixed by the challenge; external games are
# checked for contiguity alone unless their length is passed explicitly.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from games import GAMES as _GAMES
    KNOWN_FRAMES = {name: g["n_video_frames"] for name, g in _GAMES.items()}
except ImportError:  # pragma: no cover - games.py ships with the repo
    KNOWN_FRAMES = {"BAR-ATH": 199319}


class ProbsFormatError(ValueError):
    """The file breaks the contract. The message carries aggregates only."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Column order, dtypes, clipping and rounding exactly as written to disk."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ProbsFormatError("missing columns: {}".format(missing))
    out = df[COLUMNS].copy()
    out["frame"] = out["frame"].astype(np.int64)
    out["ball_src"] = out["ball_src"].fillna(0).astype(np.int64)
    for c in FLOAT_COLS:
        out[c] = out[c].astype(np.float64)
    for c in STATE_COLS + ["p_change", "ball_conf"]:
        out[c] = out[c].clip(0.0, 1.0)
    out[FLOAT_COLS] = out[FLOAT_COLS].round(DECIMALS)
    # -0.0 would print as "-0.0000"; normalise it so equal values always give equal bytes.
    out[FLOAT_COLS] = out[FLOAT_COLS] + 0.0
    return out.reset_index(drop=True)


def validate_probs(df: pd.DataFrame, game: str | None = None, n_frames: int | None = None) -> dict:
    """Check the contract. Returns aggregate statistics; raises ProbsFormatError on a breach."""
    errors = []
    cols = list(df.columns)
    if cols != COLUMNS:
        errors.append("columns must be exactly {} (got {})".format(COLUMNS, cols))
        raise ProbsFormatError("; ".join(errors))

    n = len(df)
    expected = n_frames if n_frames is not None else KNOWN_FRAMES.get(game) if game else None
    if expected is not None and n != expected:
        errors.append("{} rows, expected {}".format(n, expected))
    frame = df["frame"].to_numpy()
    if n == 0:
        raise ProbsFormatError("empty file")
    if not np.array_equal(frame, np.arange(n)):
        errors.append("frame must be 0..{} contiguous and ascending".format(n - 1))

    p = df[STATE_COLS].to_numpy(dtype=np.float64)
    nan_rows = np.isnan(p)
    partial = nan_rows.any(axis=1) & ~nan_rows.all(axis=1)
    if partial.any():
        errors.append("{} rows with some but not all of p_H/p_A/p_D empty".format(int(partial.sum())))
    pred = ~nan_rows.any(axis=1)
    if pred.any():
        dev = np.abs(p[pred].sum(axis=1) - 1.0)
        if dev.max() > SUM_TOL:
            errors.append("{} rows where p_H+p_A+p_D is off 1 by more than {} (max {:.4g})".format(
                int((dev > SUM_TOL).sum()), SUM_TOL, dev.max()))
        if (p[pred] < 0).any() or (p[pred] > 1).any():
            errors.append("state probabilities outside [0, 1]")

    pc = df["p_change"].to_numpy(dtype=np.float64)
    if np.isnan(pc[pred]).any():
        errors.append("p_change empty on rows that have state probabilities")
    if ((pc < 0) | (pc > 1)).any():
        errors.append("p_change outside [0, 1]")
    if pred[0] and pc[0] != 0:
        errors.append("p_change must be 0 on frame 0")

    bx, by = df["ball_x"].to_numpy(dtype=np.float64), df["ball_y"].to_numpy(dtype=np.float64)
    bc = df["ball_conf"].to_numpy(dtype=np.float64)
    src = df["ball_src"].to_numpy()
    has_ball = ~np.isnan(bx) & ~np.isnan(by)
    if (np.isnan(bx) != np.isnan(by)).any():
        errors.append("ball_x and ball_y must be empty together")
    lo, hi = BALL_RANGE
    if has_ball.any() and ((bx[has_ball] < lo) | (bx[has_ball] > hi) |
                           (by[has_ball] < lo) | (by[has_ball] > hi)).any():
        errors.append("ball position outside {} (normalised coordinates?)".format(BALL_RANGE))
    if np.isnan(bc).any() or ((bc < 0) | (bc > 1)).any():
        errors.append("ball_conf must be filled, in [0, 1]")
    if (bc[~has_ball] != 0).any():
        errors.append("ball_conf must be 0 where the ball position is empty")
    if not np.isin(src, list(BALL_SOURCES)).all():
        errors.append("ball_src must be one of {}".format(sorted(BALL_SOURCES)))
    if ((src == 0) != ~has_ball).any():
        errors.append("ball_src is 0 exactly where the ball position is empty")

    if errors:
        raise ProbsFormatError("; ".join(errors))

    argmax = np.full(n, "-", dtype=object)
    argmax[pred] = np.array(["H", "A", "D"])[p[pred].argmax(axis=1)]
    return {
        "rows": int(n),
        "rows_with_prediction": int(pred.sum()),
        "argmax_counts": {k: int((argmax == k).sum()) for k in ("H", "A", "D")},
        "mean_p": {c: round(float(np.nanmean(df[c])), 4) if pred.any() else None for c in STATE_COLS},
        "mean_p_change": round(float(np.nanmean(pc)), 5) if pred.any() else None,
        "ball_coverage": round(float(has_ball.mean()), 4),
        "ball_src_counts": {BALL_SOURCES[k]: int((src == k).sum()) for k in BALL_SOURCES},
    }


def write_probs(df: pd.DataFrame, path: str, game: str, meta: dict | None = None,
                n_frames: int | None = None) -> dict:
    """Validate and write the CSV plus its sidecar. Returns the sidecar dict."""
    out = _canonical(df)
    stats = validate_probs(out, game=game, n_frames=n_frames)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, index=False, float_format="%.{}f".format(DECIMALS), lineterminator="\n",
               na_rep="")
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "game": game,
        "rows": stats["rows"],
        "csv_sha256": sha256_file(path),
    }
    sidecar.update(meta or {})
    with open(path + ".meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
        f.write("\n")
    return sidecar


def read_probs(path: str, game: str | None = None, n_frames: int | None = None) -> pd.DataFrame:
    """Read and validate a probabilities file written by write_probs."""
    dtypes = {c: np.float64 for c in FLOAT_COLS}
    dtypes.update({"frame": np.int64, "ball_src": np.int64})
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    validate_probs(df, game=game, n_frames=n_frames)
    meta_path = path + ".meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ProbsFormatError("schema_version {} != {}".format(
                meta.get("schema_version"), SCHEMA_VERSION))
        if meta.get("csv_sha256") not in (None, sha256_file(path)):
            raise ProbsFormatError("csv_sha256 in the sidecar does not match the file")
    return df


def _synthetic(n: int, seed: int) -> pd.DataFrame:
    """Random but well-formed probabilities, for the self-test only."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, 3))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    has_ball = rng.random(n) < 0.6
    df = pd.DataFrame({
        "frame": np.arange(n),
        "p_H": p[:, 0], "p_A": p[:, 1], "p_D": p[:, 2],
        "p_change": np.r_[0.0, rng.random(n - 1) * 0.05],
        "ball_x": np.where(has_ball, rng.random(n), np.nan),
        "ball_y": np.where(has_ball, rng.random(n), np.nan),
        "ball_conf": np.where(has_ball, rng.random(n), 0.0),
        "ball_src": np.where(has_ball, rng.integers(1, 5, n), 0),
    })
    df.loc[:9, STATE_COLS + ["p_change"]] = np.nan  # a no-prediction stretch at the start
    return df


def selftest() -> None:
    n = 5000
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
        meta_a = write_probs(_synthetic(n, 0), a, game="SYNTH", n_frames=n, meta={"seed": 0})
        meta_b = write_probs(_synthetic(n, 0), b, game="SYNTH", n_frames=n, meta={"seed": 0})
        assert meta_a["csv_sha256"] == meta_b["csv_sha256"], "same input must give same bytes"
        back = read_probs(a, n_frames=n)
        assert list(back.columns) == COLUMNS and len(back) == n
        again = os.path.join(tmp, "again.csv")
        write_probs(back, again, game="SYNTH", n_frames=n, meta={"seed": 0})
        assert sha256_file(again) == meta_a["csv_sha256"], "read/write round trip must be stable"

        def must_fail(mutate, what):
            bad = _canonical(_synthetic(n, 0))
            mutate(bad)
            try:
                validate_probs(bad, n_frames=n)
            except ProbsFormatError:
                return
            raise AssertionError("validator accepted: " + what)

        must_fail(lambda d: d.drop(index=100, inplace=True), "a missing frame")
        must_fail(lambda d: d.loc.__setitem__((200, "p_H"), 0.9), "probabilities not summing to 1")
        must_fail(lambda d: d.loc.__setitem__((300, "p_A"), np.nan), "a partly empty row")
        must_fail(lambda d: d.loc.__setitem__((d.index[d.ball_src == 0][0], "ball_conf"), 0.5),
                  "confidence without a ball")
        try:
            validate_probs(_canonical(_synthetic(n, 0)), game="BAR-ATH")
            raise AssertionError("validator accepted the wrong length for BAR-ATH")
        except ProbsFormatError:
            pass
    print("probs_format selftest: OK (determinism, round trip, 5 rejection cases)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check a probabilities CSV; prints aggregates only")
    v.add_argument("path")
    v.add_argument("--game", help="known game name (checks the row count), e.g. BAR-ATH")
    v.add_argument("--n-frames", type=int, help="expected row count for an external game")
    sub.add_parser("selftest", help="run the built-in checks on synthetic data")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        selftest()
        return 0
    # Team stop rule (inference/quiet.py): for the provided game, pass/fail only.
    quiet = "BAR-ATH" in str(args.game).upper() or "BAR-ATH" in os.path.basename(args.path).upper()
    try:
        df = read_probs(args.path, game=args.game, n_frames=args.n_frames)
    except ProbsFormatError as e:
        print("INVALID" if quiet else "INVALID: {}".format(e))
        return 1
    stats = validate_probs(df, game=args.game, n_frames=args.n_frames)
    print("VALID")
    if not quiet:
        print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
