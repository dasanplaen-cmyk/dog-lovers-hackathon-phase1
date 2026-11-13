"""One command from the provided data to the submission file.

    python -m pipeline.run               # everything; run it where the data is
    python -m pipeline.run --from-probs  # only the last steps, from an existing probabilities file
    python -m pipeline.run --rules       # the fallback: no trained model, writes Dog_Lovers_rules.csv
    python -m pipeline.run --selftest    # synthetic probabilities, no match data, runs anywhere

Steps: model weights, game file, ball, possession probabilities, decoder, official validator. All
on CPU, which keeps the file byte-for-byte reproducible.

While the team's stop rule is on, no number computed from the provided game may reach an AI
assistant, so this prints step names, paths, sha256 and pass/fail only. The full output of every
step goes to <work>/logs/, which agents must not read, and the validator is judged by its exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import games  # noqa: E402
from decode.decoder import LETTERS, PARAMS_PATH, decode, load_params  # noqa: E402
from inference.probs_format import read_probs, write_probs  # noqa: E402
from inference.quiet import quiet_report  # noqa: E402

GAME = "BAR-ATH"
TEAM_FILE = "Dog_Lovers.csv"
# The models that make the submission. A new checkpoint also needs decode/params.json retuned.
BALL = "ball_v1"
POSSESSION = "possession_v3"
RULES_FILE = "Dog_Lovers_rules.csv"
RULES_PARAMS = os.path.join(REPO, "baseline", "decoder_params.json")
FUSION = os.path.join("ball", "configs", "fusion_v1.json")
VALIDATOR = os.path.join(REPO, "evaluation", "validate_submission.py")


def work_dir() -> str:
    return os.environ.get("MTAH_WORK_DIR") or os.path.join(REPO, "data", "work")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_play_blocks() -> list[tuple[int, int]]:
    """Half-open [start, stop) spans of each half, from the organisers' games.py (inclusive there)."""
    return [(a, b + 1) for _, a, b in games.half_video_blocks(GAME)]


def write_submission(probs: str, out: str, params: dict) -> None:
    df = read_probs(probs, game=GAME)
    p = df[["p_H", "p_A", "p_D"]].to_numpy(dtype=np.float64)
    pc = df["p_change"].to_numpy(dtype=np.float64)
    codes = decode(p, pc, params, float(games.GAMES[GAME]["fps"]), in_play_blocks())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes.tolist()))


def run_step(name: str, cmd: list[str], log: str) -> None:
    print("step  {}".format(name), flush=True)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== {}\n".format(name))
        f.flush()
        code = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise SystemExit("FAILED at step '{}' (details in {})".format(name, log))


def check_decoder_settings() -> None:
    """The decoder settings belong to one checkpoint: refuse to mix them with another."""
    with open(PARAMS_PATH, encoding="utf-8") as f:
        tuned_for = json.load(f).get("model")
    if tuned_for != POSSESSION:
        raise SystemExit("FAILED: decode/params.json was tuned for {}, not {}; retune it first".format(
            tuned_for, POSSESSION))


def finish(probs: str, out: str, log: str, params_path: str = PARAMS_PATH) -> None:
    """Decode twice to prove the file repeats, then let the organisers' validator judge it."""
    print("step  decoder", flush=True)
    params = load_params(params_path)
    again = out + ".check"
    try:
        write_submission(probs, out, params)
        write_submission(probs, again, params)
    except Exception as e:     # its message can hold figures from the game: log it, never print it
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n=== decoder\n{}\n".format(repr(e)))
        raise SystemExit("FAILED at step 'decoder' (details in {})".format(log)) from None
    same = sha256(out) == sha256(again)
    os.remove(again)
    if not same:
        raise SystemExit("FAILED at step 'decoder': two runs gave different files")
    run_step("official validator", [sys.executable, VALIDATOR, out], log)
    print(quiet_report(out))


def full(out: str, log: str, probs: str) -> None:
    check_decoder_settings()
    py = sys.executable
    run_step("model weights", [py, "-m", "inference.weights", "fetch"], log)
    from inference.weights import local_path
    run_step("game file", [py, "-m", "ball.sources", "bar-ath"], log)
    run_step("ball", [py, "-m", "ball.infer", "--weights", local_path(BALL), "--game", GAME,
                      "--device", "cpu", "--fusion", FUSION], log)
    run_step("possession probabilities", [py, "-m", "models.possession.infer", "--weights",
                                          local_path(POSSESSION), "--game", GAME, "--device", "cpu"], log)
    finish(probs, out, log)


def rules(out: str, log: str) -> None:
    py = sys.executable
    run_step("game file", [py, "-m", "ball.sources", "bar-ath"], log)
    run_step("rules", [py, "-m", "baseline.rules", "probs", GAME], log)
    finish(os.path.join(work_dir(), "probs_rules", GAME + "_probs.csv"), out, log, RULES_PARAMS)


def selftest() -> None:
    from inference.probs_format import _synthetic
    with tempfile.TemporaryDirectory() as tmp:
        n = games.source_frames()
        probs, out, log = (os.path.join(tmp, x) for x in ("probs.csv", TEAM_FILE, "logs/run.log"))
        write_probs(_synthetic(n, 0), probs, GAME)
        check_decoder_settings()
        finish(probs, out, log)
        finish(probs, os.path.join(tmp, RULES_FILE), log, RULES_PARAMS)
        rows = open(out, encoding="utf-8").read().splitlines()
        assert rows[0] == "frame,label" and len(rows) == n + 1, "header and row count"
        assert rows[1].startswith("0,") and rows[-1].startswith("{},".format(n - 1)), "frames in order"
        labels = np.array([r.split(",")[1] for r in rows[1:]])
        inside = np.zeros(n, bool)
        for a, b in in_play_blocks():
            inside[a:b] = True
        assert not (labels[inside] == "X").any(), "no X inside a half with predictions"
        assert (labels[~inside] == "X").all(), "X outside the halves"
        with open(probs, "a", encoding="utf-8") as f:
            f.write("broken\n")
        try:
            finish(probs, os.path.join(tmp, "bad.csv"), log)
        except SystemExit as e:
            assert str(e).startswith("FAILED at step 'decoder'"), "a bad file fails without details"
        else:
            raise AssertionError("a broken probabilities file was accepted")
        with open(out, "a", encoding="utf-8") as f:
            f.write("broken\n")
        try:
            run_step("validator rejects a broken file", [sys.executable, VALIDATOR, out], log)
        except SystemExit:
            print("pipeline selftest OK: file repeats, passes the validator, broken file is refused")
            return
    raise AssertionError("the validator accepted a broken file")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-probs", action="store_true", help="skip the models")
    ap.add_argument("--rules", action="store_true", help="fallback without trained models")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probs", default=None, help="only with --from-probs")
    args = ap.parse_args(argv)
    if args.probs and not args.from_probs:
        ap.error("--probs only works with --from-probs: the models always write the default path")
    probs = args.probs or os.path.join(work_dir(), "probs", GAME + "_probs.csv")
    log = os.path.join(work_dir(), "logs", "pipeline.log")
    out = args.out or os.path.join(work_dir(), "submission", RULES_FILE if args.rules else TEAM_FILE)
    if args.selftest:
        selftest()
    elif args.rules:
        rules(out, log)
    elif args.from_probs:
        check_decoder_settings()
        finish(probs, out, log)
    else:
        full(out, log, probs)


if __name__ == "__main__":
    main()
