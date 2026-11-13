"""Convert each source into a Game .npz under the work dir. Prints aggregates only.

    python -m ball.sources dfl                  # the 7 DFL open matches (downloads via kloppy)
    python -m ball.sources dfl J03WMX J03WN1    # some of them
    python -m ball.sources bar-ath              # the provided ATD; GX10 only, never labels

External games land in <work>/games/external/, BAR-ATH in <work>/games/test/. The work dir is
MTAH_WORK_DIR or data/work (gitignored, denied to coding agents).
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

from ball.game import from_kloppy  # noqa: E402
from training.runlog import work_dir  # noqa: E402

DFL_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def external_dir() -> str:
    return os.path.join(str(work_dir()), "games", "external")


def test_dir() -> str:
    return os.path.join(str(work_dir()), "games", "test")


def _report(game, path, t0):
    s = game.summary()
    s["seconds"] = round(time.time() - t0, 1)
    print("{} -> {}\n{}".format(game.name, path, json.dumps(s, sort_keys=True)))


DFL_DOI = "https://doi.org/10.6084/m9.figshare.28196177"


def find_dfl_files(folder: str, match: str) -> tuple[str, str]:
    """(match information XML, raw positions XML) of one match in a folder of DFL downloads."""
    names = [os.path.join(dp, f) for dp, _, files in os.walk(folder) for f in files
             if match in f and f.lower().endswith(".xml")]
    meta = [p for p in names if "matchinformation" in os.path.basename(p).lower()]
    raw = [p for p in names if "positions" in os.path.basename(p).lower()]
    if len(meta) != 1 or len(raw) != 1:
        raise SystemExit("{}: need exactly one *matchinformation*{}*.xml and one *positions*{}*.xml "
                         "under {}; found {} and {}".format(
                             match, match, match, folder,
                             [os.path.basename(p) for p in meta], [os.path.basename(p) for p in raw]))
    return meta[0], raw[0]


def convert_dfl(matches, folder=None):
    from kloppy import sportec
    for m in matches:
        t0 = time.time()
        if folder:
            meta, raw = find_dfl_files(folder, m)
            ds = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False)
        else:
            try:
                ds = sportec.load_open_tracking_data(match_id=m, only_alive=False)
            except Exception as e:  # figshare may serve an empty bot challenge instead of the file
                raise SystemExit(
                    "DFL {}: download through kloppy failed ({}: {}).\nDownload the match "
                    "information and raw positions XML files by hand from {} and run:\n"
                    "  python -m ball.sources dfl --dir <folder with the XML files>".format(
                        m, type(e).__name__, str(e)[:120], DFL_DOI))
        g = from_kloppy(ds, name="DFL-" + m)
        path = os.path.join(external_dir(), g.name + ".npz")
        g.save(path)
        _report(g, path, t0)


def convert_bar_ath():
    from starter.load_atd import load, check_alignment
    t0 = time.time()
    ds = load()
    report = check_alignment(ds)
    if not report["agrees"]:
        raise SystemExit("ATD alignment check failed: {}".format(report))
    g = from_kloppy(ds, name="BAR-ATH", with_labels=False)
    if g.frame_id[0] != 1 or (g.frame_id != 1 + np.arange(g.n_frames)).any():
        raise SystemExit("unexpected ATD frame ids; expected 1..n contiguous")
    path = os.path.join(test_dir(), "BAR-ATH.npz")
    g.save(path)
    from inference.quiet import quiet_report
    print(quiet_report(path))         # stop rule: alignment checked above; path, hash, pass only


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", choices=["dfl", "bar-ath"])
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--dir", help="DFL only: folder with hand-downloaded XML files (searched recursively)")
    args = ap.parse_args(argv)
    if args.source == "dfl":
        convert_dfl(args.matches or DFL_MATCHES, args.dir)
    else:
        convert_bar_ath()


if __name__ == "__main__":
    main()
