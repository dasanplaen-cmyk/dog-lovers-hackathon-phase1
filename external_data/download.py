"""Download external (other games) datasets into data/external/ (gitignored, denied to coding agents).

    python -m external_data.download dfl                 # the 7 DFL / Sportec open matches
    python -m external_data.download dfl J03WOH J03WQQ   # some of them

DFL files come straight from figshare's file endpoint (ndownloader.figshare.com), which serves the
XML without the bot challenge that kloppy's `load_open_tracking_data` hits on
springernature.figshare.com. Files already present with the expected size are skipped.

    python -m external_data.download metrica             # Metrica sample games 1, 2, 3 (dropped:
                                                         # no licence; kept for the record)

Metrica files come from the sample-data GitHub repository (raw files, via curl), so loading does
not depend on Python's SSL certificates (kloppy's own fetch fails on python.org macOS builds).

    python -m external_data.download skillcorner         # the 20 SkillCorner open matches
    python -m external_data.download skillcorner 1874553 # some of them

SkillCorner files come from one pinned commit of its GitHub repository. Tracking is stored with Git
LFS: the plain URL gives a pointer with the expected size, the media URL gives the file.
Sources and licences: external_data/SOURCES.md.
Prints file names and sizes only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_RAW = os.path.join(REPO, "data", "external")

DFL_ARTICLE = "https://api.figshare.com/v2/articles/28196177"
DFL_MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
# Per match: match information (teams, pitch), raw positions (tracking), events.
DFL_KINDS = ("matchinformation", "positions_raw_observed", "events_raw")


def _get(url: str) -> bytes:
    # curl rather than urllib: it uses the system certificates on macOS and Linux alike.
    return subprocess.run(["curl", "-sSfL", "--retry", "3", url], check=True,
                          capture_output=True).stdout


def _download(url: str, path: str, size: int) -> str:
    if os.path.exists(path) and os.path.getsize(path) == size:
        return "skip"
    tmp = path + ".part"
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", tmp, url], check=True)
    got = os.path.getsize(tmp)
    if got != size:
        os.remove(tmp)
        raise SystemExit("{}: expected {} bytes, got {} (bot challenge page?)".format(
            os.path.basename(path), size, got))
    os.replace(tmp, path)
    return "ok"


def download_dfl(matches: list[str], folder: str) -> None:
    files = json.loads(_get(DFL_ARTICLE))["files"]
    os.makedirs(folder, exist_ok=True)
    for m in matches:
        for kind in DFL_KINDS:
            hits = [f for f in files if m in f["name"] and kind in f["name"]]
            if len(hits) != 1:
                raise SystemExit("DFL {}: expected one '{}' file, found {}".format(m, kind, len(hits)))
            f = hits[0]
            status = _download(f["download_url"], os.path.join(folder, f["name"]), f["size"])
            print("{:4} {:>8.1f} MB  {}".format(status, f["size"] / 1e6, f["name"]))


METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{g}/{f}"
METRICA_FILES = {
    "1": ["Sample_Game_1_RawTrackingData_Home_Team.csv", "Sample_Game_1_RawTrackingData_Away_Team.csv",
          "Sample_Game_1_RawEventsData.csv"],
    "2": ["Sample_Game_2_RawTrackingData_Home_Team.csv", "Sample_Game_2_RawTrackingData_Away_Team.csv",
          "Sample_Game_2_RawEventsData.csv"],
    "3": ["Sample_Game_3_tracking.txt", "Sample_Game_3_metadata.xml", "Sample_Game_3_events.json"],
}


def download_metrica(games: list[str], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for g in games:
        for f in METRICA_FILES[g]:
            path = os.path.join(folder, f)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                status = "skip"
            else:
                subprocess.run(["curl", "-sSfL", "--retry", "3", "--speed-limit", "10000", "--speed-time", "60", "-o", path + ".part",
                                METRICA_RAW.format(g=g, f=f)], check=True)
                os.replace(path + ".part", path)
                status = "ok"
            print("{:4} {:>8.1f} MB  {}".format(status, os.path.getsize(path) / 1e6, f))


SKILLCORNER_REV = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/" + SKILLCORNER_REV + "/data/matches/{m}/{f}"
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/" + SKILLCORNER_REV + "/data/matches/{m}/{f}"
SKILLCORNER_MATCHES = ["1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846",
                       "1986691", "1996435", "1996436", "2006229", "2006363", "2007448", "2007721",
                       "2010085", "2011166", "2013725", "2015213", "2016236", "2017461"]


def download_skillcorner(matches: list[str], folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for m in matches:
        for f in ("{}_match.json".format(m), "{}_phases_of_play.csv".format(m)):
            path = os.path.join(folder, f)
            status = "skip"
            if not (os.path.exists(path) and os.path.getsize(path) > 0):
                with open(path + ".part", "wb") as out:
                    out.write(_get(SKILLCORNER_RAW.format(m=m, f=f)))
                os.replace(path + ".part", path)
                status = "ok"
            print("{:4} {:>8.1f} MB  {}".format(status, os.path.getsize(path) / 1e6, f))
        f = "{}_tracking_extrapolated.jsonl".format(m)
        pointer = _get(SKILLCORNER_RAW.format(m=m, f=f)).decode("ascii", "replace")
        size = re.search(r"^size (\d+)$", pointer, re.M)
        if not size:
            raise SystemExit("{}: expected a Git LFS pointer".format(f))
        status = _download(SKILLCORNER_LFS.format(m=m, f=f), os.path.join(folder, f), int(size.group(1)))
        print("{:4} {:>8.1f} MB  {}".format(status, int(size.group(1)) / 1e6, f))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", choices=["dfl", "metrica", "skillcorner"])
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--dir", default=None, help="target folder (default data/external/<source>)")
    args = ap.parse_args(argv)
    if args.source == "dfl":
        unknown = sorted(set(args.matches) - set(DFL_MATCHES))
        if unknown:
            raise SystemExit("unknown DFL match ids: {}".format(unknown))
        download_dfl(args.matches or DFL_MATCHES, args.dir or os.path.join(EXTERNAL_RAW, "dfl"))
    elif args.source == "skillcorner":
        unknown = sorted(set(args.matches) - set(SKILLCORNER_MATCHES))
        if unknown:
            raise SystemExit("unknown SkillCorner match ids: {}".format(unknown))
        download_skillcorner(args.matches or SKILLCORNER_MATCHES,
                             args.dir or os.path.join(EXTERNAL_RAW, "skillcorner"))
    else:
        games = args.matches or sorted(METRICA_FILES)
        unknown = sorted(set(games) - set(METRICA_FILES))
        if unknown:
            raise SystemExit("unknown Metrica games: {}".format(unknown))
        download_metrica(games, args.dir or os.path.join(EXTERNAL_RAW, "metrica"))


if __name__ == "__main__":
    sys.exit(main())
