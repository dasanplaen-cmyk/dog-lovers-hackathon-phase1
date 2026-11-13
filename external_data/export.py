"""Convert external games into per-game arrays for the models. Prints aggregates only.

    python -m external_data.export dfl                   # all 7 DFL matches (one process each)
    python -m external_data.export dfl J03WOH            # one match
    python -m external_data.export metrica 1 2 3         # Metrica sample games, event-derived labels
                                                         # (dropped: no licence; kept for the record)
    python -m external_data.export skillcorner 1874553   # SkillCorner open matches (default: all 20)

Input: files under data/external/<source>/ (see external_data/download.py). Output: <work>/games/external/<NAME>.npz, where <work> is MTAH_WORK_DIR or data/work
(gitignored, denied to coding agents). Only external games: the provided game is never handled here.

File format (keys of the .npz). The first block is exactly the `Game` format of ball/game.py, so
`Game.load` reads these files unchanged:

    name      str    []        e.g. "DFL-J03WOH", "SC-1874553"
    fps       f32    []        25 for all sources (SkillCorner is resampled from 10)
    pitch     f32    [2]       length, width in metres
    frame_id  int64  [T]       the source's own frame counter, in record order
                               (SkillCorner: source frame x 2.5)
    period    int8   [T]       1, 2 (extra time 3, 4)
    pos       f32    [T,P,2]   player x, y in metres, pitch centre at (0, 0), x along the length;
                               NaN = not tracked. 0-1 like the ATD: pos / pitch + 0.5
    team      int8   [P]       0 home, 1 away, 2 other / unknown
    ball      f32    [T,3]     ball x, y (metres, same frame as pos), z (metres; NaN if unknown);
                               x, y NaN = ball not observed
    label     int8   [T]       0 H, 1 A, 2 D (dead ball), -1 unknown

Extra keys (ignored by Game.load):

    ball_observed     bool [T]       ball x, y present on this frame
    home_attack_dir   int8 [n_per]   per period in sorted order: +1 home attacks towards +x,
                                     -1 towards -x, 0 ambiguous (goalkeeper rule, see attack_dirs)
    periods           int8 [n_per]   the periods home_attack_dir refers to
    label_source      str  []        how labels were made, for the technical document

Labels follow CHALLENGE.md: H / A = that team in possession with the ball alive, D = dead ball
(no owner asked for), X = no answer. DFL: provider's per-frame BallStatus and BallPossession
(alive and home -> H, alive and away -> A, dead -> D). Metrica: derived from the provider's events
by `metrica_labels` (rules in METRICA_RULES), an algorithm, no human labelling. SkillCorner: derived
from the provider's possession group and phases of play by `skillcorner_labels` (SKILLCORNER_RULES).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from external_data.download import DFL_MATCHES, EXTERNAL_RAW, SKILLCORNER_MATCHES  # noqa: E402

LABELS = {"H": 0, "A": 1, "D": 2, "X": -1}


def work_dir() -> str:
    return os.environ.get("MTAH_WORK_DIR") or os.path.join(REPO, "data", "work")


def out_dir() -> str:
    return os.path.join(work_dir(), "games", "external")


def attack_dirs(pos: np.ndarray, team: np.ndarray, period: np.ndarray):
    """Per period: +1 if home attacks towards +x, -1 towards -x.

    Goalkeeper rule (the idea of models/possession/data.py, issue #11): in each period, each team's
    goalkeeper is its well-tracked slot (tracked on at least half the period's frames) with the
    most extreme mean x. Home attacks away from its own goalkeeper. A kick-off rule is not used:
    it gave wrong signs on DFL, where teams switch sides at half-time.
    """
    periods = np.unique(period)
    dirs = []
    for p in periods:
        rows = period == p
        px = pos[rows][..., 0]                              # [t, P]
        tracked = np.isfinite(px).mean(axis=0)              # [P]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # slots never tracked in this period
            mean_x = np.nanmean(px, axis=0)
        ok = (tracked >= 0.5) & np.isfinite(mean_x)
        home = ok & (team == 0)
        away = ok & (team == 1)
        if not home.any():
            dirs.append(0)
            continue
        gk_home = mean_x[home][np.argmax(np.abs(mean_x[home]))]
        if away.any():
            gk_away = mean_x[away][np.argmax(np.abs(mean_x[away]))]
            if np.sign(gk_home) == np.sign(gk_away):
                dirs.append(0)                              # ambiguous: flag rather than guess
                continue
        dirs.append(1 if gk_home < 0 else -1)
    return periods.astype(np.int8), np.array(dirs, dtype=np.int8)


def from_kloppy(dataset, name: str, label_source: str, with_labels: bool) -> dict:
    """kloppy TrackingDataset (normalised 0-1 coordinates) -> dict of arrays (see module doc)."""
    from kloppy.domain import Ground

    md = dataset.metadata
    if not getattr(md.coordinate_system, "normalized", False):
        raise ValueError("{}: expected kloppy's normalised coordinates".format(name))
    length = float(getattr(md.pitch_dimensions, "pitch_length", None) or 105.0)
    width = float(getattr(md.pitch_dimensions, "pitch_width", None) or 68.0)
    df = dataset.to_df()

    players, team, home_id, away_id = [], [], None, None
    for t in md.teams:
        side = 0 if t.ground == Ground.HOME else 1 if t.ground == Ground.AWAY else 2
        if side == 0:
            home_id = t.team_id
        elif side == 1:
            away_id = t.team_id
        for p in t.players:
            if "{}_x".format(p.player_id) in df.columns:
                players.append(p.player_id)
                team.append(side)

    T, P = len(df), len(players)
    pos = np.full((T, P, 2), np.nan, dtype=np.float32)
    for j, pid in enumerate(players):
        pos[:, j, 0] = (df["{}_x".format(pid)].to_numpy(dtype=np.float64) - 0.5) * length
        pos[:, j, 1] = (df["{}_y".format(pid)].to_numpy(dtype=np.float64) - 0.5) * width

    ball = np.full((T, 3), np.nan, dtype=np.float32)
    ball[:, 0] = (df["ball_x"].to_numpy(dtype=np.float64) - 0.5) * length
    ball[:, 1] = (df["ball_y"].to_numpy(dtype=np.float64) - 0.5) * width
    if "ball_z" in df.columns:
        ball[:, 2] = df["ball_z"].astype("float64").to_numpy()

    label = np.full(T, LABELS["X"], dtype=np.int8)
    if with_labels:
        state = df["ball_state"].map(lambda s: getattr(s, "value", s)).astype(object).to_numpy()
        owner = df["ball_owning_team_id"].astype(object).to_numpy()
        alive, dead = state == "alive", state == "dead"
        label[alive & (owner == home_id)] = LABELS["H"]
        label[alive & (owner == away_id)] = LABELS["A"]
        label[dead] = LABELS["D"]

    team_arr = np.array(team, dtype=np.int8)
    period = df["period_id"].to_numpy(dtype=np.int8)
    fps = float(md.frame_rate)
    periods, dirs = attack_dirs(pos, team_arr, period)
    return {
        "name": np.array(name), "fps": np.float32(fps),
        "pitch": np.array([length, width], np.float32),
        "frame_id": df["frame_id"].to_numpy(dtype=np.int64), "period": period,
        "pos": pos, "team": team_arr, "ball": ball, "label": label,
        "ball_observed": ~np.isnan(ball[:, 0]), "home_attack_dir": dirs, "periods": periods,
        "label_source": np.array(label_source),
    }


def summary(g: dict) -> dict:
    """Aggregates only."""
    tracked = (~np.isnan(g["pos"][..., 0])).sum(axis=1)
    lab = g["label"]
    s = {
        "name": str(g["name"]), "frames": int(len(lab)), "fps": float(g["fps"]),
        "minutes": round(len(lab) / float(g["fps"]) / 60, 1),
        "player_slots": int(g["pos"].shape[1]),
        "team_slots": {k: int((g["team"] == i).sum()) for i, k in enumerate(("home", "away", "other"))},
        "mean_tracked_players": round(float(tracked.mean()), 2),
        "ball_observed": round(float(g["ball_observed"].mean()), 4),
        "home_attack_dir": dict(zip(g["periods"].tolist(), g["home_attack_dir"].tolist())),
        "label_share": {k: round(float((lab == v).mean()), 4) for k, v in LABELS.items()},
    }
    # H/A spells while the ball is alive: count and mean length in seconds
    alive = (lab == LABELS["H"]) | (lab == LABELS["A"])
    change = np.flatnonzero((np.diff(lab) != 0) | (np.diff(g["period"]) != 0)) + 1
    starts = np.r_[0, change]
    lens = np.diff(np.r_[starts, len(lab)])
    spell = alive[starts]
    if spell.any():
        s["possession_spells"] = int(spell.sum())
        s["spell_mean_s"] = round(float(lens[spell].mean() / float(g["fps"])), 2)
    return s


def save(g: dict) -> str:
    os.makedirs(out_dir(), exist_ok=True)
    path = os.path.join(out_dir(), str(g["name"]) + ".npz")
    np.savez_compressed(path, **g)
    return path


def find_dfl_files(folder: str, match: str) -> tuple[str, str]:
    names = sorted(f for f in os.listdir(folder) if match in f and f.endswith(".xml"))
    meta = [f for f in names if "matchinformation" in f]
    raw = [f for f in names if "positions_raw" in f]
    if len(meta) != 1 or len(raw) != 1:
        raise SystemExit("DFL {}: missing XML in {}; run: python -m external_data.download dfl {}"
                         .format(match, folder, match))
    return os.path.join(folder, meta[0]), os.path.join(folder, raw[0])


def export_dfl_one(match: str, folder: str) -> None:
    from kloppy import sportec
    t0 = time.time()
    meta, raw = find_dfl_files(folder, match)
    ds = sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False)
    g = from_kloppy(ds, "DFL-" + match, with_labels=True,
                    label_source="DFL provider per-frame BallStatus + BallPossession")
    del ds
    path = save(g)
    s = summary(g)
    s["seconds"] = round(time.time() - t0, 1)
    print("{} -> {}\n{}".format(s["name"], os.path.relpath(path, REPO), json.dumps(s, sort_keys=True)),
          flush=True)


# --- Metrica: H/A/D labels derived from the provider's events by this code (no human labels) ---

# Events that show a team controlling the ball (sets the owner from their start frame).
METRICA_CONTROL = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "CARRY"}
METRICA_RULES = (
    "Metrica events -> labels (algorithm): dead (D) from a BALL OUT end, a FAULT RECEIVED start, "
    "or the end of a SHOT that went OUT or was a GOAL, until the start of the next control event; "
    "and before every restart (a SET PIECE, or a PASS tagged GOAL KICK, except the first event of "
    "a period), from the end of the last action before it (a CARD does not count as an action, "
    "nor does the goalkeeper's RECOVERY before a goal kick) until the restart. "
    "Otherwise alive, owned (H/A) by the team of the latest PASS / SET PIECE / SHOT / RECOVERY / "
    "BALL LOST / CARRY start (CHALLENGE ignored); unknown (-1) before the first and after the last "
    "event of each period."
)


def metrica_events(folder: str, game: str, home_id: str) -> list[tuple]:
    """(period, start_frame, end_frame, side 0 home / 1 away, type, subtype) sorted by start."""
    rows = []
    if game in ("1", "2"):
        import pandas as pd
        df = pd.read_csv(os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game)))
        for r in df.itertuples(index=False):
            side = 0 if r.Team == "Home" else 1
            end = int(r[6]) if r[6] == r[6] else int(r[4])            # End Frame (NaN -> start)
            rows.append((int(r.Period), int(r[4]), end, side, str(r.Type), str(r.Subtype)))
    else:
        data = json.load(open(os.path.join(folder, "Sample_Game_3_events.json")))["data"]
        for e in data:
            sub = e.get("subtypes")
            subs = [x["name"] for x in sub] if isinstance(sub, list) else [sub["name"]] if sub else []
            side = 0 if e["team"]["id"] == home_id else 1
            start = int(e["start"]["frame"])
            end = int(e["end"]["frame"]) if e.get("end") and e["end"].get("frame") is not None else start
            rows.append((int(e["period"]), start, end, side, e["type"]["name"], "-".join(subs)))
    return sorted(rows, key=lambda r: (r[0], r[1], r[2]))


def metrica_labels(frame_id: np.ndarray, period: np.ndarray, events: list[tuple]) -> np.ndarray:
    """Per-frame labels from events (see METRICA_RULES). Frames are the tracking frame ids."""
    label = np.full(len(frame_id), LABELS["X"], dtype=np.int8)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        f = frame_id[rows]
        ev = [e for e in events if e[0] == p]
        if not ev:
            continue
        first, last = ev[0][1], max(e[2] for e in ev)
        # owner: team of the latest control event at or before each frame
        ctrl = [e for e in ev if e[4] in METRICA_CONTROL]
        owner = np.full(len(f), -1, dtype=np.int8)
        if ctrl:
            starts = np.array([e[1] for e in ctrl])
            sides = np.array([e[3] for e in ctrl], dtype=np.int8)
            idx = np.searchsorted(starts, f, side="right") - 1
            owner = np.where(idx >= 0, sides[np.clip(idx, 0, None)], -1).astype(np.int8)
        lab = np.where(owner == 0, LABELS["H"], np.where(owner == 1, LABELS["A"], LABELS["X"]))
        # dead-ball intervals end at the next control event (Metrica has no GOAL KICK set piece,
        # so waiting for the next SET PIECE would run on through goal kicks)
        restarts = np.array(sorted(e[1] for e in ctrl))
        for e in ev:
            t, sub = e[4], e[5].upper()
            if t == "BALL OUT":
                d0 = e[2]
            elif t == "FAULT RECEIVED":
                d0 = e[1]
            elif t == "SHOT" and ("OUT" in sub or "GOAL" in sub):
                d0 = e[2]
            else:
                continue
            k = np.searchsorted(restarts, d0, side="right")
            d1 = restarts[k] if k < len(restarts) else last + 1
            lab[(f >= d0) & (f < d1)] = LABELS["D"]
        # every restart means the ball was dead since the last action before it, whatever event
        # (a foul in a CHALLENGE, a BALL LOST for offside or handball) marked the stoppage
        for i, e in enumerate(ev):
            goal_kick = e[4] == "PASS" and "GOAL KICK" in e[5].upper()
            if i == 0 or not (goal_kick or e[4] == "SET PIECE"):
                continue
            skip = {"CARD", "RECOVERY"} if goal_kick else {"CARD"}
            j = i - 1
            while j > 0 and ev[j][4] in skip:
                j -= 1
            before = ev[j]
            d0 = before[1] if before[4] == "FAULT RECEIVED" else before[2]
            lab[(f >= min(d0, e[1])) & (f < e[1])] = LABELS["D"]
        lab[(f < first) | (f > last)] = LABELS["X"]
        label[rows] = lab
    return label


def export_metrica_one(match: str, folder: str) -> None:
    from kloppy import metrica
    t0 = time.time()
    if match in ("1", "2"):
        ds = metrica.load_tracking_csv(
            home_data=os.path.join(folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(match)),
            away_data=os.path.join(folder, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(match)))
    else:
        ds = metrica.load_tracking_epts(meta_data=os.path.join(folder, "Sample_Game_3_metadata.xml"),
                                        raw_data=os.path.join(folder, "Sample_Game_3_tracking.txt"))
    from kloppy.domain import Ground
    home_id = next(t.team_id for t in ds.metadata.teams if t.ground == Ground.HOME)
    g = from_kloppy(ds, "METRICA-{}".format(match), with_labels=False, label_source=METRICA_RULES)
    del ds
    g["label"] = metrica_labels(g["frame_id"], g["period"], metrica_events(folder, match, home_id))
    path = save(g)
    s = summary(g)
    s["seconds"] = round(time.time() - t0, 1)
    print("{} -> {}\n{}".format(s["name"], os.path.relpath(path, REPO), json.dumps(s, sort_keys=True)),
          flush=True)


SKILLCORNER_RULES = ("SkillCorner possession group: home or away team gives H or A. No group inside "
                     "a phase of play: a loose ball, kept by the last team of that half. No group "
                     "outside every phase of play: D. Resampled from 10 to 25 fps.")
SKILLCORNER_RATIO = 2.5                 # 25 fps / 10 fps
SKILLCORNER_SIDES = {"home team": 0, "away team": 1}
SKILLCORNER_DIRS = {"left_to_right": 1, "right_to_left": -1}


def read_skillcorner(match: dict, path: str) -> tuple:
    """Tracking at 10 fps. Ball only where detected; players as provided (extrapolated off camera)."""
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    home = match["home_team"]["id"]
    slot = {p["id"]: j for j, p in enumerate(match["players"])}
    team = np.array([0 if p["team_id"] == home else 1 for p in match["players"]], dtype=np.int8)
    frame, period = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int8)
    pos = np.full((n, len(slot), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    group = np.full(n, -1, dtype=np.int8)
    t = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("period"):             # before kick-off, half-time, after the end
                continue
            frame[t], period[t] = d["frame"], d["period"]
            b = d.get("ball_data") or {}
            if b.get("is_detected") and b.get("x") is not None:
                z = b.get("z")
                ball[t] = b["x"], b["y"], np.nan if z is None else z
            group[t] = SKILLCORNER_SIDES.get((d.get("possession") or {}).get("group"), -1)
            for pl in d.get("player_data") or []:
                if pl.get("x") is not None:
                    pos[t, slot[pl["player_id"]]] = pl["x"], pl["y"]
            t += 1
    if not (np.diff(frame[:t]) > 0).all():
        raise ValueError("{}: frames out of order".format(os.path.basename(path)))
    seen = np.isfinite(pos[:t, :, 0]).any(axis=0)       # squad players who never came on
    return frame[:t], period[:t], pos[:t][:, seen], team[seen], ball[:t], group[:t]


def skillcorner_labels(frame: np.ndarray, period: np.ndarray, group: np.ndarray,
                       phases: list[tuple[int, int]]) -> np.ndarray:
    """Per-frame labels (see SKILLCORNER_RULES). group: 0 home, 1 away, -1 none."""
    in_phase = np.zeros(len(frame), dtype=bool)
    for start, end in phases:
        in_phase[(frame >= start) & (frame <= end)] = True
    known = group >= 0
    label = np.full(len(frame), LABELS["D"], dtype=np.int8)
    label[known] = group[known]                         # LABELS H and A are 0 and 1
    last = np.maximum.accumulate(np.where(known, np.arange(len(frame)), -1))
    loose = ~known & in_phase & (last >= 0) & (period[last] == period)
    label[loose] = group[last[loose]]
    return label


def resample_skillcorner(frame: np.ndarray, period: np.ndarray) -> tuple:
    """25 fps ticks inside each half and a function that interpolates [T, ...] arrays onto them.

    Linear between two consecutive source frames; NaN if either is missing, or inside a hole in the
    source frames. A tick on a source frame takes that frame's value.
    """
    ticks = []
    for p in np.unique(period):
        f = frame[period == p]
        ticks.append(np.arange(np.ceil(f[0] * SKILLCORNER_RATIO), np.floor(f[-1] * SKILLCORNER_RATIO) + 1))
    tick = np.concatenate(ticks).astype(np.int64)
    x = tick / SKILLCORNER_RATIO
    i = np.searchsorted(frame, x, side="right") - 1
    j = np.minimum(i + 1, len(frame) - 1)
    exact = x == frame[i]
    next_frame = frame[j] - frame[i] == 1
    w = (x - frame[i]).astype(np.float32)

    def up(a: np.ndarray) -> np.ndarray:
        weight = w.reshape((-1,) + (1,) * (a.ndim - 1))
        out = a[i] * (1 - weight) + a[j] * weight
        out[exact] = a[i[exact]]
        out[~exact & ~next_frame] = np.nan
        return out

    return tick, i, up


def export_skillcorner_one(match: str, folder: str) -> None:
    t0 = time.time()
    with open(os.path.join(folder, match + "_match.json"), encoding="utf-8") as f:
        info = json.load(f)
    frame, period, pos, team, ball, group = read_skillcorner(
        info, os.path.join(folder, match + "_tracking_extrapolated.jsonl"))
    with open(os.path.join(folder, match + "_phases_of_play.csv"), encoding="utf-8", newline="") as f:
        phases = [(int(r["frame_start"]), int(r["frame_end"])) for r in csv.DictReader(f)]
    label = skillcorner_labels(frame, period, group, phases)
    tick, i, up = resample_skillcorner(frame, period)
    pos, ball, label, period = up(pos), up(ball), label[i], period[i]
    periods, dirs = attack_dirs(pos, team, period)
    g = {"name": np.array("SC-" + match), "fps": np.float32(25),
         "pitch": np.array([info["pitch_length"], info["pitch_width"]], np.float32),
         "frame_id": tick, "period": period, "pos": pos, "team": team, "ball": ball, "label": label,
         "ball_observed": ~np.isnan(ball[:, 0]), "home_attack_dir": dirs, "periods": periods,
         "label_source": np.array(SKILLCORNER_RULES)}
    path = save(g)
    s = summary(g)
    s["seconds"] = round(time.time() - t0, 1)
    s["provider_home_attack_dir"] = [SKILLCORNER_DIRS.get(d, 0) for d in info.get("home_team_side", [])]
    print("{} -> {}\n{}".format(s["name"], os.path.relpath(path, REPO), json.dumps(s, sort_keys=True)),
          flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", choices=["dfl", "metrica", "skillcorner"])
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--dir", default=None, help="folder with the downloaded files")
    args = ap.parse_args(argv)

    if args.source == "dfl":
        matches = args.matches or DFL_MATCHES
        folder = args.dir or os.path.join(EXTERNAL_RAW, "dfl")
        if len(matches) == 1:
            for m in matches:
                export_dfl_one(m, folder)
        else:
            # One process per match: a DFL match needs ~4 GB while parsing, freed on exit.
            for m in matches:
                cmd = [sys.executable, "-m", "external_data.export", "dfl", m, "--dir", folder]
                subprocess.run(cmd, check=True, cwd=REPO)
    elif args.source == "skillcorner":
        folder = args.dir or os.path.join(EXTERNAL_RAW, "skillcorner")
        for m in args.matches or SKILLCORNER_MATCHES:
            export_skillcorner_one(m, folder)
    else:
        folder = args.dir or os.path.join(EXTERNAL_RAW, "metrica")
        for m in args.matches or ["1", "2", "3"]:
            export_metrica_one(m, folder)


if __name__ == "__main__":
    main()
