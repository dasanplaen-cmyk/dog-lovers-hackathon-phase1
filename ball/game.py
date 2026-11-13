"""One game as plain arrays: the common input of the ball and sequence models.

Every source (DFL, SkillCorner, the BAR-ATH ATD) is converted once into a `.npz` with:

    frame_id  int64  [T]      the source's own frame counter, in record order
    period    int8   [T]      1, 2 (extra time 3, 4)
    pos       f32    [T,P,2]  player positions in metres, pitch centre at (0, 0); NaN = not tracked
    team      int8   [P]      0 home, 1 away, 2 other / unknown
    ball      f32    [T,3]    ball x, y, z in metres; NaN = not observed (z NaN when unknown)
    label     int8   [T]      0 H, 1 A, 2 D, -1 unknown. ALWAYS -1 for BAR-ATH: we have no labels
                              for it and must never load any.
    pitch     f32    [2]      length, width in metres; fps f32 []; name str []

Row r of the arrays is row r of the source's records. For the BAR-ATH tactical ATD that is video
frame r (records are numbered 1..199319, video_frame = frame_id - 1; see starter/load_atd.py).

Metres come from kloppy's normalised coordinates: x_m = (x - 0.5) * length, y_m = (y - 0.5) *
width. `to_normalised` is the exact inverse, so outputs go back to the ATD's 0-1 convention.

This is the ball module's own adapter. When Ying's external_data/ scripts land, sources.py reads
their files instead and this format stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

LABELS = {"H": 0, "A": 1, "D": 2, "X": -1}


@dataclass
class Game:
    name: str
    fps: float
    pitch: np.ndarray       # [2] length, width (m)
    frame_id: np.ndarray    # [T]
    period: np.ndarray      # [T]
    pos: np.ndarray         # [T, P, 2]
    team: np.ndarray        # [P]
    ball: np.ndarray        # [T, 3]
    label: np.ndarray       # [T]

    @property
    def n_frames(self) -> int:
        return len(self.frame_id)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, name=np.array(self.name), fps=np.float32(self.fps),
                            pitch=self.pitch.astype(np.float32), frame_id=self.frame_id,
                            period=self.period, pos=self.pos, team=self.team, ball=self.ball,
                            label=self.label)

    @classmethod
    def load(cls, path: str) -> "Game":
        z = np.load(path, allow_pickle=False)
        return cls(name=str(z["name"]), fps=float(z["fps"]), pitch=z["pitch"],
                   frame_id=z["frame_id"], period=z["period"], pos=z["pos"], team=z["team"],
                   ball=z["ball"], label=z["label"])

    def segments(self) -> list[tuple[int, int]]:
        """[start, end) row ranges of continuous play: same period, consecutive frame ids."""
        brk = np.flatnonzero((np.diff(self.period) != 0) | (np.diff(self.frame_id) != 1)) + 1
        edges = np.r_[0, brk, self.n_frames]
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def velocities(self) -> np.ndarray:
        """[T,P,2] player velocity in m/s by central difference inside each segment; NaN if unknown."""
        v = np.full_like(self.pos, np.nan)
        for a, b in self.segments():
            p = self.pos[a:b]
            if b - a >= 3:
                v[a + 1:b - 1] = (p[2:] - p[:-2]) * (self.fps / 2.0)
            if b - a >= 2:
                v[a] = (p[1] - p[0]) * self.fps
                v[b - 1] = (p[-1] - p[-2]) * self.fps
        return v

    def to_normalised(self, xy_m: np.ndarray) -> np.ndarray:
        """Metres (centred) back to the 0-1 coordinates of the source (= raw ATD for EPTS)."""
        return xy_m / self.pitch + 0.5

    def summary(self) -> dict:
        """Aggregates only: safe to print for any game, BAR-ATH included."""
        tracked = (~np.isnan(self.pos[..., 0])).sum(axis=1)
        out = {
            "name": self.name, "frames": self.n_frames, "fps": self.fps,
            "pitch": [round(float(v), 1) for v in self.pitch], "slots": int(self.pos.shape[1]),
            "segments": len(self.segments()), "periods": sorted(int(p) for p in np.unique(self.period)),
            "mean_tracked_players": round(float(tracked.mean()), 2),
            "ball_observed": round(float((~np.isnan(self.ball[:, 0])).mean()), 4),
            "team_slots": {k: int((self.team == i).sum()) for i, k in enumerate(("home", "away", "other"))},
        }
        if (self.label >= 0).any():
            out["labels"] = {k: int((self.label == v).sum()) for k, v in LABELS.items()}
        return out


def from_kloppy(dataset, name: str, with_labels: bool = True) -> Game:
    """Convert a kloppy TrackingDataset (default normalised coordinates) into a Game."""
    from kloppy.domain import Ground

    md = dataset.metadata
    if not getattr(md.coordinate_system, "normalized", False):
        raise ValueError("{}: expected kloppy's normalised coordinates, got {}; load without a "
                         "'coordinates' argument".format(name, type(md.coordinate_system).__name__))
    length = float(getattr(md.pitch_dimensions, "pitch_length", None) or 105.0)
    width = float(getattr(md.pitch_dimensions, "pitch_width", None) or 68.0)
    df = dataset.to_df()

    players, team = [], []
    home_id = away_id = None
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
        ball[:, 2] = np.asarray(df["ball_z"].astype("float64").to_numpy())

    label = np.full(T, -1, dtype=np.int8)
    if with_labels and "ball_state" in df.columns and "ball_owning_team_id" in df.columns:
        state = df["ball_state"].map(lambda s: getattr(s, "value", s)).astype(object)
        owner = df["ball_owning_team_id"].astype(object)
        dead = (state == "dead").to_numpy()
        alive = (state == "alive").to_numpy()
        label[alive & (owner == home_id).to_numpy()] = LABELS["H"]
        label[alive & (owner == away_id).to_numpy()] = LABELS["A"]
        label[dead] = LABELS["D"]

    return Game(name=name, fps=float(md.frame_rate), pitch=np.array([length, width], np.float32),
                frame_id=df["frame_id"].to_numpy(dtype=np.int64),
                period=df["period_id"].to_numpy(dtype=np.int8), pos=pos,
                team=np.array(team, dtype=np.int8), ball=ball, label=label)
