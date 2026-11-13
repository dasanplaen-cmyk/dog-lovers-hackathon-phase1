"""Windows, simulated ball gaps and augmentation for the ball model. numpy only.

A window is W consecutive frames inside one segment of one game. The input keeps the ball where
it was observed except on simulated gaps; the target is the observed ball (DFL: every frame).
Gap lengths are log-uniform between gap_min and gap_max frames, laid down until a per-window
missing fraction drawn from U(miss_lo, miss_hi) is reached; with probability p_blind the whole
window is blind. These are training settings chosen on external games only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Scale of the model's inputs and outputs: half pitch in metres, and a running speed.
XY_SCALE = np.array([52.5, 34.0], dtype=np.float32)
V_SCALE = 10.0
N_FEAT = 8  # player token: x, y, vx, vy, home, away, other, velocity known
N_BALL = 8  # frame ball token: x, y, visible, interp x, interp y, has interp, dt prev, dt next


def ball_features(vis_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-frame ball inputs [W, N_BALL] from the visible ball (metres, NaN = not visible).

    Besides the visible position, every frame gets the linear interpolation of the visible ball
    inside the window (held at the edges) and the distance in frames to the previous and next
    visible ball, normalised by the window. The model predicts a correction to that
    interpolation, so it starts from the baseline it has to beat.
    """
    W = len(vis_xy)
    t = np.arange(W)
    vis = ~np.isnan(vis_xy[:, 0]) & valid
    f = np.zeros((W, N_BALL), np.float32)
    f[:, 6:8] = 1.0
    if vis.any():
        tv = t[vis]
        f[vis, 0:2] = vis_xy[vis] / XY_SCALE
        f[:, 2] = vis
        f[:, 3] = np.interp(t, tv, vis_xy[vis, 0]) / XY_SCALE[0]
        f[:, 4] = np.interp(t, tv, vis_xy[vis, 1]) / XY_SCALE[1]
        f[:, 5] = 1.0
        prev = np.maximum.accumulate(np.where(vis, t, -1))
        nxt = np.minimum.accumulate(np.where(vis, t, 2 * W)[::-1])[::-1]
        f[:, 6] = np.where(prev >= 0, t - prev, W) / W
        f[:, 7] = np.where(nxt < 2 * W, nxt - t, W) / W
    f[~valid] = 0.0
    return f


@dataclass
class Batch:
    players: np.ndarray    # [B, W, K, N_FEAT] float32
    pmask: np.ndarray      # [B, W, K] bool, True = padding
    ball_in: np.ndarray    # [B, W, N_BALL] float32, see ball_features
    fmask: np.ndarray      # [B, W] bool, True = padded frame (window ran past the segment)
    target: np.ndarray     # [B, W, 2] float32 metres, NaN where unknown
    hidden: np.ndarray     # [B, W] bool, True where the ball was hidden from the input
    start: np.ndarray      # [B] row of the window's first frame in its game
    game: np.ndarray       # [B] index into the games list


class GameArrays:
    """A Game plus the derived arrays the sampler needs, computed once."""

    def __init__(self, game, max_players: int):
        self.game = game
        self.vel = game.velocities()
        self.segments = [s for s in game.segments()]
        self.max_players = max_players
        self.team_onehot = np.eye(3, dtype=np.float32)[np.clip(game.team, 0, 2)]  # [P,3]

    def window_starts(self, window: int, stride: int) -> list[int]:
        """Deterministic window starts covering every frame of every segment."""
        starts = []
        for a, b in self.segments:
            last = max(a, b - window)
            s = list(range(a, last + 1, stride))
            if s[-1] != last:
                s.append(last)
            starts.extend(s)
        return starts

    def segment_of(self, row: int) -> tuple[int, int]:
        for a, b in self.segments:
            if a <= row < b:
                return a, b
        raise IndexError(row)

    def features(self, start: int, window: int):
        """Player tokens for frames [start, start+window) clipped to the segment, padded to W."""
        a, b = self.segment_of(start)
        end = min(start + window, b)
        n = end - start
        pos = self.game.pos[start:end]                      # [n,P,2]
        vel = self.vel[start:end]
        present = ~np.isnan(pos[..., 0])                    # [n,P]
        # Keep at most K tracked objects per frame, present ones first, stable order.
        order = np.argsort(~present, axis=1, kind="stable")[:, :self.max_players]
        take = lambda arr: np.take_along_axis(arr, order[..., None], axis=1)  # noqa: E731
        p = take(pos)
        v = take(vel)
        pres = np.take_along_axis(present, order, axis=1)
        team = self.team_onehot[order]                       # [n,K,3]
        vknown = ~np.isnan(v[..., 0])
        feats = np.zeros((window, self.max_players, N_FEAT), np.float32)
        feats[:n, :order.shape[1], 0:2] = np.nan_to_num(p / XY_SCALE)
        feats[:n, :order.shape[1], 2:4] = np.clip(np.nan_to_num(v / V_SCALE), -2, 2)
        feats[:n, :order.shape[1], 4:7] = team
        feats[:n, :order.shape[1], 7] = vknown
        pmask = np.ones((window, self.max_players), bool)
        pmask[:n, :order.shape[1]] = ~pres
        fmask = np.ones(window, bool)
        fmask[:n] = False
        ball = np.full((window, 2), np.nan, np.float32)
        ball[:n] = self.game.ball[start:end, :2]
        return feats, pmask, fmask, ball


def simulate_gaps(observed: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Boolean [W] mask of frames to hide, on top of what is already missing."""
    w = len(observed)
    hide = np.zeros(w, bool)
    if rng.random() < cfg.get("p_blind", 0.1):
        hide[:] = True
        return hide
    target = rng.uniform(cfg.get("miss_lo", 0.2), cfg.get("miss_hi", 0.6))
    lo, hi = np.log(cfg.get("gap_min", 1)), np.log(cfg.get("gap_max", 250))
    for _ in range(64):
        if hide.mean() >= target:
            break
        length = int(round(np.exp(rng.uniform(lo, hi))))
        s = int(rng.integers(0, w))
        hide[s:s + length] = True
    return hide


def augment(feats, pmask, ball, rng: np.random.Generator, cfg: dict):
    """Pitch flips, team swap, position jitter and tracking dropout. In place on copies."""
    feats, pmask, ball = feats.copy(), pmask.copy(), ball.copy()
    if rng.random() < cfg.get("p_flip", 0.5):           # mirror along the length
        feats[..., 0] *= -1
        feats[..., 2] *= -1
        ball[:, 0] *= -1
    if rng.random() < cfg.get("p_flip", 0.5):           # mirror across the width
        feats[..., 1] *= -1
        feats[..., 3] *= -1
        ball[:, 1] *= -1
    if rng.random() < cfg.get("p_swap", 0.5):           # home <-> away
        feats[..., [4, 5]] = feats[..., [5, 4]]
    sigma = cfg.get("jitter_m", 0.1)
    if sigma > 0:
        feats[..., 0:2] += rng.normal(0, sigma, feats[..., 0:2].shape).astype(np.float32) / XY_SCALE
    drop = cfg.get("p_drop_track", 0.05)
    if drop > 0:                                         # a whole track lost for the window
        gone = rng.random(feats.shape[1]) < drop
        pmask[:, gone] = True
    drop = cfg.get("p_drop_frame", 0.02)
    if drop > 0:                                         # single detections lost
        pmask |= rng.random(pmask.shape) < drop
    feats[pmask] = 0.0
    return feats, pmask, ball


def make_batch(games: list[GameArrays], rng: np.random.Generator, cfg: dict, train: bool,
               starts: list[tuple[int, int]] | None = None,
               simulate_hidden: bool = True) -> Batch:
    """Random windows (train) or the given (game, start) pairs (eval, fixed masks via rng).
    simulate_hidden=False gives the model every observed ball (inference)."""
    W, B = cfg["window"], cfg["batch_size"]
    if starts is None:
        weights = np.array([g.game.n_frames for g in games], float)
        gi = rng.choice(len(games), size=B, p=weights / weights.sum())
        starts = []
        for i in gi:
            a, b = games[i].segments[rng.choice(len(games[i].segments),
                                                p=_seg_weights(games[i]))]
            starts.append((int(i), int(rng.integers(a, max(a + 1, b - W + 1)))))
    n = len(starts)
    out = Batch(players=np.zeros((n, W, cfg["max_players"], N_FEAT), np.float32),
                pmask=np.ones((n, W, cfg["max_players"]), bool),
                ball_in=np.zeros((n, W, N_BALL), np.float32), fmask=np.ones((n, W), bool),
                target=np.full((n, W, 2), np.nan, np.float32), hidden=np.zeros((n, W), bool),
                start=np.array([s for _, s in starts]), game=np.array([g for g, _ in starts]))
    for k, (gi, s) in enumerate(starts):
        feats, pmask, fmask, ball = games[gi].features(s, W)
        if train:
            feats, pmask, ball = augment(feats, pmask, ball, rng, cfg)
        observed = ~np.isnan(ball[:, 0]) & ~fmask
        hide = (simulate_gaps(observed, rng, cfg) & observed if simulate_hidden
                else np.zeros_like(observed))
        visible = observed & ~hide
        out.players[k], out.pmask[k], out.fmask[k] = feats, pmask, fmask
        out.ball_in[k] = ball_features(np.where(visible[:, None], ball, np.nan), ~fmask)
        out.target[k] = np.where(observed[:, None], ball, np.nan)
        out.hidden[k] = hide
    return out


def _seg_weights(g: GameArrays) -> np.ndarray:
    lengths = np.array([b - a for a, b in g.segments], float)
    return lengths / lengths.sum()


def interpolate_baseline(batch: Batch) -> np.ndarray:
    """Reference to beat: linear interpolation of the visible ball inside each window, holding
    the nearest visible value at the edges. [B, W, 2] metres, NaN where nothing is visible."""
    has = batch.ball_in[..., 5:6] > 0
    return np.where(has, batch.ball_in[..., 3:5] * XY_SCALE, np.nan).astype(np.float32)
