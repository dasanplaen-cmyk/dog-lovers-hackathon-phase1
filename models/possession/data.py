"""Inputs and targets for the possession model. numpy only.

Per window of W frames:
  players  [W, K, 12]  x, y, vx, vy, home, away, other, velocity known (as the ball model), plus
                       dx, dy to the ball reference (scaled), distance to it (/20 m, clipped) and
                       whether a reference exists. The reference is the visible ball, or the
                       linear interpolation of the visible ball inside the window.
  ball_in  [W, 8]      ball.batching.ball_features of the visible ball
  label    [W]         0 H, 1 A, 2 D, -1 ignore
  change   [W]         1 if label[t] != label[t-1], 0 if equal, -1 where undefined

Every game is put in one attacking direction first: x is multiplied by a per-frame sign so that
the home team attacks towards +x. The sign per half comes from the goalkeepers' mean positions
(see attack_sign), an automatic rule on the tracking alone.
"""

from __future__ import annotations

import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball.batching import GameArrays, XY_SCALE, augment as ball_augment, ball_features, simulate_gaps

N_FEAT = 12
N_BALL = 8


def half_starts(game) -> list[tuple[int, int]]:
    """[(kick_off_row, end_row)] of each half. External games: periods. BAR-ATH: the in-play
    blocks of games.py (its ATD is a single period)."""
    if game.name == "BAR-ATH":
        import games
        return [(a, b + 1) for _, a, b in games.half_video_blocks("BAR-ATH")]
    out = []
    for p in np.unique(game.period):
        rows = np.flatnonzero(game.period == p)
        out.append((int(rows[0]), int(rows[-1]) + 1))
    return out


def defended_side(pos_x: np.ndarray, min_share: float = 0.1) -> tuple[int, float]:
    """Side (+1/-1 in x) a team defends in a half: the sign of its goalkeeper's mean x, the
    goalkeeper being the slot present on at least min_share of the frames whose mean x is the
    most extreme. Returns (side, that mean x in metres)."""
    present = np.isfinite(pos_x)
    count = present.sum(axis=0)
    ok = count >= max(1, min_share * len(pos_x))
    if not ok.any():
        return 0, 0.0
    mean_x = np.where(ok, np.nansum(np.where(present, pos_x, 0.0), axis=0) / np.maximum(count, 1), 0.0)
    gk = int(np.argmax(np.abs(mean_x) * ok))
    return int(np.sign(mean_x[gk])), float(mean_x[gk])


def attack_sign(game) -> tuple[np.ndarray, list[int]]:
    """Per-frame +1/-1 so that sign * x has the home team attacking +x, and the sign per half.

    Read from the goalkeepers: in each half the home goalkeeper's side is the side home defends.
    When the two goalkeepers do not sit on opposite sides, the one further from the centre
    decides; the per-half check is in `direction_report`."""
    sign = np.ones(game.n_frames, np.float32)
    halves = half_starts(game)
    per_half = []
    for i, (a, b) in enumerate(halves):
        x = game.pos[a:b, :, 0]
        home_side, home_x = defended_side(x[:, game.team == 0])
        away_side, away_x = defended_side(x[:, game.team == 1])
        if home_side != 0 and home_side == -away_side:
            side = home_side
        else:
            side = home_side if abs(home_x) >= abs(away_x) else -away_side
        s = -side if side != 0 else 1        # defending -x means attacking +x
        per_half.append(int(s))
        lo = 0 if i == 0 else a
        hi = halves[i + 1][0] if i + 1 < len(halves) else game.n_frames
        sign[lo:hi] = s
    return sign, per_half


def direction_report(game) -> list[dict]:
    """Aggregates per half: goalkeeper mean x of each team and whether they sit on opposite sides."""
    out = []
    for a, b in half_starts(game):
        x = game.pos[a:b, :, 0]
        hs, hx = defended_side(x[:, game.team == 0])
        as_, ax = defended_side(x[:, game.team == 1])
        out.append({"home_gk_x": round(hx, 1), "away_gk_x": round(ax, 1), "opposite": hs == -as_ != 0})
    return out


class PossessionGame(GameArrays):
    """GameArrays plus the direction sign and labels."""

    def __init__(self, game, max_players: int):
        super().__init__(game, max_players)
        self.sign, self.half_signs = attack_sign(game)


def _hide_dead(label: np.ndarray, rng: np.random.Generator, p: float, margin: int) -> np.ndarray:
    """Hide the ball over whole dead-ball runs with probability p each (plus a margin): trackers
    tend to lose the ball during stoppages, and DFL never does."""
    hide = np.zeros(len(label), bool)
    dead = label == 2
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if rng.random() < p:
            hide[max(0, a - margin):b + margin] = True
    return hide


_PROFILE = {}


def hide_mask(observed: np.ndarray, label: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Frames whose ball is hidden from the model. gap_mode "renewal" (v2): with probability
    p_renewal, runs drawn from the real-tracker profile (models/possession/gaps.py), otherwise the
    synthetic gaps of v1; then whole dead-ball runs with probability p_hide_dead."""
    if cfg.get("gap_mode") == "renewal" and rng.random() < cfg.get("p_renewal", 1.0):
        from models.possession.gaps import load_profile, renewal_mask
        path = cfg["gap_profile"]
        if path not in _PROFILE:
            _PROFILE[path] = load_profile(path if os.path.isabs(path) else os.path.join(REPO, path))
        h = renewal_mask(len(observed), rng, _PROFILE[path])
    else:
        h = simulate_gaps(observed, rng, cfg)
    return h | _hide_dead(label, rng, cfg["p_hide_dead"], cfg["hide_dead_margin"])


def window(pg: PossessionGame, start: int, cfg: dict, rng: np.random.Generator | None,
           train: bool, hide: bool):
    """Inputs and targets for one window. hide=False keeps every observed ball (inference on a
    game whose missing ball is real, or already masked)."""
    W = cfg["window"]
    feats, pmask, fmask, ball = pg.features(start, W)
    n = int((~fmask).sum())
    s = np.ones(W, np.float32)
    s[:n] = pg.sign[start:start + n]
    feats[..., 0] *= s[:, None]
    feats[..., 2] *= s[:, None]
    ball[:, 0] *= s
    label = np.full(W, -1, np.int64)
    label[:n] = pg.game.label[start:start + n]

    if train:
        swap = rng.random() < cfg.get("p_swap", 0.5)
        sub = dict(cfg, p_swap=0.0)
        feats, pmask, ball = ball_augment(feats, pmask, ball, rng, dict(sub, p_flip=0.0))
        if rng.random() < cfg.get("p_flip_y", 0.5):
            feats[..., 1] *= -1
            feats[..., 3] *= -1
            ball[:, 1] *= -1
        if swap:   # home <-> away keeps "home attacks +x" only if x is mirrored too
            feats[..., [4, 5]] = feats[..., [5, 4]]
            feats[..., 0] *= -1
            feats[..., 2] *= -1
            ball[:, 0] *= -1
            label = np.where(label == 0, 1, np.where(label == 1, 0, label))

    observed = ~np.isnan(ball[:, 0]) & ~fmask
    if hide:
        observed &= ~hide_mask(observed, label, rng, cfg)
    vis_xy = np.where(observed[:, None], ball, np.nan)
    b_in = ball_features(vis_xy, ~fmask)

    ref = np.where(b_in[:, 5:6] > 0, b_in[:, 3:5], 0.0)            # scaled units
    ref_ok = b_in[:, 5] > 0
    d = (feats[..., 0:2] - ref[:, None, :]) * ref_ok[:, None, None]
    dist = np.linalg.norm(d * XY_SCALE, axis=-1) / 20.0
    extra = np.concatenate([d, np.clip(dist, 0, 3)[..., None],
                            np.broadcast_to(ref_ok[:, None, None], dist.shape + (1,))], axis=-1)
    players = np.concatenate([feats, extra.astype(np.float32)], axis=-1)
    players[pmask] = 0.0

    change = np.full(W, -1, np.int64)
    both = (label[1:] >= 0) & (label[:-1] >= 0)
    change[1:][both] = (label[1:] != label[:-1])[both]
    return players, pmask, b_in, fmask, label, change


def make_batch(games: list[PossessionGame], rng: np.random.Generator, cfg: dict):
    W, B = cfg["window"], cfg["batch_size"]
    weights = np.array([g.game.n_frames for g in games], float)
    out = [np.zeros((B, W, cfg["max_players"], N_FEAT), np.float32),
           np.ones((B, W, cfg["max_players"]), bool), np.zeros((B, W, N_BALL), np.float32),
           np.ones((B, W), bool), np.full((B, W), -1, np.int64), np.full((B, W), -1, np.int64)]
    for k in range(B):
        g = games[rng.choice(len(games), p=weights / weights.sum())]
        lens = np.array([b - a for a, b in g.segments], float)
        a, b = g.segments[rng.choice(len(g.segments), p=lens / lens.sum())]
        start = int(rng.integers(a, max(a + 1, b - W + 1)))
        for arr, val in zip(out, window(g, start, cfg, rng, train=True, hide=True)):
            arr[k] = val
    return out


def masked_copy(game, cfg: dict, seed: int):
    """The game with its ball hidden on a fixed set of simulated gaps (random + dead-ball runs),
    for validating on external games under conditions like a real ATD. Labels are untouched."""
    import copy
    rng = np.random.default_rng(seed)
    g = copy.copy(game)
    ball = game.ball.copy()
    W = cfg["window"]
    for a, b in game.segments():
        if cfg.get("gap_mode") == "renewal":   # one real-like mask over the whole segment
            h = hide_mask(~np.isnan(ball[a:b, 0]), game.label[a:b], rng, dict(cfg, p_renewal=1.0))
            ball[a:b][h] = np.nan
            continue
        for s in range(a, b, W):
            e = min(s + W, b)
            obs = ~np.isnan(ball[s:e, 0])
            h = simulate_gaps(obs, rng, cfg)[: e - s] | _hide_dead(
                game.label[s:e], rng, cfg["p_hide_dead"], cfg["hide_dead_margin"])
            ball[s:e][h] = np.nan
    g.ball = ball
    return g
