"""The ball model's estimate as an input to the possession model (v2).

Runs the frozen ball model (ball/model.py) on the same window the possession model sees, in
overlapping chunks of the ball model's own window length, and keeps for every frame the estimate
from the chunk it sits most centrally in. Output per frame: estimated x, y (scaled like the other
inputs) and the ball model's confidence.
"""

from __future__ import annotations

import numpy as np
import torch

from ball.batching import XY_SCALE, ball_features
from ball.model import confidence


def load_ball_model(path: str, device):
    from ball.train import model_from_config
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["config"]


@torch.no_grad()
def ball_estimates(ball_model, ball_cfg, players, pmask, ball_in, fmask):
    """[B, W, 3]: x, y (scaled), confidence. players' first 8 features are the ball model's."""
    B, W = players.shape[:2]
    Wb = min(ball_cfg["window"], W)
    starts = list(range(0, W - Wb + 1, max(1, Wb // 2)))
    if starts[-1] != W - Wb:
        starts.append(W - Wb)
    t = torch.arange(W, device=players.device, dtype=torch.float32)
    best = torch.full((W,), float("inf"), device=players.device)
    out = torch.zeros(B, W, 3, device=players.device, dtype=players.dtype)
    scale = torch.tensor(XY_SCALE, device=players.device)
    # The ball model was trained on ball features built over its own window: gaps measured in
    # that window's length and interpolation from balls inside it. Rebuild them per chunk from
    # the visible ball rather than slicing the possession window's features.
    vis = (ball_in[..., 2] > 0).cpu().numpy()
    vis_xy = (ball_in[..., 0:2] * scale).float().cpu().numpy()
    valid = (~fmask).cpu().numpy()
    for s in starts:
        sl = slice(s, s + Wb)
        chunk_in = np.stack([ball_features(np.where(vis[k, sl, None], vis_xy[k, sl], np.nan), valid[k, sl])
                             for k in range(B)])
        chunk_in = torch.as_tensor(chunk_in, device=players.device, dtype=players.dtype)
        mu, logvar = ball_model(players[:, sl, :, :8], pmask[:, sl], chunk_in, fmask[:, sl])
        est = torch.cat([mu / scale, confidence(logvar, ball_cfg["conf_radius_m"])[..., None]], -1)
        est = torch.nan_to_num(est)          # a chunk made only of padding gives NaN
        d = (t[sl] - (s + (Wb - 1) / 2)).abs()
        take = d < best[sl]
        out[:, sl] = torch.where(take[None, :, None], est, out[:, sl])
        best[sl] = torch.where(take, d, best[sl])
    return out * (~fmask)[..., None]
