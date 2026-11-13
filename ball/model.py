"""Ball from players: a set encoder per frame and a temporal transformer over the window.

Per frame, the tracked players (position, velocity, team) are encoded as a set with a CLS token
(always present, so an empty frame is still well defined). The CLS output, the visible ball (if
any) and a position embedding form one token per frame; a temporal transformer reads the whole
window and predicts, for every frame, the ball position and an isotropic log-variance.

Trained with a Gaussian negative log-likelihood, so sigma is a real uncertainty and
`confidence(sigma)` = P(error < radius) under the model, which is checked against the observed
rate on held-out external games.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ball.batching import N_BALL, N_FEAT, XY_SCALE

LOGVAR_MIN, LOGVAR_MAX = -4.0, 9.0   # sigma from about 0.14 m to 90 m


class BallFromPlayers(nn.Module):
    def __init__(self, d_model=128, heads=4, set_layers=2, time_layers=4, window=125,
                 dropout=0.1):
        super().__init__()
        self.player_in = nn.Linear(N_FEAT, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer(), set_layers, enable_nested_tensor=False)
        self.ball_in = nn.Linear(N_BALL, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.time_encoder = nn.TransformerEncoder(layer(), time_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)
        nn.init.zeros_(self.head.weight)   # start exactly at the interpolation, sigma = 1 m
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        self.register_buffer("xy_scale", torch.tensor(XY_SCALE), persistent=False)

    def forward(self, players, pmask, ball_in, fmask):
        """players [B,W,K,F], pmask [B,W,K] (True = pad), ball_in [B,W,N_BALL], fmask [B,W].
        Returns mu [B,W,2] in metres (interpolation + learned correction) and logvar [B,W]."""
        B, W, K, _ = players.shape
        x = self.player_in(players.reshape(B * W, K, -1))
        x = torch.cat([self.cls.expand(B * W, 1, -1), x], dim=1)
        mask = torch.cat([torch.zeros(B * W, 1, dtype=torch.bool, device=pmask.device),
                          pmask.reshape(B * W, K)], dim=1)
        frame = self.set_encoder(x, src_key_padding_mask=mask)[:, 0].reshape(B, W, -1)
        h = frame + self.ball_in(ball_in) + self.time_pos[:, :W]
        h = self.time_encoder(h, src_key_padding_mask=fmask)
        out = self.head(self.norm(h))
        base = ball_in[..., 3:5] * ball_in[..., 5:6]      # interpolation where any ball is visible
        mu = (base + out[..., 0:2]) * self.xy_scale
        logvar = out[..., 2].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar


def gaussian_nll(mu, logvar, target, weight):
    """Isotropic 2-D Gaussian NLL per frame, averaged with weights; NaN targets are ignored."""
    known = ~torch.isnan(target[..., 0])
    w = weight * known
    t = torch.nan_to_num(target)
    sq = ((mu - t) ** 2).sum(-1)
    nll = logvar + 0.5 * sq * torch.exp(-logvar)   # 2-D: log(sigma^2) + |e|^2 / (2 sigma^2)
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def confidence(logvar, radius_m: float = 2.0):
    """P(|error| < radius) for an isotropic 2-D Gaussian (Rayleigh CDF)."""
    var = torch.exp(logvar) if torch.is_tensor(logvar) else math.exp(logvar)
    if torch.is_tensor(var):
        return 1.0 - torch.exp(-(radius_m ** 2) / (2.0 * var))
    return 1.0 - math.exp(-(radius_m ** 2) / (2.0 * var))
