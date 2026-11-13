"""Possession model: per-frame set encoder over players + temporal transformer over the window.

Same backbone as ball/model.py, with players also described relative to the ball reference.
Head: three logits (H, A, D) and one change logit per frame.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ball.batching import XY_SCALE
from models.possession.data import N_BALL, N_FEAT


class PossessionNet(nn.Module):
    def __init__(self, d_model=128, heads=4, set_layers=2, time_layers=4, window=200, dropout=0.1,
                 ball_est=False):
        super().__init__()
        self.ball_est = ball_est
        extra = 3 if ball_est else 0   # v2: ball model estimate (x, y, conf); per player dx, dy, dist
        layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.player_in = nn.Linear(N_FEAT + extra, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.set_encoder = nn.TransformerEncoder(layer(), set_layers, enable_nested_tensor=False)
        self.ball_in = nn.Linear(N_BALL + extra, d_model)
        self.time_pos = nn.Parameter(torch.zeros(1, window, d_model))
        self.time_encoder = nn.TransformerEncoder(layer(), time_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 4)
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, players, pmask, ball_in, fmask, est=None):
        """Returns state logits [B,W,3] (H, A, D) and change logits [B,W]. est [B,W,3] is the
        ball model's estimate (x, y scaled, confidence), required when built with ball_est."""
        B, W, K, _ = players.shape
        if self.ball_est:
            scale = players.new_tensor(XY_SCALE)
            d = players[..., 0:2] - est[:, :, None, 0:2]
            dist = (torch.linalg.norm(d * scale, dim=-1, keepdim=True) / 20.0).clamp(max=3.0)
            players = torch.cat([players, d, dist], -1) * (~pmask)[..., None]
            ball_in = torch.cat([ball_in, est], -1)
        x = self.player_in(players.reshape(B * W, K, -1))
        x = torch.cat([self.cls.expand(B * W, 1, -1), x], dim=1)
        mask = torch.cat([torch.zeros(B * W, 1, dtype=torch.bool, device=pmask.device),
                          pmask.reshape(B * W, K)], dim=1)
        frame = self.set_encoder(x, src_key_padding_mask=mask)[:, 0].reshape(B, W, -1)
        h = frame + self.ball_in(ball_in) + self.time_pos[:, :W]
        h = self.time_encoder(h, src_key_padding_mask=fmask)
        out = self.head(self.norm(h))
        return out[..., :3], out[..., 3]


def loss_fn(state_logits, change_logits, label, change, change_weight: float):
    ce = F.cross_entropy(state_logits.reshape(-1, 3).float(), label.reshape(-1), ignore_index=-1)
    ok = change >= 0
    bce = F.binary_cross_entropy_with_logits(change_logits.float()[ok], change[ok].float()) \
        if ok.any() else state_logits.sum() * 0
    return ce + change_weight * bce, ce, bce
