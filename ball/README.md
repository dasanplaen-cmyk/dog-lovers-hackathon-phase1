# Ball from players

Owner: Chanti. Estimates the ball position and a confidence on every frame, mainly where the
tracking lost it (41 % of graded BAR-ATH frames). Trained and selected on **external games only**;
BAR-ATH is only ever inference input.

## Model

Per frame, a set encoder (transformer with a CLS token) reads the tracked players: position,
velocity and team. One token per frame (CLS output, the visible ball, the linear interpolation of
the visible ball, frames since and until the nearest visible ball) goes through a temporal
transformer over a 5 s window. The head predicts a correction to the interpolation and an
isotropic variance; the head starts at zero, so an untrained model *is* linear interpolation.
Gaussian NLL loss, so `ball_conf` = P(error < 2 m) under the model, and its calibration is
reported (`calib_gap`).

Training hides the ball on simulated gaps (log-uniform 1–250 frames, 20–60 % of each window,
10 % fully blind windows) and augments with pitch flips, home/away swap, jitter and dropped
tracks. Validation reports the model next to linear interpolation on the same frames
(`err_mean_m_where_interp` against `interp_err_mean_m`) and by gap length (short ≤ 10, mid
11–50, long > 50 frames). The model is only worth using where it beats interpolation.

## Commands (GX10, from the repo root, inside `tmux`)

```bash
.venv/bin/python -m ball.sources dfl                 # 7 DFL matches, downloaded through kloppy
.venv/bin/python -m external_data.download skillcorner 1874553 && .venv/bin/python -m external_data.export skillcorner 1874553   # real-gap external game (MIT)
.venv/bin/python -m ball.train ball/configs/v1.yaml  # logs to training/runs/<id>/
.venv/bin/python -m ball.train --resume latest       # after a crash or reboot
.venv/bin/python -m ball.fuse fit --weights data/work/weights/<id>/best.pt   # -> ball/configs/fusion_v1.json
.venv/bin/python -m ball.sources bar-ath             # converts the provided ATD (no labels)
.venv/bin/python -m ball.infer --weights data/work/weights/<id>/best.pt --game BAR-ATH --device cpu \
    --fusion ball/configs/fusion_v1.json
```

## Fusion

`ball/fuse.py` combines three sources per frame: the observed ball (`ball_src` 1), linear
interpolation on gaps bounded by observed balls on both sides and at most `max_interp_gap` frames
long (`ball_src` 4, confidence from an empirical table by gap length), and the model everywhere
else (`ball_src` 2). The threshold and the table are fitted on the checkpoint's validation games
(external only) and stored in `ball/configs/fusion_v1.json`, which is committed together with the
sha256 of the weights it belongs to; inference refuses a fusion file fitted for other weights.

Every command prints aggregates only. Games are stored as `.npz` under `data/work/games/`
(`external/` and `test/`), ball files under `data/work/ball/`; all of it is gitignored and denied
to coding agents. `--device cpu` makes the BAR-ATH ball file byte-for-byte reproducible.

`ball/game.py` defines the per-game arrays (`Game`). It is this module's own adapter until
Ying's `external_data/` scripts land; then `sources.py` reads those instead.
