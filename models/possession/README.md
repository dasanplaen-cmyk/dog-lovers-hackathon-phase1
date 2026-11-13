# Possession model

Owner: Chanti. Per-frame P(H), P(A), P(D) and P(change) for the decoder (`decode/`, Ying), written
in the agreed probabilities format (`inference/README.md`). Trained and selected on external games
with per-frame labels (DFL); BAR-ATH is only inference input.

- **Inputs per frame:** tracked players (position, velocity, team, and position relative to the
  ball reference: the visible ball or its interpolation) and the ball features of the ball model
  (visible ball, interpolation, frames since and until the nearest visible ball).
- **Direction:** every game is put in one direction, home attacking +x, with a sign per half read
  from the goalkeepers: the home goalkeeper's side is the side home defends (both goalkeepers
  are checked to sit on opposite sides; the training and inference logs print it).
- **Model:** set encoder per frame + temporal transformer over an 8 s window; heads for H/A/D and
  for a state change at that frame.
- **Training:** random ball gaps up to 20 s plus the ball hidden over most dead-ball runs (trackers
  lose it there; DFL never does), pitch flips, home/away swap (x mirrored, H/A swapped), jitter
  and dropped tracks.
- **Selection:** whole validation games with fixed simulated gaps, scored with
  `evaluation/possession_metrics.py` (spellF1, macroF1) after a fixed 3-state Viterbi. That Viterbi
  is only a yardstick for choosing checkpoints; the submission decoder is Ying's.

```bash
.venv/bin/python -m models.possession.train models/possession/configs/v1.yaml   # in tmux
.venv/bin/python -m models.possession.train --resume latest                     # after a reboot
.venv/bin/python -m models.possession.infer --weights data/work/weights/<id>/best.pt --game DFL-J03WQQ --mask-seed 1
.venv/bin/python -m models.possession.infer --weights data/work/weights/<id>/best.pt --game BAR-ATH --device cpu
```

Outputs go to `data/work/probs/` (gitignored, denied to agents); commands print aggregates only.
