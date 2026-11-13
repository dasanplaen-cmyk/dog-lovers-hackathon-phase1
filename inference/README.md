# Per-frame probabilities: the models → decoder contract

Owner: Chanti (writes it). Reader: Ying's decoder (`decode/`). Code: `inference/probs_format.py`.
Change it only after agreeing on the board; bump `SCHEMA_VERSION` when you do.

## File

One CSV per game, `<game>_probs.csv`, plus a sidecar `<game>_probs.csv.meta.json`.
One row per frame of the game's reference video, `0 … n-1`, all frames present.
For BAR-ATH that is the tactical video: **199,319 rows**, row *i* = submission row *i*.

| column | type | meaning |
|---|---|---|
| `frame` | int | 0 … n-1, contiguous, ascending |
| `p_H` | float | P(home in possession, ball alive) |
| `p_A` | float | P(away in possession, ball alive) |
| `p_D` | float | P(dead ball) |
| `p_change` | float | P(state at *t* ≠ state at *t*−1), states H/A/D (covers H↔A and live↔dead); 0 on frame 0 |
| `ball_x`, `ball_y` | float | ball position, **normalised 0–1 like the ATD**; empty when unknown |
| `ball_conf` | float | 0–1 confidence of that position; 0 when unknown |
| `ball_src` | int | 0 none · 1 ATD observed · 2 ball-from-players model · 3 video detector · 4 fused |

- `p_H + p_A + p_D = 1` within 0.001. All three empty together means "no prediction" (the decoder
  decides what to do, e.g. `X` outside play). `p_change` is empty on exactly those rows.
- Floats are written with exactly 4 decimals, `\n` line endings, no index: the same values give the
  same bytes on any machine, so the file's sha256 is a reproducibility check by itself.
- The sidecar holds the schema version, game, row count, the CSV's sha256, and the model name,
  commit, weights sha256 and seed.

## Where it lives

Under the gitignored and agent-denied `data/` tree: `data/work/probs/<game>_probs.csv`
(`MTAH_WORK_DIR` overrides `data/work`). For BAR-ATH it is frame-level match data: **never commit
it, never paste it into a chat or the board**. Anyone regenerates it with one command and the
published weights. External-game files follow the same format so the decoder can be developed and
evaluated on them.

## Commands (aggregates only)

```bash
python -m inference.probs_format validate data/work/probs/BAR-ATH_probs.csv --game BAR-ATH
python -m inference.probs_format validate data/work/probs/<ext_game>_probs.csv --n-frames <n>
python -m inference.probs_format selftest
```

Python: `from inference.probs_format import write_probs, read_probs`.
