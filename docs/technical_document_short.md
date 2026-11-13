# Technical document: Dog Lovers (More than a Hack 2027)

<!--
Short version of docs/technical_document.md (same facts, fewer details). Keep both in step.
Rule: no match data in this document. Only code names, file names and aggregate numbers.
-->

**Team:** Dog Lovers
**Members:** Santiago Plaza (team lead; ball model, sequence model, training and inference), Juan Granda (ball and vision support), Celia Barriocanal (contributor), Ying Torres (rules baseline, decoder, evaluation, pipeline, external data, document and submission)
**Repository:** https://github.com/dasanplaen-cmyk/dog-lovers-hackathon-phase1
**Full version with every detail:** `docs/technical_document.md` in the repository.

---

## 1. Summary

We label every frame of the provided match as home possession (H), away possession (A), dead ball (D) or outside play (X). A ball model fills in the ball when the tracker loses it. A sequence model over all players gives per-frame probabilities, and a Viterbi decoder turns them into clean possession spells. Everything was trained and tuned on open external data only (DFL/Sportec and SkillCorner); the provided match was never used for training or tuning. On 18 held-out SkillCorner games, scored once at the end, the final system reaches **spellF1 0.502** (90 % interval 0.487 to 0.515) and **macroF1 0.837**.

## 2. Approach

**Pipeline.** (1) Load the tactical tracking. (2) Estimate the ball where tracking lost it. (3) Sequence model: P(H), P(A), P(D) and P(change) per frame. (4) Decoder. (5) One command writes and validates the CSV. A rule-based solution uses the same format and decoder as a fallback.

**Ball model** (`ball/`). The tracking loses the ball for long stretches: on real tracker data, 80 % of the missing frames are in gaps longer than 10 s. A small transformer (1.2 M parameters) looks at all tracked players over 5 s and predicts a correction to linear interpolation plus a confidence. Trained on 5 DFL matches with the ball hidden on simulated gaps; selected on 2 others. On hidden-ball frames of the selection games it lowers the mean error from 4.45 m (interpolation) to 3.63 m, and from 5.64 to 4.42 m on long gaps. Final ball: observed where seen, interpolated on gaps up to 30 frames, the model elsewhere.

**Possession model** (`models/possession/`). Same backbone over an 8 s window. Inputs: players, the observed ball and the ball model's estimate. Every game is turned so the home team attacks towards +x, with the side of each half taken from the goalkeepers. We trained four versions and chose the last:

- **v1:** DFL labels, synthetic ball gaps.
- **v2_dfl:** v1 plus the ball model's estimate.
- **v2_sc:** ball hidden with gaps drawn from real tracker behaviour (SkillCorner 1874553 tracking, no labels), no label-driven masks.
- **v3 (submitted):** v2_sc trained on labels aligned to the challenge's controlled-possession definition (see section 5).

Each candidate got its own decoder settings and was scored once on a comparison game where nothing is tuned:

| SC-1886347, comparison game | spellF1 | macroF1 |
|---|---|---|
| v2_dfl | 0.505 | 0.847 |
| v2_sc | 0.508 | 0.843 |
| v3 | 0.507 | 0.845 |

All three are within 0.01. The rule fixed before the comparison was to prefer v3, then v2_sc, in that case, because they use no label-driven masks and match the challenge's definition.

**Decoder** (`decode/`). Viterbi over H, A and D, with separate costs for team changes, live to dead and dead to live, lowered by the model's change probability, then removal of short runs. Its settings were searched on DFL-J03WQQ, DFL-J03WR9 and SC-1874553 only. spellF1 punishes every spurious change, so the decoder matters: on the held-out games it adds 0.06 to 0.13 spellF1 over taking the most likely label per frame.

**Alternatives dropped.** A video ball detector (not built: nothing could be checked against the video, and the tracking-only model already beats interpolation). Rules only (kept as fallback; 0.424 against 0.558 for v1 on SC-1874553). An ensemble (the candidates were too close to gain from it).

## 3. Assets used

| Asset | Used? | How |
|---|---|---|
| Tactical tracking | Yes | Input to all models and the rules; never corrected by hand |
| Tactical video | No | Only its frame numbering |
| Broadcast video and tracking, Smart Tagging, homography | No | |
| Public 5-minute ground truth | Sanity check only | Automatic format and alignment checks; never used to train, tune or select |
| Pre-trained models | No | Trained from scratch on external data |
| Coding assistants | Yes, coding help | No rows, frames or video were shared. Aggregate statistics of the provided game did reach a hosted assistant and our GitHub issues; we reported this to the organisers (2026-11-04). After that, only validator pass/fail and file hashes. The public repository is a clean copy without those issues |

## 4. Training and validation data

**Nothing was trained, fitted or calibrated on the provided game or its video. We made no hand labels on any game.**

| Dataset | Licence | Size | Labels | Used for |
|---|---|---|---|---|
| DFL / Sportec open data (Bassek et al., 2025) | CC BY 4.0 | 7 matches | Provider's per-frame possession and ball status | Training (5), selection and tuning (2) |
| SkillCorner Open Data (2025) | MIT | 20 matches | Provider's possession group; dead ball derived by our code | 1874553: gap profile, tuning, selection. 1886347: comparison only. Other 18: final test |
| Metrica Sports sample data | No licence file | 3 matches | From provider events, by our code | **Dropped** (2026-11-09) because its licence was not confirmed. Used earlier in development; nothing in the submission was fitted, tuned or selected on it |

**Results on the 18 held-out games** (run once, at the end, with settings fixed before):

| 18 held-out SkillCorner games | spellF1 | macroF1 |
|---|---|---|
| v3 + decoder, mean over games | **0.502** | **0.837** |
| 90 % interval (bootstrap over games) | 0.487 to 0.515 | 0.829 to 0.846 |
| Range over games | 0.409 to 0.563 | 0.783 to 0.875 |
| Without the decoder | 0.411 | |

The result is close to the comparison game (0.507), so the model choice did not overfit it. These games use 10 fps broadcast tracking with the ball visible about 60 % of the time; the provided game may differ, so this is an estimate, not a promise.

## 5. Possession rules implemented

- **Loose balls:** possession stays with the last team that had it.
- **Dead ball (D):** from the moment play stops until the restart is taken. Never a possession, so `H H D D H H` is two possessions.
- **Challenges and 50-50s:** a touch does not change possession; the other team needs control. We approximate control with at least 2 s on the ball, from the written definition in CHALLENGE.md. This fixed 2 s aligns the training labels of v3 and the validation labels; it was never compared with the public ground truth.
- **Restarts:** dead until taken, then the restarting team has possession.
- **Minimum lengths:** possessions of at least 1 s and dead-ball runs of at least 4 s; the change costs do the rest of the smoothing.
- **X:** only outside the halves.

## 6. Limitations

- **Our labels are not the challenge's.** Provider labels and our 2 s proxy only approximate the challenge's annotation, so scores on other games are estimates.
- **The ball is often missing.** In gaps longer than 10 s the ball model has only the players to go on, and possession must come mostly from player positions. A video detector could help; we did not build one.
- **Domain gap.** We train on clean DFL tracking and apply the models to automatic tracking with broken tracks. Simulated gaps and dropped tracks narrow the gap, but do not close it.
- **Few leagues.** 7 German and 20 Australian games are a small sample of how teams play.
- **Training is not bit-reproducible on GPU.** The published weights are the reference; inference is reproducible.

## 7. Reproduction

```bash
export MTAH_DATA_DIR=/path/to/the/data/bundle   # or put it in data/ (not in the repo)
training/cpu/install.sh                          # Python 3.12, pinned, CPU torch
make submission PYTHON=.venv/bin/python          # -> data/work/submission/Dog_Lovers.csv
```

The command downloads the weights from the repository's releases and checks their sha256, runs the ball and possession models on CPU in float64, decodes twice to check that the file repeats, and runs the official validator. The submitted `Dog_Lovers.csv` came out byte-identical (same sha256) on an aarch64 and an x86_64 machine. `make selftest` checks the whole chain without match data. Training commands, seeds and run logs are in `training/` and in the full document.

## References

Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data, 12*, 195. https://doi.org/10.1038/s41597-025-04505-y

SkillCorner. (2025). *SkillCorner open data* [Data set]. GitHub. https://github.com/SkillCorner/opendata

Metrica Sports. (2020). *Metrica Sports sample data* [Data set]. GitHub. https://github.com/metrica-sports/sample-data

Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*.

PySport. (2026). *kloppy* (Version 3.18.0) [Computer software]. GitHub. https://github.com/PySport/kloppy
