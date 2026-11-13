# Technical document: Dog Lovers (More than a Hack 2027)

<!--
Full version. The submitted Dog_Lovers.pdf is built from docs/technical_document_short.md.
Owner: Ying (@yingtorres, role G). Keep both in step.
Rule: no match data in this document (no tracking rows, frame exports, CSV excerpts, video frames).
Only code names, file names and aggregate numbers.
-->

**Team:** Dog Lovers
**Members:** Santiago Plaza (team lead; compute and models: ball and vision, sequence model, training and inference), Juan Granda (contributor: ball and vision support, with Santiago Plaza), Celia Barriocanal (contributor), Ying Torres (features, rules baseline, decoding, evaluation, pipeline, external data, document and submission)
**Repository:** https://github.com/dasanplaen-cmyk/dog-lovers-hackathon-phase1
**Submission files:** `Dog_Lovers.csv` and `Dog_Lovers.pdf`

---

## 1. Summary

We label every frame of the provided match as home possession, away possession, dead ball or outside play. A ball model fills in the ball when the tracker loses it. A sequence model over all players gives per-frame probabilities, and a Viterbi decoder turns them into clean possession spells. Everything was trained and tuned on open external data only (DFL/Sportec and SkillCorner); the provided match was never used for tuning. Training hides the ball with gaps drawn from real tracker behaviour, and the labels follow the challenge's controlled-possession definition. On 18 held-out SkillCorner games, scored once at the end, the final system reaches spellF1 0.502 (90 % interval 0.487 to 0.515) and macroF1 0.837.

## 2. Approach

What we built, and why this rather than something else.

### 2.1 Overall pipeline

1. Load the tactical tracking (ATD) into one per-game file.
2. Estimate the ball where tracking lost it: observed ball, interpolation on short gaps, "ball from players" model elsewhere, each with a confidence value. The result goes into the probabilities file. Possession model v1 does not use it: it sees the observed ball and its interpolation. v2 and v3 (the submission) take the ball model's estimate as an input.
3. Sequence model: per-frame probabilities for H, A and D, and for a change of state.
4. Decoder: a Viterbi path with costs for changing state, then removal of very short runs. Frames outside the halves are X.
5. One command produces and validates the final CSV.

A rule-based solution uses the same format and decoder as a fallback and as a reference for the model.

### 2.2 Ball and vision (T1, Santiago Plaza with Juan)

The tracking loses the ball for long stretches. On real tracker data (SkillCorner's open broadcast tracking), 80 % of the missing ball frames sit in gaps longer than 10 s. We estimate the ball everywhere it is missing, with a confidence, from the players around it.

**Video detector.** Not used. The plan allowed at most two days for a pretrained detector run locally on the tactical video (never trained or tuned on it). We stopped before starting it, because the tracking-only model already beats interpolation on hidden-ball frames, and nothing could be checked against the video anyway.

**Model** (`ball/`). It looks at a 5-second window (125 frames):
- **Per frame:** a transformer over the set of tracked players (position, velocity, team), with a CLS token. The CLS output is the frame token.
- **Ball inputs per frame:** the visible ball, its linear interpolation inside the window, and the number of frames since and until the nearest visible ball.
- **Across frames:** a temporal transformer over the window.
- **Output:** a correction to the linear interpolation, plus an isotropic Gaussian variance. The head starts at zero, so the untrained model *is* linear interpolation.
- **Loss and confidence:** trained with a Gaussian negative log-likelihood, so the confidence `ball_conf` = P(error < 2 m) under the predicted variance.
- **Size:** 1.2 M parameters.

**Training data.** 5 DFL/Sportec matches (Bassek et al., 2025; CC BY 4.0), where the ball is tracked on every frame:
- **Gaps:** the ball is hidden on simulated gaps, log-uniform 1–250 frames, covering 20–60 % of each window, plus 10 % fully blind windows.
- **Augmentation:** pitch flips, home/away swap, position jitter and dropped tracks.
- **Selection:** on 2 other DFL matches. A test report on Metrica Game 3 was withdrawn when Metrica was dropped (section 4); it played no part in training or selection.

| Mean error on hidden-ball frames, DFL selection games (2 matches) | model | linear interpolation |
|---|---|---|
| All hidden-ball frames | 3.63 m | 4.45 m |
| Long gaps | 4.42 m | 5.64 m |
| Mid-length gaps | 1.31 m | 1.73 m |

Confidence is calibrated: P(error < 2 m) predicted minus observed is −0.04 on validation.

**Fusion** (`ball/fuse.py`, `ball/configs/fusion_v1.json`). Each frame takes one of three sources:
- **Observed:** where the tracker sees the ball, its position is kept.
- **Interpolated:** on gaps bounded by observed balls and at most 30 frames long, linear interpolation, with a confidence from an empirical table by gap length.
- **Model:** everywhere else.

The 30-frame threshold and the table were fitted on the DFL validation matches. Fused mean error: 3.62 m on validation.

### 2.3 Sequence model (T2)

**Architecture** (`models/possession/`). Same backbone as the ball model, over an 8-second window (200 frames).
- **Player tokens** also carry the offset and distance to the ball reference: the visible ball, or its interpolation.
- **Outputs per frame:** P(H), P(A), P(D) and P(state change at this frame). They are written in the probabilities format agreed with the decoder (`inference/README.md`).
- **Direction:** every game is put in one attacking direction (home towards +x). The sign of each half comes from the goalkeepers: the most extreme well-tracked slot of each team by mean x, checked to sit on opposite sides. A first rule based on kick-off positions gave wrong signs on DFL and was replaced. On the provided game, inference refuses to run if this check is ambiguous in any half.

**v1.** Trained on the per-frame labels of 5 DFL matches (provider BallStatus and BallPossession), with simulated ball gaps (random gaps up to 20 s, plus most dead-ball runs hidden). Selected on the 2 other DFL matches.

| v1 | spellF1 | macroF1 |
|---|---|---|
| DFL validation, fixed Viterbi yardstick, raw provider labels | 0.652 | 0.921 |
| SC-1874553 (real gaps, SkillCorner labels), fixed Viterbi yardstick, labels aligned | 0.432 | 0.805 |

The decoder, tuned for v1 on DFL validation + SC-1874553 with labels aligned to the challenge's controlled-possession definition, scores 0.746, 0.749 and 0.558 on those tuning games (mean 0.684, optimistic).

**v2 and v3** (external data only). The submission uses **`possession_v3`**.
- **`v2_dfl`:** v1 plus the frozen ball model's estimate and confidence on every frame, and each player's offset to that estimate. It keeps v1's synthetic gaps, and a mask that hides the ball on 30 % of dead-ball runs, chosen from the labels. Selected on DFL validation. On SC-1874553 (fixed Viterbi yardstick): spellF1 0.480 and macroF1 0.846, against 0.432 and 0.805 for v1.
- **`v2_sc`:** as `v2_dfl`, but in 80 % of training windows the ball of DFL games is hidden with an alternating renewal process over the visible and missing run lengths of SkillCorner 1874553's tracking (no labels used; about 40 % of frames hidden). The other windows use v1's synthetic gaps. No mask depends on the labels. Selected on DFL validation and SC-1874553.
- **`v3`:** `v2_sc` trained on labels aligned to controlled possession (`align_control`, the same fixed 2 s used for scoring).
- **Comparison:** each candidate, with its own decoder tuned on DFL-J03WQQ, DFL-J03WR9 and SC-1874553, was scored once on SC-1886347, where nothing is tuned.

| SC-1886347 (comparison game, labels aligned) | spellF1 | macroF1 |
|---|---|---|
| `v2_dfl` | 0.505 | 0.847 |
| `v2_sc` | 0.508 | 0.843 |
| `v3` | 0.507 | 0.845 |

All three are within 0.01. The rule agreed before the comparison was: within 0.01 of the best, prefer `v3`, then `v2_sc`, because they use no label-driven masks and match the challenge's definition. So `v3` was chosen. Its decoder settings are in `decode/params.json` (mean spellF1 0.700 on the tuning games, optimistic).
- A first v2 configuration (`v2.yaml`) took its gap profile and selection from Metrica 1; it was dropped with Metrica.

**Ensemble.** None.

### 2.4 Rules, dead-ball detector and decoder (Ying Torres)

Decoder (`decode/decoder.py`), tuned separately for each model: Viterbi over H, A and D, with separate costs for H/A changes, live to dead and dead to live, lowered by the model's change probability, then removal of short runs. Frames outside the in-play blocks are X. Parameters are in `decode/params.json`, chosen by coordinate search on spellF1 on DFL-J03WQQ, DFL-J03WR9 and SC-1874553 only (`eval/external.py`, which refuses BAR-ATH, Metrica and the held-out games, and never tunes on the comparison game).

Rule-based solution (`baseline/rules.py`): a fallback that needs no trained model, and a reference for the model. Possession is the team of the nearest player while a player is within 1.5 m of the ball, carried unchanged otherwise, because a loose ball stays with the last team that had it. Dead ball is a logistic score over a few cues: player speed, ball off the pitch, ball standing still with nobody near, length of the ball gap, and ball missing. Only the cue weights are fitted, on DFL-J03WQQ, DFL-J03WR9 and SC-1874553. The 1.5 m and the confidence values were fixed by hand, not searched, when the baseline was written, at a time when METRICA-1 was one of its tuning games. DFL never loses the ball, so it is hidden with gaps drawn from SkillCorner 1874553's ball-visibility profile (`baseline/gap_profile.json`), which does not depend on the labels. The output uses the same probabilities format and decoder, with its own settings (`baseline/decoder_params.json`). On SC-1874553, a tuning game for both where they see the same real gaps and labels, it scores 0.424 against 0.558 for model v1. The DFL scores are not comparable: v1's DFL probabilities hide the ball on dead-ball runs chosen from the labels, which makes D easier.

### 2.5 Alternatives we considered and why we chose this

- **Video ball detector:** not built. The tracking-only model already beats interpolation (3.63 m against 4.45 m on hidden-ball frames), and nothing could be checked against the video.
- **Rules only:** kept as the fallback. On SC-1874553 they score 0.424, against 0.558 for model v1.
- **Frame-by-frame answers without a decoder:** spellF1 0.35 to 0.58 on the tuning games, against 0.56 to 0.75 with the decoder. spellF1 punishes every spurious change.
- **Interpolation alone for the ball:** used only on gaps of at most 30 frames, a limit chosen on the DFL selection games. On the same hidden-ball frames, the model lowers the mean error from 4.45 m to 3.63 m. On fully blind windows interpolation gives nothing.
- **Ensemble:** none; the three candidates were within 0.01 on the comparison game.

## 3. Assets used

Which of the provided assets we used, how, and which we tried and dropped.

| Asset | Used? | How | Notes |
|---|---|---|---|
| Tactical video | No | | Only the frame numbering of the submission |
| Tactical tracking (ATD) | Yes | Input to the ball model, the possession model and the rules | Automatic tracking; never corrected by hand |
| Broadcast video and tracking | No | | not frame-aligned to tactical |
| Smart Tagging | No | | algorithmic phases, not possessions |
| Homography | No | | |
| Public 5-minute ground truth | Sanity check only | Automatic alignment and file format checks | Never used to train, fit, calibrate, tune, select models or features |
| Pre-trained models | No | | Our models are trained from scratch on external data |
| LLMs / coding assistants | Yes | Coding help | No rows, frame-level values or video were shared. Aggregate statistics of the provided game did reach a hosted assistant and were written in our GitHub issues; we reported this to the organisers on 2026-11-04. Their answer (2026-11-05) restated the rules without a ruling. After that report, no number computed from the provided game reached any assistant: only validator pass/fail and file hashes. The public repository is a clean copy without those issues |

## 4. Training and validation data

Every labelled dataset we used, its source, who labelled it and how much. **All training and tuning used external data only; nothing was trained, fitted or calibrated on the provided game or its video.**

| Dataset | Source and licence | Games / duration | Labels: how and by whom | Used for |
|---|---|---|---|---|
| DFL / Sportec open data (Bassek et al., 2025) | figshare, CC BY 4.0 | 7 matches, 2022/23 | Per-frame possession and ball status from the provider (DFL) | Training (5 matches) and model selection (J03WQQ, J03WR9) of the ball and possession models; decoder tuning and rules fit on J03WQQ and J03WR9 |
| SkillCorner Open Data (SkillCorner, 2025) | GitHub, MIT | 20 matches, A-League 2024/25, 10 fps broadcast tracking resampled to 25 fps | Possession group per frame from the provider; dead ball derived by our code (`skillcorner_labels`) | 1874553: ball-gap profile (tracking only), decoder tuning, rules fit, model selection. 1886347: model comparison only. Other 18: held-out final test |
| Metrica Sports sample data | GitHub, no licence file (README asks to acknowledge the source) | 3 matches | Derived from provider events by our code | **Dropped on 2026-11-09**: the organisers did not confirm we may use it. Earlier, in development: decoder tuning, rules fit, selection of a first v2 configuration and a ball-gap profile (Metrica 1), and a ball-model test report (Metrica 3). Nothing in the submission was fitted, tuned or selected on it |

### 4.1 Hand labels on other games

None. We made no hand labels on any game. DFL labels are the provider's. SkillCorner's H and A come from the provider, and its dead ball from our code (`skillcorner_labels` in `external_data/export.py`). While Metrica was in use, its labels came from its events through our code (`metrica_labels`).

### 4.2 Validation protocol

All tuning and scoring runs on external games with `eval/external.py`, which uses the organisers' own scorer (`evaluation/possession_metrics.py`) and refuses the provided game.

- **Split:** DFL-J03WQQ, DFL-J03WR9 and SC-1874553 for decoder tuning and model selection. SC-1886347 compares candidate models, each scored once; nothing is tuned on it. The other 18 SkillCorner games stay held out until the end (`--final`). DFL-J03WQQ and DFL-J03WR9 also selected the model, so their scores are optimistic.
- **Labels:** provider labels change team on any touch, while the challenge counts controlled possession only. Before scoring, a change of team in live play lasting under 2 s is removed (`align_control`). The 2 s comes from the written proxy in CHALLENGE.md, is fixed, and was never compared with the public ground truth. Scores on raw labels are reported next to the aligned ones.

- **Provided game:** an early automatic check compared the ball gaps in the provided tracking with Metrica 3 (aggregates only, no ground truth, no video). It partly prompted the idea of simulating long tracker gaps in training. No parameter came from it: gap values come from SkillCorner 1874553 tracking (rules, `v2_sc`) or are synthetic (v1, `v2_dfl`).

### 4.3 Results on the held-out games

Final test, run once with `python -m eval.external score <18 games> --final --bootstrap 1000` on the 18 held-out SkillCorner games, with `possession_v3` and its decoder settings from `decode/params.json`, fixed before the run. Probabilities were computed on GPU in float32 (release `external-probs-v3-final`); the submission itself runs on CPU in float64. Labels aligned with `align_control` unless stated.

| 18 held-out SkillCorner games | spellF1 | macroF1 |
|---|---|---|
| `possession_v3` + tuned decoder (mean over games) | **0.502** | **0.837** |
| 90 % interval, bootstrap over games | 0.487 to 0.515 | 0.829 to 0.846 |
| Range over games | 0.409 to 0.563 | 0.783 to 0.875 |
| Same probabilities, argmax without the decoder | 0.411 | |
| Same output, scored on raw provider labels | 0.429 | |

- Per game, the scorer's own 1000-sample bootstrap gives intervals about ±0.045 wide around each spellF1.
- On average the output has about 7 spells fewer than the labels, out of several hundred per game.
- The decoder adds 0.06 to 0.13 spellF1 over the argmax, and helps on every game.
- The score is close to the comparison game (0.507 on SC-1886347), so the model choice did not overfit that game.
- These are broadcast-tracking games with 10 fps data and the ball visible about 60 % of the time; the provided game may differ, so this is an estimate, not a promise.


## 5. Possession rules implemented

Where we drew the line in the ambiguous cases.

- **Loose balls:** possession stays with the last team that had it. The rule-based solution does this directly; the model learns it from the training labels.
- **Dead ball (D):** from the moment play stops (ball over a line, foul, offside, handball, goal) until the restart is taken. Never a possession, so `H H D D H H` is two possessions.
- **Challenges / 50-50 balls:** a touch does not change possession. The other team needs control, which we approximate with at least 2 s on the ball, from the written proxy in CHALLENGE.md. This 2 s is used to align validation labels. The decoder's settings (below) were tuned separately.
- **Restarts (throw-ins, free kicks, corners, goal kicks, kick-off):** dead until the restart is taken, then the restarting team has a clear possession. The decoder's change costs make very short possessions unlikely, so a very short one after a restart can be lost.
- **No answer (X):** only outside the halves, or where the model gave no prediction. Dead ball inside a half is D, not X.
- **Minimum possession length and smoothing:** the decoder charges a cost for every change and can remove short runs. Settings in `decode/params.json` for `possession_v3`: possessions of at least 1 s and dead-ball runs of at least 4 s; the change costs do the rest of the smoothing.

## 6. Limitations

Where it fails, and what we would fix with more time.

- **Our validation labels are not the challenge's.** DFL labels come from the provider; SkillCorner's possession group comes from the provider and its dead ball from our reading of it. Neither was produced the way the challenge's ground truth was, so scores on other games only estimate the real one.
- **The 2 s control rule is a proxy.** It removes short real possessions along with touches, so aligned scores are not directly comparable with raw ones. We report both.
- **Decoder tuning helps on real gaps, by an uncertain amount.** On DFL it adds about 0.02 to 0.03 spellF1. On SC-1874553 it lifts v1 from 0.43 to 0.56, but that game was also used to tune, so part of the gain is fitted to it. The best settings change with the tuning games. Better probabilities still matter most.
- **The ball is often missing.** Long tracking gaps are the hardest part: nothing tells the model where the ball is, and no video detector is used.
- **Tuning games are not clean.** DFL-J03WQQ and DFL-J03WR9 also selected the model, and with v1 their hidden-ball masks depend on the labels, so only the 18 held-out SkillCorner games give an unbiased score.
- **Few independent games.** Seven DFL games from one German league and 20 SkillCorner games from one Australian league are a small sample of how teams play.
- **SkillCorner is not the challenge's tracking.** Broadcast camera, about 16 players per frame, 10 fps resampled to 25. Scores there are a harder proxy, not an estimate of the challenge score.
- **Long ball gaps.** Most missing ball frames are in gaps of more than 10 s. There the ball model has only the players to go on, and its confidence is low by design (P(error < 2 m) is usually under 0.1). The possession model then has to work mostly from player positions.
- **The domain gap is narrowed, not closed.** We train on DFL, where the ball is never lost and the tracking is clean, and apply the models to automatic tracking with fragmented tracks and identity switches. Simulated gaps and dropped tracks narrow this gap. We did not degrade the player tracking itself (identity switches, duplicates) to match an automatic tracker.
- **Small, single-league training set.** Five DFL matches from one German league train both models, and selection uses two more DFL matches plus one SkillCorner game. Teams that play very differently may be underrepresented.
- **Label conventions differ.** DFL possession changes on any touch, whereas the challenge counts controlled possession. `v3` is trained on labels aligned with the 2 s proxy, but the proxy is ours, not the challenge's annotation.
- **Training is not bit-reproducible on GPU** (memory-efficient attention backward). Inference and the published weights are reproducible.
- **No video.** A ball detector on the tactical video could fill some long gaps. We did not build it.

## 7. Reproduction

How to regenerate the submission CSV byte for byte.

```bash
# 1. Install pinned dependencies on CPU (Linux x86_64 or aarch64, Python 3.12, uv)
training/cpu/install.sh
# 2. Place the data bundle in data/, or set MTAH_DATA_DIR to it before installing (not included in the repo)
# 3. Run everything: weights, game file, ball, possession, decoder, validator
make submission PYTHON=.venv/bin/python   # = python -m pipeline.run
# Output: data/work/submission/Dog_Lovers.csv
```

The GX10 itself was set up with `training/gx10/bootstrap.sh` (CUDA torch for training), but inference runs on CPU in float64 on both machines. `pipeline/run.py` runs on CPU and decodes twice to check that the file repeats. `make rules` writes the rules fallback (`Dog_Lovers_rules.csv`), and `make selftest` checks the whole chain on synthetic data.

**Training.** One run per config, each logged in `training/runs/<id>/` (config, commit, seed, games, scalar metrics).
- **Commands:** `python -m ball.train ball/configs/v1.yaml` and `python -m models.possession.train models/possession/configs/<v>.yaml`.
- **Seeds:** fixed in each config (ball 20261104, possession v1 20261105, `v2_dfl` 20261106, `v2_sc` 20261109, `v3` 20261110), with deterministic algorithms enabled.
- **GPU caveat:** on GPU, PyTorch's memory-efficient attention backward is not bit-deterministic. A retrain gives near-identical, not identical, weights. The published weights are therefore the reference.

**Weights.** GitHub Releases `weights-v1` (`ball_v1.pt`, `possession_v1.pt`) and `weights-v2` (`possession_v3.pt`, used by the submission), with sha256 in `inference/weights_manifest.json`. `python -m inference.weights fetch` downloads and verifies them.

**Inference.** On CPU for the provided game, with probabilities rounded to 4 decimals. The probabilities file and the submission are byte-identical across runs on the same environment. Across machines, with the pinned versions and inference on CPU in float64, the ball file and the possession probabilities of an external game (SC-1874553) are byte-identical on the GX10 (aarch64) and an x86_64 Ubuntu PC (`training/cpu/repro_check.sh`, 2026-11-09). The final test log on the 18 held-out games is also byte-identical on the GX10 and an x86_64 Mac. The submitted `Dog_Lovers.csv` itself is byte-identical (same sha256, validator OK) on the GX10 and the x86_64 PC (2026-11-12).

**Hardware.** ASUS Ascent GX10 (NVIDIA GB10, aarch64, 128 GB unified memory), Python 3.12, torch 2.14.1+cu130; exact versions in `training/gx10/requirements-gx10.lock`. Backup machine: Ubuntu 22.04 PC (Ryzen 7, RTX 4050, x86_64), CPU versions in `training/cpu/requirements-cpu-x86_64.lock`.

| Step | Time on the GX10 |
|---|---|
| Ball model training (30k steps) | about 2 h |
| Possession v1 training (20k steps) | about 1 h |
| Inference over the provided game, on CPU | a few minutes |

## 8. Compliance checklist (internal, remove before submission)

Based on the organisers' "Clarification of the rules" email (2026-11-03), which overrides all earlier documentation, and their answer of 2026-11-05 ("anything that is not explicitly allowed is not permitted").

**Training and tuning**
- [ ] Every trained or retrained model (ball detector, sequence model, HMM) was trained on external data only.
- [ ] Every tuned parameter (thresholds, HMM fitting, decoder settings, model selection) was tuned on external data only.
- [ ] Nothing was trained, fitted or calibrated on the BAR-ATH ground truth or on the provided video (no fine-tuning, no pseudo-label training on BAR-ATH).

**Provided game (BAR-ATH)**
- [ ] No hand labels, hand corrections or any manually created data on BAR-ATH.
- [ ] No human-produced data about BAR-ATH (Opta, StatsBomb, Wyscout, LaLiga, etc.).
- [ ] Output never checked against the video (watching the video only to understand what counts as possession is allowed).
- [ ] Public 5-minute GT used only for automatic sanity checks (alignment, format), never for training or tuning.
- [ ] Provided data never uploaded to AI services, including our coding agents.

**External data**
- [ ] All external data is about other games, we have the rights to use it (licence checked), it is accessible to organisers and described in section 4.
- [ ] Metrica Sports sample data dropped (no licence): nothing in the submission was fitted, tuned or selected on it; its use in development is declared in section 4.
- [ ] Aggregate statistics shared with a hosted AI assistant and written in GitHub issues: declared in section 3; the public repository is a clean copy without those issues.

**Submission**
- [ ] No match data or secrets in git; `.gitignore` rules intact.
- [ ] One documented command reproduces the CSV byte for byte.
- [ ] CSV validated with `evaluation/validate_submission.py` (the exact file uploaded).
- [ ] Files named `Dog_Lovers.csv` and `Dog_Lovers.pdf`.

## References

Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data, 12*, 195. https://doi.org/10.1038/s41597-025-04505-y

Metrica Sports. (2020). *Metrica Sports sample data* [Data set]. GitHub. https://github.com/metrica-sports/sample-data

SkillCorner. (2025). *SkillCorner open data* [Data set]. GitHub. https://github.com/SkillCorner/opendata

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*.

PySport. (2026). *kloppy* (Version 3.18.0) [Computer software]. GitHub. https://github.com/PySport/kloppy
