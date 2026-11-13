# External data sources (other games only)

Owner: Ying (@yingtorres, role G). Links and licences checked on 2026-11-03; Metrica dropped on 2026-11-09.
Only scripts and this list go in git. Downloaded data stays local and out of the repo.

## Used

| # | Dataset | Link | Tracking | Possession labels | Licence | Use |
|---|---|---|---|---|---|---|
| 1 | DFL / Sportec open data (Bassek et al., 2025) | https://doi.org/10.6084/m9.figshare.28196177 | 25 Hz, 22 players + ball (x, y, z), 7 full matches 2022/23 | Per frame: `BallPossession` (home/away) and `BallStatus` (in/out of play), i.e. H / A / D (dead ball) on every frame | CC BY 4.0 (credit DFL + cite paper) | Training (5 matches), model selection and decoder tuning (J03WQQ, J03WR9) |
| 2 | SkillCorner Open Data | https://github.com/SkillCorner/opendata | 10 fps broadcast tracking, ball with `is_detected`, 20 A-League 2024/25 matches | Per frame `possession` group (home/away/none); dead ball derived by our code | MIT (credit SkillCorner) | 1874553: ball-gap profile (tracking only), decoder tuning, rules fit, model selection. 1886347: model comparison only. Other 18: held-out final test |

### Quality check (aggregates only)

- **DFL J03WOH (2026-11-03):** 137,214 ball frames (91.5 min at 25 fps), ball present on every frame, ball in play 52.4 %, about 400 H/A possession spells (mean 7.2 s, 46 shorter than 1 s). Provider labels change team on any touch, while the challenge counts controlled possession only, so `eval/external.py` aligns labels before scoring (`align_control`: a change of team in live play needs at least 2 s on the ball, from the written proxy in CHALLENGE.md; fixed, never tuned, never compared with the public ground truth). Expect about 2,800 spells over 7 matches.

- **SkillCorner, per match (2026-11-05, 2 of 20 matches):** about 100 minutes, ball detected on 60 % of frames, about 16 players per frame (extrapolated off camera; broadcast tracking, not all 22). Possession comes per frame, but no frame is marked as dead ball, so `skillcorner_labels` in `external_data/export.py` derives it: a home or away group gives H or A; no group inside a phase of play is a loose ball kept by the last team of that half (CHALLENGE.md); no group outside every phase of play is D. This reading of the empty group is ours, not documented by SkillCorner. Result: D 43 / 44 %, 347 / 352 possession spells, mean 9.9 / 9.4 s. Harder than the DFL games: other tracker, other competition, 10 fps linearly resampled to 25, ball only where detected. Files are pinned to one commit of the repository (`SKILLCORNER_REV` in `external_data/download.py`). Split (agreed with Chanti on 2026-11-09): match 1874553 was scored while checking the source and is used for tuning and model selection; 1886347 compares candidate models, each scored once, with nothing tuned on it; the other 18 are held out until `score --final`.

Export to the shared per-game format: `python -m external_data.download dfl|skillcorner` then `python -m external_data.export dfl|skillcorner` (see the docstring of `external_data/export.py`).

Loading: `external_data/download.py` fetches the files, because kloppy's own downloaders fail here (bot challenge on figshare, SSL certificates on macOS). `external_data/export.py` reads them from disk with kloppy.

## Pending / do not use without action

- **PFF FC World Cup 2022** (64 matches): no licence declared, "All Rights Reserved" T&C. Ask PFF / Gradient Sports in writing before use.
- **SoccerTrack v2** (CC BY 4.0, gated on Hugging Face): no per-frame ball tracking, possession spells discarded. Possible extra source only.

## Dropped

- **Metrica Sports sample data** (https://github.com/metrica-sports/sample-data, 3 matches, 25 fps EPTS tracking with real ball gaps, events): no licence file; the README invites use and asks to acknowledge the source. We asked the organisers to confirm on 2026-11-04; their answer on 2026-11-05 restated the rules without a ruling, so the team dropped it on 2026-11-09. Before that it was used in development: labels derived from its events (`metrica_labels`, an algorithm), decoder tuning, rules fit and the selection of a first v2 configuration on Metrica 1, Metrica 1's ball-gap profile, and a ball-model test report on Metrica 3. Nothing in the submission was fitted, tuned or selected on it, and `eval/external.py` and `baseline/rules.py` refuse its games. Its download and export code is kept so the record can be checked.

## Rejected

- StatsBomb Open Data: user agreement forbids commercial exploitation and redistribution.
- Alfheim / Tromsø IL: non-commercial research only.
- SoccerNet Tracking / GSR, SoccerTrack v1: video boxes, no pitch ball or possession, unclear data licence.
- Last Row: hand-recreated goals only, not accurate enough.
- Wyscout (Pappalardo et al.): events only, no tracking.
- Mediacoach/LaLiga, Opta/Stats Perform, Second Spectrum: no open tracking samples.

## References (APA)

Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data, 12*, 195. https://doi.org/10.1038/s41597-025-04505-y

Metrica Sports. (2020). *Metrica Sports sample data* [Data set]. GitHub. https://github.com/metrica-sports/sample-data

SkillCorner. (2025). *SkillCorner open data* [Data set]. GitHub. https://github.com/SkillCorner/opendata
