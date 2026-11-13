# Dog Lovers: ball possession per frame (More than a Hack 2027, phase 1)

Team **Dog Lovers**: Santiago Plaza, Juan Granda, Celia Barriocanal, Ying Torres.

This repository turns the provided match data into `Dog_Lovers.csv`: one label per video frame,
H (home in possession), A (away), D (dead ball) or X (outside the halves). How it works, which data
we used and how we validated it: [`docs/technical_document.md`](docs/technical_document.md).
The organisers' original description of the challenge is kept in
[`CHALLENGE_README.md`](CHALLENGE_README.md).

## Reproduce the submission

On Linux (x86_64 or aarch64), with [uv](https://docs.astral.sh/uv/) installed:

```bash
# 1. The data bundle is not in this repository (not redistributable).
#    Put it in data/, or point to it:
export MTAH_DATA_DIR=/path/to/the/data/bundle

# 2. Pinned dependencies: Python 3.12, CPU torch 2.14.1, same versions as our reference machine
training/cpu/install.sh

# 3. Weights, game file, ball, possession, decoder (twice), official validator
make submission PYTHON=.venv/bin/python
# -> data/work/submission/Dog_Lovers.csv
```

- The model weights are downloaded from this repository's releases and checked against the sha256
  in `inference/weights_manifest.json`.
- Everything runs on CPU in float64. The decoder runs twice and the run fails if the two files
  differ. The submitted `Dog_Lovers.csv` came out byte-identical (same sha256) on an aarch64
  and an x86_64 machine.
- `make rules` writes a fallback without trained models (`Dog_Lovers_rules.csv`).
- `make selftest` checks the whole chain on synthetic data, no match data needed.

## What is where

| Folder | Content |
|---|---|
| `ball/` | ball model: fills in the ball when the tracker loses it |
| `models/possession/` | sequence model: per-frame probabilities of H, A, D |
| `decode/` | Viterbi decoder and its settings |
| `baseline/` | rule-based fallback |
| `pipeline/` | the one command that makes the submission |
| `eval/`, `external_data/` | tuning and scoring on external games only |
| `training/` | install scripts, training runs and their configs |
| `evaluation/` | the organisers' scorer and validator, unchanged |

## Data and licences

Models were trained and tuned on external open data only, never on the provided match:

- DFL / Sportec open tracking data (Bassek et al., 2025), CC BY 4.0.
- SkillCorner Open Data, MIT.

Details and sources: [`external_data/SOURCES.md`](external_data/SOURCES.md). Our code is under
the MIT licence in [`LICENSE-DOG-LOVERS`](LICENSE-DOG-LOVERS). The organisers'
starter files (scorer, validator, documentation, public 5-minute ground truth) are under the MIT
licence in [`LICENSE`](LICENSE), which does not cover the match data.
