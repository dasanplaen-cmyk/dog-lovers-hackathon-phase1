#!/usr/bin/env bash
# Cross-machine reproducibility check on an external game (SkillCorner 1874553, MIT): recompute the
# ball file and possession_v1 probabilities on CPU (float64, the default there) with the published
# weights and compare their sha256 with the GX10 reference. Prints hashes only.
#
#   training/cpu/repro_check.sh                                   # on the GX10: prints the reference
#   REF_BALL=<16 hex> REF_PROBS=<16 hex> training/cpu/repro_check.sh   # elsewhere: compares
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
# GX10 reference (CPU, float64), matched byte for byte by an x86 Ubuntu PC on 2026-11-09.
REF_BALL="${REF_BALL:-eb726c0b4df3bd79}"   # GX10, 2026-11-09
REF_PROBS="${REF_PROBS:-59a044c085794704}"  # GX10, 2026-11-09
$PY -m inference.weights fetch
$PY -m external_data.download skillcorner 1874553 >/dev/null
$PY -m external_data.export skillcorner 1874553 >/dev/null
$PY -m ball.infer --weights "$($PY -m inference.weights path ball_v1)" --game SC-1874553 \
    --fusion ball/configs/fusion_v1.json --device cpu >/dev/null
$PY -m models.possession.infer --weights "$($PY -m inference.weights path possession_v1)" \
    --game SC-1874553 --device cpu >/dev/null
ball=$(sha256sum data/work/ball/SC-1874553_ball.csv | cut -c1-16)
probs=$(sha256sum data/work/probs/SC-1874553_probs.csv | cut -c1-16)
echo "ball  $ball  (GX10 $REF_BALL)  $([ "$ball" = "$REF_BALL" ] && echo SAME || echo DIFFERENT)"
echo "probs $probs  (GX10 $REF_PROBS)  $([ "$probs" = "$REF_PROBS" ] && echo SAME || echo DIFFERENT)"
