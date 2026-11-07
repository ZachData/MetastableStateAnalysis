#!/usr/bin/env bash
# scripts/t5u5.sh — T5 U5, the controls (Blocked 135 (a); design-T5.md "U5's details"): U2's statistics on
# `pythia-410m` seed 0 (the known answer against U2's records; stops on a refusal, exit 1), seeds 1–9 at the
# 18 steps, Mistral GPT-2 small alias x21 (36 checkpoints) and battlestar x49 (35) from checkpoint 0; then
# the gate and the description (no rule). Forward passes only (GPU minutes). Resumable: written steps skip.
#   METS_DATA=<main>/data scripts/t5u5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/run/media/system/HDD_1TB/mets_data/hf}
export METS_DATA=${METS_DATA:?set METS_DATA to the main tree data dir}
OUT="$METS_DATA/trackT/t5u5/2026-11-06"
C="python -m trackT_timaeus.t5c"
$C run --out "$OUT" --unit U5 --trajs pythia-410m
$C u5 --out "$OUT" --seed0-only
$C run --out "$OUT" --unit U5 --trajs $(printf 'pythia-410m-seed%d ' 1 2 3 4 5 6 7 8 9)
$C run --out "$OUT" --unit U5 --trajs mistral-alias mistral-battlestar
$C u5 --out "$OUT"
echo "U5 read"
