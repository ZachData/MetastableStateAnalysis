#!/usr/bin/env bash
# scripts/t5u4b_replicates.sh — T5 U4b, first stage: the replicate seeds A0′ (A0p) and A0″ (A0pp) to 4369,
# then the seed-noise gate (design-T5.md "U4", Gate, last row: bar < R_A0 / 2) before A1–A3 start.
# A0″'s continuation to 5970 waits for the continuations' stage (it enters no label; design "Throughput").
# Stops at the first gate that refuses (exit 3). Resumable: rerun it.
#   METS_DATA=<main>/data scripts/t5u4b_replicates.sh   (the configuration is t5h's N_MICRO, COMPILE)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/run/media/system/HDD_1TB/mets_data/hf}
export METS_DATA=${METS_DATA:?set METS_DATA to the main tree data dir}
T="python -m trackT_timaeus.t5h"
for arm in A0p A0pp; do
    $T train --arm "$arm" --until 103
    $T read --arm "$arm"
    $T gate --arm "$arm" --stage start       # first output populated; the loss falls; init SDs; the step-0 band
    $T train --arm "$arm" --until 4369
    $T read --arm "$arm"
done
python -m trackT_timaeus.t5i replicates      # exit 3: not readable, back to the user
echo "replicates read: readable"
