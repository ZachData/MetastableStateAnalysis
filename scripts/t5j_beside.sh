#!/usr/bin/env bash
# scripts/t5j_beside.sh — T5 U4's beside stage at 4369 (Blocked 134 (a); design-T5.md "U4's beside stage
# at 4369"): the known answers first (stops on a refusal, exit 1), then per arm T2's counts (i)–(iii), U3's
# split and the onsets over its 49 checkpoints, then the reading. Forward passes only (GPU minutes).
# Resumable: onset skips written steps; counts and split rewrite.
#   METS_DATA=<main>/data scripts/t5j_beside.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/run/media/system/HDD_1TB/mets_data/hf}
export METS_DATA=${METS_DATA:?set METS_DATA to the main tree data dir}
J="python -m trackT_timaeus.t5j"
$J known
for arm in A0 A0p A0pp A1 A2 A3; do
    $J counts --arm "$arm"
    $J split --arm "$arm"
    $J onset --arm "$arm"
done
$J read
echo "beside stage read"
