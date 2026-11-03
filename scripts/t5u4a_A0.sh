#!/usr/bin/env bash
# scripts/t5u4a_A0.sh — T5 U4a: A0 through its positive control (design-T5.md "U4", gate rows 6-7;
# "U4a's details" (t6), (t7)). Stops at the first gate that refuses (exit 3). Resumable: rerun it.
#   N_MICRO=<n> [COMPILE=--compile] scripts/t5u4a_A0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/run/media/system/HDD_1TB/mets_data/hf}
export METS_DATA=${METS_DATA:?set METS_DATA to the main tree data dir}
N_MICRO=${N_MICRO:?set N_MICRO (from U4a's probe)}
COMPILE=${COMPILE:-}
T="python -m trackT_timaeus.t5h"
stage() {   # until, gate stage
    $T train --arm A0 --until "$1" --n-micro "$N_MICRO" $COMPILE
    $T read --arm A0
    $T gate --arm A0 --stage "$2"
}
stage 103 start      # first output populated; the loss falls over 1-100; init SDs; the step-0 band
stage 1253 trough    # T_A0 <= 0.25 by 1253, else A0 stops here
stage 4369 late      # L_A0 >= 0.60
echo "A0 passed its positive control"
