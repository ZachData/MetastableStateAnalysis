#!/usr/bin/env bash
# scripts/t5u4b_arms.sh — T5 U4b, second stage: the arms A1 (biases, H2), A2 (Pythia's init, H1) and A3 (both)
# to 4369 at t5h 81b2e03bdcf7, after the seed-noise gate passed (replicates.json, readable). Per arm, as
# t5u4b_replicates.sh: train to 103, read, the start gate (init SDs, the step-0 band, first output, the loss
# falls); for A1 and A3 also the step-0 equality with A0 and A2 (design-T5.md "U4", Gate: ρ_ℓ equal ≤ 1e-6),
# so A2 runs before A3. Then train to 4369 and read. The reading (t5i arms) is not run here.
# Stops at the first gate that refuses (exit 3). Resumable: rerun it.
#   METS_DATA=<main>/data scripts/t5u4b_arms.sh   (the configuration is t5h's N_MICRO, COMPILE)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/run/media/system/HDD_1TB/mets_data/hf}
export METS_DATA=${METS_DATA:?set METS_DATA to the main tree data dir}
T="python -m trackT_timaeus.t5h"
want=81b2e03bdcf7
read -r now root < <(python -c "from trackT_timaeus import t5h; print(t5h.code_sha(), t5h.ROOT)")
[ "$now" = "$want" ] || { echo "refusing: t5h code $now, the arms train at $want" >&2; exit 3; }
python -c "import json,sys; sys.exit(0 if json.load(open('$root/replicates.json'))['readable'] else 3)" \
    || { echo "refusing: $root/replicates.json not readable" >&2; exit 3; }
for arm in A1 A2 A3; do
    $T train --arm "$arm" --until 103
    $T read --arm "$arm"
    $T gate --arm "$arm" --stage start       # init SDs; the step-0 band; first output; the loss falls
    case "$arm" in A1|A3) python -m trackT_timaeus.t5i step0 --arm "$arm" ;; esac
    $T train --arm "$arm" --until 4369
    $T read --arm "$arm"
done
echo "arms A1-A3 trained and read to 4369"
