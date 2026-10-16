#!/usr/bin/env bash
# A2 unit 4 (STATE Blocked 58 (a1), (b1), (c1)): the gate at 64 nodes, the init band at step 0's βs,
# the batch over 18 steps, the report. Each stage refuses if the one before it did not pass.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${METS_DATA:?set METS_DATA to the main tree data dir}
G=$D/trackA/a2_gate64_2026-10-16
B=$D/trackA/a2_batch_2026-10-16
python -m trackA_force.a2_gate --out "$G"
python -m trackA_force.a2_batch band --out "$B" --gate "$G/a2_gate.json"
python -m trackA_force.a2_batch run --out "$B" --gate "$G/a2_gate.json"
python -m trackA_force.a2_batch report --out "$B"
