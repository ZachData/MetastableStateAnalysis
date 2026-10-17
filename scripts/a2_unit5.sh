#!/usr/bin/env bash
# A2 unit 5 (STATE Blocked 59 (b), (ii)): the re-gate at the fixed instrument (top key by mass minus
# uniform's, the sink's exact share), the rule's calibration (synthetic G1 + step 0), the init band at
# 143000's βs and at step 0's, unit 4's batch re-run at the fixed instrument, then the real reading
# (refuses unless the calibration passed). Each stage refuses if the one before it did not pass.
# Run under systemd-inhibit (the machine suspends when idle).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${METS_DATA:?set METS_DATA to the main tree data dir}
G=$D/trackA/a2_gate5_2026-10-17
B=$D/trackA/a2_batch5_2026-10-17
U=$D/trackA/a2_unit5_2026-10-17
python -m trackA_force.a2_gate --out "$G"
python -m trackA_force.a2_ablate calib --out "$U" --gate "$G/a2_gate.json" || echo "calibration failed: the reading is not made"
python -m trackA_force.a2_batch band --out "$B" --gate "$G/a2_gate.json" --beta-step 143000
python -m trackA_force.a2_batch band --out "$B" --gate "$G/a2_gate.json"
python -m trackA_force.a2_batch run --out "$B" --gate "$G/a2_gate.json"
python -m trackA_force.a2_batch report --out "$B"
python -m trackA_force.a2_ablate real --out "$U" --gate "$G/a2_gate.json" || echo "real: refused"
python -m trackA_force.a2_ablate report --out "$U"
