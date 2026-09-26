#!/usr/bin/env bash
# Phase 1d on both sweeps of the same (step, prompt) runs, so the float-noise
# drift of each tuned family and of the consensus can be compared with the
# shipped HDBSCAN's (p10_cluster_function/status-10.md §3). Reader:
# tools/run/p1d_drift.py. Usage, from the checkout to run:
#   METS_DATA=<main>/data METS_PY=<conda mets python> p1d_drift_batch.sh <out_root> [step ...]
set -euo pipefail
DATA=${METS_DATA:?set METS_DATA=<main tree>/data}
PY=${METS_PY:-python}
PILOT=/run/media/system/HDD_1TB/Mets_archive/2026-08-12_05-01-35
OUT=$1; shift
mkdir -p "$OUT/logs"
STEPS=${*:-"143000 512 32"}
LAYERS="6 12 18"
KEYS="wiki_paragraph camus_letranger hdbscan_code homer_iliad latex_monograph paper_excerpt repeated_tokens sullivan_ballou"
export CUDA_VISIBLE_DEVICES="" METS_REPO=$PWD METS_DATA=$DATA
for step in $STEPS; do for key in $KEYS; do
  s0=$($PY -c "import json;print(json.load(open('$DATA/phase12/stage0_logs/stage0_index.json'))['runs']['$step|$key'])")
  for pair in "stage0 $s0" "pilot $PILOT/pythia-410m-step${step}_$key"; do
    set -- $pair
    [ -f "$OUT/$1/$(basename "$2")/p1d_results.json" ] && continue
    echo "$(date +%T) $1 step $step $key"
    $PY -m p1d_cluster_ensemble.run_1d --v1-only --results "$2" --out "$OUT/$1" \
      --layers $LAYERS --grid quick > "$OUT/logs/$1_${step}_$key.log" 2>&1
  done
done; done
# Self-control: the same Stage 0 input twice, so 1d's own run-to-run
# difference can be read beside the between-sweep one.
for sk in "143000 wiki_paragraph" "32 camus_letranger"; do
  set -- $sk
  s0=$($PY -c "import json;print(json.load(open('$DATA/phase12/stage0_logs/stage0_index.json'))['runs']['$1|$2'])")
  [ -f "$OUT/stage0_rerun/$(basename "$s0")/p1d_results.json" ] && continue
  echo "$(date +%T) stage0_rerun step $1 $2"
  $PY -m p1d_cluster_ensemble.run_1d --v1-only --results "$s0" --out "$OUT/stage0_rerun" \
    --layers $LAYERS --grid quick > "$OUT/logs/stage0_rerun_$1_$2.log" 2>&1
done
echo "$(date +%T) done"
