#!/bin/bash
# The registered P-I1 sweep, one checkpoint at a time, resumable.
#
# The step list is READ FROM THE REGISTRY, not restated here: a second copy
# could name a grid the gate refuses without a single test failing on the
# difference. REGISTERED_P_I1_SWEEP is a superset of REGISTERED_CLAIM_B_SWEEP,
# so the twelve tables already on disk are skipped.
#
# OFFLINE BY DEFAULT. On 2026-09-01 the first launch lost two checkpoints to
# HuggingFace's Xet CDN (us.aws.cdn.hf.co) timing out: run_1 reported the
# download failure, then printed "Results in:" anyway and exited 0, so the
# script read a directory containing no prompts as a success. Every revision is
# now mirrored locally, so a run that reaches for the network is a bug and
# HF_HUB_OFFLINE makes it say so instead of stalling.
#
# COUNT THE PROMPT DIRECTORIES BY NAME. The first version of that guard used
# `ls -d ${P1}/${M}_*`, which matches the per-prompt PNGs too and reports 64
# where it means 8 -- so it rejected three checkpoints whose phase 1 had
# succeeded, at ~20 minutes each. The check now asks for the eight directories
# it actually needs, which is what the phase-7 argument loop below already did.
#
# TELL PHASE 1 AND PHASE 2 APART BY WHAT IS IN THEM. Both write
# ${model}_${prompt} subdirectories into METS_RESULTS_DIR, so a reuse search
# that only counts those directories will hand a phase-2 directory to
# --phase1-dir. On 2026-09-03 it did exactly that for step143000, because
# phase 2 had already run and its directory was the newer of the two. Nothing
# errored; run_2 simply started over on the wrong input. The two are now
# identified by a file only that phase writes -- activations.npz per prompt for
# phase 1, phase2_verdict.json per prompt plus ov_decomp_${M}.npz at the top
# for phase 2 -- and phase 2 is reused when it is already there, for the same
# reason phase 1 is.
#
# PATHS ARE NOW DERIVED, NOT HARDCODED. On 2026-09-03 the repo and the bulk
# volume were no longer at /mnt/mets and /mnt/vm_storage -- they are the same
# two filesystems, mounted by label at /run/media/system/WDS_500/Mets and
# /run/media/system/HDD_1TB/vm_storage. A script that restates an absolute
# mount point stops working when the mount moves, and it fails by running
# `cd` into nothing rather than by saying so. REPO and DATA take the
# environment's value if set, and the scratchpad follows the caller.

REPO="${METS_REPO:-/run/media/system/WDS_500/Mets}"
# Everything generated now lives under the repo, so there is no second volume
# to name. METS_VOL is gone deliberately: it named /run/media/system/HDD_1TB/
# vm_storage, which named a VM that no longer exists -- the same mistake as
# /mnt/mets, one level up. One root, derived from where this script sits.
DATA="${METS_DATA:-$REPO/data}"
cd "$REPO" || { echo "no repo at $REPO"; exit 2; }
source .venv/bin/activate

# ASSERT THE INTERPRETER, do not trust the activation. `activate` writes an
# absolute VIRTUAL_ENV recorded when the venv was created; after the 2026-09-03
# remount it still said /mnt/mets/.venv and prepended a directory that no
# longer exists. It set VIRTUAL_ENV, returned 0, and `python` fell through PATH
# to a conda env carrying a different torch and transformers -- which is the
# one difference across a checkpoint grid that nothing in the artifact would
# record, since the phase-7 manifest stores git_sha but no library versions.
# sys.prefix is the only thing that actually answers "which interpreter".
WANT="$REPO/.venv"
GOT=$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)
if [ "$GOT" != "$WANT" ]; then
  echo "WRONG INTERPRETER: sys.prefix is '$GOT', expected '$WANT'"
  echo "  (which python: $(which python))"
  echo "  The venv's activate may still name an old path; fix it before running."
  exit 2
fi
python - <<'EOF' || exit 2
import sys, torch, transformers
want = {"torch": "2.13.0+cpu", "transformers": "4.57.6"}
got = {"torch": torch.__version__, "transformers": transformers.__version__}
if got != want:
    print(f"WRONG LIBRARY VERSIONS: {got} != {want}", file=sys.stderr)
    print("  The 19-step grid must be one environment; see PROJECT.md §1 on the "
          "transformers<5 pin and core/rope.py.", file=sys.stderr)
    sys.exit(2)
EOF
echo "interpreter: $GOT"
export HF_HOME="$DATA/hf"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export METS_RESULTS_DIR="$DATA/phase12"
SP="${METS_SCRATCH:-$DATA/logs}"
mkdir -p "$SP"
OUT="$DATA"
PROMPTS="hdbscan_code paper_excerpt wiki_paragraph camus_letranger latex_monograph sullivan_ballou homer_iliad repeated_tokens"
N_PROMPTS=8

# How many of this model's eight prompt directories are in $1, counting only
# those that carry the marker file $3 -- the file the phase in question writes
# and the other phase does not.
count_prompt_dirs() {
  local d="$1" m="$2" marker="$3" n=0 p
  for p in $PROMPTS; do [ -f "${d}/${m}_${p}/${marker}" ] && n=$((n+1)); done
  echo "$n"
}

# A complete phase-1 directory for this model, newest first, or "".
# Phase 1 is ~20 minutes and its output is reusable; re-running it because a
# LATER stage failed is the expensive way to be wrong.
find_existing_p1() {
  local m="$1" d
  for d in $(ls -dt ${METS_RESULTS_DIR}/*/ 2>/dev/null); do
    d="${d%/}"
    if [ "$(count_prompt_dirs "$d" "$m" activations.npz)" -eq "$N_PROMPTS" ]; then
      echo "$d"; return
    fi
  done
}

# A complete phase-2 directory for this model, newest first, or "".
# Requires the top-level OV decomposition as well as the per-prompt verdicts,
# since run_7 reads both.
find_existing_p2() {
  local m="$1" d
  for d in $(ls -dt ${METS_RESULTS_DIR}/*/ 2>/dev/null); do
    d="${d%/}"
    [ -f "${d}/ov_decomp_${m}.npz" ] || continue
    if [ "$(count_prompt_dirs "$d" "$m" phase2_verdict.json)" -eq "$N_PROMPTS" ]; then
      echo "$d"; return
    fi
  done
}

STEPS=$(python -c "from core.changepoint_colocation import REGISTERED_P_I1_SWEEP as G; print(' '.join(map(str, G)))")
echo "grid: $STEPS"
FAILED=""
for STEP in $STEPS; do
  M="pythia-410m-step${STEP}"
  if [ -f "${OUT}/phase7/step${STEP}/interaction_table.npz" ]; then
    echo "[$M] already done, skipping"; continue
  fi

  P1=$(find_existing_p1 "$M")
  if [ -n "$P1" ]; then
    echo "=== $(date +%H:%M:%S) [$M] phase 1 REUSED: $P1 ==="
  else
    echo "=== $(date +%H:%M:%S) [$M] phase 1 ==="
    P1=$(python -m p1_mstate_tracking.run_1 --models $M --prompts $PROMPTS 2>&1 \
          | tee ${SP}/sweep_${STEP}_p1.log | grep -oP 'Results in: \K.*' | tail -1)
    if [ -z "$P1" ]; then
      echo "[$M] PHASE 1 FAILED: no results directory"; FAILED="$FAILED $STEP"; continue
    fi
    N_GOT=$(count_prompt_dirs "$P1" "$M" activations.npz)
    if [ "$N_GOT" -ne "$N_PROMPTS" ]; then
      echo "[$M] PHASE 1 FAILED: $N_GOT of $N_PROMPTS prompt dirs in $P1"
      grep -iE "Failed|Network error|Traceback" ${SP}/sweep_${STEP}_p1.log | head -3
      FAILED="$FAILED $STEP"; continue
    fi
  fi

  P2=$(find_existing_p2 "$M")
  if [ -n "$P2" ]; then
    echo "=== $(date +%H:%M:%S) [$M] phase 2 REUSED: $P2 ==="
  else
    echo "=== $(date +%H:%M:%S) [$M] phase 2 ==="
    P2=$(python -m p2_eigenspectra.run_2 --full --models $M --prompts $PROMPTS \
          --phase1-dir "$P1" --random-dir none 2>&1 \
          | tee ${SP}/sweep_${STEP}_p2.log | grep -oP 'Results in: \K.*' | tail -1)
    if [ -z "$P2" ]; then
      echo "[$M] PHASE 2 FAILED"; FAILED="$FAILED $STEP"; continue
    fi
  fi

  ARGS=""
  for P in $PROMPTS; do
    [ -d "${P1}/${M}_${P}" ] && ARGS="$ARGS --prompt ${P}=${P1}/${M}_${P}"
  done
  echo "=== $(date +%H:%M:%S) [$M] phase 7 ==="
  if ! python -m p7_motifs.run_7 --p2-dir "$P2" --model $M $ARGS \
      --sign-channel schur --revision step${STEP} \
      --out ${OUT}/phase7/step${STEP} 2>&1 | tee ${SP}/sweep_${STEP}_p7.log; then
    echo "[$M] PHASE 7 FAILED"; FAILED="$FAILED $STEP"; continue
  fi
  if [ ! -f "${OUT}/phase7/step${STEP}/interaction_table.npz" ]; then
    echo "[$M] PHASE 7 wrote no table"; FAILED="$FAILED $STEP"; continue
  fi
  echo "$P1" > ${OUT}/phase7/step${STEP}/p1_dir.txt
  echo "$P2" > ${OUT}/phase7/step${STEP}/p2_dir.txt
  echo "[$M] done $(date +%H:%M:%S)"; df -h "$DATA" | tail -1
done
echo "SWEEP COMPLETE $(date +%H:%M:%S)"
[ -n "$FAILED" ] && echo "FAILED STEPS:$FAILED" && exit 1
echo "all steps present"
