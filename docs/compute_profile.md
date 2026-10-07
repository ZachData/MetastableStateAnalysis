# Compute profile: what each job costs to run

For sizing containers (Fargate / Batch) and instances. One row per job ×
machine, measured, never estimated. `peak RSS` is the maximum resident set of
the job's process (`/usr/bin/time -v`, "Maximum resident set size"); a row
without it says **unmeasured**. Add a row whenever a job runs somewhere new.

How to measure: `/usr/bin/time -v <cmd> 2> time.log`, then read
`Elapsed (wall clock)` and `Maximum resident set size` from `time.log`.

| date | job (exact command) | input | machine | wall | peak RSS | disk | notes |
|---|---|---|---|---|---|---|---|
| 2026-09-27 | `./scripts/check.sh` (lint + gate) | `c5b56fd` + #107 | t4g.small, aarch64, py3.10 | 2 min 19 s | 222 MB | — | 4 aarch64-only failures (`LESSONS.md` §4) |
| 2026-09-27 | `SMOKE_REAL_DEPS=1 pytest -m smoke` | `c5b56fd` | t4g.small, 1.8 GB, no swap | OOM at ~6 min | > 1.1 GB (killed at 1.14 GB anon RSS) | — | kernel OOM kill, 25 of 53 tests done |
| 2026-09-27 | same | `c5b56fd` | t4g.small + 4 GB swap | 23 min 53 s | unmeasured (≥ 1.1 GB; ~580 MB in swap) | 434 MB HF cache | swapping; expect far less on ≥ 4 GB RAM |
| 2026-09-27 | venv build (`rvm env build`: CPU torch + `requirements/heavy.txt`, hdbscan from source) | `9c0773c95606ad6f` | t4g.small | ~4 min | unmeasured | 275 MB zstd tarball | one-off per dependency change |
| ≤ 2026-09-24 | Phase-1 run, one prompt × checkpoint (410m, CPU) | v1 battery | local box | ≈ 200 s | unmeasured | 260 MB output | `STATE.md`, local box |
| 2026-09-25 | 1d tuning, one layer, quick grid | smoke, L0/12/18 | local box | ~170 s / layer | unmeasured | — | `p1d_cluster_ensemble/status-1d.md` "First real run" |
| 2026-09-26 | 1d merge tree (sub-experiment F), one prompt | 8 v1 prompts, step143000 | local box | ~1 s / prompt | unmeasured | — | `status-1d.md` |
| 2026-09-28 | 1d attention nulls, `run_all.sh` (4 configurations, 100 draws, `--workers 14`) | 8 v1 prompts × step143000, step 0 × L0–23 | local box | `null` 6.5 h, `calibrate` 7.1 h, each deduped 1.4 h | unmeasured | 65 MB JSON + parts | `status-1d.md` "Attention communities…" |
| — | Stage 0 (Phase 10), whole | 410m, 19 checkpoints | local box | unmeasured here | unmeasured | 57–100 GB | `STATE.md`; needs ≥ 100 GB ephemeral storage |
| 2026-10-05 | `move_text run --kept-from … --stage0-index …` (Phase 10 R0), one checkpoint | 7 v1 passages, both joins, 115 passes | local box, CPU, 14 workers | ~6.5 min (forward ~2/3 of it) | unmeasured | ~4 MB JSON | `p10_cluster_function/status-10.md` §1.14 |
| 2026-10-05 | one 410m forward, `homer_iliad` (512 tokens), float32, eager, TF32 off | step512, step54000 | local box, **RTX 3080 (10 GB)** | **40 ms** (CPU: ~2.7 s per pass inside `move_text`) | — | — | GPU probe below |
| 2026-10-07 | `p1e_energy_field.u2_block run` (1e U2 block arm), stored activations, no forward pass, CPU float64 | 8 long passages (1,032–2,041 tokens) × 18 steps + v1 × 18 | local box, CPU, 14 workers, 1 BLAS thread each | 143 s per long run alone; ~700 s each with 14 in parallel (memory bandwidth); 1 h 50 min for 269 runs | 14 GB total in use | 49 MB JSON | `p1e_energy_field/status-1e.md` "U2's block arm" |
| 2026-10-07 | `u2_block run --device cuda` / `agree` (1e U2 reader on the GPU, float64) | same, 6 runs re-read | local box, **RTX 3080**, one process | ~20 s per long run (frozen mode), 8 s (shared); about 2.5× the 14-worker CPU pool; float32 3.6 s but fails the agreement check | — | — | `status-1e.md` "GPU reading" |

## The GPU (local box)

The local box has an **RTX 3080, 10 GB**, and the conda `mets` env's torch (2.10, cu130) sees
it. Every run on disk is CPU because the scripts set `CUDA_VISIBLE_DEVICES=""` to match the
runs already stored (`handoff-10.md` §0.2), not because there is no GPU. (`STATE.md`'s "no GPU"
row is the cloud container.)

**Probe (2026-10-05, Phase 10 R0, `homer_iliad`, float32, eager, TF32 off):** a GPU P = 0 pass
against Stage 0's stored CPU activations gives a max unit-row difference of 2.0e-7 at step 512
and **7.1e-6 at step 54000** (CPU against CPU: 2–3e-7), so the late checkpoints use most
of `move_text.P0_MATCH_TOL` (1e-5). Even so, the level-set groups were identical to unit 1's
CPU groups in 96 of 96 (layer, frame, size) cells at both steps: float64 distances absorb
it here. One passage at two steps: not a guarantee for other passages or steps.

**Rule until measured more widely:** a cloud compared with a stored CPU run (Stage 0, the 1d
units' records) is produced on CPU. A new batch can use the GPU only if every cloud
it compares comes from the same device, or if it carries a match check like
`move_text --stage0-index`. The device goes in the record. `core/config.DEVICE` picks CUDA
whenever it is visible, and `MODEL_DTYPE` must stay float32: "auto" means bfloat16 on CUDA.

**Default (user, 2026-10-06): use the GPU for forward passes wherever the rule above allows**
(drop `CUDA_VISIBLE_DEVICES=""` from that run's script, keep float32, write the device into the
record); a new batch that stays on CPU says why. R7 (`design-10.md` "R7") ran on CPU because its
gate had already run there and its `orig` check compares with Stage 0 at 1e-5. Measured there,
CPU only: at step 143000 a 2-token pass and the 512-token pass of the same prefix differ by
1.5e-5 (relative norm), so late checkpoints already sit at the 1e-5 scale without a device change.
**In a 2,048-token pass the GPU / CPU gap is 6–25× larger** at step 143000's L14–24, at early
positions as much as late ones (step 0: 2e-7; numbers in `p1e_energy_field/status-1e.md` "The 8
long passages"): long-passage clouds from different devices are not interchangeable. 1e's 144 long runs are all GPU, ~7 s each, and
`output_attentions=False` is needed at 2,048 tokens (with attention maps kept, 410m runs out of
the 10 GB).

**Where it would pay:** forward passes only, ~65× per pass. `move_text` per checkpoint ~6.5 →
~2.5 min (its HDBSCAN / level-set work stays on CPU); Phase-1 runs, `arch_null`'s re-inits and
any new sweep likewise. Clustering, Gaussian nulls, permutation nulls and the label source are
CPU-bound Python or small matrices (n ≤ 600) and gain nothing. Not built: a `--device` flag
on `move_text` / `arch_null` that writes the device into the record.
