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
| — | Stage 0 (Phase 10), whole | 410m, 19 checkpoints | local box | unmeasured here | unmeasured | 57–100 GB | `STATE.md`; needs ≥ 100 GB ephemeral storage |
