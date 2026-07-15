# Phase 1b (1h) — STATUS

**Last verified:** not recorded in source (run after Phase 1, `--phase1-dir results/2026-04-23_18-30-06`)
**Overall:** Complete. Blocks 0–4 run across all models; Blocks 5–6 not run (need Phase 2 OV artifacts, not supplied).

## Verdict table

| Block | Result |
|---|---|
| 0 — strong bipartition | **Fired (null).** 0% strong bipartition across all models. |
| 1 — identity persistence | Did not fire. Identity persistent = True for both ALBERT and GPT families. |
| 2 — HDBSCAN nesting near chance | Partial. Confirmed for GPT. Inconclusive for ALBERT. |
| 3 — cone-collapse holds everywhere | **Fired.** 100% cone-collapse, every model, every layer. Split regime never observed. |
| 5 — axis alignment | Not run (no Phase 2 OV artifacts passed in). |

**Global verdict:** paper alignment = `cone_collapse` for both ALBERT and GPT families. The
Phase 1 $k=2$ eigengap is a real, stable Fiedler axis (anisotropy direction), not an antipodal
bipartition — all tokens remain in one open hemisphere throughout.

## Known blockers

1. Blocks 5 (mechanism vs. OV/PCA/embedding/heads) and 6 (semantic MI) require Phase 2 OV
   decomposition artifacts. Attempted automatically if `--phase1-dir` is supplied and
   artifacts exist; not attempted in the run this STATUS reflects.
2. `run_1b.py`'s `_write_cross_run_md` contains hardcoded placeholder narrative (written
   pre-run) describing a split regime appearing at mid-depth. This is factually wrong given
   actual results and needs to be replaced with text conditioned on `paper_alignment`.

## Handoff notes (live constraints for later phases)

- Phase 4: don't treat the bipartition as a binary label — use the Fiedler axis as a
  continuous projection direction instead.
- Phase 5: hemisphere centroids remain usable as candidate cluster-identity vectors, but are
  the two extremes of an elongated cone, not antipodal cluster centers.
- Phase 6: the Fiedler axis difference vector and per-layer KL between centroid
  distributions remain valid probes.
