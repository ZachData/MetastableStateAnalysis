# docs/results_provenance_audit_2026-09-05.md — are the on-disk phase12 / phase7 results stale?

Prompted by the worry that `data/phase12/` and `data/phase7/` results predate a
refactor and were never deleted for a rerun. Audit run 2026-09-05 on branch
`claude/rescaler-cache-identity-test`, HEAD `73a8e63`.

**Short answer.** The data is **not old** — every on-disk run is 2026-08-31 to
2026-09-01, 4–5 days before this audit — and it **postdates every refactor I
could identify**. But the Phase 2 side has **no recorded git_sha at all**, and
the Phase 1 sweep is stitched from four commits over two days. Details and the
two real gaps below.

---

## 1. What produced each artifact

| Artifact | On disk | Produced by | Commit date(s) |
|---|---|---|---|
| **Phase 1 sweep** — 19 steps × 7 prompts, `activations.npz` + `llm_cross_run_report.txt` | `data/phase12/2026-08-31_*/`, `data/phase12/2026-09-01_*/` (non-`p2_eigenspectra_`) | `dd6b0de`, `36102e0`, `f395127`, `bbf7c0c` (manifest `git_sha`) | 2026-08-31 → 2026-09-01 |
| **Phase 2 eigenspectra** — 19 steps, `ov_projectors_*.npz` / `ov_decomp_*.npz` / `ov_weights_*.npz` / `p2_eigenspectra_cross_run.json` | `data/phase12/p2_eigenspectra_2026-08-31_*/`, `…_2026-09-01_*/` | **NOT RECORDED** — no manifest, no `experiment.txt`, no `git_sha` in any file. Directory mtime only. | dir mtime 2026-08-31 → 2026-09-01 |
| **Phase 7 interaction tables** — 19 | `data/phase7/step*/interaction_table.npz` | generated under `f395127` / `bbf7c0c` (manifest `git_sha`); 17 of 19 `.npz` **rewritten 2026-09-03 12:xx** by the recompression pass (`45b8429`), 2 (`step16000`, `step32000`) still carry 2026-09-01 mtimes | gen 2026-09-01, recompress 2026-09-03 |
| **analysis JSONs** — `curve.json`, `formation_series.json`, `behavioural_series.json`, `relay_null_series.json` | `data/analysis/` | rebuilt 2026-09-03 → 2026-09-04 on top of the above | — |
| **pre-float64 Phase 7 tables** | `data/superseded/phase7_float32/` (1.1 GB) | moved aside 2026-09-03 | handled correctly |

All four Phase-1 run commits are **linear ancestors of HEAD** (`git merge-base
--is-ancestor` = true for each) — no orphaned branch, no rebased-away history.

---

## 2. Is there a refactor *after* the runs that invalidates them?

**No.** The 15 commits from the last run commit (`bbf7c0c`, 2026-09-01) to HEAD
(`73a8e63`, 2026-09-04) are, in full: the P-I1 relay-count null construction,
the `paired_colocation_arm` per-unit skip, the attainable-floor record, the
behavioural-arm sweep, `POPPER_PLAN.md` §6v, the p=0.1414 scoring, and doc
promotions (`HANDOFF.md` → `PROJECT.md`, `OVERVIEW.md`).

`git log dd6b0de..HEAD` restricted to the modules that produced or consume these
artifacts returns **empty** for every one of:

```
core/metrics.py            core/sublayer_streams.py     core/models.py
core/dissipation.py        p2_eigenspectra/weights.py   p2_eigenspectra/decompose.py
p1_mstate_tracking/run_1.py  p1_mstate_tracking/analysis_p1.py  core/beta_eff.py
```

i.e. the energy / effective-rank / Fiedler definitions, the OV Schur
decomposition, the residual-stream extraction, and the sublayer-stream code have
**not changed since the earliest run commit**. `p1_mstate_tracking/reporting_p1.py`
(the normed-rank / `MinMass` report rewrite, D1/D3/D10) last changed
**2026-08-11** — the on-disk `llm_cross_run_report.txt` files were written by
HEAD's reporting code.

### Refactors the runs are already downstream of
- `transformers < 5` pin + smoke-tier fixes — `0ef60d0`, 2026-08-30 → runs are after.
- Phase 2 OV decompose in **float64** ("what Phase 7 refused") — `dd6b0de`,
  2026-08-31 → the Aug-31 Phase 1 runs are *at* this commit, the Sep-1 runs after.
- `core/` metric consolidation (transition plan v2), `core/sublayer_streams.py`
  rewrite, the 176 GB migration — all before 2026-08-25, well before the runs.
- float32 → float64 Phase 7 tables — old tables in `data/superseded/phase7_float32/`.

**If there is a specific refactor behind the original worry that is not in this
list, name it and I'll check the runs against it directly.**

---

## 3. Two real gaps

### 3.1 Phase 2 eigenspectra results carry no provenance
No `manifest.json`, no `git_sha`, no timestamp inside any file — not in
`p2_eigenspectra_cross_run.json` (a list of bare dicts), not in
`ov_summary_*.json`. The only evidence tying ~3 GB/checkpoint of OV projectors,
decompositions and weights to a commit is the **directory mtime**. This is the
artifact-contract bug class `INDEX.md` names ("a documentation reference that
silently resolves to nothing"), one level down: a *result* that resolves to no
commit. The co-location panel's `frac_repulsive` / `ov_frac_repulsive` series
and `tools/run/dissipation.py`'s subspace split both depend on these files.

**Fix:** Phase 2's runner should write a manifest with `git_sha` + lib versions,
same contract as Phase 1's `_write_run_manifest`. Until it does, cite the Phase 2
side of any figure as "dir mtime 2026-08-31/09-01, code assumed HEAD — §2 shows
the relevant modules are unchanged since, but this is inference not record."

### 3.2 The Phase 1 sweep is stitched from four commits over two days
Not a single-`git_sha` sweep. And at least one bucket is internally mixed:
`data/phase12/2026-09-01_16-48-38/` (step 16000) has 6 of 7 prompts under
`f395127` (2026-09-01 20:50–21:08) and **`homer_iliad` under `bbf7c0c`**
(2026-09-01 21:14) — one prompt regenerated six minutes later under the next
commit. The `f395127 → bbf7c0c` diff is "write edge and particle tables
compressed" (a Phase 7 table-writing change), so a Phase 1 activation difference
is unlikely, but the sweep is not clean.

`status-1.md` still lists D1/D3/D10 as "blocking / three verdict rows blocked."
The on-disk reports already carry the fixed columns (`MinRank` =
`effective_rank_normed`, `NormPR`, `MinMass`), so that language is at least
partly stale — the report side of D1/D3/D10 is done; D2 (the `sinkhorn.json`
per-head-Fiedler schema + rerun) is the part that may still be open.

### 3.3 Minor
- `data/phase7/step16000` and `step32000` `.npz` were not rewritten by the
  2026-09-03 recompression pass (still 2026-09-01 mtimes). Recompression is
  documented lossless (`PROJECT.md` §7.1, "0 differences" ×3), so likely benign;
  a `curve.py`-style re-read of those two would confirm.
- Every Phase 1 manifest has `sublayer_semantics: null` — the sublayer streams
  were never captured, so the dissipation attn/FFN split (Tier B) needs a fresh
  forward pass regardless. Already in
  `docs/dissipation_checkpoint_axis_scoping.md`.

---

## 4. Bottom line for current work

- **Co-location panel** (`data/analysis/colocation_panel.*`) and **dissipation
  Tier A** (`tools/run/dissipation.py`, subspace split + gradient-flow alignment
  + first-order ΔE from `activations.npz` + on-disk OV projectors): inputs are
  4–5 days old, from linear-history commits, and every producing module is
  unchanged since. **Usable, with the §3.1 provenance caveat on the Phase 2
  side recorded on the figure.**
- **The clean fix converges with work already planned.** A single-commit rerun
  of the 19 × 7 Phase 1 + Phase 2 sweep **with sublayer capture** is exactly
  Tier B of the dissipation scoping doc. Doing Tier B properly also closes §3.1
  and §3.2.
- **Independent of that:** give `p2_eigenspectra`'s runner a manifest (§3.1),
  and reconcile `status-1.md`'s defect list with what the on-disk reports
  actually contain (§3.2).
