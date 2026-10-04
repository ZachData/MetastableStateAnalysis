<!-- p1d_cluster_ensemble/status-1d.md -->
# Phase 1d — STATUS

## Revived 2026-09-25: the active thread

**Why now (user, 2026-09-25):** Phase 10's experiments are on hold until the
project can say what a cluster is, because every Phase 10 row reads one
HDBSCAN partition and that choice is a confound in all of them. Evidence
that it is: `p10_cluster_function/handoff-10.md` "Parked", first item.

**What was done to revive it.** The code was deleted on 2026-09-23 with its
branch and survived in the local tag `dead/particle-methods-comparison-vpuads`
(`010448c`, 2026-08-20). It was restored from the tag, not rewritten. Three
things had drifted:

| drift | fix |
|---|---|
| `clustering.py` now calls `HDBSCAN(**params)`; the test read inline kwargs | the test reads the `params` literal (`tests/test_phase1d_methods.py`) |
| `PHASE1D` was never on `main`'s `core/artifacts.py` | re-registered, verbatim from the tag |
| the holdout guard (`core/holdout.py`, 2026-09-24) did not exist | `run_1d.py` refuses the 12 held-out prompts; `--v1-only` / `--allow-holdout` |

1d's 112 tests pass, conda `mets` (the full gate: `./scripts/check.sh`).
Paths in the sections below are the tag's: `p1_visualization/` is now
`p1_mstate_tracking/visualization/`.

**What 1d answers, and what it does not.** It tunes seven families per layer
against subsample stability with a whole-pipeline null, and grades each token
by how many tuned families agree (core / halo / contested). That settles
"is HDBSCAN at `min_cluster_size=2` a good choice, and which tokens does the
choice matter for". It does not by itself supply the theory's definitions: the
fixed-scale one (strong Rényi centres at separation `δ`, F13/F14 in
`p10_cluster_function/math-10.md` §7) and the persistence one (tokens staying
together over a window of layers). Tuning was also against subsampling, not
the float-noise re-run that moves HDBSCAN (`p10_cluster_function/status-10.md` §3), so the
first real run should measure that drift too.

### First real run: a smoke test, not a result

**Input:** `pythia-410m-step143000_wiki_paragraph` (v1; Stage 0 pin `64a4087`,
through `data/phase12/stage0_logs/stage0_index.json`), layers 0 / 12 / 18,
467 tokens, `--grid quick`, defaults otherwise (`n_null 20`, `top_m 3`,
`alpha 0.05`, seed 0), conda `mets`, `hdbscan` 0.8.41. **Output kept:**
`data/p1d/smoke_2026-09-25/`. **Re-run:**

    METS_DATA=<main>/data python -m p1d_cluster_ensemble.run_1d --v1-only \
      --results <main>/data/phase12/2026-09-22_21-20-26/pythia-410m-step143000_wiki_paragraph \
      --out <main>/data/p1d/smoke_2026-09-25 --layers 0 12 18 --grid quick

**Cost:** 519 s wall, 6 618 s CPU (~13 cores), 280 MB RSS: about 170 s per
layer, so one run at all 25 layers is about 70 min on the quick grid.

**Two defects before it ran clean.** (1) `separation_score` crashed when a
null draw came back as one cluster per token. (2) After that fix, `calibrate`
dropped such draws as NaN, which silenced the families that behave best on
noise: agglomerative at thresholds 0.05 / 0.25 had 0 of 20 usable draws and
abstained at every layer, and HDBSCAN at L18 kept 18 draws, a p floor of 0.053
above alpha, so it could not pass. **The first version of this section
reported both abstentions as findings; they were the gate's.** Now a
degenerate null draw scores the floor (separation −1, stability 0), and a p
floor above alpha refuses under its own branch (`/challenge-pr` on #98,
finding 1; `selection.py` `DEGENERATE_NULL_*`).

Every array is populated. What the run shows, from one run, so for design
only:

| layer | families admitted | abstained | consensus strength | consensus k | core / halo / contested |
|---|---|---|---|---|---|
| 0 | 6 | gmm | 0.25 | 4 | 3 / 263 / 201 |
| 12 | 7 | — | 0.38 | 5 | 0 / 111 / 356 |
| 18 | 7 | — | 0.36 | 5 | 0 / 237 / 230 |

- **Ranking by subsample stability picks each family's extreme scale.**
  k-means, spherical k-means, spectral and GMM take k = 2 (spectral k = 3 at
  L18), their coarsest option. Agglomerative takes threshold 0.05, its
  finest, where most tokens are singletons and a partition reproduces
  trivially. HDBSCAN takes `min_cluster_size` 5 at L0 / L12 and the shipped 2
  at L18. Co-association averaged over these compares different questions, so
  consensus strength 0.25–0.38 says little about agreement. Stability's pull
  toward trivial scales is a known property (Ben-David, von Luxburg & Pál
  2006, raised in the review). **Before 1d is used to define a cluster, it
  needs a stated scale:** families compared at matched k or matched `δ`, or
  read as levels of one hierarchy. That redesign reopens `design-1d.md`, so it
  triggers a literature scan first (`CLAUDE.md` "Literature scans").
  **Scan done 2026-09-25: `lit-1d.md`.** Stage 1's ranking on raw stability is
  the known defect (von Luxburg 2010 §2: raw instability scales with k
  whatever the data). The standard remedy, normalising by the null, fails
  here: the null mean is 0 at the fine end, and it can move k to the grid's
  other end (`/challenge-pr` on #99). Agglomerative's pick at L12 is k = 452,
  95 % singletons, which the trivial filter lets through: **a defect**
  (`lit-1d.md` option A0). The scan gives the options (§5) and a recommendation
  for the user; `design-1d.md` stays unchanged until the user picks one.
- **A0 fixed 2026-09-25** (see "A0: the singleton bound" below). Agglomerative
  leaves its finest threshold at L12 / L18, not at L0.
- The driver's `P-C1`–`P-C4` lines are unregistered and meaningless on one run
  and a quick grid; they are now printed and stored under
  `verdicts_status: "UNREGISTERED, tier 1: not adjudications"`.
- The scratch agreement read that prompted the revival is in
  `p10_cluster_function/handoff-10.md` "Parked" (first item), not repeated here.

### A0: the singleton bound (2026-09-25)

**Change.** `partition_summary` now calls a partition trivial when more than
`TRIVIAL_SINGLETON_SHARE = 0.5` of all tokens are singletons (placed, a
majority rule; `selection.py`). It is the other end of `TRIVIAL_DOMINANCE`.
HDBSCAN's refusals (-1) are not counted. **Input:** the same run, layers, grid
and defaults as "First real run". **Output:** `data/p1d/smoke_a0_2026-09-25/`.
**Re-run:** the command above with `--out <main>/data/p1d/smoke_a0_2026-09-25`.

| layer | agglomerative before | after | consensus k before → after | ARI(consensus before, after) | core / halo / contested before → after |
|---|---|---|---|---|---|
| 0 | 0.05 (k 215, 31 % singletons) | **unchanged** | 4 → 4 | 1.00 | 3/263/201 → 3/263/201 |
| 12 | 0.05 (k 452, 95 %) | 0.25 (k 291, 43 %, stab 0.95) | 5 → 4 | 0.99 | 0/111/356 → **257**/88/122 |
| 18 | 0.05 (k 434, 89 %) | 0.25 (k 176, 15 %, stab 0.82) | 5 → 4 | 1.00 | 0/237/230 → 26/391/50 |

- **The pick changes where the bound bites (L12, L18), not at L0.** L0's
  pick is token identity: 215 clusters for exactly 215 distinct L0 vectors,
  since layer 0 is the embedding and repeated tokens are identical (checked
  on `activations.npz`; `/challenge-pr` on #100, finding 2). No trivial
  branch can see that. Whether L0 belongs in 1d's readouts is for the user.
  A0 removes the extreme case; it does not give agglomerative a scale. That
  is still D / C (`lit-1d.md` §5).
- A `ward, k = 2` grid point now enters the top 3 (admitted at L12, stability
  0.70; refused at L18). Stage 1 had not reached it before.
- **The consensus partition hardly moves; the grading moves a lot.** At L12,
  core goes from 0 to 257 tokens. The mechanism is in the ensemble's voting,
  and A0 does not remove it (`/challenge-pr` on #100, finding 1). A singleton
  votes *against* grouping on every pair it touches, while an HDBSCAN -1
  abstains. Each family's vote is weighted by its raw stability
  (`selection_weights`), which is highest for fine partitions. The reviewer
  rebuilt the observed ensemble without agglomerative, against the stored
  core threshold (the null thresholds were not recomputed, so this gives the
  direction only): core goes 3 → 178 at L0, 257 → 457 at L12, 26 → 285 at L18.
  **Core / halo / contested counts are not a usable readout** until singleton
  voting and the weights are decided. That is a design question for
  `design-1d.md`, together with D.
- The unregistered `P-C4` line flipped from "CONFIRMED 2/2" to "FALSIFIED 1/2"
  (`P-C3` 0 % → 7 %). These are tier 1 lines, not adjudications. The flip
  shows how little one run on the quick grid supports them.

### Float-noise drift: does tuning reduce it? (2026-09-25)

**Question.** The shipped HDBSCAN partition moves between two sweeps whose
activations differ only by float noise (`p10_cluster_function/status-10.md`
§3). Does 1d's output move less? Tier 1, descriptive, no null.

**Input.** 8 v1 prompts (V1 minus `short_heterogeneous`) × steps 32 / 512 /
143000 × layers 6 / 12 / 18 = **72 layer-records**, each run through 1d twice:
on Stage 0 (`stage0_index.json`, pin `64a4087`, battery `06790b90dcfe`) and on
the pilot (`HDD_1TB/Mets_archive/2026-08-12_05-01-35`). Steps and layers were
fixed before any 1d output was read: 32 has the most baseline drift, 512 and
143000 are the ones Phase 10 reads. L0 is left out (identical in both sweeps).
`--grid quick`, defaults otherwise, seed 0, conda `mets`, code at `ac1e188`.
**Output:** `data/p1d/drift_2026-09-25/` (`stage0/`, `pilot/`, `stage0_rerun/`,
`drift_stage0_vs_pilot.json`, `drift_self.json`). **Re-run:**
`METS_DATA=<main>/data METS_PY=<conda mets python> tools/run/p1d_drift_batch.sh
<main>/data/p1d/drift_2026-09-25` (~9 min per
run, 5.7 h total, skips done runs), then
`python -m tools.run.p1d_drift <out>/stage0 <out>/pilot` (and `<out>/stage0_rerun`
for the self-control). ARI keeps HDBSCAN
noise as a label, as §3 does.

| partition (72 records) | moved | ARI mean | p5 | min |
|---|---|---|---|---|
| shipped HDBSCAN (`mcs=2`) | 13 | 0.950 | 0.601 | 0.240 |
| tuned HDBSCAN | 7 of 70 | 0.969 | 0.886 | 0.223 |
| graph modularity | 6 of 71 | 0.994 | 0.991 | 0.615 |
| k-means | 1 of 71 (the self-control's size) | 1.000 | 1.000 | 0.991 |
| agglomerative, spherical k-means, spectral, GMM | 0 | 1.000 | 1.000 | 1.000 |
| **consensus** | **1** | 0.989 | 1.000 | **0.193** |
| core / halo / contested grade, share of tokens agreeing | 7 | 0.990 | 0.984 | 0.642 |

(A count "of n" leaves out records where the family abstained on both sides.)

- **Every family picks the same parameters on both sweeps, in all 72
  records.** The tuning stage itself does not drift.
- **Tuning HDBSCAN makes it move less often, not less far.** Where the
  shipped partition moved (13 records), tuned HDBSCAN (mostly
  `min_cluster_size` 5) moved in 7 of 11, with p5 0.28 against the shipped
  partition's 0.29.
- **The consensus moves once in 72, against 13 for shipped HDBSCAN. That
  does not show 1d finds a reproducible structure.** The families that never
  move are either coarse (k-means, spherical k-means, spectral, GMM at
  k = 2–5) or agglomerative. Agglomerative is fine, not coarse: k median 162
  over 70 records, and k ≥ 60 in 55 of them, finer than any tuned HDBSCAN pick
  (corrected after `/challenge-pr` on #101, finding 1; the first version said
  every stable family was coarse). But in those 55 records, 30 % of tokens are
  singletons and 74 % of the multi-member clusters are one repeated token
  string (`tools/run/p1d_drift_checks.py --fine`). A partition that groups
  identical strings and leaves the rest alone is stable almost by
  construction. On `repeated_tokens`, where the drift is, every
  non-density family is coarse (k 2–4) or abstains, so scale and method are
  confounded there. A matched-k run on that prompt would separate them
  (Parked).
- **When the consensus moves, it moves a lot, and it sits on a knife edge.**
  At `repeated_tokens` step 32 L6 its ARI is 0.193 (shipped: 0.240): a
  70-token cluster merges into the big one (sizes 193/70/1/1 → 251/12/1/1,
  Mirkin cut 0.465 → 0.439). Swapping the pilot's version in one family at a
  time leaves the consensus identical for every family, HDBSCAN included.
  It takes three changes together: HDBSCAN's moved labels (ARI 0.22), its
  weight (0.388 → 0.471) and graph modularity's weight (0.794 → 0.801); any
  one of them alone changes nothing
  (`tools/run/p1d_drift_checks.py --swap 32 repeated_tokens 6`; corrected
  after finding 4, the first version credited HDBSCAN alone). The weights are
  raw subsample stability (`/challenge-pr` on #100, finding 1), so drift
  enters both through labels and through weights. The one record does not
  show that the weighting makes drift worse; it shows the cut can be crossed
  by small combined shifts.
- **The drift is one prompt.** 9 of the 13 moved records are
  `repeated_tokens`, and so is every consensus or grade move larger than 1d's
  own run-to-run noise. Three grade moves elsewhere (0.996–0.998 agreement,
  `sullivan_ballou` 143000 L12, `wiki_paragraph` 143000 L12, `camus_letranger`
  512 L6) are the size of the self-control's 0.998, which was at exactly
  `wiki_paragraph` 143000 L12. On the other 7 prompts the shipped partition moved
  in 4 of 63 records (ARI ≥ 0.925), and 1d's consensus never moved.
- **The §3 baseline, per prompt** (all 13 shared steps; §3's rule, label
  vectors not identical, over all 25 layers; producer
  `tools/run/p1d_drift_checks.py --baseline`, reading only the stored
  `hdbscan_labels.json`). This is the number the other files point to:

  | | not identical, of all layers | min ARI | p5 ARI, layers ≥ 1 |
  |---|---|---|---|
  | `repeated_tokens` | **309 of 325** | 0.166 | 0.243 |
  | other 7 prompts | 126 of 2 275 | **0.925** | 0.990–1.000 per prompt |
  | all 8 (§3's 2 600) | 435 of 2 600 | 0.166 | |

  Counting ARI < 1 at layers ≥ 1 instead gives 307 of 312 and 118 of 2 184,
  because two `repeated_tokens` pairs differ in label numbering only. The
  mechanism, tested since: float32 distances (next section, `--precision`).
- **1d does not reproduce itself exactly on the same input.** The
  `stage0_rerun/` control (6 records: `wiki_paragraph` 143000,
  `camus_letranger` 32) gave k-means ARI 0.991 once, grade agreement 0.998
  once, and everything else identical. Seeds are fixed per call, so this is
  float summation order, likely threaded k-means. Six records show that
  it exists, not how big it is.

**Answer, for this design.** On the 7 prompts without mass repeats, the
shipped partition's drift is already small (ARI ≥ 0.925), and 1d's
consensus does not move at all. On `repeated_tokens` the drift is large for
every density family, tuned or not, and the consensus carries it once in 9
records. Whether 1d "reduces the drift" can't be told apart from what its
stable families pick (coarse k, or identical strings), so it waits on D / C,
and on the matched-k check on `repeated_tokens`.

### Matched k on `repeated_tokens`, and the float32 defect it led to (2026-09-25)

**Question** (Parked above). On `repeated_tokens`, is the drift a property of
density methods or of the fine scale they pick? **Input:** the 9
`repeated_tokens` records of the drift run (steps 32 / 512 / 143000 × L6 / 12 /
18), both sweeps, stored activations (float32); control: the 9
`wiki_paragraph` records. Code at `c61147e` + this unit's producer, conda
`mets`, `hdbscan` 0.8.41. **Re-run** (~2 min each): `python -m
tools.run.p1d_drift_checks <out>/stage0 <out>/pilot --matched repeated_tokens`,
and `--precision repeated_tokens`.

**Answer: neither. The drift is float32 rounding in the distance step, upstream
of every method** (found by `/challenge-pr` on #102; the first version of this
section named a method-level mechanism, withdrawn). Phase 1's stored labels and
1d's `LayerData` compute cosine distance in float32
(`p1_mstate_tracking/clustering.py` `cluster_tokens`, `pairwise_distances` on
float32 rows; `methods.py` `LayerData.from_normed`). `1 - x·y` loses its
significant digits when `x·y` is near 1, which is where `repeated_tokens`'
tokens sit. Shipped HDBSCAN refit on float64 distances from the same stored
activations (`--precision`):

| step | L6 / 12 / 18: stored A~B | float64 A~B | float64 ~ stored (A) | k stored → float64 |
|---|---|---|---|---|
| 32 | 0.240 / 0.495 / 0.320 | 1 / 1 / 1 | 0.067 / 0.082 / 0.146 | 62→30, 48→27, 53→27 |
| 512 | 0.485 / 0.688 / 0.699 | 1 / 1 / 1 | 0.279 / 0.337 / 0.656 | 50→24, 46→26, 37→25 |
| 143000 | 0.708 / 0.914 / 0.979 | 1 / 1 / 0.982 | 0.728 / 0.772 / 0.778 | 42→39, 43→42, 40→46 |

`wiki_paragraph`: float64 and stored labels are identical (ARI 1 in all 9),
and both sweeps agree (1 in 8, 0.988 in 1, the same in both precisions).

- **The stored `repeated_tokens` partitions are mostly rounding.** At steps
  32 and 512 the float64 partition shares little with the stored one (ARI
  0.07–0.66) and has about half as many clusters. The other six v1 prompts are
  unchecked; `wiki_paragraph`'s stored labels are exact. (All 8 checked
  since: next section.)
- **Matched k (`--matched`) does not separate method from scale, as first
  read.** At Stage 0's HDBSCAN k, k-means and Ward give ARI 1 across sweeps in
  16 of 16 rows, average linkage in 9 (min 0.51), spherical k-means (5 inits)
  in 10 (min 0.62), all on 1d's float32-derived distances. But k-means on one
  sweep, seed 0 against seed 1, agrees at 0.54–1.00 on `repeated_tokens` and
  **0.38–0.75 on `wiki_paragraph`**: at k ≈ 10–60 the k-means partition is
  seed-dependent everywhere, and "never moves" was the fixed seed. Average
  linkage's movement is probably the same float32 distances; not re-run on
  float64.
- **Withdrawn:** that the prompt is "one point" at early steps and its
  clusters name no structure. Float64 still finds 24–30 clusters, and whether
  their spread counts as one point depends on the scale `δ`, which waits on
  β's convention (Blocked 9). The reviewer reports the stored clusters at 32
  / 512 are runs of consecutive positions; not checked here.

**What it changes.** Phase 10 §3's floor (ARI p5 0.347, almost all
`repeated_tokens`) is mostly this defect, not HDBSCAN's sensitivity (routed to
`status-10.md`). The fix (float64 distances in `clustering.py` and 1d's
`LayerData`) changes stored labels, so it is its own unit with a re-run of §3.

**Registry.** `P-C1`–`P-C4` (`predictions-1d.md`) were never registered and
cannot be scored blind on the v1 runs already examined. They return as tier 1,
or get registered fresh against the held-out prompts (the user's call).

### Float64 distances, and Phase 10 §3's floor re-run on them (2026-09-26)

**Fix.** One route for cosine distance, `core.metrics.cosine_distance_matrix`
(float64 rows, `1 - x·y`, clipped, symmetric, zero diagonal), used by
`clustering.py` `cluster_count_sweep` (HDBSCAN, agglomerative, and the k-means
silhouette, now on the precomputed matrix), 1d's `LayerData`,
`backfill_hdbscan.labels_for_activations`, `p1d_drift_checks --precision` and
`p10_hdbscan_planted`. `clustering.json` (per layer and in its HDBSCAN block)
and backfill records now say `distance_dtype: "float64"`;
`backfill_hdbscan.labels_distance_dtype(run_dir)` reads it and returns
`"float32"` for anything unrecorded. The backfill keeps the float32 route only
for `--verify-pilot`, which still replays 3 pilot directories 25/25 identical;
its record says it verified the toolchain on that route, not the float64 labels.
Not changed: k-means still fits on float32 rows; Gram-matrix readers
(`multiscale_nesting`) do not subtract from 1 and are not affected. **Stored
labels are not rewritten**: everything in `data/phase12` and the pilot is
still float32-derived.

**Input.** The 104 (step, prompt) directories both sweeps hold: 8 v1 prompts ×
13 steps, pilot `HDD_1TB/Mets_archive/2026-08-12_05-01-35` against Stage 0
(`data/phase12`), 2 600 layer-records, activations differing by ≤ 7.9e-5
(≤ 2.5e-7 up to step 1000). Code at this PR's head, conda `mets`, hdbscan
0.8.41. **Re-run:** `python tools/run/p10_partition_stability.py --v1-only
[--refit] --out …` (47 s stored, 2 min 24 s refit, 16 cores); records
`data/analysis/p10_partition_stability_{stored,f64}_2026-09-26.json`. The
stored run reproduces §3's numbers exactly. The last column below is the f64
record's per-layer `ari_b_stored_vs_refit` (side b = Stage 0), over L ≥ 1.

| over 2 600 layer-records | stored (float32) | refit (float64) |
|---|---|---|
| label vectors identical | 83.3 % | **99.1 %** (23 differ) |
| ARI mean / p5 / min | 0.933 / 0.347 / 0.166 | 0.9997 / **1.000** / 0.471 |
| ARI noise-dropped p5 / min | 0.585 / 0.327 | 1.000 / 0.477 |
| cluster-count \|Δ\| mean / max | 0.58 / 20 | 0.008 / 9 |

| prompt (325 records each; ARI over L ≥ 1) | stored: not identical, p5, min | float64: not identical, p5, min | Stage 0 stored ~ float64: mean, p5, min |
|---|---|---|---|
| camus_letranger | 10, 1.000, 0.941 | 3, 1.000, 0.988 | 1.000, 1.000, 0.941 |
| hdbscan_code | 4, 1.000, 0.978 | 2, 1.000, 0.979 | 1.000, 1.000, 0.999 |
| homer_iliad | 40, 0.990, 0.929 | 0 | 0.998, 0.990, 0.946 |
| latex_monograph | 25, 0.993, 0.925 | 0 | 0.999, 0.996, 0.920 |
| paper_excerpt | 1, 1.000, 0.961 | 0 | 1.000, 1.000, 0.961 |
| **repeated_tokens** | **309, 0.243, 0.166** | **15, 1.000, 0.471** | **0.230, −0.044, −0.064** |
| sullivan_ballou | 27, 0.994, 0.950 | 1, 1.000, 0.994 | 0.999, 1.000, 0.945 |
| wiki_paragraph | 19, 1.000, 0.957 | 2, 1.000, 0.988 | 1.000, 1.000, 0.969 |

- **The floor was the float32 defect.** On float64 distances the two sweeps
  agree at ARI p5 1.0. Of the 23 records that still differ, 10 are at step
  143000, where the activations themselves differ by 2e-5 to 8e-5 (an input
  difference, not rounding), and 15 are `repeated_tokens` (4 are both). One is far off:
  step 8, `repeated_tokens`, L4, ARI 0.471 (37 vs 46 clusters) on activations
  1.7e-7 apart. So HDBSCAN on `repeated_tokens` still has near-ties at
  float64, rarely. **Not float64 cancellation** (`/challenge-pr` on #103
  asked): on that record, `1 - x·y` in float64 agrees with ½‖x̂ − ŷ‖² from
  direct differences (`scipy` `pdist`) to ≤ 2.2e-15, relative ≤ 2e-8 at the
  smallest distance (1.7e-8); refit on the direct route, labels are identical
  (ARI 1) and A~B stays 0.471.
- **Stage 0's stored `repeated_tokens` partitions are rounding, at every step
  and layer ≥ 1**: ARI to their float64 refit averages 0.230, p5 −0.044.
  The other 7 prompts' stored labels are the float64 labels up to small
  moves (mean ≥ 0.998, min 0.920). The previous section checked 9
  `repeated_tokens` records; this is all 325, and the other six prompts.
- **What it changes.** Every Phase 10 reader that pools `repeated_tokens`
  stored labels includes one prompt in eight whose partition is noise. The
  readers are not re-run here (Phase 10 is on hold); the labels would first
  have to be re-derived at float64 (Parked below). 1d's own drift run
  (72 records, "Float-noise drift") used float32-derived `LayerData`
  distances and is not re-run either.
- **Caveats.** One pair of sweeps, one machine; no null (a floor, as §3).
  ARI treats noise as a cluster (the noise-dropped column agrees).

**Parked** (discoveries, not followed):
1. Re-derive stored labels at float64 (a new `hdbscan_f64.json` per
   directory, or `read_labels` refitting), then re-run Phase 10's
   label readers. Why: `repeated_tokens` rows are noise. Cost: minutes for
   labels, readers an unknown hour or so. Changes: whether any Phase 10 row
   that pools prompts moves; decide when Phase 10 resumes.
2. Re-run 1d's drift (72 records) on float64 `LayerData`. Why: its
   "families never move / HDBSCAN moves" reading was on float32 distances.
   Cost: ~3 h. Changes: whether tuning matters for stability at all, now
   that the input noise is gone.

### Merge tree over scales, and clusters linked across layers (2026-09-26)

**What.** Option D (`lit-1d.md` §5), the item the user ordered first from
1d's "Next": read each layer's full agglomerative hierarchy as cluster
count `k` against distance threshold `delta`; take as that layer's
partition the longest-lived plateau with at least two substantial
clusters (>= `SUBSTANTIAL_CLUSTER_SIZE` = 4 tokens; smaller clusters are
outliers, -1); link neighbouring layers' partitions by token containment
(overlap / smaller cluster's size >= 0.5), and classify every linked
group as stable / merge / split / tangle (`merge_tree.py`, new
sub-experiment F, no prerequisites, no tuning). **Not done:** marking C's
`delta = c*beta_eff^-1/2` on the curve (waits on Blocked 9).

**Revised before merge; the first version's headline is withdrawn.** The
first version took the longest-lived plateau with no size condition and
linked on Jaccard >= 0.1, and reported "splits almost absent, merges at
L0→L2". Both were the method: at layers >= 2 that plateau is one cluster
holding 91–96 % of tokens plus a few outliers, and Jaccard cannot register
a piece leaving a large cluster (1 token of 460 is 1/460), so it records a
birth where there is a split. The L0→L2 "merges" were the pick jumping
from the token-identity partition to the blob. `/challenge-pr` on #104
did not catch this; a second read of the saved outputs did (lesson 6).
Also fixed: identical L0 vectors merge at ~0, which gave `wiki_paragraph`
L0 247 zero-width plateaus of 467 whose `k` no cut reproduces; tied merges
are now applied together (`HEIGHT_TIE_TOL`, placed).

**Input.** All 8 v1 prompts, pythia-410m step143000, all 25 layers, Stage
0: `data/phase12/2026-09-22_21-20-26/` (5 prompts) and
`2026-09-22_21-41-17/` (`camus_letranger`, `hdbscan_code`,
`latex_monograph`; `--v1-only` drops the two held-out prompts there).
Average linkage, min size 4, containment >= 0.5, conda `mets`, code at
this PR's head. **Output:** `data/p1d/merge_tree_2026-09-26/`.
**Re-run** (~3 s each):

    for R in 2026-09-22_21-20-26 2026-09-22_21-41-17; do
      METS_DATA=<main>/data python -m p1d_cluster_ensemble.run_1d --v1-only \
        --results <main>/data/phase12/$R \
        --out <main>/data/p1d/merge_tree_2026-09-26 --subexp F
    done

**A defect found building this: `--subexp F` was not actually cheap.**
`process_layer` ran the full tuning grid (stage A) regardless of
`stages`, because nothing gated it — the only genuinely skippable stage
was B. A `--subexp F` smoke run on real data (L0/12/18) took 2m51s wall
/ 39 min CPU, confirming it was paying for the untuned grid every time.
Fixed: `process_layer` now returns before stage A when `"A" not in
stages`; the same command now runs in 0.9 s, and all 25 layers of 5
prompts in about 3 s. The smoke output taken before the fix was
discarded, not reported.

**What the longest-lived scale is.** "Blob" = the longest-lived
non-trivial plateau has one substantial cluster plus outliers. "Token 0
out" = token 0 is an outlier in the chosen partition (the >= 2-cluster
plateau). Counts over L1–L24.

| prompt | n | distinct L0 vectors (= L0 floor k) | blob layers | token 0 out | chosen partition, substantial clusters L3–24 |
|---|---|---|---|---|---|
| camus_letranger | 465 | 237 | 11/24 | 17/24 | 2–26 |
| hdbscan_code | 242 | 125 | 18/24 | 19/24 | 2–16 |
| homer_iliad | 512 | 273 | 18/24 | 18/24 | 2–4 |
| latex_monograph | 446 | 171 | 20/24 | 19/24 | 2–29 |
| paper_excerpt | 286 | 163 | 20/24 | 20/24 | 2–11 |
| repeated_tokens | 265 | 3 | 8/24 | 5/24 | 2–3 |
| sullivan_ballou | 482 | 232 | 18/24 | 19/24 | 2–25 |
| wiki_paragraph | 467 | 215 | 13/24 | 14/24 | 2–7 |

- **At most layers the longest-lived scale is one cluster plus
  outliers:** 126 of 192 layer-records. Token 0, the attention sink, is
  among the outliers of the chosen partition in 131 of 192.
- **The two-cluster floor moves the extreme; it does not remove it.** At
  L3–L24 the chosen partition has exactly 2 substantial clusters in 112
  of 176 records. Ranking by absolute lifetime still favours the coarsest
  admissible scale, so the scale question D was meant to settle is open.
- L0's floor is token identity in all 8 prompts (floor k = distinct
  vectors; `repeated_tokens` has 3), and is trivial by construction now.

**Merges and splits against depth** (containment >= 0.5; 191 boundaries,
`repeated_tokens` L0→L1 skipped because L0 has no >= 2-cluster plateau):

| band | boundaries | merge | split | tangle | birth | death | stable |
|---|---|---|---|---|---|---|---|
| L0–L7 | 63 | 33 | 21 | 5 | 48 | 28 | 270 |
| L8–L15 | 64 | 16 | 12 | 19 | 7 | 4 | 75 |
| L16–L23 | 64 | 12 | 12 | 23 | 4 | 3 | 61 |

On the same labels, Jaccard >= 0.1 reports 242 births and 353 deaths
against containment's 59 and 35: most of Jaccard's births and deaths are
pieces entering or leaving a larger cluster.

- Early layers have more clusters per partition (up to 30 substantial at
  L0–L2), so the L0–L7 band has more events of every kind; compare kinds
  within a band, not counts across bands.
- A tangle is several clusters on each side sharing tokens: a reshuffle,
  not a coarsening or a refinement.
- **Caveats.** One checkpoint, one linkage (average), min size 4 and
  containment 0.5 placed, tier 1, descriptive, **no null**: a
  size-preserving random relabelling would say which of these rates a
  structureless sequence of partitions also produces.

**Answer, for this design.** Depth does not cleanly separate merges from
splits. Merges outnumber splits only in L0–L7 (33 to 21), where
partitions are still fine; from L8 on they are even (28 to 24) and
tangles are the most common change (42 of 112 non-stable events). The
firmer results are structural: at most layers the longest-lived scale is
one cluster plus outliers, token 0 usually among them, and requiring two
substantial clusters mostly picks exactly two. Which scale a cluster
lives at is still undecided.

**Parked** (discoveries, not followed):
1. Mark C's `delta = c*beta_eff^-1/2` on the merge-tree curve. Why:
   waits on beta's convention (Blocked 9). Cost: cheap once decided.
   Changes: whether the theory's scale sits in a robust plateau.
2. A null for the link counts (random relabelling preserving cluster
   sizes at each layer). Why: tangle and split rates mean nothing
   without one. Cost: minutes. Changes: whether any depth pattern above
   is real.
3. Rank plateaus by something other than absolute lifetime (relative
   lifetime, or a balance condition). Why: the two-cluster pick is the
   new extreme. Cost: small. Changes: which partition F links.
4. Drop token 0 before building the tree. Why: it is an outlier in 131
   of 192 records; whether the blob remains without the sink is the
   question. Cost: minutes. Changes: the blob count.
5. Other checkpoints (steps 32, 512 have the most drift history).

Parked 2 and 4 are done: next section.

### The link counts against a size-preserving null, and without token 0 (2026-09-26)

**What.** Parked 2 and 4 above, one unit (user, 2026-09-26). **Null:**
each draw permutes every layer's saved partition independently, which
keeps each layer's cluster sizes and outlier count and destroys which
tokens share a cluster across layers, then links as stage F does
(`merge_tree.link_chain_null`; `link_counts` is a contingency-table
version of `link_layer_pair`'s counts, equal on every tested input).
`merge_null.py` re-links the saved partitions, refuses unless the counts
equal what the run wrote, pools the 8 prompts draw by draw and bands by
`layer_from`. **Token 0:** `run_1d --drop-tokens 0` (stage F only; other
stages compare against shipped labels over every token) rebuilds each
tree without it.

**Input.** The 8 v1 prompts of the previous section, pythia-410m
step143000, 25 layers; full: `data/p1d/merge_tree_2026-09-26/`; without
token 0: `data/p1d/merge_tree_drop0_2026-09-26/`. 2000 draws, seed 0,
containment >= 0.5, conda `mets`. **Output:**
`data/p1d/merge_null_2026-09-26/null_{full,drop0}.{json,txt}`.
**Re-run** (~5 s for the trees, ~85 s per null):

    for R in 2026-09-22_21-20-26 2026-09-22_21-41-17; do
      METS_DATA=<main>/data python -m p1d_cluster_ensemble.run_1d --v1-only \
        --results <main>/data/phase12/$R --subexp F --drop-tokens 0 \
        --out <main>/data/p1d/merge_tree_drop0_2026-09-26
    done
    python -m p1d_cluster_ensemble.merge_null --v1-only --n-draws 2000 \
      --in <main>/data/p1d/merge_tree_2026-09-26 \
      --out <main>/data/p1d/merge_null_2026-09-26/null_full.json

**The null is structureless in a specific way.** With a blob of 91–96 %
of tokens on both sides, almost any small cluster is >= 50 % inside the
other layer's blob, so every cluster links through it: from L8 the null
makes one tangle per boundary (63.5 of 64) and no stable links (0.005).
Every observed count below differs from it at p = 0.0005 (the floor at
2000 draws), so "beats the null" is only informative for the two
statistics that could have gone either way.

| band | statistic | observed | null mean [2.5 %, 97.5 %] | p (upper) | without token 0: observed, null mean, p (upper) |
|---|---|---|---|---|---|
| L0–L7 | merge − split | 12 (33 − 21) | 8.0 [4, 12] | 0.061 | 17 (36 − 19), 7.9, 0.001 |
| L0–L7 | merge / (merge + split) | 0.611 | 0.679 [0.579, 0.789] | 0.89 | 0.655, 0.693, 0.70 |
| L8–L15 | tangle share of non-stable | 0.328 | 0.958 [0.912, 1] | 1.0 (lower 0.0005) | identical |
| L16–L23 | tangle share of non-stable | 0.426 | 0.974 [0.939, 1] | 1.0 (lower 0.0005) | identical |

- **The L0–L7 merge excess is not distinguishable from this null.** The
  raw difference beats it (p 0.06, or 0.001 without token 0); the merge
  fraction does not (0.61 real, 0.68 null). Neither comparison is clean:
  the null's events are of a different kind (292 births against 48 real;
  its "merges" are small random groups that happen to fall inside a large
  cluster). What needs no null: cluster counts fall across L0–L7 (net 100
  substantial clusters fewer over 63 boundaries; 25 falling, 18 rising,
  20 flat), which by itself favours merges. Whether any merge/split
  asymmetry is left beyond that fall is open (Parked 6).
- **The late-layer tangle share is below the null, not above it.** Size
  alone makes nearly every change a tangle (0.96–0.97); the real share is
  0.33–0.43. So "tangles are the most common change" is not evidence of
  reshuffling.
- **Persistence beats this null, and that is not evidence that clusters
  are real.** Neighbouring layers of a residual stream are nearly the same
  vectors, so any clustering of them carries over, and a null that
  relabels each layer independently is beaten by any carry-over. The late
  bands have about one stable link per boundary (136 over 128), mostly
  the large cluster mapping to itself; of the 108 late boundaries with a
  2-cluster side, 43 have no stable link.
- **Token 0 is not special; the blob plateau is set by one or two far
  tokens.** Without token 0, 194 of 199 chosen partitions are identical
  (the 5 that differ are L0–L7), L8–L23 link counts are identical, and the
  blob is still the longest-lived scale in 104 of 192 layer-records (126
  with it). 23 records stop being blobs and 1 becomes one; 21 of the 23
  were "everything plus one outlier token". Of the 104 left, 41 are
  "everything plus one token" and 73 have at most 2 outlier tokens: under
  absolute lifetime, any single far-away token makes "all the rest" a
  long-lived plateau, and another token takes the sink's place. This is
  Parked 3's case against ranking by absolute lifetime. Two-cluster picks:
  110 of 176 (112 with it).
- **Caveats.** This null is weak by design: it keeps nothing across
  layers, so it can say a pattern is size-driven, not that the rest is
  dynamics. One checkpoint, one linkage, min size 4 and containment 0.5
  placed, tier 1. Revised after `/challenge-pr` on #105, which found the
  first write-up claimed persistence as a positive result and the
  cluster-count fall as a tested cause.

**Answer.** Neither depth headline of the previous section is supported:
the merge excess cannot be separated from the falling cluster count under
this null, and the tangle share is below what sizes alone give. Nothing
here is positive evidence about clusters: beating an independent null is
what any carry-over between neighbouring layers does. The blob is not the
sink; it is what absolute lifetime makes of one or two far tokens.

**Parked** (discoveries, not followed):
6. A null that keeps cross-layer persistence and randomises only the
   event (e.g. move k random tokens between clusters per layer, k matched
   to the observed churn). Why: the independent null is beaten on every
   count, so it cannot rank the depth pattern. Cost: an hour. Changes:
   whether any merge/split asymmetry is left beyond the cluster-count fall.

### Matched-covariance Gaussian null: are tokens lumpier than their covariance explains? (2026-09-26)

*Later (2026-09-28, #108):* this null draws each position independently, so
structure that is only smooth along the sequence beats it; untrained step 0
beats the same null in item (3). A position-keeping residual null is Parked
under "Attention communities against three nulls".

**What.** A new unit before item (3), at the user's call (2026-09-26). The null is
SigClust's (`lit-1d.md` §7): one Gaussian with the tokens' own mean and
plug-in covariance, `n` draws per layer, renormed, then clustered by the
same routes as the tokens (`gaussian_null.py`). There are three frames:
`raw` (L2-normed, as 1d does now), `centred` (mean direction projected
out, renormed) and `centred_norogue` (the 3 coordinates with the largest
`m_i²` zeroed first; Timkey & van Schijndel's measure). The Gaussian is refitted in
each frame. The six statistics: `ci2` (SigClust's 2-means index), `nn1` (mean cosine
distance to the nearest token), `hdb_k` / `hdb_noise` (Phase 1's shipped
HDBSCAN, `min_cluster_size=2`, float64 cosine), and `mt_k` / `mt_life`
(stage F's top robust plateau: substantial clusters, lifetime).

**Input.** The 8 v1 prompts, pythia-410m, step143000 (Stage 0,
`2026-09-22_21-20-26`, `2026-09-22_21-41-17`, git `64a4087`) and step 0
(Stage 0, `2026-09-23_05-52-32`, `2026-09-23_06-02-15`), all 25 layers,
every token, 200 draws, seed 0, conda `mets`. **Deduped:** the same with
each token string's first occurrence only (125–273 tokens;
`repeated_tokens` has 3 strings and is skipped). **Calibration**, one per
input set (all tokens; deduped): each layer is replaced by one draw of its own
Gaussian and the null refitted to it, all 25 layers, 100 draws. The code is
this PR's. The full run predates the `--calibrate` / `--dedupe-strings` flags,
whose default path is unchanged. **Output:**
`data/p1d/gaussian_null_2026-09-26/`: `null`, `null_dedupe`, `calibrate`,
`calibrate_dedupe` (`.json` + `.txt`), and the tables below in
`report_{full,dedupe}.txt`. **Re-run** (~24 / 6 / 12 / 4 min at 14
workers, report ~1 min; `$RUNS` = the 16 directories in `null.json`
`inputs`):

    OMP_NUM_THREADS=1 python -m p1d_cluster_ensemble.gaussian_null --v1-only \
      --n-draws 200 --workers 14 --runs $RUNS [--dedupe-strings] \
      --out <main>/data/p1d/gaussian_null_2026-09-26/null[_dedupe].json
    # calibration: the same with --calibrate --n-draws 100, out calibrate[_dedupe].json
    python -m p1d_cluster_ensemble.gaussian_null_report --null null_dedupe.json \
      --calibrate calibrate_dedupe.json --out report_dedupe.json

**The geometry the frames remove** (the previous session's quick look,
now measured; raw frame, 7 prompts without `repeated_tokens`):

| step | band | mean-direction share `(y·m̂)²` | top-3 rogue coords, mean-square share | commonest top rogue coord | effective dims raw / centred / no-rogue (median) |
|---|---|---|---|---|---|
| 143000 | L0 | 0.08–0.17 | 0.03–0.08 | 443 | 47 / 44 / 45 |
| 143000 | L1–8 | 0.28–0.50 | 0.18–0.46 | 278 (49 of 56) | 58 / 57 / 62 |
| 143000 | L9–16 | 0.29–0.50 | 0.14–0.39 | 966 (40 of 56) | 64 / 66 / 75 |
| 143000 | L17–24 | 0.35–0.66 | 0.06–0.32 | 125 (49 of 56) | 21 / 23 / 30 |
| 0 | L1–24 | 0.13–0.58 | 0.006–0.021 | none stable | 48–53 / 44–45 / same |

**Calibration: the null is not at nominal level, and it depends on the
input set.** Records in the lumpier 2.5 % tail when the null is true,
step143000, L1–24, 7 prompts (168 per cell; `report_{full,dedupe}.txt`).
Step 0's calibrations put at most 12 of 168 in any tail.

| statistic | all tokens: raw / centred / no-rogue | deduped: raw / centred / no-rogue |
|---|---|---|
| `ci2` lower | 13 / 42 / 27 (centred L17–24: 35 of 56) | 1 / 5 / 2 |
| `nn1` lower | 6 / 19 / 8 (median z +3.1 to +4.6) | 0 / 0 / 0 (median z +6.2 to +8.8) |
| `hdb_k` upper | 0 / 0 / 2 | 0 / 1 / 0 |
| `mt_k` upper | 1 / 0 / 0 | 0 / 1 / 0 |
| `mt_life` upper | 7 / 17 / 10 | 6 / 4 / 7 |
| `hdb_noise` lower | 7 / 4 / 2 | 4 / 1 / 4 |

So each real result is read against the calibration on **its own inputs**.
`gaussian_null_report.py` refuses any other pairing. The module's first docstring called the plug-in
bias conservative. It is not, for `ci2` on all tokens.

**Results, deduped** (the reading; L1–24, 7 prompts, 168 records per cell;
tail = records in the lumpier 2.5 %, real / calibration; z = median, calibration in brackets):

| statistic | frame | step143000: tail, z | step143000 L17–24 (of 56): tail | step 0: tail |
|---|---|---|---|---|
| `ci2` | raw | 45 / 1, −1.20 [+0.49] | 23 / 1 | 0 / 0 |
| `ci2` | centred | 57 / 5, −1.27 [+0.27] | 42 / 5 | 0 / 0 |
| `ci2` | no-rogue | 28 / 2, −0.73 [+0.77] | 19 / 2 | 0 / 0 |
| `nn1` | raw | 168 / 0, −15.3 [+6.2] | 56 / 0 | 0 / 0 |
| `nn1` | centred | 168 / 0, −19.4 [+7.9] | 56 / 0 | 0 / 0 |
| `hdb_k` (median obs vs null) | raw | 67 / 0, 5 vs 3.2 | 23 / 0 | 1 / 2 |
| `hdb_k` (median obs vs null) | centred | 109 / 1, 20 vs 6.1 | 43 / 0 | 2 / 3 |
| `mt_k` (median obs vs null) | centred | 2 / 1, 2 vs 2.6 | 0 / 1 | 4 / 5 |
| `mt_life` | raw | 9 / 6 | 6 / 2 | 59 / 1 |
| `mt_life` | centred | 38 / 4, +0.96 [+0.35] | 17 / 4 | 59 / 1 |
| `hdb_noise` | raw | 37 / 4 | 16 / 2 | 2 / 1 |

**Results, all tokens** (centred; tail real / calibration): `ci2` 165 / 10
at step 0, 149 / 42 at step143000. `nn1` 168 / 0 and 168 / 19. `hdb_k` 168
/ 7 (63 vs 5.4 groups) and 168 / 0 (53 vs 4.3). `mt_k` 167 / 0 (20 vs 2.1)
and 21 / 0 (2 vs 2.1). `mt_life` 168 / 12 and 37 / 17.

**What a token's nearest neighbour is** (centred, median over
prompt-layers L1–24 [max]; `report_*.txt` has raw, which agrees):

| | same string | adjacent position | within 3 positions |
|---|---|---|---|
| step 0, all tokens | 0.66 (0.56–0.75) | 0.01 | 0.06 |
| step143000, all tokens | 0.52 (0.26–0.75) | 0.15 [0.48] | 0.22 |
| step 0, deduped | — | 0.04 | 0.08 |
| step143000, deduped | — | 0.19 [0.44] | 0.30 [0.49] |

- **With every token, both checkpoints beat the null, and step 0 beats it
  harder.** That is token identity: at step 0 a token's nearest neighbour is
  the same string for 56–75 % of tokens, flat over all 25 layers. A
  Gaussian cannot make duplicates, so any tokenised text beats this null.
- **Deduped, step 0 is at its calibration** on `ci2`, `nn1`, `hdb_k`,
  `mt_k` and `hdb_noise`. **The exception is `mt_life`:** the top plateau outlives
  the Gaussian's in 59 of 168 records (calibration 1), 37 of 56 at L17–24
  raw. So this negative control is not clean for lifetime, and step143000's
  lifetime excess (38 / 4 centred) cannot be read as learned structure.
- **Deduped, step143000 is lumpier locally.** Nearest neighbours are
  closer in all 168 records, and HDBSCAN finds 20 groups where the Gaussian
  gives 6 (centred). The HDBSCAN excess rises with depth: 22 / 44 / 43 of 56 at
  L1–8 / L9–16 / L17–24. **Part of it is position:** 30 % of nearest
  neighbours are within 3 positions (19 % adjacent), against 8 % (4 %) at
  step 0. About 70 % are farther.
- **Deduped, step143000 also has a global 2-means excess, at late
  layers, in every frame:** 42 of 56 records at L17–24 centred
  (calibration 5), 23 raw (1), 19 no-rogue (2). The median z are small
  (−1.2 to −2.5) but consistent. **The merge tree's 2-cluster pick is what
  the Gaussian gives** (`mt_k` 2 vs 2.1–3.3, tail 0–6 vs 0–1), so the
  "exactly 2 clusters in 112 of 176" of the merge-tree section is not by
  itself evidence of two clusters.

**Answer.** Yes. Once repeated strings are removed, the trained model's tokens are lumpier than a
Gaussian with their covariance in every frame. The excess is local (closer
neighbours and more small HDBSCAN groups, partly sequence position) and,
at L17–24, a 2-means split beyond the calibration. The merge tree's
2-cluster pick is Gaussian-typical. The untrained model has nothing beyond
token identity except a merge-tree lifetime excess nobody has explained.
Phase 10's `min_cluster_size=2` partitions on all tokens are mostly token
identity at step 0 (63 groups vs 5 for its Gaussian) and still carry it at
step143000.

*Revised after `/challenge-pr` on #106.* The first write-up said "only
locally; nothing global". It read deduped results against an all-token
calibration pooled over 5 layers, whose centred `ci2` tail fires 35 of 56 at
L17–24. With the calibration on the same inputs, the late 2-means excess is
there in every frame.

**Caveats.** One seed, one checkpoint pair, one model. Plug-in covariance,
not at nominal level (table above). The calibration is one pseudo-draw per
record at 100 draws, against 200 for the real runs. The first occurrence of each string is kept,
which favours early positions. "Closer neighbours than a Gaussian" is not
"clusters": position already explains 30 %, and a curve or sheet does the
same. Tier 1, unregistered.

**Parked** (discoveries, not followed):
7. The deduped step143000 close pairs beyond position: case or sub-word
   variants of one word, or something else. Why: whether the local excess
   is clusters or lexical and sequence structure. Cost: ~30 min on
   `report_dedupe.json`'s inputs. Changes: whether HDBSCAN's small groups
   mean anything past token identity and position.
8. SHC (`lit-1d.md` §7 row 2): this null at every merge-tree node, FWER-controlled.
   Why: it would test the tree's nodes, not only its top plateau. Cost: a
   few hours. Changes: which merges, if any, are more than Gaussian.
9. A null at nominal level (SigClust's soft-thresholded eigenvalues). Why:
   the all-token calibration fires 25 % on centred `ci2`. Cost: ~2 h.
   Changes: only needed if all-token results matter; the deduped
   calibration is near nominal for `ci2`.
10. What "the same null" means for item (3). Attention communities are read off
    attention matrices, which a Gaussian draw does not have. A kernel
    `softmax(β x·y)` on the draws needs β, which is Blocked 9. Needs the
    user's call when item (3) opens.
11. Step 0's deduped merge-tree lifetime excess (59 of 168, calibration 1).
    Why: an untrained model should not beat this null, so it is either
    structure at init (position through the causal mask?) or a defect in how
    lifetime is compared. Cost: ~1 h. Changes: whether `mt_life` can be
    read at all.
12. *(2026-09-29, from `FUTURE_IDEAS.md` §3 D8; possibly a confound, the
    user decides whether to attach.)* The rogue coordinates and the token-0
    sink both have optimiser-level explanations in the literature (Adam's
    diagonal preconditioning; weight decay). Why: part of what 1d reads as
    structure may be Adam-induced coordinates. Cost: reading, then a null
    that drops outlier coordinates (1d already drops 3). Changes: whether
    1d's cluster definition must be stated modulo optimiser artefacts.

Parked 10 is decided: next section.

### Attention communities against three nulls (item 3; DONE 2026-09-28)

**State: run complete; results under "Results" below.** 4 configurations ×
384 layer-records, 100 draws; output `<main>/data/p1d/attention_null_2026-09-26/`,
reports `report{,_dedupe}.{json,txt}`; code `8aab3cf` (since then only
`summarise`'s z guard changed, which moves no tail count). Page:
`viz_page.py`, published as a private artifact
(https://claude.ai/artifact/XYbhuwAvMkfkWgv27v9qVR).

**What.** Item (3), with Parked 10 decided (user, 2026-09-26: "all tests and
all nulls"). Graph per (run, layer, window w = 1–3): head-averaged attention,
sink (position 0) dropped as a node and rows renormalised, rolled out as
`prod(I/2 + M/2)`, made undirected as `mutual` `(R+Rᵀ)/2` and `coattn`
`R Rᵀ`; Leiden on weighted modularity (igraph). Statistics: modularity `Q`,
λ₂ of the normalised Laplacian, community counts, position contiguity, weight
within 3 positions, and ARI with k-means at the same k in each of #106's
three frames (the "which frame does attention agree with" readout).
`attention_graph.py`, `attention_null.py`, `neox_block.py`, report
`attention_null_report.py`, page builder `viz_page.py` + `viz/index.html`.

**"The same null" (Parked 10), answered:** the model's own block applied to
#106's null. Nothing in it needs β.

| null | construction | answers |
|---|---|---|
| A | Gaussian with the kept non-sink tokens' covariance (`gaussian_draw`, raw frame), rows at their real tokens' norms, the real sink row at position 0, read by the checkpoint's own LN1/QKV/rotary; for w > 1 carried through the real blocks, not redrawn | communities beyond what this layer's attention makes of a structureless cloud |
| B | each offset diagonal shuffled across rows, *relative to uniform* (`A[i,j]·(i+1)`), rows renormalised | beyond recency / positional heads |
| C | the theory's head `softmax(β_h⟨u_i,u_j⟩)` on unit LN1 rows, β_h fitted per head to the real attention, on A's propagated draws; plus C-real, the same head on the real tokens (`ari_kernel`) | beyond the idealised cosine coupling |
| control | step 0, all of the above | does training create it |

Calibration: the real run replaced by one null-A draw (seed offset), everything
refitted: #106's convention.

**Checks passed before the run.** The reimplemented block reproduces the stored
attention (max abs 1e-7 to 1.5e-3, largest at L18/L23) and the next residual
(rel ≤ 3.3e-5) on `wiki_paragraph` at both checkpoints (`--verify` refuses
otherwise); it matches transformers' `GPTNeoXLayer` on a random tiny config
(`tests/test_phase1d_neox_block_smoke.py`). 13 unit tests.

**Defect found and fixed before the run.** Null B first shuffled raw weights.
On step 0's near-uniform attention (`1/(i+1)` per row) that moved early rows'
large weights onto late rows and made the shuffle *more* modular than the
real matrix (Q 0.23 vs 0.14, smoke run). Fixed by shuffling relative to
uniform; uniform attention is now a fixed point (test).

**β, measured on the way (bears on Blocked 9).** On the unit LN1 frame the
slope of the softmax's own input (`core/beta_eff.py`'s `beta_raw`, row fixed
effects, offset control, no `attn_scale`) on `pythia-410m-step143000` /
`wiki_paragraph` (run `2026-09-01_18-25-12`, the one `status-1c.md` finding 2
used, sink included as `estimate_beta_all_heads` does): median **4.00**, IQR
[2.05, 6.01], range [−6.70, 17.52], R² median 0.18, 384 head-rows.
**Divided by 8 it is finding 2's numbers to the digit** (0.50, [0.26, 0.75],
[−0.84, 2.19]). The code's `attn_scale = 1/8` path gives 32.02. So "scaled"
0.50 divides the model's `1/√d_h` out a second time; `beta_raw` is already
the β of `softmax(β⟨u_i,u_j⟩)`. On the Stage 0 run of the same prompt,
without the sink: 4.44 [2.54, 6.68]; per layer, medians 1.3–8.0, R² falls
from ~0.25 (L1–16) to 0.06–0.12 (L18–23). Correction routed to
`status-1c.md` and `status-10.md` (math-10 §5.4's inversion reads "c > 1";
the exact bound is `c_min(β)`, 0.809 at β = 0.5 and 0.978 at β = 4:
`tools/math_checks/lemma51_c_bound.py`, 8/8). Which number to *adopt* is
Blocked 9, still the user's.

**Input of the results.** 8 v1 prompts × `pythia-410m` step143000 and step 0
× L0–23 × windows 1–3: the 16 run dirs `run_all.sh` lists (Phase 12
`2026-09-22_21-20-26`, `2026-09-22_21-41-17`, `2026-09-23_05-52-32`,
`2026-09-23_06-02-15`). Tables exclude `repeated_tokens` (skipped when
deduped), so a cell at L1–23 is 7 prompts × 23 layers = 161 records. "Tail" =
real's p ≤ 0.025 on the stronger side; every count reads **real /
calibration** (one null-A draw as data, same dedupe setting; #106's rule).
Headline graph: window 1, `mutual`; `coattn` and w2–3 agree in sign unless
noted.

**Results.**

| # | finding | numbers (step143000, w1 mutual; all tokens · deduped) |
|---|---|---|
| 1 | **No null is shown to be at level on real inputs.** A's calibration passes by construction (a null-A draw tested against null A); the step-0 control is the real test, and A fails it (rows 2, 7). B and C fire even on a null-A draw for Q, λ₂, k₄, local (cal 100–161 of 161), so against them only real-vs-calibration reads, and there the real run is *not* more extreme. For ARI, B's calibration is 1–13 of 161 at step143000 but 28 of 161 at step 0 | Q vs B: 127/161 · 117/161; vs C: 156/161 · 155/161 |
| 2 | **Modularity against A fires in both tails**, more often the lower: in more records real attention is *less* modular than on the matched Gaussian than more (medians 0.407 vs 0.421 · 0.360 vs 0.392). Step 0 also beats A's upper tail (34 vs 1), so A is not a clean null for Q either | Q vs A, upper / lower tail, real (cal): 26/62 (1/0) · 10/68 (1/2); step 0: 34/15 (1/3) |
| 3 | **Communities are contiguous stretches of text, and so are the null's.** Contiguity is the block's (causal mask, recency), not the tokens' | contig 0.967 vs 0.961 (11/3) · 0.948 vs 0.952 (2/2) |
| 4 | **Against A, agreement with the residual's k-means rises with depth, but A fires on position alone** (A draws each position independently, so its ARI is ≈ 0 by construction; untrained step 0 beats it in 59/161 records, 148/161 deduped). Row 4 against A is therefore not evidence of content | ARI centred vs A, L1–8 / L9–16 / L17–23: 0.006 / 0.026 / 0.089 (23/7, 41/3, 45/2) · 0.032 / 0.078 / 0.114 (42/1, 44/0, 43/2) |
| 5 | **Against B (position kept, content destroyed), a trained excess remains only at L17–23, about half the size**, and step 0 is not clean there either | step143000 vs B, L17–23: 0.089 vs 0.046, 21/49 (cal 1) · 0.114 vs 0.052, 16/49 (2). Step 0 vs B, L17–23: 13/49 (7) · 7/49 (3). L1–8 and L9–16: 10/56, 15/56 (cal 4, 2) |
| 6 | **The idealised cosine head (C-real: `softmax(β_h⟨u_i,u_j⟩)` on the real unit LN1 rows) reproduces little of the trained block's community structure and none of its locality** | Q 0.156 vs real 0.407; weight within 3 positions 0.037 vs 0.193; ARI(real, C-real) 0.22 · 0.27 |
| 7 | **Step 0: position.** Attention is near-uniform, so real ≈ C-real (ARI 0.94). Deduped, communities agree with k-means at ARI 0.29 vs A's 0, and B reproduces it (0.29): both follow position (the residual is a causal running mean). Deduped B shuffles diagonals of the *kept-token* index, not true offsets, so deduped B keeps recency only approximately | ARI centred deduped 0.295: vs A 148/2, vs B 19/13 |

*Revised after `/challenge-pr` on #108* (findings 1, 2, 5, 6 there; Claude
verified each on the reports). The first write-up read row 4 against A as an
independent check of #106; A (and #106's Gaussian, which shares the blind
spot) cannot tell position from content. What survives: against B, at
L17–23 only, trained attention groups tokens more like the residual's k-means
than position alone does (row 5, 21/49 vs cal 1, but step 0 13/49 vs 7): a
weak, late corroboration of #106's 2-means excess, not an independent
confirmation. Modularity gives no evidence either way (row 2), and the
particle picture's cosine head is a poor model of who attends to whom (row 6).

**β for Blocked 9 (recommendation; the decision is the user's).** Adopt
`beta_raw`: the slope of the softmax's own input on **unit LN1 rows**, i.e.
the β of `softmax(β⟨u_i,u_j⟩)` that the particle dynamics sees, with no
`attn_scale` division (that divides the model's `1/√d_h` out a second time;
"β, measured on the way" above). `docs/PHASE_SYNTHESIS.md` §3.2 leaned the
same way. Always report R² beside it, per band: the cosine kernel explains
about a fifth of the log-attention variance, a tenth late.

| step143000, 7 prompts × 16 heads, sink not a key | β median | IQR | R² median |
|---|---|---|---|
| L1–23, all tokens (n = 2576) | **3.88** | [1.92, 5.94] | 0.18 |
| L1–23, deduped | 4.67 | [2.63, 6.86] | 0.15 |
| L1–8 · L9–16 · L17–23, all tokens | 4.65 · 4.30 · 2.47 | | 0.22 · 0.21 · 0.10 |
| L0 | 0.96 | [−0.93, 2.90] | 0.30 |
| step 0, L1–23 | 0.004 | [−0.13, 0.14] | 0.007 |

So β ≈ 4 on 410m across prompts (the `wiki_paragraph`-only 4.00 / 4.44 above
holds), and step 0 has no β to speak of (R² ≈ 0). With β ≈ 4, C's
`δ = cβ^{-1/2}` is ≈ 0.5c, with `c_min(4) = 0.978`. **Caveat (#108 review,
finding 4):** the fit controls position offset only linearly, and the median
pools heads with negative slopes and near-zero R². Recency heads could inflate
the slope; a refit with per-offset terms (Parked below) should precede
adopting the number. The convention (`beta_raw`, not ÷ 8) does not depend on it.
*Done 2026-09-29: next paragraph.*

**β refit: per-offset fixed effects and an R² floor (2026-09-29).**
`core.beta_eff.estimate_beta_offset_fe` replaces the linear offset term with
one dummy per offset (`fe_full`), or per offset below W with a linear tail
(`fe_w4/16/64`); row fixed effects as before, row and offset effects
projected out by alternating projections. On synthetic recency heads whose
similarity falls with offset it returns the planted β exactly where the
linear control does not (`tests/test_beta_eff.py`). The floor tried was
`fe_full`'s partial R² (the share of what row and offset effects leave that
similarity explains); finding 2 says why it cannot pick the number. **Input:** #108's 16 run dirs (`run_all.sh` in
`data/p1d/attention_null_2026-09-26/`), each layer's own attention, sink
neither query nor key, `repeated_tokens` out; code
`p1d_cluster_ensemble/beta_refit.py`; output
`data/p1d/beta_refit_2026-09-29/` (`beta_refit.txt`, all bands, both token
sets). The `linear` variant reproduces #108's stored βs (6 112 heads, max
|Δ| 5e-6, 0 finite-on-one-side; the driver refuses otherwise). 6 min at 10 workers.
*Guards added after CodeRabbit on #110 (follow-up PR):* the estimator refuses
when similarity is collinear with the offset tail (column-normalised
singular-value ratio < 1e-6), and the reproduction gate refuses any fitted
head #108 did not record, or recorded twice. Re-run with both
(`beta_refit_2026-09-29/rerun_guards/`, commit `27e5c72`): summary
identical, 6 112 compared, 0 mismatched. The guard can only fire for
`fe_w4/16/64` (`fe_full` has no tail column); their smallest design ratio
over 11 488 head-fits is 0.467 (p1 0.536), so 1e-6 catches only degenerate
designs, and a refused head now shows as its variant's own count. The
page (link in STATE) now has a reading guide and carries the refit's β
beside #108's; after `/challenge-pr` on #111 it no longer calls null B a
clean judge or lists the Gaussian results as surviving position.

| step143000, L1–23, 7 prompts × 16 heads, all tokens | heads | linear | fe_w16 | fe_full |
|---|---|---|---|---|
| **all heads** | 2576 | 3.88 [1.92, 5.94] | 3.51 [1.58, 5.62] | **3.46 [1.55, 5.57]** |
| all heads, by band L1–8 · L9–16 · L17–23 | 896 · 896 · 784 | 4.65 · 4.30 · 2.47 | | 4.36 · 3.65 · 2.37 |
| deduped, all heads | 2576 | 4.67 | 4.24 | 4.15 [2.08, 6.35] |
| heads with ≤ 5 % of pairs log-clipped (1e-12) | 2300 | | | 3.57 |
| step 0, all heads | 2576 | 0.00 | 0.00 | 0.00 [−0.13, 0.14] |
| *truncated on partial R² ≥ 0.02 · 0.05 · 0.10 (not estimates: finding 2)* | 1738 · 1175 · 642 | | | *4.77 · 5.78 · 7.27* |

1. **The offset control moves the number by about a tenth.** Per head,
   `fe_full − linear` is −0.14 median [−0.46, −0.02]; most of it is taken by
   the first four offsets (fe_w4 3.61). It is concentrated at L9–16 (4.30 →
   3.65); late layers barely move (2.47 → 2.37). The recency worry was right
   in sign, small in size.
2. **A floor on partial R² is a floor on |β|, so it cannot pick the number.**
   With one regressor left after the fixed effects, a head's partial R² is
   `β̂² · Σs̃² / Σỹ²`, and `Σs̃²` is shared by a layer's 16 heads (same Gram,
   same pairs). Within a (prompt, layer), |β| and partial R² rank heads
   alike (Spearman median 0.92). The floored medians (4.77 → 5.78 → 7.27)
   are truncation of the β distribution, not a measurement of a subset.
   The first write-up read this arithmetic as "heads that run the kernel
   have steeper slopes" and "β is steadier across depth once floored";
   both are withdrawn. The review's precision-selected 1 175 heads give 2.87
   (reviewer's number, not re-run here).
3. **Step 0 does not calibrate a floor either.** Its largest partial R² is
   0.0483 (p99 0.0198), so "0.05, the smallest floor step 0 fails" was
   placed just above one head. It does show step 0 has no kernel: β 0.00,
   partial R² median 0.001.
4. **Log-clipping at 1e-12 is smaller than the offset effect.** 4.5 % of
   kept pairs are clipped (mean over heads); 276 of 2 576 heads have > 5 %,
   median β 2.39 vs 3.57 for the rest. Dropping them moves the median to
   3.57 (+3 %); heads with no clipping at all give 3.45.
5. **For C's δ = cβ^{-1/2}:** β = 3.46 gives δ ≈ 0.54c (#108's 3.88 gave
   0.51c). The per-band spread (2.37–4.36) moves δ by a factor 1.36 between
   early and late layers, more than the offset control does.

**Recommendation for Blocked 9 (the number; the decision is the user's):**
keep the convention (`beta_raw`, unit LN1 frame), fit with per-offset fixed
effects, and report the **all-head distribution per band**, headline
**β = 3.5 [1.6, 5.6]** (L1–23), not a floored subset. #108's 3.88 was 12 %
high from the offset control. A subset of "kernel heads" needs a criterion
that does not use β̂ (e.g. ablation: Parked below). *Revised after
`/challenge-pr` on #110:* the first version recommended 5.8 on heads with
partial R² ≥ 0.05 (findings 2–3 withdraw it).

**Defect fixed after the run.** `summarise` computed z whenever the null SD
was > 0, so constant draws with float-noise SDs (~1e-16) gave z ≈ 1e13
(`contig`, step 0, B/C). Now skipped below 1e-9 relative; test added. Tail
counts were never affected. **Page fixes:** no charset (mojibake) and the
attention grid overflowing to 8 800 px (`section>*{min-width:0}`).

**Parked** (discoveries, not followed):
- Token strings on the page are raw byte-level BPE (`Ġ`, `Ã«`). Cosmetic; ~10
  lines in `viz_page.token_file`; changes no decision.
- Row 6's gap could be the missing rotary / per-head `W_QK`, not the kernel's
  form. Cost: one more null (bilinear `W_QK` head, no rotary) on the same
  inputs, ~3 h. Could change whether the cosine head is the idealisation
  Phase 10's F13/F14 should be read against.
- ~~β refit with per-offset fixed effects and an R² floor.~~ Done 2026-09-29
  ("β refit" above).
- A β-free way to pick the heads that run the particle kernel (ablation, or
  each head's share of the update's norm), so a subset β can be read without
  truncating on β̂. Cost: one forward pass per ablation set. Could change
  whether the all-head β is the one C's δ should use.
- A null that keeps position and randomises content in the *residual*, not
  the attention (e.g. A's Gaussian given the real rows' positional mean, or
  a within-prompt position-block shuffle of rows). Cost: A's machinery, ~7 h
  per configuration. It is the judge row 5 needs, and #106's frames lack it
  too. Could change whether 1d counts attention as corroborating #106.

**How to re-run.** `run_all.sh` in the output directory (resumable: skips
finished configurations, reuses `<name>.parts/` when settings match), then
`attention_null_report.py --null <n>.json --calibrate <c>.json --out
report[_dedupe].json` per pair, then `viz_page.py --gauss
<main>/data/p1d/gaussian_null_2026-09-26 --attn <this dir> --out <dir>` and
copy `viz/index.html` beside it. Wall time at 14 workers: `null` 6.5 h,
`calibrate` 7.1 h (two sittings: 126 then 258 records), each deduped
configuration 1.4 h.

### Long prompts (DONE 2026-09-30; branch `claude/p1d-long-prompts`)

**What.** `lit-1d.md` §4's "length first": v1 prompts continued to Pythia's
context (2048), v1 text as an exact prefix, so under causal attention the
long run's first `n_v1` tokens reproduce the stored v1 run and length is the
only change (user, 2026-09-29: extend from source; Gaussian null, merge tree
and β on step143000 + step 0; runs on `/run/media/system/HDD_1TB/mets_data`).
Rule committed alone first (`ca787de`, `p1d_cluster_ensemble/long_prompts.py`
docstring), texts after, before any model run.

**Built (texts committed, no forward pass yet).** Long prompts hash
`91e85cc95888`; `long_prompts/provenance.json` has sources and counts.

| long key | v1 → long tokens | source |
|---|---|---|
| `wiki_paragraph_long` | 467 → 1840 | Wikipedia rev 1371774901 (next paragraph would pass 2048) |
| `sullivan_ballou_long` | 482 → 1032 | Wikisource rev 15675430; the letter ends |
| `hdbscan_code_long` | 242 → 2025 | hdbscan 0.8.41 `plots.py` |
| `latex_monograph_long` | 446 → 2036 | composed continuation, starts on a new line (tokenizer) |

Refused by the rule: `paper_excerpt` (v1 not verbatim in arXiv
2312.10794v5: math and citations removed by hand) and `repeated_tokens`
(v1's trailing lone space merges with any continuation). `homer_iliad` and
`camus_letranger` were dropped up front (copyright).

**Steps 1–4 DONE 2026-09-29; step 5 DONE 2026-09-30** (code on the branch;
outputs `<main>/data/p1d/long_prompts_2026-09-29/`, runs
`/run/media/system/HDD_1TB/mets_data/p1d_long/2026-09-29/`, ~0.9 GB each).

| step | state | result |
|---|---|---|
| 1 extractor | done | `extract_long.py`: `load_model`, tokenize **without truncation** (refuses a count ≠ `provenance.json`'s or > 2048), write through `p1_io`'s `_save_tokens/_geometry/_activations/_attentions` + `write_manifest` (`prompt_key` = long key, `long_prompts_hash`; `geometry.json` `layers` empty) |
| 2 runs | done | 8 runs (4 prompts × step143000, step 0), 26–112 s each, peak RSS 15.7 GB. First run checked populated before the rest: unit rows, finite, attention rows sum to 1 (max err 1.8e-6), causal half zero |
| 3 prefix | done | `prefix_check.json`. 6 of 8 **bit-identical** to the stored v1 run (activations, norms, attention; tokens equal; no prefix row attends past `n_v1`). `hdbscan_code` differs at float noise: step143000 act 6.3e-5, attention 3.7e-3; step 0 1e-7. Embedding identical; the difference starts at 1e-7 in L1 from token 121 and grows with depth: reduction order, not the prefix |
| 4 holdout | done | `refuse_held_out(..., drop=True)` keeps all 8; `run_1d.discover_runs` finds 8; test added |
| 5 merge tree + link null | done | below |
| 5 Gaussian null | done (deduped 2026-09-29, all tokens 2026-09-30) | below |
| 5 β | done 2026-09-30 | below; estimator sped up first |

**Merge tree (`--subexp F`) and its link null** (2000 draws, per group of
the same 4 prompts; `merge_null/{long,v1}_step{143000,0}.{json,txt}`; v1
step-0 trees were not stored before, so built here: `merge_tree_v1_step0/`).
Structure counts, summed over the 4 prompts:

| group | blob layers (L1–24) | two-cluster picks (L3–24) | token 0 out (L1–24) |
|---|---|---|---|
| v1 step143000 | 69/96 | 51/88 | 71/96 |
| long step143000 | **43/96** | 52/88 | 61/96 |
| v1 step 0 | 12/96 | 0/88 | 91/96 |
| long step 0 | 5/96 | 8/88 | 83/96 |

- **Length halves the blob at step143000** (69 → 43 of 96; per prompt
  18→10, 20→12, 18→16, 13→5): the longest-lived scale is less often one
  cluster plus outliers. The ≥ 2-cluster pick is still exactly 2 about as
  often (51 vs 52 of 88), so the scale question is unchanged.
- **Links against the null, same reading as v1's:** no merge excess
  (L0–L7 merge fraction 0.74 vs null 0.93, *below*; v1 same 4 prompts
  0.61 vs 0.67); late tangle share far below the null (L8–15 0.45, L16–23
  0.87 vs ≥ 0.99; v1 0.36, 0.48). Late layers tangle more at length (26 of
  30 L16–23 events vs 14 of 29).
- Step 0: almost every link is stable (2782 of 2785 at L0–L7): its
  partitions are fine-grained and repeat layer to layer.

**Gaussian null, deduped: DONE 2026-09-29** (`gaussian_null/{null,calibrate,report}_dedupe.*`;
200 / 100 draws, seed 0, all 3 frames, 25 layers, 8 runs, 339–699 tokens kept; v1 on the
same 4 prompts: `gaussian_null/v1_same4/report_dedupe.*`). Centred frame, records in the
lumpier tail / the same on its calibration, of 32 per band:

| stat | step | band | v1 | long |
|---|---|---|---|---|
| ci2 (2-means) | 143000 | L1–8 | 0 / 0 | **12 / 0** |
| ci2 | 143000 | L9–16 | 5 / 0 | **14 / 0** |
| ci2 | 143000 | L17–24 | 20 / 4 | 31 / **24** |
| hdb_k (HDBSCAN groups) | 143000 | L1–8 · L9–16 · L17–24 | 16 · 25 · 29 / 0 | **31 · 28 · 30** / 0 |
| nn1 (nearest neighbour) | 143000 | L1–8 · L9–16 · L17–24 | 32 · 32 · 32 / 0 | 32 · 32 · 32 / 0 · 0 · **15** |
| mt_life | 0 | L9–16 · L17–24 | 20 · 25 / 0 | 26 · 29 / 0 |
| ci2, hdb_k, nn1 | 0 | all | ≤ 1 / ≤ 1 | ≤ 1 / ≤ 1, except hdb_k L17–24 8 / 1 |

- **The 2-means excess at L1–16 is mostly power** (*revised after `/challenge-pr` on
  #118*): the counts rise (12 and 14 of 32, calibration 0; v1 0 and 5), but the effect,
  median per record of obs − null mean, is unchanged at L9–16 (−0.0035 v1, −0.0038
  long) while z doubles (−0.97 → −1.96) on 339–699 kept tokens. Only L1–8 grows
  (−0.0007 → −0.0028, ~4×). HDBSCAN groups: effect 13–16 → 30–52 groups (group counts
  scale with n too). A count of records past a tail is not an effect size. At L17–24 **the calibration itself fires**
  (24 of 32), so this null is off nominal there at this length, and v1's late 2-means
  excess cannot be read on long prompts with it.
- **More HDBSCAN groups than the Gaussian at every depth** (28–31 of 32, cal 0); v1 had
  16 at L1–8. Step 0 stays at its calibration (except hdb_k L17–24, 8 vs 1).
- Closer nearest neighbours: all records, both lengths; calibration clean except long
  L17–24 (15). A nearest neighbour within 3 positions is *less* common at length
  (median 0.23 vs 0.31, step143000 centred): more tokens to choose from.
- Step 0's merge-tree lifetime excess (Parked 11) persists and grows (26, 29 of 32).
- Tier 1, one checkpoint pair, 4 prompts.

**Gaussian null, all tokens: DONE 2026-09-30** (`gaussian_null/{null,calibrate,report}.*`;
200 / 100 draws, seed 0, 600 records each, 0 skipped, 1032–2036 tokens; v1 on the same 4
prompts: `gaussian_null/v1_same4/report.*`). Centred frame, lumpier tail / calibration,
of 32 per band (L1–8 · L9–16 · L17–24):

| stat | step | v1 | long |
|---|---|---|---|
| ci2 | 143000 | 26 · 25 · 29 / 3 · 2 · 18 | 32 · 32 · 32 / **20 · 12 · 31** |
| ci2 | 0 | 32 · 32 · 32 / 6 · 4 · 0 | 32 · 32 · 32 / **32 · 26 · 25** |
| nn1 | 143000 | 32 · 32 · 32 / 0 · 0 · 14 | 32 · 32 · 32 / **8 · 10 · 29** |
| nn1 | 0 | 32 · 32 · 32 / 0 | 32 · 32 · 32 / 8 · 9 · 12 |
| hdb_k (median groups, null ≈ 4) | 143000 | 57 · 59 · 47, all 32 / 0 | **234 · 243 · 204**, all 32 / 0 · 0 · 1 |
| hdb_k | 0 | 63 · 62 · 59, all 32 / ≤ 3 | **288 · 280 · 282**, all 32 / 0 |
| mt_k (median, null ≈ 2) | 0 | 21, 31–32 / 0 | 92, all 32 / 0 |
| mt_life | 143000 | 2 · 4 · 11 / 2 · 0 · 6 | 10 · 8 · 11 / 3 · 4 · 12 |

- **On all tokens at length the null is off nominal for ci2 and nn1:** the calibration
  (each token replaced by one draw of its own Gaussian) lands in the lumpier tail in 12–32
  of 32 records per band. v1's all-token calibration fired only at L17–24 (ci2 18, nn1 14);
  the deduped long run only at L17–24 (ci2 24, nn1 15). The miscalibration grows with n.
  So neither statistic can be read on all tokens at 2048 (Parked below).
- **What stays readable is token identity.** HDBSCAN groups (cal 0–1) and the merge
  tree's k: step 0 beats the null *more* than step143000 at every depth (288 vs 234 groups
  at L1–8; mt_k 92 vs 2–3), as #106 found at v1 length. At length a token's nearest
  neighbour is more often the same string (median 0.68 vs 0.55 at step143000, 0.79 vs 0.67
  at step 0; L1–24 centred), so the identity channel grows with the prompt.
- Trained-only signals are marginal: mt_life 8–11 of 32 vs cal 3–12; mt_k 4–6 vs ≤ 1.
- Nearest neighbour within 3 positions (step143000, centred, L1–24 median): 0.11 long vs
  0.23 v1 (deduped: 0.23 vs 0.31).
- **Reading:** length does not change #106's all-token verdict (identity), and at 2048
  the deduped table above is the one to read.

**Defects found and fixed on the way** (both would have made step 5 take
days, or lose work):
1. **β's estimator did not scale to 2048 tokens.** `_within_row_demean`
   looped over rows with a mask over all pairs (O(rows × pairs); 965 s for
   one layer's linear fit at n = 2025), and `_two_way_demean`'s alternating
   projections converge slowly on a causal design (378 s per layer, near its
   iteration cap). Now: one `bincount` pass, and an exact two-way solve
   (row effects in closed form, offset effects by Cholesky on the Schur
   complement, one bin pinned). **Identical βs:** on #110's stored heads
   (2 runs, both steps, all tokens and deduped, 1536 heads × 5 variants)
   max |Δβ| ≤ 1.2e-14. ~2 s (linear) / ~3 s (fe_full) per head at n = 2025.
   Tests: exact = dummy OLS on a random and a causal design.
2. **`gaussian_null` and `beta_refit` were all-or-nothing** (`pool.map`,
   `ex.map`), though `attention_null` had been fixed for this on 2026-09-26
   (`LESSONS.md` 11). A shutdown on 2026-09-29 cost ~5 h of the Gaussian
   null's all-token stage (~290 of 600 records, from bytes read) and ~4 h
   of β. Both now write a part per record / per (run, dedupe) job and
   resume (`<out>.parts/`; draws are seeded per record, so a resumed run
   gives the same numbers). *Corrected after `/challenge-pr` on #118:* this
   line said parts are reused only with matching settings; `beta_refit`
   reused any part. Now both check settings and the run's input files (size,
   mtime); `beta_refit` also the weights, `max_offset` and `core/beta_eff.py`'s
   hash; pre-settings parts are refitted (tests in `test_phase1d_beta_refit.py`).
3. `beta_refit --stored` may be omitted only when every run is a long prompt
   (enforced since #118's review; it was optional for any input); the output
   records `reproduction: {"not_run": ...}`.
4. Exact solver checked at length (#118 finding 5): against the iterative
   `_two_way_demean` on `hdbscan_code_long` step143000 (n = 2025), L20 h0, h7
   and L10 h3, windows 4 and full: max |Δβ| 4.4e-15
   (`beta/solver_check.json`).

**Cost, measured.** Gaussian null at n ≈ 1800: ~3 s per draw per record
under full load (k-means `n_init=10` ~40 %, cosine distances, HDBSCAN,
linkage, the draw), + ~5 s per record for the frame's decomposition. The
box is 8 cores / 16 threads, so 15 workers is the ceiling. All-token null
(600 records × 201 fits) ≈ 9 h, its calibration ≈ 4.5 h; deduped prompts
are 339–699 tokens (`hdbscan_code`, `sullivan_ballou`, `latex_monograph`,
`wiki_paragraph`: 339 / 418 / 547 / 699), so each deduped stage is roughly
a tenth. β forecast 4–5 h (16 jobs at 3 workers, ~7.5 GB RAM per worker).
**Measured:** `null` 20:45 → 12:17 less the 8 h suspend (~7.5 h),
`calibrate` 4.1 h, β **1.0 h** (3667 s; the forecast was one head timed on
a loaded box).

**Re-run** (resumable; `LESSONS.md` 11): `setsid nohup systemd-inhibit
--what=sleep:idle <main>/data/p1d/long_prompts_2026-09-29/run_chunk.sh
>> <same>/run_chunk.log 2>&1 &`. It skips finished stages (`<stage>.json`
present), resumes a stopped one from `<stage>.parts/`, and runs in order
`null_dedupe`, `calibrate_dedupe`, `null`, `calibrate`, then β. The box
suspended 2026-09-29 21:20–05:16 (`null` at 45/600) because nothing held
it awake. Reports: `python -m p1d_cluster_ensemble.gaussian_null_report
--null <O>/gaussian_null/null{,_dedupe}.json --calibrate
<O>/gaussian_null/calibrate{,_dedupe}.json --out <O>/gaussian_null/report{,_dedupe}.json`;
`python -m p1d_cluster_ensemble.beta_long_compare --long <O>/beta/beta_refit.json
--v1 <main>/data/p1d/beta_refit_2026-09-29/beta_refit.json --out <O>/beta/vs_v1.json`.

**β at length: DONE 2026-09-30** (`beta/beta_refit.{json,txt}`, `beta/vs_v1.{json,txt}`;
all 5 variants, 16 jobs, 6144 head-fits; no `--stored` reproduction, since #108 never fitted
these inputs; the estimator reproduced #108 on v1 and matched #110 to 1e-14 after the
speed-up). Against #110's heads for the **same 4 prompts** (not #110's 7-prompt 3.46),
`fe_full`, step143000, floor 0, 1472 heads (L1–23) or 512 / 512 / 448 per band; paired =
long − v1 on the same head:

| tokens | band | v1 median [IQR] | long | paired median [IQR] | heads up |
|---|---|---|---|---|---|
| all | L1–23 | 3.16 [1.36, 5.36] | **2.80** [0.78, 5.56] | −0.24 [−1.22, 0.64] | 617 / 1472 |
| all | L1–8 | 3.89 | 4.12 | −0.04 | 247 / 512 |
| all | L9–16 | 3.55 | 3.25 | −0.23 | 222 / 512 |
| all | L17–23 | 2.19 | **1.35** | −0.49 | 148 / 448 |
| deduped | L1–23 | 3.91 | 3.93 | −0.12 | 682 / 1472 |
| deduped | L9–16 | 4.70 | 5.51 | +0.47 | 336 / 512 |
| deduped | L17–23 | 2.58 | **1.36** | −0.77 | 119 / 448 |

Step 0: 0.01 on both sides, paired 0.00, half the heads up: the estimator does not drift with n.
`linear` moves the same way (all tokens L1–23: 3.59 → 3.09).

Paired median per prompt (`fe_full`, step143000, all tokens; L1–8 · L9–16 · L17–23):
`hdbscan_code` −1.52 · −2.03 · −0.48; `latex_monograph` **+1.95 · +2.43** · −0.06;
`sullivan_ballou` −0.01 · +0.05 · −0.37; `wiki_paragraph` −0.21 · −0.72 · −0.95.

**Most of the late fall is the offset mix, not length** (*revised after `/challenge-pr`
on #118*, finding 1). At full length 28–78 % of fitted pairs sit at offsets v1 never had,
and at L17–23 similarity explains ~2 % of what the fixed effects leave, so the slope moves
with the pair mix. Refit on the long runs with only the offsets their v1 prefix contains
(`beta_refit --max-offset-v1`, offset ≤ n_v1 − 2; `beta_offset_matched/{beta_refit,vs_v1}.*`,
16 jobs, 1 h):

| tokens | band | v1 | long, all offsets: paired, up | long, v1's offsets: paired, up |
|---|---|---|---|---|
| all | L1–23 | 3.16 | −0.24, 617 / 1472 | **−0.05**, 716 / 1472 (median 3.12) |
| all | L17–23 | 2.19 | −0.49, 148 / 448 | **−0.17**, 201 / 448 (median 1.88) |
| deduped | L9–16 | 4.70 | +0.47, 336 / 512 | +0.77, 385 / 512 |
| deduped | L17–23 | 2.58 | −0.77, 119 / 448 | −0.29, 183 / 448 |

Per prompt, v1's offsets, all tokens, L17–23: `hdbscan_code` +0.04, `latex_monograph` +0.23,
`sullivan_ballou` −0.19, `wiki_paragraph` −0.55. Step 0 stays at 0 (paired −0.01).

- **L17–23:** about two thirds of the paired fall (−0.49 → −0.17) is pairs at offsets v1
  did not have. On v1's offsets it no longer falls in every prompt; the rest is mostly
  `wiki_paragraph`. The first write-up's "falls ~40 % in every prompt" compared two medians
  (the paired change was −22 %) and read the offset mix as length: withdrawn.
- **At L1–16 the shift is the prompt's, not length's**, with or without the offset match
  (`latex_monograph` +2 to +3, `hdbscan_code` −1.5 to −2 all tokens; the pooled median
  cancels). `latex_monograph`'s continuation is composed, not a source text, and
  `hdbscan_code`'s is 1800 tokens of code: content and length are confounded per prompt.
- Step 0 staying flat does not clear the offset question (a zero slope stays zero on any
  pair set); it only shows the estimator has no drift with n.
- **For Blocked 9 (β's convention):** on v1's offsets the headline hardly moves with length
  (3.16 → 3.12); on all offsets it is 2.80. So β should be quoted with the **offset range**
  it was fitted on; prompt length matters through that range. A length curve (truncate
  each long run at several n, no forward pass needed) is Parked below.
- Tier 1, 4 prompts, one checkpoint pair; heads within a prompt are not independent,
  so the "heads up" counts are descriptive, not a test.

**Parked** (discoveries, not followed):
- **The 512-token cap.** `core/models.py` `extract_activations` (and 9 other
  call sites) tokenizes with `truncation=True, max_length=512`, silently.
  `homer_iliad` is 562 tokens, so every stored `homer_iliad` run is its first
  512 tokens (consistent across checkpoints; no doc said so). v2's held-out
  `scipy_linkage_code` (527) and `latex_article` (614) would be cut too. Cost:
  a note per affected phase, and a loud record (or refusal) of truncation in
  the extractor. Could change any claim quoting `homer_iliad`'s length.
- **The Gaussian null's calibration drifts with n.** One Gaussian draw per
  token, re-tested against a Gaussian refitted to it, lands in the lumpier
  ci2 / nn1 tail in 12–32 of 32 records at 1032–2036 tokens (all tokens), vs
  0–18 at 242–482 (v1). Why: untested (a guess is that re-estimating the
  covariance from the calibration sample shifts the spectrum at this n/d;
  d = 1024). Cost: a synthetic check across n on known Gaussians, ~1 h. Could
  change: whether this null is usable for any all-token claim past ~500
  tokens, and how far the deduped L17–24 calibration (24 of 32) is the same
  effect.
- **β as a function of length** (#118 finding 1's second half). Under causal
  attention a long run's first n tokens are the n-token run, so fitting each
  long run truncated at several n (e.g. 256, 512, 1024, full), on all offsets
  and on a fixed offset range, gives a length curve with no forward pass.
  Cost: `beta_refit` with a row cap, ~1–2 h compute. Could change: whether
  β's convention (Blocked 9) names a length, an offset range, or both.
- **Deduped Gaussian null at v1's token count** (#118 finding 2). Subsample
  each long deduped run to its v1 run's kept count and re-run the null, so
  counts compare at equal power. Cost: a `--subsample` option and ~1 h. Could
  change: whether L1–8's ~4× larger 2-means effect is length or sample size.

### Vote rules: can a weighting make the grading usable? (DONE 2026-09-30; branch `claude/p1d-vote-rules`)

**Question.** The weighting decision A0 left open (`/challenge-pr` on #100,
finding 1): core / halo / contested is not a usable readout until it is
decided (a) what a token outside a family's substantial structure does on a
pair (now: HDBSCAN's -1 abstains, another family's singleton votes "apart";
`refusal_fraction` already calls them the same refusal) and (b) how votes
are weighted (now: raw stability, highest at k = 2 and at fine scales).

**Rules compared**, all on the same tuned labels and null draws
(`p1d_cluster_ensemble/vote_rules.py` docstring): noise `current` /
`abstain_small` (every cluster < 4 tokens abstains) / `singleton` (-1 votes
"apart") × weights `stability` / `uniform` / `kappa` (stability above its own
null, Cohen's form). Thresholds recomputed under each rule. **Reading fixed
before the batch** (docstring): primary = single-family sway (drop one
family, nulls too; worst case over families of the core-set Jaccard, plus,
added after the one-record smoke, the worst Spearman of per-token
confidence); guard = dominance (consensus ARI to one family ≥ 0.95).

**Input.** 8 v1 prompts × step143000 / step 0 (410m; the 16 runs in
`data/p1d/gaussian_null_2026-09-26/null.json` `inputs`) × L6 / 12 / 18 = 48
layer-records; quick grid, seed 0, 5 repeats, 20 gate nulls, 10 confidence
nulls; code `2a176a4` (24 records from the first batch, which died there,
and 3 from the diagnostic re-run, the same code under a wrapper that only
acts on failure) and `631473c` (21; differs only on an invalid tree, which
would have crashed). Parts are reused on matching settings and inputs, not
on the code version. The reference rule reproduces A0's
stored `wiki_paragraph` L12 exactly (core 257 / halo 88 / contested 122).
Output `<main>/data/p1d/vote_rules_2026-09-30/vote_rules.{json,txt}`.

**Result: no rule passes.** Medians over 24 records per step:

| step143000 | worst core J | veto (J < 0.5) | worst conf. ρ | dom. ≥ 0.95 | empty core | core share | k |
|---|---|---|---|---|---|---|---|
| current · stability (run_1d's) | 0.00 | 23 | 0.12 | 9 | 5 | 0.38 | 4 |
| current · uniform | 0.10 | 23 | 0.08 | 7 | 6 | 0.41 | 6 |
| current · kappa | 0.04 | 23 | −0.01 | 11 | 4 | 0.48 | 4 |
| abstain_small · stability | 0.00 | 22 | 0.34 | 14 | 13 | 0.00 | 3 |
| abstain_small · uniform | 0.00 | 22 | 0.22 | 13 | 11 | 0.02 | 3 |
| abstain_small · kappa | 0.00 | 23 | 0.46 | 14 | 12 | 0.00 | 3 |
| singleton · stability | 0.05 | 23 | 0.14 | 7 | 5 | 0.52 | 6 |
| singleton · uniform | 0.20 | 22 | 0.32 | 4 | 3 | 0.47 | 10 |
| singleton · kappa | 0.21 | 21 | 0.28 | 8 | 5 | 0.62 | 7 |

- **Under every rule, one family's removal replaces the core set** in 21–23 of
  24 trained records, and the per-token confidence ranking after the worst
  drop agrees with the full one at ρ ≤ 0.46 (median). The same holds in each
  band (L6 / 12 / 18 separately: J ≤ 0.47, ρ ≤ 0.38 for the three rules
  broken out). Step 0 is no better (14–21 vetoes, ρ 0.28–0.55).
- **No one family is the culprit:** the worst drop is k-means, HDBSCAN or
  agglomerative about equally often (53 / 49 / 47 of 216 rule-records).
- **`abstain_small`, the design's own reading of a refusal, turns the
  consensus into k-means** (dominance ≥ 0.95 in 13–14 of 24) and empties the
  trained core in 11–13 of 24, while **step 0's core share is 0.95–1.00**:
  under it the untrained model is graded far more clustered than the trained
  one. *Revised after `/challenge-pr` on #120:* that contrast is the null
  threshold, not the vote: its median core threshold is 0.32 at step 0 and
  0.83 trained (the first write-up guessed token identity).
- What the rules do change: the consensus k (3 to 10) and the core share
  (0 to 0.62). A choice among them moves the readout without stabilising it.

**Reading.** No weighting or noise rule tried makes the grading usable.
*Revised after `/challenge-pr` on #120:* why is a hypothesis, not a
finding. The first write-up said the cause is the scale mix (a k = 2 vote
beside k ≈ 100–280 votes; `lit-1d.md` §1, option D). In a planted-caps toy
the mix is enough to cause sway and matched scale passes
(`tests/test_phase1d_vote_rules.py`), but 41 of 48 records mix k ≤ 4 with
k ≥ 50, so this batch cannot separate the two. The one near-matched slice
(step 0 `repeated_tokens`, every family at k 2–5) still sways: worst ρ
−0.18 / 0.69 / 0.53 at L6 / 12 / 18, consensus ARI 0.03–0.30, though that
prompt has 3 distinct strings. **For the user:** retire core / halo /
contested as 1d's product (P-C4, unregistered, would score noise), or first
build a matched-scale vote and ask again, with no evidence yet that it
would pass on Pythia. Claude recommends retiring it; the per-family gate,
the merge tree and the nulls are the parts that have held.

**Fixed on the way (a defect).** scipy 1.15's average linkage returned an
invalid tree (a node merged with itself) on a tied co-association (446
tokens, 27 row types, 10 values; a null draw under a leave-one-out rule).
`consensus_partition` did not check, `fcluster` raised, and the batch died
24 of 48 in. It now rebuilds an invalid tree on distances rounded to 12
decimals, and refuses if that is invalid too; a valid tree is untouched
(`ensemble.py`, regression case inline in `tests/test_phase1d_ensemble.py`;
`LESSONS.md` 2). The failure depends on token order (0 of 20 permutations
reproduce it); on tied matrices the consensus is not unique either (other
orders give ARI 0.97–1.0 at the same objective; `/challenge-pr` on #120).

**Re-run** (~25 min at 14 workers, resumable):

    OMP_NUM_THREADS=1 python -m p1d_cluster_ensemble.vote_rules --v1-only \
      --workers 14 --runs $RUNS --out <main>/data/p1d/vote_rules_2026-09-30/vote_rules.json

**Parked** (discoveries, not followed):
- **The grading's null.** Thresholds come from the shuffled-dimension null,
  which #106 and #108 showed is beaten by position and token identity alone.
  Cost: swap in the Gaussian null per draw, ~1 h. Could change: only matters
  if the grading is kept.
- **`consensus_order` (visualisation) on an invalid tree** returns an
  unsorted heatmap without saying so (`cluster_methods.py`). Cost: minutes.
  Could change: no number, only a figure's ordering.

### Design revised: the grading retired, a per-group definition proposed (2026-09-30; branch `claude/p1d-design`)

**Decision (user, 2026-09-30, Blocked 10):** core / halo / contested is retired
as 1d's product, with the consensus partition as a definition and `P-C1`,
`P-C3`, `P-C4` (`predictions-1d.md` addendum). The "Grading's null" item
Parked above is moot. Docs only; no code or data changed.

**`design-1d.md` rewritten.** It proposes the definition 1d tests next: an
HDBSCAN group (`min_cluster_size` 2, float64 cosine, centred frame, deduped
tokens) is a cluster when its excess density `S_C / |C|` exceeds the 95th
percentile of the per-draw *maximum* under the matched-covariance Gaussian,
read against its `--calibrate` run, with step 0 as the control, at v1 length.
It starts from `hdb_k` because that readout passed step 0; the merge tree's
lifetime did not (Parked 11). *Revised after `/challenge-pr` on #121:* the
first version used hdbscan's `cluster_persistence_`, which divides by the
largest λ in the whole tree, so one tight group elsewhere scores every other
group down (verified: a planted 6-point group 0.75 alone, 0.012 beside a tighter
one). Also added: a label-release rule per band, `repeated_tokens` dropping
out under dedup (7 prompts), and the centred frame marked as chosen after
seeing the data. Checks (position, cluster-wise stability,
family recovery, cross-layer links, attention, theory scale) are reported
beside each group, not voted. Literature: `lit-1d.md` §8 (3 searches;
SHC's top-down stop and Gao–Bien–Witten's isotropic test rejected there).

**Next:** build step 1, `admit.py`: done, "Admission" below.

**Parked** (discoveries, not followed):
- **Cleanup of the retired columns.** `run_1d.py`, `p1d_io.py`,
  `core/artifacts.py` and `tools/run/p1d_drift.py` still read or write
  `confidence` / core / halo / contested. Cost: an hour with tests. Could
  change: nothing measured; it stops a later reader taking them as live.

### Admission: build step 1 (2026-10-01; branch `claude/p1d-admit`)

**What.** `admit.py`, `design-1d.md` build step 1: per layer, HDBSCAN groups
of deduped tokens, each scored by `S_C / |C|` (and `log_life`) against the
per-draw maximum under the matched-covariance Gaussian (`gaussian_null.py`'s
frames, draws and `--calibrate`, unchanged; same seed, same draws). α = 0.05
per record, 200 draws, `min_cluster_size` 2 and 4, centred and raw.

**A defect in the shipped call, found by the invariance check.** The design's
synthetic check (a looser planted group must keep its statistic when an
unrelated tighter group is added) failed on hdbscan's condensed tree:
`S_C / |C|` moved 29.59 → 30.66, the same 8 members. The cause is ties:
hdbscan's core distance at `min_samples` 2 is the *second* other neighbour
(checked against its single-linkage tree), so a point's core distance is the
weight of several of its mutual-reachability edges; 13 of 115 MST edges tied
in the test input. hdbscan's binary tree orders a tie by processing order,
which other rows change: once it made a 3 + 2 "split" at one λ, once the same
points fell out one by one. It also glued a stray background row to the tight
planted cap (9 members), a set that is never a component of the graph at any
distance. **Fix:** HDBSCAN on the level-set tree (every edge of one weight
merged at once; `level_set_hdbscan`). Handed hdbscan's *own* binary tree,
the same condensing + EOM code reproduces hdbscan's labels and
`cluster_persistence_` × max λ in 400 of 400 fits (1 257 groups), so tie
merging is the only difference. Groups are now level-set ones; the shipped
call's labels, tie artefacts (`shipped_check`: groups that are never a
component, brute-force checked) and ARI to ours are written beside each
record. `design-1d.md`'s algorithm row is changed and marked.

**Synthetic** (`tests/test_phase1d_admit.py`, 17 tests): planted caps admitted
on both arms, background not; invariance holds to 1e-12 with the verdict
unchanged, while hdbscan's own persistence for the same group falls 0.785 →
0.040; pure anisotropic Gaussian, null refitted (plug-in), 100 records,
B = 39, n 60 and 150: `excess` admits in 0–3 % of records, `log_life` 0–6 %
(one cell 6/100, `min_cluster_size` 4, n 150). First real record
(`wiki_paragraph`, step143000, L12, centred) opened before the batch: shipped
k 30 and null mean 7.425 equal `null_dedupe.json`'s `hdb_k` for that record;
20 level-set groups, 12 admitted, sizes 4–14, not pairs.

**Input.** 7 v1 prompts (`repeated_tokens` has 3 strings), pythia-410m,
step143000 (Stage 0, `2026-09-22_21-20-26`, `2026-09-22_21-41-17`) and step 0
(`2026-09-23_05-52-32`, `2026-09-23_06-02-15`): the 14 directories in
`real.json` `inputs`; deduped (125–273 tokens); L1–24; conda `mets`, hdbscan
0.8.41; code on `claude/p1d-admit` from `c3fdd2e`. Calibration: same, each
layer one Gaussian draw of itself, 200 draws. **Output:**
`data/p1d/admit_2026-10-01/`: `real.json`, `calibrate.json` (672 records each,
none skipped), `report.{json,txt}`, `report_labels.json`, `run_all.sh`.
**Re-run** (~6 + 6 min at 14 workers, resumable): `run_all.sh` there
(`admit run --v1-only --n-draws 200 --workers 14 [--calibrate]`, then
`admit report`).

**Result: the step-0 control fails, so nothing trained is read** (the
design's stop rule). `excess`, `min_cluster_size` 2, records with ≥ 1
admitted group, of 56 per cell (full table, both statistics and arms:
`report.txt`):

| step | frame | L1–8 | L9–16 | L17–24 | calibration (same order) |
|---|---|---|---|---|---|
| 0 | centred | 43 | 54 | 56 | 0 · 0 · 0 |
| 0 | raw | 13 | 24 | 25 | 0 · 0 · 0 |
| 143000 | centred | 52 | 53 | 45 | 0 · 3 · 11 |
| 143000 | raw | 53 | 51 | 54 | 8 · 5 · 13 |

`min_cluster_size` 4 and `log_life` show the same step-0 failure (centred
39–56 of 56 per band, calibration 0–8).

**Why step 0 fails: groups anchored at the prompt's opening.** Every step-0
admitted group in the centred frame (153 of 153, `min_cluster_size` 2)
contains position 0 and is mostly early tokens, but only 4 of 153 are an exact
run of the first kept tokens (`def get _ plot data ( self ,` …): the median
group has 94 % of its members in the first quarter of the kept tokens, and 64
of 153 have a member past the halfway point; 4–50 members. Raw: 44 of 65
contain position 0. Those late members are the cheapest clue to what the
group is (*revised after `/challenge-pr` on #122*: the first write-up called
it the "opening stretch"). Post hoc, not a result: records admitting a group
*without* position 0 are 0 of 56 per band at step 0 centred (raw 5–10), and
45–53 (raw 48–54) at step143000, whose admitted groups contain position 0
in 21 of 990 centred (14 of 715 raw) and look like content (first names;
two-digit numbers; `a few days`). Why the untrained model's opening tokens
form a group is not checked.

**Step 0 was never at its Gaussian on `hdb_k`** (`/challenge-pr` on #122,
verified on `gaussian_null_2026-09-26/null_dedupe.json`, centred, 7 prompts,
L1–24): step 0's shipped group count is above its null mean in 160 of 168
records, against 85 of 168 in its calibration. It "passed" (2 records in the
2.5 % tail) only because its null is about twice as wide as step143000's
(median sd 8.6 vs 4.1). So admission's step-0 failure is not new, and the
premise in `design-1d.md` ("`hdb_k` passed step 0") is weaker than written
(row marked there). If step 0 is lumpier than its Gaussian beyond the
opening too, the opening alone cannot settle Blocked 11.

**The shipped call** (`min_cluster_size` 2, real records): 42 % of
step143000's groups (1 873 of 4 502) and 53 % of step 0's are tie artefacts,
about two thirds of them pairs; median ARI to the level-set labels 0.82,
p10 −0.06. At `min_cluster_size` 4, 37–47 %, median ARI 0.97–0.98. Routed to
`status-10.md`: about two in five of the groups Phase 10 reads exist only
through hdbscan's tie order.

**Labels.** *Fixed after `/challenge-pr` on #122:* the first report released
by the calibration rule only (1 064 of 1 344 label records, step 0's
included). `table` now also withholds a band when the step-0 control's same
band admits in more than 10 % of its records, never releases the control,
and releases nothing from a file without it; it also refuses a calibration
with different draws or records. Re-run: **0 of 1 344 released.** The 10 %
bound treats a band's 56 records (7 prompts × 8 adjacent layers) as
independent, which they are not; release turns on 3 vs 6 records.

**Next (the user's call, Blocked 11; the user chose (b), done 2026-10-01: "Position" below):** what to do about position before any
trained reading. (a) The position-keeping residual null (Parked under
"Attention communities", ~7 h): the design's pre-written consequence for
positional groups, and it would also test the trained groups. (b) Cheaper
first: build step 2's position check on these groups, plus why step 0's
groups form (their late members; the first positions' residuals at init).
Claude's recommendation: (b) first, under an hour, *but* with step 0 above
its Gaussian on `hdb_k` in 160 of 168 records, (b) may show that step 0's
excess is not only the opening, in which case (a) is needed.

**Parked** (discoveries, not followed):
- **Phase 1's stored labels and every `hdb_k` count carry tie artefacts.**
  `gaussian_null`'s `hdb_k` excess (20 vs 6 groups) counts shipped groups,
  ~40 % of them artefacts on both sides. Why: the excess 1d started from may
  shrink or grow. Cost: a re-run of `gaussian_null` with level-set counts,
  ~10 min. Could change: the design's starting premise.

### Position: step 0's failure is the prompt's opening (Blocked 11 option (b); 2026-10-01; branch `claude/p1d-position`)

**What.** The user's option (b) for Blocked 11: build step 2's position check
(`position_check.py`, `design-1d.md` "Checks", position row) and find what
step 0's groups are. Two instruments. (1) Per group, `span`, `near_share`
(member pairs within 3 positions), `contiguous`, `has_first`, each against
2 000 random same-size groups drawn from the *same record's kept positions*;
*positional* = `p_near` ≤ 0.05 (descriptive, ~5 % by chance). (2) A
diagnostic arm in admission, `admit run --min-position M`: only kept tokens
at absolute position ≥ M are tested (dedup unchanged; nulls and calibration
refitted on the same subset). M = 8, then M = 32 after seeing 8 (two values
tried, both shown; **post hoc**).

**Input.** #122's inputs, unchanged (14 run directories, 7 v1 prompts ×
step143000 / step 0, L1–24, centred and raw, 200 draws, seed 0). Code:
`claude/p1d-position` from `f1deefb`. **Output:** `data/p1d/admit_drop_2026-10-01/`
(`{real,calibrate,report}_m{8,32}.*`, `run_all.sh`), `data/p1d/position_2026-10-01/`
(`position_m{0,8,32}.{json,txt}`; m0 reads `admit_2026-10-01/real.json`).
**Re-run:** `run_all.sh` (~25 min at 14 workers), then
`python -m p1d_cluster_ensemble.position_check --real <real file> --out <json>`
(~2 min each).

**Why step 0 groups the opening.** At init attention is near-uniform over
the prefix: median total-variation distance to uniform over 0..t is 0.11–0.15
per layer at step 0 vs 0.55–0.99 at step143000 (all 7 prompts;
`position_m0.json` `runs`). So position t's attention output is the
mean of the first t + 1 value vectors, and position 0's share of it is
1 / (t + 1): early positions share the first few tokens' values. The centred
cosine to position 0 at step 0 falls from 0.27–0.52 (t 1–3) to 0.05–0.30
(t 4–15) and −0.02–0.07 at t 16–63 (7 prompts × L4 / 12 / 20; step143000
0.03–0.35, 0.01–0.13, −0.03–0.08). The mechanism is read off the
attention, not tested by intervention.

**Results** (`excess`, `min_cluster_size` 2, records with ≥ 1 admitted group
of 56 per band, real / calibration; full tables in the reports):

| M | step | centred L1–8 · 9–16 · 17–24 | raw L1–8 · 9–16 · 17–24 |
|---|---|---|---|
| 0 (#122) | 0 | 43 · 54 · 56 / 0 · 0 · 0 | 13 · 24 · 25 / 0 · 0 · 0 |
| 8 | 0 | 0 · 2 · 7 / 0 · 0 · 0 | 0 · 1 · 0 / 0 · 0 · 0 |
| 32 | 0 | **0 · 0 · 0** / 0 · 0 · 0 | **0 · 0 · 0** / 0 · 0 · 4 |
| 0 (#122) | 143000 | 52 · 53 · 45 / 0 · 3 · 11 | 53 · 51 · 54 / 8 · 5 · 13 |
| 32 | 143000 | **53 · 54 · 45** / 0 · 0 · 4 | 51 · 52 · 53 / 4 · 6 · 12 |

- **Step 0's admitted groups are the opening.** At M = 0 all 153 centred ones
  are positional and hold position 0; step 0's *other* groups are positional
  in 2–5 % (chance). At M = 8 the survivors (9 at `min_cluster_size` 2, 76 at
  4) all hold the new first kept token: the effect moves with the cut and
  weakens, as a prefix average predicts. At M = 32 step 0 admits nothing on
  `excess` in either arm or frame (`log_life`: at most 1 record per cell).
- **Trained admission survives the cut**: 45–54 of 56 records per band,
  calibration 0–4 (centred). At M = 32 `report_m32` releases labels for
  centred L1–8, L9–16, L17–24 and raw L1–8 (`excess`, arm 2). The 10 % bound
  still treats 56 adjacent-layer records as independent.
- **Position check on trained groups (M = 32, centred, arm 2), admitted vs
  not, by band:**

  | | L1–8 | L9–16 | L17–24 |
  |---|---|---|---|
  | positional (`p_near` ≤ 0.05) | 15 vs 19 % | 47 vs 45 % | 40 vs 32 % |
  | mostly near (≥ half of pairs within 3) | 1 vs 9 % | 10 vs 18 % | 14 vs 11 % |

  Admission does not select for position (also within size buckets,
  `/challenge-pr` on #123). *Revised after that review:* the first write-up
  said "about half are runs of nearby tokens". `p_near` is a significance
  flag, and it fires on slight tilts in large groups (of the flagged admitted
  groups, 42–71 % have under a quarter of their pairs near). Only 1–14 % of
  admitted groups are mostly near pairs. Neither column measures *how*
  positional a group is: step 0's opening groups, positional by construction,
  are mostly near in 0–9 % (a 30-token opening spans ~40 positions). M = 0
  gives the same positional shares (19 · 49 · 44 %). "Positional" is not "not
  content": a phrase is both (`a few days`), and so is a topic that stays in
  one paragraph.
- **Step 0's group count is still above its null at M = 32**, against that
  M's own calibration: level-set k above the null mean in 101 of 168 records
  vs 78 in calibration (median z +0.48 vs −0.08). M = 0: 163 vs 93 (+0.80 vs
  +0.20); M = 8: 135 vs 113 (+0.55 vs +0.68). The opening is most of #122's
  160 of 168, not all. *Revised after `/challenge-pr` on #123:* the first
  write-up waved the remainder off with the calibration's swing between arms,
  which is the wrong reference. Unexplained. Step143000: median z +4.0, +3.9,
  +3.6.

**Reading.** (b) settled step 0's *admission* failure: it is the opening,
through near-uniform attention at init, and at M = 32 step 0 admits nothing.
Two limits. M = 32 was picked because step 0 admits nothing there, so "the
control passes at 32" holds by construction (`/challenge-pr` on #123, finding
3): the control has been used to set the cut and has not yet been tested at
it. And step 0's group *count* keeps a small excess at M = 32. Content vs
position for the trained groups is not settled either, but it matters less
than the first write-up said: positional tilts are common and mostly slight.
Only a position-keeping null (option (a)) would measure it.

**For the user (Blocked 11′):** whether "tokens at absolute position ≥ 32"
joins the definition (`design-1d.md`, row "tokens"). If it does, the control
needs a test the cut was not tuned on. Candidates: step 0 at M = 32 on the
long prompts (`data/p1d/long_prompts_2026-09-29/`; expected: admits at most
its calibration; the deduped null drifts past ~1000 tokens, so read it
against its own calibration), or a cut fixed by a rule before running it.
The rule would be where step 0's centred cosine to position 0 reaches its
late-token level, ~16 here; M = 8 shows that may fail at `min_cluster_size`
4. Or option (a).
Labels at M = 32 mark positions before the cut `-3` ("not tested, before
`min_position`") and the file carries `min_position` (*fixed after the
review:* they were `-1`, the deduplication code).

**Parked** (discoveries, not followed):
- **Step 0's late members.** At M = 0, step-0 groups carry 326 members past
  position 64 (~2 per group, the same strings across layers: `perfectly`,
  `on`, `me`). Not their norm (median percentile 0.47–0.60) and not their
  embedding's similarity to the group's opening (median percentile 0.54).
  Why: they are the one part of step 0's groups the prefix average does not
  explain. Cost: < 1 h. Could change: whether the opening is the whole story.
- **Trained groups holding the first kept token** (5–18 % of admitted groups
  at M = 32; that token sits at position 32 or later, so it is not the sink).
  Not followed.
- **Theory for the control and the cut** (2026-10-01 reading list,
  `docs/readings/meanfield_reading_list_2026-10-01.md` §1 rows 1–2, [S]):
  2604.01978 and 2601.21942 analyse random-weight transformers, which is step 0;
  2605.09213 gives a closed-form primacy profile under causal attention. Why: an
  expectation for the control stated before looking. Not the route to Blocked
  11′'s cut: the prefix-average account above gives a rule without them
  (`/challenge-pr` on #124). Cost: reading, once the PDFs are supplied. Could
  change: how step 0 is read as a control.

### Identity-weights positive control (build step 1b; 2026-10-01; branch `claude/p1d-identity`)

**What.** `design-1d.md` "Identity-weights positive control", written and committed
(`53b0738`) before any run; literature `lit-1d.md` §9. The theory's causal dynamics
(2411.04990's (CSA), `Q = K = V = I`, self included, sphere, no MLP / RoPE / LN) and
the full-mask control, run from Phase 1's deduped L0 rows, then admission and #123's
position check on every snapshot. Code `p1d_cluster_ensemble/identity_sim.py`
(`simulate` / `admit` / `report`), 24 tests (`tests/test_phase1d_identity_sim.py`:
the full mask from orthogonal starts follows `gamma_ode`'s (6.9) to 1e-6, the causal pair
is (6.9) at half speed, Thm 4.1, the span reduction exact to 1e-10, the float floor),
sympy check `tools/math_checks/identity_sim_closed_form.py`.

**Input.** #122's 14 run directories (7 v1 prompts × step143000 / step 0), L0, deduped
(125–273 tokens); β ∈ {0, 0.2, 0.43, 1, 2, 3.46, 5.57, 8, 16, 64}; `t` ∈ {0, 0.5, 1, 2,
4, 8, 16}; causal and full; admission 200 draws, seed 0, both frames, `min_cluster_size`
2 / 4; theory clusters at η = 1e-3 (1e-2, 1e-4 stored). **Output:**
`data/p1d/identity_sim_2026-10-01/` (`simulate.parts/`, `traj/`, `admit_{real,calibrate}.parts/`,
`report.{json,txt}`, `run_all.sh`, `run.log`). 280 trajectories, all converged (`dt` halved
until no Gram moved > 1e-6). 3 388 snapshot-frames each real and calibration; 274 / 276
below the float floor, all at `t = 16` (collapsed). **Re-run:** `run_all.sh` (~1.5 h at 14
workers: simulate 10 min, admission ~35 min per pass). Code: `claude/p1d-identity` at
`53b0738` plus the uncommitted `identity_sim.py` (committed as `5abd28e`; the report fields
added during the batch do not touch admission); env: conda `mets` on the local box,
`OMP_NUM_THREADS=1`, 14 workers.

**Fixed on the way.** The float floor (`1 − cos` ≥ 1e-9) bounded the snapshot, not its
null: a nearly collapsed snapshot's Gaussian draws are as tight as it is, one fell below
float32 resolution, and the level-set code refused (`LESSONS.md` 2). Such records are now
skipped with the reason (56 real). And (6.9)'s reference times used `collapse_time`'s step
halving, which never converges where γ never reaches the target (150 s per test); one fixed
step now.

**What the theory does from L0.** Deduped L0 rows are near-orthogonal at both checkpoints
(cosine to position 0 ≈ 0; closest pair `1 − cos` 0.18–0.38 trained, 0.87–0.89 step 0), so this is
the `d ≥ n` regime of Thm 6.9: one global collapse, not several metastable clusters.
η-components appear only at `t ≥ 8`, one per record, at β ≤ 5.57; at β ≥ 16 the Gram does
not move by `t = 16` (self-attention takes the weight). Several theory clusters exist only
at β = 8 (step143000, `t = 16`: 5 clusters, the largest 45 % of tokens). The grid's
multi-cluster regime is that one cell.

*Revised after `/challenge-pr` on #125 (finding 2, verified):* the design said the
collapse time is nearly free of β, but `collapse_time_table` stops at β = 5. The run's own
(6.9) reference times are `t_0.9` ≈ 3.5–4.1 for β ≤ 3.46, 5.6–5.8 at 5.57, 21–24 at 8 and
`inf` at 16 and 64. So β ≥ 8 was past the time grid by construction, and the regime the
positive control was designed for (16, 64) was never reached, not tested and failed
(`LESSONS.md` 6).

**Positive control: not passed, and mostly not readable** (design outcome rows 2 and 3).

Both masks, `min_cluster_size` 4, records with a theory cluster, by the largest theory
cluster's share of tokens (post hoc strata). *Revised after `/challenge-pr` on #125
(finding 3, verified):* the first table was causal only and did not say so; it left out the
full mask at β = 8, `t = 8`, the run's richest multi-cluster cell.

| largest theory cluster | centred: records / recall / calibration admitting | raw: records / recall / calibration admitting |
|---|---|---|
| < 0.5 of tokens (several clusters) | 11 / 0.39 / 7 | **11 / 0.39 / 0**: the one readable multi-cluster cell |
| 0.5–0.95 | 30 / 0.68 / 26 | 30 / 0.23 / 0 |
| ≥ 0.95 (the whole cloud, nearly) | 254 / 0.63 / 163 | 198 / 0.06 / 2 of 195 with a calibration |

Recall against *every* HDBSCAN group, admitted or not (post hoc), is the same except raw
≥ 0.95 (0.18).

1. **The matched Gaussian is not a valid null for a collapsing cloud in the centred
   frame**: its own draws admit (196 of 295 records). So the centred column is not read.
   This is a confound for trained admission too, not only a side note (`/challenge-pr` on
   #125, finding 4): #122's own centred calibration admits in 11 of 56 records at
   step143000 L17–24. Whether those are collapse-like layers is not checked.
2. **Where the null is quiet (raw), the losses are in the group step, not the null**:
   recall against any group is barely higher. Inside a collapsing theory cluster distances
   still span 1e-9 to 1e-3, `S_C/|C|` is in units of 1 / distance, and EOM selects the
   innermost cores (a 166-token theory cluster's best match is 0.3).
3. **A cluster that is the whole cloud cannot beat the cloud's own Gaussian**, by
   construction (full mask, `t = 8`: recall 0 in both frames). That is the definition
   working as stated, and it means the theory's main object here, global collapse, is
   outside what admission can see.

**The opening: at small β, identity dynamics make an admitted group that is the prompt's
opening** (centred, `min_cluster_size` 2; records with an admitted group holding
position 0, of 7; calibration on the same snapshots in brackets):

| step 0, causal | t = 0.5 | t = 1 | t = 2 |
|---|---|---|---|
| β = 0 | 0 (0) | 6 (0): median 24 tokens, 83 % among the first \|g\| kept | 7 (0): 46 tokens, 94 % |
| β = 0.43 (÷ 8) | 0 (0) | 5 (0): 22, 82 % | 7 (0): 42, 93 % |
| β = 1 | 0 (0) | 0 (0) | 7 (0): 45, 92 % |
| β = 3.46 (`beta_raw`) | 0 | 0 | 0 (none by `t = 16` either) |
| **full mask**, any β ≤ 3.46 | 0 | 0 | 0 |

At `min_cluster_size` 4: 4–5 of 7 at `t = 1`, 7 of 7 at `t = 2` (calibration ≤ 1). The
groups' span is short for their size (`p_span` ≤ 0.05 in 67–86 %). Cosine to `x₁` at kept
ranks 1–8 vs the second half: 0.31 vs 0.04 at `t = 1`, β = 0 (primacy, `lit-1d.md` §9 row
4). Raw frame: only 1–2 of 7. Later (`t ≥ 4`) the calibration fires and the cells are not
read.

- **The full-mask control cannot fail here** (`/challenge-pr` on #125, finding 1): L0 rows
  carry no position and the full mask treats every position alike, so no positional group
  can form under it. Its zero says only that the position comes from the mask.
- **The design's row as written is not met.** It said "one *theory cluster* holding
  position 0". The η-components never isolate the opening: when they form (`t ≥ 8`) they
  are the bulk, and position 0 joins last (the late tokens share one prefix average and
  collapse onto each other first). The opening shows as a forming group that admission
  sees, not a collapsed one.
- **Against step 0's real groups holding position 0** (`data/p1d/admit_2026-10-01`, best
  over L1–24): Jaccard 0.57–0.82 in all 7 prompts at `t = 2` (β 0 or 0.43). *But a
  first-|g|-kept-tokens set of the same size matches as well* (0.57–0.85 at `t = 2`; post hoc
  baseline), so the token-level match adds nothing beyond "both are opening runs of about
  the same size".
- **Step143000** also grows an opening group at β ≤ 1 by `t = 2` (7 of 7, 73–84 %
  opening), none at β = 3.46. Its L0 already admits groups at `t = 0` (7 of 7 records in both
  frames, calibration 0; step 0's L0: 0 of 7). In 5 of 7 (centred) one holds position 0,
  with 7 tokens, 14 % opening, and it persists unchanged under the full mask, so deduped trained L0 is not featureless.

**Reading.** *Revised after `/challenge-pr` on #125 (finding 1, agreed):* the first
version said step 0's failed control is "the theory's own first-forming cluster" and that
the cut "removes a cluster the theory predicts". Neither holds. The papers do not say which
tokens join first (`lit-1d.md` §9 row 1), the run's own ground truth (η-components) never
isolates the opening, and the only instrument calling it a cluster is admission, the thing
under test, readable at `t` 1–2 only. What the run does show: #123's mechanism needs
nothing but the causal mask and near-uniform attention. With every learned weight removed,
small-β dynamics from step 0's embeddings make a group that admission admits, that is the
opening, of the size step 0's real groups have; at β = 3.46 they do not. Step 0's fitted β
is ≈ 0 ("β refit"). So the opening is a mask effect that any definition run on causal
dynamics will meet, which supports cutting it (Blocked 11′) at least as much as keeping
it. At β = 3.46 the opening
does not form within `t ≤ 16`, so on this toy the ÷ 8 convention behaves like step 0 and
`beta_raw` does not. That says nothing about which convention is Pythia's (Blocked 9):
trained attention is far from uniform. As a positive control for the definition: **read
once, and failed at the group step.** The one cell where several theory clusters exist and
the null holds (raw, < 0.5 stratum, 11 records, calibration 0) has recall 0.39, the same
against every HDBSCAN group, so EOM finds the clusters' cores, not the clusters. Elsewhere
the theory makes one global cluster, which admission cannot see by construction, and the
centred null fails. The β ≥ 16 regime the design aimed at was never reached.

**Parked** (discoveries, not followed):
- **Simulated time tracks depth for step 0.** The real layer that best matches the
  simulated opening is L8–20 at `t = 1` and L15–24 at `t = 2`. Why: a time ↔ depth map
  for step 0 is the `T_eff` that Phase 1c never measured on these runs. Cost < 1 h. Could
  change: whether a snapshot can be called "layer ℓ".
- **EOM selects the innermost cores of a collapsing cluster.** A leaf-selection or
  `cluster_selection_epsilon` arm would show whether the theory's clusters are recovered
  by a different selection rule. Cost < 1 h on the stored snapshots. Could change: the
  definition's selection rule.

### The M = 32 cut on the long prompts (Blocked 11′; 2026-10-01; branch `claude/p1d-long-m32`)

**Decision (user, 2026-10-01, with #125 in hand):** "absolute position ≥ 32" joins the
definition *provisionally* (`design-1d.md` row "tokens"), on condition that step 0 passes a
test the cut was not set on. This is that test. Rule written and committed before any run.

**Input.** The 8 long runs (`/run/media/system/HDD_1TB/mets_data/p1d_long/2026-09-29/`,
4 prompts × step143000 / step 0, long prompts hash `91e85cc95888`), L1–24, both frames,
`admit run --n-draws 200 --seed 0`, both `min_cluster_size` arms, real and `--calibrate`,
at M = 32 and at M = 0. A band is 32 records (4 prompts × 8 layers). Deduped, the runs keep
339–699 tokens, under the ~1000 where the deduped null was seen to drift (`design-1d.md`
row "scope"); the calibration is still the reference.

**What is new and what is not.** Positions below `n_v1` (467, 482, 242, 446) reproduce the v1
runs that set the cut (6 of 8 bit-identical, `prefix_check.json`). The fresh part is the
continuation: 1373, 550, 1783 and 1590 tokens. The null and its calibration are refitted on
the longer deduped set, so even groups inside the prefix face a different bar.

**Rule** (primary: `excess`, `min_cluster_size` 2; 6 cells = 2 frames × 3 bands):

| step 0 at M = 32, per cell | reading |
|---|---|
| admits in ≤ 3 of 32 records (`RELEASE_BOUND`, the report's own control condition) | pass |
| admits in > 3, and in more records than its calibration | fail |
| admits in > 3, calibration in at least as many | unreadable (null off nominal at length) |

The cut **holds** if all 6 cells pass and **fails** if any cell fails; otherwise it is
reported cell by cell as partial. On a fail the decision goes back to the user (a cut fixed by
a rule, or option (a)).

**Can the test fail?** M = 0 on the same runs is the check. On v1, step 0 at M = 0 admitted in
43 · 54 · 56 of 56 records (centred). If step 0 at M = 0 admits in ≤ 3 of 32 in most centred
cells on the long runs, the instrument does not see the opening at length, and an M = 32 pass
is not read.

**Reported, not part of the verdict:** `min_cluster_size` 4, `log_life`, step143000 at both
M, where step 0's admitted groups at M = 32 sit (inside `[32, n_v1)` or past `n_v1`), and
#123's position check on them.

**Output:** `data/p1d/long_m32_2026-10-01/` (`{real,calibrate,report}_m{32,0}.*`,
`position_m32.{json,txt}`, `run_all.sh`, `run.log`, `verdict.py` applies the rule; `pilot/`
is the one-record populated check). Code: `claude/p1d-long-m32` at `f6c8ea4`, no code change; env: conda `mets`, local
box, `OMP_NUM_THREADS=1`, 14 workers. **Re-run:** `run_all.sh` (~25 min per pass, 4 passes),
then `python -m p1d_cluster_ensemble.position_check --real real_m32.json --out
position_m32.json` (~15 min beside the batch).

**Result: the cut fails** (rule above). Step 0 at M = 32, `excess`, records admitting of 32,
real / calibration:

| arm | frame | L1–8 | L9–16 | L17–24 |
|---|---|---|---|---|
| **2 (primary)** | centred | 0 / 0 | 0 / 0 | 0 / 0 |
| **2 (primary)** | raw | 0 / 0 | 0 / 1 | **4 / 0: fail** |
| 4 | centred | 0 / 0 | 1 / 2 | **13 / 0** |
| 4 | raw | 0 / 0 | 3 / 0 | 1 / 0 |

`log_life` agrees: centred L17–24 6 / 0 (arm 2), 18 / 0 (arm 4); arm 4 centred L9–16 8 / 1.

- **The primary fail is marginal and not the opening.** Its 7 admitted groups are pairs and one
  group of 5 (`latex_monograph`, `sullivan_ballou`, `wiki_paragraph`, L19–22), p 0.015–0.05,
  members far apart (e.g. positions 208 and 832); position check: 0 positional, 0 holding the
  first kept token. One cell over the bound by one record, so by the rule it fails, and nothing
  in it looks like the mechanism the cut was for.
- **The sensitivity arm is the mechanism, moved to the cut.** All 14 of step 0's admitted
  centred arm-4 groups (13 at L17–24, 1 at L9–16) hold position 32, the first token the cut
  keeps, and all 14 are positional (`p_near` ≤ 0.05). They come from 2 of 4 prompts
  (`hdbscan_code` 8 records, `wiki_paragraph` 5; per prompt, 2 of 4 against calibration 0),
  33–81 members, 55–75 % of them among the first |g| kept tokens, median position 50–140.
  As at M = 8 on v1, the opening moves with the cut. *Location (pre-stated, added after
  `/challenge-pr` on #126, finding 3):* 13 of the 14 also hold members past `n_v1`, 2–28 per
  group, as late as position ~2000, so they are an opening core plus scattered late tokens,
  like #123's parked "step 0's late members".
- **Counted per prompt** (`/challenge-pr` on #126, finding 2): the primary fail's 4 records are
  3 prompts (one pair, ` help` / ` always` at 208 / 832, is admitted at L20 and L21), and the
  same cell's calibration admitted in 4 of 56 on v1 at M = 32. Records are 8 adjacent layers
  of one prompt, so a per-record bound overstates both cells.
- **Why v1 passed at 32 and long does not** (a reading, not tested): an opening run of ~30
  tokens is 12–29 % of v1's 103–248 kept tokens at M = 32 and 5–9 % of the long runs' 317–671,
  so at length it is a small dense part of a larger cloud, which is what beats the cloud's own
  Gaussian. A run that is most of the cloud cannot (#125's reading 3). If so, the cut passed on
  v1 partly because v1 is short. Against it: across prompts the share does not order the
  failures (`hdbscan_code`, the smallest long cloud at 317, fails; `latex_monograph` at 525
  and `sullivan_ballou` at 387 do not); within each prompt, long is larger than v1.
- **Trained admission survives at length:** step143000 admits in 27–32 of 32 records per
  cell (arm 2; calibration 2 · 2 · 9 centred, 6 · 2 · 10 raw). Released at M = 32: arm 2
  centred L1–8, L9–16 and raw L9–16; arm 4 centred L1–8, L9–16 and raw L1–8, L9–16. L17–24 is
  withheld in both frames (calibration 9–13 of 32: the deduped null is off nominal late at
  length). Position check, admitted vs not (arm 2, centred): positional 13 vs 19 %, 41 vs 40 %,
  39 vs 35 %; first kept 1–2 %.
- **The test could fail (M = 0 on the same runs):** step 0 admits in 22 / 2 · 31 / 0 · 32 / 0
  (centred) and 13 / 0 · 18 / 0 · 19 / 1 (raw) records, arm 2; arm 4 centred 23 / 0 · 30 / 1 ·
  32 / 0. All 85 centred arm-2 groups hold position 0 (median 38 members). So the instrument
  sees the opening at length, and the cut takes arm 2 from there to 0 · 0 · 0 centred; arm 4 at
  L17–24 goes from 32 to 13.

**Reading.** By the rule committed before the run, the cut fails, and the decision goes back to
the user. The primary-arm fail alone would be a weak reason (one record over the bound, not
the opening). The sensitivity arm is the strong one: the opening re-forms at the first kept
token, as it did at M = 8 on v1, so the cut moved the effect rather than removing it. It
also weakens as M grows (v1, arm 2: 153 groups at M = 0, 9 at 8, 0 at 32; long, arm 4 centred
L17–24: 32 records at M = 0, 13 at 32). *Revised after `/challenge-pr` on #126 (finding 1,
agreed):* the first version said a larger cut moves the opening "whatever M is". From #123's
prefix average, tokens just past M share positions 0..M−1, a share M / (t + 1) of their
prefix, but the average of M near-orthogonal values has norm ~1/√M, so the shared pull falls
with M. What the data support is that the cut that removes the opening depends on prompt
length: 32 does on v1, not at ~2000 tokens. A larger M (128 was proposed) is untested, and
picking it on the long runs would repeat the tuning-on-the-control problem one level up.

**For the user (Blocked 11″):** the cut is not in the definition. Options: (a) the
position-keeping null (~7 h on v1; the long runs would add about as much again), which judges
every group, trained ones included, against a null that keeps position. It addresses the
opening, **not** the primary cell that failed (raw L17–24: non-positional pairs, `/challenge-pr`
on #126 finding 2), so under (a) that cell stays as it is. A cut that scales with length, or a
rule fixed in advance, tested on data it was not set on (no third set of prompts exists yet).
A group-level rule (for example, drop groups holding the first kept token), post hoc on this
run. Also open: whether the control and release bound count prompts rather than layer-records.
Recommendation: (a), with the raw L17–24 cell read as the noise level of the deduped null at
length.

### Blocked 11″ decided, and where 1d stands (2026-10-01; docs only, nothing run)

**Decision (user, 2026-10-01, with #126 in hand):** option (a), the position-keeping null.
The cut is not in the definition. Its form, fixed before any code or run:

| | |
|---|---|
| primary | `x_t = μ̂_t + ε`, in the frame's span coordinates. `μ̂_t = Z̄ + c (P_t − P̄)`, with `P_t` the *causal* running mean of the kept rows before `t` (the first kept row gets `P̄`), `c` one least-squares scalar per record **on the centred regressor** (#123's mechanism: near-uniform attention makes early tokens share a prefix average). `ε = G R / √n` as in `gaussian_draw`, `R = Z − μ̂` the *regression* residual (mean 0 by construction); unit rows. With `c` forced to 0 it is `gaussian_draw`, draw for draw (a test). *Clarified after `/challenge-pr` on #127, finding 1:* uncentred, `P_t` is nearly `Z̄` in the raw frame and `c` collapses (0.03 on step-0 `wiki_paragraph`; centred 0.24–0.33). |
| sensitivity arm | `μ̂_t` = leave-one-out Gaussian kernel smoother over **log(1 + absolute position)**, bandwidth by leave-one-out CV on a grid ending at "no position". *Changed after `/challenge-pr` on #127, finding 3:* on absolute position CV picks 24–93 tokens, wider than the opening; the prefix pull falls as `M / (t + 1)`, so log position is the mechanism's scale (it also fitted better in 4 of 4 step-0 records the reviewer tried) |
| rule (fixed before the run; finding 4) | the verdict is the primary's step 0, `excess`, `min_cluster_size` 2, 6 cells (2 frames × 3 bands) of 56 records, #126's table: ≤ 5 records admitting (`RELEASE_BOUND`) pass; > 5 and more than its calibration fail; otherwise unreadable. All 6 pass = the null controls position, and trained cells are read through `admit report`'s release rule. Any fail: the decision returns to the user. The arm is reported beside it; where it and the primary disagree, the primary decides and the disagreement is reported. Can it fail: step 0 under the plain Gaussian at M = 0 admitted in 43 · 54 · 56 (centred) |
| known risk (finding 2) | one pooled `c` is fitted mostly where there is no opening. On step 0 the reviewer found `c` 0.21–0.34 pooled (centred, R² < 1 %) against 0.38–0.58 on the first 32 kept rows (R² 4–12 %) and ≤ 0 past 32, so the null may reproduce only about half the opening's shift, and step 0 could fail on the fit rather than the mechanism. Not redesigned before seeing the run. On a step-0 fail the report gives `c` on the first 32 kept rows beside the pooled one, as a diagnosis, not a rescue |
| calibration | one draw of the same null as pseudo-data, the null refitted to it (as `--calibrate`) |
| input, first run | v1 (#122's 14 runs, M = 0), then the long runs |

**The form above fails its synthetic positive control, so the real run was not made**
(2026-10-01, same session; code on branch `claude/p1d-position-null-run`, `c085dba`, not for
merge; scripts `p1d_cluster_ensemble/scratch_checks/syn{4,5,6}.py` there). Synthetic opening:
`Y_t = E_t + 6 · mean(V_0..V_t)`, n 150, d 300, `E`, `V` iid N(0, I), the mechanism of #123
with nothing else in it. Records admitting a group that holds position 0, of 10 seeds, 39
draws, `excess`, `min_cluster_size` 2:

| null | raw | centred |
|---|---|---|
| Gaussian (#122) | 7 | — |
| prefix, as committed (fit and draw on unit rows) | 7 | 10 |
| **oracle**: the true positional mean, drawn as committed | **7** | — |
| oracle mean, each row keeps its own residual norm | 0 | — |
| prefix, fit and draw on the un-normalised rows, then the frame | 2 | 3 |
| smooth (log position), fit and draw on the un-normalised rows | **0** | **0** |
| pure noise (`Y = E`), the last two | 0 | 0 |

**Why.** Even the true mean fails, so the defect is the draw, not only the fit. Unit rows make
the noise heteroscedastic: an early token's shared component inflates its norm, so after
normalisation its own noise is a small share, while the committed draw gives every token the
pooled residual covariance and spreads the opening wider than it is. Fitting and drawing
before normalisation, where the mechanism is additive and the noise even, and applying the
frame to each draw, fixes it for the smoother. The prefix fit still under-fits (`c` ≈ 0.5),
which is finding 2 showing on data with a known answer. **For the user (Blocked 11‴):** adopt
"fit and draw on the un-normalised activations, then the frame" (recommended: it is a defect
of the committed form, found on a synthetic, not on the control), and make the smoother the
primary with the prefix as the arm, since only the smoother passes the synthetic. Then the
build adds these synthetics as tests and runs v1 (~12 min a pass).
| counting | the verdict stays per layer-record (comparable with #122–#126); a per-prompt count is reported beside it |

**Correction to the cost.** The "~7 h" quoted for (a) since #108 is `attention_null`'s, which
pushes every draw through the model's own block. Admission's null is a Gaussian draw
(`admit.admit_record` → `gaussian_null.gaussian_draw`), and (a) changes only that draw, so it
costs about one admission pass: ~6 min real + ~6 min calibration per model on v1 (#122's
timings). Also: the Parked variant "a within-prompt position-block shuffle of rows" cannot work
for admission. `S_C / |C|` depends only on the point set, which a row shuffle leaves unchanged.

**HDBSCAN's tie order upstream** (from #122's defect; seconds, conda `mets`). Same points,
rows permuted, labels mapped back (`tools/hdbscan_tie_repro.py`, standalone, a 200 × 1024
cosine set): `hdbscan` 0.8.41 (`min_samples` 2) and scikit-learn 1.7.2 `HDBSCAN` at the matched
setting (`min_samples` 3: sklearn counts the point itself) each change their clustering in
**26 of 30** row orders; a repeat on one order is identical. Scratch checks beside it (not
committed): 11 of 30 for 3 Euclidean blobs with `hdbscan`, group count 2–5 across orders;
sklearn 9–18 tie-artefact groups per order; `level_set_hdbscan` 0 of 10. **Known upstream
as a symptom, not diagnosed or fixed** (*corrected after `/challenge-pr` on #127, finding 5*):
scikit-learn-contrib/hdbscan #265 (2018, open) is the exact row-order report (duplicate points,
Jaccard); on #409 (2020, open) a commenter names non-unique MST edge weights; on #241 (column
order) the maintainer guessed ties. New from 1d: ties come from core distances, so they are
the rule even without duplicates; a measure (groups never a graph component); a tested fix;
scikit-learn has it too (nothing found on its tracker). Reporting it (a comment on #265, an
issue on scikit-learn) is the user's call; draft in
`docs/upstream/hdbscan_ties.md`, repro `tools/hdbscan_tie_repro.py`.

**Where 1d stands** (each row's numbers are in the section named; this table is the summary
`STATE.md` points to). Three sources of structure, and what separates them: the *algorithm*
(HDBSCAN finds groups in noise; tie artefacts), separated by the calibration; the
*architecture*, not learned (identical strings; the prompt's opening through the causal mask
at init), separated by deduplication and the step-0 control; *learned*, step143000 against
step 0, which does not yet separate learned content from learned position.

| question | answer now | section |
|---|---|---|
| is the trained cloud lumpier than its covariance? | yes, locally, at every depth and length; step 0 not, once deduped | "Matched-covariance Gaussian null", "Long prompts" |
| is the token grading (core / halo / contested) usable? | no: dropping one family replaces the core set under all 9 vote rules; retired (Blocked 10), code kept | "Vote rules", "Design revised" |
| is HDBSCAN's shipped call reproducible? | no: on float32 it drifted (fixed, float64); its tied edges are ordered by row order, 42 % of trained groups are tie artefacts (replaced by the level-set tree) | "Float64 distances", "Admission" |
| does the per-group definition pass its control? | no: step 0 admits the prompt's opening (near-uniform attention); a fixed cut at 32 removes it on v1 but not at length; the position-keeping null (11‴) fails all 6 cells | "Admission", "Position", "The M = 32 cut on the long prompts", "Position-keeping null" |
| do trained groups survive the opening's removal? | yes, at v1 and at length (L17–24 withheld: the null is off nominal there) | same |
| content or position? | not the start of the context: moved behind a preamble, trained groups mostly move with the passage (centred size 2: 0.95 / 0.89 / 0.74 by band; fixed bar 0.5: 0.89 / 0.81 / 0.65), step 0's do not (0.28 / 0.09 / 0.08), its opening is opening-bound in 7 of 7 passages. Relative positions mean a preamble cannot test absolute position. "Learned" is unit 2's | "Unit 1: move the text" |
| can random re-inits stand in for Pythia's init? | yes at step 0, carried by the weights (same σ, untruncated Gaussian, float16-valued); the cloud check passes all 24 cells (largest share 0.143, bound 0.20) but fails only a gross mismatch (≥ ~1.5 SD), with a small late lean it cannot resolve; re-run on the trained union's token sets, it passes again (`ci2` raw L17–24 0.171) | "Unit 2: the architecture null — first check", "Unit 2: the trained cells" |
| is it learned? | yes, beyond 40 re-inits and replicating across 10 seeds in every band; per group concentrated early (centred size 2, replicating: 170 / 41 / 17 by band), not runs (6 of 515 contiguous) but tighter in position than chance (31 % in the tightest 5 % of random spreads; position is unit 1's test), mostly lexical-semantic classes ({was, is, were}, {year, years, months}) present from L1, 15 bulk. Which exist at L0 (the embedding) and which form with depth is open | "Unit 2: the trained cells" |
| the candidates (moves ∧ learned ∧ replicating), and their origin | 120 distinct (centred size 2); **71 carried from L0** (the embedding's word classes, through L1–8), 4 formed at L1, 45 formed later (all candidates first seen at L9–24; an upper bound, since carried classes that gain members count as formed). **Without replication** (moves ∧ learned, the outcome table's words): 205, 116 formed later, so the verdict turns on replication; Phase 10 uses the set without it (user, 2026-10-04), the replicating set stays the cross-seed definition. Raw size 2 is the other way (54 of 79 formed). Carried groups are tighter in position than chance too, so position spread cannot tell an embedding class from a formed one. "Moves" barely filters: 181 of 190 classified learned, replicating groups move. Step 0: no candidates, and 2 of 647 groups carried | "The candidates, and where each comes from" |
| does the definition recover known clusters? | partly: recall 0.39 on the identity-weights positive control (EOM finds cores) | "Identity-weights positive control" |
| does unit 3's scale rule find planted scales? | not as designed: the first check fails on the multi-scale synthetic because (b), the substantial count against the Gaussian, has the wrong tail (the Gaussian has more pieces). Without (b) the centred frame finds both scales (post hoc); raw cannot at t = 2. Fix is Blocked 15. **Re-run under it (seed 1): fails again, both arms, on the plateau's "same count"** (fringe clusters move the count while the partition holds; Gaussians 0 of 50). Under Blocked 16 (anchored ARI, opening labelled, 10 seeds): **7 of 10, fails the 8 bar**; Gaussians 0 of 50. Under Blocked 17 on fresh present seeds: 7 of 10 again (coarse 10, fine 7). **Accepted as measured (Blocked 18, option 1):** a found plateau's partition is the claim, a missing one weak evidence; real input waits on a specificity check among the inits. **That check, built and piloted: step-0 clouds' merges all fall within r ≈ 0.5–1.0, so it cannot fail (Blocked 19)** | "Unit 3: the synthetic and its first check"; "Unit 3 re-run"; "Unit 3 on seeds 2–11"; "Unit 3 on fresh seeds"; "The real-input reader, step 1: built"; `design-1d.md` "Blocked 18 decided" |
| attention communities | weak, late (L17–23) against the position-keeping attention null B | "Attention communities" |
| β (for C's scale) | 3.46 [1.55, 5.57]; the convention is Blocked 9 | "β refit" |

**Standing directions from the user** (moved here from `STATE.md` on 2026-10-01, where they
were the only record): 2026-09-25, the priority is more ways to check that a cluster is real
and not an HDBSCAN artefact; D (hierarchy across scales) and C (`δ = cβ^{-1/2}` marked on it)
yes, C waiting on β's convention (Blocked 9); prompts to maximum length yes (done, "Long
prompts"); whether L0 (token identity: 215 clusters = 215 distinct embedding vectors) stays in
1d is the user's call. 2026-09-26: the order of next items, (3) first (done). Not yet taken up:
C's `δ` on the merge tree (Parked above), and the theory's own definitions (fixed-scale F13/F14,
`p10_cluster_function/math-10.md` §7; persistence across layers). Open beside 11″: is 1d done
enough to write up (`docs/TRIAGE_2026-09.md` §4.2)?

### Position-keeping null (Blocked 11‴; 2026-10-02; branch `claude/p1d-position-null-v1`)

**Decision (user, 2026-10-02):** adopt 11‴: fit and draw on the un-normalised activations,
then apply the frame; the smoother is primary, the prefix fit the arm; the synthetics become
tests; then run v1. The verdict rule is 11″'s, unchanged, read on the new primary.

**Built.** `position_null.py` (`fit` / `draw` / `apply_frame`), `admit run --null
{smooth,prefix}`; raw rows are `norms × activations` (`activations.npz` stores unit rows and
the norms, `p1_io`). `tests/test_phase1d_position_null.py` (17 tests): the synthetic opening of
the section above, now with the 11‴ form: smooth admits it in **0 / 10** seeds raw and centred,
#122's Gaussian 7 / 10 raw (the test can fail), pure noise 0 / 10 for both nulls; prefix 2 / 10
raw, 3 / 10 centred (recorded, not asserted). Also: `gaussian` kind = `gaussian_draw(renorm =
False)` draw for draw; `apply_frame` = `frame_vectors`' rows; the driver fits on raw rows.

**Input.** #122's 14 v1 runs (7 prompts × step143000 / step 0, the `RUNS` of
`data/p1d/admit_2026-10-01/run_all.sh`), deduped, M = 0, L1–24, both frames, 200 draws, seed
0, both `min_cluster_size` arms, real and `--calibrate`; git `45ced15`, conda `mets`.
**Output:** `data/p1d/position_null_2026-10-02/{smooth,prefix}/` (`real.json`,
`calibrate.json`, 672 records each, none skipped; `report.{json,txt}`), `run_all.sh`,
`run.log`. **Re-run:** `run_all.sh` there (~7 min a pass at 14 workers, 4 passes, resumable).

**Result: the control fails, all 6 cells, on both nulls.** Step 0, `excess`,
`min_cluster_size` 2, records admitting of 56 (calibration in brackets); the rule: ≤ 5 pass,
> 5 and above its calibration fail:

| null | centred L1–8 | L9–16 | L17–24 | raw L1–8 | L9–16 | L17–24 |
|---|---|---|---|---|---|---|
| Gaussian (#122, unit rows) | 43 (0) | 54 (0) | 56 (0) | 13 (0) | 24 (0) | 25 (0) |
| **smooth (primary)** | **27 (2)** | **30 (12)** | **40 (10)** | **10 (0)** | **20 (0)** | **24 (1)** |
| prefix (arm) | 40 (0) | 53 (0) | 56 (0) | 14 (0) | 24 (0) | 24 (0) |

Per prompt (smooth): 7 · 7 · 6 of 7 centred, 4 · 5 · 5 raw. `min_cluster_size` 4 fails too
(centred 36 · 41 · 46, raw 9 · 24 · 23). The primary and the arm agree on the verdict.
**What is admitted is still the opening:** under smooth, all 97 admitted centred groups hold
position 0 and all 97 are groups #122 admitted; raw, all 56 start below position 8 (39 hold
0), and 55 of 56 are #122's. The smoother does fit position (CV picks `h` 0.68 on log position
in 136 of 168 records, R² ~2 %); the prefix's `c` is 0.46 median (0.02–0.64).

**Why (diagnosis, not a rescue; nothing refitted).** The step-0 residual looks like the
synthetic's: early / late residual norm 0.97–1.03, mean cosine among early residuals −0.02 to
−0.06, adjacent early residuals −0.01 to −0.08 (7 prompts × L4/12/20). But on 6 admitted
records (centred) the real opening is tighter than the same tokens in the null's draws
(within-group cosine distance 0.80–0.88 vs 0.86–0.92, 1.5–6 SD), and the *rest* of the tokens
are farther from their nearest neighbour than in the draws (median 0.82–0.86 vs 0.73–0.77).
So the fitted mean carries only part of the opening's shift (finding 2's risk, now on the
smoother), and the real background is more spread out than its Gaussian, which the
opening's group is measured against. The smooth calibration is itself off nominal centred
at L9–24 (12, 10 of 56).

**Confound: trained layers are not readable under this null.** Fitting before
normalisation hands token 0's massive activation (norm ~45× the median at L6–18, step143000
`wiki_paragraph`) to every draw: one row's share of the drawn noise (`resid_top_share`,
written per record) is median 0.92 at L9–16, 0.70 at L17–24 (step 0: ≤ 0.02). Each draw is
then mostly ±one direction, its groups are very tight (per-draw maximum `excess` ~21–27 vs
~0.05–0.09 at step 0, `wiki_paragraph` L12), and **step143000 admits in 0 of 56 records at
L9–16** under both nulls. That zero is the null, not the model, and its calibration (0 of
56) cannot catch it. *Fixed after `/challenge-pr` on #128, finding 1:* `admit report` now
withholds any cell with a record above `RESID_TOP_BOUND` = 0.5 (placed after this run; step 0
≤ 0.02): every step143000 cell, in 21 · 56 · 39 of 56 records by band (reports re-made, records
unchanged). Nothing trained is read.

**The synthetic's comparator** (*after `/challenge-pr` on #128, finding 2*): #122's Gaussian
differs from `smooth` both in where it draws and in keeping no position. The matched one,
`--null flat` (no position, drawn before normalisation), admits the synthetic opening in 6 /
10 centred and 2 / 10 raw (reviewer's run). So the centred test carries the evidence (6 → 0);
raw separates the two by one seed. Now `test_flat_admits_the_opening_centred`.

**The primary fails its own calibration** (*finding 3*): centred L9–16 / L17–24 admit in 12 /
10 of 56 on pseudo-data drawn from the null's own model, which points at the estimator as
well as the data. The reviewer found that the leave-one-out mean used for the draw (CV's
candidate, not refitted in-sample) recovers ~39 % of token 0's positional shift on the
synthetic at `h` 0.68, against ~68 % in-sample; the synthetic still passes at 0.68, so it is
not shown to cause the real failure. A fix can be judged first on its own calibration
(pseudo-data, no real control needed) and then on a fresh control, which makes option (ii)
less post hoc than first written.

**What this does not show.** That no position-keeping null can pass: the two tried keep the
mean only. The diagnosis was read on the control the null was set against, so a null designed
from it would be post hoc on the same 56-record cells.

**For the user (Blocked 11⁗).** Option (a) as built does not separate the opening from
learned structure, and step 0 on v1 has now set or tested four constructions (Gaussian, cut
at 8 / 32, two position nulls). The options are in `STATE.md`.

### Blocked 11⁗ decided: stop modelling nulls; intervene, use the architecture as the null, one scale axis (2026-10-02; docs only)

**Why the route changes.** Each null so far modelled one non-learned source (covariance,
token identity, the opening, token 0's massive activation) and met the next. The opening is
a real cluster at init; "is it a cluster" and "is it learned" were being asked of one null.
**Decision (user, 2026-10-02, "let's do it all"):** the programme below, one unit each, in
this order. The smoother's estimator fix (11⁗ (ii)) is not taken up; (i), the write-up,
waits for the programme.

| # | unit | what it answers | first check |
|---|---|---|---|
| 0 | **literature scan + `design-1d.md` revision** (the trigger in `CLAUDE.md`): rules for 1–4 fixed before any run; how token 0 / attention sinks are handled, once, for every method | — | PolyPythias' sizes and seeds (410m?), Pythia's init scheme, prior position-invariance tests, Markov stability, massive-activation practice |
| 1 | **move the text**: the same v1 passages after unrelated preambles of several lengths (e.g. 0, 50, 300, 1000 tokens), step 0 and step143000 | content vs position, by intervention: a cluster is a set of tokens the model keeps together when they move | step 0's opening moves to the preamble |
| 2 | **architecture null**: the same prompts through N random initialisations (re-init, or PolyPythias step 0); per-cloud standardised statistic against that spread; PolyPythias trained seeds as a replication check if available | a step-0 control that passes by construction; does a cluster replicate across training seeds | real step 0 ranks as a typical draw (else the init does not match) |
| 3 | **scale spectrum**: one family on a continuous scale (merge tree or Markov stability), per scale: subsampling stability, excess over (2), invariance under (1); other families compared only at matched scale; `δ = cβ^{-1/2}` marked when Blocked 9 allows | plateaus ("robust scales") instead of micro / meso / macro picked by seven families | the synthetic below shows its planted plateaus |
| 4 | **positive controls**: designed-content prompts (lists, interleaved code / prose, repeated entities) and a multi-scale synthetic with the opening mechanism | does the tool find what it should, at the right scale, and not position | — |

Supersedes the cut, the position nulls and the seven-family consensus as the route to a
definition; their code and results stay (each is a row in "Where 1d stands").

### Unit 0: literature scan and design (2026-10-02; docs only; branch `claude/p1d-unit0-design`)

**Done.** Scan: `lit-1d.md` §10. Rules for units 1–4 and the token rules (T1–T5):
`design-1d.md` "The programme". Nothing was run on a model; three checks were read off
the tree. Their numbers and inputs are the scan's [M] rows (`lit-1d.md` §10):

| check | row | changes |
|---|---|---|
| PolyPythias 410m on the Hub | 1, 1a | unit 2 has 10 real inits and 10 trained endpoints |
| Pythia-410m `step0` weight σ against transformers' `init_weights()` | 2a | unit 2's re-init writes Pythia's two σ; `init_weights()` rejected |
| norm / median on the v1 deduped batch | 4a | T1 excludes position 0 everywhere; T2 excludes any token over 10× in any compared run |

**`/challenge-pr` on #129** (accept with changes; answered on the PR; every finding taken):
cross-model statistics are standardised within each cloud against its own Gaussian first
(`design-1d.md` "Within-cloud scale"); unit 2's fallback refuses rather than reading a
p-floor, and its first check is per band; `sullivan_ballou`'s continuation (550 tokens) is
too short for P = 1000 and is dropped as a preamble; unit 1 reads "moves" against each
group's own subsample floor, classifies groups per preamble (moves / preamble-dependent /
opening-bound / context-bound), keeps one token set at every P, and runs the designed
prompts first. **Next: unit 1 (move the text).**

### Unit 1: move the text (2026-10-02; branch `claude/p1d-move-text`)

**Built, in this order (git shows it):** the three designed-content prompts and their
readout rule, frozen alone before any forward pass (`designed_prompts.py`, hash
`ae6a4312126c`, `4c3fded`); the runner with its first-check rules and gate (`move_text.py`,
`9718cb7`; operational readings in `design-1d.md` "Unit 1"); then the runs. P = 0
reproduces the stored v1 activations (max abs 3e-8, step 0, wiki and homer).
**Inputs:** 410m `step0` / `step143000`; the 7 v1 passages (`homer_iliad` at its first 512
of 562 tokens, as every v1 run read it); preambles = continuations of `wiki_paragraph`,
`hdbscan_code`, `latex_monograph` (`LONG_PROMPTS_HASH` in each record); P 0 / 50 / 300 /
1000; joins EOD (primary) and `\n\n`; L1–24, both frames, sizes 2 and 4. 290 forward passes,
~16 min wall at 14 workers. Output `data/p1d/move_text_2026-10-02/` (`run_first_checks.sh`,
`first_checks.json`, `report.json`); activations for P ∈ {0, 1000}, first preamble, EOD on
`HDD_1TB/mets_data/p1d_move_text_2026-10-02/`.

**T2 found** (> 10× median at L2–20): at step 0 nothing; at step143000 passage offset 0 in
every passage (ratio 46–50; excluded anyway) and the first `\n` in `hdbscan_code` (offset 34)
and `latex_monograph` (offset 10), ratio 41, dropped from every condition of those passages.

**First check 1, designed prompts (step143000, centred, size 2): pass.**

| prompt | content groups at P = 0 (L1–24) | step 0 baseline | classified: moves / other | share moves |
|---|---|---|---|---|
| category list | 75 | 1 | 56 / 10 (1 floor 0) | 0.85 |
| prose / code | 257 | 68 | 101 / 12 (91 floor 0) | 0.89 |
| entities | **0** | 13 | — | — |

The category list gives clean groups (24 animals, 24 body parts, 11–23 colours) at most
layers, which move. **The entity prediction failed**: the trained model puts all three
people's name and title tokens into one group (e.g. L4: teacher 2, sailor 3, surgeon 3),
names-as-a-kind, not one group per person. Step 0's 13 are not entities: they are its
opening group (~20 tokens), which holds the three early "teacher" names; rule 1's purity
is over labelled members only (`LESSONS.md` 6).

**First check 2, step 0's opening (v1, centred, size 2): pass.** The opening group is
opening-bound in 18–23 of 24 layers in every passage (7 of 7 modal). It is specific: of
step 0's other classified groups, 74 of 374 move (178 context-bound, 122 preamble-dependent;
336 floor 0 set apart, finding 1 below): at step 0 almost nothing moves, and the opening
differs from the rest by sitting at the start. Its members' cosine to P = 0 at P = 1000 is
0.88 (median), against ≥ 0.97 for all tokens, and its best Jaccard at P > 0 is 0.14
against a floor `J0` of 0.59.

**Trained v1 (read after both checks passed, through the gate).** Stable P = 0 groups by
class, pooled over the 7 passages and the band's layers (not independent). **Corrected after
`/challenge-pr` on #130, finding 1:** a stable group can have floor `J0 = 0`, and then "best
Jaccard ≥ J0" cannot fail; 336 of step 0's first 412 "moves" were such groups. They are now
counted apart (`floor 0`, `move_text.group_classes`), and a fixed bar of 0.5 is reported
beside. Recomputed from the stored records (no new forward pass); the pre-fix files are kept
as `*_before_floor_fix.json`.

| frame / size | band | step 0: classified, share moves (own floor / bar 0.5) | step143000: classified, share moves (own floor / bar 0.5) | floor 0, step 0 / step143000 | `\n\n` arm, step143000 (own / 0.5) | cos P=1000 vs 0, step 0 / step143000 |
|---|---|---|---|---|---|---|
| centred / 2 | L1–8 | 156, 0.28 / 0.10 | 347, **0.95 / 0.89** | 99 / 156 | 0.91 / 0.79 | 0.99 / 0.99 |
| centred / 2 | L9–16 | 205, 0.09 / 0.02 | 370, **0.89 / 0.81** | 116 / 134 | 0.84 / 0.68 | 0.98 / 0.99 |
| centred / 2 | L17–24 | 171, 0.08 / 0.00 | 231, **0.74 / 0.65** | 121 / 128 | 0.60 / 0.46 | 0.97 / 0.98 |
| centred / 4 | L17–24 | 40, 0.00 / 0.00 | 110, 0.46 / 0.58 | 16 / 9 | — | 0.97 / 0.98 |
| raw / 2 | L1–8, 9–16, 17–24 | 0.31, 0.16, 0.07 (own) | 0.95, 0.92, 0.75 (own) | — | — | — |

Opening-bound at step143000: 0, 0, 2 (centred / 2 by band). The trained opening group, after
T1: no opening group at most layers in 4 passages, *moves* in 2, unstable in 1 (modal). The
first checks are unchanged by the fix (step 0: 7 of 7 opening-bound; designed: 0.85 and 0.89
of classified content groups move, 1 and 91 floor-0 content groups set apart).
**Reading: design outcome row 1.** Step 0's groups do not move (0–28 % by band, ≤ 10 % at the
fixed bar); trained groups mostly do, less so late (L17–24) and under the `\n\n` join.

**Caveats, in order of weight.**
1. **Per-token cosine does not measure the push** (finding 2). Step 0 and trained states move
   about equally (median cosine to P = 0 ≥ 0.96 in every band, both steps, both joins), yet
   step 0's groups break and trained groups survive: a group's survival depends on its
   spacing in its own cloud, not on how far each state moved. *Retracted:* the first
   write-up's "moves is near the default" read this cosine as the push and was carried by
   the floor-0 defect. A matched-cosine random-direction baseline (finding 2) is not built.
2. **What a preamble can test** (finding 3). Pythia's positions are relative, so a preamble
   tests attachment to the start of the context, not absolute position; after EOD a trained
   model can treat the passage as a new document. The `\n\n` join, where it continues one
   context, is the harder test: 0.84 / 0.60 move at L9–16 / L17–24 against EOD's 0.89 / 0.74.
3. **The designed check shows "moves" fires on content, not that it selects content**
   (finding 4): their non-content groups also move. Specificity rests on step 0 (above).
4. Size 4 at L17–24 moves about half the time; pooled shares count dependent records; the
   step-0 check's T2 union is per step (step 0 had no massive token).

**Re-run:** `data/p1d/move_text_2026-10-02/run_first_checks.sh` (resumable; the trained step
passes the gate on `first_checks.json`), then `python -m p1d_cluster_ensemble.move_text
report --out <dir>`. Tests: `tests/test_phase1d_move_text.py`,
`tests/test_phase1d_designed_prompts.py`. **Next: unit 2 (architecture null).**

### Unit 2: the architecture null — first check (2026-10-02; branch `claude/p1d-arch-null`)

**Built:** `arch_null.py` (`e464c9f`, before any record; operational readings in
`design-1d.md` "Unit 2"): Pythia-σ re-init (`reinit_model`; refuses a tensor off its σ),
PolyPythias loading, per-cloud `z_G` of `hdb_k_2`, `hdb_k_4`, `nn1`, `ci2` against 100
draws of the cloud's own Gaussian (common random numbers across models), per-group `s`,
the first check. **Inputs:** step 0 of `pythia-410m` (seed 0) and
`pythia-410m-seed{1..9}`; re-inits 0–39; the 7 deduped v1 prompts (first 512 tokens),
L1–24, centred and raw. 50 models × 7 prompts × 48 records = 16,800 clouds, 77–82 s per
model at 14 workers (~65 min). Output `data/p1d/arch_null_2026-10-02/`
(`run_first_check.sh`, `first_check.json`, `first_check_power.json`, `token_sets.json`).
**Revised after `/challenge-pr` on #131** (`93e66b2`): the re-inits are now rounded to
float16 (finding 3: every real init is float16-valued: PolyPythias store float16, and
`pythia-410m`'s float32 `step0` holds exactly float16 values), and all 40 were re-run; the
float32 re-inits' records are kept in `superseded_fp32_reinits/` (their first check also
passed all 24 cells; the shares below moved by ≤ 0.015). The 10 real-init records are from
the first run, stamped `e464c9f` but computed with the uncommitted spawn-pool change
(`3ffdc0f`, which changes only the pool); `_git_head` now writes `-dirty` (finding 4).

**The real inits are Pythia's init, and this carries the claim** (finding "smaller" 2).
All 10: weight SDs 0.01974–0.01978 (small) and 0.00260–0.00261 (wang), every bias 0, every
LayerNorm (1, 0); the reviewer found seed 1's step-0 weights untruncated Gaussians at
those σ (kurtosis within ±0.005, KS p 0.19–0.82). Drawn from one distribution at one
precision, the two kinds of init match by construction; the cloud check below is a smoke
test of the re-init code, not independent evidence. **T2 found nothing** at step 0 in any
of the 50 models (largest norm ratio 1.35×, at position 0): one token set per prompt.

**First check (float16 re-inits): pass in all 24 cells** (share of the 70 (seed, prompt)
band-median ranks in the re-inits' outer 10 %; pass ≤ 0.20):

| statistic | centred L1–8 / 9–16 / 17–24 | raw L1–8 / 9–16 / 17–24 |
|---|---|---|
| `hdb_k_2` | 0.00 / 0.00 / 0.00 | 0.00 / 0.01 / 0.00 |
| `hdb_k_4` (arm) | 0.00 / 0.00 / 0.04 | 0.00 / 0.01 / 0.06 |
| `nn1` | 0.00 / 0.01 / 0.01 | 0.00 / 0.00 / 0.00 |
| `ci2` | 0.03 / 0.01 / 0.11 | 0.03 / 0.07 / 0.14 |

The raw statistics agree too (centred medians, init vs re-init: `hdb_k_2` 13 / 12, 11 / 11,
11 / 11 by band; `nn1` and `ci2` equal to the third decimal). Seeds 3 and 4 (PolyPythias'
outliers) are not apart from the rest: `ci2` centred L17–24 per-seed median ranks run
0.26–0.80, seed 4 at the top (0.80), seed 3 at 0.65, seeds 7 and 9 at 0.71–0.73.

**How much the check can see** (`power`, written by `check` into
`first_check_power.json`; 4 folds, each holding 10 re-inits out as pseudo-real, and real
and held-out inits **ranked against the same 30 remaining re-inits**). *Corrected after
finding 1:* the first write-up compared real inits ranked against 40 with held-out ones
ranked against 30, whose tie-driven baseline differs (0.098 vs 0.13).

| reading | held-out re-inits | real inits |
|---|---|---|
| check statistic (band medians), per cell | mean 0.00–0.11, max 0.00–0.17 | 0.00–0.16 |
| per layer (no median), per cell | 0.121–0.133 | 0.082–0.168 |

Real exceeds the held-out maximum on the check statistic in 4 cells, 3 of them late: `ci2`
raw L17–24 0.161 vs 0.100, `hdb_k_4` raw / centred L17–24 0.064 / 0.036 vs 0.029 / 0.014, and
`hdb_k_2` raw L9–16 0.018 vs 0.014 (one band median in 70). Per layer, the highest real shares are `ci2` raw
L17–24 0.168, `hdb_k_4` centred L17–24 0.160, `hdb_k_4` raw L9–16 0.158, against ~0.13.
The real set is 10 seeds whose layers and prompts are correlated, so its spread is wider
than the fold mean's. **Sensitivity** (synthetic, layers independent,
`test_first_check_sensitivity_is_between_1_and_1_5_sd`): a shift of every real init by 1 SD
of the re-inits' z passes all 24 cells; 1.5 SD fails 22–24. **Reading:** consistent with one
distribution, with a small late lean (L17–24, `ci2` and `hdb_k_4`) that this check cannot
resolve; the bound of 0.20 fails only a gross mismatch. The weights carry the claim.

**Caveats.** (1) `nn1`'s z is +17 to +27 at both kinds of init: step-0 tokens are farther
from their neighbours than their Gaussian's; a property of every init, and only
comparisons across models are read. (2) The late lean above is where the trained cells will
be read hardest; it is reported, not corrected.

**Settled for the trained cells** (`design-1d.md` "Unit 2", "For the trained cells",
written before any `step143000` record is opened; the user may overrule):
1. **Primary tail:** the lumpier one (`gaussian_null.LUMPIER`): lower for `nn1` and `ci2`,
   upper for `hdb_k`; the other tail is reported, not read.
2. **T2 across the trained comparison:** the union takes the trained runs' massive tokens
   (seed 0: the first `\n` in `hdbscan_code` and `latex_monograph`; seeds 1–9 unread), so
   every prompt whose kept set changes is recomputed for all 60 models, and **the first
   check is re-run on the recomputed records** before a trained cell is read.

**Re-run:** `data/p1d/arch_null_2026-10-02/run_first_check.sh` from the worktree root
(resumable: `norms`, `run`, `check`; `check` writes both JSONs). Tests:
`tests/test_phase1d_arch_null.py` (the re-init test needs `SMOKE_REAL_DEPS=1`).
PolyPythias `step143000` seeds 1–9 are in the HF cache, unopened. **Next: unit 2's trained
cells** (T2 union over 60 models, recompute changed prompts, re-run the check, then the
per-cloud and per-group rules and replication). *Done: next section.*

### Unit 2: the trained cells (2026-10-02; branch `claude/p1d-arch-null-trained`)

**Built before any trained cloud** (`57f12b1`; readings in `design-1d.md` "Operational
readings for the trained cells"): T2 over a named comparison (`--union first|trained`),
`cloud_rules` (rank p in the lumpier tail), `group_bars` / `group_rules` (learned = `s` above
the re-inits' 95th-percentile largest `s`), `replication`, `read`. Records stamped `cddcba7`
(`57f12b1` plus a missing `mkdir`). **Inputs:** step 0 of the 10 real inits and 40 float16
re-inits, step143000 of `pythia-410m` and `pythia-410m-seed{1..9}`; the 7 deduped v1 prompts
(first 512 tokens), L1–24, centred and raw, 100 Gaussian draws per cloud, 60 models × 7
prompts × 48 records. Output `data/p1d/arch_null_trained_2026-10-02/` (`run_trained.sh`,
`first_check.json`, `trained.json` md5 `ad32ca99`, `trained_rows.json`, `read.log`).
`trained.json` `inputs` names the inputs: sha256 of each prompt's token ids and each
checkpoint's HF snapshot (added after `/challenge-pr` on #132, finding 5; the per-record
files, as #131's, carry neither).

**T2 on the 60-model union adds each prompt's first delimiter** (first `.`, or first `\n`
in `hdbscan_code` / `latex_monograph`), massive (ratio 10–54) in 9 of the 10 trained seeds
(seed 3 none; seeds 0, 1 on `\n` only; seed 4 in one prompt). Every kept set loses one token,
so all 50 step-0 models were recomputed (~65 min) on the new sets.

**First check re-run on the recomputed records: pass in all 24 cells**, so no rule refuses.
The late lean grew: `ci2` raw L17–24 0.171 (was 0.143; bound 0.20), per layer 0.200 against
the held-out re-inits' 0.122; `hdb_k_4` raw L17–24 0.057. Every other cell ≤ 0.10.

**Per cloud** (`(prompt, layer)`s of 56 per band with p ≤ 0.05, lumpier tail; "repl" =
beyond in ≥ 8 of 10 seeds; median `z_G`: re-init / trained):

| statistic | frame | L1–8 repl | L9–16 repl | L17–24 repl | median `z_G` L1–8 / 9–16 / 17–24, re-init → trained |
|---|---|---|---|---|---|
| `hdb_k_2` | centred | 43 | 46 | 42 | 0.6 → 3.0, 0.8 → 3.8, 0.9 → 5.5 |
| `hdb_k_2` | raw | 26 | 27 | 19 | −0.6 → 3.5, −0.3 → 4.0, −0.3 → 3.2 |
| `hdb_k_4` | centred | 22 | 19 | 12 | 0.2 → 2.5, 0.3 → 3.2, 0.5 → 3.0 |
| `hdb_k_4` | raw | 21 | 26 | 12 | 0.0 → 2.8, 0.7 → 2.2, 1.0 → 2.0 |
| `nn1` | centred | 56 | 56 | 56 | 26.8 → −18.5, 20.1 → −21.0, 17.1 → −15.9 |
| `nn1` | raw | 56 | 56 | 56 | 21.7 → −11.9, 15.5 → −18.1, 13.3 → −14.4 |
| `ci2` | centred | 56 | 45 | 54 | 5.8 → −0.4, 1.2 → −1.0, 0.3 → −2.5 |
| `ci2` | raw | 53 | 46 | 49 | 5.3 → −0.9, 0.9 → −0.9, 0.2 → −1.9 |

Per-seed counts (in `read.log`) agree across seeds; seeds 3 and 4 are not apart. **Reading:**
every trained cloud is beyond the inits on `nn1` and nearly every one on `ci2`, but the
two say different things. On `nn1`, inits are far *less* lumpy than their Gaussian
(z +13 to +27) and trained clouds far *more* (−12 to −21). On `ci2`, the trained clouds are
near their own Gaussian (−0.4 to −2.5) and the excess over the inits is mostly the inits'
anti-lumpiness at L1–8 (+5.8). **"Beyond the inits" is not "lumpier than its covariance"**;
both columns are in `trained.json` (`median_z`). `hdb_k_4`: 201 of 3,360 trained records
have no z (their Gaussian's size-4 count does not vary: 0 at raw L1–8, 2 at centred L9–24)
and count as not beyond, so its replication counts are lower bounds.

**Per group** (seed 0; "learned" = `s` above its bar; "repl" = Jaccard ≥ 0.5 with a learned
group in ≥ 6 of 9 other seeds):

| frame | band | size 2: groups / `s > 1` / learned / repl | size 4: groups / `s > 1` / learned / repl |
|---|---|---|---|
| centred | L1–8 | 764 / 356 / 277 / **170** | 242 / 114 / 49 / 28 |
| centred | L9–16 | 889 / 431 / 145 / **41** | 249 / 110 / 16 / 10 |
| centred | L17–24 | 590 / 227 / 69 / **17** | 183 / 82 / 8 / 2 |
| raw | L1–8 | 381 / 230 / 206 / **92** | 95 / 80 / 78 / 26 |
| raw | L9–16 | 468 / 287 / 206 / **72** | 125 / 107 / 81 / 17 |
| raw | L17–24 | 378 / 223 / 163 / **31** | 130 / 87 / 52 / 9 |

515 of seed 0's 1,350 learned groups replicate. *Corrected after `/challenge-pr` on #132,
finding 1* (the first write-up said "content, not position", from contiguity, which a random
group almost never has and so cannot test). **They are not runs, but they are tighter in
position than chance:** 6 of 515 are contiguous, yet 162 (31 %) have a position spread in the
tightest 5 % of 2,000 random same-size groups from the prompt's kept tokens, and 330 (64 %)
have two members within 3 positions (random: 25–37 % at size 2, 50–71 % at size 4;
`position_tightness`, per cell in `trained.json`). Text puts related words near each other,
so this does not make them positional; whether they are is unit 1's test, not this one's. 89
have a member in the opening (offset < 8); 15 are **bulk** (≥ 25 % of the prompt's kept
tokens, up to ~140), not word classes, and are counted in the 515. "Learned" is not a subset
of `s > 1`: 30 of the 1,350 learned groups (9 of the 515) have `s ≤ 1`, beyond the inits'
bar but not their own Gaussian's. By token most are lexical-semantic classes, many already at
L1–2: {years, year, months}, {novelist, poet, novel, literature, collection, poems}, {never,
Never, not, always}, {was, is, were}, {did, does, do}, {sister, siblings, sisters}, {king,
gods, god}, number tokens in `latex_monograph`, French function words in `camus_letranger`.
Learned and replicating groups **thin with depth** (centred size 2: 36 % / 16 % / 12 % of
groups learned, 61 % / 28 % / 25 % of those replicating, by band): late layers have as many
groups beyond their covariance (`s > 1`), but fewer beyond what an init's cloud reaches.
Seeds 3 and 4 learn about twice as many groups at L17–24 (centred size 2: 129, 111 against
52–69); their hits on seed 0's groups are like the rest's.

**What this answers, and what it does not.** By the outcome table (`design-1d.md` "Unit 2"):
trained beyond the inits and replicating across seeds, in every band, so learned structure
is there; per group it is concentrated early. Many replicating groups are lexical-semantic
classes present at L1, which suggests the embedding's word classes; *revised after
`/challenge-pr` on #132, finding 2:* Pythia's embedding carries no position, so a pure
embedding class should have chance-level position spread, and these are tighter than chance.
~~So the groups are not simply the embedding's: L1 has had one attention layer.~~ *Retracted
after `/challenge-pr` on #133, finding 2:* the inference does not hold. Candidates carried
unchanged from L0 are also tighter than chance (`position_tightness`: median spread
percentile 0.23, 16 of 71 in the tightest 5 %), because text puts related words near each
other. Position spread cannot tell an embedding class from a formed one; the origin reading
can ("The candidates, and where each comes from"). **Open (answered there):**
which groups exist at L0 (the embedding output, in the same forward pass), which L1 adds, and
which form later. A group present at L1 and kept is not by itself evidence of clustering by
attention over depth.

**Caveats.** (1) The first check's late lean (`ci2` raw L17–24 0.171 of 0.20) is where the
late cells are read; the late `ci2` excess (repl 49–54) could carry some of it. (2) The
per-cloud rule saturates: at N = 40, p ≤ 0.05 is "at most one re-init as far", and the
trained `nn1` is 22–28 SD past the re-inits (median, by band and frame), so the rule says trained ≠ init and nothing finer.
(3) Replication matches groups at the same (prompt, layer, frame, size) only. (4) 410m only;
v1's 7 prompts. (5) `group_bars` used numpy's linear quantile, which is NaN with 3+ infinite
re-init maxima; none occurred (0 of 45,231 group rows), and it now refuses (#132 finding 3).

**Re-run:** `data/p1d/arch_null_trained_2026-10-02/run_trained.sh` from the worktree root
(resumable: `run` step 0, `check --union trained`, `run` step143000, `read`; norms from
`arch_null_2026-10-02/`). Tests: `tests/test_phase1d_arch_null.py`. **Next:** unit 1's groups
that move ∩ this unit's learned and replicating groups (the candidate definition in the
outcome table), with a per-layer origin reading **from L0** (the embedding output) so the
embedding's groups are told apart from groups formed with depth, and the 15 bulk groups
reported apart; then unit 3. *Done: next section.*

### The candidates, and where each comes from (2026-10-02; branch `claude/p1d-origin-reading`)

**Built, in this order:** the readings (`design-1d.md` "The candidates, and where each comes
from", `fbea433`, before any run); `move_text run --kept-from` and `candidates.py` (`l0`,
`read`; `a768b5b`). **Inputs:** 410m seed 0, step0 and step143000; the 7 v1 passages (first
512 tokens) on **unit 2's trained token set** (`arch_null_trained_2026-10-02/step0/token_sets.json`,
sha256 `b1eaa3abb679b36d`); unit 2's records as they are; L0–24, both frames, sizes 2 and 4.
Output `data/p1d/candidates_2026-10-02/` (`run_candidates.sh`, `unit1/` (the re-run),
`l0_*.json`, `candidates.json` md5 `575c9ae4`, `candidate_rows.json`, `run.log`); ~20 min.

**Unit 1 re-run on that set** (it drops each prompt's first delimiter too). Both first checks
pass again: step 0's opening is opening-bound in 7 of 7 passages; the designed records were
copied unchanged (their set is not in unit 2's comparison). Share of classified groups that
move, centred size 2 by band: trained 0.95 / 0.90 / 0.71 (was 0.95 / 0.89 / 0.74), step 0
0.23 / 0.10 / 0.04 (was 0.28 / 0.09 / 0.08). **Joins hold exactly:** every one of 2 × 7 × 96
records has the same member sets in unit 1 and unit 2; L1 from `l0`'s forward pass equals
unit 1's; the 515 replicating groups are reproduced.

**Unit 1 class of the learned, replicating groups** (seed 0, trained, centred size 2, not bulk;
group-layer records, pooled over prompts):

| band | learned + replicating | moves | floor 0 / unstable | preamble-dep. / context-bound | not learned: moves / classified |
|---|---|---|---|---|---|
| L1–8 | 170 | **130** | 17 / 16 | 4 / 3 | 136 / 145 |
| L9–16 | 41 | **38** | 1 / 0 | 0 / 2 | 236 / 265 |
| L17–24 | 17 | **13** | 1 / 3 | 0 / 0 | 113 / 166 |

Nearly every learned, replicating group that unit 1 can classify moves (181 of 190). So does
most of everything else: **"moves" removes almost nothing among stable trained groups.** The
candidates are, in effect, the learned, replicating groups that have a non-zero noise floor.

**Origin of the candidates** (centred size 2; distinct = distinct member sets per prompt, each
at its smallest ℓ₀):

| | carried (ℓ₀ = 0) | formed at L1 | formed later (ℓ₀ ≥ 2) | all |
|---|---|---|---|---|
| distinct candidates | **71** | 4 | 45 | **120** |
| … first a candidate at L1–8 / L9–16 / L17–24 | 71 / 0 / 0 | 4 / 0 / 0 | 16 / 20 / 9 | 91 / 20 / 9 |
| group-layer records, L1–8 / L9–16 / L17–24 | 108 / 3 / 0 | 4 / 2 / 0 | 18 / 33 / 13 | 130 / 38 / 13 |
| distinct, other arms: centred 4 / raw 2 / raw 4 | 7 / 25 / 0 | 2 / 0 / 3 | 12 / 54 / 21 | 21 / 79 / 24 |

By prompt: `wiki_paragraph` 45, `hdbscan_code` 19, `homer_iliad` 17, `sullivan_ballou` 12,
`latex_monograph` 12, `paper_excerpt` 11, `camus_letranger` 4. Carried candidates are the
embedding's word classes: {Charlotte, Emily, Anne, Jane, Maria}, {sister, siblings, sisters},
{six, five, three, two}, {was, is, has, were, became}, number tokens. As records they persist
a median 8 layers (run ends at L7, median).

**Step 0, beside:** 0 replicating groups, so 0 candidates. Of its 647 centred size-2 groups at
L1–8, **2 are carried** (73 formed at L1): at init, L0's groups do not survive one block. So
"carried" is a trained property (the trained embedding's classes kept through the trained
stack), not something any init does.

**Reading: design outcome row 1, "most carried"** (71 of 120 = 59 %; 120 ≥ 20, so not "few").
In the primary arm the candidate definition mostly picks out the embedding's word classes,
carried through L1–8. The 49 formed at L1 or later sit where the carried ones stop; that
every candidate first seen at L9–24 (29) is formed later is close to forced by the rule
(presence at every layer back to L0; only 32 of 882 trained L9–16 records are carried at all;
`/challenge-pr` on #133, finding 4).

**The verdict turns on replication** (*added after `/challenge-pr` on #133, finding 1*; the
rows below the first were computed after the run, from `candidate_rows.json`, with the same
code). Unit 2's outcome table says "moves ∩ learned"; this unit's readings added "replicates"
so a definition carries across seeds. Distinct, centred size 2, not bulk:

| definition | distinct | carried | formed at L1 | formed later | outcome row |
|---|---|---|---|---|---|
| moves ∧ learned ∧ replicates (**fixed primary**) | 120 | 71 | 4 | 45 | most carried |
| moves ∧ learned (the outcome table's words) | 205 | 81 | 8 | **116** | **most formed later** |
| learned ∧ replicates (any unit 1 class) | 147 | 89 | 6 | 52 | most carried |

Replication is what selects the embedding's classes. At L1–8 the share carried rises with
the filter: 12 % of groups that are not learned, 29 % of learned-only, 82 % of learned and
replicating (reviewer's count). Groups formed with depth are mostly seed-specific. So
"most carried" is a statement about the definition *across seeds*. A definition for one
seed's checkpoints (Phase 10 reads `pythia-410m` only) has 124 formed groups, not 49.
**Which set Phase 10 re-reads is the user's** (`STATE.md` Blocked 14).

**Caveats, in order of weight.**
1. **"Formed later" includes carried classes that grew.** 14 of the 45 were present at L0 and
   broke before re-forming. Among formed-later records, 77 % were present (Jaccard ≥ 0.5) at
   some layer before their origin. The examples are the carried classes plus new members:
   {novelist, poet, novel} + {published, literature, collection}. A fixed Jaccard bar of 0.5
   splits a group that gains members from its own past, so the 45 are an upper bound on groups
   formed from nothing. A containment reading (is the L0 class a subset?) is not built.
2. **The reading depends on the frame.** In raw size 2 the formed groups are the majority
   (54 of 79, 25 carried). The design fixed centred size 2 as primary, and the reading is
   stated on it; it is not frame-free.
3. "Moves" barely filters (above). The candidate set rests on unit 2's learned and
   replicating rules and on unit 1's noise floor, not on movement: at L1–8, `floor 0` and
   `unstable` remove 33 of 170 learned, replicating groups, "moves" another 7.
4. One seed's groups (seed 0) on 7 prompts; group-layer records are dependent, and the
   distinct counts are not independent either (one class can appear in several prompts).
   `wiki_paragraph` holds 45 of the 120.

**Re-run:** `data/p1d/candidates_2026-10-02/run_candidates.sh` from the worktree root
(resumable). Tests: `tests/test_phase1d_candidates.py`. **Next: unit 3** (one family on a
continuous scale), which reads the candidates' scale. Per the outcome table, Phase 10's rows
are re-read with formed candidates kept apart from carried ones.

### Unit 3: the synthetic and its first check (2026-10-04; branch `claude/p1d-unit3-scale`)

**Built, in this order:** the readings (`design-1d.md` "The synthetic and the first check",
`9d486c6`, before any run); `scale_spectrum.py` and `tests/test_phase1d_scale_spectrum.py`
(`ede7a05`). **Input:** the synthetic only, seed 0 (no model, no prompt). Output
`data/p1d/scale_synthetic_2026-10-04/` (`first_check.json` at `ede7a05`, `run.log`; post hoc:
`posthoc.json` at `b7c1473`, `posthoc.log`); 50 s + 77 s. **Re-run:** `python -m
p1d_cluster_ensemble.scale_spectrum synthetic --out <dir> --seed 0`, then `... posthoc --out
<dir> --seed 0`, from the worktree root. *(The post hoc numbers first came from an untracked
script under `data/`; moved into `scale_spectrum.posthoc` after `/challenge-pr` on #134,
finding 2, and re-run: same numbers.)*

**Construction probe, run before the readings were fixed** (the opening alone, no unit 3 code;
fixed-angle stand-in for vMF; centred; mean pairwise cosine distance). It set `t`:

| `identity_sim` β = 0, t | 0 | 0.5 | 1 | **2** | 4 | 8 |
|---|---|---|---|---|---|---|
| positions 1–4 | 0.87 | 0.74 | 0.60 | **0.32** | 0.04 | 0.00 |
| within a sub-group | 0.19 | 0.20 | 0.20 | **0.27** | 0.73 | 0.93 |
| between sibling sub-groups | 0.52 | 0.53 | 0.53 | **0.57** | 0.82 | 0.93 |

At t ≥ 4 every row is pulled onto the prefix mean and the planted groups are gone. The
residual form `x + λ·mean(prefix)` never put positions 1–4 below a sub-group's spread for
any λ ≤ 8 (0.40 against 0.24 at λ = 4). On the run's vMF synthetic at t = 2: opening 0.39,
within 0.25, between 0.55 (centred), so the rule holds. In raw the cloud collapses (median
0.15, within 0.03, between 0.07, opening 0.20).

**Result: the first check FAILS.** Centred, neither planted scale is found; raw, neither; the
5 Gaussians have 0 plateaus in both frames. **(b) fails, and it is (b)'s direction:**

| centred, t = 2 | r | k_sub | stability | Gaussian draws' k_sub (mean ± SD) | rank p |
|---|---|---|---|---|---|
| fine (9 sub-groups + 1) | 0.32 / 0.37 / 0.42 / 0.47 | 10 | 0.98 / 0.96 / 0.94 / 0.95 | 4.8 / 8.0 / 11.0 / 12.3 (± 1.4–1.7) | 0.02 / 0.20 / 0.82 / 0.94 |
| coarse (3 groups) | 0.79 / 0.90 / 1.02 | 3 | 0.98 / 0.98 / 0.90 | 14.0 / 8.5 / 2.8 | 1.0 / 1.0 / 0.77 |

Once the cut is past the scale where the matched Gaussian has no substantial cluster, the
Gaussian soon has **as many or more** substantial pieces than the planted structure (centred
8–15 from r 0.37 against 10, and 8–14 against 3; raw 17–30 against 3). A higher substantial
count at a fixed relative cut does not mean lumpier, so (b)'s higher tail could not have
accepted either planted scale. p ≤ 0.05 holds only at small r, where the Gaussian has none.
**This is the design row's defect, not only the synthetic's stand-in:** on real input (b) is
the count's `z_G` ranked among re-inits, and a cloud with fewer, tighter groups than its
Gaussian gets a *lower* `z_G` there too. Stability has the right direction: synthetic
0.94–0.98 against the 5 Gaussian clouds' 0.43–0.83 at r 0.32–0.90 (where they have ≥ 2
substantial clusters). At r 0.25–0.28 the Gaussians reach 0.84–0.87, so the margin is
smaller at fine scales.

**Post hoc, not a pass** (computed after the fail, same seeds; `posthoc.json`):
without (b), the robust runs and their minimum ARI over the run.

| synthetic | runs (k_sub, r) | min ARI of each run to its planted scale | both scales? |
|---|---|---|---|
| t = 2, centred | 10 at 0.32–0.47; 3 at 0.79–1.02 | fine 0.92; coarse 0.99 | yes |
| t = 2, raw | none (each planted count holds 2 grid points) | — | no |
| t = 0, centred | 9 at 0.22–0.42; 3 at 0.54–1.02 | fine 1.00; coarse 1.00 | yes |
| `d_f` = 0.20 / 0.25 / 0.30 | 3 at 0.79–1.02 / 4 at 0.61–0.90 / 4 at 0.69–0.90 | coarse 0.93 / 0.93 / 0.94 | coarse only |

So without (b): the centred frame passes on seed 0; the opening costs the fine run its lower
end (0.22 → 0.32) and adds a 10th cluster; the fine scale is lost once `d_c / d_f` ≤ 2 (with
the opening); the raw frame cannot be read at t = 2.

**Option 1 on seed 0, post hoc** (*added after `/challenge-pr` on #134, finding 1*; the same 50
draws, each with its own 50 subsamples). A draw whose cut has no substantial cluster has no
stability; two rules: **zero** (it scores 0) or **drop** (left out; p = 1 if none left).

| centred, t = 2 | zero | drop |
|---|---|---|
| runs found (k_sub, r) | 10 at 0.32–0.47; 3 at 0.79–1.02 | the same |
| both planted scales | yes | yes |
| p at r 0.12–0.19 (Gaussian draws have no substantial cluster) | 0.02 | 1.0 |
| p at r 0.25 / 0.28 / 0.32–0.61 | 0.33 / 0.08 / 0.02 | 0.63 / 0.09 / 0.02 |

Raw: nothing under either. **At matched count** instead of matched r (each draw's largest
stability at any r where it has the same k_sub): k = 10, 25 of 50 draws reach that count, max
0.82, median 0.75, 0 at or above the synthetic's weakest 0.94; **k = 3, 43 draws, max 0.93,
median 0.65, 2 at or above the synthetic's weakest 0.90** (at r 1.02, where the background
has merged in). So at matched count the coarse margin is thin; at matched r it is not.

**Options (`STATE.md` Blocked 15; the user's):**

| option | what | for | against |
|---|---|---|---|
| **1 (recommended)** | (b) := the cut's mean cluster-wise stability, rank p (higher tail) against the Gaussian draws' stability at the same r, **empty draw scores 0**; on real input its `z_G` ranked among the re-inits' as designed | the direction is "more cluster-like than the covariance" by construction; on seed 0 it finds both scales; "zero" keeps the small-r signal where only the cloud has clusters ("drop" gives p = 1 there) | stability on every draw: ~50× the cost per cloud at 50 × 50 (20 draws × 20 subsamples ≈ 8×). At matched count the coarse margin is thin (2 of 43); same r is the design's comparison, matched count would be a second reading beside it |
| 2 | drop (b) from the plateau rule (count + stability only); read "learned" per plateau among re-inits apart | simplest; seed 0 passes post hoc | "beyond covariance" then rests on the run length and count constancy, i.e. on the Gaussian check alone |
| 3 | two-sided count p | — | rejected: a count that differs either way is not a direction |

**For either option** (*added after `/challenge-pr` on #134*): the first check is re-run on
**seed 1** with the rule fixed first (seed 0 has been seen); the Gaussian negative check takes
**50** Gaussian clouds, not 5 (0 of 5 only bounds the false-plateau rate below ~45 %; the check
costs ~3 s per cloud; finding 3), with a pass bound placed before the run. Raw stays beside: at
t = 2 it cannot pass under any of the three. **Open, the user's (finding 4):** the plateau
count reads clusters of ≥ `SUBSTANTIAL_CLUSTER_SIZE` = 4 tokens, but the candidates are
level-set groups from size 2 (examples have 3 tokens) and the planted sub-groups have 33. So
unit 3 as designed asks at what scale the cloud's large structure sits, not at what scale
the candidates live. A size-2 arm (k counted from size 2) or a per-candidate reading (the
range of r over which a candidate is one cluster of the cut) would ask the second; neither is
in the design.
*Blocked 15 decided 2026-10-04 (option 1, ≤ 2 of 50, a size-2 arm); the re-run is the next section.*

### Unit 3 re-run: seed 1 under Blocked 15 (2026-10-04; branch `claude/p1d-unit3-rerun`)

**Built, in this order:** the rules (`design-1d.md` "The re-run", `a2afa0f`, before any code);
`scale_spectrum.py` with option 1's (b) inside the spectrum, both arms, the `|C ∩ S| ≥ 2` rule,
50 Gaussian clouds run in parallel (`c7137b9`, before the run). The seed-0 post hoc code is
removed (numbers above; code at `b7c1473`). **Input:** the synthetic only, **seed 1** (unseen
before this run), 50 draws × 50 subsamples per cloud, 106 clouds. Output
`data/p1d/scale_rerun_seed1_2026-10-04/` (`first_check.json` at `c7137b9`, `run.log`); 282 s
on 14 workers. **Re-run:** `python -m p1d_cluster_ensemble.scale_spectrum synthetic --out <dir>
--seed 1 --jobs 14` from the worktree root.

**Result: FAILS, both arms, and not on (b).**

| centred, seed 1 | main arm | size-2 arm |
|---|---|---|
| planted scales found | neither | neither |
| robust plateaus (k, r) | none | (2, 0.04–0.05), ARI 0 (the opening) |
| Gaussian clouds with a plateau | **0 of 50** | **0 of 50** |
| the same without (b) | 1 of 50 | 2 of 50 |
| beside, t = 0 (no opening) | both found: 9 at 0.28–0.47, 3 at 0.54–1.02 | both found |
| beside, `d_f` 0.20 / 0.25 / 0.30 | none / coarse (4 at 0.69–0.90) / none | none |
| raw | coarse found (5 at 0.47–0.69); 0 of 50 Gaussians | none |

**Why (main arm, centred).** (b) and stability are not what fails: p = 0.02 and stability
0.90–0.99 at every r from 0.28 to 1.02 except 0.25. Each planted partition is there. Fine ARI
0.87 / 0.90 / 0.90 / 0.90 / 0.80 at r 0.28–0.47; coarse 0.91 / 0.88 / 0.88 / 0.86 / 0.86 at
r 0.61–1.02. **The substantial count never holds for 3 points there:** 13, 12, 12, 10, 9
(fine); 4, 3, 4, 3, 3 (coarse). On seed 1 the opening is far tighter than on seed 0 (positions
1–4 at 0.14 centred, against 0.39; within a sub-group 0.27). It pulls 2–4 early tokens out of
several sub-groups into fringe clusters of 4–10 tokens, which merge into sub-groups or cross
the size-4 bar between consecutive cuts (at r 0.79 a 4-token fringe of sub-groups 3 and 5
appears, then merges). The size-2 arm counts more of these and churns more. **The defect is
the "robust plateau" row's "same count"**: a count is not a partition, and small clusters
move it while the partition barely changes. Real clouds have such a fringe too (token 0's
neighbours, the opening), so the rule is not only the synthetic's problem.

**Post hoc, not a pass** (seed 1, now seen; 3 of the 50 Gaussian clouds): the ARI between
consecutive cuts (all tokens). Synthetic, steps from r 0.28 to 0.47: 0.965, 0.999, 0.943,
**0.896**; from r 0.61 to 0.90: 0.964, 0.990, **0.879**. Gaussians: 0.37–0.82 at every r from
0.22 to 0.90. *(First written as "0.90" for the fourth fine step; corrected after
`/challenge-pr` on #135, finding 2: two steps in the planted ranges fall below 0.9, not one.)*

**What `/challenge-pr` on #135 added (accept with changes; reviewer's checks, not re-run here):**
(1) the opening's tightness varies about 3× across seeds: 0.20–0.40 centred on 10 synthetic-only
seeds (90001–90010; no spectrum read), seeds 0 and 1 near the ends, about 1 in 3 tighter than a
sub-group, so one fresh seed is n = 1 either way. (3) consecutive-cut ARI is nearly blind to small
clusters (30 new pairs move it by 0.004), depends on the grid spacing, and lets a run drift (one
merge of two sub-groups scores 0.89 per step while the run's ends are 0.44 alike). (4) the largest
"fringe" cluster at the fine scale (10–11 tokens, the earliest positions) is the opening group,
which unit 4 says the synthetic forms; the planted labels leave it out, so the cloud really has
12–13 groups there. (5) in the size-2 arm at r ≤ 0.13 none of the 50 draws has a pair, so any
pair passes (b) and `z_G` is None; on real input repeated tokens make such pairs.

**Options (`STATE.md` Blocked 16; the user's), revised after #135's review:**

| part | recommendation | alternatives, and why not |
|---|---|---|
| continuity, main arm | every cut of the run has **ARI ≥ 0.9 to the run's first cut** (all tokens), k ≥ 2 at every point; stability and (b) unchanged | consecutive-cut ARI (the first recommendation): allows drift (finding 3); count only stable clusters: the opening's fringe is tight, may not fix it; a relative size bar: defeats the size-2 arm. The 0.9 is placed after seeing seed 1 |
| size-2 arm | **beside, not gating**, until a per-cluster continuity rule (each cluster's Jaccard across the run) and a planted small-group control are designed | ARI cannot see pairs (finding 3) |
| the opening | the fine planted labels gain an **"opening" label** for the earliest kept positions whose distance to the prefix mean is below the planted within-sub-group spread, fixed per seed from the construction before any spectrum | leave it out (scores real structure as error); drop those tokens from the ARI |
| (b) where the draws are empty | a grid point where fewer than 5 of 50 draws have a cluster of the arm's size has (b) **uninformative** and cannot be in a plateau; on real input such a scale is reported, not ranked (`z_G` None) | "zero" there (passes any pair, finding 5) |
| seeds | **10 fresh seeds (2–11)**; pass: both scales found on **≥ 8 of 10**, and ≤ 2 of 50 Gaussian clouds with a plateau (5 per seed) | seed 2 alone (n = 1, finding 1); 50 Gaussians per seed (10× the cost, same bound) |
| rejected | a gentler opening (t = 1): changes the synthetic until the instrument passes | — |

*Decided 2026-10-04 (user): all five parts as recommended. They go into `design-1d.md` before any code.*
*Ran: next section.*

### Unit 3 on seeds 2–11 under Blocked 16 (2026-10-04; branch `claude/p1d-unit3-seeds`)

**Built, in this order:** the five parts (`design-1d.md` "The multi-seed run", `5644f90`, before
code); `scale_spectrum.py` (`e6390fc`: anchored-ARI plateaus, informative (b), `opening_extent`
/ `planted_labels`, the main arm gating, the 10-seed driver, the Clopper–Pearson bound) and
tests, before the run. **Input:** the synthetic only, seeds 2–11 (unseen), 50 draws × 50
subsamples per cloud, 160 clouds. Output `data/p1d/scale_seeds2-11_2026-10-04/`
(`first_check.json` at `e6390fc`, `run.log`); 721 s on 14 workers. **Re-run:** `python -m
p1d_cluster_ensemble.scale_spectrum synthetic --out <dir> --jobs 14` from the worktree root.

**Result: FAILS, 7 of 10 seeds** (bar 8). Specificity holds.

| | main arm, centred (gates) | size-2 arm, centred | main arm, raw |
|---|---|---|---|
| both planted scales | **7 of 10** (seeds 2–6, 10, 11); rate ≥ 0.39 at 95 % | 4 of 10 | 0 of 10 |
| coarse / fine alone | 7 / 8 | 7 / 4 | 9 / 0 |
| Gaussian clouds with a plateau | **0 of 50** (rate ≤ 0.058 at 95 %) | 0 of 50 | 0 of 50 |
| the same without (b) | 0 of 50 | 1 of 50 | 0 of 50 |

Opening extent J per seed (2–11): 5, 5, 3, 0, 9, 21, 0, 12, 6, 0; the opening's mean pairwise
distance 0.09–0.46 against within 0.25–0.30.

**Why the three seeds miss (post hoc, seeds 7–9, trees only):**

| seed | coarse | fine |
|---|---|---|
| 7 | the planted partition holds from r 0.79 (planted-only consecutive ARI 1.00), but the background's merges at r 0.90 drop the all-token ARI to 0.73, leaving 2 points | **(b)**, not resolution: the cuts at r 0.32–0.42 match the planted partition (ARI 0.80–0.82) and hold together, but (b)'s p is 0.078 at 0.32, so the run is 0.37–0.42, 2 points *(corrected after `/challenge-pr` on #136, finding 1; first written as a 2-point window)* |
| 8 | same, all-token ARI 0.75 at 0.90 | (b) admits from 0.365; fine ARI ≥ 0.8 only at 0.37–0.42: 2 points |
| 9 | same, 0.75 at 0.90 | found (0.32–0.42) |

So, on these seeds, the planted ranges are 3–4 grid points wide after the opening (centred within ≈ 0.27,
between ≈ 0.57, about 6 grid steps apart, of which the partition is exactly the planted one
over 3–4). `MIN_RUN` = 3 sits at the instrument's resolution. The construction placed the
spreads "so each plateau spans ≥ 4 grid points" by construction distances; that was never
measured on the cuts.

**Post hoc, not a pass** (seeds 2–11, seen; the same rows, trees recomputed): continuity over
the tokens in the arm's clusters at the run's first cut (background singletons joining later
do not count): **both scales on 8 of 10** (coarse 10, fine 8; seeds 7, 8 still lose fine),
Gaussians 0 of 50. At the bar, not above it: at a true rate of 0.8, ≥ 8 of 10 fresh seeds
happens with probability 0.68 (0.38 at 0.7, 0.93 at 0.9).

**The control's margin, measured** (*added after `/challenge-pr` on #136, finding 2*, which
showed the first recommendation's r-span ≥ 1.3 on an 80-point grid was stricter than today's
rule, 3 of today's points spanning 1.293, and unmeasured). `scale_spectrum windows` (`a4f72d4`):
per seed, from trees alone, the longest r-interval (400-point grid) whose cut has ARI ≥ 0.8 to
each planted labelling, as a span `r_hi / r_lo`. **Throwaway seeds 1000–1039** (never a check's
seeds); output `data/p1d/scale_windows_2026-10-04/windows.json`.

| scale | min | p10 | median | narrower than 1.47 (a 40-point grid always holds 3 points) |
|---|---|---|---|---|
| coarse | 1.59 | 1.82 | 2.00 | 0 of 40 |
| fine | 0 | 1.53 | 1.83 | **3 of 40**: 2 with no fine window at all (seeds 1006, 1024: the opening destroys the scale), 1 at 1.37 |

So the coarse misses were the all-token ARI alone, never resolution; the fine scale is absent
or narrower than the grid resolves on ~1 seed in 13, a property of the synthetic, not of the
spectrum; and the rest of the fine misses are (b) at the window's edge (seed 7). Seeds 7 and 8
have fine spans 1.385 (reviewer's measurement), under 1.47.

**Options (`STATE.md` Blocked 17; the user's), revised after #136's review:**

| option | what | for | against |
|---|---|---|---|
| **1 (recommended)** | (a) continuity over the tokens in the arm's clusters at the run's first cut; the 40-point grid and `MIN_RUN` = 3 kept; **sensitivity scored on seeds where both planted scales are present at the grid's resolution** (window span ≥ 1.47, measured from trees before any spectrum): seeds from 12 upward, skipping absent ones, until 10 present; pass ≥ 8 of 10 and ≤ 2 of 50 Gaussians, as before | (a) is what the plateau means and needs no planted labels; presence separates what the synthetic offers from what the instrument finds, and 1.47 comes from the grid, not from a failure; it states the instrument's resolution for real input (scales narrower than a span of 1.47 may be missed) | on seeds 2–11 it would drop 7 and 8 (spans 1.385), which failed: it was placed after seeing them, though from the grid's geometry; (b)-edge misses still count |
| 2 | (a) alone, seeds 12–21 as drawn | one change | windows absent or narrow on ~1 seed in 13 and (b) edges count against it: a fresh run passes ~2 times in 3 |
| 3 | accept 7 of 10 as measured sensitivity | specificity 0 of 50 is what guards against false structure | moves the bar after the result |
| 4 | widen the planted gap; an 80-point grid with an r-span bar | — | rejected: the first changes the synthetic until the instrument passes; the second is stricter than today's rule (#136, finding 2) |

*Decided 2026-10-04 (user): option 1. Ran: next section.*

### Unit 3 on fresh seeds under Blocked 17 (2026-10-04; branch `claude/p1d-unit3-blocked17`)

**Built, in this order:** the rules (`design-1d.md` "The fresh-seed run", `2d9e13b`, before
code, with `tools/math_checks/grid_resolution_span.py`: `PRESENT_SPAN` = 150^(1/13) = 1.4703,
the user's "1.47"); `scale_spectrum.py` (`eb0d761`: continuity over the first cut's clustered
tokens, `present_seeds`, refusal past seed 51) and tests. Before the run, the new code on
seeds 2–11 (seen; saved rows, trees recomputed) reproduced the post hoc above: 8 of 10, and
presence drops exactly seeds 7 and 8 (fine spans 1.369 on the 400-point grid). **Input:** the
synthetic only; presence read on seeds 12–22 (trees only); the spectrum on the first 10
present, **13–22** (unseen), 50 draws × 50 subsamples per cloud, 160 clouds. Output
`data/p1d/scale_fresh12_2026-10-04/` (`presence.json`, `first_check.json` at `eb0d761`,
`run.log`); 546 s on 14 workers. **Re-run:** `python -m p1d_cluster_ensemble.scale_spectrum
synthetic --out <dir> --jobs 14` from the worktree root.

**Presence:** seed 12 absent (fine span 0: opening extent J = 24 destroys the fine scale;
coarse 1.476); 13–22 present (coarse 1.72–2.10, fine 1.65–2.02).

**Result: FAILS, 7 of 10 present seeds** (bar 8). Specificity holds.

| | main arm, centred (gates) | size-2 arm, centred | main arm, raw |
|---|---|---|---|
| both planted scales | **7 of 10** (13, 15–20); rate ≥ 0.39 at 95 % | 4 of 10 | 0 of 10 |
| coarse / fine alone | **10** / 7 | 10 / 4 | 10 / 0 |
| Gaussian clouds with a plateau | **0 of 50** | 0 of 50 | 0 of 50 |
| the same without (b) | **29 of 50** | 39 of 50 | 0 of 50 |

The continuity change did what it was for: coarse 10 of 10 (7 under Blocked 16). **Every miss
is fine, at (b)'s lower edge** (post hoc, trees recomputed, saved rows):

| seed | the cut matches the fine labels (ARI ≥ 0.8) | (b) admits | why no run of 3 |
|---|---|---|---|
| 14 | r 0.28–0.47 | 0.32, 0.365, 0.47 (p 0.039, 0.039, 0.020); **0.415 p 0.059** | 2 points, then a gap |
| 21 | r 0.25–0.47 | from **0.365** (0.25–0.32: p 0.12–0.53) | at 0.47 one sub-group merge: continuity 0.887 < 0.9; 2 points |
| 22 | r 0.25–0.47 | from **0.365** (0.25–0.32: p 0.16–0.22) | the same, continuity 0.868; 2 points |

From r 0.28 to 0.32 the matched Gaussian draws' mean stability rises from 0.58–0.70 to
0.83–0.85 (the cloud's 0.88–0.94 at 0.32), so several draws reach the cloud there. (b) shortens the
fine window from its lower end; presence is read on trees and does not see (b). p = 0.039 is
1 of 50 draws at or above the cloud.

**What else the run shows:** (1) **(b) now carries the specificity alone:** without it 29 of
50 Gaussian clouds have a main-arm plateau (0 of 50 under Blocked 16's all-token continuity,
which the change loosened). (2) **Without (b), fine is lost on 8 of 10:** the run anchors at a
cut where the sub-groups are still fragments (only their ≥ 4-token pieces count), keeps the
formed scale (the fragments' tokens stay together), and so fails "ARI ≥ 0.8 at every point" on
its first point. Seen first on seed 0 at t = 0 (`tests/test_phase1d_scale_spectrum.py`, the
count-tail test, changed to say so). With (b) the fragment cuts are not admissible, so the
gating arm is not hit by it on these seeds; on real input it would be wherever (b) admits a
fragment cut. (3) Seeds 2–11 (seen, post hoc) and 13–22 together: 15 of 18 present seeds,
but only the second 10 are a check.

**What `/challenge-pr` on #137 added (accept with changes; reviewer's checks, not re-run
here):** (1) the new continuity rule cannot see clusters forming, only the first cut's
clusters merging, so a Gaussian run can grow from k = 2 to 15; hence the 29 of 50. The
synthetic's 0 of 50 is (b) **ranked among Gaussian draws**; on real input (b) is `z_G` ranked
among unit 2's re-inits (`design-1d.md` "The re-run", row (b)), so the synthetic's specificity
does not carry over by itself. A rule ending a run when a new cluster forms restores 0 of 50
without (b) but drops fine to 5 of 10 with it: a trade-off. (2) The same 29 of 50 holds on
seeds 2–11 under the new rule; the pre-run replay did not look (`LESSONS.md` 11). (3) Option 4
as first written needed 17 of 20 (16 gives a bound of 0.599), which passes with probability
0.41 at a true rate of 0.8, not easier than 8 of 10 (0.68): corrected below. (4) Every miss
cuts the fine window from below, so on real input a found plateau's lower end is pushed up:
**the partition is the claim, not its range of r.**

**Options (`STATE.md` Blocked 18; the user's), revised after #137's review.** The number
that should settle it: **the fine-scale sensitivity the real-input question needs.** Measured:
7 of 10 (rate ≥ 0.39 at 95 %).

| option | what | for | against |
|---|---|---|---|
| **1 (recommended)** | accept the sensitivity as measured, with its limits stated: coarse 10 of 10, fine 7 of 10 (rate ≥ 0.39 at 95 %), every miss at (b)'s lower edge; a scale narrower than a span of 1.47 may be missed; a found plateau's partition is the claim, not its r range. **A found plateau is the claim; a missing one is weak evidence of absence.** **Condition (#137 finding 1):** the real-input reader's first step is its own negative check, specificity of (b) as ranked among the re-inits (e.g. each re-init read as a cloud, ranked among the others), with a bound placed before it runs | misses are false negatives, the conservative direction; each fix since Blocked 15 met a new failure on fresh seeds; at a true rate near 0.8 the bar of 8 of 10 passes with p ≈ 0.68, so another run is close to a coin | moves the bar after the result (Blocked 17's option 3, rejected then); the synthetic's specificity is for the Gaussian-ranked (b) only |
| 2 | (b) tolerates one inadmissible point inside a run, then seeds 23 up | one change | fixes seed 14 only; 21 and 22 end on continuity at r 0.47 |
| 3 | more draws (200) so (b)'s p at the edge is resolved | p 0.039 is one draw | the edge is real (the Gaussians are stable there); 4× the cost; a true p near α stays near α |
| 4 | a larger check: 20 present seeds, pass if the 95 % lower bound on the rate is ≥ 0.6, i.e. **≥ 17 of 20** | a bound on the rate, not a point count | *corrected after #137 finding 3:* **harder** than 8 of 10 (passes with probability 0.41 at a true rate of 0.8, against 0.68); 2× the cost; a new bar, placed after this result |
| 5 (#137 finding 1) | continuity also ends a run when a new cluster of the arm's size forms | restores 0 of 50 Gaussians without (b) (reviewer) | fine drops to 5 of 10 with (b) (reviewer); another rule after a result |

**Decided (user, 2026-10-04): option 1**, the sensitivity accepted as measured with its limits,
on the condition. Limits and the specificity check's rules: `design-1d.md` "Blocked 18
decided" (docs only, nothing run). **Next: the real-input reader, step 1**: specificity on
the 50 step-0 models (10 real inits ranked among the 40 re-inits; each re-init among the
other 39), centred; pass: ≤ 5 % of re-init clouds with a plateau in every band (else
`MIN_RUN` raised, up to 5), and the real inits within the re-inits' 10-model subsets; ~2 h
(estimate). The trained reading waits on it. Revised after `/challenge-pr` on #138 (all five
findings taken: no-cluster references count below; rate gated on the re-inits, failure
response placed; cost; sensitivity is the synthetic reader's, the real reader's is unit 4's).

### The real-input reader, step 1: built; the pilot shows step-0 clouds cannot fail it (2026-10-04; branch `claude/p1d-specificity`)

**Built:** `scale_real.py` (`run`: forward each step-0 model, `scale_spectrum.spectrum`
per centred cloud, seed per (prompt, layer) in every model, rows and cut labels stored per
(model, prompt); `read`: (b) re-ranked among the reference set with the design's missing-z
rules, plateaus, the per-band rate pass with `MIN_RUN` raised to 5, the 1,000-subset route
test, the beside rows; `resolution`: trees only, below), `robust_plateaus(min_run=)`, and
`tests/test_phase1d_scale_real.py`. **Not run:** the 50-model batch. **Input:** the kept
tokens of `data/p1d/arch_null_trained_2026-10-02/step0/token_sets.json` (sha256 prefix
`b1eaa3abb679b36d`), the 7 v1 prompts, centred.

**Pilot (the design's order row):** `reinit:0` × `wiki_paragraph` × L1–24, 30 s on 14
workers (so the batch is **~3 h**, not ~2 h). Output present, but **`z_G` is defined at 1–4
of 40 grid points per layer** (main arm), not at every `r`: below r ≈ 0.7 every token is a
singleton, above ≈ 1.0 the cloud is one cluster. That is the cloud, not a bug: step-0 tokens
are nearly orthogonal (median centred cosine distance 1.006), so the whole merge tree sits in
a few grid points of `r × median`.

**Resolution, trees only** (`scale_real resolution`; `data/p1d/scale_real_pilot_2026-10-04/
resolution.{json,log}`): per cloud, the grid points whose cut has ≥ 2 main-arm clusters; a
plateau needs `MIN_RUN` = 3 of them in a row, so "≥ 3 points" is an upper bound on room for one.

| model | L1–8: median points (max); clouds with ≥ 3 | L9–16 | L17–24 | non-trivial cuts span r |
|---|---|---|---|---|
| reinit:0, reinit:1 @ step 0 | 1 (2); 0 of 56 | 1 (2–3); 0–1 of 56 | 2 (3); 3–7 of 56 | ≈ 0.5–1.0 |
| init:0, init:5 @ step 0 | 1 (1); 0 of 56 | 1 (2); 0 of 56 | 2 (3); 1–3 of 56 | ≈ 0.54–1.0 |
| init:0, init:5 @ step143000 | 3–4 (7); 50–56 of 56 | 4–5 (8); 56 of 56 | 9–10 (16–22); 56 of 56 | ≈ 0.09–1.02 |

**What this means.** (1) The step's rate pass (≤ 5 % of re-init clouds with a plateau) is
met by the grid alone in L1–16 and nearly so in L17–24, whatever (b) does: the check cannot
fail where it is meant to test (b). (2) The trained reading meets the mirror image: at
r ≲ 0.5 every re-init is all singletons, and a reference with no cluster counts as below
(#138 finding 1), so any trained cluster there gets p = 1/41 and (b) filters nothing at the
trained clouds' finer scales (the synthetic's "without (b)": 29 of 50 Gaussians with a
plateau). The relative grid (#129 finding 1) puts each cloud on its own median, but a step-0
cloud's distances have almost no spread around it. Classified as a **defect of the step's
design** (a check that could not have rejected; `LESSONS.md` 6), so the batch was not
launched.

**Options (`STATE.md` Blocked 19; the user's).**

| option | what | for | against |
|---|---|---|---|
| **1 (recommended)** | grid on each cloud's own merge range: the 40 cut heights log-spaced between the cloud's first and last non-trivial merge (absolute δ stored as now); then re-run the synthetic first check on seeds 13–22 (~10 min, the reader changed) before this step (~3 h) | step-0 and trained clouds get the same number of grid points across their structure, so (b) among re-inits compares like positions and the specificity check can fail | another reader change after a result; `PRESENT_SPAN` is defined on the `r` grid and must be re-derived; scales are then positions in each cloud's own range, not a shared `r` |
| 2 | keep the grid; (b) needs ≥ 4 of the references (10 %, as (iv)) with a cluster of the arm's size, else the point is not admissible | small; mirrors (iv); honest about what the re-inits can say | the specificity check still passes by construction; the trained reading is then readable only at r ≈ 0.5–1.0 (2–3 points), so unit 3 against re-inits reads almost nothing trained |
| 3 | run as designed (~3 h) and report the pass with this bound beside it | no change; stores the re-init `z_G` the trained reading needs | a pass that says nothing about (b); the trained reading's (b) stays empty below r ≈ 0.5 |

**Re-run:** pilot, `python -m p1d_cluster_ensemble.scale_real run --out <dir> --only reinit:0
--keys wiki_paragraph --workers 14`; probe, `... scale_real resolution --out <dir> --models
reinit:0@step0 init:0@step143000 ...`, from the worktree root with the local-box env
(`STATE.md` "Machine").

## Deleted and restored (was `FROZEN.md`)

Code deleted 2026-09-23 in a branch cleanup that should have skipped it
(`LESSONS.md` lesson 12); the docs were kept in `archive/p1d_cluster_ensemble/`
from 2026-09-24 and moved back here on 2026-09-25. The rule then (user,
2026-09-24): code may go if its intent stays. `FROZEN.md` asked a rebuild to
"first measure whether tuning reduces *that* drift" (the float-noise one);
that is still the first question.

## As built in August

**State then:** all five sub-experiments implemented and validated on synthetic data with known
answers, with a driver (`run_1d.py`) and artifact IO (`p1d_io.py`) that have been run end to
end against a synthetic Phase-1 run directory. **Not yet run against Pythia artifacts** — no
result rows below, by design. Predictions P-C1, P-C2, P-C3 and P-C4 were written in the
branch's `PREDICTIONS.md` before this code existed (never entered in the registry).

**Cost:** [R] throughout. Reads `activations.npz` and re-clusters; no weights, no forward
pass. Runnable against any existing Phase 1 run directory today.

**Why it exists:** every cluster-conditioned result in this project rests on
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")` — a library minimum and a set of
library defaults. Phase 1's three other partitions are equally untuned, so the existing
cross-method agreement statistic compares four sets of defaults rather than four methods. See
`design-1d.md`.

## Implemented

| Sub-exp | Module | Cost | State |
|---|---|---|---|
| A — tune every family per layer | `selection.py` | [R] | implemented, validated |
| — method registry and grids | `methods.py` | — | implemented, validated |
| — Phase 1 constants, read not copied | `constants.py` | — | implemented, validated |
| B — consensus and calibrated confidence | `ensemble.py` | [R] | implemented, validated |
| C — shipped partition and its refusals (P-C2, P-C3) | `comparison.py` | [R] | implemented, validated |
| D — persistence prediction (P-C4) | `comparison.py` | [R] | implemented, validated |
| E — particle-table export | `p1d_io.py` | [R] | implemented, round-trips |
| — driver | `run_1d.py` | — | implemented, end-to-end tested |
| — artifact contract | `core/artifacts.py::PHASE1D` | — | registered, validated against real output |

Seven families: `hdbscan`, `kmeans`, `spherical_kmeans`, `agglomerative` (average / complete /
single at every one of Phase 1's 12 thresholds, plus Ward on k), `spectral`, `gmm`,
`graph_modularity`. The last two of those are implemented here — neither is in sklearn.

## Validation performed

**Every family recovers planted structure.** Three tight caps on $S^{d-1}$ ($d=24$, 15 tokens
each): every one of the seven families has a grid point reaching ARI 1.000 against the planted
labels, and the tuned selection admits all seven.

**The gate separates the three regimes.** Same protocol, 20 null draws, alpha 0.05:

| regime | families admitted |
|---|---|
| three planted caps | 7 of 7 |
| i.i.d. uniform on the sphere | 0–1 of 7 |
| collapsed (all tokens near one direction) | 0 of 7 |

The structureless row is a *rate*, not a bug: the gate is per (family, candidate) with no
multiplicity correction, so 7 families × 2 gated candidates at alpha 0.05 expects ~0.7 false
admissions. Two or more would mean the gate is not working, and the test asserts that bound
rather than perfection.

**Greedy modularity is exact where an exact answer exists.** Two disjoint 10-cliques give
$Q = 0.5$ exactly, split perfectly. On a planted 3-block model ($p_{\rm in}=0.6$,
$p_{\rm out}=0.05$, $n=60$) it recovers the blocks at ARI > 0.95 and within 5% of the planted
partition's own modularity — it is a greedy heuristic with no optimality guarantee, and the
test bounds the shortfall rather than demanding the planted partition be beaten. The modularity
of the returned partition is recomputed by an independent implementation of $Q$, which is the
only way to catch a merge-bookkeeping error that still returns plausible communities.

**AUC matches sklearn including on ties.** The pairwise-concordance AUC used for P-C4 agrees
with `roc_auc_score` to floating point on continuous scores *and* on an all-ties binary
predictor — the case that matters, since the binary clustered/noise flag is nearly all ties and
an implementation resolving them differently would change the comparison the phase exists to
make.

**ΔAUC discriminates in both directions.** A graded score correlated with the target beats an
uninformative binary flag (+0.31 AUC, bootstrap CI [0.21, 0.42]); the same predictor scored
against itself returns exactly 0.000 and the FALSIFIED verdict string.

**The cross-implementation duplication is asserted, not commented.** Co-association under both
noise policies, the singleton relabeling, and the Phase 1 agreement-layer set all agree with
`p1_visualization/cluster_methods.py`'s implementations wherever that package can be imported.
The KMeans trust-gate constants, `DISTANCE_THRESHOLDS` and `K_RANGE` are read out of source
with `ast` rather than copied.

**End to end.** A synthetic Phase 1 run directory (3 layers, 30 tokens, planted caps blurring
with depth, shipped HDBSCAN labels with injected refusals) produces all four verdicts, and
`p1d_results.json`, `p1d_ensemble.npz` and `particle_table.npz` all validate against their
registered `core.artifacts` specs.

## Findings from implementation, before any data

1. **An N-sigma gate on stability discards true structure, because the statistic is bounded.**
   k-means at k=3 on three cleanly planted caps scores stability 1.000 against a null of
   0.648 ± 0.201 — **1.75σ, which fails a 2σ gate**, while exceeding 19 of 20 null draws.
   Spectral clustering on the same data scores 1.00 with two null draws tied at 1.00: a rank
   test failure by ties alone. The decision is made on a rank test, and stability is a floor
   rather than a second significance test, for this reason. This is a real departure from the
   project's N-sigma convention and is flagged wherever the numbers are written.

2. **In the collapsed regime the silhouette is not merely uninformative — it is *worse* than
   its matched null, at a value the shipped trust gate would admit.** A cloud with every token
   near one direction gives k-means at k=2 a silhouette of **0.105**, above
   `cluster_methods.py`'s `KMEANS_SIL_MIN = 0.1`, while the matched null on the same cloud
   scores **0.126** (rank p = 0.95). The shipped gate would call that layer's KMeans k
   trustworthy. Whether this survives on real deep-layer Pythia activations is exactly what
   sub-experiment A measures, but the placed-threshold gate cannot detect the case even in
   principle, and that is a statement about the instrument rather than about the data.

3. **sklearn's HDBSCAN mutates a precomputed distance matrix unless told not to.** With
   `metric="precomputed"` and the default `copy=False`, the input is modified in place; the
   default is scheduled to flip in sklearn 1.10. This phase re-fits ~100 settings against one
   cached distance matrix per layer, so the failure would be silent, cumulative and
   grid-order-dependent. `copy=True` is passed explicitly and a test asserts the matrix is
   unchanged after a fit. **Phase 1 itself uses the `hdbscan` package, which is not affected**
   — but any code that switches backends is.

4. **The two HDBSCAN backends are not interchangeable for P-C2.** `hdbscan` and
   `sklearn.cluster.HDBSCAN` do not agree bit for bit. A P-C2 verdict computed against a
   different implementation than Phase 1 ran would carry an implementation difference inside a
   comparison of settings, so the backend is recorded in every artifact and `hdbscan` is
   preferred when both are installed. **This validation ran on the sklearn backend** (the
   `hdbscan` package is not installed in the environment used); a Pythia run must use the same
   backend Phase 1 used, and the artifact will say which it was.

5. **`n_null` and `alpha` are not independent knobs.** With $n_{\rm null}$ draws the smallest
   attainable p-value is $1/(n_{\rm null}+1)$, so `--n-null 20 --alpha 0.05` leaves exactly one
   passing value (p = 0.048) and any smaller alpha makes every outcome predetermined.
   `select_family` raises rather than running such a sweep. A multiplicity-corrected gate is
   available by passing a smaller alpha, and triples the required draws.

6. **Stability alone would admit i.i.d. points, and the trap is specific.** k-means at k=2 on
   uniform sphere points is highly reproducible — the split it finds is a real property of the
   sample, just not a cluster. This is why the calibration re-runs the *whole pipeline* on each
   null draw rather than scoring the real labels against a null.

## Open items

1. **Not run against Pythia artifacts.** Nothing below the fixture level has been measured. The
   first real run should be a single checkpoint at a moderate `--layer-stride` to establish the
   cost per layer before a sweep is scheduled.
2. **P-C1's scope depends on `clustering.json` being present.** The prediction is registered
   about Phase 1's own agreement layers; `p1d_io.phase1_agreement_layers` reconstructs that set,
   and where the file is absent the verdict falls back to all layers and says so in the verdict
   string. A run adjudicated on the fallback is a weaker test than the registered one.
3. **No figures.** `p1d_ensemble.npz` holds everything a figure needs (co-association matrices,
   per-particle arrays). A `visualization/` submodule matching the other phases' pattern is the
   obvious next step and is not part of this pass.
4. **The consensus is per (run, layer), not tracked across depth or checkpoints.** Consensus
   cluster ids are not aligned between layers — P-C4's persistence target is deliberately
   defined pairwise on co-membership, which needs no alignment. A cross-layer or
   cross-checkpoint chain, the analogue of Phase 1's cluster tracking, is not built.
5. **Promotion to `core/`.** If this phase survives its first run, `co_association`,
   `noise_as_singletons`, `consensus_strength` and the agreement-layer criterion should move to
   `core/` and both this phase and the visualization package should import them. Until then the
   equivalence tests are the mechanism keeping the two copies honest.
6. **`spherical_kmeans` and `graph_modularity` are new code with no external reference
   implementation.** Both are validated against known-answer synthetic cases (planted caps;
   two disjoint cliques at the analytic $Q = 0.5$), which constrains them but is not the same
   as agreeing with an established library.

## Falsification table

Empty, and nothing here will be adjudicated. P-C1–P-C4 were never registered
(`predictions-1d.md`); P-C1, P-C3 and P-C4 were retired 2026-09-30 with the
grading (addendum there), and P-C2 stays a descriptive question. The
adjudicators (`comparison.adjudicate_p_c1..p_c4`) still write lines into
`p1d_results.json` under `verdicts`, stored as `"UNREGISTERED, tier 1: not
adjudications"`; removing the retired three is the cleanup Parked under
"Design revised".
