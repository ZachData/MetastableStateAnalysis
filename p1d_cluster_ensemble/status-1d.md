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
nulls; code `2a176a4` (27 records) and `631473c` (21; differs only on an
invalid tree, which would have crashed). The reference rule reproduces A0's
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
  one. The likely reason (not tested) is token identity, as in #106: at step
  0 the k = 2 families agree on string-identical tokens the null cannot
  reproduce.
- What the rules do change: the consensus k (3 to 10) and the core share
  (0 to 0.62). A choice among them moves the readout without stabilising it.

**Reading.** Weighting is not what makes the grading unusable. The trichotomy
is a pooled-null threshold on a confidence that mixes a k = 2 vote (0–6
families at k = 2 per record) with fine-scale votes, so each family's
partial view decides which side of the threshold a block falls. That is the
scale problem (`lit-1d.md` §1, option D) reappearing in the vote, not a
weighting problem. **For the user:** retire core / halo / contested as 1d's
product (P-C4, unregistered, would score noise), or first make the ensemble
vote at one matched scale and ask this again. Claude recommends retiring
it; the per-family gate, the merge tree and the nulls are the parts that
have held.

**Fixed on the way (a defect).** scipy 1.15's average linkage returned an
invalid tree (a node merged with itself) on a tied co-association (446
tokens, 27 row types, 10 values; a null draw under a leave-one-out rule).
`consensus_partition` did not check, `fcluster` raised, and the batch died
24 of 48 in. It now rebuilds an invalid tree on distances rounded to 12
decimals, and refuses if that is invalid too; a valid tree is untouched
(`ensemble.py`, regression case inline in `tests/test_phase1d_ensemble.py`;
`LESSONS.md` 2).

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

Empty by design — nothing has been run. The registered predictions and their falsifiers are in
`PREDICTIONS.md`; the adjudicators (`comparison.adjudicate_p_c1..p_c4`) write their verdicts
into `p1d_results.json` under `verdicts`, and every verdict string names its own prediction id
so a reader cannot mistake which claim was decided.
