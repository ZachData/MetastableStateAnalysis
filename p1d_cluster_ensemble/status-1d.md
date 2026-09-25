<!-- p1d_cluster_ensemble/status-1d.md -->
# Phase 1d — STATUS

## Revived 2026-09-25: the active thread

**Why now (user, 2026-09-25):** Phase 10's experiments are on hold until the
project can say what a cluster is, because every Phase 10 row reads one
HDBSCAN partition and that choice is a confound in all of them. Evidence
that it is (a scratch look at the stored labels, not a recorded result;
`p10_cluster_function/handoff-10.md` "Parked"): on the 8 v1 prompts,
HDBSCAN's count stays at 35–52 at every step and layer while average-linkage
at a fixed cosine distance of 0.35 goes from ~177 to 36 across depth at
step 512. The two agree at layer 0 (ARI 0.75–0.81, noise excluded) and part
in the trained model's deep layers (0.21 at step 143000, L18).

**What was done to revive it.** The code was deleted on 2026-09-23 with its
branch and survived in the local tag `dead/particle-methods-comparison-vpuads`
(`010448c`, 2026-08-20). It was restored from the tag, not rewritten. Three
things had drifted:

| drift | fix |
|---|---|
| `clustering.py` now calls `HDBSCAN(**params)`; the test read inline kwargs | the test reads the `params` literal (`tests/test_phase1d_methods.py`) |
| `PHASE1D` was never on `main`'s `core/artifacts.py` | re-registered, verbatim from the tag |
| the holdout guard (`core/holdout.py`, 2026-09-24) did not exist | `run_1d.py` refuses the 12 held-out prompts; `--v1-only` / `--allow-holdout` |

127 tests pass (1d's 109 plus `test_core_artifacts.py`), conda `mets`.
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
`alpha 0.05`, seed 0), conda `mets`, hdbscan package backend. Output in a
scratch directory, not kept. **Cost:** 442 s wall, 5 556 s CPU (~13 cores),
280 MB RSS: about 150 s per layer, so one run at all 25 layers is about an hour
on the quick grid. The first attempt crashed (the `separation_score` defect).

Every array is populated (consensus labels, confidence, population at all 3
layers). What it shows, from one run, so for design only:

| layer | families admitted | abstained | consensus strength | core / halo / contested |
|---|---|---|---|---|
| 0 | 5 | agglomerative, gmm | 0.27 | 0 / 0 / 467 |
| 12 | 6 | agglomerative | 0.40 | 0 / 253 / 214 |
| 18 | 5 | agglomerative, **hdbscan** | 0.41 | 39 / 76 / 352 |

- **The ensemble mixes scales, and that is the next design question.**
  Ranked by subsample stability, every centroid family picks k = 2 or 3 (the
  coarsest split is the most reproducible), while HDBSCAN keeps about 50 small
  clusters. Co-association averaged over both compares two different
  questions, so low consensus here says little about agreement. A cluster
  needs a stated scale: compare families at matched scale (matched k or
  matched `δ`), or read the families as levels of one hierarchy.
- **HDBSCAN at `min_cluster_size=2` fails its matched null at trained L18**
  (neither of its top 2 grid points clears separation and stability), and it
  is the selected setting at L0 and L12. One run; worth checking across runs.
- **Agglomerative abstains at every layer** on the quick grid. The quick grid
  may be too coarse for it; not yet checked.
- The driver prints `P-C1`–`P-C4` verdicts. They are unregistered, and on one
  run and a quick grid they mean nothing; do not quote them.
- `hdbscan_backend` records `version: unknown` (the package has no
  `__version__`); `p1_mstate_tracking/clustering.py::_hdbscan_version` reads
  the distribution metadata and should be reused.

**Registry.** `P-C1`–`P-C4` (`predictions-1d.md`) were never registered and
cannot be scored blind on the v1 runs already examined. They return as tier 1,
or get registered fresh against the held-out prompts (the user's call).

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
result rows below, by design. Predictions P-C1, P-C2, P-C3 and P-C4 were registered in
`PREDICTIONS.md` before this code existed.

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
