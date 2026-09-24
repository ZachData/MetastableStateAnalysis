<!-- p1d_cluster_ensemble/status-1d.md -->
# Phase 1d — STATUS

**State:** all five sub-experiments implemented and validated on synthetic data with known
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
