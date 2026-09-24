# Phase 1d — FROZEN (code deleted, intent kept)

**Code deleted 2026-09-23** with its branch
(`claude/particle-methods-comparison-vpuads`, tip `010448c`, 2026-08-20), in a
cleanup that should have skipped it (`LESSONS.md` lesson 12). **Kept here
2026-09-24, verbatim from that tip:** `design-1d.md` (why it was built this
way), `status-1d.md` (what was validated and six findings from building it)
and `predictions-1d.md` (the `P-C1`–`P-C4` text, written into the branch's
`PREDICTIONS.md` before the code and **never entered in
`claims/registry.json`**). Paths inside them are the branch's
(`p1d_cluster_ensemble/`, `p1_visualization/`), not this tree's.

**Rule (user, 2026-09-24):** code may go, as long as the intent and the reason
the code existed stay. Tooling has moved on since August, so a revival rewrites
rather than rebases. Until someone deletes the local tag
`dead/particle-methods-comparison-vpuads`, the code can still be read from it.
Nothing depends on it existing.

## What it was for, in one paragraph

Every cluster-conditioned result rests on
`hdbscan.HDBSCAN(min_cluster_size=2, metric="precomputed")`, a library minimum
nobody chose. Phase 1's three other partitions are untuned too, so its
cross-method agreement compares four sets of defaults. 1d tuned seven families
(HDBSCAN, k-means, spherical k-means, agglomerative, spectral, GMM, graph
modularity) per layer against subsample stability with a whole-pipeline null,
built a one-vote-per-family consensus, and exported a graded per-particle
confidence. It cost reads only (activations, no forward pass). It never ran on
Pythia.

## Findings worth keeping without the code (`status-1d.md` "Findings from implementation, before any data")

1. An N-sigma gate on a bounded stability statistic rejects true structure;
   use a rank test, with stability as a floor.
2. In a collapsed cloud, k-means silhouette can clear the shipped
   `KMEANS_SIL_MIN = 0.1` while scoring *below* its matched null.
3. `sklearn.cluster.HDBSCAN(metric="precomputed")` mutates the distance matrix
   unless `copy=True` (the `hdbscan` package does not).
4. The two HDBSCAN backends disagree bit for bit; record which one ran.
5. `n_null` bounds the smallest p (`1/(n_null+1)`); refuse sweeps where alpha
   leaves the outcome predetermined.
6. Stability alone admits i.i.d. points; the null has to re-run the whole
   pipeline, not score the real labels.

## When to rebuild

When Phase 10 needs a partition it can trust. Phase 10 found the HDBSCAN
partition is not reproducible run to run (`p10_cluster_function/status-10.md`
§3). Note: that is drift between near-identical re-runs, while 1d tuned
against subsampling, a different noise (`/challenge-pr` on #80). So a rebuild
should first measure whether tuning reduces *that* drift. `P-C1`–`P-C4` would
return as tier 1, or be registered fresh against the held-out prompts; they
cannot be scored blind on Phase 1 runs already examined.
