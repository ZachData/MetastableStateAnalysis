# Draft: HDBSCAN's clusters depend on row order (tied mutual-reachability edges)

**Status: draft, not posted.** Posting is the user's call (`STATE.md` Blocked 12).
Two targets: a comment on scikit-learn-contrib/hdbscan #265 ("HDBSCAN gives different
clustering results if the order of the dataframe rows changes", 2018, open; related: #409,
where a commenter names non-unique MST edge weights, and #241, column order, where the
maintainer guessed ties), and a new issue on scikit-learn (`sklearn.cluster.HDBSCAN` shares
the tree code), linking #265. What is new here: the cause is structural (core distances), a
measure, a tested fix, and scikit-learn reproduces it. Evidence:
`p1d_cluster_ensemble/status-1d.md` "Admission" and "Blocked 11″ decided". Repro:
`tools/hdbscan_tie_repro.py`.

---

**Title:** Clustering changes with row order when mutual-reachability distances tie (which is
common, not an edge case)

**Summary.** Following up #265 and #409 with a diagnosis and a fix. The same points in a
different row order give a different clustering, without duplicate points. Repeated
fits on one order are identical, so this isn't randomness. In our data (cosine distances
between transformer hidden states, `min_cluster_size=2`, `min_samples=2`) a large share of
returned clusters are not connected components of the thresholded mutual-reachability graph
at any distance: they exist only because of the order in which tied edges were joined.

**Reproduction** (numpy, scipy, scikit-learn, hdbscan; runs in seconds):

```python
<paste tools/hdbscan_tie_repro.py>
```

Output with hdbscan 0.8.41 and scikit-learn 1.7.2:

```
hdbscan (min_samples=2): repeat on one order identical: True; 26 of 30 row orders change the clustering
scikit-learn 1.7.2 HDBSCAN (min_samples=3): repeat on one order identical: True; 26 of 30 row orders change the clustering
```

**Cause.** A point's core distance is its k-th neighbour distance, and that one value becomes
the weight of every mutual-reachability edge where it is the maximum. So equal edge weights
are structural, even with continuous data and no duplicate points. The hierarchy itself is
order-free: at each level it is the set of connected components of {edges ≤ ε}. The
single-linkage tree is binary, so edges of equal weight are joined one at a time in
processing order, and condensing plus EOM selection then sees a sequence of splits that
depends on that order. It can, for example, peel tied points off one by one in one order and
split them 3 + 2 in another, or attach an unrelated point to a tight group.

**Proposed fix.** Build the condensed tree from level sets: merge every edge of one weight at
once (a node may have more than two children), then condense and run EOM as now. We
implemented this and checked it against hdbscan: given hdbscan's own binary tree, our
condensing + EOM reproduces hdbscan's labels and `cluster_persistence_` × max λ in 400 of 400
fits, so tie handling is the only difference. On the level-set tree the clustering does not
change under row permutation (0 of 10 orders), and every selected cluster is a connected
component at some level. A cheaper alternative, sorting edges with a deterministic tie-break,
makes the output repeatable but still arbitrary, and still yields non-component clusters.

**Also noted:** `cluster_persistence_` divides by the largest λ in the whole tree, so an
unrelated tight cluster lowers every other cluster's persistence (a planted group: 0.75 alone,
0.012 next to a tighter group). Maybe worth a docs note; separate issue if wanted.

Happy to open a PR if the level-set approach is acceptable.
