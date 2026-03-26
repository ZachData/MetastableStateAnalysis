# Phase 5 — Cluster Identity and Merge Event Characterization

**Status:** partially scaffolded

## Note
Cluster centroids are already computed in Phase 1 (stored in
clusters.npz and metrics.json).  This phase picks up from those.

## Goal
At each metastable window: identify which crosscoder features are
characteristically active per cluster.  At merge events: record
which features die and which activate.

## Falsification criterion
If cluster identity features are incoherent, the crosscoder hasn't
found interpretable structure.
