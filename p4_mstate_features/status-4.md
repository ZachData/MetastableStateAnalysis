# Phase 4 — STATUS

**Last verified:** 2026-05-04 (run dirs `results/p4_mstate_features/{model}_2026-05-04_*`)
**Overall:** Complete. Both models (albert-xlarge-v2, gpt2-large) run across all three
tracks. Overall verdict: `metastable_features_detected` for both, via different tracks.

## Verdict table

| Model | Track 1 (crosscoder activations) | Track 2 (direct geometric) | Track 3 (low-rank AE) | Overall |
|---|---|---|---|---|
| albert-xlarge-v2 | `crosscoder_tracks_clusters` | `strong_linear_separability` | **`v_alignment_recovered`** | metastable_features_detected |
| gpt2-large | `crosscoder_tracks_clusters` | `strong_linear_separability` | `v_alignment_still_null` | metastable_features_detected |

## Key numbers

| Metric | ALBERT | GPT-2 |
|---|---|---|
| Feature–cluster NMI (max) | 1.000 | 0.857 |
| Chorus ARI (max) | 0.001 | 0.000 |
| Linear probe accuracy (mean) | 1.000 | 0.300 (0.30 mean is layer-averaged; high within plateau windows) |
| LRAE/CC MSE ratio (mean) | 0.797 | 0.788 |
| LRAE bottleneck → V attractive dirs | 33 | 0 |
| LRAE bottleneck → V repulsive dirs | 3 | 0 |

**Headline finding:** Track 3 (low-rank autoencoder, no sparsity penalty) recovers
V-subspace alignment for ALBERT that Phase 3's sparse crosscoder could not — sparsity was
the confound, not absence of geometric structure. GPT-2 remains null even without the
sparsity constraint, consistent with Regime B (FFN-mediated) not concentrating metastable
signal into a low-rank V-aligned subspace.

## Status per transition plan (v2)

`low_rank_ae.py` (Track 3) is frozen-for-deletion alongside Phase 3's crosscoder — no further
work until activation caches exist at ≥4 checkpoints **and** a specific particle-dynamics
question requires a dictionary. Note this is the one frozen module with a positive result
(`v_alignment_recovered`, ALBERT) — see design-4.md for why that isn't a contradiction.

## Known blockers

1. Track 1's positive NMI is confounded by low crosscoder fire rate — only 12–28/2048
   (ALBERT) and 12/5120 (GPT-2) features have detectable plateaus. Not pursued further;
   Track 3 addresses the same question without the dead-feature pathology.
2. Track 2's GPT-2 mean probe accuracy (0.30) is depressed by layers outside plateau
   windows — read the max (1.0) alongside the mean, not in isolation.

## Downstream implications (binding on later phases)

- Phase 5/6: use LRAE bottleneck directions, not crosscoder features, as the primary
  cluster-identity representation.
- Phase 6: ALBERT's `v_alignment_recovered` confirms Phase 6's direct geometric tests
  (`probe_subspace.py`, `eigenspace_degeneracy.py`) operate on a real signal. GPT-2's Track 3
  null suggests Phase 6 tests for Regime B shouldn't rely on V-subspace decomposition at the
  feature level.
