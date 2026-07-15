# Phase 2 — STATUS

**Last verified:** 2026-04-28 (run dir `results/p2_eigenspectra_2026-04-28_13-22-34`)
**Overall:** Complete. 35 model×prompt runs. All four previously-documented bugs fixed.

## Verdict distribution (35 runs)

| Verdict | Count | Models |
|---|---|---|
| `V_repulsive_local` | 13 | ALBERT-xlarge (5), GPT-2-xl (3), GPT-2-large (2), GPT-2-medium (1), ALBERT-base (2, weak) |
| `V_repulsive_via_FFN` | 8 | GPT-2-small (4), GPT-2-medium (4) |
| `V_repulsive_via_FFN_confirmed` | 3 | GPT-2-xl (2), GPT-2-large (1) |
| `FFN_independent` | 1 | BERT (1, borderline) |
| `mixed_or_unattributed` | 10 | ALBERT-base (3), BERT (4), GPT-2-small (1), GPT-2-large (2) |
| `overshoot_dominant` | 0 | ruled out universally |
| `V_repulsive_via_attn` | 0 | code path exists, never fires empirically |

Core conclusion: V's mixed-sign eigenspectrum is causal for energy violations. Two regimes:
attention-mediated direct detection (ALBERT-xlarge, GPT-2-xl/large) vs. FFN-mediated global
effect (GPT-2-small/medium). BERT and ALBERT-base sit below reliable detection threshold.

## Known blockers / open items

1. GPT-2-large borderline runs (short_heterogeneous, wiki_paragraph): v-scores 0.455–0.486,
   neither test passes cleanly. Possibly genuine regime-boundary cases.
2. OV spectral norm confound significant on most GPT-2 models (partial ρ to −0.71); rescaled-
   frame result is immune, `V_repulsive_local` verdict is more vulnerable.
3. ALBERT-base: no per-layer decompose path (shared weights) — channel defaults to
   "attention" by construction, not by confirmation. FFN path unresolvable for this model.

## Not yet done (per transition plan, v2)

`decompose.py`, `ffn_subspace.py`, `ffn_contributions.py`, `run_2.py`'s decompose stage,
`analysis_extended.py`'s coverage check, and the decomposed-violations subexperiment in
`subexp_wrappers.py` remain frozen GPT-2-only against existing GPT-2-large output — that
sequential-architecture decomposition genuinely doesn't apply to Pythia's parallel residual.

**This is now explicitly an upgrade path, not a dead end (v2, item 5).** Pythia computes
attention and FFN from the same pre-block input and sums both into the residual in
parallel: Δx = attn_out + ffn_out *exactly*, with no ordering confound — a cleaner,
exactly-additive decomposition than this phase's GPT-2 module ever produced. A new parallel-
residual decomposition module (in `core/`, not this phase's directory) re-enables the
attn-vs-FFN energy panels this phase's questions feed (Phase 1's `energy_decomposition.py` /
`energy_attribution_aggregate.py`) and potentially reopens the FFN-vs-V question natively on
Pythia, rather than carrying it forward only as a frozen GPT-2-large reference point.
