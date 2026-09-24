# Phase 3 — STATUS

<!-- phase-card -->
## Card

- **Question:** Do the features a sparse cross-layer dictionary (crosscoder) learns on the residual stream line up with the value operator's attractive and repulsive eigen-subspaces, and do the long-lived ones carry the particles' metastable clusters?
- **Inputs:** `albert-xlarge-v2` and `gpt2-large`, trained weights only, no checkpoints, no Pythia; BatchTopK crosscoder, k = 64; battery not recorded. The only runs on disk (main tree `results/phase3/`, 2026-04-21) **back only 1 of the verdict table's 7 rows** (bimodality); the other six are errors or absent there — `archive/p3_crosscoder/status-3.md` "Runs on disk"
- **Results:**
  - Decoder directions align with V at chance in both models, and feature lifetime does not predict alignment (no producing run on disk) — `archive/p3_crosscoder/status-3.md` "Verdict table"
  - Feature lifetimes are bimodal on ALBERT, not on GPT-2 (the one row the runs on disk reproduce) — `archive/p3_crosscoder/status-3.md` "Verdict table"
  - Steering along features moved no merge on ALBERT; on GPT-2 the eval prompts had no merge to move (no producing run on disk) — `archive/p3_crosscoder/status-3.md` "Verdict table"
  - The null is close to a property of the instrument: a sparse objective makes decoder columns near-orthogonal, and those align with any fixed subspace at chance — `archive/p3_crosscoder/lit-3.md` §1
- **Superseded / wrong:**
  - Read as "no geometric structure at the feature level"; the literature reads it as the sparse objective's own geometry, and Phase 4's dense autoencoder recovered alignment on ALBERT — `archive/p3_crosscoder/status-3.md` "Corrections received"
  - "A `FROZEN.md` stating this trigger is still pending": written 2026-08-22 — `archive/p3_crosscoder/FROZEN.md`
- **Registry:** none, because the phase ran before the registry existed and was archived 2026-08-22 (`claims/EXPERIMENTS.md`)
- **Depends on:** 1@54c1f37216, 2@5e9fc59e62
- **Feeds:** 4
- **Open threads:**
  - Which run produced the six rows other than bimodality (decoder→V 0.484 / 0.501 among them)? Not the two on disk — `archive/p3_crosscoder/status-3.md` "Runs on disk"
  - Cross-term feature weighting and induction tagging never ran (Phase 2's `cross_term_results` not passed; `pair_agreement` empty) — `archive/p3_crosscoder/status-3.md` "Known blockers (low priority, frozen)"
  - The reintroduction trigger's first half (activation caches at ≥ 4 checkpoints) is met once Stage 0 lands 19 checkpoints of 410m; its second half (a particle question that needs a dictionary) is not — `archive/p3_crosscoder/FROZEN.md`
- **After Phase 10:**
  - None proposed. If a question needs a dictionary, use Phase 4's dense low-rank autoencoder, not a sparse one (free: trains on saved activations, CPU)
- **Reviewed:** 2026-09-24 · body `e34a8d8c72`
<!-- /phase-card -->

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-09-16 · the null is the sparse objective's near-orthogonality, not an absence of structure; Phase 4 Track 3 is the informative arm · `archive/p3_crosscoder/lit-3.md` §1

## Runs on disk (checked 2026-09-24)

Main tree `results/phase3/` (untracked) holds two runs, both 2026-04-21:
`albert-xlarge-v2_2026-04-21_13-27-04` and `gpt2-large_2026-04-21_13-28-01`
(`analysis_results.json`, `crosscoder_config.json`, `cross_phase/`). Their
feature-lifetime bimodality coefficients match the verdict table below
(0.622 / 0.514), **the only one of the table's 7 rows they reproduce.**
`v_subspace_alignment` and `lifetime_vs_alignment` are
`{"error": "v_projectors not in artifacts"}`;
`ffn_repulsive_feature_alignment` and `decoder_violation_projection` are
errors too; there is no steering or pair-tracking output at all. Of the 18
analysis keys, 10 are errors in both runs. So six rows below (decoder→V,
lifetime×V, violation projection, FFN alignment, steering, pair tracking) came
from a later run (the table was last verified 2026-04-29) that is on none of
this box's three drives. Nothing for Phases 4, 5, 5b, 5c or the frozen Phase 6
is on them either.

**Last verified:** 2026-04-29
**Overall:** Complete. Both models run. **Overall verdict: null.** Per the transition plan,
this phase is being frozen in place (relocated untouched, `FROZEN.md` to be added) rather
than revisited — see plan doc, "Scope decisions."

## Verdict table

| Test | ALBERT-xlarge-v2 | GPT-2-large |
|---|---|---|
| P1 — feature lifetime bimodality | Confirmed (BC=0.622) | Not confirmed (unimodal, BC=0.514) |
| P2 — decoder → V alignment | Null (attract_dominance 0.484, indistinguishable from random) | Null (0.501) |
| Lifetime × V alignment (Spearman) | Null (ρ=0.03) | Null (ρ=0.09, negligible effect at n=1041) |
| Violation projection (top-10 features) | 23.6% | 7.8% |
| FFN alignment (cosine sim) | 0.018 | 0.007 |
| Steering causal effect | Null (mean Δmerge = 0) | Unrunnable (no baseline merge event) |
| Pair tracking | Null (Jaccard=1.0, no perturbation) | Unrunnable |

**Interpretation (favored):** the crosscoder learned syntax/frequency/surface-form features
with the right temporal profile (short vs. long-lived) but not organized by V's
eigenstructure. Phase 2's mechanism explains *why* energy drops; it doesn't organize *what*
the model represents at the feature level. Robust across two models, multiple metrics, and
direct causal intervention — rerunning the two unrun analyses is unlikely to change this.

## Known blockers (low priority, frozen)

1. `cross_term_feature_weighting` — needs `cross_term_results` from Phase 2, not passed to
   this phase's run directory. Not run.
2. `induction_feature_tagging` — needs `pair_agreement` from Phase 1; reports 0 exclusive
   tokens, meaning the artifact was absent or empty. Not run.
3. GPT-2 steering unrunnable with current eval prompts (no baseline merge event).
4. Per-layer SAE baseline and GPT-2 two-zone crosscoders — not run, speculative given the
   global null.

## Status per transition plan (v2: hardened to frozen-for-deletion)

v1 called this "frozen, revisit once checkpoint data exists." **v2 is stricter: this is
frozen-for-deletion.** Stated status per the plan's own wording: "candidate for deletion;
git history is the archive." Rationale (v2): the project's claim to rigor rests on
theoretically grounded particle dynamics; SAE features have no comparable grounding, and
this phase already showed sparse dictionary methods underperforming dense/low-rank
alternatives (crosscoder chorus ARI = 0.000 in both models tested).

**Reintroduction trigger, stated precisely (not "once checkpoint data exists"):**
activation caches exist at ≥4 checkpoints **and** a specific particle-dynamics question
requires a dictionary — not before, and not just because more data becomes available. No
real work happens with SAEs in the meantime; a `FROZEN.md` stating this trigger explicitly
is still pending (transition plan item 4).
