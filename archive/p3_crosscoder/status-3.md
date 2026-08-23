# Phase 3 — STATUS

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
