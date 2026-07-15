# Phase 5c — STATUS

**v2 plan reframe — read this before anything below.** The transition plan's opening section
("Framing: particles first") elevates this phase's central object — the unclustered
population — to be the organizing unit for the *entire* transition project, not just this
phase's question. Quoting the plan directly: "The object of study going forward is every
particle and how it evolves. Clustering is one annotation on a particle, not the unit of
analysis." Concretely, this means: (1) the per-particle-record schema that every future phase
will build on (core infrastructure, v2 item 4) is a generalization of exactly the token-level
persistence tracking this phase already needed and specified (`noise_tracking.py`); (2)
cluster-level and population-level results across *every* phase become aggregations over that
table, not separate code paths — the population selector (v2 item 8) is a column filter on
it; (3) this phase's attention-flip finding (below) is now cited in the plan's own framing
section as part of the motivating evidence for the whole checkpoint-based redesign, not just
a Phase 5c result. This phase's status hasn't changed (still not started on causal work), but
its standing in the project has — read design-5c.md for how this connects to the rest of the
plan's infrastructure decisions.

**Last verified:** not recorded. No dedicated directory exists yet (per the transition plan,
this phase gets its own `p5c_unclustered/` directory as a sibling to
`p5_single_mstate_analysis/`; currently only a root-level README and no code directory).
**Overall:** Causal experiments (Group D) not started. Preliminary correlational evidence
exists from visualization scripts run against existing Phase 1 outputs — see findings below.
No formal investigation groups (A, C) have been run yet.

## Preliminary findings (correlational, no interventions run)

| Finding | Result |
|---|---|
| Attention flip (`noise_importance_proxy.py`) | **Strongest result so far.** Random GPT-2/ALBERT: near parity or clustered-favored. Trained GPT-2-large: unclustered tokens receive ~1.6× layer-average attention, clustered ~0.5×. Trained ALBERT-base: more extreme (unclustered >2×, clustered ~0.5×). Sign flip is trained-specific in every model examined. |
| Punctuation fraction | Not a trained-specific signal — same ratio (clustered ~20%, unclustered ~5%) under random weights. Reflects embedding-space geometry, not learned behavior. |
| Negative-IP mode | Trained-only, density ~10⁻⁴ (rare but real token pairs). Consistent with cone-collapse universality (Phase 1h) — not a contradiction, a new tail structure. |
| Within/between/noise IP decomposition | Within-cluster cohesion stays high and flat; between/noise pairs decline mid-model then rise near the known layer-35 GPT-2-large merge event. Energy plateau is carried entirely by within-cluster pairs. |
| Cluster cohesion direction (top-5) | Coin flip across depth — no consistent direction. Representative of the general ~50-cluster population, not a top-5 artifact. |
| Attractor alignment (`attractor_alignment.py`) | **Not yet run on real data** — written and tested on synthetic data only. |

## Known blockers

1. **`noise_tracking.py` does not exist yet.** Nothing downstream (Group A selection) has a
   per-token "consecutive layers unclustered" definition without it — described in the plan
   as a few-line addition, not built.
2. **Group D blocking dependency, unresolved:** `causal_tests.py`'s intervention functions
   only run through `_run_albert_with_hook`'s ALBERT branch. The GPT-2 branch loops over
   `transformer.h` directly and never calls `hook_fn` — none of the existing interventions
   currently do anything on a GPT-2 model. Must fix (or add a parallel
   `_run_gpt2_with_hook`) before any Group D sub-experiment can run on GPT-2-large, which is
   the model this phase's central question is about.
3. No loss/KL readout path exists yet for any model — new work, not a parameter to an
   existing function.
4. This phase has no code directory yet — everything above ran as standalone visualization
   scripts against Phase 1 output, not as a wired `p5c_unclustered/` package.

## Not yet started

Groups A (persistence structure) and C (effective-rank budget) — both correlational, both
gated only on `noise_tracking.py` (A) and nothing new (C, reuses existing effective-rank
code across populations). Group D (causal) — gated on the GPT-2 hook-wiring fix above.
Group B (routing/flow analysis) is explicitly descoped from this phase (see DESIGN.md).
