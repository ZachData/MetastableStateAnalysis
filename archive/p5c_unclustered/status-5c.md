# Phase 5c — STATUS

<!-- phase-card -->
## Card

- **Question:** What do the particles that never join a cluster (HDBSCAN noise) do: are they a population the trained network uses, and does clustering have a rank budget that they sit outside?
- **Inputs:** `gpt2-large` and `albert-base-v2`, trained and random weights; standalone visualisation scripts over Phase 1 outputs; no package, no manifest, date and battery not recorded, no run dir on this box's drives (checked 2026-09-24)
- **Results:**
  - The attention flip: in trained models unclustered particles receive more attention than the layer average and clustered ones less; under random weights the sign reverses — `archive/p5c_unclustered/status-5c.md` "Preliminary findings (correlational, no interventions run)"
  - The punctuation fraction is not a trained signal: random weights give the same ratio — `archive/p5c_unclustered/status-5c.md` "Preliminary findings (correlational, no interventions run)"
  - The energy plateau is carried by within-cluster pairs — `archive/p5c_unclustered/status-5c.md` "Preliminary findings (correlational, no interventions run)"
- **Superseded / wrong:**
  - On `pythia-410m` the flip is about 94 % causal mask: dividing out a content-free baseline leaves about 6 % of the gap. A different model, so not a refutation of the GPT-2/ALBERT number — `p10_cluster_function/status-10.md` §1.1
  - An ALBERT random-vs-trained ratio was misread as random weights resisting collapse more — `p5_single_mstate_analysis/math-5c.md` §2.2
  - Blockers 2 and 3 (GPT-2 hooks never fire; no model with an LM head) were closed by the `causal_tests.py` migration and `core/lm_loading.py` — `archive/p5_single_mstate_analysis/status-5.md` "v2 follow-up: causal_tests.py migration — DONE (item 3 aftermath, closed)"
- **Registry:** none, because Groups C and D were designed and never registered (`claims/CLAIMS.md`)
- **Depends on:** 1@54c1f37216
- **Feeds:** 10
- **Open threads:**
  - Token frequency confounds both stories: regress membership on log frequency, per layer and checkpoint, first — `archive/p5c_unclustered/lit-5c.md` §1
  - Position 0 is the attention sink and unclustered by construction — `p10_cluster_function/attention-10.md` §2.1
  - Group D (force-collapse, force-disperse) is not written; every primitive it needs exists
  - The rank plateau needs re-establishing on normed rank before the budget test — `p5_single_mstate_analysis/math-5c.md` §9
- **After Phase 10:**
  - The frequency regression on the 410m sweep, v1 prompt keys only (free)
  - Group C, the rank-budget test across populations (free: reuses effective-rank code)
  - Group D (forward pass: one per intervention × prompt × checkpoint)
- **Reviewed:** 2026-09-24 · body `31eefb5696`
<!-- /phase-card -->

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-07-18 · blockers 2 and 3 closed by the `causal_tests.py` migration and `core/lm_loading.py` · `archive/p5_single_mstate_analysis/status-5.md` "v2 follow-up: causal_tests.py migration — DONE (item 3 aftermath, closed)"
- 2026-08-23 · the ALBERT two-timescale ratio is plateau duration, not collapse resistance · `p5_single_mstate_analysis/math-5c.md` §2.2
- 2026-09-16 · token frequency confounds both stories · `archive/p5c_unclustered/lit-5c.md` §1
- 2026-09-20 · the flip on `pythia-410m` is ~94 % causal mask · `p10_cluster_function/status-10.md` §1.1

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
   per-token "consecutive layers unclustered" definition without it. Per-item-3's design
   note (`core/CHANGES.md`, tracking-module merge): this need is now assigned to the
   per-particle-record schema (a groupby on token/layer), not to new tracking machinery —
   but nothing populates that table with real data yet, so the blocker is still live, just
   relocated.
2. **Group D blocking dependency — partially resolved, not closed.** The original problem
   (`causal_tests.py`'s intervention functions only run through `_run_albert_with_hook`,
   whose GPT-2 branch never calls `hook_fn`) has a real fix now available:
   `core/intervention.py`'s `run_model_with_hook` (item 3, complete) is exactly the
   model-agnostic replacement this needed. **But `causal_tests.py` itself has not been
   rewired onto it** — `ablate_head`/`steer_residual`/`patch_activation` still route
   entirely through `_run_albert_with_hook`, unchanged. See status-5.md's "v2 follow-up"
   section for the rewiring shape (per-architecture dispatch, ALBERT path kept, not
   deleted).
3. **Loss/KL readout — now exists at the primitive level, still unusable for Group D.**
   `run_model_with_hook(compute_loss=True)` plus `next_token_kl`/
   `next_token_kl_all_positions` (`core/intervention.py`) implement exactly the readout
   design-5c.md specifies ("next-token cross-entropy delta and KL divergence"). Blocked from
   actual use by two things: (a) blocker 2 above — nothing calls this runner yet in a
   Group-D-shaped way; (b) **no model in the registry has an LM head** —
   `MODEL_CONFIGS`/`pythia_registry.py` load bare model classes with no `.logits` output, so
   even a correctly-wired call returns `logits=None`. A `ForCausalLM` loader, separate from
   the main extraction pipeline's, doesn't exist yet.
4. This phase has no code directory yet — everything above ran as standalone visualization
   scripts against Phase 1 output, not as a wired `p5c_unclustered/` package.
5. The Group D experiment module itself — the force-collapse/force-disperse design this
   phase actually specifies — has not been written. Every primitive it needs (the runner,
   KL/loss, population selector, dual-reading) is now built; assembling them into the
   experiment is separate, phase-level work, not covered by item 3.

## Not yet started

Groups A (persistence structure) and C (effective-rank budget) — both correlational, both
gated only on `noise_tracking.py` (A) and nothing new (C, reuses existing effective-rank
code across populations). Group D (causal) — gated on the GPT-2 hook-wiring fix above.
Group B (routing/flow analysis) is explicitly descoped from this phase (see DESIGN.md).
