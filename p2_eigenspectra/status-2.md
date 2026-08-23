<!-- p2_eigenspectra/status-2.md -->
# Phase 2 — STATUS

This phase now covers **two separate studies** against the same question and mostly the same
code. They are reported separately and their numbers are **not** comparable — see
"Why the two studies' scores don't compare" below before putting any GPT-2 and Pythia
figure in the same table.

| Study | Scope | Last verified | State |
|---|---|---|---|
| A — pre-Pythia | GPT-2 / ALBERT / BERT, 35 model×prompt runs | 2026-04-28 (`results/p2_eigenspectra_2026-04-28_13-22-34`) | Closed as reported, frozen |
| B — Pythia checkpoints | `pythia-410m`, 27 checkpoints × 9 prompts = 243 runs | 2026-08-04 (source: `p2_eigenspectra_cross_run_summary.txt`; run dir **TBD — fill in**) | Runs complete, **five verification items open before any result is load-bearing** |

---

# Study A — pre-Pythia (GPT-2 / ALBERT / BERT)

**Overall:** Complete. 35 model×prompt runs. All four previously-documented bugs fixed.
Frozen: this study is not re-run and its numbers are not retrofitted with Pythia results.

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

## Known blockers / open items (Study A)

1. GPT-2-large borderline runs (short_heterogeneous, wiki_paragraph): v-scores 0.455–0.486,
   neither test passes cleanly. Possibly genuine regime-boundary cases.
2. OV spectral norm confound significant on most GPT-2 models (partial ρ to −0.71); rescaled-
   frame result is immune, `V_repulsive_local` verdict is more vulnerable.
3. ALBERT-base: no per-layer decompose path (shared weights) — channel defaults to
   "attention" by construction, not by confirmation. FFN path unresolvable for this model.

---

# Study B — Pythia-410M checkpoint sweep

**Scope:** 27 checkpoints (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000, 5000, 7000,
9000, 11000, 13000, 15000, 17000, 19000, 40000, 60000, 80000, 100000, 120000, 143000) ×
9 prompts. β=1.0 throughout. 24 layers → 23 candidate transitions per run.

**Overall:** All 243 runs completed and produce output. **The headline developmental result
is solid. Two of the four reported columns are degenerate and one further result is
un-adjudicable until the verification items below are closed.** Nothing here should be
cited outside this file until items V1–V5 are answered.

## Headline result — collapse-resistance is learned, and it is dated

| Steps | Total violations (9 prompts) | Mean `frac_repulsive` | Character |
|---|---|---|---|
| 0–4 | 3–4 | 0.50–0.67 | sporadic; 2–3 prompts only |
| **8–64** | **0** | — | `no_violations`, 9/9 prompts, 4 consecutive checkpoints |
| 128 | 1 | 1.00 | onset |
| 256 | 33 | 0.92 | spreading across the battery |
| 512 | 68 | 1.00 | saturated |
| 1000–19000 | 67–81 | 1.00 → 0.79 | count plateaus, attribution begins decaying |
| 40000–100000 | 74 → 56 | 0.58 → 0.50 | verdicts flip to `mixed_or_unattributed` |
| 120000–143000 | 56–62 | 0.80 → 0.72 | partial recovery, 8/8 prompts same direction |

Two readings follow directly:

1. **Theorem 3.4 holds exactly at near-init and is broken by step 512.** Four consecutive
   checkpoints × nine prompts with zero violations is the cleanest confirmation of the
   monotone-energy baseline anywhere in the project. The break onset sits between step 128
   and step 512 — roughly 0.09%–0.36% of training.
2. **Violation count and violation attributability are separate curves with separate
   timescales.** Count saturates at 512 and never moves again (67–81 through step 19000,
   then a slow decline to ~60). `frac_repulsive` stays pinned at 1.00 until ~7000, decays
   monotonically for ~90k steps, then rebounds. The late-training verdict flip is driven
   *entirely* by `frac_repulsive` crossing the 0.5 threshold, not by fewer violations.

The 40000–100000 verdict flip is **knife-edge and should not be reported as a categorical
regime change.** Of the 13 `mixed_or_unattributed` runs in that window, five sit at
`frac_repulsive` exactly 0.500 against a strict `> 0.5` guard. The underlying continuous
curve (1.00 → 0.50 → 0.80) is the real object; the verdict label is an artifact of where
the threshold happens to fall.

## Verdict distribution (243 runs)

| Verdict | Count | Share |
|---|---|---|
| `V_repulsive_local` | 136 | 56.0% |
| `no_violations` | 90 | 37.0% |
| `mixed_or_unattributed` | 17 | 7.0% |
| `V_repulsive_via_FFN` | 0 | — |
| `V_repulsive_via_FFN_confirmed` | 0 | — |
| `V_repulsive_via_attn` | 0 | — |
| `FFN_independent` | 0 | — |
| `overshoot_dominant` | 0 | — |

**The classifier is effectively three-way on Pythia, and three of the five dead branches are
dead by construction, not by evidence.** `V_repulsive_via_FFN`, `V_repulsive_via_FFN_confirmed`
and `FFN_independent` all require `decompose_frac_ffn_drop`, which is unavailable because
`decompose.py` is frozen GPT-2-only. `V_repulsive_via_attn` is reachable in principle but
requires `rescaled_frac > 0.8`, which never occurs (below). See design-2.md, "Classifier
reachability on a parallel-residual architecture."

**Study A's Regime B (FFN-mediated, globally coherent) therefore has no Pythia counterpart
in this data — and this is not yet evidence that it is absent.** It is currently
unmeasurable. Do not write it up as a transfer failure until V1/V2 are closed and the
parallel-residual decomposition module exists.

Breakdown of the 90 `no_violations` runs by prompt: `repeated_tokens` 27 (all checkpoints —
see V4), `camus_letranger` 10, `sullivan_ballou` / `paper_excerpt` / `hdbscan_code` /
`latex_monograph` 9 each, `homer_iliad` 8, `wiki_paragraph` 5, `short_heterogeneous` 4.

## The rescaled frame does essentially nothing on Pythia

`rescaled_improvement_beta1.0` is not printed in the cross-run summary but is exactly
recoverable from the `v_score` identity (`0.40·R + 0.25·fr − 0.15·|ρ|`, with the
`frac_ffn_amplifies_repulsive` term at zero). The reconstruction returns clean integers —
max residual 0.0006 — so this is arithmetic, not inference.

- 19 of 243 runs eliminate ≥1 violation.
- **26 violations eliminated out of 1218 total: 2.1%.**
- Largest improvement in any single run: 3.
- 134 of the 153 runs with violations have improvement exactly 0.
- `camus_letranger` accounts for 7 of the 19 responsive runs.

Study A treated the rescaled frame as the *more trustworthy* signal where it and the
displacement test disagreed (design-2.md, "OV norm confound"). Phase 2b reported
`elim_signed = 1.0` in 35/35 GPT-2-era runs. On Pythia the full-V rescaling is inert.

**This is the single most consequential open question in Study B, and it is currently
un-adjudicable** — see V1 and V2. Three explanations are live and the data as reported
cannot separate them:

- **Numerical.** `rescaled_trajectory_perlayer` builds a cumulative product of `expm(-OV_l)`
  and bails out when it exceeds 1e15 or goes non-finite, truncating to `n_valid_layers`. If
  Pythia truncates early, "rescaling doesn't work" is an overflow report.
- **Overcorrection.** `improvement = max(0, n_phase1 − n_rescaled)` (`analysis_p2.py:153`)
  clips negatives. A zero may be hiding rescaling that makes violations *worse* — the exact
  failure mode flagged for ALBERT in status-2b caveat 1.
- **Rotational interference.** Phase 2b established that OV is 84–97% rotational energy but
  the *signed* component carries 100% of causal weight. Phase 2's rescaled frame uses full V.
  Signed-only rescaling on Pythia is the discriminating experiment.

## Degenerate columns — do not report these as findings

| Column | Reported value | Why it is not a measurement |
|---|---|---|
| `channel` | `mixed`, all 243 runs | `subexp_wrappers.py:221` falls through to `"mixed"` when both `mean_ffn_frac` and `mean_attn_frac` are 0 — which is what an empty decompose result produces. This is the frozen GPT-2-only path returning nothing, not a channel classification. |
| `frac_ffn_amplifies_rep` | `n/a`, all 243 runs | `ffn_subspace` unavailable on Pythia. Zeroes the 0.20 term in `v_score`. |
| `v_score` | 0.033–0.340 (two negatives) | With `R ≈ 0` and `ffn_amp = 0` it collapses to `0.25·fr − 0.15·|ρ|`. 134/153 violating runs match that to within 0.002. It is two other columns rearranged and carries no independent information. |
| `beta1.0_frac_repulsive` | 0.25–1.00 | **This one is real** and is the load-bearing quantity in Study B. Listed here only to say so explicitly. |

## Why the two studies' scores don't compare

`v_score`'s ceiling on Pythia is **0.65**, not 1.0, because the 0.20
`frac_ffn_amplifies_repulsive` term is structurally zero. Study A's thresholds ("scores
above ~0.5 consistently correspond to `_confirmed` or `_local`", `verdict_v2.py:61`) and its
GPT-2-large borderline band (0.455–0.486) are calibrated against a scale Pythia cannot
reach. **Any table placing a Study A v_score next to a Study B v_score is wrong.** Compare
`beta1.0_frac_repulsive` and violation counts instead, or restore the missing term first.

Same applies to the OV-norm confound: Study A reached partial ρ = −0.71; Study B's maximum
|ρ| is 0.518 and the typical magnitude is 0.10–0.25. The confound is genuinely weaker here,
but the comparison is between a 35-run and a 243-run sample and the Spearman is computed
over ~23 layers either way.

## Secondary observations (real, lower confidence)

- **The step 0–4 transient.** Steps 0, 1 and 2 produce bit-identical outputs (2 violations
  `short_heterogeneous`, 1 `wiki_paragraph`); step 4 adds `homer_iliad`; then steps 8–64 are
  clean. Untrained init is *less* monotone than 8 steps of training. This inverts what
  PREDICTIONS.md claim (a) assumed and needs V3 before it is explained either way.
- **OV-norm confound has a sign structure.** Mean partial ρ over violating runs: −0.323 at
  step 512 (co-located with count saturation), rising through zero to +0.257 at step 15000,
  back to ~0 from step 40000. Not interpreted yet.
- **Prompt ordering inverts over training.** `short_heterogeneous` violates first (step 0)
  and ends lowest (3 violations at 143000, `frac_repulsive` 0.67). Long naturalistic prompts
  end highest (mean over steps ≥512: `camus_letranger` 10.4, `wiki_paragraph` 9.6,
  `latex_monograph` 9.6, vs. `short_heterogeneous` 4.5). **Check token count before reading
  this as a content effect** — fewer particles means a noisier energy estimate.

## Verification items — open, blocking

Nothing in Study B goes into a blog post, a cross-phase reference, or PREDICTIONS.md until
these are answered. All five are cheap; four are reads against artifacts that already exist.

- **V1 — `n_valid_layers` per run.** Recorded by `rescaled_trajectory_perlayer` (line 324),
  not surfaced in the cross-run summary. If Pythia truncates well short of 24, the entire
  rescaled-frame result is an overflow report. **Highest leverage item on this list.**
- **V2 — `n_rescaled_violations`, unclipped.** Recorded in `analysis_p2.py`'s
  `rescaled_out`. Distinguishes "rescaling has no effect" from "rescaling makes it worse."
- **V3 — ΔE magnitudes at the step 0–4 violations.** If they sit at 1e-6–1e-5 against the
  `< -1e-6` threshold, the init transient is numerical noise on a near-flat energy curve and
  the story collapses to something simple.
- **V4 — effective rank on `repeated_tokens`.** Zero violations at all 27 checkpoints is
  almost certainly the `eff_rank >= 3.0` guard, not monotonicity: the prompt is `". "` ×~264,
  one distinct token id (`battery_structure.py:58`), so the particles are degenerate at
  embedding. If confirmed, `repeated_tokens` must be excluded from Study B's denominators
  rather than counted as 27 clean `no_violations` runs.
- **V5 — did the decompose subexperiment run at all**, or run and return empty? Changes
  whether `channel` should read `mixed` or `unknown`, and whether the coverage warning
  (`analysis_extended.py:671`) fired.

## Prediction adjudication (PREDICTIONS.md)

Study B speaks to two of the three project-level claims. **Do not write the addendum into
PREDICTIONS.md until V1–V4 close**; recorded here as the draft reading.

- **(a) Collapse-resistance is learned, not initial — supported, with a wrinkle.** Steps
  8–64 are perfectly monotone across all nine prompts, which is the predicted "random-like"
  behaviour and stronger than the prediction asked for. Step 0 itself is *not* clean (3
  violations), which the prediction did not anticipate. Pending V3.
- **(b) Resistance emerges at circuit-formation events — needs a dated correction.** The
  prediction named steps ~512–2000. The break begins at 128, is systemic by 256, and is
  fully saturated by 512. **The onset is earlier than predicted and is finished where the
  prediction expected it to start.** This is the p2 half only; co-location with the Fiedler
  drop is p1/p1b's to adjudicate.
- **(c) Phenomenology transfers across architecture — not adjudicable here.** Requires
  `pythia-1.4b-random` and the gpt2-large comparison, neither of which is in this run. The
  absence of Regime B and the inert rescaled frame both bear on (c) and both are currently
  confounded by missing instrumentation.

## Cross-phase conflict to resolve

`status-1.md`'s verdict table records "Monotone energy $E_\beta$ (Thm 3.4) — **Falsified
universally, including under random weights**." Blog 1 says the opposite for the random case
("the random case monotonically increases in energy"). Study B lands on Blog 1's side: steps
8–64 show zero violations across 9 prompts. The status-1.md row is wrong or is using
"random weights" to mean something narrower than near-init. **Flagged for the Phase 1 doc
pass; not fixed here.**

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

**Study B raises the priority of this module from "upgrade" to "blocking".** Three of the
classifier's eight branches and 20% of `v_score`'s weight are unreachable without it, and
the Regime A/Regime B frame that organizes Study A cannot be tested on Pythia at all until
it lands.

## Next experiments, in order

1. **Close V1–V5.** Reads against existing artifacts, no new compute.
2. **Signed-only rescaling on Pythia** (Phase 2b's `elim_signed`, not Phase 2's full-V
   frame). The discriminating test for the inert-rescaling result: if signed-only recovers
   ~1.0 while full-V gives 2.1%, the failure is rotational interference in the matrix
   exponential and V is still causal. If signed-only also fails, the mechanism genuinely
   does not transfer and claim (c) is in trouble.
3. **Parallel-residual decomposition module** in `core/`, restoring the FFN branches and the
   0.20 `v_score` term.
4. **Dense checkpoints between 128 and 512.** The whole onset happens inside one order of
   magnitude with three sampled points. This is where the pilot sweep's reserved adaptive
   slots should go.
5. **Explain the `frac_repulsive` decay and rebound** (1.00 → 0.50 → 0.80 with count roughly
   flat). Something reorganizes which subspace the violations occupy without changing how
   many there are, and the 120000–143000 rebound moves the same direction in 8 of 8 prompts.

## Maybe later (not current work, 2026-07-18)

- **SLT anchor** (plan: per-checkpoint training-loss + weight-norm logging alongside the
  cheap tier) — not implemented, not being added right now. Study B's 128–512 onset window
  is exactly the kind of thing this would help localize, so revisit if item 4 above happens.
- **LLC** — already dropped per the plan (SGLD out of compute range); no reason has come up
  to revisit it. Both come back together only if good checkpoint data exists and an actual
  question needs either.
