# Phase 5 — STATUS

**Last verified:** not recorded in source (after Phase 4, 2026-05-04)
**Overall:** Complete for 6 models (gpt2-xl, gpt2-large, gpt2-medium, bert-base-uncased,
albert-xlarge-v2, albert-base-v2). Groups D, E, and merge geometry are partially or fully
blocked by code bugs, not by absence of signal — see blockers.

## Verdict table (cross-model architecture-level findings)

| Finding | Result |
|---|---|
| Locally rotational (S/A) | Universal — every model, every cluster. Extends Phase 2b's global null to individual trajectories. |
| Attractive/repulsive centroid split | ~50/50 everywhere — mild tension with Theorem 6.3's attractive-subspace-dominance prediction. |
| FFN vs. attention role (Group C2) | Architecture-dependent: GPT-2 attn-dominant/co-dominant; ALBERT FFN-cohesive/attn-disruptive; BERT attn-cohesive/FFN-disruptive (anomalous vs. ALBERT despite architectural similarity). |
| Single dominant attractor head (Group C1) | Universal — top head 2–4× the cohesion of second place. Sharper in larger models. |
| Causal robustness (Group F) | gpt2-xl (0.80) > gpt2-large (0.53) > albert-base (0.15) ≈ albert-xlarge (0.14) > gpt2-medium (0.00) — largest model most robust to single-point intervention, *if* the identical-value issue below isn't a bug. |

## Known blockers (fix before Phase 7 or publication)

1. **`merge_verdict` always `n/a`, all 6 models.** `merge_events` isn't reaching
   `merge_event_geometry()` from `run_5.py`. Central Group B output (fusion direction vs.
   attractive subspace) is untested as a result.
2. **OV values always `n/a` in C1** *(v2: artifact-contract class — see below)*.
   `cohesion_source` is universally `inward_mass_fallback` — head rankings are valid relative
   signals, not grounded in the OV mechanism. Likely a miskeyed Phase 2 weights load in
   `head_contributions.py`.
3. **Group D blocked, all 6 models** *(v2: artifact-contract class — see below)*. Phase 4
   outputs not reaching Phase 5 — path/naming mismatch in `p5io.load_phase4()`, or Phase 4
   didn't write the expected cache files.
4. **Group E — stored probabilities round to 0.000.** Tuned lens untrained (logit-lens
   fallback in use, `used_tuned_lens=false` everywhere). Top-1 token stability (76–100%)
   stands; probability mass does not. Nothing built on top of this is trustworthy until fixed.
5. **Group F — identical `mean_frac_together` across all 4 interventions, per model.**
   Either the metric is computed once and duplicated, or all interventions hit the same
   causal bottleneck. Unresolved — verify before using F results in any analysis.
6. **bert-base Group F not run** — no blocking reason identified, just not done.

**v2 note on blockers 2 and 3:** the plan explicitly classifies both as instances of the same
underlying problem — producer/consumer mismatch, where one phase's writer and the next
phase's reader disagree on names/shapes/paths without either side erroring loudly. Rather
than patch `head_contributions.py` and `p5io.load_phase4()` independently, v2's core
infrastructure (item 2, `core/artifacts.py`) declares each phase's output contract once and
has every consumer import those constants, which is meant to kill this bug *class*, not just
these two instances. Don't fix these as isolated one-offs before that lands — check whether
the artifact contract module already resolves them first.

## Not yet done

Falsification-table retrofit (v2 item 12, was item 10 in v1) — Phase 5 currently reports
verdicts without a formal prediction/failure-reading table; format matches Phase 4/2c/6 once
added.
