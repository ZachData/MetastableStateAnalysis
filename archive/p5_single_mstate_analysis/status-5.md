<!-- p5_single_mstate_analysis/status-5.md -->
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
   *2026-07-19 addendum: training the tuned lens is no longer an unqualified fix — the
   affine-fit lens has a documented failure mode for exactly Group E's use case. See "Note
   on blocker 4" below before rerunning.*
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

## Note on blocker 4 (2026-07-19): tuned-lens skip-to-output pathology

Source: Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in
Language Models" (Transformer Circuits), §2.4 and Appendix A.5–A.6. Their finding, on the
exact class of object `train_tuned_lens.py` builds (per-layer affine translators trained to
match the final-layer representation): because the objective is correlational, the trained
lens "skips ahead" — from early layers onward it decodes the model's *eventual output
token*, not the layer's intermediate content. In their comparisons the tuned lens recovers
almost none of the known mid-layer intermediates that the logit lens and averaged-Jacobian
lens (J-lens) surface.

Why this hits Group E specifically: Group E's purpose is decoding what a mid-layer cluster
centroid represents *at that layer*. A lens biased toward the final prediction would report
converged, output-like token distributions at every depth — plausible-looking numbers that
erase precisely the per-layer differences the sibling-contrast KL and top-k stability
measurements exist to detect. Fixing blocker 4 by training the affine lens as-is could
therefore replace an obviously-broken result (probs 0.000) with a subtly-wrong one.

Options when Phase 5 resumes, in rough preference order:

1. **Averaged-Jacobian lens (J-lens) as a drop-in.** The paper's J_ℓ is a per-layer linear
   map E[∂h_final/∂h_ℓ] averaged over prompts. It is format-compatible with the existing
   tuned-lens artifact: A_L = J_ℓ, b_L = 0 in the same `{A_L{i}, b_L{i}}` npz, so
   `tuned_lens_cluster.load_tuned_lens` / `apply_tuned_lens` need no changes — only the
   fitting script does (backward passes instead of ridge regression). Their ablations (A.7,
   Fig. 59) show it beats both logit and tuned lens with as few as ~10 prompts, which also
   sidesteps the existing `n_tokens < d` ridge-domination warning: the Jacobian is
   *averaged*, not *fitted*, so it is not underdetermined by small token counts. Cost:
   d_model backward passes per prompt (batched), through the ForCausalLM at the pinned
   revision — `core/lm_loading.load_causal_lm` already provides that model, provably the
   same checkpoint as the extraction (item 2 above).
2. **Keep frozen-head decode as primary, report tuned lens only as a secondary column** with
   the skip-to-output caveat stated in the output. Frozen-head (= logit lens) is what the
   surviving top-1 stability numbers already rest on; the paper finds it captures much of
   the same structure as the J-lens in mid-to-late layers, just less reliably early.
3. **Train the affine lens anyway** and validate it against the pathology before trusting
   it: on a handful of prompts with known unspoken intermediates, check whether mid-layer
   readouts surface the intermediate or the final answer. If the latter, fall back to 1/2.

No decision is being made now — active work is Phase 2 (INDEX.md). This note exists so the
blocker-4 rerun doesn't happen on the pre-2026-07 assumption that "train the lens" is
sufficient. Related: `p2_eigenspectra/lens_band.py` and `vocab_projection.py` (see
CHANGES_jlens_adjacent.md) reuse the same paper's cheap machinery for Phase 2 purposes and
are independent of this decision.

**Deferred, not current work (2026-07-18).** Everything below is built and pure-tested but
sitting unused — active work is Phase 2, not Phase 5. See INDEX.md's "Current priority."
Revisit (and run the smoke tier) once Phase 5 work actually resumes.

## v2 follow-up: causal_tests.py migration — DONE (item 3 aftermath, closed)

Item 3 delivered `core/intervention.py` (`run_model_with_hook`,
`next_token_kl`/`next_token_kl_all_positions`); the consumer-side work that was left
described-but-undone is now done:

1. **`ablate_head`, `steer_residual`, `patch_activation` now dispatch per architecture.**
   Standard per-layer models (GPT-2, GPT-NeoX/Pythia, both bare and LM-head wrappers) route
   through `run_model_with_hook` via `forward_pre` hooks on the target block / attention
   projection; ALBERT keeps `_run_albert_with_hook` unchanged (the standard HF forward has
   no parameter for running a shared layer more times than `config.num_hidden_layers`, which
   extended-iteration ALBERT requires). Dispatch is `_use_legacy_albert_path` — ALBERT by
   class name, everything else standard. GPT-NeoX gains real support in the process
   (`_locate_blocks`, `_block_attn_projection` know `gpt_neox.layers` / `attention.dense`);
   the legacy loop never handled it at all. Return contract unchanged
   (`trajectory, attentions, tokens`, embedding at index 0). One documented difference on the
   standard path: the final trajectory entry is post-ln_f (matching `core/models.py`'s
   extraction convention, i.e. what Phase 1 labels were built on) where the manual GPT-2
   loop recorded pre-ln_f.
2. **The LM-head registry gap is closed: `core/lm_loading.py`.** `load_causal_lm(model_name)`
   resolves the same registry keys (`MODEL_CONFIGS` + `build_pythia_model_configs()`) to the
   `ForCausalLM` variant at the same repo id and pinned HF revision as the bare load — so the
   logits-bearing model is provably the same checkpoint the extraction analyzed. Masked-LM
   entries are refused (the runner's loss is the shifted causal convention; a ForMaskedLM
   load would silently compute the wrong number). `random_init` entries are refused with a
   pointer to `load_causal_lm_from_state_dict`, which rebuilds the LM-head architecture and
   overwrites the transformer body with the extraction pipeline's actual randomized weights —
   re-randomizing inside the loader would produce a *different* random model than the one the
   geometric results describe. Head caveat stated in its docstring: untied heads (Pythia
   `embed_out`) stay trained while the body is random — correct for a trained-vs-random
   readout contrast (identical head both arms), but not "a fully random causal LM."
3. **Group E's tuned-lens-untrained bug (blocker 4, above) still does not reach the
   dual-reading primitive** — `semantic_reading` calls frozen-head decode directly. Unchanged;
   flag again if a tuned-lens mode is ever wired in.

Verification: pure-logic pieces (dispatch helpers, block/projection locators,
`resolve_lm_entry` with injected registries) run for real —
`tests/test_item_completion_pure.py`, 16/16 passing in a torch-free environment. Everything
touching a live model is in `tests/test_item_completion_smoke.py` (project smoke convention,
`SMOKE_REAL_DEPS=1`, tiny-random GPT-2 + GPT-NeoX), written but not executed here — no
torch/network in the sandbox. Run it before trusting the standard path in a pipeline.

## Not yet done

Falsification-table retrofit (v2 item 12, was item 10 in v1) — Phase 5 currently reports
verdicts without a formal prediction/failure-reading table; format matches Phase 4/2c/6 once
added.
