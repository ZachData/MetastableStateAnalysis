# Item 3, sub-item 1: Population selector

New: `core/population.py` — `resolve_population_mask(cluster_labels, population)`.
`population` is `None`/`"all"` (every token), `"clustered"` (label >= 0),
`"unclustered"` (label < 0), or an int cluster id. One function; every
consumer below calls it instead of hand-rolling a mask.

Threaded through the five consumers the plan names:

| File | Function | Default | Change |
|---|---|---|---|
| `p2_eigenspectra/trajectory.py` | `displacement_projection` | `cluster_labels=None` | New optional `cluster_labels`/`population` params. Omitting `cluster_labels` (every existing call site) is byte-identical to before. |
| `p5_single_mstate_analysis/v_alignment.py` | `cluster_energy_trajectory` | — | `chain` param made optional; new `population` param is a mutually-exclusive alternative selection mode (every layer independently, not one tracked cluster's identity across depth). Exactly one of `chain`/`population` required. No prior callers existed, so nothing to break. |
| `p6_subspace/probe_subspace.py` | `probe_accuracy`, `probe_all_channels`, `run_probe_subspace` | `population="clustered"` | Replaces the hard `labels >= 0` drop. Default reproduces prior behavior exactly. `population="all"` makes `-1` its own probed class. |
| `p6_subspace/eigenspace_degeneracy.py` | `degeneracy_ratio`, `degeneracy_sweep`, `run_eigenspace_degeneracy` | `population="clustered"` | Same as above, plus removed a second, redundant `c >= 0` filter on the group list that would have silently defeated `population="all"`. |
| `p6_subspace/centroid_velocity.py` | `centroid_velocity_profile`, `run_centroid_velocity` | — | `cluster_id: int` generalized to `population` (int still works identically). `run_centroid_velocity` gained an optional `ctx["tracked_populations"]` to track named populations (e.g. `"unclustered"`) alongside specific cluster ids. |

## Why `"clustered"` rather than `"all"` as the default in two of the five

`probe_accuracy` and `degeneracy_ratio` both operate over discrete groups
(classes / clusters). Their pre-existing behavior already excluded noise,
and changing the *default* to `"all"` would silently change every existing
result (n_classes, n_clusters, accuracy, ratio) the moment this lands.
`"clustered"` as the default preserves exact prior behavior; `"all"` is one
keyword away when the unclustered population is the thing being asked
about. `displacement_projection` and `centroid_velocity_profile` had no
such discrete-grouping behavior to preserve, so their defaults are
`None`/no-op (every token) or unchanged (int cluster id).

## Verification

Not run through the project's real `pytest` (no network in this sandbox to
install it, and the real `core/config.py` imports `torch`, unavailable
here). Instead: reconstructed the real package layout in a sandbox with a
config stub exposing only `BETA_VALUES`/`MODEL_CONFIGS` (not shipped —
testing convenience only), and ran the existing `tests/test_phase6.py`
before and after every edit via a minimal manual runner. Result: identical
63 passed / 38 failed both times (the 38 are pre-existing autouse-fixture
cases the manual runner doesn't wire up — unrelated to these changes).
Confirm for real with `pytest tests/test_phase6.py -v` once these land.

45 new tests across the 5 files in `tests/`, all passing, covering: the
primitive itself, backward-compatibility of every default, and the new
population behavior (int/`"all"`/`"clustered"`/`"unclustered"`) for each
consumer, including edge cases (empty population at a transition, a
single-class/single-cluster degenerate case, argument validation on
`cluster_energy_trajectory`'s exactly-one-of check).

## Not done in this pass

The other three core-analysis-primitives sub-items from the plan:
merged intervention+logits runner (`causal_tests.py` /`dissociation.py`),
tracking-module merge (`cluster_tracking.py`), and the dual-reading
primitive (schema-first, into a DESIGN.md, per the plan's own caution
about it becoming a god-function). Each is its own scoped piece of work.

---

# Item 3, sub-item 2: Merged intervention + logits runner

**New:** `core/intervention.py` — `run_model_with_hook(model, tokenizer,
text, hooks=None, ...)`. One function, any HuggingFace model that
supports `output_hidden_states`/`output_attentions`, replacing the need
for `p5_single_mstate_analysis/causal_tests.py`'s `_run_albert_with_hook`
and `p6_subspace/dissociation.py`'s `run_intervened_forward` going
forward — see "Not refactored yet" below for why those two files
themselves aren't touched in this pass.

Also new: `next_token_kl` / `next_token_kl_all_positions` — the KL half
of design-5c.md's stated Group D readout ("next-token cross-entropy delta
and KL divergence"); `run_model_with_hook`'s `compute_loss` gives the
cross-entropy half. Neither existed anywhere before this.

## Verification — materially weaker than the population selector

**No torch in this sandbox, no network to install it** (confirmed:
`pip install pytest` and `pip install torch` both fail with no matching
distribution). This module needs torch to do anything real, so most of
it could not be run.

What I could and did verify for real: `_step_gated_hook` (pure Python
control flow — no torch touches it) and `next_token_kl` /
`next_token_kl_all_positions` / `_logsumexp` (pure numpy). 17 tests, all
passing — `tests/test_core_intervention_pure.py`. `import torch` in
`core/intervention.py` is deferred inside `run_model_with_hook` itself
(matching `causal_tests.py`'s own existing `_torch()` pattern), so the
module imports cleanly without torch and these two pieces are genuinely
exercised, not just read.

What I could **not** verify: `run_model_with_hook` itself — the
tokenize/call/extract plumbing and real hook registration against an
actual model. Wrote `tests/test_core_intervention_smoke.py` to the
project's own smoke-test convention (`SMOKE_REAL_DEPS=1 pytest -m smoke`,
`hf-internal-testing/tiny-random-gpt2`, mirroring
`tests/test_phase1_smoke.py`), but did not and could not run it. Read
`core/intervention.py` especially carefully as a result — the module's
own docstring repeats this warning at the top.

## A real gap this surfaced: no model in the registry has an LM head

Every entry in `core/config.py`'s `MODEL_CONFIGS` and
`core/pythia_registry.py` uses the bare model class — `GPT2Model`,
`GPTNeoXModel`, `AlbertModel`, `BertModel` — none of which produce
`.logits`. `run_model_with_hook`'s `logits`/`loss` output will be `None`
for every model currently loadable through the registry. This isn't a
bug in the new runner; it's a real, load-bearing gap between what Group D
needs (next-token cross-entropy, KL divergence — both need logits) and
what the registry currently loads. Something will need to load the
`ForCausalLM`/`ForMaskedLM` variant specifically for causal work,
separate from whatever the main extraction pipeline uses — the smoke test
does this directly (`AutoModelForCausalLM.from_pretrained`, bypassing the
registry) as a preview, not a fix.

## A correction, not a new choice: the embedding-layer indexing convention

`activations[0]` in the new runner is the **embedding** layer, matching
`core/models.py`'s own `extract_activations`/`extract_albert_extended` —
the functions that actually produced every existing Phase 1 `hdb_labels`
array. `dissociation.py`'s prior `run_intervened_forward` skipped the
embedding (`hidden_states[1:]`). That was internally consistent within
that one file (baseline and intervention runs were always compared to
each other, never to externally-supplied labels), but did not match
`core/models.py`'s convention — a real misalignment risk if
`ctx["baseline_labels"]` were ever supplied from genuine Phase 1 output
rather than recomputed inside the same call. Whenever `dissociation.py`
is refactored onto the new runner, this is fixed as a side effect of
adopting the shared convention, not as a separately-tracked bug.

## Why `_run_albert_with_hook`'s ALBERT branch specifically isn't replaced

The standard HuggingFace forward call can't run a shared layer for more
iterations than `model.config.num_hidden_layers` — there's no parameter
for that. `_run_albert_with_hook`'s manual reimplementation exists
specifically to support ALBERT run *beyond* its native iteration count
(Blog 1's extended-iteration methodology). Per the plan's own scope
decision ("Pythia-only, one frozen exception... the multi-architecture
comparison (ALBERT vs. BERT vs. GPT-2) is closed as a reported finding"),
no forward-going work (Phase 5b, 5c Group D, Phase 6 — all GPT-2-large or
Pythia per design-5c.md/status-5c.md) needs ALBERT's extended-iteration
mode. `run_model_with_hook` covers the native-iteration case only, which
is sufficient for every model actually in scope going forward.
`_run_albert_with_hook` is not deleted or touched — it's still there if a
legacy ALBERT extended-iteration re-run is ever needed.

## Not refactored yet: `causal_tests.py` and `dissociation.py` themselves

The plan's wording ("consolidate ... into one model-agnostic runner")
could mean either "build the shared primitive" or "and rewire the
existing files to use it." I did the first and stopped short of the
second, on purpose: rewiring `dissociation.py::run_dissociation` onto the
new runner is the safe half (single architecture, no ALBERT dispatch)
but I have no way to execute it here, and `causal_tests.py`'s three
intervention functions (`ablate_head`, `steer_residual`,
`patch_activation`) would need real per-architecture dispatch logic
(standard model → new runner; ALBERT → existing `_run_albert_with_hook`)
that's genuinely more surface area to get wrong blind. Given the
population-selector work was fully verified before shipping and this
can't be, shipping an unverified rewrite of currently-working legacy code
seemed like the wrong tradeoff. The primitive itself is what Phase 5b,
5c's (not-yet-written) Group D module, and Phase 6 actually need per
design-5c.md — that dependency is satisfied. Rewiring the two existing
files is a contained, describable follow-up once real torch is available
to verify it, not a blocker for anything downstream.

---

# Item 3, sub-item 3: Tracking-module merge

**Changed:** `p1_mstate_tracking/cluster_tracking.py`. New `label_class`
parameter on `_jaccard_overlap_matrix`, `match_layer_pair`, and
`track_clusters` (plus a new helper, `_resolve_trackable_ids`). Default
`"clustered"` reproduces the module's entire pre-existing behavior
exactly — every hardcoded `- {-1}` exclusion is now that default,
nameable and overridable rather than baked in three separate places (one
per function).

## What "one function" actually means here

Same vocabulary as `core.population`, reinterpreted for matching instead
of masking — matching needs a list of *discrete, separately-trackable*
ids (each gets its own Jaccard-matched identity across layers), not a
pooled boolean mask, so the four spellings resolve differently than they
do for `resolve_population_mask`:

| `label_class` | Trackable ids | What it answers |
|---|---|---|
| `"clustered"` (default) | every label ≥ 0 | original behavior, unchanged |
| `"unclustered"` | `{-1}` alone | does the unclustered population's own membership persist/overlap across this transition — via the *identical* Hungarian/Jaccard code path a real cluster gets, not a separate reimplementation |
| `"all"` | every label, -1 included | one matching pass where a real cluster's majority landing in the noise population (or vice versa) is visible — invisible to `"clustered"` by construction |
| int | exactly that one id | e.g. track cluster 5 alone |

`track_clusters(results, label_class="unclustered")` returns the same
shape (`events`, `trajectories`, `summary`) as the original function
always did — a trajectory chain, births/deaths/merges, lifespans — just
describing the unclustered population's own continuity instead of a real
cluster's.

## What this is not

Design-5c.md's `noise_tracking.py` need ("this specific token has been
noise for N consecutive layers") is a *per-token* streak count — a
different statistic from what Jaccard-chain tracking computes even at
`label_class="unclustered"` (which answers a *population-level*
continuity question: does the aggregate unclustered set overlap enough
across the transition to count as the same thing persisting — a token
could leave and rejoin the set while the aggregate overlap stays high
because of other tokens). Design-5c.md's own v2 update already assigns
that need to the per-particle-record schema (core infrastructure, already
built) instead: with a long table keyed by (token, layer), a per-token
consecutive-unclustered count is a groupby, not new tracking machinery.
This sub-item is the literal plan text ("generalize cluster_tracking.py's
Jaccard chaining into one function") — a real, complementary thing, not
a substitute for that groupby.

## Verification — fully verified, like the population selector

No torch anywhere in this file (`scipy.optimize.linear_sum_assignment` +
numpy only), so this one *could* be run for real in the sandbox, and was.
Reconstructed `p1_mstate_tracking/cluster_tracking.py` at its real path,
pulled the exact fixtures `tests/test_phase1_clustering.py::TestTrackClusters`
uses (`stable_tracking_results`, `one_merge_tracking_results`) from
`conftest.py` inline (pytest fixture injection isn't available to the
manual runner), and ran the existing 10 regression tests before and after
every edit: identical pass, zero regressions. 22 new tests covering
`_resolve_trackable_ids`, the overlap matrix, `match_layer_pair`, and
`track_clusters` under every `label_class` value, all passing.

Checked every call site in the project (`grep` across all files): only
`p1_mstate_tracking/analysis.py` calls `track_clusters`, single
positional argument (`track_clusters(results)`); nothing calls
`match_layer_pair` directly outside this module. The new parameter,
appended last with a default, cannot collide with anything.

## Remaining in item 3

The dual-reading primitive (schema-first, into a DESIGN.md, per the
plan's own caution about it becoming a god-function) — the last of the
four sub-items.

---

# Item 3, sub-item 4: Dual-reading primitive — item 3 complete

**New:** `core/DESIGN_dual_reading.md` (written first, per the plan's own
instruction — schema before implementation) and `core/dual_reading.py`
(`geometric_reading`, `semantic_reading`, `dual_reading`,
`to_particle_row_fields`).

## What "point of interest (token, cluster, checkpoint)" resolves to

Read literally, "checkpoint" isn't a location the way a token or cluster
is. Resolved by tying to `core/particles.py`'s existing schema rather than
inventing a third code path: a token is one particle record; a cluster is
a set of records sharing `(layer, cluster_label)`, read at its centroid
(same convention `tuned_lens_cluster.py` already uses); "checkpoint" is
the `checkpoint_step` field already in that key, not a separate kind of
subject. Both token and cluster collapse to the same thing this primitive
actually needs: one vector to read. That collapse is what keeps it thin —
full reasoning in the DESIGN doc.

## The god-function guardrail, concretely

This primitive fits probes, LDA directions, projectors — none of it. All
supplied by the caller, already computed. `geometric_reading` and
`semantic_reading` each call out to something that already exists
(`core.metrics.effective_rank`, `tuned_lens_cluster.frozen_head_decode`)
rather than reimplementing it. The one genuinely new piece of logic is
`effective_rank_contribution` — a leave-one-out delta, defined in the
DESIGN doc precisely because no existing function computes a *per-point*
contribution to a population-level metric; everything else is orchestration.

## A real caveat that testing surfaced, not just design

The first version of `effective_rank_contribution`'s test assumed a
large-norm outlier point would show high "contribution." It failed —
correctly. `effective_rank`'s `mode="raw"` is scale-sensitive: a
large-norm point can dominate the *whole* population's raw spectrum,
making the population look **more** collapsed with it in, not less —
inverting the naive expectation. What actually drives a large
contribution is occupying an underrepresented *direction*, not raw
magnitude. Rewrote the test to isolate direction from scale (matched
norms, varied only whether a direction is redundant or unique) — it
passes for the right reason now. This caveat is now in the DESIGN doc
itself, not just in the test, so a future caller comparing points of very
different norms doesn't get the same surprise I did.

## Particle-table projection

`core/particles.py` already reserves `v_attractive_proj` /
`v_repulsive_proj` columns naming this primitive by anticipation
("once core analysis primitives item 3 lands"). `to_particle_row_fields`
is the connective tissue: reduces a `dual_reading()` result to the
scalar-only subset that fits a columnar table (`v_attractive_proj`,
`v_repulsive_proj`, plus `extra__*` columns for the rest).
`decode_top1_token` and the full `decode_top_k` list are deliberately
excluded from that projection — string/nested values don't fit a
one-scalar-per-row contract, and the plan's own artifact-contract
discipline (core infrastructure item 2) is specifically about every
producer/consumer knowing the exact shape they're getting. They're still
in the direct `dual_reading()` return value for single-point, ad hoc
inspection.

## Verification

`geometric_reading` in full, and the numpy pieces of `semantic_reading`
(`lda_projection`, `probe_predicted_label`) — pure numpy/plain Python, no
torch anywhere in that path. Fully run: 19 tests, all passing (after
fixing the one real bug the tests found — see above), in
`tests/test_dual_reading.py`. The frozen-head-decode piece of
`semantic_reading` (needs a real model) is not — `tests/
test_dual_reading_smoke.py`, written to the project's smoke-test
convention, not executed here, same limitation as `core/intervention.py`.

## Item 3 — core analysis primitives — is now complete

All four sub-items done: population selector, merged intervention+logits
runner, tracking-module merge, dual-reading primitive. Population
selector and tracking-module merge are fully verified (no torch
dependency). The intervention runner and dual-reading primitive are
verified everywhere they could be (pure-logic and pure-numpy pieces) and
explicitly flagged where they couldn't be (anything touching a real
model) — run the corresponding `*_smoke.py` files with
`SMOKE_REAL_DEPS=1 pytest -m smoke` before trusting those two in a
pipeline.

