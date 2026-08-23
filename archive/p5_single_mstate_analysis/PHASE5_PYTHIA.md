<!-- p5_single_mstate_analysis/PHASE5_PYTHIA.md -->
# Phase 5 — Pythia rebuild (LIVING DOCUMENT)

**Started:** 2026-08-06
**Status:** in progress — see Work Items.
**Relationship to existing docs:** `status-5.md` and `design-5.md` describe the
frozen pre-Pythia study (6 models, GPT-2 / BERT / ALBERT). This file describes
the Pythia rebuild and supersedes them wherever the two disagree. Same
Study A / Study B split Phase 2 already made — do not retrofit Study A's
numbers with Study B's.

Update this file as work lands. Append to the Changelog; don't rewrite
earlier entries.

---

## 1. The framing decision

**The unit of analysis is a fixed set of token positions, not an HDBSCAN
cluster trajectory.**

Reasoning: a trajectory is Jaccard-chained *within one run*. Cluster ids and
trajectory ids are not comparable across checkpoints. Under a sweep, the
current `select_cluster.py` re-selects independently at every step, so every
cross-checkpoint statement would be about a different object.

The one index that carries across the whole sweep is **token position**. The
battery is fixed and `core/pythia_registry.py` records that the NeoX tokenizer
is byte-identical from step 0 to step 143,000, so position *i* is the same
particle at every checkpoint. This is what "particles first" (INDEX.md v2,
design-5c.md) means operationally.

Consequence: selection happens **once**, producing a frozen token set.
"Does this cluster exist at step 512?" stops being a selection problem and
becomes a measurement.

### Two anchors, both run

| Anchor | Question | Direction |
|---|---|---|
| **anchor-final** (step 143000) | What became of these particles? | measured backward |
| **anchor-init** (step 0) | What happened to particles that started together? | measured forward |

Selecting only at the final checkpoint biases toward mature structure;
selecting only at step 0 biases toward whatever the init happens to group.
The overlap between the two token sets is itself a reportable result and is
cheap to compute.

anchor-final also matches the control: `pythia-1.4b-random` is norm-matched to
the *final* checkpoint, not to step 0 (`core/pythia_registry.py`).

---

## 2. Scope

**In:** Pythia only (`pythia-410m-step*`, later `pythia-1.4b-step*`,
`pythia-1.4b-random`).

**Frozen as Study A:** the existing 6-model results (gpt2-xl, gpt2-large,
gpt2-medium, bert-base-uncased, albert-xlarge-v2, albert-base-v2).
`status-5.md`'s verdict table is that record. The FFN-role architecture
comparison and the "causal robustness scales with model size" hypothesis are
Study A findings and stay there — the latter was always conditional on
blocker 5, which is now diagnosed as a metric artifact (§5).

**Deleted, not ported:**

| Target | Why |
|---|---|
| Group D entirely — `feature_signature.py`, `_run_group_D`, `_compute_feature_activations`, `p5_io.load_phase3`, `p5_io.load_phase4`, `--phase3-ckpt` / `--phase3-cache` / `--phase4-dir` | Phases 3/4 out of scope (INDEX.md 2026-07-18). Blocker 3 closes as **descoped**, not fixed. |
| `causal_tests._run_albert_with_hook`, `_use_legacy_albert_path`, `_attn_submodule`, `_attn_output_projection`, `_get_input_embeds` | ALBERT/BERT closed. `_get_input_embeds`'s BERT branch is infinite recursion; deleting is the fix. |
| `find_phase1_runs`'s `_iter_depth` tiebreak | ALBERT-only. Replaced by checkpoint-step resolution. |
| `run_5.DEFAULT_MODELS`, `--phase2i-dir` | Seven-architecture list; p2i dir already documented as unused. |
| `run_5._ov_profile_from_composed_weights`, `_ov_eigval_from_composed_weights`, `_ov_eigval_from_per_layer` | Dead — never called. The first calls an unimported `ov_spectral_profile` inside a bare `except: pass`. |

---

## 3. Blocker ledger (the original six)

| # | Original | Status |
|---|---|---|
| 1 | `merge_verdict` always `n/a` | **NOT fixed — root cause found 2026-08-06, see B11.** FIX-B7 relocated the plumbing correctly but the source data was already empty. Fixed for the rebuild in `anchors.load_cluster_tracking`. |
| 2 | OV values always `n/a` in C1 | **Partially fixed.** `composed_ov` path wired into `analyze_heads`. Latent defect: three dead helpers, one calling an unimported name inside a swallowing `except`. Delete them. |
| 3 | Group D blocked | **Closed as descoped.** No producer exists. |
| 4 | Group E probabilities round to 0.000 | **Open, decision made:** keep frozen-head decode as primary. Do not train the affine tuned lens — documented skip-to-output pathology for exactly this use case (status-5, 2026-07-19 note). |
| 5 | Identical `mean_frac_together` across interventions | **Diagnosed as a metric artifact,** not duplication. See §5 Group F. |
| 6 | bert-base Group F not run | **Moot** — BERT deleted. Root cause was the `_get_input_embeds` recursion. |

---

## 4. Bugs found 2026-08-06, ordered by cost on a Pythia run

**B1 — checkpoint-name substring collision in `p5_io.find_phase1_runs`.**
Match is `model_stem in name or model_stem_hyphen in name`. Phase 1 writes
run dirs as `{model_name}_{prompt_key}` (`run_1.py:487`), so stem
`pythia-410m-step1` matches **12 of the pilot's 27 checkpoints** — every step
whose digits begin with 1: 1, 16, 128, 1000, 11000, 13000, 15000, 17000,
19000, 100000, 120000, 143000. All collapse onto the same `prompt_key` keys,
and the
`_iter_depth` tiebreak is 0 for every Pythia dir, so the winner is
`Path.iterdir()` order — filesystem-arbitrary.
**Phase 5 would silently analyze a non-deterministic mixture of checkpoints.**
Blocks everything. **FIXED — W1**, `core/run_discovery.py`.
(An earlier draft of this entry said 10 checkpoints; the regression test
corrected it to 12.)

**B2 — `run_5._load_model` ignores the registry.** Calls
`from_pretrained(model_name)` with a `MODEL_CONFIGS` key and **no
`revision=`**. `"pythia-410m-step512"` is not an HF repo id; if it resolved it
would load `main` (= step 143000) regardless of the checkpoint under analysis.
`core/lm_loading.load_causal_lm` exists for this. Groups E and F only.

**B3 — prompt text is reconstructed by detokenization.**
`prompt_text = tokenizer.convert_tokens_to_string(run["tokens"])`
(`run_5.py:1069`). Detokenize-then-retokenize is not identity under NeoX BPE.
Every position index downstream (`baseline_hdb_labels`, `member_idxs`,
`token_idx`) assumes it is. Feed battery text by `prompt_key` from
`core/prompts.py` and assert the id sequence matches.
`core/battery_structure.py` exists for this class of drift.

**B4 — Group F averages over layers the intervention cannot have touched.**
`recluster_after_intervention` iterates the full chain; interventions install
`forward_pre` hooks at `target_layer` / `mid_layer`. Everything upstream is
byte-identical to baseline, pinned at `frac_together = 1.0` by construction —
roughly half the chain for `mid_layer`. This is blocker 5.

**B5 — intervention magnitude is unnormalized.** `steer_residual` adds
`alpha * direction` with unit-norm `direction` and `alpha=2.0`. Mid-depth
Pythia residual norms are O(10²) **and grow over training**, so any
"robustness increases with training" reading is confounded with norm growth.

**B6 — layer-index convention unstated on the intervention path.** Chain
indices are activation indices with 0 = embedding (`core/models.py`;
CHANGES.md). `blocks[target_layer]` is correct for `steer`/`patch` (block *k*
reads activation *k*), but for `ablate_head` the head producing structure at
activation index *L* lives in block *L−1*. Not obviously wrong — but unwritten.

**B7 — Group C2 is dead on Pythia.** `load_ffn_deltas` reads Phase 2's frozen
GPT-2-only decompose artifacts, returns `None`, and C2 silently degrades to
LDA + centroid directions. Replace with `core/sublayer_streams.py`.

**B8 — duplicate `v_alignment` import in `run_5.py`** (absolute lines 45–53,
relative line 59). Harmless today; binds two module objects if the package is
ever imported under another name.

**B9 — `pair_agreement` reload. RESOLVED (W2): it survives.**
`p1_io._save_clustering` (line 231) copies `lr["pair_agreement"]` wholesale
into `clustering.json`, `mutual_pairs` included. No D2-style gap here.

**B12 — `dual_reading.effective_rank_contribution` hardcodes `mode="raw"`.**
`metrics.effective_rank`'s raw mode runs the SVD on unnormalized vectors, so
its spectrum is dominated by the highest-norm tokens. A per-particle
contribution in raw mode is therefore largely "how big is this token's norm."
Correct for the degeneracy gate it was written for; wrong for a
cross-checkpoint comparison, because residual norms grow across training —
D1's confound in per-particle form. Not a bug in that function's own terms, so
it is left alone; `sweep_geometry.particle_rank_contributions` takes an
explicit mode with no hidden default and computes both.

**B11 — blocker 1's real cause: `load_phase1_run["events"]` is the wrong
events. Found while checking the W3b loader contract.**

Phase 1 emits two event schemas:

| file | shape | who reads it |
|---|---|---|
| `trajectory.json` -> `cluster_tracking.events` | `{"layer_from": int, "layer_to": int, "merges": [[prev_ids], curr_id], ...}` | **nobody, until now** |
| `events.json` (Phase 3 bridge) | `{"merge_layers": [...], "energy_violations": {...}}` | `p1_io.load_phase1_run` |

`p1_io._load_events` normalises the second into
`[{"type": "merge", "layer_name": "9", "layer_from": "9"}]` — `layer_from` a
**string**, and **no `merges` key at all**. `load_phase1_run` never reads the
first. So `select_cluster._merge_event_for_trajectory`, which reads
`run["events"]`, returned `None` for every trajectory on every model.

That is blocker 1. `status-5.md` records FIX-B7 as the fix — routing
`merge_verdict` through `select_cluster`'s computation rather than
re-deriving it in `_run_group_B`. Correct plumbing, wrong cause: the value
being plumbed was already empty.

**Combined with B10, two of six selection criteria were dead — `merge`
(weight 3.0) and `semantic` (weight 2.0), 5.0 of the 9.0 scale.** Selection
ran on lifespan, size, sibling and preferred_prompt alone. This is the second
independent reason design-5.md's "9.000 for 4/6" cannot be right, and it
means Study A's selection-stability claim needs re-deriving before it is
repeated anywhere.

Fixed for the rebuild: `anchors.load_cluster_tracking` reads the real schema,
and `token_sets._assert_tracking_events` **raises** on the bridge schema
rather than returning None. Tolerance is what hid this; the two schemas must
not look alike to a consumer.

**B10 — the semantic selection criterion has never fired. Found while
checking B9.**
`clustering.pair_hdbscan_agreement` emits `tag` as a backward-compat alias for
`cross_method_tag`, whose value set is exactly
`{"same_cluster", "diff_cluster", "noise"}`.
`select_cluster._score_trajectory` tests `p.get("tag") == "semantic"` — in the
main branch **and** in the `SEMANTIC_PAIR_MIN_COUNT` fallback. That comparison
is never true, so `s["semantic"]` is `0.0` for every trajectory on every
model, and `SCORE_WEIGHTS["semantic"] = 2.0` — 2 of a 9.0 maximum — is dead
weight. Selection has been running on five criteria, not six.

The correct field is `ext_semantic_tag == "ext_semantic"`. Note where that
lands: `ext_semantic` is the quantity status-1 D6 flags as having a
*training-dependent* reference frame (cosine Gram of the model's own layer-0
embeddings against a fixed 0.5 cutoff). So repairing the criterion and
applying D6's frozen-reference fix are the same edit — do not do the first
without the second, or the criterion starts working and immediately drifts
across checkpoints.

**Consistency check owed:** design-5.md reports "all six models achieve
near-perfect scores (9.000 for 4/6)". 9.000 is the sum of all six weights, and
is unreachable if the 2.0 semantic term is structurally zero. Either those
numbers came from a different code path or the reported maximum is wrong.
Resolve before any Study A claim about selection stability is repeated.

---

## 5. What each group becomes

**A — structural profile → per-particle geometry.**
- Use **normed** effective rank, not raw. status-1 D1: raw mode mixes
  directional collapse with residual-norm growth, and Pythia's norms grow
  across training — a raw-rank trajectory over 143k steps measures norm growth
  as much as geometry.
- `core/dual_reading.effective_rank_contribution` is already a leave-one-out
  per-point contribution to a population metric. Use it.
- Emit rows into `core/particles.py`'s table. Cluster-level numbers become
  groupbys; Phase 5c reads the same table.

**B — V-alignment → projector policy must be declared before it runs.**
`_build_v_projectors_from_ov` averages `ov_total` over layers into one global
V per model. On a sweep that is 27 different V's, and `U_attractive` /
`U_repulsive` rotate between them. Projecting onto "the attractive subspace"
at step 512 and step 143000 projects onto **different subspaces**, and the
difference reads as particle motion.

| Policy | Question it answers |
|---|---|
| `own_frame` | where does this particle sit relative to *this* checkpoint's dynamics |
| `reference_frame` (V fixed at 143000) | how does this particle move in a common frame |

Pick one per figure; record it as a `core/frames.FrameSpec` field, not a
comment. Also: replace `v_alignment.estimate_effective_beta` with
`core/beta_eff.estimate_beta_all_heads` — the current one uses
`np.triu_indices(n, k=1)`, which causal attention masks to zero, and has been
returning ~0 for every head on every causal model.

**C1 — head contributions.** Highest-value group under the new frame:
status-1 D2 shows per-head Fiedler classification is vacuous (all 432 rows
STABLE-CLUSTER; thresholds calibrated for raw λ₂ on [0,1] meeting deviations
in ±0.05), so C1's cohesion ranking is the only non-degenerate per-head
discriminator in the project. Check its thresholds against Pythia's actual
distribution before it inherits the same failure. Question: does the
single-dominant-attractor-head result *emerge*, and when — against the 8→16,
256→512, 1000→3000 transitions.

**C2 — the biggest available upgrade.** Kill `load_ffn_deltas`; read
`core/sublayer_streams.py`, which already implements parallel-residual
capture: `post_attn = x + attn_delta`, `post_ffn = x + mlp_delta`, both from
the same pre-block input, neither downstream of the other, Δx = attn_out +
ffn_out **exactly**. Strictly better than the GPT-2 module ever produced.
Serves three open items at once: this group, status-1 open item 6 (late
severity peaks 0.170 @ 60000, falls to 0.101 @ 143000 with counts flat), and
status-2's blocking parallel-residual module (3 dead classifier branches,
20% of `v_score`).

**E — semantic readout.** `frozen_head_decode` is already Pythia-aware
(`embed_out` + `final_layer_norm`). Keep frozen-head primary (blocker 4
decision). Add `core/functional_distance.py` as arbiter when geometric and
functional clusterings disagree — it consumes cached decoded distributions,
which E already produces. **Caveat to state in output:** `embed_out` is
untied, so in the random control the head stays trained while the body is
randomized (`core/lm_loading` flags this). Correct for a readout contrast,
not "a fully random LM."

**F — causal, at anchors only.** Fixes required before it runs: B4 (restrict
the mean to layers strictly downstream; report per-layer regardless), B5
(scale `alpha` to the layer's median residual norm, record realized relative
perturbation), B2, B3.

**G — controls.** Replace the hand-rolled random baseline in
`sibling_contrast.py` with `core/nulls.py` (`shuffled_dimension_null`,
`label_permutation_null`, `sigma_from_null`). The three-tier ordering
primary > sibling > random licenses reading A–F as being about a real object
and currently has no significance statement. First row of the missing
falsification table.

---

## 6. Constants recalibration

The token-set reframe dissolves the worst problem: `MIN_LIFESPAN = 6` gates on
a quantity whose distribution *moves over training* (status-1: mean lifespan
7.0 → 4.5, so at late checkpoints the mean sits below the gate and survivor
bias grows monotonically with step). With a token set anchored once, there is
no per-checkpoint lifespan gate to bias.

Still open:

- **Semantic criterion (weight 2.0) rides on a confounded frame.**
  `ext_semantic` is defined against the model's own layer-0 embedding Gram at
  a fixed 0.5 cutoff (status-1 D6) — the reference frame trains. Anchored
  selection makes this a one-time cost, but the anchor's embedding still
  defines "semantic." Use a frozen final-checkpoint reference and say so.
- `LIFESPAN_FULL_SCORE = 18` is calibrated to ALBERT-xlarge's 48-iteration
  depth against Pythia-410M's 25 analyzed layers. Recalibrate or drop.
- `PREFERRED_PROMPTS` still resolves on the Pythia battery. Record token count
  per prompt in the selection artifact — status-2 flags `short_heterogeneous`
  as materially shorter, so its noisier energy estimate should be visible
  rather than inferred.

---

## 7. Questions this makes newly answerable

1. **Carrying capacity is invariant but turnover is not — same particles
   cycling faster, or different particles?** status-1: max-alive holds at
   50–55 across all 27 checkpoints while mean lifespan falls 7.0 → 4.5 and
   births rise 113 → 164. Cluster-level statistics cannot distinguish these.
   A particle table can, by groupby. Sharpest unaddressed thing in the Phase 1
   results. **Instrument built (W4b): `turnover_decomposition`.** Validated on
   two synthetic sweeps with identical falling mean lifespan — the
   same-particles case gives J=1.000 at every threshold and rank correlation
   1.000; the different-particles case gives J=0.000 first-vs-last and rank
   correlation −0.667. Awaiting the real sweep.
2. **Particle biography.** *(Instrument built, W4b: `particle_biography`.)*
   Per token position: layer of first stable-cluster
   membership, and how that date moves across the sweep. Tests directly
   whether the step-512 plateau-onset flip (weight-level SD 0.00 →
   content-driven SD 3.31 in one interval) is a change in *which* particles
   cluster or in *when* they do.
3. **The step 8→16 transient, per particle.** Rank 6.5 → 2.1, mass 0.016 →
   0.58, confined to layers 21–23, recovered by 512, unpredicted, resolved by
   a single interval. Which particles collapse — the ones that later form
   stable clusters, or the ones that end up unclustered? Answerable at
   anchor-init with no new machinery.
4. **Mid-network mass minimum.** Step 143000 / `wiki_paragraph`: layers 9–14
   at mean mass 0.0007 against a layer-0 duplicate-token value of 0.0149 — the
   trained model separating even identical tokens by mid-depth. Are the
   separated pairs the same tokens across checkpoints, and do they hand off to
   5c's unclustered population?
5. **Effective rank as a per-particle budget.** The arc — collapse to 2.1,
   recovery, overshoot to 40.4 @ 3000–5000, monotone decline for 140k steps —
   is 5c's fixed-budget hypothesis with a time axis. `effective_rank_
   contribution` in **normed** mode is the un-confounded per-particle form of
   the quantity D1 says can't currently be written as a result.

---

## 8. Work items

Order matters: 1 blocks everything else.

- [x] **W1 — checkpoint-aware run discovery. DONE 2026-08-06.**
      `core/run_discovery.py` (new, torch- and matplotlib-free) +
      `tests/test_core_run_discovery.py`, **32/32 passing, run for real.**
      Anchored `-step{N}` parsing; `{prompt_key: {step: RunRef}}` index;
      manifest > geometry > dirname provenance recorded per run;
      `DuplicateRunError` instead of silent last-wins; step 0 and the
      norm-matched random control kept as separate slots.
      `p5_io.find_phase1_runs` becomes a thin wrapper or is deleted in W3.
      Fixes B1.
- [x] **W2 — `pair_agreement` reload check. DONE 2026-08-06.** Survives (B9
      closed). Surfaced **B10** — the semantic criterion has never fired.
- [x] **W3 — token-set selection. DONE 2026-08-06.**
      `p5_single_mstate_analysis/token_sets.py` (new) +
      `tests/test_p5_token_sets.py`, **44/44 passing, run for real**
      (76/76 with W1's suite). `select_cluster.py` is superseded.
      Two decisions taken, both recorded in §11.
      **Not yet done, tracked as W3b:** wiring into `run_5.py`, the
      `RunRef` -> trajectories/events/labels loader, and emission into
      `core/particles.py`. `token_sets.py` deliberately takes plain
      arrays/dicts so it is testable without a run directory; the loader is
      a separate, thinner piece.
- [x] **W3b — wire W1+W3 together. DONE 2026-08-06.**
      `p5_single_mstate_analysis/anchors.py` (new) +
      `tests/test_p5_anchors.py`, **34/34 passing, run for real**
      (110/110 across all three suites). Contains `load_cluster_tracking`
      (fixes B11), a labels-and-chains-only run reader that loads no
      activations, the per-prompt two-anchor driver with overlap, and
      emission into `core/particles.py`.
      **Owed upstream:** `load_cluster_tracking` is a Phase 1 artifact
      reader and belongs in `p1_io.py` beside `load_phase1_run`. It lives in
      Phase 5 for now so this rebuild does not edit a frozen phase. Move it
      when Phase 1 is next touched — and while there, decide whether
      `load_phase1_run["events"]` should be renamed
      (`bridge_events`?) so no future consumer makes the same assumption.
- [x] **W4 — Group A on the sweep. DONE 2026-08-06.**
      `p5_single_mstate_analysis/sweep_geometry.py` (new) +
      `tests/test_p5_sweep_geometry.py`, **55/55 passing, run for real**
      (165/165 across four suites). Per-particle geometry for a frozen token
      set at every checkpoint; all three roles measured identically;
      normed and raw side by side; frame- and pos0-stamped.
      **Not yet done, tracked as W4b:** joining these records to
      `core/particles.py` rows (the emission path exists in `anchors.py`;
      what is missing is writing the geometry columns into it) and the
      per-checkpoint figures.
- [x] **W4b — join sweep geometry into the particle table. DONE 2026-08-06.**
      `p5_single_mstate_analysis/particle_join.py` (new) +
      `tests/test_p5_particle_join.py`, **44/44 passing, run for real**
      (252/252 across six suites), against the real `core/particles.py`.
      Contains the sweep table builder, `particle_biography` (§7 item 2) and
      `turnover_decomposition` (§7 item 1). Plotting deferred — the numbers
      are the deliverable and the figures follow the real run.
- [x] **W5 — Group G on `core/nulls.py`. DONE 2026-08-06.**
      `p5_single_mstate_analysis/tiers.py` (new) +
      `tests/test_p5_tiers.py`, **43/43 passing, run for real**
      (208/208 across five suites). Three-tier contrast, label-permutation
      and shuffled-dimension nulls, Nσ verdicts, and the falsification table
      `status-5.md` was missing. Tests run against the real `core/nulls.py`,
      not a stub.
- [ ] **W6 — Group C2 on `core/sublayer_streams.py`.** Fixes B7.
- [ ] **W7 — Group B** with the projector policy declared as a `FrameSpec`
      field; `beta_eff` swap.
- [ ] **W8 — C1 threshold calibration**, then E.
- [ ] **W9 — Group F at anchors**, after B2/B3/B4/B5.
- [ ] **W10 — deletions** (§2 table) and falsification table.

---

## 11. Decisions taken during implementation

**D-1 (W3): selection is per-prompt; no single anchor prompt is chosen.**
Token positions only index into one prompt's sequence, so a token set is
inherently per-`(prompt, anchor)` and there is no cross-prompt competition to
adjudicate. `PREFERRED_PROMPTS` is therefore removed as a scoring term rather
than recalibrated. `repeated_tokens` stays excluded as a hard gate — status-2
V4: `". "` x ~264, one distinct token id, degenerate at embedding, so its 27
zero-violation runs are the `eff_rank >= 3.0` guard firing, not monotonicity.
`anchor_overlap` refuses to compare token sets across prompts rather than
returning a number that would look valid.

**D-2 (W3): the semantic criterion is dropped from scoring and kept as an
annotation.** It has never contributed (B10), so dropping it changes no
existing result. Repairing it means switching to `ext_semantic_tag`, whose
reference frame is the model's own layer-0 embedding Gram against a fixed
cutoff — D6: the frame trains. A repaired criterion would drift across
checkpoints for reasons unrelated to any cluster, which is strictly worse
than one that does nothing, because it would look like it worked. Recorded as
`annotations["ext_semantic_frac__unfrozen_reference"]`, with the caveat text
in an adjacent key so a consumer cannot pick it up without reading it.

**D-3 (W3): the object is CORE membership, not union or single-layer.**
A trajectory's membership churns along its chain. `positions` = tokens in the
cluster at >= `CORE_MEMBERSHIP_FRACTION` (0.75) of alive layers — the
particles that actually stayed together. `union_positions` and `churn` are
recorded alongside, because a high churn with a healthy core (tokens drifting
around a stable centre) and a high churn with a small core (no stable centre)
are different findings and the scalar alone cannot distinguish them. Strict
intersection is available (`min_fraction=1.0`) and returns empty rather than
raising when a trajectory has no persistent member.

**D-4 (W3): constants recalibrated for Pythia-410M's 25 analyzed layers.**

| Constant | Was | Now | Why |
|---|---|---|---|
| `MIN_LIFESPAN` | 6 | 4 | Applied once at one anchor, not per checkpoint, so status-1's 7.0 -> 4.5 drift no longer biases a survivor set. Lowered because anchor_init's lifespan distribution is a different object from anchor_final's. |
| `LIFESPAN_FULL_SCORE` | 18 | 12 | 18 was calibrated to ALBERT-xlarge's 48-iteration depth; against 25 layers the term was effectively unsaturable. |
| `SCORE_WEIGHTS` sum | 9.0 | **1.0** | A perfect score is now 1.0 by construction. The old sum-of-weights maximum is what made design-5.md's "9.000 for 4/6" unreadable — a maximum that has to be recomputed whenever a criterion is added or silently dies is not a maximum. |
| criteria | 6 | 4 | lifespan, merge, size, sibling. See D-1, D-2. |

**D-5 (W3): the random control is seeded and frozen, not redrawn.**
`random_control_positions(seed=...)` is size-matched, disjoint from primary
and sibling, and fixed across every checkpoint the sweep touches. A control
redrawn per checkpoint would add variance exactly where the three-tier
comparison (primary > sibling > random) needs none.

**D-7 (W3b): selection loads no activations.** `load_run_for_selection`
reads `hdbscan_labels.json`, `trajectory.json`, `geometry.json` and
`clustering.json` only. Selection is a labels-and-chains operation, so a whole
27-checkpoint sweep resolves without touching a single
`(n_layers, n_tokens, d_model)` array.

**D-8 (W3b): partial coverage is reported, never fatal.** `build_anchor_token_
sets` raises nothing. A prompt whose anchor has no run, no usable clustering,
or no passing trajectory lands in `bundle.skipped` with the reason. Late
Pythia checkpoints having no passing trajectory is a fact about the
checkpoint (mean cluster lifespan 4.5), and a driver that aborted on the
first miss would make that fact invisible.

**D-9 (W3b): control seeds are per-(prompt, anchor), not global.** A single
shared seed draws the same control positions at both anchors, correlating the
two controls for no reason. Derived as
`control_seed + 1000*step + hash(prompt) % 997`.

**D-10 (W3b): the particle table keeps the complement.** Every token gets a
row at every layer, not just set members, tagged
`token_set_role in {primary, sibling, control, none}`. The complement is the
object design-5c is about — the unclustered population, "not a failure mode
but a distinct phase" — and dropping it here would make that population
unrecoverable from the table.

**D-11 (W4): the sink is excluded by MASK, never by reindexing.**
`pos0_policy="excluded"` is the default for population aggregates — position 0
is the NeoX attention sink, carries a norm one to two orders above every other
token, and can single-handedly set the raw spectrum. But the exclusion is
applied as a boolean mask over full-length arrays. Dropping row 0 would shift
every token position by one, and token position is the identity of a particle
across the entire sweep; there would be no way to detect the shift
downstream. A position excluded by policy gets **NaN**, not 0.0 — "not
measured" and "measured, contributes nothing" are different claims. If
position 0 is itself a token-set member, `sweep_geometry` emits a note rather
than handling it quietly.

**D-12 (W4): every quantity is reported in both normed and raw.**
`normed` is the cross-checkpoint-comparable one (D1). `raw` is kept because
raw-minus-normed divergence *is* the norm-growth signal, and reporting only
the corrected number would make D1 invisible rather than fixed.
`contribution_modes` lets the raw pass be dropped once it has been
characterised for a given sweep — measured cost is ~570 ms per layer at
410M/wiki_paragraph scale with both modes, so ~6.5 min per token set over a
25-layer 27-checkpoint sweep, ~1.7 h for two anchors across eight prompts.

**D-13 (W4): all three roles are measured identically.** Group G's three-tier
ordering (primary > sibling > random) is only readable if the tiers went
through the same code path. The original Group A measured the primary alone
and the comparison was assembled afterwards from differently-derived numbers.

**D-14 (W4): a token-count mismatch aborts that checkpoint.** If a
checkpoint's activations have a different token count from the anchor's, the
frozen positions no longer refer to the same particles and the correct
response is to skip with a reason, not to measure the wrong tokens. Cheap
insurance against the B3 class of tokenizer drift.

**D-15 (W5): the control tier and the null are different objects and both
are produced.** The control tier is ONE fixed random draw (D-5) measured at
every checkpoint — a comparison *object* with a developmental trajectory, whose
variance across checkpoints is signal. The label-permutation null is 200 draws
at ONE checkpoint, discarded immediately — a *distribution* whose spread is the
yardstick. Reporting only the control gives an ordering with no error bar;
reporting only the null loses the developmental comparison. Conflating them is
the easy error and the module is built around not making it.

**D-16 (W5): `ordering_holds` is a tri-state.** `True` / `False` / `None`, and
`ordering_consistency` counts evaluable cells separately from total cells. "The
claim failed" and "the claim could not be evaluated" must not be the same value
in a falsification table — a claim that held 4 of 4 times out of 27 cells is a
different object from one that held 4 of 4 out of 4. `n/a` rows are printed,
never dropped.

**D-17 (W5): correction — the pos0 exclusion is NOT about norm inflation.**
An earlier draft of `tiers.py` argued the sink had to be kept out of the
permutation pool because its norm would widen the null. That is wrong, and a
test caught it: every set-level statistic here is computed on L2-normalized
rows, so a 100x-norm token contributes a unit vector like any other and the
null's spread moves from 0.0877 to 0.0880 — nothing. The reasons that do hold
are (1) raw-mode statistics *are* sink-sensitive and the policy has to be one
decision applied identically across sphere and raw quantities, and (2) the
sink is functionally a different kind of particle (it absorbs attention mass
on behalf of every position, no BOS prepended), so whether it belongs in "a
random size-matched subset of ordinary tokens" is a question about what the
null is a null *of*. Both the module docstring and the test now say this.
Recorded because the sphere-frame invariance is easy to mistake for the policy
doing nothing.

**D-18 (W4b): an empty side gives Jaccard `None`, never 0.0.**
`clustered_set_overlap` thresholds on per-particle persistence. If the
threshold exceeds a checkpoint's maximum persistence, that side's set is
empty and the Jaccard is a mechanical 0 — which reads identically to
"completely different particles" while actually meaning "the threshold is
above this checkpoint's ceiling". Found on the first worked example: a
synthetic sweep where the SAME particles clustered throughout reported
J(k=8)=0.000 between step 0 and step 512, purely because nobody at 512
persisted 8 layers. Now returns `None` with a `degenerate` field naming which
side emptied, and the report prints an explicit note. This is the same class
of error as B10/B11 — a structurally-guaranteed value masquerading as a
measurement — caught before it reached a result this time.

**D-19 (W4b): distinct-label means are computed among clustered particles.**
The never-clustered complement contributes a structural 0 and drags the plain
mean toward it. On the worked example both checkpoints reported 0.40 while
the clustered particles' actual value was unchanged at 1.0. "Cycling through
more clusters" is a claim about particles that cluster, so
`mean_distinct_labels_among_clustered` and `mean_layers_among_clustered` are
the reported columns; the all-token means are kept alongside because their
gap is the complement's size.

**D-20 (W4b): `turnover_decomposition` returns no verdict.**
The two readings — same particles cycling faster vs. different particles
clustering — are not exhaustive, and real numbers can land between them.
Naming a winner would be the premature collapse this rebuild keeps finding in
the old results. There is a test asserting no `verdict` or `conclusion` key
exists.

**D-6 (W3): rejection is a diagnosis, not an empty result.**
`rank_trajectories` returns a per-gate tally alongside the passing pool, and
`SelectionRejected` carries it. Late Pythia checkpoints having no passing
trajectory is a fact about the checkpoint; "which gate did it" is the useful
thing to print, and `select_cluster`'s `return None` discarded it.

---

## 9. Verification conventions

Sandbox has numpy, no torch, no pytest, no network. Following the project's
existing convention (CHANGES.md):

- Pure numpy / pure-path logic: written **and run** here, via a manual test
  runner. Report pass counts honestly.
- Anything touching a live model: written to
  `tests/*_smoke.py` (`SMOKE_REAL_DEPS=1 pytest -m smoke`), **not executed
  here**. Say so explicitly at every such point.

---

## 10. Changelog

- **2026-08-06** — Document created. Framing decision (token set, two anchors)
  recorded. Blocker ledger re-checked against source. Nine new bugs logged
  (B1–B9). Work items W1–W10 defined.
- **2026-08-06** — **W1 done.** `core/run_discovery.py` +
  `tests/test_core_run_discovery.py`, 32/32 passing, executed. B1 fixed. The
  regression test corrected B1's own collision count from 10 to 12.
- **2026-08-06** — **W2 done.** B9 closed (pair_agreement persists). **B10
  logged:** the semantic selection criterion tests for a `tag` value that the
  producer never emits, so 2.0 of the 9.0 selection score has always been
  zero. Consistency check owed against design-5.md's reported 9.000 scores.
- **2026-08-06** — **W3 done.** `p5_single_mstate_analysis/token_sets.py` +
  `tests/test_p5_token_sets.py`, 44/44 passing, executed. `select_cluster.py`
  superseded. Six implementation decisions recorded as D-1..D-6 in §11.
  Also folded in, as side effects of writing the replacement rather than as
  separately-tracked fixes: the FIX-B7 merge-schema unpacking now happens
  once in `merge_event_for` instead of being re-derived per consumer; and
  `pick_sibling`'s tie-break is a total order, where
  `select_cluster._pick_sibling` dereferenced `best_id` while it could still
  be None and produced input-order-dependent results.
- **2026-08-06** — **W3b done.** `p5_single_mstate_analysis/anchors.py` +
  `tests/test_p5_anchors.py`, 34/34 passing, executed. 110/110 across all
  suites. **B11 logged and fixed:** blocker 1's real cause was
  `load_phase1_run` reading the Phase 3 bridge `events.json` instead of
  `trajectory.json -> cluster_tracking.events`, so the merge criterion
  returned None on every model. With B10 that puts two of six selection
  criteria — 5.0 of the 9.0 scale — dead throughout Study A.
  Decisions D-7..D-10 recorded.
- **2026-08-06** — **W4 done.** `p5_single_mstate_analysis/sweep_geometry.py`
  + `tests/test_p5_sweep_geometry.py`, 55/55 passing, executed. 165/165 across
  all suites. **B12 logged.** Decisions D-11..D-14 recorded. Leave-one-out
  effective rank computed from the (n,n) Gram's eigenvalues rather than a
  fresh (n,d) SVD, with the equivalence checked against the SVD definition in
  tests rather than assumed. Sweep cost measured, not estimated.
- **2026-08-06** — **W5 done.** `p5_single_mstate_analysis/tiers.py` +
  `tests/test_p5_tiers.py`, 43/43 passing, executed. 208/208 across all
  suites. Group G now produces an Nσ verdict and a falsification table
  instead of a bare ordering. Decisions D-15..D-17 recorded — D-17 is a
  correction to a claim this rebuild made two work items ago and a test
  disproved.
- **2026-08-06** — **W4b done.** `p5_single_mstate_analysis/particle_join.py`
  + `tests/test_p5_particle_join.py`, 44/44 passing, executed against the real
  `core/particles.py`. 252/252 across all suites. §7 items 1 and 2 now have
  instruments, validated on synthetic sweeps constructed to be
  indistinguishable at the cluster level. Decisions D-18..D-20 recorded —
  D-18 and D-19 were both found by running a worked example rather than by
  the unit tests, which is worth noting as a method: the tests passed and the
  output table was still misleading.
