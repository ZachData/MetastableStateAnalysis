# Phase 1b — revision log

Written against the Phase 1 rebuild and the Pythia migration. Every item
below is either a defect with an observable consequence, a capability the
rebuilt core made available, or a question that became askable.

Verification status is stated per item. Nothing model-touching has been run.

---

## A. Defects with an observable consequence

### A1. `--models` defaulted to the entire registry
`run_1b.py` had `default=list(MODEL_CONFIGS.keys())` and `choices=` over the
same list. That registry is now 48 entries; a bare invocation meant 37
checkpoint downloads across every prompt. run_1 fixed exactly this with
DEFAULT_MODELS / MODEL_GROUPS and deliberately dropped `choices=` so a typo
produces a readable error.

Resolver extracted to **`core/model_selection.py`** rather than copied a
third time. `run_1.py` should delegate to it; that edit is mechanical and
deliberately not bundled here.

*Verified:* default is now 7 models; `pythia-410m-pilot` expands to 27.

### A2. `--fast` was dead code
It assigned `_cfg.ALBERT_SNAPSHOTS` at module scope, but the name had been
bound at import. The override never reached the extraction call, so `--fast`
silently ran the full 28-snapshot sweep. Snapshots are now read off the
config module at call time.

*Not verified:* requires a model load.

### A3. The ALBERT cross-reference never resolved — and this was reported as a finding
Phase 1b built `{model}_{prompt}_d{depth}`. Phase 1 writes
`{model}_{depth}iter_{prompt}` (`run_1.py`, `effective_model_name`). No
ALBERT extended run ever resolved, so `hdbscan_labels` never loaded and Block
2's nesting test had nothing to test.

**status-1b.md records that outcome as "Inconclusive for ALBERT".** It was a
path bug. Resolution now goes through `p1_io.find_phase1_run_dir`, and the
live path emits Phase 1's own stem.

### A4. An inverted verdict field
`cone_at_long = mean(cone_fraction) < 0.5`, published as
`cone_collapse_regime_at_long_prompts` — True exactly when cone-collapse was
*rare*. The derived `paper_alignment` string read the variable rather than
the name and was correct, so the JSON carried a correct verdict beside a
boolean asserting its opposite. Renamed to `split_regime_at_long_prompts`,
with the old key retained as its correct negation.

### A5. Swapped crossref keys
`mean_crossing_at_violation` was published as `mean_crossing_at_merge_events`
and `mean_axis_rotation_at_merge` as `mean_axis_rotation_at_violation_layers`.
Both now carry their own names, and both off-event baselines are reported
beside them — a crossing count at violation layers means nothing without the
count at non-violation layers.

### A6. A hardcoded narrative contradicting the run
`_write_cross_run_md` asserted that long prompts enter a split regime at
mid-depth and that the bipartition is real geometric structure. The run found
the opposite at every layer of every model (status-1b blocker 2). The text is
generated from the verdict dict now.

### A7. Two Laplacians
`bipartition_detect` built its own with a hardcoded `1e-4/n` floor;
`core.metrics.fiedler_and_eigengap` — what Phase 1 ran — has no floor. Phase
1b's stated job is explaining Phase 1's k=2, on a different graph.

Now one implementation with `connectivity_floor` as a recorded parameter,
defaulting to `0.0` (exact Phase 1 behaviour).

*Verified:* on two tight antipodal clusters, λ₂ = 1.7e-16 unfloored vs
8.4e-6 floored — a factor of 4.9e10. Worth noting the effect is
construction-specific: X-and-−X on *random* directions does not disconnect
(λ₂ = 0.28). The floor matters precisely for the geometry this phase studies.

### A8. A capability nearly deleted during de-duplication
`clip_negative=False` was a real code path (signed Laplacian). The first
delegation attempt raised on it, justified by a claim true of `core.metrics`
but false of `bipartition_detect`. Carried through instead — and the signed
Laplacian can contain inf/nan, which `scipy.eigh` raises on rather than
returning, so `fiedler_and_eigengap` now guards it.

*Verified:* degrades to the standard degenerate return instead of raising.

### A9. Hungarian tie-breaking could invert a whole run
`design-1b.md` claimed Block 1 reuses `cluster_tracking.match_layer_pair`; it
carried its own Jaccard pair. Wiring the delegation up surfaced a live
hazard: on an exact tie the solver returns either pairing, and it did on 4 of
500 random label pairs. Since `align_hemisphere_labels` chains anchors
forward, each would have flipped hemisphere labelling for the rest of a run
on a coin toss.

*Verified:* documented tie-break to identity enforced; 0 disagreements in
3000 pairs.

### A10. `first_stable_layer` contradicted its docstring
Docstring said "matches its final-layer assignment"; the code compared
against the most-held label. A token that switches once and then stays was
recorded as never stable.

*Verified:* on a 5-layer token switching at layer 3, `final` gives 3 and
`dominant` gives −1.

### A11. Dead duplicate branch
`detect_events` had unreachable copies of birth/collapse after the `continue`
statements that preceded them. Removed.

### A12. Persistence was foreclosed, not measured
`compute_persistence_lengths` hardcoded `"strong_bipartition"`, which
cone-collapse makes unreachable, so every persistence length was 0 and the
statistic looked measured. Vocabulary parametrised via `regime_key`.

*Verified:* on a cone-compatible two-cluster stack, antipodal vocabulary
gives max persistence 0, relative gives 5.

---

## B. Claims that were weaker than reported

### B1. Block 0's null was partly structural
`strong_bipartition` requires `centroid_angle >= pi/2`. Under cone-collapse —
which Block 3 reports universally — two centroids inside one open half-space
essentially cannot be `pi/2` apart. "0% strong bipartition" and "100%
cone-collapse" are close to the same test run twice, not two independent
results, which is how `design-1b.md` framed them.

`classify_regime_relative` asks the question that survives cone-collapse: is
between-half similarity materially below within-half, whatever the absolute
angle? Both classifiers now run and are reported side by side.

*Verified:* two genuinely separated clusters 60° apart (separation ratio
0.45, minority 0.5) classify as `weak_bipartition` under the old rule and
`separated` under the new one. **"0% strong bipartition" was never the same
claim as "no bipartition".**

### B2. Block 3 had no null and PCA was documented backwards
The old note claimed the cone question "is invariant under orthogonal
projections". It is not. A reduced-space witness lifts *exactly*
(`X @ Vt[:k].T @ w_r` preserves every inner product), so a cone_collapse
verdict under PCA is sound; the converse fails, so a split verdict may be a
projection artifact. `escalate_on_split=True` re-solves at full d in the one
direction that can lie.

Separately, n points in d dimensions separate for free unless they positively
span, so "100% cone-collapse at PCA-64" may have been dimension counting.
Two matched nulls added.

*Verified:* PCA lift exact to 1e-9. On a test cap, 100% of shuffled-dimension
draws are cone-collapsed but 0% of uniform draws are — so the result
discriminates, and the uniform null is *degenerate* (zero variance), which
makes `z` nan by design and is why a cone-fraction statistic carries it.

---

## C. Carried over from the Phase 1 rebuild

| Item | Status |
|---|---|
| `FrameSpec` recorded on every Block 0 result | done |
| `core.metrics` delegation | done (A7) |
| `core.nulls` in Block 3 | done (B2) |
| `ParticleTable` emission, one row per (layer, token) | done, round-trips |
| `core.io.write_manifest` | wired, **unverified** |
| `p1_io` artifact resolution instead of hardcoded filenames | done (A3) |
| `--from-phase1`: reuse saved activations + Fiedler vectors | wired, **unverified** |
| Artifact prefix `phase1h_` → `phase1b_` | done |

LN frames are **not** wired through `run_1b`. `apply_frame` requires
per-model LN parameters, which the current entry point does not thread. The
substantive test this enables — if the k=2 axis is anisotropy, it should
attenuate under the frame attention actually reads — is left open
deliberately rather than half-wired.

---

## D. Pythia

- **Checkpoint axis.** `aggregate_by_checkpoint` groups families and reports
  against log₁₀(step+1). Previously a 27-checkpoint pilot rendered as 27
  unrelated models. `pythia-1.4b-random` correctly produces no step and stays
  off the axis.
- **pos0.** `--drop-pos0` excludes the NeoX attention sink from Block 3's LP;
  `binding_tokens` records which tokens hold the half-space up, in original
  indices.
- **Long-prompt threshold.** Was a hardcoded `n_tokens > 100`. Now
  `LONG_PROMPT_TOKENS`, recorded in the verdict, because token counts are
  tokenizer-dependent and the threshold silently changes meaning across
  families.
- **Layer 0** is the embedding output and is still averaged into per-model
  means. Not addressed.

---

## E. New: axis identity, and a defect found in it

`axis_identity.py` maps the token-space Fiedler vector to activation space
(`a = X^T f / ||X^T f||`) so it is comparable across layers, checkpoints, and
against PC1.

**The first version of this module reproduced defect B1.** It asked whether
the axis is the *mean token direction* and gave `redundancy` a
`"mean_direction"` branch. Measured across every fixture, |cos(axis, mean)|
came out between 0.000 and 0.085 — the branch is unreachable.

The reason is structural. The Fiedler vector is the second eigenvector of the
normalized Laplacian, hence orthogonal to the first (`D^(1/2)·1`); measured
`<f, D^(1/2)1>` is 1e-16 to 1e-1 against `||f|| = 1`. A mean-zero coefficient
vector makes `X^T f` cancel whatever every token shares.

So the "is the axis just the anisotropy direction?" question, as originally
posed, has the answer "no, by construction" and is not worth asking. The
revised module:

- keeps `cos_axis_mean` as a **degeneracy diagnostic** (expected ~0; a large
  value means a disconnected graph or unconverged eigensolve),
- asks redundancy against **centered PC1** and the **top-k PC subspace**
  (`pc_subspace_fraction`), both mean-removed and therefore commensurable,
- reports **`isotropic_cos` = 1/√d** beside every cosine, because 0.3 is not
  obviously different from chance at d = 1024.

A second defect in the same module: the sign convention oriented the axis by
the centroid difference between the positive-f and negative-f groups. Under
`f → −f` the axis negates *and* the groups swap, so the test gives the same
answer both times — self-consistent, and still flips with its input. Replaced
with a deterministic canonicalisation of `f` (largest-magnitude component
positive). Orienting on `sum(f)` would be worse: it is ~0 by construction.

*Verified:* axis is sign-invariant; `redundancy` never returns the removed
branch; `pc_subspace_fraction` dominates `cos_axis_pc1²`.

---

## F. New: boundary population vs unclustered population

`border_vs_noise` crosses Block 2's per-token distance from the Fiedler
boundary against HDBSCAN's −1 labels. Both were already computed; they had
never been crossed. Rank-based AUC, so no assumption about the heavy-tailed
|v| distribution.

If noise tokens *are* boundary tokens, Phase 5c's object of study gets a
geometric definition it currently lacks. If not, that constrains any account
treating noise as merely weakly-clustered.

*Verified:* 1.0 / ~0.52 / 0.0 on aligned / random / inverted synthetic noise.

---

## G. Test status

| Suite | Result |
|---|---|
| `test_phase1b.py` (pre-existing, unmodified) | 65 passed, 0 failed |
| `test_phase1b_v2.py` (new) | 66 passed, 0 failed |
| `test_core_nulls`, `test_core_frames` | 11 / 56 passed, 0 failed |
| `test_core_metrics`, `test_core_particles` | 27 failures, **identical by name to baseline** |

The 27 are pre-existing pytest-fixture cases the manual runner does not wire
up. They were confirmed by running an unmodified copy of the source tree and
diffing failure names, not counts. **Zero regressions.**

Caveat: pytest is unavailable in this environment, so the suites were run
through a manual collector against a torch stub. Everything model-touching —
`--fast`, `--from-phase1`, `write_manifest`, ALBERT extraction — is
**unverified** and needs one real run before any of it is trusted.
