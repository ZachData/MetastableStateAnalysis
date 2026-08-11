# Phase 1b (1h) — STATUS

**Last verified (results below):** run after Phase 1,
`--phase1-dir results/2026-04-23_18-30-06`. Date not recorded in source.

**Code state:** revised. See `CHANGES-1b.md`. The code has moved ahead of the
results — **the verdict table below reflects the pre-revision run and has not
been reproduced.** Two of its rows are retracted outright and two are
reinterpreted. Nothing here should be cited until a rerun.

---

## Verdict table (pre-revision run)

| Block | Result as recorded | Standing after review |
|---|---|---|
| 0 — strong bipartition | **Fired (null).** 0% strong bipartition across all models. | **Reinterpreted.** Near-unreachable by construction — see R1. |
| 1 — identity persistence | Did not fire. Identity persistent = True for ALBERT and GPT. | Holds, with a caveat — see R4. |
| 2 — HDBSCAN nesting near chance | Partial. Confirmed for GPT. Inconclusive for ALBERT. | **ALBERT row retracted** — see R2. |
| 3 — cone-collapse holds everywhere | **Fired.** 100% cone-collapse, every model, every layer. Split regime never observed. | **Unquantified.** Direction sound, magnitude unestablished — see R3. |
| 5 — axis alignment | Not run (no Phase 2 OV artifacts passed in). | Unchanged. |

**Global verdict as recorded:** paper alignment = `cone_collapse` for both
families. The Phase 1 k=2 eigengap is a real, stable Fiedler axis (anisotropy
direction), not an antipodal bipartition — all tokens remain in one open
hemisphere throughout.

That conclusion is probably right and is **under-supported by the run that
produced it.** The four points below are why.

---

## Retractions and reinterpretations

### R1. "0% strong bipartition" is not "no bipartition"

`classify_regime`'s `strong_bipartition` requires a centroid angle of at
least pi/2. Under cone-collapse — which Block 3 reports at every layer of
every model — two centroids inside one open half-space essentially cannot be
pi/2 apart. **Block 0's null and Block 3's positive are close to the same
test run twice**, not two independent findings, which is how `design-1b.md`
presented them.

A cone-compatible classifier now runs alongside. On synthetic data, two
genuinely separated clusters 60 degrees apart (separation ratio 0.45,
minority fraction 0.5) read as `weak_bipartition` under the antipodal rule
and `separated` under the relative one. The 0% figure is consistent with a
real, non-antipodal partition and with no partition at all, and the original
run could not distinguish them.

**Action:** rerun and report `separated_layer_fraction` beside the strong
fraction.

### R2. The ALBERT row was a path bug, not a measurement

"Inconclusive for ALBERT" was recorded as a result. Phase 1b built its
cross-reference path as `{model}_{prompt}_d{depth}`; Phase 1 writes
`{model}_{depth}iter_{prompt}`. **No ALBERT extended run ever resolved**, so
`hdbscan_labels` never loaded and the nesting test had nothing to test.
Fixed via `p1_io.find_phase1_run_dir`.

**Action:** rerun. This row is currently empty, not inconclusive.

### R3. Cone-collapse is unquantified against any null

The verdict is a binary regime label. n points in d dimensions admit a
separating witness for free unless they positively span, and the run used
`pca_n_components=64` on prompts of order 100–200 tokens. How much of "100%,
every layer" is transformer geometry and how much is n versus d_eff was not
established.

Two matched nulls are now available (`--n-null`). On synthetic data they
discriminate — 100% of shuffled-dimension draws are cone-collapsed, 0% of
uniform-sphere draws are — so the test has power; it simply was not run.
`normalized_margin`, not the regime label, is the quantity to report.

Separately, the run's PCA reduction was documented as invariant, which is
wrong in one direction: reduced-space *collapse* verdicts lift exactly to
full d and are sound; reduced-space *split* verdicts may be artifacts. Since
no split was observed, this does not affect the recorded result.

**Action:** rerun with `--n-null`. Report N-sigma or the null cone-fraction.

### R4. Zero events was partly foreclosed

Block 1's persistence and birth/collapse/swap detection were hardcoded to the
`strong_bipartition` label, which R1 shows is unreachable under
cone-collapse. Every persistence length was therefore 0 by construction while
appearing measured. `regime_key="regime_relative"` runs the same machinery on
the reachable vocabulary.

The identity-persistence verdict itself (mean match overlap > 0.5) does not
depend on the regime label and stands.

**Also:** wiring Block 1's matcher to `cluster_tracking.match_layer_pair`
surfaced a latent hazard — exact Jaccard ties let the assignment solver
return either pairing (4 of 500 random label pairs), and anchor chaining
would propagate a flip through the remainder of a run. Not known to have
fired in the recorded run; the tie-break is now pinned.

---

## Open blockers

1. **Nothing has been rerun.** Every result above predates the revision.
2. Blocks 5 (mechanism vs OV/PCA/embedding/heads) and 6 (semantic MI) still
   require Phase 2 OV decomposition artifacts.
3. Model-touching paths are **unverified**: `--fast`, `--from-phase1`,
   `write_manifest`, ALBERT extraction. Pure-numpy blocks are covered
   (65 + 66 tests passing, zero regressions against baseline).
4. LN frames are not threaded through `run_1b`. `apply_frame` needs per-model
   LN parameters the entry point does not supply. This blocks the sharpest
   available test of the phase's own conclusion — if the k=2 axis is
   anisotropy, it should attenuate in the frame attention actually reads.
5. Layer 0 is the embedding output, pre-any-LN, and is still averaged into
   per-model means.

---

## New results available on rerun

- `separated_layer_fraction` — the cone-compatible bipartition verdict (R1).
- `normalized_margin` + null cone-fractions — Block 3 as a continuous,
  null-referenced quantity (R3).
- `border_vs_noise` AUC — whether HDBSCAN's unclustered population is the
  Fiedler boundary population. Phase 5c's object of study, from quantities
  both already computed and never crossed.
- `axis_identity` — whether the Fiedler axis is distinguishable from centered
  PC1 and the top-k PC subspace.
- `by_checkpoint` — family aggregation on a log10(step+1) axis, plus
  `axis_settling_step`: when the Fiedler axis reaches its trained direction.
  This is the quantity PREDICTIONS.md claim (b) needs and the only thing in
  the phase tracking the axis's *direction* rather than lambda_2's magnitude.

---

## Handoff notes (live constraints for later phases)

- **Phase 4:** don't treat the bipartition as a binary label — use the Fiedler
  axis as a continuous projection direction. Unchanged, and reinforced by R1.
- **Phase 5:** hemisphere centroids remain usable as candidate
  cluster-identity vectors, but are the two extremes of an elongated cone,
  not antipodal cluster centers. **New caveat:** on synthetic data the
  activation-space axis is frequently PC1 to within |cos| >= 0.9. If that
  reproduces on real runs, Phase 5 is using PC1 under a more expensive name
  and should say so. `axis_identity` reports it per layer.
- **Phase 5c:** `border_vs_noise` gives the unclustered population a candidate
  geometric definition. Worth checking before building on "unclustered" as a
  primitive.
- **Phase 6:** the Fiedler-axis difference vector and per-layer KL between
  centroid distributions remain valid probes, subject to the same PC1 caveat.
