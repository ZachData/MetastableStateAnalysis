# Phase 1b (1h) — STATUS

<!-- phase-card -->
## Card

- **Question:** When Phase 1's token cloud shows k = 2 structure, is it the antipodal two-hemisphere split that would escape the paper's collapse-from-a-hemisphere lemma, or a direction of contrast inside one cone that still collapses?
- **Inputs:** Pre-revision: the 2026-04-23 GPT-2 / ALBERT Phase 1 run, gone from disk. Post-revision: a `pythia-410m` pilot of 27 checkpoints × 9 v1 prompts (battery `1e47918ef77a`), 243 runs on 2026-08-17 at code `3aeab20`, with no null (`n_null = 0`), read from Phase 1's 2026-08-12 pilot; output in the main tree's untracked results/p1b_pilot — `status-1b.md` "The 2026-08-17 Pythia pilot"
- **Results:**
  - Every layer of every run sits in one open half-space (cone collapse); no antipodal split, in both runs — `status-1b.md` "The 2026-08-17 Pythia pilot"
  - No antipodes: the antipodal rule reads 0 %. The relative classifier reads "separated" in most layers, but a single unclustered cone also reads "separated", so that reading is not yet evidence of a two-way split — `status-1b.md` "The 2026-08-17 Pythia pilot"
  - The activation-space Fiedler axis is PC1 at most layers, so downstream uses of it are using PC1 — `status-1b.md` "The 2026-08-17 Pythia pilot"
  - The bipartition's token identity persists across layers — `status-1b.md` "R4. Zero events was partly foreclosed"
  - HDBSCAN's unclustered tokens are barely the Fiedler-boundary tokens (border-vs-noise AUC near chance) — `status-1b.md` "The 2026-08-17 Pythia pilot"
- **Superseded / wrong:**
  - The pre-revision verdict table: Block 0's null and Block 3's positive were one test run twice, the ALBERT row was a path bug, and Block 1's zero events were foreclosed by construction — `status-1b.md` "Retractions and reinterpretations"
  - Cone collapse is not new: it is the anisotropy / common-direction literature's narrow cone — `lit-1b.md` §2
  - In the full d = 1024 stream the cone condition cannot fail for n ≤ d, so a full-dimension test would be free. The pilot tested the top 64 PCs, and a collapse verdict there lifts to full d, so the result holds at full d. It is informative only where n > 64: 8 of 9 prompts; `short_heterogeneous` (n = 20) runs in 19 dimensions, where collapse is near certain. Still no null — §3.39
  - `normalized_margin` is not scale-free as documented; 1c's exact margin `hull_min_norm` is the comparable one — `math-1b.md` §7.1
  - The code and the pilot's own report cite cone collapse as "Theorem 6.3"; it is Lemma 6.4 (for a decision) — `archive/UPDATE_PLAN.md` §0
- **Registry:** none, because the phase is exploratory by design; its findings feed `P-H1` and `CLAIM-A` instead (`claims/EXPERIMENTS.md`)
- **Depends on:** 1@6a6e6a3c1a
- **Feeds:** 4, 5, 5c, 6, 7
- **Open threads:**
  - Cone collapse against a null: `--n-null` has never been run, so how much is n versus d_eff is unknown — `status-1b.md` "R3. Cone-collapse is unquantified against any null"
  - The relative classifier needs a null too: its 0.90 cutoff is a reporting convention, and "separated" falls as the cloud concentrates — `status-1b.md` "The 2026-08-17 Pythia pilot"
  - Does the axis attenuate in the LN frame? Blocked: LN frames are not threaded through `run_1b` — `math-1b.md` §7.2
  - The pilot ran persistence on the legacy `regime` key, so R4's foreclosure still applies to it
  - Blocks 5 and 6 need Phase 2 OV artifacts; layer 0 (pre-LN) is still averaged into per-model means
  - `p7_motifs/design-7.md` carries "cone collapse is universal" as a constraint, from a result this file said not to cite
- **After Phase 10:**
  - Rerun the pilot with `--n-null` and `regime_key="regime_relative"` (free: reads Phase 1 dirs, about 22 min CPU)
  - Adopt `hull_min_norm` for the margin (free, code)
  - Thread LN frames through `run_1b` and test whether the axis attenuates (free, code and CPU)
  - Run on the Stage 0 sweep's 8 v1 prompts (free once Stage 0 lands; the 12 held-out prompts stay out)
- **Reviewed:** 2026-09-23 · body `6e39b06e9c`
<!-- /phase-card -->

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-23.

- 2026-08-23 · adopt 1c's `hull_min_norm`: two phases solve one margin problem two ways · `MATH_INDEX.md` "Structural fixes before the next run"
- 2026-09-16 · cone collapse is a rediscovery of the anisotropy literature · `lit-1b.md` §2
- 2026-09-16 · "Theorem 6.3" should be Lemma 6.4 (open: `math-1b.md` says "Lemma 6.4, feeding Theorem 6.3") · `archive/UPDATE_PLAN.md` §0
- 2026-09-20 · the cone condition cannot fail for n ≤ d in general position; the hemisphere lever is gated on context length · §3.39

**Registered predictions:** none. Exploratory by design; nothing in this
phase may carry an e-value (`claims/EXPERIMENTS.md`).

**Last verified (results below):** run after Phase 1,
`--phase1-dir results/2026-04-23_18-30-06`. Date not recorded in source.
**Superseded in part:** a post-revision run on the Pythia pilot exists
(2026-08-17), see "The 2026-08-17 Pythia pilot" below.

**Code state:** revised. See `archive/p1b_hemisphere/CHANGES-1b.md`. The code has moved ahead of the
results — **the verdict table below reflects the pre-revision run and has not
been reproduced.** Two of its rows are retracted outright and two are
reinterpreted. Nothing here should be cited until a rerun.

---

## The 2026-08-17 Pythia pilot (found on disk 2026-09-23)

Not recorded anywhere in the repo until the phase-review card session: the
only mention was a line in `PROJECT.md`'s disk inventory. **Input:**
`pythia-410m`, 27 pilot checkpoints × 9 v1 prompts (8 plus
`short_heterogeneous`), battery `1e47918ef77a`, 243 runs, `--from-phase1`
over the 2026-08-12 Phase 1 pilot, `frame = l2_sphere`, cone test in the top 64 PCs (`pca_n_components = 64`, per-run JSON), `n_null = 0`,
`regime_key = regime`, code `3aeab20` (after the revision, and no
`p1b_hemisphere/*.py` change since). 1309 s wall. **Output:**
results/p1b_pilot in the main tree (untracked; `phase1b_cross_run.md` is the
summary, `manifest.json` id `f3fdac3a27c3`). Populated: 976 files, every
table filled.

What it shows, read from `phase1b_cross_run.md`:

| quantity | reading |
|---|---|
| cone collapse | 100 % of layers at every checkpoint |
| strong (antipodal) bipartition | 0 % everywhere, except 1.3 % at step 120000 |
| separated (relative classifier) | 23 % (step 16) to 95 % (step 7000); dips at step 8–16 and at step 19000–60000 |
| mean normalized margin | 0.49 (step 3000) to 0.86 (step 16) |
| mean axis rotation per layer | rises with training, 0.08 rad (step 16) to 0.47 (step 120000) |
| border-vs-noise AUC | 0.51–0.60, falling with training |
| axis redundancy | `pc1`: the Fiedler axis is PC1 at most layers |
| identity persistence, HDBSCAN nesting | both True |
| cone vs null | not run (`n_null = 0`) |

Steps 0 and 1 have bitwise-identical axes (the same weights, §3.51). Step 2
differs slightly (axes by 3e-4), so it is a separate point.

**What it does not settle:** R3 (no null) and R4 (persistence ran on the
legacy `regime` key). **Nor R1.** The relative classifier calls a layer
"separated" when its separation ratio is ≤ 0.90, a cutoff
`bipartition_detect.py` calls "a reporting convention". `/challenge-pr` on
#76 ran `analyze_bipartition` on a single Gaussian shifted into a cone, with
no clusters: it read "separated" in 8 of 8 layers, and isotropic noise read 6
of 6. Across the 27 checkpoints, "separated %" correlates −0.76 with
concentration. So the 0 % antipodal figure is a structural null, as R1 said,
but "separated" may be measuring spread, not a split. It needs a null of its
own.

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

1. ~~**Nothing has been rerun.**~~ Wrong since 2026-08-17: the Pythia pilot
   above ran on the revised code, without the null (R3) or the relative
   persistence key (R4). The verdict table's GPT-2/ALBERT rows are still
   unreproduced.
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

## Visualization

`visualization/` draws this phase from its saved artifacts — no model load,
no recomputation, every threshold and verdict imported from the phase rather
than restated. `visualization/FIGURES-1b.md` is the catalogue: every figure,
what it shows, which artifact it reads, and its status.

Two things there that bear on this document. First, the figures are built
around R1 and R3's correction — the continuous, null-referenced quantity is
the figure and the regime label is an annotation beside it, so reading only
the labels is hard by construction. Second, building them surfaced four
quantities the blocks computed and the writer dropped before saving
(per-layer cone nulls and binding tokens, `persistence_length`, the
per-layer nesting and boundary tables, and the activation-space axes). All
four are now emitted. `phase1b_{stem}_axes.npz` is the one that matters for
the rerun: without it `axis_settling_step` — the quantity PREDICTIONS.md
claim (b) needs — had no input from disk at all.

Nothing in that package has been run against real Phase 1b output, because
there is none since the revision. It is exercised end to end against a
synthetic directory (`--fixture`); the shapes are real and the numbers are
invented.

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
