<!-- p7d_redundancy/status-7d.md -->
# Phase 7d — STATUS

<!-- phase-card -->
## Card

- **Question:** Which of `pythia-410m`'s 384 heads carry the induction readout causally, when did each form, do they act as one redundant set or several, and how are their writes arranged in the residual stream the particles move in?
- **Inputs:** `pythia-410m`, 1 to 23 checkpoints per runner, weights-only head/MLP ablation (`ov`, which is a bias-ablation, and `mean`), repeated-random-token probe (`wide`), 8–16 sequences; no battery prompts. The FV run adds four word-pair tasks. Outputs `data/analysis/*.json`, git-ignored; each section names its file, and the `mean` reruns (2026-09-11/12) are `*_pythia-410m_mean*.json`. Rerun lines: `p7d_redundancy/status-7d.md` "Reproducing"
- **Results:**
  - The set is small and heavy-tailed: about 4 substantial members and 10 with any effect, of 384, spread over layers 5–15 — `p7d_redundancy/status-7d.md` "Q1 — membership"
  - The members did not form together; the window is (512, 4000] — `p7d_redundancy/status-7d.md` "Q2 — **no, they did not form together.**"
  - One redundant set, not several; most of the pairwise interaction is the product of the two solo effects — `p7d_redundancy/status-7d.md` "Pass 2"
  - Born aligned, then one head (`L11H14`) leaves a locked core; redundancy survives the separation — `p7d_redundancy/status-7d.md` "The geometry"
  - Division of labour: one member does induction, four carry function vectors, none does both — `p7d_redundancy/status-7d.md` "The FV-head experiment"
  - `L5H2` is a previous-token relay feeding `L7H8`'s matching attention — `p7d_redundancy/status-7d.md` "The upstream-relay check"
  - The exhaustive self-repair sweep: MLP 6 is the largest stand-in, and the attention search had missed the four largest — `p7d_redundancy/status-7d.md` "Tying up the self-repair"
  - `L5H2` and MLP 6 are an OR-gate; MLP 6's repair is an active rotation aimed at the set's key read-space, and it moves the heads it points at — `p7d_redundancy/status-7d.md` "Opening MLP 6", "Geometry predicts function"
  - The catalogue, `r*`, ambient energy, pairwise matrix, formation curves and geometry are mode-invariant at 410m: `mean` reproduces the `ov` numbers. The FV experiment, the self-repair sweep and the MLP 6 work were not rerun, and `mean` cannot see MLP 6's mechanism (§3.26) — §3.15
- **Superseded / wrong:**
  - "The stand-ins are those three members" was superseded by the exhaustive sweep — `p7d_redundancy/status-7d.md` "Tying up the self-repair"
  - `ov` "zero-ablation" leaves the value bias, so it is a bias-ablation; immaterial, a hook zero reproduces it bitwise — §3.15
  - `mean` is not the conservative control when the mechanism is itself the mean (MLP 6); found at 70m — §3.26
  - The `wide` probe's token range is out of distribution; `freq` is the better probe, added as an arm — §3.15, §3.17
- **Registry:** none, because every measurement is on `pythia-410m`, which the rung policy (`check_registry` rule 3 applied forward) keeps for exploration only (`p8_scale_ladder/design-8.md`); `claims/EXPERIMENTS.md` lists 7d as a live instrument with no prediction
- **Depends on:** none
- **Feeds:** 7e, 8, 9, 10
- **Open threads:**
  - Q4 (structure per member) and Q5 (classes), mostly on disk and unread — `p7d_redundancy/status-7d.md` "What is open"
  - The step-1000 circuit (`L5H2` + `L11H14`, no `L7H8`); its pilot's sub-additive sign is outside the readout's range — `p7d_redundancy/status-7d.md` "The pilot that must not be over-read"
  - Why `L11H14` is the strongest output-side stand-in for `L5H2`; per-member attribution of MLP 6's rotation is unrun — `p7d_redundancy/status-7d.md` "Decoding the repair direction"
  - `L6H0`, not a member, falls harder than `L7H8` when `L5H2` is ablated: a second downstream matcher, unfollowed — `p7d_redundancy/status-7d.md` "Self-repair (2026-09-13)"
- **After Phase 10:**
  - Q4 from the sweeps already on disk (`qk_symmetry_sweep`, `ov_per_head_series`, `copying_score_sweep`, `behavioural_series`) (free)
  - `L11H14`'s step-1000 copying score (free: weights only)
  - The step-1000 pair on the graded readout, KL or λ, instead of raw dNLL (forward pass: one checkpoint)
  - Per-member attribution of MLP 6's rotation to `L7H1` / `L8H6` / `L8H9` (forward pass: two checkpoints)
- **Reviewed:** 2026-09-24 · body `751b162e22`
<!-- /phase-card -->

**Registered predictions:** none. Every number in this phase is a measurement
on pythia-410m, an artifact that is spent under `check_registry` rule 3 —
nothing here can be registered after the fact, and nothing in it may carry an
e-value (`claims/EXPERIMENTS.md`).

**Last verified:** 2026-09-13.
**Overall:** Q1, Q2 and Q3 are answered on pythia-410m, and **pass 2 (the
pairwise interaction matrix) and the geometric axis are now answered too**. Q4
is mostly on disk and unread; Q5 has a partial answer from geometry alone. The
`L5H2` puzzle (§3.12-U/§3.16/§3.18) is now closed on both the mechanism ("The
upstream-relay check") and the super-additivity ("Self-repair", then "Tying up
the self-repair" for the exhaustive version, which corrects it). **"Opening
MLP 6" is the current front**: `L5H2` and MLP 6 are an OR-gate over `L7H8`'s
matching, the repair is *active* — only the direction MLP 6 moves to when the
relay is ablated restores the matcher, not the one it already had — and that
direction is aimed at the redundancy set's shared key read-space, moving
precisely the heads it points at ("Decoding the repair direction", "Geometry
predicts function").
Nothing is registered and nothing can be — every measurement here is on an
artifact spent under `check_registry` rule 3. Restore checks are exact
(`0.0e+00`) on every run reported below.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`). Backfilled 2026-09-24.

- 2026-09-11 · every `ov` ablation here is a bias-ablation (the value bias stays); immaterial, a hook zero reproduces it bitwise · §3.15
- 2026-09-11 · the `wide` probe's random ids are out of distribution; the numbers here are all on `wide`, and `freq` is the better arm · §3.15
- 2026-09-11 · 410m is mode-invariant: `mean` reruns reproduce the `ov` catalogue, `r*` and interaction ratio · `p8_scale_ladder/status-8.md` "The matched cross-rung read"
- 2026-09-12 · a ceiling-contaminated cell should be retried on `freq` before it is called unmeasurable · §3.17
- 2026-09-13 · `mean` is blind to a mechanism carried by its own mean (MLP 6), so it is not always the conservative control · §3.26

## What is answered

### Q1 — membership. **~4 substantial members, ~10 with any effect, of 384.**

`redundancy_catalog.py`, single-head OV ablation at step 16000, 8 sequences, all
384 heads. The distribution is brutally heavy-tailed — the **median head moves
the readout by 0.001** — so the set is a real, small, identifiable object rather
than a gradient.

| | |
|---|---|
| median | +0.00107 |
| p99 | +0.1935 |
| above +0.05 / +0.2 / +1.0 | 10 / 4 / 2 heads |
| max / min | +1.9662 (`L5H2`) / −0.0522 (`L10H7`) |

`L5H2` +1.966, `L7H8` +1.019, **`L12H5` +0.420**, `L8H6` +0.212, `L11H14`
+0.190, `L8H9` +0.131, `L15H14` +0.087, `L7H1` +0.067, `L9H13` +0.064, `L10H9`
+0.050. **`L12H5` and `L8H6` were entirely unknown** before this sweep, and the
38-head sample of §3.12-R had missed both — which is why the full sweep was run
instead of a proxy screen.

Members are spread across layers 5, 7, 8, 8, 11, 12, 15 — **not** clustered in
the 9–20 band where the token-identity copiers live. Membership and copying are
different properties.

### Q2 — **no, they did not form together.** Q3 — **yes for `L5H2`, no for `L7H8`.**

`member_formation_curves.py`, 23 checkpoints, 16 sequences, six members plus the
joint `L5H2`+`L7H8` arm and the residual-delta cosine. `L7H8` reproduces
§3.11-A throughout and step 16000 reproduces §3.12-S to four decimals.

| member | forms in | peak | at step | at 143000 | retained |
|---|---|---|---|---|---|
| `L5H2` | **(512, 1000]** | +8.43 | 2000 | +1.26 | 15 % |
| `L11H14` | **(512, 1000]** | +3.57 | 1000 | +0.17 | 5 % |
| `L12H5` | (1000, 2000] | +0.99 | 3000 | +0.11 | 11 % |
| `L8H9` | (1000, 2000] | +0.60 | 3000 | −0.01 | −1 % |
| `L8H6` | (512, 1000] | +0.70 | 3000 | +0.15 | 22 % |
| `L7H8` | **(2000, 3000]** | +1.54 | 54000 | +1.22 | **79 %** |

`L5H2` appears in the **same interval the model acquires induction at all**
(second-copy NLL 12.63 → 4.91), and Pythia publishes no checkpoint between 512
and 1000, so that is the finest interval this axis can resolve. `L7H8` — the
head §3.11–§3.12 was largely spent on — is the **last** member to arrive, and
the only one of the six that rises monotonically.

**Read the ordering and the clustering together, not one instead of the other.**
Five of the six form inside `(512, 2000]` — one doubling of training on a
143,000-step axis, with the remaining 140,000 steps recruiting nobody, and the
window is exactly where the model acquires induction. So the members are
*ordered* at the resolution of the grid and *simultaneous* at the scale of
training. Whether that shared window is a common trigger or a cascade is not
decidable on one model; §3.14.4-A gives the pythia-70m design that decides it,
and it is the one 7d question that could carry a registered prediction.

**The redundancy postdates both heads.** The interaction is ≈0 while `L5H2` is
at its maximum (−0.03 at 1000, +0.02 at 2000), then +0.39 (3000), +1.72 (4000),
+4.15 (16000), +4.17 (32000). Dating the set by dating its members would have
been wrong by 2000 steps. The alignment behaves the same way and is not a
converged endpoint but `L7H8`'s **entry condition**: −0.17 → +0.01 → +0.26 →
**+0.83 at step 4000** → +0.91, flat after.

Full table and caveats: `PROJECT.md` §3.12-U.

### Pass 2 — **one set, not several.** 45 cells, and independence is dead.

`pairwise_interaction_matrix.py`, top 10 members, 16 sequences, steps 16000 and
143000, restore exact, no cell within 2 nats of the ceiling. The
`L5H2`×`L7H8` cell reproduces §3.12-S at **+4.1505** against its +4.151.

| | step 16000 | step 143000 |
|---|---|---|
| positive cells | **44/45** | 38/45 |
| super-additive (> +0.02) | 38 | 28 |
| r²(interaction, `d_a·d_b`) | 0.739 | 0.807 |
| r²(interaction, δ-cosine) | 0.094 | 0.073 |

No block structure at either step, so it is **one redundancy set**, not several
disjoint ones. But **74–81 % of the interaction is explained by the product of
the two heads' own effects** — most of the matrix's apparent structure is
magnitude, not pairing. A tempting "opposite directions are more redundant"
pattern shows in the tails and **does not survive the full sample**
(ρ = −0.167, p = 0.27); the honest statement is that direction and
substitutability are **decoupled**, which is §3.12-S's single-pair dissociation
generalised to 45 pairs.

### The geometry — **born aligned, then decoherent, while redundancy is kept.**

`member_subspace_geometry.py`, six members + **three near-median control heads
carrying a measured null** (`L5H5`, `L8H3`, `L12H15`), 13 checkpoints.

- **Alignment is present at birth.** At step 1000 the only extant pair,
  `L5H2`×`L11H14`, is at rank-1 cosine **0.857** and centered CKA **0.693
  against a null of 0.128**. There is no private-subspace phase.
- **Then the set fans out.** Fixed-15-pair mean cosine peaks at step 5000
  (+0.744) and falls to **+0.327** at 143000 — 56 % of the peak — while mean
  delta norm *grows* 671 → 1093. Not a fading-signal artifact. The max pair
  stays pinned at 0.87–0.97 throughout while the min pair goes +0.32 → −0.19:
  **a locked core with heads peeling off, not a uniform drift.**
- **Redundancy survives the separation.** `L5H2`×`L11H14` holds interaction
  +2.18 (16000) and +1.53 (143000) while its cosine goes 0.344 → **0.004**.
  Redundancy here is routed through downstream computation, not through writing
  the same direction.
- **No member's subspace expands.** Participation ratio sits in 8–28 and ends
  lower than it starts. Both halves of "did they start private and expand out"
  are false.

**Re-run under `mean` across the same axis (2026-09-12) — mode-invariant, and
the fan-out decomposes.** The fixed-15-pair statistic reproduces: peak
**+0.753** against `ov`'s +0.744, endpoint **+0.327** against +0.327, 57 %
against 56 %. But split on `L11H14` — which the four measurements below
singled out *before* this trajectory existed, so this is not a post-hoc cut —
and the single number turns out to average two opposite behaviours:

| step | core (5 heads, no `L11H14`) | `L11H14`'s pairs |
|---|---|---|
| 1000 | — (n=0) | **+0.888** (n=1, the birth pair) |
| 4000 | **+0.903** (n=10) | +0.303 (n=5) |
| 16000 | +0.814 (n=10) | +0.136 (n=5) |
| 143000 | **+0.751** (n=3) | **−0.099** (n=3) |

**The set does not fan out; one head leaves a locked core.** At 143000 all five
`L11H14` pairs are the bottom five of fifteen and every other pair is ≥ +0.201.
Formed-only throughout, `n` printed, per the hazard section below.

**And the two instruments disagree about that head.** `L11H14`'s rank-1
mean-delta cosine inverts to **−0.099** while its centered CKA ends at
**+0.317 against a measured null of +0.179** — still well above chance. Its
*mean write direction* anti-aligns; its *effect subspace* keeps real overlap.
That fits its participation ratio of ~60 against everyone else's 8–28: for this
head specifically a rank-1 summary is the wrong instrument. **Write it as
"anti-aligned in mean direction", never as "orthogonal to the set".**

### `L11H14` is singled out by four independent measurements

Lowest mean cosine to the set at 143000 (**−0.033**) at the **third-largest**
delta norm (10.60) — against +0.389 for `L7H1` at a magnitude-matched 9.45, and
+0.43 to +0.48 for the rest. Effect subspace **3–4× higher-dimensional** than
any other member (PR ~50 vs 8–28). **Earliest defector**, breaking away between
steps 2000 and 4000 while the rest stay locked to 9000. And it is already known
as the step-1000 co-mechanism and the top copier at 143000.

### The ambient stream is ~20-dimensional, and that broke two measures

The baseline residual's own participation ratio is **20.3 of 1024** at step
16000, falling to **8.4 at 143000** (on this probe — copied positions of
repeated *random-token* sequences, a deliberately narrow distribution; natural
text would be far higher). Members put 18 % of their effect energy in the
ambient top-10 and 59 % in the top-50.

This is why `coverage_by_others` **saturates to 1.000 and is unusable**, and why
the isotropic `k/d_model` chance value is not a baseline: unweighted principal
angles count 150–300 directions of which only ~20 carry ambient variance.
**Quote `cka` / `cka_centered`, never the unweighted `subspace_*_cos` alone.**

### The hazard that caught this session three times

**Every set-level mean over "the members" silently changes its own membership as
heads form, and it always manufactures a rising trend.** It bit `union_ratio`
(0.94 → 0.16, almost entirely arithmetic), the set-level mean cosine ("peaks at
step 1000" — an average of one pair), and centered CKA ("0.197 → 0.637", where
the 0.197 was one real pair averaged with fourteen non-existent ones). Any
trajectory here must be read over a **fixed** pair set, or over a formed-only
set with `n` printed beside it. The runner now reports `union_formed` and
`formed_members` for this reason; §3.13's report-both rule is not sufficient
protection on its own.

### The FV-head experiment (2026-09-12) — **the set divides labour; it does not transition**

`fv_score.py`, six members + six controls, four Todd-et-al. word-pair tasks
(10-shot × 16 prompts each), six checkpoints, induction and FV scored in the
same process off the same weights. Run because `PROJECT.md` §3.16 read
`2502.14010` (Yin & Steinhardt: induction heads *become* function-vector heads)
as a candidate resolution of §3.12-U's `L5H2` puzzle.

**The hypothesis is refuted for the head it was proposed about.** `L5H2`'s FV
score never leaves zero and ends **negative**:

| step | 1000 | 2000 | 4000 | 8000 | 16000 | 143000 |
|---|---|---|---|---|---|---|
| `L5H2` FV | −0.0000 | −0.0006 | −0.0004 | +0.0008 | −0.0019 | **−0.0018** |
| `L7H8` FV | −0.0000 | −0.0000 | −0.0000 | −0.0001 | +0.0001 | **+0.0001** |
| `L8H9` FV | +0.0000 | +0.0004 | +0.0060 | +0.0112 | +0.0093 | **+0.0152** |
| `L11H14` FV | +0.0010 | +0.0013 | +0.0054 | **+0.0085** | +0.0020 | +0.0012 |
| `L8H6` FV | +0.0001 | +0.0006 | +0.0037 | +0.0061 | +0.0030 | +0.0041 |
| `L12H5` FV | −0.0000 | −0.0000 | +0.0015 | +0.0054 | +0.0058 | +0.0018 |
| control mean | +0.0000 | −0.0000 | −0.0003 | −0.0001 | −0.0005 | −0.0017 |
| max abs control | 0.0000 | 0.0001 | 0.0020 | **0.0009** | 0.0035 | 0.0103 |

At 143000 `L5H2` sits *below* four of the six controls. Per task it is
sign-inconsistent — +0.0103 on country-capital against −0.0070 on past-tense at
step 8000 — which is what noise looks like, not a function vector. **§3.12-U's
puzzle stands as a puzzle.**

**What replaced it is better.** The set contains real FV heads and they are a
*different subset*. At step 8000 the **top four FV heads of the twelve scored
are all members**, and the fourth of them (+0.0054) beats the best control
(+0.0003) by 18×. `L8H9` rises monotonically to **+0.0152**, the largest score
at the endpoint, at ~8 standard errors (per-prompt sd 0.0147, n = 64).

**And `L7H8` is the dissociation.** It is the model's induction head by a
distance — induction score **0.021 → 0.947**, textbook — and its FV score is
pinned at ±0.0001 at **every** checkpoint. Meanwhile the four FV-positive
members never exceed an induction score of 0.015, which is inside the control
band (mean 0.008–0.013). So within one causally-defined redundancy set:

> **no member does both jobs.** One head carries induction and no
> function-vector role; four carry a function-vector role and no induction;
> `L5H2` carries neither, while having the largest causal effect on the
> induction readout of any member.

Yin & Steinhardt report heads *transitioning* between the two roles over
training. This set shows a **division of labour** instead — which is a
different structure, and it is only visible because membership here is defined
causally rather than by either score.

**The instrument validates itself on the task axis.** The FV effect appears
exactly where the model can do the task: per-task CIE tracks the per-task ICL
gap (past-tense and plural, gap +0.51/+0.56 at step 8000, carry the whole
effect; country-capital never gets a gap above +0.08 and shows almost nothing).
A head cannot supply a function vector for a function the model has not
learned.

**Three limits, none of them hidden.**

1. **The null grows and the separation narrows at the endpoint.** Max abs
   control goes 0.0000 → **0.0103** (that is `L14H7`, a large *negative*
   outlier). At step 8000 `L8H9` beats the best control 12×; at 143000 only
   1.5×. The result is cleanest **mid-training**, and the endpoint column
   should not be quoted alone.
2. **The FV rise is confounded with task acquisition** — the ICL gap rises over
   the same interval. What breaks the confound is that the **controls stay at
   zero while the gap grows**, so task acquisition alone does not manufacture a
   member-vs-control gap. That argument holds cleanly through step 16000 and
   weakens at 143000, where the controls spread.
3. **Six controls is a small null**, and per §3.13 both views are reported:
   `fv_score_median` agrees with the mean in sign and ordering at every
   checkpoint and is uniformly smaller (`L8H9` +0.0091 against +0.0152 at
   143000), so nothing here is a mean artifact.

**One instrument note, not a contradiction.** This runner scores induction on
the **repeated-random-token** probe and reads `L5H2` at 0.0000 at every step;
§3.12-U's twenty-fold fall (0.0046 → 0.0002) is `behavioural_series.json`'s
**natural-text** probe. Different probes, both saying `L5H2` is a negligible
attention-pattern induction head. This probe has the dynamic range to tell —
it reads `L7H8` at 0.947 on the same batch.

### The upstream-relay check (2026-09-13) — **`L5H2`'s puzzle, closed on the mechanism**

`p7d_redundancy/upstream_relay_check.py`, new this session. §3.16/§3.18 filed a
puzzle: `L5H2` has the largest single-head causal effect on the readout of any
member (+1.97 at step 16000) yet scores near zero on both instruments ever
pointed at it — the QK induction-attention score and the FV score. Both of
those measure `L5H2`'s **own** behaviour; neither can see a causal role that
runs through a downstream head. Two facts already on disk point exactly there
and had never been connected: Stage 0 (2026-09-07) found `L5H2` is a
previous-token head on its own attention (offset −1 = 0.895, ~49x the
384-head median); `induction_composition_whitening.py`'s H1-REVISED
(2026-09-09) found `L5H2`'s OV composes into `L7H8`'s Q/K read-space at rank
0 of 112 (z ≈ +6), onset between step 512 and 1000 — `L5H2`'s own formation
window. Neither result had been read as an answer to the puzzle before now.

**The functional test.** Ablate `L5H2`'s OV (weights-only, restore exact) and
read `L7H8`'s **own** induction-attention score, at 6 checkpoints, against 4
generic controls per step drawn outside the catalogue's top 6
(`member_formation_curves.members`'s own pool convention):

| step | `L7H8` baseline | after `L5H2` ablated | delta | max \|control delta\| |
|---|---|---|---|---|
| 1000 | 0.022 | 0.017 | −0.0052 | 0.0001 |
| 2000 | 0.378 | 0.165 | **−0.2128** | 0.0032 |
| 4000 | 0.919 | 0.545 | **−0.3742** | 0.0029 |
| 8000 | 0.928 | 0.719 | −0.2089 | 0.0011 |
| 16000 | 0.934 | 0.744 | −0.1901 | 0.0006 |
| 143000 | 0.946 | 0.784 | −0.1614 | 0.0013 |

At every checkpoint, ablating `L5H2` moves `L7H8`'s own attention pattern
60–600x more than any of the 4 generic controls that step. **`L5H2` does not
merely correlate with `L7H8`'s formation — removing it demonstrably breaks
part of `L7H8`'s own matching mechanism**, from the pair's shared formation
window through the trained endpoint.

**The sharper control.** A magnitude-matched head is a stronger null than a
near-zero one: `L12H5`, the model's **third-largest** single-head causal
effect (+0.42), moves `L7H8`'s induction score by **exactly 0.0000** at steps
4000, 16000 and 143000. Mattering a lot for the readout is not sufficient to
disturb `L7H8`'s attention; only `L5H2` does. The disruption is specific to
this pair, not a "removing something big" artifact.

**What this settles, and what it does not.**

- **Settled: the mechanism half.** `L5H2` has no induction score because it is
  not attending to the induction position — it is a genuine previous-token
  head — and no FV score because that is not its job either. Its causal
  weight comes from feeding `L7H8`'s own matching attention through the
  composition H1-REVISED already found as a weights-only quantity. That
  quantity is now shown to be **functional**, not just structural.
- **Not settled: §3.12-S's super-additivity.** A dependency this direct
  predicts *sub*-additive joint ablation (remove the input, the matcher has
  less to match on); S1 found the opposite — joint ablation costs 2.2x the
  sum of the parts. The natural reconciliation is that the NLL readout has
  network-wide self-repair available (§3.16's Hydra-effect citation, already
  invoked for 44/45 pairwise cells) that this attention-level probe cannot
  see: something elsewhere in the 384 heads may partly restore `L7H8`'s
  *effect on the loss* when `L5H2` alone is gone, without restoring `L7H8`'s
  *own attention pattern* — which is exactly why an attention-level probe,
  not an NLL one, is what could still see the dependency. **This is a
  hypothesis, not a measurement.** The next test, if this thread is picked up
  again: search the other ~380 heads' own induction-attention delta under
  `L5H2`-alone ablation for whichever one moves to partly cover the NLL gap.
- Step 1000 is floor-limited (`L7H8`'s own baseline there is 0.022, §3.14.4-D)
  — read the −0.0052 as a direction, not a magnitude.
- Exploratory; no p-value; `claims/registry.json` unchanged; pythia-410m spent
  under `check_registry` rule 3.

`data/analysis/upstream_relay_check.json` (git-ignored, ~3 min for the 6-step
grid): `python -u p7d_redundancy/upstream_relay_check.py --steps
1000,2000,4000,8000,16000,143000`.

### Self-repair (2026-09-13) — three redundancy-set members reproduce `L7H8`'s super-additive signature with `L5H2`

`l5h2_backup_search.py` (attention search, all 384 heads) then
`l5h2_backup_causal_check.py` (the ΔNLL-interaction version, on the search's
own candidates). Closes the half the upstream-relay check left open: §3.12-S's
`L5H2`×`L7H8` joint ablation is super-additive (2.2x the sum of parts), which
a direct dependency should NOT produce, and §3.16 named Hydra-effect
self-repair as the untested reconciliation.

**The search.** All 384 heads' own induction-attention, baseline vs.
`L5H2`-ablated, at 6 checkpoints (~13 s/checkpoint at 8 seqs — `induction_scores`
already batches every head into one attention-output pass). Two findings, one
of them not what the search was built for: `L6H0` — not a catalogue member —
falls **harder than `L7H8`** at every checkpoint from 2000 on (−0.485 at
143000 vs `L7H8`'s −0.161), so `L5H2` feeds more than one downstream matcher.
And a consistent riser cluster from step 1000 on: `L10H7`, `L10H15`, and
three known members — `L11H14`, `L8H6`, `L8H9` — all show their OWN
induction-attention rise when `L5H2` is ablated.

**The causal version**, on exactly those candidates (fixed before running,
not chosen after):

| step | `L11H14` | `L8H6` | `L8H9` | `L10H15` | `L10H7` | `L7H8` (control) |
|---|---|---|---|---|---|---|
| 4000 | **+1.578** | +0.750 | +0.500 | +0.220 | −0.011 | +1.717 |
| 16000 | **+2.180** | +1.613 | +0.543 | +0.154 | −0.220 | +4.151 |
| 143000 | **+1.532** | +0.302 | +0.083 | +0.002 | −0.101 | +3.430 |

`L7H8`'s interaction reproduces §3.12-S exactly at step 16000 (+4.1505 here
vs +4.151), which calibrates the method. `L11H14`, `L8H6` and `L8H9` are
super-additive with `L5H2` at **every** checkpoint — `L11H14`'s interaction is
8–11x its own solo effect, a bigger relative jump than `L7H8`'s own. **The
super-additivity is a property of the set, not a private feature of one
pair**: several members stand in for `L5H2` at once, and the original pairwise
arm was only seeing one of them.

**`L10H15` decays like the set's other members** (positive at 4000/16000,
+0.002 by 143000 — the same shape §3.12-U found for four of the six original
members). **`L10H7` dissociates**: its attention rises at every checkpoint but
its causal interaction is **negative** at every checkpoint (−0.01 to −0.22) —
sub-additive, the opposite of self-repair. A behavioural (attention) proxy
failing to predict a causal quantity is §3.12-R/G6's lesson one level up.

**Not settled**: whether these three are the *only* stand-ins (MLPs untested;
the search only read attention, not every head's own marginal ΔNLL), or why
`L11H14` — already the set's oddest member on four other axes (§3.19, §7e) —
is also its strongest stand-in for `L5H2` specifically.

`data/analysis/l5h2_backup_search.json`,
`data/analysis/l5h2_backup_causal_check.json` (both git-ignored). Rerun:
`python -u p7d_redundancy/l5h2_backup_search.py --steps
1000,2000,4000,8000,16000,143000` (~1.5 min), then
`python -u p7d_redundancy/l5h2_backup_causal_check.py --steps 4000,16000,143000`
(~3.5 min).

> **Superseded in part, 2026-09-13, by "Tying up the self-repair" below.** The
> set-wide conclusion stands; the claim that the stand-ins *are* those three
> members does not.

### Tying up the self-repair (2026-09-13) — the exhaustive version, and it corrects the section above

Four runners: `backup_sweep_full.py`, `prev_token_profile.py`,
`relay_selection_check.py`, `mlp_backup_check.py`. Full synthesis in
`PROJECT.md` §3.22; this is the phase-local detail.

**The sweep, and the recall failure it exposes.** Every head's solo, joint,
marginal and interaction against `L5H2` at step 16000, 8 seqs, `ov`, both arms
in one process. Validations: the **solo column reproduces
`redundancy_catalog.json` exactly** (max |diff| 0.0e+00 over 383 heads),
restore exact, and **0 of 383 arms ceiling-contaminated**. Median interaction
+0.0018; **44 heads above +0.1**; 15 below −0.1.

| rank | head | interaction | solo | in the attention search's list? |
|---|---|---|---|---|
| 1 | `L7H8` | +4.068 | +1.019 | — |
| 2 | `L5H9` | **+3.356** | +0.035 | **no** |
| 3 | `L9H5` | **+2.745** | +0.030 | **no** |
| 4 | `L1H15` | **+2.351** | +0.040 | **no** |
| 5 | `L11H14` | +1.944 | +0.190 | yes |
| 6 | `L8H6` | +1.464 | +0.212 | yes |
| 8 | `L4H9` | +1.018 | **−0.001** | no |
| 15 | `L8H9` | +0.523 | +0.131 | yes |
| 377 | `L10H7` | **−0.183** | −0.053 | yes (as a top riser) |

The attention search missed the four largest and put `L10H7` — which it ranked
as a leading candidate — at 377 of 383 with a *negative* interaction. The
cause is structural, not sloppy: it searched for heads whose own
induction-attention rose, and a backup that is not an induction head has no
such score. `L4H9` makes the point sharply: solo effect **−0.001** (invisible
on the clean model) and interaction **+1.018**.

**The MLPs, never previously tested here, contain the largest stand-in.**
MLP 6's interaction with `L5H2` is **+6.26** at step 16000 — above `L7H8`'s
+4.07 and above every head — on a solo effect of +0.14. Robust across modes
(`mean` +6.26 / `zero` +5.25), probes (`freq` +5.67 at 3.04 nats headroom),
and checkpoints (4000 +1.56, 16000 +6.26, 143000 +5.14), argmax of all 24
layers every time; MLP 5 second (+1.43/+1.72); layers 12+ under 0.25.
**Specific to `L5H2`**: MLP 6 × `L7H8` = +0.125, MLP 6 × `L12H5` = +0.127, and
with those sources no MLP exceeds +0.25. MLP 0 is excluded everywhere (solo
+12.5 puts the joint arm past `ln V`; its negative interaction is §3.12-M5's
artifact, which is why the runner flags `headroom` per arm). §3.12-Q6's
weights-only "no elevated MLP pathway" was answering a different question.

**Two classes of stand-in, separated by position.** `--background L5H2` (new
flag on `upstream_relay_check.py`) measures each candidate's effect on `L7H8`'s
attention *given the relay is already gone* — the question the unconditional
arm cannot pose, since on the clean model a backup is redundant. Conditional
null: 6 generic controls move `L7H8` by ≤ 0.0038.

| stand-in | layer | Δ `L7H8` attention, conditional | interaction |
|---|---|---|---|
| `L5H9` | 5 | −0.0870 (23x null) | +3.356 |
| `L1H15` | 1 | −0.0671 (18x null) | +2.351 |
| `L4H9` | 4 | −0.0095 | +1.018 |
| `L9H5` | 9 | +0.0000 | +2.745 |
| `L11H14` | 11 | +0.0000 | +1.944 |

Upstream stand-ins partly restore the matching pathway (`L5H9`'s conditional
effect is **3.4x its unconditional** one); the two downstream of layer 7 move
`L7H8`'s attention by exactly zero while carrying interactions of +2.7 and
+1.9, so they compensate at the readout instead. **`L11H14` is an output-side
compensator** — which reframes "why is it the strongest" rather than answering
it.

**Prev-token capacity is a threshold property, and a fresh §3.13 case.** The
three members §3.21 named rank **377th, 380th, 372nd of 384** on prev-token
attention, below the median — so re-supplying the signal is not what they do.
But across all 383 heads the rank correlation with interaction is **zero**
(Spearman −0.016, p = 0.75) while the **12 heads above prev-token 0.3 have
median interaction +0.444 against the other 371's +0.0007** (Mann-Whitney
p < 1e-5). Pearson (+0.322) is the misleading middle. Below the threshold it
predicts nothing; above it, a great deal.

**What selects the relay is composition.** 13 heads carry prev-token
attention above 0.3, so that cannot be what makes one head the relay.
Per-head composition into `L7H8`'s read-space (H1-REVISED stored only the
population summary) paired with an ablation: `L5H2` rank **0/112 at z +5.25**
→ Δattention **−0.190**; `L5H9` rank 1 → −0.025; `L6H0` rank 2 (prev-token
0.005) → −0.029; `L4H9` rank **23** (prev-token 0.805) → **+0.0008**;
`L3H1` rank 24 (0.710) → **+0.0028**. Prev-token attention without
composition is inside the control band. `L5H2` is the joint extreme at 7x the
next largest — conjunctive and strongly super-linear, no functional form
claimed. *Report-both:* Pearson(prev-token, composition) +0.42 vs Spearman
**+0.12 (p = 0.23)**, so treat the axes as near-independent.

**Already on disk, unread.** `L5H2`×`L11H14` = +2.1800 at step 16000 is in
`pairwise_interaction_matrix.json` (2026-09-10) — §3.21 reproduced rather than
discovered it, and the matrix's whole `L5H2` row was answerable without a
forward pass. Reading it against its own regression does pay: §3.12-V's
magnitude rule (r² 0.74 / 0.81, both reproduced) has **one systematic
exception — `L5H2`×`L11H14`, the largest positive residual of all 45 cells at
both checkpoints** (+1.37, +0.97), with δ-cosine **0.004** at 143000.

`data/analysis/backup_sweep_full.json` (~56 min, 383 heads x 2 arms),
`prev_token_profile.json`, `relay_selection_check.json` (weights only),
`mlp_backup_check.json` — all git-ignored. Rerun lines are each runner's
`--help`; defaults reproduce the numbers above.

### Opening MLP 6 (2026-09-13) — an OR-gate, and the repair is ACTIVE

`mlp_relay_role.py`, `mlp6_content_vs_scale.py`, `mlp6_response.py`. Synthesis
in `PROJECT.md` §3.23.

**Position first, and it is a config fact.** `use_parallel_residual = True`, so
attention and MLP at each layer read the same residual in parallel: **MLP 5
cannot see `L5H2`'s output**, and **MLP 6 is the first sublayer that can**,
writing into the residual `L7H8` reads. Two exact validity checks came free:
MLPs **7–23 move `L7H8`'s attention by exactly 0.0000**, and MLPs **0–5 are
unchanged by the ablation to the last digit** (cos 1.0000, per-position change
0.0000).

**The OR-gate.** `L7H8` induction attention, step 16000 / 143000: clean
0.938 / 0.947; `L5H2` ablated 0.751 / 0.796; MLP 6 ablated 0.876 / —; **both
0.045 / 0.012**. Either supplier alone suffices; removing both blinds the
matcher. That is what §3.22's +6.26 ΔNLL interaction looks like upstream.

**Not scale** — three ways. `zero` and `mean` leave the residual entering
layer 7 at 43.07 vs 43.46 (42.83 vs 42.91 at 143000) yet give attention 0.045
vs 0.643; a **norm-matched random constant** reproduces `zero` (0.030–0.050);
and MLP 5 removes as much norm as MLP 6 for a fraction of the damage. §3.15
puts the burden on `zero` and this is how it is discharged.

**A specific direction, carried by the mean.** Constant share is only
0.25–0.28, yet μ alone retains 0.643 / 0.567. `mean` minus `random`: **+0.60**
(MLP 6), +0.12 (MLP 5), −0.02 (MLP 3).

**Active, not pre-existing — the distinction §3.16 imported but never tested.**
MLP 6's mean rotates to cos **0.837** and grows **21 %** (the only MLP that
grows), with its per-position variation changing least of MLPs 6–23 (0.19).
Causally:

| constant in MLP 6's slot, `L5H2` ablated | 16000 | 143000 |
|---|---|---|
| zero | 0.045 | 0.012 |
| norm-matched random | 0.030–0.050 | 0.011–0.014 |
| **μ from the CLEAN state** | **0.038** | **0.016** |
| **μ after responding** | **0.643** | **0.567** |

**The direction MLP 6 already had is worth no more than noise; only the one it
moves to works.** A ~33° rotation is the whole difference between a blind
matcher and a working one.

**And it is not writing where `L5H2` wrote**: `cos(μ_cond, d_L5H2)` = −0.507,
largest magnitude of any MLP and negative (the unchanged MLPs 0–5 sit at
0.01–0.14). So this is an enabling/operating-point role, not a re-supply of
prev-token content — which a position-independent constant could not carry
anyway.

*Caveats:* all `L5H2` ablations are `ov` (a bias-ablation, §3.15), consistent
throughout; `mean` arms recompute μ inside the conditional state; 3 random
directions agreeing to ~0.02.

### Decoding the repair direction (2026-09-13) — it addresses the SET, not the matcher

`mlp6_decode_direction.py`. Synthesis in `PROJECT.md` §3.24. The object is the
**rotation component** (μ_cond minus its projection on μ_clean), which carries
the whole causal effect.

**LayerNorm guard first**: `cos(μ, uniform)` = +0.005 / −0.007, share surviving
mean-subtraction **1.0000**. None of it lives in LayerNorm's null direction.

**Aimed at attention read-space.** A random direction puts 0.0617 ± 0.0100
(n=200) of its squared norm in a head's 64-of-1024 K rowspace, taken through
the layer's LayerNorm gain. The rotation component puts **0.456 / 0.499** —
7–8x chance — while μ_clean is ordinary (0.091 / 0.120). K exceeds Q at both
steps (0.456 vs 0.323; 0.499 vs 0.398).

**Not a token signal.** The logit lens returns noise (`'urn'`, `'il'`, `'ats'`)
with entirely different token sets at the two checkpoints — consistent with a
position-independent direction, which cannot carry per-token match content.

**Aimed at the set.** Scored into all **272 heads downstream of MLP 6**,
`L7H8` ranks **5th / 4th** — high but not the target. Proximity is ruled out:
per-layer medians are flat (0.064–0.081 at 16000, layers 7–23). Membership is
what separates them:

| | members | non-members |
|---|---|---|
| within layer 7 | `L7H1`, `L7H8` — median **0.581** | 0.071 |
| within layer 8 | `L8H6`, `L8H9` — median **0.650** | 0.061 (max 0.435) |

Global ranks of 272 at 16000: `L7H1` #0, `L8H6` #1, `L8H9` #3, `L7H8` #5,
`L10H9` #6, `L11H14` #8 — **six of the top nine are catalogue members**, and
the same six lead at 143000. The members that are *not* targeted are `L12H5`
(#36/#43), `L9H13`, `L15H14` — and `L12H5` is the member that came out
uncoupled on every earlier instrument (0.0000 on `L7H8`'s attention; largest
negative residual against the magnitude rule). A fourth instrument agreeing
about `L12H5` was not designed in.

**Function-space check**, because §3.12-S's lesson binds: the other members have
no induction attention to restore (§3.18, all ≤0.015), so the readout is the
loss. With `L5H2` ablated (baseline NLL 2.551 / 1.761): zero 8.002 / 6.701;
random constant 8.09–8.99 / 7.99–8.32; **μ_clean 8.955 / 7.153**; **μ_cond
2.859 / 1.986**. One constant vector replaces MLP 6's whole position-varying
output to within **0.31 / 0.23 nats**, and μ_clean is **worse than deleting the
MLP** — the wrong operating point rather than a missing one. *Still geometry
only:* per-member attribution of the rotation to `L7H1`/`L8H6`/`L8H9`'s own
contributions is unrun; the loss arm is aggregate.

### Geometry predicts function (2026-09-13) — the rotation moves the heads it points at

`rotation_per_head_effect.py`. Synthesis in `PROJECT.md` §3.25. Closes the
weight-space gap the section above had to leave open (§3.12-S: weight-space
overlap is not function-space overlap).

**Minimal pair**, differing by exactly the rotation, both constant in MLP 6's
slot: REF = `L5H2` ablated with μ_cond (NLL 2.859), TEST = same with μ_clean
(NLL 8.955). **Readout is attention, not loss** — at 8.955 any per-head
marginal is ceiling-contaminated, while an attention distribution is still
well defined. Per head: total-variation distance between its attention in the
two states, which is role-agnostic (the FV-positive members have no induction
attention to measure, §3.18). **Zero check: layers 0–6 give TV exactly
0.00e+00.**

| layers > 6 (n=272) | members (9) | non-members (263) | Mann-Whitney |
|---|---|---|---|
| step 16000 | **0.2850** | 0.1471 | p = 1.1e-05 |
| step 143000 | **0.2483** | 0.1201 | p = 1.0e-02 |

Within layers 7, 8 and 10 the members exceed the non-member **maximum** at both
checkpoints (L7: 0.353 / 0.281 vs max 0.249 / 0.154; L8: 0.321 / 0.323 vs
0.265 / 0.225; L10: 0.321 / 0.318 vs 0.289 / 0.231).

**TV tracks `frac_K` head by head**, and layer control strengthens it:

| | raw ρ | layer-centred ρ | mean within-layer | layers positive |
|---|---|---|---|---|
| 16000 | +0.318 (7.9e-08) | **+0.325** (4.2e-08) | +0.285 | 14/17 |
| 143000 | +0.223 (2.1e-04) | **+0.337** (1.1e-08) | +0.318 | 15/17 |

TV rises with depth on its own (per-layer medians 0.03 → 0.16), so the raw
ranking carries accumulation; centring each layer on its own median removes it
and the relationship gets *stronger*.

**The misses are the geometry's own**: `L9H13` (frac_K rank #104/#89) sits
*below* its layer's non-member median at both steps; `L12H5` (#36/#43) is
barely above at 143000. **Two things not to quote**: the raw top-TV list at
143000 is dominated by layers 22–23 (accumulation — at 16000, five of the top
nine are members), and `L15H14` flips sign between checkpoints (0.264 vs 0.113
at 16000; 0.044 vs 0.101 at 143000).

`data/analysis/mlp_relay_role.json`, `mlp6_content_vs_scale.json`,
`mlp6_response.json`, `mlp6_decode_direction.json`,
`rotation_per_head_effect.json` — git-ignored.

## What is open

**Priority, set 2026-09-10.** This is the thread the project is working on, and
`PROJECT.md` §3.14.4 states the three questions driving it — what the members are
doing, whether there is any relationship between them, and why they form in one
narrow window. Read §3.14.4 before picking from the list below: it converts each
question into the specific measurement that answers it, and it flags that **the
members are not known to be independent** — one pair has ever been measured and
it is strongly redundant.

**Pass 2 is done** (above) and **the geometric axis is done**. What is left:

- **Q4 — structure per member.** Largely on disk and unread along this axis:
  `qk_symmetry_sweep.json` (384 heads × 19 steps), `ov_per_head_series.json`,
  `copying_score_sweep.json`, `behavioural_series.json`. §3.12-U says which
  *steps* matter — the action is in `(512, 4000]`, not at the endpoints.
- **Q5 — classes.** Needs Q4. Cluster on (formation step, spectral signature, QK
  symmetry trajectory, copying score, causal magnitude); report both views
  per §3.13.
- **`L11H14` has its own phase now — `p7e_consolidation/`.** Four independent
  measurements single it out (above), and the consolidation experiment designed
  there uses it as the hardest case.
- **Causal usefulness per rank, which nothing here measured.** Every rank in
  this phase is a *variance* rank — participation ratio, r90, CKA — and none of
  them says which directions carry the causal effect. `induction_rank_sweep.
  truncate` already ablates OV restricted to rank `r` in the head's 64-dim
  core; sweeping `r` per member gives `dNLL(r)` and turns "effective rank" into
  "useful rank". Note the tension worth measuring: the OV core is **rank ≤ 64**
  while the measured *effect* subspaces run to **150–300** dims — the effect is
  much wider than the head's own write rank, which is the geometric face of
  §3.12-Q's "compounds down the stack".
- **The step-1000 circuit, which is a different circuit.** At step 1000 the
  mechanism is `L5H2` (+4.97) and `L11H14` (+3.57), with `L7H8` absent and every
  other member under +0.06. `L11H14` is the top copier in the model at 143000
  but does not enter the copying top-5 until step 16000, and its step-1000 score
  is **not on disk** — `copying_score_sweep.json` keeps only the top ten and a
  named set. Measuring it is a weights-only step and comes first.

## The pilot that must not be over-read

`--heads L5H2,L11H14 --pair L5H2,L11H14 --steps 1000`, restore exact: singles
+4.973 and +3.567, joint **+6.859**, interaction **−1.682**, δ-cosine +0.857.
Sub-additive — the serial signature, opposite in sign to the `L7H8` pair.

**The sign is not evidence.** That joint arm lands at NLL 11.77 against a uniform
ceiling of 10.83, so it is outside the readout entirely, and §3.12-M5 warns that
this readout's compression biases independent contributions toward exactly this
apparent sub-additivity. Settle it with the graded readout (§3.12-M's KL / λ
scale) before running more steps at raw `dNLL`. See design-7d.md, "The readout's
ceiling".

## Reproducing

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

python -u p7d_redundancy/redundancy_catalog.py                    # Q1,  ~25 min
python -u p7d_redundancy/member_formation_curves.py --top 6 --chunk 4
python -u p7d_redundancy/member_formation_curves.py --append \
        --steps 3000,5000,7000,9000                               # Q2/Q3, ~20 min
python -u p7d_redundancy/two_big_heads.py                         # §3.12-S

python -u p7d_redundancy/pairwise_interaction_matrix.py \
        --top 10 --seqs 16 --steps 16000,143000                   # pass 2, ~50 min
python -u p7d_redundancy/member_subspace_geometry.py \
        --top 6 --seqs 16 --chunk 2 \
        --steps 1000,2000,3000,4000,5000,9000,16000,32000,54000,143000
```

`--chunk 2` on the geometry runner is not optional at 16 sequences: the first
full-grid run was **killed for memory at the last checkpoint** with `--chunk 4`.
Outputs are written per step, so a kill loses only the step in flight and
`--append --steps <the missing one>` finishes the job.

Outputs land in `data/analysis/*.json`, which is git-ignored — see PROJECT.md's
resume block for the full list. Use `--out` when running a different `--pair` or
`--heads`: a partial run replaces the main six-member curve otherwise.

The FV experiment, ~2 min per checkpoint:

```
python -u p7d_redundancy/fv_score.py \
        --steps 1000,2000,4000,8000,16000,143000 \
        --top 6 --controls 6 --n-prompts 16 --n-shot 10 --seqs 8 --chunk 4
```

It writes `data/analysis/fv_score.json` per step, so a kill loses only the step
in flight. `--controls` is the measured null and is not optional: patching any
vector into a corrupted prompt perturbs it, so the FV chance level is not zero,
and at step 143000 the largest control reads **0.0103**.
