<!-- p7d_redundancy/status-7d.md -->
# Phase 7d — STATUS

**Last verified:** 2026-09-10.
**Overall:** Q1, Q2 and Q3 are answered on pythia-410m, and **pass 2 (the
pairwise interaction matrix) and the geometric axis are now answered too**. Q4
is mostly on disk and unread; Q5 has a partial answer from geometry alone.
Nothing is registered and nothing can be — every measurement here is on an
artifact spent under `check_registry` rule 3. Restore checks are exact
(`0.0e+00`) on every run reported below.

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
