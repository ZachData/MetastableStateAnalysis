<!-- p7e_consolidation/design-7e.md -->
# Phase 7e — DESIGN

## Two objects, one phase

**`L11H14`**, the member that does the set's job from an orthogonal subspace;
and **consolidation**, the attempt to collapse the whole set into one head.
They are the same question from opposite ends. If the set's members are
functionally interchangeable but structurally distinct — which is 7d's central
finding — then either that distinctness is load-bearing, or it is incidental. A
successful consolidation says incidental. `L11H14` is the case that decides it,
because it is the furthest from everyone else.

## What 7d established, and must not be re-derived

- **One set, not several.** 44/45 pairwise cells positive at step 16000, no
  block structure. But 74–81 % of the interaction is the product of the two
  heads' own magnitudes — most of the matrix is scale, not pairing.
- **Direction and substitutability are decoupled.** δ-cosine explains 7–9 % of
  interaction across 45 pairs.
- **Born aligned, then fanning out.** Fixed-15-pair mean cosine peaks at step
  5000 (+0.744) and falls to +0.327 at 143000 while delta norms *grow*. The max
  pair stays at 0.87–0.97 throughout and the min pair goes +0.32 → −0.19.
- **`L11H14`, four ways.** Mean cosine to the set **−0.033** at 143000 at the
  third-largest delta norm (10.60) — against +0.389 for magnitude-matched
  `L7H1`. Effect subspace PR ~50 against everyone else's 8–28. Earliest
  defector (breaks away between steps 2000 and 4000). Already known as the
  step-1000 co-mechanism and the top copier at 143000.

## The capacity argument, which is why this is worth trying

The obvious objection is arithmetic: **one head's OV is rank ≤ 64** (`D_HEAD`),
while the members' measured *effect* subspaces run to **150–300 dimensions** and
their union is larger still. One head cannot span six heads' effects.

**This was checked first, and the prediction written here was wrong.** The
argument had been: the ambient participation ratio is only ~20, the members put
59 % of their energy in the ambient top-50, so the space to be spanned is small
and fits easily. `ambient_budget.py` measured it and it does not.

| | step 16000 | step 143000 |
|---|---|---|
| ambient participation ratio | 22.0 | 8.4 |
| joint effect, its own basis, 90 % | 186 dims | 141 dims |
| joint effect, **ambient ordering**, 90 % | **355 dims** | **255 dims** |
| energy inside ambient top-64 | **0.732** | **0.780** |

Reaching 90 % of the joint effect needs **more** ambient directions (355) than
the effect's own basis needs (186), against an ambient participation ratio of
22. **The set writes substantially into directions the residual stream barely
uses** — private low-variance bandwidth, not the shared trunk. That is the
opposite of the prediction and a more useful fact than the one expected: it also
explains why the members can be near-orthogonal to each other while all being
"induction heads".

**The phase still proceeds, because 73–78 % of the joint energy does fit in 64
dimensions, and energy is not usefulness.** A tail spread over hundreds of
near-unused directions may carry almost no function. The capacity question is
therefore **causal, not geometric**, and that is now the first measurement:
truncate the joint effect to rank `k` and measure how much of the `dNLL`
survives. If rank 64 recovers most of the causal effect, consolidation is
viable despite failing the energy criterion.

**A trap in that follow-up.** The joint-ablation arm lands at NLL **9.90 against
the uniform ceiling of 10.83** — within 0.93 nats. By §3.14.4-D's rule the joint
`dNLL` of +9.37 is a **floor, not a measurement**. The energy geometry above is
ceiling-immune and unaffected, but **any causal reading of the joint arm needs
§3.12-M's graded readout first**.

**Caveat carried from 7d**: that PR is measured on copied positions of repeated
*random-token* sequences, a deliberately narrow distribution. Natural text would
be far higher, and a consolidation that holds on the induction probe and breaks
on natural text is the expected failure mode, not a surprise. **Both readouts,
always** — see "What would falsify it".

## The intervention, stated precisely

The user's framing — "take a head and expand it to the head of another one,
regurgitate that, doubling the original head and deleting the second one" — has
a well-posed form and several ill-posed ones. The distinction matters.

**Ill-posed: copying weights.** Writing `A`'s OV into `B`'s slot does not give
`B` `A`'s computation. The members sit in **different layers** (5, 7, 8, 8, 11,
12), so they read different residual states and attend with different QK
patterns. A transplanted OV would be driven by the wrong attention pattern.

**Well-posed: re-fit the survivor.** Ablate member `B`, then solve for the
survivor `A`'s OV factors that best restore what the full model did:

> `min over (W_V^A, W_O^A) of  ‖ resid_full − resid_(B ablated, A refit) ‖_F`,
> subject to `rank ≤ D_HEAD`.

This asks `A` to absorb `B`'s function using `A`'s own attention pattern and
`A`'s own position in the stack — which is the only thing it could actually do.
Iterate over members in some order until one head carries the set.

**The symmetry the user asked for enters here.** `A` and `B` produce aligned
effects through different subspaces, so the map from one to the other is close
to a rotation. §2.5's **isometric path** `M(t) = γ(t) Σ γ(1−t)ᵀ` — exact
isometry, rank `k`, listed in `PROJECT.md` as the only *designed* intervention
still unrun — is machinery for exactly this, and applying it to consolidation
rather than to `L7H8` alone is a better use of it than the original plan.

## Order matters, and that is itself a measurement

There are `n!` consolidation orders. Two are principled and they test different
things:

- **Descending magnitude** (`L5H2` first): can the biggest head absorb the rest?
- **`L11H14` last**: can the *aligned core* absorb the orthogonal outlier? This
  is the interesting one. If every member folds in except `L11H14`, its
  orthogonality was load-bearing, and 7d's outlier becomes a mechanism rather
  than a curiosity.

Report both. Never pick the order that worked.

## What would falsify it

A consolidation is only interesting if it is held to the standard the original
model meets. Four checks, and **all four are required**:

1. **Second-copy NLL** on the induction probe, against the unmodified model.
2. **Natural-text loss**, on held-out text. The narrow probe is where
   consolidation is easiest and where a false success will appear.
3. **The interaction matrix goes flat.** After consolidation the surviving head
   should carry ≈ the set's joint ablation cost, and **every remaining pairwise
   interaction should fall to ~0**. This is the direct test of "the redundancy
   is gone", and it is the claim's real content.
4. **Restore-exactness** on every arm, as everywhere in 7d.

## The interpretability claim, and its limit

The motivating hope is that a network with one head per computation is easier to
read than one with six. That is plausible and worth testing, but **a
consolidated model is a different artifact from the one whose interpretability
was in question.** Success shows the redundancy was *removable*, not that the
original was secretly simple — the original still computed the behaviour six
ways. Stated as model editing this is a real result; stated as "the network was
interpretable all along" it is not supported and should not be written that way.

This is also a **second, independent reason nothing here is registrable**, on
top of the spent-artifact rule: the measurements are taken on a model this
project modified.

## The gate measurement — **run, and it splits the set in two**

`useful_rank.py`, step 16000, 16 sequences, matched-norm random controls,
restore exact. `r*` is the smallest OV rank recovering 90 % of the head's own
causal effect, out of `D_HEAD = 64`.

| member | effect `dNLL` | **`r*`** | control `r*` |
|---|---|---|---|
| `L7H8` | +1.107 | **1** | 48 |
| `L12H5` | +0.431 | **1** | 48 |
| `L8H9` | +0.127 | **2** | 48 |
| `L5H2` | +2.227 | **12** | 24 |
| `L8H6` | +0.219 | **24** | 48 |
| `L11H14` | +0.187 | **64** | *never* |

**Five of the six members are effectively low-rank operators.** `L7H8` — the
head §3.11–§3.12 was built on — recovers **97 % of its causal effect from a
single direction**, and `L12H5` and `L8H9` are the same. Their useful ranks sum
to **40, inside one head's 64-dim budget**, and because those five are also the
mutually *aligned* ones (7d cosines 0.7–0.9) their union is smaller still.
**Consolidation of the aligned core is viable**, and the energy criterion that
`ambient_budget.py` failed was indeed the wrong currency.

**`L11H14` alone blows the budget, and does something stranger.** Its recovery
curve sits **below the matched-norm random control at nearly every rank**, and
at `r = 1` recovery is **negative (−0.096)** — truncating to its top singular
direction is *worse than deleting the head entirely*. For this head the
Eckart-Young optimal approximation is **anti-informative**: singular-value
magnitude does not order its causal usefulness, and slightly anti-orders it.

That is §3.12-R's finding (`‖OV‖_F` explains 0.1 % of causal effect and runs
backwards) and §3.12-G6's (no spectral field predicts causal effect) reproduced
**inside a single head**, and it is the sharpest instance of it the project has.

### The inversion, measured directly — `useful_rank.py --bottom`

Keeping the **smallest** `r` singular directions instead of the largest turns
the inference above into a measurement. `--bottom` at step 16000, 16 sequences,
restore exact:

| | `r*` | top-1 | bottom-32 | bottom-48 |
|---|---|---|---|---|
| `L7H8` | 1 | **+0.971** | +0.041 | +0.095 |
| `L5H2` | 12 | +0.123 | +0.109 | +0.616 |
| `L11H14` | 64 | **−0.096** | **+0.717** | **+0.846** |

For `L11H14` the ordering is **bottom-`r` > matched-random > top-`r` at every
rank**: the Eckart-Young optimal approximation, provably the best approximation
of the *operator*, is the **worst** of the three at preserving the *function*.
For `L7H8` the same test is textbook — one direction carries 97 %, and the 48
smallest carry 9.5 %.

**So the set has two classes, not a gradient.** `L7H8` and `L5H2` differ in how
low-rank they are and are both ordered correctly by gain; `L11H14` differs in
kind. This is the **sixth** independent axis separating it.

**Methodological consequence, and it reaches outside this phase.** Any
rank-truncation analysis of these heads that uses SVD order will systematically
mislead on `L11H14`-like heads — including
`induction_rank_sweep`'s entire `r*` construction, whose `svd` basis is one of
its two. Its **`schur` basis orders by eigenvalue rather than gain and carries a
sign**, so it may not inherit the defect; comparing the two on `L11H14` is
weights-only and costs no forward passes. Until that is done, **no `r*` from the
`svd` basis should be quoted for a head that has not been checked with
`--bottom`.**

**So the `L11H14`-last test is already answered without surgery: its
orthogonality is load-bearing.** It is the one member that cannot be folded in,
on the same axis that makes it the outlier everywhere else.

## What is open

- **The core consolidation**, five members into one survivor, now that the rank
  budget is known to permit it. Rank is **necessary, not sufficient**: the
  survivor must reproduce those effects through its **own** QK pattern at its
  **own** layer, which the rank arithmetic does not capture.
- **`r*` at other checkpoints.** This is step 16000 only, and 7d showed the
  set's geometry moves sharply between 5000 and 143000. `r*` is also
  grid-resolved — `r* = 12` means "between 8 and 12".
- **Why `L11H14` is full-rank and anti-ordered**, which is now the most
  interesting single question in the phase.
- The joint arm still needs §3.12-M's graded readout before any causal reading
  (it sits 0.93 nats from the ceiling).

## What this phase must not lose

- **The spent-artifact rule twice over.** pythia-410m is spent under
  `check_registry` rule 3, and every measurement after the first surgery is on
  a model this project modified. Nothing here is registrable.
- **Both readouts.** Induction probe *and* natural text. The probe is where
  consolidation is easiest and where a false success appears first.
- **§3.13's report-both rule**, and 7d's harder lesson beneath it: **any
  set-level mean whose membership changes as heads form will manufacture a
  trend.** It caught 7d three times (`union_ratio`, set-level cosine, centered
  CKA). Fix the pair set, or print `n`.
- **Energy is not usefulness.** The distinction is the whole reason this phase
  survived its own first measurement.
