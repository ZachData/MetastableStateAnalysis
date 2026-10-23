<!-- trackC_writeup/draft-C2.md — rendered by `python -m trackC_writeup.numbers render` from draft-C2.src.md and numbers.json; edit the source, not this file. -->


# The interaction energy along a language model's training

## 1. Introduction

Read a transformer's depth as time and its tokens as particles on a sphere, and self-attention
becomes an interaction between them [Geshkovski et al. 2023]. Under the simplest weights (one
head with Q = K = V = I, the same in every layer) the particles attract and cluster, and the
interaction energy E_β of §2.1 never decreases with depth. That picture organises a growing
theory of clustering and metastability (§8). Trained language models do not follow it: at the
end of its training, `pythia-410m`'s E₁ falls between consecutive layers 70 times over
our seven prompts. No theorem is broken, since none covers a trained model (§2.1),
but the picture's prediction fails, and nothing says when along training it starts to fail, or
what else is changing then.

We read E₁ layer by layer at 18 checkpoints of `pythia-410m` and of nine retrained seeds,
counting V, the drops between consecutive layers (§3). We report two dated events, their place on
the model's developmental order, and one negative result:

1. **The break, dated and reproduced (§4).** Through step 128 E₁ is monotone in depth on every
   prompt (V = 0 on seed 0); V reaches half its 1000–4000 level at step
   512. A prediction registered before any seed's intermediate checkpoints were
   downloaded put the break at 256 or 512 on each of nine PolyPythias seeds, and it falls there on
   **9 of 9**. The date is the recipe's, not one run's.
2. **At its step, the break is in the tokens' common direction (§4).** With each prompt's mean
   token removed before normalising, the count at 512 on seed 0 is 5–11 (against
   56 uncentred), and the centred cloud's own break comes one grid step later, at
   1000, on seed 0 and on eight of the nine seeds at every tolerance
   read. That is the step at which the previous-token and induction heads are dated.
3. **The timeline (§5).** On seed 0 the break comes after the token cloud's rank collapse and the
   unigram and bigram minima, with the OV maps' attractive share falling, and one grid step before
   the previous-token and induction heads. **"Break, then induction, then the OV maps' repulsive
   phase" is not shown:** in either reading of that phase it ties the break or induction on this
   grid.
4. **One intervention, no reversal (§6).** Making every head's OV map symmetric, or removing its
   rotation, does not take the break back towards monotone at any step after it; giving every
   head its own step-128 OV map lowers V at 512 and 1000 but leaves it far from monotone. The
   break is neither shown to be the OV maps' nor shown not to be.

**What none of this separates (§7).** Every dated event from step 128 to 1000 falls inside the
learning-rate warmup, which runs to step 1430 and which all ten runs share. Two
further runs do not separate a training event from the schedule. On `pythia-70m` and
`pythia-160m`, which warm up over the same steps at higher peak rates, both dates are consistent
with the count of updates and with the cumulative learning rate, not with the rate itself. On
Mistral's GPT-2 small, whose warmup is logarithmic and longer, the break's larger rise falls where
the count of updates predicts, after a smaller first rise where the cumulative rate does, and the
label the frozen rule gives turns on two choices that rule made. So the dates are dates inside a
warmup, not training events told apart from it.

Everything here is tier 1 on one model size, seven hand-picked prompts and a grid
with no checkpoint between 512 and 1000; only the seed count of §4 was registered. §2 gives the
energy and the exact conditions under which it is monotone, §3 the setup, §8 related work;
appendix A reads the heads' shared subspaces along the same steps, and appendix B the gate each
reading passed before it was read.

## 2. Background

### 2.1 The energy, and when it is monotone in depth

Put each token's residual-stream state on the unit sphere, xᵢ ∈ S^{d−1}, i = 1, …, n, and let

  E_β(X) = (1 / 2βn²) Σᵢ Σⱼ exp(β ⟨xᵢ, xⱼ⟩).

E_β is large when the tokens are close together and is maximised by a single cluster; on the
sphere its minimiser is the uniform measure [Geshkovski et al. 2023, Prop. 3.4]. Read a
transformer's depth as continuous time for an interacting-particle system, with self-attention
the interaction. When every layer has the same single head with Q = K = V = I, the dynamics
ascend E_β, so E_β is non-decreasing in depth [Geshkovski et al. 2023, §3.2]; with V = −I it is
non-increasing. Wider hypotheses give monotonicity of a different, weighted energy, not of E_β:
for one head whose score matrix QᵀK is symmetric and equal to V, the dynamics ascend
(1/2β) Σᵢⱼ exp(β⟨Vxᵢ, xⱼ⟩) in a metric that weights each token by its attention normaliser
[ibid., §3.4.1], and for many heads whose score matrices are each symmetric and equal (or close)
to their value maps, a sum of energies weighted by those matrices is non-decreasing [Pendharkar
2026, Theorems 11–12], though a single head's own energy can fall through cross-head terms
[ibid., Theorem 17]. No result covers MLP blocks, values untied from the scores, weights that
change from layer to layer, or discrete layers, and trained transformers satisfy the symmetry
and tying conditions nowhere exactly [ibid., §10].

So no theorem says a trained language model's E_β should be monotone in depth, and none says
a randomly initialised one's should. We read monotonicity as the prediction of the
identity-weight picture, and a drop in E_β between consecutive layers as a departure from that
picture, not as the violation of a theorem. That a model at initialisation is (almost) monotone,
and a trained one is not, is then an empirical fact about this model, which §4 dates.

### 2.2 The developmental order

Language models acquire their internal structure in stages. In a small two-layer attention-only
model, Hoogland et al. [2024] find bigram statistics first, then positional use and common 3- and
4-grams, then previous-token heads, then induction heads, then a long stage with no significant
change, each stage a period of change in the local learning coefficient. Induction heads form at
the point where in-context learning improves abruptly, visible as a bump in the training loss
[Olsson et al. 2022]. On Pythia, a model's loss
on sequences sampled from unigram and bigram statistics falls to a trough between steps 2⁶ and
2⁸ and rises until about 2¹⁰ [Belrose et al. 2024], and task-level components emerge at similar
token counts across model sizes [Tigges et al. 2024]. §5 places the energy's break on this order
for one model, `pythia-410m`, with every row dated by the same locator.

### 2.3 Pythia's training, and its warmup

The Pythia models are trained for 143 000 steps of 1024 sequences of 2048 tokens
[Biderman et al. 2023], with the learning rate warmed up linearly over the first
1 % of training (1430 steps) and decayed by a cosine to
0.1 of its peak; the peak is 3 × 10⁻⁴ for `pythia-410m`,
6 × 10⁻⁴ for `pythia-160m` and 10⁻³ for `pythia-70m`, and the two
smaller models warm up over the same number of steps (1430 and
1430) [the released configs]. The suite
releases checkpoints at steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and then every 1000 steps. Every early event
§4 and §5 date, the energy's break included, therefore falls inside the warmup, and nothing in
one schedule separates a change caused by training from one caused by the schedule. Li et al.
[2025] tie the first of their three geometric phases to the same ramp. PolyPythias [van der Wal
et al. 2025] retrains `pythia-410m` from nine further seeds with the same schedule, which lets
§4 ask whether the break's step is the run's or the recipe's, but not whether it is the
schedule's (§7).

## 3. Setup

### 3.1 The model, the checkpoints and the prompts

`pythia-410m` has 24 layers of width 1024, each with 16 attention
heads (384 in all). We read 18 of its checkpoints, steps
0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16 000, 32 000, 54 000, 143 000. Step 1 is left out: in our runs its outputs are identical to step 0's, so
it adds no point. A grid step named as a place on the axis (512, 1000) is a label; a step that
is a result is quoted from its record. There is no checkpoint between 512 and 1000, or between
1000 and 2000, so nothing below dates a change more finely than the grid.

PolyPythias [van der Wal et al. 2025] retrains `pythia-410m` from nine further random seeds,
`pythia-410m-seed1` to `pythia-410m-seed9`, with the same code, data and schedule but a different
initialisation and data order, and releases the same checkpoints; we call the original run seed 0. PolyPythias reports two of the nine, seeds
3 and 4, as outlier runs; we keep both and flag them, as fixed before any of their
checkpoints was read.

Every seed and step reads the same seven texts: a passage of Camus's *L'Étranger* in
French, a method from the Python source of the HDBSCAN library, a passage of the *Iliad* in
English translation, the source of a LaTeX document, a paragraph from a paper on transformers,
Sullivan Ballou's 1861 letter to his wife, and a Wikipedia paragraph on Charlotte Brontë, each
cut at 512 tokens (prompt-set hash `976b62d677bc`). They are few and
hand-picked, not sampled, and every count below is a count over these seven.
Forward passes run in float32 with TF32 off, on one GPU. On seed 0 the
per-state energies reproduce an earlier CPU run of this project to a relative
1.5 × 10⁻⁶, and the counts of §3.2 exactly.

### 3.2 The energy and the drop count

At each checkpoint and prompt we take the residual stream's 25 states as the model
returns them: the token embeddings, the outputs of the first 23 blocks, and the
last block's output after the final layer norm. Each token's vector is normalised to unit
length, so a prompt's n tokens lie on the sphere, and E₁ of §2.1 (β = 1) is computed at each
state, every position included. A step from one state to the next is a *drop* when E₁ falls by
more than a fraction 10⁻³ of its value. **V**, the statistic of everything below, is
the number of drops at one checkpoint: over the 24 steps of each prompt, summed over
the 7 prompts. In the identity-weight picture V is 0. A drop is a departure from
that picture's prediction, not the violation of a theorem (§2.1). The tolerance keeps float
noise out of the count; on seed 0 the break's date below is 512 at each of the
tolerances 10⁻⁴, 10⁻³ and 10⁻², and 512 with any one prompt left out.

β = 1 is the one setting seed 0 had been read at before the seeds were registered (§3.4). A
second energy, E₃.₅ in each block's input frame (after its first layer norm, positions 1 to
n − 1, blocks 0 to 23), is read beside it in §4 and never counted.

### 3.3 Dating the break

Let m be the median of V over steps 1000, 2000 and 4000. The break's step is the first grid
step at which V ≥ m / 2, a *half-rise*. The plateau is early on purpose: V keeps moving late in
training (on seed 0, 57 on the plateau and 70 at the last step), and a
reference taken there would let a late rise move an early date. Two other rules were set aside
on seed 0 before the seeds were read: the first step with V > 0 moves with a single stray drop,
and V's change-weighted centroid in log step lands at about 827 on seed 0,
pulled late by the rise between 16 000 and 32 000.

### 3.4 What was registered, and when

Before any seed's checkpoints between its first and last were downloaded, we committed a
prediction, `P-A5`, to this project's registry of predictions, kept in its git history: on each
of the nine seeds the break falls at 256 or 512, a window (128, 512] around seed 0's step. K
counts the seeds inside. The registered falsifier was K ≤ 3. The null puts each
seed's step uniformly and independently on the 12 non-zero grid steps up to 4000,
so under it K ~ Bin(9, 1/6); p = P(Bin(9, 1/6) ≥ K), exact, and the e-value is
E = κ p^(κ−1) with κ = 0.5, a p-to-e calibrator fixed before the run [Vovk and Wang
2019]. Nobody holds that null. It says the step is the run's, scattered over the early grid; it
is the alternative to "the step is the recipe's", and beating it says only that.

The registration commit (`c2eb9a7`) is dated 2026-10-20, 13:36 UTC−5; the
first seed record was written at 18:39 UTC the same day, 3
minutes later. Every seed's first and last checkpoints were already on disk, and an earlier
phase of this project had read the first.

## 4. The break, dated and reproduced

![Figure 1](fig/fig1_break.svg)

*Figure 1. V, the number of layer-to-layer drops in E₁ over the 7 prompts, at each
of the 18 steps, for seed 0 (blue) and the nine PolyPythias seeds; seeds
3 and 9, dated 256, in orange. Left, V; right, V over each run's 1000–4000 plateau
median, with the half-rise line. Blue band: the registered window (128, 512]. Shaded: the
learning-rate warmup, to step 1430. Step 0 is drawn one octave left of step 2.*

**Seed 0.** V is 0 at every step up to 128: on no prompt does E₁ fall by more than
the tolerance from one of its 25 states to the next. It is 19 at 256 and 56 at 512, against a plateau
median of 57, so the half-rise dates the break at **512** (figure 1), and V
stays high to the end of training (70 at the last step).

**The nine seeds.** The break falls inside the window on all nine: **K = 9 of 9**,
p = 9.9 × 10⁻⁸, E = 1587 against the uniform-location null. Seven seeds date it
at 512, as seed 0 does; seeds 3 and 9 date it at 256. Both outlier runs, seeds
3 and 4, are inside. No seed is near the window's edges: at 128 every seed's V is at
most 2, against half-plateau lines of at least 22; at 512 every V is at
least 41, against lines of at most 29. A count that moved by one, the
float-noise margin seed 0's gate allows, could move a seed between 256 and 512, never out of
the window.

**What K = 9 says, and what it does not.** The break's step belongs to 410m's recipe, not
to one run: the date §5 places on the developmental order is the recipe's. It does not separate
a training event from the learning-rate schedule. Every seed shares the warmup, which runs to
step 1430, past the whole window, so a break set by the schedule's ramp would
reproduce exactly as this one does (§7). And the null it beats is one nobody holds, so a large E
here is cheap: the check makes the date reproducible, and is evidence neither about mechanism
nor that trained weights resist the clustering the particle picture describes.

**Beside the count** (fixed before the run, never scored):

- *Step 0 is not monotone on every seed.* Seeds 1, 3, 5, 8 and 9 show
  1–6 drops at step 0 (seed 0 none), up to 8 at step
  4 (seed 3); V is 0 at steps 8 to 64 on every
  seed. Against plateaus of 44–58 these are small, but "monotone
  at initialisation" holds for seed 0 and four of the nine seeds, not for all ten.
- *The second energy agrees.* E₃.₅ in each block's input frame dates the break at 512 on
  9 of the 9 seeds, and at 512 on seed 0.

### 4.1 The token-centred cloud

E_β is a moment of the tokens' Gram matrix, so a direction all tokens share, growing or
shrinking from layer to layer, moves it with no change in how the tokens sit relative to one
another (§8). To remove it we subtract each prompt's mean token at each state before normalising,
and count the drops in E₁ of the centred cloud, V_c, with the same half-rise. The rule was fixed
before any centred number was read. Centring changes E₁'s scale, so V_c is counted at four
tolerances, 7.6 × 10⁻⁵, 10⁻⁴, 3 × 10⁻⁴ and 7.6 × 10⁻⁴: two fixed, and two matched to the uncentred count's behaviour (on
re-initialised models, and on a planted spread; appendix B). A centred date counts only if it
holds at all four. No single tolerance reads a planted change alike through both counts, so we
compare their dates, never their sizes.

**Seed 0.** V_c is 0 at every step through 128 and 5–11 at 512 across
the four tolerances, against V = 56; its half-rise is at **1000** at
each. **The nine seeds.** The centred half-rise falls after the seed's uncentred break on
eight of nine seeds, and at 1000 on eight; on seed 6
one of the four tolerances dates it at the seed's own break. Read after the dates: at the
uncentred break V_c is 0.07–0.50 of its own plateau on the seeds that break at 512
and 0.00–0.04 on the two that break at 256, below half on every seed but the one
tie.

So at its step the break is the common direction's, by the half-rise: with the mean token
removed the cloud has not yet broken at 256 or 512, and it breaks one grid step later, at the
step where §5 dates the previous-token and induction heads. How much of the uncentred count the
common direction carries is not measured. The rule frozen for seed 0 carried a size label, "mostly
removed" (centred plateau 20 against 57), and we withdraw it:
at the matched tolerance 10⁻⁴ the centred plateau is 0.96 of the uncentred
one, so the label measured centring's change of scale, not a smaller break.

## 5. The timeline

Where does the break fall against what else the model learns? On seed 0's 18 steps we
dated the break among nine other measured changes with one locator (table 1, figure 2).

**The locator.** For a row x(s), signed so that its change is positive, let Δ(s) = x(s) − x(0).
The reference is the largest Δ over steps up to 4000, and the row's step is the first at which
Δ reaches half the reference. A row whose reference is under 4 bootstrap standard
errors (over the row's prompts, heads, layers or text blocks) reads "no change". The two n-gram
rows fall and then rise, and are dated at their minimum. The energy row E1 is dated by §3.3's
half-rise instead, and the learning-rate row E10 by the same half-point with no floor. Before
any real row was dated, the locator was gated on null series and on changes planted in each gap
of the grid; the first gate failed on four rows, whose spread grows by orders of magnitude over
training, and the revised gate passes on every row dated here (appendix B).

*Table 1. Seed 0's rows, ordered by their dated step. All are on `pythia-410m`'s 18
steps, v1's 7 prompts, except the n-gram rows, read on 64 held-out text blocks from
the Pile's validation split. E1c, the centred cloud (§4.1), is dated by §3.3's half-rise and was
added after the other rows were read.*

| row | what changes | dated step |
|---|---|---|
| E5 | effective rank of the token cloud on the sphere, mean over prompts, falls | 8 |
| E6u | KL divergence from a unigram predictor to the model's next-token distribution, at its minimum | 128 |
| E6b | the same for a bigram predictor (also 256 with the bigram counted on the rest of the split) | 256 |
| E1 | the energy break (§4) | 512 |
| E2 | the OV maps' attractive share: per head, the share of the OV map's squared eigenvalue mass on eigenvalues with positive real part, median over the 384 heads, falls | 512 |
| E8 | the weights' spectral tail exponent α [Martin and Mahoney 2019], median over matrix types and layers, falls | 1000 (512 from its peak) |
| E3 | induction: attention of two induction heads to the token after an earlier copy of the current one | 1000 |
| E4 | previous-token attention of head `L5H2` | 1000 |
| E1c | the token-centred cloud's break (§4.1), the same at each of its four tolerances | 1000 |
| E10 | the learning-rate schedule, its half-point | 1000 |
| E7 | relative weight change ‖W − W₀‖ / ‖W₀‖ | censored at 2000 |
| E11 | a simulator's cluster count from the measured start | refused |

![Figure 2](fig/fig2_timeline.svg)

*Figure 2. Seed 0's rows by step, each as the share of its change (Δ over its reference; E1 as
V over its plateau median, E1c as V_c at 10⁻³ over its own; the n-gram rows as the fall into
their minimum), cut at −1.2. Dot:
the dated step. Shaded: the warmup, to step 1430.*

**The order.** E5 (8) < E6u (128) < E6b
(256) < E1 = E2 (512) < E3 = E4 = E1c = E10 (1000), with
E8 between the two groups: dated from step 0 it ties the second (1000), dated from
its peak the first (512). The rank collapse and the unigram minimum come before the break; the
break falls with the OV fall, one grid step after the bigram minimum and one before the
previous-token and induction heads, which the centred cloud's break joins (on the seeds too, at
1000 on eight of nine). E1c is not one of A3's rows and its pairs are not
counted below. Of the 11 pairs read with the break and among the
circuit rows, 7 are aligned within one grid step, 2 are not (the
rank collapse and the unigram minimum come first) and 2 are unread (E7 censored,
E11 refused). That E1 and E2 align is not new: this project's earlier medians of the OV maps
already put E2's fall at the same step.

**Every aligned pair is inside the warmup.** The schedule's own row reads 1000 by construction
(the learning rate reaches half its peak at step 715, between grid steps), and inside a linear ramp "within one grid step" includes "driven by the same ramp". Nothing in
this table separates a training event from the schedule (§7).

**"Break, then induction, then the OV phase" is not shown.** In one reading the OV phase is E2's
fall, and it starts with the break, at 512; in the other it is E2's bottom, its
first step at its minimum, which is 1000, the step induction is dated. Either way the OV phase and the
break or induction are tied on this grid, not ordered.

**Inside the 1000 tie** (description, read after the dates). The share of each change done at
512, the grid step before, orders the rows the half-point ties: the break
0.98, the OV fall 0.81, previous-token
0.35, induction 0.03; the schedule is
0.36 done, beside previous-token. On the seeds, with each run's own
previous-token and induction heads picked by their scores at the last step, the two tie at
1000 on 8 of the 8 seeds where both are located; previous-token is further
along at 512 on 7 (shares 0.40–0.48 against
0.007–0.085), seed 4 the other way, and on seed
3 the induction head is not located. Previous-token attention is further
along than induction at 512, the order the stagewise literature reports (§2.2); the grid cannot
date the gap.

## 6. Is the break the OV maps'?

The monotonicity results of §2.1 need each head's value map tied to its symmetric scores, and
the OV maps' attractive share falls at the break's step (§5). So the first suspect is the OV maps
turning away from symmetric, attracting maps. On seed 0 we edited every head's OV map at once in
the forward pass and counted V again, at step 0, 256 and the five steps from 512
on. Two edits ask whether the break is the OV maps' asymmetry: (a) each map replaced by its
symmetric part; (b) each map's rotation removed, in an orthonormal basis of each of its
two-dimensional invariant planes, keeping its gain. Each is read against 40 random edits
matched to it in size and in each head's norm: a cell is *lower* or *higher* only beyond all 40,
and only beyond what the edit's change of gain alone does. In-context copying is read beside it,
as the repeat gain G of a probe of 16 sequences of 96 random tokens repeated once (the mean loss
on the first copy's positions minus that on the second's), at the four steps from 1000,
where copying exists.

![Figure 3](fig/fig3_ov.svg)

*Figure 3. V on seed 0 under each edit (blue) at the steps read, the unedited model (grey) and
the range of the 40 matched random edits (orange). Left, (a) each OV map's symmetric part;
middle, (b) its rotation removed; right, every head given its own step-128 OV map. The steps are
evenly spaced.*

**Neither edit undoes the break** (figure 3). Under (a) V is lower than its controls at
none of the five steps and higher at 512 and 143 000; under
(b) it is as its controls at every step. (b) is a small edit: removing the rotation changes a
head's map by a median 0.08–0.10 of its norm. Copying survives it: G is
8.19–12.41 at the four steps, as its controls, where zeroing the OV maps of ten
induction heads takes it to 1.06–2.57 (appendix B).

**The one OV-only edit with a known destination does not reach it either.** Giving every head
its own step-128 OV map, where V was 0, leaves V at 29–61 across the five steps.
The edit is not inert: V is lower than every control at 512 (29 against
50–66) and at 1000 (31 against 38–58), and
higher at 2000 (56 against 33–51), one grid step after the
OV maps' attractive share reaches its lowest (§5). The whole model moved to its step-128 weights, by contrast,
gives V = 0 at every step from 512 to 4000, so V does answer to a known
destination; an OV-only edit does not take it there.

So the break is neither shown to be the OV maps' nor shown not to be. One OV-only edit with a
known destination was tried, and it moves V in both directions without approaching monotone;
the two structural edits show no reversal beyond random edits of their size. None of this is
causal evidence about which components push the tokens apart.

## 7. What would separate the schedule from training

Every dated row of §5 from step 128 to 1000 falls inside the warmup, and the ten runs of §4 share
it, so a break set by the ramp would reproduce exactly as this one does. Three kinds of run
could separate the two: a model trained on the same schedule at a different peak rate; a suite
with a different warmup and checkpoints inside it; and continued training from a checkpoint
before the break, with and without the ramp. We ran the first two and read both dates, E₁'s
break and the centred cloud's, by rules fixed before their outputs.

**The predictions.** Each schedule says where a date at 410m's step should fall on another run:
under *step*, at the same count of updates; under the *cumulative* learning rate (the dose the
weights have received), where that run's summed rate reaches 410m's at the date; under the
*instantaneous* rate, where its rate does. Each is mapped from 410m's dates on the ten seeds
(256–512 for the break, 512–1000 for the centred date) onto the other
run's grid as a band, and a run's dates keep the schedules whose bands contain them. The
schedules are deterministic functions of the step, so a date consistent with one is not
evidence that it causes anything.

![Figure 4](fig/fig4_schedules.svg)

*Figure 4. Left, V over its plateau median on `pythia-70m` and `pythia-160m` (E₁ solid, the
centred cloud at 10⁻⁴ dashed) and on seed 0 of `pythia-410m`, with the half line. Right, V on
Mistral GPT-2 small's five seeds, with the bands where the cumulative rate (left) and
the count of updates (right) put E₁'s break. Shaded on the left: Pythia's warmup.*

**Pythia 70m and 160m.** They warm up over the same 1430 steps at peak rates of
10⁻³ and 6 × 10⁻⁴, against 3 × 10⁻⁴ for 410m, so the cumulative and
instantaneous rates reach 410m's earlier in steps. On both, E₁'s break is at 256
and the centred date at 1000, the same at every tolerance (figure 4). On 160m's
grid the centred date predicted under the instantaneous rate is 256–512
(128–512 on 70m), so the rate itself is excluded; step
(512–1000) and cumulative (256–1000) both contain 1000, and
for the break both contain 256. The joint reading is **consistent with step and cumulative**, and it is the same
without 70m's E₁ date, which rests on a plateau of 6 drops. Step against
the cumulative dose is not told apart on Pythia's grid; size changes with the rate, and there is
one seed per size.

**Mistral GPT-2 small.** Stanford CRFM's Mistral trained five seeds of a GPT-2 small with
checkpoints every 50 steps to 2000 [Stanford CRFM]. Its learning rate rises
logarithmically, reaching half its peak of 6 × 10⁻⁴ by step 100 and the
peak at about step 4000 (the logged rate), so the cumulative dose predicts far
earlier dates than step does, and here the schedules separate: for E₁'s break the cumulative
band is 40–150, step's 150–550. E₁'s break falls at
250 or 350 on all five seeds, in step's band only, and the frozen joint reading is
**"consistent with step"**. We do not take it as separating the two, for three reasons:

- *E₁ rises in two stages* on every seed (figure 4): to 14–17 drops by step 100,
  inside the cumulative band, then to 31–39 by 350, inside step's. The frozen plateau
  (the median over 4000–16 000, 410m's window mapped under tokens) puts its half between the
  stages; against 410m's own window, 1000–4000, the half-rise reads 80 or 200, and the
  joint would read "neither". The label holds for the larger stage and depends on the window.
- *The centred count gives no date.* Each seed's four tolerances disagree, from
  100–2000: the two fixed tolerances jump at 100–150, in the cumulative band,
  the two derived ones rise slowly and date 350–2000. By the rule a disagreement widens a
  seed's date to its span, and step survives the centred count on four seeds only
  through the tolerance matched to the re-initialised models, which is set by their single
  largest drop; at the next value the joint reads "neither".
- *Neither schedule maps the whole curve.* Under step, 410m has nothing at GPT-2's first stage
  (V is 0 through 128); under the cumulative dose, GPT-2's first stage maps onto most of 410m's
  rise, where GPT-2 is still far below its plateau.

So on another suite with a different warmup, the break's larger rise falls where the count of
updates predicts, after a first rise where the dose does, and the centred cloud's single date of
§4.1 does not transfer. Nothing here separates a training event from the learning-rate schedule
on `pythia-410m`. The run that could is the third kind: continued training from a checkpoint
before the break with the ramp replaced by a flat rate, on a model whose circuit is known.

## 8. Related work

**Geometry along the training axis.** Several works read a global geometric statistic of hidden
states along Pythia's checkpoints. Li et al. [2025] find three phases in the spectrum of the
last layer's representations (RankMe, αReQ), the first coinciding with the learning-rate ramp.
Billa [2026] finds task representations' RankMe collapsing from step 0 to steps 32–256 on
Pythia 160m and 410m and partly recovering by about step 1000. Cheng et al. [2024] follow
intrinsic dimension by layer from step 512, and Razzhigaev et al. [2023] report intrinsic
dimension rising and then falling over training. A non-monotone geometric arc along training
is therefore not new. What is new here is the interaction energy of the particle picture, its
monotonicity in depth read along a real training axis, the step at which it breaks, and that
step's reproduction across seeds.

**Anisotropy.** E_β is a moment of the tokens' Gram matrix, so it moves with anisotropy, the
tokens' shared direction. Trained transformers' representations are dominated by one principal
direction [Ben-Artzy and Schwartz 2026], and decoders' anisotropy peaks in their middle layers
[Razzhigaev et al. 2023]. Along Pythia's training, Bernas et al. [2026] study anisotropy chiefly
through the gradients, with activations read from step 10 000 on; we found no reviewed
measurement of activation anisotropy inside the break's window. §4.1 reports the energy on
token-centred clouds, the control this calls for.

**Developmental interpretability.** The stagewise picture of §2.2 comes from the local learning
coefficient [Hoogland et al. 2024], refined per head [Wang et al. 2024], and from
susceptibilities [Wang et al. 2025]; induction heads [Olsson et al. 2022] and the function-vector
heads many of them become [Yin and Steinhardt 2025] are its best-studied circuit. PolyPythias
[van der Wal et al. 2025] reports training dynamics highly consistent across seeds, with outlier
runs; §4's seeds include two such runs.

**Theory.** The particle picture of self-attention, its clustering and metastability, are
developed in Geshkovski et al. [2023, 2024], Bruno et al. [2024] and Rigollet [2025]; the
monotonicity conditions of §2.1 are in Geshkovski et al. [2023] and Pendharkar [2026].
Isobe et al. [2026] train a noisy mean-field model's feed-forward part and find tokens leaving
the clustered regime near the final layers, a predicted shape with no checkpoints. Herty and
Liu [2026] couple the tokens' flow in depth with the parameters' flow in training, as theory.

**Earlier notes.** The author's informal notes compared trained GPT-2, ALBERT and BERT models
with random weights of matched norm: under random weights the energy rose monotonically in
depth, as the identity-weight picture predicts, and in every trained model, prompt and β tested
it did not. They read only endpoints; §4 dates when, along training, the monotonicity is lost.

## Appendix A. The shared subspaces

Do the 384 heads come to query, key, read and write the same directions of the
residual stream, and when? For each family of per-head subspaces (query, key, OV write, OV read,
and the OV map's invariant subspaces with negative real, positive real and complex eigenvalues)
we average the heads' orthogonal projectors and take the participation ratio PR of the mean.
Under a null that rotates each head's subspace by an independent Haar rotation PR has an exact
expected value PR₀, and ρ = PR / PR₀ is 1 under it and falls as the heads concentrate on shared
directions. Each step's PR is read against the null's range over its draws. The null, a
norm-matched Gaussian draw of the weights and step 0 are one law, and the instrument passes it
at every step. ρ is a description: a count of shared directions cannot be calibrated on a
trained pool, and neither can ρ's effect side (the spectrum is dense at the null's edge, and a
planted effect that adds dimensions measures only its own dilution). Weights only; the
residual-stream check below used v1's prompts.

Through step 128 every family reads the null (|ρ − 1| at most 9.4 × 10⁻⁴ in the OV
families and 4.5 × 10⁻⁴ in QK). The query and key families leave it at
256, with the OV write (256) and OV complex
(256) families; OV read, negative and positive follow at
512. All of this is inside the warmup, at or before the break. OV write is the
most concentrated family at 512 and 1000 (ρ 0.89 and 0.66);
from 2000 the query family is (ρ 0.50), and from 4000 the query and key
families both are, ending at 0.51–0.53 against the OV families'
0.65–0.82 at the last step. The
shared spaces are not the residual stream's top-variance directions at any step, and at the last
step the OV write, negative and complex ones partly coincide with 410m's massive-activation
coordinates, which they avoid through 4000.

Shaikh et al. [2026] read an overlap of heads' projections in the weights of four larger models,
with checkpoints on OLMo-2 only: keys grow most, queries the same way more gently, values stay
near-orthogonal throughout. Our key family agrees in direction. Their value statistic is not our
OV read family (a different subspace, pooled differently), so the two are not compared.

## Appendix B. The gates, and why each null is the one used

Each reading above was gated before it was read: the full statistic was run on inputs whose
answer is known, on a planted null and on a planted effect at the real design points, and the
real data were read only once every arm passed. Where an arm failed, the failure and the change
are stated where the reading is.

**The break's half-rise (§3.3, §4; the gate of `P-A5` on seed 0).** *Known answer:* seed 0's
energies reproduce an earlier CPU run to a relative 1.5 × 10⁻⁶ (the gate's
tolerance 10⁻⁴), and V exactly. *Planted breaks:* a synthetic series breaking at
each grid step is read at that step for 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000; one planted at 4000, the
plateau's last step, reads 2000, since the plateau then contains the rise; a series is read
inside the window exactly when its break was planted there. *Planted null:* 20 000
draws of the nine seeds' steps from the uniform-location null, through the scorer, reproduce
Bin(9, 1/6) for K (largest standardised gap 2.4). *Planted effect:* seed 0's
series moved one grid step later reads 1000, outside the window. *Sensitivity:* the
date is the same at every tolerance and with any one prompt left out (§3.2). *Why this null:*
it is the alternative to "the recipe sets the step", scattering each run's step over the early
grid. Nobody holds it, which is why beating it is cheap (§4).

**The locator (§5).** Each row's no-change null (bootstrap draws of the row with its change
removed) reads "no change" in at least 0.972 of 1000 draws, and changes
planted in each gap of the grid at the row's own noise are placed in the right gap. The first
gate failed on four rows whose spread grows by orders of magnitude over training; the revised
gate, with each row's floor on its own spread, passes on every row dated, and the simulator's
row (E11) is refused. *Why:* a date is only as good as the locator's power to tell a change from
none in each gap.

**The OV edits (§6).** Random edits are never labelled lower or higher against their own
controls (G1: 0 labels). Zeroing ten induction heads' OV maps loses copying at
four of four steps (G2: G 1.06–2.57). Moving the whole model to its step-128
weights lowers V below its controls at four steps at full dose, to V =
0 (G3), so the count can see a reversal when one is made. A displacement guard,
meant to leave unread any cell whose edit moves the residual stream further than its controls,
refused both planted effects, and was made description before any arm was read. *Why random
edits matched in size and in each head's norm:* a structured edit has to be told from what any
edit of its size does, and the gain change alone is read beside, because the scale of a head's
map moves V by itself.

**The centred counts (§4.1, §7).** The steps no schedule reaches (0–128 on Pythia, 0 and 10 on
GPT-2 small) are read first, and a centred count there of half its plateau or more would refuse
the control; none did. The two derived tolerances are fixed per model before its records:
one so that the re-initialised models' centred drop rate matches their uncentred one, one so
that a planted spread is seen by the centred count as by the uncentred one; each tolerance is
gated on the re-initialised models and on two planted effects (a common vector added to every
token, which the centred count must not see, and a spread, which it must). One tolerance on the
centred energy's own scale, meant to compare the two counts' sizes, failed its own gate, which
is why only dates are compared. The centred rules were frozen after a preflight
(`tools.design_preflight`), which runs the statistic on step 0, a matched random
initialisation, the trained population at a step off the scored grid, and a planted control at
dose 0 and three doses; its refusals are fixed or argued in the design.

## References

Titles and authors as on each paper's arXiv page (read 2026-10-22); the year is the first
arXiv version's.

- Belrose, N., Pope, Q., Quirke, L., Mallen, A., Fern, X. (2024). Neural networks learn statistics of increasing complexity. arXiv:2402.04362.
- Ben-Artzy, A., Schwartz, R. (2026). Global divergence, local convergence: representation geometry in SSMs and transformers. arXiv:2609.08692.
- Bernas, R., Jourdan, F., Poché, A., Hudelot, C. (2026). Revisiting anisotropy in language transformers: the geometry of learning dynamics. arXiv:2604.08764.
- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., van der Wal, O. (2023). Pythia: a suite for analyzing large language models across training and scaling. arXiv:2304.01373.
- Billa, J. (2026). The geometric anatomy of capability acquisition in transformers. arXiv:2602.15997.
- Bruno, G., Pasqualotto, F., Agazzi, A. (2024). Emergence of meta-stable clustering in mean-field transformer models. arXiv:2410.23228.
- Cheng, E., Doimo, D., Kervadec, C., Macocco, I., Yu, J., Laio, A., Baroni, M. (2024). Emergence of a high-dimensional abstraction phase in language transformers. arXiv:2405.15471.
- Geshkovski, B., Letrouit, C., Polyanskiy, Y., Rigollet, P. (2023). A mathematical perspective on Transformers. arXiv:2312.10794.
- Geshkovski, B., Koubbi, H., Polyanskiy, Y., Rigollet, P. (2024). Dynamic metastability in the self-attention model. arXiv:2410.06833.
- Herty, M., Liu, H. (2026). A mean-field theory of transformers: well-posedness of the coupled data–parameter dynamics and global convergence of training. arXiv:2608.25055.
- Hoogland, J., Wang, G., Farrugia-Roberts, M., Carroll, L., Wei, S., Murfet, D. (2024). Loss landscape degeneracy and stagewise development in transformers. arXiv:2402.02364.
- Isobe, N., Inoue, D., Imaizumi, M. (2026). Training-induced escape from token clustering in a mean-field formulation of transformers. arXiv:2605.07772.
- Li, M. Z., Agrawal, K. K., Ghosh, A., Teru, K. K., Santoro, A., Lajoie, G., Richards, B. A. (2025). Tracing the representation geometry of language models from pretraining to post-training. arXiv:2509.23024.
- Martin, C. H., Mahoney, M. W. (2019). Traditional and heavy-tailed self regularization in neural network models. arXiv:1901.08276.
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., Olah, C. (2022). In-context learning and induction heads. arXiv:2209.11895.
- Pendharkar, A. (2026). Gradient flow structure and quantitative dynamics of multi-head self-attention. arXiv:2605.04279.
- Razzhigaev, A., Mikhalchuk, M., Goncharova, E., Oseledets, I., Dimitrov, D., Kuznetsov, A. (2023). The shape of learning: anisotropy and intrinsic dimensions in transformer-based models. arXiv:2311.05928.
- Rigollet, P. (2025). The mean-field dynamics of transformers. arXiv:2512.01868.
- Shaikh, K., Singh, A. K., Dsouza, R. C., Shiromani, S. (2026). Linear predictability of attention heads in large language models. arXiv:2603.13314.
- Stanford CRFM. Mistral GPT-2 small checkpoints, `stanford-crfm/alias-gpt2-small-x21`, `battlestar-gpt2-small-x49`, `caprica-gpt2-small-x81`, `darkmatter-gpt2-small-x343`, `expanse-gpt2-small-x777`, the Hugging Face Hub (tags `checkpoint-N`; read 2026-10-22). No paper is cited for them.
- Tigges, C., Hanna, M., Yu, Q., Biderman, S. (2024). LLM circuit analyses are consistent across training and scale. arXiv:2407.10827.
- van der Wal, O., Lesci, P., Muller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., Biderman, S. (2025). PolyPythias: stability and outliers across fifty language model pre-training runs. arXiv:2503.09543.
- Vovk, V., Wang, R. (2019). E-values: calibration, combination, and applications. arXiv:1912.06116.
- Wang, G., Hoogland, J., van Wingerden, S., Furman, Z., Murfet, D. (2024). Differentiation and specialization of attention heads via the refined local learning coefficient. arXiv:2410.02984.
- Wang, G., Baker, G., Gordon, A., Murfet, D. (2025). Embryology of a language model. arXiv:2508.00331.
- Yin, K., Steinhardt, J. (2025). Which attention heads matter for in-context learning? arXiv:2502.14010.
