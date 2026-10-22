<!-- trackC_writeup/draft-C2.md — rendered by `python -m trackC_writeup.numbers render` from draft-C2.src.md and numbers.json; edit the source, not this file. -->


# The interaction energy along a language model's training

## 1. Introduction

*(unit 4)*

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
2021]. Nobody holds that null. It says the step is the run's, scattered over the early grid; it
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
the Pile's validation split.*

| row | what changes | dated step |
|---|---|---|
| E5 | effective rank of the token cloud on the sphere, mean over prompts, falls | 8 |
| E6u | KL divergence from a unigram predictor to the model's next-token distribution, at its minimum | 128 |
| E6b | the same for a bigram predictor (also 256 with the bigram counted on the rest of the split) | 256 |
| E1 | the energy break (§4) | 512 |
| E2 | the OV maps' attractive share: per head, the share of the OV map's squared eigenvalue mass on eigenvalues with positive real part, median over the 384 heads, falls | 512 |
| E8 | the weights' spectral tail exponent α, median over matrix types and layers, falls | 1000 (512 from its peak) |
| E3 | induction: attention of two induction heads to the token after an earlier copy of the current one | 1000 |
| E4 | previous-token attention of head `L5H2` | 1000 |
| E10 | the learning-rate schedule, its half-point | 1000 |
| E7 | relative weight change ‖W − W₀‖ / ‖W₀‖ | censored at 2000 |
| E11 | a simulator's cluster count from the measured start | refused |

![Figure 2](fig/fig2_timeline.svg)

*Figure 2. Seed 0's rows by step, each as the share of its change (Δ over its reference; E1 as
V over its plateau median; the n-gram rows as the fall into their minimum), cut at −1.2. Dot:
the dated step. Shaded: the warmup, to step 1430.*

**The order.** E5 (8) < E6u (128) < E6b
(256) < E1 = E2 (512) < E3 = E4 = E10 (1000), with
E8 between the two groups: dated from step 0 it ties the second (1000), dated from
its peak the first (512). The rank collapse and the unigram minimum come before the break; the
break falls with the OV fall, one grid step after the bigram minimum and one before the
previous-token and induction heads. Of the 11 pairs read with the break and among the
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

*(unit 4)*

## 7. What would separate the schedule from training

*(unit 4, after units 3, 3b and 3c)*

## 8. Related work

**Geometry along the training axis.** Several works read a global geometric statistic of hidden
states along Pythia's checkpoints. Li et al. [2025] find three phases in the spectrum of the
last layer's representations (RankMe, αReQ), the first coinciding with the learning-rate ramp.
Billa [2026] finds task representations' RankMe collapsing from step 0 to steps 32–256 on
Pythia 160m and 410m and partly recovering by about step 1000. Cheng et al. [2025] follow
intrinsic dimension by layer from step 512, and Razzhigaev et al. [2024] report intrinsic
dimension rising and then falling over training. A non-monotone geometric arc along training
is therefore not new. What is new here is the interaction energy of the particle picture, its
monotonicity in depth read along a real training axis, the step at which it breaks, and that
step's reproduction across seeds.

**Anisotropy.** E_β is a moment of the tokens' Gram matrix, so it moves with anisotropy, the
tokens' shared direction. Trained transformers' representations are dominated by one principal
direction [Ben-Artzy and Schwartz 2026], and decoders' anisotropy peaks in their middle layers
[Razzhigaev et al. 2024]. Along Pythia's training, Bernas et al. [2026] study anisotropy chiefly
through the gradients, with activations read from step 10 000 on; we found no reviewed
measurement of activation anisotropy inside the break's window. §7 reports the energy on
token-centred clouds, the control this calls for.

**Developmental interpretability.** The stagewise picture of §2.2 comes from the local learning
coefficient [Hoogland et al. 2024], refined per head [Wang et al. 2024], and from
susceptibilities [Wang et al. 2025]; induction heads [Olsson et al. 2022] and the function-vector
heads many of them become [Yin and Steinhardt 2025] are its best-studied circuit. PolyPythias
[van der Wal et al. 2025] reports training dynamics highly consistent across seeds, with outlier
runs; §4's seeds include two such runs.

**Theory.** The particle picture of self-attention, its clustering and metastability, are
developed in Geshkovski et al. [2023, 2025], Bruno et al. [2025] and Rigollet [2026]; the
monotonicity conditions of §2.1 are in Geshkovski et al. [2023] and Pendharkar [2026].
Isobe et al. [2026] train a noisy mean-field model's feed-forward part and find tokens leaving
the clustered regime near the final layers, a predicted shape with no checkpoints. Herty and
Liu [2026] couple the tokens' flow in depth with the parameters' flow in training, as theory.

**Earlier notes.** The author's informal notes compared trained GPT-2, ALBERT and BERT models
with random weights of matched norm: under random weights the energy rose monotonically in
depth, as the identity-weight picture predicts, and in every trained model, prompt and β tested
it did not. They read only endpoints; §4 dates when, along training, the monotonicity is lost.

## Appendix A. The shared subspaces

*(unit 4)*

## Appendix B. The gates, and why each null is the one used

*(unit 4)*

## References

*(unit 4 builds the list, with each id's title and authors read from the paper; the ids, each in `lit-C2-unit1.md` or `trackA_force/lit-C2.md`:
2312.10794, 2605.04279, 2402.02364, 2209.11895, 2402.04362, 2407.10827, 2304.01373 (Pythia),
2509.23024, 2503.09543, 2602.15997, 2405.15471, 2311.05928, 2604.08764, 2609.08692,
2410.02984, 2508.00331, 2502.14010, 2410.23228, 2512.01868, 2605.07772, 2608.25055; Geshkovski
et al. 2025 to be identified at unit 2)*
