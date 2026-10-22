<!-- trackC_writeup/draft-C2.src.md — C2's draft, the Markdown source (design-C2.md "Numbers and figures").
`python -m trackC_writeup.numbers render` fills each {{…}} slot from numbers.json and writes
draft-C2.md; edit this file, never the render. A number from this project's records or from
Pythia's released configs is a key; a number a paper reports is that paper's and carries its
citation. State: §2 and §8 drafted at unit 1 (2026-10-22); §3–§5 and the three figures at unit 2;
§1, §6, §7, the appendices and the references at unit 4. Each citation's mark ([H], [A], [R],
[S]) is in `trackC_writeup/lit-C2-unit1.md` or `trackA_force/lit-C2.md`; a claim resting on an
[S] row is not stated here, and an [A] row is cited only for what its abstract says. -->


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

The Pythia models are trained for {{cfg.train_iters}} steps of 1024 sequences of {{cfg.seq_len}} tokens
[Biderman et al. 2023], with the learning rate warmed up linearly over the first
{{cfg.warmup_pct|int}} % of training ({{cfg.warmup_steps}} steps) and decayed by a cosine to
{{cfg.lr_floor_ratio|f1}} of its peak; the peak is {{cfg.lr_410m|sci}} for `pythia-410m`,
{{cfg.lr_160m|sci}} for `pythia-160m` and {{cfg.lr_70m|sci}} for `pythia-70m`, and the two
smaller models warm up over the same number of steps ({{cfg.warmup_160m}} and
{{cfg.warmup_70m}}) [the released configs]. The suite
releases checkpoints at steps {{cfg.early_saves|list}} and then every {{cfg.save_every}} steps. Every early event
§4 and §5 date, the energy's break included, therefore falls inside the warmup, and nothing in
one schedule separates a change caused by training from one caused by the schedule. Li et al.
[2025] tie the first of their three geometric phases to the same ramp. PolyPythias [van der Wal
et al. 2025] retrains `pythia-410m` from nine further seeds with the same schedule, which lets
§4 ask whether the break's step is the run's or the recipe's, but not whether it is the
schedule's (§7).

## 3. Setup

### 3.1 The model, the checkpoints and the prompts

`pythia-410m` has {{cfg.layers}} layers of width {{cfg.width}}, each with {{cfg.heads}} attention
heads ({{cfg.heads_total}} in all). We read {{grid.n}} of its checkpoints, steps
{{grid.steps|list}}. Step 1 is left out: in our runs its outputs are identical to step 0's, so
it adds no point. A grid step named as a place on the axis (512, 1000) is a label; a step that
is a result is quoted from its record. There is no checkpoint between 512 and 1000, or between
1000 and 2000, so nothing below dates a change more finely than the grid.

PolyPythias [van der Wal et al. 2025] retrains `pythia-410m` from nine further random seeds,
`pythia-410m-seed1` to `pythia-410m-seed9`, with the same code, data and schedule but a different
initialisation and data order, and releases the same checkpoints; we call the original run seed 0. PolyPythias reports two of the nine, seeds
{{a5.outliers|and}}, as outlier runs; we keep both and flag them, as fixed before any of their
checkpoints was read.

Every seed and step reads the same {{a5.n_prompts|word}} texts: a passage of Camus's *L'Étranger* in
French, a method from the Python source of the HDBSCAN library, a passage of the *Iliad* in
English translation, the source of a LaTeX document, a paragraph from a paper on transformers,
Sullivan Ballou's 1861 letter to his wife, and a Wikipedia paragraph on Charlotte Brontë, each
cut at {{a5.max_tokens}} tokens (prompt-set hash `{{a5.prompts_hash}}`). They are few and
hand-picked, not sampled, and every count below is a count over these seven.
Forward passes run in {{a5.dtype}} with TF32 {{a5.tf32|onoff}}, on one GPU. On seed 0 the
per-state energies reproduce an earlier CPU run of this project to a relative
{{s0.energy_rel_diff|sci}}, and the counts of §3.2 exactly.

### 3.2 The energy and the drop count

At each checkpoint and prompt we take the residual stream's {{cfg.states}} states as the model
returns them: the token embeddings, the outputs of the first {{cfg.layers_m1}} blocks, and the
last block's output after the final layer norm. Each token's vector is normalised to unit
length, so a prompt's n tokens lie on the sphere, and E₁ of §2.1 (β = 1) is computed at each
state, every position included. A step from one state to the next is a *drop* when E₁ falls by
more than a fraction {{rel_tol|sci}} of its value. **V**, the statistic of everything below, is
the number of drops at one checkpoint: over the {{cfg.layers}} steps of each prompt, summed over
the {{a5.n_prompts}} prompts. In the identity-weight picture V is 0. A drop is a departure from
that picture's prediction, not the violation of a theorem (§2.1). The tolerance keeps float
noise out of the count; on seed 0 the break's date below is {{s0.tol_steps|uniq}} at each of the
tolerances {{s0.tols|scis}}, and {{s0.loo_steps|uniq}} with any one prompt left out.

β = 1 is the one setting seed 0 had been read at before the seeds were registered (§3.4). A
second energy, E₃.₅ in each block's input frame (after its first layer norm, positions 1 to
n − 1, blocks 0 to {{cfg.layers_m1}}), is read beside it in §4 and never counted.

### 3.3 Dating the break

Let m be the median of V over steps 1000, 2000 and 4000. The break's step is the first grid
step at which V ≥ m / 2, a *half-rise*. The plateau is early on purpose: V keeps moving late in
training (on seed 0, {{s0.plateau}} on the plateau and {{s0.V_end}} at the last step), and a
reference taken there would let a late rise move an early date. Two other rules were set aside
on seed 0 before the seeds were read: the first step with V > 0 moves with a single stray drop,
and V's change-weighted centroid in log step lands at about {{a3.E1_centroid|f0}} on seed 0,
pulled late by the rise between {{s0.late_rise[0]}} and {{s0.late_rise[1]}}.

### 3.4 What was registered, and when

Before any seed's checkpoints between its first and last were downloaded, we committed a
prediction, `P-A5`, to this project's registry of predictions, kept in its git history: on each
of the nine seeds the break falls at 256 or 512, a window (128, 512] around seed 0's step. K
counts the seeds inside. The registered falsifier was K ≤ {{reg.falsifier_K}}. The null puts each
seed's step uniformly and independently on the {{grid.n_to4000}} non-zero grid steps up to 4000,
so under it K ~ Bin(9, {{reg.p0}}); p = P(Bin(9, {{reg.p0}}) ≥ K), exact, and the e-value is
E = κ p^(κ−1) with κ = {{kappa|f1}}, a p-to-e calibrator fixed before the run [Vovk and Wang
2021]. Nobody holds that null. It says the step is the run's, scattered over the early grid; it
is the alternative to "the step is the recipe's", and beating it says only that.

The registration commit (`{{reg.commit}}`) is dated {{reg.at|date}}, {{reg.at|time}} {{reg.at|tz}}; the
first seed record was written at {{reg.first_record|time}} UTC the same day, {{reg.lag_min}}
minutes later. Every seed's first and last checkpoints were already on disk, and an earlier
phase of this project had read the first.

## 4. The break, dated and reproduced

![Figure 1](fig/fig1_break.svg)

*Figure 1. V, the number of layer-to-layer drops in E₁ over the {{a5.n_prompts}} prompts, at each
of the {{grid.n}} steps, for seed 0 (blue) and the nine PolyPythias seeds; seeds
{{a5.at256|and}}, dated 256, in orange. Left, V; right, V over each run's 1000–4000 plateau
median, with the half-rise line. Blue band: the registered window (128, 512]. Shaded: the
learning-rate warmup, to step {{cfg.warmup_steps}}. Step 0 is drawn one octave left of step 2.*

**Seed 0.** V is {{s0.Vmax_to128}} at every step up to 128: on no prompt does E₁ fall by more than
the tolerance from one of its {{cfg.states}} states to the next. It is {{s0.V256}} at 256 and {{s0.V512}} at 512, against a plateau
median of {{s0.plateau}}, so the half-rise dates the break at **{{s0.break}}** (figure 1), and V
stays high to the end of training ({{s0.V_end}} at the last step).

**The nine seeds.** The break falls inside the window on all nine: **K = {{a5.K}} of {{a5.n}}**,
p = {{a5.p|sci}}, E = {{a5.E|f0}} against the uniform-location null. {{a5.n512|Word}} seeds date it
at 512, as seed 0 does; seeds {{a5.at256|and}} date it at 256. Both outlier runs, seeds
{{a5.outliers|and}}, are inside. No seed is near the window's edges: at 128 every seed's V is at
most {{a5.V128_max}}, against half-plateau lines of at least {{a5.half_min}}; at 512 every V is at
least {{a5.V512_min}}, against lines of at most {{a5.half_max}}. A count that moved by one, the
float-noise margin seed 0's gate allows, could move a seed between 256 and 512, never out of
the window.

**What K = {{a5.K}} says, and what it does not.** The break's step belongs to 410m's recipe, not
to one run: the date §5 places on the developmental order is the recipe's. It does not separate
a training event from the learning-rate schedule. Every seed shares the warmup, which runs to
step {{cfg.warmup_steps}}, past the whole window, so a break set by the schedule's ramp would
reproduce exactly as this one does (§7). And the null it beats is one nobody holds, so a large E
here is cheap: the check makes the date reproducible, and is evidence neither about mechanism
nor that trained weights resist the clustering the particle picture describes.

**Beside the count** (fixed before the run, never scored):

- *Step 0 is not monotone on every seed.* Seeds {{a5.step0_seeds|and}} show
  {{a5.step0_min}}–{{a5.step0_max}} drops at step 0 (seed 0 none), up to {{a5.early_max}} at step
  {{a5.early_max_step}} (seed {{a5.early_max_seed}}); V is {{a5.mid_max}} at steps 8 to 64 on every
  seed. Against plateaus of {{a5.plateau_min}}–{{a5.plateau_max}} these are small, but "monotone
  at initialisation" holds for seed 0 and {{a5.step0_clean|word}} of the nine seeds, not for all ten.
- *The second energy agrees.* E₃.₅ in each block's input frame dates the break at 512 on
  {{a5.ln1_512}} of the 9 seeds, and at {{s0.ln1_break}} on seed 0.

## 5. The timeline

Where does the break fall against what else the model learns? On seed 0's {{grid.n}} steps we
dated the break among nine other measured changes with one locator (table 1, figure 2).

**The locator.** For a row x(s), signed so that its change is positive, let Δ(s) = x(s) − x(0).
The reference is the largest Δ over steps up to 4000, and the row's step is the first at which
Δ reaches half the reference. A row whose reference is under {{a3.floor|int}} bootstrap standard
errors (over the row's prompts, heads, layers or text blocks) reads "no change". The two n-gram
rows fall and then rise, and are dated at their minimum. The energy row E1 is dated by §3.3's
half-rise instead, and the learning-rate row E10 by the same half-point with no floor. Before
any real row was dated, the locator was gated on null series and on changes planted in each gap
of the grid; the first gate failed on four rows, whose spread grows by orders of magnitude over
training, and the revised gate passes on every row dated here (appendix B).

*Table 1. Seed 0's rows, ordered by their dated step. All are on `pythia-410m`'s {{grid.n}}
steps, v1's {{a5.n_prompts}} prompts, except the n-gram rows, read on {{a3.ngram_blocks}} held-out text blocks from
the Pile's validation split.*

| row | what changes | dated step |
|---|---|---|
| E5 | effective rank of the token cloud on the sphere, mean over prompts, falls | {{a3.rows[E5]|step}} |
| E6u | KL divergence from a unigram predictor to the model's next-token distribution, at its minimum | {{a3.rows[E6u]|step}} |
| E6b | the same for a bigram predictor (also {{a3.E6b_full}} with the bigram counted on the rest of the split) | {{a3.rows[E6b]|step}} |
| E1 | the energy break (§4) | {{a3.rows[E1]|step}} |
| E2 | the OV maps' attractive share: per head, the share of the OV map's squared eigenvalue mass on eigenvalues with positive real part, median over the {{cfg.heads_total}} heads, falls | {{a3.rows[E2]|step}} |
| E8 | the weights' spectral tail exponent α, median over matrix types and layers, falls | {{a3.rows[E8]|step}} ({{a3.E8_peak}} from its peak) |
| E3 | induction: attention of two induction heads to the token after an earlier copy of the current one | {{a3.rows[E3]|step}} |
| E4 | previous-token attention of head `L5H2` | {{a3.rows[E4]|step}} |
| E10 | the learning-rate schedule, its half-point | {{a3.rows[E10]|step}} |
| E7 | relative weight change ‖W − W₀‖ / ‖W₀‖ | {{a3.rows[E7]|step}} |
| E11 | a simulator's cluster count from the measured start | {{a3.rows[E11]|step}} |

![Figure 2](fig/fig2_timeline.svg)

*Figure 2. Seed 0's rows by step, each as the share of its change (Δ over its reference; E1 as
V over its plateau median; the n-gram rows as the fall into their minimum), cut at −1.2. Dot:
the dated step. Shaded: the warmup, to step {{cfg.warmup_steps}}.*

**The order.** E5 ({{a3.rows[E5]|step}}) < E6u ({{a3.rows[E6u]|step}}) < E6b
({{a3.rows[E6b]|step}}) < E1 = E2 ({{a3.rows[E1]|step}}) < E3 = E4 = E10 ({{a3.rows[E3]|step}}), with
E8 between the two groups: dated from step 0 it ties the second ({{a3.rows[E8]|step}}), dated from
its peak the first ({{a3.E8_peak}}). The rank collapse and the unigram minimum come before the break; the
break falls with the OV fall, one grid step after the bigram minimum and one before the
previous-token and induction heads. Of the {{a3.n_pairs}} pairs read with the break and among the
circuit rows, {{a3.n_aligned}} are aligned within one grid step, {{a3.n_misaligned}} are not (the
rank collapse and the unigram minimum come first) and {{a3.n_unread}} are unread (E7 censored,
E11 refused). That E1 and E2 align is not new: this project's earlier medians of the OV maps
already put E2's fall at the same step.

**Every aligned pair is inside the warmup.** The schedule's own row reads 1000 by construction
(the learning rate reaches half its peak at step {{cfg.warmup_half}}, between grid steps), and inside a linear ramp "within one grid step" includes "driven by the same ramp". Nothing in
this table separates a training event from the schedule (§7).

**"Break, then induction, then the OV phase" is not shown.** In one reading the OV phase is E2's
fall, and it starts with the break, at {{a3.rows[E2]|step}}; in the other it is E2's bottom, its
first step at its minimum, which is {{a3.E2_bottom}}, the step induction is dated. Either way the OV phase and the
break or induction are tied on this grid, not ordered.

**Inside the {{a3.rows[E3]|step}} tie** (description, read after the dates). The share of each change done at
512, the grid step before, orders the rows the half-point ties: the break
{{a3.share512[E1]|f2}}, the OV fall {{a3.share512[E2]|f2}}, previous-token
{{a3.share512[E4]|f2}}, induction {{a3.share512[E3]|f2}}; the schedule is
{{a3.share512[E10]|f2}} done, beside previous-token. On the seeds, with each run's own
previous-token and induction heads picked by their scores at the last step, the two tie at
{{a5.d10_tie_steps|uniq}} on {{a5.d10_tied}} of the {{a5.d10_n}} seeds where both are located; previous-token is further
along at 512 on {{a5.d10_prev}} (shares {{a5.d10_prev_share|range2}} against
{{a5.d10_ind_share|range3}}), seed {{a5.d10_other|and}} the other way, and on seed
{{a5.d10_unlocated|and}} the induction head is not located. Previous-token attention is further
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
