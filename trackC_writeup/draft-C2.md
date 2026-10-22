<!-- trackC_writeup/draft-C2.md — C2's draft, the Markdown source (design-C2.md "Numbers and figures").
State: §2 and §8 drafted at unit 1 (2026-10-22); §3–§5 at unit 2, §1, §6, §7 and the appendices
at unit 4. Unit 2 replaces every number below that comes from a record or a config with a
numbers.py key; until then the numbers here are the configs' and the papers'. Each citation's
mark ([H], [A], [R], [S]) is in `trackC_writeup/lit-C2-unit1.md` or `trackA_force/lit-C2.md`;
a claim resting on an [S] row is not stated here, and an [A] row is cited only for what its
abstract says. -->


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

The Pythia models are trained for 143 000 steps of 1024 sequences of 2048 tokens, with the
learning rate warmed up linearly over the first 1 % of training (1430 steps) and decayed by a
cosine to a tenth of its peak; the peak is 3 × 10⁻⁴ for `pythia-410m`, 6 × 10⁻⁴ for
`pythia-160m` and 10⁻³ for `pythia-70m` [Biderman et al. 2023; the released configs]. The suite
releases checkpoints at steps 0, 1, 2, 4, …, 512 and then every 1000 steps. Every early event
§4 and §5 date, the energy's break included, therefore falls inside the warmup, and nothing in
one schedule separates a change caused by training from one caused by the schedule. Li et al.
[2025] tie the first of their three geometric phases to the same ramp. PolyPythias [van der Wal
et al. 2025] retrains `pythia-410m` from nine further seeds with the same schedule, which lets
§4 ask whether the break's step is the run's or the recipe's, but not whether it is the
schedule's (§7).

## 3. Setup

*(unit 2)*

## 4. The break, dated and reproduced

*(unit 2)*

## 5. The timeline

*(unit 2)*

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

*(unit 2 builds the list; the ids, each in `lit-C2-unit1.md` or `trackA_force/lit-C2.md`:
2312.10794, 2605.04279, 2402.02364, 2209.11895, 2402.04362, 2407.10827, 2304.01373 (Pythia),
2509.23024, 2503.09543, 2602.15997, 2405.15471, 2311.05928, 2604.08764, 2609.08692,
2410.02984, 2508.00331, 2502.14010, 2410.23228, 2512.01868, 2605.07772, 2608.25055; Geshkovski
et al. 2025 to be identified at unit 2)*
