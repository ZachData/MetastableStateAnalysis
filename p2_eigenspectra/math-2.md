# Phase 2 — MATH (study notes)

## 0. What this document is

Companion to `math-1.md`, `math-1b.md`, `math-1c.md`. **Read `math-1.md` §1.4 and §1A first** —
the attractive/repulsive dichotomy derived there is the entire subject of this phase.

Phase 1 measures the *outcome* of the attractive/repulsive tension: energy violations, plateaus,
cluster structure. **Phase 2 measures the tension itself.** That division — outcome vs.
mechanism — is why it is a separate directory rather than an extension of Phase 1's analysis
loop. Its claim is specific and falsifiable: *V's mixed-sign eigenspectrum is the cause of the
energy violations Phase 1 observes, not just a story consistent with them.*

**Scope note.** The results here are the pre-Pythia GPT-2/ALBERT/BERT study, closed as reported
(35 model×prompt runs, 2026-04-28). The Pythia rerun is active work and is not retrofitted into
this phase's verdict table. §8 covers why the port is an *upgrade* rather than a port.

---

## 1. Where $V$ enters, and why its spectrum is the question

### 1.1 The sign of the energy derivative is a statement about $V$

From `math-1.md` §1A.2, along the continuity equation

$$
\frac{d}{dt}E_\beta[\mu(t)] = \int \big\lVert\mathcal X[\mu](x)\big\rVert^2 Z_{\beta,\mu}(x)\,d\mu(x) \;\ge\;0
\qquad (V = +I_d)
$$

and the paper states plainly that $V = -I_d$ makes $E_\beta$ *decrease* along trajectories. The
dichotomy is exact at the level of this one identity: $V$ multiplies the velocity, so flipping
its sign flips $\dot E_\beta$ without changing the magnitude structure.

The two cases are named accordingly — **$V = +I_d$ attractive, $V = -I_d$ repulsive** — and
Proposition 3.4 gives them meaning: $E_\beta$ is maximized at a Dirac (full collapse) and
minimized at the uniform measure, so "energy up" is toward collapse and "energy down" is toward
spread.

### 1.2 A real $V$ is neither $+I$ nor $-I$ — it is both at once

This is the phase's founding observation and it is worth stating precisely, because the
codebase's variable names (`frac_attractive`, `frac_repulsive`) encode it.

A trained value circuit is a general $d\times d$ matrix. Decompose its action by the sign of the
real part of its eigenvalues. Along an invariant subspace where $\mathrm{Re}\,\lambda > 0$, the
map acts like $+\lambda I$ — locally attractive, energy-increasing. Along a subspace where
$\mathrm{Re}\,\lambda < 0$, it acts like $-|\lambda| I$ — locally repulsive, energy-decreasing.
**So a single trained layer runs the attractive dynamics along some directions and the repulsive
dynamics along others, simultaneously.**

That immediately supplies a mechanism for the observation Phase 1 could not explain:

> Energy monotonicity is violated. The paper's theorem is not wrong; the model is not in the
> theorem's setting. The violations should be the *repulsive subspace's* contribution showing
> through.

and it supplies the falsifiable version: **at a violation layer, the displacement should project
preferentially onto $V$'s repulsive subspace.** That is Phase 2's local test (§4.1).

The two summary scalars are just the counting measures of this split:

$$
\texttt{frac\_attractive} = \tfrac1d\#\{\mathrm{Re}\,\lambda_i > 0\},\qquad
\texttt{frac\_repulsive} = \tfrac1d\#\{\mathrm{Re}\,\lambda_i < 0\}
$$

*(Counting, not mass-weighting. See §10, open question 1 — this is a real limitation: a subspace
with one eigenvalue of magnitude 100 counts the same as one with magnitude $10^{-6}$.)*

### 1.3 The paper's own general-$V$ results: §9.2 and Table 1

The paper's Part 3 §9.2 (following [GLPR24]) drops the layer-norm projector and studies

$$
\dot x_i = \frac{1}{Z_{\beta,i}}\sum_j e^{\beta\langle Qx_i,\,Kx_j\rangle}\,V x_j
$$

Without normalization most particles diverge, because the dynamics "look like" $\dot x = Vx$
whose solutions are $e^{tV}x(0)$. The move that makes structure visible is the **rescaling**

$$
\boxed{\ z_i(t) = e^{-tV}x_i(t)\ }
$$

Differentiating, $\dot z_i = e^{-tV}(\dot x_i - Vx_i)$, which gives

$$
\dot z_i = \frac{1}{Z_{\beta,i}}\sum_j e^{\beta\langle Q e^{tV}z_i,\ K e^{tV}z_j\rangle}\,V\big(z_j - z_i\big)
$$

Two features matter and both are used by this phase:

- **The right-hand side is now a difference $(z_j - z_i)$** — a genuine interaction rather than a
  drift, so clustering statements become meaningful.
- **The attention coefficients $A_{ij}$ are identical for $z$ and $x$.** The rescaling changes the
  frame, not the routing. So it is a clean coordinate change, not a different model.

Table 1 of [GLPR24] then reports limit geometries by the spectral properties of $V$:

| $V$ | $K,Q$ | limit geometry |
|---|---|---|
| $V = I_d$ | $Q^\top K > 0$ | vertices of a convex polytope |
| $\lambda_1(V) > 0$, **simple** | $\langle Q\varphi_1, K\varphi_1\rangle > 0$ | **union of 3 parallel hyperplanes** |
| $V$ paranormal | $Q^\top K > 0$ | polytope × subspaces |
| $V = -I_d$ | $Q^\top K = I_d$ | single cluster at the origin |

Row 2 is the one this project registers a prediction on (**P-T1**, tested in Phase 2d): a real,
simple, positive top eigenvalue predicts concentration on three parallel hyperplanes normal to
$\varphi_1$ — i.e. **trimodality of $\langle\varphi_1, x_i\rangle$.** Note the row has *two*
conditions, one on $V$ and one on $QK$; the originally registered P-T1 carried only the first,
and the amendment adding $\varphi_1^\top M\varphi_1 > 0$ is on record in `PREDICTIONS.md`.

Note also what the paper says about all of this: **extending its clustering proofs to general
$(Q,K,V)$ is Problem 5, open.** Phase 2 is operating in open territory by construction.

---

## 2. What plays the role of $V$: the composed OV circuit

### 2.1 $W_V$ alone is the wrong object

Phase 1's `analyze_value_eigenspectrum` extracted $W_V$ by itself. For an individual head that is
a **non-square** $(d_{\text{head}}, d_{\text{model}})$ matrix, and **the eigendecomposition of a
non-square matrix does not exist.** Phase 2 corrects this.

The right object is the composed map from residual stream *to* residual stream:

$$
\mathrm{OV}_h = W_{V,h}\,W_{O,h}\ \in\ \mathbb R^{d\times d},
\qquad
\mathrm{OV} = \sum_{h} \mathrm{OV}_h
$$

This is the linear map the value pathway actually applies: attention decides *how much* of each
token to read; OV decides *what transformation* is applied when it is read. **$V$ in the theory
corresponds to $\mathrm{OV}$, not to $W_V$** — which also matches `math-1.md` §2.6.

`weights.py` uses the **row-vector convention** throughout ($x \mapsto x\,\mathrm{OV}$) and says
so; eigenvalues are convention-independent, but projector application is not, so the convention
has to be pinned once.

### 2.2 Four weight layouts, one formula

The composition is trivial mathematically and treacherous in code, because four architectures
store the same matrices four ways. The module handles each explicitly rather than guessing:

| family | storage | per-head slice | $\mathrm{OV}_h$ |
|---|---|---|---|
| ALBERT / BERT | `nn.Linear`, $(d_{\text{out}}, d_{\text{in}})$ | $W_V[s{:}e,\ :]$, $W_O[:,\ s{:}e]$ | $W_{V,h}^\top W_{O,h}^\top$ |
| GPT-2 | `Conv1D`, $(d_{\text{in}}, d_{\text{out}})$ | $W_V[:,\ s{:}e]$, $W_O[s{:}e,\ :]$ | $W_{V,h}\,W_{O,h}$ |
| GPT-NeoX / Pythia | fused `query_key_value`, **per-head interleaved** | split first, then ALBERT-style | $W_{V,h}^\top W_{O,h}^\top$ |

The GPT-NeoX row carries two traps, both documented in code because both have bitten:

1. **$V$ has no module of its own** — it is the third block of the fused QKV Linear, and the
   layout is `[Q_h|K_h|V_h]` contiguous *per head*, not per projection (`math-1.md` §2.3). A
   naive `weight[2d:, :]` slice silently scrambles heads. `core.pythia_weights.split_qkv_gptneox`
   handles it and is oracle-tested.
2. **The recovered $V$ and `attention.dense` are both `nn.Linear`**, so the per-head formula is
   ALBERT's, *not* GPT-2's — verified against a brute-force forward simulation in
   `tests/test_phase2_weights_gptneox.py`.

Two structural facts follow from the table that shape the whole phase: **ALBERT shares one OV
across all layers** (`is_per_layer=False`, so depth is a free variable and the map is iterated),
while GPT-2/BERT/Pythia have a different OV per layer. That single difference is the origin of
the two regimes in §7.

### 2.3 The QK side, for completeness

`extract_qk_spectrum` computes, per head, the largest singular value of $Q_hK_h^\top$ — read as
"effective $\beta$": larger spectral norm means sharper attention, stronger attraction. Note this
is a *proxy* for $\beta$, and a cruder one than Phase 1c's regression estimate (`math-1.md` §3.4)
— it is a bound on the logit magnitude, not a fit to the realized softmax. Both exist; they are
not the same number and should not be compared without saying which is which.

`extract_qk_per_head` returns $W_Q, W_K$ in a **canonical $(d_{\text{model}}, d_{\text{head}})$
orientation**, chosen so the attention bilinear is $x_i^\top(W_{Q,h}W_{K,h}^\top)x_j = x_i^\top M
x_j$ downstream. The docstring's instruction — *"Always orient WQ/WK consistently here; never
push the orientation question downstream"* — is the same discipline as the frame ledger
(`math-1.md` §3.2), applied to matrix orientation instead of activation frames.

---

## 3. Decomposing OV: two methods, and what their disagreement measures

### 3.1 Why not just `eig`?

Because OV is **not symmetric and generally not normal**, and for a non-normal matrix the
eigenvectors are not orthogonal — they can be arbitrarily ill-conditioned. Projecting onto a
"subspace spanned by eigenvectors with $\mathrm{Re}\,\lambda<0$" using a non-orthogonal basis
gives a projector that is not orthogonal, so "the fraction of displacement in the repulsive
subspace" stops being a fraction. Two independent decompositions are therefore computed.

### 3.2 Method 1 — ordered real Schur

$$
\mathrm{OV} = Z\,T\,Z^\top,\qquad Z\ \text{orthogonal},\quad T\ \text{upper quasi-triangular}
$$

with `sort='rhp'` placing eigenvalues with $\mathrm{Re}\,\lambda>0$ in the leading block, of size
$\texttt{sdim} = n_{\text{attract}}$. The key property: **the leading $n_a$ Schur vectors span an
invariant subspace of OV and are orthonormal *regardless of OV's normality*.** So

$$
P^{\text{Schur}}_{\text{attract}} = Z_{+}Z_{+}^\top,
\qquad
P^{\text{Schur}}_{\text{repulse}} = Z_{-}Z_{-}^\top
$$

are genuine orthogonal projectors onto complementary invariant subspaces. This is exactly the
right tool, and it is the reason the module uses `scipy.linalg.schur` rather than `eig`.

*(Caveat worth knowing: the Schur decomposition is not unique, and the two blocks are invariant
but the decomposition is not a direct sum unless OV is normal — the off-diagonal block of $T$
couples them. So "displacement in the repulsive subspace" is well-defined as a projection, but
the two subspaces are not dynamically independent. The `schur_cond` field — the ratio of block
norms — is a rough flag for degeneracy but is not a condition number and does not measure this
coupling.)*

### 3.3 Method 2 — the symmetric part

$$
S = \tfrac12(\mathrm{OV} + \mathrm{OV}^\top),\qquad S = U\Lambda U^\top
$$

Always symmetric, so eigenvectors are orthogonal and everything is numerically clean — but **it
discards the antisymmetric component entirely.** The antisymmetric part $\frac12(\mathrm{OV} -
\mathrm{OV}^\top)$ generates *rotations*, which move tokens without attracting or repelling them.

### 3.4 The disagreement is itself the measurement

`agree` is `True` when the two methods' sign fractions match within 10%. **When they disagree,
rotation matters** — i.e. OV's action at that layer is substantially not captured by any
attract/repel dichotomy, because a rotational component has no sign.

That is the entire premise of **Phase 2b**, which takes the antisymmetric/complex-eigenvalue
structure as its object rather than as a nuisance. The `frac_complex` field
($|\mathrm{Im}\,\lambda| > 0.01(|\mathrm{Re}\,\lambda| + \epsilon)$) is the direct measure.

### 3.5 Spectral norm vs spectral radius — a fixed bug worth understanding

The summary previously computed the OV "spectral norm" as $\sigma_{\max}(\mathrm{diag}|\lambda_i|)
= \max_i|\lambda_i|$ — which is the **spectral radius**, and equals the spectral norm *only when
OV is normal.* For non-normal matrices (precisely the GPT-2 layers with large rotational
components) they differ substantially, and always in one direction: $\rho(A) \le \lVert A\rVert_2$.

The fix computes $\sigma_{\max}(\mathrm{OV})$ directly and **keeps both**, so a reader can see
how far the two diverge per layer. That gap is a free non-normality diagnostic sitting in the
artifact, and it is the same quantity §3.4's `agree` flag is coarsely testing.

---

## 4. The three causal tests, and why three

A single test cannot distinguish *"V is locally detectable"* from *"V is globally causal but the
local signal is masked."* Three tests target different failure modes of that ambiguity.

### 4.1 Test 1 — displacement projection (local, direct)

At each layer transition, take $\Delta x = x_{\ell+1} - x_\ell$ and split its energy by subspace:

$$
\texttt{repulse\_disp\_frac}(\ell) = \frac{\lVert \Delta x\,P_{\text{repulse}}\rVert_F^2}
{\lVert \Delta x\,P_{\text{attract}}\rVert_F^2 + \lVert\Delta x\,P_{\text{repulse}}\rVert_F^2}
$$

**The prediction: at violation layers, this fraction is elevated.** This is the most direct
possible evidence, and `V_repulsive_local` (>50% of violation layers flagged) is accordingly
given precedence over every rescaling-based verdict — *direct evidence outranks indirect evidence
when both are available.*

The function also accepts `cluster_labels` + `population`, so the same projection can be
restricted to clustered or unclustered tokens (Phase 5c's object).

**Companion: the self-interaction trajectory.** $s_i = x_i\,\mathrm{OV}\,x_i^\top$ per token per
layer — positive means OV is locally attractive *for that token*, negative repulsive. This is a
finer-grained instrument than the global fraction (it is the diagonal of the quadratic form
rather than a subspace count), and `frac_negative` per layer is what produces the headline
"ALBERT: 100% negative self-interaction."

### 4.2 Test 2 — the rescaled frame (global, interventional)

Apply §1.3's coordinate change and re-run Phase 1's metrics:

$$
z(\ell) = x(\ell)\,\big((e^{-\mathrm{OV}})^{\ell}\big)^\top
$$

**The logic is causal, not correlational.** If the violations are produced by OV's own dynamics,
then factoring OV out of the coordinates should *eliminate them*. If they persist in the rescaled
frame, they were not OV's doing. `rescaled_improvement` counts violations removed, and
`rescaled_frac` = improvement / violations is the strongest single term in the V-score.

This catches the case the displacement test misses: **V causal but distributed**, where no single
layer's displacement is locally dominated by the repulsive subspace, yet the accumulated effect
is exactly OV's.

Two implementation notes that carry meaning:

- $R = e^{-\mathrm{OV}}$ is applied **incrementally** ($R^\ell$ built by repeated multiplication)
  rather than by exponentiating $-\ell\,\mathrm{OV}$, and the code watches for overflow: **if OV
  has negative eigenvalues then $R$ has eigenvalues $>1$ and $R^\ell$ diverges.** When it does,
  the rescaled trajectory is truncated at `max_valid_layer` rather than silently returning `inf`.
- The rescaled frame is where "is metastability *sharper* in OV's coordinates?" becomes askable.
  A yes would say OV is the right coordinate system for the whole phenomenon.

**§10 open question 2 argues this test is applied at the wrong time scale, and that the overflow
above is the symptom.**

### 4.3 Test 3 — attention/FFN decomposition (channel)

Splits the residual update into $\Delta x = \Delta_{\text{attn}} + \Delta_{\text{ffn}}$ and asks
which pathway carries the energy drop. This answers a different question from the first two: not
*whether* V is causal, but *how its effect reaches the trajectory.*

The finding that motivated the whole regime split: on GPT-2-small/medium the local displacement
test **fails** while the rescaled frame **succeeds**, and the FFN is the proximal dropper —
`frac_ffn_amplifies_repulsive` measures whether the FFN's update pushes into OV's repulsive
subspace. So the FFN is the *channel* and V is the *distal cause*.

**The additive split is not always sufficient.** For ALBERT-xlarge on several prompts, most of the
energy change lives in the **cross-term**

$$
\Delta_{\text{cross}} = E(x + \Delta_a + \Delta_f) - E(x + \Delta_a) - E(x + \Delta_f) + E(x)
$$

which is exactly the failure of additivity — expected, since $E_\beta$ is an exponential of an
inner product and therefore not linear in the displacement. `cross_term_analysis.py` localizes it
per token pair via

$$
C_{ij} = \Delta a_i\cdot\Delta f_j + \Delta f_i\cdot\Delta a_j
$$

(positive = the attention and FFN updates on the two tokens are aligned), and then tests
**Jaccard overlap of the top-$|C_{ij}|$ pairs against Phase 1's recorded `energy_drop_pairs`** —
tying the cross-term mechanism back to the specific token pairs Phase 1 flagged. That
cross-referencing is what makes it a mechanism claim rather than a decomposition exercise.

---

## 5. The V-score and the verdict classifier

### 5.1 A continuous score, because the two regimes need a common scale

$$
\texttt{v\_score} = 0.40\,r_{\text{resc}} + 0.25\,r_{\text{disp}} + 0.20\,r_{\text{ffn}}
\;-\; 0.15\,\big|\rho_{\text{partial}}\big|
$$

with $r_{\text{resc}}$ = fraction of violations removed by rescaling, $r_{\text{disp}}$ = fraction
of violation layers where the displacement test fires, $r_{\text{ffn}}$ = fraction where the FFN
amplifies into the repulsive subspace, and $\rho_{\text{partial}}$ the confound correlation of
§6. Range $[-0.15,\ 0.85]$ — note the three positive weights sum to $0.40+0.25+0.20 = 0.85$, so
the stated upper bound of $1.0$ (reproduced from `verdict_v2.py`'s own comment) is unattainable.
A score is being read against a ceiling it cannot reach.

The weights are described as "theory-motivated": strongest evidence (global intervention) →
direct local evidence → confirmatory channel evidence → penalty for a known confound. That
ordering is defensible. **The numbers are not derived and the score has no null distribution**
(§10, open question 3).

### 5.2 The categorical classifier

`analysis_extended._classify` is an if/elif chain — deliberately, so precedence is explicit:

```
n_violations == 0                                     → no_violations
frac_overshoot   > 0.5                                → overshoot_dominant
frac_repulsive   > 0.5                                → V_repulsive_local
rescaled_frac > 0.8  and  ffn_frac_drop  > 0.5        → V_repulsive_via_FFN
rescaled_frac > 0.8  and  ffn_frac_drop <= 0.5        → V_repulsive_via_attn
ffn_frac_drop > 0.5  and  rescaled_frac < 0.2
                     and  n_decomposed >= 3           → FFN_independent
otherwise                                             → mixed_or_unattributed
```

with an upgrade to `V_repulsive_via_FFN_confirmed` when the FFN-amplification fraction exceeds
0.5 **and** `channel == "FFN"` explicitly (the check was previously `!= "attention"`, which let
`"unknown"` through — a small fix with a real lesson: *a negated equality is not a
classification*).

`overshoot_dominant` is the alternative hypothesis being ruled out: violations caused by the
update simply being too large (a discretization artifact) rather than by OV's structure. It fires
in **0 of 35** runs, which is a genuine null and worth more than it is currently given.

**Note `FFN_independent` guards on `n_decomposed`, not `n_violations`** — the number of layers
where the decomposition actually ran, not where a violation occurred. Guarding on the wrong
denominator would let a verdict be assigned from one or two decomposed layers.

---

## 6. The confound, and how it is handled

Early- and final-layer OV **spectral norm spikes** (up to 22× the mean at GPT-2-small L11 —
plausibly the unembedding projection) produce large displacements that can be misattributed to
the repulsive subspace. Big displacement, any direction, looks like evidence.

The instrument is a **partial Spearman correlation**: rank-correlate OV norm against the
violation indicator, having residualized *both* on `rep_frac` via rank regression. If the partial
$\rho$ is large, OV norm predicts violations *beyond* what the repulsive fraction explains, and
the local verdict is suspect.

Measured: partial $\rho$ down to $-0.71$ on most GPT-2 models — **substantial**. The consequence
is stated as a policy: **the rescaled-frame result is immune to this confound and the
`V_repulsive_local` verdict is vulnerable**, so where the two disagree the rescaled frame is the
more trustworthy signal. That is why the V-score weights the rescaled term highest and subtracts
the confound explicitly.

---

## 7. The two regimes

| | Regime A — locally detectable | Regime B — globally coherent, FFN-mediated |
|---|---|---|
| models | ALBERT-xlarge, GPT-2-xl, GPT-2-large (partial) | GPT-2-small, GPT-2-medium |
| displacement test | passes | fails |
| rescaled frame | passes | passes |
| proximal channel | attention | FFN |
| architectural reading | shared weights ⟹ the OV circuit acts directly, with no layer-specific FFN amplification available — *there is nothing else to route through* | per-layer weights ⟹ the FFN can and does push into OV's repulsive subspace |

Verdict distribution over 35 runs: `V_repulsive_local` 13, `V_repulsive_via_FFN` 8,
`V_repulsive_via_FFN_confirmed` 3, `FFN_independent` 1, `mixed_or_unattributed` 10,
`overshoot_dominant` 0, `V_repulsive_via_attn` 0.

**The regime split is the phase's most consequential output**, because it becomes the organizing
frame for model selection downstream — Phase 3 selects exactly ALBERT-xlarge (A) and GPT-2-large
(B) rather than re-running the full grid, on the grounds that these are the two meaningfully
different conditions.

Three honest weaknesses, all recorded: GPT-2-large has borderline runs (v-scores 0.455–0.486,
neither test passing cleanly — possibly genuine regime-boundary cases); BERT and ALBERT-base sit
below reliable detection threshold and supply most of the `mixed_or_unattributed` bucket; and
**ALBERT-base's channel defaults to "attention" by construction** — shared weights mean there is
no per-layer decompose path, so the FFN question is unresolvable for that model rather than
answered.

---

## 8. Pythia: why the port is an upgrade

`decompose.py`, `ffn_subspace.py`, and their consumers are **frozen GPT-2-only**, and correctly
so: the decomposition relies on a real post-attention, pre-FFN intermediate state, which exists
only because GPT-2/ALBERT/BERT compute the FFN *after* adding attention's output to the residual.
Pythia has no such intermediate (`math-1.md` §2.1).

But the parallel residual is not an obstacle — it is the cleaner instrument:

$$
\Delta x = \mathrm{attn\_out} + \mathrm{ffn\_out}\quad\text{exactly, from the same input}
$$

**no ordering confound, exactly additive.** Under GPT-2's sequential form the FFN reads a state
attention has already modified, so "how much of this update was attention" has no
frame-independent answer, and the additive split is an approximation whose error is precisely the
cross-term of §4.3. On Pythia the split the theory wants is available exactly, for the first time
in this project. The plan is a new parallel-residual module in `core/` — not in this phase's
directory — that re-enables the attn-vs-FFN energy panels and reopens the FFN-vs-V question
natively.

---

## 9. Code map

| File | Role |
|---|---|
| `weights.py` | Pure weight decomposition: composed OV per head and total, Schur + symmetric eigendecomposition, subspace projectors, QK spectral norms, canonical per-head $W_Q/W_K$, $e^{-\mathrm{OV}}$, persistence. **No inference**, so Phases 2b/2c/6 reuse it without re-deriving projectors |
| `trajectory.py` | Offline analysis on saved Phase 1 activations: step sizes (overshoot test), subspace activation, self-interaction, displacement projection, centroid projection, rescaled trajectory, Phase 1 event loading |
| `trajectory_perlayer.py` | Split from `trajectory.py` because ALBERT has one shared V and GPT-2/BERT need each layer's own — keeping them separate avoids one function branching on architecture internally |
| `decompose.py` | The GPT-2-only forward-pass attn/FFN split. **Frozen against Pythia's parallel residual** |
| `cross_term_analysis.py` | Three-way (attn, FFN, cross) decomposition; per-pair $C_{ij}$; Jaccard against Phase 1 drop pairs. Kept separate from `decompose.py` because it is a genuinely different decomposition, not a refinement |
| `analysis_extended.py` | **Authoritative verdict classifier** (`_classify`), the OV-norm confound analysis |
| `verdict_v2.py` | `build_v_score`; its own `_classify` is a back-compat shim only, not a second authority |
| `subresult.py` / `subexperiments.py` / `subexp_wrappers.py` | Registry pattern: each analysis reports through a typed `SubResult` contract, so the verdict assembler needs to know no module's raw output shape. This is also what lets `--full`/`--offline` skip subexperiments cleanly and lets one subexperiment error without aborting the run |
| `head_ablation.py` / `threshold_analysis.py` | Per-head causal and $\beta$-sweep sensitivity checks layered on top of the verdict, not required for it |
| `ffn_subspace.py`, `layer_v_events.py`, `head_ov_analysis.py`, `lens_band.py`, `vocab_projection.py` | Supporting analyses |

---

## 10. Open questions

Tracked in `status-2.md`: the GPT-2-large borderline runs; the OV spectral-norm confound; and
ALBERT-base's unresolvable FFN channel.

Surfaced by writing this document:

1. **`frac_repulsive` counts eigenvalues; it does not weight them.** A direction with
   $\mathrm{Re}\,\lambda = -100$ and one with $\mathrm{Re}\,\lambda = -10^{-6}$ contribute
   equally. Since §1.1's sign argument is really about the *magnitude* of the energy
   contribution, the natural quantity is a mass-weighted fraction such as $\sum_{\lambda:\
   \mathrm{Re}<0}|\mathrm{Re}\,\lambda| \big/ \sum_i|\mathrm{Re}\,\lambda_i|$. Both are one line
   from arrays already persisted (`eig_real` is in every `ov_decomp_*.npz`), and if they diverge
   the verdict table's central column is measuring the wrong thing. **This is the same defect
   class as Phase 1's raw-vs-normed rank** — a count standing in for a magnitude.

2. **The rescaled frame is almost certainly applied at the wrong time scale, and the overflow
   guard is the symptom.** The code applies $(e^{-\mathrm{OV}})^{\ell}$ — i.e. it identifies one
   layer with one unit of ODE time. Phase 1c establishes that this is exactly wrong: the true
   Euler step is $h_\ell = \lVert P^\perp\Delta x\rVert/(\lVert x\rVert\lVert\mathcal X\rVert)$
   and $T_{\rm eff} = \sum h_\ell$ is expected to be **much smaller than $\ell$**
   (`math-1c.md` §1.2, §3.5). The frame-correct rescaling is $e^{-T_{\rm eff}(\ell)\,\mathrm{OV}}$,
   not $e^{-\ell\,\mathrm{OV}}$.

   The consequence is not subtle. Over-rescaling by a factor of $\ell/T_{\rm eff}$ inflates the
   exponent by the same factor, and `rescaled_trajectory`'s own comment records the effect: *"R^L
   can overflow when OV has negative eigenvalues"*, with a truncation at `max_valid_layer`. **A
   correctly-scaled rescaling should not overflow** — $e^{-T_{\rm eff}\mathrm{OV}}$ with
   $T_{\rm eff}\sim O(1)$ is a bounded operator. So the divergence is evidence of the mis-scaling
   rather than a numerical hazard to be guarded against.

   Since `rescaled_frac` is the **highest-weighted term in the V-score** (0.40) and the deciding
   term in four of the seven verdict branches, this is the single most consequential open item in
   the phase. It is also cheap to resolve: Phase 1c already computes $T_{\rm eff}(\ell)$ per
   layer, at report-only cost, from artifacts on disk.

3. **The V-score has no null distribution and its weights are placed, not calibrated.** What does
   `v_score = 0.46` mean? Currently: "borderline, between the empirical clusters." There is no
   answer to "what score would a random OV of matched spectrum produce on the same
   activations?" — and `core/nulls.py` exists precisely to answer that class of question
   (`math-1b.md` §6). A shuffled-eigenvalue or random-orthogonal-OV null would turn the borderline
   GPT-2-large runs from an interpretive puzzle into a measurement. This is the third independent
   instance of the project's recurring lesson: **a threshold not derived from a null is not a
   threshold.**

4. **`V_repulsive_via_attn` may be structurally unreachable, exactly as Phase 1b's
   `strong_bipartition` was.** It fires only when `rescaled_frac > 0.8` **and**
   `ffn_frac_drop <= 0.5` — but the classifier is an if/elif chain, so `V_repulsive_local`
   (`frac_repulsive > 0.5`) preempts it. The attention-mediated regime is precisely the one where
   the local displacement test *passes* (Regime A, §7), so those runs are caught by the earlier
   branch every time. For the branch to fire, a model would have to be attention-mediated *and*
   locally undetectable — a combination the regime table does not contain.

   The design doc keeps the branch on the grounds that "a future model or prompt type could
   exercise it, and removing an unused-but-reachable branch would silently narrow the
   classifier's coverage." That is a good principle. But `math-1b.md` §4.4 shows the failure mode
   of *not* checking reachability: 1b reported a 0% rate as a finding when the test was
   near-unreachable by construction. **0/35 for `V_repulsive_via_attn` should be labelled
   "unreachable given branch precedence," not read as an empirical null** — and the fix is one
   line: evaluate the branch conditions independently of precedence and record which ones *would*
   have fired.

5. **The self-interaction $x^\top\mathrm{OV}\,x$ and the Schur projection measure different
   things, and only the first is basis-free.** $s_i$ is the diagonal of the quadratic form —
   invariant, interpretable, per token. The projection fraction depends on the subspace
   *decomposition*, which §3.2 notes is not a direct sum for non-normal OV. Where the two
   disagree, the non-normality is doing the work, and that is measurable with quantities already
   persisted (`schur_cond`, `frac_complex`, and the spectral-norm/radius gap of §3.5). Nothing
   currently crosses them.

6. **The Gram-matrix connection to $E_\beta$ is never used here, though it would sharpen the
   attribution.** `math-1.md` §5.2 shows $\Delta E_\beta$ decomposes exactly into a common-mode
   ($\kappa_1$) and a spread ($\kappa_2$) term. Phase 2 asks *which subspace* the displacement
   went into; the cumulant split asks *which kind of energy change resulted*. Crossing them —
   does repulsive-subspace displacement produce $\kappa_2$ change specifically, as the mechanism
   predicts, rather than $\kappa_1$ drift? — would convert a correlation between two derived
   quantities into a mechanism test. Both quantities exist; they have never met.

7. **What does the OV eigenspectrum do over training?** Every result in this phase is from
   finished networks. Phase 1's Pythia pilot showed that the energy break is *dated* (onset step
   256, saturating by 512, severity peaking at 60k then declining). The obvious question — does
   $V$'s repulsive fraction track that curve? — is [W]-cost (weights only, no forward passes) and
   is the natural Phase 2 counterpart to Phase 1's developmental arc. **The severity decline
   after step 60k with a flat violation count is precisely a magnitude-vs-count question**, which
   open question 1's mass-weighted fraction is built to answer.
