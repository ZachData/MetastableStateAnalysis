# Phase 1c — MATH (study notes)

## 0. What this document is

Companion to `math-1.md` and `math-1b.md`. **Read `math-1.md` §1–3 first** — this phase is the
place where the dictionary built there stops being a dictionary and becomes arithmetic.

Phase 1c is the mathematical centre of the project. Every other phase measures *something about
the model*; 1c measures **the model against the theory**, quantitatively, layer by layer. It is
also entirely re-analysis — Phase 1 artifacts already on disk plus small weight-only
computations, no forward passes — which makes the fact that it has not been run the single
largest outstanding gap in the project.

Its target is one sentence in Blog 1: *"the trained network resists the collapse its
architecture drives."* That claim silently compares the observed state against $t = \infty$.
Phase 1c replaces $\infty$ with a number.

Six sub-experiments, each with a registered prediction:

| Sub-exp | Object | Prediction | Cost |
|---|---|---|---|
| **A** | effective integration time $T_{\rm eff}$ vs collapse time $t^\ast$ | **P-γ2** | [R] |
| **B** | residual of `ip_mean` against $\gamma_{\beta_{\rm eff}}(T_{\rm eff})$ | **P-γ1** | [R+W] |
| **C** | cumulant ladder + the rank reconciliation | — (settles status-1 D1/D10) | [R] |
| **D** | four measurement frames; is the sphere licensed? | — | [W] |
| **E** | hemisphere/cone feasibility and its margin | **P-H1** | [R] |
| **F** | spherical designs — is the trained configuration *sharp*? | **P-S1** | [R] |

---

## 1. The construction: a residual block is a forward-Euler step

### 1.1 Recap and sharpening

From `math-1.md` §1.2: add-then-normalize agrees with tangent-plane motion to first order, so

$$
x_{\ell+1} = x_\ell + h_\ell\,\mathcal X(x_\ell) \quad\Longrightarrow\quad
h_\ell = \frac{\lVert \Delta x_\ell^{\text{tangential}}\rVert}{\lVert \mathcal X(x_\ell)\rVert},
\qquad
T_{\rm eff} = \sum_\ell h_\ell
$$

where $\mathcal X$ is the paper's vector field
$\mathcal X(x_i) = P^\perp_{x_i}\big(\sum_j a_{ij}x_j\big)$.

Three objects follow, and only the third is the deliverable:

1. **step size** $h_\ell$ — how far one block moves the state, in the ODE's own time units;
2. **effective integration time** $T_{\rm eff}$ — compared against $t^\ast$ (P-γ2);
3. **residual** ${\rm ip\_mean}(\ell) - \gamma_{\beta_{\rm eff}}(T_{\rm eff}(\ell))$ — the part
   of the layer-wise trajectory *not* explained by identity-weight dynamics run for the observed
   amount of time (P-γ1).

> The residual is the deliverable. The fit is not.

### 1.2 The step-size definition decides the answer — and the obvious one is wrong by ~6×

`MATH.md` §8 defines $h_\ell = \lVert P^\perp(\Delta x_\ell)\rVert / \lVert x_\ell\rVert$. That
is the **displacement on the sphere** — the *numerator* of the Euler step. It equals the step
size only if $\lVert\mathcal X\rVert = 1$.

The paper's own bound is $\lVert\mathcal X\rVert \le 1$ (softmax rows sum to 1, $\lVert
x_j\rVert = 1$, and the tangent projection is a contraction), with equality **only for a fully
collapsed cloud**. For a spread cloud the field is far weaker, because the $a_{ij}$-weighted
average of many near-orthogonal directions has small norm. So `h_displacement` systematically
**understates** $T_{\rm eff}$ by exactly the factor by which the field falls short of its bound.

Measured directly, by integrating the true (SA) field forward at a known step from an orthogonal
init ($n = 40$, $d = 512$, $\beta = 1$):

| quantity | value |
|---|---|
| injected step $h$ | 0.0200 |
| recovered `h_calibrated` | **0.0200** |
| `h_displacement` (§8 as written) | 0.0035 |
| mean $\lVert\mathcal X\rVert$ over the trajectory | 0.176 |
| **understatement factor** | **5.67×** |

And the recovered time tracks the ODE: at $\gamma = 0.9$, $T_{\rm eff} = 3.040$ against the
ODE's $t^\ast = 3.015$ — 0.8%, which is Euler discretization error and not a modelling gap.

**Why this is the expensive error and not a detail.** P-γ2 predicts $T_{\rm eff}\ll t^\ast$. The
§8 definition makes that prediction nearly true *by construction*, in the direction that would
have us conclude our own headline result is an artifact of depth. Adopting it would be
confirming a prediction by choice of units. Hence: **all three definitions are computed**, and
`verdict()` returns `robust=False` when they straddle $t^\ast$ — in which case the answer is a
definition, not a measurement, and must be reported that way.

$$
\begin{aligned}
h_{\text{displacement}} &= \lVert P^\perp\Delta x\rVert/\lVert x\rVert
&&\text{(§8 as written; an underestimate)}\\
h_{\text{calibrated}} &= h_{\text{displacement}}/\lVert\mathcal X(x_\ell)\rVert
&&\text{(the actual Euler step)}\\
h_{\text{attn-only}} &= \text{as calibrated, with }\Delta x\text{ restricted to the attention branch}
&&\text{(the frame-correct one)}
\end{aligned}
$$

### 1.3 Why `h_attn_only` is the frame-correct variant

The paper writes the feed-forward layer down in §2 and then **excludes it**; every theorem in
Parts 1–2 is single-head, no-FFN. Using the full block delta credits the ODE with motion
produced by a term that is not in it — the MLP is a *token-wise force field*, not a particle
interaction at all (`math-1.md` §2.7).

Pythia's parallel residual makes the split exact:
$\Delta x = \mathrm{attn}(\mathrm{LN}_1 x) + \mathrm{mlp}(\mathrm{LN}_2 x)$, both branches
reading the same input, no ordering confound. **This is the one place GPT-2's sequential
architecture could not have supported the measurement at all** — under sequential residual the
FFN reads a state attention has already modified, so "the attention-only step" has no
frame-independent definition. The other two step definitions are upper bounds.

### 1.4 The calibrated step makes the null rate-invariant — a stronger result than planned

Damping the field by 0.3× produces a residual of $-0.0009$: essentially zero. That is **correct**
and it sharpens what the residual means. Damping is not resistance; it is *slower integration*,
and `h_calibrated` absorbs it into a shorter $T_{\rm eff}$.

So the residual does **not** measure how much of the identity-weight field the network applies.
It measures whether the network moves in a **different direction** from that field.

That is the better notion of resistance and it was not what the plan specified:

- A network that merely attenuates attention shows **zero** residual, and should — it is running
  the paper's dynamics, just fewer of them, and "depth" would be the correct explanation.
- A network with a genuinely different vector field shows a nonzero residual — verified at
  $-0.0113$ for a trajectory perturbed orthogonally to the field.

### 1.5 Implementation details that carry meaning

**Tangential displacement is computed projected, not as a unit-vector difference.**
$\lVert u_{\ell+1} - u_\ell\rVert$ and $\lVert P^\perp_{x_\ell}(\Delta x)\rVert/\lVert
x_\ell\rVert$ agree only to first order, and the projected form is the one that **isolates
motion along the sphere from residual-stream norm growth** — which is not motion on the sphere
at all, and which trained transformers do a great deal of (`math-1.md` §6.2).

**Per-token ratio, then averaged — not ratio of averages.** In `step_sizes`:

```python
safe   = np.where(mag < 1e-9, np.nan, mag)
h_cal  = np.nanmean(d_tan / safe)      # not  np.mean(d_tan) / np.mean(mag)
```

because a few sink tokens with tiny field magnitude would otherwise dominate the denominator of
a ratio-of-means and inflate the step. Same family of concern as the pos0 policy in
`math-1.md` §2.5.

**Causal vs non-causal field.** `sa_field(causal=)` exposes both. The paper's field is
non-causal — every particle interacts with every other — while a decoder-only transformer is
not. `causal=True` is the honest comparison for Pythia and is the default, but it *is* a
departure from the theory, and the masked field is systematically weaker (fewer terms, and early
tokens see almost no one). **Whichever is used for $T_{\rm eff}$ must be used for the null**, and
the choice is recorded in the artifact.

---

## 2. The $\gamma_\beta$ ODE: Theorems 6.8 and 6.9

### 2.1 Why one scalar describes $n$ particles

Theorem 6.8: for **pairwise-orthogonal** initial points, permutation equivariance forces all
pairwise angles to stay equal. The dynamics are equivariant under relabeling; the initial
configuration is invariant under it (all pairwise inner products are 0); so the whole
$n$-particle configuration remains describable by a single scalar

$$
\gamma(t) = \cos\theta(t) = \langle x_i(t), x_j(t)\rangle \quad \text{for all } i \ne j
$$

obeying, with $b = \beta$,

$$
\textbf{(SA), eq. (6.9)}\qquad
\dot\gamma = \frac{2\,e^{b\gamma}(1-\gamma)\big((n-1)\gamma + 1\big)}{e^{b} + (n-1)e^{b\gamma}}
$$

$$
\textbf{(USA), eq. (6.10)}\qquad
\dot\gamma = \frac{2}{n}\,e^{b\gamma}(1-\gamma)\big((n-1)\gamma + 1\big)
$$

both from $\gamma(0) = 0$. Note the structure: the $(1-\gamma)$ factor is what makes $\gamma=1$
(full collapse) an equilibrium and bounds the solution; the $((n-1)\gamma+1)$ factor vanishes at
$\gamma = -1/(n-1)$, the simplex configuration, the *other* equilibrium; and the denominator in
(SA) is the partition function, which is the entire difference between the two models.

**Theorem 6.9 is what makes this usable on real data**, and its exact form matters more than the
qualitative statement. For $\beta\ge0$, $n\ge2$ there is $d^\ast(n,\beta)\ge n$ such that for all
$d\ge d^\ast$, with probability at least $1 - 2n^2 d^{-1/64}$, **for all $i\ne j$ and all
$t\ge0$**:

$$
\Big|\langle x_i(t),x_j(t)\rangle - \gamma_\beta(t)\Big| \;\le\;
\min\left\{\ \underbrace{2\,c(\beta)^{nt}\sqrt{\tfrac{\log d}{d}}}_{\text{(i) stability of the flow}},
\quad \underbrace{C e^{-\lambda t}}_{\text{(ii) both have collapsed}}\ \right\},
\qquad c(\beta) = e^{10\max\{1,\beta\}}
$$

The proof is a two-part argument: (i) the flow map is Lipschitz in its initial data with constant
$c(\beta)^{nt}$, and by concentration of measure $n$ random points in high $d$ are within
$\sqrt{\log d/d}$ of being exactly orthogonal — so the real trajectory tracks the orthogonal-init
trajectory; (ii) independently, Lemma 6.4 forces both to approach 1 exponentially.

**Two things follow that bear directly on this phase, and neither is in the repo's docs.**

**Branch (i) degrades exponentially in $t$.** $c(\beta)^{nt}$ grows; at $\beta = 1$, $n = 467$
that is $e^{10\cdot467\,t}$ — the bound is vacuous almost immediately. Branch (ii) improves in
$t$. So the guarantee is strong at small $t$, strong at large $t$, and **weakest where the two
branches cross, in the middle of the trajectory.** That is *exactly* where metastable plateaus
would live. The theorem that says "everything concentrates on one curve, leaving no room for a
multi-cluster intermediate state" therefore has its weakest grip precisely on the intermediate
states. This does not rescue the $d = 1024$ tension of `math-1.md` §1.6 — the numerics of Fig. 3
are a separate line of evidence — but it does mean **the reading "concentration forbids
metastability at high $d$" is not a theorem, and the residual is measuring something the theory
genuinely does not pin down.** That is a better justification for running B than the one
currently written down.

**$d^\ast(n,\beta)$ is not computed anywhere.** The theorem is "there exists $d^\ast$"; the
proof's explicit requirement is $\frac{d}{\log d}\ge 16\,c(\beta)^2/\gamma_\beta(1/n)^2$. At
$\beta = 1$, $c(\beta) = e^{10}\approx 2.2\times10^4$, so $c^2\approx4.8\times10^8$ and — with
$\gamma_\beta(1/n)$ small for large $n$ — the requirement is astronomically beyond $d = 1024$.
**So we are not in Theorem 6.9's regime by its own sufficient condition, even though $d > n$.**
The constants are certainly loose (they come from Grönwall), but the honest statement is that
6.9's applicability here is an *assumption*, not a checked hypothesis, and the phase should say
so where it currently says "we sit in the regime where 6.9 applies."

This is also (per `math-1.md` §1.5–1.6) the theorem whose concentration argument is the paper's
own explanation for why metastability should *disappear* at high $d$ — so the comparison here is
doing double duty.

**A third hypothesis violation, from `math-1.md` §2.4.** Theorem 6.8's single-scalar reduction
is *derived from permutation equivariance* (Remark 2.2): permuting the labels of an orthogonal
initial configuration leaves it invariant, so all pairwise angles must stay equal. RoPE puts
position inside the coupling and the causal mask restricts it, and **both break permutation
equivariance.** So on Pythia the reduction's premise fails twice over. The null remains the right
null — it is the identity-weight dynamics, which is the comparison we want — but "the model
violates the hypotheses" is a longer list than orthogonality alone.

**The standing hypotheses, all of which our models violate:** $Q^\top K = V = I$, single head,
no FFN, orthogonal initialization. The point is not that the model should obey them. It is that
**the identity-weight dynamics running for the observed amount of time is the correct null**,
and "the trained network resists collapse" is only meaningful as a statement about the residual
against it.

### 2.2 Numerics that matter

**Overflow-safe factoring.** The naive form of (6.9) overflows float64 at $\beta = 5$ with
$\gamma$ near 1 and large $n$ — which is one of the corners the validation table covers.
Numerator and denominator both carry $e^{b\gamma}$ and $e^{b}$, so the code factors out
$e^{\max(b\gamma,\,b)}$ first:

```python
m   = max(b*g, b)
num = 2*exp(b*g - m) * (1-g) * ((n-1)*g + 1)
den = exp(b - m) + (n-1)*exp(b*g - m)
```

**RK4 on a fixed step, with the step validated rather than assumed.** An adaptive solver is not
worth the dependency: the RHS is smooth, monotone and bounded on $[0,1]$, and the solution is a
single sigmoid. `integrate_gamma_converged` halves $dt$ until $t(\gamma{=}0.9)$ stops moving by
more than `rtol`, and **a run that never converged reports `converged=False` rather than
silently returning the coarsest answer.**

**Clipping is a numerical guard, not a physical constraint.** $\gamma$ cannot exceed 1
analytically — the $(1-\gamma)$ factor kills the RHS there — but a fixed-step method can
overshoot in the last few steps of a stiff high-$\beta$ run, and an overshoot past 1 makes
$(1-\gamma)$ *negative* and sends the solution to $-\infty$. Hence the clip to $[g_0, 1-10^{-15}]$.

**`time_to_threshold` returns `inf`, not `t_max`, when the target is never reached.** That is a
real outcome at small $\beta$ and must not be silently reported as the grid edge.

**`gamma_at` clamps rather than extrapolates** beyond `t_max`, because $\gamma$ is asymptotic to
1 and "naive extrapolation of a saturating curve is the kind of error that looks like a result."

### 2.3 Validation, and two facts that fall out of the collapse-time table

`gamma_ode.py` reproduces **all 28 entries** of `MATH.md` §3.2's collapse-time table — both (SA)
and (USA), $n\in\{20,467\}$, $\beta\in\{0.1,1,2,5\}$, at $\gamma = 0.5$ and $0.9$ — to a maximum
absolute deviation of **0.005**.

Two facts from that table govern how everything else reads:

1. **Collapse time is short and nearly $\beta$-independent.** At $n = 467$, reaching
   $\gamma = 0.9$ takes $t^\ast \approx 4.2$ under (SA), essentially unchanged across two decades
   of $\beta$. So *"how much integration time would this network need in order to collapse"* has
   a **single answer**, and it does not depend on the $\beta$ sweep. This is what makes P-γ2 a
   clean prediction rather than a family of them.
2. **(SA) and (USA) separate at large $\beta$ and small $n$.** At $n = 20$, $\beta = 5$: $\approx
   8.3$ (SA) against $\approx 0.79$ (USA) — **a factor of ten**. Any claim reading the surrogate
   as a stand-in for the normalized dynamics is unsupported in exactly the corner where the
   paper's own metastability numerics sit (Figure 4 is $\beta = 4$ and $9$).

---

## 3. Sub-experiment B: the residual, and the sigmoid ceiling

### 3.1 The vertical residual and its sign convention

$$
\text{residual}(\ell) = {\rm ip\_mean}(\ell) - \gamma_{\beta_{\rm eff}(\ell)}\big(T_{\rm eff}(\ell)\big)
$$

| sign | reading |
|---|---|
| $< 0$ | the network is **behind** the identity-weight prediction — less clustered than pure attention would make it. **This is resistance, and its magnitude is now a number.** |
| $\approx 0$ | the network does what the null does. "Resistance" was the comparison against $t=\infty$, not a property of the weights. |
| $> 0$ | the network clusters **faster** than the null. |

The convention is written into the artifact as a string field, "because a residual whose sign
convention has to be recovered from source is a result waiting to be misread."

### 3.2 The ceiling problem, and the time residual that fixes it

$\gamma$ is a sigmoid asymptotic to 1, so once the null exceeds ~0.95 the *vertical* residual has
almost no dynamic range. On synthetic runs a perturbation that visibly altered the dynamics still
registered a final residual of $+0.0000$, purely because both curves were pinned against the
ceiling.

Inverting the null removes the compression:

$$
\boxed{\ \text{time\_residual}(\ell) \;=\; t_{\rm null}^{-1}\big({\rm ip\_mean}(\ell)\big) \;-\; T_{\rm eff}(\ell)\ }
$$

— *how much time the identity-weight dynamics would need to reach the clustering we observe*,
against *how much time the network actually spent*. Negative means the network spent longer than
the null needs: resistance.

The improvement is dramatic. On the same synthetic pair:

| | vertical residual | time residual |
|---|---|---|
| unperturbed | $+0.0001$ | $+0.002$ |
| perturbed orthogonally | $-0.0113$ | $-0.669$ |

**two and a half orders of magnitude more separation.** Both are reported: the vertical residual
is interpretable in units of inner product; the time residual is the one that stays honest late
in the stack, which is exactly where the trained-model question lives.

This is a nice general lesson about reading residuals against saturating nulls — *invert the
null and measure the horizontal gap* — and it is not specific to this project.

### 3.3 Sub-$g_0$ layers are `nan`, counted, and are the interesting ones

Layers whose observed `ip_mean` falls **below the null's starting point** are recorded as `nan`
and counted, not clipped. An observed value below $g_0$ means the network **de-clustered past its
own starting point** — the strongest possible resistance signal — and clipping it to zero would
silently render it as "on schedule."

These layers are *expected*: Phase 1 already found mid-network mass dropping a factor of 20 below
the embedding floor (`math-1.md` §13.2). Clipping would have erased the phase's most striking
observation.

### 3.4 Three assumptions, each with its own instrument

**(1) $\beta_{\rm eff}$ is not constant across layers.** The null is evaluated *per layer at that
layer's own $\beta$*, with the run median as fallback and **the fallback count recorded**. A
single global $\beta$ would fold the QK circuits' layer-wise variation into the residual and
attribute a property of the attention weights to "resistance."

**(2) Theorem 6.8 assumes orthogonal initialization; real embeddings are not orthogonal.** They
carry a large common mode. So a *second* null is integrated from the **observed** layer-0
`ip_mean` ($g_0 = {\rm ip\_mean}(0)$ rather than $0$), and `anisotropy_gap` reports the mean
distance between the two curves. Where they disagree, **the disagreement is an anisotropy effect
and not resistance** — a distinction the single-null version cannot make. This is the same
common-mode concern that runs through `math-1.md` §5.2 (the $\kappa_1$ term), reappearing as an
initial condition.

**(3) The paper's field is non-causal; the model is.** See §1.5.

### 3.5 Adjudication: falsifier separate from monotonicity

P-γ1 has two clauses — *the residual is near zero at step 0* and *it grows monotonically with
training* — and `adjudicate_p_gamma1` deliberately reports them **separately**, because they can
come apart: the residual can grow without being monotone, which is a partial confirmation and
must not be recorded as a pass. The registered falsifier is specifically that the step-0 residual
is already as large as the final one, which would mean **the gap is architectural rather than
learned**.

`collapse_fraction` reports two more numbers that answer different questions and are easy to
conflate:

- `time_fraction` $= T_{\rm eff}/t^\ast$ — the P-γ2 quantity.
- `gamma_fraction` $= \gamma_{\rm null}(T_{\rm eff})/\text{target}$ — how far identity-weight
  dynamics would have gotten in that time.

Because $\gamma$ saturates, "a small time fraction can still correspond to substantial
clustering, or to essentially none, depending on where on the curve it lands. Reporting only the
time fraction invites reading a linear relationship into a saturating one."

---

## 4. The $\beta$ reduction: dissolving a blocker with monotonicity

### 4.1 The blocker, and why it could not be dissolved the easy way

`core/beta_eff.py` returns $\beta$ **per head**; the null needs one $\beta$ **per layer**. Mean,
median, and attention-weighted give different answers, and `run_1c` refuses to invent one — so
the two highest-value measurements in the project were gated behind a choice with no principled
basis, which would have been made by whoever wrote the driver.

Unlike the clusterer question (§7.3 below), this one genuinely matters. Measured spread in
$\gamma_\beta(T_{\rm eff}{=}3)$ across $\beta\in[0.5,5]$:

| $n$ | $\gamma$ at $\beta{=}0.5$ | at $\beta{=}5$ | spread |
|---|---|---|---|
| 20 | 0.9482 | 0.0577 | **0.89** |
| 128 | 0.7545 | 0.1386 | 0.62 |
| 467 | 0.4610 | 0.1989 | 0.26 |

Larger than any residual we could hope to measure.

### 4.2 The way out: bracket it

$\gamma_\beta(t)$ is **monotone in $\beta$ at fixed $t$**, verified numerically over
$n\in\{5,20,64,128,467,512\}$, $t\in[0,8]$, $\beta\in[0.01,10]$ — **984,246 grid points per
model**:

$$
\textbf{(SA)}\ \text{monotone DECREASING in }\beta,\ \textbf{zero}\text{ violations};
\qquad
\textbf{(USA)}\ \text{monotone INCREASING},\ \sim35\%\text{ of points increase}
$$

Monotonicity means the per-head $\beta$ **range brackets the null** without any reduction being
chosen:

$$
\text{envelope}(t) = \big[\gamma_{\beta_{\max}}(t),\ \gamma_{\beta_{\min}}(t)\big] \quad\text{for (SA)}
$$

and the observed `ip_mean` either falls **outside** the envelope — in which case the conclusion
holds for *every* reduction and the decision is moot — or **inside** it, in which case the
decision genuinely matters and the envelope width is the honest uncertainty. `beta_reduction.py`
therefore reports a **bracket rather than a point estimate**.

### 4.3 The sign reversal is worth stating on its own

(SA) and (USA) respond to $\beta$ in **opposite directions**, and the partition function is what
reverses the sign. So using the surrogate as a stand-in for the normalized dynamics gets the
*direction* of the $\beta$-dependence backwards, not merely the magnitude — **and the envelope
endpoints swap between models.** This is the sharp form of the general (SA)/(USA) warning in
`math-1.md` §1.2, and a concrete instance of §2.3's fact 2.

**Remaining dependency:** `geometry.json` must carry `beta_eff_per_head`. If it carries only a
scalar `beta_eff`, the residual becomes a point estimate with an unreported error bar — and
`run_1c` says so in the artifact rather than quietly proceeding.

---

## 5. Sub-experiment C: moments and the rank reconciliation

The mathematics is `math-1.md` §5.2 and §6.2; 1c is where it gets *applied to the artifacts*.
Two questions, both settled from data already on disk.

**Is the four-$\beta$ energy sweep redundant?** `verify_moment_identity` reconstructs $E_\beta$
from $(\kappa_1,\kappa_2,\kappa_3)$ and reports the residual per $\beta$. Deliberately **not a
pass/fail gate** — the expected pattern is <1% for $\beta\le2$ and ~25% at $\beta=5$ — it
"quantifies where the ladder can stand in for the energy sweep and where it cannot, which
determines which columns the re-report can actually drop."

`ladder_from_layer` prefers the exact Gram path and falls back to the histogram with the
off-diagonal→full conversion, **recording which source was used**, because the two are not
interchangeable at small $n$ (an order of magnitude at $n = 20$) and a mixed-provenance series
would be silently inconsistent.

**Was the rank collapse a sink count?** `rank_panel` puts four rank-like quantities on the same
axes per layer — `shannon_raw`, `shannon_normed`, `pr_rank`, `norm_pr` — plus

$$
\texttt{sink\_ratio} = \frac{\texttt{shannon\_raw}}{\texttt{norm\_pr}}
$$

Near 1 ⟹ raw rank is being set by the norm distribution and carries **no geometric information**;
well above 1 ⟹ direction is doing the work.

`adjudicate_sink_hypothesis` then tests it *across the depth profile*, not per layer — "a single
layer can agree by coincidence, a whole depth profile cannot" — using the correlation of
`shannon_raw` against `norm_pr` versus against `shannon_normed`, and returns one of three
verdicts with the consequence spelled out:

- **SINKS** — status-1's rank-collapse row is a statement about outlier token norms and must be
  rewritten on the normed quantity.
- **DIRECTIONAL** — the collapse survives the frame correction; the original claim stands, now on
  the frame-correct quantity.
- **MIXED** — report normed rank and norm-PR separately and **drop the raw column rather than
  interpreting it**.

This directly closes status-1 defects D1 and D10, at report-only cost. Note this is also the
instrument that answers `math-1.md` open question 5 (do the three rank surrogates agree?) — it
computes all of them; nobody has run it.

---

## 6. Sub-experiment D: is the sphere licensed?

### 6.1 The paper's own criterion, run on a model where it can fail

§2.2 of the paper **does not assume the sphere — it measures it.** RMS-norm multiplies by a
trained diagonal, so the true state space is a time-varying axis-aligned ellipsoid; the paper
sets that matrix to $I$ and justifies it empirically (ALBERT-xlarge-v2: mean 0.44, sd 0.008
across layers).

That is a reproducible measurement, and `frame_table.py` runs it on Pythia. **This reframes what
`core/ln_frame.py` is for**: it has been described in this project as a departure from the
paper's frame; it is the opposite — *the paper's own licensing check*. `gamma_dynamic_range`
reports the range and `sphere_license` adjudicates it against the paper's own benchmark.

If Pythia's $\gamma$ has wide dynamic range per layer, the correct manifold is the ellipsoid and
**every sphere-frame metric in Phase 1 inherits a distortion** — `ip_mean`, `ip_mass_near_1`,
effective rank, and the interaction energy alike.

### 6.2 The four frames

| frame | definition | why it's here |
|---|---|---|
| `l2` | $x/\lVert x\rVert$ | the existing sphere frame |
| `ln_plain` | LN with $\gamma{=}1,\beta{=}0$ | **exactly** sphere projection in the mean-zero subspace: $\mathrm{LN}(x) = \sqrt d\,P_{\mathbf 1}x/\lVert P_{\mathbf 1}x\rVert$, constant norm $\sqrt d$ |
| `ln_learned` | $\gamma,\beta_{\rm LN}$ as trained | what attention actually reads |
| `functional` | Torgerson double-centering of the symmetrized-KL matrix | the readout's own geometry |

**`ln_plain` deserves emphasis for a reason that has nothing to do with fidelity to the paper.**
Because it forces constant norm $\sqrt d$, it *structurally restores uniform token weights* and
removes the sink domination that status-1 defect D10 identifies in raw effective rank
(`math-1.md` §6.2). It is a better frame for a reason internal to our own measurement problems.

### 6.3 The learned-bias energy floor

$\beta_{\rm LN}$ adds a **fixed vector to every token** — pure common mode. It inflates
$\langle G\rangle$ by roughly $\lVert\beta_{\rm LN}\rVert^2$ regardless of input, and
$\langle G\rangle/2 = \kappa_1/2$ is the dominant term in the small-$\beta$ expansion of
$E_\beta$ (`math-1.md` §5.2). So **the learned LN bias puts a floor under the interaction energy
that has nothing to do with the tokens**, and every absolute energy number Phase 1 reports sits
on top of it. `bias_energy_floor` isolates it by recomputing with $\beta_{\rm LN} = 0$.

### 6.4 The functional frame, and why double-centering is the right move

The functional "frame" has no Gram matrix of its own — it starts from a *divergence* matrix
(symmetrized KL, `math-1.md` §3.8). Classical MDS (Torgerson) supplies one. With $D^2$ the
matrix of squared distances and $J = I - \frac1n\mathbf 1\mathbf 1^\top$ the centering operator,

$$
B = -\tfrac12\,J\,D^2\,J
$$

**Why this is the right object.** If $D$ came from Euclidean points $x_i$, then
$D^2_{ij} = \lVert x_i\rVert^2 + \lVert x_j\rVert^2 - 2\langle x_i,x_j\rangle$. The first two
terms are constant along rows and columns respectively, so double-centering annihilates them
exactly, leaving $B_{ij} = \langle x_i - \bar x,\ x_j - \bar x\rangle$ — the **centered Gram**.
So double-centering recovers the inner-product structure from distances alone, and once you have
a Gram matrix, **every moment identity, cumulant, effective rank and eigengap in this project
applies unchanged**. That is the whole point: it lets the functional view be compared to the
geometric ones on identical machinery rather than through a bespoke statistic.

(Caveat worth carrying: symmetrized KL is *not* a metric — `math-1.md` §3.8 — so $B$ is not
guaranteed positive semidefinite, and negative eigenvalues are a real possibility rather than
numerical noise. Their magnitude is itself a measure of how non-Euclidean the functional geometry
is.)

---

## 7. Sub-experiment E: the cone condition, solved exactly

### 7.1 The condition, and why only positivity is used

Lemma 6.4: if all $x_i(0)$ lie in an open hemisphere — $\exists w$ with $\langle x_i,w\rangle>0$
for every $i$ — the dynamics collapse to a single point **exponentially**. The proof sets
$r(t) = \min_i\langle x_i(t),w\rangle$, shows $r'\ge 0$ so the hemisphere is forward-invariant,
and obtains the rate from $\alpha' \ge (1-\alpha)/(2ne^{2\beta})$.

The crucial detail, stated explicitly by the paper: **only positivity of the attention weights
$a_{ij}(t)$ is used.** That is why the lemma extends to arbitrary $Q,K$ with $V = I$. Softmax
weights are always positive. So **the hypothesis is entirely a condition on the configuration** —
nothing about the weights can rescue or break it. (Same argument as `math-1b.md` §1.2, reached
from the other side.)

### 7.2 Wendel, computed in log space

$$
P(\text{all in one hemisphere}) = 2^{-(n-1)}\sum_{k=0}^{d-1}\binom{n-1}{k} \;=\; 1 \text{ when } d \ge n
$$

Our prompts have $n\in[20,512]$ and $d = 1024$, so $d>n$ for every prompt: **at random
initialization the tokens are almost surely in an open hemisphere**, and Lemma 6.4 then predicts
exponential collapse.

`wendel_probability` computes this via `gammaln` with a log-sum-exp, because "at $n = 512$ the
binomials overflow float64 well before the sum is taken, and the naive form silently returns
inf/nan **exactly in the range our prompts occupy**."

**Why the prediction is registered in the boring direction on purpose:** P-H1 is stated as
"feasible at every checkpoint," which Wendel makes near-certain. The informative outcome is
therefore the *failure* — infeasibility, or feasibility with a margin near zero — either of which
would mean the embedding layer is doing something specific to escape a regime that otherwise
forces exponential collapse. **Report the margin, not the boolean.**

### 7.3 The exact solution by minimax duality — and why it is better than an LP

$$
m \;=\; \max_{\lVert w\rVert\le 1}\ \min_i\ \langle x_i, w\rangle
$$

The inner minimum over $i$ equals the minimum over the simplex:
$\min_i\langle x_i,w\rangle = \min_{\lambda\in\Delta}\big\langle \sum_i\lambda_i x_i,\ w\big\rangle$.
Both sets are compact and convex and the objective is bilinear, so minimax applies:

$$
m = \max_{\lVert w\rVert\le1}\min_{\lambda\in\Delta}\langle c(\lambda), w\rangle
= \min_{\lambda\in\Delta}\max_{\lVert w\rVert\le1}\langle c(\lambda), w\rangle
= \min_{\lambda\in\Delta}\lVert c(\lambda)\rVert,
\qquad c(\lambda) = \sum_i \lambda_i x_i
$$

i.e. **the margin is exactly the distance from the origin to the convex hull of the points**, and

$$
\boxed{\ m = \mathrm{dist}\big(0,\ \mathrm{conv}\{x_i\}\big) = \sqrt{\min_{\lambda\in\Delta}\ \lambda^\top G\,\lambda\ }\ }
$$

The cone condition holds iff $m>0$, i.e. iff $0\notin\mathrm{conv}\{x_i\}$.

Two consequences the module calls out, both worth internalizing:

- **It needs only $G$.** No $d$-dimensional optimization, no LP solver, no dependency. The
  problem is $n$-dimensional *regardless of model width*, and Phase 1 already computes $G$ for
  every layer of every run.
- **It is a convex QP with an exact optimum, not a feasibility heuristic.** A boolean from an LP
  with a tolerance would report "feasible" for a cloud whose margin is $10^{-9}$ — which is
  precisely the case we most want to catch.

> **Cross-phase observation.** `math-1b.md` §7.1 flags that Phase 1b answers the *same geometric
> question* with an $\ell_\infty$-ball LP whose margin is not scale-free — it divides by the max
> row norm but not by $\lVert w\rVert_2$, and the box's $\ell_2$ radius grows as
> $\sqrt{d_{\rm eff}}$, which varies across prompts. **The formulation above is exactly the fix.**
> It is the $\ell_2$-normalized margin by construction, it is dimension-free, it is exact rather
> than tolerance-gated, and it is cheaper. Two phases solve one problem two ways and only one of
> them is comparable across prompts. Phase 1b should adopt `hull_min_norm`, or at minimum the two
> should be run against each other on the same layers and the difference reported.

### 7.4 Solving the QP

Projected gradient with Nesterov acceleration on the simplex. Two implementation points that are
not incidental:

**The Lipschitz constant uses the true top eigenvalue, not the bound.** The objective's gradient
is $2G\lambda$, so $L = 2\lambda_{\max}(G) \le 2n$. Using the bound would be fine on a strongly
anisotropic cloud where $\lambda_{\max}\approx n$, but on a near-orthogonal one
$\lambda_{\max}\approx 1$ and the bound gives a step **$n$ times too small**.

**Simplex projection is exact and $O(n\log n)$** (sort, cumulative sum, threshold) rather than an
iterative inner solve.

**Non-convergence is directional and recorded.** `converged=False` means the reported margin is
an *upper bound* on the true one — and that matters, because an unconverged run can only make the
cloud look **more** feasible than it is. Knowing the direction of a numerical failure is what
lets it be reported rather than guarded against.

### 7.5 This is a different question from Phase 1b's

Stated in the module and worth repeating: `p1b_hemisphere` finds **bipartitions** — how the cloud
splits into two groups. This tests whether the **whole cloud fits in one open half-space through
the origin**. A cloud with a clean bipartition can still satisfy the cone condition, and a cloud
with none can fail it. (See `math-1b.md` §4.4 for the exact inequality linking the two.)

---

## 8. Sub-experiment F: sharp configurations and spherical designs

This is the phase's most interesting mathematics, and the only place the project's central
empirical claim has ever been given a **named limit object**.

### 8.1 Where it comes from: §9.1's rewriting

With $V = -I_d$ the interaction energy *decreases* along trajectories (`math-1.md` §1.4). Now
rewrite $E_\beta$ for unit vectors. Since $\lVert x - x'\rVert^2 = 2 - 2\langle x,x'\rangle$,

$$
e^{\beta\langle x,x'\rangle} = e^{\beta}\,e^{-\frac{\beta}{2}\lVert x-x'\rVert^2}
\quad\Longrightarrow\quad
E_\beta[\mu] = \frac{e^{\beta}}{2\beta}\iint e^{-\frac{\beta}{2}\lVert x-x'\rVert^2}\,d\mu\,d\mu
$$

So $E_\beta$ is (up to a positive constant) the **Gaussian-kernel energy** of the measure, and
minimizing it over $n$-atom empirical measures is exactly the classical **optimal point
configuration problem**.

**Theorem 9.2** (via Cohn–Kumar) then says: any global minimum is a **sharp configuration** in the
sense of Definition 9.1 —

> $m$ distinct pairwise inner products, and a spherical $(2m{-}1)$-design

— or the vertices of the 600-cell.

### 8.2 Why this is the phase's point

Blog 1's headline is "trained weights resist collapse." **If resistance means the trained model
sits in the repulsive regime, the paper predicts a specific limit geometry** — not a diffuse
spread, but a sharp configuration: *few* distinct pairwise inner products and a design condition.

That is the first target geometry the project's central empirical claim has ever had. P-S1 is
registered on it: trained centroids should be closer to a spherical $t$-design than step-0
centroids. **If they are, "resisting collapse" has a name. If they are not, the repulsive-regime
reading of the result needs revision.**

Note this is also the sub-experiment that speaks to the $d = 1024$ tension of `math-1.md`
§1.6 — because the question there is about limit **geometry**, not rate, and A and B (which are
about rate and time) cannot address it.

### 8.3 The test: Gegenbauer moments

On $\mathbb S^{d-1}$, a set is a spherical $t$-design **iff** its normalized Gegenbauer moments
vanish up to degree $t$:

$$
Q_k \;=\; \frac{1}{n^2}\sum_{i,j} P_k^{(d)}\big(\langle x_i,x_j\rangle\big) \;=\; 0,
\qquad 1\le k\le t
$$

with $P_k^{(d)}$ the ultraspherical polynomial normalized to $P_k(1) = 1$.

**Why $Q_k \ge 0$ always.** By the addition theorem for spherical harmonics,
$P_k^{(d)}(\langle x,y\rangle)$ is a positive multiple of $\sum_{\text{degree-}k} Y(x)\overline{Y(y)}$,
so

$$
Q_k \;\propto\; \Big\lVert \sum_i Y_k(x_i)\Big\rVert^2 \;\ge\; 0
$$

and $Q_k = 0$ exactly when the configuration integrates every degree-$k$ harmonic correctly. **The
non-negativity is a free correctness check on the implementation**, and it is asserted in the
code — a nice example of a theoretical identity doubling as a unit test.

**And it is the same mathematics as Proposition 3.4's proof, from the other side.** That
proposition — uniform is the unique minimizer of $E_\beta$ over *all* measures — is proved by
expanding $e^{\beta t}$ in Gegenbauer polynomials and showing every coefficient
$\hat f(k;\lambda) > 0$ for $k\ge1$, i.e. that the kernel is strictly positive definite on the
sphere (`math-1.md` §1A.3). Positive-definiteness of the kernel and non-negativity of the design
moments are one fact used twice: once to characterize the minimizer, once to test for it. Worth
knowing, because it means F is not an unrelated geometric side-quest — **it is the natural
finite-$n$ shadow of the same expansion that underwrites the energy functional's extremizer
structure.**

**The finite-$n$ deviation is the whole reason F exists.** Over all of
$\mathcal P(\mathbb S^{d-1})$ the minimizer is unique (uniform). Restricted to empirical measures
with $n$ atoms, *many* distinct global minima appear — and those are the sharp configurations.
The paper flags this explicitly as the point where "the particle dynamics and the mean-field flow
deviate." Since our data is always $n$ atoms, **the sharp configuration, not the uniform measure,
is the object the repulsive regime should be compared against** — which is exactly what P-S1
does and what a naive reading ("resistance means the cloud spreads toward uniform") would get
wrong.

This is a strong structural signature precisely because $Q_k$ is not a fitted quantity with a
threshold: *it is zero or it is not*. And it is cheap — the Gram matrix is all it needs.

### 8.4 Numerics: why `scipy.special.gegenbauer` is unusable here

The Gegenbauer parameter is $\alpha = (d-2)/2$, which is **511** for Pythia-410M, and the
coefficient representation overflows. The normalized polynomials satisfy a stable three-term
recurrence instead (Müller's "Legendre polynomials in $d$ dimensions"):

$$
P_0 = 1,\quad P_1(t) = t,\qquad
(k + d - 2)\,P_{k+1}(t) = (2k + d - 2)\,t\,P_k(t) - k\,P_{k-1}(t)
$$

evaluated pointwise on the Gram entries. $P_k(1) = 1$ follows by induction — checking
$(k+d-2)\cdot 1 = (2k+d-2)\cdot 1 - k$ — which is the second correctness check asserted in the
code.

### 8.5 The centroid problem, and how measurement dissolved it

F was originally left unwired because Phase 1 produces three clusterings per layer with different
centroid counts $m$, and $m$ was assumed to move the random baseline — so the clusterer choice
looked like a blocking decision.

**Measurement says otherwise.** $Q_k/Q_k^{\text{random}}$ for i.i.d. uniform configurations at
$d = 256$:

| $m$ | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ratio $Q_1$ | 1.104 | 1.048 | 0.967 | 0.922 | 0.951 | 0.901 |
| ratio $Q_2$ | 1.003 | 1.001 | 1.021 | 0.994 | 0.993 | 0.986 |

Flat at 1 across a **32× range in $m$**. And a genuinely sharp configuration stays low at every
$m$: the regular simplex — a spherical 1-design — gives ratio $Q_1 = 0.000$ at $m = 5,10,20,40$.

So the ratio **is** comparable across different centroid counts, and P-S1 can be adjudicated
between checkpoints whose clusterings disagree on $m$. The clusterer still has to be *fixed* for
a clean comparison, but not by matching $m$ — which is what made it look hard.

### 8.6 The effect-size floor, and a corrected intuition

The same measurement supplied what P-S1 was missing: **a threshold**. The random ratio's 2σ band:

| $k$ | 1 | 2 | 3 |
|---|---|---|---|
| $m=8$ | 0.164 | 0.015 | 0.002 |
| $m=32$ | 0.173 | 0.013 | 0.002 |

Without this, random-vs-random returns a PARTIAL verdict on nothing.

**And it corrected a claim that had been written into the code.** The author had asserted that
discriminating power sits at low $k$, because the simplex's $Q_2$ ratio is $\approx 0.98$ and
"looks like no signal." Wrong: the $k{=}2$ band is 0.014, so 0.023 is *outside* it. **Higher
degrees are more sensitive in relative terms** — both the deviation and the noise shrink with
$k$, at different rates, which is exactly why a fixed absolute tolerance is wrong in a different
direction at every degree.

This is the same lesson as `math-1.md` §8.3 (Fiedler thresholds) and `math-1b.md` §7.3
(separation-ratio thresholds), in its third independent instance: **a threshold that has not been
derived from the null's own distribution is not a threshold.**

---

## 9. What A and B settle, and what they do not

They settle whether the resistance claim **survives being made quantitative**. They do **not**
address the $d = 1024$ tension `status-1.md` opens with — whether our plateaus are the paper's
metastability at all. That needs sub-experiment F (spherical designs) and Phase 2d's Table 1
tests, because the question is about limit *geometry*, not rate.

**Sequencing.** Run `tools/preflight_1c.py` first — it reads only file headers and JSON keys, no
activations, and reports capability coverage plus the runnable count per sub-experiment *with the
blocking reason*; it exits non-zero when `norms` or `beta_eff` is incomplete, so it can gate a
scheduling script rather than only inform a reader. Then A and B (reading the residual **bracket**
rather than a point estimate), then the rest of 1c, then 2d.

**2d waits on B**, because the $T_{\rm eff}$ result determines whether the energy-monotonicity
break is even the right thing to attribute.

---

## 10. Code map

| File | Sub-exp | What it computes |
|---|---|---|
| `gamma_ode.py` | — | (6.9)/(6.10) right-hand sides with overflow-safe factoring; RK4 with step-halving convergence; `time_to_threshold`; `collapse_time`; the 28-entry validation table |
| `integration_time.py` | A | `sa_field` (causal and non-causal), field magnitude, tangential displacement, the three step definitions, `cumulative_time`, `verdict` with robustness across definitions |
| `gamma_null.py` | B | Null on a non-uniform time grid, per-layer $\beta$, the matched (observed-$g_0$) null and `anisotropy_gap`, `residual_curve`, `collapse_fraction`, `adjudicate_p_gamma1` |
| `beta_reduction.py` | B | The monotonicity-based envelope replacing the head→layer reduction decision |
| `moments.py` | C | `verify_moment_identity`, `ladder_from_layer` with provenance, `rank_panel`, `adjudicate_sink_hypothesis` |
| `frame_table.py` | D | Four frames, $\gamma$ dynamic range vs the paper's ALBERT benchmark, `sphere_license`, `bias_energy_floor`, Torgerson double-centering for the functional frame |
| `hemisphere_feasibility.py` | E | `wendel_probability` in log space; `hull_min_norm` — the exact margin via minimax duality, Nesterov projected gradient, $O(n\log n)$ simplex projection |
| `design_test.py` | F | Normalized Gegenbauer polynomials by three-term recurrence; $Q_k$; sharpness/design adjudication |
| `centroids.py` | F | Which centroids feed F; the matched-$(m,d)$ baseline; `random_band` (the effect-size floor) |
| `run_1c.py`, `p1c_io.py` | — | Driver and artifact IO |
| `tools/preflight_1c.py` | — | Artifact-readiness gate; reports per-sub-experiment runnability with blocking reasons |

**Validation that actually constrains** (from `UPDATE_PLAN.md` §5): the ODE reproduces all 28
collapse-table entries to 0.005; the step estimator recovers an injected Euler step **exactly**;
the Gegenbauer code recovers $t = 3$ for the octahedron and $t = 5$ for the icosahedron; Wendel
reproduces the textbook $n{=}3, d{=}2 \to 0.75$; the cone margin gives $\cos 30°$ on a $30°$ cone.
These are oracle tests against known exact answers, not regression tests against previous output.

---

## 11. Open questions

Tracked:

1. **Nothing has been run against Pythia artifacts.** Every sub-experiment is implemented and
   validated on synthetic data and on configurations with known exact answers. That is by design
   (the predictions were pre-registered while the artifacts were being regenerated), but it means
   the highest-value measurement in the project is sitting at zero forward-pass cost, unrun.
2. **`geometry.json` must carry `beta_eff_per_head`** or the envelope degrades to a point
   estimate with an unreported error bar (§4.3).
3. The energy-trajectory figures need regenerating — three suptitles baked a wrong citation into
   every PNG.

Surfaced by writing this document:

4. **Phase 1b and Phase 1c solve the same geometric problem with different formulations, and only
   1c's is comparable across prompts** (§7.3). This should be reconciled explicitly: either 1b
   adopts `hull_min_norm`, or both are run on the same layers and the discrepancy is reported.
   It is a rare case where a cross-phase inconsistency has an unambiguously correct resolution.
5. **The functional frame's Gram may be indefinite, and nobody has said what to do about it**
   (§6.4). Symmetrized KL is not a metric, so $B = -\frac12 JD^2J$ need not be PSD. The size of
   the negative spectrum is a *measurement* of how non-Euclidean the readout geometry is — it
   would be a shame to clip it silently, and clipping is the default behaviour of most code that
   consumes a Gram matrix.
6. **Sub-experiment F tests centroids, but the sharp-configuration prediction is about the whole
   configuration.** Definition 9.1 asks for few distinct pairwise inner products among *the
   particles*; using cluster centroids is a coarse-graining that could manufacture sharpness (a
   handful of well-separated centroids trivially has few distinct inner products) or destroy it.
   The choice is reasonable — centroids are what the theory's limit object most resembles — but
   the *number of distinct inner products* half of Definition 9.1 is checkable on the full token
   set at no extra cost, and running both would say whether the coarse-graining is doing work.
7. **The time residual has no registered prediction.** §3.2 shows it is two and a half orders of
   magnitude more sensitive than the vertical residual, and P-γ1 is registered on the vertical
   one. Since the predictions are pre-committed and must not be silently rewritten, the correct
   move is a **dated addendum** adjudicating on the time residual with its own falsifier — the
   same mechanism used for P-T1's amendment. Otherwise the phase's most sensitive instrument sits
   outside its falsification record.
8. **$t^\ast$ is computed at a fixed target of $\gamma = 0.9$, and "collapse" is a choice.** The
   collapse-time table also reports $\gamma = 0.5$. Since P-γ2 is a comparison of $T_{\rm eff}$
   against $t^\ast$, and $t^\ast$ roughly doubles between the two targets, the verdict inherits
   that choice in the same way §1.2 shows it inherits the step definition. `verdict()` guards
   against the step-definition ambiguity but not against the target ambiguity; both should be
   reported on the same footing.
