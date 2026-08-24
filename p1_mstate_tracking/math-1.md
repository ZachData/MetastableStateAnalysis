# Phase 1 — MATH (study notes)

## 0. What this document is, and isn't

`design-1.md` explains *why* the code is built the way it is. `status-1.md` reports what was
found and where the measurements break. Neither derives the mathematics from first principles,
states what the model literally computes, or walks the implementation formula-by-formula —
that's the gap this document fills. It is meant to be read slowly, with the source open next to
it, by someone who wants to be able to re-derive every number Phase 1 produces rather than just
cite it.

The spine of the document is one question: **the theory describes a specific idealized particle
system; Pythia is a specific real algorithm; what exactly is the difference, and how do we
measure it?** Sections 1–2 set up the two objects. Section 3 is the dictionary between them and
the machinery that quantifies the gap. Sections 4–11 are the per-metric derivations. Sections
12–15 are results, defects, and open questions.

Citations to the paper (Geshkovski, Letrouit, Polyanskiy, Rigollet, *A Mathematical Perspective
on Transformers*, arXiv:2312.10794v5) follow the corrected numbering in
`design-1.md`/`PREDICTIONS.md` — several theorem numbers in earlier drafts of this project were
wrong, and the corrections are load-bearing, not cosmetic. Where I'm reconstructing an argument
rather than repeating a verified citation from this repo's own docs, I say so.

---

# PART I — THE TWO OBJECTS

---

## 1. The idealized model: transformers as interacting particle systems

### 1.1 State space: why the sphere, and exactly when

Take a token's residual-stream vector $x \in \mathbb{R}^d$ at some layer. LayerNorm rescales it:

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu(x)\mathbf 1}{\sqrt{\sigma^2(x) + \epsilon}} +
\beta_{\mathrm{LN}}, \qquad
\mu(x) = \tfrac1d\textstyle\sum_c x_c, \quad \sigma^2(x) = \tfrac1d\sum_c (x_c - \mu)^2
$$

(biased variance, $\epsilon$ inside the square root — `core/ln_frame.py::ln_transform` matches
`torch.nn.LayerNorm` exactly on this, and GPT-NeoX's $\epsilon$ is `config.layer_norm_eps`,
default $10^{-5}$.)

**The idealization.** Set $\gamma = \mathbf 1$, $\beta_{\mathrm{LN}} = 0$, $\epsilon = 0$. Then
with $P_{\mathbf 1} = I - \frac1d \mathbf 1\mathbf 1^\top$ the projection onto the mean-zero
hyperplane, $\mu(x)\mathbf 1 = (I - P_{\mathbf 1})x$ and $\sigma^2(x) = \frac1d\lVert
P_{\mathbf 1}x\rVert^2$, so

$$
\mathrm{LN}(x) = \frac{P_{\mathbf 1}x}{\sqrt{\tfrac1d \lVert P_{\mathbf 1}x\rVert^2}}
= \sqrt d\,\frac{P_{\mathbf 1}x}{\lVert P_{\mathbf 1}x\rVert}
$$

That is *exactly* sphere projection: project onto the $(d-1)$-dimensional mean-zero subspace,
then rescale to constant norm $\sqrt d$. So "tokens live on a sphere" is not a metaphor
borrowed from physics — for plain LayerNorm it is literally what the operation computes, up to
the radius convention ($\sqrt d$ rather than 1, which rescales $\langle x_i,x_j\rangle$ by $d$
and is why every metric in this project normalizes to unit norm rather than using LN's own
radius).

**Where the idealization is a claim rather than an identity.** Real LayerNorm multiplies by a
trained per-channel $\gamma$, so the true state space is a time-varying axis-aligned
**ellipsoid**, and adds a trained $\beta_{\mathrm{LN}}$, which shifts the whole cloud off the
origin. The paper's own justification for the sphere is empirical, not structural: in
ALBERT-xlarge-v2 the $\gamma$ diagonal is nearly constant across channels and layers (mean
0.44, sd 0.008, paper §2.2). A near-constant $\gamma$ is a *scalar* rescaling, which preserves
all angles and therefore leaves every cosine-based metric untouched.

`core/ln_frame.py` exists to re-run that check on every model this project touches. This is the
paper's own licensing condition, not a departure from it, and **it can fail**: if a model's
$\gamma$ has wide per-channel dynamic range, the correct manifold is an ellipsoid and every
sphere-frame metric in this phase (`ip_mean`, `ip_mass_near_1`, $E_\beta$, normed effective
rank) inherits a distortion whose magnitude is exactly how far $\gamma$ is from constant.

Two structural consequences to internalize before anything else:

- **$\beta_{\mathrm{LN}}$ is a free confound on every energy number.** It adds the *same* vector
  to every token, independent of content — pure common mode. §5.2 shows the common-mode
  cumulant $\kappa_1$ dominates the small-$\beta$ expansion of $E_\beta$, so a learned LN bias
  puts a floor under the interaction energy that has nothing to do with the tokens. Every
  absolute $E_\beta$ this phase reports carries that floor.
- **Plain LayerNorm structurally erases norm outliers from the frame.** Because LN forces
  constant norm, an attention-sink token with a raw residual norm 30× the mean gets exactly the
  same weight as everything else once projected. This is the whole reason "raw" and "normed"
  effective rank are different quantities (§6) — and why one of them is a statement about
  geometry and the other is largely a sink count.

### 1.2 The dynamics: tangent flow, (SA) and (USA)

For $n$ particles $x_1,\dots,x_n \in \mathbb S^{d-1}$, the idealized continuous-time dynamics
is a tangent-space flow:

$$
\dot x_i = P^\perp_{x_i}\!\Big(\sum_{j=1}^n a_{ij}(t)\, V x_j\Big), \qquad
P^\perp_x y = y - \langle x, y\rangle x
$$

**Why the tangent projection is the right image of LayerNorm.** Suppose we move off the sphere
by a small step and renormalize: $x' = (x + \eta u)/\lVert x + \eta u\rVert$. Expanding to
first order in $\eta$ with $\lVert x\rVert = 1$,

$$
\lVert x + \eta u\rVert = 1 + \eta\langle x,u\rangle + O(\eta^2)
\;\Longrightarrow\;
x' = x + \eta\big(u - \langle x,u\rangle x\big) + O(\eta^2) = x + \eta\,P^\perp_x u + O(\eta^2)
$$

So *add-then-normalize* and *move within the tangent plane* agree to first order, and only to
first order. **This is the exact sense in which a residual block is one Euler step of the
flow** — and it is what makes the step size $h_\ell$ of §3.5 the natural clock, rather than the
layer index.

Two conventions for the interaction weights, and it matters which a given theorem is about:

- **(USA)** — unnormalized: $a_{ij} = e^{\beta\langle Qx_i,\, Kx_j\rangle}$, no partition
  function. Easier to analyze; not what a transformer computes.
- **(SA)** — the real softmax: $a_{ij} = e^{\beta\langle Qx_i, Kx_j\rangle}/Z_{\beta,i}$ with
  $Z_{\beta,i} = \sum_k e^{\beta\langle Qx_i, Kx_k\rangle}$. Row-normalized. This is the
  mechanism.

Several of the sharper results (monotonicity, the gradient-flow structure) hold for **(SA)**;
assuming otherwise is exactly the class of citation error `design-1.md` records having made and
corrected. `UPDATE_PLAN.md` §5.10 records why this is not academic: the concentration curve
$\gamma_\beta(t)$ (§1.5) is monotone *decreasing* in $\beta$ under (SA) and monotone
*increasing* under (USA) — verified over ~984k grid points per model, zero violations on the
(SA) side. Using the surrogate as a stand-in gets the **sign** of the $\beta$-dependence
backwards, not merely its magnitude.

### 1.3 The energy functional

$$
E_\beta(t) \;=\; \frac{1}{2\beta n^2}\sum_{i,j=1}^n \exp\!\big(\beta\langle x_i(t), x_j(t)\rangle\big)
$$

— exactly `core/metrics.py::interaction_energy`. Facts, from Proposition 3.4 (the one theorem
number in this phase that was never mis-cited):

- $E_\beta$ is **maximized** at full collapse (every $\langle x_i,x_j\rangle = 1$, every term
  $e^\beta$) and **minimized** at the uniform measure on the sphere. Rising $E_\beta$ ⟹
  particles moving together on average; falling ⟹ spreading apart. It is an order parameter for
  collapse.
- The diagonal always contributes $n e^\beta$, so $E_\beta \ge e^\beta/(2\beta n)$ regardless of
  geometry. Not usually dominant, but at $n = 20$ (the short prompt) it is a visible floor.
- Scale sanity: for a perfectly isotropic cloud in high $d$, off-diagonal $\langle
  x_i,x_j\rangle \approx 0$, so $E_\beta \approx \frac{1}{2\beta}\big(1 + O(1/n)\big)$ — i.e.
  $E_\beta \to \frac{1}{2\beta}$ is the "no structure" reference value, worth carrying in your
  head when reading absolute energies (at $\beta=1$: 0.5).

### 1.4 Monotonicity, and the attractive/repulsive split

For the fully-attractive idealization ($Q^\top K = I$, $V = +I_d$) the paper proves
$dE_\beta/dt \ge 0$ along the (SA) flow — **eq. (3.6)**, with **Lemma 3.7** the (USA)
counterpart — *not* Proposition 3.4
(which characterizes $E_\beta$'s extremizers, a static statement about the functional, not a
claim about trajectories; conflating the two was a corrected error). The paper also states that
$V = -I_d$ makes the same functional strictly *decrease*: the repulsive regime is a case the
theory treats, not a failure mode outside its scope.

**Where the sign comes from.** Write $G_{ij} = \langle x_i,x_j\rangle$. Then

$$
\frac{dE_\beta}{dt} = \frac{1}{2n^2}\sum_{i,j} e^{\beta G_{ij}}\,\dot G_{ij},
\qquad \dot G_{ij} = \langle \dot x_i, x_j\rangle + \langle x_i, \dot x_j\rangle
$$

and, substituting the flow with $V = I$ and using $\langle P^\perp_{x_i}y, x_j\rangle =
\langle y,x_j\rangle - G_{ij}\langle y, x_i\rangle$, every term becomes a product of positive
weights $a_{ik} > 0$ (softmax output — strictly positive, always) with Gram-matrix bilinears
that reorganize into a sum of squared tangential quantities. Two things follow immediately, and
both are used later:

1. **Positivity of the attention weights is doing the work**, not their specific functional
   form. This is why Lemma 6.4's cone-collapse hypothesis (Phase 1b's object) needs only that
   the weights are positive — which softmax guarantees unconditionally — and is therefore
   entirely a condition on the *initial configuration*, not on the weights.
2. **Flipping $V \to -I$ flips $\dot x_i$, hence every term, without changing the magnitude
   structure.** The attraction/repulsion question is entirely a question about $V$'s spectrum.
   This is the whole premise of Phase 2: if $V$ has mixed-sign eigenvalues, the flow is
   attractive along some eigendirections and repulsive along others simultaneously.

**The sharper, weight-dependent condition** is §3.4's: (SA) is a genuine gradient flow of
$E_\beta$ in the reweighted inner product $\langle a,b\rangle_X = \sum_i Z_{\beta,i}\langle
a_i,b_i\rangle$ **only when $Q^\top K$ is symmetric and $V = Q^\top K$.** Outside that
condition — which is essentially every real attention head — there is *no monotonicity
guarantee in either direction*. This converts "the theorem is violated" from a pass/fail check
into a localization question: **do violations concentrate on heads far from that condition?**
(prediction **P-M1**, tested by Phase 2d's D1). Phase 1 can only observe the violations; it has
no operators, so it cannot ask the second half.

### 1.5 Long-time behaviour: three results, three regimes

Easy to conflate, and the conflation has already cost this project one retracted verdict row.

- **Theorem 6.1 — qualitative.** For $d \ge 3$, at *any* $\beta \ge 0$, the identity-weight
  dynamics converge to a single cluster. **No rate, no dimension-dependence.** A prior version
  of the verdict table read "higher $d$ → faster convergence (Thm 6.1): unsupported" — testing a
  claim the theorem does not make. Retracted as a prediction rather than reported as a
  falsification.
- **Theorem 6.3 — exponential rate when $d \ge n$.** In that regime convergence is exponential,
  with the rate carrying an $O(e^{-\beta})$ dependence. Every prompt in this project satisfies
  $d \ge n$ on Pythia ($d = 1024$, $n \le 512$), so the hypothesis holds *everywhere* and the
  rate is a live, untested prediction — nothing in the current metric set measures a rate at
  all (§15). Note this is also the theorem that Phase 5 mis-cites for an unrelated
  intra-cluster-mass claim (`UPDATE_PLAN.md` §1).
- **Theorem 6.9 / the curve $\gamma_\beta(t)$ — concentration when $d \gg n$.** As $d\to\infty$
  with $n$ fixed, *every* pairwise inner product concentrates onto a single deterministic curve
  $\gamma_\beta(t)$ solving a scalar ODE (integrated numerically by Phase 1c's `gamma_ode.py`).
  **This is the paper's own explanation for why metastability should disappear at high $d$:**
  if every pair sits on one curve, there is no room for a genuine multi-cluster intermediate
  configuration. Figure 3's $(d,\beta)$ sweep shows the metastable band narrowing and vanishing
  by $d \approx 512$.

### 1.6 Problem 1: metastability is a conjecture, not a theorem

The single most important correction `design-1.md` makes. The paper does **not** prove that
trajectories pass through metastable multi-cluster states. It proves single-cluster convergence.
The metastable-plateau phenomenon is **Problem 1**, posed explicitly as open, with numerical
support only — Figure 4, at $d = 2$, $\beta = 4$ (two clusters) and $\beta = 9$ (three).

So "plateaus in all 216 Pythia runs" is evidence bearing on an *open conjecture*, gathered at
$d \approx 10^3$ — three orders of magnitude outside the $d\in\{2,\dots,8\}$ regime the
conjecture's own evidence lives in, and (per §1.5) inside the regime where the paper's own
Figure 3 says metastability should not exist. Two readings are consistent with everything on
disk:

1. The plateaus this detector fires on are a different object from the paper's Problem-1
   metastability.
2. The concentration argument behind Thm 6.9 (which assumes $Q^\top K = V = I$) simply fails
   under learned multi-head weights, and real high-$d$ metastability is what that failure looks
   like.

Phase 1's instrument set cannot separate these. **Read every plateau result below as "the
falsification criterion did not fire," not as "the conjecture was replicated."** Phase 1c's
$\gamma_\beta$ residual is the instrument built to force the choice (§3.6).

---

## 1A. The optimal-transport picture

Everything above is stated particle-by-particle. The paper's actual framing is one level up —
**a Transformer is a flow map on the space of probability measures** — and that framing is
where the energy functional, the (SA)/(USA) split, and the gradient-flow condition behind P-M1
all come from. This section is the measure-level story, because several things this project
treats as separate facts are one fact from up here.

### 1A.1 Transformers as measure-to-measure maps

A prompt is unordered as far as the architecture is concerned (order enters only through
positional encoding *of the initial condition*), so a sequence is perfectly encoded by the
empirical measure of its tokens. Write

$$
\mu(t,\cdot) = \frac1n\sum_{i=1}^n \delta_{x_i(t)}(\cdot) \;\in\; \mathcal P(\mathbb S^{d-1})
$$

Then (SA) is a **mean-field interacting particle system**: each particle follows a vector field
that depends on all the others only through $\mu$,

$$
\dot x_i(t) = \mathcal X[\mu(t)]\big(x_i(t)\big),
\qquad
\mathcal X[\mu](x) = P^\perp_x\!\left(\frac{1}{Z_{\beta,\mu}(x)}\int e^{\beta\langle x,y\rangle}\,y\,d\mu(y)\right)
$$

with $Z_{\beta,\mu}(x) = \int e^{\beta\langle x,y\rangle}d\mu(y)$, and the measure itself evolves
by the **continuity equation**

$$
\partial_t\mu + \mathrm{div}\big(\mathcal X[\mu]\,\mu\big) = 0
\qquad\text{on }\ \mathbb R_{\ge0}\times\mathbb S^{d-1}
$$

in the sense of distributions. So the network is a flow map $\mu(0)\mapsto\mu(T)$, and the
output measure is read as the distribution over next tokens — **clustering of $\mu(T)$ is
literally a statement that few outcomes are likely.** That is why clustering is the object of
study and not an incidental geometric curiosity.

*Practical caveat worth carrying* (the paper's Remark 3.3): the mean-field limit connecting
$\mu_n$ to a continuum $\mu$ holds by Dobrushin's argument,
$W_1(\mu_n(t),\mu(t)) \le e^{O(1)|t|}W_1(\mu_n(0),\mu(0))$, but with two weaknesses — the time
dependence is exponential, and if the tokens are i.i.d. samples then
$W_1(\mu_n(0),\mu(0))\to0$ only at rate $n^{-1/(d-1)}$, **which deteriorates badly as $d$
grows**. At $d = 1024$ that rate is meaningless. So the continuum picture is a source of
structural insight here, not of quantitative guarantees; every quantitative claim in this
project is a finite-$n$ claim.

### 1A.2 The Lyapunov identity, exactly

The derivative of the interaction energy along the continuity equation is not merely
non-negative — it has a closed form. Integrating by parts,

$$
\boxed{\ \frac{d}{dt}E_\beta[\mu(t)] \;=\; \int \big\lVert \mathcal X[\mu(t)](x)\big\rVert^2\,
Z_{\beta,\mu(t)}(x)\, d\mu(t,x)\ }
\qquad\text{with}\quad e^{-\beta}\le Z_{\beta,\mu}(x)\le e^{\beta}
$$

Three things follow that are worth having in this exact form:

1. **The rate of energy increase is the squared speed of the flow, weighted by the partition
   function.** So $E_\beta$ stops moving exactly when the particles stop moving. It is a
   Lyapunov function in the strict sense, not just a monotone quantity.
2. **The $Z_{\beta,\mu}$ weighting is bounded in $[e^{-\beta},e^{\beta}]$**, so the identity
   sandwiches $dE_\beta/dt$ between $e^{-\beta}$ and $e^{\beta}$ times $\int\lVert\mathcal
   X\rVert^2d\mu$. That bracket is what makes $\lVert\mathcal X\rVert$ — the field magnitude
   Phase 1c's `field_magnitude` computes — the natural companion quantity to an energy
   violation, and it is the same $\lVert\mathcal X\rVert$ that calibrates the Euler step
   (`math-1c.md` §1.2).
3. **$V = -I_d$ flips the sign of the whole identity**, so $E_\beta$ strictly *decreases*. The
   attractive/repulsive dichotomy is exact at the level of this one identity, not an
   approximation.

### 1A.3 Proposition 3.4, and why its proof is the same mathematics as Phase 1c-F

Over all of $\mathcal P(\mathbb S^{d-1})$: **the uniform measure $\sigma_d$ is the unique global
minimizer of $E_\beta$, and every global maximizer is a Dirac mass.** Hence "energy up ⟹ toward
collapse, energy down ⟹ toward uniform."

The proof is worth knowing because the same machinery reappears downstream. Expand the kernel
$f(t) = e^{\beta t}$ in **Gegenbauer (ultraspherical) polynomials** with
$\lambda = \frac{d-2}{2}$,

$$
f(t) = \sum_{k\ge0}\hat f(k;\lambda)\,\frac{k+\lambda}{\lambda}\,C^\lambda_k(t)
$$

The necessary and sufficient condition for Proposition 3.4 is $\hat f(k;\lambda) > 0$ for all
$k\ge1$ — i.e. **$e^{\beta\langle x,y\rangle}$ is a strictly positive-definite kernel on the
sphere**, which is established via the Rodrigues formula and the parity of $C^\lambda_k$.

That is exactly the machinery Phase 1c's sub-experiment F uses from the other direction: the
addition theorem for spherical harmonics makes each Gegenbauer moment
$Q_k = \frac1{n^2}\sum_{ij}P^{(d)}_k(\langle x_i,x_j\rangle)$ a squared norm, hence
$Q_k\ge0$ always, with $Q_j = 0$ for **all** $1\le j\le t$ characterizing a spherical
$t$-design (`math-1c.md` §8.3 — a single $Q_k$ vanishing says only that the degree-$k$ harmonic
moment does).
**The two arguments share their engine but not their whole content: Schoenberg's theorem / the
addition theorem gives $Q_k\ge0$ on its own, while Proposition 3.4 additionally needs strict
positivity of $e^{\beta t}$'s Gegenbauer coefficients $\hat f(k;\lambda)$ — the separate
Rodrigues-plus-parity computation. Same machinery, one extra ingredient on the minimizer side
fact, used once to characterize the minimizer and once to test for it.**

**One deviation to keep straight, and it is load-bearing for Phase 1c-F.** Proposition 3.4
minimizes over *all* probability measures. Restricted to **empirical** measures with $n$ atoms,
many distinct global minima appear — this is where sharp configurations and spherical designs
come from (§9.1, Cohn–Kumar). The paper flags it explicitly: *"This is one point where the
particle dynamics and the mean-field flow deviate."* Our data is always $n$ atoms, so the
relevant limit object is the sharp configuration, not the uniform measure.

### 1A.4 Why (SA) is *not* a Wasserstein gradient flow

Given the Lyapunov identity, the natural hope is that the continuity equation is the Wasserstein
gradient flow of $E_\beta$ — which would hand us the entire convexity/long-time-asymptotics
toolkit for free. It is not, and the obstruction is precise and instructive:

$$
\mathcal X[\mu](x) \;=\; \frac{1}{\beta}\,P^\perp_x\,\nabla \log\!\int e^{\beta\langle x,y\rangle}\,d\mu(y)
$$

**The field is a *logarithmic* derivative.** For a Wasserstein gradient flow, $\mathcal X[\mu]$
would have to be $\nabla\delta E_\beta[\mu]$ — the gradient of the first variation, with no log.
The log is exactly the partition function $Z_{\beta,\mu}$, i.e. **softmax's normalization is
precisely what breaks the gradient-flow structure.** (The observation extends past
$Q = K = I,\ V = \pm I$ to any $Q^\top K = K^\top Q = \pm V$; the failure is due to lack of
symmetry.)

This single fact organizes several things this project treats separately:

- It is why **(SA) and (USA) are genuinely different objects**, not a model and its convenient
  approximation — and why they can respond to $\beta$ in *opposite directions*
  (`math-1c.md` §4.3).
- It is why the paper's own theorem set splits the way it does, and why getting the convention
  wrong flips signs rather than shifting magnitudes.
- It is why the two repairs below exist at all.

### 1A.5 Repair 1 — drop the denominator: (USA) is a genuine gradient flow

Remove the log (equivalently, replace $Z_{\beta,i}$ by $n$):

$$
\textbf{(USA)}\qquad \dot x_i = P^\perp_{x_i}\Big(\frac1n\sum_j e^{\beta\langle x_i,x_j\rangle}x_j\Big)
$$

Then $\mathcal X[\mu] = \nabla\delta E_\beta[\mu]$ **exactly** (Lemma 3.6), and the continuity
equation is the Wasserstein gradient flow of $E_\beta$. Writing $G_\beta(t) = \beta^{-1}e^{\beta t}$
and using the spherical convolution $(f*g)(x) = \int f(\langle x,y\rangle)g(y)d\sigma_d(y)$,

$$
E_\beta[\mu] = \tfrac12\int (G_\beta * \mu)\,d\mu,
\qquad
\mathcal X[\mu] = \nabla(G_\beta*\mu),
\qquad
\partial_t\mu + \mathrm{div}\big(\nabla(G_\beta*\mu)\,\mu\big) = 0
$$

— an **aggregation equation**, the same family as Patlak–Keller–Segel chemotaxis models, whose
finite-time collapse to a Dirac is well studied (on $\mathbb R^d$; the sphere is what makes this
case new). Its Lyapunov identity (Lemma 3.7) is the clean one:
$\frac{d}{dt}E_\beta = \int\lVert\nabla(G_\beta*\mu)\rVert^2 d\mu$.

At the particle level (Remark 3.8), with $E_\beta(X) = \frac{1}{2\beta n^2}\sum_{ij}
e^{\beta\langle x_i,x_j\rangle}$ — **which is exactly `core/metrics.py::interaction_energy`** —
the dynamics are literally $\dot X = n\nabla_X E_\beta(X)$, gradient *ascent* on the standard
product-sphere metric. So the quantity this project computes per layer is the potential whose
gradient ascent (USA) performs.

### 1A.6 Repair 2 — change the metric: (SA) *is* a gradient flow, for a reweighted inner product

This is the construction that P-M1 is built on, and the repo's design docs state its conclusion
without its content. Assume

$$
Q^\top K \ \text{symmetric},\qquad V = Q^\top K
$$

Define a **new Riemannian metric** on $(\mathbb S^{d-1})^n$ by reweighting each particle's
tangent space by its own partition function:

$$
\big\langle (a_i),(b_i)\big\rangle_X \;=\; \sum_{i=1}^n Z_{\beta,i}(X)\,\langle a_i, b_i\rangle,
\qquad Z_{\beta,i}(X) = \sum_j e^{\beta\langle Vx_i,\,x_j\rangle}
$$

Then (SA) is **exactly** $\dot X = \nabla E_\beta(X)$ with the gradient taken in this metric.
(Verified by testing against vector fields $Y(X) = (Ax_1,0,\dots,0)$ for skew-symmetric $A$,
whose flow is $(e^{tA}x_1, x_2,\dots)$; the identity reduces to the observation that
$\langle Ax_1,y\rangle = \langle Ax_1,z\rangle$ for all skew-symmetric $A$ iff $x_1(y-z)^\top$
is symmetric.)

**This is the entire content of prediction P-M1.** The theorem guarantees monotone $E_\beta$
only under those two hypotheses; a head with asymmetric $Q^\top K$, or with $V \ne Q^\top K$, is
**outside the theorem's scope entirely** — no guarantee in either direction. So an observed
violation is not a falsification; the right question is *whether violations localize on heads
far from the condition.* Phase 2b's `rotational_schur.py` already performs the
symmetric/antisymmetric split this needs, and Phase 2d's D1 is the test.

At the measure level the same reweighting generalizes. Writing (SA) as

$$
\partial_t\mu + \mathrm{div}\!\left(\frac{\nabla\delta E_\beta[\mu]}{\delta E_\beta[\mu]}\,\mu\right) = 0,
\qquad \delta E_\beta[\mu](x) = \int e^{\beta\langle x,y\rangle}d\mu(y)
$$

one replaces the usual Otto metric tensor $\langle\nabla\psi_1,\nabla\psi_2\rangle_\mu =
\int\langle\nabla\psi_1,\nabla\psi_2\rangle d\mu$ with the **$E_\beta$-weighted** one

$$
\langle\nabla\psi_1,\nabla\psi_2\rangle_{\mu,E_\beta} = \int\langle\nabla\psi_1,\nabla\psi_2\rangle\,
\delta E_\beta[\mu](x)\,d\mu(x)
$$

which induces, through a generalization of the Benamou–Brenier formula, a **weighted Wasserstein
distance** $W_{2,E_\beta}$; (SA) is the gradient flow of $E_\beta$ in that geometry. The
literature on such weighted distances is thin, which the paper notes.

The practical reading for us: **the partition function is not noise to be normalized away — it
is a metric.** Softmax reweights how far the configuration has to travel, per token, by how much
attention mass that token commands. A high-$Z$ token (a sink, `math-1.md` §2.5) is one the
metric makes *expensive to move*. That is a strikingly good match to what attention sinks
empirically do, and to my knowledge nothing in this project has looked at
$Z_{\beta,i}$ as a per-token quantity at all (§15, open question 12).

### 1A.7 Repair 3 — Sinkhorn: the doubly-stochastic symmetrization

There is a third route, and it retroactively justifies a Phase 1 module. Sander et al.
(Sinkformers) replace the row-stochastic attention matrix with a **doubly stochastic** one
obtained by Sinkhorn iteration; that symmetrization also yields a Wasserstein gradient flow. The
paper's Remark 3.5 then says, verbatim:

> Understanding the emergence of clusters for this model is an interesting but possibly
> challenging question.

**This is a much stronger motivation for `p1_mstate_tracking/sinkhorn.py` than its own docstring
gives.** The module's stated rationale is that "the doubly stochastic form is the gradient-flow
object" and that the gap between raw attention and that form is itself a measurement — correct,
but understated. The Sinkhorn-normalized attention matrix is one of exactly **three** known ways
to recover gradient-flow structure from self-attention (drop the denominator → USA; reweight the
metric → requires symmetric $Q^\top K$ and $V = Q^\top K$; Sinkhornize → always available), and
it is the only one of the three that can be applied to a *trained* model without assuming
anything about its weights. Measuring cluster structure on the Sinkhorn-normalized graph is
therefore a direct attack on an open question the paper poses.

### 1A.8 Lineage: what family of model this is

The dynamics sit in the collective-behaviour literature and inherit intuitions from it. Closest
is the **Krause model** $\dot x_i = \sum_j a_{ij}(x_j - x_i)$ with
$a_{ij} = \varphi(\lVert x_i-x_j\rVert^2)/\sum_k\varphi(\lVert x_i-x_k\rVert^2)$ — non-symmetric
in general, exactly like (SA), and known to produce multi-cluster assembly for compactly
supported $\varphi$. Related: Vicsek, Hegselmann–Krause, Cucker–Smale, and — at $d = 2$,
$\beta = 0$ — the **Kuramoto model** of coupled oscillators. What none of them have is
*parameters inside the nonlinearity*, on the sphere, which is what makes the trained case (this
project's case) new.

The useful transfer is the vocabulary: "consensus" = single cluster, "synchronization" =
collapse, and the well-developed notion that **clustering is generic but the number and lifetime
of clusters is the hard part** — which is precisely Problem 1.

---

## 2. What Pythia actually computes

This section is the other half of the comparison: the literal algorithm, in the order the model
runs it, with every place it departs from §1 flagged. Pythia is GPT-NeoX; the architectural
facts below are read off `core/sublayer_streams.py`, `core/pythia_weights.py`, `core/rope.py`,
`core/ln_frame.py`, and `core/attn_biases.py`, each of which exists because one of these
details was previously gotten wrong in a way that produced plausible, incorrect numbers.

### 2.1 The block: parallel residual

GPT-NeoX with `use_parallel_residual=True` (Pythia's setting at every scale, though the code
reads the flag rather than assuming it) computes:

$$
\boxed{\;x_{\ell+1} \;=\; x_\ell \;+\; \mathrm{Attn}\big(\mathrm{LN}_1(x_\ell)\big)
\;+\; \mathrm{MLP}\big(\mathrm{LN}_2(x_\ell)\big)\;}
$$

Compare GPT-2's **sequential** pre-LN block:

$$
x' = x_\ell + \mathrm{Attn}(\mathrm{LN}_1(x_\ell)), \qquad
x_{\ell+1} = x' + \mathrm{MLP}(\mathrm{LN}_2(x'))
$$

and BERT/ALBERT's **post-LN** block, where each sub-block ends in add-then-LayerNorm so its
output *is* the residual stream.

This difference is not cosmetic for this project. In the parallel form both branches read
**the same input** $x_\ell$, so the decomposition

$$
\Delta x_\ell = \underbrace{\mathrm{attn\_out}_\ell}_{\text{particle coupling}} +
\underbrace{\mathrm{ffn\_out}_\ell}_{\text{no counterpart in the theory}}
$$

is **exact and symmetric** — neither term is downstream of the other, and there is no ordering
confound. Under GPT-2's sequential form the FFN reads a state that attention has already
modified, so "how much of the update was attention" has no frame-independent answer. This is
why `design-1.md` reframes the Pythia port of the attn/FFN energy decomposition as an *upgrade*
rather than a workaround: on Pythia the split the theory needs is available exactly, for the
first time in this project.

`core/sublayer_streams.py` implements the three families' stream semantics separately, and
raises `UnsupportedArchitecture` rather than returning empty for an unhandled one — its
docstring records three prior defects, of which the most instructive is that the GPT-2 path was
capturing *deltas* while the ALBERT path captured *streams*, and the arrays had the right shape
so nothing raised.

### 2.2 LayerNorm: two per block, both reading the same input

$\mathrm{LN}_1$ is `input_layernorm`; $\mathrm{LN}_2$ is `post_attention_layernorm`. **Despite
its name, `post_attention_layernorm` under parallel residual is applied to the same pre-block
input, not to a post-attention state.** Getting this wrong silently changes which frame every
MLP-side quantity lives in. `core/ln_frame.py` encodes this as `which="attn"` →
`input_layernorm`, `which="mlp"` → `post_attention_layernorm`, with `"attn"` the default since
attention is the particle-coupling mechanism.

There is also an **off-by-one** that `core/ln_frame.py::resolve_frame_index` exists to own in
one place: under this project's extraction convention, `hidden_states[L]` is the *output* of
block $L$, and the frame it is about to be read in is block $L{+}1$'s `input_layernorm`. The
last hidden state's reader is `final_layer_norm` — *unless* the extraction path already applied
it, in which case the correct frame is the identity and applying it again would be wrong. Three
cases, one function, so no call site re-derives it. (`core/frames.py` then carries the resolved
answer as data in a `FrameSpec`; §3.2.)

### 2.3 Fused QKV, and the layout trap

GPT-NeoX computes Q, K, V from **one** fused `attention.query_key_value` Linear of shape
$(3\,d, d)$. The memory layout differs from GPT-2's in a way that is easy to get wrong and
produces wrong-but-plausible results:

| | layout of the fused weight |
|---|---|
| GPT-2 `c_attn` | output blocks contiguous **across all heads**: `[Q_all | K_all | V_all]` |
| GPT-NeoX `qkv` | output blocks contiguous **per head**: for head $h$, the $3\,d_{\text{head}}$ slice at offset $h\cdot 3 d_{\text{head}}$ is `[Q_h | K_h | V_h]` |

`core/pythia_weights.py::split_qkv_gptneox` mirrors `GPTNeoXAttention.forward`'s own
`.view(..., num_heads, 3*head_size)` reshape applied to the weight instead of the activation.
Assuming GPT-2's layout on a NeoX model returns the wrong matrix for every head but the first —
this is a bug this project actually shipped, and the module carries a shape cross-check
($3 n_h d_{\text{head}}$ output rows, by construction) that "cannot drift" even if upstream
attribute names change.

The same trap recurs on the **biases** (`core/attn_biases.py`): GPT-NeoX's fused
`query_key_value.bias` indexes as $h\cdot 3d_{\text{head}} + \text{part}\cdot d_{\text{head}} +
t$, GPT-2's as $\text{part}\cdot d + h\cdot d_{\text{head}} + t$, and BERT/ALBERT use separate
per-projection biases.

### 2.4 RoPE: the attention bilinear is *not* $x^\top W_Q W_K^\top x$

This is the biggest single structural gap between the theory's bilinear and Pythia's, and the
one most likely to be silently assumed away.

Every weight-space QK quantity in this project computes $\mathrm{logit}(i,j) = x_i^\top(W_Q
W_K^\top)x_j$. **That identity holds for GPT-2. It does not hold for Pythia.** GPT-NeoX applies
a rotary position embedding to the first `rotary_ndims` dimensions of each head's query and key
*after* projection, so the true bilinear is

$$
\mathrm{logit}(i,j) \;=\; \frac{1}{\sqrt{d_{\text{head}}}}\;
\big\langle \mathcal R_i\, q_i,\; \mathcal R_j\, k_j \big\rangle
\;=\; \frac{1}{\sqrt{d_{\text{head}}}}\; q_i^\top R(j-i)\, k_j,
\qquad
\begin{aligned}
q_i &= W_Q^\top \mathrm{LN}_1(x_i) + b_q\\
k_j &= W_K^\top \mathrm{LN}_1(x_j) + b_k
\end{aligned}
$$

with $R(\Delta)$ block-diagonal and orthogonal, depending only on the **relative** offset
$\Delta = j - i$. So $W_Q W_K^\top$ is $M(\Delta{=}0)$ **only** — the diagonal of the relative
structure, not the whole thing.

The geometry, from `core/rope.py`:

$$
n_{\text{rot}} = \lfloor d_{\text{head}}\cdot \texttt{rotary\_pct}\rfloor, \qquad
\omega_t = \texttt{base}^{-2t/n_{\text{rot}}}, \qquad \theta_t(m) = m\,\omega_t
$$

with `base` $= 10^4$ and — critically — **`rotary_pct` $= 0.25$ on Pythia**, so three quarters
of every head passes through *unrotated*. Assuming full rotary silently changes every
downstream number. GPT-NeoX uses the **half-split** layout, not interleaved: within the rotary
block, dim $t$ pairs with dim $t + n_{\text{rot}}/2$ (HF's `rotate_half`). Pairing $t$ with
$t{+}1$ instead still yields an orthogonal matrix with the right Frobenius norm — i.e. it
produces plausible wrong numbers, which is exactly the failure mode this project keeps running
into.

Two consequences that matter for the particle picture:

- **Position enters the coupling multiplicatively, not additively**, and it enters through a
  factor that is orthogonal — it *rotates* the query/key geometry rather than rescaling it. So
  it cannot change $\lVert q\rVert$ or $\lVert k\rVert$, only their alignment, and only within
  a quarter of each head.
- **Rotary supplies offset-dependent antisymmetry by construction.** On each rotary plane
  $R(\Delta) = \cos(\Delta\omega_t)I + \sin(\Delta\omega_t)J$ exactly, so with $R$ orthogonal
  ($\lVert R\rVert_F^2 = d_{\text{head}}$),

  $$
  \lVert A(R)\rVert_F^2 = 2\sum_t \sin^2(\Delta\omega_t), \qquad
  a_{\text{frac}}(\Delta) = \frac{2\sum_t \sin^2(\Delta\omega_t)}{d_{\text{head}}}
  $$

  which is $0$ at $\Delta = 0$ and rises with $|\Delta|$ (`rope_sa_fractions`). **This is why
  "antisymmetry is elevated for induction pairs" is not evidence of anything until it is
  measured against this baseline** — the architecture hands you offset-dependent antisymmetry
  for free. The live question is whether the *content* bilinear $W_QW_K^\top$ carries
  antisymmetry beyond rotary's contribution at the same offsets.

**A third consequence, and it is a structural departure the paper does not cover.** In the
paper's model, position enters *only through the initial condition* — the token is
$x_i(0) = w_i + p_i$ (or $[w_i;p_i]$), and the dynamics themselves are position-blind and
**permutation equivariant** (Remark 2.2). Permutation equivariance is not decoration: it is what
makes Theorem 6.8's single-scalar reduction work at all (`math-1c.md` §2.1), and it is used
again in the definition of the phase-transition curve. **RoPE breaks it.** Position sits inside
the coupling as $R(j-i)$, so the flow map is no longer permutation equivariant — it is
*translation* equivariant in position instead. Any argument in the paper that leans on
permutation equivariance therefore does not transfer to Pythia as stated, and the causal mask
breaks it a second time and differently. This is worth flagging because the $\gamma_\beta$ null
(Phase 1c) is derived *from* the equivariance argument, so the null is being applied to a model
that violates its hypothesis in two independent ways.

`core/rope.py` also keeps two deliberately separate cost paths: logits are computed by
projecting into head space and rotating the vectors exactly as the model does
($O(nd\,d_{\text{head}} + n^2 d_{\text{head}})$), while S/A fractions use closed-form trace
identities on $d_{\text{head}}\times d_{\text{head}}$ operands —

$$
\lVert M\rVert_F^2 = \mathrm{tr}(R^\top G_Q R\, G_K),\quad
\mathrm{tr}(M^2) = \mathrm{tr}(RCRC),\quad
\lVert S\rVert^2 = \tfrac{\lVert M\rVert^2 + \mathrm{tr}(M^2)}{2}
$$

with $G_Q = W_Q^\top W_Q$, $G_K = W_K^\top W_K$, $C = W_K^\top W_Q$ — so $M(\Delta)$, which
would be $d\times d$ per head per offset, is never materialized. Cost discipline here is what
makes the analysis affordable rather than merely correct.

### 2.5 The softmax: scale, mask, and the position-0 sink

$$
A_{ij} = \frac{\exp\big(\mathrm{logit}(i,j)\big)\cdot \mathbb 1[j\le i]}
{\sum_{k\le i}\exp\big(\mathrm{logit}(i,k)\big)}
$$

Three departures from §1.2's $a_{ij}$:

- **There is no explicit $\beta$.** The inverse temperature is folded into the learned weight
  magnitudes together with the fixed $1/\sqrt{d_{\text{head}}}$ scale. So $\beta$ is not a knob
  we set on a trained model — it is a quantity that must be **estimated** (§3.4). And
  $d_{\text{head}}$ differs across architectures (64 on gpt2-large, 128 on pythia-1.4b), so any
  logit-magnitude quantity must have the scale divided out before cross-model comparison.
- **Causal masking breaks exchangeability.** The theory's particles are exchangeable and
  interact symmetrically; Pythia's token $i$ can only see $j \le i$. The interaction graph is
  lower-triangular, which by itself forces low connectivity (§8.2), and the only offsets with
  non-zero post-softmax weight are $\Delta = j - i \le 0$.
- **Position 0 becomes an attention sink.** NeoX tokenizers do not prepend BOS, so position 0
  takes on sink duty and can carry a residual norm one to two orders of magnitude above every
  other token. That single particle can dominate the raw Gram, dominate $E_\beta$ through
  $\exp(\beta\langle\cdot,\cdot\rangle)$, and dominate clustering. `core/frames.py::pos0_mask`
  makes in/out an explicit, recorded policy applied identically across every model and
  checkpoint — because a trained-vs-random energy contrast where the two sides differ in sink
  structure is partly a sink contrast.

The bias expansion is worth writing out, because dropping it is a first-order error rather than
a rounding one (`core/attn_biases.py`):

$$
\mathrm{logit} =
\underbrace{x_i^\top W_Q R W_K^\top x_j}_{\text{the only term weight-only keeps}}
+ \underbrace{b_q^\top R W_K^\top x_j}_{\text{per-key, query-independent}}
+ \underbrace{x_i^\top W_Q R\, b_k}_{\text{per-query}}
+ \underbrace{b_q^\top R\, b_k}_{\text{constant}}
$$

The second term is a **per-key logit offset applied regardless of what is querying** — which is
structurally exactly the shape of attention-sink behaviour. Measured against true logits on
synthetic weights, dropping the biases alone costs Pearson $\approx 0.99$; compounded with the
frame and rotary omissions it falls to $\approx 0.60$.

### 2.6 The value path: what plays the role of $V$

$$
\mathrm{Attn}(x)_i = \sum_{h} W_{O,h}^\top \sum_j A^h_{ij}\, \big(W_{V,h}^\top \mathrm{LN}_1(x)_j
+ b_{V,h}\big) \; + \; b_{\text{out}}
$$

So the theory's single matrix $V$ corresponds to the **composed OV circuit** $W_V W_O$ per
head, summed over heads — not to a single operator. Two immediate consequences:

- $n_h$ heads means $n_h$ *different* couplings acting simultaneously, each with its own
  attention pattern and its own OV spectrum. "The" attractive/repulsive character of a layer is
  a superposition, and Phase 2's per-head eigenspectrum work is what resolves it.
- Because attention rows sum to one, $\sum_j A_{ij}b_{V,h} = b_{V,h}$ **independent of the
  tokens**, so the value-path bias contributes a fixed write $b_V W_O^\top + b_{\text{out}}$
  every layer — identical for every token. It *displaces the whole cloud* rather than
  restructuring it: another pure common-mode term, in the same family as the LN bias of §1.1
  and detectable by the same $\kappa_1$-vs-$\kappa_2$ split (§5.2).

### 2.7 The MLP: a force with no counterpart in the theory

$$
\mathrm{MLP}(x)_i = W_{\text{out}}^\top\,\mathrm{GELU}\big(W_{\text{in}}^\top \mathrm{LN}_2(x)_i
+ b_{\text{in}}\big) + b_{\text{out}}
$$

Note the index: **the MLP acts token-wise.** It has no $j$ sum, so it is not a particle
interaction at all — it is a per-particle force field. Nothing in the paper's model corresponds
to it. Phase 2 found it to be the *dominant* pathway for the repulsive effect in GPT-2-style
models ("FFN-mediated regime"), which is a result about a term the theory does not contain.
Under Pythia's parallel residual, its contribution is exactly separable from attention's (§2.1),
which is what makes the attribution question well-posed here for the first time.

### 2.8 The complete dictionary

| Idealized model (§1) | Pythia / GPT-NeoX (§2) | Where the gap is handled |
|---|---|---|
| $x_i \in \mathbb S^{d-1}$ | $\mathrm{LN}(x)$ on an ellipsoid, learned $\gamma,\beta_{\mathrm{LN}}$ | `core/ln_frame.py`; §3.2–3.3 |
| $\dot x = P^\perp(\cdot)$, continuous | discrete residual add, $L$ blocks | Euler step $h_\ell$, $T_{\text{eff}}$; §3.5 |
| explicit $\beta$ | folded into weights + $1/\sqrt{d_{\text{head}}}$ | regression estimate $\beta_{\text{eff}}$; §3.4 |
| $Q^\top K = I$ | $W_QR(\Delta)W_K^\top$ per head, + biases | rotary module; QK fidelity score; §3.7 |
| $V = I$ (or $\pm I$) | composed $W_VW_O$ per head, mixed-sign spectrum | Phase 2 (out of scope here) |
| single interaction | $n_h$ heads superposed | per-head everything |
| no FFN | parallel token-wise MLP branch | exact split under parallel residual; §2.1, §2.7 |
| exchangeable particles | causal mask, position 0 is a sink | mask baseline §8.2; `pos0_policy` §3.2 |
| no positions | RoPE on 25% of each head | $a_{\text{frac}}(\Delta)$ null; §2.4 |
| $n$ fixed, $d\to\infty$ | $d = 1024$, $n \in [20, 512]$ | regime question; §1.5–1.6 |

**Every row of this table is a place where a number computed as if the left column were true is
a number about the wrong object.** The point of the next section is that this project's answer
to that is not "be careful" but "record the frame and refuse to compare across frames."

---

# PART II — MEASURING THE DISTANCE

---

## 3. How the deviation from the idealization is quantified

There are six distinct notions of "distance from the theory" in this codebase, and they are not
interchangeable. Keeping them separate is most of the intellectual content of the transition
work.

### 3.1 The failure mode this machinery exists to prevent

From `core/frames.py`'s docstring, and worth quoting because it is the design principle:

> The distance-measurement bug was not a coding error. Every line was correct for the object it
> named; the object was the wrong one for the model. It survived because **no result record
> stated which frame its numbers lived in**, so nothing could contradict anything else.

The patch for any single instance is local; the fix for the *class* is a ledger — every metric
record carries a `FrameSpec`, and any cross-record comparison passes through a guard that
**refuses rather than warns**. One dataclass and one assertion converts a silent failure mode
into a loud one.

### 3.2 The frame ledger

`FrameSpec` (frozen, hashable, serialized into every record) carries:

- `kind` ∈ {`raw`, `l2_sphere`, `ln_attn`, `ln_mlp`, `identity`}
- `layer_idx`, `reader_block` — **data**; records legitimately differ here
- `model_rev` — the checkpoint, not the model name (comparing `step1000` to `step143000` is the
  sweep's purpose; comparing a frame built from final-checkpoint $\gamma$ applied to a
  step-1000 checkpoint is a real and easy mistake)
- `rope_applied` — `False` on a rotary model is a **live claim that the quantity is a proxy**,
  not an accident
- `pos0_policy` ∈ {`included`, `excluded`}
- `ln_eps`

`CONVENTION_FIELDS = (kind, rope_applied, pos0_policy)` are the ones `verify_same_frame`
compares: disagreement on any of them means two numbers are not measuring the same thing
regardless of how close they look, and the guard raises `FrameMismatch` naming the specific
field. `verify_same_revision` is separate, for the reason above. `apply_frame` is the *only*
function that turns raw activations into frame activations — "a call site that normalizes
inline is a call site that cannot be audited."

### 3.3 Distance #1 — the frame gap: L2-sphere vs LN

The original particle frame L2-normalizes the residual stream. But **the model never reads the
raw residual stream**; every sub-layer reads $\mathrm{LN}(x)$, learned $\gamma$ and $\beta$
included. So there are two defensible Gram matrices per layer:

$$
G^{\text{sphere}} = \hat X\hat X^\top,\ \ \hat x_i = \frac{x_i}{\lVert x_i\rVert}
\qquad\text{vs}\qquad
G^{\text{LN}} = \hat Y\hat Y^\top,\ \ y_i = \gamma\odot\frac{x_i - \mu_i}{\sqrt{\sigma_i^2+\epsilon}}
+ \beta_{\mathrm{LN}},\ \ \hat y_i = \frac{y_i}{\lVert y_i\rVert}
$$

`ln_frame_gram` composes LN-then-L2-normalize deliberately, so $G^{\text{LN}}$ drops into every
existing metric unchanged (`ip_mean_ln`, `ip_mass_near_1_ln`, `energies_ln`, …). The learned
$\beta_{\mathrm{LN}}$ **is included on purpose**: the network reads $\gamma\hat x + \beta$, bias
and all, and that shared offset changes pairwise angles — pretending it isn't there would
measure a frame nothing in the model uses.

The *distance* here is the disagreement between the two readings — the "dual reading" of
`core/dual_reading.py` and `DESIGN_dual_reading.md`. Where the two frames agree, a clustering
claim is frame-robust. Where they disagree, something has to arbitrate (§3.8).

### 3.4 Distance #2 — the effective inverse temperature $\beta_{\text{eff}}$

$\beta$ is a free parameter in the theory and an emergent quantity in a trained model (§2.5).
To compare a real model against $\gamma_\beta(t)$ at all, we need its own $\beta$. Take logs of
the softmax:

$$
\log A_{ij} = \beta\, s_{ij} - \log Z_i
$$

so $\beta$ is the slope of $\log A_{ij}$ against the similarity $s_{ij}$ **in the reader's
frame**. Four things make the naive version of this regression wrong, and `core/beta_eff.py`
exists because the shipped version had all four:

1. **Indexing.** The original regressed over `np.triu_indices(n, k=1)` — pairs with query index
   *below* key index, which causal attention masks exactly. Every one of those entries is 0,
   clipped to $10^{-12}$, i.e. $\log A = -27.63$ constant: the fit was a varying $x$ against a
   constant $y$. On synthetic softmax data with a known $\beta = 6.0$, the estimator returned
   $-1.8\times 10^{-14}$. **It had been reporting approximately zero for every head, on every
   model, independent of the data.** `causal_pairs` fixes this, and selects on *original
   sequence position* rather than submatrix order, so restricting to a cluster's members can't
   silently change which pairs count as causal.
2. **Row-varying normalizer.** $\log Z_i$ is per query row and an intercept cannot absorb a
   per-row term; worse, later rows attend over more keys, so $\log Z_i$ correlates with
   position and therefore with offset. The fix is a **fixed-effects estimator** — demean within
   each query row. On the same synthetic data: pooled 5.937, row-demeaned **6.000**. (Demeaning
   rather than dummy columns: numerically identical, two columns instead of hundreds.)
3. **Wrong frame.** $\langle x_i, x_j\rangle$ on L2-normalized residuals is *not* $q_i\cdot k_j$
   — the head reads $\mathrm{LN}_1(x)$ and then projects. The Gram matrix is therefore an
   **argument** to the estimator rather than something it computes, so the frame is the
   caller's explicit, recorded choice (§3.2).
4. **Rotary and scale.** The logit carries $R(\Delta)$, so offset structure loads onto the slope
   unless $\Delta$ is controlled — hence the optional offset regressor. And the reported $\beta$
   is divided by `attn_scale` $= 1/\sqrt{d_{\text{head}}}$, without which values are not
   comparable across architectures with different head widths.

The returned record carries `r2`, `n_pairs`, `structural_zero_fraction`,
`zero_among_causal_pairs`, `scale_applied`, and `frame_required: True` — a standing reminder in
the record itself that the number is meaningless without the `FrameSpec` the caller attaches.
`legacy_beta` is preserved *verbatim* so the correction can be measured against what actually
ran rather than a reconstruction.

### 3.5 Distance #3 — effective integration time $T_{\text{eff}}$ (the depth confound)

This is the sharpest of the six, and status-1 calls it "the single highest-value unrun quantity
in the project."

From §1.2, add-then-normalize is one forward-Euler step of the tangent flow. Reading off the
step size:

$$
h_\ell = \frac{\big\lVert P^\perp_{x_\ell}(\Delta x_\ell)\big\rVert}{\lVert x_\ell\rVert},
\qquad
T_{\text{eff}} = \sum_{\ell} h_\ell
$$

exact for Pythia's parallel residual, where $\Delta x_\ell = \mathrm{attn\_out} +
\mathrm{ffn\_out}$ is available directly. **$T_{\text{eff}}$, not $L$, is the network's clock.**

Why this can invalidate a headline: Blog 1's claim is that trained weights resist collapse, and
that claim silently compares the observed state against $t = \infty$. The correct comparison is
against $\gamma_\beta(T_{\text{eff}})$ — a specific finite number. Integrating (6.9) at
$n = 467$ puts $\gamma_\beta = 0.9$ at $t^\ast \approx 4.2$, near-invariant in $\beta$ across
two decades. So:

- If $T_{\text{eff}} \ll t^\ast$, **the network never runs the dynamics long enough to
  collapse**, and part of what we call resistance is depth.
- If $T_{\text{eff}} \gtrsim t^\ast$ with no collapse, the claim stands and is now quantitative.

The prediction (**P-γ2**) is deliberately stated in the direction that would hurt.

Two calibration findings from `UPDATE_PLAN.md` worth carrying:

- **`MATH.md` §8's step definition understates $T_{\text{eff}}$ by ~5.7×.** It writes $h_\ell =
  \lVert P^\perp(\Delta x_\ell)\rVert/\lVert x_\ell\rVert$ where the numerator is the sphere
  *displacement* — equal to the step size only if $\lVert\mathcal X\rVert = 1$, with the
  paper's bound $\le 1$ and equality only at full collapse. Against an injected $h = 0.0200$,
  the calibrated form recovers 0.0200, §8's form reads 0.0035, the field runs at 0.176. Since
  P-γ2 predicts $T_{\text{eff}} \ll t^\ast$, the mis-definition makes the prediction nearly
  true by construction, in the direction that would make the headline an artifact. Three
  definitions are computed and the verdict *refuses a call* when they straddle $t^\ast$.
- **The calibrated step makes the residual rate-invariant, which is a better result than
  planned.** Damping the field 0.3× gives residual $-0.0009$ — correctly, because damping is
  slower integration, not resistance, and $T_{\text{eff}}$ absorbs it. So the residual measures
  whether the network moves in a **different direction** from the identity-weight field, not how
  much of it it applies. A network that merely attenuates attention reads zero, and should.

### 3.6 Distance #4 — the $\gamma_\beta$ residual (the null model)

With $\beta_{\text{eff}}$ (§3.4) and $T_{\text{eff}}$ (§3.5) in hand, the identity-weight
dynamics make a *pointwise numerical prediction* for the observed mean pairwise inner product at
each layer:

$$
\text{residual}(\ell) \;=\; \texttt{ip\_mean}(\ell) \;-\; \gamma_{\beta_{\text{eff}}}\big(T_{\text{eff}}(\ell)\big)
$$

This is the quantity that separates the two readings of §1.6: if observed `ip_mean` tracks the
curve, the concentration argument survives learned weights and our plateaus are a different
object from the paper's metastability; if it departs, the concentration argument fails and
high-$d$ metastability under learned weights is real. **The deliverable is the residual, not the
fit** — "that gap is the part of the layer-wise dynamics learned weights are responsible for,
and it is the only version of 'resistance' that is a measured quantity rather than a comparison
against an idealization."

One methodological detail worth knowing, because it is a nice piece of engineering: the
per-head $\beta$ reduction (how to collapse many heads' $\beta_{\text{eff}}$ into one number for
the curve) was expected to be a blocking decision, since the measured spread in
$\gamma_\beta(T_{\text{eff}}{=}3)$ across $\beta\in[0.5,5]$ is 0.89 at $n{=}20$ and 0.26 at
$n{=}467$ — larger than any residual we could hope to measure. It stopped being blocking when
$\gamma_\beta(t)$ was verified **monotone in $\beta$** (zero violations over 984,246 grid points
per model under (SA)). Monotonicity means the per-head $\beta$ range *brackets* the null without
any reduction being chosen: the analysis reports an envelope rather than a point estimate, and
where the observed curve falls outside the envelope the conclusion holds for every reduction.

### 3.7 Distance #5 — weight-space fidelity (is the proxy any good?)

Most weight-space analysis in this project computes a *predicted* logit from weights alone.
`qk_prediction_fidelity` scores that prediction against the model's actual logits, restricted by
`causal_pair_mask` to the pairs the softmax actually sees (the upper triangle is never
softmaxed; its disagreement is meaningless):

$$
\text{pearson}, \quad \max|p - a|, \quad \frac{\lVert p - a\rVert_F}{\lVert a\rVert_F}
$$

The intent stated in the module is worth adopting generally: **report this alongside every
weight-space QK claim, so "this is a proxy" becomes a measurement rather than a caveat.**
Rotary-dominated heads degrade; pass-through-dominated heads stay near 1. Paired with
`rope_energy_fraction` (= 0.25 on Pythia — the fraction of each head that can *possibly* carry
position), a reader can see immediately how much of a head a rotary claim can be about.

### 3.8 Distance #6 — functional distance (the arbiter)

All five distances above are geometric. The last one asks a different question: **two particles
are functionally identical iff the LM head decodes them to the same next-token distribution**,
regardless of where they sit geometrically, at what norm, in which frame.

`core/functional_distance.py` builds the full pairwise KL matrix over the vocabulary in a single
matmul — with $P$ the probabilities and $L$ the log-probabilities,

$$
\mathrm{KL}(p_i\Vert p_j) = \sum_v P_{iv}(L_{iv} - L_{jv})
= \underbrace{\textstyle\sum_v P_{iv}L_{iv}}_{-H(p_i),\ \text{a row vector}} - \;(PL^\top)_{ij}
$$

so no pair loop is needed ($n = 512$, $V = 50\text{k}$ is ~13 GFLOPs). Symmetrize as
$(K + K^\top)/2$ — documented explicitly as **not a metric** (triangle inequality fails), used
as a clustering affinity, which is all the arbiter role requires. HDBSCAN on that matrix
produces functional labels in the same $-1$-is-noise convention as the geometric pipeline, and
`frame_agreement` scores pairwise Adjusted Rand Index across `{sphere, ln, functional}`.

The reading: where all three agree, the clustering claim is frame-robust. Where the functional
labeling breaks from both geometric ones, **the geometry is seeing structure the readout does
not** — which is exactly the situation in which a "cluster" might be an artifact of the metric
rather than a fact about the computation.

---

# PART III — THE METRICS, DERIVED

---

## 4. The Gram matrix as the single sufficient statistic

Per layer, `analysis_p1.py` computes the L2-normalized activations once
(`core.models.layernorm_to_sphere`, i.e. `F.normalize(·, p=2, dim=-1)`) and the Gram
$G = \hat X\hat X^\top$ once, then threads both through every downstream metric. This isn't only
an efficiency choice (the docstring notes it removes ~8 redundant matrix multiplies per layer);
it guarantees every metric at a given layer is computed from **the same** pairwise-geometry
snapshot, so cross-metric comparisons at fixed $\ell$ aren't confounded by one metric silently
using a different normalization.

$G \in [-1,1]^{n\times n}$, symmetric, unit diagonal. It is the sufficient statistic for every
rotation-invariant quantity in this phase: energy, mass-near-1, the moment ladder, the spectral
structure, and all the clustering distances are functions of $G$ alone. Effective rank in raw
mode is the notable exception — it depends on the row norms, which $G$ has thrown away, and
that is precisely the source of its pathology (§6.2).

---

## 5. Energy: the exact formula, and the MGF reading

### 5.1 Implementation

`interaction_energy(activations, beta)`: $E_\beta = \frac{1}{2\beta n^2}\sum_{ij}e^{\beta
G_{ij}}$. `interaction_energies_batched(G, beta_values)` vectorizes over
`BETA_VALUES = [0.1, 1.0, 2.0, 5.0]` in one pass.

### 5.2 $E_\beta$ is the moment generating function of the pairwise-cosine distribution

Treat the $n^2$ entries of $G$ (diagonal included) as an empirical distribution: draw $(i,j)$
uniformly and look at $G_{ij}$. Its moments are $m_k = \langle G^k\rangle =
\frac{1}{n^2}\sum_{ij}G_{ij}^k$ (`gram_moments`). Then

$$
E_\beta = \frac{1}{2\beta}\big\langle e^{\beta G_{ij}}\big\rangle_{ij} =
\frac{1}{2\beta}\mathrm{MGF}_G(\beta)
= \frac{1}{2\beta} + \frac{m_1}{2} + \frac{\beta}{4}m_2 + \frac{\beta^2}{12}m_3 + \cdots
$$

**Consequence 1 — the four $\beta$ columns are not four independent facts.** They are four
samples of one scalar distribution's MGF. The non-redundant parameterization is the cumulant
ladder:

$$
\kappa_1 = m_1 \ (\text{common mode}),\quad
\kappa_2 = m_2 - m_1^2\ (\text{spread}),\quad
\kappa_3 = m_3 - 3m_1m_2 + 2m_1^3\ (\text{asymmetry})
$$

**Consequence 2 — participation-ratio rank is exactly the reciprocal second moment.** With unit
rows, $\mathrm{tr}(G) = n$ and $\mathrm{tr}(G^2) = \lVert G\rVert_F^2 = n^2 m_2$, so

$$
\mathrm{PR} = \frac{(\mathrm{tr}\,G)^2}{\mathrm{tr}(G^2)} = \frac{1}{m_2},
\qquad\text{hence}\qquad \frac{1}{\mathrm{PR}} = \kappa_2 + \kappa_1^2
$$

`gram_cumulants` checks this identity to $10^{-12}$ as a self-test. Note this is a *third*
rank-like quantity, distinct from the spectral-entropy effective rank of §6 — see §6.3.

**Consequence 3 — energy changes can be attributed.** Differentiating the two-term expansion:

$$
dE_\beta = \underbrace{d\kappa_1\Big(\tfrac12 + \tfrac{\beta}{2}\bar\kappa_1\Big)}
_{\text{common mode}} + \underbrace{\tfrac{\beta}{4}d\kappa_2}_{\text{spread}}
$$

(`energy_violation_attribution`, with $\bar\kappa_1$ the midpoint across the transition; exact
to second order in $d\kappa_1$). This answers a question a bare violation count cannot: is an
energy drop the cloud's internal structure loosening ($\kappa_2$ falling), or the whole cloud
losing a shared offset with its internal structure untouched ($\kappa_1$ falling)? **The second
is exactly what a learned LayerNorm bias (§1.1) or a value-path bias (§2.6) produces** — so an
unattributed violation count cannot distinguish a geometric event from a bias term.

### 5.3 Two traps in the moment machinery

**The truncation has a hard ceiling in $\beta$.** Measured on $n{=}300$ clouds, relative error
of the two-term form against exact $E_\beta$: 0.00% at $\beta{=}0.1$, 0.07% at $1.0$, 0.80% at
$2.0$, and **26.57% at $\beta{=}5.0$** (three-term: 22.48%); reaching <1% at $\beta = 5$ needs
*twelve* moments. This is the MGF's radius of usefulness, not a bug — at $\beta = 5$ the sum is
dominated by the extreme right tail of the cosine distribution, precisely where low-order
moments carry no information. Since $\beta = 5 \in$ `BETA_VALUES`, **the $\beta{=}5$ column must
remain a measured quantity.**

**Off-diagonal vs full.** The identity is over the full $n^2$ Gram *including* the unit
diagonal, but the persisted `ip_histogram` and `ip_mean` are **off-diagonal only**. Conversion
is exact,

$$
\langle G^k\rangle_{\text{full}} = \frac{1 + (n-1)\langle G^k\rangle_{\text{offdiag}}}{n}
$$

(`offdiag_to_full_moment`), but it is $O(1/n)$: at $n = 20$ the naive off-diagonal $\kappa_1$
reads $+0.0030$ against a true $+0.0523$ — an order of magnitude, on precisely the short prompts
where status-1's only $\beta$ gradient lives. `cumulants_from_ip_histogram` also reports
`quadrature_bias_m2` $=$ (binwidth$^2$/12) $= 1.3\times10^{-4}$ for the 50-bin histogram, so
the discretization error is comparable against whatever residual the energy check produces
rather than assumed negligible.

### 5.4 Violation detection

Relative, not absolute, threshold — necessary because $E_\beta$'s magnitude varies by orders of
magnitude across $\beta$:

$$
\text{violation at }\ell \iff
\frac{E_\beta(\ell-1) - E_\beta(\ell)}{\max(|E_\beta(\ell-1)|,\,10^{-12})} > \texttt{rel\_tol} = 10^{-3}
$$

`sum_severity`/`max_severity` aggregate the relative drops at violating transitions;
`total_rel_change` is the endpoint-to-endpoint change, independent of the count. Note a
violation is an **event between two adjacent layers**, so the natural series is a per-boundary
*indicator*, not a per-layer count — correlating a per-layer regime score against it is
correlating against a boolean, and layer 0 is zero by construction (reported rather than
dropped, so the series stays aligned). Two counters in the codebase currently read "violations"
under slightly different rules (status-1 D7, open).

---

## 6. Effective rank: three different notions, one name

### 6.1 Spectral entropy

$$
\mathrm{erank}(X) = \exp\Big(-\sum_i p_i\log p_i\Big), \qquad p_i = \frac{\sigma_i^2}{\sum_j\sigma_j^2}
$$

the Roy–Vetterli effective rank: a soft count of how many singular directions carry mass. Equals
1 when all mass is on one direction and $\mathrm{rank}(X)$ when the nonzero singular values are
equal.

**`mode="raw"`** runs the SVD on unnormalized activations — mixes directional collapse with
residual-stream norm growth. **`mode="normed"`** runs it after L2-normalizing rows — directional
spread on the sphere alone, the quantity §1 is about.

### 6.2 Why raw mode degenerates into a sink count

Write $n_i = \lVert y_i\rVert$ and $s_{ij}$ the cosines. The raw participation-ratio rank is
$1/\langle s^2\rangle_w$ with weights $w_i = n_i^2/\sum_j n_j^2$ — i.e. **norm-squared
weighting**. In the near-orthogonal limit ($s_{ij}\to 0$ off-diagonal) it tends to

$$
\frac{\big(\sum_i n_i^2\big)^2}{\sum_i n_i^4}
$$

the participation ratio of the **norm distribution alone**, with *zero directional content*.
`norm_participation_ratio` computes exactly this and is persisted next to raw rank, so the
hypothesis is testable rather than inferred: **if raw rank tracks `norm_participation_ratio`,
the reported "rank collapse" is a sink count, not geometry.** Reference numbers from the
derivation ($n{=}200$, $d{=}256$): uniform norms give raw PR 111.9 / norm PR 200.0; three tokens
at 30× norm give raw PR **3.44** / norm PR 3.45 — the directional geometry untouched in both.

Given §2.5 (NeoX has no BOS, so position 0 becomes a high-norm sink), this is not a hypothetical
concern on this architecture. status-1's D1 finds the headline `MinRank` column reads raw mode;
**D10** finds every rank *gate* did too, meaning the set of layers entering every gated
statistic (energy violations, CKA, NN-stability, the Fiedler mean) moved with each checkpoint's
sink structure. Both now read normed mode by policy (`DEGENERATE_RANK_MODE = "normed"`), but the
threshold value (2) was calibrated on the raw scale; `config.py`'s own comment flags it as "the
weakest-justified constant in the phase."

### 6.3 Three rank surrogates, never compared

The codebase contains: (a) spectral-entropy `effective_rank` (raw or normed); (b) the
*uncentered* Gram participation ratio `pr_rank` $= 1/m_2$ (§5.2); (c) the *centered*
participation ratio inside `linear_cka_decomposed` (§11). These coincide only for special
spectra (e.g. flat) and diverge in general — a spectral-entropy rank of 40 does not imply a
participation ratio of 40. Nothing currently plots them against each other on real data (§15,
open question 5).

---

## 7. Fiedler value and eigengap

### 7.1 Definition and the Cheeger intuition

Given a nonnegative affinity $A$, build $L = I - D^{-1/2}AD^{-1/2}$ ($D = \mathrm{diag}(A\mathbf
1)$) with eigenvalues $0 = \lambda_1\le\lambda_2\le\cdots$. The **Fiedler value** $\lambda_2$ is
bounded above and below (Cheeger) by the graph's conductance — the cost of the cheapest
bipartition. So $\lambda_2\approx 0$ ⟹ nearly disconnected components (clusters that barely
talk to each other); $\lambda_2$ large ⟹ free mixing. That is the justification, standard rather
than stated in the code comments, for reading low Fiedler as "attention has organized into
separable groups."

`fiedler_and_eigengap` clips $G$ to non-negative (a negative cosine is not a valid graph
weight), fills the diagonal to 1, and returns $\lambda_2$ plus the eigengap structure. Two
recorded options exist because two modules previously built two different Laplacians on the same
data without recording the difference:

- `connectivity_floor` adds a uniform $\texttt{floor}/n$ weight to every edge. Defensible —
  clipping negatives can disconnect an antipodal graph entirely, leaving $\lambda_2 = 0$ and a
  degenerate eigenspace — but it must be a recorded choice on one code path.
- `clip_negative=False` builds on the **signed** Gram. Explicitly **not a well-posed spectral
  clustering problem** (degrees can vanish or go negative, so the normalized Laplacian can be
  non-finite); it exists because a downstream module genuinely supported it, and silently
  removing a capability is the same class of failure as the duplication it replaced. The code
  guards for non-finite $L$ and returns the standard degenerate result rather than crashing.

### 7.2 Eigengap cluster counting, and why it's dead on Pythia

A $k$-cluster affinity has an approximately $k$-dimensional near-null Laplacian eigenspace: $k$
small eigenvalues, a gap, then larger ones. `k_eigengap` finds the largest jump in the sorted
sequence and counts below it. `sinkhorn_cluster_count_traced` applies the same idea to the
doubly-stochastic attention matrix (§8), reading eigenvalues near 1 as near-invariant subspaces
of the associated Markov chain.

On Pythia the eigengap branch essentially never fires: $k = 1.0000$ in **all 216 runs** at every
reported plateau layer (status-1 D4). Pervasive positive anisotropy means every pairwise inner
product is positive and large, so the clipped Gram is one connected component with no gap. Hence
the branch tracing (`"eigengap"` / `"hard_threshold_fallback"` / `"uniform_spectrum"` /
`"degenerate_n"`) and `sinkhorn_fallback_fraction` in the artifact. The principle, stated across
`INDEX.md` and `status-1.md` and worth generalizing: **any filter, gate, or fallback whose
behaviour depends on the data must record which branch it took** — on an architecture where the
fallback always fires, the fallback silently *is* the metric, and a reader who doesn't know that
reads a dead column as a finding.

---

## 8. Sinkhorn: attention as a Markov operator

### 8.1 Why doubly stochastic

Softmax makes attention row-stochastic but not column-stochastic, so it is not a symmetric
notion of connectivity. Sander et al. (Sinkformers) and the paper's §3.3 motivate the **doubly
stochastic** projection as the object with cleaner gradient-flow structure; the gap between raw
attention and that form is itself a measurement. Sinkhorn–Knopp alternates row and column
normalization,

$$
P \leftarrow \frac{P}{\text{row sums}}, \qquad P \leftarrow \frac{P}{\text{col sums}}
$$

until the max elementwise change falls below `SINKHORN_TOL` $=10^{-6}$, capped at
`SINKHORN_MAX_ITER` $= 500$. The cap was **raised from 100** after status-1's D9: the $n{=}20$
causal baseline alone needs **232** iterations, so every short-prompt run was silently returning
a matrix with residual $4.7\times10^{-4}$ that was not actually doubly stochastic, with no
per-layer residual recorded anywhere to reveal it. `sinkhorn_normalize_with_info` now returns
`converged`/`n_iter`/`residual`, and the batched version additionally returns
`residual_per_head` — because the batched loop breaks on the max residual *across all heads*, so
one slow head holds every head in the iteration and only a per-head residual identifies it.

### 8.2 The causal-mask confound

Causal attention is lower-triangular by construction. Sinkhorn-normalizing and symmetrizing
$(P + P^\top)/2$ on a triangular matrix forces low connectivity **regardless of learned
content**, manufacturing an apparent "100% STABLE-CLUSTER" that is a fact about the mask. Two
corrections:

1. **Baseline subtraction.** Build the content-free uniform-causal attention $A_{ij} = 1/(i+1)$
   for $j\le i$ — what a causal model produces when every QK logit is identical — Sinkhorn it,
   take $\lambda_2$ as `causal_fiedler_baseline(n)`, and classify on the **deviation**
   (actual − baseline). This answers the right question: *does this head route into clusters
   beyond what the mask already forces?* The deviation is legitimately signed, since a head can
   be more connected than the baseline.
2. **Causal-mask control on bidirectional models.** Apply an artificial causal mask to BERT and
   re-run. If BERT's heads also collapse to STABLE-CLUSTER under the control, the GPT-2/BERT
   split is mask-driven, not weight-driven, and the routing claim is withdrawn.

### 8.3 Where the fix's own calibration breaks — a reusable lesson

The CLUSTER/MIXED/MIXING cutoffs (0.3, 0.7) were calibrated for **raw** $\lambda_2$ on $[0,1]$.
After baseline subtraction the deviations live in $\pm 0.05$ and the raw values in
$[0.02, 0.07]$ — measured baselines are $\lambda_2 = 0.0640$ ($n{=}242$), $0.0654$ ($n{=}467$),
$0.0658$ ($n{=}512$), $0.1089$ ($n{=}20$). Both quantities sit **two orders of magnitude below
the CLUSTER cutoff**, so every head classifies CLUSTER at every checkpoint on every prompt: the
"100% STABLE-CLUSTER" result restates the thresholds rather than measuring the model
(status-1 D2).

The $n$-dependence is a second, separable problem: the $n{=}20$ baseline is 1.7× the $n{=}512$
one, and at $n{=}20$ the Sinkhorn fixed point puts **74.5%** of its mass on the diagonal — a
mostly-self-loop graph, qualitatively a different object. Averaging deviations across prompt
lengths averages against different baselines, with a spread comparable to the entire trained
signal. The natural repair is ratio normalization (deviation / baseline), which bounds the
quantity in $[-1,\infty)$ and makes the floor ($\lambda_2 = 0$, total separation) interpretable.

**The generalizable lesson:** a baseline-subtraction fix that changes *what a quantity measures*
does not automatically carry a re-derivation of the thresholds calibrated for the old quantity —
and forgetting to re-derive them turns a real fix into a differently-shaped tautology.

---

## 9. Clustering and cross-layer tracking

### 9.1 Per-layer clustering

Three independent clusterers, all on cosine distance over L2-normed activations:

- **Agglomerative sweep** over 12 thresholds (`np.linspace(0.05, 0.6, 12)`, average linkage,
  precomputed distance). This traces a dendrogram-cut count as a *function of threshold*, not a
  single number; the mid-threshold labels are saved for downstream spatial analysis. (The
  linspace itself is a fixed bug: it was previously `np.linspace(0.05, 0.6, 5, 12)`, where the
  4th positional argument is `endpoint` and 12 was coerced to `True` — producing 5 thresholds,
  not 12.)
- **KMeans + silhouette** over $k\in\{2,\dots,9\}$, keeping the best cosine silhouette. Model
  selection over $k$, not a fixed-$k$ clustering.
- **HDBSCAN** (`min_cluster_size=2`, precomputed cosine). The only one with a native
  noise/unclustered label ($-1$) — which matters enormously here, since the unclustered
  population is roughly half the tokens (Blog 1 Fig. 8, and the whole subject of Phase 5c).
  **Careful with Fig. 8's own reading:** the trained model has *fewer* particles in clusters,
  but **within the variance of the random model** — so population *size* is not a
  trained-vs-random signal. What is trained-specific is where the unclustered particles sit and
  how much attention they receive (`math-5c.md` §2).

HDBSCAN's labels are what tracking and the plateau vote read — not the other two.

### 9.2 Cross-layer identity: Jaccard + Hungarian

Layer $\ell$'s clustering and layer $\ell+1$'s are independent runs, so cluster identity has to
be *constructed*. `match_layer_pair` computes Jaccard overlap over token membership (noise
excluded),

$$
J(C_a, C_b) = \frac{|C_a\cap C_b|}{|C_a\cup C_b|}
$$

and finds the **maximum-weight bipartite matching** via the Hungarian algorithm
(`linear_sum_assignment` on the negated overlap, padded to square), discarding matches below
`min_jaccard = 0.1`. What survives classification:

- **births** — clusters at $\ell+1$ with no adequate match at $\ell$
- **deaths** — clusters at $\ell$ with no adequate match at $\ell+1$
- **merges** — an unmatched cluster at $\ell$ whose best overlap is with an *already-matched*
  cluster at $\ell+1$ (many-to-one)

`track_clusters` chains these into trajectories, threading a `tip_map` from (layer, cluster id)
to trajectory id so each extension is $O(1)$ rather than a linear scan. **This trajectory
structure, not any single layer's labels, is what merge-location analysis and Phase 5's
cluster-identity work are built on** — and it is the only merge instrument that transfers to
Pythia at all, since spectral-$k$ is dead there (§7.2).

### 9.3 Trajectory aggregation, and its behavioral twin

`compute_centroid_trajectories` walks a trajectory's chain, masks its member tokens at each
layer, and takes a **spherical mean** (mean, then renormalize) — matching the geometry the whole
phase works in rather than a Euclidean mean.

`compute_behavior_trajectories` is the same operation on each token's *output distribution*,
using **the same mask**, so activation-side and behavior-side descriptions are guaranteed to be
about the same population at each step. It defaults to averaging in the **Hellinger embedding**
($p\mapsto\sqrt p$, a unit vector, so mean-then-renormalize is the identical spherical-mean
operation) rather than a probability-space mixture. The reasoning is worth repeating because it
generalizes: the arithmetic mean of distributions is the operationally natural reading, but it
is *entropy-increasing* — averaging peaked distributions in probability space blurs them toward
uniform faster than in $\sqrt p$ space, compressing exactly the distances a downstream isometry
test is trying to resolve. Both are available; which was used is recorded, because it changes
the numbers.

The module also returns **coverage** rather than hiding it: a trajectory covered at 1 of 5 chain
layers and one covered at 5 of 5 are not equally good measurements, and collapsing both to "a
distribution" is the kind of silent degradation this project's artifact discipline exists to
prevent. The caller decides whether to drop low-coverage trajectories; the function does not
decide for them.

### 9.4 Nesting and induction-artifact filtering

`multiscale_nesting` asks whether the cluster structure is itself hierarchical — run the
eigengap machinery *within* each HDBSCAN cluster and check whether a coarse global bipartition
($k\le3$ globally) contains sub-clusters with their own internal structure ($k>1$ locally).
That is a genuinely different claim from "there are $k$ clusters."

`pair_hdbscan_agreement` targets a specific failure mode: a **mutual** nearest-neighbour pair
(each is the other's top-1) that HDBSCAN assigns to *different* clusters. That disagreement is a
plausible signature of an induction-style attention artifact — two tokens strongly bound for
positional/copying reasons without being semantically related — rather than genuine geometric
clustering. `emb_gram` (the layer-0 Gram) supplies an "external" semantic reference.

**But that reference trains.** On a checkpoint sweep, an `ext_sem_frac` trend conflates "deep
structure moved away from embedding space" with "embedding space moved" (status-1 D6). The code
now carries an `emb_gram_source` provenance field, and the fix is a frozen reference
(final-checkpoint embeddings, or an external encoder) — recorded, not yet applied. The
`ext_sem_threshold = 0.5` cutoff is likewise inherited from GPT-2/BERT and un-rederived under
Pythia's anisotropy, where nearly every pairwise IP is positive and large.

---

## 10. Operationalizing "metastable plateau"

The most consequential *definitional* choice in the phase, and a heuristic rather than a
theorem-backed criterion — the paper's Problem 1 specifies no operational detector, so this
codebase built one.

### 10.1 Flatness on a growing window

A window is flat when its span is small relative to its mean, or absolutely small:

$$
\mathrm{flat}(v_{i:j}) \iff (\max - \min) < \texttt{abs\_tol}
\ \ \text{OR}\ \ \frac{\max - \min}{|\bar v| + 10^{-8}} < \texttt{tol}
$$

`detect_plateaus` greedily grows a flat window from each unclaimed start index, producing
non-overlapping $(start, end, \text{mean})$ intervals. Tolerances are per-signal, calibrated to
each signal's own scale: mass-near-1 `0.10`, effective rank `0.05`, spectral/HDBSCAN $k$ `0.50`
(near-integer-valued), Fiedler `tol=0.05, abs_tol=0.01`, CKA `0.02`, NN-stability `0.02`. The
`abs_tol` on Fiedler exists because $\lambda_2$ legitimately sits near zero in the
cluster-separated regime, where the relative denominator collapses and the relative test would
never fire.

### 10.2 The multi-signal vote

`compute_plateau_layers` runs the detector independently on **seven** series (mass-near-1,
effective rank, spectral $k$, HDBSCAN $k$, Fiedler mean, CKA-vs-previous, NN stability) and
reports a layer as a plateau layer only if it falls inside at least **two** independently
detected windows. Deliberate robustness: any single metric's plateau could be an artifact of
that metric's own definition (spectral $k$'s dead-metric behaviour being the obvious case), so
agreement across independently-computed signals filters single-metric false positives. The same
computation is used for the saved `plateau_layers` and for what the report prints, so artifact
and text can't disagree.

### 10.3 What this definition does not do

It has **no connection to the paper's mathematical characterization** of a metastable state (a
long-lived approximate stationary configuration of the flow). It is curve-flatness applied to
whatever scalar summaries were cheap. Two consequences:

- It can only ever report *evidence consistent with* metastability — matching, independently,
  the epistemic status §1.6 argues for on theoretical grounds.
- It **inherits every frame problem of its seven inputs**. A plateau detected on raw-mode
  effective rank (§6.2) is evidence about sink structure, not geometry; a plateau on the Fiedler
  mean inherits the moving-baseline problem of §8.3 and the checkpoint-dependent layer filter of
  §14. The vote's robustness is against *independent* noise, and these defects are not
  independent — several of the seven signals are gated on the same rank quantity.

---

## 11. CKA, decomposed

$$
\mathrm{CKA}(X,Y) = \frac{\lVert Y^\top X\rVert_F^2}{\lVert X^\top X\rVert_F\lVert Y^\top Y\rVert_F}
= \frac{\langle G_x, G_y\rangle_F}{\lVert G_x\rVert_F\lVert G_y\rVert_F}
$$

with $G_x = X_cX_c^\top$ the centered Gram — a cosine similarity between two "shapes" of
pairwise structure, invariant to any orthogonal change of basis. `linear_cka_decomposed` factors
it (verified to $10^{-12}$):

$$
\mathrm{CKA} = \underbrace{\frac{\langle G_x, G_y\rangle_F}{\mathrm{tr}(G_x)\mathrm{tr}(G_y)}}
_{\text{overlap}}\;\times\;\underbrace{\sqrt{\mathrm{PR}_x\mathrm{PR}_y}}_{\text{rank factor}}
$$

since $\lVert G\rVert_F = \mathrm{tr}(G)/\sqrt{\mathrm{PR}}$.

The point: **a CKA drop has two logically distinct causes** — the pairwise structure genuinely
reorganized (`overlap` fell), or the effective rank moved (`rank_factor` fell) — and the bare
number cannot distinguish them. Since Phase 1 separately tracks rank collapsing across training,
a CKA-drop finding and a rank-collapse finding are **not independent evidence** unless the rank
factor is divided out. Read `overlap` when the question is "did the representation change
shape."

---

## 12. Code map

| File | What it computes | Notes |
|---|---|---|
| `core/metrics.py` | Canonical scalars: energy, effective rank (raw/normed), Fiedler/eigengap, mass-near-1, NN tracking, CKA (+decomposed), Gram moments/cumulants, energy-drop localization, norm participation ratio | numpy/scipy only; torch never imported (duck-typed `_as_numpy`) so it is oracle-testable in a stubbed session |
| `core/config.py` | `BETA_VALUES`, thresholds, Sinkhorn constants, `DEGENERATE_RANK_THRESHOLD/MODE`, dtype policy | Every data-dependent constant carries a comment on whether it is *calibrated* or merely *placed* |
| `core/ln_frame.py` | LN transform, LN-frame Gram, GPT-NeoX LN parameter extraction, the hidden-state→reader-block off-by-one | Pure/extraction split; `resolve_frame_index` is the single home of the off-by-one |
| `core/frames.py` | `FrameSpec`, `apply_frame`, `pos0_mask`, `verify_same_frame`/`verify_same_revision` | The ledger. Only place activations get transformed |
| `core/rope.py` | Rotary frequencies/angles, `apply_rope`, `rope_rotation`, closed-form S/A fractions, logits-with-rope, prediction fidelity | Two cost paths on purpose; never materializes $M(\Delta)$ |
| `core/pythia_weights.py` | Fused QKV split with per-head NeoX layout + shape cross-check | Source of a previously shipped bug |
| `core/attn_biases.py` | Q/K/V and output biases across three layouts; the token-independent OV drift | Three disagreeing conventions |
| `core/beta_eff.py` | $\beta_{\text{eff}}$ regression with causal pair selection, row fixed effects, offset control, head-size scaling | `legacy_beta` preserved verbatim for the diff |
| `core/sublayer_streams.py` | post-attention / post-FFN residual streams per architecture family | Raises rather than returning empty for unsupported families |
| `core/functional_distance.py` | Pairwise KL by matmul identity, symmetrized affinity, functional clustering, pure-numpy ARI, `frame_agreement` | The arbiter between geometric frames |
| `analysis_p1.py` | The per-layer loop: `normed`/`G` once, then every metric | Records `gate_rank`, `gate_rank_mode`, `gate_passed` per layer so gating is reconstructible from the artifact |
| `sinkhorn.py` | Sinkhorn–Knopp + Fiedler + causal baseline + branch-traced cluster count | Convergence info returned, not discarded |
| `clustering.py` | Agglomerative/KMeans/HDBSCAN, PCA/UMAP, nesting, induction-pair tagging | |
| `cluster_tracking.py` | Jaccard+Hungarian matching, trajectory chains, centroid and behaviour aggregation | |
| `reporting_p1.py` | `detect_plateaus`, `compute_plateau_layers`, the cross-run report | The cross-run report is the primary downstream artifact |
| `p1_io.py` | Artifact contract: one JSON per metric family + `.npz` arrays; run discovery/loading | |
| `run_1.py` | CLI orchestrator, random baseline, sublayer mode, ALBERT snapshot mode | |
| `visualization/checkpoint_*.py` | The checkpoint-sweep layer: aggregates a run series into the developmental arc | |

---

## 13. What was found

### 13.1 The original cross-architecture study (Blog 1)

- Clusters are real and persistent: ~50% of tokens HDBSCAN-clustered at any layer, consistently
  across models and prompts.
- **Trained weights resist collapse; random weights don't.** Random GPT-2-large pushes nearly
  all pairwise inner products above 0.9 by layer 5; the trained model's histogram stays spread.
  Energy rises monotonically under random weights and does *not* under trained weights — for
  every model, prompt, and $\beta$ tested.
- Trained models have **more, smaller, better-separated** clusters; random models have fewer,
  larger, tighter ones.
- Effective rank collapses under random weights and is maintained under trained ones (ALBERT
  maintains it exactly to its trained depth, then falls — running it deeper is out of
  distribution).
- Fiedler values are lower (more separable routing) under trained weights: training **cuts** the
  architecturally-uniform attention graph rather than assembling connections.
- Unclustered tokens absorb a growing share of attention in trained models (≈90% of attention
  mass on ≈50% of tokens by late layers) — absent in the random case.

### 13.2 The Pythia-410M checkpoint pilot: the object of study is a trajectory

27 checkpoints × 8 prompts = 216 runs, plus 27 degenerate-input controls. **"Trained vs random"
is one point on a developmental curve, and the curve is non-monotone in every component.** Four
transitions, at four different times, that any single-checkpoint study necessarily bundles:

1. **Steps 8→16 — transient late-layer collapse.** Raw rank 6.5→2.1, max mass 0.016→0.58,
   confined to the top of the stack, fully recovered by step 512. Unpredicted; possibly LR
   warmup rather than a training event.
2. **Steps 256→512 — the energy break.** 21→64 violations in one interval. **Monotonicity is
   destroyed by training, not by randomization** — it holds cleanly at steps 0 and 8. The
   GPT-2-era "falsified universally, including under random weights" was a statement about
   GPT-2's *initialization*, and cited the wrong theorem.
3. **Step 512 — plateau onset flips weight-level (SD 0.00) → content-driven (SD 3.31).**
4. **Steps 1000→3000 — Fiedler deviation crosses zero** and saturates near $-0.023$ by step 40k.

Effective rank collapses (16), recovers (512), **overshoots its initial value 3×** (3000–5000),
then declines for 140k steps. Reading any of these off two endpoints gives the wrong sign.

Two results outside the prediction list:

- The `repeated_tokens` control — deliberately excluded from the metastability analysis, since a
  degenerate input tests collapse *speed*, not metastability — carries the cleanest single
  result in the sweep. At init the network leaves a degenerate input degenerate (final mass
  0.948, rank 1.11); by step 143000 it actively **separates** it (0.379, 2.02), onset ~11k–13k.
  **Training installs a separating force** — the direct empirical counterpart to the
  attractive/repulsive tension. It would have been invisible had the control been folded in.
- Mid-network mass at step 143000 drops to 0.0007 against a layer-0 duplicate-token floor of
  0.0149 — a factor of **20 below** the embedding floor. The trained model separates even
  identical tokens by mid-depth.

Also: cluster carrying capacity is invariant (max-alive 50–55 across all 27 checkpoints) while
turnover is not (mean lifespan 7.0→4.5, births 113→164).

---

## 14. Where the measurement frame breaks

Every item is a correct formula fed the wrong input, or a gate reading the wrong signal.

- **Raw vs normed rank** (§6.2) — the `MinRank` column and, worse, *every rank gate* read raw
  mode, so the layer set entering every gated statistic moved with sink structure. D1 is a
  re-report (normed rank is on disk); D10 changes which layers enter everything.
- **Fiedler mislabel + vacuous thresholds** (§8.3) — the reported `MeanFiedler` carries the
  *deviation*, and the cutoffs classify every head identically.
- **A silent layer-inclusion fallback** — the per-head Fiedler profile filters to raw rank ≥ 10
  then falls back to *all* layers when none qualify, which fires exactly at steps 16–32, with
  nothing in the output distinguishing a filtered profile from a fallback one.
- **Mass-near-1's max-over-layers reduction is floor-dominated** — outside the transient window
  it equals the layer-0 duplicate-token fraction, a prompt-determined constant (`wiki_paragraph`
  reads 0.0148 at step 0 and 0.0149 at step 143000). The real signal is the mid-network
  *minimum*, which the reduction discards.
- **`ext_sem_frac`'s reference frame trains** (§9.4).
- **`sinkhorn.json` doesn't persist what the report reads** — `fiedler_per_head`,
  `fiedler_per_head_deviation`, `fiedler_baseline` are computed and never written, so the entire
  per-head section silently returns empty on any reload. It exists in the pilot's output only
  because that report was written in-session from memory. **This is why D1 costs a re-report and
  D2 costs a rerun.**

The standing rules this cycle produced, each earned from a defect: *if a quantity appears in a
report, it is persisted*; *every data-dependent fallback records the branch it took*; *every gate
records which quantity it read and whether it passed, per layer*; **refuse rather than degrade**
(no unit-norm substitute for missing norms, no inferred revision, no invented $\beta$, no silent
raw-frame fallback — "a number from mismatched inputs is worse than no number: it is
unfalsifiable from the output alone"); *anchors need a non-symmetric arm* (an anchor that only
tests the identity case tests almost nothing about a bilinear form — a wrong trace contraction
passed its $M = I$ sanity check while being wrong for every real head, reading $-72.08$ against
a true $+167.00$); *a threshold not derived from a distribution is labelled as placed, not
calibrated, in the code next to the value*.

---

## 15. Open questions

Tracked already:

1. Is the step-8→16 transient a training event or an LR-warmup artifact? Resolved by a single
   interval; the recommended 10/12/24/48 densification would settle it.
2. Why does violation **severity** peak at step 60k and decline to 143k while the **count** stays
   flat from 19k? Count and magnitude come apart and nothing explains it. The natural instrument
   is exactly the parallel-residual attn/FFN energy decomposition (§2.1) that Pythia uniquely
   permits.
3. Is $T_{\text{eff}} \ll t^\ast$ or $\gtrsim t^\ast$ (§3.5)? Highest-value unrun quantity, at
   report-only cost — every input is already on disk.
4. Claim (c): does the phenomenology transfer across architecture, and at 1.4B? Hard-stop gate,
   still armed.

Surfaced by writing this document:

5. **Do the three rank surrogates (§6.3) agree on real data?** Nothing plots spectral-entropy
   rank, uncentered `pr_rank`, and the centered CKA participation ratio against each other
   layer-by-layer. If they diverge, "effective rank" as a single reported number is
   underspecified — and the sphere theory arguably prefers the normed spectral entropy, since it
   is the only one manifestly invariant on the sphere without an extra centering step.
6. **Does the plateau vote count carry information beyond the binary $\ge 2$?** A layer where
   all seven signals agree is presumably stronger evidence than one that scrapes by with two,
   and the pipeline discards the gradation at the threshold. Worth checking whether vote count
   correlates with anything downstream (cluster lifespan, merge-event coherence) that would
   justify keeping it as a continuous confidence.
7. **The common-mode/spread energy attribution (§5.2) has never been run on the sweep.** It
   exists and is tested, but the developmental table reports counts, not the split. Given that
   *both* the LN bias (§1.1) and the value-path bias (§2.6) are pure common-mode terms, and that
   the severity-decline puzzle (item 2) is precisely a magnitude question, this is a
   already-built instrument sitting unused on the one question it best fits.
8. **Is the rank overshoot (3× above step-0, at steps 3000–5000) the same mechanism as the
   `repeated_tokens` separating force?** Both are training *adding* directional spread rather
   than removing it, and both land in a nearby window, but nothing connects them beyond
   co-occurrence.
9. **Does `ip_mean` on Pythia look like a failed attempt to concentrate onto $\gamma_\beta$, or
   like never approaching it at all?** These are different residual shapes — growing-then-
   saturating vs. large from layer 0 — and distinguishing them (once Phase 1c supplies
   $T_{\text{eff}}$) bears directly on which reading in §1.6 is right.
10. **How much of the observed "resistance" survives the frame correction?** Blog 1's headline
    was measured entirely in the L2-sphere frame on GPT-2, without rotary (not applicable),
    without LN $\gamma$, and with position 0 included. Three of those change on Pythia. The
    frame ledger now makes it *possible* to ask how much of the trained-vs-random contrast is
    frame-dependent — and since `pos0_policy` alone can move the raw Gram substantially when one
    particle carries 30× the norm, "is the resistance result frame-robust?" is an answerable
    question that has not been asked in either direction.
11. **Rotary's antisymmetry baseline (§2.4) is derived but the corresponding null for the
    *coupling* is not.** `rope_sa_fractions` gives the exact $a_{\text{frac}}(\Delta)$ the
    architecture supplies for free. The analogous question for Phase 1's own quantities — how
    much of the observed pairwise structure at offset $\Delta$ is what rotary alone would
    produce on random content — has no null model yet, even though the ingredients (the closed
    form, and `qk_offset_null.py`) exist.

Surfaced by reading the paper's measure-level sections (§1A):

12. **The per-token partition function $Z_{\beta,i}$ is a metric weight, and nobody has looked
    at it.** §1A.6 shows (SA) is a gradient flow of $E_\beta$ in the metric
    $\langle a,b\rangle_X = \sum_i Z_{\beta,i}\langle a_i,b_i\rangle$ — so $Z_{\beta,i}$ is
    *how expensive it is to move token $i$*. High-$Z$ tokens are pinned by the geometry. That is
    an unnervingly good description of what attention sinks do, and $Z_{\beta,i}$ is a
    one-line computation from the Gram matrix we already build every layer. Two cheap questions
    follow: does $Z_{\beta,i}$ identify the same tokens as the norm outliers driving raw
    effective rank (§6.2)? And is the *dispersion* of $Z_{\beta,i}$ across tokens — i.e. how far
    the metric is from the standard product metric — a better summary of "this layer is
    sink-dominated" than any of the rank surrogates?
13. **The Lyapunov identity gives an energy-violation diagnostic we are not computing.** §1A.2:
    $dE_\beta/dt = \int\lVert\mathcal X\rVert^2 Z_{\beta,\mu}\,d\mu \ge 0$ in the attractive
    case, sandwiched between $e^{-\beta}$ and $e^{\beta}$ times $\int\lVert\mathcal X\rVert^2$.
    So at every violation layer there is a *predicted* magnitude for the energy change, and the
    observed change can be compared against it rather than merely sign-checked. Phase 1c already
    computes $\lVert\mathcal X\rVert$ per token (`field_magnitude`) for the step calibration;
    crossing it with the per-layer $\Delta E_\beta$ Phase 1 already stores would turn the binary
    violation indicator into a signed, scaled residual — for free, from two quantities that
    exist and have never met.
14. **Are the plateaus "metastable" in the gradient-flow sense — i.e. near-critical points?**
    §1A.5 makes (USA) a gradient ascent on $E_\beta$, and the proofs of Theorems 4.3/5.1 turn on
    *strict saddle points* of $E_\beta$: for a.e. initial condition the flow avoids them, but it
    can dwell near one for a long time, which is the standard mechanism for metastability in
    gradient flows. That gives an operational test the current flatness detector (§10) does not
    have: a genuine metastable state should show $\lVert\mathcal X\rVert$ small (near-critical)
    while the configuration is *not* collapsed. Our plateau detector looks only at flatness of
    downstream summaries; **"is the field small here?" is a different and more principled
    question, and it is one function call away.**

---

## 16. Paper map

Geshkovski, Letrouit, Polyanskiy, Rigollet, *A Mathematical Perspective on Transformers*,
arXiv:2312.10794v5. Where each result this project leans on actually lives, since several were
mis-cited in earlier drafts and the corrections are load-bearing.

| Result | Content | Used here |
|---|---|---|
| §2.2, eq. (2.3) | the model: $\dot x_i = P^\perp_{x_i}\big(Z^{-1}_{\beta,i}\sum_j e^{\beta\langle Qx_i,Kx_j\rangle}Vx_j\big)$ | §1.2 |
| §2.2 (RMS-norm justification) | ALBERT-xlarge-v2 diagonal: mean 0.44, sd 0.008 | §1.1; the sphere *licensing check*, `math-1c.md` §6.1 |
| §2.3.1–2.3.2 | multi-head and FFN written down, then **excluded** from all Part 1–2 theory | §2.7; why `h_attn_only` is frame-correct |
| Remark 2.2 | permutation equivariance | §2.4 — **broken by RoPE and by causal masking** |
| §3.1, eq. (3.4) | continuity equation / measure flow | §1A.1 |
| eq. (3.6) | the exact Lyapunov identity for **(SA)** | §1A.2; open question 13 |
| **Prop. 3.4** | extremizers: uniform is the unique min, Diracs are the maxima. *Static* — says nothing about trajectories | §1A.3. **Mis-cited in earlier drafts as the monotonicity result** |
| eq. (3.7) | $\mathcal X[\mu] \propto \nabla\log\int e^{\beta\langle x,y\rangle}d\mu$ — the log-derivative obstruction | §1A.4 |
| Remark 3.5 | Sinkformers: doubly-stochastic attention *is* a Wasserstein gradient flow; clustering there is **open** | §1A.7, §8 |
| **Lemma 3.6** | (USA) is the Wasserstein gradient flow of $E_\beta$ | §1A.5 |
| **Lemma 3.7** | Lyapunov identity for (USA) | §1A.5 |
| §3.4 | (SA) is a gradient flow in the $Z_{\beta,i}$-reweighted metric, **iff $Q^\top K$ symmetric and $V = Q^\top K$** | §1A.6 — **this is P-M1** |
| Thm 4.1 / 4.2 / 4.3 | $\beta = 0$ consensus; $\beta\to0$ limit; $\beta\lesssim n^{-1}$ single cluster | context for the low-$\beta$ column |
| Thm 5.1 | $\beta\gtrsim (d-1)n^2$ single cluster | context for the high-$\beta$ column |
| **Thm 6.1** | $d\ge3$, **any** $\beta$ ⟹ single cluster. Qualitative — no rate, no $d$-dependence | §1.5. **A "higher $d$ → faster" verdict row was retracted for testing a claim this does not make** |
| **Thm 6.3** | $d\ge n$ ⟹ exponential rate, $\lambda = O(e^{-\beta})$ | §1.5; hypothesis holds for every prompt, rate **never measured** |
| **Lemma 6.4** | cone collapse; proof uses *only positivity of $a_{ij}$*; explicit $\dot\alpha\ge(1-\alpha)/(2ne^{2\beta})$ | `math-1b.md` §1.2, `math-1c.md` §7.1 |
| Thm 6.7 | Wendel: $P = 2^{-(n-1)}\sum_{k<d}\binom{n-1}{k}$, $=1$ for $d\ge n$ | `math-1c.md` §7.2 — **P-H1** |
| **Thm 6.8** | orthogonal init ⟹ one scalar $\gamma_\beta(t)$; eq. (6.9) SA, (6.10) USA | `math-1c.md` §2.1 |
| **Thm 6.9** | concentration onto $\gamma_\beta$ at $d\gg n$, w.p. $\ge 1-2n^2d^{-1/64}$ | §1.5; `math-1c.md` §2.1 — **P-γ1/P-γ2** |
| §6.3, **Problem 1** | metastability is **open**, supported only by $d=2$ numerics at $\beta = 4, 9$ | §1.6 — the epistemic status of this whole project |
| Fig. 3 | the $(d,\beta)$ phase diagram; metastable band vanishes by $d\approx512$ | §1.6 — we run at $d = 1024$ |
| Fig. 5 / Problem 2 | phase diagrams survive for random $(Q,K,V)$; generalization of 6.8–6.9 is **open** | the project's premise |
| §8, Problem 4 | BBGKY hierarchy; closure ansatz open | not pursued |
| §9.1, Def. 9.1, **Thm 9.2** | $V=-I$; $E_\beta$ as a Gaussian-kernel energy; global minima are **sharp configurations** (Cohn–Kumar) or the 600-cell | `math-1c.md` §8 — **P-S1** |
| §9.2, **Table 1** ([GLPR24]) | rescaling $z_i = e^{-tV}x_i$; row 2: $\lambda_1(V)>0$ simple **and** $\langle Q\varphi_1,K\varphi_1\rangle>0$ ⟹ three parallel hyperplanes | Phase 2/2d — **P-T1**, incl. the amendment that adds the QK condition |
| Problem 5 | extending Table 1 to other $(Q,K,V)$ is **open** | Phase 2's actual territory |

**Four of this project's six registered predictions test statements the paper poses as open
problems, not theorems** (Problem 1 for metastability, Problem 2 for general matrices, Problem 5
for Table 1's generality, and Remark 3.5's Sinkhorn question). That is the right thing to be
doing — but it means the correct framing throughout is *evidence bearing on open conjectures at
parameters far outside where their supporting numerics live*, never *replication*.
