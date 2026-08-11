# Phase 2d — DESIGN

## Core question

Phase 1 reports energy-monotonicity violations rising from 3 at step 0 to 64 by step 512 and
reads that as the paper's theorem failing. But the theorem has hypotheses, and this project has
never checked whether the model satisfies them.

Phase 2d checks them, and turns three more of the paper's results into falsifiable claims about
quantities already on disk.

## Why a Phase 2 extension rather than part of 1c

Everything here needs $M_h = W_Q^{(h)\top}W_K^{(h)}/\sqrt{d_h}$ and $W_{OV}^{(h)}$, which is
Phase 2's output. Phase 1c is deliberately Phase-2-independent so it can run now; 2d is not.

**Sequencing: 2d runs after 1c-B lands.** If $T_{\rm eff} \ll t^\ast$, the network never
integrates far enough for the asymptotic energy argument to bind, and attributing a
monotonicity break becomes attributing something that was not going to happen anyway. The
$T_{\rm eff}$ result determines whether the break is the right thing to explain.

## The four sub-experiments

### D1 — the gradient-flow condition (P-M1)

§3.4 makes (SA) a gradient flow in the reweighted metric
$\langle a,b\rangle_X = \sum_i Z_{\beta,i}\langle a_i,b_i\rangle$ **only when $Q^\top K$ is
symmetric and $V = Q^\top K$**. Heads meeting both must show monotone $E_\beta$; heads far
outside carry no guarantee at all.

So the question stops being "is the theorem violated" and becomes "which heads are outside its
hypotheses, and do the violations localize there?" Measured per head: asymmetry
$\|\mathrm{Skew}(M_h)\|_F/\|M_h\|_F$, and the signed Frobenius cosine between $W_{OV}$ and
$M_h$.

Three design points that are not obvious from the plan's one-line description:

**The cosine is signed, deliberately.** It is scale-invariant, so it tests $V \propto Q^\top K$
rather than equality — the right relaxation, since a positive rescaling of $V$ rescales time in
the ODE without changing the gradient-flow structure or the sign of $dE/dt$. A *negative*
constant does change it: that is the $V = -I_d$ repulsive case. So `align` near $-1$ is
recorded as its own regime (`repulsive_aligned`) where a violation is the *predicted*
behaviour, not as "aligned".

**Alignment is reported against $M$ and against $\mathrm{Sym}(M)$ separately.** If $M$ is far
from symmetric then no symmetric $V$ can match it, and a head can fail the condition because
$V$ is wrong or because $M$ was never symmetric. Different failures; the plain cosine conflates
them.

**A continuous `regime_distance` alongside the four-way label.** P-M1 is a correlation claim,
and a two-bin split discards most of the power.

### D2 — operator-conditioned rank

$$\mathrm{PR}_M = \frac{(\mathrm{tr}\,MC)^2}{\mathrm{tr}(M^\top C M C)}$$

with $C$ the token covariance. This is the bilinear pairing of the operator spectrum against
the activation covariance spectrum: $\mathrm{tr}(M^\top CMC) = \sum_{ab}\lambda_a\lambda_b
|\langle u_a, Mu_b\rangle|^2$. Phase 2 has the left factor, Phase 1 has the right, and nobody
has computed the product.

Heads with large $\|M\|$ and small $\mathrm{PR}_M$ are strong operators pointed where the
tokens are not. That is a candidate explanation for a specific unexplained Phase 1 observation
— the $\beta$-independence of violations after step 512. If $M$ concentrates on few directions
the higher moments collapse and only $\langle G\rangle$ survives, which would make the
violation count insensitive to $\beta$ exactly as observed.

Sanity anchor, checked rather than assumed: at $M = I$, $\mathrm{PR}_M$ reduces to
$(\mathrm{tr}\,C)^2/\mathrm{tr}(C^2)$, the ordinary participation-ratio rank. It does, to
1e-8.

`coupling_efficiency` = $\mathrm{PR}_M/\mathrm{PR}_C$ is reported because $\mathrm{PR}_M$ alone
conflates "the head is selective" with "the cloud is low-rank", and separating those is the
entire point of the pairing.

### D3 — Table 1 as a geometric prediction (P-T1)

Table 1 (§9.2) maps a classification we already have — the sign and multiplicity of
$\lambda_1(V)$ per head — onto a statement about activations. Row 2: a real, simple, positive
top eigenvalue predicts concentration on three parallel hyperplanes normal to $\varphi_1$, i.e.
trimodality of $\langle\varphi_1, x_i\rangle$.

**P-T1 as registered omits half of row 2's hypothesis.** Table 1 requires
$\langle Q\varphi_1, K\varphi_1\rangle > 0$ as well, i.e. $\varphi_1^\top M \varphi_1 > 0$. A
head with a positive simple top eigenvalue but a negative QK form is not in row 2 at all, and
testing it would falsify a prediction the paper never made — the same error the retracted
"Thm 6.1 unsupported" verdict row made. Both conditions are checked and heads only count as
row-2 candidates when both hold; `row2_eigen_only_qk_fails` is a distinct label.

**The rescaling caveat is also the falsifier.** Table 1 describes the limit geometry of
$z_i = e^{-tV}x_i$, not of $x_i$, and $t$ is not observable from a fixed-depth network. So the
raw test is a weaker related claim. `rescaled_modality` scans a few candidate $t$: if
trimodality appears at some $t$ and not at $t{=}0$, the structure is real and the rescaling is
what hides it — a different conclusion from "Table 1 does not transfer", and one the raw test
cannot reach. The eigenvector condition number is reported per $t$, since the amplification
grows with $t$ and a rescaling through an ill-conditioned basis produces noise.

**The adjudicator requires a control arm.** Trimodality rate is reported among row-2 candidates
*and* among non-candidates. If non-candidates are trimodal at the same rate, trimodality is a
property of the activations rather than of the classification, and a candidates-only number
would read as confirmation.

### D4 — the model's own energy

$E_\beta^{(h)} = \frac{1}{2\beta}\langle\exp(\beta\,y^\top M_h y')\rangle$ on LN'd states, with
the first-order term $\bar y^\top M_h \bar y$ reported alongside the identity-weight $E_\beta$.

The first-order sign *is* the attractive/repulsive call for that head, computed from the
model's own operator rather than from the $Q^\top K = I$ proxy. `monotonicity_compare` then
does the thing that has never been done: count violations under both energies and report the
ones that exist under the proxy and **disappear** under the head's own energy. Each of those is
a Phase 1 violation that was an artifact of the substitution.

## Two numerical points

**The generalized energy overflows without a shift.** At $\beta = 5$ on $d = 1024$ with an
untamed $M$, $\exp(\beta y^\top M y')$ exceeds float64 — and it surfaces as `inf` in one head of
one layer rather than as an error, which is how a NaN reaches an aggregate. The exponent is
shifted by its maximum and carried analytically, with `overflow_guarded` flagged per $\beta$.

**Rows are normalized before the energy is computed.** The paper's $E_\beta$ is defined for
unit-norm particles; leaving the norms in makes the exponent scale with $\|y\|^2$, which for a
transformer's growing residual stream would make the energy a norm measurement wearing a
geometry costume. `norm_cv` records the size of what was removed.

## What this cannot settle

D1's correlation is between a **per-layer** violation count and an aggregate of **per-head**
regime scores, because there is no per-head energy. The aggregate is a choice, and the answer
depends on it — mean, min and max are all reported, and if they disagree in sign then P-M1 is
not adjudicable from per-layer energies and needs head ablation. That is a real result about
the experiment's resolution and is recorded as one rather than resolved by picking whichever
aggregate confirms.
