# Phase 1c — DESIGN

## Core question

Blog 1's headline is that the trained network resists the collapse its architecture drives.
That claim silently compares the observed state against $t = \infty$. It has never been
compared against what the paper's own dynamics would do **in the amount of time the network
actually runs them**.

Phase 1c supplies that comparison. It is entirely re-analysis of artifacts already on disk
plus small weight-only computations.

## Why a separate phase

Phase 1's loop is a per-run analysis over a model×prompt grid; its unit of work is one forward
pass. Everything here is trajectory-level — its unit of work is the whole checkpoint series at
once. Folding it into `analysis_p1.py` would put series-level code inside a per-layer loop. It
also has its own falsification structure (P-γ1, P-γ2, P-H1, P-S1, registered in
`PREDICTIONS.md` before any of this was written), which is what has always earned a phase
directory here.

**Depends on:** Phase 1 artifacts (`geometry.json`, `energies.json`, `activations.npz`,
`sublayer_streams`), `core/ln_frame.py`, `core/beta_eff.py`, `core/functional_distance.py`.
**Does not depend on** Phase 2. Runnable now.

## The construction

A residual block is a forward-Euler step of the paper's ODE
$\dot x_i = P^\perp_{x_i}\big(\sum_j a_{ij} x_j\big)$. Three objects follow:

1. **Step size** $h_\ell$ — how far one block moves the state, in the ODE's own time units.
2. **Effective integration time** $T_{\rm eff} = \sum_\ell h_\ell$ — compared against
   $t^\ast \approx 4.2$, the time at which $\gamma_\beta$ reaches 0.9 at $n = 467$ (P-γ2).
3. **Residual** $\;{\rm ip\_mean}(\ell) - \gamma_{\beta_{\rm eff}}(T_{\rm eff}(\ell))$ — the
   part of the layer-wise trajectory *not* explained by identity-weight dynamics run for the
   observed time (P-γ1).

The residual is the deliverable. The fit is not.

## Three things that changed during implementation

### 1. The step-size definition decides the answer, and `MATH.md` §8's understates by ~6×

§8 writes $h_\ell = \|P^\perp(\Delta x_\ell)\|/\|x_\ell\|$. That is the **displacement on the
sphere**, which is the numerator of the Euler step — it equals the step size only if
$\|\mathcal{X}\| = 1$. The paper's bound is $\|\mathcal{X}\| \le 1$, with equality only for a
fully collapsed cloud; for a spread cloud the field is far weaker.

Measured directly: integrating the true (SA) field forward at a known step $h = 0.0200$ from
an orthogonal init at $n = 40$, $d = 512$, $\beta = 1$:

| quantity | value |
|---|---|
| injected step $h$ | 0.0200 |
| recovered `h_calibrated` | **0.0200** |
| `h_displacement` (§8 as written) | 0.0035 |
| mean $\|\mathcal{X}\|$ over the trajectory | 0.176 |
| **understatement factor** | **5.67×** |

And the recovered $T_{\rm eff}$ tracks the ODE: at $\gamma = 0.9$, $T_{\rm eff} = 3.040$
against the ODE's $t^\ast = 3.015$ — 0.8%, which is Euler discretization error.

This matters because the bias has a direction. P-γ2 predicts $T_{\rm eff} \ll t^\ast$; the §8
definition makes that nearly true by construction, so adopting it would come close to
confirming the prediction by choice of units, in the direction that says our headline result is
an artifact of depth. All three definitions are therefore computed
(`h_displacement`, `h_calibrated`, `h_attn_only`) and `verdict()` reports `robust=False` when
they straddle $t^\ast$ — in which case the answer is a definition, not a measurement, and must
be reported that way.

### 2. The FFN is not in the model being compared

The paper writes the feed-forward layer down in §2 and then excludes it; every theorem in
Parts 1–2 is single-head, no-FFN. Using the full block delta credits the ODE with motion
produced by a term that is not in it. Pythia's parallel residual
($\text{out} = x + \text{attn}(\text{ln}_1 x) + \text{mlp}(\text{ln}_2 x)$) makes the split
exact with no ordering confound — the one place GPT-2's sequential architecture could not have
supported this. `h_attn_only` is the frame-correct variant; the others are upper bounds.

### 3. The calibrated step makes the null rate-invariant, which is a stronger result than planned

Damping the field by 0.3× produces a residual of $-0.0009$ — essentially zero. That is
correct, and it sharpens what the residual means. Damping is not resistance; it is slower
integration, and `h_calibrated` absorbs it into a shorter $T_{\rm eff}$. So the residual does
not measure *how much* of the identity-weight field the network applies. It measures whether
the network moves in a **different direction** from that field.

That is the better notion of resistance, and it was not what the plan specified. A network
that simply attenuates attention would show zero residual and should: it is running the
paper's dynamics, just fewer of them, and the depth explanation would be the correct one. A
network with a genuinely different vector field shows a nonzero residual — verified at
$-0.0113$ for a trajectory perturbed orthogonally to the field.

## A limitation, and the companion measure it forces

$\gamma$ is a sigmoid asymptotic to 1, so once the null exceeds ~0.95 the vertical residual has
almost no dynamic range. On synthetic runs a perturbation that visibly altered the dynamics
still registered a final residual of $+0.0000$ purely because both curves were against the
ceiling.

Inverting the null removes the compression:

$$\text{time\_residual}(\ell) = t_{\rm null}^{-1}\big({\rm ip\_mean}(\ell)\big) - T_{\rm eff}(\ell)$$

— how much time the identity-weight dynamics would need to reach the clustering we observe,
against how much the network actually spent. Negative means the network spent longer than the
null needs, i.e. resistance. On the same synthetic pair where the vertical residual read
$+0.0001$ / $-0.0113$, the time residual reads $+0.002$ / $-0.669$: two and a half orders of
magnitude more separation. Both are reported. The vertical residual is interpretable in units
of inner product; the time residual is the one that stays honest late in the stack, which is
where the trained-model question lives.

Layers whose observed `ip_mean` falls below the null's starting point are recorded as `nan` and
counted, not clipped. An observed value below $g_0$ means the network **de-clustered past its
own starting point** — the strongest possible resistance signal — and clipping it to zero would
silently render it as "on schedule". Phase 1 already found mid-network mass dropping a factor
of 20 below the embedding floor, so these layers are expected and are the interesting ones.

## Three assumptions carried, and how each is handled

**$\beta_{\rm eff}$ is not constant across layers.** The null is evaluated per layer at that
layer's own $\beta$, with the run median as fallback and the fallback count recorded. A
single global $\beta$ would fold the QK circuits' layer-wise variation into the residual and
attribute it to the weights' resistance.

**Theorem 6.8 assumes orthogonal initialization.** Real layer-0 embeddings are not orthogonal;
they carry a large common mode. A second null is integrated from the *observed* layer-0
`ip_mean`, and `anisotropy_gap` reports the mean distance between the two. Where they disagree,
the disagreement is an anisotropy effect and not resistance — a distinction the single-null
version cannot make.

**The paper's field is non-causal; a decoder-only transformer is not.** `sa_field(causal=)`
exposes both. `causal=True` is the honest comparison for Pythia and is the default, but it is a
departure from the theory, and the masked field is systematically weaker (fewer terms, and
early tokens see almost no one). Whichever is used for $T_{\rm eff}$ must be used for the null;
the choice is recorded in the artifact.

## Validation

`gamma_ode.py` reproduces all 28 entries of `MATH.md` §3.2's collapse-time table — both (SA)
and (USA), $n \in \{20, 467\}$, $\beta \in \{0.1, 1, 2, 5\}$, at $\gamma = 0.5$ and $0.9$ — to a
maximum absolute deviation of **0.005**. Step size is validated by halving until
$t(\gamma = 0.9)$ stops moving, and a run that never converged reports `converged=False` rather
than silently returning the coarsest answer.

The (SA) right-hand side is written with the exponentials factored to a common scale; the naive
form of eq. (6.9) overflows float64 at $\beta = 5$ with $\gamma$ near 1 and large $n$, which is
one of the corners the table covers.

## What A and B settle, and what they do not

They settle whether the resistance claim survives being made quantitative. They do **not**
address the $d = 1024$ tension `status-1.md` now opens with — whether our plateaus are the
paper's metastability at all. That needs sub-experiment F (spherical designs) and Phase 2d's
Table 1 tests, since the question is about limit *geometry*, not rate.

Sequencing note from the update plan stands: **2d waits on B**, because the $T_{\rm eff}$
result determines whether the energy-monotonicity break is even the right thing to attribute.
