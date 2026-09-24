<!-- p9_metric_intervention/notes-9.md -->
# Phase 9 — NOTES (pre-design, nothing frozen)

**Read this header before anything below it.**

This is a **workshop record, not a design**. It came out of one session
(2026-09-18) of conversation and one pass over the tree to check which of its
claims survive contact with what is already built. No construction here is
frozen, no instrument is specified to the level `design-N.md` requires, and
`claims/registry.json` is untouched.

**`CLAUDE.md` trigger 1 applies and has NOT been discharged.** A phase opening
is one of the two moments where a literature scan still changes a decision, and
it runs *before* `design-9.md` freezes any construction. This file deliberately
stops short of that so the scan is still able to do its work. §11 names the
searches.

---

## 0. The object

Every intervention this project has run so far acts on **weights** — zero- and
mean-ablation, the `2^2` sign factorial (`MATH_SPECTRAL_OT.md` §2.4.2), the
isometric path (§2.5), rank truncation (`p7e_consolidation/useful_rank.py`).

Phase 9's object is different: **intervene on the geometry the dynamics is read
in, not on the operator that drives it.** The residual stream's metric is a
choice the architecture already makes and trains, and changing it changes what
attention sees without touching a single weight in `W_QKV` or `W_OV`.

Two motivating questions from the session, in the user's framing:

1. Can a collapse be *induced on purpose*, on an isolated structure, as an
   augmentation — the rough analogy being making a cell into a stem cell?
2. Can a local region be made *less regular* — distances stretched — to
   accelerate or decelerate a subspace, with the region located by SAE features
   or any activation-space isolation?

They are the same question from opposite signs: (1) drives a structure toward
degeneracy, (2) drives it away.

---

## 1. Two different collapses, and they must not be fused

This was the session's first correction and it governs everything after it.

- **Collapse of particles** (Geshkovski et al.; this project's Blog 1) is a
  statement about *token representations* under a fixed operator as depth
  → ∞. It is a property of the **measure**.
- **Degeneracy in SLT** is a statement about the *parameter–loss geometry* —
  directions along which the loss does not change, so the parameters are not
  determined. APD/SPD components are parameter-space objects and sit here.

Rank deficiency in `W_OV` produces both, which is why they feel like one thing.
They are not, and this repo has already measured them coming apart:
**`MATH_SPECTRAL_OT.md` §2.5.6** — the isometric path preserves every singular
value exactly, `t = 0` and `t = 1` are spectrally identical by construction, and
copying is broken at `t = 1`. **A collapse specified spectrally is not the
collapse that carries the function.** Any Phase 9 construction must say which of
the two it acts on.

---

## 2. "Revert the training" is not well-posed

SGD is not invertible and basins are not traversable backwards. The
implementable operation is:

> **project** the isolated component onto a degenerate submanifold (rank
> truncation, symmetrisation, re-randomisation confined to its subspace,
> zeroing a decomposition component's contribution), **then re-apply pressure**
> — continue training under a gradient you choose.

**Collapse is a release operation, not a deletion one.** What refills the freed
capacity is decided by the post-collapse gradient, not by the collapse. Stated
as a design rule: *collapse removes a capability only together with a change of
pressure; on the original distribution it is a plasticity operation.*

SLT's free-energy argument points the same way — lower-RLCT points are
entropically cheap, so a collapsed component re-inflates only if accuracy
demands it. **Caveat carried deliberately:** that is a Bayesian-posterior
statement and SGD's relation to it is empirical, not a theorem. It also cannot
be instrumented here — `INDEX.md` drops LLC/SGLD as out of compute range, and
that decision is not reopened by this file. Rank and participation-ratio proxies
are what is affordable.

---

## 3. What this repo has already measured that bears on it

Both principal obstacles are already on disk, which is the unusual advantage of
running this idea *here* rather than somewhere else.

**(a) Self-repair.** 44/45 pairwise co-ablation cells positive at step 16000,
no block structure (`PROJECT.md` §3.12-V). The Hydra Effect (`2307.15771`)
predicts exactly that sign, and the 2026-09-10 scan classified it as **not
new**. Collapsing one site does not remove a function — the set re-forms it, and
under continued training it re-forms it faster. Collapsing "the set" instead
runs into `status-7e.md`: members are near-orthogonal (`L11H14` mean cosine
**−0.033** to the set) and write into **private low-variance bandwidth** — 90 %
of joint effect needs **355 ambient directions** against an ambient
participation ratio of 22. The set is not a small subspace.

**(b) Identification.** `MATH_SPECTRAL_OT.md` §2.4.6: **no weights-only spectral
quantity computed by this project identifies the copier.** `L7H8` and `L2H10`
agree to three digits on every eigenvalue-derived field and their causal effects
have opposite sign. So "isolate the structure" cannot be done from the geometry
this project measures; it has to be causal, or come from a decomposition whose
components are *defined* by causal faithfulness.

**(c) The precedent that should set the prior on SAE-derived subspaces.**
`status-7e.md`: for `L11H14`, `useful_rank.py --bottom` finds bottom-`r` >
matched-random > top-`r` **at every rank**, and at `r = 1` recovery is
**−0.096** — keeping its top singular direction is *worse than deleting the
head*. The Eckart–Young optimal approximation, provably the best approximation
of the **operator**, is the **worst** of three at preserving the **function**.

Structural proxies have now failed against causal ground truth twice here.
**SAE features are fitted to reconstruct activations sparsely; there is no
reason that objective aligns them with slow directions of a dynamical
operator.** Treat an SAE-derived subspace as a *candidate generator*, never as
an identifier. `OVERVIEW.md` also records that sparse dictionary decompositions
of internal activity were already tried in this project and "came back null or
were shelved" — local precedent, and it is not encouraging.

---

## 4. The levers the paper actually exposes

| lever | what it does | where it lives here |
|---|---|---|
| `beta` | softmax inverse temperature. `gamma_beta` monotone **decreasing** in `beta` for (SA), **increasing** for (USA) — the partition function reverses the sign | `p1c_frames/beta_reduction.py`; per-head via `core/beta_eff.py`. 984k grid points, zero violations |
| `T_eff` | depth, or equivalently field amplitude | `p1c_frames/integration_time.py`; `t* ~ 4.2` |
| `V` | **`V = -I_d` flips the sign of the whole Lyapunov identity**, so `E_beta` strictly *decreases*. Exact, not an approximation (`math-1.md` §1A.2) | the S/A split; §2.4.2's four corners are its discrete version, §2.5's path the graded one |
| `Q, K` | **cannot** break Lemma 6.4 — only positivity of `a_ij` is used | `math-1c.md` §7.1 |
| initial configuration | the cone condition | `p1c_frames/hemisphere_feasibility.py`, exact QP |
| the frame | LN's learned diagonal; the LN **bias** inflates `<G>` by `||beta_LN||^2` — an energy floor with nothing to do with the tokens | `p1c_frames/frame_table.py`, `bias_energy_floor` |

Not in the paper, and therefore where resistance lives: the MLP (a token-wise
force field, not a particle interaction), causal masking and RoPE (both break
permutation equivariance, Remark 2.2 — and equivariance is what makes Theorem
6.8's single-scalar reduction work at all).

**The `T_eff` result worth restating, because it is the whole basis of §6.**
`math-1c.md` §1.4: damping the field by 0.3× produces a residual of **−0.0009**,
essentially zero, because *damping is not resistance, it is slower integration*
— `h_calibrated` absorbs it into a shorter `T_eff`. Run backwards: **scaling an
attention branch's write by `c` runs that layer's dynamics `c` times longer.**
That is the "simulate the passage of time" knob, it is local, and Pythia's
parallel residual makes the attention/MLP split exact.

---

## 5. The hemisphere lever, and the obstruction that gates it

Lemma 6.4: all `x_i` in an open hemisphere ⟹ exponential collapse to a point.
**Only positivity of the attention weights is used**, so it holds for arbitrary
`Q, K` with `V = I`. The hypothesis is entirely a condition on the
*configuration*; no choice of QK weights can rescue or break it. **Corollary
worth carrying: the QK circuit is powerless against collapse-from-a-hemisphere.
Everything that resists must come from `V` or from outside the paper's model.**

The instrument is exact and already written (`math-1c.md` §7.3):

    m = dist(0, conv{x_i}) = sqrt( min_{lambda in simplex} lambda^T G lambda )

Cone condition holds iff `0` is not in the convex hull. Needs only `G`, is
`n`-dimensional regardless of model width, and the optimal `lambda`'s support
names the **binding tokens** — so the intervention has a closed-form target,
which is rare.

**The obstruction is Carathéodory, not training.** For `0` in the convex hull of
points in general position you need `n >= d + 1`. Wendel gives probability 1
whenever `d >= n`, and `math-1c.md` §7.2 notes `d > n` holds for *every* prompt
in the current grid (`n` in [20, 512], `d = 1024`). **You cannot push a
512-token cloud off the hemisphere in a 1024-wide stream by moving points; it is
geometrically impossible, not merely hard.** The escapes are:

- run contexts past `n = d + 1` — **pythia-70m has `d = 512` and a 2048 window,
  so this is cheap and reachable**, and is the natural first rung;
- accept an exact degeneracy at small `n` (antipodal pairs, a lower-dimensional
  positive dependency), which is why §7.2 already insists: **report the margin,
  not the boolean.** A near-zero margin is the informative case and attention
  sinks are the plausible source.

---

## 6. The metric deformation: LN's gamma already is the lever

The session's framing question was "we are not actually on a hypersphere." The
precise version is better than that:

`math-1c.md` §6.2 — `ln_plain` (LN with `gamma = 1`, `beta_LN = 0`) is **exactly**
sphere projection in the mean-zero subspace, `LN(x) = sqrt(d) P_1 x / ||P_1 x||`,
constant norm `sqrt(d)`. **LN does put you on a sphere.** What takes you off it
is the *learned diagonal* `gamma`, which makes the true state space a
time-varying axis-aligned ellipsoid — and §6.1 notes the paper itself does not
assume the sphere, it measures it and sets that matrix to `I`.

So the metric deformation is already in the architecture, is trained, and sits
exactly where it is wanted: **attention reads `LN(x)`, so the inner product
driving every softmax logit is `gamma`-weighted.** `gamma` is diagonal, hence
axis-aligned, hence cannot stretch an arbitrary subspace. The generalisation is
one line:

    diag(gamma)  ->  diag(gamma) + U D U^T

with `U` spanning the isolated subspace. Rank-`k` anisotropic, one layer,
read-side only. No new plumbing — it is a modification to `core/ln_frame.py`'s
frame.

---

## 7. Three insertion points, and they are not equivalent

> **Corrected 2026-09-20** (`plan-9.md` §4.1, `docs/readings/2411.04990.md`):
> a γ-patch on `input_layernorm` feeds `W_Q`, `W_K` and `W_V` alike, so it
> cannot separate these points. The read side alone needs a different lever.

- **Read-side (before QK).** Changes the attention *pattern*: inner products in
  the stretched subspace shrink relative to everything else, logits shift, those
  tokens attend to each other less. This is "increase the distance so they
  cluster less", literally.
- **Write-side (OV output).** Changes the *displacement*, not the pattern. This
  is "accelerate/decelerate a subspace" and by §4's `T_eff` result it is exactly
  a subspace-restricted integration time.
- **State-side (the stream itself).** Changes everything downstream, MLP
  included.

**§2.5.6 is the reason this distinction is load-bearing rather than pedantic:**
read/write alignment carries real causal weight beyond the spectrum —
`(M_r)^T` preserves the truncated spectrum exactly and copying stays broken
(`1.040218472480774` both ways, cross-checked against an independently built
transpose). A Phase 9 construction that does not name its insertion point is
under-specified.

---

## 8. The OT reframe — the part that makes this a phase rather than a knob

`MATH_SPECTRAL_OT.md` §6: **there are two spectra and the project computes one.**

- Eigenvalues of `V` linearise the motion of a particle in `R^d`. That is
  `p2_eigenspectra`.
- **Metastability is a property of the measure**, governed by the **Wasserstein
  Hessian of `E_beta`**: near-zero eigenvalues are slow directions, i.e.
  long-lived states, and the escape rate over a barrier is set by the single
  negative eigenvalue at the saddle (Eyring–Kramers). The number of small
  eigenvalues counts the metastable states; the sign structure of the
  eigenvectors *is* the partition into them.

Stated in that frame, the session's proposal stops being a heuristic:

> **"isolate a subspace and stretch it" becomes "find a near-zero eigendirection
> of the Wasserstein Hessian and change its curvature."**

That is the operator whose spectrum defines the word this project has been
using, and it makes a *quantitative* prediction — stretching along a slow
direction should move the implied timescale `t_i = -1/log|lambda_i|` by a
computable amount.

> **CORRECTION 2026-09-20, from reading `2411.04990` — the ensemble form of
> this is not available under a causal mask, and the per-token form is.**
> `plan-9.md` §5.1a, `math-10.md` §7.5, `lit-10.md` §11.5. The masked dynamics
> has **no single global potential**, so "the Wasserstein Hessian of `E_beta`",
> "the number of small eigenvalues counts the metastable states" and "the sign
> structure of the eigenvectors *is* the partition" do not transfer as written.
> What the paper proves instead is that the causal dynamics is a **sequential
> gradient flow**, `phi_k' = -(1/Z_k) dE_k/dphi_k`, with a different energy
> `E_k` per particle. **So the curvature claim survives in per-token form** —
> about `d^2 E_k / d phi_k^2`, causally ordered — and that is what this section
> should say. Narrower, more specific, still testable. The `1/Z_k` prefactor is
> the quantity Phase 10's F12 measured, which makes `math-1.md` §1A.6's metric
> reading of `Z` an equation rather than an interpretation.

**This is also the honest answer to "is any of this new?"** The intervention is
not: SAE feature clamping, activation steering, concept erasure (LEACE) and
representation engineering are all populated, and scaling a feature's activation
is routine practice. What is not populated is treating the rescale as a **metric
deformation the collapse dynamics runs in**, and **predicting its effect on
cluster structure from the theory before running it.** The comparative advantage
is `gamma_beta(t)`, the monotonicity envelope, a measured null and an energy
functional with a closed-form gradient — not the knob.

> **REWRITTEN 2026-09-20 after `2605.12765` (GUARD-IT) was read as primary
> text** — `lit-10.md` §14.1. `lit-10.md` §6's earlier `[S]`-grade reading put
> the differentia on *"GUARD-IT applies a rotation, which is an isometry; a
> gamma-patch is a congruence"*. **Two thirds of that does not survive contact
> with the paper, and what remains is cleaner.**
>
> - **GUARD-IT's Eq. 8 is not a rotation.** It is
>   `h' = (h - alpha*v_hat) * ||h|| / ||h - alpha*v_hat||`: a nonlinear,
>   input-dependent self-map of the sphere of radius `||h||`. Norm preservation
>   is exact; "pure rotation in the residual stream" is the paper's prose, and
>   §6 repeated it.
> - **"Exactly invertible" is not a differentia either.** Given `v_hat` and
>   `alpha`, Eq. 8 inverts in closed form (`||h|| = ||h'||` is known, and the
>   remaining scale is the positive root of a quadratic). Any part of this
>   phase's novelty claim resting on invertibility should be **dropped**.
> - **What survives is an object-level distinction, and it is sharper.**
>   GUARD-IT acts on the **state** `h`; a gamma-patch acts on the **bilinear
>   form** `W_QK` — a congruence, `plan-9.md` §4.1. One moves where this token
>   is; the other changes how *every pair* of tokens is compared. GUARD-IT
>   selects by **content** (a similarity gate over forget-corpus embeddings,
>   and it does nothing at all when the gate is empty); a gamma-patch selects by
>   **geometry** and always applies. **State versus operator is the claim to
>   make.**
>
> **And `2505.16831` (ICML 2026) adds a requirement, not a threat.** Task-level
> metrics *"can be misleading, as models can appear to forget while their
> original behavior is easily restored through minimal fine-tuning"*. **Any
> Phase 9 forgetting result must carry a relearning arm** or it measures the
> thing that paper says is routinely mismeasured. Their four-regime taxonomy
> (reversible/irreversible x catastrophic/non-catastrophic, Definition 2.1) is
> the frame; §2's *"collapse is a release operation, not a deletion one"* is the
> same claim from the other side and should cite it. `plan-9.md` §5.3 carries
> the readout rows.

---

## 9. Correction — what is already built, and two stale statements

Checked against the tree on 2026-09-18, because the session's first draft of
this plan got it wrong in the direction that would have wasted a week.

**`core/dissipation.py` exists and is substantial.** It implements
`energy_gradient` (the closed form `dE_beta/dx_i = (1/n^2) sum_j exp(beta
<x_i,x_j>) x_j`), `dissipation`, `dissipation_by_channel` (the exact attn/FFN
split on the parallel residual), `dissipation_by_subspace` through Phase 2's
Schur projectors, and `gradient_flow_alignment`.

**And it has been run.** `PROJECT.md` §3.8.2 — Tier A
(`data/analysis/dissipation_series.json`) over the 19 x 7 grid, ~22 min, no
forward pass; Tier B (`dissipation_sublayer_series.json`) with the exact
attn/FFN split and a per-head roll-up; plus a `v2` variant. Real findings
already attached, including step 512 as a triple co-location with the
repulsive-subspace share of `|dissipation|` jumping 0.45 → 0.67.

> **Two statements are therefore stale and are recorded here rather than
> silently dropped.** `MATH_SPECTRAL_OT.md` §5.2's "Nothing in the repository
> computes it" was true when written and is not now. And the session's own
> first-draft recommendation — "build the energy gradient first, it is free and
> unblocks everything" — was wrong for the same reason. The gradient is
> available as a scoring function today.

**What is built and has NOT been run:** the transport half of the same module.
`w2_identity`, `w2_optimal` (exact linear assignment via
`scipy.optimize.linear_sum_assignment`), `sliced_w2`, `wasserstein_arc_length`
and `straightness` are implemented and tested (`tests/test_core_dissipation.py`)
and **no runner in `tools/run/` calls any of them** — verified by grep; the only
two files mentioning them are the module and its test. No result in `PROJECT.md`
or `docs/` reports a `W_2`, an arc length or a straightness. **That is the real
free action, not the gradient.**

Why it matters specifically for this phase: the gap between `w2_identity` (token
`i` to token `i`, an upper bound, the current convention) and `w2_optimal` (the
true coupling) measures **how much of a layer's displacement is tokens swapping
places — motion that leaves the distribution unchanged — versus genuine motion
of the measure.** For an intervention whose entire purpose is to spread a region
apart, that is the discriminator. Nothing else in the project separates the two.
And `straightness` (endpoint `W_2` over arc length) gives dwelling measured **on
the measure**, rather than inferred from whether HDBSCAN found clusters — which
is the readout for "did I decelerate this subspace."

---

## 10. Candidate work, ordered by cost and by what unblocks what

Ordering is a proposal, not a decision, and none of it is registered.

1. **Run the transport observables on artifacts already on disk.** No forward
   pass, scipy is a dependency, the code is written and tested. Produces the
   identity-vs-optimal gap, arc length, and straightness per layer per
   checkpoint. Needs a `tools/run/transport.py` on the pattern of
   `tools/run/dissipation.py`. **Cheapest real thing in the phase.**
2. **The cone margin at `n > d`.** `hemisphere_feasibility.py` already solves the
   QP; the gap is that no prompt in the grid has `n >= d + 1`. On pythia-70m
   (`d = 512`, 2048 window) it is reachable. Report the margin and the binding
   token set, not the boolean.
3. **The anisotropic frame intervention.** `diag(gamma) + U D U^T` at one layer,
   read-side, sweeping `D`. Readouts: `gamma` concentration, `E_beta`,
   per-token `<G_i, v_i>` from the existing dissipation module, and the
   transport quantities from (1). **Write the `gamma_beta` ODE prediction down
   before the run** — `p1c_frames/gamma_ode.py` supplies it, and §4.1's measured
   spread across `beta` in [0.5, 5] is **0.89 at n = 20**, larger than any
   residual worth reporting, so it must be read against the *envelope* rather
   than a point estimate or the effect is unreadable.
4. **Collapse-and-regrow**, if the phase goes that way at all: collapse an
   identified component at checkpoint `k`, resume training, ask whether the
   structure re-forms, where, and how fast. This is the one use of induced
   degeneracy that fits the project's own open question — §2.4.6 ends "the
   cross-sectional question has a negative answer; the developmental one is
   where the structure is." It is also **much** more expensive than 1–3 and
   needs the fork-retrain protocol (`status-8.md`; note the existing 70m retrain
   is a **fork**, not pythia-70m, and the two must stay labelled apart).
5. **Sinkhorn-normalising one head** (Remark 3.5: doubly-stochastic attention
   *is* a Wasserstein gradient flow, and clustering there is **open** in the
   paper). Converts a head into an object the theory covers exactly. Different
   in kind from 1–4: theorem-adjacent rather than heuristic.

**Standing constraint, inherited:** `MATH_SPECTRAL_OT.md` §6.1 — this is a new
subphase directory importing existing outputs **read-only**. Nothing here is
added as a stage inside `p2_eigenspectra/run_2.py`.

---

## 11. Before any of this becomes `design-9.md`

The literature scan (`CLAUDE.md` trigger 1) has not run. The searches that would
change the design rather than the citations:

- metric / frame deformation as an interpretability intervention, as against
  activation steering and feature clamping — **the novelty claim in §8 rests on
  this and is currently unverified**;
- induced degeneracy, targeted re-initialisation, plasticity restoration
  (dormant-neuron resets, later-layer re-initialisation) — prior art for §2 that
  the session named from memory and did not check;
- parameter decomposition (APD / SPD) — **attribution was uncertain in session
  (Apollo vs Goodfire) and must be confirmed before it is written down
  anywhere**;
- Wasserstein-Hessian / Eyring–Kramers spectra applied to transformer
  representations, which is §8's load-bearing framing;
- Sinkformers and what is known about clustering under doubly-stochastic
  attention since Remark 3.5.

`arxiv.org` and other scholarly hosts are blocked by the session egress proxy
(`docs/LITERATURE.md` records the same constraint), so any scan run from here
yields leads marked `[S]`/`[N]`, not readings.

---

## 12. What this phase is not

- Not a claim that induced collapse removes a capability. §3(a) says the
  opposite is the default expectation.
- Not a registration. No `P-*` id names anything here.
- Not a licence to quote an SVD-ordered truncation as "the" collapse —
  `status-7e.md`'s hold stands: **any SVD-ordered rank truncation misleads on
  `L11H14`-like heads**, and `--bottom` must be checked first.
- Not a revival of LLC/SGLD. That decision stays as `INDEX.md` records it.
