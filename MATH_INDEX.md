# MATH — study index

Ten per-phase documents deriving the mathematics behind this project and mapping it onto the code.
Written to be read with the source open. Each lives in its phase's directory:

| Doc | Phase | Lines | What it covers |
|---|---|---|---|
| `p1_mstate_tracking/math-1.md` | 1 | 1700 | **The theory doc.** Particle model, optimal-transport/Wasserstein framing, the GPT-NeoX forward pass, the six distance measures, and every Phase 1 metric derived |
| `p1b_hemisphere/math-1b.md` | 1b | 627 | Cone collapse vs bipartition; the Fiedler axis; the margin LP |
| `p1c_frames/math-1c.md` | 1c | 901 | **The quantitative core.** $T_{\rm eff}$, the $\gamma_\beta$ ODE and residual, frames, Wendel/hull duality, spherical designs |
| `p2_eigenspectra/math-2.md` | 2 | 566 | The OV mechanism: Schur decomposition, attractive/repulsive subspaces, rescaled frames, the V-score |
| `p2b_imaginary/math-2b.md` | 2b | 499 | Rotation; the withdrawn identity; Henrici; the LN Jacobian |
| `p2d_operator_activation/math-2d.md` | 2d | 440 | The paper's *hypotheses*, finally checked: gradient-flow condition, operator-conditioned rank, Table 1 |
| `p5_single_mstate_analysis/math-5.md` | 5 | 456 | One cluster end to end; the particle table; the blocker class |
| `p5b_manifold_steering/math-5b.md` | 5b | 329 | Manifold isometry; Hellinger/Fisher–Rao geometry; pre-registration done right |
| `p5c_unclustered/math-5c.md` | 5c | 337 | The unclustered population; the rank-budget hypothesis; the attention flip |
| `p6_subspace/math-6.md` | 6 | 386 | The S/A division-of-labour hypothesis; the inverted first run |

Phases 2c, 3 and 4 are out of scope per `INDEX.md` (2026-07-18) and have no document.

## Reading order

**If you want the theory:** `math-1.md` §1 → §1A → `math-1c.md` §2 → §8.
**If you want to know what the model actually does:** `math-1.md` §2 → §3.
**If you want the mechanism:** `math-2.md` → `math-2b.md` → `math-2d.md`.
**If you want to know what to run next:** the open-question register below.

`math-1.md` is the prerequisite for everything else; §1A (optimal transport) and §2 (the Pythia
forward pass) are cited constantly by the others.

---

## Six recurring failure patterns

These recur across phases with different surface symptoms. Recognizing the shape is worth more
than any individual fix.

### 1. The test that cannot come out the other way

A null result reported as evidence when the measurement was constant by construction.

| instance | why it couldn't fire |
|---|---|
| `elim_rotation = 0.0`, 35/35 runs (2b §3.1) | $e^{-A}$ is orthogonal; every measured quantity is Gram-based and Gram is orthogonally invariant |
| `strong_bipartition` at 0% (1b §4.4) | requires centroid angle $\ge\pi/2$, which the cone margin bounds away |
| `V_repulsive_via_attn` never fires in 35 runs (2 §10.4) | branch precedence — a higher-priority verdict always claims the case first |
| `ln_curvature` regression, Pearson always NaN (2b §4.3) | $\kappa \equiv 1$ by algebra, so the regressor has zero variance |
| `_classify` returns `H2_UNSUPPORTED` unconditionally (2b §4.3) | `inflation` $\le1.02$ against a $>1.5$ threshold |
| `imaginary_ablation` zeroes everything (2b §4.2) | projects onto $\mathrm{col}(A)$; real antisymmetric matrices are full rank in even $d$ |

> **The check, and it takes a minute:** before running a test, ask what its output would be on data
> where the hypothesis is *maximally true*, and on data where it is *maximally false*. If those two
> answers are the same, it is not a test.

### 2. Thresholds not derived from a null

A cutoff calibrated on one distribution, inherited into another, and read as a finding.

Fiedler CLUSTER/MIXED/MIXING at 0.3/0.7 against deviations of $\pm0.05$ (1 §8.3) · the relative
classifier's 0.90/0.98, labelled "a reporting convention" in the code (1b §7.3) · $Q_k$ against a
fixed absolute tolerance when $\mathbb E[Q_k] = 1/n$ (1c §8.6) · the V-score's placed weights and
unreachable 1.0 ceiling (2 §5.1, §10.3) · `asymmetry`'s $1/\sqrt2$ point null with no spread
(2d §7.4) · `ext_sem_threshold = 0.5` inherited from GPT-2 (1 §9.4) ·
`DEGENERATE_RANK_THRESHOLD` carried across a scale change (1 §6.2).

### 3. Dimension not controlled

Comparing alignment or capacity across subspaces of different size. $\mathbb E\lVert P_Uv\rVert^2 = k/d$
for a random unit vector, so **any** such comparison measures dimension unless normalized.

Phase 6's LDA alignment 0.887 ($U_A$) vs 0.067 ($U_{\rm neg}$) and the probe accuracies (6 §7.2) ·
Sub-exp D's S- vs A- vs full-$M_h$ isometry (5b §8.1) · the cone verdict's $n$-vs-$d_{\rm eff}$
counting (1b §6). Note `dissociation.py` **gets this right** — arm 3 is a matched-dimension random
control — so the pattern is known inside the project and applied unevenly.

### 4. Clamping hides the diagnostic

`max(0, ·)` on a quantity whose negativity is informative.

`orth_sq` when the projector bases are non-orthogonal (5 §3.1) · `max(0, n_phase1 - n_rescaled)`,
destroying the sign that distinguishes "no effect" from "made it worse" (2b §7) ·
`henrici_absolute` (handled correctly — the unclamped value is returned alongside, and a materially
negative one signals the block parse disagrees with $T$).

### 5. Producer/consumer mismatch

A writer and a reader disagreeing on names, shapes, or paths, **with neither side erroring**. The
failure always returns a plausible empty value (`n/a`, `None`, `0.0`, `[]`) that flows into a score.

Two Phase 1 event schemas, the wrong one read (5 §1.2) · OV values `n/a`, so head rankings are
ungrounded (5 §4) · Group D blocked by a path mismatch (5 §5) · `sinkhorn.json` never persisting
what the report reads (1 §14) · ALBERT run directories never resolving (1b §7.6) ·
`find_phase2_runs` substring matching, colliding 8 of 27 checkpoint stems (2b §5).

`core/artifacts.py` is the structural fix: declare each contract once, have every consumer import it.

### 6. An instrument whose failure mode looks like the result

The most dangerous pattern, because the output is plausible.

`h_displacement` understating $T_{\rm eff}$ by 5.67×, biasing toward "resistance is an artifact of
depth" (1c §1.2) · `elim_signed = 1.0` being exactly what an early-truncating frame produces for
free (2b §5) · underflow making every energy the constant $1/(2\beta)$, so the frame reports zero
violations (2b §5) · the tuned lens "skipping ahead" to decode the *output* token at every depth,
erasing precisely the per-layer differences Group E exists to measure (5 §6.2) · raw effective rank
degenerating to a sink count (1 §6.2).

---

## Open-question register

**~60 questions across the ten documents.** Roughly half were already tracked in the repo; the rest
were surfaced by writing these docs. Ordered by value-per-unit-cost. `[R]` = report-only,
`[W]` = weights, no forward pass, `[F]` = forward passes.

### Highest value, lowest cost

| # | Question | Doc | Cost |
|---|---|---|---|
| 1 | **Measure $T_{\rm eff}$ vs $t^\ast$.** Decides whether "trained weights resist collapse" is a finding or an artifact of depth. Every input is on disk | 1c §3.5 | [R] |
| 2 | **Rule out dimension as the explanation for Phase 6's inversion** before the labelling audit — `subspace_build.py` knows $\dim U_A$ and $\dim U_{\rm neg}$ per layer; that one ratio decides it | 6 §9.1 | [R] |
| 3 | **Re-establish the rank plateau on normed rank**, and reconcile ~200 against Blog 1's ~250. The budget hypothesis rests on it | 5c §9.1 | [R] |
| 4 | **Run `rank_panel`** — settles Phase 1 defects D1/D10 and answers "do the three rank surrogates agree?" in one pass | 1 §15.5, 1c §5 | [R] |
| 5 | **Null the "98% complex" headline** against a real-Ginibre baseline (expected $1 - O(d^{-1/2})\approx0.97$ at $d{=}1024$ — inside the reported range) | 2b §8.2 | [W] |
| 6 | **Compute $Z_{\beta,i}$ per token.** It is the metric weight making (SA) a gradient flow; one line from the Gram matrix; nothing in the project has looked at it | 1 §15.12, 2d §7.6 | [R] |
| 7 | **Reconcile the Phase 5 selection score** (max attainable 4.0 vs reported 9.000) with the fixed event reader | 5 §11.1 | [R] |
| 8 | **Cross the energy-violation indicator with $\lVert\mathcal X\rVert$**, which the Lyapunov identity says predicts $\Delta E_\beta$'s magnitude. Both quantities exist and have never met | 1 §15.13 | [R] |

### Structural fixes before the next run

| # | Question | Doc |
|---|---|---|
| 9 | Phase 1b should adopt `hull_min_norm` (1c's exact $\ell_2$ margin) — two phases solve one problem two ways and only one is comparable across prompts | 1c §11.4, 1b §7.1 |
| 10 | Add a dimension-matched control to Sub-exp D | 5b §8.1 |
| 11 | Restrict Group F's `mean_frac_together` to post-hook layers | 5 §11.3 |
| 12 | Unclamp `orth_sq` and record which decomposition produced the projector bases | 5 §11.2 |
| 13 | Register the time-residual prediction as a dated addendum (it is 2.5 orders of magnitude more sensitive than the vertical residual, and P-γ1 is registered on the *vertical* one) | 1c §11.7 |
| 14 | Give P6-I2 its rotary null before running on Pythia — the confound is *correlated with the contrast* | 6 §9.3 |
| 15 | Register a coverage threshold for 5b before the isometry correlation is computed | 5b §8.4 |
| 16 | Report `pos0` in/out for the attention flip | 5c §9.4 |

### Conceptual, and worth the most if they land

| # | Question | Doc |
|---|---|---|
| 17 | **Cluster tokens directly on their $S$-projections** and compare to HDBSCAN via ARI — the only cluster definition in the project derived from the operator that generates the dynamics | 6 §9.4 |
| 18 | **Is a plateau a near-critical point?** The gradient-flow framing gives an operational test ($\lVert\mathcal X\rVert$ small while not collapsed) the flatness detector does not have | 1 §15.14 |
| 19 | **Part C is the real version of the test Phase 2b could not run** — a forward-pass intervention is not orthogonally invariant | 6 §9.2 |
| 20 | Does `ip_mean` look like a *failed* attempt to concentrate onto $\gamma_\beta$, or like never approaching it? Different residual shapes; bears on which reading of §1.6 is right | 1 §15.9 |
| 21 | Four-cell cross-tab: real/complex × attractive/repulsive, energy-weighted. Phase 2 and 2b partition the same eigenvalues on different criteria and neither reports the crossing | 2b §8.4, 2d §7.3 |
| 22 | Do P-T1's row-2 candidates and P-M1's in-regime heads overlap *at all*? Row 2 constrains only $\mathrm{Sym}(M)$, so the two predictions may be about disjoint populations | 2d §7.3 |

---

## Corrections owed to the source

Found while verifying the docs. Each is an error in the repo, not in these documents.

| File | Issue |
|---|---|
| `p5b/isometry_test.py:26` | Hellinger range given as $[0,1/\sqrt2]$; it is $[0,1]$ |
| `p2b/rotational_schur.py:340-342` | Henrici described as the squared Frobenius norm of $T$'s strict upper triangle — true for the *complex* Schur form, false for the real one this phase uses ($d{=}6$: 12.26 vs 19.08) |
| `p2_eigenspectra/verdict_v2.py:61` | V-score range given as $\approx[-0.15, 1.0]$; the three positive weights sum to $0.85$, so $1.0$ is unattainable |
| `UPDATE_PLAN.md` §5.6 | Claims the buggy trace contraction coincides "at $M = I$ **and at any symmetric $M$**" — false; $\mathrm{tr}(MCMC)\ne\mathrm{tr}(M^2C^2)$ for symmetric non-commuting $M$ (28.36 vs 43.76). Only $M = I$ |
| `p1_io.py` | `load_cluster_tracking` (the correct Phase 1 event reader) currently lives in `p5/anchors.py` and belongs here as a sibling of `load_phase1_run` |
| `README_phase6.md` | Header says "Not started"; one model has been run |
| `readme-phase2c.md` | Same stale header |

---

## Two things to keep in view

**Four of the six registered predictions test open problems, not theorems.** Problem 1
(metastability), Problem 2 (general $Q,K,V$), Problem 5 (Table 1's generality), and Remark 3.5's
Sinkhorn question are all posed as open by the paper. That is the right thing to be doing — but the
framing throughout should be *evidence bearing on open conjectures at parameters far outside where
their supporting numerics live*, never *replication*. `math-1.md` §16 has the full map.

**The project sits outside the regime its own null model is proved for.** Figure 3's metastable band
vanishes by $d\approx512$; Pythia runs at $d = 1024$. Theorem 6.9's concentration bound is weakest
in the middle of the trajectory, which is exactly where metastability would live — and its own
sufficient condition on $d^\ast$ is far above 1024. So "concentration forbids metastability at high
$d$" is not a theorem, and the residual is measuring something the theory genuinely does not pin
down. That is a better justification for the whole enterprise than the one currently written down.
