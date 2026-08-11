# Phase 2d — STATUS

**State:** D1, D2, D3, D4 implemented and validated on constructed operators and synthetic
activations. **Not run against Pythia weights or artifacts.** P-M1 and P-T1 were registered in
`PREDICTIONS.md` before this code existed.

**Driver complete** as of this revision: LN frame resolution (`resolve_ln_params`) and P-M1's
violation counts (`violation_counts`) are wired, so `run_2d.py` runs end to end given Phase 2
weights and a Phase 1 run at a matching revision.

**Blocked on Phase 1c-B by design** — see design-2d.md. The $T_{\rm eff}$ result determines
whether the energy-monotonicity break is the right thing to attribute.

## Validation performed

**D1 recovers constructed regimes.** A head built with $M$ symmetric and $V = M$ classifies as
`gradient_flow` (asymmetry 0.000, alignment $+1.000$); the same head with $V = -M$ classifies as
`repulsive_aligned`; a random head reads `outside` with asymmetry 0.693, which is the
$1/\sqrt2$ a generic matrix should give.

**D2's sanity anchor holds.** At $M = I$, $\mathrm{PR}_M$ equals $\mathrm{PR}_C$ to 1e-8. A
rank-1 operator gives $\mathrm{PR}_M = 1.000$ whether aligned with the cloud's top direction or
its tail — correct, and the discriminator between the two is $\mathrm{tr}(MC)$, not the rank.

**D3's row classifier is exact on constructed cases.** $V = +2I \to$ row 1; $V = -2I \to$ row 4;
a real simple $\lambda_1 = 3$ with $M = I \to$ row 2; the *same* $V$ with $M = -I \to$
`row2_eigen_only_qk_fails`; a complex top pair $\to$ `unclassified`. All four adjudicator
branches fire on constructed inputs.

**D4's overflow guard works.** At $\|M\|$ scaled 50×, $E_{\beta=5}$ evaluates to $1.3\times10^7$
rather than `inf`.

**The join guards all fire.** Revision mismatch, unknown revision, $d_{\rm model}$ mismatch and
missing $W_Q/W_K$ arrays each raise rather than proceeding. The raw-frame warning is attached
to every record when LN parameters are absent.

**P-M1's inputs are derived, not stubbed.** `violation_counts` reconstructs the per-boundary
violation indicator from `energies.json` using `ENERGY_VIOLATION_REL_TOL` — the same relative
rule the summary table and `checkpoint_scalars.py` now share, so the three cannot drift. Tested
on a constructed series: a 0.17% drop counts, a 34% drop counts, a flat segment does not.

**Both P-M1 adjudicator branches fire.** On a constructed series where violations sit in the
high-distance layers, all three aggregates give $r = 1.00$ → CONFIRMED. On uncorrelated inputs
they give $-0.25 / -0.09 / -0.54$ → FALSIFIED.

## Findings from implementation

1. **P-T1 as registered omits half of Table 1's row-2 hypothesis.** The row requires
   $\langle Q\varphi_1, K\varphi_1\rangle > 0$ in addition to $\lambda_1(V) > 0$ simple.
   Testing without it would falsify a claim the paper does not make — structurally the same
   error as the retracted "Thm 6.1" verdict row. The prediction's *wording* in `PREDICTIONS.md`
   should be amended with a dated addendum rather than silently corrected, since it was
   pre-registered. The code checks both conditions and labels the difference.

2. **Histogram peak-counting is not a modality test.** The first implementation of
   `projection_modality` scored a plain Gaussian cloud at **nine modes** and a genuinely
   trimodal one at four, at 60 bins on 500 points. Replaced with a KDE at Silverman bandwidth:
   the Gaussian now reads 1, the trimodal reads 3 with spacing ratio 1.000, and both are stable
   across a 4× bandwidth scan.

3. **A modality claim at a single unstated bandwidth is not a measurement.** Any distribution
   can be made unimodal by over-smoothing and multimodal by under-smoothing.
   `modality_stability` scans and reports `stable_n_modes` — the count holding over at least
   three consecutive bandwidths — and `None` when the data does not determine it. P-T1 should
   be adjudicated on the stable count only.

   (Note: `p1c_frames/design_test.inner_product_modes` remains histogram-based, which is
   correct *there* — a sharp configuration's pairwise cosines concentrate at a few exact
   values, so the histogram is nearly a set of deltas and validates exactly against the
   octahedron and icosahedron. The two are different regimes and the difference is deliberate.)

4. **The signed OV/QK cosine separates two regimes the plan treated as one.** Anti-alignment is
   not "far from the condition" — it is the $V = -I_d$ case where the paper itself predicts
   decreasing energy. `repulsive_aligned` heads should be scored as *confirming* the paper,
   not as violating it, and a distance-only score would have put them at the far end of the
   same axis as genuinely unstructured heads.

5. **A wrong trace contraction that the $M = I$ anchor could not catch.** D2's denominator
   $\mathrm{tr}(M^\top CMC)$ was implemented as `sum((C@M) * (C@M.T))`, which contracts to
   $\mathrm{tr}(CMMC)$ — a different quantity that **coincides at $M = I$ and at any symmetric
   $M$**, so the sanity anchor passed while the value was wrong for every real head. It was
   caught only because `coupled_mass` came out negative, which is impossible:
   $\mathrm{tr}(M^\top CMC) = \|C^{1/2}MC^{1/2}\|_F^2 \ge 0$. Measured on a generic $M$, the
   wrong form gave $-72.08$ against a true $+167.00$. The correct contraction is
   `sum((C@M) * (M@C))`; it appeared in three places (D2, `spectral_pairing`, and D4's
   second-order term) and all three were wrong. A non-negativity assertion now runs on every
   call, so the check is permanent rather than a one-off.

   The general lesson, which applies beyond this function: **an anchor that only tests the
   identity case tests almost nothing about a bilinear form.** Every anchor in this phase
   should have a non-symmetric arm.

6. **$\mathrm{PR}_M$'s numerator is a signed trace, so cancellation reads as absence.** A head
   that couples the cloud strongly with mixed signs has $\mathrm{tr}(MC) \approx 0$ and reads
   $\mathrm{PR}_M \approx 0$, identical to a head that couples nothing. A pure rotation
   (antisymmetric $M$) reads exactly 0, since the trace of symmetric × antisymmetric vanishes.
   `coupled_mass` — sign-blind, non-negative — is reported alongside, and the pair distinguishes
   them: low $\mathrm{PR}_M$ with low `coupled_mass` is a head pointed away from the tokens
   (the $\beta$-independence hypothesis); low $\mathrm{PR}_M$ with high `coupled_mass` is a
   rotation.

7. **A violation "count per layer" is a category error.** A violation is an event between two
   adjacent layers, so there is exactly one per boundary and the series is an indicator, not a
   count. Correlating a per-layer regime score against it is correlating against a boolean, and
   `violation_counts` returns it as one rather than letting a "count" name imply otherwise.
   Layer 0 is zero by construction (no preceding layer), which biases the correlation slightly
   toward zero; it is reported rather than dropped, since dropping it would misalign the regime
   series.

## Open before running

1. **Which activations feed D2/D3/D4.** They must be the LN'd states attention actually reads,
   in the right frame — `core/ln_frame.frame_for_hidden_state` resolves the off-by-one, and it
   must be used rather than re-derived. D2 and D4 on raw residual-stream activations would be
   measuring a different operator's action.
2. **Centred vs uncentred token covariance in D2.** Both are meaningful and they answer
   different questions; the uncentred $C$ is dominated by the common mode on an anisotropic
   cloud, so $\mathrm{PR}_M$ can read $\approx 1$ purely because every token shares a
   direction. Run both wherever $\kappa_1$ is large.
3. **The `simple_tol` and `align_tol` constants are placed, not derived.** Eigenvalues of a
   $d{=}1024$ non-normal OV circuit come in near-degenerate clusters, so "simple" needs a
   tolerance, and the classification counts move with it. Both are returned in every record so
   reclassification needs no recomputation — do the sensitivity scan before quoting any rate.
4. **fp32 is mandatory.** The row classification turns on the sign and multiplicity of
   $\lambda_1(V)$ near zero, which is exactly what `core/models.py`'s precision guard exists to
   protect (and which that guard's docstring, corrected this cycle, now names correctly).
5. **The extraction convention must be passed, not guessed.** `resolve_ln_params` exposes
   `--keep-embedding` and `--last-is-post-final-ln` because the off-by-one between hidden-state
   index and reading block depends on how the activations were extracted, and
   `core/ln_frame.resolve_frame_index` cannot infer it. The defaults follow this project's Fix 4
   convention (index 0 = block-0 output) and assume final LN was *not* pre-applied. If the
   extraction did apply it, the last state's correct frame is the identity and applying final LN
   again is wrong — the driver prints which indices resolved to identity so this is checkable
   against the extraction path rather than assumed.
6. **`--pm1-beta` is not swept, deliberately.** P-M1 is a claim about *where* violations sit, and
   different $\beta$ produce different violation sets; pooling them would mix the sets and the
   correlation would be against a union that corresponds to no single energy. Report per $\beta$
   by re-running.
