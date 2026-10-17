<!-- trackB_circuit/lit-B.md -->
# Track B — literature scan at opening (2026-10-13)

Trigger 1 (`CLAUDE.md` "Literature scans"): Track B opens, so this precedes `design-B.md`'s freeze.
Scope: what B0 (the nudge harness) and B1 (`L5H2` as positive control) must not get wrong. Three
web searches (2026-10-13), read at abstract / summary level only unless a section says otherwise;
the project's own Phase 7 / 7d scans (`p7_motifs/lit-7.md`, `p7d_redundancy/`) are not repeated.

## 1. Previous-token heads form first, in Pythia's 512–1000 window

- *Fingerprint, Not Blueprint* (arXiv 2607.06621, 2026, preprint): on 22 public checkpoints of
  `pythia-410m` and `pythia-160m`, previous-token behaviour jumps first (0.37 at step 512 → 0.95
  at 1000), ahead of induction; the previous-token head's rotary-phase signature moves to the top
  decile by 1000, with neither leading at checkpoint resolution. **Bearing:** an independent
  measurement that B1's target switches on inside (512, 1000], the interval 7d found for `L5H2`'s
  causal effect (`p7d_redundancy/status-7d.md` "Q2"). B1's prediction is therefore a known answer
  from two sources, which is what a positive control should be.
- Singh et al., *What needs to go right for an induction head?* (arXiv 2404.07129, 2024): the
  induction circuit is a previous-token head feeding an induction head; it forms in a sharp phase
  change driven by the two heads' interaction (clamping experiments). **Bearing:** the
  signature should switch on, not ramp; a ramp would itself be a reading.
- *Predicting the Emergence of Induction Heads* (arXiv 2511.16893, 2025): Pythia's early public
  checkpoints are the only family dense enough at the transition. **Bearing:** Pythia publishes
  nothing between 512 and 1000, so B1 cannot place the onset finer than that interval.
- Stagewise data attribution (arXiv 2510.12071, 2025) retrains Pythia-14M with checkpoints every
  100 steps to resolve the transition. **Bearing:** finer placement needs own training (triage
  M1), out of B's scope.

## 2. Perturbing the residual stream: plateaus, and the random-direction baseline

- Heimersheim & Mendel, *Activation plateaus and sensitive directions in GPT-2* (2024; summarised
  in arXiv 2409.15019 and 2410.12555): model-generated activations resist small perturbations
  (a "plateau": the output barely moves until the perturbation crosses a threshold), and
  directions toward other real activations move the output sooner than random directions of equal
  size. Later work finds the plateaus around real activations but not around random points of the
  same distribution. **Bearing, the main one for B0:** the ± expansion that B0's antisymmetric
  part relies on (`tools/math_checks/b0_antisymmetric_nudge.py`) can be swamped on a plateau, where
  the response below threshold is not its first-order term. So B0 sweeps ε and reads the curve,
  and reports where the response turns on, rather than fixing one small ε.
- arXiv 2410.12555 (*Investigating sensitive directions in GPT-2: an improved baseline*): the
  standard protocol is `x ← x_base + α·d` at one site (layer 6 `resid_pre`), the effect read as
  next-token KL, against random unit directions; the baseline's choice changes conclusions about
  SAE directions. **Bearing:** B0's null (random directions of the same norm at the same site) is
  the field's convention, and its known weakness is that random directions are a weak baseline
  (they leave the activation manifold). B0 reports the rank against random directions *and* the
  ε at which each turns on, so a direction that is merely "on-manifold" is not read as specific.

## 3. Attention as an interacting particle system: no perturbation-response theory found

- Karagodin, Polyanskiy & Rigollet (arXiv 2411.04990, read for Phase 10): causal masking gives a
  particle system that is not a mean-field gradient flow, converging to one cluster for any
  `Q, K` with `V = I`. SympFormer (arXiv 2603.16535, 2026) and the mean-field well-posedness work
  treat layers as discretised particle dynamics.
- **Gap:** no paper found on the *linear response* of such a system to a single particle's
  nudge (which is what B1 measures: token i's response to token i−1's nudge through one head).
  B's readings are therefore descriptive, not tests of a theorem. **Bearing:** B1 is a positive
  control for the instrument, not a test of the particle picture.

## What changes in the design

| from | change |
|---|---|
| §1 (2607.06621, 7d) | B1's expected onset is (512, 1000], not ROADMAP's wider (512, 4000]; the wider window stays the pass condition, the narrower one is recorded beside it before any reading |
| §1 (2404.07129) | B1 labels a step on or off; a gradual rise across 512–4000 is reported as such |
| §2 (plateaus) | ε is swept (4 values per site), the turn-on ε is a readout, and the calibration gate is run at every ε |
| §2 (baseline) | the random-direction null is kept, with the rank and the turn-on ε both reported |
| §3 | no theory to test; B1 is an instrument check |

Not read: the primary Heimersheim & Mendel post (only its two follow-ups' summaries); 2607.06621
beyond its search summary. Both are cited at that level and flagged here.

---

# B2a, the 2D spike — scan at opening (2026-10-15)

Trigger 1 (B2a opens; precedes `design-B2a.md`'s freeze). Sources: the user's untracked LaTeX of
2609.28448 in `docs/readings/arXiv-2609.28448v1/` (§6.1 read in full, §2.1 read), 2512.01868
(HTML, §3.3 and headings), 2305.05465 (HTML, §1–5 statements), 2604.01978 v1 (HTML, all
headings and §2, §6–7 skimmed), and three web searches (2026-10-15).

## 1. The section design-A cites is in 2609.28448, not 2604.01978

`design-A.md` "Frozen 2026-10-14" cites 2604.01978 for "General value maps and effective
low-dimensional dynamics". 2604.01978 (Koubbi, Geshkovski, Rigollet, *Homogenized Transformers*)
has no such section: it is about random weights resampled per layer and head, with no value-map
reduction and no circle case. The section is **2609.28448 §6.1** (Gao, Yang, Chen, *Nonequilibrium
Phases of Repulsive Self-Attention*, `main.tex` l. 3622–3898). A correction line goes to
`trackA_force/status-A.md`.

## 2. What 2609.28448 §6.1 says (the B2a model)

| item | the paper |
|---|---|
| model (§2.1) | `x_i ∈ R^d`, `‖x_i‖ = √d`; `z_ij = x_iᵀ Qᵀ K x_j / √d`; `A_ij ∝ exp(β z_ij)`, `j ≠ i`, non-causal; `x_i ← N[(1−γ) x_i + γ V Σ_j A_ij x_j]`, with `N` restoring the norm (LN's analogue) and `0 < γ < 1` |
| scope of §6.1 | **`Q = K = I` throughout**; only V is general |
| reduction | any real invariant subspace U of V (a real eigenline, a complex-pair plane, a degenerate eigenspace), `V U = U W`: tokens `x_i = √d U v_i` obey `v_i ← N[(1−γ) v_i + γ W Σ_j A_ij v_j]` **exactly**, with `β_eff = β √(d/D)` (its score normalisation; in a cosine convention β carries over unchanged) |
| d = 2 | a degenerate negative pair (`W = −ν I₂`) gives the repulsive circle model at `γ_eff = γν / (1−γ+γν)`; a rotation pair gives `W = R_φ`; a general pair adds shear |
| which subspace is selected | for an **aligned cluster attending within itself**, components grow as `b_a = 1 − γ + γ λ_a`; the largest `|b_a|` is selected at rate `ρ_⊥ = max |b_a| / |b_top|`. For tokens in different directions, selection depends on A as well as V and "cannot be inferred from the spectrum of V alone" |
| trained models (§6.2) | V's analogue is "the complete linear path that sends a head's value vector back to the residual" (`W_O W_V`), "generally low rank and nonsymmetric"; negative eigenvalues of one head's map do not establish negative feedback |

**Bearing.** The reduction lives on **V's invariant subspaces**, not on QK; attention enters only
through `v_i · v_j` at `β`. That is why B2a waited for A1a (β per head; the block step that plays
γ). A trained head has `Q ≠ K` and rotary, so "reproduces its real attention" is a test of two
things at once: is the head in the paper's class at all (isotropic content plus position), and
if so, does V's top invariant subspace carry the content.

## 3. Bounds and neighbours

| source | finding | bearing |
|---|---|---|
| `tools/math_checks/b2a_subspace_ceiling.py` | a head's logit reads the token only through `W_Q y`, `W_K y`: projecting onto the row span of `[W_Q; W_K]` is exact at every rotary offset, so **D ≤ 128** on the head's own QK; `U = 0` leaves `b_qᵀ R(Δ) b_k`, an offset-only (positional) pattern | the QK-side question has a hard ceiling far below 1024; a positional head can be D = 0 |
| 2512.01868 §3.3 (Rigollet survey) | on the circle with `Q = K = V = I`, attention dynamics is a Kuramoto variant, `θ̇_i ∝ −Σ e^{β cos(θ_i−θ_j)} sin(θ_i−θ_j)` | the attractive d = 2 case, beside 2609.28448's repulsive one |
| 2305.05465 Thms 4.2, 5.2 | with V's top eigenvalue real, positive, simple and `⟨Qφ₁, Kφ₁⟩ > 0`, tokens cluster on ≤ 3 hyperplanes ⟂ φ₁; paranormal V with a degenerate top gives a polytope in that eigenspace | the general-QK version: the selected subspace is still V's top one, with a sign condition on QK there |
| 2607.06621 (preprint); 2410.06205 | previous-token heads concentrate on high rotary frequencies; one rotary plane suffices for an off-diagonal positional pattern | **against the user's prior's mechanism, not its verdict:** a previous-token head may need no content subspace (D = 0), which is "not the circle" |
| 2408.09503 | induction-relevant heads share a low-rank latent subspace bridging OV and QK; removing it drops copying | a shared subspace across members is plausible; B2's principal angles, not B2a |
| 2504.20938 | low-rank QK descriptions of induction heads lose fidelity; "multidimensional QK" needed | supports the prior for induction heads on the QK side |
| search: trained-model circle / Kuramoto reductions | none found that projects a trained model's tokens onto a 2D invariant subspace and tests the reduced map | B2a is new as an empirical check; descriptive, tier 1 |

## What changes in the design

| from | change |
|---|---|
| §1 | the citation is corrected (`status-A.md` "Corrections received"); B2a cites 2609.28448 §6.1 |
| §2 | the primary reduction is the paper's: U from **V's invariant subspaces** (`W_O W_V` in the LN1 frame), ordered by `|b_a|` at the block's γ; attention from `v_i·v_j` at A1a's β, Q = K = I |
| §2 (Q = K = I) | the head's offset profile is added to the reduced logit (Pythia has rotary; the paper has no position). `D = 0` is that profile alone, the positional rung |
| §2 (selection not spectral) | the share of each token's norm inside U is read beside the attention test (the paper's "attraction" question, which V alone cannot answer) |
| §3 (ceiling) | the head's own QK restricted to U is read beside, with its exact ceiling at D ≤ 128 |
| §3 (2607.06621) | `D* = 0` is a separate verdict, "positional, no content subspace"; it is not a pass of the circle case |

Not read: 2609.28448 beyond §2.1 and §6; 2604.01978 v2 (if any); 2607.06621, 2408.09503,
2504.20938 beyond search summaries. Cited at that level.

---

# B2, spectra from the weights — scan at opening (2026-10-17)

Trigger 1 (B2 opens; precedes `design-B2.md`'s freeze). Four web searches (2026-10-17), all read
at search-summary level only; nothing below was read as primary text. The B2a scan above (2609.28448
§6.1, 2305.05465, 2408.09503, 2504.20938) is not repeated.

| source | finding (summary level) | bearing on B2 |
|---|---|---|
| 2609.22990 (*On attention heads and bilinear forms*, 2026, preprint) | equal counts of positive and negative eigenvalues of a head's symmetric QK part are a consequence of the low-rank construction, not learned; training moves their *distribution*, and the symmetric part is stronger than a random baseline | **the main one.** Made exact in `tools/math_checks/b2_symmetric_inertia.py`: `sym(A Bᵀ)` has inertia (r, r) whenever `[A B]` has full column rank, for OV as for QK. ROADMAP's "dominant negative eigenpairs" cannot be a count; B2 reads \|λ\| weights and the top pair, against step 0 |
| 2607.06621 (cited above for B1) | over Pythia's public checkpoints every head's QK spectrum starts at the random-matrix (Ginibre) null; spectral "fingerprints" consolidate only after the function arrives | the step-0 band is the right null for every column, and a member's QK column should leave it at or after its function's onset (B1: (256, 512] for `L5H2`) |
| Saponati et al. (2502.10927, summary via alphaxiv) | bidirectional objectives drive `W_Qᵀ W_K` toward symmetric, autoregressive ones toward directional (asymmetric) | 2d's gradient-flow condition (QK symmetric) is expected to fail broadly on a decoder; the reading is a member against the population, not against 0 |
| 2511.01033 (2025) | in a toy model, training stays in a 19-dimensional parameter subspace and 3 directions drive the induction head's emergence | a shared low-dimensional subspace across members is plausible; B2's principal-angle column tests it on weights |
| 2601.10266 (2026, preprint) | a principal-angle "projection kernel" between head weight subspaces reads IOI head interactions better than the composition score | principal angles between head subspaces have precedent as a head-relation measure; B2 uses the plain overlap `tr(P_U P_W)`, chance r²/d |
| Products of Gaussian matrices (Burda et al.; rectangular case, search summary) | eigenvalue densities of products of square or rectangular Gaussian matrices are rotationally symmetric disks | `W_O W_V` at init has no preferred sign: the attractive share's null is ½ (fact 3 of the check); the rank-64 case's density is not tabulated, so the step-0 population is used, not a formula |
| 2503.01329 (Neural ODE transformers) | tracks OV eigenvalue signs and magnitudes over training, QK as matching and OV as copying | precedent for spectra over training; not on Pythia's public checkpoints |
| search: per-head OV spectra over Pythia checkpoints | none found | B2's OV columns over the 18 steps are new as measured; descriptive, tier 1 |

## What changes in the design

| from | change |
|---|---|
| 2609.22990 + the check | no column counts eigenvalue signs; "dominant negative pair" is a weights condition (the top two by \|λ\| negative and near-degenerate), read as a rate against step 0; an inertia other than (64, 64) is reported as read/write overlap, not attraction |
| 2607.06621 | every column is placed against step 0's 384 heads; the member statistic is read from step 512 |
| Saponati et al. | 2d's regime is read as members against the same-step population, not against the paper's symmetric ideal |
| products of Gaussians | the share's null ½ is checked on step 0 in the gate, not assumed |

Not read: all eight beyond search summaries. Cited at that level.
