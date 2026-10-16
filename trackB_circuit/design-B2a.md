<!-- trackB_circuit/design-B2a.md -->
# Track B — design: B2a, the 2D spike

Opened 2026-10-15 by Blocked 49 (a) (the user's: B2a after A1 closed). Origin: the user's question
of 2026-10-14, can a head's dynamics be read in the paper's d = 2 reduction (`trackA_force/design-A.md`
"Frozen 2026-10-14"; `docs/ROADMAP.md` §7 B2a). Scan: `trackB_circuit/lit-B.md` "B2a". Tier 1
throughout: exploratory, unregistered; 410m is an exploration rung.

**Not frozen.** Blocked 50 (`STATE.md`) asks the user to freeze it and decide the two choices
below marked *(50)*.

## The paper, corrected

The reduction is **2609.28448 §6.1** (Gao, Yang, Chen), not 2604.01978 as `design-A.md` cited
(`lit-B.md` "B2a" §1). With `Q = K = I` and norm-fixed tokens, any real invariant subspace U of the
value map (`V U = U W`, dimension D) carries an exact reduced map
`v_i ← N[(1−γ) v_i + γ W Σ_j A_ij v_j]`, attention from `v_i · v_j` at β. D = 2 with `W = −ν I₂`
is the repulsive circle; `W = R_φ` a rotating one. Which U the dynamics selects follows V's
amplification `|b_a| = |1 − γ + γ λ_a|` only for an aligned cluster; for general configurations
it "cannot be inferred from the spectrum of V alone".

## Exit condition and budget

- **Exit (as frozen in design-A):** a faithful D per named head against the null, at step 143000
  (unit 2), and over the 18 steps for the heads that have one (unit 3). "No D on the ladder is
  faithful" and "D* = 0" are verdicts, not failures.
- **Budget *(50)*:** design-A froze 3 units (weights only with a planted positive control and a
  null; real tokens at 143000; the passing heads over 18 steps) and named no design unit. This
  design, the scan and the `sympy` check are a unit of work. Recommended: **4 units on the
  record**, this design as unit 0. Alternative: 3, with unit 1's gate and weights-only rows
  folded into unit 2's session.

## The math

`tools/math_checks/b2a_subspace_ceiling.py` (`sympy`, 3 checks pass): (1) a head's logit
reads a token only through `W_Q y` and `W_K y` at every rotary offset, so its own QK is exact on
the row span of `[W_Q; W_K]`: **D ≤ 128 always**; (2) `U = 0` leaves `b_qᵀ R(Δ) b_k`, offset only,
which in one rotary plane is `|b_q||b_k| cos(Δθ − φ)` (a previous-token peak needs no content);
(3) a symmetric rank-2 form on U is the circle kernel `α r_i r_j cos(θ_i − θ_j)`, an
antisymmetric one gives `sin`. **What it does not prove:** anything about real tokens below the
ceiling; that projected tokens have equal norms (they do not, so the kernel's β is token-dependent).
The paper's `β_eff = β√(d/D)` is its √d score scale: with tokens in U and A1a's cosine
convention, `cos(x_i, x_j) = v_i · v_j`, so β carries over unchanged (one line, no check needed).

## Objects, fixed here

| object | definition |
|---|---|
| frame | A1's: block ℓ's unit LN1 rows, `u_i = unit(LN1_ℓ(x_i))`, positions 1 … n−1 (position 0 out, as A1a) |
| V of head h | the complete path back to the frame (2609.28448 §6.2): `V_h = diag(g_ℓ) C W_O^h W_V^h`, with `g_ℓ` LN1's gain, C the centring projector (LN's mean removal), biases out; LN's 1/σ is a scalar γ absorbs. Rank ≤ 64, nonsymmetric |
| β of head h | A1a's `fe_full` for that (step, prompt, head), as recorded (`trackA_force/a1a.py`) |
| γ of head h | `γ / (1 − γ)` = the median over tokens of `‖diag(g_ℓ) C o_i^h / σ_i‖ / ‖LN1_ℓ(x_i)‖`, head h's own output `o^h` mapped into the frame (σ_i LN1's scale at token i) (read in unit 2; for unit 1's weights-only ordering, γ = 0.1, placed, with the ordering by `Re λ` beside, which is γ's small-γ limit) |
| V's ladder | real Schur form of `V_h`; invariant blocks (a real eigenline, D += 1; a complex pair, D += 2, kept whole) ordered by `|b_a| = |1 − γ + γ λ_a|`, largest first. `U_D` = the span of the first blocks with total dimension D. Ladder D ∈ {0, 1, 2, 3, 4, 6, 8, 16, 32, 64}, a rung that would split a pair is moved up by one |
| reduced attention (M1, primary *(50)*) | `log Â_ij = β · cos(Uᵀu_i, Uᵀu_j) + φ_h(i − j) − log Z_i`, causal, `j ≤ i`, j ≥ 1. `φ_h` is the head's offset profile, the per-offset fixed effects of the `fe_full` model, refitted on the other 6 prompts (leave one prompt out, so the profile is not fitted on the row it predicts). `D = 0` is `φ_h` alone, the positional rung |
| M1 at full D | the same with U = I (`fe_full`'s own isotropic model): how well the paper's class fits the head at all |
| head's own QK (M2, beside *(50)*) | the real logit with `y → U Uᵀ y` in both slots, biases and rotary kept; on V's ladder and on the QK read-space ladder (SVD of `[W_Q; W_K]`, exact at its rank ≤ 128) |
| real attention | recomputed from the weights on the hooked pass's LN1 rows; **G0** checks it against the stored `attentions.npz` |
| faithfulness | per query row i ≥ 1, total variation `TV_i = ½ Σ_j |A_ij − Â_ij|`; the head's statistic is the mean over rows and the 7 prompts |
| null | K = 32 Haar-random D-dimensional subspaces of the frame per (head, D, prompt), seeded; same β and `φ_h` |
| beside | (a) **occupancy**: the median share `‖Uᵀu_i‖²` (random: D/1024), the paper's attraction question; (b) **motion**: the cosine between the reduced update `γ W Σ_j Â_ij v_j` lifted by U and head h's real output in the frame, tangent part; (c) the type of `U_2` (degenerate negative, rotation pair, mixed) and `ρ_⊥ = |b_3| / |b_2|`; (d) β refitted on the projected cosines (lenient; the primary keeps A1a's β) |

## The rule (fixed before any output)

For each named head at each step read:

1. **In the class or not.** If M1 at full D has mean TV > 0.5 *(placed)*, the head is
   *outside the paper's class* (its attention is not isotropic content plus position) and has no
   M1 verdict; M2 is reported for it.
2. **D\*.** Otherwise D\* is the smallest D on V's ladder with mean TV(D) ≤ TV(full) + 0.05
   *(placed)* **and**, for D ≥ 1, below all 32 nulls at that D (rank 1 of 33, ties counted
   against; D = 0 has no subspace to randomise, so it passes on the TV clause alone).
   **D\* = 0** reads "positional: no content subspace". **D\* = 2** reads "the circle case" and
   names `U_2`'s type. No D ≤ 64 passing reads "no small invariant subspace".
3. The planted arms of unit 1 run with these constants. If one fails, the constants change before
   unit 2, with the reason recorded, and before any real output is opened.

**Prior, before any output (the user's, design-A):** 2D fails for the previous-token and
induction heads, because their attention reads position or exact identity. In this rule a
positional head reads **D\* = 0**, which is "not the circle" and agrees with the prior's verdict.
No lean is recorded for the other heads.

## Heads (the family, fixed)

| set | heads | why |
|---|---|---|
| previous-token | `L5H2` | B1's positive control |
| induction members | `L7H8`, `L12H5`, `L8H6`, `L11H14` | 7d's substantial members (`p7d_redundancy/status-7d.md` "Q1") |
| negative interactor | `L10H7` | ROADMAP B6; the most negative 7d effect |
| reference | 8 heads drawn from the other 378 with `numpy` seed 20261015, layers 0–22 | the population a named head is compared with |

Prompts: v1's 7 (`trackA_force.a1a.V1`, A1a's β exists only there; `prompts_hash` recorded). Long
passages are not read (no stored β). Steps: 143000 in unit 2; all 18 in unit 3 for the heads with
a D\* (including D\* = 0) at 143000; the reference heads always.

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen by the user (Blocked 50) |
| 1 | **weights only.** `trackB_circuit/b2a.py`: V's ladder, M1, M2, the nulls. Gate on synthetic tokens in d = 1024 (below). Weights-only rows over the 18 steps for the 14 heads: `U_2`'s type, `ρ_⊥`, the top `|b_a|` | every arm passes; the rows are written |
| 2 | **real tokens at 143000**: G0, then the rule for the 14 heads | a D\* or a recorded verdict per head |
| 3 | **the 18 steps** for the heads unit 2 names, and the reference heads | a D\* per head per step, or the reason one is missing |

### Unit 1's gate (planted, weights only, d = 1024, n = 500, causal)

| arm | planted | passes when |
|---|---|---|
| G1, circle | V with a degenerate pair `−ν I₂` (ν = 2) as its top-`|b|` block, the rest of the spectrum of the same rank-64 shape at `|λ| ≤ 0.5`; tokens run through the paper's causal map at β = 4, γ = 0.3 for 20 steps from a Gaussian start, so they lie near the plane; attention by the paper's model at full d | D\* = 2, `U_2` within 1e-3 rad of the planted plane, type "degenerate negative" |
| G1r, rotation | as G1 with `W = R_φ`, φ = 2.5 | D\* = 2, type "rotation" |
| G1d, D = 4 | a degenerate negative block of dimension 4 | D\* = 4 |
| G2, positional | attention from an offset profile alone (a previous-token peak) | D\* = 0 |
| G3, null | tokens isotropic Gaussian on the sphere, V a random Gaussian rank-64 map, attention by the paper's model at full d | D\* ≥ 16 or none |
| G4, out of class | attention from a random full-rank `Qᵀ K` (not isotropic) | "outside the paper's class" |
| G5, arithmetic | M1 at U = I equals the full model; M2 on the QK read-space at its full rank equals the real logit | max abs diff ≤ 1e-10 (float64) |

NUMERICS: float64 on CPU for the gate (small); unit 2–3's hooked pass on the GPU float32, as A1a,
with every comparison inside one process and device. The real attention's check against the stored
`attentions.npz` (G0) uses A1a's tolerance at 143000 (`check_pass`, 1e-4).

## Fences

- **One step, not dynamics.** In an untied network each head acts once; B2a tests whether one
  application's attention is reproduced by the reduced map, not whether depth iterates it
  (2609.28448 §6.2). The motion row (b) is the only V-side read.
- **Selection is not tested.** Occupancy (a) describes where the real tokens are; it does not show
  that the dynamics attracts them there.
- **The offset profile is fitted.** M1 has `|φ_h|` free parameters per head (leave one prompt
  out); M2 has none. A positional head's D\* = 0 rests on that fit.
- **β is A1a's**, fitted on the same 7 prompts with all of the frame; M1 inherits its fit.

## Worth challenging (for the freeze)

- **M1 primary, M2 beside** *(50)*. The exit says "reduced dynamics", which is the paper's map, so
  M1 (Q = K = I on V's subspace) is primary. Rejected as primary: M2 (the head's own QK on U),
  which answers "how much of the token a head reads" with a hard ceiling at 128, a question about
  QK rank rather than the paper's reduction. If the user meant M2, the rule's step 2 reads on M2's
  ladder instead and step 1 drops.
- **V's ladder is fixed, not searched.** No subspace is optimised against the attention, so there
  is no overfitting and no held-out split, but a smaller faithful U off V's ladder would be
  missed; D\* is "smallest on the ladder", an upper bound on the smallest of all.
- **TV with 0.5 and 0.05**: placed, and the planted arms check them; a per-head null band would
  replace the 0.05 if the gate shows it is too tight.
- **The offset profile in M1.** Without it a previous-token head cannot be reproduced at any D
  (the paper has no position), and "2D fails" would be guaranteed rather than measured.
