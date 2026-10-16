<!-- trackB_circuit/design-B2a.md -->
# Track B — design: B2a, the 2D spike

Opened 2026-10-15 by Blocked 49 (a) (the user's: B2a after A1 closed). Origin: the user's question
of 2026-10-14, can a head's dynamics be read in the paper's d = 2 reduction (`trackA_force/design-A.md`
"Frozen 2026-10-14"; `docs/ROADMAP.md` §7 B2a). Scan: `trackB_circuit/lit-B.md` "B2a". Tier 1
throughout: exploratory, unregistered; 410m is an exploration rung.

**Frozen 2026-10-15 by Blocked 50 (a)** (the user's): as written, **budget 4 units on the
record** (this design as unit 0), **M1 primary**, M2 beside; the placed constants and the 8
reference heads as below. Unit 1's gate as built differs from the table below where the
construction could not test what the arm names: "Unit 1's gate, as built". **Revised after `/challenge-pr` on #207 (accept with changes), before any
output:** V's ladder on its nonzero spectrum by `|λ|` (1); position 0 out and rows renormalised
(2); β and the offset profile fitted per rung by cross-entropy, A1a's β beside (3); 320 nulls and
G3 as a false-pass rate (4); the reference pool stated (5).

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
| β of head h | fitted per rung (below); A1a's `fe_full` for that (step, prompt, head) beside, as recorded (`trackA_force/a1a.py`) |
| γ of head h | `γ / (1 − γ)` = the median over tokens of `‖diag(g_ℓ) C o_i^h / σ_i‖ / ‖LN1_ℓ(x_i)‖`, head h's own output `o^h` mapped into the frame (σ_i LN1's scale at token i) (read in unit 2; for unit 1's weights-only ordering, γ = 0.1, placed, with the ordering by `Re λ` beside, which is γ's small-γ limit) |
| V's ladder | real Schur form of `V_h`, **its nonzero spectrum only** (rank ≤ 64; the ≥ 960-dimensional kernel is the part of the frame the head does not act on, and at `b = 1 − γ` it would outrank every direction with Re λ < 0, `/challenge-pr` on #207 finding 1); invariant blocks (a real eigenline, D += 1; a complex pair, D += 2, kept whole) ordered by `|λ_a|`, largest first. `U_D` = the span of the first blocks with total dimension D. Ladder D ∈ {0, 1, 2, 3, 4, 6, 8, 16, 32, 64} capped at the rank, a rung that would split a pair is moved up by one. Beside, weights only: the paper's selection order `|b_a|` at the head's γ, and whether any nonzero block beats the kernel's `1 − γ` (if none does, the paper's selection mechanism does not operate in that head) |
| reduced attention (M1, primary *(50)*) | `log Â_ij = β_D · cos(Uᵀu_i, Uᵀu_j) + φ_h(i − j) − log Z_i`, causal, keys `1 ≤ j ≤ i`. **Position 0 out as key and query, and the real rows renormalised over keys j ≥ 1** (A1a's convention; induction heads and `L10H7` put 47–93 % on position 0 at 143000, finding 2); the sink mass per head is reported beside. **`β_D` and `φ_h` are fitted per rung** by minimising the cross-entropy of `Â` against the renormalised real rows on the other 6 prompts (leave one prompt out), so the fit and the TV score read the same object and no rung is held at the full frame's sharpness (finding 3); A1a's `fe_full` β is reported beside, and M1 at full D must reproduce it in sign and order. `D = 0` is `φ_h` alone, the positional rung |
| M1 at full D | the same with U = I (`fe_full`'s own isotropic model): how well the paper's class fits the head at all |
| head's own QK (M2, beside *(50)*) | the real logit with `y → U Uᵀ y` in both slots, biases and rotary kept; on V's ladder and on the QK read-space ladder (SVD of `[W_Q; W_K]`, exact at its rank ≤ 128) |
| real attention | recomputed from the weights on the hooked pass's LN1 rows; **G0** checks it against the stored `attentions.npz` |
| faithfulness | per query row i ≥ 1, total variation `TV_i = ½ Σ_j |A_ij − Â_ij|`; the head's statistic is the mean over rows and the 7 prompts |
| null | K = 320 Haar-random D-dimensional subspaces of the frame per (head, D), seeded, each with its own fitted `β_D` and `φ_h` by the same procedure. 320 so that "below all nulls" is 1/321 per rung and ≤ 0.03 over the ≤ 10 rungs the rule tries (finding 4) |
| beside | (a) **occupancy**: the median share `‖Uᵀu_i‖²` (random: D/1024), the paper's attraction question; (b) **motion**: the cosine between the reduced update `γ W Σ_j Â_ij v_j` lifted by U and head h's real output in the frame, tangent part; (c) the type of `U_2` (degenerate negative, rotation pair, mixed) and `|λ_3| / |λ_2|`; (d) A1a's β against `β_D`; (e) the position-0 mass |

## The rule (fixed before any output)

For each named head at each step read:

1. **In the class or not.** If M1 at full D has mean TV > 0.5 *(placed)*, the head is
   *outside the paper's class* (its attention is not isotropic content plus position) and has no
   M1 verdict; M2 is reported for it.
2. **D\*.** Otherwise D\* is the smallest D on V's ladder with mean TV(D) ≤ TV(full) + 0.05
   *(placed)* **and**, for D ≥ 1, below all 320 nulls at that D (rank 1 of 321, ties counted
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
| reference | 8 heads drawn from the other 362 heads of layers 0–22 (A1's blocks; 368 less the 6 named) with `numpy` seed 20261015 | the population a named head is compared with |

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
| G1, circle | V of rank 64 with a degenerate pair `−ν I₂` (ν = 2) as its largest-`|λ|` block, the rest of its nonzero spectrum at `|λ| ≤ 0.5`; tokens run through the paper's causal map at β = 4, γ = 0.3 for 20 steps from a Gaussian start, so they lie near the plane; attention by the paper's model at full d | D\* = 2, `U_2` within 1e-3 rad of the planted plane, type "degenerate negative" |
| G1r, rotation | as G1 with `W = R_φ`, φ = 2.5 | D\* = 2, type "rotation" |
| G1d, D = 4 | a degenerate negative block of dimension 4 | D\* = 4 |
| G2, positional | attention from an offset profile alone (a previous-token peak) | D\* = 0 |
| G3, null | tokens isotropic Gaussian on the sphere, V a random Gaussian rank-64 map unrelated to the attention, attention by the paper's model at full d; 20 seeds | D\* = none in ≥ 19 of 20 (V's ladder is no better than a random subspace, so a pass is a false pass; finding 4) |
| G4, out of class | attention from a random full-rank `Qᵀ K` (not isotropic) | "outside the paper's class" |
| G5, arithmetic | M1 at U = I equals the full model; M2 on the QK read-space at its full rank equals the real logit | max abs diff ≤ 1e-10 (float64) |

### Unit 1's gate, as built (2026-10-15, before any real token was read)

| item | the table above | as built | why |
|---|---|---|---|
| G1–G1d tokens | 20 steps of the paper's causal map at β = 4, γ = 0.3 from a Gaussian start | uniform on the planted block's unit sphere, 10 % of each token's norm² in a random direction off it (`G_OFF`); ν = 2 kept | the map cannot give spread tokens near the plane with this V: at ν = 2 the plane's `|b| = |1 − γ − γν| = 0.1` is below the kernel's `1 − γ = 0.7`, so tokens leave it (median occupancy 1e-6, scratch run); at ν = 8 they reach it but collapse to one cluster flipping sign each step (`γ_eff = 0.77 > ½` contracts deviations), whose attention is near uniform: the first G1 run read D\* = 0 (TV(0) = 0.04), the right reading for those tokens |
| G3 tokens | isotropic Gaussian on the sphere | unit vectors in a random 8-dimensional subspace unrelated to V | isotropic tokens in d = 1024 have off-diagonal cosines ±1/32, so the paper's attention at β = 4 is near uniform plus self and D = 0 would pass on its TV clause; the arm could not return "none" |
| G5 | (i) M1 at U = I; (ii) M2 at the QK space's full rank | (i), (ii) on the 14 heads' real weights at 143000 and random LN1 rows, and **(iii)** the head's attention rebuilt from the weights against HF's on a seeded random-token pass (the code G0 will use) | (iii) is new; HF's pass is float32 (its float64 causal mask overflows in transformers 4.57), so (iii) is read in units of float32 eps × the head's largest \|logit\| and passes at ≤ 10; two reference heads reach \|logit\| ≈ 3e3–8e3 on random tokens |
| φ | "the head's offset profile" | piecewise constant on 23 offset bins: 0 … 15 one each, then [16, 32), [32, 64), …, [1024, 2048) | the design named no form; the logit is linear in (β, φ), so the fit is convex (Newton, ridge 1e-6) |
| head statistic | mean over rows and the 7 prompts | mean over prompts of the per-prompt mean over rows | equal weight per prompt |
| nulls | 320 per rung | run in chunks of 32 and stopped at the first null that ties or beats V's rung (the verdict needs no more); `--all-nulls` runs all | cost |
| U₂'s type | degenerate negative, rotation, mixed | as named, "degenerate" = two real λ of one sign within 5 % (placed); "degenerate positive" added | — |
| ρ_⊥ | `max \|b_a\| / \|b_top\|` | over every other block and the kernel's `1 − γ`; ρ_⊥ > 1 reads "the kernel beats every block" | the kernel is transverse to U too |
| NUMERICS | gate float64 on CPU | gate float64 on the GPU; weights float64 on CPU, from float32 checkpoints cast exactly | the memory rule (heavy synthetic work on the GPU); float64 kept |

**Fixed for unit 2 before any real token (`/challenge-pr` on #208, findings 2 and 3):** G0's
tolerance is G5 (iii)'s, the attention difference over float32 eps × the head's largest |logit|,
≤ 10, not `check_pass`'s 1e-4 (that bound is on hidden rows; G5 (iii) already measured 3.8e-4 on
`L18H1`'s attention). Each null's median token occupancy `‖Uᵀu_i‖²` is recorded beside its TV,
so a V rung that wins only by sitting where LN1's gain puts the tokens is visible; a
token-matched comparator is not added (it would be a new null, the user's).

NUMERICS: float64 on CPU for the gate (small); unit 2–3's hooked pass on the GPU float32, as A1a,
with every comparison inside one process and device. The real attention's check against the stored
`attentions.npz` (G0) uses A1a's tolerance at 143000 (`check_pass`, 1e-4).

## Fences

- **One step, not dynamics.** In an untied network each head acts once; B2a tests whether one
  application's attention is reproduced by the reduced map, not whether depth iterates it
  (2609.28448 §6.2). The motion row (b) is the only V-side read.
- **Selection is not tested.** Occupancy (a) describes where the real tokens are; it does not show
  that the dynamics attracts them there.
- **The offset profile and β are fitted.** M1 has `|φ_h| + 1` free parameters per rung (leave
  one prompt out); M2 has none. A positional head's D\* = 0 rests on that fit. A1a's β enters
  only beside, so B2a's dependence on A1a is now the frame and the beside row, not the rule.
- **Position 0 is out.** The sink carries most of some heads' attention; B2a reads the rest.

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
