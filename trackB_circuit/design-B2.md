<!-- trackB_circuit/design-B2.md -->
# Track B — design: B2, spectra from the weights

Opened 2026-10-17 at the user's pick from `docs/ROADMAP.md` §9 order 3 (B2 over B3 and A3, asked in
session). Row: `docs/ROADMAP.md` §7 B2. Scan: `trackB_circuit/lit-B.md` "B2". Tier 1 throughout:
exploratory, unregistered; 410m is an exploration rung (`p8_scale_ladder/design-8.md`), so nothing
here can score a registered row, and B2 reads no other rung. **Proposed, not frozen (Blocked 61).**

## What B2 asks

For each member of the circuit, and for every head as the population it is compared with: what do
the weights alone say about attraction, repulsion and shared subspaces, and when does it change in
training? ROADMAP's deliverable is "a per-member table across training". The B2a close carried two
constraints into it (`status-B.md` "B2a on one page"): V's 2D blocks occur at the random-weight
rate, so **every rate is compared with step 0**; and the QK read rank is open.

## Exit condition and budget

- **Exit:** the per-member table over the 18 steps, every cell placed against step 0's 384 heads
  and against the same step's population, plus the member statistic of "The rule" per column.
  "Members do not differ from the population" is a verdict, not a failure.
- **Budget (proposed): 3 units on the record**, this design as unit 0, then a close-out page.
  Everything is weights only, the dense per-head OVs are on disk at all 19 steps, and the QK
  read curves are already in B2a's records, so no forward pass is planned. Alternative: 2 units,
  with the gate and the table in one session (the gate is cheap; B2a and A2 each ran over budget
  when a gate failed, `LESSONS.md` 10, which is why the gate is its own unit here).

## The math

`tools/math_checks/b2_symmetric_inertia.py` (`sympy` + numerics at d = 1024, r = 64, 4 checks pass):
(1) the symmetric part of any rank-r map `A Bᵀ` is `C J Cᵀ` with `C = [A B]`, so by Sylvester it has
**exactly r positive and r negative eigenvalues** whenever `[A B]` has full column rank: a head's
symmetrised OV (and symmetrised QK) has 64 negative eigenvalues by construction, and only the |λ|
weights carry information (`2609.22990`, made exact); (2) the count moves only when the read and
write spaces overlap (A = B gives (r, 0)); (3) under A → −A (Gaussian init) the attractive share
has mean ½; (4) the overlap of two independent Haar r-subspaces, `tr(P_U P_W)`, has mean r²/d.
**What it does not prove:** that a share above step 0 is attraction in the dynamics (attention
weights the write; B2a found the reduced dynamics reproduce no head); that Pythia's step 0 follows
the Gaussian law beyond the sign symmetry (the gate checks it).

## Objects, fixed here

| object | definition |
|---|---|
| frame | the raw residual frame, as Phases 2, 6, 7e and 2d (LN gains not folded; B2a's LN1-frame values for the 14 heads are beside, from unit 1's `weights.json`) |
| V of head h | the dense `ov_head<h>_layer_<l>` in `data/phase12/p2_eigenspectra_<ts>/ov_weights_pythia-410m-step<S>.npz` (the 19 runs `data/analysis/ov_per_head_series.json` names), orientation fixed by **G0** against `W_O W_V` from the checkpoint; rank ≤ 64 |
| M of head h | `M_0 = W_Qᵀ W_K / √64` at offset 0 (2d's `qk_matrix`), from the checkpoint (cached, offline). Pythia rotates 16 of the 64 head dimensions (`rotary_pct` 0.25); at Δ = 0 rotary is the identity, so `M_0` is the full form at zero offset; the unrotated 48 dimensions' form (offset-free) is beside |
| `sym`, shares | `S = ½(X + Xᵀ)`; attractive share `a(X) = Σλ₊(S) / Σ|λ(S)|` (`⟨x, Xx⟩ > 0` is S's sign, `handoff-10.md` OV1's note). For OV it is attraction's weight; for QK it is "matches like with like" |
| write / read space | V's top-r left / right singular vectors (7e's `svd` basis); r = 16 *(placed)*, 8 and 32 beside |
| steps | B2a's 18 (0, 2, 4, …, 143000; the series' step 1 dropped) |
| population | all 384 heads at the same step; **step-0 band** = 2.5–97.5 % of the 384 heads at step 0 |

## The columns (the family, fixed before any output)

| # | column | quantity | carried from |
|---|---|---|---|
| C1 | OV attractive share | `a(V)` | ROADMAP B2 "symmetrised OV"; the share A1 reads per layer from V's eigenvalues is beside (`ov_per_head_series.json`, `attractive_energy_fraction_core`) |
| C2 | OV top symmetric pair | sign of S's largest-|λ| eigenvalue, its weight `|λ₁|/Σ|λ|`, and the flag **negative pair**: λ₁, λ₂ (by |λ|) both negative and `|λ₂|/|λ₁| ≥ 0.8` *(placed)*, the weights-only form of the d = 2 repulsive reduction | "dominant negative eigenpairs"; B2a: compare rates with step 0 |
| C3 | QK symmetry | `a(M_0)` and 2d's asymmetry `‖Skew M_0‖_F / ‖M_0‖_F` | Phase 10's `⟨Qx, Ky⟩` per head; 2d's condition 1 |
| C4 | gradient-flow regime | 2d's `ov_qk_alignment(V, M_0)` and `head_regime` label (`p2d_operator_activation/gradient_flow_condition.py`, reused as is) | "2d's gradient-flow regime test per member" (weights half only; Fences) |
| C5 | Table 1 row | `classify_ov_row(V, M_0)` (`p2d_operator_activation/table1_predictions.py`, reused) | F18, the V-spectrum atlas against `2411.04990` Table 1 (left column only; Fences) |
| C6 | shared subspace | mean overlap per direction `tr(P_U P_W) / r` over the 10 pairs of the 5 circuit heads, write and read spaces separately; chance r/d = 1/64 | 7e's free item ("OV-core principal angles between members") |
| C7 | QK read rank | from B2a's records (`m2.qk`, 14 heads × 18 steps, no new run): `r90` = smallest D on the QK ladder with `TV(D) ≤ 0.1 · TV(0)` *(placed)*, the curve beside | B2a's carried "QK read rank (M2's ladder)" |

**Beside, not in the family:** the inertia of each S (fact 1 says (64, 64); any other is read/write
overlap); Phase 6's `U_pos` / `U_neg` / `U_A` dimensions of the members' union with its resolution
order, dimension-normalised (`p6_subspace/math-6.md` §7.2); the series' V-eigenvalue split for the
members; C1–C3 in B2a's LN1 frame for the 14 heads.

**Members (the family that counts):** `L5H2` (previous token), `L7H8`, `L12H5`, `L8H6`, `L11H14`
(induction, 7d's substantial members), `L10H7` (negative interactor). C6's "5 circuit heads" are
the first five; `L10H7` enters C6 beside. C7 has no 384-head population: its comparison is each
member's own step 0 and B2a's 8 reference heads.

## The rule (fixed before any output)

1. **Cells, descriptive.** Each (member, column, step) cell gets its percentile in the step-0 band
   and in its own step's population. Cells are not readings one by one: at 5 % two-sided,
   6 × 18 × 6 cells hold about 30 outside the band by chance.
2. **The member statistic (per column C1–C5).** Over steps ≥ 512 *(B1's onset, `L5H2`, (256, 512];
   7d's members by 1000)*, the share of member cells outside the step-0 band, against the same
   share for 10,000 random 6-head sets drawn from the non-members at the same steps; the rank
   is reported (one-sided toward "members differ more"). Leans below rank 0.95 are not readings.
3. **C6.** The circuit's mean overlap at each step, against Haar (1/64), the same 10 pairs at
   step 0, and 10,000 random 5-head sets from the same step; a step reads "shared" at rank ≥ 0.95
   against the random sets **and** above the step-0 value.
4. **C7.** `r90` per head per step, with the step-0 value beside; no rank (14 heads only).
5. **Negative pair (C2).** Its rate in members and in the population at each step, beside the
   step-0 rate (B2a: the 2D block rate alone is random-level).
6. Any gate arm that fails changes the constants, with the reason recorded, before unit 2 reads
   any step after 0 (step 0 is the null and is read in the gate).

No prior recorded for any column. B2a's 143000 M2 values (`status-B.md` "B2a unit 2" table, "M2 on
QK's top 64") were read before C7's 0.1 was placed; the other 17 steps' curves were not.

## Unit 1's gate (weights only, at the real shapes: d = 1024, 64 per head, 384 heads)

| arm | construction | passes when |
|---|---|---|
| G0 | stored dense OV against `W_O W_V` from the checkpoint, every head, steps 0 and 143000 | max relative error ≤ 1e-5, orientation fixed |
| G1 | the null: step 0's 384 heads split in halves 200 times; a half's rate outside the other half's band, per column C1–C4 | rate in [0.03, 0.08]; C1's mean share within 0.02 of ½ (fact 3) |
| G2 | the member statistic's false pass: 200 random 6-head sets of step 0 as "members" against the rest at step 0 | rank ≥ 0.95 in ≤ 8 % of draws, per column |
| G3 | planted attraction: in 6 step-0 heads, `W_V ← W_V + κ W_Oᵀ` scaled so `a(V)` = 0.65 *(placed dose)*; dose curve at 0.55–0.80 beside | C1's statistic reaches rank ≥ 0.95 in ≥ 80 % of 200 draws at 0.65 |
| G4 | planted repulsive pair: one step-0 head's V plus `−ν(e₁e₁ᵀ + e₂e₂ᵀ)` (e₁, e₂ orthonormal, drawn in V's write space; rank ≤ 66) at ν = 2× its top |λ(S)| | C2 flags a negative pair; at ν = 0 it does not |
| G5 | planted shared subspace: 4 of 16 write directions made common to 5 step-0 heads | C6 reads "shared"; the unplanted 5 heads read Haar (0.016 ± the random sets' spread) |
| G6 | planted gradient-flow head: `W_K := W_Q` and V := `M_0`'s symmetric part | asymmetry ≤ 1e-6, alignment ≥ 0.999, `head_regime` inside; `classify_ov_row` returns a row-1 or row-2 class as its definition implies |
| G7 | C7's reader on B2a's records: the 14 heads' `m2.qk` present at all 18 steps; `TV(128) ≤ 1e-6` (exact, B2a's G5 (ii)) | all present, all exact |

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen by the user (Blocked 61) |
| 1 | `trackB_circuit/b2.py` (columns, the rule, the gate; reuses 2d's and B2a's functions); the gate on step 0 | every arm passes, or a recorded change of constants |
| 2 | the table over the 18 steps (384 heads, 6 members), C7 from B2a's records, the beside rows | the table and every column's member statistic, with its input named |

## Fences

- **Weights only.** No column reads a token. Whether a weights-level attraction moves tokens is
  B2a's question (answered: the reduced dynamics reproduce no head) and B3/B4's.
- **2d's regime: the weights half only.** `P-M1` attributes energy-monotonicity violations to heads
  outside the regime; B2 does not read energies, and 410m cannot score `P-M1` (Blocked 6 stands).
- **F18: Table 1's left column only.** The geometric predictions (trimodal projections, `P-T1`)
  need activations and a registered rung; B2 classifies heads, it does not test configurations.
- **`attention-10.md`'s A3 goes to B3.** It is activation-based (per-head received attention joined
  to 7d's sweep); B3's row already names it, so B2 drops it from its own scope.
- **Offset 0 for QK.** Rotary makes the QK form offset-dependent on 16 of 64 dimensions; C3–C5 read
  Δ = 0 and the offset-free part beside, not the offset profile.

## Worth challenging (for the freeze)

- **The raw frame, not B2a's LN1 frame.** Chosen for comparability with Phases 2, 6, 7e and 2d's
  code; the LN1 frame changes shares where LN1's gain is far from flat. Rejected as primary
  because B2a's LN1 values exist only for 14 heads, and the population needs all 384.
- **The step-0 band as the null.** One draw per head; the band is a population spread, not a
  per-head null. Rejected: Gaussian redraws at Pythia's init scale as the band (they test the
  initialiser, not the network that was trained). G1 measures the band's calibration.
- **Steps ≥ 512 for the member statistic.** Fixed from B1 and 7d's onsets; members that act
  earlier would be missed by the statistic (the cells still show them).
- **C7's 0.1** was placed after 143000's M2 values had been seen (disclosed in "The rule").
- **Dropping counts.** ROADMAP asks for "dominant negative eigenpairs"; by fact 1 a count is
  structural, so C2 reads the top pair's weight instead. If the user meant V's own (nonsymmetric)
  eigenvalues, B2a's unit 1 rows and the series already hold them for the members.
- **Budget 3**, with the alternative of 2 above.
