<!-- trackB_circuit/design-B2.md -->
# Track B — design: B2, spectra from the weights

Opened 2026-10-17 at the user's pick from `docs/ROADMAP.md` §9 order 3 (B2 over B3 and A3, asked in
session). Row: `docs/ROADMAP.md` §7 B2. Scan: `trackB_circuit/lit-B.md` "B2". Tier 1 throughout:
exploratory, unregistered; 410m is an exploration rung (`p8_scale_ladder/design-8.md`), so nothing
here can score a registered row, and B2 reads no other rung. **Proposed, not frozen (Blocked 61).**
**Revised after `/challenge-pr` on #218 (accept with changes, all five findings taken) before the
freeze:** members are compared with their own step's population, with step 0 kept as a descriptive
column (1); one scalar per column, labels descriptive, C6 read once over steps (2); C7's inputs named,
since it reads B2a's prompts (3); the LN1 frame primary for C1–C4, raw beside (4); C6 primary is
composition, random sets layer-matched (5).

## What B2 asks

For each member of the circuit, and for every head as the population it is compared with: what do
the weights alone say about attraction, repulsion and shared subspaces, and when does it change in
training? ROADMAP's deliverable is "a per-member table across training". The B2a close carried two
constraints into it (`status-B.md` "B2a on one page"): V's 2D blocks occur at the random-weight
rate, so **every rate is compared with step 0**; and the QK read rank is open.

## Exit condition and budget

- **Exit:** the per-member table over the 18 steps, every cell placed against its own step's
  population and marked for whether training moved it out of step 0's band, plus the member
  statistic of "The rule" per column. "Members do not differ from the population" is a verdict,
  not a failure.
- **Budget (proposed): 3 units on the record**, this design as unit 0, then a close-out page.
  The dense per-head OVs are on disk at all 19 steps, the checkpoints are cached, and C7 re-reads
  B2a's records, so no forward pass is planned. Alternative: 2 units, with the gate and the table
  in one session (B2a and A2 each ran over budget when a gate failed, `LESSONS.md` 10, which is
  why the gate is its own unit here).

## The math

`tools/math_checks/b2_symmetric_inertia.py` (`sympy` + numerics at d = 1024, r = 64, 4 checks pass):
(1) the symmetric part of any rank-r map `A Bᵀ` is `C J Cᵀ` with `C = [A B]`, so by Sylvester it has
**exactly r positive and r negative eigenvalues** whenever `[A B]` has full column rank: a head's
symmetrised OV (and symmetrised QK) has 64 negative eigenvalues by construction, and only the |λ|
weights carry information (`2609.22990`, made exact); (2) the count moves only when the read and
write spaces overlap (A = B gives (r, 0)); (3) under A → −A (Gaussian init) the attractive share
has mean ½, and it is a normalised trace, `a = ½(1 + tr V / ‖sym V‖_*)`; (4) the overlap of two
independent Haar r-subspaces, `tr(P_U P_W)`, has mean r²/d. **What it does not prove:** that a share
away from the population is attraction in the dynamics (attention weights the write; B2a found the
reduced dynamics reproduce no head); that Pythia's step 0 follows the Gaussian law beyond the sign
symmetry.

**Step 0's band is narrow (seen while answering `/challenge-pr` on #218, before any member value):**
the attractive share of the 384 step-0 heads spans 0.4963–0.5037 (2.5–97.5 %, raw frame), and
99.7 % of heads at step 512 and 97.9 % at 143000 sit outside it (population medians 0.445 and
0.487, IQR 0.437–0.463 and 0.408–0.581). The reviewer found the same for C2 (0.0119–0.0126; 100 %
and 99.2 % outside). So "outside step 0" says training moved a head, which nearly every head is;
it cannot single out members, and the member statistic compares with the same step instead.

## Objects, fixed here

| object | definition |
|---|---|
| frame (C1–C4) | **B2a's LN1 frame, primary** (`design-B2a.md` "Objects"): `V_h = diag(g_ℓ) C · OV_h` and `M_0 = diag(g_ℓ) C W_Qᵀ W_K C diag(g_ℓ) / √64`, `g_ℓ` the head's block's LN1 gain, C the centring projector, biases out. **Raw frame beside** (Phases 2, 6, 7e, 2d's convention). Both computed for all 384 heads; the primary is named here, before any output |
| frame (C6) | raw residual frame (the space heads write into and read from across layers) |
| OV of head h | the dense `ov_head<h>_layer_<l>` in `data/phase12/p2_eigenspectra_<ts>/ov_weights_pythia-410m-step<S>.npz` (the 19 runs `data/analysis/ov_per_head_series.json` names), orientation fixed by **G0** against `W_O W_V` from the checkpoint; rank ≤ 64 |
| M of head h | `W_Qᵀ W_K / √64` at offset 0 (2d's `qk_matrix`), from the cached checkpoint. Pythia rotates 16 of the 64 head dimensions (`rotary_pct` 0.25); at Δ = 0 rotary is the identity, so `M_0` is the full form at zero offset; the unrotated 48 dimensions' form is beside |
| `sym`, shares | `S = ½(X + Xᵀ)`; attractive share `a(X) = Σλ₊(S) / Σ|λ(S)|` (`⟨x, Xx⟩ > 0` is S's sign). For OV it is attraction's weight; for QK, "matches like with like" |
| write / read / key space | OV's top-r left / right singular vectors (7e's `svd` basis); key space = `W_K`'s top-r right singular vectors; r = 16 *(placed)*, 8 and 32 beside |
| steps | B2a's 18 (0, 2, 4, …, 143000; the series' step 1 dropped) |
| population | the 384 heads at the same step; **step-0 band** = 2.5–97.5 % of the 384 heads at step 0 |

## The columns (one scalar each, fixed before any member output)

| # | column (the scalar the rule reads) | beside | carried from |
|---|---|---|---|
| C1 | `a(V)`, the OV attractive share | raw frame; the series' V-eigenvalue split (`attractive_energy_fraction_core`) | ROADMAP B2 "symmetrised OV" |
| C2 | signed top weight `sign(λ₁) · |λ₁| / Σ|λ|` of S (λ₁ the largest |λ|) | the **negative pair** flag (λ₁, λ₂ both negative, `|λ₂|/|λ₁| ≥ 0.8` *(placed)*, the d = 2 repulsive reduction on weights), its rate against step 0's | "dominant negative eigenpairs"; B2a: compare rates with step 0 |
| C3 | `a(M_0)`, the QK symmetric share | 2d's asymmetry `‖Skew M_0‖_F / ‖M_0‖_F`; the unrotated part | Phase 10's `⟨Qx, Ky⟩`; 2d's condition 1 |
| C4 | 2d's signed cosine `ov_qk_alignment(V, M_0)` | 2d's `head_regime` label (`gradient_flow_condition.py`, reused) | "2d's gradient-flow regime test per member" (weights half; Fences) |
| C5 | *descriptive only*: `classify_ov_row(V, M_0)` (`table1_predictions.py`, reused), the members' rows and the population's row shares per step against step 0's | — | F18, against `2411.04990` Table 1 (left column only; Fences) |
| C6 | **K-composition overlap**: mean `tr(P_W(A) P_K(B)) / r` over the 10 ordered pairs of the 5 circuit heads (A's layer below B's; A's write space, B's key space), averaged over steps ≥ 512 | write–write overlap (7e's item, as asked); per-step values; the same with `L10H7`; chance r/d = 1/64 | 7e's free item; `/challenge-pr` on #218 finding 5 |
| C7 | QK read rank `r90` = smallest D on the QK ladder with `TV(D) ≤ 0.1 · TV(0)` *(placed)*, per member per step | the curve | B2a's carried "QK read rank (M2's ladder)" |

**C7's inputs (it reads tokens, through B2a's records; no new run):** `m2.qk` in
`data/b2a/2026-10-16/real.json` (143000, code `9fe738d5a008`), `data/b2a/2026-10-16-occ/step<S>/`
(code `37d4f2487738`) and `data/b2a/2026-10-16-occ2/step<S>/` (code `939e114f7afb`), v1's 7 prompts
(`976b62d677bc`); the rule's code is unchanged between the two (`status-B.md` "B2a unit 3"). 14 heads
only: no population, so C7 is each member against its own step 0 and B2a's 8 reference heads.

**Beside, not in the family:** each S's inertia (fact 1 says (64, 64); another is read/write
overlap); Phase 6's `U_pos` / `U_neg` / `U_A` dimensions of the members' union with its resolution
order, dimension-normalised (`p6_subspace/math-6.md` §7.2).

**Members (the family that counts):** `L5H2` (previous token), `L7H8`, `L12H5`, `L8H6`, `L11H14`
(induction, 7d's substantial members), `L10H7` (negative interactor), in six distinct layers
(5, 7, 12, 8, 11, 10). C6's "5 circuit heads" are the first five.

## The rule (fixed before any member output)

1. **Cells, descriptive.** Each (member, column, step) cell gets its percentile in its own step's
   population and a mark for whether it is outside step 0's band ("moved by training"). Cells are
   not readings one by one.
2. **The member statistic (C1–C4).** Per column, `T = mean over members and steps ≥ 512 of
   |p − ½|`, p the cell's percentile in its own step's population (two-sided: members are more
   extreme than their layers' other heads, either way). Null: 10,000 **layer-matched** random sets,
   one non-member head from each member's layer, the same steps. Reported: T's rank. *(Steps ≥ 512:
   B1's onset for `L5H2`, (256, 512]; 7d's members by 1000.)*
3. **C6.** The circuit's mean K-composition overlap over steps ≥ 512, ranked against 10,000 sets of
   5 non-member heads drawn one from each circuit head's layer (5, 7, 8, 11, 12), the same ordered
   pairs and steps. Per-step values are descriptive.
4. **Reading.** Five ranked statistics (C1–C4, C6); a column reads "members differ" at rank ≥ 0.99
   (0.05 / 5, one family), and below it the column reads "no difference shown". Leans are not
   readings.
5. **C7 and the negative-pair rate** are descriptive, with step 0 beside (no rank).
6. Any gate arm that fails changes the constants, with the reason recorded, before unit 2 reads any
   member value.

No prior recorded for any column. Seen before this rule was fixed: B2a's 143000 M2 values (the
`status-B.md` "B2a unit 2" table, "M2 on QK's top 64"), before C7's 0.1 was placed; the population
C1 / C2 distributions above. No member value at any step after 0 in any column B2 adds.

## Unit 1's gate (weights only, at the real shapes: d = 1024, 64 per head, 384 heads)

| arm | construction | passes when |
|---|---|---|
| G0 | stored dense OV against `W_O W_V` from the checkpoint, every head, steps 0 and 143000; the LN1 fold against B2a's `V_h` for its 14 heads | max relative error ≤ 1e-5, orientation fixed |
| G1 | the step-0 band's calibration: step 0's 384 heads split in halves 200 times, a half's rate outside the other half's band, C1–C4 | rate in [0.03, 0.08] (descriptive column only) |
| G2 | the member statistic's false pass **at steps 512 and 143000**: 200 layer-matched random 6-head sets of non-members as "members" against the same null, C1–C4 and C6 | rank ≥ 0.99 in ≤ 2 % of draws, per column |
| G3 | planted effect **in trained weights**: at 143000, 6 non-member heads in the members' layers with V shifted along `W_Oᵀ`-aligned directions until each head's C1 reaches its step's 90th percentile *(placed dose)*; dose curve 75th–99th beside | C1's T reaches rank ≥ 0.99 in ≥ 80 % of 200 draws at the 90th |
| G4 | planted repulsive pair: one 143000 head's V plus `−ν(e₁e₁ᵀ + e₂e₂ᵀ)` (e₁, e₂ orthonormal in V's write space) at ν = 2× its top |λ(S)| | C2 < 0 and the negative-pair flag set; at ν = 0 the head's own values are returned |
| G5 | planted composition at 143000: 4 of A's 16 write directions copied into B's key space, for the 10 ordered pairs of 5 layer-matched non-member heads | C6's rank ≥ 0.99; unplanted, the same heads' rank is within the null's central 90 % |
| G6 | planted gradient-flow head: `W_K := W_Q` and V := `M_0`'s symmetric part | asymmetry ≤ 1e-6, C4 ≥ 0.999, `head_regime` inside; `classify_ov_row` returns the row its definition implies |
| G7 | C7's reader on the named records: the 14 heads' `m2.qk` at all 18 steps, prompts hash and code as named; `TV(128) ≤ 1e-6` (exact, B2a's G5 (ii)) | all present, all exact |

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen by the user (Blocked 61) |
| 1 | `trackB_circuit/b2.py` (columns, the rule, the gate; reuses 2d's and B2a's functions); the gate | every arm passes, or a recorded change of constants |
| 2 | the table over the 18 steps (384 heads, 6 members), C7 from B2a's records, the beside rows | the table and the five ranks, with the input named |

## Fences

- **Weights only, except C7.** C1–C6 read no token; C7 re-reads B2a's attention on v1's 7 prompts
  (inputs named above). Whether a weights-level attraction moves tokens is B2a's question (answered:
  the reduced dynamics reproduce no head) and B3/B4's.
- **2d's regime: the weights half only.** `P-M1` attributes energy-monotonicity violations to heads
  outside the regime; B2 does not read energies, and 410m cannot score `P-M1` (Blocked 6 stands).
- **F18: Table 1's left column only, descriptive.** The geometric predictions (trimodal projections,
  `P-T1`) need activations and a registered rung.
- **`attention-10.md`'s A3 goes to B3.** It is activation-based (per-head received attention joined
  to 7d's sweep); B3's row already names it.
- **Offset 0 for QK.** Rotary makes the QK form offset-dependent on 16 of 64 dimensions; C3–C5 read
  Δ = 0 and the offset-free part beside, not the offset profile.
- **Composition is a subspace overlap, not a path.** C6 says A writes where B's keys read; it does
  not weight by what A writes there or by attention (B4's nudges are the functional test).

## Worth challenging (for the freeze)

- **The LN1 frame primary for C1–C4** (changed from raw after `/challenge-pr` on #218, finding 4):
  it is the frame the block reads and B2a's; raw is beside for Phases 2, 6, 7e and 2d's numbers.
  C6 stays raw, since its subspaces cross layers with different gains.
- **Same-step population, not step 0, as the member null** (finding 1). Step 0's band holds almost
  no trained head, so it is only the "moved by training" mark.
- **|p − ½| two-sided.** Members could be extreme in either direction (`L10H7` is the repulsive
  candidate); a one-sided statistic per column would be stronger if a direction were predicted.
  None is.
- **Layer-matched random sets.** Controls depth trends (the share moves by layer); it also makes the
  null smaller (15 non-members per layer).
- **Steps ≥ 512** for the statistic, from B1's and 7d's onsets; earlier action shows only in cells.
- **C6's primary is K-composition** (write into key space), the relation an induction circuit is
  known by; Q- and V-composition are not read. 7e's write–write overlap is beside, as asked.
- **C7's 0.1** was placed after 143000's M2 values had been seen (disclosed).
- **Dropping counts.** By fact 1 a sign count is structural, so C2 reads the top pair's signed
  weight; V's own eigenvalues are in B2a's unit 1 rows and the series.
- **Budget 3**, with the alternative of 2 above.
