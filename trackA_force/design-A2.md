<!-- Proposed 2026-10-16, not frozen: STATE Blocked 56. Revised after /challenge-pr on #212 (findings 1–7 taken). -->
# Track A — design: A2, which components push the tokens apart?

**Question** (`docs/ROADMAP.md` §6 A2): which heads and MLPs move pythia-410m's tokens against
the self-attention field, or off it, from which step; and when the top ones are ablated, does the
cloud collapse further, does the energy over depth turn monotone, at what loss?
**Opened 2026-10-16** after B2a's close (Blocked 55 (a)). Scan: `lit-A2.md`. Tier 1, 410m,
unregistered.

What A1 left (`status-A.md` "A1 on one page"): the projected integration time `T_proj` is below
t* at every step; the net moves several times t* in total but over 90 % of that is off the field;
in 29 cells it moves against it. So two things are kept apart: **against** (a negative projected
step, which is anti-collapse by definition) and **off** (motion orthogonal to the field, which is
not anti-collapse by definition). The exit is about "against"; "off" is read beside.

## Exit condition and budget

- **Exit:** unit 5's rule reads "these N components carry the anti-collapse force" as
  **supported**, **refused** or **mixed** (one primary readout only, named), on 410m at tier 1;
  or a recorded reason it cannot be read (e.g. the step-0 null or the planted control fails).
  Before that, unit 2 says whether A1's closed number moves under the β refit.
- **Budget: 5 units**, each its own PR, plus a close-out (writing only, as B2a's #211):
  (1) this design and the scan; (2) the β refit (Blocked 54 (a)); (3) the per-component
  instrument and its calibration gate; (4) the batch over 18 steps and the ranking; (5) the
  ablation, its nulls and the reading. At the budget A2 closes at what it knows.

Every threshold below marked *placed* is set by judgement before any output, not calibrated.

## Unit 2: the β refit at 143000 (Blocked 54 (a); free, CPU, stored attention)

- **Estimators, a 2 × 2:** {M1 at full D, A1a's `fe_full`} × {all heads, heads with
  `partial_r2 ≥ 0.10`} (*placed*; a layer with fewer than 4 kept heads gets no β, *placed*).
  M1 is `trackB_circuit/b2a.py`'s `β cos(u_i, u_j) + φ(i − j)`, the softmax's own cross-entropy,
  B2a's offset bins. **Primary: M1, all heads**, because it is the likelihood of the attention the
  head produced, weighted by mass; `fe_full` weights every pair equally, near-zero entries
  included (`lit-A2.md` §3). The 2 × 2 says whether a move is the estimator's or the floor's.
- **Per prompt**, as A1's bracket is: each head fitted on one prompt's rows (B2a's Newton
  `fit`, called on that prompt alone), so each prompt has its own 23 layer βs and bracket.
- **Inputs:** A1a's records at 143000 (`data/trackA/a1a_batch_2026-10-15`, code `a08544e9b644`),
  the 7 v1 prompts (`976b62d677bc`), the stored `attentions.npz` A1a read (`a1a.RUNS`), the same
  unit LN1 rows, position 0 out.
- **Checks first, or the unit refuses:** (a) B2a's own leave-one-prompt-out path through the
  same `fit` reproduces B2a unit 2's `beta_full` on its 14 heads (1e-6); (b) the 2 × 2's
  `fe_full`, all-heads cell reproduces A1a's stored layer βs at 143000 (1e-9).
- **The rule (fixed now):** **A1's number moves** if the primary's refit p50 lies outside A1's
  [p10, p90] at 143000 in ≥ 4 of 7 prompts (*placed*). Then, in the same unit, the refit at all
  18 steps (still free) and a line under `status-A.md` "Corrections received". **Whatever the
  bar says**, A1b's cell at 143000 is re-run with the primary's layer βs (t* and `T_proj`, the
  field at block l at the refit layer β; `a1a.py` with a β override), since it is free and a
  move below the bar can still flip a cell's verdict (Blocked 56 (b), recommended; `/challenge-pr`
  on #212 agreed).
- **What it also touches:** the project's β convention (Blocked 9: 3.5 [1.6, 5.6], `fe_full`,
  1d's v1 bands) is the same estimator. M1's β per band at 143000 is read beside; if it falls
  outside [1.6, 5.6], that goes to the user as a new Blocked item (it would restale every card
  reading β), and nothing is changed silently.

## Unit 3: the per-component instrument and its gate

**Components** per block l (Pythia's parallel residual): each head's own `W_O^h` slice of
attention's output (16), the attention output bias, the MLP (with its bias): 24 × 18 = 432 per
step. The sink's part is split out of each head (Phase 10's row A5) as `A_i0 · v_0 W_O^h`
(received attention × position 0's value), and likewise for the head's most-received non-zero
key when that key is a fixed token (1e's newline sink), so "sink-only" is read for any fixed key;
sink parts are beside, not counted. The `W_O^h` split is exact as a sum but is one choice of
split (`lit-A2.md` §1, 2604.23681).

**Per-component step: the path integral (Aumann–Shapley).** A1's step runs each update through
the LN1 frame and takes medians, so it does not add over components. A2 integrates the frame
map's Jacobian along the straight path from the block's input x to its end x + v (v the block's
whole update): `Δ_c = ∫₀¹ J_l(x + s v) ds · v_c`, by Gauss–Legendre on 16 nodes (*placed*). Then
`Σ_c Δ_c` is the frame chord `frame(x + v) − frame(x)` up to quadrature error, which G0 bounds,
so no block is flagged or dropped for linearisation. The tangent projection at U is linear, so
per token `proj_c = ⟨Δ_c^tan, F⟩ / |F|²` sums exactly to the block's; A1's gnomonic lift is
applied to the total only, and its ratio to the unlifted total is recorded beside. F is the causal
field at the block's layer β (unit 2's primary). Per (block, component) the **mean** over
positions 1…n−1, so the sum over components is the block's mean.

**Off the field, additively.** The per-token off-field vector `o_c = Δ_c^tan − proj_c F` sums
over c to the block's `o`. A component's share is `⟨o_c, o⟩ / |o|²` (its projection on the total
off-field move), which sums to 1, so components that cancel each other are not both counted.
`proj_c` with the shared drift removed (A1's `drift_free_slope`, linear in the move) beside.

**The gate** (`CLAUDE.md` "Calibration gate"; on the GPU, before any real reading), on a
synthetic stack of 24 identity-QK blocks at β 3.5 with the measured start's spectrum:
- G0: Σ_c Δ_c equals the frame chord to 1e-6 relative (quadrature; *placed*).
- G1 (planted effect): one head with V = −I among 15 with V = +I: that head reads negative and
  ranks first in "against" in every block it is planted in.
- G2 (null, family-level): on a stack with Pythia's own init (normal at its σ, 50 seeds), the
  statistic unit 4 reads, `|S⁻|` (the against set's size), and the share of the against mass in
  its top 43 components, each fall inside the 50 seeds' 5–95 % range on a 51st seed. A sign at
  init is not a reading; this asks that the ranking statistic, not each of 432 components, be
  ordinary there.
- G3 (off-field): one head whose V rotates in a plane orthogonal to the field: its off share ranks
  first, `proj_c` ≈ 0.
- G4 (shared drift): a constant added to every token by the MLP: the MLP's drift-removed `proj_c`
  reads ≈ 0.
Any arm failing stops the unit; the fix is a Blocked item, not a silent change.

## Unit 4: the batch and the ranking (18 steps, v1; descriptive)

Over A1a's 18 steps and v1's 7 prompts, every component's mean `proj_c` and off share, the median
over prompts per (step, block, component).
- **Against set** `S⁻_t`: the fewest components whose negative `proj_c` sum to ≥ 50 % of all
  negative `proj_c` at step t (*placed*). **Along set** `S⁺_t`: likewise for positive `proj_c`.
  **Off set** `S⁰_t`: the fewest components whose off shares sum to ≥ 50 %.
- **Few** if `|S| ≤ 43` (10 % of 432, *placed*); **cooperating** otherwise (1e's open thread,
  `lit-A2.md` §4's prior). Gini over components beside.
- Read beside, the carried items: attention against the MLP at 1000–8000; block 23 at 143000;
  whether attention's `proj_c` is drift-robust; the sink's part, at position 0 and at any fixed
  key, and whether it stays ≤ 0.05 of the shared update (1e at close); which component carries the
  off-field share (the MLP first, `lit-A2.md` §1).
- This unit ranks; it tests nothing. Its v1 sets are not the ones unit 5 ablates.

## Unit 5: the ablation, its nulls and the reading

- **Rank on one text, test on another.** At steps 2000, 8000, 143000 (the three regimes 1e and A1
  found: OV turn, sum-over-heads ascent, end), `S⁻`, `S⁺` and `S⁰` are re-ranked by unit 4's code on
  the 8 long passages (hash `ba605f4e14b5`; GPU hooked pass at those 3 steps), and the ablation
  constant is each component's mean output over the same passages' tokens. Readouts are on v1, so
  v1 neither picks the set nor supplies the mean (`lit-A2.md` §2).
- **Ablated:** the long-passage `S⁻` (the claim's arm), and `S⁺` as the opposite-sign control
  (same count as `S⁻`, its top components), mean-ablated at positions 1…n−1, position 0 untouched.
  `S⁰` is ablated beside only.
- **Null:** 20 random sets per arm (seed 20261016), matched in count, component type
  (head / MLP / bias), block band (L1–8, L9–16, L17–24) **and size**: each drawn component comes
  from the same band, type and decile of mean `|v_c|` as the one it replaces, so "ablating big
  things makes tokens alike" is in the null.
- **Primary readouts on v1:** (i) the last block's mean cosine (re-clustering), (iii) the share of
  block transitions where `E_β` rises (monotone if 1.0). **Manipulation check, not evidence:**
  (ii) cumulative `T_proj` over t* (it rises by how `S⁻` is chosen). Beside: (iv) next-token loss
  on v1, quoted, never a condition.
- **Supported** if ablating `S⁻` raises both (i) and (iii) above all 20 of its size-matched random
  sets and above the `S⁺` ablation, at ≥ 2 of the 3 steps. **Mixed** if only one of (i), (iii)
  does (named). **Refused** otherwise. `S⁰` passing the same test is reported as "the off-field
  motion carries it", beside the exit, never as the exit's "supported". A set that is not "few"
  still runs; the claim then reads "a cooperating set of N".
- **The rule's own calibration** (`CLAUDE.md` "Calibration gate"): before the 3 real steps, units
  4–5 run end to end (rank on the long passages, ablate, readouts on v1) at **step 0** as the
  planted null: the rule must not read "supported" there. And on the synthetic stack, G1's planted
  head as `S⁻` must read "supported" against its own size-matched null. If either fails, the
  reading is not made and the failure goes to the user.

## Fences

- **No registration and no adjudication**; 410m is exploration. v2's 12 held-out prompts are not
  read (`core/holdout.py`).
- **The skip is not a component.** Ablations remove updates, never the residual, whose own
  anti-collapse share (2103.03404) is outside A2.
- **A1's closed number changes only through unit 2**, recorded under `status-A.md` "Corrections
  received".
- **LR warmup** overlaps 512 and 1000–2000; unit 4 reports breaks there with that confound.
- `a1a.ln_from` (duplicating `u2_attn.ln_from`) is merged into one in unit 3, the next code to
  extend the pass (A1 Parked; changes no number).

## Worth challenging (for the freeze)

1. **M1 as unit 2's primary.** It has an offset profile `fe_full` lacks in form; a β that drops
   because φ absorbs content is the alternative reading. The 2 × 2 separates estimator from floor.
   Rejected: a mass-weighted `fe_full` (a third estimator nobody has checked).
2. **Unit 2's "moves" bar** (4 of 7 prompts) is coarse; the free re-run of A1b's 143000 cell
   covers what it misses.
3. **The path integral at 16 nodes.** Exact to quadrature; it replaces a linearisation whose
   residual `PROJECT.md` §3.8.2 found large outside 2000–4000 (`/challenge-pr` on #212, finding
   4). Rejected: sampled Shapley through the frame (1e's method).
4. **The `W_O^h` per-head split** (2604.23681's non-identifiability). Rejected: a learned
   re-basis, which would make "a head" not the model's head.
5. **Mean ablation from the long passages**, not resample or optimal (2409.09951). Optimal
   ablation is better founded but needs a fit per component; beside only if the budget allows.
6. **Three ablation steps, not 18**, and a step-0 null rather than 50 re-inits: the re-inits would
   cost a hooked pass each; step 0 is the trained model's own init.
7. **The 50 % mass cut and "few" ≤ 10 %** are placed; a different cut changes `|S|`, and the
   size-matched null is what keeps that from deciding the reading.
