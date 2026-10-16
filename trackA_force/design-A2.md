<!-- Proposed 2026-10-16, not frozen: STATE Blocked 56. -->
# Track A — design: A2, which components push the tokens apart?

**Question** (`docs/ROADMAP.md` §6 A2): which heads and MLPs move pythia-410m's tokens against
the self-attention field, or off it, from which step; and when the top ones are ablated, does the
cloud collapse further, does the energy over depth turn monotone, at what loss?
**Opened 2026-10-16** after B2a's close (Blocked 55 (a)). Scan: `lit-A2.md`. Tier 1, 410m,
unregistered.

What A1 left (`status-A.md` "A1 on one page"): the projected integration time `T_proj` is below
t* at every step; the net moves several times t* in total but over 90 % of that is off the field;
in 29 cells it moves against it. So "the anti-collapse force" has two parts A2 keeps apart:
**against** (a negative projected step) and **off** (motion orthogonal to the field). A2 asks
which components carry each, then tests the ranking by ablation.

## Exit condition and budget

- **Exit:** "these N components carry the anti-collapse force" is **supported** or **refused**
  on 410m at tier 1 by unit 5's rule below, or a recorded reason it cannot be read. Before that,
  unit 2 says whether A1's closed number moves under the β refit.
- **Budget: 5 units**, each its own PR, plus a close-out (writing only, as B2a's #211):
  (1) this design and the scan; (2) the β refit (Blocked 54 (a)); (3) the per-component
  instrument and its calibration gate; (4) the batch over 18 steps and the ranking; (5) the
  ablation and the reading. At the budget A2 closes at what it knows.

## Unit 2: the β refit at 143000 (Blocked 54 (a); free, CPU, stored attention)

- **Estimator:** M1 at full D (`trackB_circuit/b2a.py`: `β cos(u_i, u_j) + φ(i − j)`, the
  softmax's own cross-entropy, offset bins as B2a, fitted on all 7 prompts) per head, all
  24 × 16 heads, on the same unit LN1 rows, position 0 out, as A1a's `fe_full`. Primary because
  it is the likelihood of the attention the head actually produced, weighted by mass; `fe_full`
  weights every pair equally, near-zero entries included (`lit-A2.md` §3).
- **Beside:** A1a's `fe_full` restricted to heads with `partial_r2 ≥ 0.10` (placed now: the
  floor ROADMAP names as the alternative; heads below it get no β, and a layer with fewer than
  4 kept heads gets none).
- **Inputs:** A1a's records at 143000 (`data/trackA/a1a_batch_2026-10-15`, code `a08544e9b644`),
  the 7 v1 prompts (`976b62d677bc`), the stored `attentions.npz` A1a read (`a1a.RUNS`).
- **Check first:** run through B2a's own leave-one-prompt-out path, M1's β at full D reproduces
  B2a unit 2's `beta_full` on its 14 heads (1e-6) or the unit refuses.
- **The rule (fixed now):** per prompt, the refit layer βs (median over the layer's 16 heads,
  blocks 0–22) give a refit bracket (p10 / p50 / p90), as A1's. **A1's number moves** if the
  refit p50 lies outside A1's [p10, p90] at 143000 in ≥ 4 of 7 prompts. Then, in the same unit:
  t* and `T_proj` re-run at 143000 with the refit layer βs (`a1a.py` with a β override; the
  field at block l uses the refit layer β), the verdict against t* re-read, and the refit run at
  all 18 steps (still free); a line goes under `status-A.md` "Corrections received". Otherwise
  A1's number stands and the refit is recorded beside.
- **What it also touches:** the project's β convention (Blocked 9: 3.5 [1.6, 5.6], `fe_full`,
  1d's v1 bands) is the same estimator. The unit reads M1's β per band at 143000 beside; if it
  falls outside [1.6, 5.6], that goes to the user as a new Blocked item (it would restale every
  card reading β), and nothing is changed silently.

## Unit 3: the per-component instrument and its gate

**Components** per block l: each head's own `W_O^h` slice of attention's output (16), the
attention output bias, the MLP (with its bias); the sink's part is split out of each head as
`A_i0 · v_0 W_O^h` (received attention × position 0's value, Phase 10's row A5). 24 × 18 = 432
components per step (sink parts beside, not counted). This split is exact as a sum but is one
choice of split (`lit-A2.md` §1, 2604.23681).

**Per-component step.** A1's step runs each update through the LN1 frame and takes medians, so
it does not add over components. A2 linearises: `Δ_c = J_l(x) v_c`, the frame map's Jacobian at
the block's input applied to component c's raw update, so `Σ_c Δ_c` is the block's first-order
tangent move exactly. Per token, `proj_c = ⟨Δ_c, F⟩ / |F|²` and `off_c = |Δ_c − proj_c F| / |F|`
(F the causal field at the block's layer β, after unit 2); per (block, component) the **mean**
over positions 1…n−1, so the sum over components is the block's mean. Recorded per cell: the
linearisation residual against A1's nonlinear mean step (as `PROJECT.md` §3.8.2 did; a block
whose residual exceeds 0.5 is flagged and read beside, not dropped), and `proj_c` with the shared
drift removed (A1's `drift_free_slope`, per component).

**The gate** (`CLAUDE.md` "Calibration gate"; on the GPU, before any real reading), on a
synthetic stack of 24 identity-QK blocks at β 3.5 with the measured start's spectrum:
- G0: Σ_c over components equals the block's first-order move (1e-6 relative).
- G1 (planted effect): one head with V = −I among 15 with V = +I: that head reads negative and
  ranks first in "against" in every block it is planted in.
- G2 (null): random-init weights: no component's `proj_c` outside the spread of 50 re-inits'
  (a sign is not a reading at init).
- G3 (off-field): one head whose V rotates in a plane orthogonal to the field: `off_c` ranks it
  first, `proj_c` ≈ 0.
- G4 (shared drift): a constant added to every token by the MLP: `proj_c` with drift removed
  reads ≈ 0 for the MLP.
- G5: the linearised step matches the nonlinear step within 0.1 relative on the planted stack
  at the measured step size.
Any arm failing stops the unit; the fix is a Blocked item, not a silent change.

## Unit 4: the batch and the ranking (18 steps, v1)

Over A1a's 18 steps and v1's 7 prompts, every component's mean `proj_c` and `off_c`, the median
over prompts per (step, block, component).
- **Against set** `S⁻_t`: the fewest components whose negative `proj_c` sum to ≥ 50 % of all
  negative `proj_c` at step t. **Off set** `S⁰_t`: likewise for `off_c`.
- **Few** if `|S| ≤ 43` (10 % of 432); **cooperating** otherwise (1e's open thread, `lit-A2.md`
  §4's prior). Gini over components beside.
- Read beside, the carried items: attention against the MLP at 1000–8000; block 23 at 143000;
  whether attention's `proj_c` is drift-robust; the sink's part; which component carries the
  off-field share (the MLP first, `lit-A2.md` §1).

## Unit 5: the ablation and the reading

- **Steps:** 2000, 8000, 143000 (the three regimes 1e and A1 found: OV turn, sum-over-heads
  ascent, end). **Ablated:** `S⁻_t` and, as a second arm, `S⁰_t`, each mean-ablated (positions
  1…n−1; position 0 untouched), the mean over the 8 long passages' tokens at that step (hash
  `ba605f4e14b5`; GPU forward pass, so v1 is never ablated to its own mean; `lit-A2.md` §2).
- **Null:** 20 random sets per arm, matched in count, component type (head / MLP / bias) and block
  band (L1–8, L9–16, L17–24), seed fixed now (20261016).
- **Readouts on v1:** (i) the last block's mean cosine (re-clustering), (ii) cumulative `T_proj`
  over t* at the step's bracket (unit 2's βs), (iii) the share of block transitions where the
  energy `E_β` rises (monotone if 1.0), (iv) next-token loss on v1.
- **Supported** if ablating `S⁻_t` (or `S⁰_t`) raises (i) and (ii) above all 20 of its random sets
  at ≥ 2 of the 3 steps; the loss change is quoted beside, never a condition. **Refused** if
  neither arm does. Mixed (one readout only) is reported as such. A set that is not "few"
  (unit 4) is still ablated; the claim then reads "a cooperating set of N".
- The positive control: G1's planted head ablated on the synthetic stack must turn (i) up above
  all 20 random sets, or the reading is not made.

## Fences

- **No registration and no adjudication**; 410m is exploration. v2's 12 held-out prompts are not
  read (`core/holdout.py`).
- **The skip is not a component.** Ablations remove updates, never the residual, whose own
  anti-collapse share (2103.03404) is outside A2.
- **A1's closed number changes only by unit 2's rule**, recorded under `status-A.md`
  "Corrections received".
- **LR warmup** overlaps 512 and 1000–2000; unit 4 reports breaks there with that confound.
- `a1a.ln_from` (duplicating `u2_attn.ln_from`) is merged into one in unit 3, the next code to
  extend the pass (A1 Parked; changes no number).

## Worth challenging (for the freeze)

1. **M1 as unit 2's primary.** It is the softmax's likelihood but has an offset profile `fe_full`
   lacks in form; a β that drops because φ absorbs content is the alternative reading. The
   `partial_r2` floor beside guards that. Rejected: re-deriving `fe_full` with mass weights (a
   third estimator nobody has checked).
2. **Unit 2's "moves" bar** (p50 outside A1's p10–p90 in 4 of 7 prompts) is coarse; a smaller
   move that crosses one cell's t* verdict would not trigger. Alternative: re-run A1b's cell at
   143000 whatever the refit says (it is free); taken as the recommended variant if the user
   prefers it.
3. **Linearised components** instead of A1's nonlinear step: exact sums, but a block whose
   linearisation residual is large is read beside, and §3.8.2 found it large at most steps
   outside 2000–4000. Rejected: Shapley over components through the frame (1e's method), which
   is exact but 432! orderings, sampled.
4. **The `W_O^h` per-head split** (2604.23681's non-identifiability). Rejected: a learned
   re-basis, which would make "a head" not the model's head.
5. **Mean ablation from the long passages**, not resample or optimal (2409.09951). Optimal
   ablation is the better-founded choice but needs a fit per component; beside only if the budget
   allows.
6. **Three ablation steps, not 18**: the GPU cost is small either way; three keeps the null at
   20 sets per arm in one unit. Alternative: all 18 at 10 sets.
