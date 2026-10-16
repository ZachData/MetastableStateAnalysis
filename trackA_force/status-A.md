<!-- trackA_force/status-A.md -->
# Track A — STATUS

<!-- phase-card -->
## Card

- **Question:** Does trained `pythia-410m` move its tokens far enough along the self-attention field (the identity-weight dynamics at each layer's measured β, causal mask) to reach the collapse time t* from where its own cloud starts, at each of 18 training steps (`P-gamma2`'s "never integrates far enough" against "resists")?
- **Inputs:** `pythia-410m`, the 18 Stage 0 steps, the 7 scored v1 prompts (prompts hash `976b62d677bc`), A1a's 126 records (code `a08544e9b644`, `data/trackA/a1a_batch_2026-10-15`), GPU float32 — `trackA_force/status-A.md` "A1a's gate and batch, Blocked 48 (a)"
- **Results:**
  - Opened 2026-10-14 with a scan, a `sympy` check, the t* table and a frozen design (budget 3, extended to 5 on the record) — `trackA_force/design-A.md` "Exit condition and budget"
  - Where the cloud starts, the causal mask and how β is reduced over heads each move t* several-fold; t* is the causal simulator's time from the measured cloud — `trackA_force/design-A.md` "t* before any output (no model, no data)"
  - The instrument passes its calibration gate (planted effect, null, mixed, LN1 chain) after two failed gates; a sustained shared drift is not a null — `trackA_force/status-A.md` "A1a's gate and batch, Blocked 48 (a)"
  - `P-gamma2` has a number on 410m: the projected integration time rises over training from about 0 (negative in 29 cells: moving against the field) to 0.82 of t* at 143000, and is *below* t* at all 18 steps (125 of 126 cells, one *straddles*), also with the shared drift removed; a random init (step 0) reads *below* too; the net moves several times t* in total, but over 90 % of that motion is off the field — `trackA_force/status-A.md` "A1b, the reading (2026-10-15)"
  - H-WARP does not collapse the curves (2 of 18 steps; its planted check passes); 1e's one-well question is not answered — `trackA_force/status-A.md` "A1b, the reading (2026-10-15)"
- **Superseded / wrong:** none
- **Registry:** none, because 410m is an exploration rung and A1 gives `P-gamma2` a number, not an adjudication — `trackA_force/design-A.md` "Fences"
- **Depends on:** 1c@4a06306316, 1d@4168cdd237, 1e@02fc532205
- **Feeds:** none
- **Open threads:**
  - The verdict rests on the projected step: on the calibrated step every cell is above t*, so "too shallow" and "resists" are told apart only by the field's direction — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - t* assumes V = I, while from 1000 to 4000 most heads have no attractive part — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - Is 1e's one well the causal dynamics' answer at the measured β? The causal simulator's t* doubles inside the measured range, so a plateau there is not ruled out — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - At 1000–4000 attention's own projected step is positive while the block's is near 0 (the MLP cancels it) — `trackA_force/status-A.md` "A1b, the reading (2026-10-15)"
- **After Phase 10:**
  - The long passages beside, as the design's continuity check (n > d) *(forward pass, GPU hours: attentions not stored)*
  - A2's dissipation per component, which reads the off-field share A1 found *(forward pass, GPU minutes per step)*
- **Reviewed:** 2026-10-15 · body `9b11e792bc`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check, t* table | **done 2026-10-14** (#201); frozen 2026-10-14 (Blocked 45, with the OV attractive share row beside the reading) | `design-A.md`, `lit-A.md`, `tools/math_checks/a1_collapse_time.py`, `tstar_table.py` |
| 2. A1a, the β producer over 18 steps and the T_eff instrument, with its gate | **built 2026-10-14; the gate fails (G3, G4); the real batch has not run** (Blocked 46) | `a1a.py`, `a1a_gate.py`, `p1c_frames/integration_time.unit_steps`; "A1a's gate" below |
| 3. A1a's re-gate (Blocked 46 (b)) and the chain re-gate (Blocked 47 (d)) | **done**: the re-gate failed G4 at 3 frames (#203); 47 (d)'s gate: G0–G4c pass, G2s fails on the drift-removed read (38 of 42), its planted drift not a null (#204; Blocked 48) | "A1a's gate, Blocked 47 (d)", "A1a's re-gate" below |
| 4. The gate with G2s beside (Blocked 48 (a)) and A1a's batch | **done 2026-10-15: the gate passes; the batch ran, 126 of 126 records** | "A1a's gate and batch, Blocked 48 (a)" below |
| 5. A1b, the reading (H-WARP and 1e's question beside) | **done 2026-10-15: below t* at all 18 steps; A1 closed at its exit condition** | "A1b, the reading", "A1 on one page" below |

Budget: 5 units on the record since Blocked 48 (a) (`design-A.md` "Exit condition and budget";
first 3, then 4 by Blocked 46 (b)). All 5 spent; A1 closed 2026-10-15 at its exit condition.

## A1 on one page (closed 2026-10-15)

**Asked:** does trained 410m move its tokens far enough along the self-attention field to
collapse (`P-gamma2`)? **Exit met:** `P-gamma2` has a number (below), within the 5-unit budget.

**Established (tier 1, 410m, the 7 v1 prompts, 18 steps):**

1. **Below t\* at every step, rising over training.** `T_proj`, the integration time along the
   field, is below the shortest causal t* the β bracket allows in 125 of 126 cells (one
   *straddles*), *far below* (under half) in every cell to step 16000. *Below* holds three
   cases the frozen rule does not separate (`/challenge-pr` on #206, finding 3): `T_proj` < 0,
   the net moving **against** the field (29 cells, at 0–4 and 256–4000: the "resists" side);
   about 0, including a random init (step 0 reads *below* by the widest margin); and positive
   but short (8–128, 8000 on). So the reading's content is the trend, not the label. P-gamma2's number: the median `T_proj / t*` is
   ≤ 0.39 to 16000, then 0.56, 0.74, 0.82 at 32000, 54000, 143000 (at the bracket's top β:
   ≤ 0.33 at every step). The late rise is the net moving further along the field, not t*
   shrinking much (t* at p10 3.6 → 2.6). With the shared drift removed (`T_proj_c`) every cell
   is still *below* (largest `T_proj_c / t*(p10)` 0.81).
2. **The net moves far, but not along this field.** The calibrated step `T_cal` (22–30) is
   above t* at p90 in all 126 cells (by ≥ 1.37×); over 90 % of it is off the field. So the shortfall is one of
   direction, not depth: on 1c's step the same data would read *not below* everywhere.
3. **No collapse at the last block** at 17 of 18 steps (median mean cosine at state 23
   0.45–0.85 to 256, 0.26–0.49 from 512). At step 16, 3 of 7 prompts reach 0.90 with `T_proj` ≈ 1.1, so that collapse is not
   the field's integration.
4. **H-WARP refused:** cumulative `T_proj` does not line the per-block cosine curves up across
   steps better than depth does (2 of 18 steps pass against 12 needed).
5. **1e's one-well question is not answered** (`/challenge-pr` on #206, finding 2). The
   design's "one sigmoid at β ≤ 5.6" is (6.9)'s, which has one cluster by construction; under
   the causal mask (`design-A.md` "t* before any output") t* doubles between β 3.5 and 5.6,
   inside the measured range, and per prompt the layer β's p90 at 143000 reaches 7.14. (6.9)'s
   doubling β (6.9–10.4, `a1b_report.txt`) is kept as computed, not as an answer. Carried to A3.

**Not established:**

- **That P-gamma2's "never integrates far enough" holds.** Reading 1 rests on `h_proj`, which
  A1 introduced (`design-A.md` "Worth challenging" 3); reading 2 is the falsifier's own case
  ("T_eff ≥ t* and no collapse") on the registered step. Which step is the registry's `T_eff` is
  a registration question (A5); h0's wording against its statement stays the user's (Blocked 45).
- **The comparator's V = I.** t* is the identity-weight dynamics; at 1000–4000 the layer's OV
  attractive share is 0 for most heads (`a1a_report.txt`), the steps where the model is least
  like the comparator.
- **The layer's β as the head median**, the causal mask's other reductions, and anything at
  another rung or on held-out prompts. The long passages (the design's continuity check) were
  not run: A1a's batch is v1 only.
- **That `T_attn_proj` is drift-robust**, and what the last block's negative step means.

**Parked at close:** every item below is carried or dropped ("Parked").

## A1b, the reading (2026-10-15)

**Input:** A1a's 126 records (`data/trackA/a1a_batch_2026-10-15/records`, code `a08544e9b644`,
prompts `976b62d677bc`); the rule as frozen (`design-A.md` "A1b"), the choices left to the code
in "A1b, before output" (committed and pushed before the run, `e65dd1b`). **Producer:** `python
-m trackA_force.a1b --records data/trackA/a1a_batch_2026-10-15 --out data/trackA/a1b_2026-10-15`
(2 s, CPU; `a1b_report.txt` holds the tables, its one copy; `a1b.json` sha256 `53e5760b99a0`).
Steps 0 and 2 are different checkpoints that differ only in the third decimal (checked).

**The reading:** *below* at all 18 steps (≥ 6 of 7 prompts each). 126 of 126 cells read (no
t* infinite or refused); 125 *below*, 1 *straddles* (143000, `paper_excerpt`: `T_proj` 2.65
against t* 2.61 at p10, 7.51 at p90); none *not below*. *Far below* in 7 of 7 at every step to
16000, then 2, 1, 1. Median `T_proj / t*(p10)` [prompt range]: 0.10–0.33 at 8–64, ≤ 0.10 in
absolute value at 0–4 and 128–4000 (`T_proj` near 0 or negative), 0.16 at 8000, 0.39 at 16000,
0.56 [0.34, 0.61] at 32000, 0.74 [0.45, 0.79] at 54000, 0.82 [0.40, 1.01] at 143000. At the
registered comparator ((6.9) from g0 = 0) the same ratios are smaller late (0.53 at p10 at
143000), since t* from an orthogonal start without the mask (4.2–5.5) is longer than the
causal time from the measured start (2.6). Negative `T_proj` in 29 cells (0–4, 256–4000).
On `T_proj_c` (shared drift removed): 126 of 126 *below*, largest ratio 0.81.

**Beside (medians over prompts, `a1b_report.txt`):**

- **Off-field share** `1 − T_proj / T_cal`: 0.92–1.01 at every step (cells 0.91–1.03; above 1
  where `T_proj` is negative).
- **Attention's part:** `T_attn_proj` is 0.36–1.12 at 1000–8000 while the block's `T_proj` is
  −0.05–0.47, so the MLP's move cancels attention's along-field part there (1e's MLP finding,
  `status-1e.md` "U2's per-head arm"); late, attention carries about half of `T_proj`.
- **The drift row:** `T_proj_c / T_proj` 0.69–1.37 where `T_proj` is not near 0 (meaningless at
  0–4 and 1000–4000, where it is). In 19 cells with `T_proj` > 0.2 (steps 8–128, 8000–32000)
  `T_proj_c` is the larger: there the removed shared part ran **against** the field, so "a drift
  can only add" ("A1b, before output") is wrong for the real stream (`/challenge-pr` on #206,
  finding 1); the *below* reading holds on `T_proj_c` too.
- **Blocks 20–22 at 143000** carry −0.05 of `T_proj` (median; −0.21 to 0.03), so G4c's per-block
  over-read there (1.10–1.36 of h) moves the total by almost nothing.
- **Block 23** (outside T): its projected step is −0.68 at 143000 (median), against the field.
- **Collapse at state 23:** none at 17 steps; step 16, 3 of 7 (0.901–0.905).
- **H-WARP:** the LOSO error on the T axis is ≤ ½ of the depth axis's at 2 of 18 steps (2000,
  4000), > 1× at 10; the null (17 shifted profiles) reaches 2 at most, ≥ 2 in 1 of 17.
- **H-WARP's planted check:** `tests/test_trackA_a1b.py` plants mean cosine as one function of
  cumulative T over step-varying profiles; `hwarp` reads *collapses* there (added after
  `/challenge-pr` on #206, finding 5). The cyclic-shift null is weak where neighbouring steps
  look alike; it is beside, and the refusal does not rest on it.
- **1e's question:** (6.9)'s doubling β is at the report's end; it does not answer the question
  (item 5 above).

## A1b, before output (2026-10-15)

Written before `trackA_force/a1b.py` ran. The family and the rule are `design-A.md` "A1b"
(frozen); this section answers the two Parked items carried to A1b and records where the code
left a choice. **Disclosure:** while reading the record format, one cell's totals were seen
(143000, `camus_letranger`: `T_proj` 2.22, causal t* 2.81 at p10, 6.47 at p90). Nothing below
was changed after that.

- **t\* at a negative β** (Parked). With V = I the field is still a pull toward a weighted mean
  of the other tokens; β < 0 only weights the far ones more, so the cloud still collapses and the
  time barely moves: causal `sa_sim`, n = 467, d = 1024, Gaussian start, seed 0: t* 3.78 at β
  −2.5, 3.79 at −1, 3.82 at 0, 3.93 at 1; (6.9) from g0 = 0: 4.165 at −2.5 against 4.172 at 0.
  So a cell with p10 < 0 is read against (about) the uniform-attention time, as written; the
  rule needs no change. What β < 0 says about the real heads (they favour dissimilar tokens) is
  not a statement about the comparator.
- **How far a shared drift can pass for the field** (Parked). On the gate's sustained drift
  (G2s) `proj` read 0.12–0.87 of the planted T and `proj_c` 0.08–0.70, both positive: a drift
  held over many blocks moves the cloud toward its target, which is along the field. So a drift
  in the real stream can only **add** to `T_proj`: a *below* cell is not threatened by it; a
  *not below* or *straddles* cell may be partly drift. The row beside: `T_proj_c / T_proj` per
  step (how much survives one shared vector removed per block); whether `T_attn_proj` is
  drift-robust stays unchecked. **Wrong for the real stream** (`/challenge-pr` on #206, finding
  1): in 19 cells removing the shared part raises the along-field time; see "A1b, the reading".
- **Where the code chose** (`a1b.py` docstring is the spec): the ratio `T_proj / t*` is reported
  at both bracket ends; "far below" is against t* at p10 (the shorter). H-WARP: points are
  (cumulative `T_proj` before block l, mean cosine of block l's input), blocks 0–23, the 7
  prompts pooled; isotonic **increasing**; error is the mean absolute error on the left-out step;
  the depth axis is x = l; the null's 17 draws give every step the profile k steps along (k =
  1…17). 1e's question: the β at which (6.9)'s t* doubles its β = 0 value, at n 467 and 2,041,
  from g0 = 0 and from 143000's median measured g0.

## A1a's gate and batch, Blocked 48 (a) (2026-10-15)

**Input:** gate as below (synthetic; `ln_sha` `eea24ba8de2b`, `SEED` 20261014); batch: the 7 v1
prompts (prompts hash `976b62d677bc`) × the 18 steps, `RUNS` in `a1a.py` (`data/phase12`),
pythia-410m float32 on the GPU, TF32 off. **Producer:** at `cd0a0b6` (`a1a.code_sha`
`a08544e9b644`, gate `86d5a49e6a7b`): `python -m trackA_force.a1a_gate --out
data/trackA/a1a_gate_48a_2026-10-15` (**353 s; G4c 56 s on the GPU**, against numpy 1.3e-10),
then `python -m trackA_force.a1a run --gate <that>/a1a_gate.json --out
data/trackA/a1a_batch_2026-10-15 --workers 3` (**46.6 min**, 151–170 s per step), then
`python -m trackA_force.a1a report --out <same>` (`a1a_report.txt`, the table's one copy).
Env: `METS_REPO=<worktree> METS_DATA=<main>/data`, HF offline, GPU visible.

**The gate passes (log lift):** G0–G4c as in 47 (d)'s run, both reads (G4c 0.978–1.057 on 756
of 756); **G2s beside, no bar**: `proj_c` within 0.1 T in 4 of 42 cells (the same 4 as before).
Tangent fails G3 (beside, as always).

**The batch:** 126 of 126 records. First record (143000, `camus_letranger`) populated; over all
records every head's β is finite at all 18 steps (the report's "heads finite"); `fe_full` at 143000 matches 1d's refit within 1.4e-14 (all 7
prompts); the hooked pass against the stored unit rows ≤ 2.6e-7 to step 2000, rising to 7.9e-5
at 143000 (the GPU against the CPU, as 1e's probe found; **close to `check_pass`'s 1e-4**, so a
re-run on another GPU or driver may refuse at that step, finding 5); causal t* exists at both ends of the
bracket in all 126 cells. **The layer-β bracket and the OV attractive share by step: in
`a1a_report.txt` (its one copy).** In words: the bracket sits near 0 to step 128, widens at
256–4000, and is positive from 16000; **its p10 is negative at every step to 8000** (Parked); the
OV share falls to 0 around 1000–4000 and recovers.

**What this is not:** no T has been set against t*; the reading is A1b's (unit 5), on the
family fixed in `design-A.md` "A1b".

**Fixed on the way:** the batch's first launch crashed on its first model load (`a1a.ln_from`
read `model.gpt_neox`, but `core.models.load_model` returns the bare model); fixed to the idiom
`Hooked` uses (same weights, no number moves), tested on both shapes, and the gate re-run
because the code hash changed. The first gate record (`code` `a931…`, at `d7e8cfe`) and the
crashed batch's empty output are kept under `data/trackA/*_d7e8cfe`, `*_crashed`.

**How to re-run:** the three commands above; the batch resumes from its records and refuses
records written by other code.

## A1a's gate, Blocked 47 (d) (2026-10-15)

**Input:** synthetic only: the v1 design points as below, the LN1 of every block at the 18 steps
(`ln_sha` `eea24ba8de2b`), `SEED` 20261014; G4c's 24 planted h per cell from seed + 5.
**Producer:** `python -m trackA_force.a1a_gate --out <dir>` at `8b64953` (`a1a.code_sha`
`a931b83caa19`, gate `185b34a482e3`); record `data/trackA/a1a_gate_d_2026-10-15/a1a_gate.json`,
log beside. **366 s in all; G4c 57 s on the GPU** (torch float64; against numpy on one cell
1.3e-10); G0–G3 numpy on the CPU. Spec: `design-A.md` "Blocked 47 decided".
**Verdict (log lift): fails on G2s only.**

| arm | bar | `proj` (log) | `proj_c` (log) |
|---|---|---|---|
| G0, 21 cells | ≤ 3 %, dt converged | pass (21 of 21) | – |
| G1, 42 cells | h ≤ 5 %, T ≤ 3 % | pass (h ≤ 2.3 %, T ≤ 0.7 %) | pass (2.2 %, 1.0 %) |
| G2, planted null | \|T\| ≤ 0.1 T | pass (≤ 0.0013 T) | pass (≤ 0.0027 T) |
| G2s, shared drift | \|T_c\| ≤ 0.1 T (`proj_c` only) | 0.121–0.870 of T (beside) | **fails 38 of 42** (0.077–0.696) |
| G3, mixed | ≤ 5 % | pass (≤ 4.3 %) | pass (≤ 4.1 %) |
| G4, block 5 at 143000 | T within 10 % | pass (0.988–1.004) | pass (0.986–1.003) |
| **G4c, the chain, 756 totals** | T within 10 % | **pass, 0 outside** (0.978–1.057) | **pass, 0 outside** (0.981–1.057) |

Tangent and gnomonic beside: the same, except tangent fails G3 (as before). **G4c by step**
(`proj`, log): 0.994–1.001 at every step to 16000, 0.988–1.000 at 32000, 0.978–1.002 at
54000, **1.001–1.056 at 143000** (high, as the one-frame sweep was, but inside the bar).
Block 23 (beside): 0.983–1.507 of its h. **Per block the bias is still there** (`/challenge-pr`
on #204, finding 1, checked on the record): at 143000, blocks 20–22 read 1.10–1.36 of their
planted h in 96 of the 17,388 gated block readings; every other (step, block) reads 0.94–1.05.
The totals pass because each block carries ~1/23 of the planted T. So the one-frame sweep's
failures were real per-block bias at those frames, which the reading's sum dilutes; the total
stays inside 10 % only while those three blocks carry a small share of T. A1b has to report
that share (Blocked 48).

**G2s is not a null as planted (checked after the run, synthetic only, `homer_iliad`):** the
arm moves every token toward one fixed direction for 23 steps, so the cloud converges on it
(mean cosine 0.3 → 0.96 at β 3.5 from 0.3; 0 → 0.68 at β 0 from 0). On its first step the
drift reads `proj_c`/h ≈ 0.01; by its last, 0.80 (β 3.5) and 0.99 (β 0): once the cloud has
collapsed onto the drift's target, the field points there too, and a drift and an attraction
are the same move. **One step, checked after the review** (`/challenge-pr` on #204, finding 2;
n = 300, β 3.5, h 0.1, log lift): a drift in a random direction reads ≤ 0.017 of h on both
reads at start cosines 0, 0.3, 0.6, so a one-step re-plant of G2s would pass without testing
`proj_c`; a drift toward the cloud's mean reads 0.71 (`proj`) and −0.03 (`proj_c`) from cosine
0, but 0.99 / 0.96 from 0.3 and 1.00 / 0.99 from 0.6. `proj_c` differs from `proj` only on a
near-orthogonal cloud: at positive mean cosine the field points at the mean, so a drift toward
the mean is the field's own direction and no per-step projection can separate them. The 4 passing cells are β 7 from cosine 0, where the cloud barely
converges (mean cosine ≤ 0.003; still 0.15–0.38 of h by the last step). The bar was set
without asking what a sustained drift does to the cloud (`LESSONS.md` 6). What `proj_c` does
remove (one step from an unconverged cloud) is not what the arm measured.

**What did not run:** the real batch. No real T, t*, β or cosine has been read.

**How to re-run:** the same command (needs CUDA; `--device cpu` for a slow CPU run).

## A1a's gate (2026-10-14)

**Input:** synthetic only, at the v1 design points (n = 241, 285, 445, 464, 466, 481, 511;
d = 1024; β ∈ {0, 3.5, 7}; starts at pairwise cosine 0 and 0.3; 23 planted steps,
h log-uniform in [0.02, 0.5], `SEED` 20261014); G4's frame is step 143000's block-5 LN1.
**Producer:** `python -m trackA_force.a1a_gate --lift {tangent,log} --out <dir>` at #202's
first commit (`12633c1`; `a1a.code_sha` `1cd68a0b4ba1`, gate `258cb1cb6894`, both in each record);
records `data/trackA/a1a_gate_2026-10-14/a1a_gate_{tangent,log}.json` (~5 min per lift on CPU,
G0 on the GPU). A first run on uncommitted code gave identical numbers.

| arm | bar | tangent lift (as designed) | log lift |
|---|---|---|---|
| G0, `sa_sim` full attention vs (6.9), 21 cells | ≤ 3 %, dt converged | **pass** (≤ 1.2 %) | same (no lift) |
| G1, planted effect, 42 cells | h ≤ 5 %, T ≤ 3 % | **pass** (h ≤ 3.4 %, T ≤ 1.1 %) | **pass** (2.3 %, 0.7 %) |
| G2, planted null | \|T_proj\| ≤ 0.1 T | **pass** (≤ 0.0013 T) | **pass** |
| G3, mixed (ξ ⊥ X, \|ξ\| = \|hX\|) | h and √2 h ≤ 5 % | **fails 4 of 42** (≤ 6.4 %) | **pass** (≤ 4.3 %) |
| G4, frame (raw dynamics read in LN1) | T ≤ 10 % | **fails 8 of 42** | **fails 8 of 42** |

**G3's failure is a known second-order bias, not a code defect:** the tangent part of a step
`V` followed by retraction is `V/√(1+|V|²)`; G3's ξ doubles `|V|²`, and the 4 cells are the
ones whose cloud collapses furthest (end cosine 0.73–0.99, so the field and the steps are
largest). All three lifts are functions of the angle θ a token turns: tangent `sin θ`, log `θ`,
gnomonic `tan θ` (`U'/⟨U', U⟩ − U`). The gnomonic lift is exact for one Euler step followed by
renormalisation (0.0 % on G1–G3), which is also how a block acts if its move is tangent (`/challenge-pr`
on #202, finding 4: not circular, as first written here). The log lift measures arc length, the
continuous-time step that the simulated t* counts; its bias at the planted range is about 2/3 of
the tangent lift's (≤ 4.3 % against ≤ 6.4 %), not half. Per-step only: on T, the tangent lift's
worst is 2.3 %.

**G4 by start and β** (`T_proj/T`, range over the 7 prompts; both lifts within 0.002):

| β | from cosine 0 | from cosine 0.3 |
|---|---|---|
| 0 | 0.93–0.96 | 0.994–0.998 |
| 3.5 | 0.875–0.957 (1 fail, n = 511) | 0.992–0.997 |
| 7 | **0.742–0.838 (7 fails)** | 0.993–0.998 |

Always low. The likely mechanism (not tested): LN1's bias adds a common direction, so in the
LN frame a near-orthogonal raw cloud has a field with a large part along it that raw dynamics
never follow. G4's premise (dynamics in the raw frame) is one of two; a V = I head acts on LN1
rows, bias included, which this arm does not plant.

**What the gate does not test** (`/challenge-pr` on #202, findings 2 and 5): G4 uses one LN1
(block 5 at 143000; at step 0 the bias is 0, so the factor there is likely near 1, elsewhere it
could be below 0.742); G2's random tangent directions cannot align with the field in 1,023
dimensions, so G2 cannot fail, while a drift shared by all tokens (the confound that matters,
since at positive mean cosine the field points at the cloud's mean) is not planted.

**What did not run:** the real batch (β over 18 steps, the hooked pass, t*). No real T, t*, β
or cosine has been read, so Blocked 46 is decided before any real output.

## A1a's re-gate (2026-10-14, Blocked 46 (b))

**Input:** as "A1a's gate" below, plus the LN1 weights of every block at the 18 steps (`ln_sha`
`eea24ba8de2b`); the sweep's prompt `latex_monograph`. **Producer:**
`python -m trackA_force.a1a_gate --out <dir> --workers 8` at `8c1c0b6` (`a1a.code_sha`
`dd20504f0cf4`, gate `45aca91d844f`); record `data/trackA/a1a_regate_2026-10-14/a1a_gate.json`,
1h52m (CPU; see "How to re-run"). **Verdict (log lift): G0–G3 pass, G4 fails.**

| arm | bar | log (the verdict) | tangent | gnomonic |
|---|---|---|---|---|
| G1, 42 cells | h ≤ 5 %, T ≤ 3 % | pass (h ≤ 2.3 %) | pass (3.4 %) | pass (0.0 %) |
| G3, 42 cells | ≤ 5 % | **pass** (≤ 4.3 %) | fails 4 | pass (0.0 %) |
| G4, 42 cells, block 5 at 143000 | \|T_proj/T − 1\| ≤ 0.1 | **pass** (0.988–1.004) | pass | pass |
| G4 sweep, blocks 0–22 × 18 steps, 2,484 cells | same | **fails 7** (0.913–1.344) | fails 7 | fails 7 |
| G4raw (the first G4, beside) | – | 0.742–0.998 | same | same |
| G2s, shared drift (beside, no bar) | – | **0.121–0.870 of T** | 0.121–0.868 | 0.121–0.874 |

**Where G4 fails** (`T_proj/T`, log): only step 143000, blocks 20–22, always *high*, all 7 from
the orthogonal start: block 20 β 0/3.5/7 at g0 = 0: 1.139, 1.107, 1.332; block 21 β 7: 1.183;
block 22 β 0/3.5/7: 1.162, 1.114, 1.344. From g0 = 0.3 every gated cell passes (0.913–1.076).
Without step 143000 the sweep is 0.933–1.055. Block 23 (beside) reaches 1.517 at 143000 and
1.129 at 54000. The 3 worst (β 7, g0 = 0) are cells whose planted cloud barely moves (frame cosine 0.012 →
0.042); the 4 at β 0 and 3.5 collapse (to 0.88–0.91), so that is not the mechanism; none tested.

**What the sweep does not test** (`/challenge-pr` on #203, finding 1): it runs all 23 planted steps
through one frame and gates their total. The reading (`a1a.block_steps`) reads block l's one step
in block l's frame and sums over blocks. So "fails at blocks 20–22" is not "the reading fails
there"; the record keeps totals only, so it cannot say what one step per frame would give.

**G2s** says a drift shared by all tokens reads as 12–87 % of T in `T_proj`. Beside by design
(no bar), but it is the confound the reading has to answer. A1a's batch keeps per-block medians
only, so a drift-removed statistic has to be added to `a1a.py` (and planted in the gate) before
the batch, not computed in A1b (`/challenge-pr` on #203, finding 2).

**How to re-run:** the same command. Since this record, `a1a_gate.py` changed only in its pool
(one BLAS thread per spawned worker; 8 jobs 21 s against 124 s; values within 3e-11) and its sweep
lines (both ends, the count outside the bar, the failing cells at the end); `gate_code` is now
different, so `check_gate` needs a re-run before any batch (deferred to Blocked 47's re-gate,
which changes the gate anyway; `/challenge-pr` on #203, finding 4). `check_gate` does not compare
`ln_sha`, and the batch takes LN1 from the model, not the gate's cache (finding 5; carried to the
same re-gate). ~20 min expected, not yet measured
end to end.

## Before any output (2026-10-14)

No model, no data. **Input:** eq. (6.9) only. **Producer:** `python -m trackA_force.tstar_table`;
the table is in `design-A.md` "t* before any output" (one copy). What it fixed: the registered
t* ≈ 4.2 is the orthogonal-start time; from a start at mean cosine 0.3 it is ≈ 1.5 at every β,
and from g0 = 0 it is flat in β only to ≈ 5.6. With the causal mask (6.9) lacks
(`python -m trackA_force.sa_sim`, after `/challenge-pr` on #201), t* rises steeply with β (14.8
at 5.6, n = 467), so A1's comparator is the causal simulation from the measured cloud.
`sympy`: `tools/math_checks/a1_collapse_time.py`, 4 of 4 pass; tests `tests/test_trackA_tstar.py`,
`tests/test_trackA_a1a.py`.

## Parked

At A1's close (2026-10-15) each item is *done*, *carried to X* or *dropped (why)*.

- *The 2D spike* (the user's, 2026-10-14): **done**, its slot picked (Blocked 49 (a)): B2a
  after A1 (ROADMAP §7 B2a).
- *G0–G3 on the GPU* (2026-10-15): **dropped**: A1 is closed and no re-gate is planned; if an
  A1a-style gate is ever re-run, port it then (half a unit, wall time only).
- *t\* at a negative β* (2026-10-15): **done** before A1b's output ("A1b, before output"):
  still a collapse, at about the β = 0 time; the rule unchanged.
- *A shared drift and the field* (`/challenge-pr` on #205, finding 2): **done** for `T_proj`
  (the row beside; the reading holds on `T_proj_c`; the gate's "only adds" was wrong for the real
  stream, #206 finding 1); **carried to A2**: whether `T_attn_proj` is
  drift-robust (A2 reads the per-component moves).
- *`a1a.ln_from` duplicates `u2_attn.ln_from`* (`/challenge-pr` on #205, finding 4):
  **carried to A2**, which is the next code likely to extend `a1a.py`'s pass; changes no number.
- *The off-field share* (A1b, 2026-10-15): over 90 % of every block's move is off the
  identity-weight field. **Carried to A2**: which components carry it is A2's question.
- *Attention against the MLP at 1000–8000* (A1b): attention's projected step is positive where
  the block's is near 0. **Carried to A2** (per-component dissipation).
- *Block 23's negative step at 143000* (A1b): −0.68, outside T. **Carried to A2**.
- *Step 16's collapse at state 23* in 3 of 7 prompts with `T_proj` ≈ 1.1 (A1b). **Carried to
  A3** (the joint timeline: what moves at 16 is a timeline row, not an A1 reading).
- *1e's one-well question* (`/challenge-pr` on #206, finding 2): not answered by (6.9). **Carried
  to A3**: the causal simulator's trajectory from the measured start at each step's per-prompt β
  (GPU minutes) is a timeline row beside the energy break.
- *The long passages beside* (the design's continuity check, never run): **carried to A2** as
  an input choice, since A1 is closed. Cost: β needs `attentions.npz`, stored for v1 only, so
  a new extraction plus `a1a.py`'s pass at n up to 2,041 (GPU hours, not minutes).

## Corrections received

(none)
