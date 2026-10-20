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
  - A2 opened 2026-10-16 with a scan and a frozen design (5 units); the β refit leaves A1's number in place (2 of 7 prompts outside, bar 4; at M1's β 143000 reads *below* in 7 of 7) — `trackA_force/status-A.md` "A2 unit 2, the β refit at 143000 (2026-10-16)"
  - A2 closed 2026-10-17 at "a recorded reason it cannot be read": against the field is a few components, mostly early MLPs (the last block's at 143000), against-specific at 8 of 18 steps; the MLP cancels attention at 1000–4000 and carries the off-field motion; a fixed key appears from 1000 (a period, then a newline from 8000); the ablation's calibration fails on its planted control, so nothing causal is read — `trackA_force/status-A.md` "A2 on one page (closed 2026-10-17)"
  - C2's scan (2026-10-18) found no one reading E_β along a real training axis; A5's registration is drafted on the 9 PolyPythias seeds; its text decided 2026-10-19 (D1–D9 as recommended, a reproducibility check), nothing registered or downloaded — `trackA_force/status-A.md` "A5 and C2: scan and draft (2026-10-18)"
  - A3 opened 2026-10-19 with a scan and a design frozen as revised (Blocked 70 (ii) (a); 5 units; 10 located rows on the 18 steps, E1 dated by A5's D2 (a) / D4 (a); the locator's floor 4σ, calibrated on synthetic series) — `trackA_force/design-A3.md` "Exit condition and budget"
  - A3's locator fails its gate on the four existing floored rows (E2–E5): the null arm passes, the effect arm cannot place a 9σ move in the gaps up to 128–256, because each row's spread grows 10⁴–10⁵-fold from step 2 to 4000 and the reference is the window's largest Δ; E2–E5 are refused, E1 reads 512 and E10 1000 (their anchors) — `trackA_force/status-A.md` "A3 unit 2, the existing rows and the locator's gate (2026-10-19)"
  - A3's re-gate (Blocked 71, the user's: the effect arm at 9σ at the reference step, each row's floor and power re-derived from its own per-step spread, a censored mark) passes every arm; E2 reads 512 (its anchor), E4 1000, E5 8, E1 512, E10 1000, none censored or low power; E3 reads "no change" (2.0σ at 4000), likely because it scores the same-token key, not the successor that the 2026-10-18 correction names, so E4–E3 and E3–E2 are unread — `trackA_force/status-A.md` "A3 unit 2c, the re-gate (Blocked 71, 2026-10-19)"
  - A3 unit 3 (Blocked 72, the user's: 6 units, E3 re-defined at +1 on `L8H6` and `L11H14`, every floor max(4, own)): E3 reads 1000 (a peak, then a fall), E7 weight change 2000 (censored), E8 ESD α 1000 (a rise to 256, then a fall); E11 refused as a constant row: the simulator's final count is 1 in every cell, as Theorem 4.1 of 2411.04990 makes it for V = Id at any long horizon (beside: at t = 5–10 late checkpoints stay split from 8000 on, as A1a's t* already showed); E8's 1000 may be its fit's (512 from its peak) — `trackA_force/status-A.md` "A3 unit 3, E3 at +1 and the weights-only and simulator rows (Blocked 72, 2026-10-19)"
  - A3 unit 4 (Blocked 73, the user's, all (a): E11 stays refused, censored rows' pairs unread, E8 read only where its two dates agree): the n-gram rows reproduce Belrose's trough on 410m's own grid, the KL of a unigram predictor to the model least at 128 and of a bigram at 256, then rising to 143000; both inside the warmup — `trackA_force/status-A.md` "A3 unit 4, the n-gram rows (2026-10-20)"
- **Superseded / wrong:** none
- **Registry:** none, because 410m is an exploration rung and A1 gives `P-gamma2` a number, not an adjudication; `P-A5`'s text is decided (Blocked 68), not in the registry until the calibration gate and the user's registration — `trackA_force/design-A.md` "Fences"; `trackA_force/design-A5.md`
- **Depends on:** 1c@4a06306316, 1d@4168cdd237, 1e@02fc532205
- **Feeds:** none
- **Open threads:**
  - The verdict rests on the projected step: on the calibrated step every cell is above t*, so "too shallow" and "resists" are told apart only by the field's direction — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - t* assumes V = I, while from 1000 to 4000 most heads have no attractive part — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - Is 1e's one well the causal dynamics' answer at the measured β? The causal simulator's t* doubles inside the measured range, so a plateau there is not ruled out (A3's row E11 if frozen with it) — `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)"
  - Nothing causal is known about which components push apart: A2's ablation reading was not made, and a size-matched null cannot control depth for MLPs at the real set sizes (carried to C2) — `trackA_force/status-A.md` "A2 on one page (closed 2026-10-17)"
  - M1 and `fe_full` give opposite β signs mid-depth on the code and LaTeX prompts; which is right is not known (carried to B3) — `trackA_force/status-A.md` "A2 on one page (closed 2026-10-17)"
- **After Phase 10:**
  - The ablation re-calibrated (one synthetic model on 7 starts, headroom on (iii), a depth control for MLPs), then read at 2000, 8000, 143000 *(forward pass, GPU hours)*
- **Reviewed:** 2026-10-20 · body `7f36248dda`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check, t* table | **done 2026-10-14** (#201); frozen 2026-10-14 (Blocked 45, with the OV attractive share row beside the reading) | `design-A.md`, `lit-A.md`, `tools/math_checks/a1_collapse_time.py`, `tstar_table.py` |
| 2. A1a, the β producer over 18 steps and the T_eff instrument, with its gate | **built 2026-10-14; the gate fails (G3, G4); the real batch has not run** (Blocked 46) | `a1a.py`, `a1a_gate.py`, `p1c_frames/integration_time.unit_steps`; "A1a's gate" below |
| 3. A1a's re-gate (Blocked 46 (b)) and the chain re-gate (Blocked 47 (d)) | **done**: the re-gate failed G4 at 3 frames (#203); 47 (d)'s gate: G0–G4c pass, G2s fails on the drift-removed read (38 of 42), its planted drift not a null (#204; Blocked 48) | "A1a's gate, Blocked 47 (d)", "A1a's re-gate" below |
| 4. The gate with G2s beside (Blocked 48 (a)) and A1a's batch | **done 2026-10-15: the gate passes; the batch ran, 126 of 126 records** | "A1a's gate and batch, Blocked 48 (a)" below |
| 5. A1b, the reading (H-WARP and 1e's question beside) | **done 2026-10-15: below t* at all 18 steps; A1 closed at its exit condition** | "A1b, the reading", "A1 on one page" below |
| A2 1. design and scan | **frozen 2026-10-16 as revised** (Blocked 56 (a), the user's) | `design-A2.md`, `lit-A2.md` |
| A2 2. the β refit at 143000 | **done 2026-10-16: A1's number does not move** (2 of 7 prompts outside, bar 4); A1b's 143000 cell re-run: still *below*, its one *straddles* reads *below* (Blocked 57 (a), the user's: card at `fe_full`, (i) beside) | "A2 unit 2, the β refit at 143000" below; `a2_beta.py` |
| A2 3. the per-component instrument and its gate | **done 2026-10-16: the gate passes, G0–G4** (no real reading yet); every random init is already "few" (Blocked 58) | "A2 unit 3" below; `a2_comp.py`, `a2_gate.py` |
| A2 4. the batch and the ranking (18 steps) | **done 2026-10-16** (Blocked 58 (a1), (b1), (c1)): against is few and MLP at 11–13 of 18 steps (against-specific at 6); the MLP cancels attention at 1000–4000; the top-key selector is biased (unread); block 23's step against is its MLP; the null for unit 5 is near degenerate (Blocked 59) | "A2 unit 4" below; `a2_batch.py` |
| A2 5. the ablation, its nulls and the reading | **done 2026-10-17 (Blocked 59 (b), (ii)): the rule's calibration fails on the planted control, so the reading is not made** (step 0 passes as the null); against-specific at 8 of 18 on the matched band; the sink's exact share > 0.05 from 2000; a newline key from 8000 (Blocked 60) | "A2 unit 5" below; `a2_ablate.py` |
| A2 close-out | **done 2026-10-17** (Blocked 60 (a), the user's; writing only): A2 closed at "a recorded reason it cannot be read" | "A2 on one page" below |
| A5 0. C2's scan and A5's registration draft | **done 2026-10-18 (writing only; A5 not opened, nothing registered)**; text decided 2026-10-19 (Blocked 68, the user's: D1–D9 as recommended). Next for A5: the calibration gate on seed 0, after A3 (ROADMAP §9 row 6) | "A5 and C2: scan and draft" below; `lit-C2.md`, `design-A5.md`, `tools/math_checks/a5_seed_location.py` |
| A3 1. design and scan | **frozen 2026-10-19 as revised after `/challenge-pr` on #232 (findings 1–5, 7 taken)** (Blocked 70 (ii) (a), the user's). #231 finding 4: A3 before A5's gate; no seed download before A5 registers (confirmed, 70 (i)). Locator check: 3σ keeps 76 % of flat series, 3.93σ keeps 95 %; placing right in 90 % needs 6σ (gap 2–4) to 9σ (2000–4000) | `design-A3.md`, `lit-A3.md`, `tools/math_checks/a3_locator.py` |
| A3 2. the producer, the existing rows, the locator and its gate | **done 2026-10-19: the gate fails its effect arm on E2–E5** (null passes); E2–E5 refused, E1 512 and E10 1000 (anchors pass; E2's anchor unread, refused) (Blocked 71) | "A3 unit 2" below; `a3.py` |
| A3 2c. the re-gate (Blocked 71, the user's) | **done 2026-10-19: every arm passes; E2 512 (anchor passes), E4 1000, E5 8, E3 no change (2.0σ at 4000; scored at offset 0, not +1), E1 512, E10 1000; no located row censored or low power** (Blocked 72) | "A3 unit 2c" below; `a3.py` |
| A3 3. E3 at +1, the weights-only rows, the simulator row (Blocked 72, the user's) | **done 2026-10-19: every arm passes but E11's; E3 (+1, `L8H6`, `L11H14`) 1000, E7 2000 (censored), E8 1000; E11 refused (count 1 in all 126 cells at t = 200); E2, E4, E5 unchanged at floor 4** (Blocked 73) | "A3 unit 3" below; `a3.py`, `a3_weights.py`, `a3_sim.py` |
| A3 4. the n-gram rows (Blocked 73, the user's: all (a)) | **done 2026-10-20: both arms pass (planted dip); E6u (KL to a unigram) 128, E6b (to a Witten–Bell bigram) 256, then a rise to 143000; corpus the Pile's validation split** | "A3 unit 4" below; `a3_ngram.py`, `a3.py` |

Budget: 5 units on the record since Blocked 48 (a) (`design-A.md` "Exit condition and budget";
first 3, then 4 by Blocked 46 (b)). All 5 spent; A1 closed 2026-10-15 at its exit condition.
A2: 5 units plus a close-out (`design-A2.md` "Exit condition and budget"); all 5 spent; A2 closed
2026-10-17 at its exit's "a recorded reason it cannot be read" (Blocked 60 (a)).
A3: 6 units plus a close-out (5 frozen, 6 on the record since Blocked 72 (i) (a)); 5 spent (the
re-gate, Blocked 71, took one); left: the reading, one spare.

## A3 unit 4, the n-gram rows (2026-10-20)

Blocked 73, the user's, all (a) (2026-10-20): E11 stays refused, drawn as description; a censored
row's pairs are unread; unit 5 states both of E8's dates (1000, 512) and reads a pair with E8 only
where both agree (`design-A3.md` §4 "Revised 2026-10-20 (Blocked 73)"). Unit 4's constants were
placed and committed before any output (`e63ec4e`; §4 "Placed before any unit 4 output").
**Input:** the Pile's validation split (`val.jsonl.zst` from `mit-han-lab/pile-val-backup`,
sha256 `264c875d8bbd`, 471 MB, main tree `data/trackA/a3/corpus/`); scored set 64 blocks × 1024
(tokens `cfa4542e0a81`; 149 documents: Pile-CC 38, Github 21, PubMed Abstracts 18, OpenWebText2
17, StackExchange 17, Wikipedia 12, 9 subsets with 1–7), table set 2,000,000 tokens (`817f2fdf2db8`,
1,267 documents, 18 subsets); no document shares a 13-token run with a v1 prompt (0 dropped).
`a3_ngram.py` `6057e3857228`, GPU float32 (RTX 3080), 18 steps at 7–8 s each (steps 0 and 143000
first, as the quick slice; same code). `a3.py` `12ce8a01afc0` with unit 3's inputs (weights
`eefcc5a8dea1`, simulator `a1732e526ffc`, the three `data/analysis` files as unit 2c), CPU, 4 min;
`attn_rows.json` copied from `2026-10-19-A3u3b/` (a cache of the stored attentions). Records
`data/trackA/a3/2026-10-20-A3u4-ngram/` (per step, `corpus.json`, `predictor_losses.json`) and
`…/2026-10-20-A3u4/` (gate and rows) (main tree).

**Gate** (floor used 4.00; own bounds E6u 2.34, E6b 2.50; the unit 3 rows' arms are bit-identical
to unit 3's, since E6's draws come after theirs):

| row | units | null "no change" at 4 (≥ 0.95) | planted dip at 9σ_ref, min over gaps (≥ 0.90) | power, gaps 2–4 … 2000–4000 (× σ̄_ref) | arm |
|---|---|---|---|---|---|
| E6u | 64 blocks | 1.000 | 0.968 | 6 ×9, 9, 9 | pass |
| E6b | 64 blocks | 1.000 | 0.971 | 6 ×8, 9 ×3 | pass |

**Checks placed before output:** the loss at 143000 is 2.22 nats (1.5–4); E6u at step 0 is 3.300
against KL(q_u ‖ uniform) 3.105 (6.3 %, within 10 %); no document is in both sets. All pass.

**Readings** (neither low power nor censored): **E6u at 128** (the fall into the minimum 1.845 nats,
ref / σ 116) and **E6b at 256** (3.596 nats, 143). The other rows read as in unit 3 (E1 512, E2
512, E3 1000, E4 1000, E5 8, E7 2000 censored, E8 1000, E10 1000, E11 refused); their ref / σ moves
by the bootstrap's draw (E2 219.5, E3 10.5, E4 115.5, E5 20.2, E7 29.1, E8 7.8). Anchors pass. No
pair is read here (unit 5).

| step | 0 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E6u, KL(q_u ‖ p), nats | 3.30 | 3.30 | 3.14 | 2.69 | 2.16 | 2.18 | 2.16 | **1.46** | 1.55 | 2.59 | 4.12 | 5.65 | 6.58 | 7.23 | 7.47 | 7.90 | 8.11 | 8.60 |
| E6b, KL(q_b ‖ p), nats | 5.52 | 5.52 | 5.34 | 4.84 | 4.24 | 3.93 | 3.42 | 2.55 | **1.92** | 2.39 | 3.63 | 4.78 | 5.56 | 6.12 | 6.37 | 6.71 | 6.89 | 7.32 |
| model loss (beside), nats | 11.02 | 11.02 | 10.81 | 10.24 | 9.57 | 9.20 | 8.58 | 7.51 | 6.35 | 5.28 | 3.88 | 3.10 | 2.75 | 2.56 | 2.44 | 2.36 | 2.31 | 2.22 |

**Beside, not read** (computed after the readings were seen). The minima are firm: in 10,000
resamples of the 64 blocks the minimum is at 128 (E6u) and at 256 (E6b) every time; per block, 58
of 64 and 62 of 64; E6u(256) − E6u(128) = 0.093 ± 0.013 nats. The predictors' own loss on the
scored set: unigram 7.95, bigram 7.02 nats, over the 65,472 next tokens they give mass (728, 1.1 %,
are unseen in the table set). The model's loss passes the unigram's between 64 and 128 and the
bigram's between 128 and 256, and each KL is least at the step just after. After its minimum each
KL rises to 143000 with no second fall on this grid; Belrose's loss on sampled n-gram sequences
falls again after about 1000 (`lit-A3.md`), a different observable (§2 "Worth challenging" 5).

**What it says.** Belrose's trough (64–256, quoted) is reproduced on 410m's own grid with KL on
real text: the model is closest to the unigram predictor at 128 and to the bigram at 256, unigram
first, one grid step apart, both inside the warmup (E10 reads 1000).

**Caveats.** (1) The bigram is counted on 2M tokens and is a weak predictor (its loss 7.02 against
the unigram's 7.95). A table from more tokens is sharper, and its KL minimum could sit later
(Parked below). (2) The change-mass centroid (beside) is flagged on both rows (20, 34). It reads
monotone change mass, and most of each row's fall from step 0 lies at 8–64; it is not a
turning-point statistic. (3) E6u falls in two stages: 3.30 → 2.16 by 16, flat at 16–64, then 1.46
at 128. The rule dates the minimum, not the first stage. (4) Each block holds 1–5 documents and
the 64 blocks 149, so σ over blocks is across documents and their subsets; Pile-CC and Github are
38 and 21 of them.

**Re-run:** `METS_DATA=<main>/data HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1`, then
`python -m trackA_force.a3_ngram run --corpus <main>/data/trackA/a3/corpus/val.jsonl.zst --out <n>`
(`--steps 0 143000` first), then `python -m trackA_force.a3 run --out <dir> --weights <w> --sim <s>
--ngram <n>`.

## A3 unit 3, E3 at +1 and the weights-only and simulator rows (Blocked 72, 2026-10-19)

Blocked 72, the user's: (i) **(a)** 6 units on the record; (ii) **(a)** the frozen E3 refused as
defined and re-defined at +1 on `L8H6`, `L11H14`, gated before reading; (iii) **(a)** every floor
max(4, own). `design-A3.md` §4 "Revised 2026-10-19 (Blocked 72)", with unit 3's placed constants
(E7/E8's statistic and α fit; E11's start, β, dt, horizon t = 200, cos ≥ 0.99), committed before
any output (`d0db3b1`). **Input:** `a3.py` `c5165d4f5fb8` (after `/challenge-pr` on #235: a constant row refused by name; first run `389ef9ffd7a8`, records `2026-10-19-A3u3a/`, the same readings), `a3_weights.py` `eefcc5a8dea1`,
`a3_sim.py` + `sa_sim.py` `a1732e526ffc`; unit 2c's inputs and seed (prompts `976b62d677bc`,
`colocation_panel.csv` `a9d2604f4789`, `ov_per_head_series.json` `bfc59df7d28a`,
`behavioural_series.json` `0eb229010c2c`), the 126 `attentions.npz` and `activations.npz`, A1a's
126 records (`a1a_batch_2026-10-15`), the 32 cached checkpoints. Weights: CPU float64 on float32
checkpoints, 29 min; simulator: GPU float64, 95 min (dt halving on step 143000 `homer_iliad`
agrees at every time); `a3.py`: CPU, 5 min. Records `data/trackA/a3/2026-10-19-A3u3b/` (rows and
gate), `…-A3u3-weights/` (32 steps), `…-A3u3-sim/` (126 cells) (main tree). The first `a3.py` run
refused on its own step-0 check (E7's array transposed); fixed before any gate ran.

**Gate** (floor used 4.00 for every row; own bounds E2 2.51, E3 2.76, E4 3.39, E5 3.78, E7 2.10,
E8 2.40):

| row | units | null "no change" at 4 (≥ 0.95) | effect at 9σ_ref, min over gaps (≥ 0.90) | power, gaps 2–4 … 2000–4000 (× σ̄_ref) | arm |
|---|---|---|---|---|---|
| E2 | 384 heads | 1.000 | 0.999 | 4 ×8, 6, 6, 9 | pass |
| E3 | 14 (prompt, member), pooled | 0.990 | 0.965 | 6 ×10, 9 | pass |
| E4 | 7 prompts | 0.984 | 0.964 | 4 ×7, 6, 9, 9, 9 | pass |
| E5 | 7 prompts | 0.972 | 0.971 | 4, 4, 6 ×7, 9, 9 | pass |
| E7 | 24 layers × 4 types | 1.000 | 0.978 | 4 ×8, 6, 6, 9 | pass |
| E8 | 24 layers × 4 types | 1.000 | 0.992 | 4 ×5, 6 ×5, 9 | pass |
| E11 | 7 prompts | — | — | — | **refused: constant row** |

**Readings** (none low power):

| id | row | reading | step | ref step | ref / σ | censored | centroid (beside) |
|---|---|---|---|---|---|---|---|
| E1 | energy violations (D4 (a)) | located | 512 | — | — | — | 827 |
| E2 | OV median attractive fraction | located | 512 | 512 | 225 | no | 335 |
| E3 | `L8H6`, `L11H14` at +1, pooled | located | **1000** | 1000 | 11.4 | no | 697 |
| E4 | `L5H2` previous-token score | located | 1000 | 1000 | 120 | no | 561 |
| E5 | MinRank, mean over prompts | located | 8 | 16 | 20.9 | no | 1774 (≥ 2 steps) |
| E7 | weight change, median of medians | located | **2000** | 4000 | 29.0 | **yes** | 4901 |
| E8 | ESD α, median of medians | located | **1000** | 2000 | 8.3 | no | 654 |
| E11 | simulator's cluster count at t = 200 | **refused** | — | — | — | — | — |
| E10 | LR schedule | located | 1000 | 2000 | — | — | 539 |

E2, E4, E5 read as in unit 2c (the floor at 4 changes none; ref / σ moves by the bootstrap's draw).
Anchors pass (E1 512, E2 512, E10 1000). No pair is read here (unit 5).

**E3 at +1** rises at 1000 in both heads (`L8H6` 0.005 → 0.072, `L11H14` 0.007 → 0.057; pooled
0.006 at 512, 0.065 at 1000) and then *falls* (0.021 by 143000), so its reference is the peak, not
the window's end. The offset-0 series (beside, refused) never exceeds 0.010.

**E7** rises steadily on every type through the window (median rel. change 0.18 at 512, 0.49 at
1000, 1.29 at 2000, 2.38 at 4000, 3.63 at 8000; O and MLP out reach 11 and 10 at 32000–40000, then fall), so it
is censored: its 2000 dates the window's cut, not a change. **E8** is a rise then a fall: α
13.5 at init, up to 17.1 at 256 (all four types rise at 128–256), then 10.1 at 512, 5.5 at 1000,
4.2 at 2000, and 3.3–5.7 per type thereafter. The rule dates the fall at 1000 against step 0. The gate
planted only monotone moves, so, as with E5's dip, 1000 dates the fall under a rule tested on
steady moves.

**E11 is refused: the count at t = 200 is 1 in all 126 cells**, so the row is constant (refused
by name since `/challenge-pr` on #235; the first run refused it through a degenerate effect arm).
**The cause is the row's definition, not only its horizon** (#235 finding 1): with V = Id the
causal dynamics converge to a single cluster from almost every start (Theorem 4.1 of 2411.04990,
`docs/readings/2411.04990.md`), so a *final* count is 1 at any long enough time; t = 200 (30–80×
A1a's t* at 143000, 2.5–6.4) only made it certain. **Beside, not read (seen after the refusal):** the mean count at t = 5 is 1.6–2.0
from 0 to 4000, then 3.9 at 8000, 9.6 at 16000, 21.3 at 32000, 31.4 at 54000, 25.1 at 143000; at
t = 10 it is 1 to 8000, then 1.6–5.4 from 16000; at t = 20 at most 1.6. So under the causal
dynamics at the measured β, late checkpoints' clouds stay split longer, from 8000 on; every one
reaches one cluster by t = 50. That lead is not new: A1a's causal t* (bracket p90, median over
prompts) is 3.4–3.6 to 8000, then 4.0 at 16000 and 6.2–7.2 from 32000 (`a1a_batch_2026-10-15`).
Its move is late (from 8000, outside §4's ≤ 4000 window) and *up*, against E11's declared
direction. At the start the count is each prompt's number of distinct tokens
(checked on all 7 at 512: identical tokens embed identically), the same at every step. Whether to re-place E11's horizon now that this
has been seen is Blocked 73.

**Caveats.** (1) E3 is a re-definition made after the frozen E3 read "no change", for a reason
that predates the freeze (Blocked 72 (ii)). It is on 2 heads, not 5. `L7H8` (the offset-0 member)
is outside it. (2) E7/E8's σ is the spread across layers, not noise; their floors are as placed
in §4. (3) E8's α at init (≈ 12–17) is a fit to a Marchenko–Pastur bulk, where a power law is a
poor model. Its *values* there mean little; the fall to about 4 is what the row dates. **The
rise at 128–256 coincides with the fit choosing shorter tails** (median tail 43–73 eigenvalues
per type to 64, 25–53 at 256, 57–211 at 1000–4000; #235 finding 2), so it is probably the fit's.
Dated from the peak at 256 instead of step 0, the fall reads 512. Unit 5 should not order E8
after E1 or E2 on the one step between 512 and 1000 (Blocked 73 (iii)). (4) E7's
O and MLP-out changes are about 10× their init norm, so relative change mostly reads their small
init. (5) D1's `W_V` half is not built (design §4 "Placed before any unit 3 output").

**Re-run:** `METS_DATA=<main>/data HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1`, then
`python -m trackA_force.a3_weights run --out <w>`, `python -m trackA_force.a3_sim run --out <s>`
(`--quick` first: the dt check), `python -m trackA_force.a3 run --out <dir> --weights <w> --sim <s>`.

## A3 unit 2c, the re-gate (Blocked 71, 2026-10-19)

Blocked 71, the user's: (i) **(a)**, the effect arm at 9σ with σ at the reference step; (ii) a
**censored** mark for a located row whose reference is 4000; line 102 **re-derived per row** (floor
and power from each row's own per-step spread). `design-A3.md` §4 "Revised 2026-10-19". CPU,
140 s. **Input:** code `7952a72193eb` (after `/challenge-pr` on #234: the low-power mark in the
gap's own unit; the gate is bit-identical to the PR's first run `e2e5a1c411da`, records
`2026-10-19-unit3b/`), the same inputs and seed as unit 2b (prompts
`976b62d677bc`, `colocation_panel.csv` `a9d2604f4789`, `ov_per_head_series.json` `bfc59df7d28a`,
`behavioural_series.json` `0eb229010c2c`, attention cache copied from `2026-10-19-unit2b/`), 1000
bootstraps; 1000 null series to derive each floor, 1000 fresh to check it and to plant the effect.
Records `data/trackA/a3/2026-10-19-unit3c/` (main tree).

**How it is built.** A planted shape added to every unit shifts the statistic (mean, median,
pooled) by exactly the shape and leaves the bootstrap SE unchanged (tested), so each noise draw is
bootstrapped once and serves every gap and amplitude. σ̄_ref for a gap is the largest mean null σ
over the planted plateau's steps (where its reference can fall). Power per gap: the least of 2, 3,
4, 6, 9, 12, 18, 27, 40, 60, 90 (× σ̄_ref) placed right in ≥ 90 %. Unit 2's arm (9 × σ̄ at the
step past the centre, floor 4) is recomputed beside and reproduces unit 2b's failures (E2 .00 .01
.01 .02 .04 .07 .25 .90 .95 .99 1.00, against unit 2b's .00 .01 .02 .03 .05 .06 .27 .90 .95 .99 1.00).

**A defect found and fixed in this unit, after seeing a failure.** The first run
(`2026-10-19-unit3a/`) took each floor as the bare 95th percentile and checked it at ≥ 95 % on
fresh draws, a check a correct floor passes about half the time. E2 failed it (0.936; E3–E5
0.956–0.965) and was refused. The floor is now the 95 % upper confidence bound of that percentile
(an order statistic from Binomial(1000, 0.95)). It fixes the check's construction, not E2's data,
but it was made after E2 failed (*Worth challenging* 1 on the PR). No reading turns on it: E2's ref
is 229σ, above either floor and above the frozen 4σ.

| row | floor (frozen 4) | null "no change", fresh (≥ 0.95) | at 4σ | effect at 9σ_ref, min over gaps (≥ 0.90) | power, gaps 2–4 … 2000–4000 (× σ̄_ref) | arm |
|---|---|---|---|---|---|---|
| E2 | 2.51 | 0.961 | 1.000 | 1.000 | 3 ×9, 4, 6 | pass |
| E3 | 2.44 | 0.966 | 1.000 | 1.000 | 3 ×9, 4, 6 | pass |
| E4 | 3.11 | 0.960 | 0.982 | 0.965 | 3 ×7, 6, 9, 6, 9 | pass |
| E5 | 3.72 | 0.968 | 0.973 | 0.980 | 4 ×5, 6 ×5, 9 | pass |

(`2026-10-19-unit3c/a3_rows.json` → `gate`; unit 3a's floors 2.34, 2.37, 3.01, 3.40.) **Both
arms pass by construction here, up to sampling.** As stated before the run (Blocked 71 (a)), the
9σ_ref arm vouches only for moves large against the late noise; and the floor and its fresh check
come from the same noise model, so the null arm can fail only by sampling (#234 finding 2). The
gate now tests the rule's mechanics under that model, not the model; power is the informative
number. Every reading is the same at the frozen 4σ (`reading_at_frozen_floor`).

**Readings** (§4's rule at each row's floor; none censored, none low power: ref / σ̄_ref at the
located gap against its power, E2 27.8 against 3, E4 80.0 against 9, E5 5.0 against 4):

| id | row | reading | step | ref step | ref / σ | centroid (beside) |
|---|---|---|---|---|---|---|
| E1 | energy violations (D4 (a)) | located | 512 | — | — | 827 |
| E2 | OV median attractive fraction | located | **512** (anchor passes) | 512 | 229 | 335 |
| E3 | five members' pooled induction attention | **no change** | — | 4000 | 2.0 | 1418 |
| E4 | `L5H2` previous-token score | located | **1000** | 1000 | 112 | 561 |
| E5 | MinRank, mean over prompts | located | **8** | 16 | 20.8 | 1774 (≥ 2 steps) |
| E10 | LR schedule | located | 1000 | 2000 | — | 539 |

**E3 is "no change", and its definition is the likely reason** (`/challenge-pr` on #234, finding
1). E3 scores attention at the same-token key (offset 0: `induction_candidates`' repo condition,
as `behavioural_series.json`). The correction of 2026-10-18 (`p7d_redundancy/status-7d.md`
"Corrections received"; B3 unit 1), a day before A3 froze, says that at the successor (+1, the
standard pattern) `L8H6` and `L11H14` score 0.969 and 0.953 at 143000, and `L12H5` is not an
induction head. At offset 0, by 4000 `L7H8` rises (0.004 → 0.037) while `L8H6`, `L11H14`,
`L12H5`, `L8H9` end at 0.002–0.004, and `L8H6`, `L11H14` fall below their step-0 value by 143000.
That fits heads read one position from their target, not members disagreeing. σ over (prompt,
member) counts the spread as noise. E3 at +1 has **not** been computed. Its reference is at 4000,
so a located E3 would be censored. Pairs E4–E3 and E3–E2 are unread (Blocked 72 (ii)).

**Caveats.** Unit 2's caveats 2–4 stand (σ is the SE of Δ; step 0 noiseless in the null; E2's
point mass at 0). E5's centroid is ≥ 2 grid steps from its reading (beside, not read). **E5 is a
dip and a recovery** (MinRank 14.7 → 2.1 at 16, then up to 44 by 4000), a shape the gate never
planted (only steady rises): its 8 dates the fall, under a rule tested on monotone moves (#234
finding 5); unit 5 reads E1 against E5 with this said. No pair is
read here (unit 5).

**Re-run:** as unit 2 (`METS_DATA=<main tree>/data python -m trackA_force.a3 run --out <dir>`).

## A3 unit 2, the existing rows and the locator's gate (2026-10-19)

Design frozen as revised (Blocked 70 (ii) (a), the user's; 70 (i) confirmed: no seed download
before A5 registers). `trackA_force/a3.py` builds E1–E5 and E10 on the 18 steps, gates the §4
locator at each floored row's own design, then locates. CPU, 200 s (numpy bootstraps; no forward
pass). **Input:** code `dd7edc6bdcdb` (after `/challenge-pr` on #233: E2's check added, E3's gate
noise in per-pair units; first run `4e8df8b7defc`, records `2026-10-19-unit2/`, same readings),
prompts hash `976b62d677bc` (7 v1), `colocation_panel.csv`
`a9d2604f4789`, `ov_per_head_series.json` `bfc59df7d28a`, `behavioural_series.json`
`0eb229010c2c`, the 126 stored `attentions.npz` (`a1a.RUNS`); seed 20261019, 1000 bootstraps,
1000 gate series per arm. Records `data/trackA/a3/2026-10-19-unit2b/` (main tree).

**Reproduced first (or the run refuses), all pass:** E1's per-prompt nViolRel sum to the panel's
at every step (plateau median 57, as `design-A5.md` D4 (a)); E5's mean MinRank equals the
panel's; E3's pool equals `behavioural_series.json` for each of the five members; E2's median
0.504 → 0.094 by 512 (§3.12 A).

**The gate.** Noise per row at its real per-step spread τ_s (across-unit SD of Δ_u(s)); planted
move a logistic (0.25 octave) at 9 × the mean null σ at the step past the gap's centre.

| row | units | null "no change" (≥ 0.95) | placed right (≥ 0.90), gaps 2–4 … 2000–4000 | arm |
|---|---|---|---|---|
| E2 | 384 heads, median | 1.000 | .00 .01 .02 .03 .05 .06 .27 .90 .95 .99 1.00 | **fail** |
| E3 | 35 (prompt, member), pooled | 0.998 | .00 .03 .04 .04 .06 .05 .28 .29 .24 .94 1.00 | **fail** |
| E4 | 7 prompts, mean | 0.980 | .00 .01 .03 .03 .04 .03 .08 1.00 .87 .90 .96 | **fail** |
| E5 | 7 prompts, mean | 0.965 | .00 .27 .38 .36 .57 .60 .89 .95 1.00 1.00 .98 | **fail** |

**Why (checked, not tuned):** τ_s/√n grows 10⁴–10⁵-fold across the window (E2 8e-8 at step 2
to 1.4e-2 at 4000; E4 3e-7 to 1.3e-2). A move sized to its own early step is smaller than the
late steps' noise, so the window's largest Δ, the reference, falls late and the half-point with
it. `a3_locator.py`'s calibration had iid noise of one size at every step, which the real rows
do not have. Late τ_s is mostly the units' real heterogeneity in the move, not noise.

**Readings.** E1 **512** (D4 (a); anchor passes). E10 **1000** (anchor passes). E2, E3, E4, E5
**refused** (their arms failed); E2's anchor (512) is therefore not tested. The change-mass
centroids are in `a3_rows.json`, beside and not read. No pair is read (unit 5).

**Caveats.** (1) "9σ at the step past the centre" was this unit's reading of §4's "the logistic
above at 9σ". §4 defines σ once, as the SE of Δ *at the reference step* (#233 finding 3), so
that reading is the frozen text's literal one; under it the arm passes nearly by construction and
restates the low-power mark. It is the user's (Blocked 71 (a)), not applied here. (2) σ is the bootstrap SE of Δ, √2
the per-step noise `a3_locator.py` states its floor in, so the floor as built is the stricter
reading; the null arms (0.965–1.000) show it. (3) Step 0 carries no noise in the gate's model,
so its Δs are independent across steps (a conservative null). (4) E2 is not Gaussian: more
than half the heads sit at exactly 0 at 1000 and 2000 (62 %, 65 %; 43 % at 4000), so its median's bootstrap SE there is set
by the point mass, and the Gaussian null does not model it. (5) A row whose window maximum is
at 4000 has its half-point set by the window's cut (#233 finding 4; Blocked 71 (ii)).

**Re-run:** `METS_DATA=<main tree>/data python -m trackA_force.a3 run --out
data/trackA/a3/<dir>` (the attention pass is cached in `attn_rows.json`); `report --out` reprints.

## A5 and C2: scan and draft (2026-10-18)

Writing only (ROADMAP §9 row 6's first item; §10.8 is proposed, so this is the free part).
Nothing ran on a model; the only numbers are the `sympy` check's, on seed 0's panel counts.

| item | result | where |
|---|---|---|
| C2's novelty | not refuted: no paper found reads E_β, its monotonicity or token clustering along a real training axis; the nearest read rank, ID or anisotropy (2509.23024, 2602.15997, 2405.15471, Bernas ICML 2026), or have the energy without checkpoints (2605.07772) | `lit-C2.md` §1 |
| C2's sequence | the break (128–512) sits where the LLC stages put n-grams, before previous-token (512–1000) and induction heads | `lit-C2.md` §2 |
| A5's limit | all seeds share the LR warmup (to about 1430), so A5 can show the step is not seed 0's, not that it is not the schedule's | `lit-C2.md` §3 |
| A5's draft | `P-A5`: half-rise step of the violation count in {256, 512} per seed, K of 9, K ~ Bin(9, 1/6) under H0; E ≥ 20 at K ≥ 7. Nine choices, recommendations first; revised after #227's challenge (the break against the 1000–4000 plateau, not 143000; 410m seeds 3 and 4 already known outliers, counted and flagged; E labelled against a uniform-location null; the calibration gate named) | `design-A5.md` |
| seed 0 by statistic | half-rise 512, first non-zero 256 (both in the window); `CLAIM-B`'s change-mass centroid about 830 (outside), so the centroid is not offered | `tools/math_checks/a5_seed_location.py` |

**Input:** `data/analysis/colocation_panel.csv` column `p1_energy_nviol_b1` (β = 1, sphere, the
7 v1 prompts), seed 0 only, copied into the check. Re-run: `python
tools/math_checks/a5_seed_location.py`.

## A2 on one page (closed 2026-10-17)

**Asked** (`design-A2.md`): which heads and MLPs move 410m's tokens against the self-attention
field, or off it, from which step; and does ablating the top ones re-cluster the tokens and make
the energy over depth monotone? **Exit met on its second branch**, "a recorded reason it cannot
be read" (Blocked 60 (a), the user's), with the 5 units spent (#212–#216). Writing only; no run.

**Established (tier 1, 410m, v1's 7 prompts, 18 steps; descriptive except where it says so):**

1. **A1's closed number stands under the β refit** (unit 2). M1's refit p50 is outside A1's
   bracket in 2 of 7 prompts (bar 4); A1b's 143000 cell at M1's βs reads *below* in 7 of 7 (was
   6 and one *straddles*), and every change strengthens *below*. The two estimators disagree in
   sign on the code and LaTeX prompts (M1 negative mid-depth), which is where B2a's "sign bias"
   came from (`trackB_circuit/status-B.md` "Corrections received"). M1's L17–23 band reads 0.88
   pooled, 1.66 on the five prose prompts; the convention is kept (Blocked 57 (a)).
2. **An additive per-component instrument passes its gate** (unit 3, re-gated in unit 5): 18
   components per block, each update path-integrated through the LN1 frame (64 nodes; G0 ≤ 4.9e-14
   on the synthetic stacks, ≤ 4.7e-11 on real blocks), G1–G4 pass.
3. **Against the field is a few components, mostly MLPs, and training concentrates both signs**
   (units 4, 5). `|S⁻|` is below the init band's 5 % at 11–13 of 18 steps (step 0's band) or 11
   (each step against the band whose β matches it); against-specific (`S⁻` few, `S⁺` not) at 6
   steps on step 0's band, **8 on the matched band** (16, 32, 1000–8000, 16000, 143000). The
   most-against component is an MLP at 14 of the 15 trained steps (8 on; 54000 is a block-0
   head), and also at the untrained 0–4, where the top MLPs sit at about −0.01 each: early MLPs
   (L0–L7) from 16 to 32000, the last block's MLP at 143000 (−0.49 of block 23's −0.56; the same
   at M1's βs). So "an MLP tops the list" alone does not tell training from init; the init band
   on `|S⁻|` does (`/challenge-pr` on #217, finding 3).
4. **The MLP cancels attention at 1000–4000** (attention's summed step +0.19 to +1.05, the MLP's
   −0.23 to −1.07, the blocks −0.06 to +0.03); in part at 8000; both along from 16000.
   Attention's along step is drift-robust from 2000.
5. **The off-field motion is the MLP's**: 0.68–0.92 of the pooled off motion, block 0's MLP first
   at every step. At init five early MLPs already carry half of it (a band with no width), so
   "few" is not read for `S⁰`.
6. **The sink and fixed keys** (unit 5, beside). The sink's part along the field is small (summed
   step ≤ 0.14 in size). Its per-token share of the update, `mean_i |Σ_h sink_h,i| / mean_i |v_i|`,
   is ≤ 0.01 to 512, about 0.05 at 2000–4000 and 0.08–0.11 from 8000 (over 0.05 in 17 of 24
   blocks). **That is not 1e's quantity** (the sink's share of the block's *shared* update, the
   mean over tokens, `p1e_energy_field/u2_attn.shares`), so **A2 did not test 1e's number**, and
   unit 5's "1e's ≤ 0.05 … fails" overstated it (`LESSONS.md` 11). 1e's own arms differ at their
   edges: ≤ 0.05 in the attention arm, −0.13 to +0.07 (−0.19 on v1) in the per-head arm
   (`p1e_energy_field/status-1e.md` "U2's per-head arm"; `/challenge-pr` on #217, finding 4). A fixed key
   (top key ≥ 0.05·n over uniform) appears at 1000 (0.25 of head × block): a period at 2000–4000
   (0.57–0.58), a newline from 8000 (0.25–0.32), so 1e's second sink is one case of a fixed key.
7. **The ablation reading is not made: the rule's calibration fails on its planted control**
   (unit 5). The planted heads read "not supported" because (after `/challenge-pr` on #216) the
   synthetic ranks on one stack and reads on 7 other models, (iii) sits at its ceiling in the G1
   stack, and the size-matched MLP null does not control depth (step 0's L0–L4 MLPs read 0.964
   against nulls of 0.61–0.67; depth correlates −0.91 with (i)), while `S⁻` at 2000 and 8000 is
   early MLPs. Step 0 passes as the null on (i) (16 of 20 nulls ≥ `S⁻`) and reads "mixed" on (iii)
   by one cell of 168.

**Not established:**

- **Anything causal about `S⁻`, `S⁺` or `S⁰`.** "These N components carry the anti-collapse
  force" is neither supported nor refused; readings 3–5 are rankings of a linear split.
- Whether a repaired control would pass, and **a depth control for MLPs** at the real set sizes,
  for which there is no good answer (band matching is the degenerate null of Blocked 59).
- The real steps' sets on the long passages (only step 0 was ranked there); "few" for `S⁰`;
  whether 8, 1000 and 32000 are few on step 0's band (on the cut).
- Which β estimator is right on the code and LaTeX prompts.
- 1e's ≤ 0.05 at A2's instrument (a different quantity, reading 6), and why heads that give the
  sink up to 0.6 of their attention pass it little value.
- An energy split per component (`PROJECT.md` §3.8.3's exact dissipation): A2 reads each
  component's projected step and off share, not its share of `dE_β`.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 1. design and scan | #212 | frozen as revised (Blocked 56 (a)) |
| 2. the β refit at 143000 | #213 | A1's number does not move; the estimators split on code and LaTeX |
| 3. the per-component instrument | #214 | gate passes (G0–G4); 16 nodes fail G0 on real block 0 (→ 64) |
| 4. the batch and the ranking | #215 | readings 3–5; the top-key selector biased late (fixed in 5) |
| 5. the ablation and its calibration | #216 | the calibration fails; reading 6; reading not made |
| close-out | this | writing only (Blocked 60 (a)) |

**Parked at close:** every item is carried or dropped ("Parked", "At A2's close").

## A2 unit 5, the ablation's calibration (2026-10-17): it fails, the reading is not made

Blocked 59, the user's: (b) the null matches MLPs on type and size decile only (any band; heads
and `b_O` keep the band), each set's bands reported; (ii) both instrument fixes (#215 findings 1, 3)
and the init band at 143000's βs; 58 (a1) answered as both bands, each step read against the one
whose β matches it (step 0's to 4000, β ≈ 0; 143000's from 16000, β 2.5–3.7; both at 8000, β 1.3).

**Built:** `a2_comp.py`: `top_key` (the key with the most mass *minus* its uniform expectation;
a planted fixed sink at j = 37 and a local head in the tests, where the old ratio picks the last
10), the sink's exact share `mean_i |Σ_h sink_h,i| / mean_i |v_i|`, the top key's excess, the jvp
and projections chunked (2,041-token passages), `output_attentions=False`. `a2_batch.py`: `band
--beta-step`, the 143000-band and sink/fixed-key report sections. `trackA_force/a2_ablate.py`
(new): the long-passage re-rank, the sets, the null (Blocked 59 (b)), mean-ablation hooks on the
attention dense and MLP outputs at positions 1…n−1, the readouts, the rule, `calib` / `real` /
`report`; `scripts/a2_unit5.sh`.

**Inputs:** instrument code `df0a15c12751`, `a2_ablate` `6ca248cb2965`; the 8 long passages
`ba605f4e14b5` (ranking and means), v1's 7 prompts `976b62d677bc` (readouts), A1a's layer βs.
Re-gate `data/trackA/a2_gate5_2026-10-17/a2_gate.json`: **passes** G0–G4 (G0 max 4.9e-14;
5,885 s). Calibration `data/trackA/a2_unit5_2026-10-17/` (`calib.json`, `ablate_step0.json`,
`sets_step0.json`, `a2_unit5_report.txt`). Bands and the batch re-run at the fixed instrument
`data/trackA/a2_batch5_2026-10-17/` (126 of 126 records, `a2_batch_report.txt`).

**Placed in this unit, before any output** (`a2_ablate.py` docstring): the long-passage β is the
median over v1's prompts of A1a's layer βs; (i) is the mean over prompts of the last block's
mean cosine; (iii) is pooled over 7 × 24 blocks; the arm must beat the unablated model, the `S⁺`
ablation and all 20 nulls; null draws exclude the arm's own members; a member with an empty pool
widens to any band (counted); if 20 distinct sets cannot be drawn, heads drop the band too
(`heads_any_band`); mean-ablating `b_O` is the identity; a fixed key is an excess ≥ 0.05·n.

**The rule's calibration** (design "The rule's own calibration"; both must hold):

| run | (i) unablated | `S⁻` | `S⁺` | null max | nulls ≥ `S⁻` on (i), (iii) | (iii) unablated / `S⁻` | (i) | (iii) | result |
|---|---|---|---|---|---|---|---|---|---|
| synthetic G1 stack, 7 planted heads as `S⁻` (7 of 7 found) | 0.155 | 0.160 | **0.168** | 0.157 | 0, 20 | 1.000 / 1.000 | fail | fail | not supported: **fails** (must read supported) |
| step 0, the planted null (`\|S⁻\|` 29) | 0.557 | 0.827 | 0.680 | 0.921 | 16, 0 | 0.994 / 1.000 | fail | pass | not supported: passes |

**The calibration fails, so the reading at 2000, 8000 and 143000 is not made** (the design: "If
either fails, the reading is not made and the failure goes to the user"; `real` refused, as
written). STATE Blocked 60.

Why the planted control fails (my reading of the records, not tested):
1. **(iii) sits at its ceiling.** In the G1 stack the energy already rises in every block
   unablated (1.000), so no ablation can raise the share; (iii) could not pass by construction.
   The headroom was checkable before the freeze and was not checked (`LESSONS.md` 6).
2. **On (i) the planted heads beat all 20 of their nulls, but not `S⁺`.** `S⁺` is 7 heads of
   block 13, larger than the planted ones. The synthetic means come from a separate start (my
   choice, so the arm is not ablated by its own mean), in a random basis: replacing a part by
   another cloud's mean adds a shared vector, which raises the cosine roughly with the part's
   size. The same size effect shows at step 0, where the null absorbs it.

**After `/challenge-pr` on #216 (accept with changes; all four taken, 1 checked in the code):**
1. **The synthetic swaps the model, not only the text** (the larger cause): `synthetic()` ranks
   and takes means on one stack, then reads on 7 stacks, each with its own `planted_h(seed)` and
   its own generator for the parts. The real model keeps its weights and changes only the text.
   The reviewer: L13's step is 0.491 in the ranking stack and 0.032–0.272 in the readout stacks, so
   the mean swapped in can be up to ≈ 15× the part it replaces. Reason 2 above is a part of this.
2. **The MLP null does not control depth:** at step 0 `S⁰` (L0–L4 MLPs) reads 0.964 against
   nulls of 0.61–0.67, and over the null sets mean depth correlates −0.91 with (i). At 2000 and
   8000 `S⁻` is early MLPs, so depth alone could read "supported" at 2 of 3 steps. Matching MLPs on
   band (Blocked 59 (a)) is the degenerate null unit 4 found.
3. **Step 0 passes (iii) by one cell of 168** (0.994 → 1.000; all 20 nulls tie below), so it
   reads "mixed", which the calibration (only "not supported" required) lets through.
4. **Repair the stack, not (iii):** the ceiling is the G1 stack's (1e found real energy
   non-monotone from 256); changing the readout after calibration output is a rule change.

**Step 0** (the null; S⁻ is 15 MLPs and 14 heads, none widened, `heads_any_band` false):
mean-ablating `S⁻` lifts the last block's cosine 0.56 → 0.83, but 16 of 20 size-matched random
sets match or beat that (max 0.92), so the lift is size. `S⁺` gives 0.68. `S⁰` (L0–L4 MLPs)
gives 0.96, above all 20 of its nulls on (i), not on (iii) (0.77: energy falls in a quarter of
the blocks). Loss 11.0 in every arm (an untrained model); T/t* ≈ 0.

**The batch at the fixed instrument** reproduces unit 4's tables exactly (the report's first 25
lines differ only in the code hash); only the top-key lines change.

**The band at 143000's βs** (5–95 %): `|S⁻|` 10–25.5, `|S⁺|` 11–25.1 (step 0's: 19.5–33.6,
17.8–35.0). Against the matched band (58 (a1)):

| | step 0's band (unit 4) | the matched band |
|---|---|---|
| `\|S⁻\|` few | 11–13 of 18 | 11 (16–128, 512–16000, 143000; 32000 is not few at 143000's) |
| against-specific (`S⁻` few, `S⁺` not) | 6 (16, 32, 1000–8000) | **8** (16, 32, 1000–8000, 16000, 143000; 8000 at both bands) |

**The sink's exact share** (median over blocks / max; blocks ≤ 0.05 of 24): ≤ 0.009 / 0.013 and
24 of 24 to 512; 0.021 / 0.088, 22 at 1000; 0.052–0.056, 9–11 at 2000–4000; 0.076–0.105 /
0.20–0.24, 7 from 8000. ~~1e's ≤ 0.05 holds to 1000 and fails in 17 of 24 blocks from 8000~~
*(corrected at A2's close: 1e's ≤ 0.05 is the sink's share of the block's shared update, another
quantity, so this does not test it; "A2 on one page" reading 6, `LESSONS.md` 11)*. The per-token
share exceeds 0.05 in 17 of 24 blocks from 8000; it is about 2.5× under unit 4's bound.

**Fixed keys** (share of head × block whose top key gets ≥ 0.05·n mass over uniform): 0 to 128;
0.01–0.10 at 256–512; 0.25 at 1000; 0.57–0.58 at 2000–4000 (the token most often `.`, 0.52–0.57);
0.25–0.32 from 8000 (most often `\n`, 0.72–0.83). None in the last 10 positions at any step, so
the selector bias is gone. **1e's second sink on a newline holds from 8000** (descriptive);
at 2000–4000 the fixed key is a period. The top key's part in the field is small (≤ 0.06 per
block step, −0.02 at 143000).

**Not established:** anything causal about `S⁻` (the reading is not made); whether a repaired
planted control would pass; whether the rule could pass at all with (iii) as frozen; the real
steps' sets on the long passages (only step 0 was ranked).

Re-run: `METS_DATA=<data> systemd-inhibit --what=sleep:idle ./scripts/a2_unit5.sh` (≈ 5.3 h:
re-gate 98 min, calibration 13 min, two bands about 90 min each, the batch, the report). A dry run
of the synthetic check before it was hit by a suspend (40,283 s reported); every GPU call now
runs inhibited.

## A2 unit 4, the batch and the ranking (2026-10-16)

Blocked 58, the user's: (a1) a set is **few** if its size is below the init band's 5 %, the
band re-measured at unit 4's β (≤ 43 beside); (b1) the field at A1a's `fe_full` layer βs per
(step, prompt), M1's beside at 143000 only; (c1) 64 nodes and a refusal on G0 > 1e-6.
Descriptive: this unit ranks; it tests nothing (`design-A2.md` "Unit 4").

**Built:** `a2_comp.py` at 64 nodes, refusing a block whose G0 > 1e-6; the top key chosen by mass
received over its uniform-attention expectation, its token id recorded (#214 finding 5); `off_global`
(a block's off shares × its mean `|o|²` over all blocks', so `S⁰` ranks across blocks; the design
did not say how shares from 24 blocks combine) and `set_stats`. `a2_gate.py`'s G2 reads at given βs
and records every set statistic's band. `trackA_force/a2_batch.py` (`band`, `run`, `report`),
`scripts/a2_unit4.sh`.

**Inputs:** code `05a989b874ec` (instrument), v1 prompts `976b62d677bc`, A1a's layer βs
(`data/trackA/a1a_batch_2026-10-15/records`), M1's from `data/trackA/a2_beta_2026-10-16`. Gate at 64
nodes `data/trackA/a2_gate64_2026-10-16/a2_gate.json` (**passes** G0–G4, G0 max 4.9e-14; 5,400 s);
band `data/trackA/a2_batch_2026-10-16/a2_band.json` (5,261 s); 126 of 126 records, report
`a2_batch_report.txt`, `a2_batch_summary.json` in the same directory (about 103 s per step). Real
blocks: G0 ≤ 4.7e-11 (was 1.4e-3 at 16 nodes on block 0); stored-row match and reconstruction
within their tolerances in every record.

**The init band** (50 redraws + 1, 5–95 %), at step 0's βs (≈ 0; the cut) and at 3.5 (the gate, beside):

| statistic | at step 0's β | at 3.5 | real step 0 |
|---|---|---|---|
| `\|S⁻\|` | 19.5–33.6 | 13.5–29.7 | 28 |
| `\|S⁺\|` | 17.8–35.0 | 14.0–31.1 | 33 |
| `\|S⁰\|` | 5–5 | 5–5 | 5 |
| top-43 share of against | 0.574–0.666 | 0.602–0.719 | 0.621 |

β moves the `|S⁻|` cut by 6. `S⁰`'s band has no width: at init five early MLPs carry half the off
motion, so "few" cannot be read for `S⁰`.

**The ranking** (median over 7 prompts per component; proj in field units per block step; the
attention / MLP columns summed over blocks 0–22; **bold** = below the band's 5 % at step 0's β):

| step | `\|S⁻\|` | `\|S⁺\|` | `\|S⁰\|` | top of `S⁻` | attention / MLP proj | attention proj, drift removed | sink bound (median over blocks) |
|---|---|---|---|---|---|---|---|
| 0 | 28 | 33 | 5 | MLPs, each ≈ −0.01 | +0.06 / −0.08 | +0.01 | 0.036 |
| 2 | 28 | 33 | 5 | as 0 | +0.06 / −0.08 | +0.01 | 0.036 |
| 4 | 26 | 34 | 5 | as 0 | +0.07 / −0.06 | +0.03 | 0.036 |
| 8 | 20 | 32 | 4 | L0, L1 MLP | +0.18 / +0.16 | +0.16 | 0.035 |
| 16 | **7** | 19 | 3 | L0 MLP, L0 heads | +0.46 / +0.63 | +0.42 | 0.032 |
| 32 | **5** | 20 | 3 | L0 MLP | +0.44 / +0.63 | +0.42 | 0.031 |
| 64 | **4** | **15** | 3 | L1, L0 MLP | +0.20 / +0.35 | +0.18 | 0.033 |
| 128 | **3** | **8** | 3 | L1, L2 MLP | +0.11 / +0.25 | +0.14 | 0.034 |
| 256 | 35 | **5** | 3 | L1, L0, L6 MLP | −0.11 / +0.34 | −0.20 | 0.031 |
| 512 † | **9** | **9** | 3 | L5, L1, L3 MLP | −0.09 / −0.13 | −0.22 | 0.024 |
| 1000 † | **16** | 25 | 5 | L1, L5, L4 MLP | +0.19 / −0.23 | −0.10 | 0.058 |
| 2000 † | **7** | 33 | 8 | L1–L5, L7, L10 MLP | +0.90 / −0.94 | +0.44 | 0.134 |
| 4000 | **7** | 33 | 11 | L1, L5, L3, L10 MLP | +1.05 / −1.07 | +0.84 | 0.173 |
| 8000 | **5** | 28 | 12 | L1, L0, L5, L7, L3 MLP | +1.03 / −0.49 | +0.86 | 0.223 |
| 16000 | **5** | **17** | 14 | L0, L1 MLP | +0.78 / +0.28 | +0.72 | 0.247 |
| 32000 | **14** | **14** | 14 | L0, L6 MLP, L0 heads | +0.69 / +0.88 | +0.70 | 0.278 |
| 54000 | 20 | **13** | 14 | heads L0, L9, L5, L7; L6 MLP | +0.82 / +1.26 | +0.75 | 0.268 |
| 143000 | **6** | **12** | 11 | L23 MLP −0.49, L21.h12 −0.10, L22, L20 MLP | +0.90 / +1.48 | +0.70 | 0.249 |

† LR warmup overlaps 512 and 1000–2000 (design "Fences"). Every `|S|` is ≤ 43 at every step.

**Readings (descriptive, tier 1):**
1. **Against is few, and MLP; most of "few" is training concentrating every sign** (corrected
   after `/challenge-pr` on #215, findings 2 and 4): `|S⁻|` is below the band at 11–13 of 18 steps
   (12 on the median's set; 13 taking each prompt's `|S⁻|` first; 8, 1000 and 32000 sit on the
   cut), but the 50 % set of `|proj|` of either sign also falls, from 60 at init to 11–39, and
   `|S⁺|` is below its band at 8 steps. **Against-specific** (`S⁻` few, `S⁺` not): 6 steps, 16, 32
   and 1000–8000. At 16–128 the against mass is small (0.12–0.45, along 0.97–1.23). The
   most-against component is an MLP at 17 of 18 (54000: a block-0 head); early MLPs (L0–L7) from
   16 to 32000, the last block's MLP at 143000. The trained βs are not ≈ 3 throughout: the median
   layer β is about 0 to 4000 (−0.13 to +0.36), 1.3 at 8000, 2.5–3.7 from 16000, so step 0's band
   is the matched one to 4000; at 3.5's band 1000 and 32000 would read cooperating.
2. **Attention against the MLP at 1000–4000 (A1b's carried item): the MLP cancels attention**
   (attention's summed proj +0.19 to +1.05, the MLP's −0.23 to −1.07, the blocks −0.06 to +0.03).
   At 8000 only in part (+1.03 / −0.49, block +0.45); from 16000 both are along.
3. **Attention's along step is drift-robust from 2000** (drift removed +0.44 to +0.86, the raw
   sign); at 256–1000 it is negative both ways.
4. **Block 23 at 143000 (carried): its step against the field is its MLP** (−0.49 of the block's
   −0.56; next `h12` −0.02). Same at M1's βs (`L23.mlp` −0.47; `|S⁻|` 5, `|S⁺|` 14).
5. **The off motion is the MLP's:** 0.68–0.92 of the pooled off motion; unweighted, the MLP's
   per-block share averages 0.68–0.86 and is over half in 20–24 of 24 blocks. Block 0's MLP tops
   `S⁰` at every step (block 0 is 17–32 % of the pooled off weight).
6. **The sink's part is small in the field; ≤ 0.05 of the update is not shown after 1000:** its
   summed proj is at most 0.14 in size (143000), but the bound Σ_h mean|sink_h| / mean|v| has a
   median over blocks of 0.02–0.04 up to 512 and 0.06–0.28 from 1000. The bound cannot confirm 1e's
   ≤ 0.05 there; the exact share was not measured. *(At A2's close: neither the bound nor unit 5's
   exact share is 1e's quantity, the share of the shared update, so neither tests it; "A2 on one
   page" reading 6.)* **The top-key part is not read:** the
   selector (mass over its uniform expectation) favours late keys, whose expectation is ~1/n; at
   143000 57 % of head × block pick a key in the last 10 positions (`/challenge-pr` on #215,
   finding 1, checked). 1e's newline-sink item stays open.

**Not established:** anything causal (unit 5); whether the sets carry over to the long passages
(unit 5 re-ranks there); "few" for `S⁰` (its band has no width); whether 8, 1000 and 32000 are
few; any fixed-key sink (the selector is biased); the sink's exact share of the update.

**After `/challenge-pr` on #215 (accept with changes; findings checked against the records):**
finding 1 (top-key selector biased late) confirmed, reading 6 retracted for it; the fix (mass
minus the uniform expectation, with a planted local head and sink in the tests) and finding 3's
exact sink share change `a2_comp.py`, whose hash the gate pins, so they wait for unit 5's re-gate
(STATE Blocked 59 (ii)) instead of a 3.5 h re-run here. Findings 2, 4 and 5 taken as text and as
report columns (`a2_batch_report.txt` "Beside every |S-|"; report code only, no instrument change).

**For unit 5 (Blocked 59):** at 2000, 8000 and 143000 `S⁻` has 5–7 components, all in the top
size decile (43 components, 24 of them the MLPs). A null matched on type, band and decile then
draws from about 8 MLPs in the set's own band (early at 2000 and 8000, late at 143000), so its 20 random sets would mostly be the claim's own set.

Re-run: `METS_DATA=<data> ./scripts/a2_unit4.sh` (gate, band, batch, report; the batch resumes).
The first pass ran out of GPU memory at step 64 (each record was held across prompts). The runner
freed it per record and resumed; the change touches memory only.

## A2 unit 3, the per-component instrument and its gate (2026-10-16)

Blocked 57 (a), the user's: A1's card stays at `fe_full` with unit 2's (i) beside; the β
convention stays (Blocked 9) with M1's bands beside; unit 2's (iii) (M1 negative mid-depth on code
and LaTeX) is parked for unit 4, which reads the field at A1a's βs (option (a)'s text; the
design's "Unit 3" says unit 2's primary, see Blocked 58 (b)).

**Built** (`design-A2.md` "Unit 3", as frozen): `trackA_force/a2_comp.py`. 18 components per
block (`h0`…`h15` with their value bias, `b_O`, `mlp`); each head's sink part and its
most-received-key part beside. `Δ_c` is the path integral of the LN1 frame's exact Jacobian
(Gauss–Legendre, 16 nodes); per component the mean `proj`, `proj_df` (drift removed), the off share
(pooled over tokens, sums to 1), `size` (mean raw `|v_c|`, unit 5's deciles) and `disp`. All are
linear in `Δ_c`, so they add over components (tests: `tests/test_trackA_a2_comp.py`, and
`_smoke.py` for the hooked split). The block totals carry the log and gnomonic lifted reads and
their ratio to the unlifted mean beside. Two choices the design did not fix: the off share is
pooled over tokens (a per-token mean is unstable where `|o_i|` is small); the per-component step
is unlifted (the design's "A1's gnomonic lift" is A1's *log* lift; both are recorded on the total).
`a1a.ln_from` now re-exports `u2_attn.ln_from` (fence; no number changes; A1a's code hash changes).

**The gate** (`a2_gate.py`; code `d7c4fd8dcbf8`, gate file `0acd9ffa1078`, prompts `976b62d677bc`;
output `data/trackA/a2_gate_2026-10-16/a2_gate.json`, 1,474 s on the RTX 3080). Synthetic stacks:
24 blocks × 16 identity-QK heads at β 3.5, from step 143000's embedding spectrum per v1 prompt,
identity-affine LN1 (G1 also on 143000's LN1). G2: Pythia's init redrawn at each matrix's step-0
σ, 50 seeds + a 51st, v1's 7 prompts, β 3.5.

| arm | bar | result |
|---|---|---|
| G0 quadrature | ≤ 1e-6 rel., every block of every stack | **pass**, max 7.7e-11 (G2 seeds); synthetic ≤ 4.9e-14 |
| G1 planted V = −I | < 0 and first in "against", every block | **pass**, 168 of 168 blocks; runner-up at most 0.12 of it; also on 143000's LN1 |
| G2 null | 51st seed in the 50 seeds' 5–95 % | **pass**: `|S⁻|` 25 in [13.5, 29.7]; top-43 share 0.627 in [0.602, 0.719]. A fresh draw against its own kind can fail only by chance, so this is the init band, measured, not a test of the instrument (`/challenge-pr` on #214, finding 3) |
| G3 off-field head | off share first, `|proj|` ≤ 0.05 of its size | **pass**: off share ≥ 0.905, `|proj|/size` ≤ 0.018 |
| G4 shared drift | MLP `|proj_df|` ≤ 0.1 of its size | **pass**: ≤ 0.033 (raw `proj` ≤ 0.057, beside) |

The pass's checks: heads + `b_O` against attention's output ≤ 8e-8, the counted parts against
the hidden states ≤ 8.1e-8; step 0's pass against its stored rows 2.3e-7.

**Beside, real step 0** (not gated): `|S⁻|` 25, top-43 share 0.640, `|S⁺|` 29, Gini 0.61, both
inside the redraws' range. **For unit 4 (a confound, Blocked 58 (a)):** at a random init the
against set is already "few" by the design's cut (`|S⁻|` ≤ 43): 50 of 50 seeds (8–33) and step 0.
So "few" cannot tell a trained step from init; read against the init band, it can.

Not established: anything about a trained step (no real reading was made); G4 is weak in this
frame (a constant drift barely projects on the field even before removal, raw ≤ 0.057); G1 and G3
are easy controls that catch sign and index errors, not subtle ones (finding 4).

**After `/challenge-pr` on #214 (accept with changes):** finding 1 checked and **confirmed: 16
nodes fail G0 on real block 0**, whose update is 15–48× its input (block-0 total on `homer_iliad`,
stored activations: 2.8e-6 at 2000, 5.5e-4 at 8000, 1.4e-3 at 143000; 32 nodes 3.5e-12, 1.4e-7,
9.5e-7; 64 nodes ≤ 1e-12; blocks 1 and 5 ≤ 1e-15 at 16). The gate's synthetic starts never reached
that ratio, and `CompHooked._read` refuses on the head split but not on G0. The node count is
placed and frozen, so the change is Blocked 58 (c); no code changed here. Finding 2 (the 43 cut
against 206 negative components at step 0) is Blocked 58 (a4). Finding 5: the top key is chosen by
attention received summed over all queries, which favours early keys when attention is near
uniform, and only its position is recorded (the docstring says its token); fixed with unit 4.
Finding 6: A1a's code hash changed with the `ln_from` re-export (Parked).

Re-run: `METS_DATA=<data> python -m trackA_force.a2_gate --out <dir> [--quick]`.

## A2 unit 2, the β refit at 143000 (2026-10-16)

Design `design-A2.md` "Unit 2" (frozen 2026-10-16, Blocked 56 (a)); code `trackA_force/a2_beta.py`
(`59e4193e0dde`; the first run, `87358ccfe397`, gave the same numbers, kept as
`a2_beta_2026-10-16_87358cc/`); inputs A1a's records at 143000 (`a08544e9b644`), v1 (`976b62d677bc`), the
stored `attentions.npz`, A1a's unit LN1 rows (LN cache `ln1_18steps.npz`), position 0 out;
output `data/trackA/a2_beta_2026-10-16/` (main tree; `a2_beta_report.txt`). 5 min (M1 fits
float64 on the GPU, `fe_full` on 7 CPU workers, A1a's hooked pass float32 GPU, TF32 off).
Re-run: `python -m trackA_force.a2_beta run --out <dir>` with the env of STATE "Machine".

**Checks:** (a) B2a unit 2's `beta_full` (14 heads, its own pass, `b2a.loo`): |Δ| 0;
(b) `fe_full` against A1a's 2,688 stored head βs: |Δ| 0. M1: 2,688 per-prompt fits, all
converged (gradient ≤ 1e-6 per row; some ran to the 60-iteration cap). A1a's pass at its own
βs reproduces A1a's `T_proj` and t* exactly (0) before the refit βs are read.

**The rule (primary M1, all heads; p50 outside A1's [p10, p90] in ≥ 4 of 7): 2 of 7, does
not move** (`hdbscan_code`, `latex_monograph`, both below). The bar reads the bracket's middle,
which A1's cell rule never uses (it reads t\* at p10 and p90); what A1's card quotes does move,
and every change strengthens *below* (next paragraph but one; `/challenge-pr` on #213, finding
2). That, not the bar, is why no 18-step refit is needed. Layer-β bracket p10 / p50 / p90:

| prompt | A1 (`fe_full`, all) | M1 all | M1 kept | `fe_full` kept | kept heads (0–22) |
|---|---|---|---|---|---|
| camus_letranger | 2.47 / 4.19 / 5.70 | 1.41 / 2.86 / 3.22 | 2.83 / 3.37 / 4.07 (13 layers) | 5.00 / 6.23 / 7.24 | 116 of 368 |
| hdbscan_code | 1.06 / 3.00 / 5.89 | −10.42 / −4.31 / 2.27 | −4.07 / 1.16 / 3.31 (10) | 5.82 / 6.83 / 8.44 | 93 |
| homer_iliad | 1.89 / 3.15 / 6.45 | 1.37 / 2.49 / 3.32 | 3.39 / 4.15 / 5.37 (9) | 6.16 / 7.49 / 9.13 | 91 |
| latex_monograph | 0.45 / 2.36 / 3.66 | −16.67 / −6.01 / 0.69 | 1.21 / 1.97 / 2.30 (3) | 5.82 / 7.40 / 9.41 | 41 |
| paper_excerpt | 1.04 / 4.09 / 6.10 | 0.55 / 2.70 / 3.64 | 3.55 / 4.33 / 6.27 (10) | 6.92 / 9.20 / 10.03 | 95 |
| sullivan_ballou | 1.14 / 3.08 / 6.21 | 1.13 / 2.41 / 3.44 | 3.70 / 4.34 / 5.28 (8) | 6.70 / 7.61 / 9.33 | 77 |
| wiki_paragraph | 2.17 / 4.54 / 7.14 | 1.29 / 3.11 / 3.52 | 3.31 / 3.94 / 5.57 (13) | 6.33 / 7.98 / 9.18 | 120 |

"kept" = `fe_full` `partial_r2` ≥ 0.10, ≥ 4 heads per layer; the kept columns have no β at most
layers (count in brackets), so they are beside only. What the 2 × 2 says: **the estimator lowers
β and narrows the bracket from above** (M1's p90 3.2–3.6 against A1's 5.7–7.1 on 5 prose
prompts); **the floor raises it** (`fe_full` kept 6–9, as 1d found at 0.10). On the code and
LaTeX prompts M1 is negative at most mid layers (136 and 182 of 368 heads below −5, layers
5–20; head-level correlation with `fe_full` 0.26 and 0.20, against 0.73–0.82 on the other
five): a sign the two estimators disagree on, not a small shift. **This is where B2a's "sign
bias" came from** (`/challenge-pr` on #213, finding 1, checked here from the per-prompt fits):
B2a's negative pooled heads are positive on every prose prompt (`L6H4` 2.1–3.1, `L15H7`
3.4–5.0) and negative on code and LaTeX only (−9 to −34); `L18H1` is mixed. So it is not a bias
of `fe_full` on prose; `trackB_circuit/status-B.md` "Corrections received".

**A1b's cell at 143000, re-run at M1's layer βs** (design: whatever the bar says): step reading
*below* → *below*, 6 of 7 → 7 of 7. `T_proj` falls 5–32 % (the field's direction changes with
β); t\*(p10) barely moves (2.44–2.88), t\*(p90) falls from 3.3–23.9 to 2.5–3.5, so the bracket
narrows. `paper_excerpt` *straddles* → *below* **through `T_proj`** (2.65 → 2.18, under its
t\*(p10) 2.60), not through t\*. Median `T_proj / t*(p10)` 0.82 → 0.72. Negative layer βs enter
the field and t\* as they are: t\* stays near 2.5 (the identity-weight model collapses at any β),
but on the two negative-β prompts the field at those layers points toward the *least* similar
tokens, which matters for unit 3's split of "against" from "off" (finding 3).

**β per band (heads × prompts pooled, beside; convention 3.5 [1.6, 5.6], Blocked 9):** M1
L1–8 2.71, L9–16 2.04, **L17–23 0.88 (outside)**, L1–23 1.97; `fe_full` 4.36, 3.65, 2.37, 3.46
(1d's numbers, reproduced). **Post hoc, on the five prose prompts only** (finding 1): M1 3.12,
2.79, **1.66**, 2.70, every band inside the convention; the 0.88 comes from pooling in code and
LaTeX. Sent to the user as Blocked 57; nothing changed.

Not done, by the rule: the refit at all 18 steps and a "Corrections received" line (only if
it moves). Not established: which estimator is right on the code and LaTeX prompts (M1's φ may
absorb content there, `design-A2.md` "Worth challenging" 1); whether M1's lower β holds at other
steps.

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

**From A3 unit 4 (2026-10-20):** E6b's bigram table size (why: 2M tokens make a weak bigram, its
loss 7.02 against the unigram's 7.95; cost: count the rest of the validation split, CPU minutes,
and re-run the 18-step pass, 3 GPU minutes; could change: E6b's step, so the E1–E6b pair in unit 5,
which is aligned at 256 or 512 or 1000 and misaligned only past 1000).

**From the A5 / C2 scan (2026-10-18):** `lit-C2.md` Parked 1 (full texts now fetchable) open;
Parked 2 (PolyPythias's outlier seeds) done, already in `lit-1d.md`; Parked 3 (2602.15997's
top-down collapse) carried to A3.

**At A2's close (2026-10-17)** each item opened during A2 is *done*, *carried to X* or *dropped
(why)*; the items carried into A2 at A1's close are settled below, under their own entries.

- *A1a's code hash* (A2 unit 3, 2026-10-16): `a1a.py` re-exports `u2_attn.ln_from`, so A1a's
  `code_sha` is no longer `a08544e9b644`; re-running A1a's batch would be refused against its
  old gate. No number changes. **Dropped**: A1 is closed and no re-run is planned; a re-run
  would re-gate anyway (one gate, wall time only).
- *The ablation's repair* (Blocked 60 (b), not taken): one synthetic model read on 7 starts, a
  planted stack with headroom on (iii), step 0 required to read "refused", and a depth control
  for MLPs. **Carried to C2**, as the precondition of any causal claim about which components
  push apart; the depth control is the open part and has no design yet.
- *The real ablations uncalibrated* (Blocked 60 (c)): **dropped**: nothing calibrates them.
- *M1 against `fe_full` on code and LaTeX* (unit 2): opposite signs mid-depth. **Carried to B3**,
  which reads per-head β against condensation thresholds and has to pick an estimator first.
- *The timeline rows* (units 4, 5): the against-specific steps (16, 32, 1000–16000, 143000); the
  MLP cancelling attention at 1000–4000; the fixed key moving from a period (2000–4000) to a
  newline (from 8000); the sink's per-token share rising from 1000. **Carried to A3**.
- *The sink's value against its attention* (1e's item, carried into A2 with `attention-10.md`'s
  A5): the sink's part along the field is small (reading 6, done); its share was measured on
  another quantity, so 1e's ≤ 0.05 was not tested (and 1e's per-head arm reaches −0.19).
  **Dropped** for the rest: no planned unit reads the sink as a separate source, and the
  per-token rise is an A3 row; reopen if one does.
- *An energy split per component* (ROADMAP's A2 row: `PROJECT.md` §3.8.3's exact dissipation;
  the card's "A2's dissipation per component"): A2 read projected steps and off shares instead.
  **Carried to C2**, beside the repair: any energy sentence about components needs it (a new
  instrument, `dE_β` per component through the same path integral).
- *A second sink on a newline* (1e's item): **done**: one case of a fixed key (reading 6).

At A1's close (2026-10-15) each item is *done*, *carried to X* or *dropped (why)*. Items carried
to A2 are settled at A2's close: the drift question, `ln_from`, the off-field share, attention
against the MLP and block 23 are **done** (A2 readings 2–5); the long passages were used at step
0 only (unit 5's ranking and means), and A1's continuity check on them is **dropped** (GPU hours
for a closed number that unit 2 already re-read; reopen if A5 registers on length).

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
- *A1a's β sign bias* (B2a unit 2, `/challenge-pr` on #209, finding 4; `trackB_circuit/status-B.md`
  "A1a's β against M1's"): at 143000 A1a's per-head `fe_full` β is positive in 13 of 14 B2a heads,
  and in the 6 where M1's cosine adds nothing M1's β is negative, A1a's positive. A1's layer β is
  the median over 16 heads, so the bias can move A1's bracket and `P-gamma2`'s t*. **Carried to
  A2 as its first item** (Blocked 54 (a), the user's, 2026-10-16): one CPU pass over stored
  attention; could change A1's closed number. **Done 2026-10-16** (A2 unit 2): by the bar it
  does not; Blocked 57.

## Corrections received

- 2026-10-15: `design-A.md` "Frozen 2026-10-14" cited 2604.01978 for the 2D spike's reduction; the section is 2609.28448 §6.1 (Gao–Yang–Chen). A citation only; no A1 number moves — `trackB_circuit/lit-B.md` "B2a" §1.
