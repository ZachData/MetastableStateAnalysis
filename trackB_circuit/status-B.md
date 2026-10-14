<!-- trackB_circuit/status-B.md -->
# Track B — STATUS

No card yet (`docs/phase_card.md`); one is added with B1's verdict.

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check | **done 2026-10-13**; frozen by the user 2026-10-14 (Blocked 42) | `design-B.md`, `lit-B.md`, `tools/math_checks/b0_antisymmetric_nudge.py` |
| 2. B0, the harness and its gate | **built; all 18 steps run under Blocked 43 (a)'s G4b: G0–G4a pass at every step, G4b fails at step 128 on one arm of 224, a near-tie; Blocked 44** | `core/nudge.py`, `core/position.py`, `trackB_circuit/b0_gate.py`; "B0 (2026-10-14)" below |
| 3. B1, `L5H2` as positive control | not started; runs only after B0's gate passes | `design-B.md` "B1" |

Budget: 3 units (`design-B.md` "Exit condition and budget").

## B0 (2026-10-14): the gate at its first step

**Input:** `pythia-410m` step 143000; the 7 scored v1 prompts (`b0_gate.V1_SCORED`, prompts
hash `976b62d677bc`), ≤ 512 tokens; `resid_pre` at layer 5; B1's 32 positions per prompt
(`core.nudge.draw_positions`); K = 32 directions seeded per (prompt, `resid_pre`, 5). GPU
float32 on the RTX 3080, deterministic, TF32 off (torch 2.10.0+cu130, transformers 4.44.2); G2
on CPU float64. Code `329eba22f8c6`. Records: `/run/media/system/HDD_1TB/mets_data/trackB_b0/
2026-10-14b/{gpu,cpu}_step143000.json`. GPU 660 s for the step, G2 27 s.

**Re-run after `/challenge-pr` on #198** (finding 1: `Stack.block` crashed on transformers 4.57,
and 4.57's eager attention masks only by what it is passed; the block now always passes an
explicit causal mask and 4.57's rotary embeddings; smoke tests pass on 4.44.2 and 4.57.6): the
first record (code `db4804772c65`, `2026-10-14/`) and the re-run agree in all 5,870 GPU and 388
CPU fields but timings. `report` now refuses a record from other code.

| arm | rule (`b0_gate.py` docstring, fixed before output) | at 143000 |
|---|---|---|
| G0 | loop vs HF logits ≤ 1e-5 | **pass**: 0.0 exactly, all 7 prompts |
| G1 | the add, to 8 ulp; 0 off j | **pass**: worst 0.058 of tolerance; off j exactly 0 |
| G2 | A/ε vs JVP, ≤ 5 % at ≥ 90 % of tokens, c = 0.01 | **pass**: 10424/10424; median relative error 1.1e-4, 1.0e-3, 1.1e-2, 9.9e-2 at c = 0.01, 0.03, 0.1, 0.3 (ε² growth; at c = 0.3 only 9.8 % of tokens are within 5 %: the plateau's edge) |
| G3 | planted ∇L: within 10 % and rank 1, ≥ 6/7 | **pass**: 7/7; A/ε over ‖∇L‖ 0.998–1.020 |
| G4a | causality, exact 0 before j | **pass**: 0 exactly at every later layer and the loss |
| G4b | another prompt's ∇L: rank-1 rate ≤ 0.053 (N = 224) | **fail**: 17, 17, 17, 21 of 224 (0.076, 0.076, 0.076, 0.094) at c = 0.01 … 0.3 |
| floor | four same-shape base rows | 0 exactly everywhere: no readout below floor |

**Why G4b fails** (`trackB_circuit/b0_g4b_geometry.py 143000 <out>`, no nudge): ranking
\|∇L_j · g′\| among \|∇L_j · R_k\| from the gradients alone gives **the same 17 rank-1 hits**
(the rank-1 indicator agrees with the nudged c = 0.01 at 224 of 224; the exact rank at 189,
within 1 at 217, at most 3 apart). Loss gradients at layer 5 are aligned
across prompts: mean \|cos\| over cross-prompt pairs 0.031, against 0.025 for random directions.
So the harness reads A = ε∇L·d as G3 says it does, and the excess is the model's gradient
geometry. The arm assumed an on-distribution direction from another prompt ranks like a random
one; at 143000 it does not. The frozen rule still reads **fail**: Blocked 43.

**Not established:** any step but 143000; whether G4b's excess holds at other steps; G2 at
any other step.

**Re-run:** `python trackB_circuit/b0_gate.py run --out <dir> --steps 143000`, then `g2`,
then `report` (it needs all 18 steps for a verdict).

## B0 over 18 steps (2026-10-14, Blocked 43 (a))

**Input:** as "B0 (2026-10-14)" above, all 18 steps; G4b's rule as amended by Blocked 43 (a)
(the user's), committed before the run (`8330f7b`). Code `1e78e2be9d94` in all 36 records.
Records: `/run/media/system/HDD_1TB/mets_data/trackB_b0/2026-10-14c/{gpu,cpu}_step*.json`,
`gate.json`. 207 min (GPU ~11 min per step; G2 on CPU first).

**Verdict by the rule: FAIL**, at one arm. G0, G1, G2 (10424/10424 tokens within 5 %), G3
(7/7) and G4a pass at all 18 steps; G4b passes at 17 of 18.

| step | 0 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rank-1, nudged / linear (of 224) | 32/32 | 32/32 | 32/32 | 36/36 | 34/34 | 49/49 | 86/86 | **114/115** | 119/119 | 82/82 | 36/36 | 20/20 | 15/15 | 7/7 | 14/14 | 9/9 | 12/12 | 17/17 |
| exact rank equal | 223 | 224 | 223 | 223 | 224 | 224 | 224 | 221 | 224 | 224 | 223 | 223 | 222 | 223 | 223 | 220 | 223 | 189 |
| rank-1 rate at c = 0.01 (old bound 0.053, not judged) | 0.143 | 0.143 | 0.143 | 0.161 | 0.152 | 0.219 | 0.384 | 0.509 | 0.531 | 0.366 | 0.161 | 0.089 | 0.067 | 0.031 | 0.062 | 0.040 | 0.054 | 0.076 |

(Every step: within 1 rank at 224 of 224, except 128 (223) and 143000 (217). Rates at the other
c: `gate.json`.)

**The one mismatch is a near-tie.** Step 128, `camus_letranger` j = 82, d from
`hdbscan_code`: linear |∇L·d| = 0.0725662 against the top random direction's 0.0725586 (k = 10),
a relative margin of 1.05e-4; the nudged |A|/ε at c = 0.01 is 0.072468, 1.4e-3 below linear, so
it ranks 2. Ten times the margin, and well inside G3's 10 %. This is the failure the amended
rule's text named (a near-tie inside the linear error), at 1 arm of 4,032. Re-derived by
`python trackB_circuit/b0_g4b_tie.py` (the arm alone on the GPU; ranks as in the record).

**What it shows about the model, beside the gate:** another prompt's loss gradient outranks all
32 random directions in 14 % of arms at init, rising to 53 % at step 256 and falling to
3–8 % from 8000. Loss gradients at layer 5 are aligned across prompts at every step, most
strongly at 64–512. This sharpens the B4 fence below.

**Not established:** the false-fail rate of G4b's amended rule (one near-tie in 4,032 arms is
its only reading); G2 at steps other than J[0:4] and R[0].

**Re-run:** `D=<dir>; python trackB_circuit/b0_gate.py g2 --out $D && python
trackB_circuit/b0_gate.py run --out $D && python trackB_circuit/b0_gate.py report --out $D`.

## Fences (from B0's gate)

- **B4 brings an on-distribution null.** Loss gradients at layer 5 share a direction across
  prompts: mean \|cos\| over cross-prompt pairs 0.031 against 0.025 for random directions; the
  median \|projection\| of a unit gradient on the mean gradient direction is 0.068 (a random unit
  vector: ~0.025). So "a structured direction ranked among random ones" is biased upward by the
  model itself, and B4's eigen-directions against random ones would read that bias as a finding.
  (`/challenge-pr` on #198, finding 3; carried from Parked.)
  Over the 18 steps the rank-1 rate of another prompt's gradient is 3–53 % against 3 % for a
  random direction ("B0 over 18 steps"), so the bias is largest at 64–512.
- **B1's floor.** The same-shape base floor is exactly 0 under deterministic algorithms (it
  measures repeatability, not rounding), so "below floor" never fires. B1's offsets k = 2…5 can
  sit near float32 rounding; B1 fixes a rounding floor before its run (e.g. the same pass in
  float64, or in another batch shape). (`/challenge-pr` on #198, risk 1.)

## Parked

(none)

## Corrections received

(none)
