<!-- trackB_circuit/status-B.md -->
# Track B — STATUS

<!-- phase-card -->
## Card

- **Question:** Does a paired ± nudge of one token's residual, gated against planted effects and random directions (B0), see a known circuit's particle signature (each token pulled toward its left neighbour inside `L5H2`'s output, ranked against layer 5's other heads, B1) switch on over training, so that "small" has a yardstick before the harness reads unknown mechanisms?
- **Inputs:** `pythia-410m`, the 18 Stage 0 steps, the 7 scored v1 prompts (B0's `prompts_hash`), `resid_pre` at layer 5, 32 seeded positions and K = 32 seeded random directions per prompt, c ∈ {0.01, 0.03, 0.1, 0.3}, GPU float32 — `trackB_circuit/design-B.md` "B1"
- **Results:**
  - Opened 2026-10-13 with a scan, a `sympy` check of the paired-pass algebra and a frozen design (3-unit budget) — `trackB_circuit/design-B.md` "Exit condition and budget"
  - B0's harness passes its calibration gate at every step and arm but one near-tied G4b arm at step 128, waived by the user as that arm only (Blocked 44 (a)) — `trackB_circuit/status-B.md` "B0 over 18 steps"
  - Loss gradients at layer 5 are aligned across prompts at every step, most at 64–512, so a structured direction ranked against random ones is biased upward by the model itself — `trackB_circuit/status-B.md` "Fences (from B0's gate)"
  - B1 passes by its revised pass rule (off at 0, on at 1000–8000): `L5H2`'s signature reads off at steps 0–256 and on at every step from 512; the onset falls in (256, 512], inside the range recorded before the run and before ROADMAP's first window (512, 4000]; read from block 5 alone, on one head's output — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - B2a (the 2D spike) opened 2026-10-15 with a scan, a `sympy` check (a head's own QK is exact on a subspace of dimension ≤ 128) and a design not yet frozen; its reduction is 2609.28448's general-value-map section, on V's invariant subspaces — `trackB_circuit/design-B2a.md` "The paper, corrected"
  - B2a's instrument passes its planted gate (circle, rotation, D = 4, positional, null 20 of 20, out of class, arithmetic) at the design's constants; on the weights V's top pair is a D = 2 block (complex or degenerate) in 123 of 252 cells, near-repulsive in 10, and no V block beats the kernel in 31 cells at 1000–32000 — `trackB_circuit/status-B.md` "B2a unit 1 (2026-10-15)"
  - At step 143000 the reduction's premise fails: the real tokens hold about the random share of their norm² in V's invariant subspaces, so no head of 14 has D\* ≥ 1 (no circle); 4 read positional (D\* = 0, `L5H2` post hoc), 8 outside the paper's class (at the placed 0.5), 2 "none"; the isotropic cosine adds at most 0.08 of TV over position — `trackB_circuit/status-B.md` "B2a unit 2 (2026-10-16)"
  - The rule at all 18 steps finds no head of 14 with D\* ≥ 1 at any step: to 256 attention is near uniform and every head reads outside (nothing to fit); positional heads appear from 512 (`L5H2`, `L4H9`), and V's subspaces never carry the content a head reads; B2a is at its exit with its 4 units spent — `trackB_circuit/status-B.md` "B2a unit 3 (2026-10-16)"
  - B2a closed 2026-10-16 at its exit: on 410m no head's attention lives in a small subspace of V, mostly because trained heads are not isotropic content plus position (214 of 252 cells outside); where V's ladder was reached the tokens hold about chance in V's subspaces; V's 2D blocks are at the random-weight rate, training shows in their sign — `trackB_circuit/status-B.md` "B2a on one page (closed 2026-10-16)"
  - B2 (spectra from the weights) opened 2026-10-17 with a scan, a `sympy` check (a rank-64 head's symmetrised OV or QK has exactly 64 negative eigenvalues whenever its read and write spaces are in general position, so sign counts are structural) and a design frozen with its LN1 frame's QK form corrected to B2a's — `trackB_circuit/design-B2.md` "The math"
  - B2's columns and member statistic pass their weights-only gate at every arm with the design's constants; read at one step, the statistic sees a set of members only when all sit near their step's 90th percentile (8 % of planted sets at the 80th) — `trackB_circuit/status-B.md` "B2 unit 1 (2026-10-17)"
  - On the rule's own 9-step statistic the gate still passes (100 % at the 90th), so the constants stand. The reach is 42 % at the 80th. Sets held between the 25th and 75th cannot be seen at all, because the null's mean |p − ½| is exactly ¼; sets beyond, high or low, are seen alike. Sets planted only from 8000 are seen in 67 % of draws at the 90th. A single copied direction per pair is seen in C6 in 55 % of draws — `trackB_circuit/status-B.md` "B2 unit 1b (2026-10-18)"
  - B2's table over the 18 steps: the members differ from their layers' heads in QK symmetry (C3, rank 0.9996; `L7H8` the most symmetric head from 2000, three others near the 95th, two near the 15th) and in K-composition (C6, rank 1.0, carried by `L5H2` writing into three induction heads' keys at 21–32× chance, from 512–1000, with non-member heads reading it about as much), but show no difference in the OV attractive share, its top weight or the OV–QK cosine (C1, C2, C4), within the gated reach; B2 is at its exit — `trackB_circuit/status-B.md` "B2 unit 2 (2026-10-18)"
  - B2 closed 2026-10-18 at its exit: in the weights the circuit is `L5H2` writing into three induction heads' keys (with non-member readers beside), and a QK-symmetry split (circuit or head type open); C1, C2, C4 show no difference within the gated reach — `trackB_circuit/status-B.md` "B2 on one page (closed 2026-10-18)"
  - B3 (routing) opened 2026-10-18 with a scan, a `sympy` check and a design, frozen the same day (Blocked 65 (a)). The check shows that condensation in 2609.28448 is a large-N limit with no finite-N threshold, and that a sharp head with no growing logit gap still gives a partner-growth slope well below 1 at these context lengths. So B3 reads routing from the attention itself, with β beside, and no head is placed against a threshold — `trackB_circuit/design-B3.md` "The math"
  - B3 unit 1's gate fails before any member α is read: the stored attention and head types reproduce, and planted α reads right, but the type-matched null for C3′ false-passes 6 % and R2's onset rule misses planted onsets (init structure in the early bottom decile). The induction members do not share an offset, and `L12H5` is not an induction head on 7d's probe. Revisions are tested on the gate and put to the user — `trackB_circuit/status-B.md` "B3 unit 1 (2026-10-18)"
  - Revised by the user's choice (Blocked 66 (a): each member typed at its own offset, `L12H5` untyped, a step-0 floor on R2's onset, inclusive null pools), the gate passes. The members' routing exponent α shows no difference from their layers' heads (R1, rank 0.80). Only `L5H2` of 4 condenses at its formation step (R2). Against the heads nearest in type, B2's QK symmetry does not set the members apart (C3′, 0.945): C3 is not shown to go beyond head type, though only two of the six pools are close in type — `trackB_circuit/status-B.md` "B3 unit 1b (2026-10-18)"
  - The light cone (Blocked 67 (a)): a nudge at block 4 on repeated random tokens reaches the induction partner (j + 63) about 11× more than its neighbours, from step 1000. Ablating the three typed induction members removes 0.54 of ℓ's 2.37, above all 200 random layer-matched 3-sets (L1, rank 1.000). The three nearest-type non-members remove 0.05 (0.88, not shown). Of the members' 0.54, about 0.24 is the partner's response falling and 0.30 the neighbours' rising, so about half of it is the response being kept off nearby tokens. ℓ ≈ 2.0 after block 6 is a ratio on a small signal: the partner's response there is 0.097, 8 % of its final 1.17, so the cone's size is added after block 6 (unit 2's "most of the cone is in place after block 6" was wrong). That early ratio is `L6H3`'s (ind1 0.46, untyped), and ablating it leaves the final response unchanged; `L6H0` (ind0 0.75, untyped) leaves the partner unchanged and lets the neighbours rise (Δℓ 0.18) — `trackB_circuit/status-B.md` "B3 unit 2 (2026-10-19)", "B3 unit 2b (2026-10-19)"
  - B3 closed 2026-10-19 at its exit: the light cone runs along induction partners from step 1000 and goes through the three typed induction members in part (half of their share keeps the response off nearby tokens); the partner's response is added across depth (blocks 6–8, 10–11 and 16–23); the members' routing exponent shows no difference from their layers', and B2's QK-symmetry split is not shown to go beyond head type. Path patching, the heads that add the response, a block-6 nudge site and a layer-matched same-type control are carried to B4 — `trackB_circuit/status-B.md` "B3 on one page (closed 2026-10-19)"
  - 2026-10-24: B2a's and B3's citations of 2609.28448 give section numbers the paper does not have (the paper's V A and V B were cited as 6.1, 6.2 and 7, with four more); a citation only, no reading moves — `trackB_circuit/status-B.md` "Corrections received"
  - B7 (routing activity across depth) opened 2026-10-25 with a scan, a `sympy` check and a design, frozen the same day (Blocked 110 (a)); its gate passes every arm (unit 1), no real cell read yet; the review then found its offset-profile null biased upward when partners persist on few rows. Revised over pairs of different rows (Blocked 111 (a)), the null no longer moves with persistence. The re-gate fails on one arm: with every head identical, too few rows enter at the minimum entry. With heads entering on the all-prompt best match (Blocked 112 (a), P still leave-one-out) the re-gate passes every arm (unit 1c); no real cell read yet — `trackB_circuit/status-B.md` "B7 unit 1", "B7 unit 1b", "B7 unit 1c" The check shows that on diffuse rows the argmax switch rate and the soft overlap say opposite things whatever the dynamics, and that a shuffled-layer null sees drift but not freezing. So B7 reads only condensed rows, against an offset-profile null as well, with each head matched to the next layer's best head — `trackB_circuit/design-B7.md` "The math"
  - B7's grid over 18 steps × 23 layer pairs (unit 2, B7 at its exit): nothing is condensed to step 256; from 1000–2000 mid-depth (layers 7–20) is condensed and persistent layer to layer beyond position and beyond distant layers; rerouting appears in only 7 cells (3 → 4, 5 → 6, gone from 32000), early layers are positional, and the last layer pairs are the most condensed but no more persistent to the next layer than to distant ones. 5 of 38 "frozen" and 2 of 7 "rerouting" cells rest on the interval's few-prompt draws — `trackB_circuit/status-B.md` "B7 unit 2"
- **Superseded / wrong:** B3 unit 2's "most of the cone (ℓ ≈ 2.0) is already in the residual after block 6, before any member runs" (2026-10-19, #228): ℓ is a ratio, and the partner's σ there is 8 % of its final size — `trackB_circuit/status-B.md` "B3 unit 2b (2026-10-19)"
- **Registry:** none, because B1 is a positive control on an exploration rung — `trackB_circuit/design-B.md` "Fences"
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - From step 1000 the statistic's size is a float32 detection limit, not a measurement, so only its rank is read — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - B1 reads one head's output after block 5 alone; the multi-layer path and the residual and loss readouts B2–B4 use are covered only by B0's gate — `trackB_circuit/status-B.md` "B1 (2026-10-14)"
  - G4b's false-fail rate is not derived, and G2 was checked only at its four design positions — `trackB_circuit/status-B.md` "B0 over 18 steps"
- **After Phase 10:**
  - B2–B4 on this harness, with an on-distribution null for B4 *(forward pass, GPU minutes per step)*
  - The weight-space site (a head's OV moved by ε), gated on its own *(forward pass, GPU minutes per step)*
  - From B3 at close, for B4: path patching of L1's 0.54 and the 1.84; the heads that add the partner's response (blocks 6–8, 10–11, 16–23); a block-6 nudge site; a layer-matched same-type control for L1 *(forward pass, GPU minutes at 143000)* — `trackB_circuit/status-B.md` "B3 on one page (closed 2026-10-19)"
- **Reviewed:** 2026-10-25 · body `a11aee6d20`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check | **done 2026-10-13**; frozen by the user 2026-10-14 (Blocked 42) | `design-B.md`, `lit-B.md`, `tools/math_checks/b0_antisymmetric_nudge.py` |
| 2. B0, the harness and its gate | **done 2026-10-14**: G0–G4a pass at every step; G4b at 17 of 18, failing at step 128 on one near-tied arm, **waived as that arm only** (Blocked 44 (a), the user's, 2026-10-14; `design-B.md` "If any arm fails") | `core/nudge.py`, `core/position.py`, `trackB_circuit/b0_gate.py`; "B0 over 18 steps" below |
| 3. B1, `L5H2` as positive control | **done 2026-10-14: passes** ("sees") | `trackB_circuit/b1_prev_token.py`; "B1 (2026-10-14)" below |

Budget: 3 units (`design-B.md` "Exit condition and budget"), all spent: Track B's opening is
closed at its exit condition. What follows on these instruments (B2 onward) is `docs/ROADMAP.md`
§7, after Track A (Blocked 40 (d)).

**B2a** (its own budget, `design-B2a.md` "Exit condition and budget"; Blocked 50 sets it):

| unit | state | where |
|---|---|---|
| 0. design, scan, `sympy` check | **done 2026-10-15**; frozen by the user 2026-10-15 (Blocked 50 (a): 4 units, M1 primary) | `design-B2a.md`, `lit-B.md` "B2a", `tools/math_checks/b2a_subspace_ceiling.py` |
| 1. weights only, planted gate | **done 2026-10-15: every arm passes**; weights rows written | `trackB_circuit/b2a.py`; "B2a unit 1 (2026-10-15)" below |
| 2. real tokens at 143000 | **done 2026-10-16: a verdict for all 14 heads; no head has D\* ≥ 1** | "B2a unit 2 (2026-10-16)" below |
| 3. the 18 steps | **done 2026-10-16 (Blocked 53 (a), occupancy first; then the rule at all 18 steps after `/challenge-pr` on #210): no head has D\* ≥ 1 at any step; B2a is at its exit, budget spent** | "B2a unit 3 (2026-10-16)" below |

**B2a closed 2026-10-16** at its exit condition (Blocked 55 (a), the user's): "B2a on one page" below.

**B2** (its own budget, `design-B2.md` "Exit condition and budget"; Blocked 61 sets it):

| unit | state | where |
|---|---|---|
| 0. design, scan, `sympy` check | **done 2026-10-17**; frozen by the user 2026-10-17 (Blocked 61 (a): 3 units; LN1 frame's `M_0` corrected at the freeze) | `design-B2.md`, `lit-B.md` "B2", `tools/math_checks/b2_symmetric_inertia.py` |
| 1. columns, the rule, the gate | **done 2026-10-17: every arm passes**, constants unchanged | `trackB_circuit/b2.py`, `tests/test_b2.py`; "B2 unit 1 (2026-10-17)" below |
| 1b. the 9-step re-gate (rule 6's gate, Blocked 62 (a)) | **done 2026-10-18: passes**, constants unchanged; reach at 9 steps and G5's dose curve recorded | `b2.py regate`; "B2 unit 1b (2026-10-18)" below |
| 2. the table over the 18 steps (Blocked 63 (a): the frozen T ranked) | **done 2026-10-18: C3 and C6 read "members differ", C1, C2, C4 "no difference shown"; B2 is at its exit, budget spent** | `b2.py table`, `b2_show.py`; "B2 unit 2 (2026-10-18)" below |

**B2 closed 2026-10-18** at its exit condition (Blocked 64 (a), the user's): "B2 on one page" below.

**B3** (its own budget, `design-B3.md` "Exit condition and budget"; Blocked 65 sets it):

| unit | state | where |
|---|---|---|
| 0. design, scan, `sympy` check | **done 2026-10-18**; frozen by the user 2026-10-18 (Blocked 65 (a): 3 units, the placed constants as written) | `design-B3.md`, `lit-B.md` "B3", `tools/math_checks/b3_condensation_exponent.py` (3 checks pass) |
| 1. routing (R1, R2, C3′, A3's row, β beside) | **gate run 2026-10-18: G0, G1 pass; G2 (C3′) and G3 fail, so no member α was read.** Revised (Blocked 66 (a), the user's) and **re-gated 2026-10-18: G0–G3 pass; R1, R2, C3′ read "no difference shown"** | `trackB_circuit/b3_route.py`, `b3_gate_variants.py`, `tests/test_b3_route.py`; "B3 unit 1b (2026-10-18)" and "B3 unit 1 (2026-10-18)" below |
| 2. the light cone (L0, L1; Blocked 67 (a) with the same-type arm) | **done 2026-10-19: G2 (L1), G4–G6 pass; L0 passes; L1 reads "the cone goes through the members" (rank 1.000); the same-type arm does not (0.88). B3's 3 units are spent; the close-out is next** | `trackB_circuit/b3_cone.py`, `tests/test_b3_cone.py`, `tests/test_b3_cone_smoke.py`; "B3 unit 2 (2026-10-19)" below |
| 2b. layer 6's single-head ablations (Blocked 69 (c), one small unit on the record; descriptive, outside the family) | **done 2026-10-19: block 6's output carries 8 % of the partner's final response; its high ratio there is `L6H3`'s, which the final response does not depend on; `L6H0`'s Δℓ 0.18 is the neighbours rising. Corrects unit 2's "most of the cone is in place after block 6" (`/challenge-pr` on #228's 2b commit). The close-out is next** | `trackB_circuit/b3_cone_l6.py`; "B3 unit 2b (2026-10-19)" below |

What was seen before the readings were fixed: `design-B3.md` "Seen before the readings were fixed",
and unit 1's gate record (below), which includes the members' type scores and formation steps but
no member α.

**B3 closed 2026-10-19** at its exit condition (Blocked 69 (a), the user's): "B3 on one page" below.

**B7** (its own budget, `design-B7.md` "Exit condition and budget"; Blocked 110 sets it):

| unit | state | where |
|---|---|---|
| 0. design, scan, `sympy` check | **done 2026-10-25; frozen as revised (Blocked 110 (a), the user's)**, revised after `/challenge-pr` on #271 (the first proposal's N0 sat at half of chance and its in-sample best-of-16 match 40 % above it, so a pure reshuffle would have read "persistence"; now leave-one-prompt-out matching, a row-valid binned N0, an ordered partition of descriptions, 3 units). Inputs checked: all 18 × 7 `attentions.npz` `(24, 16, n, n)` and `activations.npz` `(25, n, 1024)` present, so no forward pass | `design-B7.md`, `lit-B.md` "B7", `tools/math_checks/b7_partner_persistence.py` (7 checks pass since unit 1b; re-run: `python tools/math_checks/b7_partner_persistence.py`) |
| 1. `b7_activity.py`, gate G0–G9, code tests | **done 2026-10-25: every arm passes** on the first run; `/challenge-pr` on #272 found N0 biased under persistence (below) | "B7 unit 1" below |
| 1b. reserve (Blocked 111 (a), the user's) | **done 2026-10-25: the re-gate fails on one arm.** N0 over pairs of different rows; the review's fixes. Every arm passes but G1 at the minimum entry under the 8000 and 143000 sinks, where 16 and 18 of 23 cells do not enter (h\* selection under tied heads). The reserve is spent: **Blocked 112** | "B7 unit 1b" below |
| 1c. entry on the all-prompt h\* (Blocked 112 (a), the user's; budget 3 → 4 units on the record) | **done 2026-10-25: every arm passes**, G1 at the minimum entry 23 of 23 at every sink step | "B7 unit 1c" below |
| 2. the 18 steps × 23 layer pairs | **done 2026-10-25: the table is written with its input; B7 is at its exit, 4 of 4 units spent.** Nothing condensed to 256; mid-depth "depth-local persistence" from 1000–2000 (110 cells), rerouting in 7 cells (3 → 4, 5 → 6, gone from 32000), the last layer pairs "frozen across depth"; 5 of 38 F and 2 of 7 R cells rest on the interval's selection tail (caveat 1). The close-out page is next (writing only) | `b7_activity.py read`, `b7_show.py`; "B7 unit 2" below |

## B7 unit 2 (2026-10-25): the 18 steps × 23 layer pairs

**Input:** A1a's 18 run dirs × the 7 scored v1 prompts (`prompts_hash` `976b62d677bc`), each
attention and activation file's sha256 matched to the gate's (`read` refuses otherwise); B3's rows
at code `2c13479d759c` for α. Code `trackB_circuit/b7_activity.py` `a1de85990c74`; gate
`data/b7/2026-10-25-unit1c-r2/gate.json` (unit 1c, passes on this code); records
`data/b7/2026-10-25-unit2/step<S>.json` (main tree), 2000 draws, seed `seed_for("B7", "read", step)`;
torch on cuda; wall 12 s (143000, run first and opened to check it was populated) + 216 s (the
other 17). Tables: `python -m trackB_circuit.b7_show <dir>` (descriptions, P, P − N1, N0, entering
heads, by step); `... cells` or `... step <S>` for every cell's counts, intervals and beside columns.
Re-run: `METS_DATA=<main tree>/data python -m trackB_circuit.b7_activity read --gate <gate.json> --out <dir>`.
The design's exit (the cell table over 23 × 18, each cell with its counts, intervals and input, after
a passing gate) is met; **the 4 units are spent**, the close-out page is next.

Codes: · not read, X positional not separable, R rerouting, L depth-local persistence, F frozen
across depth (`P − N1`'s interval contains 0), ² two-cycle candidate. No cell reads "persistent, less
than at distant layers". Steps 0–256 are all · (no condensed row anywhere) and are left out; rows
with the same sequence are merged.

| l → l + 1 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | · | · | · | · | · | · | X | X | X |
| 1 | · | · | L | L | L | L | L | L | X |
| 2 | · | · | L | L | L | L | L | L | L |
| 3 | R² | R | X | X | X | X | X | X | X |
| 4 | F | X | X | X | X | X | X | X | X |
| 5 | F | R | R² | R² | R² | R² | L | L | L |
| 6 | · | · | F | L | L | F | F | F | F |
| 7, 8, 10, 16 | · | L | L | L | L | L | L | L | L |
| 9 | · | L | L | L | L | L | F | F | F |
| 11, 12, 14, 19 | · | · | L | L | L | L | L | L | L |
| 13 | · | · | L | L | F | F | F | F | F |
| 15 | · | · | L | L | L | F | L | L | L |
| 17 | · | · | L | L | F | F | F | L | L |
| 18 | · | F | L | L | L | L | L | L | L |
| 20 | · | L | F | L | L | L | F | L | F |
| 21 | F | F | F | F | F | F | F | F | F |
| 22 | F | F | F | L | F | F | L | L | F |

| step | read | X | R | L | F | condensed share (layer median) | D_1 median | α median |
|---|---|---|---|---|---|---|---|---|
| 0–256 | 0 | | | | | 0.00 | 0.22–0.30 | 0.99–1.00 |
| 512 | 5 | 0 | 1 | 0 | 4 | 0.01 | 0.21 | 0.95 |
| 1000 | 12 | 1 | 2 | 6 | 3 | 0.03 | 0.23 | 0.84 |
| 2000 | 22 | 2 | 1 | 15 | 4 | 0.11 | 0.27 | 0.73 |
| 4000 | 22 | 2 | 1 | 18 | 1 | 0.17 | 0.30 | 0.60 |
| 8000 | 22 | 2 | 1 | 15 | 4 | 0.14 | 0.32 | 0.64 |
| 16000 | 22 | 2 | 1 | 13 | 6 | 0.17 | 0.33 | 0.58 |
| 32000 | 23 | 3 | 0 | 14 | 6 | 0.19 | 0.33 | 0.50 |
| 54000 | 23 | 3 | 0 | 16 | 4 | 0.20 | 0.34 | 0.45 |
| 143000 | 23 | 4 | 0 | 13 | 6 | 0.20 | 0.43 | 0.37 |

**What the grid describes** (tier 1; cells, not heads; condensed rows only; `design-B7.md` "Fences"):
1. **Nothing to read to step 256**: no row is condensed at any layer, as B2a's near-uniform attention
   predicts. At 512, 5 cells; from 2000, 22–23 of 23.
2. **Mid-depth is condensed and persistent, not rerouting.** Of the 110 L cells, 89 are in cells
   7–20, from 1000–2000 on: there P 0.46–0.95 (median 0.86), N0 0.005–0.19 (≤ 0.05 in 67), `P − N1`
   0.07–0.52 (median 0.27). A condensed row's partner at l is kept at l + 1 by the best-matched
   head well above position alone and above distant layers.
3. **Rerouting is rare and early**: 7 cells, all in 3 → 4 (512–1000) and 5 → 6 (1000–16000); 5 of
   them are two-cycle candidates (5 → 6 at 2000–16000, 3 → 4 at 512), whose partners come back at
   l + 2. From 32000 no cell reads rerouting.
4. **Layers 0, 3, 4 (and 1 at 143000) are "positional, not separable"** (N0 0.53–0.83): offsets
   alone predict most of the agreement, as B2a's previous-token heads from 512.
5. **The last layer pairs are condensed but not depth-local.** 21 → 22 reads F at every read step, 22
   → 23 at 6 of 9, 20 → 21 at 3; at 143000 these layers have the highest condensed share (0.43–0.78)
   and the lowest α (0.03–0.09), while P falls to 0.86 / 0.50 / 0.46 and `P − N1` sits at
   −0.11 to +0.21. Cells 6, 9, 13 (and 15, 17 at some steps) also read F late, with `P − N1` points
   0.03–0.30 whose intervals reach 0 (caveat 1).
6. **Beside, over training:** `D_1`'s median falls from 0.30 (step 0) to 0.21 (512) and rises to 0.43
   (143000) as α's median falls from 1.00 to 0.37; the soft overlap of read cells rises from 0.28 to
   0.68 (medians). The pooled layer row reads in 0 cells to 512 and 21 at 143000. **Neighbour
   persistence does not separate anything:** above its N1 in every cell at every step, step 0
   included (median 0.87 against 0.69), as the design feared (shared large directions); not read.

**Caveats:**
1. **The interval's selection tail** (found reading the grid; scratch, the read's own counts, draws
   and seed, no new instrument). Each draw re-chooses h\* for a fold on that draw's other prompts
   (the design: "the folds and h\* re-done in each draw"), about 3.6 distinct on average against the
   sample's 6. Where the best match differs by prompt, P's draws fall with fewer distinct prompts
   (32000, 20 → 21: mean 0.952 at 7 distinct, 0.911 at 6, 0.794 at 4, 0.720 at 3; interval
   0.353–0.960); where it does not they barely move (10 → 11: 0.884 at 7, 0.857 at 3). N1 falls
   too, so `P − N1`'s median shift is about 0 (F +0.004, L +0.001), but its lower tail is long:
   **on draws with ≥ 5 distinct prompts, 5 of the 38 F cells and 2 of the 7 R cells would not keep
   their description; all 110 L cells would.** The gate could not see this: G1–G8 plant 16
   identical heads, so re-choosing h\* had no variance there. Not changed (the design fixes it; the
   budget is spent): F and R cells near that boundary read "not resolved from distant layers / from
   position". Which cells: not listed, so the count is not a second reading. Parked below.
2. **F is an interval containing 0**, not a shown equality: "not more persistent to the next layer
   than to distant ones, at this width".
3. **Thin heads:** 39 read cells have an entering head with < 50 leave-one-out rows (fewest 4,
   1000's 18 → 19; 9, 4000's 5 → 6); unit 1c's rule allows it, and each head's rows are in the
   record's `heads`.
4. **Entry rule:** unit 1b's leave-one-out entry would count a different number of heads in
   4 / 6 / 13 / 6 / 4 / 3 / 3 / 1 / 6 cells at 512 … 143000 (`n_enter_loo_rule`, beside; #274
   finding 2). Whether any description would change was not computed.

## B7 unit 1c (2026-10-25): the re-gate passes (Blocked 112 (a))

**Input:** as unit 1b (A1a's 18 run dirs, `prompts_hash` `976b62d677bc`, real sinks at 512, 8000,
143000, real attention at 143000 for G8, B3's rows at code `2c13479d759c`, the 252 input files'
sha256 in the record). Code `trackB_circuit/b7_activity.py` `a1de85990c74` (choices 18–19 in its
docstring), after `/challenge-pr` on #274; record `data/b7/2026-10-25-unit1c-r2/gate.json` (main
tree); torch on cuda; wall 442 s. The first run (`a39756ddfea4`, `2026-10-25-unit1c/`) gave the same
table cell for cell.
Re-run: `METS_DATA=<main tree>/data python -m trackB_circuit.b7_activity gate --out <dir>`. Code
tests `tests/test_b7_activity.py` (22: a head with no leave-one-out rows does not enter (fails on
`a39756ddfea4`); #273's strict-xfail test for G1 at the minimum entry with
sinks that differ by head now passes and is a plain test; entry against a brute force: the
all-prompt h\*, its rows, who enters at five floors, and P still leave-one-out).

**The change (Blocked 112 (a), the user's).** A head enters on its jointly condensed rows with h\*
chosen on all 7 prompts (≥ 50), for entry only; P, N0, N1 and every interval stay leave-one-out, so
a head's weight in P is its leave-one-out rows. Each head's leave-one-out rows, entry rows and entry
h\* are written in the cell's `heads`, and the minimum and median over the entering heads beside the
cell (choice 18). G3 prints the pooled r beside (choice 19, #273 finding 4). `design-B7.md` "Revised
at unit 1c"; budget 4 units + close-out.

| arm | result (per sink step 512 / 8000 / 143000) | pass |
|---|---|---|
| G0 | as unit 1b | yes |
| G1 frozen, full rows | P = 1, N0 0.008–0.009; 23 of 23 "frozen" each | yes |
| G1 frozen, minimum entry | **23 of 23 read and "frozen" at each sink step** (unit 1b: 23 / 7 / 5); P = 1, N0 0–0.026 / 0–0.059 / 0–0.039; fewest leave-one-out rows of an entering head 50 / 36 / **12**; a cell's joint rows 800 / 577–800 / 276–800 | yes |
| G2 reshuffled | 23 of 23 "rerouting" each; pooled (P − N0)/N0 −5.2 / −2.4 / −2.9 % (band 25 %); per-cell extreme beside 0.17 / 0.18 / 0.24 | yes |
| G3 drift, full rows | r 0.797–0.805; pooled r 0.800 / 0.800 / 0.800; 23 of 23 "depth-local" each | yes |
| G3 drift, minimum entry | r 0.774–0.836 (band ± 0.057); pooled r 0.801 / 0.804 / 0.799; 23 of 23 "depth-local" each | yes |
| G4, G5, G6, G9 | as unit 1 | yes |
| G7 false persistence | 0 of 1380 at full rows; **1 of 1380 (0.07 %) at the minimum entry** (unit 1b: 0), none unread; bar 5 %; fewest leave-one-out rows of an entering head 407 / 10 | yes |
| G8 planted on real | 10 → 11 "depth-local", planted heads P = 1, h\* `L11H3`; other cells' N1 moves at most 0.0063, no description changed | yes |

**What the change costs, as #273's review measured it on scratch:** at the minimum entry under
143000's sinks an entering head can carry 12 leave-one-out rows into P (here 12, as predicted), so
a cell's P can lean on heads thinner than the frozen 50. On real data the printed rows say where.
G7's one false cell at the minimum entry is new under the wider entry; at 0.07 % against a 5 % bar it
is beside, not a failure. No real cell read; unit 2 is next.

**After `/challenge-pr` on #274 (accept with changes).** Finding 1, confirmed: a head whose
all-prompt rows reach 50 but whose leave-one-out rows are 0 entered and counted toward the 2-head
rule, so a cell could be read on one head's data. A head now also needs at least one leave-one-out
row; a code test fails on `a39756ddfea4` and passes now; re-gated (`a1de85990c74`), identical, so the
guard never fires on the planted arms. Finding 2: the cause is named in the design (the gate's
thinning to exactly 50 rows a pair, with identical heads; G1's new pass sits at the floor), and every
cell now prints unit 1b's entry count beside (`n_enter_loo_rule`), so unit 2 shows where the two rules
disagree on real data. Finding 3: the design says N0 and P's weights use the leave-one-out h\* (as the
code always did). Finding 4: G7 records its fewest leave-one-out rows (10 at the minimum entry); thin
heads with P below 1 are seen only there, since G3's minimum-entry cells keep 800 rows. Answers: PR #274.

## B7 unit 1b (2026-10-25): the re-gate fails on G1 at the minimum entry (Blocked 111 (a))

**Input:** as unit 1 (A1a's 18 run dirs, `prompts_hash` `976b62d677bc`, real sinks at 512, 8000,
143000, real attention at 143000 for G8, B3's rows at code `2c13479d759c`); the 252 input files'
sha256 now in the record. Code `trackB_circuit/b7_activity.py` `7f693d7d2924` (choices 11–17 in
its docstring); record `data/b7/2026-10-25-unit1b/gate.json` (main tree); torch on cuda. Wall 435 s
(G0, reading and hashing 252 files: 184 s). Re-run:
`METS_DATA=<main tree>/data python -m trackB_circuit.b7_activity gate --out <dir>`. Code tests
`tests/test_b7_activity.py` (19: N0 against a brute force over different rows, with undefined rows;
N0 under G1 against G2 at 50 rows, which unit 1's N0 fails (0.091 against 0.028); a non-finite
interval not read; G1, G3 at the minimum entry and G2 pooled on the small cloud; the read path's
beside columns written as JSON with NaN as null; `read` refusing a gate on other code, prompts or
runs). Math check 7: at 50 rows over the 7 prompts, N0 with self-pairs reads 0.306 frozen against
0.0084 reshuffled. Over different rows it reads 0.0088 against 0.0083.

| arm | result (per sink step 512 / 8000 / 143000) | pass |
|---|---|---|
| G0 | 252 files hashed, shapes, row sums, unit norms, B3's rows, `prompts_hash` | yes |
| G1 frozen, full rows | P = 1, N1 = 1, N0 0.008–0.009 (unit 1: 0.014–0.019); 23 of 23 "frozen" each | yes |
| G1 frozen, minimum entry | read cells P = 1, N0 0–0.044, "frozen"; cells read **23 / 7 / 5** of 23, the rest "not read: 0 entering heads" | **no** |
| G2 reshuffled | 23 of 23 "rerouting" each; pooled (P − N0)/N0 −5.2 / −2.4 / −2.9 % (band 25 %); per-cell extreme beside 0.17 / 0.18 / 0.24 | yes |
| G3 drift, full rows | r 0.797–0.805; 23 of 23 "depth-local" each | yes |
| G3 drift, minimum entry | r 0.774–0.836 at 800 rows a cell (band ± 0.057); 23 of 23 "depth-local" each; 4.3–4.8 % of rows without a pair, out of N0 | yes |
| G4, G5, G6, G9 | as unit 1 | yes |
| G7 false persistence | 0 of 1380 at full rows, 0 of 1380 at the minimum entry, none unread | yes |
| G8 planted on real | 10 → 11 "depth-local", planted heads P = 1, h\* `L11H3`; other cells' N1 moves at most 0.0071, no description changed; their \|ΔP\| = 0 is a code check | yes |

**Why G1 fails at the minimum entry** (diagnosed on scratch after the gate; no real cell). G1 plants one
map in all 16 heads, so every head of l + 1 agrees with (l, h) on every joint row. h\* is the argmax of
agreeing rows on the other 6 prompts. With each head pair thinned to 50 rows, it picks for each held-out
prompt the head with the *fewest* rows in that prompt. The matched pair's rows then sum to under 50, and
no head enters. At 512's sinks every row is in for every head, so all thinned sets coincide and the arm
passes. Under the 8000 and 143000 sinks the per-head masks differ. The failure is entry under exact ties.
N0's bias, the reason the arm was added, is fixed in every read cell. Tried on scratch: thinning to
55–75 rows leaves 4–18 of 23 cells unread at 143000; one sink row shared by a layer's heads leaves 1 of
23. **Entering on the joint rows with h\* chosen on all 7 prompts (P still leave-one-out) passes G1 and
G3 at full rows and at the minimum entry at all three sink steps, with G7 at the minimum entry 0 of 345
over 5 seeds.** That changes a frozen definition (design "per-head persistence P": "enters with ≥ 50
jointly condensed rows over the 7 folds"), and the reserve is spent: Blocked 112.

**After `/challenge-pr` on #273 (accept with changes).** Finding 1: the code test for G1 at the minimum
entry used one constant sink for every head, the input that hid the failure. A strict-xfail test
with sinks that differ by head now fails on entry under this code and passes under (a). Finding 2,
measured on scratch: under (a) an entering head can have as few as 12 leave-one-out rows (G1, 143000's
sinks; median 39; G7 27), and a second floor of 25 brings G1's failure back (2 of 23 unread). This is
in Blocked 112. Finding 3: N0's own sampling noise is not in the `P − N0` interval, since its profiles
are the sample's in every draw (choice 3). G7 guards the false side, nothing guards a real persistent
cell whose N0 reads high (up to 0.044 at 50 rows). Parked below. Finding 4 (the pooled r beside G3) goes
into unit 1c if 112 is (a). Answers: PR #273.

**Caveats.** (1) On real data the totals are not fixed by thinning, so the effect there is weaker, but
it can still lower a tied head's joint rows near the entry, and its weight. (2) G7 now has its
non-finite rule (choice 12) and still reads every cell, so the rule never fired on planted data. (3) G2's
per-cell extreme is still 0.24 at 143000, now beside, not a pass condition.

## B7 unit 1 (2026-10-25): the gate passes (Blocked 110 (a))

**Input:** `pythia-410m`, A1a's 18 run dirs (`trackA_force/a1a.py` `RUNS`), the 7 scored v1 prompts
(`prompts_hash` `976b62d677bc`), real sink masses `A_i0` at 512, 8000, 143000 for the planted arms,
real attention at 143000 for G8; B3's rows `data/b3/2026-10-18-unit1/rows` (code `2c13479d759c`).
Code `trackB_circuit/b7_activity.py` `8a57c494248a`; record `data/b7/2026-10-25-unit1/gate.json` (main
tree). GPU: attention cast to float64. Wall 298 s (G0's 126 file reads 126 s). Re-run:
`METS_DATA=<main tree>/data python -m trackB_circuit.b7_activity gate --out <dir>`; code tests
`tests/test_b7_activity.py` (12: N0 against check 6's direct loop, the leave-one-out match against
a brute force, neighbours, `D`, G1–G6 on a small cloud). Ten choices the design left open are in the
module docstring, made before any output; one corrects the design's text: offsets valid at i are
0..i − 1 (keys 1..i), since a self partner is a valid key.

| arm | result (per sink step 512 / 8000 / 143000 unless one number) | pass |
|---|---|---|
| G0 | 126 attention + 126 state files, shapes right, rows sum to 1 within 1.3e-6, states unit within 4.0e-7, B3's rows and hash present, `prompts_hash` pinned | yes |
| G1 frozen | P = 1, N1 = 1, N0 0.014–0.019; 69 of 69 cells "frozen across depth" | yes |
| G2 reshuffled | 69 of 69 "rerouting"; per-cell \|P − N0\|/N0 at most 0.176 / 0.177 / **0.244** (cap 0.25); pooled P − N0 −5.4 / −2.6 / −3.2 % of N0 | yes (near the cap, see below) |
| G3 drift | r = (P − N0)/(1 − N0) in 0.795–0.804 (0.8 ± 0.03); N1 0.15–0.28; 69 of 69 "depth-local persistence" | yes |
| G4 positional | N0 = 1; 69 of 69 "positional, not separable" | yes |
| G5 period two | lag 1: 69 of 69 "rerouting"; every lag-2 cell `P_2` = 1 and a two-cycle candidate | yes |
| G6 diffuse | no row condensed; 69 of 69 "not read", no P | yes |
| G7 false persistence | 0 of 1380 read cells not "rerouting" at full rows and 0 of 1380 at the minimum entry (20 seeds × 3 sink steps × 23; cap 5 %) | yes |
| G8 planted on real | cell 10 → 11 "depth-local persistence"; the 4 planted heads P = 1 (1476–2784 joint rows), h\* = `L11H3` for all 4 (a planted head); the 20 cells not touching layers 10–11 \|ΔP\| = 0, same cells read | yes |
| G9 D | same states D_1 = D_2 = 0; antipodal flips D_1 = 2, D_2 = 0 (error 0) | yes |

**After `/challenge-pr` on #272 (accept with changes): the frozen N0 is biased under persistence.** It builds
both offset profiles from the same rows, so every row's own agreeing pair is in Σ p q, adding ≈ P/m (m rows
in the bin). The gate planted persistence only at full rows, where it is small (G1's N0 0.015 against
chance 0.009). At the 50-row entry (scratch, seed 3, 512 and 143000's sinks) N0 is 0.27–0.34 under G1
and 0.21–0.27 under G3, whose r reads 0.70–0.79 (fails 0.8 ± 0.03). Leaving out only row i's own pair
does not remove it (N0 0.27–0.49). A U-statistic over ordered pairs of different rows, both offsets
valid at i, is exact against a brute force and gives N0 0.0089 at full rows under G1, G3 and G2 (chance);
at the minimum entry, G3's r is 0.77–0.84 (sampling at ≈ 800 rows a cell), and about 5 % of rows have no
other row in their bin. Changing N0 is a frozen definition: Blocked 111. Findings 2 (uncounted dropped
draws; an all-NaN interval read as "rerouting"), 3 (G2 per cell), 4 (the read path's beside columns
untested), 5 (G8's |ΔP| cannot fail) and 6 (input hashes, the gate's prompts and runs not checked by
`read`) are accepted for unit 1b, since any code change re-binds the gate. Answers: PR #272.

**Caveats.** (1) G2's per-cell band passed by seed, not by margin: a scratch run before the gate
(143000's sinks, another seed) put one cell at 0.287. Per cell, the band also counts the cell's own
sampling noise, not only bias; the pooled deviation (−3 to −5 %) is the bias. So N0 errs above P
by a few per cent under the null, as check 6 said (+5 %); it leans to "rerouting". (2) G7's 0 of
2760 says the descriptions are conservative under the null; power is shown only by G3 (r
recovered within 0.006) and G8 (4 planted heads among 12 real). (3) G5 at 512 reads P 0.0115
against N0 0.0087 and still "rerouting": with all 16 heads of a layer identical, a cell is one
comparison and its interval is wide. (4) G8 computed the unplanted values at 143000 and kept only
differences; no real cell value is written or was seen. (5) Not built here: the soft overlap, pooled
row, neighbour and `D` columns run only in `read` (G9 tests `D`; code tests the neighbours).

## B3 on one page (closed 2026-10-19)

**Asked:** (R) how many partners does each head attend to, how does that grow with context, and
does a member's routing sharpen when it forms; (L) nudge token j on repeated random tokens: does
the response reach j's induction partner more than its neighbours, and does it go through the
members? **Exit met** (`design-B3.md` "Exit condition and budget"): R1, R2 over the 18 steps,
C3′'s rank, L0 at 143000 with its steps, L1's rank, A3's row and β's columns, in 3 units on the
record (Blocked 65 (a)) plus unit 2b (Blocked 69 (c)).

**Established (tier 1, 410m, nothing registered; R on v1's 7 prompts, `b3_route.py`
`921be18a1272`; L on 32 repeated random sequences, `b3_cone.py` `bcfc2564e89a`, nudged at block 4,
c = 0.03, GPU float32):**

1. **The light cone runs along induction partners from step 1000.** At 143000 the partner (j + 63)
   responds about 11× its six neighbours (L0, ℓ 2.375 [2.31, 2.43]). ℓ is about 0 to 128, 0.03 at
   512, and 2.96 at 1000, the formation step of `L8H6` and `L11H14`. Control sequences (same
   distance, no partner) contain 0 at every step. The partners come from the circuit, not from
   representation space, so this says the response follows routing, not that 410m is in the
   paper's regime (design "Fences"). At 143000 ℓ's size is partly a float32 detection limit (the
   neighbours' σ sits near the floor); ℓ at 32000, with no σ near it, is 2.40 ("B3 unit 2").
2. **The cone goes through the members, in part, and half of that is suppression.** Ablating
   `L7H8`, `L8H6`, `L11H14` removes 0.54 of ℓ's 2.37, above all 200 random layer-matched 3-sets
   (L1, rank 1.000). The cone stays at 1.84 without them. Of the 0.54, the partner's log falls
   0.24 and the neighbours' rises 0.30, each beyond all 200 null sets: the members both carry the
   response to the partner and keep it off nearby tokens. The three nearest-type non-members
   remove 0.05 (0.88, not shown; not layer-matched).
3. **The partner's response is added across depth, not at one block.** Its σ is 8 % of its final
   size after block 6, 0.48 by `resid_pre` 12 (blocks 6–8 and 10–11; block 9 adds nothing, block
   11 the most, 0.18), flat through block 15, and the other half in blocks 16–23. ℓ ≈ 2.0 after block 6 is a ratio on a small signal:
   `L6H3` makes it, and the final response does not depend on `L6H3`. `L6H0` keeps the response
   off the neighbours (ablated, they rise; Δℓ 0.18). Unit 2's "most of the cone is in place after
   block 6" was wrong and is corrected ("B3 unit 2b").
4. **The members' routing shows no difference from their layers'** (not shown to differ; blind to
   members held mid-population, where `L8H6` and `L11H14` sit). R1 (α, the slope of log N_eff on
   log n) ranks 0.80. R2: only `L5H2` of 4 condenses at its formation step (onset 512, α 0.98 →
   0.13), rank 0.234; weak evidence, since only 4 of 4 can read and 11 of layer 5's other 15
   heads also hit at that step. The population sharpens anyway: median α 0.95 at 512, 0.35 at
   143000, 98.7 % of heads below step 0's band.
5. **B2's QK-symmetry split is not shown to go beyond head type.** Re-ranked against the 5
   non-members nearest in type (C3′), it reads 0.945 (layer-matched C3: 0.9996). Only two of the
   six pools are close in type, so this is "not shown", not "type, not circuit".
6. **Descriptive, beside the family.** β (`fe_full`, prose) does not track sharpness: ρ(α, β) =
   0.18 over 384 heads at 143000, and the sharpest member, `L5H2`, has β −0.15. A3: received
   attention on c3x members against the rest has Spearman 0.08 with 7d's solo effect.
7. **The instruments.** Unit 1's gate failed on three defects and passed after Blocked 66 (a)'s
   revisions (each member at its own offset, `L12H5` untyped, a step-0 floor on R2's onset,
   inclusive null pools). The offsets corrected 7d: `L8H6` and `L11H14` are ordinary induction
   heads at +1, and `L12H5` is not an induction head on 7d's probe (`p7d_redundancy/status-7d.md`
   "Corrections received"). L's gate passes, but its G2 cannot fail and G6 is weak by construction
   ("B3 unit 2").

**Not established:**

- How much of the members' 0.54, and of the 1.84 left without them, is a direct path and how much
  self-repair (no path patching).
- Which heads add the partner's response (blocks 6–8, 10–11 and 16–23; outside block 6 none is
  named), and whether the neighbours' rise under ablation is direct or the downstream heads'
  response.
- That L1 marks this circuit rather than its head type: the same-type set sits in layers 8–10, not
  layer-matched, so "the cone goes through the members" is not shown to be more than "through
  induction-type heads at these layers".
- A routing difference for members held mid-population; R2 reads only at 4 of 4.
- Whether C3 marks the circuit or head type: `L5H2` and `L7H8` have no close type pool among the
  384 heads.
- Any per-head β against a condensation threshold: none is derived for a causal, rotary, untied
  head (design "Worth challenging" 1).
- Other rungs, prompts, nudge sites (block 6, after `L5H2`), and unit 2b at any step but 143000.

**Open threads, at close** (B3 had no Parked items; these are its units' "not checked" lines):

| thread | at close |
|---|---|
| Split L1's 0.54 and the 1.84 into direct path and self-repair, by path patching through the three members (Blocked 69 (b)) | **carried** to B4 (`docs/ROADMAP.md` §7): it nudges the circuit's spaces, which is B4's kind of question |
| Which heads add the partner's response (blocks 6–8, 10–11, 16–23), and whether the neighbours' rise is direct | **carried** to B4, with unit 2's per-block partner σ (`sweep/step143000.npz` `layers`) as its baseline |
| A nudge site after `L5H2` (block 6) | **carried** to B4, which nudges `L5H2`'s write space |
| A layer-matched same-type set for L1 (circuit or head type) | **carried** to B4 (`/challenge-pr` on #230, finding 3): one more configuration on unit 2's cached run, about 2 GPU minutes, where layer-7 and -8 candidates exist (C3′'s pools name none in layer 11) |
| `L6H0` against a 1-head null; unit 2b at other steps | **dropped**: descriptive, outside the family, so it can change no reading |
| C3′'s shared pool for `L8H6` and `L11H14` (a null draw can count one head twice) | **dropped**: G2's fakes share it and pass, and C3′ reads "not shown" at 0.945, where 8.2 % of fakes reach 0.95 |
| A type pool close for `L5H2` and `L7H8` | **dropped**: none exists among 384 heads (pools 0.37–0.70 and 0.14–0.61 against 0.89 and 0.67) |
| Carried in: A3's row (Phase 10), β's estimator (A2), C3's head-type control (B2) | **done**: A3's row and β's two estimators are beside (no threshold read, A2's disagreement kept); the head-type control is C3′ |

## B3 unit 2b (2026-10-19): layer 6's heads, one at a time (Blocked 69 (c))

**Input:** unit 2's (below): the 32 repeated sequences, 16 j each, K = 32, c = 0.03, step 143000,
`means.npz` (repeated), L1's truncation (tokens < j + 68); cached from block 6. Configurations:
none, each of `L6H0`–`L6H15` mean-ablated alone, and `attn6` (all 16 together, beside). Read at the
final residual (floored, as L1) and at `resid_pre` 7 (block 6's output, unfloored, as the sweep's
by-layer row). Code `b3_cone_l6.py` (the run's `code_sha()` in `l6/meta.json`; the reads below in
`l6.json` carry the later reader's, which added `parts` and `by_layer_parts` and left the run path
unchanged), on `b3_cone.py` `bcfc2564e89a`, unchanged, so unit 2's gate still applies. GPU float32, B0's NUMERICS; 36 min.
Records `data/b3/2026-10-18-unit2/l6/`, `l6.json`, `run_l6.sh`, `run_l6.log` (main tree). Re-run:
`run_l6.sh` (`python -m trackB_circuit.b3_cone_l6 run|read --out <unit 2 dir>`). ℓ_none here equals
the null pass's exactly (difference 0.0); at `resid_pre` 7 it is 2.027, as in the sweep.

**Fixed before any output** (module docstring): Δℓ at both reads per configuration; the final Δℓ
ranked against L1's null of 200 random 3-sets. That is a 3-head null for a 1-head ablation, so a
rank there means "more than three random later heads remove", not a family reading.

| ablated | Δℓ final | share of 2.373 | rank vs L1's null | Δℓ at `resid_pre` 7 | share of 2.027 |
|---|---|---|---|---|---|
| `L6H0` | 0.177 | 0.07 | 1.000 | −0.209 | −0.10 |
| all 16 (`attn6`) | 0.136 | 0.06 | 0.995 | 1.909 | 0.94 |
| `L6H12` | 0.033 | 0.01 | 0.805 | 0.011 | 0.01 |
| `L6H3` | −0.024 | −0.01 | 0.340 | **2.207** | **1.09** |
| the other 13 | −0.047 to 0.013 | | 0.155–0.665 | −0.057 to 0.005 | |

**ℓ's two sides** (added after `/challenge-pr` on the 2b commit, findings 2–3; `b3_cone_l6.parts`,
the same records re-read: `l6.json`, `l6_read2.txt`). ℓ = log σ_partner − log median σ_neighbours,
so a Δℓ can come from either side; σ is ‖A‖/ε of the un-normalised residual, a running sum, so
σ at `resid_pre` 7 and at the final residual are on one scale. Unablated: the partner's σ is
**0.097 at `resid_pre` 7 against 1.171 at the final residual (0.08)**; the neighbours' 0.0116 and
0.110. Drops (unablated − ablated; negative means it rose):

| ablated | Δ log partner, final | Δ log neighbours, final | σ partner at `resid_pre` 7 |
|---|---|---|---|
| `L6H0` | −0.016 | −0.194 | 0.0970 |
| all 16 | −0.002 | −0.131 | 0.0062 |
| `L6H3` | −0.018 | +0.002 | 0.0086 |
| the other 13 | −0.007 to +0.008 | −0.034 to +0.041 | 0.0968–0.0970 |
| L1's members (`l1/`) | **+0.242** | **−0.297** | — |
| L1's same-type set | −0.030 | −0.092 | — |

Against L1's 200 null sets read the same way (`null/`; after `/challenge-pr` on #229, finding 3):
the members' partner side is above all 200 (null max 0.111), their neighbour side below all 200
(null min −0.184).

**The partner's σ by block, unablated** (unit 2's `sweep/step143000.npz` `layers`; post hoc, after
`/challenge-pr` on #229, finding 1; `l6_read3.txt`). Share of its final 1.171 at each `resid_pre`:

| `resid_pre` | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13–16 | 17 | 18 | 19 | 20–22 | 23 | final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| partner's share | 0.00 | 0.01 | 0.08 | 0.12 | 0.20 | 0.20 | 0.30 | 0.48 | 0.45–0.48 | 0.54 | 0.63 | 0.76 | 0.75–0.80 | 0.89 | 1.00 |
| neighbours' σ | 0.004 | 0.006 | 0.012 | 0.023 | 0.036 | 0.053 | 0.061 | 0.063 | 0.058–0.062 | 0.063 | 0.071 | 0.082 | 0.085–0.091 | 0.097 | 0.110 |

The partner's response is added in steps: blocks 6–8 and 10–11 (0.48 by `resid_pre` 12; block 11,
`L11H14`'s, adds the most, 0.18), flat through block 15, then about half in blocks 16–23
(`resid_pre` b is block b − 1's output; first written "flat through block 16, then 17–23", one
block late, corrected after `/challenge-pr` on #230, finding 4).

Head types at 143000 (unit 1's `types/step143000.json`): `L6H3` ind1 0.461, ind0 0.009, prev 0.015
(an offset-+1 induction-type head below the typing cut); `L6H0` ind0 0.750 (offset 0), not a
member. In B2, `L6H3`'s keys read `L5H2`'s write space at about 9× chance (`c6_pairs.json`).

**Readings (descriptive):**

1. **The cone's size is added after block 6.** The partner's response at block 6's output is 8 %
   of its final size. ℓ ≈ 2.0 there is a high ratio on a small signal (the neighbours are
   smaller still). So unit 2's "most of the cone is in place after block 6" was wrong, and so was
   the reason Blocked 69 (c) was recommended (corrected in "B3 unit 2" below).
2. **That early ratio is `L6H3`'s, and the final response does not depend on it.** Ablating
   `L6H3` alone takes the partner's σ at `resid_pre` 7 from 0.097 to 0.009 (Δℓ there 2.21); the
   final partner and neighbours do not move (Δ log −0.018, +0.002). Nothing had to be rebuilt:
   what `L6H3` writes is a small part of a running sum that later blocks add to. No other single
   layer-6 head moves σ at `resid_pre` 7.
3. **`L6H0`'s Δℓ of 0.18 is the neighbours rising, not the partner falling.** Ablated, the partner
   is unchanged (Δ log −0.016) and the neighbours rise 0.19. So `L6H0` keeps the response off the
   partner's neighbours; it is not a route to the partner. Its rank 1.000 is against 3-head sets
   from layers 7, 8, 11; within layer 6 it is the largest of 16, and the 16 do not add (all
   together 0.136, below `L6H0` alone).
4. **L1 splits in two** (beside, the same records). Of the members' Δℓ 0.536, the partner's log
   falls 0.242 and the neighbours' rises 0.297 (medians, so the two need not sum to Δℓ). L1's rank
   stands, since its null is read the same way, but about half of "the cone goes through the
   members" is the members keeping the response off nearby tokens.

**Not established:** which heads add the partner's response after block 6. It is added in
blocks 6–8 and 10–11 (to 0.48 of its final size, the members' layers among them) and 16–23 (the
other half); the members take 0.24 of its log; no head in 12–23 is named; whether the neighbours' rise under ablation is a direct
effect or the downstream heads' response to it; `L6H0` against a 1-head null; any step but 143000.

## B3 unit 2 (2026-10-19): the light cone (Blocked 67 (a) with the same-type arm)

**Input:** 32 sequences of 64 uniform token ids in [1000, 30000) repeated twice (n = 128, no BOS)
and their controls, 16 nudge positions j ∈ [8, 56) and K = 32 directions per sequence, all seeded
in `b3_cone.py` (`tokens`, `positions`, `directions`); `resid_pre` of block 4, c = 0.03 (0.01, 0.1
beside); the 18 steps; unit 1's types (`data/b3/2026-10-18-unit1/types/`) for the offsets and the
same-type set. Code `b3_cone.py` + `core/nudge.py` `bcfc2564e89a`. GPU float32 (RTX 3080), B0's
NUMERICS; about 14 GPU hours (the sweep 11.5 min per step, G4/G6 and L1's null the rest). Records
`data/b3/2026-10-18-unit2/` (main tree): `sweep/`, `gate/`, `null/`, `l1/`, `means.npz`,
`gate.json`, `result.json`, `report.txt`, `run.sh`, `run.log`.
Re-run: `data/b3/2026-10-18-unit2/run.sh`, then `python -m trackB_circuit.b3_cone read --out <dir>
--types <unit 1 dir>/types` (`METS_DATA=<main tree>/data`).

**As fixed by Blocked 67 (a):** partner i = j + 63 (offset +1: `L8H6` 0.97, `L11H14` 0.95 at
143000; `L7H8` is at offset 0, 0.95, so j + 64 is beside). L1 mean-ablates `L7H8`, `L8H6`,
`L11H14` (each head's output replaced by its mean over the 32 sequences' positions) against 200
seeded layer-matched non-member 3-sets. The same-type arm uses the 3 non-members nearest in type:
`L9H9` (for `L7H8`, 0.61 against 0.67), `L10H9` (for `L8H6`, 0.82 against 0.83), `L8H9` (for
`L11H14`, 0.78 against 0.83). Choices the design left open, made before any output (`b3_cone.py`
docstring): G6 is ranked against L1's own null; B1's rounding floor; causal truncation and shared
early blocks (exact for every token read).

**The gate** (run before the members' configuration):

| arm | result | pass |
|---|---|---|
| G2 (L1) | each null set against the other 199: 1.0 % at rank ≥ 0.99 (cap 2 %). **Cannot fail:** for 200 distinct values a leave-one-out rank is ≥ 0.99 for exactly the top 2 (`tests/test_b3_cone.py` asserts it), so it is a code check, not calibration (`/challenge-pr` on #228, finding 3) | yes (vacuous) |
| G4 | planted route (one layer-8 non-member, query j + d → key j at mass 1; controls, 143000): interval above 0 in 50 of 50 seeds; smallest ℓ 0.30 | yes |
| G5 | step 0: ℓ 0.001 [−0.001, 0.004]; controls at 143000: −0.013 [−0.047, 0.025] | yes |
| G6 | the planted head ablated: Δℓ ≥ 0.37 in every seed, against the null's 99th percentile of 0.107: 50 of 50 | yes |

G6 is weak by construction. One head at mass 1 is a large route, so G6 shows only that an
ablation removes a route and that the null's spread is small beside it.

**The readings** (143000, c = 0.03):

| row | value | reads |
|---|---|---|
| L0 (positive control) | ℓ 2.375 [2.313, 2.426]: the partner's σ is about 11× its six neighbours' | passes |
| L1 (family of four, 0.99) | ℓ 2.373 unablated (L1's own pass), 1.837 with the 3 members ablated: Δℓ 0.536. Null median −0.011, 99th 0.107, all 200 below 0.536 | **rank 1.000: the cone goes through the members** |
| same-type arm (beside) | Δℓ 0.053 with `L9H9`, `L10H9`, `L8H9` ablated | rank 0.880: not shown |

The members' ablation removes about a fifth of ℓ (0.54 of 2.37), and the cone stays far above 0
without them (1.84). So L1 says the cone goes through the members *in part*. **Corrected
2026-10-19 (unit 2b, above):** this paragraph first said the by-layer row puts most of the cone in
the residual after block 6, before any member runs. It does not: ℓ is a ratio, and the partner's
response at block 6's output is 8 % of its final size. Of the 0.54, the partner's side is 0.24 and
the neighbours' 0.30. Where the rest of the partner's response is added, and how much of the
1.84 is self-repair (7d), are not split. The three nearest-type heads remove a tenth as much,
but at rank 0.88 that is "not shown", and it is not evidence against induction-type heads in
general: all three sit after block 6 (layers 8–10, not layer-matched), so they cannot speak for
what carries the cone there (`/challenge-pr` on #228, findings 1–2).

**Beside (descriptive):**

- **ℓ over steps:** about 0 (|ℓ| ≤ 0.004) from 0 to 128; 0.030 [0.008, 0.048] at 512; **2.96 at
  1000**, then 2.66, 2.48 and 2.33–2.40 from 8000. The rule's onset (above 0, held at the next step)
  is **512**, but the cone itself appears at 1000. That is the formation step of `L8H6` and
  `L11H14`; `L7H8` forms at 4000.
- **j + 64 (offset 0):** 0.03–0.04 at 256–512, 1.51 at 1000, 1.15–1.23 from 2000; below j + 63.
- **Against the control token** (same i, same i − j, no partner): 2.95–3.34 from 1000. The controls'
  own matched ℓ contains 0 at every step from 1000.
- **Confinement share** (the partner over every token from j + 2): 0.006–0.009 to 512, 0.16 at
  1000, 0.13 at 143000.
- **By layer** (`resid_pre` of each block from 5, i.e. the output of the block before; first
  written one block late, corrected after `/challenge-pr` on #228, finding 1): the rule's "first
  block above 0" is block 5 at steps 0–8 and at every step from 512, so it also fires before
  anything is learned (finding 4). It comes from a small early response (ℓ 0.14–0.24 at
  `resid_pre` 5–6 from 1000; 0.07 at 512). The jump is in **block 6** at 143000 (`resid_pre` 6
  0.11 → 7 2.03) and at 4000 (0.10 → 1.27, then block 7 to 2.10), and in **block 7** at 1000
  (0.43 → 2.09). It moves one block earlier with training, and from 4000 it sits before
  `L7H8`'s layer. The fixed rule picks up the small lean; the jump is read here post hoc. The
  jump is in the ratio, not in the size: at 143000 the partner's σ at `resid_pre` 7 is 8 % of its
  final value (unit 2b).
- **c:** ℓ at c = 0.01 and 0.1 is within 0.04 of c = 0.03 at every step (linear).
- **Rounding:** partner and neighbour σ's within 10× the floor: none at every step to 32000, 400
  of 3,584 at 54000, 2,597 at 143000. They are mostly the neighbours: the final residual's floor
  doubles from 54000 to 143000 (median 0.0069 → 0.0155). At 143000 ℓ's denominator is close to
  float32 resolution, so ℓ's size there is partly a detection limit. Few are actually clipped to
  the floor: 42 of 3,072 neighbour values and no partner, counted as σ equal to the floor's value
  (`/challenge-pr` on #228, finding 6, counted 65 its own way). G5's intervals contain 0 on
  the same readout, L1 is a rank against ablations read the same way, and ℓ at 32000 (no σ near
  the floor) is 2.40, as at 143000.

**Not checked:** which heads carry the block-6 jump (done since, "B3 unit 2b" above); whether the 0.54 that the three heads carry is direct or partly self-repair
(no path patching); a layer-matched same-type set; `L12H5` (untyped, not ablated); other nudge sites
(block 6, after `L5H2`).

## B3 unit 1b (2026-10-18): the re-gate passes; R1, R2, C3′ read (Blocked 66 (a))

**Input:** as unit 1 (below): the same `rows/` and `types/` (`data/b3/2026-10-18-unit1/`, produced by
`b3_route.py` `2c13479d759c` (rows) and `b7f5307150cf` (types); the row and type code is unchanged
since), `prompts_hash` `976b62d677bc`, B2's table `038b04f5c7cd`. Reading code `b3_route.py`
`921be18a1272` with Blocked 66 (a)'s four revisions (its docstring "Revised after the gate failed"),
and after `/challenge-pr` on #225: each G2 fake's C3′ pool is built as its member's, no real member
sits in a fake's pool, and C3′'s false pass is also measured on 2,000 fakes.
Records `data/b3/2026-10-18-unit1b/` (main tree): `gate.json`, `result.json`, `report.txt`. CPU, 6 s.
Re-run: `METS_DATA=<main tree>/data python -m trackB_circuit.b3_route read --src <unit1 dir> --out <dir>`.

**The gate** (before any member value; R2's attainable rank takes each member as hitting):

| arm | result | pass |
|---|---|---|
| G0, G1 | as unit 1 (rows sum to 1 within 1.3e-6; types within 5.8e-6; uniform α 1.000, two partners 0, Gaussian 0.931 / 0.940) | yes |
| G2 | false pass (200 fakes, the arm): R1 0.5 %, R2 0 %, C3′ 0.5 % (cap 2 %). C3′ on 2,000 fakes beside: **1.5 %** at 0.99, 8.2 % at 0.95 | yes |
| G3 | planted onset, primary set (`L5H7`, `L7H4`, `L8H7`, `L11H5`): 4 of 4, rank 0.9996; all 4 in 78 % of 50 sets beside | yes |

Formation steps (each member at its own key): `L5H2` 1000 (previous-token), `L7H8` 4000 (offset 0),
`L8H6` 1000 and `L11H14` 1000 (+1). R2's floor (step 0's 2.5th percentile) α = 0.9908. Attainable
rank 0.9996 at 4 of 4; 0.986 at 3 of 4, so only 4 of 4 can read. C3′'s pools (5 non-members
nearest in mean type score over steps ≥ 512, plus the member): `L5H2` ← `L4H9 L3H1 L5H9 L9H5 L7H2`;
`L7H8` ← `L9H9 L1H15 L6H0 L2H10 L7H0`; `L8H6` and `L11H14` ← the same five, `L10H9 L8H9 L7H1 L10H0 L10H3`;
`L12H5`, `L10H7` ← their layers. How near in type the pools are (mean score over steps ≥ 512; each
typed member is first or second of the 384 by its own score): `L8H6` / `L11H14` 0.83 against
0.66–0.82, close; `L5H2` 0.89 against 0.37–0.70; `L7H8` 0.67 against 0.14–0.61; the untyped two
are layer-matched, as B2's C3.

**The family** (one family of four with L1, each at 0.99):

| row | statistic | rank | reads |
|---|---|---|---|
| R1 | T_α 0.314 (null median 0.281, 99th 0.366) | 0.802 | no difference shown |
| R2 | 1 of 4: `L5H2` (onset 512, formation 1000); `L7H8`'s onset is step 16 (α 0.990, just under the floor), far from 4000; `L8H6`, `L11H14` none | 0.234 | no difference shown |
| C3′ | T 0.394 (null median 0.345, 99th 0.412) | 0.945 | C3 marks head type, not shown to mark the circuit (the line fixed in the design) |

B2's layer-matched C3 beside: the same T, rank 0.9996. Against these type-nearest pools, C3 does
not set the members apart at 0.99, though it does against their layers. The fixed line's first half
says more than the rank shows. The rank shows only that C3 was **not shown to go beyond type**, and
only two of the six pools are close in type (above). The lean (0.945) is not a reading: 8.2 % of
fakes reach 0.95.

**Reach** (`/challenge-pr` on #225, finding 5): R2 can read only at 4 of 4. Planted onsets are all
caught in 78 % of sets beside G3. `L5H2`'s hit carries almost no evidence, because 11 of layer 5's other
15 heads also hit at its formation step. So R2's "no difference shown" is weak evidence.

**Beside (descriptive):**

- Population α: median 0.95 at 512, 0.84 at 1000, 0.35 at 143000; 98.7 % of heads below step 0's band at 143000.
- `L5H2` condenses at 512 (α 0.98 → 0.13; p 0.01) and stays near 0. `L12H5`, which is not an
  induction head on the probe, condenses later (0.73 at 1000, 0.08 at 4000, about 0 from 16000; p ≤ 0.09
  from 2000). `L8H6` and `L11H14` drop to about 0.4 at 1000 (p 0.13) and stay mid-population.
  `L7H8` falls gradually (0.92 at 512 to 0.15 at 143000). `L10H7` stays diffuse, α about 1.0 and p ≥ 0.97 from 2000 on.
- β (`fe_full`, prose median, 143000): `L5H2` −0.15, `L7H8` 15.6, `L8H6` 6.2, `L11H14` 5.4,
  `L12H5` 2.4, `L10H7` 9.8 (population 10th to 90th: 0.45 to 8.3). Spearman ρ(α, β) over the 384 heads at
  143000 is 0.18. A sharp head (low α) does not have a high β: `L5H2` is the clearest case.
- A3 (received attention on c3x members against the rest, log ratio, median over steps ≥ 512):
  members −0.42 (`L12H5`) to 0.05 (`L11H14`); rest 10th to 90th −0.39 to 0.07; Spearman with 7d's solo effect 0.08.

**Not checked:** C3′'s `L8H6` and `L11H14` share one pool, so a null draw can count one non-member
twice. G2's fakes are built the same way, and G2 passes, but the effect of the shared pool alone
was not measured. The R2 onset rule takes the *first* qualifying step, so for `L7H8` an early onset
at 16 passes the floor by 0.0008. It changes nothing here, because `L7H8` never enters its step's
bottom decile near 4000 (p ≥ 0.17 after 32).

## B3 unit 1 (2026-10-18): the gate fails (Blocked 66)

**Input:** `pythia-410m`, A1a's 18 run dirs (`trackA_force/a1a.py` `RUNS`), the 7 scored v1
prompts (`prompts_hash` `976b62d677bc`), 7d's probe (16 sequences, `wide`, `EVAL_SEED`),
B2's table (code `038b04f5c7cd`), Phase 10's c3x label source `data/p10/reread_r9_2026-10-08/labels`
(its own Stage 0 runs, below). Code `b3_route.py` `b7f5307150cf`, `b3_gate_variants.py` `79bc6cab77f4`
(types, gate and variants re-run after `/challenge-pr` on #224; the gate's numbers unchanged). Records
`data/b3/2026-10-18-unit1/` (main tree): `rows/`, `types/`, `gate.json`, `gate_variants.json`.
GPU: stored attention cast to float64; forwards float32. Wall: rows about 16 s a step, types 3 s.
Re-run: `python -m trackB_circuit.b3_route rows|types --out <dir>`, then `read --out <dir> --gate-only`,
then `python -m trackB_circuit.b3_gate_variants <dir>`.

| arm | result | pass |
|---|---|---|
| G0 | 126 attention files, shapes right, rows sum to 1 within 1.3e-6; head types reproduce `prev_token_profile.json` (all 384, at 4000 / 16000 / 143000) within 5.8e-6 and `fv_score.json`'s members within 3.9e-8 | yes |
| G1 | on `L13H5`'s real rows at 512 and 143000: uniform α = 1.000, two partners −1.6e-15, Gaussian logits s = 1 0.931 and 0.940 | yes |
| G2 | false pass (200 fake sets): R1 0.5 %, R2 0 %, **C3′ 6 %** (cap 2 %) | **no** |
| G3 | planted onset, primary set: **3 of 5 caught** (needs 5); all 5 in 44 % of 50 sets beside | **no** |

Every step has an α for every head (no head fell below 10 in-rows).

**Three defects, each found before any member α was read:**

1. **The members do not share one induction offset.** At 143000, `L7H8` puts 0.945 at offset 0
   (query `N_REP + j` → key j); `L8H6` (0.969) and `L11H14` (0.953) peak at +1 (→ key j + 1);
   `L12H5` scores 0.009 and 0.002, so it is **not an induction head on this probe at either offset**
   (peak over 18 steps 0.011; its sink share on the probe is 0.49, so renormalised it is still
   about 0.02; the four's sink shares are 0.42–0.49). The repo's offset 0 is the same-token key
   (`PROJECT.md` §3.18's Q1); +1, the successor, is the standard induction pattern. So `L8H6` and
   `L11H14` are ordinary induction heads, and 7d's "function-vector only" for them was read at the
   other offset: a correction to 7d (`p7d_redundancy/status-7d.md` "Corrections received"). The design took one offset for the four (the majority, here a 2–2
   tie, broken to 0), so the type score at offset 0 gives formation step 0 for `L8H6` and `L12H5`,
   which is meaningless. It also bears on unit 2: L's partner (j + 64 or j + 63) and L1's "four
   induction members".
2. **G3: early steps' bottom decile is init structure.** R2's onset ("at or below the step's 10th
   percentile, held at the next step") is met at step 2 by any head in the bottom decile at
   initialisation, since attention barely changes to 256. Planting step 0's α before the formation
   step therefore gives onset at step 2 for 2 of 5 planted heads.
3. **G2: C3′'s null was not exchangeable with its fakes.** C3′ drew a pool head per (member, step)
   independently. That breaks the across-step correlation a fixed head carries, and each fake's
   pool was re-centred on the fake. Re-centred pools with one fixed head per member still
   false-pass at K = 5, 10, 20 (4.5 %, 8.3 %, 3.3 %, 400 fakes each).

**The revisions, each run on the gate's inputs with no member α read** (`b3_gate_variants.py`, `gate_variants.json`):

| revision | what it gives |
|---|---|
| type score at each member's own offset (143000's peak) | formation steps `L5H2` 1000, `L7H8` 4000, `L8H6` 1000, `L11H14` 1000; `L12H5` untyped (as `L10H7`), so R2 counts 4 |
| R2's onset also requires α below step 0's 2.5th percentile (0.991) | null hit rates (others in the layer) 0.73 (layer 5), 0.07 (7), 0 (8), 0 (11); G3 catches all 4 in the primary set and in 78 % of 50 sets beside. **With the frozen exclusive pools its G2 false-passes 5.5 %** (200 fakes) |
| … and **each tested head counts in its own pool** (inclusive; `/challenge-pr` on #224, finding 2) | G2 false pass 0 %; if each member hits, the rates are 0.75, 0.125, 0.0625, 0.0625, and rank is 0.9996 at 4 of 4, **0.986 at 3 of 4 (does not read)**: only 4 of 4 reads "members differ" |
| C3′: one fixed head per member from a fixed pool of the K non-members nearest in mean type score over steps ≥ 512, fakes drawn from that pool | false pass with the fake left out of its pool: 3.0 % (K = 5), 0.75 % (K = 10). **With the fake kept in (inclusive): 0.0 % (K = 5), 1.0 % (K = 10)** (400 fakes); `L12H5` and `L10H7` layer-matched |

The cause common to C3′'s and R2's failures is the same: a null pool that leaves out the head being
tested is not exchangeable with it (`/challenge-pr` on #224 reproduced 3.1 % against 0.8 % on
idealised heads). Not yet checked: the revised arms inside `b3_route.py` itself rather than the
variants script (the re-gate does that).

**A3's input:** Phase 10's label source names its own Stage 0 runs (re-runs of 2026-09-22 to 09-24),
not A1a's, at every step ≥ 512 (63 of 63 directories differ). A3 reads the label source's runs, as
Phase 10 fixed them; α reads A1a's. About two-thirds of the (layer, prompt) cells at 512 are
readable in Phase 10's own sense. **M1's β is stored per layer only** (`a2_beta_2026-10-16`), so β
beside is `fe_full` per head with M1 per layer.

## B2 on one page (closed 2026-10-18)

**Asked:** do the six members (`L5H2`, `L7H8`, `L12H5`, `L8H6`, `L11H14`, `L10H7`) differ, in their
weights alone, from the other heads of their layers at the same step, on five frozen columns (OV
attractive share, OV's top signed weight, QK symmetric share, OV–QK cosine, K-composition among the
circuit's 5)? **Exit met** (`design-B2.md` "Exit condition and budget"): the per-member table over
the 18 steps and the five ranks, in the 3 units of Blocked 61 (gate, 9-step re-gate, table).

**Established (tier 1, 410m, weights only except C7, 18 steps, 384 heads per step; code `038b04f5c7cd`):**

1. **The circuit's K-composition is in the weights, and it is `L5H2`'s.** C6 ranks 1.0 (0.138,
   8.8× chance). `L5H2` writes into `L8H6`'s, `L7H8`'s and `L11H14`'s keys at 32×, 26×, 21× chance
   (each in the top 1 % of its layer pair; 91 % of the circuit's mean), not `L12H5`'s (1.2×). It
   appears between 512 and 1000, just after B1's onset in (256, 512]. The members were found on
   activations, so this agrees with how they were found; it is not an independent discovery.
2. **`L5H2`'s write space has readers outside the members** about as strong. Over every head after
   layer 5 (`/challenge-pr` on #222, finding 1), the 11 heads after layer 5 whose keys read it at ≥ 10× chance (`c6_pairs.json` `l5h2_readers_all`, code `b12ad273a201`): members `L8H6`, `L7H8`, `L11H14`; non-members in 7d's causal top 10 `L7H1`, `L8H9`, `L10H9`; non-members outside it `L7H7`, `L9H0` (27×), `L8H7`, `L10H0`, `L11H2`; then `L6H3`, `L10H4` at about 9×. The rank says the members' keys read it more
   than a random same-layer head, not that they are its only or strongest readers. `L9H0`, 5th of
   all and outside 7d's top 10, is the contrast B4 can use.
3. **QK symmetry sets the members apart, in both directions.** C3 ranks 0.9996: `L7H8` is its step's
   most symmetric head from 2000; `L8H6`, `L11H14`, `L10H7` near the 95th; `L5H2` and the induction
   member `L12H5` near the 15th. So the split is not "induction high" (`L12H5` low, `L10H7` high);
   some same-kind non-members sit with the members (below), so circuit and head type are both open.
4. **No OV sign sets the members apart** (C1 0.933, C2 0.783, C4 0.670: "no difference shown"),
   within the gated reach: sets held between their steps' 25th and 75th, or extreme only from 8000
   below about the 90th, cannot be seen ("B2 unit 1b").
5. **Descriptive, step 0 beside.** Most heads leave step 0's band at 256 (C1) to 4000 (C4); the
   negative-pair flag rises to 0.82 of heads at 512 and falls to 0.31 at 143000, members with the
   population; no head is in 2d's regime at any step; Table 1's "unclassified" rise at 1000 is
   mostly heads with no positive-real eigenvalue of V (237 of 384); the QK read rank is small only
   for `L5H2` (D = 1–4 from 1000 to 54000): on B2a's 14 heads, at the 0.1 placed after 143000 was seen,
   and read from B2a's attention on v1's 7 prompts (C7 is the one column that is not weights only). Every symmetrised map has inertia (64, 64), as `b2_symmetric_inertia.py` says.
6. **The instrument sees what it should.** Every gate arm passes at the design's constants (unit 1),
   and on the rule's own 9-step statistic (unit 1b: 100 % at the 90th, 42 % at the 80th).

**Not established:**

- What C3 marks: the circuit, or head kinds (`L5H9`, `L4H9` sit low with `L5H2`; `L8H9` high with
  `L7H8`, `L8H6`, `L11H14`; but the induction member `L12H5` sits low); the layer-matched null cannot tell.
- That composition is a path: C6 is a subspace overlap, not weighted by what is written or
  attended (B4's nudges are the functional test).
- That any weight-level column moves tokens (B2a, B3, B4); that a symmetric QK makes a gradient-flow
  head (2d needs V aligned too, and no head is).
- Members held mid-population, or extreme only late below the 90th, on C1, C2, C4.
- Other rungs, prompts, members (410m is exploration; members from activations; nothing registered).
- Phase 6's `U_pos` / `U_neg` / `U_A` split on the members' union, named in `docs/ROADMAP.md`'s B2
  row: not built (dropped below).

**Open threads, at close:**

| thread | at close |
|---|---|
| Phase 6's `U_pos` / `U_neg` / `U_A` dimensions of the members' union (design "Beside, not in the family") | **dropped**: beside, outside the family, so it can change no reading, and no later row asks for it (`docs/ROADMAP.md` §11 gives Phase 6 to B2 only; B4 nudges by composition, C6 by pair, which needs no sign split of the members) |
| What B4 nudges (C6 by pair, `c6_pairs.json`) | **carried** to B4 (`docs/ROADMAP.md` §7): `L5H2`'s write space and the 11 heads after layer 5 whose keys read it at ≥ 10× chance (`c6_pairs.json` `l5h2_readers_all`, code `b12ad273a201`): members `L8H6`, `L7H8`, `L11H14`; non-members in 7d's causal top 10 `L7H1`, `L8H9`, `L10H9`; non-members outside it `L7H7`, `L9H0` (27×), `L8H7`, `L10H0`, `L11H2`; then `L6H3`, `L10H4` at about 9× |
| A head-type control for C3 | **carried** to B3 (`docs/ROADMAP.md` §7), which joins per-head reads to 7d's causal sweep, where the head-type list lives: one re-rank of C3 with previous-token and induction non-members as the null |

## B2 unit 2 (2026-10-18): the table over the 18 steps (Blocked 63 (a))

**Input:** code `b2.code_sha()` = `038b04f5c7cd` (`b2.py table`), records
`/run/media/system/WDS_500/Mets/data/b2/2026-10-18-table/` (`table.json`; `step<S>.npz` per step:
every head's columns in both frames, C5, inertia, the composition tables `O8`, `O16`, `O32`, `WW16`;
log `../2026-10-18-table.log`). The 18 steps' weights from the HF cache, float64 CPU, 8 min.
Members, columns, rule and null as `design-B2.md` froze them; seeds `b2-table`. Blocked 63 (a), the
user's: the frozen T is ranked, S is not computed. C7 from B2a's records as G7 checked them.
Refusals passed before the rule was read: C1–C4 and the raw beside equal unit 1's populations at
0, 512, 143000 (max diff 0); the low-rank C5 gives the dense `classify_ov_row` row on B2a's 14
heads at 0 and 143000 (λ₁ to 9e-15; also `tests/test_b2.py`). Re-run: `METS_DATA=<data> python -m
trackB_circuit.b2 table --out <dir> --gate <data>/b2/2026-10-17-gate`. Every number below:
`python -m trackB_circuit.b2_show <dir>/table.json`.

**The five ranks** (the family; T over the 9 steps ≥ 512, 10,000 layer-matched sets; reads at ≥ 0.99):

| column | statistic | rank | null 50 / 99 % | reads |
|---|---|---|---|---|
| C1 OV attractive share | T 0.283 | 0.933 | 0.230 / 0.311 | no difference shown¹ |
| C2 signed top weight | T 0.272 | 0.783 | 0.245 / 0.330 | no difference shown¹ |
| C3 QK symmetric share | T 0.394 | **0.9996** | 0.270 / 0.360 | **members differ** |
| C4 OV–QK cosine | T 0.289 | 0.670 | 0.273 / 0.359 | no difference shown¹ |
| C6 K-composition, the circuit's 5 | 0.138 (8.8× chance) | **1.0** | 0.023 / 0.037 | **members differ** |

¹ As gated in unit 1b (design rule 4's note): this cannot exclude members held between their
step's 25th and 75th, nor members extreme only from 8000 below about the 90th.

Beside, not in the family (ranks, not readings): raw frame C1 0.929, C2 0.775, C4 0.671 (the
frames agree); C3 on the unrotated 48 dimensions 1.0; 2d's asymmetry 0.999; C6 at r = 8 and 32
both 1.0; C6 with `L10H7` added (6 heads) 1.0; write–write overlap 0.913.

**The members' percentiles** (each in its own step's 384 heads; at 143000, and the range over the
9 steps ≥ 512; values and every step in `b2_show`):

| member | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| `L5H2` | 0.61 (0.61–0.95) | 0.44 (0.09–0.59) | 0.17 (0.02–0.70) | 0.08 (0.07–0.15) |
| `L7H8` | 0.32 (0.06–0.65) | 0.03 (0.03–0.85) | **1.00** (0.19–1.00; 1.00 from 2000) | 0.75 (0.01–0.88) |
| `L12H5` | **0.94** (0.32–0.96) | 0.76 (0.00–0.81) | 0.14 (0.13–0.93) | 0.77 (0.74–0.98) |
| `L8H6` | 0.80 (0.52–0.86) | 0.02 (0.01–0.80) | **0.95** (0.02–0.98) | 0.87 (0.01–0.94) |
| `L11H14` | 0.55 (0.49–0.84) | 0.56 (0.02–0.85) | 0.94 (0.77–0.98) | 0.52 (0.08–0.83) |
| `L10H7` | 0.33 (0.07–0.42) | 0.34 (0.15–0.63) | **0.95** (0.73–0.96) | 0.49 (0.18–0.59) |

**C6 by pair** (descriptive, read after the rank; `b2_show`'s `pairs`, record `c6_pairs.json`,
code `f4c0d72e689e`; mean over steps ≥ 512, r = 16; percentile among the 256 head pairs between
the same two layers). The three `L5H2` rows carry 91 % of the circuit's mean.

| A writes → B's keys read | overlap | × chance | percentile |
|---|---|---|---|
| `L5H2` → `L8H6` | 0.502 | 32 | 1.00 |
| `L5H2` → `L7H8` | 0.404 | 26 | 0.99 |
| `L5H2` → `L11H14` | 0.333 | 21 | 1.00 |
| `L5H2` → `L12H5` | 0.018 | 1.2 | 0.36 |
| the other 6 circuit pairs | 0.007–0.029 | 0.4–1.8 | 0.01–0.84 |

**Who else reads `L5H2`'s write space** (same record; `/challenge-pr` on #221, finding 1): it is
not only the members. In layer 7, `L7H1` (0.501) and `L7H7` (0.475) read it more than `L7H8` (0.404,
3rd of 16). In layer 8, `L8H6` is 1st (0.502), then `L8H9` (0.433, 7d's other induction head) and
`L8H7` (0.340). In layer 11, `L11H14` is 1st (0.333), then `L11H2` (0.221). In layer 10, `L10H9`
(0.318) and `L10H0` (0.295) read it, and `L10H7` is 10th. In layer 12 no head does (≤ 0.044). *(These are the members' layers only; over every layer after 5, `L9H0` (0.418) is 5th and `L6H3` 9×: "B2 on one page" reading 2.)*

Per step, the circuit's C6 is at chance to 256 (0.0154–0.0156), 0.017 at 512, 0.074 at 1000,
0.127 at 2000, and 0.16–0.18 from 4000.

**Descriptive columns, step 0 beside:**
- **Moved by training.** To 64 about 5 % of heads are outside step 0's band (the band's own rate).
  Most of the population leaves it at 256 for C1 (94 %), 512 for C2 (100 %), 2000 for C3 (93 %) and
  4000 for C4 (91 %), as the design expected.
- **Negative-pair flag.** The population rate is 0.10 at step 0 and 0.19 at 128. It rises to 0.72
  at 256, peaks at 0.82 at 512 and falls to 0.31 at 143000. Five of six members are flagged at 256
  and all six at 1000. At 143000 `L5H2` and `L11H14` are flagged. Members follow the population's
  rise and fall.
- **2d's regime.** Every head is "outside" at every step (no head's QK is symmetric enough).
- **C5, Table 1's row.** At step 0 the population is about ¼ row 2, ¼ row 2 without the QK
  condition, and ½ unclassified. Unclassified rises to 0.81 at 1000 and falls to 0.47 at 143000,
  when row 2 is 0.34. Members move between row 2 and unclassified with no steady pattern. No head
  is row 1 or row 4.
- **Most of that rise is heads with no eigenvalue of V with positive real part** (`/challenge-pr`
  on #221, finding 4; `trackB_circuit/b2_c5_negative.py`, record `c5_negative.json`, code
  `8638f7224bc5`). No head at 0–512, then 237 of 384 at 1000, 251 at 2000, 167 at 4000, and down to
  43 at 143000 (the raw frame within 1). Members among them: `L7H8`, `L12H5`, `L8H6`, `L10H7` at
  1000; `L10H7` until 8000. There `c5_row` reads "unclassified" (no positive top eigenvalue). The
  dense `classify_ov_row` would decide on rounding noise among the 960 zeros, so the two can differ
  only where the dense answer is undefined. Unit 2's dense check covered B2a's 14 heads only.
- **C7, the QK read rank.** `L5H2` needs D = 1–4 from 1000 to 54000 and 16 at 143000. Every other
  head of the 14 needs the full 128 at every step, except `L7H8` at 64 at 8000–16000. Only the
  positional head reaches 0.1 · TV(0) on a small subspace (B2a unit 3's reading, from the QK side).
- **Inertia.** (64, 64) for every head's sym V and sym M_0 at every step (fact 1).

**What it says.**
- **The circuit's K-composition is in the weights, and it is `L5H2`'s.** The previous-token head
  writes into the key spaces of three of the four induction heads (`L7H8`, `L8H6`, `L11H14`) at
  21–32× chance, each in the top 1 % of its layer pair. It appears between 512 and 1000, just after
  B1's onset in (256, 512] and with 7d's members by 1000. `L12H5`'s keys do not read it. The members
  were picked from activations (7d, B1), so this agrees with how they were found; it is not an
  independent discovery.
- **`L5H2`'s write space has more readers than the members.** In layers 7, 8, 10 and 11, other
  heads read it about as much (`L7H1`, `L7H7`, `L8H9`, `L8H7`, `L11H2`, `L10H9`, `L10H0`). The rank
  says members' keys sit above a random same-layer head. It does not say they are the only, or the
  strongest, readers.
- **QK symmetry sets the members apart, in both directions.** `L7H8` is the most symmetric head at
  its step from 2000 on (C3 0.94 at 143000). `L8H6`, `L11H14` and `L10H7` sit near the 95th.
  `L5H2` and `L12H5` sit near the 15th. The rank uses |p − ½|, so the two groups add.
- **No weight-level attraction or repulsion sets the members apart.** C1, C2 and C4 show no
  difference. Members spread across their steps' percentiles, and their percentiles move between
  steps (`L12H5` from 0.32 to 0.96 on C1). Within the gated reach, an OV sign is not what makes
  these heads a circuit.
- **What it does not establish.** Composition is a subspace overlap, not a path weighted by what is
  written or attended (B4's nudges test that). A symmetric QK form is not a gradient-flow head: 2d's
  regime needs V aligned too, and no head is. The members come from one rung (410m) and were found
  on activations, so nothing here is a held-out reading.
- **Not established: that C3 marks the circuit rather than a kind of head** (`/challenge-pr` on
  #221, finding 2). Non-member heads of the same kinds sit where the members do on C3 at 143000:
  `L5H9` (layer 5's second previous-token head, `design-B.md`) 0.10 and `L4H9` (positional in
  B2a) 0.18 sit low with `L5H2`; `L8H9` (7d's other
  induction head) 0.93 sits high with the induction members. The layer-matched null cannot tell
  "in this circuit" from "a previous-token or induction head". `L7H1` and `L7H7`, strong readers of
  `L5H2`, sit at 0.34 and 0.31, so the split does not follow composition either.

Not built: Phase 6's `U_pos` / `U_neg` / `U_A` dimensions of the members' union (design "Beside, not
in the family"); parked below.

## B2 unit 1b (2026-10-18): the 9-step re-gate (Blocked 62 (a), (c))

**Input:** code `b2.code_sha()` = `da75a1f92328` (`b2.py regate`: `g3m_step`, `g3m`, `g5`'s `copy`),
records `/run/media/system/WDS_500/Mets/data/b2/2026-10-18-regate/` (`regate.json`,
`g3m_step{512,…,143000}.npz`: every head's C1 and the planted heads' dosed C1 per dose; log
`../2026-10-18-regate.log`). Weights of the 9 steps ≥ 512 from the HF cache, float64 CPU, 24 min.
G3's 200 fake sets (seed `b2-g3`, 90 heads), each head dosed as G3 doses, to its own step's
quantile. No member value was read. Re-run: `METS_DATA=<data> python -m trackB_circuit.b2 regate
--out <dir>`. 62 (c), the user's: "two-sided" stays as frozen (p in the step's 384 heads, the null
layer-matched); no code change.

| G3m: C1's T over the 9 steps ≥ 512, share of 200 draws at rank ≥ 0.99 (median rank) | 70th | 75th | 80th | 90th |
|---|---|---|---|---|
| planted at all 9 steps (**gated: ≥ 80 % at the 90th**) | 0 % (0.56) | 3 % (0.87) | 42 % (0.987) | **100 %** (1.0) |
| late onset, planted at 8000–143000 only (5 of 9), beside | 0 % (0.53) | 0.5 % (0.72) | 4.5 % (0.89) | 67 % (0.996) |
| one step, 143000, G3's own seeds (wiring check) | 0 % | 0 % | 8 % | 100 % |

| G5's dose curve at 143000 (copies of A's write directions into B's keys) | planted C6 (primary) | primary rank | 200 draws at ≥ 0.99 | unplanted in central 90 % |
|---|---|---|---|---|
| 1 | 0.084 | 0.992 | 55.5 % | 95.5 % |
| 2 | 0.147 | 0.9997 | 100 % | 95.5 % |
| 4 (unit 1's G5) | 0.274 | 1.0 | 100 % | 95.5 % |

Chance C6 is 1/64 = 0.0156; the unplanted primary is 0.0169.

**What it says.**
- **The gated arm passes, so the constants stand** (rule 6). The one-step slice returns unit 1's G3
  exactly, so the 9-step run is wired the same way.
- **Averaging over steps helps for members that are extreme at every step.** At the 80th the
  9-step T sees 42 % of planted sets; one step sees 8 %.
- **The frozen T cannot see a set held between the 25th and 75th.** Under the null E|p − ½| = ¼
  exactly (`tools/math_checks/b2_two_sided_floor.py`). A set held at the 75th (or the 25th)
  scores the null's mean, and a set at the 70th scores below it. Sets beyond, high or low, are
  seen alike. The 0 % at the 70th is a property of the frozen statistic, not low power. A
  statistic that averages each member over steps before taking |·| has no such floor (below), so
  the band follows from the choice of T.
- **Members that turn extreme only late are diluted.** Planted at 8000 onward, a set at the 90th
  is seen in 2 of 3 draws, and at the 80th almost never. A "no difference shown" in unit 2 does not
  rule out a late onset below about the 90th.
- **Beside, after `/challenge-pr` on #220 (C1 only, from the saved records, not gated):**
  - G2's false pass on the 9-step T is 2.0 % (at G2_MAX).
  - The alternative S = mean over members of |mean over steps of (p − ½)| has a false pass of
    1.5 %, and sees 2 % / 13.5 % / 88 % / 100 % of consistent sets at the 70th / 75th / 80th / 90th
    (T: 0 / 3 / 41.5 / 100).
  - Planted only from 8000, S sees 0 % at the 80th and 1.5 % at the 90th (T: 4.5 %, 67 %), because
    a head's own early percentiles stay in its step mean.
  - So S trades late reach for consistent reach. Which statistic unit 2 ranks is Blocked 63.
  - Producer: `METS_DATA=<data> python -m trackB_circuit.b2_regate_alt`, from `g3m_step*.npz`.
- **G5's reach.** One copied direction per pair (C6 ≈ 5× chance) is seen in about half the draws.
  Two copies (≈ 9×) are seen in all of them.

## B2 unit 1 (2026-10-17): the columns and the gate

**Input:** code `b2.code_sha()` = `7f69b19b41a6`, records
`/run/media/system/WDS_500/Mets/data/b2/2026-10-17-gate/` (`gate.json`, `population_step{0,512,143000}.npz`
(C1–C4 and the beside values of all 384 heads), log `../2026-10-17-gate.log`). Weights of steps 0,
512 and 143000 from the HF cache, float64 CPU, 6 min. B2a's records for G7 as `design-B2.md` names
them. Constructions as built: `design-B2.md` "Unit 1's gate, as built". No member value was read
or printed. Re-run: `METS_DATA=<data> python -m trackB_circuit.b2 gate --out <dir>`.

| arm | passes when | result |
|---|---|---|
| G0 | stored OV vs `W_O W_V` ≤ 1e-5; fold = B2a's V_h; low-rank = dense | stored is `(W_O W_V)ᵀ` at both steps (row convention), max rel 1.3e-7 (other orientation ≥ 0.74); fold 0 (same algebra as `b2a.head_weights`, so (ii) checks only the wiring); dense vs low-rank 1.9e-15, regime labels equal |
| G1 | step-0 halves' outside rate in [0.03, 0.08] | C1 0.063, C2 0.060, C3 0.059, C4 0.060 (per-split 5–95 %: 0.03–0.10). Step-0 bands: C1 0.4963–0.5037 (raw 0.4963–0.5037, as the design quoted), C2 ±0.0126, C3 0.4959–0.5041, C4 ±0.0018 |
| G2 | false pass ≤ 0.02 per column | 512: C1 0.020, C2 0.010, C3 0.010, C4 0.005, C6 0.010; 143000: 0.015, 0.015, 0.015, 0.020, 0.000. Two at the bound |
| G3 | at the 90th, rank ≥ 0.99 in ≥ 80 % | dose 75th 0 %, 80th 8 %, **90th 100 %**, 95th and 99th 100 %; median rank 0.80, 0.94, 0.9998 |
| G4 | planted pair: C2 < 0, flag set; ν = 0 own values | `L22H4`, ν 1.64: C2 −0.049, flag set; ν = 0 differs by 1.1e-16 |
| G5 | planted rank ≥ 0.99, unplanted in the central 90 % | primary set (`L5H4` `L7H4` `L8H9` `L11H13` `L12H7`): planted C6 0.274, rank 1.0; unplanted 0.0169 (chance 1/64 = 0.0156), rank 0.18. Over 200 draws: planted 100 %, unplanted central 95.5 % |
| G6 | asym ≤ 1e-6, C4 ≥ 0.999, gradient-flow, Table 1 row as defined | `L17H13`: asym 0, C4 1.0, `gradient_flow`, `row2_three_hyperplanes` as expected |
| G7 | C7's inputs present and exact | 18 records, the 14 heads' `m2.qk` at each, codes and prompts hash as named, TV at D = 128 ≤ 4.5e-13 |

**What the gate says about reach, at one step only:** read at one step (143000), the member
statistic sees a set only when all its members sit near their step's 90th percentile: at the 80th
it passes 1 draw in 12 (planted upward only; by the symmetry of |p − ½| the 10th should behave the
same, untested). The rule averages the **9** steps ≥ 512, and non-members' |p − ½| barely
correlates between steps (512 against 143000: −0.04, 0.18, 0.03, −0.11 for C1–C4, `/challenge-pr`
on #219, finding 1), so the averaged null is tighter by up to about 3×. Members consistent across
steps would then be easier to see than this, and members extreme only late would be harder. The
9-step reach is not gated (Blocked 62).

## B2a on one page (closed 2026-10-16)

**Asked:** for each of 14 heads (`b2a.heads()`), the smallest subspace on V's invariant ladder whose
reduced dynamics (M1: `β cos(Uᵀu_i, Uᵀu_j) + φ(i − j)`, 2609.28448 §6.1) reproduce the head's real
attention; D = 2 is the paper's circle. **Exit met** (`design-B2a.md` "Exit condition and budget"):
a verdict per head at 143000 (unit 2) and at all 18 steps (unit 3), in the 4 units of Blocked 50.

**Established (tier 1, 410m, v1's 7 prompts `976b62d677bc`, 18 steps, 14 heads):**

1. **No head has D\* ≥ 1 at any step (no circle).** Of the 252 (head, step) cells, 214 read
   outside, 32 positional (D\* = 0), 6 "none"; a cell reaches V's ladder only when M1's full fit is
   inside and content adds, so the Haar nulls ran in 5 (`L9H8` at 8000, `L10H7` at 16000–143000),
   all "none". The largest gain any V rung gives over position alone is 0.092 (`L18H1`, 2000).
   (`/challenge-pr` on #211, findings 1–2: first read "no V subspace below its Haar nulls" and
   "because the tokens do not sit in V's subspaces".) "B2a unit 3".
2. **Why, mostly: the heads are not in M1's class.** To 256 attention is near uniform and nothing
   adds (nothing to fit); after it most heads read outside because even M1's full isotropic cosine
   plus position misses (TV(full) > 0.5). Where M1 fits, it fits by position. So "most trained
   heads on 410m are not isotropic content plus position" is the main reading, and the ladder was
   seldom reached.
3. **Where the ladder was reached, the tokens do not occupy V's subspaces.** Their share of norm²
   there is about chance (U₂ ≤ 0.01 at 143000; over 18 steps only `L5H2` late and `L18H1` stand
   outside step 0's spread), so content that helps (`L7H8`, `L10H7`) is carried by no V subspace.
   Occupancy alone does not decide a head: `L18H1` holds 37× chance in U₂ at 143000 and still
   reads outside (TV(full) 0.717). "B2a unit 2", "unit 3".
4. **Positional heads.** From 512: `L5H2`, `L4H9` (B1's onset for `L5H2` is (256, 512]), `L12H5`
   from 2000; 143000's 4 hold from 16000.
5. **V's 2D blocks are at the random-weight rate; training shows in their sign.** V's top pair is
   a D = 2 block in 123 of 252 cells, but in 7 of 14 heads at step 0 and 68 of 126 cells at
   0–256 against 55 of 126 from 512 (`/challenge-pr` on #211, finding 3). What training adds:
   near-repulsive top pairs in 10 cells, all at 256–32000, none at init; no V block beats the
   kernel in 31 cells, all at 1000–32000 (the OV repulsive phase, `PROJECT.md` §3.12 A). "B2a unit 1".
6. **The instrument sees what it should.** The planted gate passes every arm at the design's
   constants (a faithful rung at TV 0.005, the nearest random subspace ≥ 0.1); G0 at every step.
7. design-A's prior (2D fails for previous-token and induction heads) is not contradicted.

**Not established:**

- Anything off V's ladder: D\* is the smallest *on the ladder*, an upper bound (design "Worth
  challenging"). Whether a head's QK read space is low-rank (M2's ladder, read only beside) is a
  question about QK rank, not the paper's reduction (design "Worth challenging"); that a head reads
  a token only through `W_Q y`, `W_K y` is true by construction (`/challenge-pr` on #211, finding 4).
- Dynamics: each head acts once; whether depth iterates the map, and whether the dynamics
  *selects* where tokens sit, are untested (design "Fences").
- The 4 positional / 8 outside split at 143000 rests on the placed 0.5 (at 0.55 `L6H4`, `L10H14`
  read D\* = 0); no D\* ≥ 1 either way. `L5H2`'s 143000 verdict is post hoc (Blocked 52).
- Heads outside the 14, prompts outside v1, other rungs (410m is exploration; nothing registered).
- A1a's β against M1's: opposite signs in 6 of 14 heads (all M1 −, A1a +), not resolved here.

**Open threads, at close:**

| thread | at close |
|---|---|
| The gate has no position-0 sink arm | **dropped**: the rule excludes position 0 and renormalises, and a softmax row with one key removed and renormalised is exactly the softmax over the rest, so a planted sink changes nothing the rule reads (`/challenge-pr` on #211) |
| The gate has no arm with content only partly V's | **dropped**: it bears only on the 6 cells that reached V's ladder, and there reading 3 (occupancy and M2, independent of the gate) says the content is not in V's subspaces |
| Token-matched null | dropped at unit 3 ("Parked" below) |
| A1a's β against M1's | **carried** to A2 as its first item (Blocked 54 (a)) |
| QK read rank (M2's ladder: how much of the token a head reads; a QK-rank question, not the paper's reduction) | **carried** to B2 (`docs/ROADMAP.md` §7), beside its QK spectra |
| `L10H7` "none" (content helps, V carries none of it), while its V's top pair is near-repulsive at 2000 and 32000 and no V block beats the kernel at 1000–8000 (unit 1's `weights.json`, read at close) | **carried** to B6 (`docs/ROADMAP.md` §7), beside its 410m `L10H7` line |

## B2a unit 3 (2026-10-16): the 18 steps

**Input:** v1's 7 prompts (`976b62d677bc`), Stage 0's runs at each step, the 14 heads; HF forward on
the RTX 3080 float32 (TF32 off), rebuild and fits float64. Two batches (`design-B2a.md` "Unit 3 as
built" and its amendment): the occupancy screen and the rule at its 6 steps, code `37d4f2487738`
(commit `c03076a`), records `/run/media/system/WDS_500/Mets/data/b2a/2026-10-16-occ/`, 42 min; then,
after `/challenge-pr` on #210 found the screen's baseline wrong (the user's pick), the rule at the
other 11 steps, code `939e114f7afb` (commit `e0063b4`; the rule's code unchanged), records
`.../2026-10-16-occ2/` (its `occupancy.json` is the one to read: it carries `chance_ratio`), 61 min;
143000 is unit 2's. Re-run: `python -m trackB_circuit.b2a occupancy --out <dir> --unit2 <unit 2's
real.json>`, `... gate --out <dir>`, `... real --out <dir>/step<S> --gate <dir>/gate.json --step <S>`
per step, `... report --out <dir>`.

**Checks.** Both passes reproduce unit 2's rung occupancies at 143000 exactly (max |Δ| 0). Gate 9 of 9
on each code. G0 passes at all 17 steps (worst scaled 3.80, step 4; bound 10).

**Occupancy** (`b2a report`; the statistic is a median, whose Haar chance is 0.69 × D/d at D = 2 and
0.79 × at D = 3, `chance_ratio`; U₂ moves up to U₃ for `L7H8`, `L9H1`, `L18H1` at 143000, and the set
varies by step). Over chance, U₂ is 0.4–1.8× at step 0 (random weights: the spread of a median over
7 prompts) and up to 2.7–3.4× in some head from 16 on; U₆₄ is at chance to 16, then 1.2–2.2× in most
heads from 512. Only `L5H2` late (7.6× chance at 143000) and `L18H1` (8.2× at 1000, 37× at 143000)
stand well outside step 0's spread. The screen (Blocked 53 (a)'s bar, set against the mean) chose
7 steps; it did not decide anything, since the rule then ran at all 18.

**The rule at every step** (143000 from unit 2):

| step | D\* ≥ 1 | D\* = 0 | outside (TV(full) > 0.5 / nothing adds) | none | largest V-rung gain over D = 0 |
|---|---|---|---|---|---|
| 0, 2, 4, 8, 16 | 0 | 0 | 14 (0 / 14) | 0 | ≤ 0.001 |
| 32, 64, 128, 256 | 0 | 0 | 14 (0 / 14) | 0 | 0.003, 0.010, 0.019, 0.015 |
| 512 | 0 | 2 (`L5H2`, `L4H9`) | 12 (1 / 11) | 0 | 0.017 (`L18H1`) |
| 1000 | 0 | 2 (same) | 12 (3 / 9) | 0 | 0.047 (`L18H1`) |
| 2000 | 0 | 3 (+ `L12H5`) | 11 (10 / 1) | 0 | 0.092 (`L18H1`, U₂ 1.2× chance) |
| 4000 | 0 | 3 (same) | 11 (10 / 1) | 0 | 0.058 (`L7H8`) |
| 8000 | 0 | 5 (+ `L10H7`, `L6H4`) | 8 (7 / 1) | 1 (`L9H8`) | 0.047 (`L9H8`) |
| 16000 | 0 | 5 (`L5H2`, `L12H5`, `L4H9`, `L6H4`, `L9H1`) | 8 (8 / 0) | 1 (`L10H7`) | 0.037 (`L9H8`) |
| 32000, 54000 | 0 | 4 (`L5H2`, `L12H5`, `L4H9`, `L9H1`) | 9 (9 / 0) | 1 (`L10H7`) | 0.038 (`L9H8`) |
| 143000 | 0 | 4 (same) | 8 (8 / 0) | 2 (`L7H8`, `L10H7`) | 0.037 (`L9H8`) |

| reading | what it rests on |
|---|---|
| **No head of the 14 has D\* ≥ 1 at any of the 18 steps (no circle), on v1; the Haar nulls ran in 5 of 252 cells, the rest decided before the ladder** (corrected 2026-10-16, `/challenge-pr` on #211, finding 2: it first read "no V subspace up to D = 64 below its nulls"). This is the rule's verdict at every step, not the occupancy screen's. B2a is at its exit with its 4 units spent | `step<S>/real.json` in both record dirs |
| To 256, attention is near uniform (TV(uniform) 0.06–0.32) and every head reads outside by Blocked 51/52's clause (neither content nor position adds 0.05): "M1 has nothing to fit", not "M1 fits badly". The clause decided no head at 143000 | `real.json` "outside_why", "tv_uniform" |
| Positional heads appear from 512 (`L5H2`, `L4H9`; B1's onset for `L5H2` is (256, 512]), `L12H5` from 2000; `L10H7` (8000), `L6H4` (8000–16000) and `L9H1` (16000 on) come and go; 143000's 4 hold from 16000 | the table; "B1 (2026-10-14)" |
| The largest gain any V rung gives over position alone is 0.092 (`L18H1` at 2000, where its U₂ share is 1.2× chance, inside step 0's spread); where content helps (`L7H8`, `L9H8`, `L10H7`, `L18H1`, up to 0.086 from 2000), no V subspace carries it at any step | "content adds", "rungs" per step |

**Not established:** anything about heads outside the 14 or prompts outside v1; whether a token-matched
null would change a verdict (no V rung beat its Haar nulls at any step, so there is nothing for it to demote).

## B2a unit 2 (2026-10-16): real tokens at 143000

**Input:** code `b2a.code_sha()` = `9fe738d5a008` (commit `2f8c092`), v1's 7 prompts
(`prompts_hash` `976b62d677bc`), Stage 0's stored runs at 143000 (`trackA_force.a1a.run_dir`), HF's
forward on the RTX 3080 float32 (TF32 off), rebuild and fits float64. Records
`/run/media/system/WDS_500/Mets/data/b2a/2026-10-16/` (`gate.json`, `real.json`, `run.log`); 22 min,
16 of them the gate. Rule: design "The rule", with Blocked 51 (a) and 52 (a′) (both the user's, taken
before the batch; 52 after one head, `L5H2`, was read in a smoke test, so `L5H2`'s verdict is post hoc).
Re-run: `python -m trackB_circuit.b2a gate --out <dir>`, then `... real --out <dir> --gate <dir>/gate.json`,
`... report --out <dir>`.

**Checks.** The gate reran on this code and passes all 9 arms: the 7 of unit 1 unchanged, **G4w**
outside at both scales (TV(full) = TV(0) = 0.195, 0.378) and **G2n** D\* = 0 at both (TV(full) 0.163,
0.324 against TV(uniform) 0.449, 0.529). **G0** passes on all 98 (head, prompt) cells: worst
`L13H8`/`camus_letranger`, 2.8e-6 raw, 1.58 of the bound 10.

**The rule's verdicts** (TV is the mean over prompts of the per-prompt mean over rows; "content adds" =
TV(0) − TV(full), "position adds" = TV(uniform) − TV(0)):

| head | set | verdict | TV(full) | content adds | position adds | best V rung's gain over D = 0 (D ≤ 64) | M2 on QK's top 64 | sink |
|---|---|---|---|---|---|---|---|---|
| `L5H2` | previous-token | **D\* = 0** (post hoc) | 0.125 | 0.000 | 0.820 | 0.000 | 0.063 | 0.00 |
| `L7H8` | induction | none | 0.468 | 0.062 | 0.397 | 0.000 | 0.094 | 0.84 |
| `L12H5` | induction | **D\* = 0** | 0.412 | 0.006 | 0.478 | 0.013 | 0.221 | 0.38 |
| `L8H6` | induction | outside | 0.811 | 0.000 | 0.020 | 0.003 | 0.448 | 0.80 |
| `L11H14` | induction | outside | 0.742 | 0.000 | 0.025 | 0.000 | 0.554 | 0.77 |
| `L10H7` | negative interactor | none | 0.469 | 0.072 | 0.088 | 0.024 | 0.309 | 0.60 |
| `L6H4` | reference | outside | 0.522 | −0.010 | 0.262 | 0.000 | 0.335 | 0.65 |
| `L9H8` | reference | outside | 0.542 | 0.079 | 0.143 | 0.037 | 0.311 | 0.58 |
| `L4H9` | reference | **D\* = 0** | 0.261 | 0.000 | 0.646 | 0.000 | 0.135 | 0.00 |
| `L15H7` | reference | outside | 0.648 | −0.029 | 0.235 | 0.003 | 0.562 | 0.57 |
| `L13H8` | reference | outside | 0.598 | −0.003 | 0.099 | 0.001 | 0.306 | 0.43 |
| `L9H1` | reference | **D\* = 0** | 0.460 | 0.006 | 0.238 | 0.009 | 0.222 | 0.51 |
| `L18H1` | reference | outside | 0.717 | −0.009 | 0.020 | 0.000 | 0.624 | 0.75 |
| `L10H14` | reference | outside | 0.524 | −0.004 | 0.290 | 0.004 | 0.230 | 0.75 |

| reading | what it rests on |
|---|---|
| **The reduction's premise fails at 143000: the real tokens do not lie in V's invariant subspaces.** Their share of norm² in U₂ is ≤ 0.01 (below the median's Haar chance, 0.69 × 2/1024, in 8 of the 11 heads whose U₂ is 2-dimensional; above it `L5H2` 7.6×, `L6H4`, `L12H5` near it; corrected 2026-10-16, `/challenge-pr` on #210, finding 1: it first read "below the random 2/1024 in 11 of 12", against the mean), and in U₆₄ 0.6–5.3× the random 64/1024, ≤ 0.13. So no head has D\* ≥ 1 (no circle), and TV on V's ladder is flat in D (largest gain of any rung over D = 0: 0.037, `L9H8`); given the occupancy this is close to expected, not a separate finding (`/challenge-pr` on #209, finding 1) | `real.json` "rungs" "occupancy" |
| M1's isotropic cosine adds at most 0.079 of TV over the offset profile in any head; in 11 of 14 it adds < 0.01 (negative values, down to −0.029: the full model adds one parameter, β, and a β fitted on 6 prompts transfers worse to the 7th than none) | "content adds" |
| 4 heads are positional (D\* = 0): `L5H2`, `L4H9` fitted well (0.125, 0.261), `L12H5`, `L9H1` only as well as M1 fits at all (0.41, 0.46). 8 are outside, **all by TV(full) > 0.5; the clause Blocked 51 and 52 added decided no head**, and the frozen rule gives the same 14 verdicts. Under 51 (a) as first built, all 4 positional heads would have read outside. **The 4 / 8 split rests on the placed 0.5:** `L6H4`, `L10H14`, `L9H8` are outside at 0.522–0.542; at 0.55 the first two would read D\* = 0 and `L9H8` would go to its nulls at D = 64. No D\* ≥ 1 either way | "position adds" ≥ 0.24 for the 4 |
| `L7H8` and `L10H7` are "none": content helps (0.062, 0.072), but no V subspace carries it. `L10H7`'s D = 64 rung meets the TV clause (0.518 ≤ 0.519), and the first 32 Haar subspaces of dimension 64 all tie or beat it (min 0.499; occupancy V 0.07 against the nulls' median 0.06) | `real.json` "nulls" |
| **The content a head reads is in its QK read space, not in V's invariant subspaces** (a consequence of the occupancy row, not independent of it). M2 (the head's own logit with y projected) on V's ladder stays at its bias-only value (0.63–0.94) at every rung; on QK's ladder it falls to 0.06–0.62 at D = 64 and is exact at the full rank 128 (G5 (ii)) | `real.json` "m2" |
| The prior (design-A: 2D fails for previous-token and induction heads) is not contradicted: `L5H2` D\* = 0, the 4 induction members D\* = 0, none, outside, outside | the table |

**Beside.** γ (the head's output over its LN1 row, as γ/(1 − γ)) 0.005–0.146, largest `L5H2`. The
motion cosine at full D is 0.49–0.99 (median over tokens) but mostly V's own direction, since the reduced
and real updates share V; at D = 2 it is −0.10 to 0.38. Occupancy of V's U₂ by the real tokens is ≤ 0.01
except `L18H1` (0.09 at D = 3).

**A1a's β against M1's at full D (the design's "must reproduce it in sign and order"): sign agrees in 8
of 14.** The 6 that disagree (`L6H4`, `L4H9`, `L15H7`, `L13H8`, `L18H1`, `L10H14`) are the heads where
content adds ≤ 0; where it adds ≥ 0.05 (`L7H8`, `L10H7`, `L9H8`) the signs agree and the order is
the same (M1 7.9 > 6.6 > 5.6; A1a 15.5 > 10.3 > 9.8). The two are different estimators: A1a's
`fe_full` is least squares on log A with per-offset effects, every pair weighted equally; M1 is
cross-entropy, weighted by attention mass. **The disagreement has a direction** (`/challenge-pr` on #209,
finding 4): A1a's β is positive in 13 of 14 heads, and all 6 disagreements are M1 negative, A1a
positive, so one estimator is biased where the cosine explains nothing, not both noisy. The check
fails as written; nothing in B2a's rule reads it (A1a's β is beside only, design "Fences"), but A1's
layer β is the median of these per-head βs, so it goes to the user as Blocked 54.

## B2a unit 1 (2026-10-15): the gate and the weights rows

**Input:** code `b2a.code_sha()` = `72882e6f353d` (commit `32ddebd`), records
`/run/media/system/WDS_500/Mets/data/b2a/2026-10-15/` (`gate.json`, `weights.json`, `run.log`).
Gate: synthetic, d = 1024, n = 500, 7 prompts per arm, float64 on the RTX 3080, 9 min. Weights:
the 18 steps, the 14 heads (`b2a.heads()`: the 6 named, then `L6H4`, `L9H8`, `L4H9`, `L15H7`,
`L13H8`, `L9H1`, `L18H1`, `L10H14` drawn at seed 20261015), float64 CPU, 27 s. Constructions
as built, not as the design's table: `design-B2a.md` "Unit 1's gate, as built".
Re-run: `python -m trackB_circuit.b2a gate --out <dir>`, `... weights --out <dir>`, `... report --out <dir>`.

**The gate: every arm passes, with the design's constants (TV 0.5, +0.05) unchanged.**

| arm | verdict | numbers |
|---|---|---|
| G1, circle | D\* = 2, degenerate negative, angle 0 | TV(0) 0.566, TV(1) 0.448, TV(2) 0.0047; 320 nulls at D = 2: min 0.127, median 0.404 |
| G1r, rotation | D\* = 2, rotation | TV(2) 0.0047; nulls min 0.108 |
| G1d, D = 4 | D\* = 4 | TV(2) 0.389, TV(3) 0.260, TV(4) 0.0045; nulls min 0.245 |
| G2, positional | D\* = 0 | TV 2e-9 |
| G3, null | none in 20 of 20 | the best ladder rung's TV 0.099–0.152 against TV(full) ≈ 0, so no rung reached its null comparison |
| G4, out of class | outside | TV(full) 0.815 |
| G5 | (i) 1.6e-15; (ii) 1.2e-11; (iii) 0.81 of the bound (raw 3.8e-4, at `L18H1`'s \|logit\| 8.2e3) | (ii), (iii) in `weights.json` "g5_real" |

The margins are wide: a faithful rung sits at 0.005 and the nearest random subspace at ≥ 0.1. The
gate does not test a head whose content subspace is only partly V's (V's ladder is fixed), nor
attention with a sink (synthetic prompts have no position 0).

**Weights rows (252 cells; V in the LN1 frame at γ = 0.1, placed), descriptive:**

| row | reading |
|---|---|
| \|λ₁\| | 0.013–1.93; ≈ 0.014 for every head at steps 0–256, then growing to a peak at 16000–54000 and falling at 143000 |
| U₂'s type | a complex pair 120 (the paper's rotating circle `W = R_φ`, also a D = 2 case; 8 within 0.05 rad of π, 1 within 0.05 of 0), mixed 129, degenerate negative 2 (`L12H5` at 512, `L15H7` at 4000), degenerate positive 1 (`L10H14` at 16000). **Near-repulsive** (degenerate negative, or a pair within 0.05 rad of π): **10 of 252**; any D = 2 top pair (a complex pair or a degenerate real one): 123. At 143000: a pair for `L5H2`, `L12H5`, `L9H8`, `L4H9`, `L13H8`, `L10H14`; mixed for the rest. (Labelling: a real pair counts as degenerate within 5 %, a complex pair at any angle; `/challenge-pr` on #208, finding 4) |
| kernel beaten | in 221 of 252 cells; the 31 where no block beats `1 − γ` are at 1000–32000 (10 at 1000, 8 at 2000, 7 at 4000), where the top block is a negative real: the OV repulsive phase on these heads (`PROJECT.md` §3.12 A) |
| ρ_⊥ where beaten | 0.949–0.99999: at \|λ\| ≲ 2 and γ = 0.1, \|b\| ≈ 0.9 + 0.1 Re λ, so one application selects nothing; γ is read in unit 2 |
| selection order | the paper's top block (by \|b\|) is V's top by \|λ\| in 64 of 252 cells (by Re λ, γ's small-γ limit: the same 64); V's ladder and the paper's selection order disagree in most cells, as the design expected for negative λ |

## B1 (2026-10-14): `L5H2` as positive control

**Input:** as the card; B0's records `2026-10-14c` (code `1e78e2be9d94`), accepted by
`check_gate` under the waiver. Code `b1_prev_token.code_sha()` in every record. Records:
`/run/media/system/HDD_1TB/mets_data/trackB_b1/2026-10-14b/b1_step*.json`, `b1.json`, `run.log`,
on code `70b28da06186`; the first run (`2026-10-14/`, code `07b452b5a7c2`, a docstring apart)
gives the same `b1.json` steps exactly.
36 s per step on the RTX 3080, 11 min in all.

**Verdict by the frozen rule: PASS: the instrument sees `L5H2`'s signature switch on**, by the
pass rule as revised by `/challenge-pr` on #197, finding 1. The design's exit sentence and
ROADMAP's first window, (512, 4000], are not met as written: the signature is already on at 512. Off at
step 0, on at 1000, 2000, 4000 and 8000. The reading at c = 0.03 (s = log(σ₁ / median σ₂…σ₅),
median over 32 positions; *on* = `L5H2` above ≥ 14 of layer 5's other 15 heads):

| step | 0–8 | 16 | 32 | 64 | 128 | 256 | **512** | 1000 | 2000–54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|
| reading | off | off | off | off | off | off | **on** | on | on | on |
| prompts on (of 7) | 2 | 2 | 2 | 1 | 1 | 3 | 7 | 7 | 7 | 7 |
| `L5H2`'s s, median over prompts | 0.02–0.05 | 0.01 | −0.01 | 0.05 | 0.06 | 0.01 | 2.60 | 9.07 | 8.84–9.23 | 7.97 |
| cos(A, OV-mapped nudge) at k = 1 | 0.98 | 0.99 | 0.98 | 0.97 | 0.97 | 0.97 | 0.85 | 0.99 | ≥ 0.997 | 0.997 |
| ‖A‖ / r_out at k = 1 (the yardstick) | 1.1e-5 | 8.7e-6 | 8.8e-6 | 9.4e-6 | 1.1e-5 | 8.8e-6 | 6.8e-4 | 3.0e-3 | 3.2–3.6e-3 | 2.7e-3 |

(Every step, per prompt and at the other three c: `b1.json`; the reading is the same at all
four c at every step.)

- **Onset in (256, 512].** Recorded before the run: (256, 1000]. Steps 2–512 were to be recorded,
  not judged; 512 reads on at 7 of 7.
- **The yardstick for "small"** (the one place this number lives): `L5H2`'s offset −1 response,
  ‖A‖ / (c · r_out) with r_out block 5's RMS output norm, is **0.10–0.12** from 1000 to 54000
  and 0.09 at 143000, against ~3e-4 before 512: a factor of ~300. It is linear in c up to 0.1
  and 5–7 % lower at c = 0.3 (‖A‖/r_out at 4000: 1.2e-3, 3.6e-3, 1.2e-2, 3.3e-2 at the four c;
  the table's row is c = 0.03), so the ratio, not a single ‖A‖, is the yardstick. The late fade (7d: 15 % of the causal effect left at 143000) is
  ~25 % here (2.7e-3 against the 3.6e-3 peak at 4000–16000), by this measure.
- **The response is the OV-mapped nudge.** cos ≥ 0.997 from 2000 on. At init it is 0.98 too,
  since any head's response at k = 1 is its OV map of the nudge times its attention weight; the
  cosine says the readout is the head's value path, not that the head attends back.
- **`L5H9`** (the second previous-token head, the reason for "14 of 15") reaches s 0.9–2.2 from
  1000 on, against `L5H2`'s 7.3–9.7; `L5H2` beats all 15 at every prompt from 1000.
- **At step 0–256, 1–3 prompts of 7 read on by chance**: `L5H2` is one of 16 heads with s near 0,
  so it tops ≥ 14 of 15 in about 2 of 16 draws.

**The statistic's size from 1000 is a float32 detection limit.** `L5H2` is so sharp that its
attention weight on token j from 3–5 tokens later is ~1e-7–1e-10 (step 1000, `hdbscan_code`:
medians 3.6e-7, 4.8e-10, 3.5e-9; none exactly 0), so the ± change there is under float32
resolution of the head's output and A cancels to exactly 0. 47–73 % of its σ₂…σ₅ are under the rounding floor from 1000 (2 % at 512, 0 before),
and in s they are held at the floor. So s ≈ 9 says "under rounding", not e⁹. Without the
substitution the heads beaten are the same at every step and prompt (checked), so the verdict
does not rest on it. *The substitution was set after the first run (step 1000) refused on an
exact zero, before any verdict was read; `b1_prev_token.py`'s docstring says so.*

**Not established:** what B1 does not cover: it ranks one head's output against its layer's
other heads, from block 5 alone, so the multi-layer path, the ranking against random directions
and the residual and loss readouts B2–B4 read rest on B0's gate only (G2, G3, G4); the
false-on rate of the rule (≈ 2/16 per prompt at chance, read off steps
0–256, not derived); whether s's size tracks `L5H2`'s strength once rounding is not the floor
(a float64 block pass would tell); any head other than `L5H2` as a target.

**Re-run:** `python trackB_circuit/b1_prev_token.py run --out <dir> --gate
/run/media/system/HDD_1TB/mets_data/trackB_b0/2026-10-14c && python
trackB_circuit/b1_prev_token.py report --out <dir> --gate <same>`.

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
a relative margin of 1.05e-4; the nudged |A|/ε at c = 0.01 is 0.072468, 9.8e-5 below linear
(1.4e-3 of this projection, 13× the margin), so it ranks 2. In G3's unit (over ‖∇L‖ = 1.165)
the error is 8.4e-5, under the largest G3 saw at step 128 (1.7e-4) and far inside its 10 %. This is the failure the amended
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
- **B1's floor.** *(Resolved by B1: the floor is the base row in another batch shape, at
  least one ulp of the readout; `design-B.md` "B1 as built". It does bind from 1000:
  "B1 (2026-10-14)".)* The same-shape base floor is exactly 0 under deterministic algorithms, so
  every later unit fixes a rounding floor of this kind before its run. (`/challenge-pr` on #198, risk 1.)

## Parked

- B2a, a token-matched null (random subspaces drawn to match the real tokens' occupancy): why, V's subspaces share LN1's gain pattern with the tokens and may beat Haar nulls by overlap alone (`/challenge-pr` on #208, finding 2); cost, one more null family per rung; could change a unit 2 D\* from "below the nulls" to "no better than where the tokens are". Unit 2 records each null's occupancy beside its TV first. *(Unit 2, 2026-10-16: no V rung beat its nulls, so nothing for this null to demote at 143000; kept for unit 3.)* **Dropped (unit 3, 2026-10-16):** no V rung beat its nulls at any of the 7 steps read either, so it can change no B2a verdict.
- A1a's per-head `fe_full` β against B2a's M1 β, opposite signs in 6 of 14 heads at 143000, all M1 −, A1a +. **Carried to A2 as its first item** (Blocked 54 (a), the user's, 2026-10-16; `trackA_force/status-A.md` Parked).
- B2, Phase 6's `U_pos` / `U_neg` / `U_A` dimensions of the members' union (design-B2 "Beside, not in the family"), not built in unit 2: why, a subspace view of the members beside the per-head columns; cost, about one session (`p6_subspace/math-6.md` §7.2 on the stored OVs); could change nothing in the family (beside only). **Dropped at B2's close (2026-10-18):** it can change no reading, and C6 by pair already gives B4 the shared space it needs ("B2 on one page").
- B2, which pairs carry C6 was read after the rank (descriptive, `c6_pairs.json`): why, the rank is the circuit's mean; cost, none (done); could change B4's choice of what to nudge: `L5H2`'s write space and the 11 heads after layer 5 whose keys read it at ≥ 10× chance (`c6_pairs.json` `l5h2_readers_all`, code `b12ad273a201`): members `L8H6`, `L7H8`, `L11H14`; non-members in 7d's causal top 10 `L7H1`, `L8H9`, `L10H9`; non-members outside it `L7H7`, `L9H0` (27×), `L8H7`, `L10H0`, `L11H2`; then `L6H3`, `L10H4` at about 9×, not the members' keys alone (the list first read the members' layers only; `/challenge-pr` on #222, finding 1). **Carried to B4** at B2's close (2026-10-18).
- B2, a head-type control for C3 (`/challenge-pr` on #221, finding 2): why, non-member previous-token and induction heads sit with the members; cost, a list of head types from 7d's sweep and one re-rank; could change C3 from "members differ" to "these head types differ". **Carried to B3** at B2's close (2026-10-18; B3 joins per-head reads to 7d's sweep).
- B7, N0's own sampling noise in the `P − N0` interval (`/challenge-pr` on #273, finding 3): why, N0's profiles are the sample's in every draw (choice 3), so a cell whose N0 reads high by chance (up to 0.044 against about 0.009 at 50 rows) can be read "rerouting" when it persists; cost, re-estimating N0 per draw (the U-statistic, so no duplicated self-pairs) and one re-gate; could change a sparse cell from "rerouting" to "persistence". Unit 2 prints N0's rows beside each cell; decided at B7's close.
- B7, the interval's selection tail (unit 2, caveat 1): why, each draw re-chooses h\* on about 3.6 distinct other prompts against the sample's 6, so where the best match differs by prompt the interval's lower tail is long; 5 of 38 F and 2 of 7 R cells rest on draws with < 5 distinct prompts; cost, an interval with h\* re-chosen on a fixed number of distinct prompts (or a prompt jackknife) and a gate arm whose heads differ by prompt, one re-gate and one re-read (about 10 min of GPU); could change those 7 cells' descriptions, not the 110 L cells. Decided at B7's close.
- B7, the 5 → 6 two-cycle candidates (2000–16000) and the late layers' F (20–22): why, layer 5 holds `L5H2` (B1) and the late layers are the most condensed (α 0.03–0.09) with the lowest P; cost, a per-head read of those cells from the records (`heads`), no new run; could change nothing in B7's grid; it could point S4's tied arm or B4 at a layer. Decided at B7's close.

## Corrections received

- 2026-10-24: B2a and B3 (and `lit-B.md`, `design-B2a.md`, `b2a.py`'s docstring, ROADMAP's B2a row) cite 2609.28448 by section numbers the paper does not have: §6.1 is Sec. V A, §6.2 and §7 are Sec. V B, §4.2 is III G, §4.3 is III H, §5 is IV, §2 is II B. A citation only; the content each points to is the one read, so no reading moves — `docs/readings/2609.28448.md` (`/challenge-pr` on #270, finding 1).
- 2026-10-16: B2a unit 2's "A1a's β against M1's" (6 of 14 heads, M1 −, A1a +) is the code and LaTeX prompts pulling the pooled M1 fit, not a bias of A1a's `fe_full`: fitted per prompt, those heads are positive on all five prose prompts (`L6H4` 2.1–3.1, `L15H7` 3.4–5.0) and negative on `hdbscan_code` and `latex_monograph` only. B2a's verdicts do not read β's sign, so none moves — `trackA_force/status-A.md` "A2 unit 2, the β refit at 143000 (2026-10-16)".
