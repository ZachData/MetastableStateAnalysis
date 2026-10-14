<!-- trackB_circuit/design-B.md -->
# Track B — design: the nudge harness (B0) and `L5H2` as positive control (B1)

Opened 2026-10-13 by the user's Blocked 40 (c) ("do a, b, and c in one go"), after Phase 10 and
1e closed and B-pre found no Phase 7 row adjudicable on 410m (`p7_motifs/status-7.md` "B-pre").
Plan: `docs/ROADMAP.md` §7 (B0, B1), §9 (order 2). Scan: `trackB_circuit/lit-B.md`. Tier 1
throughout: exploratory, unregistered; 410m is an exploration rung.

**Frozen 2026-10-14** (Blocked 42, the user's: "Blocked 42 frozen; build B0"), as revised by
`/challenge-pr` on #197. Choices B0 made where this text left one are in "B0 as built" below.

**Taken with the user's (c):** ROADMAP §10.5, "`L5H2` as the positive control and first anchor",
as ROADMAP wrote it. If the user meant otherwise, B1 is the unit to change.

## Exit condition and budget

- **Exit:** B1 has a verdict, "the instrument sees `L5H2`'s signature switch on inside
  (512, 4000]" or "it does not", with the signature's size recorded as the yardstick for "small"
  (ROADMAP §7 B1). On "does not", Track B stops on these instruments and the next unit is fixing
  them (ROADMAP §9's gate).
- **Budget: 3 units.** (1) this design, the scan and the `sympy` check; (2) B0, the harness and its
  calibration gate; (3) B1. At the budget the track's opening closes at what it knows.

## The math

`tools/math_checks/b0_antisymmetric_nudge.py` (`sympy`): for a response `f` smooth at the base
state, `A(ε) = (f(x+εd) − f(x−εd))/2 = εJd + O(ε³)` and `S(ε) = (f(x+εd) + f(x−εd))/2 − f(x) =
ε²H[d,d]/2 + O(ε⁴)`; the ± squared displacements agree at ε² and differ at ε³ by `2⟨Jd, H[d,d]⟩`;
`d → −d` flips A and keeps S. So A is the response to the direction's sign with no norm
matching, S the part blind to it. **What it does not prove:** the ε range where the expansion
holds; on an activation plateau (`lit-B.md` §2) it may hold nowhere B reads, which is why ε is
swept.

## B0, the harness (`core/nudge.py`, `core/position.py`)

| part | fixed here |
|---|---|
| site | **activation only in B0:** `resid_pre` at layer ℓ (a forward pre-hook on block ℓ), for a token set T (one position, or all). The weight site (a head's OV moved to `K ± ε·S`) is deferred to the first unit that reads it (B2/B4), with its own gate |
| direction | a unit vector `d ∈ R^1024`; the same `d` at every token of T |
| ε | relative to the site's RMS token norm `r = sqrt(mean_i ‖x_i‖²)`: `ε = c·r`, `c ∈ {0.01, 0.03, 0.1, 0.3}`, all four always run |
| passes | base, `+εd`, `−εd` for the direction and for **K = 32** random directions (Gaussian, normalised, seeded per (prompt, site)), at each ε; base run twice (the floor below) |
| readouts | per token and later layer: the displacement `A` and `S` of `resid_pre` (and, at the last layer, of the final residual); per position, the next-token loss's `A` and `S`; truncated forwards where a unit reads only up to some layer |
| null | the direction's readout is ranked among the K random directions' at the same site, ε and readout (exact rank, floor `1/(K+1)` ≈ 0.03); the turn-on ε (the smallest c whose `|A|` clears 10× the floor) is reported for the direction and for the nulls' median |
| position | `core/position.py`: one shared covariate for every per-token readout, a cubic in `log(i+1)` plus an indicator for position 0, residualised membership-blind (Phase 10 had three ad hoc ones, `p10_cluster_function/questions-10.md` §7 item 4) |

**NUMERICS** (ROADMAP §7's B0 row, carried from 1e): every pass of one comparison (base, ±, nulls)
runs in **one process, one device, one dtype, the same batch shape**; GPU float32 is the default
(the RTX 3080; the user's standing instruction), CPU float64 only for the gate's JVP check. **A
cross-device comparison is not allowed:** on long passages GPU against CPU reaches ~1.8e-4 at
143000's deep layers (`p1e_energy_field/status-1e.md` "The 8 long passages"), above the smallest
responses B reads. **Floor:** the two base passes' max `|Δ|` per readout; a readout whose `|A|` is
under 10× the floor at some ε is marked *below floor* there and not ranked. Deterministic
algorithms on; a run records device, dtype, torch and transformers versions.

### The calibration gate (before any real reading; CLAUDE.md "Calibration gate")

Run on B1's design points (below), at all four ε:

| arm | planted | passes when |
|---|---|---|
| G1, arithmetic | `d` at layer ℓ, read at layer ℓ | `A = εd` and `S = 0` at every token of T to float tolerance (the hook adds what it says) |
| G2, linear response | `d` random, read at ℓ + 1 | on **CPU float64 throughout** (the ± passes and `torch.func.jvp` of the same block on one device, so no cross-device comparison): at c = 0.01, `A/ε` agrees with the JVP to ≤ 5 % relative at ≥ 90 % of tokens; reported at every c, so the plateau's departure is seen, not assumed away. The GPU run's own check is G2′: its ± passes against its own base, the floor below |
| G3, planted effect | `d = ∇ₓ L_i` (the next-token loss at position i, w.r.t. `resid_pre` at ℓ, token i), normalised | the loss's `A/ε` is within 10 % of `‖∇ₓ L_i‖` at c = 0.01, and `d` ranks first of K + 1 on `|A|` of the loss at every c ≥ 0.03 in ≥ 6 of 7 prompts. Near-guaranteed while the response is linear: it checks the harness's arithmetic and ranking, not its power |
| G4a, structural null | nudge token j, read the loss and the state at every position i < j | `A = S = 0` exactly (causality): an arm that fails if the harness leaks across positions |
| G4b, structured null | `d = ∇ₓ L` of a token in **another prompt** (a structured, on-distribution direction unrelated to this token) | over 7 prompts × 32 positions, its rank among the K random directions on the loss's `|A|` rejects (rank 1) at ≤ 1/(K+1) + binomial 2σ at every c. *Changed after `/challenge-pr` on #197, finding 2: the first G4 drew one random direction against the others, which rejects at 1/33 whatever the harness does* **Changed after output (Blocked 43 (a), the user's, 2026-10-14):** at step 143000 the rate failed (17/224) and the gradients alone gave the same 17, so the pass is now: at c = 0.01 the nudged rank-1 set equals the linear response's (\|∇ₓL_j·d\| among \|∇ₓL_j·R_k\|, no nudge), at every step; the rank-1 rate is reported, not judged (`status-B.md` "B0 (2026-10-14)") |

If any arm fails, B0 is not done and B1 does not run. **One named exception** (Blocked 44 (a),
the user's, 2026-10-14): G4b's arm at step 128, `camus_letranger` j = 82, code `1e78e2be9d94`,
records `2026-10-14c`, a near-tie (`status-B.md` "B0 over 18 steps"), is waived. The waiver is
that arm only: B1 refuses any `gate.json` with another failure, another failing arm, or other code
(`b1_prev_token.check_gate`). It is not a tie clause; the next gate on other code gets none.

### B0 as built (2026-10-14, fixed in `trackB_circuit/b0_gate.py`'s docstring before any output)

| where the text left a choice | taken | why |
|---|---|---|
| the mechanism ("a forward pre-hook on block ℓ") | `core.nudge.Stack` runs the blocks in its own loop from a cached block input; the nudge is added to block ℓ's input there | one mechanism gives the nudge, truncation and block-only passes (G2's JVP, B1's cost); a hook on the HF forward gives only the first |
| that loop is the model | **G0 (added):** the loop's logits against the HF forward, max \|Δ\| ≤ 1e-5 at every (step, prompt) | the loop replaces the forward, so it is gated too |
| "base run twice" | every batch carries a base row; the four c's batches of one comparison are four same-shape base passes; floor = their max pairwise \|Δ\| | the floor costs no extra pass. Under deterministic algorithms it is expected to be 0, and then nothing is below floor; that is reported, not assumed away |
| G1's tolerance | at j, ‖A − εd‖∞ and ‖S‖∞ ≤ 8·u₃₂·(‖x_j‖∞ + ε); off j, exactly 0; all 32 positions × 32 directions × 4 c | rounding of one add and one subtract |
| G2's design points | positions J[0:4], direction R[0], tokens i ≥ j pooled over the 7 prompts | tokens before j have Jd = 0 exactly |
| G3's "≥ 6 of 7" | both clauses (10 % at c = 0.01, rank 1 at every c ≥ 0.03) per prompt, at i = J[0] | |
| G4b's "another prompt" | the next prompt in `V1_SCORED` (cyclic), same-index position | fixed, not chosen per pair |
| loss-only reads | the sequence is cut at j + 1 tokens | causal: exact for every kept position (G4a checks it) |
| B0's cost | ~10 GPU-minutes per step, ~3 h for 18 (G4b's 224 × 4 batches of 67 rows dominate); G2 on CPU beside it | the units table's "GPU minutes" was per step |

## B1, `L5H2` as positive control

**Signature** (ROADMAP §7): each token pulled toward its left neighbour inside `L5H2`'s output
subspace. **Measured as:** nudge token `j`'s `resid_pre` at layer 5 by `±εd` (K = 32 random `d`,
the four ε); read **`L5H2`'s own output at token i** (its per-head attention output after its
`W_O` slice, hooked inside block 5), for `j = i − k`, `k ∈ {0, 1, 2, 3, 4, 5}`. Forwards are
truncated after block 5. Per (step, prompt, i, k): `σ_k = median_d ‖A‖ / ε` over the K directions.

- **Statistic:** `s = log(σ₁ / median(σ₂ … σ₅))`, with `σ_k` read at `i = j + k` from one nudge of
  token `j`; the median over 32 nudge positions `j` (drawn once, seeded, from `j ≥ 8` with `j + 5`
  inside the prompt; the same positions at every step), **at c = 0.03**, fixed here; the other
  three c beside. No position correction: `s` is a ratio of offsets at the same position, and a
  per-head intercept would centre every head's median at zero (`/challenge-pr` on #197, finding 3).
- **Null for "on":** the same statistic for **every other head of layer 5** (15 heads) at that
  step, prompt and c. A step is **on** when `L5H2`'s `s` exceeds **at least 14 of the 15** in
  ≥ 6 of 7 prompts; **leans** at 5 of 7; otherwise off. "14 of 15" because layer 5 has a second
  previous-token head, `L5H9` (`data/analysis/prev_token_profile.json`, 0.55–0.62 against
  `L5H2`'s 0.93–0.97, quoted by the review; finding 4), known before this rule was fixed.
- **What `s` is, to first order:** `L5H2`'s offset −1 attention ratio times its OV gain, which the
  stored attention arrays already carry. B1 is an instrument check because that answer is known.
- **Beside, not labelled:** `cos(A, W_OV^{L5H2} d)` at k = 1 (is the response the OV-mapped nudge);
  the yardstick in **residual units**, `‖A‖` of `L5H2`'s contribution to the residual at token i
  over the RMS token norm at block 5's output, the units B2–B4 read.
- **Inputs:** `pythia-410m`, 1e's 18 steps (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000,
  4000, 8000, 16000, 32000, 54000, 143000; Stage 0's index also has step 1, not read), the 7 scored v1 prompts (hash pinned at run
  time from `core/prompts.py`), not the held-out 12; GPU float32.
- **Pass (exit):** off at step 0, and on at every step from 1000 through 8000. *Changed after
  `/challenge-pr` on #197, finding 1: the first rule also failed on "on at ≤ 512", but
  2607.06621 reports previous-token behaviour already at 0.37 at step 512, so a working
  instrument could read 256–512 as on or lean.* Steps 2–512 place the onset; they are recorded,
  not judged. **Recorded before any reading** (`lit-B.md` §1, 7d): the onset falls in
  (256, 1000]. A late fade (7d: 15 % of the causal effect retained at 143000) is expected and
  does not fail the pass; it is reported.
- **Fail:** on or leaning at step 0 (the instrument reads something `L5H2` is not), or off or
  leaning anywhere in 1000–8000.

### B1 as built (2026-10-14, fixed in `trackB_circuit/b1_prev_token.py`'s docstring before any output)

| where the text left a choice | taken | why |
|---|---|---|
| "hooked inside block 5" | a pre-hook on block 5's `attention.dense` takes the merged heads; head h's output is its 64 columns times its slice of `W_O` (bias shared, cancels in A) | NeoX merges heads as [h, e], so the slice is exact |
| the passes | per (prompt, j, c): one batch of 65 rows (base, ±ε·R_k at j), block 5 only, sequence cut at j + 6 | causal, exact for i ≤ j + 5; B0's design points (`b0_gate.site`), so the gate covers them |
| the rounding floor (`status-B.md` Fences) | per (prompt, j): max(‖Δ‖ between the batch's base row and the base alone (batch 1), u₃₂·max‖y‖); σ_k below floor when median ‖A‖ ≤ 10× it; a prompt cannot count as on where `L5H2`'s σ₁ is below floor at > 16 of 32 j | another batch shape measures rounding where the same shape gives 0; one ulp is the minimum |
| `W_OV d` | `W_O W_V` of `L5H2` applied to block 5's input LayerNorm's exact JVP at x_j along d | the head reads LN(x), not x |
| the yardstick | median ‖A‖ of `L5H2`'s output at k = 1 over the RMS token norm of block 5's output, every c | the design's "residual units" |

## Units (each its own PR)

| unit | output | cost |
|---|---|---|
| 1. this design | `design-B.md`, `lit-B.md`, the `sympy` check | writing |
| 2. B0 | `core/nudge.py`, `core/position.py`, tests, the gate's record | GPU minutes |
| 3. B1 | `trackB_circuit/b1_prev_token.py`, record, `status-B.md` | the base pass caches block 5's input once per (step, prompt); each nudge of token j runs **block 5 only** and reads every offset `i = j + k` from that one pass: 18 steps × 7 prompts × 32 positions × 33 directions × 2 signs × 4 ε ≈ 1.1 M single-block passes, batched; minutes to an hour on the RTX 3080 (estimate, checked on the first step before the batch). *Corrected after `/challenge-pr` on #197, finding 5: the first estimate was for 6.4 M six-layer passes* |

## Fences

- No registration. `claims/registry.json` untouched; B1 is a positive control on an exploration
  rung.
- `P-AB1`'s producer is not built here; it is the harness's natural second use, after B1.

## Worth challenging

- **Activation site only.** ROADMAP's B0 row names weight-space nudges too; deferring them
  keeps B0's gate on one site. The alternative is to build both now and gate both.
- **The head-rank null in B1** compares `L5H2` with its own layer's heads, which may include a
  second previous-token head; then `L5H2` would fail to beat "all 15" while the instrument is
  right. The alternative null, offsets `k ≥ 2` only, cannot see a head that attends everywhere.
- **6 of 7 prompts** is the field's sign-test habit, not derived for this statistic.
- **32 random directions** sets the rank floor at ~0.03; more cost linearly.
- **ε relative to the RMS token norm** is dominated by the massive-activation tokens in later
  layers; at layer 5 that is smaller, but `r` per token would be the alternative.
