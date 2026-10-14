<!-- trackB_circuit/design-B.md -->
# Track B — design: the nudge harness (B0) and `L5H2` as positive control (B1)

Opened 2026-10-13 by the user's Blocked 40 (c) ("do a, b, and c in one go"), after Phase 10 and
1e closed and B-pre found no Phase 7 row adjudicable on 410m (`p7_motifs/status-7.md` "B-pre").
Plan: `docs/ROADMAP.md` §7 (B0, B1), §9 (order 2). Scan: `trackB_circuit/lit-B.md`. Tier 1
throughout: exploratory, unregistered; 410m is an exploration rung.

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
| G2, linear response | `d` random, read at ℓ + 1 | at c = 0.01, `A/ε` agrees with `torch.func.jvp` of the same block (CPU, float64) to ≤ 5 % relative, at ≥ 90 % of tokens; reported at every c, so the plateau's departure is seen, not assumed away |
| G3, planted effect | `d = ∇ₓ L_i` (the next-token loss at position i, w.r.t. `resid_pre` at ℓ, token i), normalised | the loss's `A/ε` is within 10 % of `‖∇ₓ L_i‖` at c = 0.01, and `d` ranks first of K + 1 on `|A|` of the loss at every c ≥ 0.03 in ≥ 6 of 7 prompts |
| G4, planted null | `d` one more random direction | over 200 leave-one-out draws from the K + 1 pool (no extra passes), the rank test rejects at ≤ 0.05 + its binomial 2σ, at every ε and readout |

If any arm fails, B0 is not done and B1 does not run.

## B1, `L5H2` as positive control

**Signature** (ROADMAP §7): each token pulled toward its left neighbour inside `L5H2`'s output
subspace. **Measured as:** nudge token `j`'s `resid_pre` at layer 5 by `±εd` (K = 32 random `d`,
the four ε); read **`L5H2`'s own output at token i** (its per-head attention output after its
`W_O` slice, hooked inside block 5), for `j = i − k`, `k ∈ {0, 1, 2, 3, 4, 5}`. Forwards are
truncated after block 5. Per (step, prompt, i, k): `σ_k = median_d ‖A‖ / ε` over the K directions.

- **Statistic:** `s = log(σ₁ / median(σ₂ … σ₅))`, position-corrected per prompt (`core/position.py`),
  then the median over 32 positions `i` (drawn once, seeded, from `i ≥ 8`; the same positions at
  every step) at the gate's c.
- **Null for "on":** the same statistic for **every other head of layer 5** (15 heads) at that
  step, prompt and c. A step is **on** when `L5H2`'s `s` exceeds all 15 in ≥ 6 of 7 prompts
  (sign-test floor 8/128 = 0.0625 per prompt count; the "all 15" rank has floor 1/16 per prompt).
  **Leans** at 5 of 7. Otherwise off.
- **Beside, not labelled:** `cos(A, W_OV^{L5H2} d)` at k = 1 (is the response the OV-mapped nudge),
  and `σ₁` in absolute units (the yardstick).
- **Inputs:** `pythia-410m`, the 18 Stage 0 steps, the 7 scored v1 prompts (hash pinned at run
  time from `core/prompts.py`), not the held-out 12; GPU float32.
- **Pass (exit):** on at every step from some step in (512, 4000] onward to at least 8000, and
  off at 0 and at every step ≤ 512. **Recorded before any reading** (`lit-B.md` §1): the expected
  onset is (512, 1000]. A late fade (7d: 15 % of the causal effect retained at 143000) is
  expected and does not fail the pass; it is reported.
- **Fail:** on at step 0 or ≤ 512 (the instrument reads something `L5H2` is not), or never on.

## Units (each its own PR)

| unit | output | cost |
|---|---|---|
| 1. this design | `design-B.md`, `lit-B.md`, the `sympy` check | writing |
| 2. B0 | `core/nudge.py`, `core/position.py`, tests, the gate's record | GPU minutes |
| 3. B1 | `trackB_circuit/b1_prev_token.py`, record, `status-B.md` | 18 steps × 7 prompts × 6 offsets × 32 positions × 33 directions × 2 signs × 4 ε of 6-layer forwards; batched on the GPU, ~1 h (estimate) |

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
