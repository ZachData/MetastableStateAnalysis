<!-- trackA_force/design-A.md -->
# Track A — design: A1, does the trained net integrate far enough to collapse?

Opened 2026-10-14 by the user's Blocked 40 (d) ("Track A next session"), after Phase 10, 1e and
Track B's opening (B0, B1) closed. Plan: `docs/ROADMAP.md` §6 (A1), §9. Scan:
`trackA_force/lit-A.md`. Tier 1 throughout: exploratory, unregistered; 410m is an exploration
rung, so A1 gives `P-gamma2` a **number**, not an adjudication (that is A5 / ROADMAP §10.7).
`claims/registry.json` is untouched.

**What A1 carries in** (each home row in ROADMAP §6 names it): Phase 10's F0b (the `beta_eff`
slope across training, `p10_cluster_function/handoff-10.md` "At close (10.5)"); Stage 4's H-WARP
(`handoff-10.md` "Stage 4"); 1e's "is one well at β 3.5 what 2312.10794 predicts at n ≈ 2,000,
d = 1,024" (`p1e_energy_field/status-1e.md` "The phase on one page"); 1c's instrument A
(`p1c_frames/integration_time.py`, validated on synthetic data, never run on real data:
`p1c_frames/status-1c.md` card).

## Exit condition and budget

- **Exit:** `P-gamma2` has a number on 410m (T_eff per prompt and step, against t* at the
  measured start, with the rule below saying *below*, *straddles* or *not below*), or a recorded
  reason it cannot. H-WARP and 1e's question are read beside it if the budget allows, else carried.
- **Budget: 3 units.** (1) this design, the scan, the `sympy` check and the t* table; (2) A1a:
  the β producer over 18 steps and the T_eff instrument extended, with its calibration gate;
  (3) A1b: the reading. At the budget A1 closes at what it knows. **Extended to 4 units on the
  record (Blocked 46 (b), the user's, 2026-10-14):** unit 3 is the re-gate and A1a's batch,
  unit 4 is A1b.

## The math

`tools/math_checks/a1_collapse_time.py` (`sympy`, all pass): (1) at β = 0, eq. (6.9) is
`dg/dt = 2(1−g)((n−1)g+1)/n` and `t*(g) = ½ log(((n−1)g+1)/(1−g))`, so t*(0.9) = 4.17 at
n = 467, 4.91 at n = 2041; (2) g = 1 and g = −1/(n−1) are fixed at every β; (3) as β → ∞ the
speed at g < 1 falls like `e^{−β(1−g)}`, so t* from g = 0 grows exponentially in β; (4) for a move
`dx = hX + ξ`, ξ ⊥ X, the projected step `⟨dx, X⟩/|X|²` is h exactly while 1c's calibrated step
`|dx|/|X|` is `√(h² + |ξ|²/|X|²)`: **the calibrated step counts motion off the field as time.**
**What it does not prove:** that a real cloud starts pairwise orthogonal (Theorem 6.8's premise),
that d ≥ d*(n, β) (Theorem 6.9; false for the long passages, n > d), that one β stands for 16
heads, or that the MLP's move belongs in the sum.

## t* before any output (no model, no data)

`python -m trackA_force.tstar_table` (eq. (6.9), adaptive solver; `gamma_ode.collapse_time`'s
fixed step at t_max = 12 returns inf from β = 8, where the time is finite; β 7 it gives 8.70):

| start g0 | n | β 0 | 1.6 | 3.5 | 5.6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 0 (orthogonal) | 467 | 4.17 | 4.19 | 4.33 | 5.36 | 8.70 | 16.0 | 86.0 |
| 0 | 2041 | 4.91 | 4.92 | 4.96 | 5.28 | 6.33 | 8.66 | 31.3 |
| 0.3 | 467 | 1.52 | 1.52 | 1.53 | 1.56 | 1.60 | 1.67 | 2.03 |
| 0.6 | 467 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.92 |

Two facts fixed here, before any reading: **the registered t* ≈ 4.2 is the time from an
orthogonal start**; a cloud already at mean cosine 0.3 needs ~1.5, at 0.6 ~0.9, at every β. And
from g0 = 0 the time is flat only to β ≈ 5.6 (the measured interval's top); per-head β at 143000
(`fe_full`, no R² floor) reaches 7.11 at p90 in L1–8 (`p1d_cluster_ensemble/status-1d.md` "β
refit"; `beta_refit.txt`), where t* is ~2× longer.

**(6.9) has no causal mask; Pythia does** (`/challenge-pr` on #201, finding 2). The identity-weight
dynamics simulated on a cloud (`python -m trackA_force.sa_sim`: n = 467, d = 1024, Gaussian
start, seed 0, Euler dt 0.01, time to mean cosine 0.9):

| β | 0 | 3.5 | 5.6 | 7 |
|---|---|---|---|---|
| full attention (known answer: (6.9) 4.17, 4.33, 5.36, 8.70) | 4.25 | 4.41 | 5.45 | 8.80 |
| **causal** | **3.82** | **5.20** | **14.8** | **46.0** |

The full-attention column reproduces (6.9) within 2 % (Euler's bias at dt 0.01 and a Gaussian
start's |cos| ~ 0.03), which checks the simulator; the causal column is the comparator Pythia
needs, and at the bracket's top it is ~3× (6.9)'s. So **where the cloud starts, the mask, and how
β is reduced over heads decide the comparator**, and all three are fixed below: **t* is the
causal simulator's time from the measured cloud itself** (no equal-cosine start assumed, which
also answers finding 3), the (6.9) table beside as the registered comparator.

## A1a, the instrument (unit 2)

| part | fixed here |
|---|---|
| model, steps | `pythia-410m`, the 18 Stage 0 steps (`trackB_circuit/b0_gate.STEPS`) |
| prompts, primary | **the 7 scored v1 prompts** (`b0_gate.V1_SCORED`, hash pinned at run time), n ≤ 512 < d = 1024: n < d is necessary for Theorem 6.8's orthogonal start; whether d ≥ d*(n, β) (6.9's bound) holds at n ≈ 467 is not known here. The held-out 12 refused (`core/holdout.py`) |
| prompts, beside | 1e's 8 long passages (`LONG8_HASH`, n 1,032–2,041 > d): outside 6.9's regime, read as a continuity check, never in the verdict (see "Worth challenging" 1) |
| inputs | `activations.npz` (unit rows + norms, 25 states; state 24 is post-final-LN, so the step sum runs over blocks 0–22, block 23's move beside); `attentions.npz` for β (v1 only; all 18 steps exist on disk, 28 runs per step) |
| frame | **unit LN1 rows of the reading block, at both ends**: `u = unit(LN1_l(x))` with `FrameSpec(kind="ln_attn", reader_block=l)`; the step at block l is `f_l(x_{l+1}) − f_l(x_l)`, the field `X_β` is evaluated on `f_l(x_l)`, β is fit in the same frame. Position 0 is neither query nor key (as β's fit); 1c's raw-sphere frame beside |
| β | `core.beta_eff.estimate_beta_offset_fe` (`fe_full`, Blocked 9's convention), per head × step × prompt, from `attentions.npz`; **the layer's β = the median over its 16 heads**, the band's p10 and p90 as a bracket. At step 0 the fit is ≈ 0 (`beta_refit.txt`); it is used as fitted, not floored. F0b is this table's slope over steps |
| steps | four per block, per token then the median over tokens 1…n−1: `h_disp` (1c), `h_cal` (1c), **`h_proj = ⟨Δ_tan, X⟩/‖X‖²`** (new, the math's item 4), and `h_attn` (`h_cal` on attention's part of Δ only, the parallel residual's exact split) |
| `h_attn` input | a hooked GPU pass saving attention's per-token output per block (1e's `u2_attn.py` hook, all tokens, float32); one comparison per device (Track B's NUMERICS): every step definition for a (step, prompt) comes from the same pass or the same file |
| start and t* | **the measured cloud `f_0(x_0)`** (the embedding in block 0's LN1 frame, position 0 out) is the simulator's start; **t* = `sa_sim.tstar(f_0(x_0), β, causal=True, check_dt=True)`** to mean cosine 0.9, at the layer-median β of each bracket end. Beside: (6.9)'s t* from g0 = that cloud's mean cosine and from g0 = 0 (registered). Also beside: g0 in the affine-free LN frame (centre and scale only), since LN1's learned bias is 0 at step 0 and moves the start over training for a reason that is not the embedding (finding 3) |
| T_eff | the sum over blocks 0–22 of each step; reported per prompt and step, the median over the 7 prompts and their range |

### The calibration gate (before any real reading; `CLAUDE.md` "Calibration gate")

On synthetic clouds at the real design points (n of each v1 prompt, d = 1024, β ∈ {0, 3.5, 7}):

| arm | planted | passes when |
|---|---|---|
| G0, comparator | `sa_sim` with full attention from a Gaussian start, at each design n and β ∈ {0, 3.5, 5.6} | within 3 % of (6.9) (passes now at n = 467: ≤ 2 %, "t* before any output"); `check_dt` converged on the causal arm |
| G1, planted effect | 23 forward-Euler steps of the identity-weight SA field (V = I, the causal mask) with known `h_l` (log-uniform in [0.02, 0.5], seeded), from orthogonal and from g0 = 0.3 starts | `h_proj` and `h_cal` recover each `h_l` within 5 % and T within 3 %, at every n and β |
| G2, planted null | the same `Δ` norms per token, directions random in the tangent plane | `T_proj` within ±0.1·T of 0 (its sign is the reading); `h_cal` unchanged from G1 (it cannot see direction: reported as the arm's point) |
| G3, mixed | `Δ = hX + ξ`, ξ random tangent with `‖ξ‖ = ‖hX‖` | `h_proj` within 5 % of h; `h_cal` ≈ √2·h (the math's item 4) |
| G4, frame | G1's cloud passed through a real LN1 (step 143000's block 5 weights) before unit-normalising | `T_proj` within 10 % of T (the LN's centring and gain move the frame; this arm measures by how much) |

If any arm fails, A1a is not done and A1b does not run. No positive control exists for real
collapse on disk (ALBERT and random GPT-2 are B0′, not cached); recorded as a gap, not filled.

### A1a as built (2026-10-14), where the text left a choice

`trackA_force/a1a.py` (the batch), `a1a_gate.py` (the gate), `integration_time.unit_steps` (the
per-token steps). Recorded here, before any real output:

- **Inputs:** the one `data/phase12` directory per step that holds all 7 v1 prompts with
  attentions (`a1a.RUNS`, checked: exactly one each). n = 242–512, so 241–511 tokens read.
- **The bracket:** p10 / p50 / p90 over the 23 layer βs of blocks 0–22 (each the median over its
  16 heads), per (step, prompt). The field at block l uses that block's own layer β.
- **β's rows** come from the stored `activations.npz`, the steps from the GPU pass (checked against
  the stored unit rows at 1e-4, `u2_attn.check_pass`); at 143000 `fe_full` must match 1d's refit
  where the inputs coincide (1e-4) or the batch refuses.
- **Block 23's end** is rebuilt as `x_23 + attn + mlp` (state 24 is post-final-LN); beside, as designed.
- **`attn_proj`** (the projected step on attention's part) is recorded beside `h_attn` (= `attn_cal`).
- **t\*:** `sa_sim` horizon 200 (an infinite t* is "not read"); `check_dt` on every cell.
- **Gate:** the orthogonal start is exactly orthonormal (QR), the 0.3 start has every pairwise
  cosine exactly 0.3; G3's ξ is ⊥ X exactly; 23 steps, h seeded per cell (`SEED` 20261014).
- **The batch refuses** without a passing gate record for its code and lift (`a1a.check_gate`).

### The gate's first run (2026-10-14): fails, G3 and G4 (STATE Blocked 46)

The numbers are in `trackA_force/status-A.md` "A1a's gate (2026-10-14)" (one copy). G0–G2 pass.
**G3** fails on its largest steps by the tangent lift's second-order bias, which this design did
not compute before fixing its bar; the log lift passes. **G4** fails from orthogonal starts at high
β, always low, with either lift; it plants raw-frame dynamics, while the reading's field is on
LN1 rows (`/challenge-pr` on #202, finding 1). The real batch has not run; nothing real was read.

### Blocked 46 decided (2026-10-14, the user's: (b), a 4th unit on the record)

Fixed before the re-gate's output (`a1a_gate.py` docstring is the spec):

- **G4 re-planted as a V = I head acts in the reading's frame**: the field on `unit(LN1(x))`, the
  raw move `h·|Cx|·X` added to the raw residual, read back through the same LN1. The scale makes
  the affine-free frame G1 exactly (tested). The radial part a literal V = I head would add
  (`⟨U, ΣAU⟩U`) is left out, so G4 isolates LN1's w and b. Bar unchanged (T within 10 %), on block
  5 at 143000 for all 42 cells **and** on every block 0–22 × step LN1 (432 frames, block 23
  beside) for `latex_monograph`'s six (β, start) cells (the first G4's lowest prompt). The first
  G4 is kept beside as `G4raw`.
- **Lift: log** (arc length, the continuous time t* counts); its leading bias ≈ |V|²/3, quoted
  beside G3's 5 % bar. Gnomonic and tangent are read beside from the same trajectories.
- **G2s, shared drift, beside G2 (not gated)**: the same norms per token, all toward one fixed
  random direction; reported as `T_proj/T`. No bar: at positive mean cosine the field points at
  the mean, so a shared drift projects onto it by geometry; the arm says by how much.
- **`check_gate`** also requires the gate file's own hash (`gate_code`).
- If the gate passes, A1a's batch runs (unchanged otherwise); if it fails, A1 goes back to the
  user before any real output.

**Ran 2026-10-14: fails on G4 at 3 frames** (step 143000, blocks 20–22; numbers in `status-A.md`
"A1a's re-gate"); back to the user before any real output (STATE Blocked 47).

## A1b, the reading (unit 3; unit 4 after Blocked 46)

**The family, fixed before output:** 18 steps × 7 v1 prompts; the statistic is `T_proj`
against the causal t* from the measured cloud per (step, prompt).

- **Reading per (step, prompt), on `T_proj` alone** (`/challenge-pr` on #201, finding 1: since
  `h_cal ≥ |h_proj|`, requiring both lets `T_cal`, the step that counts off-field motion, decide):
  *below* if `T_proj` < t* at the bracket's p10 β (t* rises with β, so this is the shortest t*
  the bracket allows, the comparator least favourable to P-gamma2); *not below* if `T_proj` ≥ t*
  at the p90 β; else *straddles*. A t* that is infinite or refused (start at or above 0.9) makes
  the cell *not read*, never *below*.
- **Rows beside, not in the reading:** `T_cal`, `T_disp`, `T_attn`, and the off-field share
  `1 − T_proj / T_cal` (large = the net moves a lot, but not along this field: the case that
  separates "resists" from "too shallow"). **"Far below"** (the registered "≪") is `T < t*/2`, this design's number, not the
  registry's.
- **Per step:** the reading of ≥ 6 of 7 prompts, else *mixed*.
- **P-gamma2's number:** the median ratio `T_proj / t*` per step, with its prompt range; and the
  same at the registered comparator (g0 = 0, t* from n and β), beside, so the effect of the start
  is visible.
- **Collapse beside:** the measured mean cosine at state 23 in the same frame (P-gamma2's
  falsifier needs "no collapse observed": ≥ 0.9 there is collapse).
- **H-WARP (beside, if the budget allows):** per step, the per-block mean cosine against cumulative
  `T_proj`; a pooled isotonic fit over the other 17 steps predicts the left-out one; the warp
  *collapses the curves* if the leave-one-step-out error on the T axis is ≤ ½ of the same on the
  depth axis in ≥ 12 of 18 steps. Null: each step given another step's T profile (17 draws).
- **1e's question (beside, if the budget allows):** set 1e's one well against (6.9) at the
  measured g0 and β: at β ≤ 5.6 the trajectory is one sigmoid (no plateau), so one well is the
  identity-weight dynamics' own answer at that β; multi-cluster metastability (`lit-A.md` §1)
  needs β past the measured range. A1b states which β the long passages would need; no new run.

## Units (each its own PR)

| unit | what | state |
|---|---|---|
| 1 | this design, `lit-A.md`, `a1_collapse_time.py`, `tstar_table.py`, `sa_sim.py` | done (#201); frozen 2026-10-14 (Blocked 45) |
| 2 | A1a: β over 18 steps (CPU, `attentions.npz`), the hooked pass for `h_attn` (GPU), `h_proj` in `integration_time.py`, the gate | built 2026-10-14; **the gate fails (G3, G4)**, the batch not run; Blocked 46 |
| 3 | A1a's re-gate (Blocked 46 (b)) and batch | re-gate ran 2026-10-14, **fails on G4 at 3 frames**; Blocked 47 |
| 4 | A1b: the reading, H-WARP and 1e's question beside | after A1a's gate |

## Fences

- **No registration and no adjudication.** 410m is an exploration rung; A5 and the reserved rung
  (ROADMAP §10.7) decide where `P-gamma2` is first scored.
- **The MLP.** The paper's dynamics have no FFN; `h_cal` and `h_proj` take the whole block's
  move, `h_attn` only attention's. 1e found the MLP supplies most of the shared update to 1000
  (`status-1e.md` "U2's per-head arm"), so the gap between `T_attn` and the others is read, not
  assumed.
- **LR warmup** overlaps 512 and 1000–2000 (ROADMAP §6 risks): any break in T there is reported
  with that confound beside.
- **`P-gamma2`'s registered h0 reads against its statement** (`/challenge-pr` on #201, question
  2): h0 says "T_eff comparable to t*, so non-collapse is explained by depth", but T_eff ≈ t*
  without collapse is resistance, the falsifier's own case. A1 reads the statement and the
  falsifier; amending h0 is a registry edit, so it is the user's (STATE Blocked 45).

## Frozen 2026-10-14 (Blocked 45, the user's: as recommended)

- **Frozen as revised**, with one row added beside the reading (not in it): **the layer's OV
  attractive share**, per block and step, read off `data/analysis/ov_per_head_series.json`
  (weights only, no new run): the median over the layer's 16 heads of
  `attractive_energy_fraction_core`, and the share of heads at exactly 0 (`PROJECT.md` §3.12 A).
  Why: t* assumes V = I (pure attraction), while at 1000–2000 62 % of heads have no attractive
  part at all; the row shows where that assumption is furthest from the weights.
- **`P-gamma2`'s h0 is left as registered** until A5 registers; A1 reads the statement and the
  falsifier ("Fences").
- **The 2D spike** (the user's question of 2026-10-14: can a head's dynamics be read in the
  paper's d = 2 reduction, 2604.01978 §"General value maps and effective low-dimensional
  dynamics"?) runs **after A1a**, which supplies its two inputs (β per head, the step γ): either
  before A1b or as Track B's B2a after A1 closes, the slot picked at A1a's close. **Exit: a
  faithful D per named head against the null** (the smallest subspace whose reduced dynamics
  reproduce the head's real attention, for `L5H2`, the induction heads, `L10H7` and random
  reference heads; 2D is the circle case). **Budget: 3 units** (weights only with a planted
  positive control and a null; real tokens at 143000; the heads that pass over 18 steps). Prior,
  before any output: 2D fails for the previous-token and induction heads (their attention reads
  position or exact identity). `docs/ROADMAP.md` §7 B2a.

## Worth challenging (for the freeze)

1. **v1, not the long passages, as primary.** The user moved 1e to long passages (2026-10-06);
   A1 goes back to v1 because Theorem 6.9 needs d ≥ d*(n, β) and the long passages have n > d.
   The alternative is long passages primary with the theorem's regime as a caveat.
2. **t* from the causal simulator on the measured embedding cloud.** The other choices are (6.9)
   at g0 = 0 as registered, or (6.9) at the measured g0; both are beside. The start alone moves
   t* 3–5×, the mask ~3× at the bracket's top. Least settled: the frame of the start (LN1 with
   its learned bias, β's frame), with the affine-free frame beside.
3. **`h_proj` alone decides the reading.** It is new; 1c's three are what the registry names and
   are reported beside. Without it, motion off the field counts as integration time.
4. **The layer's β as the head median.** Alternatives: the per-head mean of t*, or a weight per
   head by its share of the update (not yet measurable β-free, `status-1d.md` Parked).
