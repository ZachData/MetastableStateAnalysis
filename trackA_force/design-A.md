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
  (3) A1b: the reading. At the budget A1 closes at what it knows.

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
fixed step at t_max = 12 returns inf from β ≈ 7, where the time is finite):

| start g0 | n | β 0 | 1.6 | 3.5 | 5.6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| 0 (orthogonal) | 467 | 4.17 | 4.19 | 4.33 | 5.36 | 8.70 | 16.0 | 86.0 |
| 0 | 2041 | 4.91 | 4.92 | 4.96 | 5.28 | 6.33 | 8.66 | 31.3 |
| 0.3 | 467 | 1.52 | 1.52 | 1.53 | 1.56 | 1.60 | 1.67 | 2.03 |
| 0.6 | 467 | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 | 0.91 | 0.92 |

Two facts fixed here, before any reading: **the registered t* ≈ 4.2 is the time from an
orthogonal start**; a cloud already at mean cosine 0.3 needs ~1.5, at 0.6 ~0.9, at every β. And
from g0 = 0 the time is flat only to β ≈ 5.6 (the measured interval's top); per-head β above
the 0.05 R² floor at 143000 reaches 7.85 at p90 (`p1d_cluster_ensemble/status-1d.md` "β
refit"), where t* is 2–4× longer. So **where the cloud starts and how β is reduced over heads
decide the comparator**, and both are fixed below.

## A1a, the instrument (unit 2)

| part | fixed here |
|---|---|
| model, steps | `pythia-410m`, the 18 Stage 0 steps (`trackB_circuit/b0_gate.STEPS`) |
| prompts, primary | **the 7 scored v1 prompts** (`b0_gate.V1_SCORED`, hash pinned at run time), n ≤ 512 < d = 1024: inside Theorem 6.9's regime. The held-out 12 refused (`core/holdout.py`) |
| prompts, beside | 1e's 8 long passages (`LONG8_HASH`, n 1,032–2,041 > d): outside 6.9's regime, read as a continuity check, never in the verdict (see "Worth challenging" 1) |
| inputs | `activations.npz` (unit rows + norms, 25 states; state 24 is post-final-LN, so the step sum runs over blocks 0–22, block 23's move beside); `attentions.npz` for β (v1 only; all 18 steps exist on disk, 28 runs per step) |
| frame | **unit LN1 rows of the reading block, at both ends**: `u = unit(LN1_l(x))` with `FrameSpec(kind="ln_attn", reader_block=l)`; the step at block l is `f_l(x_{l+1}) − f_l(x_l)`, the field `X_β` is evaluated on `f_l(x_l)`, β is fit in the same frame. Position 0 is neither query nor key (as β's fit); 1c's raw-sphere frame beside |
| β | `core.beta_eff.estimate_beta_offset_fe` (`fe_full`, Blocked 9's convention), per head × step × prompt, from `attentions.npz`; **the layer's β = the median over its 16 heads**, the band's p10 and p90 as a bracket. At step 0 the fit is ≈ 0 (`beta_refit.txt`); it is used as fitted, not floored. F0b is this table's slope over steps |
| steps | four per block, per token then the median over tokens 1…n−1: `h_disp` (1c), `h_cal` (1c), **`h_proj = ⟨Δ_tan, X⟩/‖X‖²`** (new, the math's item 4), and `h_attn` (`h_cal` on attention's part of Δ only, the parallel residual's exact split) |
| `h_attn` input | a hooked GPU pass saving attention's per-token output per block (1e's `u2_attn.py` hook, all tokens, float32); one comparison per device (Track B's NUMERICS): every step definition for a (step, prompt) comes from the same pass or the same file |
| start | **g0 = mean pairwise cosine of `f_0(x_0)`** (the embedding in block 0's LN1 frame, position 0 out); t*(n, β, g0) from `tstar_table.tstar` |
| T_eff | the sum over blocks 0–22 of each step; reported per prompt and step, the median over the 7 prompts and their range |

### The calibration gate (before any real reading; `CLAUDE.md` "Calibration gate")

On synthetic clouds at the real design points (n of each v1 prompt, d = 1024, β ∈ {0, 3.5, 7}):

| arm | planted | passes when |
|---|---|---|
| G1, planted effect | 23 forward-Euler steps of the identity-weight SA field (V = I, the causal mask) with known `h_l` (log-uniform in [0.02, 0.5], seeded), from orthogonal and from g0 = 0.3 starts | `h_proj` and `h_cal` recover each `h_l` within 5 % and T within 3 %, at every n and β |
| G2, planted null | the same `Δ` norms per token, directions random in the tangent plane | `T_proj` within ±0.1·T of 0 (its sign is the reading); `h_cal` unchanged from G1 (it cannot see direction: reported as the arm's point) |
| G3, mixed | `Δ = hX + ξ`, ξ random tangent with `‖ξ‖ = ‖hX‖` | `h_proj` within 5 % of h; `h_cal` ≈ √2·h (the math's item 4) |
| G4, frame | G1's cloud passed through a real LN1 (step 143000's block 5 weights) before unit-normalising | `T_proj` within 10 % of T (the LN's centring and gain move the frame; this arm measures by how much) |

If any arm fails, A1a is not done and A1b does not run. No positive control exists for real
collapse on disk (ALBERT and random GPT-2 are B0′, not cached); recorded as a gap, not filled.

## A1b, the reading (unit 3)

**The family, fixed before output:** 18 steps × 7 v1 prompts; the statistic is `T_proj` and
`T_cal` against t*(n, β_layer-median, g0) per (step, prompt).

- **Reading per (step, prompt):** *below* if both `T_proj` and `T_cal` < t* at the bracket's
  p10 β and at g0 (t* rises with β, so this is the shortest t* the bracket allows, the hardest
  to be below: the comparator least favourable to P-gamma2); *not below* if both ≥ t* at the
  p90 β; else *straddles*. **"Far below"** (the registered "≪") is `T < t*/2`, this design's number, not the
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
| 1 | this design, `lit-A.md`, `a1_collapse_time.py`, `tstar_table.py` | this PR; freeze is Blocked 45 |
| 2 | A1a: β over 18 steps (CPU, `attentions.npz`), the hooked pass for `h_attn` (GPU), `h_proj` in `integration_time.py`, the gate | after the freeze |
| 3 | A1b: the reading, H-WARP and 1e's question beside | after A1a's gate |

## Fences

- **No registration and no adjudication.** 410m is an exploration rung; A5 and the reserved rung
  (ROADMAP §10.7) decide where `P-gamma2` is first scored.
- **The MLP.** The paper's dynamics have no FFN; `h_cal` and `h_proj` take the whole block's
  move, `h_attn` only attention's. 1e found the MLP supplies most of the shared update to 1000
  (`status-1e.md` "U2's per-head arm"), so the gap between `T_attn` and the others is read, not
  assumed.
- **LR warmup** overlaps 512 and 1000–2000 (ROADMAP §6 risks): any break in T there is reported
  with that confound beside.

## Worth challenging (for the freeze)

1. **v1, not the long passages, as primary.** The user moved 1e to long passages (2026-10-06);
   A1 goes back to v1 because Theorem 6.9 needs d ≥ d*(n, β) and the long passages have n > d.
   The alternative is long passages primary with the theorem's regime as a caveat.
2. **The start g0 at the embedding.** The other choice is g0 = 0 as registered; the t* table
   shows that choice alone can move t* 3–5×, so both are reported and the measured one rules.
3. **`h_proj` as co-primary.** It is new; 1c's three are what the registry names. Without it,
   motion off the field counts as integration time.
4. **The layer's β as the head median.** Alternatives: the per-head mean of t*, or a weight per
   head by its share of the update (not yet measurable β-free, `status-1d.md` Parked).
