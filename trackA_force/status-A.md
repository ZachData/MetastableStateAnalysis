<!-- trackA_force/status-A.md -->
# Track A — STATUS

No card yet (`docs/phase_card.md`); one is added with A1b's reading.

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check, t* table | **done 2026-10-14** (#201); frozen 2026-10-14 (Blocked 45, with the OV attractive share row beside the reading) | `design-A.md`, `lit-A.md`, `tools/math_checks/a1_collapse_time.py`, `tstar_table.py` |
| 2. A1a, the β producer over 18 steps and the T_eff instrument, with its gate | **built 2026-10-14; the gate fails (G3, G4); the real batch has not run** (Blocked 46) | `a1a.py`, `a1a_gate.py`, `p1c_frames/integration_time.unit_steps`; "A1a's gate" below |
| 3. A1a's re-gate (Blocked 46 (b)) and the chain re-gate (Blocked 47 (d)) | **done**: the re-gate failed G4 at 3 frames (#203); 47 (d)'s gate: G0–G4c pass, G2s fails on the drift-removed read (38 of 42), its planted drift not a null (#204; Blocked 48) | "A1a's gate, Blocked 47 (d)", "A1a's re-gate" below |
| 4. The gate with G2s beside (Blocked 48 (a)) and A1a's batch | **done 2026-10-15: the gate passes; the batch ran, 126 of 126 records** | "A1a's gate and batch, Blocked 48 (a)" below |
| 5. A1b, the reading (H-WARP and 1e's question beside) | not started; next | `design-A.md` "A1b" |

Budget: 5 units on the record since Blocked 48 (a) (`design-A.md` "Exit condition and budget";
first 3, then 4 by Blocked 46 (b)). Units 1–4 spent; A1b is the last.

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

**The batch:** 126 of 126 records. First record (143000, `camus_letranger`) populated: every
head's β finite at all 18 steps; `fe_full` at 143000 matches 1d's refit within 1.4e-14 (all 7
prompts); the hooked pass against the stored unit rows ≤ 2.6e-7 to step 2000, rising to 7.9e-5
at 143000 (the GPU against the CPU, as 1e's probe found); causal t* exists at both ends of the
bracket in all 126 cells. **The layer-β bracket** (p10 / p50 / p90, median over prompts) sits at
about ±0.1 to step 128, widens at 256–4000 (p10 down to −2.47 at 2000), and turns positive from
16000 (143000: 1.14 / 3.15 / 6.10). **The p10 is negative at every step to 8000.** The OV
attractive share falls from 0.50 to 0 at 1000–4000 and recovers to 0.39 at 143000.

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

- *The 2D spike* (the user's, 2026-10-14): homed as ROADMAP §7 B2a with its exit and budget;
  its slot (before A1b, or B2a after A1 closes) is picked at A1a's close.
- *G0–G3 on the GPU* (2026-10-15): they stay numpy on the CPU (~5 of the gate's 6 min); a
  torch port like G4c's would cut a re-gate to ~2 min. Why: the user's GPU rule; cost: half a
  unit with its own numpy check; changes no decision, only wall time.
- *t\* at a negative β* (2026-10-15, A1a's batch): the bracket's p10 is below 0 at every step to
  8000, and the reading's *below* uses t* at p10 as the shortest the bracket allows (t* rising
  with β). Why: t* exists there, but whether `sa_sim`'s collapse at β < 0 means what (6.9)'s does
  was never checked; cost: a look in A1b before its output; could change which cells read
  *below* to 8000. Carried to A1b.

## Corrections received

(none)
