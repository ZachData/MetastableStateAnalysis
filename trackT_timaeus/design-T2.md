<!-- trackT_timaeus/design-T2.md -->
# Track T — design: T2, V and E_β on Dodecahedron (proposed 2026-10-29; frozen 2026-10-29; Blocked 120 (b) section proposed 2026-10-30)

Opened 2026-10-29 by `STATE.md` Blocked 118 (b), the user's: T2 tests the LR dose against step;
the user asks Timaeus for the global batch, and T2 tests tokens too if it is 524k. Row:
`docs/ROADMAP.md` §7b T2. The windows come from `status-T.md` "T1, the inventory" Q3, which stays
their one home; the bands and the outcome table below are printed by
`tools/math_checks/t2_grid.py` (6 checks pass; what it does not prove is in its docstring).
Scan: `docs/lit-T.md` (T1's, and T2's line, 2026-10-29: nothing published on Dodecahedron's
training dynamics); the schedule question's own scan is C2's (`trackA_force/lit-C2.md`,
`trackC_writeup/lit-C2-unit1.md`). Tier 1, description throughout. **Frozen 2026-10-29 on the
user's calls, D1–D4 all (a) (Blocked 119), before any Dodecahedron weight was downloaded**; the
"Decisions" table below keeps the options as written.

## What T2 is for

C2's fence F1: on Pythia the break's date (E₁'s at 128–512, the centred count's at 1000) cannot
be told from the LR warmup, because every Pythia size shares it. 3b (70m, 160m) left step and the
cumulative LR not told apart; 3c (GPT-2 small, log warmup) read "consistent with step", fragile on
its plateau window (`trackC_writeup/status-C.md` "C2 on one page", item 5). Dodecahedron is a
fourth schedule where the two come apart on a linear warmup: its peak LR is 11.7× Pythia's
(3.5e-3 against 3e-4) over a shorter warmup (1000 steps against 1430), so equal summed LR falls at
0.245 × Pythia's step. 53 checkpoints, log-spaced at about 1.17× a step, give a finer grid in the
window than Pythia's octaves.

**What it cannot show.** Architecture (Llama, RMSNorm, RoPE, GQA, tied embeddings), data
(FineWeb), tokenizer (16k BPE) and depth (16, 20 layers against 24) all differ, so a date that
matches no schedule is not read as one cause, and a match is description across models. The
schedule is the card's (warmup linear from 0, cosine after); no training log has been read.

## The mapping (fixed now)

A Pythia grid date g stands for a crossing in (prev(g), g]; mapped to Dodecahedron's steps and
read on its grid as 3c read GPT-2's (`trackC_writeup/design-C2.md` "Unit 3c"). 410m's dates are
`REF` as 3b and 3c: E₁'s break {256, 512}, the centred count {512, 1000} (10 seeds, union).

| schedule | Dodecahedron step for Pythia's t | E₁ band | centred band |
|---|---|---|---|
| instantaneous (equal LR level) | t × p_p W_d / (p_d W_p) = 0.0599 t | (7.7, 30.7] | (15.3, 59.9] |
| cumulative (equal summed LR) | t × √(p_p W_d / (p_d W_p)) = 0.245 t | (31.3, 125.4] | (62.7, 244.8] |
| fraction (of the warmup) | t × 1000 / 1430 | (89.5, 358] | (179, 699] |
| step | t | (128, 512] | (256, 1000] |
| tokens at 4,194,304 a step (leaning) | t / 2 | (64, 256] | (128, 500] |
| tokens at 524,288 a step (the card's figure) | 4 t | (512, 2048] | (1024, 4000] |
| *beside, not scored:* cumulative with the peak LR scaled by d_model / 1024 (μP's convention for hidden matrices; 384 and 512 against 1024) | 0.400 t (32m), 0.346 t (67m) | (51.2, 204.7], (44.3, 177.3] | (102.3, 399.8], (88.6, 346.2] |

Every warmup-dependent mapped step sits inside Dodecahedron's warmup (T3 of the check). "LR dose"
in the scored rows means the raw LR; the width-scaled row lands on step's band at 141–205 (E₁),
so a step reading does not exclude a width-scaled dose, and every reading below says "raw"
(`/challenge-pr` on #284, finding 2).

**The outcome table** (each Dodecahedron grid date → the schedules that predict it; others none):

| count | dates |
|---|---|
| E₁'s break | 8–30 inst.; 35 inst., cum.; 40–55 cum.; 65–88 cum., tok4M; 103–121 cum., frac., tok4M; 141 cum., frac., step, tok4M; 165–263 frac., step, tok4M; 308–359 frac., step; 420–491 step; 574 step, tok524k; 671–2340 tok524k; else none |
| centred | 16–55 inst.; 65 inst., cum.; 75–121 cum.; 141–165 cum., tok4M; 193–225 cum., frac., tok4M; 263 cum., frac., step, tok4M; 308–574 frac., step, tok4M; 671–785 frac., step; 917 step; 1072 step, tok524k; 1253–4369 tok524k; else none |

So **T2 separates step from cumulative** under either batch: E₁ at 35–121 or at 165–574 (one
shared date, 141); centred at 65–225 or 308–1072 (one shared, 263). It **cannot split fraction
from step** (Dodecahedron's warmup is 0.7 of Pythia's, so fraction is 0.7 × step; they part only
at 420–574 for E₁, 917–1072 centred). **At 4.19M, tokens overlaps cumulative and step** (the step
window one octave down), so a cumulative or step reading at 65–263 cannot exclude tokens; at 524k
its band lies past step's and is separable from all but step's last date.

## The reading (fixed now)

| item | fixed as | why |
|---|---|---|
| models | D1 | |
| the grid | the 53 released steps 1–9536; 9536 is the preflight's trained step only (below) | all the checkpoints there are |
| per model, per count | D4 (a)'s half-rise (A5) on this grid: the first step with V ≥ half the plateau median; beside, `sustained` (3c): the first step from which V stays ≥ half to 4369 | as 3c |
| the plateau | D3: early 1253–3738 unless tokens is scored (D4), then late 5108–8158; the other computed too, and the date set is the union of the two plateaus' dates | a later window reads a later break (A5 D4); a disagreement widens, as 3b's arms |
| unreadable | the plateau is 0, or V at step 1 or 2 is ≥ half of it | 92 (i)'s early check; every predicted date is ≥ 8 (T5) |
| the uncentred count | D2's 1e-3, as A5, 3b and 3c; its preflight report cited | refuse rather than degrade on a new model |
| the centred count | D2 | |
| the scored schedules | instantaneous, cumulative, fraction, step; **tokens at 524k only on D4's condition**; otherwise both tokens columns are printed beside, not scored | Blocked 118 (b) |
| **the reading** (`joint`) | the scored schedules consistent with every readable date (models × counts); all → **"not told apart"**; none → **"neither"**; else **"consistent with …"**. Beside: `with_3b` (the same less instantaneous, which 3b excluded), 3c's reading, and which tokens columns each date also fits | one statistic, as 3b and 3c |
| prompts, states | the 7 v1 prompts (hash at run time), Dodecahedron's tokenizer, ≤ 512 tokens; every hidden state HF returns (17 for 32m, 21 for 67m, the last after the final norm, as in GPT-NeoX and GPT-2: unit 1 checks) | C2's set and convention |
| numerics | GPU, float32 (weights stored in bf16 are upcast) | as A5 |

**What each outcome changes (fixed now).** None of it edits C2's paper, which is the user's
rewrite; T2's reading goes to `status-T.md` and the user decides any F1 sentence.

| reading | what it says |
|---|---|
| step (alone, or with fraction) | the dates track the updates applied across an 11.7× peak LR, agreeing with 3c; with 3b's, F1 narrows to "a count of updates (or the warmup's fraction, not split here), not the raw LR dose", description across models; a width-scaled dose is not excluded where the date fits its band. If tokens was not scored and a date also fits tok4M, that is said beside |
| cumulative without step | the dates track the raw LR dose, against 3c's step; F1 stands, the disagreement named with model and data as the confounds |
| tokens at 524k (scored) without step or cumulative | the dates track the data seen; F1 narrows to tokens, description |
| instantaneous alone | conflicts with 3b, which excluded it; F1 stands |
| not told apart, neither, unreadable | F1 stands; "neither" names the model and data |

## Calibration gate (unit 1, before the scored batch)

- **Known answer:** the producer pointed at `pythia-410m` reproduces seed 0's E₁ counts (0
  through 128, 19 at 256, 56 at 512, plateau median 57) and half-rise 512, so the
  model-agnostic path (hidden states, sphere, violations) is the one C2 used. On Dodecahedron,
  step 9536's record is populated (17 or 21 states, finite).
- **Planted effect:** per scored schedule, a synthetic V series on Dodecahedron's grid with its
  crossing at the middle of that schedule's band reads a set that contains it and excludes what
  the outcome table excludes.
- **Planted null:** crossings drawn uniform in log step over (1, 4369] through `joint`; the
  share of each reading by chance is reported beside the real one (it is not a p).
- **The centred arms (D2):** 3b's procedure per model: null-matched and B-matched arms derived on 3
  re-inits (HF's Llama init at the config's `initializer_range` 0.02, their weight SDs beside step
  1's, which is one update from init) and step 9536, beside 1e-4 and 3e-4; each gated as 3b
  (re-init rate ≤ 5 %, plant B ≥ 0.4 at the smallest dose the uncentred count sees at 0.4, plant
  A at 0.5 ≤ 0.1); frozen into `TAU_ARMS`; refuses if a re-run derives other arms.
- Any arm failing sends the design back to you before the batch.
- **Ran 2026-10-29 (unit 1):** known answer, planted effect and planted null pass; every centred
  arm fails on both models (plant A 0.21–0.51; 67m's null-matched τ ≤ 0, refused), and the
  uncentred count barely sees plant A (0.04–0.07), the unit cloud near one direction (mean cosine
  0.9); back to the user as `STATE.md` Blocked 120. Numbers: `status-T.md` "T2 unit 1".

## Decisions (each yours; recommendation first)

| # | choice | options | recommended, and why |
|---|---|---|---|
| D1 | which models | (a) `-32m` and `-67m` both scored (same recipe and LR, two depths), `joint` over 2 models × 2 counts; (b) `-32m` scored, `-67m` beside; (c) as (a), with `-pilot-dense` (every step 1–160) beside | **(a).** Two sizes on one recipe show whether a date moves with size, as 3b's did; downloads are about 6.7 GB and 14.3 GB (126 MB and 270 MB a checkpoint). (b) halves it but one model's date is one draw. (c) is a different run with no stated schedule (`status-T.md` Q3), so its dates map under no rule |
| D2 | which counts | (a) E₁'s break and the centred count, both primary, the centred arms derived per model as 3b; (b) E₁'s break only, the centred count described if computed | **(a).** C2's draft heads on two events (98 (c)), and the centred count's band separates step from cumulative as cleanly (65–225 against 308–1072). (b) saves the re-init preflight, about half of unit 1 |
| D3 | the plateau | **Revised after `/challenge-pr` on #284 (finding 1).** (a) early, the median V over 1253–3738 (Pythia's 1000–4000 under step, less 1072, so past every predicted date but tok524k's; LR 77–98 % of peak), unless tokens is scored under D4, then late, 5108–8158 (past every date, T5; 9536 kept for the preflight); both computed, and a seed's date set is the union of the two plateaus' dates, which can only keep a schedule; (b) late always, the early beside and reported (the first proposal); (c) early always, tokens never scored | **(a).** A later window reads a later break, which is why A5's D4 refused a late reference: on the 9 A5 seeds V's median rises from 1000–4000 to 8000–16000 on every seed (by 3–20, `claims/audits/p_a5_real_run.json`; no date moves on Pythia's octave grid, but Dodecahedron's is 1.17× a step). The late window sits in the cosine decay (53 % of peak at 5108 to 6 % at 8158, if it decays to 0) and leans every date toward step and tok524k, both under test. So it is used only when tokens needs it. (b) pays that lean in the likely case (T1 leans 4.19M) for nothing; (c) gives up Blocked 118 (b) |
| D4 | when Timaeus's answer counts | (a) tokens at 524k enters `joint` only if a reply confirming 524,288 tokens a step is quoted in `status-T.md`, with its date, before T2's first scored record is written; a later reply is recorded and tokens stays beside. A reply that changes the warmup (length, shape, start) or the peak LR sends the mapping and this design back to you before the batch. The batch does not wait beyond unit 1; (b) as (a), but the scored batch waits up to 14 days for a reply | **(a).** Forward passes are cheap and the December target is close; unit 1 already takes a session, so a quick reply still counts. (b) is right if testing tokens matters more than the date |

## The question for Timaeus (yours to send; outward)

A draft, for email (contact on timaeus.co) or the model repo's Community tab (public):

> We are reading the energy of the residual stream along training on `dodecahedron-32m` and
> `-67m` (forward passes on your released checkpoints). Two questions about the run: (1) the
> global batch, in tokens per optimizer step: the card's 64 × 4 × 2048 = 524,288, or that per GPU
> (8 × 524,288 = 4,194,304), which is what 9,536 steps and 40B tokens imply? (2) the LR schedule
> as trained: warmup linear from 0 over 1,000 steps to 3.5e-3, and cosine to what floor at step
> 9,536? A training log, if one is public, would answer both. Thank you for releasing these.

## Exit and budget

**Exit:** `joint` (or a recorded refusal) on D1's models × D2's counts, with `with_3b`, 3c beside,
and the tokens columns' status stated. **Budget: 3 units + close-out** after this design: (1) the
producer (`trackT_timaeus/t2.py`, pure tests, `t2_grid.py` re-pointed at its `predicted`), the
calibration gate and the preflight on step 1, step 9536 and the re-inits; (2) the download and
the batch (`runner`); (3) the reading. **Revised by Blocked 120 (b)** (the user's, 2026-10-30): 4
units + close-out, unit 1b and its pre-committed exit in "Blocked 120 (b)" below. **No lean:** nothing of Dodecahedron has been read but its
cards, configs, branch listings and four LFS hashes (T1); no weight is downloaded before the freeze.

## Worth challenging (for the PR)

- Scoring fraction at all: on this grid it is 0.7 × step and parts from step only at 420–574 (E₁),
  so "consistent with step and fraction" will be the usual step reading; dropping it would make
  the step reading look sharper than the grid allows.
- D3 (a)'s plateau switching on D4's trigger (revised after `/challenge-pr` on #284): the late
  window, in the cosine decay, only when tokens is scored; the union of both plateaus' dates.
- "Raw LR dose": a width-scaled dose (beside) lands on step's band at 141–205, so a step reading
  excludes only the raw dose.
- The centred arms re-derived per model on step 9536 (end of decay), the only trained step not in
  the reading; 3b and 3c used a step past their plateau at higher LR.

## Blocked 120 (b): a centred count fit to Llama (proposed 2026-10-30; not frozen)

Taken 2026-10-30 by the user (Blocked 120 (b), over the recommended (c)): T2 continues with a
centred count built for Dodecahedron and a new arm family, the budget extended on the record
(below). **This is a statistic and a rule chosen after the frozen ones failed**, so everything
here is fixed before any Dodecahedron value of the new count exists, and the exit is pre-committed.

**Why unit 1's count failed (L3, `tools/math_checks/t2_llama_count.py`).** e1c normalises, then
centres: on n unit rows at mean cosine ρ the centred rows keep a squared norm (n − 1)(1 − ρ)/n,
about 0.10 on Dodecahedron (ρ 0.90–0.91) against 0.83 on 410m at 4000 (ρ 0.17). So e1c there
re-normalises a tenth of each row, and anything that turns the mean direction moves it. L3 says
how much is left, not that this is why plant A leaked (measured, `status-T.md` "T2 unit 1").

**Lean.** No value of the new count on Dodecahedron has been computed. On 410m, e1r (below) is
in every C2 record, and its V per seed under 3s's arms is printed beside in
`data/trackC/c2u3s/2026-10-22-run/reading.json` (seed 0's also in `trackC_writeup/status-C.md`).
Seeds 0–7 were printed again in the session that wrote this, before the rule: they rise at
512–1000, as e1c's `REF` does. So D8's reference is not blind; the Dodecahedron side is.

| # | choice | options | recommended, and why |
|---|---|---|---|
| D5 | the count | (a) **e1r**: centre the raw states over tokens, then normalise (C2's "raw-centred", beside in 3a–3c); (b) e1 on the unit cloud with its first principal direction projected out; (c) e1c with position 0 dropped | **(a).** e1r is in every record on both sides (410m's 10 seeds, C2 3a–3c), so the reference needs no new Pythia pass, and it never re-normalises the small residual L3 describes. (b) needs 10 seeds × 18 steps of new 410m passes and is a statistic C2 never computed. (c) does not touch the shared direction: position 0 is one row of up to 512, and the mean cosine is over all pairs |
| D6 | the plants | (a) plant A is kept and recorded, but it **cancels exactly** from e1r (L1), so it cannot fail. **A_s** is added as the leak test: every row moved by 0.5 ‖h_i‖ u, a common direction on the sphere, the shape the diagnostic found. Gate: A_s ≤ 0.1, as A was. Plant B is unchanged; (b) plant A only | **(a).** A_s reaches e1r only through the spread of token norms (L2), and that spread is large at 9536 (position 0 at 13.9–23.8× the median), so the gate can fail. Under (b) the leak gate is vacuous |
| D7 | the arms | (a) **scale-matched arms gate**: the re-init quantile arm τ_q (the re-inits' 95th percentile of e1r drops; its re-init rate is ≤ 5 % by construction, so its gate is A_s and B) and the B-matched arm. 1e-4 and 3e-4 are reported beside and do not gate. The null-matched arm is dropped; (b) unit 1's family with null-matched replaced by τ_q, every arm gating | **(a).** The null-matched arm's re-init rate equals the uncentred count's (6.2 %, 8.1 %) by construction, so it fails the 5 % bar on Dodecahedron whatever the count. 1e-4 and 3e-4 are absolute tolerances set on Pythia's scale of energy drops. (b) can fail on a fixed tolerance that measures nothing about the count |
| D8 | the reference | (a) **REF_r on 410m by the same rule**: D5–D7 run on 410m's 3 re-inits and step 7000, as in 3s. REF_r is the set of the 10 seeds' e1r half-rise dates over 410m's gated arms, as `REF`'s centred entry was built. e1r must pass the gate on 410m too; (b) `REF`'s centred entry {512, 1000} reused for e1r | **(a).** The same statistic and the same rule on both sides. (b) maps e1c's Pythia event onto e1r's Dodecahedron count |
| D9 | what enters `joint` | (a) **e1r alone** over 2 models; E₁'s break beside, read but not scored; (b) E₁ and e1r both, 2 × 2 | **(a).** Unit 1's diagnostic says E₁ at cosine 0.9 is not Pythia's event (its plant A share is 0.04 / 0.07 against 0.79–0.98). The cost: over 2 cells, chance survival per schedule was 5.4–6.6 % for E₁ (unit 1). It is re-computed for REF_r in unit 1b and stated beside any reading |

**Unit 1b (one unit; GPU, no new download):** `t2.py` gains e1r and A_s in the preflight
(plants per `_plants`, reading the raw states), the arm rule of D7, the 410m preflight of D8 and
REF_r from the seed records, plus pure tests (L1 and L2 on the code's own functions, τ_q's rate).
It is then run on 410m, 32m and 67m: the re-inits, step 1 and step 9536 only (Dodecahedron), and
step 7000 (410m). The planted effect and planted null are re-run on REF_r.

**Exit, pre-committed.** If e1r fails a gating arm on any of 410m, 32m or 67m, **T2 closes at
unit 1b with what it knows; no third count is built**. If it passes, unit 2 (the download and
the batch) and unit 3 (the reading) run as frozen, with D9's `joint`. **Budget: 4 units +
close-out** (the frozen 3, plus 1b; the user's, Blocked 120 (b)). The first reading is the
pre-registered half-rise on D3's plateau. Everything else is beside.

**Worth challenging (for the PR):** D6's A_s is a plant chosen to fit the nuisance the diagnostic
found, after the data showed it; a plant that e1r passes because of how it is built (as plant A
is) would be no test, and the converse risk is a plant e1r fails for a reason Pythia's e1c was
never asked about (A_s has not been run on 410m's e1c). D7 drops the absolute arms from the gate
after they failed: that is defensible on scale, but it is a rule loosened after seeing a failure.
