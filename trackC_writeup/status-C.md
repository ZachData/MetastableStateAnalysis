<!-- trackC_writeup/status-C.md -->
# Track C — STATUS

<!-- phase-card -->
## Card

- **Question:** When, along `pythia-410m`'s training, does the tokens' interaction energy stop being non-decreasing in depth, as the identity-weight particle picture predicts, is that step the run's or the recipe's, and where does it fall against the model's known developmental order (n-gram stages, previous-token and induction heads, the OV maps turning repulsive)?
- **Inputs:** none run under C's name; C2 reads Track A's and Track S's records (`trackC_writeup/design-C2.md` "What the record supports")
- **Results:**
  - C2 opened 2026-10-22 and its design froze the same day: seven lines the record supports, nine fences, an outline, a render step so every number comes from a record; Blocked 91–94 decided as recommended (left to Claude by the user): the dated, reproduced break as the headline, the anisotropy control and the LR-dose arm as new runs, Markdown rendered from records, S2 in the main text and S1 an appendix — `trackC_writeup/design-C2.md` "Decided 2026-10-22"
  - Unit 1, the reading: the plain E_β is proved monotone only at identity weights; wider hypotheses (symmetric scores tied to the values) make a weighted energy monotone, not E_β; so the break violates no theorem (F5 worded from the texts); published anisotropy along Pythia starts at step 10 000, not in the break's window (F9 narrowed); Mistral GPT-2 small (4000-step warmup, checkpoints every 50 steps to 2000) makes 92 (ii) feasible, so 3c runs (7 units + close-out); §2 and §8 drafted (revised after `/challenge-pr` on #253, findings 1–5) — `trackC_writeup/status-C.md` "C2 unit 1"
  - Unit 2, the render step: every number in the draft is a key that `numbers.py` reads from its record (A5's seeds and gate, A3's rows, S2's readings, Pythia's released configs; 13 records, each sha256 in the tracked `numbers.json`), and CI checks the draft against that table; §3–§5 drafted and the three figures drawn; no reading moved — `trackC_writeup/status-C.md` "C2 unit 2"
  - Unit 3, the anisotropy control (tier 1, description): the frozen rule reads **"mostly removed"** (centred plateau 20 against 57 at D2's 1e-3; Blocked 95 (a)), but that label's size half is a scale effect: centring makes E₁'s layer-to-layer changes about 30× smaller, and at a matched tolerance (1e-4, 1e-5; post hoc, `/challenge-pr` on #255) it reads "moved" (ratio 0.90–1.10, centred date 1000). What holds at every tolerance: at 512, the break's step, the centred count is 4–13 against 56–60, so the break at 512 is mostly the common direction's (F9 a finding), and a full-size centred break appears at 1000, induction's grid step; seed 0 only. 91 reopens (Blocked 96) — `trackC_writeup/status-C.md` "C2 unit 3"
  - Unit 3s, the centred count on the 9 PolyPythias seeds (tier 1, description; Blocked 96 (b)): one centred tolerance on the centred energy's own scale failed its own gate, because no tolerance reads a planted change alike through both counts, so no size comparison between them is calibrated; the rule reads dates only at four centred tolerances (Blocked 97 (a)). The centred date reproduces: the centred half-rise is at 1000, one grid step after the uncentred break, on 8 of 9 seeds (K1 and K2 are one reading on this grid; seed 6 ties on one arm), so F9 is a finding on the seeds as a date (how much of the break is the common direction's is not calibrated), and the centred break shares 1000 with D10's heads; revised after `/challenge-pr` on #256 (findings 1–4); 91 goes to the user, decided (c) (Blocked 98): the draft heads on the two events — `trackC_writeup/status-C.md` "C2 unit 3s"
  - Unit 3b, the LR-dose arm on `pythia-70m` and `pythia-160m` (tier 1, description; Blocked 99 (a), 100 (a)): both dates on both sizes, against three schedules' predictions on Pythia's grid. E₁'s break at 256 and the centred date at 1000 on both: **consistent with step and cumulative** (the same without 70m's fragile E₁ date), an LR level excluded on one seed per size; step against cumulative is not told apart on this grid, and size is confounded. F1 stands — `trackC_writeup/status-C.md` "C2 unit 3b"
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@2db9b14ba1, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the dates are fixed in steps (or tokens) or wait for a cumulative LR dose (F1); 3b could not separate them on Pythia's grid, 3c reads it on another suite's warmup (description only) — `trackC_writeup/status-C.md` "C2 unit 3b"
- **After Phase 10:**
  - none
- **Reviewed:** 2026-10-22 · body `2d3a890240`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: frozen, Blocked 91–94 decided as recommended (left to Claude)** | `design-C2.md` "Decided 2026-10-22" |
| C2 1. reading (writing only) | **done 2026-10-22: F5 worded, F9 narrowed, 92 (ii) feasible (GPT-2 small), §2 and §8 drafted; revised after `/challenge-pr` on #253** | `lit-C2-unit1.md`, `draft-C2.md`; "C2 unit 1" below |
| C2 2. `numbers.py`, figures, §3–§5 | **done 2026-10-22: 102 keys from 13 records, 24 sentence claims checked, §3–§5 render, figures 1–3 drawn; revised after `/challenge-pr` on #254** | `numbers.py`, `numbers.json`, `draft-C2.src.md` → `draft-C2.md`, `fig/`; "C2 unit 2" below |
| C2 3. anisotropy control | **done 2026-10-22: frozen before output (95 (a), the user's), preflight cited; reads "mostly removed", the centred date 1000; Blocked 96** | `c2_aniso.py`, design "Unit 3"; "C2 unit 3" below |
| C2 3s. the centred count on the seeds (96 (b)) | **done 2026-10-22: 96 (b)'s tolerance refused by its gate; dates at four arms (97 (a), the user's); K1 = K2 = 8 of 9, "reproduced"; Blocked 98** | `c2_seeds.py`, design "Unit 3s"; "C2 unit 3s" below |
| C2 3b. LR-dose arm (70m, 160m) | **done 2026-10-22: 98 (c), 99 (a), 100 (a), the user's; frozen before output; "consistent with step and cumulative" (instantaneous excluded); F1 stands** | `c2_dose.py`, design "Unit 3b"; "C2 unit 3b" below |
| C2 3c. warmup arm (Mistral GPT-2 small) | next: both dates primary (98 (c)), read against the same three schedules on its own grid | design 92 (ii); 3c's separations `lit-C2-unit1.md` §5 |
| C2 4, 5, close-out | after 3c | — |

## C2 unit 3b, the LR-dose arm (2026-10-22)

Blocked 92 (iii), read under 98 (c) (both dates primary), 99 (a) (the labels re-cut by which
schedule predicts each date on the grid) and 100 (a) (plant B's gate dose per size), all the
user's, in session, before any scored record. The rule is in `design-C2.md` "Unit 3b", frozen and
pushed (`89022d1`) before the run.

**Input:** `pythia-70m` (peak LR 1e-3, 6 layers) and `pythia-160m` (6e-4, 12 layers), one seed
each, at A5's 18 grid steps × the 7 v1 prompts (`prompts_hash` `976b62d677bc`), float32, eager,
TF32 off, RTX 3080; checkpoints fetched to `HDD_1TB/mets_data/hf` (`model.safetensors`, 19
revisions per size with step 7000), the run offline; code `02a3cae5c04f`
(`trackC_writeup/c2_dose.py`). Records `data/trackC/c2u3b/2026-10-22-run/{70m,160m}/records/`
(126 each, main tree; about 3.5 min); reading `…/reading.json` sha256 `fd864233d350…`. 410m enters
as its 10 seeds' dates only (`REF`: E₁'s break {256, 512}, centred {512, 1000}). Preflight and its
v0: design "Unit 3b". No known answer: neither size had a record of E₁ by step before this one;
the code path is units 3 and 3s's (`c2_aniso.states`, `state_row`), whose known answers passed.

| size | units per step | V_u at 128, 256, 512, 1000 (plateau) | E₁'s break | V_c at 128, 256, 512, 1000, per arm (plateau) | centred date | consistent |
|---|---|---|---|---|---|---|
| 70m | 42 | 0, 3, 4, 8 (6) | **256** | 9.61e-5, 1e-4: 1, 1, 5, 14 (15); 3e-4: 1, 1, 5, 12 (15); 4.78e-3: 0, 0, 0, 3 (3) | **1000 ×4** | E₁: all three; centred: step, cumulative |
| 160m | 84 | 0, 18, 16, 16 (16) | **256** | 1.25e-4: 1, 1, 11, 22 (26); 1e-4: 1, 1, 11, 22 (27); 3e-4: 1, 1, 9, 17 (22); 8.22e-4: 0, 0, 5, 14 (19) | **1000 ×4** | E₁: all three; centred: step, cumulative |

**The reading, by the frozen rule:** nothing unreadable (steps 0–128 at most 1 on any count).
The schedules consistent with all four dates: **step and cumulative: "consistent with step and
cumulative"**; instantaneous is excluded by both sizes' centred dates (it predicts 128–512 on 70m
and 256–512 on 160m). By the outcomes fixed in the design: §7 says the dates do not follow the LR
level; **F1 stands**, the warmup's timing (or the tokens seen) and the cumulative dose not told
apart, as the design said 3b could not.

**Beside, not the reading:**

| what | reads |
|---|---|
| 3s's two events on the other sizes | on both sizes the centred date (1000) is later than E₁'s break (256), and at 1000: unit 3s's events 1 and 2 hold on 70m and 160m too, one seed each. At the break step the centred count is at most 1 (plateau 15–27) |
| E₁'s break at 256 on both | earlier than 8 of 410m's 10 seeds (512), in the dose's direction, but inside 410m's own seed range (seeds 3 and 9 at 256), so not a reading. One seed per size: their own spread is unmeasured |
| margins | 70m's E₁ break is a tie: V_u at 256 is 3, exactly half its plateau of 6. It is fragile both ways: one unit later it reads 512 ("step, cumulative"), but read against its later level (15 from 8000 on) it reads 1000, which no schedule predicts for 70m, and the joint would be "neither" (`/challenge-pr` on #257, finding 1). **The joint is the same with 70m's E₁ date removed** (160m's E₁ date and both centred dates give "step and cumulative"), so it rests on those three. The joint needs both sizes' centred dates to move before instantaneous survives: 70m is 2.5 units below half at 512 (5 against 7.5), 160m 2–4.5 units per arm |
| 70m's uncentred count | the weakest reading (the design tool's `power` refusal, argued): its plateau over 1000–4000 is 6 of 42 units, and it keeps rising after (15 at 8000–32000, 5 at 143000), so its "plateau" is a shelf |
| **post hoc (`/challenge-pr` on #257, finding 2)**: the two counts lean different ways | "step" survives the E₁ dates only through 410m's two seeds at 256 (3 and 9; 8 of 10 read 512), and "cumulative" survives 70m's centred date only if 410m's centred crossing is in (935, 1000] (160m's: (724, 1000]), the top of its interval. So the E₁ dates lean toward a dose and the centred dates toward step. On 410m seeds 3 and 9 (E₁ at 256, centred at 1000 at all four arms, unit 3s) step accounts for both counts on one seed, the pattern 70m and 160m show. Not a reading: the frozen statistic is the joint |
| the layer-mean cosine | on both sizes it rises to a peak at 32 (0.54, 0.67), falls to its lowest at 1000–2000 (0.20–0.23) and rises after; 410m's per-layer profile was not compared here |

**What it changes.** §7 can say: across a 3.3× range of peak LR at a shared warmup (one seed
per size, size confounded), the centred date stays at 1000 and E₁'s break at 256–512, so the
dates are not shown to wait for the LR to reach a level; whether they are fixed in steps (or tokens) or wait for a cumulative LR dose, Pythia's
grid cannot say (`tools/math_checks/c2_dose_grid.py`'s outcome table). Size is confounded
throughout. 3c, on GPT-2 small's finer grid and longer warmup, is the arm that can separate
step from cumulative.

**Not established:** step against cumulative; anything about size apart from the LR (6, 12, 24
layers move with it); the smaller sizes' seed spread (one seed each; PolyPythias' smaller sizes
are Parked 1); that 70m's uncentred count sees what 410m's does (its power refusal).

**Parked:** (1) PolyPythias' 70m and 160m seeds, if released (unverified): their own seed spread
would replace 410m's as each size's reference; cost a download of 9 × 18 checkpoints per size and
one run; could change whether a size's date sits inside its own spread, not this reading's
exclusion of instantaneous (which uses 410m's band, the conservative one). Carried to the close-out
for a decision. (2) `numbers.py` keys for this `reading.json`: unit 4, with unit 3s's Parked 1.

**Re-run:** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_dose
preflight --size {70m,160m} --out <dir>` (a few minutes each; refuses unless it re-derives
`TAU_ARMS`), then `python -m tools.design_preflight <dir>/<size>/preflight_input_{c,u}.json --out
…`; `… c2_dose run --out <run>` (about 3.5 min on the GPU, offline), then `read --out <run>
--preflight <dir>`. Tests: `tests/test_trackC_dose.py` (pure, 19); the predictions:
`python3 tools/math_checks/c2_dose_grid.py` (D1–D4).

## C2 unit 3s, the centred count on the seeds (2026-10-22)

Blocked 96 (b), the user's. The rule is in `design-C2.md` "Unit 3s", frozen and pushed (`98617c7`)
before any seed's centred record. 96 (b)'s single median-matched tolerance failed its own gate,
so it reads dates only, at four centred arms (Blocked 97 (a), the user's, in session).

**Input:** PolyPythias `pythia-410m-seed1`–`seed9` at A5's 18 grid steps (`HDD_1TB/mets_data/hf`,
offline; nothing downloaded) × the 7 v1 prompts (`prompts_hash` `976b62d677bc`), float32, eager,
TF32 off, RTX 3080; code `e7a1e159f247` (`trackC_writeup/c2_seeds.py`). Records
`data/trackC/c2u3s/2026-10-22-run/seed{1..9}/records/` (126 each, main tree; about 41 min);
reading `…/reading.json` sha256 `8d5ed734f665…`. Known answer: every record's e1 equals A5's
record for that seed and step (largest deviation 0.0), tokens identical, and each seed's V equals
A5's `score.json` V. Seed 0 is unit 3's records, read beside.

| seed | uncentred break (A5) | V_u at 512 | V_c at 512, per arm (7.55e-5, 1e-4, 3e-4, 7.58e-4) | V_c at 1000, per arm | centred date, per arm | event 1 | event 2 | D10's prev / ind |
|---|---|---|---|---|---|---|---|---|
| 0 (beside) | 512 | 56 | 11, 11, 10, 5 | 56, 55, 35, 19 | 1000 ×4 | yes | yes | — |
| 1 | 512 | 47 | 10, 10, 9, 7 | 36, 36, 32, 14 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 2 | 512 | 41 | 16, 16, 11, 5 | 35, 34, 26, 16 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 3 | 256 | 49 | 5, 5, 3, 2 | 57, 55, 46, 20 | 1000 ×4 | yes | yes | 4000 / not located |
| 4 | 512 | 58 | 9, 9, 7, 2 | 38, 38, 30, 19 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 5 | 512 | 55 | 14, 14, 8, 7 | 51, 48, 33, 16 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 6 | 512 | 53 | 14, 14, 11, 5 | 25, 24, 20, 14 | 1000, 1000, **512**, 1000 | no | no | 1000 / 1000 |
| 7 | 512 | 43 | 13, 12, 9, 6 | 42, 39, 25, 11 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 8 | 512 | 58 | 15, 13, 9, 7 | 49, 46, 33, 18 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 9 | 256 | 52 | 20, 17, 10, 6 | 42, 42, 35, 18 | 1000 ×4 | yes | yes | 1000 / 1000 |

**The reading, by the frozen rule:** no seed unreadable at any arm (steps 0–128's centred counts
at most 4). **K1 = 8 of 9, "reproduced"** (the centred half-rise later than the uncentred break);
**K2 = 8 of 9, "reproduced"** (the centred half-rise at 1000). By the outcomes fixed in the design,
96 (c) is open on the seeds and the choice goes to the user: **Blocked 98**. *On this grid K1 and K2
are one reading, not two (`/challenge-pr` on #256, finding 2): every uncentred break is at 256 or
512, so a centred date at 1000 implies one later than the break. The two events would differ only
for a seed whose centred date was 2000 or 4000, and none was.*

**Beside, not the reading:**

| what | reads |
|---|---|
| seed 6 | its one miss is a tie on one arm: at 3e-4 the centred plateau is 22 and V_c at 512 is 11, exactly half, so the half-rise lands at 512. At the other three arms it reads 1000 |
| the size at 512 | across the 10 seeds and 4 arms V_c at 512 is 2–20 against V_u 41–58; at 1e-3 it is 1–7 at 512, 10–16 at 1000. That holds on every seed. It is not a ratio label: the gate showed the two counts have no common scale |
| seeds 3 and 9 | their uncentred break is at 256 (V_u 35, 29); their centred counts at 256 are 0–2 and their centred date is 1000 too, so the centred date does not move with the uncentred one |
| the centred date against D10 | on the 7 seeds where D10 dates both heads at 1000 and the rule reads 1000 (1, 2, 4, 5, 7, 8, 9), the centred date is the heads' grid step. Seed 3's heads are not both located. With no grid step between 512 and 1000, that is one grid step shared, not an order |
| the centred plateau's size | on its own count it falls with the arm (e.g. seed 0: 59, 55, 41, 22), as a larger tolerance should; the date does not move. At the B-matched arm the centred count keeps rising after 1000 (to 34–71 at 143000), so there the date is a half-rise on a rising curve, not a step onto a flat plateau (finding 2) |
| **post hoc (`/challenge-pr` on #256, findings 1, 3, 4)**: the centred count at the break, on its own scale | V_c at the seed's break is 0.17–0.50 of its own plateau on the 8 seeds that break at 512 (5–16 units at the null-matched arm) and 0–0.04 on seeds 3 and 9 (at 256). Below half on every seed but the tie; well above the early-step floor (≤ 4). It is not "near the floor" |
| post hoc: the common direction by layer, per seed | the layer-mean cosine rises with depth at all 23 block transitions at 128 on all 10 seeds (none falling); at 512, 6–10 falling on every seed (seeds 3 and 9: 3 and 6 falling at 256, 9 and 8 at 512). So seed 0's turn of the cosine profile at the break is on every seed. It is E₁'s first-order term, so this is the same break decomposed, not separate evidence |
| post hoc: margins | per seed and arm, how far the half-rise is from moving: below half before 1000 by 0 (seed 6 at 3e-4, the tie), 2.0 (seed 9, null-matched), 2.5 (seeds 1, 6), 3.0–3.5 (seeds 2, 6, 7); above half at 1000 by 1.5 (seed 7, B-matched), 4.5 (seeds 1, 2, 5), then ≥ 5. All the near misses are at the outer arms, mostly the B-matched one. One unit's change flips seed 6 back, and about two flip seed 7 or 9. "Reproduced" (≥ 7) holds with 1 seed to spare |

**What it changes.** The centred date reproduces across the PolyPythias seeds: on 8 of 9 (9 of
10 with seed 0) at every arm, the token-centred count reaches half its own plateau at 1000, one
grid step after E₁'s break (512 on 7 seeds, 256 on 2), the step where D10 dates previous-token and
induction on 8 seeds. At the break step itself the centred count has not yet reached half of its
own plateau (0.17–0.50 of it), and the layer-mean cosine turns from rising with depth at the break
on every seed. F9 is a finding on the seeds, as a date: the break comes before the centred
cloud's. How *much* of the break is the common direction's is not calibrated (the gate). 91's
headline goes back to the user as Blocked 98: (a) or (c).

**Not established:** that the centred break at 1000 is a separate event rather than the same
change read later (no step between 512 and 1000; the same limit as unit 3); any size comparison
between the two counts (the gate showed none is calibrated); that the centred date follows
induction or previous-token rather than the LR half-point or anything else at 1000 (one grid
step; 3b and 3c are what could separate the schedule); anything on another model size.

**Parked:** (1) `numbers.py` keys for this `reading.json` (K1, K2, the per-seed dates) and the
centred row beside fig 1: unit 4, with unit 3's Parked 1 (cost: one render pass).

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_seeds preflight --out <dir>` (about
3 min; refuses unless it re-derives `TAU_ARMS`), then `python -m tools.design_preflight
<dir>/preflight_input.json --out <dir>/preflight_report.json`; `METS_DATA=<main>/data
HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_seeds run --out <dir>` (about 41 min on
the GPU, offline), then `read --out <dir>`. Tests: `tests/test_trackC_seeds.py` (pure, 13).

## C2 unit 3, the anisotropy control (2026-10-22)

92 (i) with Blocked 95 (a)'s size label, frozen in `design-C2.md` "Unit 3" and committed (`7878f5d`)
before the scored run; the preflight's report and its lean are there.

**Input:** `pythia-410m` (seed 0) at A5's 18 grid steps (HF cache, offline) × the 7 v1 prompts
(`prompts_hash` `976b62d677bc`), float32, eager, TF32 off, RTX 3080; code `d90bb06b7910`
(`trackC_writeup/c2_aniso.py`). Records `data/trackC/c2u3/2026-10-22-run/records/` (126, main
tree); reading `…/reading.json` sha256 `46f7654c96ba…`. Known answer: every record's E₁ equals A5's
seed-0 record (`trackA/a5/2026-10-20-u1/seed0`) to 0.0 (tolerance 1e-4), tokens identical.

| step | 0–128 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V, uncentred (A5's) | 0 | 19 | 56 | 56 | 61 | 57 | 59 | 64 | 76 | 72 | 70 |
| V_c, centred, 1e-3 (the label's) | 0 | 0 | 4 | 15 | 20 | 24 | 26 | 29 | 32 | 39 | 59 |
| V_c at 1e-4 | 1, 1, 1, 1, 0, 0, 0, 0 | 1 | 11 | 55 | 56 | 55 | 57 | 56 | 51 | 58 | 75 |
| V_c at 1e-2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 4 | 10 |
| V, raw-centred, 1e-3 (beside) | 0 except 3 at 128 | 0 | 5 | 15 | 21 | 26 | 29 | 40 | 52 | 58 | 72 |
| mean cosine, layer mean | 0.42, 0.42, 0.42, 0.51, 0.70, 0.62, 0.39, 0.30 | 0.30 | 0.33 | 0.24 | 0.18 | 0.18 | 0.23 | 0.36 | 0.44 | 0.45 | 0.40 |

**The reading, by the frozen rule:** steps 0–128's centred counts first: all 0. Centred plateau
median 20 against 57 uncentred, ratio 0.35 < ½: **"mostly removed"**, with the centred half-rise
at 1000 beside (uncentred 512).

**Beside, not the label:**

| what | reads |
|---|---|
| the rule at the other tolerances | 1e-4: plateaus 55 / 61 (0.90), centred date 1000: "moved"; 1e-2: centred plateau 0: "removed". So the size half of the label depends on the tolerance: centring compresses E₁'s range (a centred cloud's mean cosine is near 0), so the same layer-to-layer change is a smaller relative drop |
| what holds at every tolerance | at 512, the break's step, the centred count is 4 (1e-3), 11 (1e-4) or 13 (sign level) against the uncentred 56, 60 or 60; the centred half-rise falls at 1000 wherever one is located, and the raw-centred cloud's at 1000 too |
| **post hoc, the scale (`/challenge-pr` on #255, finding 1)** | the centred E₁ sits about 0.002 above its floor (≈ 0.504 at step 0); its mean per-transition relative change is 3.9e-5 at step 0, 1.2e-3 at 512, 8.1e-4 at 1000 and 8.4e-4 at 4000, against 2.2–2.9e-2 uncentred: about 30× smaller, so one relative tolerance is far stricter on it. Matched more closely, the rule reads "moved": at 1e-4 the plateaus are 55 / 61 (0.90), at 1e-5 67 / 61 (1.10), the centred date 1000 in both; steps 0–128 at 1e-4 are 0–1. At sign level, float noise reads 24–35 at steps 0–8 ("unreadable"). From 1000 to 54000 the centred count at 1e-4 is flat (51–58), then 75 at 143000 |
| the common direction, by layer | at 128 the mean cosine rises with depth at all 24 transitions (0.02 → 0.71); at 256, 2 falling; at 512 it peaks at state 5 (0.40) and falls to 0.32 by state 15 (10 falling transitions); 8 at 1000, 10 at 4000. The mean cosine is E₁'s first-order term at β = 1, so this decomposes the same break; it is not separate evidence (finding 4) |
| the lean (design "Unit 3") | step 7000's 0.43 pointed to "mostly removed" and the re-inits away from "unreadable"; both held, and the 0.43 was the same scale effect |

**What it changes.** The frozen label stays on the record as "mostly removed". On a matched scale
the reading is 92 (i)'s "moved". Two parts hold at every tolerance. First, at 512 E₁'s
layer-monotonicity breaks mostly because the common direction stops growing with depth, so F9
becomes a finding. Second, the token-centred cloud breaks at full size one grid step later, at
1000, the step where L3 puts induction, previous-token and the LR schedule's half-point. L1–L4's
dates and `P-A5` stand as facts about E₁, the energy that was registered. **91 reopens because of
the count at 512, not the plateau ratio: Blocked 96, the user's.** 3b and 3c wait on it.

**Not established:** that the centred break at 1000 is a separate event rather than the same change
seen more weakly (the grid has no step between 512 and 1000); any centred-count rule calibrated on
its own scale (the post-hoc rows are description); any cause for the cosine profile's turn (no
component analysis); anything on the other 9 seeds (no centred records; A5's records hold E₁ only).

**Parked:** (1) `numbers.py` keys for `reading.json` (the label, the plateaus, the cosine profile)
and a figure beside fig 1: unit 4 (cost: one render pass; changes nothing until §4 is rewritten).
(2) The centred count on the 9 PolyPythias seeds, to say whether the 1000 date reproduces: about
20 GPU minutes (seed 0's 18 steps took 125 s); it could change 96's wording ("reproduced" or
"seed 0 only"); an option in 96. *Carried to unit 3s (96 (b)). "After a re-download of about 160
checkpoints" was wrong: all 162 are in `HDD_1TB/mets_data/hf`, where A5 put them (`LESSONS.md` 1).*

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_aniso run --out <dir>` (about 2 min
on the GPU, offline), then `read --out <dir>`; preflight `… preflight --out <dir>`, then
`python -m tools.design_preflight <dir>/preflight_input.json --out <dir>/preflight_report.json`.
Tests: `tests/test_trackC_aniso.py` (pure, 16).

## C2 unit 2, the render step, the figures and §3–§5 (2026-10-22)

Writing and tooling; no forward pass. `trackC_writeup/numbers.py`:

| step | what it does | needs |
|---|---|---|
| `produce` | reads the 13 records, computes 102 keys, writes `numbers.json` (each key's value and record id, each record's sha256, the code hash). Refuses: a record whose sha256 differs from the table's, or a key whose value or record moved (a new run directory, a code change), until named with `--reread`; any of `CLAIMS` false (24 assertions the draft's sentences make beyond the printed values: orders, "every", "none", margins, tallies); `score.json` not the one `claims/audits/p_a5_real_run.json` hashes; a V series off the 18-step grid | the main tree's `data/` |
| `render` / `check` | `draft-C2.src.md` + `numbers.json` → `draft-C2.md`; an unknown key or format, or a non-integer with no format, refuses; `check` fails if the tracked draft is not the render | any checkout (CI: `tests/test_trackC_numbers.py`, pure, 14, the claims on the tracked table among them) |
| `figures` | `fig/fig1_break.svg`, `fig2_timeline.svg`, `fig3_ov.svg` from `numbers.json` alone; byte-identical on re-run (one matplotlib); CI checks that they draw, not their bytes | matplotlib (`tests/test_trackC_figures.py`, deps) |

**Input set:** the record ids and sha256s in `numbers.json` `records` (A5: `trackA/a5/2026-10-20-u1/seed0/{gate,plan}.json`, `…-u2/score.json`; A3: `trackA/a3/2026-10-20-A3u5/a3_rows.json`; S2: `trackS/s2/2026-10-21-u2/{readings,gate_read}.json`; Pythia's `models/{70M,160M,410M}/pythia-*.yml` from `EleutherAI/pythia` main at `a19eecb807ec` (its newest commit, 2025-11-15), fetched 2026-10-22 to `data/trackC/configs/`, read by a line parser only; tracked: `claims/audits/p_a5_real_run.json`, `claims/registry.json`, `core/metrics.py`'s and `core/evalues.py`'s constants). Code hash in the table.

**What it found:** no number moved. Every rendered value equals its pointer in "What the record
supports" (L1–L4) and in A5's, A3's and S2's status sections. Two sentences the draft had carried
from summaries were checked against the source and reworded: the prompts are "hand-picked, not
sampled" (not "outside the training distribution": a Wikipedia paragraph and the *Iliad* may be in
the Pile), and PolyPythias's seeds differ in initialisation *and* data order.

**`/challenge-pr` on #254: accept with changes.** Findings taken: (1) §5's order line rendered
"E8 (1000) < E3 … (1000)", copied from A3's order, which placed E8 by its peak date: now E8 is
written between the groups with both dates; (2) the sentences' assertions are now checked (`CLAIMS`),
in `produce` and in CI, and the 23 unused keys are gone or used by a claim; (3) the guard now
compares key values and records, not only a record's hash at a fixed path; (4) the typed results
are keys (the late rise's steps, the tolerances, the block count, the time zone); "1024
sequences" stays as the Pythia paper's, cited; (5) grid positions are looked up on the 18-step
grid, and a series of another length refuses. (6) in part: the configs' upstream commit is
recorded here; CI checks that the figures draw but not their bytes (matplotlib's SVG moves
between versions). Answered on the PR.

**Choices made in the writing** (each in the PR's "Worth challenging"):

- A grid step named as a place on the axis ("between 512 and 1000") is a label and stays literal;
  a step that is a result is a key. The source header states the rule.
- F7 is written in the paper's terms (the check "is evidence neither about mechanism nor that
  trained weights resist the clustering"); `H-RESIST` and the relevance weight are ledger
  vocabulary and stay out of the text.
- Figure 3 (S2's R1 cells and G3′ against their 40 controls) is drawn now; §6, which cites it,
  is unit 4's. Figure 2 cuts each lane at −1.2 (E5's and E6u's late falls), said in its caption.
- §3.4 cites Vovk and Wang [2021] for the p-to-e calibrator (`core/evalues.py`); it is not in
  either scan's rows.

**Parked** (one line each: why / cost / which decision it could change):

1. The references list moves from unit 2 to unit 4 (the draft's own note had it here): titles and
   authors must be read from each paper, 22 ids plus Vovk–Wang and an ESD source for E8 /
   one metadata read / none, the list only. Carried to unit 4.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`, then `render` and
`figures` (`check` to verify). A changed record: read the change, then `produce --reread <id>`.

## C2 unit 1, the reading (2026-10-22)

Writing only; nothing ran. Read: the gradient-flow results (2312.10794 §3.2 and §3.4.1 in its
local LaTeX, 2605.04279), 2603.13314, the [S] rows §2 and §8 cite (each now [H] or [A]), a delta
scan since 2026-10-18, the Hub's checkpoint branches and tags for every candidate suite, and the
released configs. Record: `lit-C2-unit1.md`. `/challenge-pr` on #253 (accept with changes) found
the GPT-2 small tags, the weighted energy, an unrecorded [A] row, Blog 1's misquote and the [H]
mark's limits; all five taken (PR thread). What it changed:

| item | result | where it lands |
|---|---|---|
| F5 | the plain E_β is proved monotone only at Q = K = V = I (one constant head, continuous time, sphere, no MLP); symmetric scores tied to the values make a V-weighted energy monotone, not E_β; Pythia at every step and S2's edits meet none, so the break departs from a picture's prediction, not a theorem | `design-C2.md` F5; `draft-C2.md` §2.1 |
| the energy's sign | non-decreasing in every text; `lit-S2.md`'s "decreases" corrected, routed to S (no S2 reading moves) | `status-S.md` "Corrections received" |
| F9 | Bernas et al. read Pythia's activations from step 10 000 on and chiefly study the gradients; nothing reviewed measures activation anisotropy in 128–512; an unreviewed 70m project reports it half-built by 512, which raises unit 3's stakes | `design-C2.md` F9; `draft-C2.md` §8 |
| F1 | 2509.23024 ties its own first geometric phase to the warmup | `design-C2.md` F1; `draft-C2.md` §2.3 |
| 92 (ii) | **feasible on Mistral GPT-2 small**: 5 seeds, a 4000-step linear warmup to 6e-4, 604–609 checkpoint tags each (every 10 steps to 100, every 50 to 2000); medium's are gone; OLMo-2 1B has steps 0, 300, 10 000. It can separate steps or LR dose from warmup fraction or tokens, not within either pair | unit 3c runs; budget 7 units + close-out |
| 92 (iii) | the configs confirm the arm: 70m, 160m, 410m share the warmup (1430 steps), batch and schedule; peak LR 1e-3, 6e-4, 3e-4 | unit 3b as designed |
| 2603.13314 | weights on OLMo-2 7B; keys rise most, values flat; S1's `ov_read` is a different statistic | appendix A (unit 4) |
| novelty | "nobody reads this energy across training" not refuted as of 2026-10-22 | `draft-C2.md` §8 |

| Blog 1 | its random-weight energy rises monotonically; "under random weights too" is not its line (attribution corrected, routed to A) | `draft-C2.md` §8; `status-A.md` "Corrections received" |

Parked (`lit-C2-unit1.md`): GPT-2 medium's checkpoints off the Hub (carried to C2's close-out);
token-drift's per-step table (carried to unit 3b); where "under random weights too" comes from
(carried to unit 4).

## C2 unit 0, the design (2026-10-22)

Writing only; nothing ran. The design names what the record supports (L1–L7, each pointing to
the one place its numbers live), what the draft may not go past (F1–F9), the outline, and the
exit (a complete draft, every number rendered from a record, a fresh-context referee read
answered). F9, the anisotropy objection, is new to C2's row: `lit-C2.md` §1 named it and no
ROADMAP row carried it. Inputs read: the one-page closes of A3, A5, S1 and S2 and `lit-C2.md`.

## Corrections received

none
