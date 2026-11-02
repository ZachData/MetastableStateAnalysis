<!-- trackC_writeup/design-C6.md -->
# Track C — design: C6, V and the centred count on the released long-warmup 70m

Opened 2026-10-30 by the user ("open it and do what's best", C2p's readings audit;
`docs/ROADMAP.md` §8 row C6; `lit-C2p-fulltext.md` M1, M2, Parked 1–2). U0 (this file) ran no
energy on the run. **Frozen 2026-10-31, before any output on `pythia-70m-warmup01`**: the bands
per schedule (`c6_warmup.REF70`, "The bands" below), the reading rule and the run's gate. What
was read off the run before the freeze is only its weights' differences ("What the run is").

The scan is C2p's readings audit (`lit-C2p-fulltext.md`, 2026-10-30), plus in this unit
Belrose et al. 2402.04362 §5 and appendix (the run's only description), PolyPythias 2503.09543 §2
(what a seed varies), EleutherAI's `features-across-time` repository (`scripts/plot/plot_warmups.py`
pairs `pythia-70m` "fast warmup" with `pythia-70m-warmup01` "slow warmup") and gpt-neox
`configs/pythia/70M.yml` (`"warmup": 0.01`, lr 1e-3, min_lr 1e-4, cosine to 143 000). One web
search found no other use of the run.

## What C6 is for

`paper/` §7 says nothing on Pythia separates the dates from the schedule: C2 unit 3b read 70m
and 160m against 410m and could not tell step from cumulative LR (size confounded, one seed
each). `pythia-70m-warmup01` is 70m retrained with its warmup ten times longer, everything else
Pythia's: C4's longer-ramp arm, already trained. Read C2's two dates on it (E₁'s break in V, the
token-centred cloud's date) against `pythia-70m`'s, and label each by which schedule predicts it.
A clean label narrows §7 for 70m only; 410m's dates stay unseparated (C4).

## What the run is (U0, read off the weights)

Producer `trackC_writeup/c6_schedule.py` (code `798c2da33a8a` as run; record
`data/trackC/c6/2026-10-31-u0-schedule/schedule.json`, main tree, sha256 `294a981bc314…`;
CPU, differences of fp16 checkpoints, no forward pass). Re-run:
`HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_schedule --out <dir>`.

| question | read | answer |
|---|---|---|
| architecture | `config.json` at every branch against `pythia-70m`'s | identical but `classifier_dropout` (0.1, unused by a causal LM) and the `transformers` version; 155 branches, the same grid |
| initialisation | step 0, tensor by tensor | **the same**: 76 of 76 shared tensors bit-equal (26 of 26 that vary at init); control `pythia-70m-seed1`: 0 of 26 |
| first batch | step 2's change (step 1 equals step 0 in both: NeoX's rate is 0 at iteration 0); Adam's first step is ≈ lr · sign(g) | **the same**: sign agreement 1.000 on the 270 116 entries resolved in both; seed 1 against `pythia-70m` 0.501 |
| data order after it | bias increments' signs over each grid interval | agreement 1.000 to step 4, 0.996 to 8, 0.95 to 16, 0.84 to 32, 0.72 to 64, then at the control's ≈ 0.5–0.6: **consistent with the same order through ~16 steps; not shown after**, since the runs' parameters part (their rates differ 10×) |
| peak / warmup | zero-initialised biases' |change| quantiles, run over `pythia-70m`, at 4–512 | **0.094–0.10 at 4–32 (every quantile), the 99th 0.098–0.103 to 128**: the in-warmup rate is a tenth of Pythia's (H0, W 14 300 at peak 1e-3, predicts 0.100). The median drifts up from 32 (0.13 → 0.62 at 256), where 70m's parameters have moved enough for the gradients to part |
| the peak itself | the weight change between grid steps, run over `pythia-70m`, against the summed-rate ratio | from 16 000: 1.10–1.16 against H0's 1.04–1.11 (half the peak predicts 0.52–0.56, twice it 2.1–2.2): **Pythia's peak, within ~10 %**. In mid-warmup the proxy fails (2000–4000: 0.67 against 0.21; 8000–16000: 1.19 against 0.84): the run's low-rate steps add up more coherently than 70m's |
| the verdict (fixed before the file first ran) | median \|log(measured / predicted)\| over all 32 readouts < log 1.25 and the smallest | **H0 confirmed** (0.156; half peak 0.85, twice 0.67, no change 1.74) |

So the bands are computed on H0. Inside both warmups (every date the bands use below 1430 on 70m)
the maps depend only on peak / W (`tools/math_checks/c6_bands.py` B2), which the early biases pin
to 2–6 %; the peak alone matters only past the run's step 14 300, through the cosine.

**Because the run shares seed 0's init and first batch, `pythia-70m` is its matched control**;
the other nine seeds vary both init and order (PolyPythias §2), so their spread is wider than
anything but the schedule separates the run from seed 0. The union over ten seeds is the
conservative reference; seed 0's own dates are read beside.

## The reading, frozen

Code `trackC_writeup/c6_warmup.py` (`9e9aa860bdf2`), pure tests `tests/test_trackC_c6.py`,
closed forms `tools/math_checks/c6_bands.py` (B1–B6 and what it does not prove). A first version
was committed and pushed in `fbc2fdd` before any 70m seed's date was read; **the window, the
per-arm reference and the every-arm rule below were revised after the ten seeds' profiles were
read** (the reference, not the run), because the reading planted on those profiles could not
separate step from cumulative ("Calibration"). Nothing on the run had been computed.

| item | fixed as | why |
|---|---|---|
| the scored family | `pythia-70m-warmup01` × the 18 grid steps × the 7 v1 prompts | unit 3b's family, on the run |
| the records | unit 3b's (e1, e1c, e1r, cos per state; float32, eager, TF32 off) | the same instrument |
| the counts | V at D2's 1e-3; the centred V at 70m's four frozen arms (9.61e-5 null-matched, 1e-4, 3e-4, 4.78e-3 B-matched; `c2_dose.TAU_ARMS["70m"]`), not re-derived | the dates are compared with 70m's at the same tolerances |
| the reference (`REF70`) | each count's readable dates over `pythia-70m` and `pythia-70m-seed1..9`, each read exactly as unit 3b read 70m (the half-rise on the 1000–4000 plateau, 92 (i)'s unreadable check); **the centred count's per arm** | the ten seeds' spread, not one seed's date; the union is conservative, as 3b's `REF`; pooled over arms, the 4.78e-3 arm's later dates widened every arm's band |
| the schedules | **step** (t′ = t); **cumulative** (∫ rate equal; √10 · t inside both warmups); **instantaneous** (the rate equal, on the run's rising warmup; 10 · t for t ≤ 1430, and ≈ 14 300 for any 70m step past its peak, where 70m's rate falls) | unit 3b's three, with the run's real schedule (NeoX warmup then cosine) in place of a factor |
| a schedule's band | the run's grid steps s whose (prev(s), s] meets m_h's range over some (prev(g), g], g in `REF70` | unit 3b's `predicted`, with the instantaneous map's turn at 70m's peak handled |
| **a schedule's window** | under h the run's grid step s *holds* 70m's latest grid state h has reached by s (`held`: the largest 70m grid step g with max over t ≤ g of m_h(t) ≤ s). h's window is the run's steps holding a 70m state in 1000–4000 (step 1000, 2000, 4000; cumulative 4000, 8000; instantaneous none, so its first later step, 16 000), its early set the steps holding a state ≤ 128 (tops 128, 512, 2000) | "Why each schedule has its own window" |
| a count under h | per counted arm (gated, with a reference date), the half-rise on h's window; **h is inconsistent with an arm whose run count has a plateau of 0 there, or is ≥ half of it in h's early set, or rises outside that arm's band**; **consistent only if every counted arm's date is in its band**; the count unreadable only if no arm counts | h maps 70m's whole readable profile onto the run (low early, a plateau, the rise in the band), so a profile that breaks any part of it, on any arm, contradicts h |
| **calibration** (`calibrate`, recorded in `seeds.json`) | each seed's own V and centred V planted under each schedule (`plant`: the run's V at s is 70m's at `held(s)`), all arms gated, read against `REF70` and, beside, against the other nine seeds | the planted effect over the real design points ("Calibration") |
| **the reading** (`joint`) | the schedules consistent on both counts (where readable): all three → **"not told apart"**; none → **"neither"**; else **"consistent with …"**; none readable → **"unreadable"** | unit 3b's statistic |
| beside, not scored | each count's dates on C2's 1000–4000 window; the reading against seed 0's dates alone; V_c at 1e-3, the raw-centred V, the layer-mean cosine | description |

### Why each schedule has its own window

C2's date is the first grid step at half the median of V over 1000–4000. If the run's clock is
stretched, that window catches it mid-rise and the date comes out early, which favours step:
under the instantaneous map 70m's 1000–4000 is the run's 10 000–14 300. `pythia-70m`'s own V
shows the stakes: 0 to 128, 3 at 256, 4 at 512, 8 at 1000, 6 at 2000–4000, 15 from 8000 (unit
3b's record), so the half-rise depends on which plateau it is measured against. Each schedule is
a claim about the whole timeline, so the reading maps the window, the early set and the band
together. Under step it is exactly unit 3b's (`date_on(v, "step") == c2_dose.date_reading(v)`,
tested).

The window is the set of run steps that *hold* a 70m state in 1000–4000, not the grid steps
whose interval contains m_h(1000), m_h(2000), m_h(4000) (the first version). The grid is coarse
past 4000, so the interval rule put cumulative's window at 16 000, which holds 70m's state at
8000, where V rises a second time (6 → 15 on seed 0). That raised the plateau and pushed the
planted date out of the band. The instantaneous map jumps the window: every 70m step from its
peak on maps to ≈ 14 300, and no grid step lies between 10 000 and 14 300, so its window is
16 000, which under that map holds 70m's last state. B6 checks that a one-step rise planted at
any readable 70m date is read inside its band under each schedule.

## The reference: 70m's dates over ten seeds

Records `data/trackC/c6/2026-10-31-u0-seeds/<seed>/records` (126 per seed: 18 steps × 7
prompts; forward code `5aa5eb7a067b`, GPU, float32, eager, TF32 off); reading
`data/trackC/c6/2026-10-31-u0-seeds/seeds.json` (code `9e9aa860bdf2`, sha256
`237fea0ffcfe…`; `ref70_frozen_equal: true`). **Seed 0 is unit 3b's 70m**: its records equal
3b's (max relative difference in e1 0.0, the tokens' hash equal), and its dates are 3b's (256, 1000).
Re-run: `python -m trackC_writeup.c6_warmup run --out <dir> --models seed0 … seed9` (offline,
`HF_HOME=<HDD_1TB>/mets_data/hf`), then `seeds --out <dir>`.

| seed | V (1e-3) | centred, 9.61e-5 | 1e-4 | 3e-4 | 4.78e-3 |
|---|---|---|---|---|---|
| 0 (`pythia-70m`) | 256 | 1000 | 1000 | 1000 | 1000 |
| 1 | 512 | 1000 | 1000 | 1000 | 2000 |
| 2 | 512 | 1000 | 1000 | 1000 | 1000 |
| 3 | 512 | 512 | 512 | 1000 | 1000 |
| 4 | 512 | 1000 | 1000 | 1000 | 1000 |
| 5 | 512 | 1000 | 1000 | 1000 | 2000 |
| 6 | 256 | 512 | 512 | 512 | 1000 |
| 7 | 256 | 512 | 512 | 512 | 1000 |
| 8 | 256 | 1000 | 1000 | 1000 | 1000 |
| 9 | 256 | 1000 | 1000 | 1000 | 1000 |

Every count is readable on every seed. The break in V is at 256 or 512 (5 and 5), so 70m's
seeds span two grid steps where 3b's one seed gave one; the centred date is 512–1000 at the
three small arms and 1000–2000 at the B-matched arm.

## The bands (frozen into `c6_warmup.REF70`)

`REF70 = {"u": (256, 512), "c": {9.61e-5: (512, 1000), 1e-4: (512, 1000), 3e-4: (512, 1000),
4.78e-3: (1000, 2000)}}`, pinned by `test_ref70_is_frozen_per_arm`. The run's grid steps each
schedule predicts:

| count | step | cumulative | instantaneous |
|---|---|---|---|
| V (1e-3) | 256, 512 | 512, 1000, 2000 | 2000, 4000, 8000 |
| centred, 9.61e-5 / 1e-4 / 3e-4 | 512, 1000 | 1000, 2000, 4000 | 4000, 8000, 16 000 |
| centred, 4.78e-3 | 1000, 2000 | 2000, 4000, 8000 | 8000, 16 000 |

Each window and early set: step 1000/2000/4000, early ≤ 128; cumulative 4000/8000, early
≤ 512; instantaneous 16 000, early ≤ 2000. On V the shared steps are not shared in effect
(`/challenge-pr` on #295, finding 3). A date at 512 lies in cumulative's early set and a date
at 2000 in instantaneous's, so each excludes the later schedule. V's effective bands are step
256–512, cumulative 1000–2000, instantaneous 4000–8000. On the centred count the overlaps are
real: 1000 for step and cumulative, 4000 for cumulative and instantaneous. Step's plants still
read "step and cumulative" 3 times in 10, because V's second rise at 8000 lifts cumulative's
plateau (its window is 4000/8000). Against seed 0 alone (beside, `bands_seed0`), the bands are disjoint:
V step 256, cumulative 512–1000, instantaneous 2000–4000; centred step 1000, cumulative
2000–4000, instantaneous 8000–16 000.

## Calibration (before the freeze, on the reference's own profiles)

Each of the ten seeds' profiles, planted under each schedule (`plant`) and read by the frozen
rule (`seeds.json` `calibration`; the leave-one-seed-out reference, `calibration_loo`, gives the
same table):

| planted | read as (of 10) |
|---|---|
| step | consistent with step 7; step and cumulative 3 |
| cumulative | consistent with cumulative 4; cumulative and instantaneous 5; **instantaneous 1** (seed 5) |
| instantaneous | consistent with instantaneous 9; **neither 1** (seed 6) |

So the truth survives on 10, 9 and 9 of 10; it survives alone on 7, 4 and 9. Each of the five
labels the plants produce comes from one truth only, except a lone "instantaneous" (1 of its 10
came from a cumulative plant). Both misses are V's date landing one grid step outside its
band. The centred count keeps the truth on 30 of 30 plants but excludes little: the
schedules' bands for it overlap more than V's do. **The first version** (`fbc2fdd`: the
interval window, one band pooled over arms, any arm sufficing) read the same plants as step
alone 1, cumulative alone 1 and instantaneous alone 9 of 10, and lost cumulative on 2. Its
centred count read "consistent" under every schedule on every plant. Not calibrated: a run whose
own init and order spread is narrower than ten seeds' (it shares seed 0's, "What the run is").
The plants carry no noise beyond the seeds' own. **The table is in-sample**: the rule was revised
on these 30 plants, and the leave-one-out version re-derives only the bands, so nothing tuned is
held out (#295, finding 2). Two of the four arms (9.61e-5, 1e-4) give the same date on every seed,
so "every arm" counts about three distinct arms (finding 5).

**The instantaneous plant depends on an untested assumption** (#295, finding 1). Under that
schedule the window is the single step 16 000. What the run holds there depends on how 70m's
states past its peak are reached. Taken literally, all are reached at once, so 16 000 holds 70m's
143 000, where V has fallen (1–5 on 8 of 10 seeds). Two softer versions are recorded in
`seeds.json` `calibration_late`; step and cumulative plants are unaffected:

| instantaneous's late states | its plants read |
|---|---|
| literal (all at once; the table above) | instantaneous 9, neither 1 |
| at 70m's same step | **instantaneous 2, neither 8** |
| at 70m's summed rate (cumulative) | **instantaneous 4, neither 6** |

Every failure is V's count; the centred count stays consistent with instantaneous on every
plant. So under this rule a "neither" on the run cannot be told from an instantaneous run whose
late states are not literal.

**Decided 2026-11-01, Blocked 127 (a), the user's, before any output on the run: V is not read
under instantaneous** (`c6_warmup.NOT_READ`; code `9c9e91f715ad`). Its window depends on the
late states, which are untested. That count gives no verdict there ("not read", which `joint`
treats as unreadable), so the instantaneous verdict rests on the centred count alone. Everything
else is as frozen above. This is a second in-sample revision, so `read` scores this rule and
records the rule as frozen on 2026-10-31 beside it (`reading.json` `beside.as_frozen`).
Re-derived by `recalibrate` from U0's `seeds.json` (record `calibration_127a.json`), the plants
read identically under all three late-state versions:

| planted | scored rule (127 (a)), every late version | as frozen, literal (above) |
|---|---|---|
| step | step 7; step and cumulative 3 | the same |
| cumulative | cumulative 2; **cumulative and instantaneous 7**; instantaneous 1 | 4; 5; 1 |
| instantaneous | **instantaneous 10** | instantaneous 9, neither 1 |

The cost: on a cumulative plant, instantaneous is now excluded on 2 of 10 (4 of 10 as frozen),
because V no longer votes against it. With leave-one-seed-out bands, the truth survives on
10 / 9 / 10 and alone on 7 / 2 / 10.

## The run's gate (U1, before any scored record on the run)

| item | fixed as | why |
|---|---|---|
| the populations | the 3 re-inits of 70m's config (`arch_null.reinit_model`, seeds 0–2) and the run's off-grid steps **7000 and 20 000** | 7000 is unit 3b's step; 20 000 lies past every band, so a rise that the run's schedule delays is inside one of them |
| per step | unit 3b's preflight: plant A (a common vector) and plant B (token-independent spread) per transition, through both counts; plant B's dose by 3b's rule 100 (a) (the smallest dose the uncentred count sees at a share ≥ 0.4) | the procedure that gated 70m |
| an arm is gated | it passes at both steps: re-init centred rate ≤ 5 %, plant B ≥ 0.4, plant A at 0.5 ≤ 0.1 (unit 3s's thresholds) | conservative across the two |
| the uncentred count is read | if rule 100 (a) finds plant B's dose at both steps | 3b reported the uncentred count's preflight without a gate; here a refusal is possible |
| known answer | the re-inits' centred rates equal unit 3b's 70m preflight (≤ 0.005 at each arm), else refuse; seed 0's records, run into U1's own directory first, equal unit 3b's e1 (≤ 1e-4, the tokens' hash equal), else `gate` refuses | the same config, seeds and code path |
| one version | `read` refuses unless `gate.json` reads readable and the run's records, the gate and the code share one code hash | added after `/challenge-pr` on #295, finding 4 |
| refused | no arm gated and the uncentred count not read: the exit's "read refused by its gate" | refuse rather than degrade |

`run --models run` refuses until `REF70` is set and `gate.json` reads readable.

**U1 (2026-11-01): refused.** Rule 100 (a) found no plant-B dose on the run at either step, and
the centred arms take their dose from it, so no arm is gated. "V read" and "an arm gated" were
not separable as this table implies (`status-C.md` "C6 unit 1"; Blocked 128).

### U1b: the centred count alone (Blocked 128 (a), the user's, 2026-11-01; frozen before `run --models run`)

One unit beyond the budget, extended by the user on the record. A revision made **after seeing
a gate output** (U1's refusal and its top-dose arms, beside), **before any output on the run's
grid**. Nothing else in this design changes.

| item | U1b |
|---|---|
| plant B's dose for the centred arms | **0.5, fixed**: unit 3b's 70m dose (`preflight_plants.json` `gate.b_dose`; `gate` refuses if they differ), not re-derived on the run; rule 100 (a)'s outcome recorded beside (`rule_100a_beside`) |
| V | **not read** (`uncentred_read` false), so the joint is the centred count's statuses alone |
| an arm is gated | as frozen: passes at both 7000 and 20 000 (null ≤ 5 %, B ≥ 0.4, A at 0.5 ≤ 0.1) |
| known answers | as frozen (seed 0's e1, the re-inits' null rates), **plus**: the arms' B, A and null shares equal U1's `arms_beside_top_dose` (≤ 1e-12; the same draws), else refuse |
| expected from U1's beside | 9.61e-5, 1e-4, 3e-4 gated; 4.78e-3 not (B 0.39 at 20 000) |
| the reading | `count_reading` and `joint` as frozen, on the gated arms; Blocked 127 (a) moot (it concerns V) |
| calibration (in-sample, computed after U1's gate output) | the 3 arms on the seeds' plants: truth survives 10 / 10 / 10, alone 1 / 2 / 10 (step / cumulative / instantaneous), the same under every late version (`calibration_127a.json` `centred_only_3arms`) |
| what it can separate | **step from the other two** (step survives on 10 of 10 step plants and 0 of 20 others); not cumulative from instantaneous (8 of 10 cumulative plants read both), nor step alone from step with cumulative (9 of 10 step plants read both) |

Labels as in "What each outcome changes", with one reading fixed now: "consistent with step"
(alone) here means the centred dates do not move with a ten-fold warmup on 70m; "consistent
with cumulative and instantaneous" means they move, the two not told apart; any label holding
step and another reads as not separating step. Run: `run --models seed0`, `gate --centred-only`,
`run --models run`, `read`, all into `data/trackC/c6/2026-11-01-u1b/`, one code hash.

## What each outcome changes (fixed now)

| label | §7 and F1 for 70m |
|---|---|
| consistent with step | on 70m the dates do not move with a ten-times warmup: fixed in steps (or tokens, not told apart), not by the LR; F1 narrowed for 70m |
| consistent with cumulative (alone, or with instantaneous) | the dates move with the warmup, as the summed rate does: a dose, not a fixed step, on 70m |
| consistent with instantaneous alone | the dates wait for the rate to reach a level; reported with its calibration (a cumulative plant read so once in 10) |
| consistent with step and cumulative | as unit 3b: the LR level excluded on 70m, step and dose not told apart (in calibration only step's plants read so) |
| not told apart | the ten seeds' range is too wide on this grid; §7 unchanged |
| neither | the run's dates follow none of the three; reported with the counts' profiles; §7 says so |
| unreadable or refused | §7 unchanged; the refusal reported |

## Not this design

410m's dates (C4); any registration; tokens against steps (the batch is Pythia's in both); the
data order after ~16 steps; the run's V at other tolerances; `pythia-14m-warmup01`.

## Exit condition and budget

**Exit:** both dates read on the run and labelled by the frozen bands, or the read refused by
its gate. **Budget: U0 + U1 + close-out (3 units)**, under an hour on the RTX 3080; at the budget
it closes at what it knows unless the user extends it on the record.

| unit | what | done when |
|---|---|---|
| 0 (this) | the run's schedule, init and first batch from the weights; 70m's date range from seeds 0–9; the bands, the reading and the gate frozen | this file frozen, `REF70` set and pushed |
| 1 | Blocked 127 decided; `run --models seed0`, then `gate` on the run's steps 7000 and 20 000; `run --models run` (18 steps, offline); `read` | the label, or the gate's refusal |
| 1b (Blocked 128 (a), beyond the budget) | the centred count alone, plant B at 0.5 ("U1b" above): `gate --centred-only`, `run --models run`, `read` | the centred label, or the gate's refusal |
| close-out | writing only: what C6 established, §7 and F1 for 70m in `paper/` if the label is clean, Parked items homed, the card current | the card current (**done 2026-11-01**, Blocked 129 (a), the user's: `status-C.md` "C6 on one page (closed 2026-11-01)") |

## Worth challenging (for the PR)

1. **The rule was revised after the reference's profiles were read.** No output on the run
   existed, but the window, the per-arm reference and the every-arm rule were chosen after
   seeing which plants the first version lost. The alternative was to freeze `fbc2fdd`'s rule as
   it was and report its weak calibration (step alone 1 in 10). I took the revision because the
   protocol's calibration gate exists to catch exactly this. A reviewer could ask for the run to
   be read under both and the first reported beside.
2. **Ten seeds as the reference for a run that shares seed 0's init and first batch.** The
   union over seeds is conservative and makes the bands meet (V at 512 and 2000). Against seed 0
   alone they are disjoint (beside). Rejected: seed 0 alone as the scored reference, since the
   run's order diverges from seed 0's after ~16 steps and one seed's date is one draw.
3. **"Every counted arm must fit."** It is stricter than 3b's reading. In calibration it never
   cost the truth (30 of 30 on the centred count), but a run with one odd arm reads "neither" or
   loses a schedule. A majority of arms calibrated identically on these plants; "every" was
   taken as the stricter at equal measured power. "Any arm" (the first version) excluded nothing.
4. **The instantaneous plant assumes everything past the rate's peak is reached at once.** That
   is the hypothesis taken literally (a date waits for a rate level), and it makes the run's
   16 000 hold 70m's final state. Two softer versions, measured after `/challenge-pr` on #295,
   leave the instantaneous truth at 2 and 4 of 10 ("Calibration"). Blocked 127 (a), the user's:
   V is not read under instantaneous; the rule as frozen is read beside.
5. **70m's arms are not re-derived on the run** (the gate checks them on its steps 7000 and
   20 000). The dates are compared at the same tolerances, but the run's energy scale at a given
   step could differ from 70m's.
6. **The peak is pinned to ~10 % only from 16 000 on, and the mid-warmup proxy failed**
   ("What the run is"). Inside both warmups the maps depend on peak / W alone, which the early
   biases pin to 2–6 %. H0 rests on Belrose et al. plus the weights, not on a released config.
7. **The coarse grid.** The design reads the 18 steps 3b read, so between 4000 and 16 000 only
   8000, and both calibration misses are one grid step. The run releases every 1000 steps (155
   branches, as `pythia-70m`; the seeds' branches not checked), so a finer grid there is possible. It was rejected for U1 because
   it changes 3b's instrument and needs about 24 more checkpoints per model on a nearly full
   disk. It is the first thing to add if U1's label sits on a shared band step.
