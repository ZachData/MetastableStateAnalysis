<!-- trackC_writeup/design-C4pc.md -->
# Track C — design: C4′ (c), the replicate pair

**Status: frozen 2026-11-18 on Blocked 166 all (a)** (decided by Claude on the user's "Decide
Blocked 166 (D1–D4) for C4′ (c)'s design [your choices]"; yours to overrule): both arms to 4000,
π a permutation of C4's rows, five entries, C4's training order. Proposed 2026-11-17 (U0); revised
after `/challenge-pr` on #361 (accept with changes, findings 1–5 taken). **U1 built 2026-11-18**
("U1 as built" below; `status-C.md` "C4′ (c) unit 1"); the five entries are drafted below and
are STATE Blocked 167, yours. No arm has taken an update.
Opened on Blocked 165 (a), the user's ("Do Blocked 165 (a): C4′ (c)'s design"). Row:
`docs/ROADMAP.md` §8 C4′ (c), §10 item 18. Scan: `trackC_writeup/lit-C4pc.md`. Nothing was
computed, trained, downloaded or registered. The arithmetic below is
`tools/math_checks/c4pc_nearer.py`'s.

## What it is for

C4 trained one pair from seed 0's init on Pythia's data order: A0 (Pythia's schedule) and F (the
ramp removed). Its registered reading failed (`P-SCH1`, "neither"), and its close-out read leans
after the label (`status-C.md` "C4 on one page", items 6–10). C4′ held items 9 and 10 at tier 1
and left 6–8 untouched (`status-C.md` "C4′ on one page"). Every lean rests on one pair. **(c)
trains a second pair, A0′ and F′, from the same init on another order of the same rows.** It
does three things C4 could not:

1. **It measures the run-to-run spread** of every date at a fixed init (item 6: the summed rate
   is excluded only by a factor of 1.73 in A0's date, against a tolerance that is the grid's).
2. **It re-reads R2 and R3 under design-C4's rule as written**, whose tolerance includes "A0's
   rerun ratio if one exists" (`design-C4.md` "The predictions on the flat arm"). No rerun
   existed, so `c4_read.rho_at` never implemented the term. A0′ is that rerun. Only V's label
   can change this way; the centred label cannot (M2).
3. **It tests C4's leans on a held-out pair, with their rules fixed and registered before
   either arm trains** (items 7–10: the transient and the shared end set, V one block earlier,
   induction's date and the heads' layer, F′ off A0′'s path, and F's late departure against a
   same-init replicate, U2's Parked 3).

It reaches nothing on `pythia-410m`, the paper's model.

## Found while designing (before any decision)

1. **A0′ cannot reach 4000 on shard 0's unused rows.** Shard 0 holds 7 320 644 whole rows. C4
   used 4 096 000 for training and 64 for H4's held-out loss. That leaves 3 224 580 rows, which
   is 3149 updates of 1024 rows (math check (3)). ROADMAP C4′ and Blocked 165 assumed "the unused
   3.2M rows" carry A0′ to 4000. They carry either arm to 3149 at most. D2 below.
2. **No date that C4's rule reads exists on a pair stopped at 1000.** `c4_read.date_on` takes
   the plateau as the median at 1000, 2000 and 4000, and C4's gate (V and the τ arms) runs at
   3500. So on a pair stopped at 1000, R2, R3, item 6, the end sets (item 7), V's sets (item 8)
   and induction's date (item 9) can be read only under new rules with their own gates, not
   C4's. D1 below.
3. **R3 needs A0′ only; item 6 needs the pair.** R3 is "neither" by one count on A0's side
   (`status-C.md` "C4 on one page" item 4); with A0′'s dates, C4's own F is re-read against A0 at
   ρ = max(the grid's ratio, A0's rerun ratio), design-C4's rule (M2), and V's label can flip.
   The centred label cannot flip that way, since its bands meet at ρ 1.647, below item 6's margin
   of 1.727 (M2; `/challenge-pr` on #361, finding 1). Item 6 is read from the paired replicate
   (M3), which needs F′ to 4000 too.
4. **The heads' overlap count mostly measures the layer** (C4′ U1's A2: every head ≥ 0.5 on the
   ten 70m seeds and on warmup01 is in layer 3 or 4; F's top head is L2H6). So item 9's head
   lean is stated as the **layer** of the top head, not as an overlap count.

## The arms

| arm | schedule | init | data |
|---|---|---|---|
| **A0′** | Pythia's (`c4_train.lr_pythia`), as A0 | seed 0's (the released step 0), as A0 and F | order π (D2) |
| **F′** | flat at 1e-3 to W, then Pythia's cosine (`lr_flat`), as F | the same | the same order π |

Everything else is C4's: batch 1024 × 2049, Adam, weight decay 0.1, clip 1.0, bf16 autocast with
fp32 master weights, checkpoints on `c4_train.SAVE_STEPS` (the 28-step dense grid and 3500).
**`c4_train.py` is not edited**: its `code_sha` (`1c1e1d2df083`) is the C4 arms'. A new module,
`c4pc_train.py`, supplies the rows and the arm names and calls C4's step, optimizer and schedule
functions unchanged, recording its own code hash beside C4's.

**D2, the order π.** (a), recommended: **a seeded permutation of C4's 4 096 000 training rows**,
written once in order to one contiguous file (16.8 GB on `HDD_1TB`, 155 GB free), its sha256
recorded. By 4000 A0′ and F′ have seen exactly C4's rows, in another order and in other batches.
Only the order varies, which is what a replicate at a fixed init should vary (`lit-C4pc.md` 1–2),
and no download is needed. (b): fresh rows in Pythia's order from row 4 096 064, contiguous as
C4's. They reach 3149 updates in shard 0; past that, shard 1 is a 30 GB download. This varies the
rows and the order together. H4's 64 held-out rows (iteration 4000's first 64 in Pythia's order,
`c4_train.HELD_OUT`) stay unseen by both arms under either option, so C4′ U2's KL rows serve
unchanged.

**D1, the horizon.** C4 measured 29.5 h an arm to 4000 (26.5 s an update), or 7.4 h to 1000.

| reading | (a) both to 4000, ≈ 59 h | (b) A0′ to 4000, F′ to 1000, ≈ 37 h | (c) both to 1000, ≈ 15 h |
|---|---|---|---|
| M1 A0′'s dates, the rerun ratio | yes | yes | no (found 2) |
| M2 R3 on C4's pair (V can flip; the centred label cannot) | yes | yes | no |
| M3 R2′ and R3′ on the new pair | yes | no (F′ undated) | no |
| L7, L8, L9, L9h | yes | no (each needs 4000 or the gate at 3500) | no |
| L10a, F′ off A0′'s path early | yes | yes | yes |
| L10b, F's late departure against A0′ | yes | yes | no |

**Recommended: (a).** It is the only option that reads every lean with C4's own definitions. (b)
and (c) leave the "no"s readable only under rules redefined on a 1000 horizon, each a new
instrument needing its own gate. That costs more design than (a)'s extra GPU hours.

## The harness's checks (U1, before either arm trains past step 64)

| check | against | pass |
|---|---|---|
| H1′ init | the released step 0 | bit-equal (C4's `h1`) |
| H2′ rows | the source shard | the file's row k equals shard row π(k) at every k (lengths, and the bytes at 4096 sampled k, seed fixed); its sha256 recorded; under π = identity, `c4pc_train`'s batches equal `c4_train.batch_of`'s (a CPU test) |
| H3′ divergence | C4's rule (`c4_train.diverged`) | each arm's loss watched as it trains (after registration, finding 3); a divergence is a refusal |
| H4′ loss | A0 at 1000 and 4000, on the 64 held-out rows | A0′ within max(2 %, the largest gap between `pythia-70m` and seeds 1–9 on the same rows at 1000 and 4000), the tolerance computed in U1 before A0′ trains (finding 5: 2 % was set for a same-order copy, and an order replicate may differ more); outside is a refusal and nothing is read |

## The readings

All are read in U3, after both arms are at 4000, with code frozen and pushed in U1. Instruments
are C4's and C4′'s, unchanged, on the new arms' records: `c4_read` (records on v1's 7 prompts,
`date_on`, `label`, `centred_label`, C4's gate at 3500 on F′), `c4_close` (`per_transition`,
`majority`, `settled`, `induction_of`, `rho_star`), `c4p_path` (K on the 64 held-out rows, `δ`,
`window_reading`). New code goes in `c4pc_read.py`, so the registered hashes stay put.

### Measured (not registered; each is reported whatever it reads)

| | what | rule |
|---|---|---|
| M1 | A0′'s date on V and each τ arm, and its rerun ratio q = max(t_A0′ / t_A0, t_A0 / t_A0′) | `date_on` on the dense grid |
| M2 | **R2 and R3 on C4's own pair**: F's dates against A0's at ρ = max(`rho_at`(t_A0), q) | design-C4's rule with its rerun term. `P-SCH1` is not re-scored: its registered instrument names the grid's ratios only |
| M3 | R2′ and R3′: F′ against A0′, as C4 read F against A0 | `label`, `centred_label`, at ρ = max(grid, q) |
| M4 | F′'s rerun ratio against F, per count | as M1 |

**What M2 can and cannot change** (corrected after `/challenge-pr` on #361, finding 1). `label`
stops at "A0 past the separable range" once the two bands meet, at ρ ≥ (2W / t_A0)^{1/3}. For
the centred date (t_A0 = 640) that cap is 1.647, below item 6's margin of 1.727. **So M2's
centred label is "neither" (q < 1.647) or "A0 past the separable range" (q ≥ 1.647). It can
never read "consistent with cumulative".** "Past the separable range" would say the measured
spread makes step and the summed rate indistinguishable at that date. For V (t_A0 = 384, cap 1.95,
margin 1.47) **R3 can flip** to "consistent with cumulative" if q ≥ 1.47 (A0′'s V date at 256 or
earlier, or 512 or later). Beside, q is printed against `rho_star` for each count, as numbers.
One rerun is a spread of one draw, and M2 says so.

**M2 is unpaired, and it reports; it does not relabel** (finding 2). F and A0 share init and
order, so the right spread for "F against A0" is a paired one: M3 (F′ against A0′, same init,
another order) is the replicate of C4's comparison. **§7's item-6 wording follows M3's label and
dates**, with M2 beside. `P-SCH1` was registered with the grid's ρ and stays scored as read, and
M2 changes no sentence of §7 by itself.

### The registered family (D3): five entries, each scored alone, all five reported

Each lean is stated **against A0′, never against A0**, so a large order spread breaks no
prediction by itself. τ arms are those C4's gate passes on F′ at 3500. If none pass, L7 and the
centred parts of L9 are "unreadable" and score nothing; if V is refused, L8 is unreadable.

| entry | C4 item | statistic and rule | h1, the lean (holds) | on C4's pair (known answer) |
|---|---|---|---|---|
| **L7** | 7 | the centred count at each gated τ arm: the prompts violating at each transition on the dense grid (`per_transition`); a transition is majority-violated at > 3.5 of 7. **End set** is the majority set at 4000. **Transient** is a transition majority-violated at some grid step ≤ 1000 and not in the end set | at every gated τ arm: **F′ has a transient, A0′ has none, and both end sets are {T1, T2}** | holds at all three gated arms: F's transient {T3, T4, T5}, A0 none, both end on {T1, T2} (`close.json` `transitions`, read 2026-11-17 for this table) |
| **L8** | 8 | V's (uncentred) majority set at 1000, 2000 and 4000 | at each of the three, both sets non-empty and **min(F′'s) = min(A0′'s) − 1** | holds at 1000, 2000 and 4000: F's minimum is T1, A0's T2 |
| **L9** | 9 | induction's half-rise (`induction_of`, the top head's score, `date_on`'s plateau); d_X = \|ln(t_F′ / m_cum(t_A0′))\| for event X, m_cum(t) = t² / 2W | **(i) t_F′ is nearer m_cum(t_A0′) than t_A0′ in log**, which is t_F′ < t_A0′^{3/2} / √(2W) (math check; 398 at t_A0′ = 768), **and (ii) d_induction < d_centred at every gated τ arm** | holds: 320 < 398; d 0.44 against 1.09 |
| **L9h** | 9 (the heads) | the top head at 4000 by induction score | **F′'s is in layer 2 and A0′'s in layer 3 or 4** | holds: F L2H6, A0 L3H6 |
| **L10** | 10 | (a) `c4p_path`'s early reading with F′ as F and A0′ as A0. (b) Over F's 12 late steps (320–4000), r_F(s) (U2's, F against A0) against r_A0′(s) = min over A0's states of K(A0′_s, A0_t); both full precision from one harness | **(a) "another path" early and (b) r_F(s) > r_A0′(s) at ≥ ⅔ of the late steps (8 of 12)** | (a) holds (U2); (b) has no known answer |

Each entry's falsifier is its lean not holding, with every cell readable. "Unreadable", a
refused gate and a harness refusal score nothing. **The family is fixed now:** these five
statements, these cells, all five reported in the close-out whatever they read. Nothing is
added after output. L9's (i) is weak alone, since C4's centred date also passed it (48 < 303 at
t_A0 = 640). (ii) is the part that says "more nearly than the counts".

### The gates (U1 for L7–L9h; U3 for L10 (b), which needs A0′)

The calibration gate (`CLAUDE.md`): each rule run on a planted effect and a planted null over the
real design points before it freezes.

| entry | planted effect | planted null | passes iff |
|---|---|---|---|
| L7, L8, L9, L9h | **C4's pair (F, A0)**: the known answer above | **(released `pythia-70m`, A0)** as (F′, A0′): same init, order and schedule, so no lean should hold. Read on Pythia's 13 grid steps ≤ 4000, which A0's dense grid contains | the effect reads "holds" and the null "does not hold", per entry |
| L10 (a) | C4′ U2's gate (seeds 1–9, "another path" 9 of 9) | U2's (the released run against fp16-rounded A0, "one path") | passed 2026-11-17 (`data/trackC/c4p/2026-11-17-u2/gate.json`); stands unchanged |
| L10 (b) | seeds 1–9 as pseudo-F against A0 on their late steps (512–4000, 4): r_seed(s) > r_A0′(s) | the released run against fp16-rounded A0 (U2's null path) | the null reads "beyond A0′" at < ⅔ of its 4 late steps and ≥ 8 of 9 seeds at ≥ ⅔. If refused, (b) is reported as description and L10 scores on (a) alone |

**What these gates do not test** (finding 4: both sides pass by construction). C4's pair is where
the leans were formed, so "holds" there only checks that the code computes the lean as C4 read
it. The (released, A0) null is nearly an identity, so "does not hold" there only checks that the
rule does not fire on a copy. **Neither side tests whether run-to-run variation alone can read
"holds".** So a third gate is added for L7, L8, L9 and L9h:

| entry | null with real variation | passes iff |
|---|---|---|
| L7, L8, L9, L9h | each of `pythia-70m-seed1..9` as pseudo-F′ against the released `pythia-70m` as pseudo-A0′: same schedule, another init and order, on the 13 Pythia grid steps ≤ 4000 (records and heads cached from C6 U0 and C4′ U1). No schedule differs, so every "holds" is a false positive | ≤ 1 of 9 seeds reads "holds", per entry |

An entry whose seed null fails is reported as description and scores nothing. L9 is computed with
m_cum's W as on A0′. For a pseudo-pair on one schedule, (i) asks whether one run's date is below
the other's to the 3/2 power over √(2W). It is a test of the rule on two dates that differ only by
spread. L10 (b)'s seed gate already has real variation on its effect side.

## Order of work and refusals

1. U1: `c4pc_train.py`, π's file, H1′–H2′ (no training step); `c4pc_read.py` and its tests;
   the gates for L7–L9h (known answer, the (released, A0) null and the seed null); H4′'s
   tolerance; the five entries drafted (fields as `P-SCH1`'s; ids the next free `P-SCH` numbers,
   2–6, since A1's draft was declined unregistered). **The user registers (D3 (a)) before either
   arm takes its first update** (finding 3: in C4, F's transient was already a majority at 48 and
   64, so a look at step 64 would have shown part of L7's answer). `c4pc_train` refuses to train
   until each entry is in `claims/registry.json` or recorded as declined, and `c4pc_read` refuses
   to score before the same.
2. U2 (runner): A0′ to 1000 and H4′; F′ to 4000, its loss watched for divergence, and C4's gate at
   3500 on F′; A0′ 1000 → 4000; H4′ at 4000. C4's order (D4).
3. U3: records, induction and heads on both new arms; the K passes (F′ × A0′, A0′ × A0; about an
   hour, as U2's); L10 (b)'s gate; then M1–M4 and L7–L10.

Refusals: a divergence, an H1′, H2′ or H4′ miss (nothing is read); `c4pc_read` refuses to score
an entry before its gate's record exists, and refuses L10 (b) before its gate.

## Cost

GPU ≈ 59 h for D1 (a) (29.5 h an arm, C4's measured rate), plus about 3 h for records, heads and
K. Disk: 2 × 8.4 GB of checkpoints and 16.8 GB for π's file (D2 (a)). In wall-clock terms that is
about 3 days of runner time and 3–4 sessions, which fits before mid-December.

## Exit condition and budget

**Exit:** the five entries scored, or each refused by its gate, and M1–M4 written. Items 6–10
are marked held, broken or untouched in a close-out page. `paper/` §7 gets a sentence for each
outcome that changes what §7 says (M3 for item 6, M2 for R3, L10 (b) for the late window). **Budget:
4 units + close-out** (U0 this design, U1, U2, U3). At the budget it closes at what it knows.

## Carried in

- **The two `c2_seeds` refusals** (C4 U1's Parked 0): not needed. No unit derives a centred arm
  (70m's four frozen τ are read), so they stay with ROADMAP C4′.
- **U2's Parked 2, the finer A0 grid** (≈ 1 h): not folded in; it stays beside (c) in ROADMAP
  C4′. It could share a runner session in U2 if the user wants it.
- **`lit-C4pc.md` Parked 1** (Kosson et al.'s full text): read before U3, or dropped with why.

## Decisions (STATE Blocked 166; each yours, recommendation first)

| | question | (a), recommended | others |
|---|---|---|---|
| D1 | horizon | both arms to 4000, ≈ 59 GPU-hours: every lean is read with C4's own definitions | (b) A0′ to 4000, F′ to 1000, ≈ 37 h: M1, M2 (R3 can flip, the centred label cannot), L10 only, and not M3, the paired spread §7's item-6 wording follows; (c) both to 1000, ≈ 15 h: L10 (a) only, unless new rules are designed |
| D2 | the order | a seeded permutation of C4's 4 096 000 rows, one 16.8 GB file: varies only the order, no download | (b) fresh rows in Pythia's order: varies the rows too, and shard 1 (30 GB) past update 3149 |
| D3 | registration | the five entries L7–L10 (the next `P-SCH` numbers, 2–6), by you after U1's gates and before either arm trains at all | (b) one joint entry (all five must hold), which is stricter and says less when it fails; (c) tier 1, rules frozen and pushed |
| D4 | order of training | C4's: A0′ to 1000 (H4′), F′ to 4000, A0′ to 4000 | (b) both to 4000 back to back, A0′ first |

**Decided 2026-11-18: all (a)**, by Claude on the user's delegation. Why: D1 (a) is the only
horizon that reads every lean with C4's own definitions, and the 22 GPU-hours over (b) fit before
mid-December; D2 (a) varies only the order and needs no download; D3 (a) keeps each lean scorable
on its own, with all five reported; D4 (a) is C4's, so H4′ at 1000 is read before F′ trains.

## U1 as built (2026-11-18)

Code `c4pc_train.py` (`6517b85f75b0`) and `c4pc_read.py` (`2de07248985b`; pushed as
`32a57885224d` in `52303d8` before any output of the new arms, then threshold labels added in
comments only, and the gate re-run); tests `tests/test_trackC_c4pc.py` (11, CPU). Three
choices made while building, none seen against an output:

1. **The gate pairs use C4's gated τ arms** (9.61e-5, 1e-4, 3e-4; `data/trackC/c4/2026-11-12-u2/
   gate.json`) and read V, since no released run has a step 3500 to gate on.
2. **`init`** saves an arm's step 0 with no update, so H1′ runs before registration; `train`
   refuses until every entry is registered or declined (`<root>/declined.json`, the user's).
3. **L10's K pass** (the `c4p_path` instrument on keys A0′@29 steps, F′@28, and A0′ against A0's
   29) is wiring, written in U3 before any K on the new arms; the rule (`c4pc_read.l10`, `l10b`,
   `l10b_gate`, and `c4p_path.read_pseudo` unchanged) is frozen here.

**The gates for L7–L9h pass** (`data/trackC/c4pc/2026-11-18-u1/gate.json`, code `2de07248985b`):
each reads "holds" on C4's pair, "does not hold" on (released, A0), and 0 of 9 seeds hold. **The
seed null is weak for L9**: on Pythia's 13 steps every seed and the released run date induction
at 1000 (no grid step lies between 512 and 1000), so (i) fails on every seed by the grid alone
(1000 against the bar 591). It is weaker for L7 than the effect side too, since Pythia's grid
has 10 steps ≤ 1000 to the dense grid's 23. Near misses in the seed null: seed 1 has L8's
pattern at 4000 but not at 1000; seeds 5 and 8 have a top head in layer 4 (L9h's A0′ side), none
in layer 2.

## The five draft entries (Blocked 167; yours to register or decline)

Ids are the next `P-SCH` numbers, 2–6 (`c4pc_train.ENTRY_IDS`; not written whole, since
`tools/check_registry.py` fails on an unregistered id in the tree). **Shared fields:** claim
`H-RESIST` (as `P-SCH1`); phase C, experiment C4′ (c); cost F; evaluable measurement; relevance
0.6; `registered_provenance` gated; `gate` `data/trackC/c4pc/2026-11-18-u1/gate.json` (record
path; main tree); `source` this file; **instrument** `c4pc_train.py` `6517b85f75b0` (the arms,
calling `c4_train.py` `1c1e1d2df083`'s recipe unchanged), records by `c4_read records` (C2 unit
3b's per-state row on the 7 v1 prompts, float32 eager), the rule in `c4pc_read.py`
`2de07248985b`. **Common to every statement:** *on `pythia-70m` trained from seed 0's init on
order π (seed 20261118, a permutation of C4's 4 096 000 rows) under Pythia's schedule (A0′) and
with the LR warmup removed (F′: 1e-3 to step 1430, then Pythia's cosine), both to 4000 on C4's
dense grid*; τ arms are those C4's gate passes on F′ at 3500. **Common falsifier:** the lean does
not hold with every cell readable. "Unreadable" (no gated τ arm, V refused, an undated event), a
harness refusal (H1′, H2′, H4′) or a divergence score nothing.

| entry | statement (h1) | h0 |
|---|---|---|
| 2 (L7) | At every gated τ arm, F′'s centred count has a transient (a transition violated on > 3.5 of 7 prompts at some grid step ≤ 1000 and not in the 4000 majority set), A0′'s has none, and both 4000 majority sets are {T1, T2}. | The transient and the shared end set were C4's pair's: F′ has none, A0′ has one, or an end set differs. |
| 3 (L8) | V's majority sets at 1000, 2000 and 4000 are non-empty on both arms, and F′'s lowest transition is one below A0′'s at each. | V's lowest majority transition is not one block earlier on F′ at some of the three steps. |
| 4 (L9) | Induction's half-rise (the top head's score, `c4_read.date_on`) satisfies (i) t_F′ < t_A0′^{3/2} / √(2W), W = 1430 (nearer the summed-rate map than the step map in log), and (ii) \|ln(t_F′ / m_cum(t_A0′))\| < the same distance for the centred date at every gated τ arm, m_cum(t) = t² / 2W. | Induction's date is no nearer the summed-rate map than the centred date's, or is nearer step. |
| 5 (L9h) | The top induction head at 4000 is in layer 2 on F′ and in layer 3 or 4 on A0′. | The flat ramp does not move the top head to layer 2. |
| 6 (L10) | (a) F′ reads "another path" in the early window (1–256) against A0′'s path (`c4p_path`'s per-token KL on C4's 64 held-out rows, A0′ as the reference), and (b) over F's 12 late steps (320–4000), C4′ U2's r_F(s) exceeds r_A0′(s) = min over A0's states of K(A0′_s, A0_t) at ≥ 8 of 12. If (b)'s gate (U3: the released run's null < ⅔ and ≥ 8 of 9 seeds ≥ ⅔) refuses, L10 scores on (a) alone. | F′ is on A0′'s path early, or (with (b) gated) F's late distance from A0 is within an order replicate's. |

`null_construction` for 2–5: the planted (released, A0) pair and seeds 1–9 against the released
run, on Pythia's 13 steps ≤ 4000 (no schedule differs; "holds" there is a false positive), with
the seed null's grid limit above. For 6: C4′ U2's gate for (a); (b)'s in U3. `notes` for all:
70m only; the leans were formed post hoc on C4's pair, so a held entry is one replication at
another order and the same init; bf16 against Pythia's fp16.

## Worth challenging (for the PR)

1. **The registered leans were formed post hoc on the pair whose order (c) changes.** A lean
   that holds on (c) has replicated once on another order at the same init. That is not a
   test at another init, and not on 410m.
2. **L10 (a) reuses U2's gate**, which was calibrated with A0 as the reference, not A0′. No
   released twin of A0′ exists to rebuild the null. The gate is the instrument's, at this model
   and grid; A0′'s δ is its own.
3. **L10 (b)'s null mixes precisions**: the released run is read against fp16-rounded A0 (as
   U2's null), while A0′ is read against A0 at full precision.
4. **D2 (a) shares C4's rows.** By 1000 a quarter of A0′'s rows are among A0's first 1000
   updates' (at random positions); by 4000 all are. Rejected alternative: fresh rows (D2 (b)),
   which vary content as well as order.
5. **L9's (ii) compares two noisy log distances** whose difference on C4 (0.44 against 1.09) is
   one pair's. Rejected: dropping (ii), which leaves a lean the counts also satisfy.
6. **Five entries can be read as a menu.** All five are reported, and none is added or dropped
   after output. The joint entry (D3 (b)) is the stricter alternative.
