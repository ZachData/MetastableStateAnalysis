<!-- trackC_writeup/design-C4pdf.md -->
# Track C — design: C4′ (d) and (f), the transient's dose and a schedule switch

**Status: proposed 2026-11-23 (U0, writing only); decisions D1–D4 are STATE Blocked 172.**
Opened on Blocked 169, decided by the user 2026-11-23: "(e) and (g) first", then (d), (f). (e)+(g)
closed in #370. (d) and (f) both train short new runs from states C4 already holds, and both ask
what sets F's transient, so they share one design and one harness. Scan: `lit-C4pdf.md`.
Arithmetic: `tools/math_checks/c4pdf_bands.py`. **Nothing below has been computed or trained.**
Every number quoted from the four trained arms (A0, F, A0′, F′) was read by earlier units and is
cited to them.

## What it is for

C4 found that F (flat at the peak from step 0) rises through a **transient** that A0 (Pythia's
ramp) lacks: from step 48, blocks 1–5 lower the centred energy on most prompts (T3–T5 majority
violated, 48–320), before both arms settle on the same end set {T1, T2} (`status-C.md` "C4 on one
page", item 7). F′ repeated it; A0′ showed {T3} at one grid step (640) only ("C4′ (c) on one
page", item 5). Nothing so far says **what sets the transient**: the rate's size, its dose (the
summed rate), or the early state that a flat rate meets at its first updates. Induction's date has
the same open question in a different form: `P-SCH4` held it nearer the summed rate than the
counts, and the rate and loss clocks were not told apart (C4′ (a)). The ROADMAP rows (§8 C4′):

- **(d) the transient's dose** (`lit-C4.md` Parked 1, Qu & Liu's peak dependence): flat arms at a
  fraction r of the peak. Is the transient a threshold in the rate, and does induction's date
  scale as 1/r?
- **(f) a schedule switch from a saved state:** A0's step-256 weights continued at the peak, against
  the same restart at A0's own rate. Is the transient the early state's or the rate's?

**Found while writing (corrects the ROADMAP row's horizons).** The row costs (d) as F½ to 768 and
F⅒ to 384. Under the dose clock F⅒'s transient would begin in [384, 600] and its induction in
[2560, 4000] (F's 48 and 320, divided by r, ρ = 1.25; the math check). So F⅒ stopped at 384 could
not see the transient where the dose clock puts it, and "none by 384" would not separate a
threshold from a dose. F½ stopped at 768 reaches the dose band's upper edge (800) without passing
it. D1 below therefore proposes F½ to 1000 and F⅒ to 768, each one grid step past its band.

**Also found.** A0 itself already argues against a pure dose clock for the transient. Its summed
rate reaches F's onset dose (48 peak-steps) at step 370, where its rate is 0.26 of the peak, and it
has no transient there (`status-C.md` "C4 on one page", item 7's table: 320 "1 0 1 0 0 0", 512
"0 0 3 1 1 0"). So either the transient needs a rate above some threshold, or it needs the early
state (A0 at 370 has trained), or both. (d) varies the rate at the init's state; (f) varies the
state at the peak's rate. Together they separate the two.

## The arms (new runs; C4's recipe with one thing changed)

Every arm is `pythia-70m`'s architecture under C4's recipe (`c4_train.build`, `make_opt`,
`step_once`, `save_ckpt`, called unchanged, so `c4_train`'s hash `1c1e1d2df083` stays): AdamW
(0.9, 0.95), ε 1e-8, weight decay 0.1, clip 1.0, bf16 autocast, batch 1024 rows of 2049 tokens, on
**C4's data order** (`c4_train.batch_of(rows, i)`, iteration i reads A0's and F's i-th batch).

| arm | from | rate | updates | saved steps | change against C4 |
|---|---|---|---|---|---|
| F½ | seed 0's init (`pythia-70m` step0, as A0, F) | 0.5 × 1e-3, flat | 0 → 1000 (D1 (a)) | C4's `GRID` ≤ 1000 (23 states) | the rate only (F is the same at r = 1) |
| F⅒ | seed 0's init | 0.1 × 1e-3, flat | 0 → 768 (D1 (a)) | C4's `GRID` ≤ 768 (22 states) | the rate only |
| B_P | A0's saved step-256 weights (fp32), **fresh AdamW** (zero moments, counter 0) | 1e-3, flat | 256 → 512 | 256 + k, k ∈ K | the state (against F: same rate, same fresh Adam) |
| B_O | the same weights, fresh AdamW | A0's own, `lr_pythia(256 + k)` (0.179 → 0.358 of the peak) | 256 → 512 | 256 + k, k ∈ K | the control for the restart |

K = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256): C4's early grid as
offsets, so a branch's transient is seen at F's own resolution from its first update. The
branches read A0's batches 256–511, the ones A0 itself took next. **Checkpoints hold weights only**
(`HDD_1TB/mets_data/c4/A0/ckpt/step256/model.safetensors`, 282 MB; `resume.pt` is A0's at 4000), so
A0's moments at 256 do not exist: D2 (b) costs rebuilding them.

GPU, at C4's measured ≈ 26.5 s an update (C4′ (c): 8000 updates in ≈ 59 h): F½ 7.4 h, F⅒ 5.7 h,
each branch 1.9 h; **≈ 17 h in all** at D1 (a), D2 (a). Records (c4_read's per-checkpoint e1 / e1c
on v1's 7 prompts, float32 eager) and heads (`c4_close heads`) on the 78 new states: under an hour.
Disk ≈ 22 GB on `HDD_1TB` (123 GB free 2026-11-23), under `mets_data/c4pdf/{Fh,Ft,BP,BO}/`.

## The readings (rules fixed here; constants placed in U1 before any new arm trains)

The instruments are C4's, unchanged: `c4_close.per_transition` (prompts violating at each
transition T0–T5, by `c4_read.series`' rule) at **C4's three gated centred arms** (τ 9.61e-5,
1e-4, 3e-4), and the induction score per head (`c4_close heads`). A reading holds "at every gated
arm" as L7's did. **No new centred arm is derived**, so the arms' τ are not re-gated on the new
runs (no new run reaches C4's gate step 3500); "Worth challenging" 2.

### T, the transient (both (d) and (f))

On a series of states s₀ < s₁ < … (an arm's grid, or a branch's offsets), the **transient** is a
run of **≥ 2 consecutive saved states** at each of which some transition outside the end set
{T1, T2} is majority violated (> 3.5 of 7 prompts). Its **onset** is the run's first state.

Placed so that the four trained arms read as C4 and C4′ (c) described them: F and F′ have one from
48; A0 has none; A0′'s {T3} at 640 is one state, so none. **This placement was made after seeing
A0′** (a one-state blip is excluded by construction); the gate checks it on all four, and the new
runs are unseen. The size of the transient (the count outside {T1, T2}, summed over the run) is
reported beside, never read.

### d1, the transient against the rate (F½, F⅒)

The reference is F's onset, t(1) = 48 (F and F′ both; U1 confirms per gated arm). Per arm r, with
ρ = 1.25 (C4's `RHO_MIN`, the order-only spread having been one grid step):

| label | when | F½ (r = ½) | F⅒ (r = ⅒) |
|---|---|---|---|
| step | onset in [t(1)/ρ, t(1)ρ] | [38.4, 60] → {48} | the same |
| dose | onset in [t(1)/(rρ), t(1)ρ/r] | [76.8, 120] → {80, 96} | [384, 600] → {384, 512} |
| none | no transient by the horizon H, with H past the dose band | H 1000 | H 768 |
| neither | an onset in no band | | |

The two bands are disjoint at both rates (the math check). **The joint reading:**

| F½ | F⅒ | d1 reads |
|---|---|---|
| dose | dose | a dose clock, no threshold down to a tenth of the peak |
| dose or step | none | **a threshold between a tenth and a half of the peak** (or, for F⅒, a clock slower than the dose: not told apart by 768) |
| none | none | a threshold above half the peak |
| step | step | a step clock: the transient is the first updates', at any rate |
| any other pair | | mixed: each arm's label reported, no joint reading |

**Lean d1, stated before output** [scan, arms]: F½ has the transient, F⅒ none by 768: a threshold
between a tenth and a half. From the scan, the catapult line (`lit-C4pdf.md` §1: a regime set by
the rate against the early curvature). From the arms, A0's having none after passing F's onset dose
at 0.26 of the peak, and A0′'s one-state {T3} at 640, at 0.45 of the peak. **Lean d1′** (weaker,
[none]: a guess with no source): F½'s onset in the dose band.

### d2, induction's date against the rate

**The bar rule.** The short arms have no 1000–4000 plateau, so C4's half-rise (`c4_read.date_on`)
cannot run on them. The date is the first saved state at which the top head's induction score
(the maximum over 48 heads at that state) reaches b = ½ P̄, where P̄ is the median, over A0, F, A0′
and F′, of each arm's `date_on` plateau. P̄ is placed in U1 from the four arms' head records
before any new arm trains (each is near 0.9–0.96: C4's item 9).

| label | F½ | F⅒ |
|---|---|---|
| step | date in [256, 400] → {256, 320, 384} | the same |
| dose | date in [512, 800] → {512, 640, 768} | [2560, 4000]: past the horizon, not readable |
| not formed by H | no state at b by 1000: "slower than the dose" | no state at b by 768: **"not step"** |
| neither | any other date | a date in (400, 768] |

**Lean d2** [arms]: F½'s induction in the dose band (`P-SCH4` held it nearer the summed rate on
two pairs). **Lean d3** [arms, from d2]: F⅒ not formed by 768.

**Beside, read never** (description): each arm's training loss at its induction date (F 4.13, A0
4.02, so a loss clock predicts about 4.0–4.2); the top head's layer at the horizon (F's in layer 2,
A0's in 3: `P-SCH5`); whether F⅒ forms induction at all by 768 is d3 itself.

### f, the transient after a switch (B_P, B_O)

Rule T on each branch's 17 offsets.

| B_P | B_O | f reads |
|---|---|---|
| transient | none | **the rate's**: a jump to the peak makes it on a trained state too |
| none | none | **the state's**: the peak makes it only at the early state (F's) |
| transient | transient | **the restart's**: a fresh Adam disturbs a trained state at either rate |
| none | transient | unreadable: the control misbehaves, so neither branch is read |

**Leans, stated before output, opposite ways** (`lit-C4pdf.md` "What the scan changes" 1). f-state
[arms, scan]: F's transient follows the cloud's collapse at F's first updates (er_mid 1.22 at
step 4 against A0's trough of 5.23 at 48: C4's item 7), and the catapult's threshold is the early
curvature's. f-rate [scan]: Gupta et al.'s direct-to-peak spike on a trained Pythia, and a fivefold
jump in η past Adam's edge of stability. Neither is the design's; the reading decides.

**Beside, read never:** B_P's onset in updates against F's 48; the training loss on each branch at
every update (Gupta's spike); A0's own states 256–512 (no transient, above); the top head's
induction score on each branch (A0 reaches b near 768).

## The gates (U1, before any new arm trains)

| gate | effect (must read) | null (must read) |
|---|---|---|
| G-T | F and F′, each cut at 1000 and at 768: a transient, onset 48, at every gated arm | A0 and A0′ cut the same: none |
| G-d1 | F's matrices stretched by 1/r onto the arm's grid (the state at s takes F's at the largest F state ≤ r s): "dose" at r = ½ and r = ⅒ | F unstretched: "step" |
| G-I | the bar rule on A0, F, A0′, F′ cut at 1000: each arm's frozen half-rise date (768, 320, 768, 320) | |
| G-d2 | F's top-head series stretched by 1/r: "dose" at ½; at ⅒, "not formed by 768" | F unstretched: "step" at both |
| G-f | F's states 0–256 read as offsets: a transient | A0's states 256–512 as offsets (only 256, 320, 384, 512 exist): none |
| H-d | an arm at r = 1 under the new driver equals F's saved weights bit for bit at 1, 2, 4, 8 | |
| H-f | a branch's state at k = 0 equals A0's step-256 file bit for bit; its batch k is A0's batch 256 + k (row hashes) | |

G-d1 and G-d2 are plants built from the effect arm itself: they test that the code reads a stretched
F as the dose, not that the dose clock is right. A gate that fails is fixed or argued before any new
arm trains, and its failure is recorded (`LESSONS.md` 6).

## Exit condition and budget

**Exit:** d1's joint reading (or each arm's label), d2 and d3, and f each read or refused by its
rule. C4's item 7 ("the transient") and `lit-C4.md` Parked 1 (does a lower flat peak form the
heads) each say what (d) and (f) added. **Budget: 4 units + close-out** (U0 this design; U1 the
driver, the rules, the gates and, at D3 (b), the drafted entries; U2 the four runs and their
records, run-and-watch to the `runner` agent; U3 the reads; close-out). GPU ≈ 17 h at D1 (a), D2 (a).
At the budget it closes at what it knows.

## Carried in

- **The two `c2_seeds` refusals** (C4 U1's Parked 0, carried through C4′, C4′ (c), (e)+(g)): not
  taken. No unit here derives a centred arm. Proposed **dropped at this thread's close-out** unless
  a unit by then derives one: they have been carried through five designs and none needed them.
- **Keep every registered hash:** new code goes in `trackC_writeup/c4pdf_train.py` (the driver: a
  rate factor, a branch start), `c4pdf_rules.py` (the rules above, pure; its hash is the one a D3 (b)
  entry registers) and `c4pdf_read.py` (the gates' drivers and the reads). `c4_train`, `c4_read`,
  `c4_close`, `c4pc_rules` stay as they are.
- `lit-C4pdf.md` Parked 1 (Gupta et al.'s full text): read before U2's records are read.

## Decisions (each yours; recommendation first)

| | question | (a), recommended | (b) | (c) |
|---|---|---|---|---|
| D1 | the (d) horizons | F½ to 1000, F⅒ to 768 (13.0 h): each one grid step past its own dose band, so "none" means past the band | the ROADMAP's 768 and 384 (8.5 h): F⅒ cannot reach its transient's dose band, and F½ ends inside induction's | |
| D2 | (f)'s optimiser state | the ROADMAP's: two branches from the saved weights with fresh moments (B_P, B_O; 3.8 h). B_P against F changes only the state, since F also began with a fresh Adam at the peak | also rebuild A0's moments at 256 (re-train A0 0 → 256, 1.9 h, checked bit-equal against the saved file) and add B_K, the peak with moments kept (+3.8 h): the switch as practised | B_K alone with A0 as its control (3.8 h), if the rebuild is bit-equal; otherwise also a kept-moment branch at A0's rate |
| D3 | registration | none: tier 1, description with the rules fixed here | you register d1 and d2 as two entries (the next `P-SCH` numbers, 7 and 8) after U1's gates pass and before either (d) arm's first update; (f) stays tier 1 either way | |
| D4 | the order | (f)'s branches first (3.8 h, tier 1), then F⅒, then F½: nothing waits on D3 while the branches run | (d) first | |

**Recommended: D1 (a), D2 (a), D3 (b), D4 (a).** D3 (b) because these four runs are unseen, the
leans are stated above with their sources, and d1 and d2 can each lose: they are the only
confirmatory readings left in the C4 line. It costs you a registry commit and holds the (d) arms,
not (f), until you make it. Against it: one run per rate, 70m only, and bands that rest on F's two
dates at one grid.

## Worth challenging (for the PR)

1. **T's "≥ 2 consecutive states" was placed after seeing A0′.** It makes the harness replicate's
   one-state {T3} read "none". The alternative, a size bar (the count outside {T1, T2} at least half
   F's peak), takes its scale from the effect arm itself. On a branch's dense offsets two states can
   be one update apart (k = 1, 2), so a run there is shorter in updates than on an arm's grid at 640.
2. **The centred arms are not re-gated on the new runs.** C4 gated them on F at 3500; the new runs
   stop at ≤ 1000. The tolerances are the instrument's, but whether each still reads a planted
   spread on F½'s states is assumed, not checked.
3. **"Threshold" on F⅒ is "none by 768"**, which a clock slower than the dose (1/r², say) also
   gives. The design does not separate them and says so in d1's table.
4. **One run per rate.** ρ = 1.25 rests on C4′ (c)'s order-only spread (one grid step), from one
   reordering at r = 1.
5. **The bar rule replaces the half-rise for induction** on the short arms. G-I checks that it
   returns the frozen dates on the four trained arms; it cannot check it on arms whose plateau is
   unknown (F⅒'s heads may plateau lower, and then b is too high and the date late).
6. **(f)'s fresh moments.** At D2 (a) neither branch is a schedule switch as practised; B_P against
   F isolates the state, B_O the restart at a low rate. Rejected as the primary: D2 (c), since a
   kept-moment branch compares with nothing trained so far except A0 itself, and needs a 1.9 h
   rebuild whose bit-equality is not known.
