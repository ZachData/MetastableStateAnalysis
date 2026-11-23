<!-- trackC_writeup/design-C4pdf.md -->
# Track C — design: C4′ (d) and (f), the transient's dose and a schedule switch

**Status: proposed 2026-11-23 (U0, writing only); revised after `/challenge-pr` on #371 (accept
with changes, findings 1–5 taken: the reference onset per gated arm, interval bands on a dense
grid with a refusal when they meet, (f)'s labels, rule T in updates, two slips). Decisions D1–D4
are STATE Blocked 172.**
Opened on Blocked 169, decided by the user 2026-11-23: "(e) and (g) first", then (d), (f). (e)+(g)
closed in #370. (d) and (f) both train short new runs from states C4 already holds, and both ask
what sets F's transient, so they share one design and one harness. Scan: `lit-C4pdf.md`.
Arithmetic: `tools/math_checks/c4pdf_bands.py`. **Nothing below has been computed or trained.**
Every number quoted from the four trained arms (A0, F, A0′, F′) was read by earlier units, or by
the reviewer of #371 from their records, and is cited to them.

## What it is for

C4 found that F (flat at the peak from step 0) rises through a **transient** that A0 (Pythia's
ramp) lacks. From step 48, blocks 1–5 lower the centred energy on most prompts: at τ 1e-4, T3 is
majority violated from 48 to 512 and T4, T5 from 48–64 to 256. Then both arms settle on the same
end set {T1, T2} (`status-C.md` "C4 on one page", item 7). F′ repeated it, its run lasting past
1000; A0′ showed {T3} at one grid step (640) only ("C4′ (c) on one page", item 5). Nothing so far
says **what sets the transient**: the rate's size, its dose (the summed rate), or the early state
that a flat rate meets at its first updates. Induction's date has the same open question in
another form: `P-SCH4` held it nearer the summed rate than the counts, and the rate and loss clocks
were not told apart (C4′ (a)). The ROADMAP rows (§8 C4′):

- **(d) the transient's dose** (`lit-C4.md` Parked 1, Qu & Liu's peak dependence): flat arms at a
  fraction r of the peak. Is the transient a threshold in the rate, and does induction's date
  scale as 1/r?
- **(f) a schedule switch from a saved state:** A0's step-256 weights continued at the peak, against
  the same restart at A0's own rate. Is the transient the early state's or the rate's?

**Found while writing (corrects the ROADMAP row's horizons).** The row costs (d) as F½ to 768 and
F⅒ to 384. Under the dose clock F⅒'s transient begins in [256, 800] (the bands below, over the
three gated arms) and its induction past 2000. So F⅒ stopped at 384 cannot see the transient where
the dose clock puts it. F½ stopped at 768 ends inside induction's dose band (to 800). D1 (a)
therefore runs both to 1000.

**Found at review (#371, finding 2): F's onset is not known finely enough to tell a half-rate dose
from a step.** F's grid holds 32, 48 and 64, so its onset is in (32, 48] or (48, 64]. With that
interval and the spread ρ₀ = 1.25, the step and dose bands for F½ meet (they are disjoint only for
r < 0.43 at an onset of 48). So (d) needs F's onset to a few updates: **F¹, F re-run 0 → 128 with a
state saved every 4 updates** (≈ 1 h), checked bit-equal against F's saved states (H-d).

**Also found.** A0 already argues against a pure dose clock and against the rate alone. Its summed
rate reaches F's onset dose (48 peak-steps) at step 370, at 0.26 of the peak, with no transient
(item 7's table: 320 "1 0 1 0 0 0", 512 "0 0 3 1 1 0"). And A0 reaches the peak itself at 1430,
trained, with none through 4000 (#371, finding 3). So the rate alone, on a trained state, does not
make it. Either it needs the early state, or a fresh Adam at a high rate (as F had at step 0, and as
a branch has), or both. (d) varies the rate at the init's state; (f) varies the state with a fresh
Adam at the peak. Together they separate the early state from the rest.

## The arms (new runs; C4's recipe with one thing changed)

Every arm is `pythia-70m`'s architecture under C4's recipe (`c4_train.build`, `make_opt`,
`step_once`, `save_ckpt`, called unchanged, so `c4_train`'s hash `1c1e1d2df083` stays): AdamW
(0.9, 0.95), ε 1e-8, weight decay 0.1, clip 1.0, bf16 autocast, batch 1024 rows of 2049 tokens, on
**C4's data order** (`c4_train.batch_of(rows, i)`, so iteration i reads A0's and F's i-th batch).

**The dense grid D:** every 4 updates to 128, every 16 to 256, every 32 to 992, and 1000 (65
states). Saves hold weights only (282 MB each), so density costs disk, not GPU.

| arm | from | rate | updates | saved states | changes against C4 |
|---|---|---|---|---|---|
| F¹ | seed 0's init (`pythia-70m` step0, as A0, F) | 1e-3, flat (F's) | 0 → 128 | D ≤ 128 (33) | nothing: F on a finer grid, the reference |
| F½ | seed 0's init | 0.5 × 1e-3, flat | 0 → 1000 | D (65) | the rate only |
| F⅒ | seed 0's init | 0.1 × 1e-3, flat | 0 → 1000 | D (65) | the rate only |
| B_P | A0's saved step-256 weights (fp32), **fresh AdamW** (zero moments, counter 0) | 1e-3, flat | 256 → 512 | 256 + k, k ∈ K (17) | the state (against F: same rate, same fresh Adam) |
| B_O | the same weights, fresh AdamW | A0's own, `lr_pythia(256 + k)` (0.179 → 0.358 of the peak) | 256 → 512 | 256 + k, k ∈ K | the control for a low-rate restart |

K = (0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256): C4's early grid as
offsets. The branches read A0's batches 256–511, the ones A0 itself took next. **Checkpoints hold
weights only** (`HDD_1TB/mets_data/c4/A0/ckpt/step256/model.safetensors`; `resume.pt` is A0's at
4000), so A0's moments at 256 do not exist: D2 (b) costs rebuilding them.

GPU, at C4's measured ≈ 26.5 s an update (C4′ (c): 8000 updates in ≈ 59 h): F¹ 0.9 h, F½ and F⅒
7.4 h each, each branch 1.9 h; **≈ 19.4 h in all** at D1 (a), D2 (a). Records (c4_read's
per-checkpoint e1 / e1c on v1's 7 prompts, float32 eager) and heads (`c4_close heads`) on the 197
new states: about an hour. Disk ≈ 56 GB on `HDD_1TB` (123 GB free 2026-11-23), under
`mets_data/c4pdf/{F1,Fh,Ft,BP,BO}/`.

## The readings (rules fixed here; constants placed in U1 before any new arm trains)

The instruments are C4's, unchanged: `c4_close.per_transition` (prompts violating at each
transition T0–T5, by `c4_read.series`' rule) at **C4's three gated centred arms** (τ 9.61e-5,
1e-4, 3e-4), and the induction score per head (`c4_close heads`). Every reading is made **per
gated arm, with that arm's own reference**, and holds only where it holds at every gated arm, as
L7's did. **No new centred arm is derived**, so the arms' τ are not re-gated on the new runs (none
reaches C4's gate step 3500); "Worth challenging" 2.

### T, the transient (both (d) and (f))

On a series of saved states, a **transient** is a maximal run of consecutive states at each of
which some transition outside the end set {T1, T2} is majority violated (> 3.5 of 7 prompts), and
whose last state is **at least 16 updates after its first** (#371, finding 4: counted in updates,
not states, so the rule means the same on F's grid, on D and on a branch's offsets). Its **onset**
is the run's first state.

Placed so that the four trained arms read as C4 and C4′ (c) described them: F and F′ have one
(from 48 at 9.61e-5 and 1e-4, from 64 at 3e-4: #371, finding 1); A0 has none; A0′'s {T3} at 640
is one state, so none. **This placement was made after seeing A0′** (a one-state blip is
excluded by construction); the gate checks it on all four, and the new runs are unseen. The size
of the transient (the count outside {T1, T2}, summed over the run) is reported beside, never read.

### d1, the transient against the rate (F½, F⅒)

The reference, per gated arm, is F¹'s onset t with the saved state before it t⁻ (on D, t⁻ = t − 4),
so F's onset lies in (t⁻, t]. The spread ρ₀ = 1.25 is C4's `RHO_MIN` (placed there as the grid's,
not measured; C4′ (c)'s order-only spread was one grid step). Per arm r:

| label | the arm's onset in |
|---|---|
| step | [t⁻/ρ₀, t ρ₀] |
| dose | [t⁻/(r ρ₀), t ρ₀/r] |
| none | no transient by 1000, and 1000 past the dose band's top |
| neither | an onset in no band |
| refused | the two bands meet (they are disjoint iff r < t⁻/(t ρ₀²)): no label at that arm |

At F¹'s grid (t⁻ = t − 4) the bands are disjoint at both rates for any onset from 48 to 64 (r < 0.59
at 48). On C4's grid alone they would meet at r = ½ (r < 0.43 at 48), which is why F¹ runs. For
example, at an onset of 48: step [35.2, 60], F½ dose [70.4, 120], F⅒ dose [352, 600]; at 64: step
[48, 80], F½ [96, 160], F⅒ [480, 800]. **The joint reading:**

| F½ | F⅒ | d1 reads |
|---|---|---|
| dose | dose | a dose clock, no threshold down to a tenth of the peak |
| dose | none | **a threshold between a tenth and a half of the peak** (or, for F⅒, a clock slower than the dose: not told apart by 1000) |
| step | none | the first updates' at a half, nothing at a tenth: a threshold between them, on a step clock above it (#371's question 7) |
| none | none | a threshold above half the peak |
| step | step | a step clock: the transient is the first updates', at any rate |
| any other pair, or a refusal | | mixed: each arm's label reported, no joint reading |

**Lean d1, stated before output** [scan, arms]: F½ has the transient, F⅒ none by 1000: a threshold
between a tenth and a half. From the scan, the catapult line (`lit-C4pdf.md` §1: a regime set by
the rate against the early curvature). From the arms, A0's having none after passing F's onset dose
at 0.26 of the peak, and A0′'s one-state {T3} at 640, at 0.45 of the peak. **Lean d1′** (weaker,
[none]: a guess with no source): F½'s onset in the dose band.

### d2, induction's date against the rate

**The bar rule.** The short arms have no 1000–4000 plateau, so C4's half-rise (`c4_read.date_on`)
cannot run on them. The date is the first saved state at which the top head's induction score
(the maximum over 48 heads at that state) reaches b = ½ P̄, where P̄ is the median, over A0, F, A0′
and F′, of each arm's `date_on` plateau. P̄ is placed in U1 from the four arms' head records
before any new arm trains (the reviewer of #371 found the rule returns 768, 320, 768, 320 on them).

The reference is F's date, 320, with the state before it 256 (F¹ stops at 128, so F's own grid):
F's date lies in (256, 320], and the bands are the d1 formulas with t⁻ = 256, t = 320.

| label | F½ (r = ½) | F⅒ (r = ⅒) |
|---|---|---|
| step | [204.8, 400] | the same |
| dose | [409.6, 800] (disjoint from step: r < 0.512, by a thin margin) | [2048, 4000]: past 1000, not readable |
| not formed by 1000 | "slower than the dose" | **"not step"** |
| neither | any other date | a date in (400, 1000] |

**Lean d2** [arms]: F½'s induction in the dose band (`P-SCH4` held it nearer the summed rate on
two pairs). **Lean d3** [arms, from d2]: F⅒ not formed by 1000.

**Beside, read never** (description): each arm's training loss at its induction date (F 4.13, A0
4.02, so a loss clock predicts about 4.0–4.2); the top head's layer at 1000 (F's in layer 2, A0's
in 3: `P-SCH5`).

### f, the transient after a fresh start on a trained state (B_P, B_O)

Rule T on each branch's 17 offsets. A0 reaches the peak at 1430 with no transient, so **the rate on
a trained state is already excluded**; what B_P adds to that is a fresh Adam at the peak, as F had
at step 0. B_P against F changes only the state; B_O controls a restart at a low rate. In B_P the
restart's first updates are near the rate itself per coordinate, so the jump and the restart are
not separated (#371, finding 3; D2 (b) would separate them).

| B_P | B_O | f reads |
|---|---|---|
| transient | none | **not the early state's**: a fresh Adam at the peak makes it on a trained state too (the jump and the restart not told apart) |
| none | none | **the early state's**: a fresh Adam at the peak makes it only at F's state |
| transient | transient | **the restart's**: a fresh Adam disturbs a trained state even at a low rate |
| none | transient | unreadable: the control misbehaves, so neither branch is read |

**Leans, stated before output, opposite ways** (`lit-C4pdf.md` "What the scan changes" 1). f-state
[arms, scan]: F's transient follows the cloud's collapse at F's first updates (er_mid 1.22 at
step 4 against A0's trough of 5.23 at 48: C4's item 7), and the catapult's threshold is the early
curvature's. f-jump [scan]: Gupta et al.'s direct-to-peak spike on a trained Pythia, and a
fivefold jump in η past Adam's edge of stability. Neither is the design's; the reading decides.

**Beside, read never:** B_P's onset in updates against F¹'s; the training loss on each branch at
every update (Gupta's spike); A0's own states 256–512 (no transient, above); the top head's
induction score on each branch (A0 reaches b near 768).

## The gates (U1, before any new arm trains, except H-d, which F¹ itself answers)

| gate | effect (must read) | null (must read) |
|---|---|---|
| G-T | F and F′, each cut at 1000: a transient, onset 48, 48, 64 at the three gated arms | A0 and A0′ cut the same: none |
| G-d1 | F's matrices stretched by 1/r onto D (the state at s takes F's at the largest F state ≤ r s), with F's coarse reference: "dose" at r = ⅒; at r = ½, "refused" (the bands meet on F's grid, the reason F¹ runs) | F unstretched: "step" at ⅒ |
| G-I | the bar rule on A0, F, A0′, F′ cut at 1000: 768, 320, 768, 320 | |
| G-d2 | F's top-head series stretched by 1/r: "dose" at ½; at ⅒, "not formed by 1000" | F unstretched: "step" at both |
| G-f | F's states 0–256 read as offsets: a transient | A0's states 256–512 as offsets (only 256, 320, 384, 512 exist): none |
| H-d | F¹ equals F's saved weights bit for bit at every F state ≤ 128 (1, 2, 4, …, 128); then G-d1 is re-run with F¹'s reference: "dose" at ½ and ⅒ | |
| H-f | a branch's state at k = 0 equals A0's step-256 file bit for bit; its batch k is A0's batch 256 + k (row hashes) | |

G-d1 and G-d2 are plants built from the effect arm itself: they test that the code reads a stretched
F as the dose, not that the dose clock is right. If H-d fails (F¹ not bit-equal to F), F¹ is a
replicate of F and its onset the reference anyway, recorded so. A gate that fails is fixed or argued
before any new arm trains, and its failure is recorded (`LESSONS.md` 6).

## Exit condition and budget

**Exit:** d1's joint reading (or each arm's label), d2 and d3, and f each read or refused by its
rule. C4's item 7 ("the transient") and `lit-C4.md` Parked 1 (does a lower flat peak form the
heads) each say what (d) and (f) added. **Budget: 4 units + close-out** (U0 this design; U1 the
driver, the rules, the gates, F¹ and, at D3 (b), the drafted entries; U2 the other four runs and
their records, run-and-watch to the `runner` agent; U3 the reads; close-out). GPU ≈ 19.4 h at
D1 (a), D2 (a). At the budget it closes at what it knows.

## Carried in

- **The two `c2_seeds` refusals** (C4 U1's Parked 0, carried through C4′, C4′ (c), (e)+(g)): not
  taken. No unit here derives a centred arm. Proposed **dropped at this thread's close-out** unless
  a unit by then derives one: they have been carried through five designs and none needed them.
- **Keep every registered hash:** new code goes in `trackC_writeup/c4pdf_train.py` (the driver: a
  rate factor, a branch start, grid D), `c4pdf_rules.py` (the rules above, pure; its hash is the
  one a D3 (b) entry registers) and `c4pdf_read.py` (the gates' drivers and the reads). `c4_train`,
  `c4_read`, `c4_close`, `c4pc_rules` stay as they are.
- `lit-C4pdf.md` Parked 1 (Gupta et al.'s full text): read before U2's records are read.

## Decisions (each yours; recommendation first)

| | question | (a), recommended | (b) | (c) |
|---|---|---|---|---|
| D1 | the (d) arms | F¹ (0.9 h), then F½ and F⅒ both to 1000 on grid D (14.7 h): each past its own transient dose band, and F½ past induction's | the ROADMAP's: F½ to 768, F⅒ to 384 on C4's grid (8.5 h): F⅒ cannot reach its transient's dose band, F½ ends inside induction's, and at r = ½ the transient's bands meet | |
| D2 | (f)'s optimiser state | the ROADMAP's: two branches from the saved weights with fresh moments (B_P, B_O; 3.8 h). B_P against F changes only the state | also rebuild A0's moments at 256 (re-train A0 0 → 256, 1.9 h, checked bit-equal against the saved file) and add B_K, the peak with moments kept (+3.8 h): separates the jump from the restart | B_K alone with A0 as its control (3.8 h), if the rebuild is bit-equal; otherwise also a kept-moment branch at A0's rate |
| D3 | registration | none: tier 1, description with the rules fixed here | you register d1 and d2 as two entries (the next two `P-SCH` numbers) after U1's gates pass and before F½'s or F⅒'s first update; (f) and F¹ stay tier 1 | |
| D4 | the order | F¹ and (f)'s branches first (4.7 h, tier 1), then F⅒, then F½: nothing waits on D3 while they run | (d) first | |

**Recommended: D1 (a), D2 (a), D3 (b), D4 (a).** D3 (b) because these runs are unseen, the leans
are stated above with their sources, and d1 and d2 can each lose: they are the only confirmatory
readings left in the C4 line. It costs you a registry commit and holds F½ and F⅒, not F¹ or (f),
until you make it. Against it: one run per rate, 70m only, ρ₀ placed and not measured, and d2's
bands disjoint by a thin margin (r < 0.512 at r = ½). D2 (a) stays recommended after #371's
finding 3 because it answers the question asked (the early state or not), and says plainly that
it does not separate the jump from the restart; D2 (b) does, for 3.8 h more.

## Worth challenging (for the PR)

1. **Rule T's 16-update span was placed after seeing A0′'s one-state {T3} at 640.** The rejected
   alternative is a size bar, the count outside {T1, T2} at least half F's peak, which takes its
   scale from the effect arm itself. A real transient shorter than 16 updates would read "none".
2. **C4's centred τ arms are not re-gated on the short runs**, none of which reaches 3500. The
   tolerances are the instrument's, but whether each still reads a planted spread on F½'s states
   is assumed, not checked.
3. **"None by 1000" on F⅒** does not separate a threshold from a clock slower than the dose
   (1/r², say).
4. **One run per rate, and ρ₀ = 1.25 is placed.** C4 placed it as the grid's; C4′ (c) measured the
   order-only spread as one grid step, from one reordering at r = 1.
5. **The bar rule replaces the half-rise for induction** on the short arms. G-I checks it only on
   arms with a known plateau. If F⅒ plateaus lower, b is too high and the date reads late.
6. **d2 at r = ½ is disjoint by a thin margin** (0.5 against 0.512), because F's induction date is
   known only to (256, 320]. Rejected: extending F¹ to 400 for a finer induction reference
   (+2 h). The thin margin is accepted and stated.
7. **(f) at D2 (a) does not separate the jump from the restart.** B_P against F isolates the state.
   B_O controls only a low-rate restart.
