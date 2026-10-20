<!-- Proposed 2026-10-19; revised after /challenge-pr on #232 (findings 1-5, 7 taken); frozen as revised 2026-10-19 (STATE Blocked 70 (ii) (a), the user's; 70 (i): no seed download before A5 registers, confirmed). -->
# Track A — design: A3, one timeline

**Question** (`docs/ROADMAP.md` §6 A3): on pythia-410m's checkpoint axis, do the energy break,
the OV repulsive phase, induction formation, the n-gram milestones and the weights-only rows
change at the same steps, and in what order? This is the core of C2's sequence (§8).
**Opened 2026-10-19** by §9 row 6 (Blocked 68). #231's finding 4 (gate first or A3 first): A3
first, as already ordered; the user took "A3 first, start the seed download now", and Claude
dropped the download half because `design-A5.md` registers before the seeds are downloaded
(its step 3), which is to be confirmed by the user (Blocked 70). Scan: `lit-A3.md` (C2's
`lit-C2.md` not repeated). Tier 1, 410m, unregistered.

## Exit condition and budget

- **Exit:** one table and one figure. Every family row (§2) has a located change step, or
  "no change", or "refused" (its gate arm failed). Every read pair (§3) is stated as
  **aligned**, **misaligned** or **unread**, and the located steps are given as an order (ties
  at grid resolution), with the warmup named on each pair at or below 2000. Or a recorded
  reason the table cannot be made (e.g. the locator's gate fails).
- **Budget: 5 units** (6 on the record since Blocked 72 (i) (a), §4 "Revised 2026-10-19 (Blocked 72)"), each its own PR, plus a close-out (writing only):
  (1) this design and the scan; (2) the producer, the existing rows, the locator and its gate;
  (3) the weights-only rows and the simulator row; (4) the n-gram rows (M2);
  (5) the reading: table, figure, pairs, a page. At the budget A3 closes at what it knows.

Every threshold marked *placed* is set by judgement before any output; *calibrated* ones come
from `tools/math_checks/a3_locator.py` (synthetic, seeded), before any output.

## 1. The axis

The 19-step grid (0, 1, 2, 4, …, 512, 1000, 2000, 4000, …, 143000). Steps 0 and 1 are the same
weights (`p1b_hemisphere/status-1b.md`, §3.51), so there are **18 distinct points**, and the
locator reads 18. All 19 grid checkpoints and 14 more (3000, 5000, 7000, 9000, 11000, 13000,
15000, 17000, 19000, 40000, 60000, 80000, 100000, 120000) are on disk (`data/hf`): 32 distinct
steps. The weights-only rows are computed on all 32 and drawn beside the grid, but they are
**located on the 18**, so that every row has the same resolution. Nothing exists between 512
and 1000 or between 1000 and 2000.

## 2. The family (fixed before any output)

Each row has a **declared direction** (the sign of the move it is located by) and a unit set its
σ is bootstrapped over.

| id | row | dir. | σ over | source | cost |
|---|---|---|---|---|---|
| E1 | energy violations, Σ nViolRel at β = 1, 7 prompts | up | — | `colocation_panel.csv` `p1_energy_nviol_b1`; **located by `design-A5.md` D2 (a) / D4 (a) exactly**, not by §4 | free |
| E2 | OV: median `attractive_energy_fraction_core` over 384 heads | down | heads | `data/analysis/ov_per_head_series.json` (§3.12 A) | free |
| E3 | induction: mean pooled induction attention of the five typed members (`L5H2`'s partners `L8H6`, `L7H8`, `L11H14`, `L12H5`, `L8H9`) | up | prompts × members | `data/analysis/behavioural_series.json` (P-I1) | free |
| E4 | previous-token score of `L5H2` | up | prompts | B1's records if they cover the 18 steps; else one hooked pass per step on the 7 prompts | free / GPU min |
| E5 | effective rank on the sphere, mean over 7 prompts | down | prompts | `colocation_panel.csv` `p1_eff_rank_normed_mean` | free |
| E6u, E6b | KL of the model's next-token distribution to a unigram (u) and a bigram (b) predictor | **turning point** (§4) | tokens (blocks) | new, unit 4 (corpus in §5) | GPU, ~1 h |
| E7 | weight change, `‖W_t − W_0‖_F / ‖W_0‖_F` per matrix type (QKV, O, MLP in, MLP out), median over 24 layers; one row per type, located on the median over types | up | layers | new, unit 3 | GPU min |
| E8 | ESD tail exponent α per matrix type, median over layers (HTSR) | down | layers | new, unit 3 | GPU min |
| E11 | 1e's one-well question: the causal simulator (`sa_sim.py`) from the measured start at each step's per-prompt β, final cluster count | down | prompts | unit 3 (A1 at close) | GPU min |
| E10 | the LR schedule (linear warmup to about step 1430, then cosine) | up | — | Pythia's config, analytic; **the confound row**, located by §4 with no floor: it reads 1000 (half of 1430 is 715) | free |

So **ten located rows** (E1–E8 with E6 as two, E11) and E10. **Beside, not in the family:**
ΔW's stable rank (undefined at step 0, where ΔW = 0; drawn from step 2), and D1's `W_V` half
(the share of heads with a negative-definite core: a negative-definite core forces E2's
quantity to exactly 0 for that head, so it is not independent of E2, and it is 0 at random
init, so it has no floor).

**Quoted rows** (drawn on the figure as bands, never located by the rule and never in a pair):
Belrose's n-gram-sequence loss trough at 64–256 and rise until about 1000 (2402.04362, Fig. 6,
Pythia); RankMe's collapse at 32–256 and recovery by about 1000 (2602.15997, 410m); the
induction phase change 1200–2400 (2209.11895); R6f's embedding move 512 → 2000 (Phase 10); U2's
regime change at 2000–4000 (1e); step 16's collapse at state 23 in 3 of 7 prompts (A1); A2's
against-specific steps (16, 32, 1000–16000, 143000), the MLP cancelling attention at
1000–4000, the fixed key from a period (2000–4000) to a newline (from 8000), and the sink's
per-token share rising from 1000.

## 3. The pairs that are read

Named before output, **eleven pairs**: **E1 against each of E2–E8 and E11** (nine, with E6u and
E6b as two), and the two that carry C2's order inside the circuit: **E4 against E3**
(previous-token before induction) and **E3 against E2** (induction against the OV phase).
Other pairs are drawn, not read. **Aligned** (*placed*): the located steps are equal or one
grid step apart. **Misaligned**: two or more grid steps apart, and the order is stated.
**Unread**: either row is "no change" or "refused". One grid step is a factor of about 2, so
"aligned" at 512 means anything in 256–1000; the text says so on every aligned pair.

**What "aligned" can and cannot carry, stated before output.** (i) E10 reads 1000 under the
rule, so every row located at 512–2000 is aligned with the warmup by construction; a pair at or
below 2000 carries "inside warmup (E10)", and A3 cannot separate a shared cause from the
schedule (`lit-C2.md` §3). (ii) E1 against E2 is already known: §3.12 A's published medians put
E2 at 512 under this rule (0.504 → 0.094 by 512), and E1 is at 512 (D4 (a)). That pair is a
reproduction on one table, not a finding. (iii) The new information is the pairs whose rows
have not been on this axis before (E4, E6, E7, E8, E11) and the order of E4, E3, E2.

## 4. The locator, and its gate (unit 2)

**Rule.** For a row x over the 18 steps, let Δ(s) = x(s) − x(0), signed in the row's declared
direction. The reference is the largest such Δ over steps ≤ 4000 (the same plateau end as D4
(a)); the change step is the first step where Δ ≥ half the reference. **Turning-point rows**
(E6u, E6b): Belrose et al. find the unigram and bigram losses fall, then rise to a lesser
extent, then fall (`lit-A3.md`), so these are located at the step of the minimum over ≤ 4000,
and the floor applies to the fall into it. **No change** if the reference is below **4σ**
(*calibrated*: 3.93σ gives "no change" on 95 % of flat series with iid noise of size σ at
every step, step 0 included; 3σ gives 76 %). σ is the bootstrap standard error (1000
resamples, over the row's units in §2) of Δ at the reference step: the noise in the move
itself, for activation and weights-only rows alike. For weights-only rows the floor is the
same number but not calibrated for their noise, which is over heads or layers, not over draws
(*placed*). Unit 2 re-runs the calibration on each row's own σ at every step (re-derived
per row as floor and power: "Revised 2026-10-19" below).

**Power** (*calibrated*, the same script): with the floor at 3.93σ and a logistic step in
log2(step) of scale 0.25 octave (*placed*) centred mid-gap, the rule places the step just past
the centre in ≥ 90 % of draws once the amplitude reaches 6σ (gap 2–4) to 9σ (gap 2000–4000);
the need rises with the gap's position because more steps precede the move. A row located
with a reference under 9σ is marked **low power** on the table.

**Gate, before any real row is located** (`CLAUDE.md` "Calibration gate"), on the 18 steps:
- *planted null:* 1000 flat series with iid noise at each row's own σ: "no change" in ≥ 95 %.
- *planted effect:* the logistic above at 9σ, centred in each gap up to 2000–4000, 1000 each:
  placed right in ≥ 90 %. A turning-point row gets a planted dip (down, then up by half) at
  the same amplitude.
- *the real anchors:* E1 must come out at 512 (D4 (a) on the same file); E2 at 512 (from
  §3.12 A's table); E10 at 1000.
A failed arm refuses its rows; it is not tuned after the fact. Beside, described and not read:
`core.changepoint_colocation.change_profile`'s change-mass centroid per row, flagged where it
and the rule are two or more grid steps apart.

**Revised 2026-10-19 (Blocked 71, the user's; unit 3).** Unit 2's gate failed its effect arm
on E2–E5 (`status-A.md` "A3 unit 2"). (i) **(a)**: the effect arm's 9σ is σ *at the reference
step* (the paragraph above's definition, read literally), taken as the largest mean null σ over
the planted plateau's steps; the arm then vouches only for moves large against the late noise.
**Line 102 re-derived per row**: each floored row's floor is the 95 % upper confidence bound of the
95th percentile of ref / σ over 1000 null series at its own per-step spread (added in unit 2c,
after the bare percentile failed its own check: `status-A.md` "A3 unit 2c"), checked on 1000
fresh ones (≥ 95 %), replacing 4σ for that row; its power is, per gap, the least amplitude (grid
2 … 90, units of that σ) placed right in ≥ 90 %, and a located row whose ref, in that unit, is
below its gap's power is **low power**,
replacing the flat 9σ. (ii) A located row whose reference is at 4000 (still rising at the
window's end) is marked **censored**, and its pairs are read with that mark (unit 5).

**Revised 2026-10-19 (Blocked 72, the user's; unit 3).** (i) **(a)**: the budget is **6 units**
on the record (the re-gate took one); unit 3 is the weights-only rows, the simulator row and E3,
unit 4 the n-gram rows, unit 5 the reading, unit 6 spare. (ii) **(a)**: the frozen E3 (five
members at offset 0) is **refused as defined**, because the 2026-10-18 correction that predates
the freeze says it scores the wrong key. **E3 is re-defined** as the pooled attention of `L8H6` and
`L11H14` (the two that correction names as induction heads at +1) at the standard pairs
(`induction_candidates(strict=True)`: the key follows an earlier copy of the query's token),
averaged over the two, with σ over the 14 (prompt, member) units, and it is gated before it is read.
The offset-0 series is drawn beside, not read. (iii) **(a)**: from unit 3 on, every floored row's
floor is **max(4, its own bound)**; the null arm and the power are taken at that floor.

**Revised 2026-10-20 (Blocked 73, the user's, all (a); for unit 5).** (i) E11 stays refused (a
constant row: Theorem 4.1 of 2411.04990 makes the final count 1 for V = Id); its trajectory is
drawn as description and nothing is carried. (ii) A **censored** row's pairs are **unread**
(replacing (ii) of Blocked 71's revision, "read with that mark"), decided before any pair is seen.
(iii) Unit 5 states both of E8's dates (1000 against step 0, 512 from its peak at 256) and reads
a pair with E8 only where both give the same verdict.

**Placed before any unit 3 output (2026-10-19)** (§2's rows leave these open):
- *E7, E8:* located on the median over the four types of the per-type median over 24 layers, with
  σ over layers (the same layers resampled for every type); per type drawn beside. E8's α is the
  Clauset–Shalizi–Newman continuous power-law MLE on the eigenvalues of W Wᵀ, λ_min by least KS
  distance over tails of at least 10 (as WeightWatcher); the weight, not the bias.
- *E11:* the start is A1a's (`f_0(x_0)`, position 0 out, rebuilt from the stored activations and
  checked against A1a's `g0`), β the prompt's bracket p50 (median over blocks 0–22 of the layer
  β), the causal dynamics at dt 0.01 (checked by halving on one cell), and the count is read at
  t = 200 (A1a's `T_MAX`). A cluster is a connected component of "cos ≥ 0.99". Counts at t = 1, 2,
  5, 10, 20, 50, 100 are drawn beside.
- D1's `W_V` half (beside in §2) is not built in unit 3; ΔW's stable rank is.

**Placed before any unit 4 output (2026-10-20)** (§2's E6 and §5's corpus leave these open;
Blocked 73, the user's, all (a), changes none of them):
- *Corpus:* the Pile's validation split, `the-eye.eu`'s `val.jsonl.zst` as copied to
  `mit-han-lab/pile-val-backup` (sha256 `264c875d8bbd…`, its LFS oid; all 22 subsets; held out of
  Pythia's training). Documents in file order, each tokenized with 410m's tokenizer and followed
  by `<|endoftext|>` (as Pythia trains). A document sharing any 13-token run with one of the 7 v1
  prompts is dropped (count recorded).
- *Scored set:* **64 blocks of 1024 tokens**, each its own forward pass from an empty context.
  A block starts at a document's start and takes that document and its successors until it
  holds 1024 tokens; the rest of its last document is discarded and the next block starts at the
  next document. So no document is in two blocks (the blocks are the bootstrap's units) and a long
  document does not fill many. *Table set:* the documents after the scored set's, in order,
  until **2,000,000** tokens.
- *Predictors:* unigram q_u, the table set's token frequencies (MLE; unseen tokens 0). Bigram
  q_b(·|v), Witten–Bell interpolated with q_u: (c(v, ·) + T(v) q_u) / (c(v) + T(v)), T(v) the
  number of distinct successors of v; q_u where c(v) = 0. Parameter-free.
- *E6u, E6b:* KL(q ‖ p_model) in nats at every position of a block (q given that position's
  token, against the model's next-token distribution), mean over the block's 1024 positions; the
  row is the mean over the 64 blocks, σ over blocks. This direction because the model's softmax
  has full support, so the KL is finite where the MLE n-gram has zeros; the other direction would
  be set by the smoothing. Model float32 on the GPU, logits over 50,304 (q is 0 on the padding).
  **Beside, not located:** the model's next-token loss on the scored set at each step, and the two
  predictors' own (constants).
- *Located* by §4's turning-point rule (the minimum over ≤ 4000; the floor on x(0) − x(min)); a
  minimum at 4000 is **censored**, as for half-rise rows.
- *Gate (§4's dip):* the null arm and the own floor as for every row, ref / σ taken by the
  turning-point rule. The effect arm, per gap (i, i + 1) from 2–4 to 2000–4000: a logistic fall of
  the amplitude centred in that gap and a rise of half of it centred in the next gap, both of
  scale 0.25 octave, so the planted minimum is step i + 1, and placed right is located there.
  Amplitude 9 × σ̄_ref, σ̄_ref as for half-rise rows; the power on the same grid.
- *Checks before reading* (refuse on failure): at 143000 the model's loss on the scored set is
  1.5–4 nats; at step 0, E6u is within 10 % of KL(q_u ‖ uniform over 50,304) (an untrained softmax
  is near flat); no document is in both sets.

## 5. Fences

- **No registration and no adjudication.** 410m is exploration; the v1 prompts only, so v2's 12
  held-out prompts are not read (`core/holdout.py`).
- **A3 does not redefine A5's break.** E1 is dated by D2 (a) / D4 (a). If A3's figure suggests
  another statistic dates the break better, that is a note to the user, and `design-A5.md`'s
  text goes back to the user before A5 opens (#231 finding 3's clause), not a change made here.
- **No seed download.** A5 registers before the seeds are downloaded (`design-A5.md` step 3;
  space is not the reason: `HDD_1TB`, A5's target, had 265 GB free on 2026-10-19). The 9 seeds'
  steps 0 and 143000 already cached (`p7_motifs/status-7.md`) are not read by A3.
- **M2's corpus** (unit 4): a sample of the Pile's validation split (Pythia's own distribution),
  about 2M tokens for the n-gram tables and 64k held-out tokens scored at each step. Downloading
  it is a corpus, not seed data; it is checked for overlap with the 7 prompts.
- **Steps 0 ≡ 1** are one point; no row may use their difference.

## Worth challenging (for the freeze)

1. **The half-rise against a ≤ 4000 reference.** It favours early moves by construction,
   which is what C2 sequences; a row whose move is late (E8 may be) reads "no change". Rejected:
   a full-axis reference, which lets 143000's drift set every row's half-point late.
2. **"Aligned" at one grid step.** It is the grid's resolution, so the reading cannot claim
   co-location finer than a doubling, and (§3) the warmup aligns with everything at 512–2000.
   Rejected: `CLAIM-B`'s pairing null, which is a test and belongs to a registration.
3. **Locating weights-only rows on the 18, not the 32.** It keeps rows comparable and throws away
   3000 (between 2000 and 4000), which is where U2's regime change sits. The dense curve is
   drawn, so the loss is visible.
4. **E11 in A3**, not left to the toy model. It answers 1e's one-well question as a row, which A1
   carried here; the cost is GPU minutes on existing code. Alternative: drop it, with why.
5. **E6 computed, against Belrose quoted.** Belrose already shows Pythia's n-gram-sequence loss
   turning at 64–256, on the energy break (`lit-A3.md`). Computing E6 on 410m's own 18 steps is
   what lets it be located by this table's rule and read against E1; the alternative is the
   quoted band only and 4 units. Their measure is loss on synthetic max-entropy sequences; E6
   is KL on real text, a different observable.
6. **E3 on the five typed members**, not the max over 384 heads (a bootstrap SE of a max is
   unreliable). It ties E3 to Track B's circuit; a head outside it that forms first is missed,
   which the drawn max over heads would show.
7. **The weights-only floor is placed.** Their noise is across heads or layers, not draws; the
   iid calibration may be too strict or too loose for them.
