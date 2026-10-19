<!-- Proposed 2026-10-19; not frozen. The freeze is the user's (STATE Blocked 70). -->
# Track A — design: A3, one timeline

**Question** (`docs/ROADMAP.md` §6 A3): on pythia-410m's checkpoint axis, do the energy break,
the OV repulsive phase, induction formation, the n-gram milestones and the weights-only rows
change at the same steps, and in what order? This is the core of C2's sequence (§8).
**Opened 2026-10-19** by §9 row 6 (Blocked 68). Finding 4 on #231 (gate first or A3 first) was
decided by the user: A3 first, as already ordered. Scan: `lit-A3.md` (C2's `lit-C2.md` not
repeated). Tier 1, 410m, unregistered.

## Exit condition and budget

- **Exit:** one table and one figure. Every family row (§2) has a located change step, or
  "no change", or "refused" (its gate arm failed). Every named pair (§3) is stated as
  **aligned**, **misaligned** or **unread**, with the warmup named on each pair at or below
  2000. Or a recorded reason the table cannot be made (e.g. the locator's gate fails).
- **Budget: 5 units**, each its own PR, plus a close-out (writing only):
  (1) this design and the scan; (2) the producer, the existing rows, the locator and its gate;
  (3) the weights-only rows and the simulator row; (4) the n-gram milestones (M2);
  (5) the reading: table, figure, pairs, a page. At the budget A3 closes at what it knows.

Every threshold marked *placed* is set by judgement before any output, not calibrated.

## 1. The axis

The 19-step grid (0, 1, 2, 4, …, 512, 1000, 2000, 4000, …, 143000). Steps 0 and 1 are the same
weights (`p1b_hemisphere/status-1b.md`, §3.51), so there are **18 distinct points**, and the
locator reads 18. All 19 grid checkpoints and 15 more (3000, 5000, 7000, 9000, 11000, …,
120000) are on disk (`data/hf`). The weights-only rows are also computed on all 33 distinct
steps and drawn beside the grid, but they are **located on the 18**, so that every row has
the same resolution. Nothing exists between 512 and 1000 or between 1000 and 2000.

## 2. The family (fixed before any output)

| id | row | source | cost | located by |
|---|---|---|---|---|
| E1 | energy violations, Σ nViolRel at β = 1, 7 prompts | `data/analysis/colocation_panel.csv` `p1_energy_nviol_b1` (Phase 1 reports) | free | **`design-A5.md` D2 (a) / D4 (a) exactly**: the first step with V ≥ half the median of V over 1000–4000 |
| E2 | OV repulsive phase: median `attractive_energy_fraction_core` over 384 heads | `data/analysis/ov_per_head_series.json` (§3.12 A) | free | the generic rule (§4) |
| E3 | induction: max pooled induction attention over 384 heads | `data/analysis/behavioural_series.json` (P-I1) | free | generic |
| E4 | previous-token score of `L5H2` (and the max over heads) | B1's records if they cover 18 steps; else one hooked pass per step on the 7 prompts | free / GPU minutes | generic |
| E5 | effective rank on the sphere, mean over 7 prompts | `colocation_panel.csv` `p1_eff_rank_normed_mean` | free | generic |
| E6 | KL of the model's next-token distribution to a unigram and to a bigram predictor | new, unit 4 (corpus in §5) | GPU, ~1 h | generic, one row each |
| E7 | ΔW stable rank, `‖W_t − W_0‖_F² / ‖W_t − W_0‖₂²`, per matrix type (QKV, O, MLP in, MLP out), median over 24 layers | new, unit 3, weights only | GPU minutes | generic |
| E8 | ESD tail exponent α per matrix type, median over layers (HTSR) | new, unit 3, weights only | GPU minutes | generic |
| E9 | `W_V`'s symmetric part: share of heads with a negative-definite core (D1's V half; QK symmetry is block J's) | new, unit 3, weights only | minutes | generic |
| E10 | the LR schedule (warmup to about step 1430, then cosine) | Pythia's config, analytic | free | not located; the confound row |
| E11 | 1e's one-well question: the causal simulator (`sa_sim.py`) from the measured start at each step's per-prompt β, final cluster count | unit 3 (A1 at close) | GPU minutes | generic |

**Quoted rows** (drawn on the figure as bands, not located by the rule and not in the pairs):
RankMe's collapse at 32–256 and recovery by about 1000 (2602.15997, 410m); Belrose's unigram by
1000, bigram after 8000 (2402.04362); the induction phase change 1200–2400 (2209.11895);
R6f's embedding move 512 → 2000 (Phase 10); U2's regime change at 2000–4000 (1e); step 16's
collapse at state 23 in 3 of 7 prompts (A1); A2's against-specific steps (16, 32,
1000–16000, 143000), the MLP cancelling attention at 1000–4000, the fixed key from a period
(2000–4000) to a newline (from 8000), and the sink's per-token share rising from 1000.

## 3. The pairs that count

Named before output: **E1 against each of E2–E9 and E11** (nine pairs). Pairs among E2–E11 are
drawn and described, not read. **Aligned** (*placed*): the two located steps are equal or
one grid step apart. **Misaligned**: two or more grid steps apart. **Unread**: either row is "no
change" or "refused". One grid step is a factor of about 2 in step, so "aligned" at 512 means
anything in 256–1000; the grid cannot do better, and the text says so on every aligned pair.
Every pair whose later step is at or below 2000 carries "inside warmup (E10)": A3 cannot
separate a shared cause from the schedule (`lit-C2.md` §3).

## 4. The generic locator, and its gate (unit 2)

**Rule** (*placed*): for a row x over the 18 steps, let Δ(s) = x(s) − x(0) and let the reference
be the Δ of largest magnitude over steps ≤ 4000. The change step is the first step where |Δ|
≥ half that magnitude. **No change** if the reference is below 3σ, with σ the standard error of
the row's aggregate at step 0: a bootstrap over the units it averages (prompts, heads or
layers), 1000 resamples. Not the raw spread, which for a median over 384 heads would hide a
move of the median itself.
The 4000 cut is the same plateau end as D4 (a). It reads an early move only; a row whose main
move is after 4000 reads "no change" if the early move is under 3σ, and the figure shows it.
Beside (described, not read): `core.changepoint_colocation.change_profile`'s change-mass
centroid, so a row that the half-rise and the centroid place far apart is flagged.

**Gate, before any real row is located** (`CLAUDE.md` "Calibration gate"), on the 18 steps:
- *planted null:* 1000 flat series with noise at each row's own σ: "no change" in ≥ 95 %.
- *planted effect:* a sigmoid in log step centred between each adjacent pair of grid steps,
  amplitude 5σ, 1000 each: the located step is the one just past the centre in ≥ 90 %.
- *the real anchor:* E1 must come out at 512, as D4 (a) gave on the same file
  (`design-A5.md`); E2's floor at 1000 (§3.12 A) must read 512 or 1000 under the rule.
A failed arm refuses its rows; it is not tuned after the fact.

## 5. Fences

- **No registration and no adjudication.** 410m is exploration; the v1 prompts only, so v2's 12
  held-out prompts are not read (`core/holdout.py`).
- **A3 does not redefine A5's break.** E1 is dated by D2 (a) / D4 (a). If A3's figure suggests
  another statistic dates the break better, that is a note to the user, and `design-A5.md`'s
  text goes back to the user before A5 opens (#231 finding 3's clause), not a change made here.
- **No seed download.** A5 registers before the seeds are downloaded (`design-A5.md` step 3);
  the 9 seeds' steps 0 and 143000 already cached (`p7_motifs/status-7.md`) are not read by A3.
- **M2's corpus** (unit 4): a sample of the Pile's validation split (Pythia's own distribution),
  about 2M tokens for the n-gram tables and 64k held-out tokens scored at each step. Downloading
  it is a corpus, not seed data; it is checked for overlap with the 7 prompts.
- **Steps 0 ≡ 1** are one point; no row may use their difference.

## Worth challenging (for the freeze)

1. **The half-rise rule against a ≤ 4000 reference.** It favours early moves by construction,
   which is what C2 sequences; a row whose move is late (E8 may be) reads "no change". Rejected:
   a full-axis reference, which lets 143000's drift set every row's half-point late.
2. **"Aligned" at one grid step.** It is the grid's resolution, so the reading cannot claim
   co-location finer than a doubling. Rejected: `CLAIM-B`'s pairing null, which is a test and
   belongs to a registration, not to a tier-1 table.
3. **Locating weights-only rows on the 18, not the 33.** It keeps rows comparable and throws away
   3000 (between 2000 and 4000), which is where U2's regime change sits. The dense curve is
   drawn, so the loss is visible.
4. **E11 in A3**, not left to the toy model. It answers 1e's one-well question as a row, which A1
   carried here; the cost is GPU minutes on existing code. Alternative: drop it, with why.
5. **M2 as KL on real text**, not Belrose's loss on synthetic max-entropy sequences. It reads a
   different observable (how close the model is to n-gram statistics) and needs a corpus.
   Alternative: quote Belrose only and save unit 4 (lit Parked 1 could decide it).
6. **The quoted rows sit outside the pairs.** Their locations come from other definitions; laying
   them on the figure as bands without a rule is honest only if they are never counted.
