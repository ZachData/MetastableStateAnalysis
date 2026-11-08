<!-- trackN_neurons/design-N1.md -->
# Track N — design: N1, the scan, the inventory and the unit

**Frozen 2026-11-08** (Blocked 144 decided D1 (d), D2 (a), D3 (a), the user's, as recommended;
before any N1 output). Proposed 2026-11-08; revised the same day after
`/challenge-pr` on #327 (findings 1–5 taken, "Revised after #327" below). Row:
`docs/ROADMAP.md` §7c N1. Opened by the order of 2026-11-02 (the user's: "add the motions to
analyse the MLP after part T"); Track T closed 2026-11-08, so N1 is next. The scan before it:
`docs/lit-N.md` (this unit). Tier 1. N1 registers nothing and scores nothing.

## What N1 answers (fixed before reading)

| # | question | what N2 / N3 takes from it |
|---|---|---|
| Q1 | The essay's method as written: what a pole is, the blind-maximum test, how thresholds are set; is the companion paper out? | N3's explanation step, if it has one |
| Q2 | Is N3's question answered on Pythia already? Read neuron-lineage's code: **its checkpoint list (are 512 and 1000 in it?)**, which sign it reads, its planted controls, its random-direction null, its licence; and the MBZUAI release's method | N3's scope (D3): replicate, extend, or drop |
| Q3 | Which Pythia neurons have a known job, with what evidence: `pythia-410m` 23.417 (token frequency; Stolfo et al.), `pythia-70m` L3N669 (German; which step is "checkpoint 10"), 2509.24198's Wasserstein neurons | N2: a natural control beside the planted one |
| Q4 | What a pole is under GeLU (below) | the unit N2 and N3 count |
| Q5 | The inventory: checkpoints on disk, the text to read neurons on, how many of each reading are populated at 143000 and at step 0 under the gated rule, and the cost of one step's capture on the 3080 | N3's sizing |
| Q6 | Given Q2–Q5, what N3 should be (put to the user at N1's close, not decided here) | N3's design |

**Grades** (as `trackT_timaeus/design-T1.md`): **confirmed** = read in a primary source (paper
text, code, `config.json`) or measured here; **leaning** = a README, blog or summary only;
**unknown** = neither within the budget. Each question gets one, with its source, in
`trackN_neurons/status-N.md`.

## What a pole is under GeLU (proposed; D1)

Pythia's neuron *i* at a token: pre-activation z = (W_in LN(x) + b_in)_i, post-activation
h = GeLU(z), write h · w_out,i. `hidden_act` is `gelu` (410m's config, read 2026-11-08), the exact
GeLU(z) = z Φ(z) in transformers, which every reader here uses. Checked in
`tools/math_checks/n1_gelu_pole.py` (8 of 8):

- h ≥ −0.16997, its one minimum at z* = −0.75179; h falls in z below z* and rises above it (G1).
- sign(h) = sign(z): no double-negative regime, which is SwiGLU's (G2, G5).
- At z ≤ −3, |h| < 0.0041: the most negative pre-activations write almost nothing (G3).
- Ranking by h reverses z below z* (G4): the most negative h are a band at z*, not the
  strongest negative votes.

**Three readings per neuron** (D1 (d), decided):

- **P+**: contexts ranked by z from the top (the same ranking as h, since h rises in z above z*).
- **P−**: contexts ranked by z from the bottom: the inputs that close the gate hardest, which
  write almost nothing (G3). Its support is the essay alone, which reads a ReLU neuron's
  negative side through the pre-nonlinearity votes (line 238) and the negative pole as a gate
  (line 496).
- **B**, the write band: contexts ranked by −h, i.e. by closeness to z*, where the negative write
  is largest (|h| ≤ 0.17). 2509.24198's evidence that the negative side matters is about this
  reading: it removes the output wherever z < 0, which is largest near z* and ≈ 0 at the most
  negative z (`docs/lit-N.md` §2).

U1 counts all three; which of them N3 explains is fixed in N3's design, before N3's output.

**Populated (gated in U1).** Per neuron and step, z is standardised over the library's tokens
(z̃ = (z − median) / MAD-scale), so the init's and the trained weights' scales do not decide it.
A reading's statistic is the mean z̃ of its top k contexts (for B, the mean closeness to z* of
its top k, on h). The reading is **populated** if its statistic exceeds the 95th percentile of
the same statistic over **512 random rows** at the same layer and step: Haar directions scaled to
the neuron's ‖W_in,i‖, with its own bias b_in,i (neuron-lineage's null, made per neuron). The
step-0 count is reported beside, not used as the floor. k = 10, 20, 50 are all reported, none
picked.

## U1 (D2 (a), decided)

1. Q1–Q3: read neuron-lineage's code (Q2's list first: checkpoints, sign), 2509.24198's Pythia
   sections, Stolfo's appendices F–H, 2311.00863's checkpoint index (its code; a question to
   the authors is the user's), and re-check for the companion paper.
2. **The gate, before any count** (calibration, `CLAUDE.md`): at layer 0 of `pythia-410m` step
   143000, where a token-indicator direction is defined (the LN'd embedding of one token), plant
   two rows: W_in,i set to the indicator (its P+ must be that token's positions, populated) and
   to its negation (its P− must be those positions, populated); and for B, a row whose bias puts
   that token's z at z* (its B must be those positions). On 512 held-out random rows the rule
   reads populated at ≤ 10 % for every reading and every k. A failed arm stops the counts.
3. Q5: `pythia-410m` steps 0 and 143000 (cached) on the library's `val` (1024 Pile documents of
   512 tokens, `library_hash` `47feaa321fa0`, through `core/library.py`), all 24 × 4096 neurons:
   each reading's top 50, streamed (never all z stored), z's quantiles per neuron; populated
   counts per reading, layer and k; 23.417's three readings printed as a look at a known neuron
   (its job is a write, so its contexts need not show it). GPU, wall time recorded for N3's
   sizing. Description: no reading is scored.

## Exit and budget

**Exit:** Q1–Q5 each carry a grade with its source; the readings are frozen with the math check
and the gated rule; Q6's options for N3 are written for the user. **Budget (D2):** 2 units, U0
(this: scan, design, math check) and U1 (the readings, the gate, the capture), the close-out
inside U1 if nothing is Parked. At the budget N1 closes at what it has, the rest marked unknown.

## Decisions (Blocked 144; decided 2026-11-08, the user's: D1 (d), D2 (a), D3 (a), as recommended)

| | options | recommended |
|---|---|---|
| D1 the readings | (a) P+ and P− by z, B beside, not counted; (b) by h: P− becomes B; (c) P+ only (the MBZUAI release and, as far as its README says, neuron-lineage read where a unit "fires most"); (d) all three, P+, P− and B, each counted under the same gated rule | **(d)**: (b) ranks the negative side backwards (G4); (c) drops both readings the essay and 2509.24198 point at; (a) leaves B, the reading 2509.24198 actually supports, uncounted; capturing three costs one top-k each |
| D2 budget | (a) 2 units as above; (b) close N1 now on this scan, U1's gate and capture moved into N2 | **(a)**: Q2's grade (code, not a README) decides D3, and N3 counts with the gated rule |
| D3 N3's direction (provisional; decided at N1's close) | (a) extend: date births against C2's 512 and 1000 on the 18-step grid, the three readings, the 9 seeds, with neuron-lineage's instrument re-run as the replication arm if its code runs; (b) N3 as written in ROADMAP §7c; (c) close Track N after N1, the question answered enough outside | **(a)**, held until Q2's checkpoint list is read: if neuron-lineage already reads 512 and 1000, (a)'s dated-births part is a replication, and (c) gains |

## Revised after #327

`/challenge-pr` on #327: accept with changes; all five findings taken.

1. The populated rule compared raw z with step 0 and could not fail (weights and biases grow).
   Now z is standardised per neuron and step and the rule is gated against random rows of
   matched norm, with planted rows as the effect arm (U1 step 2).
2. 2509.24198 was cited for P−; it supports B. P− rests on the essay alone; B is counted (D1 (d)).
3. The known answer checked P+ only and named no layer: now layer 0, P+, P− and B each planted.
4. Which GeLU Pythia trained with is dropped from Q4: every reader here uses the exact form, and
   G6 bounds the tanh form's difference at 4.7e-4.
5. "Already run" rested on a README that does not list the checkpoints: Q2 reads the list first,
   and D3 waits on it.

## Worth challenging

- The random-row null keeps the neuron's norm and bias but not its alignment with the stream's
  top variance directions; a trained row aligned with a large direction could read populated for
  that reason alone. The alternative, rows drawn from the stream's own covariance, is closer to
  "what any direction of this size sees" but harder to plant against.
- 524k tokens (the library) puts a top-20 context at about the 99.996th percentile; the essay
  read 10B tokens, neuron-lineage 2M. Whether the library is enough is U1's measurement, not
  assumed.
- Lecomte et al. (2312.03096) predict structure from init; step 0's counts are reported beside
  rather than subtracted for that reason.
