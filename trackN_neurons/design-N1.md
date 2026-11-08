<!-- trackN_neurons/design-N1.md -->
# Track N — design: N1, the scan, the inventory and the unit

**Proposed 2026-11-08, not frozen** (Blocked 144, the user's). Row: `docs/ROADMAP.md` §7c N1.
Opened by the order of 2026-11-02 (the user's: "add the motions to analyse the MLP after part
T"); Track T closed 2026-11-08, so N1 is next. The scan before it: `docs/lit-N.md` (this unit).
Tier 1. N1 registers nothing and scores nothing; its one forward pass (U1) is description.

## What N1 answers (fixed before reading)

| # | question | what N2 / N3 takes from it |
|---|---|---|
| Q1 | The essay's method as written: what a pole is, the blind-maximum test, how thresholds are set; is the companion paper out? | N3's explanation step, if it has one |
| Q2 | Is N3's question answered on Pythia already? Read neuron-lineage's code (sign handling, checkpoints, planted controls, the random-direction null, licence) and the MBZUAI release's method | N3's scope (D3): replicate, extend, or drop |
| Q3 | Which Pythia neurons have a known job, with what evidence: `pythia-410m` 23.417 (token frequency; Stolfo et al.), `pythia-70m` L3N669 (German; which step is "checkpoint 10"), 2509.24198's Wasserstein neurons | N2: a natural control beside the planted one |
| Q4 | What a pole is under GeLU (below), and which GeLU Pythia trained with (exact or tanh: GPT-NeoX's config and fused kernel) | the unit N2 and N3 count |
| Q5 | The inventory: checkpoints on disk, the text to read neurons on, how many poles are populated at 143000 against step 0, and the cost of one step's capture on the 3080 | N3's sizing |
| Q6 | Given Q2–Q5, what N3 should be (put to the user at N1's close, not decided here) | N3's design |

**Grades** (as `trackT_timaeus/design-T1.md`): **confirmed** = read in a primary source (paper
text, code, `config.json`) or measured here; **leaning** = a README, blog or summary only;
**unknown** = neither within the budget. Each question gets one, with its source, in
`trackN_neurons/status-N.md`.

## What a pole is under GeLU (proposed; D1)

Pythia's neuron *i* at a token: pre-activation z = (W_in LN(x) + b_in)_i, post-activation
h = GeLU(z), write h · w_out,i. `hidden_act` is `gelu` (410m's config, read 2026-11-08), the exact
GeLU(z) = z Φ(z) in transformers. Checked in `tools/math_checks/n1_gelu_pole.py` (8 of 8):

- h ≥ −0.16997, its one minimum at z* = −0.75179; h falls in z below z* and rises above it (G1).
- sign(h) = sign(z): no double-negative regime, which is SwiGLU's (G2, G5).
- At z ≤ −3, |h| < 0.0041: the most negative pre-activations write almost nothing (G3).
- Ranking by h reverses z below z* (G4): the most negative h are a band at z*, not the
  strongest negative votes.

**Proposed unit.** A pole is (layer, neuron, sign). **P+** is the neuron's contexts ranked by z
from the top (the same ranking as h, since h rises in z above z*). **P−** is its contexts ranked
by z from the bottom: the inputs that close its gate hardest, which write almost nothing (G3).
P− is read as what the gate keeps out, on its inputs, as the essay reads a ReLU neuron's
negative side through the pre-nonlinearity votes (essay line 238) and as 2509.24198 finds the
negative pre-activations functional on a Pythia (`docs/lit-N.md` §2). Beside each neuron, not a
pole: the **write band** B (z within a fixed window around z*, where the negative write is
largest, |h| ≤ 0.17), with its share of the neuron's write.

**Populated.** A pole is populated at a step if its k-th context lies beyond the matched
pole of the same neuron at step 0 (each neuron's initial weights as its own floor). k and the
margin are D1's; U1 reports the counts at several (k = 10, 20, 50) without picking one.

## U1 (if D2 (a))

Reading only, plus one capture:

1. Q1–Q3: read neuron-lineage's code, 2509.24198's Pythia sections, Stolfo's appendices F–H,
   2311.00863's checkpoint index (its code or a question to the authors is the user's), and
   re-check for the companion paper.
2. Q4: GPT-NeoX's GeLU at Pythia's training commit (config `bias_gelu_fusion` and its kernel).
3. Q5: `pythia-410m` steps 0 and 143000 (cached) on the library's `val` (1024 Pile documents of
   512 tokens, `library_hash` `47feaa321fa0`, through `core/library.py`), all 24 × 4096 neurons:
   per neuron z's quantiles and each pole's top 50 by z, streamed (never all z stored); the
   populated counts at k = 10, 20, 50; 23.417's poles printed as a look at a known neuron (its job
   is a write, so its contexts need not show it). GPU, wall time recorded for N3's sizing.
   Known answer before the batch: on a planted neuron (a copy of one 143000 neuron with W_in row
   replaced by a token-indicator direction) the capture's P+ is that token's positions.

No rule is read and nothing is scored, so no calibration gate; the known answer guards the
capture.

## Exit and budget

**Exit:** Q1–Q5 each carry a grade with its source; the pole is frozen with its math check;
Q6's options for N3 are written for the user. **Budget (D2):** 2 units, U0 (this: scan, design,
math check) and U1 (the readings and the capture), the close-out inside U1 if nothing is
Parked. At the budget N1 closes at what it has, the rest marked unknown.

## Decisions (Blocked 144)

| | options | recommended |
|---|---|---|
| D1 the pole | (a) P+ and P− by z, the write band beside; (b) poles by h: P− becomes the band at z*; (c) P+ only (the MBZUAI release and, as far as its README says, neuron-lineage read where a unit "fires most") | **(a)**: (b) ranks the negative side backwards (G4); (c) drops the side 2509.24198 finds functional and the essay calls half the reading |
| D2 budget | (a) 2 units as above; (b) close N1 now on this scan, U1's capture moved into N2 | **(a)**: Q2's grade (code read, not a README) decides D3, and Q5's step-0 floor is the populated rule N3 counts with |
| D3 N3's direction (provisional; decided at N1's close) | (a) extend: date births against C2's 512 and 1000 on the 18-step grid, both poles, the 9 seeds, with neuron-lineage's instrument re-run as the replication arm if its code runs; (b) N3 as written in ROADMAP §7c; (c) close Track N after N1, the question answered enough outside | **(a)**, held until Q2: what is new here is the dated events, the negative pole and the seeds, not the local-growth test |

## Worth challenging

- P− by the most negative z reads inputs whose write is ≈ 0 (G3). If the job lives in the
  write, P− has none, and the band B (z ≈ z*) would be the negative pole. Kept beside, not chosen.
- The step-0 floor per neuron assumes the init's poles are noise; Lecomte et al. (2312.03096)
  predict structure from init, which the floor would subtract.
- 524k tokens (the library) puts a top-20 context at about the 99.996th percentile; the essay
  read 10B tokens, neuron-lineage 2M. Whether the library is enough is U1's measurement, not
  assumed.
