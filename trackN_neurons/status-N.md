<!-- trackN_neurons/status-N.md -->
# Track N — STATUS

<!-- phase-card -->
## Card

- **Question:** Before Track N reads what the MLP's neurons write across training: is it already answered on Pythia, what is a neuron's pole under GeLU, and can a rule tell a populated reading from a random row of the same size?
- **Inputs:** `pythia-410m` step 143000, layer 0, on the prompt library's `val` (k = 1024, `library_hash` `47feaa321fa0…`, 524,274 tokens), float32 on the RTX 3080; code `trackN_neurons/n1_capture.py`; records `trackN_neurons/n1_gate_u1.json` and `trackN_neurons/n1_sizing_u1.json`; outside code read: `skydancerosel/neuron-lineage` `03ad1ab`, 2311.00863's `contextual-ngrams`
- **Results:**
  - Scan and design: the companion paper is not out; the readings P+, P− and B defined under GeLU, the math checked (8 of 8) — `trackN_neurons/design-N1.md`, `trackN_neurons/status-N.md` "N1 U0, the scan and the design (2026-11-08)"
  - The populated rule refused by its own gate at layer 0: a perfect token detector clears the Haar null by only 1.0–1.7×, a covariance null sits above every plant, and B's arm could not fail, so no counts — `trackN_neurons/status-N.md` "N1 U1, the reads and the gate (2026-11-08): the gate failed, no counts"
  - neuron-lineage already dates births on `pythia-410m-deduped` at 512 and 1000, with its random basis peaking at the same steps; 70m's L3N669 is detected at step 512 and 410m's 23.417 is confirmed as a token-frequency neuron — `trackN_neurons/status-N.md` "N1 U1, the reads and the gate (2026-11-08): the gate failed, no counts"
  - Closed at its budget (Blocked 145 (a), the user's), Q6 put to the user — `trackN_neurons/status-N.md` "N1 on one page (closed 2026-11-08)"
- **Superseded / wrong:** none
- **Registry:** none, because N1 is tier 1 description and registers nothing
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - A populated statistic a random row cannot pass at layer 0, and a B statistic that can fail (carried to the ROADMAP's Track N row N2, no repair prescribed) — `docs/ROADMAP.md`
  - Q6: what follows N1 (Blocked 146)
  - Q5's counts and the cost of one step's capture; whether neuron births are basis-specific on 410m
- **After Phase 10:**
  - 23.417 and L3N669 as natural controls for any neuron statistic, read at their known steps (forward pass, 410m and 70m, GPU minutes)
  - neuron-lineage's instrument re-run on `pythia-410m` (not deduped) at C2's 512 and 1000, against a planted gate (forward pass, cached checkpoints, unmeasured)
- **Reviewed:** 2026-11-08 · body `b99bb89e8d`
<!-- /phase-card -->

What the MLP's neurons do across training (`docs/ROADMAP.md` §7c, the user's, 2026-11-02).

## N1 U0, the scan and the design (2026-11-08)

Writing only, no forward pass. Branch `claude/n1-design`.

- **Scan:** `docs/lit-N.md`. The companion paper to the essay is not out. **N3's core test is
  reported run on `pythia-410m` outside** (neuron-lineage: new jobs sprout next to old ones more
  often than for random directions, 10.4 % vs 6.5 % on its role embedding, most births at
  256–2000; README summary only), and on `pythia-160m` with LLM grading (two agents agree on
  ~40 %). Known-job neurons exist on Pythia in the literature: 410m 23.417 (token frequency),
  70m L3N669 (German, dated across checkpoints). 2509.24198 finds the negative side
  of GeLU neurons functional on a Pythia, through the output near z* (the write band).
- **The unit:** proposed in `trackN_neurons/design-N1.md` "What a pole is under GeLU": three
  readings per neuron, P+ and P− on the pre-activation z and the write band B at
  z* = −0.75179, each populated under a rule gated against random rows of matched norm. The closed-form
  facts it rests on pass (`tools/math_checks/n1_gelu_pole.py`, 8 of 8; re-run:
  `python3 tools/math_checks/n1_gelu_pole.py`). What the check does not prove is in its docstring.
- **Grades so far** (design-N1.md's scale): Q1 confirmed for the essay (local copy), unknown for
  the companion paper (not out); Q2 leaning (READMEs and a blog, not code); Q3 confirmed for
  L3N669's job (2311.00863's text), leaning for 23.417 (summary of 2406.16254), unknown for
  L3N669's step; Q4 the math confirmed (the training GeLU dropped, G6's bound); Q5 not started; Q6 open.
- **Blocked 144** (the user's): D1 the pole, D2 the budget, D3 N3's direction (design-N1.md
  "Decisions").
- **`/challenge-pr` on #327:** accept with changes, findings 1–5 taken (design-N1.md "Revised
  after #327"): the populated rule standardised and gated against random rows, planted P+, P−
  and B at layer 0; 2509.24198 re-cited to the band; the training-GeLU question dropped; Q2
  reads neuron-lineage's checkpoint list before D3.
- **Blocked 144 decided 2026-11-08** (the user's, as recommended): D1 (d) P+, P− and B all
  counted under the gated rule; D2 (a) 2 units; D3 (a) extend, provisional until Q2's checkpoint
  list is read. `design-N1.md` frozen before any N1 output.

## N1 U1, the reads and the gate (2026-11-08): the gate failed, no counts

Branch `claude/n1-u1`. Code `trackN_neurons/n1_capture.py` (tests `tests/test_n1_capture.py`,
3 of 3: band, null and readings against brute force). Input: library `val` k = 1024,
`library_hash` `47feaa321fa0…`, T = 524,274 tokens; `pythia-410m` step 143000, layer 0; float32,
RTX 3080, 4 s. Record `trackN_neurons/n1_gate_u1.json` (its `code` field is `0c93f0a`, the freeze;
the script ran uncommitted and is committed unchanged with this unit).
Re-run: `METS_DATA=<main>/data HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 python -m
trackN_neurons.n1_capture gate` (conda `mets`).

**The gate (design-N1 U1 step 2), 8 planted tokens (seeded among 247 with 200–5000 positions):**

| arm | result | numbers |
|---|---|---|
| checks | pass | the hook reads LN(embedding) at layer 0 exactly (max diff 0.0); a token's input is its id's alone (0.0) |
| planted P+ | **fail**, 1 of 24 cells | `' ('` at k = 10: 4.64 vs threshold 4.70 (precision 1.0); all other cells pass |
| planted P− | pass, at the edge; **not a separate arm** | same token, k = 10: 4.64 vs 4.63. P−'s plant is P+'s negated, so its statistic is P+'s exactly; only the threshold differs (a second draw of the same null) |
| planted P± margin | | stat / threshold 0.99–1.73 for a perfect token indicator |
| planted B | pass, **could not fail** | planted stat 0 (the token sits at z*), the maximum; null thresholds −20 to −29: Haar rows of median norm never come within ~20 MADs of z* at layer 0 |
| held-out (512 Haar rows) | pass, **~5 % by construction** | populated rate P+ 4.3–5.1 %, P− 2.5–4.7 %, B 5.7–7.0 % (bar ≤ 10 %): the same distribution as the null, so a check of the code, not of power |

**Read:** the rule is not powered where it was gated: a row that *is* a token detector clears the
Haar null by 1.0–1.7×, because Haar projections of a token-structured stream already have heavy
top tails (P± thresholds 4.3–4.7 MADs). B's null is degenerate: a random row of the neuron's norm
barely moves z, so B "populated" mostly says the neuron reaches z* at all. This is design-N1's
"Worth challenging" item 1 (random rows ignore the stream's large directions), measured. By the
frozen rule a failed arm stops the counts: **Q5's counts not run**, nothing captured at steps 0 or 143000.

**After `/challenge-pr` on #328** (accept with changes, findings 1–5 taken; sizing post hoc, for
Blocked 145, not a gate: `python -m trackN_neurons.n1_capture sizing`, record
`trackN_neurons/n1_sizing_u1.json`, code `c0ea485`): the k = 10 Haar threshold over seeds 0, 2, 3, 4
is 4.70, 4.55, 4.60, 4.57, so `' ('` (4.64) fails under seed 0 only: **the fail is the null's
sampling noise, and the gate had one live arm (P+) sitting at that noise**. A covariance null (rows
~ N(0, Cov X), top eigenvalue 25 % of the trace) puts the threshold at 8.51 at every k, above all
eight plants (4.64–7.47): at layer 0 every row's top k is one token, so **the statistic, not the
null, cannot separate a detector from a random row**. The covariance repair is dropped.
The script now refuses to run uncommitted and records its file hash (finding 4).

**Grades (design-N1's scale), at U1:**

| Q | grade | source |
|---|---|---|
| Q1 | essay confirmed (local copy); companion paper **unknown, not out** (search 2026-11-08: press coverage says unreleased) | |
| Q2 | **confirmed** (code, `skydancerosel/neuron-lineage` `03ad1ab`, MIT) | `run_main.py`: checkpoints 128, 256, **512, 1000**, 2000 … 143000; `pythia-410m-deduped` (not 410m), layers 6, 12, 18, 512 neurons each; reads the post-GeLU h, top 200 (P+ only); null = a fixed random rotation of h-space; fp16. Births (L12, neuron basis): 128: 303, 256: 105, 512: 203, 1000: 337, 2000: 345, then falling; the random basis peaks at the same steps (L12 rand 512: 133, 1000: 269) |
| Q3 | 23.417 **confirmed** (2406.16254 text, §4, App. G); L3N669 **confirmed: "checkpoint 10" is step 512** | 2311.00863's code loads `checkpoint_index` through TransformerLens, whose Pythia list is 0, 1, 2 … 512, 1000 … 143000; its data has 154 indices (`contextual-ngrams` `data/checkpoint_ablation_data.csv`). Ablation loss rise: step 512 0.001, 1000 0.002, 10k 0.013, 20k 0.05, 90k 0.33 |
| Q3′ | 2509.24198 **confirmed** for B | §3: "zeroing only their negative pre-activations immediately before the nonlinearity", a′ = max(a, 0): removes the output where z < 0, largest at z*; §6's negative-negative pairs (two inputs both pushed negative "to different degrees") are the only text near P−; dynamics on the second MLP block of 70M–12B, the ablation on 1.4B |
| Q4 | confirmed (math check), the populated rule **refused by its gate** | above |
| Q5 | **partly**: inputs confirmed, counts not run (the gate) | checkpoints on disk: `pythia-410m` 34 revisions cached (0 … 512, 1000 … 143000, C2's 18 among them); text: library `val`, 524,274 tokens; layer 0's gate 4 s, the capture's cost unmeasured |
| Q6 | open: Blocked 145 | D3 (a)'s dated births are a replication on the steps neuron-lineage already reads (512, 1000), so D3 (c) gains, as design-N1 said it would |

## N1 on one page (closed 2026-11-08)

**Asked** (`design-N1.md`; ROADMAP §7c N1): Q1–Q6, the scan, the inventory and the unit that N2 and
N3 would count. **Closed at its budget** (Blocked 145 (a), the user's, 2026-11-08): 2 units of 2
spent, Q1–Q4 graded with sources, Q5 partly (the counts stopped by the frozen rule), Q6 put to the
user (Blocked 146). Tier 1, description, nothing registered. Writing only at close.

**Established (tier 1):**

1. **What a pole is under GeLU**: three readings, P+ and P− on z and the write band B at
   z* = −0.75179, with the closed-form facts checked (`tools/math_checks/n1_gelu_pole.py`, 8 of 8).
   Ranking by h reverses z below z*, so "the negative pole" names two different readings.
2. **The populated rule fails its own gate** at layer 0 of `pythia-410m` 143000. A perfect token
   detector clears the Haar null by only 1.0–1.7×. The one fail (`' ('`, k = 10) is that null's
   seed noise. A covariance null sits above all eight plants. B's arm could not fail. At layer 0
   every row's top k is one token, so **the statistic cannot tell a detector from a random row
   there** ("N1 U1", the gate and the sizing).
3. **The dating is already done outside, on a neighbour model**: neuron-lineage reads
   `pythia-410m-deduped` at 128, 256, 512, 1000, 2000 … 143000 (P+ by h only). Births peak at
   512–2000, **and its random basis peaks at the same steps**, so the neuron basis leads the random
   one by 1.5× at 512 (203 vs 133) and 1.25× at 1000 (337 vs 269) at L12, not by a date (Q2).
4. **Two natural controls exist on Pythia**, each with a primary source: 410m's 23.417 (token frequency)
   and 70m's L3N669 (German), whose "checkpoint 10" is **step 512**. Its ablation cost there is
   0.001 and grows to 0.33 by 90000 (Q3).

**Not established:**

- **Q5's counts**: how many readings are populated at 0 or 143000, on any layer, and the cost of
  one step's capture. Unknown, by the frozen rule.
- **Whether any rank-and-threshold statistic works past layer 0.** The gate stood at layer 0
  because a planted detector is defined there, and that is also where every random row is a
  token detector. Deeper layers were never read.
- **Whether neuron births are specific to the neuron basis** on 410m. neuron-lineage's own numbers
  do not separate them by date.
- **The companion paper's method** (not out, 2026-11-08).

**Carried:** "a populated statistic a random row cannot pass at layer 0, and a B statistic that can
fail" goes to ROADMAP §7c N2 as an open problem, with no repair prescribed (the covariance null
was sized and fails every plant). **Parked:** none were raised. **Units:** U0 (scan, design, math
check), U1 (reads, gate, sizing after `/challenge-pr` on #328), and this close-out, writing only.

**Q6, put to the user (Blocked 146; options in `STATE.md`):** what follows N1. The options are
close Track N, an N2 that builds the instrument with the two natural controls, or N3 on
neuron-lineage's instrument, gated first.

## Corrections received

