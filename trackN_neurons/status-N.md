<!-- trackN_neurons/status-N.md -->
# Track N — STATUS

What the MLP's neurons do across training (`docs/ROADMAP.md` §7c, the user's, 2026-11-02).
No card yet: the first result lands at N1 U1.

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
| planted P+ | **fail**, 1 of 24 cells | `' ('` at k = 10: 4.60 vs threshold 4.70 (precision 1.0); all other cells pass |
| planted P− | pass, at the edge | same token, k = 10: 4.60 vs 4.63 |
| planted P± margin | | stat / threshold 0.99–1.73 for a perfect token indicator |
| planted B | pass, **could not fail** | planted stat 0 (the token sits at z*), the maximum; null thresholds −20 to −29: Haar rows of median norm never come within ~20 MADs of z* at layer 0 |
| held-out (512 Haar rows) | pass | populated rate P+ 4.3–5.1 %, P− 2.5–4.7 %, B 5.7–7.0 % (bar ≤ 10 %) |

**Read:** the rule is not powered where it was gated: a row that *is* a token detector clears the
Haar null by 1.0–1.7×, because Haar projections of a token-structured stream already have heavy
top tails (P± thresholds 4.3–4.7 MADs). B's null is degenerate: a random row of the neuron's norm
barely moves z, so B "populated" mostly says the neuron reaches z* at all. This is design-N1's
"Worth challenging" item 1 (random rows ignore the stream's large directions), measured. By the
frozen rule a failed arm stops the counts: **Q5 not run**, nothing captured at steps 0 or 143000.

**Grades (design-N1's scale), at U1:**

| Q | grade | source |
|---|---|---|
| Q1 | essay confirmed (local copy); companion paper **unknown, not out** (search 2026-11-08: press coverage says unreleased) | |
| Q2 | **confirmed** (code, `skydancerosel/neuron-lineage` `03ad1ab`, MIT) | `run_main.py`: checkpoints 128, 256, **512, 1000**, 2000 … 143000; `pythia-410m-deduped` (not 410m), layers 6, 12, 18, 512 neurons each; reads the post-GeLU h, top 200 (P+ only); null = a fixed random rotation of h-space; fp16. Births (L12, neuron basis): 128: 303, 256: 105, 512: 203, 1000: 337, 2000: 345, then falling; the random basis peaks at the same steps (L12 rand 512: 133, 1000: 269) |
| Q3 | 23.417 **confirmed** (2406.16254 text, §4, App. G); L3N669 **confirmed: "checkpoint 10" is step 512** | 2311.00863's code loads `checkpoint_index` through TransformerLens, whose Pythia list is 0, 1, 2 … 512, 1000 … 143000; its data has 154 indices (`contextual-ngrams` `data/checkpoint_ablation_data.csv`). Ablation loss rise: step 512 0.001, 1000 0.002, 10k 0.013, 20k 0.05, 90k 0.33 |
| Q3′ | 2509.24198 **confirmed** for B | §3: "zeroing only their negative pre-activations immediately before the nonlinearity", a′ = max(a, 0): removes the output where z < 0, largest at z*; §6's negative-negative pairs (two inputs both pushed negative "to different degrees") are the only text near P−; dynamics on the second MLP block of 70M–12B, the ablation on 1.4B |
| Q4 | confirmed (math check), the populated rule **refused by its gate** | above |
| Q5 | **not run** (the gate) | wall time: layer 0's gate 4 s; the capture's cost unmeasured |
| Q6 | open: Blocked 145 | D3 (a)'s dated births are a replication on the steps neuron-lineage already reads (512, 1000), so D3 (c) gains, as design-N1 said it would |

## Corrections received

