<!-- trackC_writeup/lit-C5.md -->
# C5, M2 (the massive-activation coordinates projected out): literature scan before its design freezes (2026-10-23)

Trigger (`CLAUDE.md` "Literature scans"): C5 opens on M2 (Blocked 103 (a), the user's).
Scope: (1) when massive activations, attention sinks and outlier dimensions appear along a
training run, Pythia's above all; (2) whether they are on every token (a common direction, so
‖m‖²) or on a few (position 0, delimiters); (3) whether anyone has removed those coordinates and
re-read anisotropy or a Gram-matrix statistic, and against what control; (4) what fixes which
coordinates they are. Five web searches, three abstract pages, two HTML full texts read in part
(2026-10-23). Not repeated: `trackS_subspace/lit-S1.md` (Sun et al. 2402.17762 and 2607.09279 as
[S] rows), `trackC_writeup/lit-C2-unit1.md` (anisotropy along Pythia: Bernas et al.,
Razzhigaev et al.), `trackA_force/lit-C2.md`.

**Marks.** [A] the arXiv abstract page was fetched and read this session; [H] the HTML full text,
read in part (the first 100 000 characters); [S] a search engine's summary only.

## 1. Verdict on "nobody reads what the massive coordinates do to the break"

**Not refuted, but the nearest work is closer than C2's §8 says.** No paper found projects a
model's massive coordinates out and re-reads an energy or a monotonicity count across training.
One paper reads a Gram-type statistic on Pythia-410m across its checkpoints and dates its rise:

| work | what it does | bearing on M2 |
|---|---|---|
| Queipo-de-Llano, Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv, 2510.06477 (2025, rev. 2026-02) [H] | proves massive activations force representational compression (one row of large norm dominates σ₁); reads matrix entropy, the anisotropy p₁ = σ₁² / ‖X‖²_F, the sink rate and the BOS norm across "training checkpoints (1–143k steps) for Pythia 410M, 6.9B": **"all three phenomena emerge together around step 1k"**; in 410M the transition is at layer 5 for every input, the BOS norm dominates layers 5–15, and 410M grows massive activations "in more than one stage" | (i) a published date for 410m's massive activations, **about 1000, not 512**: a lean (stated, not a reading) that the coordinates arrive with C2's centred event, after the break; (ii) their p₁ is uncentred and close kin to ‖m‖², so C2's §8 and F9 ("published anisotropy along Pythia from step 10 000") are out of date: carried to C2p (Parked 1); (iii) their effect sits on one row (the first token), which a count over all tokens on the sphere weighs as 1/n |
| Gallego-Feliciano, McClendon, Morinelli, Zervoudakis, Saravanos, 2508.03616 (2025, rev. 2026-02) [H] | tracks the largest activation per layer across ≥ 37 Pythia checkpoints, 14M–12B, Sun et al.'s rule (|a| > 100 and ≥ 1000× the median), relaxed to "the top activation" for small models; "not present at model initialization"; early and late layers peak early and decay, middle layers rise log-like, "particularly stronger for … 410M or more"; dataset released | the threshold M2 cannot use as is: 410m's end-state ratios are 140–350× (`trackS_subspace/status-S.md` "S1 unit 3"), under 1000×. No warmup discussion; no 410m date in the text read |
| McClendon, Gallego-Feliciano, Saravanos, 2610.00423 (2026-09-30) [A] | the "life cycle": stochastic birth, growth driven by weight decay, consolidation onto a few redundant carriers; the sink channels **"vary across random seeds but stabilize early within each run"** | M2's coordinate set must be **per seed**; a set read at a seed's end state is the same set it had early, if "stabilize early" holds for Pythia (the abstract does not say when). The search summary's "410m separation settles around 10⁴ steps" was not on the abstract page: [S] |
| Gu et al., 2410.10781 (NeurIPS 2024 / ICLR 2025) [S] | the sink emerges "after effective optimization on sufficient training data"; smaller peak LRs give a weaker sink, also with steps matched | the sink is LR-dose-sensitive: whatever M2 dates shares F1 (C4's question, not M2's) |
| 2607.09279 [S] (already in `lit-S1.md`) | clamps the three massive dimensions to zero or their per-layer median at every residual boundary, against **random-dimension controls** | the control M2 uses: the same number of coordinates, drawn at random |
| Machina & Mercer, NAACL 2024 [S]; Rudman & Eickhoff, 2305.19358 [S] | large Pythia models' embedding spaces are isotropic through the final LayerNorm; LLM representations dominated by a few outlier dimensions, isotropy regularisers | context only; neither reads early checkpoints |

## 2. On every token, or on a few?

The sink papers put massive activations on the first token and delimiters (2510.06477, Sun et
al.). S1's read of 410m at 143000 found coordinates 357, 130 and 966 at |x| ≈ 500–650 at
position 0 *and* in the maximum over the other positions (`massive.json`, layers 7–22). A maximum
is not a mean: whether the coordinates are on every token, and so carry ‖m‖², is unread. **That
share, ‖m restricted to the set‖² / ‖m‖² on the normalised tokens, is the first thing M2 reports**
(design R0), before any count.

## What the scan changes in the design

1. **Per-seed coordinate sets** (2610.00423): no seed-0 set is carried to the other 9 seeds.
2. **A ratio threshold below Sun's 1000×** (2508.03616 relaxes it for small models; 410m sits
   at 140–350×): the rule is a ratio to the layer's median |x|, fixed in the design.
3. **The random-coordinate control** (2607.09279): the same number of coordinates, drawn 20
   times per seed outside the set.
4. **A date for the coordinates themselves** (2510.06477's "around 1k"): M2 reports each seed's
   first grid step at which the set passes the threshold, beside the two C2 dates. The lean that
   it is 1000 is the paper's, on seed 0's model, and is written down here before any output.

## Parked

1. **C2's §8 and F9 against 2510.06477** (why: a published Gram-type statistic on Pythia-410m
   dated "around step 1k" is closer to the draft's centre than anything §8 cites, and F9's
   "published from step 10 000" is no longer the published record; cost: a paragraph and a
   reference, plus reading its Figure 2 at the grid steps; could change: §8's novelty sentence,
   F9's wording, possibly §4.1's comparison). *Carried to C2p* (the draft's posting thread owns
   every edit after C2's close).
2. **2508.03616's released dataset** (why: per-layer top activations for 410m at ≥ 37 steps, a
   check on M2's birth dates from outside the repo; cost: a download and a join; could change:
   nothing in M2's reading, only its outside agreement). *Carried to M2's unit 1 as a beside
   row if the data is reachable without an account; else dropped there with why.*
