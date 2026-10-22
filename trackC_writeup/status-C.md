<!-- trackC_writeup/status-C.md -->
# Track C — STATUS

<!-- phase-card -->
## Card

- **Question:** When, along `pythia-410m`'s training, does the tokens' interaction energy stop being non-decreasing in depth, as the identity-weight particle picture predicts, is that step the run's or the recipe's, and where does it fall against the model's known developmental order (n-gram stages, previous-token and induction heads, the OV maps turning repulsive)?
- **Inputs:** none run under C's name; C2 reads Track A's and Track S's records (`trackC_writeup/design-C2.md` "What the record supports")
- **Results:**
  - C2 opened 2026-10-22 and its design froze the same day: seven lines the record supports, nine fences, an outline, a render step so every number comes from a record; Blocked 91–94 decided as recommended (left to Claude by the user): the dated, reproduced break as the headline, the anisotropy control and the LR-dose arm as new runs, Markdown rendered from records, S2 in the main text and S1 an appendix — `trackC_writeup/design-C2.md` "Decided 2026-10-22"
  - Unit 1, the reading: the plain E_β is proved monotone only at identity weights; wider hypotheses (symmetric scores tied to the values) make a weighted energy monotone, not E_β; so the break violates no theorem (F5 worded from the texts); published anisotropy along Pythia starts at step 10 000, not in the break's window (F9 narrowed); Mistral GPT-2 small (4000-step warmup, checkpoints every 50 steps to 2000) makes 92 (ii) feasible, so 3c runs (7 units + close-out); §2 and §8 drafted (revised after `/challenge-pr` on #253, findings 1–5) — `trackC_writeup/status-C.md` "C2 unit 1"
  - Unit 2, the render step: every number in the draft is a key that `numbers.py` reads from its record (A5's seeds and gate, A3's rows, S2's readings, Pythia's released configs; 13 records, each sha256 in the tracked `numbers.json`), and CI checks the draft against that table; §3–§5 drafted and the three figures drawn; no reading moved — `trackC_writeup/status-C.md` "C2 unit 2"
  - Unit 3, the anisotropy control (tier 1, description): the frozen rule reads **"mostly removed"** (centred plateau 20 against 57 at D2's 1e-3; Blocked 95 (a)), but that label's size half is a scale effect: centring makes E₁'s layer-to-layer changes about 30× smaller, and at a matched tolerance (1e-4, 1e-5; post hoc, `/challenge-pr` on #255) it reads "moved" (ratio 0.90–1.10, centred date 1000). What holds at every tolerance: at 512, the break's step, the centred count is 4–13 against 56–60, so the break at 512 is mostly the common direction's (F9 a finding), and a full-size centred break appears at 1000, induction's grid step; seed 0 only. 91 reopens (Blocked 96) — `trackC_writeup/status-C.md` "C2 unit 3"
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@2db9b14ba1, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the break is the LR schedule's or a training event's (F1); units 3b and 3c read it, 3c on another suite's warmup (description only) — `trackC_writeup/design-C2.md` "What C2 may not go past"
  - What the headline says now that the break at 512 is mostly the common direction's (Blocked 96, the user's) — `trackC_writeup/status-C.md` "C2 unit 3"
- **After Phase 10:**
  - none
- **Reviewed:** 2026-10-22 · body `2b2e40a850`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: frozen, Blocked 91–94 decided as recommended (left to Claude)** | `design-C2.md` "Decided 2026-10-22" |
| C2 1. reading (writing only) | **done 2026-10-22: F5 worded, F9 narrowed, 92 (ii) feasible (GPT-2 small), §2 and §8 drafted; revised after `/challenge-pr` on #253** | `lit-C2-unit1.md`, `draft-C2.md`; "C2 unit 1" below |
| C2 2. `numbers.py`, figures, §3–§5 | **done 2026-10-22: 102 keys from 13 records, 24 sentence claims checked, §3–§5 render, figures 1–3 drawn; revised after `/challenge-pr` on #254** | `numbers.py`, `numbers.json`, `draft-C2.src.md` → `draft-C2.md`, `fig/`; "C2 unit 2" below |
| C2 3. anisotropy control | **done 2026-10-22: frozen before output (95 (a), the user's), preflight cited; reads "mostly removed", the centred date 1000; Blocked 96** | `c2_aniso.py`, design "Unit 3"; "C2 unit 3" below |
| C2 3b, 3c. LR-dose arm (70m, 160m); warmup arm (Mistral GPT-2 small) | next, after Blocked 96 | design 92 (iii), (ii); 3c's separations `lit-C2-unit1.md` §5 |
| C2 4, 5, close-out | after 3c | — |

## C2 unit 3, the anisotropy control (2026-10-22)

92 (i) with Blocked 95 (a)'s size label, frozen in `design-C2.md` "Unit 3" and committed (`7878f5d`)
before the scored run; the preflight's report and its lean are there.

**Input:** `pythia-410m` (seed 0) at A5's 18 grid steps (HF cache, offline) × the 7 v1 prompts
(`prompts_hash` `976b62d677bc`), float32, eager, TF32 off, RTX 3080; code `d90bb06b7910`
(`trackC_writeup/c2_aniso.py`). Records `data/trackC/c2u3/2026-10-22-run/records/` (126, main
tree); reading `…/reading.json` sha256 `46f7654c96ba…`. Known answer: every record's E₁ equals A5's
seed-0 record (`trackA/a5/2026-10-20-u1/seed0`) to 0.0 (tolerance 1e-4), tokens identical.

| step | 0–128 | 256 | 512 | 1000 | 2000 | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V, uncentred (A5's) | 0 | 19 | 56 | 56 | 61 | 57 | 59 | 64 | 76 | 72 | 70 |
| V_c, centred, 1e-3 (the label's) | 0 | 0 | 4 | 15 | 20 | 24 | 26 | 29 | 32 | 39 | 59 |
| V_c at 1e-4 | 1, 1, 1, 1, 0, 0, 0, 0 | 1 | 11 | 55 | 56 | 55 | 57 | 56 | 51 | 58 | 75 |
| V_c at 1e-2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 4 | 10 |
| V, raw-centred, 1e-3 (beside) | 0 except 3 at 128 | 0 | 5 | 15 | 21 | 26 | 29 | 40 | 52 | 58 | 72 |
| mean cosine, layer mean | 0.42, 0.42, 0.42, 0.51, 0.70, 0.62, 0.39, 0.30 | 0.30 | 0.33 | 0.24 | 0.18 | 0.18 | 0.23 | 0.36 | 0.44 | 0.45 | 0.40 |

**The reading, by the frozen rule:** steps 0–128's centred counts first: all 0. Centred plateau
median 20 against 57 uncentred, ratio 0.35 < ½: **"mostly removed"**, with the centred half-rise
at 1000 beside (uncentred 512).

**Beside, not the label:**

| what | reads |
|---|---|
| the rule at the other tolerances | 1e-4: plateaus 55 / 61 (0.90), centred date 1000: "moved"; 1e-2: centred plateau 0: "removed". So the size half of the label depends on the tolerance: centring compresses E₁'s range (a centred cloud's mean cosine is near 0), so the same layer-to-layer change is a smaller relative drop |
| what holds at every tolerance | at 512, the break's step, the centred count is 4 (1e-3), 11 (1e-4) or 13 (sign level) against the uncentred 56, 60 or 60; the centred half-rise falls at 1000 wherever one is located, and the raw-centred cloud's at 1000 too |
| **post hoc, the scale (`/challenge-pr` on #255, finding 1)** | the centred E₁ sits about 0.002 above its floor (≈ 0.504 at step 0); its mean per-transition relative change is 3.9e-5 at step 0, 1.2e-3 at 512, 8.1e-4 at 1000 and 8.4e-4 at 4000, against 2.2–2.9e-2 uncentred: about 30× smaller, so one relative tolerance is far stricter on it. Matched more closely, the rule reads "moved": at 1e-4 the plateaus are 55 / 61 (0.90), at 1e-5 67 / 61 (1.10), the centred date 1000 in both; steps 0–128 at 1e-4 are 0–1. At sign level, float noise reads 24–35 at steps 0–8 ("unreadable"). From 1000 to 54000 the centred count at 1e-4 is flat (51–58), then 75 at 143000 |
| the common direction, by layer | at 128 the mean cosine rises with depth at all 24 transitions (0.02 → 0.71); at 256, 2 falling; at 512 it peaks at state 5 (0.40) and falls to 0.32 by state 15 (10 falling transitions); 8 at 1000, 10 at 4000. The mean cosine is E₁'s first-order term at β = 1, so this decomposes the same break; it is not separate evidence (finding 4) |
| the lean (design "Unit 3") | step 7000's 0.43 pointed to "mostly removed" and the re-inits away from "unreadable"; both held, and the 0.43 was the same scale effect |

**What it changes.** The frozen label stays on the record as "mostly removed". On a matched scale
the reading is 92 (i)'s "moved". Two parts hold at every tolerance. First, at 512 E₁'s
layer-monotonicity breaks mostly because the common direction stops growing with depth, so F9
becomes a finding. Second, the token-centred cloud breaks at full size one grid step later, at
1000, the step where L3 puts induction, previous-token and the LR schedule's half-point. L1–L4's
dates and `P-A5` stand as facts about E₁, the energy that was registered. **91 reopens because of
the count at 512, not the plateau ratio: Blocked 96, the user's.** 3b and 3c wait on it.

**Not established:** that the centred break at 1000 is a separate event rather than the same change
seen more weakly (the grid has no step between 512 and 1000); any centred-count rule calibrated on
its own scale (the post-hoc rows are description); any cause for the cosine profile's turn (no
component analysis); anything on the other 9 seeds (no centred records; A5's records hold E₁ only).

**Parked:** (1) `numbers.py` keys for `reading.json` (the label, the plateaus, the cosine profile)
and a figure beside fig 1: unit 4 (cost: one render pass; changes nothing until §4 is rewritten).
(2) The centred count on the 9 PolyPythias seeds, to say whether the 1000 date reproduces: about
20 GPU minutes (seed 0's 18 steps took 125 s) after a re-download of about 160 checkpoints (the
seed checkpoints are no longer in the HF cache); it could change 96's
wording ("reproduced" or "seed 0 only"); an option in 96.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_aniso run --out <dir>` (about 2 min
on the GPU, offline), then `read --out <dir>`; preflight `… preflight --out <dir>`, then
`python -m tools.design_preflight <dir>/preflight_input.json --out <dir>/preflight_report.json`.
Tests: `tests/test_trackC_aniso.py` (pure, 16).

## C2 unit 2, the render step, the figures and §3–§5 (2026-10-22)

Writing and tooling; no forward pass. `trackC_writeup/numbers.py`:

| step | what it does | needs |
|---|---|---|
| `produce` | reads the 13 records, computes 102 keys, writes `numbers.json` (each key's value and record id, each record's sha256, the code hash). Refuses: a record whose sha256 differs from the table's, or a key whose value or record moved (a new run directory, a code change), until named with `--reread`; any of `CLAIMS` false (24 assertions the draft's sentences make beyond the printed values: orders, "every", "none", margins, tallies); `score.json` not the one `claims/audits/p_a5_real_run.json` hashes; a V series off the 18-step grid | the main tree's `data/` |
| `render` / `check` | `draft-C2.src.md` + `numbers.json` → `draft-C2.md`; an unknown key or format, or a non-integer with no format, refuses; `check` fails if the tracked draft is not the render | any checkout (CI: `tests/test_trackC_numbers.py`, pure, 14, the claims on the tracked table among them) |
| `figures` | `fig/fig1_break.svg`, `fig2_timeline.svg`, `fig3_ov.svg` from `numbers.json` alone; byte-identical on re-run (one matplotlib); CI checks that they draw, not their bytes | matplotlib (`tests/test_trackC_figures.py`, deps) |

**Input set:** the record ids and sha256s in `numbers.json` `records` (A5: `trackA/a5/2026-10-20-u1/seed0/{gate,plan}.json`, `…-u2/score.json`; A3: `trackA/a3/2026-10-20-A3u5/a3_rows.json`; S2: `trackS/s2/2026-10-21-u2/{readings,gate_read}.json`; Pythia's `models/{70M,160M,410M}/pythia-*.yml` from `EleutherAI/pythia` main at `a19eecb807ec` (its newest commit, 2025-11-15), fetched 2026-10-22 to `data/trackC/configs/`, read by a line parser only; tracked: `claims/audits/p_a5_real_run.json`, `claims/registry.json`, `core/metrics.py`'s and `core/evalues.py`'s constants). Code hash in the table.

**What it found:** no number moved. Every rendered value equals its pointer in "What the record
supports" (L1–L4) and in A5's, A3's and S2's status sections. Two sentences the draft had carried
from summaries were checked against the source and reworded: the prompts are "hand-picked, not
sampled" (not "outside the training distribution": a Wikipedia paragraph and the *Iliad* may be in
the Pile), and PolyPythias's seeds differ in initialisation *and* data order.

**`/challenge-pr` on #254: accept with changes.** Findings taken: (1) §5's order line rendered
"E8 (1000) < E3 … (1000)", copied from A3's order, which placed E8 by its peak date: now E8 is
written between the groups with both dates; (2) the sentences' assertions are now checked (`CLAIMS`),
in `produce` and in CI, and the 23 unused keys are gone or used by a claim; (3) the guard now
compares key values and records, not only a record's hash at a fixed path; (4) the typed results
are keys (the late rise's steps, the tolerances, the block count, the time zone); "1024
sequences" stays as the Pythia paper's, cited; (5) grid positions are looked up on the 18-step
grid, and a series of another length refuses. (6) in part: the configs' upstream commit is
recorded here; CI checks that the figures draw but not their bytes (matplotlib's SVG moves
between versions). Answered on the PR.

**Choices made in the writing** (each in the PR's "Worth challenging"):

- A grid step named as a place on the axis ("between 512 and 1000") is a label and stays literal;
  a step that is a result is a key. The source header states the rule.
- F7 is written in the paper's terms (the check "is evidence neither about mechanism nor that
  trained weights resist the clustering"); `H-RESIST` and the relevance weight are ledger
  vocabulary and stay out of the text.
- Figure 3 (S2's R1 cells and G3′ against their 40 controls) is drawn now; §6, which cites it,
  is unit 4's. Figure 2 cuts each lane at −1.2 (E5's and E6u's late falls), said in its caption.
- §3.4 cites Vovk and Wang [2021] for the p-to-e calibrator (`core/evalues.py`); it is not in
  either scan's rows.

**Parked** (one line each: why / cost / which decision it could change):

1. The references list moves from unit 2 to unit 4 (the draft's own note had it here): titles and
   authors must be read from each paper, 22 ids plus Vovk–Wang and an ESD source for E8 /
   one metadata read / none, the list only. Carried to unit 4.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`, then `render` and
`figures` (`check` to verify). A changed record: read the change, then `produce --reread <id>`.

## C2 unit 1, the reading (2026-10-22)

Writing only; nothing ran. Read: the gradient-flow results (2312.10794 §3.2 and §3.4.1 in its
local LaTeX, 2605.04279), 2603.13314, the [S] rows §2 and §8 cite (each now [H] or [A]), a delta
scan since 2026-10-18, the Hub's checkpoint branches and tags for every candidate suite, and the
released configs. Record: `lit-C2-unit1.md`. `/challenge-pr` on #253 (accept with changes) found
the GPT-2 small tags, the weighted energy, an unrecorded [A] row, Blog 1's misquote and the [H]
mark's limits; all five taken (PR thread). What it changed:

| item | result | where it lands |
|---|---|---|
| F5 | the plain E_β is proved monotone only at Q = K = V = I (one constant head, continuous time, sphere, no MLP); symmetric scores tied to the values make a V-weighted energy monotone, not E_β; Pythia at every step and S2's edits meet none, so the break departs from a picture's prediction, not a theorem | `design-C2.md` F5; `draft-C2.md` §2.1 |
| the energy's sign | non-decreasing in every text; `lit-S2.md`'s "decreases" corrected, routed to S (no S2 reading moves) | `status-S.md` "Corrections received" |
| F9 | Bernas et al. read Pythia's activations from step 10 000 on and chiefly study the gradients; nothing reviewed measures activation anisotropy in 128–512; an unreviewed 70m project reports it half-built by 512, which raises unit 3's stakes | `design-C2.md` F9; `draft-C2.md` §8 |
| F1 | 2509.23024 ties its own first geometric phase to the warmup | `design-C2.md` F1; `draft-C2.md` §2.3 |
| 92 (ii) | **feasible on Mistral GPT-2 small**: 5 seeds, a 4000-step linear warmup to 6e-4, 604–609 checkpoint tags each (every 10 steps to 100, every 50 to 2000); medium's are gone; OLMo-2 1B has steps 0, 300, 10 000. It can separate steps or LR dose from warmup fraction or tokens, not within either pair | unit 3c runs; budget 7 units + close-out |
| 92 (iii) | the configs confirm the arm: 70m, 160m, 410m share the warmup (1430 steps), batch and schedule; peak LR 1e-3, 6e-4, 3e-4 | unit 3b as designed |
| 2603.13314 | weights on OLMo-2 7B; keys rise most, values flat; S1's `ov_read` is a different statistic | appendix A (unit 4) |
| novelty | "nobody reads this energy across training" not refuted as of 2026-10-22 | `draft-C2.md` §8 |

| Blog 1 | its random-weight energy rises monotonically; "under random weights too" is not its line (attribution corrected, routed to A) | `draft-C2.md` §8; `status-A.md` "Corrections received" |

Parked (`lit-C2-unit1.md`): GPT-2 medium's checkpoints off the Hub (carried to C2's close-out);
token-drift's per-step table (carried to unit 3b); where "under random weights too" comes from
(carried to unit 4).

## C2 unit 0, the design (2026-10-22)

Writing only; nothing ran. The design names what the record supports (L1–L7, each pointing to
the one place its numbers live), what the draft may not go past (F1–F9), the outline, and the
exit (a complete draft, every number rendered from a record, a fresh-context referee read
answered). F9, the anisotropy objection, is new to C2's row: `lit-C2.md` §1 named it and no
ROADMAP row carried it. Inputs read: the one-page closes of A3, A5, S1 and S2 and `lit-C2.md`.

## Corrections received

none
