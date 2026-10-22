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
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@2db9b14ba1, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the break is the LR schedule's or a training event's, and whether it is anisotropy's (F1, F9); units 3, 3b and 3c read each, 3c on another suite's warmup (description only) — `trackC_writeup/design-C2.md` "What C2 may not go past"
- **After Phase 10:**
  - The anisotropy control: V on token-centred clouds at the 18 steps on seed 0 *(forward pass, cached checkpoints)*
- **Reviewed:** 2026-10-22 · body `7493815de2`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: frozen, Blocked 91–94 decided as recommended (left to Claude)** | `design-C2.md` "Decided 2026-10-22" |
| C2 1. reading (writing only) | **done 2026-10-22: F5 worded, F9 narrowed, 92 (ii) feasible (GPT-2 small), §2 and §8 drafted; revised after `/challenge-pr` on #253** | `lit-C2-unit1.md`, `draft-C2.md`; "C2 unit 1" below |
| C2 2. `numbers.py`, figures, §3–§5 | **done 2026-10-22: 109 keys from 13 records, §3–§5 render, figures 1–3 drawn** | `numbers.py`, `numbers.json`, `draft-C2.src.md` → `draft-C2.md`, `fig/`; "C2 unit 2" below |
| C2 3, 3b, 3c. anisotropy control; LR-dose arm (70m, 160m); warmup arm (Mistral GPT-2 small) | next (3 first) | design 92 (i), (iii), (ii); 3c's separations `lit-C2-unit1.md` §5 |
| C2 4, 5, close-out | after 3c | — |

## C2 unit 2, the render step, the figures and §3–§5 (2026-10-22)

Writing and tooling; no forward pass. `trackC_writeup/numbers.py`:

| step | what it does | needs |
|---|---|---|
| `produce` | reads the 13 records, computes 109 keys, writes `numbers.json` (each key's value and record id, each record's sha256, the code hash); refuses a record whose sha256 differs from the table's until it is named with `--reread`, and refuses unless `score.json` is the one `claims/audits/p_a5_real_run.json` hashes | the main tree's `data/` |
| `render` / `check` | `draft-C2.src.md` + `numbers.json` → `draft-C2.md`; an unknown key or format, or a non-integer with no format, refuses; `check` fails if the tracked draft is not the render | any checkout (CI: `tests/test_trackC_numbers.py`, pure, 11) |
| `figures` | `fig/fig1_break.svg`, `fig2_timeline.svg`, `fig3_ov.svg` from `numbers.json` alone; byte-identical on re-run | matplotlib |

**Input set:** the record ids and sha256s in `numbers.json` `records` (A5: `trackA/a5/2026-10-20-u1/seed0/{gate,plan}.json`, `…-u2/score.json`; A3: `trackA/a3/2026-10-20-A3u5/a3_rows.json`; S2: `trackS/s2/2026-10-21-u2/{readings,gate_read}.json`; Pythia's `models/{70M,160M,410M}/pythia-*.yml` from `EleutherAI/pythia` main, fetched 2026-10-22 to `data/trackC/configs/`, read by a line parser only; tracked: `claims/audits/p_a5_real_run.json`, `claims/registry.json`, `core/metrics.py`'s and `core/evalues.py`'s constants). Code hash in the table.

**What it found:** no number moved. Every rendered value equals its pointer in "What the record
supports" (L1–L4) and in A5's, A3's and S2's status sections. Two sentences the draft had carried
from summaries were checked against the source and reworded: the prompts are "hand-picked, not
sampled" (not "outside the training distribution": a Wikipedia paragraph and the *Iliad* may be in
the Pile), and PolyPythias's seeds differ in initialisation *and* data order.

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
