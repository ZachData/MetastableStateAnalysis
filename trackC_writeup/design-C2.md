<!-- trackC_writeup/design-C2.md -->
# Track C — design: C2, the interaction energy along Pythia's training axis (the draft)

Opened 2026-10-22 by `docs/ROADMAP.md` §9 row 6 (… → S1 → S2 → **C2's draft**). Row:
`docs/ROADMAP.md` §8 C2. Scan: `trackA_force/lit-C2.md` (2026-10-18; not repeated here, its
Parked 1 and 3 are unit 1's). **Proposed, not frozen: Blocked 91–94 are the user's.** Revised after `/challenge-pr` on #252 (accept with changes, findings 1–5 taken: the energy's sign, 92's LR-dose and continued-training options, L1's step-0 count, the tracked numbers table, the anisotropy rule). Unit 0
(this) is writing only; nothing ran. Every reading C2 quotes is tier 1 on 410m except `P-A5`,
the one adjudicated row (`claims/adjudications/P-A5.json`).

## What C2 is for

One paper: the interaction energy E_β of Geshkovski et al. (β = 1, tokens on the unit sphere,
`core.metrics.interaction_energy`), read layer by layer on `pythia-410m` along its training, with
the break in its layer-monotonicity dated, reproduced on 9 seeds and laid on the known
developmental order. The deliverable is a draft complete enough for the user to decide whether
and where to post it. Posting is public, so it is the user's and outside the exit.

Not C2: a mechanism for the break (A2 refused its ablation reading; S2 could not reach it), the
anti-collapse force by component, clusters (C1, S3), any rung but 410m (§10.3, §10.7).

## What the record supports

Each line is a sentence C2 may write, at the weight given, and the one place its numbers live.
The draft states these in words and takes its numbers from that place (§ "Numbers and figures").
The numbers below are quoted 2026-10-22 so the decisions can be weighed; where one disagrees
with its pointer, the pointer wins, and none of them is carried into the draft from here.

| # | sentence | weight | source |
|---|---|---|---|
| L1 | Through step 128, E_β is monotone across 410m's 24 layers on the 7 prompts (V = 0 on seed 0); V is 19 at 256, 56 at 512 and stays high after. At step 0 five of the 9 PolyPythias seeds show 1–6 violations (up to 8 at step 4, seed 3; none at 8–64), against plateaus of 44–58 | tier 1, description | `trackA_force/status-A.md` "A5 units 2 and 3"; `p1_mstate_tracking/status-1.md` "The developmental arc" |
| L2 | The break's step is the recipe's: on all 9 seeds the half-rise falls at 256 or 512 (7 at 512), registered before any seed's intermediate step was downloaded | **adjudicated** (`P-A5`): a reproducibility check against a uniform-location null nobody holds, at the relevance floor | `trackA_force/status-A.md` "A5 on one page (closed 2026-10-20)" |
| L3 | On seed 0's 18 steps the dated order is rank (8) < unigram (128) < bigram (256) < break = the OV attractive fraction's fall (512) < ESD (1000 / 512) < induction = previous-token = the LR schedule's half-point (1000); 7 of 11 read pairs aligned, all inside the warmup | tier 1, description, with a locator gate that failed once and passes | `trackA_force/status-A.md` "A3 on one page (closed 2026-10-20)" |
| L4 | The share of each move done at 512 orders the 1000 tie (break 0.98, OV 0.81, previous-token 0.35, induction 0.03; the LR schedule 0.36); on the seeds previous-token is further along than induction at 512 on 7 of 8 | description, seen after the readings | the same, reading 4; `trackA_force/design-A5.md` D10 |
| L5 | The heads' subspaces leave a random arrangement at 256 (QK, OV write, OV complex) or 512 (the rest), all inside the warmup | description against an exact null; ρ's effect side uncalibrated | `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)" |
| L6 | Editing every head's OV at once (its symmetric part, its rotation removed, or its own step-128 map) does not undo the break; the step-128 maps lower V at 512 and 1000 and raise it at 2000, beyond 40 random edits of their size | tier 1; the last clause description | `trackS_subspace/status-S.md` "S2 on one page (closed 2026-10-22)" |
| L7 | In-context copying survives removing the rotation at an edit of about 0.1 of a head's norm | tier 1, at that edit's size only | the same, item 3–4 |

## What C2 may not go past

Each is written into the draft beside the line it limits, not only in a limits section.

| # | fence | why | source |
|---|---|---|---|
| F1 | Nothing separates a training event from the LR schedule: every dated row from 128 to 1000 sits inside the warmup (to about step 1430), and the seeds share it | the scan's named confound | `trackA_force/lit-C2.md` §3; A5 "Not established" |
| F2 | "Break, then induction, then the OV phase" is not shown, in either reading of the OV phase (its fall starts with the break; its bottom ties induction at 1000) | A3 reading 5 | A3 on one page |
| F3 | The break is neither shown OV's nor shown not OV's (one OV-only edit with a known destination; V 29–61 against ≤ 7) | G3′'s reach | S2 on one page, "Parked and open at close" |
| F4 | 410m only; v1's 7 prompts (the Pile's validation split for the n-gram rows); nothing between grid steps (no checkpoint between 512 and 1000) | the rung policy and the grid | `CLAUDE.md` "Name the family"; A3 "Not established" |
| F5 | "No theorem predicts monotone" for trained or edited weights is worded from summaries until the gradient-flow results' hypotheses are read in full | unread | `trackS_subspace/lit-S2.md` Parked 1 |
| F6 | Nothing causal about which components push apart; no energy sentence about components before an energy split per component | A2's ablation calibration failed | `trackA_force/status-A.md` "A2 on one page (closed 2026-10-17)" |
| F7 | H-RESIST's `reject_null` on `P-A5` is not evidence that trained weights resist the collapse dynamics | one cheap factor; its validity for the e-process unargued | A5 on one page, item 3 and "Not established" |
| F8 | A non-monotone geometric arc along training (2509.23024, 2602.15997) and the stagewise order (the LLC literature) are not new; E_β's monotonicity read along a real training axis, dated and reproduced, is what is | the scan | `trackA_force/lit-C2.md` §1–§2 |
| F9 | Whether the break is anisotropy's: E_β is a moment of the Gram matrix, and anisotropy along Pythia's training is published (Bernas et al., ICML 2026); not measured here | the scan's one unmeasured objection | `trackA_force/lit-C2.md` §1; Blocked 92 (i) |

## The outline

| § | section | lines and fences | state |
|---|---|---|---|
| 1 | Introduction: the particle picture's prediction (E_β non-decreasing in depth: the identity-weight dynamics ascend it, and a drop between layers is a violation, `core/metrics.py` `energy_violation_severity`), what trained 410m does at step 143000, and the question "when does it stop?"; contributions L1–L3, L6 with F1 and F2 stated in the first page | L1–L3, L6; F1, F2 | to write (unit 4) |
| 2 | Background: the energy and its monotonicity (exact hypotheses), the developmental stages (n-grams, previous-token, induction), Pythia's warmup | F5, F8 | after unit 1 |
| 3 | Setup: 410m and its 18 steps, the 9 PolyPythias seeds, the prompts, E_β and V, the half-rise statistic, the registration order (`P-A5` before the download) | F4 | to write |
| 4 | The break, dated and reproduced (figure 1: V by step, 10 seeds) | L1, L2; F1, F7 | to write |
| 5 | The timeline (figure 2: A3's rows by step; the table) | L3, L4; F1, F2 | to write |
| 6 | Is the break the OV maps'? (figure 3: S2's R1 and G3′ cells) | L6, L7; F3 | to write (Blocked 94) |
| 7 | What would separate the schedule from training, and the anisotropy control | F1, F9 | Blocked 92 |
| 8 | Related work | F8 | after unit 1 |
| A | Appendix: the shared subspaces (L5) | L5 | Blocked 94 |
| B | Appendix: the gates (A3's locator, A5's gate on seed 0, S2's G0–G3) and why each null is the one used | — | to write |

## Numbers and figures

"Write once" applies to a paper too. The draft's Markdown source (under `trackC_writeup/`, made in
unit 2) carries a key where a number goes; a script under `trackC_writeup/` (numbers.py) reads each key from its
record (the A5 seeds' records, A3's table, S2's `gate_read.json`, the panel CSV) and renders
the draft; a key with no record is an error. Its producer step writes a tracked table under
`trackC_writeup/` (each key's value, record path and record sha256, as
`claims/audits/p_a5_real_run.json` does for A5), and the draft renders from that table on any
checkout, so CI can check the draft against it. Only the producer step needs the main tree's
`data/` (never added); a re-run whose record hash differs from the table's is refused until
the change is read. Every figure is drawn by the same script from the same records. Whether the
records C2 quotes are released beside a preprint, and where, is part of Blocked 93.

## Exit condition and budget

**Exit:** a complete draft: every section of the outline written (or dropped by a decision on
the record); every number rendered from a record by one command and every figure drawn by it;
each fence written beside the line it limits; a fresh-context referee read of the whole draft
answered point by point (fixed, or a reply saying why not). Or a recorded reason a section
cannot be written. **Budget: 4 units, plus one unit per run Blocked 92 takes, plus a
writing-only close-out** (as recommended: 6 units, 7 if unit 1 finds a suite for (ii)).

| unit | what | done when |
|---|---|---|
| 0 | this design (writing only) | Blocked 91–94 decided; the design frozen |
| 1 | reading: the gradient-flow results' hypotheses in full (`lit-S2.md` Parked 1), 2603.13314 in full (`lit-S1.md` Parked 1), the scan's [S] rows that §2 and §8 cite (`lit-C2.md` Parked 1 and 3), a delta scan since 2026-10-18, and whether a public suite with a different warmup and early checkpoints exists (for 92 (ii)) | §2 and §8 drafted; F5 worded from the texts; 92 (ii) feasible or not |
| 2 | `numbers.py`, the three figures, the draft's source with §3–§5 | §3–§5 render with every number from a record |
| 3, 3b, 3c | the new runs Blocked 92 takes, one unit each | each run's reading rule (below, for (i) and (iii)) frozen with a preflight report before its output |
| 4 | the rest of the draft (§1, §6, §7, appendices) | every section written |
| 5 | the referee read: `/challenge-pr` on the draft's PR, plus one fresh-context read of the draft as a reviewer would (no conversation history, the draft only), answered on the PR | every point answered |
| close-out | writing only: what C2 established, what it left open, every Parked item carried or dropped | the card current |

The ROADMAP gives C2 a target, mid-December 2026 (the user's), not a unit count. At the budget
C2 closes at what it has, unless the user extends it on the record.

## Decisions (each yours; recommendation first)

| # | decision | options | recommendation |
|---|---|---|---|
| 91 | the headline | (a) a reproducible, dated break in E_β's layer-monotonicity during warmup, on the developmental order (L1–L4), with "break, then induction, then the OV phase" stated as not shown (F2); (b) keep the sequence as the headline and hold the draft until something orders the OV phase against induction (Pythia has no step between 512 and 1000, so only S4's toy could, 3–4 units, past December); (c) a methods paper: the energy across training as the worked example of the gates and nulls (`PUBLICATION_IDEAS.md` Idea 6) | **(a).** It is what the record supports; (b) waits on a model that does not exist yet; (c) is a different paper, and the A3, A5 and S2 gates can be its appendix B here |
| 92 | new runs inside C2 | (i) **the anisotropy control**: V on the token-centred, re-normalised clouds at the 18 steps on seed 0, with each layer's mean cosine beside (F9); one unit, forward passes on cached checkpoints. *Rule, fixed here:* steps 0–128's centred counts are reported first; if any is ≥ half the centred 1000–4000 plateau median the control is refused as unreadable (a limit, stated); else A5's D4 (a) half-rise on the centred V reads **"survives"** if in (128, 512], **"removed"** if there is none by 4000, else **"moved"** to its step. *What it changes:* "survives", §4 says the break is not the common direction's; "removed", the break is anisotropy's rise, F9 becomes a finding and 91 reopens; "moved", §4 gives both dates. (ii) **the warmup arm**: V on one public training suite with a different warmup and checkpoints inside it, if unit 1 finds one (candidates unverified); one unit. (iii) **the LR-dose arm**: V at the 18 steps on `pythia-70m` and `pythia-160m`, which share 410m's 1430-step warmup at a higher peak LR (1e-3 and 6e-4 against 3e-4, by the Pythia paper's table; 70m's is in `PROJECT.md`, the others unit 1 checks against the released configs); one unit, a download. *Rule:* the same half-rise per size; a break at the same step on all three reads "step-locked" (the warmup's timing or the tokens seen, not told apart), an earlier one on the higher-LR sizes "dose-ordered" (it tracks how far training has moved the weights); size (width, depth) is confounded throughout, so description. 1b shares 410m's LR and cannot help. (iv) **continued training**, two arms from a pre-break checkpoint, warmup against flat LR: the only within-model test, but the 70m harness `PROJECT.md` describes is gone from disk, so ≥ 2 units to rebuild plus GPU hours per arm, and its results are reachability, not development. (v) none: F1 and F9 stated as limits. S4 (91 (b)) would also vary the warmup, with a known circuit, after December | **(i) and (iii), and (ii) if unit 1 finds a suite.** F9 and F1 are the objections a reader raises first. (i) is cheap and could change §4's wording. (iii) is the cheapest run that can move F1 (it separates the warmup's timing from the amount learned, not the schedule from the data). (ii) separates the schedule's timing if a suite exists. (iv) costs more than the rest together, for a reachability answer |
| 93 | the form | (a) Markdown in the repo (diffable, reviewed by PR), rendered by `numbers.py`, converted to LaTeX at the close-out for an arXiv preprint (cs.LG); a LessWrong post that supersedes Blog 1's "under random weights too" line is a separate decision, after, and so is releasing the quoted records beside the preprint (only the rendered table is tracked); (b) a LessWrong / Alignment Forum post only; (c) a workshop paper, whose page limit shapes the draft from the start | **(a).** It keeps the draft under the same review as the code and leaves the venue open until the draft exists. Where to post stays yours at the close |
| 94 | what S1 and S2 contribute | (a) S2 in the main text (§6) as a bounded negative with F3 beside it; S1 as appendix A, description only; (b) both in the main text; (c) both left out | **(a).** S2 is the only intervention on the break and its G3′ reading is the one that speaks to the sequence; S1's readings carry no calibrated effect size, so they support, they do not lead |

## Worth challenging (for the PR)

- **No fresh literature scan in unit 0.** The C2 scan is 4 days old and was made for this row;
  unit 1 does a delta scan and the full texts. The alternative was a full re-scan now, which
  would repeat `lit-C2.md` and could not change a decision above except 92 (ii)'s feasibility,
  which unit 1 checks anyway.
- **The anisotropy control (92 (i)) is new to C2's row.** The scan named it ("C2 states E_β's
  anisotropy share"), but no ROADMAP row carried it. Leaving it out keeps C2 writing-only after
  unit 1.
- **A render step for the numbers** costs a unit's share of tooling. The alternative is numbers
  typed into the draft with pointers, checked by hand at the referee read; with about 40 numbers
  across three tracks, a stale one is likely (`LESSONS.md` 1, docs wrong about what is merged,
  exists, populated).
- **Track C as its own directory** (`trackC_writeup/`), with `lit-C2.md` left in `trackA_force/`
  where it was written and pointed to, rather than moved.
