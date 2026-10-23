<!-- trackC_writeup/design-C2.md -->
# Track C — design: C2, the interaction energy along Pythia's training axis (the draft)

Opened 2026-10-22 by `docs/ROADMAP.md` §9 row 6 (… → S1 → S2 → **C2's draft**). Row:
`docs/ROADMAP.md` §8 C2. Scan: `trackA_force/lit-C2.md` (2026-10-18; not repeated here, its
Parked 1 and 3 are unit 1's). **Frozen 2026-10-22: Blocked 91–94 decided as recommended, left to Claude by the user ("Blocked 91–94: <your picks>"): 91 (a), 92 (i) and (iii) and (ii) if unit 1 finds a suite, 93 (a), 94 (a)** ("Decided 2026-10-22" below). Revised after `/challenge-pr` on #252 (accept with changes, findings 1–5 taken: the energy's sign, 92's LR-dose and continued-training options, L1's step-0 count, the tracked numbers table, the anisotropy rule). Unit 0
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
| F1 | Nothing separates a training event from the LR schedule: every dated row from 128 to 1000 sits inside the warmup (to about step 1430), and the seeds share it | the scan's named confound; 2509.23024 ties its own first phase to the warmup ([H], unit 1); 3b (70m, 160m): consistent with step and cumulative LR, an LR level excluded, step against cumulative not told apart on the grid (`status-C.md` "C2 unit 3b"); 3c (GPT-2 small, a log warmup): E₁'s larger rise at step's band, a first stage at the dose's, no centred date; the frozen "consistent with step" turns on the plateau window and one re-init unit, so it does not narrow F1 (Blocked 101 decided (a), the user's, 2026-10-22, against the outcome frozen in "Unit 3c"; `status-C.md` "C2 unit 3c", "C2 unit 4") | `trackA_force/lit-C2.md` §3; A5 "Not established"; `lit-C2-unit1.md` §3 |
| F2 | "Break, then induction, then the OV phase" is not shown, in either reading of the OV phase (its fall starts with the break; its bottom ties induction at 1000) | A3 reading 5 | A3 on one page |
| F3 | The break is neither shown OV's nor shown not OV's (one OV-only edit with a known destination; V 29–61 against ≤ 7) | G3′'s reach | S2 on one page, "Parked and open at close" |
| F4 | 410m only; v1's 7 prompts (the Pile's validation split for the n-gram rows); nothing between grid steps (no checkpoint between 512 and 1000) | the rung policy and the grid | `CLAUDE.md` "Name the family"; A3 "Not established" |
| F5 | The break departs from the identity-weight picture's prediction; it violates no theorem. The plain E_β is proved monotone only at Q = K = V = I (one constant head, continuous time, the sphere, no MLP); under symmetric scores tied to the values (one head, QᵀK = V; many heads, each Mₕ = W_V,h or nearly) the monotone quantity is a V-weighted energy, not E_β. Pythia's weights at every step, step 0 included, and S2's edits meet none of these, so L1's monotone steps 0–128 are an empirical fact *(worded from the texts at unit 1, 2026-10-22; the weighted energy after `/challenge-pr` on #253)* | 2312.10794 §3.2, §3.4.1 ([R], the local LaTeX); 2605.04279 Theorems 11–12, §10 ([H]) | `lit-C2-unit1.md` §1 |
| F6 | Nothing causal about which components push apart; no energy sentence about components before an energy split per component | A2's ablation calibration failed | `trackA_force/status-A.md` "A2 on one page (closed 2026-10-17)" |
| F7 | H-RESIST's `reject_null` on `P-A5` is not evidence that trained weights resist the collapse dynamics | one cheap factor; its validity for the e-process unargued | A5 on one page, item 3 and "Not established" |
| F8 | A non-monotone geometric arc along training (2509.23024, 2602.15997) and the stagewise order (the LLC literature) are not new; E_β's monotonicity read along a real training axis, dated and reproduced, is what is | the scan | `trackA_force/lit-C2.md` §1–§2 |
| F9 | Whether the break is anisotropy's: E_β is a moment of the Gram matrix; anisotropy along Pythia's training is published from step 10 000 on (Bernas et al., ICML 2026, App. D; its main measure is the gradients'), not in the break's window, where only an unreviewed 70m project reports it rising by 512; measured here by unit 3 *(narrowed at unit 1, 2026-10-22: the scan's row was abstract-level; read at unit 3, 2026-10-22: "mostly removed" as frozen, "moved" to 1000 on a matched scale (post hoc); at 512 mostly the common direction's at every tolerance, so a finding, not a fence; `status-C.md` "C2 unit 3", Blocked 96)* | the scan's one unmeasured objection | `trackA_force/lit-C2.md` §1; `lit-C2-unit1.md` §3–§4; Blocked 92 (i) |

## The outline

| § | section | lines and fences | state |
|---|---|---|---|
| 1 | Introduction: the particle picture's prediction (E_β non-decreasing in depth: the identity-weight dynamics ascend it, and a drop between layers is a violation, `core/metrics.py` `energy_violation_severity`), what trained 410m does at step 143000, and the question "when does it stop?"; contributions L1–L3, L6 with F1 and F2 stated in the first page | L1–L3, L6; F1, F2 | written (unit 4; headed on the two events, 98 (c)) |
| 2 | Background: the energy and its monotonicity (exact hypotheses), the developmental stages (n-grams, previous-token, induction), Pythia's warmup | F5, F8 | drafted (unit 1, `draft-C2.md`) |
| 3 | Setup: 410m and its 18 steps, the 9 PolyPythias seeds, the prompts, E_β and V, the half-rise statistic, the registration order (`P-A5` before the download) | F4 | drafted (unit 2, `draft-C2.src.md`) |
| 4 | The break, dated and reproduced (figure 1: V by step, 10 seeds) | L1, L2; F1, F7 | drafted (unit 2); §4.1, the centred cloud and F9, added at unit 4 |
| 5 | The timeline (figure 2: A3's rows by step; the table) | L3, L4; F1, F2 | drafted (unit 2); E1c added at unit 4 |
| 6 | Is the break the OV maps'? (figure 3: S2's R1 and G3′ cells) | L6, L7; F3 | written (unit 4; 94 (a)) |
| 7 | What would separate the schedule from training (figure 4); the anisotropy control moved to §4.1 by 98 (c) | F1 | written (unit 4; 3b, 3c; F1 stands, Blocked 101 (a)) |
| 8 | Related work | F8, F9 | drafted (unit 1, `draft-C2.md`) |
| A | Appendix: the shared subspaces (L5) | L5 | written (unit 4; 94 (a)) |
| B | Appendix: the gates (A3's locator, A5's gate on seed 0, S2's G0–G3) and why each null is the one used | — | written (unit 4; with the centred counts' arms) |

## Numbers and figures

"Write once" applies to a paper too. The draft's Markdown source (`trackC_writeup/draft-C2.src.md`,
made in unit 2; rendered to `draft-C2.md`) carries a key where a number goes; a script under `trackC_writeup/` (numbers.py) reads each key from its
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
writing-only close-out** (as decided: 6 units, 7 if unit 1 finds a suite for (ii)).

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
| 93 | the form | (a) Markdown in the repo (diffable, reviewed by PR), rendered by `numbers.py`, converted to LaTeX at the close-out for an arXiv preprint (cs.LG); a LessWrong post that updates Blog 1 is a separate decision (*the "under random weights too" line once named here is not Blog 1's, whose random-weight energy rises monotonically: `lit-C2-unit1.md` Parked 3*), after, and so is releasing the quoted records beside the preprint (only the rendered table is tracked); (b) a LessWrong / Alignment Forum post only; (c) a workshop paper, whose page limit shapes the draft from the start | **(a).** It keeps the draft under the same review as the code and leaves the venue open until the draft exists. Where to post stays yours at the close |
| 94 | what S1 and S2 contribute | (a) S2 in the main text (§6) as a bounded negative with F3 beside it; S1 as appendix A, description only; (b) both in the main text; (c) both left out | **(a).** S2 is the only intervention on the break and its G3′ reading is the one that speaks to the sequence; S1's readings carry no calibrated effect size, so they support, they do not lead |

## Decided 2026-10-22

Left to Claude by the user ("Blocked 91–94: <your picks>"); every pick is the recommendation
above, for the reason given there. The options stay as written in the table.

| # | taken | what it fixes |
|---|---|---|
| 91 | (a) | the headline is the dated, reproduced break on the developmental order; F2 stated on the first page |
| 92 | (i) and (iii); (ii) if unit 1 finds a suite | units 3 (anisotropy, seed 0) and 3b (LR dose, 70m and 160m), each rule as written above, each with a preflight report before its output; 3c only if (ii) is feasible. (iv) and (v) not taken |
| 93 | (a) | Markdown under `trackC_writeup/`, rendered by `numbers.py`; LaTeX at the close-out; venue, a LessWrong post and the records' release decided after the draft exists |
| 94 | (a) | S2 is §6 with F3 beside it; S1 is appendix A, description only |

Budget as decided: units 0–5 (with 3 and 3b) + close-out; 3c adds one. **Unit 1 found a suite
for (ii)**: Mistral GPT-2 small, 5 seeds, a 4000-step warmup at 6e-4, checkpoints every 10 steps
to 100 and every 50 to 2000 as Hub tags (`lit-C2-unit1.md` §5; first read as gone, corrected
after `/challenge-pr` on #253). So 3c runs, and the budget is 7 units + close-out. What 3c can
tell apart is fixed there before its rule: optimizer steps or LR dose against warmup fraction
or tokens, not within either pair; description, since model, data and tokenizer differ.
Blocked 96 (b), the user's, 2026-10-22, adds unit 3s (the centred count on the seeds): **8 units +
close-out**.

## Unit 3: the anisotropy control (frozen 2026-10-22, before any scored output)

Code `trackC_writeup/c2_aniso.py` (`d90bb06b7910`); pure tests `tests/test_trackC_aniso.py`. The
rule is 92 (i)'s with one addition, the size label (Blocked 95 (a), the user's, taken before any
centred number existed, the preflight's included). The scored family is seed 0's 18 grid steps ×
the 7 v1 prompts, as in A5's records; nothing else enters the label.

| item | fixed as | why |
|---|---|---|
| the cloud | U = the L2-normalised hidden states, the cloud A5's E₁ reads (all positions, 25 states incl. the post-final-LN one); centred: U − its token mean, each row re-normalised (float64); a row shorter than 1e-6 refuses | F9's anisotropy is a Gram moment of U: its mean off-diagonal cosine is (‖Σ Uᵢ‖² − n) / (n (n − 1)), so subtracting U's mean is what removes it; a vector added on the sphere is removed exactly (test) |
| the energy and count | E₁ (β = 1) on the centred cloud; a violation at D2 (a)'s 1e-3; V_c summed over 7 prompts × 24 transitions; tolerances 1e-4 and 1e-2 beside | A5's D2 (a), unchanged except for the centring |
| beside, not in the label | the raw-centred cloud (x − token mean, then normalised); each layer's mean cosine of U, prompt-averaged; A5's E₁ recomputed per record | the other reading of "centred"; F9's anisotropy as a number; A5's e1 is the known answer (≤ 1e-4 per layer and the tokens' hash equal, else refuse) |
| **95 (a), the size label** | first: centred plateau median (1000, 2000, 4000) 0 → **removed**; centred plateau < ½ the uncentred plateau on the same records → **mostly removed**, the date and ratio beside | 92 (i)'s half-rise reads only the shape: with any positive plateau it finds a step ≤ 4000, so a plateau that fell from 57 to 3 would read "survives", which licenses "not the common direction's" (test) |
| 92 (i), then | steps 0–128's centred counts reported first; any ≥ ½ the centred plateau → **unreadable**; else the half-rise (`a3.d4a`) at 256 or 512 → **survives**; at 1000–4000 → **moved** | as written in 92 (i); exhaustive once the plateau is positive |
| what each changes | removed / mostly removed: F9 becomes a finding (the break's count is mostly the common direction's), 91 reopens; survives: §4 says the break is not the common direction's; moved: §4 gives both dates; unreadable: a limit, stated | 92 (i), with 95 (a)'s label read as "removed" |

**The preflight** (`data/trackC/c2u3/2026-10-22-preflight/`, main tree; same code). Report
`preflight_report.json` sha256 `ca14c358404d3427c524e32b5edc400d960010946ee8e06e3c628c83c314e217`
(input `490707e60ab6…`, plants `preflight_plants.json` `47ce90a1eac7…`). The statistic per unit is
the centred relative drop per (prompt, transition). The null is 3 matched re-inits (Pythia's σ,
every tensor within 1 %: small 0.01974–0.01978, wang 0.00260–0.00261), 504 values; trained is the
off-grid step 7000 (168); the planted control (B) is on the re-inits at 0.25, 0.5 and 1 × the
median token norm.

| check | result |
|---|---|
| refusals | `floor` only (1/505 → E 11.2 < 20): C2 computes no E; the label is a count against a tolerance, so `decision_p` 0.05 stands in. No band refusal (trained outside the null's band 82.7 % < 90 %), no size refusal (dose 0 at α 4.8 %), no dose-side or power refusal (every dose 100 %) |
| (B) spread on the later state, centred, units over 1e-3 (of 504) | 0 → 501 → 504 → 504: the centred count sees a planted break that is not a common direction |
| (A) a common vector on the earlier state, re-inits (of 504) | uncentred 9 → 415 → 504 → 504; centred 0 → 0 → 0 → 0 (largest shift 3.9e-5): the centred count does not see it |
| (A) at step 7000 (of 168) | uncentred 61 → 152 → 161 → 168; centred 26 → 27 → 27 → 29 (largest shift 2.6e-3): it leaks, by at most 3 units, at a plant that saturates the uncentred count |

**A lean, stated before the run.** At 7000, off the grid, 26 of 168 units violate centred against
61 uncentred (0.43), and the re-inits' centred drops sit far below the tolerance (band
[−1.7e-4, 5.7e-5]). Both point toward "mostly removed" and away from "unreadable" on the scored
steps. They are not readings; the label is read only off the 18 steps. 95 (a) was taken before
either number existed.

## Unit 3s: the centred count on the seeds (frozen 2026-10-22, before any seed's centred record)

Blocked 96 (b), the user's: one unit on the 9 PolyPythias seeds, "with a rule on the centred
energy's own scale (its tolerance set from an off-grid step's per-transition spread, and a plant
read on both counts) frozen before the download". No download was needed: all 162 checkpoints
are in `HDD_1TB/mets_data/hf` (`LESSONS.md` 1). Code `trackC_writeup/c2_seeds.py`, pure tests
`tests/test_trackC_seeds.py`. The scored family is seeds 1–9 × the 18 grid steps × the 7 v1
prompts; seed 0 (unit 3's records) is read beside as the development set.

**96 (b)'s tolerance was refused by its own gate** (preflight v0,
`data/trackC/c2u3s/2026-10-22-preflight-v0/`, before the rule froze, no seed read). The
median-matched τ_c = 1e-3 × s_c / s_u = 2.81e-5 (s the median |relative change| per (prompt,
transition) at seed 0's step 7000: 1.41e-2 uncentred, 3.96e-4 centred) failed all three checks:
plant B (token-independent spread, a change both counts should see) turned 2.0× and 5.0× more
of the units centred than uncentred at its two partial doses; plant A (a common vector) leaked
0.24 into the centred count at dose 1; the re-inits' centred rate was 8.5 %. A sweep over τ_c
from 1e-5 to 1e-3 on the same populations found none that reads plant B alike at more than one
dose: the centred count's dose response is steeper. Read at step 7000, the centred/uncentred
rate ratio is 1.0 at the median-matched τ, 0.93 null-matched, 0.44 B-matched. **So a size
comparison between the two counts has no calibration, and the rule reads dates only.**

**Blocked 97, decided (a), the user's, in session after preflight v0 and before any seed
record.** The options were: (a) dates only, each on its own count's plateau, at four centred
tolerances, a seed counting only if the event holds at every arm; (b) one τ_c, B-matched, with the
size event kept (it still fails plant B at dose 0.1); (c) the seeds as description with no frozen
rule; (d) no seeds, 96 (a).

| item | fixed as | why |
|---|---|---|
| the arms | τ_c ∈ {7.55e-5, 1e-4, 3e-4, 7.58e-4}: the null-matched τ (the re-inits' centred false-positive rate equal to the uncentred one at 1e-3, 1.8 %), unit 3's 1e-4, 3e-4, and the B-matched τ (plant B at 0.25 turns the same share of step 7000's units both ways, 0.467); uncentred at D2's 1e-3 throughout | each arm is "matched" by a different defensible criterion, and they span a factor of 10; an event that holds at all four does not rest on which criterion is right |
| the gate, per arm (the thresholds set after v0, with 97 (a)) | the re-inits' centred rate ≤ 5 %; plant B at 0.25 turns ≥ 0.4 of step 7000's units not already violating; plant A at 0.5 turns ≤ 0.1 | the null is clean; the count can see a change that is not the common direction's; it does not see one that is |
| per seed | centred V per arm; per arm, `a3.d4a`'s half-rise on the centred V's own plateau; unreadable if that plateau is 0 or a step 0–128 count is ≥ half of it (92 (i)) | the half-rise is relative to its own count's plateau, so no cross-count scale enters |
| event 1 | the centred half-rise is later than the seed's uncentred break (A5's date, recomputed from the records and checked against A5's V) | "the break at its step is mostly the common direction's", read on each count's own scale |
| event 2 | the centred half-rise is at 1000 | unit 3's date on seed 0; L3's grid step of induction, previous-token and the LR half-point |
| a seed | carries an event only if it holds at all four arms; unreadable at any arm reads both False, flagged | conservative across arms, as D7 (a) |
| the tally | K1, K2 of 9; ≥ 7 "reproduced", ≤ 2 "seed 0's only", else "mixed" | at most two seeds the other way (PolyPythias names two 410m outliers, seeds 3 and 4); description, no null |
| known answer | each record's e1 equals A5's record for that seed and step (≤ 1e-4, tokens' hash equal), else refuse; each seed's V equals A5's `score.json` V, else refuse | the records are the ones A5 scored |
| beside | V_c at 1e-3; the raw-centred V per arm; the layer-mean cosine per step; D10's previous-token and induction dates per seed (A5's `score.json`) | description |

**What each outcome changes (fixed now).** K1 and K2 both "reproduced": 96 (c) is open on the
seeds, and the choice between (a) and (c) goes to the user (Blocked 98). K1 "reproduced", K2 not:
(a), with F9 a finding on the seeds and the 1000 date as seed-0 description (or "on K2 of 9").
K1 not "reproduced": F9's finding stays seed 0's, and (a) is written with that limit. "Mixed"
is reported as the count, and the user decides.

**Plant A's leak.** At dose 1 (a common vector of the median token's norm) plant A turns 0.05–0.24
of step 7000's units centred, depending on the arm. A leak can only make the centred count rise
with the common direction, so its date earlier. Both events say the centred date is *later*, so
the leak works against them.

**The preflight** (`data/trackC/c2u3s/2026-10-22-preflight/`, main tree, code `e7a1e159f247`, the
frozen code; it re-derives the two matched arms and refuses unless they round to `TAU_ARMS`).
Report `preflight_report.json` sha256 `6c584c1aafb89e191a78ce69cfb524b24c6ffffd7022a24bebc2d338d269cd43`
(input `f27bbafd9494…`; per-arm gate `preflight_plants.json` `0d566ced2973…`; per-unit drops
`preflight_units.json` `df989090c7b0…`). The tool refuses `floor` only (1/505 → E 11.2 < 20), as
in unit 3: C2 computes no E, and the reading is a date against a tolerance. Plant B on the
re-inits at τ = 7.55e-5: 0.08, 0.17, 0.88, 0.99, 1.00 at 0.01–0.25.

| arm | re-init centred rate | B at 0.25, share | A at 0.5, share | gate | beside: step 7000 centred / uncentred rate; A at dose 1; B at 0.1 |
|---|---|---|---|---|---|
| 7.55e-5 (null-matched) | 0.020 | 0.94 | 0.07 | pass | 0.34 / 0.36; 0.22; 0.16 |
| 1e-4 | 0.004 | 0.94 | 0.06 | pass | 0.33 / 0.36; 0.21; 0.15 |
| 3e-4 | 0 | 0.92 | 0.02 | pass | 0.27 / 0.36; 0.13; 0.05 |
| 7.58e-4 (B-matched) | 0 | 0.46 | 0.01 | pass | 0.16 / 0.36; 0.05; 0.01 |

**A lean, stated before the run.** Seed 0 reads both events at 1e-4 (centred date 1000 against
512) and at 1e-3 (1000). It is the development set, not a reading.

## Blocked 98 and 99, decided 2026-10-22 (the user's, in session, before any 3b output)

| # | taken | the options as written | what it fixes |
|---|---|---|---|
| 98 | (c), as recommended | (a) keep `P-A5`'s E₁ break as the headline, restated as the common direction's at its step, the centred date at 1000 beside; (c) re-head the draft on the two events | the draft is headed on two events: the common direction stops growing with depth at E₁'s break (`P-A5`, the registered anchor), and the token-centred cloud breaks at 1000, beside D10's heads (tier 1, 9 + 1 seeds); 3b and 3c read both dates as primary; §1, §4, §5 rewritten at unit 4 (no extra unit) |
| 99 | (a), as recommended | (a) re-label 92 (iii) by which schedule predicts each date on the grid, run both sizes; (b) the same on 70m only; (c) drop 3b, F1 stays a limit, 3c carries the schedule | 92 (iii)'s labels overlapped on Pythia's grid (below); 3b reads each size's two dates against three schedules' grid predictions |

## Unit 3b: the LR-dose arm (frozen 2026-10-22, before any scored record)

Code `trackC_writeup/c2_dose.py`; pure tests `tests/test_trackC_dose.py`; the predictions'
check `tools/math_checks/c2_dose_grid.py` (D1–D4, and what it does not prove). The scored
family is `pythia-70m` and `pythia-160m` × the 18 grid steps × the 7 v1 prompts; 410m enters
only as its 10 seeds' dates (`REF`). Description throughout: size (6, 12, 24 layers; width 512,
768, 1024) moves with the LR, and there is one seed per size.

**Why 92 (iii)'s labels had to change.** All three sizes share the warmup (1430 steps, linear;
the released configs), the batch and so the tokens per step, and the data order; only the peak
LR differs (1e-3, 6e-4, 3e-4). 410m's seeds already break anywhere in (128, 512]. If the break
waits for the weights to have moved a fixed amount (the cumulative LR, p t² / 2W), 70m's
crossing is at (70, 280], which the grid reads as 128, 256 or 512: the steps "step-locked"
predicts, plus 128. So "the same step on all three" would have read "step-locked" off outcomes
the dose reading predicts too.

| item | fixed as | why |
|---|---|---|
| the schedules | **step**: 410m's date (the warmup's timing or the tokens seen, not told apart); **cumulative**: t · √(3e-4 / lr); **instantaneous**: t · 3e-4 / lr | the two dose forms bracket "how far the weights have moved" (Adam's step is about lr per coordinate) and "the LR reaches a level"; the check's D1–D2 |
| 410m's dates (`REF`) | E₁'s break {256, 512} (A5, seeds 0–9); centred {512, 1000} (unit 3s, any seed at any arm; 512 is seed 6 at 3e-4, a tie) | a schedule is excluded only if no 410m seed at any arm maps to the date: the union is the conservative band |
| a prediction | a grid date g stands for a crossing in (prev(g), g]; scaled by the schedule's factor, the grid steps whose interval meets it (`predicted`) | the grid is all the records see; D4 checks it against a scan |
| per size, per count | `a3.d4a`'s half-rise on its own plateau; unreadable if the plateau is 0 or a step 0–128 count is ≥ half of it (92 (i)) | unit 3s's per-count dating |
| the centred arms, per size | unit 3s's procedure on that size: null-matched and B-matched re-derived on 3 re-inits of its config and its off-grid step 7000, beside 1e-4 and 3e-4; each arm gated as unit 3s (re-init rate ≤ 5 %, plant B at 0.25 ≥ 0.4, plant A at 0.5 ≤ 0.1); frozen into `TAU_ARMS` from the preflight, which refuses if a re-run derives other arms | 410m's arms are on 410m's scale; the procedure, not the numbers, transfers |
| the centred date | the dates of the arms that pass their gate and are readable; a schedule is consistent if it predicts any of them; unreadable if no arm is both | arms that disagree widen the date, which can only keep a schedule |
| the uncentred count | D2's 1e-3, as A5; its preflight (`preflight_input_u.json`, the design tool) reported, since 1e-3 was never gated off 410m | refuse rather than degrade on a new model |
| **the reading** (`joint`) | the schedules consistent with every readable date (2 sizes × 2 counts); all three → **"not told apart"**; none → **"neither"**; else **"consistent with …"** | one statistic, fixed before output |

**What 3b can and cannot tell apart (the check's outcome table, fixed now).** Every date
step-locked predicts, cumulative predicts too, so **3b cannot separate step-locked from
cumulative**; it can exclude step-locked (a date earlier than 410m's: E₁'s break at ≤ 128, the
centred date at ≤ 256) or instantaneous (E₁'s break at 512, the centred date at 1000), and a
date at 2000 or later excludes all three.

| size | E₁'s break: date → consistent | centred: date → consistent |
|---|---|---|
| 70m | 64 → inst.; 128 → cum., inst.; 256 → all; 512 → step, cum. | 128 → inst.; 256 → cum., inst.; 512 → all; 1000 → step, cum. |
| 160m | 128 → cum., inst.; 256 → all; 512 → step, cum. | 256 → cum., inst.; 512 → all; 1000 → step, cum. |

**What each outcome changes (fixed now).** "Not told apart": §7 says the LR-dose arm could not
separate the schedules on Pythia's grid, and F1 stands as written. "Consistent with step and
cumulative": §7 says the dates do not follow the LR level; F1 stands (the timing and the
cumulative dose not told apart). Instantaneous or cumulative without step: §7 says the dates move
earlier with the peak LR, size confounded; F1 is narrowed to "a dose, not a fixed step". "Neither":
reported, with size the named confound. 3c is read the same way on its own grid.

**Blocked 100, decided (a), the user's, in session after preflight v0 and before any scored
record.** Preflight v0 (`data/trackC/c2u3b/2026-10-22-preflight-v0/`, unit 3s's gate as frozen)
passed 160m at every arm and refused 70m at every arm: plant B at 0.25 turned 0.06–0.21 of 70m's
units centred and 0.07 uncentred (the gate needs 0.4). 70m's layers move about 5× more than
410m's (median |relative change| at step 7000: 0.068 uncentred against 410m's 0.014; 0.0033
centred against 0.0004), so a plant sized in token norms is small against them. A sweep on 70m's
step 7000 (off the grid; scratch, not kept) saw B at 0.5 at 1.0 centred and 0.45 uncentred, with
A at 0.5 leaking 0.04. The options were (a) **plant B's gate dose per size: the smallest dose at
which the uncentred count at 1e-3 turns ≥ 0.4** (0.25 on 410m and 160m, the frozen dose; 0.5 on
70m), thresholds unchanged, or (b) the gate as frozen, 70m unreadable and 3b on 160m alone. The
rule was written after seeing 70m's sweep, so it is not blind to 70m's dose.

**The preflight** (`data/trackC/c2u3b/2026-10-22-preflight/`, main tree; code `02a3cae5c04f`, the
frozen code, which refuses unless a re-run derives `TAU_ARMS`). Per size: 3 re-inits of its config
(Pythia's σ) and its step 7000; 126 re-init and 42 (70m) or 84 (160m) trained units.

| size | B's dose | arm | re-init centred rate | B share | A at 0.5 | gate |
|---|---|---|---|---|---|---|
| 70m | 0.5 (uncentred 0.41) | 9.61e-5 null-matched; 1e-4; 3e-4 | 0 | 1.00 | 0.04 | pass |
| 70m | | 4.78e-3 B-matched | 0 | 0.41 | 0.03 | pass |
| 160m | 0.25 (uncentred 0.44) | 1.25e-4 null-matched | 0.004 | 0.78 | 0.08 | pass |
| 160m | | 1e-4; 3e-4 | 0.012; 0 | 0.79; 0.77 | 0.05; 0.02 | pass |
| 160m | | 8.22e-4 B-matched | 0 | 0.44 | 0 | pass |

Reports (`tools/design_preflight.py`): 70m centred `d61bc9aab699…`, uncentred `3516fef2d5a5…`;
160m centred `a9d1e243eac6…`, uncentred `392f13ea7a80…` (inputs and `preflight_plants.json`
hashed in each size's directory). Refusals, each argued:

- `floor` on all four (E ≤ 5.6 < 20): C2 computes no E, and the reading is a date against a
  tolerance, as in units 3 and 3s.
- 70m centred `band:reinit` (92.9 % of step 7000's units outside the re-inits' band): the re-inits
  set only the null-matched arm's false-positive rate; no unit is scored against their band, and
  the date is read on the count's own plateau. The re-inits' centred drops are 20× smaller than
  the trained ones (scale 1.7e-4 against 3.3e-3), as on 410m.
- 70m uncentred `power` (B at 0.5 reaches p ≤ 0.05 against the re-inits in 69 % of draws, under 80 %):
  **not fixed: 70m's uncentred count is the weakest reading in 3b.** On the trained step 7000 it
  turns 0.41 of the units at that dose. It is kept because the date reads the count's rise to half
  its own plateau over 42 units per step, not single units; it is a limit stated beside 70m's
  E₁ date.

**No lean.** No record of `pythia-70m`'s or `pythia-160m`'s E₁ along training exists in the
repo (searched: `data/analysis` holds Phase 8's per-head readings of 70m, none an energy by
step); nothing of either size was read before the freeze except the preflight's off-grid step 7000
and re-inits.

## Unit 3c: the warmup arm (frozen 2026-10-22, before any scored record)

Code `trackC_writeup/c2_warmup.py`; pure tests `tests/test_trackC_warmup.py`; the predictions'
check `tools/math_checks/c2_warmup_grid.py` (W1–W5, and what it does not prove); the logged LR
`trackC_writeup/gpt2_small_lr_log.json`. The scored family is Stanford CRFM Mistral GPT-2 small's
5 seeds (alias, battlestar, caprica, darkmatter, expanse) × the 35 tags of `GRID` × the 7 v1
prompts (GPT-2's tokenizer, ≤ 512 tokens); 13 states, so 84 units a step. 410m enters only as its
10 seeds' dates (`REF`, as 3b). Description throughout: model (12 × 768, 1/layer attention
scaling), data (OpenWebText) and tokenizer all differ. Both dates primary (98 (c)).

**The schedule is logarithmic, not linear (a correction to unit 1).** `lit-C2-unit1.md` §5 read
the warmup as "4000 steps, linear" from the trainer config. Every seed's logged LR
(`trainer_state.json` at `checkpoint-4100`) is 6e-4 · ln(n) / ln 4000 with n = t − K integer:
DeepSpeed's log warmup, with K = 15–18 early fp16-overflow skips (W2: every logged LR from 50 to
4000, all five seeds). At step 100 the LR is already 3.2e-4, half the peak. The batch is checked
too: each seed's epoch count at 4100 gives 524 989 tokens a step against 512 × 1024 = 524 288
(W2). So the dose schedules predict GPT-2's dates far earlier than step-locking does, and 3c can
separate step from cumulative LR, which 3b could not.

| item | fixed as | why |
|---|---|---|
| the schedules (`to_gpt2`) | **step**: t + K (equal updates applied); **cumulative**: equal summed LR (410m's p t² / 2W against GPT-2's p (n ln n − n + 1) / ln W); **instantaneous**: equal LR level (n = 4000^(t / 2860)); **fraction**: K + t · 4000 / 1430; **tokens**: 4 t | lit §5's four, plus instantaneous so 3c is read as 3b; W1, W3 |
| 410m's dates (`REF`) | E₁'s break {256, 512}, centred {512, 1000}: 3b's union over 410m's 10 seeds and unit 3s's arms | the conservative band; 70m's and 160m's dates (3b) lie inside it |
| a prediction | a GPT-2 grid date s stands for a crossing in (prev(s), s]; the GPT-2 grid steps whose interval meets a mapped 410m interval, the union over K = 15–18 (`predicted`) | where the skips fell in steps 1–50 is unknown; W5 checks it against a scan |
| the grid | 0, 10, 20, 40–100 by 20, 150–600 by 50 (no 300), 700–1200 by 100, 1400–2000 by 200, 2400–4000 by 400, 6000, 8000, 12000, 16000; tags present on all five seeds (expanse lacks 30 and 300) | 50-step resolution where the schedules' bands meet (outcome table); 46 GB |
| the plateau | median V over the grid steps in [4000, 16000] | 410m's 1000–4000 under tokens, the latest map; past every schedule's predicted date. Beside: the half-rise on [1000, 4000] (410m's window under step) |
| unreadable | the plateau is 0, or V at step 0 or 10 is ≥ half of it | 92 (i)'s early check on the steps no schedule reaches (the earliest predicted crossing is at 16) |
| per seed, per count | `half_rise` (D4 (a) on this grid); beside, the first step from which V stays ≥ half to 4000 (`sustained`) | one crossing on a dense grid can be a spike; the beside says so, it does not decide |
| the centred arms | unit 3b's procedure on `alias`: null-matched and B-matched derived on 3 re-inits (HF's GPT-2 init, float16-rounded; their weight SDs set beside alias's step 0) and the off-grid step 20000, beside 1e-4 and 3e-4; each arm gated as 3b (re-init rate ≤ 5 %, plant B ≥ 0.4 at the smallest dose the uncentred count sees at 0.4, plant A at 0.5 ≤ 0.1); frozen into `TAU_ARMS`, the same arms on all five seeds | 3b's procedure transfers, not its numbers; one seed's preflight as 3s's |
| the uncentred count | D2's 1e-3, as A5 and 3b; its preflight report cited | refuse rather than degrade on a new model |
| the centred date | the dates of the gated, readable arms; a schedule is consistent if it predicts any of them | as 3b |
| **the reading** (`joint`) | the schedules consistent with every readable date (5 seeds × 2 counts); all five → **"not told apart"**; none → **"neither"**; else **"consistent with …"**. Beside it, `with_3b`: the same less instantaneous (3b excluded it; on Pythia's sizes step, fraction and tokens are one prediction, 3b's "step") | one statistic, fixed before output |

**What 3c can and cannot tell apart (the check's outcome table, fixed now).** Each GPT-2 date
→ the schedules consistent with it (others: none):

| count | dates |
|---|---|
| E₁'s break | 20 inst.; 40 inst., cum.; 60–100 cum.; 150 cum., step; 200–350 step; 400–500 step, frac.; 550 step, frac., tokens; 600–1600 frac., tokens; 1800–2400 tokens; else none |
| centred | 20–40 inst.; 60–250 cum.; 350 cum., step; 400–700 step; 800–1000 step, frac.; 1100 step, frac., tokens; 1200–3200 frac., tokens; 3600–4000 tokens; else none |

So 3c separates step from cumulative LR (E₁ at 200–350 or ≤ 100; centred at 400–700 or ≤ 250),
and both from {fraction, tokens} (E₁ ≥ 600, centred ≥ 1200). It cannot split fraction from tokens
(their bands overlap at 600–1600 and 1200–3200). With one seed's date outside every band the joint
reads "neither": 5 seeds make that more likely than one; each seed's reading is reported.

**What each outcome changes (fixed now).** "Consistent with step" (alone, or with fraction from
E₁ at 400–500): §7 says the dates track the updates applied across a 4× change in tokens a step
and a log warmup; F1 narrows to "a count of updates, not the LR's dose or shape, nor the tokens
seen", description across models. Cumulative without step: the dates track the LR dose; F1
narrows to "a dose". Fraction and / or tokens without step or cumulative: the warmup's timing or
the data seen; F1 narrows to that pair. "Neither": the timing does not carry over to GPT-2 small
under any of the five; F1 stands, the model and data named. "Not told apart" or unreadable: F1
stands as written. Combined with 3b, `with_3b` is quoted.

**The preflight** (`data/trackC/c2u3c/2026-10-22-preflight/`, main tree; code `15217ea3e56d`
with `TAU_ARMS` unset; the frozen code refuses unless a re-run derives the same arms). On alias:
3 re-inits (weight SDs 0.997–1.003 of step 0's, every matrix) and step 20000; 252 re-init and 84
trained units. Plant B's dose is 0.5 (the uncentred count turns 0.67 at it; 0.25 does not reach
0.4), as 70m's.

| arm | re-init centred rate | B share | A at 0.5 | trained base rate | gate |
|---|---|---|---|---|---|
| 2.45e-3 null-matched | 0.004 | 0.83 | 0.09 | 0.36 | pass |
| 1e-4 | 0.032 | 1.00 | 0 | 0.64 | pass |
| 3e-4 | 0.020 | 1.00 | 0 | 0.63 | pass |
| 3.42e-3 B-matched | 0 | 0.66 | 0.02 | 0.27 | pass |

Uncentred at 1e-3: re-init rate 0, trained base rate 0.50, B at 0.5 0.67, A at 0.5 0.98 (the
common direction is what it sees, as on 410m). The null-matched arm sits 25× above Pythia's
(7.55e-5 to 1.25e-4) because the re-inits' uncentred rate at 1e-3 is 0 here, so matching it
asks for a centred rate near 0; it is gated like the others. Reports
(`tools/design_preflight.py`): centred `cfefbce5a851…`, uncentred `4645b68b24ac…` (inputs
`1794a1d17f12…`, `93bdc8651099…`; `preflight_plants.json` `9e39d44b213e…`). The one refusal,
argued: `floor` on both (E ≤ 7.95 < 20): C2 computes no E, and the reading is a date against a
tolerance, as in units 3, 3s and 3b. No `band` or `power` refusal (the uncentred count's B at 0.5
reaches p ≤ 0.05 against the re-inits in 83 % of draws).

**No lean.** No record of any Mistral checkpoint exists in the repo (searched: no code names
`stanford-crfm` or Mistral); before the freeze nothing of GPT-2 small was read but the logged LR,
the configs, alias's step-0 weight SDs and the preflight's re-inits and step 20000.

## Unit 5b: the referee's readings from records (frozen 2026-10-23, before any 5b output)

**Decided (a), 2026-10-23, the user's** (Blocked 102; `referee-C2.md` "Blocked 102"): one more
unit, readings from records on disk, then the close-out. **The budget is extended on the record
to 9 units + close-out.** (b), the runs, and (c), the schedule intervention, go to ROADMAP rows,
(c) the first post-C2 candidate.

**Inputs.** Seed 0: unit 3's records (`trackC/c2u3/2026-10-22-run/records`); seeds 1–9: unit
3s's (`trackC/c2u3s/2026-10-22-run/seed<s>/records`); each holds, per step and prompt, `e1` and
`cos` at the 25 states (`n` tokens). ‖m‖² = (1 + (n − 1) cos) / n exactly, and E₁ = ½ + ½‖m‖² +
R with R ≥ 0, since each higher term is a squared norm, Σᵢⱼ⟨xᵢ,xⱼ⟩ᵏ = ‖Σᵢ xᵢ^⊗k‖²
(`tools/math_checks/c2_energy_moments.py` M4, M5). **Refuse** unless each seed's V from these
records equals A5's (`score.json` for 1–9, seed 0's gate record) at every step. Nothing is run
on a model; no download. `trackC_writeup/c2_readings.py`, committed with this section before its
first run.

A *drop* at transition t (state t − 1 → t, t = 1..24) is D2 (a)'s: E₁ falls by more than 10⁻³ of
its value. The *frame* transitions are t = 1 (embeddings → block 0) and t = 24 (block 23 → final
LN); the *inner* ones are 2..23. Dates are D4 (a)'s half-rise (`a3.d4a`) unless stated; the
window is D5 (a)'s, {256, 512}.

| # | referee | reading (all description: nothing registered, no label, no threshold) |
|---|---|---|
| R1 | M8 | **per-prompt V**: V_p by step for the 70 (seed, prompt) series, each dated by the half-rise on its own 1000–4000 plateau ("no change" if that is 0). Reported: how many of the 70 are dated in the window, the dates' spread, and at each seed's break the largest share of V one prompt holds |
| R2 | M8 | **the layer × step map**: drops at each t and step summed over 7 prompts and 10 seeds (at most 70), drawn as figure 5's left panel; reported, at 128, 256, 512 and 1000, the median t of the drops (pooled) and the three t holding most |
| R3 | M8 | **without the frame transitions**: V_in over t = 2..23, dated per seed; reported, the dates, how many of 10 are in the window, and the frame transitions' share of V at 256, 512, 1000 and 143000 (pooled) |
| R4 | M8 | **E₁'s net change**: (E₁ at state 24 − at state 0) / at state 0 per series; reported, at each step, how many of the 70 are > 0 and the median; the inner change (state 23 against 1) beside |
| R5 | m27 | **a later plateau**: the half-rise on the median of V over 8000, 16000 and 32000 (the referee's 8k–32k); reported, each seed's date and plateau, and how many of 10 are in the window |
| R6 | M1, M2 | **the mean's term at each drop**: for every drop, ΔE₁ = ½Δ‖m‖² + ΔR. Reported at each step, pooled over seeds and prompts: (a) the share of drops at which ‖m‖² falls, beside the share of *all* transitions at which it falls; (b) the share of drops with s = ½Δ‖m‖² / ΔE₁ > ½ (the mean's term carries most of that drop) and the median s; (c) V_R, the count of transitions at which R alone falls by more than 10⁻³ of E₁ (the drop E₁ would show with ‖m‖² held still), dated per seed. Per seed, (a) and (b) at the seed's own break step |
| R7 | M8 | **E₁ by depth**: the prompt-mean E₁ at the 25 states at steps 128, 512, 1000 and 143000, seed 0 drawn and the other nine as a range; figure 5's right panel; no number reported from it |
| R8 | m16 | **figure 4's right panel normalised**: each GPT-2 small seed's V over its frozen plateau (3c's record), the half line, the centred V at each of 3c's four arms (the median over seeds of V_c over its own plateau), and the logged LR over its peak; no number |

**What it can change.** §3.2's, §3.3's and §4.1's wording (the "not reported" and "not
measured" sentences become numbers), a new figure 5, figure 4's right panel. **Not** any date
already read: R3's, R5's and R6 (c)'s dates sit beside the registered break, never in its place.
V_R is E₁ without its first-order term, not a mean-free energy (R's k ≥ 2 terms still carry the
mean); the centred count (unit 3s) is the mean-free one.

**Exit:** every reading above in a record (`data/trackC/c2u5b/2026-10-23/readings.json`), quoted
through `numbers.py` keys, the draft and figures re-rendered, `check` passing.

**Added after `/challenge-pr` on #261 (post hoc, after R6's output).** R6 (a)'s comparison, the
share of *all* transitions at which ‖m‖² falls, is the wrong base rate (most transitions are
rises of E₁); R6′ adds, per step, the share of transitions at which E₁ and ‖m‖² change in the
same direction, the share of any fall of E₁ at which ‖m‖² falls, and s > ½ over all moves and
over rises. R6 (c)'s V_R stays in the record but out of the draft: its tolerance is 10⁻³ of E₁,
not of R, which tightens on R's scale along training.

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
