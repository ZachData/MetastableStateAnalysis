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
  - Unit 3s, the centred count on the 9 PolyPythias seeds (tier 1, description; Blocked 96 (b)): one centred tolerance on the centred energy's own scale failed its own gate, because no tolerance reads a planted change alike through both counts, so no size comparison between them is calibrated; the rule reads dates only at four centred tolerances (Blocked 97 (a)). The centred date reproduces: the centred half-rise is at 1000, one grid step after the uncentred break, on 8 of 9 seeds (K1 and K2 are one reading on this grid; seed 6 ties on one arm), so F9 is a finding on the seeds as a date (how much of the break is the common direction's is not calibrated), and the centred break shares 1000 with D10's heads; revised after `/challenge-pr` on #256 (findings 1–4); 91 goes to the user, decided (c) (Blocked 98): the draft heads on the two events — `trackC_writeup/status-C.md` "C2 unit 3s"
  - Unit 3b, the LR-dose arm on `pythia-70m` and `pythia-160m` (tier 1, description; Blocked 99 (a), 100 (a)): both dates on both sizes, against three schedules' predictions on Pythia's grid. E₁'s break at 256 and the centred date at 1000 on both: **consistent with step and cumulative** (the same without 70m's fragile E₁ date), an LR level excluded on one seed per size; step against cumulative is not told apart on this grid, and size is confounded. F1 stands — `trackC_writeup/status-C.md` "C2 unit 3b"
  - Unit 3c, the warmup arm on Mistral GPT-2 small's 5 seeds (tier 1, description): its warmup is logarithmic, not linear (unit 1 corrected before the freeze), so the LR dose predicts far earlier dates than step. E₁'s break falls at 250 or 350 on all 5 seeds (step's band), and the frozen joint reads **"consistent with step"**, but it turns on two choices: the plateau window (against 1000–4000, "neither") and, on 4 of 5 seeds, a centred arm whose tolerance is one re-init unit (at the next, "neither"; `/challenge-pr` on #258). The centred arms disagree from 100 to 2000, so no centred date; E₁ rises in two stages (a first at 60–100, in the dose's band) that neither schedule maps whole. Blocked 101 decided (a), the user's: F1 stands — `trackC_writeup/status-C.md` "C2 unit 3c"
  - Unit 4, the rest of the draft (writing only): the draft's introduction, its centred-cloud section, the OV and schedule sections, appendices A and B, figure 4 and the references written; the break and timeline sections re-headed on the two events (98 (c)); 20 records, 200 keys, 40 sentence claims checked; the unit-3s span "0.17–0.50" corrected to 0.07–0.50 from the record; no reading moved; `/challenge-pr` on #259: accept with changes, all 7 findings taken (a wrong-record key, §1's 3b exclusion, sizes compared across counts, the frozen-rule departure disclosed) — `trackC_writeup/status-C.md` "C2 unit 4"
  - Unit 5, the referee read (writing only): a fresh agent given only the draft recommended major revision (9 major, 27 minor points); every point answered (`referee-C2.md`): the draft gains an abstract, a title that says what is found, E₁'s exact series (its first-order term is ½‖m‖², the anisotropy; sympy check), the centred section's conclusion as a date statement, the introduction's order language matched to the timeline's, the band rule, (b)'s construction, arm (a)'s step-0 and 256 cells, ten references; after `/challenge-pr` on #260 the ‖m‖² profile (read post hoc at 3s, unsaved) reported from a record (never falls at 128, 6–10 of 24 transitions at 512 on every seed) and the registration's outside timestamp quoted (GitHub's branch creation, 60 s before the first record); 207 keys, 49 claims; no reading moved. The readings and runs it asks for are Blocked 102 — `trackC_writeup/status-C.md` "C2 unit 5"
  - Unit 5b, the referee's readings from records (Blocked 102 (a), the user's; budget extended to 9 units + close-out; no forward pass; description): at every one of the 1224 E₁ drops from step 128 to 1000, on every seed, ‖m‖² falls, and its term ½Δ‖m‖² is more than half the drop at all but one, so at the break V counts the tokens' mean shrinking between layers (exact split, not a separation: R's higher terms carry the mean too). No date moves: the frame-changing transitions hold none of the drops at 256–1000, a later plateau moves only seed 9 (256 → 512), 65 of 70 per-prompt series break in the window; E₁ rises first-to-last on every series; figure 5 and appendix C; (b), (c) to ROADMAP C5, C4 — `trackC_writeup/status-C.md` "C2 unit 5b"
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@2db9b14ba1, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the dates are fixed in steps (or tokens) or wait for a cumulative LR dose (F1, standing by Blocked 101 (a)); 3b could not separate them on Pythia's grid; 3c, on GPT-2 small, reads step for E₁'s larger stage, a first stage in the dose's band, no centred date, and a frozen label that turns on the window and one re-init unit (description only); the run that could separate them, continued training from a pre-break checkpoint with a flat rate, is not in C2 — `trackC_writeup/status-C.md` "C2 unit 3c"
  - Why ‖m‖² falls mid-stack at the break (massive activations, sinks), E_β at other β, component transplants, the small sizes' seeds, a sampled prompt set (ROADMAP C5) and the schedule intervention (C4, the first post-C2 candidate): runs, not in C2 — `trackC_writeup/status-C.md` "C2 unit 5b"
- **After Phase 10:**
  - none
- **Reviewed:** 2026-10-23 · body `ff8d5b53d3`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: frozen, Blocked 91–94 decided as recommended (left to Claude)** | `design-C2.md` "Decided 2026-10-22" |
| C2 1. reading (writing only) | **done 2026-10-22: F5 worded, F9 narrowed, 92 (ii) feasible (GPT-2 small), §2 and §8 drafted; revised after `/challenge-pr` on #253** | `lit-C2-unit1.md`, `draft-C2.md`; "C2 unit 1" below |
| C2 2. `numbers.py`, figures, §3–§5 | **done 2026-10-22: 102 keys from 13 records, 24 sentence claims checked, §3–§5 render, figures 1–3 drawn; revised after `/challenge-pr` on #254** | `numbers.py`, `numbers.json`, `draft-C2.src.md` → `draft-C2.md`, `fig/`; "C2 unit 2" below |
| C2 3. anisotropy control | **done 2026-10-22: frozen before output (95 (a), the user's), preflight cited; reads "mostly removed", the centred date 1000; Blocked 96** | `c2_aniso.py`, design "Unit 3"; "C2 unit 3" below |
| C2 3s. the centred count on the seeds (96 (b)) | **done 2026-10-22: 96 (b)'s tolerance refused by its gate; dates at four arms (97 (a), the user's); K1 = K2 = 8 of 9, "reproduced"; Blocked 98** | `c2_seeds.py`, design "Unit 3s"; "C2 unit 3s" below |
| C2 3b. LR-dose arm (70m, 160m) | **done 2026-10-22: 98 (c), 99 (a), 100 (a), the user's; frozen before output; "consistent with step and cumulative" (instantaneous excluded); F1 stands** | `c2_dose.py`, design "Unit 3b"; "C2 unit 3b" below |
| C2 3c. warmup arm (Mistral GPT-2 small) | **done 2026-10-22: frozen before output (`0623dd9`); the warmup is logarithmic (unit 1 corrected); the frozen joint "consistent with step", turning on the plateau window and a one-unit centred arm; no centred date; revised after `/challenge-pr` on #258; Blocked 101** | `c2_warmup.py`, design "Unit 3c"; "C2 unit 3c" below |
| C2 4. the rest of the draft | **done 2026-10-22: every section written; 20 records, 200 keys, 40 claims; revised after `/challenge-pr` on #259; figure 4; references from arXiv; Blocked 101 (a), the user's** | `draft-C2.src.md` → `draft-C2.md`, `numbers.py`, `fig/fig4_schedules.svg`; "C2 unit 4" below |
| C2 5. the referee read | **done 2026-10-23: a fresh agent, the draft only: major revision, 9 major and 27 minor points, every one answered; the draft revised (abstract, title, E₁'s series, §4.1 as dates, ten references); Blocked 102** | `referee-C2.md`; "C2 unit 5" below |
| C2 5b. the referee's readings from records (102 (a)) | **done 2026-10-23: frozen before output (`9629d78`); no forward pass; R1–R8 description; ‖m‖² falls at every E₁ drop from 128 to 1000 and its term is most of the drop at all but 1 of 1224; the frame transitions, a later plateau and per-prompt counts move no date; figure 5, figure 4's right panel normalised** | `c2_readings.py`, design "Unit 5b"; "C2 unit 5b" below |
| C2 close-out | next | — |

## C2 unit 5b, the referee's readings from records (2026-10-23)

Blocked 102 decided (a), the user's: one more unit, readings from records already on disk, then
the close-out; **the budget is extended on the record to 9 units + close-out**; (b) and (c) are
ROADMAP rows C5 and C4 (C4 the first post-C2 candidate). The rules (R1–R8, all description) were
frozen in `design-C2.md` "Unit 5b" and committed with the code (`9629d78`) before the first run.

**Input.** Unit 3's (seed 0) and unit 3s's (seeds 1–9) records, `e1` and `cos` at the 25 states
per step and prompt; each seed's V from them equals A5's at every step (the refusal check). E₁ =
½ + ½‖m‖² + R with R ≥ 0 (math check M4, M5, 7 of 7; the code refuses on any R < −10⁻⁹, none
found). Code `c2_readings.py` (record code hash in `readings.json`); record
`data/trackC/c2u5b/2026-10-23/readings.json`.

| # | reading | result (keys `rd.*` in `numbers.json`) |
|---|---|---|
| R1 | per-prompt V, each dated on its own plateau | 65 of 70 series in the window (32 at 256, 33 at 512); the other 5 are one prompt (Camus) at 1000 on five seeds; at each seed's break one prompt holds at most 0.17–0.24 of V (even split 0.14) |
| R2 | the layer × step map | the median drop at transition 12 (256), 11 (512), 9 (1000); most at 10–13 at 256 and 512, at 5, 7, 10 at 1000; late in training a band at 20–24 (figure 5) |
| R3 | without t = 1, 24 | the frame transitions hold 0 drops at 256, 512, 1000 and 0.05 at 143000; V over 2..23 dates every seed exactly as V (10 of 10 in the window) |
| R4 | net change, state 0 → 24 | > 0 on all 70 series at every step (median 1.02 at 128, 0.54 at 512, 0.36 at 1000, 0.41 at 143000); state 1 → 23 on all 70 before 8000, 39–63 of 70 from 8000 |
| R5 | plateau at 8000–32000 | plateaus 59–74; every seed in the window; only seed 9 moves, 256 → 512 |
| R6 (a) | ‖m‖² at the drops | falls at every drop at 128, 256, 512, 1000 (1224 drops); at all transitions it falls at 0.20–0.35 (256–1000); later steps ≥ 0.98 of drops |
| R6 (b) | s = ½Δ‖m‖² / ΔE₁ | > ½ at all but 1 of the 1224 drops at 128–1000 (median 0.81–1.02 by step at 256–1000; above 1 means R rose); later 0.91–0.99 of drops |
| R6 (c) | V_R, drops of R alone | half-rise at 512 on all 10 seeds; on seeds 3 and 9 (V's break at 256) V_R is 2 and 1 there |
| R7 | E₁ by depth at 128, 512, 1000, 143000 | figure 5, right (no number) |
| R8 | figure 4's right panel | normalised per seed, half line, centred per arm (seed median), LR / peak |

**What it changes in the draft.** §3.2: the frame transitions and per-prompt sentences are
numbers; §3.3: the later plateau is read; §4.1: "how much of V it carries is not measured"
replaced by R6; the abstract and §1 item 2 say the mean's term falls at every drop and is most of
it; a new appendix C with figure 5; figure 4's caption. 23 records, 241 keys, 56 claims (7 new).
**No date already read moved.**

**What it means.** The referee's M1 was right in substance: at the break, V counts the tokens'
mean shrinking between layers. Every drop V counts at 128–1000 is one at which ‖m‖² falls, and
the mean's term is more than half of it. The split is exact but not a separation: R's higher
terms carry the mean too (R is not mean-free; the centred count is), so V_R breaking at 512 does
not say a second, mean-free break exists there. With unit 3s's centred date (1000), the reading
is: the break at 256–512 is the mean's norm starting to fall mid-stack; the tokens' arrangement
around the mean starts to fall one grid step later.

**Not established.** Why ‖m‖² falls mid-stack (massive activations, sinks: C5's projection is the
run that could say); whether V_R's 512 is the mean through R or something else; anything on a
prompt set not chosen by hand.

**Parked:** none new. Still carried: 3c's Parked 0 and `lit-C2-unit1.md` Parked 1, to the
close-out.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_readings`, then
`python -m trackC_writeup.numbers produce`, `render`, `figures`, `check`;
`python3 tools/math_checks/c2_energy_moments.py`.

## C2 unit 5, the referee read (2026-10-23)

Writing only; no run. The design's unit 5 is a fresh-context read of the whole draft as a
reviewer would, answered point by point, then `/challenge-pr` on the unit's PR.

**The read.** A fresh general-purpose agent (no conversation history) was given only
`draft-C2.md` at `9840633` and its four figures, with a referee's brief and none of the
project's reasoning. Report and answers: `trackC_writeup/referee-C2.md`. Recommendation **major
revision**; the two reasons it gives: by the paper's own account the break is neither separated
from the warmup nor attributed, and at β = 1 E₁ is close to a first-moment anisotropy statistic,
so the framing oversells; and the experiments that would settle it are feasible and absent.

**What changed in the draft** (each in the answers table with its point):

| where | change |
|---|---|
| title, abstract | added; the title says what is found |
| §2.1 | E₁ = ½ + ½‖m‖² + Σ_{k≥2}(1/2n²k!)Σᵢⱼ⟨xᵢ,xⱼ⟩ᵏ, exact, the higher terms not small (`tools/math_checks/c2_energy_moments.py`, 4 of 4; it does not prove which term moves at the break) |
| §1, §4.1, §8 | the centred result as a date statement ("not seen once the mean is removed"; "does not say the break is anisotropy"); §8: the statistic is not new in kind, the prediction is what the picture adds |
| §1 item 3 | one-step gaps "aligned, not ordered", as §5 counts them |
| §3.4 | the window chosen after seed 0, its step at the right edge; the falsifier's level 0.048 (key `reg.falsifier_p`); times in UTC; the expected outcome stated |
| §4, §5 | E₃.₅ differs from E₁ on seeds 3 and 9; the 11 pairs defined; no null for the aligned count; E5's climb back past its start (512–1000) and E3's fall after 1000 described |
| §6 | (b)'s Schur construction; the controls' double matching; arm (a)'s step-0 (11, inside 0–17) and 256 (30, above 9–17) cells; G3 as a reach check; MLPs and other components never moved |
| §7 | the band rule as `c2_dose.predicted` computes it; each band's model named; GPT-2's spacing and LR source; the tokens mapping |
| figures | figure 1's legend names the seven grey seeds; figure 3's third panel retitled; captions for 2–4 say what is and is not drawn (unit 4's Parked 1, 70m's plateau of 6, closed in figure 4's caption) |
| §8, references | ten papers the referee named (massive activations, sinks, anisotropy, rank collapse, warmup, sudden drops, induction formation), titles from the arXiv API, cited for their abstracts (`lit-C2-unit1.md` §7) |

`numbers.py`: one key (`reg.falsifier_p`), formats `utc`, `v`, `span`, five claims (E₃.₅ against
E₁ on the 256 seeds; E5's reversal; E3's peak; which lanes run below figure 2's cut; arm (a)'s two
cells): 201 keys, 46 claims; `produce` needed no `--reread`.

**After `/challenge-pr` on #260** (accept with changes, all 5 findings taken; table in
`referee-C2.md` "After `/challenge-pr` on #260"): §1's "later" squared with "aligned" (the two
counts are one statistic, read by grid order under a rule fixed before the run); **the ‖m‖²
profile, read post hoc at units 3 and 3s but never saved, reported in §4.1 from a new record**
(`c2_mean_profile.py`, no forward pass: never falls at 128 on any seed, 6–10 of 24 transitions at
512 on every seed, the 256 seeds most at 256; the frame transitions hold no fall at 128–1000, some
later); §3.4's registration order rests on GitHub's branch creation at 18:38:17Z carrying the
commit, 60 s before the first record (copied to `p_a5_push_record.json`; the reviewer had found
only the 18:45 push); §4.1's centring order corrected; §8's novelty sentence made one claim. 207
keys from 22 records, 49 claims.

**Not done here, and why:** the readings the referee asks for that the records allow (per-prompt
V, the layer × step map, the frame-changing transitions, the net change, a later plateau, the ‖m‖²
count on seed 0, figure 4's right panel normalised) and the runs it asks for (other β, the
massive coordinates removed, component transplants, the small sizes' seeds, a sampled prompt set,
the schedule intervention) are past C2's budget: **Blocked 102**, the user's. The registration
cannot be made public, nor the prompt texts given, without the release decision (Blocked 93).

**Not established here:** anything new; whether the readings of 102 (a) change §3.2's or §4.1's
wording.

**Parked:** none new. Unit 4's Parked 1 closed (figure 4's caption). Still carried: 3c's Parked 0
and `lit-C2-unit1.md` Parked 1, to the close-out.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`, then `render`,
`figures`, `check`; `python3 tools/math_checks/c2_energy_moments.py`.

## C2 unit 4, the rest of the draft (2026-10-22)

Writing only; no run. Blocked 101 decided (a) by the user at the start of the session: F1
stands, and §7 reports 3c's frozen label with both fragilities and "neither schedule maps the
whole curve". The design's exit asks for every section written; it is.

**What was written** (`draft-C2.src.md`, rendered to `draft-C2.md`):

| § | content | keys from |
|---|---|---|
| 1 | the picture, the four contributions (the dated break, the centred cloud's later break, the timeline with F2, S2's negative with F3), F1 on the first page, the scope | A5, 3s, A3, S2, the configs |
| 4.1 | the centred cloud (98 (c)): the four tolerances, seed 0, the seeds' tally, the post-hoc fractions at the break, the size label "mostly removed" withdrawn (matched ratio 0.96) | 3, 3s |
| 5 | E1c added to table 1 and figure 2 (a lane from unit 3's V_c at 1e-3), joining the 1000 tie; marked as added after A3's rows were read and kept out of the pair count | 3, 3s |
| 6 | arms a and b against their 40 controls, the edit's size, copying under b against G2, G3′ lower at 512 / 1000 and higher at 2000, G3 reaching 0; F3 | S2's readings and gates |
| 7 | the three kinds of run, the schedules' bands, 3b's reading, 3c's three reasons it does not separate (two stages and the window; no centred date, step kept through one arm; neither schedule maps the curve); F1 stands | 3b, 3c, GPT-2's logged LR |
| A | S1's ρ: the null to 128, who leaves at 256 and 512, OV write then QK most concentrated, not top variance, the massive coordinates; 2603.13314 stated against it | S1's two ρ records |
| B | each gate: A5's (known answer, planted breaks, the null's K, the planted effect), A3's locator, S2's G1–G3 and the guard made description, the centred counts' arms and their failed size gate | A5, A3, S2, 3–3c |
| References | 27 entries, titles and authors from each arXiv page (fetched 2026-10-22); Martin and Mahoney added for E8; the Mistral checkpoints by repo | — |

**The render step.** 7 records added (`c3`, `c3s`, `c3b`, `c3c`, two S1 ρ records, the tracked
GPT-2 LR log), 95 keys, 15 sentence claims (39 in all), formats `ands` and `or`, figure 2's E1c
lane and figure 4. `produce` needed no `--reread` (no earlier key moved). Pytest 14 of 14
(`tests/test_trackC_numbers.py`), `check` matches.

**Choices made here** (each also in the PR's "Worth challenging"):

- *Citation years are the first arXiv version's* (Razzhigaev 2023, Cheng 2024, Bruno 2024,
  Rigollet 2025, Geshkovski 2024 for 2410.06833, Vovk and Wang 2019), since venue years cannot be
  read from the session; §8's years changed to match. The Mistral checkpoints are cited by repo
  with no year.
- *The size label withdrawn in the text.* §4.1 says unit 3's "mostly removed" measured centring's
  change of scale; the record still carries it, so a claim pins both the label and the matched
  ratio.
- *"The break is the common direction's, by the half-rise"* (§4.1, §1 item 2) is a date claim:
  the centred cloud has not broken at 256 or 512. How much of the count the common direction
  carries is said to be unmeasured.
- *Figure 4 drawn* (3c's Parked 2): it shows the two stages against both bands, which §7's first
  reason rests on.
- *"Step survives through the null-matched arm only" is computed* (`c3c.null_only`, 4 of 5): on
  alias the B-matched arm also dates in step's band.

**A correction found here.** The span "0.17–0.50 of its own plateau" (unit 3s's post-hoc row,
STATE, `PROJECT.md` §3.67) left out seed 4's B-matched arm (2 of 28): it is **0.07–0.50** over
the four arms, 0.19–0.41 at the null-matched arm. Corrected in place in all three; no reading
moves ("below half on every seed but the tie" holds); `LESSONS.md` 1.

**Carried items closed:** unit 2's Parked 1 (the references), unit 3's and 3s's Parked 1 and
3b's and 3c's (keys), 3c's Parked 2 (figure 4), `lit-C2-unit1.md` Parked 3 (the phrase "under
random weights too" is this project's GPT-2-era run record, `p1_mstate_tracking/design-1.md`
line 97 and `status-1.md` "Reverses", not Blog 1; §8 does not use it). Still carried: 3c's
Parked 0 (a refusal in `c2_seeds.null_matched_tol`, to the next unit that derives a centred
arm); `lit-C2-unit1.md` Parked 1 (GPT-2 medium, to the close-out).

**After `/challenge-pr` on #259** (accept with changes; all 7 findings taken):

| # | finding | done |
|---|---|---|
| 1 | tier 3 red: `tests/test_trackC_figures.py` listed three figures | figure 4 added to the list; tier 3 was not in the default `check.sh` run |
| 2 | §4.1's "V_c is 0 through 128" came from unit 3's one tolerance (10⁻³), not 3s's four (1–3 there at the small arms on seed 0) | key `c3s.s0_early` from 3s's record; the sentence reads "at most 3" |
| 3 | §1 said both 3b dates exclude the rate itself; E₁'s 256 lies in all three bands | §1 and §7 say only the centred date excludes it; a claim pins E₁'s 256 inside all three |
| 4 | §1 item 2 and §4.1 set 5–11 beside 56, a size comparison across counts | each count stated as a share of its own plateau (0.98 against 0.19–0.24), "only dates compared" said in §1 |
| 5 | §7 did not disclose that the frozen rule said "consistent with step" narrows F1, or when the decision was made | one sentence: the rule said so; decided after the run, before §7 was written |
| 6 | figure 2's E1c lane at 10⁻³, not one of the four tolerances; §5's tie paragraph left E1c out | the lane from 3s's seed-0 series at 10⁻⁴; E1c's share at 512 (0.19–0.24) placed between induction's and previous-token's, with a claim |
| 7 | the corrected span hand-typed in `status-C.md`, `PROJECT.md`, STATE (how "0.17" spread) | `PROJECT.md` and STATE point to the key `c3s.cfrac512`; `status-C.md`'s post-hoc row is the one typed copy, the rest point to it |

`numbers.py`: 200 keys, 40 claims; `c3.lane` removed (its record's tolerance was not one of
the four).

**Not established here:** anything new; the draft's argument has not had a fresh-context read
as a reviewer would (unit 5).

**Parked:** (1) Figure 4's left panel shows 70m's E₁ over a plateau of 6 drops, reaching 2.5;
whether to clip it or say so in the caption (why: legibility; cost: one line; could change:
nothing read). Carried to unit 5's referee read.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`, then `render` and
`figures`; `check` to verify.

## C2 unit 3c, the warmup arm (2026-10-22)

Blocked 92 (ii), read under 98 (c) (both dates primary). The rule is `design-C2.md` "Unit 3c",
frozen and pushed (`0623dd9`, `09af6bf`) before any scored record, after the preflight.

**Input:** Stanford CRFM Mistral GPT-2 small, 5 seeds (alias, battlestar, caprica, darkmatter,
expanse), at the 35 tags of `GRID` × the 7 v1 prompts (GPT-2's tokenizer, 512 tokens each),
float32, eager, TF32 off, RTX 3080; checkpoints fetched to `HDD_1TB/mets_data/hf`
(`pytorch_model.bin`, float16; 176 tags with alias's 20000), the run offline (17 min); code
`409833b1edf0` (`trackC_writeup/c2_warmup.py`) in every record. Records
`data/trackC/c2u3c/2026-10-22-run/<seed>/records/` (245 each, main tree); reading
`…/reading.json` sha256 `76eefb1f26ab…`. 410m enters as its 10 seeds' dates only (`REF`).
Preflight: design "Unit 3c". No known answer: no Mistral checkpoint had been read in the repo;
the code path is units 3, 3s and 3b's (`c2_aniso.states`, `state_row`, `c2_dose.record`).

**A correction to unit 1, found before the freeze.** The warmup is logarithmic (6e-4 · ln n /
ln 4000, n = t − 15…18), not linear as `lit-C2-unit1.md` §5 had it from the trainer config; the
LR is half its peak by step 100 (`tools/math_checks/c2_warmup_grid.py` W2, all five seeds'
logged LR). It moved the dose schedule's prediction for E₁ from about (303, 605] to (29, 119],
which is what lets 3c separate step from cumulative; annotated in place (§5).

| seed | E₁: V at 60, 100, 200, 250, 350 (plateau) | E₁'s break | centred dates (2.45e-3, 1e-4, 3e-4, 3.42e-3) | consistent |
|---|---|---|---|---|
| alias | 13, 17, 19, 31, 39 (41) | **250** | 350, 100, 100, 900 | E₁: step; centred: cum., step, frac. |
| battlestar | 16, 14, 19, 24, 38 (53) | **350** | 450, 150, 150, 1800 | E₁: step; centred: cum., step, frac., tokens |
| caprica | 13, 14, 20, 27, 31 (47) | **250** | 450, 150, 150, 1400 | E₁: step; centred: cum., step, frac., tokens |
| darkmatter | 6, 17, 18, 21, 34 (47) | **350** | 1000, 100, 100, 2000 | E₁: step; centred: cum., step, frac., tokens |
| expanse | 9, 17, 17, 22, 37 (48) | **350** | 900, 150, 150, 1600 | E₁: step; centred: cum., step, frac., tokens |

**The reading, by the frozen rule:** nothing unreadable (steps 0 and 10 at most 1 on E₁ and at
most 4 on any centred arm, against plateaus of 22–54; every arm gated). The schedules consistent
with all 10 dates: **step: "consistent with step"** (`with_3b`: step). E₁'s dates are step-only
on every seed; each seed's centred arms disagree (100–2000), and by the rule a disagreement
widens the date. **But step survives the centred count on battlestar, caprica, darkmatter and
expanse only through the null-matched arm** (`/challenge-pr` on #258, finding 1): the fixed arms
date in the cumulative band and the B-matched arm in fraction / tokens'. That arm's tolerance
(2.45e-3) is the single largest of the 252 re-init centred drops, 3.3× the next (7.32e-4),
because the re-inits' uncentred rate at 1e-3 is 0 (every re-init drop is negative), so matching
it asks for a centred rate of 0. At the next value the arm dates 150–350 and the joint reads
"neither". So the widening rule, which in 3b only kept schedules, here holds up the positive
label (finding 3). By the outcomes fixed in the design "consistent with step" narrows F1 for
E₁'s break; whether it should, given this and the window below, is **Blocked 101** (the user's;
recommended: F1 stands).

**The margin, stated beside the reading (post hoc, seen at reading).** E₁ rises in two stages on
every seed: to 14–17 by 100 (from 60–80, the cumulative band 40–150), then to 31–39 by 350
(step's band, 200–550); it then dips to 21–33 at 800–2000 and rises again to 41–53 by
8000–16000. The frozen plateau is that late level, so its half (20.5–26.5) falls between the two
stages, 1.5–3.5 units above V at the grid step before each date. Against 410m's window under step
(1000–4000, the beside fixed in the design) the half-rise reads 80 on alias, darkmatter and
expanse (cumulative) and 200 on battlestar and caprica (step), and the joint would read "neither".
So the step label holds for the second, larger stage and depends on the plateau window; the
first stage sits where the dose predicts. Neither schedule maps the whole curve (finding 2):
under step, 410m has nothing at GPT-2's first stage (seed 0's V is 0 through 128); under the
cumulative dose, GPT-2's step 100 maps to 410m's step 450, most of the way up 410m's rise (19 at
256, 56 at 512), where GPT-2 is at about 0.3 of its own plateau. Not a reading:
the frozen statistic is the joint at its window.

**Beside, not the reading:**

| what | reads |
|---|---|
| the centred arms split by tolerance | the two fixed arms (1e-4, 3e-4) jump at 100–150 on every seed (to 24–47, plateaus 47–54), in the cumulative band; the two derived arms (2.45e-3, 3.42e-3) rise slowly from 350 to 900–2000. On 410m (3s) and 70m / 160m (3b) the arms agreed at 1000; here they do not, so 3s's second event (the centred cloud breaking after E₁, as one date) is not reproduced on GPT-2 small |
| an early collapse at 20 | the layer-mean cosine jumps from 0.37–0.40 (steps 0, 10) to 0.88–0.90 at 20, the first steps with updates, and is back near 0.5 by 80; V is 0 on every count at 20 |
| sustained crossings | E₁: 250, 900, 250, 350, 350 (battlestar dips below half at 800); the fixed centred arms' equal their dates |

**What it changes.** §7 can say: on another suite with a log warmup 2.8× longer and 4× fewer
tokens a step, E₁'s larger rise falls where the count of updates predicts (250–350 against
410m's 256–512) on 5 of 5 seeds, after a smaller first rise where the LR dose predicts; neither
schedule maps the whole curve; the frozen joint's "step" turns on the plateau window and on one
re-init unit. The centred date does not transfer as one date. F1: Blocked 101.

**Not established:** that Pythia's break follows updates (3b could not separate step from
cumulative there); anything about the model, the data or the tokenizer; a centred date on GPT-2
small; whether E₁'s two stages are two events or one rise that Pythia's grid (nothing between
128 and 256) could not split.

**Parked:** (0) A refusal in `c2_seeds.null_matched_tol` when the null's uncentred rate is 0
(finding 1's ask): not made here, because `c2_seeds.py` is frozen code that 3s's and 3b's
preflights re-derive their arms with, and a refusal could stop their re-runs; carried to the next
unit that derives a centred arm, which adds it before its freeze (`LESSONS.md` 6). (1) `numbers.py` keys for this `reading.json`: unit 4, with 3b's Parked 2.
(2) Whether §7 shows the two-stage rise and the window margin as a figure (V by step, 5 seeds,
both windows marked): unit 4's call; cost a figure function in `numbers.py`; could change how
strongly §7 words F1's narrowing.

**Re-run:** `HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_warmup fetch` (about 25
min), `… preflight --out <dir>` (refuses unless it re-derives `TAU_ARMS`), then
`python -m tools.design_preflight <dir>/preflight_input_{c,u}.json --out …`; `… run --out <run>`
(17 min on the GPU, offline), then `read --out <run> --preflight <dir>`. Tests:
`tests/test_trackC_warmup.py` (pure, 12); the predictions: `python3
tools/math_checks/c2_warmup_grid.py` (W1–W5).

## C2 unit 3b, the LR-dose arm (2026-10-22)

Blocked 92 (iii), read under 98 (c) (both dates primary), 99 (a) (the labels re-cut by which
schedule predicts each date on the grid) and 100 (a) (plant B's gate dose per size), all the
user's, in session, before any scored record. The rule is in `design-C2.md` "Unit 3b", frozen and
pushed (`89022d1`) before the run.

**Input:** `pythia-70m` (peak LR 1e-3, 6 layers) and `pythia-160m` (6e-4, 12 layers), one seed
each, at A5's 18 grid steps × the 7 v1 prompts (`prompts_hash` `976b62d677bc`), float32, eager,
TF32 off, RTX 3080; checkpoints fetched to `HDD_1TB/mets_data/hf` (`model.safetensors`, 19
revisions per size with step 7000), the run offline; code `02a3cae5c04f`
(`trackC_writeup/c2_dose.py`). Records `data/trackC/c2u3b/2026-10-22-run/{70m,160m}/records/`
(126 each, main tree; about 3.5 min); reading `…/reading.json` sha256 `fd864233d350…`. 410m enters
as its 10 seeds' dates only (`REF`: E₁'s break {256, 512}, centred {512, 1000}). Preflight and its
v0: design "Unit 3b". No known answer: neither size had a record of E₁ by step before this one;
the code path is units 3 and 3s's (`c2_aniso.states`, `state_row`), whose known answers passed.

| size | units per step | V_u at 128, 256, 512, 1000 (plateau) | E₁'s break | V_c at 128, 256, 512, 1000, per arm (plateau) | centred date | consistent |
|---|---|---|---|---|---|---|
| 70m | 42 | 0, 3, 4, 8 (6) | **256** | 9.61e-5, 1e-4: 1, 1, 5, 14 (15); 3e-4: 1, 1, 5, 12 (15); 4.78e-3: 0, 0, 0, 3 (3) | **1000 ×4** | E₁: all three; centred: step, cumulative |
| 160m | 84 | 0, 18, 16, 16 (16) | **256** | 1.25e-4: 1, 1, 11, 22 (26); 1e-4: 1, 1, 11, 22 (27); 3e-4: 1, 1, 9, 17 (22); 8.22e-4: 0, 0, 5, 14 (19) | **1000 ×4** | E₁: all three; centred: step, cumulative |

**The reading, by the frozen rule:** nothing unreadable (steps 0–128 at most 1 on any count).
The schedules consistent with all four dates: **step and cumulative: "consistent with step and
cumulative"**; instantaneous is excluded by both sizes' centred dates (it predicts 128–512 on 70m
and 256–512 on 160m). By the outcomes fixed in the design: §7 says the dates do not follow the LR
level; **F1 stands**, the warmup's timing (or the tokens seen) and the cumulative dose not told
apart, as the design said 3b could not.

**Beside, not the reading:**

| what | reads |
|---|---|
| 3s's two events on the other sizes | on both sizes the centred date (1000) is later than E₁'s break (256), and at 1000: unit 3s's events 1 and 2 hold on 70m and 160m too, one seed each. At the break step the centred count is at most 1 (plateau 15–27) |
| E₁'s break at 256 on both | earlier than 8 of 410m's 10 seeds (512), in the dose's direction, but inside 410m's own seed range (seeds 3 and 9 at 256), so not a reading. One seed per size: their own spread is unmeasured |
| margins | 70m's E₁ break is a tie: V_u at 256 is 3, exactly half its plateau of 6. It is fragile both ways: one unit later it reads 512 ("step, cumulative"), but read against its later level (15 from 8000 on) it reads 1000, which no schedule predicts for 70m, and the joint would be "neither" (`/challenge-pr` on #257, finding 1). **The joint is the same with 70m's E₁ date removed** (160m's E₁ date and both centred dates give "step and cumulative"), so it rests on those three. The joint needs both sizes' centred dates to move before instantaneous survives: 70m is 2.5 units below half at 512 (5 against 7.5), 160m 2–4.5 units per arm |
| 70m's uncentred count | the weakest reading (the design tool's `power` refusal, argued): its plateau over 1000–4000 is 6 of 42 units, and it keeps rising after (15 at 8000–32000, 5 at 143000), so its "plateau" is a shelf |
| **post hoc (`/challenge-pr` on #257, finding 2)**: the two counts lean different ways | "step" survives the E₁ dates only through 410m's two seeds at 256 (3 and 9; 8 of 10 read 512), and "cumulative" survives 70m's centred date only if 410m's centred crossing is in (935, 1000] (160m's: (724, 1000]), the top of its interval. So the E₁ dates lean toward a dose and the centred dates toward step. On 410m seeds 3 and 9 (E₁ at 256, centred at 1000 at all four arms, unit 3s) step accounts for both counts on one seed, the pattern 70m and 160m show. Not a reading: the frozen statistic is the joint |
| the layer-mean cosine | on both sizes it rises to a peak at 32 (0.54, 0.67), falls to its lowest at 1000–2000 (0.20–0.23) and rises after; 410m's per-layer profile was not compared here |

**What it changes.** §7 can say: across a 3.3× range of peak LR at a shared warmup (one seed
per size, size confounded), the centred date stays at 1000 and E₁'s break at 256–512, so the
dates are not shown to wait for the LR to reach a level; whether they are fixed in steps (or tokens) or wait for a cumulative LR dose, Pythia's
grid cannot say (`tools/math_checks/c2_dose_grid.py`'s outcome table). Size is confounded
throughout. 3c, on GPT-2 small's finer grid and longer warmup, is the arm that can separate
step from cumulative.

**Not established:** step against cumulative; anything about size apart from the LR (6, 12, 24
layers move with it); the smaller sizes' seed spread (one seed each; PolyPythias' smaller sizes
are Parked 1); that 70m's uncentred count sees what 410m's does (its power refusal).

**Parked:** (1) PolyPythias' 70m and 160m seeds, if released (unverified): their own seed spread
would replace 410m's as each size's reference; cost a download of 9 × 18 checkpoints per size and
one run; could change whether a size's date sits inside its own spread, not this reading's
exclusion of instantaneous (which uses 410m's band, the conservative one). Carried to the close-out
for a decision. (2) `numbers.py` keys for this `reading.json`: unit 4, with unit 3s's Parked 1.

**Re-run:** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_dose
preflight --size {70m,160m} --out <dir>` (a few minutes each; refuses unless it re-derives
`TAU_ARMS`), then `python -m tools.design_preflight <dir>/<size>/preflight_input_{c,u}.json --out
…`; `… c2_dose run --out <run>` (about 3.5 min on the GPU, offline), then `read --out <run>
--preflight <dir>`. Tests: `tests/test_trackC_dose.py` (pure, 19); the predictions:
`python3 tools/math_checks/c2_dose_grid.py` (D1–D4).

## C2 unit 3s, the centred count on the seeds (2026-10-22)

Blocked 96 (b), the user's. The rule is in `design-C2.md` "Unit 3s", frozen and pushed (`98617c7`)
before any seed's centred record. 96 (b)'s single median-matched tolerance failed its own gate,
so it reads dates only, at four centred arms (Blocked 97 (a), the user's, in session).

**Input:** PolyPythias `pythia-410m-seed1`–`seed9` at A5's 18 grid steps (`HDD_1TB/mets_data/hf`,
offline; nothing downloaded) × the 7 v1 prompts (`prompts_hash` `976b62d677bc`), float32, eager,
TF32 off, RTX 3080; code `e7a1e159f247` (`trackC_writeup/c2_seeds.py`). Records
`data/trackC/c2u3s/2026-10-22-run/seed{1..9}/records/` (126 each, main tree; about 41 min);
reading `…/reading.json` sha256 `8d5ed734f665…`. Known answer: every record's e1 equals A5's
record for that seed and step (largest deviation 0.0), tokens identical, and each seed's V equals
A5's `score.json` V. Seed 0 is unit 3's records, read beside.

| seed | uncentred break (A5) | V_u at 512 | V_c at 512, per arm (7.55e-5, 1e-4, 3e-4, 7.58e-4) | V_c at 1000, per arm | centred date, per arm | event 1 | event 2 | D10's prev / ind |
|---|---|---|---|---|---|---|---|---|
| 0 (beside) | 512 | 56 | 11, 11, 10, 5 | 56, 55, 35, 19 | 1000 ×4 | yes | yes | — |
| 1 | 512 | 47 | 10, 10, 9, 7 | 36, 36, 32, 14 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 2 | 512 | 41 | 16, 16, 11, 5 | 35, 34, 26, 16 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 3 | 256 | 49 | 5, 5, 3, 2 | 57, 55, 46, 20 | 1000 ×4 | yes | yes | 4000 / not located |
| 4 | 512 | 58 | 9, 9, 7, 2 | 38, 38, 30, 19 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 5 | 512 | 55 | 14, 14, 8, 7 | 51, 48, 33, 16 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 6 | 512 | 53 | 14, 14, 11, 5 | 25, 24, 20, 14 | 1000, 1000, **512**, 1000 | no | no | 1000 / 1000 |
| 7 | 512 | 43 | 13, 12, 9, 6 | 42, 39, 25, 11 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 8 | 512 | 58 | 15, 13, 9, 7 | 49, 46, 33, 18 | 1000 ×4 | yes | yes | 1000 / 1000 |
| 9 | 256 | 52 | 20, 17, 10, 6 | 42, 42, 35, 18 | 1000 ×4 | yes | yes | 1000 / 1000 |

**The reading, by the frozen rule:** no seed unreadable at any arm (steps 0–128's centred counts
at most 4). **K1 = 8 of 9, "reproduced"** (the centred half-rise later than the uncentred break);
**K2 = 8 of 9, "reproduced"** (the centred half-rise at 1000). By the outcomes fixed in the design,
96 (c) is open on the seeds and the choice goes to the user: **Blocked 98**. *On this grid K1 and K2
are one reading, not two (`/challenge-pr` on #256, finding 2): every uncentred break is at 256 or
512, so a centred date at 1000 implies one later than the break. The two events would differ only
for a seed whose centred date was 2000 or 4000, and none was.*

**Beside, not the reading:**

| what | reads |
|---|---|
| seed 6 | its one miss is a tie on one arm: at 3e-4 the centred plateau is 22 and V_c at 512 is 11, exactly half, so the half-rise lands at 512. At the other three arms it reads 1000 |
| the size at 512 | across the 10 seeds and 4 arms V_c at 512 is 2–20 against V_u 41–58; at 1e-3 it is 1–7 at 512, 10–16 at 1000. That holds on every seed. It is not a ratio label: the gate showed the two counts have no common scale |
| seeds 3 and 9 | their uncentred break is at 256 (V_u 35, 29); their centred counts at 256 are 0–2 and their centred date is 1000 too, so the centred date does not move with the uncentred one |
| the centred date against D10 | on the 7 seeds where D10 dates both heads at 1000 and the rule reads 1000 (1, 2, 4, 5, 7, 8, 9), the centred date is the heads' grid step. Seed 3's heads are not both located. With no grid step between 512 and 1000, that is one grid step shared, not an order |
| the centred plateau's size | on its own count it falls with the arm (e.g. seed 0: 59, 55, 41, 22), as a larger tolerance should; the date does not move. At the B-matched arm the centred count keeps rising after 1000 (to 34–71 at 143000), so there the date is a half-rise on a rising curve, not a step onto a flat plateau (finding 2) |
| **post hoc (`/challenge-pr` on #256, findings 1, 3, 4)**: the centred count at the break, on its own scale | V_c at the seed's break is 0.07–0.50 of its own plateau on the 8 seeds that break at 512 across the four arms, 0.19–0.41 at the null-matched arm (5–16 units there) *(was "0.17–0.50", which left out seed 4's B-matched arm, 2 of 28; corrected at unit 4 from the record, `numbers.py` `c3s.cfrac512`)* and 0–0.04 on seeds 3 and 9 (at 256). Below half on every seed but the tie; well above the early-step floor (≤ 4). It is not "near the floor" |
| post hoc: the common direction by layer, per seed | the layer-mean cosine rises with depth at all 23 block transitions at 128 on all 10 seeds (none falling); at 512, 6–10 falling on every seed (seeds 3 and 9: 3 and 6 falling at 256, 9 and 8 at 512). So seed 0's turn of the cosine profile at the break is on every seed. It is E₁'s first-order term, so this is the same break decomposed, not separate evidence |
| post hoc: margins | per seed and arm, how far the half-rise is from moving: below half before 1000 by 0 (seed 6 at 3e-4, the tie), 2.0 (seed 9, null-matched), 2.5 (seeds 1, 6), 3.0–3.5 (seeds 2, 6, 7); above half at 1000 by 1.5 (seed 7, B-matched), 4.5 (seeds 1, 2, 5), then ≥ 5. All the near misses are at the outer arms, mostly the B-matched one. One unit's change flips seed 6 back, and about two flip seed 7 or 9. "Reproduced" (≥ 7) holds with 1 seed to spare |

**What it changes.** The centred date reproduces across the PolyPythias seeds: on 8 of 9 (9 of
10 with seed 0) at every arm, the token-centred count reaches half its own plateau at 1000, one
grid step after E₁'s break (512 on 7 seeds, 256 on 2), the step where D10 dates previous-token and
induction on 8 seeds. At the break step itself the centred count has not yet reached half of its
own plateau (the span is the key `c3s.cfrac512` in `numbers.json`; the row above, corrected at
unit 4), and the layer-mean cosine turns from rising with depth at the break
on every seed. F9 is a finding on the seeds, as a date: the break comes before the centred
cloud's. How *much* of the break is the common direction's is not calibrated (the gate). 91's
headline goes back to the user as Blocked 98: (a) or (c).

**Not established:** that the centred break at 1000 is a separate event rather than the same
change read later (no step between 512 and 1000; the same limit as unit 3); any size comparison
between the two counts (the gate showed none is calibrated); that the centred date follows
induction or previous-token rather than the LR half-point or anything else at 1000 (one grid
step; 3b and 3c are what could separate the schedule); anything on another model size.

**Parked:** (1) `numbers.py` keys for this `reading.json` (K1, K2, the per-seed dates) and the
centred row beside fig 1: unit 4, with unit 3's Parked 1 (cost: one render pass).

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_seeds preflight --out <dir>` (about
3 min; refuses unless it re-derives `TAU_ARMS`), then `python -m tools.design_preflight
<dir>/preflight_input.json --out <dir>/preflight_report.json`; `METS_DATA=<main>/data
HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c2_seeds run --out <dir>` (about 41 min on
the GPU, offline), then `read --out <dir>`. Tests: `tests/test_trackC_seeds.py` (pure, 13).

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
20 GPU minutes (seed 0's 18 steps took 125 s); it could change 96's wording ("reproduced" or
"seed 0 only"); an option in 96. *Carried to unit 3s (96 (b)). "After a re-download of about 160
checkpoints" was wrong: all 162 are in `HDD_1TB/mets_data/hf`, where A5 put them (`LESSONS.md` 1).*

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
