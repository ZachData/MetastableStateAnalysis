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
  - Unit 5b, the referee's readings from records (Blocked 102 (a), the user's; budget extended to 9 units + close-out; no forward pass; description): by E₁ = ½ + ½‖m‖² + R, E₁ and ‖m‖² change in the same direction at 0.96–1.00 of transitions at every step, so V at β = 1 is close to a count of the layers at which the tokens' mean norm falls, along all of training (post hoc, after `/challenge-pr` on #261 corrected the base rate); at each of the 1224 drops at 128–1000 ‖m‖² falls and its term is more than half of all but one (exact split, not a separation: R's higher terms carry the mean too). No date moves: the frame-changing transitions hold none of the drops at 256–1000, a later plateau moves only seed 9 (256 → 512), 65 of 70 per-prompt series break in the window; E₁ rises first-to-last on every series; figure 5 and appendix C; (b), (c) to ROADMAP C5, C4 — `trackC_writeup/status-C.md` "C2 unit 5b"
  - **C2 closed 2026-10-23 at its exit** (9 units + close-out, as extended by Blocked 102 (a)): a complete draft, every number rendered from a record, a fresh-context referee read answered. It establishes a dated, reproduced break in E₁'s monotonicity (the registered seed count), a second event after it in the token-centred cloud (at 1000), and that V at β = 1 nearly counts falls in the tokens' mean norm; it does not separate either date from the LR warmup. Every Parked item carried (C4, C5, C2p, the draft's posting thread) or dropped (GPT-2 medium, token-drift, an aligned-pair null); what happens to the draft was Blocked 103, decided (a), the user's: C5's M2 first, then C2p; nothing posted to arXiv by Claude — `trackC_writeup/status-C.md` "C2 on one page (closed 2026-10-23)"
  - C5 opened 2026-10-23 on M2 only (Blocked 103 (a); writing only): a scan and a proposed design that projects each seed's massive-activation coordinates (read at its own step 143000) out of every token and recounts both C2 counts against 20 random coordinate sets; the scan found a Pythia-410m anisotropy-type statistic dated "around step 1k" (2510.06477) that C2's §8 and F9 do not cite (carried to C2p); the set and scope are Blocked 104, the user's — `trackC_writeup/status-C.md` "C5 unit 0"
  - C5 unit 1, M2 (Blocked 104 (a), the user's; tier 1, description): each seed's massive-activation coordinates (2–3 per seed, ≥ 100× the median coordinate's max at some grid step) are ordinary at the break (no coordinate of any seed above 2.4× the median's at 128–1000) and reach 100× only at step 16000 (32000 on one seed), long after both events; they carry none of the falls V counts (their share above ½ at 0 transitions), so the mean whose falls V counts at the break is not theirs; projected out, the break's date is kept on 7 of 9 seeds (two cross the half line by one violation at one arm, as random sets also do) and the centred event's on 9 of 9, as expected when nothing large is removed; the lean from 2510.06477 ("around 1k") did not hold for this statistic; whether ‖m‖² sits in a few other coordinates is Blocked 105 — `trackC_writeup/status-C.md` "C5 unit 1"
  - C5 unit 2, M2's other half (Blocked 105 (a), the user's; budget 3 units + close-out): frozen before output; its gate, on seed 0's off-grid step 7000, refused the absolute "a few coordinates hold half a fall" (falls turned to random bases read "few" at 0.65, since coordinates move both ways); the rotation-relative reading passes both sides; Blocked 106 (a), the user's, made it the primary: on the ten seeds the falls V counts at the break are **no more concentrated in the model's coordinates than in a random basis** (9 of 9; post hoc, slightly less on every seed), no coordinate recurs, and the mean itself is spread there (half of ‖m‖² in 132–135 of 1024, a random basis 127–129), below a random basis from 4000 (104–113), in 6–24 coordinates at 8000 and 2–3 at 16000; a few coordinates hold half of each net fall in any basis (cancellation), so the result is about the basis, not a count — `trackC_writeup/status-C.md` "C5 unit 2"
  - **C5 closed on M2 2026-10-24 at its exit** (3 units + close-out, as extended by Blocked 105 (a); writing only): at the break's steps neither the massive-activation coordinates (ordinary there, 100× only from 16000, outside the falls) nor any recurring few carry the falls V counts, and the falls are not basis-aligned, against a gate that planted 3 coordinates (a set of 8–16 not tested; "not in a few coordinates" may not be said); projecting the massive ones out keeps both of C2's dates, a check rather than a test that could have moved (nothing large removed); a read-out, not an intervention. Whether the draft's section 4.1 leads with the level is Blocked 107, the user's. Parked items carried (when the mean becomes basis-aligned → ROADMAP C5; the late carriers' wording → C2p) or dropped (unit 1's set-aside pass); the draft's edits from M2 belong to C2p, next — `trackC_writeup/status-C.md` "C5 on M2, on one page (closed 2026-10-24)"
  - C2p opened 2026-10-24 at the user's request (2 units + close-out); unit 1: the draft converted to LaTeX for Overleaf (`trackC_writeup/tex/`, compiled, CI-checked against the Markdown) and ten figures to cull from, five new (the identity-weight picture, an illustration; the two events on the ten seeds; E₁ against its mean term; the massive coordinates' late arrival; the mean spread at the break), each but the illustration from a record; where the draft lives once the user edits it is Blocked 108 — `trackC_writeup/status-C.md` "C2p unit 1, the LaTeX and ten figures to cull from (2026-10-24)"
  - C2p unit 2 (107 (a), 108 (a), the user's; writing only): the draft's source is now `trackC_writeup/paper/` (the Overleaf folder; `paper_check` lists any number no record renders to, CI-checked); M2's results in the draft's section 4.1, level first; §8 and F9 meet 2510.06477, whose Pythia grid reads no step between 1 and 1000, so its "around 1k" bounds rather than dates (a correction to C5); m15, m19, m25 done; pdfLaTeX compiles (26 pages); a second fresh read (major revision, 6 major, 20 minor) answered in `referee-C2p.md`, its framing points Blocked 109 — `trackC_writeup/status-C.md` "C2p unit 2"
  - **C2p closed 2026-10-24 at its exit** (2 units + close-out; Blocked 109 (a), the user's; writing only): `paper/` is an Overleaf-ready draft (pdfLaTeX, 28 pages, `paper_check` 0), cut to seven figures in the main text and three in appendices (the illustration out), with the second read's framing points made: the title and abstract lead with the count, its drops at β = 1 being falls of the mean (one direction), §4 leads with a per-seed table and p and E follow it, out of the abstract, §5 states the alignments' base rate (anchored at the break too), a glossary and per-row definitions; revised after `/challenge-pr` on #269 (findings 1–5 taken); no number moved; no Parked items; m11, m12, M3 (iii), a code and data statement and posting are the user's — `trackC_writeup/status-C.md` "C2p on one page (closed 2026-10-24)"
  - After C2p (2026-10-24, the user's; writing only): `paper/` cites 2609.28448 in §6 (Sec. V B: a head's negative eigenvalues do not establish negative feedback) and §8, and §5 says the OV phase is named for E2's share; 28 pages, `paper_check` 0. 2026-10-25 (the user's): appendix D takes the timeline's rows one by one, each with its panel and its floor numbers; 34 pages. 2026-10-28 (the user's): figure 12, the energy ridgeline with the identity-weight reference, drawn from a new record, not yet in `paper/` — `trackC_writeup/status-C.md` "Units"
  - Figure 13 (2026-10-29, the user's; tier 1, description, no forward pass): effective rank, mass near 1, energy change and the Schur ratio by depth and step on seed 0, from `data/phase12`, drawn only after those runs reproduced A5's seed-0 energies (max relative difference 1.5e-6, tokens and V equal); `fig/paper/fig13_grid.pdf`, not yet in `paper/` — `trackC_writeup/status-C.md` "Figure 13, the layer × step small multiples (2026-10-29)"
  - C2p readings audit (2026-10-30, the user's; writing only): every claim `paper/` attributes to a paper checked against its source in `docs/readings/`; 6 corrected (Pendharkar's section 10 is about orthogonal subspaces, its Theorem 17 conditional; 2604.01978 proves ‖m‖² rises in depth under per-layer random attention weights, so the draft's "no result" is narrowed; Razzhigaev's anisotropy is centred; Ethayarajh's non-input layers; Gao's circle), 9 missed items, 8 added to the text; a released `pythia-70m` with a ten-times warmup (M1), on which the n-gram KL minima come 1–2 octaves later (M2), opens C6 — `trackC_writeup/status-C.md` "C2p readings audit (2026-10-30)"
  - C6 unit 0, the design (2026-10-31; tier 1, description; no energy on the run): `pythia-70m-warmup01` read off its weights is Pythia-70m's init and first batch (76 of 76 tensors bit-equal at step 0), its in-warmup rate a tenth of Pythia's (W 14 300 at peak 1e-3, H0 confirmed), its peak Pythia's within ~10 %; 70m's dates over `pythia-70m` and PolyPythias seeds 1–9: V 256 or 512 (5 and 5), the centred date 512–1000 (1000–2000 at 4.78e-3); bands per schedule and arm frozen into `c6_warmup.REF70`, with the reading revised after it was calibrated on the seeds' own profiles (planted truth survives 10, 9, 9 of 10; alone 7, 4, 9; the instantaneous truth only 2–4 of 10 if 70m's late states are planted less literally, all on V's count: Blocked 127); the gate fixed for U1 — `trackC_writeup/status-C.md` "C6 unit 0, the design (2026-10-31)"
  - C6 unit 1 (2026-11-01; Blocked 127 (a), the user's: V not read under instantaneous, frozen before output): seed 0's known answer exact, but **the run's gate refuses**. On the run, V sees plant B at 0.38 (7000; 13 of 34 units, 14 needed) and 0.33 (20 000) against the bar of 0.4 (`pythia-70m` at 7000, its own plant draws: 0.45, 13 of 29). The frozen gate takes the centred arms' dose from V's rule, so no arm is gated, though three would pass at the top dose (beside). Nothing on the run's grid was computed; the exit is met as "refused by its gate"; whether to read the centred count alone is Blocked 128 — `trackC_writeup/status-C.md` "C6 unit 1, Blocked 127 (a) and the run's gate (2026-11-01)"
  - C6 unit 1b (2026-11-01; Blocked 128 (a), the user's, one unit beyond the budget; frozen after U1's gate output, before any output on the run's grid): the centred count alone, plant B at 3b's 0.5, V not read. Gate: 9.61e-5, 1e-4, 3e-4 gated (4.78e-3 not, B 0.39 at 20 000), equal to U1's top-dose arms. **The centred count reads "consistent with cumulative"**: dated 2000 on all three arms against 70m's 512–1000 over ten seeds; step excluded (its band 512–1000), instantaneous excluded (the run is already at 16–18 at 2000, a step its early set holds); in the in-sample calibration that label came only from cumulative plants (2 of 10, none from the 20 others). So on 70m the centred date comes later than a fixed step allows, compatible with the summed rate; the grid places the shift at a factor of about 1 to 3.9 against seed 0 (`/challenge-pr` on #298, finding 1). V not read (its profile on disk and seen: a lean, finding 2); one count's three near-equal arms — `trackC_writeup/status-C.md` "C6 unit 1b"
  - **C6 closed 2026-11-01 at its exit** (U0, U1, U1b + close-out; Blocked 128 (a) extended it by one unit; 129 (a), the user's; writing only): V refused by its gate on the run; the centred count "consistent with cumulative", written into `paper/` scoped to that count (the abstract's schedule sentence scoped to 410m, §1, a §7 paragraph, "compatible with" the summed rate; 17 `c6.*` keys, 4 claims, `paper_check` 0, 35 pages; `/challenge-pr` on #299: accept with changes, findings 1–5 taken); F1 narrowed for 70m's centred date only (a fixed count of updates excluded there); V's date, the shift's size and 410m not established; U0's Parked 1 dropped, V on the run carried to ROADMAP C6 — `trackC_writeup/status-C.md` "C6 on one page (closed 2026-11-01)"
  - C7 (i) opened 2026-11-09 (STATE Blocked 153 revised; writing only): a scan (2605.07772 v2 read in full) and a proposed design that reads Isobe et al.'s three phases on 410m's stored runs by the paper's own fixed criterion, the FFN against attention per block, and the interaction matrix's gap per layer; the scan corrects the row's wording (the gap is QᵀK's, a weight matrix, not the attention pattern's; the escape's rate is the loss's, not the gap's); Blocked 154 — `trackC_writeup/status-C.md` "C7 (i) unit 0"
  - C7 (i) unit 1 (2026-11-09; Blocked 154 all (a); tier 1, description, seed 0, no forward-pass batch): by the paper's own criterion there is no terminal escape at any of the 18 steps, through block 23's own output (computed post hoc by a hooked pass); E₁ at the end of the stack sits above the interior plateau on 7 of 7 prompts everywhere, and through 8000 it is highest there. By the exact split, the FFN moves the tokens out on each of blocks 20–22 on 1–2 of 7 prompts at 32000–54000 only; the frozen first-order label is unreadable at 9 of 18 steps. Each head's symmetrised QᵀK is indefinite by construction, and each layer's sum is indefinite too (positive share 0.43–0.67), so the theorem's Assumption 3.1 fails everywhere — `trackC_writeup/status-C.md` "C7 (i) unit 1"
  - **C7 (i) closed 2026-11-09 at its exit** (3 units: U0, U1, close-out; writing only): on Pythia-410m, Isobe et al.'s own criterion finds no terminal escape on the block inputs at any checkpoint (through block 23's output, 1 of 7 prompts just passes at 143000 on states 1–24), and their positive-definiteness assumption fails at every layer; a §8 sentence and the email's paragraph written for the user; seeds and the ViT comparison carried to ROADMAP C7, the rest dropped with why — `trackC_writeup/status-C.md` "C7 (i) on one page (closed 2026-11-09)"
  - C4 opened 2026-11-09 (STATE Blocked 153; writing only): a scan and a proposed design that trains `pythia-70m` from seed 0's init on Pythia's data order with a flat rate at the peak and with Pythia's own schedule (the harness check), and labels the flat arm's dates against the harness arm's on a shared dense grid (step against the summed rate: about 256 against 23 for V, 1000 against 350 for the centred date; disjoint bands, sympy-checked); revised after `/challenge-pr` on #344, since C6's ten-seed bands overlap; V's gate runs on the flat arm; Blocked 155 taken all (a) (Claude, on the user's delegation) — `trackC_writeup/status-C.md` "C4 unit 0"
  - C4 U1 (2026-11-09; tier 1, harness): the harness reproduces Pythia's run to step 1000: the init bit-equal, the data rows verified against the Hub's hashes, the first updates' signs agreeing with the released run at 0.9994 (step 2) and 0.981 (step 4), the loop exact in float64, the reader exact against C2 3b's records, the held-out loss at 1000 within 0.38%, the counts and the induction head (L3 H6) tracking the released run's; the flat arm trains to 64 without diverging; the rule frozen in code before any count; weight decay corrected to the config's 0.1; the reading's registration is the user's (STATE Blocked 156) — `trackC_writeup/status-C.md` "C4 unit 1"
  - C4's design gains a description beside (2026-11-10, the user's; writing only): ρ_mid, ρ_1:5, m_ℓ and the sphere's effective rank on every saved checkpoint of both arms, read in U3 after R1–R3; no label, not in the registration — `trackC_writeup/status-C.md` "C4, an addition to the design"
  - C4 U2 (2026-11-12; `P-SCH1` registered first by the user, Blocked 156 (a)): the flat arm trained to 4000 without diverging; V's gate at the off-grid step 3500 passes as frozen, so V is read (plant B seen at 0.5, by one unit over the bar, where C6's run failed), and three of the four centred arms are gated (4.78e-3 not, as in C6); the flat arm's records written, no count read — `trackC_writeup/status-C.md` "C4 unit 2"
  - C4 U3 (2026-11-14; registered, `P-SCH1`): the harness arm trained to 4000 and matches the released run's dates on Pythia's grid on every count (R1); on the flat arm the centred date (R2, the registered reading) reads "neither": the flat arm dates before the bands of both the step count and the summed rate (48 against the harness arm's 640, summed-rate band from 92), so `P-SCH1`'s falsifier is met; V's date (R3, beside) also reads "neither" but one count on the harness arm's side would make it "consistent with cumulative", so it is undetermined; the flat arm's centred count overshoots its plateau about 2.4× before settling; one seed per arm, bands set by the grid; the cloud's concentration and effective rank described beside, no label — `trackC_writeup/status-C.md` "C4 unit 3"
  - **C4 closed 2026-11-14 at its exit** (4 units + close-out, the budget): on 70m with the ramp removed the registered prediction (the summed rate) failed, the centred date reading "neither"; a fixed count of updates is excluded from both sides with C6 (later under a longer ramp, earlier with none); read after the label, as description: the summed rate is excluded only by a factor of 1.73 in the harness arm's date (the grid's tolerance is 1.25, no run-to-run spread measured); the flat arm's count rises through a transient the harness arm lacks (the centred energy falling at nearly every transition past the first at 64–256) and both arms end on the same two transitions; the induction witness the design asked for, run here, dates induction at 768 on the harness arm and 320 on the flat arm at near-equal loss, in four heads that share none with the harness arm's four (the released run's); the label in `paper/` §7 and §1; candidates carried to ROADMAP C4′ — `trackC_writeup/status-C.md` "C4 on one page (closed 2026-11-14)"
  - C4′ opened 2026-11-15 (STATE Blocked 158 (a), the user's; writing only, nothing computed): a scan and a proposed design for two readings without training: induction's date on the released long-warmup 70m by C6's frozen rule, against C4's lean that it follows the summed rate, and whether the flat arm's function at each step is the harness arm's at some step (per-token KL against the harness arm's own step-to-step KL), each gated on the real design points before output; budget 3 units + close-out; decisions STATE Blocked 159 — `trackC_writeup/status-C.md` "C4′ unit 0"
  - C4′ U1 (2026-11-15; Blocked 159 all (a), the user's; rule frozen and pushed before output): on `pythia-70m` and PolyPythias seeds 1–9 every top induction head rises between 512 and 1000, so the reference is {1000}; the rule's leave-one-out gate passes 10 of 10 on every schedule (easy: each rise is one grid step); on this grid "consistent with cumulative" means a delay of 1× to about 8× over the seeds, not the summed rate in particular; A1's entry drafted and declined (Blocked 160 (b)), so A1 is tier 1: **the run reads "consistent with cumulative"** (half-rise in (2000, 3000] with 3000 beside, a 2× to 5.9× delay inside the summed rate's (1619, 3162]), a loss clock not excluded; A2: every seed's and the run's heads ≥ 0.5 are in layers 3–4, while C4's F's top head is L2H6; the run's centred set has a T3 majority at 2000–4000 that seed 0 never has, but whether either has a transient turns on the horizon (description) — `trackC_writeup/status-C.md` "C4′ unit 1"
  - C4′ U2 (2026-11-17; Blocked 163 (a), the user's; tier 1; open details fixed and pushed before output): per-token KL on 64 held-out rows between every flat-arm and harness-arm checkpoint. The gate passes in both windows: the released run against the harness arm in fp16 reads "one path" (identity 8 of 9 and 4 of 4; late at exactly the ⅔ bar), and 9 of 9 seeds read "another path". **The flat arm reads "another path" early and late**: 0 of 28 steps are within the harness arm's own step-to-step KL of any of its states, at 1.3–4.6× that resolution. At each shared step from 8 to 256 its residual exceeds every seed's distance from the released run at that step. Late, its ratio over the same-init null's residual falls from 6.7 at 512 to 1.9 at 4000. The matches are monotone and near equal loss. From step 48, in attention and MLP weights its displacement is near orthogonal to the harness arm's (cos 0.06–0.19, against the harness arm's own 0.26–1.0). So C4's item 10 is held: firmly early, with a narrowing margin late — `trackC_writeup/status-C.md` "C4′ unit 2"
  - **C4′ closed 2026-11-17 at its exit** (3 units + close-out, the budget; Blocked 164 (a), the user's; writing only): of C4's post hoc leans, item 9 (induction follows the summed rate, or a function of it; loss not excluded) and item 10 (the flat arm is not the harness arm on another clock: at none of its 28 saved steps within the harness arm's grid resolution of a saved state; over a same-init rerun by a wide margin early and a narrowing one late, in that resolution by 1.25–1.35 at the nearest) are held at tier 1 on one run or pair each; items 6–8 untouched; (b) in `paper/` §7 as one paragraph (14 keys, 2 claims, `paper_check` 0, 37 pages); nothing registered; U2's finer grid carried to ROADMAP C4′, its late window to C4′ (c), whose opening and registration are STATE Blocked 165 — `trackC_writeup/status-C.md` "C4′ on one page (closed 2026-11-17)"
  - C4′ (c) opened 2026-11-17 (STATE Blocked 165 (a), the user's; writing only, nothing computed): a scan and a proposed design for the replicate pair (A0′ and F′ from seed 0's init on another order). It measures A0′'s rerun ratio and re-reads item 6 and R3 on C4's own pair under design-C4's rerun term. Five leans (items 7–10, each stated against A0′) are drafted for registration, gated on C4's pair and on (released, A0). Found: shard 0's unused rows reach 3149 updates, not 4000, and a pair stopped at 1000 reads none of C4's dates; decisions STATE Blocked 166 — `trackC_writeup/status-C.md` "C4′ (c) unit 0"
  - C4′ (c) U2 (2026-11-18 to 2026-11-22; `P-SCH2`–`6` registered first by the user, Blocked 167 (a)): A0′ and F′ trained to 4000 on order π without diverging (four stops, each resumed with the re-trained step's loss reproduced); H4′ passes at 1000 (−0.13 %) and 4000 (−0.096 %, tol 2 %), so A0′ is read as a harness arm; nothing read — `trackC_writeup/status-C.md` "C4′ (c) unit 2"
  - C4′ (c) U3 (2026-11-22): C4's gate at 3500 on F′ as frozen (V and τ 9.61e-5, 1e-4, 3e-4), pinned in git before any record; of the five registered leans, L7 does not hold (A0′ has a one-step transient {T3} at 640), L8, L9, L9h and L10 hold; M2 leaves C4's R2 and R3 "neither"; M3 (the replicate) reads V "consistent with cumulative", robust to a one-step move of either arm, with C4's "neither" the reading at the band's edge, and the centred count "neither"; L10 (b) holds at 12 of 12, but from 1500 on by a factor ≤ 1.32; scoring in the registry is Blocked 168 — `trackC_writeup/status-C.md` "C4′ (c) unit 3"
  - **C4′ (c) closed 2026-11-22 at its exit** (4 units + close-out, the budget): items 8, 9, 10 held by registered entries (10 thin after 1500, so "early"); item 7 broken as registered (A0′ has a one-step transient), its F′ side replicating; item 6 (that the exclusion is fragile) split: not borne out for the centred count (order spread 1.0, the replicate "neither"), borne out on V, where the replicate reads "consistent with cumulative"; Kosson et al. read, a mechanism-level reference for item 7, no cell moved; one paragraph in `paper/` §7, and §1's sentence on the flat arm corrected to match — `trackC_writeup/status-C.md` "C4′ (c) on one page (closed 2026-11-22)"
  - C4′ (e)+(g) opened 2026-11-23 (STATE Blocked 169 (d), the user's: "(e) and (g) first"; writing only, nothing computed): a scan and a proposed design that reads the four trained arms. (e): the tokens' mean through W_U against unigram log-frequency, the exact split of each fall of ‖m‖² into its mode (removal or dilution) and writer (attention or MLP), and the arms' mean directions against the replicate's. (g): the loss barrier between the two schedules' end states, scaled by the replicate pairs, and each arm's induction heads mean-ablated. Leans stated before output (g1: warmup helps order-only connectivity); budget 3 units + close-out; (e) frozen on Blocked 170 D1, D2 (a) (Claude, on the user's "continue here"); (g)'s D3, D4 are Blocked 171 — `trackC_writeup/status-C.md` "C4′ (e)+(g) unit 0"
  - C4′ (e)+(g) U1, (e) (2026-11-23; tier 1; rules pushed before output): on all four arms the tokens' mean, read through the unembedding, is aligned with log unigram frequency at every readable step. The alignment is weak in the harness arm's earliest window and strong later; it is nearest just before or at V's date (78–90 % of the least-squares reference mid-stack), and F gets there about 12× earlier. The mean is written mostly by the MLPs. At the block where V's falls sit, both schedules start with the MLP diluting the mean and end with attention diluting it; only the flat schedule has a stage between (128–640) where the MLP writes against the mean, and the replicate shows the same stages. The two schedules' mean directions are near orthogonal by 4000 (cos 0.14 against a reordering's 0.94), though both carry the unigram prior; revised after `/challenge-pr` on #367 — `trackC_writeup/status-C.md` "C4′ (e)+(g) unit 1"
  - C4′ (e)+(g) U2, (g) (2026-11-23; tier 1; rules pushed before output): the two schedules' end states, from one init, are not one basin. The held-out loss rises 3.56 nats along the straight line between them at 4000 (2.45 at 1000), against 0.13 between two orderings on Pythia's schedule and 1.03 on the flat one, so warmup keeps reorderings connected and the schedules sit beyond the order-only spread; the replicate pair gives the same. Mean-ablating each arm's induction heads breaks induction, ablating the other schedule's indices does nothing; on the first pair one random set, holding a single early head, hurts as much, so the frozen 'own heads carry induction' fails there, but on a loss saturated at a uniform guess, so it is unresolved (revised after `/challenge-pr` on #369) — `trackC_writeup/status-C.md` "C4′ (e)+(g) unit 2"
  - **C4′ (e)+(g) closed 2026-11-23 at its exit and budget** (3 units + close-out): the tokens' mean carries the unigram prior on both schedules; at V's block both end with attention diluting it; the two schedules' end states sit in different basins, their mean directions are near orthogonal, and mid-stack their readouts differ about as much as another init's (post hoc); the other schedule's induction-head indices do nothing when ablated, and each arm's previous-token heads alone remove as much copying as its induction heads, so g3 is unresolved (post hoc; revised after `/challenge-pr` on #370) — `trackC_writeup/status-C.md` "C4′ (e)+(g) on one page (closed 2026-11-23)"
  - C4′ (d)+(f) opened 2026-11-23 (STATE Blocked 169's order, after (e)+(g); writing only, nothing trained): a scan and a proposed design. (d): flat arms at a half and a tenth of the peak, the transient's onset and induction's date read against a step and a dose band; (f): A0's step-256 weights continued at the peak against the same restart at A0's own rate. The ROADMAP's horizons were too short for the dose clock (F⅒'s transient band reaches 800), and F's own grid cannot tell a half-rate dose from a step, so F is re-run to 128 on a 4-update grid (F¹); A0 passes F's onset dose at 0.26 of the peak, and the peak itself at 1430, with no transient. Revised after `/challenge-pr` on #371. Decisions D1–D4, STATE Blocked 172 — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 0"
  - C4′ (d)+(f) U1 (2026-11-23; tier 1): design frozen on Blocked 172 (as recommended, Claude on the user's delegation); rules pushed before output; all five gates pass on the four trained arms. F¹ (the flat arm re-run to 128 on a 4-update grid) is not bit-equal to F from update 4 (the harness is not deterministic past one update), so it is a near-replicate; its onsets (48, 52, 52 at the three gated τ arms) make the half-rate step and dose windows disjoint. Two entries drafted for the user's registration — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 1"
  - C4′ (d)+(f) U2a (2026-11-23; tier 1, nothing read): (f)'s branches B_P and B_O trained from A0's step-256 weights (fresh AdamW; the peak, A0's own rate) to 512; each starts from A0's state and batch (step 257's loss equal to A0's); records and heads written. F½, F⅒ wait on the user's Blocked 173 — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 2a"
  - C4′ (d)+(f) U3a, f (2026-11-24; tier 1; Gupta et al. read and the D2 (b) rule pushed before the read): **f reads "not the early state's" at all three gated τ arms.** A fresh AdamW at the peak makes the transient on A0's trained step-256 state as on F's init, and the same restart at A0's own rate makes none. Not separated: the rate's jump from the restart's first updates (by design), so D2 (b) is worth its hours by the rule fixed before the read; STATE Blocked 174 — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 3a"
  - C4′ (d)+(f) U3b, K (2026-11-24; tier 1; Blocked 174 (a), Claude on the user's delegation; rule K pushed before output): **K reads "the restart's" at all three gated τ arms.** A0's state at 256 with its Adam moments kept (rebuilt, 0.98 % from the saved weights) jumped to the peak makes no transient by 384, nor at A0's own rate (BKO), where B_P (fresh moments, the same jump) makes one at 280–288. So on a trained state the transient needs fresh moments *and* a high rate; neither the jump alone nor the restart at a low rate makes it. Fresh moments also make the first updates 3× larger, so the moments and the early steps' size are not separated (`/challenge-pr` on #375). A second F¹ replicate moves 1e-4's reference to (44, 48]. P-SCH7, P-SCH8 registered (the user's) — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 3b"
  - C4′ (d)+(f) U2b (2026-11-25; under P-SCH7, P-SCH8; nothing read): F⅒ and F½ (seed 0's init, C4's order, flat at a tenth and a half of the peak) trained 0 → 1000, 77 states each on grid D, every loss finite; records and heads written. Blocked 175 (a), (a), the user's — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 2b"
  - C4′ (d)+(f) U3c, d (2026-11-25; P-SCH7, P-SCH8 scored by their frozen rules): **neither entry holds.** F⅒ has a transient too by rule T (onset (752, 784] at every gated arm, past its dose band; cut off at 1000, its size about a sixth of F½'s), so the registered threshold is not borne out, and a weaker effect at a low rate or a much slower clock is not separated from "no threshold"; F½'s onset is labelled step beside ((56, 60] to (60, 64], one on the band's edge; not scored) and its induction date (304, 320] is in the step band; d1 joint "mixed"; d3 "not step" (F⅒ not formed by 1000). Scoring sentences drafted, Blocked 177 — `trackC_writeup/status-C.md` "C4′ (d)+(f) unit 3c"
  - C4′ (d)+(f) closed 2026-11-25 at its exit and extended budget (7 units + close-out): the transient is not the early state's (a trained state shows it after a fresh Adam at the peak), nor the rate's jump alone (kept moments at the peak show none within 128 updates), nor confined to the peak (present at a half and a tenth from init); no single condition yet covers every arm (F⅒'s comes at 1e-4 after ~750 updates, long after its moments adapted; `/challenge-pr` on #379, finding 0); neither a step nor a dose clock dates it across rates; at half the peak induction comes at F's update count; P-SCH7, P-SCH8 falsified; the trigger is ROADMAP §8 C8's — `trackC_writeup/status-C.md` "C4′ (d)+(f) on one page (closed 2026-11-25)"; `PROJECT.md` §3.88
  - C9 opened 2026-11-25 (Blocked 176 (a), the user's; writing only, nothing read): scan and a proposed design. U1, the mean's term on every model, is a CPU read on records that already exist; U2, A3's pairs on GPT-2 small × 5 and the harness arms; U3, the 160m seeds' dates (downloading); U4, the paper. D1–D6, Blocked 178 — `trackC_writeup/status-C.md` "C9 unit 0"
  - C9 U1, the mean's term on every model (2026-11-25; Blocked 178 all (a), Claude on the user's "merged, continue"; tier 1, no forward pass; rule pushed before output): the share of counted drops whose mean's term carries over half reads "carries" (≥ 410m's 0.963 less 0.1) on 410m, 160m, 70m (by 0.02), GPT-2 small and the harness and flat arms, and "partly" on `pythia-70m-warmup01` (0.826, one run, low at 8000–54000); warmup01's 0.826 lies inside 70m's seeds' 0.802–0.981, so not shown to be its warmup's; the paper's S_any is 0.961–0.988 pooled on every model; at the break ‖m‖² falls at every counted drop on every model (the abstract's sentence), but its term is over half of each only on the Pythia-shaped runs, not on GPT-2 small (0.76–0.87; §4's clause) — `trackC_writeup/status-C.md` "C9 unit 1"
  - C9 U2a, the locator by grid and the weights rows (2026-11-26; D1–D4, D6 confirmed by the user; tier 1, CPU, nothing located off 410m; rules pushed before output): A3's locator and gate parametrised by grid reproduce A3 unit 5's dates and floors exactly on its own units; the weights producer reproduces A3's E2, E7, E8 on 410m; E2, E7, E8 are gated on GPT-2 small × 5 (35 tags) and A0, A0′ (28 steps), and all 21 pass at floor 4.0 — `trackC_writeup/status-C.md` "C9 unit 2a"
  - C9 U2b, the GPU rows and the pairs (2026-11-26; Blocked 179 D7 (a), D8 (a), the user's; tier 1): every known answer exact. The snapped gate refuses every GPT-2 row, so no GPT-2 pair is read; one pair is read on the harness arms (A0's E1–E3 aligned, replicating 410m's verdict). Accepted as frozen (Blocked 180 (a), the user's) — `trackC_writeup/status-C.md` "C9 unit 2b"
  - C9 U3, the 160m seeds' two dates (2026-11-26; D5 (a), the user's: tier 1, nothing registered; rules pushed before output): known answer bit-equal on seed 0. E₁'s break is 256 on all ten 160m seeds. The centred date is 1000 at every arm on 6 seeds, and 512–1000 across arms on 4. U1's M on the ten seeds is 0.927, "carries" — `trackC_writeup/status-C.md` "C9 unit 3"
  - C9 closed 2026-11-26 at its exit and budget (5 units + close-out with U4, Blocked 181 (i)–(iii) (a), the user's; writing only): `paper/` now carries M on every model (table 2's last column; §4 scoped to the Pythia-shaped runs at the break), 160m's ten seeds (256 on 10 of 10; centred 1000 on 6, 512–1000 on 4 with a note at the bar), and §5 "On other models" (the order unread off 410m; one pair read on A0); `PROJECT.md` §3.89 — `trackC_writeup/status-C.md` "C9 on one page (closed 2026-11-26)"
  - C8 opened 2026-11-27 (the user's: "U0 ... candidate (i) first"; writing only, nothing computed): a scan and a proposed design for what triggers the transient. (i)'s rule is fixed in full: loss-rise and grad-spike events in a 16-update lead window on five arms with a transient, against each arm's own background, with a gate on the controls' real logs; "marked", "silent" or "mixed". (ii) sharpness against Adam's 38/η, (iv) causal arms and (iii)+(v) are fixed in shape. Budget 4 units + close-out, GPU ≈ 15–21.5 h; decisions Blocked 182 — `trackC_writeup/status-C.md` "C8 unit 0"
  - Paper: every model read (2026-11-24; writing only, nothing computed): `paper/` §4.2 tabulates every model's two dates (on every warmup run that dates both, E₁'s break comes first) and Dodecahedron's gate refusal; §5 adds the finer 70m grid (E₁ before the centred break, which is one saved step before induction on both harness arms: aligned by §5's convention); the rest of the plan is ROADMAP §8 C9 — `trackC_writeup/status-C.md` "Paper: every model read"
- **Superseded / wrong:**
  - C6 U0's schedule form (`c6_schedule.lr`: a cosine to `min_lr`) is not the one the runs trained under; the summed rate differs by 6.2 % at 143 000 and under 0.11 % to 20 000; U0's H0 statistic becomes 0.197 against 0.223 (still confirmed), no band or label moves — `trackC_writeup/status-C.md` "Corrections received"
- **Registry:** `P-SCH1`, C4's reading on the flat arm, registered by the user 2026-11-10 before F trained past step 64 (Blocked 156 (a)); read 2026-11-14, R2 "neither", its falsifier met (scoring the entry decided 2026-11-15, STATE Blocked 157 (a): a sentence in its `notes`, the user's commit); C4′ registered nothing (A1's draft entry declined, Blocked 160 (b); (b) tier 1 by design); C4′ (c)'s five leans `P-SCH2`–`P-SCH6` (L7, L8, L9, L9h, L10) registered by the user 2026-11-18 (Blocked 167 (a), `87e996b`) after U1's gates, before either arm's first update; read 2026-11-22: `P-SCH2` does not hold, `P-SCH3`–`6` hold, scoring in `notes` the user's (STATE Blocked 168; `trackC_writeup/status-C.md` "C4′ (c) unit 3"); C4′ (d)+(f)'s d1 and d2 registered by the user 2026-11-24 as `P-SCH7`, `P-SCH8` (Blocked 173 (a), `4ebc723`) after U1's gates and both F¹ replicates, before F½'s or F⅒'s first update; read 2026-11-25, both do not hold (`trackC_writeup/status-C.md` "C4′ (d)+(f) unit 3c"); every P-SCH entry's scoring sentence is in its `notes` since 2026-11-26 (STATE Blocked 157, 168, 177, the user's instruction; `status` stays `active`, the registry having no "falsified"); C2 quotes `P-A5`, Track A's (its `real_run_record` set 2026-11-26, Blocked 79)
- **Depends on:** A@2db9b14ba1, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the dates are fixed in steps (or tokens) or wait for a cumulative LR dose (F1, standing by Blocked 101 (a) for 410m; narrowed by C6 for 70m's centred date only, where a fixed count of updates is excluded); 3b could not separate them on Pythia's grid; 3c, on GPT-2 small, reads step for E₁'s larger stage, a first stage in the dose's band, no centred date, and a frozen label that turns on the window and one re-init unit (description only); ROADMAP C4, the schedule intervention (with 3c's Parked 0), separates them on 70m only, 410m by inference (C4 trains `pythia-70m`; corrected 2026-11-08, `/challenge-pr` on #330, finding 3); V on 70m's long-warmup run is unread (its gate refused), a reopen candidate in ROADMAP C6 — `trackC_writeup/status-C.md` "C6 on one page (closed 2026-11-01)"
  - After C4 (closed 2026-11-14): on 70m the dates are not fixed in steps, and on the flat arm they are not the summed rate's either, by a margin the run-to-run spread could cover; the flat arm's counts date a transient, so what sets the dates is open, and so are the spread, whether the schedule chooses the induction heads, and whether induction's date follows the summed rate; 410m is still unseparated. Seven candidates, none scheduled: ROADMAP §8 C4′ — `trackC_writeup/status-C.md` "C4 on one page (closed 2026-11-14)"
  - After C4′ (closed 2026-11-17): induction on warmup01 follows the summed rate or a function of it (rate or loss not told apart), and the flat arm takes another path; whether a finer harness grid moves the steps nearest the bar stays open (ROADMAP §8 C4′, unscheduled) — `trackC_writeup/status-C.md` "C4′ on one page (closed 2026-11-17)"
  - After C4′ (c) (closed 2026-11-22): on one reordering the spread is a grid step at most; the centred date is "neither" on both pairs, V "consistent with cumulative" on the second; open are a second replicate or another init, whether a finer grid moves L7's one-step transient or V's one-step margin, F's late distance after 1500 against a reordering's, and the clock inside the summed rate's family; 410m still unseparated — `trackC_writeup/status-C.md` "C4′ (c) on one page (closed 2026-11-22)"
  - When the tokens' mean, and the falls V counts, become basis-aligned: the level equals a random basis's at 2000 and is below it at 4000 on every seed; the falls at 2000–8000 not read (records on disk, no forward pass; C5's Parked, carried at its close): ROADMAP C5, unscheduled — `trackC_writeup/status-C.md` "C5 on M2, on one page (closed 2026-10-24)"
  - Whether R breaks on its own, E_β at other β, component transplants, the small sizes' seeds, a sampled prompt set: ROADMAP C5, unscheduled — `trackC_writeup/status-C.md` "C2 on one page (closed 2026-10-23)"
  - The draft's rewrite and posting (anonymity and the timestamp narrative, m11; the abstract's density, m12; M3 (iii); a code and data statement; the venue; the records' release): the user's, Blocked 103 (a); the rewrite done (the user, 2026-11-09; its `paper/` commit pending), F1 still open (posting before F1 is the user's); C7 (i) closed with a §8 sentence for the user to carry; C4's design next (Blocked 153, revised 2026-11-09) — `trackC_writeup/referee-C2p.md`
  - C7 (i) on the other seeds (if the authors ask) and the ViT comparison for §8 (2511.21635): ROADMAP §8 C7, unscheduled — `trackC_writeup/status-C.md` "C7 (i) on one page (closed 2026-11-09)"
  - After C9 (closed 2026-11-26): the timeline's order off 410m (unread on GPT-2 at Pythia's resolution; A0′'s pair and A0's E4 wait on an unbiased max-over-heads null, ROADMAP §8 C9's standing note); why 160m breaks a step before most 410m seeds (unregistered); a per-transition split of R at GPT-2's breaks (ROADMAP §8 C5); the `c2_seeds` refusals (ROADMAP §8 C8) — `trackC_writeup/status-C.md` "C9 on one page (closed 2026-11-26)"
- **After Phase 10:**
  - none
- **Reviewed:** 2026-11-27 · body `ee50237fc8`
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
| C2 5b. the referee's readings from records (102 (a)) | **done 2026-10-23: frozen before output (`9629d78`); no forward pass; R1–R8 description; E₁ and ‖m‖² move together at ≥ 0.96 of transitions at every step (post hoc, after `/challenge-pr` on #261), so V nearly counts the mean's falls; the mean's term is more than half of all but 1 of the 1224 drops at 128–1000; the frame transitions, a later plateau and per-prompt counts move no date; figure 5, figure 4's right panel normalised** | `c2_readings.py`, design "Unit 5b"; "C2 unit 5b" below |
| C2 close-out | **done 2026-10-23: writing only; exit met at 9 units + close-out; every Parked item carried or dropped; where to post was Blocked 103, decided (a)** | "C2 on one page" below |
| C5 0. M2's scan and design (writing only) | **done 2026-10-23: proposed; Blocked 104 (the set and scope)** | `lit-C5.md`, `design-C5.md`; "C5 unit 0" below |
| C5 1. M2's preflight, run and reading | **done 2026-10-23: 104 (a), the user's; frozen before output (`1b4e2c6`, `6a38770`); P1 kept 7 of 9, P2 kept 9 of 9; R0′ 0 on every seed; R0b 16000; the exit met** | `c5_massive.py`; "C5 unit 1" below |
| C5 2. M2's other half (105 (a), the user's; budget 3 units + close-out) | **done 2026-10-24**: the gate refused the absolute "few" (`ded356f`); 106 (a), the user's: the rotation reading primary, re-frozen (`6db31cf`); **"not basis-aligned", 9 of 9** (post hoc, slightly more spread than a random basis); no recurring coordinate; the mean spread at the break (half in 132–135 of 1024); revised after `/challenge-pr` on #265; the exit met | `c5_mean.py`; "C5 unit 2" below |
| C5 close-out (M2) | **done 2026-10-24: writing only; exit met at 3 units + close-out; Parked items carried (ROADMAP C5, C2p) or dropped** | "C5 on M2, on one page" below |
| C2p 1. the LaTeX and ten figures (opened at the user's request) | **done 2026-10-24: `tex/` compiles (22 pages); figures 6–10 new, 9 and 10 C5's; `numbers.py`'s docstring; Blocked 108** | `design-C2p.md`, `to_tex.py`, `tex/`; "C2p unit 1" below |
| C2p 2. the revisions and the second fresh read (107 (a), 108 (a), the user's) | **done 2026-10-24: `paper/` is the source; M2's results level first, §8 and F9 against 2510.06477 (its grid reads no step in (1, 1000)), m15, m19, m25 (`fig11_rho`); `paper_check` 0; the read: major revision, 6 + 20 points answered; Blocked 109** | `paper/`, `paper_check.py`, `referee-C2p.md`; "C2p unit 2" below |
| C2p close-out (109 (a), the user's) | **done 2026-10-24: writing only; the cut and the read's framing points made in `paper/` (28 pages, `paper_check` 0); exit met at 2 units + close-out; no Parked items** | "C2p on one page" below |
| after C2p: one citation (the user's, 2026-10-24) | **done 2026-10-24: writing only; 2609.28448 (Gao, Yang, Chen) cited in `paper/` §6 (its Sec. V B: negative eigenvalues of one head's map do not establish negative feedback, beside E2's fall) and §8 (the repulsive case as a discrete map that need not settle); `refs.bib` `gao2026`; after `/challenge-pr` on #270, §5 says the OV phase is named for E2's share, not a shown negative feedback (so §6's sentence and §5 agree); still 28 pages, no undefined reference, `paper_check` 0; no number added** | `paper/main.tex`, `paper/refs.bib` |
| after C2p: appendix D, the rows one by one (the user's, 2026-10-25) | **done 2026-10-25: writing only, on records already read; appendix D rewritten as one entry per row of table 2 with its own panel of figure 5: what it measures, why it is on the timeline, how it is dated, and whether its change clears the locator's floor (reference, σ, ref / σ, the null arm's no-change share, all from A3's `a3_rows.json`); a new table 3 of those numbers; table 2's row names and figure 5's caption link to the entries; `numbers.py` new keys `a3.x0`, `a3.ref`, `a3.ref_step`, `a3.x_ref`, `a3.sigma`, `a3.ref_sigma`, `a3.n_units`, `a3.own_floor`, `a3.null_no_change`, `a3.low_power`, `a3.low_power_9`, `a3.x_last`, `a3.E8_peak_ref`, `a3.E8_peak_step` and `row_panels` (`fig/paper/figD_<row>.pdf`, copied into `paper/fig/`; no half-line on the n-gram rows, dated at their minimum); 93 slots + one literal (the gate's 95 %) in `paper_slots.json`; 34 pages, no undefined reference, `paper_check` 0; the page republished. After `/challenge-pr` on #275 (accept with changes): E8's peak step (256, from the right key), the n-gram panels' half-line, the registration's wording (intermediate checkpoints downloaded) and E1's reason for no σ corrected, panels renamed `figD_`; whether to generate table 3 into `paper/` left to the user (Blocked 108 keeps `paper/` hand-edited)** | `paper/main.tex` appendix D; re-run: `python -m trackC_writeup.numbers produce && python -m trackC_writeup.numbers paper-figures`, then copy `fig/paper/figD_*.pdf` into `paper/fig/` |
| after C2p: figure 12, the energy ridgeline (the user's, 2026-10-28) | **drawn 2026-10-28, not yet in `paper/` (yours to include): tier 1, description, no forward pass; from the page https://claude.ai/artifact/TBed9FFnPDLDWKAeN1duSQ, its identity prototype ported to a tracked producer** | `c2_ridge.py`, `numbers.py` `ridge_figure`, `fig/paper/fig12_ridge.pdf`; "Figure 12, the energy ridgeline" below |
| after C2p: figure 13, the layer × step small multiples (the user's, 2026-10-29) | **done 2026-10-29: tier 1, description; no forward pass; `data/phase12` checked against A5's seed 0 first (passes); `fig/paper/fig13_grid.pdf`, not in `paper/main.tex`; after `/challenge-pr` on #281 (accept with changes): the input hash covers the gate and A3's rows, the check has refusing tests, and the energy-change and Schur panels' limits are stated; which quantity those two panels draw is Blocked 116** | `c2_grid.py`, `numbers.py` `grid_figure`; "Figure 13" below |
| after C2p: the readings audit (the user's, 2026-10-30) | **done 2026-10-30: writing only; the draft's citations against the full texts; W1–W6 fixed, M1–M8 in the text (M9 not), C6 opened; 34 pages, `paper_check` 0** | `lit-C2p-fulltext.md`, `paper/`, `tools/math_checks/c2p_mean_cosine.py`; "C2p readings audit" below |
| C6 0. the design (opened 2026-10-30, the user's; budget 3 units) | **done 2026-10-31: the run's schedule, init and first batch from its weights (H0); 70m's dates on seeds 0–9; `REF70` and the bands frozen, the reading calibrated on the seeds' profiles and revised before freezing; the gate fixed; no energy on the run** | `design-C6.md`, `c6_schedule.py`, `c6_warmup.py`, `tools/math_checks/c6_bands.py`; "C6 unit 0" below |
| C6 1. Blocked 127 (a), the gate | **done 2026-11-01: 127 (a) frozen before output (`98d3f38`); seed 0's known answer exact; the run's gate refuses (no plant-B dose for the uncentred count at 7000 or 20 000, so no arm gated); no record on the run's grid; the exit met ("refused by its gate"); Blocked 128** | `c6_warmup.py`; "C6 unit 1" below |
| C6 1b. the centred count alone (Blocked 128 (a), the user's; beyond the budget) | **done 2026-11-01: frozen before output on the run's grid (`fd18085`); 3 arms gated, equal to U1's top-dose arms; "consistent with cumulative" (dated 2000; step and instantaneous excluded); V not read** | `c6_warmup.py gate --centred-only`; "C6 unit 1b" below |
| C6 close-out (Blocked 129 (a), the user's) | **done 2026-11-01: writing only; the label in `paper/` (abstract, §1, a §7 paragraph), scoped to the centred count; 17 `c6.*` keys from three records, 4 claims; `paper_check` 0, 35 pages; F1 narrowed for 70m's centred date; exit met; revised after `/challenge-pr` on #299 (findings 1–5 taken)** | "C6 on one page" below |
| C7 (i) 0. the scan and design (Blocked 153 revised; budget 3 units) | **done 2026-11-09: writing only; 2605.07772 v2 read in full; design proposed, revised after `/challenge-pr` on #341; frozen on Blocked 154 all (a) (Claude, on the user's delegation)** | `lit-C7.md`, `design-C7.md`; "C7 (i) unit 0" below |
| C7 (i) 1. the read | **done 2026-11-09: tier 1, description; gate and k1–k3 pass; no terminal escape at any of the 18 steps (0 of 7 prompts, E₁ and ‖m‖², states 0–23 and 1–23, and through block 23's output, post hoc); by the exact split the FFN moves the tokens out on blocks 20–22 on 1–2 of 7 at 32000–54000 only (the frozen first-order label unreadable at 9 of 18 steps); Assumption 3.1 fails at every layer and step; revised after `/challenge-pr` on #342** | `c7_profile.py`; "C7 (i) unit 1" below |
| C7 (i) close-out | **done 2026-11-09: writing only; exit met at 3 units; Parked items carried (ROADMAP C7) or dropped; a §8 sentence and the email's paragraph for the user** | "C7 (i) on one page" below |
| C4 0. the scan and design (Blocked 153; budget 4 units + close-out) | **done 2026-11-09: writing only; frozen on Blocked 155 all (a) (Claude, on the user's delegation)** | `lit-C4.md`, `design-C4.md`; "C4 unit 0" below |
| C4 1. the harness, A0 to 1000, F to 64 | **done 2026-11-09: data verified (H2); H1, H3, H4 (−0.38% at 1000), H5 pass; the reader's known answer exact; A0's counts and induction head track the released run's (H6); F to 64 without divergence; the rule frozen in code; registration Blocked 156** | `c4_train.py`, `c4_read.py`; "C4 unit 1" below |
| C4 2. F to 4000 and V's gate (`P-SCH1` registered first, Blocked 156 (a)) | **done 2026-11-12: F trained to 4000 in 6 segments without diverging (loss 2.990); V's gate at 3500 passes as frozen (plant B at 0.5, 13 of 28 units, 12 needed); centred arms 9.61e-5, 1e-4, 3e-4 gated, 4.78e-3 not; 203 records; nothing counted** | `data/trackC/c4/2026-11-12-u2/`; "C4 unit 2" below |
| C4 3. A0 1000 → 4000, A0's records, R1–R3, the description beside | **done 2026-11-14: A0 1000 → 4000 (3 segments, loss 3.042); R1 passes; R2 (centred) "neither": F dates before both maps' bands; R3 (V) "neither" on one count of A0's, so undetermined; `P-SCH1`'s falsifier met (scoring, Blocked 157); the description beside written** | "C4 unit 3" below |
| C4 close-out | **done 2026-11-14: exit met at the budget (4 units + close-out); the label in `paper/` §7 and §1 (28 `c4.*` keys, 5 claims; `paper_check` 0, 36 pages); the design's induction witness run on both arms (not run in U3): 768 on A0, 320 on F, the two arms' four heads disjoint; post hoc beside: F's count rises through a transient absent on A0, both end on the same two transitions; Parked carried (ROADMAP C4′) or dropped** | `c4_close.py`, `data/trackC/c4/2026-11-14-close/`; "C4 on one page" below |
| C4′ 0. the scan and design (Blocked 158 (a), the user's; budget 3 units + close-out) | **done 2026-11-15: writing only, nothing computed; (a) and (b) designed, each gated on the real design points before output; proposed, D1–D5 Blocked 159** | `lit-C4p.md`, `design-C4p.md`; "C4′ unit 0" below |
| C4′ 1. (a): A1's gate, the draft entry, A3; then the run's scores, A1, A2 | **done 2026-11-15 (Blocked 159 all (a); 160 (b), entry declined): the ten seeds all dated 1000; gate 10 of 10; A1 (tier 1) "consistent with cumulative" (run 0.011 at 1000, 0.43 at 2000, 0.87 at 3000; half-rise in (2000, 3000], a 2× to 5.9× delay, inside the summed rate's (1619, 3162]); a loss clock not excluded; A2: every seed's and the run's heads ≥ 0.5 are in layers 3–4, F's top head is L2H6; A3 written** | `c4p_read.py`, `data/trackC/c4p/2026-11-15-u1/`; "C4′ unit 1" below |
| C4′ 2. (b): F against A0's path by per-token KL (Blocked 163 (a), the user's) | **done 2026-11-17: tier 1; open details fixed before output (`231cb77`, `c8876cd`); gate passes both windows (null "one path", 8 of 9 and 4 of 4 identity; seeds 9 of 9 "another path"); F "another path" early and late, 0 of 28 steps on the path; r 1.3–4.6 × δ; early r beyond any seed's, late F's r over the same-init null's falls 6.7 → 1.9; `/challenge-pr` on #359 accept with changes, 1–5 taken** | `c4p_path.py`, `data/trackC/c4p/2026-11-17-u2/`; "C4′ unit 2" below |
| C4′ close-out (Blocked 164 (a), the user's) | **done 2026-11-17: writing only; exit met at the budget (3 units + close-out); C4's items 9 and 10 held, 6–8 untouched; (b) in `paper/` §7 (one paragraph, 14 `c4p.*` keys, 2 claims; `/challenge-pr` on #360 accept with changes, 1–4 taken; `paper_check` 0, 37 pages); Parked carried (ROADMAP C4′, (c)) or dropped; whether (c) opens and is registered is Blocked 165** | "C4′ on one page" below |
| C4′ (c) 0. the scan and design (Blocked 165 (a), the user's; budget 4 units + close-out) | **done 2026-11-17: writing only, nothing computed; proposed, D1–D4 Blocked 166** | `lit-C4pc.md`, `design-C4pc.md`, `tools/math_checks/c4pc_nearer.py`; "C4′ (c) unit 0" below |
| C4′ (c) 1. harness, π, rules, gates, entries drafted (Blocked 166 all (a), Claude on the user's delegation) | **done 2026-11-18: gates for L7–L9h pass; no arm has trained; registration Blocked 167** | `c4pc_train.py`, `c4pc_read.py`, `data/trackC/c4pc/2026-11-18-u1/`; "C4′ (c) unit 1" below |
| C4′ (c) 2. registration, then the arms (Blocked 167 (a), the user's) | **done 2026-11-22: `P-SCH2`–`6` registered (`87e996b`) before any update; both arms to 4000 without divergence; H4′ passes at 1000 (−0.13 %) and 4000 (−0.096 %); nothing read** | "C4′ (c) unit 2" below |
| C4′ (c) 3. the 3500 gate pinned, records, heads, K, the five scores, M1–M4 | **done 2026-11-22: gate as frozen (V read, τ 9.61e-5, 1e-4, 3e-4); L7 does not hold; L8, L9, L9h, L10 hold; M3's V reads "consistent with cumulative", its centred count "neither"; scoring the entries is Blocked 168** | "C4′ (c) unit 3" below |
| C4′ (c) close-out (writing only) | **done 2026-11-22: items 6–10 marked; Kosson et al. read; one paragraph in `paper/` §7 (20 keys, 4 claims, `paper_check` 0)** | "C4′ (c) on one page (closed 2026-11-22)" below |
| C4′ (e)+(g) 0. the scan and design (Blocked 169, the user's: "(e) and (g) first"; budget 3 units + close-out) | **done 2026-11-23: writing only, nothing computed; (e) frozen on Blocked 170 D1, D2 (a) (Claude, on the user's "continue here"); (g)'s D3, D4 are Blocked 171** | `lit-C4peg.md`, `design-C4peg.md`; "C4′ (e)+(g) unit 0" below |
| C4′ (e)+(g) 1. (e) on the four arms | **done 2026-11-23: tier 1; every known answer exact; e1 held on both pairs; e2 held on A0, broken on F as pooled, while by step both end in attention's dilution and F alone has an MLP-removal stage; e3's end half broken; revised after `/challenge-pr` on #367** | `c4peg_read.py`; "C4′ (e)+(g) unit 1" below |
| C4′ (e)+(g) 2. (g) on the four arms (Blocked 171 D3 (a), D4 (a), E (a), Claude on the user's "do the next 5 items") | **done 2026-11-23: tier 1; G1 gated and readable; (A0, F) "beyond it" (B 3.56 nats against S 1.03), pair 2 the same; g1 held, g2 held; g3 broken as frozen on A0 and F (one random set as harmful as the own heads), held on A0′ and F′, unresolved since the induction loss saturates; the other schedule's indices carry nothing on any arm; revised after `/challenge-pr` on #369** | `c4peg_g.py`; "C4′ (e)+(g) unit 2" below |
| C4′ (e)+(g) close-out (two post hoc reads, minutes) | **done 2026-11-23: at the budget (3 units + close-out); at mid-stack the schedules' means read through W_U agree no more than another init's (the final state is at the read's ceiling for any two models); g3 unresolved on both pairs, each arm's previous-token heads alone remove as much of the copy gain as its induction heads; revised after `/challenge-pr` on #370** | `c4peg_close.py`; "C4′ (e)+(g) on one page (closed 2026-11-23)" below |
| C4′ (d)+(f) 0. the scan and design (Blocked 169's order; budget 4 units + close-out) | **done 2026-11-23: writing only, nothing trained; proposed, D1–D4 Blocked 172** | `lit-C4pdf.md`, `design-C4pdf.md`, `tools/math_checks/c4pdf_bands.py`; "C4′ (d)+(f) unit 0" below |
| C4′ (d)+(f) 1. frozen (Blocked 172, Claude on the user's delegation); rules, driver, reader; gates; F¹; entries drafted | **done 2026-11-23: gates G-T, G-d1, G-I, G-d2, G-f pass; H-d fails (not bit-reproducible past update 1), F¹ a near-replicate; references (44, 48], (48, 52], (48, 52]; G-d1 re-run on them passes; registration Blocked 173** | `c4pdf_rules.py`, `c4pdf_train.py`, `c4pdf_read.py`, `data/trackC/c4pdf/2026-11-23-u1b/` (revised after `/challenge-pr` on #372); "C4′ (d)+(f) unit 1" below |
| C4′ (d)+(f) 2a. (f)'s branches B_P, B_O (D2 (a); not gated on Blocked 173) | **done 2026-11-23: both trained 256 → 512, the state half of H-f passes and step 257's loss matches A0's to every digit; records and heads written (17 saves each); nothing read. F½, F⅒ (U2's other half) wait on Blocked 173** | `data/trackC/c4pdf/2026-11-23-u2/`; "C4′ (d)+(f) unit 2a" below |
| C4′ (d)+(f) 3a. f's read (not gated on Blocked 173) | **done 2026-11-24: Gupta et al. read first (silent on moments and on the spike's length); f "not the early state's" at 9.61e-5, 1e-4, 3e-4 (B_P a transient from 280, 280, 288 to 512; B_O none); D2 (b) is Blocked 174** | `data/trackC/c4pdf/2026-11-24-u3/reading_f.json`; "C4′ (d)+(f) unit 3a" below |
| C4′ (d)+(f) 3b. B_K (Blocked 174 (a)), F¹'s second replicate, the registration | **done 2026-11-24: A0 rebuilt 0 → 256 (0.98 % from the saved file at 256); B_K and BKO 256 → 384, H-K and G-K pass; K "the restart's" at all three τ arms; F¹ r2's references (44, 48], (44, 48], (48, 52], G-d1 passes on them; P-SCH7 (d1), P-SCH8 (d2) registered by the user before F½, F⅒** | `c4pdf_bk.py`; `data/trackC/c4pdf/2026-11-24-u3b/`; "C4′ (d)+(f) unit 3b" below |
| C4′ (d)+(f) 2b. F⅒, F½ (Blocked 175 (i) (a), the user's: budget extended to U2b, U3c + close-out) | **done 2026-11-25: both 0 → 1000, exit 0, 77 states each, all losses finite; step 1's loss equal to F¹'s; records 539 per arm, heads 77 states; the first launch was stopped by the user at Ft 481 and resumed from 480 with step 481's loss equal to every digit; nothing read. U3c (the d read) next** | `data/trackC/c4pdf/2026-11-24-u2b/`; "C4′ (d)+(f) unit 2b" below |
| Paper: every model read (the user's, 2026-11-24; writing only, outside any thread) | **done 2026-11-24: `paper/` §4.2 (table 2, every model's two dates; Dodecahedron refused), §5 "On a finer grid" (70m), §1, §3.1 and the abstract's scope; no number computed; `paper_check` 0, 38 pages; the rest is ROADMAP §8 C9** | "Paper: every model read" below |
| C4′ (d)+(f) 3c. the d read (P-SCH7, P-SCH8) | **done 2026-11-25: P-SCH7 does not hold (F⅒ has a transient, onset 784, at every gated arm); P-SCH8 does not hold (F½'s induction (304, 320], the step band); d1 joint "mixed" (F½ step, F⅒ neither); d3 "not step"; f re-read unchanged. Scoring sentences drafted, Blocked 177. Close-out next** | `data/trackC/c4pdf/2026-11-25-u3c/reading.json`; "C4′ (d)+(f) unit 3c" below |
| C4′ (d)+(f) close-out | **done 2026-11-25: writing only; closed at its exit and extended budget (7 units + close-out); five findings, P-SCH7 and P-SCH8 falsified; Parked carried to C8 or dropped; `PROJECT.md` §3.88** | "C4′ (d)+(f) on one page" below |
| C9 0. the scan and design (Blocked 176 (a), the user's) | **done 2026-11-25: writing only, nothing read; proposed, D1–D6 Blocked 178; U1 is a CPU read (every model's records exist); the 160m seeds downloading** | `lit-C9.md`, `design-C9.md`; "C9 unit 0" below |
| C9 1. the mean's term on every model (Blocked 178 all (a)) | **done 2026-11-25: known answer exact, plants exact on all 7 models; M "carries" on 6, "partly" on warmup01; the 160m download complete** | `c9_u1.py`, `data/trackC/c9/2026-11-25-u1/u1.json`; "C9 unit 1" below |
| C9 2a. the locator by grid, the weights rows and their gates (D1–D4, D6 confirmed by the user) | **done 2026-11-26: CPU, nothing located off 410m; known answer exact (A3's dates and floors; E2, E7, E8 from the weights); E2, E7, E8 gated on GPT-2 × 5, A0, A0′: all 21 pass, floor 4.0. U2b next** | `c9_u2.py`, `data/trackC/c9/2026-11-26-u2a/`; "C9 unit 2a" below |
| C9 2b. the GPU rows, the snapped gate and the pairs (Blocked 179 D7 (a), D8 (a), the user's) | **done 2026-11-26: every known answer exact; the snapped arm refuses every row on GPT-2 × 5 and 6 of 9 on each harness arm; one pair read (A0, E1–E3 aligned, replicates 410m's); Blocked 180** | `c9_u2b_pass.py`, `c9_u2b.py`, `data/trackC/c9/2026-11-26-u2b/`; "C9 unit 2b" below |
| C9 3. the 160m seeds' two dates (Blocked 180 (a); D5 (a), the user's) | **done 2026-11-26: tier 1, nothing registered; known answer bit-equal; gate passes; E₁ 256 on 10 of 10 seeds; centred 1000 at every arm on 6, 512–1000 across arms on 4; U1's M 0.927 "carries"** | `c9_u3_pass.py`, `c9_u3.py`, `data/trackC/c9/2026-11-26-u3/`; "C9 unit 3" below |
| C9 close-out, with U4 (Blocked 181 (i) (a), (ii) (a), (iii) (a), the user's) | **done 2026-11-26: closed at its exit and budget (5 units + close-out); `paper/` edited (table 2's 160m row and M column, §4, §5 "On other models", §7, the scope lines), 27 new slots, `paper_check` 0; `PROJECT.md` §3.89** | `numbers.py` (`c9u1`, `c9u2b`, `c9u3` records); "C9 on one page" below |
| C8 0. the scan and design (the user's, 2026-11-27) | **done 2026-11-27: writing only, nothing computed; proposed, D1–D5 Blocked 182; (i)'s rule fixed in full, (ii)–(v) in shape; budget 4 units + close-out** | `lit-C8.md`, `design-C8.md`, `tools/math_checks/c8_thresholds.py`; "C8 unit 0" below |

## C8 unit 0, the scan and design (2026-11-27)

Writing only. Nothing computed or trained, and no log value opened. Opened by the user
2026-11-27 ("U0, the literature scan and design ... candidate (i) first"; ROADMAP §8 C8).

- **Scan** `trackC_writeup/lit-C8.md` (nine searches, [S] only). Cohen et al.'s threshold form
  checked: (2 + 2β₁)/((1 − β₁)η), 38/η at β₁ 0.9 (`tools/math_checks/c8_thresholds.py` (1)). For
  (i), a grad-norm spike marks a stage jump (2508.12837), against the branches' seen spikes at
  the restart. For (v), Edelman et al.'s stages are in-context statistics, the ROADMAP's lean the
  corpus's, so U4 reads both. Parked: 2508.12837's and Cohen §3's full text, each before the
  output it bears on. 2602.15997's title in `paper/refs.bib` is v4's (`docs/readings/`); the
  search returned v1's, so it is not a defect.
- **Design** `trackC_writeup/design-C8.md`, proposed. **U1 (i)**: the loss-rise and grad-spike
  events in a 16-update lead window, on five conditions (F, F′, F½, F⅒, B_P), each against its
  own log's background (8–46 blocks, `c8_thresholds.py` (2)). Labels: "marked", "silent",
  "mixed"; the clip's share beside. The constants are placed on A0's updates 2001–4000. G-null
  and G-plant (on the controls' real logs) pass before any window of an arm with a transient is
  opened. F′'s and F's references are computed by rule T first. Every input is hashed. **U2
  (ii)** raw and preconditioned sharpness; v̂ from F², B_P², B_K² re-runs. **U3 (iv)** F_w (moments
  warmed by 64 updates at rate 0) and B_K↑ (a rate overshoot), with (viii). **U4 (iii) + (v)**.
- **Seen before the design:** F's log at 1, 4, 64; the branches' peaks at 257–259 and their
  values at 384; F⅒'s and F½'s at 1000. No window value is recorded in any file.
- **Not triggered:** the `c2_seeds` refusals (no centred arm derived); C9's max-over-heads note.
- **Corrects ROADMAP §8 C8's costing:** GPU ≈ 15–21.5 h, not ≈ 5. HDD_1TB has 37 GB free, so
  new outputs go to `Mets_hdd_storage/c8/`.

**Next:** Blocked 182, then U1 (CPU; fresh session). Parked: the scan's two.

## C9 on one page (closed 2026-11-26)

**Closed at its exit and its budget** (5 units + close-out): U0 the design (#380), U1 the mean's
term (#381), U2a the locator by grid (#382), U2b the GPU rows and pairs (#383), U3 the 160m seeds
(#384), and this close-out, which made U4's `paper/` edits (Blocked 181 (i) (a), the user's:
writing only, no new read). The exit's four parts: U1 read on every model in the design's table,
U2's pairs read or refused on GPT-2 × 5 and A0, A0′, U3's dates on the nine seeds, U4's edits.
Nothing registered (D5 (a)). GPU ≈ 1.5 h (U2b 75 min, U3 14 min) against the design's 4–6 h.

**What it established** (numbers in the unit sections; each is in `paper/` from a key):

| # | finding | where in `paper/` |
|---|---|---|
| 1 | **The count's drops are mostly the mean's falls on every model.** M (drops whose mean's term is over half the drop) is 0.83–0.96 pooled per model: "carries" on 410m, 160m (0.93 on ten seeds), 70m, GPT-2 small and the four 70m arms, "partly" on warmup01 (0.83, one run, inside 70m's seeds' 0.80–0.98; its count failed its gate, so the share describes undated drops, and the abstract's range, 0.88–0.96, leaves it out). At the break ‖m‖² falls at every counted drop on each of the 21 runs U1 dated (160m's seeds 1–9 were dated by U3 and not read for it); the mean's term is over half of each drop there on all of them but GPT-2 small's five (0.76–0.87). | table 2's last column; §4 "How much of V the mean's term carries", last sentences; abstract |
| 2 | **The timeline's order is unread off 410m at Pythia's resolution.** GPT-2 small: every row's snapped arm fails its gate, so no pair. Harness arms: one pair, A0's E₁–E3 aligned (384 → 512, 768 → 1000), replicating seed 0's verdict on unlike instruments (like for like on 410m, E3 is censored). | §5 "On other models"; §5 "On a finer grid"'s last sentence; §1, abstract |
| 3 | **160m's ten seeds: E₁ 256 on 10 of 10; centred 1000 on 6, 512–1000 on 4.** The four split seeds' arms at 512 are at 0.50–0.58 of plateau (the rest at most 0.48): about one drop at the bar. 410m's ten break at 512 × 8, 256 × 2: an unregistered contrast, size and peak LR confounded. | table 2's 160m row and note a; §4.2 "Read across the table"; §7 "Pythia 70m and 160m" |

**Not established.** Any cross-model comparison of dates (C9 reads orders within a model). The
order of the break, induction and the OV phase on any model. Whether A0′'s E₁–E3 pair or A0's E4
would read under an unbiased null (Blocked 180 (e), not taken). How stable each snapped verdict is
under the bootstrap ((d), not taken). Why 160m breaks a grid step before most 410m seeds, and
whether the four 512 arms are the seeds or the arms' scale. Whether warmup01's "partly" is its
warmup's. M's bar is placed, not calibrated, and the plants test the code, not M's power.

**Parked and carried, each carried or dropped:**
1. `lit-C9.md` Parked 1 (PolyPythias' full text): **done** in U3, before any seed 1–9 record.
   Parked 2 (2509.22831's full text): **done** in U2b, before its rows.
2. U1's Parked (GPT-2's below-410m M at its break, warmup01's late dip; a per-transition split of
   R against the mean at those states): its paper half is **done** here (§4 scopes the stronger
   clause to the Pythia-shaped runs). The split itself is **carried to ROADMAP §8 C5**, beside
   "whether R breaks on its own", which needs the same per-transition R.
3. U2a's Parked (E10 on GPT-2 from Mistral's logged LR): **done** in U2b (beside).
4. The two `c2_seeds` refusals (carried in from C4′ (d)+(f)): **carried to ROADMAP §8 C8** by
   Blocked 181 (ii) (a), the user's, ratifying U3's reversal: added before the freeze of any C8
   unit that derives a centred tolerance arm. This is their fourth home; none of the six designs
   that carried them derived an arm.
5. Blocked 180 (d) and (e), not taken (Blocked 180 (a), the user's): **dropped for C9** and
   stated as limits in §5. (e) is a defect of A3's synthetic null for any max-over-heads row, so
   it is **carried to ROADMAP §8 C9's row** as a standing note: fix it before any later gate of a
   max-over-heads row.
6. Blocked 177 (P-SCH7, P-SCH8 and the earlier entries' scoring sentences) is not C9's; it stays
   open in STATE until `git log` shows the user's registry commit. *Done 2026-11-26 on the user's
   instruction: all eight sentences in the registry (#386, `52470c8`).*

**Units at close:** 5 + close-out, the budget. Decisions: Blocked 178 (D1–D6; Claude on the user's
"merged, continue", confirmed by the user), 179 (D7, D8), 180 (a), D5 (a), 181 (i)–(iii) (a), all
the user's except 178's first pass.

**U4, as it ran (this close-out).** `numbers.py` gains three records (`c9u1` U1's `u1.json`,
`c9u2b` U2b's `reading.json`, `c9u3` U3's `reading.json`) and 21 keys; `paper_slots.json` 27
slots (as first pushed; after `/challenge-pr` 8 records, 29 keys, 29 slots, below). `produce` ran under the repo `.venv` (py3.14), which reproduces every existing key; conda
py3.10 refused on four keys' last ulp (the known interpreter noise). Table 2's centred 160m cell is
3b's frozen set reading with a note at the bar (Blocked 181 (iii) (a)). §7's "one seed per size
(… seeds of 70m and 160m, not read here)" was stale for both and is replaced. **Caught before
commit:** the first §4 sentence gave the other models' M as 0.83–0.94; the flat arms' is 0.96.
**Re-run:** `METS_DATA=<main>/data .venv/bin/python -m trackC_writeup.numbers produce`, then
`python -m trackC_writeup.paper_check` (0); compile `paper/` with `pdflatex main; bibtex main;
pdflatex main; pdflatex main`.

**After `/challenge-pr` on #385** (accept with changes; comment
https://github.com/ZachData/MetastableStateAnalysis/pull/385#issuecomment-7206591503, reviewed at
`cf8fc0a`). All four taken, in wording and checks; nothing read anew:

| # | finding | done |
|---|---|---|
| 1 | warmup01's count is "refused" in table 2 yet its M is printed and pooled into the abstract's range | caption: its share describes drops never dated; abstract's range is over models whose count was read (`c9u1.M_read`, 0.88–0.96); §4 names the failed gate |
| 2 | "every dated run of every model" at the break covers 160m seeds 1–9, which U1 never read at the break | narrowed to the 21 runs U1 dated (`c9u1.n_break_runs`), here, in §4 and in `PROJECT.md` §3.89; reading seeds 1–9 would be a new read, not made |
| 3 | the worded claims had keys but no `CLAIMS` entry; the GPT-2 row list was hand-picked | five `CLAIMS` added (at-break, labels, 160m's tallies and the bar, GPT-2's gates, the one pair); GPT-2's rows now come from each `gate/<seed>.json` (five records added: 8 records, 29 keys in all, 29 slots) |
| 4 | the abstract's new sentence did not parse and sat beside the other 0.96 | split into its own sentence, "a share of drops, not of transitions as above" |

## C9 unit 3, the 160m seeds' two dates (2026-11-26)

Tier 1, description. The decisions came first: **Blocked 180 decided (a)** (the user's: GPT-2's
refusal accepted as frozen, so U2 is done) and **D5 decided (a)** (the user's, before any seed 1–9
record: nothing registered). `lit-C9.md` Parked 1 was read before any record: PolyPythias names
no 160m outlier seed, so none is flagged. `design-C9.md` "U3 as it runs", the producer
`c9_u3_pass.py` and the reading `c9_u3.py` were pushed before any output (`9ec5282`). One gate
check was dropped before that push (the design says why).

**Input.** One GPU pass per (seed, step): `pythia-160m` seed 0 (again, the known answer) and
`pythia-160m-seed1`–`seed9`, Pythia's 18 steps, 7 v1 prompts, float32, tf32 off. That is 1260
records with 13 states each, producer code `d4510f86170e`, 14 min on the 3080. The arms are C2
3b's four centred τ (1.25e-4, 1e-4, 3e-4, 8.22e-4; derived and gated on 160m seed 0), not
re-derived. Reading at code `bf65c8396e27`. Records: `data/trackC/c9/2026-11-26-u3/`
(`seed{0..9}/records/`, `plan.json`, `known.json`, `gate.json`, `reading.json` sha256
`8a0540bc513b…`). *The reading was re-run once after CodeRabbit on #384: the producer-version
check now takes the union of codes (one code, `d4510f86170e`, on all 1260 records), and each
seed's count of gated, readable arms is kept. Every other value is identical to the first
reading (`4d324752a028`, sha256 `cebbfdec0a10…`).*

**Known answer** (`known.json`): seed 0's rerun against 3b's 126 records. Ids are equal, and
e1, e1c, e1r and cos are bit-equal (max relative difference 0, tolerance 1e-4). V_u and all
four V_c are equal, so the dates are 256 and 1000 × 4. 3b's gate passed all four arms.
**Gate** (`gate.json`, on seed 0's series): every check passes. The planted null reads inside
A5's window exactly at 256 and 512, of the 12 null steps. The planted effect at 1000 reads 1000,
not inside. Each arm's V_c moved one step earlier reads 512, readable.

**Reading** (per seed; the centred date is the set over the four gated, readable arms):

| seed | E₁'s break | plateau | centred date(s) | arms at 512 | V_c(512) / plateau at τ 1.25e-4 · 1e-4 · 3e-4 · 8.22e-4 (beside) | V_u(256) / plateau (beside) |
|---|---|---|---|---|---|---|
| 0 | 256 | 16 | 1000 | — | 0.42 · 0.41 · 0.41 · 0.26 | 1.12 |
| 1 | 256 | 20 | 512, 1000 | 1e-4 | 0.48 · **0.50** · 0.37 · 0.30 | 1.10 |
| 2 | 256 | 18 | 512, 1000 | 1.25e-4, 1e-4 | **0.54** · **0.54** · 0.42 · 0.29 | 1.06 |
| 3 | 256 | 19 | 512, 1000 | 1.25e-4, 1e-4 | **0.58** · **0.56** · 0.46 · 0.17 | 1.21 |
| 4 | 256 | 19 | 1000 | — | 0.31 · 0.30 · 0.29 · 0.29 | 0.63 |
| 5 | 256 | 24 | 1000 | — | 0.39 · 0.42 · 0.29 · 0.12 | 0.54 |
| 6 | 256 | 23 | 1000 | — | 0.25 · 0.25 · 0.26 · 0.27 | 0.61 |
| 7 | 256 | 22 | 1000 | — | 0.41 · 0.41 · 0.30 · 0.05 | 0.68 |
| 8 | 256 | 23 | 1000 | — | 0.26 · 0.26 · 0.26 · 0.25 | 0.57 |
| 9 | 256 | 15 | 512, 1000 | 3e-4 | 0.48 · 0.48 · **0.52** · 0.30 | 1.07 |

The two beside columns were added after `/challenge-pr` on #384 (findings 1, 4), post hoc, from
`reading.json`. Bold marks an arm at or over the half-plateau bar (`>=`).

- **E₁'s break is 256 on all ten 160m seeds.** None is unreadable: the step 0–128 counts are at
  most 2 against plateaus of 15–24. Table 2's 160m row becomes "256 (10 of 10)".
- **The centred date is 1000 at every arm on 6 seeds** (all four arms gated and readable on all
  ten seeds, `E1c_arms_read`). On the other 4, one or two arms read 512
  and the rest 1000, so the date is "512–1000, arms disagree". Every arm on every seed is later
  than that seed's break (C2 3s's event 1, 10 of 10). Event 2 (1000 at all four arms) holds on 6
  of 10.
- **U1's M on 160m pooled over the ten seeds: 0.927, "carries"** (bar 0.863; 2328 drops). Both
  plants are exact (1.0 and 0.0), R ≥ 0 (minimum 0.0043), and S_any is 0.983. Per seed M runs
  0.894–0.958, every seed above the bar. U1's seed-0 value was 0.911.

**Beside, never read** (D5 (a): no label): 410m's ten seeds break at 512 × 8 and 256 × 2, and
160m's ten at 256 × 10. Every 160m break is inside A5's window. Under 3b's schedule sets, every
seed's E₁ date is consistent with all three schedules. The centred date excludes
"instantaneous" on the six 1000-only seeds, and on the four with a 512 arm all three survive.
The difference from 410m is a description of an unregistered contrast on Pythia's
one-octave grid (C9 design D5 (c) was not taken), and size is confounded with the peak LR (6e-4
against 3e-4).

**Not established.** Why 160m breaks a grid step earlier than most 410m seeds: on this grid
"256" is a crossing in (128, 256]. It does not tell step, cumulative and instantaneous apart (3b's
predictions overlap). The 512 arms are at the two smallest τ (1e-4, 1.25e-4) on 3 of the 4
seeds and at 3e-4 on seed 9. Whether that is the arms' scale or the seeds is not read.
**The 512 / 1000 split sits at the half-plateau bar** (`/challenge-pr` on #384, finding 1). The
arms that read 512 have V_c(512) at 0.50–0.58 of their plateau, and seed 1's only 512 arm is
exactly 15 / 30, read 512 by `>=`. *(Corrected at the close-out: this said 0.48–0.58. 0.48 is
the largest of the arms on those seeds that read 1000, by `c9u3.at512_read1000_max`; the
table above was right.)* Seed 0's arms are at 0.41–0.42. The two smallest τ give
near-identical series (the same count at 512 on 8 of 10 seeds), so "two arms at 512" is about one
piece of evidence. So the 6 / 4 split is a one-count margin at a frozen threshold, not a
separation. *Post hoc, description only, 10 seeds* (finding 4): the four seeds with a 512 arm
(1, 2, 3, 9) are four of the five whose uncentred count is over its plateau at 256; the fifth
is seed 0. The other five are at 0.54–0.68. That points to faster seeds rather than the arms'
scale. It was looked at after the reading.
**The gate does not test that region** (finding 2). The "one step earlier" check puts V_c(512)
at about 0.85–1.0 of a moved plateau, so it passes whatever the reader does near 0.5. The
planted-null row at 4000 reads `None` and passes as "not inside". Both plants move the
reference they are read against, which is the case this unit's `LESSONS.md` line names. The
gate is frozen and not re-run. These limits bear mostly on the centred cells. E₁'s count at
256 is 0.54–1.21 of its plateau (seed 5's 0.54 is near the bar too, so 512 is one count away
there), and at 128 it is at most 1 count (≤ 0.05), so no seed is near reading 128.

**Carried in, not made:** the two `c2_seeds` refusals. U3 derived no arm, so neither applies. They go
to the close-out, which gives them a home (ROADMAP C8 or the next unit that derives an arm).
*This reverses the frozen design's "Carried in" ("U3 derives centred tolerances on 160m, so it
takes them"). That text did not leave the choice open, so the reversal is Claude's, not yours.
It parks the refusals a third time (`/challenge-pr` on #384, finding 3): STATE Blocked 181 (ii).*

**Re-run:** `METS_REPO=$PWD METS_DATA=<main>/data HF_HUB_OFFLINE=1 python -m
trackC_writeup.c9_u3_pass --out <dir>` (GPU, about 14 min; seeds 1–9 from
`WDS_500/Mets_hdd_storage/hf/hub`, seed 0 from `HDD_1TB/mets_data/hf/hub`), then
`METS_DATA=<main>/data python -m trackC_writeup.c9_u3 known --out <dir>` and `METS_DATA=<main>/data
python -m trackC_writeup.c9_u3 read --out <dir>` (CPU, seconds).

**Next:** U4 (`paper/`: table 2's 160m row, U1's column, §5's "on other models" paragraph with
Blocked 180 (a)'s wording, the scope line) and the close-out. The budget's five units are spent.
D6 (a) makes U4 a unit of its own, and the budget line counted it only under D6 (b), so **Blocked
181**: (i) U4 inside the close-out (writing only), or a budget extension; (ii) ratify or reverse
the "Carried in" line above; (iii) table 2's centred 160m cell. **Parked:** none new.

**After `/challenge-pr` on #384** (accept with changes; comment on #384, reviewed at `07eb3ce`):

| # | finding | done |
|---|---|---|
| 1 | the 512 / 1000 split is at the half-plateau bar (seed 1's arm exactly 15 / 30; the two smallest τ near-identical) | taken: the V_c(512) / plateau column and a "Not established" paragraph; table 2's cell is Blocked 181 (iii) |
| 2 | the gate does not exercise the region the reading turns on; the null's 4000 row passes on `None`; both plants move their reference | taken as a stated limit ("Not established"); the gate is frozen and was not re-run |
| 3 | the "Carried in" reversal is Claude's, against the frozen text | taken: marked as Claude's, put to the user (Blocked 181 (ii)) |
| 4 | the 512 seeds are four of the five at plateau by 256 (post hoc) | taken: the V_u(256) / plateau column and one post hoc line |

## C9 unit 2b, the GPU rows, the snapped gate and the pairs (2026-11-26)

Tier 1. The decision came first: **Blocked 179 decided D7 (a), D8 (a)** (the user's, before any
U2b output). `design-C9.md` "U2b as it runs", the producer `c9_u2b_pass.py` and the reading
`c9_u2b.py` were pushed before any of their output (`9eb0efc`). `lit-C9.md` Parked 2
(2509.22831's full text) was read before any record. It measures D2 (a)'s statistic, the max
over heads of the mean 1-back attention. It gives no onset for 70m, 160m or 410m, so nothing
moves.

**Input.** One GPU pass per (run, step) on 410m seed 0 (18 states), GPT-2 small × 5 (35 each)
and A0, A0′ (28 each): 251 records, producer code `49c1638e235c`, about 75 min on the 3080
(`pass/`). The weights rows (E2, E7, E8) are U2a's records (`2026-11-26-u2a/weights`).
E1 and E1c come from the paper's records: C2 3c (GPT-2 E₁ 250, 350, 250, 350, 350, with no
centred date), C4 U2 (A0: 384, 640) and C4′ (c) U3 (A0′: 320, 640). Gates and reading at code
`109a7e86d345`. Records: `data/trackC/c9/2026-11-26-u2b/` (`known.json`, `gate/<run>.json`,
`reading.json` sha256 `8f0b8ff2…`, `logs/`).

**Known answers** (`known.json`, every one before any gate). All pass:

| check | result | tolerance |
|---|---|---|
| U2a's known answer, its null rates equal (#382 finding 3) | pass | — |
| E5 on 410m against the reports' `MinRank` | 0.00499 (the reports' 2-decimal rounding) | 0.005 |
| E5's mean against the panel | 3.9e-4 relative | 5e-4 |
| E4 at `L5H2` against A3's `prev` | 5.8e-8 | 1e-3 |
| E3's mean over sequences against C4's heads files | 7.5e-8 (A0), 9.7e-8 (A0′) | 1e-5 |
| E6u, E6b, E6b′ against A3's records | ≤ 3.2e-14 nats | 1e-4 |
| prompt ids against each run's energy records | 0 mismatches of 7 runs × all steps × 7 | 0 |

**Gates (D7 (a): a row is gated only if U2a's own-grid arm and the snapped arm both pass).**
Entries are the snapped arm's minimum over gaps of "placed right" (bar 0.90). **P** = both
pass; **o** = the own arm fails too.

| run | E2 | E7 | E8 | E3 | E4 | E5 | E6u | E6b | E6b′ |
|---|---|---|---|---|---|---|---|---|---|
| 410m (bridge only) | | | | **P** 0.91 | **P** 0.90 | | | | |
| GPT-2 alias | 0.48 | 0.62 | 0.49 | 0.00 | 0.32 | 0.43 | | | |
| GPT-2 battlestar | 0.47 | 0.66 | 0.50 | o 0.00 | o 0.18 | o 0.42 | | | |
| GPT-2 caprica | 0.46 | 0.59 | 0.48 | 0.00 | o 0.11 | 0.44 | | | |
| GPT-2 darkmatter | 0.43 | 0.60 | 0.52 | o 0.00 | o 0.15 | o 0.44 | | | |
| GPT-2 expanse | 0.46 | 0.60 | 0.50 | 0.00 | 0.24 | o 0.43 | | | |
| A0 | **P** 0.94 | **P** 0.91 | 0.88 | **P** 0.92 | 0.65 | o 0.45 | 0.02 | 0.04 | 0.03 |
| A0′ | **P** 0.94 | **P** 0.91 | 0.898 | 0.84 | o 0.73 | o 0.47 | 0.02 | 0.03 | 0.03 |

**On GPT-2 every row is refused, so no pair is read on any GPT-2 seed.** The snapped arm's
failures are at the gaps that straddle a snap boundary. For example, in 450–500, 450 and 500
snap to 128 and 550 to 256. A 0.25-octave change spans several of GPT-2's 0.13–0.15-octave
gaps, so noise moves the located step by one own-grid step, and that step snaps to either side
of the boundary. This is what D7's arm was added to detect, and it fires: **at the locator's
power, a GPT-2 date cannot be carried onto Pythia's grid to one Pythia step.** E3's 0.00 is at
450–500 on every seed whose own arm passes. Its cause is the null's bias for a max over heads
(finding 1 below). On the harness arms, E6's
refusals say more about the plant than the rows. A3's fixed dip, with its rise one gap later,
has a bottom several steps wide on C4's fine grid, so argmin is noisy (0.02–0.04 even after the
fix below). E8 misses by 0.02 and 0.002.

**Corrected after the first gate pass, before the reading** (`7459cd1`; the first pass kept as
`gate-v0/`): the snapped arm took the planted step as `i + 1`. For the fixed dip on a
sub-octave grid that is wrong (48–64's dip bottoms at 80), so E6 read 0.000. Now the arm
locates the noise-free plant and snaps that (`LESSONS.md` 3). Half rows were unchanged, so every
GPT-2 entry above is the same in both passes.

**Reading** (`reading.json`):

| run | located (own step → snapped) | pairs read | replicates 410m's? |
|---|---|---|---|
| A0 | E1 384 → 512; E3 768 → 1000; E2 "no change" (a dip 0.514 → 0.40 at 1000–1250 and back, 1.6 σ over 48 heads, floor 4); E7 1500, censored | **E1–E3 aligned** (one Pythia step; three own-grid steps apart, E1 first: the finer order §5 already prints) | **yes**, against A3's named-head E3 (another instrument; finding 2 below) |
| A0′ | E1 320 → 512; E2 "no change" (2.2 σ); E7 censored | none | — |
| GPT-2 × 5 | E1 only (from C2 3c) | none | — |

The bridge, beside: 410m with D2 (a)'s instrument gives E4 1000 (as the named head) and E3 1000
censored (its reference at 4000, where the named heads' is not). So the bridge reads 6 of the 9
pairs A3 reads (the three with E3 unread) and agrees on all 6. E10, beside: A0, A0′ 768 → 1000; GPT-2 (50, 100],
which snaps to 16 or 32.

**Not established / weak.**
- **Whether the timeline's order holds off 410m.** On GPT-2 it is unread: refused by the gate
  you chose, not by a contrary reading. On the arms it rests on one pair on one run.
- The snapped arm uses A3's fixed width and amplitude on grids it was not built for. On GPT-2
  that is the test you asked for. For the turn rows the plant is a weak one, so E6's refusal
  says little about E6.
- E2's "no change" on 70m means not at the floor over 48 heads. It does not mean no OV phase
  (the median dips by 0.11).
- E3, E4 by D2 (a) differ from 410m's named-head rows. The bridge measures that on 410m only.
  **The one "replicates" compares two instruments** (`/challenge-pr` on #383, finding 2). A0's
  E3 is the max over heads; the reference is A3's named heads. With the same instrument on 410m
  (the bridge), E3 is censored, so like for like the pair is unread.

**After `/challenge-pr` on #383** (accept with changes; comment
https://github.com/ZachData/MetastableStateAnalysis/pull/383#issuecomment-7202347427):
- **Finding 1, the null is biased for the max-over-heads rows (taken as a finding, not fixed:
  that would change a gate after its output).** A3's synthetic null gives every unit zero noise
  at step 0. For E3 and E4's statistic, the max over 48 or 144 tied noisy heads is therefore
  positive at every later step: 1.2–3.2 SE on alias's E3, about 2.4 on A0's E4, by the
  reviewer's rebuild. That moves a smooth plant's located step one step late. The rebuild (no
  floor, approximate SEs; not re-run here) gives:

  | row | recorded | rebuild, biased | rebuild, unbiased |
  |---|---|---|---|
  | GPT-2 E3, gap 450–500 | 0.00 | 1 of 300 | 0.64 |
  | A0 E4 | 0.65 | 0.67–0.74 | 1.00 |
  | A0′ E3 | 0.84 | 0.87 | 1.00 |

  So the cause of E3's 0.00 is this, and it plausibly decides A0's E4 and A0′'s E3 refusals
  (A0′'s E1–E3 would be the second arm's pair). The GPT-2 rows stay refused either way (E2, E7,
  E8 do not use the max). It is in Blocked 180.
- **Finding 2:** the caveat above, also in STATE and the PR title.
- **Finding 3:** Blocked 180 gains (d), the stability of each pair's snapped verdict under the
  bootstrap. That tests what D4 reads, rather than every gap. Beside it: GPT-2's own E₁ dates
  (250, 350) already snap to two steps (64, 128).
- **Finding 4, the snapped arm's known answer, run post hoc** (`c9_u2b_snapcheck.py`,
  `snapped_known.json`). On Pythia's grid, on A3's units at A3's floors, it gives A3's recorded
  effect rates to ≤ 0.013 (E2 0.002, E5 0.013, E6u 0.003; fresh draws; tolerance 0.03, placed
  at about 3 binomial SE). It would not have caught `7459cd1`'s defect, which needs a grid
  finer than an octave.
- **Finding 5:** three verdicts sit within one Monte Carlo SE (≈ 0.0095 at n = 1000) of the
  0.90 bar: 410m E4 0.904, 410m E3 0.914, A0′ E8 0.898. The seed decides them.

**Re-run** (GPU ≈ 75 min, then CPU ≈ 40 min with 8 processes): `c9_u2b_pass --run <run> --out
<dir>` for the 8 runs (HF_HOME: `data/hf` for 410m and the arms' tokenizer, `HDD_1TB/mets_data/hf`
for GPT-2), then `c9_u2b known`, `c9_u2b gate --run <run>` × 8, `c9_u2b read`; tests
`tests/test_trackC_c9_u2b.py`.

**Next: Blocked 180** (yours: what U2 does with GPT-2's refusal), then U3 (the 160m seeds).
**Parked:** none new. U2a's E10 item is done (beside, from the logged LR).

## C9 unit 2a, the locator by grid, the weights rows and their gates (2026-11-26)

Tier 1; CPU, no forward pass; **nothing located off 410m** (a date there is U2b's reading).
**D1–D4, D6 confirmed by the user** (2026-11-25, after U1's output: `/challenge-pr` on #381,
finding 3). The split of U2 into U2a (CPU) and U2b (GPU), the grids, windows, axis, plant width
and seeds were placed in `design-C9.md` "U2a as it runs" and pushed with `c9_u2.py` before any
output (`09c9117`). One change after the first record and before the known answer ran
(`fe1bd4c`): (ii) compares E7, E8 at relative 1e-9, not exactly, since the first 410m record's α
differed from `a3_weights`' by 1e-13 relative (thread count, not the code).

**Input.** Checkpoints: 410m seed 0's 18 (`data/hf`), GPT-2 small × 5 on C2 3c's 35 tags
(`HDD_1TB/mets_data/hf`), A0 (`HDD_1TB/mets_data/c4`) and A0′ (`…/c4pc`) on C4's 28-step grid.
A3 unit 5's record `data/trackA/a3/2026-10-20-A3u5/a3_rows.json` and its inputs (the known
answer). Weights records by `c9_u2.py` at code `a3eedc0b9808` (249: 18 + 175 + 56, each run's
W₀ its own step 0); known answer and gates at `c0cd011e8b22`. Records in
`data/trackC/c9/2026-11-26-u2a/` (`weights/`, `known.json`, `gate/<run>.json`, `logs/`).

**Known answer** (`known.json`): (i) the grid-parametrised gate and locator on A3's own units, in
A3's order and seed, give every date, censoring and floor of A3 unit 5 (E2 512, E3 1000, E4 1000,
E5 8, E6u 128, E6b 256, E6b′ 256, E7 2000 censored, E8 1000, E11 refused). These are the first
dates of the gated rows; E8's second date (from its peak) and the offset-0 E3 are not replayed.
The null rate is equal on every row too (`null_no_change_equal`), but it is recorded, not part
of the pass flag (`/challenge-pr` on #382, finding 3; folded into the flag in U2b). (ii) The weights producer on 410m seed 0 gives `a3_weights`' E7, E8 to 4.4e-14 and
2.5e-13 relative and `ov_per_head_series.json`'s per-head E2 to 5.8e-8 absolute (bar 1e-6);
swapped into (i), no date moves. **Pass.**

**Gates** (A3's rule; plant width a quarter of its gap; null ≥ 95 %, effect ≥ 90 % in every gap):

| run | grid (gaps) | E2 own floor / null / effect min | E7 | E8 |
|---|---|---|---|---|
| GPT-2 alias | 35 (33) | 3.24 / 0.992 / 0.993 | 2.72 / 0.999 / 0.983 | 3.11 / 0.999 / 0.986 |
| GPT-2 battlestar | 35 (33) | 3.23 / 0.993 / 0.994 | 2.68 / 1.000 / 0.979 | 2.99 / 0.996 / 0.989 |
| GPT-2 caprica | 35 (33) | 3.17 / 0.992 / 0.996 | 2.68 / 0.999 / 0.972 | 3.22 / 0.989 / 0.992 |
| GPT-2 darkmatter | 35 (33) | 3.19 / 0.994 / 0.992 | 2.52 / 0.998 / 0.978 | 3.18 / 0.996 / 0.992 |
| GPT-2 expanse | 35 (33) | 3.25 / 0.994 / 0.995 | 2.66 / 1.000 / 0.978 | 3.10 / 0.997 / 0.991 |
| A0 | 28 (26) | 2.73 / 0.995 / 0.994 | 2.78 / 0.995 / 0.982 | 3.09 / 0.992 / 0.985 |
| A0′ | 28 (26) | 3.03 / 0.998 / 0.993 | 2.79 / 0.999 / 0.983 | 3.15 / 0.994 / 0.980 |

Every own floor is below 4, so the floor used is 4.0 everywhere (A3's max(4, own)). **All 21
pass.** Power (the least amplitude reaching 90 %) is 4–9 σ̄_ref by gap, and 6–9 for GPT-2's E8
on 3 seeds. Units: 144 heads (GPT-2) and 48 (70m) for E2; 12 and 6 layers × 4 types for E7, E8.

**Not established / weak.** The gates test the rule against flat noise at each run's own
spread. They say nothing of the rows' power against a real change's shape. The plant width (a
quarter of the gap) is placed. On GPT-2's finest gaps (0.13 octave, 1100–1200) it plants a near-step,
the easiest change the locator can place. The gate scores the own-grid step, but D4 reads the
step snapped onto Pythia's grid. So a pass there is about resolving the grid, not about a
timescale, and not about what U2b reads (`/challenge-pr` on #382, finding 1: **Blocked 179**,
yours, before U2b). The producers' logs print each step's E2 median. They were seen at the first steps
while checking the output was populated; no row was located or plotted.

**Re-run** (CPU, ≈ 25 min with 8 processes): `c9_u2 weights --run <run>` for the 8 runs (HF_HOME
as in `c9_u2.py`'s docstring) **into a fresh `--out`**, or at `09c9117`. The records' code hash
covers all of `c9_u2.py`, so at a later commit `weights` refuses to resume into this unit's
dir (finding 2; the hash narrowed to the producer in U2b). Then `c9_u2 known`, then `c9_u2 gate --run <run>`; tests
`tests/test_trackC_c9_u2.py`.

**Next: U2b** (the GPU rows: E3, E4 by D2 (a) with per-sequence units, E5, E6u/E6b on A0, A0′;
each producer's known answer on 410m seed 0 first; then the reading, after `lit-C9.md` Parked 2;
fresh session; **waits on Blocked 179**, D7 the gate's snapped arm and D8 the snap's units).
`/challenge-pr` on #382: accept with changes, findings 2–5 taken (re-run line, known answer
wording, the copied grids tested against their sources, the finest gap 0.13 octave), 1 to the
user. **Parked:** E10 on GPT-2 is Mistral's logged schedule
(`gpt2_small_lr_log.json`), not analytic: why / the confound row needs a series on 35 tags /
cost minutes / could change which pairs carry "inside warmup" on GPT-2. Carried to U2b.

## C9 unit 1, the mean's term on every model (2026-11-25)

Tier 1, description under D1 (a)'s placed label; no forward pass. **Blocked 178 taken all (a)** by
Claude on the user's "merged, continue" (yours to overrule); `design-C9.md` frozen ("Decided
2026-11-25": the models, M₄₁₀'s pooling, plants held at the series' first state, S_break's
dates) and `trackC_writeup/c9_u1.py` pushed before any output (`5dd99ca`). The first launch
refused at step (1), before anything was computed: its check wanted one `n` per run, but `n` is
each prompt's token count; fixed to one `n` per prompt across steps (code `81578cbf7a39`).

**Input** (each run's sha256 in the record): 410m `c2u3` seed 0 + `c2u3s` seeds 1–9; 160m `c2u3b`;
70m C6 U0 seeds 0–9; warmup01 C6 U1b; GPT-2 small `c2u3c` × 5 (35 steps); A0, F `c4/2026-11-12-u2`,
A0′, F′ `c4pc/2026-11-22-u3` (29 steps). Every (step, prompt) record present on every run.

**Gate.** Known answer: record by record U1's series equals `c2_readings.one_series`, and on all 18
steps C2 5b's R6 shares equal U1's exactly; seed hashes equal C2 5b's inputs. **M₄₁₀ = 0.9628**
(5438 of 5648 drops), so the bar is **0.8628**. Both plants exact on all 7 models (mean held:
M = S_any = 1; ‖m‖² held: M = 0, "not"); R ≥ 0 everywhere (min 1.6e-3). None refused.

| model | runs | drops | **M** | label | S_any | per run M | M at the break (drops) |
|---|---|---|---|---|---|---|---|
| `pythia-410m` | 10 | 5648 | 0.963 | carries | 0.988 | 0.939–0.979 | 1.0 on all 10 (29–58) |
| GPT-2 small | 5 | 4823 | 0.944 | carries | 0.988 | 0.931–0.952 | 0.76–0.87 (27–38) |
| 70m flat (F, F′) | 2 | 539 | 0.959 | carries | 0.982 | 0.964, 0.954 | 1.0, 1.0 (8, 7) |
| 70m harness (A0, A0′) | 2 | 152 | 0.941 | carries | 0.980 | 0.920, 0.961 | 1.0, 1.0 (4, 4) |
| `pythia-160m` | 1 | 213 | 0.911 | carries | 0.986 | 0.911 | 1.0 (18) |
| `pythia-70m` | 10 | 828 | 0.883 | carries | 0.961 | 0.802–0.981 | 1.0 on seed 0 (3); others undated |
| `pythia-70m-warmup01` | 1 | 109 | **0.826** | **partly** | 0.964 | 0.826 (inside 70m's seeds' 0.802–0.981) | no E₁ date (refused) |

At the break ‖m‖² falls at every counted drop on every dated run of every model (1.0; the
abstract's sentence); the column above is M, §4's stronger clause ("more than half the drop").

Beside, never read (all in `u1.json`): the base rate (‖m‖² falls at 0.06–0.34 of all
transitions), per-step M (lowest 0.59, 70m at 143000 on 56 drops; warmup01 0.71–0.75 at
8000–54000 on 11–20 drops a step), per-step S_any (lowest 0.80, warmup01).

**What it says.** On every model but warmup01, the drops V counts are mostly the mean's falls at
about 410m's rate. The paper's S_any holds pooled on every model (0.961–0.988), as the identity
nearly forces. At the break, the abstract's "every drop the count counts is a layer at which
the mean's norm falls" holds on every model (1.0 on every dated run); §4's stronger clause, that
the mean's term is more than half of each drop there (M = 1 on every Pythia-shaped run), does not
hold on GPT-2 small (M 0.76–0.87 at its breaks). Corrected after `/challenge-pr` on #381
(finding 1): the first push pinned GPT-2's at-break result on the abstract's sentence.

**Not established / weak.** The bar is placed (0.1), not calibrated. 70m clears it by 0.02, and
4 of its 10 seeds (0.80–0.86) fall below it alone. warmup01's "partly" rests on one run and 109
drops, concentrated where its LR is still warming (to 14 300) and after; its 0.826 lies inside
`pythia-70m`'s own seeds (0.802–0.981, three of them below it), so the label difference is one
run against a pool of ten, not shown to be the longer warmup's (`/challenge-pr` on #381,
finding 2). The gate checks the code only: the plants are exact by construction, so they test
nothing of M's power (finding 4). At-break values on 70m
and the harness arms rest on 3–4 drops. M pools every checkpoint, so it says nothing about when.

**Re-run:** `METS_REPO=$PWD METS_DATA=<main>/data python -m trackC_writeup.c9_u1` (seconds, CPU);
tests `tests/test_trackC_c9_u1.py`.

**The download** (Blocked 176) finished 2026-11-25: 162 of 162 (9 seeds × 18 steps), every
`pytorch_model.bin` present at 375 036 845 bytes, 57 GB in `Mets_hdd_storage/hf/hub`; still not
linked into HDD_1TB's hub (U3 does it).

**Next:** U2a (A3's rows generalised by model and grid, their gates and the known answer), a fresh
session. `lit-C9.md` Parked 2 before U2b's rows are read. **Parked:** GPT-2's below-410m M at
its break (0.76–0.87) and warmup01's late-step dip are discoveries: why / a per-transition
split of R against the mean at those states, minutes / could change whether §4's "more than half
the drop at all but one" is scoped to Pythia-shaped runs (U4's wording; the abstract's sentence
holds on every model).

## C9 unit 0, the scan and design (2026-11-25)

Writing only; nothing read. Opened by Blocked 176 (a), the user's: C9 after C4′ (d)+(f)'s
close-out, U0 before any read.

- **Scan** `trackC_writeup/lit-C9.md` (abstracts and snippets; no full text). Nothing scooped.
  Two inputs: heads move between indices over training (2407.10827), so heads are picked per
  model by a rule; and induction's step follows batch × context, not size (2511.16893), so C9
  reads orders within a model, not dates across models. Parked: PolyPythias' and 2509.22831's
  full text, each before the records it bears on.
- **Design** `trackC_writeup/design-C9.md`, proposed. U1 is the mean's term on every model, a CPU
  read: every model's records hold `e1`, `cos` and `n` (one record per folder checked, and the
  file counts). This corrects the ROADMAP row's "GPU hours". U2 is A3's eleven pairs on GPT-2
  small × 5 and the harness arms A0, A0′, with the locator gated on each grid. U3 is the 160m
  seeds' two dates. U4 is the paper. Budget 5 units + close-out, GPU ≈ 4–6 h.
- **Decisions D1–D6, Blocked 178** (recommended: all (a)).
- **`/challenge-pr` on #380: accept with changes, findings 1–7 taken in the design before any
  decision.**
  1. "Aligned" now snaps each date onto Pythia's grid, then applies A3's one-step rule. A factor of 2 would have flipped §5's finer-grid pair.
  2. U1 reads the magnitude share M. S_any ≥ 0.96 is nearly forced by E₁ = ½ + ½‖m‖² + R, so no shuffle is a null for it.
  3. No registration is recommended: the 256-or-512 window is already near-settled.
  4. Heads are read by the max over heads, the paper's harness-arm rule. A5's D10 rule would have given a second induction date.
  5. `one_series` hard-codes 25 states, so U1 takes the state count from the record.
  6. The 0.96 is labelled 410m's pooled minimum, and the duplicate 70m row is merged.
  7. Another design's decisions are named with that design.
- **The download (Blocked 176):** `Mets_hdd_storage/c9_download.py`, the nine
  `pythia-160m-seed{1..9}` repos at A5's 18 steps (`config.json`, `pytorch_model.bin`, 375 MB
  each, ≈ 61 GB), into `Mets_hdd_storage/hf/hub`. It is idempotent and stops past 240 GB. Its
  first launch's size guard read 0 GB (`du` on the symlinked root measured the link); stopped
  after 5 files, fixed (`du -sb root/`), relaunched. Log `Mets_hdd_storage/c9_logs/download.log`.
  Not yet linked into HDD_1TB's hub (U3 does it).

**Next:** Blocked 178, then U1. No Parked items beyond the scan's.

## C4′ (d)+(f) on one page (closed 2026-11-25)

**Closed at its exit and at its extended budget** (Blocked 175 (i) (a)): U0 the design (#371),
U1 the rules, gates and F¹ (#372), U2a (f)'s branches (#373), U3a f (#374), U3b B_K and the
registration (#375), U2b F⅒ and F½ (#377), U3c d (#378), and this close-out. d1, d2, d3 and f are
each read by their rules (the exit), and K was added beside. P-SCH7 and P-SCH8 were registered
before either arm trained, and both are falsified. The rest is tier 1. 70m only, seed 0's init,
C4's order, one run per arm.

**What it established** (numbers in the unit sections):

| # | finding | lean |
|---|---|---|
| 1 | C4's harness does not reproduce F's weights by update 4 (cause untested). F¹ is a near-replicate: its transient starts at (44, 48] / (48, 52] / (48, 52] on the three gated τ arms, and a second replicate at (44, 48] / (44, 48] / (48, 52]: one grid step of spread at 1e-4. | H-d failed (gate row) |
| 2 | On A0's trained state at 256, a fresh AdamW at the peak makes the transient (onset 16–32 updates after, on F's transitions T3–T5). The same restart at A0's own rate (1.79e-4 → 3.57e-4) makes none **within its 256 updates**, a summed rate (≈ 0.07) below F⅒'s at its onset (0.075–0.078). | f "not the early state's"; f-state broken |
| 3 | On that trained state, the moments kept make no transient at the peak (B_K) or at A0's rate (BKO) within 128 updates. **On a trained state**, then, the transient came only with fresh moments at the peak, and fresh moments also make the first updates larger (the two not separated). From init it is not so tied: F⅒'s comes at 1e-4, below B_O's rate, after ~750 updates, long after Adam's moments (β 0.9, 0.95) have adapted. So no single condition yet covers every arm (`/challenge-pr` on #379; the first push said "every arm with it had un-adapted moments meet a high rate", written before F⅒ was read). | K "the restart's"; f-jump broken |
| 4 | **P-SCH7 falsified.** A flat rate at half the peak has the transient, labelled step beside (onset (56, 60] to (60, 64]). So does a tenth, at (752, 784], past its dose band, small (a sixth of F½'s size) and cut off at 1000. | d1, d1′ broken; joint "mixed" |
| 5 | **P-SCH8 falsified.** At half the peak, induction reaches the bar at (304, 320], F's update count (the step band), not the dose band. At a tenth it has not formed by 1000. | d2 broken; d3 held ("not step") |

**For C4's item 7 ("the flat arm's date is the onset of a transient").** The transient is not
the early state's, since a trained state shows it after a fresh Adam at the peak. It is not the
rate's jump alone, since kept moments at the peak show none within 128 updates. It is not confined
to the peak or to fresh moments, since from init it appears at half and at a tenth, at a tenth
only after ~750 updates. Its onset is near F's update count at half the peak and far
past the dose clock at a tenth. So neither a step clock nor a dose clock dates it across rates,
and what triggers it is ROADMAP §8 C8's question.

**For `lit-C4.md` Parked 1 (does a lower flat peak form the heads).** On 70m at half the peak,
induction forms at F's update count (320), with its top head in layer 3 (F's in layer 2, A0's in
3). At a tenth it has not formed by 1000. Qu & Liu's non-forming plateau was at a higher peak
(6e-3), so this does not test their case. It says only that halving 70m's peak does not delay
induction.

**For `paper/` §7** (the user's rewrite; not edited here). §7's second registered bullet
(P-SCH4, held) says the flat and harness arms' induction dates (320, 768) lie "nearer the summed
rate than the centred date is". d2 compares two flat rates, not those two arms, so it does not
contradict the bullet. It limits it: between the peak and half the peak, induction keeps the
update count (`/challenge-pr` on #379, finding 3; the first push called this a tension). A
sentence on it is yours to place, if any.

**Not established.** What the transient is at a tenth of the peak: no threshold, an effect that
weakens with the rate, or a much slower clock (its run is never seen ending). That B_O, B_K or
BKO have none past their 256 or 128 updates: F⅒ needed ~750 (finding 1). Whether moments matter
as such or only through the first updates' size (C8 (iv)). Any of this at init with kept moments.
Anything on 410m.

**Parked and carried, each carried or dropped:**
1. `lit-C4pdf.md` Parked 1 (Gupta et al.'s full text): **done** in U3a, before f's read.
2. U3a Parked 1 (the reading names its input): **done** in U3b. Its remainder, that the reader
   records paths and not the input files' hashes (`/challenge-pr` on #378, finding 3): **carried
   to ROADMAP §8 C8**, whose new reader should hash its inputs (`status-C.md` "unit 3c" holds the
   hashes by hand).
3. F⅒ continued from its `resume.pt` at 1000 (why: to see whether its transient ends, and on
   which transitions; cost ≈ 7.7 h at 1e-4 to 2000; could change: finding 4's "not established").
   **Carried to ROADMAP §8 C8**, unscheduled.
4. B_K with a rate overshoot sized to B_P's early drift: already **C8 (iv)**.
4b. B_O continued from its `resume.pt` at 512 at A0's rate (why: its "none" stops at 256 updates,
   below F⅒'s onset dose; cost ≈ 7 h to 1000 (its 256 updates took 2.0 h); could change finding 2's
   "the same restart at a low rate makes none"): **carried to ROADMAP §8 C8 (viii)**
   (`/challenge-pr` on #379, finding 1).
5. The two `c2_seeds` refusals (C2 3c's Parked 0; carried as C4 U1's): **carried to ROADMAP §8
   C9**, not dropped as first pushed (`/challenge-pr` on #379, finding 2): `b_matched_tol` is still
   called by `s4.py`, `t2.py` and `t5e.py`, and C9 may derive tolerance arms on new models. None of
   the five designs that carried them (C4, C4′, (c), (e)+(g), (d)+(f)) derived a centred arm.
6. Blocked 177 (P-SCH7 and P-SCH8's scoring sentences, with P-SCH1's and P-SCH2–6's, none yet
   in the registry): **given a home**. It is the user's commit and stays open in STATE until
   `git log` shows it. *Done 2026-11-26 on the user's instruction (#386, `52470c8`).*

**Units at close:** 7 + close-out, the extended budget (the design's 4, plus U3b on Blocked 174 and
U2b, U3c on Blocked 175). GPU ≈ 25.5 h by the logged walls: B_P, B_O 4.1 h, the A0 rebuild 2.0 h,
B_K, BKO 2.0 h, F½ 7.7 h, F⅒ ≈ 7.7 h over two segments, F¹ ≈ 62 min and its replicate 3649 s.
The design's figure was 19.4 h.

**Re-run:** each unit's section below names its commands. The reads are JSON only:
`c4pdf_read read` (d and f, U3c), `read-k` (U3b).

## C4′ (d)+(f) unit 3c, the d read (2026-11-25)

Under P-SCH7 (d1) and P-SCH8 (d2), each scored by its frozen rule. Code at `f4ecb2c`, unchanged
since registration: `c4pdf_rules` `284f0d10f5ba`, `c4pdf_train` `84662428cf73` (both checked
against the entries before the read). Input: the gate `data/trackC/c4pdf/2026-11-23-u1b/gate.json`
(sha256 `078bf9315744`, pass, gated τ arms 9.61e-5, 1e-4, 3e-4); F¹'s records and heads
(`2026-11-23-u1/`), B_P's and B_O's (`2026-11-23-u2/`), F½'s and F⅒'s (`2026-11-24-u2b/`), each
linked into `2026-11-25-u3c/`; checkpoint root `HDD_1TB/mets_data/c4pdf` (the reader checks each
arm's states are all there). Output `2026-11-25-u3c/reading.json`, sha256
`16150f06cc1b8a682eef49266d9911345ab142ef6b70baf500f03869d28acccd`, made read-only after the read
(a re-run to the same folder now fails rather than overwrite it; `/challenge-pr` on #378, finding 3).
`c4pdf_read read`, conda `mets`, CPU, JSON only, seconds.

The input's hashes (the reader records paths only; finding 3). Heads, sha256 of the file; records,
sha256 of the sorted `sha256sum` listing of the arm's folder:

| arm | heads | records (files) |
|---|---|---|
| F¹ | `9860afc86bac` | `e00a680d20a2` (231) |
| F½ | `cab13fbe4e09` | `9b82c9afcfb7` (539) |
| F⅒ | `317689664d38` | `9b628eef3a05` (539) |
| B_P | `393a2152ca66` | `c2e8812621f1` (119) |
| B_O | `1f3c6df41a88` | `9dc75a302f35` (119) |

**d1 (P-SCH7)**, against F¹'s references (rule T; an onset read at x after p lies in (p, x]):

| τ arm | F¹'s reference | F½ (Fh) onset, run | label | F⅒ (Ft) onset, run | label |
|---|---|---|---|---|---|
| 9.61e-5 | (44, 48] | (56, 60], 60 → 640 | step [35.2, 60] | (752, 784], 784 → 1000 | neither (dose [352, 600]) |
| 1e-4 | (48, 52] | (56, 60], 60 → 592 | step [38.4, 65] | (752, 784], 784 → 1000 | neither (dose [384, 650]) |
| 3e-4 | (48, 52] | (60, 64], 64 → 368 | step [38.4, 65] | (752, 784], 784 → 1000 | neither (dose [384, 650]) |

- **P-SCH7 does not hold; its falsifier is met at every gated arm:** F⅒ has a transient by 1000.
  F½ has one, as stated. The entry's h0 (presence does not change between a tenth and a half)
  is what the arms show: both have one.
- **Joint reading: "mixed"** at every arm (F½ step, F⅒ neither). So the lean (a threshold between
  a tenth and a half) is broken, and so is d1′ (F½ in the dose band).
- F½'s step label at 9.61e-5 sits on the band's top edge: (56, 60] against 60.0. One grid step
  later would read "neither".
- F⅒'s onset comes after its dose band ends, and its run reaches 1000 (the horizon), so its end
  and size are cut off.

**d2 (P-SCH8) and d3**, the bar b = 0.4819 on the top head's induction score:

| arm | date (p, x] | label | top score at 256 / 304 / 320 / 1000 | top head at 1000 (beside) |
|---|---|---|---|---|
| F½ | (304, 320] | **step** [204.8, 400] | 0.087 / 0.388 / 0.526 / 0.934 | L3H1 |
| F⅒ | not reached by 1000 | **not step** (d3) | 0.018 / 0.017 / 0.016 / 0.057 | L3H1 |

- **P-SCH8 does not hold; its falsifier is met:** (304, 320] lies inside the step band. F's own
  date is (256, 320], so at half the peak, induction forms at F's update count.
- The lean d3 (F⅒ not formed by 1000) held. That is "not step", not "dose": F⅒'s dose band
  [2048, 4000] is past the horizon.

**f, re-read on the same input:** "not the early state's" at all three arms, as U3a (in
`reading.json` beside d; U3a's `2026-11-24-u3/reading_f.json` is untouched).

**Beside (description, not read):**

| | F¹ / F | F½ | F⅒ |
|---|---|---|---|
| transient onset × rate (summed rate at onset, peak 1e-3; F¹'s references) | 0.044–0.052 | 0.028–0.032 | 0.075–0.078 |
| transitions majority-violated in the run (states) | T3–T5 | T3 (32–46), T4 (7–16), T5 (15–17) | T4 (8 of 8), T3 (4 at 9.61e-5 and 1e-4, none at 3e-4) |
| size (summed over the run) | | 485 / 468 / 335 | 78 / 78 / 54 (cut at 1000) |
| training loss at the induction date | F 4.13 (A0 4.02) | 4.40 at 320 | — (4.61 at 1000) |

- Neither the update count nor the summed rate lines the three onsets up. F½'s comes at F's
  update count, near its summed rate times 0.6. F⅒'s comes at about 1.5 times F's summed rate.
- In F, the transient (48) comes long before induction (320). In F⅒, it starts at 784, while the
  top head's score is just leaving its floor (0.021 at 784, 0.057 at 1000).
- F½'s loss at its induction date is 4.40, outside the 4.0–4.2 that a loss clock predicted (F
  4.13, A0 4.02).

**What this does not establish.**

- There is one run per rate, and F¹'s near-replicate references are one grid step apart
  (U3b). F½'s step label at 9.61e-5 is on the band's edge.
- Above 656, F⅒ is saved every 32 updates. Its onset is known only to (752, 784], and whether
  its run would end, and when, is past 1000.
- "Mixed" makes no joint claim. Neither a step clock nor a dose clock is established across the
  three rates, and F½'s step labels are beside, not scored.
- **What F⅒'s transient is, is not established** (reworded after `/challenge-pr` on #378,
  finding 1; the first push said "no threshold lies above a tenth"). Rule T counts it, so the
  registered threshold is not borne out. But its run is cut off at 1000 and never seen ending,
  its size is about a sixth of F½'s, and the reviewer's post hoc recomputation from the records
  puts its energy drops at about a tenth of F¹'s (still 3–8 times τ, so not a threshold artefact).
  "No threshold", "an effect that weakens with the rate" and "a much slower clock" are not told
  apart.
- The transient's trigger stays C8's (ROADMAP §8).

**Scoring.** As Blocked 157 (a) set for `P-SCH1` and Blocked 168 drafted for `P-SCH2`–`6`,
scoring is one `notes` sentence per entry, and the commit is yours (STATE Blocked 177). None of
those earlier sentences is in the registry yet (finding 2), so Blocked 177 carries all eight
*(all eight appended verbatim 2026-11-26 on the user's instruction, #386, `52470c8`)*. Each
here is prefixed "Read 2026-11-25 (C4′ (d)+(f) U3c, `data/trackC/c4pdf/2026-11-25-u3c/reading.json`
sha256 16150f06cc1b8a682eef49266d9911345ab142ef6b70baf500f03869d28acccd, rules `284f0d10f5ba`,
the gate passed):":

| entry | draft sentence |
|---|---|
| `P-SCH7` (d1) | does not hold, every cell readable; falsifier met. F½ has a transient at every gated arm (onsets (56, 60], (56, 60], (60, 64]), and so does F⅒ ((752, 784] at each, running to 1000). |
| `P-SCH8` (d2) | does not hold; falsifier met. F½'s induction date is (304, 320], inside the step band [204.8, 400], not the dose band [409.6, 800]. |

**Next: the close-out** (`design-C4pdf.md` "Exit condition"): the card, C4's item 7, `lit-C4.md`
Parked 1, and the one page. No Parked items.

**Re-run.** From `../Mets-work`, the `mets` env's python,
`METS_DATA=<main>/data python -m trackC_writeup.c4pdf_read read --out
<main>/data/trackC/c4pdf/2026-11-25-u3c --gate <main>/data/trackC/c4pdf/2026-11-23-u1b` (the
folder holds links to F1, BP, BO, Fh, Ft and their `heads_*.json`).

## C4′ (d)+(f) unit 2b, F⅒ and F½ (2026-11-25)

Registered: trained under P-SCH7 (d1) and P-SCH8 (d2), each arm's `meta.json` recording
"registered as P-SCH7 / P-SCH8" (the guard read `c4pdf_entries.json`). Blocked 175 (i) (a), (ii)
(a), the user's (`70f66a5`): the budget extended to U2b, U3c + close-out, and one sentence added
to each entry's `notes` before either arm's first update. Code at `70f66a5`: `c4pdf_train`
`84662428cf73`, `c4pdf_rules` `284f0d10f5ba`, `c4_train` `1c1e1d2df083`; heads' code
`019846e1561b` (U2a's). Batch `HDD_1TB/mets_data/c4pdf/u2b_batch.sh`, run by the `runner`
agent, no edits. Conda `mets` (torch 2.10.0+cu130), RTX 3080, n_micro 512, compiled, bf16
autocast, fp32 master. Input: seed 0's init, C4's batches 1–1000; records on C4's 7 prompts.

| step | state |
|---|---|
| Ft (F⅒, 1e-4 flat) | 0 → 1000, exit 0, two segments (0 → 481, stopped by the user; 480 → 1000, 3 h 59 min); 1000 logged steps, all finite, lr 1e-4 at every step |
| Fh (F½, 5e-4 flat) | 0 → 1000, exit 0, one segment, 7 h 42 min; 1000 logged steps, all finite, lr 5e-4 at every step |
| same start | step 1's loss is 11.081148006021976 on Ft, Fh and F¹ (same init, same batch, before the first update); the step-0 records of Ft and Fh are equal |
| the resume | Ft resumed from `resume.pt` at 480; step 481's loss 5.243332610465586 on both segments, equal to every digit (grad norm equal to 6 digits); `train_log.jsonl` keeps both rows with a segment line between |
| records | `records --arm Ft Fh`: 539 files per arm (77 states × 7 prompts), none empty, n 512 |
| heads | `heads_Ft.json` sha256 `317689664d38`, `heads_Fh.json` `cab13fbe4e09`, 77 states each |
| disk | `HDD_1TB/mets_data/c4pdf/{Ft,Fh}/` ≈ 21 GB each; HDD_1TB 37 GB free after |

**Seen before the read:** the runner's report gave each arm's training loss and grad norm at
step 1000 (and Ft's at 481), because its brief asked for them (finding 4: a brief for a run under a
registered entry asks "finite: yes / no" instead). Those are log numbers, not P-SCH7's or P-SCH8's statistic (rule T on
the centred records, the induction bar on the heads), and neither entry nor rule can change now.
They are left out here.

**The first launch was stopped by the user** (launched 2026-11-24 16:39 with the `runner`'s
detached template; Ft at step 481 when the user said "stop all computations, we will begin again
tomorrow" at 20:25; that session stopped the runner, then killed the batch, which had survived
it, by hand, so no exit file was written). The stop was not recorded on disk. This unit's first
commit took it for a silent death tied to the launching session and wrote that cause here, in
STATE and in `LESSONS.md`; `/challenge-pr` on #377 (finding 1) found the stop in that session's
transcript, and all three are corrected (`LESSONS.md` 1). Resumed 2026-11-25 04:58. The first
launch's `Ft.log` was overwritten by the relaunch (finding 2); `train_log.jsonl` keeps every row
of both segments.

**The resume check** (finding 3): step 481's loss and grad norm show that the weights and the
batch order resumed. They do not show that the optimiser state did; `train` restores it from the
same `resume.pt` (`c4pdf_train.py` `train`, `opt.load_state_dict`), which was written before
the stop.

**Next: U3c**, the d read: `c4pdf_read read` on F⅒, F½ against F¹'s references, P-SCH7 and
P-SCH8 scored by their frozen rules, then the close-out (`design-C4pdf.md`). No Parked items.

**Re-run.** From `../Mets-work`, `METS_DATA=<main>/data HF_HOME=HDD_1TB/mets_data/hf
HF_HUB_OFFLINE=1`, the `mets` env's python: `bash HDD_1TB/mets_data/c4pdf/u2b_batch.sh` (every
step resumes), or by hand `-m trackC_writeup.c4pdf_train train --arm Ft`, then `--arm Fh`;
`-m trackC_writeup.c4pdf_read records --arm Ft Fh --out <dir>`, then `heads --arm Ft Fh --out
<dir>`.

## Paper: every model read (2026-11-24; writing only)

The user asked whether the other models' checkpoints could strengthen the paper ahead of their
rewrite. Nothing was computed. Every number added was already printed in `paper/`, except three
Timaeus checkpoint steps (9536, 491, 917), which went into `paper_slots.json` as literals with
their records (`trackT_timaeus/status-T.md` "T2 unit 1", "T5 U2b").

| where | what was added |
|---|---|
| §4.2 (new), table 2 | each model's warmup, checkpoints and two dates: 410m × 10, 160m, 70m (centred × 10), warmup01, GPT-2 small × 5, the harness and flat arms × 2, Dodecahedron (refused) |
| §4.2, "Read across the table" | description: on every run with a warmup that dates both counts, E₁'s break comes first (the flat arms left out: their half-rise dates a transient); on Pythia's own schedule E₁'s break is 256–512 and the centred one 512–1000 |
| §4.2, "Where the count could not be read" | Dodecahedron: every centred tolerance fails its gate at step 9536 (T2; step 1 read only for loss and the cosine); the uncentred count was never gated there and misses plant A (0.04 / 0.07); in the trough (491–917, `-32m` only) the uncentred count sees plant A as on Pythia (its re-init rate 6.2 %, over T2's 5 % bar) and one fitted centred tolerance passes, not §4.1's four (T5 U2b); no date; concentration a co-occurrence, not a cause |
| §5, "On a finer grid" | description: on both harness arms E₁ (384, 320) before the centred break (640, 640), which is one saved step before induction (768, 768): aligned by §5's one-step convention, the same sign on both (shared init); thinned to Pythia's grid, 640 and 768 tie. Limits: 70m only; induction by §7's rule, not E3's locator; OV, previous-token and n-gram rows unread |
| abstract, §1, §3.1 | the scope line names the other models, read for the dates alone |

**Caught before commit:** the first Dodecahedron wording said "the same gates pass" in the trough.
T5 U2b's table shows only the uncentred count and T2's B-matched tolerance pass there, on `-32m`
only; corrected. **Not established:** any new date, the mean-falls share on a model other than
410m (not checked whether it was ever read), or the order of the break, induction and the OV phase
on any model. **`/challenge-pr` on #376: accept with changes, findings 1–5 taken** (wording only):
the finer-grid gap is "aligned" by §5's convention, not "ordered"; Dodecahedron was gated at 9536
only and its uncentred count never; §4.2's gating sentence names which count was gated where; the
causal last sentence dropped; "Read across the table" leaves out the flat arms. **Re-run:** `python -m trackC_writeup.paper_check`; compile `paper/` with
`pdflatex main; bibtex main; pdflatex main; pdflatex main`.

## C4′ (d)+(f) unit 3b, B_K, F¹'s second replicate, the registration (2026-11-24)

Tier 1. Blocked 174 (a) taken by Claude on the user's "continue on the next 5 items in STATE.md"
(as recommended; ≈ 4 GPU-h beyond the design's 19.4). `trackC_writeup/c4pdf_bk.py`
`003164e7af61` (the arms, rule K, G-K, H-K, the rebuild's gap) pushed at `8490624` before the
rebuild or any B_K output; G-K checked then on the already-read B_P, B_O (passes). `c4pdf_rules`
`284f0d10f5ba` and `c4pdf_train` `84662428cf73` unchanged. Run by the `runner` agent at
`81f7bac` (batch `HDD_1TB/mets_data/c4pdf/u3b_batch.sh`; two non-numeric edits: `HF_HOME` added,
and the pairs run one after the other after a CUDA OOM side by side; the OOM'd A0 segment logged
no update). Conda `mets`, RTX 3080, n_micro 512, compiled, bf16 autocast, fp32 master.
Output `data/trackC/c4pdf/2026-11-24-u3b/` (main tree); checkpoints `HDD_1TB/mets_data/c4pdf/{a0r,BK,BKO,rep2}/`.

**BKO is beside Blocked 174 (a)**, which named B_K alone: the rebuilt moments belong to a
near-replicate, laid on the saved weights, and BKO (the same start at A0's own rate) is the null
for that mismatch disturbing the state by itself.

| step | result |
|---|---|
| rebuild (`c4_train` unchanged, A0 0 → 256, 7191 s) | `gap.json` (`0fb90ae1aed3`): bit-equal to A0's files at 0 and 1; differs from 2 (63 of 76 tensors, max \|Δ\| 2.4e-7), relative L2 1.1e-4 at 128, **9.8e-3 at 256**; the loss first differs at update 3 (max \|Δ\| 0.026 over 1–256, 0.008 at 256). Its `resume.pt` at 256 holds the moments (`c4_train` `1c1e1d2df083`) |
| B_K, BKO (A0's saved 256, rebuilt moments; the peak / `lr_pythia`; 256 → 384) | exit 0, 3571 s and 3609 s; every loss finite |
| H-K | `hk.json` (`2404e65dc962`) pass: k = 0 is A0's file on all 76 tensors on both arms; step 257's loss 5.98879364784807 on B_K, BKO and A0's own log |
| G-K | pass: B_P cut at 384 a transient at every gated arm (runs from 280, 280, 288), B_O none |
| **K** (`reading_k.json` `80df84354494`) | **"the restart's" at 9.61e-5, 1e-4 and 3e-4**: B_K and BKO have no state with a majority outside {T1, T2} at all (no run, not a short one) |

**What K says, with f.** On A0's trained state at 256 the transient appears with fresh moments at
the peak (B_P) and not with fresh moments at A0's rate (B_O), nor with the moments kept at the peak
(B_K) or at A0's rate (BKO). So the jump in rate alone does not make it; the restart's first updates
at a high rate do. With F (fresh moments at init, at the peak) and A0 (fresh moments at init, the
rate rising over 1430), every arm so far with the transient has had un-adapted moments meet a high
rate. **What "the restart's" does not separate (`/challenge-pr` on #375, finding 1):** fresh
moments make the first updates larger, so the label cannot say whether the moments matter as such
or only through the size of the earliest steps. `drift.json` (`c4pdf_read drift`, beside, read
never): the relative L2 distance from A0's 256 file is 0.033 on B_P against 0.011 on B_K after one
update, and 0.17 against 0.12 at 272. B_K catches up by 320 (0.36 against 0.34) and is further by
384 (0.55 against 0.51), still with no transient. So the transient follows how fast the first ~16
updates move, not B_K's total movement; ROADMAP C8 (iv) carries the arm that separates them (B_K
with a short rate overshoot sized to B_P's early drift). **BKO's use (finding 2):** a "the jump's"
reading, the one it guards, did not occur, so its null did not bind. It does show that the rebuilt
moments behave like A0's own: BKO is 1.2 % and 2.0 % from A0's saved states at 320 and 384, where
B_O (fresh moments, the same rate) is 3.7 % and 4.5 %. **Not shown:** that this holds at init (no kept-moment arm starts at 0: C8 (iv)'s
moment-warmed flat arm asks it); a transient later than 384 on B_K (B_P's began 16–32 updates after
the jump; B_K's horizon is 128 updates); that the moments are A0's own (a near-replicate's, 0.98 %
off; BKO shows the mismatch alone does not disturb). The lean f-jump (`design-C4pdf.md`) is broken
on this arm; the restart's reading is the one left.

**Beside, read never (the training logs, 257–384):** the loss peaks at 9.57 on B_P (259), 7.42 on
B_K (259), 7.46 on B_O (258) and 5.99 on BKO; the grad norm peaks at 10.9, 11.3, 7.3, 2.8. So a loss
spike of B_O's size comes without the transient; only B_P's larger one comes with it (one arm
each; ROADMAP C8 (i) reads the logs properly). At 384: 5.33, 4.84, 5.30, 5.29.

**F¹'s second replicate (Blocked 173's question, before registering).** `c4pdf_train train --arm F1`
into a second root (3649 s; it ran beside the OOM'd A0 segment). Against F, not bit-equal from
update 4 (`rep2/hd.json`: max \|Δ\| 0.0016 at 4, 0.033 at 128; F¹'s was 0.067). Its references
(`rep2/reference.json` `01b578b6e822`): **(44, 48] at 9.61e-5, (44, 48] at 1e-4, (48, 52] at 3e-4**,
against F¹'s (44, 48], (48, 52], (48, 52]. So the spread between replicates is one grid step (4
updates) at 1e-4, where r2's interval now holds F's and F′'s 48, and none at the other two. G-d1
re-run on r2's references passes ("dose" at ½ and ⅒ at every gated arm). Neither registered score
reads the reference (d1 on presence, d2 on F's own date), so this firms the onset labels beside d1
only. The read uses F¹'s reference as frozen.

**The registration (Blocked 173 (a), the user's, 2026-11-24, "register d1 and d2 as drafted"):**
P-SCH7 (d1) and P-SCH8 (d2) in `claims/registry.json` as drafted (`4ebc723`), P-SCH8's
null_construction opened "Calibration, not a p-value:" (d1's words, for `check_registry`); 
`trackC_writeup/c4pdf_entries.json` names them; `c4pdf_train.registration()` reads both registered.
No F½ or F⅒ update had been taken. The claims tables re-rendered (`b3ab38c`, `1159b48`, `f4bf117`).

**U3a Parked 1, done:** `c4pdf_read.read` and `read-k` write an `input` block (the gate's path and
sha256, the record folder, the checkpoint root; `read-k` also H-K's and the branches' root).

**Re-run.** From `../Mets-work`, the `mets` env, `HF_HOME=<HDD_1TB>/mets_data/hf`: the batch above;
then `METS_DATA=<main>/data python -m trackC_writeup.c4pdf_read read-k --out <u3b> --gate
<main>/data/trackC/c4pdf/2026-11-23-u1b --hk <u3b> --f-reading <main>/data/trackC/c4pdf/2026-11-24-u3` (with `BP/`, `BO/` and their
heads linked into `<u3b>`; it now refuses unless f's reading was made on the same gate and rules,
the check `c4pdf_bk`'s docstring promised, #375 finding 5; the reading is unchanged apart from its
`input` block), `… drift --out <u3b>` (CPU), and `… reference --out <u3b>/rep2 --gate <same>`.

**Next:** F⅒ then F½ (D4 (a); ≈ 14.7 h), now unblocked; then U3's d half and the close-out. ROADMAP
§8 C8 (added 2026-11-24, the user's) takes up the trigger. No Parked items.

## C4′ (d)+(f) unit 3a, f's read (2026-11-24)

Tier 1. Order, each pushed before the next: `lit-C4pdf.md` Parked 1 (Gupta et al.) read and the
D2 (b) rule fixed (`7a71b60`), then `read --f-only`. No U2 record had been opened before
that commit. Input: the gate `data/trackC/c4pdf/2026-11-23-u1b/gate.json` (sha256 `078bf9315744`,
pass, rules `284f0d10f5ba`, gated τ arms 9.61e-5, 1e-4, 3e-4); F¹'s records and heads
(`2026-11-23-u1/`); B_P's and B_O's (`2026-11-23-u2/`), linked into `2026-11-24-u3/`. Output
`2026-11-24-u3/reading_f.json` (sha256 `90870c5065b9`). Code at `7a71b60`; conda `mets`, CPU,
0.5 s (JSON only).

**Gupta et al. 2308.04014** (HTML v2, §1–§6): silent on whether the moments were reset, and on the
spike's size and length ("its first few iterations"); max rate 3e-4, from the final checkpoint,
on new data. So f's wording does not change, and f-jump keeps Gupta as a weaker lean
(`lit-C4pdf.md` Parked 1).

| τ arm | F¹'s reference (t⁻, t] | B_P (rule T) | B_O | f |
|---|---|---|---|---|
| 9.61e-5 | (44, 48] | transient, run 280 → 512: onset in (16, 24] updates after the jump | none (no run) | **not the early state's** |
| 1e-4 | (48, 52] | transient, run 280 → 512: (16, 24] | none | **not the early state's** |
| 3e-4 | (48, 52] | transient, run 288 → 512: (24, 32] | none | **not the early state's** |

`f_joint`: **"not the early state's"**. A fresh AdamW at the peak makes the transient on A0's
trained state at 256 as it did on F's init. The same restart at A0's own rate (0.18 → 0.36 of the
peak) makes none. Together with A0 itself, which reaches the peak gradually at 1430 with no
transient, a restart alone at a low rate does not make it, and neither does the peak reached
gradually. **Why it is F's transient:** B_P's run is a majority of the 7 prompts at T3, T4 and T5
(T1 and T2 are the end set), the same transitions as F¹'s run at every gated arm. B_O never has a
majority outside the end set at any state. It has minority hits at T2–T4 from 352, at most 1 of 7
outside the end set. Checked after `/challenge-pr` on #374, finding 2. **What f does not separate
(by design, #371 finding 3):** the jump in the rate from the restart's first updates at the peak.
The lean f-state [arms, scan] is broken. The result fits f-jump and a fresh Adam's first updates
at the peak equally well; it does not favour f-jump (reworded after #374, finding 3).

**Beside, read never (description):** B_P's onset in updates after the jump (16–32) is earlier
than F¹'s (44–52). Both runs last to the arm's last state (F¹'s to 128, B_P's to 512), so in
256 updates B_P's transient does not end under rule T. F′'s own transient ran 64 → 1000, so no end
was expected (#374).

**D2 (b).** By the rule fixed before the read (`lit-C4pdf.md` Parked 1), "not the early state's"
is the row where D2 (b) earns its hours: B_K (A0's moments kept, jump to the peak) tells the jump
from the restart. A0's moments at 256 do not exist, and H-d showed the harness is not
bit-reproducible past one update. So a rebuild (A0 re-trained 0 → 256 with its moments saved) is a
near-replicate of A0's 256, not the saved file. That is STATE Blocked 174, the user's (it adds GPU
hours beyond the design's 19.4). The options were revised after `/challenge-pr` on #374, finding 1:

| | B_K's start | cost | against it |
|---|---|---|---|
| (a), recommended | A0's **saved** 256 weights with the rebuild's moments, at the peak, to 384 (horizon fixed now) | rebuild ≈ 1.9 h + ≈ 1 h | the moments belong to a near-replicate's weights (their gap from the saved file is recorded first); B_K and B_P then differ only in the optimiser state |
| (b) | B_O's step-512 `resume.pt` (state and moments exact, on disk), at the peak | ≈ 1 h to 640 | the jump is 2.8× (0.357 of the peak), not 5.6×, on B_O's state, not A0's; "none" from it would be weak |
| (c) | none: f closes here, the jump and the restart not separated | 0 | |

**Re-run.** From `../Mets-work`, the `mets` env's python: `-m trackC_writeup.c4pdf_read read
--f-only --out <dir holding F1/, BP/, BO/ and heads_{F1,BP,BO}.json> --gate
<main>/data/trackC/c4pdf/2026-11-23-u1b`.

**Parked.** 1. `reading_f.json` does not name its input: its `gate` field repeats the rules hash,
and no gate path or record folder is written (#374, finding 4; pre-existing in `c4pdf_read.read`).
Why: a result should state its input. Cost: three fields. Changes: nothing in f (this section
names the inputs, and both hashes match). Carried to U3b: fixed before the d read.

## C4′ (d)+(f) unit 2a, (f)'s branches B_P and B_O (2026-11-23)

Run by the `runner` agent from `../Mets-work` at `334ee9b` (code `c4pdf_train` `84662428cf73`,
`c4pdf_rules` `284f0d10f5ba`; records' code `220161b9159b`, heads' `019846e1561b`); no edits,
tier 1, nothing read. Input: A0's step-256 file and π's batches 256 + k; records on C4's 7
prompts.
Conda `mets` (torch 2.10.0+cu130), RTX 3080, n_micro 512, compiled, bf16 autocast, fp32 master.

| step | state |
|---|---|
| B_P (A0's 256, fresh AdamW, 1e-3 flat) | 256 → 512, exit 0, wall 7302 s; 256 logged steps, all finite; lr 1e-3 at every step |
| B_O (same weights, fresh AdamW, `lr_pythia(256 + k)`) | 256 → 512, exit 0, wall 7350 s; lr 1.790e-4 at 257 rising to 3.573e-4 at 512 (0.179 → 0.357 of the peak, as the design) |
| H-f, state half | `hf.json` (sha256 `bb1e9ea3ca69`, "pass": true): each branch's saved k = 0 state equals A0's step-256 file on all 76 tensors (max abs 0). This checks the save after loading, not the forward pass |
| H-f, batch half | `hf.json` only hashes batches 256 + k (k = 0, 1, 255); it compares them with nothing. **The check that covers both halves:** step 257's loss is 5.98879364784807 in B_P's, B_O's and A0's own log (`mets_data/c4/A0/train_log.jsonl`), equal to every digit (same state, same batch, before the first update). Corrected after `/challenge-pr` on #373, finding 2 |
| records | `records --arm BP BO`: 119 files per arm, 17 saves (256 … 512) × 7 prompts; first line n 465, finite, equal at 256 on both arms (same state) |
| heads | `heads_BP.json` `393a2152ca66`, `heads_BO.json` `1f3c6df41a88`, 17 saves each |
| disk | checkpoints `HDD_1TB/mets_data/c4pdf/{BP,BO}/` 5.3 GB each; HDD_1TB 88 % used before the runs |

Training logs: every loss finite on both arms, no divergence. **Seen before Gupta was read
(2026-11-23):** the runner's report and this session quoted both branches' training-loss peaks and
the step-512 losses (in the session and in this PR's first commit). Those are log numbers, not
f's rule, and the rule is frozen. But `lit-C4pdf.md` Parked 1 (Gupta et al.) is to be read before
U2's records, since it can change f's wording and whether the kept-moment branch (D2 (b)) is
worth its hours. So that reading and the D2 (b) choice were made with the size of the spike
known. The numbers are left out here (`/challenge-pr` on #373, finding 1).

**U2's first half is done** (both branches trained, the state matches A0's, records written).
f does not wait on Blocked 173: `c4pdf_read read --f-only` needs only F¹ (the reference), B_P and B_O, all trained. So the
next step is U3's f half: read Gupta et al. (Parked 1), then `read --f-only`, then decide on D2 (b).
This can run while the user decides Blocked 173. That decision then gates F½, F⅒ (U2's
other half, ≈ 14.7 h). No Parked items.

**Re-run.** From `../Mets-work`, `METS_DATA=<main>/data HF_HOME=HDD_1TB/mets_data/hf
HF_HUB_OFFLINE=1`, the `mets` env's python: `-m trackC_writeup.c4pdf_train train --arm BP`, then
`--arm BO`; `c4pdf_train hf --out <dir>`; `-m trackC_writeup.c4pdf_read records --arm BP BO --out
<dir>`, then `heads --arm BP BO --out <dir>`.

## C4′ (d)+(f) unit 1, the rules, the gates and the reference run (2026-11-23)

Blocked 172 taken as recommended (D1 (a), D2 (a), D3 (b), D4 (a)) by Claude on the user's
"continue for the next 5 state items"; the user's to overrule. Rules, driver and reader pushed
before any output (`8ff5463`: `c4pdf_rules` `bc45f4a54d84`, `c4pdf_train` `f46abc7de0e8`). Tier 1;
nothing registered.

**Input set.** Records and heads of A0, F (`data/trackC/c4/2026-11-12-u2`, `2026-11-14-close`) and
A0′, F′ (`data/trackC/c4pc/2026-11-22-u3`), C4's gated τ arms (9.61e-5, 1e-4, 3e-4, from
`c4/2026-11-12-u2/gate.json`); F¹'s 33 states (`HDD_1TB/mets_data/c4pdf/F1/ckpt/`, 9.5 GB).

| gate | result |
|---|---|
| G-T | pass: F's transient 48–512 (48, 48, 64 at the three arms), F′'s from 48 / 48 / 64 to past 1000; A0 none; A0′'s {T3} at 640 a one-state run, none |
| G-d1 (F's coarse reference) | pass: F stretched by 10 reads "dose" (onset 480, 480, 640); at ½ "refused" (the bands meet on F's grid); unstretched "step" |
| G-I | pass: b = 0.4819 (plateaus A0 0.962, F 0.927, A0′ 0.973, F′ 0.966); bar dates 768, 320, 768, 320 = the half-rise dates |
| G-d2 | pass: stretched by 2 "dose" (640), by 10 "not step"; unstretched "step" (320) |
| G-f | pass: F's states 0–256 a transient (48 / 48 / 64); A0's 256–512 none |
| H-d | **fail**: F¹'s update-1 loss equals F's to every digit, but its weights differ from update 4 (76 tensors, max abs 0.0017 at 4, 0.067 at 128) |
| G-d1 on F¹'s reference | pass: "dose" at ½ and ⅒ at every arm |

**F¹ as the reference.** C4's harness does not reproduce F's weights by update 4, the first state compared after 0 (cause
not tested; `LESSONS.md` 4). By the design's gate row F¹ is then a near-replicate of F; its cells differ
from F's at 3–6 of 72 on the 12 shared states. Its transient begins at **48 at τ 9.61e-5 and 52 at
1e-4 and 3e-4** (F's coarse 64 at 3e-4 lies in (48, 64]), so the references are (44, 48], (48, 52],
(48, 52]. F½'s windows: step [35.2, 60] / [38.4, 65], dose [70.4, 120] / [76.8, 130]; F⅒'s dose
window reaches 650, inside its horizon. H-d's code first listed F's states 1 and 2, which grid D
does not save; the runner stopped on it, and it was fixed to the 12 shared states before H-d was
read (`c4pdf_train` hash moves; F¹ trained under `f46abc7de0e8`).

**Drafted:** the two entries for d1 and d2 (`design-C4pdf.md` "The two draft entries"), H-d dropped
from their falsifiers. Registering them is Blocked 173, the user's; F½ and F⅒ refuse until
`trackC_writeup/c4pdf_entries.json` is committed.

**Re-run.** `METS_DATA=<main>/data python -m trackC_writeup.c4pdf_read gate --out <dir>` (no GPU);
`… c4pdf_train train --arm F1` (≈ 62 min on the 3080) and `… c4pdf_train hd --out <dir>`; `…
c4pdf_read records --arm F1 --out <dir>` and `heads` (2 min). The reference and G-d1's re-run were
computed inline and saved as `reference.json` (sha256 `ff097573de52…`); U3's `read` recomputes the
reference from the records.

**Parked:** none new. `lit-C4pdf.md` Parked 1 stays before U2's records.

**After `/challenge-pr` on #372** (accept with changes; reviewed `97c6523`). Taken before any new
arm trained, rules re-pushed first (`2ab4b14`, rules `284f0d10f5ba`, driver `84662428cf73`), gates
and reference re-run and passing (`data/trackC/c4pdf/2026-11-23-u1b/`; the first record set aside):
(1) the drafts named a driver hash that had moved; they now name the code that will run. (2) d1
scored "dose or step" while the code could return "mixed"; the entry now scores presence only
(`d1_presence`), labels beside. (3) F¹'s reference at 1e-4, (48, 52], excludes F's and F′'s 48, and
G-d1's re-run passes by 2 updates at 3e-4: stated in the design and the d1 notes; a second F¹
replicate (≈ 0.9 h) is put to the user in Blocked 173. (4) a band label now needs the whole interval
(previous state, event] inside the band, and grid D is every 16 updates over 144–656 (77 states).
(5) the registration guard has a test. (6) H-d's failure is stated as measured (the weights differ
by update 4), its cause untested; D2 (b) and (c)'s bit-equal rebuild is void. (7) the grid above
656 (every 32) can miss a transient shorter than 32 updates there; stated in the d1 notes.

## C4′ (d)+(f) unit 0, the scan and design (2026-11-23)

Blocked 169, decided by the user 2026-11-23: (e) and (g) first, then (d) and (f); (e)+(g) closed
in #370. One design for both, since both ask what sets F's transient and both train short runs on
C4's harness. Writing only; nothing trained, no record read beyond what earlier units showed.

| item | where |
|---|---|
| scan: six searches, every outside row [S]; the catapult line (a threshold set by the early state's curvature) and Gupta et al.'s direct-to-peak spike on a trained Pythia give (f) two leans pointing opposite ways, both stated before output; fresh Adam moments matter for tens of updates (Ibrahim et al.), which is F's window, so (f) keeps a restart control | `lit-C4pdf.md` |
| design: arms F½ and F⅒ (flat at a half and a tenth of the peak, C4's order and init), branches B_P and B_O from A0's step-256 weights with fresh AdamW (the peak; A0's own rate); F¹ (F re-run to 128 on a 4-update grid, the reference); rule T (a run of states with a majority outside {T1, T2} spanning ≥ 16 updates); d1 the onset against step and dose bands from F¹'s interval, refused where they meet; d2 induction by a bar rule; f by the branches' pair | `design-C4pdf.md` |
| arithmetic | `tools/math_checks/c4pdf_bands.py`: the bands disjoint for r < 1/ρ², the grid steps in each, the horizons, A0's crossing of F's onset dose |
| leans, stated now | d1 (a threshold between a tenth and a half), d1′ (F½'s onset in the dose band; no source), d2 (F½'s induction in the dose band), d3 (F⅒ not formed by 1000); f-state and f-jump, opposite |
| exit, budget | 4 units + close-out; GPU ≈ 19.4 h at D1 (a), D2 (a) |
| decisions | D1–D4, STATE Blocked 172 |

**Found while writing:** (1) the ROADMAP row's horizons (F½ to 768, F⅒ to 384) cannot read its own
question: under the dose clock F⅒'s transient falls in [256, 800] and F½'s induction band reaches
800, so D1 (a) runs both to 1000. (2) A0's summed rate reaches F's onset dose (48 peak-steps)
at step 370, at 0.26 of the peak, with no transient there: a pure dose clock for the transient is
already argued against unless the state matters, which is (f)'s question. (3) Checkpoints hold
weights only (`resume.pt` is A0's at 4000), so A0's moments at 256 would have to be rebuilt by
re-training (D2 (b), (c)).

**Checked against the tree:** A0's `step256/model.safetensors` exists (fp32, 282 MB);
`c4_train.lr_flat` and `lr_pythia` take no rate factor, so a new driver is needed and `c4_train`
stays unedited; `HDD_1TB` has 123 GB free (the runs need ≈ 22 GB).

**Parked:** `lit-C4pdf.md` Parked 1 (Gupta et al.'s full text; read before U2's records are read).

**After `/challenge-pr` on #371** (accept with changes; all five taken; the reviewer recounted the
four arms' transitions from their records): (1) F's onset is 64, not 48, at τ 3e-4, so the
reference is per gated arm and F⅒ runs to 1000 (its dose band reaches 800 there). (2) ρ = 1.25 is
C4's floor, not its band: on F's grid (32, 48, 64) the half-rate step and dose bands meet, so F is
re-run to 128 on a 4-update grid (F¹, ≈ 1 h) and the bands are built from its interval, with a
refusal where they meet; the new arms save on a dense grid D (65 states). (3) A0 reaches the peak at
1430 with no transient, so the rate alone on a trained state is already excluded; (f)'s labels
now read "the early state's or not", with the jump and the restart not separated at D2 (a).
(4) Rule T counts updates (a run spanning ≥ 16), not states. (5) F's transient runs 48–512 (T3);
the scan's Parked read is before U2's records. GPU ≈ 19.4 h, disk ≈ 56 GB. `LESSONS.md` 6 gains
the instance.

## C4′ (e)+(g) on one page (closed 2026-11-23)

**Closed at its exit and budget:** U0 the design (#366), U1 (e) (#367, #368), U2 (g) (#369), and
this close-out. Every lean is marked, and items 8 and 10 each say what (e) and (g) added. Tier 1;
nothing registered (Blocked 170 D4, 171 D4 (a)). 70m only, two pairs of one init.

**What it established** (each on both pairs; numbers in the unit sections):

| # | finding | lean |
|---|---|---|
| 1 | The tokens' mean, read through the unembedding, correlates with log unigram frequency at every readable window on all four arms, weakly in A0's collapse window and strongly from about A0's 128 and F's 8. It is closest to the least-squares direction just before or at V's date. | e1 held |
| 2 | At V's block both schedules start with the MLP diluting the mean and end with attention diluting it; only the flat schedule has a middle stage (128–640) where the MLP writes against it. | e2 held on A0, broken on F as pooled |
| 3 | The two schedules' mean directions at mid states are near orthogonal by 4000 (cos 0.14 against a reordering's 0.94). **Post hoc (close-out), read through W_U at 4000:** at state 3 the schedules rank the vocabulary alike at r 0.81 (0.69 with the unigram ranks removed), against two orderings' 0.92–0.94 (0.87–0.91) and another init's (`pythia-70m-seed1`, `-seed2`) 0.71–0.81 (0.53–0.68). So mid-stack the schedules differ in readout about as much as different inits do, not only in coordinates. At the final state every pair of trained models agrees at 0.968–0.981, other inits included, so that read cannot tell schedules apart (first written "agree in what they predict"; withdrawn after `/challenge-pr` on #370, finding 1). Beside: the per-context KL between F and A0 at 4000 is 0.36 against A0's own last step's 0.15 (U2). | e3 end half broken |
| 4 | The schedules' end states, from one init, are not one basin: the held-out loss rises 3.56 nats on the line between A0 and F (pair 2: 3.54), beyond the order-only spread (1.03 flat, 0.13 Pythia's). | g2 held |
| 5 | Warmup keeps two orderings linearly connected on 70m; the flat schedule does not (0.13 against 1.03 at 4000). | g1 held |
| 6 | Mean-ablating the other schedule's induction heads does nothing on any arm (copy gain −6 % to +4 %). The own heads remove 64–86 % of it. **Post hoc (close-out):** each arm's previous-token heads, ablated alone, remove as much or more: L2H1 (score 0.94–0.95) 68 % on A0 and 75 % on A0′; on the flat arms two heads share the job, L1H7 and L1H3 on F (36 % and 18 % alone, 79 % together), L1H6 and L1H3 on F′ (39 %, 18 %, 81 %). The random sets that matched the own heads are the ones holding these. The frozen null did not exclude induction's upstream half, so both pair 1's break and pair 2's hold (A0′ by 0.01) are the null's. | g3 unresolved on both pairs (broken as frozen on pair 1, held on pair 2) |

**For C4's items.** Item 8 (V one block earlier on F): the mean that V counts is the unigram
direction's carrier, and its falls at V's block end as attention's dilution on both schedules. The
earlier block is not a different writer at the end, only a different path to it. Item 10 (F off
A0's path): the paths end in different basins, with mid-stack readouts as far apart as another
init's, through induction circuits (previous-token and induction heads) whose indices do not
transfer.

**Not established.** Whether a permutation of units connects A0 and F. How the arms'
predictions differ on real text (finding 3 reads m̄ only, a logit lens at mid states). Cause for any of E2's writers (a read-out, not an intervention). The
previous-token heads' role beyond their score. Anything on 410m.

**Parked, each carried or dropped:**
1. U1's Parked 1, the cross-arm logits: **done** at this close-out (finding 3), post hoc, with
   two other inits as its scale (added after `/challenge-pr` on #370).
2. U2's Parked 1, an unsaturated g3: **done** at this close-out (finding 6), post hoc.
3. Permutation alignment of A0 and F (the design's "Worth challenging" 4): **carried to ROADMAP
   C4′**, unscheduled (minutes to hours; could change finding 4's "not one basin").
4. A `paper/` paragraph: **not written here**. §7 already carries C4's line, and the user's rewrite
   chooses whether (e)+(g) enter it (Blocked 103's posting items stay the user's).

**Units at close:** 3 + close-out, the budget. GPU ≈ 1.5 h in all, under the 2 h budget (U1 twice, U2, the close-out's
reads).

**Re-run (the close-out's reads):** `METS_DATA=<main>/data python -m trackC_writeup.c4peg_close
logits --u1 <U1 dir> --out <dir>` (CPU), and `… g3 --u2 <U2 dir> --out <dir>` (GPU, `HF_HOME` as
U2). Records `data/trackC/c4peg/2026-11-23-close/` (code `f427728524d1`; the first pass, without
the other inits and the previous-token sets, set aside as `-close-v0`). Known answer: m̄ of A0 at
4000 recomputed on CPU equals U1's record to 9 × 10⁻⁹.

## C4′ (e)+(g) unit 2, (g): do the end states differ in function? (2026-11-23)

Tier 1, description; nothing registered (Blocked 171 D4 (a)). Rules and code pushed before any
output (`c4peg_g.py`, `bc84cd0`; the batch cut from 8 rows to 2 after an out-of-memory stop that
wrote nothing, `design-C4peg.md` "(g) as U2 runs it"). Records `data/trackC/c4peg/2026-11-23-u2/`
(`g1.json`, `g2.json`, `reading_g.json`; main tree, untracked), code `c4peg_g` at the records'
hash. GPU about 25 minutes.

**Known answers:** every interpolation's endpoints reproduce each checkpoint's own held-out loss to
10⁻⁵ (A0 3.1070 and A0′ 3.1041 equal H4′'s records); an empty ablation through the hooks leaves
the induction loss unchanged; the own sets and the 11-head union equal the design's.

**G1, the loss along the line** (held-out rows, nats; B is the excess over the chord, at α 0.5
unless stated; ± 0.05 beside agrees to ≤ 0.2):

| pair | step | L(0), L(1) | L(0.5) | B | role |
|---|---|---|---|---|---|
| (A0, F) | 4000 | 3.107, 3.058 | 6.638 | **3.56** | the question |
| (A0′, F′) | 4000 | 3.104, 3.061 | 6.618 | 3.54 | beside, pair 2 |
| (A0, A0′) | 4000 | 3.107, 3.104 | 3.240 | 0.13 | order-only scale, Pythia's schedule |
| (F, F′) | 4000 | 3.058, 3.061 | 4.094 | 1.03 | order-only scale, flat |
| (released, A0) | 4000 | 3.099, 3.107 | 3.129 | 0.026 | null |
| (A0, seed 1) | 4000 | 3.107, 3.108 | 9.476 | 6.37 | effect (another init and order) |
| (A0, F) | 1000 | 3.720, 3.373 | 5.850 | 2.45 (α 0.4) | beside |
| (A0′, F′) | 1000 | 3.715, 3.376 | 5.878 | 2.45 (α 0.4) | beside |
| (A0, A0′) | 1000 | 3.720, 3.715 | 3.699 | 0.00 | beside: linearly connected |
| (F, F′) | 1000 | 3.373, 3.376 | 3.817 | 0.44 | beside |

The gate passes (6.37 > 0.1 and > 4 × 0.026). S = max(0.13, 1.03, 0.026, 0.01) = 1.03, so **(A0, F)
reads "beyond it"** (3.56 > 2S), and pair 2 the same (3.54). **g1 held**: B(F, F′) − B(A0, A0′) =
0.90 > 0.026, so on 70m warmup leaves two orderings linearly connected (0.13 at 4000, 0.00 at
1000) where the flat schedule does not (1.03, 0.44), as Altıntaş et al. found; the 4000 reading
carries it (at 1000 F has had about 2.9× A0's summed rate, 1.2× by 4000, so the 1000 rows mostly
say A0 has moved less; `/challenge-pr` on #369, finding 4). **g2 held**: the two schedules from one
init end in different basins (3.56 nats on the line; no weight distance is reported, so barriers
are not compared as distances). The barrier is 2.45 at 1000, beside. Beside: K(F₄₀₀₀, A0₄₀₀₀) =
0.36 against A0's own last step 0.15 (C4′ (b)'s `K_F.npz`).

**G2, the heads mean-ablated** (Δ of the induction loss, nats; base A0 2.67, F 4.58, A0′ 2.02, F′
3.65; random sets 20 per arm, outside the 11-head union):

| arm | own set | Δ own | Δ other's indices | random: median, max | own > every random? |
|---|---|---|---|---|---|
| A0 | L3H0, 1, 5, 6 | 9.94 | 0.15 | −0.02, **10.09** | **no** (by 0.15) |
| F | L2H6, L3H3, 4, 7 | 6.62 | −0.12 | 0.14, **7.49** | **no** (by 0.87) |
| A0′ | A0's four | 8.79 | −0.43 | 0.12, 8.73 | yes (by 0.06) |
| F′ | L2H2, 5, 6, L3H7, L4H1 | 6.68 | 0.22 | 0.20, 4.12 | yes (by 2.56) |

The held-out loss moves little under any of these (own sets +0.22 to +0.37). **The other
schedule's indices carry nothing on any arm** (|Δ| ≤ 0.43, a loss fall at most). As frozen, **g3
broke on A0 and F and held on A0′ and F′**: on A0 and F one of the 20 random sets hurts induction
as much as the own heads. **But the instrument is saturated where the label is decided**
(`/challenge-pr` on #369, finding 1): with the own set ablated the induction loss is 10.3–12.6
nats, at or past a uniform guess (ln 50304 = 10.83), and the worst random sets reach it too. Both
then remove the copying entirely, so "own > the max" ranks other damage. A0 fails by 0.15 and A0′
passes by 0.06 with the same own set. **So g3's split between the pairs is not a difference
between them**, and g3 is read as **unresolved by this instrument**, not as broken. Post hoc, the
random sets that reach saturation hold L2H1 on A0 (10.09, 8.23) and on A0′ (8.73, 8.03, 7.98). On
F the worst holds L0H6, L1H3, L1H5 and L1H7 (7.49); the next are 3.75 and 3.56. A0 also has a set
without L2H1 at 2.36. F′'s worst sets (4.12, 3.63, 2.68) share no head. **Random sets also
lower the loss**: by up to 3.24 nats on F and 2.04 on F′, every such set holding L1H4. So the
random distribution is wide on the flat arms, not "near 0 except one head" (corrected after
finding 2). What survives without the max: **the other schedule's indices do nothing on any arm**,
and the own sets sit at or near saturation on all four. The ratio Δ(own)^F / Δ(own)^A0 is 0.67,
and F′/A0′ is 0.76. Both ratios are compressed by the ceiling. The flat arms' base induction loss
is higher (F 4.58 against A0 2.67; F′ 3.65 against A0′ 2.02), so they induce less well on random
repeats at 4000.

**Leans:** g1 held (on the 4000 reading); g2 held (both pairs); g3 broken as frozen on pair 1 and
held on pair 2, but with the instrument saturated it is unresolved. Nothing registered.

**What it says for C4's item 10** (F off A0's path): the end states are not one basin. The
straight line between them rises 3.56 nats, about 3.4 times the larger reordering's barrier.
"The schedule chooses the heads" goes past their indices in one direction: ablating the other
schedule's induction heads does nothing to either arm. Whether each arm's induction runs through
its own heads *alone* is the frozen test, which broke on pair 1 on a saturated loss (finding 3).

**Not established.** Whether a permutation of units connects A0 and F (no alignment was tried;
the design's "Worth challenging" 4). What the early heads do (L2H1 on A0, L1H4 on F). Anything on
410m.

**Parked:** 1. **An unsaturated g3** (why: the frozen max-of-20 rule ranks saturated losses;
cost: minutes of GPU, re-scoring each ablation as the share of the copy gain it removes (first-copy
loss minus second-copy loss), plus a previous-token score per head at 4000; could change: how the
close-out words g3 and which heads the random maxima are). To the close-out, post hoc.

**Re-run:** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4peg_g
{g2,g1,read} --out <dir>` (the `mets` env; GPU for g1, g2).

## C4′ (e)+(g) unit 1, (e): the common direction (2026-11-23)

Blocked 170's D1 and D2 were taken (a) by Claude on the user's "Let's just continue here", which
may have meant only "stay in this session". D2 is now spent, since pair 2 has been read. D3 and D4
went back to the user as Blocked 171 after `/challenge-pr` on #367, finding 3. 2609.02959 was read
first (D1): its fit is OLS on the centred unembedding with nKL and
R², and nothing in it argues against a rank correlation, so Spearman stayed primary. The open
details were fixed in `design-C4peg.md` "(e) as U1 runs it" and pushed with the code before any
output (`4d97ce6`; `c4peg_read` `e6e0a12eb2cc`; 10 pure tests, `tests/test_trackC_c4peg.py`).
Writing that section found that the design's "P centres" was wrong: `layernorm_to_sphere` is
plain L2, as C2's records are.

**Input:** A0 and F (`HDD_1TB/mets_data/c4/`), A0′ and F′ (`…/c4pc/`), all 29 saved steps;
v1's 7 prompts; unigram counts over the labels of rows [0, 4 096 000), 8 388 608 000 tokens,
50 061 types (`unigram_counts.npy`, sha256 `a7e757d85dab2322…`). Records:
`data/trackC/c4peg/2026-11-23-u1/{A0,F,A0p,Fp}/step*.json` (main tree), `reading.json` (sha256
`2286814845acef51…`). GPU about 30 minutes, counts one CPU pass of about 5 minutes.

**Known answers, all 116 checkpoints:** the parallel residual and the final LN reproduce HF's
states to ≤ 6.4 × 10⁻⁸; every per-prompt e1 equals C4's or C4′ (c)'s record exactly (0.0); the
split and the component means are exact to ≤ 2 × 10⁻⁸. At step 0 the gate refuses on every arm
(r(u\*) 0.088 against the permutation bar 0.096: a random W_U fits noise as well as frequency).
**From the literature:** u\*'s nKL is 0.038–0.055 at the trained steps, against 2609.02959's
0.05–0.07 on Pythia 410M–6.9B.

**E1: the mean reads "aligned" with log unigram frequency at every readable window, on all four
arms** (e1 held, pair 2 replicating). The alignment is weak in A0's collapse window: r 0.03–0.34
at state 6 over 24–64, which passes the null's bar (at step 24 by 0.002 at mid states) but is
far below the least-squares reference. It is strong (r ≥ 0.6, ≥ 0.7 of the reference) from about A0's 128 and
F's 8 on (corrected after `/challenge-pr` on #367, finding 4: first headed "the mean is the
unigram direction"). Spearman r of m̄ through W_U against log f, mid state 3 and final state 6,
with the least-squares reference r(u\*) (not an upper bound on r; first called "the ceiling",
corrected after CodeRabbit on #367) and the null's bar q:

| arm, step | r(u\*) | q | r, state 3 | r, state 6 | ‖m̄‖², state 3 |
|---|---|---|---|---|---|
| A0 24 (collapse) | 0.10 | 0.014 | 0.03 | 0.03 | 0.50 |
| A0 192–256 (peak) | 0.93–0.95 | 0.24–0.31 | 0.73–0.81 | 0.88 | 0.20–0.21 |
| A0 384 (V's date) | 0.92 | 0.35 | 0.74 | 0.78 | 0.16 |
| A0 4000 | 0.89 | 0.35 | 0.64 | 0.58 | 0.15 |
| F 4 (collapse) | 0.51 | 0.10 | 0.47 | 0.48 | 0.96 |
| F 16–24 (peak; V's date 24) | 0.92–0.94 | 0.33–0.39 | 0.78–0.82 | 0.90–0.91 | 0.24–0.41 |
| F 4000 | 0.89 | 0.36 | 0.60 | 0.57 | 0.21 |
| A0′, F′ 4000 | 0.89 | 0.34–0.35 | 0.63, 0.61 | 0.58, 0.58 | |

On both schedules the mean turns towards the unigram direction as the cloud collapses. It is
nearest it (0.78–0.90 of the least-squares reference at state 3, nKL 0.06 at state 6) just before or at V's date,
and it drifts off it as ‖m̄‖² falls. F does this about 12× earlier (peak 16–24 against 192–256).
The bar q rises to 0.35 because random directions read through a trained W_U already correlate
with frequency, and every mid-state r stays above it. The embeddings' mean is not aligned
(|r| ≤ 0.22). The final LN's bias reads r 0.71–0.94 by itself (from 128 on A0, from 8 on F),
beside.

**Per prompt** (the design's beside row; the code had dropped it, CodeRabbit on #367, so the
records were regenerated with it, every other value equal to the first pass's): each prompt's own
r at state 3 (γ readout) / state 6, and how many (prompt, step) cells are above that checkpoint's q
at state 3 / 6:

| arm | collapse | V's date | centred | 4000 |
|---|---|---|---|---|
| A0 | 0.02–0.25 / 0.03–0.33; 28 / 28 of 28 | 0.39–0.70 / 0.54–0.74; 7 / 7 of 7 | 0.32–0.57 / 0.42–0.61; 6 / 7 of 7 | 0.48–0.59 / 0.42–0.56; 7 / 7 of 7 |
| F | 0.03–0.86 / 0.03–0.90; 35 / 35 of 42 | 0.47–0.79 / 0.72–0.89; 7 / 7 of 7 | 0.36–0.74 / 0.56–0.84; 4 / 7 of 7 | 0.44–0.60 / 0.39–0.56; 7 / 7 of 7 |
| A0′ | 0.02–0.25 / 0.03–0.33; 28 / 28 of 28 | 0.49–0.76 / 0.57–0.83; 7 / 7 of 7 | 0.30–0.56 / 0.45–0.59; 6 / 7 of 7 | 0.44–0.57 / 0.41–0.58; 7 / 7 of 7 |
| F′ | 0.03–0.85 / 0.03–0.90; 35 / 35 of 42 | 0.35–0.82 / 0.65–0.89; 4 / 7 of 7 | 0.32–0.75 / 0.56–0.85; 4 / 7 of 7 | 0.45–0.62 / 0.41–0.56; 7 / 7 of 7 |

At state 6 every prompt is above q at every step after F's step 1 (the 7 cells below in F's and
F′'s collapse window are all step 1, where m̄ is still the init's); at state 3 a few prompts fall
below q at the centred date and at F′'s V date. So the aligned m̄ is not one or two prompts' doing.

**E2: what writes the mean, and its falls.** The mean is written mostly by the MLPs on both arms
(shares at state 3: A0 at 384 f1 0.30, f2 0.23, f0 0.23; F at 24 f1 0.46, f0 0.34; at 4000 f0
0.33 on A0, 0.56 on F). The falls V counts (e2 cells: V's transition, from V's date to 4000,
per prompt), pooled as the frozen rule pools them:

| arm | transition | falls | dilution, attention | dilution, MLP | removal, MLP | removal, attention |
|---|---|---|---|---|---|---|
| A0 | T2 | 66 | 59 | 7 | 0 | 0 |
| A0′ | T2 | 69 | 60 | 9 | 0 | 0 |
| F | T1 | 154 | 57 | 31 | 63 | 3 |
| F′ | T1 | 147 | 60 | 28 | 58 | 1 |

So, as frozen, **e2 held on A0 and broke on F**, and both replicate. **But the pooled count
hides a sequence** (`/challenge-pr` on #367, finding 1). F's window holds 22 saved steps against
A0's 11, so F's early steps weigh more. By step, all 7 prompts at V's transition:

| arm | stage 1: dilution, MLP | stage 2: removal, MLP | stage 3: dilution, attention |
|---|---|---|---|
| A0 (T2) | 384–512 | none | 640–4000 |
| F (T1) | 24–64 | 128–640 (80–96 mixed) | 768–4000 (1000 on, 7 of 7) |
| F′ (T1) | 32–64 | 128–512 (80–96 mixed) | 640–4000 |

**Both schedules end in the same mode: attention diluting the mean.** Both begin at V's date with
the MLP diluting it. Only the flat schedule has a middle stage, five to eight saved steps where
the MLP writes against the mean on every prompt. The replicate shows the same three stages.
The collapse windows hold almost no falls (‖m̄‖² rises there; F 5, A0 0).

**E3: the two schedules' mean directions differ, and they differ more as training goes on.**
cos(m̄^A0, m̄^F) at the same step, mid states 2–4, against the replicates' minimum:

| step | c(A0, F), states 2–4 | replicates' min | c(A0′, F′) |
|---|---|---|---|
| 4 | 0.32–0.49 | 1.00 | 0.32–0.50 |
| 24 | 0.13–0.49 | 0.98 | 0.24–0.57 |
| 384 | 0.38–0.53 | 0.90–0.94 | 0.40–0.54 |
| 1000 | 0.34–0.44 | 0.88–0.93 | 0.36–0.44 |
| 4000 | 0.14–0.15 | 0.94 | 0.11–0.13 |

All 84 mid-state cells from step 1 on read "schedule-specific", on both pairs. So, as frozen,
**e3's collapse half held and its end half broke**. **The collapse half is uninformative**
(`/challenge-pr` on #367, finding 2). At steps 1–8 A0 has not moved from its init (cos(m̄₃(s),
m̄₃(0)) 1.000, 1.000, 0.999, 0.989), so a same-step cosine there measures only how far F has
moved. The one stage-matched pair, ρ_mid's peak (A0 48, F 4), reads 0.62–0.86 over states 1–6,
and the design gives it no scale. **The end half's break is real**: by 4000 the two directions
are near orthogonal (0.14), while a reordering keeps 0.94. The lean was "one corpus, one prior,
so one direction". Both readouts still correlate with log unigram frequency through their own
W_U (E1), despite their different mean coordinates; whether they predict the same tokens is not
read, so shared function is open (Parked 1; worded after CodeRabbit on #367, which caught a
first "suggests the shared thing is the function"). The other event-matched pairs
give 0.21–0.81.

**Leans:** e1 held (both pairs; weak in A0's collapse window); e2 held on A0 and A0′ and broke on
F and F′ as pooled, while by step both schedules end in attention's dilution and F alone has a
stage of MLP removal; e3's collapse half held but measures F's displacement, and its end half
broke. Nothing is registered.

**What it says for C4's items.** Item 8 (V one block earlier on F): at the block where V's
falls sit, both schedules end in attention's dilution. F reaches it after a stage
(128–640) of the MLP writing against the mean, which A0 does not go through. Item 10 (another
path): the mean's direction is schedule-specific by 4000 and on the way there; whether its
function is too is Parked 1.

**Not established.** Whether the two arms' means predict the same tokens: E1 says both correlate
with log f, but their logits were not compared with each other (Parked 1). Cause: a read-out,
not an intervention. E1 at mid states is a logit lens (the design's "Worth challenging" 1).
Anything on 410m.

**Parked:**
1. **Cross-arm logit agreement** (why: e3's end half broke on coordinates while E1 holds on both,
   and the two together leave "different basis, same function" open; cost: minutes, from the
   records' m̄ and each checkpoint's W_U, post hoc; could change: how the close-out words e3's
   break). To the close-out, marked post hoc.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c4peg_read counts --out <dir>`, then
`HF_HOME=<HDD_1TB>/mets_data/hf … arm --out <dir> --arm {A0,F,A0p,Fp}` (GPU), then `… read --out
<dir>` (the `mets` env).

## C4′ (e)+(g) unit 0, the scan and design (2026-11-23)

Blocked 169, decided by the user 2026-11-23: of ROADMAP C4′'s unscheduled rows, the two that need
no training, **(e)** (what the tokens' common direction is and what writes it) and **(g)** (whether
the two end states differ in function), come first, then (d) and (f). One design for both, as
C4′ (a)+(b) had. Writing only; no forward pass, no record read beyond what earlier units showed.

| item | where |
|---|---|
| scan: five searches, every outside row [S]; Altıntaş et al. (Adam breaks linear connectivity, warmup restores it) gives lean g1 before output; 2609.02959's least-squares unigram direction is E1's planted effect | `lit-C4peg.md` |
| design: E1 (the mean through W_U against log unigram frequency, gated per checkpoint), E2 (the exact split of each fall of ‖m‖² into renormalisation, attention, MLP), E3 (cosine between the arms' means, scaled by the replicate pair); G1 (the barrier along the line, null released vs A0, effect seed 1), G2 (mean-ablating each arm's heads ≥ 0.5, against 20 random layer-matched sets) | `design-C4peg.md` |
| cells, fixed now | all 29 steps on A0, F, A0′, F′; four windows per arm (the collapse, V's date, the centred date, 4000) |
| leans, stated now (from the scan) | e1–e3, g1–g3 (`design-C4peg.md`) |
| exit, budget | the design's "Exit condition and budget": 3 units + close-out, GPU under 2 h |
| decisions | D1–D4, STATE Blocked 170 |

**Checked against the tree:** pair 2's π permutes rows below `HELD_OUT` (`c4pc_train` refuses
otherwise), so one set of unigram counts and one held-out set serve both pairs. The checkpoints'
config: `use_parallel_residual` true, no tied embeddings, float32. Pythia's `embed_out` has no bias.

**After `/challenge-pr` on #366** (accept with changes, reviewed `649e7db`; all five taken):
(1) G2's random sets, as first written (layer-matched, score < 0.2), could not be built. A0 had
one such set, mostly F's heads; A0′ had none. The head records give A0′ = A0's four and F′ five
(L2H2, L2H5, L2H6, L3H7, L4H1). The sets are now drawn outside the 11-head union and are not
layer-matched. (2) E2's R is the writes' dilution, not a writer, so falls are labelled mode
(removal or dilution) × writer. (3) G1's scale is floored at the null's barrier and 0.01 nats,
and the labels are renamed to "within / beyond the order-only spread". (4) The C5 promise is
dropped, since nothing here measures coordinate-basis alignment. (5) Each lean is marked [scan]
or [arms], and D4 says which leans could be registered. `LESSONS.md` 11 gains the instance.

**Parked:** `lit-C4peg.md` Parked 1 (2609.02959's full text; read before U1's code, D1).

## C4′ (c) on one page (closed 2026-11-22)

**Asked** (`design-C4pc.md` "What it is for"): one pair of runs carried all of C4's post hoc
leans. So (c) trained a second pair, A0′ and F′, from seed 0's init on another order of C4's
rows (π, a seeded permutation). It asked three things: the run-to-run spread of every date
(item 6), R2 and R3 under design-C4's rule with its rerun term (M2), and five of the leans as
registered entries on a pair they were not formed on (`P-SCH2`–`6`, items 7–10). **Exit met at
the budget** (U0 design, U1 harness and gates, U2 the arms, U3 reads and scores, this close-out;
no extension). The five entries were scored, every gate passing, and M1–M4 written. About 59
GPU-hours of training and 15 minutes of reads.

**Established** (70m, one init, one reordering; four registered results, the rest tier 1):

1. **The order-only spread is small on the dense grid.** A0′'s centred date is A0's on every
   gated τ (640), and its V date is one grid step earlier (320 against 384; q 1.2). F′'s centred
   date is F's (48); its V date is 32 against 24 (M1, M4).
2. **C4's labels do not change with that spread** (M2): R2 "neither", R3 "neither".
3. **On the replicate pair the centred label is again "neither" (48 against 640), but V reads
   "consistent with cumulative"** (M3): F′'s 32 is in [22.9, 55.9], and stays there if either
   arm moves one grid step. C4's "neither" on V is the reading at the edge (24 against 29.0).
   So C4's "undetermined" R3 went the summed rate's way on the second pair.
4. **Four of the five registered leans hold** (U3; scoring Blocked 168):
   - L8 (`P-SCH3`): V one block earlier on F′.
   - L9 (`P-SCH4`): induction at the first pair's dates (320, 768), nearer the summed rate than the centred date.
   - L9h (`P-SCH5`): F′'s top head in layer 2, A0′'s in layer 3.
   - L10 (`P-SCH6`): F′ off A0′'s path early; F beyond A0′'s distance from A0 at 12 of 12 late steps.
5. **L7 (`P-SCH2`) does not hold.** F′'s transient {T3, T4, T5}, both end sets {T1, T2} and
   T2-before-T1 (A0′ T2 at 640, T1 at 1000; F′ 48, 64) all replicate. But A0′ has a transient
   {T3} at one grid step (640) on 5 of 7 prompts, at two of three τ, so "A0′ has none" fails.

**C4's leans (`status-C.md` "C4 on one page", items 6–10), after (c):**

| item | lean | C4′ (c) | where |
|---|---|---|---|
| 6 | the label's margin: the summed rate excluded only by 1.73 in A0's date, so the exclusion may not survive a measured spread | **split** (the lean is that the exclusion is fragile; corrected after `/challenge-pr` on #365, finding 2, which found the first wording inverted). **Not borne out for the centred count**: the order spread is 1.0, below 1.73, and the replicate is "neither" too, so the exclusion survives one draw. **Borne out on V**: the replicate reads "consistent with cumulative", so on V the summed rate is not excluded on 70m | M1–M3 |
| 7 | F's transient, a shared end set {T1, T2}, T2 before T1; the harness arm has no transient | **broken as registered** (`P-SCH2` fails): A0′ has a one-step transient. F′'s transient, the end set and the order replicate (description) | U3 L7 |
| 8 | V one block earlier on F | **held** (`P-SCH3`, registered) | U3 L8 |
| 9 | induction follows the summed rate more nearly than the counts; F's top head in layer 2 | **held** (`P-SCH4`, `P-SCH5`, registered) | U3 L9, L9h |
| 10 | F is not A0 on another clock; F's late departure beyond a same-init replicate | **held early; late held by the rule but thin**: the factor over A0′ falls from 7.5 at 320 to ≤ 1.32 from 1500 on, so the departure is early (`P-SCH6`, registered) | U3 L10 |

**Not established.**
- The spread from one reordering is one draw. q = 1.0 on the centred count is at the dense
  grid's resolution (512–640–768).
- L7's miss turns on one grid step (640) on A0′. V's "cumulative" survives a one-step move of
  either arm, but both pairs sit near the band's lower edge. A second replicate could move
  either. A finer grid would have to be on A0′, which means re-training it over that span, and
  that is not costed. The parked finer grid (U2's Parked 2) is on A0 near 192–384 and sees
  neither (corrected after `/challenge-pr` on #365, finding 3).
- After 1500, F's late distance from A0 is not shown to exceed what a reordering gives.
- Cause. What clock induction follows within the summed rate's family (rate or loss).
- Another init. Anything on 410m, the paper's model.

**Kosson et al. (2410.23922), `lit-C4pc.md` Parked 1, read at this close-out.** GPT-2 124M on
OpenWebText (AdamW β2 0.95, 5000 iterations). Without warmup they show large early ℓ2 and
angular updates and a "large, sudden" representation change (their §§3, 5, 6, the last a
conjecture), but they date no transient by step. They report nothing on data order or on
heads. So for item 7 they give a mechanism-level reference (early updates too large, which
predicts the transient survives a reordering, as F′'s did), not a dated one. It changes no
cell. §7's new paragraph does not cite it.

**In `paper/`** (this close-out): §7 gains one paragraph after C4′'s, "A replicate pair":
- the design: same init, a reordering, five readings registered before the first update;
- the spread (640, 640; 320 against 384);
- M2 unchanged, and M3: centred "neither" (48 against 640), V "consistent with the summed rate" (32 in 23–56, robust to a one-step move);
- four registered readings that hold, with the late margin thin (7.5 at 320, ≤ 1.32 from 1500);
- L7's failure.

The record side: 20 `c4pc.*` keys from `scores.json` and `measured.json` (two new records),
4 claims, 14 slots and 1 literal (1500). `numbers produce` (`.venv`) re-read the registry
(changed by `87e996b`); no other key moved. `paper_check` 0; 37 pages, no undefined reference.
`scores.json` was re-written at this close-out to carry r_F and r_A0′ per late step (the five
scores and cells unchanged; `c4pc_read` `320a41794f52`). The abstract is not touched. In §1,
one sentence is corrected to match (after `/challenge-pr` on #365, finding 1): it said the flat
arm dates both counts earlier than either map and that the harness schedule shows no transient.
It now says that holds for the centred count on two pairs, that E1's is undetermined on the first
and compatible with the summed rate on the second, and that the harness arm's transient is at most
brief. §7's C4 paragraph now says "the centred date" where it said "them". The new paragraph says
"farther, but by less than a third" late and "both pairs near the band's lower edge". The claims
check both, and not the one-step robustness, which `status-C.md` U3 checks by hand.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan and design | #361 | proposed; D1–D4 all (a), Claude on the user's delegation (Blocked 166) |
| 1. harness, π, rules, gates, drafts | #362 | gates for L7–L9h pass; H1′, H2′ pass; registered hash `d7e674313486` |
| 2. registration, then the arms | #363 | `P-SCH2`–`6` registered before any update (Blocked 167 (a)); both arms to 4000; H4′ passes |
| 3. the 3500 gate pinned, reads, scores, M1–M4 | #364 | L7 fails, L8–L10 hold; M3's V "cumulative" |
| close-out | this | this page; `paper/` §7; Blocked 168 still the user's (*done 2026-11-26, #386*) |

**Parked items at close:**
- U0, `lit-C4pc.md` Parked 1 (Kosson et al.): **read here** (above). Not read before U3's
  reading as the scan asked; it could change wording only, and it did not.
- Carried in and not taken: the two `c2_seeds` refusals and U2's Parked 2 (the finer A0 grid,
  ≈ 1 h) **stay with ROADMAP §8 C4′**, unscheduled. The finer grid is on A0, so it bears
  on C4′'s nearest approaches (192–384), not on L7's transient or M3 (both on A0′).
- U1–U3: no Parked items.

**Re-run.** U3: its section below. This page's numbers:
`METS_DATA=<main>/data <main>/.venv/bin/python -m trackC_writeup.numbers produce`, then
`python -m trackC_writeup.paper_check` (0).

## C4′ (c) unit 3, the reads and the scores (2026-11-22; done)

**Input:** A0′ and F′ (`HDD_1TB/mets_data/c4pc/{A0p,Fp}/ckpt/`, the 29 saved steps, `c4pc_train`
`95e71d5e7147`); v1's 7 prompts; C4's 64 held-out rows; C4's pair and C4′ U2's K records (main
tree). Out: `data/trackC/c4pc/2026-11-22-u3/` (main tree). Code `c4pc_read.py` `9b6646587037`
(pushed `8a40fb8` before any output; `scores.json` re-written at the close-out by `320a41794f52`,
which adds r_F and r_A0′ per late step and leaves the scores unchanged); rules `c4pc_rules.py` `d7e674313486` (the registered hash,
unchanged); instruments `c4_read` `220161b9159b`, `c4_close` `019846e1561b`, `c4p_path`
`8dffeea7c9ed`, all unchanged. GPU (RTX 3080), float32 eager, K in float64.

| step | state |
|---|---|
| C4's gate at 3500 on F′ (`gate3500`; `c4_read.gate` on an alias root A0 → A0p, F → Fp) | **as frozen**: plant B's dose 0.5 by rule 100 (a), V read; τ 9.61e-5, 1e-4, 3e-4 pass, 4.78e-3 does not (C4's F: the same three); re-init rates equal unit 3b's. 23 s. Record `gate3500/gate.json` |
| the pin (`/challenge-pr` on #363, finding 4) | `trackC_writeup/c4pc_gate3500.json` (the record's sha256 `6f12d0b7…`) committed `402d01d` before any record of the new arms existed. `require_scorable`, and every U3 driver, refuse unless the pin is tracked, clean and matches the record. The first commit was refused: `.gitignore`'s allowlist ignored the pin (and `c4pc_declined.json`); both allowlisted |
| records, heads, K (`u3_chain.sh`, runner) | exit 0, 13 min 48 s (`stages.txt`). Records on both arms' 29 steps; heads (`heads_{A0p,Fp}.json`); K jobs `A0p` (A0's 29 cached states linked from C4′ U2, against A0′'s 28) and `Fp` (A0′'s 29 against F′'s 28), M 57, 226 s and 222 s, logits check 0.0 |
| L10 (b)'s gate (`l10b_gate.json`, before `K_Fp`) | **passes**: the null beyond r_A0′ at 0 of 4 late steps; 9 of 9 seeds at 4 of 4 |
| scores (`scores.json`) | below |
| M1–M4 (`measured.json`) | below |

**The five registered entries** (every gate passed; each scored alone):

| entry | holds? | cells |
|---|---|---|
| L7 (`P-SCH2`) | **no** | F′'s transient {T3, T4, T5} at all three arms and both end sets {T1, T2}, as C4's pair. A0′ also has a transient, {T3}, at 9.61e-5 and 1e-4 (not at 3e-4), so "A0′ has none" fails at 2 of 3 arms. A0′'s is one grid step, 640 (its centred date), on 5 of 7 prompts; F′'s runs 64–1000 on 6–7 of 7 (beside, not in the rule) |
| L8 (`P-SCH3`) | yes | V's minimum at 1000 / 2000 / 4000: F′ T1 (sets {1, 2}, {1}, {1, 5}), A0′ T2 ({2} at each) |
| L9 (`P-SCH4`) | yes | induction F′ 320, A0′ 768 (C4's pair: the same dates); (i) 320 < 397.98; (ii) d_induction 0.439 < d_centred 1.093 (F′ 48, A0′ 640) at all three arms |
| L9h (`P-SCH5`) | yes | top head at 4000: F′ L2H2, A0′ L3H6 (C4's F L2H6, A0 L3H6) |
| L10 (`P-SCH6`) | yes | (a) F′ "another path" against A0′ early: 0 of 16 on, monotone (late: also "another path", 0 of 12). (b) r_F(s) > r_A0′(s) at 12 of 12 late steps, by a falling factor: 7.5 at 320, 3.8 at 512, 2.0 at 1000, 1.3 at 2000, 1.2 at 3500 and 4000 |

**Measured (not registered):**

| | what | read |
|---|---|---|
| M1 | A0′'s dates against A0's, q | V 320 against 384 (q 1.2, one grid step); each centred arm equal (640, 640, 640, 768; q 1.0) |
| M2 | R2 and R3 on C4's pair at ρ = max(`rho_at`(t_A0), q) | unchanged: R2 "neither", R3 "neither" (V: ρ 1.33, cumulative band [29.0, 91.7], F's 24 outside). q 1.2 is below R3's flip at 1.47 |
| M3 | R2′ and R3′, F′ against A0′ | **R2′ (centred) "neither"** at every gated arm (F′ 48, A0′ 640). **R3′ (V) "consistent with cumulative"**: F′ 32, A0′ 320, ρ 1.25, band [22.9, 55.9] (step's [256, 400]) |
| M4 | F′'s dates against F's | V 32 against 24 (q 1.33); centred equal (48, 48, 48, 64) |
| beside | A0′ on A0's path (`l10b_gate.json` `A0p_vs_A0`) | early "one path", 16 of 16 on, all at their own step; late "mixed", 6 of 12 |

**What this does not establish.** One order replicate is one draw of the spread.

- **On V the fragile reading is C4's "neither", not the replicate's "cumulative"** (corrected
  after `/challenge-pr` on #364, finding 1; the first push said the reverse). C4's pair sits at
  the band's lower edge (F's 24 against 29.0). R3′ holds if either arm moves one grid step either
  way: A0′ 256 or 384, F′ 24 or 48. Both cross pairings also read cumulative (F against A0′,
  24 / 320; F′ against A0, 32 / 384).
- That the centred dates match exactly on both arms is at the dense grid's resolution (the
  512–640–768 steps).
- L7's miss is a one-step, 5-of-7 event on A0′. The rule as registered counts it, and calling
  it noise would be a post hoc reading.
- **L10 (b)'s hold is thin, and its departure is early** (finding 2). r_F / r_A0′ is 7.5, 6.8,
  3.8, 2.8, 2.1, 2.0 and 1.7 at 320–1250, then 1.28, 1.32, 1.16, 1.19 and 1.17 at 1500–4000. The
  rule needs 8 of 12, so the hold depends on at least one of those last five. From 1500 on, F is
  about as far from A0 as one order replicate is.
- Items 6–10 are not yet marked held or broken, and `paper/` §7 is not edited: both are the
  close-out's. Item 10's wording should say the departure is early.

**Provenance of the 3500 gate** (finding 3): the record names no checkpoint. The alias root's
links (`gate3500/root/F` → `HDD_1TB/mets_data/c4pc/Fp`, `A0` → `…/A0p`) are what bind it to F′.
`heads` now refuses a reused file whose code is not `c4_close`'s (this run's are `019846e1561b`).

**Not done here:** scoring `P-SCH2`–`6` in the registry is a `notes` sentence per entry, as
`P-SCH1`'s was (Blocked 157 (a)), and the commit is the user's (STATE Blocked 168; *made
2026-11-26 on the user's instruction, #386, `52470c8`*). The drafts,
each to be appended to the entry's `notes` and prefixed "Read 2026-11-22 (C4′ (c) U3,
`data/trackC/c4pc/2026-11-22-u3/scores.json`, rules `d7e674313486`, every gate passed):":

| entry | draft sentence |
|---|---|
| `P-SCH2` (L7) | does not hold, every cell readable; falsifier met. F′'s transient {T3, T4, T5} and both end sets {T1, T2} are as stated, but A0′ has a transient {T3} at τ 9.61e-5 and 1e-4 (one grid step, 640, 5 of 7 prompts). |
| `P-SCH3` (L8) | holds at 1000, 2000 and 4000: F′'s lowest V transition T1, A0′'s T2. |
| `P-SCH4` (L9) | holds: induction F′ 320, A0′ 768; (i) 320 < 398; (ii) 0.44 < 1.09 at the three gated arms. |
| `P-SCH5` (L9h) | holds: F′ L2H2, A0′ L3H6. |
| `P-SCH6` (L10) | holds: (a) "another path" early, 0 of 16 steps on A0′'s path; (b)'s gate passed and r_F > r_A0′ at 12 of 12 late steps, by a factor falling from 7.5 at 320 to 1.2 at 3500 and 4000. |

`lit-C4pc.md`
Parked 1 (Kosson et al.'s full text) was not read before this reading. It can change item 7's
wording, not a cell, so it goes to the close-out.

**Re-run:** from `../Mets-work`, the commands in `c4pc_read.py`'s docstring in that order. The
forward passes are `data/trackC/c4pc/2026-11-22-u3/u3_chain.sh` (idempotent). `gate3500` refuses a
second run, and the pin binds the first; `score` and `measure` take seconds.

## C4′ (c) unit 2, registration and the arms (2026-11-18 to 2026-11-22; done)

| step | state |
|---|---|
| Blocked 167 | decided (a), the user's, 2026-11-18: `P-SCH2` (L7), `3` (L8), `4` (L9), `5` (L9h), `6` (L10) registered as drafted in `design-C4pc.md`, commit `87e996b`, hashes verified then (`c4pc_rules` `d7e674313486`, `c4pc_train` `95e71d5e7147`, `c4_train` `1c1e1d2df083`). One field moved: `gate` stays empty and `calibration_record` null (`check_registry` refuses a gate on a measurement entry, and wants the record tracked); the U1 gate record is cited in `null_construction` with its sha256 `952b73ba…83c0` |
| the chain | D4 (a), each step gating the next: `train --arm A0p --until 1000` → `h4p --steps 1000` → `train --arm Fp --until 4000` → `train --arm A0p --until 4000` → `h4p --steps 4000`; started 2026-11-18 08:53 CST under `systemd-inhibit`; conda `mets` (torch 2.10.0+cu130, transformers 4.44.2), RTX 3080, n_micro 512, compiled, bf16 autocast, fp32 master |
| first output | populated: A0′ step 1 loss 11.083 (C4's A0 11.081 on its own batch, same init), ≈ 27 s an update (C4's 26.5) |
| A0′ → 1000 | done 2026-11-18 16:22 CST (7.5 h; loss 3.648 at 1000, no divergence) |
| H4′ at 1000 | **passes**: held-out loss A0′ 3.7151, A0 3.7200, rel. diff −0.13 % (tol 2 %; the seeds' gaps 1.51 % at 1000, 0.57 % at 4000); `HDD_1TB/mets_data/c4pc/A0p/h4p_1000.json`. In the chain it ran out of GPU memory (fp32 logits on 8 rows × 2048 × 50 304, 2.31 GiB reserved but free: fragmentation) and stopped it; rerun alone with `PYTORCH_ALLOC_CONF=expandable_segments:True`, an allocator setting, no code or number changed; the chain's last `h4p` carries it |
| F′ → 4000, then A0′ → 4000, H4′ at 4000 | chain resumed 2026-11-18 16:22 CST; F′ step 1 loss equal to A0′'s (same init, same first batch, the rate not yet applied), steps 2–3 10.115 / 9.509 (C4's F 10.097 / 9.521) |
| stopped, resumed | stopped by the user 2026-11-18 21:19 CST at F′ step 636 (log `u2_logs/u2_seg2.log`); resumed 2026-11-19 04:56 CST from `Fp/resume.pt` at 612 (segment 3 of `u2_chain.sh`, under `setsid`). Step 613's loss equals the first pass's to every printed digit (3.564080652548); 614–620 differ ≈ 1e-5 relative (non-deterministic kernels; the log's last entry per step wins, `prior_losses`). Stopped again 2026-11-19 19:53 CST at F′ step 2587, exit 143 (a stop the user began; the session dropped mid-stop; log `u2_logs/u2_seg3.log`); resumed 2026-11-20 04:59 CST from `resume.pt` at 2575 (segment 4, `setsid nohup systemd-inhibit`). Step 2576's loss equals the first pass's to every printed digit (3.088707319344); step 2580 3.09461 vs 3.0946. F′ reached 4000 2026-11-20 15:46 CST. A0′ 1000 → 4000 stopped by the user 2026-11-20 23:45 CST at step 2059, exit 143 (log `u2_logs/u2_seg4.log`; `A0p/resume.pt` 23:41); resumed 2026-11-21 09:45 CST from `A0p/resume.pt` at 2050 (segment 5 of `u2_chain.sh`, `setsid nohup systemd-inhibit`, log `u2_logs/u2_seg5.log`), then H4′ at 4000. Step 2051's loss equals the first pass's to every printed digit (3.189226919785142); 2052–2059 differ ≤ 3e-5 relative |
| A0′ → 4000, H4′ at 4000 | done 2026-11-22 00:51 CST; **H4′ at 4000 passes**: held-out loss (rows 4 096 000–4 096 064) A0′ 3.10407, A0 3.10705, rel. diff −0.096 % (tol 2 %); `HDD_1TB/mets_data/c4pc/A0p/h4p_4000.json`, code `95e71d5e7147`. Chain exit 0 at 00:52 (`u2_logs/stages.txt`, `u2_seg5.log`) |
| the arms at the end | both `train_log.jsonl` hold every step 1–4000, all losses finite (no divergence). Re-trained windows (logged twice; the last entry wins): F′ 613–636 and 2576–2587, A0′ 2051–2059. The first step of each is equal to every digit; the largest relative gap from the first pass is 2.4e-4 (F′ 636), 7.3e-5 (F′ 2587) and 2.7e-5 (A0′ 2059), from non-deterministic kernels, which grow over the window (corrected after `/challenge-pr` on #363, finding 3; the row above quotes 614–620 only). Train-batch loss A0′ / F′: 3.648 / 3.300 at 1000, 3.248 / 3.123 at 2000, 3.123 / 3.058 at 3000, 3.037 / 2.993 at 4000 (C4's A0 / F: 3.682 / 3.333, 3.240 / 3.116, 3.119 / 3.056, 3.042 / 2.990; different batches, so beside only). Checkpoints on both: 0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1000, 1250, 1500, 2000, 3000, 3500, 4000 (`{A0p,Fp}/ckpt/`) |
| C4's gate at 3500 on F′ | moved to U3: it reads F′'s records, which U3 computes. The design's plan item 2 puts it in U2, and `c4pc_read.require_scorable` checks only U1's gate record, so **U3 must first commit that gate's record and add a refusal for it to `require_scorable`, both before any score** (`/challenge-pr` on #363, finding 4) |

**U2's exit is met** (the design's plan item 2, except the 3500 gate above): both arms trained; H1′, H2′ and H4′ pass; no divergence. Nothing read; no Parked items. Next, U3: the 3500 gate's refusal and record; records, induction and heads on both arms; the K passes; L10 (b)'s gate; then M1–M4 and L7–L10.

**Re-run from scratch** (the chain run end to end; `u2_chain.sh` was rewritten for each resume and now holds only segment 5, A0′ → 4000 and H4′ at 4000, so on its own it would skip F′; `/challenge-pr` on #363, finding 1). From `../Mets-work`, with `METS_REPO=$PWD METS_DATA=<main>/data HF_HOME=HDD_1TB/mets_data/hf HF_HUB_OFFLINE=1`, the `mets` env's python: `c4pc_train train --arm A0p --until 1000` → `PYTORCH_ALLOC_CONF=expandable_segments:True c4pc_train h4p --steps 1000` → `train --arm Fp --until 4000` → `train --arm A0p --until 4000` → `PYTORCH_ALLOC_CONF=expandable_segments:True c4pc_train h4p --steps 4000`, each with `&&`. Every `train` resumes from its arm's `resume.pt`; a finished arm prints `nothing_to_train`. The stages and segment 5's exit are in `u2_logs/stages.txt`; segments 1–4 also wrote `u2_seg{1..4}.exit`. There is no `u2.exit`.

## C4′ (c) unit 1, the harness, π and the gates (2026-11-18)

Blocked 166 decided all (a) by Claude, on the user's "[your choices]" (`design-C4pc.md` "Decided
2026-11-18"; yours to overrule). **The registered rules: `c4pc_rules.py` `d7e674313486`**;
drivers `c4pc_read.py` `fcb8d2eb3f91`; arms `c4pc_train.py` `95e71d5e7147`; 13 CPU tests. The
first push (`52303d8`) kept the rules in `c4pc_read.py`; after `/challenge-pr` on #362 (accept with
changes, findings 1–4 taken) they were split out, the decline record moved into git, and C4's
divergence rule put inside the loop, all before any update; H1′ and H2′ ran on `6517b85f75b0`,
whose `write_permuted`, `verify` and `init` are unchanged since. **No arm has taken an update**:
`train` refuses until the five entries are registered or declined (Blocked 167).

| check | record | reads |
|---|---|---|
| L7–L9h gates (effect C4's pair; null (released, A0); seeds 1–9 against released) | `data/trackC/c4pc/2026-11-18-u1/gate.json` | **all four pass**: effect holds, null readable and not holding, 0 of 9 seeds hold |
| H1′ (step 0 of A0′, F′ bit-equal to the released) | `HDD_1TB/mets_data/c4pc/{A0p,Fp}/h1.json` | **pass**, 76 tensors each, none unequal |
| H2′ (π's file: 4096 sampled rows against the shard) | `HDD_1TB/mets_data/c4pc/perm_meta.json` | **pass**, 0 mismatches; 16 785 408 000 bytes; π (seed 20261118) sha256 `06887e00e6eb…`, file `b55ea350c548…`; batch 0 `7cc18a4b63be…`, max token 50276 |
| H4′'s tolerance (max(2 %, the seeds' largest gap at 1000 and 4000)) | `c4/2026-11-09-u1/h4_controls_1000.json`, `c4pc/2026-11-18-u1/h4_controls_4000.json` | **2 %** (the base): the seeds' largest gap is 1.51 % at 1000 (seed 3) and 0.57 % at 4000 (seed 9); warmup01, not a seed, 44.6 % and 12.3 % |

**What the seed null shows** (corrected after finding 3 on #362): C4's own pair re-read on
Pythia's 13 steps still holds all four (L9: 512 against 1000, bar 591; `gate.json` `beside`), so
the rules can fire at the nulls' resolution. The seed null therefore says order and init spread
on one schedule do not separate induction's dates there (every seed and the released run date
1000). It cannot see spread below one grid interval (512–1000). **L9 is one dense step from
failing on a replay of F**: an A0′ date of 640 instead of 768 puts the bar at 303, below F's 320.
Near misses: seed 1 has L8's pattern at 4000 only; seeds 5 and 8 put the top head in layer 4,
none in 2.

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c4pc_read gate --out <dir>` (seconds,
no GPU); `c4pc_train write-perm --tmp <nvme dir>`, `verify`, `init --arm {A0p,Fp}`; `c4_read
h4-controls --step 4000` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the first try ran
out of GPU memory from fragmentation).

## C4′ (c) unit 0, the scan and design (2026-11-17)

Writing only (Blocked 165 (a), the user's). Nothing trained, computed, downloaded or registered.
The one read of a record: C4's `close.json` `transitions`, to fill the design's known-answer
column for L7 and L8. Those transitions were seen at C4's close-out; the per-arm sets at the other
τ arms were not tabulated there.

- **Scan** (`lit-C4pc.md`, three searches, summaries only): no order-only variance study in
  pretraining and no replicated warmup-removed run was found. Dodge et al. (fine-tuning) find
  init and order contribute comparably. 2511.16893 finds induction's emergence set by batch and
  context and robust over 3 seeds. Kosson et al. tie warmup to the early updates' size. So the
  design states every lean against A0′, and recommends an order-only replicate.
- **Design** (`design-C4pc.md`): A0′ and F′ from seed 0's init on order π, C4's harness through a
  new module (`c4_train.py` untouched). Measured: M1 A0′'s dates and rerun ratio q; M2 R2 and
  R3 on C4's pair at ρ = max(grid, q) (design-C4's rule; `P-SCH1` not re-scored; V can flip,
  the centred label cannot); M3 R2′ and R3′, the paired replicate that item 6's wording follows;
  M4 F′ against F. Registered family, five entries: L7 transient and end set, L8 V one block
  earlier, L9 induction nearer the summed-rate map than step and than the centred date, L9h the
  top head's layer, L10 F′ off A0′'s path early and F's late departure beyond A0′'s. Gates: C4's
  pair (known answer; holds by construction), (released, A0) as the null, a seed null (seeds 1–9
  against the released run), L10 (b) on the seeds. Registration before either arm's first update.
- **Found** (each changes Blocked 165's premises): (1) shard 0's unused rows hold 3 224 580 rows,
  3149 updates, so A0′ cannot reach 4000 on them (math check (3)); (2) C4's rule takes the
  plateau at 1000–4000 and the gate at 3500, so a pair stopped at 1000 reads none of C4's dates;
  (3) R3 needs A0′ only (item 6, first written so too, needs the pair: #361, finding 1); (4) the head overlap measures the layer (U1's A2), so L9h is a
  layer statement. (1) is in `LESSONS.md` 1.
- **Math check** `tools/math_checks/c4pc_nearer.py`: log-nearer iff below the geometric mean;
  the threshold t^{3/2}/√(2W), 398 at 768; the label's cap (2W/t)^{1/3}; the shard's row count.

**`/challenge-pr` on #361: accept with changes, findings 1–5 taken** (the PR's thread):
(1) M2 cannot flip the centred label (its bands meet at ρ 1.647 < 1.727; math check (2b)), only
V's; (2) §7's item-6 wording follows the paired M3, not the unpaired M2; (3) registration moves
before any update, since F's transient is a majority at 48–64; (4) the gates for L7–L9h pass by
construction on both sides, so a seed null (seeds 1–9 against the released run, ≤ 1 of 9 "holds")
is added; (5) H4′'s tolerance is max(2 %, the seeds' held-out loss gap).

**Parked:** `lit-C4pc.md` 1, Kosson et al.'s full text (could change item 7's wording in §7, not
the reading; read before U3 or dropped at the close-out).

## C4′ on one page (closed 2026-11-17)

**Asked** (`design-C4p.md` "What it is for"): two of C4's post hoc leans, tested without
training, each with its rule fixed before its output. (a) Does induction's date on a run C4 did
not train, `pythia-70m-warmup01` (ten times the ramp), follow the summed rate (C4 item 9)? (b) Is
C4's flat arm F the harness arm A0 on another clock (item 10 says no)? **Exit met at the budget**
(U0 design, U1 (a), U2 (b), this close-out; no extension): A1 labelled, A2 and A3 written, (b) read
in both windows, items 6–10 marked below. About 2 h of GPU in all (U1 about 80 min at 23 s a model, U2 about 1 h). Nothing registered: A1's
entry was drafted and declined (Blocked 160 (b)), and (b) was tier 1 by design (D4).

**Established** (tier 1; one run or one pair of runs, 70m):

1. **Induction on warmup01 reads "consistent with cumulative"** (A1, U1): the top head's
   half-rise is in (2000, 3000] against the ten 70m seeds' (512, 1000], a delay of 2× to 5.9×
   inside the summed rate's (1619, 3162]. The rule's leave-one-out gate passed 10 of 10 on every
   schedule, but easily (each seed rises in one grid step). The label covers "the summed rate or
   a function of it"; a fixed-loss clock is not excluded (the loss bracket is 0.73 nats wide).
2. **F is not on A0's path at any saved step** (U2): 0 of 28 F steps are within A0's own
   step-to-step KL of any saved A0 state (1.25–4.64 δ at the nearest; ≥ 1.25 δ over all states),
   with the gate passing in both windows (the released run "one path" at 8 of 9 and 3 of 4 steps,
   the late window exactly at the ⅔ bar; 9 of 9 seeds "another path"). **Against the same-init
   null, early (1–256) by a wide margin**: the null is within 9.2 × 10⁻⁵ through 128 and 0.028 at
   256, F 0.063–0.67; and at each shared step from 8 to 512, F's distance to its nearest A0 state
   exceeds every seed's distance from the released run at that step (by 1.07× at 8; not at 2 and
   4). **Late (320–4000) by a narrowing one**: F's r over the null's falls from 6.7 at 512 to 1.9
   at 4000, as the null itself drifts off A0. **In δ the margin is not wide in either window**:
   the four nearest approaches, 1.25–1.35 δ, are at 192–384, two of them in the early window.
   (Corrected after `/challenge-pr` on #360, finding 1: first written "early by a wide margin"
   with no reference, and Blocked 164 had quoted the early window as "3–5 δ"; it is 1.25–4.64 δ.)

**Beside (description; no label):** A2, every head ≥ 0.5 on the ten seeds and on warmup01 is in
layer 3 or 4, while F's top head is L2H6 (U1). A3, warmup01's per-transition centred set has a
T3 majority at 2000–4000 that seed 0 never has; whether either has a transient turns on the
horizon (U1). From step 48, F's attention and MLP displacement from init is near orthogonal to
A0's at the match (cos 0.06–0.19 against A0's own 0.26–1.0); F's state sits near A0's state of
equal loss (U2).

**C4's leans (`status-C.md` "C4 on one page", items 6–10):**

| item | lean | C4′ | where |
|---|---|---|---|
| 6 | the label's margin: the summed rate excluded only by 1.73 in A0's date | **untouched** (no run-to-run spread measured; the seeds show none on Pythia's grid) | U1 |
| 7 | the arms' counts rise for different reasons (F's transient), a shared end set {T1, T2}, T2 before T1 | **untouched** as a reading; A3 beside: the end set and T2-before-T1 hold on warmup01, the transient turns on the horizon | U1 "A3" |
| 8 | V sits one block earlier on F | **untouched** | — |
| 9 | induction's date follows the summed rate more nearly than the counts | **held** (tier 1, one run; "the summed rate or a function of it", loss not excluded) | U1 "Second half" |
| 10 | F is not A0 on another clock | **held**: against the same-init null, by a wide margin early and a narrowing one late; in δ, by 1.25–1.35 at the nearest (192–384) | U2 |

**Not established.** Whether F's late departure is beyond what two same-init runs on other data
orders show (the null drifts to 0.19 by 4000; Parked 3 below). Whether a finer A0 grid moves the
steps nearest the bar, 192–384 at 1.25–1.35 δ (it lowers r and δ both). The run-to-run spread of
any date. What clock induction follows inside the summed rate's family (rate or loss). Cause.
Anything on 410m, the paper's model.

**In `paper/`** (this close-out): §7 gains one paragraph after C4's two, "Is the flat arm the
harness arm on another clock?": the rule (fixed before output, not registered), the gate (the
null's 8 of 9 and 3 of 4, the late window at the bar), none of 28 steps on the path at 1.25–4.64
times the resolution, the margin against the same-init null (early within 9.2 × 10⁻⁵ through 128
against F's 0.06–0.67; beyond every seed at each step from 8 to 512, not at 2 and 4; late the
null's 0.064 → 0.19 against F's 0.43 → 0.36), "not wide anywhere" in δ, and the finer-grid caveat.
14 `c4p.*` keys from `reading.json` and `gate.json` (no other key moved), 2 claims, 17 slots
(revised after `/challenge-pr` on #360, findings 1, 2, 4); `paper_check` 0; 37 pages, no undefined reference. A1 is not added (Blocked 164 (a) named
the (b) sentence only); the abstract and §1 are not touched.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan and design | #350 | (a) and (b), each gated on the real design points; D1–D5 frozen all (a) |
| 1. (a): A1, A2, A3 | #351, #352 | gate 10 of 10; A1 "consistent with cumulative" (entry declined, tier 1) |
| 2. (b) | #359 | gate passes both windows; F "another path" early and late |
| close-out | this | this page; `paper/` §7; Blocked 165 |

**Parked items at close:**
- U0, `lit-C4p.md` Parked 1 (2509.26643's full text, "before U2's code"): **dropped**. It was not
  read before U2 (`LESSONS.md` 1); what it would have supplied, a seed scale for (b), U2
  computed itself (K(released_t, seed_t) on all nine seeds at the 12 shared steps), so it could
  no longer change (b).
- U1: none.
- U2 Parked 1, F against A0's path at 410m: **dropped** (two 410m arms are out of reach on the
  3080, ROADMAP §8 C4′ "not reachable"); the §7 paragraph says 70m.
- U2 Parked 2, a finer A0 grid near F's induction (≈ 1 h): **carried to ROADMAP §8 C4′**, beside
  (c), unscheduled.
- U2 Parked 3, the late window against a same-init replicate: **carried to C4′ (c)**, whose
  opening and registration are Blocked 165.
- Carried in from C4 and not taken: the two `c2_seeds` refusals (no unit here derived a centred
  arm) and Qu & Liu's peak-dependence stay with ROADMAP C4′ ((d)).

**Found at the close-out.** (1) U2's line "F's r (0.33–0.67) exceeds every seed's distance (≤
0.40)" put two overlapping ranges side by side; the comparison holds step by step (at 128, 0.33
against 0.19), not between the ranges. The claim check on the new keys refused it as first
written; U2's text reworded below. (2) `lit-C4p.md` Parked 1 lapsed silently (above).

**For the user's Overleaf copy.** The diff of `trackC_writeup/paper/` from `c08de0c` to this PR's
merge: one paragraph at the end of §7's "Pythia 70m with the warmup removed".

**Re-run.** `METS_DATA=<main>/data <main>/.venv/bin/python -m trackC_writeup.numbers produce`,
then `python -m trackC_writeup.paper_check` (0). U1's and U2's records: their sections' re-run
lines.

## C4′ unit 2, (b): is F A0 on another clock? (2026-11-17)

Blocked 163 (a), the user's. The design's open details (A0's 29 states, merging equal states,
the null read against A0 in fp16 throughout, monotonicity against the running max, the
thresholds, float64 sums) were fixed in `design-C4p.md` "(b) as U2 runs it", committed and
pushed before any K (`231cb77`). A smoke test in scratch then showed that A0₀ and A0₁ differ only
in signed zeros, so states are compared by value (`c8876cd`, `LESSONS.md` 3). Code
`trackC_writeup/c4p_path.py` `8dffeea7c9ed`, with 15 pure tests (`tests/test_trackC_c4p_path.py`).
Records are in `data/trackC/c4p/2026-11-17-u2/` (main tree; sha256, first 12): `gate.json`
`528eaa4b293b`, `reading.json` `25ef5d0cbaa8`, `theta.json` `297441f75360`, `K_F.npz` `4eeac0c1664c`,
`K_null.npz` `12ca4718f511`, `K_seed1..9.npz`. GPU time: caching 207 models took about 20 min; the
11 K jobs took 200–223 s each. **Input:** rows `HELD_OUT` = [4 096 000, 4 096 064) of the C4
shard, all 131 072 positions, none subsampled. `cache/` (72 GB of hidden states and
unembeddings) is regenerated by `cache`; you may delete it.

**Known answers.** Every model's h·Wᵀ equals its own logits exactly (max |Δ| 0.0, 207 of 207).
K(A0₀, A0₁) = 0; A0₁, the released step 1 and its fp16 rounding are equal in value. The float64
floor: |K| ≤ 1.6 × 10⁻⁹ between models that differ only by rounding. A0's δ runs from
8.4 × 10⁻⁶ (steps 0–2) through 0.19 at 256 to 0.15–0.34 from 512 on.

**The gate (before F was cached): both windows readable.**

| window | null (released vs A0 in fp16) | identity | seeds "another path" |
|---|---|---|---|
| early (1–256) | "one path", 8 of 9 on | 8 of 9 | 9 of 9 (1 of 8 steps on, each seed) |
| late (512–4000) | "one path", 3 of 4 on, **exactly the ⅔ bar** (4000 off: r 0.19 > δ 0.15) | 4 of 4 | 9 of 9 (0–1 of 4 on) |

The null's r is ≤ 9.2 × 10⁻⁵ through 128 (written "≤ 9 × 10⁻⁵" until the close-out), then rises: 0.028 (256), 0.064 (512), 0.11 (1000), 0.14
(2000), 0.19 (4000). So a rerun with A0's init and order drifts away from A0, and by 4000 it is
off A0's path itself. A seed's distance to the released run at the same step,
K(released_t, seed_t), is 0.16–0.47 (0.40 at steps 2–16, a minimum of 0.16–0.20 at 256).
**The design expected the late window might be refused** (the seeds reconverge near 2k). It
was not, but only just: at 4000 a seed is still 0.37–0.40 from the released run, and the null
passes the late window at the bar. The gate's limit stands: the seeds differ in init and order,
F in neither. (Corrected after `/challenge-pr` on #359, finding 1: first written "the released
run sits within δ of A0".)

**The reading (tier 1): F is on "another path" in both windows.** 0 of 16 early steps and 0 of 12
late steps are on A0's path. The rule tests the nearest state only. Checked over every state
(after CodeRabbit on #359), no F step is within δ(g) of any A0 state g either: the smallest
K(F_s, g) / δ(g) over all 28 × 28 pairs is 1.25. The matches are monotone, so F's function moves
forward through A0's states, but never within A0's own resolution of one:

| F step | 1 | 8 | 64 | 192 | 256 | 320 | 512 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|
| match (A0 step) | 24 | 128 | 256 | 512 | 640 | 768 | 1000 | 1500 | 2000 | 4000 |
| r (nats) | 0.063 | 0.43 | 0.40 | 0.37 | 0.42 | 0.45 | 0.43 | 0.40 | 0.38 | 0.36 |
| r / δ(match) | 1.7 | 3.0 | 2.1 | 1.3 | 1.3 | 1.3 | 1.7 | 1.9 | 1.4 | 2.4 |
| summed-rate map (A0 step) | 54 | 152 | 428 | 742 | 856 | 957 | 1211 | 1716 | 2716 | past 4000 |

**Early** (1–256), r is 0.063–0.67, against the null's ≤ 9.2 × 10⁻⁵ through 128. At each shared
step from 8 to 256, F's r exceeds every seed's distance from the released run at that step
(at 8, 0.43 against 0.40; at 128, 0.33 against 0.19), although F shares A0's init and data order.
(Reworded at the close-out: first written "F's r (0.33–0.67) exceeds … (≤ 0.40)", two ranges
that overlap; the comparison is per step.) The early reading has a wide margin over the
same-init null, not in δ (see the close-out, `/challenge-pr` on #360, finding 1).
**Late** (320–4000), r is 0.34–0.47, while the same-init null's r rises toward it. F's r over the
null's is 6.7 at 512, 3.5 at 1000, 2.7 at 2000 and 1.9 at 4000. So late, "another path" is beyond
a same-init rerun's spread, but by a factor that falls, and part of it may be two runs drifting
apart (`/challenge-pr` on #359, finding 2). The closest approach to the path is 1.3–1.4 δ, at
192–384, just before and at F's induction (320). The lean (C4 item 10, "F is not A0 on another
clock") **is held**: firmly early against the same-init null (not in δ; close-out), with a narrowing margin late. (First written "r 0.06–0.45,
the size of a seed's distance"; corrected, finding 3.) No window reads "one
path", so the clock beside is not read; the matches are printed above only as description. They
run later than the step map and earlier than the summed-rate map at every step through 1000.

**Beside (description).**
- **Loss at the match.** The held-out loss of F_s minus A0's at match(s) is −0.004 to +0.66 nats
  through 32, and −0.16 to +0.30 from 48 on (within ±0.14 from 128). So F's state is near A0's
  state of equal loss, but r says the two functions still differ by a seed's distance.
- **K in reverse,** K(A0, F_s): the same match, or the next grid state, at 26 of 28 steps
  (s = 2000 matches 3500, two states later); r 0.07–0.58.
- **K(F_s, A0_s) at the same step:** 2.6–3.4 nats at 8–64, falling to 0.36 at 4000.
- **Parameters** (`theta.json`). Take cos(θ^F_s − θ₀, θ^A0_match − θ₀). For attention weights it
  is 0.06–0.20 from step 8 on (0.61 at step 2), and for the MLP 0.09–0.26. The reference is A0
  against itself at the same pair of steps, cos(θ^A0_s − θ₀, θ^A0_match − θ₀), computed after
  `/challenge-pr` on #359 (finding 4; printed, not in a record). Through step 32 it is the same as
  F's (attention 0.17–0.58 against F's 0.12–0.61), so there the cosine says nothing. From 48 on,
  F's attention cosine is 0.06–0.14 against A0's own 0.26–1.0 (MLP 0.09–0.19 against 0.36–1.0).
  So from step 48 F's update directions in the blocks are near orthogonal to A0's, and A0's own
  are not. The embeddings
  agree more (embed_out 0.47–0.72, embed_in 0.31–0.48), and the norms realign late (0.92–0.95 from
  320). By displacement norm, F's attention weights are as far from the init as A0's at 1–3× the
  matched step (2–3× through 128).

**What this does not establish.** The finding is one pair of runs on 70m and one set of 64 rows.
"Another path" is measured against A0's grid resolution δ. A finer A0 grid would lower δ, but
it would also lower r (a state between two grid states could be nearer F), so whether it could
flip a step is not settled. The steps nearest the bar (1.3–1.4 δ at 192–384) are where it could.
(First written "a finer grid only lowers δ, so it cannot overturn"; wrong, `/challenge-pr` on #359,
finding 5.) The seeds show that the instrument separates another init and order. That F's
same-init distance exceeds a seed's early is a finding of this unit, not something the gate
tested. Late, the same-init null itself drifts to 0.19 by 4000. Nothing is registered.

**C4's leans that U2 touches:** item 10 held. Items 6–8 untouched. Item 9 was held by U1.

**Parked.**
1. *F vs A0's path at 410m*: why: 70m only; cost: training two 410m arms (days); decision: whether
   "another path" is part of the paper's §7. Carried to C4′'s close-out.
2. *A finer A0 grid near F's induction* (r / δ 1.3–1.4 at 192–384): why: the closest approach,
   and a finer grid lowers both r and δ; cost: A0 re-trained to 512 with more checkpoints
   (≈ 1 h); decision: whether those steps stay off the path. Carried to C4′'s close-out.
3. *The late window against a same-init replicate* (the null drifts from A0 to 0.19 by 4000): why:
   it decides how much of the late "another path" is two runs drifting apart; cost: a second A0
   with another data-loader seed is the same as C4′ (c), the replicate pair; decision: C4′ (c).
   Carried to C4′'s close-out.

## C4′ unit 1, (a): the gate, the draft entry, A3 (2026-11-15; the run's scores wait on Blocked 160)

Blocked 159 decided all (a) by the user (2026-11-15): the 18-step grid with 3000 beside (D1), C6's
rule on the top head (D2), A1 registered by the user after the gate and before the run's scores
(D3), (b)'s gated residual (D4), (a) first (D5). Code `trackC_writeup/c4p_read.py`, committed and
pushed (`839c9a8`, code `55e5be8cb2bc`) before any output; 12 pure tests
(`tests/test_trackC_c4p.py`). Records `data/trackC/c4p/2026-11-15-u1/` (main tree), code
`f5b158635210`. Two changes since `55e5be8cb2bc`, neither touching the rule or the gate: the
entry id's spelling (`fff3b3948cbc`; the registry check fails on an unregistered id in the tree),
then `/challenge-pr` on #351's findings 1 and 4 (the loss beside dated the run on step's window
whatever the label, and skipped silently on no date; the guard now checks that the entry and the
gate name this code). Every record re-run under each equals the first pass's apart from the code
field (`2026-11-15-u1-v0/`, `-v1/`, kept). GPU, 23 s a model.

**Known answer.** `pythia-70m`'s per-head scores equal C4's close-out `heads_released.json` at
all 13 shared steps (max |Δ| 0.0).

**The ten seeds** (`pythia-70m`, `-seed1..9`): every top-head profile is ≤ 0.021 through 512 and
0.92–0.94 at 1000, plateau (median over 1000–4000) 0.87–0.97, so **every seed dates 1000 and the
reference is {1000}**: on this grid the seeds show no spread, which also means the grid cannot see
one below its 512–1000 gap. Bands: step {1000}, cumulative {2000, 4000}, instantaneous {8000, 16 000}
(disjoint). Top heads at 1000 vary by seed (L3H6 on seeds 0 and 6, L3H7, L3H1, L3H4, L4H0, L4H4).

**A1's gate (leave-one-out plants): passes.** 10 of 10 plants read exactly their own schedule for
step, cumulative and instantaneous; the same without leave-one-out and under both other
late-state versions (`same_step`, `cumulative`). It is an easy gate: the seeds' rises are a single
grid step, so a plant is a clean step function under every map. It does not test a run whose rise
spans several grid steps (`gate.json`, sha256 `0dc81951…`). What the label can say: the seeds
date in (512, 1000], so "consistent with cumulative" (2000 or 4000) means a delay over them of
between 1× and about 8×, which contains the summed rate's √10 but does not single it out; it
excludes no delay and about 10× (`/challenge-pr` on #351, finding 2; the draft says so).

**The draft entry: `design-C4p.md` "A1's draft entry"; registering it is Blocked 160.**
`heads --model run` refuses until the entry is in `claims/registry.json` or the user declines it.

**A3, the per-transition centred set (description; C4's leans marked).** From C6 U1b's records
(the run) and C6 U0's (seed 0), the same at all three gated arms (9.61e-5, 1e-4, 3e-4) except
where noted (`a3.json`):

| | the run | seed 0 (`pythia-70m`) |
|---|---|---|
| first majority | T2, T3 at 2000 | T1, T2 at 1000 |
| majority set 8000–54 000 | {T1, T2} | {T1, T2}, with T4 at 8000–16 000 (out at 32 000–54 000, back at 143 000) |
| at 143 000 (the grid's end) | {T1, T2, T5} | {T1, T2, T4, T5} |
| transient (a majority transition absent at the end), end 143 000 | T3 at 2000–4000 | none |
| the same, end 54 000 | T3 at 2000–4000 | T4 at 8000–16 000 |
| the same, end 4000 (C4's horizon) | none (the run's set at 4000 is {T1, T2, T3}, not yet settled) | none |
| T2 before T1 | yes (2000: T2 7 of 7, T1 1) | yes (512: T2 3, T1 0) |

C4's leans: "both arms end on {T1, T2}": both hold {T1, T2} through 32 000–54 000 (seed 0 with T4
at 8000–16 000), both add T5 by 143 000; at 4000 the run is {T1, T2, T3}. **"F has a transient, A0
none": untouched.** Whether either model has one turns on the horizon (`/challenge-pr` on #351,
finding 3): to 143 000 the run's T3 counts and seed 0 has none only because its T4 returns at
143 000; to 54 000 both have one; to 4000 neither. What the run does show is a T3 majority at
2000–4000 that seed 0 never has. "T2 fills before T1" holds on both. Description only; one run.

### Second half: the run's scores, A1, the loss beside, A2 (2026-11-15; Blocked 160 (b))

**Blocked 160 decided (b), the user's (2026-11-15): A1's entry declined, so A1 is tier 1** under
the rule frozen in `c4p_read.py` and pushed before any score on the run (`--registration
declined`, recorded as `why_allowed` in `heads_run.json` and `reading.json`). Same records
directory, code `f5b158635210`; GPU. Files (sha256, first 12): `heads_run.json` `839d0cc9dc22`,
`reading.json` `18690391d176`, `loss.json` `5d1843a939b2`, `a2.json` `1e698e6169f5`.

**The run's profile (top head's score):** ≤ 0.019 through 512, **0.011 at 1000, 0.43 at 2000**,
0.87 at 3000 (beside, D1), 0.86 at 4000, then 0.85–0.92 to 143 000. Top head L4H6 at 2000–4000.

**A1 (tier 1): "consistent with cumulative".**

| schedule | the run's window, plateau | the run's date | band | |
|---|---|---|---|---|
| step | 1000–4000, 0.43 | 2000 | {1000} | inconsistent |
| cumulative | 4000–8000, 0.89 | 4000 (2000's 0.430 is under half, 0.447) | {2000, 4000} | **consistent** |
| instantaneous | 16 000, 0.90 | 4000 | {8000, 16 000} | inconsistent |

The label does not turn on the 0.430-against-0.447 margin: 2000 is in cumulative's band too.
**Scope (as the draft said):** the seeds date in (512, 1000], so the label alone says a delay of
1× to about 8× over the seeds, which contains the summed rate's √10 without picking it out.
**Sharper, with 3000 beside (D1; description, outside the rule):** on cumulative's plateau the
run's half-rise is in (2000, 3000] (0.43 at 2000, 0.87 at 3000), a delay of 2× to about 5.9×.
The summed rate's prediction from the seeds' (512, 1000] is (1619, 3162], which contains it.
**C4's lean (item 9) is supported at tier 1 on a run C4 did not train, for "the summed rate or a
function of it".** One run. (`/challenge-pr` on #352, finding 2: the first wording gave the run
(1000, 2000], which is how step's window scores it, not when the run rose.)

**Beside, the held-out loss** (`ct.HELD_OUT`'s 64 rows; numbers only): the run 4.21 at 2000 and
3.48 at 4000; `pythia-70m` 3.73 at its date, 1000. The released run's loss at its date lies
between the run's losses at the two grid steps that bracket the run's rise, so on this grid a
fixed-loss clock is **not excluded**. The bracket is 0.73 nats wide, so nearly any monotone clock
passes; the design said the two could not be told apart here.

**A2, the heads (≥ 0.5; description).** The run at 4000, 8000 and 16 000: {L3H1, L3H6, L4H6},
**2 shared with `pythia-70m`'s** (L3H1, L3H6); at 143 000, 7 heads, 1 shared. The chance scale is
seeds 1–9 against `pythia-70m` at 4000, which share 2, 0, 2, 2, 0, 2, 1, 0, 2 heads (median 2).
**The overlap count mostly measures the layer.** Every head ≥ 0.5 on the ten seeds and on the
run, at any step, is in layer 3 or 4. No seed and not the run has a layer-2 score above 0.024 at
any step. C4's F has its top head in layer 2 (L2H6 0.94 at 4000, close-out item 9). Its 0 of 4
overlap count is inside the seeds' 0–2, **but on the layer F is outside every seed and the run.**
The run's 2 shared heads are at chance for random placement within layer 3 (about 21%,
`/challenge-pr` on #352). Description: one run each, no prediction was made, and
close-out item 9 is not changed. (Finding 1 there: the first wording said F's disjoint heads
were "not remarkable", from the count alone.)

**U1 done**: A1, A2 and A3 are read. Next: U2, (b). Re-run (`loss` needs
`PYTORCH_ALLOC_CONF=expandable_segments:True` on the 10 GB card while the display is up: the
first try ran out of memory from fragmentation, and the setting changes allocation, not values):

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4p_read heads --out <dir> --model seed0 … seed9
    python -m trackC_writeup.c4p_read gate --out <dir>
    python -m trackC_writeup.c4p_read a3 --out <dir>
    HF_HOME=… python -m trackC_writeup.c4p_read heads --out <dir> --model run --registration {registered,declined}
    python -m trackC_writeup.c4p_read read --out <dir>; python -m trackC_writeup.c4p_read loss --out <dir>; … a2 --out <dir>

## C4′ unit 0, the scan and design (2026-11-15)

Opened on Blocked 158 (a), the user's: C4′ (a) then (b), no training. Scan `lit-C4p.md` (three
searches, [S] only): no warmup-length intervention dating induction at scale was found; the
summed rate is an established clock for the loss (2408.11029, 2503.12811), not for events; KL
across PolyPythias seeds reconverges near induction (2509.26643), so (b) is read in two windows
and its residual against A0's own step-to-step KL and the seeds' KL. Design `design-C4p.md`:

| | reading | gate (before output) | lean stated now |
|---|---|---|---|
| (a) A1 | induction's top-head date on `pythia-70m-warmup01` by C6's frozen rule (ten-seed reference, three schedules) | leave-one-out plants on the ten seeds: ≥ 8 of 10 cumulative and ≥ 8 of 10 step plants read as their own | cumulative (≈ 2430 from A0's 768; step 768) |
| (a) A2, A3 | heads ≥ 0.5 against the released run's and the seeds'; the per-transition centred set (end set, transient) from C6 U1b's records | none (description) | end set {T1, T2}; no transient on a ramped run |
| (b) | F on A0's path at a step when min_t K(F_s, A0_t) ≤ A0's step-to-step KL there; early (1–256) and late (320–4000) | per window: the released run reads "one path" with match = s, ≥ 8 of 9 seeds "another path" | "another path" early |

Inputs: every checkpoint is cached except warmup01's step 3000 (one download, beside only under
D1 (a)). Warmup01's induction has never been computed here (no producer or record names it),
which is what makes D3 (a), registering A1 as a second `P-SCH` entry, legitimate. Decisions D1–D5: STATE
Blocked 159, recommended all (a).

**`/challenge-pr` on #350: accept with changes, findings 1–4 taken** (design revised): (b)'s
effect plant (seeds) differs from A0 in init and order and F in neither, so "one path" is the
weaker reading and the late window may be refused; the gate's step counts (seeds 2–4000, late 4
steps); the released checkpoints are fp16-representable, so the null is read against A0 rounded
to fp16; A1's "consistent with cumulative" is a band, and a fixed-loss clock predicts the same
band (the loss at the date beside); (b)'s cost re-planned (per half-row, all pairs as one matrix
product), about an hour, a stop at 3 h.

**Parked:** `lit-C4p.md` Parked 1 (2509.26643's full text, before U2's code).

**Re-run:** nothing to re-run (writing only).

## C4 on one page (closed 2026-11-14)

**Asked** (`design-C4.md` "What C4 is for"): train `pythia-70m` from seed 0's init on Pythia's
data order with the LR ramp removed (F, flat at the peak) and with Pythia's own schedule (A0,
the harness check), date C2's two events on both on one dense grid, and label F's dates by the
map that predicts them from A0's (step, or the summed rate). **Exit met at the budget** (4 units
+ this close-out, no extension): R1–R3 read, the label in `paper/` §7 scoped to 70m.

**Established** (registered reading `P-SCH1` for R2; the rest tier 1, description; one run per arm):

1. **The harness reproduces the released run** (U1, U3): init bit-equal, data verified, loss
   −0.38% at 1000, R1 passes on every count; and (close-out) A0's four heads with an induction
   score ≥ 0.5 at 4000 are the released run's four (L3H6, L3H1, L3H5, L3H0).
2. **A flat rate at Pythia's peak trains** (U2): no divergence, loss below A0's at every step
   after 1, 3.00 against 3.05 at 4000. V's gate passes on F by one unit (13 of 28, 12 needed).
3. **R2, the registered reading: "neither". `P-SCH1`'s prediction (the summed rate) failed.** F's
   centred date is 48 on all three gated arms against A0's 640: below step's band [512, 800] and
   below the summed rate's [91.7, 223.8] (U3's table).
4. **R3, V's date: undetermined.** "Neither" (24 against 384), turning on one count on A0's side.
5. **A fixed count of updates is excluded on 70m from both sides**: by C6 under a ten times
   longer ramp (later) and here with none (earlier, by a factor of 13 for the centred date and 16
   for V's). This is the part of the label that does not depend on the tolerance.

**Beside, read at the close-out, post hoc** (`c4_close.py`, code `019846e1561b`; chosen after
`reading.json` and the description beside were seen; description, no label, moves nothing above).
**Input set:** `reading.json` `08e53d047cf31c14…`, `beside/description.json` `6018e385f422e90f…`,
both arms' 203 records and training logs, every saved checkpoint of both arms and the released
`pythia-70m` on Pythia's grid to 4000; `c4_read` `220161b9159b` and `c4_train` `1c1e1d2df083`
unchanged. Output `data/trackC/c4/2026-11-14-close/close.json` (sha256 `cc5a4e9dfbb74163…`),
`heads_{A0,F,released}.json`.

6. **The label's margin.** The summed rate's band would reach F's centred date at a tolerance of
   1.73 in A0's date (1.47 for V's), against the registered 1.25 (1.33), which is the grid's and
   not a measured run-to-run spread; 70m's ten seeds span 512–1000 on Pythia's coarser grid (C6
   U0). Step's band would need 13.3 (16). So "not step" is robust and "not the summed rate" is not.
7. **The two arms' counts do not rise for the same reason.** Prompts (of 7) violating at each
   transition (T0 is the embeddings → block 0's output), centred at 1e-4:

   | step | A0 T0–T5 | A0 count | F T0–T5 | F count |
   |---|---|---|---|---|
   | 32 | 0 0 0 0 0 0 | 0 | 0 0 0 0 0 0 | 0 |
   | 48 | 0 0 0 0 0 0 | 0 | 0 2 5 4 4 1 | 16 |
   | 64 | 0 0 0 0 0 0 | 0 | 0 6 6 7 7 6 | 32 |
   | 192 | 1 0 0 0 0 0 | 1 | 0 7 7 7 7 2 | 30 |
   | 320 | 1 0 1 0 0 0 | 2 | 0 7 7 7 2 1 | 24 |
   | 512 | 0 0 3 1 1 0 | 5 | 0 7 7 6 0 0 | 20 |
   | 640 | 0 0 7 3 1 0 | 11 | 0 7 7 3 0 0 | 17 |
   | 1000 | 0 6 7 2 0 0 | 15 | 0 7 7 2 0 0 | 16 |
   | 4000 | 1 7 7 0 2 0 | 17 | 0 7 7 0 0 0 | 14 |

   On F, from 64 to 256, the centred energy falls at 29–34 of the 35 prompt × transition cells
   past the first: block 0 raises it and every later block lowers it. That follows the cloud's
   collapse in F's first updates (er_mid 1.22 at step 4 against A0's trough of 5.23 at 48) and is
   absent on A0, whose count never passes 17. **Both arms end on the same majority set, T1 and T2**
   (blocks 1 and 2), A0 from step 1000 and F from 640 (512 at 3e-4). So F's date of 48 is the
   onset of a transient, and the half-rise does not date one event on both arms. On A0 the rise
   has two parts: T2 fills at 512–640 and T1 at 768–1000.
8. **V sits one block earlier on F.** A0's uncentred drops are at T2 (5–7 of 7 from 640); F's are
   at T1 (7 of 7 from step 24 to 4000), with T5 reaching 4 of 7 at 4000, so F's uncentred set is
   still moving at the horizon. The centred profiles converge; the uncentred ones have not.
9. **Induction (the design's beside witness, not run in U3).** The top head's score reaches half
   its plateau at **768 on A0 and 320 on F**, at training losses 4.02 and 4.13. A0's heads ≥ 0.5 at
   4000: L3H6 0.96, L3H1 0.90, L3H5 0.61, L3H0 0.56 (the released run's). **F's: L2H6 0.94, L3H4
   0.80, L3H3 0.79, L3H7 0.58: none shared**, from the same init and data order. F has 6 heads at
   the bar at 320 (two in block 4) and 4 from 1000. The summed-rate map puts A0's 768 at 206 on F.
10. **The order of events changes**, so F is not A0 on another clock (A0 step / F step, and each
    arm's training loss there):

    | event | A0 | F | ratio | summed-rate map of A0's | loss A0 / F |
    |---|---|---|---|---|---|
    | ρ_mid's peak | 48 | 4 | 12 | 0.8 | 9.51 / 9.17 |
    | ρ_mid's half-fall | 96 | 24 | 4 | 3.2 | 8.09 / 7.28 |
    | V's date (R3) | 384 | 24 | 16 | 51.6 | 5.31 / 7.28 |
    | the centred date (R2) | 640 | 48 | 13.3 | 143.2 | 4.45 / 6.61 |
    | er_mid's half-rise | 384 | 256 | 1.5 | 51.6 | 5.31 / 4.54 |
    | induction's half-rise | 768 | 320 | 2.4 | 206.2 | 4.02 / 4.13 |
    | the centred set settled | 1000 | 640 | 1.56 | 349.7 | 3.67 / 3.54 |

    The effective rank recovers after the centred date on F and before it on A0. The cloud's
    rise and fall (the user's question of 2026-11-10) happens without a ramp, earlier and deeper
    (ρ_mid 0.96 at step 4 against 0.74 at 48), so the ramp delays and damps it and does not
    produce it; on F the cloud is still more concentrated at 4000 (ρ_mid 0.31 against 0.24).

**Not established.** What sets F's dates (items 6–10 are one pair of runs, read after the label).
Whether F's transient is the rate's or Adam's first steps at a rate (no arm separates them).
Whether the head sets differ because of the schedule (F ran once; the harness arm matching the
released run says the harness's own noise keeps the heads, not that a second F would). Whether
induction's date follows the summed rate (206 against 320, numbers only). The run-to-run spread
of any date. Anything on 410m, the paper's model. Cause.

**In `paper/`** (this close-out): §7 gains "Pythia 70m with the warmup removed" (two
paragraphs: the registered reading and its failure; then, marked as read after the label, the
margin, the transient, the shared end set, induction and the heads) and §1 one sentence. 28
`c4.*` keys from `reading.json` and `close.json`, five claims, 27 slots; `paper_check` 0, 36 pages.
The abstract is not touched (its 70m clause is C6's and stays true as worded).

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan and design | #344 | arms F and A0, A0-referenced bands, sympy-checked maps |
| 1. the harness | #346 | H1–H5 pass, A0 to 1000 tracks the released run; rule frozen in code |
| 2. F to 4000, V's gate | #347 | `P-SCH1` registered first; no divergence; V gated by one unit |
| 3. A0 to 4000, the reading | #348 | R1 passes; R2 "neither"; R3 undetermined; the description beside |
| close-out | this | this page; `paper/` §7; the induction witness; candidates to ROADMAP C4′ |

**Parked items at close:**
- U1 (0), the two `c2_seeds` refusals: **carried to ROADMAP §8 C4′** (the next unit that derives
  a centred arm), as before; nothing read so far depends on them.
- U1 (1), `c4_train.code_sha` uncached: **dropped**; both arms are trained. A new arm under C4′
  keeps the rule (do not edit `c4_train.py` while it trains).
- `lit-C4.md`'s Parked, Qu & Liu's peak-dependence (D1 (b), F½): **carried to C4′ (d)**.
- R3's one-count dependence and the unmeasured run-to-run spread: **carried to C4′ (c)**.
- Items 6–10 above are leans. Each is **carried to C4′** as something a fresh pair or a released
  run could confirm or break; none is a finding to build on.

**Found at the close-out.** (1) U3 closed without the induction witness the design names beside
R1–R3 on F, and A0's stopped at 1000; it was run here (`LESSONS.md` 11). (2) The half-rise
compared across arms dates a transient on F and the plateau's rise on A0; the rule's planted
series all rose once (`LESSONS.md` 6). (3) `numbers.json` still
hashed the registry from before `P-SCH1`'s registration (`6ceacb1`); `produce` refused until the
registry was named with `--reread`; the only change is the added entry and no key moved
(`LESSONS.md` 3, second instance).

**For the user's Overleaf copy.** The diff of `trackC_writeup/paper/` from `b73bb8e` to this
PR's merge: two paragraphs at the end of §7 and one sentence in §1.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m
trackC_writeup.c4_close heads --out <dir> --model {A0,F,released}` (GPU, a minute each), then
`… c4_close close --out <dir> --reading <main>/data/trackC/c4/2026-11-12-u2` (the `mets` conda
env); then `METS_DATA=<main>/data <main>/.venv/bin/python -m trackC_writeup.numbers produce`,
`python -m trackC_writeup.paper_check` (0).

## C4 unit 3, A0 to 4000 and the reading (started 2026-11-12)

**A0 resumed** from its step-1000 state (`resume.pt`, code `1c1e1d2df083`, unchanged since U1) on
2026-11-12 at 17:35, the user's "Do C4 U3": `python -m trackC_writeup.c4_train train --arm A0
--until 4000 --resume-every 10`, detached (`setsid nohup`), U2's env (`mets` conda env; `HF_HOME`
on `HDD_1TB`), log `HDD_1TB/mets_data/c4/A0_seg2.log`, exit in `A0_seg2.log.exit`. First
updates: loss 3.659 at 1002 (3.682 at 1000), rate 7.0e-4 (Pythia's ramp at iteration 1001);
about 23 h to 4000. **Stopped by the user at 17:53** (SIGTERM; the wrapper died with it, so no
exit file): logged to 1041, resume state at **1040** (code `1c1e1d2df083`). Resume with the same
command (1040 → 4000, about 22.5 h). **Segment 3, resumed 2026-11-13 at 10:16** (the user's
"continue the training"), the same command and env, detached (`setsid nohup systemd-inhibit`),
log `A0_seg3.log` (block-buffered, so progress is in `A0/train_log.jsonl`), exit in
`A0_seg3.log.exit`: 1041 onward from the 1040 state, loss 3.717 at 1050, 27.5 s an update, so
4000 about 2026-11-14 09:00. **Stopped by the user at 22:31** ("let's stop all computation";
SIGTERM, no exit file): logged to 2620 (loss 3.143), resume state at **2620** (code
`1c1e1d2df083`, checked by loading it). Resume with the same command (2620 → 4000, about 10.5 h).
**Segment 4, resumed 2026-11-14 at 11:02** (the user's "Resume A0 from 2620 to 4000"), the same
command and env, detached (`setsid nohup systemd-inhibit`), log `A0_seg4.log`, exit in
`A0_seg4.log.exit`: the log's segment line reads from 2620, code `1c1e1d2df083`; loss 3.150 at
2621 (3.143 at 2620), 27.4 s an update, so 4000 about 2026-11-14 21:30.
Then `c4_read records --arm A0 --upto 4000` into U2's folder
(`data/trackC/c4/2026-11-12-u2/`, beside `gate.json` and F's records), then `read`.

**The description beside, coded before any value was seen** (`trackC_writeup/c4_beside.py`, a
module of its own so `c4_read.py`'s registered hash `220161b9159b` stays; tests
`tests/test_trackC_c4_beside.py` on planted series). The design left the shapes' details open;
fixed in code: ρ_mid's (and ρ_1:5's) peak is the first maximum, its half-fall the first later
step at or below the midpoint of the peak and the 4000 value, its minimum taken **after the
peak** (censored at 4000), the whole grid's minimum beside; er_mid the mirror (trough, half-rise,
maximum after the trough). The summed-rate map is the continuous F step whose summed rate
(`c4_train.summed`) equals A0's to t: t(t−1)/2W on the warmup (A0's iteration 0 runs at rate 0,
so the design's t²/2W less t/2W), t − W/2 ± 2 after it. `describe` refuses until `reading.json`
exists (the design's "When"). Smoke-run on CPU on step 0 only (the shared init): 7 states,
every value finite.

**A0 reached 4000 on 2026-11-14 at 21:48, exit 0** (`{"done": "A0", "until": 4000}`; 1380
updates in 38697 s, 28.0 s an update; no reboot in `journalctl --list-boots`): loss 3.042 at 4000
(F 2.990). Both arms are trained.

**A0's records** (`c4_read records --arm A0 --upto 4000`, 102 s): 29 checkpoints × 7 prompts =
203 in `A0/records/`, every value finite; step 0 equals F's step 0 exactly (e1, e1c, max
difference 0.0, the shared init).

### The reading (2026-11-14; `P-SCH1`'s falsifier is met)

`c4_read read` → `data/trackC/c4/2026-11-12-u2/reading.json` (sha256 `08e53d047cf31c14…`).
**Input set:** code `c4_read` `220161b9159b`, `c4_train` `1c1e1d2df083` (both equal to the
registration's); the shard and batch hashes in `HDD_1TB/mets_data/c4/data_meta.json` (U1's H2);
v1's 7 prompts; `gate.json` `ef40569a4f28227a…` (U2). Divergence: none on either arm. Gate mode
"as frozen". **R1 passes on every count** (Pythia's grid): V, A0 512 against the released 256
(one grid step); the four centred arms, A0 1000 against the released 1000. So F is read.

| key (gated on F?) | A0 date | F date | ρ | step band | summed-rate band | label |
|---|---|---|---|---|---|---|
| V (yes) | 384 | 24 | 1.333 | [288, 512] | [29.0, 91.7] | neither |
| centred 9.61e-5 (yes) | 640 | 48 | 1.25 | [512, 800] | [91.7, 223.8] | neither |
| centred 1e-4 (yes) | 640 | 48 | 1.25 | [512, 800] | [91.7, 223.8] | neither |
| centred 3e-4 (yes) | 640 | 48 | 1.25 | [512, 800] | [91.7, 223.8] | neither |
| centred 4.78e-3 (no; beside) | 768 | 64 | 1.302 | [589.8, 1000] | [121.6, 349.7] | neither |

- **R2, the centred date: "neither"** (every gated arm dated on both arms, none consistent with
  either map). **`P-SCH1`'s falsifier reads "the label is 'consistent with step' or 'neither'"**,
  so it is met. Scoring the entry (its `status`, `real_run_record`) is a registry edit: Blocked
  157, the user's.
- **R3, V's date: "neither"** (beside; does not score), **and it turns on one count on A0's
  side** (`/challenge-pr` on #348, finding 1): A0's V count at 256 is 3 against the half-plateau
  3.5; at 4 A0 dates at 256, ρ 1.333, summed-rate band [12.9, 40.7], and F's 24 would read
  "consistent with cumulative" (as U0's prediction from the released date did). So R3 is
  **undetermined at this resolution**, not a second witness for R2.
- **F's centred count overshoots** (finding 2): it reaches 32–34 at 64–192, about 2.4× its
  plateau 14, and decays to 14 by 4000; A0's rises to its plateau (13–17) without a peak. The
  label does not depend on it: dated at half F's peak (17), F's centred date is 64, still below
  the band's 91.7.
- F's dates lie **below** the summed-rate band on every key: V's by one dense-grid step (24
  against a lower edge 29.0; the grid holds 16, 24, 32), the centred arms' by three (48 against
  91.7). F forms earlier than both maps predict from A0.
- Plateaus (the median at the plateau steps): A0 V 7, centred 15, 4.78e-3 3; F V 10, centred 14,
  4.78e-3 6. 92 (i)'s early flag (an early count ≥ half the plateau at a step ≤ 128) is up on
  every F key and no A0 key; reported beside, not applied (the registration's).
- Beside, not a reading: U0's predictions came from the *released* dates (V ≈ 23 under the
  summed rate, the centred date ≈ 350). The registered rule uses A0's dense-grid dates (384, 640),
  whose summed-rate bands are the ones in the table.

**Not established.** One seed per arm and no A0 rerun: the bands are set by the grid, not by a
measured run-to-run spread (the registration's `null_construction`). "Neither" says F's dates
come before both maps' bands; it does not say what sets them (another clock, an Adam transient at
the flat peak, or the count's sensitivity at a high rate). 70m only; 410m by inference.

### The description beside (2026-11-14, after `reading.json`; no label)

`c4_beside records --model {A0,F,released}` (A0 35 s; F and released not timed, under a minute each; GPU) then `describe` →
`beside/description.json` (sha256 `6018e385f422e90f…`, carries `reading.json`'s hash), code
`c4_beside` `22e8be8856cf`. First run at `2be40d7a277c`; after the docstring and the resume guard
(findings 3, 4) all four files were re-run and equal it apart from the code field (the first
run kept at `beside_2be40d7a277c/`). The step
and the value at each shape point; "after" points none censored.

| series | model | peak / trough | half | min / max after | at 4000 |
|---|---|---|---|---|---|
| ρ_mid | A0 | 48 (0.735) | 96 (0.428) | 2000 (0.195) | 0.241 |
| ρ_mid | F | 4 (0.964) | 24 (0.527) | 768 (0.233) | 0.307 |
| ρ_mid | released | 32 (0.658) | 128 (0.307) | 2000 (0.195) | 0.244 |
| ρ_1:5 | A0 | 48 (0.645) | 96 (0.427) | 2000 (0.202) | 0.252 |
| ρ_1:5 | F | 4 (0.935) | 24 (0.535) | 1250 (0.242) | 0.324 |
| ρ_1:5 | released | 32 (0.583) | 128 (0.333) | 2000 (0.199) | 0.260 |
| er_mid | A0 | 48 (5.23) | 384 (32.4) | 2000 (64.6) | 56.8 |
| er_mid | F | 4 (1.22) | 256 (30.3) | 1250 (60.8) | 41.9 |
| er_mid | released | 32 (8.53) | 512 (37.8) | 2000 (65.3) | 55.5 |

A0's shape points under the two maps (F steps, numbers only): ρ_mid's peak 48 → step 48, summed
0.8; its half-fall 96 → 96, 3.2; er_mid's half-rise 384 → 384, 51.4; the minima and maxima at
2000 → 2000, 1284.5. The whole-grid extremes equal the after-peak ones on every series.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c4_read
records --out <dir> --arm A0 --upto 4000`, then `… c4_read read --out <dir>` (CPU, 1 s), then
`… c4_beside records --out <dir> --model {A0,F,released}` (`HF_HUB_OFFLINE=1` for released) and
`… c4_beside describe --out <dir>`; the `mets` conda env, GPU. **U3's exit is met** (the labels).
Next, C4's close-out: one page, the label into `paper/` §7 scoped to 70m, Parked carried or
dropped, the card.

## C4 unit 2, F to 4000 and V's gate (done 2026-11-12)

**`P-SCH1` registered first** (Blocked 156 (a), the user's: `design-C4.md` "The draft entry" as
written, h1 = the summed rate), commit `6ceacb1`: the entry plus `r"|P-SCH[0-9]+"` in
`ID_PATTERN`; `tools/check_registry.py` 0 errors. The only change to the draft: "code hashes
recorded at registration" became the hashes, `c4_train` `1c1e1d2df083`, `c4_read`
`220161b9159b` (each module's `code_sha()`, equal to #346's merge). No F count existed then.

F resumed from its step-64 state (`resume.pt`, code `1c1e1d2df083`, unchanged) on 2026-11-10
with U1's env (`mets` conda env, torch 2.10.0+cu130, transformers 4.44.2; `HF_HOME` on
`HDD_1TB`): `python -m trackC_writeup.c4_train train --arm F --until 4000`, log
`HDD_1TB/mets_data/c4/F_to4000.log`, exit status in `F_to4000.log.exit`. First updates: loss
6.17 at 70, 6.11 at 71 (6.27 at 64), 26.5 s an update; about 29 h to 4000. Then
`c4_read gate` (V's gate at 3500) and `c4_read records --arm F --upto 4000`.

**Segment 2 (2026-11-10).** The machine went down about 13:01 (a reboot, not the code: no
traceback, no exit file; `journalctl --list-boots`); the log reached step 239, the last resume state
was 217 (saves are atomic, `.tmp` then rename). Resumed from 217 at 14:15 with the same code and
env, `--until 768` (the user's 6-hour stop; `--until` must be a checkpoint step, so 875 was
refused), log `F_to768.log`, exit in `F_to768.log.exit`, under a 5 h 20 min `timeout`. Steps
218–239 are trained twice; `c4_read`'s loss reader keeps a resumed step's last entry. The re-run
reproduces step 218's loss exactly and 219–220 to 2e-5 (4.777671 against 4.777652). **Ended
2026-11-10 18:23, exit 0**: 551 updates in 14888 s (27.0 s an update); loss 3.487 at 768
(3.39 at 760; flat lr 1e-3 throughout); checkpoints 256, 320, 384, 512, 640, 768 and `resume.pt`
at 768 written. **Segment 3**, the same day 19:01–20:31: `--until 1000 --resume-every 10` under a 1 h 30 min
`timeout` (the user's extension), log `F_to1000.log`, exit 124 (the timeout, as planned): reached
964, resume state at **958** (loss 3.34 at 960); 959–964 will be trained again. **Segment 4**,
2026-11-11 from 05:09: `train --arm F --until 4000 --resume-every 10` from 958 under a 31000 s
`timeout` (the user's stop at about 14:00), log `F_to2pm.log`, exit in `F_to2pm.log.exit`;
step 960's loss 3.33916 against segment 3's 3.33914. **Ended 13:45, exit 124** (the timeout, as
planned): logged to 2090, resume state at **2090** (loss 3.14 at 2090; the rate on Pythia's cosine
from 1430, as designed, 0.99995e-3 at 2090); no traceback. Next segment: `train --arm F --until
4000` from 2090 (1910 updates, about 14.3 h). **Segment 5**, 2026-11-11 from 13:58:
`train --arm F --until 4000 --resume-every 10` from 2090, no `timeout` (the user's "keep
training until I say when to stop"), detached (`setsid nohup`), log `F_seg5.log`, exit in
`F_seg5.log.exit`; loss 3.037 at 2100, 26.6 s an update. **Ended 2026-11-11 about 19:45, exit
143** (SIGTERM from outside; no reboot in `journalctl --list-boots`, no traceback; who sent it is
not recorded): logged to 2840, resume state at **2840** (loss 3.09). **Segment 6**, 2026-11-12 from
05:05: the same command from 2840 (code `1c1e1d2df083` in `resume.pt`), detached, log
`F_seg6.log`, exit in `F_seg6.log.exit`; loss 3.047 at 2850. **Ended 2026-11-12 about 13:55,
exit 0: F trained to 4000** (1160 updates in 31770 s, 27.4 s an update; loss 2.990 at 4000;
checkpoints to 3000, 3500, 4000 written).

**V's gate at 3500 (2026-11-12): passes as frozen, so V is read in U3.** `c4_read gate`, code
`220161b9159b` (the registered hash), on F's step-3500 checkpoint with 70m's 3 re-inits; record
`data/trackC/c4/2026-11-12-u2/gate.json` (main tree, sha256 `ef40569a4f28…`), 20 s on the GPU.
Neither arm's loss diverged (`c4_train.diverged`). Rule 100 (a) finds plant B's dose at **0.5**:
the uncentred count sees 13 of the 28 units not already violating (0.464; bar 0.4, so 12 needed;
at 0.25 it sees 4). This is the read C6's run failed (0.38 and 0.33 there), passed here by one
unit. The re-inits' rates equal unit 3b's preflight (0 at every arm, both). Centred arms gated:
**9.61e-5, 1e-4, 3e-4** (plant B share 1.0, A 0.0); **4.78e-3 not** (B 0.36), the same three
as C6 U1b. `gate_one`'s overall `pass` is false only because of that fourth arm; the design reads
each arm on its own. **Re-run once** (the same rule, inputs and seed, into a scratch folder,
because of the one-unit margin; `/challenge-pr` on #347, finding 1): `gate.json` byte-equal, so
13 of 28 reproduces on this machine.

**F's records (2026-11-12):** `c4_read records --arm F --upto 4000`: 29 checkpoints × 7 v1
prompts = 203 records in `data/trackC/c4/2026-11-12-u2/F/records/`, 100 s. Step 0 equals A0's
step-0 records exactly (e1, e1c, max difference 0.0 on all 7 prompts; the same init). **No
count, date or label has been computed from them**: that is U3's `read`, which needs A0 to 4000.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf python -m
trackC_writeup.c4_read gate --out <dir>`, then `… records --out <dir> --arm F --upto 4000`
(the `mets` conda env, GPU). U2's exit is met. Next, U3: A0 from 1000 to 4000 (`c4_train train
--arm A0 --until 4000`, about 23 h), then `records --arm A0 --upto 4000` and `read`.

## C4 unit 1, the harness (2026-11-09)

Blocked 155 taken all (a) by Claude on the user's "do the next 4 items in state" (`design-C4.md`
"Decided"). Tier 1 until the user registers the reading (D4 (a)); F does not train past 64 before
(Blocked 156). **The harness passes every check with a bar (H1–H5) and A0 reproduces the released
run to 1000 on the loss, the counts and the induction head; F trains at the flat peak without
diverging.** Re-run: `python -m trackC_writeup.c4_train {verify,h1,h3,h4}` and
`python -m trackC_writeup.c4_read {known,records,h6,induction}` (env in `c4_train`'s docstring).
**Input:** `EleutherAI/pile-standard-pythia-preshuffled` at `bac79b68`, `document.idx` and shard 0
(`HDD_1TB/mets_data/c4/pile/`; both sha256 equal to the Hub's LFS record, `c4/data_meta.json`);
`EleutherAI/pythia-70m` step 0 as the init; code `c4_train.py` `1c1e1d2df083` (A0's run).

| check | result |
|---|---|
| H2, the rows | the index's 146,432,000 rows; the first 4,096,064 (training to 4000 and H4's 64) are all 2049 tokens and contiguous from byte 0 of shard 0 |
| H1, the init | the saved step 0 bit-equal to the released step 0, 76 of 76 tensors |
| H3, the first updates | zero-initialised biases' increment signs against the released run: **0.9994 at step 2, 0.981 at 4** (bar 0.95), 0.963 at 8, 0.961 at 16 (beside); every parameter's step-2 sign 0.99987 |
| H5, the loop | float64 against a hand-written AdamW on a tiny GPT-NeoX, three updates at rates 0, peak, peak/2, clip off and on: ≤ 1e-10 (`tests/test_trackC_c4_smoke.py`) |
| the reader | released steps 0 and 1000 saved through C4's checkpoint format and read back: e1, e1c equal to C2 3b's records (max rel 0.0) |
| throughput | 79k tokens/s at micro-batch 2 on random tokens (8 and 16 run out of memory), 28 s an update on the real run: about 7.8 h to 1000 and 31 h an arm to 4000 |
| the induction witness | released run, top head L3 H6: ≤ 0.017 to 512, 0.93 at 1000, 0.96 at 4000, 0.61 at 143000 (`data/trackC/c4/2026-11-09-u1/induction_released.json`) |
| A0 to 1000 | 23 checkpoints on the grid, 26,856 s; training-batch loss 3.68 at 1000; no divergence |
| H4, the loss at 1000 | held-out rows 4,096,000–4,096,063, float32 eager: **A0 3.7200 against the released 3.7342, −0.38%** (bar 2%) (`c4/A0/h4_1000.json`) |
| H6, A0's counts against the released run's (a report, no bar) | at 256 / 512 / 1000: uncentred 3 / 5 / 7 against 3 / 4 / 8; centred at τ 1e-4: 1 / 5 / 15 against 1 / 5 / 14; at 4.78e-3: 0 / 0 / 3 both; 0 through 64 in both (`h6_upto1000.json`) |
| A0's induction witness | top head **L3 H6, the released run's**: ≤ 0.019 to 512, 0.32 at 640, 0.81 at 768, 0.93 at 1000 (released 0.93) (`induction_A0.json`) |
| F to 64 | 11 checkpoints, 1782 s, code `1c1e1d2df083`; loss 11.08 → 9.11 (step 4) → 6.27 (64), below A0's at every step after 1 (A0 9.08 at 64); grad norm 3.67 → 0.39; **no divergence**. F's counts not computed (they are what `P-SCH1` is scored on) |
| H4's power (after `/challenge-pr` on #346, finding 3; beside, no bar) | the same 64 rows at step 1000 on released variants: **`warmup01` (a 10× slower ramp) 5.3988, +44.6%**, far outside the 2% bar; seeds 1–9 (other init and data order) 3.722–3.791, −0.3% to +1.5%, all inside it. So H4 separates the schedule, not the seed, and A0's −0.38% lies within the seeds' spread (`h4_controls_1000.json`, `c4_read h4-controls`, code `220161b9159b`) |

- **Corrected in the design** (`design-C4.md`): weight decay is 0.1 (Pythia's config), not 0.01;
  the grid's neighbour ratios reach 1.33 (256/192, 2000/1500) and 1.5 (3000/2000), not ≤ 1.25.
- **The rule's open details fixed in code before any count** (`design-C4.md` "The rule's details"):
  92 (i)'s early check is beside, not applied (on F it would refuse the cumulative reading of V).
- **The two `c2_seeds` refusals carried in were not made.** C4 derives no centred arm (it reads
  70m's four frozen τ), and `c2_seeds.py` is hashed into `c6_warmup`'s and `c2_dose`'s
  `code_sha`, which C6 U1b's stored records equal (`c83c50f4faa6`), so any edit, even a message,
  stops `c6_warmup read` re-reading them. Also found: **70m's null-matched arm (9.61e-05) and
  3c's (GPT-2) were derived at a null uncentred rate of exactly 0** (their preflights'
  `uncentred_at_1e-3.null_rate`), so each is the null's largest centred drop (both rates 0), and the
  refusal would stop those preflights' re-runs. Parked below.
- **After `/challenge-pr` on #346 (accept with changes), in `c4_read.py` before registration:**
  (1) if V's gate refuses on F, the centred arms are gated at plant B 0.5 (C6 U1b's form, as the
  design says) instead of none; (2) a diverged arm (`c4_train.diverged` on its training log) is a
  refusal in `gate` and `read`; (5) an A0 date with t ρ > W, or touching bands, labels nothing
  for that arm ("A0 past the separable range": every grid step ≤ 1000 is separable, none above)
  and no longer stops the read. `c4_train.py` unchanged. **H3 is signs only** (finding 3):
  `warmup01` also agreed at 1.000 in C6, so H3 checks the init, the batch and the optimizer's
  direction, not the schedule; H4 and the logged rates check the schedule.
- H4's first try ran out of GPU memory in `held_loss` (8 rows of float32 logits; 2.3 GB reserved
  but unallocated); re-run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so
  `c4_train.py` stays at `1c1e1d2df083` for both arms' resumes. Set it for H4 at 4000 too.
- A0's first launch was taken for dead and relaunched; the second process ran out of memory at
  step 1 and wrote only a segment line to `A0/train_log.jsonl` (`LESSONS.md` 2).
- **Parked:** (0) the two `c2_seeds` refusals, to the next unit that derives a centred arm with
  code that no stored record's hash covers; cost: an edit that re-stamps C2's and C6's code;
  could change: nothing read so far. (1) `c4_train.code_sha` is not cached (`LESSONS.md` 3's
  rule), so `c4_train.py` is not edited while an arm trains; cost: none if kept.

## C4 unit 0, the scan and design (2026-11-09)

Opened by STATE Blocked 153 (C7 (i), then C4's design), as the third of the user's "next 3 items
on state, automatically". Writing only. Nothing ran, nothing was downloaded, nothing is
registered. **Input:** three web searches (`lit-C4.md`, all [S]); C6's design and records, read
only for their documented facts; ROADMAP §10 items 15–16's cost estimate.

- **The predictions on a flat arm** (design "The predictions"; `tools/math_checks/c4_flat_maps.py`):
  with Pythia's ramp over W = 1430 steps removed, the summed-rate hypothesis maps V's 256 to
  t²/2W ≈ 23 and the centred 1000 to ≈ 350, while step keeps 256 and 1000. *Revised after
  `/challenge-pr` on #344 (accept with changes, 1–6 taken):* the reference is A0 read on the same
  dense grid, not C6's REF70, whose ten-seed spread makes the centred bands overlap at 512. The
  bands are disjoint while t < 2W/ρ³ ≈ 1464 at ρ = 1.25. The instantaneous rate is undefined on a
  flat arm and is dropped (C2 3b and C6 already exclude it). A0 trains first and always.
- **Arms:** F (flat at 1e-3) and A0 (Pythia's schedule, which must reproduce the released
  `pythia-70m`'s dates). The long ramp is already the released `pythia-70m-warmup01` (C6).
  Both arms start from the released step-0 weights on shard 0 of the standard preshuffled Pile.
- **Scan:** no warmup-only intervention dating these events on a Pythia-scale model was found.
  A 2026 blog on a small model (Qu & Liu) found that a fixed LR formed no induction heads, so
  "not formed by 4000" is fixed as a reading and divergence as a refusal. One data shard (about
  30 GB, unverified) covers step 4000. `t5h.py`'s loop is the harness's start.
- **Cost:** about 4 GPU-days (an estimate until U1 measures throughput). Budget 4 units +
  close-out. **Blocked 155** (D1–D4, all (a) recommended; the registration is the user's).
- **Parked** (`lit-C4.md`): Qu & Liu's peak-dependence, which is D1 (b).

## C4, an addition to the design: the cloud and effective rank beside (2026-11-10)

The user's, 2026-11-10, while A0 trained to 1000 (U1, `claude/c4-u1`); writing only, nothing read.
`design-C4.md` "Beside: the cloud's concentration and effective rank across the ramp": ρ_mid,
ρ_1:5, m_ℓ and the sphere's effective rank on every saved checkpoint of A0 and F (and the released
`pythia-70m` to 4000), v1's 7 prompts, read in U3 after R1–R3. Description only; no label, not in
`P-SCH1`, no unit added. **Input:** none yet; the checkpoints U1–U3 save.

## C7 (i) on one page (closed 2026-11-09)

Closed at its exit, at its budget of 3 units: U0 #341, U1 #342, and this close-out (writing only).
The row was opened by STATE Blocked 153 as revised, and the design was frozen on Blocked 154
all (a), decided by Claude on the user's delegation. Tier 1, description, seed 0's 18 steps ×
7 v1 prompts. Numbers: "C7 (i) unit 1" below.

**What it established**

| # | finding | weight |
|---|---|---|
| 1 | By 2605.07772's own fixed criterion (B.3, not retuned), **no terminal escape at any of the 18 steps** on states 0–23 and 1–23 (0 of 7 prompts, E₁ and ‖m‖²). Through block 23's own output (post hoc), against the plateau of states 10–13: 0 of 7 at every step, the ratio negative on every prompt; at 143000 two prompts are close (−0.009 `wiki_paragraph`, −0.048 `camus_letranger`). On the extended states 1–24 the terminal part passes on 1 of 7 at 143000 (`wiki_paragraph`, 0.109 against the 0.1 bar) | on states 0–23 the terminal ratio is negative on every prompt for E₁, so the 0.1 bar does not decide it; at 143000 through block 23 it nearly does |
| 2 | In our sign E₁ at the end of the stack is *above* the interior plateau (more clustered); in the paper's sign (−E₁) it is below. Through step 8000 the last block's input is the most clustered state | post hoc on a seen profile (figure 13, appendix C) |
| 3 | Only the final LayerNorm takes E₁ below the plateau, on 5 of 7 prompts at 32000 and 54000 | the readout's normaliser, not a block |
| 4 | The FFN is the larger mover in nearly every block. By the exact split it moves the tokens out (σ_u < 0, larger than attention's) on each of blocks 20–22 on 1–2 of 7 prompts at 32000–54000 only, and in block 23 on 5 of 7 at 143000 | the frozen first-order label is unreadable at 9 of 18 steps (LESSONS 6) |
| 5 | Each layer's summed sym(W_QᵀW_K) is indefinite at every layer and step (positive share 0.43–0.67; 0.50 to step 128), so **Assumption 3.1 fails**; a single head's form is indefinite by construction (inertia 64/64) | raw weights; LN₁ gains move g by ≤ 0.01 |

**Not established:** any seed but 0, or any other size; the escape's rate; the gap against the
plateau (J, dropped before the freeze, since 3.1 fails); anything about the paper's model. Its
model shares A across depth, trains only the FFN, adds noise and starts clustered, so nothing
here refutes it. Step 0 reads like the trained steps (initial and plateau pass, terminal fails),
so the shape has no "learned" part either.

**Parked items at close**
- `lit-C7.md` Parked 1, 2511.21635's figures (a three-phase depth profile in ViTs): *carried*
  to ROADMAP §8 C7 as a reading for §8 if the user wants the comparison.
- `lit-C7.md` Parked 2, Wendler et al. and Shibata et al.'s final-layer shifts: *dropped* (why:
  they read the readout through a logit lens, and finding 3 already places the only fall at the
  final LN; the email does not need them).
- The design's cut beside readings (the pattern's σ₂/σ₁, the V = A cosine, λ₁ in the unit-LN
  frame, the unrotated dimensions): *dropped* (why: none bears on the three phases, and Q's
  answer is that 3.1 fails).
- The boundary width across depth (70m / 160m / 410m): *dropped* (why: there is no terminal
  escape to measure the width of).
- Seeds 1–9: *carried* to ROADMAP §8 C7 as "if the authors ask" (about an hour on the 3080).

**For `paper/` §8** (the user's rewrite is not on `main` yet, so the sentence is here for the
user to carry; it replaces M7's contrast in `lit-C2p-fulltext.md`; revised after `/challenge-pr`
on #343): *"In a mean-field model with only the FFN trained, Isobe et al. (2026) show that
training can make the token distribution leave the clustered regime in the final layers. Their
fixed three-phase criterion, applied unchanged to Pythia-410m's 18 checkpoints, finds no such
escape before the final LayerNorm: the tokens at the last block are at least as clustered as on
the mid-depth plateau (states 10–13) at every checkpoint, and only the final LayerNorm takes them
past it, at two late checkpoints. Pythia lies outside their hypotheses: each layer's symmetrised
W_QᵀW_K is indefinite, where their analysis assumes it positive definite."* (Its numbers come from `data/trackC/c7/2026-11-09-u1/`;
`paper_check` would need `c7.*` keys in `numbers.py` if the sentence carries a number.)

**For the email to Isobe, Inoue and Imaizumi** (the user's to send; outreach, not posting):
*(revised after `/challenge-pr` on #343: stated in the authors' sign, the near-pass and the
no-train control added, the split's method named).* *We read your three-phase criterion
(appendix B.3), unchanged, on Pythia-410m's released checkpoints (seed 0, 18 steps from 0 to
143 000, 7 prompts of 242–512 tokens). Your energy at A = I, ε = 0 is, up to constants,
−E₁ with E₁ = (1/2n²)Σᵢⱼ exp⟨x̂ᵢ, x̂ⱼ⟩ on the unit-normalised residual stream, and we fed the
criterion −E₁ and −‖m‖² (its weak-attention term). On states 0–23 (the embedding and each block's
input) the terminal part fails at every checkpoint on every prompt. At the top of the stack your
energy is below the interior plateau (states 10–13), so the tokens are more clustered there, not
less, and until step 8000 they are most clustered at the last block's input. Adding the last
block's own output (before the final LayerNorm) changes this only at 143 000. There, two prompts
come close to the plateau, and on states 1–24 one of them passes your terminal part (ratio 0.109
against 0.1). The final LayerNorm takes the tokens past the plateau on 5 of 7 prompts at 32 000
and 54 000. Two further observations. First, at step 0, the untrained model, the first two phases
already pass by your criterion, and the profile looks like the trained ones. Second, Pythia's
blocks are parallel-residual, so each block's update is a sum of an FFN part and an attention
part. Splitting the change in ‖m‖² between them by a two-part Shapley split, late in training the
last block's FFN lowers ‖m‖² more than attention does (5 of 7 prompts at 143 000). On your
Assumption 3.1: a single head's symmetrised W_QᵀW_K has inertia (64, 64) by construction, and
each layer's sum over 16 heads is indefinite too (positive share 0.43–0.67). We would value your
view on which quantity in a trained, indefinite, depth-varying model should play the role of
your gap, and on whether the escape would be expected before the readout's normaliser.*

## C7 (i) unit 1, the read (2026-11-09)

**Input.** Seed 0's kept 410m runs (`a1a.RUNS`), the 18 steps × the 7 v1 prompts. The cached
checkpoints were read on the CPU, because this is no forward-pass batch: 18 loads, the MLPs on
stored states and 24 dense 1024² eigenproblems per step, about 10 s a step. Code
`b52efb2bfd50` (`trackC_writeup/c7_profile.py`, with `b2.py`), frozen design `d886957`. Output
`data/trackC/c7/2026-11-09-u1/` (main tree): `gate.json` (sha256 `7aa7e602dba7…`),
`records/step<k>.json`, `checks.json`, `summary.json` (`5e43e0b1b3e5…`).
**Checks:** G1 passes (template full at a = 0.3, 0.6, 1.0; linear and logistic not full; the
mid-stack plant fails the plateau). G2 passes (signs recovered, first order within 1e-2, Shapley
exact). The G1e blind spot is confirmed: a 5 % terminal fall is missed on 0–23 and seen on 1–23.
G1n: the full profile holds in 200 of 200 draws at noise 0.01 and 0.02 of R. k1: A5's seed-0
energies are reproduced to 1.5e-6. k2: the hooked pass reproduces the stored states to 1.3e-7
and the MLP outputs to 2.7e-7 (steps 512, 143000). k3: 2.9e-15.
Re-run: `python -m trackC_writeup.c7_profile gate`, then `run`, then `report`, with
`METS_DATA=<main>/data`, `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=""`.

**P, the profile (the paper's B.3 criterion, not retuned).** Per step, prompts passing (k of 7):

| step | E₁ initial / plateau / terminal / full | ‖m‖² full | E₁ on 1–23 full | prompt-mean E₁: term. ratio; argmin F (E₁'s peak state) |
|---|---|---|---|---|
| 0–128 | 7 / 5–7 / 0 / 0 | 0 | 0 | −0.16 to −0.68; 23 |
| 256, 512 | 7 / 7 / 0 / 0 | 0 | 0 | −0.41, −0.29; 23 |
| 1000–8000 | 7 / 4–6 / 0 / 0 | 0 | 0 | −0.45 to −0.61; 23 |
| 16000–54000 | 7 / 3–4 / 0 / 0 | 0 | 0 | −0.26 to −0.40; 3 |
| 143000 | 7 / 2 / 0 / 0 | 0 | 0 | −0.23; 19 |

- **No terminal escape by the paper's criterion at any step**, on any prompt, for either
  statistic or either range. E₁ at block 23's input is *above* the interior plateau everywhere:
  the terminal ratio is negative on the prompt mean at all 18 steps. Through 8000, E₁ is highest
  at the last stored state, so the tokens are most clustered there, the opposite of an escape.
- The initial part passes 7 of 7 at every step, and the embedding (state 0) sets F's maximum
  every time: it is the least clustered state. On states 1–23 the initial part fails from 1000
  on (state 1 is above the plateau).
- **Step 0 reads like the trained steps**: initial and plateau pass, terminal fails. So nothing
  here is "learned" in the paper's sense (B.2).
- Beside (declared): at the post-final-LN state 24, E₁ falls below the plateau on 5 of 7 prompts
  at 32000 and 54000 (median ratio 0.17, 0.18). It does so on 0 of 7 at every other step,
  143000 included (median −0.03). That fall mixes block 23 with the readout's normaliser, which is
  why it is not in the profile (D1 (a)).
- **Block 23's own output** (post hoc, after `/challenge-pr` on #342, finding 2:
  `c7_profile.py b23`, a hooked pass per prompt and step on the stored tokens, records
  `records_b23/`; the pass reproduces the stored states to ≤ 1.9e-7, and E₁ at state 23 matches
  `energies.json` to 1e-6). Before the final LN, **E₁ after block 23 is still above U1's interior
  plateau on 7 of 7 prompts at all 18 steps** (median ratio −0.12 at 143000, −0.31 and −0.32 at
  54000 and 32000). ‖m‖² is the same. On the extended states 0–24 the criterion's terminal part
  passes on 0 of 7 everywhere, and on 1–24 on 1 of 7 at 143000. So the post-LN fall at 32000 and
  54000 belongs to the final LayerNorm, not to block 23. In block 23 itself, the exact split has the
  FFN lowering ‖m‖² more than attention on 5 of 7 prompts at 143000, 1 of 7 at 32000 and 54000,
  and 0 elsewhere. It moves the tokens out at the very end late in training, but not below the
  plateau.

**F, which part moves the tokens out, blocks 20–22.**
*Revised after `/challenge-pr` on #342, finding 1:* the frozen first-order label cannot be read
at most steps, so the exact Shapley form (declared beside at the freeze) leads.
- **By the exact split, the FFN moves the tokens out at the end (σ_u < 0 and |σ_u| > |σ_a| on
  each of blocks 20–22) on 2 of 7 prompts at 32000, 1 of 7 at 54000, and 0 of 7 at the other 16
  steps**, step 0 included.
- **The frozen first-order label is unreadable at 9 of 18 steps.** It reads 0 of 7 everywhere,
  but over blocks 20–22 × 7 prompts its remainder exceeds half of Δ‖m‖² on 19 of 21 cases at
  steps 0–4, 8 at step 8, and 13–21 from 8000 on. From 4000 on, its δ_u has the opposite sign to
  the exact σ_u on 5–13 of 21. It is close to exact only at 16–2000 (remainder above half on ≤ 1
  of 21), and there it agrees with the Shapley label (0 of 7). The first draft said "fails late";
  it fails at the no-train control too (LESSONS 6).
- On the prompt mean at 143000 (Shapley), the FFN lowers ‖m‖² at block 20 (−4.0e-2, attention
  +2.5e-3), raises it at 21 (+3.3e-2), and lowers it at 22 by less than attention does (−9.5e-3
  against −2.1e-2). At 32000 the FFN lowers ‖m‖² at all three blocks, but not more than attention
  at 22. Blocks 20–22 lower ‖m‖² at some late steps, then, but not enough to bring state 23 below
  the plateau.
- The unsigned FFN speed exceeds attention's in blocks 20–22 at every step (ρ 1.4–3.1), and in
  all but 1–2 of blocks 0–22 at every step (ρ below 1 in at most 2 blocks, from 2000 on). As 1e
  U2 found, the FFN is the larger mover nearly everywhere, not only at the end.

**Q, the interaction form.** The positive share of Σ_h sym(W_QᵀW_K)/8 is 0.50 at every layer
from step 0 to 128, which is what a random form gives. It spreads to 0.43–0.67 from 2000 on and
never comes near 1. **Assumption 3.1 fails at every layer and step.** The median interior gap
(blocks 10–13) is 0.015 to 128, 0.17 at 512, 0.42 at 2000, and then 0.16–0.26. The per-head
medians are in the records. By the design, g is the gap of an indefinite form, which the theory
does not cover. *Added after `/challenge-pr` on #342, finding 3:* **for a single head the failure
is structural, not empirical.** sym(W_QᵀW_K) = ½[A B]J[A B]ᵀ with J = [[0, I₆₄], [I₆₄, 0]], so by
Sylvester's law of inertia each full-rank head has exactly 64 positive and 64 negative
eigenvalues (the per-head share is therefore not a finding). Only the sum over 16 heads could have
been positive definite. That it is not (0.43–0.67) is the empirical part, and the 0.50 up to step
128 is this structure on random weights. *Finding 4:* Q is read on the raw weights, with the LN₁
gains not folded in. The reviewer on #341 found that folding them in moves g by ≤ 0.01 at three
layers at 32000, so "fails" is unaffected; the numbers above are the raw frame's.

**What it means for the email (description, one seed):** on Pythia-410m, by the paper's own
criterion, there is no terminal escape at any of 18 checkpoints through block 23's output. The
energy at the last block is above the interior plateau, and through step 8000 it is highest at
the end. Late in training the last block's FFN does lower ‖m‖² more than attention (5 of 7 prompts
at 143000), but only the final LayerNorm takes the energy below the plateau, and only at 32000
and 54000. The FFN is the larger mover in nearly every block. Each head's symmetrised QᵀK is
indefinite by construction (inertia 64/64), and the layer's sum is indefinite too, so the
theorem's Assumption 3.1 does not hold. **Not established:** any seed but 0; the escape's rate;
anything about the paper's model, which this cannot refute.

## C7 (i) unit 0, the scan and design (2026-11-09)

Opened by STATE Blocked 153 as revised (#340), at the user's word that the rewrite of `paper/`
is done. Writing only; nothing ran. **Input:** 2605.07772 v2 (2026-10-16), the PDF's text
extracted locally (the HTML is a 404), plus six web searches. From `data/`, only the stored
files' shapes were checked; no value was read. Revised after `/challenge-pr` on #341 (accept
with changes, findings 1–6 taken; the answers are on the PR).

- **What the paper gives a measurement** (`lit-C7.md` §1): the three phases; the energy
  E_{ε,A} (clustering lowers it); Assumption 3.1 (A = QᵀK positive definite, simple λ₁); the
  plateau's rate tied to 1 − λ₂/λ₁ and λ₁e^{−λ₁} (Remark 3.10, A.53); W*_t exponentially small
  away from the terminal layer (Theorem 3.9); and **a fixed three-phase criterion, used without
  retuning** (appendix B.3), which the design adopts as written.
- **Two corrections to C7 (i)'s wording in ROADMAP and STATE (#340, finding 3).** (1) The gap is
  the weight matrix QᵀK's, not the attention pattern's. A causal pattern is lower triangular, so
  its eigenvalues are its diagonal and its eigen-gap is empty; the pattern's singular-value gap
  (2410.07799) is a different mechanism (parked, for scope). (2) The terminal escape's rate is
  R_ρ(g_ℓ), the loss gradient's Rayleigh quotient, so "the gap against the escape's rate" is not
  read.
- **What is new for a reading:** Pythia-410m is parallel-residual (`use_parallel_residual`
  true, `rotary_pct` 0.25, checked in the config), so each block's FFN update is exact from the
  stored input state. That makes the paper's Phase 3 mechanism ("the FFN moves the tokens out of
  the cluster in the final layers") a signed, per-block split of Δ‖m‖².
- **Pretrained models:** no test of the three phases found. A ViT paper (2511.21635) reports a
  three-phase depth profile of similarity whose plateau grows with depth. Parked for the
  close-out.
- **Design** (`design-C7.md`): P, F (signed: each part's first-order share of Δ‖m‖²) and Q on
  seed 0's 18 steps × 7 prompts, step 0 the no-train control, gate G0–G2 before any label, leans
  written down (1e U2's MLP split among them). J was dropped: Assumption 3.1 very likely fails
  (the reviewer's look at Q's inputs, disclosed on #341: the summed form's positive share is
  0.57–0.63 at three layers at 32000). Budget 3 units. **Blocked 154** (D1–D4, all (a)
  recommended).
- **Parked** (`lit-C7.md`): 2511.21635's figures; Wendler et al. and Shibata et al.'s final-layer
  shifts. Both are for the close-out's §8 sentence and the email.

## C6 on one page (closed 2026-11-01)

**Asked** (`design-C6.md` "What C6 is for"): read C2's two dates (E₁'s break in V, the
token-centred cloud's date) on `pythia-70m-warmup01`, 70m retrained with a ten times longer
warmup, against 70m's own dates, and label each by the schedule that predicts it. **Exit met**
(U0, U1, U1b + close-out; one unit beyond the budget of 3, extended by the user on the record,
Blocked 128 (a)): for V, "the read refused by its gate" (U1); for the centred count, a label (U1b).
Blocked 129 decided (a), the user's (2026-11-01): the close-out writes the label into `paper/`,
scoped to the centred count.

**Established** (tier 1, description; one run):

1. **The run is 70m's init and first batch with a tenth of the in-warmup rate** (U0, read off the
   weights: 76 of 76 tensors bit-equal at step 0; W 14 300, peak 1e-3 within ~10 %).
2. **70m's centred date over ten seeds is 512–1000; V's 256 or 512** (U0; `pythia-70m` and
   PolyPythias seeds 1–9).
3. **V cannot be read on the run under the frozen gate** (U1): at the top planted dose V sees plant
   B at 0.38 (7000) and 0.33 (20 000) against 0.4.
4. **The centred count on the run reads "consistent with cumulative"** (U1b): dated 2000 on three
   gated arms against step's band 512–1000; instantaneous excluded by its early set; in the
   in-sample calibration the label came from 2 of 10 cumulative plants and 0 of 10 instantaneous
   ones (the 10 step plants are the seeds' own profiles and could not give it, so the calibration
   speaks against instantaneous, not step: `/challenge-pr` on #299, finding 2). Against step, the
   direct evidence is seed 0 (same init and first batch): its centred count is 12–14 at 1000, the
   run's 2. So on 70m the centred date comes later than a fixed count of updates allows,
   compatible with the summed rate (not its point prediction: seed 0's cumulative plant dates at
   4000; finding 1); the shift is a factor of 1.0 to 3.9 against seed 0 (1.0 to 7.8 against the
   ten seeds).
5. **In `paper/`** (this close-out): the abstract's schedule sentence is scoped to 410m and names
   the run's centred result; §1's "What none of this separates" says the run tells the centred date
   apart for 70m; §7 gets "Pythia 70m with a ten times longer warmup" (the bands from 70m's dates,
   the gate, the label, seed 0's count at 1000, the shift, the calibration, the three revisions (two
   in-sample before the run was read, one after the gate's refusal), and what it does not say);
   410m's separating run stays the third kind; "compatible with", not "predicts" (finding 1).
   Every new number is a `c6.*` key (17; `numbers.py`; records U1b's `reading.json`, U1's
   `calibration_127a.json` and 3b's for seed 0) or a stated literal; four claims check the
   sentences (the label and its statuses; √10 inside both shift ranges, with seed 0's date 3b's;
   the calibration the same under every late version; seed 0's count at 1000 far above the run's).
   F1's fence in `design-C2.md` is narrowed to match. A new test recomputes every key from the
   in-repo records and fails if a value moved (finding 3).

**Not established:** whether V's date (E₁'s break) moves with the warmup; the shift's size (any
partial shift fits the grid); anything on 410m (C4); a replicate of the long-warmup recipe. V's
profile on the run and the ungated 4.78e-3 arm's are on disk and were seen: leans, not readings.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. the design | #295 | the run read off its weights; `REF70`, bands, rule, gate frozen |
| 1. 127 (a), the gate | #297 | the gate refuses; no arm gated; a vacuous known-answer check fixed |
| 1b. the centred count alone (128 (a)) | #298 | "consistent with cumulative"; V not read |
| close-out (129 (a)) | this | `paper/` abstract, §1, §7; F1 narrowed for 70m's centred date |

**Parked items at close:**
- U0's Parked 1 (a finer grid between 4000 and 16 000): **dropped**. It could only change
  instantaneous's late window, and U1b excluded instantaneous by its early set (the run already at
  16–18 at 2000); the step/cumulative edge is at 1000/2000, which the grid already holds.
- V on the run (unread, not Parked): **carried to ROADMAP C6's row** as a reopen candidate. A gate
  that reaches plant B ≥ 0.4 on the run needs a design that does not take the dose from rule
  100 (a); no unit is scheduled.

**Found at the close-out.** `numbers.json` still hashed the registry from before CLAIM-C's
withdrawal (#296 changed `claims/registry.json` without re-producing the table), so `produce`
refused until the registry was named with `--reread`; no key reading it moved. Under conda's
Python 3.10, 4 keys move in the last ulp (≤ 4.4e-16: `ridge.e1`, `ridge.ident_*`,
`reg.falsifier_p`); `.venv`'s 3.14 reproduces the table exactly, so this close-out used it.

**For the user's Overleaf copy.** Carry over this close-out's edits: the diff of
`trackC_writeup/paper/` from `d7a0795` (its last change on `main`) to this PR's merge. Build:
`pdflatex main && bibtex main`, then `pdflatex main` twice (35 pages).

**Re-run.** `METS_DATA=<main>/data <main>/.venv/bin/python -m trackC_writeup.numbers produce`,
then `python -m trackC_writeup.paper_check` (0) and `python -m trackC_writeup.numbers check`.

## C6 unit 1b, the centred count alone (2026-11-01)

The user decided Blocked 128 (a) and extended C6 by one unit on the record. Frozen and pushed
before any output on the run's grid (`fd18085`, code `c83c50f4faa6`; `design-C6.md` "U1b"),
after U1's gate output had been seen. Plant B's dose for the centred arms is fixed at unit 3b's
70m dose, 0.5 (checked against 3b's preflight by `gate`), V is not read, and every other row is
as frozen.

| item | result | record |
|---|---|---|
| seed 0's known answer | e1 equal to unit 3b's (max relative difference 0.0) | `seed0/records/`, `gate.json` |
| the re-inits' null rates | 0.0 at every arm, as unit 3b's | `gate.json` `null_vs_3b` |
| U1's top-dose arms | B, A and null shares equal (≤ 1e-12), so the same draws | `gate.json` `per_step` |
| arms gated (both steps) | **9.61e-5, 1e-4, 3e-4** (B 1.0 at 7000, 0.96 at 20 000; A ≤ 0.04; null 0); 4.78e-3 not (B 0.40, then 0.39) | `gate.json` `arms_gated` |
| rule 100 (a), beside | still refuses at both steps (V's B share 0.38, 0.33) | `gate.json` `rule_100a_beside` |
| **the centred count on the run** | **"consistent with cumulative"** | `reading.json` `joint` |
| under step | inconsistent: dated 2000 on all three arms, band 512–1000 | `reading.json` `counts.c.step` |
| under cumulative | consistent: dated 2000, band 1000–4000; its early set (≤ 512) at 0–1 | `counts.c.cumulative` |
| under instantaneous | inconsistent: its early set reaches step 2000 (which holds a 70m state ≤ 128), where the count is already 16–18 | `counts.c.instantaneous` |
| as frozen (`not_read=()`), beside | the same label (127 (a) concerns V only) | `beside.as_frozen` |
| the run's counts, 0 → 143 000 (3e-4 arm) | 0 through 256, 1 at 512, 2 at 1000, 16 at 2000, 17, 15, 14, 15, 14, 22 at 143 000; 4.78e-3 (not gated) 1 at 2000, 11 from 16 000 | `beside.V_c` |

What the label means. The frozen outcome table words it "a dose, not a fixed step". Narrowed
after `/challenge-pr` on #298 (finding 1): **on 70m the centred count's date comes later than a
fixed step allows, compatible with the summed learning rate.** The grid does not pin the shift
finely. The run rises in (1000, 2000] and seed 0 in (512, 1000], a factor of about 1 to 3.9;
the ten seeds rise in (256, 1000], a factor of 1 to 7.8. The summed rate's √10 lies inside both
ranges, but so does any partial shift, and the calibration planted only the three schedules,
none in between. On 70m's own ten seeds the date is 512–1000; on the run it is 2000. In the
calibration on the seeds' plants (in-sample, 3 arms, the same under every late-state version),
that label came only from cumulative plants: 2 of 10 cumulative plants read it, and none of the
10 step or 10 instantaneous plants did.

**Caveats.**
- **V was not read.** The label is the centred count's alone, so it does not say whether V's
  date moves with the warmup too. But V's profile on the run, and the ungated 4.78e-3 arm's, are
  on disk (`reading.json` `beside`, written by `read`). Both were seen before Blocked 129 was
  written (`/challenge-pr` on #298, finding 2). Both missed their gates, so anything said from
  them is a lean, not a reading.
- **Which plants read the label.** The two cumulative plants that read "cumulative" alone are
  seeds 3 and 6, the early risers. Seed 0's own cumulative plant dates at 4000 and reads
  "cumulative and instantaneous" (`/challenge-pr` on #298, finding 1).
- **The revision came after a gate output.** U1's refusal and its top-dose arms had been seen
  before the dose was fixed, though no output on the run's grid had been.
- **The arms are not independent.** The three gated arms (9.61e-5, 1e-4, 3e-4) give the same
  date. They are one count at near-equal tolerances, not three replicates.
- **The calibration supports the label weakly in sensitivity, strongly in specificity.** The
  label is rare on a cumulative truth (2 of 10; 8 of 10 also kept instantaneous). Here the
  run's early rise excludes instantaneous outright.
- **The grid is coarse.** The run's 2000 is one grid step past step's band (512–1000) and
  inside cumulative's (1000–4000), so the separation rests on one step of the grid. At 1000 the
  run's count is 2 of a plateau of about 16, so the rise is not near that edge. U0's Parked 1
  (a finer grid between 4000 and 16 000) would not resolve it either.
- **One run.** The run is one seed of the long-warmup recipe.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf HF_HUB_OFFLINE=1 python -m
trackC_writeup.c6_warmup run --out <dir> --models seed0`, `… gate --out <dir> --centred-only`,
`… run --out <dir> --models run`, `… read --out <dir>` (GPU, about 3 min together; seed 0 43 s,
the run 68 s). Records: `data/trackC/c6/2026-11-01-u1b/` (main tree), one code hash
`c83c50f4faa6`. `pytest tests/test_trackC_c6.py` (28; one skips without `data/`). Untested: the
gate's refusal on arms that differ from U1's (it needs the forward passes). Moving it into a
pure function would change the code hash the U1b records carry (`/challenge-pr` on #298,
finding 3).

**Parked.** None new. Whether and how the label enters `paper/` §7 and F1 is the close-out's
call (Blocked 129).

## C6 unit 1, Blocked 127 (a) and the run's gate (2026-11-01)

The user decided Blocked 127 (a): V is not read under instantaneous. It was frozen and pushed
before any output on the run (`98d3f38`, code `9c9e91f715ad`; `design-C6.md` "Decided
2026-11-01"). Then seed 0 ran into U1's directory, and the gate ran on the run's steps 7000 and
20 000. **The gate refuses**, so nothing on the run's grid was computed and no date was read.

| item | result | record |
|---|---|---|
| 127 (a)'s calibration, from U0's `seeds.json` | the same under all three late-state versions: step 7 alone (3 with cumulative); cumulative 2 alone (7 with instantaneous, 1 instantaneous); instantaneous 10 alone; as frozen, the first table again | `calibration_127a.json` `scored`, `as_frozen` (+ `_loo`) |
| seed 0's known answer | e1 equal to unit 3b's (max relative difference 0.0), tokens' hashes equal | `seed0/records/`, `gate.json` |
| the re-inits' known answer | every arm's null rate 0.0, as unit 3b's; the uncentred re-init null rate 0.0, as 3b's (recorded, not checked by code) | `gate.json` `null_vs_3b`, `uncentred_null_rate` |
| plant B through V (rule 100 (a), bar 0.4) | **the run: 0.38 at 7000 (13 of 34 units not already violating; 14 needed), 0.33 at 20 000 (7 of 21), at the top dose 0.5; no dose found, so V is not read and no arm is gated**. Beside, `pythia-70m` at 7000 with the same plant direction but its own plant-B draws (drawn last from the shared stream): 0.45 (13 of 29); unit 3b's draw: 0.41 (12 of 29) | `gate.json` `per_step`, `beside_not_gating` |
| the centred arms at the top dose, beside (not gating) | 9.61e-5, 1e-4 and 3e-4 would pass at both steps (B 0.96–1.0, A ≤ 0.04, null 0); 4.78e-3 passes at 7000 (B 0.40), fails at 20 000 (B 0.39) | `gate.json` `arms_beside_top_dose` |
| a centred-only reading, for Blocked 128 | on the 3 arms: truth survives 10 / 10 / 10 of 10, alone 1 / 2 / 10, the same under every late version; all 4 arms: alone 3 / 2 / 10 | `calibration_127a.json` `centred_only_3arms`, `centred_only_4arms` |

Records: `data/trackC/c6/2026-11-01-u1/` (main tree): `gate.json` (code `412e53d8eca4`),
`calibration_127a.json` (`fe0b7b132612`, from `seeds.json` sha256 `237fea0ffcfe…`), and
`seed0/records/` (`9c9e91f715ad`, 126 records). Set aside: `gate_v0_9c9e91f715ad.json` (the same
refusal, made before the fix below) and `calibration_127a_9c9e91f715ad.json` (equal on its keys).

**The defect, fixed before the re-gate.** The first gate (`9c9e91f715ad`) refused in 26 s and
recorded no shares. Its known-answer row compared the re-inits' rates through the per-arm
results, and a refused dose leaves those empty, so the check passed on nothing. Now the null
rates are computed whatever the dose, the check refuses unless it covers every arm, and a refused
step records its B shares and its arms at the top dose (beside). Rule and outcome unchanged;
LESSONS 2.

**Caveats.** The refusal is about the denominator more than the sensitivity
(`/challenge-pr` on #297, finding 3). At 7000 the plant turns 13 units on the run, as it turns
12–13 on `pythia-70m` over two draws. But only 8 of the run's 42 units already violate, against
13 of 42 on 70m, so the run needs 14 of 34 where 70m needs 12 of 29. One draw moved 70m's count
by one unit, which is the run's whole margin at 7000. At 20 000 it misses by two (7 of 21; 9
needed). The frozen gate takes the centred arms' plant-B dose from V's rule. That is unit 3b's
design, intended, so V's refusal also stops the centred count, whose own plants pass. C6's design
table read "V read" and "an arm gated" as separable ("The run's gate"); they are not. The
`pythia-70m` row was added after the first refusal. Its plant-B draws are its own, so it shows
that the population at 7000 passes, not that this draw passes. The centred-only calibration is
in-sample (the same plants) and was computed after the gate's output was seen, though not after
any output on the run's grid. It separates step from the other two (step survives on 10 of 10
step plants, 0 of 20 others), not cumulative from instantaneous (8 of 10 cumulative plants read
both). The key-set guard added to the known-answer check cannot fire as the code now stands, so
it guards only against a later edit.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<HDD_1TB>/mets_data/hf HF_HUB_OFFLINE=1 python -m
trackC_writeup.c6_warmup run --out <dir> --models seed0`, then `… gate --out <dir>` (GPU, ~2 min
together); `… recalibrate --seeds <U0 dir>/seeds.json --out <dir>` (CPU, no forward pass);
`pytest tests/test_trackC_c6.py` (25).

**Parked.** None new. U0's Parked 1 (a finer grid between 4000 and 16 000) stands.

## C6 unit 0, the design (2026-10-31)

The user's request: design C6's read of V and the centred count on `EleutherAI/pythia-70m-warmup01`.
The design was to confirm the run's schedule, peak, init and data order first, get 70m's date range
from PolyPythias' `pythia-70m-seed1..9`, and freeze the bands per schedule and the gate before any
output on the run. The numbers live in `design-C6.md` ("What the run is", "The reference", "The
bands", "Calibration"); this section says how to check them.

| item | result | record |
|---|---|---|
| the run, from its weights | same init (76 of 76 tensors), same first batch (sign agreement 1.000), same order to ~16 steps; in-warmup rate 0.094–0.103 of Pythia's (H0, W 14 300 at peak 1e-3); peak within ~10 % | `data/trackC/c6/2026-10-31-u0-schedule/schedule.json` (code `798c2da33a8a`) |
| seed 0 against unit 3b | records equal (max relative difference in e1 0.0); dates 256 and 1000 | `seeds.json` `seed0_known_answer_max_rel` |
| 70m's dates, seeds 0–9 | V 256 ×5, 512 ×5; centred 512–1000 at the three small arms, 1000–2000 at 4.78e-3; all readable | `data/trackC/c6/2026-10-31-u0-seeds/seeds.json` (code `9e9aa860bdf2`, sha256 `237fea0ffcfe…`) |
| `REF70` | `{"u": (256, 512), "c": per arm}`, pinned by a test | `c6_warmup.py` |
| calibration (planted on the seeds' profiles) | truth survives 10 / 9 / 9 of 10 (step / cumulative / instantaneous), alone 7 / 4 / 9; the first version (`fbc2fdd`) 10 / 8 / 9 and 1 / 1 / 9; instantaneous 2 or 4 of 10 under softer late-state plants | `seeds.json` `calibration`, `calibration_loo`, `calibration_late` |

**Caveats.** The reading was revised after the seeds' profiles were read (not the run's), which
`design-C6.md` "Worth challenging" 1 states. The seeds' records ran on forward code `5aa5eb7a067b`.
The revision changed only the reading, and the seeds' reading was re-run on `9e9aa860bdf2`. The
calibration is in-sample to the revision and has no noise beyond the seeds' own. **The
instantaneous truth survives 9, 2 or 4 of 10** depending on how 70m's states past its peak are
planted (`calibration_late`; `/challenge-pr` on #295, finding 1). The failures are all V's
count, so a "neither" on the run is ambiguous under this rule; whether to stop reading V under
instantaneous (then 10 of 10 under every version) is Blocked 127.

**Re-run.** `HF_HOME=<HDD_1TB>/mets_data/hf python -m trackC_writeup.c6_schedule --out <dir>` (CPU);
`METS_DATA=<main>/data HF_HOME=… python -m trackC_writeup.c6_warmup run --out <dir> --models seed0 … seed9`
(GPU, offline, ~70 s a seed), then `… c6_warmup seeds --out <dir>`;
`python3 tools/math_checks/c6_bands.py` (B1–B6); `pytest tests/test_trackC_c6.py` (22).

**Parked.** 1. A finer grid between 4000 and 16 000 (the run releases every 1000 steps).
*Why:* both literal-plant misses are one grid step, and instantaneous's window is the single
step 16 000 because no grid step lies in 10 000–14 300. *Cost:* about 24 checkpoints per model,
and the disk is nearly full (13 GB free on HDD_1TB). *Could change:* instantaneous's window and its
dependence on the late states. 2. (Was: the softer instantaneous plants are not tested.) Measured
after `/challenge-pr` on #295 (`calibration_late`); now Blocked 127.

## C2p readings audit (2026-10-30)

The user's request, after populating `docs/readings/` with the PDFs and arXiv sources: look through
the papers in depth for anything missed and for incorrect claims in `paper/`. Writing only; no
forward pass, no number from a record moved. The table, with marks and line references, is
`lit-C2p-fulltext.md`; this section holds what changed in `paper/` and how to check it.

| in `paper/main.tex` | change | source |
|---|---|---|
| §2.1 | Pendharkar cited for Theorem 17 as conditional (§4.2), for tying as Remark 5's idealisation; the §10 misattribution dropped; "no result" made "no monotonicity result for E_β"; 2604.01978 added (‖m‖² logistic in depth, per-layer random attention weights, small β, from all-positive overlaps) as context, not a prediction, since 0.48–0.50 of each prompt's step-0 embedding pairs have a negative cosine (measured after `/challenge-pr`, finding 1); the mean pairwise cosine = (n‖m‖² − 1)/(n − 1) | W1–W3, M8 |
| §1, §7 | the released `pythia-70m-warmup01` (14 300-step warmup) named as a run that could separate the schedule for 70m, not 410m, not read; on point dates only the instantaneous rate stands apart, step against cumulative needs 70m's date range; Belrose's KL minima (on sampled sequences) later under it | M1, M2 |
| §2.2 | Hoogland's model at a constant rate; Tigges's induction heads at 2 × 10⁹ tokens, about step 950 | M3, M4 |
| §3.1 | PolyPythias' "batch composition" | wording |
| §8 | Ethayarajh non-input layers; Razzhigaev's measure centred, so not ‖m‖²; warmup precedents (Olsson, Chen, PolyPythias); Gao on the circle; Isobe's final-layer escape against our mid-stack drops; Koubbi et al. in Theory | W4–W6, M5–M7 |
| appendix D, E6 | Belrose also read this KL, same direction, on sampled sequences | minor |

Not taken into the text: M9 (Godey's drift and attention sharpness), as nothing here reads it.
`paper_slots.json` gains two literals (950, 14 300), each a cited paper's number with its reason.

**Caveats.** Mostly targeted reads of each source at the claim, not cover to cover. M2 is a figure
read by eye (octaves, not values). `pythia-70m-warmup01` has no model card: the 14 300 steps are
Belrose's text; its peak rate and data order are unconfirmed, so §7 says "its other settings are not
published with it".

**Re-run.** `python -m trackC_writeup.paper_check` (0); `python3 tools/math_checks/c2p_mean_cosine.py`;
pdfLaTeX + BibTeX on a copy of `paper/` (34 pages, no undefined citation).

**Parked.** 1 (run the counts on `warmup01`) and 2 (confirm its schedule): *carried to C6*
(ROADMAP §8, opened 2026-10-30, the user's).

## Figure 12, the energy ridgeline (2026-10-28)

The user's request: the page's ridgeline as a paper figure, with the identity reference panel. Tier 1, a
description; nothing registered, no threshold, no date.

| | |
|---|---|
| figure | `fig/paper/fig12_ridge.pdf` (`numbers.py paper-figures`, `ridge_figure`); deterministic on this machine (two draws byte-equal); refuses if any ridge is covered by one in front of it (`ridge_hidden`, sampled at 20 points a segment) |
| record | `data/trackC/c2ridge/2026-10-28/ridge.json` (sha256 in `numbers.json`), producer `c2_ridge.py` (code in the record), CPU 2.5 min |
| inputs | A5's seed-0 records `data/trackA/a5/2026-10-20-u1/seed0/records` (126 files, sha256 over the files in name order `6ecb21551a8d…`); V checked against that run's `gate.json` known answer at every step; step-0 `embed_in` of `EleutherAI/pythia-410m`, tokenizer A5's (`step143000`), offline |
| right panel | E₁ (β = 1) at the 25 states, mean over the 7 prompts, the 18 steps on one height scale (each ridge 2.4 rows tall), step 0 at the back; a transition is drawn in the drop colour where **that mean** drops by A5's D2 rule (10⁻³); V at the right counts each prompt's drops (claim: equal to `s0.V`); where the drops sit by transition, per prompt, is figure 5's map |
| left panel | Q = K = V = I from each prompt's step-0 embedding rows (E at t = 0 equal to the record's `e1[0]` to 10⁻⁷, observed ≤ 4 × 10⁻⁹; token ids hashed to the record's), β = 1 (a key; the ceiling exp(β)/2β and the limits follow it), 2400 Euler steps to T = 8 renormalised to the sphere, read at 25 equal times; both masks, mean over the prompts; the model at step 0 beside |
| what it shows | full mask: smallest fine-step change 3.7 × 10⁻⁸ (it refuses on a fall), within 1 % of e/2 (one point) by read 14 of 24; causal (no theorem): smallest change 1.7 × 10⁻⁵, ends 1.33–1.34; the model at step 0 rises 0.52 → 0.88; ridges at 0–128 have no drop; the highest ridge is step 16's (1.23); the mean ends above the embeddings at every step (claim) |
| matches | the page's identity numbers to 7 × 10⁻¹⁶ (its scratch `identity_ref.py`, ported unchanged in what it computes) |

**Caveats.** t is not depth: T = 8 is figure 6's choice, so the identity curve's rate against the
model's states means nothing; its shape (monotone, to a ceiling) is the reference. The causal mask's
monotonicity is an observation on 7 prompts at one step size (dt = 1/300), not a theorem. The ridges
are the mean over prompts: a prompt's drop where the mean rises is in V, not in the colour. Seed 0 only.

**After `/challenge-pr` on #280** (accept with changes; findings 1–6 taken): at 3.4 rows step 16's ridge
hid 69 % of step 8's, now 2.4 with a refusal on any cover (the page TBed9FFnPDLDWKAeN1duSQ draws its
ridges taller still, so it likely hides step 8 too; not republished here); the colour was "any prompt
drops" on the mean's line (64 of 145 coloured segments rose), now the mean's own drops; a key on the
right panel and the caption below; the claim split by subject; β a key; the E(0) tolerance 10⁻⁷; the
drop rule imported from `c2_readings`, the model name from `a5_seeds.repo_for`; tests
(`tests/test_trackC_ridge.py`).

**Caption (draft, for `paper/`; yours to edit).** *The interaction energy through depth, at every
checkpoint (seed 0). Left: $E_1$ under identity weights ($Q = K = V = I$, $\beta = 1$), run from step 0's
embedding rows of each prompt and read at 25 equal times to $t = 8$; $t$ is not depth. Under the full
mask $E_\beta$ is non-decreasing [Geshkovski et al.] and rises to $e/2$, every token at one point; under
the causal mask it also never falls here (observed on these prompts, no theorem). The model at step 0
(black) rises too, more slowly. Right: the model's $E_1$, averaged over the 7 prompts, against the 25
states, one ridge per checkpoint, step 0 at the back; orange where that average falls by more than
$10^{-3}$ of its value; at the right, V, the drops of the 7 prompts summed. Through step 128 no prompt
drops; from 256 the ridges fold. The average ends above the embeddings at every step.*

**Also in this unit.** `p_a5_push_record.json` (a `repo:` record of `numbers.py`, §3.4's outside
timestamp) was gitignored and its one copy sat untracked in `../Mets-work`; now whitelisted and tracked,
sha256 equal to the table's (`LESSONS.md` 2). `reg.falsifier_p` moved by one ulp (…338 → …339) under
Python 3.14.8's float `pow`, named with `--reread`; its rendering is unchanged. This venv redraws
`fig11_rho` and `figD_*` with other bytes (matplotlib), so the tracked ones were kept.

**Re-run.** `METS_DATA=<main>/data HF_HOME=<main>/data/hf HF_HUB_OFFLINE=1 python -m trackC_writeup.c2_ridge`,
then `METS_DATA=<main>/data python -m trackC_writeup.numbers produce` and `python -m trackC_writeup.numbers
paper-figures`; copy `fig/paper/fig12_ridge.pdf` into `paper/fig/` when `paper/` includes it.

## Figure 13, the layer × step small multiples (2026-10-29)

The user's request: effective rank, mass near 1, energy change and the Schur ratio by depth and
step, for the paper, from `data/phase12`, after a check that those runs reproduce A5's energies.
Tier 1, a description: nothing registered, no threshold, no date, no forward pass. One unit; exit:
the figure drawn from the table after the check passes.

**Input set:** seed 0 (`pythia-410m`), the 18 steps (`a1a.STEPS`), the 7 v1 prompts; Phase 1's
runs `a1a.RUNS` (`geometry.json`, `energies.json`, `layer_metrics.csv`, `tokens.txt`) and one
Phase 2 run per step (`c2_grid.P2_RUNS`, `ov_summary`); A5's seed-0 records
`data/trackA/a5/2026-10-20-u1/seed0/` and gate, A3's rows. Code `d84d6186e3ef`; inputs sha256
`f7574b8cdac1` (every file read, the gate and A3's rows included after `/challenge-pr` on #281,
finding 4); record `data/trackC/c2grid/2026-10-29/grid.json` (sha256 `815063581cc3`), CPU, 2 s.

**The check first (refuses on any failure, before a grid value is read): passes.**

| arm | result |
|---|---|
| `tokens.txt` against A5's record tokens | equal in 126 of 126 |
| E₁ at the 25 states against A5's `e1` | max relative difference 1.5e-6 (A5's ENERGY_TOL 1e-4; A5's own gate read 1.5e-6, GPU against CPU) |
| `layer_metrics.csv` `energy_beta1.0` against `energies.json` | equal (exact) in 126 of 126 |
| V counted from these energies by A5's rule | equal to A5's gate known answer at all 18 steps (also a sentence claim in `numbers.py`) |
| beside: each prompt's minimum effective rank over states, prompt mean, against A3's E5 | max relative difference 4.2e-4 (A3's own rtol 5e-4; the reports print 2 decimals) |

**The quantities** (prompt means; the Schur ratio is the weights' alone):

| panel | depth | source | colour |
|---|---|---|---|
| effective rank of the unit tokens (paper E5's quantity) | 25 states | `geometry.json` `effective_rank_normed` | log, one hue |
| mass near 1, the share of pairs with cosine > 0.9 | 25 states | `ip_mass_near_1` | log, one hue |
| energy change (E₁(t) − E₁(t−1)) / E₁(t−1) | 24 transitions | `energies.json`, β = 1 | symmetric log, linear inside ±1e-3 (A5's REL_TOL) |
| Schur ratio log₁₀ ‖T₊₊‖ / ‖T₋₋‖ | 24 blocks | `ov_summary` `schur_cond` | diverging at 0 |

**What it shows (description, read after drawing; leans, not readings):**

- Effective rank: the last state falls from 14.7 (step 0) to 2.1 at step 16 and climbs back past
  its start by 1000 (31.1); the minimum of the grid is that cell.
- Mass near 1: state 0 is the duplicate-token floor, 0.0166 at every step (`design-1.md`); the
  transient reaches 0.65 at step 16, state 24; at 143000 the middle states fall to 0.00034, below
  the floor.
- Energy change: no mean change below −1e-3 through 256; 7–10 of 24 transitions below it from 512
  on. **The mean hides step 256** (`/challenge-pr` on #281, finding 2): V is 19 there, but no cell's
  mean falls below −1e-3 (the lowest −0.0004), because one prompt's drop averages out against the
  others' rises. So this panel is not figure 5's drop map; it shows the mean change's size, and the
  break's first step does not appear on it.
- Schur ratio: within ±0.019 of 0 through step 128; every block from 1 to 23 negative at 256, 512,
  1000 and 2000, the minimum −1.10 (block 14, step 2000); mixed after, positive in the last blocks
  at 143000. `math-2.md` §3.2: a degeneracy flag, not a condition number, and blind to the blocks'
  coupling. **About half of it is the blocks' sizes** (finding 1, re-computed here): the Frobenius
  norms grow with block dimension, so over the 432 cells log₁₀ `schur_cond` correlates 0.982 with
  ½ log₁₀(`schur_dim_attract` / `schur_dim_repulse`); with that subtracted, every negative cell
  stays negative, the range narrows from [−1.10, 0.58] to [−0.79, 0.38] and 47 % of the variance
  remains. "The repulsive block is larger" is therefore partly a count of repulsive directions.
- Which quantity each of these two panels should draw is Blocked 116, the user's.

**Not established:** anything per prompt (means only); other seeds; that the Schur ratio's swing
and E₁'s break are related (beside each other on one grid, no test); the figure's place in
`paper/` (the user's: copy `fig/paper/fig13_grid.pdf` to `paper/fig/` and include it).

**Re-run:** `METS_DATA=<main>/data python -m trackC_writeup.c2_grid`, then
`METS_DATA=<main>/data python -m trackC_writeup.numbers produce` and `python -m trackC_writeup.numbers
paper-figures` (two draws byte-equal). Tests: `tests/test_trackC_grid.py`, `test_trackC_figures.py`.

## C2p on one page (closed 2026-10-24)

**Asked** (`design-C2p.md` "What C2p is for"): make C2's draft ready for the user to rewrite and
post (the LaTeX, figures to cull from, the ROADMAP row's revisions), with no new reading.
**Exit met** at its budget, 2 units + close-out (#267, #268 and this): `paper/` compiles with
pdfLaTeX; every item of the ROADMAP row is done or handed to the user by name; the second
fresh-context read is answered (`referee-C2p.md`). Blocked 109 decided (a), the user's
(2026-10-24): Claude makes the recommended cut and framing points at the close-out.

**Established** (writing only; no number moved; the draft's claims are C2's and C5's):

1. **`trackC_writeup/paper/` is an Overleaf-ready draft**: pdfLaTeX, 28 pages, no undefined
   reference; every number in it is a record key or a literal with a stated reason
   (`paper_check` 0, CI). The check catches 42 % of one-digit changes, so it is not a proofread.
2. **The cut** (109 (i)): seven figures in the main text (`fig1_break`, `fig7_events`,
   `fig8_mean`, `fig10_level`, `fig2_timeline`, `fig3_ov`, `fig4_schedules`). Three are in
   appendices: `fig11_rho` (A), `fig5_count` and `fig9_massive` (C). `fig6_picture`, an
   illustration, is out of the text; its file stays in `paper/fig/` so it can be put back.
3. **The second read's framing points** (109 (ii); each row in `referee-C2p.md`). The title and
   abstract now lead with the count, and with its drops at β = 1 being falls of the tokens' mean
   (one direction only: E₁'s falls are the mean's, the converse not read), with the particle
   picture as motivation (M1 (d)). §4 leads with a per-seed table (checked against `a5.V` and
   `s0.V`), and p and E come after it in one sentence, out of the abstract (M2). The abstract says
   "one grid step later" in place of the heads' step, §1 item 2 and §4.1 apply §5's convention to
   that coincidence, and §5 states the alignments' base rate, plain and anchored at the break
   (`tools/math_checks/c2p_alignment_base_rate.py`), so the alignments are not read as links (M3).
   There is a glossary (§3.5) and per-row definitions with bootstrap units (appendix D) (M6).
   m6, m13 (checked in `c2_aniso.py`), m17 (dropped) and m19 are stated.
4. **One correction to C5** (unit 2, already under `## Corrections received`): 2510.06477's
   Pythia grid reads no step in (1, 1000), so its "around 1k" bounds, not dates.

**Not established / not done:** anything new about the model. **The user's:** m11 (anonymity, the
timestamp narrative, "the author's notes"), m12 (the abstract's density beyond the reframing),
M3 (iii) (how the shares at 512 are used), a code and data statement, the venue, the records'
release and posting (Blocked 103 (a)). No third fresh read of the reframed draft was run (the
budget closed it). **ROADMAP C5:** the readings the second read asks for (V_m dated, a magnitude
statistic, a β sweep, both normalisations on GPT-2, the set-aside dating rules on the nine seeds,
a per-cell OV table, the substitution on more seeds).

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 1. the LaTeX and ten figures | #267 | `tex/` from the Markdown, CI-checked; figures 6–10 new |
| 2. the revisions and the second read | #268 | `paper/` the source; `paper_check`; major revision, 6 + 20, answered |
| close-out (109 (a)) | this | the cut and the framing points in `paper/` |

**Parked items at close:** none (units 1 and 2 parked none; every referee point has a row in
`referee-C2p.md`).

**For the user's rewrite.** *It is the next item (STATE Blocked 152 (d), 2026-11-09, the
user's): no new row opens before it, so the rewrite proceeds with F1 open as §7 words it;
whether the paper is posted before F1 is read stays the user's (Blocked 40, 103). After it,
C7 (i), Isobe et al.'s depth profile on 410m with attention's spectral gap per layer, opens
first; its reading may enter §8 before you post. C4's design follows it (STATE Blocked 153,
revised 2026-11-09; the order assumes posting before C4 reads F1).*
If the Overleaf copy has been edited since #268, carry over this
close-out's edits: the diff of `trackC_writeup/paper/` from `7d9cfe6` (#268's merge) to this PR's merge.
Re-run as in "C2p unit 2" below. The 28-page build is `pdflatex main && bibtex main` and then
`pdflatex main` three times.

**Cited context for the rewrite, from Track N's close (2026-11-08; not results, neither re-run):**
70m's German context neurons forming as a group at 512, "coinciding with the LR warm-up"
(2311.00863), a dated event at C2's step on F1; and neuron births on `pythia-410m-deduped`
(neuron-lineage) at every step of a five-point grid, which date nothing. The wording and the
numbers: `trackN_neurons/status-N.md` "Track N on one page (closed 2026-11-08)".

**Cited context for the rewrite, from S4's scan (2026-11-08; a fetch summary, not read in full):**
Gopalani and Hu (2506.13688, NeurIPS 2025) report that in Pythia's last layer the token states'
mean pairwise cosine goes from 0.4–0.65 at init to above 0.9 within a few steps, "followed by a
decrease around step 1000". That bears on §8 (C2's mean term, ‖m‖², and the centred date at
1000) and is not cited in `paper/`. Read its §6 and figure 8 before quoting it:
`trackS_subspace/lit-S4.md` §2.

**Cited context for the rewrite, from S4's close (2026-11-09; description, not a result on 410m):**
on HAPAX's Pythia-1b-shaped pair (2511.05743, trained with and without the loss on
induction-predictable tokens), both energy counts rise alike, within 5 counts through step 800,
whether or not the paper's induction transition happens. It bears on F2 ("break, then induction"
not shown causal), at its weight: one seed a side, 1B on the Pile, Hapax weakens induction rather
than removing it, and no noise floor (`pythia-1b`'s checkpoints stay reserved by `design-8.md`,
Blocked 7), so no gap is read. Before quoting it, date Hapax's own 1–2 induction heads from the
paper's second panel (by eye the top head rises from ~1000 to ~2500, after both counts'
plateaus; one colour-and-alpha pass on the figure). `trackS_subspace/status-S.md` "S4 on one
page (closed 2026-11-09)".

## C2p unit 2, the revisions in `paper/` and the second fresh read (2026-10-24)

Blocked 107 (a) and 108 (a), the user's (2026-10-24). Writing only: no forward pass and no new
reading. Every number added is a key from a record already on disk, or a cited paper's.

**Where the draft lives now.** Under 108 (a) the draft's source is `trackC_writeup/paper/`, the
user's Overleaf folder, which nothing generated writes into. The user had uploaded `tex/` and not
yet edited it (the Desktop zip `c2_draft_overleaf.zip` was byte-identical to `tex/`), so `paper/`
starts as `tex/`, and `git diff --no-index trackC_writeup/tex trackC_writeup/paper` is exactly this
unit's edits. If the Overleaf copy has already been edited, that diff is what to carry over.
`tex/` and `draft-C2.md` stay the record-backed version (CI unchanged).
`python -m trackC_writeup.paper_check` lists every number in `paper/*.tex` that no key renders to
(a slot of `draft-C2.src.md`, or of `paper_slots.json` for numbers new to `paper/`) and that the
record-backed draft does not write literally. `tests/test_trackC_paper.py` fails on any such
number, so the user's rewrite is checked in CI. It reads 0 now. What it does not catch: it
compares tokens, not contexts, so a changed small integer that some key also prints passes.
It was planted-tested on 1590, 9.8 and 12 000: each listed. Measured by `/challenge-pr` on #268, a one-digit change to each of the 1051 numbers in `paper/main.tex` is caught 441 times (42 %); a seed count or a grid date passes (stated in the docstring). A number no record holds goes into `paper_slots.json` as a literal with a required reason, and `stale_figures` checks `paper/fig/` against the generated copies (both added after `/challenge-pr` on #268).

**pdfLaTeX.** Installed as TeX Live 2026 from Fedora's packages in a distrobox (`latex`), with
`pdflatex` and `bibtex` exported to `~/.local/bin`; TinyTeX's install script was refused by auto
mode. `tex/` compiles clean (24 pages; the labels settle on the fourth pass), and so does
`paper/` after this unit (26 pages, no undefined reference). The one BibTeX warning is the Mistral
checkpoints' missing year.

**The revisions** (the ROADMAP row's items):

| item | what changed in `paper/main.tex` |
|---|---|
| M2's results, level first (107 (a)) | §4.1 gains "Which coordinates carry the mean". It opens with the level (half of ‖m‖² in 132–135 of 1024 coordinates at 512 and 1000, random bases 127–129). Then the frozen rotation reading: not basis-aligned on 9 of 9, no recurring carrier. Then the massive set: 2–3 per seed, ≤ 2.35× at 128–1000, 100× at 16 000–32 000, more than half of none of the 1224 falls, and projected out, dates kept on 7 of 9 and 9 of 9 ("a check, not a test"). Then the gate's reach (3 coordinates) and unit 2's rule: about the basis, not a count. Then the level after the break. One clause each in the abstract and §1 item 2. Appendix C's paragraph becomes a pointer |
| the late carriers' wording | "one set per seed, the union over the grid, not the set at the last step, which differs on some seeds", and the massive-set figure's caption explains the fall at 143 000 on seeds 3 and 4 |
| §8 and F9 against 2510.06477 | Read from the paper's own HTML: its appendix B.1 reads Pythia at steps 1, 1k, 2k, 4k, … on one GSM8K prompt. So its "around step 1k" bounds its changes to (1, 1000], which holds both of C2's dates and orders neither. It is cited as the nearest published reading; its criterion for massive activations (the first token's norm) is not ours, so the two dates are not one quantity. F9's "no reviewed measurement inside the window" is gone. §8's novelty sentence now says "on checkpoints inside the first 1000 steps". Massive activations are no longer left open as the mean's carrier at the break. This corrects C5 (`## Corrections received`) |
| m15 | §7 cites the Mistral release post (Karamcheti and Orr 2021, read 2026-10-24): five GPT-2 small seeds, each its own seed. The checkpoint spacing stays the repository's, since the post gives none |
| m19 | §7: 70m and 160m have 6 and 12 layers, so V is at most 6 × 7 and 12 × 7 against 24 × 7 (keys `cfg.layers_70m`, `cfg.layers_160m`) |
| m25 | Appendix A gains ρ by family and step (`fig11_rho`, S1's records, drawn by the new `numbers.py paper-figures` into `fig/paper/`, copied into `paper/fig/` by hand) |
| the second fresh read | `referee-C2p.md`: major revision, 6 major and 20 minor. Every point is answered there. The fixes that need no reading are made (the 0.956 rounding, seed 6's centred date, §2.1's citation, §6's rationale, the abstract on the step-128 substitution, the GPT-2 bands, captions, internal labels). The framing points are Blocked 109's. The ones that need a reading are carried to ROADMAP C5 |

Keys: 270 (was 254) from 27 records (was 26; `c5_mean.py` added for its frozen constants).
`numbers.py`'s `constant()` now reads a constant with a trailing comment. The level figure draws
the random-basis band's edges over the seeds' lines (`fig10_level`, both drafts). The record-backed
Markdown keeps its "0.96 or more" (true minimum 0.956); it is frozen under 108 (a), and `paper/`
has the fix.

**The figures**, by file (LaTeX numbers by position; the map in "C2p unit 1" with this change):
`fig11_rho` is new and sits in appendix A, so in `paper/` it is figure 8, and `fig5_count`,
`fig9_massive` and `fig10_level` are 9, 10 and 11. Eleven files to cull from; the cut is the
user's (Blocked 109).

**Not established / not done:** the user's figure cut and the framing points (Blocked 109). V_m
dated, a magnitude statistic, both normalisations on GPT-2, the set-aside dating rules on the nine
seeds (carried to ROADMAP C5). m13 and m17 are not checked here (Blocked 109).

**Parked:** none (each referee point has a home in `referee-C2p.md`'s table).

**Re-run.** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`, then `paper-figures`;
`python -m trackC_writeup.paper_check`; compile: `cd trackC_writeup/paper && pdflatex main && bibtex
main && pdflatex main && pdflatex main && pdflatex main`, or upload the folder to Overleaf.

## C2p unit 1, the LaTeX and ten figures to cull from (2026-10-24)

Opened at the user's request in the session that closed C5 (`design-C2p.md`: 2 units +
close-out). Writing and drawing only: no forward pass, no new reading; every number in the new
captions and paragraph is a key from a record already on disk.

**The LaTeX.** `trackC_writeup/to_tex.py` converts the rendered `draft-C2.md` into
`trackC_writeup/tex/` (`main.tex`, `refs.bib`, `fig/*.pdf`), the folder to upload to Overleaf.
It targets pdfLaTeX (Overleaf's default): the draft's Unicode mathematics becomes math mode, the
two display equations are set by hand, citations become natbib author–year keys (36 entries, each
arXiv one with its link), and §N, figure N, table 1 and appendix X become `\ref`s. It refuses a display
equation, citation, figure reference or character it cannot convert. Compiled here with tectonic
(XeTeX): 22 pages, no undefined reference, every figure before the references; the one BibTeX
warning is the Mistral checkpoints' missing year (the draft gives none). pdfLaTeX itself was not
run here (not installed); the converter refuses any character outside T1's accented Latin, which is
the condition pdfLaTeX needs. CI (`tests/test_trackC_tex.py`) checks that the tracked `tex/` is the
conversion of the tracked draft and that every figure has its PDF. LaTeX numbers figures by
position, so the Markdown's figure 6 is the LaTeX's figure 1; the `\ref`s follow.

**The figures** (1–5 unchanged; 6–10 new, drawn by `numbers.py figures` from the table). LaTeX
numbers figures by position, so the two drafts number them differently; cull by file
(after `/challenge-pr` on #267, finding 4):

| file | Markdown | LaTeX | where | what | from |
|---|---|---|---|---|---|
| `fig6_picture` | 6 | 1 | §2.1 | the identity-weight picture: 40 tokens on S² clustering, E₁ never falling | `_toy`, a simulation (an illustration, refused if its E₁ ever falls); **not a record** |
| `fig1_break` | 1 | 2 | §4 | V by step on the ten seeds | (C2 unit 2) |
| `fig7_events` | 7 | 3 | §4.1 | the two events on the ten seeds, uncentred and centred (10⁻⁴), each over its plateau | unit 3s's `reading.json` |
| `fig8_mean` | 8 | 4 | §4.1 | seed 0's E₁ and ½ + ½‖m‖² by state at 512 and 1000; the share of transitions where the two move together, by step | unit 5's profile, unit 5b's readings |
| `fig2_timeline` | 2 | 5 | §5 | the rows on seed 0 | (C2 unit 2) |
| `fig3_ov` | 3 | 6 | §6 | S2's OV edits | (C2 unit 2) |
| `fig4_schedules` | 4 | 7 | §7 | the other schedules | (C2 unit 4) |
| `fig5_count` | 5 | 8 | appendix C | where the drops sit; E₁ by depth | (C2 unit 5b) |
| `fig9_massive` | 9 | 9 | appendix C | each seed's massive set over the median coordinate, by step, against 100× | C5 unit 1's `reading.json`, seed 0's `set.json` |
| `fig10_level` | 10 | 10 | appendix C | coordinates holding half of ‖m‖², the model's basis against random bases, by step | C5 unit 2's `reading.json` |

Appendix C's paragraph "Which coordinates carry the mean" exists for `fig9_massive` and
`fig10_level`; it goes if both are cut. The converter sets the draft's Unicode mathematics, but
single-letter variables written in plain letters (V, R, m, n, Q = K = V = I) stay in text font,
and `E = κ p^(κ−1)`'s E with them (`/challenge-pr` on #267, finding 2, which also found that
exponent printed as a literal caret: now set, and a bare caret refuses).

Records: 26 (was 23), 254 keys (was 241), 60 sentence claims (3 new: the massive coordinates
ordinary at 128–1000 and at 100× only after 1000; the mean spread at the break and below a random
basis's count by 4000; the centred half-rise at 1000 on most seeds); no existing key or record
moved. Appendix C gains one paragraph, "Which coordinates carry the mean", for figures 9 and 10
(the read-out caveat stated); §4.1's own M2 text is unit 2's, after Blocked 107. A first render
said the massive coordinates' largest ratio at 128–1000 "exceeds" none at 2.3× when the record's
is 2.35; caught before commit and printed to two decimals. The PDFs are deterministic (two draws,
equal bytes).

**Not established / not done:** which figures to keep (the user's cull); m15, m19, m25, §8 and F9
against 2510.06477, the second fresh read (unit 2); a pdfLaTeX run.

**Parked:** none.

**Re-run.** `METS_DATA=<main>/data python -m trackC_writeup.numbers produce`; then `render`,
`figures`, and `python -m trackC_writeup.to_tex build`. Conda `mets` env (matplotlib 3.10.8).
Compile: upload `trackC_writeup/tex/` to Overleaf, or `tectonic -X compile main.tex` inside it.

## C5 on M2, on one page (closed 2026-10-24)

**Asked** (`design-C5.md` "What this opening of C5 is for"): the referee's M2. C2 unit 5b showed
that V at β = 1 nearly counts the layers at which the tokens' mean norm ‖m‖² falls; is that mean
the massive-activation coordinates, or any few coordinates, and do C2's two dates survive
removing them? **Exit met** (budget extended on the record to 3 units + close-out by Blocked
105 (a); #263–#265 and this): R0, R0′, R0b, P1, P2 (unit 1) and Q1–Q3 (unit 2) read on the 9
PolyPythias seeds with seed 0 the development set, both controls, each gate passed before any
scored record (unit 2's after one refusal and Blocked 106 (a)). Writing only here.

**Established** (tier 1, description; `pythia-410m` seeds 0–9, the 18 grid steps, v1's 7
prompts; numbers in the unit sections below):

1. **The massive-activation coordinates are not the break's.** Each seed's 2–3 coordinates are
   ordinary at 128–1000 (≤ 2.4× the median coordinate's max), reach 100× only at step 16000
   (32000 on one seed), and carry none of the falls V counts (R0′: share above ½ at 0
   transitions on every seed). Projected out, the break's date is kept on 7 of 9 seeds (the two
   others cross the half line by one violation at one arm, as random sets also do) and the
   centred event's on 9 of 9 (unit 1): a check, not a test that could have moved, since the
   coordinates removed are ordinary at the break.
2. **No other few coordinates are the break's either.** The falls V counts at 128–1000 (1224, 5b's
   family) are no more concentrated in the model's coordinates than in a random basis, 9 of 9
   ("not basis-aligned"; post hoc, slightly more spread on every seed); no coordinate recurs in
   ¼ of any seed's falls; the mean itself is spread there (half of ‖m‖² in 132–135 of 1024 at 512
   and 1000, a random basis 127–129). It equals a random basis's at 2000 (126–131), is below it
   on every seed at 4000 (104–113), and sits in 6–24 coordinates at 8000 and 2–3 at 16000 (unit 2).
3. **So at the break's steps no privileged coordinates of the kind tested carry the break**:
   neither the massive set nor a concentration like the gate's plant (3 coordinates). A
   statement about the basis, not a count: a few coordinates hold half of each *net* fall in any
   basis, by cancellation (unit 2's gate), so the draft may not say "not in a few coordinates".
4. **The lean from 2510.06477 ("around 1k") did not hold for this statistic** on 410m's seeds:
   the coordinates reach 100× an order of magnitude later (unit 1; `lit-C5.md`).

**Not established:** any causal statement (a read-out of the states, not a clamp in the forward
pass, 2607.09279's intervention); a direction that is not a coordinate (rotation-invariant, which
C2's centred count already reads); a privileged set of 8–16 coordinates carrying part of each
fall, or the plant on the scored steps (unit 2's gate planted 3 coordinates at 7000); when the
falls V counts become basis-aligned after 1000 (the level turns between 2000 and 4000); the attention sink's share per head; other
β, rungs or prompts; any separation from the LR warmup (F1, C4).

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. scan and design | #263 | proposed; 104 (a), the user's, froze the set and scope |
| 1. M2: projection and recount | #264 | P1 kept 7 of 9, P2 9 of 9, R0′ 0, R0b 16000 |
| 2. M2's other half (105 (a)) | #265 | the absolute "few" refused by its gate; 106 (a): "not basis-aligned", 9 of 9 |
| close-out | this | writing only |

**Parked items at close:**

| item | from | outcome |
|---|---|---|
| When m becomes basis-aligned (the level as a random basis's at 2000, below it at 4000 on every seed, 6–24 coordinates at 8000; corrected at the close, `/challenge-pr` on #266, finding 3: unit 2's Parked had "spread at 4000") | unit 2 | **carried to ROADMAP C5's row** (unscheduled): Q2 on the falls at 2000, 4000 and 8000 is a read of unit 2's records (`c5_mean.family` takes the steps), no forward pass; dating the level inside (2000, 4000] needs off-grid checkpoints. C2p quotes the recorded levels as description, with R0b's 16000 beside |
| The carriers change late on seeds 3 and 4 (seed 3's end-state set is not its 16000–54000 one; seed 4's are gone at 143000) | unit 1 | **carried to C2p**: the draft says the set is per seed, the union over the grid, not an end-state set; S3's choice of coordinates reads the same caveat in unit 1 |
| Unit 1's first pass 1 (`2026-10-23-u1-v0`), kept as the defect's evidence | unit 1 | **dropped**: `LESSONS.md` 3 records the defect; the directory under `data/` (untracked) is the user's to delete |
| C2's §8 and F9 against 2510.06477 | unit 0 (`lit-C5.md` Parked 1) | carried to C2p at C5's opening (unchanged) |

**What it changes for the draft** (each an edit in C2p, none made here): §4.1 gains M2's two
results (the massive coordinates ordinary at the break and outside the falls; the falls not
basis-aligned, the mean spread), with the read-out caveat, the gate's reach (a 3-coordinate
plant) and unit 2's rule that it does not say "not in a few coordinates"; whether it leads with
the level (Q3) or the frozen rotation reading (Q2) is **Blocked 107, the user's** (`/challenge-pr`
on #265, finding 6, homed at the close after #266's finding 4); §8's "Anisotropy, outlier dimensions
and sinks" and appendix A's note no longer leave massive activations open as the mean's carrier
at these steps; §8 and F9 meet 2510.06477 (its "around 1k" not reproduced for this statistic).

## C5 unit 2, M2's other half: the falls at the break are not concentrated in the model's coordinates (2026-10-24)

### The gate refused, first

Blocked 105 (a), the user's; budget 3 units + close-out on the record. Design `design-C5.md`
"Unit 2", code `trackC_writeup/c5_mean.py` and `tests/test_trackC_mean.py`, frozen and pushed in
`ded356f` before any output. Code hash `code_sha()` as recorded in `gate.json`. Tier 1, description.

**Input.** `pythia-410m` seed 0, off-grid step 7000 (trained, not scored), the 7 v1 prompts,
float32 forward on the RTX 3080, TF32 off, m float64. Its 60 transitions at which E₁ drops by
more than 1e-3 and ‖m‖² falls. Record `data/trackC/c5/2026-10-24-u2/gate/` (main tree):
`gate.json`, `preflight_input.json` (as refused, now `gate-v0-ded356f/`).

| population (60 transitions) | read "few" (k½ ≤ 16) | f_rot ≤ 0.05 | median k½ | bar |
|---|---|---|---|---|
| **planted null** (each fall turned by a Haar rotation) | **0.65** | 0.03 | 11 | few ≤ 0.05 (**fails**), f_rot ≤ 0.10 (passes) |
| plant, f = 0.5 (a third of the fall in 3 coordinates) | 0.90 | 0.85 | 8 | beside |
| plant, f = 1 (half) | 1.00 | 0.93 | 3 | beside |
| plant, f = 2 (two thirds) | 1.00 | 0.97 | 3 | ≥ 0.9 both (passes) |
| trained at 7000 (seen; not a scored cell) | 0.98 | 0.95 | 2 | — |

**Why the absolute bar fails.** k½ is taken on the *net* fall Δ = Σδᵢ, and the coordinates
move both ways: ‖m‖²'s net fall is small against its per-coordinate moves, so in any basis about
11 coordinates of 1024 already hold half of it. The pure tests' spread fall was a pure scaling
(every δᵢ negative), so they never met the cancellation (`LESSONS.md` 6). The rotation reading
does not depend on that: it compares each fall with the same two vectors in random bases, and it
separates the null (0.03) from the plant (0.85 already at f = 0.5).

**Blocked 106 decided (a), the user's, 2026-10-24**: Q2 (the rotation reading) is the primary,
Q1's k½ beside unlabelled. The gate re-ran under the revised code `0b4cac7817e2` with the same
draws (k½ identical) and passes (null 0.033, plant 0.97); the refused run is kept as
`gate-v0-ded356f/`. `tools/design_preflight` `7ab262644eb7` refuses on floor, power (no e-value
taken) and band (argued in the design). Frozen and pushed in `6db31cf` before any grid-step record.

### The reading (2026-10-24)

**Input.** Seeds 0–9 (`pythia-410m`, `-seed1`…`-seed9`), the 18 grid steps, the 7 v1 prompts,
float32 forward, TF32 off, m float64; code `0b4cac7817e2` for every record (180 npz, 36 min,
runner). Known answers met on every record: tokens' hash, e1 and Σmᵢ² (and Σ over S) equal to unit
1's records (first record: e1 0, ‖m‖² 4e-16, over S 2e-16). Records
`data/trackC/c5/2026-10-24-u2/seed{0..9}/` (main tree), `reading.json` sha256 `12fcbe65d923`
(re-read under `bcabd4e430f6` to add `post_hoc_ties`; every other key equal to the first read's,
`d1116b2eaf98`).
**The family is unit 5b's:** 101–135 falls per seed, 1224 over the 10 seeds at 128–1000 (5b's
1224), ‖m‖² falling at every one (none skipped).

| reading | result |
|---|---|
| **Q2, the primary: is a fall more concentrated in the coordinates than in a random basis?** | **"not basis-aligned", 9 of 9** (0 aligned). The fraction of a seed's falls at f_rot ≤ 0.05 is 0–0.022 on every seed (the gate's turned null: 0.033; ties keep it below 0.05), the median f_rot 0.85–0.96 (the turned null's 0.655); per step ≤ 0.07 at 256 and ≤ 0.04 at 512 and 1000. Seed 0 (development): 0.008 |
| **The other way** (post hoc, `/challenge-pr` on #265, finding 2; `reading.json` `post_hoc_ties`) | **the model's basis spreads the falls slightly more than a random basis does, on every seed**: ties apart, a rotation holds half the fall in fewer coordinates at 41–50 % of draws and in more at 15–24 % (ties 30–38 %); the fall sits above 95 % of its rotations at 5–15 % of a seed's falls, below 5 % of them at 0–2.2 %. Seed 0: 43 %, 23 %, 6.1 %, 0.8 %. Not a frozen reading; no label is taken from it |
| Q1, k½ (beside) | half a fall takes a median of 5–8 coordinates (quartiles 3–14), against 4–8 for the same two vectors turned at random; 0.78–0.91 of falls at ≤ 16. **So by the design's own count the falls do sit in a few coordinates, in either basis**: a net fall is 6–10 % of the coordinates' gross movement Σ\|δᵢ\| (median per seed), and a few of the moving coordinates outweigh that net. "A few" is the cancellation's, not the model's coordinates' (the gate) |
| Q3, recurring coordinates | **none**: no coordinate is in the k½ set at ≥ ¼ of a seed's falls, on any seed (so none of S or M) |
| Q3, the level of ‖m‖² | at the break the mean itself is spread: half of ‖m‖² takes 132–135 coordinates at 512 and 1000 on every seed (participation ratio 352–366), against 127–129 (341–346) turned at random; also at step 0 (123–125) and 128; as a random basis's at 2000 (126–131), below it at 4000 (104–113, corrected at the close: first written as spread). By 8000 it is 6–24, by 16000 2–3 (ratio 3–7): the massive coordinates arriving (unit 1's R0b); at 143000 2–16, except seed 4 (133; its carriers are gone by then, unit 1's Parked) |

**What it changes** (the design's table, fixed before output; worded after `/challenge-pr` on
#265, findings 1–3): "not basis-aligned". §4.1 in C2p says the falls V counts at the break are no
more concentrated in the model's coordinates than in a random basis (9 of 9 seeds; post hoc,
slightly less), and the mean whose falls they are is spread over the coordinates there (half of
‖m‖² in 132–135 of 1024, a little more than a random basis's 127–129); with unit 1, the break is
carried by no privileged coordinates: neither the massive-activation ones nor any recurring set.
It does not say "not in a few coordinates": a few hold half of each net fall in any basis. The
lean ("not basis-aligned") held.

**Not established.** Anything about a direction that is not a coordinate (a rotation-invariant
question, which the centred count of C2 unit 3 already reads); a causal statement (a read-out of
m, not an intervention); other β, rungs or prompts. **The gate's reach** (finding 4): its plant
was a fall in 3 coordinates at step 7000; a privileged set of 8–16 coordinates carrying part of
each fall, and the plant on the scored steps' falls, were not tested, so the negative result
covers concentration of the planted kind, read at the break (the post hoc lean toward spread
argues against a hidden set but is not a test of one). The gate's trained population (7000, off the
grid) reads aligned at 0.95: whether alignment begins between 1000 and 8000, and where, is not
read (the family is 128–1000; the level's numbers at 4000 and 8000 are beside).

**Parked** (discoveries; not followed):
- When m becomes basis-aligned: the level reads below a random basis at 4000 (104–113 against
  127–129; first written "spread", corrected at the close) and concentrated at 8000 (6–24). Why: it would date the massive coordinates' takeover of the mean with a statistic that
  needs no 100× threshold. Cost: a read of these records, no forward pass. Decision it could
  change: how C2p words unit 1's R0b date (16000 at 100×). Carry or drop at the close-out.

**Re-run.** `METS_DATA=<main>/data`; `c5_mean gate --out <dir>` (`HF_HOME=<main>/data/hf`);
`c5_mean pass --out <dir> --seeds 0` (the same `HF_HOME`) and `--seeds 1 … 9` with
`HF_HOME=<HDD_1TB>/mets_data/hf`; `c5_mean read --out <dir>`. Conda `mets` env.

## C5 unit 1, M2: the massive-activation coordinates projected out (2026-10-23)

Blocked 104 decided (a), the user's. Design frozen and pushed before pass 1 (`1b4e2c6`: the
unit-1 table in `design-C5.md`), τ_p and τ_M frozen by the preflight and pushed before any pass-2
record (`6a38770`). Code `7e45858c218d` (`c5_massive.py`); every record under one hash. Tier 1,
description; nothing registered.

**Input.** `pythia-410m` seed 0 (the development set) and `-seed1`…`-seed9`, the 18 grid steps, the
7 v1 prompts (prompts hash as A5's; tokens' hash checked against A5's and unit 3s's records),
forward float32 on the RTX 3080, TF32 off, energies float64. Known answers met: S1's top-3
ratios at seed 0's 143000 equal to `massive.json` (deviation 0); the unprojected e1 equal to
A5's records (≤ 1e-4 relative; seed 0's first record 0) and its V equal to A5's at every step on
every seed; the unprojected e1c equal to unit 3s's (unit 3's for seed 0) to 8e-12; the float64
unprojected count equal to A5's float32 V at all three P1 arms on every seed (precision check).
Records `data/trackC/c5/2026-10-23-u1/` (main tree): `seed{0..9}/pass1/` (18 each),
`set.json`, `records/` (126 each), `preflight/`, `reading.json`. Wall time 71 min (runner).

**The sets (pass 1, written before any projected energy).** No seed refused: |S| = 2 or 3 on
every seed, each seed's own coordinates (seed 0: 130, 357, 966, S1's). M is the next 2–3 by
largest r (26–100×).

**Read first: nothing is massive at the break** (`/challenge-pr` on #264, finding 1; producer
`reading.json` `max_ratio_all_by_step`, from pass 1). At 128–1000 the largest ratio over *every*
coordinate at any state 0–23 is 1.6–2.4× the median coordinate's on every seed (6.8× at 2000 on
seed 0, 3.9–10.5× across seeds). So the projection at the break removes 2–3 ordinary
coordinates: their share of ‖m‖² and of its falls is at the random sets' level, the energy's
scale does not change (s_p / s_u 0.98–1.02), and every control keeps its dates too. P1 and P2
"kept" are what that predicts. They read as a check that nothing large enough to matter is
removed, not as a test that could have moved; the informative reading is R0b, when these
coordinates become massive.

| reading | result |
|---|---|
| **R0′, the set's share of ‖m‖²'s falls** (reported first) | at every violating transition at 128–1000 (1–2 per seed at 128 on six seeds, none on four; 14–58 per seed and step at 256–1000), the median signed share Δ‖m_S‖² / Δ‖m‖² is between −0.008 and 0.016 at 256–1000 (−0.048 to 0.061 on 128's few), and the fraction above ½ is **0 on all 10 seeds at every step**. M's share is as small (−0.009 to 0.027 at 256–1000). The falls V counts at the break are not in these coordinates, which at that step are not massive (above) |
| R0, the level | q = ‖m_S‖² / ‖m‖² at its largest state: 0.003–0.007 at 512 and 0.004–0.010 at 1000 on every seed, against 0.002–0.004 for the random sets' mean (at chance); 0.016–0.048 at 4000; 0.19–0.73 at 16000; at 143000 0.12–0.61, except seed 4 (0.006) |
| R0b, the coordinates' own date | **16000 on 9 of 10 seeds, 32000 on seed 6.** At 512 S's largest ratio is 1.1–1.5× the median coordinate's (not massive); at 1000, 1.3–2.3×; it grows from 2000 (4–10×) through 4000 (19–35×) and 8000 (50–88×) |
| **P1, the break projected** | **kept on 7 of 9 seeds: "kept"** (≥ 7, none to spare). Seeds 2, 3, 5–9 keep their date at all four arms (1e-4, 1e-3, 1e-2, τ_p 1.15e-3). Seed 1 reads mixed: at 1e-3 the projected date is 256 against 512 (earlier; at 256 the count is 23 against 22, the half line 22.5); seed 4 reads mixed: at 1e-2 1000 against 512 (the plateau 25 against 23, the count at 512 12 against 13). Both are one violation across the half line at one arm; the random sets cross it there too (`reading.json` `random_by_arm_post_hoc`, every random set dated at every arm, added after `/challenge-pr` on #264: 2 of 20 earlier on seed 1 at 1e-3; 1 of 20 later on seed 4 at 1e-2; seed 0's 2 of 20 earlier at 1e-4; no other random set moves at any P1 or P2 arm on any seed). Seed 0 (development): kept |
| **P2, the centred event projected** | **kept on 9 of 9: "kept"**, every arm, against unit 3s's dates (1000 at every arm, except seed 6's 512 at 3e-4, kept). Seed 0: kept |
| controls | M keeps its dates at every arm of P1 and P2 on every seed; at the frozen random arms (P1 1e-3, P2 7.58e-4) 20 of 20 random sets keep theirs on every seed but seed 1's P1 (18). "Specific" is moot: nothing moved |
| scale | s_p / s_u, the projected uncentred energy's own scale over the unprojected one's: 0.98–1.02 at 256–1000 on every seed (the projection leaves the break's relative drops their size, so the scale arm is a check, not a correction there); 1.09–2.71 at 16000, where the coordinates are large |

**What it changes** (the design's table, fixed before output, with the order changed after
`/challenge-pr` on #264, finding 1). P1 kept and P2 kept, but C2p leads with the date: §4.1 says
the coordinates that end training massive are ordinary at the break (no coordinate of any seed
above 2.4× the median's at 128–1000) and reach 100× only at 16000, long after both events, so
the mean whose falls V counts at the break is not theirs; projecting them out leaves the break's
date on 7 of 9 seeds (the other two move by one grid step at one arm, at the half line, as random
sets also do) and the centred event's on 9 of 9, as expected when nothing large is removed.
Whether ‖m‖² sits in a few *other* coordinates is not tested (Blocked 105).

**The lean was wrong.** Stated before output from 2510.06477 ("around step 1k"): R0b reads 16000
(at 100×). The paper reads σ₁²/‖X‖²_F, dominated by the first token's row; this reads each
coordinate's max over its median, so the two dates are of different statistics, and nothing here
tests the paper's.

**Not established.** That the model does not need the coordinates at the break (projection is a
read-out, not an intervention); anything at other β or on other rungs; whether the falls V counts
sit in some other small set of coordinates (R0′ reads only S and M; the referee's M2 also named
"outlier dimensions", a set chosen by the mean, which the design did not consider: Blocked 105).
The code hash (`code_sha`) covers the C5, C2 and A5 modules, not `core/metrics.py` or
`core/models.py` (finding 5): the known answers (e1 equal to A5's, e1c to unit 3s's) pin their
behaviour on these records, and the hash is not changed after the run.

**Parked** (discoveries; not followed):
- The carriers change late on 2 of 10 seeds: seed 3's set at 143000 (409, 524) is not its
  16000–54000 one (1019); seed 4's coordinates are at 100× at 16000–54000 and gone at 143000
  (its S's largest ratio 3×, q 0.006). Why: 2610.00423 says the carriers stabilise early, and
  S3 or C2p might take an end-state set for granted. Cost: none (in pass 1's records). Decision it
  could change: how C2p describes the set; S3's choice of coordinates. Carry or drop at the close-out.
- Unit 1's first pass 1 (`2026-10-23-u1-v0`) is kept on disk only as the defect's evidence
  (`LESSONS.md` 3); drop at the close-out.

**Re-run.** `METS_DATA=<main>/data`; `c5_massive pass1 --out <dir> --seeds 0` with
`HF_HOME=<main>/data/hf` and `--seeds 1 … 9` with `HF_HOME=<HDD_1TB>/mets_data/hf`; `select`;
`preflight` (refuses unless it re-derives τ_p, τ_M); `pass2` (as pass1); `read`. Conda `mets`
env (the repo `.venv` has no CUDA torch).

## C5 unit 0, M2's scan and design (2026-10-23)

Blocked 103 decided (a), the user's, with "do not post anything to arXiv": C5 opens on
M2 (2 units + close-out), then C2p prepares the draft and the user posts. Writing only; nothing
ran. Inputs read: `referee-C2.md` M2 and its answers, S1's `massive.json` (seed 0 at 143000:
357, 130, 966 at 100–350× the median coordinate's max at states 6–22, at position 0 and in the
maximum over the rest; 278, 335, 914 at ≤ 21× at states 1–4), `design-C2.md` "Unit 3s",
`c2_aniso.py` and `c2_seeds.py` (unit 1 reuses their passes, plants and arms).

**The scan** (`lit-C5.md`; five searches, three abstracts, two HTML texts in part). Not refuted:
no paper projects massive coordinates out and re-reads an energy or monotonicity count along
training. Closest: **2510.06477** reads σ₁²/‖X‖²_F (uncentred, kin to ‖m‖²), the BOS norm and the
sink on Pythia-410m's checkpoints, "all three emerge together around step 1k", the transition at
layer 5. Its consequences: (1) a lean for M2, stated before any output: the coordinates may be
born at C2's centred event (1000), after the break, so P1 would read "kept" nearly by
construction and P2 is the reading that can move; (2) C2's §8 novelty sentence and F9's
"published anisotropy along Pythia from step 10 000" are out of date (Parked 1, carried to C2p;
`LESSONS.md` 7). 2610.00423 (the carriers differ by seed, stabilise early in a run) makes the
set per seed; 2508.03616 (Sun's 1000× rule relaxed for small models) and 410m's end-state 140–350×
put the threshold at 100×; 2607.09279's random-dimension control is M2's.

**The design** (`design-C5.md`, proposed; revised after `/challenge-pr` on #263, accept with
changes, findings 1–5 taken, the budget point answered): per seed, S = the coordinates at ≥ 100×
at any of states 0–23 at any grid step (pass 1, written before any projected energy); each
token's S-coordinates zeroed, then normalised; R0, the set's share of ‖m‖² (level); R0′, its
signed share of ‖m‖²'s falls at each violating transition (what V counts); R0b, the coordinates'
own first step at 100×; P1, the projected uncentred half-rise at 1e-4, 1e-3, 1e-2 and a
scale-matched τ_p, each against the unprojected date at the same arm; P2, the projected centred
half-rise at unit 3s's four arms (re-gated), each against its own unprojected date; "moved" is
specific only if a magnitude-matched set (the next |S| coordinates by ratio) and ≥ 18 of 20
random sets keep their dates; tallies of 9 at ≥ 7. What each outcome changes in §4.1 is fixed. **Blocked 104, the user's:** (a) as
designed, recommended; (b) seed 0 only with S1's six; (c) a per-step set; (d) no M2.

## C2 on one page (closed 2026-10-23)

**Asked** (`design-C2.md` "What C2 is for"): one paper on E_β (β = 1) read layer by layer along
`pythia-410m`'s training: the break in its layer-monotonicity dated, reproduced on the 9
PolyPythias seeds and laid on the developmental order, complete enough for the user to decide
whether and where to post it. **Exit met** (budget extended on the record to 9 units + close-out
by Blocked 102 (a); #252–#261 and this): every section of the outline written; every number
rendered from a record by one command (`numbers.py`: 23 records, 241 keys, 57 sentence claims
checked in `produce` and CI) and every figure (1–5) drawn by it; each fence F1–F9 beside the
line it limits; a fresh-context referee read answered point by point (`referee-C2.md`). Writing
only here. The draft: `trackC_writeup/draft-C2.md` (rendered from `draft-C2.src.md`).

**Established** (exploratory except the registered seed count; 410m; v1's 7 prompts; numbers in
the unit sections below and `numbers.json`):

1. **The break is dated and is the recipe's.** V = 0 through step 128 on seed 0, half its plateau
   at 512; `P-A5`, registered before the seeds' intermediate steps were downloaded, put each
   seed's break at 256 or 512 and held on 9 of 9 (adjudicated by A5; F7 beside it).
2. **A second event, after the break.** With each prompt's mean token removed, the count
   breaks at 1000 on seed 0 and on 8 of 9 seeds (one grid step after a break at 512, two after
   seeds 3 and 9's at 256), the step at which the previous-token and induction heads are dated; at 512 the uncentred break is mostly the common direction's (unit
   3; seeds, unit 3s). Dates only: no size comparison between the two counts is calibrated.
3. **V at β = 1 nearly counts the layers where the tokens' mean norm falls.** E₁ = ½ + ½‖m‖² + R
   exactly (R ≥ 0, sympy); E₁ and ‖m‖² change in the same direction at 0.96–1.00 of transitions
   at every step, and the mean's term is more than half of all but 1 of the 1224 drops at
   128–1000 (unit 5b). An exact split, not a separation; the referee's sharpest point (M1), so
   §8 says the statistic is not new in kind.
4. **The dates are robust to how V is read:** neither the frame-changing transitions, a later
   plateau (one seed moves one step inside the window) nor per-prompt counts (65 of 70 series in
   the window) move a date (unit 5b).
5. **The schedule is not separated (F1 stands, Blocked 101 (a)).** On 70m and 160m (higher peak
   LR, the same warmup) E₁ breaks at 256 and the centred count at 1000: consistent with step and
   cumulative LR, an LR level excluded (3b). On Mistral GPT-2 small (a logarithmic warmup) E₁'s
   larger rise falls at step's band, a first stage at the dose's; the frozen "consistent with
   step" turns on the plateau window and one re-init unit; no centred date (3c).
6. **Carried in and fenced:** A3's order (L3, with F2: "break, then induction, then the OV phase"
   not shown) and S2's negative (L6, with F3: the break neither shown OV's nor shown not OV's).

**Not established:** anything that separates the dates from the LR warmup (F1; the run is C4);
that the centred break is a separate event rather than the same change seen more weakly (no
step between 512 and 1000; units 3 and 3s); why ‖m‖² falls mid-stack (massive activations, sinks; C5's projection); whether R breaks on its
own (no tolerance on R's scale is calibrated); how much of the break each frame carries; E_β at
other β; any causal statement about components (F6); a prompt set not chosen by hand; any rung
but 410m for the registered claim; anything between 512 and 1000.

**Units at close:**

| unit | PR | outcome |
|---|---|---|
| 0. design | #252 | frozen; 91–94 as recommended (left to Claude) |
| 1. reading | #253 | F5 worded, F9 narrowed, the GPT-2 small suite found (3c) |
| 2. `numbers.py`, §3–§5, figures 1–3 | #254 | every number from a record, checked in CI |
| 3. anisotropy control (seed 0) | #255 | "mostly removed" as frozen, a scale effect; centred date 1000 (Blocked 96) |
| 3s. the centred count on the seeds | #256 | K1 = K2 = 8 of 9; the draft heads on two events (98 (c)) |
| 3b. LR-dose arm (70m, 160m) | #257 | consistent with step and cumulative; F1 stands |
| 3c. warmup arm (GPT-2 small) | #258 | "consistent with step", fragile; no centred date (101 (a)) |
| 4. the rest of the draft | #259 | every section written; references from arXiv |
| 5. referee read | #260 | major revision, 9 major and 27 minor, every one answered (Blocked 102) |
| 5b. readings from records | #261 | V nearly counts the mean's falls; no date moves |
| close-out | this | writing only |

**Parked and open at close:**

| item | fate |
|---|---|
| 3c's Parked 0 (a refusal in `c2_seeds.null_matched_tol` when the null's uncentred rate is 0) | carried to C4 (ROADMAP §8): it reads the centred count, so it derives a centred arm and adds the refusal before its freeze; C5 likewise if it runs first |
| 3b's Parked 1 (PolyPythias' 70m and 160m seeds) | carried to C5, already there as the referee's M7 |
| `lit-C2-unit1.md` Parked 1 (Mistral GPT-2 medium's checkpoints, by email) | dropped: it removes only 3c's shape confound in an arm that is description either way, and C4 varies the schedule inside one model, which is the test F1 needs |
| unit 5b's R6 (c), whether R breaks on its own | carried to C5 beside M1 and M2: it needs a tolerance gated on R's own scale, and M2's projection changes R |
| the referee's M1 (other β), M2, M6, M7, M8's sampled prompts; M3 | carried to C5 and C4 (Blocked 102 (b), (c)) |
| `lit-C2-unit1.md` Parked 2 (token-drift's per-step table, an unreviewed 70m anisotropy series; sent to unit 3b, never picked up) | dropped: units 3 and 3s measured the mean cosine on 410m itself at every step, and the draft does not cite the project; it could change only §7's words, no rule |
| the referee's M5 remainder (A3's other rows on the 9 seeds; a null for the aligned-pair count) | the rows carried to C5 (a run: no record holds them on the seeds); the null dropped as a statistic: §5 already says no null gives the aligned count's chance level, so it is a stated limit |
| the referee's m15 (the CRFM blog, "at the close-out's LaTeX pass"), m19 (70m's and 160m's layer counts as keys, "at the close-out"), m25 (appendix A's ρ figure, "at the close-out") | carried to C2p, the draft's posting thread (ROADMAP §8, Blocked 103): each is an edit to the draft or `numbers.py`, and the close-out is writing only. *The referee answers promised them here; that promise was not kept at the close-out (`/challenge-pr` on #262)* |
| `numbers.py`'s module docstring names figures 1–3 (5 exist) | carried to C2p with m19: the file's own hash is in `numbers.json`, so the fix re-runs `produce` once |
| Blocked 93's remainder: where to post, a LessWrong post, the records' release, the LaTeX conversion | Blocked 103, the user's (below). *93 (a) put LaTeX at this close-out; it is held for 103, since a LessWrong post needs none and the venue decides the format* |

**Blocked 103 (yours): what happens to the draft.** After the close every edit to the draft
belongs to **C2p** (ROADMAP §8): m15, m19, m25 and the docstring above, the LaTeX, and **a second
fresh-context read of the whole draft**, which every option includes, because the first read
predates unit 5b's rewrite of the abstract, §1, §3.2 and §4.1 (`/challenge-pr` on #262).
(a) **C5's M2 first, then arXiv:** C5 opens (design, scan, the coordinates frozen) and runs M2
(the massive-activation coordinates projected out, V and the centred count recounted; cached
checkpoints), about 2–3 units; then C2p revises the draft, re-reads it and posts a preprint
(cs.LG) with the quoted records, about 2 units: **4–5 units in all**; C4 later, its own paper or
a v2. (b) **C2p now:** the carried edits, the second read, LaTeX and arXiv, about 2 units; M2
and C4 as a v2. (c) Hold the draft for C4, the schedule intervention (a new training harness,
GPU-days; weeks), and post with F1 answered. (d) A LessWrong / Alignment Forum post first
(updating Blog 1; C2p without the LaTeX), arXiv later. **Recommendation: (a), with (b) close
behind.** Unit 5b made ‖m‖² the paper's centre, and which coordinates carry it is the first
question a reader of §4.1 asks; at this thread's pace 4–5 units fit well inside the
mid-December target. (b) is the right call if a dated preprint soon matters more than a §4.1 the
next run could rewrite in public; (c) trades weeks for the one fence a reader will accept as
stated; (d) is cheap and can follow any of them. Posting is public, so it is yours whichever is
taken.

**Re-run** (the draft from the tracked table, any checkout): `python -m trackC_writeup.numbers
render`, `figures`, `check`; from the records: `METS_DATA=<main>/data python -m
trackC_writeup.numbers produce` first.

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
| R6 (a) | ‖m‖² at the drops | falls at every drop at 128, 256, 512, 1000 (1224 drops); later steps ≥ 0.98 of drops. *(The frozen comparison, ‖m‖² falling at 0.20–0.35 of all transitions, is the wrong base rate: most transitions are rises of E₁. Withdrawn; R6′.)* |
| R6′ (post hoc, `/challenge-pr` on #261 finding 1) | the base rate over every move | E₁ and ‖m‖² change in the same direction at 0.96–1.00 of transitions at every step; at any fall of E₁, ‖m‖² falls at 0.96–1.00; on rises, the mean's term is > ½ at 0.30 and 0.43 of them at steps 16 and 32, ≥ 0.80 from 128 on |
| R6 (b) | s = ½Δ‖m‖² / ΔE₁ | > ½ at all but 1 of the 1224 drops at 128–1000 (median 0.81–1.02 by step at 256–1000; above 1 means R rose); later 0.91–0.99 of drops |
| R6 (c) | V_R, drops of R alone | half-rise at 512 on all 10 seeds; on seeds 3 and 9 (V's break at 256) V_R is 2 and 1 there. **Not quoted in the draft** (finding 3): the tolerance is 10⁻³ of E₁ while R is 3–5 % of E₁ and shrinks along training, so on R's own scale it tightens with the step; no tolerance on R's scale is calibrated |
| R7 | E₁ by depth at 128, 512, 1000, 143000 | figure 5, right (no number) |
| R8 | figure 4's right panel | normalised per seed, half line, centred per arm (seed median), LR / peak |

**What it changes in the draft.** §3.2: the frame transitions and per-prompt sentences are
numbers; §3.3: the later plateau is read; §4.1: "how much of V it carries is not measured"
replaced by R6 and R6′; the abstract and §1 item 2 say E₁ and ‖m‖² move together at ≥ 0.96 of
transitions at every step, so V nearly counts the mean's falls; a new appendix C with figure 5;
figure 4's caption. 23 records, 241 keys, 57 claims (8 new; first written 56, before R6′'s claim, corrected at the close-out). **No date already read moved.**

**What it means.** The referee's M1 was right, more strongly than the draft first said: at β = 1
E₁ moves with ‖m‖² at 0.96–1.00 of transitions at *every* step, so V is close to a count of the
layers at which the tokens' mean norm falls, along the whole training axis. The break is when
those falls start, mid-stack; the drops at the break are no exception to the rule, not a special
property of the break (first written as one, against the wrong base rate; `/challenge-pr` on
#261). The split is exact but not a separation: R's higher terms carry the mean too (R is not
mean-free; the centred count nearly is). With unit 3s's centred date (1000): the mean's norm starts to
fall mid-stack at 256–512; the tokens' arrangement around the mean starts to drop one grid step
later.

**Not established.** Why ‖m‖² falls mid-stack (massive activations, sinks: C5's projection is the
run that could say); whether R has a break of its own (no tolerance on R's scale is calibrated);
anything on a prompt set not chosen by hand.

**After `/challenge-pr` on #261** (accept with changes; reviewed `9594b02`; all four taken): (1)
R6's base rate replaced by R6′, added to `c2_readings.py` after R6's output and marked post hoc
in the record (R1–R5 and R6's frozen fields unchanged on the re-run); the abstract, §1 item 2,
§4.1, `PROJECT.md` §3.70 and the claim reworded. (2) The abstract's "most of" every drop is now
the same-direction share; "all but one" stays in §1 and §4.1. (3) V_R's date taken out of the
draft (above). (4) `readings.json` now records each seed's input sha256 and the code hash covers
`a1a.py` and `a5_seeds.py`.

**CodeRabbit on #261** (4 findings, each checked): `cos` is from a float64 normalisation of the
same states and `e1` from a float32 one (both plain L2), so the split mixes precisions. **Not
regenerated** (that is every forward pass again): 23 of the 30 240 transitions have a change below
10⁻⁵ of E₁ in either term, and the smallest mean's-term change at any drop is 6.8 × 10⁻⁶ of E₁,
against a float32 normalisation error near 10⁻⁷, so no share can move by more than 0.001. The
abstract's exception (already fixed after `/challenge-pr`), "the centred cloud is mean-free"
(now "nearly"), and §1's and §8's first-order coefficient (now ½‖m‖²): fixed.

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

- 2026-11-15 (training-symmetry §8.4; rewritten after `/challenge-pr` on #354, finding 1): `c6_schedule.lr`, and its restatements in `c4_train.lr_pythia`, `c6_warmup.py` and `tools/math_checks/c6_bands.py`, decay by a cosine *to* `min_lr` over T − W, which is today's NeoX; the released weights' zero-gradient embedding rows read the schedule these runs trained under as max(`min_lr`, peak/2 · (1 + cos(π(i − W)/T))) at λ = 0.0995–0.1001 on every checkpoint from 512 (`pythia-70m`, `pythia-410m`) and 0.0998–0.1007 on `pythia-70m-warmup01` (W 14 300). The two forms agree in the summed rate within 0.01 % to 8000 and 0.11 % to 20 000, and differ by 1.4 % at 64 000 and 6.2 % at 143 000. **What moves:** C6 U0's schedule check reads intervals to 143 000, and its H0 statistic goes from 0.156 to 0.197 against the threshold 0.223, so H0 is still confirmed, with a third of its margin (`python tools/ts_weights.py c6-h0`). **What does not:** C6's, C4's and C4′'s bands and labels, all read at steps ≤ 20 000, and C4's A0 arm to 4000 (the rate differs by under 0.01 % there). The code is left as frozen under its hashes. Pointer: `MATH_TRAINING_SYMMETRY.md` §8.4, `tools/ts_weights.py`.
- 2026-11-14 (C4 close-out): C4 U3 reported its exit met without the induction witness that `design-C4.md` "Readings and outcomes" names beside R1–R3 (F never read; A0 only to 1000, from U1). Run at the close-out: induction's half-rise is 768 on A0 and 320 on F, so on F it does not share a step with the centred date (48) as it does on Pythia's schedule. No label changes — "C4 on one page (closed 2026-11-14)", item 9.
- 2026-11-09 (C4 U1): C2's "null-matched" centred arms, 70m's 9.61e-05 (unit 3b's preflight) and 3c's GPT-2 arm, were derived at a null uncentred rate of exactly 0 (each preflight's `uncentred_at_1e-3.null_rate`), so each is the null's largest centred drop, not a τ matched to a non-zero rate. On 70m it counts as 1e-4 does (H6: equal at every step to 1000 on A0 and on the released run), so C6 U1b's and C4's "every gated arm" is in effect three arms. No label changes; the refusal is Parked (0) — "C4 unit 1"
- 2026-10-30 (C2p readings audit): C2 unit 1's F5 row (`lit-C2-unit1.md` §1) gave Pendharkar §10 as "the conditions are never exactly satisfied in trained transformers"; §10 says it of Condition 6 (orthogonal subspaces) only, and tying is Remark 5's idealisation. Its "no result covers … weights that change from layer to layer" misses 2604.01978 (‖m‖² logistic in depth under per-layer random attention weights). §8's Razzhigaev comparison read a centred measure as ‖m‖². No date or reading moves; `paper/` corrected — "C2p readings audit (2026-10-30)", `lit-C2p-fulltext.md` W1–W4.
- 2026-10-24 (C2p unit 2): C5's finding 4 ("the lean from 2510.06477, 'around 1k', did not hold") and `lit-C5.md`'s row read the paper's "around step 1k" as a date; its appendix B.1 reads Pythia at steps 1, 1k, 2k, 4k, … only, so it bounds its changes to (1, 1000] (both C2 dates inside), and its training figure reads entropy, sink rate and the BOS norm, not σ₁²/‖X‖²_F. The coordinates' 16000–32000 stands; it is not a date the paper's grid could contradict. "C2p unit 2" below; `LESSONS.md` 7.
