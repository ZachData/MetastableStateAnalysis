<!-- trackC_writeup/referee-C2p.md -->
# C2p unit 2: the second fresh-context read, and its answers (2026-10-24)

**How it was run.** A fresh general-purpose agent with no conversation history was given only a
copy of `trackC_writeup/paper/` as it stood before the answers below (main.tex, refs.bib, fig/,
and the pdfLaTeX build, 24 pages). It was told to review it as a TMLR-style referee, to open no
other file on the machine, and to check cited papers on the web if needed. It could not write its
report file (subagents may not), so it returned the report as text; it is kept below as it was
returned. The first read, on the Markdown draft before unit 5b, is `referee-C2.md`.

**Recommendation: major revision** (6 major, 20 minor). Its deciding reasons: (1) at β = 1, V is
in effect a count of the layers where the mean token's norm falls, and the framing obscures it;
(2) the registered test is scored against a null nobody holds, and the timeline's alignments have
no base rate; (3) internal vocabulary and undefined statistics.

## Answers

"Fixed" means changed in `paper/` in this unit (each edit is in `git diff --no-index tex/ paper/`).
"Yours (109)" means the point is a framing or venue choice that the user's rewrite decides
(STATE Blocked 109, with a recommendation). "C5" means it needs a reading or a run, which is not
this thread's (`design-C2p.md`: no new reading), so it is carried to ROADMAP C5's row.
**At the close-out** (Blocked 109 decided (a), the user's, 2026-10-24) Claude made the recommended
points in `paper/`. M1 (d), M2, M3 (i)–(ii), M6, m6, m13, m17 and m19 now read "fixed at the
close-out". Still the user's: m11 (anonymity, the timestamp narrative, "the author's notes"), m12
(the abstract's density beyond the reframing), M3 (iii), and a code and data statement.

| point | answer | what changed, or why not |
|---|---|---|
| M1 | fixed in part; (a), (b), (c), (e) C5; (d) fixed at the close-out (109 (a)) | The two sentences the referee set against each other are reconciled. §4.1 now says that at β = 1 every drop V counts at the break is a fall of the mean's norm, so the break at 512 is, descriptively, the tokens' anisotropy ceasing to rise with depth at some layers, and that what drives it is not read. The abstract, §1 item 2 and §4.1 are scoped to the direction that was read: at the break ‖m‖² falls at every drop V counts, and at every step it falls at ≥ 0.96 of E₁'s falls. The converse, how many of the mean's falls V misses, is stated as not read. V_m dated by the same rule, a magnitude statistic, a β sweep in this frame (C5's M1), and the centred count's second moment against E5: these are new readings, so C5. (d), at the close-out (revised after `/challenge-pr` on #269, findings 1–2): the title leads with the count ("Counting where a language model's token energy falls with depth: a break early in training, at falls of the tokens' mean"), and the abstract opens with the count and its β = 1 reading in the direction read (E₁'s falls are the mean's, the conditional share, not the same-direction one), the particle picture moved to one sentence of motivation and §1's second paragraph says so |
| M2 | fixed at the close-out (109 (a)); the timestamp narrative yours (m11) | The arithmetic is confirmed by the referee. §4 now leads with a per-seed table (V at 0, 128, 256, 512, 1000, the plateau, the break; checked against `a5.V` and `s0.V`), p and E are one sentence after it ("reported because they were registered, not as the strength of the result") and are out of the abstract. The timestamp narrative stays: it is m11's (anonymity, venue-dependent) |
| M3 | (i), (ii) fixed at the close-out (109 (a)); (iii) not changed; (iv) in part | (i): §5 states the base rate for two dates on four consecutive grid steps, and the higher one anchored at the break, which seven of the nine pairs read include (`tools/math_checks/c2p_alignment_base_rate.py` B1, B4; the anchor after `/challenge-pr` on #269, finding 4), and that seven of nine is about chance placement, so the alignments are not evidence of a link. (ii): the abstract drops "the step at which the previous-token and induction heads are dated" and says "one grid step later on seed 0"; §1 item 2 and §4.1 say the uncentred break is as close to the heads by §5's convention (§4.1 after #269's finding 3). (iii): how the shares at 512 are used was not in 109's recommendation; left to the rewrite. (iv): the E8 caption is fixed (m10); appendix D now says which locator each row uses |
| M4 | (i) fixed; (ii) C5; (iii) no change; (iv) C5 | (i): §7 now says how GPT-2 small's bands are computed (its rate p′ ln n / ln W′, the cumulative map inverting its integral, as `c2_warmup.py` does). (ii): reporting both normalisations is a new reading. (iii): §7 already states the frozen label, "consistent with step", as the result and the decision after it as discussion. (iv): the 70m/160m PolyPythias seeds are C5's M7, already on its row |
| M5 | (i), (ii) fixed; (iii) yours / C5 | (i): §6 says that no edit puts the model under the theorems' hypotheses, so neither outcome tests the theory. (ii): the abstract now says the step-128 OV map lowers V at 512 and 1000 beyond every random edit, to about half. (iii): a per-cell table can be built from S2's records, at the rewrite's choice. More seeds, per-layer substitution and a QK substitution are C5's M6 |
| M6 | fixed (109 (a)); the code and data statement yours | "3c's rule", G1–G3 and `tools.design_preflight` are removed from the text. At the close-out: §3.5 "Notation and terms" (x, m, E_β, drop, V, V_c, plateau, half-rise, break, grid step, K, fixed before the run, gate, refused, arm, description, locator), and appendix D defines each row of the timeline table (the report's table 1, table 2 in `paper/` once the seeds table precedes it; E11's β after #269's finding 5) (from `trackA_force/design-A3.md` and `a3.py`) with its bootstrap units. A code and data statement depends on the release, which is the user's (Blocked 103 (a)) |
| m1 | fixed | §2.1: monotone [§3.2], gradient ascent in a reweighted metric [§3.4]. Checked against `docs/readings/arXiv-2312.10794v5/survey.tex` (§3.2 "The interaction energy", §3.4 "(SA) is a gradient flow for a modified metric") |
| m2 | fixed | The minimum is 0.956 (key `rd.same_dir`), so "0.96 or more" rounded up. Now 0.956 (abstract, §1) and 0.956–1.000 (§4.1). The frozen Markdown `draft-C2.md` keeps "0.96" |
| m3 | fixed | §4.1's M2 paragraph says which counts are over the nine scored seeds and which are over all ten |
| m4 | fixed | §4.1: seed 6 has no centred date by the four-tolerance rule (three tolerances date it 1000, one 512, its own break; `c2u3s` reading) |
| m5 | fixed | §2.3 |
| m6 | fixed at the close-out | §3.1: the grid is the set earlier readings of this model used, which is why it holds 54 000 rather than 64 000 |
| m7 | in part | "3c's rule" fixed. The legend overlap and drawing the spread are figure 4's, left to your cull |
| m8 | fixed | The level figure draws the random-basis band's edges over the seeds, and its caption says the 132–135 against 127–129 gap is too small to see at log scale. The massive-set figure's caption explains the fall at the last step on seeds 3 and 4: their carriers change late, so the set is the union over the grid (C5's Parked late-carrier item) |
| m9 | fixed | Figure 6's caption |
| m10 | in part | The caption says E8 is drawn from step 0. Marking the −1.2 cut is a figure change, left to your cull |
| m11 | yours (109) | Anonymity is a venue choice |
| m12 | yours (109) | The abstract is the rewrite's |
| m13 | fixed at the close-out | Checked in `c2_aniso.py` (`plant_common`): the vector is added to each token's residual state before normalising, scaled by the median token norm. Appendix B now says so, and that centring does not remove it exactly |
| m14 | first half fixed; second declined | §2.1 states Σᵢⱼ⟨xᵢ,xⱼ⟩ᵏ = ‖Σᵢ xᵢ^{⊗k}‖² ≥ 0 (`tools/math_checks/c2_energy_moments.py` M4). The proposed bound R ≥ (e − 2)/(2n) does not hold: for odd k the off-diagonal terms can be negative, so a term can fall below its diagonal's n |
| m15 | fixed | §2.1: seed 0 has no drop at step 0, and five of the nine seeds have a few |
| m16 | fixed | Appendix C: the Camus passage is the only prompt not in English |
| m17 | fixed at the close-out (dropped) | "the float-noise margin seed 0's gate allows" is gone; the sentence reads "a count that moved by one drop" |
| m18 | in part | Godey et al. 2024 (2401.12143) is cited for what its abstract says: anisotropy across objectives and modalities. Its checkpoint reading, which the referee cites, was not read here, so it is not stated. Gao et al. 2019 and Puccetti et al. 2022 are not added (not read) |
| m19 | fixed at the close-out (stated) | Appendix A kept, with its figure (the cut puts `fig11_rho` in the appendix), and it now opens by saying it is context for §6 and that no claim rests on it; whether its OV readings predict the substitution is not read |
| m20 | C5 | The set-aside rules on the nine seeds are a new reading |

---

## The report, as returned

# Referee report: "When a language model's interaction energy stops rising with depth"

## 1. Summary

The paper takes the interaction energy E_β of the self-attention particle model (Geshkovski et al.), which is non-decreasing in depth under identity weights, as a reference prediction for a trained model. It measures E_1 layer by layer on seven hand-picked 512-token prompts at 18 checkpoints of `pythia-410m` and of its nine PolyPythias seeds, and counts layer-to-layer drops (V; relative tolerance 1e-3, at most 168). V is zero through step 128 and reaches half its 1000–4000 median at step 512 on seed 0. A window (128, 512], registered after seed 0 was read and before the other seeds' intermediate checkpoints were downloaded, contains the break on all nine seeds (seven at 512, two at 256). The paper shows that E_1's first-order term is ½‖m‖² (m the mean unit token) and that E_1's changes almost always follow ‖m‖²'s; removing the mean ("centred" count V_c) moves the break to step 1000, which is also where the previous-token and induction heads, and the learning-rate half-point, are dated. A table of ten other developmental quantities places the break among them; three OV-map edits on seed 0 do not bring V back to zero; and, because every dated event falls inside the 1430-step warmup, the paper reads `pythia-70m`/`160m` and Mistral GPT-2 small and finds that none separates training events from the schedule. The authors present the work as exploratory except for the registered seed count.

## 2. Recommendation

**Major revision.**

Deciding reasons:
1. **The central object is not what the title and framing suggest.** By the paper's own evidence (§4.1), V is in effect a count of depth transitions where the mean token's norm falls; the "break" is the onset of a non-monotone anisotropy profile across depth. The particle-picture framing and the hedge "it does not say the break is anisotropy" sit uneasily with the paper's own finding that ‖m‖² falls at every one of the 1224 drops between steps 128 and 1000 (M1).
2. **The confirmatory-looking evidence is non-discriminating.** The registered test is scored against a null the authors themselves say nobody holds, yet p = 9.9e-8 and E = 1587 are reported; the timeline "alignments" are on a log-2 grid where chance agreement is high and use a convention that the abstract then applies selectively (M2, M3).
3. **Exposition relies on internal project vocabulary and undefined statistics**, so several claims cannot be checked from the text as submitted (M6), and the schedule and intervention sections have specification gaps that affect their conclusions (M4, M5).

The measurements appear careful, the paper is unusually candid about what it does not show, and most of the stated limits are correct. A revision that recentres the claims on what V measures, and removes or properly calibrates the decorative statistics, could make this a sound TMLR paper.

## 3. Major issues

### M1. The break is, descriptively, a mean-norm (anisotropy) profile; the framing and β choice obscure this
*Where:* abstract (main.tex l.26); §2.1 eq. (2) (l.48–64); §4.1 "The mean's term by layer" and "How much of V the mean's term carries" (l.143–147); the sentence "It does not say the break is anisotropy" (l.135); Fig. 4; §8 "Anisotropy" (l.246).

*Problem.* At β = 1 on the unit sphere, exp⟨x_i, x_j⟩ is close to affine, so eq. (2) is dominated by ½‖m‖². The paper's own numbers confirm it: E_1 and ‖m‖² move in the same direction at 0.96–1.00 of transitions, ‖m‖² falls at all 1224 drops between 128 and 1000, and its term is more than half of each drop at all but one (median share 0.81–1.02). So the uncentred break is, at minimum, "the depth profile of ‖m‖² stops being monotone", a mid-depth anisotropy peak of the kind Razzhigaev et al. (2023) describe for decoders. The paper then declines to call it that ("It does not say the break is anisotropy", l.135) while also saying "V nearly counts the layers at which the mean's norm falls" (l.37); those need reconciling. The particle-picture vocabulary ("interaction energy", clustering, "the picture's prediction fails") and the title imply more physical content than a count of mean-norm falls carries.

Two further problems. (i) The abstract's step is incomplete. A same-direction share over all transitions, most of them rises, does not by itself show that V counts ‖m‖² falls. That needs P(‖m‖² falls | E_1 falls) and P(E_1 falls | ‖m‖² falls); the body gives only the first ("whenever E_1 falls … ‖m‖² falls at 0.96–1.00"). (ii) V counts sign events and ignores magnitude. Fig. 4 shows the drops at step 512 are about 0.01 each on a 0.5–0.83 scale, so a profile with one shallow interior bump spread over ten layers gives V ≈ 10 per prompt, the same as a deep collapse.

*Fix.* (a) Report V_m, the count of ‖m‖² falls, dated by the same rule, beside V for every seed. If it dates identically, say plainly that the uncentred break is a mean-norm event; if not, that is a finding. (b) Add a magnitude statistic (e.g. Σ max(0, −ΔE), or max_ℓ E_ℓ − E_final) dated by the same rule. (c) Read a β sweep (e.g. 1, 3.5, 10) in the same frame. The clustering and metastability literature invoked concerns the regime where higher-order terms matter, and E_3.5 is currently reported only in a different frame. (d) Retitle, or reword the abstract, so that what is measured (a depth-monotonicity count dominated by the mean token at β = 1) leads and the particle picture is motivation. (e) For the centred count, note that its leading term is the centred cloud's second moment, a spectral concentration measure, and discuss whether its 1000 date and E5's reversal "between 512 and 1000" (l.183) are the same event.

### M2. The registered prediction is scored against a straw null; the reported p and E overstate what was tested
*Where:* abstract; §1 item 1; §3.4 (l.104–106); §4 "The nine seeds" (l.110); App. B (l.276).

*Problem.* The arithmetic is right: the uniform null on 12 grid steps gives K ~ Bin(9, 1/6), and I verified P(K ≥ 4) = 0.048, (1/6)^9 = 9.9e-8 and E = 0.5·p^(−1/2) = 1587. But the authors say nobody holds this null and that 9 of 9 "was the expected outcome" given shared data, schedule and code. The window was chosen after seed 0 was read, with seed 0's date at its right edge. It spans two points of a log-2 grid, so the prediction is "the half-rise is not before 256 and not after 512". The step-0 and final checkpoints had already been read. In this setting a headline p of 1e-7 and an e-value of 1587 are decoration that readers skimming §4 will take as strong evidence, and the abstract's "it held on 9 of 9" invites the same. The registration-timing narrative (commit hash, author-set timestamps, a GitHub activity log "kept with the code", l.106) cannot be verified by a reviewer, and it identifies the authors at a double-blind venue.

*Fix.* Present §4 as a reproducibility result: a per-seed table of V at 128, 256, 512 and 1000, the plateau medians, and the half-rise dates. Drop p and E, or move them to a footnote with the caveat. If a test is wanted, pick an alternative someone might hold, e.g. "the seed-to-seed spread of the break date exceeds one grid step", or compare with the seed dispersion of the other rows in Table 1. Replace the timestamp narrative with an anonymised, third-party-timestamped preregistration (OSF, an anonymous archived link), or shorten it to one sentence.

### M3. The timeline (§5) and the abstract's coincidence with the induction heads lack discriminative power and are applied inconsistently
*Where:* abstract ("the step at which the previous-token and induction heads are dated"); §1 items 2–3; §4.1 (l.135); §5 "The order" and "Inside the 1000 tie" (l.192–200); Table 1; Fig. 5.

*Problem.* (i) Base rate. Most rows fall on {256, 512, 1000, 2000}. Two events placed independently and uniformly there are "within one grid step" with probability 10/16 = 0.625, so "seven of 9 aligned" is near chance. The paper concedes that "no null says how many pairs … would align" but still reports the count and lists alignments in §1. (ii) Selective convention. By the paper's own convention a one-step gap is aligned, so the uncentred break (512) is just as aligned with the induction and previous-token heads (1000) as the centred break is. Yet the abstract singles out the centred date of 1000 as "the step at which the previous-token and induction heads are dated", which reads as a link the convention does not support. E1c was also added after the other rows were read (Table 1 caption). (iii) The shares contradict the "tie" reading. At 512 the break is 0.98 done, induction 0.03 and previous-token 0.35 (l.200), so on this evidence the uncentred break precedes induction-head formation regardless of the grid. The paper sets that aside under a convention that treats one-step gaps as ties. Either the shares are informative and the break precedes induction, or they are not and the centred-count coincidence is not informative either; the paper uses both readings in different places. (iv) The dates are not commensurable across rows. E1 uses the 1000–4000 plateau median; the others use the largest Δ up to 4000 with a bootstrap floor; E10 reads 1000 by construction; E8 is dated twice (1000 from step 0, 512 from its peak) and Fig. 5 draws one. No row whose plateau or peak is at 1000 can be dated after 1000, which covers E3 and every series entering a plateau window that begins at 1000.

*Fix.* Remove the head-date coincidence from the abstract, or give a calibrated null for alignment on this grid (e.g. random sigmoids with log-uniform onsets passed through each row's own locator). Treat the ordering evidence from the shares consistently. Say in Fig. 5 which locator each row used.

### M4. §7's schedule comparison is under-specified, and its one external test was overridden after reading
*Where:* §7 (l.218–240), Fig. 7.

*Problem.* (i) The mapping f·t (f = 1, √(p/p′), p/p′) is derived "inside a linear warmup" of equal length. Mistral GPT-2's schedule is described as logarithmic, reaching half its peak by step 100, so these formulas do not apply, and the text does not say how the "cumulative" band 40–150 was computed (presumably by integrating the logged LR). I re-derived the Pythia bands; they are correct. (ii) The plateau window is mapped by tokens seen (4000–16 000) while the dates are mapped by update count. Mixing the two normalisations drives the outcome: on 410m's own window the half-rise reads 80 or 200 and the joint label becomes "neither", as the authors show. (iii) The frozen rule's label "consistent with step" was set aside after reading (l.234). The three reasons may be sound, but this is a post-hoc override and should be presented as such, with the frozen label as the result. (iv) On `pythia-70m`/`160m`, size and peak LR are confounded, there is one seed per size, and 70m's plateau is 6 drops (Fig. 7's ratio of 2.5 shows how fragile that is), yet the text says the centred date "excludes the rate itself". PolyPythias provides 70m and 160m seeds, and they were not read.

*Fix.* Specify the GPT-2 band computation. Fix one normalisation (tokens or updates) for both the plateau and the dates, or report both. State the frozen label as the result and the override as discussion. Read the PolyPythias 70m/160m seeds before claiming any exclusion. Shorten the abstract sentence on Mistral, which is hard to parse.

### M5. The OV interventions (§6) are weakly motivated by the theory, asymmetrically summarised, and not verifiable from the figure
*Where:* §6 (l.202–216), Fig. 6, App. B "The OV edits".

*Problem.* (i) Rationale. The monotonicity results cited in §2.1 require QᵀK symmetric and equal to V. Making OV symmetric, or removing its rotation, while QK, the MLPs and the LayerNorms are untouched, does not put the model under any of those hypotheses, so neither a positive nor a negative result bears on the theory. That should be said up front. (ii) Summary. The step-128 OV substitution lowers V from 56 to 29 at 512 (controls 50–66) and to 31 at 1000 (controls 38–58), beyond all 40 random edits. That is about half the break and the strongest causal signal in the paper, yet the abstract summarises it as "does not undo the break", which is literally true but undersells it. (iii) Verifiability. A cell's lower/higher label also depends on a gain-only control that is not drawn. In Fig. 6(a) the edited values at 1k–4k sit at the lower edge of the control ranges, so a reader cannot check "lower at none". Each cell is judged against 40 draws (about p = 1/41) across roughly 15 cells, with no multiplicity correction. It is seed 0 only, with every head in every layer edited at once.

*Fix.* A per-cell table (edited, unedited, control range, gain-only value). Repeat the substitution on two or three seeds. A per-layer (or layer-band) substitution to localise the halving. Consider substituting QK at step 128 too, the other half of the theory's hypothesis.

### M6. Definitions, vocabulary, and reproducibility
*Where:* throughout; especially §5/Table 1, App. A–B, Fig. 7 caption, abstract.

*Problem.* Much of the paper is in an internal register. Undefined terms: "gate", "arm", "refused", "description", "read after the dates", "P-A5", "3c's rule" (Fig. 7 caption, l.227; no section 3c exists), "G1/G2/G3" (App. B), `tools.design_preflight`, and "the label the rule fixed before that run gives turns on two of its choices" (abstract). Undefined statistics: E5 (effective rank of which state, computed how?); E2 ("squared eigenvalue mass on eigenvalues with positive real part" of a rank-64 1024×1024 map: the nonzero eigenvalues only?); E8 (the α-fit procedure, and the layers and matrix types included); E3/E4 (positions, prompts, scoring rule); E6u/E6b (how the n-gram predictors are estimated, on what text); the locator's bootstrap units; and PR and PR_0 in App. A (PR_0 is called an "exact expected value", but PR is a ratio of traces, so the expectation needs a derivation or a citation). There is no code or data availability statement.

*Fix.* A short notation and glossary paragraph in §3; a one-line formula for every Table 1 row (in an appendix table or in the table); descriptive phrases in place of project labels; an anonymised code link.

## 4. Minor issues

- **m1.** §2.1 (l.50): the statement that the normalised dynamics are a "gradient ascent in a metric that reweights the tokens" is Geshkovski et al. (2023) §3.4.1 (particle case; §3.4 says (SA) is a gradient flow for a modified metric), while §3.2 shows monotonicity through the continuity equation (identity (3.6)). Cite accordingly. Those results are in continuous time, and the discrete-layer caveat should sit next to Fig. 1.
- **m2.** "0.96 or more" (abstract, l.26/37): Fig. 4 (right) shows about 0.956 at step 256. Give three digits or write "≥ 0.95".
- **m3.** Inconsistent denominators: e.g. "on 7 of the 9 seeds" and "not basis-aligned on 9 of the 9 seeds" (§4.1, l.153) against "the ten seeds" elsewhere. Say each time whether seed 0 is included.
- **m4.** Seed 6's centred date under the four-tolerance rule is never stated. §1 says "eight of the nine seeds at every tolerance read", while §4.1 says one tolerance dates seed 6 "at the seed's own break". Does seed 6 then have no centred date?
- **m5.** §2.3 (l.80): "the same warmup (1430 steps for 160m, 1430 for 70m)" is redundant and reads like a leftover; say "both warm up over 1430 steps".
- **m6.** The checkpoint grid includes 54 000 rather than a power-of-two-like 64 000; say why.
- **m7.** Fig. 7 caption: "left"/"right" refer both to panels and to bands within the right panel; "3c's rule" is undefined; the legend overlaps the "step" band label; a median of four tolerances across seeds is plotted while the text says they disagree, so draw the spread instead.
- **m8.** Fig. 11: the random-basis grey range is nearly invisible, and the §4.1 claim (132–135 vs 127–129) depends on it. Fig. 10: one seed's maximum falls from about 200× to about 3× at step 143k, and another drops markedly; this is unexplained. Did the massive coordinate move?
- **m9.** The Fig. 6 caption says the gain-only control "is not drawn", yet cell labels depend on it (M5).
- **m10.** Fig. 5: E8's dot is drawn at 1000, while Table 1 also gives 512 "from its peak". The caption says the axes are "cut at −1.2", but the plotted ranges and ticks make that hard to see; mark the cut.
- **m11.** Anonymity: "The author's informal notes" (l.254), "earlier work of ours", "earlier work on this model", the commit hash and the GitHub branch-log timing all identify the authors at a double-blind venue.
- **m12.** The abstract is very dense, with about eight results and internal qualifiers. Cut it to the three findings that survive M1–M3, and state the β and the frame.
- **m13.** App. B, the planted "common vector": the centred count "must not see" it, but centring happens after normalisation to the sphere, so a vector added before normalisation is not removed exactly. State where it is added.
- **m14.** Eq. (2): give the one-line reason each higher term is non-negative (Σ_ij⟨x_i,x_j⟩^k = ‖Σ_i x_i^{⊗k}‖²). Note also that the diagonal contributes a constant (e − 2)/(2n) to R, so R is bounded below by a known constant, not only by 0.
- **m15.** §2.1 (l.58) calls the model "(almost) monotone" at initialisation, but §4 shows five of ten seeds with 1–6 drops at step 0 (up to 8 at step 4). Say the baseline is seed-dependent wherever the identity-weight prediction is invoked.
- **m16.** App. C, by prompt: the only five off-window prompt series are all the Camus passage, the only non-English prompt. Comment; it bears on how general the seven-prompt result is.
- **m17.** §4, "a count that moved by one, the float-noise margin seed 0's gate allows": App. B says V was reproduced "exactly". Clarify which margin is meant.
- **m18.** Missing related work: Godey et al. (2024), "Anisotropy is inherent to self-attention in transformers" (arXiv:2401.12143), which tracks query/key anisotropy along training checkpoints (MultiBERT) and reports that the common-direction dynamics appear in the first few steps, directly relevant to M1; Gao et al. (2019) on representation degeneration; work linking outlier dimensions to token frequency during training (e.g. Puccetti et al., 2022).
- **m19.** App. A is only loosely tied to the paper's claims and adds a further unexplained statistic. Connect it (e.g. does ρ for OV write predict the substitution result in §6?) or remove it.
- **m20.** §3.3: two alternative dating rules were "set aside on seed 0" (first V > 0; a change-weighted centroid ≈ 827). Report what they give on the nine seeds as a robustness row, since they were rejected before the seeds were read.

**Characterisations checked on arXiv:** Geshkovski et al. 2023 (2312.10794): Prop. 3.4, §3.2, §3.4.1, accurate apart from the section pointer in m1. Belrose et al. 2024 (2402.04362): trough at 2^6–2^8 and rise to about 2^10, accurate. Queipo-de Llano et al. 2025 (2510.06477): Pythia 410M/6.9B/12B at steps 1, 1k, 2k, 4k, …, one GSM8K prompt, "emerge together around step 1k", accurate. Hoogland et al. 2024 (2402.02364): stage order LM1–LM5, accurate. Godey et al. 2024 (2401.12143): for m18.

**Numbers checked:** the binomial, p, E and 0.048; the §7 Pythia bands (step, cumulative, instantaneous); 1/7 = 0.14; the 18-step grid; the 11 timeline pairs (7 aligned, 2 not, 2 unread). These cross-section repetitions are consistent: V = 19/56/57/70 on seed 0; the 0.19–0.24 and 0.98 shares; 1224; 29/31/56 under the substitution; the ρ values against Fig. 8.

## 5. What the paper does well

- Honest scoping. Exploratory status, seven prompts, one size, the grid gap 512–1000, everything inside the warmup, and the uniform null being one nobody holds are all stated plainly and mostly correctly.
- A correct analysis of eq. (2): the first-order term identified, and the obvious control (token centring) run with tolerances set before reading.
- Ten seeds including the known outlier runs, with dates robust to tolerance, leave-one-prompt-out and the plateau window.
- Interventions with random controls matched in size per head, and the recognition that the "whole model at step 128" reversal is close to a tautology.
- A real attempt at the schedule confound (§7) rather than ignoring it, reported with a negative outcome.
- Figures that mostly show what their captions claim (I checked Figs. 2, 3, 4, 6, 8, 9 and 10 against the text at higher resolution).
