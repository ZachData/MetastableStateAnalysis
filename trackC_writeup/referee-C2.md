<!-- trackC_writeup/referee-C2.md -->
# C2 unit 5: the fresh-context referee read, and the answers

**What ran (2026-10-23).** A fresh agent (no conversation history, general-purpose, the
session's model) was asked to referee `draft-C2.md` as of `9840633` (#259 merged) as an
anonymous reviewer would, reading only the draft and its four figures (web allowed for cited
papers). Its brief asked for a summary, a recommendation, numbered major and minor points, and
what it could not understand; it was given none of this project's reasoning. Its report came back
as a message (its file write failed: the scratchpad was withdrawn mid-run) and is copied below
with its wording unchanged; only the nested bullet lists under each "Fix" and in the minor points
are joined into one line each. **Recommendation: major revision; 9 major points, 27 minor.**

The answers follow the report. Each point is **fixed** (in `draft-C2.src.md` at this unit),
**not fixed, because** (a reply), or **to Blocked 102** (a run or reading beyond C2's budget,
the user's to take or carry). `/challenge-pr` on this unit's PR is the design's second half of
unit 5.

## The report (verbatim)

**Counts: 9 major points (M1–M9) and 27 minor points (m1–m27).**

I read only the draft and its four figures. I turned the figures into images, and for Figure 4 I also checked the band edges against the coordinates in the SVG. On the web I checked only two facts: PolyPythias' model sizes, and Mistral's checkpoint spacing. I did not check the theorem and section numbers cited from Geshkovski et al. or Pendharkar.

### 1. Summary of claims

The authors normalise each token's residual-stream state onto the unit sphere. At each of 25 depth states they compute the interaction energy E₁ from the clustering theory of self-attention (Geshkovski et al.). V is the number of layer-to-layer drops in E₁, summed over 7 hand-picked 512-token prompts. They read V at 18 checkpoints of pythia-410m and of the 9 PolyPythias seeds. The claims are:

1. **The break.** V is 0 through step 128 and reaches half its 1000–4000 plateau at step 512 on seed 0. A prediction registered in advance put each seed's break in (128, 512]; it held on 9 of 9 seeds (p = 9.9e-8, E = 1587 against a uniform-location null the authors concede no one holds).
2. **The centred break.** After subtracting each prompt's mean token, the count breaks at 1000 instead. The authors read this as "the break is the common direction's".
3. **A timeline on seed 0.** The break comes after the rank collapse and the n-gram minima, coincides with the fall in the OV maps' attractive share, and comes one grid step before the previous-token and induction heads. The ordering "break, then induction, then OV repulsive phase" is explicitly not shown.
4. **OV edits.** Symmetrising the OV maps or removing their rotation does not undo the break. Swapping in each head's step-128 OV map lowers V at 512 and 1000 but leaves it far from monotone.
5. **Schedule.** Every early event lies inside the 1430-step warmup. Runs at 70m/160m (higher peak rate) and on Mistral GPT-2 small (different warmup) do not separate a training event from the schedule.

### 2. Overall assessment and recommendation

**Recommendation: major revision.** This is a workshop-level paper as it stands. The authors are unusually candid about their limits, and the internal numbers mostly agree with each other. Two things decide the recommendation:

1. **The headline finding cannot be interpreted with what the paper shows, by the paper's own account.** The break cannot be separated from warmup (§7) and is not attributed to the OV maps (§6). §4.1 suggests it belongs to the shared mean direction, i.e. anisotropy, a known phenomenon. At β = 1, E₁ is close to a first-moment anisotropy statistic (M1). What remains is that a grid-coarse count crosses half its plateau at step 512 reproducibly across seeds. That is solid, but thin as a contribution, and the "interaction energy of the particle picture" framing oversells it.
2. **The experiments that would settle it are feasible and absent.** These are:
   - a schedule intervention on a small Pythia that can be reproduced (M3);
   - per-component weight transplants from step 128 (M6);
   - a direct measurement of the mean direction (M1, M2);
   - the PolyPythias 70m/160m seeds that already exist (M7).

### 3. Major points

**M1. At β = 1, E₁ is close to an anisotropy measure, and the paper never says so (§2.1, §3.2, §4.1, §8).**
- For unit vectors, ⟨xᵢ,xⱼ⟩ = t ∈ [−1, 1]. Expanding the exponential gives E₁ = ½ + ½‖m‖² + (1/4n²)Σᵢⱼ tᵢⱼ² + O(t³), where m is the mean of the normalised tokens.
- So the uncentred V largely counts layer-to-layer decreases in ‖m‖², the norm of the shared direction, plus a second-moment term.
- §4.1's main result, that removing the mean delays the break, is then close to expected from the statistic itself.
- The centred E₁ is dominated by the Gram matrix's Frobenius norm, which is an inverse effective-rank quantity. V_c is therefore related to row E5, which the paper treats as a separate change.
- The novelty claim in §8 ("What is new here is the interaction energy of the particle picture…") is weakened, because at β = 1 nothing particular to the particle picture is being measured.
- **Fix:** state the expansion; report ‖m‖² per layer and the count of its drops beside V; read E_β at several β on the same frame and pipeline (§3.2 says only β = 1 was read; E₃.₅ uses a different frame and is never counted); relate V_c to E5; reframe the contribution accordingly.

**M2. "So at its step the break is the common direction's" (§4.1, last two paragraphs) is a mechanism claim drawn from dates alone.**
- The same section says sizes cannot be compared and that "how much of the uncentred count the common direction carries is not measured".
- Centring plus a separate set of tolerances could delay a half-rise for reasons of sensitivity, not mechanism. The four-tolerance robustness reduces but does not remove this.
- The obvious candidates, massive activations and outlier dimensions (named only in Appendix A) and attention sinks, are not examined in the main text.
- **Fix:** do one of: decompose E₁ directly into mean and centred parts; measure ‖m‖ and its angle across layers and steps; project out the massive-activation coordinates and recount; or soften the claim to "the centred count's half-rise comes one grid step later".

**M3. The central confound is left open although the decisive experiment is affordable (§7, §1 "What none of this separates").**
- Every event from 128 to 1000 lies inside the warmup, and the schedule's own row (E10) dates to 1000 by construction.
- The authors name the experiment that would separate schedule from training (training with and without the ramp) but do not run it.
- Pythia publishes its configs and data order. Training 70m from scratch for about 2000 steps (about 4B tokens) with (i) the original ramp, (ii) a flat rate and (iii) a longer ramp is a few GPU-days.
- The same run would give a dense checkpoint grid, which removes the 512–1000 resolution limit that bounds every claim in §4–§5. Continuing from a released checkpoint is weaker, because optimiser state is not released and a fresh Adam state is its own confound.
- Without this, both "dated events" may be properties of warmup, and the paper's claims reduce to dates of the schedule.

**M4. The registered test is weak but presented prominently (§1 item 1, §3.4, §4).**
- **Built-in prior.** The window (128, 512] was set after seeing seed 0, and seed 0 sits on its right edge, so it is not "around" seed 0's step. PolyPythias already reports seed-consistent dynamics, and all seeds share the schedule, so 9/9 was close to certain before the run.
- **Lenient falsifier.** The falsifier K ≤ 3 corresponds to P(K ≥ 4) ≈ 0.048 under the null.
- **Strawman null.** The uniform null includes steps 2–64, where V = 0 on every seed while the learning rate is near zero.
- **Inflated prominence.** Putting "**9 of 9**", p = 9.9e-8 and E = 1587 in bold in the introduction gives the result more weight than the authors' own caveat in §4 ("a large E here is cheap") supports.
- **Not verifiable.** Readers cannot check the registration: the registry is "this project's git history", commit c2eb9a7 is not public, and the first record came 3 minutes after the commit.
- **Fix:** move p and E out of the headline; add a continuous measure of seed spread (interpolate each seed's half-rise in log-step and report the spread, since grid dates hide variation at a factor-2 resolution); make the timestamped registration publicly checkable (a public repository or OSF).

**M5. The §5 timeline has no null, uses coarse alignment, and its locator ignores larger later changes.**
- **No null for alignment.** With ten rows dated by the same half-point on a log grid during a linear ramp, "aligned within one grid step" (7 of 11 pairs) is close to guaranteed. Nothing tells the reader how many aligned pairs independent warmup-driven quantities would produce.
- **Inconsistent language.** The introduction (item 3) treats one-grid-step separations as order: the break comes "after" the bigram minimum and "one grid step before" the heads. §5 counts those same pairs as aligned and says ties are "not ordered". Pick one convention.
- **The locator misses reversals.** Its reference is the largest Δ up to 4000, so it dates the first excursion only. In Figure 2, E5's effective rank dips (dated 8), then crosses back between 512 and 1000 and ends far beyond its initial level (the line leaves the panel). That is a large change at the break's own step, and it is undated and undiscussed. E3 (induction) peaks at 1000 and then declines, also not discussed.
- **"Fraction done at 512" is not an ordering.** Rows with different functional forms (a count, an attention weight, a learning rate) cannot be ordered by their fraction completed at 512.
- **Seed coverage.** Most rows are dated on seed 0 only.
- **Fix:** add a null for the number of aligned pairs; bootstrap the dates (over prompts, heads and blocks); date all rows on the nine seeds; date reversals as separate events.

**M6. The OV experiments have low power and localise nothing (§6, Figure 3, Appendix B).**
- **Controls move V a lot.** Under edit (a) at step 2000, the unedited V is 61 while the 40 random edits give about 29–41 (Figure 3a). "As its controls" therefore means only that the edit behaves like a large random perturbation.
- **Size of edit (a) not reported.** Only (b)'s size is given (0.08–0.10 of the norm).
- **Unreported cells under (a).** Figure 3a shows V above the controls at step 256 (30 against roughly 9–20), and V = 11 at step 0 where the unedited model has 0 (so the edit makes the initialised model non-monotone). The text reports only the five steps from 512.
- **The positive control is nearly a tautology.** G3 moves the whole model to its step-128 weights, which by §4 has V = 0, so it is effectively the step-128 model; at full dose it shows little beyond that. Intermediate doses are mentioned ("at full dose") but not shown.
- **MLPs are never tested.** §2.1 notes that no theorem covers MLP blocks, and Pythia's parallel attention/MLP residual is never mentioned.
- **Fix:** transplant step-128 weights into the step-512 and step-1000 models one component class at a time (embeddings, layer-norm gains, QK, OV, MLP), each alone and leave-one-out, with dose curves; report the size of edit (a) and the gain-only control; report the step-0 and step-256 cells.

**M7. The §7 schedule analysis rests on thin data and choices not specified in the paper.**
- **Seeds exist but are not used.** "There is one seed per size" is fixable: PolyPythias releases 9 seeds at each of 14m, 31m, 70m, 160m and 410m (arXiv:2503.09543 abstract). 14m and 31m would also add more peak-rate conditions.
- **70m's break rests on about 3 drops.** Its E₁ plateau is 6 drops, so the half-line is 3. Figure 4 (left) shows 70m's curve at roughly 0.5–0.65 at steps 256–512.
- **Band construction unspecified.** The rule for mapping 410m's dates onto another run's grid as a band is not given, and the sentence giving the 70m versus 160m bands is ambiguous about which band belongs to which model.
- **Mistral's schedule is asserted, not shown.** The learning rate "rises logarithmically, half its peak by 100" is pivotal for the cumulative band but rests only on "the logged rate". No curve or citation is given, and a log-shaped warmup is unusual.
- **Plateau window mapped inconsistently.** The Mistral plateau window is mapped by tokens while the hypotheses are tested in steps.
- **Frozen label overridden after reading.** The rule's label was set aside after the run. This is honest, but it leaves §7 with no usable conclusion beyond "neither".
- **Fix:** plot the instantaneous and cumulative learning rate for every run; state the band rule; use the PolyPythias 70m/160m (and 14m/31m) seeds; give the source for the Mistral schedule.

**M8. The core object is never shown, and the count conflates frames (§3.2, §4).**
- The reader never sees E₁ plotted against depth at any checkpoint, nor which layers carry the drops, nor any per-prompt breakdown.
- V is a sum over 7 hand-picked prompts and 24 transitions (maximum 168).
- The first transition starts from the embeddings and the last ends after the final layer norm. Both change frame, so up to 7 drops per checkpoint could be frame artefacts.
- The net change in E₁ from first to last state is not reported. "Non-monotone" is a weaker departure if E₁ still rises overall.
- **Fix:** add a layer × step heatmap of drops; add E₁-versus-depth curves at, say, steps 128, 512, 1000 and 143k; report per-prompt V; recount with the frame-changing transitions excluded; repeat on a larger sampled prompt set (the 64 Pile validation blocks already used for E6 would do).

**M9. The paper is not self-contained and reads as a lab notebook.**
- Undefined internal labels: "tier 1", "A3's rows", "v1's prompts", "G3′", "P-A5", "the frozen rule", "the joint reading", "re-initialised models", "a simulator's cluster count from the measured start".
- Undefined terms in Table 1: E7 "censored", E11 "refused".
- There is no abstract.
- Process narration (gates, refusals, what was frozen when) crowds out the science. Some sentences are opaque: "A grid step named as a place on the axis (512, 1000) is a label; a step that is a result is quoted from its record."
- **Fix:** add an abstract, define or remove every internal label, and move the process material to an appendix.

### 4. Minor points

- **m1.** Title and §1: there is no abstract, and "the interaction energy along training" does not say what is found.
- **m2. §1 item 2, "breaks one grid step later".** This holds only for seeds whose break is at 512. Seeds 3 and 9 (break 256, centred 1000), 70m and 160m (256 → 1000) are two grid steps apart.
- **m3. §4, "The second energy agrees".** E₃.₅ dates 512 on all nine seeds, while E₁ dates seeds 3 and 9 at 256, so it disagrees on 2 of 9. The choice of β = 3.5 and of the input frame is unexplained.
- **m4. §3.4.** The window (128, 512] is not "around" seed 0's step, which sits on its right edge. Mixed time zones (13:36 UTC−5 against 18:39 UTC): give both in UTC. Give the falsifier's level under the null (about 0.048).
- **m5. §2.3.** "(1430 and 1430)" is a rendering artefact. "[the released configs]" is not a citation; the Pythia paper lists the learning rates.
- **m6. §4.1 tolerances.** Say which two of the four tolerances are fixed and which two are derived. 7.6e-5 and 7.6e-4 look like one value and ten times it. "Centred plateau 20 against 57": at which tolerance? It does not seem to be one of the four. Add a table of V_c at 512 and the plateau for each tolerance, so that the pairing of "5–11 drops" with "0.19–0.24" can be checked.
- **m7. §4.1.** It says sizes are "never compared", then compares plateau sizes (0.96) to withdraw the label. Reconcile the two.
- **m8. §5.** "Nine other measured changes" does not match Table 1's eleven other rows. Say which rows are excluded (E10, E1c?).
- **m9. §5, "Of the 11 pairs read".** Define the pairs. I reconstructed them as 10 pairs with E1 plus E3–E4, which gives 7 / 2 / 2 as stated, but the reader should not have to.
- **m10. Figure 2.** It omits E7 and E11 without saying so. E5 and E6u leave the panel and the reader cannot see by how much. Only 0 and 1 are labelled on the y-axes. E2 and E8 are labelled "falls" but plotted rising (signed). Say so in each panel.
- **m11. §5 and Table 1 head selection.** How were seed 0's "two induction heads" and L5H2 selected? Choosing heads by their scores at the last step (for the seeds) can bias the dates. Say whether seed 0 used the same rule.
- **m12. §5.** "This project's earlier medians of the OV maps" is an internal, unpublished reference. Remove it or cite it.
- **m13. §6 and Figure 3.** "The five steps from 512 on" are 512, 1k, 2k, 4k and 143k, not 512–54k; say so. The panel title "G3′" is undefined. Give the unedited G for the copying probe beside 8.19–12.41. The gain-only control, which is part of the labelling rule, is not shown in the figure. The random controls in the third panel: matched in what, for a substitution edit?
- **m14. §6(b).** "Removing the rotation in an orthonormal basis of each two-dimensional invariant plane" is unclear for non-normal OV maps, whose invariant planes are not orthogonal. Give the construction.
- **m15. §7, Mistral.** Checkpoints are every 10 steps to 100, then every 50 to 2000 (Appendix B and Figure 4 use steps 10 and 20). "Every 50 steps to 2000" is incomplete. The GPT-2 state definition (13 states; frame of the final layer norm) is not given. The Mistral release blog (crfm.stanford.edu, 2021-08-26) can be cited.
- **m16. Figure 4.** Right panel: V is raw, not normalised as on the left. It has no plateau window or half line (so "half between the stages" cannot be checked), no centred curves although the text relies on them, no learning-rate curve, and the five seeds are indistinguishable. Step 0 is drawn at the position of step 1 without saying so. Left panel: the 70m E₁ curve is normalised by a plateau of 6; flag it or show raw counts. I checked the band edges against the SVG: cumulative 40–150 and step 150–550 match the text.
- **m17. Figure 1.** The legend "seeds 1–9, the rest" is ambiguous. The registered-window band is hard to tell from the warmup shading. Seed 4, the other outlier, is not marked. Individual seeds cannot be identified.
- **m18. §8 related work.** Missing: massive activations (Sun et al. 2024); attention sinks (Xiao et al. 2023); anisotropy and outlier dimensions (Ethayarajh 2019; Timkey and van Schijndel 2021); rank collapse in attention (Dong et al. 2021; Noci et al. 2022); warmup mechanics (Liu et al. 2020; Kalra and Barkeshli 2024); abrupt structure formation during training (Chen et al. 2023, "Sudden drops in the loss"); induction-head formation dynamics (Singh et al. 2024).
- **m19. Architecture details.** Pythia's parallel attention/MLP residual and rotary embeddings are not stated, though they matter for the mapping onto the particle picture and for §6. The layer counts of 70m and 160m are not given (the maximum V differs: 42 / 84 / 168).
- **m20. §3.1.** Why 54 000 rather than another step? "Step 1's outputs are identical to step 0's in our runs": say whether the weights are identical (a known duplicate checkpoint?) or merely the outputs.
- **m21. §2.1.** Clarify that the normalised self-attention flow is a gradient ascent of E_β in a modified metric, which is why monotonicity holds. Give the conditions on d and n under which the uniform measure (or a configuration) minimises E_β for finite n < d. I did not check the cited section, proposition and theorem numbers.
- **m22. Prompts.** Give the seven prompts' exact text in an appendix; the hash is useless to readers. "Hand-picked" needs a sentence on how they were picked.
- **m23. §4 "Beside the count".** On five seeds the step-0 drops vanish by step 8. That is an event where the model *becomes* monotone, and it coincides with E5's date of 8. It is not discussed.
- **m24. §1.** "Falls 70 times over our seven prompts": give the denominator (168 transitions).
- **m25. Appendix A.** There is no table or figure; tabulate ρ by family and step. These sentences are opaque: "The null, a norm-matched Gaussian draw of the weights and step 0 are one law"; "a planted effect that adds dimensions measures only its own dilution"; "the residual-stream check below".
- **m26. Appendix B.** The planted-null check of 20 000 draws only checks the scorer's arithmetic (that K follows Bin(9, 1/6)), not how robust the half-rise rule is. The displacement guard refused both planted effects and was demoted to description. Say what it reported for the real edits.
- **m27. §3.3 and §4.** The plateau window 1000–4000 was chosen on seed 0. Report how the seed dates change if the plateau is taken at, say, 8k–32k (V keeps rising late; Figure 1 right reaches about 1.7).

### 5. What I could not understand from the paper alone

- What E11 is: "a simulator's cluster count from the measured start". Also what "refused" means for it, and what "censored at 2000" means for E7.
- Which of the four centred tolerances are "fixed" and which "derived", and what the "re-initialised models" used to derive them are.
- How a 410m date is mapped onto another run's grid "as a band" (the rounding rule), and which band in the §7 sentence about 70m/160m belongs to which model.
- The source and shape of Mistral's learning-rate schedule, and the batch size behind "410m's window mapped under tokens".
- How edit (b) is built for non-normal maps, what the 40 random controls are matched to for the step-128 OV substitution, and how the "change of gain alone" control is computed.
- The "two choices that rule made" that the Mistral label turns on. I inferred the plateau window and the tolerance from §7, but the introduction does not name them.
- Appendix A's null construction and its "check below".
- What "tier 1", "A3", "v1", "P-A5" and "G3′" denote.
- Whether the registration and its timestamp can be checked by anyone outside the project.

## The answers

Each answer is checked on the rendered draft (`python -m trackC_writeup.numbers check`); five new
sentence claims pin the new statements (46 claims, 201 keys). Section numbers are the revised
draft's.

### Major points

| # | verdict | what was done, or why not |
|---|---|---|
| M1 | **fixed in text; the reading to Blocked 102** | §2.1 states the exact series E₁ = ½ + ½‖m‖² + Σ_{k≥2}(1/2n²k!)Σᵢⱼ⟨xᵢ,xⱼ⟩ᵏ (`tools/math_checks/c2_energy_moments.py`, 4 of 4; it notes the higher terms are not small, since diagonal terms are 1, so "close to ‖m‖²" is not claimed); §4.1 opens on it; §8 says the statistic is not new in kind, the particle picture contributes the prediction against which a drop is read; the abstract and §1 item 2 name it. *Not done:* the ‖m‖² drop count beside V (seed 0's per-state mean cosine is already in unit 3's records, so this is a reading, not a run), several β in the same frame (a forward-pass run), and V_c against E5 (both are Gram statistics; no relation is claimed). All three are Blocked 102 (a)/(b). |
| M2 | **fixed** (the softening) | §4.1's conclusion now reads "by the half-rise, the break at its step is not seen once the mean token is removed … This is a statement about dates. It does not say the break is anisotropy", and names the ‖m‖² count as the unread measure; §1 item 2 likewise. §8 names massive activations and attention sinks as candidates for the mean and says neither is examined. The decomposition or a projection of the massive coordinates is Blocked 102 (a). |
| M3 | **not fixed: Blocked 102 (c)** | Agreed that it is the decisive experiment; it was already named in §7 as the third kind of run. Training 70m from scratch with three schedules is a new harness (the earlier 70m training harness is gone from disk), GPU-days on the local RTX 3080, and outside C2's design (Blocked 92 (iv) was declined for cost). It would also need its own preregistered reading rule. Carried to the user as a post-C2 phase; C2's text keeps F1 as the headline limit. |
| M4 | **fixed in part** | §1 item 1 no longer bolds 9 of 9, and adds that with seeds sharing the recipe this was the expected outcome and the registration guards only the window's choice; p and E stay out of §1 (they were only in §4). §3.4: the window is "chosen after seed 0 was read, with seed 0's step at its right edge"; the falsifier's level under the null is a key (`reg.falsifier_p`, 0.048); both times in UTC; "checkable in the project's git history". *Not fixed:* the null's inclusion of steps 2–64 is the registered null, and §3.4 already says nobody holds it and beating it is cheap; changing it now would be scoring against a null chosen after the data. Interpolated seed dates: a new statistic chosen after the reading; the grid's resolution is stated in §3.1 and the dates are not reported more finely than it. Making the commit public is a release decision (Blocked 93, the user's: venue and records' release after the draft). |
| M5 | **fixed in part** | §1 item 3 now uses §5's convention (within one grid step = aligned, not ordered); §5 defines the 11 pairs, says a one-step gap counts as aligned here and in §1, and that no null exists for the aligned count, so it is description; the locator dates first moves only, and E5's climb back past its start between 512 and 1000 and E3's peak at 1000 then fall are now described (claims pin both); the share-at-512 paragraph says shares of unlike quantities order only as description. *Not done:* a null for the aligned-pair count, bootstrapped dates, rows on the nine seeds, dated reversals: Blocked 102 (a) for what the records allow, else carried. |
| M6 | **fixed in part** | §6 now names the five labelled steps, reports (a)'s step-0 cell (11, inside its controls' 0–17, so a random edit of its size also breaks monotonicity at init) and step-256 cell (30, above all 40 at 9–17), says (a) is a large edit by what its controls do, gives (b)'s Schur construction, says how the controls are matched (twice: change norm and edited-map norm, per head), that the gain-only control is read and not drawn, and that G3 is close to a tautology and only shows the count can register a reversal; the MLPs (in parallel with attention) and every non-OV component are named as never moved. *Not done:* (a)'s edit size as a number (it is in S2's records, not yet a key; a one-key addition if wanted), and per-component transplants with dose curves: Blocked 102 (b), a run. |
| M7 | **fixed in part** | §7 now states the band rule from `c2_dose.predicted` (a grid date stands for a crossing in (g′, g], carried by f = 1, √(p/p′) or p/p′ inside a linear warmup, and a grid step is predicted when its own interval meets the carried one); the 70m/160m sentence names each band's model; 70m's E₁ plateau of 6 is flagged in the text and figure 4's caption (Parked 1 of unit 4, closed); GPT-2's schedule is sourced to each seed's released `trainer_state.json`; the tokens mapping of the plateau window is explained (a GPT-2 small step is a quarter of a Pythia step's tokens); the checkpoint spacing corrected (every 10 to 100, every 50 to 2000). *Not done:* the PolyPythias 70m/160m (and 14m/31m) seeds and LR curves in a figure: Blocked 102 (b). On the overridden label: §7 already reports the frozen label, the decision (Blocked 101 (a), the user's) and when it was taken; "neither schedule maps the whole curve" is the conclusion. |
| M8 | **fixed in part** | §3.2 gives V's maximum (24 × 7) and names the two frame-changing transitions, counted like the rest, with their share not reported. *Not done:* the layer × step map, E₁-by-depth curves, per-prompt V, a recount without the frame transitions and the net first-to-last change are all readable from the A5 records already on disk (`e1` per state per prompt per seed): Blocked 102 (a). A larger sampled prompt set is a forward-pass run: Blocked 102 (b). |
| M9 | **fixed in part** | An abstract; a title that says what is found; "tier 1" → "exploratory"; "A3's rows", "v1's prompts", "the frozen rule", "G3′" (text and figure 3's panel), "the joint reading" kept only where defined; "re-initialised models" defined (appendix B and §4.1); "censored", "refused" and E11 defined in table 1's caption; the opaque §3.1 sentence removed; "this project's earlier medians" removed. *Not done:* moving all process material to an appendix. The gates are already in appendix B; what remains in the text is when a rule was fixed against when its data were read, which is what the reader needs to weigh each reading. `P-A5` stays as the registered prediction's name, defined in §3.4. |

### Minor points

| # | verdict | answer |
|---|---|---|
| m1 | fixed | abstract added; title "When a language model's interaction energy stops rising with depth" |
| m2 | fixed | §1 item 2: one grid step after on the seeds that break at 512, two on those at 256; heading "breaks later" |
| m3 | fixed | §4: "dates the break inside the window too", differing from E₁ on seeds 3 and 9 (claim); §3.2 says why β = 3.5 and the frame |
| m4 | fixed | window's choice and right edge; UTC (new format `utc`); level 0.048 (key) |
| m5 | fixed | the parenthetical rewritten; Biderman et al. 2023 cited for the peak rates, the released configs for the warmup |
| m6 | fixed in part | which two tolerances are fixed and which derived, and how; "20 against 57" placed at 10⁻³; *not* a per-tolerance table (the pairing is one number per end; the per-tolerance values are in the record, one key each if wanted) |
| m7 | fixed | the 0.96 is "quoted only to withdraw the label; it is not a size reading" |
| m8 | fixed | "eight other changes measured on the model, the learning-rate schedule and a simulator's row" |
| m9 | fixed | the 11 pairs defined: the break with each of the nine others, E4–E3, E3–E2 |
| m10 | fixed in part | caption: signs, the cut and which rows run below it (claim), E7 and E11 not drawn. *Not* more y-tick labels (the lanes are shares; 0 and 1 are the two values the reading uses) |
| m11 | fixed | E3's heads named (`L8H6`, `L11H14`); caption: E3's and E4's heads were named in earlier work, not by the seeds' rule |
| m12 | fixed | sentence removed |
| m13 | fixed in part | the five steps named; G3′ retitled in the figure; the unedited G stated as about equal; the gain-only control said to be read and not drawn; the third panel's controls "matched to that substitution as above". *Not* the unedited G's numbers (a key from S2's base record if wanted) |
| m14 | fixed | the real Schur construction on the head's write space |
| m15 | fixed in part | the spacing corrected; GPT-2's states defined as for Pythia. *Not* the CRFM blog: unit 4 cited the checkpoints by repository because no paper exists, and the blog was not read here; added at the close-out's LaTeX pass if the user wants it |
| m16 | fixed in part | caption: raw V, seeds not told apart, centred counts not drawn, step 0's position, 70m's plateau. *Not* the right panel redrawn with the plateau, half line, centred curves and LR: Blocked 102 (a) (a figure from records) |
| m17 | fixed | legend lists the seven grey seeds; caption names the band and shading and says seed 4 is among the grey; individual seeds not labelled (stated) |
| m18 | fixed | all ten added, titles and authors from the arXiv API (2026-10-23), each cited only for what its abstract says; Liu et al. is 2019 (first arXiv version), not 2020 |
| m19 | fixed in part | parallel attention and MLP, rotary embeddings (§3.1); V's maximum for 410m. *Not* 70m's and 160m's layer counts: they are in the fetched configs but not keys; one key each at the close-out |
| m20 | fixed in part | "we did not compare the weights". *Not* why 54 000: the grid is the one this project's earlier phases read; the reason for that step is not recorded in the material read this session |
| m21 | fixed | gradient ascent in a reweighting metric (§2.1); the minimiser stated over probability measures on the sphere (the mean-field statement the citation makes). Finite n < d conditions not added: the draft does not use the minimiser |
| m22 | fixed in part | a sentence on how they span languages and registers, chosen by hand. *Not* the texts in an appendix: one is a copyrighted novel's passage; releasing the prompt set with the records is Blocked 93 (the user's) |
| m23 | fixed | §4: the early drops are gone by step 8, E5's date, "seen, not read further" |
| m24 | fixed | "out of 24 × 7 layer-to-layer transitions" |
| m25 | fixed in part | the three opaque sentences rewritten; "ρ by family and step is not tabulated here" stated. *Not* a table (S1's records hold it; a figure at the close-out if appendix A stays) |
| m26 | fixed | the planted null "checks the scorer, and the planted breaks check the half-rise rule"; the guard "is not reported here" (its real-edit readings are description in S2's records) |
| m27 | not fixed: Blocked 102 (a) | the dates at a later plateau are readable from the A5 records; §3.3 now says "how the dates move under a later plateau is not read here" |

### Blocked 102 (the user's)

The referee's readings and runs go past C2's budget (8 units + close-out; the exit asks for every
point answered, which the table does). Options:

- **(a) One more unit, 5b, readings from records already on disk** (no forward pass, no
  download), each rule fixed before its output: on the ten seeds, per-prompt V, the layer × step
  map of drops, V without the two frame-changing transitions, E₁'s net first-to-last change, and
  the dates under a later plateau (M8, m27); how often a drop in E₁ comes with a fall in ‖m‖²,
  per prompt and transition (M1, M2; *the ‖m‖² profile itself is now reported, after
  `/challenge-pr` on #260*); figure 4's right panel normalised with the half line
  and the centred counts (m16). About one unit; it can change §3.2's and §4.1's wording, not a
  date already read.
- **(b) Runs** (forward passes or downloads): E_β at several β in the uncentred frame (M1); the
  massive coordinates projected out and recounted (M2); per-component step-128 transplants with
  dose curves (M6); the PolyPythias 70m/160m seeds (M7); a larger sampled prompt set (M8). Each
  about one unit; together a new phase, not C2.
- **(c) The schedule intervention** (M3): training 70m from scratch with the ramp, a flat rate and
  a longer ramp, with a dense checkpoint grid. GPU-days and a new harness; the only run that can
  separate F1. A phase of its own after C2 (it is also what S4's toy would do with a known
  circuit).
- **(d) None:** C2 closes at its budget with the answers above; (a)–(c) go to `docs/ROADMAP.md`
  as rows.

**Recommended: (a), then the close-out; (b) and (c) to ROADMAP rows**, with (c) the first
post-C2 candidate. (a) is cheap and answers the referee's most direct question (what does the
count look like, and is it the mean?) without new data; (b) and (c) are new experiments that
change what the paper can claim and deserve their own design and budget.

## After `/challenge-pr` on #260 (accept with changes; reviewed `6af0635`)

All five findings taken. The counts in the answers above ("46 claims, 201 keys") are the
pre-review figures; after these fixes the table has **207 keys from 22 records and 49 claims**.

| # | finding | done |
|---|---|---|
| 1 | §1 (abstract, item 2) said the centred count breaks "later" at 1000 against 512 while item 3 called one-step gaps "aligned, not ordered"; §5 still said "one grid step after / before" | §4.1 and §1 item 3 say why the two differ: the two counts are one statistic on the same states dated by one rule fixed before the run, which reads grid order (unit 3s's K1); §5's convention is for rows of different kinds. §5's sentence now reads "within one grid step of" |
| 2 | "We have not read" the ‖m‖² term was false: units 3 and 3s read the layer-mean cosine profile on all ten seeds (post hoc), which fixes ‖m‖² per prompt, and the answer to M1/M2 withheld it | reported in §4.1 as post-hoc description from a record: `c2_mean_profile.py` (no forward pass; recomputes the per-transition counts from the per-seed records, since the 3s record kept only one mean per step) writes `data/trackC/c2u5/2026-10-23-mean/profile.json`; keys `mean.falls`, `mean.frame_falls`. ‖m‖² never falls at 128 on any seed, falls at 6–10 of 24 transitions at 512 on every seed, and the two 256 seeds fall most at 256 (it reproduces the counts unit 3s quoted). The frame-changing transitions hold none of the falls at 128–1000 but do later (8000 on), so the sentence says "at steps 128 to 1000". §8 ties it to Razzhigaev et al.'s mid-depth peak as "compatible with", the peak not located. Blocked 102 (a)'s item becomes the co-occurrence of the two counts' drops |
| 3 | "Checkable in the project's git history" overclaims: commit dates and the record's time are self-written; the reviewer found GitHub's earliest event at 18:45:17Z, after the first record | GitHub's activity log also holds an earlier event: the **branch creation at 18:38:17Z** with tip `c337961`, which contains `c2eb9a7`, 60 s before the first record's 18:39:17Z. §3.4 now says both of our times are self-written and quotes this one as the earliest outside timestamp; the events are copied to `trackC_writeup/p_a5_push_record.json` (GitHub keeps them about 90 days); keys `reg.pushed_at`, `reg.pushed_kind`, `reg.pushed_has_commit`, and a claim that the event is a branch creation carrying the commit, before the first record |
| 4 | §4.1 described the side-by-side centring (centre raw, then normalise), not the scored one | "take the unit token cloud, subtract the prompt's mean unit token and normalise each row again" (`c2_aniso.centred`, as `e1c`) |
| 5 | §8 called the energy new and the statistic "not new in kind" in adjacent sentences; the math check's M3 docstring named one zero-mean condition and tested another | §8: the statistic is not new in kind; what is new is reading the prediction along a training axis. M3's docstring names the symmetric cloud it tests |

## Blocked 102 decided (a), and unit 5b's readings (2026-10-23)

The user took (a): unit 5b, readings from records on disk, rules frozen before output
(`design-C2.md` "Unit 5b", committed `9629d78`); budget extended on the record to 9 units +
close-out. (b) and (c) are `docs/ROADMAP.md` rows C5 and C4. Numbers: `status-C.md` "C2 unit 5b".
The points left open above now read:

| point | was | now |
|---|---|---|
| M1, M2 (how much of V the mean carries) | "not measured" | **read** (R6): ‖m‖² falls at every one of the 1224 drops at 128–1000 and its term is more than half of all but one; V_R (R's drops alone) breaks at 512 on all ten. §4.1, the abstract, §1 item 2. The split is exact, not a separation (R carries the mean too). β, the massive-coordinate projection: C5 |
| M5 (rows on the nine seeds, nulls, reversals) | carried | still carried: no record holds A3's other rows on the seeds; the aligned-pair null is a new statistic (C5 or C2's close-out names it as a limit) |
| M8 (the layer × step map, E₁ by depth, per-prompt V, without the frame transitions, net change) | "readable from records" | **read** (R1–R4, R7): appendix C, figure 5, §3.2. No date moves. A sampled prompt set: C5 |
| m16 (figure 4's right panel) | carried | **fixed** (R8): normalised, half line, centred arms, LR / peak |
| m27 (a later plateau) | "not read here" | **read** (R5): every seed stays in the window; only seed 9 moves, 256 → 512. §3.3 |
