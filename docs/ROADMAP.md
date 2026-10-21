# Roadmap: what comes next, in what order, and when each piece is done

**Written:** 2026-10-10, docs-only session at the user's request ("map out everything that we
want to do and figure out the best way to actually get there … finish Phase 10 first"). It
replaces `docs/TRIAGE_2026-09.md` (renamed here; the 2026-09-29 triage is in
`git show d50d2e5:docs/TRIAGE_2026-09.md`). **Inputs:** `origin/main` `d50d2e5`: `STATE.md`, every
phase card (`docs/PHASES.md`), the 2026-09-29 triage, `p10_cluster_function/handoff-10.md`
Stages 3–7, `LESSONS.md`, `docs/readings/2609.28448v1` (Gao–Yang–Chen, repulsive self-attention).
**Nothing was run.** Tier 1: a plan, not a decision; the user's decisions are §10.
**Revised 2026-10-18** (the user's check-in, docs only): Track S added as candidates (§7a), C2
re-scoped and given the user's target (§8), process items (§5), a proposed order after B3 (§9
row 6) and §10 items 8–10. Nothing was run; Hugging Face and the PolyPythias refs were checked (§6).

**How to use it.** `STATE.md` says what is true now; this file says what comes next. Update it
when a phase closes, a track's order changes, or a decision in §10 is taken (`CLAUDE.md` Stop).
Numbers are quoted with a pointer to the file that owns them.

## 1. Where the project stands (one line each)

| finding | strength | owner |
|---|---|---|
| Monotone energy holds exactly at steps 8–64 and breaks at 128–512 | strongest card on the training axis | `p2_eigenspectra/status-2.md` "Headline result" |
| The OV spectrum goes fully repulsive at 1000–2000 (62 % of heads exactly repulsive at 1000) | measured, never set beside the energy curve | `PROJECT.md` §3.12 A |
| Cutting each head's OV to its attractive part collapses the stream at 16000 (2.5–3.4 groups against ~15) | large, global; the sign-versus-distance split is muddy | `p10_cluster_function/status-10.md` §1.37–§1.39 |
| The trained cloud at the theory's scale is one cone plus outliers; cluster structure sits near a matched Gaussian | the cluster object is weak in trained Pythia | `p1d_cluster_ensemble/status-1d.md`; `p1b_hemisphere/status-1b.md` |
| At the measured β (3.5 [1.6, 5.6]) the field has one well; its structure is mostly there at init (Phase 1e closed 2026-10-13) | small effects (`|X|` ≤ 0.06) | `p1e_energy_field/status-1e.md` "The phase on one page (closed 2026-10-13)" |
| Phase 10's group-level effects are small, often carried by step 0, and c3x's groups are never singled out (Phase 10 closed 2026-10-11) | many tier-1 labels, none registered | `p10_cluster_function/status-10.md` "The phase on one page (10.5, closed 2026-10-11)" |
| The induction set on 410m: ~4 substantial members in L5–15, formed in (512, 4000]; `L5H2` relays the previous token to `L7H8`; MLP 6 is an OR-gate with `L5H2`; `L11H14` leaves the core | causal (ablation) | `p7d_redundancy/status-7d.md`, `p7e_consolidation/status-7e.md` |
| Trained 410m's integration time along the identity-weight field is below the causal collapse time at all 18 steps, while over 90 % of its motion is off that field (Track A, A1, closed 2026-10-15) | `P-gamma2` has a number, not an adjudication; rests on the projected step | `trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)" |
| The heads' subspaces concentrate on shared directions of the residual stream from 256 (QK, OV write, OV complex) or 512 (the rest), QK most from 2000 on; not the stream's top variance; at 143000 partly the massive-activation coordinates (Track S, S1, closed 2026-10-21) | described against an exact null; the effect side uncalibrated | `trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)" |
| 39 registered predictions, zero adjudications | the registry has never scored a real run | `STATE.md` "Where things stand" |

**Blog 1 (2026-07-08, gpt2-large), its claims now.** Collapse resistance is the part that has held
best (the first two rows above), but claim (a) "step 0 looks random-like" is still recorded as not
adjudicated (`PREDICTIONS.md`) and `P-gamma2` ("never integrates far enough" against "resists") is
unscored. The ~50-cluster universality is mostly a repeated-token count (`status-10.md` §1.8); the
attention flip to unclustered tokens is position 0 plus the causal mask (§1.17, §1.19); the ~250
effective rank is not re-established on normed rank (`MATH_INDEX.md` item 3).

**The reading this roadmap is built on.** The big signals are global (collapse against
resistance); everything asked of particular clusters came out small. Blog 1's own headline
predicts that: a trained, untied model is trained against the collapse the theory describes. So
the plan stops asking trained Pythia for vivid clusters and asks two other questions (§2).

## 2. What we are after

| track | question | why it matters |
|---|---|---|
| **A, the anti-collapse force** (the flagship, kept from the triage) | What force keeps the trained stream from collapsing, which components carry it, and when in training does it switch on? | the project's most robust phenomenon; publishable on its own |
| **B, particles in a known circuit** (new) | What does a known mechanism (induction first, copy suppression next) look like as particles: which subspace, which pulls and pushes, which routing? | the bridge from the mean-field picture to mechanistic interpretability; it starts from a mechanism that is known to be real, so it doubles as the positive control the particle instruments never had |
| **C, write-ups** | 1d + Phase 10 as a null-model audit; A as the flagship; B later | the year's output |

**Where A and B meet.** The energy break (128–512), the induction set's formation window
(512, 4000] and the OV's fully repulsive phase (1000–2000) overlap on the checkpoint axis.
Nobody has put them on one timeline (`PROJECT.md` §3.12 A says so for two of them). A3 and B5 do.

**The paper connection (2609.28448).** Gao–Yang–Chen study `Q = K = I`, `V = −I`. Two results map
onto real heads: (i) for a symmetric V, a cluster's direction is drawn exponentially into the
eigenspace of V's most negative eigenvalue, at a rate set by the gap to the next one (their
eq. 190–196), so the dynamics can live in a small subspace and a degenerate negative pair behaves
like their d = 2 circle model; (ii) attention condenses onto single routing partners (at β ~ N²
for d = 2, at β = O(1) when d = N), and in the hard-routing limit a perturbation spreads only
along routing partners (an emergent light cone). Induction heads route each token to one
partner; copy-suppression heads have a repulsive OV. Both are hypotheses to test on Pythia, not
expectations: the paper's model is recurrent, tied, without MLPs.

## 3. Rules for phases (why this file exists)

The repeated failure: a phase finds a problem, a new phase opens to go around it, and the old one
is left with open Blocked and Parked items (Phase 10 held for 1d, 1d's definition re-read by
Phase 10 while 1e opened and paused; over 20 Parked items in `handoff-10.md`; 1e's U4 / U5; 1d's
write-up). `LESSONS.md` 10 records it; the calibration gate exists because `LESSONS.md` 6 records
the null built after the output, again and again (E1m, E1p, OV1d).

**The rules live in one place: `CLAUDE.md` "Phases and tests"** (exit condition and unit budget,
close-out before the next phase opens, the calibration gate, the named family and the
exploration rungs). This file only applies them. A **unit** is one session and one PR, as
`CLAUDE.md` "One session per unit" uses the word.

## 4. Step 1: finish Phase 10

**Exit condition.** Phase 10 closes when (i) the OV thread is closed, (ii) the parking account has
been tested on its own terms (F13 with the anchor/ballast cross), (iii) the one functional test
the phase never ran (loss coupling) has a number, and (iv) the close-out is done.
**Budget: 4 units for 10.1–10.4, each including its calibration gate, plus the close-out (10.5),
which is never cut.** If 10.2 or 10.4 needs a second unit, the last of 10.3 / 10.4 drops, not the
close-out.

| # | unit | what | cost | home |
|---|---|---|---|---|
| 10.1 | close the OV thread | if the user takes Blocked 34 (b) (§10.1): tier 1 at "the sign matters beyond distance at 8000 L17–24 on 7 exploration passages, not shown at 16000; c3x not singled out". **The full-cut question (what removing the repulsive part does at 16000) closes unanswered; nothing below takes it over.** B0's weight-space nudges ask a different, local question (the sign of a small change at the trained weights). Writing only | none | `status-10.md` §1.39, handoff |
| 10.2 | F13 + A9 + F15 | the Rényi / strong-Rényi centre scan swept in δ, geodesic and `⟨Qx, Ky⟩`, joined to received attention, `Z_i/(i+1)` and membership; the coverage curve (`handoff-10.md` Stage 3). Stored activations, no partition needed. **Falsifier:** centres and members receive indistinguishable attention once position is divided out | free (CPU, stored) | Stage 3 |
| 10.3 | F14, if registered | observed strong-centre count against Lemma C.1 with an exact i.i.d. null; the handoff says "register before looking" and keeps 1b / 1.4b reserved until a prediction names them. **To settle first (§10.3):** the rung policy keeps 410m exploration-only, while `status-10.md`'s card holds the 12 v2 prompts out on 410m as a confirmation set (scoring set and order undecided, `docs/PHASE_REVIEW.md` "Open" 1–4). F14 either registers on a reserved rung, or runs exploratory on 410m. **Not run in Phase 10 (closed 2026-10-11, Blocked 39 (a)): the budget was spent (10.2 took three units, 10.4 two), so by the rule above it drops; F14 goes to the first-registration decision (§10.7)** | free | Stage 5 item 2 |
| 10.4 | loss coupling | per-token surprisal against c3x membership, position as a covariate (`handoff-10.md` Stage 6 item 1); does the gap open where A0's residual and §1.1's window do? One forward pass per prompt per checkpoint (GPU). **Done 2026-10-11, one unit: no gap, so no onset (`status-10.md` §1.44). A second unit, the user's extension (STATE Blocked 38 (b), 2026-10-11; 10.3 had not run, so none dropped): LCo, the members' own-token surprisal adjusted for its frequency and class, early gap the token's, late less surprising beyond them from 4000 (§1.45)** | ~2 GPU minutes (the guess was ~1 hour); LCo CPU minutes | Stage 6 item 1 |
| 10.5 | close-out | one-page summary of 10 (and 1d, 1e as they bear on it); `status-10.md` §6; card; every Parked item in `handoff-10.md` marked carried or dropped; Blocked 26, 27, 30, 34 closed. **Done 2026-10-11: `status-10.md` "The phase on one page (10.5, closed 2026-10-11)", §6 rewritten, the card's open threads each with a home, `handoff-10.md` "At close (10.5)"; Phase 10 is closed** | writing | §3 rule 2 |

**Leaves Phase 10, with a home.**

| item | goes to | why |
|---|---|---|
| Stage 4: H-THERMOSTAT, H-WARP (`T_eff`) | Track A (A1, A2) | global, about the force, not the clusters |
| Stage 5 rest: `d₁`, F16, the carrying-capacity join | dropped unless 10.3 runs; then F14's design decides | the carrying capacity is mostly a repeat count (§1.8) |
| Stage 6 items 2–4, the J-lens rows F2–F10 | dropped | they read a cluster object 1d and Phase 10 found weak |
| Stage 7, the drive as an instrument | Phase 9 (parked), revisited after B4 | B gives a lever on a known mechanism first |
| ALBERT (Blocked 26 (e′)) | B0′, optional | it is the theory's own regime (tied weights): the positive control for the clustering instruments |
| The OV thread's parked refinements (finer dose, matched death share, effect-matched controls, one block at a time) and E1's (placebo, planted pulls, per-head kernels) | dropped with their threads, if Blocked 34 closes at (b) | each refines a question being closed; the full-cut sign question goes with them (B0 asks only the local one) |

## 5. Step 2: other open ends (in parallel with Step 1 where free)

| item | cost | action |
|---|---|---|
| Stale worktrees | minutes | **done**: `../Mets-sign`, `../Mets-dose` removed 2026-10-10, `../Mets-roadmap` 2026-10-13 (`STATE.md` "Open PRs and branches") |
| Phase 1e close-out | writing | **done 2026-10-13** (Blocked 40 (a)): `p1e_energy_field/status-1e.md` "The phase on one page (closed 2026-10-13)"; U4 and U5 *not pursued*; Parked items carried to A1 (1), A2 (3), A3 (1), B0 (1), C1 (1); 5 dropped, 3 done |
| Claim (a), step 0 against the random criteria | check first | Phase 2's steps 8–64 may already answer it; if so, write it up and adjudicate |
| `P-gamma2` / `T_eff` | check first | 1c says no `beta_eff` in any run dir; β is now measured (Blocked 9). Goes to A1 if it needs a run |
| Normed-rank re-report (Blog 1's ~250) | free | A, as a Blog 1 retest |
| `CLAIM-C` calibration to 20 prompts (with it, from Phase 10's card at close: its two HDBSCAN metrics against the reproducibility floor, now float64's, the only Phase 10 item bearing on a registered prediction) | ~45 min | deferred by the user; stays deferred unless the registry write-up needs it |
| `P-S1` draws 500 → ≥ 1 600 | free | with A if P-S1 is ever scored (1e U4 waits on it) |
| 5c frequency regression | free | only if a "what is in a cluster" claim is written (C1) |
| Blocked 2 (`P-I5`'s target), Blocked 7 (which rung registers first) | the user's | both before Track B registers anything |
| Blocked 3, 4, 12, 13 | the user's | independent; no work waits on them |
| Design preflight tool (`LESSONS.md` 6; 2026-10-18) | 1 unit | given a statistic, run it on step 0, on matched random weights, on the trained population at one trained step and on a planted effect at three doses; report the attainable floor and the detection curve; refuse on "a band nearly everything leaves", "a control already at its ceiling", "a dose below the null's mean". A design after it cites its output before the freeze. Built before S1's design freezes. **Done 2026-10-20:** `tools/design_preflight.py` (values in, a JSON report out, exit 1 on refusal; also refuses a floor below 1/α, a control reading an effect unplanted, a largest dose seen < 80 % at the E-bar's p unless the design names its level, a trained step among the scored cells, and an unknown or missing key); `tests/test_tools_design_preflight.py` replays B2's band, A2's control, B2's re-gate dose, `P-S1`'s 500 draws and A3 unit 2's shape as known answers (what it does not check: `LESSONS.md` 6's status line); `tools/math_checks/design_preflight_floor.py` (n ≥ 1599 draws for E ≥ 20). "A design cites its report before the freeze" as a `CLAUDE.md` rule is the user's (STATE Blocked 80) |
| Alternatives at design time (2026-10-18) | per design | before a design freezes, a fresh-context pass lists five ways to answer its question, simplest first; the design says why it took its own (`/challenge-pr` does this only after the run) |
| More than one null (2026-10-18) | per design | where several nulls exist, the design fixes how they combine before any is read: all must agree, or all are reported side by side. "One may fail and we still claim" is not a combination rule (it picks the null that passes) |
| Pre-authorised fallbacks (2026-10-18) | per design | a design names its likely gate failures and the revision each gets ("if G2 fails by X, apply Y and re-gate"), so a planned revision runs in the same unit instead of ending it on a Blocked item; scope (a phase opened or closed, what counts as a claim, registration) still goes to the user. A `CLAUDE.md` rule, so the user's |
| `STATE.md` byte cap (`LESSONS.md` 9, 2026-10-18) | minutes | the 150-line cap does not bound bytes, so the start hook's output is persisted to a file and re-read. A byte cap in `tools/lint_repo.py`, the 1d / 1e / phase-review rows moved to their status files. After B3 unit 2's PR merges, since that unit is editing `STATE.md` |
| Start hook reads `origin/main` (`LESSONS.md` 1) | the user's | `git fetch -q` then `git show origin/main:STATE.md` in `.claude/settings.json`; the hook has printed a stale `STATE.md` six times |
| Per-read cost attribution (`LESSONS.md` 9, 2026-10-18) | small | `tools/session_cost.py` ranks tool results by size × calls remaining after them, naming the reads that drive the per-call context |
| Refuse-on-empty writer (`LESSONS.md` 2) | check first | whether every runner writes through one `core/` writer that refuses empty, NaN or constant outputs and records the counts; if not, build it |

## 6. Track A: the anti-collapse force

Kept from the triage (its §4.1 and §6a); what changed since 2026-09-29 is that β is decided
(Blocked 9: 3.5 [1.6, 5.6], unit LN1) and 1e's U2 split each block's update into attention, MLP,
sink and bias at that β. **Opened 2026-10-14: `trackA_force/design-A.md`** (A1: exit condition,
a 3-unit budget, the instrument, its gate, the reading rule), scan `trackA_force/lit-A.md`.

| # | unit | what | done when |
|---|---|---|---|
| A1 | β per head × checkpoint; `T_eff` against `t*` (`P-gamma2`); H-WARP (from Phase 10 at close: F0b, the `beta_eff` slope; from 1e at close: is one well at β 3.5 what 2312.10794's thresholds predict at n ≈ 2,000, d = 1,024) | does the trained net integrate far enough to collapse at all? | `P-gamma2` has a number, or a recorded reason it cannot. **Done 2026-10-15** (5 units): below t* at all 18 steps on `T_proj`, H-WARP refused (`trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)") |
| A2 | **Opened 2026-10-16: design frozen (Blocked 56 (a)), `trackA_force/design-A2.md`, scan `lit-A2.md`; 5 units + close-out. Unit 2 (the first item) done 2026-10-16: A1's number does not move by its bar (`trackA_force/status-A.md` "A2 unit 2"; STATE Blocked 57). Unit 3 done (the instrument passes its gate). Unit 4 done 2026-10-16 (Blocked 58): against the field is a few MLPs at 11–13 of 18 steps (against-specific at 6), the MLP cancels attention at 1000–4000, block 23's step against is its MLP (`status-A.md` "A2 unit 4"). Unit 5 done 2026-10-17 (Blocked 59 (b), (ii)): the ablation rule's calibration fails on the planted control, so the reading is not made (`status-A.md` "A2 unit 5"). **Closed 2026-10-17 at its exit's "a recorded reason it cannot be read"** (Blocked 60 (a), the user's; `status-A.md` "A2 on one page (closed 2026-10-17)"): against the field is a few components, mostly early MLPs, against-specific at 8 of 18 steps; nothing causal is read. Carried: the ablation's repair and a depth control for MLPs → C2; M1 against `fe_full` on code and LaTeX → B3; four timeline rows → A3.** **First item (Blocked 54 (a), the user's, 2026-10-16):** refit A1a's per-head β at 143000 by M1's cross-entropy (or keep heads with A1a's `partial_r2` above a floor placed first) and see whether A1's layer medians move past its bracket (`trackA_force/status-A.md` Parked, "A1a's β sign bias"); then per-component dissipation at the measured β, then mean-ablation (from Phase 10 at close: `attention-10.md`'s row A5, paid against received attention, the sink's part; from 1e at close: the late sum-over-heads ascent, cooperating heads or a few large ones; the sink's value ≤ 0.05 of the shared update; a second sink on a newline, so whether "sink-only" means any fixed key; from A1 at close: which components carry the >90 % off-field motion, attention's along-field step cancelled by the MLP at 1000–8000, block 23's step against the field at 143000, whether `T_attn_proj` is drift-robust, the long passages as an input choice, `a1a.ln_from`'s duplicate) | which heads and MLPs push apart, from which step; ablate the top ones: does energy turn monotone, do tokens re-cluster, at what loss? (`PROJECT.md` §3.8.3's exact split) | "these N components carry the anti-collapse force" is supported or refused |
| A3 | **Opened 2026-10-19: design frozen as revised (Blocked 70 (ii) (a)), `trackA_force/design-A3.md`, scan `lit-A3.md`; 5 units + close-out; 10 located rows, E1 dated by A5's D2 (a) / D4 (a), 11 read pairs (E1 against nine, E4–E3, E3–E2). Unit 2 done 2026-10-19: the locator's gate fails its effect arm on E2–E5. Re-gate (unit 2c, Blocked 71, the user's) passes every arm: E1 512, E2 512, E4 1000, E5 8, E10 1000, E3 "no change". Unit 3 done 2026-10-19 (Blocked 72, the user's: 6 units on the record; E3 re-defined at +1 on `L8H6`, `L11H14`; floors max(4, own)): E3 1000, E7 2000 (censored), E8 1000, E11 refused (`trackA_force/status-A.md` "A3 unit 3"). Unit 4 done 2026-10-20 (Blocked 73 (a), the user's: E11 stays refused, censored rows' pairs unread, E8 read only where its two dates agree): E6u 128, E6b 256 on the Pile's validation split (`status-A.md` "A3 unit 4"). Unit 5, the reading, done 2026-10-20 (Blocked 74 (a), the user's: the bigram also on 340M tokens, still 256): 7 of 11 pairs aligned, all inside the warmup; E5 (8) and E6u (128) before E1 (512); E7, E11 unread; E4–E3 not ordered by the half-point rule, though the shares done at 512 order E1, E2 < E4 < E3 (`status-A.md` "A3 unit 5"; 6 of 6 units spent). **Closed 2026-10-20 at its exit** (Blocked 75 (c); `status-A.md` "A3 on one page (closed 2026-10-20)"): one table and one order, every aligned pair inside the LR warmup; C2's sequence is shown in neither reading of the OV phase. Carried: a tie-breaker inside one grid step → A5 (`design-A5.md` D10, proposed, yours); induction at +1 → A5's notes row; C2's sequence → C2; the quoted rows drawn, none read; E7's date and `lit-A3.md` Parked 2 dropped with why.** One timeline (2026-10-18: the core of C2, §8; from Phase 10 at close: R6f's embedding move at 512 → 2000 against the n-gram stage; from 1e at close: U2's regime change at 2000–4000 beside the OV turn; from A1 at close: step 16's collapse at the last block in 3 of 7 prompts without the field's integration; 1e's one-well question, the causal simulator's trajectory from the measured start at each step's β; from A2 at close: the against-specific steps (16, 32, 1000–16000, 143000), the MLP cancelling attention at 1000–4000, the fixed key from a period (2000–4000) to a newline (from 8000), the sink's per-token share rising from 1000) | the energy break, the OV repulsive phase, the induction formation window, n-gram milestones (triage M2) and weights-only D1 / D3 / D4 (`FUTURE_IDEAS.md`) on the same 18 checkpoints | the curves are side by side, with their alignment or misalignment stated |
| A4 | claim (a) and the Blog 1 retests | §5 rows 3–5 | adjudicated or recorded as unadjudicable |
| A5 | generality | register one prediction (e.g. "the energy break falls in [128, 512]") on the rung §10.7 picks or on PolyPythias seeds, then run it. **Proposed 2026-10-18 (§10.9):** the 9 PolyPythias 410m seeds, whose intermediate steps nobody here has read, registered before they are downloaded; beside it, descriptive: whether each seed's energy break and induction onset move together (Pythia's grid has no step between 512 and 1000, so a lag inside that interval cannot be resolved). **Drafted 2026-10-18; text decided 2026-10-19** (Blocked 68, the user's: nine choices D1–D9 as recommended, a reproducibility check): `trackA_force/design-A5.md` (`P-A5`, window (128, 512]); not registered until the calibration gate on seed 0. From A3 at close: the induction row read at +1; D10, a tie-breaker (the share done at the grid step before a tie, break–induction and previous-token–induction), descriptive beside K, **decided (a) 2026-10-20 (Blocked 76, the user's)**. **Unit 1 done 2026-10-20: the calibration gate on seed 0 passes, every arm** (`trackA_force/status-A.md` "A5 unit 1"; producer `trackA_force/a5_seeds.py`, gate `p_value_p_a5`); **registered by the user 2026-10-20** (`c2eb9a7`, Blocked 77 (a)); **units 2–3 done 2026-10-20: K = 9 of 9** (`trackA_force/status-A.md` "A5 units 2 and 3"); **adjudicated and closed 2026-10-20** (Blocked 78 (a), the user's: `claims/adjudications/P-A5.json`, the project's first ledger record; H-RESIST `reject_null` on this one factor, a reproducibility check against a null nobody holds, quoted with its notes; `trackA_force/status-A.md` "A5 on one page (closed 2026-10-20)"). C2 receives: the break's step is the recipe's on 410m's seeds; the shared warmup; D10's per-seed order as description | the project's first adjudication (done) |

Risks (from the triage, still open): Pythia's LR warmup overlaps 512 and 1000–2000; sink and
compression prior art (`2510.06477`); literature trigger 1 before A2's design freezes.
**Models on disk** (`/challenge-pr` on #185, finding 6; corrected 2026-10-18): the PolyPythias
seeds are cached only at step 0 and 143000. Hugging Face is **not** blocked from the local box
(2026-10-18: its API answers; `HF_HUB_OFFLINE=1` is the scripts' setting, not a block), and each
410m seed has all 154 branches (`step0` … `step143000`, Pythia's grid) at 0.91 GB per checkpoint,
so the 16 missing steps of the 18-step grid are about 131 GB for 9 seeds (`HDD_1TB` had 265 GB
free on 2026-10-18; S1's 8-step grid needs less). ALBERT (B0′) and the CRFM GPT-2 checkpoints (B6) are not cached, not checked.

## 7. Track B: particles in a known circuit

**Start from Phase 7, not from scratch.** Phase 7 already asks whether induction is a motif in the
particles' interaction graph ("a two-stage relay of attention forces, split into the value
operator's sign channels"), with nine `H-BRIDGE` rows: `P-I1` ran (INSUFFICIENT); `P-ST1`,
`P-AB1`, `P-I3` are built and calibrated but unrun. B's design reads `p7_motifs/` first and
reuses what holds. **Opened 2026-10-13: `trackB_circuit/design-B.md`** (B0 and B1: exit condition, a 3-unit budget, the harness, its gate, B1's rule), scan `trackB_circuit/lit-B.md`. **A candidate first unit, B-pre (free):** check whether those three registered
rows can still be scored as built (inputs on disk, rung, gates; `P-AB1` and `P-I3` have no
known-answer dry run). If they can, they may be a cheaper first adjudication than A5 and a
better start for B than B0 (§10.4). **Done 2026-10-13** (`p7_motifs/status-7.md` "B-pre (2026-10-13)"): on 410m none
can be an adjudication (an exploration rung, and `P-I1` already read `P-I3`'s motif side).
`P-AB1` needs B0's harness; `P-I3` meets its requirement only with weak members; `P-ST1` is built
and, on the reserved rung, the one candidate cheaper than B0 (projectors from weights, one forward
pass, linear algebra), its readability open. B starts from B0; `P-ST1` goes to §10.7.

| # | unit | what | done when |
|---|---|---|---|
| B0 | the nudge harness (from Phase 10 at close: one shared position correction in `core/`, where Phase 10 had three; from 1e at close: whether a GPU-against-CPU comparison is allowed on long passages, ~1e-4 at 143000's deep layers) | choose a direction, in the weights (a head's OV moved to `K ± ε·S₊`, one eigenvector's weight) or in the activations (a subspace, one token's state); nudge by ±ε; read each token's displacement, energy change and next-token loss; null = random directions of the same size; for small ε the ± distances agree to second order, so the antisymmetric response is the sign effect with no matching (a `sympy` check). Passes the calibration gate (§3 rule 4) before any real read | planted null and planted effect pass; its NUMERICS say whether a GPU-against-CPU comparison is allowed on long passages (from 1e) |
| B0′ | optional: ALBERT or random GPT-2 (from Phase 10 at close: 5c's flip on ALBERT with the sink out, both arms) | the clustering instruments where the theory's collapse is vivid | large effects there, or the instruments are suspect |
| B1 | positive control: `L5H2` | its particle signature (each token pulled toward its left neighbour inside `L5H2`'s output subspace) should switch on inside (512, 4000] | it does (the instruments see a real mechanism, and "small" has a yardstick) or it does not (fix the instruments first). **Done 2026-10-14: it does** by the revised pass rule (off at 0, on at 1000–8000; `design-B.md` "B1"): on from step 512, so the onset is in (256, 512], before this row's window; the yardstick and what B1 does not cover: `trackB_circuit/status-B.md` "B1 (2026-10-14)" |
| B2 | spectra from the weights (from Phase 10 at close: `⟨Qx, Ky⟩` per head, F18's `V`-spectrum atlas, `attention-10.md`'s A3) | each member's symmetrised OV and QK spectra across checkpoints; dominant negative eigenpairs (the d = 2 reduction); principal angles between members (7e's free item); 2d's gradient-flow regime test per member; Phase 6's `U_neg` / `U_A` split on the circuit's subspace; from B2a at close: the d = 2 reduction reproduced no head (most are not isotropic content plus position, and V's 2D blocks occur at the random-weight rate, so compare any spectrum rate with step 0), and QK read rank (M2's ladder, `b2a.py`) is open (`trackB_circuit/status-B.md` "B2a on one page"). **Opened 2026-10-17: `trackB_circuit/design-B2.md`** (proposed, Blocked 61; weights only, 3 units proposed; "dominant negative eigenpairs" read as the top pair's weight, since sign counts of a rank-64 symmetrised map are structural, `tools/math_checks/b2_symmetric_inertia.py`; `attention-10.md`'s A3 moved to B3, which already names it). **Closed 2026-10-18 at its exit (3 units, Blocked 61; Blocked 64 (a)): in the weights the members differ in K-composition (`L5H2` writing into `L8H6`'s, `L7H8`'s, `L11H14`'s keys at 21–32× chance, with 8 non-member readers at ≥ 10× chance) and in QK symmetry (both directions; circuit or head type open); C1, C2, C4 show no difference within the gated reach** (`trackB_circuit/status-B.md` "B2 on one page (closed 2026-10-18)"). Phase 6's split not built (dropped: beside, can change no reading). Carried: what to nudge to B4, a head-type control for C3 to B3 | a per-member table across training (**done**) |
| B2a | the 2D spike (the user's, 2026-10-14; `trackA_force/design-A.md` "Frozen 2026-10-14"): for each named head (`L5H2`, the induction heads, `L10H7`, random reference heads), the smallest subspace whose reduced dynamics reproduce its real attention; d = 2 is the paper's circle case. Three units: weights only with a planted positive control and a null; real tokens at 143000; the heads that pass over 18 steps. Needs A1a's β and step; runs before A1b or here, the slot picked at A1a's close. **Closed 2026-10-16 at its exit (4 units, Blocked 50): no head of 14 has D\* ≥ 1 at any of the 18 steps, mostly because trained heads are not isotropic content plus position** (`trackB_circuit/status-B.md` "B2a on one page (closed 2026-10-16)"; design `design-B2a.md`, the reduction 2609.28448 §6.1). Carried: A1a's β bias to A2, QK read rank to B2, `L10H7` to B6 | does a head's particle dynamics live in a small subspace, and how small? | a faithful D per named head against the null (**done**) |
| B3 | routing (from Phase 10 at close: `attention-10.md`'s A3, per head joined to 7d's causal sweep; from A2 at close: M1 and `fe_full` give opposite per-head β signs mid-depth on the code and LaTeX prompts, so pick and justify the estimator before reading β against thresholds, `trackA_force/status-A.md` "A2 unit 2"; from B2 at close: a head-type control for B2's C3, re-ranking QK symmetry with 7d's previous-token and induction non-members as the null, since `L5H9`, `L4H9`, `L8H9` sit with the members, `trackB_circuit/status-B.md` "B2 on one page") | effective attention partners and per-head β against the paper's condensation thresholds (does a head condense when it forms?); the light-cone test: nudge token j, does the response spread to tokens whose induction partner is j, against matched others? **Opened 2026-10-18: `trackB_circuit/design-B3.md`** (proposed, Blocked 65; 3 units proposed). The paper's condensation is a large-N limit with no finite-N line, so routing is read β-free as the slope α of log N_eff on log n, with β beside and no per-head threshold (`tools/math_checks/b3_condensation_exponent.py`). The carried items: A3 as a descriptive row, C3′ as a type-matched re-rank, β with both estimators on prose prompts. **Closed 2026-10-19 at its exit (3 units + unit 2b, Blocked 65 (a), 69 (c); Blocked 69 (a)): the light cone runs along induction partners from step 1000 (ℓ 2.37 at 143000) and goes through the three typed induction members in part (Δℓ 0.54, rank 1.000; half of it keeps the response off nearby tokens); the partner's response is added across depth (blocks 6–8, 10–11, 16–23); no difference shown in the members' α against their layers' (R1 0.80; R2 only `L5H2`), and C3 is not shown to go beyond head type (C3′ 0.945)** (`trackB_circuit/status-B.md` "B3 on one page (closed 2026-10-19)"). The three carried items done. Carried to B4: path patching, the heads that add the response, a block-6 nudge site, a layer-matched same-type control | both have a number and a direction (**done**) |
| B4 | particles to function (from B2 at close: what to nudge is `L5H2`'s write space and the 11 heads after layer 5 whose keys read it at ≥ 10× chance (`c6_pairs.json` `l5h2_readers_all`, code `b12ad273a201`): members `L8H6`, `L7H8`, `L11H14`; non-members in 7d's causal top 10 `L7H1`, `L8H9`, `L10H9`; non-members outside it `L7H7`, `L9H0` (27×), `L8H7`, `L10H0`, `L11H2`; then `L6H3`, `L10H4` at about 9×, not the members' keys alone (`L9H0`, outside 7d's top 10, is a contrast); C6 is a subspace overlap, so B4 is its functional test; `trackB_circuit/status-B.md` "B2 on one page", record `c6_pairs.json`; from B3 at close: split the members' Δℓ 0.54 on the light cone, and the 1.84 left without them, into direct path and self-repair by path patching; name the heads that add the partner's response (blocks 6–8, 10–11, 16–23; unit 2's per-block σ in `data/b3/2026-10-18-unit2/sweep/step143000.npz` `layers` is the baseline) and whether the neighbours' rise under ablation is direct; a nudge site after `L5H2` (block 6); a layer-matched same-type control for L1 (circuit or head type; about 2 GPU minutes on unit 2's cache); `trackB_circuit/status-B.md` "B3 on one page") | nudge members' eigen-directions; read the induction loss on repeated tokens | a particle-level move is tied to the circuit's output, or not |
| B5 | one head across training | `L5H2` or `L11H14` over the 18 checkpoints with B1's signature; onto A3's timeline | tracked, or a recorded reason it cannot be |
| B6 | copy suppression | GPT-2 small `L10H7`, whose OV is dominated by negative diagonal values (`2310.04625`; `p2_eigenspectra/lit-2.md` §1.2): a trained head with a repulsive V, the real-model analogue of the paper's `V = −I`. On 410m, `L10H7` (same index, unrelated) interacts negatively with the induction set at every checkpoint (`PROJECT.md` "`L10H7` is the dissociation"); from B2a at close: its content adds to attention but no V subspace carries it, while its V's top pair is near-repulsive at 2000 and 32000 and no V block beats the kernel at 1000–8000 (`trackB_circuit/status-B.md` "B2a on one page"). GPT-2 small training checkpoints exist from Stanford CRFM's Mistral runs (from memory; to verify) | B0–B4 repeated on it |

Later, only when a B unit asks for them: SAE features (Phase 3's crosscoder aligned with V at
chance, so expect little), the low-rank autoencoder (Phase 4, ALBERT/GPT-2, revivable), bigram /
n-gram directions (needs triage M2's producer), LLC on 70m / 160m (dropped for 410m on compute;
triage M3), own small models (triage M1; now S4, §7a).

## 7a. Track S: shared subspaces and the forces (candidates, 2026-10-18; S1 closed 2026-10-21)

From the user's check-in (2026-10-18). Track B ranked named heads against their layers' other
heads and mostly read "no difference shown" (B2a, B2's C1, C2, C4, B3 unit 1). Its one positive
weight-level row is a subspace one head writes and others read (B2's C6). So S asks about
subspaces shared across heads and layers, and treats a head as where routing lives, not as the
unit of the mechanism. Each S unit opens under §3 (scan, exit condition, unit budget).

**The picture, one claim per line.**

- Particles are the tokens' residual states. A dictionary feature is a direction (a coordinate),
  not a particle; a transcoder feature is closer to a mechanism (an input-to-output map).
- QK sets who interacts and how strongly. Its symmetric part is an indefinite similarity form on
  the read space (a metric tensor in the pseudo-Riemannian sense, **not a distance**), so each head
  scores closeness in its own warped geometry; for a rank-64 product its signature is split, 64
  and 64, by construction (`tools/math_checks/b2_symmetric_inertia.py`), so the information is in
  which directions and how large, not in how many are negative. Only its positive part is an inner
  product, and a distance. Its antisymmetric part is a directed term. Pythia rotates 16 of each
  head's 64 dimensions by position (rotary), so a QK form is read at one offset (B2 used 0).
- OV sets what an interaction does: its symmetric part pushes toward or away; a complex pair
  rotates content inside a fixed plane; a map from one subspace into an orthogonal one moves
  content between subspaces with every eigenvalue zero.
- So eigenvalues see attraction, repulsion and in-plane rotation, and only the write / read
  (singular) side sees transfer between subspaces, which is how the previous-token head feeds the
  induction keys (B2's C6).
- Complex pairs are the random default (a real Gaussian n × n matrix has about √(2n/π) real
  eigenvalues, Edelman–Kostlan–Shub 1994), and 2b's headline was an identity under global orthogonal maps
  (`p2b_imaginary/status-2b.md`), so every rotation reading needs a null and a test that is not
  invariant under such a map.
- Two archived leads are on ALBERT (tied weights, the theory's setting): Phase 4's dense low-rank
  AE found 33 V-attractive bottleneck directions (0 on GPT-2), and Phase 6's cluster-separating
  directions aligned with the imaginary subspace (0.887 against 0.067; not dimension-matched,
  unresolved). S4's tied arm is where both can be checked with a known circuit.

| # | unit | what | done when |
|---|---|---|---|
| S1 | the shared-subspace atlas (weights only; est. 2–3 units) | at steps 0, 128, 256, 512, 1000, 2000, 4000, 143000 on 410m: each head's OV invariant subspaces (real negative, real positive, complex pairs with their angle) and its OV and QK write / read singular subspaces; the mean of the heads' projectors, whose eigenvalues above the nulls mark the shared directions. Nulls: step 0, a norm-matched random draw, and (since trained heads mostly fail isotropic nulls, B2a's 214 of 252 cells outside) the trained heads at the same step each rotated by its own random orthogonal map, which keeps every head's spectrum and kills only the alignment between heads; beside it, the shared directions' share in the residual stream's top-variance directions, so a few global anisotropic directions are not read as sharing. Per shared subspace, its radial sign, rotation angle and negative-pair gap (the d = 2 reduction's "collapsibility") across the 8 steps | the number of shared subspaces and their spectra at each step, each against step 0 and the random draw. **Closed 2026-10-21** as revised: ρ described, no count (below) |
| S2 | what rotation does (est. 2 units) | through `core/intervention.py`, two arms: (a) 2b's after-Phase-10 item, `W_OV := S` (each OV replaced by its symmetric part; this doubles the rank to 128 and writes into the read space, so it is not rotation alone); (b) rotation alone: in OV's real block form each complex pair's angle set to 0 (λ → Re λ), keeping its invariant planes, rank and write space. Does the energy turn monotone after 512, does induction survive (7d's score), and is S1's shared rotation what carries it. **Received from S1 (2026-10-21):** R1b (overlap between families' shared spaces, write against key), R2 (within-layer sharing, description), and R3 on a leave-the-layer-out S for the OV families (S1's OV readings carry a selection bias: S is built from the heads it compresses); S1's S is P̄'s top 10 per family and step | each arm read against a norm-matched ablation of the same size |
| S3 | clusters inside subspaces (est. 2 units) | cluster tokens in each S1 subspace, and in a head's QK positive read subspace at offset 0 (where the form is an inner product, so distances exist), with 1d's instruments, against the same projection at step 0 and 1d's Gaussian null: are clusters too faint in the full space (1d) present in a shared subspace? The user's guess, recorded as a guess. **Received from S1:** at 143000 the write, negative and complex shared spaces are partly the massive-activation coordinates (357, 130, 966; `trackS_subspace/status-S.md` "S1 unit 3"), so S3 states them or projects them out | a plateau count per subspace against both nulls |
| S4 | a toy induction model (est. 3–4 units) | a 2-layer attention-only transformer (and one with MLPs) on an induction task, tied and untied, with and without repeated tokens in the data, every step saved, several seeds on the RTX 3080: a known circuit for S1–S3 and S5, and an energy curve across its training (does the break move when induction cannot form? the causal arm C2 lacks) | S1's atlas and the energy curve on it, with the circuit known |
| S5 | the carve (est. 3 units) | on S4, then on 410m's induction set: ablate S1's subspaces across heads (not whole heads), keeping what the induction score needs, until a minimal set is left (parameter decomposition in spirit, on subspaces) | a minimal subspace set that keeps the induction score, or a recorded reason none is found |

**S1 opened 2026-10-20** (`trackS_subspace/design-S1.md`, frozen as written, Blocked 81 (a); 3 units +
close-out). What the design changed from the row above: P̄ sees a head only through its subspace,
so the rotation null, the norm-matched random draw and step 0 are one law (Haar subspaces of the
heads' own dimensions; `tools/math_checks/s1_projector_null.py` fact 4); the preflight (step 8000)
refused per-unit verdicts against it (every trained layer shares), so S1 reads counts calibrated
by planting into step 8000's real pool, and within-layer sharing (a layer-shuffle null) is
description only. Unit 1's gate (2026-10-20): the instrument and the null pass; the count fails its
planted effect on the real pool in all 7 families, since a trained step's spectrum is dense at the
null's edge, so N is refused and R1's statistic is the user's (STATE Blocked 82).
**S1 closed 2026-10-21** (3 units + close-out, #243–#245; `trackS_subspace/status-S.md` "S1 on one
page"): R1′ = the participation ratio against its exact null, read as description (no plant
could calibrate its effect side); QK, OV write and OV complex concentrate from 256, the rest from
512, QK most from 2000 on; inside the shared spaces the radial sign reads nothing of S's own from
512 (random subspaces are as repulsive); no shared space is the
residual stream's top variance; at 143000 three are partly the massive-activation coordinates.
Units: `trackS_subspace/status-S.md`.
**S2 opened 2026-10-21; Blocked 86 decided (a) and (i), the user's (arm b in Schur planes; 3 units
+ close-out, the third for S1's carried items and Q3); unit 1 done 2026-10-21, the freeze is
Blocked 87** (`trackS_subspace/design-S2.md`, `status-S.md` "S2 unit 1"). What the design changed from the row above:
"angle set to 0" and "λ → Re λ" differ: λ → Re λ in A's eigenbasis edits a 410m head by 1.6× its
own size at 512 (`/challenge-pr` on #246), so the recommended arm (b) removes the rotation in an
orthonormal basis of each invariant plane (the real Schur blocks made symmetric; Blocked 86 is the
choice); both arms and their controls are one hook with a per-head core
(`tools/math_checks/s2_ov_arms.py`), since arm (a) has rank 128 and cannot be written into a
64-wide head; the controls are random edits matched to the arm in size *and* in the edited head's
norm (40 each), and a label is read only beyond the arm's gain change alone; every cell carries
its displacement and is not read where the arm moved the stream further than every control
(OV1's lesson); no theorem predicts "monotone" for (a), and in a toy symmetrising V barely changes
it (math check C6).

Order inside S: the preflight tool (§5; **built 2026-10-20**) before S1's design freezes; S1, then S2 (C2 needs it);
S4 can start beside S1; S3 and S5 after S1.

## 8. Track C: write-ups

| # | paper | inputs | when |
|---|---|---|---|
| C1 | "What counts as a token cluster?", a null-model audit | 1d, Phase 10 §1, F13 (10.2); the 8 items carried to C1 at Phase 10's close, and 1e's (c3x purity in the centred frame against any proximity partition), each needed only if C1 quotes the reading it refines (`p10_cluster_function/handoff-10.md` "At close (10.5)") | after Phase 10 closes |
| C2 | **re-scoped 2026-10-18 (the user's check-in): the interaction energy along Pythia's training axis, with the energy break (128–512), induction formation ((512, 4000]) and the OV repulsive phase (1000–2000) read as a sequence; target about two months, mid-December 2026 (the user's).** Inputs: A3, A5 on the seeds, S2 as the intervention, S4 as the causal arm if it is ready; a literature scan first (the induction phase change, 2209.11895; developmental stages by LLC), since "nobody reads this energy across training" is unverified (**scan done 2026-10-18**, `trackA_force/lit-C2.md`: not refuted; the nearest work reads rank, ID or anisotropy along training; the LR warmup is the confound to name). From A3 at close (`trackA_force/status-A.md` "A3 on one page (closed 2026-10-20)"): the order on 410m is rank (8) < unigram (128) < bigram (256) < break = OV fall's onset (512) < induction = previous-token (1000), every aligned pair inside the warmup; the OV attractive fraction starts falling with the break and reaches 0 on induction's step, so "break, then induction, then the OV phase" is shown in neither reading (fall or bottom). From A5 at close (`trackA_force/status-A.md` "A5 on one page (closed 2026-10-20)"): the break's step is the recipe's on the 9 PolyPythias seeds (256 or 512 on each; `P-A5` adjudicated), but the seeds share the warmup, so it cannot separate training from the schedule; previous-token is further along than +1 induction at 512 on 7 of 8 seeds where they tie at 1000 (description, not a reading fixed for C2). From S1 at close (`trackS_subspace/status-S.md` "S1 on one page (closed 2026-10-21)"): the heads' subspaces concentrate from 256 (QK, OV write, OV complex) or 512, all inside the warmup; inside the shared spaces the layers are as repulsive as in random subspaces from 512 (the OV phase seen in compressions); state the key and value rows against 2603.13314 after reading its full text (`lit-S1.md` Parked 1, carried). Kept: the anti-collapse force along Pythia's training axis (from A2 at close: no causal claim about which components push apart until the ablation is re-calibrated: one synthetic model on 7 starts, headroom on (iii), step 0 reading "refused", and a depth control for MLPs, which has no design yet; and an energy split per component (`PROJECT.md` §3.8.3) before any energy sentence about components; `trackA_force/status-A.md` "A2 on one page") | Track A | after A3 and A5; a draft by mid-December if §10.8 is taken |
| C3 | particles in a known circuit | Track B | after B4 |

## 9. Order and budget

| order | work | units (estimates) | gate |
|---|---|---|---|
| 1 | Step 1, finish Phase 10 (**done 2026-10-11**, 7 units: 10.1 one, 10.2 three, 10.4 two, 10.5 one; 10.3 dropped) | ≤ 5 | §4's exit condition |
| 1′ | Step 2's free items, beside it | 1–2 | — |
| 2 | B0 + B1 (the harness and the positive control) (**done 2026-10-14**, 3 units with the design: B1 passes, B continues on these instruments; `trackB_circuit/status-B.md` "B1 (2026-10-14)") | 2–3 | B1's verdict decides whether B continues on these instruments |
| 3 | A1, A2 and B2, B3 interleaved (**A1 done 2026-10-15, 5 units; B2a added and done 2026-10-16, 4 units + close-out; A2 done 2026-10-17, 5 units + close-out; B2 done 2026-10-18, 3 units + close-out; B3 done 2026-10-19, 3 units + unit 2b + close-out**: 21 units against the 4–6 estimated; row 6's order taken 2026-10-19, Blocked 68) | 4–6 | each unit's own "done when" |
| 4 | A3 (the joint timeline), B4, B5 (*superseded 2026-10-19 by row 6: B4, B5 after C2*) | 3–4 | — |
| 5 | A5 (registration, generality), C1 / C2 (*superseded 2026-10-19 by row 6*) | — | §10.7 |
| 6 | **Taken 2026-10-19 (§10.8, Blocked 68, the user's); proposed 2026-10-18:** C2's literature scan and A5's registration text (free; **both done**, the text decided 2026-10-19) → A3 (**closed 2026-10-20**, before A5's gate: #231 finding 4, the user's) → A5 on the seeds (**closed 2026-10-20**, adjudicated) → the preflight tool (**built 2026-10-20**) → S1 (**closed 2026-10-21**, 3 units + close-out) → S2 → C2's draft; S4 beside S1 as budget allows; S3, S5, B4–B6 after C2 | ~10–12 | C2's draft by mid-December |

**Trade-off to decide (§10.4).** The triage aimed C2 at arXiv in December. Putting Track B's
first units before Track A's pushes that back. The order above puts B0–B1 first because the
positive control also tells Track A whether "small" means small. Against it (`/challenge-pr` on
#185, finding 7): Track A's quantities are global and already large in places (§1), so A needs
that yardstick less than the cluster rows did, and B-pre may give an adjudication sooner.

## 10. Decisions for the user, by leverage

1. *Decided 2026-10-10 (Blocked 34 (b)):* **close the OV thread at (b)** (10.1). Recommended: yes, knowing the full-cut
   question then closes unanswered; (a) is a finer dose on the same far-from-base design.
2. *Decided (Blocked 35–36), ran:* **F13 in Phase 10** (10.2). Recommended: yes; without it the parking account closes untested.
3. **F14 and the rung policy** (10.3; Phase 10 closed without it, so this now joins item 7). The policy keeps 410m exploration-only; `status-10.md`'s
   card holds 12 prompts out on 410m as a confirmation set. Either F14 runs exploratory on 410m,
   or it registers on a reserved rung, or the policy gets an exception on the record.
4. *Decided 2026-10-13 (Blocked 40, the user's):* **1e's close-out, then B-pre, then B0 + B1, then
   Track A (A1).** Whether the December target for C2 stands is still open.
5. *Taken 2026-10-13 with Blocked 40 (c):* **`L5H2` as the positive control and first anchor** (B1; `trackB_circuit/design-B.md`).
6. **`P-I5`'s target** (`STATE.md` Blocked 2) before Track B registers anything.
7. **Which rung carries the first registration** (`STATE.md` Blocked 7): A5, `P-I7` and F14
   compete for it. *For the first adjudication, answered 2026-10-20 by A5 on a held-out set (the
   PolyPythias seeds, item 9); `P-I7`'s rung and the first Phase 8 registration stay open.*
Items 8 and 9 were decided 2026-10-19 (Blocked 68, the user's, as recommended); item 10 is
proposed, not decided.

8. *Taken 2026-10-19 (Blocked 68):* **The order after B3** (§9 row 6, proposed 2026-10-18): C2's path first (scan, A3, A5 on the
   seeds, S2), S4 beside (S1 in sequence before S2, as row 6 has it), B4–B6 after C2. The alternative is Track B to B5 first, which
   puts C2 past December. Recommended: C2's path first, since the user set the two-month target.
9. *Text decided 2026-10-19 (Blocked 68: `trackA_force/design-A5.md` D1–D9 as recommended, A5
   a reproducibility check, not a test of mechanism; D10 (a) 2026-10-20, Blocked 76; the
   calibration gate on seed 0 passes 2026-10-20, `trackA_force/status-A.md` "A5 unit 1";
   registered by the user 2026-10-20, Blocked 77 (a); K = 9 of 9, adjudicated, A5 closed
   2026-10-20, Blocked 78 (a).)* **A5 on the PolyPythias seeds** (§6): the statement (decided: "each seed's energy break falls in
   (128, 512]"), the energy definition and its rule, registered before the seeds' intermediate
   steps are downloaded. This would also answer item 7 for A5 (a held-out set, not the 410m
   exploration rung). Registering is the user's (`CLAUDE.md`).
10. **The §5 process items:** the preflight tool first (one unit, before S1 freezes; **built
    2026-10-20**, its citation rule STATE Blocked 80); pre-authorised fallbacks and the
    alternatives step as `CLAUDE.md` rules; the start-hook fix.

## 11. What each phase gives the plan

| phase | what to reuse | goes to |
|---|---|---|
| 1 | energy, plateaus, Fiedler, effective rank instruments | A |
| 1b | one cone, no antipodal split (background); `--n-null` rerun open | A (cheap) |
| 1c | β, `T_eff`, γ_β; `P-gamma1/2` (blocked on per-run β, now measurable) | A1 |
| 1d | null-model audit, Gaussian null, float64 distances, the working definition | C1; every instrument |
| 1e | the field `φ_β`, U2's attention / MLP / per-head split, wells | A2; B0's energy readout |
| 2 | V's attractive / repulsive eigen-subspaces; the OV repulsive phase at 1000–2000; Study A (V's mixed sign causal in GPT-era models) | A3, B2 |
| 2b | rotation inert (headline withdrawn as an identity); complex pairs mostly at their init value to 128, a small drop at 256–1000, no null | S1, S2 (2026-10-18; was "not pursued") |
| 2d | operator regime (QK symmetric, V = QK); `P-T1` / `P-M1` registered and calibrated | A (D1 is the same question); B2 per member |
| 3 | crosscoder directions align with V at chance | a caution for SAE work in B and S |
| 4 | dense low-rank AE finds V-attractive directions in ALBERT | optional instrument in B; S3, S4's tied arm |
| 5 / 5b | one cluster end to end; manifold steering | not pursued (B goes head-first instead) |
| 5c | the unclustered population and the rank budget; its flip is superseded | the "unclustered tokens do the work" idea, tested through 10.4 and B4 |
| 6 | `U_neg` against `U_A`; `P6-R2` / `P6-R4` registered, never run; ALBERT's imaginary-subspace alignment (unresolved) | B2's subspace question; S1, S4's tied arm |
| 7 | induction as a particle motif; nine `H-BRIDGE` rows, three built and unrun | B's starting design |
| 7d / 7e | the induction set, `L5H2` → `L7H8`, `L11H14`, MLP 6, the formation window, `L10H7`'s negative interaction, useful rank | B1–B6 |
| 8 | the same signatures on 70m | B's generality |
| 9 | LayerNorm-γ and attention-logit (β) levers for making crests and troughs | after B4 |
| 10 | §4 | Step 1 |
