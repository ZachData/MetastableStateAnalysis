# Roadmap: what comes next, in what order, and when each piece is done

**Written:** 2026-10-10, docs-only session at the user's request ("map out everything that we
want to do and figure out the best way to actually get there … finish Phase 10 first"). It
replaces `docs/TRIAGE_2026-09.md` (renamed here; the 2026-09-29 triage is in
`git show d50d2e5:docs/TRIAGE_2026-09.md`). **Inputs:** `origin/main` `d50d2e5`: `STATE.md`, every
phase card (`docs/PHASES.md`), the 2026-09-29 triage, `p10_cluster_function/handoff-10.md`
Stages 3–7, `LESSONS.md`, `docs/readings/2609.28448v1` (Gao–Yang–Chen, repulsive self-attention).
**Nothing was run.** Tier 1: a plan, not a decision; the user's decisions are §10.

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

## 6. Track A: the anti-collapse force

Kept from the triage (its §4.1 and §6a); what changed since 2026-09-29 is that β is decided
(Blocked 9: 3.5 [1.6, 5.6], unit LN1) and 1e's U2 split each block's update into attention, MLP,
sink and bias at that β. **Opened 2026-10-14: `trackA_force/design-A.md`** (A1: exit condition,
a 3-unit budget, the instrument, its gate, the reading rule), scan `trackA_force/lit-A.md`.

| # | unit | what | done when |
|---|---|---|---|
| A1 | β per head × checkpoint; `T_eff` against `t*` (`P-gamma2`); H-WARP (from Phase 10 at close: F0b, the `beta_eff` slope; from 1e at close: is one well at β 3.5 what 2312.10794's thresholds predict at n ≈ 2,000, d = 1,024) | does the trained net integrate far enough to collapse at all? | `P-gamma2` has a number, or a recorded reason it cannot. **Done 2026-10-15** (5 units): below t* at all 18 steps on `T_proj`, H-WARP refused (`trackA_force/status-A.md` "A1 on one page (closed 2026-10-15)") |
| A2 | **first item (Blocked 54 (a), the user's, 2026-10-16):** refit A1a's per-head β at 143000 by M1's cross-entropy (or keep heads with A1a's `partial_r2` above a floor placed first) and see whether A1's layer medians move past its bracket (`trackA_force/status-A.md` Parked, "A1a's β sign bias"); then per-component dissipation at the measured β, then mean-ablation (from Phase 10 at close: `attention-10.md`'s row A5, paid against received attention, the sink's part; from 1e at close: the late sum-over-heads ascent, cooperating heads or a few large ones; the sink's value ≤ 0.05 of the shared update; a second sink on a newline, so whether "sink-only" means any fixed key; from A1 at close: which components carry the >90 % off-field motion, attention's along-field step cancelled by the MLP at 1000–8000, block 23's step against the field at 143000, whether `T_attn_proj` is drift-robust, the long passages as an input choice, `a1a.ln_from`'s duplicate) | which heads and MLPs push apart, from which step; ablate the top ones: does energy turn monotone, do tokens re-cluster, at what loss? (`PROJECT.md` §3.8.3's exact split) | "these N components carry the anti-collapse force" is supported or refused |
| A3 | one timeline (from Phase 10 at close: R6f's embedding move at 512 → 2000 against the n-gram stage; from 1e at close: U2's regime change at 2000–4000 beside the OV turn; from A1 at close: step 16's collapse at the last block in 3 of 7 prompts without the field's integration; 1e's one-well question, the causal simulator's trajectory from the measured start at each step's β) | the energy break, the OV repulsive phase, the induction formation window, n-gram milestones (triage M2) and weights-only D1 / D3 / D4 (`FUTURE_IDEAS.md`) on the same 18 checkpoints | the curves are side by side, with their alignment or misalignment stated |
| A4 | claim (a) and the Blog 1 retests | §5 rows 3–5 | adjudicated or recorded as unadjudicable |
| A5 | generality | register one prediction (e.g. "the energy break falls in [128, 512]") on the rung §10.7 picks or on PolyPythias seeds, then run it | the project's first adjudication |

Risks (from the triage, still open): Pythia's LR warmup overlaps 512 and 1000–2000; sink and
compression prior art (`2510.06477`); literature trigger 1 before A2's design freezes.
**Models on disk** (`/challenge-pr` on #185, finding 6): Hugging Face downloads are blocked from
this machine, and the PolyPythias seeds are cached only at step 0 and 143000, so A5 on seeds
needs steps 128–512 fetched elsewhere first; ALBERT (B0′) and the CRFM GPT-2 checkpoints (B6)
are not cached at all.

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
| B2 | spectra from the weights (from Phase 10 at close: `⟨Qx, Ky⟩` per head, F18's `V`-spectrum atlas, `attention-10.md`'s A3) | each member's symmetrised OV and QK spectra across checkpoints; dominant negative eigenpairs (the d = 2 reduction); principal angles between members (7e's free item); 2d's gradient-flow regime test per member; Phase 6's `U_neg` / `U_A` split on the circuit's subspace | a per-member table across training |
| B2a | the 2D spike (the user's, 2026-10-14; `trackA_force/design-A.md` "Frozen 2026-10-14"): for each named head (`L5H2`, the induction heads, `L10H7`, random reference heads), the smallest subspace whose reduced dynamics reproduce its real attention; d = 2 is the paper's circle case. Three units: weights only with a planted positive control and a null; real tokens at 143000; the heads that pass over 18 steps. Needs A1a's β and step; runs before A1b or here, the slot picked at A1a's close. **Opened 2026-10-15: `trackB_circuit/design-B2a.md`** (not frozen, Blocked 50; the reduction is 2609.28448 §6.1, not 2604.01978 as first cited; scan `lit-B.md` "B2a") | does a head's particle dynamics live in a small subspace, and how small? | a faithful D per named head against the null (3 units) |
| B3 | routing (from Phase 10 at close: `attention-10.md`'s A3, per head joined to 7d's causal sweep) | effective attention partners and per-head β against the paper's condensation thresholds (does a head condense when it forms?); the light-cone test: nudge token j, does the response spread to tokens whose induction partner is j, against matched others? | both have a number and a direction |
| B4 | particles to function | nudge members' eigen-directions; read the induction loss on repeated tokens | a particle-level move is tied to the circuit's output, or not |
| B5 | one head across training | `L5H2` or `L11H14` over the 18 checkpoints with B1's signature; onto A3's timeline | tracked, or a recorded reason it cannot be |
| B6 | copy suppression | GPT-2 small `L10H7`, whose OV is dominated by negative diagonal values (`2310.04625`; `p2_eigenspectra/lit-2.md` §1.2): a trained head with a repulsive V, the real-model analogue of the paper's `V = −I`. On 410m, `L10H7` (same index, unrelated) interacts negatively with the induction set at every checkpoint (`PROJECT.md` "`L10H7` is the dissociation"). GPT-2 small training checkpoints exist from Stanford CRFM's Mistral runs (from memory; to verify) | B0–B4 repeated on it |

Later, only when a B unit asks for them: SAE features (Phase 3's crosscoder aligned with V at
chance, so expect little), the low-rank autoencoder (Phase 4, ALBERT/GPT-2, revivable), bigram /
n-gram directions (needs triage M2's producer), LLC on 70m / 160m (dropped for 410m on compute;
triage M3), own small models (triage M1).

## 8. Track C: write-ups

| # | paper | inputs | when |
|---|---|---|---|
| C1 | "What counts as a token cluster?", a null-model audit | 1d, Phase 10 §1, F13 (10.2); the 8 items carried to C1 at Phase 10's close, and 1e's (c3x purity in the centred frame against any proximity partition), each needed only if C1 quotes the reading it refines (`p10_cluster_function/handoff-10.md` "At close (10.5)") | after Phase 10 closes |
| C2 | the anti-collapse force along Pythia's training axis | Track A | after A2 and A5 |
| C3 | particles in a known circuit | Track B | after B4 |

## 9. Order and budget

| order | work | units (estimates) | gate |
|---|---|---|---|
| 1 | Step 1, finish Phase 10 (**done 2026-10-11**, 7 units: 10.1 one, 10.2 three, 10.4 two, 10.5 one; 10.3 dropped) | ≤ 5 | §4's exit condition |
| 1′ | Step 2's free items, beside it | 1–2 | — |
| 2 | B0 + B1 (the harness and the positive control) (**done 2026-10-14**, 3 units with the design: B1 passes, B continues on these instruments; `trackB_circuit/status-B.md` "B1 (2026-10-14)") | 2–3 | B1's verdict decides whether B continues on these instruments |
| 3 | A1, A2 and B2, B3 interleaved | 4–6 | each unit's own "done when" |
| 4 | A3 (the joint timeline), B4, B5 | 3–4 | — |
| 5 | A5 (registration, generality), C1 / C2 | — | §10.7 |

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
   compete for it.

## 11. What each phase gives the plan

| phase | what to reuse | goes to |
|---|---|---|
| 1 | energy, plateaus, Fiedler, effective rank instruments | A |
| 1b | one cone, no antipodal split (background); `--n-null` rerun open | A (cheap) |
| 1c | β, `T_eff`, γ_β; `P-gamma1/2` (blocked on per-run β, now measurable) | A1 |
| 1d | null-model audit, Gaussian null, float64 distances, the working definition | C1; every instrument |
| 1e | the field `φ_β`, U2's attention / MLP / per-head split, wells | A2; B0's energy readout |
| 2 | V's attractive / repulsive eigen-subspaces; the OV repulsive phase at 1000–2000; Study A (V's mixed sign causal in GPT-era models) | A3, B2 |
| 2b | rotation inert (headline withdrawn as an identity) | not pursued |
| 2d | operator regime (QK symmetric, V = QK); `P-T1` / `P-M1` registered and calibrated | A (D1 is the same question); B2 per member |
| 3 | crosscoder directions align with V at chance | a caution for SAE work in B |
| 4 | dense low-rank AE finds V-attractive directions in ALBERT | optional instrument in B |
| 5 / 5b | one cluster end to end; manifold steering | not pursued (B goes head-first instead) |
| 5c | the unclustered population and the rank budget; its flip is superseded | the "unclustered tokens do the work" idea, tested through 10.4 and B4 |
| 6 | `U_neg` against `U_A`; `P6-R2` / `P6-R4` registered, never run | B2's subspace question |
| 7 | induction as a particle motif; nine `H-BRIDGE` rows, three built and unrun | B's starting design |
| 7d / 7e | the induction set, `L5H2` → `L7H8`, `L11H14`, MLP 6, the formation window, `L10H7`'s negative interaction, useful rank | B1–B6 |
| 8 | the same signatures on 70m | B's generality |
| 9 | LayerNorm-γ and attention-logit (β) levers for making crests and troughs | after B4 |
| 10 | §4 | Step 1 |
