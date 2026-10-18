<!-- trackB_circuit/design-B3.md -->
# Track B — design: B3, routing

Opened 2026-10-18 by `docs/ROADMAP.md` §9 order 3 (B3 next after B2's close, Blocked 64 (a)). Row:
`docs/ROADMAP.md` §7 B3. Scan: `trackB_circuit/lit-B.md` "B3". Math:
`tools/math_checks/b3_condensation_exponent.py`. Tier 1 throughout: exploratory, unregistered;
410m is an exploration rung (`p8_scale_ladder/design-8.md`), so nothing here can score a registered
row, and B3 reads no other rung. **Proposed, not frozen** (Blocked 65).
**Revised after `/challenge-pr` on #223 (accept with changes, all six findings taken) before the
freeze:** R2's onset is read against the same step's population, since step 0's band sits at α ≈ 1
and its edge marks only leaving uniform (1); α weights each row by its non-sink mass (2); type is
read at the members' measured offset, with the nearest non-members in type score as the pool (3);
L's 143000 read becomes a positive control (L0), and the family's L reading is the members'
ablation (4); the seen-before list is here (5); the numbers are corrected (6).

## What B3 asks

ROADMAP's row, in two parts, with "done when: both have a number and a direction":

- **R, routing.** How many partners does each head attend to, how does that number grow with the
  context, and does a head's routing sharpen when it forms ("does a head condense when it forms?")?
- **L, the light cone.** Nudge token j: does the response spread to the tokens whose induction
  partner is j, more than to matched others?

Carried into B3 (each must be done or dropped at close):

| item | from | B3 takes it as |
|---|---|---|
| `attention-10.md`'s A3: received attention per head, joined to 7d's 384-head causal sweep and `attention_entropy_per_head` | Phase 10 at close; B2 moved it here | **R's descriptive A3 row** (unit 1): per head, received attention on Phase 10's populations, rank-correlated with 7d's solo effect (`redundancy_catalog.json`, step 16000). Beside, not in the family |
| M1 and `fe_full` give opposite per-head β signs mid-depth on code and LaTeX; pick and justify the estimator before reading β against thresholds | A2 at close (`trackA_force/status-A.md` "A2 unit 2") | **β beside, no threshold read** ("The math", "Worth challenging" 1): the reading is β-free; both estimators are shown, prose prompts only for the per-head column, the disagreement kept as A2 left it |
| a head-type control for B2's C3 (QK symmetry): re-rank with 7d's previous-token and induction non-members as the null | B2 at close (`status-B.md` "B2 on one page") | **C3′** (unit 1): B2's T for C3, re-ranked with a type-matched null instead of B2's layer-matched one |

## Exit condition and budget

- **Exit:** R's member statistics and onset table over the 18 steps, C3′'s rank, L's partner
  ratio at 143000 (L0) with its steps and the members' ablation rank (L1), each with its sign and
  the input named. "Members do not differ"
  and "no cone along partners" are verdicts, not failures. A3's row and β's columns are present.
- **Budget (proposed): 3 units on the record**, this design as unit 0, then a close-out page
  (writing only, as B2a, A2 and B2 closed). Unit 1 reads stored attention and runs one short
  forward pass per step for head types; unit 2 is B0's harness, the one new GPU batch.
  Alternative: 4 units, splitting L's gate from L's read (B2a and A2 overran when a gate failed,
  `LESSONS.md` 10). At the budget B3 closes at what it knows unless you extend it on the record.

## The math

`tools/math_checks/b3_condensation_exponent.py` (`sympy` + numerics, 3 checks pass):
(1) uniform attention over n keys has N_eff = n, and k fixed partners N_eff = k, so the slope
α = d log N_eff / d log n is 1 for diffuse and 0 for condensed routing; (2) i.i.d. logits of
fixed spread s have N_eff → n e^{−s²}, so α → 1 at any fixed s (the paper's bounded-logit bound),
**but at s = 2 the slope over n = 16–256 is about 0.6 (asserted < 0.8), and over n = 4096–65536
it is higher by more than 0.1 (asserted; 0.84–0.87 by seed)**:
a sharp head with no growing gap can show a low α at B3's context lengths; (3) the paper's d = 2
window holds N (√2β)^{−1/2} tokens, O(1) iff √2β = χN², which is the n^{2/(d−1)} critical scale of
2605.08505 at d = 2. **What it does not prove:** that a finite-n α is the limit (it is the
crossover the paper says finite N gives); that a low α is the paper's condensation (the paper
routes by representation-space neighbours; a previous-token head routes by position); any
per-head threshold (n^{2/(d−1)} needs the local key dimension, not estimated here).

**Why no "β against the thresholds".** The paper's condensation is a limit with no finite-N line;
its d = N line β_G = O(1) is a finite-size scan of `Q = K = I`, `V = −I`, recurrent, non-causal;
2605.08505's n^{2/(d−1)} and 2510.05554's log n are critical scales under particular key laws. None
places a trained 64-dimensional rotary head. And A2 left the two estimators disagreeing in sign
on code and LaTeX. So B3 reads condensation from the attention itself, and β is beside.

## Objects, fixed here

| object | definition |
|---|---|
| attention | stored `attentions.npz`, `(24, 16, n, n)`, in A1a's run directories (`trackA_force/a1a.py` `RUNS`), the 18 steps × the 7 scored v1 prompts (hash pinned at run time, `core/prompts.py`; not the held-out 12, `core/holdout.py`) |
| a row | query i ≥ 16, **position 0 dropped as key and query and the row renormalised over keys 1..i** (the sink; B2a's convention); the sink share `A_i0` beside. The row's weight is its non-sink mass `w_i = 1 − A_i0`; rows with `w_i < 0.05` *(placed)* are out, their count reported per head and step. α describes a head **while it routes**: B2a records 47–93 % of the induction heads' and `L10H7`'s mass on position 0 at 143000 (`design-B2a.md`), so unweighted rows would read their logits while parked |
| N_eff,i, n_i | `1 / Σ_j Ã_ij²` over the renormalised row; n_i = i, the keys it can see |
| α of a head | weighted least-squares slope of log N_eff,i on log n_i over its rows (weights `w_i`), pooled over the 7 prompts; per-prompt slopes and the unweighted slope beside; for the four induction members, α on the repeated-token probe (the head types' pass) beside |
| logit spread (beside) | per row, the std of log Ã_ij over j (the logits' std, up to a row constant), and its slope on log n_i: a growing gap is the condensation route the math names |
| N_eff at 256 (beside) | median N_eff over rows with n_i ∈ [192, 320]: the number of partners at a fixed context |
| head types | per step, every head's **previous-token score** (`prev_token_profile.py`'s convention: mean attention i → i−1, all positions) and **induction score at the repo's offset** (query `N_REP + j` → key j, `fv_score.induction_scores`), on 7d's probe (16 sequences, `N_REP` = 96); one forward per step, GPU float32. The members' offset is on disk: `L7H8` puts 0.934 at offset 0 and 1.3e-5 at +1 (`data/analysis/what_l7h8_writes.json` `probe1_offsets`); the +1 score is beside only |
| type pools (C3′) | at each step ≥ 512, for each typed member the **5 non-members nearest its own type score** *(placed)* (previous-token for `L5H2`; induction for the four), so a pool never empties; for `L10H7` the non-members of layer 10. A fixed cut was dropped: at 0.4 the induction non-members number 3–4 at every stored step (`/challenge-pr` on #223, finding 3) |
| formation step of a member | first step where its own type score reaches half its maximum over the 18 steps: previous-token for `L5H2`, induction for `L7H8`, `L12H5`, `L8H6`, `L11H14`. `L10H7` has no type score and no formation step |
| members | B2's six: `L5H2`, `L7H8`, `L12H5`, `L8H6`, `L11H14`, `L10H7` |
| population, step-0 band | the 384 heads at the same step; 2.5–97.5 % of the 384 heads at step 0 |
| steps | B2a's 18 (0, 2, 4, …, 143000) |

## R: the readings (fixed before any member output)

| # | statistic | null | reads "members differ" at |
|---|---|---|---|
| R1 | `T_α = mean over members and steps ≥ 512 of |p − ½|`, p the member's α percentile in its step's population (B2's rule 2, on α) | 10,000 layer-matched sets (one non-member per member's layer), the same steps | rank ≥ 0.99 |
| R2 | **condenses when it forms**: the count, over the 5 members with a formation step, of members whose **α onset** (the first grid step from step 2 on where its α is at or below its step's population 10th percentile *(placed)*, and still there at the next grid step) is within one grid step of their formation step | 10,000 sets of 5 layer-matched non-members, each given its member's formation step | rank ≥ 0.99 |
| C3′ | B2's C3 statistic (`a(M_0)`, `T` over members and steps ≥ 512, B2's `table.json`, code `038b04f5c7cd`) | 10,000 **type-matched** sets: for each member one head drawn from its type pool at each step ("Objects") | rank ≥ 0.99: "C3 sets the members apart beyond head type"; below it: "C3 marks head type, not shown to mark the circuit" |

R1, R2, C3′ and L1 (below) are one family of four; 0.99 ≥ 1 − 0.05/4. Leans are not readings.
**Rank** is the share of null draws strictly below the observed value (ties count against).
**R2's null is the point of R2:** most heads leave uniform at 256–1000 (B2a, B2), so a member that
sharpens at its formation step is only notable if same-layer heads do not do so at that step.
Step 0's band could not serve: its lower edge sits at α ≈ 0.99, so leaving it marks only the first
departure from uniform, and a layer-matched head given a formation step of 1000 would hit with
probability ≈ 0.53–0.60 (finding 1). **R2's attainable rank** is 1 − Π_m p_m, p_m the null hit rate
for member m's layer and formation step; unit 1 reports it from the null before any member value
is read, and **if it is below 0.99, R2 reads descriptive only** (fixed here, not after the null).
**C3′'s depth:** the type pools are not layer-matched (the induction non-members sit in layers 1,
2, 6, 9 at 0.4; the nearest-k pool may too); B2's layer-matched rank stands beside, so C3′ and C3
together say type, depth or circuit, not one alone.

**Beside, not in the family:** each member's α, logit-spread slope and N_eff at 256 per step;
the population's α distribution per step against step 0's; the sink share; β per head from
A1a's 2,688 stored `fe_full` βs and A2's M1 fits at 143000 (`data/trackA/a2_beta_2026-10-16/`),
**on the five prose prompts only** where the two agree in sign (head-level correlation 0.73–0.82),
code and LaTeX shown apart; A3's row: per head and step ≥ 512, received attention (position 0 out)
on c3x members against the rest (Phase 10's populations, `p10_cluster_function/status-10.md`),
its Spearman correlation with 7d's solo effect over the 384 heads at 16000, and the stored
`attention_entropy_per_head` beside α (the same attention, another summary).

## L: the light cone (fixed before any output)

**Inputs.** 32 sequences of 64 uniform random token ids in [1000, 30000) repeated twice (n = 128),
seeded; and their **controls**, the same first halves followed by a fresh random second half (no
partners). Steps: the 18. GPU float32, B0's NUMERICS (`trackB_circuit/design-B.md` "B0").

**Partners.** For nudged token j in the first half, its partner is the second-half token i whose
members attend to j: i = j + 63 if the members' induction attention peaks at offset j + 1
(query `64 + j′` → key `j′ + 1`), i = j + 64 at offset j. The offset is the one unit 1 measures for
`L7H8`, `L8H6`, `L11H14`, `L12H5` at 143000 (majority; recorded before L reads anything).
**Matched non-partners:** the second-half tokens at i ± 2, ± 3, ± 4 in the same sequence (same
distance band, their partners elsewhere), and token i in the control sequence (same i − j, same
position, no partner).

**Nudge and read** (B0's harness, `core/nudge.py`): `resid_pre` of block 4 *(placed: before
`L5H2`'s layer, so both stages of the circuit lie downstream)* at token j, ±εd for K = 32 random
unit d, c = 0.03 primary (0.01, 0.1 beside); read the displacement `‖A‖/ε` of the final residual
at each later token. j over 16 positions in [8, 56) per sequence, seeded, the same at every step.

`ℓ = median over (sequence, j) of log(σ_partner / median σ_matched)`, the matched set the six
same-sequence neighbours (i ± 1 is out: block 4's nudge reaches j + 1's key through `L5H2`, so it
is a secondary partner).

| # | statistic | reads |
|---|---|---|
| L0 (positive control, not in the family) | ℓ at 143000 | must be > 0 with its 99 % bootstrap interval over the 32 sequences above 0, or the instrument fails (as B1 for B0). By construction the induction heads move j to its partner, so this is expected; it is the yardstick, not the test (finding 4) |
| L1 | **the cone goes through the members:** at 143000, `Δℓ = ℓ − ℓ_abl`, ℓ_abl with the four induction members' OV mean-ablated (mean over the probe's positions) | rank of Δℓ ≥ 0.99 against 200 sets of 4 layer-matched non-members ablated the same way |

**Beside:** ℓ against the control-sequence token (same i − j, no partner); ℓ per step and its
onset (first step whose 99 % interval is above 0, held at the next) against the members'
formation steps from R; a confinement share (response at the partner over the summed response at
every token after j + 1); the response by layer at the partner (`resid_pre` of each block from 5:
which block it first appears after). 7d found self-repair among the members' stand-ins
(`p7d_redundancy/status-7d.md` "Self-repair"), so a small Δℓ is a reading, not an instrument
failure. Readout at the final residual only for the family.

## Units 1 and 2's gates (before any member value; CLAUDE.md "Calibration gate")

| arm | construction | passes when |
|---|---|---|
| G0 | A1a's 18 × 7 attention files present, shape `(24, 16, n, n)`, rows sum to 1 within 1e-5; head types reproduce `prev_token_profile.json` at 4000, 16000, 143000 for all 384 heads and `fv_score.json`'s induction scores for its members | all present; max |Δ| ≤ 1e-4 (same probe, same device class); else the input is recorded and the gap explained before unit 1 reads anything |
| G1 | planted α on real rows: one non-member head's stored rows replaced by (i) uniform over keys 1..i, (ii) 2 fixed partners (i−1, i−2), (iii) i.i.d. Gaussian logits at s = 1, each with the head's own sink mass kept (so the weights are real) | α ∈ [0.97, 1.0], [−0.02, 0.02], ≥ 0.85 (the math's check 2). (ii) is exact by construction: a code test, not calibration |
| G2 | R1, R2, C3′, L1 false pass at their real pools: 200 random non-member sets (layer-matched for R1, R2, L1; drawn from the type pools for C3′) as "members" against the same null | rank ≥ 0.99 in ≤ 2 % of draws, per statistic |
| G3 | R2 planted onset: 5 non-member heads whose rows are step 0's before a chosen formation step and (ii)'s 2-partner pattern from it on | R2's count is 5 and its rank ≥ 0.99 |
| G4 | L planted routing: at 143000, a hook sets one non-member head of layer 8's attention (query i → key j) to mass 1 for a planted non-partner pair (i, j), on the control sequences | ℓ computed with that pair as "partner" exceeds 0 with its 99 % interval in ≥ 80 % of 50 seeds |
| G5 | L null: at step 0, and at 143000 on the control sequences with i as "partner" | ℓ's 99 % interval contains 0 in both |
| G6 | L1 planted: G4's planted head, ablated | Δℓ's rank ≥ 0.99 in ≥ 80 % of 50 seeds |

An arm that fails changes the constants, with the reason recorded, before any member value is
read (B2's rule 6).

## Seen before the readings were fixed

Before this design: B2a's 18-step table (attention near uniform to 256, positional heads from
512: `L5H2`, `L4H9`), B2a's sink shares (47–93 % for the induction heads and `L10H7` at 143000),
B1's onset (256, 512], B2's C3 ranks and member percentiles, the previous-token values of `L5H9`
(0.55–0.62) and `L4H9` (0.805), and `L7H8`'s offset profile (0.934 at 0). From `/challenge-pr` on
#223, before the revision: the population's α under the first proposal (step 0's 2.5th percentile
0.991; onset counts by step; the layer-matched hit rates 0.53–0.60), the induction non-members
above 0.4 at 4000, 16000, 143000 (`L1H15`, `L6H0`, `L9H9`, `L2H10`), and the slopes of check 2. The
reviewer also saw member α values in a scratch run and did not quote them; Claude has not seen them,
and the scratch output stays unopened until the freeze. No partner ratio or type score at the 18
steps has been computed.

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen by the user (Blocked 65) |
| 1 | `trackB_circuit/b3_route.py`: α and the beside columns over 18 steps × 384 heads from stored attention; head types per step (one forward per step); G0–G3; R1, R2, C3′; A3's row; β beside | every arm passes (or a recorded change), and R1, R2, C3′ ranked with the input named |
| 2 | `trackB_circuit/b3_cone.py` on B0's harness: G4–G6; L0 and L1 at 143000, ℓ over the 18 steps, the beside rows | the gate and L0 pass and L1 reads, with the input named |

## Fences

- **No per-head threshold.** β is beside; "condensed" and "diffuse" are not labels a head gets.
- **Partners from the circuit, not from representation space.** The paper's cone runs along
  representation-space neighbours under hard routing; L reads a sequence-space graph (induction)
  that is known to be real, so a cone here says the response follows routing, not that 410m is
  in the paper's regime.
- **Finite depth.** One forward pass; no Lyapunov exponent (the paper's §7: untied networks need
  finite-depth amplification).
- **A3 is descriptive.** Phase 10 is closed; its populations are read as Phase 10 fixed them, and
  no Phase 10 reading is re-opened.
- **v1's 7 prompts for R; random repeated tokens for L.** The 12 held-out prompts are not read.

## Worth challenging (for the freeze)

1. **β beside, not against a threshold** (ROADMAP's row asked for the latter; A2 asked that the
   estimator be picked first). The alternative is to estimate each head's local key dimension D
   and place `fe_full`'s β against n^{2/(D−1)}: one more unit, and an estimator of D with its own
   gate. Rejected for now because no threshold was derived for a causal, rotary, untied head.
2. **α as the routing number.** By the math's check 2, a sharp head can show α ≈ 0.6 with no
   growing gap; the logit-spread slope is beside to separate the two, but it is not in the family.
   The alternative primary is N_eff at a fixed n (a level, not a growth), which says "how many
   partners" but not "does it condense".
3. **R2's ±1 grid step** on an 18-step log grid (512 and 1000 are neighbours, so are 1000 and
   2000), the onset at the population's 10th percentile, and formation as half the maximum type
   score. All placed before any output; R2 may turn out descriptive by its attainable rank.
4. **Type pools as the 5 nearest non-members in type score** for C3′, not a cut (the cut left
   3–4 induction non-members); the pools are not depth-matched, so C3′ is read beside B2's C3.
5. **L's nudge site, block 4,** and the final-residual readout. Nudging after `L5H2` (block 6)
   would test the induction stage alone; the by-layer row beside shows where the response first
   reaches the partner. **L1 is the ablation**, not ℓ's onset or the confinement share, both
   beside: the ablation has a rank against random heads, the onset has no null.
6. **Partners by the members' measured offset**, read in unit 1 at 143000; `L7H8`'s 0.934 at
   offset 0 says the key is j (query `64 + j` → key j), so the partner of j is i = j + 64.
7. **Weighting rows by non-sink mass** (α while the head routes) over reading the induction
   members on repeated tokens only (beside).
8. **Budget 3**, with the alternative of 4 above.
