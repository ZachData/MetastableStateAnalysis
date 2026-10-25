<!-- trackB_circuit/design-B7.md -->
# Track B — design: B7, routing activity across depth

Opened 2026-10-25 at the user's request ("Open B7"). Row: `docs/ROADMAP.md` §7 B7 (set
2026-10-24 from 2609.28448). Scan: `trackB_circuit/lit-B.md` "B7". Math:
`tools/math_checks/b7_partner_persistence.py` (7 checks pass). Tier 1 throughout: exploratory,
unregistered; 410m is an exploration rung (`p8_scale_ladder/design-8.md`), so nothing here can
score a registered row. **Frozen 2026-10-25 as revised, 3 units + close-out (Blocked 110 (a), the
user's).** Unit 1's `b7_activity.py` docstring records ten choices the design left open, made
before any output.
**Revised after `/challenge-pr` on #271 (accept with changes) before the freeze:** the next-layer
head is matched leave-one-prompt-out, and N0 is a row-valid offset chance within position bins,
since the first proposal's best-of-16 match and pooled-profile N0 put a pure reshuffle 0.007
above N0 (1); the descriptions are an ordered partition with "positional, not separable" (2);
3 units (3); representation-space neighbour persistence beside (4); > 1/2, and checks 2 and 6 (5).
**Revised at unit 1b (Blocked 111 (a), the user's, 2026-10-25), before any real cell was read:**
`/challenge-pr` on #272 (finding 1) showed the frozen N0 counted each row's own agreeing pair, so it
rose with persistence by about P/m on m rows a bin (0.27–0.34 under G1 at the 50-row entry). N0 is
now over pairs of different rows (check 7); G1 and G3 also run at the minimum entry; G2 is read on
its pooled deviation; findings 2, 4–6 are code (`b7_activity.py` docstring, choices 11–17). The
rows below say "as revised" where they changed.
**Revised at unit 1c (Blocked 112 (a), the user's, 2026-10-25), before any real cell was read:**
unit 1b's re-gate failed G1 at the minimum entry under the 8000 and 143000 sinks: with identical
heads the leave-one-out h\* picks the head with fewest rows in the held-out prompt, so no head
reached 50 rows. A head now enters on its joint rows with h\* chosen on all 7 prompts, used for
entry only; P, N0, N1 and the intervals stay leave-one-out, and each head's leave-one-out rows are
printed beside its cell (`b7_activity.py` choices 18–19). The budget goes from 3 to 4 units, on the
record.

## What B7 asks

2609.28448 keeps three properties apart (l. 196–209): **condensation** (how a row's weight is
spread; B3 read it as α), **clustering** (where tokens sit; 1d, 1e) and **temporal activity** (how
the configuration moves with depth), which nothing here has read. "Condensed attention can remain
active through persistent rerouting." ROADMAP's row: at a fixed step, how often a token's top
attention partner changes from one layer to the next, so that "condensed and rerouting" and
"condensed and frozen" are told apart, with B3's α beside. Done when: a switch rate per layer and
step against the nulls, as description.

**The inputs cover it with no forward pass** (checked 2026-10-25): A1a's run directories
(`trackA_force/a1a.py` `RUNS`, the ones B3 read) hold `attentions.npz` `(24, 16, n, n)` float32 for
all 18 steps × 7 v1 prompts (none missing), and `activations.npz` `(25, n, 1024)`, unit norm.

## Exit condition and budget

- **Exit:** the cell table below over 23 layer pairs × 18 steps, each cell with its counts,
  intervals and the input named, after the gate passes. "Rerouting everywhere" or "nothing
  readable before step 512" are verdicts, not failures.
- **Budget: 4 units on the record** (3 at the freeze; unit 1c added by Blocked 112 (a)), this design as unit 0, then a close-out page (writing only).
  Unit 1 builds `trackB_circuit/b7_activity.py` and runs the gate; unit 1b is a reserve, spent
  only if an arm fails (B2a, A2 and B3 each overran on one, `LESSONS.md` 10); unit 2 reads the
  18 steps. Alternative: 2 units, no reserve. At the budget B7 closes at what it knows unless you
  extend it on the record.
- **Compute:** reads stored arrays only, about 19 GB of attention; I/O-bound. Torch on the GPU as
  B3 (`b3_route._load_attn`), CPU allowed, since the operations are argmax and counting, not
  float-sensitive.

## The math

`tools/math_checks/b7_partner_persistence.py` (`sympy` + numerics, 7 checks pass; check 7, unit 1b:
profiles that keep each row's own pair have Σ p q with mean c + (P − c)/m, pairs of different rows
mean c, and at 50 rows a frozen map puts the first at 0.306 against 0.0084 reshuffled, the second
at 0.0088 against 0.0083): (1) on
diffuse rows the two obvious measures disagree in opposite directions: independent uniform
argmaxes agree with probability 1/n (switch rate ≈ 1, "rerouting"), while two uniform rows have
soft overlap 1 − TV = 1 ("frozen"); independent s = 1 Gaussian-logit rows at n = 256 still overlap
0.48. (2) A row whose top share exceeds 1/2 has a unique argmax. (3) Two independent partners
drawn from offset profiles p, q agree with probability c = Σ p_k q_k, which is 1 for two
previous-token layers. (4) If a row keeps its partner with probability r per layer and otherwise
redraws (collision probability c), the lag-k agreement is **P(k) = r^k + (1 − r^k) c**: frozen
(r = 1) agrees with distant layers as much as with the next, so **a shuffled-layer null sees drift
but not freezing**; reshuffled (r = 0) sits at c; r = (P(1) − c)/(1 − c). A period-two routing has
P(1) = c, P(2) = 1. (5) D on unit states lies in [0, 2]; random directions in 1024 dimensions give
√2. (6) **The nulls at the real design points** (the 7 prompts' n, 16 heads, a pure reshuffle):
the per-row chance is 0.0085 (rows pooled); a best-of-16 match chosen on the read prompts gives P
0.0118, and an offset profile pooled over all positions gives c 0.0043 (`/challenge-pr` on #271,
finding 1); matching leave-one-prompt-out gives P 0.0084, and the row-valid offset chance within
32-position bins gives N0 0.0089, asserted within −3 % to +15 % of chance (conservative: it
leans to "rerouting").
**What it does not prove:** that 410m's rows follow check 4's chain (the real lag profile is
read, not fitted); that a kept partner in an untied net is one geometry persisting rather than
two layers' weights agreeing (Sec. V B); that top-1 is the right summary of a row whose mass sits
on 2–3 partners (the soft overlap on condensed rows is beside).

## Objects, fixed here

| object | definition |
|---|---|
| attention, rows | B3's: stored `attentions.npz`, query i ≥ 16, **position 0 dropped as key and query, the row renormalised over keys 1..i** (Ã); rows with non-sink mass `w_i = 1 − A_i0 < 0.05` are out (B3's `W_MIN`) |
| a condensed row | Ã's top share > **0.5** *(placed)*; its **partner** π(i) is that key (unique, check 2) and its **offset** i − π(i) |
| per-head persistence P (primary) | for head (l, h): the fraction of rows i condensed in both (l, h) and (l + 1, h\*) with the same partner, where **h\* is the head of layer l + 1 with the most agreement with (l, h)** (2110.06821's best-match; head indices carry nothing across layers), **chosen leave-one-prompt-out**: for each prompt q, h\* is chosen on the other 6 and P counted on q, then the 7 folds are pooled (so the max over 16 does not inflate P, check 6). A head enters with ≥ **50** *(placed)* jointly condensed rows ~~over the 7 folds~~ *(as revised, unit 1c)* with the h\* of all 7 prompts, used for entry only; its weight in P is its leave-one-out rows, printed beside. The cell value for (l → l + 1, step) is the mean over the entering heads, weighted by their jointly condensed row counts |
| pooled persistence (beside) | the layer's row: the mean of its 16 heads' Ã weighted by `w_i`, renormalised; condensed and P as above, one value per layer pair |
| N1, shuffled-layer null | the same statistic, leave-one-prompt-out best-match included, between l and every layer l′ with \|l′ − l\| ≥ 2 (both directions in depth), averaged. Excess `P − N1` > 0 is **depth-local persistence** (drift) |
| N0, offset-profile null (as revised) | for each entering pair (l, h) → (l + 1, h\*) and each read row i: c_i is the share of ordered pairs (j, j′) of **different** jointly condensed rows in i's **32-position bin** *(placed)* over all 7 prompts, j's offset in (l, h) and j′'s in (l + 1, h\*) both valid at i (0..i − 1), whose offsets agree; N0 is the mean of c_i over the read rows, weighted as P; a row with no such pair is left out of N0 and counted. Excess `P − N0` > 0 is **persistence beyond position**. Profiles pooled over all positions are biased low (check 6); profiles that keep each row's own pair are biased up by about P/m (check 7; the frozen definition, Σ_k p(k) q(k), until unit 1b) |
| step 0 | the same statistics at step 0 where rows pass; expected mostly unreadable (near-uniform attention to step 256, B2a). Step 0 is the reference for `D_ℓ` (check 1) |
| P_2 (beside) | P at lag 2 (l → l + 2, best-match), so a two-cycle (lag-1 at N0, lag-2 high) is not read as rerouting |
| D_1, D_2 (beside) | the paper's `D_ℓ` (l. 482–490) on `activations.npz` (unit states; index l is the residual attention layer l reads, B3's convention), position 0 dropped, mean over tokens and prompts; per layer and step |
| soft overlap (beside) | 1 − ½‖Ã_{l,h,i} − Ã_{l+1,h\*,i}‖₁ on the jointly condensed rows |
| neighbour persistence (beside) | the paper's notion of partner: each token i's nearest earlier token by cosine among 1..i − 1 in the unit states at index l, and whether it is the same at l + 1; per layer pair and step, with N1 as for P (distant layers). Beside, not primary: ROADMAP's question is routing, and 410m's states share large common directions (C5's massive tokens), so cosine neighbours may follow those rather than what attention reads |
| condensed share (beside) | the fraction of rows (`w_i ≥ 0.05`) condensed, per head and per layer: whether the cell routes sparsely at all |
| α (beside) | B3's per-head α (`b3_route.alpha_of`, weighted), its layer median, read from B3's stored rows (`data/b3/2026-10-18-unit1/rows`) with B3's code hash checked |
| interval | percentile 2.5–97.5 % over 2000 *(placed)* bootstrap draws of the 7 prompts (the folds and h\* re-done in each draw) |
| steps, prompts | B2a's 18 steps; the 7 scored v1 prompts (hash pinned at run time, `core/prompts.py`); the 12 held-out (`core/holdout.py`) are not read |

## The cells (fixed before any output)

Family: the 23 layer pairs × 18 steps; the statistic is the per-head P against N1 and N0 with
their intervals. Each cell gets exactly one description, the first that applies (from check 4):

| order | description | when |
|---|---|---|
| 1 | not read | fewer than 2 entering heads *(placed)*; the count is printed instead |
| 2 | positional, not separable | N0 ≥ 0.5 *(placed)*: the offset profiles alone predict most of the agreement (two previous-token layers), so persistence beyond position cannot be read |
| 3 | rerouting | `P − N0`'s interval does not lie above 0 |
| 4 | depth-local persistence | `P − N1`'s interval above 0 |
| 5 | frozen across depth | `P − N1`'s interval contains 0 |
| 6 | persistent, less than at distant layers | `P − N1`'s interval below 0 |

Beside, not a class: **a two-cycle candidate** is a cell read "rerouting" at lag 1 whose
`P_2 − N0` interval lies above 0.

These are descriptions of cells, not labels a head gets; the 23 × 18 grid is read as a picture
(where in depth, from which step), with no count of cells tested against a rate. A cell's reading
is about its condensed rows only, and its condensed share is printed beside it.

## Unit 1's gate (before any real cell; CLAUDE.md "Calibration gate")

Planted inputs keep the real design points: each prompt's real n, every row's real sink mass
`A_i0` at steps 512, 8000, 143000, the `(24, 16)` shape and the 7 prompts. A planted row puts
0.8 of its non-sink mass on its planted partner and spreads the rest uniformly over keys 1..i.
Partners are drawn uniformly from keys 1..i − 2 (never i − 1) unless the arm says otherwise.

| arm | construction | passes when |
|---|---|---|
| G0 | inputs: 18 × 7 attention and activation files present; shapes `(24, 16, n, n)`, `(25, n, 1024)`; rows sum to 1 within 1e-5; states unit norm within 1e-5; B3's rows present with their code hash | all hold, else the gap is recorded before anything is read |
| G1 frozen | one partner map π(i), the same at every layer and head; at full rows and (as revised) at the minimum entry (G7's thinning) | P ≥ 0.99, N1 ≥ 0.99, `P − N0` ≥ 0.9; every cell "frozen across depth" |
| G2 reshuffled | an independent π per layer and head | (as revised) pooled over a sink step's cells, rows weighted, \|Σ P − Σ N0\| ≤ 0.25 · Σ N0 (N0 is about 0.009 here, check 6, so an absolute band would check nothing; per cell the band also counted sampling noise, finding 3), the per-cell extreme beside; every cell "rerouting" |
| G3 drift | each row keeps its partner with r = 0.8 per layer, else redraws; at full rows and (as revised) at the minimum entry | (P − N0)/(1 − N0) within 0.8 ± 0.03 at full rows, and at the minimum entry within ± 4 binomial standard errors on the cell's joint rows (± 0.057 at 800); every cell "depth-local persistence" |
| G4 positional | every head previous-token (π(i) = i − 1) | N0 ≥ 0.99; every cell "positional, not separable" |
| G5 period two | two maps alternating by layer | lag-1 cells "rerouting", every cell a two-cycle candidate, `P_2` ≥ 0.99 |
| G6 diffuse | uniform rows | no row condensed; every cell "not read"; no P printed (refuse rather than degrade) |
| G7 false persistence | G2 at the full row counts, and at the minimum entry (50 jointly condensed rows per head: rows thinned at random) | a cell read anything but "rerouting" in ≤ 5 % of cells at both, over 20 seeds |
| G8 planted on real | real attention at 143000; in layers 10 and 11, heads 0–3 overwritten with one shared fixed partner map | cell 10 → 11 "frozen across depth" or "depth-local", with those 4 heads' P ≥ 0.95 (h\* found); every other cell's P within 0.01 of the unplanted run, except cells touching layers 10–11 |
| G9 D | D_1 on a planted state sequence: the same states at every layer (D = 0), and antipodal flips (D = 2) | within 1e-6 |

An arm that fails changes the constants, with the reason recorded, before any real cell is read
(B2's rule 6). G1–G6 also run as code tests on a small synthetic cloud.

## Seen before the cells were fixed

The array shapes, file presence and unit norms above; one stored residual norm (position 0 at
layer 12 of `hdbscan_code`, 143000: 986, the sink); B3's recorded results (`status-B.md` "B3 on
one page"), B2a's (attention near uniform to step 256, positional heads from 512), B2's. No
partner, persistence, condensed share or `D_ℓ` value has been computed on real attention. The
reviewer's and Claude's simulations of G2 on synthetic partners (check 6) have been seen.

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen (Blocked 110) |
| 1 | `trackB_circuit/b7_activity.py`: the objects above, G0–G9, code tests | every arm passes, or a recorded change |
| 1b | reserve: a failed arm's revision, re-gated (spent only if unit 1 fails). **Spent** on N0's revision (Blocked 111 (a)) | every arm passes |
| 1c | entry on the all-prompt h\* (Blocked 112 (a)), the pooled r beside G3, re-gated | every arm passes |
| 2 | the 18 steps × 23 layer pairs, the beside columns, the cell picture, with the input named | the table is written with its input |

## Fences

- **Description, not dynamics.** 410m is untied: layer l + 1 has its own Q and K, so a kept partner
  can be two layers' weights agreeing, not one geometry persisting (2609.28448 Sec. V B). No cell
  says "the paper's active phase". The separation is S4's tied arm, carried there at close.
- **No head identity across layers.** "Per head" means best-matched (2110.06821); nothing says
  head h of layer l becomes head h of layer l + 1.
- **Condensed rows only.** On diffuse rows the rate has no meaning (check 1); the condensed share
  says how much of a layer a cell describes.
- **Partners in sequence space.** The paper's partners are neighbours in representation space; B7's
  are attention argmaxes over positions, and N0 takes the positional part out, not all of it.
- **Stored attention, 7 v1 prompts, 18 steps, no forward pass.** The held-out 12 are not read.

## Worth challenging (for the freeze)

1. **Per-head best-match as primary, pooled beside.** ROADMAP named both. Pooling 16 heads makes
   a layer row condensed only when most heads agree, which may leave few rows; best-match keeps
   heads but takes a max over 16, removed by choosing it leave-one-prompt-out (check 6) and
   calibrated by G7. The alternatives are the pooled row as primary, or a max taken under every
   null (the first proposal; N0 did not take it, finding 1).
2. **N0 added, step 0 demoted.** ROADMAP named a shuffled-layer null and step 0. Check 4 shows the
   shuffled-layer null alone cannot see freezing, and check 1 that step 0's near-uniform rows say
   nothing about partners; so N0 (offset profiles) is the second null and step 0 is `D_ℓ`'s
   reference. The alternative is ROADMAP's pair as written, with step 0 read where rows pass.
   N0 treats a bin's offset profile as all of "positional": a head whose partner depends on
   content and offset together is only partly removed. A within-bin offset permutation was
   tried in simulation and ran about 10 % below chance (permuted offsets fall outside a row's
   range), so the analytic row-valid chance, which errs above, was kept.
3. **Top share > 0.5 as "condensed".** It makes the partner unique without a tie rule. A head
   splitting its mass over two partners (0.45 / 0.45) is out. The alternative is N_eff ≤ 2 with
   a top-2 set match; or top share ≥ 0.3 with ties broken by index.
4. **`D_ℓ` beside, not primary.** It is the paper's activity measure. It is beside because in
   410m a layer's displacement mixes attention and the MLP, and ROADMAP's question is the routing
   side. The alternative is `D_1`, `D_2` as the family and the switch rate as its mechanism.
5. **Budget 3 with a reserve**, with 2 as the alternative above (the first proposal; finding 3).
6. **No cell count against a rate.** The grid is read as a picture; the alternative is a count of
   "frozen" cells against G2's false rate, which would make B7 a test it was not set up to be.
7. **"Positional, not separable" at N0 ≥ 0.5.** Below it, a previous-token-heavy pair is read
   against N0 as usual; above it, nothing is said. The alternative is no positional class, with
   G4 read as "rerouting" (the first proposal: the opposite of what G4 plants, finding 2).
8. **Neighbour persistence beside, not in the family** (finding 4): the paper's partners are
   representation-space neighbours, and the column is cheap from the stored states; it is beside
   because cosine neighbours in 410m may follow shared large directions, not what attention reads.
