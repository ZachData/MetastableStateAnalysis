<!-- trackC_writeup/design-C4peg.md -->
# Track C — design: C4′ (e) and (g), no training

**Status: (e) frozen 2026-11-23 on Blocked 170 D1, D2 (a)** (Claude, on the user's "Let's just
continue here"; "Decided 2026-11-23" below) **and read in U1; (g) waits on D3, D4 (STATE Blocked
171).** Proposed 2026-11-23 (U0, writing only); revised after `/challenge-pr` on #366 (accept
with changes, findings 1–5 taken: G2's random sets and the four arms' head lists, E2's dilution
term, G1's scale floor, the C5 promise dropped, each lean's source marked); decisions D1–D4 are
STATE Blocked 170.**
Opened on Blocked 169, decided by the user 2026-11-23: "(e) and (g) first", then (d), (f). (e) and
(g) need no training and read the four arms C4 and C4′ (c) already trained, so they share one
design, as C4′ (a) and (b) did. Scan: `lit-C4peg.md`. **Nothing below has been computed.** Of the
inputs, the C4 records (e1, e1c, cos per state) and the heads ≥ 0.5 at 4000 have been seen. No
component split, no reading through W_U, no interpolation and no ablation has been run on any arm.

## What it is for

C2 5b showed that V nearly counts the layers where the tokens' mean norm ‖m‖² falls. C4 found
those falls one block earlier on F (T1) than on A0 (T2), and `P-SCH3` held this on the replicate.
Neither says **what the mean is** or **what writes it**. C4 and C4′ (c) also found that the two
schedules end on different induction heads, but not **whether the two end states compute
differently**. The ROADMAP rows (§8 C4′):

- **(e)** Read the mean's direction through the unembedding against the corpus's unigram
  log-frequencies. Measure each component's share of the mean and of its falls. Take the cosine
  between the arms' mean directions. Could give §4's "the count's drops are falls of the mean" a
  mechanism. The row also says it "feeds C5's 'when the mean becomes basis-aligned'". **Not taken
  here** (`/challenge-pr` on #366, finding 4): C5's question is alignment with the coordinate
  basis on 410m's seeds, and nothing below measures that. It stays with ROADMAP C5.
- **(g)** Compare F and A0 at 4000 by the loss along the line between them and by ablating each
  arm's induction heads. Could change whether "the schedule chooses the heads" matters beyond the
  heads' indices.

## Inputs (both units)

| item | value |
|---|---|
| arms | A0, F (`HDD_1TB/mets_data/c4/{A0,F}/ckpt/`), A0′, F′ (`…/c4pc/{A0p,Fp}/`), all 29 saved steps (`c4_train.SAVE_STEPS`); `pythia-70m` (released) and `pythia-70m-seed1` on Pythia's grid, cached |
| prompts, (e) | v1's 7 prompts, `c2_aniso.states`' tokenisation, float32 eager, no TF32, as C4's records |
| rows, (g) | `c4_train.HELD_OUT` (64 rows, 131 072 positions): read by none of the four arms, since π permutes rows below it (`c4pc_train` refuses otherwise) |
| unigram counts | the target tokens (positions 1–2048) of rows [0, `N_ROWS`) of C4's shard: every token both pairs trained on (π is a permutation of the same rows). About 8.4 × 10⁹ tokens, one CPU pass. `f(v)` = count / total; the vocabulary read is the tokens with count ≥ 1 |

## (e) The common direction (U1)

**The mean.** As in C2 and C4's records: at state l (0 the embeddings, 1–5 the block outputs, 6
after the final LN, as HF's `hidden_states` gives it), uᵢ = P(hᵢ)/‖P(hᵢ)‖, where P centres over the
512 coordinates (`layernorm_to_sphere`). m_l = the mean of uᵢ over a prompt's tokens, and
**m̄_l = the mean of m_l over the 7 prompts** (each prompt weighted equally). Per prompt, beside.

**E1, the direction through W_U.** At each checkpoint, z(x) = W_U (γ_f ⊙ x) for a direction x in
the residual space. W_U is that checkpoint's `embed_out` and γ_f its final LN weight. The final LN's
bias b_f is left out: it adds the constant W_U b_f, read separately as its own row. The statistic is
**r(x) = Spearman(z(x), log f)** over the vocabulary read.

| item | fixed as |
|---|---|
| the effect (planted, per checkpoint) | **u\*** = the least-squares solution of (W_U diag γ_f) u ≈ log f − mean (2609.02959's direction, at this checkpoint's W_U); r(u\*) is the ceiling a direction reaches |
| the null (planted, per checkpoint) | 1000 Gaussian directions in ℝ⁵¹² (seeded); q = their 99.9th percentile of r |
| **gate**, per checkpoint | readable iff r(u\*) > q **and** r(u\*) > the 99th percentile of r(u\*_π) over 100 permutations π of log f across tokens (a fitted direction finds some fit to anything) |
| a cell's label (arm, step, state) | **"aligned"** iff r(m̄_l) > q; **"anti-aligned"** iff r(m̄_l) < −q; else "not aligned"; "unreadable" if the gate fails. The fraction of the ceiling, r(m̄_l)/r(u\*), is reported beside, with no bar |
| beside | r(W_U b_f), the final LN bias's own; cos(m̄_l, γ_f-weighted u\*); r per prompt |

**E2, what writes the mean.** Pythia uses a parallel residual: h_{l+1} = h_l + a_l + f_l, the
attention and MLP outputs (biases included). P is linear, so with the weights wᵢ = 1/‖P(hᵢ^{l+1})‖ at
state l+1, m_{l+1} = m_old + m_a + m_f exactly. Here m_old = mean wᵢ P(hᵢ^l), and m_a, m_f are the same
means of P(aᵢ), P(fᵢ). For a transition T_l (state l → l+1), the change splits exactly:

  Δ‖m‖² = **R** + **A** + **M**, with
  R = ‖m_old‖² − ‖m_l‖² (renormalisation),
  A = 2⟨m_old, m_a⟩ + ‖m_a‖² + ⟨m_a, m_f⟩,
  M = 2⟨m_old, m_f⟩ + ‖m_f‖² + ⟨m_a, m_f⟩.

At T5 (into the final LN's output) the components are mapped through γ_f/σᵢ, and the bias b_f is
a fourth term B.

**R is not a separate cause** (`/challenge-pr` on #366, finding 2). It is the mean diluted by what
attention and the MLP write, so it is non-zero only when they write. A fall caused entirely by
attention writing large vectors off the mean reads R ≪ 0 and A ≈ 0 (the review's toy: R −0.553,
A −0.0003). So the split names a **mode**, and the writer is found separately in each mode:

- **removal**: the writes oppose the mean (A or M most negative). The writer is that term's
  component.
- **dilution**: the writes grow the tokens' norms off the mean (R most negative). The writer is
  the component that grows them more. Per token, ‖P(hᵢ^{l+1})‖² − ‖P(hᵢ^l)‖² = gᵃᵢ + gᶠᵢ splits
  exactly, as A and M do (gᵃ = 2⟨P h, P a⟩ + ‖P a‖² + ⟨P a, P f⟩, gᶠ likewise), and the writer is
  the larger of the token means of gᵃ, gᶠ.

| item | fixed as |
|---|---|
| a fall | a (prompt, transition) cell with Δ‖m‖² < 0, per prompt (V's cells are per prompt) |
| a fall's label | mode × writer: **"removal, attention"**, **"removal, MLP"**, **"dilution, attention"**, **"dilution, MLP"** (, "LN bias" at T5) |
| the counts | per arm, window and transition, the number of falls by label; the writer's total (removal + dilution) beside each |
| the shares of the mean | s_c = ⟨m_c, m̄_l⟩/‖m̄_l‖² for c ∈ {embedding, a_0 … a_5, f_0 … f_5 (, b_f)}, each with its weights at state l, so Σ s_c = 1 |
| refusals | the components' sum ≠ m_l beyond 10⁻⁵ relative, or Σ s_c ≠ 1 beyond 10⁻⁵: refused. Each record's e1 must equal C4's record at that (arm, step, prompt) within 10⁻⁶ relative (the known answer): refused otherwise |

**E3, the two arms' directions.** c(X, Y; l, s) = cos(m̄_l^X(s), m̄_l^Y(s)). It is taken at the same
step and at the seven matched events of C4's item 10 (A0's and F's step of each).

| item | fixed as |
|---|---|
| the question | **c(F, A0)** at each state and step |
| the order-only scale | c(A0, A0′) and c(F, F′), same schedule, another order |
| a cell's label | **"shared"** iff c(F, A0) ≥ min(c(A0, A0′), c(F, F′)); **"schedule-specific"** iff c(F, A0) < min(…) − 0.1; else "between". Step 0 is 1 on every pair by construction, and is not read |

**Cells (fixed now, before output).** All 29 steps of each of the four arms are computed. The
labels are summarised in four windows per arm:

| window | A0 | F | A0′ | F′ | why |
|---|---|---|---|---|---|
| the cloud's collapse | 24–64 | 1–16 | 24–64 | 1–16 | the ROADMAP row's (ρ_mid's peak: A0 48, F 4) |
| V's date | 384 | 24 | 320 | 32 | C4 R3, C4′ (c) M1 |
| the centred date | 640 | 48 | 640 | 48 | R2, M1 |
| the end | 4000 | 4000 | 4000 | 4000 | |

A window reads "aligned" (E1) or "shared" (E3) at a state iff ≥ ⅔ of its steps do. E2's counts
are not labelled.

**The leans, stated now.** Each lean is marked with its source: **[scan]** from the literature,
**[arms]** formed on C4's pair, which these units read again (`/challenge-pr` on #366, finding 5).

- **e1 [scan]**: at the final state, the mean is "aligned" on both arms in every window where the
  gate is readable (2609.02959: learned early; Pythia has no output bias).
- **e2 [scan]**: at the transitions where V's falls sit (T2 on A0, T1 on F, from V's date on), the
  falls' writer (removal and dilution together) is mostly attention, not the MLP (2401.12143 ties
  the common direction to attention). An MLP majority breaks it. No lean is stated on the mode.
- **e3**: at the end, the mid states (2–4) read "shared" **[scan]** (one corpus, so one unigram
  prior). In the collapse window they read "schedule-specific" **[arms]** (C4 item 10: F takes
  another path early, read on this pair).

**Cost.** 4 arms × 29 steps × 7 prompts, one forward pass each with hooks on every block's attention
and MLP outputs: minutes on the 3080. E1's nulls are 1000 directions × 50 304 tokens, one matrix
product per checkpoint. The unigram counts take one CPU pass over 16.8 GB, which `c4pc_train`'s
writer did in about 10 minutes.

## (g) Do the two end states differ in function? (U2)

**G1, the loss along the line.** θ(α) = (1−α) θ_X + α θ_Y over every parameter (LN included), for α
in 0, 0.1, …, 1, and L(α) = the held-out loss in float32. The barrier is **B(X, Y) = max_α [L(α) −
((1−α) L(0) + α L(1))]**. Around the grid's maximum, α is refined at ± 0.05, beside.

| pair | role |
|---|---|
| (A0, F) at 4000 | **the question** |
| (A0, A0′), (F, F′) at 4000 | the order-only scale per schedule (Frankle's test) |
| (released, A0) at 4000 | the null: same init, order and schedule; the harness's noise |
| (A0, `pythia-70m-seed1`) at 4000 | the effect: another init and order; no alignment |
| **gate** | readable iff B(effect) > 0.1 nats **and** B(effect) > 4 B(null) |
| the scale | **S = max(B(A0, A0′), B(F, F′), B(null), 0.01 nats)**: the order-only spread, never below the harness's noise or a fixed floor (`/challenge-pr` on #366, finding 3: the replicate barriers may be near 0, so a ratio to them alone would read noise as structure) |
| **the label** | **"within the order-only spread"** iff B(A0, F) ≤ S; **"beyond it"** iff B(A0, F) > 2S; else "between". Every barrier is reported in nats beside its label. If S is itself large (> 0.1 nats), "within" means only that the schedules differ no more than the orders do, not that the pair is connected |
| beside | K(F₄₀₀₀, A0₄₀₀₀) from C4′ (b)'s `K_F.npz` (not recomputed); the endpoints' losses; the same barrier at 1000 for the four C4 pairs |

**G2, the heads ablated.** Each arm's heads with an induction score ≥ 0.5 at 4000
(`c4_close.HEAD_BAR`), read from the head records (`data/trackC/c4/2026-11-14-close/heads_{A0,F}.json`,
`data/trackC/c4pc/2026-11-22-u3/heads_{A0p,Fp}.json`; checked 2026-11-23 after `/challenge-pr` on
#366, finding 1):

| arm | heads ≥ 0.5 at 4000 |
|---|---|
| A0 | L3H0, L3H1, L3H5, L3H6 |
| A0′ | L3H0, L3H1, L3H5, L3H6 (A0's four) |
| F | L2H6, L3H3, L3H4, L3H7 |
| F′ | L2H2, L2H5, L2H6, L3H7, L4H1 (five; two shared with F) |

The union is 11 of the 48 heads. **The ablation is a mean ablation:** the head's slice of the
attention output, before the dense projection, is replaced by its mean over the held-out
positions.

| item | fixed as |
|---|---|
| the induction loss | the mean next-token loss on the second copy of 64 sequences of 64 random tokens repeated twice (C4's `IND_LEN`, a new seed, `IND_N` raised to 64) |
| sets ablated on arm X | X's own set; the other schedule's set of the same pair (F's on A0, A0's on F; F′'s on A0′, A0′'s on F′); 20 random sets of X's own set's size, drawn (seeded) from the heads with score < 0.2 on X **and outside the 11-head union**, so no random set holds an induction head of any arm. **Not layer-matched**: layer 3 holds 6 of the union's heads and A0′ has 3 heads below 0.2 there, so a matched set cannot be drawn (finding 1). Each random set's layers are reported beside |
| Δ | the induction loss with the set ablated, minus without; the held-out loss's Δ beside |
| per arm, the label | **"own heads carry induction"** iff Δ(own) > the max of the 20 random Δ. **"index carries it too"** iff Δ(other's indices) > that max as well |
| the arms' comparison | Δ(own)^F / Δ(own)^A0 beside Δ(own)^F′ / Δ(own)^A0′; no bar (a ratio of two pairs) |

**The leans, stated now** (sources marked as for e1–e3):

- **g1 [scan]** (Altıntaş et al., `lit-C4peg.md` §3): B(F, F′) − B(A0, A0′) > max(B(null), 0.01
  nats), so warmup helps order-only connectivity on 70m. A smaller difference reads "not told
  apart".
- **g2 [arms]** (C4 item 10, `P-SCH6`, read on these arms): (A0, F) reads "beyond it" or
  "between", not "within the order-only spread".
- **g3 [arms]** (C4's disjoint head sets): on every arm the own heads carry induction and the
  other schedule's set does not, so the two schedules end on different heads that do the same job.
  On (A0′, F′) the other set shares L2H6 and L3H7 with F's, so g3 there is partly an index test.

**Cost.** G1: 5 pairs × 11 α (+ 4 refinements) × 131 072 positions, a few seconds a point. G2: 4 arms
× 23 sets × 2 losses. Under 30 minutes of GPU together.

## Exit condition and budget

**Exit:** E1–E3 computed on the four arms, with each window labelled or refused by its gate. G1's
gate decided and (A0, F) labelled or refused. G2's labels on the four arms. Each lean e1–e3 and
g1–g3 marked held, broken or unreadable. C4's items 8 and 10 each say what (e) and (g) added. **Budget: 3 units + close-out** (U0 this design,
U1 (e), U2 (g), close-out). GPU under 2 hours in all. At the budget it closes at what it knows.

## Carried in

- **The two `c2_seeds` refusals** (C4 U1's Parked 0): not taken. No unit here derives a centred
  arm, so they stay with ROADMAP C4′, for (d).
- **Keep every registered hash**: new code goes in `trackC_writeup/c4peg_read.py`, so `c4_read`
  (`220161b9159b`), `c4pc_rules` (`d7e674313486`) and `c4p_path` stay as they are.
- `lit-C4peg.md` Parked 1 (2609.02959's full text): read before U1's code (D1).

## Decisions (each yours; recommendation first)

| | question | (a), recommended | (b) |
|---|---|---|---|
| D1 | E1's statistic | Spearman of the logits through W_U against log f, gated per checkpoint by u\* and random directions; 2609.02959's normalised KL beside, its full text read before U1's code, and swapped in as primary only if the reading shows the rank correlation misreads it (decided before any output) | the normalised KL primary now |
| D2 | the replicate pair in (e) | read both pairs together, tier 1: pair 2 is the order-only scale for E3, and its E1 and E2 a replication reported cell by cell. The [scan] leans were not formed on either pair; the [arms] halves were formed on pair 1 by C4 | hold pair 2 back: develop on pair 1, then you register the leans that formed there and pair 2 is read against them (E3 waits for it; + 1 unit) |
| D3 | the ablation | mean ablation over the held-out positions | zero ablation (harsher; moves the residual off its distribution) |
| D4 | registration | none: (e) and (g) are description with their rules fixed here, tier 1. Of the leans, e1, e2, g1 and e3's end half come from the scan; g2, g3 and e3's collapse half were formed on C4's pair, which these units read again, so those cannot be held-out readings here | register g1 (the one [scan] lean in (g)) before U2; it is stated before output, and its pairs exist |

## Decided 2026-11-23 (Blocked 170)

**D1 and D2 (a), by Claude on the user's "Let's just continue here" after Blocked 170 was put**
(the user's to overrule): D1 Spearman primary, D2 both pairs together at tier 1. (e) is frozen
with the details below, fixed before any output. D2 is spent, since U1 read pair 2. First
recorded as "all (a)". After `/challenge-pr` on #367, finding 3 ("continue here" may have meant
only "stay in this session"), **D3 and D4, which (g) needs, went back to the user as STATE
Blocked 171** before U2.

### (e) as U1 runs it (written 2026-11-23, before any output)

2609.02959 was read before the code, as D1 required (its §3.1–3.5, App. B, C, G; `lit-C4peg.md`
Parked 1). It fits d with unregularised least squares on the centred unembedding rows against
centred log-frequency. It excludes zero counts and scores with nKL and R². It never argues
against rank correlation, so **Spearman stays primary**. The details the frozen text left open:

| detail | fixed as | why |
|---|---|---|
| uᵢ | **hᵢ/‖hᵢ‖, plain L2** (`core.models.layernorm_to_sphere`; C2 5b: "both plain L2"). The design's "P centres over the 512 coordinates" was wrong; P is the identity | the code the records use |
| state 6 | HF's last hidden state, after the final LN; the code refuses unless it equals LN_f(h₅ + a₅ + f₅) to 10⁻⁵ relative | the parallel residual and the LN's place are checked, not assumed |
| E1's readout | states 0–5: z(x) = W_U(γ_f ⊙ x); state 6: z(x) = W_U x (already through the LN). u\*, the 1000 random directions and the gate are computed for each readout | reading a post-LN state through γ again would apply it twice |
| u\* | OLS on the support, the readout's rows centred (W̃ = W − w̄) and ℓ̃ = ℓ − ℓ̄ (2609.02959, App. B) | the paper's fit; centring changes no Spearman |
| ties | average ranks (log f has many ties among rare tokens) | |
| the support | tokens with count ≥ 1, as frozen; beside, the paper's top 0.7 \|V\| by count | |
| beside, nKL | KL(p_uni ‖ softmax(β z)) / KL(p_uni ‖ uniform) on the support, β fitted by a scalar search over [−200, 200]; for m̄_l and u\*; R² for u\* | the paper's metric, scale fitted since m̄ has no scale |
| E2 at T5 | the components pass through the LN's linear part, c ↦ γ_f ⊙ (c − c̄)/σᵢ (σᵢ from the pre-LN state, eps 1e-5), and b_f is the term B. R then includes the LN's re-weighting of the old stream, not dilution alone; dilution's writer is read from the pre-LN norm growth | P = I elsewhere, so this is the only non-trivial map |
| E2's cells for e2 | T2 on A0 and A0′, T1 on F and F′ (V's transitions: C4 item 8, `P-SCH3`), at every step from the arm's V date to 4000; "mostly attention" iff attention is the writer at > ½ of those cells' falls | |
| E3's windows | the same-step cosines summarised on each arm's windows of the pair (A0's steps and F's steps), and at C4 item 10's seven event pairs on pair 1 only (pair 2's events are not all dated) | a cosine is one cell for two arms |
| unigram counts | `c4_train.open_rows()`, rows [0, `N_ROWS`), columns 1–2048 (the labels), `np.bincount` per 16 384-row chunk, `minlength` 50 304; saved with its sha256 | |
| known answers | every recomputed per-prompt e1 equals C4's or C4′ (c)'s record at the same (arm, step, prompt) to 10⁻⁵ relative; the components sum to m_l and the shares to 1 to 10⁻⁵ | refuse rather than degrade |
| code | `trackC_writeup/c4peg_read.py`, with pure tests; pushed before any output | |

## Worth challenging (for the PR)

1. **E1 reads the mean through the final LN's weight at every state**, a logit lens. At mid
   states the model never reads its states through W_U. "Aligned" means the direction would
   predict frequent tokens if it reached the output, not that it does. The alternative, a tuned
   lens per state, needs training, and it was rejected.
2. **E2's label is the most negative term**, with dilution's writer taken from the norm growth.
   A fall made of a large negative A and a positive M reads "removal, attention" even when the net
   change is small. The shares are reported beside so that such cases show. (First written with
   R as a third writer, "renormalisation"; `/challenge-pr` on #366, finding 2, showed it is the
   writes' dilution, not a cause of its own.)
3. **E3's "schedule-specific" margin of 0.1** is chosen, not measured. The replicate cosines are
   the scale and the margin only keeps "between" from vanishing. Rejected: no margin, which makes
   every cell binary at the replicate's minimum.
4. **G1 has no permutation alignment.** Every pair except the effect shares an init, so units
   correspond. The effect pair (another init) will show a large barrier partly because nothing
   aligns it. That is why it is the gate's effect and not a comparison for (A0, F).
5. **G2's "other schedule's indices"** tests the index, not the function. F's heads in A0 may hold
   another job (C4 found none shared), so a small Δ there says the index does not carry
   induction, not that the head is idle.
