<!-- trackC_writeup/design-C9.md -->
# C9 — the paper across every model: design (proposed 2026-11-25; frozen 2026-11-25, Blocked 178 all (a))

ROADMAP §8 C9; opened by Blocked 176 (the user's, 2026-11-24: "I would prefer to have all of the
models there"). Scan: `lit-C9.md`. **Frozen 2026-11-25** ("Decided 2026-11-25" below; D1–D6
were STATE Blocked 178). Every rule here is fixed before any of its output exists.

## What it is for

`paper/` makes three claims on `pythia-410m` that table 2 (§4.2) does not yet carry to the other
models:

1. **The mean's term** (abstract, §4 "How much of V the mean's term carries"). At every
   checkpoint, wherever E₁ falls by any amount, ‖m‖² falls at 0.96–1.00 of those transitions
   (`c2_readings` R6 `m_fall_share_at_any_e_fall`, 410m × 10 seeds, 7 prompts). At the break,
   every drop the count counts is a fall of ‖m‖² (`m_fall_share_at_break` 1.0).
2. **The timeline** (§5, table 3). On 410m seed 0: E5 8, E6u 128, E6b 256, E1 512, E2 512, E8
   1000, E3 1000, E4 1000, E1c 1000, E7 censored. A3's eleven pairs are read on seed 0 only.
3. **The two dates on the 160m seeds** (§4.2's table has 160m as one run).

C9 asks, of each claim, whether it holds on every model whose early checkpoints exist. **It reads
orders within a model, not dates across models**: Pythia and GPT-2 small differ in batch and
context, and induction's step is predicted by those, not by size (`lit-C9.md`: 2511.16893).

## The models (each with records or checkpoints already, except U3)

| model | runs | grid | E₁ / E1c records | checkpoints |
|---|---|---|---|---|
| `pythia-410m` | 10 seeds | Pythia's 18 | C2 3s (`c2u3s`) | cached (the known answer) |
| `pythia-160m` | 1 (seed 0) | 18 | C2 3b (`c2u3b/…/160m`) | cached |
| `pythia-160m-seed1..9` | 9 | 18 | **none: U3** | **downloading** (`Mets_hdd_storage`, ≈ 61 GB, Blocked 176) |
| `pythia-70m` seeds 0–9 | 10 | 18 | C6 U0 (`c6/2026-10-31-u0-seeds/seed{0..9}`; seed 0 is `pythia-70m`, also in C2 3b, not counted twice) | cached |
| Mistral GPT-2 small | 5 seeds | 35 (0–16000) | C2 3c (`c2u3c`) | cached (`HDD_1TB/mets_data/hf`) |
| 70m harness arms A0, A0′ | 2 | C4's 28–29 | C4 U2, C4′ (c) U3 | `HDD_1TB/mets_data/c4`, `c4pc` |
| 70m flat arms F, F′ | 2 | C4's 28–29 | the same | the same |
| `pythia-70m-warmup01` | 1 | 18 | C6 U1b (`c6/2026-11-01-u1b/run`) | cached |

The ROADMAP row said U1 needs GPU hours. **It does not**: the records above hold `e1`, `cos` and
`n` per state (checked 2026-11-25 on one record per folder, and the file counts: 126 = 18 × 7 per
Pythia-grid run, 245 = 35 × 7 per GPT-2 seed). ‖m‖² follows from `cos` (math check M5). U1 is a
CPU read. U1's first step refuses unless every (state, prompt) record of every run is present.

## U1, the mean's term on every model (no forward pass)

**Code.** `c2_readings.one_series` hard-codes 410m's 25 states (its `net_inner` reads
`e[23]`), so it refuses on 70m's 7 (`/challenge-pr` on #380, finding 5). U1 uses a copy that
takes the state count from the record, and its known answer is equality with `one_series` on
410m's records.

**What is forced, and what is not** (finding 2). E₁ = ½ + ½‖m‖² + R with R ≥ 0, so
ΔE₁ = ½Δ‖m‖² + ΔR. Where E₁ falls, ‖m‖² can rise only if R falls by more. So S_any, the paper's
sentence, is near-forced by the algebra, and no shuffle or shift of ‖m‖² is a null for it: each
would test independence, which the identity already excludes. U1 therefore reads two statistics,
per model, pooled over its runs, prompts, transitions and checkpoints:
- **S_any** (the paper's sentence, carried as description): #(E₁ falls with ‖m‖² falling) /
  #(E₁ falls).
- **M, the magnitude share** (the reading): over the drops the count counts (C2's D2 (a)
  REL_TOL), the share at which the mean's term carries more than half of E₁'s fall,
  s = ½Δ‖m‖² / ΔE₁ > ½ (`c2_readings`' `s_gt_half_share`). M can fail: a model whose drops are
  R's has M near 0 and S_any still high.

Beside, never read: the base rate (‖m‖² falls over all transitions), same-direction share,
per-checkpoint and per-run values, and S_break (S_any at the model's dated break; undefined where
table 2 has no date).

**Reading (D1).** Per model: **the mean's term carries the drops** if M ≥ M₄₁₀ − 0.1, where M₄₁₀
is 410m's pooled M from C2 5b's record (read at U1 before any other model; **placed** from
410m, so the bar is the paper's model's own level less a margin, and the margin is placed, not
calibrated). **Partly** if ½ ≤ M < M₄₁₀ − 0.1; **not** if M < ½. S_any is reported against 0.96
as the paper words it, with no label: the 0.96 was 410m's pooled minimum, and a single-run model
has 42–84 transitions a checkpoint against 410m's 1680, so a per-checkpoint bar would turn on
one transition (finding 6).

**Gate (before any model is read).** It checks the code only (the plants are exact by
construction, so they test nothing of M's power; corrected after `/challenge-pr` on #381,
finding 4). It is not a null for S_any.
- *Known answer:* on 410m's records, U1's code returns C2 5b's R6 numbers exactly.
- *Planted mean:* each model's records with R held at its step-0 value (E₁ rebuilt as
  ½ + ½‖m‖² + R₀) read M = 1 and S_any = 1.
- *Planted R:* each model's records with ‖m‖² held at its step-0 value (E₁ rebuilt as
  ½ + ½‖m₀‖² + R) read M = 0 and the label "not". If either plant misreads on a model, U1
  refuses that model.

## U2, the timeline's order on GPT-2 small and the harness arms

**Models:** GPT-2 small × 5 (35 states), A0 and A0′ (C4's 28–29 states). Not the flat arms
(their counts date a transient, §7). Not 70m / 160m on Pythia's grid (one run each, and §5
already notes 70m's finer grid on the harness arms).

**Rows** (A3's definitions where they are model-free, A3's code generalised by model and grid):

| row | on GPT-2 small | on A0, A0′ | change from A3 |
|---|---|---|---|
| E1, E1c | C2 3c's dates (E1c has none there) | C4's dates | none (records) |
| E2 OV attractive share, median over heads | weights | weights | heads 144 / 48 |
| E3 induction | **D2** | **D2** | 410m's named heads do not exist elsewhere |
| E4 previous-token | **D2** | **D2** | as E3 |
| E5 effective rank on the sphere | one pass per state, 7 prompts | the same | none |
| E6u, E6b n-gram KL | **D3** | A3's tables (Pythia's tokenizer, same corpus) | the model's grid |
| E7, E8 weights | weights | weights | none |
| E10 the LR schedule | Mistral's logarithmic warmup to ≈ 4000 | Pythia's | per model |
| E11 | not run (refused on 410m by construction) | not run | dropped |

**The locator and its gate:** A3 §4 unchanged in rule; its planted gaps become each model's own
grid gaps. Each row is gated on each model's grid and units before that model's row is read.
Null ≥ 95 % "no change", planted effect placed in ≥ 90 % per gap; a failed row is refused on
that model. *Known answer:* the generalised code on 410m seed 0 reproduces A3's dates for E2,
E5, E6u, E6b, E7, E8 exactly. Under D2 it also reads E3 and E4 on seed 0 by the chosen rule,
beside the named-head dates.

**What is read (D4):** A3's eleven pairs, per model. Each located step is **snapped onto Pythia's
18-step grid** (the first grid step at or above it), and A3's rule is applied unchanged:
**aligned** if equal or one grid step apart, else **ordered** (which first). Then **the order
replicates on model M** for a pair if M's verdict equals 410m seed 0's. The model's own-grid
distance is reported beside, never read, and so is a pair that is aligned after snapping but
ordered on the model's finer grid (§5's "On a finer grid" is such a pair: E₁ 384 and the centred
640 on the harness arms snap to 512 and 1000, aligned). GPT-2's five seeds are read one by one,
and the count agreeing is reported. Why snapping (finding 1): a fixed factor of 2 does not mean
what A3's one step means. Two adjacent Pythia steps can hide true dates nearly 4× apart, so a
finer grid would fail to "replicate" 410m's aligned pairs from resolution alone.

## U3, the 160m seeds' two dates

Records on 9 seeds × 18 steps × 7 prompts by C2 3b's producer on 160m (`c2_dose` records, the
same code that made 160m seed 0's), parametrised by repo. Then E₁'s break by A5's half-rise
(A5's D4 (a)) and the centred break by C2 3s's tolerances, gated on 160m seed 0 first (A5's gate,
160m's records). U1's statistics run on them too. **D5** is whether anything is registered first.

## U4, the paper

Table 2's 160m row → 10 seeds; a column or sentence for U1 per model; §5 gains a short
"on other models" paragraph with U2's pair table; abstract scope line. `paper_check` 0; every
number from a record (`numbers.py`). Whether U4 waits for your rewrite is **D6**.

## Exit condition and budget

**Exit:** U1 read or refused on every model in the table; U2's pairs read or refused on GPT-2 × 5
and A0, A0′; U3's two dates on the nine 160m seeds (registered or not, D5); U4's edits, or D6 (b).
**Budget: 5 units + close-out** (U0 this; U1; U2a the generalised rows, their gates and the
known answer; U2b the GPU rows and the reading; U3; U4 folded into the close-out if D6 (b)).
GPU ≈ 4–6 h (U2b ≈ 2–3 h on GPT-2's 175 states and the arms' 57; U3 ≈ 1–2 h on 162 states).
Disk: the download's 61 GB, under the 250 GB cap.

## Carried in

- The two `c2_seeds` refusals (ROADMAP C9, from C4′ (d)+(f)'s close): U3 derives centred
  tolerances on 160m, so it **takes them**. `null_matched_tol` refuses at a zero uncentred rate,
  and `b_matched_tol` is checked against its three other callers before the change.
- `lit-C9.md` Parked 1 before U3's records are read; Parked 2 before U2b's rows are read.

## Decisions (each yours; recommendation first)

Decisions are this design's; another design's carry its name (A5's D4 (a), C2's D2 (a);
finding 7).

- **D1, U1's reading.** (a) As above: the magnitude share M is labelled against 410m's own M
  less 0.1 (placed). S_any is reported against the paper's 0.96 with no label. (b) Description
  only: both statistics, no label. (c) The first proposal: S_any ≥ 0.96 at every checkpoint.
  Withdrawn, since it is near-forced (finding 2) and turns on one transition on a single run
  (finding 6).
- **D2, which heads E3 and E4 read on a model without named heads.** (a) **The maximum over
  heads at each state.** This is C4's induction instrument (`c4_close.heads`, random repeated
  sequences, the bar's series), already the rule behind the paper's induction dates on the
  harness arms (768, 768). The same applies to previous-token (the maximum over heads of the
  mean attention to i − 1 on the 7 prompts). 410m seed 0 is read this way too, beside its
  named-head rows, as the bridge. (b) A5's D10 rule, one head of each kind per run (the top score
  at the last state), then its series. This was the first recommendation, withdrawn: it would
  give the harness arms a second induction date beside the paper's (finding 4), and it misses a
  head that peaks and is replaced (2407.10827).
- **D3, E6 on GPT-2 small.** (a) Not read on GPT-2. Its corpus (OpenWebText) and tokenizer
  differ, so a Pile n-gram table measures distance to another corpus. Read on A0, A0′ with A3's
  tables. (b) Tables rebuilt from the same Pile split in GPT-2's tokenizer (≈ 1 h CPU), read
  with that caveat.
- **D4, "aligned" across grids.** (a) Snap each located step onto Pythia's 18-step grid, then
  A3's one-step rule (above). (b) One step of each model's own grid: GPT-2's 50-step grid would
  make most pairs "ordered". (c) Within a factor of 2 in steps: the first proposal, withdrawn
  (finding 1).
- **D5, U3's registration.** (a) **None, tier 1 description**, as table 2 reports every model
  but 410m. (b) Register first. But the window "256 or 512" already holds on the nine 410m seeds
  and the ten 70m seeds, and on 160m seed 0 (256). Its null is one this repo calls "a null nobody
  holds", so a registration would add little (finding 3). If you want one with teeth: (c)
  register the split, "the 160m seeds break at 256 more often than the 410m seeds do" (the
  larger model later), a direction the data have not shown on 160m.
- **D6, the paper.** (a) U4 edits `paper/` as the last unit (as "Paper: every model read" did).
  (b) U4 writes a `paper/`-ready block in `status-C.md` and leaves `paper/` to your rewrite.

**Recommended: D1 (a), D2 (a), D3 (a), D4 (a), D5 (a), D6 (a)** (revised after `/challenge-pr`
on #380, accept with changes, findings 1–7 taken; D1, D2, D4 and D5's options rewritten before
any decision).

## Decided 2026-11-25

**Blocked 178: D1–D6 all (a)**, taken by Claude as recommended on the user's "merged, continue"
(after #378–#380 merged); yours to overrule. D5 bites only when U3 reads a 160m seed's record,
so (c) stays open until that session starts.

**U1 as it runs** (`trackC_writeup/c9_u1.py`, pushed before any output). Where the text above left
a choice:
- *Models* are the rows of the table above, each pooled over its runs: 410m (10), 160m (1),
  70m (C6's 10; C2 3b's copy of seed 0 not read), warmup01 (1), GPT-2 small (5), the harness arms
  (A0, A0′), the flat arms (F, F′). Every checkpoint on each run's own grid counts.
- *M₄₁₀* is C2 5b's R6 pooled over its 18 steps: Σ (s > ½ at drops) / Σ drops, from the record's
  per-step shares and counts, and must equal U1's own pooled M on 410m's records.
- *The plants hold at the series' first state* (depth), not at training step 0: only then are
  they exact (R constant over depth gives s = 1 at every drop; ‖m‖² constant gives s = 0), which
  is what "read M = 1" and "M = 0" require. A plant with no drop reads "refused".
- *S_break* (beside) uses table 2's E₁ dates from the records they were read from; none for
  70m seeds 1–9 (centred only) or warmup01 (refused).

## Worth challenging (for the PR)

1. U2 leaves out 70m and 160m on Pythia's grid (one run each) and the flat arms. The harness
   arms are 70m on Pythia's schedule with a finer grid, so 70m is covered there.
2. D1 (a)'s margin of 0.1 below 410m's M is placed, not calibrated. Single-run models' M rests on
   fewer drops.
3. D4 (a) makes §5's finer-grid pair on the harness arms read "aligned" under the reading, with
   its finer order beside. The paper already prints that order as a description, and C9 does not
   overturn it.
