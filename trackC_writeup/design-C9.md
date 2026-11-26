<!-- trackC_writeup/design-C9.md -->
# C9 — the paper across every model: design (proposed 2026-11-25)

ROADMAP §8 C9; opened by Blocked 176 (the user's, 2026-11-24: "I would prefer to have all of the
models there"). Scan: `lit-C9.md`. **Proposed; nothing read.** Decisions D1–D6 below are yours
(STATE Blocked 178). Every rule here is fixed before any of its output exists.

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
| `pythia-70m` | 1 | 18 | C2 3b (`c2u3b/…/70m`) | cached |
| `pythia-70m` seeds 0–9 | 10 | 18 | C6 U0 (`c6/2026-10-31-u0-seeds/seed{0..9}`) | cached |
| Mistral GPT-2 small | 5 seeds | 35 (0–16000) | C2 3c (`c2u3c`) | cached (`HDD_1TB/mets_data/hf`) |
| 70m harness arms A0, A0′ | 2 | C4's 28–29 | C4 U2, C4′ (c) U3 | `HDD_1TB/mets_data/c4`, `c4pc` |
| 70m flat arms F, F′ | 2 | C4's 28–29 | the same | the same |
| `pythia-70m-warmup01` | 1 | 18 | C6 U1b (`c6/2026-11-01-u1b/run`) | cached |

The ROADMAP row said U1 needs GPU hours. **It does not**: the records above hold `e1`, `cos` and
`n` per state (checked 2026-11-25 on one record per folder, and the file counts: 126 = 18 × 7 per
Pythia-grid run, 245 = 35 × 7 per GPT-2 seed). ‖m‖² follows from `cos` (math check M5). U1 is a
CPU read. U1's first step refuses unless every (state, prompt) record of every run is present.

## U1, the mean's term on every model (no forward pass)

**Statistic** (`c2_readings.one_series`, unchanged, on each model's records): per checkpoint,
pooled over the model's runs, prompts and transitions,
- S_any = #(transitions where E₁ falls and ‖m‖² falls) / #(transitions where E₁ falls);
- S_break = the same over the drops the count counts (D2 (a)'s REL_TOL), at the model's dated
  E₁ break (table 2). Undefined where table 2 has no date (warmup01's E₁ "refused").

Beside, never read: the base rate (‖m‖² falls over all transitions), same-direction share,
S_any per run.

**Reading (D1).** Per model: **extends** if S_any ≥ 0.96 at every checkpoint and S_break = 1;
**extends at the break only** if S_break = 1 but some checkpoint has S_any < 0.96; **does not
extend** if S_break < 1. A checkpoint with fewer than 5 E₁ falls pooled is reported, not read.

**Gate (before any model is read; the calibration gate on the real design points).**
- *Known answer:* on 410m's records, U1's code returns C2 5b's R6 numbers exactly.
- *Planted effect:* each model's records with R held at its step-0 value (E₁ rebuilt as
  ½ + ½‖m‖² + R₀) read S_any = 1 at every checkpoint.
- *Planted null:* each model's ‖m‖² differences permuted across transitions within each record
  (1000 draws). S_any falls to its base rate ± 0.05, and "extends" occurs in < 5 % of draws.
  This is the forced-agreement check (`LESSONS.md` 11). If the null reads "extends" often, the
  0.96 says nothing on that model and is reported as such.

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
E5, E6u, E6b, E7, E8 exactly. With D2 (a), it also reads E3 and E4 by the seeds' rule on seed 0,
beside the named-head dates.

**What is read (D4):** A3's eleven pairs, per model. A pair is **aligned** if the two located
steps are within a factor of 2 (410m's grid step, so "aligned" means the same thing on every
grid), else **ordered** (which first). The model's own grid distance is reported beside. Then
**the order replicates on model M** for a pair if M's verdict equals 410m seed 0's. GPT-2's five
seeds are read one by one, and the count of seeds agreeing is reported.

## U3, the 160m seeds' two dates

Records on 9 seeds × 18 steps × 7 prompts by C2 3b's producer on 160m (`c2_dose` records, the
same code that made 160m seed 0's), parametrised by repo. Then E₁'s break by D4 (a) and the
centred break by C2 3s's tolerances, gated on 160m seed 0 first (A5's gate, 160m's records). U1's
statistic runs on them too. **D5** is whether the break is registered first.

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

- **D1, U1's bar.** (a) As above: the abstract's own 0.96 at every checkpoint, and every drop
  at the break. (b) Description only, no label.
- **D2, which heads E3 and E4 read on a model without named heads.** (a) **A5's D10 rule**,
  which already dated heads on the nine 410m seeds: one head of each kind per run, the top score
  at the run's last state (143000 on Pythia; 16000 on GPT-2; 4000 on the arms), then that head's
  series. (b) Head-free: at each state the maximum over heads (C4's induction instrument, and the
  same for previous-token). This tracks "some head", not one head, and differs from §5's 410m
  rows.
- **D3, E6 on GPT-2 small.** (a) Not read on GPT-2. Its corpus (OpenWebText) and tokenizer
  differ, so a Pile n-gram table measures distance to another corpus. Read on A0, A0′ with A3's
  tables. (b) Tables rebuilt from the same Pile split in GPT-2's tokenizer (≈ 1 h CPU), read
  with that caveat.
- **D4, "aligned" across grids.** (a) Within a factor of 2 in steps, on every grid. (b) One grid
  step of the model's own grid: on GPT-2's 50-step grid that is ~10 % at 500, so few pairs align.
- **D5, U3's registration.** (a) Register before any 160m-seed record exists, as P-A5 on
  410m: "In each of pythia-160m-seed{1..9}, E₁'s break (D4 (a)) is step 256 or 512", K and
  falsifier as P-A5's. 160m seed 0's 256 is the exploration rung, and the nine seeds are held
  out. (b) Tier 1, description.
- **D6, the paper.** (a) U4 edits `paper/` as the last unit (as "Paper: every model read" did).
  (b) U4 writes a `paper/`-ready block in `status-C.md` and leaves `paper/` to your rewrite.

**Recommended: D1 (a), D2 (a), D3 (a), D4 (a), D5 (a), D6 (a).**

## Worth challenging (for the PR)

1. U2 leaves out 70m and 160m on Pythia's grid (one run each) and the flat arms. The harness
   arms are 70m on Pythia's schedule with a finer grid, so 70m is covered there.
2. D2 (a) picks heads at the last state, after the window. That is A5's rule, so 410m's seeds and
   C9 read heads the same way. But a head that peaks and is replaced (2407.10827) is missed.
3. D5 (a) adds a registration for the paper's sake. The alternative is to report the nine seeds
   as description, as table 2 does for every model but 410m.
4. U1's planted null permutes ‖m‖²'s moves within a record. A null that keeps each record's
   autocorrelation (a circular shift) would be stricter.
