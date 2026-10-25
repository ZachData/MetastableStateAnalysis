<!-- trackB_circuit/design-B7p.md -->
# Track B — design: B7′, sink-excluded routing activity

Opened 2026-10-25 at the user's request ("Open B7′ … design, gate arm with a planted second
sink"). Row: `docs/ROADMAP.md` §7 B7′ (set at B7's close, Blocked 113 (a)). Scan:
`trackB_circuit/lit-B.md` "B7′". Math: `tools/math_checks/b7p_sink_exclusion.py` (5 checks pass).
Tier 1 throughout: exploratory, unregistered; 410m is an exploration rung, so nothing here can
score a registered row. **Not frozen: Blocked 114 (yours).** Everything below "Objects" is fixed
before any sink-excluded row is computed on real attention.

## What B7′ asks

B7 read 23 layer pairs × 18 steps of partner persistence on condensed rows (top share > ½ of the
row with position 0 dropped). At mid-depth (7–20) from 1000–2000 it read "depth-local persistence"
in 89 cells, but what persisted was mostly a second sink: at 2000 an early `.` or `\n` in every
prompt (0.50–0.83 of each prompt's mid-depth condensed rows), from 8000 a newline in `hdbscan_code`
(key 34) and `latex_monograph` (key 10), which carry about 0.9 of the rows (`status-B.md` "B7 on
one page"). B7's N0 takes out offsets; a fixed key's offset moves with the row, so N0 could not
remove it (check 3). **So routing activity of content partners is not established.** B7′ asks
B7's question again with each prompt's second-sink keys dropped as position 0 is dropped, under a
rule fixed here, re-gated with arms that plant a second sink and heads and prompts that differ.
ROADMAP's done-when: whether content partners persist or reroute across depth, as description.

**The inputs cover it with no forward pass:** B7's (A1a's 18 run dirs × the 7 scored v1 prompts,
`attentions.npz` `(24, 16, n, n)` float32, `activations.npz` `(25, n, 1024)`), each file's sha256
pinned in B7's gate (`data/b7/2026-10-25-unit1c-r2/gate.json`, main tree). **Whether enough rows
are left is not known:** scaling every head's rows evenly in B7's records, 85 of 89 mid-depth L
cells keep ≥ 2 entering heads at 1/5 of their rows and 37 at 1/10 (`/challenge-pr` on #277,
finding 1), but the sink is not removed evenly across heads, so that bounds neither side. "Not
read at mid-depth" is a verdict.

## Exit condition and budget

- **Exit:** the sink-excluded cell table over 23 layer pairs × 18 steps, each cell with its counts,
  intervals and the input named, with each prompt's sink set per step, the rows each cell keeps of
  B7's, and the table of B7 → B7′ description changes, after the gate passes.
- **Budget: 3 units + a close-out page** (writing only): this design as unit 0; unit 1 builds
  `trackB_circuit/b7p_sink.py` and runs the gate; unit 1b is a reserve, spent only if an arm fails
  (B7 spent two, `LESSONS.md` 10); unit 2 reads the 18 steps. The alternative is 2 units (gate and
  read in one, no reserve), which ROADMAP's "about one unit" sized for the read alone. At the
  budget B7′ closes at what it knows unless you extend it on the record.
- **Not in the budget:** the long prompts (`p1d_cluster_ensemble/long_prompts/`,
  `p1e_energy_field/long_prompts_1e/`; forward passes, GPU minutes a step). Whether to run them is a
  decision at close, on what the 7 prompts leave.
- **Compute:** stored arrays only, about 19 GB read, I/O-bound; torch on the GPU as B7 (B7's read
  was 12 s + 216 s). N0 per draw (below) adds a contraction over 2000 draws per bin, chunked.

## The rule, fixed here

| step | definition |
|---|---|
| R1 condensed rows, first pass | B7's: rows i ≥ 16, position 0 dropped as key, renormalised over keys 1..i; out where `w_i = 1 − A_i0 < 0.05`; condensed when the top share > ½, partner that key |
| R2 a sink of (prompt q, layer l) | a key k ≥ 1 holding **≥ ¼** *(placed)* of layer l's condensed rows over its 16 heads, counted only where that layer has **≥ 50** *(placed)* condensed rows in q |
| R3 the prompt's sink set S_q | the union over all 24 layers of R2's keys, per prompt and step |
| R4 the drop | every key in S_q is dropped **as key and as query**, as position 0 is: each row renormalised over keys 1..i outside S_q, `w_i = 1 − A_i0 − Σ_{k ∈ S_q, k ≤ i} A_ik`, out below 0.05; condensed rows and partners recomputed (R1) at every layer and head |
| R5 passes | R2–R4 repeated on the new condensed rows until no new key qualifies, **at most 3 passes** *(placed)*; a prompt still adding a key at pass 3 is recorded, read as it stands, and named |
| R6 recorded | per (prompt, step): S_q's keys, their tokens, the layers where each qualified, its share there, the pass; whether each key is the prompt's first `.` or `\n` (beside, 2410.13835) |

Then B7's instrument, unchanged except N0 (below), on the dropped rows: P per head against the
next layer's best-matched head chosen leave-one-prompt-out, the cell value the entering heads'
weighted mean, N1 (layers ≥ 2 apart), N0, the ordered partition, the beside columns
(`design-B7.md` "Objects, fixed here"; `b7_activity.py` choices 1–19).

**Why these.** ¼ sits in the gap of B7's sink table (0.50–0.94 of a prompt's mid-depth rows on the
key; ≤ 0.06 on any one key elsewhere); that table was seen, so ¼ is placed in a known gap, not
blind. Under B7's planted nulls the largest key share is about 0.05 at full rows and 0.10 at 50
rows (check 4), so the rule finds nothing there. Per layer and then a union, rather than ROADMAP's
example (the modal key over layers 7–20 at ≥ ½): 7–20 and ½ are B7's own seen values, `homer_iliad`
sits at 0.50 at 2000, and secondary sinks arise in middle layers and persist a variable number of
layers (2512.22213), so a fixed window would miss one that lives in 9–12. One set per prompt, so
every layer pair, N1's distant pairs included, reads the same rows. Keys dropped, not rows: a row on
the sink with a content partner under it is read (check 2), a row with nothing under it goes
diffuse and is out.

## Objects changed from B7

| object | definition |
|---|---|
| rows | R1–R5 above; B7's rows where S_q is empty |
| N0, per draw *(B7 Parked; `/challenge-pr` on #273, finding 3)* | B7's N0 (unit 1b: ordered pairs of different jointly condensed rows of i's 32-position bin over the 7 prompts, offsets valid at i, whose offsets agree), **with the bin's offset profiles re-built from each bootstrap draw's prompts**: rows weighted by their prompt's multiplicity m_q, pairs of different *original* rows weighted m_q m_q′ (a row's own copies never pair, check 5: pairing them brings back unit 1b's self-pair bias, 0.24 against 0.009 at 50 frozen rows). Closed form: Σ_k p_m(k) q_m(k) − Σ_j m_j² [self agree], over C_p C_q − Σ_j m_j² [both valid]. Row 0 (the sample) equals B7's N0 exactly. The `P − N0` interval then carries N0's own noise |
| everything else | B7's, by reference (`design-B7.md`, `b7_activity.py` `a1de85990c74`, imported, not edited) |

**Beside, not in the family:** (1) P with the 7 prompts weighted equally (each prompt's P over
the entering heads, averaged; its interval), since two prompts carried B7's rows; (2) each cell's
description under unit 1b's entry rule (≥ 50 leave-one-out rows) and the count that differ (B7
printed only the count of heads); (3) the count of cells whose description changes with R2's ¼ at
⅛ and at ½ (counts only, no cell listed); (4) per cell, the jointly condensed rows kept of B7's;
(5) the share of `L6H4`'s and `L15H7`'s condensed rows on S_q in `hdbscan_code` and
`latex_monograph` per step, a pointer for whether the same newline flipped their pooled M1 β (A2
unit 2; `status-B.md` "Corrections received" 2026-10-16), not a reading of β.

## The math

`tools/math_checks/b7p_sink_exclusion.py` (`sympy` + numerics, 5 checks pass): (1) dropping S
scales every kept entry by 1 / (1 − s), so the order among kept keys and any condensation on a kept
key survive; (2) a planted row with 0.7 / 0.2 / 0.1 of its non-sink mass on the sink, a content
partner and uniform is condensed on the sink before the drop (0.7 + 0.1 / i) and on the partner
after ((2i + 1) / (3i − 1) > ½); a 0.8 / 0.2 row is condensed on nothing after (1 / (i − 1)); (3)
B7's N0 on bins of frozen rows on one key per prompt is 0.014 with P = 1, which is why B7 read
those cells "beyond position"; (4) under G1, G2, G4 at the real n's the largest key share of a
layer's condensed rows is 0.049 at full rows and 0.100 at 50 rows thinned, against ¼; (5) the
U-statistic identity Σ_{j ≠ j′} m_j m_j′ x_j y_j′ = (Σ m x)(Σ m y) − Σ m² x y, and at 50 rows a
draw's N0 over different original rows stays at the reshuffled value (0.0080 against 0.0087)
while counting copies puts it at 0.24.
**What it does not prove:** that 410m's sink rows are check 2's (what is under the sink is read);
that ¼ separates sinks from content in 410m (a content key a quarter of a layer's rows point at is
dropped, whatever its token, and R6 names it); that what is left is content and not position (N0
takes the positional part out, as in B7); check 5 is for the counts, c_i a ratio of them.

## The cells (fixed before any output)

Family: the 23 layer pairs × 18 steps on R1–R5's rows; the statistic is the per-head P against N1
and N0 (per draw) with their intervals; B7's ordered partition, unchanged (not read; positional,
not separable; rerouting; depth-local persistence; frozen across depth; persistent, less than at
distant layers), and B7's two-cycle candidate beside. The grid is read as a picture, with no count
against a rate (`design-B7.md` "The cells"). **The readout B7 lacked is the transition table:**
B7's description against B7′'s per cell, counted, with mid-depth (7–20) apart. A B7 L cell that
reads L again says content partners persist there; one that becomes R says the persistence was
the sink's; one that becomes "not read" says the 7 prompts cannot tell.

## Unit 1's gate (before any real cell; CLAUDE.md "Calibration gate")

At B7's design points (each prompt's real n, every row's real `A_i0` at 512, 8000, 143000, the
`(24, 16)` shape, the 7 prompts). B7's arms put one map in every head and prompt and planted no
second sink, so they could not see the two things the real data had (`status-B.md` "B7 on one
page", reading 7); the new arms plant both.

**The second-sink plant (G10–G12).** Per seed, two of the 7 prompts are *heavy*: prompt A has one
sink key k_A in layers 7–20; prompt B has a key k_B in layers 9–12 only (a transient sink,
2512.22213) and a second key k_B′ in 14–20; keys drawn from 10..54 (B7's real sinks sat at 10–54).
In a heavy prompt's sink layers each head h gets its own sink share s_{q,h} ~ U(0.5, 0.95); a row
i > k assigned to the sink is *hidden* (0.7 of its non-sink mass on k, 0.2 on its content partner,
0.1 uniform over keys 1..i) or *pure* (0.8 on k, 0.2 uniform), half each. In the five *light*
prompts, layers 7–20 plant content partners on 1/10 of rows (the rest diffuse) and a key of their
own at share U(0, 0.04) per head, below R2; so the heavy prompts carry most of mid-depth's rows,
as B7's two did. Heads differ by prompt (below).

**Heads that differ (G1d, G3d and every G10–G12):** a content map per (prompt, head) and a random
head permutation per layer, the same in every prompt, so h\* must be found and P cannot come from
one map shared by all heads.

| arm | construction | passes when |
|---|---|---|
| G0 | B7's G0, and B7's gate and records present with their hashes | all hold |
| G1–G9 | B7's arms with R1–R5 and the per-draw N0 | R2 finds no key in any prompt or step (S_q empty, so the rows are B7's); every arm's B7 pass condition holds |
| G1d, G3d | G1 and G3 with heads that differ | as G1, G3 (full rows and minimum entry) |
| G10 sink over reshuffled | the plant, content partners as G2 | (a) S_A = {k_A}, S_B = {k_B, k_B′}, every light S_q empty; (b) **B7's instrument without the drop reads ≥ ¾ of cells 7 → 8 … 19 → 20 L or F** (the arm reproduces B7's confound; otherwise it does not plant what B7 saw, and is revised before any real cell); (c) with the drop, G2's pooled band \|Σ P − Σ N0\| ≤ 0.25 Σ N0 and every read cell "rerouting"; (d) ≥ 0.99 of hidden rows condensed on their planted partner after the drop, no pure row condensed |
| G11 sink over frozen | the plant, content as G1d | (a), (d) as G10; with the drop, G1's condition (P ≥ 0.99, N1 ≥ 0.99, `P − N0` ≥ 0.9, every cell "frozen across depth") at full rows and the minimum entry |
| G12 sink over drift | the plant, content as G3d | (a), (d); with the drop, G3's band on (P − N0) / (1 − N0) at full rows and the minimum entry |
| G13 false persistence | G10 at full rows and at the minimum entry, 20 seeds | a read cell described as anything but "rerouting" in ≤ 5 % of cells at both (G7's rule) |
| G8 on real | B7's G8 (143000, layers 10–11, heads 0–3 one shared fixed map, keys outside S_q) with R1–R5 | S_q the same with and without the plant; G8's condition against the unplanted sink-excluded run; records differences only, and S_q (an input, not a cell) |

An arm that fails changes the constants, with the reason recorded, before any real cell is read
(B2's rule 6); a failure of G10 (b) changes the plant, not the rule. G10–G12 also run as code
tests on a small synthetic cloud.

## Seen before the rule was fixed

B7's grid, its sink table (the modal key, token and shares by prompt at 2000, 8000, 143000), its
caveats and its records' scaled-row counts (`status-B.md` "B7 on one page", "B7 unit 2"); the
math check's synthetic values; the scan's summaries. **Not computed on real attention:** any R2
share or S_q, any sink-excluded partner, P, N0, cell or count. G8 will compute S_q at 143000 in
unit 1, before unit 2.

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen (Blocked 114) |
| 1 | `trackB_circuit/b7p_sink.py` (imports `b7_activity`, unchanged): R1–R6, N0 per draw, the beside columns, G0–G13, code tests | every arm passes, or a recorded change |
| 1b | reserve: a failed arm's revision, re-gated (only if unit 1 fails) | every arm passes |
| 2 | the 18 steps × 23 layer pairs, S_q per prompt and step, the transition table, the beside columns, with the input named | the table is written with its input |

## Fences

- B7's fences stand (`design-B7.md` "Fences"): description, not dynamics (410m is untied; a kept
  partner can be two layers' weights agreeing); no head identity across layers; condensed rows
  only; partners in sequence space; stored attention, 7 v1 prompts, 18 steps, the 12 held-out not
  read.
- **"Sink" is R2's operational definition**, not a claim about what the key does: a key that a
  quarter of a layer's condensed rows point at. R6 names every token dropped, so a content word is
  visible if one is dropped.
- **What is left is "non-sink", not "content".** Positional rows stay and are N0's to take out;
  rows on 2–3 content keys are out (top share ≤ ½), as in B7.
- **Two prompts may still carry the rows.** If they do, the equal-weight P beside says what the
  other five say; it describes no cell.

## Worth challenging (for the freeze)

1. **¼ placed in a seen gap.** It is not blind: B7's table showed 0.50–0.94 against ≤ 0.06. The
   alternatives are ROADMAP's ½ over layers 7–20 (rejected above: B7's seen window and `homer_iliad`
   at 0.50), or a key's share against its own null (the share a key 1..i gets from rows that can
   reach it, check 4's model), which is blind but adds a constant to place. ⅛ and ½ are beside as
   counts.
2. **One sink set per prompt, every layer.** A key that is a sink in 9–12 only is also dropped at
   layer 22, where it may be a content partner; per-layer sets would keep those rows but give every
   layer pair, and each of N1's distant pairs, its own rows. Union chosen for one row set.
3. **Keys dropped, not rows.** Dropping the rows on the sink would also drop the hidden partners
   under it and condition on the sink; dropping the keys reads them (check 2), at the risk that a
   row's remaining top share > ½ is the uniform tail's (check 2: never at i ≥ 16).
4. **G10 (b) as a pass condition.** The arm must reproduce B7's failure before its fix is trusted;
   the alternative is to read (b) beside. Chosen because B7's arms passed while missing the real
   data's shape (`LESSONS.md` 6).
5. **N0 per draw is in, for every cell.** It widens `P − N0`'s interval; B7's 7 R cells are the
   ones it could move. The alternative is B7's fixed-profile N0 with the per-draw one beside;
   rejected because B7's close carried it here as part of the interval.
6. **3 units with a reserve**, with 2 as the alternative above.
7. **The long prompts out of the budget.** They add content rows at the cost of forward passes; the
   7 prompts are read first, so the long-prompt decision is made on what is left.
