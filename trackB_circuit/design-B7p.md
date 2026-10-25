<!-- trackB_circuit/design-B7p.md -->
# Track B — design: B7′, sink-excluded routing activity

Opened 2026-10-25 at the user's request ("Open B7′ … design, gate arm with a planted second
sink"). Row: `docs/ROADMAP.md` §7 B7′ (set at B7's close, Blocked 113 (a)). Scan:
`trackB_circuit/lit-B.md` "B7′". Math: `tools/math_checks/b7p_sink_exclusion.py` (5 checks pass).
Tier 1 throughout: exploratory, unregistered; 410m is an exploration rung, so nothing here can
score a registered row. **Frozen 2026-10-25 as revised after #278's review (Blocked 114 (a), the
user's): the per-head rule, its gate with G10 (b) as a pass condition, 3 units with a reserve +
close-out.** Everything below "Objects" is fixed before any sink-excluded row is computed on real
attention; what the gate's construction left open is fixed in `b7p_sink.py`'s docstring before
any output.
**Revised after `/challenge-pr` on #278 (accept with changes), before the freeze:** R2 counts per
head, not per layer, since the sink table ¼ was placed against counted entering heads only, not a
layer's 16 (1); a head-sparse arm G14 and a sub-threshold leak arm G15, with the near-miss share
beside on real data (2); a row's sink membership follows a chain across layers and G10 (b) asks
for L, the description B7 read (3); the mass a kept row rests on beside (4); a key-agreement null
beside (5); "mass off position 0" for "non-sink mass", and the rule's constants live here only (6).

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
| R2 a sink of (prompt q, layer l, head h) *(as revised, #278 finding 1)* | a key k ≥ 1 holding **≥ ¼** *(placed)* of head (l, h)'s condensed rows in q, counted only where that head has **≥ 50** *(placed; B7's `MIN_JOINT`)* condensed rows in q |
| R3 the prompt's sink set S_q | the union over all 24 × 16 heads of R2's keys, per prompt and step |
| R4 the drop | every key in S_q is dropped **as key and as query**, as position 0 is: each row renormalised over keys 1..i outside S_q, `w_i = 1 − A_i0 − Σ_{k ∈ S_q, k ≤ i} A_ik`, out below 0.05; condensed rows and partners recomputed (R1) at every layer and head |
| R5 passes | R2–R4 repeated on the new condensed rows until no new key qualifies, **at most 3 passes** *(placed)*; a prompt still adding a key at pass 3 is recorded, read as it stands, and named |
| R6 recorded | per (prompt, step): S_q's keys, their tokens, the heads (l, h) where each qualified and its share there, the pass; whether each key is the prompt's first `.` or `\n` (beside, 2410.13835) |

Then B7's instrument, unchanged except N0 (below), on the dropped rows: P per head against the
next layer's best-matched head chosen leave-one-prompt-out, the cell value the entering heads'
weighted mean, N1 (layers ≥ 2 apart), N0, the ordered partition, the beside columns
(`design-B7.md` "Objects, fixed here"; `b7_activity.py` choices 1–19).

**Why these.** ¼ is placed against the null: under B7's planted nulls one head's largest key share
is about 0.05 at full rows and 0.12 at 50 rows over all 2688 heads of the 7 prompts in 20 draws
(check 4), so the union over heads finds nothing in a null prompt. It sits below what B7's entering
heads put on the modal key (median 0.50–0.83 at 2000, 0.85–0.94 from 8000), but that table pooled
the heads entering each cell (`b7_diag.py prompts`), so it is **not R2's own count, and no gap at ¼
on R2's count has been seen** (`/challenge-pr` on #278, finding 1: the first version said ¼ sat in
a seen gap, and counted a layer's 16 heads, while at 2000 `homer_iliad`'s key was the modal
partner of only 63 of the 224 heads in 7–20). Per head, because that is how B7 reads a cell (a
head enters on its own rows) and a sink held by 4 of 16 heads is below ¼ of a layer. The near-miss
share (beside 6) shows on real data whether a key sits just under ¼. Per head and then a union,
rather than ROADMAP's example (the modal key over layers 7–20 at ≥ ½): 7–20 and ½ are B7's own seen
values, and secondary sinks arise in middle layers and persist a variable number of layers
(2512.22213), so a fixed window would miss one that lives in 9–12. One set per prompt, so every
layer pair, N1's distant pairs included, reads the same rows. Keys dropped, not rows: a row on the
sink with a content partner under it is read (check 2), a row with nothing under it goes diffuse
and is out.

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
unit 2; `status-B.md` "Corrections received" 2026-10-16), not a reading of β; (6) *(#278 finding
2)* **the near-miss share**: per (prompt, step, layer, head) with ≥ 50 condensed rows, the share
of its largest key *not* dropped, with the count of heads in [⅛, ¼), so a sink R2 missed shows;
(7) *(finding 4)* per cell, the quartiles of the kept rows' `w_i` after the drop (the mass a kept
row rests on, down to 0.05); (8) *(finding 5)* **a key-agreement null N0k**, N0 with keys in place
of offsets (pairs of different rows of i's bin whose partner keys agree), and each cell flagged
where `P − N0`'s interval lies above 0 and `P − N0k`'s does not: persistence a fixed key explains,
a residual sink. N0k changes no description.

## The math

`tools/math_checks/b7p_sink_exclusion.py` (`sympy` + numerics, 5 checks pass): (1) dropping S
scales every kept entry by 1 / (1 − s), so the order among kept keys and any condensation on a kept
key survive; (2) a planted row with 0.7 / 0.2 / 0.1 of its mass off position 0 on the sink, a content
partner and uniform is condensed on the sink before the drop (0.7 + 0.1 / i) and on the partner
after ((2i + 1) / (3i − 1) > ½); a 0.8 / 0.2 row is condensed on nothing after (1 / (i − 1)); (3)
B7's N0 on bins of frozen rows on one key per prompt is 0.014 with P = 1, which is why B7 read
those cells "beyond position"; (4) under G1, G2, G4 at the real n's one head's largest key share
is 0.049 at full rows and at most 0.120 at 50 rows thinned over 53 760 heads, against ¼; (5) the
U-statistic identity Σ_{j ≠ j′} m_j m_j′ x_j y_j′ = (Σ m x)(Σ m y) − Σ m² x y, and at 50 rows a
draw's N0 over different original rows stays at the reshuffled value (0.0080 against 0.0087)
while counting copies puts it at 0.24.
**What it does not prove:** that 410m's sink rows are check 2's (what is under the sink is read);
that ¼ separates sinks from content in 410m (a content key a quarter of one head's rows point at
is dropped, whatever its token, and R6 names it; the near-miss share says what sits under ¼); that what is left is content and not position (N0
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
In a heavy prompt's sink layers each head h gets its own sink share s_{q,h} ~ U(0.5, 0.95). **Which
rows sit on the sink** *(fixed here, #278 finding 3)*: each row i > k carries one uniform u_i per
layer, kept from layer to layer with probability 0.8 and otherwise redrawn (G3's chain, on sink
membership), and is on the sink in head (l, h) when u_i < s_{q,h}; so sink rows persist more to the
next layer than to distant ones, as B7's L cells did. A sink row is *hidden* (0.7 of its mass off
position 0 on k, 0.2 on its content partner, 0.1 uniform over keys 1..i) or *pure* (0.8 on k, 0.2
uniform), fixed per row, half each. In the five *light*
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
| G10 sink over reshuffled | the plant, content partners as G2 | (a) S_A = {k_A}, S_B = {k_B, k_B′}, every light S_q empty; (b) **B7's instrument without the drop reads ≥ ¾ of cells 7 → 8 … 19 → 20 "depth-local persistence"** (the arm reproduces B7's confound as B7 read it; otherwise it does not plant what B7 saw, and the plant is revised before any real cell; as revised, #278 finding 3: "L or F" let either membership model pass); (c) with the drop, G2's pooled band \|Σ P − Σ N0\| ≤ 0.25 Σ N0 and every read cell "rerouting"; (d) ≥ 0.99 of hidden rows condensed on their planted partner after the drop, no pure row condensed |
| G11 sink over frozen | the plant, content as G1d | (a), (d) as G10; with the drop, G1's condition (P ≥ 0.99, N1 ≥ 0.99, `P − N0` ≥ 0.9, every cell "frozen across depth") at full rows and the minimum entry |
| G12 sink over drift | the plant, content as G3d | (a), (d); with the drop, G3's band on (P − N0) / (1 − N0) at full rows and the minimum entry |
| G13 false persistence | G10 at full rows and at the minimum entry, 20 seeds | a read cell described as anything but "rerouting" in ≤ 5 % of cells at both (G7's rule) |
| G14 head-sparse sink *(#278 finding 2)* | G10, but in each heavy prompt's sink layers only 3–6 of 16 heads (drawn per layer) carry the sink, at s_{q,h} ~ U(0.3, 0.6); a layer's share then about 0.06–0.22, which a per-layer count at ¼ would miss | G10's (a), (c), (d), and (b) beside |
| G15 sub-threshold leak *(#278 finding 2)* | G14 with s_{q,h} ~ U(0.10, 0.20), below R2 in every head | **no pass condition: it measures the rule's blind spot.** Recorded: S_q (expected empty), the share of cells 7 → 8 … 19 → 20 read other than "rerouting" with the drop, the near-miss counts (beside 6) and the N0k flags (beside 8). Unit 2 prints the real near-miss counts beside this arm's, so a real prompt that looks like G15 is named |
| G8 on real | B7's G8 (143000, layers 10–11, heads 0–3 one shared fixed map, keys outside S_q) with R1–R5 | S_q the same with and without the plant; G8's condition against the unplanted sink-excluded run; records differences only, and S_q (an input, not a cell) |

An arm that fails changes the constants, with the reason recorded, before any real cell is read
(B2's rule 6); a failure of G10 (b) changes the plant, not the rule. G10–G12 and G14 also run as
code tests on a small synthetic cloud.

## Seen before the rule was fixed

B7's grid, its sink table (the modal key, token and shares by prompt at 2000, 8000, 143000, over
the heads entering each cell), its caveats and its records' scaled-row counts (`status-B.md` "B7 on
one page", "B7 unit 2"); the math check's synthetic values; the scan's summaries; the reviewer's
count on #278 (at 2000 `homer_iliad`'s key the modal partner of 63 of the 224 heads in 7–20).
**Not computed on real attention:** any R2 share, near-miss share or S_q, any sink-excluded
partner, P, N0, N0k, cell or count. G8 will compute S_q at 143000 in unit 1, before unit 2.

## Units

| unit | what | done when |
|---|---|---|
| 0 | this design, the scan, the `sympy` check | frozen (Blocked 114) |
| 1 | `trackB_circuit/b7p_sink.py` (imports `b7_activity`, unchanged): R1–R6, N0 per draw, the beside columns, G0–G15, code tests | every arm with a pass condition passes, or a recorded change; G15 recorded |
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

1. **¼ per head, placed against the null, not in a seen gap** (as revised, #278 finding 1). The
   first version counted a layer's 16 heads and said ¼ sat in B7's seen gap; that gap was over
   entering heads only. Per head is closer to how B7 reads a cell and catches a sink held by a few
   heads (G14), at the cost that one head's focus on a content key drops it prompt-wide (R6 names
   it). The alternatives are ROADMAP's ½ over layers 7–20 (B7's seen window and value), per layer
   at ¼ (misses G14), or computing R2's shares on real attention before the freeze (it would place
   ¼ on seen data). ⅛ and ½ are beside as counts; G15 and the near-miss share say what lies under ¼.
2. **One sink set per prompt, every layer.** A key that is a sink in 9–12 only is also dropped at
   layer 22, where it may be a content partner; per-layer sets would keep those rows but give every
   layer pair, and each of N1's distant pairs, its own rows. Union chosen for one row set.
3. **Keys dropped, not rows.** Dropping the rows on the sink would also drop the hidden partners
   under it and condition on the sink; dropping the keys reads them (check 2), at the risk that a
   row's remaining top share > ½ is the uniform tail's (check 2: never at i ≥ 16).
4. **G10 (b) as a pass condition, on L.** The arm must reproduce B7's failure as B7 read it before
   its fix is trusted; the alternative is to read (b) beside. Chosen because B7's arms passed while
   missing the real data's shape (`LESSONS.md` 6). With the per-head rule, (b) is still expected to
   pass easily in G10 (shares 0.5–0.95); G14 and G15 are where the rule is tested near its edge.
5. **N0 per draw is in, for every cell.** It widens `P − N0`'s interval; B7's 7 R cells are the
   ones it could move. The alternative is B7's fixed-profile N0 with the per-draw one beside;
   rejected because B7's close carried it here as part of the interval.
6. **3 units with a reserve**, with 2 as the alternative above.
7. **The long prompts out of the budget.** They add content rows at the cost of forward passes; the
   7 prompts are read first, so the long-prompt decision is made on what is left.
