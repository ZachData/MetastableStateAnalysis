<!-- p10_cluster_function/design-10.md -->
# Phase 10 — the re-read design (on 1d's working definition)

**Status: frozen 2026-10-05.** The user merged #144 and asked for R0, which Claude took as
accepting it (`STATE.md` Blocked 23). R0 has run (`status-10.md` §1.14); its two corrections
are marked in place below. Tier 1, exploratory, unregistered:
`claims/registry.json` is untouched and no row here adjudicates anything.

**What this is.** `handoff-10.md`'s banner lifts the hold for a re-read: every Phase 10 row
read one `HDBSCAN(min_cluster_size=2)` partition, and 1d now gives a working definition
(`p1d_cluster_ensemble/design-1d.md` "The working definition"; counts `status-1d.md`
"Blocked 22 decided"). This file fixes, row by row, what is re-read and how, on which
checkpoints and prompts, in what order, and what is read beside each row. It does not
change the definition. Literature scan: `lit-10.md` §16 (done first).

## Inputs, fixed here

| point | value | why |
|---|---|---|
| model | `pythia-410m`, seed 0 (the original run) | Phase 10's model; the definition's "replicates" column exists only where 1d ran seeds |
| checkpoints | **all 18 distinct Stage 0 steps**: 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000 (step 1 = step 0's weights, `PROJECT.md` §3.51.3) | the rows' findings sit at 32–512 (F1, §1.5), at 512 (§1.9–§1.10) and at 2000–4000 (A0); a thinned set would miss one of them. Change is fastest in the first 1 % of training (`lit-10.md` §16 row 1) |
| prompts | the **7 v1 passages** with a cloud: `repeated_tokens` has none after T3; `homer_iliad` at its first 512 tokens, as every v1 run read it | the 12 new v2 prompts stay held out (`core/holdout.py`); the definition was built on v1 |
| run dirs | Stage 0's v1 directories, **selected through `stage0_logs/stage0_index.json` only** | one input set for every row (the older readers' default glob mixes sweeps; `status-10.md` §0) |
| token rules | T1–T3 of `design-1d.md` "Token rules"; **T2's union over all 18 checkpoints** per prompt, so a prompt has one token set along the whole axis | massive tokens appear during training (`lit-10.md` §16 row 6); a per-checkpoint T2 would change the token set along the axis being read |
| layers | L1–24 (L0 has no cloud under the definition) | definition scope |
| label rule | per (step, prompt, layer): a token in a definition group carries its group id; a kept token in no group is **−1 (noise)**; T1–T3 tokens are **outside the domain** (a `kept` position list per record, not a label) | groups are disjoint (`admit.level_set_hdbscan` returns one flat labelling), so this is a partition of the kept tokens. Assigning the −1 tokens to a nearest group was rejected: it invents membership the definition did not give |

**T2 over all 18 checkpoints** gives the same two tokens as over steps 0 and 143000
(`hdbscan_code` offset 34, `latex_monograph` offset 10, both above 10× from step 8000;
read from Stage 0's stored norms by `/challenge-pr` on #144, no forward pass), so unit 1's
records and the 62 / 810 counts are on the re-read's token set.
*Corrected at R0 (2026-10-05):* the 62 / 810 counts are on **unit 2's** token set
(`arch_null_trained_2026-10-02/step0/token_sets.json`, sha `b1eaa3ab`), the union over unit
2's comparison (10 seeds at steps 0 and 143000, 40 re-inits). Beyond the two `\n` tokens it
also drops each prose passage's first `.` (massive in seed 7's trained run). The re-read
uses that set (`move_text run --kept-from`), so the counts, the "learned" join and the rows
share one token set. The 18-checkpoint union is checked as a subset at every step:
`--kept-from` refuses a checkpoint whose own massive tokens fall outside the set.

**The check that joins the two passes.** "Moves" comes from unit 1's own P = 0 pass; the rows
read Stage 0's stored activations. Per (step, prompt), the P = 0 cloud must match the stored
`activations.npz` within **1e-5** (placed) in direction and relative norm, or that (step,
prompt) refuses. **Measured on unit 1's stored P = 0 states against Stage 0, 7 passages
(*added after `/challenge-pr` on #144, finding 3*):** max 2.6e-8 (direction) and 1.3e-7
(relative norm) at step 0; 2.9e-7 and 1.3e-7 at step 143000. The tolerance is 35× the
largest. The cross-sweep drift the review cites (pilot vs Stage 0, up to 7.9e-5 after step
1000) is between two sweeps, not between a re-run and its own sweep.

## The ladder: what is read beside every row

One column per choice, so a changed row says which choice changed it (`lit-10.md` §16 row 7).
*Rebuilt after `/challenge-pr` on #144, finding 2*: the first version moved four choices in
one column (tree, frame, stability, bulk) and dropped L0 inside c0 → c1.

| col | partition | the one choice it adds |
|---|---|---|
| **c0** | the stored labels, the row's reader **re-run unchanged on the 7 prompts, Stage 0 dirs, L1–24** (F1: boundaries from L1 → L2) | none: the old reading on the re-read's inputs (the published number pooled 8 prompts, another sweep, or L0) |
| c0f | the same shipped call on **float64** distances, every stored position (*added at R0 after `/challenge-pr` on #145, finding 1*: c0 was fitted on float32 distances in all 380 Stage 0 runs, so c0 → c1 moved two choices) | precision |
| c1 | shipped `HDBSCAN(min_cluster_size=2)`, float64, kept tokens (T1–T3; A0 also T4), the stored (uncentred) frame | the token rules |
| c1c | as c1, **centred** frame | the frame |
| c2a | level-set groups, all of them, centred | the tie-free tree (`admit.layer_groups` returns c1c and c2a from the same distances) |
| c2b | c2a minus bulk groups (their members join the rest) | bulk. Probably the largest step on the binary rows, so it gets its own column |
| **c2** | c2b minus unstable groups | stability: the definition without the move filter |
| **c3** | c2 ∧ moves: **the definition**, the primary column | the filter |
| floor | the count of c3 records at step 0 | the definition at init (62 step-0 records on v1, against 810 trained); a count, not a baseline (below) |
| arms | c3 at centred size 4, raw size 2 | quoted only where a reading label differs from c3's |
| learned | c3 split by unit 2's learned bar, at step 143000 | "learned beside, not required" (Blocked 22 (a)); at other steps only if the bar reads from stored re-init records without new passes (the build checks; otherwise 143000 only) |

**Step-0 baselines are read on c2, not c3** (*changed after `/challenge-pr` on #144, finding
1*). The move filter removes almost all of step 0 by design (62 of 520 stable step-0 groups
pass, against 810 of 926 trained), so a baseline on c3 at step 0 would be the few chance
passes. Every "Δ vs step 0" and "vs baseline" below subtracts c2's step-0 value, at every
column; c3 at step 0 is reported as the floor count only.
*Departure at R1 (2026-10-05, flagged by `/challenge-pr` on #146; **accepted by the user
2026-10-05**, `STATE.md` Blocked 24 (b)):* the ladder subtracts c2's step 0 from c3, the arms and
the learned split, and each of c0–c2b's own step 0, so c0 stays the published reading. Read
literally, "at every column" puts c2's step 0 under c0 too; `p10_r1_ladder.py` counts both
(`status-10.md` §1.15). *From R2 on (user, Blocked 24 (a)):* the raw value and the baseline are
printed beside every Δ.

**Readable.** A (prompt, layer) record is readable when it has **≥ 10 members and ≥ 10 of the
rest** (placed). At a step where fewer than half its records are readable on c3, that step's
reading is reported on c2 and labelled so; an unreadable record is counted, never read as
"no effect". **A row "holds"** when its reading rule (below) gives the same label on c3 as on
c0 (A0: on c1), at every step and layer the rule names; when it does not, the first column
whose label differs is named. One rule, applied per row. The re-read's binary statistic is
called **members − rest**, not "clustered − noise": the rest is kept tokens in no group of
the column, a different population from HDBSCAN's −1.

## Row by row

| row (`status-10.md`) | what its reader takes | re-read? | how, and its reading rule |
|---|---|---|---|
| **§1.7**, unique-token part (rank does nothing; class does, weakly) | clustered rate of unique tokens by rank bin and class (`p10_token_composition`) | **yes**, unique tokens only | member share among kept unique tokens by rank and class; §1.7's own rule. Its copy-count headline is out (T3 keeps one copy). Not in `design-1d.md` "Scope", which listed §1.6–§1.8 together |
| **§1.9** co-membership | focal = clustered unique token at position > 0; co-members; 5 lifts against a random draw (`p10_comembership`) | **yes** | draw from kept tokens only; layers 12, 24 and the mean over L1–24 (L0 out); rule as written there: Δ vs step 0 above / as / below at ±0.05. `copy_share` now means "the co-member's string recurs later" (the copies are dropped); `adjacent` counts a ±1 neighbour only if kept. Δ subtracts c2's step-0 lift (above), so the rule does not change at layers where c3 has no step-0 focal token |
| **§1.10** lexical carry | as §1.9, plus each token's layer-0 vector (`p10_lexical_carry`) | **yes** | layer 0 enters as a covariate (the frozen frame), not as a cloud, so the definition's L0 gap does not bite; rule as written there |
| **F1** clustered − noise step (the 32–512 window) | binary labels at each layer boundary, permutation null (`tools/run/transport.py`) | **yes** (correcting `design-1d.md` "Scope", which said F1 needs a full partition: its labelled statistic reads clustered vs noise only) | members − rest, permuted among kept tokens, 2 000 draws. Label per step: **negative** (median p ≤ 0.05, sign −), **positive**, or **none**; with the readability rule above, so "none" at 0–16 is a reading on enough members, not an empty column. The identity-coupling result is partition-free and is not re-read. *Found at R2:* the rule reads significance, not size; on c0 it is negative to 54000, so §1.3's 32–512 window was a magnitude reading (`status-10.md` §1.16) |
| **F12** members − rest, **against its step-0 baseline** (§1.4's "vs baseline" column, the finding; the sign alone is "mostly definitional", §1.4) | binary labels, `Z` per token (`p10_partition_function`) | **that column only** | per step, the statistic minus c2's step-0 value; label **below / as / above** baseline at ±0.05 (placed, as §1.9's floor). c0's labels: below at 32–4000, as at 8000, above from 16000. *Rule replaced after `/challenge-pr` on #144, finding 1*: the first version tested the sign. Raw `log Z` (99.5 % position) and the sink results are out (T1 drops the sink) |
| **§1.5** F1 + F12, "parked, not pinned" at 32–64 | the two above | follows them | read off F1 and F12's re-read; no reader of its own |
| **A0** attention flip vs the causal mask | attention received, binary labels, mask baseline (`p10_attention_baseline`) | **yes, with T4** (sink columns dropped, rows renormalised) | members − rest. T4 changes the statistic itself, so **c3's labels are compared with c1's, not c0's**. Labels: mask share **≥ 0.9 or not** (0.9 placed, below c0's ~0.94); the step interval where the learned residual first appears (c0: 2000–4000); whether it persists to 143000. *As run (R3, 2026-10-05):* the share is the sweep's (pooled as §1.1), labelled **no flip** where the raw gap is ≤ 0; "residual" is corrected gap ≥ 0.05 (placed, as F12's floor; the published per-step E never rejects, so §1.1's interval was a size reading, and this floor reproduces it on the published record). Under T4 every column reads no flip (`status-10.md` §1.17) |

**Not re-read, and why.**

| row | why |
|---|---|
| §1.6 (`ext_sem_threshold`), §1.8 (step 0 L0, `max_alive`) | repeats and L0 (`design-1d.md` "Scope"); the definition cannot see either |
| §1.7's copy-count headline | T3 keeps one copy of each string |
| §1.11–§1.13 (the pilot sweep) | they asked whether a second partition of the same activations agrees; the level-set route on float64 is tie-free, so the question does not arise. The pilot's 14 extra steps are not on Stage 0's axis |
| F0, the anchor test | superseded by F13, which needs no partition (`lit-10.md` §11.4) |
| `CLAIM-C`'s two HDBSCAN metrics | a registered gate's instrument; changing its partition is the user's call, not a tier-1 re-read |
| F13–F20 | partition-free or not yet run; not re-reads |

## Checkpoints: the cost

Unit 1 has run at steps 0 and 143000. "Moves" at the 16 others needs its preamble passes,
EOD join only: 7 + 3 P × 18 (passage, preamble) pairs = **61 passes per checkpoint, 976 in
all**. Unit 1 ran 290 passes in ~16 min at 14 workers, so about an hour, an estimate; all 19
checkpoints are cached. Records only: no activations are stored beyond Stage 0's (unit 1's
P = 1000 activations on `HDD_1TB` stay its own). **Gate before the batch:** run two
checkpoints end to end, 512 and 54000 (the latest new one, where any drift in the P = 0
match would show), open their records, and check they are populated (groups per layer,
classes, the P = 0 match) before launching the other 14.

*As run (R0, 2026-10-05):* `move_text` runs both joins (the `\n\n` join is "beside, per
group" in the definition, and steps 0 and 143000 carry it), so 7 + 2 × 54 = 115 passes per
checkpoint, and steps 0 and 143000 are re-run too, so one directory holds all 18 steps from
one commit, each with its P = 0 match. Timings: `status-10.md` §1.14.

Rejected: a thinned axis (e.g. 0, 32, 512, 4000, 143000), which saves ~45 min and loses the
window edges (16 → 32, 512 → 1000) that F1's reading rule is stated on.

## Order

| # | unit | needs | why here |
|---|---|---|---|
| R0 | **build**: unit 1's runner at the 16 checkpoints; a labels source with the `kept` domain and columns c1 → c3, which the readers take by flag and refuse without | ~1 h of forward passes | every row reads it |
| R1 | §1.7 (unique), §1.9, §1.10 | R0 | the thread's own question (Stage 1: what is in a cluster); their focal unit, unique tokens off position 0, is already what the definition keeps, so they change least in meaning |
| R2 | F1 + F12 (its gap against step 0), then §1.5 | R0 | the phase's one dynamic finding, and its density confound was argued from step 0; the definition's step-0 floor is a stronger control than the old partition had |
| R3 | A0 under T4 | R0 | last: T4 changes the statistic, so it is the least comparable |

| R2m | F12 against a matched control, §1.5 again (below) | R2 | R2 left F12's label as the baseline's (`STATE.md` Blocked 25, (iv) taken by the user 2026-10-05) |
| R6 | the cross-checkpoint matcher (below) | R0 | every later row that follows a group through training needs it (`STATE.md` Blocked 26 (b)) |
| R6w | §1.9–§1.10's drift within lineages or by replacement (below) | R1, R6 | the question the matcher was parked for (`STATE.md` Blocked 26 (b′)) |
| R6f | R6w's within split into members against frame, in one fixed layer 0 (below) | R6w | R6w's "within" cannot tell the groups moving from the frame moving under them (`STATE.md` Blocked 26 (b″)) |
| R7 | the context-shuffle test: does a group need its context or only its tokens (below) | R0 | every row so far reads composition and the embedding; none asks whether a group survives losing its context (`STATE.md` Blocked 26 (c)) |
| R8 | the own-floor check on "moves": does the definition's move filter pass groups by chance (below) | R0, R7 | R7 found `J0` below chance for 32–46 % of c3's groups, and "moves" uses the same `J0`; the one item that could change c3 itself (`STATE.md` Blocked 26 (f)) |
| R9 | every re-read row again with c3x leading (below) | R8x | R8x's drop held by the rule's letter; the user took c3x as the definition's column (`STATE.md` Blocked 27 (a)) |

**R2m, F12's matched control (rule fixed 2026-10-05, before any control output was read).**
Each column's readable (prompt, layer) records at step s are scored on **step 0's stored
activations** for the same prompt: same positions (`tokens.txt` must match, else refuse), same
statistic (members − rest in corrected log Z over every stored position, on the column's
domain at s), same β grid, same unit generator. Per unit, **matched Δ = trained − control**; per
step, the mean over paired units, labelled **below / as / above control** at ±0.05 (F12's
floor). Δ is 0 at step 0 by construction (a check, not a reading). Beside every Δ: the raw sign
(Blocked 25 (ii); trained mean, below / as / above the rest at ±0.05), the control's mean and
median p (did these tokens already differ at init?), and the prompts with mean Δ < 0, of 7.
Primary column, "holds" and §1.5 (F1 negative ∧ matched below; F1 is R2's, unchanged) as R2.
c0 is read on the same control, so c0's matched label is new too; R2's labels stay as recorded.
It does not control: step 0's context (Z at init is computed from init activations at every
position, so the control asks whether those positions were already dense at init, not whether
training changed their neighbours), the norm (§1.5's random-twin hazard), or the init seed (one).

**R5c, 5c's flip with the sink out (`STATE.md` Blocked 26 (e), taken by the user 2026-10-05;
rule fixed before any output was read).** Inputs: `gpt2-large` and `gpt2-large-random`, 21
prompts each, the stored runs `data/phase12/2026-09-19_13-40-48/` and `…_15-47-26/` (the
9-prompt pair 10-51-00 / 11-52-10 is checked for identity on its overlap, not read twice). Labels:
each run's own `hdbscan_labels.json`, the partition 5c read; attention block L paired with labels
at L, L = 0–35, as `noise_importance_proxy`. **T1–T2** per prompt: position 0 ∪ every position
whose residual norm exceeds 10× its layer's median at any hidden-state layer 2–32 (Pythia's 2–20
of 24, the last four left out the same way), **union over both arms**, so trained and random drop
the same positions. Arms as R3's sink split: `all` (5c as published), `drop` (T1–T2 out of the
means, attention untouched), `drop+T4` (their columns also dropped, rows renormalised); each raw
(diagonal zeroed, 5c's statistic) and causal-mask-corrected. Per unit, gap = unclustered −
clustered enrichment; a prompt's gap is its mean over readable layers; **the prompt is the unit**
(layers within a prompt are not independent). **Primary: `drop`, raw.** **Survives** iff the
trained mean gap > 0 with ≥ 16 of 21 prompts positive **and** trained − random > 0 in ≥ 16 of 21
(one-sided sign test, p = 0.013 at 16); **sign-flipped as 5c stated** if, in addition, the random
mean gap < 0. Otherwise **does not survive**. Beside: `all`'s enrichments against 5c's 1.6× /
0.5× (5c's prompts and date are unrecorded, so a mismatch is reported, not fixed); the share of
units where every T1–T2 position is unclustered (the Parked question); means by third of depth;
the corrected gap in R3's 8 position bins. It does not control token frequency
(`archive/p5c_unclustered/lit-5c.md` §1), and the random arm is a default re-init, not
norm-matched. **5c's own random arms cannot be this one:** neither `gpt2-large-random` nor
`albert-base-v2-random` could load through `run_1` before 2026-09-17 (`e191d77`), so both are
of unknown provenance (ALBERT's added after `/challenge-pr` on #150; not part of the rule).
**ALBERT is not read here:** no run survives, and its sinks ([CLS], [SEP]; bidirectional; 60
shared-weight iterations) need their own definition (`handoff-10.md` Parked).

**R6, the cross-checkpoint matcher (`STATE.md` Blocked 26 (b), taken by the user 2026-10-05;
rule fixed before any matcher output was read).** An instrument with a first descriptive
reading, not a re-read: no claim is labelled here. **Input:** R0's label source only; columns
**c0** (stored, every position: the old partition), **c2a** (every level-set group, no filter),
**c2** and **c3** (the definition), each over its own domain, which is the same at every step
of a prompt (one token set). **Pairs:** the 17 adjacent steps of the 18, same (prompt, layer
L1–24); every record is read (no readability bar: the matcher compares groups, not members
against the rest; c3's 10–34 readable records at 0–32 are reported as counts, not as a
reading). **Link:** `merge_tree.link_layer_pair`, **containment ≥ 0.5** (overlap over the
smaller group), components classified stable / split / merge / tangle / birth / death: MONIC's
taxonomy (`lit-10.md` §16 row 2), whose own overlap is relative to one cluster, not Jaccard.
Rejected as the link: Jaccard (`STATE.md`'s wording), which records a piece leaving a group as
a death plus a birth (`status-1d.md` "Merge tree", the withdrawn #104 headline); Hungarian
one-to-one at r ≥ 0.5 (`lit-10.md` §16 row 1), which cannot report a split. **Beside:** on
every stable link, the Jaccard (identical sets; ≥ 0.5, unit 1's and row 1's bar), so
"survives" splits into "same group" and "changed members". **Filter flips** (c2 and c3): a
birth or death in the column whose group id (shared with c2a) is linked in c2a's matching at the
same boundary is a flip (the group was there and failed a filter), else **new / gone**.
**Null:** `merge_tree.link_chain_null`, per (prompt, layer) chain and column, 200 draws (seed
0), each step's labels permuted within its domain (sizes kept, which tokens share a group
destroyed); per boundary, the pooled stable count against its draws (rank p). **Lineage:** for
each group at 143000, the earliest step reached by an unbroken chain of stable links back from
it, on its own column and on c2a (by id). **First check (gates the rest; refuse rather than
degrade):** at 0 → 2 (weights barely move; R0's c1 counts 1000 → 997), c2a's stable share of
step-0 groups ≥ 0.9 (**placed**); if it fails the run stops and the labels are checked.
**Readouts:** per boundary and column, pooled over 7 prompts × 24 layers, with prompts beside:
the share of the earlier step's groups in each kind, births, the stable links' Jaccard
(median, share identical, share ≥ 0.5), flips against new / gone, and the null's p; per column,
the lineage-origin histogram at 143000. The churn at 0 → 2 and 2 → 4 is the matcher's own
floor on that column (no learning there; *corrected at reading: 2 → 4 already moves, so only
0 → 2 is a floor*, `status-10.md` §1.20), and c3's step-0 groups are mostly chance passes
(R0's floor), so c3 churn below 64 is the filter's, not the model's. **It does not:** follow
a group across layers (`merge_tree` does, per step), read §1.9–§1.10's drift within
lineages (the next row that needs this), or say why a group changed. *After `/challenge-pr` on
#151 (not part of the rule as fixed):* columns c1 and c1c added (c0's call on the kept positions,
so c0's lineages compare on the same tokens); flips split into **kept** (c2a's component 1–1) and
**restructured** (c2a split / merge / tangle); lineages read beside an independence baseline (the
product of each crossed boundary's backward survival rate); a c2 / c3 id missing from c2a and a
missing `summary.json` refuse.

**R6w, §1.9–§1.10's drift within lineages (`STATE.md` Blocked 26 (b′), taken by the user
2026-10-06; rule fixed before any output was read).** The question parked with the matcher: when
a §1.9–§1.10 lift changes between checkpoints, do the groups that persist change, or do the groups
that die differ from the ones born? Descriptive, tier 1, no null; it labels where the drift
happens, not whether it is real (R1 read that).
**Statistics** (per focal token, as R1's readers compute them, all-positions draw): §1.9's five
lifts (`same_class`, `copy_share`, `no_copy`, `adjacent`, `emb_pct_own`; `emb_pct`, the frozen
frame, out as in §1.9) and §1.10's `emb_given_class` (10 bins) and **CGE40 − kNN40** (the
deciding reading, per focal token). Carry is out: it compares focal with unclustered tokens,
which belong to no group.
**Columns:** **c3** (primary), c2a (no filter) and c0 (the published partition) beside.
**Spans:** primary **512 → 143000** (8 boundaries; on c3: §1.10's class beyond the lexical
cluster +0.16 → +0.04, `emb_pct_own` and `emb_given_class` rising, `no_copy` rising at 2000;
`same_class` flat, the row with no drift); secondary **64 → 512** (3 boundaries; c3 is primary
from 64). **Records:** the (prompt, layer L1–24) records readable on the column at **every**
step of the span (count reported), so each step's population is the same on both boundaries it
borders and the sum over boundaries telescopes; R1's own change over the span (all readable
records per step) is printed beside, and a label differing between the two is named.
**Weights:** each focal token carries its R1 aggregation weight (1 / (records at its layer ×
focal tokens in its record); for the layer mean also 1 / layers present), so the weighted mean
is R1's value on this record set. **Kinds:** a focal token's group, from R6's `link` on the
column's full labels (containment ≥ 0.5): at the earlier step **stable**, **restructured**
(split / merge / tangle) or **death**; at the later step stable, restructured or **birth**. On
c3, a birth or death splits by R6's `flips`: **flip kept** (c2a 1–1; the group was there and
passed or failed a filter), **flip restructured**, **new / gone**.
**Decomposition** (the form of Melitz & Polanec's dynamic Olley–Pakes decomposition, 2015,
recalled, not re-read; exact): with M the weighted mean, S the stable tokens' mean, X_k and
w_k a kind's mean and weight share, at the earlier step s and later step t,
M_t − M_s = **(S_t − S_s)** [within] + Σ_k w_k,t (X_k,t − S_t) − Σ_k w_k,s (X_k,s − S_s)
[one term per non-stable kind]. Within is the persisting groups' change; each other term is
how far that kind of entry or exit moves the mean, measured against the persisting groups.
Summed over the span's boundaries, per term. Read at L12, L24 and the layer mean, pooled; and
per prompt (layer mean), for counts. A boundary with no stable focal token at a level has no
S: the run reports it and that level's span is unlabelled.
**Labels** (placed): only where |total over the span| ≥ 0.05 (§1.9's floor); otherwise "no
drift" and the terms are reported unlabelled. Within share = within / total: **within** if
≥ 2/3, **replacement** if ≤ 1/3, **both** between. Beside: the largest replacement term by
size (on c3: flip kept, flip restructured, new / gone, restructured), and the prompts whose
within term has their own total's sign, of those with |total| ≥ 0.05.
**Beside, within only:** each stable link's paired change (the group's focal mean at t minus
at s), split into **identical** links (Jaccard 1) and changed ones. On an identical link
`same_class`, `copy_share`, `no_copy` and `adjacent` cannot move (the domain is one token set
along the axis, so co-members and the draw are the same), and `emb_pct_own`, `emb_given_class`
and CGE40 − kNN40 move only because each step's own layer 0 moves. So within on identical links
is the embedding drifting under fixed members.
**First checks (gate; refuse rather than degrade):** (a) every per-record mean reproduces R1's
stored record for that column and step within 1e-6 (`data/p10/reread_r1_2026-10-05/cm_<c>.json`,
`lc_<c>.json`); (b) the terms sum to the total within 1e-9 at every boundary and level; (c) on
identical links the four composition lifts' paired change is exactly 0; (d) the matcher's
stable components per boundary equal R6's `r6.json` for the column.
**It does not:** test anything (no null; 0.05 and 2/3 are placed); follow a group across
layers; separate a stable group's member turnover from its frame beyond the identical / changed
split; say why. "Within" includes changed members (R6: median Jaccard 0.75–0.83), so a lineage
that swaps a quarter of its members per step counts as within. The reference (the persisting
groups) is one choice: entry and exit measured against the overall mean (Foster–Haltiwanger–
Krizan's form) would move weight between the terms; rejected because it makes "within" depend
on the entrants.
*At build, before any output was read:* a record must also have **≥ 1 focal token** at every
step (R1 drops a record without one, so "readable" alone does not fix the population). A first
run refused on a level with no such record (c3, L24 at 64–512). A level with none is now reported
as "no records" and not read. Results: `status-10.md` §1.21.
*After `/challenge-pr` on #152 (not part of the rule as fixed):* R1's own records per step read
beside, with records readable at one end only as a **records** term, so the span sums to R1's
change (check (e)); identical / changed links per boundary; an `opposing` flag; the runner's
dirty state recorded. The reviewer's main point is now in the reading: for statistics measured
in each step's own frame, a shift shared by every group lands wholly in "within", so "within"
there means "not replacement", not "lineage carries it".

**R6f, R6w's within split into members against frame (`STATE.md` Blocked 26 (b″), taken by
the user 2026-10-06; rule fixed before any output was read).** R6w measured each step's groups in
that step's own layer 0, so "within" mixes two changes: the groups' members changing, and the
embedding moving under them. Scoring every step's groups in **one fixed layer 0** holds the
frame still. Descriptive, tier 1, no null; it decides how §1.10's "trained, mostly what the
embedding groups" reads: the groups came to hold what the embedding groups (**members**), or the
embedding came to group what the groups already held (**frame**).
**Statistics:** R6w's three measured in the embedding: `emb_pct_own`, `emb_given_class` (10
bins), CGE40 − kNN40, per focal token exactly as R6w computes them (same members, pool, class,
weights, kinds) but with **frame F's layer-0 Gram** in place of the step's own. The four
composition lifts have no frame and are out. **Frames:** F = **512** and F = **143000**, the
span's two ends, for the same prompt (its `tokens.txt` must match, else refuse). **Span, records,
columns:** R6w's primary span 512 → 143000 only (its 64 → 512 is not readable on c3); R6w's
fixed record set (the rule) and R1's own records beside; c3 primary, c2a and c0 beside.
**Split** (exact, each of R6w's terms is linear in the per-token values at fixed weights and
kinds): for every term T (within, each replacement kind, records, the total), T = T_F + (T − T_F),
where T is R6w's own-frame term and T_F the same term on frame-F values. **Members** = T_F: the
change the groups' membership makes in a frame that does not move. **Frame** = T − T_F. Per
boundary and summed over the span.
**Labels** (placed, as R6w's): only where |within| ≥ 0.05 over the span (else "no drift");
members share = within_F / within. Per frame: **members** if ≥ 2/3, **frame** if ≤ 1/3, **both**
between. The row's label is the two frames' label where they agree, else **frame-dependent**
with both shares. Read on c3, layer mean, fixed set (primary); L12 / L24 not read (R6w: too
thin). Beside: the Shapley share (the mean of the two frames' shares, the two-factor split of
the span between members and frame); the total's split; per boundary, where each part sits; per
prompt (layer mean), the label count among prompts with |within| ≥ 0.05; on identical stable
links, members is 0 by construction (check), so their whole change is frame.
**First checks (gate; refuse rather than degrade):** (a) the own-frame per-record values
reproduce R1's stored records within 1e-6 (R6w's check (a)); (a′) with F = 143000, each record's
`emb_pct_own` equals R1's stored frozen-frame `emb_pct` within 1e-6 (`p10_comembership`, the same
statistic in step 143000's layer 0); (b) at step F itself, the frame-F values equal the own-frame
values exactly; (c) on identical stable links the three statistics' paired change in frame F is
exactly 0; (e) the own-frame span terms equal `r6w.json`'s within 1e-9.
**It does not:** say which frame is right (the two shares differ by the interaction of members
and frame, reported, not assigned); chain frames per boundary (one frame per run keeps "members"
on one yardstick over the span); test anything; follow groups across layers; read the
composition lifts, L12 / L24 or 64 → 512. One seed, 7 passages.
*After `/challenge-pr` on #153 (not part of the rule as fixed):* per-prompt values stored; per
prompt, the label is read on the two frames' mean (each fixed frame favours its own step's
groups), with an `opposing` flag (members and frame both ≥ 0.05, opposite signs). The pooled
labels and terms are unchanged. Results: `status-10.md` §1.22.

**R7, the context-shuffle test (`STATE.md` Blocked 26 (c), taken 2026-10-06; rule fixed
2026-10-06, before any shuffled pass ran).** Scan: `lit-10.md` §17 (and §12.2, §16 row 5).
**Question:** does a definition group need its passage's context, or only its tokens? (`handoff-10.md`
Parked, "Is the computed class grouping context, or a per-token feature?": §1.10 removes only
the token's own layer-0 vector, so a per-token feature an early MLP computes would look the
same as context.) **Inputs:** R0's labels and unit 1's records (each c2a group's member offsets
and own floor `J0`, centred, size 2); the 7 v1 passages at their 512-token cap; **all 18 steps**
(passes are cheap; the gate below runs first). **Conditions**, per (step, passage), position 0
fixed (T1 drops it; the sink token is the same in every condition), offsets 1 … n−1 rearranged:
- `orig`: the passage as read. Its pass must match Stage 0's `activations.npz` within
  `P0_MATCH_TOL` (unit 1's `p0_match`), else the (step, passage) refuses.
- `block b`, b ∈ {64, 16, 4, 1}: offsets 1 … n−1 cut into consecutive blocks of b tokens (the last
  shorter), the blocks permuted uniformly (redrawn if the identity), **K = 5** permutations per
  level, seeded by (passage, b, k) only, so every step sees the same permutations. A graded
  disruption with the unigram set fixed (`lit-10.md` §12.2); b = 1 is a full token shuffle (a
  bag of the passage's tokens); b ≥ 4 keeps most previous tokens and short words.
- `alone`: each kept token t as the 2-token sequence [passage offset 0, t]; its state at
  position 1 is the token with no context but the sink. One batched pass per (step, passage).
**Clouds:** in each condition, the same kept offsets (R0's token set) at their new positions,
layer ℓ ∈ L1–24, centred, level-set HDBSCAN size 2, **all** groups (no filter on the condition
side, as unit 1's P > 0 conditions). A kept offset that T2's rule (`move_text.massive_positions`)
marks massive in a condition is dropped from that condition's cloud and from the groups compared
in it (as unit 1's subsample floor restricts a group); counted. **Per group** g of c3 (c2 beside;
c2 at step 0 is the baseline, as the ladder), at each layer: J(g, condition) = best-match Jaccard
on offsets (`move_text.best_jaccard`). **Survives** a level when the median of its K Jaccards is
≥ g's own `J0` (unit 1's rule for "moves", the same yardstick); `alone` survives when J ≥ `J0`.
Fixed bar 0.5 beside. **Group label:** **token-borne** = survives `alone`; **bag-borne** = fails
`alone`, survives b = 1; **order-borne** = fails both. Beside: survival share at each b (the
dose–response), and the token-borne groups that fail b = 1 (counted, not relabelled).
**Reading**, per step, over c3's group-layer records pooled over prompts and L1–24: the three
shares; the step's label **token** if token-borne ≥ 2/3, **context** if ≤ 1/3, **mixed** between
(placed, as R6f's). Primary steps: those where c3 is readable (from 64, `status-10.md` §1.14);
c2 at every step beside, and c2's step 0 as the baseline. Beside: per band (L1–8, L9–16,
L17–24; detokenization sits in the first layers, `lit-10.md` §17 row 3); per prompt, the label
on its own records (prompts with ≥ 5 records only); per token, the centred cosine between each
kept token's state in a condition and in `orig` (Ethayarajh's self-similarity, `lit-10.md` §16
row 5), members against the rest; the same-class share of member pairs (`p10_token_composition.token_class`,
class from the **original** order) per label; a chance level per group (best-match Jaccard of 20
random same-size sets of kept tokens against the condition's groups, its 95th percentile) and
the count of groups whose `J0` is below it.
**First checks (gate; refuse rather than degrade):** (a) `orig` matches Stage 0 (above); (b)
`orig`'s groups are R0's c2a member sets at every layer (else that record refuses), so
J(g, `orig`) = 1; (c) each shuffled sequence is a permutation of the passage's ids with offset 0
fixed, and each kept offset's token sits at its mapped position; (d) the `alone` state of the
kept token at offset 1 equals `orig`'s position 1 (the same two tokens) within `P0_MATCH_TOL`.
*Departure at the gate (2026-10-06): (d)'s tolerance is 1e-4.* `latex_monograph` at step
143000 refused at 1.06e-5 (relative norm). An unbatched 2-token pass is 1.5e-5 from the
passage's pass there too, so the gap is float32 sequence-length noise, not the batch;
`P0_MATCH_TOL` was placed for re-running the same input. A wrong pairing or mask differs at O(1).
Results: `status-10.md` §1.23.
Run steps 512 and 143000 first and open the output (groups per layer, Jaccards not all 0 or 1,
the three shares) before the other 16.
**It does not:** say what a group carries (token-borne is not "lexical" and order-borne is not
"semantic": class is orthographic); control position inside `alone` (every token at position 1;
c3's groups survive moved offsets by construction, so absolute position is controlled for the
shuffles, not for `alone`); keep words whole at b = 1 (a word-level shuffle is Parked); use a
foreign context (other passages' tokens change the token set); test anything. One seed, 7 passages.

**R8, the own-floor check on "moves" (`STATE.md` Blocked 26 (f), taken 2026-10-06; rule fixed
2026-10-06, before any chance level was computed).** **Question:** does unit 1's "moves" (best
Jaccard ≥ the group's own `J0` at every P > 0 under every preamble, EOD join) pass c2 groups that a
random set of the same size would pass as well? **Inputs:** R0's unit 1 records and labels, all 18
steps, 7 v1 passages, centred, size 2, L1–24; no forward pass. **Chance level**, per (step, passage,
layer) and group size s: N = 2000 random s-subsets of the kept offsets (seeded by step, passage,
layer, s), each scored by best-match Jaccard against the **P = 0** partition (unit 1's stored
groups, every group). `Jc` = the smallest drawn value v with share(J ≥ v) ≤ 0.05 (the discrete
95 %; R7's interpolated 20-draw percentile is beside). *Proxy, stated:* unit 1 stored each P > 0
condition's group count `k` but not its labels, so P = 0's partition stands in for every
condition's; the median ratio of `k` to P = 0's count is reported to show how close they are.
**Chance-aware moves:** every EOD condition's J ≥ max(`J0`, `Jc`), then unit 1's `classify`;
**c3c** = c2 ∧ that class is `moves`. **Reading**, per step, c3's group-layer records pooled
over prompts: the share with `J0` < `Jc`; the median share of random sets clearing `J0`; c3c / c3;
by band and by size beside; step 0's floor (62 records) and 143000 (810) under both bars.
**Decision (placed):** if c3c keeps ≥ 90 % of c3's records at every primary step (64 on), "moves"
does not pass groups by chance at the definition's own grain and c3 stands; otherwise the
definition needs a chance-aware bar, c3c is the candidate, and choosing it is the user's (it would
change every re-read row's c3). **First checks (refuse rather than degrade):** (a) the stored P = 0
groups are disjoint and equal R0's c2a labels at every layer; (b) the recomputed c2 and c3 equal
R0's; (c) c3's totals match R0's (step 0: 62, 143000: 810). **It does not:** use the real P > 0
partitions (the proxy above); ask whether the floor `J0` itself is the right yardstick (its 10th
percentile is placed); test anything. Results: `status-10.md` §1.24.

**R8x, the exact check (`STATE.md` Blocked 27 (c), the user's, 2026-10-08; rule fixed before
any P > 0 label was written).** **Question:** with each P > 0 condition's own partition in place
of R8's P = 0 proxy, does c3 keep ≥ 90 % of its records at every primary step, and how large was
the proxy's error? **Inputs:** R8's (R0's unit 1 records and labels, 18 steps, 7 v1 passages,
centred, size 2, L1–24), plus unit 1's EOD-join P > 0 passes re-run: `move_text.conditions`'
EOD conditions (the preambles other than the passage, P 50 / 300 / 1000; 6 or 9 per passage),
R0's environment (CPU, float32, 14 torch threads, `OMP_NUM_THREADS=1`), each condition's
partition of R0's kept offsets by `move_text.groups_of` (centred, size 2), the labels written
per (step, passage, condition, layer). **Chance level** per condition: R8's (N = 2000 random
same-size subsets of the kept offsets, the discrete 95 %), against **that condition's** partition,
seeded by (step, passage, layer, condition, s). **c3x** = c2 ∧ unit 1's class is `moves` when each
EOD condition's J ≥ max(`J0`, that condition's `Jc`). **Reading:** R8's table with c3x beside
c3c; c3x / c3 per step with a prompt bootstrap (5,000 draws, 95 %, as R8); per step the records
where c3c and c3x disagree, each way; the proxy's error, `Jc` (exact) − `Jc` (proxy) per (record,
condition), and the condition's group count over P = 0's; band and prompt beside. **Decision
(placed; Blocked 27's "(c), then (a) if the drop at 64–1000 holds"):** the drop holds if c3x keeps
< 0.90 at any step of 64–1000 (R8's letter, the proxy replaced); then (a), c3x as the definition's
new column with every re-read row run again on it, is recommended to the user. If c3x keeps ≥ 0.90
at every primary step, c3 stands and c3x goes beside. The bootstrap intervals are beside, not in
the rule: R8 showed 7 prompts cannot resolve 0.90, and a rule that needed them to would not
decide. **First checks (refuse rather than degrade):** R8's three; per (condition, layer) the
recomputed partition reproduces unit 1's stored group count `k` and every stored best Jaccard
exactly, so the labels written are the ones unit 1 classified on; the first step run (64) is
opened and checked populated before the rest. **It does not:** re-examine `J0`; read the `\n\n`
join (not part of "moves"); change any other column; test anything. Results: `status-10.md` §1.25.

**R9, the re-read on c3x (`STATE.md` Blocked 27 (a), confirmed by the user 2026-10-08: "Start
Phase 10's c3x re-read"; rule fixed before any c3x reading).** **Question:** which re-read labels
change when c3x, not c3, is the definition's column? **The column:** c3x = c3 minus the groups
R8x's rows do not mark `c3x` (each EOD condition's J ≥ max(`J0`, that condition's `Jc`)); centred,
size 2 only (R8x built no arm). Written by `p10_label_source extend` into a new source
(`data/p10/reread_r9_2026-10-08/labels`): R0's step files with a `c3x` labelling per layer, every
other column copied unchanged; `c3x_learned` / `c3x_unlearned` at step 143000 as c3's split.
**The ladder:** c3x appended after c3 (the one choice it adds: the chance-aware move filter).
**Primary per step: c3x**, read on c2 where fewer than half its records are readable (as before);
its Δ rows take **c2's step 0**, as c3's; the floor is c3x's step-0 count. Each row's reading
rule, floors and "holds" (primary against c0; A0 against c1) are unchanged. **Beside, the answer
this re-read exists for:** per row, the cells where c3x's label differs from c3's, at steps both
read on their own column. The arms stay c3's and are quoted against c3 as before. **Rows, in
order, each its own PR:** R1 (§1.7, §1.9, §1.10); R2 (F1, F12's gap, §1.5); R2m; R3 (A0 under T4);
R6 → R6w → R6f; R7. Not re-read: R5c (`gpt2-large`, 5c's own partition; c3x is not defined
there), R8 and R8x (c3x's source). **First checks (refuse):** `extend` refuses unless every c3
group at every (step, passage, layer) has an R8x row, every row R8x marks `c3x` is a c3 group, and
each such group's member count is R8x's `size`; the new source's c0–c3 labels equal R0's exactly;
each row's c0–c3 records read on the new source reproduce that row's stored R-records (the
summaries the ladder reads, exactly), so a changed label is the column and not the run; the
first c3x record is opened and checked populated before the rest. **It does not:** re-examine
"moves", `J0` or `Jc`; add a row; change a floor; test anything. Results: `status-10.md` §1.26.

**R9 / R6 (fixed 2026-10-09, before any c3x matcher output).** R6 labels nothing (an
instrument), so "the cells where c3x's label differs" reads here as **§1.20's headline clauses
on c3x beside c3**. **Re-run, not filtered:** the matcher's kinds depend on every group at both
steps (a c3 split whose second piece c3x drops is a c3x stable link), so c3's stored links
cannot be filtered into c3x's. `p10_r6_matcher.py` with c3x added as a column (flips against
c2a, as c3; lineage on its own links and by id on c2a), everything else R6's. **A clause
changes when:** (i) survival: c3x's stable share differs from c3's by > 0.05 (R1–R3's floor,
placed) at any boundary from 4000 → 8000 to 32000 → 54000, or at 54000 → 143000; (ii) "changed
members": at any of those boundaries the stable links' median Jaccard < 0.5 or their identical
share > 0.5; (iii) "few new": at any boundary from 4000 → 8000, new is ≥ ¼ of c3x's births +
deaths (placed; c3's largest is 49 of 694); (iv) "none past 256": any c3x lineage at 143000
starts before 256; "89 % after 2000" moves by > 0.05; (v) "longer than independent": c3x's
share reaching ≤ 2000 is not above its independence baseline. Every boundary's c3x numbers sit
beside c3's whatever the clauses say. **First checks (refuse):** R6's c2a gate; c0, c1, c1c,
c2a, c2 and c3 pooled on the R9 source equal R6's `r6.json` exactly (`--reproduce`, with the
summary of the source it names); c3x holds groups at 143000 before any chain runs. Results:
`status-10.md` §1.30.

Each of R0–R3 is its own PR, and none is re-read before the one ahead of it is merged. Unlocked
by R0, not re-reads, for after R3 or for the user to bring forward: the cross-checkpoint
matcher (MONIC's transitions on unit 1's Jaccard and its fixed bar, `lit-10.md` §16 rows 1–2),
the context-shuffle test (§16 row 5; run as R7), Stage 2, F13.

## Parked

- **An n-gram control for §1.10 at step 512** (`lit-10.md` §16 row 3): the class effect falls
  in Pythia's unigram stage. Cost: a unigram-matched prompt per v1 passage. Could change:
  whether §1.10's "the network computes it" is n-gram statistics. Not taken: the re-read
  changes no construction.
- **The designed prompts are not used.** §1.9's "class" is orthographic (`token_class`), so the
  category list is no known answer for it; unit 4's semantic reading stays in 1d.

## Worth challenging

- **c0 re-runs the old readers on 7 prompts of Stage 0's dirs** instead of quoting the
  published numbers, so c0 can differ from what `status-10.md` says. Rejected: quoting the
  published number, which mixes sweeps and the eighth prompt into the comparison.
- **F1, F12 and A0 are taken into scope** against `design-1d.md`'s "Scope" paragraph. The
  argument is that their statistics read a binary split; the counter is that "noise" now
  means "kept but in no group that moves", a different population from HDBSCAN's −1.
- **All 16 checkpoints**, for ~1 h, against a thinned set.
- **"Holds" is judged on c3 against c0** (c1 for A0). An alternative is c3 against c2,
  which asks only whether the move filter matters; the ladder reports both.
- **Step-0 baselines on c2** (after review): the baseline then carries no move filter while
  the trained reading does. The alternative, c3 at step 0, is mostly chance passes.
- **The readability bar (≥ 10 members, ≥ 10 rest; half the records) is placed**, as are
  A0's 0.9 and F12's ±0.05.
