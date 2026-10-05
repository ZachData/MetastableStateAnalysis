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
| **A0** attention flip vs the causal mask | attention received, binary labels, mask baseline (`p10_attention_baseline`) | **yes, with T4** (sink columns dropped, rows renormalised) | members − rest. T4 changes the statistic itself, so **c3's labels are compared with c1's, not c0's**. Labels: mask share **≥ 0.9 or not** (0.9 placed, below c0's ~0.94); the step interval where the learned residual first appears (c0: 2000–4000); whether it persists to 143000 |

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

Each of R0–R3 is its own PR, and none is re-read before the one ahead of it is merged. Unlocked
by R0, not re-reads, for after R3 or for the user to bring forward: the cross-checkpoint
matcher (MONIC's transitions on unit 1's Jaccard and its fixed bar, `lit-10.md` §16 rows 1–2),
the context-shuffle test (§16 row 5), Stage 2, F13.

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
