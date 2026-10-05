<!-- p10_cluster_function/design-10.md -->
# Phase 10 — the re-read design (on 1d's working definition)

**Status: proposed 2026-10-05, not frozen.** It freezes when the user accepts it
(`STATE.md` Blocked 23); nothing below has run. Tier 1, exploratory, unregistered:
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

**The check that joins the two passes.** "Moves" comes from unit 1's own P = 0 pass; the rows
read Stage 0's stored activations. Per (step, prompt), the P = 0 cloud must match the stored
`activations.npz` (unit 1 found max abs 3e-8 at step 0); a mismatch above 1e-5 refuses that
(step, prompt).

## The ladder: what is read beside every row

One column per choice, so a changed row says which choice changed it (`lit-10.md` §16 row 7).

| col | partition | isolates |
|---|---|---|
| **c0** | the stored labels, all tokens, the row's reader **re-run unchanged on the 7 prompts, Stage 0 dirs** | the old reading on the re-read's inputs (the published number pooled 8 prompts or another sweep) |
| **c1** | shipped `HDBSCAN(min_cluster_size=2)`, float64, on the **kept** tokens (T1–T3) | the token rules (`admit.layer_groups` returns it from the same distances) |
| **c2** | level-set groups, stable, non-bulk, **without** the move filter | the group route |
| **c3** | **the definition** (c2 ∧ moves) | the filter. **The re-read's primary column** |
| floor | c3 at step 0 | the definition at init (62 step-0 records on v1, against 810 trained) |
| arms | c3 at centred size 4, raw size 2 | quoted only where a sign or a reading label differs from c3's |
| learned | c3 split by unit 2's learned bar, at step 143000 | "learned beside, not required" (Blocked 22 (a)); at other steps only if the bar reads from stored re-init records without new passes (the build checks; otherwise 143000 only) |

**A row "holds"** when c3 gives the reading c0 gave, by the row's own reading rule (below).
When it does not, the first column where the reading changes is named. A record with no
group, or fewer than 2 kept non-members, refuses and is counted; the count is reported per
step, since at step 0 most records will refuse.

## Row by row

| row (`status-10.md`) | what its reader takes | re-read? | how, and its reading rule |
|---|---|---|---|
| **§1.7**, unique-token part (rank does nothing; class does, weakly) | clustered rate of unique tokens by rank bin and class (`p10_token_composition`) | **yes**, unique tokens only | member share among kept unique tokens by rank and class; §1.7's own rule. Its copy-count headline is out (T3 keeps one copy). Not in `design-1d.md` "Scope", which listed §1.6–§1.8 together |
| **§1.9** co-membership | focal = clustered unique token at position > 0; co-members; 5 lifts against a random draw (`p10_comembership`) | **yes** | draw from kept tokens only; layers 12, 24 and the mean over L1–24 (L0 out); rule as written there: Δ vs step 0 above / as / below at ±0.05. `copy_share` now means "the co-member's string recurs later" (the copies are dropped); `adjacent` counts a ±1 neighbour only if kept. Where step 0 has no focal token at a layer, Δ is unreadable and the trained lift is reported against the random draw alone |
| **§1.10** lexical carry | as §1.9, plus each token's layer-0 vector (`p10_lexical_carry`) | **yes** | layer 0 enters as a covariate (the frozen frame), not as a cloud, so the definition's L0 gap does not bite; rule as written there |
| **F1** clustered − noise step (the 32–512 window) | binary labels at each layer boundary, permutation null (`tools/run/transport.py`) | **yes** (correcting `design-1d.md` "Scope", which said F1 needs a full partition: its labelled statistic reads clustered vs noise only) | members vs kept non-members, permuted among kept tokens, 2 000 draws. Holds iff negative at median p ≤ 0.05 at every step 32–512 and not at 0–16. The identity-coupling result is partition-free and is not re-read |
| **F12** clustered − noise on corrected `Z` | binary labels, `Z` per token (`p10_partition_function`) | **the corrected-`Z` column only** | as F1; holds iff its sign matches c0's at each step. Raw `Z` and the sink results are position, out (T1 drops the sink) |
| **§1.5** F1 + F12, "parked, not pinned" at 32–64 | the two above | follows them | read off F1 and F12's re-read; no reader of its own |
| **A0** attention flip vs the causal mask | attention received, binary labels, mask baseline (`p10_attention_baseline`) | **yes, with T4** (sink columns dropped, rows renormalised) | members vs kept non-members. T4 changes the statistic itself, so **c3 is compared with c1, not c0**. Holds iff the mask share stays ≥ 0.9 and the learned residual appears in the same step range (2000–4000) and persists |

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
P = 1000 activations on `HDD_1TB` stay its own). **Gate before the batch:** run one
checkpoint (512) end to end, open its record, and check it is populated (groups per layer,
classes, the P = 0 match) before launching the other 15.

Rejected: a thinned axis (e.g. 0, 32, 512, 4000, 143000), which saves ~45 min and loses the
window edges (16 → 32, 512 → 1000) that F1's reading rule is stated on.

## Order

| # | unit | needs | why here |
|---|---|---|---|
| R0 | **build**: unit 1's runner at the 16 checkpoints; a labels source with the `kept` domain and columns c1–c3, which the readers take by flag and refuse without | ~1 h of forward passes | every row reads it |
| R1 | §1.7 (unique), §1.9, §1.10 | R0 | the thread's own question (Stage 1: what is in a cluster); their focal unit, unique tokens off position 0, is already what the definition keeps, so they change least in meaning |
| R2 | F1 + F12 (corrected `Z`), then §1.5 | R0 | the phase's one dynamic finding, and its density confound was argued from step 0; the definition's step-0 floor is a stronger control than the old partition had |
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
