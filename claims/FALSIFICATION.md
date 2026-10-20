# FALSIFICATION.md — the adjudication ledger

**Generated** by `tools/render_falsification.py` from `claims/registry.json`
and `claims/adjudications/`. Do not edit by hand; `--check` fails CI when this
file disagrees with the records it summarises.

Threshold: a claim is supported when its accumulated evidence reaches 
**E ≥ 1/α = 20** (α = 0.05, κ = 0.5).

The decision is never "null accepted". Failing to accumulate evidence against
a null is not evidence for it, and an e-process has no way to express one — which
is the Popperian asymmetry kept visible in the artifact rather than only in the
prose around it.

## Status

1 prediction(s) adjudicated across 1 claim(s).

## Per-claim evidence

| claim | adjudicated | log E | E | decision |
|---|---|---|---|---|
| `H-BRIDGE` | 0 | 0 | 1 | not adjudicated |
| `H-EMERGE` | 0 | 0 | 1 | not adjudicated |
| `H-OPERATOR` | 0 | 0 | 1 | not adjudicated |
| `H-RESIST` | 1 | 7.37 | 1587 | reject_null (read with its notes, `H-RESIST` below) |
| `H-TRANSFER` | 0 | 0 | 1 | not adjudicated |

### H-BRIDGE

18 registered · 4 adjudicable now · 9 dormant

*No adjudications.*

<details><summary>Registered but not adjudicable (14)</summary>

| prediction | why |
|---|---|
| `P-I2` | needs-null — null not yet constructed |
| `P-I4` | needs-null — null not yet constructed |
| `P-I5` | needs-null — null not yet constructed |
| `P-I7` | needs-null — null not yet constructed |
| `P-SA1` | needs-null — null not yet constructed |
| `P5b-A1` | dormant — instrument archived |
| `P5b-A2` | dormant — instrument archived |
| `P5b-B1` | dormant — instrument archived |
| `P5b-B2` | dormant — instrument archived |
| `P5b-B3` | dormant — instrument archived |
| `P5b-C1` | dormant — instrument archived |
| `P5b-C3` | dormant — instrument archived |
| `P5b-D1` | dormant — instrument archived |
| `P5b-D2` | dormant — instrument archived |

</details>

### H-EMERGE

1 registered · 1 adjudicable now · 0 dormant

*No adjudications.*

### H-OPERATOR

14 registered · 4 adjudicable now · 10 dormant

*No adjudications.*

<details><summary>Registered but not adjudicable (10)</summary>

| prediction | why |
|---|---|
| `P6-A2` | dormant — instrument archived |
| `P6-C1` | dormant — instrument archived |
| `P6-D5` | dormant — instrument archived |
| `P6-DD1` | dormant — instrument archived |
| `P6-DD2` | dormant — instrument archived |
| `P6-I1` | dormant — instrument archived |
| `P6-I2` | dormant — instrument archived |
| `P6-R1` | dormant — instrument archived |
| `P6-R3` | dormant — instrument archived |
| `P6-R5` | dormant — instrument archived |

</details>

### H-RESIST

6 registered · 2 adjudicable now · 0 dormant

| # | prediction | p | e | running E | decision after |
|---|---|---|---|---|---|
| 1 | `P-A5` | 9.923e-08 | 1587 | 1587 | reject_null |

- **`P-A5`, notes on record:** First ledger record (STATE Blocked 78 (a), the user's, 2026-10-20). A reproducibility check, not a test of mechanism (design-A5.md D9 (a)): E is against a uniform-location null nobody holds (D6 (b)), so a large E is cheap; relevance is the registry's floor (0.6); all 9 seeds share Pythia's LR warmup (to about step 1430), so K = 9 says the step is the recipe's, not whether it is training's or the schedule's. H-RESIST's reject_null after this record rests on this entry alone; every quote of that decision carries this line.

The threshold is crossed (E ≥ 20).

<details><summary>Registered but not adjudicable (4)</summary>

| prediction | why |
|---|---|
| `CLAIM-A` | needs-null — null not yet constructed |
| `P-H1` | measurement — no valid null exists |
| `P-gamma1` | needs-null — null not yet constructed |
| `P-gamma2` | needs-null — null not yet constructed |

</details>

### H-TRANSFER

1 registered · 1 adjudicable now · 0 dormant

*No adjudications.*

## Ledger integrity

Every claim's E was recomputed from the committed records and agrees with
what those records state. Each e-value was recalibrated from its p-value
rather than trusted as stored, so a hand-edited record would surface here
rather than propagate.

