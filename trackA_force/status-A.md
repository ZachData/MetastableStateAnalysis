<!-- trackA_force/status-A.md -->
# Track A — STATUS

No card yet (`docs/phase_card.md`); one is added with A1b's reading.

## Units

| unit | state | where |
|---|---|---|
| 1. design, scan, `sympy` check, t* table | **done 2026-10-14**, awaiting the user's freeze (Blocked 45) | `design-A.md`, `lit-A.md`, `tools/math_checks/a1_collapse_time.py`, `tstar_table.py` |
| 2. A1a, the β producer over 18 steps and the T_eff instrument, with its gate | not started | `design-A.md` "A1a" |
| 3. A1b, the reading (H-WARP and 1e's question beside) | not started; runs only after A1a's gate passes | `design-A.md` "A1b" |

Budget: 3 units (`design-A.md` "Exit condition and budget").

## Before any output (2026-10-14)

No model, no data. **Input:** eq. (6.9) only. **Producer:** `python -m trackA_force.tstar_table`;
the table is in `design-A.md` "t* before any output" (one copy). What it fixed: the registered
t* ≈ 4.2 is the orthogonal-start time; from a start at mean cosine 0.3 it is ≈ 1.5 at every β,
and from g0 = 0 it is flat in β only to ≈ 5.6. With the causal mask (6.9) lacks
(`python -m trackA_force.sa_sim`, after `/challenge-pr` on #201), t* rises steeply with β (14.8
at 5.6, n = 467), so A1's comparator is the causal simulation from the measured cloud.
`sympy`: `tools/math_checks/a1_collapse_time.py`, 4 of 4 pass; tests `tests/test_trackA_tstar.py`.

## Parked

(none yet)

## Corrections received

(none)
