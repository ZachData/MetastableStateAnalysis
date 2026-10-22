<!-- trackC_writeup/status-C.md -->
# Track C — STATUS

<!-- phase-card -->
## Card

- **Question:** When, along `pythia-410m`'s training, does the tokens' interaction energy stop being non-decreasing in depth, as the identity-weight particle picture predicts, is that step the run's or the recipe's, and where does it fall against the model's known developmental order (n-gram stages, previous-token and induction heads, the OV maps turning repulsive)?
- **Inputs:** none run under C's name; C2 reads Track A's and Track S's records (`trackC_writeup/design-C2.md` "What the record supports")
- **Results:**
  - C2 opened 2026-10-22 and its design froze the same day: seven lines the record supports, nine fences, an outline, a render step so every number comes from a record; Blocked 91–94 decided as recommended (left to Claude by the user): the dated, reproduced break as the headline, the anisotropy control and the LR-dose arm as new runs, Markdown rendered from records, S2 in the main text and S1 an appendix — `trackC_writeup/design-C2.md` "Decided 2026-10-22"
  - Unit 1, the reading: monotone E_β is a theorem only for constant, symmetric, value-tied scores on the sphere with no MLP, so the break violates no theorem (F5 worded from the texts); published anisotropy along Pythia starts at step 10 000, not in the break's window (F9 narrowed); no public suite has a different warmup with checkpoints inside it, so 92 (ii) is not run (6 units + close-out); §2 and §8 drafted — `trackC_writeup/status-C.md` "C2 unit 1"
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@6b6559eb1a, S@a1fb89438b
- **Feeds:** none
- **Open threads:**
  - Whether the break is the LR schedule's or a training event's, and whether it is anisotropy's (F1, F9); units 3 and 3b read each, no public suite can vary the warmup (unit 1) — `trackC_writeup/design-C2.md` "What C2 may not go past"
- **After Phase 10:**
  - The anisotropy control: V on token-centred clouds at the 18 steps on seed 0 *(forward pass, cached checkpoints)*
- **Reviewed:** 2026-10-22 · body `1c7fc5fcd4`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: frozen, Blocked 91–94 decided as recommended (left to Claude)** | `design-C2.md` "Decided 2026-10-22" |
| C2 1. reading (writing only) | **done 2026-10-22: F5 worded, F9 narrowed, 92 (ii) not feasible, §2 and §8 drafted** | `lit-C2-unit1.md`, `draft-C2.md`; "C2 unit 1" below |
| C2 2. `numbers.py`, figures, §3–§5 | next | — |
| C2 3, 3b. anisotropy control; LR-dose arm (70m, 160m) | after 2 | design 92 (i), (iii) |
| C2 4, 5, close-out | after 3b | — |

## C2 unit 1, the reading (2026-10-22)

Writing only; nothing ran. Read: the gradient-flow results in full (2312.10794 §1–§5,
2605.04279), 2603.13314, the [S] rows §2 and §8 cite (each now [H] or [A]), a delta scan since
2026-10-18, the Hub's checkpoint branches for every candidate suite, and Pythia's released
configs. Record: `lit-C2-unit1.md`. What it changed:

| item | result | where it lands |
|---|---|---|
| F5 | monotone E_β is proved only for constant weights with symmetric scores tied to the values (one head or many), on the sphere, no MLP; Pythia at every step and S2's edits meet none, so the break departs from a picture's prediction, not a theorem | `design-C2.md` F5; `draft-C2.md` §2.1 |
| the energy's sign | non-decreasing in every text; `lit-S2.md`'s "decreases" corrected, routed to S (no S2 reading moves) | `status-S.md` "Corrections received" |
| F9 | Bernas et al. read Pythia's activations from step 10 000 on and chiefly study the gradients; nothing reviewed measures activation anisotropy in 128–512; an unreviewed 70m project reports it half-built by 512, which raises unit 3's stakes | `design-C2.md` F9; `draft-C2.md` §8 |
| F1 | 2509.23024 ties its own first geometric phase to the warmup | `design-C2.md` F1; `draft-C2.md` §2.3 |
| 92 (ii) | **not feasible**: the Mistral GPT-2 runs (4000-step warmup, dense early checkpoints in their docs) now hold only `main` on the Hub; OLMo-2 1B has steps 0, 300, 10 000; the rest start later or coarser | unit 3c not run; budget 6 units + close-out |
| 92 (iii) | the configs confirm the arm: 70m, 160m, 410m share the warmup (1430 steps), batch and schedule; peak LR 1e-3, 6e-4, 3e-4 | unit 3b as designed |
| 2603.13314 | weights on OLMo-2 7B; keys rise most, values flat; S1's `ov_read` is a different statistic | appendix A (unit 4) |
| novelty | "nobody reads this energy across training" not refuted as of 2026-10-22 | `draft-C2.md` §8 |

Parked (`lit-C2-unit1.md`): the Mistral checkpoints off the Hub (carried to C2's close-out);
token-drift's per-step table (carried to unit 3b).

## C2 unit 0, the design (2026-10-22)

Writing only; nothing ran. The design names what the record supports (L1–L7, each pointing to
the one place its numbers live), what the draft may not go past (F1–F9), the outline, and the
exit (a complete draft, every number rendered from a record, a fresh-context referee read
answered). F9, the anisotropy objection, is new to C2's row: `lit-C2.md` §1 named it and no
ROADMAP row carried it. Inputs read: the one-page closes of A3, A5, S1 and S2 and `lit-C2.md`.

## Corrections received

none
