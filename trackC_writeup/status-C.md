<!-- trackC_writeup/status-C.md -->
# Track C — STATUS

<!-- phase-card -->
## Card

- **Question:** When, along `pythia-410m`'s training, does the tokens' interaction energy stop being non-increasing in depth, as the identity-weight particle picture predicts, is that step the run's or the recipe's, and where does it fall against the model's known developmental order (n-gram stages, previous-token and induction heads, the OV maps turning repulsive)?
- **Inputs:** none run under C's name; C2 reads Track A's and Track S's records (`trackC_writeup/design-C2.md` "What the record supports")
- **Results:**
  - C2 opened 2026-10-22 with a proposed design: seven lines the record supports, nine fences the draft may not go past, an outline, a render step so every number comes from a record, 5 units + close-out; four decisions are the user's (headline, new runs, form, S1 and S2's place) — `trackC_writeup/design-C2.md` "Decisions (each yours; recommendation first)"
- **Superseded / wrong:** none
- **Registry:** none, because Track C writes up rows scored elsewhere (`P-A5`, Track A's)
- **Depends on:** A@6b6559eb1a, S@e7ac5c3113
- **Feeds:** none
- **Open threads:**
  - Whether the break is the LR schedule's or a training event's, and whether it is anisotropy's (F1, F9) — `trackC_writeup/design-C2.md` "What C2 may not go past"
- **After Phase 10:**
  - The anisotropy control: V on token-centred clouds at the 18 steps on seed 0 *(forward pass, cached checkpoints)*
- **Reviewed:** 2026-10-22 · body `d76d2df9a2`
<!-- /phase-card -->

## Units

| unit | state | where |
|---|---|---|
| C2 0. design (writing only) | **done 2026-10-22: proposed, Blocked 91–94 the user's** | `design-C2.md` |

## C2 unit 0, the design (2026-10-22)

Writing only; nothing ran. The design names what the record supports (L1–L7, each pointing to
the one place its numbers live), what the draft may not go past (F1–F9), the outline, and the
exit (a complete draft, every number rendered from a record, a fresh-context referee read
answered). F9, the anisotropy objection, is new to C2's row: `lit-C2.md` §1 named it and no
ROADMAP row carried it. Inputs read: the one-page closes of A3, A5, S1 and S2 and `lit-C2.md`.

## Corrections received

none
