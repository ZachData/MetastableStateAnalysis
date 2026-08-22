# archive/ — frozen phases

Phases 3, 4, 5, 5b, 5c and 6. All of this ran against GPT-2 / ALBERT / BERT,
before the project moved to Pythia checkpoints and before the "particles
first" reframe. None of it is current work. Archived 2026-08-22.

## The three rules

**1. Not maintained, not imported, not collected.**
No live module imports anything under `archive/` — this was verified before the
move, and it is the property that made the move mechanical rather than a
refactor. `pytest.ini` sets `testpaths = tests` and `norecursedirs = archive`,
so nothing here is collected by default.

The tests moved with their code, but they will not run as-is: they depend on
shared fixtures and the heavy-dependency stub installer that stayed in
`tests/conftest.py`. `archive/tests/conftest.py` carries only the pieces that
existed solely for archived tests (the `p5b_manifold` package bootstrap and the
Phase 5b manifold fixtures). Reconnecting the rest is part of a deliberate
reintroduction, not something the archive offers ready-made. Saying so plainly
is better than shipping a conftest that looks runnable and isn't.

**2. Nothing is salvaged by copying.**
If a future phase needs a capability that lives here, it is **rebuilt against
the particle schema** (`core/particles.py`), not lifted. Every module in this
directory predates the reframe and keys its own bespoke structures — cluster
chains, per-phase result dicts, hand-rolled `labels >= 0` masks. Copying one
forward would reintroduce exactly the producer/consumer mismatches that
`core/artifacts.py` exists to kill.

Reading them for design is not only allowed but expected. `p3_crosscoder/
steering.py` already implements a steering intervention with a merge-event
readout and a recorded null; new steering work should be written having read
it, so it does not silently re-derive an evaluation design that has already
been tried.

**3. Archiving the code does not retract the findings.**
Each phase keeps its `status-N.md` and `design-N.md` verbatim. These contain
real results and real blocker diagnoses that stay citable:

- **Phase 3** — the sparse-crosscoder null: decoder→V alignment 0.484 / 0.501,
  indistinguishable from random, in both models.
- **Phase 4** — Track 3's `v_alignment_recovered` for ALBERT, the finding that
  *sparsity* was the confound rather than absence of geometric structure.
- **Phase 5** — the tuned-lens skip-to-output note (2026-07-19), which says why
  "just train the lens" would replace an obviously-broken result with a subtly
  wrong one.
- **Phase 5c** — the attention-flip result: trained models route attention
  *toward* unclustered tokens at 1.6-2x, sign-flipped against random weights.
  This is cited in `PREDICTIONS.md` claim (a) and is not archived reasoning.
- **Phase 6** — the LDA-alignment inversion (0.887 on the imaginary subspace vs
  0.067 on real repulsive, 0/49 layers in the predicted direction), still
  carrying two live explanations, neither ruled out.

## Reintroduction

Phases 3 and 4 carry an explicit trigger; see the `FROZEN.md` in each. The
others have no trigger — they are archived because the project moved, not
because they failed, and reviving one is a scoping decision, not a threshold
being crossed.
