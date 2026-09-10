"""
p8_scale_ladder — the same induction measurements across the Pythia size
ladder, so that single-model observations can become population claims.

Everything 7d and 7e found is `n = 1`. "`L11H14` is anti-ordered" is an
anecdote about one head in one model; "induction sets contain a full-rank,
anti-ordered member at every scale" is a result. This phase is the machinery
that separates the two, and `p7d_redundancy/design-7d.md` reserved the slot for
it: *a shared signature over independent circuits is a population claim no
single circuit can make.*

THE RUNG POLICY, which is the one thing here that must not be violated:
**explore low, validate high.** pythia-70m and pythia-410m carry exploration;
**pythia-1b and pythia-1.4b are reserved** and are not to be measured on any
induction quantity until a prediction naming them is registered. See
design-8.md; the policy is the phase's whole epistemic value and it is one
careless sweep away from being lost.

NOTHING HERE TRANSFERS BY HEAD NAME. `L5H2`, `L7H8` and `L11H14` are 410m
coordinates. The ladder compares **structural invariants** — the shape of the
membership tail, the formation window, ordering versus window, the alignment
trajectory, the rank profile, the interaction structure — never a head index and
never an absolute threshold. `d_head` also varies across the ladder (64 at 70m
and 410m, larger above), so every rank is reported as a **fraction of the
model's own budget**.

Upstream sibling: `Lora_inductionhead` (GitHub, ZachData) supplies the pythia-70m
dense-onset bracket — 82 checkpoints at stride 4 through an induction onset. It
is an independent repository and stays one; this phase imports its artifacts
one-way and sends instrument fixes back. `PROJECT.md` §3.9 / §3.9-A are the
record, including the caveat that its retrain is **reachability, not
development**.

Read design-8.md for the six invariants and the sequencing, status-8.md for what
has actually run. Nothing here is registered yet.
"""
