# PREDICTIONS.md — Pythia Transition, Project-Level Falsification Record

This file is distinct from any phase's own falsification table (those live in each phase's
`status-N.md`). This one covers the transition project's own claims — the things the move
from a multi-architecture GPT-2/BERT/ALBERT study to a Pythia checkpoint study is supposed to
show. Written and committed **before the replication gate (execution-order item 6) runs**, so
the timestamp on this file precedes any result it's checked against. Don't edit the
predictions after seeing gate results — if a prediction needs revising, add a dated
addendum below it rather than changing the original text.

## Why these three claims, specifically

The whole point of moving to a checkpoint suite is to ask whether what Blog 1 found on
GPT-2-large — trained weights resisting the collapse the architecture drives, with a
large, informative unclustered population — is (a) something training does or something
architecture already gives you for free, (b) tied to the same circuit-formation events the
literature already anchors checkpoint schedules to, and (c) actually the same phenomenon on
a different architecture rather than a GPT-2-large idiosyncrasy the whole project has been
building on without knowing it. Every one of these has a clean failure mode that would change
what the rest of the plan is even for — which is why they're gated (claim (c) specifically
stops the sweep if it fails) rather than just noted and moved past.

## The three claims

| Claim | Prediction | Failure reading |
|---|---|---|
| (a) Collapse-resistance is learned, not initial | Steps 0 and 8 look "random-like": monotone energy, rank collapse, high stationary Fiedler | Resistance is partly architectural/init-borne; the trained-vs-random contrast (the load-bearing comparison across Phases 1, 2c, and 5c) needs restating for what it actually isolates |
| (b) Resistance emerges at circuit-formation events | The energy-monotonicity break and Fiedler drop co-locate with steps ~512–2,000 (the Pythia-410M pilot, execution-order item 8, tests this directly rather than assuming it) | Clustering dynamics and circuit formation are independent — itself a real result, and it re-anchors the 1.4B checkpoint schedule rather than invalidating the sweep |
| (c) Phenomenology transfers across architecture | `pythia-1.4b-random` (norm-matched to the final trained checkpoint) reproduces `gpt2-large-random` phenomenology; trained checkpoint 143,000 reproduces trained `gpt2-large` phenomenology | The Blog 1 contrast is architecture-dependent, not a general property of trained transformers. **Stop and re-baseline before any checkpoint sweep** — this is the one claim with a hard stop attached, since every downstream phase's Pythia rerun assumes this transfer holds |

## How each claim gets adjudicated, and by what

- **(a)** — the replication gate (item 6): Phase 1 run at Pythia step 0 and step 8, checked
  against the same pass criteria Blog 1 established for gpt2-large-random (monotone energy,
  rank collapse, high stationary Fiedler). This is a cheap-tier check, no expensive-tier
  compute needed.
- **(b)** — the Pythia-410M pilot (item 8): a dense 20–30 checkpoint sweep through the cheap
  tier, explicitly built to test co-location of the energy/Fiedler/effective-rank transitions
  against the circuit-formation-event literature's checkpoint anchors, not to assume the
  anchors are right and just fill in data around them. A failure here is informative on its
  own terms (see table) and directly changes the 1.4B anchor schedule rather than being
  treated as a pilot that "didn't work."
- **(c)** — also the replication gate (item 6), using both the true step-0 init and the
  norm-matched `pythia-1.4b-random` as the two separate objects the plan's two-baseline
  policy requires (see Phase 5c's `design-5c.md` for why these can't be collapsed into one
  "random" condition). **If this fails, no checkpoint-sweep work (items 9–11) proceeds past
  the gate.**

## Status

Not yet adjudicated — the replication gate (item 6) has not run. This file exists to make
sure that when it does, the prediction was on record first.
