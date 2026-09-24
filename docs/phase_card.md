# The phase card

A card sits at the top of each `status-N.md`, right under its title: one screen
saying what the phase asked, what ran on which inputs, what it found, what
superseded it and what it left open. `tools/render_phases.py` lines the cards
up into `docs/PHASES.md`; lint rule `phase-card` keeps them current. Why cards
and not per-phase STATE files: `docs/PHASE_REVIEW.md`.

## Fields (fixed, in this order)

| field | content | the lint requires |
|---|---|---|
| Question | the question the phase asked, one sentence, in the particle/OT vocabulary | filled |
| Inputs | model(s), checkpoints, battery hash, run dirs; "none" if never run | filled |
| Results | one sub-item per result, each with its pointer | ≥ 1 item; each has a pointer that resolves |
| Superseded / wrong | what later work corrected, one sub-item each, with the pointer; or "none" | each item has a pointer that resolves |
| Registry | prediction ids and their state, or "none, because …" | filled |
| Depends on | phases whose results this card reads, comma-separated `<phase>@<hash>`, written by `--stamp`; or "none" | each id is a phase; each hash matches that phase's `## Corrections received` |
| Feeds | phases that read this one's results, comma-separated; or "none" | agrees with their Depends on, once they have cards |
| Open threads | questions the phase itself raised and did not answer | filled |
| After Phase 10 | candidate experiments, each tagged *(free)* or *(forward pass …)* | every item has its cost |
| Reviewed | `YYYY-MM-DD · body <hash>`, written by `--stamp` | hash current (below) |

**Pointers.** A backticked repo path (`p1c_frames/status-1c.md`), optionally
followed by §N or by a "Heading" in quotes; or a bare §N, which means
`PROJECT.md`. Each must resolve: the file exists, the heading exists. Paths
under `data/` are accepted unchecked, since no checkout carries them.
**Numbers stay where they already live** ("write once"); a card states the
finding in words and points.

**Staleness.** The body hash is a hash of the status file with the card cut
out. If anything in the file outside the card changes, the card is **stale**
and the gate fails. Each Depends on entry is `<phase>@<hash>`, and that hash
covers **only** the upstream phase's `## Corrections received` section
(user, 2026-09-24, `docs/PHASE_REVIEW.md` "Decisions"). So an ordinary edit
to a phase stales only that phase's own card, and phases can be edited
independently. A correction routed to a phase (below) also stales every
card that reads it, one hop. Stamping a card never touches a Corrections
section, so two phases that read each other do not re-stamp each other.
To clear it: read what changed (`git log -p -- <status file>`), fix the card if
the change touches it, then run `python3 tools/render_phases.py --stamp <phase>`
and `python3 tools/render_phases.py`. The stamp records that someone looked.
It does not check what they concluded.

**Corrections received: how a later correction reaches the card.** Most
corrections to a phase are written by a *later* phase, in its own files or
in `PROJECT.md` §3.x, and never touch the corrected phase's status file.
(Phase 1's first card: 4 of its 7 corrections came from there.) So each
status file carries, outside the card, a section

    ## Corrections received

    - 2026-09-20 · what was corrected, in words · `pointer`

and whoever writes a correction to an earlier phase adds a line there
(`CLAUDE.md` Stop step 2). The line changes the body hash, so the card goes
stale, and it changes the Corrections hash, so every card that reads this
phase goes stale too. Settled by the user 2026-09-23 (`docs/PHASE_REVIEW.md` "Decisions").
It relies on people following the rule: nothing checks that a correction was
routed.

**What staleness does not see.** A change to phase A outside its
`## Corrections received`: the phases reading A stay current, even if the
change matters to them. A correction two hops away: if A is corrected and B
reads A, B's card goes stale, but C, which reads only B, does not until
someone routes a line to B. And a correction nobody routed stays invisible, so each card session still greps for the phase
outside its directory (`p1b_`, "Phase 1b") before stamping.

Phases without a card are listed in `docs/PHASES.md` as "no card yet" and are
not checked.

## Skeleton

Copy everything between the two markers, markers included, to just under the
status file's title. Every `TODO` must go before the lint passes.

<!-- phase-card -->
## Card

- **Question:** TODO one sentence
- **Inputs:** TODO models, checkpoints, battery hash, run dirs
- **Results:**
  - TODO a result, with its pointer
- **Superseded / wrong:** none
- **Registry:** TODO ids and state, or "none, because …"
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - TODO
- **After Phase 10:**
  - TODO an experiment (free)
- **Reviewed:** TODO run --stamp
<!-- /phase-card -->
