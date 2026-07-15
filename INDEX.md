# STATUS/DESIGN Split — Index

Output of transition-plan execution-order item 1 (v2): every phase README split into a short
`status-N.md` (verdict table, blockers, last-verified date) and a long-form `design-N.md`
(rationale — why the phase is built the way it is, not just what it found). `N` is the phase
identifier used everywhere else in the plan (1, 1b, 2, 2b, 2c, 3, 4, 5, 5b, 5c, 6), so each
file is identifiable on its own, outside its folder. Directory names still match the real
repo layout, so these drop in directly.

Read `status-N.md` when you need the current state of a phase. Read `design-N.md` when
you're about to touch the code and need the reasoning that isn't visible from the code
itself. `PREDICTIONS.md` (repo root, alongside this index) is new in v2 — the transition
project's own falsification record, separate from any single phase's.

## What changed from v1 to v2 (this pass)

The plan v2 document changed three things that actually touch content already written, not
just renumbering:

1. **Naming reconciliation, done here.** `p2b_imaginary/`'s own README and result files call
   it "Phase 2i"; the directory name is canonical per the plan. All prose references across
   every phase's docs now say "Phase 2b." On-disk artifact filenames (`phase2i_results.json`,
   `phase2i_summary.txt`) are untouched — renaming those is separate, unscoped work.
2. **Parallel-residual decomposition reframed as an upgrade, not a skip.** Phase 2's
   `status-2.md`/`design-2.md` and Phase 1's `design-1.md` now explain why Pythia's parallel
   residual (Δx = attn_out + ffn_out, exact, no ordering confound) is a *cleaner*
   decomposition than GPT-2's sequential one ever gave, not just an architecture Phase 2's
   existing module can't reach. The GPT-2-only files stay frozen either way; what's new is a
   planned `core/` module that reopens the question natively on Pythia.
3. **SAE/crosscoder freeze hardened to frozen-for-deletion.** Phase 3 and Phase 4's
   `low_rank_ae.py` status docs now state the precise reintroduction trigger (activation
   caches at ≥4 checkpoints **and** a specific particle-dynamics question needing a
   dictionary) rather than the vaguer v1 "revisit once checkpoint data exists."

One more change worth flagging on its own, since it's a reframe rather than a correction:

**Phase 5c's object of study is now the project's object of study.** v2's opening section
("Framing: particles first") promotes this phase's unclustered-population work — every
particle and how it evolves, clustering as an annotation rather than the unit of analysis —
to be the organizing principle for the whole transition, including the per-particle-record
schema every other phase's aggregations will be built on. `status-5c.md` and `design-5c.md`
now carry this explicitly; nothing about Phase 5c's own blockers changed.

Mechanical-only fixes made throughout (no content change, just correctness): every reference
to a v1 execution-order item number updated to v2's renumbering (e.g., the falsification-
table retrofit moved from item 10 to item 12; the freeze/bug-fix step moved from item 3 to
item 4), and Phase 5's two artifact-loading blockers (OV values, Group D) now flagged as
instances of the "artifact-contract" bug class v2 names explicitly, fixed once at the
infrastructure level rather than patched per-phase.

## Phase status at a glance

| Files | Phase | State | Flag |
|---|---|---|---|
| `p1_mstate_tracking/{status-1,design-1}.md` | 1 | Complete | — |
| `p1b_hemisphere/{status-1b,design-1b}.md` | 1b (1h) | Complete | — |
| `p2_eigenspectra/{status-2,design-2}.md` | 2 | Complete | — |
| `p2b_imaginary/{status-2b,design-2b}.md` | 2b *(reconciled from "2i" — see status-2b.md)* | Complete | — |
| `p2c_churchland/{status-2c,design-2c}.md` | 2c | **Complete** | README header says "Not started" — **stale, contradicted by results**. |
| `p3_crosscoder/{status-3,design-3}.md` | 3 | Complete, **frozen-for-deletion** (v2) | Null result. Reintroduction trigger stated precisely — see status-3.md. |
| `p4_mstate_features/{status-4,design-4}.md` | 4 | Complete | `low_rank_ae.py` frozen-for-deletion alongside Phase 3, despite being this phase's one positive result. |
| `p5_single_mstate_analysis/{status-5,design-5}.md` | 5 | Complete, partially blocked | 6 code-level blockers; 2 reclassified as "artifact-contract class" (v2). |
| `p5b_manifold_steering/{status-5b,design-5b}.md` | 5b | Not started (execution) | Code and tests built, never run. |
| `p5c_unclustered/{status-5c,design-5c}.md` | 5c | Not started (causal); preliminary correlational only | **v2 reframe: this phase's object of study is now the project's** — read status-5c.md first. |
| `p6_subspace/{status-6,design-6}.md` | 6 | **Partial run, one model** | README header says "Not started" — **stale**. |

## Two things worth knowing before reading further

**Two README headers were wrong** — carried over from v1, still true. `readme-phase2c.md`
and `README_phase6.md` both say "Not started" while their own results data show partial or
complete runs. The corrected `status-2c.md` / `status-6.md` are the source of truth going
forward, not the old headers.

**Dates are inconsistent across phases.** Recovered from run-directory names / report
timestamps where present: Phase 1 — 2026-04-23; Phase 2 — 2026-04-28; Phase 2b, Phase 3 —
2026-04-29; Phase 2c — 2026-05-02; Phase 4 — 2026-05-04; Phase 5 — not recorded (after
2026-05-04); Phase 5b — never run; Phase 5c — not recorded; Phase 6 — recorded only as
"2026-04-xx" in the source, itself a gap. Worth adopting a single timestamp convention going
forward (v2's run-manifest infrastructure, item 2, effectively forces this for every future
run — every manifest carries a real timestamp — so this problem should stop recurring once
that lands).

## Not done as part of this pass

- The actual file split/move into each phase's real directory, or writing `PREDICTIONS.md`
  into the actual repo root — this is drafted content, ready to drop in. Nothing has been
  written back to `/mnt/project`, which is read-only.
- `FROZEN.md` for Phase 3 / `low_rank_ae.py` (v2 item 4, separate from this item) — the
  precise wording and reintroduction trigger are drafted in each phase's `status`/`design`
  doc, but the actual `FROZEN.md` files aren't written yet.
- Correcting the stale "Not started" headers in the actual `readme-phase2c.md` /
  `README_phase6.md` files, or deleting those files in favor of the split.
- Renaming the on-disk `phase2i_results.json` / `phase2i_summary.txt` artifacts to match the
  Phase 2b naming reconciliation — left alone deliberately, see status-2b.md.
