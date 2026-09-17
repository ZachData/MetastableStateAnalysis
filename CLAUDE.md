<!-- CLAUDE.md -->
# Working agreements

`PROJECT.md` is the living state of the repository and **the file to read
first** — where the work stands, what is registered and may not be re-decided,
and how to reproduce any number in it. `INDEX.md` maps phases to directories.

This file is not a second copy of that. It holds only the working agreements:
the habits that kept getting forgotten, and are therefore written down.

## Keep the handoff current as you go, not at the end

**Update `PROJECT.md` when a unit of work closes** — a result landed, a defect
found, a decision taken — **not in a sweep before the session ends.** Sessions
end abruptly (context runs out, a background job gets killed by the memory
watchdog, the machine sleeps), and that file is what the next session starts
from. Forgetting it has already cost twice: `cf700f9` had to correct a git line
the commit before it made stale, and on 2026-09-12 §3.15 was still ending on an
open hedge that the very next commit had already closed and recorded elsewhere.

Bump the **"Last updated" header line** in the same edit. Phase detail belongs
in that phase's `status-N.md`; `PROJECT.md` carries only what a fresh session
needs in order to start working.

## Open PRs at natural boundaries

CodeRabbit reviews PRs, and a PR carrying an era of work is not reviewable by a
human in any useful way — the 2026-09-12 merge was **51 commits and ~9,500
lines**, which is the thing not to repeat. **Open a PR when a coherent piece of
work closes**: an invariant read, a defect fixed, a runner parametrised. The
point is a review a person can actually get through, so the bound is reviewer
attention, not commit count.

## Literature scans: two triggers, both rare

Not every derivation — most derivations here are instrument plumbing (a
transpose convention, a z-score leave-one-out fix) where novelty is irrelevant.
Not every PR either; PR cadence is a review-load question. Scan at the two
moments where *"this is already known"* still changes a decision:

1. **When a phase or subphase opens, before its `design-N.md` freezes the
   constructions.** The high-value one, because it changes what gets built.
   Precedent: `5dae0aa`, "let the literature scan tell us what of it is actually
   new".
2. **Before an entry lands in `claims/registry.json`.** Registration freezes the
   wording and the statistic, so it is the last moment a literature fact can
   still change the definition, the null, or the name.

An **end-of-phase** scan buys citations and positioning rather than design, so
run one when a phase produced something publishable — not as a ritual.

## Checking the math

`sympy` is a base dependency, and symbolic checks belong in `tools/math_checks/`.
A closed-form derivation (an identity, a gradient, a sign/transpose convention)
can be checked mechanically and should be; an empirical number measured off real
checkpoints cannot, and needs its producer re-run instead — `PROJECT.md` §7
lists how. Note in each check what it does *not* prove: an identity instantiated
at `n = 4` is evidence, not a general-`n` proof.
