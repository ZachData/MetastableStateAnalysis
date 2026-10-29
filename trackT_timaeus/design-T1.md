<!-- trackT_timaeus/design-T1.md -->
# Track T — design: T1, the Timaeus inventory

Opened 2026-10-29 by `STATE.md` Blocked 117 (a), the user's: T1 and T2 first. Row:
`docs/ROADMAP.md` §7b T1. The search before it: `docs/lit-T.md` (abstracts and listings only).
Free: no download beyond configs, cards and file listings; nothing is run. Tier 1.

## What T1 answers (fixed before any reading)

| # | question | what T2 / T3 / T4 takes from it |
|---|---|---|
| Q1 | Does any released Timaeus model have a published LLC curve, and is any of them deeper than 2 layers? | a yes on a deep one replaces T3–T4 by the cheapest overlay (ROADMAP §7b) |
| Q2 | Which repo holds 2402.02364's language model, and what are its LLC settings (ε, γ / localisation, nβ, chains, draws, batch)? | T3's positive control |
| Q3 | Dodecahedron's provenance: tokens per step, total tokens, the LR schedule as trained, which repo T2 should read (`-32m` / `-67m`, `-v2`, `-archived`, `-pilot-dense`) and its checkpoint steps | T2's mapping rule (step, cumulative LR, tokens) |
| Q4 | What `timaeus/gpt-2-small` (and `-log-spacing`) is: architecture, data, schedule, steps, any published LLC on it | a second deep candidate for Q1 |
| Q5 | Does the negative stand: no published LLC along training for any Pythia model (primary text of 2402.03698 and 2606.22389 read, not abstracts)? | T4's novelty |

## What counts as an answer

- **Confirmed**: read in a primary source (the paper's text, a `config.json`, a checkpoint's
  file listing, the training code), with the source named.
- **Leaning**: a card, a blog or a search summary only.
- **Unknown**: neither found within the budget.

Each question gets one of the three, with its source, in `status-T.md` "T1".

## Exit and budget

**One unit.** T1 closes when Q1–Q5 each carry one of the three grades, or at the budget with what
it has, the rest marked unknown. Nothing is downloaded beyond metadata, and nothing is computed,
so there is no calibration gate here; T2's design carries the first one.
