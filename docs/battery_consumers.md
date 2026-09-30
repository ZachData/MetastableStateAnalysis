# Who reads the live prompt battery

`LESSONS.md` lesson 3: anything whose result is recorded names the exact
input set it ran on and refuses on a mismatch. This is the audit #115 left
open: every live file that reads `core.config.PROMPTS`.

**Input:** `git grep -l -E '\bPROMPTS\b' -- '*.py' ':!tests/' ':!archive/'`
at `13225de` (2026-09-29): 31 files. The ones that only name `PROMPTS` in a
docstring or comment (`core/particles.py`, `core/interactions.py`,
`core/battery_structure.py`, `p7_motifs/motif_stats.py`,
`p1d_cluster_ensemble/p1d_io.py`, `long_prompts.py`, `analysis_p1.py`,
`analysis_p2.py`) are left out of the table. A second grep, for
`PROMPT_BATTERY_HASH`, adds `p2b_imaginary/p2b_io.py`, which stamps the
live hash without reading `PROMPTS` (first version of this file missed it;
`/challenge-pr` on #116).

**Two risks, not one.** *Growth*: a loop over the live battery silently
widens when prompts are added (P-I5's defect). *Text drift*: a key whose
text changed after a run was extracted. Code that rebuilds positions from
the live text and reads them against a stored run then indexes different
tokens. No v1 text has changed yet (#115 checked `bff93d7` against now).

| File | Feeds | Prompt set | Names its input | Refuses on drift | Status |
|---|---|---|---|---|---|
| `p7_motifs/p_i5_*.py`, `tools/calibrate_p_i5_joint_null.py` | `P-I5` (registered) | pinned 8, by key tuple | `battery_hash` (texts) | yes | ✅ #115 |
| `p7_motifs/run_7.py` | the interaction tables `CLAIM-B`, `P-AB1`, `P-I3`, `P-ST1` read (registered) | `--prompt KEY=DIR` | was the sorted key list; now a text hash of the prompts used, plus per prompt: run dir, sha256 of its `tokens.txt`, Phase 1 manifest id and battery hash | now yes: live text vs the run's `tokens.txt` (before the admissibility gate, which now judges the run's prefix), `tokens.txt` vs the activations' width, and the Phase 1 manifest's checkpoint vs `--model`'s (when both are recorded) | ✅ #116 |
| `tools/score_claim_c.py` | `CLAIM-C` (registered) | live battery minus `repeated_tokens`; 20 today, **grows with the battery** | per-arm key list + sha256 of every artifact read | n/a (reads stored artifacts, rebuilds no positions) | ⚠️ growth: user's call (below) |
| `tools/run/behavioural.py` | `P-I1` B arm (registered) | keys with all 19 steps on disk; refuses held-out | `scored_prompts` (keys) | yes (`tokens.txt`) | ✅ |
| `tools/run/relay_null.py` | `P-I1` relay null (registered) | keys in `formation_series.json` | upstream record only | yes (`tokens.txt`) | ✅ |
| `tools/run/stage0_chunk.py` | Phase 10 Stage 0 | live battery, pinned | `BATTERY_HASH_V2` | yes (hash) | ✅ |
| `p1_mstate_tracking/run_1.py`, `p1b_hemisphere/run_1b.py` | Phase 1 / 1b artifacts (`CLAIM-C`'s inputs) | `--prompts`, default all live keys | whole live battery's hash in each manifest | n/a (producer: extracts from the text it hashes) | ✅, except `run_1 --length-sweep`: it adds `wiki_<N>` keys to `PROMPTS` after the hash is fixed at import (`run_1.py:775`, `core/prompts.py:139`), so those runs' hash does not cover their text ⚠️ |
| `p2b_imaginary/p2b_io.py` `write_run_manifest` | Phase 2b manifests | none read | stamps **today's** live battery hash, not its input run's; `"unavailable"` on any import error | n/a | ⚠️ tier 1, not fixed: should carry its Phase 2 input's hash, as `run_2d.py` does |
| `p2_eigenspectra/run_2.py` | Phase 2 (tier 1) | `--prompts`, default all live keys | no battery hash | `_run_decompose` re-extracts from the live text; a missing key returns `None` | ⚠️ tier 1, not fixed |
| `tools/run/dissipation_sublayer.py` | `PROJECT.md` §3.8.3 (tier 1) | fixed 7 + `repeated_tokens` | keys | token **count** only, vs Phase 1 activations | ⚠️ tier 1, not fixed |
| `tools/run/induction_composition_whitening.py` | whitening diagnostic (tier 1) | `natural` arm: every live text, **grows with the battery** | no | n/a | ⚠️ tier 1, not fixed |
| `tools/token_counts.py`, `p1_mstate_tracking/reporting_p1.py`, `core/run_discovery.py` | nothing recorded (printout, report text, an optional argument) | — | — | — | n/a |
| `data/analysis/dissipation_v2_violation_restricted.py`, `p7d_redundancy/fv_score.py` | own list (read from a record, or built in the file) | — | — | — | n/a |

**Open for the user.** `CLAIM-C`'s scorer reads whatever metastability keys
the live battery holds. The registry fixes no prompt count. Today a
battery growth past 20 makes the gate refuse, but only because no
homogeneity correction is tabulated past 12 (`PROJECT.md` §3.46). Once
the n = 20 row is calibrated (`docs/PHASE_SYNTHESIS.md` §3.1), adding a
prompt would change the test again. Pin it the way P-I5 is pinned, or
accept growth with a recalibration per count.

**Parked** (discoveries, not followed):
- `run_7.py` takes `KEY=DIR` for any key and has no holdout guard;
  `tests/test_holdout.py` scans `tools/run/` only. Cost: one
  `refuse_held_out` call plus a test. Could change: whether a Phase 7
  table built on 410m can read the twelve.
- `p1_io._load_tokens` and `core/artifacts.py`'s `tokens` spec say
  `tokens.txt` is tab-separated; the writer uses two spaces, so the
  loader returns whole lines (`"  0  tok"`). Cost: switch it to
  `core.battery_structure.phase1_tokens`. Could change: anything that
  reads `run["tokens"]` from `p1_io.load_run`.
