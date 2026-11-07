<!-- prompt_library/status-L.md -->
# The prompt library — STATUS

<!-- phase-card -->
## Card

- **Question:** How many real, held-out prompts does a token-cloud reading (ρ_mid on Tracks A and T, read today on 7 hand-picked v1 prompts) need, drawn from which source by which rule fixed before any model reads them, and pinned how, so that later readings name their exact input?
- **Inputs:** none run. U0 read T5 U2's records (`data/trackT/t5u2/2026-10-30/`, the 7 v1 prompts, code `3d425b32db90`) through `prompt_library/sizing_u0.py`, and the Hugging Face API listing of `monology/pile-uncopyrighted` at `3be90335b66f`
- **Results:**
  - Opened 2026-11-07 by the user's order in the roadmap's decisions, item 14 (T5′ paused for it): the 7 prompts' spread of ρ_mid is small at most steps (SD 0.02–0.04) and 0.07–0.23 in the transitions, so a transition's level is known to ±0.03–0.09 at 7 prompts; a design proposed with a sealed confirmation split, 3 units + close-out; D1–D5 are Blocked 140 — `prompt_library/status-L.md` "U0, the design (2026-11-07)"
- **Superseded / wrong:** none
- **Registry:** none, because the library is an input set; D4's sealed split is held for a later registration, which is the user's
- **Depends on:** none
- **Feeds:** none
- **Open threads:**
  - Blocked 140: D1–D5 (`prompt_library/design-L.md` "Decisions")
  - whether random Pile documents spread more than the 7 v1 prompts (U1 measures it)
- **After Phase 10:**
  - near-duplicate check of the library against the Pile's train split *(free: CPU, an n-gram pass over one train shard)*
- **Reviewed:** 2026-11-07 · body `7325903b70`
<!-- /phase-card -->

## Corrections received

None.

## U0, the design (2026-11-07)

Branch `claude/prompt-library-u0`; writing and one reader of existing records. Design
`prompt_library/design-L.md`; scan `docs/lit-L.md`.

| item | value | how to re-run |
|---|---|---|
| across-prompt SD of per-prompt ρ_mid on the 7 v1 prompts, median over steps | 0.023 (70m) – 0.041 (`-67m`) | `METS_DATA=<main>/data python prompt_library/sizing_u0.py` |
| its maximum, and where | 0.069–0.228; Pythia at step 256, Dodecahedron at 88, `gpt-2-small` at 1100 | same |
| SE of a k-prompt mean at the maximum | k = 7: 0.026–0.086; 256: 0.004–0.014; 1024: 0.002–0.007 | same |
| per-prompt median against the recorded ρ_mid | within 0.018 at every record | same |
| held-out text on hand | Pile val and test (`pile-uncopyrighted`, 338 MB each, LFS sha256s in `docs/lit-L.md` §1); FineWeb `sample-10BT` on `HDD_1TB` (no validation split; held out only from T5's arms) | `docs/lit-L.md` §1 |

**Caveats.** The SDs are the 7 hand-picked prompts', not real draws'. The per-prompt median is a
proxy for the recorded statistic. T5 U2 scored three trajectories; Pythia's were beside.

**Parked.** None yet.
