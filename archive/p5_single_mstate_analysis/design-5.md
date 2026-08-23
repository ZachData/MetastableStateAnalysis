# Phase 5 — DESIGN

## Core question

Every previous phase worked in aggregate — 35 model×prompt verdicts, thousands of features
ranked by F-statistic, bulk V-alignment scores. Phase 5 inverts the approach: take a single
HDBSCAN cluster trajectory and reconstruct, end to end, the mechanism that creates,
maintains, and dissolves it, cross-referenced against every framework the project has built
(Phases 1, 2, 2i, 3, 4). The deliverable is an interpretable narrative of one piece of the
model's computation, not another aggregate statistic.

## Why a six-criterion selection score, not the "best" cluster

Cluster selection (`select_cluster.py`) scores candidates on lifespan (≥6 layers), merge
participation, semantic content (Phase 1's P1-4 tag, excluding induction artifacts), prompt
context (`sullivan_ballou`/`paper_excerpt` preferred, `repeated_tokens` avoided), size (≥4
tokens), and sibling availability. This is deliberately not "pick the cluster with the
longest lifespan" — a single-axis choice would risk selecting a degenerate or artifact-driven
trajectory. Six independent gates, each individually motivated by a known project pitfall
(the induction-head confound from Phase 1, the repeated-tokens degeneracy from Phase 1),
make the selection auditable rather than arbitrary. All six models achieve near-perfect
scores (9.000 for 4/6), and each model's runner-up trajectory shares the same prompt and
scores within 0.3 points — evidence the selection is stable, not cherry-picked.

## Why seven investigation groups, and why they're ordered A→G

Each group targets a different layer of explanation, roughly in the order a mechanistic
story would need them:

- **A (structural profile)** — is this actually a coherent object? (compactness, silhouette,
  CKA). Prerequisite for everything else meaning anything.
- **B (paper-theoretical alignment)** — does the object behave the way Geshkovski's framework
  predicts (attractive/repulsive centroid split, local S/A, merge geometry)?
- **C1/C2 (head and FFN contributions)** — which components *cause* the cohesion in A? Split
  into two groups (attention heads vs. FFN) because Phase 2's OV-centric framework and the
  FFN pathway are mechanistically distinct enough to need separate treatment — this split is
  exactly what surfaces the ALBERT/GPT-2 FFN-role inversion.
- **D (feature signatures)** — connects back to Phase 4's crosscoder/LRAE work, asking
  whether this specific cluster's identity shows up in feature space.
- **E (tuned-lens decoding)** — what does the model predict while this cluster exists?
  Adds a semantic reading on top of B/C's geometric one.
- **F (causal interventions)** — the only test that moves from correlational profile to
  causal necessity. Deliberately last, since it's the most expensive and most directly
  falsifying group.
- **G (sibling and random control)** — run throughout the pipeline as a validity check: does
  the primary cluster actually stand out from a same-size random baseline and from its
  geometric sibling? The clean three-tier IP/silhouette ordering (primary > sibling > random)
  across all six models is what licenses treating A–F's findings as about a real object.

## Findings and what they mean for the framework

- **Locally rotational, universal.** Phase 2b's global rotation-neutral result reproduces at
  the level of individual trajectories — no cluster shows a locally non-rotational profile.
  This closes a possible gap (global average could in principle have masked
  locally-rotational exceptions; it doesn't).
- **~50/50 attractive/repulsive centroid split, universal.** This is flagged as a mild
  tension with Theorem 6.3's prediction that cluster tokens sit primarily in the attractive
  subspace during stable phases — not resolved here, explicitly deferred to Phase 6.
- **FFN role is architecture-dependent, and BERT is the outlier even relative to ALBERT.**
  ALBERT (FFN-cohesive, attention-disruptive) is explained by weight-sharing: the same
  attention weights applied iteratively produce fragmented attention, while FFN — also
  shared but applied to a changing residual state — stabilizes. BERT is attention-cohesive,
  FFN-disruptive (−13.19) despite architectural similarity to ALBERT, attributed
  speculatively to its bidirectional masked-LM pretraining producing a different
  computational role balance. This is flagged as unresolved, not asserted as settled.
- **Causal robustness scales with GPT-2 model size.** Read as a hypothesis (larger models
  distribute cluster identity across more components, so no single intervention dissolves
  it), explicitly conditional on the identical-`mean_frac_together` issue not being a bug —
  the design doesn't treat this result as clean until that's resolved.

## Known-issue-to-fix mapping (why these are tracked here rather than silently patched)

Each of the six known issues names a specific code path to check (e.g., `merge_verdict`
n/a → trace the `merge_events` argument from `run_5.py` through to `merge_event_geometry()`)
rather than just noting "value looks wrong" — this is the same discipline the transition
plan asks of every phase (named regression test before fixing), applied here as a diagnosis
even before a test is written.

## Deferred work, explicitly separated from Phase 6's scope

A semantic-decode enrichment for the Phase 1 blog post (decoding every tracked cluster
member's next-token prediction, not just the centroid) is logged here as deferred, explicitly
*not* Phase 6's work — an earlier draft conflated it with Phase 6's real/imaginary
decomposition question, and the current README corrects that conflation rather than
silently dropping the record of it. Blocked on fixing Known Issue 4 (probabilities rounding
to 0.000) first, since nothing built on the tuned-lens output is trustworthy until then.

## Code structure

`select_cluster.py` (selection), `cluster_profile.py` (A), `v_alignment.py` (B),
`head_contributions.py` (C1), `ffn_contributions.py` (C2), `feature_signature.py` (D),
`tuned_lens_cluster.py` (E), `causal_tests.py` (F), `sibling_contrast.py` (G), `report.py`
(assembles `cluster_report.txt`), `run_5.py` (CLI).
