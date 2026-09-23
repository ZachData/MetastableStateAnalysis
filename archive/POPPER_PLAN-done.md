<!-- archive/POPPER_PLAN-done.md -->
# POPPER_PLAN.md — the chunks marked DONE, archived 2026-09-22

Moved verbatim out of `POPPER_PLAN.md` at `64a4087`: A0–A7, B1–B5, B7, C1, C2.
B6 (first one done), B8 and C3–C6 stayed, being open. Section ids are unchanged,
so a citation `POPPER_PLAN.md` §B3 now means this file §B3 (`archive/MOVED.md`).

---

### A0. Baseline: what the suite does today · **DONE** · [C]

Recorded before changing anything, so later greenness means something. See
`docs/CI_BASELINE.md`.

### A1. Dependency manifest · **DONE** · [C] · S

`pyproject.toml` plus `requirements/{base,test,heavy}.txt`. Three tiers, matching the three
things the code actually needs:

- **base** — `numpy`, `scipy`. Everything in `core/` that is torch-optional by design.
- **test** — base + `pytest`.
- **heavy** — test + `torch`, `transformers`, `matplotlib`, `scikit-learn`, `hdbscan`.

Acceptance: `pip install -r requirements/test.txt && pytest -m pure` succeeds in a container
with no torch.

### A2. Unblock the pure tier · **DONE** · [C] · S

Remove `tests/conftest.py`'s module-scope `import torch` (line 20) and the second stray
`import torch` at line 273; route every use through the stub that the file already installs,
and make the real-torch fixtures skip cleanly when torch is genuinely absent.

Acceptance: `pytest --collect-only` succeeds with numpy+scipy+pytest and nothing else.

**This is the single highest-leverage fix in workstream A.** Until it lands, no CI tier
exists that can run in under ten minutes, and a fast tier is the difference between CI that
gates merges and CI that gets ignored.

### A3. Marker taxonomy · **DONE** · [C] · S

Register markers in `pyproject.toml` (not a separate `pytest.ini` — one config file):

- `pure` — numpy/scipy/pytest only. The gating tier.
- `deps` — needs torch/sklearn/matplotlib importable, but no model download and no artifacts.
- `smoke` — real torch/transformers, tiny HF models, network once. Already partly built
  (`SMOKE_REAL_DEPS=1`); this only registers and documents it.
- `heavy` — needs real artifacts on disk. **Never** run in CI; marked so it can be
  deselected deterministically rather than by filename convention.

Four tiers rather than the three originally planned. `deps` was added after measuring: the
planned `pure`/`smoke`/`heavy` split has no home for the ~34 modules that need real tensors
but no model and no artifacts, and without it they fall into whichever half the partition
happens to leave them in.

**Tier assignment is measured, not assigned.** A module is `pure` only if its *whole* test
set passes with torch, transformers, scikit-learn and matplotlib all made unimportable — 59
of 95 modules qualify, 1532 tests in ~10 seconds. Collecting-without-error is not sufficient
and was the first thing tried: 72 modules collect clean but 13 of those fail once run.

**`-m` alone is not enough**, which is the non-obvious part. pytest imports every module
before deselecting, so one `deps` module raises at collection and takes the run down before
any deselection happens — 19 modules do exactly this. `tests/conftest.py` therefore carries a
`pytest_ignore_collect` hook that reads each module's declared marker as *text* (no import,
which is the whole point) and skips `deps` modules when their dependencies are genuinely
absent. Keyed on real importability rather than an env var, so a runner that has torch runs
the deps tier without being told.

Acceptance: `pytest --markers` lists all four; `pytest -m pure` is green with no heavy deps
installed (enforced in CI by an explicit "assert torch is absent" step, so the tier cannot
silently stop testing what it claims to).

### A4. Workflows · **DONE** · [C] · M

- `.github/workflows/ci.yml` — on push and PR. Job `lint` (tier 0, no deps, seconds) then
  job `pure` (tier 1). Both required.
- `.github/workflows/smoke.yml` — nightly + `workflow_dispatch`. Tier 2. Not required for
  merge; a failure opens an issue rather than blocking.

Note for whoever runs this: the CPU torch wheel comes from `download.pytorch.org`, which is
reachable from GitHub Actions but **not** from this sandbox's egress proxy. The smoke job's
install step is written against the CPU index deliberately; do not "fix" it to plain PyPI,
which pulls the 4.9 GB CUDA wheel.

### A5. Repo-hygiene lint · **DONE (first four rules)** · [C] · M

`tools/lint_repo.py`, run in tier 0. Encodes the project's own standing rules
(`UPDATE_PLAN.md` §6) as machine checks. Rules implemented this pass:

1. **No orphan modules.** A `.py` under a package directory that no import path can reach
   (catches `core/.py`).
2. **Every test file carries exactly one tier marker.** Prevents the "unmarked straggler"
   drift A3's partition depends on.
3. **No hand-synced constant literals.** A numeric constant defined in `core/` and repeated
   verbatim elsewhere with a "keep in sync" comment is an error — this is exactly the defect
   `checkpoint_scalars.py`'s `ast`-parsing fix closed once, and the rule stops it recurring.
4. **Doc-claim staleness.** A `status-*.md` or README header asserting "Not started" while a
   results artifact for that phase is referenced elsewhere is an error. `INDEX.md` already
   lists two live instances (`readme-phase2c.md`, `README_phase6.md`).

Queued rules (A5b, [C], S each): **5.** every threshold literal is annotated *placed* or
*calibrated* (standing rule 6); **6.** every data-dependent fallback records its branch
(standing rule 2); **7.** every anchor test has a non-symmetric arm (standing rule 5) — this
one is a heuristic and should warn, not fail.

### A6. `scripts/check.sh` · **DONE** · [C] · S

One script that runs exactly what tier 0 + tier 1 run, so local and CI cannot diverge. CI
calls the script; it does not reimplement it.

### A7. Delete `core/.py` · **DONE** · [C] · XS

Verified unreachable and contradicted by the live `core/models.py`. Removed. A5 rule 1 keeps
it gone.

---

### B1. `core/evalues.py` — the kernel · **DONE** · [C] · M

Pure numpy. No project imports, no torch. Contents:

- `calibrate(p, kappa=0.5)` → e. Domain-checked; refuses `p ∉ [0,1]` and `κ ∉ (0,1)`.
- `EProcess` — accumulates `(prediction_id, p, e)` in order, exposes `E`, `log_E`, and
  `decision(alpha)`.
- `sufficient_evidence(E, alpha)` → `E ≥ 1/α`.
- `required_p_for_rejection(alpha, kappa, n_prior)` — the diagnostic that says, before running
  anything, how small a p-value the next experiment must return to cross the threshold. Cheap
  and it stops people running underpowered experiments.

Acceptance: unit tests covering (i) the calibrator's null property `E[e] ≤ 1` under
`p ~ U(0,1)` by Monte Carlo, (ii) `p = 1 → e = κ < 1` (a non-falsification accumulates
evidence *against*, which is the property that makes the process honest), (iii) the
super-martingale property under a sequence of null p-values, (iv) Markov's-inequality
Type-I control at nominal α by simulation, (v) log-space accumulation agreeing with the
direct product to floating tolerance over ≥ 500 terms.

### B2. `claims/` — the machine-readable registry · **DONE** · [D + C] · L

Two files plus a directory.

**`claims/CLAIMS.md`** — the main hypotheses, which currently exist only implicitly. Every
prediction must name exactly one. Proposed initial set, drawn from what the project already
argues:

- **H-RESIST** — trained weights actively resist the architecture's collapse dynamics.
- **H-TRANSFER** — that phenomenology is a property of trained transformers, not a
  GPT-2-large idiosyncrasy. *(This is `PREDICTIONS.md` claim (c), the one with a hard stop
  attached.)*
- **H-EMERGE** — resistance emerges at circuit-formation events.
- **H-BUDGET** — the network spends a bounded dimensionality budget on particles that must
  stay individuated (Phase 5c's effective-rank plateau).
- **H-OPERATOR** — collapse/anti-collapse is attributable to stated operator conditions
  (V eigenstructure, QK symmetry) rather than being an aggregate curiosity.
- **H-BRIDGE** — the natural-language-interpretability constructs are particle-dynamical
  objects, and the particle account makes *correct differential* predictions (workstream C).

**`claims/registry.json`** — one record per prediction. JSON, not YAML: the project has no
YAML dependency and `core/particles.py` documents the deliberate no-new-dependency norm.
Schema:

```
id, claim, statement, h0, h1, falsifier, instrument, cost,
evaluable: "e-value" | "measurement" | "needs-null",
null_construction, relevance, kappa,
registered_commit, registered_date, superseded_by, notes
```

**`claims/adjudications/<id>.json`** — one record per adjudicated prediction: the p-value, the
artifact hash it was computed from, the run manifest, the resulting e, and the cumulative E
for its claim at that point.

Acceptance: `tools/check_registry.py` validates schema, uniqueness, that every `claim` names a
row of `CLAIMS.md`, that every prediction ID appearing anywhere in `*.py`/`*.md` has a registry
entry, and that `relevance ≥ r₀`.

### B3. Pre-registration gate — **where CI and Popper fuse** · **DONE** · [C] · M

POPPER's Assumption 2 (sequential validity) is the assumption that actually carries the Type-I
guarantee, and it says: *the choice of sub-hypothesis and test function must not depend on the
data used to test it.* `PREDICTIONS.md` asserts this by convention ("Written and committed
before the replication gate runs, so the timestamp on this file precedes any result it's
checked against").

**Git can check that.** `tools/check_preregistration.py`, run in tier 0 — built as
`check_preregistration` *inside* `tools/check_registry.py` rather than as a file of its own,
so that one tool reads the registry once; the name here is the plan's, not a path:

- For every `claims/adjudications/<id>.json`, resolve the commit that introduced the registry
  entry for `<id>` and the commit that introduced the adjudication. Fail if the registration
  does not strictly precede.
- Fail if a registry entry's `statement`, `h0`, `h1`, `falsifier`, or `null_construction` was
  **modified** after its first adjudication — amendments go in `notes` as dated addenda, which
  is the mechanism the P-T1 amendment already used correctly by hand.
- Warn when an adjudication's artifact hash matches one already consumed by a *different*
  prediction registered later. Reusing an artifact is fine; registering a new prediction
  *after* seeing that artifact and then testing it on the same artifact is the conditional-
  validity violation, and it is invisible to every other check.

Acceptance: a deliberately back-dated fixture (adjudication committed before registration)
fails the gate; the existing `PREDICTIONS.md` history passes it once backfilled.

**Caveat to state honestly in the doc this produces:** predictions registered *before* this
machinery existed get their `registered_commit` backfilled from `PREDICTIONS.md`'s own history.
That is a claim about the past based on git, which is good evidence but not the same as having
run the gate. Backfilled entries carry `registered_provenance: "backfilled"` and are reported
separately in every summary.

### B4. `core/adjudication.py` — emission · **DONE** · [C] · M

The thin layer between a science module and the registry. Takes `(prediction_id, p_value,
artifact_hashes, run_manifest)`, looks up the registry, refuses if `evaluable != "e-value"`,
computes `e`, appends to that claim's e-process, and writes the adjudication record.

Two refusals, both instances of standing rule 4 ("refuse rather than degrade"):

- **No registry entry → refuse.** An unregistered prediction cannot be adjudicated, because
  its Assumption-2 status is unknown.
- **`evaluable != "e-value"` → refuse.** See B5. Emitting an e-value from an invalid null is
  strictly worse than emitting nothing: it is unfalsifiable from the artifact alone, and it
  silently voids the Type-I guarantee for *every other prediction on that claim*, because
  the product is only as valid as its weakest factor.

### B5. The evaluability audit — **the honest part** · **DONE** · [D + R] · M

Not every registered prediction can carry an e-value, and pretending otherwise would be the
exact pseudo-rigor this workstream exists to prevent. Each of the ~26 IDs gets classified:

- **`e-value`** — a valid null and a real p-value exist or can be constructed. Example: **P-S1**
  is already adjudicated on the ratio to a *matched random baseline* (`UPDATE_PLAN.md` §5.7,
  §5.8) — that is a permutation null, so a p-value follows directly. **P6-I1** already runs a
  Mann-Whitney U. **P-M1** correlates a per-boundary violation indicator against a regime
  score; a permutation test over boundaries gives a valid p. **P-T1** compares a trimodality
  rate among row-2 candidates against the required control arm — a two-proportion test.
- **`measurement`** — no valid null exists and the honest output is a number with an interval.
  Example: **P-H1**. Wendel's theorem gives probability 1 for `d > n`, which every prompt
  satisfies; the project's own §5.7 already says the boolean is "nearly vacuous" and the
  reportable object is the *margin*. Forcing an e-value here would manufacture evidence from a
  theorem.
- **`needs-null`** — the prediction is testable but the null has to be built first, and that
  construction is its own chunk. Example: **P-γ2** (`T_eff ≪ t*`) is a point estimate against a
  constant; a bootstrap over the eight prompts gives a p-value, with the small-n caveat stated
  in the record rather than buried.

Deliverable: `claims/EVALUABILITY.md` with one row per ID, the classification, and the reason.
Every `needs-null` row spawns a chunk in the queue.

### B7. `FALSIFICATION.md` generation + CI recomputation · **DONE** · [C] · M

`tools/render_falsification.py` builds, per claim, the ordered table of adjudicated
predictions with p, e, running E, and the decision at α — replacing the hand-maintained
verdict tables in each `status-N.md` with a generated artifact.

And the check that makes the committed record self-verifying: **CI recomputes E from the
committed adjudication records and fails if any reported decision disagrees.** Deterministic,
no artifacts, no heavy deps, runs in tier 0. It catches arithmetic drift and, more usefully,
catches a verdict word that was updated by hand without its evidence.

### C1. `docs/PARTICLE_ONTOLOGY.md` · **DONE** · [D] · L

Written before any code, per the norm `core/DESIGN_dual_reading.md` already follows. One
section per construct: standard definition, particle-paradigm definition, what already exists
in this repo, and the differential prediction. Initial six:

| Construct | Particle-paradigm reading | Already here |
|---|---|---|
| **Induction head** | An inter-particle coupling with a matching kernel: a term in the velocity field that transports a particle toward the successor of its earlier occurrence, i.e. a *non-local* attraction not explained by current position | `p6_subspace/induction_ov.py` (P6-I1 already asks whether induction OV writes into the imaginary/rotational subspace) |
| **Activation steering** | An exogenous impulse added to a particle's velocity. Its effect is predicted by its decomposition in the V-eigenbasis: components on the attractive subspace accelerate collapse, on the repulsive subspace disperse | `p5b_manifold_steering/`, `core/intervention.py` |
| **SAE / crosscoder feature** | A claimed *coordinate of the particle configuration* — a direction some subpopulation concentrates on. Falsifier: a feature that is not a stable direction of the population is a decomposition artifact, not a mechanism | `p3_crosscoder/`, `p4_mstate_features/low_rank_ae.py` — both **frozen-for-deletion**, so this entry is definition + deferred experiment, not code |
| **Logit / tuned lens** | The readout map from particle position to the vocabulary simplex; "the lens is linear" is the claim that this map does not itself curve the geometry | `p2_eigenspectra/lens_band.py`, `p5_.../tuned_lens_cluster.py` |
| **Ablation / activation patching** | Deleting or substituting one coupling term in the velocity field, then measuring the trajectory difference | `p2_eigenspectra/head_ablation.py`, `core/intervention.py` |
| **Probe / LDA direction** | A hyperplane in configuration space; a probe's accuracy is a statement about particle separation, not about a "feature" | `p6_subspace/probe_subspace.py`, `core/dual_reading.py` |

**`core/dual_reading.py` is already the bridge primitive** — its whole design is a paired
*geometric* reading (V-projection, real/imaginary split, effective-rank contribution) and
*semantic* reading (frozen-head decode, probe membership) of the same point. C1 should be
written against that schema rather than inventing a parallel one.

### C2. Differential predictions, registered · **DONE** · [D] · M

One registered prediction per bridged construct, entering `claims/registry.json` under
H-BRIDGE *before* any of them is run. Draft directions (to be sharpened into falsifiers with
instruments in the chunk itself):

- **Induction.** If induction is a matching-kernel coupling rather than a feature-copying
  circuit, then ablating it should change *inter-particle* geometry (pair-distance
  distribution at the matched positions) and not merely the logit at the copied token. The
  standard account predicts the logit effect and is silent on the geometry; a null geometric
  effect with a large logit effect falsifies the particle reading.
- **Steering.** The particle account predicts steering effect size is a function of the
  vector's *V-eigenbasis decomposition*, not of its norm alone — so two steering vectors of
  equal norm with opposite attractive/repulsive projections should have opposite-signed
  effects on effective rank. The standard account predicts effect scales with norm along the
  "feature direction" and says nothing about sign reversal.
- **SAE.** Dictionary elements should preferentially align with the *repulsive* subspace (the
  directions along which particles stay individuated), because those are the directions the
  population actually spans. A dictionary whose elements distribute isotropically across
  attractive and repulsive subspaces would falsify it. **Deferred**: Phase 3/4 are frozen; this
  is registered and left unrun, which is the correct state for a prediction whose instrument
  is frozen.

