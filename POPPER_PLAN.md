<!-- POPPER_PLAN.md -->
# POPPER_PLAN — Popperian adjudication, CI/CD, and the particle-paradigm bridge

**Status:** plan, written 2026-08-23. Nothing in this document has been adjudicated.
Chunks marked **DONE** below were executed in the same pass that wrote this file; everything
else is queued.

This is a *work-breakdown* document, not a design document. Each chunk states a deliverable, a
dependency, an acceptance test, and a cost tag, so the work can be handed out one chunk at a
time and picked up cold. Design rationale that outlives the chunk goes into the module's own
`DESIGN_*.md`, per the split `INDEX.md` already establishes.

Cost tags follow `UPDATE_PLAN.md`: **[D]** doc-only · **[R]** report-only, data already on
disk · **[W]** needs weights, no forward pass · **[F]** needs forward passes · **[C]** CI
infrastructure, no science data.

---

## 0. The three things being built, and why they are one project

1. **CI/CD** (workstream A). There is none. Nothing else in this document is checkable
   without it.
2. **A Popperian adjudication layer** (workstream B) — POPPER's sequential-falsification
   machinery (Huang, Jin, Li, Li, Candès & Leskovec, arXiv:2502.09858) attached to the
   predictions this project already registers, so that "the prediction survived" becomes a
   number with a Type-I guarantee rather than a verdict word.
3. **A particle-paradigm bridge** (workstream C) — definitions, in the interacting-particle
   paradigm this project works in, for the interpretability constructs that already have
   natural-language-interpretability definitions (induction heads, activation steering,
   SAEs/crosscoders, lenses, ablation/patching, probes).

They are one project because each is worthless alone. A bridge without falsifiers is a
metaphor. Falsifiers without CI are a convention that decays the first time someone is in a
hurry. CI without either is a green checkmark on a repo whose central claims are adjudicated
by prose.

The ordering is forced: **A gates B gates C.** The pre-registration guarantee that makes the
e-values valid (§B3) is a CI check or it is nothing, and the bridge's whole point is to emit
predictions into the machinery B builds.

---

## 1. Where things actually stand (verified this pass, not inherited)

| Fact | Evidence |
|---|---|
| No CI of any kind | no `.github/` directory exists |
| No dependency manifest | no `pyproject.toml`, `setup.py`, `requirements*.txt` anywhere in the tree |
| The pure test tier cannot run without heavy deps | `tests/conftest.py:20` hard-imports `torch` at module scope — which defeats the `sys.modules` stubbing the same file installs 26 lines later, and blocks collection of all ~100 test modules |
| `pytest.ini` does not exist | `tests/SMOKE_TESTS_NOTES.md` says it does and that the `smoke` marker is registered there; neither is true, so `-m smoke` currently selects nothing and every marker is unregistered |
| A stale orphan sits in `core/` | `core/.py` — 196 lines, an older truncated copy of `models.py` whose docstring asserts **bfloat16** loading while the live `core/models.py` asserts **float32** and calls that choice load-bearing. Unimportable (not a valid module name), so nothing catches the contradiction |
| 30 predictions are registered in prose | `P-γ1 P-γ2 P-H1 P-S1 P-T1 P-M1` and the three transition claims (`PREDICTIONS.md`), nine `P5b-*` (`p5b_report.py`, `logit_cache.py`), twelve `P6-*` (`report_6.py::_PREDICTIONS`) — scattered across `.md` and `.py`, with no machine-readable index. (`P1-P5` in `core/run_policy.py` is *not* one of them: it means "policies P1 through P5". A naive ID pattern counts it as a prediction, which is why `tools/check_registry.py`'s pattern excludes it explicitly.) |
| p-values already exist in the science code | `p6_subspace/head_classify.py`, `induction_ov.py`, `qk_decompose.py`, `p5_single_mstate_analysis/tiers.py`, `p5b_manifold_steering/isometry_test.py`, `merge_teleportation_subspace.py`, `core/nulls.py`, `core/qk_offset_null.py` — the e-value layer has real inputs to attach to on day one |
| Baseline suite health | **zero tests collected** before this pass; see `docs/CI_BASELINE.md` |

Two of these are load-bearing for the plan and worth stating plainly. **The project's
falsification discipline is already better than most published work** — `PREDICTIONS.md`
pre-registers with falsifier, instrument and cost; the P-T1 addendum corrects a
pre-registration by dated addendum rather than edit; `UPDATE_PLAN.md` §6 lists six standing
rules each derived from a defect that cost real work. Workstream B is not introducing rigor.
It is **mechanizing rigor that currently depends on the author remembering it**, and adding
the one thing prose cannot supply: a Type-I error rate.

And: **the discipline is already load-bearing enough that its gaps are the expensive bugs.**
`UPDATE_PLAN.md` §5.6 (a wrong trace contraction that the identity-case anchor could not
catch) and §5.5 (two peak-detection bugs, opposite in direction) are both cases where a
result would have been reported had a check not existed. Every one of those checks is
currently a test nobody runs on push.

---

## 2. Workstream A — CI/CD

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

## 3. Workstream B — the Popperian adjudication layer

### What is being adopted from POPPER, and what is not

POPPER is an *agentic* framework: an LLM design agent proposes sub-hypotheses, a relevance
checker screens them, an execution agent produces a p-value, and the p-values are combined
into an anytime-valid e-process. This project is not agentic and should not become so. What
transfers is the statistical core; what does not transfer is replaced by the discipline the
project already practises.

| POPPER component | Here |
|---|---|
| Experiment design agent | **Human/pre-registered.** `PREDICTIONS.md` already does this, with falsifier and instrument stated up front |
| Relevance checker `R(h) ∈ [0.1, 1.0]`, threshold `r₀` | **A declared field** on each registry entry, scored against the paper's own rubric (Listing 4), reviewed by a human. CI enforces `relevance ≥ r₀`; it does not compute relevance |
| Experiment execution agent → p-value | **The existing science code.** Eight modules already produce p-values (§1) |
| p-to-e calibrator `eᵢ = κ·pᵢ^(κ−1)` | **Adopted verbatim.** κ = 0.5 fixed and declared (the paper's Table 1 confirms this choice: p = 1.0 → cumulative e = 0.5) |
| Product e-process `E = ∏ eᵢ`, reject at `E ≥ 1/α` | **Adopted verbatim** |
| Assumptions 1–3 | **Adopted, and mechanized** — see B3, which is where CI and Popper actually fuse |

The reason to adopt the e-value core rather than, say, Fisher's combined test is the one the
paper measures: Fisher's method fails Type-I control here (0.311 against a nominal 0.1 on
DiscoveryBench) precisely because it assumes independent p-values and cannot survive optional
stopping. This project stops when it stops — the sweep is gated, phases are descoped
mid-flight, `INDEX.md` records three phases going out of scope in one day. **Optional
stopping is not a hypothetical here; it is the project's actual operating mode**, and it is
the exact thing an e-process is valid under and a p-value combination is not.

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

**Git can check that.** `tools/check_preregistration.py`, run in tier 0:

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

### B6. Retrofit the existing p-value sites · **FIRST ONE DONE** · [R] · L — **split one module per chunk**

`p6_subspace/induction_ov.py` (P6-I1) is threaded end to end and is the worked example the other seven follow. Three things it settled that the plan had not:

- **Adjudication is opt-in per run, not automatic.** `run_induction_ov` is exercised by fixtures, and `adjudicate` refuses to overwrite an existing record — correctly, since silent replacement is how evidence disappears without trace. So one accidental fixture run would permanently occupy P6-I1's slot in the real ledger with a synthetic p-value. Every retrofit must carry the same `ctx["adjudicate"]` gate and the same `adjudications_dir` override.
- **A test that could not run is not a refusal.** `compare_induction_vs_semantic` returns `mwu_pvalue=None` when an arm has fewer than two heads. That passes through quietly; putting it in the log as a refusal would file a data limitation under policy violation.
- **The summary must say when adjudication did *not* happen**, and say that a refusal is not a failed prediction. A silent absence reads as a passed test.


In dependency order, cheapest first: `core/nulls.py` (`sigma_from_null` → add
`p_from_null`, one-sided and two-sided, with the ≤(n+1)⁻¹ resolution floor stated),
`core/qk_offset_null.py`, `p6_subspace/induction_ov.py` (P6-I1, already MWU),
`p6_subspace/head_classify.py`, `p6_subspace/qk_decompose.py`,
`p5b_manifold_steering/isometry_test.py`, `p5b_manifold_steering/merge_teleportation_subspace.py`,
`p5_single_mstate_analysis/tiers.py`.

Each is one chunk: thread `prediction_id` through, call `core.adjudication.adjudicate`, assert
in tests that the emitted record round-trips. No science changes, no threshold changes.

### B7. `FALSIFICATION.md` generation + CI recomputation · [C] · M

`tools/render_falsification.py` builds, per claim, the ordered table of adjudicated
predictions with p, e, running E, and the decision at α — replacing the hand-maintained
verdict tables in each `status-N.md` with a generated artifact.

And the check that makes the committed record self-verifying: **CI recomputes E from the
committed adjudication records and fails if any reported decision disagrees.** Deterministic,
no artifacts, no heavy deps, runs in tier 0. It catches arithmetic drift and, more usefully,
catches a verdict word that was updated by hand without its evidence.

### B8. Wire the gate that already exists · [R] · S

`PREDICTIONS.md` claim (c) carries a hard stop: *"If this fails, no checkpoint-sweep work
(items 9–11) proceeds past the gate."* Once H-TRANSFER has an e-process, that sentence becomes
a check: `tools/gate_status.py` reports the claim's E and refuses to mark sweep chunks runnable
below threshold. A stop rule that is written down but not wired is a stop rule that gets
argued with at exactly the moment it matters.

---

## 4. Workstream C — the particle-paradigm bridge

### The problem, stated precisely

This project studies tokens as **particles on `S^{d-1}` evolving under an interacting-particle
system** — the Geshkovski et al. dynamics, with residual blocks read as forward-Euler steps.
Mainstream interpretability studies the same objects with a different vocabulary: induction
heads, steering vectors, SAE features, lenses, patching. Both describe the same forward pass.

The bridge is only worth building if it is **falsifiable**, and a translation table alone is
not. A definition that merely re-describes an induction head in particle language predicts
nothing and cannot be wrong. So every entry in the bridge must come with a **differential
prediction**: something the particle account says that the standard account does not, where
the two disagree observably.

That constraint is what makes C a research workstream rather than a glossary, and it is why C
is sequenced after B — each entry emits into the registry.

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

### C3. Claims layer over the existing phases · [D] · M

The `p1…p6` numbering is historical accretion, and `INDEX.md` documents the strain: 2b is
called "2i" on disk, 2c/3/4 went out of scope in one day, and 5c's frame was promoted to be
the project's frame. **Do not rename the directories** — every path, artifact stem and test
fixture depends on them, and the churn buys nothing.

Instead, add the claim layer on top: `claims/CLAIMS.md` maps each main hypothesis to its
predictions to its instruments to the phase directories that implement them. Phases become
*instruments*; claims become the organizing unit. This is the reorganization the user asked
for, done additively.

### C4. Wire the ontology into `dual_reading` · [R] · M
### C5. Write `FROZEN.md` for Phase 3 and `low_rank_ae.py` · [D] · S

Both are listed as outstanding in `INDEX.md`'s "not done" section, and A3's marker taxonomy
needs them: a frozen module should be `heavy`-marked or excluded explicitly, not silently
untested. Small chunk, unblocks a lint rule.

### C6. Figure/catalogue conventions for particle-paradigm plots · [D] · M

---

## 5. Execution order

```
A1 ─┬─ A2 ── A3 ── A4 ── A6          (CI runs at all)
    └─ A5 ── A7                       (CI enforces the project's own rules)
             │
B1 ──────────┴─ B2 ── B3 ── B4 ── B5 ── B6 (×8, parallel) ── B7 ── B8
                                     │
                          C1 ── C2 ──┴─ C3 ── C4 ── C5 ── C6
```

**Critical path: A2 → B1 → B2 → B3.** Everything else can be parallelized or deferred. A2
because no tier is fast without it; B1 because it is the only piece with a correctness
proof to get right; B2/B3 because the e-values are meaningless without the pre-registration
guarantee they enforce.

**Parallelizable immediately after B4:** all eight B6 retrofits, and C1 (doc-only, no code
dependency).

**Deliberately last:** C3's claims layer, because writing it before B5's evaluability audit
would fix a claim structure before knowing which predictions can actually adjudicate it.

---

## 6. Four risks worth naming now

1. **A mis-specified null voids the whole claim, not just its own prediction.** E-values
   multiply; the product is only as valid as its weakest factor. This is why B5 exists and why
   B4 refuses rather than degrades. It is the same failure POPPER measures when it removes its
   relevance checker: Type-I error jumps from 0.082 to 0.340 on TargetVal-IL2, purely from
   irrelevant nulls being "falsified."
2. **Reusing one artifact across predictions can break conditional validity** even though
   e-values need no independence. The violation is not reuse — it is *registering a prediction
   after seeing the artifact and then testing it on that artifact*. B3's third check is the
   only thing that detects it.
3. **κ must be fixed in advance.** Choosing κ after seeing p-values invalidates the
   calibration. Fixed at 0.5, declared per-entry, and immutable-after-adjudication under B3.
4. **Backfilled pre-registration is evidence, not proof.** See B3's caveat. Backfilled entries
   should be reported separately forever, not merged into the same table as prospectively
   gated ones.

---

## 6b. What the first pass actually found

Executed 2026-08-23: workstream A in full, B1–B3 and B5, C1.

Four things were discovered by doing the work rather than by planning it, and
each changes something downstream:

1. **The suite collected zero tests**, not "most of them" — a conftest-level
   import failure takes the whole directory down. And two packages
   (`p3_crosscoder`, `p4_mstate_features`) could not be imported at all because
   their `__init__` referenced `.analysis` where the module is `analysis_p4`,
   so those phases' tests had never run once. See `docs/CI_BASELINE.md`.

2. **`.gitignore` whitelisted only `*.py` and `*.md`.** Every file this plan
   needs — `pyproject.toml`, `requirements/*.txt`, the workflows, the registry
   JSON — was unstageable. This would have blocked workstream A on its first
   commit, and it is invisible until you try.

3. **Seven of thirty predictions can carry an e-value.** Twenty need a null
   constructed; three admit none at all. That number is the real finding of
   workstream B, and it re-weights the plan: B6's retrofit of existing p-value
   sites is much smaller than expected, while a *new* workstream — constructing
   the twenty missing nulls — is the actual bulk of the Popperian work.
   `claims/EVALUABILITY.md` names three recurring patterns behind it (a
   threshold is not a null; an equivalence claim needs an equivalence test; the
   same data cannot settle two entries).

4. **The bridge's only adjudicated prediction has already failed.** P6-R2/R4
   came back inverted — 0 of 49 layers in the predicted direction. `C1` records
   this rather than routing around it, and declines to register a fresh probe
   prediction until the inversion has a null construction, because inventing
   one now would be fitting theory to a result already in hand.

**Revised critical path**, replacing §5's: A is done, so the path is now
**B5 → the twenty null constructions → B4 → B6 → B7**, with C2 (registering the
four bridge predictions prospectively) runnable in parallel since it depends on
no artifact. The first prediction to adjudicate is `PB-STEER1` from C1 — cheap,
no new instrument, and the one place the two paradigms make *incompatible*
rather than merely different predictions.

## 6c. What main's archive + Phase 7 changed (2026-08-23)

Main archived Phases 3, 4, 5, 5b, 5c and 6 and opened **Phase 7 — the
mechinterp/particle bridge** while this branch was in flight. Two decisions
followed, both the author's:

**Workstream C folded into Phase 7.** `docs/PARTICLE_ONTOLOGY.md` and
`p7_motifs/design-7.md` were the same project, built independently, down to the
same anti-glossary constraint. Phase 7's formalization is sharper — a named
mechinterp phenomenon is a *motif*, a recurring structure of typed particle
interactions, countable against a matched null — and it has running code. So the
ontology was deleted and its four differential predictions renumbered into Phase
7's own scheme (`P-I5`, `P-ST1`, `P-AB1`, `P-SA1`), rather than kept under a
second `PB-*` convention inside one phase. C1, C2 and C4–C6 are therefore closed
by Phase 7 rather than by this branch; C3's claims layer stands.

**Dormant status.** 21 of 38 registered predictions instrument an archived
phase — all twelve `P6-*`, all nine `P5b-*`. They carry `status: "dormant"`:
`core/adjudication.py` refuses them and they contribute nothing to any claim's E,
but they stay registered, counted, and visible with their falsifiers intact.
Deleting a pre-registered prediction because its apparatus went away would leave
the record as the flattering subset of what was actually predicted, which is the
exact failure the pre-registration gate exists to prevent. `status` is orthogonal
to `evaluable`: `P6-I1` is `e-value` *and* dormant — its Mann-Whitney U was and
remains a valid test; what it lacks is a live module to run it.

**What this costs, stated plainly.** H-OPERATOR now has almost no live path to
adjudication (12 of 14 dormant; only `P-T1` and `P-M1` survive, both on Phase 2d).
Three predictions are adjudicable today: `P-S1`, `P-T1`, `P-M1`. The B6 retrofit
of existing p-value sites is mostly moot — those sites are archived — so what
remains of workstream B is B5's null constructions and B7's generated
falsification table, now pointed at Phase 7 and Phases 1c/2d rather than at 5b/6.

## 7. What this plan does *not* do

- It does not run any science. No chunk here adjudicates a prediction; B6 makes adjudication
  *possible*, and the first real adjudication is a separate decision gated on
  `tools/preflight_1c.py` per `UPDATE_PLAN.md` §3.
- It does not unfreeze Phase 2c/3/4. C2's SAE prediction is registered and left unrun.
- It does not rename phase directories (§C3).
- It does not add an LLM to the loop. POPPER's design and relevance agents are replaced by the
  human pre-registration the project already practises (§3).
