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

### B7. `FALSIFICATION.md` generation + CI recomputation · **DONE** · [C] · M

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

## 6d. First live null construction (2026-08-23, post-merge)

**P-S1 now has a calibrated p-value.** Its null machinery already existed --
`random_band` sampled 200 draws of the Q_k ratio -- but stopped at "Nσ from
null", a summary that reads like significance without being calibrated as one.
`core/nulls.p_from_null` supplies the missing step for every such null in the
project, with the `(n_extreme + 1)/(n + 1)` floor: a p of exactly 0 calibrates
to an infinite e-value, asserting more evidence than a finite sample can carry.

Two choices had to be fixed before running, because neither was settled by the
original wording and both are selections if made afterward:

- **Multi-degree combination.** One pre-declared scalar -- the sum over degrees
  1–3 of the (step0 − trained) ratio, each standardised by that degree's own
  null sd -- rather than three per-degree tests plus a correction chosen after
  seeing three p-values. Standardising is load-bearing: `UPDATE_PLAN.md` §5.8
  measured the band narrowing from ~0.17 at k=1 to ~0.002 at k=3, so an
  unstandardised sum is dominated by k=1 and discards the degrees that are more
  sensitive in relative terms.
- **Direction.** One-sided `greater`, since P-S1 predicts trained ratios
  *smaller*. Verified: reversed arms give p = 1.000.

**A calibration defect was found and fixed by measuring rather than reasoning.**
The first implementation referenced the observed statistic against
`design_report`'s internal baseline and the null against its own — two
Monte-Carlo estimates of the same quantity. Null-vs-null pairs came back with a
mean p of 0.40 against the 0.50 a calibrated statistic must give, in the
anticonservative direction and invisible in any single result. Re-referencing
both arms against one shared baseline brings the rejection rate to nominal
(frac p ≤ 0.05 = 0.050 at n = 40); power is 88% at α = 0.05 against a
simplex-like configuration.

**B7 is done.** `tools/render_falsification.py` generates
`claims/FALSIFICATION.md`, and tier 0 now recomputes every claim's E from the
committed records — recalibrating each e-value from its p-value rather than
trusting the stored number — so a decision word edited by hand surfaces in CI.

## 6e. All three live predictions now have calibrated nulls

`P-S1`, `P-T1`, `P-M1` — the complete set of predictions that are both
`e-value` and `active` — can now each produce a p-value. Every one is a
permutation test, and in each case the permutation is of the thing the
prediction's own falsifier names:

| | statistic | null permutes | falsifier it realises |
|---|---|---|---|
| `P-S1` | sum over degrees 1–3 of the (step0 − trained) Q_k ratio, standardised per degree | two independent i.i.d. configurations at matched (m, d) | "no difference between trained and step-0" |
| `P-T1` | trimodal-rate(candidates) − trimodal-rate(controls) | the row-2 classification labels | "trimodality is a property of the activations, not of the classification" |
| `P-M1` | correlation of per-layer mean regime distance with the violation series | the violation series against layers | "violations are not explained by leaving the gradient-flow regime" |

That pattern is worth noticing rather than treating as coincidence: **a
falsifier stated well enough to be falsifiable usually names its own null.**
Where these three were easy to construct, it was because `PREDICTIONS.md` had
already done the hard part. The remaining `needs-null` entries are the ones
whose falsifiers are stated as thresholds instead.

Two decisions each had to be fixed before running, since either is a selection
if made after seeing results — the multi-degree combination and direction for
P-S1, the mode-count source and the spacing question for P-T1, the aggregate
choice and direction for P-M1. All are module constants rather than parameters,
so a per-run choice is not available, and all are recorded in the registry's
`null_construction`.

**Two findings from doing it:**

- **`adjudicate_p_t1` contradicts the P-T1 amendment.** It reads
  `modality["trimodal"]` — a mode count at one bandwidth — while the dated
  addendum says "adjudicate on `stable_n_modes` only ... a mode count at a
  single bandwidth is a choice, not a measurement." The two disagree wherever
  the bandwidth scan does not settle. `p_value_p_t1` implements the amended
  version; a note now sits on `adjudicate_p_t1` itself so a reader is not
  misled by the name.
- **P-M1 refuses rather than choosing.** When the mean/min/max head-to-layer
  aggregates disagree in sign, no p-value is emitted. `adjudicate_p_m1` already
  established that this means per-layer energies cannot resolve a per-head
  claim; producing a number for one chosen aggregate would convert a resolution
  limit into a result.

**What is still missing is data, not apparatus.** No artifacts exist in this
repo (`BASE_RESULTS_DIR` does not), so all three are validated on synthetic
inputs with known answers — calibration under H0, power against a planted
effect, and p ≈ 1 when the effect is in the wrong arm. The ledger is still
empty, and `claims/FALSIFICATION.md` says so.

## 6f. CLAIM-C's null, and the four things its wording left open (2026-08-24)

`CLAIM-C` — the transfer claim under H-TRANSFER, and the only prediction in the
registry carrying a hard stop — now has a construction, in
`p1_mstate_tracking/replication_gate.py`. It is the first `needs-null` entry
converted, and `claims/EVALUABILITY.md` had already named it the one to do
first: *"a stop rule that cannot be adjudicated is a stop rule that gets argued
with at the moment it binds."*

**The criterion.** "Reproduces" is read as sign-concordance of the
trained-minus-random *contrast*, over the six per-layer series
`CHECKPOINT_METRICS` already registers, on a common normalized-depth grid
(gpt2-large has 36 layers, pythia-1.4b has 24 — nothing is comparable before
depth is normalized). Blog 1's phenomenology is a contrast, not a set of
absolute levels, so the object that has to transfer is the contrast.

**The null is the same shape §6e found in the other three.** Permute the
trained/random condition label on the candidate side, gpt2-large held fixed as
the reference. `delta` is antisymmetric in (trained, random), so a swap is an
exact sign flip and the null has a closed form — which lets it be enumerated
**exhaustively** rather than sampled: 256 patterns for the eight metastability
prompts, so the test is exact rather than Monte-Carlo. That is the first exact
null in the project.

**The exchangeable unit is the prompt, not the cell.** The label "trained"
attaches to a run, so swapping it moves all six metrics of one prompt together.
Flipping cells independently would treat six metrics on one prompt as six
independent observations — the same error `status-6.md` records for "49 ALBERT
layers are not 49 independent observations", and the reason that result is
still `needs-null`. Choosing the unit is where the p-value was won or lost here,
not choosing the statistic.

**A refusal that is new in kind.** The existing refusals are about the data
(P-M1's aggregates disagreeing in sign). This one is about the *design*: with n
prompts the enumeration's smallest expressible p is `2/(2^n + 1)`, so at four
prompts a **perfect** result gives p = 0.118 and the test cannot reject at
α = 0.05 however clean the data is. The module refuses rather than reporting
"not significant" on nothing — which on a hard-stop claim would read as
evidence against transfer. Six prompts is the first workable gate. Worth
generalising: several remaining `needs-null` entries are small-n permutation
designs, and the attainable floor should be checked before the null is built,
not after a result comes back null.

**And a second refusal, found by measuring the limitation instead of noting
it.** The plan was to document "prompts on one model are not independent runs"
and move on. Measuring it changed what the code does. With independent sign
rows the rejection rate at α = 0.05 is ≈0.015 — conservative, as a discrete
statistic should be. With every row identical it is **≈0.34** (3000 draws
each): numerically the
same fourfold-plus inflation §6 lists as risk 1, POPPER's 0.082 → 0.340 when
its relevance checker is removed, arriving here from a completely different
direction. So the exactly-degenerate case is refused rather than documented —
identical rows mean the prompts contribute one observation, and enumerating
2^n patterns over one observation is the wrong null, not a conservative one —
and `sign_homogeneity` is reported in between so a reader can place a real run
between the two measured rates. This is a degeneracy and not a tolerance: rows
are either all equal or they are not, so the ordinal criterion stays free of
the magnitude cut it was chosen to avoid needing.

The transferable lesson is the one §6d already recorded in a different form:
**measure the calibration, do not reason about it.** §6d found P-S1
anticonservative at a null-p mean of 0.40 by simulating; this pass found the
size of a limitation everyone would have been content to describe in prose.

**The stop rule is three-way, and only one branch is a falsification.**
TRANSFERS (`p_greater ≤ α`), FAILS-TO-TRANSFER (the reciprocal test rejects —
the contrast systematically *inverts*), INSUFFICIENT (neither). The hard stop
fires on both of the latter — an unadjudicated gate stops the sweep — but only
FAILS-TO-TRANSFER enters the ledger as a falsification, because an e-process
records "insufficient evidence" and never "null accepted". **Only `p_greater`
is calibrated into an e-value.** `p_reciprocal` is a stop-rule input, recorded
in the record's notes and kept out of H-TRANSFER's product; two one-sided tests
on one statistic would otherwise double the claim's Type-I rate for free.

**Four things the registered wording did not settle**, decided here and written
into `null_construction` so a later reader is not left to infer them:

1. **The criterion adjudicates the contrast, not the two absolute reproductions
   the statement's words name.** A pythia pair whose levels both sit far from
   gpt2-large's but whose difference has the same sign passes. The cost is that
   the criterion is scale-blind; the absolute per-arm profile distances are
   computed and reported as a diagnostic and enter no p-value.
2. **The two-baseline policy.** `PREDICTIONS.md` attaches it to this claim
   specifically. The p-value runs on the norm-matched `pythia-1.4b-random`,
   which is what the statement names; the true step-0 init is a *mandatory*
   sensitivity arm — refused on omission, the same refusal
   `centroids.load_centroids` makes — reported beside the result and kept out
   of the p-value, since step 0 is CLAIM-A's object and one dataset must not
   settle two entries. Direction disagreement between the two baselines is
   flagged in the record.
3. **`effective_rank` is read from `effective_rank_normed`.** `status-1.md`
   defect D1: the raw field mixes directional collapse with residual-stream
   norm growth. Baking the known-defective field into the gate that carries the
   hard stop would be knowingly wrong.
4. **Full normalized depth, no band restriction.** Blog 1 quotes layers 5–30 of
   gpt2-large, but a depth band is a choice with as many options as there are
   bands.

**The limitation that does not go away.** Prompts run on one model share that
model's weights, so the rows are not fully independent either: a pythia-wide
effect present in every prompt is invisible to the enumeration. The prompt is
the coarsest unit this design provides — a coarser one would need independent
training runs, which do not exist. Refusing at the degenerate end bounds the
damage; it does not remove it, and every record the module emits carries both
measured rates in its notes rather than leaving them to a reader.

**A second agreement axis, added the same day and still pre-data.** The author
asked whether agreement across *tools* could sit alongside agreement across
*architectures*, on the reasoning that together they are a stronger argument
while individually a disagreement is not a death sentence — there are
instrument quirks nobody is privy to. It was not too late: `null_construction`
freezes only on a prediction that has been adjudicated
(`check_registry.py:359` iterates `sorted(adjudications)`), the ledger is
empty, and no gate data exists. That window closes at the first adjudication
and not before.

The axis is **metric leave-one-out**: the whole cross-architecture test is
re-run once per subset with one metric dropped, and the gate requires
**unanimity in both directions**. Four things about it are worth carrying
forward:

1. **The two axes stay separate factors.** The gpt/pythia axis is the claim;
   the metric axis is a statement about the instrument. Folded into one
   p-value, a failure is ambiguous between "the phenomenology does not
   transfer" and "one of our six measurements is quirky", and those have
   opposite consequences for the sweep.
2. **Intersection-union, so no multiplicity correction.** The alternative is a
   conjunction, and max(p) is a valid p-value for a conjunction *regardless of
   dependence* — which is what makes it right here, since six leave-one-out
   runs share five sixths of their data and any Bonferroni-style correction
   over them would be absurd.
3. **The hard stop is not weakened.** Both directions get harder and the
   INSUFFICIENT middle grows, but the stop already fires on INSUFFICIENT. Only
   the word *falsified* is reserved for an inversion no single metric carries.
4. **The rule is "no subset may fail", not "no metric may dissent"** — a
   distinction that only showed up when a test written on the looser reading
   failed. Five of six metrics inverting on every prompt survives every
   leave-one-out and is correctly a falsification. The looser reading would
   let one quirky measurement veto a real result and make the gate
   unfalsifiable in practice, which is the failure the whole apparatus exists
   to prevent. `TestToolAxis` pins both halves.

The attainable-floor refusal is unchanged — a max of p-values each at or above
`2/(2^n + 1)` is at or above it — and a new refusal joins it: if any subset
cannot carry a p-value, the gate refuses rather than taking a max over a set
with an undefined member, which would silently drop whichever subset was
hardest to satisfy.

**Still no data.** As in §6e, the apparatus exists and the artifacts do not.
Validation is on synthetic inputs with known answers: exactness of the
enumeration, validity under H0, power against perfect transfer, p = 1.000 with
the arms reversed, and every refusal. `claims/adjudications/` remains empty.

## 6g. CLAIM-C's homogeneity calibration curve (2026-08-24)

§6f ended with a limitation described as bounded: the sign-flip null treats n
prompts as n pieces of information, they are not independent, and the cost was
**measured** at the two ends — ≈0.015 at α = 0.05 with independent sign rows,
≈0.34 with identical ones. The gate refused at the exactly-degenerate end and
reported `sign_homogeneity` in between.

Two endpoints are not a correction. **Everything between them was
uncontrolled, and a real run lands in the middle** — so the middle is now
measured too, offline, once, and committed:
`tools/calibrate_claim_c_homogeneity.py` →
`claims/calibration/claim_c_homogeneity.json`. The gate reads

    R(h, p) = P( it reports a p ≤ p │ it reported one at all, under H0,
                 at prompt sign-row homogeneity h )

and reports `max(p_exact, R(sign_homogeneity, p_exact))`.

**Two decisions had to be fixed first, and both were put to the author before
any curve existed** — picking either after seeing gate data would void the
guarantee the curve exists to restore, the same way choosing a tail afterward
would.

- **The correction ADJUSTS the reported p; it is not a diagnostic beside it.**
  The corrected number is what enters H-TRANSFER's e-value. `p_exact` stays in
  the record. The alternative — report both, let the reader apply it — leaves
  a p the project has already measured to be anticonservative sitting in the
  ledger, and `EVALUABILITY.md`'s opening argument is that one bad factor
  voids the product silently.
- **Blunt, never sharpen.** `max(·)`, not `R(·)`. At the independent end the
  exhaustive enumeration is genuinely *conservative*, and calibrating there
  would be a real power gain — the corrected p would fall. It is refused
  anyway: that trades an exact conditional guarantee for a simulated one, on
  the one claim carrying a hard stop. The correction may only cost power.

**The refusal is derived from α, not placed.** The second question was whether
some homogeneity is too high to correct at all. It is, and the cut needs no
tolerance: refuse when `R(h, 2/(2^n + 1)) > α`, i.e. when even a **perfect**
result would not survive its own correction. That is §6f's attainable-floor
refusal one level up, from the same two inputs (α and the null size). No
homogeneity constant appears anywhere in the module, and the boundary moves
when α does — which is what a test distinguishes a derived cut from a placed
one by. Measured, it lands near homogeneity **0.80–0.85** for every tabulated
prompt count.

**Five things worth carrying forward.**

1. **The simulation is exact, not sampled.** A subset's p depends only on the
   *multiset* of per-row concordant counts — the null lets row i contribute
   `conc_i` or `m − conc_i`, so both the observed statistic and the whole null
   distribution follow from the histogram. There are only C(n+m, m) of those
   (3003 at eight prompts and six metrics), so every attainable p is tabulated
   once by integer convolution and each draw is a lookup. No Monte-Carlo error
   enters the p-values; only the rates carry sampling error. It is a second
   implementation of the gate's arithmetic, which is a real risk, so
   `TestFastPathMatchesTheGate` pins it against `p_value_claim_c` cell by cell.

2. **Rates are conditional on the gate EMITTING a p, and that is the subtle
   part.** Unconditional rates would let the gate look calibrated *by
   refusing*: at high homogeneity most draws hit the identical-rows refusal,
   and counting those as non-rejections pushes the measured rate down exactly
   where the inflation is worst. The ledger only ever receives runs that
   emitted, so the conditional rate is the one that governs it.

3. **It was validated OUT of sample, and that is the claim worth making.**
   `sign_homogeneity` is a scalar summary and a scalar cannot determine a
   distribution, so the open question was never "does this work on the family
   it was fitted to" but "does indexing by it transfer to a different
   *mechanism* of dependence". Re-measured on a duplicate-prompt mixture
   ("some prompts are redundant") rather than the fitted per-metric propensity
   ("some metrics are architecture-wide"), the uncorrected rate still inflates
   to 0.23 while the corrected one stays at or below nominal across the range.

   | | uncorrected | corrected |
   |---|---|---|
   | worst fitted configuration | 0.199 | 0.046 |
   | mixture, ρ = 0.4 | 0.036 | 0.008 |
   | mixture, ρ = 0.6 | 0.094 | 0.008 |
   | mixture, ρ = 0.8 | 0.175 | 0.008 |

4. **§6f's two headline numbers describe a gate that no longer exists.** The
   0.015 and 0.34 were measured *before* the metric-leave-one-out axis was
   added the same day. With the axis the independent-rows rate is ≈0.003,
   because the reported p is a max over seven subsets. Neither number was
   wrong when written; both stopped being about the live gate a few hours
   later, and nothing would have noticed. They are kept in the record as
   history, and the code now reads a curve that is checked against the gate it
   corrects (`check_curve`, pinned in the pure tier) instead of prose that
   cannot be.

5. **A defect found by measuring, again.** The stored quantiles were first
   rounded to 8 decimals. The derived refusal turns on whether a stored value
   is at or below `2/(2^n + 1)`, and rounding moves that value *up* for
   n ∈ {7, 9, 10, 11} and down for n ∈ {6, 8, 12} — so the refusal silently
   switched off for four of the seven tabulated prompt counts while the file
   looked entirely normal. It surfaced as an implausible pattern in the
   refusal map, not as any failure. Values are now **truncated**, which can
   only make a stored rate more conservative. §6d's lesson holds a third time:
   *measure the calibration, do not reason about it* — and look at the result
   even when nothing failed.

**What this deliberately did not do.** It adds no third robustness axis. Each
axis moves probability mass into INSUFFICIENT, which fires the hard stop, and a
stop rule that always fires carries no information. The correction is a
calibration of the existing p, not a new agreement requirement.

**What it leaves open, stated now.** Every simulated draw has a complete
(prompt × metric) table. A real run that drops cells — a non-finite or exactly
zero contrast — has a coarser statistic than anything tabulated, so its
correction is read off a table measured on a slightly different design. The
gate reports `n_cells_dropped` beside the correction and every record says so;
the honest fix is a second dimension on the curve, and it is not built.

**Still no data.** As in §6e and §6f, the apparatus exists and the artifacts do
not. `claims/adjudications/` remains empty, and `null_construction` — which
freezes at the first adjudication and has still not frozen — now records the
correction, both refusals, the H0 family the rates are rates under, and what
the curve does not cover.

## 6h. P6-R2/R4: the strongest recorded result, and why it is not a p-value (2026-08-24)

`EVALUABILITY.md` called the Phase 6 inversion "the strongest single result in
the registry" and `CLAIMS.md` called it a falsification H-OPERATOR already
carries: mean LDA alignment 0.887 with the imaginary subspace $U_A$ against
0.067 with the real repulsive $U_\text{neg}$, **0 of 49 layers** in the
predicted direction, probe accuracy real-only 0.152 against imaginary-only
0.564. Both entries sat at `needs-null` and `dormant` for one stated reason:
49 ALBERT layers are not 49 independent observations.

The reason was true. It was not the binding one, and the entry that was binding
was not on the list.

**The prerequisite came first, because it was pre-registered.** `status-6.md`
item 5 names a projector-construction error — Schur block mislabelling swapping
$U_\text{neg}$ and $U_A$ — as a live alternative explanation, and `design-6.md`
had already fixed the ordering: *"the design explicitly prioritizes ruling out
the first (a Schur sign/block-type convention check) before treating the second
as established."* Nothing had done it in four months. Emitting a p-value with it
open would be putting a calibrated number on possibly-broken instrumentation,
which is standing rule 4's whole subject.

`tools/audit_p6_projector_labels.py` settles it and commits the record to
`claims/audits/p6_projector_labels.json`. It is a **tool and not a test**
because settling it means running the archived code that produced the number,
and `archive/README.md` rule 1 is that nothing under `archive/` is imported by
anything live. The tool loads it by file path when run;
`tests/test_p6_projector_audit.py` pins the committed result together with the
sha256 of the file it describes, so the record going stale is a failure rather
than a silence. Same division of labour as CLAIM-C's committed curve, for the
same reason: a finding that takes ~100 seconds to recompute should not have to
be recomputed to be trusted, but it must not be able to drift.

**Explanation (a) is RULED OUT, on two independent routes** — and the second
route exists only because the sensitivity arm caught the first one being
incapable of failing.

Arm L plants OV matrices whose real-positive, real-negative, rotation and kernel
structure is known by construction and checks that each bucket recovers its own
span: worst principal angle 3.3e-08 rad against a 1e-05 cut. Arm S then runs the
same assertions against two deliberate breakages — a relabelling that swaps the
neg and rotation buckets, and the transposed-subdiagonal bug (`T[i, i+1]`
instead of `T[i+1, i]`) that real Schur form makes easy to write. The swap was
caught at $\pi/2$. **The transposition was not caught at all, and could not
have been.**

To plant known real-versus-rotational structure you build a block-diagonal
matrix of scaled rotations and real eigenvalues — and that matrix is **normal**.
A normal matrix's real Schur form is block *diagonal*, so its superdiagonal is
zero outside the 2×2 blocks, and the two index conventions return bit-identical
answers. *The family that makes ground truth unambiguous is exactly the family
that cannot express the bug.* Arm C was added for it: on non-normal matrices the
spectrum still fixes how many eigenvalues are real-positive, real-negative and
complex, `np.linalg.eigvals` reaches that without touching the Schur form, and
the transposed extractor disagrees with it. Arm L's PASS on that breakage is now
pinned as a **positive** assertion about the method, not left as an absence.

Without arm C the audit would have reported RULED-OUT on the strength of a check
that could not fail. That is the same defect class as §6d's and §6g's, arriving
from a third direction: **an audit needs its own null.**

**Explanation (c), which nobody had listed, is the binding one.**
`p6_subspace/math-6.md` §7.2 states it and this pass measured it. For a random
unit vector and a $k$-dimensional subspace, $\mathbb{E}[\lVert P_U v\rVert^2] =
k/d$ — alignment scales with dimension — and the projector build's resolution
order removes span($U_\text{pos}$) from $U_\text{neg}$ and span($U_S$) from
$U_A$, making $U_\text{neg}$ the doubly-shrunk bucket. Arm D measures what that
produces at `albert-xlarge-v2`'s exact $(d, \text{heads}, \text{head dim}) =
(2048, 16, 128)$:

| | value |
|---|---|
| $\dim U_A / \dim U_\text{neg}$, ALBERT's shape | **24.89** |
| observed alignment ratio $0.887/0.067$ | 13.24 |
| observed / chance | 0.53 |
| chance-normalized alignment, $U_A$ | 0.960 |
| chance-normalized alignment, $U_\text{neg}$ | **1.805** |

**The correction is nearly twice the effect it would explain.** Normalized, the
comparison does not merely stop favouring $U_A$ — it lands in the *predicted*
direction. The reference is random OV matrices at the right shape and not
ALBERT's trained weights, so this bounds the correction rather than reporting a
result; the actual dims are computed by the projector build on every run and
were never reported, and recovering them is one number that settles the reading
outright.

So the recorded inversion is **withdrawn as a falsification** and kept as a
measurement whose reading is unresolved. It is not weak evidence against
H-OPERATOR; it is not evidence either way, and **no choice of exchangeable unit
would have rescued it.** The apparatus question the plan had queued was real and
downstream of a question nobody had asked.

**The instrument is live again.** Taking `P6-R2` and `P6-R4` out of `dormant`
means an instrument that can produce their p-value, and `archive/README.md`
rule 2 is that nothing is salvaged by copying — so the projector path is
**rebuilt** in `p6_subspace/` against `core/particles.py` and `core/nulls.py`,
with the chance normalization built in rather than available. Both entries are
now `e-value` and `active`; ten of H-OPERATOR's twelve `P6-*` remain dormant.

**The null is a matched-dimension random subspace, and choosing it dissolved
the problem the plan had been braced for.** H0-OPERATOR is "the operator
classification carries no information about activation geometry" — realise it
directly by replacing the operator-derived subspaces with random ones *of the
same dimension*. Everything the statistic could read off dimension is held
fixed and only operator content moves.

That changes the attainable-floor arithmetic completely. Under a `CLAIM-C`-style
sign-flip enumeration the coarsest honest unit here is "one model", $n = 1$,
floor $2/(2^1+1) = 0.667$, and the design cannot reject however clean the data
is — the refusal §6f established one level up. Under randomisation over
**subspaces** rather than over units, $n = 1$ is no obstacle: the floor is
$1/(\text{draws}+1) = 0.0005$. **The binding constraint was the choice of null,
not the choice of exchangeable unit.** `attainable_floor_report` prints both
framings side by side rather than leaving that as a docstring claim, and it
generalises: several queued rows (`P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1`,
`P-I4`) name a matched control that is a subspace or a magnitude rather than a
unit.

**A second calibration defect, found the same way as the first two.** The first
implementation drew the null's two subspaces independently. But $U_\text{neg}$
and $U_A$ are **orthogonal by construction** — the resolution order guarantees
it — so an orthogonal observed pair was being compared against overlapping null
pairs. Measured over 400 replicates, the H0 rejection rate came back at
**0.0875** against a nominal 0.05, anticonservative and invisible in any single
result. Drawing the pair mutually orthogonal (one Stiefel draw, split, so each
half stays marginally uniform) brings it to **0.045**. Power is 100% against a
direction planted in $U_\text{neg}$ and $p = 1.000$ with the arms reversed.
§6d's lesson holds a fourth time: *measure the calibration, do not reason about
it.*

**And a plain bug, found by a test that expected a refusal and got none.**
`attainable_floor_report(n_units, n_draws=N_NULL_DRAWS)` binds its default once
at definition time, so the floor **refusal** was reading whatever the constant
held at import and would have reported a too-small null as safe. Resolved inside
the function now.

**What the exchangeable unit does control, measured rather than argued.** It
decides whether one null draw is shared across layers or drawn independently.
Over 400 replicates, as the layers come to share one direction:

| shared component ρ | unit = model | unit = layer |
|---|---|---|
| 0.0 (independent) | 0.0525 | 0.0525 |
| 0.5 | 0.0450 | 0.0800 |
| 0.9 | 0.0450 | **0.2325** |
| 1.0 (identical) | 0.0575 | **0.2800** |

The per-model unit holds nominal across the whole range; the per-layer unit
reaches 0.28, the same fourfold-plus inflation POPPER reports at 0.082 → 0.340
and §6f measured at 0.015 → 0.34. Third independent arrival at that number. The
mechanism is that the layer unit averages $n$ independent null draws where the
model unit averages $n$ copies of one, so its null is narrower by $\sqrt{n}$ —
and *that* is what the pure tier pins, deterministically and in milliseconds,
rather than resampling the consequence.

**Nothing is adjudicated, and the refusal is structural.**
`REGISTERED_EXCHANGEABLE_UNIT` is `None`, `adjudicate_p6_r2_r4` raises while it
is, and passing `unit=` does not route around it — that argument selects what to
**compute**, the module constant decides what may enter an e-process. The
evidence points unambiguously at `"model"`; registering it is a separate
decision, of the same class as CLAIM-C's criterion, and making it after seeing a
p-value would void the guarantee. `claims/adjudications/` is still empty.

**Both entries are registered as adjudicable, and their dependence is recorded.**
`P6-R2` and `P6-R4` are *not* the `P5b-B1`/`P5b-B3` pattern of one test with two
thresholds — R2 compares alignments, R4 compares probe accuracies, two
statistics on two instruments. But they share one projector, so a projector
defect moves both, and `null_construction` says so: a reader must not take their
product for two independent factors.

**What this leaves open, stated now.** The audit's dimension arm runs on random
OV matrices at ALBERT's shape, not on ALBERT's weights, so it bounds the
correction and does not report the run's own dims. `P6-R4`'s statistic has power
only while $U_S$ is a small fraction of $d_\text{model}$ — at 14 of 24 both arms
saturate and a planted effect reads $p = 1.0$, which was found by a fixture that
did exactly that. And nothing here produces a p-value, because as in §§6e–6g the
apparatus exists and the artifacts do not.

## 6i. CLAIM-B + P-I1: one construction, and the null the wording named was invalid (2026-08-24)

`claims/EVALUABILITY.md` closed on these two: *"`CLAIM-B` is next by the same
reasoning, and it shares a construction with `P-I1` — the same changepoint
co-location across a checkpoint sweep — so the two should be built together
rather than each inventing one."* They are.
`core/changepoint_colocation.py` is the construction and CLAIM-B's gate;
`p7_motifs/formation_gate.py` is P-I1's thin half. Both entries are now
`e-value`; eight predictions are adjudicable in principle and
`claims/adjudications/` is still empty.

Unlike `P6-R2`/`P6-R4` there was no dormancy to settle first and unlike
`CLAIM-C` no hard stop rides on it. What there was instead is a registered
null that does not work, and an obvious estimator that cannot carry the test.
**Both were found by computing before building, and neither would have shown up
in any single result.**

**The estimator is not `detect_transitions`, and the floor is why.** The
existing estimator returns the intervals of largest change per unit log-step —
adopting it is the reuse this project prefers and it fixes the choice in advance
for free. It was checked first. `interval_rates` divides by the log-step
spacing, and that spacing varies 4.6× across a 25-checkpoint Pythia sweep
(0.065 where the every-1000 releases compress at the top against 0.301 at the
bottom), so with the value series permuted against that fixed step grid the
argmax interval lands on the tightest-spacing one **44.7%** of the time. A
binary co-location statistic — "the two top intervals
coincide" — is therefore floored at ~0.29 typical and 0.45 worst case
and **cannot reject at any sensible alpha however clean the data is.** That is
§6f's attainable-floor refusal reached before building rather than after a null
result, and §6h's sharper form: check the floor against the null you *could*
build. `detect_transitions` also takes `n_top` and `min_abs`, both selections if
set after seeing the sweep.

**What replaced it carries no placed constant.** The location of a series'
change is the centroid, on the log-step axis, of the **change-mass profile**
$w_i \propto \max(\text{direction} \cdot (v_{i+1} - v_i),\, 0)$ — the share of
the series' total registered-direction change that happened in interval $i$. No
`n_top`, no `min_abs`, no tolerance on what counts as co-located, no smoothing
bandwidth. `EVALUABILITY.md` had asked whether there was an ordinal formulation
needing none, the way CLAIM-C's sign-concordance avoided a magnitude cut; this
is the answer, a distance in log10-step compared against a null.

**It is deliberately NOT divided by the log-step spacing, which departs from
`checkpoint_frames`, and the reason is power rather than validity.** Both
weightings are valid — H0 rejection 0.043–0.073 either way under the pairing
null that is actually used — but their power diverges as the sweep densifies.
Measured at 8 units and α = 0.05, change mass holds **1.000** across 20, 35, 80
and 143 checkpoints while rate falls **0.995, 0.970, 0.685, 0.090**: dividing by
$dx$ amplifies per-checkpoint noise exactly where the spacing is tight, and a
denser sweep makes every $dx$ tighter. **The log-step axis is right for plotting a
derivative, which is what `checkpoint_frames` built it for, and wrong for
weighting a location.** A change-mass profile takes no derivative at all, so
`spacing_change_steps`' "an index-based derivative will place a peak here by
construction" cannot reach it — and the spacing report is emitted in every
record anyway, so a reader checks that rather than taking it.

**The registered null was measured to be invalid, and the failure is large.**
Both entries said "a permutation null over checkpoint order gives a valid p once
the changepoint estimator is fixed in advance." Four permutation-family nulls
were built and their H0 rejection rate measured against a nominal 0.05:

| null | H0 rejection at α = 0.05 |
|---|---|
| permute the value series against the fixed step grid | 0.45 |
| permute the interval increments | 0.32 |
| circular shift of the increments, **sampled** | 0.13 |
| circular shift of the increments, **enumerated** | 0.065 |
| enumerated shift, both onsets drawn early | **0.103** |

The first three fail for one reason: **the statistic is built on a concentrated
profile and those nulls dissolve the concentration.** A permuted series' change
is scattered across every interval, so the null has far too little variance and
any partial overlap of two real profiles reads as significant. The sampled
circular shift is additionally wrong in a way worth naming — $m$ rotations are
not $m$ independent draws, so sampling 199 of them and dividing by 200
understates $p$; the same class of error as reading a discrete design's floor
off a Monte-Carlo size. And the enumerated shift, which is honest, assumes
changepoints are uniform on the interval grid. They are not: **everything moves
early in training**, and with both onsets early it rejects at twice nominal.

**The null that survives is a matched control series, and the control is
another unit's copy of the same series.** For series B at unit $u$ the control
is series B at another unit — same metric, same construction, same sweep — and
those controls are combined across units as a permutation of the **pairing**
between the two series' units. Under H0 the two series' per-unit locations are
independent, so which unit of A is paired with which unit of B is arbitrary and
the permutation is exact. It disposes of the common-trend confound for free,
because both series keep their real per-unit locations under every permutation:
the trend is held fixed rather than assumed away. Measured offline at 300
replicates and committed to
`claims/calibration/changepoint_colocation.json`:

| H0 family | 8 units | 16 units | 24 units |
|---|---|---|---|
| onsets independent | 0.040 | 0.040 | 0.060 |
| onsets both early (common trend) | 0.050 | 0.037 | 0.047 |
| **shared per-unit factor** | **0.997** | **1.000** | **1.000** |
| reversed pairing (reciprocal fires) | 0.983 | 1.000 | 1.000 |
| power, planted co-location | 1.000 | 1.000 | 1.000 |

Making it a permutation over **pairings** rather than one test per unit is also
what keeps it clear of `status-6.md`'s "n layers are not n independent
observations" — the arm is one test over all units.

**§6h's lesson arrives a second time from a different direction.** There the
attainable floor moved from 0.667 to 0.0005 on the question of what is
randomised. Here *validity* moves from 0.103 to 0.050 on the same question,
with the same data, the same estimator and the same claim. And it adds a third
kind of matched control to the two `EVALUABILITY.md` lists — a subspace, a
magnitude, and now **another series**.

**The limitation is severe, it was measured rather than described, and it is
not fixed.** The pairing null tests *association*. A common per-unit factor — a
layer that changes late changing late in *both* series, for a reason with
nothing to do with the claim — is an association, and the measured rejection
rate under exactly that is **1.00 against 0.05**. No null over the pairing
separates them: a confound present at every unit is present under every
permutation. Every record therefore carries a `shared_unit_factor_diagnostic`,
which reports each series' rank correlation with the unit index — catching a
confound *monotone* in that index and catching nothing else — and the analyst
must name the independence source, which `PREDICTIONS.md`'s Phase 7
adjudication constraint 2 already required and now has a number behind. The
honest fix is a confound-control arm against other per-unit series; it needs
the same 19 control series CLAIM-B's anchor arms need and it is not built. This
is the same shape as §6f's "prompts on one model are not independent runs":
bounded by measurement, not removed by it.

**CLAIM-B is three arms under unanimity; P-I1 is one.** CLAIM-B's statement
names two co-locations at once, so the mutual arm (energy break against Fiedler
drop, paired over layers) runs beside one anchor arm per series against the
pre-registered ~512–2000 window, and the reported p is the intersection-union
**max** — a valid p for a conjunction regardless of dependence, so no
multiplicity correction, which matters because the arms share two series
between them. Same precedent as CLAIM-C's metric-leave-one-out axis. A third
axis is affordable here in a way §6g records it is not on CLAIM-C, because
CLAIM-B carries no hard stop. P-I1 names no literature anchor, so it gets no
anchor arm — inventing one would be the glossary error Phase 7 is designed
against.

**The anchor arms are the ones that will refuse, and the arithmetic says so
before the pilot runs.** An anchor arm has no relabeling available: there is no
permutation that realises "unrelated to the literature's anchors", so it needs
a reference *population* of change locations and its floor is
$1/(n_\text{controls} + 1)$. At α = 0.05 that is **19 control series** measured
on the same sweep at the same layers. A cheap-tier sweep measuring six metrics
has six, and under unanimity a refusing arm refuses the whole gate. **That is a
requirement on what the pilot must measure, computed before it runs** — the
§6f pattern ("six prompts is the first workable gate") one level up. P-I1,
having no anchor arm, is the likelier of the two to produce a number.

**The stop rule is three-way and only one branch is a falsification**, CLAIM-C's
shape, and CLAIM-B's falsifier is why the branch exists at all: *"No
co-location. Itself a real result: it re-anchors the 1.4B schedule rather than
invalidating the sweep."* CO-LOCATES on `p_greater ≤ α`; RE-ANCHORS when the
reciprocal test rejects — the changes sit demonstrably *further* apart than the
matched controls, a separation positively shown rather than inferred from a
failure to reject; INSUFFICIENT otherwise, because an e-process records
insufficient evidence and never a null accepted. Only `p_greater` is calibrated
into a claim's E. **And the RE-ANCHORS branch was checked to be one that can
actually fire** — a deliberately anti-aligned pairing returns p = 1.000 with the
reciprocal test rejecting at 0.98–1.00 — because §6h found an audit arm
reporting PASS while incapable of failing, and a verdict nothing can trigger is
that defect wearing a different hat.

**Two plain bugs, both found by a test that asserted an exact value.** The
sampled pairing regime reported its attainable floor as $1/(n+1)$ while the
identity pairing is already in the draw list, so the reported floor was
*smaller* than any p the arm can express — the same class of slip as §6h's
default argument bound at definition time, and found the same way, by a test
asserting a perfect result lands exactly on the floor. And the anchor arm's
minimum control count is **19, not 20**: the floor must not *exceed* α, so
$1/20 = 0.05$ is admissible and rejects exactly when the observed statistic
beats every control, which is nominal rather than lucky. Both were in prose
before they were in a test.

**Still no data.** As in §§6e–6h the apparatus exists and the artifacts do not.
`INDEX.md` records the dense pilot sweep as not executed, validation is on
synthetic inputs with known answers, and `claims/adjudications/` remains empty.

## 6j. CLAIM-C run on inputs whose answer is known (2026-08-25)

Five passes built apparatus and `claims/adjudications/` was still empty. This
one adds none. It runs the gate §§6f–6g shipped on two families of input whose
correct verdict is fixed a priori, and commits what came back to
`claims/audits/claim_c_dry_run.json` (`tools/dry_run_claim_c.py`, ~5 minutes,
committed for the same reason as the other three artifacts).

**The sharp input is the self-comparison.** One model as *both* the reference
and the candidate. The contrast tables are then identical, every cell is
concordant, and the statistic is at its maximum in the full set and in all six
leave-one-out subsets at once. If the gate does not return TRANSFERS on that,
the criterion does not mean what it says — and there was a live reason to
expect it might not, since a self-comparison inherits whatever prompt-to-prompt
sign consistency the reference has, §6g's derived refusal fires above
homogeneity ≈0.80–0.85, and §6f's identical-rows refusal fires at exactly 1.0.

**The criterion is sound, and the tool axis is inert where it should be.** On a
perfect input every subset returns exactly the attainable floor
$2/(2^n + 1)$ and the intersection-union max over the seven is that same floor.
Unanimity does not bite on a unanimous input. That had never been checked; it
is now pinned at every tabulated prompt count.

**What the sweep found instead is an ADMISSIBLE BAND, outside which the gate is
a constant function.** At eight prompts, `sign_homogeneity` at or below 0.8125:

| candidate sign homogeneity | what the gate does to a PERFECT input |
|---|---|
| ≤ 0.8125 | TRANSFERS, p = max(2/257, R(h, 2/257)) |
| 0.8333 – 0.9583 | refused: corrected attainable floor |
| 0.9792 | refused: the curve's top bin has no measurement |
| 1.0000 | refused: identical sign rows |

0.8125 is **at least 9 of the 48 candidate cells carrying the minority sign for
their metric** — on average at least 1.5 of the 8 prompts dissenting on each of
the six metrics. Above it the power curve confirms refusal at *every*
concordance count from 0 to 48, so neither TRANSFERS nor FAILS-TO-TRANSFER is
reachable and **the hard stop fires unconditionally**. §6g's own caution against
a third robustness axis was that *"a stop rule that always fires carries no
information"*; this is where CLAIM-C's stop rule has such a region, and it was
found by running the gate rather than by reading it.

**It is not a Type-I defect and not an argument for a weaker correction.**
`sign_homogeneity` is a *within-candidate* statistic. Under H0 it measures the
prompt redundancy §6g measured and corrected for; under H1 the same number also
rises with the strength and *uniformity* of a real effect, and the correction
cannot tell the two apart. The cost therefore lands as power, and it lands
hardest where the effect is most uniform. **The gate is powered against a
contrast carrying a prompt-specific signature that transfers, and unpowered
against a contrast carrying one uniform direction that transfers.** Blog 1's
phenomenology is the second kind.

**Two references fix the scale, and both were computed rather than guessed.**
Under *independent* prompt signs — the most favourable candidate the design can
be handed — homogeneity concentrates at **0.637** at eight prompts (exact, by
convolution over the binomial majority counts) and the refusal fires with
probability **1e-4**. So the band is not tight against chance. It is tight
against a clean effect: a contrast pointing the same way on every prompt sits at
exactly 1.0 and is refused with certainty. And more prompts do not help.
Expressed as the curve bin the refusal starts in — the unit comparable across
prompt counts, since the attainable homogeneities themselves lie on a grid of
step $1/(n\,m)$ — the boundary is 0.800–0.825 at six prompts, 0.850–0.875 at
seven and nine, and 0.825–0.850 at eight, ten, eleven and twelve. Three bins of
0.025, with no trend.

**So this is a requirement on what the pilot must measure, computed before it
runs** — the §6i pattern a second time, where CLAIM-B's anchor arms turned out
to need 19 control series a six-metric sweep does not have. Here: at least ~19%
of the candidate's 48 cells must dissent in sign. Whether they do is an
empirical fact about pythia-1.4b that nothing in this repository yet knows, and
it is now on the record as a precondition rather than as a surprise waiting at
the far end of a sweep.

**One positive result, and it answers §6h's question asked of a refusal.** §6h
found an audit arm reporting PASS while incapable of failing. The dual question
for a refusal is whether it ever refuses something that would have passed.
R(h, ·) is non-decreasing in p in all 264 tabulated bins, so
`R(h, floor) > α` implies `R(h, p) > α` for every attainable p: **whenever the
derived refusal fires, no input whatsoever could have cleared α.** It is tight.
It never costs a verdict the gate could otherwise have reached; it converts an
uninformative "p > α" into a refusal that says why.

**The power curve, which was the second queued question.** At eight prompts and
48 cells, over randomly placed arrangements at a fixed candidate sign table:

| homogeneity | TRANSFERS ≥50% | TRANSFERS always | FAILS ≥50% | INSUFFICIENT band |
|---|---|---|---|---|
| 0.625 | 35/48 | 38/48 | ≤13/48 | 14–34 |
| 0.750 | 37/48 | 42/48 | ≤10/48 | 11–36 |
| 0.8125 | 39/48 | 44/48 | ≤10/48 | 11–38 |

The queued form of the question was whether the answer is ~44, which would make
the gate nearly all-or-nothing. It is 44 *at the boundary* and 38 well inside
it, so the gate is demanding rather than binary — and the requirement tightens
as the candidate's contrast becomes more uniform, which is the same finding
arriving from the other direction. The INSUFFICIENT band is 26 of the 49
possible concordance counts at homogeneity 0.75, and the hard stop fires across
all of it.

**Decomposed, because "the gate is demanding" is not a finding until you know
which part of it is demanding.** Two counterfactual rates are recorded beside
each row: the full-set-only p and the uncorrected intersection-union max. The
metric-leave-one-out axis moves the 50% point by **1–3 cells**; the homogeneity
correction moves it by **0 at homogeneity 0.625, rising to 5 at 0.8125**. Most
of the cost is the correction, and it is concentrated at the top of the band.

**What this deliberately did not do.** It changes nothing in
`p1_mstate_tracking/replication_gate.py`. It adds no third robustness axis —
§6g records why CLAIM-C in particular cannot afford one, and the region found
here is a concrete instance of that argument rather than a reason to add
another. And it adjudicates nothing: a dry run on synthetic inputs is not
evidence about pythia-1.4b, `claims/adjudications/` is still empty, and the
record says so in a field the staleness check reads.

**The lesson, which is the one the last three passes kept paying for.** §6g's
rounding defect, §6h's audit arm incapable of failing and §6i's power figures
measured under a discarded null were all found by *looking at an output*, never
by a failing test. This pass looked at an output nobody had generated: the
gate's verdict on an input whose answer was already known. Neither of the two
things it found — the admissible band and the tightness of the refusal — is
visible in any single result, and no synthetic unit test in the suite was
failing.

## 7. What this plan does *not* do

- It does not run any science. No chunk here adjudicates a prediction; B6 makes adjudication
  *possible*, and the first real adjudication is a separate decision gated on
  `tools/preflight_1c.py` per `UPDATE_PLAN.md` §3.
- It does not unfreeze Phase 2c/3/4. C2's SAE prediction is registered and left unrun.
- It does not rename phase directories (§C3).
- It does not add an LLM to the loop. POPPER's design and relevance agents are replaced by the
  human pre-registration the project already practises (§3).
