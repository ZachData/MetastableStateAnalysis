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
a constant function.** At eight prompts, `sign_homogeneity` at or below 0.8125
— *superseded 2026-08-25: the band is now ≤ 0.7708, see §6l. The finding stands;
the number moved when the informative-row floor changed what "conditional on
emission" conditions on.* The table below is the measurement as it stood:

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
of the candidate's 48 cells must dissent in sign (**revised to 11 of 48, ~23%,
in §6l**). Whether they do is an
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

## 6k. P-ST1: the entry that can lose, and four things the wording left open (2026-08-25)

`P-ST1` is H-BRIDGE's cheapest entry and the only registered bridge prediction
where the particle and standard accounts make **incompatible** rather than
merely different predictions — which is what makes it the only one that can
genuinely lose. It now has a construction: `p7_motifs/steering_gate.py`, with
`tools/calibrate_steering_sign.py` → `claims/calibration/steering_sign.json`
behind it. Nine predictions are adjudicable in principle;
`claims/adjudications/` is still empty.

It is unlike every other entry in one way that made this pass possible. The
whole intervention is exact linear algebra — add `α·v` to every row of an
activation block and recompute effective rank — so the statistic can be
exercised **end to end on populations with a planted answer, with no model at
all**. Every number below was measured rather than argued, and four of them
changed a decision.

**Two scientific calls were put to the author before any code existed**, the
§6i pattern, and both were answered. A third and fourth were settled by
measurement without needing to be asked.

**Steering is a pure mean effect, and that is algebra rather than simulation.**
Adding `α·v` to every token is exactly a shift of the population mean, so
`ER(X′ − mean X′) = ER(X − mean X)` identically: a pipeline that re-centres
after injection measures nothing at all. The consequence the registered wording
could not have anticipated is that the cloud's **pre-existing** mean offset
competes directly with the injected one. Per-pair `P(D = +2)`, each
configuration at its own best α:

| cloud mean norm | H1 | H0 |
|---|---|---|
| 0 × spread | 0.970 | 0.010 |
| 2 × spread | 0.230 | 0.180 |
| 5 × spread | 0.110 | **0.250** |

A real residual stream sits at the bottom of that table — at five spreads the
design rejects *more often under H0 than under H1*. Removing the **baseline**
mean before injecting, keeping the injected offset, restores it to 1.000
against 0.000 at every offset. That was the first call to the author; its cost
is that the criterion becomes about the injected direction relative to the
*centred* population, which is narrower than the words "the effective rank of
the token population".

**α was a third placed constant the registry never flagged.** It flags
`n_pairs` and the "predominantly" threshold. Measured, α decides whether the
prediction is readable at all: below ~0.05 × spread both arms move the same way
in every pair and the statistic is *identically zero*; above ~0.26 the rank-1
spike `nα²vvᵀ` dominates the Gram matrix and both arms fall for any direction;
in between is a **plateau at 0.17–0.24** where the per-pair rate is 1.000 under
H1 and 0.000 under H0. The second call to the author chose one pre-registered
α labelled `placed` per Phase 7 adjudication constraint 4. What is *not* placed
is the scale it multiplies — the population's own RMS deviation from its mean —
and that is what buys the plateau being **the same four fractions** at mean
offsets of 0, 2 and 5 spreads.

**The first value written into the module was 0.1, and it was wrong.** The
plateau was missed on a grid of (0.03, 0.1, 0.3), which reads 0.1 as the peak
because both its neighbours are zero. On a finer grid 0.1 sits on the shoulder
at a sixth of the plateau's rate. §6g's rounding defect and §6i's discarded-null
power figures were both found by looking at a generated table; this is the third,
and the only thing that found it was printing a finer sweep.

**"Predominantly" was removed rather than thresholded.** The registry asks for
vectors "predominantly" in one subspace — a magnitude cut with as many values
as there are cuts. Each arm is instead drawn *uniformly from the subspace
itself*: 100% by construction. That is CLAIM-C's sign-concordance and CLAIM-B's
change-mass centroid escape a third time, and it retires one of the two
constants the registry flagged.

**`ER_MODE` is `raw`, against the CLAIM-C precedent, and the reason is
structural.** `status-1.md` D1 is why CLAIM-C reads `effective_rank_normed`,
and at the working α the two modes are indistinguishable — which is exactly how
the wrong one nearly shipped. Away from that point they are not
interchangeable. With the baseline mean removed the centred population has zero
mean, so the first-order Gram term `α(baseᵀ1vᵀ + v1ᵀbase)` **vanishes
identically**, `dER` is O(α²) and even in `v`. L2 row-normalization is not
linear and puts an odd term back: `normed` agrees with itself under `v → −v` in
0 of 60 draws at small α and *manufactures* `D = −2` in 20–22% of pairs there,
where `raw` gives 0%. A criterion that answers differently for `v` and `−v` is
not a criterion about a steering **direction**.

**The registered null does not hold, and the failure grows with the pair
count.** "The same injection procedure with the decomposition label permuted
across pairs" treats *m* pairs as *m* exchangeable units. Every pair at one
layer sees the same tokens and the same two subspaces, so a chance tilt of the
cloud moves them together; more pairs shrink the permutation null's spread like
√m and leave the tilt untouched. Rejection rate under a noisy H0 at α = 0.05,
**conditional on the gate emitting**:

| pairs | 8 | 24 | 40 | 150 |
|---|---|---|---|---|
| weakly concentrated H0 | 0.000 | 0.031 | 0.030 | **0.220** |
| slightly concentrated H0 | 0.000 | 0.012 | 0.082 | **0.170** |

That is `status-6.md`'s "49 layers are not 49 independent observations" for the
third time. It is invisible in the clean regime, where under H0 every pair is
uninformative and the gate *refuses*: the unconditional rate reads 0.000, and
it is 0.000 by refusal rather than by control. **Conditioning on emission is
the only thing that makes it visible**, which is §6g's lesson exactly, and it
is the second registered null in three passes that measurement retired.

**What replaces it is §6h's construction, arriving for the fourth time:
randomise over subspaces, not over units.** H0 here is "the sign is independent
of the U_pos/U_neg decomposition", realised directly by replacing the two
operator-derived subspaces with **random ones of the same dimensions**, at the
same layer, on the same population. Every chance tilt is present in the null
exactly as in the observed value, so the confound the permutation cannot see is
what the null is made of. The pair is drawn *mutually orthogonal* from one
Stiefel draw, because the real pair is orthogonal by the projector build's
resolution order and §6h measured the cost of forgetting that at 0.0875 against
a nominal 0.05. Measured: 0.000–0.040 under H0, power 1.000 in both directions
at 8 and 24 pairs. The registered permutation is still computed and reported
beside every result, never adjudicated, so the difference is visible in the
record rather than asserted in a docstring.

**Replacing the null also removed a floor the registered design could not
meet.** Under the permutation, a pair whose two arms move the same way
contributes `D = 0`, and a zero contributes identically to the observed sum and
to every null pattern — so with *k* of *m* pairs informative the best
attainable p is `(2^(m−k) + 1)/(2^m + 1) ≈ 2^−k`, set by the **informative**
pairs and not the drawn ones. Five is the first *k* that clears α = 0.05, at
every *m*, so a hundred pairs at a 2% informative rate buy two informative
pairs and a best possible p of 0.25. The subspace null's floor is
`1/(draws + 1)`, fixed by how many draws are taken and independent of the data:
a single informative pair can reject, and correctly so, since if random
subspaces of the same dimensions essentially never inform then an
operator-derived pair that does is exactly the surprise the claim is about.
§6h's "the binding constraint was the choice of null, not the choice of unit"
holds here too — this time it moved a power requirement rather than a floor.

**The precondition that remains, and it is a requirement on the pilot.** A
uniform draw from `U_pos` carries only `dim(occupied)/dim(U_pos)` of its energy
into the subspace the cloud lives in. Per-pair informative rate against that
ratio: 1.000, 0.710, 0.320, 0.030, 0.005, 0.000 at ratios 1, 1.5, 2, 3, 4, 6.
§6h already measured that `U_pos` is the **un-shrunk** bucket in the projector
build's resolution order, which is the unfavourable side. So the pilot must
report `dim U_pos` at the injection layer against the population's effective
rank — the third pre-computed requirement in three passes, after CLAIM-B's 19
control series and CLAIM-C's 19% dissenting cells. The obvious fix is refused:
drawing from the intersection of `U_pos` with the occupied subspace would
restore the rate and is circular, since a probe aligned with the cloud by
construction concentrates it by construction.

**The registered falsifier is not one an e-process can carry**, and that is
recorded here rather than discovered when it binds. *"Both arms move effective
rank the same way, or the effect tracks ‖s‖ and is insensitive to the
decomposition"* — both clauses describe the **null**, and an e-process records
insufficient evidence and never a null accepted. They map to INSUFFICIENT. The
falsification branch is **INVERTS**: attractive-dominant steering demonstrably
*raising* effective rank while repulsive-dominant lowers it, a reversal
positively shown. §6i's requirement that such a branch be checked to be one
that can actually fire is met — it fires at 1.000 under a planted inversion.
The falsifier's second clause is designed out rather than tested: at matched
norm ‖s‖ cannot vary *within* a pair, so the norm dependence lives in the
α-profile, which every record carries and no p-value reads.

**What is still weakly measured, stated rather than left to be found.** The
reciprocal tail — the INVERTS branch, the one that would enter the ledger — is
measured under H0 at 0.02–0.10 over fifty gate runs per cell. Fifty runs
resolve a rate to about ±0.03, so that is consistent with nominal and it is not
a tight bound; the adjudicated `greater` tail is 0.000–0.040 across the same
cells. Before anything is adjudicated on INVERTS specifically, that cell wants
more replicates than a committed artifact can afford to carry.

**Still no data.** As in §§6e–6j, the apparatus exists and the artifacts do
not: the gate needs activations and the Phase 2 attractive/repulsive
projectors, and neither is in this repository. Validation is on synthetic
populations with known answers, and `claims/adjudications/` remains empty.

## 6l. CLAIM-C's cell-drop dimension, and the floor that was never tight (2026-08-25)

§6g closed with a gap it named as the honest fix and did not build: every draw
in the homogeneity curve had a **complete** (prompt × metric) table, so a real
run that drops cells reads its correction — and therefore its refusal — off a
table measured on a design it does not have. §6j promoted that gap by showing
the correction is what drives the refusal boundary. It is now built, and
building it turned up a second thing that had been live in the gate since the
day it was written.

**The second thing came first, because it is exact and needs no simulation.**
Flipping prompt *i*'s condition label swaps its concordant and discordant cells,
so row *i* contributes `conc_i` unflipped and `valid_i − conc_i` flipped and its
**swing** is `|valid_i − 2 conc_i|`. A row with swing 0 contributes the *same*
number to the observed sum and to every one of the 2^n null patterns: it is
enumerated and never counted. With *k* rows that do move,

$$\text{floor} = \frac{2^{\,n-k}+1}{2^{\,n}+1} \;\approx\; 2^{-k}$$

which is `2/(2^n + 1)` exactly when *k* = *n* — so the floor the module already
refused on is the special case, not a different rule. Checked against the gate's
own enumeration at every *k*, it agrees to the digit.

**Five informative rows is the first count that clears α = 0.05, at every
tabulated prompt count.** That is `P-ST1`'s informative-*pair* floor (§6k)
arriving in CLAIM-C from the other direction, down to the same k ≥ 5, which is
now the third construction where the binding quantity turned out to be the units
that carry information rather than the units that were run.

| informative rows *k* | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| floor at 8 prompts | 0.128 | 0.066 | **0.035** | 0.020 | 0.012 | 0.008 |

**Two ways a row lands there, and the one nobody was looking for is on a
perfectly complete table.** All its cells dropped is the obvious one. The other
is an **even** number of usable cells splitting exactly half and half: with the
full six metrics, a prompt concordant on three and discordant on three. Under H0
that is 20/64 of rows. So the gate could be handed a table that was *perfect on
four prompts* and 3–3 on two, return p = 0.0769 — precisely that table's own
floor — and report it as "not significant". Measured at six prompts, **61% of H0
draws** could not have rejected however the statistic fell; at eight prompts 22%,
at twelve 1.2%.

It does **not** bite on the six leave-one-out subsets of a complete table: five
metrics is an odd count and an odd swing cannot be zero. The full set is the
binding subset until drops make a subset's usable count even or empty it — which
is the join between the two halves of this pass.

**Why a data-dependent refusal is safe, and why that had to be measured rather
than argued.** The null is symmetric under a global flip, so both tails share
the floor: when it exceeds α neither `p_greater` nor `p_reciprocal` can reach α,
TRANSFERS and FAILS-TO-TRANSFER are both unreachable, and the verdict was
INSUFFICIENT whatever the statistic came to. The refusal therefore removes no
verdict; it replaces a p above α — which on this claim reads as evidence
*against* CLAIM-C — with a record saying the design could not have rejected.
Measured against the counterfactual at five H1 strengths, P(TRANSFERS) is
identical to four decimals with the refusal and without it, including at a
strength where the refusal fires on 15% of draws. Zero power, exactly. That is
§6j's tightness result asked of the refusal that was missing, and the dry run
now re-scores every table it refuses rather than restating the argument.

**And the cost of stating it that way: `costs_no_power` is `None`, never
`True`, when the refusal never fired.** A sweep with nothing to re-score would
report success while being incapable of reporting anything else — §6h's audit
arm again, caught this time before it shipped rather than after.

### The drop dimension

**Dropping cells is not the same statistic made noisier**, which is why it
needed a dimension rather than a caveat. It changes three things at once: the
sum runs over fewer cells, the per-row null weights stop being equal, and a row
can lose its swing outright.

**The tabulation was affordable only after re-keying the exact fast path.** §6g's
table was keyed by the histogram of per-row *concordant counts*, which assumes
every row has all *m* cells. Once `valid_i` varies, the obvious key is the pair
`(valid_i, conc_i)` — 28 types at six metrics, and multisets of eight rows over
28 types is **23 million** keys. The null, though, is exactly the distribution of
`Σ ±g_i`: it depends only on the multiset of **swings**, which has *m* + 1 types,
so the key count is C(*n* + *m*, *m*) = **3003** at eight prompts — the same size
as the complete-table version. Dropping cells costs the tabulation nothing. The
observation `Σ e_i` is not determined by the swing multiset, so each key stores
the whole null distribution and the observation indexes into it. Pinned against
the gate cell by cell on holed tables, not only on the complete ones it reduces
to.

**The curve is now indexed by (n_prompts, drop bin, homogeneity bin).** Drop bin
0 is `n_cells_dropped == 0` *exactly*, tested as integers — a complete table is
the design the gate was built on and the common case, and deciding that boundary
by comparing a float against an epsilon is how §6g's rounding defect got in.
Above the last tabulated edge the gate **refuses** rather than reading the
nearest row.

**Three drop mechanisms reach each rate**, for the same reason the bias family
has three shapes, and each (h, d) cell keeps the worst configuration over bias
shapes and mechanisms together. The first version pinned the concentrated
mechanisms to a single line and it was wrong: one metric column is 1/6 of the
table and one prompt row is 1/*n* of it, so both saturated and left the upper
drop bins measured by the benign mechanism alone — "worst configuration" over
one candidate, exactly where the design is most stressed. They now spread over
as few lines as the rate allows and no fewer.

**Nothing is filled across the drop dimension, and that is a measurement.**
Coarsening pushes p-values up; selecting for the tables that survived the
informative-row floor pushes the conditional rate down. The two point opposite
ways and they do not resolve — at every tabulated prompt count, of the ~118
adjacent drop-bin pairs at a fixed homogeneity, the overwhelming majority are
neither non-decreasing nor non-increasing:

| prompts | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|
| rises | 12 | 9 | 5 | 2 | 1 | 0 | 0 |
| falls | 7 | 8 | 19 | 5 | 1 | 2 | 2 |
| **neither** | **93** | **100** | **93** | **110** | **116** | **116** | **116** |

The small prompt counts even carry pairs going *both* ways at once — 12 rising
and 7 falling at six prompts — which is the strongest possible form of the
answer: there is no direction to fill in.

So a hole in that dimension is a refusal, and `drop_monotone_in_d` is stored per
prompt count so a later reader does not take the fill rule on trust.

**The draw count stopped being one number, and the reason is the refusal above.**
Every rate here is conditional on the gate emitting, so what a bin needs is a
fixed number of *emitted* draws rather than of drawn ones — and the
informative-row floor refuses far more often at small prompt counts: measured,
six prompts emit on 39% of independent-row H0 draws, eight on 78%, twelve on
99%. Drawing 40000 everywhere therefore measures six prompts to a coarser
resolution than twelve, and it showed: the first generation of this curve left
the six-prompt **(0, 5%] drop slab with no measured bin at all**, so the gate
refused there outright whatever the data was, and the other five six-prompt
slabs stood on one to three measured bins apiece. Nothing failed; it was found
by printing the coverage. The draw count is now scaled by `1/P` with *P* in
closed form, rounded and capped — 3× at six prompts, 2× at seven, unchanged from
eight up — which closes it: every slab at every prompt count now carries 12 to 18
measured bins of 20. Every curve also carries a `coverage` block naming any slab
with no measurement, so the next hole is visible in the artifact rather than
discovered by a run being turned away.

That is the fifth session running in which looking at a generated table found
something no test was failing on, and the second in which the thing found was a
consequence of the same session's own change.

**What the family assumes, stated because a family is only as good as its
boundary.** Drops are independent of concordance *given the position*: which
cells go is modelled, whether a surviving cell agrees is not conditioned on it. A
mechanism that preferentially removes discordant cells is outside it and is not
corrected by this curve.

### The band moved, and that is this pass's real cost

§6j measured CLAIM-C's admissible band at **≤ 0.8125** at eight prompts, and
restated it as a precondition on the pilot: at least 9 of the candidate's 48
cells must dissent in sign. Both numbers are now superseded. Re-measured on the
regenerated curve, the derived refusal starts in the **0.775–0.800** bin at six,
seven and eight prompts and 0.825–0.850 at nine and up, and the requirement is
**11 of 48 cells**, about 23%.

**The cause is the informative-row floor, and it is not a regeneration
artefact.** Every rate in the curve is conditional on the gate emitting. Before,
a draw whose rows could not move the statistic was emitted with a p above α and
counted in the denominator as a non-rejection, diluting *R* downward. It is now
refused, so it reaches no ledger and belongs in no denominator — and the rate
among the draws that do reach a ledger is higher. A higher *R* is a stronger
correction, and a stronger correction refuses at a lower homogeneity. Both
curves were internally consistent; each matches the gate it was measured on, and
only the new one matches the gate that exists.

So the refusal that costs no power (above) does cost *band*, and the two are not
in tension: it removes no verdict from any individual table, and it tightens the
region in which the gate can reach a verdict at all. That is worth stating
plainly because §6g's caution — each thing added moves probability mass into
INSUFFICIENT, and a stop rule that always fires carries no information — applies
to a refusal exactly as it applies to a robustness axis, and this pass added a
refusal.

The rest of §6j survives it. The criterion is still sound on a perfect input;
`R(h, ·)` is still non-decreasing in *p*, now over **1914** tabulated bins
rather than 264, so the derived refusal is still tight across the drop dimension
as well; the boundary still does not move with prompt count (two bins of 0.025,
every count moving down together); and an independent-prompt candidate still
sits comfortably inside — the share of that distribution above the boundary is
0.0017 at eight prompts, against 8e-5 before. At six prompts it is 0.026, which
is the clearest statement yet that six prompts is the marginal design.

**What the pilot must now produce**, replacing §6j's line: at least **11 of 48**
cells dissenting in sign, and at least five prompts whose usable metrics do not
split evenly.

**What it cost.** The configuration count went from 35 per prompt count to 770,
and — with the draw scaling above — generation from about 50 seconds to about 40
minutes, with the artifact growing from roughly 0.4 MiB to roughly 3 MiB.
`docs/CI_BASELINE.md` records it, because a tool that used to be worth running
while waiting is not any more.

**Still no data.** As in §§6e–6k, the apparatus exists and the artifacts do not.
`claims/adjudications/` is empty for CLAIM-C and `null_construction` has still
not frozen.

### And the first refusal lifted by a decision rather than by apparatus

`P6-R2` and `P6-R4` refused to adjudicate while `REGISTERED_EXCHANGEABLE_UNIT`
was `None`. §6h recorded why it was left there: which unit may enter an
e-process is a scientific decision of the same class as CLAIM-C's criterion, not
a measurement, and taking it after seeing a p-value would void the guarantee.
It was put to the author in the previous pass, who chose to build `P-ST1`
instead; it was put again and **the author registered `"model"`**.

It was safe to take now for the reason that made it worth asking early: no
p-value on real activations exists — `claims/adjudications/` is empty, no run
artifact is in the repository, and every number either unit has produced came
from synthetic populations. What it was decided *against* is on the record
rather than on anyone's authority: measured at 400 replicates, as the layers
come to share one direction, `"layer"` runs 0.0525 → 0.0800 → 0.2325 → 0.2800
while `"model"` holds at 0.045–0.0575 throughout. Under ALBERT's weight-tying
the layers are as far from independent as they get — one OV matrix, one Schur
decomposition, one projector pair, 49 activation snapshots of it — so `"model"`
is the conservative choice at every point of the range and not a trade: at ρ = 0,
where the two agree, it costs nothing.

**These are the first two entries whose refusal is lifted by a recorded decision
rather than by new apparatus**, which is a category this project had not used
yet. Nothing is adjudicated: there is still no run artifact, and the live
refusal simply moved — `adjudicate_p6_r2_r4` now turns away a result computed
under `"layer"` instead of turning away everything.

**One consequence worth not rediscovering.** While no unit was registered, that
refusal was doubling as the safety catch keeping a synthetic p-value out of
P6-R2's ledger slot: `adjudicate_p6_r2_r4(res, adjudicate=True)` could not reach
`core.adjudication` at all. It can now, `P6-R2` is classified `e-value`, and
`adjudicate` refuses to overwrite a record once written — so an accidental
fixture run would occupy the slot permanently. Every test that both asks to
adjudicate and uses the registered unit passes an isolated `adjudications_dir`,
and the test class says so at the top.

## 6m. P-ST1 run on inputs whose answer is known, and the null that did not hold (2026-08-26)

`claims/EVALUABILITY.md` closed §6j's section on a queue: seven adjudicable rows
had been validated by unit tests and none had been run on an input whose correct
verdict is fixed a priori, so *"the queue that used to read 'convert the next
needs-null row' now has a second entry ahead of it for each row already
converted."* This pass does that for `P-ST1` —
`tools/dry_run_p_st1.py` → `claims/audits/p_st1_dry_run.json`, about ten
minutes, committed for the same reason as the other five artifacts.

`P-ST1` was the one to do next for the reason it was built early: it can
genuinely lose, and its whole intervention is exact linear algebra, so the gate
runs end to end on populations with a planted answer and no model at all. §6k
applied the dry-run discipline at *construction* time rather than after, which
is better and which made this a real check rather than a formality. The question
was whether anything survives being run on inputs the construction did not
already have in mind.

Two things did not, and the first was found before the dry run was finished
being written.

### The adjudicated null is invalid, and the family that shows it is the realistic one

§6k replaced the registered label permutation with §6h's construction arriving
for the fourth time: randomise over **subspaces**, not over units — replace the
two operator-derived subspaces with random ones **of the same dimensions**,
drawn mutually orthogonal from one Stiefel draw. Matching the dimensions holds
fixed everything the statistic could read off dimension.

It does not hold fixed how much of the population each subspace **contains**,
and that is what `dER` is driven by. Injecting along a direction the cloud
already occupies reinforces a large Gram eigenvalue and lowers effective rank;
injecting along one it does not adds a new eigenvalue and raises it. A random
*k*-dimensional subspace captures *k/d* of the population's energy in
expectation — and `U_pos` and `U_neg` are cut from the model's own OV
eigenstructure, on a residual stream orthogonal to neither, so both capture
more. Against random pairs, such a pair is unusual **whichever arm is called
attractive**, and the sign of the observed difference is then whichever way the
layer's realized asymmetry happens to fall.

Measured on an H0 family in which both arms are occupied above chance and the
two are *identical by construction* — so a label swap is a distributional
identity, the correct verdict is INSUFFICIENT, and P(TRACKS) must equal
P(INVERTS) exactly — against a nominal 0.05, on the same runs and the same
drawn pairs so the comparison is paired:

| family | pairs | occupancy of each arm | retired null | adjudicated null |
|---|---|---|---|---|
| H0-both-arms(weak) | 8 | 1.27 | 0.080 | 0.020 |
| H0-both-arms(weak) | 24 | 1.27 | **0.200** | 0.020 |
| H0-both-arms(strong) | 8 | 2.24 | 0.120 | 0.040 |
| H0-both-arms(strong) | 24 | 2.25 | 0.160 | 0.000 |

**The inflation grows with the pair count**, which is the signature of the
mechanism rather than a coincidence: more pairs tighten the null's spread
without touching the union's unusualness, so what is left of the statistic is
the realized asymmetry between two subspaces that are both special. It is the
same shape §6k measured for the *registered* permutation and did not think to
ask of its own replacement.

**§6k's calibration could not see it, and the reason is the transferable
lesson.** All three of its H0 families put the cloud in a subspace **orthogonal
to both arms**, which leaves both at chance occupancy — precisely the one case
in which a matched-dimension random pair *is* exchangeable with the observed
pair. The families a calibration measures are part of the measurement, and their
absence is invisible: a calibration whose H0 families cannot express the failure
it is meant to rule out is §6h's audit arm incapable of failing, one level up.
`check_record()` now fails if no such family is present — and fails again if the
retired null does **not** come back anticonservative, since an artifact that no
longer supports a retirement is a problem with the retirement rather than
something to pass over.

### What replaces it randomises the split, not the subspaces

The diagnosis names the fix. The old null moved the union and the split
**together**, so it rejected on either — and "this pair of subspaces holds more
of the cloud than a random pair would" is a statement about the *union*, which
is not what `P-ST1` claims. The claim is about the **labelled split**: does
calling one of the two attractive predict which way effective rank moves?

So hold the union fixed and randomise only the split. The null draws a uniformly
random *k*<sub>pos</sub>-dimensional subspace of span(`U_pos` + `U_neg`) and
takes its orthogonal complement **within that union** as the other arm.
Dimensions, orthogonality, occupancy and the whole spectral relationship to the
layer's cloud are held exactly fixed, and the observed split is one point of the
same Grassmannian the null draws from — so **exchangeability under H0 is by
construction rather than by measurement**, which no other null in this project
can say.

**This is §6h's question — what is being randomised? — arriving for the fifth
time, and the first time the answer is to randomise LESS.** §6h moved `P6-R2`
from units to subspaces because units were too coarse; here subspaces are too
coarse, and the exchangeable object is the assignment.

It costs power, and the cost is stated rather than hidden. Where the cloud fills
the whole arm — dim `U_pos` = dim(occupied), the calibration's H1 and INVERTED
families — both nulls reach 1.000 in both directions and the change is free. The
cost appears as that ratio grows, and the dry run's band below is the whole-gate
version of it. Power lost that way was never power about the decomposition — it
was the union's unusualness being read as the split's. **Both retired nulls are
computed and reported beside every result**, never adjudicated: the registered
permutation, and §6k's matched-dimension pair. Two retired nulls in one record
is not clutter; it is the only way a reader sees the size of the difference
between a null that was believed and the one that holds, and that difference has
now been large twice.

**And a check the module had never made.** The two arms' orthogonality was
*assumed* from the projector build's resolution order from the day the module
was written. A caller passing overlapping arms used to get a null quietly drawn
on a geometry the observed pair does not have; the union's rank is now checked,
and the refusal covers "the arms overlap" and "their dimensions exceed d_model"
as the same fact about the data.

### Each arm's occupancy is now reported, and it costs no injection

`occupancy_pos` and `occupancy_neg` are the share of the centred population's
energy inside each arm, divided by the *k/d* a random subspace of that dimension
would hold. That is §6h's `E[||P_U v||² ] = k/d` with the population in place of
a single vector, and §6h's whole finding was a comparison read without that
normalization. They need no injection and no null, so a pilot can read them off
the activations and the two projectors **before** spending a sweep, and a reader
deciding whether a TRACKS verdict has a non-particle explanation can look at the
quantity the verdict is made of rather than infer it.

The dry run asks how far that goes, and the answer is *informative, not
determinative*. Over 450 whole-gate runs spanning the sweep, the probability
that a TRACKS run has a larger occupancy log-ratio than an INSUFFICIENT one is
**0.875** — well clear of the 0.5 that would mean no information, and well
short of the 1.0 that would mean the gate is an expensive way to compute a
ratio. The two distributions overlap on both sides: the smallest log-ratio that
ever tracked is −0.034, which is a Type-I event on a symmetric input, and the
largest that came back INSUFFICIENT is 1.91, which is a strongly asymmetric pair
the design had no power on. So the diagnostic is worth reading before a sweep
and is not a substitute for running the gate — which is the honest version of
both things it was added for.

### What the dry run itself found: a reported floor that was not attainable

The gate reported `1/(draws + 1)` as the smallest p it could express — a floor
fixed by the draws and independent of the data, which is what made §6k's "a
single informative pair can reject" true. `sum(D)` cannot exceed 2*m*, so the
smallest p a run can actually express is what an observation of 2*m* would
receive, and **every null re-split that already reaches 2*m* ties it**. On a
union the cloud occupies, re-splits inform often. Measured on a perfect input at
one pair with 99 draws, the attainable floors are **0.11–0.17** in both
directions where `1/(draws + 1)` says 0.01.

Until this pass the gate would report one of those as "not significant" — a design
that could not have rejected returning a number that, on an entry whose whole
value is that it can lose, reads as a loss. That is §6l's defect for `CLAIM-C`
arriving here from the other side, and §6i's optimistically-reported floor for
`CLAIM-B`'s sampled pairing regime arriving for the second time. It had been
live since the module was written and no test was failing on it.

The gate now computes both tails' attainable floors from the null it already has
— which costs nothing — and **refuses when neither can reach α**. 2*m* is an
upper bound on the observation rather than an attainable value, so the floor
computed at it is a *lower* bound on what the run can express: the refusal can
never turn away a result that would have cleared α. When only **one** tail is
out of reach the gate does not refuse, because one reachable tail is one
reachable verdict and §6l's rule is that a refusal costs none.

**Unlike §6l's, this one costs no power by construction rather than by
measurement, and the difference is worth naming.** CLAIM-C's informative-row
refusal had to be re-scored against a counterfactual — every refused table put
back through the gate's own subset arithmetic — because the floor it refuses on
and the p it would otherwise report are computed by different code. Here they
are the same quantity: the refusal condition *is* "the best p this run could
express exceeds α". Re-scoring it would be tautological, so the argument is the
2*m* bound and not a measured zero, and this record says which of the two it is
rather than reporting a count that could not have come out any other way.

**Which surfaced a category this project had not named.** The two tails' floors
are computed separately and they are not equal, so a run can have exactly one
reachable tail — and when the reachable one is INVERTS, the design can return a
**falsification or nothing** and nothing else. On a cloud planted in `U_pos`,
few re-splits reach −2*m* while many reach +2*m*, which is the direction that
produces it; `TestTheSubspaceNulls` pins a one-pair case where the attainable
floors are 0.175 in the predicted direction and 0.025 in the reciprocal one, and
the gate correctly emits rather than refusing.

At the dry run's own geometry the one-pair cells fall the other side of that
line — both tails out of reach, so all six refuse — which is why the category is
recorded here from the arithmetic and the unit test rather than from the
artifact. Every record now carries `reachable_tails`, because a run whose only
reachable verdict is the one that enters the ledger as a falsification is a run
a reader has to be told about, and nothing else in the record would say it.

### What the four other arms came back with

**The sharp input.** Cloud planted entirely in one arm at dim `U_pos` =
dim(occupied), so every draw lands in it and the statistic is at its maximum:
the correct verdict is TRACKS-DECOMPOSITION a priori, and INVERTS with the arms
mirrored. At two pairs and above the gate returns the planted verdict in every
cell of both directions and every emitted p lands exactly on that run's own
**attainable** floor — never on the draw-count floor at two pairs, where the
attainable floors measure 0.02–0.05 against `1/(draws + 1)` = 0.01. At one pair
it refuses, in both directions, in all six cells: the attainable floors there
are 0.11–0.17, and 99 draws cannot help because the ties are the layer's.

**So two pairs is the smallest design that emits at all**, and the binding
quantity is how often a re-split of the *same union* reaches the maximum rather
than how many null draws were taken. That is the §6f pattern — "six prompts is
the first workable gate" — arriving for `P-ST1`, and it is a different number
from §6k's five *informative* pairs, which was the retired permutation's floor.

**And a small thing the arm's own staleness check caught, which is worth keeping
because of what it corrects.** The first version of this arm asserted that a
perfect input puts the statistic at its maximum, and one row failed: at sixteen
pairs, one drawn direction in forty-eight did not inform. The *planting* is
perfect — the cloud lies entirely in one arm at dim `U_pos` = dim(occupied), so
every drawn direction is inside it — but the *statistic* is not deterministic,
because a direction can land where both arms' effective-rank changes happen to
share a sign. The measured rate is 0.979 at worst rather than 1.000, the record
now carries it, and the check asserts a rate rather than an identity: an
assertion that fails once in fifty runs is an assertion about the draw and not
about the gate.

**The exchangeable input, which is this entry's sharpest statement and has no
counterpart elsewhere in the registry.** Draw the observed pair as a random
re-split of a fixed union and it is exchangeable with the null draws *by
construction*, so P(p ≤ α) ≤ α exactly, for any population, with no modelling
assumption at all. Over 200 draws on a population occupying both arms at
chance-normalized occupancy 2.24, all 200 emitted, and the adjudicated tail
rejected on **0.020** of them with the reciprocal tail at 0.070 — both inside
one standard error of nominal at that replicate count, and the mean p is 0.555
rather than the 0.5 an exactly-uniform p would give, which is the discreteness
of a rank statistic over 100 values showing up as mild conservatism. None of
that conservatism is ties: the fraction of runs where both tails read exactly
1.0 is 0.000, so it is control rather than refusal — §6g's distinction, asked of
a rank statistic. Every other validity number in this project is a
rate under a modelled H0 family; this one's answer follows from the
construction, so a failure would localise to the implementation rather than to
the choice of family. It is deliberately run on a population that occupies both
arms — the family that retired the previous null — because running it on a bland
one would be an arm incapable of failing.

**The verdict band, swept over the occupancy asymmetry and the precondition
ratio.** No cell is dead in the strict sense — every one of the eighteen reached
a verdict on at least one of its thirty draws — but the perfect-input
counterfactual separates them sharply, and it separates them along the
**precondition ratio** rather than along the asymmetry:

| dim `U_pos` / dim(occupied) | perfect input reaches a verdict | P(TRACKS) across the row |
|---|---|---|
| 1 | 5/5 in every cell | 0.00 → 1.00 |
| 2 | 2/5 to 5/5 | 0.00 → 0.60 |
| 3 | **0/5 in five of six cells** | 0.00 → 0.12 |

At ratio 3 the design is not merely underpowered: the strongest input the cell
can be handed comes back INSUFFICIENT essentially every time, and the verdicts
that do appear are stray draws at rates indistinguishable from the Type-I rate.
That is the whole-gate version of the per-pair informative rate the registry
already recorded as a precondition (1.000, 0.710, 0.320, 0.030 at ratios 1, 1.5,
2, 3) — and it turns a rate into a statement about what the gate can return.

It stops short of §6j's claim, and the field names say so. `CLAIM-C`'s band was
settled by enumerating every concordance count, which *proves* the gate is a
constant function there. This statistic has no such enumeration, so what is
recorded is `no_verdict_in_any_draw` — a measured zero over a stated number of
draws — and at ratio 3 even that is not reached, because stray draws do get
through.

**Refusals and branches.** Every refusal is reached by an input built to trigger
it and re-scored to check it turns away nothing that could have cleared α;
every verdict branch fires on the input built for it. The refusal this pass
added is exercised there too, because a refusal no input in the record reaches
is a refusal nothing has checked.

### The defect this pass found in its own arm, which is the sixth session running

The band sweep re-scores a "perfect input" in every cell to separate *the data
was not strong enough here* from *nothing reached a verdict at all*. The first
version read that counterfactual off a **single draw**, and marked cells as
reaching no verdict whose own twenty-five draws reached one 28% of the time. It
now runs five seeds and the field is called `no_verdict_in_any_draw`. Nothing
failed; printing the table is what showed it — after §6g's rounding defect,
§6h's audit arm, §6i's discarded-null power figures, §6k's α on a shoulder and
§6l's empty drop slab.

### The cost, and what carries forward

`claims/calibration/steering_sign.json` went from about twenty-five minutes to
about forty. Every gate run now computes two subspace nulls on the same draws so
the comparison is paired; two `H0-both-arms` families were added; and a
dedicated `reciprocal_tail` section measures the INVERTS branch at **four times
the replicates** at one pair count, which closes the measurement §6k named as
this construction's weakest — fifty runs resolve a rate to ±0.03 and cannot
separate nominal from twice nominal. Against that, the calibration's null draws
dropped from 99 to 49 (floor 0.02, still far under α), which is where most of
the doubling was paid for.

**And the higher-replicate section earned itself immediately.** The main
validity table, at fifty runs a cell, shows one H0 cell with a reciprocal rate
of 0.10 — twice nominal, and exactly the kind of number that reads as a defect.
It is not one: at fifty runs a true 0.05 lands on 0.10 about once in ten cells
and that table has twenty of them. The dedicated section measures the same
family at two hundred runs and reports 0.015. §6k said fifty runs cannot
separate nominal from twice nominal; this is what that looks like when it
happens, and the reason the assertion on that table is now α plus one standard
error of a proportion — derived from the replicate count — rather than a placed
tolerance.

**A separate cost, and it is the one to carry.** Both retired nulls' rates were
quoted inline in the module's docstring, and cross-checking them against the
regenerated artifact found three of them stale — a permutation figure that no
section contains, a plateau rate off by 0.04, and an inversion rate that
contradicted a second copy of itself eleven lines away. None of them was wrong
when written; each stopped being about the artifact and nothing noticed, which
is precisely what §6g item 4 records for §6f's two headline numbers. The module
now carries **pointers rather than digits** for every rate an artifact holds,
and says so where a reader would otherwise wonder why. That is the seventh
instance of the pattern this pass — and the only one that was found by checking
prose against a file rather than by looking at the file.

**Still no data.** As in §§6e–6l, the apparatus exists and the artifacts do not:
the gate needs activations and the Phase 2 attractive/repulsive projectors, and
neither is in this repository. `claims/adjudications/` is empty and
`null_construction` has still not frozen — which is the only reason the null
could be replaced at all, and it is the second time that window has been used.

## 6n. The retired null, checked where it came from: P6-R2 and P6-R4 (2026-08-26)

§6m retired the null `P-ST1` adjudicated — a matched-dimension random
orthogonal subspace pair — because it randomises the union of the two subspaces
together with the split between them, and so rejects when the pair is unusual
*as a pair* rather than when the labelling predicts anything.

**That construction is §6h's, and §6h introduced it here.** `P6-R2` and
`P6-R4` are where it was built, and `P-ST1` borrowed it. A defect found in a
borrowed construction is a defect to check at its source, which
`claims/EVALUABILITY.md`'s opening argument makes non-optional rather than
tidy-minded: the product is only as valid as its weakest factor, and three
entries across two claims shared this one. `tools/dry_run_p6_r2_r4.py` →
`claims/audits/p6_r2_r4_dry_run.json` is the check, and it doubles as the dry
run `EVALUABILITY.md`'s queue owed both entries.

The two entries came back differently, and the difference is the finding.

### P6-R2 has it, and the evidence is a trend rather than a rate

The H0 is exact: each layer's split is drawn uniformly at random inside its own
union, so the operator's labelling carries no information by construction and
the correct answer is *do not reject* at every point of the sweep. What moves
along the sweep is only how far the union sits above chance against that
layer's separating direction — the quantity a matched-dimension null does not
reproduce. Measured at 250 replicates a cell, both nulls scored on the same
runs:

| union alignment (chance = 1.0) | retired null | adjudicated null |
|---|---|---|
| 1.00 | 0.000 | 0.052 |
| 3.43 | 0.096 | 0.040 |
| 3.70 | 0.108 | 0.044 |
| 3.94 | **0.156** | 0.048 |

The replacement holds the union fixed and re-splits it at the observed
dimensions, so exchangeability under H0 is by construction rather than by
measurement — the same fix as §6m's, in the same shape.

**What the replacement's own rate is, stated precisely rather than rounded to
"nominal".** 250 replicates separate 0.05 from 0.15 and do not separate 0.05
from 0.07, so the artifact carries a `precision_check` section at 600. Two
independent measurements of the aligned end — that section, and a
1000-replicate run during construction — gave **0.068** and **0.048**, pooling
to about **0.056**. So the replacement is at or *marginally above* 0.05 there,
and saying otherwise would be rounding a measurement into a claim. What is
established is the thing that matters: it does not **trend** with the union's
alignment (0.047 at chance against 0.068 at 3.9×, within sampling error of each
other) where the retired null runs 0.000 to 0.155, and at the aligned end the
two are **9.7 standard errors apart**.

**A trend is the point, not a rate.** A single cell is a proportion over a few
hundred draws, and this pass produced a 0.076 at 250 replicates that came back
0.050 at 1000. What makes this a mechanism rather than a cell is that the rate
rises monotonically in exactly the quantity the retired null fails to hold
fixed, and vanishes when that quantity is at chance.

**And the same fact cost this file a red gate, which is worth recording.** The
`precision_check` section's first bound was alpha plus 1.96 standard errors
applied to each of its two cells — a bound that fails once in twenty
regenerations when the null is exactly nominal, *by construction*, because two
one-sided cells at 2.5% each is a 5% family. It duly failed, on a 0.068. The
bound now carries a Bonferroni allowance derived from the cell count at a
family-wise level tighter than alpha, and the reason is about the check rather
than the science: a gate that cries wolf once per twenty regenerations is a gate
people re-run rather than read. Placing a threshold on a *proportion in a
regenerated artifact* needs the regeneration count in it, and nothing in this
project had said so before.

**Power is unchanged**, so the fix costs nothing here: against the union's
content concentrated in `U_neg` — P6-R2's predicted direction — both nulls
reject at 1.000 across the sweep.

### P6-R4 does not have it, and that is structural

`P6-R4` compares **one** subspace against matched-dimension random ones. There
is no union and no split for this defect to reach. The analogous question is
whether matching the dimension is enough when the observed subspace is not a
random one — a probe's accuracy inside a projection depends on the retained
signal and the retained within-cluster noise together, and which way that cuts
is not predictable, so it was measured rather than argued. Where a
high-variance `U_S` captures 3.4× the population variance a random subspace of
its dimension would, the rate holds at **0.040–0.048**.

So `P6-R4` is left alone — and the measurement is in the record precisely
because leaving it alone is a decision. Without that arm the difference between
the two entries would rest on an argument about their statistics rather than on
a number, which is the position §6h's construction was in for two passes.

### What actually decides it: the statistic, not the claim

Three entries now share §6h's construction and it is valid in one of them,
mildly invalid in another and badly invalid in the third. The property that
separates them is **whether the statistic cancels a common elevation of both
arms**:

| entry | statistic | behaviour under an elevated union |
|---|---|---|
| `P-ST1` | the **sign** of a difference | saturating, so no cancellation at all — 0.20 where each arm held 1.27× chance (§6m) |
| `P6-R2` | a **difference** of two chance-normalized alignments | cancels to *first order*, so it survives until the union is strongly aligned — 0.14 at 3.9× |
| `P6-R4` | a **single** subspace against matched controls | no common elevation to mismatch — 0.04–0.05 at 3.4× |

That is the transferable result, and it is forward-looking: `EVALUABILITY.md`
lists `P6-R1`, `P6-C1`, `P5b-A1`, `P5b-A2`, `P-SA1` and `P-I4` as queued rows
whose predictions already name a matched control. **Matched on what** is now a
question with a checkable answer rather than a preference, and the answer
depends on the statistic each row builds.

It also sharpens §6h's own lesson. §6h moved `P6-R2` from units to subspaces on
the question *what is being randomised?* and that was right; what it did not
ask is *what else moves when the subspace does*. Randomising a subspace
randomises everything about it, including the properties the observed one has
for reasons the claim is not about.

### What this pass did not do

It does not touch `P6-R4`, whose null is unchanged and now has a measurement
behind that. It adjudicates nothing: the populations are synthetic, no ALBERT
run artifact is in this repository, `claims/adjudications/` is still empty, and
`null_construction` has still not frozen — which is again the only reason the
null could be replaced at all, and the third time that window has been used.

And one small thing worth recording because writing it down did not prevent it.
The dry-run tool bound a module constant as a **default argument**, so a caller
overriding the constant did not reach it — §6h found that bug in
`attainable_floor_report`, §6m found it again in `tools/dry_run_p_st1.py` and
wrote a comment about it, and this file reproduced it a third time anyway,
caught by a smoke run taking implausibly long rather than by anything failing.
A comment is not a guard.

## 6o. CLAIM-B and P-I1 run on inputs whose answer is known, and the location that is partly the grid's (2026-08-27)

`claims/EVALUABILITY.md`'s queue owes every converted row a run on an input
whose correct verdict is fixed a priori. Four were done; these two are the
fifth and sixth, and they share one estimator —
`core/changepoint_colocation.py` — so one dry run covers both, which is why
§6i built them together. `tools/dry_run_claim_b_p_i1.py` →
`claims/audits/claim_b_p_i1_dry_run.json`, committed for the same reason as the
other six artifacts — and it is the first whose generation cost is measured on
every write and stored in the record as `elapsed_seconds`, rather than quoted in
a docstring where §6n's had to be corrected in three places after a section was
added.

They came back differently, which is the second pass running where a shared
construction is valid in one entry and not in the other.

### A change location is partly a property of the sweep grid, and it was measured rather than argued

The location of a series' change is the centroid of its change-mass profile: a
weighted mean of the sweep's interval midpoints. So mass spread evenly over the
sweep lands on the grid's **own midpoint** — exactly, by the definition of a
mean, not as an approximation. Per-checkpoint noise puts rectified mass in every
interval, so any real location is a **mixture** of where the series changed and
where the grid's midpoint is, weighted by the noise's share of total change
mass.

That share is `n_intervals · σ · √2 / √(2π)` against the series' own range, so
it grows with the interval count and **a denser sweep is worse rather than
better.** The dry run predicts the centroid from those two numbers and measures
it beside the prediction, over three grids and four noise levels; the worst
disagreement is 0.061 in log10-step. At σ = 0.02 — the committed calibration's
own noise level — a change planted at step ~1000 reads at step ~1000 on the
25-checkpoint sweep and at step **~10,000** on the 154-checkpoint one, where 0 of
300 draws put it inside the anchor window at all.

**The module's power argument for change-mass weighting is correct and does not
cover this.** §6i measured change mass holding power at 1.000 from 20 to 143
checkpoints where rate weighting falls to 0.090. That measurement is about the
**mutual** arm: both series are dragged the same way and the pairing null holds
both marginals fixed, so the pull cancels. It does not transfer to an arm whose
reference is a fixed window. §6n's question — does the statistic cancel a common
elevation of both arms? — arriving as a common pull toward one point.

### `CLAIM-B`'s anchor arms have it, and on the registered sweep it is total

The registry names CLAIM-B's instrument as a "20-30 checkpoint cheap-tier
sweep". That grid's uniform-profile midpoint is step **955** — inside CLAIM-B's
own registered 512–2000 anchor window. So a series that changes **nowhere**
receives the anchor arm's maximum statistic, and against controls that all carry
a located change:

| input | anchor arm rejects at α = 0.05 |
|---|---|
| change planted inside the window | **1.000** |
| change somewhere else (H0) | 0.075 |
| **no located change at all** | **1.000** |

The arm's discriminating power there is **0.000**, measured as the difference
between the first and third rows on the same cell against the same control
draws. It is not a Type-I rate in the arm's own terms — a series with no change
is outside the design's domain — which is exactly why it was invisible: no H0
family anyone would write down contains it. §6m's eighth lesson, that the
families a calibration measures are part of the measurement, arriving in the
form where the missing family is not an H0 at all.

**The rate is exactly `1/(k+1)`**, with *k* the number of controls that are
themselves change-free, measured against that closed form at every *k* from 0 to
19 (worst error 0.024). A change-free series beats every control that has a
located change, because on a grid whose midpoint sits inside the window a real
change is usually further from the window than the midpoint is; its only
competition is the change-free controls.

### The refusal, and the first attempt at it that the measurement threw out

`anchor_arm` now refuses when the change-free reference lands **inside** the
window, where it attains the arm's maximum. The condition reads the step grid
against the registered window and nothing else — no controls, no observation, no
α — so it is decidable **before a checkpoint is sampled**, which is where a
requirement on a pilot belongs. Nothing is placed: the reference is the uniform
profile on the grid, and "inside the window" is a comparison of two numbers the
prediction already fixes.

**It was first written as the reference's RANK among the controls, and that was
wrong.** It looked right — rank it the way the observation is ranked, refuse
when a change-free series would clear α — and the family sweep is what showed
it is not: the reference is a *noiseless* profile and a realised change-free
series is a noisy one, so the reference outranks even the change-free members of
a family and its rank pegs at the floor whatever it is handed. Across the whole
*k* = 0…19 sweep the rank is flat at **0.050–0.051** while the rate it was
meant to track runs 1.000 → 0.050. A condition built on it cannot see the axis
that matters. The sweep is in the artifact because it is what corrected the
change, not as background.

**Unlike §6l's and §6m's, this refusal costs verdicts, and the record says so
rather than claiming otherwise.** §6l's informative-row refusal removed no
reachable verdict and was measured at zero power cost; §6m's attainable-floor
refusal could not cost one by construction. This one turns away inputs that
would have rejected, including inputs whose change really is at the anchor — on
the registered sweep it costs **1.000**, the whole arm. What it refuses is a
verdict the design cannot **support** rather than one it could not **reach**,
which is a third category this project had not used, and arm D re-scores the
counterfactual in every cell instead of asserting the cost is small.

**And what it deliberately does not refuse.** On a grid whose midpoint sits just
*outside* the window the reference no longer attains the ceiling but a
change-free series still clears α on about a quarter of draws — five times
nominal, and unrefused.
That residual is a rate rather than a ceiling, so it is reported beside every
result and the analyst must state that the series under test has a located
change, the same posture §6i takes toward the shared-unit-factor rate of 1.00
that no null over the pairing can separate.

### `P-I1` does not have it, and that is structural

`P-I1` is the mutual arm alone: a difference of two locations, with a null that
permutes the pairing and therefore keeps both series' real per-head locations on
both sides of every draw. A pull that moves every location the same way is
present in the null exactly as in the observation. Measured on the **registered
cheap sweep** — the grid where the anchor arm fails hardest, because measuring
it anywhere else would be choosing the easy case:

| H0 family | mutual arm rejects |
|---|---|
| both series change nowhere | 0.065 |
| one series changes nowhere | 0.045 |
| both change, independent onsets | 0.055 |
| both change early (common trend) | 0.050 |

So `P-I1` is left alone, and the measurement is committed because leaving an
entry unchanged is a decision — the precedent `P6-R4` set in §6n one pass
earlier, now used twice.

### What actually decides it, which sharpens §6n rather than repeating it

§6n's taxonomy asked whether a statistic cancels a common **elevation** of both
arms, and put `P6-R4` — one subspace against matched-dimension controls — in the
safe column with "nothing to mismatch". This is the counter-example that says
what that column requires:

| entry | statistic | the quantity it degenerates on | matched on it? |
|---|---|---|---|
| `P6-R4` | one subspace against matched controls | subspace dimension | **yes**, by construction |
| `CLAIM-B` anchor | one location against a fixed window | where the grid puts an unlocated profile | **no** |
| `CLAIM-B`/`P-I1` mutual | a difference of two locations | — cancels | n/a |

**An absolute quantity against matched controls is safe only when the controls
are matched on the quantity the statistic degenerates on.** "Matched on what" has
to name the statistic's degenerate input, and for the six queued rows
`EVALUABILITY.md` lists that is a question to answer before the control is built
rather than after a rate comes back.

### Three smaller things, and the first is the sixth session running

**CLAIM-B's falsification branch fires, and with no margin.** RE-ANCHORS needs
unanimity in the reciprocal direction, and both anchor arms' reciprocal p is
floored at `1/(n_controls + 1)` — which is *exactly* α at nineteen controls. So
the branch requires the observed series to rank strictly worst of twenty in both
anchor arms at once. It fires on most of the inputs built for it — the artifact
carries the rate, which is a proportion over twenty draws — and a single control
further from the window than the observation removes it. The nineteen
control series `EVALUABILITY.md` records as the minimum for CO-LOCATES are
therefore also the exact minimum for the falsifier, with no margin either way —
a second reading of a number that has been in the registry since §6i.

**Two rows of the known-answer arm asserted a draw rather than a gate**, and both
were caught by running them. An H0 input reaches a verdict branch at α by
design, so "five INSUFFICIENTs out of five" fails about once in ten runs, and did
— on the anchor arms' H0 row at exactly the floor. Both rows now assert a rate
with an allowance derived from α and the seed count. §6m recorded the same
correction for `P-ST1`'s sharp input; writing it down did not prevent it.

**And a falsy zero in this pass's own staleness check.** `check_record` guarded
the discrimination with `(value or 1.0) > 0.10` — and the value it guards is a
number that *should* be 0.0, which is falsy, so the fallback fired on the healthy
artifact and reported the finding missing. Found by running `--check` on the file
just generated, not by a test. That is the seventh session running in which
looking at a generated output found something nothing was failing on, after
§6g's rounding defect, §6h's audit arm, §6i's discarded-null power figures,
§6k's α on a shoulder, §6l's empty drop slab, §6m's single-draw counterfactual
and §6n's default argument.

### What the pilot must now produce, and it is about which checkpoints rather than which metrics

The fifth pre-computed requirement in five passes, and the first that constrains
the **sweep grid** rather than the measurements taken on it: CLAIM-B's anchor
arms need a sweep whose uniform-profile midpoint falls **outside** the 512–2000
window. The registered 25-checkpoint cheap-tier sweep puts it at step 955 and
fails; Pythia's full every-1000 schedule puts it at step 31496 and passes the
condition — but at that density the noise share is 0.63 and a real change at the
anchor is dragged out of the window, so the arm has no power there either. **The
two failure modes are one mechanism read at two grid geometries, and the sweep
that satisfies both is neither of the two the project has.** That is a design
question for the pilot, computed before it runs, and it sits beside the
nineteen-control requirement rather than replacing it.

The coincidence that hid all of this is worth stating plainly: the cheap sweep's
midpoint sits almost exactly where CLAIM-B's anchor is. On the one grid this
construction was calibrated for, the bias is invisible **because it points at
the answer**.

**Still no data.** As in §§6e–6n, the apparatus exists and the artifacts do not:
`INDEX.md` records the dense pilot sweep as not executed,
`claims/adjudications/` is empty, and `null_construction` has still not frozen
— the fourth time that window has been used.

## 6p. The last three dry runs: P-S1, P-T1 and P-M1, and a floor that was never the design's (2026-08-27)

`claims/EVALUABILITY.md`'s queue is finished. Six of the nine adjudicable rows
had been run on inputs whose correct verdict is fixed a priori; these are the
last three. `tools/dry_run_p_s1.py` → `claims/audits/p_s1_dry_run.json` and
`tools/dry_run_p_t1_p_m1.py` → `claims/audits/p_t1_p_m1_dry_run.json`.

**Nine for nine, every one of them changed something.** That is the fact worth
recording about the exercise as a whole: not one converted row survived being
run on an input whose answer was already known, and no test was failing on any
of them.

### `P-T1` and `P-M1`: a reported floor that belongs to the call, not the design

`core/nulls.p_from_null` reports `resolution` = 1/(n_draws + 1) and calls it
"the honest resolution limit". It is — of the **sample**. It is the smallest p
a run can actually express only when the statistic is continuous, so that ties
with the observation have probability zero. Both of these statistics are
**discrete**: a rate difference over tens of heads, a correlation against a
binary series. The null puts a lump of mass exactly on the observed value, and
the smallest expressible p is set by the data's own marginals.

Measured on a **perfect** input — the most extreme arrangement the marginals
admit:

| | design | the design's floor | reported resolution | ratio |
|---|---|---|---|---|
| `P-T1` | 2 candidates, 3 controls | **0.100** | 0.0005 | 200× |
| `P-M1` | 12 layers, 1 violation | **0.083** | 0.0005 | 167× |
| `P-M1` | 6 layers, 1 violation | **0.167** | 0.0005 | 333× |

Those floors are exact, not sampled, and a perfect input is now refused at each
of the three rather than returning a p just above α. Both were returning "not
significant" from designs that could not have rejected, which on a prediction
reads as evidence against it. §6f established that refusal for `CLAIM-C`, §6i
reached it for `CLAIM-B` before building, §6l found it again on `CLAIM-C`'s
informative rows and §6m found it on `P-ST1`'s reported floor. This is the
fifth arrival and the first where two independently written constructions had
it at once.

**Both floors are exact and neither contains a draw count.** `P-T1`'s statistic
is monotone in how many trimodal heads land in the candidate arm and the null
holds both marginals fixed, so that count is hypergeometric and the floor is
the tail at the most extreme table — exact, by `math.comb`. `P-M1`'s
permutations that only swap equal violation values give the same correlation,
so the floor is `∏ (multiplicity)! / n!`, which for a binary series is
`1/C(n, T)`. A tied regime score makes the true floor larger, so that one is a
**lower bound** and refusing on it can only under-refuse — §6m's 2m argument in
a second setting.

**The refusal costs nothing, and here that is enumerated rather than measured.**
§6l had to re-score `CLAIM-C`'s counterfactual because there the floor and the p
come from different code. Here the floor *is* the smallest value of the same
discrete p, so every attainable arrangement at every refused configuration can
be listed: none clears α. §6m distinguished a zero that is proved from a zero
that is measured, and this is the second of the first kind.

**And the old resolution was not wrong everywhere, which is the honest reading.**
The smallest p a run can express is the **max** of the design floor and the
sampling resolution, and they bind at opposite ends: at 12 candidates and 36
controls `P-T1`'s design floor is below 1e-6 and the draw count binds again. It
was wrong exactly where these two entries live — tens of heads, tens of layers,
few violations — and the record says which constraint binds in every row rather
than implying one of them always does.

**Three designs now never emit at all**, and that is the pre-computed
requirement in the form a reader can see: `P-T1` at 2 candidates against 3
controls, `P-M1` at 6 layers with one violation and at 12 layers with one. A
36-layer model with a single energy-monotonicity violation is not an exotic
case; it is the case a mostly-monotone trained model produces.

### And a dependence neither entry recorded, on the pair where it matters most

`P-T1` and `P-M1` classify the same head's `Wq`, `Wk` and `W_OV` — one on the
eigenstructure of `V` and the QK form, the other on `M = QᵀK`'s symmetry and
`V`'s alignment with it. The classifications differ; the objects classified do
not, and neither does the extraction that decides which head's weights are
which. §6h spent an audit ruling out exactly that class of defect one phase
over.

`P6-R2`/`P6-R4` record their shared projector and `CLAIM-B`/`P-I1` record their
shared estimator. These two recorded nothing — **and they are the pair where it
matters most, because both are H-OPERATOR's.** CLAIM-B and P-I1 sit under
different claims, so a shared defect does not multiply inside one product.
Two e-values that one defect moves together, multiplied into one claim's E, is
the specific way a product inflates without anyone editing a number, which is
`EVALUABILITY.md`'s opening argument. Both entries now say so.

### `P-S1`: the largest Type-I number the registry has produced

`p_value_p_s1` takes `m` and `d` from the **trained** arm, draws its null
there, and re-references both arms against that one baseline. That is §6d's fix
and it is right when the two arms sit at the same configuration. Nothing
checked that they did.

For i.i.d. points on the sphere `E[Q_k] = 1/m` exactly, so the baseline scales
like `1/m`. A step-0 arm at a different cluster count is divided by a baseline
that is not its own and its ratio is off by roughly `m_trained/m_step0` — which
enters the statistic as a **difference between the arms**, the exact shape of
the effect P-S1 predicts. Measured on **two i.i.d. arms**, where the correct
verdict is "no difference" at every row:

| trained | step 0 | rejection rate | mean p |
|---|---|---|---|
| 32 | 32 | 0.075 | 0.508 |
| 32 | **30** | **1.000** | 0.008 |
| 32 | 28 | **1.000** | 0.008 |
| 32 | 24 | **1.000** | 0.008 |
| 32 | 36 | 0.000 | **1.000** |
| 32 | 40 | 0.000 | **1.000** |

**Two clusters out of thirty-two** — six percent — is enough to take an input
whose correct verdict is "no difference" to certain rejection. And the error
runs both ways: fewer step-0 clusters inflates the step-0 ratio and therefore
*confirms* the prediction, while more sends p to 1.000 and the design can never
win. Neither is an answer an analyst would notice.

**Unequal cluster counts are the expected case.** Clustering runs per
checkpoint, and a random-weight model's activation geometry is not a trained
one's. This is not a stress test of the gate; it is what the gate was going to
be handed.

**The refusal is more than a guard.** `p_value_p_s1` now refuses when the two
arms report different `(m, d)`, and refuses when the step-0 arm does not report
them at all. It is a degeneracy and not a tolerance — the counts are equal or
they are not. And it is a statement about the statistic rather than about the
code: `Q_k`'s i.i.d. floor depends on `m`, so "closer to a spherical design" is
not a comparison that exists across different `m`, and no choice of baseline
rescues the row. It therefore costs nothing **by construction** — there was no
correct p there to remove — which is the third refusal of that kind after
§6m's and this pass's other one.

**The sixth pre-computed requirement in six passes**, and the first that
constrains how a run is *clustered* rather than what it measures or where it
samples: both arms must be clustered to the same count, rather than each to its
own best `k`.

### A number that stopped describing its own path, for the second time

The module warned that its `Q_ratio` fallback — taken when raw `Q` is absent —
leaves the p-value "mildly anticonservative", citing a null-p mean of **0.40**
against the 0.50 a calibrated statistic must give. Measured on the same draws,
the two paths are indistinguishable: 0.508 against 0.507, and rejection 0.058
against 0.058.

The 0.40 was real when it was written, on the pre-2026-08-24 code where
observed and null sat on genuinely different baselines. The fix changed the
structure: the statistic is now a **difference** of two ratios formed against
the *same* caller baseline, so a common per-degree factor cancels to first
order and what is left is a rescaling of about a percent. The note stopped
describing the path it was attached to and nothing noticed — §6m found three
such numbers in one docstring and the module now carries pointers; this is the
second arrival, and the reason to state the mechanism in a note rather than a
rate.

### What did not change, and the reason it is here

`P-S1`'s reported floor **is** attainable: its statistic is continuous, ties
have probability zero, and a strongly-spread trained arm lands exactly on
1/(n_null + 1) on every draw. That is the claim that failed for `P-ST1`, `P-T1`
and `P-M1` — all three discrete — so it was checked rather than assumed, and
the measurement is committed because a check that comes back clean is a
decision too. `P6-R4`'s precedent, used a third time.

### The taxonomy the nine dry runs leave behind

Every one of the nine found something, and sorted by what was wrong they fall
into four kinds rather than nine:

| what was wrong | where |
|---|---|
| a reported floor that was not the design's | `P-ST1`, `CLAIM-C`, `P-T1`, `P-M1` |
| a null that randomised more than the claim is about | `P-ST1`, `P6-R2` |
| a statistic partly determined by the measurement grid rather than the data | `CLAIM-B` |
| an input the design cannot compare, scored anyway | `P-S1` |

The first is by far the commonest, it is cheap to check, and it is checkable
**before** any data exists — the floor is arithmetic on the design. That is the
one to ask of the next construction first.

### What this leaves

`claims/adjudications/` is empty, `null_construction` has still not frozen, and
no run artifact is in this repository. The queue `EVALUABILITY.md` opened on
2026-08-25 — *"a second entry ahead of it for each row already converted"* — is
now closed, and what stands ahead of converting the next `needs-null` row is
the list of pre-computed requirements six passes have produced, none of which
any existing sweep satisfies.

## 6q. `P-AB1`: the last unbuilt bridge entry, and an exponent that is not monotone in what it measures (2026-08-27)

`claims/EVALUABILITY.md`'s dry-run queue closed in §6p, and what stood ahead of
converting the next `needs-null` row was that document's own instruction for
every row naming a matched control: *"compute the attainable floor, name what
the statistic degenerates on, check what the measurement grid contributes, and
only then build the control."* `P-AB1` is the last unbuilt Phase 7 bridge entry
with a live instrument and `design-7.md` calls it the one entry in its
translation table "where the particle account plausibly says something the
mechinterp framing does not already say". It is now built —
`p7_motifs/patching_gate.py`, with
`tools/calibrate_patching_exponent.py` → `claims/calibration/patching_exponent.json`
behind it. Ten predictions are adjudicable in principle; `claims/adjudications/`
is still empty.

**This is the first construction built in that order, and each of the first
three steps changed the design before any control existed.** That is the
argument for the order, and it is worth stating as a result rather than as a
process note: three of the four defect kinds §6p catalogued are checkable
before any data exists, and all three were missed on the nine rows that had
them. Asked first, they were all found.

### 1. The floor, and the registered null that cannot reach any

The registry's `null_construction` read *"Permutation over ablation points once
the fitted exponent is the statistic."* Read literally — permute which ablation
point's real exponent is compared against which point's control exponent — it is
**degenerate**, and pure algebra says so. The natural statistic on a matched
design is the mean paired difference, and permuting the pairing gives
mean β_real − mean β_control **for every permutation**. The null has no spread,
every draw ties the observation, and the smallest p the design can express is
**1.000**.

Not a small floor: the largest one there is. §6p's seventeenth lesson — a floor
is a claim about the design and not about the call — in the one form where the
design can never reject whatever it is handed, and reachable with no data and no
simulation.

The reading that works is the other one. The exchangeable object under H0 is the
**label** on the two directions at one ablation point: if ablation removes a
value from a sum, a real direction and a structureless one of equal magnitude at
the same layer are exchangeable, so swapping their labels is exact by
construction. That is a sign-flip null with floor `(2^(n−k)+1)/(2^n+1)` in the
informative units — CLAIM-C's rule from §6l, reached by a second construction
because it is the same group — so **six informative units is the first design
that can reject at all**, after CLAIM-C's six prompts (§6f), CLAIM-C's five
informative rows (§6l) and `P-ST1`'s two pairs (§6m).

**A second floor binds at the other end, and it was found the way the last nine
were.** Under the per-ablation-point unit the sign-flip group is 2^42 and is
sampled, so the run cannot express anything below `2/(n_patterns + 1)` either:
the design floor is 4.5e-13 there and a *perfect* input returns 4.0e-4 — nine
orders of magnitude apart, and the arm was reporting the first. That is §6i's
defect exactly, where CLAIM-B's sampled pairing regime reported a floor smaller
than any p it could express, and §6p's rule is the fix: the smallest expressible
p is the **max** of the design floor and the sampling resolution, and the two
bind at opposite ends — under the per-prompt unit the group enumerates and the
design floor is the whole story. Nothing was failing. It was found by printing
what a perfect input returns beside what the arm claimed it could return, which
is the **tenth session running** in which looking at a generated output found
something no test caught, after §6g's rounding defect, §6h's audit arm, §6i's
discarded-null power figures, §6k's α on a shoulder, §6l's empty drop slab,
§6m's single-draw counterfactual, §6n's default argument, §6o's falsy zero and
§6p's stale fallback note.

**And a second thing the same reading caught**, smaller and the same shape as
§6p's: the magnitude check the registry requires was being run on whatever array
it was handed, without checking that array indexes the ablation grid the curves
describe. Magnitudes for a different grid satisfied it silently — an unchecked
match, in the refusal that exists for unchecked matches. Refused now, and pinned.

**And §6l's informative-row structure arrives with it, exactly.** A prompt
contributes the SUM of its ablation points' signs, so a prompt with an **even**
number of points can split evenly and contribute nothing to the observation or
to any sign pattern. Under H0 each sign is a fair coin, so at six prompts:

| ablation points per prompt | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| P(the design can reject at all) | 0.274 | **1.000** | 0.394 | **1.000** | 0.479 | **1.000** |

Six points per prompt leaves the design unable to reject on 61% of draws and
seven leaves it able on all of them. **An odd number of ablation points is free
and an even number is not**, it is exact arithmetic on the grid rather than a
fact about any model, and it is in `grid_arithmetic` in closed form.

### 2. What the statistic degenerates on: the fit window, which the ablation point sets

§6o's rule is that "matched on what" has to name the quantity the statistic
degenerates on. A fitted growth exponent degenerates on the **window it was
fitted over**, and the ablation point fixes that window: ablating at layer ℓ of
an L-layer model leaves K = L − ℓ downstream layers and no more.

Divergence saturates — two trajectories can only diverge so far — and the
log-log slope of a saturating curve is a **decreasing function of K**. On one
fixed set of dynamics, `D(k) = 1 − exp(−(k/τ)^β)` at β = 2.0 and τ = 4:

| K | 3 | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| fitted exponent | 1.79 | 1.71 | 1.53 | 1.35 | 1.07 | 0.88 | **0.66** |

A factor of nearly three on nothing but where the measurement stopped.
**"Superlinear" is not a window-free statement**, and the registry's own reason
for requiring a control — "later layers have more opportunity to diverge for
reasons unrelated to field structure" — is right but is not the binding one. The
binding one is that β has no meaning at all until a window is named.

The pairing disposes of it: both arms at one ablation point are fitted over the
same window, so its contribution is identical on both sides of the difference.
That puts `P-AB1` in the safe column of §6o's table — **and only because it
pairs at the ablation point.** The comparison across points, which the
registered null's literal reading performs, is between exponents fitted over
different windows and is not a comparison at all.

### 3. What the measurement grid contributes: a common window, and a sign rather than a mean

The window cancels within a pair; it does not make the pairs commensurable. The
sampling spread of a log-log slope is `σ/√Sxx` with `Sxx` fixed by K alone, so
the fitted exponent's spread runs a factor of five across one model's ablation
grid — exact arithmetic, tabulated in `sampling_spread`, no replicates. A **mean**
of paired differences is then dominated by whichever ablation point sits nearest
the output and carries the least information.

Two consequences, both taken. The gate fits every point over one **common
window**, the largest every included point can supply, read off the input rather
than placed — and what IS a choice is the caller's ablation grid, which trades
directly against the floor, since n points and a window of W need n + W ≤ L. On
a 12-layer model six ablation points leave a window of six; on a six-layer model
there is no design at all. And the units are combined by a **sign**, so a point
fitted over a short window counts exactly as much as one fitted over a long one
and no more. The cost was measured rather than assumed: against a mean of paired
differences on a grid whose windows deliberately run 3 to 27, power at a planted
gap of 0.10 is **0.910 against 0.872** and at 0.05 **0.429 against 0.383**, with
the H0 rate lower too, 0.030 against 0.045. It also places no constant, which is
§6i's change-mass centroid, CLAIM-C's sign concordance and `P-ST1`'s
sign-of-a-difference escaping the same way a fourth time.

### 4. And only then the control, where the unit is the open question

The control is the registry's — a matched random-direction ablation of equal
magnitude at the same layer — and **equal magnitude is now checked rather than
assumed**, §6p's `P-S1` precedent, where a null drawn at one arm's configuration
and applied to the other's rejected at 1.000 on two i.i.d. arms because nothing
checked they matched.

What the sign-flip needs beyond per-point exchangeability is that the signs are
independently flippable, and that is a claim about the units. `status-6.md`'s
"49 layers are not 49 independent observations" applies to ablation points
inside one run exactly as it applied to ALBERT's layers, so both readings were
measured on the same draws, at six prompts × seven ablation points, against a
per-prompt shared component:

| shared share ρ | unit = ablation point | unit = prompt |
|---|---|---|
| 0.0 (independent) | 0.050 | 0.018 |
| 0.5 | **0.141** | 0.016 |
| 1.0 | **0.235** | 0.029 |

The per-prompt unit holds across the range; the per-ablation-point unit — the
reading the registry's wording implies — reaches 0.235. A fourth independent
arrival of the fourfold-plus inflation POPPER reports at 0.082 → 0.340 and §§6f,
6h and 6k each measured. **Which unit may enter an e-process is a scientific
decision of the same class as CLAIM-C's criterion**, so
`REGISTERED_EXCHANGEABLE_UNIT` is `None`, `adjudicate_p_ab1` raises while it is,
and `unit=` selects what to compute and never what may be adjudicated — §6h's
construction, used a second time and for its reason.

**The limitation no label swap removes, measured rather than described.** A
shared component that is a random per-prompt draw is what the prompt unit
disposes of. A **fixed** offset — real ablation directions are not isotropic and
the controls are, so every cell is nudged the same way for a reason with nothing
to do with the field account — is not, by either unit: at one jitter it rejects
at **1.000** under the point unit and **0.895** under the prompt unit. That is
§6i's shared-per-unit-factor at 1.00 in this design's clothing, and every record
carries a `shared_prompt_factor_diagnostic` that catches a factor varying
*between* prompts and catches this one not at all — which is pinned by a test,
so it cannot later be mistaken for coverage.

### The finding: a fitted exponent is not monotone in what the prediction is about

Divergence is **bounded**. So the arm whose divergence is larger at every layer
reaches its ceiling sooner inside a fixed window and its log-log slope
**flattens**: at one true exponent of 2.0 over eight layers, τ = 4 fits 1.35 and
τ = 16 fits 1.95 — and τ = 4 is the arm that dominates everywhere.

On two arms carrying the **same** true exponent where only the real one
saturates sooner — which is what a real ablation that propagates *does* — the
gate returned **`RECAPTURES`, its registered falsification branch, on 0.98 of
draws under the prompt unit and 1.00 under the other.** An input on which the
prediction holds, scored as the prediction refuted. It is §6p's fourth defect
kind ("an input the design cannot compare, scored anyway") in its worst form,
because the verdict it produces is the one that would enter the ledger as a
falsification — and it is not visible in any single result, because the number
the gate reports is the same number either way.

**The first attempt at the refusal was the paired contrast, and the measurement
threw it out.** An equal bend cancels inside the pair, so the confound is a bend
*contrast*, and it is testable two-sided with the gate's own exact null. Right
shape, too weak: on the differential-saturation family it turned away 52 of 100
draws under the prompt unit and the 48 it let through **still returned
RECAPTURES on 0.979 of them**; under the other unit 23 got through and 1.000 of
those did. A refusal that thins a defect is not a refusal. It is kept as a
reported diagnostic naming the *direction* of the confound, and the sweep is in
the artifact because it is what corrected the design — §6o's discarded
rank-based refusal, same shape, one pass later.

**What refuses instead asks the per-arm question.** The exponent is a growth
exponent only where the curve is a **power law** over the window, and that is a
property of one arm rather than of the pair. Each curve's `window_sensitivity`
divided by the spread its own OLS residual gives it is a z under "this curve is
a power law"; pooled over the arm's cells it is standard normal, and each arm is
tested two-sided at α/2 — Bonferroni over the two arms, which is §6n's rule
about putting a threshold on a family without the family's size in it. Measured
over the design's own 42 cells: **0.060 on pure power laws and at τ = 30, 0.360
at τ = 15, 0.970 at τ = 8, 1.000 at τ = 5 and below.** Nominal on the shape it
is meant to admit, certain on the shape it is not.

**It costs verdicts, and the record says which kind of cost.** §6p names three
kinds of "this costs nothing" — measured, proved, enumerated — and §6o names a
fourth category that is none of them: a refusal that is right and costs verdicts
anyway. This is the second of those. It turns away every draw of *both*
saturating families, including the symmetric one where both arms bend equally
and the contrast measured nominal. That case is refused deliberately: an
exponent fitted through a ceiling is a property of the window, so a `PROPAGATES`
verdict there would say the real arm's window artifact exceeded the control's,
which is not what `P-AB1` predicts. What is refused is a verdict the design
cannot **support** rather than one it could not **reach**, and every family in
the calibration carries the counterfactual rate re-scored on the refused draws
instead of a claim that the cost is small.

### The falsifier, and the branch that can carry it

*"Divergence flat in remaining depth, or growing linearly, across ablation
points"* describes the **null**, and an e-process records insufficient evidence
and never a null accepted — §6k recorded the same for `P-ST1` and the resolution
is the same. Both clauses map to INSUFFICIENT. The falsification branch is
`RECAPTURES`: the real ablation's exponent demonstrably *below* its matched
control's, the perturbation reabsorbed faster than a structureless one, which is
`design-7.md`'s own other side of the recapture-vs-propagation question and a
reversal positively shown. §6i's requirement that such a branch be checked to be
one that can fire is met — it fires at 0.457–0.803 against a planted inversion
and at the floor exactly on a perfect one. `p_reciprocal` is a stop-rule input
only and never enters a claim's E, which is CLAIM-B's and CLAIM-C's division
doing real work here rather than being a convention: the branch differential
saturation can manufacture is exactly the one that reaches no ledger.

### What the pilot must now produce

Two more requirements, and the second constrains something none of the previous
six did — **the intervention itself** rather than what is measured, where it is
sampled, how it is clustered or how large the run is:

- **Six informative units, with an odd number of ablation points per prompt.**
  Under the per-prompt unit that is six prompts; the ablation grid and the
  common fit window trade against the model's depth as `n + W ≤ L`.
- **An ablation magnitude and a fit window that keep BOTH arms inside the
  power-law regime.** Checkable off the divergence curves before any p-value is
  computed, refused by the gate when it fails, and not obtainable any other way.

That takes `claims/EVALUABILITY.md`'s list to eight, and no existing sweep
satisfies any of them.

### What this pass did not do

It does not touch `CLAIM-B`'s grid question or the centroid de-biasing decision
(§6o) — both are decisions for the author rather than construction, and the
window to take them is open only while `claims/adjudications/` is empty. It
builds no confound-control arm for `CLAIM-B`/`P-I1`, which remains blocked on
nineteen control series *and* a grid. And it adjudicates nothing: the divergence
curves are synthetic with planted answers, no ablation run artifact is in this
repository, `claims/adjudications/` is empty, and `null_construction` has still
not frozen — the fifth time that window has been used, and the first time it was
used to record that the registered null was not one.

## 7. What this plan does *not* do

- It does not run any science. No chunk here adjudicates a prediction; B6 makes adjudication
  *possible*, and the first real adjudication is a separate decision gated on
  `tools/preflight_1c.py` per `UPDATE_PLAN.md` §3.
- It does not unfreeze Phase 2c/3/4. C2's SAE prediction is registered and left unrun.
- It does not rename phase directories (§C3).
- It does not add an LLM to the loop. POPPER's design and relevance agents are replaced by the
  human pre-registration the project already practises (§3).
