<!-- README.md -->
# Mets — metastable-state / interacting-particle analysis of transformers

Tokens as particles on a sphere, attention as the interaction kernel, a
residual block as one Euler step, depth as integration time. This repository
takes the Geshkovski et al. picture (*A Mathematical Perspective on
Transformers*) literally, measures it inside real trained language models,
and — since August 2026 — tries to write mechanistic interpretability's named
objects (induction heads first) in that language, with every claim
pre-registered and adjudicated by a falsification ledger rather than by prose.

It is a solo research project: Python, CPU-only, ~150k lines including tests,
2,873 test functions, and a living state file that is longer than most papers.
**This README is the front door** — what the project is, where everything
lives, how to run it, and which of the many documents answers which question.
It is deliberately light on results, because results live in files that are
kept current and this one is a map.

**Snapshot: 2026-09-17.** For the current state of the work, read
[`PROJECT.md`](PROJECT.md) — always, and first.

---

## Contents

1. [The idea, in one screen](#1-the-idea-in-one-screen)
2. [Where the work stands](#2-where-the-work-stands)
3. [Which document answers which question](#3-which-document-answers-which-question)
4. [Repository layout](#4-repository-layout)
5. [The phases](#5-the-phases)
6. [The discipline: predictions, nulls, e-values](#6-the-discipline-predictions-nulls-e-values)
7. [Getting started](#7-getting-started)
8. [Testing and CI](#8-testing-and-ci)
9. [Reproducing a number](#9-reproducing-a-number)
10. [How work is organised](#10-how-work-is-organised)
11. [References](#11-references)

---

## 1. The idea, in one screen

Geshkovski, Letrouit, Polyanskiy and Rigollet model a transformer as an
interacting-particle system: each token is a point on the unit sphere in
$\mathbb{R}^d$, self-attention is a kernel that moves the points, and the
layers integrate that flow in time. Their theorems say that, run long enough,
every particle collapses to a single point. Along the way — in their toy
numerics, not as a theorem — the particles pass through **metastable states**:
plateaus where they sit in a handful of clusters before merging further.
They call the mechanism open.

The project's founding question was whether that plateau behaviour shows up in
*real, trained* models with everything the toy leaves out (multi-head
attention, MLPs, layer norm, learned weights). It does, consistently, across
several model families — and the more interesting observation was that trained
weights actively **resist** the collapse their own architecture drives, while
random-initialised weights of the same architecture do not. That contrast is
what the project's internal "Blog 1" wrote up, and everything since has been
built on it:

- **What is the geometry of the force?** The OV circuit's action splits, via a
  real Schur decomposition, into an attractive and a repulsive subspace, and a
  large part of what looks like translation is rotation (Phases 2, 2b, 2d).
- **How much of it is training, and when does it appear?** The project moved
  from a multi-architecture GPT-2 / BERT / ALBERT study to Pythia checkpoints,
  where the *same* model can be watched across training (Phases 1, 1c, 2 rerun;
  `PREDICTIONS.md`).
- **Can the particle language say what mechinterp says, without going through
  natural language?** Attention is a transport plan — row-stochastic, queries
  onto keys — so an induction head *is* a near-Monge map and a relay *is*
  transport by fixed displacement. Phase 7 and its sub-phases test whether that
  reading does explanatory work: whether it predicts something a rival account
  would not (`p7_motifs/design-7.md`, `PROJECT.md` §3.29).

The plain-English version of this story, with no cross-references, is
[`OVERVIEW.md`](OVERVIEW.md).

## 2. Where the work stands

A one-paragraph summary as of 2026-09-17; `STATE.md` is the authority and
will be newer than this.

- **The programme is the particle / optimal-transport reading** (`PROJECT.md`
  §3.29, a user decision on 2026-09-13). Induction heads are the *current
  instance*, not the object; the induction thread (Phases 7 / 7d / 7e / 8) is
  a large body of exploratory measurement on `pythia-410m` and `pythia-70m`
  that is now past diminishing returns. The next objects for the same pass —
  *why did this form, what were the mechanisms, what happened after* — are
  SAE features and the clusters themselves.
- **The e-value audit is finished** (§3.36, §3.40, §3.43–§3.45, completed
  2026-09-19). All **39** registered predictions examined across five units,
  and **zero e-values** — the ledger is empty, honestly. Its result is not a
  list of things left to run: **wherever the inputs exist, the blocker is a
  decision nobody has taken**, and none of the four costs a forward pass —
  β's scale convention, `P-T1`'s wording, `P6-R2`/`R4`'s exchangeable unit,
  `P-S1`'s matched-k clustering.
- **`CLAIM-C`'s hard-stop gate has now been run, three times, and refused
  three different ways** (§3.41, §3.46): a metric dead in every arm because
  `hdbscan` was never a declared dependency and the writer forged zeros in its
  absence; then a floor eight prompts cannot reach; then a homogeneity
  correction tabulated only to twelve prompts, which extending the battery to
  twenty invalidated. Each fix revealed the next, and **not one of them was a
  fact about the phenomenology.** The arms it needed — `gpt2-large` and
  `pythia-1.4b`, never run before — exist now, twice over.
- **Parked, not closed:** `P-I5`'s joint permutation null — two
  head-comparison controls failed to discriminate an induction head from two
  uninvolved heads, and a single-head diagnostic explained why the readout
  cannot see constant substitution (§3.31–§3.34); its statistic was corrected
  to an intersection-union test on 2026-09-17.
- **Just run:** the only *designed* particle intervention in the project,
  §2.5's isometric path on `L7H8` (§3.35) — an exact isometry whose two
  endpoints have identical singular values but different causal effect, so
  read/write alignment carries weight beyond the spectral sign.

## 3. Which document answers which question

The repository has many top-level documents because each one is the answer to
a different question, and the questions recur. Reading order for a fresh
session is `PROJECT.md` → `INDEX.md` → the phase you are about to touch.

| Question | File |
|---|---|
| What is this, in plain English, no cross-references | [`OVERVIEW.md`](OVERVIEW.md) |
| **Where does the work stand right now, what is blocking** | [`STATE.md`](STATE.md) — **read first** |
| **What went wrong before, and the rule it produced** | [`LESSONS.md`](LESSONS.md) |
| **What is registered, how do I reproduce a number, the full history** | [`PROJECT.md`](PROJECT.md) |
| Which phase lives in which directory; what is archived; what is referenced but absent | [`INDEX.md`](INDEX.md) |
| The working agreements (handoff cadence, PR size, when to scan the literature, checking math) | [`CLAUDE.md`](CLAUDE.md) |
| The six claims and what adjudicates each | [`claims/CLAIMS.md`](claims/CLAIMS.md) |
| What is pre-registered, its null, its falsifier, its instrument | [`PREDICTIONS.md`](PREDICTIONS.md) (prose), [`claims/registry.json`](claims/registry.json) (machine-checked) |
| Which predictions may carry an e-value, and why | [`claims/EVALUABILITY.md`](claims/EVALUABILITY.md) (generated, current state only) |
| Which phase carries which prediction, its gate, and what evidence stands behind it | [`claims/EXPERIMENTS.md`](claims/EXPERIMENTS.md) (generated) |
| The per-phase literature review, all sixteen phases | [`docs/LITERATURE.md`](docs/LITERATURE.md) → `<phase>/lit-N.md` |
| How each null was built — the thirteen dated construction passes | [`claims/EVALUABILITY_LOG.md`](claims/EVALUABILITY_LOG.md) |
| The adjudication ledger and per-claim E | [`claims/FALSIFICATION.md`](claims/FALSIFICATION.md) (generated; empty as of this writing) |
| The method plan — CI, the Popperian layer, the particle bridge — and its dated construction log §6a–§6za | [`POPPER_PLAN.md`](POPPER_PLAN.md) |
| The mathematics, phase by phase, and the six recurring failure patterns | [`MATH_INDEX.md`](MATH_INDEX.md) → each phase's `math-N.md` |
| Spectral analysis, optimal transport, and the "second spectrum" argument | [`MATH_SPECTRAL_OT.md`](MATH_SPECTRAL_OT.md) (symbolically checked, §7.4 of `PROJECT.md`) |
| The standing rules | `tools/lint_repo.py` docstring (history: [`archive/UPDATE_PLAN.md`](archive/UPDATE_PLAN.md) §6) |
| Paper / blog sketches and the prior art each sits against | [`PUBLICATION_IDEAS.md`](PUBLICATION_IDEAS.md) |
| Why a phase is built the way it is | `<phase>/design-N.md` |
| A phase's current state, opening with its registered predictions | `<phase>/status-N.md` |
| Dated one-offs (literature scans, the provenance audit, the CI baseline, deleted branches) | [`archive/docs/`](archive/docs/); old → new paths in [`archive/MOVED.md`](archive/MOVED.md) |
| What changed and when | `git log` |

Two conventions worth knowing: many `.md` files open with an
`<!-- filename -->` comment so they identify themselves when pasted; and
`PROJECT.md`'s §3 is numbered in the order things were *learned*, not in file
order — §3.36 sits above §3.35; `docs/index/PROJECT.idx.md` lists them.

## 4. Repository layout

```
Mets/
├── README.md, OVERVIEW.md, PROJECT.md, INDEX.md, CLAUDE.md   # the front matter (§3)
├── PREDICTIONS.md, POPPER_PLAN.md
├── MATH_INDEX.md, MATH_SPECTRAL_OT.md, PUBLICATION_IDEAS.md
│
├── core/                      # shared library — the particle schema, metrics,
│                              #   Pythia loading, RoPE, Schur/OV machinery, nulls,
│                              #   interventions, the e-value kernel, adjudication
├── p1_mstate_tracking/        # live phases: one directory each (§5)
├── p1b_hemisphere/
├── p1c_frames/
├── p2_eigenspectra/
├── p2b_imaginary/
├── p2d_operator_activation/
├── p6_subspace/               # only the rebuilt P6-R2/R4 null lives here now
├── p7_motifs/                 # the mechinterp/particle bridge
├── p7d_redundancy/            # the causal-ablation line (a different programme
│                              #   from p7_motifs — INDEX.md says how to tell them apart)
├── p7e_consolidation/
├── p8_scale_ladder/           # the same measurements across Pythia sizes
│
├── claims/                    # the falsification ledger — tracked deliberately
│   ├── registry.json          #   39 predictions, frozen wording, evidence paths
│   ├── CLAIMS.md              #   the six claims
│   ├── EVALUABILITY.md        #   generated by tools/render_evaluability.py
│   ├── EVALUABILITY_LOG.md    #   the construction diary
│   ├── FALSIFICATION.md       #   generated by tools/render_falsification.py
│   ├── audits/                #   dry runs on inputs whose answer is known
│   ├── calibration/           #   calibration artifacts for each null
│   └── adjudications/         #   (empty — nothing has been adjudicated)
│
├── tools/                     # one-off scripts: registry checks, calibrations,
│   ├── run/                   #   the real-model sweeps and runners (sweep.sh, curve.py, …)
│   └── math_checks/           #   seven sympy scripts, 47 checks, exit non-zero on failure
├── tests/                     # 107 modules; every module carries a tier marker (§8)
├── scripts/check.sh           # what CI runs, runnable locally
├── .github/workflows/         # ci.yml (gating) and smoke.yml (nightly)
│
├── archive/                   # Phases 3, 4, 5, 5b, 5c, 6 — frozen, not collected
├── docs/                      # scans, audits, baselines
├── data/                      # ALL generated bulk, git-ignored by `*` except the
│   └── analysis/*.py          #   builder scripts (tracked); their .json outputs are not
├── results/                   # 132 GB PILOT grid — DO NOT DELETE (PROJECT.md §5.2)
└── requirements/, pyproject.toml, pytest.ini
```

`.gitignore` is a **whitelist**: everything is ignored except `.py`, `.md`,
config, CI, shell, and `claims/**/*.json`. When a new config file mysteriously
will not commit, `git check-ignore -v <path>` is the thing to run.

## 5. The phases

Phases are the project's **instruments**, and are numbered in the order they
were built. They are not renamed to match the claim taxonomy — that would break
every artifact stem and test fixture to buy nothing (`POPPER_PLAN.md` §C3).
Each live phase has a `design-N.md` (why), a `status-N.md` (state, opening with
its registered predictions), and usually a `math-N.md` (the derivations).

### Live

| Phase | Directory | What it asks | State (2026-09-17) |
|---|---|---|---|
| 1 | `p1_mstate_tracking/` | Do metastable plateaus appear in trained models? Energy, Fiedler value, effective rank, per layer | Complete; the Pythia-410M checkpoint pilot ran (27 checkpoints × 8 prompts). `CLAIM-C`'s replication gate is built and calibrated; `tools/score_claim_c.py` now runs it and it **refuses** — the arms it needs are not on disk |
| 1b | `p1b_hemisphere/` | Cone collapse vs bipartition — the Fiedler axis | Complete |
| 1c | `p1c_frames/` | The quantitative theory: $T_{\rm eff}$, the $\gamma_\beta$ ODE, frames, Wendel/hull duality, spherical designs | Implemented and validated on synthetic data; **not yet run against Pythia** |
| 2 | `p2_eigenspectra/` | The OV mechanism: real Schur decomposition into attractive/repulsive subspaces, the V-score | Complete; the Pythia rerun is done and the 19-step registered sweep is on disk |
| 2b | `p2b_imaginary/` | Rotation: the imaginary channel, Henrici departure, the LN Jacobian | Complete |
| 2d | `p2d_operator_activation/` | The source paper's *hypotheses* checked: gradient-flow condition, operator-conditioned rank | Implemented, validated on constructed operators; **not run**, blocked on 1c by design |
| 7 | `p7_motifs/` | The bridge: mechinterp phenomena as particle **motifs** — typed-edge interaction tables, the motif alphabet, `P-I1` (induction as a two-stage `relay`) | `P-I1` ran end to end and scored INSUFFICIENT (not falsified, not validated). The co-location frame was retired for a construction-level circularity (§3.10); `P-I5`, the frame's own differential test, is parked |
| 7d | `p7d_redundancy/` | Which heads hold the induction regime on `pythia-410m`, when each formed, how they interact (single- and pairwise ablation) | Q1–Q3 answered, both open axes closed. The `L5H2` puzzle closed on MLP 6's *active* self-repair (§3.20–§3.25) |
| 7e | `p7e_consolidation/` | Whether the redundancy set collapses into one head (`L11H14`) | Gate measurement answered: the aligned core can consolidate; `L11H14` cannot be folded in. No surgery performed |
| 8 | `p8_scale_ladder/` | The 7d/7e measurements repeated across Pythia sizes so `n = 1` becomes a population | 70m rung run. Invariants 2 and 4 replicate, 5 does not, 6 needs a ceiling-immune instrument. **1b and 1.4b are reserved** for a registered prediction |

Phase 6's directory survives at the top level only for the rebuilt
`P6-R2`/`P6-R4` null (`p6_subspace/r2_r4_null.py`); its original study is in
`archive/`. `7a`/`7b`/`7c` are labels in `PROJECT.md` §3.14.1 with no directory.

### Archived — `archive/`, moved 2026-08-22

All pre-Pythia, all GPT-2 / BERT / ALBERT. Not maintained, not imported by
anything live, not collected by pytest. **Archiving the code does not retract
the findings**; `archive/README.md` states the policy once.

| Phase | What it found |
|---|---|
| 3 `p3_crosscoder` | **Null.** Sparse-crosscoder decoder directions align with V at chance (0.484 / 0.501), both models |
| 4 `p4_mstate_features` | **Not null.** A dense low-rank AE recovered V-alignment for ALBERT — "sparsity was the confound" |
| 5 `p5_single_mstate_analysis` | One cluster end to end, 6 models; carries the tuned-lens skip-to-output note |
| 5b `p5b_manifold_steering` | Built and tested, never run |
| 5c `p5c_unclustered` | Docs only. Trained models route attention *toward* unclustered tokens at 1.6–2×, sign-flipped against random weights |
| 6 `p6_subspace` | LDA-alignment inversion (0.887 imaginary vs 0.067 real repulsive), unresolved, two live explanations |

Two more branches carry work that exists nowhere else — Phase 1d (clusterer
comparison) and a cross-phase visualisation CLI — and are listed in `INDEX.md`
"In flight on other branches" so their age does not get them deleted.

## 6. The discipline: predictions, nulls, e-values

Geometry research on neural networks is easy to fool yourself with, so a large
share of the engineering here goes into not believing things too early. The
apparatus has four layers, and CI enforces the parts of it that can be
enforced.

**Predictions are registered before the code that tests them exists.**
`claims/registry.json` holds every prediction with its `h0`, `h1`, `falsifier`,
`instrument`, and the commit and date it was registered at. Wording is
frozen on registration; a change is a dated addendum, never an edit. Each
entry also carries the **phase join** — `phase`, `experiment`, and a `gate`
(`module:function`) that CI resolves against the tree — and two **evidence
paths**, `calibration_record` and `real_run_record`, each a git-tracked path
or `null`, so "this null is built and calibrated" is a file CI can check
rather than a claim in prose. `claims/EXPERIMENTS.md` is the generated
phase → prediction → gate → evidence view; `claims/EVALUABILITY.md` is the
per-prediction one.

**Every prediction names exactly one of six claims** (`claims/CLAIMS.md`):
`H-RESIST` (trained weights resist collapse), `H-TRANSFER` (it is a property of
trained transformers, not GPT-2-large), `H-EMERGE` (resistance appears at
circuit-formation events), `H-BUDGET` (a bounded dimensionality budget),
`H-OPERATOR` (collapse is attributable to stated operator conditions), and
`H-BRIDGE` (mechinterp constructs are particle-dynamical objects). A claim's
evidence is the product of e-values from *its* predictions and nothing else.

**Every prediction is in one of three states** (`claims/EVALUABILITY.md`):

- **`e-value`** — a valid null exists and its p-value is calibrated under H0.
  May contribute to a claim's product. 15 as of this writing.
- **`needs-null`** — testable, but most are *threshold comparisons* (a number
  against a number, with no distribution behind it — a decision rule, not a
  test). 21.
- **`measurement`** — no valid null exists and forcing one would manufacture
  evidence; the honest output is a number with an interval. 3.

The reason for the classification is arithmetic, not philosophy: one invalid
null in a product voids the Type-I guarantee for every other factor, silently.
`core/adjudication.py` *refuses* to emit an e-value for the latter two states
rather than emitting a neutral one — the project's standing rule 4, **refuse
rather than degrade** (`archive/UPDATE_PLAN.md` §6).

**Adjudication is sequential and anytime-valid.** The e-process follows POPPER
(Huang et al., 2025): a claim is supported when its accumulated $E \geq 1/\alpha
= 20$, and "null accepted" is not an outcome the machinery can express. As of
this writing **no prediction has been adjudicated** — twelve nulls are built
and calibrated on known-answer dry runs (`claims/audits/`,
`claims/calibration/`), one has been run on real artifacts, and
`claims/adjudications/` is empty. `FALSIFICATION.md` says so in as many words.

The order to build a null in — floor first, then the known-answer dry run,
then real artifacts — was distilled from thirteen construction passes in which
four kinds of defect kept recurring, and is written up in `EVALUABILITY.md`.
`P-I1` is the cautionary instance: a pairing null that was near-degenerate on a
36-head tie coset, so its p-value moved from 0.14 to 0.89 between 50 and 100
replicates on the *same* observed statistic.

## 7. Getting started

### Dependencies

Three tiers, declared in `pyproject.toml` and mirrored in `requirements/`:

| tier | installs | for |
|---|---|---|
| `base` | numpy, scipy, sympy | everything in `core/` that is torch-optional by design |
| `test` | base + pytest | the gating CI tier |
| `heavy` | test + torch, transformers `<5`, matplotlib, scikit-learn | real forward passes, clustering, figures |

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel; the PyPI default is the 4.9 GB CUDA build
pip install -r requirements/heavy.txt
pip install -e . --no-deps
```

`transformers` is pinned `<5`: on 5.x GPT-NeoX moved its rotary parameters
into `config.rope_parameters`, `core/rope.py`'s default then fires silently,
and it reports the wrong rotary dimension for `pythia-410m`. Python `>=3.10`;
the development machine runs 3.14 with torch 2.13 CPU.

### Environment

```bash
export HF_HOME=$PWD/data/hf              # the mirrored Pythia revisions
export METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1                   # the development machine runs offline
export HF_HUB_DISABLE_XET=1
```

`METS_REPO` and `METS_DATA` are the only two path overrides the run scripts
honour, and both must be the canonical path (the runners assert `sys.prefix`,
so a bind-mount alias fails the interpreter check). There is deliberately no
scratch-volume variable: paths that encode transient infrastructure fail
silently when the infrastructure changes.

### Data

Nothing under `data/` or `results/` is in git, and the numbers in the
documents were measured against roughly 300 GB of it: 33 mirrored
`pythia-410m` revisions (51 GB), the Phase 1/2 sweeps (118 GB), the 19
interaction tables of Phase 7 (6 GB), and the 132 GB PILOT grid under
`results/` that covers 27 checkpoints appearing nowhere else. A fresh clone
can run every test tier and every `tools/math_checks/` script without any of
it; reproducing a *measured* number needs the relevant sweep on disk, and
`PROJECT.md` §7 lists which producer writes what.

### First command

```bash
./scripts/check.sh gate     # tier 0 + 1, ~40 s
```

If the gate is green the tree is consistent: the registry validates, the
generated documents are in step with it, the ledger recomputes to what it
claims, and the pure test tier passes with torch genuinely unimportable. If it
fails on a `sha256` mismatch, a module carrying a record's hash was edited —
`PROJECT.md` §7.2 says how to rewrite the record; it is a chore, not a bug.

## 8. Testing and CI

Tests are partitioned by **what a runner must have installed**, and each
module's tier was measured rather than assigned — a module is `pure` only if
its whole test set passes with torch, transformers, scikit-learn and
matplotlib all made unimportable.

| tier | marker | needs | where it runs |
|---|---|---|---|
| 0 | — | nothing (stdlib only) | `check.sh lint`; CI on every push |
| 1 | `pure` | numpy, scipy, pytest | `check.sh iso`; CI, **gates merge** |
| 2 | `smoke` | real torch, tiny HF models, network | `smoke.yml`, nightly and on demand |
| 3 | `deps` | the heavy tier importable, no downloads | `check.sh all` |
| — | `heavy` | real run artifacts on disk | never in CI |

Tier 0 is where the project's own rules live: `tools/lint_repo.py` encodes the
standing rules as machine checks (orphan modules, unmarked test modules,
hand-synced constants, stale status lines, unlabelled thresholds);
`tools/check_registry.py` validates the registry and the pre-registration gate
(registration commit precedes adjudication commit — the check that carries the
Type-I guarantee); `render_evaluability.py --check` and
`render_falsification.py --check` fail when a generated document has drifted
from the records it summarises; and `core.adjudication --verify` replays every
claim's e-process from the committed records rather than trusting stored
numbers.

`scripts/check.sh` is the *only* command list — CI calls it rather than
reimplementing it, because the first two red CI runs were exactly that kind of
divergence (`pytest` vs `python -m pytest`; a stub bug invisible on any machine
with real torch). Tier 1 runs with the heavy modules **shadowed by packages
that raise `ImportError`**, not merely uninstalled, for the same reason.

The gate on the current tip: **2332 passed / 5 skipped / 45 deselected**.
`archive/docs/CI_BASELINE.md` records what the suite did before CI existed — zero
tests collected — so that green means something.

## 9. Reproducing a number

The rule is that every number in a committed document is measured on the
development machine and has a named producer. `PROJECT.md` §7 is the full list
with timings; the shape of it:

```bash
./scripts/check.sh all                    # adds the deps tier, ~2:15

bash tools/run/sweep.sh                   # the registered 19-step Phase 7 sweep; resumable
python tools/run/curve.py                 # curve.json — the artifact that gets diffed (§7.1)
python3 -m tools.run.behavioural --write  # the behavioural induction series
python3 -m tools.run.relay_null           # P-I1's degree-preserving null
python3 -m tools.score_p_i1               # P-I1's p-value

for f in tools/math_checks/*.py; do python3 "$f"; done   # 47 sympy checks, seconds
```

Three records hash `core/changepoint_colocation.py` or
`p7_motifs/formation_gate.py` and must be rewritten when either changes; the
gate fails loudly when they are stale, which is intended (§7.2).

Two traps the machine has actually set, recorded so they are not re-learned:
`source .venv/bin/activate` can succeed and hand you the wrong interpreter if
the repo has moved (check `sys.prefix`, never `VIRTUAL_ENV`) — this cost a
Phase 7 checkpoint computed against the wrong library; and "no output yet" is
not evidence a background job died — check `pgrep`, and write the pattern as a
real ERE.

## 10. How work is organised

The habits are in [`CLAUDE.md`](CLAUDE.md), written down because each was
forgotten at least twice:

- **`STATE.md` is overwritten when a unit of work closes** (`CLAUDE.md`,
  Stop), not in a sweep at session end. Sessions end abruptly; that file is
  what the next one starts from. Phase detail goes in the phase's `status-N.md`.
- **PRs open at natural boundaries** — an invariant read, a defect fixed, a
  runner parametrised — sized for a reviewer, not for a commit count. Every PR
  targets `main`, never another PR's branch (`CLAUDE.md`, Git and PRs); open
  ones are listed in `STATE.md`. CodeRabbit does not trigger on its own for
  this repo, so each PR gets an `@coderabbitai review` comment, and every PR
  gets a `/challenge-pr` review.
- **Literature scans at two triggers only:** when a phase opens, before its
  `design-N.md` freezes the constructions; and before an entry lands in the
  registry, since registration freezes the wording. Scans live in `docs/` and
  the phase's `literature-N.md`.
- **Closed-form math is checked mechanically** with `sympy` in
  `tools/math_checks/`, and each check states what it does *not* prove. Four
  of the six corrections owed to the source math were the kind these catch.

Three standing rules from `archive/UPDATE_PLAN.md` §6 that shape most of the code: a
quantity that appears in a report is persisted; every data-dependent fallback
records the branch it took; and a threshold not derived from a distribution is
labelled *placed*, not *calibrated*, in the code next to the value.

Two further constraints are easy to trip over. **SAEs are an object of study,
never an instrument** — `core/DESIGN_dual_reading.md` forbids SAE/LRAE features
in any measurement path. And **`pythia-410m` is spent** for registration purposes
(`check_registry` rule 3): a prediction may not be adjudicated on an artifact
the project measured before the prediction was registered, so nothing observed
there can be registered after the fact. `pythia-70m` is the exploration rung;
the 1b and 1.4b rungs are reserved for a prediction that names them first.

## 11. References

- B. Geshkovski, C. Letrouit, Y. Polyanskiy, P. Rigollet. *A Mathematical
  Perspective on Transformers.* arXiv:2312.10794. The particle model, the
  collapse theorems, and the metastability observation this project began from.
- K. Huang, Y. Jin, R. Li, M. Y. Li, E. Candès, J. Leskovec. POPPER —
  agentic sequential falsification. arXiv:2502.09858. The e-process and
  relevance-checking discipline behind `claims/`.
- S. Biderman et al. *Pythia: A Suite for Analyzing Large Language Models
  Across Training and Scaling.* arXiv:2304.01373. The checkpoint suite every
  live phase runs on.
- The induction-head literature and the verified scans that position this
  project against it: `p8_scale_ladder/literature-8.md`,
  `archive/docs/literature_scan_2026-09-10.md`, `archive/docs/literature_scan_2026-09-13.md`.
- `MATH_SPECTRAL_OT.md` §8 and `PUBLICATION_IDEAS.md` carry the wider
  reference lists.
