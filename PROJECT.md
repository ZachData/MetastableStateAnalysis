<!-- PROJECT.md -->
# PROJECT — the living state of this repository

The file to read first, and the one to keep current. It answers: what machine
this runs on, where the work stands, what is blocking, what has been registered
and may not be re-decided, and how to reproduce anything.

**It is not a session diary.** What changed and why lives in `git log`, and the
reasoning behind a construction lives in `POPPER_PLAN.md`'s numbered sections.
This file carries only what a fresh session needs in order to start working,
and every number in it is measured on this machine.

| | |
|---|---|
| Branch | `claude/rescaler-cache-identity-test` — nothing merged, no PR open |
| Last updated | 2026-09-08 |
| Structural map | `INDEX.md` — which phase lives in which directory, and what is archived |
| Method and construction log | `POPPER_PLAN.md` §6a–§6t |
| Pre-registered predictions | `PREDICTIONS.md`, `claims/registry.json` |
| What can carry an e-value | `claims/EVALUABILITY.md` |

---

## 1. Start here

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf
export METS_RESULTS_DIR=$PWD/data/phase12
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

./scripts/check.sh gate     # 2270 passed / 5 skipped / 30 deselected, ~40 s
```

If the gate is green the tree is consistent. If it fails on a `sha256` mismatch,
a module carrying a record's hash was edited — see §6.3, it is a chore and not a
bug.

### Resume here (2026-09-09)

- **Git is clean and everything is pushed.** `claude/rescaler-cache-identity-test`
  is up to date with origin. The 13-commit batch was split into **three stacked
  topic branches**, all pushed, each gate-green; PRs are **not yet opened** (no
  `gh` CLI — web UI):
  | PR | branch | base |
  |---|---|---|
  | 1 | `claude/p-i1-build-null-score` | `main` |
  | 2 | `claude/spectral-dissipation-infra` | PR1 branch |
  | 3 | `claude/induction-programme-stage012` | PR2 branch |

  `https://github.com/ZachData/MetastableStateAnalysis/compare/<base>...<head>?expand=1`.
  PR3's tree is byte-identical to `claude/rescaler-cache-identity-test`'s tip.
- **The repo's `github_key` is DEAD** — GitHub rejects it (`Permission denied
  (publickey)`). Push with the default `~/.ssh/id_ed25519`: plain
  `git push origin <branch>`, no `GIT_SSH_COMMAND`. The old instruction in this
  block was wrong and cost a debugging pass.
- **§3.12 is the live block, and it supersedes §3.11's registration advice.**
  The "repulsiveness is a population baseline" premise was measured against the
  wrong reference class, and the matched-norm control is degenerate. **Nothing
  is registered; `claims/registry.json` is untouched.**
- **The Stage 3 entry was drafted and is NOT registrable as drafted** (§3.12 block C —
  three defects, all decidable before a forward pass). The draft and its
  decisions memo were session-scratchpad only and are **gone** if that session
  is gone; §3.12 blocks C and D carry everything needed to rebuild them.
- **The block-E diagnostics HAVE RUN** (§3.12 block F, 2026-09-09,
  `data/analysis/induction_diagnostics_7b.py`). Three of four changed the
  design: the S-flip is exact and writable back (F1); the stable object is the
  **top-2 subspace**, not the top-1 direction, and there is a violent rotation
  at 512–2000 that §3.11 missed (F2); and `L2H10` is spectrally
  indistinguishable from `L7H8` with **opposite-signed** causal copying, which
  makes it the control head and cuts against the spectral frame (F3).
- **All 7b diagnostics have run** (§3.12 blocks F, G, H). Net effect: the
  Stage 3 entry §3.11 asked for is **not the one the measurements support**.
  `−Mᵀ` alone confounds the `S` sign with a read/write swap (G2), no
  spectral-*sign* quantity identifies the copier across 112 heads while every
  *concentration* quantity does (G6), the weight-identified direction does not
  survive whitening though the concentration does (H2), and the top-1 direction
  is not stable across checkpoints (F2).
- **NEXT ACTION: draft §6x around GAIN CONCENTRATION, not a named direction**,
  as the full **2² factorial** `{M, Mᵀ, −M, −Mᵀ}` (all four isometric; main
  effects clean, interaction aliased with the role swap), controlled against
  `L2H10`, with the pre-run prediction from G6 recorded: **`−Mᵀ` should not
  destroy copying, and `M` vs `Mᵀ` is the informative contrast.**
- **Both follow-ups have run** (H1-REVISED, H2-CONFIRMED). The composition is
  **not** K-specific — `W_Q` and `W_K` converge onto one subspace (principal
  cosine 0.21 → 0.83) so the score cannot separate the paths; the timing and
  magnitude stand, and the **V arm does** discriminate, so it is composition
  into the *attention* pathway. Whitening is confirmed by a split-half control
  (halves agree at 0.990 with each other, both ~0.22 against raw).
- **NEW, and the strongest single result (§3.12-I):** `L7H8`'s static QK
  operator becomes **symmetric** — 0.50 (random baseline) → **0.956**, layer-7
  median stays 0.520, **rank 0 of 16 from step 2000 on**. The first weights-only
  quantity that cleanly identifies the head; it *derives* block C's "near-normal
  matcher"; and it makes "matching kernel" a measurement rather than a metaphor.
  **Answered in §3.12-J: it tracks MATCHING, not induction**, and block I's
  claim 1 is corrected there. `L5H2` (a pure *positional* matcher) never becomes
  symmetric — rank ~310 of 384 throughout — so symmetry marks **content**
  matching; and `L2H10` reaches symmetry rank 3 with a **negative** OV copy
  effect, so matching and copying are separable. Two halves: QK = symmetry
  (sharp, 11 of 384), OV = gain concentration (weak). **Induction is both.**
- **The §3.13.3 position-axis check has run and came back NEGATIVE** (§3.13.3,
  §3.12-K): the effect is not position-concentrated (37 of 96 positions carry
  half the mass) and `r*_SVD = 1` is robust — 0.824 all-positions against 0.791
  / 0.858 on the concentrated / dilute halves. **The 82 % is not an artifact of
  averaging.** Two things it found instead: `j = 0`, where induction *cannot*
  fire, is the only position with a negative effect (−0.048) and the lowest
  attention (0.779) — the first end-to-end validation that `second_copy_nll`
  measures what it should; and attention is flat at 0.90–0.93 while ΔNLL varies
  fourfold, the QK/OV dissociation on a third axis.
- **The S/A factorial is RETIRED before registration** (§3.12-M). Its own pilot
  killed it: `−M` is off-scale at 5.13× ablation (KL 0.97 vs 0.09) so additivity
  fails, and the role swap that §2.4.3 proved inseparable from any sign contrast
  **destroys copying on its own** (`Mᵀ` = 0.94× ablation), so the S-sign effect
  can only be measured on a floor. §3.12-L's rank result shows no other points
  exist, so it is not repairable. `L2H10` is also not usable as a control (§M4).
- **What replaced it is a stronger result: ALIGNMENT is the mechanism.** `Mᵀ` —
  same singular values, same `‖·‖_F`, same eigenvalue moduli, only the read and
  write subspaces exchanged — destroys copying as completely as deleting the
  head. Gain concentration (G6) is a **marker** of which head copies; it does
  not carry the copying. G6's own pre-run prediction is falsified, and that is
  the finding.
- **The graded perturbation now exists** (`MATH_SPECTRAL_OT.md` §2.5): with
  `M = UΣVᵀ` and any Stiefel path `γ` from `U` to `V`, `M(t) = γ(t)Σγ(1−t)ᵀ` is
  an **exact isometry at every `t`** — same singular values, same `‖·‖_F`, rank
  exactly `k` — with `M(0)=M`, `M(1)=Mᵀ`, and `M(½)` symmetric PSD. Explicit
  `γ` = polar retraction of the chord. It sweeps the head from 100 % repulsive
  through 100 % attractive and back **at constant singular values**, which is the
  causal test of the spectral frame §3.12-D wanted and the four corners could not
  give. A second family `U R Σ Vᵀ`, `R ∈ O(k)`, fixes both subspaces and moves
  only the read/write correspondence, with a Haar null whose unit is the draw.
- **A literature check ran first** (§3.12-N) and changes what may be claimed: QK
  symmetry is known at the population level (Saponati et al.) — our tail result
  is the complement, not a duplicate; the symmetric/skew split has a published
  mechanistic reading (**filtering vs routing**) that names §3.12-J; the
  512–2000 window is **established in the literature** and must stop being
  implied as ours; and **we have never computed Elhage's copying score**, which
  is a different matrix from the one every "100 % repulsive" claim rests on.
- **The copying score has run (§3.12-O) and CONFIRMS §3.11 emphatically.**
  `L7H8`'s token-basis copying score is at or below zero at every step, falling
  to **−0.094 (rank 328 of 384)** at 143000 while its causal `ΔOV_nll` grows
  ~6×. The model has plenty of real copiers (max +0.723) — **all in layers
  9–20, downstream**. Two unarranged consistency checks passed (`L9H8`'s
  negative effect ↔ negative score; `L9H9` the one induction head that both
  copies and sits in the copier band), and the LN caveat is discharged to the
  fourth decimal.
- **THE LIVE QUESTION IS NOW A THREE-STAGE CIRCUIT.** `L7H8` has the largest OV
  causal effect in its layer *and* is an anti-copier by the token test — which
  **falsifies the standard account for this head on the standard account's own
  measure**. Reading: `L5H2` (positional match) → `L7H8` (content match, writes
  something that is not token identity) → layers 9–20 (token copying). §3.11 may
  have been calling `L7H8` "the copier" when it is **upstream** of the copier.
- **The three-stage reading is FALSIFIED (§3.12-P).** No sub-additivity
  anywhere — three copiers are **super**-additive, `L11H14` most sharply
  (alone +0.187, `L7H8` alone +1.107, both **+2.275**), which is redundancy
  between parallel paths, not mediation; and the sign is conservative because
  §3.12-M5's compressive readout biases toward *sub*-additivity. Composition
  gives no specific pathway either: on both Q and V a **non-copier control beats
  every copier** (`L11H0` Q rank 0, `L9H15` V rank 4). Without the per-path
  controls this would have read as support.
- **The puzzle is sharper, not resolved.** `L7H8`'s OV write is causally
  enormous (+1.107, ~2× baseline NLL), is not token-identity copying, and does
  not route through the copiers. **The obvious gap: every composition score in
  §3.12 is head→head, and the MLPs — the majority of the parameters — have never
  been measured.**
- **Five probes run (§3.12-Q); three negative, and the survivor is geometry.**
  MLPs: **negative** (best z +0.57, most layers below median). Composed
  `L5H2→L7H8` circuit: **more** anti-copying (−0.0997 vs −0.0496). Offset: L7H8
  attends **93.4 % at exactly `j`**, so the repo's non-standard convention is
  right for this head — and copying from there would return the *current* token,
  not the successor. What survives is **residual-stream geometry**: ablating
  `L7H8`'s OV shifts *every* logit by ~4 nats and removes **16 %** of the final
  residual norm, against <0.4 % for ordinary heads.
- **FRAMING CORRECTION: `L5H2` has TWICE `L7H8`'s OV effect** (ΔNLL +2.227 vs
  +1.107). §3.11's "largest of all 16 layer-7 heads" is true but has been read
  too broadly — `L7H8` is not the largest OV effect in the circuit; its own
  prev-token partner is. And QK ablation is worth 56 % of OV ablation on NLL,
  which §3.11 never measured (it read attention only).
- **NEXT, in order:** (1) close Q3's open caveat — sweep `‖OV‖_F` against
  `Δ‖resid‖` across heads, to rule out norm-proportionality before the geometry
  reading is leaned on; (2) the §2.5 isometric path on `L7H8`; (3) then §6x.
- **§3.14 records the organisation**: three objects (7a population / 7b
  mechanism / 7c formation), the author's queued **7d case-study programme**
  (follow all three circuit stages across every checkpoint, look for phase
  correlations and particle dynamics, then repeat over many circuits and ask
  whether the dynamics correlate *across* cases), and the **three code defects**
  found this session — all reporting-only, none touching a p-value.
- **`data/analysis/*.json` are git-ignored** — the batch outputs
  (`induction_rank_sweep_s*`, `induction_qk_sweep_s*`,
  `induction_subspace_characterize_*`, `induction_developmental_series`,
  `ov_per_head_series`, `dissipation_v2_series`, `dissipation_sublayer_series`)
  are on disk but not in git; rerun their `.py` producers (§7) to regenerate.
- **`POPPER_PLAN.md` §6x is still unwritten** (§6w points to it). A
  doc-consolidation pass would move the §3.11 + §3.12 design into it.

### The machine

| | |
|---|---|
| Repo | `/run/media/system/WDS_500/Mets` (NVMe, `/dev/nvme0n1p1`, 458 GB) |
| venv | `<repo>/.venv` — Python 3.14.7, torch 2.13.0+cpu, transformers 4.57.6, numpy 2.5.2, scipy 1.18.1 |
| CPU / RAM | 16 cores, 31 GB |
| Free | 95 GB on WDS_500, 440 GB on HDD_1TB |

### The tree

Everything generated lives under the repo, on one root. `data/` is git-ignored
by `*`.

```
Mets/
├── data/                            # all generated bulk
│   ├── hf/                 51 GB    # HF_HOME — 33 mirrored pythia-410m revisions
│   ├── phase12/           118 GB    # METS_RESULTS_DIR — phase 1 and phase 2
│   ├── phase7/            6.1 GB    # the 19 interaction tables
│   ├── analysis/                    # curve.json, formation_series.json
│   ├── logs/
│   └── superseded/phase7_float32/   # 1.1 GB, pre-float64 tables
├── results/               132 GB    # the PILOT grid — §5.2, DO NOT DELETE
└── tools/run/                       # sweep.sh, curve.py — tracked
```

`METS_REPO` and `METS_DATA` are the only two overrides. There is deliberately no
`METS_VOL`: it named a VM scratch volume, which is the class of path that
encodes transient infrastructure and fails silently when the infrastructure
changes. Both run scripts derive everything from `METS_REPO`.

`transformers` is pinned `<5`. On 5.x GPT-NeoX moved rotary parameters into
`config.rope_parameters`, `core/rope.py`'s `rotary_pct` default then fires, and
it reports `rotary_ndims=64` where pythia-410m rotates 16.

### Traps this machine sets

**`source .venv/bin/activate` can succeed and give you the wrong interpreter.**
`activate` carries the absolute `VIRTUAL_ENV` recorded at creation. If the repo
has moved, it prepends a directory that does not exist, sets the variable, and
returns 0 — and `python` falls through `PATH` to whatever else is installed (here
a conda env at `miniforge3/envs/mets`, Python 3.10 with a pre-4.45
transformers). This cost a phase-7 checkpoint computed against the wrong library
with nothing in the artifact to record it: the phase-7 manifest stores
`git_sha`, `hf_revision` and `seeds`, but **no library versions**.
`tools/run/sweep.sh` now asserts `sys.prefix` and the torch/transformers
versions rather than trusting activation. Check `sys.prefix`, never
`VIRTUAL_ENV`.

**"No output yet" is not evidence a job died.** The check is `pgrep`, not the
log — and write the pattern as a real ERE, since `pgrep -f "a\|b"` matches
nothing and reports success. Two writers on one temp path can replace a good
table with a corrupt one.

Both of the other traps this repo has hit are now guarded in code with tests:
the phase-1/phase-2 reuse selector identifies each phase by a file only that
phase writes (`tests/test_run_scripts.py`), and `tools/recompress_tables.py` no
longer globs its own temp file (`tests/test_tools_recompress_tables.py`).

---

## 2. Where the work stands

Active work is **Phase 7** — the mechinterp/particle bridge. **The live thread
is the bottom-up induction programme (§3.11)**: the co-location / `relay`-motif
frame (`P-I1`) hit a construction-level circularity and was retired (below,
§3.10, `POPPER_PLAN.md` §6w); it was replaced by isolating one induction head
(`L5H2 → L7H8`) and characterising its circuit directly. As of 2026-09-08
Stages 0–2, the QK half, and a generalisation batch have all run — everything
**exploratory, nothing registered**. **2026-09-09 (§3.12) reopened the design**:
the OV repulsive collapse is a model-wide developmental phase with its floor in
`CLAIM-B`'s own anchor window, §3.11's baseline argument used the wrong
reference class, and the Stage 3 control needs rebuilding. Next action is
§3.12's diagnostics, not a registration. §3.11 and §3.12's dated blocks are the
detail; §§3.6–3.10 below are the retired-frame history, kept as the
construction log.

`P-I1`, induction-head formation as a two-stage `relay` motif tracked across the
checkpoint axis, ran end to end and scored **INSUFFICIENT** — not
falsified, not validated — at both a 50- and a 100-replicate null (§3.6, §3.7).
The K = 100 rerun (§3.7) settled the sensitivity question: the verdict is
stable but **the p-value is not** (0.14 at K = 50, 0.89 at K = 100, same
observed statistic) — the pairing null is near-degenerate on a 36-head
change-centroid tie coset, so P-I1's arm returns the verdict but cannot
quantify how insufficient. `INDEX.md`'s phase table is still accurate for
everything else.

**Two analysis tracks, run 2026-09-05/06** (`docs/`, §3.8): a co-location panel
putting the Phase 1 / Phase 2 global observables on the P-I1 19-step axis
(`data/analysis/colocation_panel.*`), and the dissipation-identity
decomposition — Tier A (`data/analysis/dissipation_series.json`) and Tier B
(`data/analysis/dissipation_sublayer_series.json`, the exact attn/FFN split +
per-head roll-up). **Tier B's headline**: the per-head attention-dissipation
series breaks §3.7's tie coset and `|d_attn_repulsive|` co-locates with the
behavioural induction rise at a *stable* p ≈ 0.004 (CO-LOCATES) — but this is
**exploratory** (three anchors tried, no differential falsifier registered
first), so it is a hypothesis to register and re-test, not an adjudication.
`claims/adjudications/` stays empty.

**The co-location programme was stopped on 2026-09-07 and NOT registered**
(§3.10, `POPPER_PLAN.md` §6w). `relay` membership requires attractive-channel
edges defined by `U_pos` — the same OV Schur projector whose sign split the
proposed prediction would measure — so the obvious entry would have
"confirmed" at `p_less` = 0.018 a claim its own head-selection presupposes.
Stripping the filter moves into the *other* tautology `find_relays` guards
against, so the circularity is a property of the motif's construction. Work
has pivoted to a bottom-up interventional programme (§3.11).

**Open, in rough priority order** (for the next session):
1. **The bottom-up induction programme** (§3.11) — isolate one induction head,
   find the minimal rank `r*` that carries induction, characterise its shape,
   then perturb it. Replaces the population/co-location frame entirely: no
   projector-defined population, so no §3.10 circularity, and `n = 1` is
   sound because the null randomises over *subspaces* rather than units
   (`claims/EVALUABILITY.md`'s own unused advice). **Stages 0–2 have run,
   plus a generalisation batch** (§3.11, dated blocks). Settled: the
   representational picture — OV core 100 % repulsive, non-normal but not
   outlier, no token-identity copy structure — is **universal** across the top
   8 behavioural induction heads and stable across development; `r*_SVD ≪
   r*_Schur` holds across the whole trained regime. The causal OV→copying
   effect is **concentrated in L7H8** (~10× any other). The **QK half is done**
   and is the spectral opposite — the matcher is a ~12-dim high-eigenvalue
   near-normal invariant subspace (`r*_Schur` 12 < `r*_SVD` 32, SVD below the
   random control). **Pick up at: the four diagnostics of §3.12 block E** —
   §3.11's registration advice is superseded by §3.12. The reference class was
   wrong ("100 % repulsive" is 9/9 against *the induction heads*, but 0.435 →
   0.109 of the 384-head population at matched step), the matched-norm control
   is degenerate, and the drafted Stage 3 entry has three defects decidable before a
   forward pass. Diagnostics first, then the §3.12-D **S-flip** entry, then
   **Stage 3**. `claims/registry.json` unchanged.
2. ~~Violation-restricted subspace split (dissipation v2)~~ **— done
   2026-09-08, `status-2.md` item 5 ~resolved.** The clean per-particle
   version (`v2_attn_pos_*` in `tools/run/dissipation_sublayer.py` →
   `data/analysis/dissipation_v2_series.json`) does **not** reproduce Phase
   2's `frac_repulsive` decay: the energy-weighted repulsive share of the
   positive first-order term at ΔE>0 boundaries stays 0.6–0.9 across the
   trained regime. So `frac_repulsive` (a count with a hard `>0.5` per-
   violation threshold) falls because marginal violations drift across the
   line, not because the energy leaves the repulsive subspace. Nothing
   further for the dissipation runner here.
3. **The pythia-70m dense-onset sub-study** (§3.9) — new registered ground,
   fills the 512 → 1000 gap, would let §3.8's ODE-regime finding be checked at
   4-step resolution. Now also the natural second model for anything §3.11
   produces.
4. **Phase 2's runner needs a manifest** (`_write_run_manifest`) —
   `docs/results_provenance_audit_2026-09-05.md` §3.1.

*(The former item 3, "per-head OV Schur projectors", is **done** — the
machinery already existed in `p2b_imaginary/head_circuits.py` and the sweep is
`data/analysis/ov_per_head_series.json`. See §3.10.)*

**Committed 2026-09-08, on `claude/rescaler-cache-identity-test`, not yet
pushed / no PR.** Commits `b44c3e9`..`HEAD` (10): the `head_spectrum` sign
split, the dissipation runners + v2, `ov_per_head`, the induction programme
Stages 0–2 + generalisation batch, the provenance docs, and the PROJECT /
POPPER / status-2 sync. `./scripts/check.sh gate` was green before the batch
(2270 passed / 5 skipped). The tracked new code is under `tools/run/` and
`data/analysis/*.py`; every `data/analysis/*.json` series is git-ignored (§6)
and reproducible via §7.

**The registered 19-step sweep is complete.** All 19 interaction tables are on
disk under `data/phase7/`.

**The degeneracy that blocked `P-I1` has cleared.** On the twelve-step CLAIM-B
grid every head's change centroid was one number, so the pairing null permuted a
constant and the attainable floor was 1.000. The five registered log-spaced
fills inside (1000, 54000) fixed it:

| `relay_owner` | heads scored | distinct centroids | span (log-step) | sd |
|---|---|---|---|---|
| `tag_writer` | 102 | 68 | 3.8898 – 4.9439 | 0.2813 |
| `matcher` (registered) | 116 | **79** | 4.1604 – 4.9439 | 0.2374 |
| `both` | 122 | 86 | 4.1604 – 4.9439 | 0.2641 |

The relay counts behind it:

| step | relays | ex-`repeated_tokens` | heads (matcher) |
|---|---|---|---|
| 0 – 2000 | 0 | 0 | 0 |
| 4000 | 15,030 | 5,563 | 9 |
| 8000 | 232,568 | 83,659 | 25 |
| 16000 | 509,646 | 216,528 | 46 |
| 32000 | 1,176,478 | 582,796 | 63 |
| 54000 | 2,560,483 | 1,008,553 | 80 |
| 143000 | **2,407,556** | **1,465,052** | **114** |

**Two things that must not be skipped when this is scored.**

*The series is not monotone.* The total relay count FALLS from 54000 to 143000
while heads carrying relays rises 80 → 114 and the ex-`repeated_tokens` count
keeps climbing — the signal spreads across heads and away from the one
combinatorially-loaded prompt while the raw total drops. `change_profile`
rectifies, so that decline lands in `reverse_change_mass` and will inflate
`noise_mass_share_estimate` on a series whose reverse motion is real structure.
That field is documented "REPORTED, NEVER SCORED"; this is the case that earns
the distinction.

*"79 distinct centroids" is not 79 classes.* It is **77 singletons, one class of
three, and one class of thirty-six** — 31% of the heads still put their change in
a single interval. Harmless at 116 heads and not harmless at forty; see §3.2.

---

## 3. `P-I1`: built, run, and scored — INSUFFICIENT (2026-09-04)

**The relay-count null did not exist through 2026-09-03.** `formation_gate`
requires the series to be the excess above a null envelope, and
`core/qk_offset_null.py` computes that for the **QK antisymmetry statistic**,
not for relay counts. `formation_curve.assert_gate_ready` refused the raw
series, correctly. §3.1–§3.4 below is the construction log, kept as it happened
rather than rewritten now that §3.6 has the answer.

`claims/EVALUABILITY.md` prescribes the order — compute the attainable floor,
name what the statistic degenerates on, check what the measurement grid
contributes, and only then build the control. **All three steps before the
control are done.** `POPPER_PLAN.md` §6t is the write-up.

* **Step 2.** Across the 8 battery prompts at step 54000 the raw relay count
  against the prompt's own induction-pair supply runs **r = +0.9958** — 99% of
  the cross-prompt variance is the prompt's combinatorics, not the model's
  circuitry. Excluding `repeated_tokens`, +0.8908. Nothing else is close:
  n_tokens −0.39, n_same_content −0.36, n_distinct_tokens −0.79.
* **Step 3.** §2's table is the answer: the grid contributed the entire previous
  failure and the fills fixed it.
* **Step 1.** `claims/audits/p_i1_attainable_floor.json`. Two findings, below.

### 3.1 The gate cannot score the axis the pipeline builds

`formation_curve_payload` takes its head axis from the **behavioural** series,
dense over all 384 heads (24 × 16), and zero-fills the relay side. But
`paired_colocation_arm` calls `change_profile` on every unit with **no per-unit
skip**, and `change_profile` refuses a series with no rise. 116 heads carry
relays and **268 never do**, so the arm refuses on the first all-zero unit and
`p_value_p_i1` returns **no p-value at all**. On the 116 forming heads the
identical input emits.

The message names none of it — "the series has no rise anywhere in the sweep",
no arm, no head index, no unit count. Pinned as it is, in
`tests/test_p_i1_attainable_floor.py`.

**Fixed, both halves, 2026-09-04 — the author's decision was both, not
either.** Pre-filter the axis to the 116 heads that carry a relay anywhere in
the raw sweep (a static, pre-registrable population — §3.4's null still runs
on the raw series before subtraction, so nothing here depends on the null),
**and** give the arm a per-unit skip for whatever residual heads have zero
above-null EXCESS once §3.4's null is subtracted (a head can be in the 116 and
still have the null absorb its entire signal). `paired_colocation_arm` and
`p_value_p_i1`/`adjudicate_p_i1` now take `skip_no_rise: bool = False` —
default off, CLAIM-B untouched, `p7_motifs/formation_gate.py` — dropping a
unit only on `change_profile`'s "no location to measure" refusal specifically,
reporting the count as `n_skipped_no_rise` on the arm's own record and naming
it in every refusal the arm can still raise afterward. The real scoring call
is `tools/score_p_i1.py`, which pre-filters to the forming axis and passes
`skip_no_rise=True`.

### 3.2 The pairing arm's floor has two halves

Permuting units within a class of equal change locations leaves
`-mean|ca - cb[p]|` exactly unchanged, so every pairing ties a coset of order
`prod(m!)` and no input can express a p below `prod(m!) / n!`. The arm reported
`1 / n_draws` alone. At nine of ten units sharing one location it reported
**0.000500** against an attainable **0.100000** — 200×, above α, emitted with no
refusal. Seven of ten tied is 0.00139 and emits legitimately: **the halves cross
within two units.**

`core.changepoint_colocation.pairing_floor_report` now owns both halves and the
arm refuses on the max. On the real head set the tie half does not bind (116
heads, tie floor 1e-148); it binds on the set a relay-count null *leaves*.

### 3.3 What that constrains — the point of doing it first

A relay-count null turns the series into an above-null excess, and a head whose
excess stops rising leaves the scored set. So the null chooses `n_units`, and
`n_units` with the tie structure chooses the floor.

| survivors | max tied | tie floor there |
|---|---|---|
| 4 | 1 | 0.0417 |
| 6 | 4 | 0.0333 |
| 8 | 6 | 0.0179 |
| 12 | 10 | 0.0076 |
| 19 | 17 | 0.0029 |
| 20 | **19** | 0.0500 |

Not monotone: `k = n − 1` gives exactly `1/n`, so all-but-one-tied clears 0.05
from n = 20 upward and fails at n = 19. Full table in the record.

> **The relay-count null must leave at least four heads with a rising above-null
> excess, and among them no more than k sharing one change location.**

### 3.4 The null — built 2026-09-04, degree-preserving at the head level

The author's decision, walked through and registered rather than started from
the code: degree-preserving at the **head** level, not per particle.
`p7_motifs/relay_count_null.py`. `pair_type` and `offset` are pure facts about
where an edge points, given the prompt's tokenisation; `attractive_frac` /
`repulsive_frac` / `force_magnitude` / `weight` are facts about its force —
independent axes of the same edge-row. So the null is a payload shuffle: for
each (prompt, layer, head), draw `len(group)` DISTINCT positions uniformly at
random from the prompt's full causal pool and reattach each real edge's entire
force-derived payload to it unchanged, recomputing `offset`/`pair_type` from
the new position.

This holds `n_induction` fixed per prompt automatically (the pool and the
induction/strict/same-content candidate sets are properties of the prompt's
tokenisation alone, `PromptNullContext`, identical at every checkpoint and
replicate — no separate bookkeeping needed), and preserves each head's edge
count and its **entire** force distribution exactly, not just an aggregate like
"attractive fraction". Per-particle in/out-degree is NOT held fixed — a
heavier double-edge-swap configuration-model null was considered and not
chosen. The relay count itself, a two-edge composition rather than a single
masked edge, is scored by Monte Carlo — reshuffle, rerun
`find_relays`/`per_head_relay_strength` unchanged, K replicates → mean/sd —
rather than a closed form, to avoid re-deriving the composition's null
distribution by hand. 18 tests, including a planted-relay oracle and
calibration on a structureless table; caching the null's per-prompt grouping
(a ~6.8× speedup, needed to make the real run feasible) also caught a genuine
cross-prompt position-leak bug before it reached the real sweep.

Run over the real 19-step sweep, 50 replicates/checkpoint
(`tools/run/relay_null.py` → `data/analysis/relay_null_series.json`):

| step | raw relays | null mean | excess | excess / null |
|---|---|---|---|---|
| 0 – 2000 | 0 | 0 | 0 | — |
| 4000 | 15,030 | 2,968 | 12,063 | 4.1× |
| 8000 | 232,568 | 27,719 | 204,849 | 8.4× |
| 16000 | 509,646 | 64,176 | 445,473 | 6.9× |
| 32000 | 1,176,478 | 128,647 | 1,047,832 | 8.1× |
| 54000 | 2,560,483 | 346,008 | 2,214,479 | 6.4× |
| 143000 | 2,407,556 | 241,229 | 2,166,327 | 8.9× |

The raw count sits 4–9× the chance level at every formation-window checkpoint
— real excess above what the induction-pair supply and edge counts alone would
produce — and that excess is what §3.6's gate is scored on.

### 3.5 Done: the behavioural arm over the sweep

`tools/run/behavioural.py` → `data/analysis/behavioural_series.json`, run
2026-09-03 (`POPPER_PLAN.md` §6u). Pooled mean post-softmax attention on
induction pairs per (layer, head) per checkpoint, on the same pair set `run_7.py`
types the A side with, tokenisation verified token-for-token against each run's
`tokens.txt`. Cross-prompt convention registered by the author: **mirror the
relay side** — pool the seven non-`repeated_tokens` prompts, carry the
eight-prompt series beside it, never scored. **10,618** pooled pairs (44,809 with
`repeated_tokens`), asserted constant across all 19 steps.

The result: flat at `≈ 1/n_tokens` through step 128 (0 heads elevated), first
rise at 512→1000, sharp climb 2000–8000 — L7H8 peaks **0.0368 at step 4000**,
L6H0 peaks **0.0306 at step 16000**. **Non-monotone in the §2 shape**: leaders
recede (L7H8 → 0.0160, L6H0 → 0.0247 by 143000) while the elevated-head count
runs 0 → 14 (step 8000) → 9. Endpoint precondition clean both ends: step 0 all
384 heads at baseline, step 143000 has 7–9 heads clearly elevated. B leads A by
an interval or two on inspection; the co-location itself needs the gate, which is
blocked on §3.1 and §3.4.

**The floor record is rewired to it** (`schema_version` 2, same session).
`tools/p_i1_attainable_floor.py` arm A now pairs the real relay series against
`series_excl_repeated`, not the synthetic located rise it used before;
`b_side_is_synthetic` is `False` and `--check` verifies the input hash. The
dense-axis refusal is unchanged (`paired_colocation_arm` profiles the A side
first, so the 268 all-zero relay heads decide it regardless of B). What is new:
the 116 forming heads emit **p = 0.420** against the measured B side,
**mean_distance_log_step = 2.02** — on the raw count the two curves do not
co-locate per head, behaviour leading. Not P-I1's test (raw count, not the
above-null excess §3.4's null would produce) and partly a floor effect — the
relay count is structurally zero until step 4000 so its change can't be located
below log-step 3.6 — but it is the number the null now has to move.

### 3.6 The real result: `tools/score_p_i1.py`, p = 0.1414, INSUFFICIENT

`p_value_p_i1` on the real above-null excess series (§3.4) against the real
behavioural series (§3.5), pre-filtered to the 116 forming heads,
`skip_no_rise=True`:

| | |
|---|---|
| p_value | **0.1414** |
| p_reciprocal | 1.0 |
| verdict | **INSUFFICIENT** |
| n_units | 116 (`n_skipped_no_rise` = 0 — every forming head's excess still located a rise) |
| attainable_floor | 0.0005 |
| mean_distance_log_step | 2.018 |

**Barely moved from the raw-series number** (§3.5's 2.02): subtracting the
null rescales the curves' magnitude far more than it moves where each head's
change is located, at least at this replicate count. p dropped from 0.420
(raw) to 0.141 (excess) — real movement, and still nowhere near α = 0.05.

**Both endpoint failure modes are clear**, reported and entering no p-value
per §3.3: 0 of 116 heads are already above-null at step 0, and of the 2 heads
absent at step 143000, 0 had a high behavioural score. Neither disjunct of the
falsifier's second half fires.

**Not adjudicated.** `claims/adjudications/` is untouched — `tools/
score_p_i1.py` deliberately does not call `adjudicate_p_i1(..., adjudicate=
True)`. INSUFFICIENT is not RE-ANCHORS: the design did not fail, and nothing
here falsifies `P-I1`; it means the two curves' rises do not co-locate across
heads more than an arbitrary pairing allows, at the registered sweep and the
50-replicate null.

### 3.7 Answered (2026-09-05): the p-value moves 0.14 → 0.89, and the reason is structural

The K = 100 rerun ran end to end — **3 h 12 m**, matching the estimate — and
wrote `data/analysis/relay_null_series.json` (`n_replicates: 100`). All three
replicate series are now durable in the repo tree:
`data/analysis/relay_null_series_k50.json`, `…_k100.json`,
`relay_null_full_k100.log`.

| | K = 50 | K = 100 |
|---|---|---|
| `p_value` | 0.14143 | **0.89355** |
| `p_reciprocal` | **1.0** | 0.89305 |
| `mean_distance_log_step` | 2.017944 | 2.018009 |
| `n_units` / `n_skipped_no_rise` | 116 / 0 | 116 / 0 |
| `attainable_floor` | 0.0004998 | 0.0004998 |
| verdict | INSUFFICIENT | INSUFFICIENT |

**The p-value moved 6.3×. The observed statistic did not** — `mean_distance_
log_step` changed by 6.5e-5 (0.003%). So this is not the point estimate
shifting; it is the permutation null that the p is read against.

**Mechanism.** 50 more null replicates moved the above-null excess series by
< 0.06 % at every step (0 sign flips, largest single-cell change 91 relays
against millions). That nudged **77 of the 116 heads' change-centroids by
~1e-4 – 1e-3 log-step** — enough to move the observed `-mean|c_a − c_b|`
across the body of the permutation null. The tie structure itself did **not**
change: 79 distinct centroids both times, one class of **36** heads pinned at
log-step 4.9439 (rise entirely in the 54000 → 143000 interval), one class of
3, 77 singletons — exactly §2's "77 singletons, one class of three, and one
class of thirty-six". That 36-head coset dominates the null's spread, so the
permutation distribution of `-mean|c_a − c_b[perm]|` is a near-spike and the
observed value sits at its median. On a near-vertical CDF a 6e-5 shift in x is
a 0.75 shift in F(x). `p_reciprocal` collapsing from exactly 1.000 (pinned to
the atom) to 0.893 is the same tell.

**This realises §3.3's stated precondition as a failure.** "No more than k
heads sharing one change location" — 36 of 116 share one. §3.2's tie *floor*
is 1e-148 here and does not bind, but the tie-driven *atomicity* of the null
still makes the p-value a step function of an input that the replicate count
perturbs at the 4th–5th decimal.

**What is robust, and what is not.** Robust: the verdict (INSUFFICIENT at both
K), the observed mean distance (~2.018 log-step — the two families of rises do
not co-locate per head, nowhere near α), `n_units`, the floor. Not robust:
the p-value itself. **§3.6's p = 0.1414 is not a quotable number, and neither
is 0.8936.** P-I1's pairing arm can return the verdict but cannot quantify how
insufficient the evidence is, on this sweep.

**To get a stable p (not required — the verdict stands):** break the 36-head
coset. Either a denser sweep inside 54000 → 143000 so those heads' rises spread
across more intervals, or a B-anchor that is not centroid-tied the same way —
the per-head attention-dissipation series from
`docs/dissipation_checkpoint_axis_scoping.md` Tier B is the candidate, and it
is defined at every checkpoint rather than structurally zero before step 4000.

---

### 3.8 The two analysis tracks opened 2026-09-05 (context: while K = 100 ran)

Both are in `docs/` and `data/analysis/`; both build only on artifacts already
on disk. Neither adjudicates anything — measurement, with the provenance caveat
from `docs/results_provenance_audit_2026-09-05.md` §3.1 (the Phase 2 OV
projectors carry no `git_sha`; every producing module is unchanged since the
runs, but that is inference not record).

### 3.8.1 Co-location panel — `data/analysis/colocation_panel.{png,csv}`

The Phase 1 / Phase 2 global observables parsed onto the P-I1 19-step axis from
the committed run reports: energy-monotonicity violations (both phases),
`frac_repulsive`, `ov_frac_repulsive`, effective rank (normed + raw), raw λ₂
and the Fiedler deviation, beside the P-I1 behavioural and relay-excess series.
The reading: the transitions fire in a fixed order across one decade —
**128→256** energy break + first behavioural rise; **512** plateau-onset flips
weight→content, `frac_repulsive` hits 1.0, raw λ₂ starts its monotone fall;
**2000–4000** effective rank peaks (44.5) with the behavioural leaders; **4000**
the relay motif first appears (structural zero before). The relay count is the
*lagging* indicator — every other signal turns 3–4 checkpoints earlier and on
more points, which is why §3.7's pairing arm, anchored on the relay side, has
so little to locate against.

### 3.8.2 Dissipation identity, Tier A — `data/analysis/dissipation_series.json`, `dissipation_panel.{png,csv}`

`core/dissipation.py` evaluated per (step, prompt, layer) over the 19 × 7 grid ×
24 layer boundaries, from `activations.npz` (total dX) + the on-disk per-checkpoint
OV Schur projectors. No forward pass. `tools/run/dissipation.py`, ~22 min,
`max_subspace_sum_check = 2e-12` (the split is exact). β ∈ {1.0, 2.0}, sphere
frame. **Caveat, recorded in the artifact:** the subspace split projects the
*total* dX (attn+ffn), not the attention channel — the attention-only and
per-head versions need the Tier B sublayer-capture pass.

Four findings:

1. **The forward-Euler / ODE framing the project rests on is checkpoint- and
   layer-dependent, and mostly does not hold.** The relative linearisation
   residual `|ΔE − Σᵢ⟨Gᵢ,vᵢ⟩| / max(|ΔE|,|Σ⟨G,v⟩|)`:
   - **layer 0 is never in the regime** — residual ≈ 1.0 at every checkpoint
     (step size ~450 in raw coords). Any per-layer dissipation reading must
     drop layer 0.
   - deep layers (4–23) reach residual **~0.17 only at steps 2000–4000** — the
     ODE picture is decent *in the induction-formation window* — and degrade to
     ~0.87 both before (steps 8–256) and after (steps ≥ 16000).
   This is `MATH_SPECTRAL_OT` §5.3(d)'s "a large residual is a finding about the
   project's framing", quantified: it is large outside a narrow window.

2. **Step 512 is a triple co-location in the dissipation view** — same
   checkpoint as the energy break / plateau flip / Fiedler turn. The
   repulsive-subspace share of `|dissipation|` jumps 0.45 → 0.67 and **stays**
   0.63–0.76 for the rest of training; the gradient-flow alignment
   `mean cos(−G, v)` is at its least anti-aligned (≈ −0.01, i.e. motion most
   orthogonal to the energy landscape) with `frac_descending` at its peak 0.55.

3. **The layer motion is weakly anti-aligned with the energy gradient almost
   everywhere** — `mean cos(−G, v)` ∈ [−0.10, −0.01], most negative (−0.10) at
   the step-128–256 energy-break onset, drifting back to −0.08 late. Trained
   layers are, on average, very slightly *ascending* E_β — the activation-side
   reading of the monotone-energy break, and the measured counterpart of Phase
   2d's weights-only D1.

4. **Phase 2 open-item-5's `frac_repulsive` decay (1.00 → 0.56 over steps
   8000–143000) is NOT reproduced** in the total-displacement subspace split —
   the dissipation repulsive-share is flat ~0.64 from step 8000 on. So the
   reorganisation Phase 2 sees is either specific to the *attention* channel
   (Tier B would show it) or to the *violation* mass rather than the
   displacement magnitude. First thing for Tier B to resolve.

### 3.8.3 Dissipation identity, Tier B — `data/analysis/dissipation_sublayer_series.json`, `dissipation_tierB_panel.png`

`tools/run/dissipation_sublayer.py`, run 2026-09-06 (20 min, 133 forward passes
with sublayer capture, `max channel sum_check = 4e-15` — the attn/FFN split is
exact, and the per-head projection `Σ_h dX_attn[h] + bias = dX_attn` holds to
1e-5 at every layer). All 24 layers kept (author's call: fine for L0/L23 to have
unexplained behaviour in analysis, not fine to exclude them from the
measurement). β = 1.0.

1. **The `frac_repulsive` decay is NOT in the attention channel's displacement
   geometry.** The attention channel's repulsive-subspace share of `|dissipation|`
   is near-cancelling at the aggregate (Σ d_attn passes through ~0 several times)
   and does not track Phase 2's violation-based `frac_repulsive`. On the forming
   layers L8–23 it rises to 0.82 at step 2000, dips, returns to 0.85 at
   step 32000, drops to 0.54 at 143000 — a trajectory, but not the smooth
   1.00 → 0.56 decay. The FFN-channel projection nominally tracks it better
   (0.97 → 0.82 over 8000 → 143000) but projecting an FFN output through an OV
   subspace is not physically meaningful. **Net: still not localised to a
   displacement channel — the clean test is a violation-restricted subspace
   split (share of the *positive* first-order term, not of `|dissipation|`),
   which is a v2.**

2. **Step 512 co-location holds in the attention channel specifically** — attn
   `mean cos(−G, v)` crosses positive (+0.03) at step 512, its only positive
   value, against −0.11 to −0.13 through steps 2000–4000.

3. **The per-head co-location test — one stable CO-LOCATES, exploratory.** The
   per-head attention-dissipation series **breaks §3.7's 36-head centroid tie
   coset**: change-centroids are 116 singletons, no ties. Three candidate
   B-anchors were tried post hoc against the behavioural rise, on the same
   `p_value_p_i1` gate, `skip_no_rise=True`:

   | B-anchor (per forming head) | p_value | verdict | mean_dist (log-step) |
   |---|---|---|---|
   | `d_attn_total` (head's attn contribution to first-order ΔE) | 0.333 | INSUFFICIENT | 0.566 |
   | **`|d_attn_repulsive|`** (attn displacement projected on repulsive OV subspace) | **0.0040** | **CO-LOCATES** | 0.576 |
   | `-gfa_cos` (attn anti-alignment with the energy gradient) | 0.587 | INSUFFICIENT | 0.508 |

   The `|d_attn_repulsive|` result is **stable** where §3.7's relay-based p was
   not: deterministic on re-run, seed-stable (10 seeds, p ∈ [0.0015, 0.0045],
   sd 0.0009), head-jackknife-stable (drop any 1 of 116 → p ∈ [0.0005, 0.0065],
   0 flips above α), and not a magnitude tautology (centroid corr with the
   behavioural side r = 0.21; observed mean pairing distance sits at the 0.1st
   percentile of the label-permutation null). Mechanism reading: **as a head
   becomes an induction head, its attention output starts moving the residual
   stream along the repulsive (individuating) OV directions in a way that pushes
   on E_β, and that onset tracks the behavioural onset across all 116 heads.**

   **This is exploratory, not an adjudication.** Three anchors were tried and one
   was significant — multiple comparisons, and no differential falsifier was
   registered before running (POPPER_PLAN §C2's requirement). It is a
   **hypothesis to register and re-test**, e.g. on held-out heads or a second
   model, not a validated P-I1. `claims/adjudications/` stays empty. What it
   does settle: §3.7's "get a stable p by using a non-centroid-tied B-anchor"
   works — the anchor exists and the tie coset is gone.

---

### 3.9 External resource — the pythia-70m dense-onset run (`lora_ind`)

Logged 2026-09-06 from an inspection of a sister project. **Not consumed by any
Mets code yet.** This is a resource note plus where it could plug in.

### What it is

A **full-parameter** training continuation of `pythia-70m` from `step512` to
`step2000`, on an RTX 3080 (6.9 h, torch 2.11.0+cu130, `lora_ind` git
`1ebbe33`). Not LoRA, not that project's M1–M8 protocol — a plain
`GPTNeoXForCausalLM` (`attn_implementation="eager"`, bf16), AdamW (0.9, 0.95),
wd 0.01, grad-clip 1.0, micro-bs 2 × grad-accum 512 = a 1024-sequence batch,
`monology/pile-uncopyrighted` streamed and packed to 2048.

**The load-bearing detail is the LR schedule**: it *continues* Pythia's own —
peak 1e-3, 1430-step linear warmup, cosine to 0.1× over 143k. Step 512 sits
*inside* the warmup at 3.58e-4 and climbs to 1e-3 by ~step 1500, so the onset
window runs at the intended rate, not at peak LR (which is what a
default-configured continuation would do, and would make the transition an
artefact of the harness).

**It is not the Pythia trajectory.** Adam is cold-started at step 512, so it
diverges from published `pythia-70m` from step 1. These checkpoints are "a
clean, correctly-scheduled induction onset in the loss landscape near
`step512`", not "`pythia-70m` steps 528–852". The sister project's own spec
phrases every such result as *reachability, not developmental* — Mets would
inherit that constraint.

### On disk

`/home/iron/Desktop/lora_ind/data/retrain/cb25e3f6c2185c1e/` (67 GB):

| file | content |
|---|---|
| `ckpt/step_NNNNNN.pt` | **249 weight snapshots**, every 4 optimiser steps, 528 → 1860 (one gap, 1184 → 1528). Each a **plain HF `GPTNeoXForCausalLM` state_dict** — fp32, 281 MB, **no optimiser state, no RNG**. Loads with `core.lm_loading.load_causal_lm_from_state_dict`. |
| `probe.jsonl` | 381 rows, step 516 → 2000 every 4 steps: `pms` (prefix-matching score), `icl` (first−second copy NLL), `nll_first/second`, `train_loss`, `lr`. |
| `final/model.safetensors` | the step-2000 model = the sister project's checkpoint **B**. |
| `provenance.json` | `onset_step: 620`. **Its `retained_checkpoints` list (1528–1860) is wrong** — a double-resume reset the buffer bookkeeping. |

**Clean bracket: steps 528 → 852, 82 checkpoints, all present, no gaps**, onset
at 620 (PMS 0.019 @ 592 → 0.051 @ 620 → 0.36 @ 676 → 0.51 @ 700; ICL 0.03 →
1.86; loss declines smoothly, no LR kink). Session 1 ran 516 → 868 correctly;
only steps 856 / 860 are contaminated. Everything 864 → 1860 is from two resumes
that replayed different batches (visible as different `train_loss` at identical
steps) — usable as a rough long tail, not for anything tight.

**State as of 2026-09-06:** the local `lora_ind` repo is **very out of date**,
and a **re-probe is running now** — it re-scores PMS/ICL at `n_eval=512`
(the run used 128, the sister project's §5 wants 512) and raises rather than
logging a 0.0 when attentions come back empty. It emits `onset_bracket.json`,
which is **authoritative** over `provenance.json` for the bracket and the
contaminated steps. Wait for it before consuming the bracket.

### The sister project, in one paragraph

`lora_ind` = "Induction Bandwidth": the minimum-rank weight update that installs
a working induction circuit into a pre-induction `pythia-70m` checkpoint, and
whether the update's antisymmetric fraction φ differs between the
prefix-matching (QK) and copying (OV) halves. **Shared spine with Phase 2 / 2b**:
`M_QK = W_Q W_Kᵀ`, `M_OV = W_Oᵀ W_Vᵀ`, the `S + Λ` split, `φ(M) = ‖Λ‖²/‖M‖²` —
the same operator decomposition `p2b_imaginary` and `core/dual_reading.py` use.
Its gates G0–G2 passed (checkpoint A = step 512, B = step 2000, induction head
**L3H6**, previous-token head **L2H1**). **G3 — the positive control — failed**:
a generous-rank QK-only update did not reach criterion, and the run could not
distinguish "composition rule too restrictive" from "optimisation broken". M1–M8
(the actual rank sweeps) are blocked on it. **So there is no rank-r induction
subspace yet.**

### How / where it could plug into Mets

1. **Dense-onset sub-study — the thing the Pythia mirror cannot give.**
   `p2_eigenspectra/status-2.md` item 4 and §3.7 both note there is no released
   Pythia checkpoint between 512 and 1000. This run has one every 4 steps
   through the onset. It would let the §3.8 findings be checked at real
   resolution instead of 3 sparse points — in particular **§3.8.2 finding 1**
   (the forward-Euler residual "sharpens across 512 → 1000, sharpest at
   2000–4000"): with 4-step spacing the claim becomes testable rather than
   interpolated. `pythia-70m` is 6 layers / d=512 / 8 heads, so a full Phase 1 +
   Phase 2 + dissipation pass over all 82 clean checkpoints × the battery is
   cheap on CPU.

2. **A non-tie-coset behavioural / energy co-location axis.** §3.7's whole
   problem was 36 of 116 heads sharing one change-centroid because the axis is
   too sparse in the formation window. A 4-step axis through the onset spreads
   the centroids by construction — the co-location arm gets a well-conditioned
   null without needing §3.8.3's post-hoc anchor hunt.

3. **The real merge, once `lora_ind` unblocks G3.** When that project produces
   `r*_QK`, `r*_OV` and the φ signatures, Mets can project the rank-r induction
   subspace **in and out** of the residual stream at each dense checkpoint and
   re-measure the particle dynamics — the dissipation identity, the energy
   violations, the relay motif — with vs without the induction contribution.
   That is the concrete form of "isolate what induction does to the particle
   picture", and it is **blocked on `lora_ind` G3**, not on anything here.

### Cost of adoption, stated honestly

- **New registered ground.** Every Mets registered decision is 410m-specific
  (`REGISTERED_P_I1_SWEEP`, `P_I1_RELAY_OWNER`, the CLAIM-B grid, the battery
  tokenisation). A 70m dense run needs its **own** registered grid, forming-head
  axis and battery — no 410m number transfers, and this must be a labelled
  sub-study, not an extension of the 410m sweep.
- **Reachability, not developmental** (above) — any co-location result on this
  run generalises to Pythia only as far as the loss landscape near `step512` is
  representative.
- **No activations saved** — weights + scalar probe trace only. Adoption means
  pointing the Phase 1/2/7 runners at local `.pt` paths and re-running forward
  passes. `core.lm_loading.load_causal_lm_from_state_dict` already exists; a thin
  "checkpoint dir → (model, tokenizer)" adapter and a `REGISTERED_P70M_*` grid
  are the whole lift.
- **torch 2.11 (run) vs 2.13 (this venv)** — irrelevant for loading, the
  snapshots are plain state_dicts.

---

### 3.10 Per-head OV, and the circularity that stopped a registration (2026-09-07)

Full narrative in `POPPER_PLAN.md` §6w. What a fresh session needs:

**Nothing was registered.** `claims/registry.json` is unchanged,
`claims/adjudications/` is still empty. Every number below is post-hoc on
artifacts that already existed and may not be adjudicated on them.

**The instrument.** `p2b_imaginary/head_circuits.py` already had the per-head
OV machinery (`head_core`, `head_spectrum`, `sym_antisym_factors`,
`apply_factored`). It gained an energy-weighted sign split —
`repulsive_energy_fraction_core` / `attractive_energy_fraction_core` /
`repulsive_dim_fraction_core`, +7 tests — which is `MATH_SPECTRAL_OT` §3's own
proposed discriminator, energy-weighted rather than bulk-edge-restricted
because a bulk edge is a placed constant and a weighting is not.
`tools/run/ov_per_head.py` → `data/analysis/ov_per_head_series.json`
(45 min, weights only, no forward pass, no model load — every checkpoint's
per-head dense OV is already on disk as `ov_head<h>_layer_<l>`).

**It is calibrated**, which is the strongest null check in the sequence:
chance is 0.500 ± 0.033 (300 matched-shape random heads at the real 1024/64
geometry) and **step 0 measures 0.4982**. Not dominated by one eigenvalue
(top-eigenvalue energy share median 0.057, exceeds 0.5 nowhere in 384 heads).
The 21% of heads pinned at exactly 0.0/1.0 are pinned *structurally* — median
`min|Re λ|/|λ|` = 0.873, none under 1e-2 against an fp32 floor of ~1e-8.

**The trajectory is a real finding regardless of the circularity.** Every head
at chance through step 64; both populations swing to near-total repulsive
dominance by step 1000 (forming 0.989, non-forming 0.948); from step 2000 the
relay-carrying heads reverse to **0.156** while the rest hold at **0.701**.
The divergence opens in the formation window and grows monotonically.

**Two population facts that bear on `P-I1` and `P-I3` directly.**
The 116-head relay axis and the behavioural induction population are nearly
disjoint — **1 of the top 9** behavioural heads is on it, the relay axis sits
in layers 8–23 (mass 21–23) against the behavioural leaders in layers 1–10
(mass 6, 7, 9), and the relay axis's mean peak behavioural score (0.00536) is
*below* the off-axis mean (0.00618). `spearman(max relay excess, peak
behavioural)` = −0.230 (p = 0.013), **but the partial controlling for layer is
+0.004** and the mean within-layer ρ is +0.090 — so it is **no association
within layer**, not an inversion, and the raw negative is two opposite depth
trends multiplying. This is a more direct account of §3.6's INSUFFICIENT than
§3.7's tie coset: `P-I1` pairs two per-head series that are unrelated within
layer, over a population where the behavioural side sits near baseline.

**The summed-OV projector is a fiction, quantified.** The on-disk
`schur_repulse_layer_*` come from `ov_total = sum_h ov_per_head`, and the
summed value lands within 0.05 of only **21.2%** of the heads in its own layer.
So the Tier B per-head anchor (§3.8.3) projects each head's write onto a
subspace of an operator the model never forms.

### 3.11 The bottom-up induction programme — decisions taken before running

Replaces the co-location frame. Isolate one induction head, find the minimal
structure that carries induction, perturb it, then build outward. Machinery is
live: `p2_eigenspectra/head_ablation.py` (per-head OV ablation for GPT-NeoX),
`core/intervention.py` (`run_model_with_hook`, `next_token_kl`),
`head_circuits.py` (factored S/A surgery), `tools/run/behavioural.py`.

**Stages.** 0: target `L7H8` at step 4000 (peak behavioural 0.0368) and locate
its stage-1 prev-token partner. 1: rank sweep on QK and OV separately → `r*`.
2: characterise the `r*` subspace (Schur sign, φ, token-subspace alignment).
3: small variations at matched norm against a matched-norm random control,
joint behavioural + logit + geometry readout (`P-I5`'s registered shape).
4: repeat on the other elevated heads — generalisation *after* mechanism.

**Three decisions, taken now because taking them after seeing a curve would
void the guarantee** (§6l's timing argument):

1. **Both bases, and they answer different questions.** SVD finds `r*` — it
   measures gain, is Eckart–Young optimal for "minimum rank that carries the
   action", and orders unambiguously. Schur characterises what is *in* `r*` —
   it is the only one carrying a **sign**, so the attractive/repulsive
   differential falsifier cannot be posed in the SVD frame at all. Cost is not
   a consideration: 0.40 ms (SVD) and 0.99 ms (real Schur) per 64×64 core,
   ~25 s for the whole 19-step × 384-head grid. **Their disagreement is itself
   a registered outcome**: these cores are strongly non-normal (Henrici median
   **0.450**; rank for 90% of action 36.5 by SVD against 41.5 by eigenvalue
   ordering, median per-head gap 5, max 13), so `r*_SVD ≈ r*_Schur` says the
   induction-relevant part is near-normal and the eigenvalue picture is
   trustworthy, while `r*_SVD ≪ r*_Schur` says induction lives in high-gain
   non-invariant directions and the project's whole attractive/repulsive frame
   is measuring something other than what the head does — `MATH_SPECTRAL_OT`
   §5.3(d)'s "a large residual is a finding about the framing", reached from a
   second direction.
2. **`r*` is derived, not thresholded.** No "induction collapsed" constant.
   Report the score-vs-`r` curve and define `r*` as where it crosses the
   matched-norm control band — the same move that made
   `N_CONTROLS_PER_INDUCTION_HEAD` a frontier rather than a placed number.
3. **Readouts are paired to the operator, and getting this wrong measures
   nothing.** The behavioural induction score is *mean post-softmax attention
   on induction pairs* — a pure QK quantity — so **ablating OV cannot move it
   within the layer**. QK sweep → behavioural score. OV sweep → logit/copying
   effect (`next_token_kl`). Either → particle-geometry delta. A flat OV curve
   read against the attention score would be misread as "OV does not matter".

**Still to register before stage 2 reads anything**: the differential
prediction itself — particle account says the causally-identified induction
subspace is repulsive/individuating, standard account says it is a copier
(attractive, token-aligned). Neither is silent, so INVERTS can fire. Stating
it after seeing the `r*` subspace repeats §6w's mistake one level down.

### What has run (2026-09-07) — Stages 0–1 done, Stage 2 invalid, nothing registered

All exploratory. `claims/registry.json` and `claims/adjudications/` unchanged.

**Stage 0 — the circuit is `L5H2 → L7H8`.** From `attentions.npz` at step 4000,
no forward pass: `L5H2` is an overwhelming previous-token head (mean attention
at offset −1 = **0.895** against a 384-head median of 0.018, ~49×). `L7H8` is
the behavioural leader (§3.5). Textbook two-stage shape: prev-token head in L5,
matcher in L7.

**Stage 1 — OV half only, `tools/run/induction_rank_sweep.py` →
`data/analysis/induction_rank_sweep.json`.** QK half deferred: Pythia rotates
only `rotary_ndims = 16` of 64 head dims, so an `M_QK = W_Q W_Kᵀ` rank
truncation is not a truncation of what the model computes — needs a
RoPE-aware treatment. Readout is the **copying** side (decision 3): second-copy
NLL on repeated uniform-random sequences (`N_REP=96`, 8 seqs) plus KL from the
unablated model, through the full forward. Weights saved/restored around every
measurement; end-of-run restore check exact (`abs_diff` 0.0).

The OV effect on this head is **modest**: full OV ablation (`r=0`) moves the
second-copy NLL only **0.779 → 1.023** (KL 0.043). Fraction of that effect
recovered by a rank-`r` truncation, `(nll₀ − nllᵣ)/(nll₀ − nll_base)`:

| r | SVD | Schur | random control |
|---|---|---|---|
| 1 | **82 %** | 12 % | ~0 % |
| 2 | 85 % | 15 % | ~0 % |
| 6 | 93 % | 75 % | 16 % |
| 16 | ~99 % | 81 % | 32 % |
| 24 | ~99 % | 88 % | 52 % |
| 48 | 100 % | 97 % | 90 % |

**`r*_SVD ≪ r*_Schur`** — SVD rank 1 already carries 82 % and is flat past
r≈6; Schur needs r≈16 to match rank-1 SVD, and beats the random control only
modestly below r≈24. This is the **pre-registered branch of decision 1**:
induction's copying action lives in a **high-gain, non-invariant** direction,
and the attractive/repulsive (eigenvalue-sign) frame is not the natural
description of this head's OV — `MATH_SPECTRAL_OT` §5.3(d) reached from a
second direction.

**Stage 2 — redone 2026-09-08, `tools/run/induction_subspace_characterize.py`
→ `data/analysis/induction_subspace_characterize.json`.** (An earlier inline
Stage 2 was discarded: its copying readout was a direct logit attribution with
no `final_layer_norm` and no control heads.) This version reads copying
**causally through the full forward** — same readout as Stage 1, LN present by
construction — and calibrates every number against all 16 heads of layer 7 and
24 matched-Frobenius-norm random OV operators. Weights save/restore, end
restore check 0.0.

*L7H8's OV is the causally load-bearing half, by a wide margin.* Full OV
ablation moves the second-copy NLL **0.779 → 1.023** (ΔNLL **+0.244**, KL
0.043) — **the largest of all 16 layer-7 heads**, ~10× the layer mean (0.021 ±
0.059). So the mid-Stage-2 doubt ("is OV even the right half") is settled for
this readout: it is.

*`r*_SVD ≪ r*_Schur` holds under the causal readout too.* Rank-1 SVD carries
**82.0 %** of that ΔNLL (rank 1 of 16; layer mean 15 %), rank-1 Schur only
**11.9 %**. The top singular value holds 17.6 % of the OV Frobenius energy
(rank 2 of 16) — one unusually dominant gain direction, which is where the
action is.

*The subspace is entirely repulsive — and that turns out not to be the
description that matters.* Every eigenvalue of the 64×64 core has Re < 0:
`attractive_energy_fraction_core` is **exactly 0.0**, top |λ| has Re −0.165,
the rank-1 SVD mode's own eigenvalue is Re −0.177, the top-16 Schur subspace
is 100 % repulsive. But φ = 0.398 (rank 10/16, *below* the layer mean) and
Henrici = 0.353 (rank 8/16, *less* non-normal than a random operator's 0.71),
so L7H8 is not a spectral outlier in its layer. The `r*_SVD ≪ r*_Schur` result
says the copying action lives in a **high-gain, non-invariant** direction, so
sorting the operator by eigenvalue sign is not sorting it by what it does —
the repulsive sign is *true but not the mechanism*.

*And it is not a token-identity copier.* The direct copy score of the rank-1
mode (LN mean-scale folded, `s_in` 1.36, `s_final` 0.61, descriptive only):
the diagonal is the row-max for **1 token in 4000 — exactly chance** —
diag z-mean 0.03, diag-positive 0.514. No `W_E → W_U` diagonal structure.

**So Stage 2 cuts against both accounts.** The standard account (induction OV =
attractive, token-aligned copier) fails on both counts. The particle account
(repulsive/individuating) has the sign right but the wrong frame — the effect
is carried by an SVD gain direction, not an eigen-mode. What L7H8's OV
actually is, on this evidence: a **functional** copier (ablating its one
high-gain direction measurably degrades repeated-token prediction) that is
neither a **representational** copier (no token diagonal) nor an **eigen-mode**
(SVD, not Schur, is where the rank collapses).

**Caveats.** One head, one checkpoint. The causal ΔNLL / KL are solid; the
"not token-aligned" half leans on the approximate LN-folded copy score. The
per-head SVD/Schur fractions are only interpretable for L7H8 — it is the only
layer-7 head with a non-noise OV effect, so the others' ratios divide by
~0.01. The **QK half is now done** — see block C below.

`POPPER_PLAN.md` §6x is still unwritten — §6w refers forward to it; this §3.11
is currently the only home for the design.

### Generalisation batch + QK half — results (2026-09-08, exploratory, nothing registered)

`data/analysis/induction_developmental_analysis.py` →
`induction_developmental_series.json`. Two questions: does the Stage 2 picture
hold across development, and across the other behavioural induction heads?

**A. L7H8 OV rank sweep across all 19 axis steps.** Through step 2000 the OV
copying effect is ~zero (`ΔOV_nll` ≤ |0.014|; L7H8 cannot do induction yet) —
the rank fractions there are noise. From step 4000 it switches on and grows
monotonically: `ΔOV_nll` **+0.24 → +0.73 → +1.02 → +1.47** (steps
4000/8000/16000/54000), KL 0.04 → 0.57, slight pullback to +1.18 at 143000.
The `r*` story:

| regime | steps | `svd@1` | `svd@2` | `schur@1` | `schur@8` |
|---|---|---|---|---|---|
| formation | 4000–16000 | 0.82–0.96 | 0.85–0.99 | 0.12–0.49 | 0.77–0.93 |
| consolidation | 32000–143000 | **0.68 → 0.20** | 0.85–0.97 | 0.25–0.34 | 0.83–0.89 |

So the copying action is **one SVD direction during formation**, then spreads
to a **second** during consolidation (`r*_SVD` grows 1 → 2). Schur rank-1 is
weak throughout; Schur needs ~8 modes at every step. **`r*_SVD ≪ r*_Schur`
holds across the entire trained regime** — a stable developmental property,
not a snapshot. Caveat: late full-ablation KL is 0.57, so the rank-1-vs-2 gap
there is partly nonlinearity.

**B. Stage 2 on the top 8 behavioural heads** (L7H8, L7H0, L7H3, L7H12, L6H0,
L2H10, L9H9, L9H8, L1H15), each at its peak step, each calibrated against its
own layer.

*Universal — the representational claim generalises.* **Every** head's OV core
is 100 % repulsive (`attractive_energy_fraction_core` = 0.000, top-λ
repulsive, 9 of 9), φ ∈ [0.29, 0.46], Henrici ∈ [0.23, 0.53] (no spectral
outliers), and **none is a token-identity copier** (copy z ≈ 0, diag-is-row-max
at chance). The standard "attractive, token-aligned copier" account fails for
every behavioural induction head, not just L7H8.

*Concentrated — the functional claim does not.* The causal OV→copying effect is
L7H8's at scale (`ΔOV_nll` +0.244) against L9H9 +0.054, L1H15 +0.040, L6H0
+0.021, and the rest ≤ 0.01 or negative (L2H10, L9H8 negative — their OV is
not a copier). The "top behavioural heads" are ranked by attention pattern (a
QK quantity) and are mostly not OV copiers — decision 3, confirmed. Where the
OV effect is real (L7H8, L9H9, L1H15, L6H0) rank-1 SVD carries it (0.68–0.97)
and Schur rank-1 is weaker, so `r*_SVD ≲ r*_Schur` generalises there too;
L1H15 is the near-exception (Schur rank-1 0.76 vs SVD 0.83 — more normal).

**C. The QK half — `tools/run/induction_qk_sweep.py`, L7H8 across 8 steps.**
Rank-truncates the **static** (rows 16–63, non-rotary) QK operator — the
content-match half; RoPE carries the positional part and its 16 dims are left
intact — and reads the induction attention directly (mean post-softmax weight
on the repo's pairs: query `N_REP+t`, key `t`). Full static-QK ablation drops
L7H8's induction attention from **0.92 to 0.02**, so the readout bites.

The matcher is the **spectral opposite of the copier.** Once formed
(step 4000+, induction attn 0.92 → 0.95, stable):

| | `r*` (rank reaching ½ the effect) | low-rank basis vs random |
|---|---|---|
| **OV copier** | `r*_SVD` 1–2 · `r*_Schur` ~8–16 | SVD ≫ random ≫ Schur |
| **QK matcher** | `r*_Schur` **12** · `r*_SVD` **32** | Schur > random > **SVD (below random)** |

So the copy lives in a **rank-1 high-gain non-normal** direction; the match
lives in a **~12-dim high-eigenvalue near-normal invariant subspace**, and its
top-*gain* directions are worse than random. `MATH_SPECTRAL_OT` §5.3(d)'s
"the eigenvalue frame is or isn't the right description" resolves *per half*:
right for the matcher, wrong for the copier. The Schur<SVD pattern for QK
appears exactly when the match forms (step 2000→4000) and holds through 143000.
(Schur reordering is non-monotone above r≈16 on near-degenerate |λ|; the
r ≤ 16 Schur values are clean.)

*Across the other genuine induction heads (L1H15, L9H9, L6H0 — the heads that
both match on the repo-convention pairs and have an OV copy effect; the other
"behavioural leaders" do neither).* **L9H9** shows the same Schur ≫ random > SVD
matcher pattern as L7H8. **L6H0** (weakest matcher, 0.76) leans the same at low
rank but is noisy. **L1H15** is **generic** — SVD ≈ Schur ≈ random, ~40 of 48
dims needed, no spectral structure — and it was also the near-normal outlier on
OV (Schur rank-1 ≈ SVD rank-1). So L1H15's whole circuit is spectrally
unstructured; the QK dissociation holds for 3 of 4 but is not as clean as the
OV representational claim's 9/9.

**Consequence for the registration.** Two things are now population-level
baselines, not distinguishing features: "OV core 100 % repulsive" (every
behavioural induction head) and the **OV/QK spectral dissociation** (a generic
property of a copy+match circuit, plausibly). The Stage 3 differential
prediction has to be about the **rank-1 high-gain OV direction of L7H8**
specifically — under matched-norm perturbation does it act as an individuating
channel, or does copying survive (a copier the token-alignment test missed) —
not about repulsiveness or rank per se. Register that, then Stage 3.

*Superseded in part by §3.12 (2026-09-09): the "repulsiveness is a population
baseline" premise was measured against the wrong reference class, and the
matched-norm control is degenerate. Left unedited — the reasoning is the
construction log.*

### 3.12 The reference-class error, and what it opens (2026-09-09)

Exploratory. `claims/registry.json` and `claims/adjudications/` unchanged.
Nothing here is registered and nothing here is a p-value.

**A. The OV repulsive collapse is a model-wide developmental phase, not an
induction-head property.** Read off `data/analysis/ov_per_head_series.json`
(commit `08175a8`, weights-only, all 384 heads × 19 steps — the file was
computed for §3.10's per-head projector item and its population axis was never
looked at). Median `attractive_energy_fraction_core` over all 384 heads, and
the share of heads whose core is *exactly* 100 % repulsive:

| step | 0–64 | 128 | 256 | 512 | **1000** | **2000** | 4000 | 8000 | 16000 | 32000 | 54000 | 143000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| median attractive | ~0.504 | 0.481 | 0.385 | 0.094 | **0.000** | **0.000** | 0.001 | 0.026 | 0.066 | 0.134 | 0.225 | 0.392 |
| frac heads exactly 0 | 0.000 | 0.000 | 0.000 | 0.000 | **0.617** | **0.654** | 0.435 | 0.297 | 0.219 | 0.188 | 0.164 | 0.109 |

The whole model slides off the random-matrix baseline into a **fully repulsive
phase whose floor is steps 1000–2000**, then partially recovers. 1030 of 7296
head-steps are exactly 0.0 overall (14.1 %).

*Checked, not assumed:* `attractive + repulsive = 1.000` exactly at every step,
so this is not energy vanishing into an unclassified bucket — the `< 0` / `> 0`
split in `head_spectrum` leaves nothing at zero. And step 0's ~0.504 is the
**random-real-matrix baseline** (eigenvalues symmetric about the imaginary
axis), so the measurement carries its own control and the model demonstrably
moves off it.

**Two things this collides with.** The floor sits *exactly inside* `CLAIM-B`'s
registered anchor window (512–2000) — either a striking convergence or a shared
cause, and which one matters before either is scored. And Phase 2's activation-
level `frac_repulsive` decay (§2 item 2) runs the *other* way over the trained
regime while this weights-only quantity recovers; they are different
measurements and the shapes have never been put side by side.

**B. §3.11's exclusion of repulsiveness rests on a reference-class error.**
"OV core 100 % repulsive, 9 of 9 induction heads" was read as a population
baseline — but the population it was compared against was *the induction heads
themselves*. Against the 384-head population **at matched step**, exactly-0.0
runs 0.435 at step 4000 down to **0.109 at 143000**. `L6H0` peaks at 143000,
where only 11 % of heads are fully repulsive. The 9/9 is not a baseline.

This does **not** resurrect the observational claim as a registrable
differential — 9/9 with no control arm is still not a test. What it removes is
the *reason given* for excluding repulsiveness from Stage 3, which reopens the
**causal** test of it (block D).

**Before anything is built on A:** check the count version
(`repulsive_dim_fraction_core`) against the energy version, in case one dominant
eigenvalue carries it; and name weight decay as the leading alternative
mechanism, though the recovery after step 2000 argues against a pure-decay
story.

**C. The Stage 3 entry was drafted and is not registrable as drafted.**
A draft entry (H-BRIDGE, geometry-vs-copying at the rank-1 OV direction) was in
the session scratchpad, not in the repo. Three defects, all decidable before a
forward pass.

*Three things the gate taught us while writing this section, and they decide
where this discussion lives.* (1) The draft reused the **induction id that
`POPPER_PLAN.md` §6w already spent** on the co-location registration that was
*not* made; the next free one is one higher. (2) `tools/check_registry.py`'s
coverage rule fired the moment that id was named here — working exactly as
intended, since an id in a scanned file with no registry entry is the state the
registry exists to forbid. (3) It did **not** fire on §6w's use, because
`POPPER_PLAN.md` sits in `SCAN_EXCLUDE` beside the three generated files — so
the design narrative can burn an id invisibly to coverage, and no *uniqueness*
check spans the excluded files.

**Consequence, and it is structural rather than a nuisance:** `PROJECT.md`
cannot name an unregistered prediction id at all, so **the id-level design
discussion belongs in `POPPER_PLAN.md` §6x** — which §6w already points forward
to and which is still unwritten. This section is the summary; §6x is where the
candidate entry, its id, and its wording go before registration.

The three defects:

1. **The copy-matched control is probably unbuildable.** Random rank-1
   directions recover ~0 % at matched norm (§3.11 Stage 1), so reaching the
   observed ΔNLL needs hard rescaling, which drives KL up — matching a
   *copying* effect against a *destruction* effect. §3.11's own nonlinearity
   caveat (KL 0.57 late) bites here. This replaced one degenerate null with
   another.
2. **The geometry contrast has a mechanical component with no individuation
   content.** Perturbing `W_OV` along `v` moves every position the head writes
   to by a near-rank-1 update; the matched positions share attention structure
   *by construction*, so they move together and their pairwise distances change
   *less* than unmatched ones — contraction contrast in the wrong direction, for
   algebraic reasons. `P-ST1`'s "steering is a pure mean effect" one level over:
   settle it on paper, not by simulation.
3. **The direction is identified weights-only.** The top singular direction of
   `W_OV` ignores the residual-stream distribution; the 82 % is a causal
   measurement but the identification is not, and Stage 3 perturbs the
   identified object.

Also missing, and every other registry entry has one: a **precondition on the
pilot, computed before it runs**. Expected verdict distribution as drafted is
INSUFFICIENT-dominated, because the direction was *selected* for carrying the
copy effect and is then asked to beat a bar its own selection sets.

**D. The proposal that replaces it: flip the symmetric part.**
For `M = S + A` with `S = (M+Mᵀ)/2`, `A = (M−Mᵀ)/2`: `Re(λ)` is governed by `S`,
and `tr(SᵀA) = 0`, so `‖−S+A‖_F = ‖S+A‖_F`. **`S → −S` flips every eigenvalue's
real part at exactly matched Frobenius norm, by construction** — no rescaling,
no matching search, no tolerance. `p2b_imaginary/head_circuits.py` already does
factored S/A surgery.

- Standard account: copying depends on gain (σ and singular vectors), not on
  eigenvalue sign → flip `S`, **copying survives**.
- Particle account: individuation depends on repulsive character → flip `S`,
  **matched particles collapse and copying degrades**.

Both non-silent, so the falsification branch can fire; the matching problem and
the mechanical-geometry confound both dissolve. And it is the *causal* test of
repulsiveness, which is not a baseline even where the observational version is:
if every induction head is 100 % repulsive and flipping it changes nothing, the
repulsive frame is decoratively true and causally empty — a real registrable
negative. If flipping it destroys induction, the frame is load-bearing. **Either
outcome is informative**, which is the property the drafted entry lacks.

**E. Four absences, all cheap, none producing a p-value.**

1. **The composition has never been measured.** Grep confirms no K/Q/V-
   composition score anywhere in the repo — the only "composition" hits are the
   retired `relay` motif. Induction is `L5H2 → L7H8` and only the *endpoints*
   are characterised. Weights-only, milliseconds, and it yields a developmental
   series: when the **composition** forms against when each half forms. No §6w
   circularity, since composition is defined by weights, not by the projector
   whose sign a prediction would read.
2. **Rotated or spread?** `svd@1` falls 0.68 → 0.20 across consolidation while
   `svd@2` stays high; §3.11 reads this as spreading to a second direction.
   Equally consistent: the direction **rotated** and rank 2 tracks a moving
   target. Distinguish by top-singular-vector overlap between adjacent steps.
3. **The anti-copiers.** `L2H10` and `L9H8` have *negative* `ΔOV_nll` —
   ablating their OV *improves* second-copy prediction — while carrying the
   universal representational description. Stage 2's "representation is not the
   mechanism" in its sharpest available form, already computed.
4. **The 16 rotary dims were never examined.** The QK sweep truncates rows
   16–63 and leaves rotary intact; but `L5H2` is a *pure positional* head and
   "previous token" is positional. Block C describes the static half only.

**Order of work decided here:** the four diagnostics run *before* the entry is
written, because two of them (1 and 3 above, plus the input-whitening check in
C3) can change what a valid control even is. Registering first and diagnosing
after is the §6l timing argument pointed the wrong way — nothing here is a
p-value, so none of it spends the registration.

**F. Diagnostics run (2026-09-09) — `data/analysis/induction_diagnostics_7b.py`
→ `induction_diagnostics_7b.json`.** Weights-only, no model load, no forward
pass, ~1 min 43 s. All four returned, and three of them change the design.

*F0 — the U-shape survives the count check, and gains an ordering.* The
`repulsive_dim_fraction_core` count tells the same story as the energy version
— identical `frac heads exactly 1` at every step (0.617 / 0.654 / 0.435 / …),
so **the collapse is not one dominant eigenvalue**. What the disagreement shows
is *how* it happens: in the ramp, repulsive **energy leads repulsive dimension**
— 0.615 vs 0.562 at step 256, **0.906 vs 0.734 at step 512** — and the two
meet at 1.000 by step 1000. **The large eigenvalues go repulsive first and the
count catches up.** This is exactly the bulk-vs-outlier reading
`ov_per_head.py`'s own docstring said the disagreement was for; nobody had read
it.

*F1 — block D is exact, and feasible.* `(−S+A)` and `−Mᵀ` agree **bit for bit**
(`max|Δ| = 0.000e+00`) on all four heads. The flip preserves every singular
value to `≤1.3e-15`, the Frobenius norm to printed precision, and every
eigenvalue modulus to `≤1.1e-15`, while `attractive_energy_fraction_core` goes
**0.000 → 1.000** for `L7H8`, `L2H10`, `L9H8` (and 0.444 → 0.556 for the mixed
`L5H2`). The factored write-back `W_O' = −W_Vᵀ`, `W_V' = W_Oᵀ` reproduces it at
`1.2e-07` relative — the **fp32 storage floor** the OV artifact already carries
(`s[64]/s[0] = 4.9e-8`), i.e. exact to the data's own precision. So the Stage 3
control needs no rescaling, no matching search, no tolerance, and no new
machinery.

*F2 — "rotated or spread?" is BOTH, at different times, and §3.11 saw only the
second.* Overlap of `L7H8`'s top right-singular vector between adjacent steps,
with the top-2 subspace's principal cosines beside it:

| window | top-1 overlap | top-2 principal cos | reading |
|---|---|---|---|
| 0 → 512 | 1.000 → 0.882 | [0.940, 0.890] | stable |
| **512 → 1000** | **0.188** | **[0.352, 0.122]** | **rotation** |
| **1000 → 2000** | **0.282** | **[0.287, 0.002]** | **rotation** |
| 2000 → 16000 | 0.653 → 0.940 | rising to [0.943, 0.915] | re-forming |
| **16000 → 32000** | **0.490** | **[0.961, 0.951]** | **reordering** |
| 32000 → 143000 | 0.870 → 0.978 | [0.979, 0.974] | stable |

Two distinct events. **512–2000 is a genuine rotation** — the whole plane moves,
second principal cosine reaching **0.002** (orthogonal). **16000–32000 is
reordering inside a stable plane** — top-1 halves while the subspace holds at
[0.961, 0.951], which is `svd@1` falling 0.68 → 0.20 while `svd@2` stays
0.85–0.97, seen from the other side. §3.11's consolidation reading is right; the
earlier event was missed entirely.

**And the rotation window is the repulsive-collapse window is `CLAIM-B`'s
anchor window.** Three independent quantities — a population spectral phase, one
head's OV direction, and a registered literature anchor — all name 512–2000.
Whether that is one event or three is now the question worth asking.

**Design consequence, forced by measurement:** the stable object is the
**top-2 subspace**, not the top-1 direction. The Stage 3 entry must perturb the
plane. "The rank-1 direction" is well defined only inside the formation regime
(4000–16000).

*F3 — the found control is decisive, and it cuts against the spectral frame.*
At step 4000:

| head | attr. frac | complex frac | ‖M‖_F | σ₁ share | σ₁₂ share | participation |
|---|---|---|---|---|---|---|
| `L7H8` copier (ΔOV_nll **+0.244**) | 0.000 | 0.787 | 1.665 | 0.176 | 0.241 | 19.9 |
| `L2H10` anti-copier (**negative**) | 0.000 | 0.881 | 1.287 | 0.166 | 0.226 | 21.5 |
| `L9H8` anti-copier (**negative**) | 0.000 | 0.728 | 1.749 | 0.125 | 0.174 | 30.8 |
| `L5H2` prev-token | **0.444** | 0.896 | 2.568 | 0.062 | 0.118 | 27.5 |

**`L2H10` is spectrally near-indistinguishable from `L7H8`** — same repulsive
fraction (0.000), same gain concentration (0.166 vs 0.176), same participation
ratio (21.5 vs 19.9) — and its causal copying effect has **the opposite sign**.
So the OV spectral description **does not determine the causal role**. That is
§3.11's "the repulsive sign is true but not the mechanism" in its sharpest
available form, obtained from a *found* control rather than a constructed one,
and therefore immune to the matching problem that killed both constructed
designs. `L2H10` is the control head the S-flip experiment should run against.

Separately: `L5H2` is the **only** spectrally mixed head of the four (0.444) and
the least gain-concentrated. The two stages of the circuit are spectrally
unlike, which no one had checked.

**G. The derivation, and the negative it produced (2026-09-09).**
`MATH_SPECTRAL_OT.md` gains **§2.4, "The S/A split as an intervention"**, built
on §§2.1–2.3 rather than re-deriving them. Four results, three of which changed
the design and one of which killed a hypothesis of its own making.

*G1 — the readout Stage 3 depends on is a quadratic form in `S` (§2.4.1).* §2.1
derives the first-order result for `‖x‖`; the particle account is about
*inter-particle* distance. For `δ = x − y` the update is linear, so
`‖δ + Mδ‖² = ‖δ‖² + 2·δᵀSδ + ‖Mδ‖²` — identical form. Hence "individuating" is
literally the statement `δᵀSδ > 0` on the matched differences, `A` contributes
**exactly nothing** at first order, and **§3.12-C2's mechanical confound becomes
computable in closed form and subtractable** rather than something to control
for. `P-I5`'s pairwise readout and `P-ST1`'s effective-rank readout are the same
quadratic form on different arguments.

*G2 — the four sign choices are a group, and block D was under-specified
(§§2.4.2–2.4.3).* `{M, Mᵀ, −M, −Mᵀ} = {S+A, S−A, −S−A, −S+A}` is a complete
**2² factorial** in (sign `S`, sign `A`), and **every arm is an isometry** —
same singular values, same `‖·‖_F`, same eigenvalue moduli, no rescaling, no
draws. But `M = UΣVᵀ ⟹ Mᵀ = VΣUᵀ`, so **transposing swaps the read subspace
with the write subspace**, which for a copier is a change of function. It
resolves exactly: role-swap `⟺ sign(S) ≠ sign(A)`, i.e. the `S×A` interaction.
So **both main effects are clean and the interaction is aliased with the role
swap** — and block D's single `−Mᵀ` arm, run alone, confounds the `S` sign with
the swap completely. **Stage 3 must run all four cells.**

*G3 — the flip's residual is the non-normality (§2.4.4).* Applying `M` and
`−Mᵀ` to the same `δ`, the first-order terms cancel identically and the
difference is `4·δᵀSδ + δᵀ[Mᵀ,M]δ`. So the S-flip inverts the first order
*exactly*, and the whole deviation is a quadratic form in the self-commutator —
a third readout for free, and the one that answers §5.3(d) per head by
intervention rather than by residual.

*G4 — §2.4.5's own prediction, measured and FALSIFIED
(`induction_abscissa_7b.py`, §2.4.6).* Bendixson gives only the inclusion
`Re λ(M) ∈ [λ_min(S), λ_max(S)]`, so two heads can agree on every
eigenvalue-derived field and differ on the numerical abscissa — which would have
explained F3. **It does not.** At step 4000:

| | `attr_frac` | `λ_max(S)` | `λ_min(S)` | `S_pos_E` | `‖[Mᵀ,M]‖` |
|---|---|---|---|---|---|
| `L7H8` (ΔOV_nll **+0.244**) | 0.000 | 0.2644 | −0.4392 | 0.150 | 0.2968 |
| `L2H10` (**negative**) | 0.000 | 0.2238 | −0.3038 | 0.202 | 0.2924 |

The two candidate separators differ by ~15 % and ~35 % **in opposite
directions**, and the commutator agrees to two digits. Across its own layer
`L7H8` ranks **8th of 16** on `λ_max(S)` while carrying ~10× the layer's causal
effect.

**So `L7H8` is unremarkable in the eigenvalue frame, unremarkable in the
symmetric frame, and unremarkable in the gain frame — and it is the head that
does induction.** No weights-only spectral quantity this project computes
identifies the copier. (Population test of that claim:
`induction_spectral_predicts_7b.py`, 112 heads with a measured causal readout.)

This does **not** show the spectral character is causally inert — a static
property failing to *predict* which head copies is not the same as flipping it
failing to *change* what the head does. The G2 factorial is now the thing that
decides it. But it moves the prior hard, and it puts the *representational*
reading of induction in trouble on a third independent front, after the missing
token diagonal and after F3.

*G5 — and the developmental signal is where the structure is.* `L7H8`'s
`λ_max(S)` holds its initialisation value (~0.040) through step 1000, then rises
~20× — 0.104 at 2000, 0.264 at 4000, 0.455, 0.627, peaking **0.729 at 32000**
before falling to 0.385 at 143000. Its `S_pos_E` traces a **U-shape with floor
at step 2000** (0.076). That is the **fifth** quantity to name 512–2000, after
the population repulsive collapse (A), this head's OV plane rotation (F2), the
eigenvalue-frame U-shape (A), and `CLAIM-B`'s registered anchor. The
cross-sectional question has a negative answer; **the developmental one is the
live one**, which is sub-phase 7a rather than 7b.

*G6 — the population test, and it splits the fields cleanly
(`induction_spectral_predicts_7b.py`).* G4 was two heads. Seven
`induction_subspace_characterize*` runs already carry a measured
`full_ablation_delta_nll` for **every head of a layer** at four checkpoints —
112 heads with a causal readout and a dense `W_OV` on disk. Asking whether any
weights-only field rank-correlates with the causal effect: **Spearman ρ is ≈ 0
for all ten fields** (best `|mean ρ|` = 0.133; **no field reaches `|ρ| > 0.5` in
any of the 7 populations**).

**But the correlation was the wrong instrument, and the rank table is the right
one.** Fifteen of sixteen heads per layer carry causal noise, so Spearman
averages the signal away. What matters is where the *causally-top* head sits on
each field (rank 0 = that field's own maximum of 16):

| field | top-head rank across the 7 populations | extreme in |
|---|---|---|
| `sv12_share` | 2, 1, 11, 0, 0, 1, 1 | **6/7 top-3** |
| `lambda_min_S` | 13, 15, 11, 15, 15, 15, 14 | **6/7 bottom-3** |
| `sv1_share` | 2, 2, 12, 0, 0, 1, 3 | 5/7 top-3 |
| `nonnormality` | 2, 2, 6, 0, 0, 1, 3 | 5/7 top-3 |
| `participation_ratio` | 13, 13, 7, 15, 15, 14, 12 | 5/7 bottom-3 |
| `attractive_energy_fraction_core` | 8, 9, 2, 0, 14, 10, 2 | — |
| `sym_pos_energy_fraction` | 10, 8, 1, 0, 15, 14, 13 | — |
| `max_re_lambda` | 11, 9, 2, 0, 14, 14, 13 | — |
| `lambda_max_S` | 7, 7, 3, 0, 5, 8, 5 | — |
| `frobenius` | 8, 6, 1, 2, 13, 13, 5 | — |

Under a uniform-rank null the top group sits at ~2.6e-4 and ~3.5e-3, the bottom
group at 0.13–0.77. **That is arithmetic, not an adjudication** — the
populations share checkpoints, heads within a layer are not independent, and the
fields were not chosen in advance. No p-value is emitted and
`claims/registry.json` is untouched.

**The split is exact and it is the finding.** *Concentration and magnitude*
quantities identify the copier — `σ₁₂` share, `σ₁` share, participation ratio,
non-normality, and the single most-negative `λ(S)`. *Sign-balance* quantities do
not — the attractive fraction, `max Re λ`, `S_pos_E`, and (worst of all ten, at
0.77) `λ_max(S)`, the field §2.4.5 nominated. `‖M‖_F` alone does not either, so
it is concentration rather than size.

**Three consequences.**

1. **`r*_SVD ≪ r*_Schur` — the one Stage 1/2 finding that survived — is now a
   population result**, confirmed across 7 layer-populations and 4 checkpoints
   rather than one head at one step. The copier is the most gain-concentrated
   head in its layer, and that is what marks it.
2. **A pre-run prediction for Stage 3, which the drafted entry lacked.** The
   S-flip preserves *every singular value*, so it preserves gain concentration
   exactly. **Predicted before running: `−Mᵀ` will not destroy copying.**
3. **That sharpens the G2 factorial rather than weakening it.** If concentration
   is the whole story, even `Mᵀ` preserves copying; if the read/write
   *alignment* matters, `Mᵀ` destroys it while `−M` does not. So the informative
   contrast is **`M` vs `Mᵀ`** — alignment at fixed gain — and not the `S` sign
   at all. The factorial computes both for the same four forward passes.

**H. Composition and whitening (2026-09-09,
`tools/run/induction_composition_whitening.py`).** The two checks that needed a
model. Both returned, and both matter.

*H1 — the composition switches on between step 512 and step 1000, and it is the
cleanest circuit-level signal in the programme.* K-composition
`‖W_K^{L7H8} · W_OV^{L5H2}‖_F / (‖W_K‖_F‖W_OV‖_F)`, scored against **all 112
upstream heads** into the same key path, so "elevated" is against the model's
own distribution and no threshold is placed:

| step | K | pop median | **rank / 112** | z | Q | V |
|---|---|---|---|---|---|---|
| 0 – 256 | 0.0311 | 0.0313 | **83** | −0.56 | 0.0311 | 0.0320 |
| 512 | 0.0313 | 0.0313 | **66** | −0.08 | 0.0312 | 0.0320 |
| **1000** | 0.0346 | 0.0314 | **0** | **+5.78** | 0.0331 | 0.0317 |
| 2000 | 0.0502 | 0.0317 | **0** | +6.87 | 0.0486 | 0.0377 |
| 4000 | 0.0551 | 0.0318 | **0** | +5.74 | 0.0541 | 0.0438 |
| 143000 | 0.0778 | 0.0315 | **0** | +6.93 | 0.0778 | 0.0470 |

`L5H2 → L7H8` goes from **below the population median (rank 83 of 112)** to
**the single strongest composition into that key path (rank 0, z ≈ +6)** in one
step interval, and never leaves rank 0 again. **Sixth quantity to name
512–2000**, and the first that is unambiguously about the *circuit* rather than
about spectra.

**The honest caveat, and it is not small: K and Q rise together.** By step 8000
they are equal to three digits (0.0778 both at 143000). V is clearly lower
(0.0470) so the measurement is not purely generic, but **the Q/V arms were
computed without their own population controls**, so this does *not* yet isolate
the K-specific induction story — only that `L5H2`'s output becomes strongly
aligned with `L7H8`'s read subspaces. Scoring Q and V against their own 112-head
populations is the immediate next step and is the same cost.

*H2 — §3.12-C3 confirmed: the weight-identified direction is not the
data-identified one.* At step 4000, comparing the top singular directions of
`W_OV` against those of `W_OV Σ^{1/2}` (pulled back to input space):

- top-1 overlap **0.207**
- top-2 plane principal cosines **[0.293, 0.091]** — very nearly orthogonal planes
- but `σ₁` energy share 0.176 → 0.148 and `σ₁₂` **0.241 → 0.270**

So the *direction* does not survive whitening and the **concentration does**.

**Caveats, both real.** Σ is estimated from only **1536 tokens** for a 1024-dim
space (effective rank 238), so part of the low overlap is estimation noise —
raising `N_SEQS` is cheap and should be done before this is leaned on. And Σ
here comes from the repeated-random-token battery, not natural text; that is
arguably the *right* metric since it is the distribution the causal readout
uses, but it is not the model's operating distribution and the two should be
compared.

**H1-REVISED (2026-09-09, rerun with per-path population controls).** The
caveat was right and it is now settled: **the composition is not K-specific.**
Scoring Q and V against their own 112-head populations:

| step | K rank / z | Q rank / z | V rank / z |
|---|---|---|---|
| 0 – 512 | 83 / −0.56 | 72 / −0.38 | 2 / +1.94 |
| **1000** | **0 / +5.78** | 2 / +3.25 | 21 / +0.81 |
| 2000 | 0 / +6.87 | **0 / +6.40** | 0 / +6.07 |
| 8000 | 0 / +4.94 | **0 / +5.41** | 4 / +2.32 |
| 143000 | 0 / +6.93 | **0 / +6.79** | 6 / +1.54 |

**And the reason is an artifact, identified rather than guessed:** `L7H8`'s
`W_Q` and `W_K` **converge onto the same subspace** over training — mean
principal cosine between their rowspaces **0.213 → 0.829**, top principal cosine
**0.476 → 0.995**. Any operator composing into `K` therefore composes into `Q`,
and the composition score cannot separate the two pathways for this head.

*What survives, and it is most of it.* The **timing** (rank 83 of 112 → rank 0
between 512 and 1000) and the **magnitude** (z ≈ +6, sustained) stand
untouched — those never depended on which read path. What must be dropped is the
phrase "K-composition": the right description is **composition into `L7H8`'s
attention read-space**, which is *one* object because Q and K share it. And the
**V arm does discriminate** — it drifts to rank 4–6 at z ≈ +1.5 while K and Q
hold rank 0 at z ≈ +6 — so the composition is into the **attention** pathway and
not the **value** pathway, which is the induction-shaped result and is the part
that was actually worth having. At the onset step alone (1000) K does lead Q,
rank 0 / z +5.78 against rank 2 / z +3.25; one step and a small gap, recorded
and not leaned on.

**H2-CONFIRMED, by a control that needed no model of the noise.** Three arms
plus a split-half at step 4000:

| arm | tokens | top-1 overlap | plane cos | Σ eff. rank |
|---|---|---|---|---|
| battery | 24,576 | 0.217 | [0.329, 0.088] | 327 |
| battery half 1 | 12,288 | 0.222 | [0.320, 0.088] | — |
| battery half 2 | 12,288 | 0.213 | [0.336, 0.088] | — |
| natural text | 3,185 | 0.266 | [0.270, 0.104] | 281 |

**Split-half: half 1 against half 2 agrees at top-1 0.990 and plane cos
[1.000, 0.778].** The two independent halves agree with *each other* at 0.99
while both disagree with raw at ~0.22 — so **the low overlap is signal, not
estimation noise**, and H2's caveat is discharged. Sixteen times the tokens
moved the answer 0.207 → 0.217, so the original estimate was already sound. And
`Σ(battery)` vs `Σ(natural)` have cosine **0.453** — substantially different
metrics giving the same answer, which is the robustness the caveat asked for.

*H3 — three independent results now converge on the same correction.* G6 says
**concentration** identifies the copier and directional/sign quantities do not.
H2 says concentration **survives whitening** and the direction does not. F2 says
the top-1 direction is not even stable across checkpoints while the top-2
*plane* is. So F2's "perturb the top-2 subspace" fix is **still not enough** —
the object has to be defined in the whitened metric, or, better, **the Stage 3
entry should be about gain concentration rather than about any named
direction.** That is a different prediction from the one §3.11 asked for, and it
is the one the measurements support.

**I. The matcher becomes a similarity kernel, and this is the first quantity
that cleanly identifies the head (2026-09-09).** H1-REVISED's artifact — `W_Q`
and `W_K` converging onto one subspace — is not only an artifact. A head whose
query and key read the *same* subspace computes an attention logit
`qᵀk = xᵀ(W_Qᵀ W_K)y` that is close to a **similarity form**, i.e. the static QK
operator should be becoming **symmetric**. Measured directly, on
`M = W_Q[16:]ᵀ W_K[16:]` (the same static operator §3.11 block C sweeps), as
`‖S‖²_F / ‖M‖²_F`:

| step | `L7H8` sym. fraction | `‖M‖_F` | **layer-7 median** | `L7H8` rank / 16 |
|---|---|---|---|---|
| 0 | 0.5017 | 2.78 | 0.5006 | 2 |
| 512 | 0.5019 | 2.79 | 0.5008 | 1 |
| 1000 | 0.5042 | 2.87 | 0.5034 | 6 |
| **2000** | **0.5742** | 3.50 | 0.5095 | **0** |
| 4000 | 0.7398 | 4.44 | 0.5181 | **0** |
| 8000 | 0.8807 | 5.90 | 0.5200 | **0** |
| 16000 | 0.9311 | 7.85 | 0.5263 | **0** |
| 32000 | 0.9465 | 9.86 | 0.5257 | **0** |
| 143000 | **0.9561** | 7.91 | 0.5203 | **0** |

A random real matrix splits its energy 50/50 between `S` and `A`, and 0.50 is
where every head starts. **`L7H8` goes to 0.956 while its fifteen neighbours
stay at the random baseline (layer median 0.520 at step 143000), and it is rank
0 of 16 at every step from 2000 on.** Takeoff is 1000 → 2000 — the **seventh**
quantity to name that window.

**Three things this is** *(claim 1 below was overstated and is corrected in
block J — left standing because the correction is the point)*.

1. **The first weights-only quantity in the entire programme that cleanly
   identifies the head.** §3.12-G6's negative — no spectral field picks out the
   copier — surveyed the **OV** operator only. On the **QK** side, symmetry
   identifies the matcher decisively: 0.956 against a layer median of 0.520,
   rank 0 of 16, sustained over seven checkpoints.
2. **It explains a previously descriptive finding.** §3.11 block C reports the
   matcher as a "near-normal" invariant subspace with `r*_Schur` 12 < `r*_SVD`
   32. A symmetric operator **is** normal, so near-symmetric *derives*
   near-normal rather than restating it, and it explains why the Schur frame is
   the right one for the matcher and the wrong one for the copier.
3. **"Matching kernel" stops being a metaphor.** The particle account's own
   framing is that induction is "a matching-kernel coupling rather than a
   feature-copying circuit" (`POPPER_PLAN.md` §C2). The static QK half of this
   head *is* a similarity kernel, at 0.956. That is the account's language
   arriving as a measurement — **on the half nobody was testing**.

**What it does not do, stated because the temptation is obvious.** It adjudicates
nothing: `claims/registry.json` is untouched, this is one head in one model, and
the differential prediction §3.11 wanted was about **OV**, where the evidence
still runs the other way. A symmetric *content* operator is also exactly what
one would expect architecturally — RoPE carries the positional asymmetry in the
16 excluded dims and the causal mask carries the rest, so "symmetric content
part plus positional part" is the ordinary way to build a content matcher, and
that reading has to be ruled out before any of this is registered. The obvious
next measurements: the same sweep on the other genuine induction heads
(`L9H9`, `L6H0`, `L1H15` — block C's 3-of-4), and on non-induction heads that
also match, to see whether symmetry tracks *induction* or merely tracks
*matching*.

**J. It tracks MATCHING, not induction — block I's claim 1 is corrected
(2026-09-09, `tools/run/qk_symmetry_sweep.py`, all 384 heads × 19 steps).**
The symmetric fraction reduces to 48×48 algebra — `‖M‖²_F = tr((AᵀA)(BBᵀ))` and
`tr(M²) = tr((BA)²)` for `M = AB` — so no (1024,1024) matrix is ever formed;
the identity is asserted against the direct computation on the first head of
every run rather than trusted.

*The population.* Every head sits at the random baseline (median 0.5005, max
0.5027, **none above 0.7**) through step **512**. First movement at 1000; by
143000 the median is 0.5231 and only **11 of 384** heads exceed 0.7, **one**
exceeds 0.9. So high symmetry is genuinely rare — but it is **not unique**.

*The decisive cell, at 143000 — top-10 symmetry heads and their induction rank
of 384:* `L7H8` 0.956 (rank 2), `L6H0` 0.873 (**0**), `L1H15` 0.840 (5),
`L2H10` 0.835 (**1**), `L7H0` 0.796 (3), `L9H9` 0.738 (7), **`L1H4` 0.734
(40)**, **`L8H13` 0.730 (47)**, `L9H8` 0.727 (4), **`L7H4` 0.708 (91)**.

**Seven of the top ten are top-8 induction heads — and three are not.** At step
4000 the mismatch was starker still (`L10H3`, symmetry rank 5, induction rank
**382 of 384**). So symmetry is close to *necessary* for induction and clearly
not *sufficient*: **block I's "the first quantity that cleanly identifies the
head" was too strong.** What symmetry identifies is the **matching half of the
circuit**, and induction heads are matchers.

*Two controls settle what "matching" means here, and neither was arranged.*

- **`L5H2`, the previous-token head, never becomes symmetric** — 0.500 → 0.504,
  rank 322 → 310 of 384, flat across the entire axis while its induction
  partners climb past 0.8. It is an overwhelming matcher (attention 0.895 at
  offset −1, ~49× the head median) but it matches on **position**. So symmetry
  marks **content** matching specifically, and the architectural reading block I
  flagged is *partly right*: it is about matching, not about induction.
- **`L2H10` matches without copying.** It reaches symmetry rank 3 (0.835) and
  induction rank 1 while its OV copy effect is **negative**. Its trajectory —
  rank 131 at 4000, 23 at 8000, 4 at 16000, 3 thereafter — means the step-4000
  reading ("the anti-copier is not symmetric") **does not survive the full
  axis**, and what replaces it is better: **matching (QK, symmetric) and copying
  (OV) are separable, and this head has one without the other.**

*What the two halves now look like together.* §3.12-G6: on the **OV** side no
sign quantity identifies the copier and only *gain concentration* does, weakly.
Block J: on the **QK** side *symmetry* identifies the matcher, sharply and
rarely (11 of 384). **Induction is both**, and `L2H10` is the existence proof
that they come apart. That is the dissociation §3.11 block C was reaching for,
now on a population footing rather than one head.

*Developmentally — the eighth quantity to name the window.* Flat at baseline
through 512; first movement at 1000; by 2000 `L9H9` leads at rank 0 (0.595)
with `L7H8` at rank 3, and from 4000 `L7H8` takes rank 0 and never gives it up.
And the high-symmetry set **purifies toward induction heads over training** —
the worst induction rank in the symmetry top-10 goes from 382 at step 4000 to 91
at 143000.

*A §3.13 footnote that is not a footnote.* Spearman of symmetry against
induction over all 384 heads is ≈ 0 at every step **except step 1000, where it
is +0.343**. The correlation is the right instrument exactly once — at the onset,
when many heads move slightly together — and the wrong one everywhere after,
when the signal concentrates into a handful. **The right summary changes with
training stage**, which is a sharper version of §3.13 than that section states:
the mean-versus-extremum choice is not only per-quantity, it is per-regime.

**K. What the position profile found instead (2026-09-09).** §3.13.3's own
hypothesis came back negative, but the profile carries two things nobody had
looked for.

*K1 — a mechanism check the readout has never made, and it passes.* At the first
second-copy position `j = 0` the model predicts `ids[N_REP]` from the first copy
alone: **no earlier occurrence of the current token exists yet, so induction
cannot fire there.** Measured, `j = 0` is the **only** position with a negative
effect — ΔNLL **−0.0484**, and it is the run's minimum — so ablating `L7H8`'s OV
slightly *helps* exactly where induction is impossible. Its induction attention
is also the lowest of any position, **0.779** against a mean of 0.910. The
prediction was made from the slice convention before the run and both halves
hold. This is the first end-to-end validation that `second_copy_nll` is
measuring what it is supposed to measure.

*K2 — there is real position structure; it just does not bias the mean.* ΔNLL
runs **−0.048 at j = 0**, peaks at **+0.570 at j = 4**, falls to **+0.390 at
j = 8**, and settles onto a plateau near **+0.22** for the rest of the sequence.
So the copying effect is largest immediately after induction becomes possible
and then decays to about a third of its peak — consistent with the model having
progressively more non-induction evidence as the repeat proceeds, and worth a
look on its own.

*K3 — and the QK/OV dissociation appears on a third axis.* Across the same
positions **induction attention is flat at 0.90–0.93** while **ΔNLL varies
roughly fourfold (0.20 → 0.57)**. The matcher fires essentially uniformly; the
copier's *payoff* does not. That is §3.11 block C's dissociation and §3.12-J's
two-halves picture arriving from the position axis, which neither was derived
from.

**L. The Stage-1 table compares operators of different size (2026-09-09).**
Found while stress-testing the Stage 3 design, and it revises a headline.

`tools/run/induction_rank_sweep.py::truncate` returns, at rank `r`: for `svd` a
genuine rank-`r` truncation (`U[:, :r] * s[:r], Vt[:r]`), and for `schur` and
`random` a **projection of `A`** onto an `r`-dimensional subspace (`A @ P, B`).
**None preserves Frobenius norm, and they lose it at very different rates.**
Measured on `L7H8` at step 4000, energy retained as a fraction of the full OV:

| r | svd | schur | random | svd/rand | svd/schur |
|---|---|---|---|---|---|
| **1** | **0.1755** | **0.0373** | **0.0108** | **16.2×** | **4.7×** |
| 4 | 0.3368 | 0.1398 | 0.0518 | 6.5× | 2.4× |
| 16 | 0.6153 | 0.4039 | 0.2315 | 2.7× | 1.5× |
| 48 | 0.9263 | 0.8017 | 0.7050 | 1.3× | 1.2× |

The `random` branch's own docstring says *"Matched-norm random rank-r control …
so the operator norm scale and the factor structure match the real
truncation."* **At `r = 1` it is off by 16×.** So §3.11's Stage-1 table — the
one reading `r*_SVD ≪ r*_Schur` and "random ≈ 0 %" — **compares operators of
different size at the same rank.**

*Re-read at matched ENERGY instead of matched rank*, using the NLLs already in
`induction_rank_sweep.json` (no new forward passes):

| energy | svd | schur | random |
|---|---|---|---|
| 0.175 (= svd's r=1) | **0.820** (r 1) | **0.599** (r≈4.8) | **0.353** (r≈12.6) |
| 0.25 | 0.847 | 0.760 | 0.427 |
| 0.40 | 0.924 | 0.807 | 0.586 |
| 0.60 | 0.964 | 0.893 | 0.814 |

**The ordering survives — SVD > Schur > random at every matched energy — but the
gaps collapse.** SVD-over-Schur goes from **6.9×** at matched rank (0.820 vs
0.119) to **1.37×** at matched energy. Random is not "≈ 0 %"; at `r = 1` it
holds 1.08 % of the energy, and given 17.55 % it recovers **0.353**.

*What stands and what does not.* The **structural** claim stands: at matched
energy the top singular directions beat structureless ones **2.3×**, so this is
not energy alone. What does not stand is the *magnitude* of `r*_SVD ≪
r*_Schur` — 1.37× is much weaker support for "the copying action lives in a
high-gain non-invariant direction, and the attractive/repulsive frame is the
wrong description" than 6.9× was. Eckart–Young makes "SVD retains most energy at
rank `r`" a **theorem**, so at matched rank part of the gap was never a finding.
**§3.12-G6's population result is unaffected** — it correlates gain
concentration with *which head* is the copier and never uses this comparison.

*And this is §6n's own rule applied to this project's headline for the first
time*: match the control on **the quantity the statistic degenerates on**. It
degenerates on energy; the controls were matched on rank.

**Consequence for Stage 3, and it is a convergence.** Any Stage 3 arm must be
**energy-matched, not rank-matched**. The `{M, Mᵀ, −M, −Mᵀ}` factorial is
*exactly* energy-matched — every arm is an isometry (§2.4.2) — so the design
motivated by the S/A group structure turns out to be the fix for this confound
as well, from a completely independent direction.

*Precondition on the S-flip, computed before it runs.* If `S` carried little of
the OV's energy, "the flip does nothing" would be guaranteed by magnitude rather
than by mechanism. It does not: `‖S‖²/‖M‖²` for `L7H8`'s OV is **0.602** at step
4000 and 0.543 at 143000, so `‖M − (−Mᵀ)‖ = ‖2S‖ ≈ 1.55‖M‖` — **the change is
larger than the operator**. The precondition passes.

*But it also shows the two halves are structurally different in exactly the way
§3.12-J found.* `L7H8`'s **OV** symmetric fraction is 0.602 against a layer
median of 0.584 — at the baseline, carrying no signal. Its **QK** symmetric
fraction is **0.956 against a layer median of 0.520**. The matcher becomes a
similarity kernel; the copier does not, and never does.

**M. The S/A factorial is not a viable Stage 3 design — its own pilot killed it,
and produced a better result (2026-09-09,
`tools/run/induction_sa_pilot.py`).** Step 4000, 16 sequences, restore exact
(`0.0e+00`) on every head. ΔNLL against the unablated model:

| arm | `L7H8` ΔNLL | KL | /ablation | `L9H9` | `L2H10` |
|---|---|---|---|---|---|
| ablation | +0.2420 | 0.093 | — | +0.0752 | −0.0129 |
| scale 0.25 | +0.1412 | 0.036 | | +0.0457 | −0.0096 |
| scale 0.50 | +0.0729 | 0.011 | | +0.0242 | −0.0034 |
| scale 0.75 | +0.0279 | 0.002 | | +0.0094 | −0.0015 |
| **`Mᵀ`** | **+0.2278** | 0.089 | **0.94×** | +0.0928 (1.23×) | −0.0112 |
| **`−M`** | **+1.2409** | **0.969** | **5.13×** | +0.3163 (4.21×) | −0.0076 |
| **`−Mᵀ`** | **+0.4000** | 0.181 | **1.65×** | +0.1113 (1.48×) | −0.0083 |

*M1 — `−M` is off-scale, exactly as predicted.* **5.13×** ablation on `L7H8`,
4.21× on `L9H9`, with KL **0.969** against ablation's 0.093 — an order of
magnitude. Reversing the write is far more destructive than removing it, because
the head actively suppresses the token it used to promote. **So the balanced
main effects — which average `−M` with `−Mᵀ` — are dominated by one catastrophic
arm, and additivity fails.** The prediction was made from the algebra before the
run and both halves hold.

*M2 — and the design cannot answer its own question.* §2.4.3 proved every
pairwise contrast isolating a sign **also swaps the read/write role**. The pilot
now measures what that swap costs on its own: **`Mᵀ` alone destroys copying at
0.94× ablation.** So by the time the transpose has been applied the effect is
already gone, and the `S`-sign contrast (`Mᵀ` → `−Mᵀ`, 0.94× → 1.65×) is
measured **on top of a floor**. The main effects cannot rescue it because of M1.
**Both routes to the `S` sign are blocked, and §3.12-L's rank result shows no
other points exist** — so this is not a design to be repaired. It is retired
before registration, which is what the pilot was for.

*M3 — `G6`'s pre-run prediction is FALSIFIED, and the falsification is the
finding.* G6 predicted `−Mᵀ` would **not** destroy copying, because it preserves
every singular value and hence gain concentration exactly. It destroys it at
**1.65× ablation**. And `Mᵀ` — same singular values, same Frobenius norm, same
eigenvalue moduli, *nothing changed but which subspace reads and which writes* —
destroys it at **0.94×**, as completely as deleting the head.

**So the copier's function is carried by read/write ALIGNMENT, not by its
spectrum and not by its gain profile.** §3.12-G6 showed gain concentration
identifies *which head* is a copier; it does not carry *the copying*. That is
consistent with §3.12-H2 — the top OV direction does not survive whitening
(overlap 0.217) while the concentration does — and it completes that reading:
concentration is a **marker**, alignment is the **mechanism**.

*M4 — `L2H10` is not usable as a control, measured rather than argued.* Its
whole dynamic range is |ΔNLL| ≤ 0.013 and the arm ratios (0.87×, 0.59×, 0.65×)
are noise on that scale. A head whose baseline effect is ~2 % of the target's
cannot calibrate an intervention on the target. The §3.12-J finding that it
matches without copying stands; its use as the Stage 3 control does not.

*M5 — a nonlinearity worth carrying forward.* The scale curve is strongly
**sublinear**: halving `L7H8`'s OV costs only +0.073 against ablation's +0.242,
so **50 % of the operator does 30 % of the damage**. Any future arm reported in
"equivalent λ" is on a compressive scale and must say so.

**N. Literature check before registering (2026-09-09).** Run because §3.12-I/J
and §2.4 are close enough to published work that registering without checking
would risk re-deriving it. Four threads, and they change what should be claimed.

*N1 — QK symmetry is known at the population level, and our result is the
complement rather than a duplicate.* Saponati et al., *"The underlying
structures of self-attention: symmetry, directionality, and emergent dynamics in
Transformer training"* (arXiv 2502.10927), decomposes `W_QK` into symmetric and
skew parts and defines a Frobenius symmetry score. **Their finding is that
bidirectional training induces symmetry while autoregressive training induces
directionality** — decoder-only models score *more directional* than
encoder-only. That is a **median across layers, model-level, with no per-head
breakdown, no head-type analysis, no mention of induction or previous-token
heads, and no exclusion of positional dimensions.**

**This does not contradict §3.12-J; it frames it.** Our population median is
**0.523** — essentially neutral, consistent with their directional aggregate.
What §3.12-J adds is the **tail**: 11 of 384 heads exceed 0.7, one reaches
0.956, and those heads are the content matchers. Their claim is about the
median; ours is about which heads leave it, and why. The RoPE exclusion matters
here too — pythia's positional (directional) information lives in the 16 rotary
dims we remove, so we measure the *content* operator they do not separate.

*N2 — the mechanistic reading of the S/A split exists, and it gives §3.12-J its
name.* *"The Routing and Filtering Structure of Attention"* (arXiv 2605.18826)
splits the pre-softmax score matrix into a symmetric **"filtering"** part
(undirected mutual relevance) and a skew **"routing"** part (directional
transport, purely imaginary eigenvalues), and finds routing removal catastrophic
(699 PPL against a 34.99 baseline). Since `A = X M X^T` for `M = W_Q W_K^T`,
their split of the *scores* is our split of the *weights* conjugated by the
activations — the same decomposition at two levels, which should be said rather
than discovered later. **In their vocabulary §3.12-J reads: `L7H8`'s content
operator becomes almost pure filtering (0.956) with almost no routing** — the
routing being supplied by RoPE and the causal mask, which is exactly the
architectural division block I flagged. They report **no transposition
experiment, no induction-head connection, and no OV analysis.**

*N3 — the induction-head emergence window is established, so §3.12's "eighth
quantity" claim needs restating.* Published checkpoint studies of Pythia put
induction-head emergence at **around step 1000 of 143000**. The 512–2000 window
is therefore **not our finding**, and this document should stop implying
novelty for the window itself. What is ours is the **set of quantities that
co-locate in it** — the population OV repulsive collapse, `L7H8`'s OV plane
rotation, the eigenvalue-frame U-shape, the numerical-abscissa takeoff, the
composition switch-on, and the QK symmetry takeoff — several of which are
weights-only and none of which is the behavioural score the literature dates the
window by.

*N4 — and the actionable one: we have not computed the field's own copying
score.* Elhage et al.'s copying test takes the eigenvalues of the **token-basis**
OV circuit `W_E W_V W_O W_U` and summarises their positiveness as
`sum(lambda) / sum(|lambda|)`; a copier has **positive** eigenvalues.
**This project has never computed that matrix.**
`p2b_imaginary/head_circuits.head_core` returns `W_V W_O` — the `(64,64)` core in
the **residual** basis — and every "100 % repulsive" statement in §3.11 and
§3.12-A is about *that*, not about the token-basis circuit. The two differ by the
vocabulary round-trip `W_U W_E` sandwiched between the factors, and **nothing
guarantees they share a sign**.

It is cheap: with `W_OV = A B`, the nonzero spectrum of `W_E A B W_U` equals that
of the `(64,64)` matrix `B W_U W_E A`, so it costs one `64x64` eigendecomposition
per head — the same price as the core we already compute. **If `L7H8` scores as
a copier on the field-standard measure, then §3.11's "not a token-identity
copier" — which rests on the weaker LN-folded diagonal check the section itself
flags as its softest half — is measuring a different thing, and the
"repulsive/individuating" reading cannot be carried over to it.** This is the
next measurement, before any registration.

**O. It was measured, and it CONFIRMS §3.11 — emphatically (2026-09-09,
`tools/run/copying_score_sweep.py`).** All 384 heads, eight checkpoints, weights
only. A convention trap was checked rather than assumed first: `ov_factors`
returns `OV_h = (W_O W_V)ᵀ`, the **transpose** of the residual operator (verified
at relative error 0.0). Harmless for every quantity read off it so far — all
transpose-invariant — but **not** for a copying score, which is directional, so
this runner uses `W_O W_V` from the model.

*O1 — `L7H8` is not a token-identity copier at any point in training, and it
becomes less of one as its causal effect grows.*

| step | `L7H8` copying score | rank / 384 | `ΔOV_nll` (§3.11-A) |
|---|---|---|---|
| 512 | −0.038 | 324 | ~0 |
| 1000 | −0.050 | 305 | ~0 |
| 2000 | −0.043 | 258 | ~0 |
| 4000 | **+0.062** | 139 | **+0.24** |
| 8000 | +0.020 | 181 | **+0.73** |
| 16000 | −0.050 | 267 | **+1.02** |
| 32000 | −0.115 | 322 | — |
| 143000 | **−0.094** | **328** | +1.18 |

**Its causal OV effect grows roughly sixfold while its copying score stays at or
below zero and its rank falls to 328 of 384.** The two run in opposite
directions. §3.11's conclusion was reached on the LN-folded diagonal check it
called its own softest half; it now stands on the field's instrument, across the
whole axis, and stronger than when it was stated.

*O2 — and the model is full of real copiers, none of them these heads.* Scores
above 0.4 go 0 (step 512) → 10 → 24 → 56 (32000) → 41 (143000), with the maximum
rising to **+0.723**. The top ten at 143000 — `L11H14` 0.723, `L13H5` 0.708,
`L18H8` 0.675, `L12H8` 0.661, `L10H0` 0.649, `L9H0` 0.635, `L17H10`, `L20H15`,
`L11H2`, `L17H6` — sit in **layers 9–20, all downstream of `L7H8`**. So the
measure is not blind: it finds copying where copying is, and reports its absence
at the induction heads.

*O3 — two internal consistency checks nobody arranged.* `L9H8`, whose `ΔOV_nll`
is **negative**, has a **negative** copying score at every step (−0.096 to
−0.109, rank ~305–312). And `L9H9` — the one induction head with a real positive
OV effect besides `L7H8` — is the one induction head that **does** become a
modest copier (+0.206 at 16000, rank ~104), and it sits in layer 9, at the
boundary of the copier band. The measure tracks the causal readout where the two
should agree.

*O4 — LN sensitivity is discharged.* Median raw against final-LN-gain-and-
centring folded, at all eight steps: differences in the **fourth decimal**
(+0.0324 vs +0.0293 at 143000). The softest half of §3.11's copy-score reading
is no longer load-bearing anywhere.

**What this sharpens into, and it is the live question now.** `L7H8` has the
largest OV causal effect in its layer (10× the layer mean, growing sixfold across
training) **and is an anti-copier by the token-identity test**. Both now rest on
solid instruments. So its OV write is causally important for repeated-token
prediction *without* being token-identity copying — which **falsifies the
standard account's central claim for this head**, on the standard account's own
measure.

The natural reading is that the circuit has **three stages, not two**:

    L5H2  (layer 5)     positional matcher, prev-token
    L7H8  (layer 7)     content matcher (QK symmetry 0.956), writes NOT token identity
    L9-L20              the actual token-identity copiers

If that holds, §3.11 has been calling `L7H8` "the copier" because ablating its OV
moves second-copy NLL — but it may be **upstream** of the copier and its ΔNLL
mediated. Testable with machinery already built: composition `L7H8 →` the
downstream copiers' `V` and `K` paths, and whether ablating `L7H8`'s OV
suppresses their contribution. **One cross-reference already points that way:**
`L10H3` is a top-ten copier (0.530 at step 4000) and was §3.12-J's counterexample
— high QK symmetry, induction rank 382 of 384. It copies without matching;
`L7H8` matches without copying. The dissociation now has named heads on both
sides.

**P. The three-stage reading is FALSIFIED, on both tests (2026-09-09,
`tools/run/three_stage_mediation.py`).** Step 16000, 16 sequences, restore exact.
`ΔNLL(ablate L7H8 alone) = +1.1069` against a baseline of 0.5335.

*P1 — mediation: no sub-additivity anywhere, and three copiers are
**super**-additive.* `I = ΔNLL(both) − ΔNLL(L7H8) − ΔNLL(C)`:

| head | kind | ΔNLL(C) | ΔNLL(both) | **I** | reading |
|---|---|---|---|---|---|
| `L13H5` | copier | −0.0000 | +1.1259 | +0.019 | independent |
| `L12H8` | copier | +0.0104 | +1.1331 | +0.016 | independent |
| `L9H0` | copier | +0.0320 | +1.1810 | **+0.042** | super-additive |
| `L10H0` | copier | +0.0452 | +1.2703 | **+0.118** | super-additive |
| **`L11H14`** | copier | **+0.1869** | **+2.2753** | **+0.982** | **super-additive** |
| `L11H10` | ctrl | +0.0004 | +1.1252 | +0.018 | independent |
| `L11H0` | ctrl | +0.0157 | +1.1324 | +0.010 | independent |
| `L9H15` | ctrl | +0.0037 | +1.1117 | +0.001 | independent |

A serial circuit predicts `I < 0` — once the upstream stage is gone there is less
for the downstream one to do. **Every measured `I` is ≥ 0.** The clearest case
inverts the prediction outright: `L11H14` alone costs +0.187 and `L7H8` alone
costs +1.107, but **both together cost +2.275**. With `L7H8` intact, `L11H14`
barely matters; with `L7H8` gone, removing it costs five times as much. That is
**redundancy between parallel paths that partially substitute for each other**,
not mediation.

*And the sign is conservative.* §3.12-M5 measured the readout as **compressive**
(half the operator does 30 % of the damage), and a compressive readout pushes
genuinely independent contributions toward *apparent sub-additivity*. Observing
super-additivity against that bias strengthens the reading rather than weakening
it.

*P2 — composition: the prediction fails, and the controls are why we know.*
`L7H8`'s OV into each copier's read paths, ranked against **every head in the
layers below it**:

| head | kind | Q rank (z) | K rank (z) | V rank (z) |
|---|---|---|---|---|
| `L9H0` | copier | **4** (+2.81) | 63 (−0.06) | 8 (+1.39) |
| `L11H14` | copier | 6 (+2.67) | 142 (−0.54) | 15 (+1.07) |
| `L10H0` | copier | 9 (+1.85) | 99 (−0.31) | 73 (+0.25) |
| `L13H5` | copier | 54 (+0.37) | 59 (+0.39) | 10 (+1.59) |
| `L12H8` | copier | 33 (+0.86) | 58 (+0.32) | 102 (−0.12) |
| **`L11H0`** | **ctrl** | **0 (+3.02)** | 61 (+0.21) | 48 (+0.54) |
| **`L9H15`** | **ctrl** | 39 (+0.33) | 52 (+0.02) | **4 (+2.83)** |
| `L11H10` | ctrl | 113 (−0.51) | 77 (−0.09) | 69 (+0.04) |

The three-stage prediction was **elevated V-composition** — the copier copies
what `L7H8` wrote. Some copiers are modestly elevated on V (ranks 8–15), **but
the control `L9H15` ranks 4th at z +2.83, above every copier.** Q looks elevated
for copiers until the control `L11H0` ranks **0th at z +3.02**, above every
copier. K is flat everywhere. **On both paths a non-copier control beats the
copiers, so composition supplies no evidence for a specific `L7H8` → copier
pathway.** Without the controls, "V rank 10, z +1.59" would have been read as
support; this is the §3.12-H1 lesson (a population control *per path*) paying for
itself a second time.

*P3 — what survives, and it is a sharper puzzle than before.* `L7H8`'s OV write
is **causally enormous** (+1.107 at step 16000, ~2× the baseline NLL), is **not
token-identity copying** (§3.12-O), and **does not route through the heads that
do token-identity copying** (P1, P2). Three explanations remain, and the first is
a gap in this test rather than a hypothesis:

1. **It writes to MLPs, which were never measured.** Every composition score in
   §3.12 is head→head. MLPs are the majority of the parameters and the obvious
   place for a non-token-identity signal to be read. **This is the next
   measurement.**
2. It acts on the unembedding directly but not by token identity — boosting a
   *class*, or suppressing alternatives.
3. It acts on residual-stream geometry rather than any single readable
   direction — which is the particle account's own claim, and the one §3.14.2's
   case-study programme is built to examine.

**Q. Five probes: three negative, and the surviving one corrects a framing
(2026-09-09, `tools/run/what_l7h8_writes.py`, step 16000).** Ordered by what
could reframe the question rather than by convenience; MLPs deliberately last.

*Q1 — the offset. My off-by-one worry was WRONG, and that deepens the puzzle.*
`induction_candidates` documents two conventions and the repo uses the
non-standard one (`ids[key-1] == ids[query-1]`, pairing query `N_REP+j` with key
`j`, the **same-token** position) rather than the Anthropic one
(`ids[key-1] == ids[query]`, pairing with `j+1`, the **successor**). Measured
attention from query `N_REP+j`:

| offset | j−2 | j−1 | **j** | j+1 | j+2 |
|---|---|---|---|---|---|
| `L7H8` | 0.0012 | 0.0013 | **0.9339** | 0.0000 | 0.0000 |

**93.4 % at exactly `j`, and nothing at `j+1`.** So the repo's convention
correctly describes this head. But copying from the same-token position returns
the **current** token when the answer is the **successor** — so a token-identity
copier here would be actively wrong, which is consistent with §3.12-O's near-zero
score and makes "what does it write" harder, not easier.

*Q2 — the effect is a large GLOBAL logit shift, not a targeted promotion.*
`ΔNLL = −(Δlogit_correct − Δlogsumexp)` exactly. Ablating `L7H8`'s OV gives
`Δlogit_correct = −5.02` **and** `Δlogsumexp = −3.91` — every logit falls by
about four nats and the correct one by five, netting +1.11. This is not a
delicate promotion of one token; it is the removal of a large component of the
residual.

*Q3 — and the control says that is specific, not generic — while correcting a
framing this document has carried.* Ablating each head's OV, against a baseline
logsumexp of 16.05 and final residual norm of 52.95:

| head | ΔNLL | Δlogit(correct) | Δlogsumexp | Δ‖resid‖ |
|---|---|---|---|---|
| **`L5H2`** (prev-token) | **+2.227** | −5.89 | −3.67 | **−5.92** |
| `L7H8` | +1.107 | −5.02 | −3.91 | **−8.38** |
| `L11H14` (top copier) | +0.187 | −1.02 | −0.83 | −0.40 |
| `L13H5` (top copier) | −0.000 | −0.02 | −0.02 | −0.06 |
| `L2H10`, `L0H0`, `L20H7` | ≤ +0.008 | ≤ 0.10 | ≤ 0.10 | ≤ 0.19 |

Two things. **The global shift is not generic** — ordinary heads move logsumexp
by under 0.1 where these two move it by ~4, and they remove 11–16 % of the final
residual norm against under 0.4 % for the rest. And **`L5H2` has TWICE `L7H8`'s
effect**. §3.11's "largest of all 16 layer-7 heads" is true and has been read
too broadly: `L7H8` is not the largest OV effect in the circuit, its own
prev-token partner is. Every "the causally load-bearing half" statement needs
that qualifier.

**Caveat, not yet closed:** the two large heads may simply have larger OV norms,
and norm-proportionality is not ruled out here. The check is one sweep of
`‖OV‖_F` against `Δ‖resid‖` across heads.

*Q4 — QK against OV on the same readout.* Ablating the static QK costs +0.616,
**56 % of the OV ablation's +1.107**. §3.11 only ever read attention for the QK
half (0.92 → 0.02); on NLL, attending correctly is worth a bit over half of what
the head is worth in total.

*Q5 — the composed circuit is MORE anti-copying, not less.* §3.12-O's score uses
the raw embedding as the OV's input, but the residual at the attended position
has already been written by `L5H2`. The composed path
`W_U OV(L7H8) OV(L5H2) W_Eᵀ` scores **−0.0997** against the direct
`W_U OV(L7H8) W_Eᵀ` at **−0.0496** — twice as negative. So it is not copying the
prev-token signal either.

*Q6 — the MLPs are negative, which is why they were not run first.* `L7H8`'s OV
into each downstream MLP's input projection, ranked against every head below that
layer: the best is layer 10 at **rank 44 of 159, z +0.57**, layer 9 at rank 47 of
143 (z +0.31), and **every other layer is at or below the population median**
(z −0.16 to −0.73). No elevated MLP pathway anywhere.

**Where this leaves it.** Of the three §3.12-P explanations, (1) MLPs is
**negative** (Q6) and (2) unembedding-without-token-identity is **negative in
the targeted sense** (Q2: the shift is global, not selective). What survives is
**(3), residual-stream geometry** — and it now has a measurement behind it rather
than being the leftover option: `L5H2` and `L7H8` each remove a tenth or more of
the final residual norm, two orders of magnitude more than an ordinary head,
while shifting every logit by ~4 nats. That is the particle account's own claim
arrived at by eliminating the alternatives on their own instruments, and it is
exactly what §3.14.2's case-study programme was queued to examine.

---

## 3.14 How this work is organised, and what is queued (2026-09-09)

### 3.14.1 Three objects, not one thread

§§3.11–3.12 have been running as a single "induction programme", and they are
not one. Three research objects are tangled in them, with **different nulls,
different exchangeable units, and different failure modes**, and separating them
is the largest structural improvement available:

| | object | exchangeable unit | state |
|---|---|---|---|
| **7a** | population spectral development — the OV repulsive collapse (§3.12-A), its relation to `CLAIM-B`'s window, the Phase-2 `frac_repulsive` comparison | head, with a shared-model factor | biggest signal, hardest null |
| **7b** | circuit mechanism at a fixed step — rank sweeps, S/A interventions, composition, alignment | the perturbation draw, so `n = 1` is sound | most of §3.12; closest to registrable |
| **7c** | circuit formation — the backward search from a behavioural anchor, the pythia-70m study (§3.9) | checkpoint / window — where co-location circularity lives | scoped, not started |

Splitting them says immediately what can be registered when, and stops 7a's hard
null blocking 7b's clean one. Note that the labels are for *this document's*
organisation; they are not new phase directories, and §3's numbering is
unchanged (`INDEX.md`'s rule: do not rename directories).

### 3.14.2 Queued: `7d`, the case-study programme — **the author's direction,
recorded 2026-09-09, not started**

Take **one induction circuit as a case study** and follow **each of its three
stages** (§3.12-O: positional matcher → content matcher → token copier) across
**every checkpoint**, asking three things of each stage:

1. **How does it evolve?** Per-stage developmental series, not just the endpoint
   snapshots §3.11 has been taking.
2. **Does it correlate with a phase?** Against the transitions already on the
   axis — the population repulsive collapse and the 512–2000 window (§3.12-A,
   noting §3.12-N3: that window is the literature's, not ours), the OV plane
   rotation (§3.12-F2), the numerical-abscissa takeoff (§3.12-G5), the
   composition switch-on (§3.12-H1), the QK symmetry takeoff (§3.12-I).
3. **Is there particle dynamics inside it?** This is where the project's own
   frame (`core/particles.py`, `core/dual_reading.py`, `P-I5`'s geometric
   reading) attaches to a circuit rather than to a population — the thing the
   retired co-location frame was reaching for and could not have without
   circularity (§3.10, `POPPER_PLAN.md` §6w).

Then **repeat over as many circuits as can be found** — other induction heads in
410m, the copier band of layers 9–20, and 70m under §3.9's own registered grid —
and ask the question the single case cannot: **is the particle dynamics
correlated ACROSS cases?** A shared dynamical signature over independent circuits
is a population claim that no single circuit can make, and it is the natural home
for anything §3.12 produces that wants to generalise.

*Why it is queued rather than started.* It needs 7b's mechanism settled first —
§3.12-O has just moved the target from "which head copies" to "what does `L7H8`
write", and a case study built on the two-stage reading would be built on a
picture that is currently being revised. It also needs the `dual_reading`
pairwise-field extension that `P-I5` is blocked on (§3.12-C).

### 3.14.3 Defects found this session, not yet fixed

All three are **reporting or documentation** defects. None touches a p-value,
`claims/registry.json` is unchanged and `claims/adjudications/` is empty, so
nothing registered is affected — but each is live in code a reader would trust.

1. **`induction_rank_sweep.truncate`'s `random` branch docstring is false**
   (§3.12-L). It claims *"Matched-norm random rank-r control … the operator norm
   scale and the factor structure match the real truncation."* At `r = 1` the
   energy ratio to the `svd` arm is **16×**. Either the docstring changes or the
   branch rescales; the §3.11 Stage-1 table must be read against §3.12-L's
   energy-matched version either way.
2. **`target_vs_reference` computes a self-inclusive z** (§3.13.1), which is
   capped at `(n−1)/√n = 3.75` for `n = 16`. `L7H8` at step 8000 reads **3.74** —
   saturated, against a leave-one-out value of **50.2**. Fix is leave-one-out.
3. **`ov_factors` returns `OV_h = (W_O W_V)ᵀ`**, the transpose of the
   residual-stream operator (§3.12-O, verified at relative error 0.0). Harmless
   for every quantity read off it so far — all transpose-invariant — and the
   docstring does state the convention, but nothing warns that a **directional**
   read (any circuit with `W_E` on one side and `W_U` on the other) must not use
   it. A one-line warning would have saved a careful check.

---

## 3.13 When the mean is the wrong instrument (2026-09-09)

`§3.12-G6` found a signal that Spearman could not see: fifteen of sixteen heads
per layer carry causal noise, so a correlation over all sixteen averages the
signal away, while the *rank of the top head* recovers it at ~2.6e-4. That is a
general lesson and this section is where it is logged, because the sites it
applies to are spread across the repo.

**The principle, and its edge.** A mean or correlation is the right summary when
the effect is *distributed* across the population; an extremum — max, argmax,
top-rank — is right when it is *concentrated* in a few members. **Choosing
between them after seeing the data is exactly the selection `claims/registry.json`
exists to forbid.** So the rule is not "use max": it is

- **exploratory work reports both, always** (`§3.12-G6` was legitimate because
  nothing is registered and both were reported);
- **registered work has already frozen the choice**, and the alternative may be
  *reported beside* the result but never swapped in — the same discipline the
  registry already applies to `p_reciprocal`, which is a stop-rule input and
  enters no `E`.

### 3.13.1 The arithmetic sub-case, which is not a judgment call

A z-score computed against a population **that contains the point being
scored** is capped. For a sample of `n`, the largest attainable studentized
deviate is `(n − 1)/√n` — at `n = 16`, **3.75**. The statistic cannot report a
larger effect however large the effect is.

`induction_subspace_characterize`'s `target_vs_reference` computes the target
head's z against `layer_mean` / `layer_sd` **over all 16 heads, the target
included**. Measured on the seven populations already on disk:

| population | top head | ΔOV_nll | z (self-included) | z (leave-one-out) | |
|---|---|---|---|---|---|
| 4000 / L7 | H8 | 0.244 | 3.69 | **20.59** | 5.6× |
| **8000 / L7** | **H8** | **0.725** | **3.74** | **50.17** | **13.4×** |
| 8000 / L9 | H5 | 0.187 | 2.93 | 4.84 | 1.7× |
| 2000 / L9 | H5 | 0.317 | 3.60 | 13.25 | 3.7× |
| 16000 / L1 | H15 | 0.040 | 3.35 | 7.72 | 2.3× |
| 16000 / L6 | H0 | 0.021 | 2.23 | 2.86 | 1.3× |
| 32000 / L2 | H2 | 0.056 | 2.48 | 3.41 | 1.4× |

**`L7H8` at step 8000 measures 3.74 against a ceiling of 3.75 — saturated.** A
reader comparing it to step 4000's 3.69 would conclude the effect barely moved;
the leave-one-out z went **20.6 → 50.2**. The reported number stopped being a
measurement and became the ceiling.

This is a **reporting defect, not a scoring one** — no p-value is computed from
this z and `claims/adjudications/` is empty, so nothing registered is affected.
`core/nulls.py`'s z is *not* subject to it: there `observed` is scored against a
null distribution it is not a member of, and where the identity permutation *is*
included the draw count is large enough that `(P−1)/√P` is far above anything
attainable. **The fix is leave-one-out, and it is a docstring and three lines.**

### 3.13.2 Sites surveyed

*Already handled — recorded so they are not re-litigated.* `tools/run/
behavioural.py` prints `mean` **and** `max` side by side. `P-M1`'s registry
entry computes the mean/min/max head-to-layer aggregates, declares `mean`
primary *in advance* so it cannot be picked after the fact, and **refuses a
p-value when the three disagree in sign**. `CLAIM-B` reports dispersion beside
every centroid precisely so a bimodal profile is visible. `ov_per_head.py`
records the energy split *and* the count because their disagreement is the
reading (`§3.12-F0` finally read it).

*Worth a look, none registered-blocking.* `p7_motifs/motif_stats.py`'s
`mean_ind` / `mean_non` are means over head sets — `P-I3`'s registered statistic
already superseded them with a rank-based matched contrast, so the means are
diagnostics; they should say so. `p2d_operator_activation/run_2d.py`'s
`head_mean` energy series is `P-M1`'s aggregate and inherits that entry's
refusal.

*A frozen entry whose stated reason does not distinguish the two.* `CLAIM-C`
takes `delta = mean over normalized depth`. Its `null_construction` justifies
the choice against a depth band: *"Blog 1 quotes layers 5-30 of gpt2-large, but
a depth band is a choice with as many options as there are bands."* That
argument is sound against a **band** — and **a max over depth places no constant
either**, so it does not distinguish mean from max at all. The registered
wording is frozen and stays; this is recorded because the reasoning has a gap,
not because the entry should change. Blog 1 quoting a band is itself weak
evidence the contrast is depth-concentrated.

### 3.13.3 What this suggests we have overlooked: the position axis

Every readout in `§3.11`–`§3.12` is a mean over **token positions**, and nothing
has ever looked at that axis:

- `behavioural_induction_score` and `induction_attention` are
  `picked.mean()` over `N_REP = 96` second-copy positions × 8 sequences;
- `second_copy_nll` — the readout the **entire** OV rank sweep, the 82 %, and
  every ΔOV_nll in this document rest on — is a mean over the repeated half.

If copying is concentrated at particular positions (later second-copy positions
have more context, so concentration is the expected shape rather than an exotic
one), then the mean dilutes it and the rank sweep's `r*` is being read off a
diluted curve. **This is the same error `§3.12-G6` found, one axis over, in the
measurement everything else depends on.** It is one forward pass to check: emit
the per-position NLL delta instead of its mean and look at the profile. Report
mean and max together, per 3.13's own rule.

**MEASURED 2026-09-09 (`tools/run/induction_position_profile.py`, `L7H8` at
step 4000, 64 sequences — eight times the original readout's sample count).
THE HYPOTHESIS ABOVE IS WRONG, and the negative is worth more than the positive
would have been.**

*The effect is not position-concentrated.* **37 of 96** positions carry half the
ΔNLL mass (uniform would be 48); ninety percent needs **80 of 96**; the top
decile of positions carries **0.178** of the total against 0.100 for uniform.
Mildly above uniform, nowhere near "a few members".

*And `r*` does not move.* Recovered fraction by rank, computed as a **ratio of
sums** over each position set rather than a mean of per-position ratios (the
per-position denominator is near zero where the head does nothing):

| rank | all positions | concentrated half | dilute half | excluding j=0 |
|---|---|---|---|---|
| 1 | **0.824** | 0.791 | 0.858 | 0.822 |
| 2 | 0.850 | 0.832 | 0.869 | 0.849 |
| 4 | 0.871 | 0.866 | 0.877 | 0.870 |
| 8 | 0.945 | 0.943 | 0.947 | 0.944 |
| 16 | 0.971 | 0.972 | 0.971 | 0.971 |
| 64 | 1.001 | 1.001 | 1.000 | 1.001 |

The concentrated and dilute halves differ by at most **0.067** at rank 1 and are
identical to three digits by rank 8. **The 82 % is not an artifact of
averaging**, and `r*_SVD = 1` survives the axis that could have dissolved it.
Restore check exact (`0.000e+00`).

*The methodological reading, which is the point of §3.13.* The principle says
**report both**, not *expect the extremum to win*. Here the mean was the right
instrument and the check confirms a result rather than overturning one. §3.13
is not a licence to prefer extrema — it is a requirement to look, and looking is
cheap.

---

## 4. Open, analysed, not yet acted on: the scoring threshold

Investigated 2026-09-03, nothing changed in code. Recorded here because it is
measured and it affects every gate.

Every gate refuses when `attainable_floor > alpha`. The e-process validates at
`E >= 1/alpha`. With κ = 0.5 those are different requirements: for a claim
carrying k factors each at its floor, `p <= (κ·α^(1/k))^(1/(1−κ))`.

| k factors | required p | vs. the α the gates check |
|---|---|---|
| 1 | 6.25e-4 | **80× stricter** |
| 2 | 0.0125 | 4× stricter |
| 4 | 0.0559 | `p ≤ α` suffices |

`H-EMERGE`, `H-TRANSFER` and `H-RESIST` each have **exactly one** active
e-value row. `H-BRIDGE` and `H-OPERATOR` have four each.

**CLAIM-B on a perfect input returns p = 0.05 on all five seeds** — its arms
combine by max and the anchor arm is floored at `1/(n_controls+1)` with the 19
controls its dry run uses. `claims/audits/claim_b_p_i1_dry_run.json` already
carries `floor_equals_alpha: True`. That is e = 2.24 against a threshold of 20:
falsifiable via the RE-ANCHORS branch, **not validatable**, at any data.

`core/evalues.py:216`'s `required_p_for_rejection` already computes the right
number and **no gate calls it**. That is the whole defect in one line.

Best floors across the committed audits as e-value ceilings: `P-T1`/`P-M1` 183,
`CLAIM-C` 22.6, `P-I1` 22.4, `P-ST1` 1.58.

Measured, on 400k H0 replicates at N = 2000: a randomization p is discrete on a
known grid, so `e = (N+1)/(R·H_{N+1})` is a valid e-value directly (E[e] =
1.0005 under H0) and returns **244.7** at rank 1 where `calibrate(p)` returns
22.4. κ registered per prediction from its own floor (κ\* = 1/ln(1/p_floor))
gives 96.8. Neither fixes the tie floor: with heavy ties and the conservative
convention `paired_colocation_arm` correctly uses, E[e] = 0.28 under H0 — the
design cannot produce evidence under any scoring rule.

Nothing here has been implemented. The cheapest structural win is **more factors
per claim**, not a better calibrator.

---

## 5. Registered decisions, and disk that must not be deleted

### 5.1 Registered — do not re-decide from the code

1. **`P-I1`'s grid** — `REGISTERED_P_I1_SWEEP`, 19 steps: `0, 1, 2, 4, 8, 16,
   32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000`.
   A superset of the CLAIM-B sweep. All 19 tables are on disk.
2. **`P_I1_RELAY_OWNER = "matcher"`** — `p7_motifs/formation_gate.py:143`.
3. **Endpoints** — steps 0 and 143000 are in the grid, which is what
   `endpoint_flags` needs.
4. **`P_I1_DOMINANT_PROMPT = "repeated_tokens"`** — kept, carried beside the
   excluding-it series, reported and never scored. It holds **34,191** induction
   pairs against the next prompt's 2,873, because every repeated token pairs with
   every other; its 61% share is a fact about the prompt, not the checkpoint.
5. **`CLAIM-B`'s sweep** — `REGISTERED_CLAIM_B_SWEEP`, chosen 2026-08-28 from the
   computed feasible set (`POPPER_PLAN.md` §6r).
6. **`P-I3`'s matching** — `"score_and_layer"`, registered 2026-08-30 with both
   sides measured (§6s).

### 5.2 `results/` holds 132 GB and must keep it

`2026-08-12_05-01-35` (56.5 GB) and `p2_eigenspectra_2026-08-13_05-13-52`
(74.2 GB) each cover **27 steps on the PILOT schedule** — 11000, 13000, 15000,
17000, 19000, 100000, 120000 and so on. Those steps appear in nothing else on
disk, and `core/pythia_registry.py` keeps `PYTHIA_410M_PILOT_STEPS` loadable for
exactly this reason. `p1b_pilot`, `p2b_pilot`, `p2d_pilot` and `phase3` are
small; `phase3` is referenced from `archive/`.

---

## 6. Untouched, and named so it is not mistaken for done

* `core/precision_policy.py`'s **P2** (Pythia ships fp16; an fp16-epsilon
  perturbation splits a genuinely real eigenvalue pair into a complex one) and
  **item 13** (the forward pass runs under bf16 autocast).
* **`real_frac`/`imag_frac` are NaN in every row of every table** — deliberate
  and correctly recorded (`rotational_channel: "absent"` in the manifest), not a
  silent gap. **Both open questions answered 2026-09-04, nothing changed in
  code.** No registered prediction needs the rotational channel: `P-I2` names
  only the sign channel (`U_pos`), and `P-I1`/`P-I3`/`P-I4` don't reference
  `real_frac`/`imag_frac` at all. And no consumer reads them for a computation —
  grepped across every `.py` file: `run_7.py` writes the NaN, `p7_io.py` is the
  seam that would fill it (`rotational_channel_from_blocks`, unwired), and
  `core/interactions.py` / `core/artifacts.py` only carry the schema and
  validation. None of `motif_stats.py`, `formation_gate.py`,
  `formation_curve.py`, `cross_head_gate.py`, `patching_gate.py`, `events.py` or
  `motif_alphabet.py` touch either column. (`core/dual_reading.py` computes
  fields with the same names but is an unrelated per-particle primitive from an
  earlier phase, not a Phase 7 consumer.) So the columns are exactly what §6
  asked whether they were: schema nothing fills and nothing reads. Left as is —
  removing them would touch `InteractionTable`'s hashed schema for a channel
  Phase 2b's `extract_schur_blocks` could still wire in later, and no registered
  prediction is asking for the removal either. `p7_io.rotational_channel_from_blocks`
  stays the seam if that changes.
* **The phase-7 manifest records no library versions.** §1's first trap is the
  argument for adding them; not done, because it changes the manifest schema and
  every record that hashes it.
* **The in-memory categorical option** (int8 codes for `model`/`prompt_key`/
  `pair_type`, 5.49 GB → 1.89 GB expanded). Compression fixed disk and does
  nothing for RAM.
* **Eleven predictions are adjudicable in principle and
  `claims/adjudications/` is empty.**
* **`data/analysis/` is git-ignored, so every number quoted in a committed
  document is reproducible only by re-running its producer.** `data/` is
  ignored by `*` (§1), which is right for the 118 GB of bulk and wrong for the
  small JSON series and the builder scripts sitting beside them:
  `dissipation_series`, `dissipation_sublayer_series`, `ov_per_head_series`,
  `relay_null_series_k50`/`_k100`, `behavioural_series`, and the five/six
  `build_*.py` / `*_analysis.py` scripts. `POPPER_PLAN.md` §6w, §3.8 and
  `p2_eigenspectra/status-2.md`'s dated section all quote figures whose only
  provenance is a file outside version control. The producers are tracked and
  deterministic, so this is recoverable rather than lost — but it is inference
  from a rerun, not a record, which is the same class of gap
  `docs/results_provenance_audit_2026-09-05.md` §3.1 raises against Phase 2's
  missing manifest. **Fix: un-ignore `data/analysis/*.json` and
  `data/analysis/*.py`** (a `!` rule under the `data/` ignore), or move them to
  a tracked `results/analysis/`. Deferred deliberately — it is a chore, and the
  decision on which of the two shapes to take is not yet made.

---

## 7. Reproducing anything

```bash
cd /run/media/system/WDS_500/Mets && source .venv/bin/activate
export HF_HOME=$PWD/data/hf METS_RESULTS_DIR=$PWD/data/phase12 HF_HUB_OFFLINE=1

./scripts/check.sh gate     # tier 0 + 1, what gates a merge; ~35 s
./scripts/check.sh all      # adds the deps tier; ~2:15

bash tools/run/sweep.sh     # resumable; all 19 steps present, so it is a no-op
python tools/run/curve.py   # ~2:43, writes data/analysis/curve.json AND
                            # data/analysis/formation_series.json

python3 -m tools.run.behavioural --write    # ~1:06, reads 19x8 attentions.npz,
                                            # writes data/analysis/behavioural_series.json
python3 -m tools.run.behavioural --check    # structural checks on the written series

python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s, needs the series
python3 -m tools.p_i1_attainable_floor --check     # needs no data

METS_NULL_REPLICATES=50 python3 -m tools.run.relay_null
                            # ~1:33 at 50 reps (measured); ~3:00+ at 100 --
                            # scales close to linearly in the replicate count.
                            # Writes data/analysis/relay_null_series.json.
                            # Prints one line per checkpoint as it goes, so a
                            # long run can be judged and killed early; the
                            # output file is only written at the very end, so
                            # killing it mid-run loses nothing already on disk
                            # (the PREVIOUS successful --write, if any, is
                            # untouched until the new run's last line prints).
python3 -m tools.score_p_i1
                            # needs relay_null_series.json and
                            # behavioural_series.json; prints P-I1's p-value

METS_REPO=$PWD METS_DATA=$PWD/data python3 -m tools.run.dissipation
                            # ~22 min. Dissipation identity Tier A per
                            # (step, prompt, layer) from activations.npz +
                            # the on-disk OV Schur projectors, no forward
                            # pass. Writes data/analysis/dissipation_series.json.
METS_REPO=$PWD METS_DATA=$PWD/data python3 -m tools.run.dissipation_sublayer
                            # ~20 min, 133 forward passes (loads pythia-410m
                            # checkpoints). Exact attn/FFN split + per-head
                            # roll-up. Writes dissipation_sublayer_series.json.
# Panels + the per-head co-location test are rebuilt by the scripts in
# data/analysis/ (build_colocation_panel.py, build_dissipation_panel.py,
# build_earlylayer_and_512to1k.py, tierB_panel_and_colocation.py).

METS_REPO=/run/media/system/WDS_500/Mets METS_DATA=$METS_REPO/data \
  python3 -m tools.run.induction_rank_sweep
                            # ~4 min, loads pythia-410m-step4000. Stage 1 of
                            # sec 3.11: OV rank sweep of L7H8 in the SVD, Schur
                            # and random bases. Writes induction_rank_sweep.json.
METS_REPO=/run/media/system/WDS_500/Mets METS_DATA=$METS_REPO/data \
  python3 -m tools.run.induction_subspace_characterize
                            # ~5.5 min. Stage 2: L7H8 OV r* characterisation
                            # (Schur sign, phi, Henrici, rank-1 mode) calibrated
                            # against all 16 layer-7 heads + 24 random OVs.
                            # Copying read causally (final LN present). Writes
                            # induction_subspace_characterize.json.
                            # NB METS_REPO must be the canonical path (matches
                            # sys.prefix); $PWD via the bind mount fails the
                            # interpreter check.
```

### 7.1 `curve.json` is the artifact that gets diffed

Every change to the storage or estimator layer is verified by re-running
`curve.py` and diffing `curve.json` against the pre-change copy. It has come
back **0 differences** three times: after the single-pass rewrite and table
compression, after the 176 GB migration, and after adding the per-head series
dump. That is why the series went into a **second** file — a file whose content
is diffed is not the place to add a key.

### 7.2 Records that carry file hashes

Three records hash `core/changepoint_colocation.py` or
`p7_motifs/formation_gate.py` and must be rewritten whenever either changes. The
gate fails loudly if they are stale, which is the intended behaviour.

```bash
python3 -m tools.dry_run_claim_b_p_i1 --write      # ~4 min
python3 -m tools.claim_b_grid_feasibility --write  # ~3:45
python3 -m tools.p_i1_attainable_floor --write     # ~0.2 s
```

### 7.3 One gotcha

`pythonpath = .` in `pytest.ini` applies to pytest only. A plain
`python script.py` needs `PYTHONPATH` set, which `tools/run/curve.py` does for
itself.

---

## 8. Where to read next

| Question | File |
|---|---|
| Which phase lives where, what is archived | `INDEX.md` |
| Why a construction is the way it is | `POPPER_PLAN.md` §6a–§6t |
| What is pre-registered, and its falsifier | `PREDICTIONS.md`, `claims/registry.json` |
| Which predictions can carry an e-value, and the order to build a null in | `claims/EVALUABILITY.md` |
| Phase 7's translation table and motif alphabet | `p7_motifs/design-7.md` |
| A phase's current state | `<phase>/status-N.md` |
| The dissipation-identity run — what it is, Tiers A/B, v2 list | `docs/dissipation_checkpoint_axis_scoping.md`, §3.8 |
| Are the on-disk phase12/phase7 results stale? | `docs/results_provenance_audit_2026-09-05.md` |
| The pythia-70m dense-onset run, and how it could plug in | §3.9 |
| What changed and when | `git log` |
