# docs/dissipation_checkpoint_axis_scoping.md — the dissipation-identity run on the P-I1 checkpoint axis

**Status:** scoped, not built. Written 2026-09-05 while the K=100 relay-null rerun
(PROJECT.md §3.7) was in flight. Nothing here is registered; §5 says what would
have to be before any of it adjudicates a prediction.

**One line.** `core/dissipation.py` is built and tested (commit `821a1ef`,
`tests/test_core_dissipation.py`) and **has never been evaluated on a Pythia
checkpoint**. This is the plan to run it across the registered 19-step P-I1 grid
× 7 prompts × 24 layer boundaries, producing the per-particle energy-dissipation
decomposition MATH_SPECTRAL_OT §5 describes and PROJECT.md's co-location work
needs.

---

## 0. Why

Three questions it addresses, in priority order:

1. **A per-head co-location anchor for P-I1 that is not floored at step 4000.**
   The relay count is a structural zero until step 4000 (PROJECT.md §3.5), so it
   has 4 informative points on the axis and its change "can't be located below
   log-step 3.6". ΔE_β dissipation is defined at *every* checkpoint including
   step 0. A per-head series — "where head *h*'s behavioural score rises, does
   its attention dissipation on E_β turn repulsive-dominant / does its
   gradient-flow alignment move" — can feed the same
   `core.changepoint_colocation.paired_colocation_arm` the relay side uses, as
   an alternative B-anchor.
2. **Phase 2 open item 5.** `frac_repulsive` decays 1.00 → 0.56 → 0.80 over steps
   8000–143000 with the violation *count* flat (status-2.md; the co-location
   panel, `data/analysis/colocation_panel.png`). "Something reorganizes which
   subspace the violations occupy without changing how many." `dissipation_by_
   subspace` per head per checkpoint is the dissipation-side counterpart: if the
   induction heads' `d_repulsive` share is what declines across that window, that
   is the mechanism, tied to the local circuit event.
3. **MATH_SPECTRAL_OT §5.2 / §5.3(d).** "E_beta is the project's central object
   … the gradient that produces [every violation's] sign has never been
   evaluated." This computes it, and reports the linearization residual
   `actual ΔE_β − Σᵢ⟨Gᵢ,vᵢ⟩` as its own quantity — a test of whether the
   forward-Euler/ODE framing the project assumes holds at each layer.

It is also the measured counterpart of Phase 2d's D1 (`gradient_flow_alignment`
asks, on activations, the "is this head a gradient flow of E_β" question D1 asks
from the weights).

---

## 1. Tier A — computable now, from artifacts already on disk, no forward pass

| Input | Where | Shape |
|---|---|---|
| raw residual-stream states | `data/phase12/<ts>/pythia-410m-step<S>_<prompt>/activations.npz` key `activations` | `(25, n_tok, 1024)` = emb + 24 blocks |
| per-checkpoint OV Schur projectors | `data/phase12/p2_eigenspectra_<ts>/ov_projectors_pythia-410m-step<S>.npz` | per-layer `schur_attract_layer_ℓ` / `schur_repulse_layer_ℓ`, `(1024,1024)` |

Total per-layer displacement `dXℓ = activations[ℓ+1] − activations[ℓ]` for
ℓ = 0..23, all 19 × 7 = 133 runs. With that plus the on-disk projectors,
**these run with zero model loads:**

- `dissipation(X=activations[ℓ], dX=dXℓ, beta)` → `first_order`, `per_particle`
  (sums to `first_order` exactly), `actual_delta_E`, `residual`,
  `relative_residual`, `step_size`.
- `dissipation_by_subspace(X, dXℓ, beta, P_attract=schur_attract_layer_ℓ,
  P_repulse=schur_repulse_layer_ℓ)` → `attractive`, `repulsive`,
  `per_particle_*`, `sum_check`. **This splits the *total* dX**, not the
  attention channel specifically — see §3.3.
- `gradient_flow_alignment(X, dXℓ, beta)` → per-particle cos(−Gᵢ, vᵢ) with
  `mean`, `median`, `q10`, `q90`, `frac_descending`.

Runtime: a Gram + `exp(βG)` + a few `(n,d)` matmuls per (run, layer), n≈250–512,
d=1024. **Minutes for the whole grid.**

**Not in Tier A:** the attention-vs-FFN split. `dissipation_by_channel` needs
`dX_attn` and `dX_ffn` separately, and every Phase 1 run wrote
`sublayer_semantics: null` (manifest.json) — the streams were not captured.

---

## 2. Tier B — the attn/FFN split, one forward pass per (step, prompt)

`core.sublayer_streams.extract_sublayer_streams(model, tokenizer, text, name)`
returns `post_attn`, `post_ffn` per layer with `semantics="pre-ln-parallel"` on
Pythia, from which `dX_attnℓ = post_attnℓ − x_inℓ`,
`dX_ffnℓ = post_ffnℓ − x_inℓ`, and `dX_attn + dX_ffn == dXℓ` **exactly** (the
property `dissipation_by_channel`'s `sum_check` verifies and §3.4 says to
assert).

- 19 steps × 7 prompts = **133 forward passes**, CPU, ≤512 tokens, d=1024,
  24 layers, `torch.no_grad()`. Model load per checkpoint via
  `core.lm_loading.load_causal_lm(f"pythia-410m-step{S}")` (pinned revision).
- Estimate: ~20–40 s model load per checkpoint (19×) + ~5–15 s per prompt
  forward+capture+write (133×). **~20–45 min total.** Cheap.
- Writes `dX_attn` / `dX_ffn` (or `post_attn` / `post_ffn`) to a sidecar npz per
  run, then Tier-A math re-runs with `dissipation_by_channel` added.

---

## 3. Design decisions to settle before coding

### 3.1 Frame — **sphere, v1**
`core/dissipation.py` computes on the L2 sphere (`l2_normalize`), matching
`interaction_energy` and every E_β number in the project. Its docstring flags
the LN-frame alternative (the operator *attention* applies reads `LN1(x)`;
Phase 2d open item 1). Decision: **sphere only for v1** — consistency with the
whole energy series matters more than the attention-operator reading here. An
LN-frame variant can be carried beside, never scored, using the
`carried_beside` convention P-I1's behavioural arm already uses.

### 3.2 β — **1.0, v1**
Project β set is {0.1, 1.0, 2.0, 5.0}; `frac_repulsive` and every co-location
number is at β=1.0. Run β=1.0; carry β=2.0 beside. (Overflow guard only bites
at β>700, irrelevant here.)

### 3.3 Subspace-projector fidelity — **per-layer on-disk projectors, v1**
The on-disk projectors are a **per-layer aggregated-OV** Schur decomposition
(one `schur_attract_layer_ℓ` per layer, not per head). The honest per-head
version — project each head's own contribution through *its own* head's OV Schur
subspace — needs per-head attention-delta capture (a heavier hook than
`extract_sublayer_streams`) and a per-head decomp. v1 uses the on-disk per-layer
projectors and **documents this as an approximation** (the layer's OV is the
head sum; this is also exactly how Phase 2 itself defines the subspaces). Per-head
is a v2 refinement (§7 B1).

### 3.4 Invariants to assert (not log)
- `dissipation_by_channel.sum_check < 1e-10` — the evidence the parallel-residual
  identity is exact. **Fail loud** if not.
- `dissipation_by_subspace.sum_check` likewise.
- On projector load: symmetry (`‖P − Pᵀ‖`) and idempotency (`‖P² − P‖`), and
  `P_attract + P_repulse ≈ I` within tol.
- `activations.npz` has 25 rows (emb + 24); sublayer streams have 24; align on
  block index; assert `n`.
- Re-tokenisation matches `tokens.txt` (the check `relay_null.py` /
  `behavioural.py` already do).

### 3.5 `repeated_tokens`
Excluded from scored analyses everywhere (degenerate input). Carry beside,
never scored — same as the behavioural and relay sides.

### 3.6 Structural-zero check
Unlike the relay count, ΔE_β dissipation is defined at step 0 (there *are*
energy violations there — 1 in Phase 2, per the panel). No structural-zero
floor. This is the property that makes it a usable co-location anchor below
log-step 3.6.

---

## 4. Artifact

`data/analysis/dissipation_series.json` — a **new** file (curve.json §7.1
lesson: a diffed artifact is not where new keys go). Schema:

```
_what_this_is, steps, prompts_scored (7), carried_beside ("repeated_tokens"),
beta, frame ("l2_sphere"), projector_source, git_sha, hf_revision_scheme,
lib_versions {torch, transformers, numpy, scipy},          # PROJECT.md §1 trap
per_step_layer: { "<step>|<prompt>|<layerℓ>": {
    first_order, actual_delta_E, residual, relative_residual, step_size,
    d_attn, d_ffn, attn_share, sum_check_channel,           # Tier B only
    d_attractive, d_repulsive, sum_check_subspace,
    gfa_mean, gfa_median, gfa_q10, gfa_q90, gfa_frac_descending }, ... },
per_head:  { "<Lℓ,Hh>": {                                   # P-I1 116-forming axis
    d_repulsive_share:   [19 values],   # this head's attn contribution
    d_attractive_share:  [19 values],
    gfa_cos:             [19 values] } }
```

Per-particle arrays (`per_particle`, `per_particle_attn/ffn`,
`per_particle_attr/rep`, gfa per-particle) are large — 133 runs × 24 layers ×
~465 tokens × ~6 float64 arrays ≈ a few GB. **Decision up front:** JSON carries
reductions only; per-particle arrays go to an optional sidecar
`dissipation_particles/step{S}_{prompt}.npz` **only if** a downstream analysis
needs them. v1: reductions only.

The `per_head` roll-up needs per-head attention contributions, which is Tier B.
In Tier A the `per_head` block is omitted and the co-location question is
answered at layer granularity (which layers' dissipation turns repulsive, when)
rather than head granularity.

---

## 5. What would have to be registered before this adjudicates anything

Nothing here is a prediction yet. To take the dissipation co-location series
into `claims/adjudications/` it would need, per POPPER_PLAN §C2's pattern:

- a `claims/registry.json` entry with a **differential falsifier** (what the
  particle account says that the standard account does not, where they disagree
  observably) — e.g. "the induction heads' attention dissipation turns
  repulsive-dominant in the same checkpoint window as their behavioural rise;
  falsified if it turns attractive-dominant, or does not move, or moves at an
  unrelated checkpoint";
- an evaluability classification (`claims/EVALUABILITY.md`) — likely
  `needs-null`, with the null being the same pairing permutation
  `paired_colocation_arm` runs, so the attainable-floor / tie-structure
  analysis of PROJECT.md §3.2–§3.3 applies unchanged;
- the `shared_unit_factor_diagnostic` PROJECT.md §3.3 requires.

v1 is **measurement**: a number with the co-location panel beside it, not an
e-value.

---

## 6. Risks / gotchas

- **venv trap** (PROJECT.md §1): the runner asserts `sys.prefix` and
  torch/transformers versions, never trusts `activate`.
- **No library versions in the phase-7 manifest** (§1, already cost one
  checkpoint): the artifact records `lib_versions` explicitly.
- **Second-order residual size is unknown on real data.** The tests use
  synthetic small-step clouds. If `relative_residual` is O(1) at some layers,
  the per-particle *attribution* is still exact (it is a definition), but its
  tie to `actual ΔE_β` weakens there. Report it per layer; it is a finding
  about the ODE framing, not an error to swallow (§5.3(d)).
- **`sum_check` is the guard, not a model-name branch** — assert it (§3.4).
- **Projector convention** is row-vector `dX @ P`; the on-disk arrays are
  symmetric so `@P` and `P@` agree, but assert symmetry + idempotency on load.
- **Disk**: keep the per-particle-array decision explicit (§4). Default:
  reductions only.
- **Tier ordering**: Tier A answers questions 1 (at layer granularity) and 2
  in minutes. The head-granularity co-location anchor and the attn/FFN split
  need Tier B (~30 min). Ship Tier A, decide Tier B from what it shows.

---

## 7. Phased plan

| Step | What | Cost |
|---|---|---|
| **A0** | `tools/run/dissipation.py` — sibling of `tools/run/relay_null.py`. `METS_REPO`/`METS_DATA` derived; `sys.prefix` + lib-version asserts; loads `activations.npz` + on-disk `ov_projectors_*.npz`; runs `dissipation` + `dissipation_by_subspace` + `gradient_flow_alignment` per (step, prompt, layer); writes `data/analysis/dissipation_series.json` (reductions only, no `per_head` block yet). | ~1 h dev, minutes runtime |
| **A1** | Extend `data/analysis/colocation_panel.png` with the layer-level dissipation series (first_order, `d_repulsive` share, `gfa_frac_descending`) beside the behavioural/relay rows. | ~20 min |
| **A2** *(opt)* | Wire the dissipation series into `paired_colocation_arm` as an alternative B-anchor; print p beside §3.6's 0.1414. Adjudication needs §5. | ~1 h |
| **B0** *(if A shows signal)* | `tools/run/sublayer_capture.py` — 133 forward passes via `extract_sublayer_streams`, write `dX_attn`/`dX_ffn` npz per run; re-run the A0 math with `dissipation_by_channel`; add the `per_head` roll-up on the 116-forming-head axis. | ~30 min runtime |
| **B1** *(v2)* | Per-head OV Schur projectors (per-head attn-delta capture + per-head decomp) — removes the §3.3 approximation. | ~half day |

---

## 8. What this is not

- Not an adjudication of P-I1. INSUFFICIENT at PROJECT.md §3.6 stands; this adds
  an instrument.
- Not a new registered prediction — measurement first (§5).
- Not touching `curve.json`, `formation_series.json`, or any file-hashed record
  (PROJECT.md §7.2).


---

## UPDATE 2026-09-06 — Tier A is run

`tools/run/dissipation.py` built and run over the full 19x7 grid (~22 min, `data/analysis/dissipation_series.json`, `max_subspace_sum_check` 2e-12). Panel: `data/analysis/dissipation_panel.{png,csv}`. Findings are in `PROJECT.md` §3.8.2 — the headline is that the linearisation residual is O(1) outside steps ~2000-4000 (and O(1) at layer 0 always), so the forward-Euler framing holds only in a narrow window; and step 512 is a repulsive-share / gradient-flow-alignment co-location. Tier B (sublayer capture -> attn/FFN split + per-head roll-up) is still the next step and is unchanged from S7 below.
