# Phase 6 — STATUS (live instrument only)

<!-- phase-card -->
## Card

- **Question:** Does the value operator split the residual stream into channels that do different jobs, so that the direction separating the particles' clusters lies in the real repulsive subspace $U_{\rm neg}$ rather than the rotational $U_A$ (`P6-R2`), and projecting onto the symmetric part alone keeps cluster membership (`P6-R4`)?
- **Inputs:** none; the live instrument has never run on real activations. Synthetic calibration (`claims/audits/p6_r2_r4_dry_run.json`) and the projector-label audit (`claims/audits/p6_projector_labels.json`). On disk for `P6-R4`: 19 `p2_eigenspectra_*` dirs of `sym_*` projectors, Phase 1 activations and cluster labels; no Phase 6 run dir in `results/` or `data/` (checked 2026-09-23). The frozen ALBERT study is `6-frozen`
- **Results:**
  - Projector path and matched-dimension random-subspace null rebuilt live, with gates for `P6-R2` and `P6-R4` — `p6_subspace/r2_r4_null.py`
  - `P6-R2` cannot run: no artifact carries an antisymmetric/rotational subspace, by the project's own choice — `p6_subspace/status-6.md` "E-value audit, Phases 5b / 6 (2026-09-19)"
  - `P6-R4` is the one row in the phase whose inputs exist today — `p6_subspace/status-6.md` "E-value audit, Phases 5b / 6 (2026-09-19)"
  - Four dormant rows are classified e-value with no gate, a blind spot in `tools/check_registry.py` — `p6_subspace/status-6.md` "E-value audit, Phases 5b / 6 (2026-09-19)"
- **Superseded / wrong:**
  - The audit said both active rows lacked a registered exchangeable unit and `P6-R4` was blocked on it alone; the unit was registered as `model` on 2026-08-25 — `p6_subspace/status-6.md` "Corrections received"
- **Registry:** twelve `P6-*` rows under `H-OPERATOR`: ten `dormant`; `P6-R2`, `P6-R4` `active`, e-value, gated and calibrated, unit `model`, never run. Both registry `notes` fields still say no unit is registered (stale; the user's to amend)
- **Depends on:** 1@d9236160c9, 2@5e9fc59e62, 2b@69b0e40a60, 6-frozen@d5c92fe31a
- **Feeds:** none
- **Open threads:**
  - Does unit `model` fit Pythia? It was argued from ALBERT's weight tying (one OV, one projector pair); Pythia has a different OV per layer — `p6_subspace/r2_r4_null.py`
  - `P6-R2` needs $U_A$: wire `p7_io.rotational_channel_from_blocks`, or amend the row
  - The frozen study's LDA inversion (0.887 with $U_A$ vs 0.067 with $U_{\rm neg}$) is unresolved — `archive/p6_subspace/status-6.md`
- **After Phase 10:**
  - `P6-R4` on the 410m sweep, v1 prompt keys only, after the unit question (free: projectors and activations on disk)
  - Produce $U_A$ from saved OV weights, then `P6-R2` (free, weights only)
  - Registry check: an `e-value` row naming no gate warns unless `dormant` (free)
- **Reviewed:** 2026-09-24 · body `ddd1bc3049`
<!-- /phase-card -->

**Registered predictions:** twelve `P6-*` rows under `H-OPERATOR`. Ten are
`dormant` — their instrument is `archive/p6_subspace/`, frozen 2026-08-22 and
not maintained, imported or collected. Two are `active`: **`P6-R2`** and
**`P6-R4`**, whose projector path was rebuilt live in this directory on
2026-08-24 against `core/particles.py` and `core/nulls.py`, per
`archive/README.md` rule 2 that nothing is salvaged by copying. Both have a
gate (`p6_subspace.r2_r4_null:p_value_p6_r2` / `_r4`) and a shared calibration
record (`claims/audits/p6_r2_r4_dry_run.json`); neither has been run.

This file covers the live instrument. The frozen phase's own history stays in
`archive/`.

## Corrections received

Corrections to this phase written elsewhere, one line each (`CLAUDE.md` Stop
step 2; `docs/phase_card.md`).

- 2026-09-23 · findings 2 and 3 below (`PROJECT.md` §3.44) are wrong about the unit: `P6-R2` / `P6-R4`'s exchangeable unit was **registered as `model`** by the author on 2026-08-25, before any real p-value (`r2_r4_null.py` `REGISTERED_EXCHANGEABLE_UNIT`, the registry's `null_construction`, `POPPER_PLAN.md` 6l, `1cf825f`). The audit read the entries' stale `notes` field, which still says no unit is registered. So `P6-R4` has inputs, a gate and a unit · `POPPER_PLAN.md` "6l. CLAIM-C's cell-drop dimension, and the floor that was never tight (2026-08-25)"

## E-value audit, Phases 5b / 6 (2026-09-19)

Fourth unit of the audit (`PROJECT.md` §3.44), after Phases 1, 1c and 2/2d.
Twenty-one registered rows, **nineteen dormant and two active**. Findings, in
the order they bite.

**1. `P6-R2` cannot be run, and the reason is that half of what it compares
does not exist on disk.** The statement is "LDA aligns more with the real
repulsive subspace `U_neg` than with the imaginary subspace `U_A`" — a
*contrast between two channels*. `LayerChannels` accordingly requires four
bases per layer: `u_pos`, `u_neg`, `u_a`, `u_s`. Checked across **all 19**
`p2_eigenspectra_*` directories, the projector artifacts carry exactly four
array kinds and no others:

    schur_attract_layer_N   schur_repulse_layer_N
    sym_attract_layer_N     sym_repulse_layer_N

That is `{Schur, symmetric} × {attracting, repelling}` — the **sign** channel,
twice. **There is no antisymmetric, imaginary or rotational subspace anywhere
in the artifacts**, which is consistent rather than surprising: the project
records the rotational channel as *deliberately* absent, with
`rotational_channel: "absent"` written into every Phase 7 manifest
(`p7_motifs/run_7.py:389`) and `real_frac`/`imag_frac` NaN in every row of
every table (`PROJECT.md` §6, where that absence is examined and found to cost
no registered prediction — **`P6-R2` is the row that contradicts this**, since
`U_A` is its second arm).

So `P6-R2` is not blocked on compute or on a null. It is blocked on a channel
the project decided not to measure. Reviving it means either producing the
antisymmetric projectors (`p7_io.rotational_channel_from_blocks` is the named,
unwired seam) or amending the prediction — and the second is a registry
amendment, not a convenience.

**Findings 2 and 3 are wrong (2026-09-23): the unit was registered, `model`,
on 2026-08-25. See "Corrections received".**

**2. Both active rows are missing a registered exchangeable unit, and the gate
refuses rather than choosing one.** `EXCHANGEABLE_UNITS = ("model", "layer")`
and `_check_unit` raises `NullRefused` on anything else, deliberately: "the two
units differ by orders of magnitude in the p they produce, and a default would
silently pick one." The registry entries name no unit. This is the same class
of defect as `P-T1`'s wording (§3.43) — **a registration gap that has to be
closed before the gate is run**, because a unit chosen afterwards cannot be
told apart from a unit chosen for its p-value. `P6-R4` ("S-only projection
preserves cluster membership") shares the gap and, unlike `P6-R2`, does **not**
need `U_A`: it projects onto `u_s`, which `sym_*` supplies. **`P6-R4` is
therefore the one row in this phase whose inputs exist today**, blocked on the
unit alone.

**3. The entries' own "no run artifacts" note is now stale, in `P6-R4`'s
favour.** Both notes say no p-value is emitted "because this repository holds
no run artifacts and because no exchangeable unit is registered". The first
clause was written 2026-08-24; since 2026-09-01 the tree holds **19
checkpoints of per-layer projectors** plus the Phase 1 activations and cluster
labels an LDA direction needs. The second clause stands and is now the only
thing between `P6-R4` and a p-value.

**4. Four dormant rows are classified `e-value` with no gate and no
calibration record** — `P6-I1`, `P6-I2`, `P5b-B1`, `P5b-C1`. That is legal
under `EVALUABILITY.md`'s definition, which reads "a valid null exists **or is
directly constructible**", and all four carry the same `dormant_reason`
(instrument archived 2026-08-22, adjudication needs a deliberate
reintroduction). But "directly constructible" is the whole claim here, and with
the instrument frozen **nobody can check it** — the classification is
unfalsifiable while the row is dormant, and it inflates the count of rows that
look one step from an e-value. `tools/check_registry.py` warns when a row names
a gate but is classified `needs-null` (`P-I5`); it does **not** warn on the
converse, an `e-value` row naming no gate. Four rows sit in that blind spot.
Recommended, not done: either a check that `e-value` requires a gate unless
`dormant`, or a dated note on each of the four saying what construction is
meant.

**What this unit did not do.** No statistic was computed. `P6-R4` could not be
run in any case without registering its unit, and doing that as part of an
audit — with the artifacts sitting right there — is exactly the ordering the
audit exists to protect.
