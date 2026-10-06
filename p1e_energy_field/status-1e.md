<!-- p1e_energy_field/status-1e.md -->
# Phase 1e — STATUS

<!-- phase-card -->
## Card

- **Question:** Read the residual stream as the theory's field rather than as clusters: where are the wells and crests of the energy at the measured β, and does each token's update ascend the field (attraction) or descend it (repulsion, the packing end)?
- **Inputs:** none run yet. Proposed: `pythia-410m`, the 18 Stage 0 steps, the 7 v1 passages, unit LN1 rows, β = 3.5 [1.6, 5.6] — `p1e_energy_field/design-1e.md` "Inputs, fixed here"
- **Results:**
  - Opened 2026-10-06 with a literature scan and a proposed design, not frozen; no measurement yet. The closed-form steps the design uses are checked (M6 corrected after review: in a cloud-centred frame the field also has a floor off the tokens' span) — `p1e_energy_field/design-1e.md` "The math", `p1e_energy_field/lit-1e.md`
- **Superseded / wrong:** none
- **Registry:** none, because the phase is proposed and unregistered; it is fenced off P-S1, P-γ1/P-γ2 and P-M1 — `p1e_energy_field/design-1e.md` "Fences"
- **Depends on:** 1d@4168cdd237, 10@c159a02c9b
- **Feeds:** none
- **Open threads:**
  - Whether the design freezes as proposed, and which units run first: the user's — `STATE.md` Blocked 28
  - The correlated against anti-correlated question at the token level needs a corpus for co-occurrence — `p1e_energy_field/design-1e.md` "Units"
- **After Phase 10:**
  - U1, U3, U4 from stored activations *(free)*; U2 *(free for the block arm; forward pass per step and passage, GPU, for the attention and per-head arms)*
  - U5, after a corpus download and a co-occurrence count *(free)*
- **Reviewed:** 2026-10-06 · body `316886011f`
<!-- /phase-card -->

Nothing has run under this phase. The 2026-10-06 probe that prompted it ran under Phase 10's
handoff (`p10_cluster_function/handoff-10.md` Parked) on frames 1e does not use.

## Corrections received
