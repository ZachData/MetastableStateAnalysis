"""
p7_motifs/events.py — the particle-event level of the motif alphabet.

An edge-level motif can be real and inconsequential: a head can attend,
write a force, and move nothing that matters, because the force was small,
orthogonal to what the next layer reads, or cancelled by another head. The
edge level says who pushed whom; this level says whether it mattered, and
P-I4 is adjudicated here.

Events are `extra__` columns on the EXISTING core/particles.py
ParticleTable rather than a new artifact. That is not a storage
convenience: the plan's framing makes the particle table the unit of
analysis, cluster- and population-level results are aggregations over it,
and an event is an annotation on a particle in exactly the way a cluster
label is.

The five events
---------------
    capture       joined a cluster at this layer (was unclustered at the
                  previous layer, clustered at this one)
    hold          stayed unclustered — with `hold_run`, the number of
                  consecutive layers it has been unclustered. This is the
                  primitive Phase 5c specified as `noise_tracking.py` and
                  never built ("this token has been noise for N
                  consecutive layers"); as the plan predicted, it is a
                  groupby on the particle table rather than new tracking
                  machinery.
    escape        left a cluster (clustered previously, unclustered now)
    relay_target  was the tag-carrying particle of a `relay` motif — the
                  target of a stage-1 prev_token edge that later sourced a
                  stage-2 match edge
    moved_fraction
                  share of this particle's layer-to-layer displacement
                  attributable to a given motif's edges

Layer 0 has no previous layer. Its events are NaN/False by construction
and are reported that way rather than dropped, for the same reason
UPDATE_PLAN.md 5.9 gives about layer 0's violation indicator: dropping it
would misalign every series built alongside it.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from core.particles import ParticleTable

EVENT_COLUMNS = ("capture", "hold", "hold_run", "escape", "relay_target",
                 "moved_fraction")

UNCLUSTERED_LABEL = -1   # HDBSCAN noise, same convention as Phase 1


def _ordered_layers(table: ParticleTable) -> np.ndarray:
    return np.unique(table.columns["layer"])


def transition_events(
    table: ParticleTable,
    prompt_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    capture / hold / hold_run / escape for every row of `table`, computed
    per particle across layers.

    A particle is identified by (model, checkpoint_step, prompt_key,
    token_position) — everything in the key except `layer`, which is the
    axis the transition is measured along. Rows are matched by that
    identity rather than by position in the array, because a
    ParticleTable is a concat of per-layer tables in no guaranteed order.

    A particle absent from the previous layer (layer 0, or a token that
    simply has no row there) gets capture=False, escape=False,
    hold_run=1 if it is unclustered now. Its transition is unknown, not
    absent, and False is the honest reading of "no transition observed"
    — but `hold_run` deliberately starts the count rather than reporting
    0, since the particle IS unclustered at this layer whatever happened
    before.
    """
    cols = table.columns
    n = len(table)
    if n == 0:
        return {c: np.array([]) for c in ("capture", "hold", "hold_run", "escape")}

    if prompt_key is not None:
        table = table.filter(prompt_key=prompt_key)
        cols = table.columns
        n = len(table)

    labels = cols["cluster_label"]
    layers = cols["layer"]
    unclustered = labels < 0

    # identity -> {layer: row index}
    ident = list(zip(cols["model"], cols["checkpoint_step"],
                     cols["prompt_key"], cols["token_position"]))
    by_particle: Dict[tuple, Dict[int, int]] = {}
    for i, key in enumerate(ident):
        by_particle.setdefault(key, {})[int(layers[i])] = i

    capture = np.zeros(n, dtype=bool)
    escape = np.zeros(n, dtype=bool)
    hold = unclustered.copy()
    hold_run = np.zeros(n, dtype=np.int64)

    for key, rows in by_particle.items():
        ordered = sorted(rows)
        run = 0
        for pos, layer in enumerate(ordered):
            i = rows[layer]
            if unclustered[i]:
                run += 1
            else:
                run = 0
            hold_run[i] = run

            if pos == 0:
                continue
            j = rows[ordered[pos - 1]]
            was_unclustered = unclustered[j]
            capture[i] = was_unclustered and not unclustered[i]
            escape[i] = (not was_unclustered) and unclustered[i]

    return {"capture": capture, "hold": hold, "hold_run": hold_run, "escape": escape}


def relay_target_flags(
    table: ParticleTable,
    relays: Iterable,
    prompt_key: Optional[str] = None,
) -> np.ndarray:
    """
    Mark every particle that carried a relay's tag.

    `relays` is p7_motifs.motif_alphabet.find_relays' output. A relay's
    `tag_position` is the particle written into at stage 1 and read from
    at stage 2 — the composition point, and the particle P-I4 asks about.

    Flagged at the stage-1 layer (`layer_1`), not at every layer: the tag
    is written there, and marking the particle at every depth would make
    the flag a property of the token rather than of the event.
    """
    cols = table.columns
    flags = np.zeros(len(table), dtype=bool)
    wanted = {(int(r.tag_position), int(r.layer_1)) for r in relays}
    if not wanted:
        return flags
    for i, (pos, layer, pk) in enumerate(zip(
        cols["token_position"], cols["layer"], cols["prompt_key"]
    )):
        if prompt_key is not None and pk != prompt_key:
            continue
        if (int(pos), int(layer)) in wanted:
            flags[i] = True
    return flags


def moved_fraction(
    displacement: np.ndarray,
    motif_force: np.ndarray,
) -> np.ndarray:
    """
    P-I4's readout: what share of each particle's layer-to-layer
    displacement is attributable to a motif's edges.

    displacement : (n_particles, d) — the particle's actual movement
        between this layer and the previous one.
    motif_force  : (n_particles, d) — the summed force from just the
        motif's edges into that particle.

    Defined as the PROJECTION of the motif force onto the displacement
    direction, over the displacement norm:

        <motif_force, displacement> / ||displacement||^2

    and not as ||motif_force|| / ||displacement||, which was the obvious
    first choice and is wrong in a way that matters: a large force
    orthogonal to the actual motion would score high while having moved
    the particle nowhere along its path, and a force opposing the motion
    would be indistinguishable from one driving it. The signed projection
    reads ~1 when the motif accounts for the movement, ~0 when it is
    orthogonal to it, and NEGATIVE when the motif pushed against the
    direction the particle actually went — which is a real and reportable
    outcome, not an error to clip away.

    Particles that did not move (zero displacement) give NaN: "what
    fraction of no movement" has no answer, and 0.0 would read as "the
    motif explained none of it".
    """
    displacement = np.asarray(displacement, dtype=np.float64)
    motif_force = np.asarray(motif_force, dtype=np.float64)
    if displacement.shape != motif_force.shape:
        raise ValueError(
            f"displacement {displacement.shape} and motif_force "
            f"{motif_force.shape} must have the same shape"
        )
    if displacement.ndim == 1:
        displacement = displacement[None, :]
        motif_force = motif_force[None, :]

    denom = np.sum(displacement * displacement, axis=1)
    num = np.sum(motif_force * displacement, axis=1)
    out = np.full(len(denom), np.nan, dtype=np.float64)
    nz = denom > 0
    out[nz] = num[nz] / denom[nz]
    return out


def annotate(
    table: ParticleTable,
    relays: Optional[Iterable] = None,
    displacement: Optional[np.ndarray] = None,
    motif_force: Optional[np.ndarray] = None,
    prompt_key: Optional[str] = None,
) -> ParticleTable:
    """
    Return a copy of `table` with the event columns attached as `extra__`
    entries.

    Every input beyond the table is optional and independently so: without
    `relays`, `relay_target` is absent rather than all-False; without both
    `displacement` and `motif_force`, `moved_fraction` is absent rather
    than NaN-filled. An absent column says "not computed"; a present
    all-False column says "computed, none found". Collapsing those two is
    the degradation standing rule 4 forbids, and it is the difference
    between "P-I4 was not run" and "P-I4 failed".
    """
    events = transition_events(table, prompt_key=prompt_key)
    extra = dict(table.extra)
    extra.update({
        "capture": events["capture"],
        "hold": events["hold"],
        "hold_run": events["hold_run"],
        "escape": events["escape"],
    })

    if relays is not None:
        extra["relay_target"] = relay_target_flags(table, relays, prompt_key=prompt_key)

    if displacement is not None and motif_force is not None:
        mf = moved_fraction(displacement, motif_force)
        if len(mf) != len(table):
            raise ValueError(
                f"moved_fraction has length {len(mf)}, expected {len(table)} "
                "(one row per particle in the table)"
            )
        extra["moved_fraction"] = mf
    elif (displacement is None) != (motif_force is None):
        raise ValueError(
            "displacement and motif_force must be supplied together; got "
            f"displacement={'set' if displacement is not None else 'None'}, "
            f"motif_force={'set' if motif_force is not None else 'None'}."
        )

    return ParticleTable(columns=dict(table.columns), extra=extra)
