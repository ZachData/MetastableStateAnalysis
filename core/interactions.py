"""
core/interactions.py — Typed interaction edges, the interaction-graph
analogue of core/particles.py's ParticleTable (Phase 7, p7_motifs/design-7.md).

Where a ParticleTable row is a particle (a token, at one layer, at one
checkpoint, in one prompt), an InteractionTable row is a *directed edge
between two particles at the same (model, checkpoint, prompt, layer)*,
carried by one attention head. Motif counts are aggregations over this
table — groupby / filter — not a separate producer, exactly as
cluster-level results are aggregations over ParticleTable.

Why the edge carries force and not just attention
--------------------------------------------------
Post-softmax attention A_ij is the interaction *kernel*: how much particle
i looks at particle j. It is not the interaction. Two heads with identical
attention patterns and opposite-signed OV circuits move particles in
opposite directions, and an analysis keyed on A_ij alone cannot tell them
apart. So the edge's primary quantity is the displacement contribution
A_ij * (V x_j), decomposed through the Phase 2 / 2b projectors into:

    attractive_frac / repulsive_frac   (U_pos / U_neg — the sign channel)
    real_frac       / imag_frac        (U_S   / U_A   — the rotational channel)

The sign channel is what makes this a dynamical object rather than a
routing diagram, and it is the reason Phase 7 is not a re-description of
attention-pattern analysis.

No pandas, no dtype=object, one npz per table — the same storage contract
ParticleTable follows, for the same reasons (see that module's docstring:
np.load(allow_pickle=False) must be able to read anything we write).

Schema
------
Key columns (identify a row uniquely, together):
    model           : str   — MODEL_CONFIGS key
    checkpoint_step : int   — -1 sentinel for non-checkpointed models
    prompt_key      : str   — key into core.config.PROMPTS
    layer           : int
    head            : int   — -1 for a head-agnostic (layer-summed) edge
    target          : int   — token_position of the particle being moved (i)
    source          : int   — token_position of the particle doing the moving (j)

Value columns:
    weight           : float — post-softmax attention A_ij
    force_magnitude  : float — ||A_ij * V x_j||
    attractive_frac  : float — squared-norm fraction of the force in U_pos
    repulsive_frac   : float — squared-norm fraction of the force in U_neg
    real_frac        : float — fraction in U_S; NaN if no 2b projectors given
    imag_frac        : float — fraction in U_A; NaN if no 2b projectors given
    offset           : int   — target - source
    pair_type        : str   — "induction" | "strict" | "same_content" | "neither"

Anything else goes in `extra`, {name: (n_edges,) array}, saved with an
"extra__" prefix — same convention as ParticleTable.

A note on absence
-----------------
Edge tables are large: n_tokens^2 per head per layer per checkpoint. A
producer is expected to apply a top-k-by-force retention cutoff and record
it in the run manifest. **An absent edge is therefore not a zero-force
edge**, and no consumer may treat it as one. `retention` on this class
carries the cutoff so a table that has been thinned says so about itself
rather than relying on the reader to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

CHECKPOINT_STEP_SENTINEL = -1   # matches core/particles.py
HEAD_AGNOSTIC = -1              # "this edge is not attributed to one head"

PAIR_TYPES = ("induction", "strict", "same_content", "neither")

KEY_COLUMNS = ("model", "checkpoint_step", "prompt_key", "layer", "head",
               "target", "source")
REQUIRED_VALUE_COLUMNS = ("weight", "force_magnitude", "attractive_frac",
                          "repulsive_frac", "offset", "pair_type")
OPTIONAL_VALUE_COLUMNS = ("real_frac", "imag_frac")
ALL_COLUMNS = KEY_COLUMNS + REQUIRED_VALUE_COLUMNS + OPTIONAL_VALUE_COLUMNS


def projection_fractions(force: np.ndarray, U: Optional[np.ndarray]) -> np.ndarray:
    """
    Squared-norm fraction of each force vector lying in the subspace
    spanned by U's columns: ||U^T f||^2 / ||f||^2.

    force : (n_edges, d)
    U     : (d, r) orthonormal columns, or None -> all-NaN, which is the
            correct answer when the projector was not supplied. Never a
            silent 0.0: "no U_A was given" and "this force has no
            imaginary component" are different facts and must not collapse
            (standing rule 4, refuse rather than degrade).

    Zero-norm forces give 0.0 rather than NaN — an edge that moves nothing
    has a well-defined answer, namely that none of its (absent) motion is
    in any subspace.
    """
    force = np.asarray(force, dtype=np.float64)
    if force.ndim == 1:
        force = force[None, :]
    n = force.shape[0]
    if U is None:
        return np.full(n, np.nan, dtype=np.float64)

    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        U = U[:, None]
    if U.shape[0] != force.shape[1]:
        raise ValueError(
            f"projector has d={U.shape[0]} but force vectors have "
            f"d={force.shape[1]}; these must match (frame mismatch)."
        )

    total = np.sum(force * force, axis=1)
    proj = np.sum((force @ U) ** 2, axis=1)
    out = np.zeros(n, dtype=np.float64)
    nz = total > 0
    out[nz] = proj[nz] / total[nz]
    return out


def classify_pair_types(
    targets: Sequence[int],
    sources: Sequence[int],
    induction_pairs: Optional[Iterable] = None,
    strict_pairs: Optional[Iterable] = None,
    same_content_pairs: Optional[Iterable] = None,
) -> np.ndarray:
    """
    Label each (target, source) edge with its pair type, from the pair sets
    core.battery_structure already computes.

    Precedence is induction > strict > same_content > neither. The first
    two overlap by construction (they are different formulations of the
    same idea — see core/battery_structure.py's header on the repo's
    condition vs the Anthropic one) and this project reports both, so an
    edge in both sets is labelled "induction" and the divergence between
    the two sets stays visible where it belongs: in battery_structure's own
    report, not smeared into per-edge labels here.

    Pairs are accepted as any iterable of 2-sequences, in (query, key) =
    (target, source) order — the orientation battery_structure emits.
    """
    targets = np.asarray(targets, dtype=np.int64)
    sources = np.asarray(sources, dtype=np.int64)
    if targets.shape != sources.shape:
        raise ValueError("targets and sources must have the same length")

    def _as_set(pairs):
        if pairs is None:
            return set()
        return {(int(a), int(b)) for a, b in pairs}

    ind = _as_set(induction_pairs)
    strict = _as_set(strict_pairs)
    same = _as_set(same_content_pairs)

    out = np.empty(len(targets), dtype="<U12")
    for i, (t, s) in enumerate(zip(targets, sources)):
        key = (int(t), int(s))
        if key in ind:
            out[i] = "induction"
        elif key in strict:
            out[i] = "strict"
        elif key in same:
            out[i] = "same_content"
        else:
            out[i] = "neither"
    return out


@dataclass
class InteractionTable:
    """
    Columnar typed-edge table. Every array has length n_edges; row i across
    all columns describes one directed interaction.

    `retention` records how the table was thinned, if it was. None means
    "every causal edge is present." A dict means a cutoff was applied and
    absent edges are unknown, not zero — see the module docstring.

    Construct via `from_head` (one head's worth of edges) or `concat`.
    """
    columns: Dict[str, np.ndarray]
    extra: Dict[str, np.ndarray] = field(default_factory=dict)
    retention: Optional[dict] = None

    def __post_init__(self):
        missing = set(KEY_COLUMNS + REQUIRED_VALUE_COLUMNS) - set(self.columns)
        if missing:
            raise ValueError(f"InteractionTable missing required columns: {sorted(missing)}")
        lengths = {k: len(v) for k, v in self.columns.items()}
        lengths.update({f"extra__{k}": len(v) for k, v in self.extra.items()})
        if len(set(lengths.values())) > 1:
            raise ValueError(f"InteractionTable columns have mismatched lengths: {lengths}")

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_head(
        cls,
        model: str,
        prompt_key: str,
        layer: int,
        head: int,
        targets: Sequence[int],
        sources: Sequence[int],
        weight: Sequence[float],
        force: np.ndarray,
        U_pos: Optional[np.ndarray] = None,
        U_neg: Optional[np.ndarray] = None,
        U_S: Optional[np.ndarray] = None,
        U_A: Optional[np.ndarray] = None,
        pair_type: Optional[Sequence[str]] = None,
        checkpoint_step: Optional[int] = None,
        retention: Optional[dict] = None,
        extra: Optional[Dict[str, Sequence]] = None,
    ) -> "InteractionTable":
        """
        Build the rows for one (model, checkpoint, prompt, layer, head).

        force : (n_edges, d) — the per-edge displacement contribution
                A_ij * (V x_j), in the SAME frame as the projectors. This
                function does not normalize anything and does not check the
                frame; callers hold a FrameSpec for that (see core/frames.py).
                Passing L2-normalized activations to a rotary head is the
                error that core/qk_offset_null.py's frame discipline exists
                to make visible, and it is not detectable from shapes alone.
        """
        targets = np.asarray(targets, dtype=np.int64)
        sources = np.asarray(sources, dtype=np.int64)
        n = len(targets)
        if len(sources) != n:
            raise ValueError(f"sources has length {len(sources)}, expected {n}")

        force = np.asarray(force, dtype=np.float64)
        if force.ndim != 2 or force.shape[0] != n:
            raise ValueError(
                f"force must be (n_edges, d) with n_edges={n}; got {force.shape}"
            )

        weight = np.asarray(weight, dtype=np.float64)
        if len(weight) != n:
            raise ValueError(f"weight has length {len(weight)}, expected {n}")

        step_val = CHECKPOINT_STEP_SENTINEL if checkpoint_step is None else int(checkpoint_step)

        if pair_type is None:
            pt = np.array(["neither"] * n, dtype="<U12")
        else:
            pt = np.array([str(p) for p in pair_type], dtype="<U12")
            if len(pt) != n:
                raise ValueError(f"pair_type has length {len(pt)}, expected {n}")
            bad = sorted(set(pt.tolist()) - set(PAIR_TYPES))
            if bad:
                raise ValueError(
                    f"unknown pair_type(s) {bad}; expected one of {list(PAIR_TYPES)}"
                )

        cols: Dict[str, np.ndarray] = {
            "model":           np.array([model] * n),
            "checkpoint_step": np.full(n, step_val, dtype=np.int64),
            "prompt_key":      np.array([prompt_key] * n),
            "layer":           np.full(n, int(layer), dtype=np.int64),
            "head":            np.full(n, int(head), dtype=np.int64),
            "target":          targets,
            "source":          sources,
            "weight":          weight,
            "force_magnitude": np.linalg.norm(force, axis=1),
            "attractive_frac": projection_fractions(force, U_pos),
            "repulsive_frac":  projection_fractions(force, U_neg),
            "real_frac":       projection_fractions(force, U_S),
            "imag_frac":       projection_fractions(force, U_A),
            "offset":          targets - sources,
            "pair_type":       pt,
        }

        extra_cols: Dict[str, np.ndarray] = {}
        for k, v in (extra or {}).items():
            v = np.asarray(v)
            if len(v) != n:
                raise ValueError(f"extra column {k!r} has length {len(v)}, expected {n}")
            if v.dtype == object:
                raise ValueError(
                    f"extra column {k!r} has dtype=object, which cannot be saved "
                    "without allow_pickle. Use a numeric array or a plain list of str."
                )
            extra_cols[k] = v

        return cls(columns=cols, extra=extra_cols, retention=retention)

    @classmethod
    def concat(cls, tables: Iterable["InteractionTable"]) -> "InteractionTable":
        """
        Stack several InteractionTables' rows. Empty input -> empty table.

        Refuses to merge tables with different `retention` settings: two
        tables thinned by different cutoffs cannot be counted together
        without the counts meaning different things per row, and silently
        picking one of the two would be exactly the kind of degradation
        standing rule 4 forbids.
        """
        tables = list(tables)
        if not tables:
            empty = {c: np.array([], dtype=np.int64 if c in
                                 ("checkpoint_step", "layer", "head", "target",
                                  "source", "offset")
                                 else (np.float64 if c in
                                       ("weight", "force_magnitude", "attractive_frac",
                                        "repulsive_frac")
                                       else "<U1"))
                     for c in KEY_COLUMNS + REQUIRED_VALUE_COLUMNS}
            return cls(columns=empty, extra={})

        retentions = [t.retention for t in tables]
        first = retentions[0]
        if any(r != first for r in retentions[1:]):
            raise ValueError(
                "concat: refusing to merge InteractionTables with different "
                f"retention settings: {retentions}. An absent edge means "
                "different things under different cutoffs."
            )

        col_names = set(tables[0].columns)
        for t in tables[1:]:
            if set(t.columns) != col_names:
                raise ValueError(
                    "concat: all tables must share the same column set; "
                    f"got {sorted(col_names)} vs {sorted(t.columns)}"
                )

        merged_cols = {
            name: np.concatenate([t.columns[name] for t in tables])
            for name in col_names
        }
        extra_names = set()
        for t in tables:
            extra_names |= set(t.extra)
        merged_extra = {
            name: np.concatenate([
                t.extra[name] if name in t.extra else np.full(len(t), np.nan)
                for t in tables
            ])
            for name in extra_names
        }
        return cls(columns=merged_cols, extra=merged_extra, retention=first)

    # -----------------------------------------------------------------
    # Selection
    # -----------------------------------------------------------------

    def filter(self, **equals) -> "InteractionTable":
        """
        Subset of rows where every given column equals the given value,
        e.g. `table.filter(pair_type="induction", layer=4)`. The motif
        predicates in p7_motifs/motif_alphabet.py are built out of this
        plus threshold masks, rather than each hand-rolling its own
        indexing.
        """
        mask = np.ones(len(self), dtype=bool)
        for key, value in equals.items():
            if key in self.columns:
                mask &= (self.columns[key] == value)
            elif key in self.extra:
                mask &= (self.extra[key] == value)
            else:
                raise KeyError(f"Unknown column {key!r} for filter()")
        return self.mask(mask)

    def mask(self, mask: np.ndarray) -> "InteractionTable":
        """Subset by an explicit boolean mask, preserving retention."""
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self):
            raise ValueError(f"mask has length {len(mask)}, expected {len(self)}")
        return InteractionTable(
            columns={k: v[mask] for k, v in self.columns.items()},
            extra={k: v[mask] for k, v in self.extra.items()},
            retention=self.retention,
        )

    def to_records(self) -> List[dict]:
        """List-of-dicts view. Debugging convenience, not the storage format."""
        keys = list(self.columns) + list(self.extra)
        return [
            {k: (self.columns[k][i] if k in self.columns else self.extra[k][i])
             for k in keys}
            for i in range(len(self))
        ]

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save to one npz. String columns must be fixed-width unicode, not
        dtype=object — enforced here so a hand-built table fails at save
        time rather than at the next load (see load()'s allow_pickle=False).

        `retention` rides along as two scalar arrays rather than a pickled
        dict, so the thinning is recoverable from the file itself.
        """
        for name, arr in {**self.columns, **self.extra}.items():
            if np.asarray(arr).dtype == object:
                raise ValueError(
                    f"column {name!r} has dtype=object, which save() refuses "
                    "to write. Convert to a numeric array or a list of str."
                )
        payload = {k: np.asarray(v) for k, v in self.columns.items()}
        payload.update({f"extra__{k}": np.asarray(v) for k, v in self.extra.items()})
        if self.retention is not None:
            for rk, rv in self.retention.items():
                payload[f"retention__{rk}"] = np.asarray(rv)
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InteractionTable":
        data = np.load(Path(path), allow_pickle=False)
        cols, extra, retention = {}, {}, {}
        for key in data.files:
            if key.startswith("extra__"):
                extra[key[len("extra__"):]] = data[key]
            elif key.startswith("retention__"):
                v = data[key]
                retention[key[len("retention__"):]] = v.item() if v.ndim == 0 else v
            else:
                cols[key] = data[key]
        return cls(columns=cols, extra=extra, retention=retention or None)
