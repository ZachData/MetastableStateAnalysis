"""
core/particles.py — Per-particle records, the canonical artifact shape
(transition plan v2, core infrastructure item 4).

"The object of study going forward is every particle and how it evolves.
Clustering is one annotation on a particle, not the unit of analysis."
(plan, "Framing: particles first"). Concretely: the extraction layer's
canonical output is a long table of particle records keyed by
(model, checkpoint_step, prompt_key, layer, token_position), carrying a
cluster label, a population tag, and (once core analysis primitives item
3 lands) a V-projection and dual-reading-primitive output. Cluster- and
population-level results become aggregations (groupby / filter) over this
table rather than separate code paths, and the population selector (item
8, threaded through displacement_projection / v_alignment / etc. in a
later pass) reduces to ParticleTable.filter(population=...).

No pandas dependency — nothing else in this project uses pandas (checked:
every existing loader/analysis module reads json/npz into plain dicts and
numpy arrays), so this module follows suit: a ParticleTable is a plain
columnar dict-of-arrays, saved as one npz per table. This keeps it
importable and testable wherever numpy is available, consistent with
core.metrics's torch-optional design.

Schema
------
Key columns (identify a row uniquely together with token_position):
    model           : str   — MODEL_CONFIGS key, e.g. "pythia-1.4b-step1000"
    checkpoint_step : int   — -1 sentinel for non-checkpointed models
                              (gpt2/albert/bert), a real HF step otherwise.
    prompt_key      : str   — key into core.config.PROMPTS
    layer           : int   — 0-indexed layer position
    token_position  : int   — 0-indexed position within the prompt

Value columns:
    cluster_label       : int   — HDBSCAN label; -1 is noise (same
                                   convention as Phase 1's hdbscan_labels.json)
    population          : str   — annotation on cluster_label, not a
                                   separate measurement; defaults to
                                   "clustered" / "unclustered" via
                                   default_population_tag, override for
                                   any future population scheme.
    token_str            : str   — decoded token text (optional)
    v_attractive_proj    : float — optional; NaN until core analysis
                                    primitives (item 3) computes it
    v_repulsive_proj     : float — optional; NaN until item 3

Anything else (dual-reading-primitive output, once it exists) goes into
`extra`, a dict of {name: (n_rows,) array}, saved as additional npz keys
prefixed "extra__".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

CHECKPOINT_STEP_SENTINEL = -1  # "no checkpoint" (non-Pythia models)

KEY_COLUMNS = ("model", "checkpoint_step", "prompt_key", "layer", "token_position")
REQUIRED_VALUE_COLUMNS = ("cluster_label", "population")
OPTIONAL_VALUE_COLUMNS = ("token_str", "v_attractive_proj", "v_repulsive_proj")
ALL_COLUMNS = KEY_COLUMNS + REQUIRED_VALUE_COLUMNS + OPTIONAL_VALUE_COLUMNS


def default_population_tag(cluster_labels: np.ndarray) -> np.ndarray:
    """
    "clustered" / "unclustered" from an HDBSCAN label array (-1 = noise),
    matching the split Phase 5c's preliminary work already uses. This is
    the default population tag; a future dual-reading primitive or a
    hand-picked selection (Phase 5c's five-criterion scoring) may assign
    a different tag, which is exactly why `population` is stored as its
    own column rather than derived on the fly from cluster_label at every
    call site — one place computes the default, every consumer can
    override without touching this module.
    """
    cluster_labels = np.asarray(cluster_labels)
    return np.where(cluster_labels < 0, "unclustered", "clustered")


@dataclass
class ParticleTable:
    """
    Columnar particle-record table. Every array has the same length
    (n_rows); row i across all columns describes one particle
    (token, at one layer, at one checkpoint, in one prompt, in one model).

    Construct via `from_layer` (one layer's worth of rows) or `concat`
    (stack several ParticleTables, e.g. across layers/checkpoints/prompts).
    Don't construct directly unless you're already holding validated,
    equal-length arrays for every column.
    """
    columns: Dict[str, np.ndarray]
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        missing = set(KEY_COLUMNS + REQUIRED_VALUE_COLUMNS) - set(self.columns)
        if missing:
            raise ValueError(f"ParticleTable missing required columns: {sorted(missing)}")
        lengths = {k: len(v) for k, v in self.columns.items()}
        lengths.update({f"extra__{k}": len(v) for k, v in self.extra.items()})
        if len(set(lengths.values())) > 1:
            raise ValueError(f"ParticleTable columns have mismatched lengths: {lengths}")

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def from_layer(
        cls,
        model: str,
        prompt_key: str,
        layer: int,
        cluster_labels: Sequence[int],
        checkpoint_step: Optional[int] = None,
        population: Optional[Sequence[str]] = None,
        token_str: Optional[Sequence[str]] = None,
        v_attractive_proj: Optional[Sequence[float]] = None,
        v_repulsive_proj: Optional[Sequence[float]] = None,
        extra: Optional[Dict[str, Sequence]] = None,
    ) -> "ParticleTable":
        """
        Build the rows for one (model, checkpoint, prompt, layer) — one
        row per token. `cluster_labels` fixes n_tokens; every other
        per-token argument must match its length if given.
        """
        cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
        n = len(cluster_labels)

        step_val = CHECKPOINT_STEP_SENTINEL if checkpoint_step is None else int(checkpoint_step)

        # String columns deliberately built via np.array(list_of_str) rather
        # than np.full(..., dtype=object): numpy infers a fixed-width
        # unicode dtype ('<U...') from a list of plain str, which npz
        # round-trips with allow_pickle=False. dtype=object would instead
        # require allow_pickle=True on every future load — checked directly
        # (np.load raises "Object arrays cannot be loaded when
        # allow_pickle=False" otherwise), not assumed.
        cols: Dict[str, np.ndarray] = {
            "model":           np.array([model] * n),
            "checkpoint_step": np.full(n, step_val, dtype=np.int64),
            "prompt_key":      np.array([prompt_key] * n),
            "layer":           np.full(n, int(layer), dtype=np.int64),
            "token_position":  np.arange(n, dtype=np.int64),
            "cluster_label":   cluster_labels,
            "population": (
                np.array([str(p) for p in population])
                if population is not None
                else default_population_tag(cluster_labels)
            ),
        }

        def _optional_float(values, name):
            if values is None:
                return np.full(n, np.nan, dtype=np.float64)
            arr = np.asarray(values, dtype=np.float64)
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")
            return arr

        cols["v_attractive_proj"] = _optional_float(v_attractive_proj, "v_attractive_proj")
        cols["v_repulsive_proj"] = _optional_float(v_repulsive_proj, "v_repulsive_proj")

        if token_str is not None:
            if len(token_str) != n:
                raise ValueError(f"token_str has length {len(token_str)}, expected {n}")
            cols["token_str"] = np.array([str(t) for t in token_str])
        else:
            cols["token_str"] = np.array([""] * n)

        extra_cols: Dict[str, np.ndarray] = {}
        for k, v in (extra or {}).items():
            v = np.asarray(v)
            if len(v) != n:
                raise ValueError(f"extra column {k!r} has length {len(v)}, expected {n}")
            if v.dtype == object:
                raise ValueError(
                    f"extra column {k!r} has dtype=object, which cannot be "
                    "saved without allow_pickle. Use a numeric array or a "
                    "plain list of str (numpy will infer a fixed-width "
                    "unicode dtype)."
                )
            extra_cols[k] = v

        return cls(columns=cols, extra=extra_cols)

    @classmethod
    def concat(cls, tables: Iterable["ParticleTable"]) -> "ParticleTable":
        """Stack several ParticleTables' rows into one. Empty input -> empty table."""
        tables = list(tables)
        if not tables:
            empty = {c: np.array([], dtype=object) for c in KEY_COLUMNS + REQUIRED_VALUE_COLUMNS}
            return cls(columns=empty, extra={})

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
        merged_extra = {}
        for name in extra_names:
            pieces = [
                t.extra[name] if name in t.extra else np.full(len(t), np.nan)
                for t in tables
            ]
            merged_extra[name] = np.concatenate(pieces)

        return cls(columns=merged_cols, extra=merged_extra)

    # -----------------------------------------------------------------
    # The population selector, as a filter
    # -----------------------------------------------------------------

    def filter(self, **equals) -> "ParticleTable":
        """
        Return the subset of rows where every given column equals the
        given value, e.g. `table.filter(population="unclustered")` or
        `table.filter(model="pythia-1.4b-step1000", layer=12)`. This is
        the population selector (plan item 8) reduced to its primitive:
        once every phase's data lives in a ParticleTable, threading the
        selector through a given consumer (v_alignment.py,
        probe_subspace.py, ...) is calling this instead of hand-rolling
        a `labels >= 0` mask at each call site.
        """
        mask = np.ones(len(self), dtype=bool)
        for key, value in equals.items():
            if key in self.columns:
                mask &= (self.columns[key] == value)
            elif key in self.extra:
                mask &= (self.extra[key] == value)
            else:
                raise KeyError(f"Unknown column {key!r} for filter()")
        new_cols = {k: v[mask] for k, v in self.columns.items()}
        new_extra = {k: v[mask] for k, v in self.extra.items()}
        return ParticleTable(columns=new_cols, extra=new_extra)

    # -----------------------------------------------------------------
    # Row / dict views
    # -----------------------------------------------------------------

    def to_records(self) -> List[dict]:
        """List-of-dicts view, one dict per row. Convenience for small
        tables / debugging; not the storage format (see save/load)."""
        n = len(self)
        keys = list(self.columns) + list(self.extra)
        return [
            {k: (self.columns[k][i] if k in self.columns else self.extra[k][i])
             for k in keys}
            for i in range(n)
        ]

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save to one npz file. String columns must already be fixed-width
        unicode arrays ('<U...', what plain `np.array(list_of_str)`
        infers) rather than dtype=object — object arrays serialize via
        pickle and can't be read back with allow_pickle=False, which
        `load` below deliberately uses. `from_layer` already builds every
        string column this way; this is enforced again here so a
        hand-built ParticleTable fails at save time, not at the next
        load.
        """
        for name, arr in {**self.columns, **self.extra}.items():
            if np.asarray(arr).dtype == object:
                raise ValueError(
                    f"column {name!r} has dtype=object, which save() "
                    "refuses to write (see load()'s allow_pickle=False). "
                    "Convert to a numeric array or a plain list of str "
                    "before saving."
                )
        path = Path(path)
        payload = {k: np.asarray(v) for k, v in self.columns.items()}
        payload.update({f"extra__{k}": np.asarray(v) for k, v in self.extra.items()})
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParticleTable":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        cols = {}
        extra = {}
        for key in data.files:
            if key.startswith("extra__"):
                extra[key[len("extra__"):]] = data[key]
            else:
                cols[key] = data[key]
        return cls(columns=cols, extra=extra)
