"""
p1b_hemisphere/visualization/loaders.py

Every disk read in this package, in one module. No figure module opens a
file; they take a `Run` (or a list of them) and draw.

Two things this module is deliberately strict about.

**Optional means optional, and says which.** Phase 1b runs standalone: Block
3 is skippable (`--no-cone`), its nulls are opt-in (`--n-null`), and Block 2's
nesting and boundary tables only exist when the Phase 1 cross-reference
resolved and produced `hdbscan_labels`. Four more tables (`cone_per_layer`,
`hdbscan_nesting`, `border_vs_noise`, `persistence_length`) plus the axes npz
are newer than the first Phase 1b runs and are simply absent from those
directories. So a `Run` reports `missing` — a list of human-readable reasons —
and every figure that needs one of them checks and skips with that reason
printed. Never a KeyError, and never a figure that silently drops half its
layers.

**JSON object keys are strings, layer indices are not.** `hdbscan_nesting`
and `border_vs_noise` are keyed by layer int in memory and by `"7"` after a
round trip through JSON. `_int_keyed` normalizes once, here, rather than at
seven call sites — this is exactly the artifact-contract bug class
`core/artifacts.py` exists to kill, and the version of it that bit Phase 1b
before (the ALBERT path, status-1b R2) was the same shape: a key convention
assumed rather than resolved.

Nothing here recomputes an analysis. `per_layer` arrays come out as float
arrays with JSON `null` mapped to NaN — a dropped row would silently shift
every depth profile in the package by one layer, which is the failure mode
worth being paranoid about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ARTIFACT_PREFIX = "phase1b"

__all__ = [
    "Run", "discover_runs", "load_run", "load_cross_run", "layer_field",
    "layer_strings", "token_field", "token_strings", "particles",
    "checkpoint_families", "describe_runs",
]


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One (model, prompt) Phase 1b result, plus whatever else is on disk."""
    model: str
    prompt: str
    stem: str
    path: Path
    data: dict
    #: Human-readable reasons a figure might have to skip. Populated at load.
    missing: List[str] = field(default_factory=list)
    _particles: Optional[dict] = None
    _axes: Optional[dict] = None

    # -- identity -----------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return int(self.data.get("n_layers") or 0)

    @property
    def n_tokens(self) -> int:
        return int(self.data.get("n_tokens") or 0)

    @property
    def checkpoint_step(self) -> Optional[int]:
        s = self.data.get("checkpoint_step")
        return int(s) if s is not None else None

    @property
    def label(self) -> str:
        return f"{self.model} / {self.prompt}"

    @property
    def summary(self) -> dict:
        return self.data.get("summary") or {}

    @property
    def per_layer(self) -> List[dict]:
        return self.data.get("per_layer") or []

    @property
    def per_token(self) -> List[dict]:
        return self.data.get("per_token") or []

    @property
    def events(self) -> List[dict]:
        return self.data.get("events") or []

    # -- optional blocks ----------------------------------------------------

    @property
    def cone(self) -> Optional[dict]:
        return self.data.get("cone")

    @property
    def cone_per_layer(self) -> Optional[List[dict]]:
        return self.data.get("cone_per_layer")

    @property
    def axis_identity(self) -> Optional[dict]:
        return self.data.get("axis_identity")

    @property
    def nesting(self) -> Optional[dict]:
        n = self.data.get("hdbscan_nesting")
        if not n:
            return None
        return {"per_layer": _int_keyed(n.get("per_layer") or {}),
                "overall": n.get("overall") or {}}

    @property
    def border_vs_noise(self) -> Optional[dict]:
        b = self.data.get("border_vs_noise")
        if not b:
            return None
        return {"per_layer": _int_keyed(b.get("per_layer") or {}),
                "overall": b.get("overall") or {}}

    @property
    def persistence_length(self) -> Optional[np.ndarray]:
        p = self.data.get("persistence_length")
        return _floats(p) if p else None

    def axes(self) -> Optional[dict]:
        """
        Activation-space Fiedler axes, (n_layers, d), lazily.

        Written by `run_1b` as `phase1b_{stem}_axes.npz`. Absent from runs
        made before that emission landed, which is not an error — it means
        the cross-checkpoint axis figures skip.
        """
        if self._axes is None:
            p = self.path.parent / f"{ARTIFACT_PREFIX}_{self.stem}_axes.npz"
            if not p.exists():
                self._axes = {}
            else:
                try:
                    d = np.load(p)
                    self._axes = {"axes": np.asarray(d["axes"]),
                                  "valid": np.asarray(d["valid"])
                                  if "valid" in d.files else None}
                except Exception:
                    self._axes = {}
        return self._axes or None

    def particles(self) -> Optional[dict]:
        """
        The ParticleTable as plain arrays, lazily.

        Loaded through `core.particles.ParticleTable` so this package inherits
        the schema check rather than reimplementing the column grammar; the
        `extra__` prefix stripping lives there too. Returns a flat
        {column: array} dict — one row per (layer, token).
        """
        if self._particles is None:
            self._particles = _load_particles(
                self.path.parent / f"{ARTIFACT_PREFIX}_{self.stem}_particles.npz")
        return self._particles or None

    # -- convenience --------------------------------------------------------

    def field(self, name: str) -> np.ndarray:
        """`per_layer` column as a float array, JSON null -> NaN."""
        return layer_field(self.per_layer, name)

    def strings(self, name: str) -> List[str]:
        """`per_layer` column as strings, missing -> 'invalid'."""
        return layer_strings(self.per_layer, name)

    def has(self, what: str) -> bool:
        """
        Is an optional input present? `what` is one of the reasons listed in
        `missing`, so a figure's guard and its skip message cannot disagree.
        """
        return what not in self.missing


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(p1b_dir: Path,
                  models: Optional[Sequence[str]] = None,
                  prompts: Optional[Sequence[str]] = None) -> List[Run]:
    """
    Every `phase1b_{stem}.json` under `p1b_dir`, as `Run`s.

    Sorted by (model, prompt) so figure order is stable across invocations —
    an output directory that reorders itself between runs is one nobody can
    diff. The cross-run digest is excluded here; `load_cross_run` owns it.
    """
    p1b_dir = Path(p1b_dir)
    if not p1b_dir.exists():
        return []

    runs: List[Run] = []
    for path in sorted(p1b_dir.glob(f"{ARTIFACT_PREFIX}_*.json")):
        if path.name == f"{ARTIFACT_PREFIX}_cross_run.json":
            continue
        run = load_run(path)
        if run is None:
            continue
        if models and run.model not in models:
            continue
        if prompts and run.prompt not in prompts:
            continue
        runs.append(run)

    runs.sort(key=lambda r: (r.model, r.prompt))
    return runs


def load_run(path: Path) -> Optional[Run]:
    """One per-run JSON, with its `missing` list populated."""
    path = Path(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  ⚠  unreadable, skipping: {path.name} ({exc})")
        return None

    if "per_layer" not in data or "summary" not in data:
        print(f"  ⚠  not a Phase 1b per-run JSON, skipping: {path.name}")
        return None

    stem = path.stem[len(ARTIFACT_PREFIX) + 1:]
    run = Run(model=str(data.get("model", "?")),
              prompt=str(data.get("prompt", "?")),
              stem=stem, path=path, data=data)

    # Each reason is phrased as what is absent and what caused it, because
    # this string is what a skipped figure prints. "cone" alone would send a
    # reader to the wrong place; "--no-cone" sends them to the flag.
    if data.get("cone") is None:
        run.missing.append("cone (Block 3 not run — --no-cone?)")
    if data.get("cone_per_layer") is None:
        run.missing.append("cone_per_layer (run predates the per-layer cone emission)")
    elif not any(e.get("z_vs_uniform") is not None
                 for e in data.get("cone_per_layer") or []):
        run.missing.append("cone nulls (Block 3 run without --n-null)")
    if data.get("axis_identity") is None:
        run.missing.append("axis_identity (Block A failed or not run)")
    if data.get("hdbscan_nesting") is None:
        run.missing.append("hdbscan_nesting (no Phase 1 cross-reference, or predates emission)")
    if data.get("border_vs_noise") is None:
        run.missing.append("border_vs_noise (no HDBSCAN labels, or predates emission)")
    if data.get("persistence_length") is None:
        run.missing.append("persistence_length (predates emission)")
    if not (path.parent / f"{ARTIFACT_PREFIX}_{stem}_particles.npz").exists():
        run.missing.append("particles npz (not written for this run)")
    if not (path.parent / f"{ARTIFACT_PREFIX}_{stem}_axes.npz").exists():
        run.missing.append("axes npz (predates emission)")

    return run


def load_cross_run(p1b_dir: Path) -> Optional[dict]:
    """`phase1b_cross_run.json`, or None."""
    p = Path(p1b_dir) / f"{ARTIFACT_PREFIX}_cross_run.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  ⚠  unreadable cross-run digest ({exc})")
        return None


def describe_runs(runs: Sequence[Run]) -> str:
    """The `--list_runs` report: what was found and what each run is missing."""
    if not runs:
        return "no runs found"
    lines = []
    for r in runs:
        step = "" if r.checkpoint_step is None else f"  step={r.checkpoint_step}"
        lines.append(f"  {r.stem:<52} {r.n_layers:>3} layers  "
                     f"{r.n_tokens:>4} tokens{step}")
        for reason in r.missing:
            lines.append(f"      missing: {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Column extraction
# ---------------------------------------------------------------------------

def layer_field(per_layer: Sequence[dict], name: str) -> np.ndarray:
    """
    A `per_layer` column as float, JSON null -> NaN, length preserved.

    Length preservation is the whole contract. `crossing_count`,
    `axis_rotation`, and `match_overlap` are defined for transitions L -> L+1
    and are therefore null at the last layer; dropping those rows rather than
    NaN-ing them would shift every layer index in the figure by one, silently
    and only at the end of the depth axis where nobody looks.
    """
    out = np.full(len(per_layer), np.nan, dtype=np.float64)
    for i, entry in enumerate(per_layer):
        v = entry.get(name)
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def layer_pair_field(per_layer: Sequence[dict], name: str) -> np.ndarray:
    """A `per_layer` column holding a 2-list (e.g. within_half_ip)."""
    out = np.full((len(per_layer), 2), np.nan, dtype=np.float64)
    for i, entry in enumerate(per_layer):
        v = entry.get(name)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        for j in range(2):
            try:
                out[i, j] = float(v[j])
            except (TypeError, ValueError):
                pass
    return out


def layer_strings(per_layer: Sequence[dict], name: str,
                  default: str = "invalid") -> List[str]:
    """A `per_layer` categorical column, missing/None -> `default`."""
    out = []
    for entry in per_layer:
        v = entry.get(name)
        out.append(default if v is None else str(v))
    return out


def token_field(per_token: Sequence[dict], name: str) -> np.ndarray:
    """A `per_token` column as float, null -> NaN."""
    return layer_field(per_token, name)


def token_strings(per_token: Sequence[dict], name: str = "token_str") -> List[str]:
    return [("" if e.get(name) is None else str(e.get(name))) for e in per_token]


def token_trajectories(run: "Run") -> Optional[np.ndarray]:
    """
    (n_tokens, n_layers) hemisphere assignments from `per_token`.

    The particle table carries the same thing in long form and is preferred
    when present; this is the fallback for runs written without one, so the
    ribbon figures work either way.
    """
    per_token = run.per_token
    if not per_token:
        return None
    n_L = run.n_layers
    rows = []
    for e in per_token:
        traj = e.get("hemisphere_trajectory")
        if not traj or len(traj) != n_L:
            return None
        rows.append([int(x) for x in traj])
    return np.asarray(rows, dtype=np.int8) if rows else None


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------

def _load_particles(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        from core.particles import ParticleTable
        table = ParticleTable.load(path)
    except Exception as exc:
        print(f"  ⚠  particle table unreadable ({path.name}): {exc}")
        return {}

    out: dict = {}
    for source in (getattr(table, "columns", None), getattr(table, "extra", None)):
        if isinstance(source, dict):
            for k, v in source.items():
                out[k] = np.asarray(v)
    return out


def particles(run: "Run") -> Optional[dict]:
    """`run.particles()`, as a module-level function for symmetry."""
    return run.particles()


def particle_grid(run: "Run", column: str) -> Optional[np.ndarray]:
    """
    One particle column reshaped to (n_layers, n_tokens).

    The table is emitted with `np.meshgrid(..., indexing="ij")` over
    (layer, token) and raveled, so a plain reshape recovers the grid — but
    only if the row count matches. It is checked rather than assumed: a
    mismatched reshape produces a plausible-looking image of nothing.
    """
    cols = run.particles()
    if not cols or column not in cols:
        return None
    n_L, n_T = run.n_layers, run.n_tokens
    arr = np.asarray(cols[column])
    if arr.size != n_L * n_T:
        print(f"  ⚠  {run.stem}: particle column {column!r} has {arr.size} rows, "
              f"expected {n_L * n_T} — not reshaping")
        return None
    return arr.reshape(n_L, n_T)


# ---------------------------------------------------------------------------
# Checkpoint families
# ---------------------------------------------------------------------------

def checkpoint_families(runs: Sequence[Run]) -> Dict[str, Dict[int, List[Run]]]:
    """
    Group runs into {base_model: {step: [runs]}}.

    Uses `p1b_report.checkpoint_step` / `checkpoint_base` — the phase's own
    parser, which defers in turn to Phase 1's. Models with no step (gpt2,
    albert-base-v2, and deliberately pythia-1.4b-random) produce no family and
    are not placed on a step axis.
    """
    from p1b_hemisphere.p1b_report import checkpoint_base, checkpoint_step

    fams: Dict[str, Dict[int, List[Run]]] = {}
    for r in runs:
        step = r.checkpoint_step
        base = checkpoint_base(r.model)
        if step is None:
            step = checkpoint_step(r.model)
        if step is None or base is None:
            continue
        fams.setdefault(base, {}).setdefault(int(step), []).append(r)
    return fams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_keyed(d: dict) -> dict:
    """{"7": ...} and {7: ...} both -> {7: ...}. Non-numeric keys dropped."""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _floats(seq) -> np.ndarray:
    out = np.full(len(seq), np.nan, dtype=np.float64)
    for i, v in enumerate(seq):
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out
