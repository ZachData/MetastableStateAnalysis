"""
p1c_frames/visualization/loaders.py

Every disk read in this package, in one module. No figure module opens a
file; they take a `Run` (or a list of them) and draw.

THE ARTIFACT GRAMMAR, WHICH IS THE WHOLE REASON THIS MODULE IS LONG

`p1c_io.save_p1c` splits one result dict into two files by a rule that is
about SIZE, not about meaning:

    {run}/p1c.json          {"scalars": {section: {key: value}},
                             "array_keys": [dotted paths]}
    {run}/p1c_curves.npz    {dotted path: ndarray}

with `_split` walking nested dicts into dotted section names — `"A"`,
`"B.time_domain"`, `"B.residual_bracket"`, `"provenance.geometry.json"` —
and sending any ndarray with **more than 8 entries** to the npz while
everything smaller stays in the JSON as a list.

That threshold is the trap. `E.margins` is an ndarray of length n_layers:
in the npz for a 24-layer model, in the JSON for a 6-layer one. A loader
that reads only one of the two places works on Pythia-410M and silently
returns nothing on a small fixture, or the reverse. `series()` therefore
resolves BOTH, npz first, and callers never learn which file a quantity
came from. This is the artifact-contract bug class `core/artifacts.py`
exists to kill, in its size-dependent flavour.

Three more conventions worth stating before reading the code.

**NaN survives, holes do not.** `_NpEncoder` writes non-finite floats as
JSON `null` for 1-D arrays, so `series()` maps null back to NaN and never
drops a row. The step arrays are length n_layers-1 and the layer series are
length n_layers; a dropped row would shift a depth profile by one, silently,
at whichever end nobody looks.

**Absent means absent, and says which sub-experiment.** A run carries only
the blocks its inputs supported: no `norms` kills A and C, no `beta_eff`
kills A and B, no `clusters.npz` kills F, and D is never written at all
(FIGURES-1c.md G1). `Run.missing` holds one human-readable reason per
absent block — the same string a skipped figure prints — and `run.skipped`
holds the driver's own explanation where it recorded one, which is more
specific than anything this module could infer.

**Nothing here recomputes an analysis.** Verdict strings are read from the
artifact; adjudicators are imported from `p1c_frames` by the figure modules
that need them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ARTIFACT_NAME = "p1c"

#: The sub-experiments `run_1c` can write, in the order they are read.
SUBEXPS = ("A", "B", "C", "D", "E", "F")

#: What each block is, for the `--list_runs` report and the availability
#: matrix. Kept here rather than in each figure module so one rename cannot
#: leave two spellings in the output.
SUBEXP_TITLES = {
    "A": "effective integration time",
    "B": "the gamma_beta null",
    "C": "cumulant ladder / rank",
    "D": "four-frame comparison",
    "E": "hemisphere feasibility",
    "F": "spherical designs",
}

__all__ = [
    "Run", "SUBEXPS", "SUBEXP_TITLES", "discover_runs", "load_run",
    "describe_runs", "checkpoint_families", "checkpoint_step",
    "checkpoint_base", "records", "record_field", "record_strings",
    "record_matrix", "floats",
]

_STEP_RE = re.compile(r"^(?P<base>.+?)-step(?P<step>\d+)$")


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """One Phase 1c result directory, plus whatever it managed to compute."""
    model: str
    prompt: str
    stem: str
    path: Path                      # the p1c.json itself
    scalars: Dict[str, dict]
    array_keys: List[str]
    missing: List[str] = field(default_factory=list)
    _arrays: Optional[dict] = None

    # -- identity -----------------------------------------------------------

    @property
    def top(self) -> dict:
        """The unprefixed scalars: model, prompt, n_tokens, beta_used, …"""
        return self.scalars.get("", {})

    @property
    def n_tokens(self) -> int:
        return int(self.top.get("n_tokens") or 0)

    @property
    def n_layers(self) -> int:
        """
        Layer count, from whichever block recorded one.

        `A.n_layers` is authoritative when A ran. Otherwise it is inferred
        from a per-layer series, and 0 when nothing did — a figure asking
        for a depth axis on a run with no depth-indexed block should skip,
        not draw an empty one.
        """
        a = self.block("A").get("n_layers")
        if a:
            return int(a)
        for key in ("B.ip_mean", "E.margins"):
            v = self.series(key)
            if v.size:
                return int(v.size)
        for block in ("E", "F"):
            recs = records(self, block)
            if recs:
                return len(recs)
        return 0

    @property
    def beta(self) -> float:
        return float(self.top.get("beta_used") or np.nan)

    @property
    def beta_source(self) -> str:
        return str(self.top.get("beta_source") or "unknown")

    @property
    def causal(self) -> Optional[bool]:
        c = self.block("A").get("causal")
        return None if c is None else bool(c)

    @property
    def checkpoint_step(self) -> Optional[int]:
        return checkpoint_step(self.model)

    @property
    def label(self) -> str:
        return f"{self.model} / {self.prompt}"

    @property
    def skipped(self) -> dict:
        """`{sub-experiment: reason}` as the driver recorded it."""
        return self.scalars.get("skipped", {}) or {}

    @property
    def available(self) -> List[str]:
        """Phase 1 inputs the driver found (geometry / energies / …)."""
        v = self.top.get("available")
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else []

    @property
    def subexps(self) -> List[str]:
        """Which sub-experiment blocks this run actually carries."""
        return [s for s in SUBEXPS if self.has(s)]

    # -- blocks -------------------------------------------------------------

    def has(self, subexp: str) -> bool:
        """
        Did this block land? True when the section exists in either file.

        A block can be scalars-only (`F` with every layer erroring) or
        arrays-only (`A` on a long model, whose four series all exceed the
        size threshold and whose scalars land under `"A"` regardless), so
        both are checked.
        """
        pre = f"{subexp}."
        return (subexp in self.scalars
                or any(k.startswith(pre) for k in self.scalars)
                or any(k.startswith(pre) for k in self.array_keys))

    def block(self, section: str) -> dict:
        """Scalars for one dotted section, `{}` when absent."""
        return self.scalars.get(section, {}) or {}

    def scalar(self, section: str, key: str, default=np.nan) -> float:
        v = self.block(section).get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def text(self, section: str, key: str, default: str = "") -> str:
        v = self.block(section).get(key)
        return default if v is None else str(v)

    # -- arrays -------------------------------------------------------------

    def arrays(self) -> dict:
        """The curves npz, lazily. `{}` when the run wrote no large array."""
        if self._arrays is None:
            p = self.path.parent / f"{ARTIFACT_NAME}_curves.npz"
            if not p.exists():
                self._arrays = {}
            else:
                try:
                    with np.load(p) as z:
                        self._arrays = {k: np.asarray(z[k]) for k in z.files}
                except Exception as exc:
                    print(f"  ⚠  curves npz unreadable ({p.name}): {exc}")
                    self._arrays = {}
        return self._arrays

    def series(self, dotted: str) -> np.ndarray:
        """
        One array by its dotted path — `"A.h_calibrated"`, `"B.residual"` —
        from wherever `save_p1c`'s size rule happened to put it.

        Returns an empty float array when absent, so every caller can test
        `.size` and none has to distinguish "run did not compute this" from
        "run computed it and it was short".
        """
        arrs = self.arrays()
        if dotted in arrs:
            return floats(arrs[dotted])
        section, _, key = dotted.rpartition(".")
        v = self.block(section).get(key)
        if v is None:
            return np.zeros(0, dtype=np.float64)
        return floats(v)

    def has_series(self, dotted: str) -> bool:
        s = self.series(dotted)
        return bool(s.size) and bool(np.isfinite(s).any())


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(p1c_dir: Path,
                  models: Optional[Sequence[str]] = None,
                  prompts: Optional[Sequence[str]] = None) -> List[Run]:
    """
    Every `p1c.json` under `p1c_dir`, as `Run`s.

    `run_1c` writes one directory per Phase 1 run (`{--out}/{run_dir.name}/`),
    so the search is one level of nesting plus the directory itself, and the
    run's identity comes from the JSON rather than from the directory name —
    a Phase 1 run directory named by date would otherwise become the model
    name in every legend.

    Sorted by (model, prompt) so figure order is stable across invocations;
    an output directory that reorders itself between runs is one nobody can
    diff.
    """
    p1c_dir = Path(p1c_dir)
    if not p1c_dir.exists():
        return []

    paths = sorted(set(list(p1c_dir.glob(f"{ARTIFACT_NAME}.json"))
                       + list(p1c_dir.glob(f"*/{ARTIFACT_NAME}.json"))))
    runs: List[Run] = []
    for path in paths:
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
    """One `p1c.json`, with `missing` populated."""
    path = Path(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  ⚠  unreadable, skipping: {path} ({exc})")
        return None

    if "scalars" not in data:
        print(f"  ⚠  not a Phase 1c artifact, skipping: {path}")
        return None

    scalars = {str(k): (v or {}) for k, v in (data.get("scalars") or {}).items()}
    top = scalars.get("", {})
    stem = (path.parent.name if path.name == f"{ARTIFACT_NAME}.json"
            else path.stem)

    run = Run(model=str(top.get("model") or "?"),
              prompt=str(top.get("prompt") or "?"),
              stem=stem, path=path, scalars=scalars,
              array_keys=[str(k) for k in (data.get("array_keys") or [])])

    # Each reason names the block and why it is absent, because this string
    # is what a skipped figure prints. The driver's own explanation is more
    # specific than anything inferable here, so it wins where it exists.
    skipped = run.skipped
    for sub in SUBEXPS:
        if run.has(sub):
            continue
        why = skipped.get(sub)
        if why:
            run.missing.append(f"{sub} ({SUBEXP_TITLES[sub]}): {why}")
        elif sub == "D":
            run.missing.append(
                "D (four-frame comparison): run_1c has no D branch — "
                "FIGURES-1c.md G1, not a property of this run")
        else:
            run.missing.append(
                f"{sub} ({SUBEXP_TITLES[sub]}): not selected (--subexp)")

    if run.has("B") and not run.has_series("B.envelope_lower"):
        note = run.text("B", "envelope_note") or run.text("B", "envelope_error")
        run.missing.append(
            "B beta envelope: " + (note or "no per-head beta_eff in "
                                   "geometry.json — FIGURES-1c.md G4"))
    if run.has("A") and not run.has_series("A.h_attn_only"):
        run.missing.append(
            "A h_attn_only: no sublayer streams, so the frame-correct step "
            "definition is nan — FIGURES-1c.md G3")
    if run.has("C") and not run.block("C").get("moment_identity"):
        run.missing.append(
            "C moment_identity: no energies.json for this run")

    return run


def describe_runs(runs: Sequence[Run]) -> str:
    """The `--list_runs` report: what landed, and what each run is missing."""
    if not runs:
        return "no runs found"
    lines = []
    for r in runs:
        step = "" if r.checkpoint_step is None else f"  step={r.checkpoint_step}"
        got = "".join(s if r.has(s) else "·" for s in SUBEXPS)
        lines.append(f"  {r.stem:<44} {r.n_layers:>3} layers  "
                     f"{r.n_tokens:>4} tokens  [{got}]  "
                     f"beta={r.beta:.3g} ({r.beta_source}){step}")
        for reason in r.missing:
            lines.append(f"      missing: {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Records — the per-layer lists that stayed in the JSON
# ---------------------------------------------------------------------------

def records(run: Run, section: str, key: str = "per_layer") -> List[dict]:
    """
    A block's per-layer list of dicts (`E.per_layer`, `F.per_layer`,
    `C.panels`), as stored.

    These never reach the npz — `_split` only diverts ndarrays — so they are
    read straight out of the scalars. Non-dict entries are dropped rather
    than crashing a comprehension three modules away.
    """
    v = run.block(section).get(key)
    return [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []


def record_field(recs: Sequence[dict], name: str,
                 n: Optional[int] = None) -> np.ndarray:
    """
    One column out of a per-layer record list, as float, missing -> NaN.

    `n` pads to a known layer count. F's `per_layer` skips layers whose
    centroids could not be loaded, so its records are NOT one-per-layer and
    are indexed by their own `layer` key; passing `n` scatters them back
    onto the depth axis instead of compressing the gaps, which would draw a
    24-layer profile as an 18-layer one.
    """
    if n is None:
        out = np.full(len(recs), np.nan, dtype=np.float64)
        for i, r in enumerate(recs):
            out[i] = _as_float(r.get(name))
        return out

    out = np.full(int(n), np.nan, dtype=np.float64)
    for i, r in enumerate(recs):
        idx = r.get("layer", i)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        if 0 <= idx < out.size:
            out[idx] = _as_float(r.get(name))
    return out


def record_strings(recs: Sequence[dict], name: str,
                   default: str = "") -> List[str]:
    return [default if r.get(name) is None else str(r.get(name)) for r in recs]


def record_matrix(recs: Sequence[dict], name: str, width: int,
                  n: Optional[int] = None) -> np.ndarray:
    """
    A per-layer column that is itself a list — `Q_ratio`, `random_band`,
    `outside_band` — as a (layers, width) array.

    Short rows are NaN-padded rather than raising: `t_max` is a CLI choice
    and a directory can hold runs made at two different ones.
    """
    rows = int(n) if n is not None else len(recs)
    out = np.full((rows, int(width)), np.nan, dtype=np.float64)
    for i, r in enumerate(recs):
        idx = i
        if n is not None:
            try:
                idx = int(r.get("layer", i))
            except (TypeError, ValueError):
                idx = i
        if not (0 <= idx < rows):
            continue
        v = r.get(name)
        if not isinstance(v, (list, tuple)):
            continue
        for j, x in enumerate(v[:width]):
            out[idx, j] = _as_float(x)
    return out


# ---------------------------------------------------------------------------
# Checkpoint families
# ---------------------------------------------------------------------------

def checkpoint_step(model: str) -> Optional[int]:
    """'pythia-410m-step2000' -> 2000; None for non-checkpoint names."""
    try:
        from p1_mstate_tracking.visualization.checkpoints import _checkpoint_step
        return _checkpoint_step(model)
    except Exception:
        m = _STEP_RE.match(str(model))
        return int(m.group("step")) if m else None


def checkpoint_base(model: str) -> Optional[str]:
    """'pythia-410m-step2000' -> 'pythia-410m'; None for non-checkpoints."""
    try:
        from p1_mstate_tracking.visualization.checkpoints import _checkpoint_base
        return _checkpoint_base(model)
    except Exception:
        m = _STEP_RE.match(str(model))
        return m.group("base") if m else None


def checkpoint_families(runs: Sequence[Run],
                        ) -> Dict[str, Dict[str, Dict[int, Run]]]:
    """
    Group runs into `{base_model: {prompt: {step: run}}}`.

    Nested by prompt as well as by step because $t^\\ast$ is n-dependent and
    the prompts span 20-512 tokens (status-1c open item 4): a step axis
    pooled over prompts would compare each checkpoint against a different
    collapse time and call the difference training. Models with no step —
    gpt2, and deliberately any `-random` control — produce no family and are
    not placed on a step axis at all.
    """
    fams: Dict[str, Dict[str, Dict[int, Run]]] = {}
    for r in runs:
        step, base = checkpoint_step(r.model), checkpoint_base(r.model)
        if step is None or base is None:
            continue
        fams.setdefault(base, {}).setdefault(r.prompt, {})[int(step)] = r
    return fams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def floats(seq) -> np.ndarray:
    """Anything list-like -> float array, None/non-numeric -> NaN."""
    arr = np.asarray(seq, dtype=object).ravel()
    out = np.full(arr.size, np.nan, dtype=np.float64)
    for i, v in enumerate(arr):
        out[i] = _as_float(v)
    return out


def _as_float(v) -> float:
    if v is None or isinstance(v, (dict, list, tuple, str, bytes)):
        return float(v) if isinstance(v, str) and _is_number(v) else np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
