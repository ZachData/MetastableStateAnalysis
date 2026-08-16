"""
p2b_imaginary/visualization/loaders.py

Every disk read in this package, in one module. No figure module opens a
file; they take a `Sweep` (or a `Checkpoint`, or one Block 1b record) and
draw.

Three things this module is deliberately strict about.

**The combined file is the artifact, and the subdirectories are the
fallback.** `run_2b.run_sweep` embeds each checkpoint's Block 1a JSON and
every prompt's Block 1b JSON inside `phase2b_results.json`, so one read gets
the whole sweep. But a sweep interrupted partway through has written every
`{stem}/block1a_rotational_spectrum.json` it got to and no combined file at
all, and that directory is exactly the one someone wants to look at. So
discovery tries the combined file first and reconstructs from the
subdirectories when it is absent, reporting which path it took.

**A pre-rewrite directory is refused, by the phase's own function.**
`p2b_io.refuse_legacy_run_dir` raises on a directory holding
`phase2i_results.json`, because those numbers were scored with an absolute
1e-6 threshold and a 3.0 rank gate and their `elim_rotation` column is an
algebraic identity. That check is imported, not restated — a second copy here
would be a second place for the refusal to go stale.

**Gaps are DETECTED, not assumed.** Four of the seven data gaps in
FIGURES-2b.md are unconditional properties of the serializers as they stand
today (G1, G2, G3, G7), and it would be easy to hardcode "these figures
cannot be drawn". Instead `Sweep.gaps` looks for the key in the artifact. The
day someone lands the emission in `p2b_imaginary/`, the figures that need it
start drawing against new directories with no change here, and keep skipping
against old ones — which is the only behaviour that lets both kinds of
directory stay readable.

Nothing here recomputes an analysis. Per-layer columns come out as float
arrays with JSON `null` mapped to NaN; a dropped row would silently shift
every depth profile in the package by one layer, which is the failure mode
worth being paranoid about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from p2b_imaginary import p2b_io

__all__ = [
    "Checkpoint", "Sweep", "load_sweep", "describe_sweep",
    "layer_field", "layer_strings", "depth_matrix", "reference_beta",
    "frame_counts", "elim_row", "frame_series", "rel_drops", "null_stat",
    "GAPS",
    "checkpoint_out", "prompt_out", "cross_out",
]


#: The gaps this package can detect, and the key whose presence closes each.
#: `where` is the function that drops it, so the fix has an address. Kept as
#: data rather than prose because `--list_runs` prints it and FIGURES-2b.md's
#: table is generated from the same source of truth by hand.
GAPS: Tuple[dict, ...] = (
    {"id": "G1", "key": "per_layer",
     "what": "per-layer energies / effective rank / IP summaries per frame",
     "where": "rotational_rescaled.comparison_to_json"},
    {"id": "G2", "key": "r_cum_max_abs",
     "what": "the rescaler growth curve (only its maximum survives)",
     "where": "rotational_rescaled.comparison_to_json"},
    {"id": "G3", "key": "planes.npz",
     "what": "the per-plane rho / theta / sign spectrum",
     "where": "rotational_schur.summary_to_json + run_2b.run_block_1a"},
    {"id": "G4", "key": "head_circuits",
     "what": "per-head circuit results",
     "where": "run_2b.run_head_circuits"},
    {"id": "G5", "key": "precision",
     "what": "the fp16 precision surface (--with-precision)",
     "where": "run_2b.run_block_1a"},
    {"id": "G7", "key": "rel_drops",
     "what": "per-transition violation severity",
     "where": "rotational_rescaled.comparison_to_json"},
)


# ---------------------------------------------------------------------------
# One checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """One checkpoint's Phase 2b result: Block 1a once, Block 1b per prompt."""
    stem: str
    step: Optional[int]
    data: dict
    #: Human-readable reasons a figure might have to skip, phrased as what is
    #: absent AND what caused it — this string is what a skipped figure
    #: prints, and "block1b" alone would send a reader to the wrong place.
    missing: List[str] = field(default_factory=list)
    #: The directory this checkpoint's subresults were written to, when the
    #: sweep is being read from disk. `None` for a Sweep assembled in memory,
    #: which is why every sidecar read goes through `planes()` and returns
    #: `None` rather than raising.
    path: Optional[Path] = None
    _planes: Optional[dict] = None

    # -- identity -----------------------------------------------------------

    @property
    def status(self) -> str:
        return str(self.data.get("status") or "ok")

    @property
    def label(self) -> str:
        return (self.stem if self.step is None
                else f"{self.stem}  (step {self.step})")

    @property
    def n_ov_layers(self) -> int:
        n = self.data.get("n_ov_layers")
        if n:
            return int(n)
        return len(self.per_layer)

    @property
    def wall_time_seconds(self) -> Optional[float]:
        w = self.data.get("wall_time_seconds")
        return None if w is None else float(w)

    @property
    def failures(self) -> List[dict]:
        return list(self.data.get("failures") or [])

    # -- Block 1a -----------------------------------------------------------

    @property
    def block1a(self) -> Optional[dict]:
        return self.data.get("block1a")

    @property
    def per_layer(self) -> List[dict]:
        return list((self.block1a or {}).get("per_layer") or [])

    @property
    def summary(self) -> dict:
        return dict((self.block1a or {}).get("summary") or {})

    @property
    def layer_names(self) -> List[str]:
        names = (self.block1a or {}).get("layer_names")
        if names:
            return [str(n) for n in names]
        return [f"layer_{i}" for i in range(len(self.per_layer))]

    @property
    def has_nulls(self) -> bool:
        return any(rec.get("nulls") for rec in self.per_layer)

    def null_statistics(self) -> List[str]:
        """Which statistics carry a null, in first-layer order."""
        for rec in self.per_layer:
            if rec.get("nulls"):
                return list(rec["nulls"])
        return []

    # -- Block 1b -----------------------------------------------------------

    @property
    def block1b(self) -> Dict[str, dict]:
        """{prompt: record}. Records with no `interpretation` are refusals
        (`no_activations`, `failed`) and are kept, not filtered — a prompt
        that failed must be visible as a failure, not as an absence."""
        return dict(self.data.get("block1b") or {})

    def block1b_scored(self) -> Dict[str, dict]:
        """Only the prompts that produced a comparison."""
        return {k: v for k, v in self.block1b.items() if "interpretation" in v}

    # -- the per-plane spectrum (sidecar) ------------------------------------

    def planes(self) -> Optional[dict]:
        """
        `planes.npz` as `{layer_name: {rho, theta, sign, idx}}`, lazily.

        The distribution the per-layer angle statistics summarise. Loaded
        through `p2b_io.load_sidecar` so the filename comes from the artifact
        contract rather than from a string here, and returns None when the
        sidecar is absent — which is a real state (`--no-planes`, or a run
        predating the emission) and not an error.
        """
        if self._planes is None:
            self._planes = self._load_planes()
        return self._planes or None

    def _load_planes(self) -> dict:
        if self.path is None or not self.path.exists():
            return {}
        try:
            arrays = p2b_io.load_sidecar(self.path, "planes")
        except Exception as exc:
            print(f"  ⚠  {self.stem}: planes sidecar unreadable ({exc})")
            return {}
        if not arrays:
            return {}
        names = [str(n) for n in arrays.get("layer_names", [])]
        out: dict = {}
        for name in names:
            rec = {f: np.asarray(arrays[f"{name}__{f}"])
                   for f in ("rho", "theta", "sign", "idx")
                   if f"{name}__{f}" in arrays}
            if rec:
                out[name] = rec
        return out

    def plane_column(self, field_name: str) -> np.ndarray:
        """
        One per-plane field pooled over every layer of this checkpoint.

        For the distribution figures, which are about the spectrum rather
        than about depth. `plane_layer_index` gives the matching depth label
        for a coloured version.
        """
        planes = self.planes() or {}
        parts = [planes[name][field_name] for name in self.layer_names
                 if name in planes and field_name in planes[name]]
        return (np.concatenate(parts) if parts
                else np.zeros(0, dtype=np.float64))

    def plane_layer_index(self) -> np.ndarray:
        """Depth index per pooled plane, matching `plane_column`'s order."""
        planes = self.planes() or {}
        parts = []
        for i, name in enumerate(self.layer_names):
            if name in planes and "rho" in planes[name]:
                parts.append(np.full(planes[name]["rho"].size, i, dtype=int))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=int)

    # -- the other two late emissions ---------------------------------------

    @property
    def head_circuits(self) -> Optional[dict]:
        """Per-head circuit results, or None when the block did not run."""
        return self.data.get("head_circuits") or (self.block1a or {}).get(
            "head_circuits")

    @property
    def precision(self) -> Optional[dict]:
        """The fp16 / tolerance surface, or None (`--with-precision` is
        opt-in and costs ~10 eigendecompositions per layer)."""
        return (self.block1a or {}).get("precision")

    def field(self, name: str) -> np.ndarray:
        """A Block 1a `per_layer` column as floats, JSON null -> NaN."""
        return layer_field(self.per_layer, name)

    def has(self, what: str) -> bool:
        """Is an optional input present? `what` is one of the strings in
        `missing`, so a figure's guard and its skip message cannot disagree."""
        return not any(what in m for m in self.missing)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

@dataclass
class Sweep:
    """One Phase 2b output directory."""
    path: Path
    combined: dict
    checkpoints: List[Checkpoint]
    #: "combined" when `phase2b_results.json` was read, "subdirectories" when
    #: it was reconstructed from an interrupted sweep.
    source: str = "combined"

    # -- run configuration --------------------------------------------------

    @property
    def base(self) -> Optional[str]:
        return self.combined.get("base")

    @property
    def blocks(self) -> List[str]:
        return [str(b) for b in (self.combined.get("blocks") or [])]

    @property
    def betas(self) -> List[float]:
        return [float(b) for b in (self.combined.get("betas") or [])]

    @property
    def counting_rule(self) -> dict:
        return dict(self.combined.get("counting_rule") or {})

    @property
    def missing_checkpoints(self) -> List[int]:
        return [int(s) for s in (self.combined.get("missing_checkpoints") or [])]

    @property
    def n_failed(self) -> int:
        return int(self.combined.get("n_failed") or 0)

    # -- selections ---------------------------------------------------------

    @property
    def with_1a(self) -> List[Checkpoint]:
        return [c for c in self.checkpoints if c.block1a]

    @property
    def with_1b(self) -> List[Checkpoint]:
        return [c for c in self.checkpoints if c.block1b_scored()]

    @property
    def stepped(self) -> List[Checkpoint]:
        """Checkpoints with a step, in training order. A stem with no
        `-step{N}` has no place on a training axis and is not given one."""
        return sorted((c for c in self.with_1a if c.step is not None),
                      key=lambda c: c.step)

    @property
    def steps(self) -> List[int]:
        return [int(c.step) for c in self.stepped]

    @property
    def prompts(self) -> List[str]:
        out: List[str] = []
        for c in self.checkpoints:
            for p in c.block1b:
                if p not in out:
                    out.append(p)
        return sorted(out)

    @property
    def has_trajectory(self) -> bool:
        """Two stepped checkpoints is the minimum for anything on the
        training axis. One is a point, and a point drawn as a trajectory is
        how a single measurement acquires a slope."""
        return len(self.stepped) >= 2

    @property
    def has_nulls(self) -> bool:
        return any(c.has_nulls for c in self.checkpoints)

    @property
    def combined_view(self) -> dict:
        """
        `combined`, with `results` restricted to the checkpoints actually
        loaded.

        Everything in `p2b_report` takes the combined dict, and passing the
        unfiltered one would make `--steps 512 3000` silently draw a
        trajectory over all 27 — the filter would apply to the per-checkpoint
        figures and not to the cross-checkpoint ones, which is worse than not
        filtering at all. `missing_checkpoints` is left alone: a step the
        SWEEP never produced and a step the CALLER excluded are different
        facts, and only the first belongs on a coverage figure.
        """
        out = dict(self.combined)
        out["results"] = {c.stem: c.data for c in self.checkpoints}
        return out

    # -- gaps ---------------------------------------------------------------

    @property
    def gaps(self) -> List[dict]:
        """Which of the FIGURES-2b.md data gaps are open in THIS directory."""
        return [g for g in GAPS if not self._gap_closed(g)]

    def has_gap(self, gap_id: str) -> bool:
        """True when `gap_id` is OPEN — i.e. the quantity is absent."""
        return any(g["id"] == gap_id for g in self.gaps)

    def gap_reason(self, gap_id: str) -> str:
        for g in GAPS:
            if g["id"] == gap_id:
                return (f"{g['id']}: {g['what']} — dropped by {g['where']} "
                        f"(see FIGURES-2b.md)")
        return gap_id

    def _gap_closed(self, gap: dict) -> bool:
        key = gap["key"]
        if gap["id"] in ("G1", "G2", "G7"):
            for c in self.with_1b:
                for js in c.block1b_scored().values():
                    for fr in (js.get("frames") or {}).values():
                        if key in fr:
                            return True
                        for counts in (fr.get("counts") or {}).values():
                            if key in counts:
                                return True
            return False
        if gap["id"] == "G3":
            # The arrays live in a sidecar, so presence is a file question,
            # not a key question — and the Block 1a JSON says which state a
            # run is in (`has_plane_arrays`), so a run that deliberately
            # passed --no-planes is distinguishable from one that predates
            # the emission.
            return any(c.planes() for c in self.with_1a)
        if gap["id"] == "G4":
            return any(key in c.data or key in (c.block1a or {})
                       for c in self.checkpoints)
        if gap["id"] == "G5":
            return any(key in (c.block1a or {}) for c in self.with_1a)
        return False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def load_sweep(p2b_dir, steps: Optional[Sequence[int]] = None,
               prompts: Optional[Sequence[str]] = None) -> Optional[Sweep]:
    """
    One Phase 2b output directory as a `Sweep`, or None.

    `steps` and `prompts` filter what is loaded. Filtering by step keeps the
    combined file's `missing_checkpoints` intact, because a step the SWEEP
    never produced and a step the CALLER excluded are different facts and only
    the first belongs on the coverage figure.
    """
    p2b_dir = Path(p2b_dir)
    if not p2b_dir.exists():
        print(f"⚠  no such directory: {p2b_dir}")
        return None

    # Imported from the phase: the refusal and its reasoning live there.
    p2b_io.refuse_legacy_run_dir(p2b_dir)

    combined_path = p2b_dir / p2b_io.COMBINED_RESULTS
    if combined_path.exists():
        try:
            with open(combined_path) as f:
                combined = json.load(f)
            source = "combined"
        except Exception as exc:
            print(f"⚠  unreadable {combined_path.name} ({exc}) — "
                  "falling back to the per-checkpoint subdirectories")
            combined, source = _reconstruct_from_dirs(p2b_dir), "subdirectories"
    else:
        combined, source = _reconstruct_from_dirs(p2b_dir), "subdirectories"

    results = combined.get("results") or {}
    if not results:
        print(f"⚠  no Phase 2b results under {p2b_dir}")
        return None

    wanted_steps = None if steps is None else {int(s) for s in steps}
    checkpoints: List[Checkpoint] = []
    for stem, rec in results.items():
        step = rec.get("checkpoint_step")
        step = None if step is None else int(step)
        if wanted_steps is not None and step not in wanted_steps:
            continue
        if prompts is not None and rec.get("block1b"):
            rec = dict(rec)
            rec["block1b"] = {k: v for k, v in rec["block1b"].items()
                              if k in set(prompts)}
        checkpoints.append(_as_checkpoint(stem, rec, root=p2b_dir))

    checkpoints.sort(key=lambda c: (c.step is None, c.step or 0, c.stem))
    return Sweep(path=p2b_dir, combined=combined, checkpoints=checkpoints,
                 source=source)


def _as_checkpoint(stem: str, rec: dict,
                   root: Optional[Path] = None) -> Checkpoint:
    step = rec.get("checkpoint_step")
    if step is None:
        # The stem grammar is stdlib-only and lives in core/model_family.py;
        # `p2b_io` re-exports it so a caller needs one import for one grammar.
        step = p2b_io.checkpoint_step_of(stem)
    ck = Checkpoint(stem=str(stem), step=None if step is None else int(step),
                    data=rec, path=None if root is None else Path(root) / stem)

    if rec.get("status") == "no_ov_weights":
        ck.missing.append("everything (Phase 2 wrote no OV weights for this step)")
        return ck
    if not rec.get("block1a"):
        ck.missing.append("block1a (sweep did not run block 1a)")
    if not rec.get("block1b"):
        ck.missing.append("block1b (sweep ran --blocks 1a, or found no Phase 1 "
                          "run with activations)")
    elif not any("interpretation" in v for v in rec["block1b"].values()):
        ck.missing.append("block1b comparisons (every prompt refused or failed)")
    if rec.get("block1a") and not any(
            r.get("nulls") for r in rec["block1a"].get("per_layer") or []):
        ck.missing.append("nulls (Block 1a run without --with-nulls)")
    if rec.get("failures"):
        ck.missing.append(f"{len(rec['failures'])} prompt(s) failed "
                          "(--continue-on-error was set)")
    return ck


def _reconstruct_from_dirs(p2b_dir: Path) -> dict:
    """
    Rebuild a combined-file-shaped dict from the per-checkpoint subresults.

    For an interrupted sweep, which has written every subresult it reached and
    no combined file. The reconstruction is explicitly partial: `betas` and
    `counting_rule` are recovered from the Block 1b records where present,
    `missing_checkpoints` is left empty because nothing on disk records what
    was ASKED for, and `source` on the resulting Sweep says which path was
    taken so a coverage figure does not claim a completeness it cannot know.
    """
    b1a_name = p2b_io.subresult_filename("block1a_rotational_spectrum")
    b1b_name = p2b_io.subresult_filename("block1b_rescaled_comparison")

    results: dict = {}
    betas: List[float] = []
    counting_rule: dict = {}

    for sub in sorted(p for p in p2b_dir.iterdir() if p.is_dir()):
        if sub.name.startswith("_"):
            continue
        rec: dict = {"model_stem": sub.name,
                     "checkpoint_step": p2b_io.checkpoint_step_of(sub.name),
                     "status": "ok", "failures": []}
        b1a = _read_json(sub / b1a_name)
        if b1a:
            rec["block1a"] = b1a
            rec["n_ov_layers"] = len(b1a.get("per_layer") or [])
        per_prompt: dict = {}
        for pdir in sorted(p for p in sub.iterdir() if p.is_dir()):
            js = _read_json(pdir / b1b_name)
            if js:
                per_prompt[pdir.name] = js
                counting_rule = counting_rule or (js.get("counting_rule") or {})
                for b in (js.get("comparison") or {}):
                    if float(b) not in betas:
                        betas.append(float(b))
        if per_prompt:
            rec["block1b"] = per_prompt
        if "block1a" in rec or "block1b" in rec:
            results[sub.name] = rec

    blocks = []
    if any("block1a" in r for r in results.values()):
        blocks.append("1a")
    if any("block1b" in r for r in results.values()):
        blocks.append("1b")

    return {
        "phase": "2b",
        "base": None,
        "blocks": blocks,
        "betas": sorted(betas),
        "counting_rule": counting_rule,
        "n_checkpoints": len(results),
        "n_failed": 0,
        "missing_checkpoints": [],
        "steps": sorted(r["checkpoint_step"] for r in results.values()
                        if r.get("checkpoint_step") is not None),
        "results": results,
        "reconstructed": True,
    }


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  ⚠  unreadable, skipping: {path} ({exc})")
        return None


def describe_sweep(sweep: Sweep) -> str:
    """The `--list_runs` report: what was found, and what is missing."""
    if sweep is None:
        return "no sweep loaded"
    lines = [
        f"  source: {sweep.source}"
        + ("   (interrupted sweep — no combined file)"
           if sweep.source == "subdirectories" else ""),
        f"  base: {sweep.base}   blocks: {', '.join(sweep.blocks) or 'none'}"
        f"   betas: {sweep.betas}",
        f"  counting rule: {sweep.counting_rule}",
        f"  checkpoints: {len(sweep.checkpoints)} "
        f"({len(sweep.with_1a)} with block 1a, {len(sweep.with_1b)} with 1b)",
        f"  prompts: {', '.join(sweep.prompts) or 'none'}",
        f"  missing checkpoints: {sweep.missing_checkpoints or 'none'}",
        f"  failed prompts: {sweep.n_failed}",
        "",
    ]
    for c in sweep.checkpoints:
        lines.append(f"  {c.stem:<40} step={str(c.step):>7}  "
                     f"{c.n_ov_layers:>3} OV layers  "
                     f"{len(c.block1b_scored())} scored prompt(s)")
        for reason in c.missing:
            lines.append(f"      missing: {reason}")
    open_gaps = sweep.gaps
    lines.append("")
    if open_gaps:
        lines.append("  open data gaps (fix belongs in p2b_imaginary/, not here):")
        for g in open_gaps:
            lines.append(f"      {g['id']}  {g['what']}")
            lines.append(f"           dropped by {g['where']}")
    else:
        lines.append("  no open data gaps — every emission has landed")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
#
# Mirrors the INPUT layout, deliberately: `{out}/{stem}/{prompt}/` is where
# the artifact being drawn lives under `--p2b_dir`, so a figure and the JSON
# it came from sit at the same relative path in two trees. `_cross/` is
# underscore-prefixed so it sorts above the checkpoint directories and is
# obviously not a checkpoint.

def checkpoint_out(out_dir, checkpoint: Checkpoint) -> Path:
    return Path(out_dir) / checkpoint.stem


def prompt_out(out_dir, checkpoint: Checkpoint, prompt: str) -> Path:
    return Path(out_dir) / checkpoint.stem / (prompt or "_no_prompt")


def cross_out(out_dir) -> Path:
    return Path(out_dir) / "_cross"


# ---------------------------------------------------------------------------
# Column extraction
# ---------------------------------------------------------------------------

def layer_field(per_layer: Sequence[dict], name: str) -> np.ndarray:
    """
    A `per_layer` column as float, JSON null -> NaN, length preserved.

    Length preservation is the contract. `p2b_io.json_default` maps a
    non-finite float to JSON `null`, and NaN is a real value here — a layer
    with no 2x2 blocks has NaN for every angle statistic by construction
    (`rotation_angle_stats`), which is a fact about that layer, not a missing
    row. Dropping it would shift every depth profile in the package by one.
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


def layer_strings(per_layer: Sequence[dict], name: str,
                  default: str = "invalid") -> List[str]:
    """A `per_layer` categorical column, missing/None -> `default`."""
    return [default if e.get(name) is None else str(e.get(name))
            for e in per_layer]


def depth_matrix(checkpoints: Sequence[Checkpoint],
                 key: str) -> Tuple[List[int], np.ndarray]:
    """
    (steps, matrix) with matrix (n_steps, n_layers) for one per-layer scalar.

    Checkpoints with a different layer count are padded with NaN rather than
    dropped or truncated: a family whose 410m and 1.4b entries got into one
    directory is a mistake worth SEEING as a ragged heatmap, not one worth
    silently reconciling into a rectangle.
    """
    stepped = [c for c in checkpoints if c.step is not None and c.block1a]
    stepped.sort(key=lambda c: c.step)
    if not stepped:
        return [], np.zeros((0, 0))
    n_layers = max(len(c.per_layer) for c in stepped)
    mat = np.full((len(stepped), n_layers), np.nan, dtype=np.float64)
    for i, c in enumerate(stepped):
        col = layer_field(c.per_layer, key)
        mat[i, :col.size] = col
    return [int(c.step) for c in stepped], mat


# ---------------------------------------------------------------------------
# Block 1b accessors
# ---------------------------------------------------------------------------

def reference_beta(js: dict) -> Optional[str]:
    """
    The beta a Block 1b record's verdict was taken at, as the STRING key the
    counts and comparison dicts are keyed by.

    `interpretation.reference_beta` is a float and survives JSON as one;
    `frames[*].counts` and `comparison` are keyed by `str(beta)`. Resolving
    that once here rather than at nine call sites is the same
    artifact-contract discipline `core/artifacts.py` exists for — and the
    float-to-string round trip is not free: `str(1.0)` is `"1.0"`, but a beta
    of `0.1` written by one path and `float("0.1")` by another must still
    land on the same key, so the lookup falls back to a numeric match.
    """
    interp = js.get("interpretation") or {}
    ref = interp.get("reference_beta")
    if ref is None:
        return None
    counts_keys = list((js.get("comparison") or {}).keys())
    if str(ref) in counts_keys:
        return str(ref)
    for k in counts_keys:
        try:
            if float(k) == float(ref):
                return k
        except (TypeError, ValueError):
            continue
    return str(ref)


def frame_counts(js: dict, frame: str, beta: Optional[str] = None) -> dict:
    """One frame's violation counts at `beta` (default: the reference beta)."""
    beta = beta or reference_beta(js)
    fr = (js.get("frames") or {}).get(frame) or {}
    return dict((fr.get("counts") or {}).get(str(beta)) or {})


def elim_row(js: dict, beta: Optional[str] = None) -> dict:
    """
    The elimination-rate row at `beta`: {name: {rate, status, ...}}.

    `rate` is None for every refusal, and callers must keep it None. Mapping
    it to 0.0 anywhere in this package would reproduce the exact defect the
    phase was reopened to fix.
    """
    beta = beta or reference_beta(js)
    return dict((js.get("comparison") or {}).get(str(beta)) or {})


def frame_series(js: dict, frame: str, name: str,
                 beta: Optional[str] = None) -> np.ndarray:
    """
    One per-frame per-layer series as floats, JSON null -> NaN.

    `name` is `energies` (which is per beta), or any of `effective_rank`,
    `ip_mean`, `ip_mass_near_1`, or `r_cum_max_abs` — the last living beside
    `per_layer` rather than inside it, because the rescaler's growth does not
    depend on beta. Returns an empty array when the series is absent, which is
    what a run predating the emission looks like.
    """
    fr = (js.get("frames") or {}).get(frame) or {}
    if name == "r_cum_max_abs":
        values = fr.get("r_cum_max_abs")
    else:
        per_layer = fr.get("per_layer") or {}
        if name == "energies":
            values = (per_layer.get("energies") or {}).get(
                str(beta or reference_beta(js)))
        else:
            values = per_layer.get(name)
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.array([np.nan if v is None else float(v) for v in values],
                    dtype=np.float64)


def rel_drops(js: dict, frame: str, beta: Optional[str] = None) -> np.ndarray:
    """
    Per-transition relative energy drop, NaN where unscored.

    Length `n_layers - 1`; position L-1 is the transition L-1 -> L, matching
    `violation_layers`' L. Getting that off by one would put every marked
    violation one layer from its severity.
    """
    c = frame_counts(js, frame, beta)
    values = c.get("rel_drops")
    if values is None:
        return np.zeros(0, dtype=np.float64)
    return np.array([np.nan if v is None else float(v) for v in values],
                    dtype=np.float64)


def null_stat(checkpoint: Checkpoint, statistic: str,
              key: str = "z_score") -> np.ndarray:
    """
    One null field per layer, e.g. `z_score` or `percentile`, NaN where the
    layer carries no null.
    """
    out = np.full(len(checkpoint.per_layer), np.nan, dtype=np.float64)
    for i, rec in enumerate(checkpoint.per_layer):
        res = (rec.get("nulls") or {}).get(statistic)
        if not res:
            continue
        v = res.get(key)
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out
