"""
core/run_discovery.py — Resolve run directories to (model, checkpoint step,
prompt) without guessing.

Why this module exists
----------------------
`p5_io.find_phase1_runs` matched run directories with

    model_stem in name or model_stem_hyphen in name

Phase 1 writes run dirs as `{model_name}_{prompt_key}` (run_1.py:487), so on a
checkpoint sweep the stem `pythia-410m-step1` substring-matches 12 of the
410M pilot's 27 checkpoints — every step whose digits begin with 1: steps 1,
16, 128, 1000, 11000, 13000, 15000, 17000, 19000, 100000, 120000 and 143000.
All of them collapse onto the same `prompt_key` keys, and the surviving tiebreak
(`_iter_depth`, an ALBERT-only quantity) is 0 for every Pythia dir — so the
winner was whichever `Path.iterdir()` happened to yield first. Phase 5 would
have analyzed a non-deterministic mixture of checkpoints and said nothing.

Three commitments follow from that, and they are the whole module:

1. **Anchored, never substring.** A run belongs to base `pythia-410m` iff its
   parsed base string is exactly `pythia-410m`. `parse_checkpoint_name` does
   the parsing once; no call site does its own string containment test.

2. **A checkpoint sweep returns every checkpoint.** `find_phase1_runs`
   returned `{prompt_key: run_dir}` — one directory per prompt, which cannot
   represent a sweep at all. The unit here is `{prompt_key: {step: RunRef}}`.

3. **Collisions raise.** Two directories claiming the same
   (model, prompt_key) is a real condition (a re-run written to a new
   timestamped dir, a partially-deleted sweep) and the correct response is to
   say so, not to pick one. `DuplicateRunError` carries both paths.

Provenance, in precedence order
-------------------------------
`manifest.json` (core.io.write_manifest — carries `model`, `checkpoint_step`,
`prompt_key`, `hf_revision` explicitly) > `geometry.json`'s `prompt` field >
parsing the directory name. Every RunRef records which one answered, so a
sweep assembled from directory-name guesses is visible as such rather than
indistinguishable from one assembled from manifests.

Torch-free and matplotlib-free by construction: `p1_visualization/
checkpoints.py` owns the same `-step{N}` grammar but imports matplotlib, and
`core.config` (which owns `PROMPTS`) imports torch. Neither is importable from
an analysis module that only needs to find files. `known_prompt_keys` is
therefore an optional argument — pass `core.config.PROMPTS.keys()` from a call
site that already has torch, or omit it and accept the directory-name
heuristic documented on `_split_dirname`.

Note on duplication: `_STEP_RE` here is identical to the one in
`p1_mstate_tracking/visualization/checkpoints.py:62`, which deliberately kept
the grammar out of `core/naming.py`. This module is torch- and
matplotlib-free, so `checkpoints.py` can import the grammar from here in a
follow-up rather than the two drifting. Until that happens the two regexes
must stay in sync; they are the only two copies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "RunRef",
    "DuplicateRunError",
    "parse_checkpoint_name",
    "read_run_ref",
    "discover_runs",
    "index_by_prompt_and_step",
    "sweep_for_prompt",
    "resolve_anchor",
    "sweep_report_lines",
]


# ---------------------------------------------------------------------------
# Name grammar
# ---------------------------------------------------------------------------

# Must stay in sync with p1_mstate_tracking/visualization/checkpoints.py:62.
_STEP_RE = re.compile(r"^(?P<base>.+)-step(?P<step>\d+)$")

_RANDOM_SUFFIX = "-random"


def parse_checkpoint_name(model: str) -> Tuple[str, Optional[int], bool]:
    """
    Split a model-variant string into (base, step, is_random).

        "pythia-410m-step2000" -> ("pythia-410m", 2000, False)
        "pythia-410m-step0"    -> ("pythia-410m", 0,    False)
        "pythia-1.4b-random"   -> ("pythia-1.4b", None, True)
        "gpt2-large"           -> ("gpt2-large",  None, False)

    step 0 and the random control are deliberately distinguishable: they are
    two different objects (core/pythia_registry.py's docstring is explicit
    that collapsing them is a mistake that was already made once). step 0 is
    the developmental origin and carries step=0; the norm-matched control
    carries step=None and is_random=True, so it can never land on a step axis
    by accident.
    """
    if model.endswith(_RANDOM_SUFFIX):
        return model[: -len(_RANDOM_SUFFIX)], None, True
    m = _STEP_RE.match(model)
    if m:
        return m.group("base"), int(m.group("step")), False
    return model, None, False


# ---------------------------------------------------------------------------
# RunRef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRef:
    """One Phase 1 run directory, fully resolved.

    `source` records which provenance answered — "manifest", "geometry" or
    "dirname" — so a sweep assembled from directory-name guesses is visible
    as such. `hf_revision` is present only from a manifest; it is the one
    field that can independently confirm the weights a run actually used, so
    a None here on a checkpoint run is worth noticing.
    """
    run_dir: Path
    model: str
    base: str
    step: Optional[int]
    is_random: bool
    prompt_key: str
    source: str
    hf_revision: Optional[str] = None

    @property
    def is_checkpoint(self) -> bool:
        return self.step is not None

    def label(self) -> str:
        if self.is_random:
            return f"{self.base}-random/{self.prompt_key}"
        if self.step is None:
            return f"{self.model}/{self.prompt_key}"
        return f"{self.base}@{self.step}/{self.prompt_key}"


class DuplicateRunError(RuntimeError):
    """Two run directories claim the same (model, prompt_key)."""


# ---------------------------------------------------------------------------
# Reading one directory
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _split_dirname(
    name: str,
    known_prompt_keys: Optional[Iterable[str]] = None,
) -> Optional[Tuple[str, str]]:
    """
    (model, prompt_key) from a `{model}_{prompt_key}` directory name.

    With `known_prompt_keys`: match the longest key that the name actually
    ends with, after an underscore. Unambiguous, and the only correct method
    when a model name could itself contain an underscore.

    Without it: split on the FIRST underscore. This is correct for every
    registry key in use — Pythia keys ("pythia-410m-step512"), GPT-2 keys
    ("gpt2-large"), and the suffix grammar ("@48iter", "@attn", "-random")
    are all hyphen/at-separated and contain no underscore. It becomes wrong
    the moment a model key with an underscore is added (an org-prefixed HF id
    goes through `model_name.replace('/', '_')` in run_1.py:487), which is
    exactly why `known_prompt_keys` exists and why callers that can supply it
    should.
    """
    if known_prompt_keys:
        matches = [k for k in known_prompt_keys if name.endswith("_" + k)]
        if matches:
            pk = max(matches, key=len)
            return name[: -(len(pk) + 1)], pk
        return None
    if "_" not in name:
        return None
    model, pk = name.split("_", 1)
    return model, pk


def read_run_ref(
    run_dir: Path,
    known_prompt_keys: Optional[Iterable[str]] = None,
) -> Optional[RunRef]:
    """
    Resolve one run directory. Returns None if it cannot be identified at all
    (which is a legitimate answer for a stray directory, not an error).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None

    model: Optional[str] = None
    prompt_key: Optional[str] = None
    step: Optional[int] = None
    revision: Optional[str] = None
    source = "dirname"

    # --- 1. manifest.json (core.io.write_manifest) ---
    man = _read_json(run_dir / "manifest.json")
    if man:
        m_model = man.get("model")
        m_pk = man.get("prompt_key")
        if isinstance(m_model, str) and isinstance(m_pk, str) and m_model and m_pk:
            model, prompt_key, source = m_model, m_pk, "manifest"
            revision = man.get("hf_revision")
            raw_step = man.get("checkpoint_step")
            if isinstance(raw_step, int):
                step = raw_step

    # --- 2. geometry.json (canonical for prompt; model where present) ---
    if model is None or prompt_key is None:
        geo = _read_json(run_dir / "geometry.json")
        if geo:
            g_pk = geo.get("prompt")
            g_model = geo.get("model")
            if isinstance(g_pk, str) and g_pk:
                prompt_key = g_pk
                source = "geometry"
                if isinstance(g_model, str) and g_model:
                    model = g_model
                elif run_dir.name.endswith("_" + g_pk):
                    model = run_dir.name[: -(len(g_pk) + 1)]

    # --- 3. directory name ---
    if model is None or prompt_key is None:
        split = _split_dirname(run_dir.name, known_prompt_keys)
        if split is None:
            return None
        model, prompt_key = split
        source = "dirname"

    base, parsed_step, is_random = parse_checkpoint_name(model)

    # A manifest's checkpoint_step wins over the parsed one, but a
    # disagreement is worth surfacing rather than silently preferring either.
    if step is not None and parsed_step is not None and step != parsed_step:
        raise DuplicateRunError(
            f"{run_dir}: manifest.json checkpoint_step={step} disagrees with "
            f"model name '{model}' (parses to step {parsed_step}). One of the "
            "two is wrong; refusing to guess."
        )
    if step is None:
        step = parsed_step

    return RunRef(
        run_dir=run_dir,
        model=model,
        base=base,
        step=step,
        is_random=is_random,
        prompt_key=prompt_key,
        source=source,
        hf_revision=revision,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(
    root: Path,
    base: Optional[str] = None,
    prompt_keys: Optional[Iterable[str]] = None,
    steps: Optional[Iterable[int]] = None,
    include_random: bool = True,
    known_prompt_keys: Optional[Iterable[str]] = None,
) -> List[RunRef]:
    """
    Every identifiable run directory under `root`, filtered.

    Parameters
    ----------
    base         : exact base-model match, e.g. "pythia-410m". Anchored —
                   "pythia-410m" does NOT match "pythia-410m-step..." by
                   containment; it matches because those parse to that base.
                   A full variant string ("pythia-410m-step512") is also
                   accepted and narrows to that one checkpoint.
    prompt_keys  : keep only these prompts.
    steps        : keep only these checkpoint steps. Runs with step=None
                   (non-checkpointed models, the random control) are dropped
                   when this is given, except that the random control is
                   governed by `include_random` instead.
    include_random : keep `{base}-random` runs. They carry step=None and must
                   never enter a step-indexed structure, but they are the
                   continuity control and dropping them by default would
                   quietly remove PREDICTIONS.md claim (c)'s object.

    Returns runs sorted by (prompt_key, step-or-minus-one, model) — total and
    deterministic, so two invocations on the same tree return the same list.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    want_prompts = set(prompt_keys) if prompt_keys is not None else None
    want_steps = set(steps) if steps is not None else None

    exact_model: Optional[str] = None
    want_base: Optional[str] = None
    if base is not None:
        b, s, r = parse_checkpoint_name(base)
        if s is not None or r:
            exact_model = base
            want_base = b
        else:
            want_base = base

    out: List[RunRef] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        ref = read_run_ref(child, known_prompt_keys=known_prompt_keys)
        if ref is None:
            continue
        if want_base is not None and ref.base != want_base:
            continue
        if exact_model is not None and ref.model != exact_model:
            continue
        if want_prompts is not None and ref.prompt_key not in want_prompts:
            continue
        if ref.is_random:
            if not include_random:
                continue
        elif want_steps is not None and ref.step not in want_steps:
            continue
        out.append(ref)

    out.sort(key=lambda r: (r.prompt_key,
                            -1 if r.step is None else r.step,
                            r.model))
    return out


def index_by_prompt_and_step(
    refs: Iterable[RunRef],
) -> Dict[str, Dict[Optional[int], RunRef]]:
    """
    {prompt_key: {step: RunRef}}. The random control is indexed under the key
    None, alongside step 0 rather than in place of it.

    Raises DuplicateRunError on a collision. Two directories for the same
    (prompt, step) is a condition with no correct silent resolution — a stale
    partial re-run and a legitimate re-run look identical from here.
    """
    index: Dict[str, Dict[Optional[int], RunRef]] = {}
    for ref in refs:
        key: Optional[int] = None if ref.is_random else ref.step
        bucket = index.setdefault(ref.prompt_key, {})
        prior = bucket.get(key)
        if prior is not None and prior.run_dir != ref.run_dir:
            raise DuplicateRunError(
                f"Two runs claim prompt={ref.prompt_key!r} "
                f"step={key!r} ({ref.model!r}):\n"
                f"  {prior.run_dir}\n  {ref.run_dir}\n"
                "Remove or rename one; refusing to pick."
            )
        bucket[key] = ref
    return index


def sweep_for_prompt(refs: Iterable[RunRef], prompt_key: str) -> List[RunRef]:
    """Checkpoint runs for one prompt, ascending by step. Excludes the random
    control and any run without a step — a sweep is a step-indexed object."""
    out = [r for r in refs
           if r.prompt_key == prompt_key and r.step is not None and not r.is_random]
    out.sort(key=lambda r: r.step)
    return out


def resolve_anchor(
    refs: Iterable[RunRef],
    prompt_key: str,
    step: int,
) -> RunRef:
    """The run for one (prompt, step). Raises with the available steps listed
    rather than returning None — an anchor that isn't there is a setup error,
    and the useful thing to print is what IS there."""
    sweep = sweep_for_prompt(refs, prompt_key)
    for ref in sweep:
        if ref.step == step:
            return ref
    available = [r.step for r in sweep]
    raise KeyError(
        f"No run for prompt={prompt_key!r} step={step}. "
        f"Available steps: {available}"
    )


def sweep_report_lines(refs: Iterable[RunRef]) -> List[str]:
    """Human-readable coverage summary. Prints the provenance mix explicitly:
    a sweep resolved entirely from directory names is a weaker object than one
    resolved from manifests, and that difference should be visible before the
    numbers are, not inferred afterwards."""
    refs = list(refs)
    lines: List[str] = []
    if not refs:
        return ["(no runs discovered)"]

    by_source: Dict[str, int] = {}
    for r in refs:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    lines.append(
        "provenance: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items()))
    )

    index = index_by_prompt_and_step(refs)
    for pk in sorted(index):
        steps = sorted(s for s in index[pk] if s is not None)
        extras = []
        if None in index[pk]:
            extras.append("random")
        lines.append(
            f"  {pk:24s} {len(steps):3d} checkpoints"
            + (f"  [{', '.join(extras)}]" if extras else "")
            + (f"  steps {steps[0]}..{steps[-1]}" if steps else "")
        )

    missing_rev = [r for r in refs if r.is_checkpoint and r.hf_revision is None]
    if missing_rev:
        lines.append(
            f"  [warn] {len(missing_rev)} checkpoint run(s) carry no hf_revision "
            "— weights identity is unconfirmed for those (no manifest)."
        )
    return lines
