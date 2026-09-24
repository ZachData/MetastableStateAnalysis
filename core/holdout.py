"""
core/holdout.py — The Phase 10 confirmation set, enforced in code.

The rule (user, 2026-09-23; `p10_cluster_function/handoff-10.md` §0.4): the
twelve prompts v2 added to the battery are Phase 10's confirmation set.
Stage 0 runs all twenty on pythia-410m, but Stages 1-5 read only the eight
v1 metastability prompts until the predictions scored on the twelve are in
`claims/registry.json`. A rule kept only in docs has been missed before
(`LESSONS.md`), so every runner that reads `data/phase12` passes its inputs
through `refuse_held_out` before opening them, or is exempt for a stated
reason, and `tests/test_holdout.py` fails if one is neither.

What counts as held out, by path alone:

1. A run directory for a held-out prompt, by its `manifest.json`
   `prompt_key` when it has one, else by its `{model}_{prompt_key}` name.
   Any model, not only 410m: `CLAIM-C` already ran the twelve on 1.4b and
   gpt2-large (`data/phase12/2026-09-19_*`), and each further read of those
   runs makes the twelve less blind.
2. Any file or directory whose name carries a held-out key as a whole
   `_`-separated token: the per-prompt plots Phase 1 writes beside its run
   directories (`{model}_{key}_pca.png`, `cross_model_{key}.png`).
3. A directory that contains a held-out run directory, or a file beside
   one whose name carries no v1 key: a timestamp directory's
   `pair_agreement.json` or `llm_cross_run_report.txt` pools every prompt of
   that invocation.
4. `CLAIM-C`'s per-prompt record, `claim_c_real_run.json`, which carries
   cluster count, membership, effective rank and Fiedler for the twelve, and
   Stage 0's `stage0_logs/*.out` / `*.log`, which print per-prompt values.

Rules 3 and 4 are "pooled": they hold v1 values too, so `--v1-only` refuses
them rather than drop them; a reader of one keeps only `V1_PROMPT_KEYS`.

Checking that a Stage 0 output is populated is not reading it
(`handoff-10.md` §0.4), so producers (`stage0_chunk.py`,
`backfill_hdbscan.py`) and `CLAIM-C`'s own scorer are not guarded.

A reader refuses by default; `--v1-only` drops held-out per-prompt inputs, and
`--allow-holdout` reads them, which is for after the user releases the set
(`docs/PHASE_REVIEW.md` "Open"). Either way the output record says so.

Torch-free: `core.config` owns `PROMPTS` but imports torch, so the keys are
listed here and `tests/test_holdout.py` checks both sets against `PROMPTS`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

__all__ = [
    "HELD_OUT_PROMPT_KEYS",
    "V1_PROMPT_KEYS",
    "HoldoutError",
    "held_out_reason",
    "refuse_held_out",
    "add_holdout_args",
]

# The twelve v2 additions (core/config.py, "v2 extension, 2026-09-19").
HELD_OUT_PROMPT_KEYS = frozenset({
    "wiki_photosynthesis", "wiki_byzantium",
    "lincoln_letter_short", "lincoln_letter_long",
    "paper_attention", "paper_svflow",
    "quijote_capitulo", "moby_loomings",
    "sklearn_kmeans_code", "scipy_linkage_code",
    "latex_beamer", "latex_article",
})

# v1's nine, which stay readable. Listed so a file named for one of them is
# known to be per-prompt rather than pooled (rule 3).
V1_PROMPT_KEYS = frozenset({
    "short_heterogeneous", "wiki_paragraph", "repeated_tokens",
    "sullivan_ballou", "paper_excerpt", "homer_iliad", "hdbscan_code",
    "camus_letranger", "latex_monograph",
})

HELD_OUT_FILES = frozenset({"claim_c_real_run.json"})
# Stage 0's per-invocation logs print effective rank and spectral k per prompt.
# stage0_index.json (paths only, no values) is the selection source and stays open.
LOG_DIRS = frozenset({"stage0_logs"})
LOG_SUFFIXES = frozenset({".out", ".log"})


def _token_re(keys: frozenset) -> "re.Pattern[str]":
    # Longest first, so the alternation cannot stop at a shorter key.
    alt = "|".join(sorted(map(re.escape, keys), key=len, reverse=True))
    return re.compile(r"(?:^|_)(" + alt + r")(?:_|\.|$)")


_KEY_TOKEN = _token_re(HELD_OUT_PROMPT_KEYS)
_V1_TOKEN = _token_re(V1_PROMPT_KEYS)

PathLike = Union[str, Path]


class HoldoutError(RuntimeError):
    """A Phase 10 reader was handed a held-out input without `allow`."""


def _manifest_key(run_dir: Path) -> Optional[str]:
    try:
        with open(run_dir / "manifest.json") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return None
    pk = man.get("prompt_key") if isinstance(man, dict) else None
    return pk if isinstance(pk, str) else None


def _names_held_out(name: str) -> Optional[str]:
    m = _KEY_TOKEN.search(name)
    return m.group(1) if m else None


def _classify(path: Path) -> Optional[Tuple[str, str]]:
    """(kind, reason) or None. kind "prompt" is one held-out prompt's own
    output, safe to drop whole; kind "pooled" mixes held-out and v1 prompts,
    so dropping it would lose v1 values and reading it would leak."""
    if path.name in HELD_OUT_FILES:
        return "pooled", f"{path.name} carries per-prompt values for the twelve"
    if path.parent.name in LOG_DIRS and path.suffix in LOG_SUFFIXES:
        return "pooled", f"a {path.parent.name} log, which prints held-out values"

    if path.is_dir():
        pk = _manifest_key(path)
        if pk is not None:
            if pk in HELD_OUT_PROMPT_KEYS:
                return "prompt", f"run directory for held-out prompt {pk!r} (manifest)"
            return None       # a v1 run, by its own manifest
        # A directory that pools held-out runs, e.g. a timestamp directory.
        for child in path.iterdir():
            if child.is_dir() and _names_held_out(child.name):
                return "pooled", f"contains held-out run {child.name}"

    key = _names_held_out(path.name)
    if key is not None:
        return "prompt", f"named for held-out prompt {key!r}"

    # A pooled file beside a held-out run: pair_agreement.json and the like.
    # A file named for a v1 prompt is per-prompt, so it is not pooled.
    if path.is_file() and not _V1_TOKEN.search(path.name):
        for sib in path.parent.iterdir():
            if sib.is_dir() and _names_held_out(sib.name):
                return "pooled", f"beside held-out run {sib.name}, so it may pool it"
    return None


def held_out_reason(path: PathLike) -> Optional[str]:
    """Why `path` is held out, or None if it is not."""
    c = _classify(Path(path))
    return c[1] if c else None


def refuse_held_out(
    paths: Iterable[PathLike],
    *,
    allow: bool = False,
    drop: bool = False,
    context: str = "",
) -> Tuple[List[Path], dict]:
    """
    Screen a reader's inputs. Returns `(kept, record)`.

    Default: raise HoldoutError if any input is held out. `drop=True` (a
    reader's `--v1-only`) removes them instead; `allow=True` (its
    `--allow-holdout`, for after the user releases the set) keeps them.
    `record` goes under "holdout" in the reader's output, so a result says
    which input set it ran on.
    """
    if allow and drop:
        raise ValueError("allow and drop are exclusive")
    paths = [Path(p) for p in paths]
    hits = []
    for p in paths:
        c = _classify(p)
        if c is not None:
            hits.append((p, c[0], c[1]))
    who = context or "reader"
    blocking = [h for h in hits if not allow and not (drop and h[1] == "prompt")]
    if blocking:
        shown = "\n".join(f"  {p}: {r}" for p, _, r in blocking[:5])
        more = f"\n  ... and {len(blocking) - 5} more" if len(blocking) > 5 else ""
        pooled = any(k == "pooled" for _, k, _ in blocking)
        raise HoldoutError(
            f"{who}: {len(blocking)} input(s) are Phase 10's held-out "
            f"confirmation set (core/holdout.py):\n{shown}{more}\n"
            + ("A pooled input mixes held-out and v1 prompts, so --v1-only "
               "cannot drop it without losing v1 values: read it per prompt "
               "key, keeping only V1_PROMPT_KEYS. "
               if pooled else "Pass --v1-only to drop them. ")
            + "--allow-holdout reads everything, once the user has released "
            "the set."
        )
    held = {p for p, _, _ in hits}
    kept = [p for p in paths if p not in held] if drop else paths
    if hits:
        verb = "dropped" if drop else "reading"
        print(f"[holdout] {who}: {verb} {len(hits)} held-out input(s)",
              file=sys.stderr, flush=True)
    record = {"allowed": bool(allow), "n_held_out": len(hits),
              "n_dropped": len(hits) if drop else 0,
              "held_out_keys": sorted(HELD_OUT_PROMPT_KEYS)}
    return kept, record


def add_holdout_args(ap) -> None:
    """The two flags every Phase 10 reader takes."""
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--v1-only", action="store_true",
                   help="drop held-out inputs (core/holdout.py) instead of refusing")
    g.add_argument("--allow-holdout", action="store_true",
                   help="read held-out inputs; only once the user has released them")
