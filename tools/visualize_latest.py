#!/usr/bin/env python3
"""
tools/visualize_latest.py — draw every phase's catalogue from the newest
results directory that phase actually has.

    python -m tools.visualize_latest --results results/ --out blog_figures/
    python -m tools.visualize_latest --results results/ --dry-run

WHY THIS EXISTS

Each phase ships its own `visualization` package with its own entry point
and its own flag for the input directory (`--results_dir`, `--p1b_dir`,
`--p1c_dir`, `--p2_dir`, `--p2b_dir`), and each of those refuses another
phase's directory rather than drawing nonsense from it. That is the right
behaviour per phase and a nuisance across all of them: a `results/` tree
mixes bare-timestamp Phase 1 roots, `p2_eigenspectra_<ts>/`, and
hand-named pilots like `p1b_pilot/`, and the name alone does not say which
phase wrote which — `run_1` names its directory with a bare timestamp and
`run_1b` will too whenever `--output-dir` is passed by hand.

So the phase is read off the artifacts instead of off the directory name,
using the same marker file each phase's own loader discovers:

    Phase 1    {run}/geometry.json          (loaders.discover_runs)
    Phase 1b   phase1b_*.json               (ARTIFACT_PREFIX glob)
    Phase 1c   p1c.json or {run}/p1c.json
    Phase 2    ov_summary_*.json
    Phase 2b   phase2b_results.json, or an interrupted sweep's
               {stem}/block1a_rotational_spectrum.json

A directory matching nothing is reported as unclassified rather than
guessed at, and the newest match per phase (by mtime) is the one drawn.
Phases with no matching directory are skipped and said so — except Phase
1c, whose `theory` class draws the null model the phase compares against
without needing a run at all, so that much is still offered.

Nothing here plots anything or opens an artifact; it only decides which
directory goes to which entry point, and then calls it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# One row per phase: the module to call, the flag it takes its input
# directory under, the output subdirectory, and the predicate that
# recognises a directory that phase wrote. Keeping this as data means a new
# phase's visualization package is a row rather than another branch, and —
# more importantly — that the marker files are readable in one place next
# to the loaders they mirror.
#
# Order matters: `_classify` returns the first match, so the narrower
# markers come before the broader ones.
PHASES = [
    {
        "key": "p2b",
        "module": "p2b_imaginary.visualization",
        "flag": "--p2b_dir",
        "out": "p2b",
        "marker": lambda d: ((d / "phase2b_results.json").exists()
                             or any(d.glob("*/block1a_rotational_spectrum.json"))),
        "describe": "phase2b_results.json",
    },
    {
        "key": "p2",
        "module": "p2_eigenspectra.visualization",
        "flag": "--p2_dir",
        "out": "p2",
        "marker": lambda d: any(d.glob("ov_summary_*.json")),
        "describe": "ov_summary_*.json",
    },
    {
        "key": "p1b",
        "module": "p1b_hemisphere.visualization",
        "flag": "--p1b_dir",
        "out": "p1b",
        "marker": lambda d: any(d.glob("phase1b_*.json")),
        "describe": "phase1b_*.json",
    },
    {
        "key": "p1c",
        "module": "p1c_frames.visualization",
        "flag": "--p1c_dir",
        "out": "p1c",
        "marker": lambda d: ((d / "p1c.json").exists()
                             or any(d.glob("*/p1c.json"))),
        "describe": "p1c.json",
    },
    {
        "key": "p1",
        "module": "p1_mstate_tracking.visualization",
        "flag": "--results_dir",
        "out": "p1",
        "marker": lambda d: any(d.glob("*/geometry.json")),
        "describe": "{run}/geometry.json",
    },
]

PHASE_KEYS = [p["key"] for p in PHASES]

# Directories that are inputs to a phase's visualization but are not
# themselves a run root: Phase 1's random-seed controls are passed under
# --random_seed_dirs, not --results_dir, and folding them in as a Phase 1
# root would draw the control as if it were the experiment.
RANDOM_DIR_NAMES = ("p1_random",)


def _classify(d: Path) -> Optional[str]:
    """Which phase wrote `d`, by marker file, or None."""
    for phase in PHASES:
        try:
            if phase["marker"](d):
                return phase["key"]
        except OSError:
            return None
    return None


def _scan(results: Path) -> Dict[str, List[Path]]:
    """{phase key: [dirs, newest first]} plus '_unclassified' and '_random'."""
    found: Dict[str, List[Path]] = {k: [] for k in PHASE_KEYS}
    found["_unclassified"] = []
    found["_random"] = []

    for d in sorted(results.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name in RANDOM_DIR_NAMES:
            found["_random"].append(d)
            continue
        key = _classify(d)
        found[key if key else "_unclassified"].append(d)

    for key in PHASE_KEYS:
        found[key].sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return found


def _commands(found: Dict[str, List[Path]], out: Path,
              wanted: List[str]) -> List[List[str]]:
    """The argv for each phase that has something to draw."""
    cmds: List[List[str]] = []
    for phase in PHASES:
        key = phase["key"]
        if key not in wanted:
            continue
        dirs = found[key]
        if not dirs:
            if key == "p1c":
                # The theory class needs no run directory — it draws the
                # phase's own null model — so it is still worth offering
                # when no Phase 1c sweep has landed.
                cmds.append([sys.executable, "-m", phase["module"],
                             "--classes", "theory",
                             "--out", str(out / phase["out"])])
            continue
        argv = [sys.executable, "-m", phase["module"],
                phase["flag"], str(dirs[0]),
                "--out", str(out / phase["out"])]
        if key == "p1" and found["_random"]:
            argv += ["--random_seed_dirs"] + [str(p) for p in found["_random"]]
        cmds.append(argv)
    return cmds


def _row(key: str, describe: str, text: str) -> str:
    """One scan line, with the directory column aligned across phases."""
    marker = f"[{describe}]" if describe else ""
    return f"  {key:<4} {marker:<34} {text}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run every phase's visualization package against the "
                    "newest results directory that phase wrote.")
    ap.add_argument("--results", type=Path, default=Path("results"),
                    help="Results root holding one directory per run "
                         "(default: results).")
    ap.add_argument("--out", type=Path, default=Path("blog_figures"),
                    help="Figure root; each phase writes {out}/{phase}/ "
                         "(default: blog_figures).")
    ap.add_argument("--phases", nargs="*", default=None, choices=PHASE_KEYS,
                    help="Limit to these phases (default: all found).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands and the directory each would "
                         "read, without drawing anything.")
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="Print what each results directory was classified "
                         "as, then exit.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Carry on after a phase exits non-zero instead of "
                         "stopping. One phase's missing data does not "
                         "invalidate another's figures.")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"ERROR: results dir not found: {args.results}", file=sys.stderr)
        sys.exit(1)

    found = _scan(args.results)

    print(f"Scanning {args.results}")
    for phase in PHASES:
        dirs = found[phase["key"]]
        print(_row(phase["key"], phase["describe"],
                   f"{dirs[0].name}   <- newest, will be drawn"
                   if dirs else "none"))
        for extra in dirs[1:]:
            print(_row("", "", extra.name))
    for d in found["_random"]:
        print(_row("p1", "random control", d.name))
    for d in found["_unclassified"]:
        print(_row("--", "no phase marker", d.name))
    print()

    if args.list_only:
        return

    wanted = args.phases if args.phases else PHASE_KEYS
    cmds = _commands(found, args.out, wanted)
    if not cmds:
        print("Nothing to draw.", file=sys.stderr)
        sys.exit(1)

    failures = []
    for argv in cmds:
        print("$ " + " ".join(shlex.quote(a) for a in argv))
        if args.dry_run:
            continue
        rc = subprocess.call(argv)
        if rc != 0:
            failures.append((argv[3], rc))
            print(f"  ^ exited {rc}", file=sys.stderr)
            if not args.keep_going:
                sys.exit(rc)
        print()

    if args.dry_run:
        return
    if failures:
        print(f"\n{len(failures)} phase(s) failed:", file=sys.stderr)
        for module, rc in failures:
            print(f"  {module}: exit {rc}", file=sys.stderr)
        sys.exit(1)
    print(f"Done. Figures under {args.out.resolve()}")


if __name__ == "__main__":
    main()
