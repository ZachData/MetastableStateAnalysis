"""R9's flags, shared by the re-read ladders (`p10_cluster_function/design-10.md` "R9").

Every ladder that re-reads a row on the c3x source takes the same three things: ``--lead c3x``
(the primary column), ``--reproduce <that row's record> [...] --reproduce-labels <its source>``
(R9's first check: c0–c3 read on the new source equal the row's stored record, so a changed
label is the column and not the run), and the lead's step-0 floor in its output. The ladders
keep their own ``load`` and ``reproduce`` (each compares a different record shape); this module
holds the flags, the "go together" refusal, the refusal on a mismatch, and the output header.
Built at R9 / R3, the fourth copy (`/challenge-pr` on #170, finding 6).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

# `p10_r1_ladder` imports this module, so its LEADS and LadderError are imported where used.


def add_args(ap: argparse.ArgumentParser, row: str, also: Sequence[Tuple[str, str]] = ()) -> None:
    """``--lead``, ``--reproduce`` (metavar ``<row>_RECORD``), each ``also`` (flag, help) input
    the reproduced run read, and ``--reproduce-labels``."""
    from tools.run.p10_r1_ladder import LEADS
    flags = ", ".join([f for f, _ in also] + ["--reproduce-labels"])
    ap.add_argument("--lead", choices=tuple(LEADS), default="c3", help="the primary column (R9: c3x)")
    ap.add_argument("--reproduce", type=Path, default=None, metavar=f"{row.upper()}_RECORD",
                    help=f"refuse unless c0–c3's records equal that {row} run's (with {flags})")
    for flag, help_ in also:
        ap.add_argument(flag, type=Path, default=None, help=help_)
    ap.add_argument("--reproduce-labels", type=Path, default=None, help="the label source that run read")


def reproduce_inputs(args: argparse.Namespace, also: Sequence[str] = ()) -> Optional[tuple]:
    """``(record, *also, labels)`` when ``--reproduce`` is given; ``None`` when none of them is;
    refuses (``SystemExit``) when only some are: the run it reproduces read its own inputs."""
    names = ("--reproduce", *also, "--reproduce-labels")
    vals = tuple(getattr(args, n.lstrip("-").replace("-", "_")) for n in names)
    given = [v is not None for v in vals]
    if not any(given):
        return None
    if not all(given):
        raise SystemExit(f"refusing: {', '.join(names[:-1])} and {names[-1]} go together (the run it "
                         "reproduces read its own inputs)")
    return vals


def check_reproduces(args: argparse.Namespace, data: Dict, load: Callable[..., Dict],
                     reproduce: Callable[[Dict, Dict], list], what: str,
                     also: Sequence[str] = ()) -> Optional[str]:
    """Refuse (``LadderError``) unless ``reproduce(data, load(*inputs))`` is empty; the reproduced
    record's path, or ``None`` without ``--reproduce``. ``load`` reads with the lead c3."""
    from tools.run.p10_r1_ladder import LadderError
    inputs = reproduce_inputs(args, also)
    if inputs is None:
        return None
    bad = reproduce(data, load(*inputs))
    if bad:
        raise LadderError(f"refusing: records differ from {inputs[0]}: {bad}")
    print(f"reproduces {inputs[0]}: every c0–c3 record equal ({what})")
    return str(inputs[0])


def floor(summary: Dict, lead: str) -> Dict:
    """The lead's step-0 count from the label source's ``summary.json`` (the floor)."""
    f = summary["step0"]["columns"][lead]
    return {f"{lead}_group_layer_records_step0": f["groups"], f"{lead}_readable_step0": [f["readable"], f["records"]]}


def floor_line(prim: Dict[int, str], summary: Dict, lead: str) -> str:
    f = summary["step0"]["columns"][lead]
    return (f"primary: c2 at steps {[s for s, c in prim.items() if c == 'c2']}, {lead} elsewhere; floor "
            f"{f['groups']} {lead} records at step 0 ({f['readable']} of {f['records']} readable)")


def header(labels: Path, sha: str, lead: str, reproduces: Optional[str], prim: Dict[int, str],
           summary: Dict) -> Dict:
    """The fields every R9 ladder output opens with."""
    return {"label_source": str(labels), "summary_sha256": sha, "lead": lead, "reproduces": reproduces,
            "primary": prim, "floor": floor(summary, lead)}
