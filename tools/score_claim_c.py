#!/usr/bin/env python3
"""
tools/score_claim_c.py — run CLAIM-C's replication gate on real Phase 1 run
directories, and write the record whether it produced a p-value or refused.

CLAIM-C is the one registered prediction with a hard stop: if the Blog-1
trained-minus-random contrast does not transfer from gpt2-large to
pythia-1.4b, no checkpoint-sweep work was to proceed. The gate
(`p1_mstate_tracking/replication_gate.py`) was built and calibrated on
2026-08-24/25 and never pointed at a real run directory, because nothing
assembled its five arms from Phase 1 artifacts. This is that plumbing and
nothing more: no statistic, tolerance or prompt set is chosen here.

The five arms are MODULE CONSTANTS because the registered statement names
them. Changing an arm is a registry amendment, not a flag.

Layout expected, the one `run_1.py` writes:

    <run_dir>/<model>_<prompt>/{geometry,clustering,sinkhorn}.json

Several run directories may be given; each arm is taken from the first one
that has it. `repeated_tokens` is a collapse control, not a metastability
prompt, and is never an exchangeable unit here.

A record is written to `claims/audits/claim_c_real_run.json` by default,
including refusals — a gate that refuses on real data has still been run on
real data, and the P-I1 scorer's lesson (a p-value that lived only in prose
and a git-ignored series) is not repeated. `claims/adjudications/` is untouched
unless `--adjudicate` is passed, which is the author's call.

Usage
-----
    python3 -m tools.score_claim_c --run-dir data/phase12/<run> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(os.environ.get("METS_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

from p1_mstate_tracking.replication_gate import (   # noqa: E402
    CLAIM_C_METRICS,
    adjudicate_claim_c,
    profiles_from_run_dir,
)

RECORD_PATH = REPO / "claims" / "audits" / "claim_c_real_run.json"

#: The arms, as the registered statement names them. Reference = gpt2-large
#: (Blog 1's phenomenology); candidate = pythia-1.4b, with the norm-matched
#: random baseline carrying the p-value and the published step-0 checkpoint as
#: the mandatory sensitivity arm (registry, CLAIM-C null_construction item 2).
REFERENCE_TRAINED = "gpt2-large"
REFERENCE_RANDOM = "gpt2-large-random"
CANDIDATE_TRAINED = "pythia-1.4b-step143000"
CANDIDATE_RANDOM = "pythia-1.4b-random"
CANDIDATE_STEP0 = "pythia-1.4b-step0"
ARMS = (REFERENCE_TRAINED, REFERENCE_RANDOM, CANDIDATE_TRAINED,
        CANDIDATE_RANDOM, CANDIDATE_STEP0)

#: The eight metastability prompts. `repeated_tokens` is the collapse control
#: and is excluded by `run_1.py` from every metastability analysis (P1-2).
CONTROL_PROMPTS = ("repeated_tokens",)

ARTIFACT_FILES = ("geometry.json", "clustering.json", "sinkhorn.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metastability_prompts() -> List[str]:
    from core.config import PROMPTS
    return [p for p in PROMPTS if p not in CONTROL_PROMPTS]


def discover_arm(run_dirs: Sequence[Path], model: str,
                 prompts: Sequence[str]) -> Tuple[Dict[str, dict], List[Path], str]:
    """
    `{prompt: {metric: series}}` for one arm, the artifact files it read, and
    the run directory it came from ('' when the arm is absent everywhere).

    An arm is taken whole from the first run directory holding ANY of its
    prompt directories; prompts are not stitched across run directories,
    because a run directory is one model load and one seed.
    """
    for rd in run_dirs:
        found = {p: rd / f"{model}_{p}" for p in prompts}
        present = {p: d for p, d in found.items() if d.is_dir()}
        if not present:
            continue
        profiles: Dict[str, dict] = {}
        files: List[Path] = []
        for p, d in present.items():
            prof = profiles_from_run_dir(d)
            if prof:
                profiles[p] = prof
                files.extend(d / f for f in ARTIFACT_FILES if (d / f).exists())
        return profiles, files, str(rd)
    return {}, [], ""


def assemble(run_dirs: Sequence[Path]) -> dict:
    prompts = metastability_prompts()
    arms = {}
    files: List[Path] = []
    missing = []
    for model in ARMS:
        prof, arm_files, src = discover_arm(run_dirs, model, prompts)
        arms[model] = prof
        files.extend(arm_files)
        if not prof:
            missing.append(model)
    return {"prompts": prompts, "arms": arms, "files": files, "missing": missing}


def run(run_dirs: Sequence[Path], *, adjudicate: bool = False,
        n_perm: Optional[int] = None, seed: int = 0) -> dict:
    found = assemble(run_dirs)
    record = {
        "prediction": "CLAIM-C",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dirs": [str(rd) for rd in run_dirs],
        "arms": {m: {"n_prompts": len(found["arms"][m]),
                     "prompts": sorted(found["arms"][m])} for m in ARMS},
        "metrics": list(CLAIM_C_METRICS),
        "artifact_hashes": sorted({_sha256(f) for f in found["files"]}),
        "adjudicate": adjudicate,
    }
    required_missing = [m for m in found["missing"] if m != CANDIDATE_STEP0]
    if required_missing:
        record["refused"] = (
            f"arm(s) absent from every run directory given: {required_missing}. "
            f"The gate needs Phase 1 runs of all of {list(ARMS[:4])} on the "
            f"metastability prompts; nothing here substitutes one model for "
            f"another.")
        record["result"] = None
        return record

    step0 = found["arms"][CANDIDATE_STEP0] or None
    kwargs = dict(
        candidate_step0=step0,
        step0_absent_reason=(None if step0 else
                             f"{CANDIDATE_STEP0} absent from every run directory"),
        seed=seed,
        artifact_hashes=record["artifact_hashes"],
        run_manifest={"run_dirs": record["run_dirs"], "arms": record["arms"]},
        adjudicate=adjudicate,
    )
    if n_perm is not None:
        kwargs["n_perm"] = n_perm
    res = adjudicate_claim_c(
        found["arms"][REFERENCE_TRAINED], found["arms"][REFERENCE_RANDOM],
        found["arms"][CANDIDATE_TRAINED], found["arms"][CANDIDATE_RANDOM],
        **kwargs)
    record["refused"] = None
    record["result"] = res
    return record


def _print(record: dict) -> None:
    print("CLAIM-C replication gate on real Phase 1 artifacts\n")
    for m, a in record["arms"].items():
        print(f"  {m:<26} {a['n_prompts']} prompt(s)")
    print(f"  {len(record['artifact_hashes'])} artifact file(s) hashed\n")
    if record["refused"]:
        print(f"REFUSED: {record['refused']}")
        return
    res = record["result"]
    for k in ("p_value", "p_reciprocal", "verdict", "hard_stop", "reason",
              "n_prompts", "prompts_dropped", "best_attainable_p",
              "sign_homogeneity", "binding_subset"):
        if k in res:
            print(f"  {k:<20} = {res[k]}")
    s0 = res.get("step0_sensitivity") or {}
    if s0:
        print(f"  step0 arm            = "
              f"{'reported' if s0.get('available') else 'absent'}"
              f"{' — DISAGREES with primary' if s0.get('disagrees_with_primary') else ''}")
    if res.get("adjudication"):
        print(f"\nADJUDICATED: {res['adjudication']}")
    else:
        print("\nNOT adjudicated: claims/adjudications/ is untouched. "
              "--adjudicate is the author's call.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True, type=Path,
                    help="a Phase 1 run directory; repeatable")
    ap.add_argument("--adjudicate", action="store_true",
                    help="write the e-value into claims/adjudications/ (author only)")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write claims/audits/claim_c_real_run.json")
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    for rd in args.run_dir:
        if not rd.is_dir():
            print(f"{rd} is not a directory")
            return 1

    record = run(args.run_dir, adjudicate=args.adjudicate,
                 n_perm=args.n_perm, seed=args.seed)
    _print(record)
    if not args.no_write:
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_PATH.write_text(json.dumps(record, indent=2, default=str) + "\n",
                               encoding="utf-8")
        print(f"\nrecord: {RECORD_PATH.relative_to(REPO)}")
    return 0 if not record["refused"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
