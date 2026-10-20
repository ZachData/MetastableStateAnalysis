"""
trackA_force/a5_record.py — `P-A5`'s real-run record, tracked (A5 unit 3).

Reads the `score.json` that `python -m trackA_force.a5_seeds score` wrote over
seeds 1–9 and writes `claims/audits/p_a5_real_run.json`: the score verbatim,
SHA-256 of every record it was computed from (per seed, and one over all), and
the per-seed margins of V against the half-plateau line at steps 128, 256 and
512, so a reader can see how far each seed sits from the window's edges.

Kept out of `a5_seeds.py` so the producer's code hash (`7ae0bff62ada`, named in
the registry entry) does not change after registration. It computes no p-value;
the p and E it carries are `score`'s.

    METS_DATA=<main>/data python -m trackA_force.a5_record --out <main>/data/trackA/a5/2026-10-20-u2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from .a1a import DATA, REPO, STEPS
from .a5_seeds import SEEDS, code_sha

RECORD = "claims/audits/p_a5_real_run.json"
EDGES = (128, 256, 512)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_hashes(d: Path) -> dict:
    """SHA-256 per record file, and one over the sorted (name, hash) lines."""
    files = sorted((d / "records").glob("*.json"))
    if len(files) != len(STEPS) * 7:
        raise SystemExit(f"refusing: {d} holds {len(files)} records, not {len(STEPS) * 7}")
    lines = [f"{f.name} {sha256(f)}" for f in files]
    return {"n_records": len(files), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
            "plan_sha256": sha256(d / "plan.json")}


def margins(score: dict) -> dict:
    out = {}
    for seed, note in score["notes"].items():
        half = score["per_seed"][seed]["plateau_median"] / 2
        out[seed] = {"half": half, **{f"V{s}": note["V"][STEPS.index(s)] for s in EDGES}}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P-A5's real-run record")
    ap.add_argument("--out", required=True, help="the directory holding score.json and seed1 … seed9")
    a = ap.parse_args(argv)
    root = Path(a.out)
    score = json.loads((root / "score.json").read_text())
    if score["code"] != code_sha() or score["input_set"]["record_codes"] != [code_sha()]:
        raise SystemExit(f"refusing: score or records not at code {code_sha()}")
    per = {str(s): seed_hashes(root / f"seed{s}") for s in SEEDS[1:]}
    rec = {"prediction": "P-A5", "producer": "trackA_force.a5_seeds score", "code": score["code"],
           "records": str(root.relative_to(DATA)) if root.is_relative_to(DATA) else str(root),
           "score_sha256": sha256(root / "score.json"),
           "artifact_hashes": sorted(v["sha256"] for v in per.values()),
           "per_seed_hashes": per, "margins": margins(score), "score": score,
           "adjudicated": False}
    (REPO / RECORD).write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {RECORD}: K {score['K']}, p {score['p']:.3g}, E {score['e']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
