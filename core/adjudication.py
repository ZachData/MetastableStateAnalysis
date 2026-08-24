"""
core/adjudication.py — the emission layer (POPPER_PLAN.md item B4).

The thin layer between a science module that produced a p-value and the
falsification ledger in `claims/`. It does four things and refuses to do a
fifth:

  1. look the prediction up in `claims/registry.json`
  2. refuse if the prediction may not carry an e-value
  3. calibrate p -> e and append to its claim's e-process
  4. write one `claims/adjudications/<id>.json` record

What it will not do is emit a number when the inputs do not support one. Three
refusals, all instances of the project's standing rule 4 ("refuse rather than
degrade" -- `UPDATE_PLAN.md` §6):

**No registry entry -> refuse.** An unregistered prediction has no recorded
null, no evaluability classification, and no pre-registration timestamp, so its
Assumption-2 status is unknown. An e-value computed from it looks identical in
the artifact to one computed from a properly registered prediction.

**`evaluable != "e-value"` -> refuse.** This is the load-bearing one and it is
worth being precise about why, because the intuition that a doubtful e-value is
"just one weak piece of evidence" is exactly backwards. E-values multiply. A
factor derived from a null that is not valid under H0 does not merely contribute
noise -- it voids `E[E] <= 1` for the whole product, so *every other prediction
on that claim* loses its guarantee too. POPPER measures the size of that effect
directly: removing the relevance checker, whose only job is keeping such nulls
out, raises Type-I error from 0.082 to 0.340 on TargetVal-IL2.

So a prediction classified `measurement` or `needs-null` in
`claims/EVALUABILITY.md` cannot be adjudicated here, and the right response to
"but we have a number for it" is to record the number as a measurement in the
phase's own artifact, not to route it through this module.

**`status == "dormant"` -> refuse.** The prediction's instrument was archived,
so nothing live can produce its p-value. This is deliberately a *status* and not
a deletion: the prediction was pre-registered, its falsifier is unchanged, and it
has not been withdrawn. Deleting a pre-registered prediction because its
apparatus went away is precisely the selective-record problem the gate exists to
prevent -- it would let the surviving record be the flattering subset. Dormant
keeps it visible and uncounted, and reverses if the instrument is rebuilt.

Separation of concerns
----------------------
`core/evalues.py` owns the arithmetic and knows nothing about predictions.
This module owns the bookkeeping and does no arithmetic beyond calling that
one. `tools/check_registry.py` owns the pre-registration gate and reads what
this module writes. Keeping the three apart is what lets the arithmetic stay
small enough to be obviously correct.

Dependencies: stdlib only (plus `core.evalues`, itself stdlib-only at
import time). This runs
in CI tier 0 with no `pip install` step, and imports without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from core.evalues import DEFAULT_ALPHA, DEFAULT_KAPPA, EProcess, calibrate

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "claims" / "registry.json"
ADJUDICATIONS_DIR = PROJECT_ROOT / "claims" / "adjudications"

#: Record-schema version. Bump when a field's meaning changes, so a reader can
#: tell a v1 record from a v2 one rather than inferring it from which keys are
#: present.
RECORD_SCHEMA_VERSION = 1


class AdjudicationRefused(Exception):
    """
    Raised when a prediction may not be adjudicated.

    Deliberately not a subclass of ValueError: a caller writing
    `except ValueError` around a numeric pipeline should not accidentally
    swallow a refusal, because a swallowed refusal is indistinguishable from
    an adjudication that never happened.
    """


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegistryEntry:
    """One prediction as the registry declares it."""
    id: str
    claim: str
    evaluable: str
    relevance: float
    statement: str
    h0: str
    null_construction: str
    status: str
    dormant_reason: str
    raw: dict


def load_registry(path: Optional[Path] = None) -> dict:
    path = Path(path) if path is not None else REGISTRY_PATH
    if not path.exists():
        raise AdjudicationRefused(f"registry not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def registry_entry(prediction_id: str, registry: Optional[dict] = None) -> RegistryEntry:
    reg = registry if registry is not None else load_registry()
    for p in reg.get("predictions", []):
        if p.get("id") == prediction_id:
            return RegistryEntry(
                id=p["id"], claim=p["claim"], evaluable=p["evaluable"],
                relevance=float(p["relevance"]), statement=p["statement"],
                h0=p["h0"], null_construction=p.get("null_construction", ""),
                status=p.get("status", "active"),
                dormant_reason=p.get("dormant_reason", ""),
                raw=p,
            )
    raise AdjudicationRefused(
        f"{prediction_id!r} has no entry in claims/registry.json. An unregistered "
        f"prediction cannot be adjudicated: its null, its evaluability and its "
        f"pre-registration timestamp are all unknown, and an e-value computed from "
        f"it is indistinguishable in the artifact from a valid one."
    )


# ---------------------------------------------------------------------------
# Artifact identity
# ---------------------------------------------------------------------------

def hash_artifact(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    SHA-256 of a file, as the identity an adjudication records for its inputs.

    Recorded rather than the path, because a path says which file was read and
    a hash says which *bytes* were. The gate's third rule -- an adjudication may
    not consume an artifact that a later-registered prediction also consumes --
    is a statement about bytes: re-using an artifact is fine, but registering a
    prediction after seeing that artifact and then testing it on the same one is
    the conditional-validity violation, and only the hash makes it visible.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def hash_artifacts(paths: Iterable[Path]) -> List[str]:
    return [hash_artifact(Path(p)) for p in paths]


# ---------------------------------------------------------------------------
# Reading the ledger
# ---------------------------------------------------------------------------

def _record_path(prediction_id: str, adjudications_dir: Optional[Path] = None) -> Path:
    d = Path(adjudications_dir) if adjudications_dir is not None else ADJUDICATIONS_DIR
    return d / f"{prediction_id}.json"


def load_adjudications(adjudications_dir: Optional[Path] = None) -> List[dict]:
    """Every adjudication record, ordered by when it was adjudicated."""
    d = Path(adjudications_dir) if adjudications_dir is not None else ADJUDICATIONS_DIR
    if not d.is_dir():
        return []
    records = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))]
    # Order by adjudication time, not filename. An e-process is a sequence, and
    # the interim E after each step is what optional stopping is valid against;
    # alphabetical order would reproduce the right final product with the wrong
    # trajectory, which is the part a reader actually looks at.
    #
    # `claim_sequence_index` breaks ties before the prediction id does, and that
    # ordering matters more than it looks: two adjudications in the same batch
    # can share a timestamp, at which point an id tiebreak alone silently sorts
    # them alphabetically. The sequence index is assigned at write time from the
    # claim's own length, so it is monotonic within a claim regardless of clock
    # resolution. (Timestamps carry microseconds for the same reason -- see
    # `adjudicate` -- but a resolution argument is a weaker guarantee than a
    # counter, so both are here.)
    return sorted(records, key=lambda r: (r.get("adjudicated_at", ""),
                                          r.get("claim_sequence_index", 0),
                                          r.get("prediction_id", "")))


def claim_process(
    claim: str,
    adjudications_dir: Optional[Path] = None,
    registry: Optional[dict] = None,
) -> EProcess:
    """
    Rebuild one claim's e-process from the committed records.

    Recalibrates each p-value rather than trusting the stored e-value, so a
    record edited by hand does not survive the reload. That is what makes CI's
    recomputation check (item B7) meaningful rather than a tautology.
    """
    reg = registry if registry is not None else load_registry()
    alpha = float(reg.get("alpha", DEFAULT_ALPHA))
    proc = EProcess(claim=claim, alpha=alpha)
    for rec in load_adjudications(adjudications_dir):
        if rec.get("claim") != claim:
            continue
        proc.add(
            prediction_id=rec["prediction_id"],
            p_value=float(rec["p_value"]),
            kappa=float(rec.get("kappa", DEFAULT_KAPPA)),
        )
    return proc


def all_claim_processes(
    adjudications_dir: Optional[Path] = None,
    registry: Optional[dict] = None,
) -> Dict[str, EProcess]:
    reg = registry if registry is not None else load_registry()
    claims = sorted({p["claim"] for p in reg.get("predictions", [])})
    return {c: claim_process(c, adjudications_dir, reg) for c in claims}


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def adjudicate(
    prediction_id: str,
    p_value: float,
    artifact_hashes: Sequence[str],
    run_manifest: Optional[dict] = None,
    *,
    test_name: str = "",
    notes: str = "",
    registry: Optional[dict] = None,
    adjudications_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    """
    Adjudicate one registered prediction and write its record.

    Parameters
    ----------
    prediction_id : str
        Must exist in `claims/registry.json` and be classified `e-value`.
    p_value : float
        Valid under the prediction's H0. Validity is this function's
        precondition and cannot be checked here -- it is established by the
        null construction the registry names, which is why that field is
        required for `e-value` entries.
    artifact_hashes : sequence of str
        SHA-256 of every input the p-value was computed from. Empty is allowed
        only for a synthetic or self-contained test, and is recorded as such;
        for a real run it is what gate rule 3 reads.
    run_manifest : dict, optional
        The run's own manifest (`core.io.write_manifest` output or equivalent).
        Stored verbatim so the record is self-describing rather than pointing
        at a directory that may be regenerated.
    test_name : str
        What statistical test produced `p_value` (e.g. "mannwhitneyu, one-sided
        greater"). Free text, recorded because "which test" is the first
        question a reader has and reconstructing it from the phase code later
        is guesswork.
    dry_run : bool
        Compute and return the record without writing it. For previewing the
        effect on a claim's E before committing to it.

    Raises
    ------
    AdjudicationRefused
        If the prediction is unregistered, is not classified `e-value`, sits
        below the registry's relevance floor, or has already been adjudicated.

    Returns
    -------
    dict
        The record, as written.
    """
    reg = registry if registry is not None else load_registry()
    entry = registry_entry(prediction_id, reg)

    kappa = float(reg.get("kappa", DEFAULT_KAPPA))
    alpha = float(reg.get("alpha", DEFAULT_ALPHA))
    r0 = float(reg.get("relevance_threshold", 0.6))

    # --- refusals ---------------------------------------------------------
    if entry.evaluable != "e-value":
        raise AdjudicationRefused(
            f"{prediction_id} is classified {entry.evaluable!r} in the registry, not "
            f"'e-value', so it may not contribute to claim {entry.claim}.\n"
            f"  reason on record: {entry.null_construction}\n"
            f"E-values multiply, so admitting a factor whose null is not valid under H0 "
            f"voids E[E] <= 1 for every other prediction on this claim, not just for this "
            f"one. Record the number as a measurement in the phase's own artifact, or "
            f"construct the null first and re-classify the entry (claims/EVALUABILITY.md)."
        )

    if entry.status == "dormant":
        raise AdjudicationRefused(
            f"{prediction_id} is dormant: no live instrument can currently adjudicate it.\n"
            f"  reason on record: {entry.dormant_reason}\n"
            f"The prediction stands and has NOT been withdrawn -- its falsifier is unchanged "
            f"and it is still counted in the registry. What it cannot do is contribute to "
            f"{entry.claim}'s e-process, because the number would have to come from somewhere "
            f"other than the instrument it was registered against. Reviving it means reviving "
            f"the instrument, not relaxing this check."
        )

    if entry.relevance < r0:
        raise AdjudicationRefused(
            f"{prediction_id} has relevance {entry.relevance} below the registry's "
            f"floor r0={r0}. Below that the sub-hypothesis is not strongly enough implied "
            f"by its main hypothesis for a falsification to count as evidence for it -- "
            f"which is the inflation POPPER's relevance checker exists to prevent."
        )

    existing = _record_path(prediction_id, adjudications_dir)
    if existing.exists():
        raise AdjudicationRefused(
            f"{prediction_id} has already been adjudicated ({existing.name}). "
            f"Re-adjudicating against new data is a NEW prediction with its own registry "
            f"entry and its own pre-registration timestamp -- overwriting this record "
            f"would let one experiment's evidence be replaced without any trace that it "
            f"had been."
        )

    # --- emission ---------------------------------------------------------
    e_value = calibrate(p_value, kappa)

    # Rebuild the claim's process, then append this one, so the record carries
    # the E *after* it rather than in isolation.
    proc = claim_process(entry.claim, adjudications_dir, reg)
    proc.add(prediction_id=prediction_id, p_value=float(p_value), kappa=kappa)

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "prediction_id": prediction_id,
        "claim": entry.claim,
        "statement": entry.statement,
        "h0": entry.h0,
        "test_name": test_name,
        "p_value": float(p_value),
        "e_value": e_value,
        "kappa": kappa,
        "alpha": alpha,
        "artifact_hashes": list(artifact_hashes),
        "run_manifest": run_manifest or {},
        # Microseconds, not seconds: a batch adjudicating several predictions
        # writes them within the same second, and a second-resolution stamp
        # makes their order depend on the tiebreak rather than on what happened.
        "adjudicated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "claim_sequence_index": len(proc.adjudications),
        "claim_log_E_after": proc.log_E,
        "claim_E_after": proc.E,
        "claim_decision_after": proc.decision(),
        "next_p_needed": proc.next_p_needed(kappa=kappa),
        "notes": notes,
    }

    if not artifact_hashes:
        record["notes"] = (
            (notes + " " if notes else "")
            + "NO ARTIFACT HASHES RECORDED: gate rule 3 (artifact reuse by a "
              "later-registered prediction) cannot be checked for this adjudication."
        ).strip()

    if not dry_run:
        d = Path(adjudications_dir) if adjudications_dir is not None else ADJUDICATIONS_DIR
        d.mkdir(parents=True, exist_ok=True)
        _record_path(prediction_id, d).write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return record


def adjudicate_if_registered(
    prediction_id: str,
    p_value: Optional[float],
    artifact_hashes: Sequence[str] = (),
    **kwargs,
) -> Optional[dict]:
    """
    Adjudicate, or return None with a reason on stderr, without raising.

    For call sites inside a phase runner, where a refusal should not abort a
    run that is otherwise producing valid science. The refusal is *printed*
    rather than swallowed: a phase whose adjudication was refused should say so
    in its own log, since the alternative is a run that silently produced no
    ledger entry and looks identical to one that was never asked to.

    A `None` p_value returns None without complaint -- that is the shape
    `compare_induction_vs_semantic` and friends already use for "the test could
    not be run" (too few heads, missing prerequisite), which is not a refusal.
    """
    if p_value is None:
        return None
    try:
        return adjudicate(prediction_id, p_value, artifact_hashes, **kwargs)
    except AdjudicationRefused as exc:
        import sys
        print(f"[adjudication refused] {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Verification (seed of item B7)
# ---------------------------------------------------------------------------

def verify_ledger(
    adjudications_dir: Optional[Path] = None,
    registry: Optional[dict] = None,
) -> List[str]:
    """
    Recompute every claim's E from the committed records and report
    disagreements. Returns a list of problem strings; empty means consistent.

    This is what makes the ledger self-verifying rather than trusted. It
    catches two things worth catching: arithmetic that drifted, and -- more
    usefully -- a `claim_decision_after` that was updated by hand without its
    evidence.
    """
    reg = registry if registry is not None else load_registry()
    problems: List[str] = []
    records = load_adjudications(adjudications_dir)
    by_id = {p["id"]: p for p in reg.get("predictions", [])}

    for rec in records:
        pid = rec.get("prediction_id")
        entry = by_id.get(pid)
        if entry is None:
            problems.append(f"{pid}: adjudicated but not in the registry")
            continue
        if entry["evaluable"] != "e-value":
            problems.append(
                f"{pid}: adjudicated while classified {entry['evaluable']!r}; "
                f"this record must not contribute to {entry['claim']}"
            )
        if entry.get("status") == "dormant":
            problems.append(
                f"{pid}: adjudicated while dormant (instrument archived); "
                f"this record must not contribute to {entry['claim']}"
            )
        expected_e = calibrate(float(rec["p_value"]), float(rec.get("kappa", DEFAULT_KAPPA)))
        if not _close(expected_e, float(rec["e_value"])):
            problems.append(
                f"{pid}: stored e_value {rec['e_value']} != calibrate(p={rec['p_value']}) "
                f"= {expected_e}; the record has been edited"
            )

    # Replay each claim and compare against the E each record claimed at its
    # own step, which is the check that catches a reordered or deleted record.
    for claim in sorted({r.get("claim") for r in records if r.get("claim")}):
        proc = EProcess(claim=claim, alpha=float(reg.get("alpha", DEFAULT_ALPHA)))
        for rec in [r for r in records if r.get("claim") == claim]:
            proc.add(rec["prediction_id"], float(rec["p_value"]),
                     float(rec.get("kappa", DEFAULT_KAPPA)))
            if not _close(proc.log_E, float(rec["claim_log_E_after"])):
                problems.append(
                    f"{rec['prediction_id']}: claim_log_E_after {rec['claim_log_E_after']} "
                    f"does not match the replayed {proc.log_E}; records may have been "
                    f"reordered, deleted, or edited"
                )
            if proc.decision() != rec.get("claim_decision_after"):
                problems.append(
                    f"{rec['prediction_id']}: claim_decision_after "
                    f"{rec.get('claim_decision_after')!r} does not match the replayed "
                    f"{proc.decision()!r}"
                )
    return problems


def _close(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    import math
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or verify the falsification ledger.")
    ap.add_argument("--verify", action="store_true",
                    help="recompute every claim's E from the records and report disagreements")
    ap.add_argument("--status", action="store_true", help="print each claim's e-process")
    args = ap.parse_args(argv)

    if args.verify:
        problems = verify_ledger()
        for p in problems:
            print(f"ERROR   {p}")
        print(f"\nledger: {len(problems)} problem(s) across "
              f"{len(load_adjudications())} adjudication(s)")
        return 1 if problems else 0

    procs = all_claim_processes()
    print(f"{'claim':<12} {'n':>3} {'log E':>10} {'E':>12}  decision")
    print("-" * 60)
    for claim, proc in procs.items():
        n = len(proc.adjudications)
        print(f"{claim:<12} {n:>3} {proc.log_E:>10.4f} {proc.E:>12.4g}  "
              f"{proc.decision() if n else '-- not adjudicated --'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
