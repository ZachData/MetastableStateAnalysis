"""
tools/recompress_tables.py — rewrite existing .npz artifacts compressed.

`InteractionTable.save` and `ParticleTable.save` wrote uncompressed until
2026-09-01. Everything written before that is readable but large, and most of
the size is not measurement:

    model, checkpoint_step   one value repeated once per row
    prompt_key, pair_type    eight values and four
    real_frac, imag_frac     all NaN whenever the rotational channel was not
                             supplied, which p7_motifs/run_7.py documents as
                             its normal case

Measured on the step-54000 sweep table, 19,077,120 edges: 5.49 GB -> 0.35 GB.
Per column the ratio tracks how much of it is structure rather than signal —
layer 693x, checkpoint_step 686x, real_frac/imag_frac 686x, model 295x,
pair_type 174x, target 64x, source 11x — against `weight` at 1.9x, the one
column that is nearly all information.

SAFETY. Each file is rewritten to a temporary path in the SAME directory,
verified by reloading it and comparing every array against the original
(NaN-aware), and only then moved over the original with os.replace, which is
atomic within a filesystem. A file that fails verification is left untouched
and named in the summary. Nothing is deleted before its replacement is
checked, because an artifact this expensive to recompute is not worth a
clever in-place write.

    python3 -m tools.recompress_tables --dry-run /mnt/vm_storage/mets_runs/p7
    python3 -m tools.recompress_tables /mnt/vm_storage/mets_runs/p7
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def is_compressed(path: Path) -> bool:
    """True when every member is deflated. A file with no members is not a
    table and is left alone."""
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            return bool(infos) and all(
                i.compress_type == zipfile.ZIP_DEFLATED for i in infos)
    except zipfile.BadZipFile:
        return False


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        # equal_nan matters here specifically: real_frac and imag_frac are
        # all-NaN in every table this tool will touch, and a rewrite that
        # turned NaN into 0.0 would be exactly the degradation the schema
        # forbids -- "not measured" must stay distinct from "measured zero".
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def recompress(path: Path, dry_run: bool = False) -> dict:
    before = path.stat().st_size
    if is_compressed(path):
        return {"path": path, "status": "already", "before": before,
                "after": before}
    if dry_run:
        return {"path": path, "status": "would", "before": before, "after": None}

    src = np.load(path, allow_pickle=False)
    payload = {k: src[k] for k in src.files}

    tmp = path.with_suffix(".npz.recompress-tmp")
    t0 = time.time()
    try:
        np.savez_compressed(tmp, **payload)
        # savez_compressed appends .npz when the name lacks it; ours has it.
        written = tmp if tmp.exists() else Path(str(tmp) + ".npz")

        check = np.load(written, allow_pickle=False)
        if set(check.files) != set(payload):
            raise ValueError(
                f"rewritten file has keys {sorted(set(check.files))}, "
                f"original had {sorted(payload)}")
        for k in payload:
            if not arrays_equal(check[k], payload[k]):
                raise ValueError(f"array {k!r} differs after rewrite")
        del check
        after = written.stat().st_size
        os.replace(written, path)      # atomic within one filesystem
    except Exception as exc:
        for leftover in (tmp, Path(str(tmp) + ".npz")):
            if leftover.exists():
                leftover.unlink()
        return {"path": path, "status": f"FAILED: {exc}", "before": before,
                "after": None}

    return {"path": path, "status": "done", "before": before, "after": after,
            "seconds": time.time() - t0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path,
                    help="directories to walk, or .npz files")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be rewritten and stop")
    ap.add_argument("--pattern", default="*.npz")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for root in args.roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(root.rglob(args.pattern)))
    if not files:
        print("no .npz files found", file=sys.stderr)
        return 1

    tot_before = tot_after = 0
    failed, done, already = [], 0, 0
    for f in files:
        r = recompress(f, dry_run=args.dry_run)
        tot_before += r["before"]
        tot_after += r["after"] if r["after"] is not None else r["before"]
        name = str(r["path"])
        if r["status"] == "done":
            done += 1
            print(f"  {name}\n     {r['before']/1e9:.2f} GB -> "
                  f"{r['after']/1e9:.2f} GB  "
                  f"({r['before']/max(r['after'],1):.1f}x, {r['seconds']:.0f}s)",
                  flush=True)
        elif r["status"] == "already":
            already += 1
        elif r["status"] == "would":
            print(f"  would rewrite {name} ({r['before']/1e9:.2f} GB)", flush=True)
        else:
            failed.append((name, r["status"]))
            print(f"  {name}\n     {r['status']}", file=sys.stderr, flush=True)

    print(f"\n{len(files)} file(s): {done} rewritten, {already} already "
          f"compressed, {len(failed)} failed")
    if not args.dry_run:
        print(f"total {tot_before/1e9:.1f} GB -> {tot_after/1e9:.1f} GB "
              f"(saved {(tot_before-tot_after)/1e9:.1f} GB)")
    for name, why in failed:
        print(f"  FAILED {name}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
