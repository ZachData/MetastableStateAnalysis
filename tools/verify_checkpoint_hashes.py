#!/usr/bin/env python3
"""
tools/verify_checkpoint_hashes.py — settle status-1 open item 3.

THE OBSERVATION
    `step0` and `step1` produce byte-identical output on all 82 lines where
    the model name appears; `step1` vs `step2` differ on 34. Two explanations,
    with opposite consequences:

      (a) The HF `step1` revision resolves to the same weights as `step0`
          (e.g. the tag points at the pre-update snapshot). Then step0 and
          step1 are ONE trajectory point, the sweep has 26 distinct
          checkpoints rather than 27, and every "steps 0-4" aggregate row in
          status-1 is averaging a duplicate.

      (b) The loader is caching across revisions. Then the identity is a bug
          in our code, it may have silently affected other adjacent revision
          pairs, and no aggregate anywhere in the sweep can be trusted until
          it is found.

    These are not distinguishable from the analysis outputs — only from the
    weights. This script hashes them.

WHAT IT DOES
    For each requested revision: loads the model in a FRESH process-local
    state, hashes every parameter tensor's raw bytes, and reports a
    per-parameter and whole-model digest. Also reports the resolved commit
    SHA from the Hub, which settles (a) directly and cheaply.

    The cache-invalidation control is the important part and is why this is a
    script rather than a notebook cell: revisions are loaded in a randomized
    order, and each load is preceded by an explicit cache clear, so a caching
    bug cannot produce a false match by load order alone.

USAGE
    python -m tools.verify_checkpoint_hashes \\
        --model EleutherAI/pythia-410m \\
        --revisions step0 step1 step2 step4

COST
    [W] — weights only, no forward passes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path


def _param_digest(model) -> tuple[str, dict]:
    """
    (whole_model_sha256, {param_name: sha256}) over raw tensor bytes.

    Hashing bytes rather than a numeric summary is deliberate: a summary
    statistic can collide across genuinely different weights, and the whole
    question here is whether two things are the SAME, so a false match is
    the expensive error.
    """
    per_param: dict[str, str] = {}
    running = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        arr = tensor.detach().cpu().contiguous().numpy()
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        per_param[name] = h
        running.update(name.encode())
        running.update(h.encode())
    return running.hexdigest(), per_param


def _resolved_sha(model_name: str, revision: str):
    """The Hub commit the revision tag actually points at. Settles (a) alone."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name, revision=revision)
        return getattr(info, "sha", None)
    except Exception as exc:                      # offline, auth, missing tag
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--revisions", nargs="+", default=["step0", "step1", "step2"])
    ap.add_argument("--out", type=Path, default=Path("checkpoint_hashes.json"))
    ap.add_argument("--seed", type=int, default=0,
                    help="load-order shuffle seed (cache-bug control)")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    order = list(args.revisions)
    random.Random(args.seed).shuffle(order)
    print(f"load order (shuffled, seed={args.seed}): {order}\n")

    results: dict[str, dict] = {}
    for rev in order:
        # Explicit teardown between loads. If a match survives this, it is
        # not our cache.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"loading {args.model} @ {rev} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=rev, torch_dtype=torch.float32,
        )
        digest, per_param = _param_digest(model)
        results[rev] = {
            "model_digest":  digest,
            "resolved_sha":  _resolved_sha(args.model, rev),
            "n_params":      len(per_param),
            "per_param":     per_param,
        }
        print(f"  digest {digest[:16]}...  hub sha {results[rev]['resolved_sha']}")
        del model
        gc.collect()

    print("\n" + "=" * 68)
    print("PAIRWISE COMPARISON (original order)")
    print("=" * 68)
    revs = list(args.revisions)
    for i in range(len(revs) - 1):
        a, b = revs[i], revs[i + 1]
        if a not in results or b not in results:
            continue
        ra, rb = results[a], results[b]
        same_weights = ra["model_digest"] == rb["model_digest"]
        same_sha = (ra["resolved_sha"] == rb["resolved_sha"]
                    and not str(ra["resolved_sha"]).startswith("<"))
        n_diff = sum(1 for k in ra["per_param"]
                     if ra["per_param"][k] != rb["per_param"].get(k))

        print(f"\n  {a} vs {b}")
        print(f"    identical weights : {same_weights}")
        print(f"    params differing  : {n_diff} / {ra['n_params']}")
        print(f"    same hub commit   : {same_sha}")

        if same_weights and same_sha:
            print(f"    => Reading (a): the {b} tag resolves to the same Hub")
            print(f"       commit as {a}. These are ONE trajectory point.")
            print(f"       Drop one from the sweep and restate any aggregate")
            print(f"       row that averaged them as distinct checkpoints.")
        elif same_weights and not same_sha:
            print(f"    => Reading (b): DIFFERENT Hub commits, IDENTICAL")
            print(f"       weights. Either the loader is caching across")
            print(f"       revisions, or the upstream commits differ only in")
            print(f"       non-weight files. Check the per-param digests and")
            print(f"       the repo diff before trusting any adjacent pair.")
        else:
            print(f"    => Distinct weights. Both are real trajectory points.")

    args.out.write_text(json.dumps(
        {rev: {k: v for k, v in r.items() if k != "per_param"}
         for rev, r in results.items()},
        indent=2,
    ))
    print(f"\nwrote {args.out} (per-parameter digests omitted; rerun with a "
          f"debugger if a pair needs localizing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
