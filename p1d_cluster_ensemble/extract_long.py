"""
p1d_cluster_ensemble/extract_long.py — the forward pass for the long prompts
(`long_prompts.py`), writing only what Phase 1d reads.

`run_1` is not used: its Sinkhorn / spectral / UMAP pass is slow at 2048
tokens and nothing in 1d reads it. What is written goes through Phase 1's
own save helpers (`p1_mstate_tracking/p1_io.py`), so a long run directory
reads like a v1 one to every 1d loader:

- ``activations.npz`` (``activations`` on the sphere + ``norms``),
- ``attentions.npz`` (float32, ``(n_blocks, n_heads, n, n)``),
- ``geometry.json`` (``tokens`` + extraction provenance; ``layers`` is
  empty, since the per-layer geometry is `run_1`'s),
- ``tokens.txt``, ``manifest.json`` (``prompt_key`` = the long key,
  ``long_prompts_hash``, and ``prompt_battery_hash`` set to the same, since
  the long prompts are not in the battery; rule 6).

**No 512 cap.** `core.models.extract_activations` tokenizes with
``truncation=True, max_length=512``; this module tokenizes without
truncation and refuses a prompt whose token count is not the one
``provenance.json`` recorded, or is above ``MAX_TOKENS``.

`prefix_check` is step 3 of `status-1d.md` "Long prompts": the long run's
first ``n_v1`` rows against a stored v1 run, activations and attention.

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .long_prompts import DATA_ROOT, HERE, MAX_TOKENS, load, long_prompts_hash


def _expected() -> Dict[str, Dict]:
    return json.loads((HERE / "provenance.json").read_text())["prompts"]


def run_one(model, tokenizer, model_name: str, key: str, text: str, out_root: Path) -> Path:
    """One forward pass, one run directory ``<out_root>/<model>_<key>``."""
    import torch
    from core.io import write_manifest
    from core.models import describe_extraction
    from p1_mstate_tracking.p1_io import (_save_activations, _save_attentions,
                                          _save_geometry, _save_tokens)

    exp = _expected()[key]
    t0 = time.monotonic()
    ids = tokenizer(text, return_tensors="pt")["input_ids"]      # no truncation
    n = int(ids.shape[1])
    if n != exp["n_tokens"] or n > MAX_TOKENS:
        raise ValueError(f"{key}: {n} tokens, provenance.json says {exp['n_tokens']} "
                         f"(cap {MAX_TOKENS}); refusing")
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, output_attentions=True)
    if not out.attentions:
        raise RuntimeError(f"{model_name}: no attention weights returned (eager pin dropped?)")
    hidden = [h[0].to(torch.float32).cpu() for h in out.hidden_states]
    attn = [a[0].to(torch.float32).cpu() for a in out.attentions]
    del out
    tokens = tokenizer.convert_ids_to_tokens(ids[0])

    run_dir = Path(out_root) / f"{model_name}_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = describe_extraction(model, model_name, hidden, attn)
    results = {"model": model_name, "prompt": key, "n_layers": len(hidden), "n_tokens": n,
               "d_model": int(hidden[0].shape[-1]), "tokens": tokens, "layers": [], **meta}
    _save_tokens(results, run_dir)
    _save_geometry(results, run_dir)
    _save_activations(hidden, run_dir)
    del hidden
    _save_attentions(attn, run_dir)
    del attn
    h = long_prompts_hash()
    write_manifest(run_dir, model=model_name, prompt_battery_hash=h,
                   wall_time_seconds=time.monotonic() - t0,
                   hf_revision=meta.get("revision"), checkpoint_step=meta.get("checkpoint_step"),
                   prompt_key=key, seeds={"torch": 0},
                   config={"weight_dtype": meta.get("weight_dtype"), "device": meta.get("device"),
                           "truncation": None, "max_tokens": MAX_TOKENS},
                   extra={"phase": "p1d_long", "long_prompts_hash": h, "hf_repo": meta.get("hf_repo"),
                          "random_init": meta.get("random_init"), "n_tokens": n,
                          "n_v1_tokens": exp["n_v1_tokens"], "n_layers_analyzed": results["n_layers"]})
    return run_dir


def prefix_check(long_dir: Path, v1_dir: Path) -> Dict:
    """Max abs difference between the long run's first ``n_v1`` rows and the v1 run."""
    long_dir, v1_dir = Path(long_dir), Path(v1_dir)
    rec: Dict = {"long": str(long_dir), "v1": str(v1_dir)}
    a1 = np.load(v1_dir / "activations.npz")
    al = np.load(long_dir / "activations.npz")
    n = a1["activations"].shape[1]
    rec["n_v1"] = int(n)
    t1 = json.loads((v1_dir / "geometry.json").read_text())["tokens"]
    tl = json.loads((long_dir / "geometry.json").read_text())["tokens"]
    rec["tokens_equal"] = t1 == tl[:n]
    rec["act_max_abs"] = float(np.abs(al["activations"][:, :n] - a1["activations"]).max())
    rel = np.abs(al["norms"][:, :n] - a1["norms"]) / np.maximum(a1["norms"], 1e-12)
    rec["norm_max_rel"] = float(rel.max())
    del a1, al
    A1 = np.load(v1_dir / "attentions.npz")["attentions"]
    Al = np.load(long_dir / "attentions.npz")["attentions"]
    rec["attn_max_abs"] = float(np.abs(Al[:, :, :n, :n] - A1).max())
    # causal: no row of the prefix may attend past n_v1
    rec["attn_leak_max"] = float(np.abs(Al[:, :, :n, n:]).max()) if Al.shape[-1] > n else 0.0
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="forward passes")
    r.add_argument("--models", nargs="+", default=["pythia-410m-step143000", "pythia-410m-step0"])
    r.add_argument("--keys", nargs="*", default=None, help="default: every built long prompt")
    r.add_argument("--out", type=Path, default=Path(DATA_ROOT) / "p1d_long" / date.today().isoformat())
    r.add_argument("--skip-existing", action="store_true")
    c = sub.add_parser("prefix", help="step 3: long run vs stored v1 run")
    c.add_argument("--pairs", nargs="+", required=True, help="LONG_DIR=V1_DIR ...")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "prefix":
        recs = []
        for p in args.pairs:
            l, _, v = p.partition("=")
            recs.append(prefix_check(Path(l), Path(v)))
            print(json.dumps(recs[-1]), flush=True)
        args.out.write_text(json.dumps(recs, indent=1) + "\n")
        return 0

    from core.models import load_model
    texts = load()
    keys = args.keys or sorted(texts)
    for m in args.models:
        model, tok = load_model(m)
        for k in keys:
            d = args.out / f"{m}_{k}"
            if args.skip_existing and (d / "manifest.json").exists():
                print(f"already {d}", flush=True)
                continue
            t0 = time.monotonic()
            run_one(model, tok, m, k, texts[k], args.out)
            print(f"done {d} {time.monotonic() - t0:.0f}s", flush=True)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
