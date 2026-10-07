"""
p1e_energy_field/extract_long8.py — the forward pass for 1e's 8 long passages
(`long_prompts_1e.load8`) at the 18 Stage 0 steps of `pythia-410m`.

As `p1d_cluster_ensemble/extract_long.py`, with three differences:

- **Activations only.** ``activations.npz`` (unit rows + ``norms``),
  ``geometry.json`` (tokens, provenance), ``tokens.txt``, ``manifest.json``.
  No ``attentions.npz``: U2's block arm reads residuals only, and 2048² maps
  are ~3 GB per run. An arm that needs attention runs its own pass.
- **GPU** where visible (`docs/compute_profile.md` "The GPU": every cloud a
  1e unit compares comes from this batch, so from one device), float32,
  eager attention, TF32 off; the device goes into the manifest.
- **8 passages, one hash.** Token counts are checked against both
  provenance files; the manifest records ``long8_hash``.

No 512 cap: tokenized without truncation, refused if the count is not the
recorded one or is above 2048. Tier 1: exploratory, unregistered.
    python -m p1e_energy_field.extract_long8 [--steps ...] [--keys ...] --out <dir> [--skip-existing]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from .long_prompts_1e import DATA_ROOT, MAX_TOKENS, expected_tokens, load8, long8_hash

#: The 18 distinct Stage 0 steps (`p10_cluster_function/status-10.md` §1.14's index).
STEPS = (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 54000, 143000)


def run_one(model, tokenizer, model_name: str, key: str, text: str, n_expected: int,
            h: str, out_root: Path) -> Path:
    """One forward pass, one run directory ``<out_root>/<model>_<key>``."""
    import torch
    from core.io import write_manifest
    from core.models import describe_extraction
    from p1_mstate_tracking.p1_io import _save_activations, _save_geometry, _save_tokens

    t0 = time.monotonic()
    ids = tokenizer(text, return_tensors="pt")["input_ids"]      # no truncation
    n = int(ids.shape[1])
    if n != n_expected or n > MAX_TOKENS:
        raise ValueError(f"{key}: {n} tokens, provenance says {n_expected} (cap {MAX_TOKENS}); refusing")
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=ids.to(dev), output_hidden_states=True, output_attentions=False)
    hidden = [x[0].to(torch.float32).cpu() for x in out.hidden_states]
    del out
    tokens = tokenizer.convert_ids_to_tokens(ids[0])
    run_dir = Path(out_root) / f"{model_name}_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # describe_extraction counts attention layers from its list; none are saved here
    meta = describe_extraction(model, model_name, hidden, [None] * (len(hidden) - 1))
    results = {"model": model_name, "prompt": key, "n_layers": len(hidden), "n_tokens": n,
               "d_model": int(hidden[0].shape[-1]), "tokens": tokens, "layers": [], **meta}
    _save_tokens(results, run_dir)
    _save_geometry(results, run_dir)
    _save_activations(hidden, run_dir)
    write_manifest(run_dir, model=model_name, prompt_battery_hash=h,
                   wall_time_seconds=time.monotonic() - t0,
                   hf_revision=meta.get("revision"), checkpoint_step=meta.get("checkpoint_step"),
                   prompt_key=key, seeds={"torch": 0},
                   config={"weight_dtype": meta.get("weight_dtype"), "device": str(dev),
                           "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                           "truncation": None, "max_tokens": MAX_TOKENS, "attentions_saved": False},
                   extra={"phase": "p1e_long8", "long8_hash": h, "hf_repo": meta.get("hf_repo"),
                          "random_init": meta.get("random_init"), "n_tokens": n,
                          "n_layers_analyzed": results["n_layers"]})
    return run_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--steps", nargs="*", type=int, default=list(STEPS))
    ap.add_argument("--keys", nargs="*", default=None, help="default: all 8")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)

    import torch
    from core.models import load_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    texts, n_exp, h = load8(), expected_tokens(), long8_hash()
    keys = a.keys or sorted(texts)
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        m = f"pythia-410m-step{s}"
        todo = [k for k in keys if not (a.skip_existing and (a.out / f"{m}_{k}" / "manifest.json").exists())]
        if not todo:
            print(f"already {m}", flush=True)
            continue
        model, tok = load_model(m)
        for k in todo:
            t0 = time.monotonic()
            d = run_one(model, tok, m, k, texts[k], n_exp[k], h, a.out)
            print(f"done {d} {time.monotonic() - t0:.1f}s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (a.out / "long8.json").write_text(json.dumps({"long8_hash": h, "keys": keys, "steps": a.steps,
                                                  "data_root": DATA_ROOT}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
