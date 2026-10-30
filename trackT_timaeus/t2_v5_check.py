"""
trackT_timaeus/t2_v5_check.py — T2's Llama path against the library that saved the checkpoints
(`/challenge-pr` on #285, finding 4).

Dodecahedron was saved by transformers 5.2.0 and tokenizers ≥ 0.20; `t2.py` reads it with 4.44.2
and 0.19.1, joining the merge pairs and trusting 4.44's Llama on a 5.2 config. This dumps the token
ids and every hidden state for the 7 v1 prompts at one step, on the CPU in float32, under whichever
env runs it (`dump`; the 4.44 env goes through `t2.tokenizer` and `t2.load`, the 5.2 env through its
own tokenizer and loader, nothing shared but the prompts), and `compare` reports ids equal and the
largest absolute and relative state differences.

    HF_HOME=<HDD_1TB>/mets_data/hf python -m trackT_timaeus.t2_v5_check dump --model 32m --out <dir>/v444.npz
    HF_HOME=<HDD_1TB>/mets_data/hf <v5 venv>/bin/python -m trackT_timaeus.t2_v5_check dump --model 32m --out <dir>/v52.npz
    python -m trackT_timaeus.t2_v5_check compare <dir>/v444.npz <dir>/v52.npz
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import numpy as np

MODELS = {"32m": "timaeus/dodecahedron-32m", "67m": "timaeus/dodecahedron-67m"}
V1 = ("camus_letranger", "hdbscan_code", "homer_iliad", "latex_monograph",
      "paper_excerpt", "sullivan_ballou", "wiki_paragraph")       # = trackA_force.a1a.V1
MAX_TOKENS = 512


def dump(a) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import transformers
    from core.config import PROMPTS
    repo, rev = MODELS[a.model], f"step{a.step}"
    if transformers.__version__.startswith("4."):
        from trackT_timaeus import t2
        tok, model = t2.tokenizer(a.model), t2.load(repo, rev, "cpu")
    else:
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(repo, "tokenizer.json", revision=rev),
                                      model_input_names=["input_ids", "attention_mask"])
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32,
                                                     attn_implementation="eager").eval()
    arrs = {"meta": np.array(json.dumps({"transformers": transformers.__version__, "model": a.model, "step": a.step}))}
    for p in V1:
        enc = tok(PROMPTS[p], return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        with torch.no_grad():
            o = model(input_ids=enc["input_ids"], output_hidden_states=True)
        arrs[f"{p}/ids"] = enc["input_ids"][0].numpy()
        arrs[f"{p}/states"] = torch.stack(o.hidden_states)[:, 0].float().numpy()
    np.savez(a.out, **arrs)
    print(f"wrote {a.out} ({transformers.__version__})")
    return 0


def compare(a) -> int:
    x, y = np.load(a.a), np.load(a.b)
    out = {"a": json.loads(str(x["meta"])), "b": json.loads(str(y["meta"])), "per_prompt": {}}
    for p in V1:
        ids = bool(np.array_equal(x[f"{p}/ids"], y[f"{p}/ids"]))
        sa, sb = x[f"{p}/states"], y[f"{p}/states"]
        r = {"ids_equal": ids, "n": int(len(x[f"{p}/ids"]))}
        if ids and sa.shape == sb.shape:
            d = np.abs(sa - sb)
            r.update(max_abs=float(d.max()), max_rel=float((d.max(axis=-1) / np.linalg.norm(sa, axis=-1)).max()))
        out["per_prompt"][p] = r
    out["all_ids_equal"] = all(r["ids_equal"] for r in out["per_prompt"].values())
    out["max_abs"] = max(r.get("max_abs", np.inf) for r in out["per_prompt"].values())
    out["max_rel"] = max(r.get("max_rel", np.inf) for r in out["per_prompt"].values())
    print(json.dumps(out, indent=1))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("dump")
    sp.add_argument("--model", choices=list(MODELS), required=True)
    sp.add_argument("--step", type=int, default=9536)
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("compare")
    sp.add_argument("a")
    sp.add_argument("b")
    a = ap.parse_args(argv)
    return {"dump": dump, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
