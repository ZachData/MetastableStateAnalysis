"""
trackS_subspace/s4_tl_check.py — S4 U1: `s4.toy_forward` against TransformerLens itself.

TransformerLens is not in the project env; run this under a venv that has it (2.16.1 was used:
`python -m venv --system-site-packages <v>; <v>/bin/pip install --no-deps transformer_lens==2.16.1
jaxtyping einops fancy_einsum better_abc typeguard==4.4.1 wadler_lindig`, with an empty `wandb`
package on PYTHONPATH). Per (arm, step): the max |logit difference| on `wiki_paragraph` and TL's
loss. Writes <out>/tl_check.json.

    HF_HOME=… HF_HUB_OFFLINE=1 <v>/bin/python -m trackS_subspace.s4_tl_check --out <dir>
"""
import argparse
import json
from pathlib import Path

CASES = (("R2", 75000), ("R3", 75000), ("R4", 1035), ("R4", 75000), ("R5", 75000), ("R6", 75000))


def main() -> int:
    import torch
    import transformer_lens
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from trackS_subspace import s4
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    texts = s4.prompt_texts()
    rows = []
    for arm, step in CASES:
        tok, _ = s4.toy_tokenizer(arm)
        x = torch.tensor([tok.encode(texts["wiki_paragraph"], add_special_tokens=False).ids])
        cfg = HookedTransformerConfig(n_layers=s4.n_layers(arm), d_model=256, n_ctx=1024, d_head=32,
                                      n_heads=8, act_fn="gelu", d_vocab=5000, attn_only=True,
                                      positional_embedding_type="shortformer", device="cpu")
        m = HookedTransformer(cfg)
        w, _ = s4.toy_load(arm, step, "cpu")
        sd = m.state_dict()
        missing = [k for k in sd if k not in w and not k.endswith(("mask", "IGNORE"))]
        if missing:
            raise SystemExit(f"REFUSE: {arm} {step}: TL keys not in the checkpoint: {missing[:4]}")
        m.load_state_dict({**sd, **{k: v for k, v in w.items() if k in sd}})
        m.eval()
        with torch.no_grad():
            ref = m(x)
            _, _, mine = s4.toy_forward(w, x, True)
        rows.append({"arm": arm, "step": step, "max_abs_dlogit": float((ref - mine).abs().max()),
                     "tl_loss": float(torch.nn.functional.cross_entropy(ref[0, :-1], x[0, 1:]))})
        print(rows[-1], flush=True)
    rec = {"transformer_lens": transformer_lens.__version__ if hasattr(transformer_lens, "__version__")
           else "2.16.1", "s4_code": s4.code_sha(), "rows": rows}
    Path(a.out, "tl_check.json").write_text(json.dumps(rec, indent=1) + "\n")
    bad = [r for r in rows if r["max_abs_dlogit"] > 1e-3]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
