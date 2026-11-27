"""C9 unit 3's producer: C2 3b's 160m records on the nine PolyPythias 160m seeds (`design-C9.md`
"U3 as it runs"; D5 (a), tier 1). Seed 0 (`pythia-160m`) is passed again by the same code as the
known answer against 3b's records (`c9_u3 known`).

One pass per (seed, step): `c2_aniso.states` (truncation 512, the 410m tokenizer as 3b), then
`c2_aniso.state_row` per state (e1, e1c, e1r, cos). Each record's ids hash must equal 3b's seed-0
record for that (step, prompt), or the pass refuses. Weights float32, offline; seeds 1–9 from
`Mets_hdd_storage/hf/hub` (C9's download), seed 0 from `HDD_1TB/mets_data/hf/hub`.

Run (GPU):
    METS_DATA=<main>/data HF_HUB_OFFLINE=1 python -m trackC_writeup.c9_u3_pass --out <dir> [--seeds 0 1 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

from trackA_force.a1a import DATA, REPO, STEPS, V1, prompts_hash
from trackA_force.a5_seeds import TOKENIZER
from trackC_writeup import c2_aniso as c
from trackC_writeup import c2_dose as cd

SEEDS = tuple(range(10))
HUB = {"seed0": Path("/run/media/system/HDD_1TB/mets_data/hf/hub"),
       "seeds": Path("/run/media/system/WDS_500/Mets_hdd_storage/hf/hub")}
REF = DATA / "trackC" / "c2u3b" / "2026-10-22-run" / "160m" / "records"     # 3b's seed-0 records


class PassRefused(RuntimeError):
    pass


def repo_for(seed: int) -> str:
    return "EleutherAI/pythia-160m" if seed == 0 else f"EleutherAI/pythia-160m-seed{seed}"


def hub_for(seed: int) -> Path:
    return HUB["seed0"] if seed == 0 else HUB["seeds"]


def code_sha() -> str:
    """This file and the code it calls (not the reading)."""
    h = hashlib.sha256()
    for f in ("trackC_writeup/c9_u3_pass.py", "trackC_writeup/c2_aniso.py", "trackC_writeup/c2_dose.py",
              "core/metrics.py", "core/models.py"):
        h.update((REPO / f).read_bytes())
    return h.hexdigest()[:12]


def rec_path(out: Path, seed: int, step: int, prompt: str) -> Path:
    return out / f"seed{seed}" / "records" / f"step{step}_{prompt}.json"


def load(seed: int, step: int, device):
    import torch
    from transformers import GPTNeoXForCausalLM
    from core.models import from_pretrained_eager
    m = from_pretrained_eager(GPTNeoXForCausalLM, repo_for(seed), revision=f"step{step}",
                              cache_dir=str(hub_for(seed)), local_files_only=True,
                              torch_dtype=torch.float32).to(device).eval()
    if next(m.parameters()).dtype != torch.float32:
        raise PassRefused(f"seed {seed} step {step}: weights are not float32")
    return m


def run(out: Path, seeds: Sequence[int]) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch, device = c._torch_setup()
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    code = code_sha()
    plan = {"design": 'trackC_writeup/design-C9.md "U3 as it runs"', "code": code, "seeds": list(seeds),
            "repos": {s: repo_for(s) for s in seeds}, "hubs": {s: str(hub_for(s)) for s in seeds},
            "prompts_hash": prompts_hash(), "prompts": list(V1), "tokenizer": list(TOKENIZER),
            "steps": list(STEPS), "dtype": "float32", "tf32": False,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    first, t0 = True, time.time()
    for seed in seeds:
        rec_path(out, seed, 0, V1[0]).parent.mkdir(parents=True, exist_ok=True)
        for step in STEPS:
            todo = [p for p in V1 if not rec_path(out, seed, step, p).exists()]
            if not todo:
                continue
            model = load(seed, step, device)
            n_states = model.config.num_hidden_layers + 1
            for p in todo:
                hs, ids = c.states(model, tok, p, device)
                rec = cd.record(hs, ids, p, n_states)
                ref = json.loads((REF / f"step{step}_{p}.json").read_text())
                if rec["ids_sha"] != ref["ids_sha"] or rec["n"] != ref["n"]:
                    raise PassRefused(f"seed {seed} step {step} {p}: tokens differ from 3b's seed-0 record")
                rec.update(seed=seed, repo=repo_for(seed), step=step, revision=f"step{step}", code=code)
                rec_path(out, seed, step, p).write_text(json.dumps(rec) + "\n")
                if first:
                    print(f"first record populated: seed {seed} step {step} {p}, n {rec['n']}, "
                          f"states {n_states}, e1[0] {rec['e1'][0]:.6f}, e1c[0] {rec['e1c'][0]:.6f}, "
                          f"cos[-1] {rec['cos'][-1]:.4f}", flush=True)
                    first = False
            del model
            torch.cuda.empty_cache()
        print(f"seed {seed} done ({time.time() - t0:.0f} s)", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    a = ap.parse_args(argv)
    return run(Path(a.out), a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
