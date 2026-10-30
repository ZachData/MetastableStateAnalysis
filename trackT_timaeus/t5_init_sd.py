"""
trackT_timaeus/t5_init_sd.py — the init scales behind T5's H1 (`docs/ROADMAP.md` §7b row T5).

Prints, per tensor class (layer index folded to N), the SD of every 2-D weight and the largest
|value| of every 1-D one (norm gains, biases), from one checkpoint file. Read-only; no forward
pass. The numbers it gave are in `trackT_timaeus/status-T.md` "T5's proposal: the init scales
measured"; their closed forms are checked in `tools/math_checks/t5_init.py`.

    python -I trackT_timaeus/t5_init_sd.py <model.safetensors | pytorch_model.bin>
"""

import re
import sys
from collections import defaultdict

import torch


def load(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def main(path: str) -> None:
    groups = defaultdict(list)
    for k, v in load(path).items():
        if v.ndim < 1 or not v.is_floating_point():
            continue
        name = re.sub(r"\.\d+\.", ".N.", k)
        kind = "sd" if v.ndim == 2 else "max|x|"
        groups[(name, kind)].append(v.float().std().item() if v.ndim == 2 else v.float().abs().max().item())
    for (name, kind), xs in sorted(groups.items()):
        print(f"{name:60s} n={len(xs):2d} {kind:6s} min={min(xs):.4g} max={max(xs):.4g}")


if __name__ == "__main__":
    main(sys.argv[1])
