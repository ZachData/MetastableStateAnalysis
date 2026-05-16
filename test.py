# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("albert-base-v2")
# import json
# prompts = json.load(open("p2c_churchland/prompt_grids/matched_length.json"))
# for p in prompts:
#     print(len(tok(p["prompt"])["input_ids"]), p["prompt"][:40])


# from transformers import AutoTokenizer
# import json

# tok = AutoTokenizer.from_pretrained("albert-base-v2")
# prompts = json.load(open("p2c_churchland/prompt_grids/induction_prompts.json"))

# for group in ("induction", "control"):
#     print(f"\n--- {group} ---")
#     for p in prompts[group]:
#         n = len(tok(p["prompt"])["input_ids"])
#         print(f"{n:3d}  [{p['tier']}]  {p['prompt'][:50]}")


import numpy as np
d = np.load("results/p2_eigenspectra_2026-04-28_13-22-34/ov_projectors_albert-xlarge-v2.npz")
print(d.files)