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


# import numpy as np
# d = np.load("results/p2_eigenspectra_2026-04-28_13-22-34/ov_projectors_albert-xlarge-v2.npz")
# print(d.files)


# from p6_subspace.subspace_build import load_projectors, print_projector_summary
# proj = load_projectors("./results/phase6/albert_xlarge_v2/projectors.npz")
# print_projector_summary(proj)


# quick check before full rebuild
from p6_subspace.subspace_build import _extract_schur_subspaces
from pathlib import Path
import numpy as np


# --- load one head's OV ---
phase2_dir = Path("results/phase2")          # adjust to your actual path
stem       = "albert-xlarge-v2"              # or albert_xlarge_v2 — try both

weights_path = "./results/p2_eigenspectra_2026-04-28_13-22-34/ov_weights_albert-xlarge-v2.npz"
data = np.load(weights_path)
print(sorted(data.files)[:10])
# ALBERT uses shared weights; keys are ov_head0_shared, ov_head1_shared, ...
OV_head_0 = data["ov_head0_shared"].astype(np.float64)

# --- run the diagnostic ---
sub = _extract_schur_subspaces(OV_head_0, eig_rel_tol=1e-8)
print("eig_tol_used:", sub["eig_tol_used"])
print("max_eig_mag: ", sub["max_eig_mag"])
print(f"real_pos={len(sub['real_pos_vecs'])}  "
      f"real_neg={len(sub['real_neg_vecs'])}  "
      f"rot={len(sub['rot_vecs']) // 2}  "
      f"kernel_real={sub['n_kernel_real']}")

# also useful: see the full eigenvalue magnitude distribution
T, _ = __import__('scipy').linalg.schur(OV_head_0, output='real')
d = OV_head_0.shape[0]
eig_mags = []
i = 0
while i < d:
    if i < d-1 and abs(T[i+1, i]) > sub["eig_tol_used"]:
        a, c = T[i,i], T[i+1,i]
        eig_mags.append(float(np.sqrt(max(a*T[i+1,i+1] - T[i,i+1]*c, 0))))
        i += 2
    else:
        eig_mags.append(abs(T[i,i]))
        i += 1
eig_mags = np.sort(eig_mags)[::-1]
print("top-10 eig magnitudes:", eig_mags[:10].round(4))
print("eig_tol_used:         ", sub["eig_tol_used"])
print("modes above tol:      ", int((eig_mags > sub["eig_tol_used"]).sum()),
      "of", len(eig_mags))