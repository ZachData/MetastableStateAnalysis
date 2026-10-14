"""Diagnostic for G4b at one step: the rank from gradient geometry alone vs the nudged rank."""
import sys, json; sys.path.insert(0, '.')
from core import nudge as nd
nd.set_numerics()
import torch, numpy as np
from trackB_circuit import b0_gate as bg
step = int(sys.argv[1]); out = sys.argv[2]
rec = json.load(open(f"{out}/gpu_step{step}.json"))
m, tok = bg.load(step, "cuda", torch.float32); st = nd.Stack(m)
S, G = {}, {}
for key in bg.V1_SCORED:
    ids = bg.token_ids(tok, key); x, r, J, R = bg.site(st, ids, key)
    S[key] = (R, J); G[key] = np.array([bg.grad_at(st, x, ids, int(j)) for j in J])
agree = tot = hits_geom = same1 = 0; shifts = []; cos_other, cos_rand = [], []
for a, key in enumerate(bg.V1_SCORED):
    other = bg.V1_SCORED[(a + 1) % 7]; R, J = S[key]
    for k in range(32):
        g = G[key][k]; d = nd.unit(G[other][k])
        rg = nd.exact_rank(abs(g @ d), np.abs(R @ g))
        rn = rec["prompts"][key]["G4b"]["arms"][k]["c"]["0.01"]["rank"]
        agree += rg == rn; same1 += (rg == 1) == (rn == 1); shifts.append(abs(rg - rn)); tot += 1; hits_geom += rg == 1
        gu = nd.unit(g); cos_other.append(abs(gu @ d)); cos_rand.append(np.median(np.abs(R @ gu)))
# cross-prompt gradient alignment: mean |cos| between all pairs of gradients from different prompts
allg = np.vstack([np.array([nd.unit(v) for v in G[k]]) for k in bg.V1_SCORED]); lab = np.repeat(np.arange(7), 32)
C = np.abs(allg @ allg.T); off = lab[:, None] != lab[None, :]
mean_g = nd.unit(allg.mean(0)); 
print(json.dumps({"step": step, "rank_agree_geom_vs_nudged_c0.01": f"{agree}/{tot}", "rank1_geom": hits_geom, "rank1_indicator_agree": f"{same1}/{tot}", "max_rank_shift": int(max(shifts)), "rank_shift_le_1": f"{sum(x <= 1 for x in shifts)}/{tot}",
  "median_abs_cos_other": float(np.median(cos_other)), "median_abs_cos_random_median": float(np.median(cos_rand)),
  "mean_abs_cos_cross_prompt_pairs": float(C[off].mean()), "random_expect_abs_cos": float(np.sqrt(2/np.pi/1024)),
  "share_of_grad_on_mean_grad_dir_median": float(np.median(np.abs(allg @ mean_g)))}, indent=1))
