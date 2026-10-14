"""B0 over 18 steps: the one G4b mismatch (step 128, camus_letranger j = 82, d from hdbscan_code), its linear margin
against its nudged |A|/eps at every c. Run from the repo root on the GPU."""
import sys; sys.path.insert(0, '.')
from core import nudge as nd
nd.set_numerics()
import torch, numpy as np
from trackB_circuit import b0_gate as bg
m, tok = bg.load(128, "cuda", torch.float32); st = nd.Stack(m)
S = {}
for key in ("camus_letranger", "hdbscan_code"):
    ids = bg.token_ids(tok, key); x, r, J, R = bg.site(st, ids, key); S[key] = (ids, x, r, J, R)
ids, x, r, J, R = S["camus_letranger"]; k = [int(j) for j in J].index(82)
g = bg.grad_at(st, x, ids, 82)
d = nd.unit(bg.grad_at(st, S["hdbscan_code"][1], S["hdbscan_code"][0], int(S["hdbscan_code"][3][k])))
lin0 = abs(g @ d); linR = np.abs(R @ g); top = int(np.argmax(linR))
print("linear: |g.d|", lin0, "max |g.R|", linR[top], "(k=%d)" % top, "rel gap", (lin0 - linR[top]) / lin0, "||g||", np.linalg.norm(g))
dirs = np.vstack([d, R])
with torch.no_grad():
    for c in nd.C_GRID:
        eps = c * r; delta = st.deltas(dirs, eps, [82], 83)
        out = st.run(x[:, :83].expand(len(delta), -1, -1), bg.LAYER, delta=delta, loss_ids=ids, loss_at=[82])["loss"][:, 0].double().cpu().numpy()
        b, p, mm = nd.split_rows(out); A = nd.anti_sym(p, mm, b)[0]
        print(f"c={c}: |A0|/eps {abs(A[0])/eps:.6g}  max|A_R|/eps {np.abs(A[1:]).max()/eps:.6g} (k={int(np.argmax(np.abs(A[1:])))})  lin pred {lin0:.6g} / {linR[top]:.6g}")
