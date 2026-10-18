"""B3 unit 1's gate failed (G2's C3′, G3; `status-B.md` "B3 unit 1"): the revisions put to the
user as Blocked 66, each run on the gate's own inputs with no member α read.

(1) The induction score at each member's own offset at 143000, and the formation steps it gives.
(2) R2's onset with a step-0 floor (bottom decile of its step *and* below step 0's 2.5th
    percentile): the layer-matched null hit rates, the attainable rank, and G3's planted count
    (primary set and 50 beside), with `L12H5` untyped.
(3) C3′ with one fixed pool head per member over the 9 steps (pools by mean type score over steps
    ≥ 512): K = 5, 10, 20 with re-centred fakes (unit 1's G2 construction), and K = 5, 10
    exchangeable (the fake drawn from the member's pool, its null that pool without it); 400 fakes.

Tier 1: exploratory, unregistered.
    python -m trackB_circuit.b3_gate_variants <unit 1's out dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from trackB_circuit import b3_route as b3
from trackB_circuit.b2a import seed_for

N_FAKE = 400


def main(out: Path) -> dict:
    D = b3.load_all(out)
    A = b3.alpha_grid(D)
    t = D["types"][143000]
    ser = {k: b3.type_series(D, k) for k in ("prev", "ind0", "ind1")}
    offs = {m: ("ind0" if t["ind0"][b3.head_index(m)] >= t["ind1"][b3.head_index(m)] else "ind1")
            for m in b3.INDUCTION}
    key = {"L5H2": "prev", **offs}
    peak = {m: float(ser[key[m]][:, b3.head_index(m)].max()) for m in key}
    form = {m: b3.formation_index(ser[key[m]][:, b3.head_index(m)]) for m in key}
    rec = {"offsets": offs, "peak_type_score": peak, "formation_steps": {m: b3.STEPS[k] for m, k in form.items()}}

    # (2) R2 with a step-0 floor; L12H5 untyped (its peak induction score is 0.011)
    lo0 = float(np.nanquantile(A[0], 0.025))
    q = b3.q10(A)

    def onset(a, qq):
        hit = (a <= qq) & (a < lo0)
        for k in range(1, len(a) - 1):
            if hit[k] and hit[k + 1]:
                return k
        return None

    def hit(a, f, qq):
        k = onset(a, qq)
        return k is not None and abs(k - f) <= 1

    typed = ["L5H2", "L7H8", "L8H6", "L11H14"]
    fs = [form[m] for m in typed]
    rates = []
    for m, f in zip(typed, fs):
        l = b3.parse_head(m)[0]
        pool = [l * 16 + j for j in range(16) if l * 16 + j != b3.head_index(m)]
        rates.append(float(np.mean([hit(A[:, h], f, q) for h in pool])))
    F = b3.fake_sets([b3.parse_head(m)[0] for m in typed], 51, seed_for("b3-g3"))
    full = []
    for f in F:
        Ap = A.copy()
        for h, fk in zip(f, fs):
            Ap[:fk, h] = A[0, h]
            Ap[fk:, h] = 0.0                          # the two-partner rows' α (G1 (ii): 0 to 1e-15)
        qq = b3.q10(Ap)
        full.append(sum(hit(Ap[:, h], fk, qq) for h, fk in zip(f, fs)) == len(typed))
    # inclusive pools (`/challenge-pr` on #224, finding 2): the tested head counts in its own
    # layer's pool. Members' hits are not read, so the rates assume each member hits; the rank
    # at a count c is then P(null count < c), computed exactly from the per-layer rates.
    inc = [(r * 15 + 1) / 16 for r in rates]

    def rank_at(c, rs):
        dist = np.array([1.0])
        for r in rs:
            dist = np.convolve(dist, [1 - r, r])
        return float(dist[:c].sum())

    # G2 for R2 under the floor: 200 layer-matched fake sets, exclusive pools as frozen
    Ff = b3.fake_sets([b3.parse_head(m)[0] for m in typed], b3.N_DRAW, seed_for("b3-var-g2r2"))
    g2 = {"exclusive": [], "inclusive": []}
    for f in Ff:
        c = sum(hit(A[:, h], fk, q) for h, fk in zip(f, fs))
        rs_ex, rs_in = [], []
        for h, fk in zip(f, fs):
            l = h // 16
            others = [l * 16 + j for j in range(16) if l * 16 + j != h]
            r = float(np.mean([hit(A[:, o], fk, q) for o in others]))
            rs_ex.append(r)
            rs_in.append((r * 15 + hit(A[:, h], fk, q)) / 16)
        g2["exclusive"].append(rank_at(c, rs_ex))
        g2["inclusive"].append(rank_at(c, rs_in))
    rec["R2_floor"] = {"step0_p2.5": lo0, "members": typed, "rates": rates,
                       "attainable": float(1 - np.prod(rates)), "G3_primary_all": bool(full[0]),
                       "G3_beside_all_rate": float(np.mean(full[1:])),
                       "inclusive_rates_if_hit": inc,
                       "inclusive_rank_at": {str(c): rank_at(c, inc) for c in range(1, len(typed) + 1)},
                       "G2_false_pass": {k: float(np.mean(np.array(v) >= b3.RANK_PASS)) for k, v in g2.items()}}

    # (3) C3′ with a fixed head per member
    P3, _ = b3.c3_values()
    idx = [b3.STEPS.index(s) for s in b3.STAT_STEPS]
    mem = [b3.head_index(m) for m in b3.MEMBERS]
    ckey = {"L5H2": "prev", "L7H8": offs["L7H8"], "L8H6": offs["L8H6"], "L11H14": offs["L11H14"]}

    def pools(refs, excl, K):
        out = []
        for m, h in refs.items():
            if m in ckey:
                sc = ser[ckey[m]][idx].mean(0)
                out.append(b3.nearest_pool(sc, sc[h], excl, K))
            else:                                     # L12H5, L10H7: layer-matched
                l = b3.parse_head(m)[0]
                out.append(np.array([l * 16 + j for j in range(16) if l * 16 + j not in excl]))
        return out

    def rank(heads, pl, seed, n=2000):
        T = float(np.abs(P3[:, heads] - 0.5).mean())
        g = np.random.default_rng(seed)
        null = np.mean([np.abs(P3[:, p[g.integers(0, len(p), n)]] - 0.5).mean(0) for p in pl], 0)
        return b3.rank_of(T, null)

    c3 = {}
    for K in (5, 10, 20):
        real = pools({m: b3.head_index(m) for m in b3.MEMBERS}, mem, K)
        g = np.random.default_rng(seed_for("b3-var-c3", K))
        r = []
        for k in range(N_FAKE):
            fake = {m: int(real[i][g.integers(0, len(real[i]))]) for i, m in enumerate(b3.MEMBERS)}
            r.append(rank(list(fake.values()), pools(fake, mem + list(fake.values()), K), seed_for("v", K, k)))
        c3[f"recentred_K{K}"] = float(np.mean(np.array(r) >= b3.RANK_PASS))
    for K in (5, 10):
        real = pools({m: b3.head_index(m) for m in b3.MEMBERS}, mem, K)
        g = np.random.default_rng(seed_for("b3-var-c3x", K))
        r = []
        for k in range(N_FAKE):
            pick = [int(p[g.integers(0, len(p))]) for p in real]
            pl = [np.array([h for h in p if h != f]) for p, f in zip(real, pick)]
            r.append(rank(pick, pl, seed_for("vx", K, k)))
        c3[f"exchangeable_K{K}"] = float(np.mean(np.array(r) >= b3.RANK_PASS))
    for K in (5, 10):                             # the fake stays in its own pool (finding 2)
        real = pools({m: b3.head_index(m) for m in b3.MEMBERS}, mem, K)
        g = np.random.default_rng(seed_for("b3-var-c3i", K))
        r = []
        for k in range(N_FAKE):
            pick = [int(p[g.integers(0, len(p))]) for p in real]
            r.append(rank(pick, real, seed_for("vi", K, k)))
        c3[f"inclusive_K{K}"] = float(np.mean(np.array(r) >= b3.RANK_PASS))
    rec["C3p_false_pass"] = c3
    rec["n_fake"] = N_FAKE
    rec["code"] = b3.code_sha()
    rec["code_variants"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    (out / "gate_variants.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
