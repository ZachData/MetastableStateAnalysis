"""Checks the init scales T5's H1 rests on (`docs/ROADMAP.md` §7b row T5, proposed 2026-10-30;
the SDs measured by `trackT_timaeus/t5_init_sd.py`, recorded in `trackT_timaeus/status-T.md`
"T5's proposal: the init scales measured").

I1  GPT-NeoX's small_init sqrt(2 / (5 d)) and wang_init 2 / (L sqrt(d)) give the measured step-0
    SDs of pythia-410m (d 1024, L 24: 0.0198, 0.0026) and pythia-70m (d 512, L 6: 0.0280,
    0.0147), and GPT-2's 0.02 / sqrt(2 L) gives Mistral GPT-2 small's output projections (d 768,
    L 12: 0.0041). Dodecahedron uses 0.02 on every weight (measured; `initializer_range` 0.02).
I2  The rough per-block ratio of a block's write to the embedding, r = sd_out · sd_in · d / sd_emb
    (a normed input has norm sqrt(d); each d × d Gaussian map multiplies a norm by sd · sqrt(d);
    the embedding has norm sd_emb · sqrt(d)), and r · L as an upper bound on the depth sum (every
    write in one direction): printed for the five shapes. Per block, 70m (7.5) is as large as
    Dodecahedron-32m (7.7); over depth, Dodecahedron stands out (123, 205 against 38–64).

NOT proved: that r predicts the cloud's mean cosine at init. r ignores attention's averaging
(which shrinks a write's norm and makes it shared), the MLP, the norm gains and the 4d or SwiGLU
widths, and treats the weights as independent Gaussians. It only says which shapes H1's mechanism
would mark, so 70m's step-0 cloud is the cheap test of H1 in its per-block form
(`/challenge-pr` on #289, finding 1).
"""

import sympy as sp

d, L = sp.symbols("d L", positive=True)
SMALL = sp.sqrt(sp.Rational(2, 5) / d)
WANG = 2 / (L * sp.sqrt(d))
GPT2_OUT = sp.Rational(2, 100) / sp.sqrt(2 * L)

# (name, d, L, sd_emb, sd_in, sd_out); the measured SDs to 3 significant figures
SHAPES = [
    ("pythia-410m", 1024, 24, SMALL, SMALL, WANG),
    ("pythia-160m", 768, 12, SMALL, SMALL, WANG),
    ("pythia-70m", 512, 6, SMALL, SMALL, WANG),
    ("gpt2-small (Mistral)", 768, 12, sp.Rational(2, 100), sp.Rational(2, 100), GPT2_OUT),
    ("dodecahedron-32m", 384, 16, sp.Rational(2, 100), sp.Rational(2, 100), sp.Rational(2, 100)),
    ("dodecahedron-67m", 512, 20, sp.Rational(2, 100), sp.Rational(2, 100), sp.Rational(2, 100)),
]
MEASURED = {  # (sd_in, sd_out) from the checkpoints' weights
    "pythia-410m": (0.0198, 0.0026),
    "pythia-70m": (0.0280, 0.0147),
    "gpt2-small (Mistral)": (0.0200, 0.0041),
    "dodecahedron-32m": (0.0200, 0.0200),
}


def i1() -> None:
    for name, dv, Lv, _, s_in, s_out in SHAPES:
        if name not in MEASURED:
            continue
        got = tuple(float(x.subs({d: dv, L: Lv})) for x in (s_in, s_out))
        want = MEASURED[name]
        assert all(abs(g - w) / w < 0.02 for g, w in zip(got, want)), (name, got, want)
    print("I1 ok: the init formulas give the measured step-0 SDs (within 2 %)")


def i2() -> None:
    r_sym = sp.Symbol("sd_out") * sp.Symbol("sd_in") * d / sp.Symbol("sd_emb")
    print(f"I2: r = {r_sym}; r and r·L per shape")
    rs = {}
    for name, dv, Lv, s_emb, s_in, s_out in SHAPES:
        sub = {d: dv, L: Lv}
        r = float((s_out * s_in * d / s_emb).subs(sub))
        rs[name] = r
        print(f"   {name:22s} d {dv:4d} L {Lv:2d}  r {r:5.2f}  r·L {r * Lv:6.1f}")
    assert abs(rs["pythia-70m"] - rs["dodecahedron-32m"]) / rs["dodecahedron-32m"] < 0.05
    assert rs["pythia-410m"] < rs["pythia-160m"] < rs["pythia-70m"]
    print("I2 ok: per block 70m is within 5 % of Dodecahedron-32m; Pythia's r falls with width")


if __name__ == "__main__":
    i1()
    i2()
