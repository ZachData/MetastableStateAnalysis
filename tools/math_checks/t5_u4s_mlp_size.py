"""The MLP's write size at init across U4s's `gelu` split (`trackT_timaeus/status-T.md` "T5 U4s";
`/challenge-pr` on #308, finding 3: "the gate" may stand for the write's size).

S1  A normed input x (unit RMS per coordinate, gains 1) through an input map of SD sd_in gives
    pre-activations z ~ N(0, s²), s = sd_in · sqrt(d), independent across units. P: sd_in =
    small_init(d) = sqrt(2 / (5 d)), so s = sqrt(2/5) ≈ 0.632 at any d; F: 0.02 · sqrt(384) ≈ 0.392.
S2  The write's RMS per coordinate is sd_out · sqrt(width · E[h²]), with h = act(z) ungated and
    h = act(g) · u gated (g, u independent, each N(0, s²)), so E[h²] = E[act(z)²] · s² gated.
    sd_out is the same for every MLP form under one init (wang_init or 0.02), so the ratio of two
    forms' writes is sqrt(width_a E_a[h²] / (width_b E_b[h²])), and the base (gated SiLU 1024) is 1.

Printed: each of U4s's MLP cells' write RMS relative to the base, under P and F, beside its ΔE
share from the records (copied here, not recomputed).

NOT proved: that the write's size, rather than the gate, moves ρ. This is one block's MLP on a
Gaussian input at init (real inputs are the stream after attention, not Gaussian, and the stream
grows with depth), and it says nothing about direction or how shared the write is. It only shows
whether the split's cells differ in write size enough to confound "the gate"; a size-matched
control (Parked 2 in "T5 U4s") would tell them apart.
"""
import math

from scipy import integrate, stats

D = 384


def silu(z):
    return z / (1 + math.exp(-z))


def gelu(z):
    return z * stats.norm.cdf(z)


def second_moment(act, s):
    f = lambda z: act(z) ** 2 * stats.norm.pdf(z, scale=s)
    return integrate.quad(f, -12 * s, 12 * s, limit=200)[0]


def write_rms(gated, act, width, s):
    m = second_moment(act, s) * (s * s if gated else 1.0)
    return math.sqrt(width * m)


CELLS = {  # name: (gated, act, width, ΔE share of E0 from data/trackT/t5u4s/2026-11-03/reading.json)
    "base (gated SiLU 1024)": (True, silu, 1024, 0.0),
    "gate (SiLU 1024)": (False, silu, 1024, 0.45),
    "act (gated GeLU 1024)": (True, gelu, 1024, -0.15),
    "width (gated SiLU 1536)": (True, silu, 1536, -0.34),
    "gate+act (GeLU 1024)": (False, gelu, 1024, 0.49),
    "gate+width (SiLU 1536)": (False, silu, 1536, 0.32),
    "act+width (gated GeLU 1536)": (True, gelu, 1536, -0.44),
    "gelu (GeLU 1536)": (False, gelu, 1536, 0.45),
}


def main():
    s_P, s_F = math.sqrt(2 / 5), 0.02 * math.sqrt(D)
    assert abs(math.sqrt(2 / (5 * D)) * math.sqrt(D) - s_P) < 1e-15
    # sanity: SiLU's second moment at small s tends to s²/4 (silu(z) ≈ z/2)
    assert abs(second_moment(silu, 1e-3) / (1e-6 / 4) - 1) < 1e-3
    base = {k: write_rms(True, silu, 1024, s) for k, s in (("P", s_P), ("F", s_F))}
    print(f"s_P {s_P:.3f}, s_F {s_F:.3f}")
    print(f"{'cell':30s} {'P':>6s} {'F':>6s}  share of E0")
    for name, (g, a, w, sh) in CELLS.items():
        r = {k: write_rms(g, a, w, s) / base[k] for k, s in (("P", s_P), ("F", s_F))}
        print(f"{name:30s} {r['P']:6.2f} {r['F']:6.2f}  {sh:+.2f}")
    ung = write_rms(False, silu, 1024, s_F) / base["F"]
    assert 2.0 < ung < 3.5, ung       # ungating multiplies the write at F by about 1 / s_F · const


if __name__ == "__main__":
    main()
