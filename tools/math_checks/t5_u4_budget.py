"""Checks the arithmetic of T5 U4's design (`trackT_timaeus/design-T5.md` "U4, the causal arm,
designed"; `STATE.md` Blocked 132). Inputs are the design's: Dodecahedron's card (524,288 tokens a
step; T1 Q3's leaning 4,194,304), its LR schedule (linear warmup from 0 to 3.5e-3 over 1000 steps,
cosine to step 9536, floor taken as 0), U2's ρ_mid record at `-32m` 917, 3738, 4369, the 3 GPU-day
cap and `dodecahedron-32m`'s `config.json`.

B1  Tokens per arm to step 4369: 2.29e9 at 524,288 a step; 1.83e10 at 4,194,304.
B2  Hours per arm at a throughput θ (tokens/s), and the least θ at which k arms fit in 72 h:
    5 arms 44.2k, 4 arms 35.4k, 3 arms 26.5k. At 4,194,304 a step one arm reaches only
    ~ 72 h · θ / 4.19M steps in the whole cap (θ 60k: 3707, inside the rise but short of 4369
    for one arm alone, and 741 per arm for five, inside the trough).
B3  The LR at 4369 is 0.662 of the peak with floor 0 (0.696 with floor 0.1): the floor, which the
    card does not give, moves the read window's LR by 5 %.
B4  Dodecahedron's reference numbers for the rule: L = mean ρ_mid at 3738 and 4369 = 0.7875;
    T = 0.111 (917, the minimum over 100 … 4369); R = L − T = 0.6765; R / 2 = 0.338.
B5  Parameters of the 32m arm (tied unembedding): 31,469,952 (125.9 MB in float32); biases on
    every linear add 3456 a block (55,296).

NOT proved: the throughput. 60k tokens/s is the ROADMAP's rough guess, never measured on this
GPU; U4a measures it and the arm count follows B2's thresholds. Nothing here says the rise will
appear at 524k a step: that is A0's positive control.
"""

import sympy as sp

PEAK, WARM, TOTAL, READ = sp.Rational(35, 10000), 1000, 9536, 4369
B_CARD, B_LEAN = 524_288, 4_194_304
CAP_S = 3 * 24 * 3600

# B1
tok_card = READ * B_CARD
tok_lean = READ * B_LEAN
assert tok_card == 2_290_614_272 and tok_lean == 18_324_914_176
print(f"B1 tokens per arm to {READ}: {tok_card:.4g} (524k), {tok_lean:.4g} (4.19M)")

# B2
for theta in (30_000, 45_000, 60_000):
    print(f"B2 θ {theta}: {tok_card / theta / 3600:.1f} h per arm at 524k")
thr = {k: sp.Rational(k * tok_card, CAP_S) for k in (5, 4, 3)}
for k, v in thr.items():
    print(f"B2 {k} arms in 72 h need θ ≥ {float(v):.0f} tokens/s")
assert round(float(thr[5])) == 44_186 and round(float(thr[4])) == 35_349 and round(float(thr[3])) == 26_512
reach_one = CAP_S * 60_000 // B_LEAN
assert reach_one == 3707 and reach_one < READ
print(f"B2 at 4.19M a step and θ 60k: one arm reaches step {reach_one} in the whole cap; "
      f"five share it, {reach_one // 5} each")


# B3
def lr(step, floor=0):
    if step <= WARM:
        return PEAK * sp.Rational(step, WARM)
    p = sp.Rational(step - WARM, TOTAL - WARM)
    return PEAK * (floor + (1 - floor) * (1 + sp.cos(sp.pi * p)) / 2)


f0 = float(lr(READ) / PEAK)
f1 = float(lr(READ, sp.Rational(1, 10)) / PEAK)
assert abs(f0 - 0.6627) < 1e-3 and abs(f1 - 0.6965) < 1e-3
print(f"B3 LR at {READ} / peak: {f0:.4f} (floor 0), {f1:.4f} (floor 0.1)")

# B4 (U2's record, `status-T.md` "T5 U2", "T5 U3, first read")
rho = {917: sp.Rational(111, 1000), 3738: sp.Rational(763, 1000), 4369: sp.Rational(812, 1000)}
Lv = (rho[3738] + rho[4369]) / 2
Rv = Lv - rho[917]
assert Lv == sp.Rational(7875, 10000) and Rv == sp.Rational(6765, 10000)
print(f"B4 L {float(Lv):.4f}, T {float(rho[917]):.3f}, R {float(Rv):.4f}, R/2 {float(Rv / 2):.4f}")

# B5 (`dodecahedron-32m` config.json: 16 layers, 384, 6 heads / 2 KV of 64, SwiGLU 1024, vocab 16384, tied)
d, ff, kv, Lyr, V = 384, 1024, 2 * 64, 16, 16384
attn = d * d + d * kv + d * kv + d * d
mlp = 3 * d * ff
per_block = attn + mlp + 2 * d
n = V * d + Lyr * per_block + d
assert n == 31_469_952
bias_block = d + kv + kv + d + ff + ff + d
assert bias_block == 3456
print(f"B5 params {n} ({n * 4 / 1e6:.1f} MB fp32); biases {bias_block} a block, {Lyr * bias_block} in all")
