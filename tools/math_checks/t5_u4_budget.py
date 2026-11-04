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
B6  (after `/challenge-pr` on #306, finding 2) An LR arm cannot be read at matched cumulative LR
    inside Dodecahedron's schedule: A0's cumulative LR to 4369 is 3469.7 peak-steps, while an arm
    at half the peak has 2384.0 over the whole schedule to 9536 (at 1e-3, 0.286 of the peak,
    1362.3). Its whole run never reaches A0's dose at the read.
B7  (finding 1) The delay check continues A0 and up to two arms from 4369 to 5970: 1601 steps,
    0.839e9 tokens an arm. Four arms plus three continuations fit in 72 h at θ ≥ 45.1k tokens/s;
    four arms alone at θ ≥ 35.4k (B2).
B8  (finding 3) A rough throughput ceiling: about 6 N + 12 L d n_ctx = 339.8 MFLOP a token
    (N 31.47M, attention at 2048), so 60k tokens/s is 20.4 TFLOP/s, 34 % of the RTX 3080's
    59.5 TFLOP/s dense bf16 with float32 accumulate (NVIDIA's sheet, not measured). With d 384 the
    matmuls are small, so θ of 25–45k is as likely as 60k: the rows of B2 below 44.2k are live.
B9  (Blocked 132 D5 (b), the user's, 2026-11-03: A3 added and the cap raised to 4 GPU-days, 96 h)
    Five arms plus four continuations (A0, A1, A2, A3) fit in 96 h at θ ≥ 42.9k tokens/s; five
    arms alone at θ ≥ 33.1k. A3 and its continuation alone cost 3.13e9 tokens (19.3 h at 45.1k).
B10 The interaction I = ΔL_A3 − ΔL_A1 − ΔL_A2 = L_A1 + L_A2 − L_A0 − L_A3 (positive: together
    lower L by more than the two changes' sum). With each arm's L
    independent with SD σ, SD(I) = 2σ, while a pair difference (τ's |L_A0 − L_A0′|) has SD √2 σ:
    I's noise is √2 times a pair's, so its bar is √2 · max(0.1, 2τ), floor included (/challenge-pr
    on #307, finding 3: an unscaled floor 0.1 doubles the null's false 'interaction' rate).
B11 (Blocked 133, the user's, 2026-11-03: (i) (d), floor 0.2 and a second replicate seed A0″ trained to
    5970 with its continuation) Six arms plus five continuations (A0, A0″, A1, A2, A3) fit in 96 h at
    θ ≥ 51.9k tokens/s; six arms alone at θ ≥ 39.8k. At the measured 104.8k: 47.6 h (39.3 h
    before). The floor: bar = max(0.2, 2τ), bar_I = √2 · bar, so bar_I's floor 0.2828; on
    Dodecahedron's R/2 = 0.338 the rule stays readable while τ < 0.169. τ is the largest of the
    three pairs, max(L) − min(L). With 410m's split (2 of 10 seeds low, so p = 0.2), one pair lands
    in a single group with probability 0.68 and three seeds 0.52, so a third seed raises the
    chance of seeing a split by 0.16.

NOT proved: the throughput (B8 is a ceiling argument, not a measurement). 60k tokens/s is the ROADMAP's rough guess, never measured on this
GPU; U4a measures it and the arm count follows B9's thresholds (B2's at the old 3-day cap). Nothing here says the rise will
appear at 524k a step: that is A0's positive control. B10 assumes the arms' L are independent with
equal SD; two replicate seeds (A0′, A0″; B11) estimate that SD, so the bar is only as good as that estimate.
B11's 0.68 / 0.52 treat 410m's 2-in-10 split as the probability for Dodecahedron's recipe, which is not known.
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

# B6
def cum(step, peak_frac=1):
    """Cumulative LR in units of the peak, summed over steps 1 … step (float: a 9536-term sum)."""
    import math
    tot = 0.0
    for t in range(1, step + 1):
        if t <= WARM:
            tot += t / WARM
        else:
            tot += (1 + math.cos(math.pi * (t - WARM) / (TOTAL - WARM))) / 2
    return tot * float(peak_frac)


cA0 = float(cum(READ))
c_half = float(cum(TOTAL, sp.Rational(1, 2)))
c_1e3 = float(cum(TOTAL, sp.Rational(1, 1) * sp.Rational(1000, 3500)))
assert abs(cA0 - 3469.7) < 0.1 and abs(c_half - 2384.0) < 0.1 and c_half < cA0 and c_1e3 < cA0
print(f"B6 cumulative LR / peak: A0 to {READ} {cA0:.1f}; half peak to {TOTAL} {c_half:.1f}; 1e-3 to {TOTAL} {c_1e3:.1f}")

# B7
CONT_TO = 5970
cont = (CONT_TO - READ) * B_CARD
need = sp.Rational(4 * tok_card + 3 * cont, CAP_S)
assert CONT_TO - READ == 1601 and round(float(need)) == 45_064
print(f"B7 continuation {READ}->{CONT_TO}: {cont:.4g} tokens an arm; 4 arms + 3 continuations need θ ≥ {float(need):.0f}")

# B8
n_ctx = 2048
flop_tok = 6 * n + 12 * Lyr * d * n_ctx
tflops = flop_tok * 60_000 / 1e12
assert abs(flop_tok / 1e6 - 339.8) < 0.1 and abs(tflops / 59.5 - 0.343) < 0.01
print(f"B8 {flop_tok / 1e6:.1f} MFLOP a token; 60k tokens/s = {tflops:.1f} TFLOP/s = {tflops / 59.5:.0%} of 59.5")

# B9
CAP4_S = 4 * 24 * 3600
need5c = sp.Rational(5 * tok_card + 4 * cont, CAP4_S)
need5 = sp.Rational(5 * tok_card, CAP4_S)
assert round(float(need5c)) == 42_855 and round(float(need5)) == 33_140
a3_h = float(sp.Rational(tok_card + cont, 1) / 45_064 / 3600)
assert abs(a3_h - 19.29) < 0.01
print(f"B9 96 h: 5 arms + 4 continuations need θ ≥ {float(need5c):.0f}; 5 arms ≥ {float(need5):.0f}; "
      f"A3 + its continuation {(tok_card + cont):.4g} tokens, {a3_h:.1f} h at 45.1k")

# B10
sig = sp.symbols("sigma", positive=True)
L0, L0p, L1, L2, L3 = sp.symbols("L0 L0p L1 L2 L3")
I = (L0 - L3) - (L0 - L1) - (L0 - L2)  # ΔL_A3 − ΔL_A1 − ΔL_A2, ΔL_k = L_A0 − L_k
assert sp.expand(I - (L1 + L2 - L0 - L3)) == 0
var = lambda expr: sum(sp.diff(expr, v) ** 2 for v in (L0, L1, L2, L3, L0p)) * sig ** 2
ratio = sp.sqrt(var(I)) / sp.sqrt(var(L0 - L0p))
assert sp.simplify(ratio - sp.sqrt(2)) == 0
bar_I = lambda tau: sp.sqrt(2) * sp.Max(sp.Rational(1, 10), 2 * tau)
assert sp.simplify(bar_I(sp.Rational(1, 100)) - sp.sqrt(2) / 10) == 0  # the floor scales too
print(f"B10 SD(I) / SD(pair difference) = {ratio}; bar_I = sqrt2 * max(0.1, 2 tau), floor {float(bar_I(0)):.4f}")

# B11
need6c = sp.Rational(6 * tok_card + 5 * cont, CAP4_S)
need6 = sp.Rational(6 * tok_card, CAP4_S)
assert round(float(need6c)) == 51_911 and round(float(need6)) == 39_768
h_at = lambda arms, conts, th: float(sp.Rational(arms * tok_card + conts * cont, 1) / th / 3600)
assert abs(h_at(6, 5, 104_800) - 47.55) < 0.01 and abs(h_at(5, 4, 104_800) - 39.26) < 0.01
bar = lambda tau: sp.Max(sp.Rational(2, 10), 2 * tau)
assert sp.simplify(sp.sqrt(2) * bar(0) - sp.sqrt(2) / 5) == 0 and abs(float(sp.sqrt(2) / 5) - 0.2828) < 1e-4
R2 = (sp.Rational(7875, 10000) - sp.Rational(111, 1000)) / 2      # B4's R / 2
assert bar(sp.Rational(169, 1000)) < R2 <= bar(sp.Rational(170, 1000))
pr = sp.Rational(2, 10)
same2, same3 = (1 - pr) ** 2 + pr ** 2, (1 - pr) ** 3 + pr ** 3
assert (same2, same3) == (sp.Rational(68, 100), sp.Rational(52, 100))
print(f"B11 96 h: 6 arms + 5 continuations need θ ≥ {float(need6c):.0f}; 6 arms ≥ {float(need6):.0f}; "
      f"at 104.8k {h_at(6, 5, 104_800):.1f} h (was {h_at(5, 4, 104_800):.1f}); bar_I floor {float(sp.sqrt(2) / 5):.4f}; "
      f"readable while τ < {float(R2) / 2:.4f}; one pair in one group {same2}, three seeds {same3}")
