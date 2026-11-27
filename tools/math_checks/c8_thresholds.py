"""C8's closed forms and arithmetic (trackC_writeup/design-C8.md, lit-C8.md §2).

Proves: (1) Cohen et al.'s adaptive-EoS threshold (2 + 2β₁)/((1 − β₁)η) is 38/η at β₁ = 0.9, and
its values at C8's rates (1e-3, 5e-4, 1e-4); (2) (i)'s windows (t⁻ − 16, t], less a branch's first
8 updates, and the background blocks each condition gets on its log span, all ≥ 4 (the refusal
bar), at the onsets earlier units recorded (F's (32, 48] and (48, 64], F½'s (56, 60] and (60, 64],
F⅒'s (752, 784], B_P's (272, 280] and (280, 288]); F¹'s own window gets < 4, which is why F reads
and F¹ stands beside; (3) the Π bar over factors (F and F½ merged, since their windows share
updates; /challenge-pr on #387, finding 1): F and F½'s windows overlap at every gated τ, F⅒'s and
B_P's overlap no other on C4's order; k factors at a common b give Π = b^k ≤ 0.05 iff
b ≤ 0.05^(1/k), ≈ 0.473 at k = 4 ("marked") and ≈ 0.368 at k = 3 ("marked at ≥ half the peak");
(4) Adam's second moment at β₂ = 0.95 after k zero-rate updates holds
1 − β₂^k of its weight on the gradients seen, ≈ 0.963 at k = 64 (U3's warmed arm), with time
constant 1/(1 − β₂) = 20 updates; β₁'s at k = 64 is ≈ 0.9988.

Does NOT prove: that the threshold holds for minibatch Adam at C4's batch (Cohen et al. is
full-batch; lit-C8 Parked 2), that any event marks the onset, that the background is stationary
over an arm's log, or that the conditions are independent (Π assumes it; F, F¹ and F¹'s
replicate are not, so only F reads). F′'s onset is computed in U1 and is not checked here.
"""
from fractions import Fraction as Fr

import sympy as sp

b1, eta = sp.symbols("beta1 eta", positive=True)
thr = (2 + 2 * b1) / ((1 - b1) * eta)
assert sp.simplify(thr.subs(b1, sp.Rational(9, 10)) - 38 / eta) == 0
for lr, want in ((sp.Rational(1, 1000), 38000), (sp.Rational(1, 2000), 76000), (sp.Rational(1, 10000), 380000)):
    assert thr.subs({b1: sp.Rational(9, 10), eta: lr}) == want

LEAD, RESTART, GUARD, MIN_BLOCKS = 16, 8, 16, 4


def window(tm, t, branch=None):
    lo = tm - LEAD
    if branch is not None:
        lo = max(lo, branch + RESTART)
    return lo, t                                   # updates s with lo < s <= t


def blocks(tm, t, span, branch=None):
    lo, hi = window(tm, t, branch)
    n = hi - lo
    a, b = span
    count, s = 0, a
    while s + n - 1 <= b:                           # block [s, s + n − 1]
        e = s + n - 1
        if e <= lo - GUARD or s > hi + GUARD:
            count += 1
        s += n
    return n, count


cases = {  # condition: (t⁻, t, log span, branch)
    "F 9.61e-5/1e-4": (32, 48, (9, 1000), None),
    "F 3e-4": (48, 64, (9, 1000), None),
    "Fh 9.61e-5/1e-4": (56, 60, (9, 1000), None),
    "Fh 3e-4": (60, 64, (9, 1000), None),
    "Ft": (752, 784, (9, 1000), None),
    "BP 9.61e-5/1e-4": (272, 280, (265, 512), 256),
    "BP 3e-4": (280, 288, (265, 512), 256),
}
got = {k: blocks(*v[:3], branch=v[3]) for k, v in cases.items()}
assert got["F 9.61e-5/1e-4"] == (32, 29)
assert got["Fh 9.61e-5/1e-4"][0] == 20 and got["Ft"][0] == 48
assert got["BP 9.61e-5/1e-4"] == (16, 13) and got["BP 3e-4"][0] == 24
assert all(c >= MIN_BLOCKS for _, c in got.values()), got
assert blocks(44, 48, (9, 128))[1] < MIN_BLOCKS     # F¹ on its own 128-update log: refused

def wins(key):
    tm, t, _, br = cases[key]
    return window(tm, t, br)


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


for f, h in (("F 9.61e-5/1e-4", "Fh 9.61e-5/1e-4"), ("F 3e-4", "Fh 3e-4"),
             ("F 9.61e-5/1e-4", "Fh 3e-4"), ("F 3e-4", "Fh 9.61e-5/1e-4")):
    assert overlap(wins(f), wins(h))                    # F and F½ share updates: one factor
for f in ("F 9.61e-5/1e-4", "F 3e-4", "Fh 9.61e-5/1e-4", "Fh 3e-4", "BP 9.61e-5/1e-4", "BP 3e-4"):
    assert not overlap(wins("Ft"), wins(f))
for f in ("F 9.61e-5/1e-4", "F 3e-4", "Fh 9.61e-5/1e-4", "Fh 3e-4"):
    for bp in ("BP 9.61e-5/1e-4", "BP 3e-4"):           # B_P's update s reads A0's batch s − 1
        lo, hi = wins(bp)
        assert not overlap(wins(f), (lo - 1, hi - 1))
assert 0.4728 < 0.05 ** 0.25 < 0.4729 and 0.3684 < 0.05 ** (1 / 3) < 0.3685
assert Fr(47, 100) ** 4 <= Fr(1, 20) and Fr(37, 100) ** 3 > Fr(1, 20)

k = 64
assert abs((1 - 0.95 ** k) - 0.9625) < 1e-3 and abs((1 - 0.9 ** k) - 0.99882) < 1e-4
assert sp.Rational(1, 1) / (1 - sp.Rational(95, 100)) == 20

if __name__ == "__main__":
    for kk, v in got.items():
        print(f"{kk:18s} window {v[0]:3d}  background blocks {v[1]:3d}")
    print("ok")
