"""Closed forms behind `MATH_TRAINING_SYMMETRY.md` §10.1 (E0's backward pass; Parked 1).

D12  Adam's bias correction one step ahead of its moments. With n equal gradients g in the
     moments and the counter at t, Adam's update is
         u(n, t) = [(1 - b1^n)/(1 - b1^t)] / sqrt[(1 - b2^n)/(1 - b2^t)]  (times sign g; eps -> 0).
     Textbook: t = n, u = 1. One zero-gradient step before training (DeepSpeed v0.7.0's ZeRO
     `initialize_optimizer_states` calls optimizer.step() on zero grads; zeros leave m and v at 0
     but advance the counter) gives t = n + 1. NeoX's ramp runs iteration 0 at rate 0, so the
     first moving update is n = 2, t = 3: u = 0.848 (observed 0.852 on 70m's last writer bias).
     Over (2, 4], (4, 8], (8, 16] with rate ∝ i: 0.916, 0.964, 0.991 (observed 0.92, 0.96, 0.99).
D13  Where E0's gradient comes from. dS_ik = P_ik (dP_ik - d_i) with d_i the row's subtracted
     term. The non-rotary key bias b enters every score of row i as q_i·b, so
         dL/db = sum_i (sum_k dS_ik) q_i / sqrt(d),   sum_k dS_ik = sum_k P_ik dP_ik - d_i sum_k P_ik.
     With d_i = sum_k P_ik dP_ik and sum_k P_ik = 1 it is 0 (exact arithmetic). FlashAttention
     uses d_i = rowsum(dO_i ∘ O_i) with O stored in low precision, so the residual is
     dO_i·(O_i^exact - O_i^stored): any bias in that rounding lands on dL/db.

Not proved here: that the gradients after iteration 1 are steady (D12's later intervals assume
it); that Pythia's DeeperSpeed fork kept v0.7.0's setup step (only upstream v0.7.0 was read);
that the rounding residual of D13 has a nonzero mean (that is measured, §10.1, not derived).
"""
import sympy as sp

b1, b2 = sp.Rational(9, 10), sp.Rational(95, 100)


def u(n, t):
    return ((1 - b1 ** n) / (1 - b1 ** t)) / sp.sqrt((1 - b2 ** n) / (1 - b2 ** t))


def interval(a, b, off):
    """Rate-weighted mean u over iterations a..b-1 (rate ∝ i), counter = i + 1 + off."""
    return sum(i * u(i + 1, i + 1 + off) for i in range(a, b)) / sum(range(a, b))


def main():
    # D12
    assert u(2, 2) == 1 and all(sp.simplify(u(n, n) - 1) == 0 for n in range(1, 6))
    first = float(u(2, 3))
    assert abs(first - 0.84797) < 1e-4, first
    # a zero gradient in the moments is the same thing as the counter one ahead
    m = (1 - b1) * (b1 ** 2 * 0 + b1 * 1 + 1)
    v = (1 - b2) * (b2 ** 2 * 0 + b2 * 1 + 1)
    assert sp.simplify((m / (1 - b1 ** 3)) / sp.sqrt(v / (1 - b2 ** 3)) - u(2, 3)) == 0
    pred = [round(float(interval(a, b, 1)), 3) for a, b in ((1, 2), (2, 4), (4, 8), (8, 16))]
    obs = [0.85, 0.92, 0.96, 0.99]                          # §9.3, ts_first_step.py
    assert pred == [0.848, 0.916, 0.964, 0.991], pred
    assert all(abs(p - o) <= 0.005 for p, o in zip(pred, obs))
    # the two-step reading of §9.3 (g2 = r g1, counter = n) needed r in [3.4, 8.2]; at r = 1 it is 1
    assert u(2, 2) == 1
    # D13: symbolic row of 3 keys
    P = sp.symbols("p0:3", positive=True)
    dP = sp.symbols("e0:3")
    d = sp.Symbol("d")
    res = sum(P[k] * (dP[k] - d) for k in range(3))
    exact = res.subs(d, sum(P[k] * dP[k] for k in range(3))).subs(P[2], 1 - P[0] - P[1])
    assert sp.simplify(exact) == 0
    assert sp.simplify(res.subs(P[2], 1 - P[0] - P[1]) - (sum(P[k] * dP[k] for k in range(3)) - d)
                       .subs(P[2], 1 - P[0] - P[1])) == 0
    print("D12 first update", round(first, 5), "intervals", pred, "observed", obs)
    print("D13 residual = sum_k P dP - d (zero iff d is the exact row sum): ok")


if __name__ == "__main__":
    main()
