"""Closed form behind `MATH_TRAINING_SYMMETRY.md` §11.1 (b_K zeroed; `/challenge-pr` on #357, finding 1).

D14  Why §11's answer rule had one reachable outcome. Let z be the exact logits, and let the intact
     and zeroed arms' logits be z + eps a and z + eps b (a, b their rounding errors). With p = softmax(z),
     to second order in eps
         KL(p_z || p_{z+eps a}) = (eps^2 / 2) Var_p(a)
         KL(p_{z+eps a} || p_{z+eps b}) = (eps^2 / 2) Var_p(a - b)
     so, with e_I = KL(exact || intact), e_Z = KL(exact || zeroed), d = KL(intact || zeroed),
         r = d / e_I = 1 + e_Z / e_I - rho,   rho = 2 Cov_p(a, b) / Var_p(a).
     b_K is the floor (b ~ 0):          r -> 1.
     b_K irrelevant, errors re-rolled (Var b = Var a, Cov = 0):  r -> 2.
     Both land in "comparable" (1 < r < 10) of §11's rule, which mapped to "matters".

Checked below on a 3-token softmax with symbolic logits and errors. Not proved here: that the
recorded KLs are in the second-order regime (on 410m R32 the formula gives r = 1.0043 - rho
against 1.0033 recorded, so rho ~ 0.001; on the fp16 arms e_I ~ 3e-2 nats, where the third-order
terms are not bounded here), and that per-token averages keep the identity (it holds token by
token, so the ratio of averages is 1 + mean e_Z / mean e_I - a weighted rho).
"""
import sympy as sp

eps = sp.symbols("eps")
z = sp.symbols("z0:3", real=True)
a = sp.symbols("a0:3", real=True)
b = sp.symbols("b0:3", real=True)


def logsm(v):
    lse = sp.log(sum(sp.exp(x) for x in v))
    return [x - lse for x in v]


def kl(u, v):
    lu, lv = logsm(u), logsm(v)
    return sum(sp.exp(x) * (x - y) for x, y in zip(lu, lv))


def second_order(expr):
    return sp.simplify(sp.diff(expr, eps, 2).subs(eps, 0) / 2)


def var_p(w):
    p = [sp.exp(x) for x in logsm(z)]
    m = sum(pi * wi for pi, wi in zip(p, w))
    return sum(pi * (wi - m) ** 2 for pi, wi in zip(p, w))


def main() -> int:
    za = [zi + eps * ai for zi, ai in zip(z, a)]
    zb = [zi + eps * bi for zi, bi in zip(z, b)]
    e_I = second_order(kl(list(z), za))
    d = second_order(kl(za, zb))
    want_e = var_p(a) / 2
    want_d = var_p([ai - bi for ai, bi in zip(a, b)]) / 2
    pt = {z[0]: sp.Rational(3, 10), z[1]: -1, z[2]: sp.Rational(7, 5),
          a[0]: 1, a[1]: -2, a[2]: sp.Rational(1, 2), b[0]: sp.Rational(-1, 3), b[1]: 2, b[2]: 1}
    for got, want, name in ((e_I, want_e, "e_I = Var_p(a)/2"), (d, want_d, "d = Var_p(a-b)/2")):
        diff = sp.N((got - want).subs(pt), 30)
        assert abs(diff) < 1e-25, (name, diff)
        print(f"D14 {name}: holds (residual {diff:.1e} at a rational point)")
    # the two regimes, as exact algebra on the variances
    Va, Vb, C = sp.symbols("Va Vb C", positive=True)
    r = (Va + Vb - 2 * C) / Va
    assert sp.simplify(r - (1 + Vb / Va - 2 * C / Va)) == 0
    print("D14 r = 1 + e_Z/e_I - rho: holds;",
          "floor (Vb -> 0, C -> 0):", sp.limit(r.subs(C, 0), Vb, 0),
          "| re-rolled (Vb = Va, C = 0):", r.subs({Vb: Va, C: 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
