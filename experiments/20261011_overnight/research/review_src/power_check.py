"""Review check: conditional binomial (exact) one-sided 5% power for two Poisson rates, n laps each."""
from math import comb, exp, lgamma, log
def pois(k, lam):
    return exp(k*log(lam) - lam - lgamma(k+1)) if lam > 0 else (1.0 if k == 0 else 0.0)
def binom_cdf(c, m):
    return sum(comb(m, i) for i in range(c+1)) / 2**m
def power(l1, l2, n, alpha=0.05):
    a, b = l1*n, l2*n
    tot = 0.0
    for m in range(0, int(a+b+12*(a+b)**0.5)+20):
        pm = pois(m, a+b)
        if pm < 1e-12 and m > a+b: break
        # critical c: largest c with P(X2<=c | m, p=.5) <= alpha
        c = -1
        while c+1 <= m and binom_cdf(c+1, m) <= alpha: c += 1
        if c < 0: continue
        p = b/(a+b)
        tot += pm * sum(comb(m, i)*p**i*(1-p)**(m-i) for i in range(c+1))
    return tot
for l2 in (0.75, 0.6, 0.4):
    print(1.2, '->', l2, [round(power(1.2, l2, n), 2) for n in (12, 24, 43, 60, 100)])
# what the abort rule does to a model that is exactly as good as baseline
import math
for lam in (1.24, 1.3, 1.4):
    mu = lam*10
    p_gt15 = 1 - sum(pois(k, mu) for k in range(16))
    print('abort P(>15 impacts in 10 laps | rate %.2f) = %.2f' % (lam, p_gt15))
