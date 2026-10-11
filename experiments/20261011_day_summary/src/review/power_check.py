"""レビュー用: 1回の A/B/A(A 44周・B 43周)で何が言えるかの検出力と、ラップの判定の幅(CPU・numpy だけ)。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    cd ~/jetformer_reports/summary_20261010/src/review && nice -n 15 python power_check.py > power_check.json

- 衝突/周: ポアソン、条件付き二項検定(片側 5%)、モンテカルロ。基準 1.17回/周(006/007、回復中を除く)。
- 割合(衝突なしの周の割合・R1 の強い衝突/通過): Fisher の正確検定(片側 5%)、モンテカルロ。
- ラップ: AI の R1 起点の周(explore_control/sections.json)の標準偏差から、差の 90% 信頼区間の半幅。
"""
import json, math, os
from functools import lru_cache
import numpy as np

rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(None)
def binom_cdf(k, n, p):
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log(1 - p)) for i in range(k + 1))


def pow_rate(rA, rB, nA, nB, sims=4000):
    a = rng.poisson(rA * nA, sims); b = rng.poisson(rB * nB, sims)
    p0 = nB / (nA + nB)
    return float(np.mean([binom_cdf(int(y), int(x + y), p0) < 0.05 for x, y in zip(a, b)]))


@lru_cache(None)
def hyper_sf(x, N, K, n):
    lo, hi = max(0, n - (N - K)), min(K, n)
    c = math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)
    def lp(i):
        return (math.lgamma(K + 1) - math.lgamma(i + 1) - math.lgamma(K - i + 1) + math.lgamma(N - K + 1)
                - math.lgamma(n - i + 1) - math.lgamma(N - K - n + i + 1) - c)
    return sum(math.exp(lp(i)) for i in range(max(x, lo), hi + 1))


def pow_prop_up(p1, p2, n1, n2, sims=3000):
    """群2の「成功」の割合が群1より高いことを片側で検出する確率。"""
    x1 = rng.binomial(n1, p1, sims); x2 = rng.binomial(n2, p2, sims)
    return float(np.mean([hyper_sf(int(b), n1 + n2, int(a + b), n2) < 0.05 for a, b in zip(x1, x2)]))


out = {"assumptions": "A 44周(前後 22+22)・B 43周。片側 5%。衝突はポアソン(同じセッション内の周ごとのばらつきはポアソンより小さい: ロードマップ課題8)。"}
out["collisions_per_lap_power_44v43"] = {f"1.17->{r}": pow_rate(1.17, r, 44, 43) for r in (0.9, 0.86, 0.7, 0.585, 0.5)}
need = {}
for r in (0.9, 0.86, 0.7, 0.585):
    for n in (43, 60, 80, 100, 120, 150, 200, 250, 300):
        p = pow_rate(1.17, r, n, n, 2500)
        if p >= 0.8:
            need[f"1.17->{r}"] = {"laps_per_arm": n, "power": p}; break
out["laps_per_arm_for_power_0.8"] = need
out["noninferiority_upper95_ratio_when_equal"] = {str(ev): round(math.exp(1.645 * math.sqrt(2 / ev)), 2) for ev in (25, 50, 100)}
out["clean_lap_fraction_power"] = {f"0.33->{p2} n={n}": pow_prop_up(0.33, p2, n, n) for p2, n in ((0.4, 43), (0.5, 43), (0.6, 43), (0.4, 100), (0.4, 200))}
# R1 の強い衝突/通過の低下 = 「当たらない」割合の上昇
out["r1_strong_per_pass_power"] = {f"0.25->0.10 n={n}": pow_prop_up(0.75, 0.90, n, n) for n in (43, 60, 100)}

sec = json.load(open(os.path.join(HERE, "..", "..", "explore_control", "sections.json")))
ai = [l for L in sec["logs"] for l in L["laps"] if l["driver"] == "ai"]
t_all = np.array([l["lap_s"] for l in ai]); t_clean = np.array([l["lap_s"] for l in ai if l["n_imp"] == 0])
lap = {"ai_laps": len(t_all), "sd_all_s": round(float(t_all.std(ddof=1)), 2),
       "ai_clean_laps": len(t_clean), "sd_clean_s": round(float(t_clean.std(ddof=1)), 2)}
for lab, sd, n in (("clean_15v15_sd0.16", 0.16, 15), ("clean_15v15_sd0.30", 0.30, 15), ("all_44v43_sd0.71", 0.71, 43)):
    se = sd * math.sqrt(2 / n)
    lap[lab] = {"ci90_halfwidth_s": round(1.645 * se, 2), "detectable_80pct_s": round(2.486 * se, 2)}
out["lap_time"] = lap
print(json.dumps(out, ensure_ascii=False, indent=1))
