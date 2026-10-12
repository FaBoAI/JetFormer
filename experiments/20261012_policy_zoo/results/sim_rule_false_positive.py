"""R1・DE の「良い」の規則が、学習のばらつきだけ(本当の差 0)で偶然に通る割合の見積もり(CPU・numpy・数秒)。
模型: 学習結果 r の通過 p の出力 = 共通の値 + o_r(場所によらない全体のずれ = 学習のばらつき、SD σ_t)+ u_rp(通過ごとの揺れ、SD σ_u)。
シーンの平均を引いた差(D2)は o_r を消すと仮定(実際は一部しか消えない = 甘めの見積もり)。
旧: d̄ = P̄ − R̄ の 95% 区間(通過のブートストラップ)の下限 > 0 かつ 2 つの種の両方で d > 0。
新: 99.375% 区間(d̄ と、全体のずれを除いた d̄)の下限 > 0 かつ 4 組すべて d_ij ≥ 10 µs。
"""
import json
import numpy as np


def one(rng, n_pass, s_t, s_u, B=2000):
    o = rng.normal(0, s_t, 4)                                # P0 P1 R0 R1 の全体のずれ
    u = rng.normal(0, s_u, (4, n_pass))
    y = o[:, None] + u
    ym = u                                                   # 全体のずれを除いた(D2 の理想)
    d = (y[0] + y[1]) / 2 - (y[2] + y[3]) / 2
    dm = (ym[0] + ym[1]) / 2 - (ym[2] + ym[3]) / 2
    idx = rng.integers(0, n_pass, (B, n_pass))
    bs, bsm = d[idx].mean(1), dm[idx].mean(1)
    pair = [(y[i] - y[j]).mean() for i in (0, 1) for j in (2, 3)]
    seeds = [(y[i] - (y[2] + y[3]) / 2).mean() for i in (0, 1)]
    old = np.quantile(bs, 0.025) > 0 and min(seeds) > 0
    lo = 0.05 / 8 / 2
    new = np.quantile(bs, lo) > 0 and np.quantile(bsm, lo) > 0 and min(pair) >= 10.0
    return old, new


def main():
    rng = np.random.default_rng(0)
    out = {}
    for n_pass in (17, 14):
        for s_t in (3.0, 5.0):
            for s_u in (5.0, 10.0):
                r = np.array([one(rng, n_pass, s_t, s_u) for _ in range(4000)])
                p_old, p_new = r.mean(0)
                out['n%d_t%g_u%g' % (n_pass, s_t, s_u)] = {
                    'old_per_item': round(float(p_old), 3), 'new_per_item': round(float(p_new), 4),
                    'old_any_of_8': round(1 - (1 - float(p_old)) ** 8, 3), 'new_any_of_8': round(1 - (1 - float(p_new)) ** 8, 4)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
