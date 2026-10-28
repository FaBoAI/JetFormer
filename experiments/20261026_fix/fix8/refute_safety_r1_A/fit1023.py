"""票 A: 本物の 10/23 20261023-205458(No.6。見張りは無かった = OFF の手順)のリカバリーの書いた値と eRPM から、
ESC の後退の模型を合わせる(読むだけ)。書いた時刻 = 撮像時刻 + latency_ms。合わせる区間は 'brake' を書いてから後ろの当たりの前まで。
模型: 後退側(> 中立 + 40)を書くと、待っている(ready)・止まっている/後ろへ動いているなら rdelay 秒あとから後退で走る。
走っている間の加速は const(A·s、上限 Vmax·s。s = (thr − 中立)/300)か fo(k·(Vmax·s − |v|))。中立の帯では後退が切れて
coast [m/s²] で遅くなる。中立のあとの後退側は rdelay2 秒あとから。ESC は lag 秒前の値を見る。
    python -I -B fit1023.py → 標準出力・out/fit1023.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rec_io  # noqa: E402

KV = 0.5e-3
DT = 0.01


def sim(cmds, n, t_a, t_b, prm):
    """cmds = [(書いた時刻 s, thr)]。t_a から t_b まで DT で進めた (t, v)。v < 0 = 後ろへ"""
    acc, A, Vmax, coast, rdelay, rdelay2, lag = (prm[k] for k in ('acc', 'A', 'Vmax', 'coast', 'rdelay', 'rdelay2', 'lag'))
    ts = np.array([c[0] for c in cmds])
    th = np.array([c[1] for c in cmds])
    v, rev_on, ran, wait = 0.0, False, False, 0.0
    out = []
    t = t_a
    while t <= t_b + 1e-9:
        j = np.searchsorted(ts, t - lag, side='right') - 1
        thr = th[j] if j >= 0 else n
        if thr > n + 40:
            s = (thr - n) / 300.0
            if rev_on:
                if acc == 'const':
                    v = max(v - A * s * DT, -Vmax * s)
                else:
                    v = v - A * (Vmax * s + v) * DT
            elif v <= 0.075:
                wait += DT
                if wait >= (rdelay2 if ran else rdelay) - 1e-9:
                    rev_on, ran = True, True
        else:
            wait = 0.0
            rev_on = False
            v = min(0.0, v + coast * DT) if v < 0 else max(0.0, v - coast * DT)
        out.append((round(t, 4), v))
        t += DT
    return out


def main():
    log = '20261023-205458'
    cat, L, t0 = rec_io.load_log(log)
    n = float(L['drive']['throttle_neutral_us'])
    A_, V = cat['AI_CMD'], cat['VEHICLE']
    ta = (A_['timestamps_us'] - t0) / 1e6
    tw = ta + np.where(np.isfinite(A_['latency_ms']), A_['latency_ms'], 0.0) / 1000.0
    i0 = int(np.where((A_['assist'] == 2) & (ta > 23.0))[0][0])       # 'brake' の始め
    t_brake = float(tw[i0])
    cmds = [(float(tw[i]), float(A_['out'][i][1])) for i in range(i0, i0 + 40)]
    tv = (V['timestamps_us'] - t0) / 1e6
    ve = V['erpm'].astype(float)
    t_hit = 24.19                       # 後ろの当たりの前(24.197 秒に当たった。fix2 の記録)
    sel = (tv >= t_brake) & (tv <= t_hit)
    obs_t, obs_v = tv[sel], -ve[sel] * KV
    # 跳ね(前後と 0.4 m/s 以上違う 1 サンプル)は合わせに使わない
    keep = np.ones(len(obs_v), bool)
    for k in range(1, len(obs_v) - 1):
        if abs(obs_v[k] - obs_v[k - 1]) > 0.4 and abs(obs_v[k] - obs_v[k + 1]) > 0.4:
            keep[k] = False
    obs_t, obs_v = obs_t[keep], obs_v[keep]
    print('brake write %.3f  cmds:' % t_brake, [(round(a, 3), b) for a, b in cmds[:26]])
    best = []
    grid = list(itertools.product(('const',), (4.0, 5.0, 6.0, 7.0, 8.0), (2.0, 2.2, 2.4, 2.6, 3.0),
                                  (0.0, 0.4, 0.8, 1.5, 3.0), (0.0, 0.01, 0.02, 0.03, 0.05, 0.08),
                                  (0.0, 0.03, 0.05, 0.08), (0.0, 0.02, 0.04, 0.06)))
    grid += list(itertools.product(('fo',), (1.0, 1.5, 2.0, 2.5, 3.0, 4.0), (3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
                                   (0.0, 0.2, 0.4, 0.8, 1.5), (0.0, 0.01, 0.02, 0.03, 0.05, 0.08),
                                   (0.0, 0.03, 0.05, 0.08), (0.0, 0.02, 0.04, 0.06)))
    for acc, Ak, Vm, co, rd, rd2, lag in grid:
        prm = dict(acc=acc, A=Ak, Vmax=Vm, coast=co, rdelay=rd, rdelay2=rd2, lag=lag)
        tr = sim(cmds, n, t_brake, t_hit, prm)
        tt = np.array([a for a, _ in tr])
        vv = np.array([b for _, b in tr])
        pred = np.interp(obs_t, tt, vv)
        err = float(np.sqrt(np.mean((pred - obs_v) ** 2)))     # どちらも後ろが負
        best.append((err, prm))
    best.sort(key=lambda x: x[0])
    for e, p in best[:12]:
        print('rmse %.4f m/s' % e, p)
    fit = {'best': [{'rmse': e, 'prm': p} for e, p in best[:30]], 't_brake': t_brake,
           'obs': [(round(float(a), 3), round(float(b), 3)) for a, b in zip(obs_t, obs_v)]}
    # いちばん合う const・fo それぞれで、時刻ごとの見比べ
    for acc in ('const', 'fo'):
        e, p = next((e, p) for e, p in best if p['acc'] == acc)
        tr = sim(cmds, n, t_brake, t_hit, p)
        tt = np.array([a for a, _ in tr]); vv = np.array([b for _, b in tr])
        pred = np.interp(obs_t, tt, vv)
        print('==', acc, 'rmse %.4f' % e, p)
        for k in range(0, len(obs_t), 6):
            print('  %.3f real %.2f model %.2f' % (obs_t[k], obs_v[k], pred[k]))
        x = float(np.sum(vv) * DT)
        print('  backed by model until %.2f: %.3f m (real integral of eRPM: %.3f m)' % (
            t_hit, -x, float(np.trapezoid(obs_v, obs_t))))
        fit['best_' + acc] = {'rmse': e, 'prm': p, 'model_back_m': -x}
    fit['real_back_m_erpm_integral'] = float(np.trapezoid(obs_v, obs_t))
    (HERE / 'out' / 'fit1023.json').write_text(json.dumps(fit, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
