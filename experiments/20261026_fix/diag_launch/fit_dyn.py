"""原因 C の診断(2): 前後の動きの簡単なモデル(スロットル → 加速)を 10/23・10/24 の記録から作る。2026-10-26

読むだけ。
  a = (c·f(t − δ) − v) / τ を、まっすぐ走っているところ(|舵| < 150 µs・|ヨー| < 60 deg/s・衝突の前後 1 秒を除く)で合わせる。
  f = 前進の指令 [µs](中立 − 車に出した値。AI のフレームは out、人のフレームはスティック human_pwm)。
  加速 a は IMU の前の加速(止まっているときのずれを引き、9 サンプル平均)と、車輪の速さの微分の両方。
  c(電池の効き)は回ごと、τ・δ・加速の上限は全体で 1 つ。
  空転の目安: 低い速さ(v < 1.5 m/s)で、車輪の加速 ÷ IMU の加速(指令の帯ごと)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I fit_dyn.py     # → out/dyn_fit.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect as C  # noqa: E402

OUT = C.OUT
DELAYS = (0.0, 0.033, 0.067, 0.1, 0.133, 0.167)
F_BINS = (40, 100, 150, 200, 250, 300, 400)
V_BINS = (0.0, 1.0, 2.0, 3.0, 4.5)


def samples(d, delay):
    z = d['z']
    t = z['t_s']
    c, n = d['ref']
    ai = z['ai'].astype(bool)
    f_ai = n - z['out'][:, 1]
    f_hu = n - z['human_pwm'][:, 1]
    f = np.where(ai, f_ai, f_hu)
    s = np.where(ai, z['out'][:, 0] - c, z['human_pwm'][:, 0] - c)
    vt = d['vt']
    k = np.clip(np.searchsorted(t, vt - delay, side='right') - 1, 0, len(t) - 1)
    fresh = (vt - delay - t[k] >= 0) & (vt - delay - t[k] <= 0.07)
    fv, sv, aiv = f[k], s[k], ai[k]
    v = d['v']
    vs = C.smooth(v, 9)
    aw = np.gradient(vs, vt)
    ay = np.interp(vt, d['it'], C.smooth(d['ay'], 9))
    gz = np.interp(vt, d['it'], C.smooth(d['gz'], 9))
    near_hit = np.zeros(len(vt), bool)
    for h in d['hits']:
        near_hit |= np.abs(vt - h[0]) < 1.0
    ok = fresh & np.isfinite(fv) & np.isfinite(sv) & ~near_hit & (np.abs(sv) < 150) & (np.abs(gz) < 60) & (v > -0.05)
    ok &= np.isfinite(ay)
    return {'f': fv[ok], 'v': v[ok], 'aw': aw[ok], 'ay': ay[ok], 'ai': aiv[ok], 'log': np.full(ok.sum(), d['log'])}


def session_key(log, lab):
    no = C.ai_no(lab.get('description'))
    return C.session_of(log, no)


def main():
    data = {}
    for log in C.logs():
        try:
            d = C.load(log)
        except Exception as e:      # noqa: BLE001
            print('skip', log, e)
            continue
        data[log] = d
    res = {'created_by': 'experiments/20261026_fix/diag_launch/fit_dyn.py', 'fits': {}, 'bins': {}, 'slip_bins': {}}
    best = None
    for delay in DELAYS:
        S = {}
        for log, d in data.items():
            s = samples(d, delay)
            key = session_key(log, d['lab'])
            S.setdefault(key, []).append(s)
        S = {k: {kk: np.concatenate([x[kk] for x in v]) for kk in v[0]} for k, v in S.items()}
        # 前進の指令(f > 20 µs)・速さ 0〜4.5 m/s。a = α f − β v + γ(回ごとの α、共通の β・γ)
        keys = sorted(S)
        rows, ys = [], []
        for i, k in enumerate(keys):
            s = S[k]
            m = (s['f'] > 20) & (s['v'] < 4.5) & (s['f'] < 450)
            X = np.zeros((m.sum(), len(keys) + 2))
            X[:, i] = s['f'][m]
            X[:, -2] = -s['v'][m]
            X[:, -1] = 1.0
            rows.append(X)
            ys.append(s['ay'][m])
        X, y = np.concatenate(rows), np.concatenate(ys)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        beta = coef[-2]
        fit = {'delay_s': delay, 'r2': round(float(r2), 4), 'n': int(len(y)), 'tau_s': round(float(1 / beta), 3),
               'gamma': round(float(coef[-1]), 3),
               'c_mps_per_us': {k: round(float(coef[i] / beta), 5) for i, k in enumerate(keys)},
               'n_by_session': {k: int((S[k]['f'] > 20).sum()) for k in keys}}
        res['fits']['%.3f' % delay] = fit
        print('delay %.3f r2 %.3f tau %.3f gamma %.2f c %s' % (delay, r2, 1 / beta, coef[-1], fit['c_mps_per_us']))
        if best is None or r2 > best[0]:
            best = (r2, delay, S, fit)
    r2, delay, S, fit = best
    res['best'] = fit
    # 指令 × 速さ の帯ごとの加速(IMU・車輪)と、車輪 ÷ IMU
    allS = {kk: np.concatenate([S[k][kk] for k in S]) for kk in ('f', 'v', 'aw', 'ay', 'ai')}
    tab = []
    for v0, v1 in zip(V_BINS[:-1], V_BINS[1:]):
        for f0, f1 in zip(F_BINS[:-1], F_BINS[1:]):
            m = (allS['v'] >= v0) & (allS['v'] < v1) & (allS['f'] >= f0) & (allS['f'] < f1)
            if m.sum() < 20:
                continue
            ay, aw = allS['ay'][m], allS['aw'][m]
            pos = ay > 1.0
            tab.append({'v': [v0, v1], 'f': [f0, f1], 'n': int(m.sum()), 'n_ai': int(allS['ai'][m].sum()),
                        'a_imu_med': round(float(np.median(ay)), 2), 'a_imu_p90': round(float(np.percentile(ay, 90)), 2),
                        'a_wheel_med': round(float(np.median(aw)), 2),
                        'wheel_over_imu_med': round(float(np.median(aw[pos] / ay[pos])), 3) if pos.sum() >= 10 else None,
                        'model_pred_a': round(float((fit['c_mps_per_us'][max(fit['n_by_session'], key=fit['n_by_session'].get)]
                                                     * (f0 + f1) / 2 - (v0 + v1) / 2) / fit['tau_s'] + fit['gamma']), 2)})
    res['bins'] = tab
    # 加速の上限の目安: IMU の前の加速の 99% 点(v < 2 m/s)
    m = allS['v'] < 2.0
    res['a_imu_p99_low_speed'] = round(float(np.percentile(allS['ay'][m], 99)), 2)
    res['a_imu_max_low_speed'] = round(float(np.max(allS['ay'][m])), 2)
    m2 = m & (allS['f'] >= 250)
    res['a_imu_p90_low_speed_f250'] = round(float(np.percentile(allS['ay'][m2], 90)), 2) if m2.sum() > 20 else None
    (OUT / 'dyn_fit.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res['best'], ensure_ascii=False))
    for r in tab:
        print(r)
    print('a_imu p99/max (v<2):', res['a_imu_p99_low_speed'], res['a_imu_max_low_speed'], 'p90 f>=250:', res['a_imu_p90_low_speed_f250'])


if __name__ == '__main__':
    main()
