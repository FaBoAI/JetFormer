"""原因 C の診断(4): 車のモデル(sim.py の当てはめ)を、周の中のまっすぐな加速(発進ではないところ)で確かめる。2026-10-26

窓 = AI の 0.6 秒。窓の間ずっと |舵| < 150 µs・|ヨー| < 60 deg/s・衝突なし・指令は前進。始めの速さ v0 は 0.5〜3.5 m/s。
記録の指令を時刻の順に入れてシミュレーションし、窓の終わりの速さを記録と比べる(v0 と 指令の平均 の帯ごと)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I check_dyn.py   # → out/check_dyn.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect as C  # noqa: E402
import sim as S  # noqa: E402

WIN = 0.6
STEP = 0.2


def main():
    sim = json.loads((S.OUT / 'sim.json').read_text())
    fit = json.loads((S.OUT / 'dyn_fit.json').read_text())
    cm = sim['car_model']
    p = [cm['alpha_mps2_per_us'], cm['f0_us'], cm['beta_per_s'], cm['gamma_mps2'], cm['delay_s']]
    rows = []
    for log in C.logs():
        try:
            d = C.load(log)
        except Exception:      # noqa: BLE001
            continue
        z = d['z']
        if not z['ai'].any():
            continue
        t = z['t_s']
        c, n = d['ref']
        ai = z['ai'].astype(bool)
        f = n - z['out'][:, 1]
        s = z['out'][:, 0] - c
        gz_f = np.interp(t, d['it'], C.smooth(d['gz'], 9))
        hit_t = np.array([h[0] for h in d['hits']]) if d['hits'] else np.zeros(0)
        lab = d['lab']
        ratio = S.c_ratio(fit, C.session_of(log, C.ai_no(lab.get('description'))))
        car = S.Car(p, ratio, cm['a_cap_mps2'])
        t0 = t[0]
        while t0 + WIN < t[-1]:
            m = (t >= t0) & (t <= t0 + WIN)
            if m.sum() >= 15 and ai[m].all() and np.all(np.diff(t[m]) < 0.07) and np.all(np.abs(s[m]) < 150) \
                    and np.all(np.abs(gz_f[m]) < 60) and np.all(f[m] > 0) and not np.any(np.abs(hit_t - t0) < 1.5):
                v0 = float(np.interp(t0, d['vt'], d['v']))
                v1 = float(np.interp(t0 + WIN, d['vt'], d['v']))
                if 0.5 <= v0 <= 3.5:
                    # 指令は窓の前 δ から(遅れの分)
                    mm = (t >= t0 - 0.2) & (t <= t0 + WIN)
                    ts, vs, xs = S.sim_time_indexed(car, t[mm] - t0, f[mm], v0, WIN)
                    rows.append({'log': log, 't0': round(float(t0), 2), 'v0': round(v0, 2), 'v1': round(v1, 2),
                                 'v1_sim': round(float(vs[-1]), 2), 'f_mean': round(float(np.mean(f[m])), 1)})
                    t0 += WIN
                    continue
            t0 += STEP
    tab = []
    for v0a, v0b in ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5)):
        for fa, fb in ((100, 175), (175, 225), (225, 275), (275, 400)):
            sel = [r for r in rows if v0a <= r['v0'] < v0b and fa <= r['f_mean'] < fb]
            if len(sel) < 5:
                continue
            dv = np.array([r['v1'] - r['v0'] for r in sel])
            dvs = np.array([r['v1_sim'] - r['v0'] for r in sel])
            tab.append({'v0': [v0a, v0b], 'f': [fa, fb], 'n': len(sel), 'dv_rec_med': round(float(np.median(dv)), 2),
                        'dv_sim_med': round(float(np.median(dvs)), 2),
                        'err_med': round(float(np.median(dvs - dv)), 2)})
    out = {'created_by': 'experiments/20261026_fix/diag_launch/check_dyn.py', 'win_s': WIN, 'n': len(rows), 'table': tab}
    (S.OUT / 'check_dyn.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print('windows', len(rows))
    for r in tab:
        print(r)


if __name__ == '__main__':
    main()
