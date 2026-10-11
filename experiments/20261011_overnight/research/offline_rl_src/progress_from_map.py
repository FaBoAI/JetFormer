"""コース図に合わせた軌跡(course/tracks_*.npz)から、オフラインRLの報酬に使う「コース上の進み」を確かめる。
読み取りのみ。npz だけ使う(画像は読まない)。
出力: progress_from_map.json
- 周ごとの時間(track の lap 番号が増えた時刻の差。AI/人の別は周の中で多い方)
- 走った距離(speed の積分)に対する、コース上の進み(progress_cum_m の増分)の比 = 「役に立った距離」の割合
- 進みが後ろ向き(後退・スピン)のフレームの割合
- 区間(section)ごとの、AIの進みの速さ(m/s)の中央値
"""
import json, os, sys
import numpy as np
D = '/home/jetson/jetformer_reports/overnight_20261011/course'
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007', '20261010-203924': 'ai_run_009',
        '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}
res = {}
for log, name in LOGS.items():
    z = np.load(os.path.join(D, 'tracks_%s.npz' % log))
    t, v, cum, lap, sec, valid, ai = z['t'], z['speed'], z['progress_cum_m'], z['lap'], z['section'], z['valid'], z['ai']
    names = list(z['section_names'])
    L = float(json.load(open(os.path.join(D, 'tracks_%s.json' % log)))['centerline_length_m'])
    r = {'name': name, 'frames': int(len(t)), 'valid_frac': float(valid.mean()), 'centerline_length_m': L}
    # 周
    ok = valid & np.isfinite(cum)
    idx = np.nonzero(ok)[0]
    laps = []
    if len(idx):
        lp = lap[idx]; tt = t[idx]
        ch = np.nonzero(np.diff(lp) == 1)[0] + 1
        for a, b in zip(ch[:-1], ch[1:]):
            ia, ib = idx[a], idx[b]
            fr = slice(ia, ib)
            gap = np.max(np.diff(t[fr])) if ib - ia > 1 else 0
            if not valid[fr].all() or gap > 0.2:
                continue
            laps.append({'t0': float(t[ia]), 'dt': float(t[ib] - t[ia]), 'ai_frac': float(ai[fr].mean())})
    r['laps'] = laps
    aiL = [l['dt'] for l in laps if l['ai_frac'] > 0.9]; huL = [l['dt'] for l in laps if l['ai_frac'] < 0.1]
    r['lap_ai_s'] = sorted(round(x, 2) for x in aiL); r['lap_human_s'] = sorted(round(x, 2) for x in huL)
    r['lap_mixed_n'] = len(laps) - len(aiL) - len(huL)
    # 距離と進み
    dt = np.r_[np.diff(t), 0]; dt = np.where(dt > 0.2, 0, dt)
    dcum = np.r_[np.diff(cum), 0]
    good = ok & np.r_[ok[1:], False] & (dt > 0)
    for who, m in (('ai', good & ai), ('human', good & ~ai)):
        dist = float(np.nansum(np.abs(v[m]) * dt[m])); prog = float(np.nansum(dcum[m]))
        mov = m & (np.abs(v) > 0.3)
        back = float((dcum[mov] < -0.005).mean()) if mov.any() else None
        r[who] = {'seconds': float(dt[m].sum()), 'dist_from_speed_m': round(dist, 1), 'progress_m': round(prog, 1),
                  'progress_over_dist': round(prog / dist, 3) if dist > 0 else None, 'frac_backward_when_moving': back}
    # 区間ごと AI の進みの速さ
    ps = {}
    for k, nm in enumerate(names):
        m = good & ai & (sec == k)
        if m.sum() > 30:
            rate = dcum[m] / dt[m]
            ps[str(nm)] = {'n': int(m.sum()), 'progress_rate_med_mps': round(float(np.median(rate)), 2),
                           'speed_med_mps': round(float(np.median(np.abs(v[m]))), 2)}
    r['ai_by_section'] = ps
    res[log] = r
    print(name, 'valid %.2f' % r['valid_frac'], 'laps ai', r['lap_ai_s'], 'hum', r['lap_human_s'], 'mixed', r['lap_mixed_n'])
    print('   ai', r['ai'], '\n   hu', r['human'])
    print('   sec', ps)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress_from_map.json'), 'w'), ensure_ascii=False, indent=1)

# 追記: 区間ごとに「進み合計 / 走った距離合計」(フレーム単位の量子化を避ける)
sec_ratio = {}
for log, name in LOGS.items():
    z = np.load(os.path.join(D, 'tracks_%s.npz' % log))
    t, v, cum, sec, valid, ai = z['t'], z['speed'], z['progress_cum_m'], z['section'], z['valid'], z['ai']
    names = list(z['section_names'])
    dt = np.r_[np.diff(t), 0]; dt = np.where(dt > 0.2, 0, dt); dcum = np.r_[np.diff(cum), 0]
    ok = valid & np.isfinite(cum); good = ok & np.r_[ok[1:], False] & (dt > 0) & ai
    out = {}
    for k, nm in enumerate(names):
        m = good & (sec == k)
        dist = float(np.sum(np.abs(v[m]) * dt[m])); prog = float(np.sum(dcum[m]))
        if dist > 5:
            out[str(nm)] = {'dist_m': round(dist, 1), 'progress_m': round(prog, 1), 'ratio': round(prog / dist, 3),
                            'mean_speed_mps': round(dist / dt[m].sum(), 2), 'mean_progress_rate_mps': round(prog / dt[m].sum(), 2)}
    sec_ratio[name] = out
    print(name, out)
res['_ai_section_ratio'] = sec_ratio
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress_from_map.json'), 'w'), ensure_ascii=False, indent=1)
