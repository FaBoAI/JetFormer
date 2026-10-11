"""AI走行 006/007(コース図に当てはめた軌跡)の区間ごとの時間と速度。CPUのみ・読み取りのみ。
入力: ~/jetformer_reports/overnight_20261011/course/tracks_*.npz
出力: section_time.json(このフォルダ)
"""
import json, os, numpy as np
BASE = os.path.expanduser('~/jetformer_reports/overnight_20261011/course')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'section_time.json')
LOGS = {'20261010-172910': 'ai_run_006', '20261010-173445': 'ai_run_007'}
res = {}
for log, name in LOGS.items():
    z = np.load(os.path.join(BASE, f'tracks_{log}.npz'), allow_pickle=True)
    names = list(z['section_names'])
    t, sec, ai, valid, sp, lap, prog = z['t'], z['section'], z['ai'], z['valid'], z['speed'], z['lap'], z['progress_cum_m']
    dt = np.diff(t, append=t[-1]); dt = np.clip(dt, 0, 0.1)
    m = ai & valid
    per = {}
    # 区間の通過(連続した同じ区間の並び)ごとの所要時間。AIのみ・有効のみ・途中で切れていない通過
    passes = {n: [] for n in names}
    i = 0; N = len(t)
    while i < N:
        j = i
        while j + 1 < N and sec[j + 1] == sec[i]:
            j += 1
        seg = slice(i, j + 1)
        full = (i > 0 and j < N - 1 and sec[i - 1] != sec[i] and sec[j + 1] != sec[i])
        if full and m[seg].all() and (t[j] - t[i]) < 8:
            passes[names[sec[i]]].append(float(t[j] - t[i] + dt[j]))
        i = j + 1
    for k, n in enumerate(names):
        mm = m & (sec == k)
        per[n] = {
            'time_s_total': round(float(dt[mm].sum()), 2),
            'n_pass_full': len(passes[n]),
            'pass_time_median_s': round(float(np.median(passes[n])), 2) if passes[n] else None,
            'speed_ms_p50': round(float(np.median(sp[mm])), 2) if mm.any() else None,
            'speed_ms_p90': round(float(np.percentile(sp[mm], 90)), 2) if mm.any() else None,
        }
    tot = sum(v['time_s_total'] for v in per.values())
    for n in names:
        per[n]['time_share'] = round(per[n]['time_s_total'] / tot, 3)
    res[name] = {'log': log, 'sections': per, 'ai_valid_time_s': round(tot, 1)}
json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
for name, r in res.items():
    print(name, r['ai_valid_time_s'])
    for n, v in r['sections'].items():
        print('  ', n, v)
