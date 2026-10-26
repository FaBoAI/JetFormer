#!/usr/bin/env python3
"""原因 A の診断 M2(numpy だけ・読むだけ): 門「低速のブレーキ」の分子(10/10 の AI 走行の TensorRT・v < 1.0 m/s・
行 0 のスロットル > 1598 µs)が、どんな場面のフレームかを分ける(衝突の前後・止まっているか・交代の前・区間・ToF の前の空き)。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I m2_brake_frames.py

フレームの並びは desk_gpu.ai_frames(ai_mix_20261011 の source == 1 & vehicle_valid、記録・時刻の順)の写し。
書くのは out/m2_brake_frames.json だけ。
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path('/home/jetson/jetformer/data')
AI_DS = DATA / 'datasets' / 'ai_mix_20261011'
LABELS = DATA / 'labels'
TRT = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk/trt')
OUT = Path(__file__).resolve().parent / 'out'
AI_LOGS = ('20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
           '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
           '20261010-204605')
LOC = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']
SPEED_PER_ERPM = 0.5e-3
BRAKE = 1598.0


def ai_frames():
    sp = json.loads((AI_DS / 'splits.json').read_text())
    rows = []
    for split in ('train', 'val'):
        for s in sp[split]:
            if s[:15] not in AI_LOGS:
                continue
            with np.load(AI_DS / 'aligned' / (s + '.npz')) as z:
                ok = (z['source'] == 1) & z['vehicle_valid']
                info = {k: z[k] for k in ('timestamps_us', 'near_collision', 'collision', 'loc_code', 'end_code',
                                          'time_to_end_s', 'vehicle')}
                tof = z['tof']
                for i in np.nonzero(ok)[0]:
                    d = tof[i] & 0x1FFF
                    c = tof[i] >> 13
                    mid = d[14:28, 18:36].astype(np.float64)
                    cm = c[14:28, 18:36] > 0
                    front = float(np.median(mid[cm])) if cm.sum() > 10 else np.nan
                    rows.append((s, int(i), int(info['timestamps_us'][i]), bool(info['near_collision'][i]),
                                 bool(info['collision'][i]), int(info['loc_code'][i]), int(info['end_code'][i]),
                                 float(info['time_to_end_s'][i]), float(info['vehicle'][i, 3]), front))
    rows.sort(key=lambda r: (r[0][:15], r[2], r[0], r[1]))
    return rows


def label_ctx(log, t_us, cache={}):
    """そのフレームの前後の出来事: 直前の衝突からの秒・次の交代までの秒"""
    if log not in cache:
        j = json.loads((LABELS / (log + '.json')).read_text())
        ev = j['events']
        cache[log] = ([c['t_us'] for c in ev.get('collisions', [])], [t['t_us'] for t in ev.get('takeovers', [])])
    cs, tks = cache[log]
    prev_c = [t_us - c for c in cs if c <= t_us]
    next_t = [t - t_us for t in tks if t >= t_us]
    return (min(prev_c) / 1e6 if prev_c else np.inf), (min(next_t) / 1e6 if next_t else np.inf)


def latest_trt():
    out = {}
    for f in sorted(TRT.glob('*/*.npz')):
        out[f.stem] = f              # 後の仕事で上書き(どの仕事も同じフレームの並び)
    return out


def main():
    rows = ai_frames()
    n = len(rows)
    v = np.array([-r[8] * SPEED_PER_ERPM for r in rows])
    near = np.array([r[3] for r in rows])
    loc = np.array([r[5] for r in rows])
    front = np.array([r[9] for r in rows])
    ctx = np.array([label_ctx(r[0][:15], r[2]) for r in rows])
    since_c, to_tk = ctx[:, 0], ctx[:, 1]
    files = latest_trt()
    res = {'n_frames_1010': n, 'runs': {}}
    low = v < 1.0
    cats = {
        '止まっている v<0.3': low & (v < 0.3),
        '動き出し・止まりかけ 0.3-1.0': low & (v >= 0.3),
        '衝突の後 3 秒以内': low & (since_c < 3.0),
        '衝突の後 3 秒より後': low & (since_c >= 3.0),
        '交代の前 2 秒以内(人が取る直前)': low & (to_tk < 2.0),
        '前が近い(ToF の前 < 400 mm)': low & (front < 400),
        '前が空いている(ToF の前 >= 1000 mm)': low & (front >= 1000),
        'near_collision': low & near,
    }
    for name, f in files.items():
        with np.load(f) as z:
            pr, erpm, part = z['pred'].astype(np.float64), z['erpm'], z['part']
        m0 = part == 0
        if m0.sum() != n:
            print('並びが合わない', name, m0.sum(), n)
            continue
        if not np.allclose(-erpm[m0] * SPEED_PER_ERPM, v):
            print('erpm が合わない', name)
            continue
        thr0 = pr[m0][:, 0, 1]
        brake = low & (thr0 > BRAKE)
        r = {'brake_low': int(brake.sum()), 'n_low': int(low.sum()), 'frac': float(brake.sum() / low.sum()),
             'by_cat': {}, 'by_loc': {}, 'by_log': {}}
        for cn, cm in cats.items():
            r['by_cat'][cn] = [int((brake & cm).sum()), int(cm.sum())]
        for li, ln in enumerate(LOC):
            cm = low & (loc == li)
            if cm.any():
                r['by_loc'][ln] = [int((brake & cm).sum()), int(cm.sum())]
        logs = np.array([x[0][:15] for x in rows])
        for lg in AI_LOGS:
            cm = low & (logs == lg)
            if cm.any():
                r['by_log'][lg] = [int((brake & cm).sum()), int(cm.sum())]
        res['runs'][name] = r
    OUT.mkdir(exist_ok=True)
    (OUT / 'm2_brake_frames.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    keys = list(cats)
    print('low-speed frames', int(low.sum()), {k: int(c.sum()) for k, c in cats.items()})
    for name, r in res['runs'].items():
        print('%-56s %4d/%d %.3f | %s' % (name[:56], r['brake_low'], r['n_low'], r['frac'],
                                         ' '.join('%d' % r['by_cat'][k][0] for k in keys)))


if __name__ == '__main__':
    sys.exit(main())
