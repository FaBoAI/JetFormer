"""打ち消しの票 B(反証 8-2・docs の指摘 f8r2-docs-1023-first-recovery-off): 本物の記録(data/raw。読むだけ)の
リカバリー(AI_CMD の assist == 2)を 1 本の記録の場面をつないで探し(fix8/refute_safety_r1_B/real_recoveries.py と同じ数え方)、
  ・日ごとの記録の数・AI 側のフレームのある記録の数・リカバリーの数・その日の最初のリカバリー
  ・始めから後ろへ動き出すまで(eRPM > 300)の秒(10/23 の形 = すぐ下がる の目安)
  ・後ろへ下がった距離・後ろの当たり(後ろへ動いている間の水平 2.5G 以上)とそこまでに下がった距離(= 止まった所から後ろの物まで)
を出す。装置・GPU・API に触れない。
    jf-heavy … python -I -B real_rec_b.py → out/real_rec_b.json
"""
import json
from pathlib import Path

import numpy as np

RAW = Path('/home/jetson/jetformer/data/raw')
HERE = Path(__file__).resolve().parent
MPS_PER_ERPM = 0.5 / 1000.0


def cat(dirs, name, keys):
    parts = [np.load(d / (name + '.npz')) for d in dirs]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys if all(k in p for p in parts)}


def main():
    rows, days = [], {}
    for logdir in sorted(p for p in RAW.glob('2026*') if p.is_dir()):
        day = logdir.name[:8]
        D = days.setdefault(day, {'logs': 0, 'logs_ai': 0, 'logs_assist': 0, 'recoveries': 0})
        D['logs'] += 1
        if not (logdir / 'log.json').exists():
            continue
        scenes = [d for d in sorted(logdir.glob('2026*')) if d.is_dir()
                  and all((d / f).exists() and (d / f).stat().st_size > 0 for f in ('AI_CMD.npz', 'VEHICLE.npz', 'IMU.npz'))]
        if not scenes:
            continue
        try:
            z = cat(scenes, 'AI_CMD', ('timestamps_us', 'assist', 'select', 'out', 'model_out'))
            v = cat(scenes, 'VEHICLE', ('timestamps_us', 'erpm', 'select'))
            im = cat(scenes, 'IMU', ('timestamps_us', 'accel'))
        except Exception:   # noqa: BLE001
            continue
        if 'select' in z and bool(np.any(z['select'])):
            D['logs_ai'] += 1
        if 'assist' not in z:
            continue
        D['logs_assist'] += 1
        t = z['timestamps_us'].astype(np.int64)
        t0 = int(t[0])
        ass = z['assist'].astype(int)
        vt = v['timestamps_us'].astype(np.int64)
        ve = v['erpm'].astype(float)
        vs = v['select'].astype(bool)
        it = im['timestamps_us'].astype(np.int64)
        acc = im['accel'].astype(float)
        hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
        i, n = 0, len(t)
        while i < n:
            if ass[i] != 2 or (i > 0 and ass[i - 1] == 2):
                i += 1
                continue
            j = i
            while j + 1 < n and ass[j + 1] == 2:
                j += 1
            ts, te = int(t[i]), int(t[j]) + 34_000
            m = (vt >= ts) & (vt <= te + 1_500_000)
            vv, tt, ss = ve[m], vt[m], vs[m]
            back = back_ai = 0.0
            first_back_s = None
            hit = None
            for k in range(1, len(tt)):
                dt = (tt[k] - tt[k - 1]) / 1e6
                e = vv[k]
                if e > 0:
                    back += e * MPS_PER_ERPM * dt
                    if tt[k] <= te:
                        back_ai += e * MPS_PER_ERPM * dt
                if e > 300 and first_back_s is None:
                    first_back_s = round((tt[k] - ts) / 1e6, 3)
                if e > 300 and hit is None:
                    ii = (it > tt[k] - 60_000) & (it <= tt[k] + 60_000)
                    if ii.any() and float(hg[ii].max()) >= 2.5 and tt[k] > ts + 100_000:
                        hit = {'g': round(float(hg[ii].max()), 2), 'back_m': round(back, 3),
                               'v_mps': round(e * MPS_PER_ERPM, 2), 'ai_side': bool(ss[k])}
            D['recoveries'] += 1
            rows.append({'log': logdir.name, 'start_s': round((ts - t0) / 1e6, 3), 'dur_s': round((te - ts) / 1e6, 2),
                         'first_back_s': first_back_s, 'back_m_rec': round(back_ai, 3),
                         'back_m_incl_after': round(back, 3), 'rear_hit': hit})
            i = j + 1
    first = {}
    for r in rows:
        first.setdefault(r['log'][:8], r['log'] + '@' + str(r['start_s']))
    out = {'days': days, 'first_recovery_of_day': first, 'rows': rows}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'real_rec_b.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(days, ensure_ascii=False))
    print('first', json.dumps(first, ensure_ascii=False))
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print('n', len(rows))


if __name__ == '__main__':
    main()
