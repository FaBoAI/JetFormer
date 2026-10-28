"""打ち消しの票 B(指摘 rs8-1-backearly-rear-faster): 本物の記録(data/raw。読むだけ)のリカバリー(AI_CMD の assist == 2)を
1 本の記録の場面をつないで探し、それぞれについて
  始め(秒)・始める前の 0.6 秒のモデルの出力がブレーキ側(中立 + 40 µs より上)だったフレームの割合(10/23 の形の目安)・
  始めの eRPM・リカバリーの間に後ろへ下がった距離(eRPM > 0 の積分。1000 eRPM = 0.5 m/s)・後ろへの最高の速さ・
  後ろへ動いている間の IMU の水平 G の最大(2.5G 以上なら後ろの当たりの見込み)とそのときまでに下がった距離・
  リカバリーの終わり方(人側にした・中立止め・走りに戻った)
を出す。装置・GPU・API に触れない。 python -I -B real_recoveries.py → out/real_recoveries.json
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
    rows = []
    for logdir in sorted(p for p in RAW.glob('2026*') if p.is_dir()):
        if not (logdir / 'log.json').exists():
            continue
        scenes = [d for d in sorted(logdir.glob('2026*')) if d.is_dir() and all((d / f).exists() for f in ('AI_CMD.npz', 'VEHICLE.npz', 'IMU.npz'))]
        if not scenes:
            continue
        try:
            z = cat(scenes, 'AI_CMD', ('timestamps_us', 'assist', 'select', 'hold', 'out', 'model_out'))
            v = cat(scenes, 'VEHICLE', ('timestamps_us', 'erpm', 'select'))
            im = cat(scenes, 'IMU', ('timestamps_us', 'accel'))
            L = json.loads((logdir / 'log.json').read_text(encoding='utf-8'))
        except Exception as ex:   # noqa: BLE001
            continue
        if 'assist' not in z:
            continue
        dr = L.get('drive') or {}
        N = float(dr.get('throttle_neutral_us', 1496))
        t = z['timestamps_us'].astype(np.int64)
        t0 = int(t[0])
        ass = z['assist'].astype(int)
        mo = z['model_out'].astype(float) if 'model_out' in z else z['out'].astype(float)
        vt = v['timestamps_us'].astype(np.int64)
        ve = v['erpm'].astype(float)
        vs = v['select'].astype(bool)
        it = im['timestamps_us'].astype(np.int64)
        acc = im['accel'].astype(float)
        hg = np.hypot(acc[:, 0], acc[:, 1]) / 9.80665
        i = 0
        n = len(t)
        while i < n:
            if ass[i] != 2 or (i > 0 and ass[i - 1] == 2):
                i += 1
                continue
            j = i
            while j + 1 < n and ass[j + 1] == 2:
                j += 1
            ts, te = int(t[i]), int(t[j]) + 34_000
            pre = (t >= ts - 600_000) & (t < ts)
            brake_frac = float(np.mean(mo[pre, 1] > N + 40)) if pre.any() else None
            m = (vt >= ts) & (vt <= te + 1_500_000)      # リカバリーと、そのあとの 1.5 秒(人側の惰性も)
            vv, tt, ss = ve[m], vt[m], vs[m]
            # リカバリーの間(AI 側)の後ろへの積分
            back = 0.0
            back_ai = 0.0
            vmax = 0.0
            hit = None
            for k in range(1, len(tt)):
                dt = (tt[k] - tt[k - 1]) / 1e6
                e = vv[k]
                if e > 0:
                    back += e * MPS_PER_ERPM * dt
                    if tt[k] <= te:
                        back_ai += e * MPS_PER_ERPM * dt
                    vmax = max(vmax, e * MPS_PER_ERPM)
                if e > 300 and hit is None:
                    ii = (it > tt[k] - 60_000) & (it <= tt[k] + 60_000)
                    if ii.any() and float(hg[ii].max()) >= 2.5 and tt[k] > ts + 100_000:
                        hit = {'s': round((tt[k] - t0) / 1e6, 3), 'g': round(float(hg[ii].max()), 2), 'back_m': round(back, 3),
                               'v_mps': round(e * MPS_PER_ERPM, 2), 'ai_side': bool(ss[k])}
            end_sel = bool(z['select'][j]) if j + 1 >= n else bool(z['select'][j + 1])
            nxt = int(ass[j + 1]) if j + 1 < n else None
            rows.append({'log': logdir.name, 'start_s': round((ts - t0) / 1e6, 3), 'end_s': round((te - t0) / 1e6, 3),
                         'dur_s': round((te - ts) / 1e6, 2), 'pre_brake_frac': brake_frac,
                         'erpm_at_start': float(ve[max(0, np.searchsorted(vt, ts) - 1)]),
                         'back_m_rec': round(back_ai, 3), 'back_m_incl_after': round(back, 3), 'vmax_back_mps': round(vmax, 2),
                         'rear_hit': hit, 'next_assist': nxt, 'next_select_ai': end_sel})
            i = j + 1
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'real_recoveries.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print('n', len(rows))


if __name__ == '__main__':
    main()
