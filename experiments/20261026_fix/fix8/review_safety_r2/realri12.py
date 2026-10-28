"""本物の記録(10/10・10/23・10/24。data/raw は読むだけ)を、場面をつないで Assist に開いたループで流し(tests/test_rec_fix8_safety_r1
の assist_replay。今の見張り ON / OFF)、見張り ON の出来事 rear_impact の前後を出す: 衝撃(IMU の水平 G 2.5 以上)の時刻と、
その手順で最初に後ろへ動いたサンプル(eRPM > 300)の時刻・下がった秒・ON と OFF の出す値の違い。
あわせて、リカバリーの 'reverse' の始めの前後 0.3 秒の IMU の最大(後退に入るときの揺れ)を全部の手順で数える。
    python -I -B realri12.py → out/realri12.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT))
import test_rec_fix8_safety_r1 as T  # noqa: E402


def main():
    out = {'events': [], 'reverse_starts': [], 'logs': 0}
    spec = {'now_on': ('now', True, True), 'now_off': ('now', False, False)}
    for log in T.raw_logs():
        for scenes in T.units(log, 'concat'):
            cat, meta, sk = T.load_unit(log, scenes)
            name = log
            if cat is None:
                out.setdefault('skipped', []).append([log, str(sk)[:120]])
                continue
            out['logs'] += 1
            r = T.assist_replay(cat, meta, spec=spec)
            t0 = T.log_t0_us(log)
            im = cat['IMU']
            it = im['timestamps_us'].astype(np.int64)
            hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
            v = cat['VEHICLE']
            vt = v['timestamps_us'].astype(np.int64)
            ve = v['erpm'].astype(float)
            ts = (r['t_us'] - t0) / 1e6
            on, off = r['out']['now_on'], r['out']['now_off']
            for k, what in r['ev']['now_on']:
                if what not in ('rear_impact', 'rear_impact_hold'):
                    continue
                tk = int(r['t_us'][k])
                # その手順の始め(段 None → brake)を前へ探す
                ks = max([j for j in r['rec_starts'] if j <= k] or [0])
                tb = int(r['t_us'][ks])
                m = (vt >= tb) & (vt <= tk)
                back = vt[m][ve[m] > 300]
                gm = (it >= tk - 300_000) & (it <= tk + 50_000)
                gi = it[gm][hg[gm] >= 2.5]
                dif = [int(j) for j in range(k, min(k + 40, len(ts))) if np.any(on[j] != off[j])]
                out['events'].append({
                    'log': name, 'what': what, 's': round(float(ts[k]), 3), 'rec_start_s': round(float((tb - t0) / 1e6), 3),
                    'first_back_s': round(float((back[0] - t0) / 1e6), 3) if len(back) else None,
                    'shock_s': [round(float((x - t0) / 1e6), 3) for x in gi[:6]],
                    'shock_g_max': round(float(hg[gm].max()), 2) if gm.any() else None,
                    'phase0': r['phase0'][k], 'erpm': r['erpm'][k],
                    'on_out_next': [list(map(float, on[j][:2])) for j in range(k, min(k + 6, len(ts)))],
                    'off_out_next': [list(map(float, off[j][:2])) for j in range(k, min(k + 6, len(ts)))],
                    'frames_on_ne_off_40': len(dif)})
            # 'reverse' に入ったフレームの前後 0.3 秒の IMU(今の OFF の段は分からないので、ON の段で見る)
            for j in range(1, len(ts)):
                if r['phase'][j] == 'reverse' and r['phase0'][j] in ('gap',):
                    tj = int(r['t_us'][j])
                    gm = (it >= tj - 300_000) & (it <= tj + 300_000)
                    out['reverse_starts'].append({'log': name, 's': round(float(ts[j]), 3),
                                                  'g_max': round(float(hg[gm].max()), 2) if gm.any() else None})
    gs = [x['g_max'] for x in out['reverse_starts'] if x['g_max'] is not None]
    out['reverse_starts_n'] = len(out['reverse_starts'])
    out['reverse_starts_ge2p5'] = sum(1 for g in gs if g >= 2.5)
    (HERE / 'out' / 'realri12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for e in out['events']:
        print(json.dumps(e, ensure_ascii=False, default=str))
    print('logs', out['logs'], 'reverse_starts', out['reverse_starts_n'], '>=2.5G', out['reverse_starts_ge2p5'],
          sorted(gs)[-8:])


if __name__ == '__main__':
    main()
