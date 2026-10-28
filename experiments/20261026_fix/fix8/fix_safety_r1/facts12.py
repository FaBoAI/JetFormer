"""fix8 係 safety 直す 8-1 回目: 文書に書く本物の記録の数(data/raw。読むだけ)。
  1. 10/24 20261024-160847 の 39.9〜41.3 秒: 書いた値(記録の AI_CMD の out)・CH3(VEHICLE の select)・eRPM・hall_count・後ろへの
     距離(eRPM の積分。safety_caps.KV = 0.5e-3 m/s / eRPM)・IMU の水平 G の最大
  2. 10/23 20261023-205458(No.6)のリカバリー: 'brake'(後退側)を書き始めた時刻・後ろへ動き始めた時刻・後ろの当たり
     (水平 G 2.5 以上の最初と最大)までに下がった距離・後退側を書いていた間の速さ
時刻は記録の最初の場面の scene.json の start_us から(tests/test_rec_fix8_safety_r1.log_t0_us)。
    python3 -I -B facts12.py → out/facts12.json・標準出力
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

KV = 0.5e-3


def tables(log):
    cat, meta, sk = T.load_unit(log, T.scene_dirs(log))
    t0 = T.log_t0_us(log)
    return cat, meta, t0


def f160847():
    cat, meta, t0 = tables(T.LOG)
    z, v, im = cat['AI_CMD'], cat['VEHICLE'], cat['IMU']
    zs = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
    vs = (v['timestamps_us'].astype(np.int64) - t0) / 1e6
    its = (im['timestamps_us'].astype(np.int64) - t0) / 1e6
    hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
    rows = []
    for k in np.nonzero((zs >= 39.9) & (zs <= 41.3))[0]:
        rows.append({'s': round(float(zs[k]), 3), 'out': [round(float(x), 1) for x in z['out'][k]], 'select': bool(z['select'][k]),
                     'hold': bool(z['hold'][k]), 'assist': int(z['assist'][k]), 'model_thr': round(float(z['model_out'][k][1]), 1)})
    sel = v['select'].astype(bool)

    def span(a, b):
        m = (vs >= a) & (vs <= b)
        h = v['hall_count'].astype(np.int64)[m]
        e = v['erpm'].astype(float)[m]
        tt = vs[m]
        dist = float(np.sum(np.maximum(e[1:], 0) * KV * np.diff(tt)))
        st = np.diff(h)
        return {'from': a, 'to': b, 'hall': [int(h[0]), int(h[-1])], 'hall_up_steps': int(np.sum(st > 0)),
                'hall_down_steps': int(np.sum(st < 0)), 'erpm_min': float(e.min()), 'erpm_max': float(e.max()),
                'back_m': round(dist, 3), 'select_all_ai': bool(np.all(sel[m])), 'select_any_ai': bool(np.any(sel[m]))}
    ai_end = float(vs[(vs > 40.0) & ~sel][0])
    out = {'frames': rows, 'pre_stop': span(39.44, 40.03), 'ai_part': span(40.03, round(ai_end, 3)),
           'after_human': span(round(ai_end, 3), 41.21), 'total': span(40.03, 41.21), 'ch3_human_from_s': round(ai_end, 3),
           'imu_max_g_40_40p2': round(float(hg[(its >= 40.0) & (its <= 40.2)].max()), 2),
           'first_brake_s': next(r['s'] for r in rows if r['out'][1] > 1700 and r['assist'] == 2)}
    return out


def fno6():
    cat, meta, t0 = tables(T.LOG_1023)
    z, v, im = cat['AI_CMD'], cat['VEHICLE'], cat['IMU']
    zs = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
    vs = (v['timestamps_us'].astype(np.int64) - t0) / 1e6
    its = (im['timestamps_us'].astype(np.int64) - t0) / 1e6
    hg = np.hypot(im['accel'][:, 0], im['accel'][:, 1]) / 9.80665
    N = float(meta['drive']['throttle_neutral_us'])
    rec = np.nonzero((z['assist'].astype(int) == 2) & (zs > 20) & (zs < 26))[0]
    tb = float(zs[rec[0]])
    e = v['erpm'].astype(float)
    m = (vs >= tb) & (vs <= tb + 1.3)
    tt, ee = vs[m], e[m]
    t_back = float(tt[np.argmax(ee > 300)])
    x = np.concatenate([[0.0], np.cumsum(np.maximum(ee[1:], 0) * KV * np.diff(tt))])
    gm = (its >= tb) & (its <= tb + 1.3)
    gi = its[gm][hg[gm] >= 2.5]
    hits = []
    for g_t in gi[:1]:
        k = np.searchsorted(tt, g_t) - 1
        hits.append({'s': round(float(g_t), 3), 'back_m': round(float(x[max(k, 0)]), 3), 'v_mps': round(float(ee[max(k, 0)] * KV), 2)})
    kmax = int(np.argmax(np.where(gm, hg, 0)))
    k2 = np.searchsorted(tt, its[kmax]) - 1
    out = {'brake_write_s': round(tb, 3), 'first_backing_sample_s': round(t_back, 3), 'back_total_m': round(float(x[-1]), 3),
           'first_hit_ge2p5g': hits, 'max_g': {'s': round(float(its[kmax]), 3), 'g': round(float(hg[kmax]), 2),
                                               'back_m': round(float(x[max(k2, 0)]), 3)},
           'vmax_mps': round(float(ee.max() * KV), 2),
           'cmd': [(round(float(zs[k]), 3), round(float(z['out'][k][1]) - N, 1)) for k in rec[:40]]}
    return out


def main():
    res = {'160847': f160847(), 'no6': fno6()}
    (HERE / 'out' / 'facts12.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    r = res['160847']
    print('160847', {k: v for k, v in r.items() if k != 'frames'})
    for x in r['frames'][:20]:
        print('   ', x)
    print('no6', {k: v for k, v in res['no6'].items() if k != 'cmd'})
    print('   cmd', res['no6']['cmd'])


if __name__ == '__main__':
    main()
