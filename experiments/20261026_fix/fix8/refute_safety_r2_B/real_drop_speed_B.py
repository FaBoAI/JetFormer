"""票 B: カメラが来なかった AI 側の間(real_cam_B)の、速さごとの起きやすさ(1 フレームあたり)。data/raw は読むだけ。
速さは VEHICLE の eRPM(1000 eRPM = 0.5 m/s。負 = 前へ)。全部の AI 側の処理したフレームを速さの帯に分けて数え、
その帯で起きた間の数を割る。
    python -I -B real_drop_speed_B.py → out/real_drop_speed_B.json
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

EDGES = [-99.0, -1.0, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 99.0]   # 前へ +(m/s)。-1 より下 = 後ろへ 1 m/s より速い


def band(vf):
    for q in range(len(EDGES) - 1):
        if EDGES[q] <= vf < EDGES[q + 1]:
            return q
    return len(EDGES) - 2


def main():
    cam = json.loads((HERE / 'out' / 'real_cam_B.json').read_text(encoding='utf-8'))
    drops = {}
    for r in cam['rows']:
        if r['cam_frames_inside'] == 0:
            drops.setdefault(r['log'], []).append(r['s'])
    nf = np.zeros(len(EDGES) - 1, int)
    nd = np.zeros(len(EDGES) - 1, int)
    for log in T.raw_logs():
        for scenes in T.units(log, 'concat'):
            cat, meta, sk = T.load_unit(log, scenes)
            if cat is None:
                continue
            z = cat['AI_CMD']
            t0 = T.log_t0_us(log) or int(z['timestamps_us'][0])
            ts = (z['timestamps_us'].astype(np.int64) - t0) / 1e6
            sel = z['select'].astype(bool)
            hold = z['hold'].astype(bool)
            dry = z['dry_run'].astype(bool) if 'dry_run' in z else np.zeros(len(ts), bool)
            ok = sel & ~hold & ~dry
            v = cat['VEHICLE']
            vt = (v['timestamps_us'].astype(np.int64) - t0) / 1e6
            ve = v['erpm'].astype(float)
            k = np.clip(np.searchsorted(vt, ts) - 1, 0, max(len(ve) - 1, 0))
            vf = -ve[k] * 0.5 / 1000.0 if len(ve) else np.zeros(len(ts))
            for i in np.nonzero(ok)[0]:
                nf[band(vf[i])] += 1
            for s in drops.get(log, []):
                kk = int(np.clip(np.searchsorted(vt, s) - 1, 0, len(ve) - 1))
                nd[band(-ve[kk] * 0.5 / 1000.0)] += 1
    out = []
    for q in range(len(EDGES) - 1):
        out.append({'v_fwd_from': EDGES[q], 'to': EDGES[q + 1], 'frames': int(nf[q]), 'drops': int(nd[q]),
                    'per_frame': round(nd[q] / max(nf[q], 1), 5)})
    tot = {'frames': int(nf.sum()), 'drops': int(nd.sum()), 'per_frame': round(nd.sum() / max(nf.sum(), 1), 5)}
    res = {'bands': out, 'total': tot}
    (HERE / 'out' / 'real_drop_speed_B.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
