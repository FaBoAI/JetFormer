"""10/24 20261024-160847 を本物の drive._loop に開いたループで流し直し(rec_replay11 と同じ)、今の ON・fix7 の前(f7b)の ON・
今の OFF の PCA9685 に書いた値が違うフレームを並べる(読むだけ)。
    python -I -B probe_rec160847.py → out/probe_rec160847.json・標準出力
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rec_replay11 as R  # noqa: E402

h11, RR = R.h11, R.RR


def main(log='20261024-160847'):
    h11.check_now()
    RR._EPOCH_OFF[0] = int(R.clock.to_epoch_us(0)) * 1000
    meta, rec, problems = RR.load_rec(log)
    dr = meta.get('drive') or {}
    p = dict(h11.drive.DEFAULT_PARAMS)
    p.update({k: v for k, v in (dr.get('params') or {}).items() if k in h11.drive.DEFAULT_PARAMS})
    p.update(speed_ceiling=0, speed_auto=0, launch_no_kick=0, launch_boost=0)
    p['_ref'] = (float(dr.get('steer_center_us', 1441)), float(dr.get('throttle_neutral_us', 1496)))
    Z = rec['AI_CMD']
    t_lo, t_hi = int(Z['timestamps_us'][0]), int(Z['timestamps_us'][-1])
    fpath = Path(h11.drive.FLOOR_REF_PATH)
    floor = np.load(fpath) if fpath.exists() else None
    out = {}
    try:
        rr = {k: R.run_v(v, rec, f(p), t_lo, t_hi, floor) for k, (v, f) in
              {'now_on': ('now', h11.ON), 'f7b_on': ('f7b', h11.ON), 'now_off': ('now', h11.OFF),
               'r1_on': ('r1', h11.ON)}.items()}
    finally:
        R.shutil.rmtree(R.TMP, ignore_errors=True)
        h11.cleanup()
    w = {k: v[2].w for k, v in rr.items()}
    t0 = t_lo * 1000
    diffs = [(i, round((a[0] - t0) / 1e9, 3), a[1:], b[1:], c[1:]) for i, (a, b, c) in
             enumerate(zip(w['now_on'], w['f7b_on'], w['now_off'])) if a != b or a != c]
    print('writes', {k: len(v) for k, v in w.items()}, 'diff frames(now_on vs f7b_on / now_off)', len(diffs))
    for d in diffs[:60]:
        print(d)
    ev = {k: [(e[0], {kk: e[1].get(kk) for kk in ('what', 'phase', 'backed_s')},
               round((int(e[1].get('cam_epoch_us', 0)) * 1000 - RR._EPOCH_OFF[0] - t0) / 1e9, 3) if e[1].get('cam_epoch_us') else None)
              for e in v[1].events if e[0] == 'recovery_guard'] for k, v in rr.items()}
    print('guard events', json.dumps(ev, ensure_ascii=False, default=str))
    # 違ったあたりの車両の値(記録)
    if diffs:
        ts = diffs[0][1]
        V = rec['VEHICLE']
        vt = (V['timestamps_us'] - t_lo) / 1e6
        m = (vt > ts - 0.6) & (vt < ts + 1.2)
        print('veh erpm around', [(round(float(a), 3), int(b), bool(s)) for a, b, s in zip(vt[m], V['erpm'][m], V['select'][m])][::2])
        zt = (Z['timestamps_us'] - t_lo) / 1e6
        mz = (zt > ts - 0.6) & (zt < ts + 1.2)
        print('record out', [(round(float(a), 3), [round(float(x)) for x in o], int(k) if 'assist' in Z else None)
                             for a, o, k in zip(zt[mz], Z['out'][mz], Z['assist'][mz] if 'assist' in Z else [0] * int(mz.sum()))])
    out = {'diffs': diffs, 'events': ev}
    (HERE / 'out' / 'probe_rec160847.json').write_text(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main(*sys.argv[1:2])
