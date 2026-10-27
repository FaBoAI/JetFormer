"""safety1-2 の確かめ: 発進の前進が安全の理由(near・tof)で始めなかった区間で、StartKick が弱い前進を 2 秒出すか。
偽物の走行ループ(tests/test_launch_drive.py の run・build)を使う。実機・GPU・torch は使わない。
    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix2/safety1_2/repro_kick_after_skip.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

import test_launch_drive as T
from jetformer import drive

C, N = T.REF


def scene(d_mm):
    # 人側 1.0 秒(止まっている)→ AI 側のまま 3 秒。ずっと止まっている(壁に当たって動けない形)。モデルは中立
    tof = None if d_mm is None else (lambda s: T.tof_near(d_mm))
    return T.build(4.2, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof)


def one(mod, params, d_mm, floor=True):
    frames, imu = scene(d_mm)
    mp = MonkeyPatch()
    with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as td:
        try:
            d, pca, rec = T.run(mod, mp, Path(td), params, frames, imu, floor=floor)
        finally:
            mp.undo()
    thr = [p[1] for p in pca.pairs]
    kick = [x for x in thr if abs(x - (N - 80.0)) < 0.5]
    skips = [e[1].get('why') for e in rec.events if e[0] == 'launch_boost']
    ttc = [r.get('tof_ttc', r.get('ttc')) for _, r in rec.rows if r.get('select') or True]
    rows = [r for _, r in rec.rows]
    ttcs = sorted({round(float(r['ttc']), 2) for r in rows if 'ttc' in r and np.isfinite(r['ttc'])})
    return dict(n_frames=len(thr), kick_frames=len(kick), kick_s=round(len(kick) / 30.0, 2),
                thr_set=sorted({round(x, 1) for x in thr}), skips=sorted(set(skips)),
                start_kicks=d.st['stats'].get('start_kicks'), launch_boosts=d.st['stats'].get('launch_boosts'),
                ttc_range=(ttcs[0], ttcs[-1]) if ttcs else None, row_keys=sorted(rows[0].keys())[:0])


def main():
    before = T.load_before()
    for d_mm in (350, 500, 900, None):
        for sk in (1, 0):
            p = dict(T.PARAMS, tof_guard=1, start_kick=sk, launch_boost=1)
            r = one(drive, p, d_mm)
            pb = {k: v for k, v in dict(T.PARAMS, tof_guard=1, start_kick=sk).items() if k in before.DEFAULT_PARAMS}
            r0 = one(before, pb, d_mm)
            print('d=%s start_kick=%d | ON: kick %d frames (%.2f s) thr=%s skips=%s start_kicks=%s boosts=%s ttc=%s'
                  ' | before(launch 無し): kick %d frames thr=%s' % (
                      d_mm, sk, r['kick_frames'], r['kick_s'], r['thr_set'], r['skips'], r['start_kicks'],
                      r['launch_boosts'], r['ttc_range'], r0['kick_frames'], r0['thr_set']))
    # 床の基準が無い(ToF ガードが作れない)
    for sk in (1, 0):
        p = dict(T.PARAMS, tof_guard=1, start_kick=sk, launch_boost=1)
        r = one(drive, p, 500, floor=False)
        print('no floor ref start_kick=%d | ON: kick %d frames (%.2f s) thr=%s skips=%s' % (
            sk, r['kick_frames'], r['kick_s'], r['thr_set'], r['skips']))
    # tof_guard 0
    p = dict(T.PARAMS, tof_guard=0, start_kick=1, launch_boost=1)
    r = one(drive, p, 500)
    print('tof_guard=0 start_kick=1 d=500 | ON: kick %d frames (%.2f s) thr=%s skips=%s' % (
        r['kick_frames'], r['kick_s'], r['thr_set'], r['skips']))


if __name__ == '__main__':
    main()
