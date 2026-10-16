#!/usr/bin/env python3
"""人のきれいな走行(clean_driving_20261011 の 52 シーン)の各フレームに「コースのどこか」を付ける(CPU だけ、約 1 分)

tools/course_fit.py の 1 段目の当てはめ(fit_log_stage1: ジャイロの向き vs ホールセンサーの距離をコース図の基準線と
全体照合 → 壁・基準線への引き寄せで補正。カメラの目印は使わない)をそのまま使い、カメラのコマごとの位置 (X, Z, 向き) を
基準線へ対応づけて、1 周の中の進み sp [m](0〜L)・横ずれ off [m](左 +)・周の番号を作る。後処理(未来のデータも使う)。

    cd ~/jetformer && python experiments/20261015_real_accuracy/ideas_ceiling/lapcoord.py
出力: cache/lapcoord.npz(frames.npz の行ごと。clean_driving 以外の行は NaN)
"""
import os
import sys

import numpy as np

REPO = '/home/jetson/jetformer'
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, REPO)
import course_fit as cf  # noqa: E402
import course_map as cm  # noqa: E402
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    z = np.load(rp.rl_dir(ev.DATASET) / 'frames.npz', allow_pickle=False)
    f = {k: z[k] for k in ('scene', 'frame', 'timestamps_us', 'log')}
    clean_scenes = sorted(x[:-4] for x in os.listdir(os.path.join(REPO, 'data/datasets/clean_driving_20261011/aligned')))
    rows = np.nonzero(np.isin(f['scene'], clean_scenes))[0]
    logs = sorted(set(f['log'][rows]))
    cl = cm.Centerline()
    n = len(f['scene'])
    out = {k: np.full(n, np.nan) for k in ('sp', 'off', 'X', 'Z', 'h', 'score')}
    lapkey = np.full(n, -1, np.int64)
    info = {}
    for li, lg in enumerate(logs):
        d, segs = cf.fit_log_stage1(lg)
        r = rows[f['log'][rows] == lg]
        tcam = (f['timestamps_us'][r].astype(np.int64) - d.t0_us) / 1e6
        info[lg] = []
        for si, (S, inf) in enumerate(segs):
            if not inf.get('fitted'):
                info[lg].append({'fitted': False})
                continue
            X, Z, h, _ = cf.frame_pose(S, S.x, tcam)
            m = np.isfinite(X)
            if not m.any():
                continue
            sp, cum, lap, sec, off = cl.progress(X[m], Z[m], h[m])
            rr = r[m]
            out['sp'][rr] = sp
            out['off'][rr] = off
            out['X'][rr], out['Z'][rr], out['h'][rr] = X[m], Z[m], h[m]
            out['score'][rr] = inf['reg']['score']
            lapkey[rr] = li * 100000 + si * 1000 + lap
            info[lg].append({'fitted': True, 'score': round(inf['reg']['score'], 3), 'frames': int(m.sum()),
                             'matched': int(np.isfinite(sp).sum())})
        print(lg, info[lg])
    mm = np.isin(f['scene'], clean_scenes)
    print('coverage(clean の行で sp あり)', round(float(np.isfinite(out['sp'][mm]).mean()), 3), 'L', round(cl.L, 2))
    np.savez_compressed(os.path.join(HERE, 'cache', 'lapcoord.npz'), lapkey=lapkey, scene=f['scene'], frame=f['frame'],
                        L=cl.L, **out)


if __name__ == '__main__':
    main()
