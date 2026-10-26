#!/usr/bin/env python3
"""解析 B の机上の試し 5(2026-10-25): 入れ替えの相手を「同じ区間・同じ進み具合・同じ速さ(eRPM が近い)」の No.9 のフレームに
したら、画像・IMU を入れ替えてもスロットルが上がらないか(= モデルは画像・IMU から自分の速さ・加速を読み取って、
速いときほど踏む、という自分を強める動きをしているか)を確かめる。開ループ。車は動かさない。

    jf-heavy --lock big --mem 3G -- /home/jetson/miniconda/envs/jetracer/bin/python -I cond_replay5.py

相手の選び方: 同じ区間の No.9 の全フレームから、費用 = |進み具合の差| + |eRPM の差| / 1500 がいちばん小さいもの。
比べるもの: もと / 速さをそろえた相手の画像 / IMU / 入力すべて(と、そのときの eRPM の差)
出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_replay5.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261024_report/cond')
from cond_replay2 import Log, normalize_inputs, runtime, post_mod, RUN, OUT, LOGS, DET, NEUTRAL  # noqa: E402

PAIRS = [(11, 9), (12, 9), (18, 9)]
VAR = ['none', 'donor', 'img', 'imu']
STEP = 3
BATCH = 48


def erpm_at(lg):
    e = np.interp(lg.t_us.astype(np.float64), lg.V['timestamps_us'].astype(np.float64), lg.V['erpm'].astype(np.float64))
    return -e


def main():
    model, cfg, _, _ = runtime.load_run(RUN, 'best', device='cuda')
    nm = cfg['num_modes']
    logs = {no: Log(lg) for no, lg in LOGS.items()}
    out = {'doc': __doc__, 'pairs': {}}
    for a, b in PAIRS:
        src, dst = logs[a], logs[b]
        es, ed = erpm_at(src), erpm_at(dst)
        sel = np.flatnonzero(src.drv & np.isfinite(src.p) & np.isin(src.loc, range(1, 8)))[::STEP]
        mt = []
        for i in sel:
            cand = np.flatnonzero(dst.drv & (dst.loc == src.loc[i]) & np.isfinite(dst.p))
            if not len(cand):
                mt.append(None)
                continue
            cost = np.abs(dst.p[cand] - src.p[i]) + np.abs(ed[cand] - es[i]) / 1500.0
            mt.append(int(cand[np.argmin(cost)]))
        keep = [k for k, m in enumerate(mt) if m is not None]
        sel = sel[keep]
        mt = np.array([mt[k] for k in keep])
        thr = {s: np.full(len(sel), np.nan) for s in VAR}
        for b0 in range(0, len(sel), BATCH):
            P = [src.parts(i) for i in sel[b0:b0 + BATCH]]
            Q = [dst.parts(j) for j in mt[b0:b0 + BATCH]]
            for s in VAR:
                ims, tofs, imus, sts, modes = [], [], [], [], []
                for p, q in zip(P, Q):
                    x = dict(q) if s == 'donor' else dict(p)
                    if s in ('img', 'imu'):
                        x[s] = q[s]
                    veh = np.array([1441.0, NEUTRAL, 2000.0, x['erpm']], np.float32)
                    im, tf, iu, st = normalize_inputs(x['img'], x['tof'], x['imu'], veh, use_command=False)
                    ims.append(im); tofs.append(tf); imus.append(iu); sts.append(st)
                    oh = np.zeros(nm, np.float32)
                    oh[p['mode']] = 1.0
                    modes.append(oh)
                r = runtime.predict(model, {'image': np.stack(ims), 'tof': np.stack(tofs), 'imu': np.stack(imus),
                                            'state': np.stack(sts), 'mode': np.stack(modes)})
                act = r['actions'] * post_mod.PWM_SCALE + post_mod.PWM_CENTER
                thr[s][b0:b0 + len(P)] = NEUTRAL - act[:, 2, 1]
            print(a, b, b0, len(sel), flush=True)
        loc = src.loc[sel]
        res = {'frames': int(len(sel)), 'sections': {}}
        for c, name in [(-1, 'all')] + list(enumerate(DET)):
            m = np.ones(len(sel), bool) if c < 0 else loc == c
            if m.sum() < 15:
                continue
            row = {'n': int(m.sum()), 'erpm_self': round(float(np.mean(es[sel][m]))), 'erpm_donor': round(float(np.mean(ed[mt][m])))}
            for s in VAR:
                row['thr_' + s] = round(float(np.mean(thr[s][m])), 1)
            res['sections'][name] = row
        out['pairs']['%d<-%d' % (a, b)] = res
        print(a, b, json.dumps(res['sections']['all']), flush=True)
    (OUT / 'cond_replay5.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
