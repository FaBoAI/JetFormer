#!/usr/bin/env python3
"""解析 B の机上の試し 3(2026-10-25): cond_replay2.py で「IMU を入れ替えるとスロットルの差が約半分埋まる」と出たので、
IMU のどの成分が効いているかを分ける。3 回目(No.11・12・18)のフレームの IMU(直近 10 個)のうち 1 つの成分だけを、
No.9 の同じ区間・同じ進み具合のフレームのものに入れ替える。開ループ。車は動かさない。

    jf-heavy --lock big --mem 3G -- /home/jetson/miniconda/envs/jetracer/bin/python -I cond_replay3.py

IMU の軸(記録から確かめた): 加速度 0 = 横(v·gz と相関 −0.95)、1 = 前後(速さの変化と相関 0.53)、2 = 上下(9.7 m/s²)。
  acc_lat(横の加速度)・acc_lon(前後の加速度)・acc_vert(上下)・gyro_z(ヨー)・gyro_xy(ロール・ピッチ)・imu(全部)
出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_replay3.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261024_report/cond')
import cond_replay2 as r2  # noqa: E402
from cond_replay2 import Log, match, normalize_inputs, runtime, post_mod, RUN, OUT, LOGS, DET, NEUTRAL  # noqa: E402

PAIRS = [(11, 9), (12, 9), (18, 9)]
COMP = {'imu': list(range(6)), 'acc_lat': [0], 'acc_lon': [1], 'acc_vert': [2], 'gyro_z': [5], 'gyro_xy': [3, 4]}
STEP = 3
BATCH = 48


def main():
    model, cfg, _, _ = runtime.load_run(RUN, 'best', device='cuda')
    nm = cfg['num_modes']
    logs = {no: Log(lg) for no, lg in LOGS.items()}
    out = {'doc': __doc__, 'pairs': {}}
    for a, b in PAIRS:
        src, dst = logs[a], logs[b]
        sel = np.flatnonzero(src.drv & np.isfinite(src.p) & np.isin(src.loc, range(1, 8)))[::STEP]
        state = {c: 0 for c in range(8)}
        mt = [match(src, dst, i, state) for i in sel]
        keep = [k for k, m in enumerate(mt) if m is not None]
        sel = sel[keep]
        mt = [mt[k] for k in keep]
        names = ['none', 'donor'] + list(COMP)
        thr = {s: np.full(len(sel), np.nan) for s in names}
        for b0 in range(0, len(sel), BATCH):
            P = [src.parts(i) for i in sel[b0:b0 + BATCH]]
            Q = [dst.parts(j) for j in mt[b0:b0 + BATCH]]
            for s in names:
                ims, tofs, imus, sts, modes = [], [], [], [], []
                for p, q in zip(P, Q):
                    x = dict(q) if s == 'donor' else dict(p)
                    if s in COMP:
                        h = p['imu'].copy()
                        h[:, COMP[s]] = q['imu'][:, COMP[s]]
                        x['imu'] = h
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
            base, don = float(np.mean(thr['none'][m])), float(np.mean(thr['donor'][m]))
            row = {'n': int(m.sum()), 'thr_self': round(base, 1), 'thr_donor': round(don, 1)}
            for s in COMP:
                v = float(np.mean(thr[s][m]))
                row['thr_' + s] = round(v, 1)
                row['closed_' + s] = round((v - base) / (don - base), 2) if abs(don - base) >= 3 else None
            res['sections'][name] = row
        out['pairs']['%d<-%d' % (a, b)] = res
        print(a, b, json.dumps(res['sections']['all']), flush=True)
    (OUT / 'cond_replay3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
