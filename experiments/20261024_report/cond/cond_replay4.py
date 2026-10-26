#!/usr/bin/env python3
"""解析 B の机上の試し 4(2026-10-25): cond_replay2.py で「画像を入れ替えると、スロットルの差が約半分埋まる」と出たので、
画像のどの部分が効いているかを分ける。3 回目(No.11・12・18)のフレームの画像のうち、
  hood  = 下の 56 画素(ボンネットと、修理のあとに立って写るようになった黒い線)だけ
  scene = 上の 168 画素(コースの見え方 = 走る位置・向き)だけ
を No.9 の同じ区間・同じ進み具合のフレームのものに入れ替えて、モデル B のスロットルを測る。開ループ。車は動かさない。

    jf-heavy --lock big --mem 3G -- /home/jetson/miniconda/envs/jetracer/bin/python -I cond_replay4.py

出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_replay4.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261024_report/cond')
from cond_replay2 import Log, match, normalize_inputs, runtime, post_mod, RUN, OUT, LOGS, DET, NEUTRAL  # noqa: E402

PAIRS = [(11, 9), (12, 9), (18, 9)]
CUT = 168
VAR = ['none', 'donor', 'img', 'hood', 'scene']
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
        thr = {s: np.full(len(sel), np.nan) for s in VAR}
        for b0 in range(0, len(sel), BATCH):
            P = [src.parts(i) for i in sel[b0:b0 + BATCH]]
            Q = [dst.parts(j) for j in mt[b0:b0 + BATCH]]
            for s in VAR:
                ims, tofs, imus, sts, modes = [], [], [], [], []
                for p, q in zip(P, Q):
                    x = dict(q) if s == 'donor' else dict(p)
                    if s == 'img':
                        x['img'] = q['img']
                    elif s == 'hood':
                        im = p['img'].copy()
                        im[CUT:] = q['img'][CUT:]
                        x['img'] = im
                    elif s == 'scene':
                        im = p['img'].copy()
                        im[:CUT] = q['img'][:CUT]
                        x['img'] = im
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
            for s in ('img', 'hood', 'scene'):
                v = float(np.mean(thr[s][m]))
                row['thr_' + s] = round(v, 1)
                row['closed_' + s] = round((v - base) / (don - base), 2) if abs(don - base) >= 3 else None
            res['sections'][name] = row
        out['pairs']['%d<-%d' % (a, b)] = res
        print(a, b, json.dumps(res['sections']['all']), flush=True)
    (OUT / 'cond_replay4.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
