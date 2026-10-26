#!/usr/bin/env python3
"""解析 B の机上の試し 2(2026-10-25): 入れ替えの試し。3 回目(No.11・12・18)のフレームに、2 回目の No.9 の
「同じ区間・同じ進み具合」のフレームの入力を 1 つだけ入れ替えてモデル B に入れ、スロットルが No.9 にどれだけ近づくかを測る。
(逆向きに、No.9 のフレームに No.11 の入力を入れたものも測る)。開ループ。車は動かさない。

    jf-heavy --lock big --mem 3G -- /home/jetson/miniconda/envs/jetracer/bin/python -I cond_replay2.py

入れ替えるもの: img(カメラ画像)・imu(直近 10 個の IMU)・tof(STEVAL-VL53L9 の距離)・erpm(回転数)。
対応づけ: 区間(ラベルの loc_detail)の 1 回の通過の中の進み具合 p(0〜1)がいちばん近いフレーム(相手の全通過から、
通過の番号を順に回して選ぶ)。埋まった差の割合 = (入れ替えた後 − もとの値) ÷ (相手のもとの値 − もとの値)(区間ごとの平均で)。
出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_replay2.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
import cv2  # noqa: E402
from jetformer.train_data import normalize_inputs  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.policies import post as post_mod  # noqa: E402

RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
RUN = JF / 'data' / 'runs' / 'clean_driving_20261011-nocmd-s3-20261011-024626'
OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report/cond_data')
LOGS = {9: '20261024-162033', 11: '20261024-183123', 12: '20261024-183437', 18: '20261024-185703'}
PAIRS = [(11, 9), (12, 9), (18, 9), (9, 11)]          # (フレームの走行, 入れ替えの相手)
DET = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
SWAPS = ['none', 'img', 'imu', 'tof', 'erpm']
NEUTRAL = 1496.0
STEP = 2
BATCH = 48


def cat(lg, name, keys):
    parts = {k: [] for k in keys}
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/' + name + '.npz') as z:
                for k in keys:
                    parts[k].append(z[k])
        except Exception:
            continue
    o = {k: np.concatenate(v) for k, v in parts.items()}
    idx = np.argsort(o['timestamps_us'], kind='stable')
    return {k: v[idx] for k, v in o.items()}


class Log(object):
    def __init__(self, lg):
        with np.load(LAB / (lg + '.npz')) as z:
            lab = {k: z[k] for k in ('t_us', 'ai', 'dry_run', 'hold', 'assist', 'loc_detail')}
        self.A = cat(lg, 'AI_CMD', ['timestamps_us', 'mode_id', 'model_pred'])
        self.T = cat(lg, 'TOF_FRONT', ['timestamps_us', 'depth'])
        I = cat(lg, 'IMU', ['timestamps_us', 'accel', 'gyro'])
        self.It = I['timestamps_us']
        self.imu = np.concatenate([I['accel'], np.deg2rad(I['gyro'])], axis=1).astype(np.float32)
        self.V = cat(lg, 'VEHICLE', ['timestamps_us', 'erpm'])
        files, ct = [], []
        for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
            with np.load(s + '/CAM_FRONT.npz') as c:
                files += [s + '/' + f for f in c['files']]
                ct.append(c['timestamps_us'])
        self.fmap = dict(zip(np.concatenate(ct).tolist(), files))
        drv = lab['ai'] & ~lab['dry_run'] & ~lab['hold'] & (lab['assist'] == 0)
        k = np.clip(np.searchsorted(self.A['timestamps_us'], lab['t_us']), 0, len(self.A['timestamps_us']) - 1)
        drv &= self.A['timestamps_us'][k] == lab['t_us']
        self.t_us, self.loc, self.drv, self.ak = lab['t_us'], lab['loc_detail'].astype(int), drv, k
        # 区間の通過ごとの進み具合
        self.p = np.full(len(drv), np.nan)
        self.pass_id = np.full(len(drv), -1)
        edges = np.flatnonzero(np.diff(np.r_[-1, self.loc, -1]) != 0)
        pid = 0
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a >= 5:
                self.p[a:b] = (np.arange(b - a) + 0.5) / (b - a)
                self.pass_id[a:b] = pid
                pid += 1

    def parts(self, i):
        cam_t = int(self.t_us[i])
        img = cv2.imread(self.fmap[cam_t])
        jt = max(np.searchsorted(self.T['timestamps_us'], cam_t, side='right') - 1, 0)
        ji = np.searchsorted(self.It, cam_t, side='right')
        jv = max(np.searchsorted(self.V['timestamps_us'], cam_t, side='right') - 1, 0)
        hist = self.imu[max(ji - 10, 0):ji]
        if len(hist) < 10:
            hist = np.vstack([np.repeat(hist[:1], 10 - len(hist), 0), hist])
        return {'img': img, 'tof': self.T['depth'][jt], 'imu': hist, 'erpm': float(self.V['erpm'][jv]),
                'mode': int(self.A['mode_id'][self.ak[i]])}


def match(src, dst, i, rng_state):
    """src のフレーム i に対応する dst のフレーム(同じ区間・p がいちばん近い。通過は順に回す)"""
    c = src.loc[i]
    cand = np.flatnonzero(dst.drv & (dst.loc == c) & np.isfinite(dst.p))
    if not len(cand):
        return None
    pids = np.unique(dst.pass_id[cand])
    pid = pids[rng_state[c] % len(pids)]
    rng_state[c] += 1
    cc = cand[dst.pass_id[cand] == pid]
    return int(cc[np.argmin(np.abs(dst.p[cc] - src.p[i]))])


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
        thr = {s: np.full(len(sel), np.nan) for s in SWAPS + ['donor']}
        for b0 in range(0, len(sel), BATCH):
            P = [src.parts(i) for i in sel[b0:b0 + BATCH]]
            Q = [dst.parts(j) for j in mt[b0:b0 + BATCH]]
            for s in SWAPS + ['donor']:
                ims, tofs, imus, sts, modes = [], [], [], [], []
                for p, q in zip(P, Q):
                    x = dict(q) if s == 'donor' else dict(p)
                    if s in ('img', 'imu', 'tof', 'erpm'):
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
        for c, name in enumerate(DET):
            m = loc == c
            if m.sum() < 15:
                continue
            base, don = float(np.mean(thr['none'][m])), float(np.mean(thr['donor'][m]))
            row = {'n': int(m.sum()), 'thr_self': round(base, 1), 'thr_donor': round(don, 1)}
            for s in ('img', 'imu', 'tof', 'erpm'):
                v = float(np.mean(thr[s][m]))
                row['thr_' + s] = round(v, 1)
                row['closed_' + s] = round((v - base) / (don - base), 2) if abs(don - base) >= 3 else None
            res['sections'][name] = row
        base, don = float(np.mean(thr['none'])), float(np.mean(thr['donor']))
        res['all'] = {'thr_self': round(base, 1), 'thr_donor': round(don, 1)}
        for s in ('img', 'imu', 'tof', 'erpm'):
            v = float(np.mean(thr[s]))
            res['all']['thr_' + s] = round(v, 1)
            res['all']['closed_' + s] = round((v - base) / (don - base), 2) if abs(don - base) >= 3 else None
        out['pairs']['%d<-%d' % (a, b)] = res
        np.savez_compressed(OUT / ('replay2_%d_from_%d.npz' % (a, b)), sel=sel, match=np.array(mt), loc=loc,
                            **{'thr_' + s: v for s, v in thr.items()})
        print(a, b, json.dumps(res['all']), flush=True)
    (OUT / 'cond_replay2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
