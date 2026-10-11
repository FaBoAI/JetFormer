#!/usr/bin/env python3
"""AI走行中の交代のあとの「人の操作」のうち、HG-DAgger の正例に使えそうな量を数える(読み取りのみ、CPU)

    nice -n 15 python intervention_budget.py > intervention_budget.json

交代 = VEHICLE.select が True → False(直前 0.5秒以内に途切れがないもの。基板の再起動の読み取りの切れ目は
0.5秒以内に AI 側へ戻るので除く)。交代のあと AI に戻るまで(最大 5秒)の人の区間で、
  - 前進フレーム: カメラフレームのうち、前進 eRPM > 1500 かつ スロットル指令 < 1478µs(前進側)
  - 後退・ブレーキを含むか: スロットル指令 > 1538µs のサンプルがあるか
  - 交代の前 3秒に衝突(水平 2.5G、跳ね除く)があったか
を数える。HG-DAgger の正例の候補は「前進フレーム」(後退・止まっている・位置直しは除く)。
"""
import json
import os
import sys

import numpy as np

RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665
AI = {'20261010-155820': 'AIトライ', '20261010-160306': 'run_001', '20261010-163247': 'run002',
      '20261010-171639': 'ai_run_003', '20261010-172505': 'ai_run_004', '20261010-172910': 'ai_run_006',
      '20261010-173445': 'ai_run_007', '20261010-181504': 'ai_run_008', '20261010-203924': 'ai_run_009',
      '20261010-204208': 'ai_run_010', '20261010-204605': 'ai_run_101'}


def load(log):
    root = os.path.join(RAW, log)
    V, I, C = [], [], []
    for s in sorted(x for x in os.listdir(root) if x.startswith(log + '-')):
        try:
            v = np.load(os.path.join(root, s, 'VEHICLE.npz'))
            i = np.load(os.path.join(root, s, 'IMU.npz'))
            c = np.load(os.path.join(root, s, 'CAM_FRONT.npz'), allow_pickle=True)
            V.append((v['timestamps_us'], v['select'], v['erpm'], v['pwm']))
            I.append((i['timestamps_us'], i['accel']))
            C.append(c['timestamps_us'])
        except Exception:
            continue
    vt = np.concatenate([x[0] for x in V]) / 1e6
    t0 = vt[0]
    d = dict(vt=vt - t0, sel=np.concatenate([x[1] for x in V]).astype(bool), erpm=np.concatenate([x[2] for x in V]).astype(float),
             pwm=np.concatenate([x[3] for x in V]).astype(float), it=np.concatenate([x[0] for x in I]) / 1e6 - t0,
             acc=np.concatenate([x[1] for x in I]).astype(float), ct=np.concatenate(C) / 1e6 - t0)
    hz = np.hypot(d['acc'][:, 0], d['acc'][:, 1]) / G
    ev = []
    for k in np.nonzero(hz >= 2.5)[0]:
        t = d['it'][k]
        if ev and t - ev[-1] <= 0.6:
            continue
        near = (np.abs(d['it'] - t) <= 0.05) & (hz > 1.0)
        if near.sum() > 1:
            ev.append(t)
    d['ev'] = np.array(ev)
    return d


def main():
    rows = []
    for log, name in AI.items():
        d = load(log)
        vt, sel = d['vt'], d['sel']
        idx = np.nonzero(sel[:-1] & ~sel[1:])[0]
        for k in idx:
            t = vt[k + 1]
            if vt[k + 1] - vt[k] > 0.5:
                continue
            nxt = np.nonzero((vt > t) & sel)[0]
            t_back = vt[nxt[0]] if len(nxt) else vt[-1]
            if t_back - t < 0.5:          # 基板の再起動などの読み取りの切れ目
                continue
            t_end = min(t_back, t + 5.0)
            mv = (vt >= t) & (vt < t_end)
            rev = bool(np.any(d['pwm'][mv, 1] > 1538))
            cm = (d['ct'] >= t) & (d['ct'] < t_end)
            ce = np.interp(d['ct'][cm], vt, -d['erpm'])
            cthr = np.interp(d['ct'][cm], vt, d['pwm'][:, 1])
            fwd = (ce > 1500) & (cthr < 1478)
            pre = d['ev'][(d['ev'] <= t) & (d['ev'] > t - 3.0)]
            rows.append(dict(log=log, run=name, t=round(float(t), 2), human_s=round(float(t_back - t), 2),
                             impact_before=bool(len(pre)), reverse=rev, frames=int(cm.sum()), fwd_frames=int(fwd.sum()),
                             erpm_at=int(np.interp(t, vt, -d['erpm']))))
    s = {'takeovers': len(rows), 'with_impact_before': sum(r['impact_before'] for r in rows),
         'with_reverse': sum(r['reverse'] for r in rows), 'frames_5s': sum(r['frames'] for r in rows),
         'fwd_frames_5s': sum(r['fwd_frames'] for r in rows),
         'fwd_frames_no_reverse': sum(r['fwd_frames'] for r in rows if not r['reverse']),
         'fwd_frames_moving_at_takeover': sum(r['fwd_frames'] for r in rows if r['erpm_at'] > 3000 and not r['impact_before']),
         'n_moving_at_takeover_no_impact': sum(1 for r in rows if r['erpm_at'] > 3000 and not r['impact_before'])}
    json.dump({'summary': s, 'rows': rows}, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
