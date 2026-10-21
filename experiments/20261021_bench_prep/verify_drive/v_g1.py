#!/usr/bin/env python3
"""V-drive: 走行ループ(今の木の Driver._loop)を 10/10 の記録で回した結果の cap_* を、G1 の再生の作り
(experiments/20261016_sim_calibration/g1_replay.py の prepare = 記録の全部の IMU で位相・shadow = 上限の層)と、
フレームごとに比べる。層への入力(スロットル・舵・予測・AI 側か・車両の値の鮮度)は、ループの記録の値を使う。

    python -I v_g1.py <new_*_plain_z1.pkl> [<new_*_plain_z0.pkl> ...]
"""
import collections
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
from jetformer import safety_caps as sc  # noqa: E402

spec = importlib.util.spec_from_file_location('g1_replay_v', str(JF / 'experiments/20261016_sim_calibration/g1_replay.py'))
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)
ZI = {z: i for i, z in enumerate(sc.ZONES)}
MI = {'off': -1, 'pass': 0, 'cap': 1, 'neutral': 2, 'brake': 3}
_prep = {}


def check(path):
    with open(path, 'rb') as f:
        r = pickle.load(f)
    log = r['log']
    if log not in _prep:
        _prep[log] = g1.prepare(log)
    d = _prep[log]
    t0 = d['info']['t0_us']
    rows = r['rows']
    t_ns = np.array([t for t, _ in rows], np.int64)
    at = (t_ns // 1000 - t0) / 1e6
    it, vt, erpm = d['it'], d['vt'], d['erpm']
    ki = g1.prev_idx(it, at)
    kv = g1.prev_idx(vt, at)
    v_at = -erpm[kv] * g1.KV
    fresh = (at - vt[kv]) < 0.1
    hold = np.array([bool(x['hold']) for _, x in rows])
    sel = np.array([bool(x['select']) and not bool(x['dry_run']) for _, x in rows]) & ~hold & fresh
    pred_min = np.array([np.nanmin(x['pred'][:, 0]) if np.isfinite(x['pred'][:, 0]).any() else np.inf for _, x in rows])
    pred_sig = (pred_min <= g1.PRED_TH) & sel
    zg = d['ver']['v1.1']['zone_g'][ki]
    tr = d['ver']['v1.1']['turning'][ki]
    zone = np.array([sc.combine_zone(z, bool(t), bool(p), v=float(v), pred_min_speed=sc.PRED_MIN_SPEED)
                     for z, t, p, v in zip(zg, tr, pred_sig, v_at)], dtype=object)
    thr = np.array([float(x['cap_in_us']) for _, x in rows])
    steer = np.array([float(x['out'][0]) for _, x in rows])
    d2 = {'at': at, 'sel': sel, 'v_at': v_at, 'thr': thr, 'steer_norm': np.array([sc.steer_us_to_norm(s) for s in steer]),
          'ver': {'v1.1': {'zone': zone}}}
    sh = g1.shadow(d2, 'v1.1', sc.params_real_car())
    bad = collections.Counter()
    first = {}
    n_cmp = 0
    max_thr = 0.0
    for i, (_, x) in enumerate(rows):
        if hold[i]:
            if x['cap_mode'] not in (-1, -2) or x['cap_zone'] != -1:
                bad['hold_row'] += 1
            continue
        n_cmp += 1
        checks = {'zone': x['cap_zone'] == ZI[zone[i]], 'mode': x['cap_mode'] == MI[sh['mode'][i]],
                  'brake_start': bool(x['cap_brake_start']) == bool(sh['brake_start'][i]),
                  'thr': abs(float(x['cap_thr_us']) - float(sh['thr_out'][i])) < 1e-6}
        max_thr = max(max_thr, abs(float(x['cap_thr_us']) - float(sh['thr_out'][i])))
        if r['zone'] == 1 and sel[i]:
            checks['out_eq_shadow'] = abs(float(x['out'][1]) - float(sh['thr_out'][i])) < 1e-6
        for k, ok in checks.items():
            if not ok:
                bad[k] += 1
                first.setdefault(k, (i, str(zone[i]), int(x['cap_zone']), str(sh['mode'][i]), int(x['cap_mode']),
                                     float(x['cap_thr_us']), float(sh['thr_out'][i])))
    modes = collections.Counter(str(m) for m in sh['mode'])
    return {'file': Path(path).name, 'log': log, 'zone': r['zone'], 'frames_compared': n_cmp, 'mismatch': dict(bad),
            'first': first, 'max_thr_abs_diff_us': max_thr, 'g1_modes': dict(modes),
            'brake_starts': int(np.sum(sh['brake_start'])), 'ok': not bad}


if __name__ == '__main__':
    out = [check(p) for p in sys.argv[1:]]
    for o in out:
        print(json.dumps(o, ensure_ascii=False))
