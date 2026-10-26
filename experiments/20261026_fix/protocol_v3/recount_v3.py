#!/usr/bin/env python3
"""protocol_v3 の閾値を別のコードで数え直す(desk_v2.py の car_out・go_frac と eval_v2.py の metrics を使う)。numpy だけ。
流し方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I recount_v3.py"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

C3 = Path('/home/jetson/jetformer/experiments/20261025_train_v2/c3/desk_v2.py')
EV = Path('/home/jetson/jetformer/experiments/20261025_train_v2/eval_v2.py')
P3 = json.loads(Path('/home/jetson/jetformer_reports/fix_20261026/protocol_v3.json').read_text())
DESK = Path('/home/jetson/jetformer_reports/train_v2_20261025/desk')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(m)
    return m


def main():
    D = load(C3, 'desk_v2_copy')
    E = load(EV, 'eval_v2_copy')
    S = np.load(DESK / 'sets_v2.npz')
    st, ss, sd = S['straight'].astype(bool), S['sess'], S['stand'].astype(bool)
    pb = np.load(DESK / 'pred' / ('%s_tab.npz' % D.B_RUN))['pred'].astype(np.float64)
    refb = (1441.0, 1496.0)
    v1023, v1010, back = [], [], []
    for rid, run, *_ in E.FIVE:
        pc = np.load(DESK / 'pred' / ('%s_tab.npz' % run))['pred'].astype(np.float64)
        v1010.append(D.thr_vs_b(pc, pb, refb, refb, st & (ss == 0), 0.9, 0.9))
        v1023.append(D.thr_vs_b(pc, pb, refb, refb, st & (ss == 1), 0.9, 0.9))
        t0 = D.car_out(pc[sd][:, 0], refb, 0.9)[:, 1]
        back.append(float(np.mean(t0 > refb[1] + 30.0)))
    proto2 = json.loads(Path('/home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json').read_text())
    Sb = E.load_bundle(proto2)
    five = E.five_preds(proto2)
    ivs = []
    for rid, *_ in E.FIVE:
        d = np.abs(five[rid]['best'] - Sb['y'])
        ivs.append(float(d[Sb['interv']][:, :, 0].mean()))
    got = {'g1023': max(10.0, 2 * np.std(v1023, ddof=1) * math.sqrt(1.2)),
           'g1010': max(10.0, 2 * np.std(v1010, ddof=1) * math.sqrt(1.2)),
           'g_back': float(np.mean(back) + 2 * np.std(back, ddof=1) * math.sqrt(1.2)),
           'ivs_mean': float(np.mean(ivs)), 'ivs_n4': float(2 * np.std(ivs, ddof=1) * math.sqrt(1 / 4 + 1 / 5))}
    T = P3['thresholds_us']
    want = {'g1023': T['gate_straight_1023']['value'], 'g1010': T['gate_straight_1010']['value'],
            'g_back': T['gate_standstill_back']['value'], 'ivs_mean': T['interv_steer']['five_mean_best'],
            'ivs_n4': T['interv_steer']['thresholds_best']['vs5_n4']}
    ok = all(abs(got[k] - want[k]) <= (0.002 if k != 'g_back' else 0.0001) for k in got)
    out = {'got': {k: round(v, 5) for k, v in got.items()}, 'protocol': want, 'ok': ok}
    Path('/home/jetson/jetformer_reports/fix_20261026/recount_v3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
