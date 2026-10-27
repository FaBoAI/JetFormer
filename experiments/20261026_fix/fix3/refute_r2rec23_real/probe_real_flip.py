"""反証 r2-rec2-3: 今までの本物の記録で、AI の区間の終わり(人側にした)のすぐあとに衝突がある場面がどれだけあるか。読むだけ。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        /home/jetson/jetformer/experiments/20261026_fix/fix3/refute_r2rec23_real/probe_real_flip.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/diag_launch')
sys.path.insert(0, '/home/jetson/jetformer/tools')
import collect as C  # noqa: E402
import label_events as le  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def main():
    logs = sorted(p.name[:-4] for p in C.LAB.glob('*.npz'))
    res = {'n_logs': 0, 'n_seg': 0, 'n_seg_human_end': 0, 'after_0_0p2': [], 'after_0p2_0p5': [], 'before_0p3': [],
           'errors': []}
    for log in logs:
        try:
            d = C.load(log)
        except Exception as e:  # noqa: BLE001
            res['errors'].append([log, repr(e)[:120]])
            continue
        res['n_logs'] += 1
        z = d['z']
        t, ai = z['t_s'], z['ai'].astype(bool)
        A = d['anchors']
        for i0, i1 in le.runs(ai):
            t0, t1 = float(t[i0]), float(t[i1])
            res['n_seg'] += 1
            human_end = i1 < len(t) - 1
            if not human_end:
                continue
            res['n_seg_human_end'] += 1
            e0 = float(np.interp(t0, d['vt'], d['erpm']))
            launch = abs(e0) < C.STOP_ERPM
            a_in = [float(a) for a in A if t0 < a <= t1]
            v1 = float(np.interp(t1, d['vt'], d['v']))
            info_base = {'log': log, 't0': round(t0, 2), 't1': round(t1, 3), 'launch': launch,
                         'anchors_in_seg': len(a_in), 'since_last_anchor_s': round(t1 - a_in[-1], 2) if a_in else None,
                         'v_at_t1': round(v1, 2)}
            for h in d['hits']:
                dt = h[0] - t1
                rec = dict(info_base, hit_t=h[0], g=round(h[1], 1), dt=round(dt, 3))
                if 0 < dt <= 0.2:
                    res['after_0_0p2'].append(rec)
                elif 0.2 < dt <= 0.5:
                    res['after_0p2_0p5'].append(rec)
                elif -0.3 <= dt <= 0:
                    res['before_0p3'].append(rec)
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_real_flip.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('logs', res['n_logs'], 'seg', res['n_seg'], 'human_end', res['n_seg_human_end'], 'errors', len(res['errors']))
    for k in ('after_0_0p2', 'after_0p2_0p5', 'before_0p3'):
        print('==', k, len(res[k]))
        for r in res[k]:
            print('  ', r)


if __name__ == '__main__':
    main()
