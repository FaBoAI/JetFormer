"""反証 3 回目・係 record: car_check の R1 の板(r1_bumps)は AI の通過だけを見る。人が R1 の直前に人側にした通過の板の衝撃は
「R1 の板 7G 以上 1 回」に入るか。本物の 10/23・10/24 の記録を読むだけ(書くのは out/ の JSON だけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python \
        experiments/20261026_fix/fix3/refute3_record/probe_r1_flip.py
"""
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
import label_events as le  # noqa: E402
import section_scorecard as ss  # noqa: E402
from jetformer import config  # noqa: E402

OUT = Path(__file__).resolve().parent / 'out'


def main():
    raw = config.RAW_DIR
    labels_dir = config.DATA_DIR / 'labels'
    rows = []
    for d in sorted(raw.glob('2026102[34]-*')):
        lg = d.name
        try:
            meta = json.loads((d / 'log.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not (meta.get('drive') or {}) or (meta.get('drive') or {}).get('dry_run'):
            continue
        labels = ss._labels(lg, labels_dir, raw)
        info, lab, _ = labels
        if lab is None:
            continue
        L = le.load_log(lg, raw)
        ex = {}
        sc = ss.score_log(lg, labels_dir, raw, labels=labels, loaded=L, extras=ex)
        if sc.get('error'):
            continue
        I = L['IMU']
        t0 = info['t0_us']
        it = (np.asarray(I['timestamps_us'], np.int64) - t0) / 1e6
        t = np.asarray(lab['t_s'], np.float64)
        ai = np.asarray(lab['ai'], bool) & ~np.asarray(lab['dry_run'], bool)
        ta_ai = t[ai]
        a3 = np.abs(np.linalg.norm(np.asarray(I['accel'], np.float64), axis=1) / cc.G - 1.0)
        res = cc.r1_bumps(sc.get('passes') or [], it, I['accel'])
        for ps in sc.get('passes') or []:
            if ps.get('ai') or not len(ta_ai):
                continue
            ta = ps['ta_s']
            j = int(np.searchsorted(ta_ai, ta, side='right')) - 1
            if j < 0:
                continue
            dt = ta - float(ta_ai[j])
            if not (0.0 < dt <= 1.0):
                continue
            m = (it >= ta + cc.BUMP_W[0]) & (it <= ta + cc.BUMP_W[1])
            if m.sum() < cc.BUMP_MIN_SAMPLES:
                continue
            rows.append({'log': lg, 'ta_s': ta, 'flip_before_s': round(dt, 3), 'a3_max_human_pass': round(float(a3[m].max()), 2),
                         'car_check_ai_passes': res['passes'], 'car_check_a3_max': res['a3_max']})
    OUT.mkdir(exist_ok=True)
    (OUT / 'probe_r1_flip.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    for r in rows:
        print(r)
    print('人側にしてから 1 秒以内の R1 の通過', len(rows), '・うち板 7G 以上', sum(1 for r in rows if r['a3_max_human_pass'] >= 7.0),
          '・4.5G 以上', sum(1 for r in rows if r['a3_max_human_pass'] >= 4.5))


if __name__ == '__main__':
    main()
