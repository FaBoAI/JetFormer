#!/usr/bin/env python3
"""参考(原因 1 の係へ): v2 の学習に入った AI のフレームの正解が、ゲインのあと(out)でどれだけ縮んだか(読むだけ・numpy)。
drop_v2h / drop_bfast で残した・source 1・future_valid・vehicle_valid・衝突の前後でない aligned のフレームの、
行 0 の正解(= out)と AI_CMD の raw(ゲインの前)の前進の量を、記録ごとのゲインで分けて数える。
    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I <この台本>
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
DS = REPO / 'data/datasets/dataset_v2_20261025'
TABLES = REPO / 'data/derived/dataset_v2_20261025'
OUT = Path('/home/jetson/jetformer_reports/fix_20261026/diag_bfast_v2_ai_labels.json')


def main():
    res = {}
    for tab in ('drop_v2h', 'drop_bfast'):
        d = np.load(TABLES / (tab + '.npz'), allow_pickle=False)
        keep = ~d['drop']
        by_gain = defaultdict(lambda: [0, 0.0, 0.0])      # n, raw の前進, 正解の前進
        tot_frames = 0
        for sc in np.unique(d['scene']):
            m = d['scene'] == sc
            fr = d['frame'][m][keep[m]]
            if not len(fr):
                continue
            z = np.load(DS / 'aligned' / (sc + '.npz'), allow_pickle=False)
            sel = np.zeros(len(z['timestamps_us']), bool)
            sel[fr] = True
            ok = sel & z['future_valid'] & z['vehicle_valid'] & ~z['near_collision']
            tot_frames += int(ok.sum())
            ai = ok & (z['source'] == 1)
            if not ai.any():
                continue
            lg = sc.rsplit('-', 1)[0]
            p = REPO / 'data/raw' / lg / sc / 'AI_CMD.npz'
            a = np.load(p, allow_pickle=False)
            ta, tz = a['timestamps_us'].astype(np.int64), z['timestamps_us'].astype(np.int64)
            ii = np.clip(np.searchsorted(ta, tz), 0, len(ta) - 1)
            has = ta[ii] == tz
            idx = np.nonzero(ai & has)[0]
            info = json.loads((REPO / 'data/labels' / (lg + '.json')).read_text(encoding='utf-8'))
            neutral = float((info.get('ref_us') or {}).get('throttle_neutral', 1500.0))
            g = a['gains'][ii[idx], 1]
            raw = a['raw'][ii[idx], 1]
            lab = z['future_actions'][idx, 0, 1]
            for gv in np.unique(np.round(g, 2)):
                mm = np.round(g, 2) == gv
                e = by_gain[float(gv)]
                e[0] += int(mm.sum())
                e[1] += float(np.sum(neutral - raw[mm]))
                e[2] += float(np.sum(neutral - lab[mm]))
        n_ai = sum(v[0] for v in by_gain.values())
        rows = {'%.2f' % k: {'frames': v[0], 'mean_fwd_raw_us': round(v[1] / v[0], 2), 'mean_fwd_label_us': round(v[2] / v[0], 2),
                         'mean_removed_us': round((v[1] - v[2]) / v[0], 2)} for k, v in sorted(by_gain.items())}
        res[tab] = {'frames_all_kept': tot_frames, 'ai_frames': n_ai, 'ai_share': round(n_ai / max(tot_frames, 1), 4),
                    'by_gain': rows,
                    'ai_mean_removed_us': round(sum(v[1] - v[2] for v in by_gain.values()) / max(n_ai, 1), 2),
                    'ai_mean_fwd_raw_us': round(sum(v[1] for v in by_gain.values()) / max(n_ai, 1), 2)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
