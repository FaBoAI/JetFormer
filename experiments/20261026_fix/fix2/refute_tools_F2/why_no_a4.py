#!/usr/bin/env python3
"""指摘 tools:F2 の確かめ(読むだけ): A3+dist になった発進で、A4 が無かったわけ(AI の区間の終わり・次の起点・記録の終わり)"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as S  # noqa: E402
import section_scorecard as ss  # noqa: E402
import label_events as le  # noqa: E402

names = S.candidate_names()
for day in sys.argv[1:] or ('20261023', '20261024'):
    logs, _ = S.day_logs(day)
    for lg in logs:
        try:
            r = S.analyze_log(lg, names=names)
        except Exception as e:  # noqa
            continue
        for s in r.get('standing') or []:
            if s.get('method') != 'A3+dist':
                continue
            # 全部の起点と AI の区間の終わり・記録の終わり
            L = le.load_log(lg)
            labels = ss._labels(lg, None, None) if False else None
            ex = {}
            sc = ss.score_log(lg, ROOT / 'data' / 'labels', ROOT / 'data' / 'raw', extras=ex)
            an = np.asarray(ex['anchors'], float)
            vt = np.asarray(ex['vt'], float)
            A = s['anchors']
            after = an[an > A[-1]]
            print(lg, 'No.', r.get('ai_no'), 'anchors', A, 'end3', s['end_s'], 't_ai_end', s['t_ai_end'],
                  'next_anchor', after[:2].tolist(), 'rec_end_vt', round(float(vt[-1]), 2),
                  'erpm_at_t_ai_end', round(float(np.interp(s['t_ai_end'], vt, ex['verpm'])), 0),
                  'erpm_+1s', round(float(np.interp(s['t_ai_end'] + 1.0, vt, ex['verpm'])), 0),
                  'from_move', s['from_move_s'], 'd_pre', s['d_pre_m'], 'Lrun', s['lap_dist_median_m'])
