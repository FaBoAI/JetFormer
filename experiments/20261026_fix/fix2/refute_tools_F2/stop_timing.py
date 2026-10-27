#!/usr/bin/env python3
"""指摘 tools:F2 の確かめ(読むだけ): 3 周を走った発進ごとに、3 周目の線 → 人側(AI の区間の終わり)の秒と A4 の有無"""
import sys
from pathlib import Path
ROOT = Path('/home/jetson/jetformer')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import standing_3lap as S  # noqa: E402
names = S.candidate_names()
print('| 日 | No. | 記録 | method | valid | 動いてから 3 周 | 線 → 人側 [秒] | A4 − 線 [秒] | 線のあとの AI の秒 | 衝突 |')
for day in sys.argv[1:] or ('20261023', '20261024'):
    for lg in S.day_logs(day)[0]:
        try:
            r = S.analyze_log(lg, names=names)
        except Exception:  # noqa
            continue
        for s in r.get('standing') or []:
            if s.get('from_move_s') is None:
                continue
            A = s['anchors']
            a4 = (A[3] - s['end_s']) if len(A) == 4 else None
            print('| %s | %s | %s | %s | %s | %.2f | %.2f | %s | %s | %s |' % (
                day, r.get('ai_no'), lg, s['method'], s['valid'], s['from_move_s'], s['t_ai_end'] - s['end_s'],
                '—' if a4 is None else '%.2f' % a4, '%.1f' % (s['t_ai_end'] - s['end_s']), (s.get('collisions') or {}).get('n')))
