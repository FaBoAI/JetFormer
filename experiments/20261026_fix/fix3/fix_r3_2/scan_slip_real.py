"""本物の 10/23・10/24 の記録で、③ standing_3lap の発進ごとの空転の比(launch_slip)が出ない発進を数える(読むだけ)。
2026-10-27 fix3 直す 3 回目・2 組目 r3-r3-rec-4。比が出ない発進に印を付けると、本物の出力が変わるかを見る"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/tools')
sys.path.insert(0, '/home/jetson/jetformer')
import standing_3lap as s3  # noqa: E402

out = {'n_start': 0, 'with_a1': 0, 'slip_none': [], 'ratio_none': [], 'ratios': [], 'dv_imu_min': None}
for day in ('20261023', '20261024'):
    logs, _ = s3.day_logs(day)
    for lg in logs:
        res = s3.analyze_log(lg)
        for s in res.get('standing') or []:
            out['n_start'] += 1
            if not s.get('anchors'):
                continue
            out['with_a1'] += 1
            sl = s.get('launch_slip')
            if sl is None:
                out['slip_none'].append([lg, s['t_ai']])
            elif sl.get('ratio') is None:
                out['ratio_none'].append([lg, s['t_ai'], sl])
            else:
                out['ratios'].append(sl['ratio'])
                di = sl['dv_imu']
                out['dv_imu_min'] = di if out['dv_imu_min'] is None else min(out['dv_imu_min'], di)
out['ratio_max'] = max(out['ratios']) if out['ratios'] else None
out['n_ratio'] = len(out.pop('ratios'))
print(json.dumps(out, ensure_ascii=False))
