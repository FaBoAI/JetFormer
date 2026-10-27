"""読むだけ: 本物の記録の AI 側(select・dry でない)で、中立止め(hold の行が続いた区間)の長さと、2 つの中立止めが近く(前の終わりから
次の始めまで 0.4 秒以内)に続く組を数える。    python -I scan_hold_pairs.py /home/jetson/jetformer/data/raw '2026102[34]'
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
pat = sys.argv[2]
out = {'recs': 0, 'ai_s': 0.0, 'holds': 0, 'hold_len_hist': collections.Counter(), 'pairs_0.4': [], 'holds_in_rec': 0}
for ai in sorted(root.glob(pat + '-*/*/AI_CMD.npz')):
    try:
        a = np.load(ai)
    except Exception:
        continue
    out['recs'] += 1
    ta = a['timestamps_us'].astype(np.int64)
    hold = a['hold'].astype(bool)
    sel = a['select'].astype(bool) & ~a['dry_run'].astype(bool)
    ai_rows = sel
    if ai_rows.sum() > 1:
        idx = np.where(ai_rows)[0]
        dt = np.diff(ta[idx]) / 1e6
        out['ai_s'] += float(dt[dt < 1.0].sum())
    runs = []
    i, n = 0, len(ta)
    while i < n:
        if hold[i] and sel[i]:
            j = i
            while j + 1 < n and hold[j + 1] and sel[j + 1]:
                j += 1
            # 中立を書いていた長さ: 最初の hold の行から、次の hold でない行まで
            t1 = ta[j + 1] if j + 1 < n else ta[j]
            runs.append((ta[i], t1))
            i = j + 1
        else:
            i += 1
    out['holds'] += len(runs)
    for t0, t1 in runs:
        L = (t1 - t0) / 1e6
        out['hold_len_hist'][('<0.05' if L < 0.05 else '0.05-0.08' if L < 0.08 else '0.08-0.13' if L < 0.13 else
                              '0.13-0.3' if L < 0.3 else '0.3-0.5' if L < 0.5 else '>=0.5')] += 1
    for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
        if (b0 - a1) / 1e6 <= 0.4:
            out['pairs_0.4'].append((ai.parent.name, round((a0 - ta[0]) / 1e6, 2), round((a1 - a0) / 1e6, 3),
                                     round((b0 - a1) / 1e6, 3), round((b1 - b0) / 1e6, 3)))
out['hold_len_hist'] = dict(out['hold_len_hist'])
out['n_pairs_0.4'] = len(out['pairs_0.4'])
out['pairs_0.4'] = out['pairs_0.4'][:30]
out['ai_s'] = round(out['ai_s'], 1)
print(json.dumps(out, ensure_ascii=False, indent=1))
