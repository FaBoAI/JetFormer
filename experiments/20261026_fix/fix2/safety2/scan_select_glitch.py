"""反証 2 回目・safety: CH3(select)の短い人側(AI 側の間に 1〜3 フレームだけ人側)が実際の記録にあるか(読むだけ)。
短い人側は Assist.step(select False) で リカバリー・発進の補助・発進の前進の区間 をすべて戻す。0 バイトの npz は飛ばす"""
import glob, json, os, sys
import numpy as np
root = '/home/jetson/jetformer/data/raw'
out = {'scenes': 0, 'ai_frames': 0, 'short_human': [], 'by_len': {}}
for sc in sorted(glob.glob(root + '/2026*/*/AI_CMD.npz')):
    if os.path.getsize(sc) == 0:
        continue
    try:
        z = np.load(sc)
        sel = z['select'].astype(bool); hold = z['hold'].astype(bool); t = z['timestamps_us'].astype(np.int64)
        dry = z['dry_run'].astype(bool) if 'dry_run' in z.files else np.zeros(len(sel), bool)
        a = z['assist'].astype(int) if 'assist' in z.files else np.zeros(len(sel), int)
    except Exception:
        continue
    out['scenes'] += 1
    out['ai_frames'] += int((sel & ~dry).sum())
    n = len(sel)
    i = 0
    while i < n:
        if not sel[i] and i > 0 and sel[i - 1]:
            j = i
            while j < n and not sel[j]:
                j += 1
            L = j - i
            if j < n and L <= 3:
                dt = (t[j] - t[i - 1]) / 1e6
                out['by_len'][L] = out['by_len'].get(L, 0) + 1
                out['short_human'].append({'scene': os.path.relpath(sc, root), 'row': int(i), 'len': int(L), 'gap_s': round(float(dt), 3),
                                           'assist_before': int(a[i - 1]), 'assist_after': int(a[j]), 'hold_in': bool(hold[i:j].any())})
            i = j
        else:
            i += 1
print(json.dumps({k: (v if k != 'short_human' else v[:40]) for k, v in out.items()} | {'n_short': len(out['short_human'])},
                 ensure_ascii=False, indent=1))
