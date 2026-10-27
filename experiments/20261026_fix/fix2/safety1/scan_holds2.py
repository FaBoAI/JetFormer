"""AI 側の中立止め(hold)の起き方と、リカバリー中止(assist=3)が人の交代まで続いた長さ(読むだけ)"""
import glob, json, os, sys
import numpy as np
root = '/home/jetson/jetformer/data/raw'
iso, giveups, runs_per_scene = [], [], []
for sc in sorted(glob.glob(root + '/20261*/*/AI_CMD.npz')):
    if os.path.getsize(sc) == 0:
        continue
    try:
        z = np.load(sc); hold = z['hold'].astype(bool); sel = z['select'].astype(bool); a = z['assist'].astype(int)
        t = z['timestamps_us'] / 1e6
    except Exception:
        continue
    ai = sel & ~z['dry_run'].astype(bool)
    # 孤立した hold(前後が AI のフレーム)の数・長さ
    i, n = 0, len(hold)
    while i < n:
        if ai[i] and hold[i]:
            j = i
            while j < n and ai[j] and hold[j]:
                j += 1
            iso.append({'scene': os.path.relpath(sc, root), 'row': i, 'n': j - i,
                        'before_ai': bool(i > 0 and ai[i - 1] and not hold[i - 1]),
                        'after_ai': bool(j < n and ai[j] and not hold[j])})
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if a[i] == 3:
            j = i
            while j < n and (a[j] == 3 or (hold[j] and sel[j])):
                j += 1
            giveups.append({'scene': os.path.relpath(sc, root), 'row': i, 'dur_s': round(float(t[j - 1] - t[i]), 2),
                            'ended_by': 'human' if j < n and not sel[j] else ('end' if j >= n else 'other a=%d' % a[j])})
            i = j
        else:
            i += 1
mid = [x for x in iso if x['before_ai'] and x['after_ai']]
print(json.dumps({'hold_runs': len(iso), 'hold_runs_mid_ai(前後がAI)': len(mid),
                  'mid_lengths': sorted(collections.Counter(x['n'] for x in mid).items()) if False else
                  {str(k): sum(1 for x in mid if x['n'] == k) for k in sorted(set(x['n'] for x in mid))},
                  'mid_examples': mid[:8], 'giveups': giveups}, ensure_ascii=False, indent=1))
