"""反証 2 回目・safety: VEHICLE のサンプルで、select = 0 かつ CH1〜CH3 の valid が全部 False(空のサンプル)が、
select = 1 の間に挟まる回数(日ごと)。走行ループはこれを「CH3 を人側」と読む。読むだけ。0 バイトは飛ばす"""
import collections, glob, json, os
import numpy as np
root = '/home/jetson/jetformer/data/raw'
by_day = collections.defaultdict(lambda: {'scenes': 0, 'samples': 0, 'sel1': 0, 'blank_in_sel1': 0, 'blank_any': 0, 'ex': []})
for vp in sorted(glob.glob(root + '/2026*/*/VEHICLE.npz')):
    if os.path.getsize(vp) == 0:
        continue
    try:
        v = np.load(vp)
        sel = v['select'].astype(bool); val = v['valid'].astype(bool); t = v['timestamps_us'].astype(np.int64)
        e = v['erpm'].astype(int); pwm = v['pwm'].astype(int)
    except Exception:
        continue
    day = os.path.basename(os.path.dirname(os.path.dirname(vp)))[:8]
    b = by_day[day]
    b['scenes'] += 1
    b['samples'] += len(sel)
    b['sel1'] += int(sel.sum())
    blank = (~sel) & (~val.any(axis=1))
    b['blank_any'] += int(blank.sum())
    for i in np.nonzero(blank)[0]:
        if 0 < i < len(sel) - 1:
            # 前後 0.1 秒に select = 1 がある(AI 側の間に挟まった)
            w = (t >= t[i] - 100_000) & (t <= t[i] + 100_000) & sel
            if w.any():
                b['blank_in_sel1'] += 1
                if len(b['ex']) < 3:
                    b['ex'].append({'scene': os.path.relpath(os.path.dirname(vp), root), 'i': int(i), 'erpm': int(e[i]),
                                    'pwm': pwm[i].tolist(), 'erpm_around': e[max(0, i - 2):i + 3].tolist()})
print(json.dumps(by_day, ensure_ascii=False, indent=1))
