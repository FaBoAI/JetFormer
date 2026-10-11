"""AI走行中の人の交代(介入)を分類する: 直前に衝突/停止があったか(=失敗の信号)、運用上の交代か。
RLIF / HG-DAgger / IWR / RECAP で「介入=負の報酬」「介入区間=正例」として使えるかの見積もり。"""
import os, sys, glob, json
import numpy as np
RAW = '/home/jetson/jetformer/data/raw'
G = 9.80665
logs = [l for l in sorted(os.listdir(RAW)) if 'ai-drive' in json.load(open(os.path.join(RAW, l, 'log.json'))).get('tags', [])]
allrows = []
for log in logs:
    desc = json.load(open(os.path.join(RAW, log, 'log.json'))).get('description', '')
    VT, SEL, ER, IT, HG, AT, AS = [], [], [], [], [], [], []
    for sc in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        try:
            V = np.load(os.path.join(sc, 'VEHICLE.npz')); I = np.load(os.path.join(sc, 'IMU.npz'))
        except Exception:
            continue
        VT.append(V['timestamps_us'] / 1e6); SEL.append(V['select'].astype(bool)); ER.append(V['erpm'].astype(float))
        a = I['accel']; h = np.hypot(a[:, 0], a[:, 1]); IT.append(I['timestamps_us'] / 1e6); HG.append((h - np.median(h)) / G)
        p = os.path.join(sc, 'AI_CMD.npz')
        if os.path.exists(p):
            A = np.load(p)
            if 'assist' in A.files:
                AT.append(A['timestamps_us'] / 1e6); AS.append(A['assist'])
    vt = np.concatenate(VT); sel = np.concatenate(SEL); er = np.concatenate(ER); it = np.concatenate(IT); hg = np.concatenate(HG)
    at = np.concatenate(AT) if AT else np.array([]); asg = np.concatenate(AS) if AS else np.array([])
    imp = []; last = -1e9
    for t, g in zip(it, hg):
        if g >= 2.5 and t - last > 0.6:
            imp.append(t); last = t
    imp = np.array(imp)
    to = np.where(sel[:-1] & ~sel[1:])[0]
    for k in to:
        t = vt[k + 1]
        if k > 0 and vt[k + 1] - vt[k] > 0.5:   # 途切れ(シーン境界)
            continue
        prev_imp = imp[(imp <= t) & (imp > t - 3.0)]
        w = (vt > t - 1.0) & (vt <= t)
        stopped = bool(np.all(np.abs(er[w]) < 300)) if w.any() else False
        w2 = (vt > t - 0.3) & (vt <= t)
        fast = float(np.max(np.abs(er[w2]))) if w2.any() else 0.0
        rec_active = bool(len(at) and np.any(asg[(at > t - 3) & (at <= t)] >= 2))
        # 交代後の人の操作が何秒続いたか(次にAIへ戻るまで)
        nxt = np.where((vt > t) & sel)[0]
        hum_s = float(vt[nxt[0]] - t) if len(nxt) else float(vt[-1] - t)
        if len(prev_imp):
            cls = 'impact'
        elif stopped:
            cls = 'stuck'
        elif fast > 3000:
            cls = 'moving_fast'   # 走行中の先回りの交代(危険を見て)
        else:
            cls = 'slow/other'
        allrows.append(dict(log=log, desc=desc, t=round(float(t - vt[0]), 2), cls=cls, dt_imp=round(float(t - prev_imp[-1]), 2) if len(prev_imp) else None, rec=rec_active, human_s=round(hum_s, 1), erpm_last=int(fast)))
from collections import Counter
c = Counter(r['cls'] for r in allrows)
print('takeovers', len(allrows), dict(c))
d = [r['dt_imp'] for r in allrows if r['dt_imp'] is not None]
print('impact->takeover s: median %.2f IQR %.2f-%.2f' % (np.median(d), np.percentile(d, 25), np.percentile(d, 75)))
print('human segment after takeover s: median %.1f, sum %.0f' % (np.median([r['human_s'] for r in allrows]), sum(min(r['human_s'], 30) for r in allrows)))
print('with recovery active in prior 3s:', sum(r['rec'] for r in allrows))
for log in sorted(set(r['log'] for r in allrows)):
    rr = [r for r in allrows if r['log'] == log]
    print(log, rr[0]['desc'], len(rr), dict(Counter(r['cls'] for r in rr)))
for r in allrows:
    if r['log'] >= '20261010-203924':
        print(r)
json.dump(allrows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interventions.json'), 'w'), ensure_ascii=False, indent=1)
