"""オフラインRL用ラベルの在庫: カメラフレーム(30fps)単位で、
・完了した周の中にあるか(=「周の残り時間」= モンテカルロ回帰の目標が作れる)
・衝突の1秒前(失敗の直前=負例の核)、交代の1.5秒前
・人/AI の別
を数える。さらに衝突率の差を実車で見分けるのに必要な周回数(ポアソン、条件付き二項検定)を計算。"""
import os, sys, glob, json
import numpy as np
from math import comb
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.laps import LapCounter
RAW = '/home/jetson/jetformer/data/raw'; G = 9.80665
tot = {}
def add(k, v):
    tot[k] = tot.get(k, 0) + v
for log in sorted(os.listdir(RAW)):
    ai_log = 'ai-drive' in json.load(open(os.path.join(RAW, log, 'log.json'))).get('tags', [])
    CT, VT, SEL, ER, IT, GZ, HG = [], [], [], [], [], [], []
    for sc in sorted(glob.glob(os.path.join(RAW, log, log + '-*'))):
        try:
            V = np.load(os.path.join(sc, 'VEHICLE.npz')); I = np.load(os.path.join(sc, 'IMU.npz')); C = np.load(os.path.join(sc, 'CAM_FRONT.npz'))
        except Exception:
            continue
        CT.append(C['timestamps_us'] / 1e6); VT.append(V['timestamps_us'] / 1e6); SEL.append(V['select'].astype(bool)); ER.append(V['erpm'].astype(float))
        a = I['accel']; h = np.hypot(a[:, 0], a[:, 1]); IT.append(I['timestamps_us'] / 1e6); GZ.append(I['gyro'][:, 2].astype(float)); HG.append((h - np.median(h)) / G)
    ct = np.concatenate(CT); vt = np.concatenate(VT); sel = np.concatenate(SEL); er = np.concatenate(ER); it = np.concatenate(IT); gz = np.concatenate(GZ); hg = np.concatenate(HG)
    ie = np.interp(it, vt, er); isel = sel[np.searchsorted(vt, it).clip(0, len(vt) - 1)]
    lc = LapCounter(); laps = []
    for k in range(len(it)):
        lap = lc.add(it[k], gz[k], ie[k], ai=bool(isel[k]), impact_g=float(hg[k]))
        if lap: laps.append((lap['t_end'] - lap['time_s'], lap['t_end'], lap['impacts']))
    imp = []; last = -1e9
    for t, g in zip(it, hg):
        if g >= 2.5 and t - last > 0.6:
            imp.append(t); last = t
    imp = np.array(imp)
    to = vt[1:][sel[:-1] & ~sel[1:]]
    csel = sel[np.searchsorted(vt, ct).clip(0, len(vt) - 1)]
    cer = np.interp(ct, vt, er)
    moving = np.abs(cer) > 200
    inlap = np.zeros(len(ct), bool); cleanlap = np.zeros(len(ct), bool)
    for t0, t1, ni in laps:
        m = (ct >= t0) & (ct < t1); inlap |= m
        if ni == 0: cleanlap |= m
    pre_imp = np.zeros(len(ct), bool)
    for t in imp: pre_imp |= (ct > t - 1.0) & (ct <= t)
    pre_to = np.zeros(len(ct), bool)
    for t in to: pre_to |= (ct > t - 1.5) & (ct <= t)
    who = 'ai' if ai_log else 'human_log'
    for name, m in [('frames', np.ones(len(ct), bool)), ('moving', moving), ('in_lap', inlap & moving), ('clean_lap', cleanlap & moving), ('pre_impact_1s', pre_imp), ('pre_takeover_1.5s', pre_to)]:
        if ai_log:
            add('AIlog/AIsel/' + name, int((m & csel).sum())); add('AIlog/human/' + name, int((m & ~csel).sum()))
        else:
            add('humanlog/' + name, int(m.sum()))
for k in sorted(tot): print('%-32s %7d  (%.1f min)' % (k, tot[k], tot[k] / 30 / 60))

# 実車評価の検出力: 衝突/周 λA vs λB を n周ずつで比べる(片側 α=0.05、条件付き二項)
def power(l1, l2, n, alpha=0.05):
    lam = (l1 + l2) * n; p_alt = l1 / (l1 + l2)
    # Nの分布(ポアソン)で平均
    from math import exp, lgamma, log
    pw = 0.0
    Nmax = int(lam + 8 * lam ** 0.5) + 5
    for N in range(1, Nmax):
        pN = exp(N * log(lam) - lam - lgamma(N + 1))
        # 臨界値: P(X>=c | p=0.5) <= alpha
        cdf = 0.0; c = N + 1
        tail = 0.0
        for x in range(N, -1, -1):
            tail += comb(N, x) * 0.5 ** N
            if tail > alpha: c = x + 1; break
        pa = sum(comb(N, x) * p_alt ** x * (1 - p_alt) ** (N - x) for x in range(c, N + 1))
        pw += pN * pa
    return pw
for l1, l2 in [(1.2, 0.75), (1.2, 0.6), (1.2, 0.4)]:
    out = []
    for n in (12, 24, 43, 60, 100):
        out.append('%d周:%.2f' % (n, power(l1, l2, n)))
    print('衝突/周 %.2f→%.2f (比 %.1f): ' % (l1, l2, l1 / l2) + ' '.join(out))
