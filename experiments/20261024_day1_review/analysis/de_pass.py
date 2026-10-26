#!/usr/bin/env python3
"""10/23 の AI の DE 区間の通過を 1 回 1 行に(読むだけ)。DE = loc_code 3 が続く区間で、ずっと AI が操作していたもの。
入る速さ = 区間の最初のフレームの eRPM(負 = 前進。絶対値で出す)、区間の中の最大の速さ・ステア・スロットル(出した値 out)、
衝突 = 区間の中か、区間の終わりから 0.3 秒以内の AI の衝突(collision_g の最大)。
    python3 de_pass.py   → de_pass.json と、速さの帯ごとの衝突の割合"""
import glob, json, os, re
import numpy as np
LAB = '/home/jetson/jetformer/data/labels'
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {r['log']: r for r in json.load(open(os.path.join(HERE, 'runs.json'), encoding='utf-8'))}
out = []
for log, r in sorted(RUNS.items()):
    z = np.load('%s/%s.npz' % (LAB, log))
    de = (z['loc_code'] == 3)
    idx = np.flatnonzero(np.diff(np.r_[0, de.astype(int), 0]))
    t = z['t_s']
    for a, b in zip(idx[::2], idx[1::2]):
        if not z['ai'][a:b].all() or b - a < 5:
            continue
        b2 = min(len(t), b + 9)                       # 0.3 秒の余り
        cg = z['collision_g'][a:b2]
        hit = bool((z['collision'][a:b2] & z['ai'][a:b2]).any())
        e = -z['erpm'][a:b]
        out.append(dict(log=log, ai_no=r['ai_no'], group=r['group'], model=r['model'], steer=r['steer'], thr=r['thr'],
                        tof=r['tof_guard'], t_s=round(float(t[a]), 2), dur_s=round(float(t[b - 1] - t[a]), 2),
                        entry_erpm=float(e[0]), max_erpm=float(e.max()), mean_erpm=float(e.mean()),
                        steer_out_min=float(z['out'][a:b, 0].min()), steer_out_max=float(z['out'][a:b, 0].max()),
                        thr_out_min=float(z['out'][a:b, 1].min()), guard_frames=int((z['guard'][a:b] != 0).sum()),
                        hit=hit, g=float(cg.max()) if hit else 0.0))
json.dump(out, open(os.path.join(HERE, 'de_pass.json'), 'w'), indent=1)
for grp in (('OFFICIAL',), ('OFFICIAL', 'EXP', 'TRIAL')):
    ps = [p for p in out if p['group'] in grp and p['ai_no'] >= 3 and not (p['group'] == 'OFFICIAL' and p['ai_no'] < 8)]
    print('==', grp, '通過', len(ps), '衝突', sum(p['hit'] for p in ps))
    for lo, hi in ((0, 5000), (5000, 5500), (5500, 6000), (6000, 6500), (6500, 7000), (7000, 99999)):
        q = [p for p in ps if lo <= p['entry_erpm'] < hi]
        if q:
            print('  入り %5d-%5d: %2d 通過 衝突 %2d (%.0f%%)  最大 eRPM 中央 %d' % (lo, hi, len(q), sum(p['hit'] for p in q),
                  100 * sum(p['hit'] for p in q) / len(q), np.median([p['max_erpm'] for p in q])))
    for lo, hi in ((0, 6000), (6000, 6500), (6500, 7000), (7000, 99999)):
        q = [p for p in ps if lo <= p['max_erpm'] < hi]
        if q:
            print('  最大 %5d-%5d: %2d 通過 衝突 %2d (%.0f%%)' % (lo, hi, len(q), sum(p['hit'] for p in q), 100 * sum(p['hit'] for p in q) / len(q)))
ps = [p for p in out if p['group'] == 'OFFICIAL' and p['ai_no'] >= 8]
print('公式 衝突あり', [(p['ai_no'], p['model'], int(p['entry_erpm']), int(p['max_erpm']), round(p['g'], 1), int(p['steer_out_min']), int(p['steer_out_max']), p['guard_frames']) for p in ps if p['hit']])
print('公式 衝突なし 入り中央', np.median([p['entry_erpm'] for p in ps if not p['hit']]), '最大中央', np.median([p['max_erpm'] for p in ps if not p['hit']]))
print('公式 衝突あり 入り中央', np.median([p['entry_erpm'] for p in ps if p['hit']]), '最大中央', np.median([p['max_erpm'] for p in ps if p['hit']]))
