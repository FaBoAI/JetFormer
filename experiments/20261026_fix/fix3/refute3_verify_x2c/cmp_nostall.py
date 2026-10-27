"""反証の確かめ(r3-r3-safety-1): 同じ前の壁の当たりで、カメラが止まらないときのリカバリーの下がる量と、
keep=1 で 'none'(段なし)の形、keep=0 の形を比べる。装置・GPU・API に触れない。直さない。"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix3/refute3_safety')
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402

cl3.TMP = Path('/home/jetson/jetformer/experiments/20261026_fix/fix3/tmp_verify_x2c')
cl3.CL.TMP = cl3.TMP
res = []
for keep, guard in ((1, 1), (0, 1), (0, 0)):
    for L in (0.0, 0.6, 2.0):
        for xf in (2.00, 2.03, 2.06):
            sc_ = {'dur': 8.0, 'xf': xf, 'model': (C, N - 120.0), 'drops': []}
            if L:
                sc_['hit_drop'] = (0.0, L)
            p = dict(PARAMS, recovery=1, start_kick=1, rec_back_guard=guard, rec_hold_keep=keep)
            sim, rec, pca, ph = cl3.run(p, sc_)
            t_hit = sim.car.hits[0][0]
            back = t_hit + L
            rows = [((t - T0) / 1e9, r) for t, r in rec.rows if not r['hold']]
            rev = [t for t, r in rows if int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40]
            gu = any(int(r['assist']) == A.ASSIST_GIVEUP for t, r in rows)
            xs = [r[1] for r in sim.log if r[0] >= t_hit]
            push = sum(1 for t, r in rows if t > back and r['out'][1] < N - 20 and int(r['assist']) == A.ASSIST_NONE)
            res.append({'keep': keep, 'guard': guard, 'L': L, 'xf': xf, 't_hit': t_hit,
                        'giveup': gu, 'first_rev_after_hit_s': round(rev[0] - t_hit, 2) if rev else None,
                        'backed_m': round(xf - min(xs), 3), 'model_push_frames_after_cam': push,
                        'hits': sim.car.hits})
for r in res:
    print(r)
shutil.rmtree(cl3.TMP, ignore_errors=True)
