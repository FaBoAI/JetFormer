"""r3-r3-safety-1 の確かめ(直さない)。refute3_safety/cl3.py の閉じたループ(本物の drive._loop・偽物の車と ESC)で、
前の壁 1.8 m/s 前後の当たり → カメラ L 秒来ない を、壁の位置 8 通り(位相)× L 3 通り × keep 0/1、と中立止めなしで回す。
一時フォルダは fix3/tmp_verify_r3r3safety1(終わったら消す)。"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute3_safety'))
import cl3                                    # noqa: E402
import closed_loop as CL                      # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402

TMP = HERE.parent / 'tmp_verify_r3r3safety1'
cl3.TMP = TMP
CL.TMP = TMP

res = []
for keep in (1, 0):
    for L in (None, 0.6, 1.0, 2.0):
        for j in range(8):
            xf = 2.00 + 0.01 * j
            sc_ = {'dur': 8.0, 'xf': xf, 'model': (C, N - 120.0), 'drops': [], 'hit_drop': (0.0, L) if L else None}
            p = dict(PARAMS, recovery=1, start_kick=1, rec_back_guard=1, rec_hold_keep=keep)
            sim, rec, pca, ph = cl3.run(p, sc_)
            t_hit, _, v_hit, g_hit = sim.car.hits[0]
            back = t_hit + (L or 0.0)
            proc = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold']]
            last_before = max(t for t in proc if t <= t_hit + 0.001)
            after = [t for t in proc if t > t_hit + 0.001]
            first_after = min(after)
            rev = [(t - T0) / 1e9 for t, r in rec.rows
                   if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40]
            gu = any(int(r['assist']) == A.ASSIST_GIVEUP for t, r in rec.rows if not r['hold'])
            xs = [r[1] for r in sim.log if r[0] >= back]
            push = sum(1 for r in sim.log if r[0] >= back and r[4] == 'front' and r[3] < N - 20) * 0.01
            res.append({'keep': keep, 'L': L, 'xf': round(xf, 2), 'v_hit': v_hit, 'g_hit': g_hit,
                        'last_before_ms': round((t_hit - last_before) * 1000, 1),
                        'frame_before_drop': (first_after - t_hit) < (L or 9) and L is not None and (first_after - t_hit) < 0.1,
                        'result': 'giveup' if gu else ('reverse %.2f s after camera' % (rev[0] - back) if rev else 'none'),
                        'backed_m': round(xf - min(xs), 3), 'push_wall_s': round(push, 2),
                        'ev': [(e[1]['what'], e[1].get('hold_s')) for e in rec.events if e[0] == 'recovery_guard']})
for r in res:
    print(r)
(HERE / 'out' / 'closed_phase.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
shutil.rmtree(TMP, ignore_errors=True)
