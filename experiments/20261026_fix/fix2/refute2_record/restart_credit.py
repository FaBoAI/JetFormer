"""反証 2 回目・record: 1 本の中で、ぶつかったあとに人が止めて AI 側へ戻した(止まった状態から 2 回目の発進)とき、
standing_3lap・quick_eval_by_params が 2 回目の発進を「止まってから 3 周」に数え、衝突をその発進からしか数えないので、
ぶつかった本が「止まってから 3 周 …(衝突 0)」に見えることを確かめる。作り物の記録だけ(tmp_record の下)。"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import label_events as le  # noqa: E402
import standing_3lap as s3  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
from test_lap_anchor import REF, T0_US, course_profile  # noqa: E402

G = le.G
V = 2.0
TMP = JF / 'experiments/20261026_fix/fix2/tmp_record/restart'
if TMP.exists():
    shutil.rmtree(TMP)
raw = TMP / 'raw'
log = '20001103-140000'
d = raw / log
d.mkdir(parents=True)
(d / 'log.json').write_text(json.dumps({
    'description': 'ai_no=20 bench=real_bench_20261023 day=2000-11-03 block=20 cand=EXP tune=4-1', 'mode_id': 5,
    'drive': {'run': 'runX', 'steer_center_us': REF[0], 'throttle_neutral_us': REF[1], 'dry_run': False,
              'params': {'throttle_gain': 0.9}}}), encoding='utf-8')
dur = 47.0
ft = np.arange(0, dur, 1 / 30.0)
vt = np.arange(0, dur, 0.01)
it = vt + 0.003
# 1 回目: 1 秒に AI 側・1.5 秒に動く。8.0 秒に 6G で当たる。10 秒に人が止める(人側・止まる)。
# 2 回目: 14 秒に AI 側(止まった状態)・14.5 秒に動く・そのあと 3 周きれい
seg1 = (1.5, 10.0)
seg2 = 14.5
g1, _, _ = course_profile(dur, 4.0, ())
g2, _, _ = course_profile(dur, 2.5, ())
gz = np.zeros(len(it))
for a, b, g in g1:
    m = (it >= a) & (it < b) & (it >= seg1[0]) & (it < seg1[1])
    gz[m] = g
for a, b, g in g2:
    m = (it >= a + seg2) & (it < b + seg2) & (it >= seg2)
    gz[m] = g
vv = np.where(((vt >= seg1[0]) & (vt < seg1[1])) | (vt >= seg2), V, 0.0)
e = -vv / le.KV
sel_v = ((vt >= 1.0) & (vt < 10.0)) | (vt >= 14.0)
sel_f = ((ft >= 1.0) & (ft < 10.0)) | (ft >= 14.0)
acc = np.zeros((len(it), 3))
acc[:, 2] = G
k = int(np.searchsorted(it, 8.0))
acc[k:k + 3, 1] += [-3.0 * G, -6.0 * G, -1.8 * G]            # 6G の衝突(1 回目の発進の 2 周目の手前)
v_f = np.maximum(np.interp(ft, vt, vv), 0.1)
kap = np.radians(np.interp(ft + 0.1, it, gz)) / v_f
steer = REF[0] + np.clip(kap * 1e4 / 40.0, -480, 480)
out = np.stack([steer, np.full(len(ft), REF[1] - 200.0)], axis=1)
us = lambda x: (T0_US + np.round(x * 1e6)).astype(np.int64)      # noqa: E731
for si in range(3):
    a, b = si * 20.0, min((si + 1) * 20.0, dur)
    s = d / ('%s-%04d' % (log, si + 1))
    s.mkdir()
    fm, vm, im = (ft >= a) & (ft < b), (vt >= a) & (vt < b), (it >= a) & (it < b)
    n = int(fm.sum())
    np.savez(s / 'AI_CMD.npz', timestamps_us=us(ft[fm]), raw=out[fm].astype(np.float32), out=out[fm].astype(np.float32),
             gains=np.tile([1.0, 0.9], (n, 1)).astype(np.float32), trim=np.zeros(n, np.float32),
             lookahead=np.full(n, 3, np.int16), hold=np.zeros(n, bool), select=sel_f[fm], dry_run=np.zeros(n, bool),
             assist=np.zeros(n, np.int8), model_out=out[fm].astype(np.float32), launch_on=np.zeros(n, bool),
             launch_end=np.zeros(n, np.int8))
    nv = int(vm.sum())
    np.savez(s / 'VEHICLE.npz', timestamps_us=us(vt[vm]), pwm=np.tile([1441, 1496, 2000], (nv, 1)).astype(np.int32),
             valid=np.ones((nv, 3), bool), select=sel_v[vm], erpm=np.round(e[vm]).astype(np.int32),
             board_t_us=(us(vt[vm]) - T0_US + 10_000_000))
    gy = np.zeros((int(im.sum()), 3))
    gy[:, 2] = gz[im]
    np.savez(s / 'IMU.npz', timestamps_us=us(it[im]), accel=acc[im].astype(np.float32), gyro=gy.astype(np.float32))
    np.savez(s / 'CAM_FRONT.npz', timestamps_us=us(ft[fm]), files=np.array(['x.jpg'] * n))

res = s3.analyze_log(log, raw, TMP / 'labels', {'runX': 'B'})
print('この本の衝突(全部):', res['collisions'])
for r in res['standing']:
    print('発進 t_ai=%s: valid=%s from_move=%s collisions=%s why=%s' % (r['t_ai'], r['valid'], r['from_move_s'],
                                                                     (r.get('collisions') or {}).get('n'), r['why']))
b = s3.best_standing(res)
print('best_standing(表の「止まってから 3 周(衝突)」):', b['from_move_s'] if b else None, '衝突', (b.get('collisions') or {}).get('n') if b else None)
rows = qp.run_rows([res])
print('quick_eval_by_params 1 章の欄: 衝突(強)=%s(%s)・止まってから 3 周=%s(%s)' % (
    rows[0]['collisions'], rows[0]['strong'], rows[0]['standing_best_s'], rows[0]['standing_coll']))
g = qp.group_rows([res])[0]
print('2 章: 止まってから 3 周の本 %d・衝突なし %d' % (g['standing_n'], g['standing_clean_n']))
