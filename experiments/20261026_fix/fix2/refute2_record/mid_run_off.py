"""反証 2 回目・record: 走行中に天井・自動合わせを OFF にした本(画面は「走行中は OFF にだけできる」)が、道具の表で
ON の本として数えられ、印が出ないことを確かめる。作り物の記録だけ(tmp_record の下)。本物の data は読まない・書かない。"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import section_scorecard as ss  # noqa: E402
import quick_eval_by_params as qp  # noqa: E402
import standing_3lap as s3  # noqa: E402
from test_lap_anchor import write_lap_log  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix2/tmp_record/midoff'
if TMP.exists():
    shutil.rmtree(TMP)
raw = TMP / 'raw'
raw.mkdir(parents=True)
log = '20001103-130000'
P = {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 1,
     'auto_target_mps': 4.27, 'launch_boost': 0}
d, anchors, _ = write_lap_log(raw, log, dur=60.0, false_laps=(), description='ai_no=8 bench=real_bench_20261023 '
                              'day=2000-11-03 block=8 cand=EXP tune=2-3', params=P)
OFF_AT = 25.0          # 3 周目の途中で「揺れています」を見て、走行中に天井(と自動合わせ)を OFF にした
for s in sorted(d.glob('*/AI_CMD.npz')):
    z = dict(np.load(s))
    t = (z['timestamps_us'] - z['timestamps_us'][0]) / 1e6 + (int(s.parent.name[-4:]) - 1) * 20.0
    on = t < OFF_AT
    n = len(t)
    g = np.where(on, 0.95 * 1.04, 0.95)                    # ON の間は自動合わせの m = 1.04、OFF のあとは設定のまま
    z['gains'] = np.stack([np.ones(n), g], axis=1).astype(np.float32)
    z.update(ceil_on=on, ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool), ceil_left=np.zeros(n, bool),
             sa_on=on, sa_m=np.where(on, 1.04, np.nan).astype(np.float32), sa_gain=g.astype(np.float32),
             sa_src=np.ones(n, np.int8), sa_event=np.zeros(n, np.int8), launch_on=np.zeros(n, bool),
             launch_end=np.zeros(n, np.int8))
    np.savez(s, **z)

res = s3.analyze_log(log, raw, TMP / 'labels', {'x': 'B'})
key, txt = qp.key_of(res)
sp = ss.score_log(log, TMP / 'labels', raw)['speed_ctrl']
print('実際: 0〜%.0f 秒は天井・自動 ON、そのあと OFF(AI のフレーム %d のうち ON %d・自動 ON %d)' % (
    OFF_AT, sp['ai_frames'], sp['ceil']['on_frames'], sp['auto']['on_frames']))
print('quick_eval_by_params の設定の文:', txt)
print('quick_eval_by_params の印:', qp.flags_of(res))
print('scorecard・car_check の天井・自動の欄:', ss.ctrl_cells({'speed_ctrl': sp, 'vbat': {}})[:2], '印', sp['flags'])
print('周(どれも「天井 4.1・自動」の本の周として数える): 通過 %d・きれいな周 %d(OFF のあとの周 %d)' % (
    res['laps']['passes'], res['laps']['clean'], sum(1 for a in anchors if a >= OFF_AT) - 1))
print('log.json の drive.params:', {k: P[k] for k in ('speed_ceiling', 'speed_auto')})
