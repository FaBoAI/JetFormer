"""反証 2 回目・record: 自動合わせが電圧の源「なし」(sa_src = 0)で走った本でも、走ったあとに説明欄へ V_before を書くと、
car_check の文は「電圧 …(description)」になり、自動合わせの源(sa_src)はどの文にも出ないことを確かめる(手順書 6 章の
「+ 自動合わせ ON: 電圧の源が 基板 か 説明欄」を道具の文で決められない)。作り物の記録だけ(tmp_record の下)。"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF / 'tests'))
sys.path.insert(0, str(JF / 'tools'))
import car_check as cc  # noqa: E402
from test_lap_anchor import write_lap_log  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix2/tmp_refute_R2rec5/autosrc'
if TMP.exists():
    shutil.rmtree(TMP)
raw = TMP / 'raw'
raw.mkdir(parents=True)
log = '20001103-131000'
P = {'throttle_gain': 0.95, 'speed_ceiling': 1, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 1, 'auto_target_mps': 4.27}
desc0 = 'ai_no=9 bench=real_bench_20261023 day=2000-11-03 block=8 cand=EXP tune=2-3 battery=#B V_before=__ V_after=__ check=__ memo='
d, anchors, _ = write_lap_log(raw, log, dur=60.0, false_laps=(), description=desc0, params=P)
for s in sorted(d.glob('*/AI_CMD.npz')):
    z = dict(np.load(s))
    n = len(z['timestamps_us'])
    z.update(ceil_on=np.ones(n, bool), ceil_mode=np.zeros(n, np.int8), ceil_applied=np.zeros(n, bool),
             ceil_left=np.zeros(n, bool), sa_on=np.ones(n, bool), sa_m=np.ones(n, np.float32),
             sa_gain=np.full(n, 0.95, np.float32), sa_src=np.zeros(n, np.int8),       # 0 = 電圧の源なし(ファーム無し・V_before 空)
             sa_event=np.zeros(n, np.int8))
    np.savez(s, **z)
    # 走ったあとのメモ(走り終わりの欄)で V_before を書いた(説明欄はシーンの scene.json に入る)
    (s.parent / 'scene.json').write_text(json.dumps({'description': desc0.replace('V_before=__', 'V_before=8.31')
                                                     .replace('V_after=__', 'V_after=8.02')}), encoding='utf-8')
res = cc.check_log(log, TMP / 'labels', raw)
txt = cc.to_text(res)
print('\n'.join(x for x in txt.splitlines() if '電圧' in x or '天井' in x or '自動' in x))
print('JSON にだけある自動合わせの源 sa_src:', res['speed_ctrl']['auto']['src'], '(0 = なし・1 = 基板・2 = 説明欄)')
print('vbat.source:', res['vbat']['source'])
# 追加: 同じ記録で、走る前に V_before を書いた本(sa_src = 2)と比べる。文が同じなら、文だけでは見分けられない
for s in sorted(d.glob('*/AI_CMD.npz')):
    z = dict(np.load(s)); z['sa_src'] = np.full(len(z['timestamps_us']), 2, np.int8); np.savez(s, **z)
res2 = cc.check_log(log, TMP / 'labels', raw)
t2 = '\n'.join(x for x in cc.to_text(res2).splitlines() if '電圧' in x or '天井' in x or '自動' in x)
print('--- sa_src=2 の本 ---'); print(t2)
print('文が同じか:', t2 == '\n'.join(x for x in txt.splitlines() if '電圧' in x or '天井' in x or '自動' in x))
shutil.rmtree(TMP.parent)
