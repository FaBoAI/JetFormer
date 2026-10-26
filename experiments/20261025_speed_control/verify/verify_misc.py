"""反証 V-drive: 時間・status の JSON(NaN なし)・走行のあとも ON が残るか・起動で OFF・走行中の決まり(サーバーの API の形で)"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261025_speed_control/verify')
import vdrive_harness as H  # noqa: E402
from jetformer import drive  # noqa: E402
from verify_off import PB, vb_good  # noqa: E402

# 1) 時間(No.9・ON・電圧あり)と status の JSON
src = H.LogSource('20261024-162033', vbat=vb_good)
r = H.run_loop(drive, dict(PB, speed_ceiling=1, speed_auto=1), src, src.pred_of, description='V_before=8.30')
s = r['st']['stats']
print('時間: ceil_ms %.3f (最大 %.3f)・auto_ms %.3f (最大 %.3f)・ゾーン %.3f' % (
    s['ceil_ms'], s['ceil_ms_max'], s['auto_ms'], s['auto_ms_max'], s['zone_ms']))
d = r['d']
st = d.status()
try:
    json.dumps(st, allow_nan=False, default=str)
    print('status の JSON: NaN なし OK')
except ValueError as e:
    print('status の JSON: NaN あり NG', e)
print('now.ceil', st['now'].get('ceil'), '\nnow.auto', st['now'].get('auto'))
# 2) 走行のあと: params は ON のまま(次の走行に持ち越す)
print('走行のあとの params: speed_ceiling=%s speed_auto=%s' % (d.params['speed_ceiling'], d.params['speed_auto']))
# 次の start で params に speed_ceiling を入れない(古い画面・別の道具)と ON のまま
p2 = d._speed_ctrl_rules(d._clean_params({'throttle_gain': 0.9}), False)
print('次の start(speed_ceiling を送らない)→ speed_ceiling=%s speed_auto=%s' % (p2['speed_ceiling'], p2['speed_auto']))


# 3) 走行中の決まり(偽の thread)
class Alive(object):
    def is_alive(self):
        return True


d.thread = Alive()
res = {}
for body in ({'speed_ceiling': 1}, {'speed_auto': 1}, {'ceil_left_mps': 4.2}, {'auto_vref_v': 8.0},
             {'speed_ceiling': '1'}, {'speed_ceiling': 0.6}, {'speed_ceiling': True}, {'speed_auto': 0}, {'speed_ceiling': 0},
             {'speed_ceiling': 1}, {'speed_ceiling': 0.49, 'speed_auto': 0}):
    try:
        out = d.set_params(body)
        res[json.dumps(body)] = 'OK → ceil %s auto %s' % (out['speed_ceiling'], out['speed_auto'])
    except (ValueError, TypeError) as e:
        res[json.dumps(body)] = '断る: %s' % str(e)[:40]
for k, v in res.items():
    print('走行中 set_params %-40s %s' % (k, v))
d.thread = None
# 4) 範囲の外・数でない
for body in ({'ceil_all_mps': float('nan')}, {'ceil_all_mps': 'abc'}, {'ceil_left_mps': 9.0}, {'auto_target_mps': 1.0},
             {'speed_ceiling': -5}, {'speed_ceiling': 7}):
    try:
        out = d.set_params(body)
        print('止まって set_params %-30s OK ceil_all %s left %s target %s ceil %s' % (
            json.dumps(body), out['ceil_all_mps'], out['ceil_left_mps'], out['auto_target_mps'], out['speed_ceiling']))
    except (ValueError, TypeError) as e:
        print('止まって set_params %-30s 断る %s' % (json.dumps(body), e))
