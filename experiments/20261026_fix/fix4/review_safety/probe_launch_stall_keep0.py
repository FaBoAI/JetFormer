"""probe_launch_stall.py の「発進の前進の途中のカメラの止まりの間に前の壁に当たる」場面を keep = 0 でも流す(比べ)。
出力: out/probe_launch_stall_keep0.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_launch_stall as P     # noqa: E402
import rs_common as R              # noqa: E402

out = {}
try:
    for keep in (1, 0):
        for xf in (1.3, 1.6):
            for L in (1.0, 2.0):
                p = dict(P.PARAMS, recovery=1, rec_back_guard=1, rec_hold_keep=keep, tof_guard=1, launch_no_kick=0,
                         launch_boost=1, start_kick=0)
                sc = {'dur': 1.3 + L + 5.0, 'xf': xf, 'model': P.model, 'drops': [(1.3, 1.3 + L)], 'veh_skip': []}
                r = P.one(p, sc)
                out['launch_boost=1 keep=%d wall=%.1f stall=1.30+%.1f' % (keep, xf, L)] = r
finally:
    R.cleanup()
(HERE / 'out' / 'probe_launch_stall_keep0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
for k, v in out.items():
    print(k, '| hits', v['hits'][:1], '| launch', v['launch_rows'], '| giveup', v['giveup_first'], '| rev', v['reverse_first'],
          '| backed', v['backed_m_after_hit'], '| push', v['push_wall_after_camera_s'], '| first', v['first_row_after_camera'],
          '| call', v['first_recovery_call_after_hold'])
