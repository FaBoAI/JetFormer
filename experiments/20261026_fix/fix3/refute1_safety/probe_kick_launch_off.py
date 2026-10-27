"""反証 1 回目・safety: safety1-2 の直し(launch_no_kick)は launch_boost = 0 の本では効かない(仕様)。前 0.5 m・0.9 m の物・
床の基準なしで、launch_no_kick = 1 でも StartKick が出る長さを、本物の走行ループで数える(装置は開かない)。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_refute1 as P                  # noqa: E402
from probe_refute1 import A, C, N, PARAMS, build, run, tl, tof_near   # noqa: E402

out = {}
for name, tof, floor in (('front_0.5m', tof_near(500), True), ('front_0.9m', tof_near(900), True), ('no_floor_ref', None, False)):
    for lb in (0, 1):
        frames, imu = build(4.0, sel=lambda s: s >= 1.0, v=0.0, pred=(C, N), tof=tof)
        p = dict(PARAMS, start_kick=1, launch_boost=lb, launch_no_kick=1)
        rows = tl(frames, run(p, frames, imu, floor=floor)[2])
        k = [x for x in rows if x[1] == A.ASSIST_START]
        out['%s launch_boost=%d launch_no_kick=1' % (name, lb)] = {'kick_s': round(len(k) / 30.0, 2),
                                                                  'kick_us': sorted({x[3] for x in k})}
(P.OUT / 'probe_kick_launch_off.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
