"""plan_check(読むだけ)にトリム 15 と 0 を渡し、EXP の印が付くかを見る"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.web import bench  # noqa: E402
inp = json.loads(Path(sys.argv[1]).read_text())
d = inp['bench']; day = [x for x in d['days'] if x['date'] == '2026-11-03'][0]; blk = day['blocks'][1]
base = dict(d['baseline']['params']); base.update(blk.get('params') or {})
for trim in (0.0, 15.0):
    p = dict(base, steer_trim_us=trim)
    desc, diffs = bench.plan_check(blk.get('description', ''), blk['run'], 5, p)
    print('trim', trim, 'diffs', diffs, 'cand=EXP' in desc)
print('PLAN_KEYS has trim:', 'steer_trim_us' in bench.PLAN_KEYS)
