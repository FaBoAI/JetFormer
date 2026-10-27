"""r2-r2-rs-1 の反証: refute2_safety の閉じたループ(本物の drive._loop)で、指摘の場面を一部だけ流し直す(直さない)"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/jetson/jetformer/experiments/20261026_fix/fix3/refute2_safety')
import closed_loop as CL          # noqa: E402
CL.TMP = Path('/home/jetson/jetformer/experiments/20261026_fix/fix3/tmp_r2r2rs1')
import probe_hold_retry as PR     # noqa: E402
import probe_hold_ready as PY     # noqa: E402

KEYS = ('back_m', 'v_back_max', 'rear_hits', 'rev_us_max_written', 'guard_events')
tr = 2.52
print('(a) nohold', {k: PR.one(1, 1)[k] for k in KEYS})
for o, L in ((0.03, 0.45), (0.06, 0.6), (0.10, 1.0)):
    for keep in (0, 1):
        r = PR.one(keep, 1, drop=(tr + o, tr + o + L))
        print('(a) cam o=%.2f L=%.2f keep=%d' % (o, L, keep), {k: r[k] for k in KEYS}, r['phases'][:7])
r = PR.one(1, 1, vskip=(tr + 0.06, tr + 0.56))
print('(a) veh o=0.06 L=0.5 keep=1', {k: r[k] for k in KEYS})
# 後ろの物なしで下がる距離(keep=1, 中立止めあり)
r = PR.one(1, 1, drop=(tr + 0.03, tr + 0.48), xr=None)
print('(a) cam o=0.03 L=0.45 keep=1 no-rear', {k: r[k] for k in KEYS})
# (b)
K2 = ('back_m', 'v_back_max', 'rear_hits', 'guard_events')
for keep, guard in ((0, 0), (1, 1)):
    print('(b) nohold keep=%d guard=%d' % (keep, guard), {k: PY.one(keep, guard)[k] for k in K2})
b = 2.42
for o, L in ((0.10, 0.8), (0.15, 0.5)):
    for keep in (0, 1):
        r = PY.one(keep, 1, drop=(b + o, b + o + L))
        print('(b) brake-hold o=%.2f L=%.1f keep=%d' % (o, L, keep), {k: r[k] for k in K2}, r['phases'][:6])
