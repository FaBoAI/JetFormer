"""escape のあとの cooldown 中(slow_since = None)に 0.5 秒より長い中立止め → 明けて止まっている。中止になるか(直さない)"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A
C, N = 1500.0, 1508.0
MODEL = (C, N - 120.0)
FR = 1 / 30.
for L in (0.6, 1.0, 2.0):
    a = A.Assist((C, N))
    r = a.rec
    now = 100.0
    for _ in range(30):
        a.step(MODEL, -5000.0, True, now, rec_guard=True); now += FR
    # escape が終わったところを作る: 段 None・cooldown 1 秒・slow_since None
    r.phase, r.slow_since, r.cooldown_until, r.t_last = None, None, now + 1.0, now
    r.starts.append(now - 3.0)
    hold_t0 = now + 0.1
    a.reset(keep_recovery=True)
    now += L
    first, res = True, []
    t0 = now
    while now < t0 + 3.0:
        kw = dict(hold_g=0.0, hold_since=hold_t0, hold_back_s=0.0) if first else {}
        o, k = a.step(MODEL, 0.0, True, now, rec_guard=True, **kw)
        ev = r.guard_event; r.guard_event = None
        res.append((round(now - t0, 3), o[1], A.ASSIST_NAMES[k], ev)); first = False; now += FR
    gu = [t for t, thr, k, e in res if k == 'リカバリー中止']
    rv = [t for t, thr, k, e in res if k == 'リカバリー' and thr > N + 40]
    print('L=%.1f giveup_at=%s first_reverse_side_at=%s events=%s' % (L, gu[:1], rv[:1], [e for *_, e in res if e]))
