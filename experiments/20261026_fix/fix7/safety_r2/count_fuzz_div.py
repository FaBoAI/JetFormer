"""直す 7-2 回目・係 safety: 試験 test_recovery_on_diverges_only_by_s4c と同じでたらめな入力で、直す前と違ったフレームの数・
そのうちの early_none の数・skip_gap の 'reverse' に入った回数を数える(確かめ。装置に触れない)"""
import sys
sys.path[:0] = ['/home/jetson/jetformer/tests', '/home/jetson/jetformer']
import test_rec_fix7_safety_r2 as T  # noqa: E402
from jetformer import assist as A  # noqa: E402

amod, _ = T.load_r1()
tot = {'diverge': 0, 'skip_entries': 0, 'frames': 0, 'skip_frames': 0}
for seed in range(12):
    r0, r1 = amod.Recovery((T.C, T.N)), A.Recovery((T.C, T.N))
    for model, e, t, kw in T._fuzz_stops(500 + seed):
        sk = r1.skip_gap and r1.phase == 'reverse'
        a = r0.step(model, e, t, back_guard=True, **kw)
        b = r1.step(model, e, t, back_guard=True, **kw)
        tot['frames'] += 1
        tot['skip_frames'] += sk
        if not sk and r1.skip_gap and r1.phase == 'reverse':
            tot['skip_entries'] += 1
        if a != b or any(getattr(r0, k) != getattr(r1, k) for k in T.STATE):
            tot['diverge'] += 1
            r0, r1 = amod.Recovery((T.C, T.N)), A.Recovery((T.C, T.N))
        r0.guard_event = r1.guard_event = None
        if r1.phase == 'giveup':
            r0, r1 = amod.Recovery((T.C, T.N)), A.Recovery((T.C, T.N))
print(tot)
