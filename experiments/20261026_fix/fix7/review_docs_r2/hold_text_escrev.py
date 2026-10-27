"""反証 7-2・docs: 手順書 7 章「中立で停止中」の中止の形が、fix7 s4-a(esc_rev)のあとも同じかを本物の assist.Recovery で見る(読むだけ)。
docs_r1/check_hold_text.py と同じ場面を esc_rev = None(前と同じ)・False(drive が 'brake' 'wait' 'gap' で渡す値)で流して比べる。"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A  # noqa: E402

C = N = 1500.0
FDT = 1 / 30.0
MODEL = (C, N - 120.0)
R = A.Recovery


def armed():
    r = R((C, N))
    t = 100.0
    for _ in range(40):
        r.step(MODEL, -6000.0, t, 0.2, back_guard=True)
        t += FDT
    return r, t - FDT


def wake(phase, erpm, g, hold, esc_rev, hb):
    r, last = armed()
    if phase:
        r._go(phase, last)
        r.esc_ran = False          # 手順の始め(_start)と同じ: この手順ではまだ ESC が後退で走らせていない
    out, kind = r.step(MODEL, erpm, last + hold, 0.0, back_guard=True, hold_g=g, hold_since=last + 0.05, hold_back_s=hb,
                       esc_rev=esc_rev)
    return r.phase, out, A.ASSIST_NAMES.get(kind, kind)


diff = 0
n = 0
for phase in (None, 'escape', 'brake', 'wait', 'gap', 'reverse', 'coast'):
    for erpm in (-6000.0, -2000.0, -1001.0, -999.0, 0.0, 500.0):
        for g in (0.0, 3.0):
            for hold in (0.4, 1.0):
                for hb in (0.0, 0.2):
                    a = wake(phase, erpm, g, hold, None, hb)
                    b = wake(phase, erpm, g, hold, False, hb)
                    n += 1
                    if a != b:
                        diff += 1
                        print('違う: 段 %s eRPM %+.0f 衝撃 %.1fG 止まり %.1f 秒 中立止めの間の後退 %.1f 秒: esc_rev None → %s / False → %s'
                              % (phase, erpm, g, hold, hb, a, b))
print('場面 %d・違う %d' % (n, diff))
