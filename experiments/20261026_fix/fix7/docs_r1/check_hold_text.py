"""fix7 係 docs 直す 7-1(w1-docs-r71-docs-3): 手順書 7 章「中立で停止中」の行と approvals 3「ON の形」に書く文を、本物の
assist.Recovery で確かめる(読むだけ。車・装置に触れない。作り物の値だけ)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I experiments/20261026_fix/fix7/docs_r1/check_hold_text.py
"""
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
    for _ in range(40):                       # 前へ 3 m/s で走っている(armed・段なし)
        r.step(MODEL, -6000.0, t, 0.2, back_guard=True)
        t += FDT
    return r, t - FDT


def wake(phase, erpm, g, hold=1.0):
    r, last = armed()
    if phase:
        r._go(phase, last)
    out, kind = r.step(MODEL, erpm, last + hold, 0.0, back_guard=True, hold_g=g, hold_since=last + 0.05, hold_back_s=0.0)
    return r.phase, out, A.ASSIST_NAMES.get(kind, kind)


print('定数: HOLD_KEEP_MAX_S %s 秒・IMPACT_G %s G・HOLD_IMPACT_FWD_ERPM %s eRPM(= %.1f m/s。2000 eRPM = 1 m/s)・STOP_ERPM %s・'
      'BACKING_ERPM %s・ESCAPE_US %s µs・HOLD_HALT_MAX_S %s 秒' % (
          R.HOLD_KEEP_MAX_S, R.IMPACT_G, R.HOLD_IMPACT_FWD_ERPM, R.HOLD_IMPACT_FWD_ERPM / 2000.0, R.STOP_ERPM,
          A.BACKING_ERPM, R.ESCAPE_US, R.HOLD_HALT_MAX_S))
rows = []
for phase in (None, 'escape', 'brake', 'wait', 'gap', 'reverse', 'coast'):
    for erpm in (-6000.0, -2000.0, -1001.0, -999.0, 0.0, 500.0):
        for g in (0.0, 3.0):
            for hold in (0.4, 1.0):
                p, out, kind = wake(phase, erpm, g, hold)
                rows.append((phase, erpm, g, hold, p, out, kind))
                print('段 %-7s 明けたとき %+6.0f eRPM(前へ %+.2f m/s)衝撃 %.1fG 止まり %.1f 秒 → 段 %-7s 出す値 %-16s 種類 %r' % (
                    phase, erpm, -erpm / 2000.0, g, hold, p, out, kind))
