"""反証 7-1・docs F2: 手順書 7 章「中立で停止中」の行(11/3 = 残す ON の日)の文
「0.5 秒より長く止まると、…止まっている間に 2.5G 以上の衝撃があった車は「リカバリー中止」(中立で人を待つ)になる」
が、今の assist.Recovery(fix5 s1: 衝撃だけで中止にするのは、明けたとき前へ 0.5 m/s より遅いか後ろへ動いているときだけ)と
合っているかを、本物の assist.py で流して見る(読むだけ。車・装置に触れない)。

    cd ~/jetformer && python3 -I experiments/20261026_fix/fix7/review_docs_r1/demo_f2_hold_shock.py
"""
import sys

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A  # noqa: E402

C = N = 1500.0
FDT = 1 / 30.0
MODEL = (C, N - 120.0)


def armed():
    r = A.Recovery((C, N))
    t = 100.0
    for _ in range(40):                       # 前へ 3 m/s で走っている(armed・段なし)
        r.step(MODEL, -6000.0, t, 0.2, back_guard=True)
        t += FDT
    return r, t - FDT


print('Recovery.HOLD_KEEP_MAX_S =', A.Recovery.HOLD_KEEP_MAX_S, '・IMPACT_G =', A.Recovery.IMPACT_G,
      '・HOLD_IMPACT_FWD_ERPM =', A.Recovery.HOLD_IMPACT_FWD_ERPM)
for erpm, g in ((-4400.0, 3.0), (-6000.0, 5.7), (-999.0, 3.0), (0.0, 3.0)):
    r, last = armed()
    out, kind = r.step(MODEL, erpm, last + 1.0, 0.0, back_guard=True, hold_g=g, hold_since=last + 0.1, hold_back_s=0.0)
    print('中立止め 1.0 秒・その間の衝撃 %.1fG・明けたとき eRPM %6.0f(%.1f m/s 前へ)→ 段 %-7s 出す値 %s 種類 %r' % (
        g, erpm, -erpm / 2000.0, r.phase, out, A.ASSIST_NAMES.get(kind, kind)))
print('手順書 7 章の文どおりなら、どの行も「リカバリー中止」(段 giveup)のはず')

print()
print('リカバリーの途中(段 escape = 当たったあとの前進)で 1.0 秒の中立止め:')
for erpm, g in ((-4400.0, 0.0), (-4400.0, 4.0), (0.0, 0.0)):
    r, last = armed()
    r._go('escape', last)
    out, kind = r.step(MODEL, erpm, last + 1.0, 0.0, back_guard=True, hold_g=g, hold_since=last + 0.1, hold_back_s=0.0)
    print('  衝撃 %.1fG・明けたとき %.1f m/s 前へ → 段 %-7s 種類 %r' % (g, -erpm / 2000.0, r.phase, A.ASSIST_NAMES.get(kind, kind)))
