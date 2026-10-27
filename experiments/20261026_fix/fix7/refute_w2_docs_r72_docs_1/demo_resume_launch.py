"""反証 7-2・docs: 手順書 7 章の中立止めの行のとおり、リカバリー中止のあと「CH3 を人側にしてから AI 側に戻す」と、発進の前進 ON の
本(回 3 の ON・回 4)では発進の前進がその場でもう一度出るかを、本物の assist.LaunchBoost で見る(読むだけ。作り物の値だけ)。"""
import sys
sys.path.insert(0, '/home/jetson/jetformer')
from jetformer import assist as A  # noqa: E402

N = 1500.0
lb = A.LaunchBoost(N)
FDT = 1 / 30.0


def inp(ttc):
    return A.LaunchIn(us=250.0, mps=2.5, max_s=1.2, v=0.0, a=0.0, gz=0.0, steer_rel=0.0, ttc=ttc, ttc5=(ttc,) * 5,
                      tof_ok=True, veh_ok=True, impact_g=0.0, guard_changed=False)


t = 0.0
for _ in range(10):                       # HS に置いて人側で待つ
    lb.human(); t += FDT
out, code = lb.step(N - 40.0, 0.0, t, inp(10.0))      # AI 側(1 回目の発進)
print('1 回目の発進(HS): 出す値 %s µs・始めた回数 %d・出来事 %s' % (out, lb.count, lb.event))
for _ in range(20):                       # 走って、そのあと中止で止まった(AI 側のまま。中止の間は中立)
    t += FDT
# 人が CH3 を人側(手順書 7 章「CH3 を人側にしてから AI 側に戻す」)→ その場で AI 側。前は 1.2 m 空き(TTC 3.3 秒 = 止まっていれば約 1 m)
for _ in range(30):
    lb.human(); t += FDT
out, code = lb.step(N - 40.0, 0.0, t, inp(3.6))
print('中止のあと人側 → その場で AI 側: 出す値 %s µs・始めた回数 %d・出来事 %s' % (out, lb.count, lb.event))
