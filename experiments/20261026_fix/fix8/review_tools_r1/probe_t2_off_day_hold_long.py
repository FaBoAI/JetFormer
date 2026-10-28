"""反証 8-1 回目・係 tools・台本 2(読むだけ): approvals 3「途中で OFF にした日」= (a)「そのまま ⑤ の行で決める」の前提の外の 1 形
(hold_long。0.5 秒より長い AI 側の中立止め(カメラの止まり)の間に、走ったあとの車が止まった発進)が、発進の前進 OFF の本で起きたとき、
⑤ の答えが「使ってよい」の側に動くか。

main session の決め(approvals 3 の「途中で OFF にした日」の決まりの行・手順書 3 章): 「両方 ON は明けて中止(人を待つ)、OFF は走り続ける。
⑤ はその発進を「R1 に届かなかった発進」(ON の組なら満たさない)、その場で戻した発進を走り直しにする =「使ってよい」の側には動かない」。
ここでは発進の前進 OFF の組(比べの OFF 側)の本で同じ形が起きたときを見る。

作り物の回 3(8 本。ON = 発進の前進 ON・OFF = 発進の前進 OFF):
  3-1 OFF 3.013 秒・3-2 ON 2.813・3-3 OFF 3.043・3-4 ON 2.863・3-5 OFF(この本で形が起きる)・3-6 ON 2.943・3-7 OFF 3.253・3-8 ON 2.973
  3-5: 発進(1.0 秒)→ 1.6〜2.5 秒にカメラが止まる(AI_CMD・カメラのフレームが無い)。その間に車は中立の惰性で止まる(2.4 秒に 0)。
    A 両方 ON の日(朝から全部の本で見張り・残す ON): 明けたフレーム(2.5 秒)から中止(assist 3。出来事 hold_long)→ 5.5 秒に人側 → 本の終わり
    B 3-5 から先をわざと OFF にした日(紙の本 X = 3-5): 明けたらモデルのまま走り続ける(2.6 秒から出て 3.2 秒に 2 m/s)
    C 両方 ON の日で、同じ止まりの間に 0.7 m/s まで落ちるが止まらない(明けたとき 0.5 m/s 以上 = 中止しない。走り続ける)
  比べ(参考): カメラの止まりが無ければ 3-5 は 3.033 秒前後(ほかの OFF と同じ作り)
⑤ は手順書どおり --ignore-gain。出力 out/probe_t2_off_day_hold_long.txt

  jf-heavy --lock none --mem 1500M -- python -B probe_t2_off_day_hold_long.py
"""
import json

import common as C
from common import t7, PRM, ON

OFFOPT = {'rec_back_guard': 0, 'rec_hold_keep': 0}


def day(tmp, case, boost_of_35=False):
    D = t7.Day(tmp)
    off_from = 16 if case == 'B' else 99           # 3-5 = block 16
    p = lambda blk, prm: dict(prm, **OFFOPT) if blk >= off_from else prm      # noqa: E731
    D.add(12, p(12, PRM), 0.20, 0.6)
    D.add(13, p(13, ON), 0.15, 0.3)
    D.add(14, p(14, PRM), 0.23, 0.6)
    D.add(15, p(15, ON), 0.20, 0.3)
    prm35 = p(16, ON if boost_of_35 else PRM)
    # 1.22 秒に動き出し、1.6 秒に 1.27 m/s(0.6 秒で 2 m/s に向かう途中)→ カメラが止まった間(1.6〜2.5 秒)に中立の惰性で 2.4 秒に止まる
    base = [(0.0, 0.0), (1.22, 0.0), (1.6, 2.0 * 0.38 / 0.6), (2.4, 0.0)]
    if case == 'N':             # 止まりの無い 3-5(比べの元)
        log35 = D.add(16, prm35, 0.22, 0.6)
    elif case == 'A':
        dur = 8.0
        vk = base + [(dur, 0.0)]
        log35 = D.add(16, prm35, 0.22, 0.6, dur=dur, vk=vk, ai=[(1.0, 5.5)], stall=[(1.6, 2.5)], giveup=[(2.5, 5.5)],
                      events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.9,
                               'run': 'runB', 'cam_epoch_us': int(t7.T0_US + 2.5e6), 't_us': int(t7.T0_US + 2.5e6)}])
    elif case == 'B':
        dur = 30.0
        vk = base + [(2.6, 0.0), (3.2, 2.0), (dur, 2.0)]
        log35 = D.add(16, prm35, 0.22, 0.6, dur=dur, vk=vk, ai=[(1.0, dur - 2.0)], stall=[(1.6, 2.5)])
    else:                       # C: 両方 ON の日。止まりの間に 0.7 m/s まで落ちるが止まらない(明けて 0.5 m/s 以上 = 中止しない)
        dur = 30.0
        vk = [(0.0, 0.0), (1.22, 0.0), (1.6, 2.0 * 0.38 / 0.6), (2.5, 0.7), (3.1, 2.0), (dur, 2.0)]
        log35 = D.add(16, prm35, 0.22, 0.6, dur=dur, vk=vk, ai=[(1.0, dur - 2.0)], stall=[(1.6, 2.5)])
    D.add(17, p(17, ON), 0.28, 0.3)
    D.add(18, p(18, PRM), 0.44, 0.6)
    D.add(19, p(19, ON), 0.31, 0.3)
    return D, log35


def main():
    lines = []
    for boost in (False, True):
        for case in ('N', 'A', 'B', 'C'):
            D, log35 = day(C.fresh('t2_%s_%d' % (case, boost)), case, boost)
            j, out, _ = C.runc(D, 'new', C.IG, 't2%s%d' % (case, boost))
            lines.append('== %s・3-5 は発進の前進 %s(%s)' % ({'N': 'N 両方 ON の日・カメラの止まりなし', 'A': 'A 両方 ON の日', 'B': 'B 3-5 から OFF にした日', 'C': 'C 両方 ON の日・止まりの間に止まらない'}[case],
                                                         'ON' if boost else 'OFF', log35))
            for r in j['launches']:
                if r['log'] == log35:
                    lines.append('  3-5 の発進の行: to_r1_s %s・why_no_r1 %s・launch_state %s・settings の見張り %s' % (
                        r.get('to_r1_s'), r.get('why_no_r1'), r.get('launch_state'), (r.get('settings') or {}).get('rec_back_guard')))
            for g in j['summary']:
                lines.append('  ' + C.group_text(g))
                off = (g.get('by_launch') or {}).get('off') or {}
                lines.append('  OFF の組の発進: %s' % json.dumps(off.get('logs'), ensure_ascii=False))
            lines += ['  | ' + x for x in out.splitlines() if x.lstrip().startswith(('ON − OFF', '注意', 'R1 に届かなかった'))
                      or '→ 使ってよい' in x or '→ 満たさない' in x]
    C.write('probe_t2_off_day_hold_long.txt', lines)
    C.cleanup()


if __name__ == '__main__':
    main()
