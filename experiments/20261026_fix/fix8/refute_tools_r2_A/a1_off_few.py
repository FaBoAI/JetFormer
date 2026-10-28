"""票 A・指摘 1(w2-tools-r8-2-off-few-left)を自分の作り物で確かめる。

回 3(自分で決めた数字): OFF 3-1 2.98・3-3 3.05・3-5 3.10・3-7 3.40 秒 / ON 3-2 2.90・3-4 2.92・3-6 2.95・3-8 2.97 秒(起点まで。
⑤ の起点の見つけ方でそれぞれ約 +0.01 秒)。止まりが無ければ ON 中央値 2.935 − OFF 中央値 3.075 = −0.14 秒「満たさない」。

  S*  OFF の本の 3.0 秒(2 m/s に届いたあと・R1 の約 1 秒前)のフレームのあと、フレームの間 0.300 秒(8 枚無い)。
      flat = 速さはそのまま(フレームだけ無い)/ coast = drive の中立(3.1〜3.3 秒)の間だけ惰性 −0.9 m/s²、明けたら元の加速で 2 m/s へ
  N*  OFF の本が R1 に届かない(8-1 の前からの形): human = 2.6 秒に人側(AI の区間の終わり)/ hold = 両方 ON の日の hold_long の形
      (1.4 秒から 0.8 秒カメラが止まり、その間に止まる → 明けて中止(assist 3)→ 5.0 秒に人側)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/refute_tools_r2_A/a1_off_few.py
  → out/a1_off_few.txt
"""
import statistics

import common_a as C

BOOKS = ((12, C.PRM, 0.18), (13, C.ON, 0.25), (14, C.PRM, 0.25), (15, C.ON, 0.27),
         (16, C.PRM, 0.30), (17, C.ON, 0.30), (18, C.PRM, 0.60), (19, C.ON, 0.32))
NAME = {12: '3-1', 14: '3-3', 16: '3-5', 18: '3-7'}


def day(name, stall_books=(), shape='flat', nr_books=(), nr_kind='human'):
    D = C.Day(name)
    for blk, prm, dm in BOOKS:
        if blk in stall_books:
            st, neu, _ = C.gap_pattern(3.0, 9, 1)          # 3.0 秒のフレームのあと 0.300 秒
            D.add(blk, prm, dm, stall=st, neutral=neu if shape == 'coast' else ())
        elif blk in nr_books and nr_kind == 'human':
            vk = C.sim(30.0, C.T_AI + dm, C.ACC_OFF, [(2.6, 30.0)])
            D.add(blk, prm, dm, vk=vk, ai=[(C.T_AI, 2.6)])
        elif blk in nr_books and nr_kind == 'hold':
            t_move = C.T_AI + dm
            v14 = C.ACC_OFF * (1.4 - t_move)
            vk = [(0.0, 0.0), (t_move, 0.0), (1.4, v14), (1.5, v14), (2.1, 0.0), (30.0, 0.0)]
            ev = [{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.8, 'run': 'runB',
                   'cam_epoch_us': int(t7_T0 + 2.2e6), 't_us': int(t7_T0 + 2.2e6)}]
            D.add(blk, prm, dm, vk=vk, ai=[(C.T_AI, 5.0)], stall=[(1.4, 2.2)], giveup=[(2.2, 5.0)], events=ev)
        else:
            D.add(blk, prm, dm)
    return D


t7_T0 = C.t7.T0_US


def main():
    C.check_sha()
    L = ['# 指摘 1: OFF の本が比べから外れて OFF が 2 本・1 本残る形(今 = d2b210cc / 直す前 = fix8_tools_r1_before 26dfb087)。--ignore-gain']
    cases = [('base 止まりなし', {}),
             ('S2 3-1・3-3(速い OFF 2 本)flat', dict(stall_books=(12, 14))),
             ('S2 3-1・3-3 coast', dict(stall_books=(12, 14), shape='coast')),
             ('S2 3-5・3-7(遅い OFF 2 本)flat', dict(stall_books=(16, 18))),
             ('S2 3-1・3-7 flat', dict(stall_books=(12, 18))),
             ('S1 3-1 flat', dict(stall_books=(12,))),
             ('S3 3-1・3-3・3-5 flat', dict(stall_books=(12, 14, 16))),
             ('S3 3-3・3-5・3-7 flat', dict(stall_books=(14, 16, 18))),
             ('N2 3-1・3-3 が R1 に届かない(2.6 秒に人側)', dict(nr_books=(12, 14))),
             ('N2 3-1・3-3 が hold_long の形で R1 に届かない', dict(nr_books=(12, 14), nr_kind='hold')),
             ('N1 3-1 が hold_long の形', dict(nr_books=(12,), nr_kind='hold')),
             ('N2 3-5・3-7 が R1 に届かない(2.6 秒に人側)', dict(nr_books=(16, 18))),
             ('S1+N1 3-1 止まり・3-3 hold_long', dict(stall_books=(12,), nr_books=(14,), nr_kind='hold'))]
    try:
        for i, (label, kw) in enumerate(cases):
            D = day('a1_%d' % i, **kw)
            L.append('## %s' % label)
            for ver in ('old', 'now'):
                j, o = C.run(D, ver, C.IG, 'a1_%d' % i)
                L.append('  ' + C.brief(j, o, '直す前' if ver == 'old' else '今    '))
                gs = C.cmp_groups(j)
                if gs:
                    g = gs[0]
                    offs = [r['to_r1_s'] for r in j['launches'] if r.get('launch_state') == 'off' and r.get('to_r1_s') is not None
                            and not r.get('launch_stall')]
                    if len(offs) == 2:
                        L.append('      OFF 2 本: statistics.median %s・平均 %.4f・⑤ の中央値 %s(2 本の中央値 = 2 本の平均)' % (
                            round(statistics.median(offs), 4), sum(offs) / 2, g['by_launch']['off']['to_r1_median_s']))
                    for x in C.onoff_line(o):
                        L.append('      | ' + x[:150])
                    st = [r for r in j['launches'] if r.get('launch_stall')]
                    for r in st:
                        L.append('      launch_stall %s@%.2f %s 起点まで %s' % (r['log'][-6:], r['t_ai'], r['launch_stall'], r['to_r1_s']))
            # 止まりのない同じ本の起点まで(止まりで伸びた分)
        # 引数なし(ゲインは全部 0.9 なので組は同じ)でも同じか: S2 3-1・3-3 flat
        D = day('a1_noarg', stall_books=(12, 14))
        L.append('## 引数なし: S2 3-1・3-3 flat')
        for ver in ('old', 'now'):
            j, o = C.run(D, ver, (), 'a1_noarg')
            L.append('  ' + C.brief(j, o, '直す前' if ver == 'old' else '今    '))
    finally:
        C.cleanup()
    C.write('a1_off_few.txt', L)


if __name__ == '__main__':
    main()
