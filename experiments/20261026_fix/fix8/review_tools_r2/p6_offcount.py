"""反証 8-2 回目・係 tools: OFF の本が比べから外れて 1〜2 本だけ残る日(p2 の e の続き)と、短い止まりが続く日(p1 の C・p4 の S8)の行。

  1  p2 の e の日(止まりが無ければ ON − OFF −0.10「満たさない」)で、OFF の本を外す形を変える:
       stall  2 m/s に届いたあと(2.30 秒から)にカメラが 0.3 秒止まる(起点までは約 0.01〜0.02 秒しか伸びない。8-1 の直しで外れる)
       A      両方 ON の日の hold_long の形(1.6〜2.5 秒にカメラが止まって止まる → 明けて中止 → 5.5 秒に人側 = R1 に届かない。前から外れる)
       mix    1 本は stall・1 本は A
  2  p4 の S8(3-5・3-7 の 2 本に 0.200 秒 × 4 の短い止まり)の標準出力の行

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p6_offcount.py
  → out/p6_offcount.txt
"""
import common as C
from common import T, t7, PRM, ON, IG
import p4_args as P4


def day(tmp, kinds):
    """kinds = {block: 'stall' | 'A'}(OFF の本だけ)"""
    D = t7.Day(tmp)
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.20), (14, PRM, 0.23), (15, ON, 0.25), (16, PRM, 0.22), (17, ON, 0.30),
                         (18, PRM, 0.44), (19, ON, 0.33)):
        k = kinds.get(blk)
        if k == 'stall':
            gaps = [(69, 8)]
            D.add(blk, prm, dm, 0.6, vk=C.sim_vk(30.0, 1.0 + dm, C.neutral_of(gaps)), stall=C.gaps_to_frames(gaps))
        elif k == 'A':
            s0, s1, dur = 1.6, 2.5, 8.0
            v_s0 = 2.0 * (s0 - (1.0 + dm)) / 0.6
            vk = [(0.0, 0.0), (1.0 + dm, 0.0), (s0, v_s0), (s1 - 0.1, 0.0), (dur, 0.0)]
            D.add(blk, prm, dm, 0.6, dur=dur, vk=vk, ai=[(1.0, 5.5)], stall=[(s0, s1)], giveup=[(s1, 5.5)],
                  events=[{'kind': 'recovery_guard', 'what': 'hold_long', 'phase': None, 'backed_s': 0.0, 'hold_s': 0.9,
                           'run': 'runB', 'cam_epoch_us': int(t7.T0_US + s1 * 1e6), 't_us': int(t7.T0_US + s1 * 1e6)}])
        else:
            D.add(blk, prm, dm, 0.6 if prm is PRM else 0.3)
    return D


def main():
    lines = ['# 1 OFF の本が外れて残りが少ない日(ON 4 本 2.863・2.913・2.963・2.993 / OFF 3-1 3.013・3-3 3.043・3-5 3.033・3-7 3.253)']
    try:
        for name, kinds in (('なし', {}), ('stall 3-1・3-3', {12: 'stall', 14: 'stall'}), ('A 3-1・3-3', {12: 'A', 14: 'A'}),
                            ('mix 3-1 stall・3-3 A', {12: 'stall', 14: 'A'}), ('stall 3-1', {12: 'stall'}), ('A 3-1', {12: 'A'}),
                            ('stall 3-1・3-3・3-5', {12: 'stall', 14: 'stall', 16: 'stall'}),
                            ('A 3-1・3-3・3-5', {12: 'A', 14: 'A', 16: 'A'})):
            D = day(C.fresh('o%d' % len(lines)), kinds)
            lines.append(' OFF の本を外す形: %s' % name)
            for ver in ('r1old', 'r1new'):
                j, o, _ = C.runc(D, ver, IG, 'o%s%d' % (ver, len(lines)))
                g = T.g0(j)
                lines.append('  %s %s' % ('直す前' if ver == 'r1old' else '今    ', C.gtext(g, ('launch_stall_excluded', 'not_reached'))))
        lines.append('')
        lines.append('# 2 3-5・3-7 の 2 本に 0.200 秒 × 4 の短い止まり(p4 の S8)の今の標準出力(組の行)')
        D = P4.s8(C.fresh('s8'))
        j, o, _ = C.runc(D, 'r1new', IG, 's8n')
        g = T.g0(j)
        lines.append('  ' + C.gtext(g, ('launch_gap_short', 'launch_stall_excluded')))
        for x in o.splitlines():
            if x.startswith('  ON − OFF') or x.startswith(T.GAPNOTE_HEAD) or x.startswith('  off ') or x.startswith('  on '):
                lines.append('  行: %s' % x)
        offs = sorted((r['log'][-6:], r['to_r1_s']) for r in j['launches'] if r.get('launch_state') == 'off')
        lines.append('  OFF の本の起点まで: %s(止まりが無ければ 3-5 3.033・3-7 3.253)' % offs)
    finally:
        C.cleanup()
    C.write('p6_offcount.txt', lines)


if __name__ == '__main__':
    main()
