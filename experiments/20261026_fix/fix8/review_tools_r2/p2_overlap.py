"""反証 8-2 回目・係 tools: カメラの止まり(launch_stall)と R1 に届かなかった発進・走り直しの重なり、ON が全部外れた日・OFF が少し
だけ残った日。⑤ を 8-1 の直す前の写し(r1old)と今(r1new)で流して並べる(--ignore-gain。手順書 6 章の読み方)。

  a  ON の発進でカメラが止まり、そのあと R1 の前で人側(R1 に届かない)
  b  リカバリー中止のあとの走り直し(その場から AI 側)の発進の 0.6〜1.5 秒にカメラの止まり(ON / OFF)
  c  ON の 4 本が全部カメラの止まり(OFF 4 本はふつう)
  d  OFF の 4 本が全部カメラの止まり・ON は全部 R1 に届かない(当たって人側)
  e  OFF の 4 本のうち 1・2・3 本で、2 m/s に届いたあと(2.30 秒から)にカメラが 0.3 秒止まる(起点までは約 0.02 秒しか伸びない)。
     止まりが無ければ ON − OFF −0.10 秒「満たさない」(OFF の 1 本 3-7 は遅い 3.253 秒)
  e2 同じ 1・2・3 本で、止まりが発進の始め(1.60 秒から 0.3 秒 = 伸びる形)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p2_overlap.py
  → out/p2_overlap.txt
"""
import common as C
from common import T, t7, t72, PRM, ON, IG


def both(lines, D, tag, logs=(), args=IG, heads=()):
    res = {}
    for ver in ('r1old', 'r1new'):
        j, o, _ = C.runc(D, ver, args, tag + ver)
        res[ver] = (j, o)
        lines.append('  %s %s' % ('直す前' if ver == 'r1old' else '今    ', ' / '.join(C.gtext(g) for g in j['summary'])))
        for lg in logs:
            for r in T.rows_of(j, lg):
                lines.append('      発進 %s@%.2f %s 起点まで %s・launch_stall %s・after_abort %s' % (
                    lg, r['t_ai'], r.get('launch_state'), r.get('to_r1_s'), r.get('launch_stall'),
                    (r.get('after_abort') or {}).get('kind')))
        for x in C.compare_lines(o):
            lines.append('      行: %s' % x[:200])
        for x in C.lines_with(o, *heads):
            lines.append('      行: %s' % x[:200])
    return res


def case_a(lines):
    lines.append('# a ON の発進でカメラが 1.6〜2.5 秒止まり、3.0 秒に人側(R1 に届かない)')
    D = t7.Day(C.fresh('a'))
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.15), (14, PRM, 0.23), (15, ON, 0.20), (16, PRM, 0.22)):
        D.add(blk, prm, dm, 0.6 if prm is PRM else 0.3)
    lg = D.add(17, ON, 0.28, 0.3, dur=8.0, vk=t7.launch_keys(1.0, 0.28, 0.3) + [(1.6, 1.0), (2.5, 0.2), (3.0, 0.5), (4.0, 0.0),
                                                                            (8.0, 0.0)],
               ai=[(1.0, 3.0)], stall=[(1.6, 2.5)])
    D.add(18, PRM, 0.44, 0.6)
    D.add(19, ON, 0.31, 0.3)
    both(lines, D, 'a', [lg], heads=('  R1 に届かなかった発進', T.STALL_HEAD, T.STALL_ON_HEAD))
    lines.append('')


def case_b(lines):
    lines.append('# b 走り直しの発進にカメラの止まり(t_res + 0.6〜1.5 秒)')
    for name, b38 in (('ON', ON), ('OFF', PRM)):
        D, lg, tt = t72.day(C.fresh('b' + name), 'abort', b38=b38,
                            stall=lambda tt: [(tt['t_st'], tt['t_st'] + 0.8), (tt['t_res'] + 0.6, tt['t_res'] + 1.5)])
        lines.append(' 3-8 = %s' % name)
        both(lines, D, 'b' + name, [lg], heads=('  リカバリー中止のあとの走り直し', T.STALL_HEAD, T.STALL_ON_HEAD))
    lines.append('')


def case_c(lines):
    lines.append('# c ON の 4 本が全部カメラの止まり(1.6〜2.5 秒)')
    D = t7.Day(C.fresh('c'))
    logs = []
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.15), (14, PRM, 0.23), (15, ON, 0.20), (16, PRM, 0.22), (17, ON, 0.28),
                         (18, PRM, 0.44), (19, ON, 0.31)):
        if prm is ON:
            logs.append(D.add(blk, prm, dm, 0.3, stall=[(1.6, 2.5)]))
        else:
            D.add(blk, prm, dm, 0.6)
    both(lines, D, 'c', [], heads=(T.STALL_HEAD, T.STALL_ON_HEAD))
    lines.append('')


def case_d(lines):
    lines.append('# d OFF の 4 本が全部カメラの止まり・ON の 4 本は全部 3.0 秒に人側(R1 に届かない。2.5 秒に 6G)')
    D = t7.Day(C.fresh('d'))
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.15), (14, PRM, 0.23), (15, ON, 0.20), (16, PRM, 0.22), (17, ON, 0.28),
                         (18, PRM, 0.44), (19, ON, 0.31)):
        if prm is ON:
            D.add(blk, prm, dm, 0.3, dur=8.0, vk=t7.launch_keys(1.0, dm, 0.3) + [(3.0, 2.0), (4.0, 0.0), (8.0, 0.0)],
                  ai=[(1.0, 3.0)], hits=[(2.5, 6.0)])
        else:
            D.add(blk, prm, dm, 0.6, stall=[(1.6, 2.5)])
    both(lines, D, 'd', [], heads=(T.STALL_HEAD, '  → OFF の発進が全部', '  → ON の発進が R1 の前で終わった', '  R1 に届かなかった発進'))
    lines.append('')


def eday(tmp, stalled, k0=69, n=8):
    """回 3。OFF 3-1 3.013・3-3 3.043・3-5 3.033・3-7 3.253 秒、ON(ゆっくりめ)2.863・2.913・2.963・2.993 秒 → 中央値 3.038・2.938 =
    ON − OFF −0.10「満たさない」。stalled の OFF の本(block)に、フレーム k0 のあと n 枚が無いカメラの止まり(本物に近い速さ)"""
    D = t7.Day(tmp)
    logs = {}
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.20), (14, PRM, 0.23), (15, ON, 0.25), (16, PRM, 0.22), (17, ON, 0.30),
                         (18, PRM, 0.44), (19, ON, 0.33)):
        if prm is PRM and blk in stalled:
            gaps = [(k0, n)]
            logs[blk] = D.add(blk, prm, dm, 0.6, vk=C.sim_vk(30.0, 1.0 + dm, C.neutral_of(gaps)), stall=C.gaps_to_frames(gaps))
        else:
            logs[blk] = D.add(blk, prm, dm, 0.6 if prm is PRM else 0.3)
    return D, logs


def case_e(lines):
    for tag, k0 in (('e', 69), ('e2', 48)):
        lines.append('# %s OFF の本のカメラの止まり(フレーム %d = %.2f 秒のあと 8 枚が無い = 間 0.300 秒)。本数を変える' % (tag, k0, k0 / 30.0))
        for stalled in ((), (12,), (12, 14), (12, 14, 16), (14, 16), (12, 16)):
            D, logs = eday(C.fresh('%s_%s' % (tag, '_'.join(map(str, stalled)) or 'none')), stalled, k0)
            lines.append(' 止まった OFF の本: %s' % (', '.join('3-%d' % (b - 11) for b in stalled) or 'なし'))
            res = both(lines, D, '%s%s' % (tag, ''.join(map(str, stalled))), [logs[b] for b in stalled])
            for ver in ('r1old', 'r1new'):
                j, _ = res[ver]
                offs = sorted(r['to_r1_s'] for r in j['launches'] if r.get('launch_state') == 'off')
                lines.append('      %s: OFF の本の起点まで %s' % (ver, offs))
        lines.append('')


def main():
    lines = []
    try:
        case_a(lines)
        case_b(lines)
        case_c(lines)
        case_d(lines)
        case_e(lines)
    finally:
        C.cleanup()
    C.write('p2_overlap.txt', lines)


if __name__ == '__main__':
    main()
