"""反証 8-2 回目・係 tools: 8-1 の直しの場面を、⑤ の 4 通りの引数(--day だけ・--ignore-gain・--first-only・両方)で流し、答え(使ってよい /
満たさない・False の条件)が同じかを並べる。直す前(r1old)も並べる。

  S1 3-5(OFF)の発進の 1.6〜2.5 秒にカメラの止まり(sday B)          S2 同じ(3-5 = ON)
  S3 天井 ON の試しの本(3-8)の 6G(tfday)                             S4 ゲイン 0.92 の試しの本の空転(tfday gain)
  S5 OFF の 2 本(3-1・3-3)に 2 m/s に届いたあとのカメラの止まり(p2 の e)  S6 ON の 4 本が全部カメラの止まり(p2 の c)
  S7 ゲイン 0.92 の走り直しの 6G(t8.rday)                               S8 3-5 と 3-7 の 2 本に 0.200 秒 × 4 の短い止まり(p1 の C)
  S9 OFF の本(3-5)の 2 回目の発進にカメラの止まり

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p4_args.py
  → out/p4_args.txt
"""
import common as C
from common import T, t7, t8, PRM, ON
import p1_boundary as P1
import p2_overlap as P2


def plan_groups(j):
    return [g for g in j['summary'] if 'compare' in g]


def row_of(j):
    out = []
    for g in plan_groups(j):
        c = g['compare']
        out.append('%s %s %s False%s' % ('g%s' % g['gain'] if g['gain'] is not None else 'g*', C.verdict(g), c['d_to_r1_s'],
                                       sorted(k for k, v in c['checks'].items() if not v)))
    return ' / '.join(out) or '比べなし'


def s9(tmp):
    """OFF の 3-5 の本: 1 回目の発進(1.0 秒)→ 8.5 秒に人側で止める → 21.0 秒に置いた所から AI 側(2 回目の発進)の 21.6〜22.5 秒に
    カメラの止まり"""
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.23, 0.6), (15, ON, 0.20, 0.3)):
        D.add(blk, prm, dm, ramp)
    dur = 40.0
    vk = t7.launch_keys(1.0, 0.22, 0.6) + [(8.0, 2.0), (9.0, 0.0), (21.22, 0.0), (21.6, 1.3), (22.5, 0.5), (23.2, 2.0), (dur, 2.0)]
    D.add(16, PRM, 0.22, 0.6, dur=dur, vk=vk, ai=[(1.0, 8.5), (21.0, dur - 2.0)], stall=[(21.6, 22.5)])
    for blk, prm, dm, ramp in ((17, ON, 0.28, 0.3), (18, PRM, 0.44, 0.6), (19, ON, 0.31, 0.3)):
        D.add(blk, prm, dm, ramp)
    return D


def s8(tmp):
    D = t7.Day(tmp)
    burst = [(39, 5), (45, 5), (51, 5), (57, 5)]
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.23, 0.6), (15, ON, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, PRM, 0.44, 0.6), (19, ON, 0.31, 0.3)):
        if blk in (16, 18):
            D.add(blk, prm, dm, ramp, vk=C.sim_vk(30.0, 1.0 + dm, C.neutral_of(burst)), stall=C.gaps_to_frames(burst))
        else:
            D.add(blk, prm, dm, ramp)
    return D


def main():
    lines = []
    scen = [
        ('S1 3-5 OFF のカメラの止まり', lambda d: T.sday(d, 'B')[0]),
        ('S2 3-5 ON のカメラの止まり', lambda d: T.sday(d, 'B', boost35=True)[0]),
        ('S3 天井 ON の試しの 6G', lambda d: T.tfday(d, 'trial', 'hit', 'ceil')[0]),
        ('S4 ゲイン 0.92 の試しの空転', lambda d: T.tfday(d, 'trial', 'spin', 'gain')[0]),
        ('S5 OFF 2 本の 2 m/s のあとの止まり', lambda d: P2.eday(d, (12, 14))[0]),
        ('S6 ON 4 本が全部止まり', None),
        ('S7 ゲイン 0.92 の走り直しの 6G', lambda d: t8.rday(d, 3.5, b38=dict(ON, throttle_gain=0.92),
                                                     hits=lambda tt: [(tt['t_res'] + 2.0, 6.0)])[0]),
        ('S8 OFF 2 本に 0.200 秒 × 4', s8),
        ('S9 OFF の 2 回目の発進のカメラの止まり', s9),
    ]
    try:
        for name, mk in scen:
            d = C.fresh('s%d' % len(lines))
            if mk is None:
                D = t7.Day(d)
                for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.15), (14, PRM, 0.23), (15, ON, 0.20), (16, PRM, 0.22),
                                     (17, ON, 0.28), (18, PRM, 0.44), (19, ON, 0.31)):
                    D.add(blk, prm, dm, 0.3 if prm is ON else 0.6, **({'stall': [(1.6, 2.5)]} if prm is ON else {}))
            else:
                D = mk(d)
            lines.append('# %s' % name)
            for an, args in C.ARGS:
                cells = []
                for ver in ('r1old', 'r1new'):
                    j, o, _ = C.runc(D, ver, args, '%s%s%d' % (name[:2], ver, len(an)))
                    cells.append('%s: %s' % ('直す前' if ver == 'r1old' else '今', row_of(j)))
                lines.append('  %-28s %s' % (an, '  |  '.join(cells)))
    finally:
        C.cleanup()
    C.write('p4_args.txt', lines)


if __name__ == '__main__':
    main()
