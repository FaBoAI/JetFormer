"""反証 8-2 回目・係 tools: other_group_on(w1-tools-r8-1-trial-form の直し)が、数えてはいけない本を数えないか・数えるべき本を
数えるか。回 3(tests/test_fix8_tools_r1.tfday と同じ: OFF 4・ON 3(200 µs)= −0.165 秒「使ってよい」)に足す本を変える。

  1  別の回(回 4 のブロック。bench の 4-1)の、形の違う ON の試しの本の最初の周の 6G
  2  別の回(回 4)の、計画どおりの形の ON のふつうの本(試しでない)の最初の周の 6G
  3  別の日(11/04。--day 20261103,20261104)の回 3 の、形の違う ON の試しの本の 6G
  4  同じ回・同じ形・ゲインだけ違う(0.92)ON のふつうの本(試しでない)の 6G。--ignore-gain なし / あり
  5  同じ回・形の違う(天井 ON)ON のふつうの本(試しでない)の 6G(form_split のほかに other_group_on を付けないか)
  6  同じ回・形の違う OFF の試しの本の 6G(数えない)
  7  同じ回・形の違う ON の試しの本の 2 回目の発進(人が止めて戻した)の 6G。引数なし / --first-only
  8  同じ回・形の違う ON の試しの本(launch_boost 1 なのに出ない = on_not_started)の 6G
  9  同じ回・形の違う ON の走り直し(リカバリー中止のあと)で、R1 に届かない走り直しの 6G
  10 同じ回で、形の違う試しが 2 組(天井 ON・自動 ON)にあり、計画の組が 2 組(ゲインで分かれた。--ignore-gain なし)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p3_other_group.py
  → out/p3_other_group.txt
"""
import json

import common as C
from common import T, t7, t8, PRM, ON, IG

ON200 = dict(ON, launch_us=200.0)


def base_day(tmp, extra_blocks=None, day2=False):
    """tfday の 7 本(3-1〜3-7)。bench に回 4(block 20〜23 → 4-1〜4-4)・回 2(block 11 → 2-8)を足す"""
    D = t7.Day(tmp)
    b = json.loads(D.bench.read_text(encoding='utf-8'))
    b['days'][0]['blocks'] += [{'block': 11, 'run_id': '2-8'}] + [{'block': x, 'run_id': '4-%d' % (x - 19)} for x in range(20, 24)]
    if day2:
        b['days'].append({'date': '2026-11-04', 'blocks': [{'block': x, 'run_id': '3-%d' % (x - 11)} for x in range(12, 20)]})
    D.bench.write_text(json.dumps(b), encoding='utf-8')
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON200, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, ON200, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, ON200, 0.28, 0.3), (18, PRM, 0.23, 0.6)):
        D.add(blk, prm, dm, ramp)
    return D


def show(lines, D, tag, args_list=(('--ignore-gain', IG),), days=None):
    for an, args in args_list:
        for ver in ('r1old', 'r1new'):
            j, o, _ = run(D, ver, args, tag + ver + an, days)
            lines.append('  %s %s:' % (an, '直す前' if ver == 'r1old' else '今'))
            for g in j['summary']:
                lines.append('    %s' % C.gtext(g, ('other_group_on', 'on_faults_counted_in', 'form_split', 'gain_split',
                                                   'trial_excluded', 'trial_on_collisions', 'not_reached',
                                                   'abort_resume_excluded')))
            for x in C.lines_with(o, T.OTHER_HEAD, T.SRC_HEAD):
                lines.append('    行: %s' % x[:220])


def run(D, ver, args, tag, days=None):
    if not days:
        return C.runc(D, ver, args, tag)
    # --day を 2 日に(t8.runc は --day 20261103 の固定なので、同じ流し方で日だけ変える)
    import contextlib
    import io
    import sys
    pc = t8.VER[ver][0]
    m = t8.load_mod('c8r2_%s_%s' % (ver, abs(hash(tag)) % 100000), pc)
    m.RAW, m.LAB, m.BENCH, m.OUT = D.raw, D.tmp / 'nolab', D.bench, D.tmp / 'out'
    out = D.tmp / ('%s_%s.json' % (ver, abs(hash(tag)) % 100000))
    buf = io.StringIO()
    saved = sys.modules.get('standing_3lap')
    sys.modules['standing_3lap'] = t8.mods(ver)[0]
    try:
        with contextlib.redirect_stdout(buf):
            m.main(['--day', days, '--out', str(out)] + list(args))
    finally:
        if saved is not None:
            sys.modules['standing_3lap'] = saved
        else:
            sys.modules.pop('standing_3lap', None)
    return json.loads(out.read_text(encoding='utf-8')), buf.getvalue(), None


def add_trial(D, block, prm, **kw):
    lg = D.add(block, prm, 0.15, 0.3, hits=[(3.0, 6.0)], **kw)
    t8.make_kind(D.raw, lg, 'trial')
    return lg


def main():
    lines = []
    ceil_on = dict(ON200, speed_ceiling=1)
    try:
        lines.append('# 0 対照: 3-8 = 天井 ON の試しの本の 6G(数える = 8-1 の直し)')
        D = base_day(C.fresh('p0'))
        add_trial(D, 19, ceil_on)
        show(lines, D, 'p0')

        lines.append('# 1 回 4(4-1)の天井 ON の試しの本の 6G(別の回 = 数えない)')
        D = base_day(C.fresh('p1'))
        add_trial(D, 20, ceil_on)
        show(lines, D, 'p1')

        lines.append('# 2 回 4(4-1)の計画どおりの形の ON のふつうの本の 6G(別の回 = 数えない)')
        D = base_day(C.fresh('p2'))
        D.add(20, ON200, 0.15, 0.3, hits=[(3.0, 6.0)])
        show(lines, D, 'p2')

        lines.append('# 3 11/04 の回 3 の天井 ON の試しの本の 6G(別の日 = 数えない)。--day 20261103,20261104')
        D = base_day(C.fresh('p3'), day2=True)
        lg = D.add(19, ceil_on, 0.15, 0.3, hits=[(3.0, 6.0)])
        t8.make_kind(D.raw, lg, 'trial')
        new = lg.replace('20261103', '20261104')
        (D.raw / lg).rename(D.raw / new)
        for p in sorted((D.raw / new).glob(lg + '-*')):
            p.rename(p.parent / p.name.replace(lg, new))
        t8.edit_log(D.raw, new, lambda j: j.update(description=j['description'].replace('day=2026-11-03', 'day=2026-11-04')))
        show(lines, D, 'p3', days='20261103,20261104')

        lines.append('# 4 同じ回・同じ形・ゲイン 0.92 の ON のふつうの本の 6G(試しでない)。引数なし / --ignore-gain')
        D = base_day(C.fresh('p4'))
        D.add(19, dict(ON200, throttle_gain=0.92), 0.15, 0.3, hits=[(3.0, 6.0)])
        show(lines, D, 'p4', (('--day', []), ('--ignore-gain', IG)))

        lines.append('# 5 同じ回・天井 ON の ON のふつうの本の 6G(試しでない = form_split で満たさない。other_group_on は付けない)')
        D = base_day(C.fresh('p5'))
        D.add(19, ceil_on, 0.15, 0.3, hits=[(3.0, 6.0)])
        show(lines, D, 'p5')

        lines.append('# 6 同じ回・天井 ON の OFF の試しの本の 6G(条件は ON だけ = 数えない)')
        D = base_day(C.fresh('p6'))
        add_trial(D, 19, dict(PRM, speed_ceiling=1))
        show(lines, D, 'p6')

        lines.append('# 7 同じ回・天井 ON の ON の試しの本の 2 回目の発進(1 回目のあと人側 → 置いて AI 側)の 6G。--ignore-gain / --ignore-gain --first-only')
        D = base_day(C.fresh('p7'))
        dur = 40.0
        vk = t7.launch_keys(1.0, 0.15, 0.3) + [(8.0, 2.0), (9.0, 0.0), (21.15, 0.0), (21.45, 2.0), (dur, 2.0)]
        lg = D.add(19, ceil_on, 0.15, 0.3, dur=dur, vk=vk, ai=[(1.0, 8.5), (21.0, dur - 2.0)],
                   launch=[(1.0, 0.5), (21.0, 0.5)], hits=[(23.0, 6.0)])
        t8.make_kind(D.raw, lg, 'trial')
        show(lines, D, 'p7', (('--ignore-gain', IG), ('--ignore-gain --first-only', C.IGF)))

        lines.append('# 8 同じ回・天井 ON の ON の試しの本(launch_boost 1 なのに出ない = on_not_started)の 6G')
        D = base_day(C.fresh('p8'))
        add_trial(D, 19, ceil_on, launch=[])
        show(lines, D, 'p8')

        lines.append('# 9 同じ回・ゲイン 0.92 の ON の本の走り直し(中止のあと)が R1 に届かず 6G(--ignore-gain なし)')
        D, lg, tt = t8.rday(C.fresh('p9'), 3.5, b38=dict(ON, throttle_gain=0.92), resume=True,
                            ai=lambda tt: [(1.0, tt['t_flip']), (tt['t_res'], tt['t_res'] + 2.0)],
                            hits=lambda tt: [(tt['t_res'] + 1.5, 6.0)])
        show(lines, D, 'p9', (('--day', []), ('--ignore-gain', IG)))

        lines.append('# 10 計画の組がゲインで 2 組(0.90・0.92。--ignore-gain なし)+ 天井 ON の試し・自動 ON の試し(どちらも 6G)')
        D = t7.Day(C.fresh('p10'))
        for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON200, 0.15, 0.3), (14, PRM, 0.21, 0.6), (15, ON200, 0.20, 0.3),
                                   (16, dict(PRM, throttle_gain=0.92), 0.22, 0.6), (17, dict(ON200, throttle_gain=0.92), 0.28, 0.3),
                                   (18, dict(PRM, throttle_gain=0.92), 0.23, 0.6)):
            D.add(blk, prm, dm, ramp)
        add_trial(D, 19, ceil_on)
        lg2 = D.add(19, dict(ON200, speed_auto=1), 0.15, 0.3, hits=[(3.5, 6.0)])
        t8.make_kind(D.raw, lg2, 'trial')
        show(lines, D, 'p10', (('--day', []),))
    finally:
        C.cleanup()
    C.write('p3_other_group.txt', lines)


if __name__ == '__main__':
    main()
