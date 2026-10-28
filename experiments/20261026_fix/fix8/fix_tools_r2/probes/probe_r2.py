"""fix8 係 tools 直す 8-2 回目: 反証 8-2 回目 係 tools の 2 件が起きるのを、直す前(tests/fixtures/fix8_tools_r2_before/collect.py.orig)と
今の ⑤ で確かめる(作り物の日は tests/test_fix8_tools_r2.py の fday・gday。⑤ は手順書どおり --day 20261103 --ignore-gain)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/fix_tools_r2/probes/probe_r2.py old|new|both
  → ../out/probe_r2_<old|new>.txt(作り物は fix8/tmp_tools_r2/probe_<..> に作って消す)
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix8_tools_r2 as T  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r2'


def fresh(name):
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def gtext(g):
    c = g.get('compare') or {}
    on = (g.get('by_launch') or {}).get('on') or {}
    off = (g.get('by_launch') or {}).get('off') or {}
    ex = {k: g[k] for k in ('off_dropped', 'launch_stall_excluded', 'launch_stall_sum', 'launch_sum_short', 'launch_gap_short')
          if g.get(k)}
    if g.get('not_reached'):
        ex['not_reached'] = {k: v['logs'] for k, v in g['not_reached'].items()}
    return 'ON n %s(%s)・OFF n %s(%s)・ON − OFF %s・%s・False %s・%s' % (
        on.get('n'), on.get('to_r1_median_s'), off.get('n'), off.get('to_r1_median_s'), c.get('d_to_r1_s'), T.verdict(g),
        sorted(k for k, v in (c.get('checks') or {}).items() if v is not True), json.dumps(ex, ensure_ascii=False))


NEW_HEADS = (T.OFFDROP_HEAD, T.SUM_HEAD, T.SUMNOTE_HEAD, T.T1.STALL_HEAD, T.T1.GAPNOTE_HEAD, T.T1.STALL_ON_HEAD,
             '  → OFF の発進が全部カメラの止まりで比べから外れた')


def run(D, ver, args, tag, show_lines=False):
    j, o, _ = T.runc(D, ver, args, tag)
    g = T.g0(j)
    out = ['    %s' % gtext(g)]
    if show_lines:
        for x in o.splitlines():
            if x.startswith(T.COMPARE_HEAD) or any(x.startswith(h) for h in NEW_HEADS) or ' カメラの止まり' in x:
                out.append('      行: %s' % x[:420])
    return out, j


def main(which):
    vers = {'old': ['r2old'], 'new': ['r2new'], 'both': ['r2old', 'r2new']}[which]
    L = []
    try:
        # ---------------------------------------------------------------- 1 w2-tools-r8-2-off-few-left
        L.append('# 1 w2-tools-r8-2-off-few-left: 票 A の形の日(止まりが無ければ ON − OFF −0.145「満たさない」。OFF 3-1 2.963・3-3 3.053・'
                 '3-5 3.103・3-7 3.403)')
        cases = (('なし', {}), ('late 3-1(いちばん速い OFF に 2 m/s のあとの 0.3 秒)', {12: 'late'}),
                 ('late 3-7(いちばん遅い OFF)', {18: 'late'}), ('late 3-1・3-3', {12: 'late', 14: 'late'}),
                 ('early 3-1(発進の始めの 0.3 秒)', {12: 'early'}), ('A 3-1(hold_long = R1 に届かない)', {12: 'A'}),
                 ('sum 3-1(0.133 秒 × 9 = 中立の合計 0.3 秒)', {12: 'sum'}),
                 ('blip 3-1(0.5 秒の AI 側 → 5 秒にもう一度走った = 本は比べに残る)', {12: 'blip'}),
                 ('tail 3-1(走ったあとの 2 回目の 0.5 秒の AI 側 = R1 に届かない 2 回目)', {12: 'tail'}),
                 ('A の OFF 4 本(OFF が全部 R1 に届かない)', {12: 'A', 14: 'A', 16: 'A', 18: 'A'}))
        for name, kinds in cases:
            D, logs = T.fday(fresh('f%d' % len(L)), kinds)
            L.append('  %s' % name)
            for ver in vers:
                for args in ((T.IG, T.IGF) if 'blip' in name or 'tail' in name else (T.IG,)):
                    out, _ = run(D, ver, args, 'f%s%d%d' % (ver, len(L), len(args)), show_lines=(ver == 'r2new'))
                    L.append('   %s %s' % ('直す前' if ver == 'r2old' else '今    ', ' '.join(args)))
                    L += out
        D, logs = T.fday(fresh('redo'), {12: 'A'})
        D.add(12, T.PRM, 0.15, 0.6)                       # 同じ block 12 で別の記録にやり直し(起点まで 2.963 秒)
        L.append('  A 3-1 + 同じ block 12 で別の記録にやり直し(回の計画の本は比べに残る)')
        for ver in vers:
            out, _ = run(D, ver, T.IG, 'redo%s' % ver, show_lines=(ver == 'r2new'))
            L.append('   %s --ignore-gain' % ('直す前' if ver == 'r2old' else '今    '))
            L += out
        L.append('')
        L.append('# 1b 反証の係の日(review_tools_r2/p6_offcount.py と同じ数。止まりが無ければ −0.10「満たさない」)')
        for name, kinds in (('late 3-1・3-3', {12: 'late', 14: 'late'}), ('A 3-1・3-3', {12: 'A', 14: 'A'}),
                            ('mix 3-1 late・3-3 A', {12: 'late', 14: 'A'}), ('late 3-1・3-3・3-5', {12: 'late', 14: 'late', 16: 'late'}),
                            ('early 3-1・3-5', {12: 'early', 16: 'early'})):
            D, logs = T.fday(fresh('p%d' % len(L)), kinds, T.PBOOKS)
            L.append('  %s' % name)
            for ver in vers:
                out, _ = run(D, ver, T.IG, 'p%s%d' % (ver, len(L)))
                L.append('   %s' % ('直す前' if ver == 'r2old' else '今    '))
                L += out
        L.append('')
        # ---------------------------------------------------------------- 2 w2-tools-r8-2-short-gaps-add-up
        L.append('# 2 w2-tools-r8-2-short-gaps-add-up: 8-1 の sday と同じ 8 本(止まりが無ければ 3-5 3.033・ON − OFF −0.135「満たさない」)')
        L.append('   (中立 = drive の中立の作り物の合計。速さは中立の間だけ惰性 −0.9 m/s²)')
        g2 = (('0.133 × 3(1.30 秒から)', {16: T.burst(39, 3, 3)}, False), ('0.133 × 4', {16: T.burst(39, 3, 4)}, False),
              ('0.133 × 6', {16: T.burst(39, 3, 6)}, False), ('0.133 × 7', {16: T.burst(39, 3, 7)}, False),
              ('0.133 × 9(反証の係の形)', {16: T.burst(39, 3, 9)}, False),
              ('0.167 × 7', {16: T.burst(39, 4, 7)}, False), ('0.200 × 2', {16: T.burst(39, 5, 2)}, False),
              ('0.200 × 3', {16: T.burst(39, 5, 3)}, False),
              ('0.200 × 4(1.90 秒から)3-5・3-7(反証の係の S8)', {16: T.burst(57, 5, 4), 18: T.burst(57, 5, 4)}, False),
              ('中立止めの行 0.133 × 4(行 3 枚 × 4)', {16: T.burst(39, 3, 4)}, True),
              ('中立止めの行 0.067 × 6(行 1 枚 × 6)', {16: T.burst(39, 1, 6)}, True))
        for name, gaps, hold in g2:
            D, logs = T.gday(fresh('g%d' % len(L)), gaps, hold)
            ns = T.neutral_sum(gaps[16], hold)
            L.append('  %s: 3-5 の中立の作り物の合計 %.3f 秒' % (name, ns))
            for ver in vers:
                out, j = run(D, ver, T.IG, 'g%s%d' % (ver, len(L)), show_lines=(ver == 'r2new'))
                r = T.rows_of(j, logs[16])[0]
                L.append('   %s 3-5 の起点まで %.3f(+%.3f)・launch_stall %s' % ('直す前' if ver == 'r2old' else '今    ', r['to_r1_s'],
                                                                         r['to_r1_s'] - 3.033, r.get('launch_stall')))
                L += out
        D, logs = T.gday(fresh('gon'), {16: T.burst(39, 3, 9)}, on35=True)
        L.append('  3-5 を ON にして 0.133 × 9')
        for ver in vers:
            out, _ = run(D, ver, T.IG, 'gon%s' % ver, show_lines=(ver == 'r2new'))
            L.append('   %s' % ('直す前' if ver == 'r2old' else '今    '))
            L += out
    finally:
        if TMP.exists():
            shutil.rmtree(TMP)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ('probe_r2_%s.txt' % which)).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'both')
