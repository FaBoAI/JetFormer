"""fix8 係 tools 直す 8-1 回目の台本(読むだけ。作り物の記録は ../../tmp_tools_r1/ の下に作り、終わりに消す)。

指摘 2 件の形を、⑤ collect.py の直す前の写し(tests/fixtures/fix8_tools_r1_before/collect.py.orig)と今の版で流して並べる。
作り物の日の作り方は tests/test_fix8_tools_r1.py(tfday・sday)と同じ。

  T1 w1-tools-r8-1-trial-form: 回 3(OFF 4・ON 3 = −0.165 秒「使ってよい」)の 3-8 を試し / 途中で変えた本の ON にし、最初の周に
     空転・いちばんひどい空転・比が出ない・6G の衝突・IMU の欠け、R1 に届かない ON の衝突。3-8 の形は計画どおり / 天井 / 左 / 自動 /
     目標 / ゲイン(--ignore-gain なし)
  T2 w1-tools-r8-1-stall-inflates-off: 回 3 の 3-5 の発進の 1.6〜2.5 秒にカメラが止まる(N 止まりなし・A 両方 ON で中止・B 3-5 から
     わざと OFF・C 両方 ON で止まらない・H 中立止めの行)× 3-5 の発進の前進 OFF / ON。止まりの長さ 0.2〜0.9 秒(B)も

  cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
      /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/fix_tools_r1/probes/probe_r1.py [old|new|both]
  → ../out/probe_r1_<版>.txt
"""
import json
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix8_tools_r1 as T  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'out'
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r1/probe'
VERS = {'old': 'r1old', 'new': 'r1new'}


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
    extra = {k: g[k] for k in ('trial_excluded', 'midrun_excluded', 'trial_on_collisions', 'midrun_on_collisions', 'trial_slip',
                               'midrun_slip', 'trial_slip_unmeasured', 'midrun_slip_unmeasured', 'first_lap_gap_on',
                               'not_reached', 'form_split', 'gain_split', 'other_group_on', 'launch_stall_excluded',
                               'launch_stall_on', 'on_faults_counted_in', 'launch_gap_short', 'mixed_options') if g.get(k)}
    if 'not_reached' in extra:
        extra['not_reached'] = {k: {'n': v['n'], 'logs': v['logs']} for k, v in extra['not_reached'].items()}
    return ('[回 %s・g %s・天井 %s(%s)・自動 %s(%s)] ON n %s(%s)・OFF n %s(%s)・ON − OFF %s・%s・False %s・%s' % (
        g['session'], g['gain'], g['speed_ceiling'], g['ceil_left_mps'], g['speed_auto'], g['auto_target_mps'], on.get('n'),
        on.get('to_r1_median_s'), off.get('n'), off.get('to_r1_median_s'), c.get('d_to_r1_s'), T.verdict(g),
        sorted(k for k, v in (c.get('checks') or {}).items() if v is not True), json.dumps(extra, ensure_ascii=False)))


def run_t1(ver, lines):
    lines.append('# T1 形の違う試し・途中で変えた本(3-8)の ON の不具合')
    for kind in ('trial', 'midrun'):
        for shape in ('spin', 'extreme', 'unk', 'hit', 'gap', 'noreach_hit', 'nospin'):
            for form in T.FORMS:
                D, log = T.tfday(fresh('t1_%s_%s_%s' % (kind, shape, form)), kind, shape, form)
                args = [] if form == 'gain' else T.IG
                j, out, _ = T.runc(D, VERS[ver], args, 't1%s%s%s' % (kind[:2], shape[:3], form[:3]))
                g = T.plan_group(j)
                lines.append('== %s・%s・形 %s(%s)→ 計画の組: %s' % (kind, shape, form, ' '.join(args) or '引数なし', T.verdict(g)))
                for gg in j['summary']:
                    lines.append('   ' + gtext(gg))
                lines += ['   | ' + x for x in out.splitlines() if x.startswith('  注意: 同じ回・同じモデルの別の組')
                          or x.startswith('  → この組の比べから外した本')]
                shutil.rmtree(D.tmp)


def run_t2(ver, lines):
    lines.append('# T2 発進 → R1 のカメラの止まり(3-5)')
    for boost in (False, True):
        for case in ('N', 'A', 'B', 'C', 'H'):
            D, log35 = T.sday(fresh('t2_%s_%d' % (case, boost)), case, boost)
            j, out, _ = T.runc(D, VERS[ver], T.IG, 't2%s%d' % (case, boost))
            r35 = [r for r in j['launches'] if r['log'] == log35][0]
            g = T.g0(j)
            lines.append('== %s・3-5 は発進の前進 %s → %s' % (case, 'ON' if boost else 'OFF', T.verdict(g)))
            lines.append('   3-5 の発進の行: to_r1_s %s・launch_stall %s' % (r35.get('to_r1_s'), r35.get('launch_stall')))
            lines.append('   ' + gtext(g))
            lines += ['   | ' + x for x in out.splitlines() if x.startswith('  ON − OFF') or x.startswith('  発進から R1')
                      or x.startswith('  → ON の発進で、発進から R1') or x.startswith('  → OFF の発進が全部')
                      or 'カメラの止まり(フレームの間' in x]
            shutil.rmtree(D.tmp)
    lines.append('# T2b 止まり(フレームの無い間)の長さ(R: 両方 ON の日・本物に近い速さ(止まりの 0.1 秒あとから明けるまで中立の惰性 '
                 '−0.9 m/s²)・3-5 は発進の前進 OFF。1.6 秒から)。止まりなしの 3-5 は 3.033 秒(ON − OFF −0.135 秒「満たさない」)')
    for ln in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9):
        D, log35 = T.sday(fresh('t2b_%s' % ln), 'R', False, stall=(1.6, 1.6 + ln))
        j, out, _ = T.runc(D, VERS[ver], T.IG, 't2b%d' % int(ln * 100))
        r35 = [r for r in j['launches'] if r['log'] == log35][0]
        g = T.g0(j)
        off = g['by_launch']['off']
        lines.append('== R・止まり %.2f 秒: 3-5 起点まで %s(遅れ %+.3f)・launch_stall %s・OFF n %s 中央値 %s・ON − OFF %s → %s・注意 %s' % (
            ln, r35.get('to_r1_s'), (r35.get('to_r1_s') or 0) - 3.033, r35.get('launch_stall'), off['n'], off['to_r1_median_s'],
            g['compare']['d_to_r1_s'], T.verdict(g), g.get('launch_gap_short')))
        shutil.rmtree(D.tmp)


def main(argv):
    which = argv[0] if argv else 'both'
    vers = ('old', 'new') if which == 'both' else (which,)
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for ver in vers:
            lines = ['版: %s(%s)' % (ver, T.t8.VER[VERS[ver]][0])]
            run_t1(ver, lines)
            run_t2(ver, lines)
            txt = '\n'.join(lines) + '\n'
            (OUT / ('probe_r1_%s.txt' % ver)).write_text(txt, encoding='utf-8')
            print(txt)
    finally:
        if TMP.exists():
            shutil.rmtree(TMP)


if __name__ == '__main__':
    main(sys.argv[1:])
