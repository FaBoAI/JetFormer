"""fix8 係 tools-decided の確かめ(作り物の記録だけ。読むだけ): 決め ①(リカバリーの途中で人側 → その場で AI 側の走り直し)と
②(試し・途中で変えた本の ON の空転・最初の周の欠け)を、直す前の写し(tests/fixtures/fix8_tools_decided_before)と今で流す。
作り物は ../../tmp_tools_decided/probe/ に作る(終わったら消す)。出力は ../out/probe_decided_<tag>.txt

  jf-heavy --lock none --mem 1500M -- python3 probe_decided.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    sys.path.insert(0, str(p))
import test_fix7_tools as t7  # noqa: E402
import test_fix7_tools_r2 as t72  # noqa: E402
import test_fix8_tools_decided as t8  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_decided/probe'


def summ(j):
    out = []
    for g in j['summary']:
        c = g.get('compare') or {}
        on = g['by_launch'].get('on') or {}
        out.append('ON n %s・起点まで %s・ON − OFF %s・%s・checks %s・%s' % (
            on.get('n'), on.get('to_r1_median_s'), c.get('d_to_r1_s'), '使ってよい' if c.get('all_ok') else '満たさない',
            {k: v for k, v in (c.get('checks') or {}).items() if v is not True},
            {k: g[k] for k in ('abort_resume_excluded', 'midrun_slip', 'trial_slip', 'midrun_slip_unmeasured', 'trial_slip_unmeasured',
                               'first_lap_gap_on', 'trial_excluded', 'midrun_excluded') if g.get(k)}))
    return out


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    lines = []
    for stop in (0.5, 1.5, 2.5, 3.5, 4.5):
        D, log, tt = t8.rday(TMP / ('r%d' % int(stop * 10)), stop=stop)
        for ver in ('old', 'new'):
            j, o, _ = t8.runc(D, ver, t8.IG, 'p%d' % int(stop * 10))
            ab = [r.get('after_abort') for r in t7.rows_of(j, log)]
            R, md, qmd, runs, groups = t8.run_s3(D, ver)
            s2 = t72.st_of(R, log)
            g = t72.on250(groups)
            lines.append('① stop %.1f %s ⑤ %s 印 %s / ③ counted %s / ④ 止まってから 3 周 %s 数えない %s' % (
                stop, ver, summ(j), ab, [s['counted'] for s in s2], g.get('standing_list_s') or g.get('standing_best_s'),
                g.get('standing_excluded_n')))
    for kind in ('trial', 'midrun'):
        for shape in ('spin', 'nospin', 'gap', 'extreme', 'unk', 'off_spin'):
            D, log = t8.mtday(TMP / ('%s_%s' % (kind, shape)), kind, shape)
            for ver in ('old', 'new'):
                j, o, _ = t8.runc(D, ver, t8.IG, '%s%s' % (kind, shape))
                lines.append('② %s %s %s: %s' % (kind, shape, ver, summ(j)))
    txt = '\n'.join(lines) + '\n'
    (HERE.parent / 'out' / 'probe_decided.txt').write_text(txt, encoding='utf-8')
    print(txt)
    shutil.rmtree(TMP)


if __name__ == '__main__':
    main()
