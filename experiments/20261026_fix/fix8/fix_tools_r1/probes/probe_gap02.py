"""fix8 係 tools 直す 8-1 回目(読むだけ): しきい 0.22 秒のすぐ下(フレームの間 0.200 秒 = 6 フレーム分)の止まりが、境目の日
(tests/test_fix8_tools_r1.sday の R。止まりなしで −0.135 秒「満たさない」)の答えを変えるか。止まりの始めをずらして中立の長さを変える
(作り物の中立は「止まりの始め + 0.1 秒」から明けるまで)。出力 ../out/probe_gap02.txt

  PYTHONDONTWRITEBYTECODE=1 jf-heavy --lock none --mem 1500M -- python -B probe_gap02.py
"""
import shutil
import sys
from pathlib import Path

JF = Path('/home/jetson/jetformer')
for p in (JF, JF / 'tools', JF / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import test_fix8_tools_r1 as T  # noqa: E402

TMP = JF / 'experiments/20261026_fix/fix8/tmp_tools_r1/gap02'
OUT = Path(__file__).resolve().parent.parent / 'out' / 'probe_gap02.txt'
lines = []
try:
    for s0, s1 in ((1.601, 1.799), (1.6005, 1.7995), (1.61, 1.795), (1.63, 1.799), (1.401, 1.599), (1.801, 1.999), (2.201, 2.399)):
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)
        D, log35 = T.sday(TMP, 'R', stall=(s0, s1))
        for ver in ('r1old', 'r1new'):
            j, out, _ = T.runc(D, ver, T.IG, 'g02%s%d' % (ver, int(s0 * 1000)))
            r = [x for x in j['launches'] if x['log'] == log35][0]
            g = T.g0(j)
            lines.append('止まり %.4f〜%.4f 秒(作り物の中立 %.3f 秒)・%s: 3-5 起点まで %s(遅れ %+.3f)・launch_stall %s・OFF n %s・ON − OFF %s → %s・注意 %s' % (
                s0, s1, max(0.0, s1 - min(s0 + 0.1, s1)), ver, r['to_r1_s'], r['to_r1_s'] - 3.033, r.get('launch_stall'),
                g['by_launch']['off']['n'], g['compare']['d_to_r1_s'], T.verdict(g), g.get('launch_gap_short')))
finally:
    shutil.rmtree(TMP, ignore_errors=True)
OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
