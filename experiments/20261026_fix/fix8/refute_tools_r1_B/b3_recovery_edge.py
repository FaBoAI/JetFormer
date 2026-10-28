"""反証 8-1 回目・打ち消しの票 B・指摘 w1-tools-r8-1-recovery-just-ended(読むだけ。コード・文書は変えない)。

review_tools_r1/probe_t3_recovery_edge.py と同じ作り物(tests/test_fix8_tools_decided.rday)で、今の版の答えを流し直す(k = 0・1)。
足したもの: 止まった場所が R1 の 5.0 m 手前(= 回 3 の本を置く HS と同じ距離。tests/test_fix7_tools_r2.kday の「置いた場所と同じ HS」)
= 手順書 7 章「人側で HS のいつもの場所に戻してから」AI 側にした走り直しと同じ起点までの道のり、のときの答え。
作り物の置き場は ../tmp_refute_tools_B(終わって消す)。出力 out/b3_recovery_edge.txt

  PYTHONDONTWRITEBYTECODE=1 jf-heavy --lock none --mem 1500M -- python -B b3_recovery_edge.py
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'review_tools_r1'))
import common as C  # noqa: E402
from common import t8  # noqa: E402

C.TMP = HERE.parent / 'tmp_refute_tools_B'


def main():
    lines = []
    for stop in (3.5, 5.0):
        for k in (0, 1):
            D, log, tt = t8.rday(C.fresh('b3_%d_%d' % (int(stop * 10), k)), stop,
                                 rec=lambda tt, k=k: (tt['t_st'] + 0.8, tt['t_flip'] - k / 30.0 + 1e-4))
            j, out, _ = C.runc(D, 'new', C.IG, 'b3%d%d' % (int(stop * 10), k))
            g = j['summary'][0]
            rows = [r for r in j['launches'] if r['log'] == log]
            lines.append('止まった場所 R1 の %.1f m 手前・リカバリーが人側の %d フレーム前に終わる: ⑤ %s・ON n %s・ON − OFF %s・'
                         '3-8 の発進の起点まで %s・印 %s' % (
                             stop, k, C.verdict(g), g['by_launch']['on']['n'], g['compare']['d_to_r1_s'],
                             [r.get('to_r1_s') for r in rows], [(r.get('after_abort') or {}).get('kind') for r in rows]))
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out' / 'b3_recovery_edge.txt').write_text(txt, encoding='utf-8')
    print(txt)
    if C.TMP.exists():
        shutil.rmtree(C.TMP)


if __name__ == '__main__':
    main()
