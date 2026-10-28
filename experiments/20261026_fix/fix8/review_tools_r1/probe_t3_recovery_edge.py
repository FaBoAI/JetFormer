"""反証 8-1 回目・係 tools・台本 3(読むだけ): 決め ①「リカバリーの途中で人が CH3 を取った区間のあとの走り直し」の印(kind 'recovery')の
境目 = assist の段の変わり目を突く。

tests/test_fix8_tools_decided.rday(回 3 の 7 本 + 3-8: 2 周目に R1 の stop m 手前で止まり、止まり始めの 0.8 秒あとから
リカバリー(assist 2)→ 人側 1.0 秒 → その場で AI 側)の、リカバリーの終わりだけを動かす:
  k フレーム = リカバリーが人側にする k フレーム前(k / 30 秒前)に終わり、そのあとは assist 0(モデルのまま。車は止まったまま =
  リカバリーの前進(escape)が前の物で止まり、モデルの前進が壁を押している形 = 手順書 7 章「当たって止まった車が、AI 側のまま前へ押し
  続けている → すぐ CH3 を人側」)。k = 0 は決めの形(最後のフレームがリカバリー)。
⑤ は --ignore-gain、③④ は作り物の日で。出力 out/probe_t3_recovery_edge.txt

  jf-heavy --lock none --mem 1500M -- python -B probe_t3_recovery_edge.py
"""
import common as C
from common import t8


def main():
    lines = []
    for stop in (3.5, 4.5):
        for k in (0, 1, 2, 9, 30):
            D, log, tt = t8.rday(C.fresh('t3_%d_%d' % (int(stop * 10), k)), stop,
                                 rec=lambda tt, k=k: (tt['t_st'] + 0.8, tt['t_flip'] - k / 30.0 + 1e-4))
            j, out, _ = C.runc(D, 'new', C.IG, 't3%d%d' % (int(stop * 10), k))
            g = j['summary'][0]
            rows = [r for r in j['launches'] if r['log'] == log]
            R, md, qmd, runs, groups = t8.run_s3(D, 'new')
            s2 = t8.st_of(R, log)
            g250 = t8.on250(groups)
            lines.append('stop %.1f m・リカバリーが人側の %d フレーム前に終わる: ⑤ %s ON n %s・ON − OFF %s・印 %s / ③ counted %s / '
                         '④ 発進 250 の組の止まってから 3 周 %s(数えない %s)' % (
                             stop, k, C.verdict(g), g['by_launch']['on']['n'], g['compare']['d_to_r1_s'],
                             [r.get('after_abort') for r in rows], [s['counted'] for s in s2],
                             g250.get('standing_list_s') or g250.get('standing_best_s'), g250.get('standing_excluded_n')))
    C.write('probe_t3_recovery_edge.txt', lines)
    C.cleanup()


if __name__ == '__main__':
    main()
