"""反証 8-2 回目・係 tools: 短い止まり(フレームの間 0.22 秒以下 = launch_stall の外)が 1 本の発進の中で続く形(カメラの落ちが続く・
fps が下がる)。回 3 は sday と同じ 8 本(止まりが無ければ ON − OFF −0.135 秒「満たさない」)。OFF の 3-5(と 3-7)の発進の 1.30 秒から。

  間 0.133 秒(3 枚無い = 7.5 fps)× 9 / 間 0.167 秒(4 枚無い = 6 fps)× 7 / 間 0.200 秒(5 枚無い = 5 fps)× 6、それぞれ 1 本・2 本。
  速さは drive の中立の間(前のフレームの 0.1 秒あと〜次のフレーム)だけ惰性 −0.9 m/s²(p1 と同じ)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p8_burst.py
  → out/p8_burst.txt
"""
import common as C
from common import T, t7, PRM, ON, IG


def day(tmp, gaps, books):
    D = t7.Day(tmp)
    for blk, prm, dm, ramp in ((12, PRM, 0.20, 0.6), (13, ON, 0.15, 0.3), (14, PRM, 0.23, 0.6), (15, ON, 0.20, 0.3),
                               (16, PRM, 0.22, 0.6), (17, ON, 0.28, 0.3), (18, PRM, 0.44, 0.6), (19, ON, 0.31, 0.3)):
        if blk in books:
            D.add(blk, prm, dm, ramp, vk=C.sim_vk(30.0, 1.0 + dm, C.neutral_of(gaps)), stall=C.gaps_to_frames(gaps))
        else:
            D.add(blk, prm, dm, ramp)
    return D


def main():
    lines = ['# 短い止まりが続く OFF の発進(1.30 秒から)']
    try:
        for name, n_miss, cnt in (('0.133 × 9', 3, 9), ('0.167 × 7', 4, 7), ('0.200 × 6', 5, 6)):
            gaps = [(39 + (n_miss + 1) * i, n_miss) for i in range(cnt)]
            neu = sum(b - a for a, b in C.neutral_of(gaps))
            for books in ((16,), (16, 18)):
                D = day(C.fresh('b%d%d' % (n_miss, len(books))), gaps, books)
                for ver in ('r1old', 'r1new'):
                    j, o, _ = C.runc(D, ver, IG, 'b%s%d%d' % (ver, n_miss, len(books)))
                    g = T.g0(j)
                    offs = sorted(r['to_r1_s'] for r in j['launches'] if r.get('launch_state') == 'off')
                    flagged = [x[:60] for x in C.lines_with(o, T.GAPNOTE_HEAD, T.STALL_HEAD)]
                    lines.append('  %s・%s(中立の合計 %.3f 秒/本)%s: OFF の起点まで %s → ON − OFF %s %s・印の行 %s' % (
                        name, '・'.join('3-%d' % (b - 11) for b in books), neu, '直す前' if ver == 'r1old' else '今    ', offs,
                        g['compare']['d_to_r1_s'], C.verdict(g), flagged or 'なし'))
        lines.append('  (止まりが無ければ OFF 3.013・3.033・3.043・3.253 → −0.135 満たさない)')
    finally:
        C.cleanup()
    C.write('p8_burst.txt', lines)


if __name__ == '__main__':
    main()
