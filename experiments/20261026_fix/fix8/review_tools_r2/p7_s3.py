"""反証 8-2 回目・係 tools: ⑤ が外す「発進 → R1 のカメラの止まり」の発進を、③ standing_3lap(止まってから 3 周)・④ quick_eval_by_params
はどう数えるか(③④ は 8-1 で変えていない)。3-5 の本を 45 秒にして 3 周を走らせる(ほかの 7 本は sday と同じ)。

  N 止まりなし / S フレームが 1.6〜2.5 秒に無い(速さは同じ)/ H 同じ間が中立止めの行(フレームはある)/
  g200 フレームの間 0.200 秒(1.60 秒から 5 枚無い)/ g233 0.233 秒(6 枚無い)/ h233 6 枚が中立止めの行

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p7_s3.py
  → out/p7_s3.txt
"""
import common as C
from common import T, t7, t8, PRM, ON, IG


def day(tmp, case):
    D = t7.Day(tmp)
    for blk, prm, dm in ((12, PRM, 0.20), (13, ON, 0.15), (14, PRM, 0.23), (15, ON, 0.20)):
        D.add(blk, prm, dm, 0.6 if prm is PRM else 0.3)
    dur = 45.0
    kw = {}
    vk = None
    if case == 'S':
        kw = {'stall': [(1.6, 2.5)]}
    elif case == 'H':
        kw = {'hold': [(1.6, 2.5)]}
    elif case in ('g200', 'g233', 'h233'):
        gaps = [(48, 5 if case == 'g200' else 6)]
        vk = C.sim_vk(dur, 1.22, C.neutral_of(gaps))
        kw = {'hold' if case == 'h233' else 'stall': C.gaps_to_frames(gaps)}
    lg = D.add(16, PRM, 0.22, 0.6, dur=dur, ai=[(1.0, dur - 1.0)], **({'vk': vk} if vk else {}), **kw)
    for blk, prm, dm in ((17, ON, 0.28), (18, PRM, 0.44), (19, ON, 0.31)):
        D.add(blk, prm, dm, 0.6 if prm is PRM else 0.3)
    return D, lg


def main():
    lines = ['# ③ standing_3lap の 3-5 の発進(止まってから 3 周)と ⑤ の印']
    try:
        for case in ('N', 'S', 'H', 'g200', 'g233', 'h233'):
            D, lg = day(C.fresh('s3' + case), case)
            R, md, qmd, runs, groups = t8.run_s3(D, 'r1new')
            st = [x for x in R if x['log'] == lg][0]['standing']
            j, o, _ = C.runc(D, 'r1new', IG, 's3c' + case)
            r = T.rows_of(j, lg)[0]
            for s in st:
                lines.append('  %-5s ③: t_ai %s・起点まで %s・止まってから 3 周 %s 秒・valid %s・counted %s・why %s | ⑤: launch_stall %s・'
                             '比べに入る %s' % (case, s.get('t_ai'), s.get('to_r1_s'), s.get('from_ai_s'), s.get('valid'),
                                                s.get('counted'), s.get('why'), r.get('launch_stall'),
                                                not r.get('launch_stall')))
            q = [x for x in qmd.splitlines() if '止まってから' in x][:3]
            for x in q:
                lines.append('        ④: %s' % x[:200])
    finally:
        C.cleanup()
    C.write('p7_s3.txt', lines)


if __name__ == '__main__':
    main()
