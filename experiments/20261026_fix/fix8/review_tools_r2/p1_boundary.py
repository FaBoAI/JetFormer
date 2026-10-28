"""反証 8-2 回目・係 tools: launch_stall の 0.22 秒の境目(0.200・0.233・0.267 秒)と中立止めの行の除き方、短い止まりの重なり。

  A  launch_stall だけ(本物と同じ µs の時刻・30 fps。ずれ ±1 ms も): フレームの無い間・中立止めの行・両方が並ぶ形
  B  作り物の回 3(tests/test_fix8_tools_r1.sday と同じ 8 本。3-5 = 発進の前進 OFF。止まりが無ければ ON − OFF −0.135 秒「満たさない」)の
     3-5 の発進の 1.40・1.60 秒からに、フレームの間 0.200・0.233・0.267 秒(30 fps の 6・7・8 フレーム分)の止まりを、フレームの無い形と
     中立止めの行(hold = True)の形で。速さは本物に近い形(止まりの 0.1 秒あとから中立の惰性 −0.9 m/s²)
  C  短い止まり(フレームの間 0.200 秒 = 外さない)が 1 本の発進の中で何度も起きる形(カメラの落ちが続く)

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/review_tools_r2/p1_boundary.py
  → out/p1_boundary.txt
"""
import numpy as np

import common as C
from common import T, t7, PRM, ON, IG


def unit(lines):
    m = C.load_new()
    lines.append('# A launch_stall(z, i0, i1, t0, t_end)。フレームは本物と同じ µs の整数(33333/33334 µs)。発進 1.0 秒・起点 4.0 秒')
    us = np.round(np.arange(0, 180) * 1e6 / 30.0).astype(np.int64)
    rng = np.random.default_rng(1)

    def z_of(drop=(), hold=(), jitter=0):
        t_us = us + (rng.integers(-jitter, jitter + 1, len(us)) if jitter else 0)
        t = (t_us - t_us[0]) / 1e6
        keep = np.ones(len(t), bool)
        h = np.zeros(len(t), bool)
        for k, n in drop:
            keep[k + 1:k + 1 + n] = False
        for k, n in hold:
            h[k + 1:k + 1 + n] = True
        return {'t_s': t[keep], 'hold': h[keep]}

    cases = []
    for n in (3, 4, 5, 6, 7):
        cases.append(('フレームが %d 枚無い(間 %.3f)' % (n, (n + 1) / 30.0), dict(drop=[(60, n)])))
        cases.append(('中立止めの行が %d 枚(間 %.3f)' % (n, (n + 1) / 30.0), dict(hold=[(60, n)])))
    cases += [('無い 3 枚 + 中立止め 3 枚が続く(間 0.233)', dict(drop=[(60, 3)], hold=[(63, 3)])),
              ('中立止め 2 枚 + 無い 3 枚(間 0.200)', dict(hold=[(60, 2)], drop=[(62, 3)])),
              ('発進のフレームから中立止め 7 枚(1.0 秒から。間 0.233)', dict(hold=[(29, 7)])),
              ('起点の直前 6 枚無い(3.767〜。間 0.233)', dict(drop=[(112, 6)]))]
    for jit in (0, 1000):
        for name, kw in cases:
            zz = z_of(jitter=jit, **kw)
            st = m.launch_stall(zz, 0, len(zz['t_s']) - 1, float(zz['t_s'][30]), 4.0)
            nt = m.launch_stall(zz, 0, len(zz['t_s']) - 1, float(zz['t_s'][30]), 4.0, over=m.LAUNCH_GAP_NOTE_S)
            lines.append('  ずれ ±%d µs・%s → launch_stall %s・注意(0.15 秒より長い)%s' % (jit, name, st, nt))
    lines.append('')


def bday(tmp, gaps, mode='stall', boost35=False):
    """sday('R')と同じ 8 本。3-5 の発進(1.0 秒・1.22 秒に動き出す)に gaps [(k, n)] = フレーム k のあと n 枚を、mode 'stall' なら
    抜く(フレームが無い)・'hold' なら中立止めの行にする。速さは drive の中立の間(前のフレームの 0.1 秒あと〜次のフレーム)だけ惰性"""
    D = t7.Day(tmp)
    D.add(12, PRM, 0.20, 0.6)
    D.add(13, ON, 0.15, 0.3)
    D.add(14, PRM, 0.23, 0.6)
    D.add(15, ON, 0.20, 0.3)
    dur = 30.0
    vk = C.sim_vk(dur, 1.22, C.neutral_of(gaps))
    iv = C.gaps_to_frames(gaps)
    kw = {'stall': iv} if mode == 'stall' else {'hold': iv}
    log35 = D.add(16, ON if boost35 else PRM, 0.22, 0.6, dur=dur, vk=vk, ai=[(1.0, dur - 2.0)], **kw)
    D.add(17, ON, 0.28, 0.3)
    D.add(18, PRM, 0.44, 0.6)
    D.add(19, ON, 0.31, 0.3)
    return D, log35


def one(lines, tag, gaps, mode='stall', boost35=False, args=IG):
    D, log35 = bday(C.fresh(tag), gaps, mode, boost35)
    out = {}
    for ver in ('r1old', 'r1new'):
        j, o, _ = C.runc(D, ver, args, tag + ver)
        g = T.g0(j)
        r = T.rows_of(j, log35)[0]
        out[ver] = (g, r, o)
    gn, rn, on_ = out['r1new']
    go, ro, _ = out['r1old']
    lines.append('  %s: 3-5 起点まで %.3f(止まりなし 3.033)・launch_stall %s・launch_gap_short %s' % (
        tag, rn['to_r1_s'], rn.get('launch_stall'), gn.get('launch_gap_short')))
    lines.append('    直す前 ON − OFF %s %s → 今 OFF n %s・ON − OFF %s %s・False %s' % (
        go['compare']['d_to_r1_s'], C.verdict(go), gn['by_launch'].get('off', {}).get('n'), gn['compare']['d_to_r1_s'],
        C.verdict(gn), sorted(k for k, v in gn['compare']['checks'].items() if not v)))
    for x in C.lines_with(on_, T.GAPNOTE_HEAD):
        lines.append('    注意の行: %s' % x[:160])
    return gn, rn


def e2e(lines):
    lines.append('# B 回 3 の 3-5(OFF)の発進に止まり。フレームの無い形(stall)と中立止めの行(hold)')
    lines.append('  止まりなし:')
    one(lines, 'none', [])
    for k0, at in ((42, '1.40'), (48, '1.60')):
        for n, gap in ((5, '0.200'), (6, '0.233'), (7, '0.267')):
            for mode in ('stall', 'hold'):
                one(lines, 'b%d_%d_%s' % (k0, n, mode), [(k0, n)], mode)
    lines.append('')


def burst(lines):
    lines.append('# C 短い止まり(フレームの間 0.200 秒 = 5 枚無い)が 1 本の発進の中で続く形(3-5 = OFF)')
    for name, gaps in (('0.200 × 1(1.40 秒から)', [(42, 5)]),
                       ('0.200 × 2(1.40・1.70 秒から)', [(42, 5), (51, 5)]),
                       ('0.200 × 3(1.40・1.70・2.00 秒から)', [(42, 5), (51, 5), (60, 5)]),
                       ('0.200 × 4(1.30〜2.20 秒。6 枚に 1 枚 = 5 fps が 1.2 秒)', [(39, 5), (45, 5), (51, 5), (57, 5)]),
                       ('0.200 × 6(1.30〜2.50 秒。5 fps が 1.8 秒)', [(39 + 6 * i, 5) for i in range(6)]),
                       ('0.167 × 6(4 枚無い。1.30 秒から 6 fps)', [(39 + 5 * i, 4) for i in range(6)]),
                       ('対照: 0.233 × 1(1.60 秒から。外す)', [(48, 6)])):
        for mode in ('stall', 'hold'):
            gn, rn = one(lines, 'c_%s_%s' % (abs(hash(name)) % 10000, mode), gaps, mode)
            neu = sum(b - a for a, b in C.neutral_of(gaps))
            lines.append('    ↑ %s・%s・drive の中立の合計 %.3f 秒' % (name, mode, neu))
    # ON の本で同じ形(ON は遅くなる側 = 比べが厳しくなる。参考)
    lines.append('  参考: 3-5 を ON にして 0.200 × 4(遅い ON が比べに残る = 厳しい側)')
    one(lines, 'c_on_burst', [(39, 5), (45, 5), (51, 5), (57, 5)], 'stall', boost35=True)
    lines.append('')


def main():
    lines = []
    try:
        unit(lines)
        e2e(lines)
        burst(lines)
    finally:
        C.cleanup()
    C.write('p1_boundary.txt', lines)


if __name__ == '__main__':
    main()
