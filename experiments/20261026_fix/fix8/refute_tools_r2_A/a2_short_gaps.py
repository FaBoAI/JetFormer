"""票 A・指摘 2(w2-tools-r8-2-short-gaps-add-up)を自分の作り物で確かめる。

1) 1 本の OFF の発進(3-5 = 起点まで 3.10 秒の本)に、短い間(0.22 秒以下)を続けて入れたとき、自分の速さの作り(drive の中立 =
   前のフレームの 0.1 秒あと〜次のフレームの間だけ惰性 −0.9 m/s²、ほかは 0.6 秒で 2 m/s の加速)で起点までがどれだけ伸びるか、
   ⑤(今)がその本を外すか・注意の行を出すか。
2) 回 3 の答え: 境目に近い日(止まりなしで −0.14 秒)と遠い日(−0.10 秒)で、OFF の 3-5(と 3-3)に入れたとき(今と直す前)。

    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python -B experiments/20261026_fix/fix8/refute_tools_r2_A/a2_short_gaps.py
  → out/a2_short_gaps.txt
"""
import common_a as C

NEAR = ((12, C.PRM, 0.18), (13, C.ON, 0.25), (14, C.PRM, 0.25), (15, C.ON, 0.27),
        (16, C.PRM, 0.30), (17, C.ON, 0.30), (18, C.PRM, 0.60), (19, C.ON, 0.32))      # −0.14
FAR = ((12, C.PRM, 0.18), (13, C.ON, 0.29), (14, C.PRM, 0.25), (15, C.ON, 0.31),
       (16, C.PRM, 0.30), (17, C.ON, 0.34), (18, C.PRM, 0.60), (19, C.ON, 0.36))      # −0.10

# (名前, 始めのフレームの時刻, 間のフレーム数, 回数)。30 fps: 4 = 0.133 秒(3 枚無い = 7.5 fps)・5 = 0.167・6 = 0.200・7 = 0.233
PATS = [('0.133×9 1.30 秒から', 1.30, 4, 9),
        ('0.133×9 2.30 秒から(2 m/s のあと)', 2.30, 4, 9),
        ('0.133 が R1 まで続く(7.5 fps)1.00 秒から', 1.00, 4, 23),
        ('0.100×12 1.30 秒から(10 fps)', 1.30, 3, 12),
        ('0.167×7 1.30 秒から', 1.30, 5, 7),
        ('0.200×4 1.30 秒から', 1.30, 6, 4),
        ('0.200×6 1.30 秒から', 1.30, 6, 6),
        ('0.200×1 1.30 秒から(対照: 注意の行)', 1.30, 6, 1),
        ('0.233×1 1.30 秒から(対照: 外す)', 1.30, 7, 1)]


def day(name, books, pat=None, targets=(16,)):
    D = C.Day(name)
    for blk, prm, dm in books:
        if pat is not None and blk in targets:
            st, neu, _ = C.gap_pattern(pat[1], pat[2], pat[3])
            D.add(blk, prm, dm, stall=st, neutral=neu)
        else:
            D.add(blk, prm, dm)
    return D


def main():
    C.check_sha()
    L = ['# 指摘 2: 短い間が続く OFF の発進(今 = d2b210cc / 直す前 = 26dfb087)。--ignore-gain']
    L.append('## 1) 3-5(OFF・止まりなしで起点まで 3.10 + 約 0.01 秒)だけ。境目に近い日(止まりなし −0.14)')
    try:
        base_vk = C.sim(30.0, C.T_AI + 0.30, C.ACC_OFF)
        t_base = C.time_to(base_vk, 5.0) - C.T_AI
        for i, pat in enumerate(PATS):
            st, neu, gaps = C.gap_pattern(pat[1], pat[2], pat[3])
            vk = C.sim(30.0, C.T_AI + 0.30, C.ACC_OFF, neu)
            dt = C.time_to(vk, 5.0) - C.T_AI - t_base
            neu_s = sum(b - a for a, b in neu if a < C.T_AI + t_base + dt)
            D = day('a2_%d' % i, NEAR, pat)
            res = []
            for ver in ('old', 'now'):
                j, o = C.run(D, ver, C.IG, 'a2_%d' % i)
                r35 = [r for r in j['launches'] if r['log'] == D.logs[16]][0]
                g = C.cmp_groups(j)[0]
                notes = [x.strip()[:40] for x in o.splitlines()
                         if x.startswith('  注意: 発進から R1 の起点までにフレームの間が') or x.startswith('  発進から R1 の起点までにカメラが止まった発進')]
                res.append('%s 起点まで %s・ON − OFF %s %s・行 %s%s' % (
                    '直す前' if ver == 'old' else '今', r35['to_r1_s'], g['compare']['d_to_r1_s'], C.verdict(g), notes or 'なし',
                    '・launch_stall %s' % r35['launch_stall'] if r35.get('launch_stall') else ''))
            L.append('  %s: 間 %s 秒 × %d・中立の合計(起点まで)%.3f 秒・作りで起点まで +%.3f 秒 | %s' % (
                pat[0], gaps[0], len(gaps), neu_s, dt, ' / '.join(res)))
        L.append('## 2) 回 3 の答え(OFF に入れる)')
        for bname, books in (('近い日(−0.14)', NEAR), ('遠い日(−0.10)', FAR)):
            for tg in ((16,), (14, 16)):
                for k, pat in enumerate(PATS):
                    if pat[0].startswith('0.233') or pat[0].startswith('0.100'):
                        continue
                    D = day('a2b_%s_%d_%d' % (bname[:2], len(tg), k), books, pat, tg)
                    out = []
                    for ver in ('old', 'now'):
                        j, o = C.run(D, ver, C.IG, 'a2b')
                        g = C.cmp_groups(j)[0]
                        notes = [x.strip()[:22] for x in o.splitlines()
                                 if x.startswith('  注意: 発進から R1 の起点までにフレームの間が')
                                 or x.startswith('  発進から R1 の起点までにカメラが止まった発進')]
                        offs = sorted(r['to_r1_s'] for r in j['launches'] if r.get('launch_state') == 'off')
                        out.append('%s %s %s 行 %s' % ('直す前' if ver == 'old' else '今', g['compare']['d_to_r1_s'], C.verdict(g),
                                                      notes or 'なし'))
                    L.append('  %s・%s・%s: OFF の起点まで %s | %s' % (bname, '・'.join('3-%d' % (b - 11) for b in tg), pat[0], offs,
                                                                ' / '.join(out)))
            D = day('a2b_%s_none' % bname[:2], books)
            j, o = C.run(D, 'now', C.IG, 'a2n')
            g = C.cmp_groups(j)[0]
            L.append('  %s・止まりなし: 今 %s %s' % (bname, g['compare']['d_to_r1_s'], C.verdict(g)))
    finally:
        C.cleanup()
    C.write('a2_short_gaps.txt', L)


if __name__ == '__main__':
    main()
