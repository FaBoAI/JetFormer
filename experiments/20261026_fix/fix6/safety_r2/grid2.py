"""fix6 係 safety 直す 6-2 回目: 指摘の形と、その近くを広く振る(読むだけ)。どの場面も既定の rev_s 0.4・rev_us 300・keep 1・見張り 1。
グリッド:
  fr_<rev|brake>[_<esc>]  指摘 1 の形(反証の grid_false_ready と同じ 480 組): 最初のリカバリーの 'brake' の始め tb から a0 秒に
                          長さ aL のカメラの止まり A、その終わりから b0 秒に長さ bL の車両の値の止まり B
  rt_<neu>               指摘 2 の形(反証の probe_retry_est と同じ 81 組): 中立 neu 秒で下がれる偽物の ESC。'reverse' の始めから
                          d0 秒に長さ L の車両の値の止まり
  w61_<rev|brake>         fix6 r1 の 170 組(偽物の ESC が後退に入る時刻から d0 秒に長さ L の車両の値の止まり。r1 の指摘の形)
esc: orig(元の偽物 = R2.Car)・fake(中立 0.1 秒・0.08 秒。1e-9 の余裕)・slow(中立 0.12 秒・0.10 秒)・n15(中立 0.15 秒・0.08 秒)
比べるもの: now(fix6 r1)・before(fix5 のあと)・cand の候補(VARIANTS)・k0(now の keep 0)。止まりなし(ng)も 1 回。
    python -I grid2.py <grid> [part/parts]     → out/grid2_<grid>[_part].json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402
from test_launch_drive import N   # noqa: E402

ESC = {'orig': None, 'fake': (0.1, 0.08), 'slow': (0.12, 0.10), 'n15': (0.15, 0.08)}
VARIANTS = ['now', 'before', 'cand:so,oc,dr', 'cand:so,oc,dr,dv', 'cand:so,oc,dr,rn12', 'cand:so,oc,dr,dv,rn12']
if os.environ.get('R2_VARIANTS'):          # 2 回目の組: 候補だけ(before・k0 は 1 回目の出力から。analyze2 が場面の鍵で合わせる)
    VARIANTS = os.environ['R2_VARIANTS'].split('|')
PREFIX = os.environ.get('R2_OUT', 'grid2')
T_ON = {'rev': 2.60, 'brake': 2.73}


def cells(grid, ng_res, p):
    kind = grid.split('_')[1] if not grid.startswith('rt_') else 'rev'
    if grid.startswith('fr_'):
        tb = E.metrics(ng_res, p)['t_brake']
        for a0 in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2):
            for aL in (0.13, 0.15, 0.17, 0.19, 0.21):
                for b0 in (0.0, 0.03, 0.06, 0.1):
                    for bL in (0.17, 0.2, 0.25, 0.3):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        yield 'A=%s B=%s' % (a, b), E.scene(kind, skips=[b], drops=[a])
    elif grid.startswith('rt_'):
        sim, rec, pca = ng_res
        tb = E.metrics(ng_res, p)['t_brake']
        tg = next(t for t, s, th in pca.w if t > tb + 0.2 and abs(th - (N + 10)) < 1e-6)
        tr = next(t for t, s, th in pca.w if t > tg and abs(th - (N + 300)) < 1e-6)
        for d0 in (-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2):
            for L in (0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45):
                b = (round(tr + d0, 3), round(tr + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), E.scene('rev', skips=[b])
    else:
        t0 = T_ON[kind]
        for d0 in (-0.2, -0.15, -0.1, -0.06, -0.04, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.45):
            for L in (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
                b = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), E.scene(kind, skips=[b])


def esc_of(grid):
    if grid.startswith('rt_'):
        return (float(grid.split('_')[1]), 0.08)
    parts = grid.split('_')
    return ESC[parts[2]] if len(parts) > 2 else None


def main(grid, part='0/1'):
    pi, pn = (int(x) for x in part.split('/'))
    esc = esc_of(grid)
    p = E.P1()
    kind = 'rev' if grid.startswith('rt_') else grid.split('_')[1]
    out = {'grid': grid, 'esc': esc, 'variants': VARIANTS, 'cells': {}}
    try:
        ng = E.run('now', p, E.scene(kind), esc=esc)
        out['ng'] = E.metrics(ng, p)
        if PREFIX == 'grid2':
            out['ng_k0'] = E.metrics(E.run('now', E.P1(rec_hold_keep=0), E.scene(kind), esc=esc), p)
        for i, (key, sc) in enumerate(cells(grid, ng, p)):
            if i % pn != pi:
                continue
            row = {}
            for v in VARIANTS:
                name, fl = (v.split(':', 1) + [''])[:2] if ':' in v else (v, '')
                row[v] = E.metrics(E.run(name, p, sc, esc=esc, flags=fl), p)
            if PREFIX == 'grid2':
                row['k0'] = E.metrics(E.run('now', E.P1(rec_hold_keep=0), sc, esc=esc), p)
            out['cells'][key] = row
            if (i // pn) % 30 == 0:
                print(grid, part, i, key, {v: (m['back'], m['push']) if m else None for v, m in row.items()},
                      flush=True)
    finally:
        E.cleanup()
    suf = '' if pn == 1 else '_%d' % pi
    (HERE / 'out' / ('%s_%s%s.json' % (PREFIX, grid, suf))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print('done', grid, part, len(out['cells']))


if __name__ == '__main__':
    main(*sys.argv[1:3])
