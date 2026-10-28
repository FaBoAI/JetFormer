"""fix8 係 safety-proposal: fix7/safety_r2/g9.py の写し(h9 → h10 だけ変えた)。
反証 7-1 回目・safety: 狙った場面のグリッドで ON(rec_back_guard 1・rec_hold_keep 1)と OFF(両方 0)を同じ場面で並べる。
どれも今の版(jetformer/)・回 1〜4 の形の基本(rev_s 0.4・rev_us 300・start_kick 1・ToF ガード 1)。onc = ON の止まりなし。
場面(fix6 r1・r2 の exp と同じ形): rev = 前の壁 2.0 m・モデルは前進 120 µs、brake = 前の壁 2.4 m・壁の 0.45 m 手前からブレーキ側
(中立 + 200。止まると ESC は後退を待っている)。rear = 前の壁から後ろの物までの m(None = なし)。
グリッド:
  fr_<kind>      fix6 r2 の fr(480 組): 'brake' の始めから a0 秒に aL のカメラの止まり、その b0 秒あとに bL の車両の値の止まり
  w61_<kind>     fix6 r1 の 170 組: 偽物の ESC が後退に入る時刻から d0 秒に L 秒の車両の値の止まり
  sv_<kind>      0.15 秒より短い車両の値の止まり(中立止めにならない): 後退に入る時刻から d0 秒に L 秒(0.04〜0.14)
  cs_<kind>      'coast'(惰性)の頃の長いカメラの止まり: 後退側を出し終える時刻から d0 秒に L 秒(0.3〜1.5)
    python -I grid_onoff.py <grid> <rear|None> [esc(neu,rdelay) か orig] [part/parts]   → out/grid_<grid>_r<rear>[_esc][_part].json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h10 as rv7  # noqa: E402

C, N, A = rv7.C, rv7.N, rv7.A
XF = {'rev': 2.0, 'brake': 2.4}
T_ON = {'rev': 2.60, 'brake': 2.73}      # 止まりなしで偽物の ESC が後退に入る時刻(fix6 r1)


def model_brake(t, car):
    return (C, N - 120.0) if car.x < 2.4 - 0.45 and t < 6.0 else (C, N + 200.0)


def scene(kind, skips=(), drops=(), rear=None, dur=7.0, esc=None):
    xf, model = (2.0, (C, N - 120.0)) if kind == 'rev' else (2.4, model_brake)
    sc = {'dur': dur, 'xf': xf, 'model': model, 'drops': list(drops), 'veh_skip': list(skips)}
    if rear is not None:
        sc['xr'] = xf - rear
    if esc is not None:
        sc['esc'] = esc
    if os.environ.get('RV7_VEHDT'):              # 車両の値の間隔(本物は 100 Hz = 0.01)
        sc['veh_dt'] = float(os.environ['RV7_VEHDT'])
    return sc


def first_rec(res, t_from=1.5):
    sim, rec, pca, ph = res
    tb = next((t for t, s, th in pca.w if t > t_from and abs(th - (N + 300.0)) < 1e-6), None)
    m = rv7.metrics(sim, rec, pca, t_from=t_from)
    out = {'t_brake': tb, 'rear': m['rear'], 'rear_v': m['rear_max_v'], 'front_push': m['front_push_s'],
           'rear_push': m['rear_push_s'], 'giveup': m['giveup_t'], 'guards': m['guards'][:6], 'rev_dead': m['rev_dead_s']}
    if tb is None:
        out.update(back=None, esc_push=None, t_escape=None)
        return out
    esc = N - A.Recovery.ESCAPE_US
    te = next((t for t, s, th in pca.w if t > tb and abs(th - esc) < 1e-6), None)
    end = (te + 0.6) if te is not None else tb + 3.0
    x_b = next(x for t, x, *_ in sim.log if t >= tb)
    xs = [x for t, x, *_ in sim.log if tb <= t <= end]
    push = sum(rv7.DT for t, x, v, th, *_ in sim.log if te is not None and te <= t < te + 0.6
               and x >= sim.s['xf'] - 1e-9 and abs(th - esc) < 1e-6)
    out.update(back=round(x_b - min(xs), 3), esc_push=round(push, 2), t_escape=te,
               rear_first=[h for h in sim.car.hits if h[1] == 'rear' and tb <= h[0] <= end])
    return out


def cells(grid, rear, esc, ng):
    kind = grid.split('_')[1]
    if grid.startswith('fr_'):
        tb = ng['t_brake']
        for a0 in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2):
            for aL in (0.13, 0.15, 0.17, 0.19, 0.21):
                for b0 in (0.0, 0.03, 0.06, 0.1):
                    for bL in (0.17, 0.2, 0.25, 0.3):
                        a = (round(tb + a0, 3), round(tb + a0 + aL, 3))
                        b = (round(a[1] + b0, 3), round(a[1] + b0 + bL, 3))
                        yield 'A=%s B=%s' % (a, b), scene(kind, skips=[b], drops=[a], rear=rear, esc=esc)
    elif grid.startswith('w61_'):
        t0 = T_ON[kind]
        for d0 in (-0.2, -0.15, -0.1, -0.06, -0.04, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.45):
            for L in (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
                b = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), scene(kind, skips=[b], rear=rear, esc=esc)
    elif grid.startswith('sv_'):
        t0 = T_ON[kind]
        for i in range(31):
            d0 = round(-0.3 + 0.03 * i, 3)
            for L in (0.04, 0.06, 0.08, 0.1, 0.12, 0.14):
                b = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), scene(kind, skips=[b], rear=rear, esc=esc)
    elif grid.startswith('rt_'):
        tr = ng['t_rev_start']
        for d0 in (-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2):
            for L in (0.16, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45):
                b = (round(tr + d0, 3), round(tr + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), scene(kind, skips=[b], rear=rear, esc=esc)
    elif grid.startswith('cs_'):
        te = ng['t_rev_end']
        for i in range(21):
            d0 = round(-0.3 + 0.04 * i, 3)
            for L in (0.3, 0.45, 0.55, 0.7, 1.0, 1.5):
                a = (round(te + d0, 3), round(te + d0 + L, 3))
                yield 'd0=%+.2f L=%.2f' % (d0, L), scene(kind, drops=[a], rear=rear, esc=esc)


def main(grid, rear='None', esc='orig', part='0/1'):
    rv7.check_now()
    pi, pn = (int(x) for x in part.split('/'))
    rear = None if rear == 'None' else float(rear)
    esc_t = None if esc == 'orig' else tuple(float(x) for x in esc.split(','))
    kind = grid.split('_')[1]
    p = rv7.base_params()
    out = {'grid': grid, 'rear': rear, 'esc': esc_t, 'cells': {}}
    try:
        ngr = rv7.run('now', rv7.ON(p), scene(kind, rear=rear, esc=esc_t))
        ng = first_rec(ngr)
        pca = ngr[2]
        tr = [t for t, s, th in pca.w if t > ng['t_brake'] + 0.3 and th > N + 40]
        ng['t_rev_end'] = max(tr) if tr else ng['t_brake'] + 1.0
        tg = next((t for t, s, th in pca.w if t > ng['t_brake'] + 0.2 and abs(th - (N + 10)) < 1e-6), None)
        ng['t_rev_start'] = next((t for t, s, th in pca.w if tg is not None and t > tg and abs(th - (N + 300)) < 1e-6), None)
        out['ng_on'] = ng
        out['ng_off'] = first_rec(rv7.run('now', rv7.OFF(p), scene(kind, rear=rear, esc=esc_t)))
        for i, (key, sc) in enumerate(cells(grid, rear, esc_t, ng)):
            if i % pn != pi:
                continue
            row = {'on': first_rec(rv7.run('now', rv7.ON(p), sc)), 'off': first_rec(rv7.run('now', rv7.OFF(p), sc))}
            out['cells'][key] = row
            if (i // pn) % 40 == 0:
                print(grid, rear, i, key, {k: (m['back'], m['rear_v'], m['esc_push'], m['front_push']) for k, m in row.items()},
                      flush=True)
    finally:
        rv7.cleanup()
    suf = ('' if esc == 'orig' else '_e' + esc.replace(',', '-')) + ('' if pn == 1 else '_%d' % pi)
    if os.environ.get('RV7_VEHDT'):
        suf += '_v' + os.environ['RV7_VEHDT']
    (HERE / 'out' / ('grid_%s_r%s%s.json' % (grid, rear, suf))).write_text(json.dumps(out, ensure_ascii=False, default=str))
    print('done', grid, rear, esc, part, len(out['cells']))


if __name__ == '__main__':
    main(*sys.argv[1:5])
