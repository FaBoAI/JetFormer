"""反証 r2-r2-rs-1 の打ち消し(直さない): refute2_safety/closed_loop.py の閉じたループで
(1) 指摘の場面の再現 (2) 中立止めの始まりの時刻を細かく動かし、keep=1 で中立止めなしより多く下がる窓の幅
(3) 中立止めなしで rev_us を 350(やり直しの強さ)にしたときの下がる距離(設計のやり直しの範囲)を見る。
装置は開かない。書くのは標準出力だけ(tmp は fix3/tmp_r2r2rs1 に作って消す)"""
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'refute2_safety'))
import closed_loop as CL          # noqa: E402
from closed_loop import C, N, PARAMS   # noqa: E402

CL.TMP = HERE.parent / 'tmp_r2r2rs1'
XF = 2.0


def one(keep, guard=1, drop=None, vskip=None, rev_us=300.0, xr=None):
    sc_ = {'dur': 7.0, 'xf': XF, 'xr': xr, 'model': (C, N - 120.0), 'drops': [drop] if drop else [],
           'veh_skip': [vskip] if vskip else []}
    p = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=guard, rec_hold_keep=keep, rev_us=rev_us)
    sim, rec, pca = CL.run(p, sc_)
    ph = [(t, p0, p1) for t, p0, p1, hg, ig in CL.PH if p0 != p1]
    t_esc = next((t for t, p0, p1 in ph if p1 == 'escape'), 99)
    xs = [(t, x, v) for t, x, v, *_ in sim.log if 2.0 <= t <= t_esc + 0.5]
    retries = sum(1 for t, p0, p1 in ph if p0 == 'reverse' and p1 == 'gap' and t <= t_esc)
    revmax = max((th for t, s, th in pca.w if 2.0 <= t <= t_esc), default=None)
    return {'back': round(XF - min(x for _, x, _ in xs), 3), 'vmax': round(min(v for _, _, v in xs), 2),
            'retries': retries, 'rev_max': revmax, 'rear': [h for h in sim.car.hits if h[1] == 'rear'],
            'ph': ph[:10]}


def main():
    t0 = time.time()
    base = one(1)
    tr = next(t for t, p0, p1 in base['ph'] if p1 == 'reverse')
    print('base keep=1', base['back'], base['vmax'], base['ph'][:6], 'reverse_at', tr, '%.1fs' % (time.time() - t0))
    for ru in (350.0, 400.0):
        r = one(1, rev_us=ru)
        print('nohold rev_us=%d' % ru, r['back'], r['vmax'])
    # 指摘の場面の再現
    for o, L in ((0.03, 0.45), (0.06, 0.6), (0.10, 1.0)):
        for keep in (0, 1):
            r = one(keep, drop=(tr + o, tr + o + L), xr=XF - 1.45)
            print('repro cam o=%.2f L=%.2f keep=%d' % (o, L, keep), r['back'], r['vmax'], 'retries', r['retries'],
                  r['rev_max'], r['rear'])
    # 窓の幅(後ろの物なし)
    rows = []
    for L in (0.2, 0.3, 0.45, 0.6, 1.0):
        for i in range(-10, 41, 2):
            o = i / 100.0
            r = one(1, drop=(tr + o, tr + o + L))
            rows.append((L, o, r['back'], r['vmax'], r['retries']))
    print('sweep (L, o, back, vmax, retries) keep=1:')
    for L in (0.2, 0.3, 0.45, 0.6, 1.0):
        sel = [x for x in rows if x[0] == L]
        over = [x[1] for x in sel if x[2] > base['back'] + 0.02]
        print(' L=%.2f over_base_o=%s max_back=%.3f' % (L, over, max(x[2] for x in sel)))
        print('   ', [(x[1], x[2], x[4]) for x in sel])
    print('elapsed %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.rmtree(CL.TMP, ignore_errors=True)
