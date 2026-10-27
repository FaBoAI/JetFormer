"""反証 6-1・安全: probe_s3_reverse_start の 'brake' 版。モデルが壁の手前でブレーキ(後退側)を出して止まり(ESC は
ブレーキ → 中立で「後退できる」状態)、止まったまま 0.8 秒でリカバリーが始まると、'brake' の後退側で ESC はすぐ下がり始める
(後ろの見張りの back_early の場面)。その入りぎわに車両の値が 0.45〜0.6 秒来ないとき、下がった距離を値の止まりなし・keep 0・
直す前の写しと比べる。閉じたループ(本物の drive._loop)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/probe_s3_brake_ready.py
出力: out/probe_s3_brake_ready.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402
from r6_common import A, C, N, PARAMS, T0, R, PH   # noqa: E402

XF = 2.4


def model(t, car):
    return (C, N - 120.0) if car.x < XF - 0.45 and t < 6.0 else (C, N + 200.0)


def run(p, skip, mod=None, xr=None):
    del PH[:]
    sc = {'dur': 7.0, 'xf': XF, 'model': model, 'drops': [], 'veh_skip': [skip] if skip else []}
    if xr is not None:
        sc['xr'] = xr
    kw = {} if mod is None else {'mod': mod[0], 'amod': mod[1]}
    sim, rec, pca, d = R.run_closed(p, sc, wrap=K.wrap, **kw)
    ph = list(PH)
    tb = next((t for t, p0, p1, *_ in ph if p1 == 'brake'), None)
    x_b = next((x for t, x, *_ in sim.log if tb is not None and t >= tb), None)
    win = [x for t, x, *_ in sim.log if tb is not None and t >= tb]
    first_rev_on = next((t for t, x, v, thr, con, rv, sl in sim.log if tb is not None and t >= tb and rv), None)
    return {'t_brake': tb, 'x_at_brake': x_b, 't_first_rev_on': first_rev_on,
            'backed_m': round(x_b - min(win), 3) if win else None,
            'hits': sim.car.hits[:4],
            'after_hold': [x for x in ph if x[3]][:3],
            'phases': [(t, p0, p1, b1) for t, p0, p1, hs, hb, hu, b0, b1, e in ph if p0 != p1][:10],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events
                       if e[0] == 'recovery_guard']}


def main():
    BEF = R.load_snapshot('fix5_safety_before')
    out = {}
    try:
        p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=0.4, rev_us=300.0)
        base = run(p1, None)
        out['no_gap keep1'] = base
        out['no_gap keep0'] = run(dict(p1, rec_hold_keep=0), None)
        print('base', base, flush=True)
        t0 = base['t_first_rev_on']
        if t0 is None:
            return
        for d0 in (-0.10, -0.06, -0.04, -0.02, 0.0):
            for L in (0.45, 0.52, 0.6):
                skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                k = 'gap=rev_on%+.2f..+%.2f' % (d0, L)
                out[k + ' now_keep1'] = run(p1, skip)
                out[k + ' now_keep0'] = run(dict(p1, rec_hold_keep=0), skip)
                out[k + ' before_keep1'] = run(p1, skip, BEF)
                print(k, {nm: (out[k + ' ' + nm]['backed_m'], out[k + ' ' + nm]['events'][:2],
                               out[k + ' ' + nm]['after_hold'][:1]) for nm in ('now_keep1', 'now_keep0', 'before_keep1')},
                      flush=True)
    finally:
        R.cleanup()
        (HERE / 'out').mkdir(exist_ok=True)
        (HERE / 'out' / 'probe_s3_brake_ready.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
