"""反証 6-1・安全: s3 の直しのあと、車両の値の止まりが 'reverse' の下がり始め(ESC が後退に入った直後)にかかる場面。
値が古くなってから中立止めになるまで(max_sensor_age 150 ms)の間、Recovery は古い eRPM(< 300)のまま後退を出し続け、
車は下がり始める。止まりの前後のサンプルがどちらも「後ろへ動いていない」(止まりの前 = 下がり始める前・明け = 惰性で止まった)
なら、その間は「値が来なかっただけ」(hold_unseen_s)になり、下がり始める前の 'reverse' では数えない(backed_s = 0)。
→ 明けて t_check から rev_s をもう一度まるごと下がる。下がった距離を、値の止まりなし・keep 0・直す前の写しと比べる。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/probe_s3_reverse_start.py
出力: out/probe_s3_reverse_start.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402
from r6_common import A, C, N, PARAMS, T0, R, PH   # noqa: E402

XF = 2.0


def run(p, skip, mod=None, xr=None):
    del PH[:]
    sc = {'dur': 7.0, 'xf': XF, 'model': (C, N - 120.0), 'drops': [], 'veh_skip': [skip] if skip else []}
    if xr is not None:
        sc['xr'] = xr
    kw = {} if mod is None else {'mod': mod[0], 'amod': mod[1]}
    sim, rec, pca, d = R.run_closed(p, sc, wrap=K.wrap, **kw)
    ph = list(PH)
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows]
    t_hit = next((h[0] for h in sim.car.hits if h[1] == 'front'), None)
    tb = next((t for t, p0, p1, *_ in ph if p1 == 'brake'), None)
    tr = next((t for t, p0, p1, *_ in ph if p1 == 'reverse'), None)
    te = next((t for t, p0, p1, *_ in ph if p1 == 'escape' and t > (tb or 0)), None)
    tg = next((t for t, p0, p1, *_ in ph if p1 == 'giveup'), None)
    end = te if te is not None else (tg if tg is not None else (tb or 0) + 3.0)
    win = [x for t, x, *_ in sim.log if tb is not None and tb <= t <= end + 0.6]
    first_rev_on = next((t for t, x, v, thr, con, rv, sl in sim.log if tb is not None and t >= tb and rv), None)
    after = [x for x in ph if x[3]]
    return {'t_hit': t_hit, 't_brake': tb, 't_reverse': tr, 't_escape': te, 't_giveup': tg,
            't_first_rev_on': first_rev_on,
            'backed_m': round(XF - min(win), 3) if win else None,
            'rear_hits': [h for h in sim.car.hits if h[1] == 'rear'],
            'front_hits': [h for h in sim.car.hits if h[1] == 'front'][:4],
            'hold_rows': sum(1 for s, r in rows if r['hold']),
            'after_hold': after[:3],
            'phases': [(t, p0, p1, b1) for t, p0, p1, hs, hb, hu, b0, b1, e in ph if p0 != p1][:10],
            'events': [(e[1].get('what'), e[1].get('phase'), e[1].get('backed_s')) for e in rec.events
                       if e[0] == 'recovery_guard']}


def main():
    BEF = R.load_snapshot('fix5_safety_before')
    out = {}
    try:
        for rev_s in (0.4, 0.3):
            p1 = dict(PARAMS, recovery=1, start_kick=0, rec_back_guard=1, rec_hold_keep=1, tof_guard=1, rev_s=rev_s)
            base = run(p1, None)
            out['rev_s=%.1f no_gap keep1' % rev_s] = base
            out['rev_s=%.1f no_gap keep0' % rev_s] = run(dict(p1, rec_hold_keep=0), None)
            t0 = base['t_first_rev_on']       # 偽物の ESC が後退に入った時刻(模型の 10 ms 刻み)
            for d0 in (-0.06, -0.04, -0.02, 0.0, 0.02):
                for L in (0.35, 0.45, 0.5, 0.55, 0.6):
                    skip = (round(t0 + d0, 3), round(t0 + d0 + L, 3))
                    k = 'rev_s=%.1f veh_gap=rev_on%+.2f..+%.2f' % (rev_s, d0, L)
                    out[k + ' now_keep1'] = run(p1, skip)
                    out[k + ' now_keep0'] = run(dict(p1, rec_hold_keep=0), skip)
                    out[k + ' before_keep1'] = run(p1, skip, BEF)
    finally:
        R.cleanup()
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'probe_s3_reverse_start.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    for k, v in out.items():
        print(k, {kk: v[kk] for kk in ('t_reverse', 't_first_rev_on', 't_escape', 't_giveup', 'backed_m', 'hold_rows',
                                      'events')}, v['after_hold'][:1])


if __name__ == '__main__':
    main()
