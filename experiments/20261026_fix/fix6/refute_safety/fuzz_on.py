"""反証 6-1・安全: ON(rec_hold_keep = 1・rec_back_guard = 1)の でたらめな閉じたループを、fix5 と別の種で回す。
場面・スイッチは fix4/review_safety/fuzz_closed.py の rand_scene・rand_params(読み込むだけ。launch_boost・launch_no_kick・
start_kick・tof_guard・速さの天井・rev_s・rev_us は でたらめ)。決まりは fuzz_closed の K1〜K7・K9 と fix5 の R1・R2・R4・R5・R7
(同じ中身をここに写した。fix5 の台本は読み込まない = 一時の場所が fix5 に向くため)。
足した見方 X1: 車両の値の止まりのある場面で、止まりを抜いた同じ場面(now1_nogap)より 0.15 m 以上多く下がった
(最初のリカバリーの区間)・後ろに当たった。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \
        experiments/20261026_fix/fix6/refute_safety/fuzz_on.py [本数] [種]
出力: out/fuzz_on_<種>.json
"""
import collections
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_common as K          # noqa: E402
from r6_common import R, CL, A, C, N, T0   # noqa: E402
import fuzz_closed as FC       # noqa: E402


def extra_rules(sim, rec, pca, p):
    bad = []
    for t, s_, th in pca.w:
        if not (math.isfinite(s_) and math.isfinite(th) and 1000.0 <= s_ <= 2000.0 and 1000.0 <= th <= 2000.0):
            bad.append(('R1_pca', t, s_, th))
    for t, r in rec.rows:
        s = (t - T0) / 1e9
        o_s, o_t = float(r['out'][0]), float(r['out'][1])
        k = int(r['assist'])
        if r['hold']:
            if o_t != N or k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R7_hold_not_neutral', round(s, 3), o_t, k))
            continue
        if not r['select']:
            if k != A.ASSIST_NONE or r['launch_on']:
                bad.append(('R5_assist_on_human', round(s, 3), k))
            continue
        fwd = max(N - o_t, 0.0)
        m_fwd = max(N - float(r['model_out'][1]), 0.0)
        lim = {A.ASSIST_NONE: m_fwd, A.ASSIST_ESC: m_fwd, A.ASSIST_START: max(m_fwd, A.StartKick.KICK_US),
               A.ASSIST_LAUNCH: max(m_fwd, float(p['launch_us'])), A.ASSIST_RECOVERY: A.Recovery.ESCAPE_US,
               A.ASSIST_GIVEUP: 0.0}[k]
        if fwd > lim + 1e-6:
            bad.append(('R2_forward_over_source', round(s, 3), k, round(fwd, 1), round(lim, 1)))
        if k in (A.ASSIST_START, A.ASSIST_LAUNCH, A.ASSIST_GIVEUP) and o_t > N + 40.0:
            bad.append(('R4_reverse_side', round(s, 3), k, o_t))
    return bad


def min_x_after_first_hit(sim):
    t_hit = next((h[0] for h in sim.car.hits if h[1] == 'front'), None)
    if t_hit is None:
        return None, None
    xs = [x for t, x, *_ in sim.log if t >= t_hit]
    return sim.s['xf'], min(xs)


def main(n=150, seed=6102):
    rng = random.Random(seed)
    out = {'seed': seed, 'runs': 0, 'bad': collections.Counter(), 'examples': [], 'x1': []}
    orig = CL.Sim
    CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            p0 = dict(p1, rec_hold_keep=0)
            del FC.PH[:]
            s1, r1, pc1, _ = R.run_closed(p1, sc, wrap=FC.wrap)
            ph1 = list(FC.PH)
            s0, r0, pc0, _ = R.run_closed(p0, sc)
            m1, m0 = FC.metrics(s1, r1), FC.metrics(s0, r0)
            bad = []
            cont = [t for t, a, b, hs in ph1 if hs and a in A.Recovery.HOLD_HIT_PHASES]
            for h in m1['rear']:
                if not any(h[0] - 2.0 <= t <= h[0] for t in cont):
                    continue
                twin = [x for x in m0['rear'] if abs(x[0] - h[0]) <= 0.3]
                if not twin or -h[2] > max(-x[2] for x in twin) + 0.05:
                    bad.append(('K1_rear_after_kept_hold_worse_than_keep0', h, twin))
            if m1['front_push_s'] > m0['front_push_s'] + 0.5:
                bad.append(('K6_front_push_longer_than_keep0', m1['front_push_s'], m0['front_push_s']))
            bad += FC.rules_one(s1, r1, p1)
            for t, a, b, hs in ph1:
                if b == 'halt' and FC.v_at(s1, t) > 0.3:
                    bad.append(('K7_halt_while_moving_forward', t, round(FC.v_at(s1, t), 2)))
            bad += extra_rules(s1, r1, pc1, p1)
            if sc.get('veh_skip'):
                sn = dict(sc, veh_skip=[])
                s2, r2, pc2, _ = R.run_closed(p1, sn)
                xf, mx1 = min_x_after_first_hit(s1)
                _, mx2 = min_x_after_first_hit(s2)
                rear1 = [h for h in s1.car.hits if h[1] == 'rear']
                rear2 = [h for h in s2.car.hits if h[1] == 'rear']
                if mx1 is not None and mx2 is not None and (mx2 - mx1 > 0.15 or (rear1 and not rear2)):
                    out['x1'].append({'run': i, 'more_back_m': round(mx2 - mx1, 3), 'rear_now': rear1[:2],
                                      'rear_nogap': rear2[:2], 'veh_skip': sc['veh_skip'],
                                      'rev_s': p1['rev_s'], 'rev_us': p1['rev_us']})
            out['runs'] += 1
            for b in bad:
                out['bad'][b[0]] += 1
                if sum(1 for x in out['examples'] if x['kind'] == b[0]) < 6:
                    out['examples'].append({'kind': b[0], 'run': i, 'detail': b[1:],
                                            'params': {k: p1[k] for k in ('launch_boost', 'launch_no_kick', 'start_kick',
                                                                          'tof_guard', 'rev_s', 'rev_us')},
                                            'scene': {k: v for k, v in sc.items() if k != 'model'}})
            if (i + 1) % 25 == 0:
                print(i + 1, dict(out['bad']), len(out['x1']), flush=True)
    finally:
        CL.Sim = orig
        R.cleanup()
    out['bad'] = dict(out['bad'])
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / ('fuzz_on_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}, ensure_ascii=False, default=str)[:3000])
    for x in out['examples']:
        print(x['kind'], x['run'], x['detail'], x['params'])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
