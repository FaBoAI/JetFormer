"""fuzz_fix6(= fix5 の fuzz_fix5 の場面)の K1(残した中立止めのあと 2 秒以内の後ろの当たりが、keep 0 の同じ頃の当たりより
0.05 m/s 以上速い・keep 0 では当たらない)を、今(now1)と直す前の写し(bef1 = fix5 のあとの版)の両方で場面ごとに数え、
今だけに出た場面・直す前だけの場面・両方の場面を分ける(読むだけ)。値の止まりを抜いた同じ場面の今の keep 1 の当たりも出す。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r1/k1_cmp.py [本数] [種]
出力: out/k1_cmp_<種>.json
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuzz_fix6        # noqa: E402,F401  (fix6_before に差し替えた fuzz_fix5)
F = fuzz_fix6.F
R, FC, A = F.R, F.FC, F.A


def k1(run, ph, m0):
    s1, rc1, pc1, _ = run
    m1 = FC.metrics(s1, rc1)
    cont = [t for t, a0, a1, hs in ph if hs and a0 in A.Recovery.HOLD_HIT_PHASES]
    out = []
    for h in m1['rear']:
        if not any(h[0] - 2.0 <= t <= h[0] for t in cont):
            continue
        twin = [x for x in m0['rear'] if abs(x[0] - h[0]) <= 0.3]
        if not twin or -h[2] > max(-x[2] for x in twin) + 0.05:
            out.append(h)
    return out, m1['rear']


def main(n=300, seed=4404):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')     # fuzz_fix6 が fix6_before に差し替える
    res = {'seed': seed, 'only_now': [], 'only_before': [], 'both': []}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            p0 = dict(p1, rec_hold_keep=0)
            del FC.PH[:]
            r1 = R.run_closed(p1, sc, wrap=FC.wrap)
            ph1 = list(FC.PH)
            del FC.PH[:]
            rb = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1], wrap=FC.wrap)
            phb = list(FC.PH)
            r0 = R.run_closed(p0, sc)
            m0 = FC.metrics(r0[0], r0[1])
            a, rear1 = k1(r1, ph1, m0)
            b, rearb = k1(rb, phb, m0)
            if not a and not b:
                continue
            nv = R.run_closed(p1, dict(sc, veh_skip=[]))
            rec = {'run': i, 'now1': a, 'bef1': b, 'now0_rear': m0['rear'], 'now1_rear': rear1, 'bef1_rear': rearb,
                   'now1_noveh_rear': FC.metrics(nv[0], nv[1])['rear'], 'veh_skip': sc.get('veh_skip'),
                   'drops': sc.get('drops'), 'params': {k: p1[k] for k in ('rev_s', 'rev_us', 'tof_guard', 'launch_boost')}}
            key = 'both' if a and b else ('only_now' if a else 'only_before')
            res[key].append(rec)
            print(key, i, 'now1', a, 'bef1', b, 'k0', m0['rear'], 'noveh', rec['now1_noveh_rear'], flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    (HERE / 'out' / ('k1_cmp_%d.json' % seed)).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print({k: len(v) for k, v in res.items() if isinstance(v, list)})


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
