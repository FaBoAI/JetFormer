"""fix5 係 safety: fuzz_closed の K1(残した中立止めのあとの後ろの当たりが keep 0 より悪い)を、今(now1)と直す前の写し(bef1)の
両方で全部の場面について数え、どの本が増えた・減ったかを出す。増えた本は、車両の値の止まり・カメラの止まりを抜いた同じ場面
(今)の後ろの当たりも出す。
    ... python -I k1_all.py 本数 種   出力: out/k1_all_<種>.json"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fuzz_fix5 as F          # noqa: E402
R, FC = F.R, F.FC
A = R.A


def k1(ph, m1, m0):
    cont = [t for t, a0, a1, hs in ph if hs and a0 in A.Recovery.HOLD_HIT_PHASES]
    bad = []
    for h in m1['rear']:
        if not any(h[0] - 2.0 <= t <= h[0] for t in cont):
            continue
        twin = [x for x in m0['rear'] if abs(x[0] - h[0]) <= 0.3]
        if not twin or -h[2] > max(-x[2] for x in twin) + 0.05:
            bad.append(h)
    return bad


def main(n, seed):
    rng = random.Random(seed)
    BEF = R.load_snapshot('fix5_safety_before')
    out = {'seed': seed, 'now1': {}, 'bef1': {}, 'new': {}}
    orig = R.CL.Sim
    R.CL.Sim = FC.FSim
    try:
        for i in range(n):
            sc = FC.rand_scene(rng)
            p1 = FC.rand_params(rng)
            p0 = dict(p1, rec_hold_keep=0)
            del FC.PH[:]
            a = R.run_closed(p1, sc, wrap=FC.wrap)
            pha = list(FC.PH)
            del FC.PH[:]
            b = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1], wrap=FC.wrap)
            phb = list(FC.PH)
            z = R.run_closed(p0, sc)
            m1, mb, m0 = FC.metrics(a[0], a[1]), FC.metrics(b[0], b[1]), FC.metrics(z[0], z[1])
            ka, kb = k1(pha, m1, m0), k1(phb, mb, m0)
            if ka:
                out['now1'][i] = ka
            if kb:
                out['bef1'][i] = kb
            if ka and not kb:
                v = {}
                for nm, s_ in (('no_veh_skip', dict(sc, veh_skip=[])), ('no_cam_drop', {k: x for k, x in dict(sc, drops=[]).items()
                                                                                      if k != 'hit_drop'})):
                    r_ = R.run_closed(p1, s_)
                    v[nm] = [h for h in r_[0].car.hits if h[1] == 'rear']
                out['new'][i] = {'now1': ka, 'bef1_rear': mb['rear'], 'veh_skip': sc['veh_skip'], 'drops': sc['drops'],
                                 'variants': v, 'rev_s': p1['rev_s'], 'rev_us': p1['rev_us']}
                print(i, out['new'][i], flush=True)
    finally:
        R.CL.Sim = orig
        R.cleanup()
    (HERE / 'out' / ('k1_all_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print('now1', sorted(out['now1']), 'bef1', sorted(out['bef1']))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
