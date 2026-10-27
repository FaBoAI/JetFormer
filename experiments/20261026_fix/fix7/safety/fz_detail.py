"""直す 7-1 回目・係 safety: fuzz_fix7.py の「下がらない(0.1 m 未満)まま escape が前の壁を 0.2 秒より押した」リカバリーを、場面を
作り直して(同じ乱数の並び)今(now1)・直す前(bef1)・OFF で 1 つずつ見る。リカバリーの区切りごとに 始め・escape の時刻・下がった距離・
押した秒・段の移り・見張りの出来事・その区切りの間の跳ね(scene['glitch'])・カメラ/車両の値の止まり・後ろの物までの距離を出す。
    python -I fz_detail.py <種> <本数> [場面の番号 ...]   (番号を省くと、now1 に 1 つでもあった場面を全部)
出力は標準出力。装置・GPU・API には触れない
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.argv += [] if len(sys.argv) > 2 else ['7401', '150']
SEED, NRUN = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], str(NRUN), str(SEED)] + sys.argv[3:]
sys.path.insert(0, str(HERE))
import fuzz_fix7 as Z  # noqa: E402

R, FC, A, N, T0 = Z.R, Z.FC, Z.A, Z.N, Z.T0
R.TMP = Z.FIX / 'fix7' / 'tmp_safety' / ('fzd_%d' % SEED)
R.CL.TMP = R.TMP


def segs(sim, rec):
    rows = [((t - T0) / 1e9, r) for t, r in rec.rows if not r['hold']]
    out, cur = [], []
    for s, r in rows:
        if int(r['assist']) == A.ASSIST_RECOVERY and r['select']:
            cur.append((s, float(r['out'][1])))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    esc = N - A.Recovery.ESCAPE_US
    res = []
    for sg in out:
        t0 = sg[0][0]
        te = next((s for s, th in sg if abs(th - esc) < 1e-6), None)
        if te is None:
            continue
        xs = [x for t, x, *_ in sim.log if t0 <= t <= te]
        if not xs:
            continue
        back = xs[0] - min(xs)
        push = sum(R.CL.DT for t, x, v, thr, con, rv, sl in sim.log if te <= t < te + 0.5 and con == 'front'
                   and abs(thr - esc) < 1e-6)
        res.append((round(t0, 2), round(te, 2), round(back, 3), round(push, 2)))
    return res


def main(want):
    rng = random.Random(SEED)
    BEF = R.load_snapshot('fix7_safety_before')
    orig = R.CL.Sim
    R.CL.Sim = Z.GSim
    try:
        for i in range(NRUN):
            sc = Z.add_glitch(rng, FC.rand_scene(rng))
            p1 = FC.rand_params(rng)
            if want and i not in want:
                continue
            del FC.PH[:]
            r_now = R.run_closed(p1, sc, wrap=FC.wrap)
            ph = list(FC.PH)
            sn = segs(r_now[0], r_now[1])
            if not want and not any(b < 0.1 and pu > 0.2 for _, _, b, pu in sn):
                continue
            r_bef = R.run_closed(p1, sc, mod=BEF[0], amod=BEF[1])
            r_off = R.run_closed(dict(p1, **Z.OFF4), sc)
            print('=== run', i, {k: p1[k] for k in ('rev_s', 'rev_us', 'launch_boost', 'start_kick', 'tof_guard', 'speed_ceiling')},
                  'xf', round(sc['xf'], 2), 'xr', sc.get('xr') and round(sc['xf'] - sc['xr'], 2), 'front_soft', sc.get('front_soft'),
                  'hit_drop', sc.get('hit_drop'))
            print('  drops', [(round(a, 2), round(b, 2)) for a, b in sc['drops']], 'veh_skip',
                  [(round(a, 2), round(b, 2)) for a, b in sc['veh_skip']], 'human', sc.get('human'))
            print('  glitch', sc['glitch'])
            for nm, rr in (('now1', r_now), ('bef1', r_bef), ('off', r_off)):
                print('  %s segs' % nm, segs(rr[0], rr[1]), 'hits', rr[0].car.hits[:6])
                print('  %s guards' % nm, [(round((e[1]['cam_epoch_us'] - T0 / 1000) / 1e6, 2) if False else None, e[1]['what'],
                                            e[1]['phase']) for e in rr[1].events if e[0] == 'recovery_guard'][:12])
            last = None
            seq = []
            for t, p0, p1_, hs in ph:
                if p1_ != last:
                    seq.append((t, p1_, 'H' if hs else ''))
                    last = p1_
            print('  now1 phases', seq[:40])
            if want:
                for nm, rr in (('now1', r_now), ('bef1', r_bef)):
                    print('  %s x/v/thr' % nm, [(t, round(x, 2), round(v, 2), int(th), con) for t, x, v, th, con, *_ in rr[0].log
                                               if abs(round(t * 100) % 5) == 0][:0])
    finally:
        R.CL.Sim = orig
        R.cleanup()


if __name__ == '__main__':
    main([int(x) for x in sys.argv[3:]])
