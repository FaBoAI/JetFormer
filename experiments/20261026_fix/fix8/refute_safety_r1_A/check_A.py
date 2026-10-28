"""票 A: (1) 後ろの物を本当に置いた走りで、最初の後ろの当たりの速さが「物なしの走りから出した速さ」(sweep_A の hit_profile)と
同じかを照らす。(2) fix7 の前(tests/fixtures/fix7_safety_before の drive.py・assist.py)の両方 ON が、今の両方 ON と同じ当たりかを見る
(指摘の「fix2 の見張りからある形」)。
    python -I -B check_A.py → 標準出力・out/check_A.json
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cl_A  # noqa: E402

CL = cl_A.CL
FIX = cl_A.ROOT / 'tests' / 'fixtures' / 'fix7_safety_before'
SUMS = {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
        'assist.py': '951a4c122c954a12ecc5056d823729ab9934112977f30645f92b14b556bf50f4'}


def load_f7b():
    for fn, h in SUMS.items():
        assert hashlib.sha256((FIX / fn).read_bytes()).hexdigest() == h, fn
    aspec = importlib.util.spec_from_loader('jetformer._chkA_assist_f7b', loader=None)
    amod = importlib.util.module_from_spec(aspec)
    exec(compile((FIX / 'assist.py').read_text(encoding='utf-8'), str(FIX / 'assist.py'), 'exec'), amod.__dict__)
    spec = importlib.util.spec_from_loader('jetformer._chkA_drive_f7b', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile((FIX / 'drive.py').read_text(encoding='utf-8'), str(FIX / 'drive.py'), 'exec'), mod.__dict__)
    mod.assist_mod = amod
    return mod, amod


def first_rear(sim):
    h = [x for x in sim.car.hits if x[1] == 'rear']
    return None if not h else round(-h[0][2], 3)


def main():
    out = {'obstacle': [], 'f7b': []}
    try:
        # (1) 物を置いた走り(当たりの速さは closed_loop.Car と同じく 0.01 m/s に丸めた値)
        for esc in ('fit', 'human', 'cl'):
            for sw in ('on', 'off'):
                sim, rec, pca, ph = cl_A.run(cl_A.ESC[esc], sw)
                t0, t1 = cl_A.first_rec(ph)
                ds = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.25]
                prof, x0, back, vpk = cl_A.hit_profile(sim, t0, t1 or 99, ds)
                for d in ds:
                    s2, *_ = cl_A.run(cl_A.ESC[esc], sw, rear=x0 - d)
                    real = first_rear(s2)
                    pr = prof[d]
                    ok = (real is None and pr is None) or (real is not None and pr is not None and abs(real - pr) <= 0.06)
                    out['obstacle'].append({'esc': esc, 'sw': sw, 'd': d, 'profile': pr, 'placed': real, 'ok': ok})
        bad = [r for r in out['obstacle'] if not r['ok']]
        print('obstacle check: %d runs, mismatches %d' % (len(out['obstacle']), len(bad)), bad[:5])
        # (2) fix7 の前の両方 ON
        drv0, A0 = CL.drive, CL.A
        f7d, f7a = load_f7b()
        rows = []
        for esc in ('fit', 'human', 'cl'):
            for m in (120.0, 200.0):
                now = {}
                for ver in ('now', 'f7b'):
                    if ver == 'f7b':
                        CL.drive, CL.A = f7d, f7a
                    try:
                        sim, rec, pca, ph = cl_A.run(cl_A.ESC[esc], 'on', m=m)
                    finally:
                        CL.drive, CL.A = drv0, A0
                    t0, t1 = cl_A.first_rec(ph)
                    prof, x0, back, vpk = cl_A.hit_profile(sim, t0, t1 or 99, [0.3, 0.5, 0.8, 1.0, 1.2])
                    now[ver] = {'w': [(round(t, 3), s, th) for t, s, th in pca.w], 'back': back, 'prof': prof}
                same = now['now']['w'] == now['f7b']['w']
                rows.append({'esc': esc, 'm': m, 'writes_same': same, 'back_now': now['now']['back'],
                             'back_f7b': now['f7b']['back'], 'n_writes': len(now['now']['w'])})
                print('f7b vs now (ON) esc %s m %s: writes same %s back %s / %s' % (esc, m, same, now['now']['back'],
                                                                                    now['f7b']['back']))
        out['f7b'] = rows
    finally:
        cl_A.cleanup()
    (HERE / 'out' / 'check_A.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
