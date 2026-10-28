"""fix8 係 safety 直す 8-1 回目: sweep_r1 の「後ろの物なしの走りから出した当たりの速さ」(hit_profile)が、物を本当に置いた走りの
最初の後ろの当たりと合うかを、案 qA・qB・qC と今の ON・OFF で照らす(票 A の check_A.py の (1) と同じ照らし方。ESC は fit・human・cl、
距離は止まった所から 0.2〜1.25 m の 7 点。差 0.06 m/s 以内なら合う)。
    python -B check_r1.py → 標準出力・out/check_r1.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('R1_TAG', 'check_%d' % os.getpid())
import cl_r1  # noqa: E402


def first_rear(sim):
    h = [x for x in sim.car.hits if x[1] == 'rear']
    return None if not h else round(-h[0][2], 3)


def main():
    cl_r1.check_now()
    out = []
    ds = [0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.25]
    try:
        for esc in ('fit', 'human', 'cl'):
            for ver, sw in (('now', 'on'), ('now', 'off'), ('qA', 'on'), ('qB', 'on'), ('qC', 'on')):
                sim, rec, pca, ph = cl_r1.run(cl_r1.ESC[esc], sw, ver=ver)
                t0, t1 = cl_r1.first_rec(ph)
                prof, x0, back, vpk = cl_r1.hit_profile(sim, t0, t1 or 99, ds)
                for d in ds:
                    s2, *_ = cl_r1.run(cl_r1.ESC[esc], sw, rear=x0 - d, ver=ver)
                    real = first_rear(s2)
                    pr = prof[d]
                    ok = (real is None and pr is None) or (real is not None and pr is not None and abs(real - pr) <= 0.06)
                    out.append({'esc': esc, 'ver': ver, 'sw': sw, 'd': d, 'profile': pr, 'placed': real, 'ok': ok})
    finally:
        cl_r1.cleanup()
    bad = [r for r in out if not r['ok']]
    print('runs %d mismatches %d' % (len(out), len(bad)), bad[:8])
    (HERE / 'out' / 'check_r1.json').write_text(json.dumps({'rows': out, 'mismatches': len(bad)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
