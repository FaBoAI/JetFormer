"""票 A: 1 場面だけ(fit の ESC・前進 120・壁 2.4・手前 0.45)で ON / OFF / 見張りだけの書いた値と下がり方を並べる。
    python -I -B smoke_A.py [esc] → 標準出力"""
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cl_A  # noqa: E402

NOW = {'drive.py': '3f2e31a22ecb9abdd9c8f84b3f787d075ef636b6d5679bad5c0fc28aee18388e',
       'assist.py': 'ad6db15b4b7cabb9e4d8caf6c78a437f0123ab858e0bad9629d78c56328c138f'}


def main():
    for fn, h in NOW.items():
        assert hashlib.sha256((cl_A.ROOT / 'jetformer' / fn).read_bytes()).hexdigest() == h, fn
    esc = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    try:
        for sw in ('on', 'off', 'guard'):
            t = time.time()
            sim, rec, pca, ph = cl_A.run(cl_A.ESC[esc], sw)
            t0, t1 = cl_A.first_rec(ph)
            prof, x0, back, vpk = cl_A.hit_profile(sim, t0, t1 or 99, [0.3, 0.5, 0.8, 1.0, 1.2])
            seq, prev = [], None
            for tt, s, th in pca.w:
                if t0 - 0.05 <= tt <= (t1 or 99) and th != prev:
                    seq.append((round(tt, 3), th))
                    prev = th
            phs, pp = [], None
            for tt, p0, p1, *_ in ph:
                if t0 - 0.01 <= tt <= (t1 or 99) and p1 != pp:
                    phs.append((tt, p1))
                    pp = p1
            ev = [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']
            print('%-5s rec %.3f-%s x0 %.3f back %.3f m peak %.2f m/s hit@d %s  (%.1fs)' % (
                sw, t0, t1, x0, back, vpk, prof, time.time() - t))
            print('      writes', seq[:10])
            print('      phases', phs[:10], 'events', ev)
    finally:
        cl_A.cleanup()


if __name__ == '__main__':
    main()
