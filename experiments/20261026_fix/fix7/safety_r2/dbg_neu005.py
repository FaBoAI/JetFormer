"""直す 7-2 回目・係 safety: 中立 0.05 秒で下がれる偽物の ESC で、今と直す前(r1)が違うところをたどる(確かめ。装置に触れない)"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', 'dbg005')
sys.path.insert(0, str(HERE))
import h9  # noqa
from g9 import first_rec, scene  # noqa
N = h9.N
p = h9.base_params()
try:
    for two in (False, True):
        sc = dict(scene('rev', dur=9.0, esc=(0.05, 0.08)), drops=[(2.52, 2.62)], glitch=[(2.68, 1322), (2.88, 1322)] if two else [])
        for name in ('now', 'r1'):
            sim, rec, pca, ph = h9.run(name, h9.ON(p), sc)
            m = first_rec((sim, rec, pca, ph))
            print('two', two, name, m['back'], m['esc_push'], [(e[1]['what'], e[1]['phase'], e[1]['backed_s'], round((e[2] if len(e) > 2 else 0), 3)) for e in rec.events if e[0] == 'recovery_guard'])
            for x in ph:
                if 2.5 <= x[0] <= 3.4:
                    print('   ', x)
            print('   pca', [(t, round(th - N)) for t, s, th in pca.w if 2.5 <= t <= 3.4])
            print('   rev_on', next((t for t, r, c in sim.car.revlog if r), None))
finally:
    h9.cleanup()
