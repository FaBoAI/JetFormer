"""verify/gap_check.py — scorecard に無い周の、AI_CMD と VEHICLE の途切れを見る。"""
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, rawload
for lg, a, b in [('20261024-185703', 25.9, 35.5), ('20261023-210840', 55.2, 65.0), ('20261023-211547', 7.1, 17.1), ('20261024-162033', 22.4, 117.3)]:
    r = rawload.load(lg)
    t = r['a_t']; w = (t >= a) & (t <= b)
    d = np.diff(t[w]); k = np.argmax(d)
    vt = r['v_t']; wv = (vt >= a) & (vt <= b); dv = np.diff(vt[wv])
    sel = r['a_sel'][w]
    print(lg, 'AI_CMD max gap %.3f at t=%.2f' % (d[k], t[w][k]), 'n gaps>0.2:', (d > 0.2).sum(), '| VEH max gap %.3f' % dv.max(), '| AI_CMD select all', sel.all(), 'dry any', r['a_dry'][w].any())
