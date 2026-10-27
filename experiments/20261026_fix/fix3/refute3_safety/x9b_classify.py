"""X9 の R4(長く見ていない中立止めのあとに後退側)を分ける: 中立止めの前の段・止まりかけか・車の速さ"""
import collections
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'refute2_safety'))
import cl3                                    # noqa: E402
from cl3 import A, C, N, PARAMS, T0           # noqa: E402
import fuzz_cl as F                           # noqa: E402
import x9_fuzz as X                           # noqa: E402

rng = random.Random(1103)
cls = collections.Counter()
ex = []
for i in range(150):
    sc_ = F.scene_of(rng)
    p = X.params_on(rng)
    sim, rec, pca, ph = cl3.run(p, sc_)
    rows = [(t, r) for t, r in rec.rows if not r['hold']]
    for (ta, ra), (tb, rb) in zip(rows, rows[1:]):
        ga, gb = (ta - T0) / 1e9, (tb - T0) / 1e9
        if gb - ga <= 0.5 or not (ra['select'] and rb['select']):
            continue
        rev = [(t - T0) / 1e9 for t, r in rec.rows if not r['hold'] and int(r['assist']) == A.ASSIST_RECOVERY and r['out'][1] > N + 40 and gb <= (t - T0) / 1e9 <= gb + 1.6]
        if not rev:
            continue
        before = [x for x in ph if x[0] <= ga + 0.02]
        after = [x for x in ph if x[0] >= gb - 0.02]
        ph_before = before[-1][2] if before else 'no-step'
        ph_after = after[0][1:3] if after else None
        v_a = next((r[2] for r in sim.log if r[0] >= ga), None)
        v_b = next((r[2] for r in sim.log if r[0] >= gb), None)
        hits_in = [h for h in sim.car.hits if ga <= h[0] <= gb]
        key = (ph_before, 'moving' if v_a is not None and abs(v_a) > 0.3 else 'slow', 'stopped' if v_b is not None and abs(v_b) < 0.05 else 'moving', 'hit-in-gap' if hits_in else 'no-hit')
        cls[key] += 1
        if len(ex) < 12:
            ex.append({'i': i, 'gap': (round(ga, 3), round(gb, 3)), 'first_rev_after': round(rev[0] - gb, 3), 'giveup_after': any(int(r['assist']) == 3 for t, r in rec.rows if not r['hold'] and gb <= (t - T0) / 1e9 <= gb + 0.1), 'key': key,
                       'ph_after': ph_after, 'hits_in': hits_in, 'assist_a': int(ra['assist']), 'assist_b': int(rb['assist'])})
print(json.dumps({'classes': {' / '.join(map(str, k)): v for k, v in cls.items()}, 'examples': ex}, ensure_ascii=False, indent=1, default=str))
(Path(__file__).resolve().parent / 'out' / 'x9b_classify.json').write_text(json.dumps({'classes': {' / '.join(map(str, k)): v for k, v in cls.items()}, 'examples': ex}, ensure_ascii=False, indent=1, default=str))
shutil.rmtree(cl3.TMP, ignore_errors=True)
