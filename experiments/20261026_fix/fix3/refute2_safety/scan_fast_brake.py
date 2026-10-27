"""fuzz_cl の場面で、ON(keep 1・guard 1)のとき AI 側で前へ 1 m/s 以上なのに後退側(> 中立 + 40)を書いた行の補助の種類と段"""
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop as CL          # noqa: E402
import fuzz_cl as F               # noqa: E402
from closed_loop import A, N, T0  # noqa: E402


def main(n=150, seed=2):
    rng = random.Random(seed)
    tot = collections.Counter()
    ex = []
    for i in range(n):
        sc_ = F.scene_of(rng)
        p0 = F.params_of(rng)
        for keep, guard in ((1, 1),):
            sim, rec, pca = CL.run(dict(p0, rec_hold_keep=keep, rec_back_guard=guard), sc_)
            ph = list(CL.PH)
            v_at = {round(t, 2): v for t, x, v, *_ in sim.log}
            for t, r in rec.rows:
                if r['hold'] or not r['select']:
                    continue
                s = (t - T0) / 1e9
                v = v_at.get(round(s + 0.01, 2), v_at.get(round(s, 2)))
                o = float(r['out'][1])
                if o > N + 40 and v is not None and v > 1.0:
                    k = int(r['assist'])
                    phase = next((p1 for tt, p0_, p1, hg, ig in reversed(ph) if tt <= s - 100 + 100 + 0.02), None)
                    key = (A.ASSIST_NAMES[k] or 'none', 'guard%d' % int(r['guard']), 'cap%d' % int(r.get('cap_mode', -1)),
                           'model>%d' % int(float(r['model_out'][1]) > N + 40))
                    tot[key] += 1
                    if k == A.ASSIST_RECOVERY and len(ex) < 8:
                        ex.append((i, round(s, 3), round(v, 2), round(o, 1), [x for x in ph if abs(x[0] - s) < 0.15]))
    for k, v in tot.most_common():
        print(k, v)
    for e in ex:
        print(e)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
