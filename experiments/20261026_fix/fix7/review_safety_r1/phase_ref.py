"""止まりなしの下がる距離の、フレームの並びの位相 10 通りの幅(ON・OFF、場面 rev・brake、回 1〜4 の形の基本)。読むだけ。
出力 out/phase_ref.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
from grid_onoff import first_rec, scene  # noqa: E402

out = {}
try:
    p = rv7.base_params()
    for kind in ('rev', 'brake'):
        for nm, pp in (('on', rv7.ON(p)), ('off', rv7.OFF(p))):
            xs = []
            for i in range(10):
                sc = scene(kind)
                sc['cam0'] = round(0.15 + i * 0.0033, 4)
                m = first_rec(rv7.run('now', pp, sc))
                xs.append(m['back'])
            out['%s_%s' % (kind, nm)] = {'min': min(xs), 'max': max(xs), 'all': xs}
            print(kind, nm, out['%s_%s' % (kind, nm)], flush=True)
finally:
    rv7.cleanup()
(HERE / 'out' / 'phase_ref.json').write_text(json.dumps(out, indent=1))
