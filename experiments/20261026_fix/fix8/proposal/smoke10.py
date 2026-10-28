"""fix8 係 safety-proposal: 案の写しが読めて、残り 1(revslow の失敗の組)で違いが出るかの smoke。装置・GPU・API なし"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', 'smoke')
sys.path.insert(0, str(HERE))
import h10  # noqa: E402
from g10 import scene  # noqa: E402
import exp10  # noqa: E402

try:
    p = h10.base_params()
    for tg in (2.98, 3.02):
        sc = dict(scene('rev', dur=12.0, esc=(0.2, 0.08)), glitch=[(tg, 1322)])
        for v in ('on', 'pA', 'pB', 'pC', 'off', 'pA_off', 'pAx_off', 'pBx_off', 'pCx_off'):
            name, f, force, patch = exp10.VARIANTS[v]
            m = exp10.summ(h10.run(name, f(p), sc, force=force, patch=patch))
            print(tg, v, m['back'], m['esc_push'], m['all_guards'], m['rev_on_t'], m['t_escape'], m['t_brake'])
finally:
    h10.cleanup()
