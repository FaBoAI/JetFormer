"""スイッチを OFF にしたときのビット単位の同じ(読むだけ)。でたらめな場面(fix4 の rand_scene)× 回 1〜4 の形で、今の版と
f6b(fix5 のあと)・f5b(fix4 のあと)を、(1) 4 つのスイッチ全部 0、(2) 見張り 1・残す 0(片方だけ ON)、(3) 見張り 0・残す 0・発進の前進は
でたらめ で比べる。PCA9685 に書いた値・記録の行・出来事。出力 out/ident_more_<種>.json
  python -I ident_more.py [本数] [種]"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rv7  # noqa: E402
import fuzz_onoff as FZ  # noqa: E402


def main(n=150, seed=7301):
    rv7.check_now()
    rng = random.Random(seed)
    out = {'seed': seed, 'n': 0, 'same': {}, 'diff': []}
    try:
        for i in range(n):
            sc = FZ.FC.rand_scene(rng)
            p = FZ.rand_params(rng)
            forms = {'off4': dict(p, rec_back_guard=0, rec_hold_keep=0, launch_boost=0, launch_no_kick=0),
                     'guard_only': dict(p, rec_back_guard=1, rec_hold_keep=0),
                     'off2': dict(p, rec_back_guard=0, rec_hold_keep=0)}
            for nm, pp in forms.items():
                rn = rv7.run('now', pp, sc, wrap=False)
                for snap in ('f6b', 'f5b'):
                    ok, why = rv7.same_run(rn, rv7.run(snap, pp, sc, wrap=False))
                    k = '%s_vs_%s' % (nm, snap)
                    out['same'][k] = out['same'].get(k, 0) + int(ok)
                    if not ok:
                        out['diff'].append((i, k, str(why)[:300]))
            out['n'] += 1
            if (i + 1) % 25 == 0:
                print(i + 1, out['same'], len(out['diff']), flush=True)
    finally:
        rv7.cleanup()
    (HERE / 'out' / ('ident_more_%d.json' % seed)).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:3]))
