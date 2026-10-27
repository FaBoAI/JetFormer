"""指摘 2 件の場面を、今(now = fix6 r1)・直す前の写し(before = fix5 のあと)・候補(cand + スイッチ)で流す(読むだけ)。
  f1: 元の偽物の ESC・場面 rev・カメラの止まり 2.11〜2.26 秒・車両の値の止まり 2.26〜2.45 秒(w2-safety-r62-safety-1)
  f2: 中立 0.2 秒で下がれる偽物の ESC・場面 rev・車両の値の止まり 2.52〜2.72 秒(w2-safety-r62-safety-2)
    python -I repro2.py [variant ...]    variant = now / before / cand:<flags>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp2 as E   # noqa: E402

CASES = {
    'f1': dict(esc=None, sc=lambda: E.scene('rev', skips=[(2.26, 2.45)], drops=[(2.11, 2.26)])),
    'f1_ng': dict(esc=None, sc=lambda: E.scene('rev')),
    'f2': dict(esc=(0.2, 0.08), sc=lambda: E.scene('rev', skips=[(2.52, 2.72)])),
    'f2_ng': dict(esc=(0.2, 0.08), sc=lambda: E.scene('rev')),
}


def variant(v):
    if ':' in v:
        n, f = v.split(':', 1)
        return n, f
    return v, ''


def main(vs):
    out = {}
    try:
        for case, c in CASES.items():
            for v in vs:
                name, fl = variant(v)
                p = E.P1()
                m = E.metrics(E.run(name, p, c['sc'](), esc=c['esc'], flags=fl), p)
                out['%s|%s' % (case, v)] = m
                print(case, v, {k: m[k] for k in ('back', 'push', 't_on', 'dead', 'retry', 't_escape', 'guards')},
                      flush=True)
    finally:
        E.cleanup()
    (HERE / 'out' / 'repro2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1:] or ['now', 'before', 'r1', 'final', 'final:rn12', 'cand:so,oc', 'cand:so,oc,dr'])
