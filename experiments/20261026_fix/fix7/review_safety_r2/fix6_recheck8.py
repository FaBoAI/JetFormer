"""反証 7-2 回目(fix7 r1 のあとの今の版で流し直す。review_safety_r1/fix6_recheck.py の写し)・safety: fix6 の指摘 3 件(w1-safety-r61-safety-1・w2-safety-r62-safety-1・w2-safety-r62-safety-2)の入力そのものを
今の版で流し直し、直ったかと、同じ入力の OFF(両方 0)を並べる。fix6 r2 の exp2(本物の drive._loop・閉じたループ)をそのまま読み込む
(台本は変えない)。比べ: f6b = fix5 のあと(指摘 r61 の直す前)・r1 = fix6 r1 のあと(指摘 r62 の直す前)。出力 out/fix6_recheck.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'fix6' / 'safety_r2'))
import os
os.environ.setdefault('R2_TAG', 'fix7_review_safety_r2')
import exp2 as E   # noqa: E402

E.TMP = HERE.parents[1] / 'fix7' / 'tmp_review_safety_r2' / 'fix6_recheck'

CASES = {
    'r61 rev d0=0 L=0.55': ('rev', [(2.60, 3.15)], [], None),
    'r61 brake d0=0 L=0.60': ('brake', [(2.73, 3.33)], [], None),
    'r62-1 rev A=(2.11,2.26) B=(2.26,2.45)': ('rev', [(2.26, 2.45)], [(2.11, 2.26)], None),
    'r62-2 rev neu0.2 B=(2.52,2.72)': ('rev', [(2.52, 2.72)], [], (0.2, 0.08)),
}


def main():
    out = {}
    try:
        for name, (kind, skips, drops, esc) in CASES.items():
            sc = E.scene(kind, skips=skips, drops=drops)
            row = {}
            for v, mod, p in (('now_on', 'now', E.P1()), ('now_off', 'now', E.P1(rec_back_guard=0, rec_hold_keep=0)),
                              ('before_on', 'before', E.P1()), ('r1_on', 'r1', E.P1())):
                m = E.metrics(E.run(mod, p, sc, esc=esc), p)
                row[v] = None if m is None else {k: m[k] for k in ('back', 'push', 'front', 'rear', 't_on', 'dead', 'retry', 'guards')}
            ng = E.metrics(E.run('now', E.P1(), E.scene(kind), esc=esc), E.P1())
            row['nostall_on'] = ng['back']
            out[name] = row
            print(name, {k: ((v['back'], v['push'], v['retry']) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
    finally:
        E.cleanup()
        import shutil
        shutil.rmtree(E.TMP, ignore_errors=True)
    (HERE / 'out' / 'fix6_recheck.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
