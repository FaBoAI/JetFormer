"""同じ窓を今の既定(rec_back_guard=0・rec_hold_keep=0)と keep=0・guard=1 で回す(比べる相手)。標準出力だけ"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep as S   # noqa: E402


def main():
    tr = 2.52
    for keep, guard in ((0, 0), (0, 1), (1, 1)):
        print('== keep=%d guard=%d nohold' % (keep, guard), S.one(keep, guard)['back'])
        for L in (0.3, 0.45, 0.6, 1.0):
            sel = []
            for i in range(-10, 41, 2):
                o = i / 100.0
                r = S.one(keep, guard, drop=(tr + o, tr + o + L))
                sel.append((o, r['back'], r['retries']))
            print(' L=%.2f max=%.3f  >1.30 at o=%s' % (L, max(x[1] for x in sel), [x[0] for x in sel if x[1] > 1.30]))
        for L in (0.3, 0.5):
            sel = []
            for i in range(-10, 41, 2):
                o = i / 100.0
                r = S.one(keep, guard, vskip=(tr + o, tr + o + L))
                sel.append((o, r['back'], r['retries']))
            print(' veh L=%.2f max=%.3f  >1.30 at o=%s retries_at=%s' % (L, max(x[1] for x in sel), [x[0] for x in sel if x[1] > 1.30],
                                                                         [x[0] for x in sel if x[2]]))


if __name__ == '__main__':
    try:
        main()
    finally:
        shutil.rmtree(S.CL.TMP, ignore_errors=True)
