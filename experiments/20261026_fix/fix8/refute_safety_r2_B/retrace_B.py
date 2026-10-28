"""票 B: 指摘の 1 場面(trace12.py の入力そのまま)を、一時の場所だけこの係のもの(fix8/tmp_refute_safety_r2_B)にして流し直す。
係 safety の台本(review_safety_r2/h12.py・exp12.py・trace12.py)は読み込むだけで変えない。
    python -I -B retrace_B.py → 標準出力(out/retrace_B.txt に写す)
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = HERE.parent / 'review_safety_r2'
os.environ['RS82_TAG'] = 'refB'
sys.path.insert(0, str(REV))
import h12  # noqa: E402
import trace12  # noqa: E402

MY = h12.FIXD / 'fix8' / 'tmp_refute_safety_r2_B' / 'h' / 'refB'
h12.h11.TMP = MY
h12.TMP = MY

SD = {"base": "rev", "esc": "human", "form": "kai1", "pert": [["drop", 4.042, 3]]}

if __name__ == '__main__':
    sds = [SD]
    if len(sys.argv) > 1:
        sds = [json.loads(a) for a in sys.argv[1:]]
    try:
        for sd in sds:
            print('######', json.dumps(sd))
            trace12.trace(sd, 3.9, 4.75, ('on', 'off', 'base_off'))
    finally:
        h12.cleanup()
        for q in (MY.parent, MY.parent.parent):
            try:
                q.rmdir()
            except OSError:
                pass
