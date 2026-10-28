"""fix8 係 safety-proposal: exp10 のくみごとの組の数と最初の数組(走らせない。基準の時刻を出すための 1〜数本だけ走る)"""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ.setdefault('RV9_TAG', 'count')
sys.path.insert(0, str(HERE))
import h10  # noqa: E402
import exp10  # noqa: E402
try:
    for kind in sys.argv[1:]:
        cs = list(exp10.cells(kind))
        print(kind, len(cs), [c[0] for c in cs[:3]], cs[-1][0], flush=True)
finally:
    h10.cleanup()
