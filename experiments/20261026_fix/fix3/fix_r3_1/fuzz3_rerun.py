"""係 safety の fix3/safety/fuzz_loop3.py(決まり 9 つ × でたらめな走行ループ)を、直す 3 回目・1 組目(r3-r3-safety-1・safety-2)の直しのあとの drive.py・
assist.py で流し直す(台本はそのまま読み込み、出力と一時置き場だけこの組の場所にする。係 safety の out/ は書き換えない)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261026_fix/fix3/fix_r3_1/fuzz3_rerun.py [本数] [種]
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'safety'))
import fuzz_loop3 as F      # noqa: E402

F.OUT_DIR = HERE / 'out'
F.TMP = HERE.parent / 'tmp_fixr31_fuzz3'

if __name__ == '__main__':
    try:
        F.main(int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 1103)
    finally:
        shutil.rmtree(F.TMP, ignore_errors=True)
