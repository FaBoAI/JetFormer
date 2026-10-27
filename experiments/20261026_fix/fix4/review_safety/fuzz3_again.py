"""反証 4 回目・安全: 係 safety の fix3/safety/fuzz_loop3.py(開いたループ・決まり 12)を、今の drive.py・assist.py で
別の種・本数で流す(台本はそのまま読み込み、出力と一時置き場だけこの場所にする。係 safety・直す組の out/ は書き換えない)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -B -I \
        experiments/20261026_fix/fix4/review_safety/fuzz3_again.py [本数] [種]
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rs_common  # noqa: E402,F401  (sys.path を合わせる)
import fuzz_loop3 as F      # noqa: E402

F.OUT_DIR = HERE / 'out'
F.TMP = HERE / 'tmp_fuzz3'

if __name__ == '__main__':
    try:
        F.main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 4403)
    finally:
        shutil.rmtree(F.TMP, ignore_errors=True)
