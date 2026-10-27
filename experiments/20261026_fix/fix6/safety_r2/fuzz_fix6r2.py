"""fix6 係 safety 直す 6-2 回目: 直したあとの でたらめな閉じたループ。fix5 の fix5/safety/fuzz_fix5.py(反証 4 回目の
fix4/review_safety/fuzz_closed.py の場面と決まり K1〜K7・K9 と、fix5 が足した R1・R2・R4・R5・R7)を、台本は変えずに読み込み、
「直す前の写し」だけを tests/fixtures/fix6_safety_r2_before(= fix6 r1 のあとの版)に差し替えて流す(fix6 r1 の fuzz_fix6.py と同じ形)。
1 つの場面を 6 通り: now1 = 今(keep 1・見張り 1)・now0 = 今の keep 0・bef1 = 直す前(fix6 r1)の keep 1・bef0 = 直す前の keep 0
(now0 とビット単位で同じか)・4 つのスイッチ全部 0 を今と直す前で(ビット単位で同じか)
装置・GPU・API には触れない。一時の場所は fix6/tmp_safety_r2_fz/<種>(終わったら消す)。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        experiments/20261026_fix/fix6/safety_r2/fuzz_fix6r2.py [本数] [種]
出力: fix6/safety_r2/out/fuzz_fix5_<種>.json(fuzz_fix5 の出力の名前のまま、この台本の out/ に書く)
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
sys.path.insert(0, str(FIX / 'fix5' / 'safety'))
import fuzz_fix5 as F          # noqa: E402

R = F.R
SEED = sys.argv[2] if len(sys.argv) > 2 else '4402'
TMPROOT = FIX / 'fix6' / 'tmp_safety_r2_fz'
R.TMP = TMPROOT / SEED
R.CL.TMP = R.TMP
R.SNAP_SHA['fix6_safety_r2_before'] = {'drive.py': '696ec9e8e4b853ebd221428768a23e1a48ddf566d11d81df7c5279f82ea15fec',
                                       'assist.py': 'd1d62c66bef856002c14db8dd2fd9c35279a36878b2e4817a279e9fb8bf64165'}
_load = R.load_snapshot


def load_snapshot(name):
    return _load('fix6_safety_r2_before' if name == 'fix5_safety_before' else name)


R.load_snapshot = load_snapshot
F.HERE = HERE

if __name__ == '__main__':
    try:
        F.main(*(int(x) for x in sys.argv[1:3]))
    finally:
        shutil.rmtree(R.TMP, ignore_errors=True)
        try:
            TMPROOT.rmdir()
        except OSError:
            pass
