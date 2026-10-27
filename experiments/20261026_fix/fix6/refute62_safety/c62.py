"""反証 6-2 回目・安全(fix6/refute62_safety): 共通の部品。直さない・装置に触れない・GPU を使わない。
fix4/review_safety/rs_common.py(本物の drive._loop を閉じたループで回す)と fix6/safety_r1/exp_common.py を読み込むだけ(変えない)。
一時の場所は fix6/tmp_refute62_safety(終わったら消す)。
"""
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
TMPROOT = FIX / 'fix6' / 'tmp_refute62_safety' / os.environ.get('R62_TAG', 'main')   # 並べて流すときは R62_TAG を分ける
sys.path.insert(0, str(FIX / 'fix4' / 'review_safety'))
sys.path.insert(0, str(FIX / 'fix6' / 'safety_r1'))
import rs_common as R          # noqa: E402
import exp_common as E         # noqa: E402

R.TMP = TMPROOT / 'rs'
R.CL.TMP = R.TMP
E.TMP = TMPROOT / 'exp'
R.SNAP_SHA['fix5_safety_before'] = {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                    'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}
R.SNAP_SHA['fix6_safety_before'] = {'drive.py': '041a66de7590d9915f3a3008b03aa08cec01b81fbc2a292f90f65875ee0be39c',
                                    'assist.py': '80f20890804170bede9c7df6e7f05892ab4e5397f8c5f837fda49ddba64d1276'}
A, C, N, PARAMS, T0 = R.A, R.C, R.N, R.PARAMS, R.T0
CL = R.CL


def cleanup():
    shutil.rmtree(TMPROOT, ignore_errors=True)
    try:
        TMPROOT.parent.rmdir()      # 空なら
    except OSError:
        pass
