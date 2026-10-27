"""反証 6-1 回目・安全(fix6/refute_safety): 共通の部品。直さない・装置に触れない・GPU を使わない。
fix4/review_safety/rs_common.py(本物の drive._loop を閉じたループで回す)を読み込むだけ(変えない)。
一時の場所は fix6/tmp_safety(終わったら消す)。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
RS = FIX / 'fix4' / 'review_safety'
sys.path.insert(0, str(RS))
import rs_common as R          # noqa: E402

R.TMP = FIX / 'fix6' / 'tmp_safety' / 'rs'
R.CL.TMP = R.TMP
R.SNAP_SHA['fix5_safety_before'] = {'drive.py': 'c575af1262836c3e10bfe32b2ee34293117a2144b3e568acb326ada57d59a471',
                                    'assist.py': 'b2bc818c5d8c8d410269f3aaf852d458fcc0c31d8f514002d8e34aa2c163fae6'}
A, C, N, PARAMS, T0 = R.A, R.C, R.N, R.PARAMS, R.T0
CL = R.CL

PH = []


def wrap(orig):
    def step(self, model_us, erpm, now, impact_g=0.0, *a, **kw):
        p0 = self.phase
        b0 = self.backed_s
        r = orig(self, model_us, erpm, now, impact_g, *a, **kw)
        PH.append((round(now - 100.0, 3), p0, self.phase, kw.get('hold_since') is not None,
                   round(float(kw.get('hold_back_s', 0.0)), 3), round(float(kw.get('hold_unseen_s', 0.0)), 3),
                   round(b0, 3), round(self.backed_s, 3), round(float(erpm), 1)))
        return r
    return step
