"""vd_recount.r1 と A10 の並べ方(同点 → r1_app の大きい順 → id)の小さな試験(torch を読まない)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vd_recount import r1  # noqa: E402


def order(F):
    return sorted(F, key=lambda c: (r1(F[c][0]), -r1(F[c][1]), c))


def test_round():
    assert r1(13.45) == 13.5 and r1(13.549) == 13.5 and r1(-0.05) == -0.1 and r1(25.16) == 25.2


def test_tie_r1app_then_id():
    assert order({'X': (13.46, 21.0), 'Y': (13.54, 25.0)}) == ['Y', 'X']
    assert order({'X': (13.46, 25.04), 'A': (13.54, 24.96)}) == ['A', 'X']


def test_real_values():
    F = {'SLOW-SEC1': (13.524, 26.126), 'SLOW1': (25.16, 22.749), 'SLOW': (34.4, 31.859)}
    assert order(F)[0] == 'SLOW-SEC1'
