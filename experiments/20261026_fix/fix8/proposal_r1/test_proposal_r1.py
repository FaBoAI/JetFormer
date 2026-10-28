"""fix8 係 safety 直す 8-1 回目: 11/3 のあとに向けた案 qA・qC(写し。本物の jetformer/ には書かない)の形を固定する試験。
本当に入れるときは、この試験の決まりを tests/ に移す(入れる前に 11/3 の記録で 'gap' の間の ESC の速さを見る。proposal_safety_after1103.md 9 章)。
  - 写しは今の jetformer/assist.py から make_copies_r1.py で作り直せる(変えたのは Recovery._guard の back_early の分かれと reset・_start の印)
  - 見張り OFF は今とビット単位で同じ(閉じたループの 10/23 の形 × ESC 3 種)
  - 10/23 の形で、案の ON の後ろの当たりは、どの距離でも今の OFF より 0.1 m/s を超えて速くない(qA は今の ON とも)
  - qC は escape の間に前の壁まで 0.2 m より近づかない(qA は cl・human の模型で前の壁に触れる = 9 章の注意)
  - わざと壊す: 案の写しの代わりに今の ON を入れると「OFF より速くない」が落ちる
    cd ~/jetformer && PYTHONDONTWRITEBYTECODE=1 /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \\
        /home/jetson/miniconda/envs/jetracer/bin/python -m pytest -q -p no:cacheprovider -o tmp_path_retention_policy=failed \\
        --basetemp=<消せる場所> experiments/20261026_fix/fix8/proposal_r1/test_proposal_r1.py
"""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('R1_TAG', 'test_%d' % os.getpid())
import cl_r1  # noqa: E402
import make_copies_r1  # noqa: E402
import qload  # noqa: E402

DS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5)
ESCS = ('fit', 'human', 'cl')


@pytest.fixture(scope='module', autouse=True)
def _cleanup():
    yield
    cl_r1.cleanup()


def test_copies_rebuild_from_current_assist():
    res = make_copies_r1.main(check=True)          # 今の assist.py から作り直した写しが、置いてある写しと同じ
    for name in ('qA', 'qC'):
        assert res[name]['sha256'] == qload.QSHA[name]


_RUNS = {}


def runs(esc, ver, sw):
    key = (esc, ver, sw)
    if key not in _RUNS:
        sim, rec, pca, ph = cl_r1.run(cl_r1.ESC[esc], sw, ver=ver)
        t0, t1 = cl_r1.first_rec(ph)
        prof, x0, back, vpk = cl_r1.hit_profile(sim, t0, t1 if t1 is not None else 99.0, DS)
        esc_fwd = cl_r1.N - cl_r1.A.Recovery.ESCAPE_US
        te = next(t for t, s, th in pca.w if t >= t0 and abs(th - esc_fwd) < 1e-6)
        gap = 2.4 - max(r[1] for r in sim.log if te <= r[0] <= te + 0.6)
        _RUNS[key] = {'w': list(pca.w), 'prof': prof, 'back': back, 'front_gap': gap,
                      'events': [e[1]['what'] for e in rec.events if e[0] == 'recovery_guard']}
    return _RUNS[key]


@pytest.mark.parametrize('esc', ESCS)
@pytest.mark.parametrize('q', ['qA', 'qC'])
def test_off_bit_identical(esc, q):
    assert runs(esc, q, 'off')['w'] == runs(esc, 'now', 'off')['w']


def not_faster(a, b, th=0.1):
    """a の当たり(None = 当たらない)が b より th を超えて速くない(b が当たらず a が当たるのも「速い」)"""
    return a is None or (b is not None and a <= b + th)


@pytest.mark.parametrize('esc', ESCS)
@pytest.mark.parametrize('q', ['qA', 'qC'])
def test_rear_not_faster_than_off(esc, q):
    p, off = runs(esc, q, 'on')['prof'], runs(esc, 'now', 'off')['prof']
    assert all(not_faster(p[d], off[d]) for d in DS), {d: (p[d], off[d]) for d in DS}
    assert runs(esc, q, 'on')['events'][:1] == ['back_early']          # 10/23 の形として見張りが見つけている


@pytest.mark.parametrize('esc', ESCS)
def test_qA_not_faster_than_now_on(esc):
    p, on = runs(esc, 'qA', 'on')['prof'], runs(esc, 'now', 'on')['prof']
    assert all(not_faster(p[d], on[d]) for d in DS), {d: (p[d], on[d]) for d in DS}


@pytest.mark.parametrize('esc', ESCS)
def test_escape_front_gap(esc):
    assert runs(esc, 'qC', 'on')['front_gap'] >= 0.2
    if esc in ('cl', 'human'):
        assert runs(esc, 'qA', 'on')['front_gap'] < 0.1           # qA の注意(9 章): 下がりが短く、escape が前の壁に近づく


@pytest.mark.parametrize('esc', ESCS)
def test_broken_now_on_is_faster_than_off(esc):
    """わざと壊す: 案の代わりに今の ON を入れると、近い物で OFF より速い(上の試験の決まりが今の ON の形を見つける)"""
    on, off = runs(esc, 'now', 'on')['prof'], runs(esc, 'now', 'off')['prof']
    assert not all(not_faster(on[d], off[d]) for d in DS)
