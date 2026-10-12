"""方策の後処理(jetformer/policies/post.py)と予測の重ね合わせ(ensemble.py)のテスト(numpy だけ・CPU)

    cd ~/jetformer && python -m pytest -q tests/test_policy_post.py
"""

import numpy as np
import pytest

from jetformer.policies import ensemble, post

FRAME_NS = int(1e9 / 30)


def _pred(v):
    """全部の行が同じ値でなく、行ごとに違う予測 [10,2](行 s = v + s)"""
    return np.stack([np.arange(10) + v, np.arange(10) + v + 100], axis=1).astype(np.float32)


# ---------------------------------------------------------------- 重ね合わせ
def test_ensemble_off_returns_copy():
    e = ensemble.TemporalEnsembler()
    p = _pred(1500)
    out = e.push(0, p, 0, 0.5)
    assert np.array_equal(out, p) and out is not p and out.dtype == np.float32


def test_ensemble_newest_is_heaviest_and_aligned_in_time():
    e = ensemble.TemporalEnsembler()
    e.push(0, _pred(1000), 2, 0.5)
    out = e.push(FRAME_NS, _pred(2000), 2, 0.5)
    w0, w1 = 1.0, np.exp(-0.5)
    # 行 s: 新しい予測の s 行と、1 フレーム前の予測の s+1 行(s ≤ 8)
    expect0 = (w0 * (2000 + 0) + w1 * (1000 + 1)) / (w0 + w1)
    assert out[0, 0] == pytest.approx(expect0, rel=1e-6)
    assert out[9, 0] == pytest.approx(2000 + 9)                       # 最後の行は新しい予測だけ
    assert abs(out[0, 0] - 2000) < abs(out[0, 0] - 1001)               # 新しいほど重い


def test_ensemble_counts_frames_by_time_and_drops_old():
    e = ensemble.TemporalEnsembler()
    e.push(0, _pred(1000), 1, 0.0)
    out = e.push(3 * FRAME_NS, _pred(2000), 1, 0.0)                    # 2 フレーム抜けた: j = 3 > j_max は使わない
    assert np.allclose(out, _pred(2000))
    e = ensemble.TemporalEnsembler()
    e.push(0, _pred(1000), 3, 0.0)
    out = e.push(2 * FRAME_NS, _pred(2000), 3, 0.0)                    # j = 2(時刻で数える)
    assert out[0, 0] == pytest.approx((2000 + 1002) / 2)
    e.reset()
    assert np.allclose(e.push(5 * FRAME_NS, _pred(3000), 3, 0.0), _pred(3000))


# ---------------------------------------------------------------- 後処理の作り方
class FakePost(object):
    kind = 'fake'

    def __init__(self, pp, ref):
        self.pp, self.ref = pp, ref
        self.resets = 0
        self.calls = []

    def reset(self):
        self.resets += 1

    def step(self, outputs, v_now, params, t_ns):
        self.calls.append((outputs, v_now, params.get('tcp_mode'), t_ns))
        pred = outputs['actions'] * 500 + 1500
        pred[:, 1] = self.ref[1] - 100 * v_now                       # スロットルだけ変える
        if params.get('boom'):
            raise ValueError('boom')
        if params.get('nan'):
            pred[0, 0] = np.nan
        return pred.astype(np.float32), {'mode': int(params.get('tcp_mode', 0)), 'branch': 'fake', 'v_hat': 1.5,
                                         'v_cmd': 1.2, 'v_curv': np.nan, 'cap': 0.0}


@pytest.fixture
def fake_kind(monkeypatch):
    import types
    import sys
    mod = types.ModuleType('jetformer_test_fake_post')
    mod.FakePost = FakePost
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(post.KINDS, 'fake', (mod.__name__, 'FakePost'))
    return 'fake'


def test_make_without_or_unknown_postprocess():
    assert post.make({'num_modes': 6}, (1443, 1498)) == (None, '')
    obj, note = post.make({'postprocess': {'kind': 'no_such_kind'}}, (1443, 1498))
    assert obj is None and '操作の枝' in note
    obj, note = post.make({'postprocess': 'garbage'}, (1443, 1498))
    assert obj is None and note


def test_make_with_fake_kind(fake_kind):
    obj, note = post.make({'postprocess': {'kind': fake_kind, 'version': 1}}, (1443.0, 1498.0))
    assert isinstance(obj, FakePost) and obj.ref == (1443.0, 1498.0) and obj.pp['version'] == 1
    assert note


def test_make_when_controller_module_breaks(monkeypatch):
    monkeypatch.setitem(post.KINDS, 'broken', ('jetformer.policies.no_such_controller', 'X'))
    obj, note = post.make({'postprocess': {'kind': 'broken'}}, (1443, 1498))
    assert obj is None and '操作の枝' in note


# ---------------------------------------------------------------- Stage(走行と評価の共通の手順)
def _res(v=0.1):
    return {'actions': np.full((1, 10, 2), v, np.float32), 'waypoints': np.zeros((1, 10, 2), np.float32)}


def test_stage_without_post_is_identity():
    st = post.Stage({'num_modes': 6}, (1443, 1498))
    res = _res()
    mp = st.model_pred(res)
    assert np.array_equal(mp, res['actions'][0] * 500.0 + 1500.0)
    pred, info = st.apply(res, mp, 1.0, {'ensemble_j': 0}, 0)
    assert pred is mp and info is None
    pred, _ = st.apply(res, mp, 1.0, {}, FRAME_NS)                      # パラメータが無くても既定(重ね合わせなし)
    assert pred is mp


def test_stage_runs_post_then_ensemble(fake_kind):
    st = post.Stage({'postprocess': {'kind': fake_kind}}, (1443, 1498))
    mp = st.model_pred(_res())
    pred, info = st.apply(_res(), mp, 2.0, {'tcp_mode': 1, 'ensemble_j': 1, 'ensemble_m': 0.0}, 0)
    assert info['branch'] == 'fake' and np.allclose(pred[:, 1], 1498 - 200)
    outputs, v_now, mode, t_ns = st.post.calls[-1]
    assert set(outputs) == {'actions', 'waypoints'} and outputs['actions'].shape == (10, 2)   # バッチの次元を外す
    assert v_now == 2.0 and mode == 1
    pred2, _ = st.apply(_res(), mp, 0.0, {'tcp_mode': 1, 'ensemble_j': 1, 'ensemble_m': 0.0}, FRAME_NS)
    assert pred2[0, 1] == pytest.approx(((1498 - 0) + (1498 - 200)) / 2)          # 後処理の後に重ね合わせ
    st.reset()
    assert st.post.resets == 1


def test_stage_errors_become_post_error(fake_kind):
    st = post.Stage({'postprocess': {'kind': fake_kind}}, (1443, 1498))
    mp = st.model_pred(_res())
    with pytest.raises(post.PostError, match='後処理でエラー'):
        st.apply(_res(), mp, 0.0, {'boom': True}, 0)
    with pytest.raises(post.PostError, match='数値'):
        st.apply(_res(), mp, 0.0, {'nan': True}, 0)


def test_replay_resets_on_gaps_and_scenes(fake_kind):
    n = 6
    t = np.array([0, 1, 2, 6, 7, 8]) * FRAME_NS                      # 3 つ抜けた所で reset
    outputs = {'actions': np.zeros((n, 10, 2), np.float32)}
    preds, model_preds, infos = post.replay({'postprocess': {'kind': fake_kind}}, (1443, 1498), outputs,
                                            erpm=np.full(n, -2000.0), t_ns=t, params={'tcp_mode': 0})
    assert preds.shape == (n, 10, 2) and np.allclose(model_preds, 1500.0)
    assert np.allclose(preds[:, :, 1], 1498 - 100 * 1.0)             # v_now = -eRPM × 0.5e-3 = 1.0 m/s
    assert all(i['branch'] == 'fake' for i in infos)
    preds, _, infos = post.replay({}, (1443, 1498), outputs, np.zeros(n), t, {'ensemble_j': 0},
                                  scene=np.array(['a', 'a', 'a', 'b', 'b', 'b']))
    assert np.allclose(preds, 1500.0) and infos == [None] * n


def test_speed_of_sign():
    assert post.speed_of(-4000) == pytest.approx(2.0)                  # eRPM が負 = 前進
    assert post.speed_of(1000) == pytest.approx(-0.5)


def test_stage_reapplies_cap_after_ensemble():
    """上限のブレーキ(中立 + 150 µs)が、重ね合わせで前のフレームの前進の行と平均されて弱まらない(レビュー 2026-10-12)。
    本物の TCP の制御器で: 1 フレーム目は上限の手前(前進)、2 フレーム目は上限 + 0.3 m/s を超えてブレーキ"""
    from jetformer.policies import controller as C
    offs = np.arange(-575.0, 576.0, 50.0)
    k = np.maximum.accumulate(np.clip(0.004 * offs, -1.5, 1.5)) + np.arange(len(offs)) * 1e-6
    pp = {'kind': 'tcp_controller', 'version': 1, 'speed_bins': [0.5 * i for i in range(10)],
          'steer_table': {'offset_us': offs.tolist(), 'bands_mps': [2.0, 3.0], 'curv': [k.tolist()] * 3},
          'throttle_ff': {'c0': 1530.2, 'c1': -82.7, 'c2': -25.4, 'tau_s': 0.33, 'fwd_max_us': 300.0}}
    ref = (1441.0, 1496.0)
    st = post.Stage({'postprocess': pp}, ref)
    assert isinstance(st.post, C.TCPPost) and not st.failed
    res = {'actions': np.full((1, 10, 2), -0.6, np.float32), 'waypoints': np.zeros((1, 10, 2), np.float32),
           'speed_logits': np.zeros((1, 10), np.float32)}
    params = {'tcp_mode': 0, 'speed_cap': 1.0, 'ensemble_j': 3, 'ensemble_m': 0.0}
    mp = st.model_pred(res)
    st.apply(res, mp, 0.5, params, 0)                                   # 上限の手前: 前進
    pred, info = st.apply(res, mp, 2.0, params, FRAME_NS)               # 2.0 > 1.0 + 0.3: ブレーキ
    assert info['thr_cap'] == pytest.approx(ref[1] + 150.0)
    assert np.all(pred[:, 1] >= ref[1] + 150.0 - 1e-3)                  # 重ね合わせの後も全部の行がブレーキ
    # 上限なしなら重ね合わせはそのまま(掛け直さない)
    st2 = post.Stage({'postprocess': pp}, ref)
    p0 = dict(params, speed_cap=0.0)
    st2.apply(res, mp, 0.5, p0, 0)
    pred2, info2 = st2.apply(res, mp, 2.0, p0, FRAME_NS)
    assert np.isnan(info2['thr_cap']) and np.allclose(pred2, mp)


def test_stage_marks_failed_post():
    assert post.Stage({'postprocess': {'kind': 'no_such_kind'}}, (1443, 1498)).failed is True
    assert post.Stage({'num_modes': 6}, (1443, 1498)).failed is False
    assert post.cap_limit(None) is None and post.cap_limit({'thr_cap': float('nan')}) is None
    assert post.cap_limit({'thr_cap': 1600.0}) == 1600.0 and post.cap_limit({'thr_cap': 'x'}) is None
