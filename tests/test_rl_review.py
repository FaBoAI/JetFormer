"""オフライン強化学習のレビューで足した確認(tools/rl_review.py)と、rl_evaluate.py・rl_policies.py の小さな変更のテスト。
GPU・データ不要(CPU、数秒)

    cd ~/jetformer && python -m pytest -q tests/test_rl_review.py
"""

import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import rl_evaluate as ev  # noqa: E402
import rl_policies as rp  # noqa: E402
import rl_review as rv  # noqa: E402


def test_sign_test_matches_binomial():
    r = rv.sign_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, 0])
    assert (r['pos'], r['n']) == (13, 14)
    assert r['p'] == pytest.approx(2 * (1 + 14) / 2 ** 14, abs=1e-4)
    assert rv.sign_test([1, -1])['p'] == 1.0
    assert rv.sign_test([0, 0])['p'] is None


def test_range_summary():
    s = rv.range_summary([[5, 1, 9], [2, 0.5, 4], [None, None, None]])
    assert s['points'] == [2, 5] and s['all_pos'] and not s['all_neg'] and s['any_excludes_zero']
    s = rv.range_summary([[-5, -9, -1], [-1, -3, 1]])
    assert not s['all_neg'] and s['any_excludes_zero']
    assert rv.range_summary([])['points'] is None


def test_minus_group_mean_removes_scene_offset():
    """シーンごとの一定のずれは消え、窓に固有の差だけが残る。base に入らない行もそのシーンの平均で引く"""
    d = np.array([10, 10, 14, -3, -3, -3, 7.0])
    g = np.array(['s1', 's1', 's1', 's2', 's2', 's2', 's3'])
    base = np.array([1, 1, 1, 1, 1, 0, 0], bool)
    out = rv.minus_group_mean(d, g, base)
    assert out[:3] == pytest.approx([-4 / 3, -4 / 3, 8 / 3])
    assert out[3:6] == pytest.approx([0, 0, 0])
    assert np.isnan(out[6])                      # s3 は base の行が無い


def test_location_adjusted_share():
    """空転が「良い」の多い場所に集まっているだけなら、場所でそろえると差は 0"""
    loc = np.array([2] * 10 + [1] * 10)
    good = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0] + [0, 0, 0, 1, 0, 0, 0, 1, 0, 0], bool)
    flag = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 10, bool)      # 場所 2 の2枚だけ
    ref = np.ones(20, bool)
    r = rv.location_adjusted_share(flag, good, loc, ref)
    assert r['n'] == 2 and r['observed'] == 1.0
    assert r['expected'] == pytest.approx(6 / 8, abs=1e-4)              # 場所 2 の空転でない 8枚のうち 6枚が良い
    assert r['excess'] == pytest.approx(0.25, abs=1e-4)
    assert r['by_loc'][2] == (2, 1.0, 0.75)
    assert rv.location_adjusted_share(np.zeros(3, bool), good[:3], loc[:3], ref[:3])['n'] == 0


def test_auroc_matches_pairwise_count():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, size=40).astype(float)        # 同じ値を多く含む
    y = rng.random(40) < 0.4
    pos, neg = s[y], s[~y]
    want = np.mean([(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)])
    assert rv.auroc(s, y) == pytest.approx(want)
    assert rv.auroc(s, np.zeros(40, bool)) is None


def test_human_only_drop_keeps_val_like_c():
    drop_c = np.array([0, 1, 0, 0, 0, 0], bool)
    source = np.array([0, 0, 1, 1, 0, 1])
    split = np.array(['train', 'train', 'train', 'val', 'val', 'val'])
    got = rv.human_only_drop(drop_c, source, split)
    assert got.tolist() == [False, True, True, False, False, False]


def test_disable_engine_is_idempotent(tmp_path):
    (tmp_path / 'jetformer_fp16.engine').write_bytes(b'x')
    assert not rp.engine_disabled(tmp_path)
    assert rv.disable_engine(tmp_path) == 'renamed'
    assert not (tmp_path / 'jetformer_fp16.engine').exists()
    assert rp.engine_disabled(tmp_path)
    assert rv.disable_engine(tmp_path) == 'already'
    assert rv.disable_engine(tmp_path / 'none') == 'missing'
    (tmp_path / 'jetformer_fp16.engine').write_bytes(b'y')
    with pytest.raises(SystemExit):
        rv.disable_engine(tmp_path)


def test_disabled_engine_is_not_listed_by_drive(tmp_path, monkeypatch):
    """名前を変えたエンジンは、管理画面の走行の一覧(drive.model_info)に出ない"""
    from jetformer import drive
    d = tmp_path / 'run1'
    d.mkdir()
    (d / 'jetformer_fp16.engine').write_bytes(b'x')
    (d / 'model_config.json').write_text('{"num_modes": 6, "trained_modes": [4, 5]}')
    monkeypatch.setattr(drive, 'RUNS_DIR', tmp_path)
    assert drive.model_info('run1') is not None
    rv.disable_engine(d)
    assert drive.model_info('run1') is None


def _table(n_ok, a_ok, b_ok):
    """verdict に渡す最小の res(項目 2〜5 の合否だけ)"""
    pas = {'a_good': a_ok, 'b': b_ok, 'n_good': n_ok, 'a_base': a_ok, 'n_base': n_ok}
    res = {'item1_value': {'pass': False}, 'item2_r1': {'vs': {}}, 'item3_de': {'vs': {}}, 'item4_collapse': {'vs': {}},
           'item5_labels': {'labels': {'I': {'pass': a_ok['5_labels']}, 'I_neg': {'pass': n_ok['5_labels']}}}}
    for x, ok in pas.items():
        res['item2_r1']['vs']['%s-c' % x] = {'pass': ok['2_r1']}
        res['item3_de']['vs']['%s-c' % x] = {'pass': ok['3_de']}
        res['item4_collapse']['vs']['%s-B' % x] = {'pass': ok['4_collapse']}
    return res


def test_verdict_gate_discriminates_needs_a_separating_item():
    """(n) が落ちても、(a)(b) も同じ項目で落ちたなら「見分けた」にしない(先に決めた規則は negative_control_rule_met に残す)"""
    fail = {'2_r1': False, '3_de': False, '4_collapse': True, '5_labels': False}
    v = ev.verdict(_table(fail, fail, fail))
    assert v['negative_control_rule_met'] is True
    assert v['gate_discriminates'] is False and v['separating_items'] == []
    a_ok = dict(fail, **{'3_de': True})
    v = ev.verdict(_table(fail, a_ok, fail))
    assert v['gate_discriminates'] is True and v['separating_items'] == ['3_de']


def test_pred_path_names():
    assert ev.pred_path('c').name == 'pred_c.npz'
    assert ev.pred_path('c', 'last').name == 'pred_c_last.npz'
    with pytest.raises(ValueError):
        ev.pred_path('c', 'mid')
    assert ev.config_of('h')[0] == ev.H_RUN and ev.config_of('B')[0] == ev.BASE_RUN


def test_load_state_dict_reads_best_and_last(tmp_path):
    torch = pytest.importorskip('torch')
    sd = {'w': torch.ones(2)}
    torch.save(sd, tmp_path / 'best.pth')
    torch.save({'model': sd, 'optimizer': {'state': {}}, 'epoch': 3}, tmp_path / 'last.pth')
    for name in ('best.pth', 'last.pth'):
        got = ev.load_state_dict(tmp_path / name)
        assert list(got) == ['w'] and float(got['w'].sum()) == 2.0


def test_gpu_patterns_cover_rl_evaluate_predict():
    assert rp.is_gpu_cmd(['python', '-u', 'tools/rl_evaluate.py', 'predict', '--ckpt', 'last'])
    assert rp.is_gpu_cmd(['python', 'tools/rl_evaluate.py', 'engine-check'])
    assert not rp.is_gpu_cmd(['python', 'tools/rl_evaluate.py', 'evaluate'])
    assert not rp.is_gpu_cmd(['python', 'tools/rl_review.py', 'analyze'])
