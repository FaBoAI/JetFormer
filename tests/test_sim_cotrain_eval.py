"""シミュレーターと実機を混ぜた学習の評価(tools/sim_cotrain_eval.py)のテスト(numpy だけ・CPU・実データなし)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_sim_cotrain_eval.py
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import policy_zoo as pz  # noqa: E402
import rl_policies as rp  # noqa: E402
import sim_cotrain as sc  # noqa: E402
import sim_cotrain_eval as se  # noqa: E402


# ---------------------------------------------------------------- 名前・保存先
def test_name_table_reuses_1012_predictions_for_A_and_B():
    t = se.name_table()
    assert list(t) == ['A_s0', 'A_s1', 'B', 'C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'AFT_s0', 'AFT_s1']
    assert t['A_s0']['zoo'] == 'jetformer_v0_s0' and t['B']['zoo'] == 'B'
    assert t['A_s1']['run'] == sc.RUN_A % 1 and t['B']['run'] == sc.RUN_B
    assert t['CFT_s1']['run'] == sc.RUN_CFT % 1 and t['AFT_s0']['run'] == sc.RUN_AFT % 0
    assert se.new_names() == ['C_s0', 'C_s1', 'CFT_s0', 'CFT_s1', 'AFT_s0', 'AFT_s1']
    # A・B は 10/12 の予測の場所(読むだけ)、新しい 6 本は別の場所
    assert se.pred_file('A_s0') == pz.pred_path('jetformer_v0_s0')
    assert se.pred_file('B', 'last') == pz.pred_path('B', 'last')
    assert se.pert_file('A_s1') == pz.pert_path('jetformer_v0_s1')
    assert pz.zoo_dir() not in se.pred_file('C_s0').parents
    assert se.pred_file('C_s0', 'last').name == 'C_s0_last.npz'
    assert se.sim_file('A_s0').parent.name == 'sim'
    with pytest.raises(ValueError):
        se.pred_file('C_s0', 'mid')


def test_arm_names():
    assert se.arm_names('A') == ['A_s0', 'A_s1']
    assert se.arm_names('C+FT') == ['CFT_s0', 'CFT_s1']
    assert se.arm_names('B') == ['B']
    with pytest.raises(ValueError):
        se.arm_names('D')


def test_gpu_patterns_have_eval_commands():
    assert rp.is_gpu_cmd(['python', '-u', 'tools/sim_cotrain_eval.py', 'predict'])
    assert rp.is_gpu_cmd(['python', 'tools/sim_cotrain_eval.py', 'crosscheck'])
    assert not rp.is_gpu_cmd(['python', 'tools/sim_cotrain_eval.py', 'evaluate'])
    assert not rp.is_gpu_cmd(['python', 'tools/sim_cotrain_eval.py', 'figures'])


# ---------------------------------------------------------------- シミュレーターの頭
def test_heads_for_and_onehot():
    assert se.heads_for(9) == [4, 5, 8]
    assert se.heads_for(6) == [4, 5]                 # A・A+FT・B: 頭 8 が無い
    o = se.onehot(3, 8, 9)
    assert o.shape == (3, 9) and o.dtype == np.float32 and (o[:, 8] == 1).all() and o.sum() == 3


def test_masked_l1_values():
    y = np.zeros((4, 10, 2))
    p = np.zeros((4, 10, 2))
    p[:, :, 0] = 10.0          # ステア +10
    p[:, :, 1] = -4.0          # スロットル −4(前進が強い)
    p[3] = 1000.0              # mask で外す行
    m = se.masked_l1(p, y, [True, True, True, False])
    assert m['n'] == 3 and m['l1_us'] == pytest.approx(7.0) and m['l1_steer_us'] == pytest.approx(10.0)
    assert m['l1_throttle_us'] == pytest.approx(4.0) and m['fwd_bias_us'] == pytest.approx(-4.0)
    assert se.masked_l1(p, y, np.zeros(4, bool))['l1_us'] is None


# ---------------------------------------------------------------- 判定
def _vals(best, last):
    return {'best': list(best), 'last': list(last)}


def test_compare_uses_protocol_thresholds():
    a = _vals([49.146, 49.457], [49.235, 49.752])            # 10/12 の A
    s = {'best': 0.313, 'last': 0.661}
    # 両方の種が A の両方より小さく、平均の改善 > 0.626 → best は良い。last も良い → 良い
    r = se.compare(_vals([48.3, 48.4], [47.9, 48.0]), a, s)          # last の改善 1.54 > 1.322
    assert r['best']['class'] == 'better' and r['last']['class'] == 'better' and r['final_class'] == 'better'
    assert r['best']['threshold_us'] == pytest.approx(0.626) and r['last']['threshold_us'] == pytest.approx(1.322)
    assert not r['practical']
    assert r['paired_best_us'] == pytest.approx([48.3 - 49.146, 48.4 - 49.457])
    # best だけ良い → 同等に下げる
    r = se.compare(_vals([48.3, 48.4], [49.0, 49.1]), a, s)
    assert r['best']['class'] == 'better' and r['final_class'] == 'equivalent'
    # 悪い(best のまま)
    r = se.compare(_vals([50.2, 50.4], [49.5, 49.6]), a, s)
    assert r['final_class'] == 'worse'
    # 差は小さい → 同等
    r = se.compare(_vals([49.5, 49.9], [49.6, 50.0]), a, s)
    assert r['final_class'] == 'equivalent'


def test_compare_spread_is_max_of_ref_range_and_floor():
    ref = _vals([49.0, 50.0], [49.0, 50.0])                  # 基準の種の範囲 1.0 > s_A
    r = se.compare(_vals([47.0, 47.5], [47.0, 47.5]), ref, {'best': 0.313, 'last': 0.661})
    assert r['best']['ref_spread_us'] == pytest.approx(1.0) and r['best']['threshold_us'] == pytest.approx(2.0)
    assert r['final_class'] == 'better'


def test_own_spread_large():
    assert not se.own_spread_large([49.0, 49.6], 0.313)
    assert se.own_spread_large([49.0, 49.7], 0.313)


def test_speed_habit_threshold():
    r = se.speed_habit([20.0, 22.0], [9.3, 10.8])
    assert r['shift_us'] == pytest.approx(10.95) and r['moved']
    assert not se.speed_habit([12.0, 14.0], [9.3, 10.8])['moved']
    assert se.speed_habit([-1.0, 0.0], [9.3, 10.8])['moved']          # 前進が強くなる向きも数える


def test_forgetting_share():
    r = se.forgetting(40.0, 45.0, 60.0)       # C の得 20 µs のうち 5 µs を返した
    assert r['increase_us'] == pytest.approx(5.0) and r['gain_c_us'] == pytest.approx(20.0)
    assert r['share_lost'] == pytest.approx(0.25)
    assert se.forgetting(60.0, 61.0, 50.0)['share_lost'] is None     # C が A より悪ければ割合は出さない


def test_conclusion_rules():
    c = se.conclusion('equivalent', 'equivalent', 'equivalent')
    assert c['key'] == 'no_gain' and not c['recommend_change'] and c['candidate'] is None
    assert '上がらなかった' in c['text'] and 'モデル B のまま' in c['text']
    c = se.conclusion('better', 'equivalent', 'better', practical_cft=False)
    assert c['key'] == 'sim_helped' and c['candidate'] == 'C+FT' and not c['recommend_change']
    assert '10 µs 未満' in c['text']
    c = se.conclusion('better', 'equivalent', 'better', practical_cft=True)
    assert c['recommend_change']
    c = se.conclusion('better', 'equivalent', 'equivalent')
    assert c['key'] == 'not_separable' and '区別できない' in c['text']
    c = se.conclusion('equivalent', 'better', 'equivalent')
    assert c['key'] == 'mix_only' and c['candidate'] == 'C'
    c = se.conclusion('worse', 'equivalent', 'worse')
    assert c['key'] == 'worse' and 'C+FT' in c['text']
    c = se.conclusion('better', 'equivalent', 'better', practical_cft=True, ok_cft=False)   # 崩れ・ゲート不合格は推奨しない
    assert c['candidate'] is None and not c['recommend_change']


# ---------------------------------------------------------------- 図(評価の結果があるときだけ)
def test_figures_size_and_font(tmp_path):
    """図は 2160×1350・文字 16 pt 以上(evaluation.json が無い環境では飛ばす)"""
    import json
    p = se.REPORT_DIR / 'evaluation.json'
    if not p.exists():
        pytest.skip('evaluation.json がない')
    import sim_cotrain_figures as sf
    from PIL import Image
    res = json.loads(p.read_text(encoding='utf-8'))
    paths = sf.make_all(res, tmp_path)
    assert len(paths) == 5
    for q in paths:
        assert Image.open(q).size == (2160, 1350)
    plt = sf._setup()
    for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize', 'axes.labelsize'):
        assert float(plt.rcParams[k]) >= 16
