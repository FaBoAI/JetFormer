"""方策の比較の評価の道具(tools/policy_zoo.py)のテスト(numpy だけ・CPU・実データなし。合成の数字で判定の規則を確かめる)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_policy_zoo.py
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import policy_zoo as pz  # noqa: E402
import rl_policies as rp  # noqa: E402

POLS = pz.POLICIES


def _rules():
    runs = {'%s_s%d' % (p, s): 'run-%s-s%d' % (p, s) for p in POLS for s in (0, 1)}
    return {'runs': {'policy': runs, 'model_B': 'run-B'}}


# ---------------------------------------------------------------- 名前
def test_run_names_order_and_policy_of():
    names = pz.run_names(_rules())
    assert list(names)[0] == 'B' and names['B'] == 'run-B'
    assert list(names)[1:3] == ['jetformer_v0_s0', 'jetformer_v0_s1'] and len(names) == 11
    assert pz.policy_of('jetformer_v0_s1') == 'jetformer_v0' and pz.policy_of('act_s0') == 'act' and pz.policy_of('B') == 'B'
    assert pz.seed_names('tcp') == ['tcp_s0', 'tcp_s1']


def test_load_rules_missing_stops(tmp_path):
    with pytest.raises(SystemExit):
        pz.load_rules(tmp_path / 'nothing.json')


def test_gpu_patterns_have_zoo_commands():
    for k in ('policy_zoo.py predict', 'policy_zoo.py engine-check', 'benchmark_model.py'):
        assert k in rp.GPU_PATTERNS
    assert rp.is_gpu_cmd(['python', 'tools/policy_zoo.py', 'predict'])
    assert not rp.is_gpu_cmd(['python', 'tools/policy_zoo.py', 'evaluate'])


# ---------------------------------------------------------------- ずれの試験
def _img():
    return (np.arange(224)[:, None, None] * np.ones((1, 224, 3))).astype(np.uint8)     # 行の番号が値


def test_perturb_image_shift_duplicates_edge_rows():
    img, tof = _img(), np.arange(42 * 54, dtype=np.uint16).reshape(42, 54)
    d, t = pz.perturb('img_down3', img, tof)
    assert d.shape == img.shape and d.dtype == np.uint8
    assert (d[:4, 0, 0] == [0, 0, 0, 0]).all() and d[4, 0, 0] == 1 and d[-1, 0, 0] == 220
    assert np.array_equal(t, tof)
    u, _ = pz.perturb('img_up3', img, tof)
    assert u[0, 0, 0] == 3 and (u[-4:, 0, 0] == 223).all()


def test_perturb_brightness_is_clipped():
    img = np.full((224, 224, 3), 250, np.uint8)
    img[0, 0] = 10
    b, _ = pz.perturb('bright_1.2', img, np.zeros((42, 54), np.uint16))
    assert b.dtype == np.uint8 and b[5, 5, 0] == 255 and b[0, 0, 0] == 12
    d, _ = pz.perturb('bright_0.8', img, np.zeros((42, 54), np.uint16))
    assert d[5, 5, 0] == 200 and d[0, 0, 0] == 8


def test_perturb_tof_down_makes_top_row_invalid():
    tof = np.full((42, 54), 7, np.uint16)
    tof[-1] = 9
    img, t = pz.perturb('tof_down1', _img(), tof)
    assert (t[0] == 0).all() and (t[1:] == 7).all() and np.array_equal(img, _img())
    with pytest.raises(ValueError):
        pz.perturb('blur', _img(), tof)


def test_pert_rows_every_third_frame_of_clean():
    frame = np.arange(12)
    clean = np.ones(12, bool)
    clean[3] = False
    assert pz.pert_rows(frame, clean).tolist() == [0, 6, 9]


# ---------------------------------------------------------------- 主指標
def test_l1_rows_and_delta():
    y = np.zeros((2, 10, 2))
    p = np.ones((2, 10, 2)) * np.array([[[2.0, 4.0]], [[0.0, 0.0]]])
    assert pz.l1_rows(p, y).tolist() == [3.0, 0.0]
    assert pz.delta_of([49.1, 49.3, 49.2]) == 10.0
    assert pz.delta_of([40.0, 55.0, 45.0]) == 15.0              # v0 の3本の範囲が広ければ δ も広い


def test_classify_primary_good_bad_equivalent():
    r = [50.0, 50.4]
    assert pz.classify_primary([38.0, 39.0], r, 10.0)['class'] == 'good'
    assert pz.classify_primary([39.0, 51.0], r, 10.0)['class'] == 'equivalent'     # 平均は良いが範囲が重なる
    assert pz.classify_primary([45.0, 46.0], r, 10.0)['class'] == 'equivalent'     # δ より小さい
    assert pz.classify_primary([61.0, 62.0], r, 10.0)['class'] == 'bad'
    c = pz.classify_primary([51.0, 51.4], r, 10.0)
    assert c['class'] == 'equivalent' and abs(c['diff'] - 1.0) < 1e-9


def test_final_class_downgrades_good_when_last_differs():
    assert pz.final_primary_class('good', 'equivalent') == 'equivalent'
    assert pz.final_primary_class('good', 'good') == 'good'
    assert pz.final_primary_class('bad', 'equivalent') == 'bad'
    assert pz.final_primary_class('equivalent', 'good') == 'equivalent'


def test_consistent_direction():
    r = {'best': [49.0, 49.2], 'last': [49.1, 49.3]}
    assert pz.consistent_direction({'best': [50.0, 50.5], 'last': [50.1, 49.9]}, r) == 'worse'
    assert pz.consistent_direction({'best': [50.0, 49.1], 'last': [50.1, 49.9]}, r) is None
    assert pz.consistent_direction({'best': [48.0, 48.5], 'last': [48.1, 48.9]}, r) == 'better'


# ---------------------------------------------------------------- R1・DE(D1〜D3)
def _passes(n=17, seed=0):
    return np.random.default_rng(seed).normal(1300.0, 30.0, n)


def test_judge_pass_item_good_needs_all_parts():
    base = _passes()
    idx = pz.bootstrap_index(len(base), 4000, 0)
    R = np.stack([base, base + 1.0])
    noise = np.random.default_rng(1).normal(0, 1.0, (2, len(base)))
    P = R.mean(0) + 15.0 + noise                                  # 全部の通過で約 15 µs 遅い
    adj = np.full(len(base), 15.0) + noise.mean(0)                # 場所に固有の差(差の差)も 15 µs
    j = pz.judge_pass_item(P, R, adj, idx, 0.99375, 0.95, 10.0)
    assert j['good'] and not j['bad'] and j['n_pass'] == 17
    assert len(j['d_pairs']) == 2 and len(j['d_pairs'][0]) == 2
    # 全体のずれだけ(差の差は 0 のまわり)なら「良い」にしない(D2)
    j2 = pz.judge_pass_item(P, R, noise.mean(0), idx, 0.99375, 0.95, 10.0)
    assert not j2['good'] and not j2['good_parts']['adj_ci_lower_gt0']
    # 1 組でも 10 µs 未満なら「良い」にしない(D1)
    P3 = R.mean(0) + 6.0 + noise * 0.1
    j3 = pz.judge_pass_item(P3, R, np.full(len(base), 6.0), idx, 0.99375, 0.95, 10.0)
    assert j3['good_parts']['ci_lower_gt0'] and not j3['good_parts']['all_pairs_ge_min'] and not j3['good']
    # 種の片方が逆向きなら「良い」にしない
    P4 = np.stack([R.mean(0) + 40.0, R.mean(0) - 2.0])
    j4 = pz.judge_pass_item(P4, R, np.full(len(base), 19.0), idx, 0.99375, 0.95, 10.0)
    assert not j4['good_parts']['each_seed_gt0'] and not j4['good']


def test_judge_pass_item_bad_and_nan_windows():
    base = _passes(14, 3)
    idx = pz.bootstrap_index(len(base), 2000, 0)
    R = np.stack([base, base])
    P = R - 12.0 + np.random.default_rng(2).normal(0, 1, (2, 14))
    adj = np.full(14, -12.0)
    adj[3] = np.nan                                               # 窓にフレームの無い通過は差の差から除く
    j = pz.judge_pass_item(P, R, adj, idx, 0.99375, 0.95, 10.0)
    assert j['bad'] and not j['good'] and j['ci_bad'][2] < 0
    one = pz.judge_pass_item(P[:1], R, adj, idx, 0.99375, 0.95, 10.0)   # 陰性対照の形(1 本 × 基準 2 本)
    assert len(one['d_seed']) == 1 and len(one['d_pairs'][0]) == 2


def test_ci_of_uses_given_indices():
    v = np.arange(10.0)
    idx = pz.bootstrap_index(10, 500, 0)
    a = pz.ci_of(v, idx, 0.95)
    b = pz.ci_of(v, idx, 0.95)
    assert a == b and a[0] == 4.5 and a[1] < 4.5 < a[2]
    assert pz.ci_of(v, idx, 0.99375)[1] <= a[1]                  # 広い水準ほど区間が広い


# ---------------------------------------------------------------- 崩れ・ずれ
def test_judge_collapse_parts():
    c, parts, drop = pz.judge_collapse(196.0, 200.0, -10.0, 0.98, 0.985)
    assert not c and all(parts.values()) and abs(drop - 0.02) < 1e-9
    c, parts, _ = pz.judge_collapse(185.0, 200.0, -10.0, 0.98, 0.985)
    assert c and not parts['straight']
    c, parts, _ = pz.judge_collapse(200.0, 200.0, -31.0, 0.98, 0.985)
    assert c and not parts['fwd_bias']
    c, parts, _ = pz.judge_collapse(200.0, 200.0, 0.0, 0.97, 0.985)
    assert c and not parts['sign_acc']


def test_judge_robust():
    assert pz.judge_robust([3.0, 3.5], 5.0)
    assert not pz.judge_robust([3.0, 5.5], 5.0)                  # 種の片方が R より悪い
    assert not pz.judge_robust([4.5, 4.4], 5.0)                  # 0.8 倍より大きい


# ---------------------------------------------------------------- 判定
def _inputs(**over):
    pols = POLS
    d = dict(gate={p: True for p in pols}, primary={p: 'equivalent' for p in pols}, collapse={p: False for p in pols},
             r1={p: {'good': False, 'bad': False} for p in pols}, de={p: {'good': False, 'bad': False} for p in pols},
             robust={p: False for p in pols}, nc_invalidates=False,
             lbar={'jetformer_v0': 49.2, 'act': 50.4, 'tcp': 49.4, 'msfusion': 49.3, 'dinov2': 50.5},
             speed={'jetformer_v0': 2.37, 'act': 2.55, 'tcp': 2.6, 'msfusion': 3.19, 'dinov2': 5.05}, delta=10.0)
    d.update(over)
    return d


def _decide(**over):
    a = _inputs(**over)
    return pz.decide(POLS, a['gate'], a['primary'], a['collapse'], a['r1'], a['de'], a['robust'], a['nc_invalidates'],
                     a['lbar'], a['speed'], a['delta'])


def test_decide_rule4_when_nothing_beats_v0():
    v = _decide()
    assert v['rule'] == 4 and v['best'] == 'jetformer_v0'
    assert set(v['candidates']) == {'act', 'tcp', 'msfusion', 'dinov2'}


def test_pick_by_speed_d11():
    lbar = {'a': 40.0, 'b': 41.0, 'c': 48.0}
    assert pz.pick_by_speed(['a', 'b'], lbar, {'a': 3.0, 'b': 2.5}, 10.0) == 'b'       # 0.5 ms 速い・L̄ の差 < δ/2
    assert pz.pick_by_speed(['a', 'b'], lbar, {'a': 3.0, 'b': 2.8}, 10.0) == 'a'       # 0.2 ms は揺れの内
    assert pz.pick_by_speed(['a', 'c'], lbar, {'a': 5.0, 'c': 2.0}, 10.0) == 'a'       # L̄ の差 8 ≥ δ/2
    assert pz.pick_by_speed([], lbar, {}, 10.0) is None


def test_decide_rule2_picks_lowest_lbar():
    prim = {p: 'equivalent' for p in POLS}
    prim.update(act='good', tcp='good')
    v = _decide(primary=prim)
    assert v['rule'] == 2 and v['best'] == 'tcp' and set(v['rule2']) == {'act', 'tcp'}
    gate = {p: True for p in POLS}
    gate['tcp'] = False                                           # ゲートに落ちた方策は候補にならない
    v = _decide(primary=prim, gate=gate)
    assert v['best'] == 'act' and 'tcp' not in v['candidates']
    col = {p: False for p in POLS}
    col['act'] = True
    v = _decide(primary=prim, gate=gate, collapse=col)
    assert v['rule'] == 4 and v['best'] == 'jetformer_v0'


def test_decide_rule3_and_negative_control_invalidates():
    r1 = {p: {'good': False, 'bad': False} for p in POLS}
    de = {p: {'good': False, 'bad': False} for p in POLS}
    r1['msfusion'] = {'good': True, 'bad': False}
    de['act'] = {'good': True, 'bad': False}
    rob = {p: False for p in POLS}
    rob['act'] = True
    v = _decide(r1=r1, de=de, robust=rob)
    assert v['rule'] == 3 and v['best'] == 'act' and v['score'] == {'msfusion': 1, 'act': 2}
    v = _decide(r1=r1, de=de, robust=rob, nc_invalidates=True)
    assert v['rule'] == 4 and v['best'] == 'jetformer_v0' and set(v['rule3']) == {'msfusion', 'act'}
    r1['act'] = {'good': False, 'bad': True}                       # R1 が「悪い」なら候補から外れる
    v = _decide(r1=r1, de=de, robust=rob)
    assert 'act' not in v['candidates'] and v['best'] == 'msfusion'


def test_decide_bad_primary_is_not_candidate():
    prim = {p: 'equivalent' for p in POLS}
    prim['dinov2'] = 'bad'
    assert 'dinov2' not in _decide(primary=prim)['candidates']


# ---------------------------------------------------------------- ゲート
def _check(acc=1.0, med=2.4, p99=2.7, drivable=True, io=True, **extra):
    a = {'actions_max_us': acc, 'finite': True}
    a.update(extra)
    return {'check_io': {'ok': io}, 'model_info': {'drivable': drivable}, 'accuracy': a,
            'latency_ms': {'run_numpy_median': med, 'run_numpy_p99': p99}}


def test_gate_of_items():
    ok, parts = pz.gate_of(_check(), {'peak_mem_mb': 1300}, True, 'act')
    assert ok and set(parts) == {'engine', 'drivable', 'accuracy', 'speed', 'train_mem', 'finite'}
    assert not pz.gate_of(_check(acc=2.85), {'peak_mem_mb': 400}, True, 'dinov2')[0]
    assert not pz.gate_of(_check(med=6.2), {'peak_mem_mb': 400}, True, 'dinov2')[0]
    assert not pz.gate_of(_check(), {'peak_mem_mb': 5000}, True, 'act')[0]
    assert not pz.gate_of(_check(), {'peak_mem_mb': 1300}, False, 'act')[0]
    assert not pz.gate_of(None, {'peak_mem_mb': 1300}, True, 'act')[0]
    ok, parts = pz.gate_of(_check(waypoints_max_m=0.005, speed_exp_max_mps=0.003), {'peak_mem_mb': 1300}, True, 'tcp')
    assert ok and parts['tcp_waypoints'] and parts['tcp_speed']
    assert not pz.gate_of(_check(waypoints_max_m=0.03, speed_exp_max_mps=0.003), {'peak_mem_mb': 1}, True, 'tcp')[0]
    assert not pz.gate_of(_check(), {'peak_mem_mb': 1300}, True, 'tcp')[0]      # tcp の追加の出力の確認が無い


# ---------------------------------------------------------------- そのほか
def test_boundary_mask_validation_side():
    scene = np.array(['20261010-152405-0002'] * 5 + ['20261010-152707-0001'] * 3 + ['x'] * 2)
    t = np.array([0, 0.5e6, 2e6, 3.5e6, 4.2e6, 0, 1e6, 3e6, 0, 1], np.int64)
    m = pz.boundary_mask(scene, t)
    assert m.tolist() == [True, True, False, True, True, False, False, True, False, False]


def test_softmax_expectation_and_wp_errors():
    bins = [0.0, 1.0, 2.0]
    assert abs(pz.softmax_expectation(np.array([0.0, 0.0, 0.0]), bins) - 1.0) < 1e-12
    assert abs(pz.softmax_expectation(np.array([[0.0, -50.0, 50.0]]), bins)[0] - 2.0) < 1e-9
    pred = np.zeros((1, 10, 2))
    lab = np.zeros((1, 10, 2))
    lab[0, :, 1] = 0.3
    lab[0, -1] = [0.0, 0.4]
    ade, fde = pz.wp_errors(pred, lab)
    assert abs(ade[0] - (0.3 * 9 + 0.4) / 10) < 1e-12 and abs(fde[0] - 0.4) < 1e-12


def test_tcp_replay_uses_tcppost_and_mode0_is_actions():
    """tcp の順の評価は走行と同じ post.Stage(TCPPost)を使う。tcp_mode 0・上限 0 なら actions の µs と同じ"""
    pytest.importorskip('jetformer.policies.controller')
    from jetformer.policies import controller as C
    from jetformer.policies import post
    offsets = np.arange(-575.0, 576.0, 50.0)
    k = np.maximum.accumulate(np.clip(0.004 * offsets, -1.5, 1.5)) + np.arange(len(offsets)) * 1e-6
    pp = {'kind': 'tcp_controller', 'version': 1, 'pwm_center': 1500.0, 'pwm_scale': 500.0,
          'speed_bins': [0.5 * i for i in range(10)], 'wp_dt': 0.1, 'fps': 30,
          'steer_table': {'lag_ms': 40, 'offset_us': offsets.tolist(), 'bands_mps': [2.0, 3.0], 'curv': [k.tolist()] * 3},
          'throttle_ff': {'c0': 1490.7, 'c1': -72.1, 'c2': -13.6, 'tau_s': 0.33, 'fwd_max_us': 361.0},
          'rules': dict(C.RULES), 'pursuit': dict(C.PURSUIT), 'fusion': dict(C.FUSION)}
    cfg = {'postprocess': pp}
    rng = np.random.default_rng(0)
    n = 40
    outputs = {'actions': rng.normal(0, 0.2, (n, 10, 2)).astype(np.float32),
               'waypoints': np.cumsum(np.tile([[0.2, 0.01]], (n, 10, 1)), axis=1).astype(np.float32),
               'speed_logits': rng.normal(0, 1, (n, 10)).astype(np.float32)}
    t_ns = (np.arange(n) * 1e9 / 30).astype(np.int64)
    t_ns[20:] += int(3e9 / 30)                                    # 途中でフレームが抜ける(reset)
    erpm = np.full(n, -4000.0)
    p0 = {'tcp_mode': 0, 'speed_cap': 0.0, 'ensemble_j': 0}
    pred, mp, infos = post.replay(cfg, (1441.0, 1496.0), outputs, erpm, t_ns, p0, scene=np.array(['s'] * n))
    assert np.array_equal(pred, mp) and np.array_equal(mp, outputs['actions'] * 500.0 + 1500.0)
    assert all(i['branch'] == 'control' for i in infos)
    p1 = dict(p0, tcp_mode=1, speed_cap=2.75)
    pred1, _, infos1 = post.replay(cfg, (1441.0, 1496.0), outputs, erpm, t_ns, p1, scene=np.array(['s'] * n))
    assert np.array_equal(pred1[:, :, 0], mp[:, :, 0]) and np.all(np.isfinite(pred1))      # hybrid のステア = 操作の枝
    assert all(i['cap'] == 2.75 for i in infos1)


# ---------------------------------------------------------------- 図
def _fig_res():
    """図の関数に渡す最小の合成の結果(主指標・速さだけ)"""
    names = ['B'] + ['%s_s%d' % (p, s) for p in POLS for s in (0, 1)]
    rng = np.random.default_rng(0)
    L = {ck: {n: 49.0 + float(rng.uniform(0, 2)) for n in names} for ck in ('best', 'last')}
    ref = float(np.mean([L['best']['jetformer_v0_s0'], L['best']['jetformer_v0_s1']]))
    return {'primary': {'L_us': L, 'delta_us': {'best': 10.0, 'last': 10.0},
                        'per_policy': {'jetformer_v0': {'best': {'ref_mean': ref}}}},
            'latency': {'per_run': {n: {'run_numpy_median_ms': 2.4 + 0.1 * i} for i, n in enumerate(names)}}}


def test_zoo_figures_labels_do_not_overlap_title_or_axis(tmp_path, monkeypatch):
    """主指標の図: δ の帯の説明は一番下の行より下(題と重ならない)、値の文字は best・last の印より右。
    速さの図: モデル B の名前は印の横(軸の目盛りの下に出ない)"""
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    import zoo_figures
    figs = []
    orig = Figure.savefig

    def keep(self, *a, **k):
        figs.append(self)
        return orig(self, *a, **k)
    monkeypatch.setattr(Figure, 'savefig', keep)
    res = _fig_res()
    zoo_figures.fig_primary(res, tmp_path)
    zoo_figures.fig_speed(res, tmp_path)
    for f in ('eval_primary.png', 'eval_primary_zoom.png', 'eval_speed_vs_l1.png'):
        assert (tmp_path / f).stat().st_size > 0
    ax = figs[0].axes[0]
    band = [t for t in ax.texts if 'δ' in t.get_text()]
    assert len(band) == 1 and band[0].get_position()[1] < 0 < ax.get_ylim()[1]
    L = res['primary']['L_us']
    rows = ['B'] + POLS
    for t in ax.texts:
        if t.get_text().endswith('µs') and 'δ' not in t.get_text():
            y = round(t.get_position()[1])
            p = rows[len(rows) - 1 - y]
            names = ['B'] if p == 'B' else pz.seed_names(p)
            assert t.get_position()[0] > max(L[ck][n] for ck in ('best', 'last') for n in names)
    ax = figs[2].axes[0]
    tb = [t for t in ax.texts if t.get_text() == 'モデル B'][0]
    assert abs(tb.get_position()[1] - L['best']['B']) < 1e-9 and ax.get_ylim()[0] < L['best']['B']
