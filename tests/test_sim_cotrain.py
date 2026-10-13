"""シミュレーターと実機を混ぜた学習の準備(tools/sim_cotrain.py)のテスト(numpy だけ・CPU・実データなし)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_sim_cotrain.py
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import sim_cotrain as sc  # noqa: E402
from jetformer.train_data import JetFormerDataset, load_frame_table  # noqa: E402


def _aligned(n=30, t0=1_000_000, step=None, seed=0):
    rng = np.random.default_rng(seed)
    t = t0 + (np.round(np.arange(n) * 33333.333).astype(np.int64) if step is None else np.arange(n) * step)
    veh = np.stack([1442 + rng.normal(0, 50, n), np.full(n, 1300.0), np.full(n, 1500.0), np.full(n, -4000.0)], 1)
    fut = np.stack([veh[np.clip(np.arange(n) + k + 1, 0, n - 1), :2] for k in range(10)], axis=1)
    return {'timestamps_us': t, 'cam_files': np.array(['samples/CAM_FRONT/s/%d.jpg' % x for x in t]),
            'tof': np.full((n, 42, 54), (4 << 13) | 1200, np.uint16), 'tof_age_ms': np.zeros(n, np.float32),
            'imu': rng.normal(0, 1, (n, 10, 6)).astype(np.float32), 'vehicle': veh.astype(np.float32),
            'vehicle_valid': np.ones(n, bool), 'vehicle_age_ms': np.zeros(n, np.float32),
            'future_actions': fut.astype(np.float32), 'future_valid': np.r_[np.ones(n - 10, bool), np.zeros(10, bool)],
            'erpm_valid': np.ones(n, bool), 'mode': np.full(n, 8, np.int16)}


# ---------------------------------------------------------------- 比べる・戻す
def test_compare_arrays_equal_and_different():
    a, b = _aligned(), _aligned()
    cmp = sc.compare_arrays(a, b, sc.COMPARE_KEYS)
    assert sc.all_equal(cmp) and cmp['future_actions']['max_abs'] == 0.0
    b['future_actions'] = b['future_actions'].copy()
    b['future_actions'][3, 2, 1] += 7.0
    b['future_valid'] = b['future_valid'].copy()
    b['future_valid'][0] = False
    b['cam_files'] = np.array([str(x).replace('/s/', '/t/') for x in b['cam_files']])
    cmp = sc.compare_arrays(a, b, sc.COMPARE_KEYS)
    assert not sc.all_equal(cmp)
    assert cmp['future_actions']['mismatch'] == 1 and abs(cmp['future_actions']['max_abs'] - 7.0) < 1e-4
    assert cmp['future_valid']['mismatch'] == 1 and cmp['future_valid']['max_abs'] is None
    assert cmp['cam_files']['mismatch'] == 30
    b['imu'] = b['imu'][:5]
    del b['mode']
    cmp = sc.compare_arrays(a, b, ('imu', 'mode'))
    assert cmp['imu']['shape_equal'] is False and cmp['mode']['missing'] == ['b']


def test_restore_from_original_keeps_paths_and_checks_time():
    orig = _aligned(seed=1)
    rebuilt = _aligned(seed=2)                                       # 同じ時刻・違う中身
    rebuilt['cam_files'] = np.array([str(x).replace('/s/', '/ds-x-0001/') for x in rebuilt['cam_files']])
    out = sc.restore_from_original(rebuilt, orig)
    assert sc.all_equal(sc.compare_arrays(out, orig, sc.RESTORE_KEYS))
    assert (out['cam_files'] == rebuilt['cam_files']).all() and (out['timestamps_us'] == rebuilt['timestamps_us']).all()
    assert not sc.all_equal(sc.compare_arrays(rebuilt, orig, ('imu',)))   # 元の dict は変えない
    shifted = dict(rebuilt, timestamps_us=rebuilt['timestamps_us'] + 1)
    with pytest.raises(ValueError):
        sc.restore_from_original(shifted, orig)
    with pytest.raises(ValueError):
        sc.restore_from_original(rebuilt, {k: v for k, v in orig.items() if k != 'imu'})


def test_restore_log_for_resets_when_dataset_rebuilt():
    log = {'card_created_at': '2026-10-13T07:53:10', 'scenes': {'ds-x-0001': {'restored': True}}}
    assert sc.restore_log_for(log, '2026-10-13T07:53:10') is log                 # 同じデータセット = 続きから
    new = sc.restore_log_for(log, '2026-10-13T11:50:00')                         # 作り直した = 最初から
    assert new == {'card_created_at': '2026-10-13T11:50:00', 'scenes': {}}
    assert sc.restore_log_for({'scenes': {'a': {}}}, 'x')['scenes'] == {}         # 印の無い前の記録も使わない
    assert sc.restore_log_for({}, None)['scenes'] == {}


def test_future_shift_stats_counts_frames_and_steps():
    a = _aligned(n=20)
    b = {k: v.copy() for k, v in a.items()}
    b['future_actions'][[2, 5], 1, 0] += 50.0
    b['future_actions'][5, 4, 1] += 0.5                              # 1 µs 以下は数えない
    st = sc.future_shift_stats(b, a)
    assert st['frames'] == 20 and st['frames_gt_tol'] == 2 and abs(st['max_abs_us'] - 50.0) < 1e-3
    assert st['per_step_frames_gt_tol'][1] == 2 and sum(st['per_step_frames_gt_tol']) == 2


# ---------------------------------------------------------------- ドロップの表
def test_drop_frame_table_rows_and_unknown():
    t = sc.drop_frame_table({'real-1': 3, 'ds-sim-0001': 2, 'real-2': 1}, ['ds-sim-0001'])
    assert list(t['scene']) == ['ds-sim-0001'] * 2 + ['real-1'] * 3 + ['real-2']
    assert list(t['frame']) == [0, 1, 0, 1, 2, 0] and list(t['drop']) == [True, True, False, False, False, False]
    with pytest.raises(ValueError):
        sc.drop_frame_table({'a': 1}, ['b'])


def test_drop_mask_with_jetformer_dataset_keeps_only_real(tmp_path):
    """混ぜたデータセット + ドロップの表 → 学習のフレームは実機のシーンだけ(train.py --drop-mask と同じ読み方)"""
    root = tmp_path / 'mix'
    (root / 'aligned').mkdir(parents=True)
    modes = [{'id': i, 'name': str(i)} for i in range(9)]
    (root / 'dataset_card.json').write_text(json.dumps({'modes': modes}), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': ['r1', 'ds-sim-0001'], 'val': ['r2']}), encoding='utf-8')
    frames = {}
    for name, mode, seed in (('r1', 5, 0), ('r2', 4, 1), ('ds-sim-0001', 8, 2)):
        a = _aligned(n=40, seed=seed)
        a['mode'] = np.full(40, mode, np.int16)
        np.savez(root / 'aligned' / ('%s.npz' % name), **a)
        frames[name] = 40
    np.savez(tmp_path / 'drop.npz', **sc.drop_frame_table(frames, ['ds-sim-0001']))
    drop = load_frame_table(str(tmp_path / 'drop.npz'), 'drop')
    kw = dict(sc.TRAIN_KW)
    full = JetFormerDataset(str(root), 'train', **kw)
    ft = JetFormerDataset(str(root), 'train', drop=drop, **kw)
    val = JetFormerDataset(str(root), 'val', drop=drop, **kw)
    assert full.num_modes == 9 and set(full.mode_counts()) == {5, 8}
    assert set(ft.mode_counts()) == {5} and ft.mode_counts()[5] == full.mode_counts()[5]
    assert ft.dropped == 30 and val.dropped == 0 and len(val) == 10      # 30 = 未来が有効な 30 フレーム(間引きの前)


# ---------------------------------------------------------------- 判定の規則
A_BEST = (49.146, 49.457)                       # 10/12 の jetformer_v0 種 0・1(best.pth)


def test_judge_primary_better_needs_both_seeds_and_margin():
    j = sc.judge_primary((48.0, 48.3), A_BEST)
    assert j['class'] == 'better' and abs(j['threshold_us'] - 2 * 0.311) < 1e-9 and not j['practical']
    assert sc.judge_primary((48.0, 49.2), A_BEST)['class'] == 'equivalent'      # 片方の種が A の種 0 より悪い
    assert sc.judge_primary((49.0, 49.1), A_BEST)['class'] == 'equivalent'      # 両方良いが差 0.25 < 0.622
    assert sc.judge_primary((38.0, 39.0), A_BEST)['practical']                   # 10 µs 以上
    assert sc.judge_primary((50.5, 50.2), A_BEST)['class'] == 'worse'
    assert sc.judge_primary((49.5, 49.6), A_BEST)['class'] == 'equivalent'      # 悪いが差 < 閾値
    j = sc.judge_primary((48.0, 48.3), A_BEST, spread_floor=1.0)                # 閾値 2.0 > 改善 1.15
    assert j['ref_spread_us'] == 1.0 and j['class'] == 'equivalent' and abs(j['threshold_us'] - 2.0) < 1e-9
    assert sc.judge_primary((47.0, 47.2), A_BEST, spread_floor=1.0)['class'] == 'better'


def test_final_verdict_downgrades_best_only():
    b = {'class': 'better'}
    assert sc.final_verdict(b, {'class': 'better'}) == 'better'
    assert sc.final_verdict(b, {'class': 'equivalent'}) == 'equivalent'
    assert sc.final_verdict({'class': 'worse'}, {'class': 'equivalent'}) == 'worse'
    assert sc.final_verdict({'class': 'equivalent'}, {'class': 'better'}) == 'equivalent'


# ---------------------------------------------------------------- 腕の学習の引数・手順の固定
def test_train_args_recipes():
    c = sc.train_args('C', 0)
    assert c[0] == 'tools/train.py' and '--init' not in c and '--drop-mask' not in c
    assert c[c.index('--data') + 1] == 'data/datasets/' + sc.MIX
    assert c[c.index('--epochs') + 1] == '30' and c[c.index('--lr') + 1] == '3e-4'
    for k, v in (('--stride', '3'), ('--batch', '64'), ('--workers', '4'), ('--impact-g', '2.5'), ('--seed', '0')):
        assert c[c.index(k) + 1] == v
    assert '--no-command-input' in c and c[c.index('--policy') + 1] == 'jetformer_v0'
    f = sc.train_args('C+FT', 1)
    assert f[f.index('--init') + 1] == 'data/runs/%s/best.pth' % (sc.RUN_C % 1)          # 同じ種の C から
    assert f[f.index('--drop-mask') + 1] == 'data/derived/%s/drop_sim_frames.npz' % sc.MIX
    assert f[f.index('--epochs') + 1] == '12' and f[f.index('--lr') + 1] == '1e-4' and f[f.index('--seed') + 1] == '1'
    assert f[f.index('--out') + 1] == 'data/runs/' + sc.RUN_CFT % 1
    a = sc.train_args('A+FT', 0, resume=True)
    assert a[a.index('--init') + 1] == 'data/runs/%s/best.pth' % (sc.RUN_A % 0) and a[-1] == '--resume'
    assert a[a.index('--data') + 1] == 'data/datasets/' + sc.REAL_DATASET and '--drop-mask' not in a
    assert sc.run_name('B', 0) == sc.RUN_B
    with pytest.raises(ValueError):
        sc.train_args('A', 0)                                   # A は新しく学習しない
    with pytest.raises(ValueError):
        sc.run_name('C', 2)


def test_run_names_follow_task():
    """C・C+FT の学習結果の名前は 2026-10-13 の指示の形(simreal_mix_20261013-cotrain-s<種>・-cotrain-ft-s<種>)。
    A・A+FT・モデル B の名前は変えていない"""
    assert [sc.run_name('C', s) for s in sc.SEEDS] == ['simreal_mix_20261013-cotrain-s0', 'simreal_mix_20261013-cotrain-s1']
    assert [sc.run_name('C+FT', s) for s in sc.SEEDS] == ['simreal_mix_20261013-cotrain-ft-s0',
                                                          'simreal_mix_20261013-cotrain-ft-s1']
    assert sc.run_name('A', 1) == 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30'
    assert sc.run_name('A+FT', 0) == 'clean_driving_20261011-jetformer_v0-s0-ft-real-lr1e-4-e12'
    f = sc.train_args('C+FT', 0)
    assert f[f.index('--init') + 1] == 'data/runs/simreal_mix_20261013-cotrain-s0/best.pth'
    assert f[f.index('--out') + 1] == 'data/runs/simreal_mix_20261013-cotrain-ft-s0'


def _fake_zoo():
    names = ['jetformer_v0_s0', 'jetformer_v0_s1', 'B']
    acc = {k: 1.0 for k in ('n', 'l1_us', 'fwd_bias_us', 'steer_sign_acc', 'steer_mae_k3', 'throttle_mae_k3')}
    return {
        'primary': {'L_us': {'best': {'jetformer_v0_s0': 49.146, 'jetformer_v0_s1': 49.457, 'B': 49.144, 'act_s0': 50.0},
                             'last': {'jetformer_v0_s0': 49.235, 'jetformer_v0_s1': 49.752, 'B': 49.091}}},
        'accuracy': {'metrics': {n: {s: dict(acc) for s in ('human_clean', 'human_clean_genre4', 'human_clean_genre5')}
                                 for n in names}},
        'r1': {'window_mean_us': {'best': {n: 1.0 for n in names}, 'last': {n: 1.0 for n in names}},
               'nc': {'NC-A': {'d_mean': 7.69, 'bad': False}, 'NC-B': {'d_mean': -7.47, 'bad': True}}},
        'de': {'window_mean_us': {'best': {n: 1.0 for n in names}, 'last': {n: 1.0 for n in names}},
               'nc': {'NC-A': {'d_mean': -2.78, 'bad': False}, 'NC-B': {'d_mean': 2.32, 'bad': False}}},
        'robustness': {'per_run': {n: {'dL1_us': d, 'per_pert_dL1_us': {}, 'frames': 1559}
                                   for n, d in zip(names, (0.699, 0.887, 0.791))}},
        'collapse': {'per_run': {n: {} for n in names}},
        'latency': {'per_run': {n: {'run_numpy_median_ms': 2.37} for n in names}},
        'frames': {'pred_table': 10985, 'human_clean': 4670, 'pred_table_key': 'k'},
    }


def test_baseline_and_protocol_thresholds():
    base = sc.baseline_from_zoo(_fake_zoo())
    assert set(base['L_us']['best']) == {'jetformer_v0_s0', 'jetformer_v0_s1', 'B'}
    assert abs(base['spread_us']['best'] - 0.313) < 1e-9 and abs(base['spread_us']['best_two_seeds'] - 0.311) < 1e-9
    assert abs(base['spread_us']['last'] - 0.661) < 1e-9
    assert abs(base['ref_mean_us']['best'] - 49.3015) < 1e-9 and abs(base['robust_ref_dL1_us'] - 0.793) < 1e-9
    assert base['nc_abs_max_us'] == {'r1': 7.69, 'de': 2.78}
    tf = {k: {'train': n, 'val': 1559, 'num_modes': m, 'mode_counts': {}, 'action_reference': [1441.0, 1496.0]}
          for k, n, m in (('A', 5912, 6), ('C', 17732, 9), ('C+FT', 5912, 9))}
    check = {'card': {}, 'train_frames': tf, 'ok': {'x': True}, 'sim_ref_scenes': []}
    proto = sc.build_protocol(check, base, {})
    thr = proto['primary']['threshold_us']
    assert abs(thr['best'] - 0.626) < 1e-9 and abs(thr['last'] - 1.322) < 1e-9
    assert set(proto['arms']) == {'A', 'C', 'C+FT', 'A+FT'}
    assert proto['arms']['C+FT']['commands']['0']['train'].endswith(' '.join(sc.train_args('C+FT', 0)))
    assert proto['arms']['C']['commands']['1']['resume'].endswith('--resume')
    json.dumps(proto, ensure_ascii=False)                                      # JSON に書ける
    # 判定は固定した閾値(2 × s_A)で: 両方の種が A の両方より良く、平均の改善 > 0.626 µs
    j = sc.judge_primary((48.5, 48.6), (49.146, 49.457), spread_mult=2.0, spread_floor=base['spread_us']['best'])
    assert j['class'] == 'better' and abs(j['threshold_us'] - 0.626) < 1e-9
    assert sc.judge_primary((48.8, 48.9), (49.146, 49.457), spread_floor=base['spread_us']['best'])['class'] == 'equivalent'
