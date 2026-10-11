"""オフライン強化学習の方策の学習(tools/train.py・jetformer/train_data.py に足した引数、tools/rl_policies.py)のテスト。
GPU不要(CPU だけ。小さな合成データセットを tmp_path に作る)

    cd ~/jetformer && python -m pytest -q tests/test_rl_policy_training.py
"""

import json
import os
import sys
import types

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import model as jm  # noqa: E402
from jetformer.train_data import GRAVITY, JetFormerDataset, load_frame_table  # noqa: E402
import train  # noqa: E402
import rl_policies as rp  # noqa: E402

MODES = [{'id': i, 'name': n, 'description': ''} for i, n in
         enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]
N = 150                                  # 1シーンのフレーム数(30fps で 5秒)


def _scene_arrays(n=N, mode=5, impact_at=None, seed=0):
    rng = np.random.default_rng(seed)
    t = (np.arange(n) * 1e6 / 30).astype(np.int64)
    imu = np.zeros((n, 10, 6), np.float32)
    imu[:, :, 2] = GRAVITY
    if impact_at is not None:
        imu[impact_at, -1, 1] = 5.0 * GRAVITY                       # 水平 5G の衝突
    fv = np.ones(n, bool)
    fv[-10:] = False
    return {'timestamps_us': t, 'cam_files': np.array(['img.jpg'] * n), 'tof': np.zeros((n, 42, 54), np.uint16),
            'imu': imu, 'vehicle': np.tile(np.array([1500, 1500, 1000, -3000], np.float32), (n, 1)),
            'future_actions': (1500 + rng.normal(0, 50, (n, 10, 2))).astype(np.float32),
            'future_valid': fv, 'vehicle_valid': np.ones(n, bool), 'mode': np.full(n, mode, np.int16)}


@pytest.fixture
def dsroot(tmp_path):
    (tmp_path / 'aligned').mkdir()
    cv2.imwrite(str(tmp_path / 'img.jpg'), np.full((16, 16, 3), 128, np.uint8))
    with open(tmp_path / 'dataset_card.json', 'w', encoding='utf-8') as f:
        json.dump({'modes': MODES}, f, ensure_ascii=False)
    with open(tmp_path / 'splits.json', 'w') as f:
        json.dump({'train': ['s1', 's2'], 'val': ['s3']}, f)
    np.savez(tmp_path / 'aligned' / 's1.npz', **_scene_arrays(mode=5, impact_at=120, seed=1))   # 4秒に衝突
    np.savez(tmp_path / 'aligned' / 's2.npz', **_scene_arrays(mode=4, seed=2))
    np.savez(tmp_path / 'aligned' / 's3.npz', **_scene_arrays(mode=5, seed=3))
    return str(tmp_path)


def _table(tmp_path, name, scenes, key, values):
    """フレームの表の npz を書く。values: {シーン: [N] 値}"""
    scene = np.concatenate([[s] * len(values[s]) for s in scenes]).astype('U')
    frame = np.concatenate([np.arange(len(values[s])) for s in scenes]).astype(np.int32)
    val = np.concatenate([np.asarray(values[s]) for s in scenes])
    path = os.path.join(str(tmp_path), name)
    np.savez(path, scene=scene, frame=frame, **{key: val})
    return path


# ---------------------------------------------------------------- jetformer/train_data.py

def test_default_dataset_unchanged(dsroot):
    ds = JetFormerDataset(dsroot, 'train', stride=3, mask_impacts=True, impact_g=4.0)
    ok = np.ones(N, bool)
    ok[-10:] = False
    ok &= (np.arange(N) % 3) == 0
    t = np.arange(N) / 30
    imp = (t >= 4.0 - 1.0 - 1e-6) & (t <= 4.0 + 2.0)          # 衝突の 1秒前〜2秒後
    expect = [(0, int(i)) for i in np.nonzero(ok & ~imp)[0]] + [(1, int(i)) for i in np.nonzero(ok)[0]]
    assert ds.index == expect
    assert ds.num_modes == 6 and ds.modes == MODES
    assert ds.dropped == 0 and ds.impact_kept == 0
    item = ds[0]
    assert len(item) == 6 and item[4].shape == (6,) and float(item[4][5]) == 1.0
    assert ds.mode_counts() == {5: sum(1 for k, _ in expect if k == 0), 4: sum(1 for k, _ in expect if k == 1)}


def test_drop_mask(dsroot, tmp_path):
    drop = {s: np.zeros(N, bool) for s in ('s1', 's2', 's3')}
    drop['s2'][:30] = True
    path = _table(tmp_path, 'drop.npz', ['s1', 's2', 's3'], 'drop', drop)
    base = JetFormerDataset(dsroot, 'train')
    ds = JetFormerDataset(dsroot, 'train', drop=load_frame_table(path, 'drop'))
    assert ds.index == [(k, i) for k, i in base.index if not (k == 1 and i < 30)]
    assert ds.dropped == 30


def test_frame_table_missing_scene_is_an_error(dsroot, tmp_path):
    path = _table(tmp_path, 'drop.npz', ['s1'], 'drop', {'s1': np.zeros(N, bool)})
    with pytest.raises(KeyError):
        JetFormerDataset(dsroot, 'train', drop=load_frame_table(path, 'drop'))
    bad = _table(tmp_path, 'bad.npz', ['s1', 's2'], 'drop', {'s1': np.zeros(N + 5, bool), 's2': np.zeros(N, bool)})
    with pytest.raises(ValueError):
        JetFormerDataset(dsroot, 'train', drop=load_frame_table(bad, 'drop'))


def test_keep_impact_exempts_frames(dsroot, tmp_path):
    keep = {s: np.zeros(N, bool) for s in ('s1', 's2', 's3')}
    keep['s1'][95:105] = True
    path = _table(tmp_path, 'keep.npz', ['s1', 's2', 's3'], 'keep_impact', keep)
    masked = JetFormerDataset(dsroot, 'train', mask_impacts=True, impact_g=4.0)
    kept = JetFormerDataset(dsroot, 'train', mask_impacts=True, impact_g=4.0, keep_impact=load_frame_table(path, 'keep_impact'))
    back = sorted(set(kept.index) - set(masked.index))
    assert back == [(0, i) for i in range(95, 105)]
    assert kept.impact_kept == 10 and kept.impact_masked == masked.impact_masked - 10


def test_sample_weights(dsroot, tmp_path):
    w = {s: np.arange(N, dtype=np.float32) + 10 * k for k, s in enumerate(('s1', 's2', 's3'))}
    path = _table(tmp_path, 'w.npz', ['s1', 's2', 's3'], 'weight', w)
    ds = JetFormerDataset(dsroot, 'train', weights=load_frame_table(path, 'weight'), return_weight=True)
    item = ds[3]
    k, i = ds.index[3]
    assert len(item) == 7 and float(item[6]) == pytest.approx(i + 10 * k)
    assert np.allclose(ds.sample_weights(), [i + 10 * k for k, i in ds.index])
    w['s1'][5] = np.nan                         # 使うフレームに重みが無い
    path = _table(tmp_path, 'w2.npz', ['s1', 's2', 's3'], 'weight', w)
    with pytest.raises(ValueError):
        JetFormerDataset(dsroot, 'train', weights=load_frame_table(path, 'weight'))


def test_mode_remap_and_dropout(dsroot, tmp_path):
    remap = {s: np.full(N, -1, np.int16) for s in ('s1', 's2', 's3')}
    remap['s1'][:20] = 6
    remap['s2'][:20] = 7
    remap['s3'][:20] = 6
    path = _table(tmp_path, 'm.npz', ['s1', 's2', 's3'], 'mode', remap)
    extra = [{'id': 6, 'name': '低速走行・良'}, {'id': 7, 'name': '高速走行・良'}]
    table = load_frame_table(path, 'mode')
    val = JetFormerDataset(dsroot, 'val', mode_remap=table, extra_modes=extra)
    assert val.num_modes == 8 and [m['id'] for m in val.modes] == list(range(8))
    for n, (k, i) in enumerate(val.index[:25]):
        assert int(val[n][4].argmax()) == (6 if i < 20 else 5)
    tr = JetFormerDataset(dsroot, 'train', mode_remap=table, extra_modes=extra, remap_dropout=1.0)
    assert all(int(tr[n][4].argmax()) in (4, 5) for n in range(len(tr)))   # 必ず元のジャンルに戻る
    counts = tr.mode_counts()                                                # 数えるのは付け替え先
    assert counts[6] == 20 and counts[7] == 20 and counts[5] == N - 30 and counts[4] == N - 30
    np.random.seed(0)
    half = JetFormerDataset(dsroot, 'train', mode_remap=table, extra_modes=extra, remap_dropout=0.3)
    draws = [half.mode_of(0, 0, train_draw=True) for _ in range(2000)]
    assert 0.25 < np.mean(np.array(draws) == 5) < 0.35
    with pytest.raises(ValueError):                                          # 表に無いジャンルへの付け替え
        JetFormerDataset(dsroot, 'val', mode_remap=table)


# ---------------------------------------------------------------- tools/train.py

def test_expand_mode_state_copies_genre_heads():
    torch.manual_seed(0)
    old = jm.JetFormer(num_modes=6).eval()
    new = jm.JetFormer(num_modes=8).eval()
    state = train.expand_mode_state(old.state_dict(), new.state_dict(), train.parse_mode_copy('6:5,7:4'))
    new.load_state_dict(state)
    x = (torch.randn(1, 3, 224, 224), torch.rand(1, 2, 42, 54), torch.zeros(1, 10, 6), torch.zeros(1, 3))
    with torch.no_grad():
        for g_new, g_old in ((4, 4), (5, 5), (6, 5), (7, 4)):
            a = new(*x, jm.mode_onehot([g_new], 8))
            b = old(*x, jm.mode_onehot([g_old], 6))
            assert torch.allclose(a, b, atol=1e-5)
    assert train.expand_mode_state(old.state_dict(), old.state_dict()) is not None
    with pytest.raises(ValueError):
        train.expand_mode_state(old.state_dict(), new.state_dict(), {6: 7})       # 元のジャンルが範囲外
    with pytest.raises(ValueError):
        train.expand_mode_state(new.state_dict(), old.state_dict())               # 減る


class _Fixed(torch.nn.Module):
    """入力の state[:,0] をそのまま全出力にする(誤差を決めて試す)"""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, tof, imu, state, mode):
        return state[:, :1, None].expand(-1, 10, 2) + 0 * self.p


def _batch(errs, weights=None):
    b = len(errs)
    out = [torch.zeros(b, 3, 2, 2), torch.zeros(b, 2, 2, 2), torch.zeros(b, 10, 6),
           torch.tensor([[e, 0, 0] for e in errs], dtype=torch.float32), jm.mode_onehot([5] * b, 6), torch.zeros(b, 10, 2)]
    if weights is not None:
        out.append(torch.tensor(weights, dtype=torch.float32))
    return out


def test_run_epoch_weighted_and_default():
    m = _Fixed()
    errs = [0.1, 0.2, 0.3, 0.4]
    mean, per = train.run_epoch(m, [_batch(errs[:2]), _batch(errs[2:])], torch.device('cpu'), amp=False)
    assert mean == pytest.approx(0.25) and per[5] == pytest.approx(0.25)
    w = [1.0, 3.0, 0.0, 4.0]
    st = {}
    mean, _ = train.run_epoch(m, [_batch(errs[:2], w[:2]), _batch(errs[2:], w[2:])], torch.device('cpu'), amp=False, stats=st)
    assert mean == pytest.approx(sum(e * x for e, x in zip(errs, w)) / sum(w))
    assert st['l1'] == pytest.approx(0.25)
    st = {}
    mean1, _ = train.run_epoch(m, [_batch(errs, [1.0] * 4)], torch.device('cpu'), amp=False, stats=st)
    assert mean1 == pytest.approx(0.25) and st['l1'] == pytest.approx(0.25)


def test_resume_args_and_rl_args_used():
    old = {'data': 'd', 'epochs': 30, 'impact_g': 2.5}            # 前からの last.pth(新しい引数の記録なし)
    assert train.args_diff(old, dict(old)) == {}
    assert train.args_diff(old, dict(old, seed=0)) == {'seed': (None, 0)}
    assert train.args_diff(dict(old, drop_mask='x.npz'), old) == {'drop_mask': ('x.npz', None)}
    args = types.SimpleNamespace(**{k: d for k, d in train.RL_ARGS.items()})
    assert train.rl_args_used(args) == {}
    args.seed = 0                                                  # 種 0 も「使った」
    assert train.rl_args_used(args) == {'seed': 0}
    assert train.parse_mode_copy('') == {} and train.parse_mode_copy('6:5, 7:4') == {6: 5, 7: 4}


# ---------------------------------------------------------------- tools/rl_policies.py

def test_failure_window_is_ai_only_same_scene():
    scene = np.array(['a'] * 6 + ['b'] * 3)
    t = np.array([0, 1, 2, 3, 4, 5, 3.5, 4.0, 4.5])
    src = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1])
    flag = np.zeros(9, bool)
    flag[4] = True                                                # シーン a の 4秒
    w = rp.failure_window(scene, t, src, flag, window_s=2.0)
    assert w.tolist() == [False, False, False, True, True, False, False, False, False]


def test_frame_sets_rules():
    eligible = np.array([1, 1, 1, 1, 1, 0, 1], bool)
    A = np.array([0.1, -1, 2, 0.5, np.nan, 0, 1])
    I = np.array([0, 0, 1, 1, 0, -1, -1])
    src = np.array([0, 0, 1, 1, 1, 1, 1])
    nc = np.array([0, 1, 1, 0, 0, 0, 0], bool)
    fw = np.array([0, 0, 0, 1, 0, 0, 0], bool)
    s = rp.frame_sets(eligible, A, I, src, nc, fw)
    assert s['valid'].tolist() == [True, True, True, True, False, False, False]
    assert (~s['drop_clean']).tolist() == [True, False, False, False, False, False, False]
    assert (~s['drop_all']).tolist() == [True, False, True, True, False, False, False]   # 人の衝突の前後は外したまま
    assert s['keep_impact'].tolist() == [False, False, True, True, False, False, False]
    assert s['fail_adj_ai'].tolist() == [False, False, True, True, False, False, False]


def test_awr_weights_and_remap():
    A = np.array([-3.0, 0.0, 1.0, 10.0, np.nan])
    valid = np.array([1, 1, 1, 1, 0], bool)
    w, beta = rp.awr_weights(A, valid, beta=1.0, clip=20.0)
    assert np.allclose(w[:4], [np.exp(-3), 1.0, np.e, 20.0]) and np.isnan(w[4]) and beta == 1.0
    _, beta2 = rp.awr_weights(A, valid)
    assert beta2 == pytest.approx(np.std(A[:4]))
    I = np.array([1, 1, 0, 1, -1])
    genre = np.array([4, 5, 5, 5, 4])
    m = rp.remap_modes(I, genre, np.array([1, 1, 1, 0, 1], bool), {5: 6, 4: 7})
    assert m.tolist() == [7, 6, -1, -1, -1]


def test_register_good_modes_is_idempotent():
    table = [dict(m) for m in MODES]

    def add(name, desc=''):
        m = {'id': max(x['id'] for x in table) + 1, 'name': name, 'description': desc}
        table.append(m)
        return m
    fake = types.SimpleNamespace(load=lambda: table, add=add)
    assert rp.register_good_modes(fake) == {5: 6, 4: 7}
    assert rp.register_good_modes(fake) == {5: 6, 4: 7}
    assert [m['name'] for m in table[6:]] == ['低速走行・良', '高速走行・良']


def test_train_commands_differ_only_in_data_options():
    cmds = {v: rp.train_command(v, '/tmp/out_' + v, 12, good_of={5: 6, 4: 7}) for v in rp.ORDER}
    common = set(cmds['c']) - {'/tmp/out_c', str(rp.policy_dir() / 'drop_clean.npz')}
    for v, c in cmds.items():
        assert common <= set(c), v
        assert c[c.index('--init') + 1].endswith(rp.BASE_RUN + '/best.pth')
        assert '--no-command-input' in c and c[c.index('--impact-g') + 1] == '2.5' and c[c.index('--seed') + 1] == '0'
    assert '--sample-weights-npz' in cmds['b'] and '--sample-weights-npz' not in cmds['c']
    for v in ('a', 'n'):
        c = cmds[v]
        assert c[c.index('--init-mode-copy') + 1] == '6:5,7:4'
        assert c[c.index('--mode-remap-npz') + 1].endswith('modes_%s.npz' % v)
        assert c[c.index('--mode-remap-dropout') + 1] == '0.3'
    assert rp.train_command('c', '/tmp/o', 12, good_of={})[-1] != '--resume'
    assert rp.train_command('c', '/tmp/o', 12, resume=True)[-1] == '--resume'


def test_is_gpu_cmd_ignores_shells():
    assert rp.is_gpu_cmd(['/home/x/envs/jetracer/bin/python', '-u', '/home/x/jetformer/tools/train.py', '--data', 'd'])
    assert rp.is_gpu_cmd(['python3.12', 'tools/build_engine.py', 'a.onnx', 'b.engine'])
    assert rp.is_gpu_cmd(['/usr/src/tensorrt/bin/trtexec', '--onnx=a.onnx'])
    assert not rp.is_gpu_cmd(['/bin/bash', '-c', 'until grep -q x log; do sleep 1; done; ps aux | grep tools/train.py'])
    assert not rp.is_gpu_cmd(['python', '-m', 'pytest', 'tests/test_train.py'])
    assert not rp.is_gpu_cmd(['python', '-m', 'jetformer.web.server'])
    assert not rp.is_gpu_cmd([])


def test_cluster_bootstrap_mean():
    d = np.r_[np.full(50, 1.0), np.full(50, 3.0)]
    cl = np.r_[np.zeros(50, int), np.ones(50, int)]
    m, lo, hi = rp.cluster_bootstrap_mean(d, cl, n_boot=500)
    assert m == pytest.approx(2.0) and lo >= 1.0 - 1e-9 and hi <= 3.0 + 1e-9 and lo < 2.0 < hi
    m, lo, hi = rp.cluster_bootstrap_mean(np.full(30, 0.5), np.arange(30) % 3)
    assert m == lo == hi == pytest.approx(0.5)
