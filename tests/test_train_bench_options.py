"""ベンチマークの学習(2026-10-21)に足した引数のテスト: tools/train.py の --label-override-npz・--aug-*、
jetformer/train_data.py の label_override・aug。既定(使わない)なら今までと同じサンプル・同じ乱数の使い方であることも確かめる。
GPU 不要(CPU だけ。小さな合成データセットを tmp_path に作る。torch を読むので jf-heavy --lock big で回す)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big -- python -m pytest -q tests/test_train_bench_options.py
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
from jetformer.train_data import GRAVITY, JetFormerDataset, augment_image, augment_tof, load_frame_table  # noqa: E402
import train  # noqa: E402

MODES = [{'id': i, 'name': n, 'description': ''} for i, n in
         enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]
N = 60
SMALL_V0 = {'dim': 32, 'layers': 1, 'heads': 4}


@pytest.fixture
def dsroot(tmp_path):
    root = tmp_path / 'ds'
    (root / 'aligned').mkdir(parents=True)
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :, 0] = np.arange(224, dtype=np.uint8)[None, :]              # 横の勾配(ずれが分かるように)
    img[:, :, 1] = np.arange(224, dtype=np.uint8)[:, None]
    img[:, :, 2] = 100
    cv2.imwrite(str(root / 'img.png'), img)
    (root / 'dataset_card.json').write_text(json.dumps({'modes': MODES}, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': ['s1', 's2'], 'val': ['s3']}))
    rng = np.random.default_rng(0)
    for k, s in enumerate(['s1', 's2', 's3']):
        imu = np.zeros((N, 10, 6), np.float32)
        imu[:, :, 2] = GRAVITY
        fv = np.ones(N, bool)
        fv[-10:] = False
        tof = rng.integers(0, 4000, (N, 42, 54)).astype(np.uint16)
        np.savez(root / 'aligned' / (s + '.npz'), timestamps_us=(np.arange(N) * 1e6 / 30).astype(np.int64),
                 cam_files=np.array(['img.png'] * N), tof=tof, imu=imu,
                 vehicle=np.tile(np.array([1500, 1400, 1000, -3000], np.float32), (N, 1)),
                 future_actions=(1450 + rng.normal(0, 50, (N, 10, 2))).astype(np.float32), future_valid=fv,
                 vehicle_valid=np.ones(N, bool), mode=np.full(N, 4 + k % 2, np.int16))
    return str(root)


def _override(tmp_path, scenes, frames_rows, value=1498.0):
    """future_throttle_us の表。frames_rows: {シーン: [(フレーム, 行), ...]} を value に、ほかは NaN"""
    sc, fr, val = [], [], []
    for s in scenes:
        v = np.full((N, 10), np.nan, np.float32)
        for f, k in frames_rows.get(s, []):
            v[f, k] = value
        sc += [s] * N
        fr.append(np.arange(N))
        val.append(v)
    path = os.path.join(str(tmp_path), 'ov.npz')
    np.savez(path, scene=np.array(sc), frame=np.concatenate(fr).astype(np.int32), future_throttle_us=np.concatenate(val))
    return path


def _items(ds, idx, seed):
    np.random.seed(seed)
    out = [ds[i] for i in idx]
    return out, np.random.get_state()[1].copy()


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


# ---------------------------------------------------------------- 既定は今までと同じ
def test_default_items_and_rng_unchanged(dsroot):
    base = JetFormerDataset(dsroot, 'train', color_jitter=True, stride=3)
    for kw in ({'label_override': None, 'aug': None},
               {'aug': {'brightness': 0, 'contrast': 0.0, 'shift_px': 0, 'tof_noise': 0}}):
        ds = JetFormerDataset(dsroot, 'train', color_jitter=True, stride=3, **kw)
        assert ds.aug is None and ds.index == base.index
        a, ra = _items(base, range(len(base)), 3)
        b, rb = _items(ds, range(len(ds)), 3)
        assert all(_same(x, y) for x, y in zip(a, b))
        assert np.array_equal(ra, rb)                       # 乱数を余分に引かない(同じ種で同じバッチ)


def test_train_args_defaults_are_unused():
    args = types.SimpleNamespace(**dict(train.BENCH_ARGS))
    assert train.bench_args_used(args) == {}
    assert train.aug_of(args) is None
    args.aug_shift_px = 3
    assert train.bench_args_used(args) == {'aug_shift_px': 3}
    assert train.aug_of(args) == {'brightness': 0.0, 'contrast': 0.0, 'shift_px': 3, 'tof_noise': 0.0}
    args.aug_tof_noise = -0.1
    with pytest.raises(SystemExit):
        train.aug_of(args)
    old = {'data': 'd', 'epochs': 30}                     # 前からの last.pth(新しい引数の記録なし)で再開できる
    assert train.args_diff(old, dict(old, **train.bench_args_used(types.SimpleNamespace(**train.BENCH_ARGS)))) == {}


# ---------------------------------------------------------------- 正解の差し替え
def test_label_override_throttle_only(dsroot, tmp_path):
    path = _override(tmp_path, ['s1'], {'s1': [(0, 0), (0, 9), (3, 4)]}, value=1600.0)
    table = load_frame_table(path, 'future_throttle_us')
    base = JetFormerDataset(dsroot, 'train')
    ds = JetFormerDataset(dsroot, 'train', label_override=table)
    assert ds.label_overridden == 3 and ds.label_overridden_used == 3
    fa0 = base.scenes[0]['future_actions']
    fa1 = ds.scenes[0]['future_actions']
    assert fa1[0, 0, 1] == 1600.0 and fa1[0, 9, 1] == 1600.0 and fa1[3, 4, 1] == 1600.0
    changed = np.zeros_like(fa0, bool)
    changed[0, 0, 1] = changed[0, 9, 1] = changed[3, 4, 1] = True
    assert np.array_equal(fa1[~changed], fa0[~changed])     # ステア・ほかの行・ほかのフレームはそのまま
    assert np.array_equal(ds.scenes[1]['future_actions'], base.scenes[1]['future_actions'])   # 表に無いシーンもそのまま
    # 元のデータセットのファイルは変えない(読み直しても同じ)
    again = JetFormerDataset(dsroot, 'train')
    assert np.array_equal(again.scenes[0]['future_actions'], fa0)
    # 学習に使わないフレーム(future_valid=False)の行は used に数えない
    path = _override(tmp_path, ['s1'], {'s1': [(N - 1, 0)]})
    ds = JetFormerDataset(dsroot, 'train', label_override=load_frame_table(path, 'future_throttle_us'))
    assert ds.label_overridden == 1 and ds.label_overridden_used == 0


def test_label_override_rejects_val_and_bad_shape(dsroot, tmp_path):
    path = _override(tmp_path, ['s1', 's3'], {})
    with pytest.raises(KeyError):                            # s3 は val の分け方 → 学習に渡すとエラー
        JetFormerDataset(dsroot, 'train', label_override=load_frame_table(path, 'future_throttle_us'))
    path = _override(tmp_path, ['s1'], {})
    with pytest.raises(KeyError):                            # 学習の表を val に渡してもエラー
        JetFormerDataset(dsroot, 'val', label_override=load_frame_table(path, 'future_throttle_us'))
    bad = {'s1': (np.arange(N), np.zeros((N, 9), np.float32))}
    with pytest.raises(ValueError):
        JetFormerDataset(dsroot, 'train', label_override=bad)
    bad = {'s1': (np.array([N + 5]), np.zeros((1, 10), np.float32))}
    with pytest.raises(ValueError):
        JetFormerDataset(dsroot, 'train', label_override=bad)


# ---------------------------------------------------------------- 入力の拡張
def test_aug_changes_inputs_not_targets(dsroot):
    base = JetFormerDataset(dsroot, 'train')
    ds = JetFormerDataset(dsroot, 'train', aug={'brightness': 0.2, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02})
    np.random.seed(0)
    a = base[0]
    b = ds[0]
    assert not torch.equal(a[0], b[0]) and not torch.equal(a[1][0], b[1][0])
    assert torch.equal(a[1][1], b[1][1])                     # ToF の信頼度は変えない
    assert torch.equal(a[5], b[5]) and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])
    assert float(b[1][0].min()) >= 0.0 and float(b[1][0].max()) <= 1.0


def test_augment_image_and_tof_units():
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :, 0] = np.arange(224, dtype=np.uint8)[None, :]
    np.random.seed(1)
    out = augment_image(img, shift_px=3)
    assert out.shape == img.shape and out.dtype == np.uint8
    dx = int(out[100, 100, 0]) - int(img[100, 100, 0])       # 横の勾配なので、ずれた画素数が値の差に出る
    assert -3 <= dx <= 3
    np.random.seed(2)
    out = augment_image(np.full((224, 224, 3), 100, np.uint8), brightness=0.1)
    assert abs(int(out[0, 0, 0]) - 100) <= 26
    t = np.full((2, 42, 54), 0.5, np.float32)
    np.random.seed(3)
    o = augment_tof(t, 0.02)
    assert np.array_equal(o[1], t[1]) and 0.005 < float(np.std(o[0])) < 0.04
    assert augment_tof(t, 0.0) is t


# ---------------------------------------------------------------- train.py を通す(CPU・1 ステップ)
def test_train_main_with_bench_options(dsroot, tmp_path, monkeypatch):
    ov = _override(tmp_path, ['s1', 's2'], {'s1': [(0, 0), (3, 1)], 's2': [(6, 2)]}, value=1550.0)
    w = os.path.join(str(tmp_path), 'w.npz')
    sc = np.array(['s1'] * N + ['s2'] * N + ['s3'] * N)
    np.savez(w, scene=sc, frame=np.tile(np.arange(N), 3).astype(np.int32), weight=np.ones(3 * N, np.float32) * 2)
    out = tmp_path / 'run'
    argv = ['train.py', '--data', dsroot, '--out', str(out), '--epochs', '1', '--batch', '4', '--workers', '0',
            '--max-steps', '1', '--stride', '3', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy-cfg', json.dumps(SMALL_V0), '--seed', '0', '--label-override-npz', ov, '--sample-weights-npz', w,
            '--aug-brightness', '0.2', '--aug-contrast', '0.2', '--aug-shift-px', '4', '--aug-tof-noise', '0.02']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    train.main()
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    bo = res['bench_options']
    assert bo['label_override_npz'] == ov and bo['label_overridden_rows']['all'] == 3
    assert bo['aug'] == {'brightness': 0.2, 'contrast': 0.2, 'shift_px': 4.0, 'tof_noise': 0.02}
    ck = torch.load(out / 'last.pth', weights_only=False)
    assert ck['args']['label_override_npz'] == ov and ck['args']['aug_shift_px'] == 4
    # 既定のまま: result.json に bench_options が無く、last.pth の引数にも新しい名前が無い(前からの再開の規則と同じ)
    out2 = tmp_path / 'run2'
    argv = ['train.py', '--data', dsroot, '--out', str(out2), '--epochs', '1', '--batch', '4', '--workers', '0',
            '--max-steps', '1', '--stride', '3', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy-cfg', json.dumps(SMALL_V0), '--seed', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    train.main()
    res2 = json.loads((out2 / 'result.json').read_text(encoding='utf-8'))
    assert 'bench_options' not in res2
    ck2 = torch.load(out2 / 'last.pth', weights_only=False)
    assert not set(train.BENCH_ARGS) & set(ck2['args'])
