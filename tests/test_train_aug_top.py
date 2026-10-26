"""C1(2026-10-25、データセット v2 の段 3 の AUG2): tools/train.py の --aug-top-brightness と
jetformer/train_data.py の top_brightness(画像の上 1/3 の明るさ ±F。一番上の行で全部、上 1/3 の境で 0)のテスト。

既定(0 = 使わない)なら、C1 の前(commit c247daa。protocol_v2.json の defaults_untouched.sha256_now と同じ中身)の
train_data.py・train.py と同じサンプル・同じ乱数・同じ損失・同じ重みになることを確かめる。10/21 の拡張(上 1/3 なし)も同じ。
前からの last.pth で --resume できること、AUG2 の last.pth で再開できること、値が違えば再開を断ることも確かめる。
GPU 不要(CPU だけ。小さな合成データセットを tmp_path に作る。torch を読むので jf-heavy --lock big で回す)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock big -- python -m pytest -q tests/test_train_aug_top.py
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import train_data as new_td  # noqa: E402
from jetformer.train_data import GRAVITY, JetFormerDataset, augment_image, top_ramp  # noqa: E402
import train  # noqa: E402

BASE_REV = 'c247daa'                     # C1 の前の commit
BASE_SHA256 = {'tools/train.py': '6547618cd5a7f53079f1e02ffe1c3e60e3f794f7e8a3afcdde42ddf798d54d0f',
               'jetformer/train_data.py': '8762dda4bc8c7f347bc8f706743a35f4f93e5b15a8d3e487ea39d59509d717aa'}
MODES = [{'id': i, 'name': n, 'description': ''} for i, n in
         enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]
N = 60
SMALL_V0 = {'dim': 32, 'layers': 1, 'heads': 4}
AUG_1021 = {'brightness': 0.2, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02}      # 10/21 の AUG(上 1/3 なし)
AUG2_ARGS = ['--aug-brightness', '0.15', '--aug-contrast', '0.2', '--aug-shift-px', '4', '--aug-tof-noise', '0.02',
             '--aug-top-brightness', '0.15']                                          # protocol_v2 の AUG2
AUG_1021_ARGS = ['--aug-brightness', '0.2', '--aug-contrast', '0.2', '--aug-shift-px', '4', '--aug-tof-noise', '0.02']


def _base_source(path):
    try:
        src = subprocess.run(['git', '-C', ROOT, 'show', '%s:%s' % (BASE_REV, path)], check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git で %s:%s を読めない' % (BASE_REV, path))
    assert hashlib.sha256(src).hexdigest() == BASE_SHA256[path]       # protocol の「変える前」と同じ中身
    return src


def _load(name, path, src, package=None):
    d = os.path.join(os.path.dirname(path), '')
    os.makedirs(d, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if package:
        mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    """C1 の前の train_data(jetformer パッケージの中のモジュールとして)と train(その train_data を使う)"""
    d = tmp_path_factory.mktemp('base')
    td = _load('jetformer.train_data_c1base', str(d / 'train_data_c1base.py'), _base_source('jetformer/train_data.py'),
               package='jetformer')
    tr = _load('train_c1base', str(d / 'train_c1base.py'), _base_source('tools/train.py'))
    tr.JetFormerDataset, tr.load_frame_table = td.JetFormerDataset, td.load_frame_table
    return td, tr


@pytest.fixture
def dsroot(tmp_path):
    root = tmp_path / 'ds'
    (root / 'aligned').mkdir(parents=True)
    img = np.zeros((224, 224, 3), np.uint8)
    img[:, :, 0] = np.arange(224, dtype=np.uint8)[None, :]
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


def _items(ds, idx, seed):
    np.random.seed(seed)
    out = [ds[i] for i in idx]
    return out, np.random.get_state()[1].copy()


def _same_items(a, b):
    return len(a) == len(b) and all(len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)) for x, y in zip(a, b))


# ---------------------------------------------------------------- 既定は C1 の前と同じ(データセット)
@pytest.mark.parametrize('jitter', [True, False])
def test_dataset_default_same_as_base(base, dsroot, jitter):
    td, _ = base
    old = td.JetFormerDataset(dsroot, 'train', color_jitter=jitter, stride=3)
    for kw in ({}, {'aug': None}, {'aug': {'top_brightness': 0.0}},
               {'aug': {'brightness': 0, 'contrast': 0, 'shift_px': 0, 'tof_noise': 0, 'top_brightness': 0}}):
        ds = JetFormerDataset(dsroot, 'train', color_jitter=jitter, stride=3, **kw)
        assert ds.aug is None and ds.index == old.index
        a, ra = _items(old, range(len(old)), 5)
        b, rb = _items(ds, range(len(ds)), 5)
        assert _same_items(a, b)
        assert np.array_equal(ra, rb)                         # 乱数を余分に引かない


def test_dataset_aug_1021_same_as_base(base, dsroot):
    """10/21 の拡張(上 1/3 なし): 鍵が無くても top_brightness 0 でも、C1 の前と同じサンプル・同じ乱数"""
    td, _ = base
    old = td.JetFormerDataset(dsroot, 'train', color_jitter=True, stride=3, aug=AUG_1021)
    for aug in (AUG_1021, dict(AUG_1021, top_brightness=0.0)):
        ds = JetFormerDataset(dsroot, 'train', color_jitter=True, stride=3, aug=aug)
        a, ra = _items(old, range(len(old)), 11)
        b, rb = _items(ds, range(len(ds)), 11)
        assert _same_items(a, b)
        assert np.array_equal(ra, rb)


# ---------------------------------------------------------------- 上 1/3 の明るさ(単体)
def test_top_ramp_shape():
    w = top_ramp(224)[:, 0, 0]
    assert w.shape == (224,) and w[0] == 1.0
    assert np.all(np.diff(w) <= 0)                            # 上ほど強い
    assert w[74] > 0 and np.all(w[75:] == 0)                  # 224 / 3 = 74.7 行より下は 0
    assert abs(float(w[37]) - (1 - 37 / (224 / 3))) < 1e-6    # 直線


def test_augment_image_top_only_upper_third():
    img = np.full((224, 224, 3), 100, np.uint8)
    assert augment_image(img, top_brightness=0.0) is img      # 0 なら何もしない
    np.random.seed(4)
    s0 = np.random.get_state()[1].copy()
    out = augment_image(img, top_brightness=0.0)
    assert np.array_equal(np.random.get_state()[1], s0)       # 0 なら乱数を引かない
    for seed in range(20):
        np.random.seed(seed)
        u = np.random.uniform(-0.15, 0.15)                    # 1 回だけ引く値
        after_one = np.random.get_state()[1].copy()
        np.random.seed(seed)
        out = augment_image(img, top_brightness=0.15)
        assert np.array_equal(np.random.get_state()[1], after_one)          # 乱数はちょうど 1 回
        assert np.array_equal(out[75:], img[75:])                           # 上 1/3 より下は変えない
        exp0 = np.clip(np.float32(100) + np.float32(u * 255.0), 0, 255).astype(np.uint8)
        assert np.all(out[0] == exp0)                                       # 一番上の行は ±F × 255 の全部
        d = np.abs(out[:, 0, 0].astype(int) - 100)
        assert np.all(np.diff(d) <= 0) and d.max() <= int(0.15 * 255) + 1   # 上ほど強く、なだらかに
        assert np.all(out == out[:, :1, :1])                                # 行の中は同じ量(横・色で変えない)


def test_augment_image_top_keeps_brightness_draw():
    """乱数の順(コントラスト → 明るさ → 上 1/3 → ずれ): 上 1/3 を足しても、全体の明るさの値は同じ"""
    img = np.full((224, 224, 3), 120, np.uint8)
    for seed in range(10):
        np.random.seed(seed)
        a = augment_image(img, brightness=0.1, contrast=0.2)
        np.random.seed(seed)
        b = augment_image(img, brightness=0.1, contrast=0.2, top_brightness=0.15)
        assert np.array_equal(a[75:], b[75:])


def test_dataset_top_changes_only_image_top(dsroot):
    base_ds = JetFormerDataset(dsroot, 'train')
    ds = JetFormerDataset(dsroot, 'train', aug={'top_brightness': 0.15})
    assert ds.aug == {'brightness': 0.0, 'contrast': 0.0, 'shift_px': 0.0, 'tof_noise': 0.0, 'top_brightness': 0.15}
    changed = 0
    for i in range(10):
        np.random.seed(i)
        a, b = base_ds[i], ds[i]
        assert torch.equal(a[0][:, 75:], b[0][:, 75:])        # 画像の上 1/3 より下・ToF・IMU・状態・ジャンル・正解は同じ
        assert all(torch.equal(x, y) for x, y in zip(a[1:], b[1:]))
        changed += int(not torch.equal(a[0][:, :75], b[0][:, :75]))
    assert changed >= 9


# ---------------------------------------------------------------- train.py の引数
def test_train_args_top_default_unused():
    args = type('A', (), dict(train.BENCH_ARGS))()
    assert train.BENCH_ARGS['aug_top_brightness'] == 0.0
    assert train.bench_args_used(args) == {} and train.aug_of(args) is None
    args.aug_brightness, args.aug_contrast, args.aug_shift_px, args.aug_tof_noise = 0.2, 0.2, 4, 0.02
    assert train.aug_of(args) == AUG_1021                     # 上 1/3 を使わないなら 10/21 と同じ dict(鍵なし)
    args.aug_top_brightness = 0.15
    assert train.aug_of(args) == dict(AUG_1021, top_brightness=0.15)
    assert train.bench_args_used(args)['aug_top_brightness'] == 0.15
    only = type('A', (), dict(train.BENCH_ARGS, aug_top_brightness=0.1))()
    assert train.aug_of(only) == {'brightness': 0.0, 'contrast': 0.0, 'shift_px': 0, 'tof_noise': 0.0, 'top_brightness': 0.1}
    args.aug_top_brightness = -0.1
    with pytest.raises(SystemExit):
        train.aug_of(args)
    # 前からの last.pth(新しい引数の記録なし)で再開できる。AUG の last.pth を AUG2 で再開はできない(違う学習)
    old = {'data': 'd', 'epochs': 30}
    assert train.args_diff(old, dict(old, **train.bench_args_used(type('A', (), dict(train.BENCH_ARGS))()))) == {}
    old_aug = dict(old, aug_brightness=0.15)
    assert train.args_diff(old_aug, dict(old_aug, aug_top_brightness=0.15)) == {'aug_top_brightness': (None, 0.15)}


# ---------------------------------------------------------------- train.py を通す(CPU)
def _argv(dsroot, out, extra=(), epochs=2):
    return ['train.py', '--data', dsroot, '--out', str(out), '--epochs', str(epochs), '--batch', '4', '--workers', '0',
            '--max-steps', '2', '--stride', '3', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy-cfg', json.dumps(SMALL_V0), '--seed', '0'] + list(extra)


def _run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    mod.STOP['flag'] = False
    mod.main()


def _strip(res):
    res = dict(res)
    for k in ('duration_s', 'peak_mem_mb'):
        res.pop(k, None)
    res['history'] = [{k: v for k, v in h.items() if k != 'seconds'} for h in res['history']]
    return res


def _same_state(p, q):
    a, b = torch.load(p, map_location='cpu', weights_only=True), torch.load(q, map_location='cpu', weights_only=True)
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize('extra', [[], AUG_1021_ARGS], ids=['default', 'aug1021'])
def test_train_main_same_as_base(base, dsroot, tmp_path, monkeypatch, extra):
    """既定(と 10/21 の AUG)の学習が C1 の前と同じ: result.json(損失の履歴)・best.pth の重み・last.pth の引数・model_config"""
    _, old_train = base
    o, n = tmp_path / 'old', tmp_path / 'new'
    _run(old_train, _argv(dsroot, o, extra), monkeypatch)
    _run(train, _argv(dsroot, n, extra), monkeypatch)
    ro = json.loads((o / 'result.json').read_text(encoding='utf-8'))
    rn = json.loads((n / 'result.json').read_text(encoding='utf-8'))
    assert _strip(ro) == _strip(rn)                           # 同じ損失(学習・検証・ジャンル別)・同じ数
    assert len(rn['history']) == 2
    assert _same_state(o / 'best.pth', n / 'best.pth')        # 同じ重み
    co, cn = torch.load(o / 'last.pth', weights_only=False), torch.load(n / 'last.pth', weights_only=False)
    assert co['args'] == cn['args'] and 'aug_top_brightness' not in cn['args']
    assert all(torch.equal(co['model'][k], cn['model'][k]) for k in co['model'])
    assert json.loads((o / 'model_config.json').read_text()) == json.loads((n / 'model_config.json').read_text())
    if not extra:
        assert 'bench_options' not in rn
    else:
        assert rn['bench_options']['aug'] == AUG_1021 and 'top_brightness' not in rn['bench_options']['aug']


def _rewind(path):
    """last.pth を 1 エポック目のあとに戻す(電源断で 2 エポック目の途中で止まった形)"""
    ck = torch.load(path, weights_only=False)
    ck['epoch'], ck['history'] = 1, ck['history'][:1]
    torch.save(ck, path)


@pytest.mark.parametrize('extra', [[], AUG_1021_ARGS], ids=['default', 'aug1021'])
def test_resume_from_base_lastpth(base, dsroot, tmp_path, monkeypatch, extra, capsys):
    """C1 の前の train.py が書いた last.pth を、C1 のあとの train.py で --resume できる"""
    _, old_train = base
    out = tmp_path / 'run'
    _run(old_train, _argv(dsroot, out, extra), monkeypatch)
    _rewind(out / 'last.pth')
    capsys.readouterr()
    _run(train, _argv(dsroot, out, extra) + ['--resume'], monkeypatch)
    assert '再開: 1 エポック目まで済み' in capsys.readouterr().out
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert res['state'] == 'done' and res['epochs_done'] == 2


def test_train_main_aug2_and_resume(dsroot, tmp_path, monkeypatch, capsys):
    out = tmp_path / 'aug2'
    _run(train, _argv(dsroot, out, AUG2_ARGS), monkeypatch)
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert res['bench_options']['aug'] == {'brightness': 0.15, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02,
                                           'top_brightness': 0.15}
    ck = torch.load(out / 'last.pth', weights_only=False)
    assert ck['args']['aug_top_brightness'] == 0.15
    # 同じ引数なら再開できる
    _rewind(out / 'last.pth')
    capsys.readouterr()
    _run(train, _argv(dsroot, out, AUG2_ARGS) + ['--resume'], monkeypatch)
    assert '再開: 1 エポック目まで済み' in capsys.readouterr().out
    assert json.loads((out / 'result.json').read_text(encoding='utf-8'))['epochs_done'] == 2
    # 上 1/3 の値が違う・上 1/3 を外した、では再開しない
    for extra in (AUG2_ARGS[:-1] + ['0.1'], AUG2_ARGS[:-2]):
        with pytest.raises(SystemExit, match='前回と引数が違います'):
            _run(train, _argv(dsroot, out, extra) + ['--resume'], monkeypatch)
    # 上 1/3 は学習だけ(既定のままの学習と検証の損失の数は同じ・正解は変えない)
    plain = tmp_path / 'plain'
    _run(train, _argv(dsroot, plain), monkeypatch)
    rp = json.loads((plain / 'result.json').read_text(encoding='utf-8'))
    assert rp['train_samples'] == res['train_samples'] and rp['val_samples'] == res['val_samples']


def test_new_td_is_the_one_train_uses():
    assert train.JetFormerDataset is new_td.JetFormerDataset
