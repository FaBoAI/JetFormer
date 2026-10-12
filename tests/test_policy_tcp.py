"""TCP / TransFuser++ 型の方策(jetformer/policies/tcp.py)と、その正解・較正(tcp_labels.py)のテスト。
CPU だけ・GPU・実データ・ネットなし(合成の can_bus の記録とデータセットを tmp に作る)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_policy_tcp.py

車の約束: 自車の座標 x = 前・y = 左、ヨーレート z 正 = 左旋回、eRPM 負 = 前進(v = −eRPM × 0.5e-3)、ステア > 中心 = 左
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from jetformer import policies  # noqa: E402
from jetformer.policies import controller, post, runtime, tcp, tcp_labels  # noqa: E402
from jetformer.train_data import JetFormerDataset, impact_mask  # noqa: E402

T0_US = 1_791_600_000_000_000
KV = 0.5e-3
SMALL = {'dim': 32, 'enc_layers': 1, 'dec_layers': 1, 'heads': 2, 'ffn': 64, 'dropout': 0.0}
MODES = [{'id': i, 'name': n, 'description': ''} for i, n in enumerate(['未設定', '駐車(1)', '駐車(2)', '駐車(3)', '高速走行', '低速走行'])]


@pytest.fixture(autouse=True)
def derived_tmp(tmp_path, monkeypatch):
    """正解・較正のキャッシュを tmp に(data/derived/ に書かない)"""
    monkeypatch.setattr(tcp_labels, 'DERIVED_ROOT', tmp_path / 'derived')
    return tmp_path / 'derived'


# ==================================================================== 合成のデータ
def _write_scene(root, scene, t, v, gz, steer=None, thr=None, ax=None, imu_keep=None, mode=5):
    """can_bus の JSON(IMU 100 Hz・車両は 3 ms 後)と aligned の npz(30 fps)を書く"""
    root.mkdir(parents=True, exist_ok=True)
    (root / 'can_bus').mkdir(exist_ok=True)
    (root / 'aligned').mkdir(exist_ok=True)
    n = len(t)
    steer = np.full(n, 1441.0) if steer is None else np.asarray(steer, float)
    thr = np.full(n, 1496.0) if thr is None else np.asarray(thr, float)
    ax = np.zeros(n) if ax is None else np.asarray(ax, float)
    keep = np.ones(n, bool) if imu_keep is None else imu_keep
    imu = [{'utime': int(T0_US + t[i] * 1e6), 'linear_accel': [float(ax[i]), 0.0, 9.8],
            'rotation_rate': [0.0, 0.0, float(gz[i])], 'temperature': 40.0} for i in range(n) if keep[i]]
    veh = [{'utime': int(T0_US + t[i] * 1e6 + 3000), 'steering_us': round(float(steer[i]), 3),
            'throttle_us': round(float(thr[i]), 3), 'mode_us': 974, 'erpm': round(float(-v[i] / KV), 3),
            'ai_driving': False, 'valid': [True, True, True]} for i in range(n)]
    (root / 'can_bus' / (scene + '_ms_imu.json')).write_text(json.dumps(imu))
    (root / 'can_bus' / (scene + '_vehicle_monitor.json')).write_text(json.dumps(veh))
    tf = np.arange(0.0, t[-1], 1 / 30)
    nf = len(tf)
    vehicle = np.stack([np.interp(tf, t, steer), np.interp(tf, t, thr), np.full(nf, 974.0),
                        np.interp(tf, t, -np.asarray(v) / KV)], 1).astype(np.float32)
    j = np.clip(np.searchsorted(t, tf, 'right')[:, None] - 10 + np.arange(10)[None], 0, n - 1)
    imu_a = np.zeros((nf, 10, 6), np.float32)
    imu_a[:, :, 0], imu_a[:, :, 2], imu_a[:, :, 5] = ax[j], 9.8, np.asarray(gz)[j]
    fut = np.stack([vehicle[np.minimum(np.arange(nf)[:, None] + 1 + np.arange(10)[None], nf - 1), c] for c in (0, 1)], -1)
    np.savez(root / 'aligned' / (scene + '.npz'), timestamps_us=(T0_US + tf * 1e6).astype(np.int64),
             cam_files=np.array(['img.jpg'] * nf), tof=np.zeros((nf, 42, 54), np.uint16), imu=imu_a, vehicle=vehicle,
             future_actions=fut.astype(np.float32), future_valid=np.arange(nf) < nf - 10,
             vehicle_valid=np.ones(nf, bool), erpm_valid=np.ones(nf, bool), mode=np.full(nf, mode, np.int16))
    return tf


def _finish(root, train, val):
    cv2.imwrite(str(root / 'img.jpg'), np.full((16, 16, 3), 128, np.uint8))
    (root / 'dataset_card.json').write_text(json.dumps({'modes': MODES}, ensure_ascii=False), encoding='utf-8')
    (root / 'splits.json').write_text(json.dumps({'train': list(train), 'val': list(val)}))
    return str(root)


def _times(seconds):
    return np.round(np.arange(0.0, seconds, 0.01), 6)


def _const(t, x):
    return np.full(len(t), float(x))


def _drive_scene(seconds=20.0, lag=0.04, slope=0.004, sat=1.5, bias=0.002, phase=0.0):
    """較正用: 2 秒止まってから、ステアを ±550 µs で振り、速さ 1.1〜3.9 m/s で走る。
    κ(t) = clip(slope·ずれ(t − lag), ±sat)、gz = κ·v + bias、スロットル = 1530 − 80·v − 25·(v_tgt − v)/0.33"""
    t = _times(seconds)
    run = np.clip(t - 2.0, 0.0, None)
    v = np.where(t < 2.0, 0.0, (2.5 + 1.4 * np.sin(2 * np.pi * run / 11 + phase)) * np.minimum(1.0, run))
    off = np.where(t < 2.0, 0.0, 550.0 * np.sin(2 * np.pi * run / 7 + phase))
    kappa = np.clip(slope * np.interp(t - lag, t, off), -sat, sat)
    gz = kappa * v + bias
    tv = t + 0.003                                                     # 車両の時刻(_write_scene と同じ)
    vt = tcp_labels.target_speed(tv, tv, v)
    thr = np.where(v >= 0.3, 1530.0 - 80.0 * v - 25.0 * (vt - v) / 0.33, 1496.0)
    return t, v, gz, 1441.0 + off, thr


@pytest.fixture(scope='module')
def drive_ds(tmp_path_factory):
    """較正と学習の通しに使う合成のデータセット(学習 2・検証 1 シーン)"""
    root = tmp_path_factory.mktemp('tcpds') / 'drive_ds'
    for k, s in enumerate(['20261012-100000-0001', '20261012-100000-0002', '20261012-110000-0001']):
        t, v, gz, st, th = _drive_scene(phase=k * 1.3)
        _write_scene(root, s, t, v, gz, st, th, mode=4 + k % 2)
    return _finish(root, ['20261012-100000-0001', '20261012-100000-0002'], ['20261012-110000-0001'])


# ==================================================================== モデル
def _x(batch=2, num_modes=6, seed=7):
    return runtime.example_inputs(num_modes, batch=batch, seed=seed)


def test_registry_entry_is_ready_and_default_size():
    spec = policies.get('tcp')
    assert spec.ready() == (True, '')
    assert spec.outputs == ('actions', 'waypoints', 'speed_logits') and spec.postprocess == 'tcp_controller'
    m = policies.build_policy('tcp', 6)
    total, trainable = runtime.count_params(m)
    assert abs(total / 1e6 - 15.6) <= 0.3 and total == trainable            # spec.md §3.2.1: 15.6M ± 0.3M
    out = runtime.check_forward(m, 6)                                        # eval・有限・EXPORT_OUTPUTS が全部ある
    assert out['actions'].shape == (2, 10, 2) and out['waypoints'].shape == (2, 10, 2)
    assert out['speed_logits'].shape == (2, 10)
    names = set(m.state_dict())
    for k in ('image_backbone.0.weight', 'image_proj.weight', 'tof_encoder.net.0.0.weight', 'imu_mlp.0.weight',
              'state_mlp.0.weight', 'mode_embed.weight', 'pos', 'queries', 'heads_wp.weight', 'heads_v.weight', 'heads.weight'):
        assert k in names                                                    # v0 と同じトークン化の名前・一番上のジャンルの層
    assert m.queries.shape == (1, 21, 256) and 'tril' not in names and 'bins' not in names


def test_cumsum_op_and_matmul_give_the_same_waypoints():
    torch.manual_seed(0)
    a = tcp.build(6, cumsum='op', **SMALL).eval()
    b = tcp.build(6, cumsum='matmul', **SMALL).eval()
    b.load_state_dict(a.state_dict())
    x = _x()
    with torch.no_grad():
        ya, yb = a(**x), b(**x)
    for k in ya:
        assert torch.allclose(ya[k], yb[k], atol=1e-6)
    with pytest.raises(ValueError):
        tcp.build(6, cumsum='loop', **SMALL)


def test_twohot():
    bins = torch.tensor(tcp.bin_centers(10, 4.5))
    q = tcp.twohot(torch.tensor([2.3, 4.6, -0.1, 0.0, 4.5, 1.0]), bins)
    assert torch.allclose(q.sum(1), torch.ones(6))
    assert q[0, 4] == pytest.approx(0.4, abs=1e-5) and q[0, 5] == pytest.approx(0.6, abs=1e-5)
    assert q[1, 9] == 1.0 and q[2, 0] == 1.0 and q[3, 0] == 1.0 and q[4, 9] == 1.0 and q[5, 2] == pytest.approx(1.0)
    assert (q[0] > 0).sum() == 2                                             # 隣り合う2クラスだけ
    assert torch.allclose((q * bins).sum(1)[:1], torch.tensor([2.3]), atol=1e-5)   # 期待値が元の速さ


def _extras(batch, valid, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {'y_waypoints': torch.cumsum(torch.rand(batch, 10, 2, generator=g) * 0.3, 1),
            'y_v_tgt': torch.rand(batch, generator=g) * 4.0, 'y_wp_valid': torch.tensor(valid)}


def test_policy_loss_masks_invalid_frames():
    torch.manual_seed(0)
    m = tcp.build(6, **SMALL)
    out = m(**_x())
    target = torch.zeros(2, 10, 2)
    per = m.policy_loss(out, target, _extras(2, [True, False]))
    l1 = runtime.l1_per_sample(out['actions'], target)
    assert per.shape == (2,) and torch.allclose(per[1], l1[1])                # 無効: 通過点・速度の損失は 0
    assert per[0] > l1[0] + 0.1                                              # 有効: 0.5·wp + 0.1·ce が足される
    assert set(m.loss_parts) == {'wp_m', 'ce'}
    per2 = m.policy_loss(out, target, _extras(2, [False, False]))
    assert torch.allclose(per2, l1) and float(m.loss_parts['wp_m']) == 0.0
    assert runtime.sample_loss(m, out, target, _extras(2, [True, True])).shape == (2,)
    with pytest.raises(RuntimeError, match='正解'):
        m.policy_loss(out, target, None)


def test_genre_heads_select_and_expand():
    torch.manual_seed(0)
    m6 = tcp.build(6, **SMALL).eval()
    x = _x(batch=1)
    x4 = dict(x, mode=torch.nn.functional.one_hot(torch.tensor([4]), 6).float())
    x5 = dict(x, mode=torch.nn.functional.one_hot(torch.tensor([5]), 6).float())
    with torch.no_grad():
        y4, y5 = m6(**x4), m6(**x5)
        for k in y4:
            assert not torch.allclose(y4[k], y5[k])
        # ジャンル 5 の頭の行だけを変えるとジャンル 5 の出力だけが変わる(行の並び = ジャンル × 行)。
        # 重みに定数を足しても hs(LayerNorm の後で平均 0)との積は変わらないので、バイアスを動かす
        for layer, rows in policies.get('tcp').genre_keys['heads'].items():
            getattr(m6, layer).bias[5 * rows:(5 + 1) * rows] += 1.0
        z4, z5 = m6(**x4), m6(**x5)
    for k in ('actions', 'waypoints', 'speed_logits'):
        assert torch.allclose(z4[k], y4[k])
    assert torch.allclose(z5['actions'], y5['actions'] + 1.0, atol=1e-5)
    assert torch.allclose(z5['speed_logits'], y5['speed_logits'] + 1.0, atol=1e-5)
    assert torch.allclose(z5['waypoints'], y5['waypoints'] + torch.arange(1, 11.0).view(1, 10, 1), atol=1e-4)   # 累積和
    # ジャンルを 6 → 8 に広げ、7 を 5 から写すと、7 の出力 = 元の 5 の出力
    m8 = tcp.build(8, **SMALL).eval()
    st = runtime.expand_modes(m6.state_dict(), m8.state_dict(), policies.get('tcp').genre_keys, {7: 5})
    m8.load_state_dict(st)
    x7 = dict(x, mode=torch.nn.functional.one_hot(torch.tensor([7]), 8).float())
    x8_5 = dict(x, mode=torch.nn.functional.one_hot(torch.tensor([5]), 8).float())
    with torch.no_grad():
        a, b = m8(**x7), m8(**x8_5)
    for k in a:
        assert torch.allclose(a[k], z5[k], atol=1e-5) and torch.allclose(b[k], z5[k], atol=1e-5)


class _Fixed(torch.nn.Module):
    """決まった特徴を返すだけ(凍結した ResNet の出力を1回だけ計算して使い回す。CPU のテストの速さのため)"""

    def __init__(self, feat):
        super().__init__()
        self.register_buffer('feat', feat)

    def forward(self, image):
        return self.feat


def test_loss_decreases_on_a_tiny_batch():
    torch.manual_seed(0)
    m = tcp.build(6, **SMALL)
    x = _x(batch=2, seed=3)
    with torch.no_grad():
        m.image_backbone = _Fixed(m.image_backbone.eval()(x['image']))
    target = torch.tensor(np.random.default_rng(0).uniform(-0.3, 0.3, (2, 10, 2)), dtype=torch.float32)
    ex = _extras(2, [True, True], seed=4)
    opt = torch.optim.Adam(runtime.param_groups(m, 2e-3), lr=2e-3)
    losses = []
    for _ in range(25):
        out = runtime.forward(m, x, target=target, train=True)
        loss = runtime.sample_loss(m, out, target, ex).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], losses


def _onnx_ops(path):
    import onnx
    return {n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node}


def test_onnx_export_cpu_has_three_outputs(tmp_path):
    torch.manual_seed(0)
    m = tcp.build(6, **SMALL)
    io = runtime.export_onnx(m, tmp_path / 'jetformer.onnx', 6)
    assert io['inputs'] == list(policies.BASE_INPUTS)
    assert io['outputs'] == ['actions', 'waypoints', 'speed_logits']
    assert io['output_shapes'] == {'actions': [1, 10, 2], 'waypoints': [1, 10, 2], 'speed_logits': [1, 10]}
    assert policies.check_io(io['input_shapes'], io['output_shapes'], 6) == (True, '')
    ops = _onnx_ops(tmp_path / 'jetformer.onnx')
    assert 'CumSum' in ops
    assert not ops & {'Loop', 'If', 'Scan', 'GRU', 'LSTM', 'RNN'}            # TensorRT でループ・分岐を作らない
    assert not (tmp_path / 'jetformer.onnx.tmp').exists()


# ==================================================================== 正解(numpy)
def test_labels_on_circles_and_straight(tmp_path):
    root = tmp_path / 'ds'
    t = _times(6.0)
    for s, w in (('logL-0001', 1.0), ('logR-0001', -1.0), ('logS-0001', 0.0)):
        _write_scene(root, s, t, _const(t, 2.0), _const(t, w))
    root = _finish(root, ['logL-0001', 'logR-0001'], ['logS-0001'])
    tcp_labels.build_labels(root)
    lab = tcp_labels.labels_by_scene(root)
    tau = 0.1 * np.arange(1, 11)
    for s, w in (('logL-0001', 1.0), ('logR-0001', -1.0)):
        d = lab[s]
        ok = d['wp_valid']
        assert ok.sum() > 100
        r = 2.0 / abs(w)                                                     # 半径 = v / ω = 2 m
        exp = np.stack([r * np.sin(abs(w) * tau), np.sign(w) * r * (1 - np.cos(abs(w) * tau))], 1)
        assert np.abs(d['waypoints'][ok] - exp).max() < 0.03                # 円の上(推測航法の誤差 数 cm)
        assert np.all(np.sign(d['waypoints'][ok, :, 1]) == np.sign(w))      # 左旋回 = y 正、右 = y 負
        assert np.all(d['waypoints'][ok, :, 0] > 0)                          # 前進 = x 正
        assert np.allclose(d['v_tgt'][ok], 2.0, atol=1e-3)
    d = lab['logS-0001']
    assert np.abs(d['waypoints'][d['wp_valid'], :, 1]).max() < 1e-3
    assert np.allclose(d['waypoints'][d['wp_valid'], -1, 0], 2.0, atol=0.03)


def test_label_validity_rules(tmp_path):
    root = tmp_path / 'ds'
    t = _times(6.0)
    v = _const(t, 2.0)
    tf = {}
    tf['logA-0001'] = _write_scene(root, 'logA-0001', t, v, _const(t, 0.0))                      # きれい
    tf['logB-0001'] = _write_scene(root, 'logB-0001', t, v, _const(t, 0.0),
                                   imu_keep=~((t > 2.0) & (t < 2.08)))                           # IMU の切れ目
    tf['logC-0001'] = _write_scene(root, 'logC-0001', t, np.where((t >= 3.0) & (t <= 3.3), -0.5, 2.0), _const(t, 0.0))  # 後退
    tf['logD-0001'] = _write_scene(root, 'logD-0001', t, v, _const(t, 0.0), ax=np.where(np.isclose(t, 3.0), 30.0, 0.0))  # 衝突
    tf['logE-0001'] = _write_scene(root, 'logE-0001', t, v, np.where(np.isclose(t, 3.0), 20.0, 0.0))  # ジャイロの飽和
    scenes = sorted(tf)
    root = _finish(root, scenes[:3], scenes[3:])
    summary = tcp_labels.build_labels(root)
    lab = tcp_labels.labels_by_scene(root)
    a, ta = lab['logA-0001']['wp_valid'], tf['logA-0001']
    assert not a[0]                                                          # 車両の記録が 3 ms 後から: 記録の外
    assert np.all(a[(ta > 0.01) & (ta <= 4.98)]) and not np.any(a[ta > 5.0])   # シーンの終わり(1 秒の窓)
    b = lab['logB-0001']['wp_valid']
    tb = tf['logB-0001']
    assert not np.any(b[(tb > 1.1) & (tb < 2.0)]) and np.all(b[((tb > 0.01) & (tb < 0.95)) | ((tb > 2.1) & (tb < 4.9))])
    c = lab['logC-0001']['wp_valid']
    tc = tf['logC-0001']
    assert not np.any(c[(tc > 2.05) & (tc < 3.25)]) and np.all(c[((tc > 0.01) & (tc < 1.9)) | ((tc > 3.4) & (tc < 4.9))])
    d = lab['logD-0001']['wp_valid']
    td = tf['logD-0001']
    assert not np.any(d[(td > 1.05) & (td < 4.95)]) and np.all(d[(td > 0.01) & (td < 0.95)])   # 窓が [衝突 − 1 秒, + 2 秒] に掛かる
    e = lab['logE-0001']['wp_valid']
    te = tf['logE-0001']
    assert not np.any(e[(te > 2.05) & (te < 2.98)]) and np.all(e[((te > 0.01) & (te < 1.95)) | ((te > 3.05) & (te < 4.9))])
    r = summary['invalid_reasons']
    assert r['imu_gap'] > 0 and r['reverse'] > 0 and r['impact'] > 0 and r['gyro_sat'] > 0 and r['outside'] > 0
    assert all(np.all(lab[s]['waypoints'][~lab[s]['wp_valid']] == 0) for s in scenes)   # 無効の通過点は 0


def test_gyro_bias_is_removed_per_log(tmp_path):
    root = tmp_path / 'ds'
    t = _times(7.0)
    v = np.where(t < 2.0, 0.0, 2.0)
    _write_scene(root, 'logB-0001', t, v, _const(t, 0.05))                    # 2 秒止まる(200 サンプル)、ずっと 0.05 rad/s
    root = _finish(root, ['logB-0001'], [])
    summary = tcp_labels.build_labels(root)
    assert summary['gyro_bias_rad_s']['logB'][0] == pytest.approx(0.05)
    d = tcp_labels.labels_by_scene(root)['logB-0001']
    moving = d['wp_valid'] & (d['v_tgt'] > 1.9)
    assert moving.sum() > 50 and np.abs(d['waypoints'][moving, :, 1]).max() < 5e-3   # 除かなければ 1 秒で約 5 cm


def test_target_speed_is_mean_of_0_2_to_0_5_s_and_clipped(tmp_path):
    root = tmp_path / 'ds'
    t = _times(7.0)
    v = 1.0 + 0.8 * t
    tf = _write_scene(root, 'logV-0001', t, v, _const(t, 0.0))
    root = _finish(root, ['logV-0001'], [])
    tcp_labels.build_labels(root)
    d = tcp_labels.labels_by_scene(root, keys=('v_tgt', 'wp_valid'))['logV-0001']
    lo = tf < 4.0
    tv = tf + 0.003
    exp = np.clip(1.0 + 0.8 * (tf + 0.35 - 0.003), 0, 4.5)
    assert np.allclose(d['v_tgt'][lo & (tf > 0.05)], exp[lo & (tf > 0.05)], atol=2e-3)
    assert np.all(d['v_tgt'][(tv > 4.2) & (tf < 5.9)] == np.float32(4.5))    # 4.5 m/s で切る
    assert d['v_tgt'].dtype == np.float32


def test_label_cache_rebuilds_when_splits_change(tmp_path):
    root = tmp_path / 'ds'
    t = _times(4.0)
    for s in ('logA-0001', 'logA-0002'):
        _write_scene(root, s, t, _const(t, 2.0), _const(t, 0.3))
    _write_scene(root, 'logX-0001', t, _const(t, 2.0), _const(t, 0.3))       # split に無いシーン
    root = _finish(root, ['logA-0001'], ['logA-0002'])
    path, built = tcp_labels.ensure_labels(root)
    assert built and path.exists() and path.parent.name == 'ds'
    assert not tcp_labels.ensure_labels(root)[1]                             # 済み → 何もしない
    lab = tcp_labels.load_labels(path)
    assert set(lab['scene']) == {'logA-0001', 'logA-0002'} and int(lab['version']) == 1
    for k in ('frame', 'waypoints', 'v_tgt', 'v_now', 'wp_valid', 'splits_sha1'):
        assert k in lab
    with open(os.path.join(root, 'splits.json'), 'w') as f:
        json.dump({'train': ['logA-0002'], 'val': ['logA-0001']}, f)
    assert tcp_labels.labels_fresh(root)[0] is False
    assert tcp_labels.ensure_labels(root)[1]                                 # sha1 が変わった → 作り直す
    extra = tcp_labels.labels_by_scene(root, scenes=['logX-0001'])['logX-0001']    # npz に無いシーン → 0 と無効
    n = len(tcp_labels.aligned_times(root, 'logX-0001'))
    assert extra['waypoints'].shape == (n, 10, 2) and not extra['wp_valid'].any() and not extra['v_tgt'].any()
    p2, b2 = tcp_labels.ensure_labels(root, wp_steps=5, wp_dt=0.2)          # 別の形は別のファイル
    assert b2 and p2.name != path.name and tcp_labels.load_labels(p2)['waypoints'].shape[1] == 5


def test_impact_mask_copy_matches_train_data():
    rng = np.random.default_rng(0)
    ts = (np.arange(300) * 33333).astype(np.int64)
    imu = rng.normal(0, 5, (300, 10, 6)).astype(np.float32)
    imu[[40, 200], 3, 0] = 40.0
    for g in (2.5, 4.0):
        assert np.array_equal(tcp_labels.impact_mask_frames(ts, imu, g), impact_mask(ts, imu, g))


def test_prepare_needs_can_bus(tmp_path):
    root = tmp_path / 'nocan'
    (root / 'aligned').mkdir(parents=True)
    _finish(root, [], [])
    with pytest.raises(SystemExit, match='can_bus'):
        tcp.prepare(str(root), policies.get('tcp').resolve({}))


# ==================================================================== 較正(numpy)
def test_calibration_recovers_synthetic_car(drive_ds):
    cal = tcp_labels.fit_calibration(drive_ds)
    st, ff = cal['steer_table'], cal['throttle_ff']
    assert st['lag_ms'] == 40 and st['r2_by_lag']['40'] > 0.95                # 遅れ 40 ms を見つける
    assert st['center_us'] == 1441.0 and ff['neutral_us'] == 1496.0           # model_config と同じ中心・中立
    assert st['slope_left_right_per_us'] == pytest.approx([0.004, 0.004], abs=2e-4)
    offs = np.array(st['offset_us'])
    assert offs[0] == -575.0 and offs[-1] == 575.0 and len(offs) == 24
    for row in st['curv']:
        row = np.array(row)
        assert np.all(np.diff(row) > 0)                                       # 狭義単調(逆引きできる)
        lin = np.abs(offs) <= 300
        assert np.abs(row[lin] - 0.004 * offs[lin]).max() < 0.05
        assert row[-1] == pytest.approx(1.5, abs=0.02) and row[0] == pytest.approx(-1.5, abs=0.02)   # いっぱいに切った曲率
        # 逆引き: 表の外は端(いっぱい)に張り付く、中は戻る
        assert np.interp(9.0, row, offs) == 575.0 and np.interp(-9.0, row, offs) == -575.0
        assert np.interp(0.4, row, offs) == pytest.approx(100.0, abs=8.0)
    assert ff['c0'] == pytest.approx(1530.0, abs=1.0) and ff['c1'] == pytest.approx(-80.0, abs=0.5)
    assert ff['c2'] == pytest.approx(-25.0, abs=0.5) and ff['train_mae_us'] < 1.0
    assert 250.0 < ff['fwd_max_us'] < 420.0
    assert cal['gyro_bias_rad_s']['20261012-100000'][0] == pytest.approx(0.002)
    assert tcp_labels.action_reference(drive_ds, ['20261012-100000-0001', '20261012-100000-0002']) == \
        JetFormerDataset(drive_ds, 'train').action_reference()


def test_monotone_table_and_postprocess_shape(drive_ds):
    k = tcp_labels.monotone_table([0.0, -0.1, 0.2, 0.2, 0.1])
    assert np.all(np.diff(k) > 0) and k[1] == pytest.approx(1e-6)
    path, built = tcp_labels.ensure_calibration(drive_ds)
    assert built and not tcp_labels.ensure_calibration(drive_ds)[1]
    tcp_labels.ensure_labels(drive_ds)
    pp = tcp.finalize('run', policies.get('tcp').resolve({}), drive_ds)['postprocess']
    assert pp['kind'] == 'tcp_controller' and pp['version'] == 1 and pp['fps'] == 30 and pp['wp_dt'] == 0.1
    assert pp['speed_bins'] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    assert pp['pursuit'] == {'t_look_s': 0.25, 'l_min_m': 0.3, 'min_path_m': 0.2}
    assert pp['rules']['brake_us'] == 150.0 and pp['rules']['cap_ramp_mps2'] == 1.5
    assert pp['calib_source']['scenes'] == 'train' and len(pp['calib_source']['labels_sha1']) == 40
    assert set(pp['throttle_ff']) >= {'c0', 'c1', 'c2', 'tau_s', 'fwd_max_us'}
    json.dumps(pp)                                                            # model_config.json に書ける
    assert isinstance(controller.TCPPost(pp, (1441.0, 1496.0)), controller.TCPPost)


# ==================================================================== 学習の通し(CPU・合成データ・小さい形)
def test_train_main_end_to_end_tcp(drive_ds, tmp_path, monkeypatch):
    import train
    out = tmp_path / 'run'
    argv = ['train.py', '--data', drive_ds, '--out', str(out), '--epochs', '1', '--batch', '4', '--workers', '0',
            '--max-steps', '1', '--stride', '30', '--no-pretrained', '--no-command-input', '--impact-g', '2.5',
            '--policy', 'tcp', '--policy-cfg', json.dumps(SMALL), '--seed', '0']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    train.main()
    cfg = json.loads((out / 'model_config.json').read_text(encoding='utf-8'))
    res = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    assert cfg['policy'] == 'tcp' and cfg['outputs'] == ['actions', 'waypoints', 'speed_logits']
    assert cfg['output_shapes']['speed_logits'] == [1, 10] and cfg['inputs'] == list(policies.BASE_INPUTS)
    assert cfg['postprocess']['kind'] == 'tcp_controller'
    assert policies.drivable(cfg) == (True, '')
    assert set(res['history'][0]['train_parts']) == {'wp_m', 'ce'} and 'train_loss' in res['history'][0]
    model, cfg2, spec, pcfg = runtime.load_run(out)
    assert spec.name == 'tcp' and pcfg['dim'] == 32
    feed = {k: v.numpy() for k, v in runtime.example_inputs(cfg['num_modes'], batch=1).items()}
    pred = runtime.predict(model, feed)
    assert set(pred) == {'actions', 'waypoints', 'speed_logits'}
    stage = post.Stage(cfg2, (cfg2['steer_center_us'], cfg2['throttle_neutral_us']))
    assert isinstance(stage.post, controller.TCPPost)
    mp = stage.model_pred(pred)
    y0, info = stage.apply(pred, mp, 2.0, {'tcp_mode': 0, 'speed_cap': 0.0, 'ensemble_j': 0}, 0)
    assert np.array_equal(y0, mp) and info['branch'] == 'control'            # 走行の既定 = 比較で評価する actions
    y1, info1 = stage.apply(pred, mp, 2.0, {'tcp_mode': 1, 'speed_cap': 1.0, 'ensemble_j': 0}, 1)
    assert info1['branch'] == 'hybrid' and info1['cap'] == 1.0 and np.all(np.isfinite(y1))


def test_finalize_does_not_stop_after_training(tmp_path, capsys):
    root = tmp_path / 'ds'
    t = _times(4.0)
    _write_scene(root, 'logA-0001', t, _const(t, 0.0), _const(t, 0.0))      # 止まっているだけ: 較正できない
    root = _finish(root, ['logA-0001'], [])
    assert tcp.finalize(str(tmp_path / 'run'), policies.get('tcp').resolve({}), root) == {}
    assert '警告' in capsys.readouterr().out
