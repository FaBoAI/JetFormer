"""AI走行・収集まわりのテスト(実機・GPU不要。センサー・PCA9685・TensorRTは偽物を使う)

    cd ~/jetformer && python -m pytest -q tests
"""

import json
import threading
import time

import numpy as np
import pytest

from jetformer import clock, config, dataset, drive
from jetformer.recorder import Recorder
from jetformer.sensors.base import Sample
from jetformer.train_data import JetFormerDataset, normalize_inputs

REF = (1450.0, 1500.0)
PARAMS = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'steer_trim_us': 0.0, 'lookahead': 1, 'osc_percent': 0.0}


# ---------------------------------------------------------------- 偽物
class FakeRunner(object):
    def __init__(self, pred_us):
        self.actions = ((np.asarray(pred_us, np.float32) - drive.PWM_CENTER) / drive.PWM_SCALE)[None]
        self.calls = []

    def run_numpy(self, **kw):
        self.calls.append({k: v.copy() for k, v in kw.items()})
        return {'actions': self.actions.copy()}


class FakePCA(object):
    def __init__(self, fail=0):
        self.fail = fail
        self.pairs = []
        self.single = []

    def _maybe_fail(self):
        if self.fail:
            self.fail -= 1
            raise OSError(121, 'Remote I/O error')

    def set_pair_us(self, sc, s, tc, t):
        self._maybe_fail()
        self.pairs.append((s, t))

    def set_us(self, ch, us):
        self._maybe_fail()
        self.single.append((ch, us))

    def set_osc_scale(self, scale):
        pass


class FakeRecorder(object):
    recording = False

    def __init__(self):
        self.rows = []

    def add_ai(self, t_ns, data):
        self.rows.append((t_ns, data))


def make_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    d.params = dict(PARAMS)
    d.hb_token = 'tok'
    d.st = {'phase': 'driving', 'mode_id': 4}
    return d


def run_loop(d, runner, pca, veh_data, frames=12, lookahead=1, dry_run=False):
    """偽のセンサー値を30fpsで流しながら _loop を回す。戻り値: (例外 or None)"""
    d.params = dict(PARAMS, lookahead=lookahead)
    out = drive._Output(pca, config.PCA9685, REF, lambda: d.params)
    tof_buf, imu_buf, veh_buf = [__import__('collections').deque(maxlen=n) for n in (10, 60, 50)]
    cam_slot, cam_evt = [None], threading.Event()
    err = []

    def loop():
        try:
            d._loop(d.st, runner, normalize_inputs, pca, out, REF, 6, 0.0, cam_slot, cam_evt,
                    tof_buf, imu_buf, veh_buf, dry_run)
        except Exception as e:
            err.append(e)
    th = threading.Thread(target=loop, daemon=True)
    now = clock.now_ns()
    for k in range(12):                        # IMUの履歴を先に貯める
        imu_buf.append((now - (12 - k) * 10_000_000, np.zeros(6, np.float32)))
    th.start()
    for i in range(frames):
        if not th.is_alive():
            break
        t = clock.now_ns()
        tof_buf.append((t - 5_000_000, np.full((42, 54), 1000, np.uint16)))
        imu_buf.append((t - 2_000_000, np.zeros(6, np.float32)))
        veh_buf.append((t - 3_000_000, dict(veh_data)))
        cam_slot[0] = Sample(t, {'image': np.zeros((224, 224, 3), np.uint8), 'seq': i})
        cam_evt.set()
        d.heartbeat('tok')
        time.sleep(1 / 30)
    d._stop.set()
    th.join(3)
    d._stop.clear()
    return err[0] if err else None


def veh(select, valid=(True, True, True), pwm=(1450, 1500, 2000), erpm=0):
    return {'pwm': list(pwm), 'valid': list(valid), 'select': select, 'moving': False, 'erpm': erpm}


# ---------------------------------------------------------------- 走行ループ
def test_ch3_lost_on_ai_side_holds_neutral(monkeypatch, tmp_path):
    """送信機の電源断・圏外(SELECT=AI側のまま CH3 が無効)では推論の値を出さず中立にする"""
    d = make_driver(monkeypatch, tmp_path)
    pred = np.tile([[1700.0, 1650.0]], (10, 1))
    runner, pca = FakeRunner(pred), FakePCA()
    assert run_loop(d, runner, pca, veh(True, valid=(False, False, False))) is None
    assert runner.calls == []
    assert pca.pairs and all(p == REF for p in pca.pairs)
    assert 'CH3' in d.st['hold_reason']
    assert d.st['now']['select'] is True and d.st['now']['hold'] is True


def test_ai_side_outputs_prediction_and_feeds_back_next_frame(monkeypatch, tmp_path):
    """先読み3でも、次のフレームのモデル入力には「次の時刻に当たる予測」(pred[0])を戻す"""
    d = make_driver(monkeypatch, tmp_path)
    pred = np.array([[1460.0 + 10 * k, 1510.0 + 5 * k] for k in range(10)], np.float32)
    runner, pca = FakeRunner(pred), FakePCA()
    assert run_loop(d, runner, pca, veh(True), lookahead=3) is None
    assert len(runner.calls) >= 3
    assert pca.pairs[-1] == pytest.approx((pred[2, 0], pred[2, 1]))        # 出力は先読み3フレーム目
    st0 = runner.calls[0]['state'][0]
    assert st0[:2] == pytest.approx(((REF[0] - 1500) / 500, (REF[1] - 1500) / 500))   # 最初は中心・中立
    fed = runner.calls[-1]['state'][0][:2] * 500 + 1500
    assert fed[0] in (pytest.approx(pred[0, 0], abs=0.01), pytest.approx(pred[1, 0], abs=0.01))  # 取りこぼし時は1つ先
    assert fed[0] != pytest.approx(pred[2, 0], abs=0.01)


def test_human_side_feeds_rc_values(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    runner, pca = FakeRunner(np.tile([[1600.0, 1550.0]], (10, 1))), FakePCA()
    assert run_loop(d, runner, pca, veh(False, pwm=(1400, 1480, 1000))) is None
    fed = runner.calls[-1]['state'][0][:2] * 500 + 1500
    assert fed == pytest.approx((1400, 1480))


def test_dry_run_writes_only_neutral(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    runner, pca = FakeRunner(np.tile([[1700.0, 1700.0]], (10, 1))), FakePCA()
    assert run_loop(d, runner, pca, veh(True), dry_run=True) is None
    assert runner.calls and pca.pairs and all(p == REF for p in pca.pairs)


def test_transient_i2c_errors_are_retried(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    runner, pca = FakeRunner(np.tile([[1600.0, 1550.0]], (10, 1))), FakePCA(fail=2)
    assert run_loop(d, runner, pca, veh(True)) is None
    assert pca.pairs[-1] == (1600.0, 1550.0)
    assert d.st['stats']['i2c_errors'] >= 1


def test_persistent_i2c_errors_stop_the_drive(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    runner, pca = FakeRunner(np.tile([[1600.0, 1550.0]], (10, 1))), FakePCA(fail=10 ** 6)
    err = run_loop(d, runner, pca, veh(True), frames=60)
    assert isinstance(err, RuntimeError) and 'PCA9685' in str(err)


def test_neutral_safe_writes_throttle_first_and_retries():
    pca = FakePCA(fail=1)
    out = drive._Output(pca, config.PCA9685, REF, lambda: PARAMS)
    assert out.neutral_safe() is True
    assert pca.single[0] == (config.PCA9685['throttle_ch'], REF[1])
    assert pca.single[1] == (config.PCA9685['steering_ch'], REF[0])


def test_predicted_now_uses_frame_gap():
    pred = np.arange(20, dtype=np.float32).reshape(10, 2)
    t0 = 1_000_000_000
    assert drive.predicted_now(pred, t0, t0 + 33_333_333, 30) == (0.0, 1.0)
    assert drive.predicted_now(pred, t0, t0 + 3 * 33_333_333, 30) == (4.0, 5.0)
    assert drive.predicted_now(pred, t0, t0 + 10 ** 10, 30) == (18.0, 19.0)


def test_heartbeat_needs_the_start_token(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    d.heartbeat_t = 0.0
    assert d.heartbeat('') is False and d.heartbeat('other') is False and d.heartbeat_t == 0.0
    assert d.heartbeat('tok') is True and d.heartbeat_t > 0


def test_params_are_partial_and_clamped(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    d.params = dict(PARAMS, throttle_gain=0.3)
    p = d.set_params({'steer_trim_us': 999})
    assert p['throttle_gain'] == 0.3 and p['steer_trim_us'] == 200.0
    with pytest.raises(ValueError):
        d.set_params({'steer_gain': float('nan')})


def test_apply_gains():
    assert drive.apply_gains((1550, 1600), REF, dict(PARAMS, steer_gain=2, throttle_gain=0.5), 1000, 2000) == (1650.0, 1550.0)
    assert drive.apply_gains((1950, 1000), REF, dict(PARAMS, steer_gain=2, throttle_gain=2), 1000, 2000) == (2000.0, 1000.0)


# ---------------------------------------------------------------- 収集
class FakeHub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


def test_late_ai_row_does_not_delete_saved_scene(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    rec = Recorder(FakeHub())
    rec._warm_up = lambda: None
    log = rec.start(description='t')
    t0 = rec.log['start_ns']
    for i in range(20):
        rec._on_sample('CAM_FRONT', Sample(t0 + i * 33_000_000, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
    rec._io.join()
    rec._finalize(0, partial=False)
    scene_dir = tmp_path / log / ('%s-0001' % log)
    assert (scene_dir / 'scene.json').exists()
    rec.add_ai(t0 + 100_000_000, {'late': True})          # 保存済みのシーンの時刻
    assert rec.late_dropped == 1 and 0 not in rec.scenes
    assert rec.stop(only_log='other') is False and rec.recording
    assert rec.stop(only_log=log) is True and not rec.recording
    assert (scene_dir / 'scene.json').exists()


def test_action_reference_ignores_frames_without_erpm():
    ds = JetFormerDataset.__new__(JetFormerDataset)
    moving_legacy = {'vehicle': np.tile([[1500.0, 1580.0, 1000.0, 0.0]], (50, 1)), 'vehicle_valid': np.ones(50, bool),
                     'erpm_valid': np.zeros(50, bool)}
    stopped = {'vehicle': np.tile([[1444.0, 1499.0, 1000.0, 0.0]], (20, 1)), 'vehicle_valid': np.ones(20, bool),
               'erpm_valid': np.ones(20, bool)}
    ds.scenes = [moving_legacy, stopped]
    assert ds.action_reference() == (1444.0, 1499.0)


# ---------------------------------------------------------------- 実データがあれば
def _dataset_or_skip(name):
    root = config.DATASET_DIR / name
    if not (root / 'splits.json').exists():
        pytest.skip('データセット %s がありません' % name)
    return root


def test_replay_matches_training_inputs():
    """収録シーンを走行ループの入力組み立てに流すと、学習データ(aligned)と同じ入力になる"""
    root = _dataset_or_skip('v2_check')
    splits = json.load(open(root / 'splits.json'))
    checked = 0
    for name in splits['train'] + splits['val']:
        a = dict(np.load(root / 'aligned' / (name + '.npz')))
        p = dataset.scene_path(name)
        tof, imu, veh_npz = (dict(np.load(p / f)) for f in ('TOF_FRONT.npz', 'IMU.npz', 'VEHICLE.npz'))
        us = 1000
        tof_buf = [(int(t) * us, x) for t, x in zip(tof['timestamps_us'], tof['depth'])]
        feats = np.concatenate([imu['accel'], np.deg2rad(imu['gyro'])], axis=1)
        imu_buf = [(int(t) * us, f) for t, f in zip(imu['timestamps_us'], feats)]
        veh_buf = [(int(t) * us, {'pwm': veh_npz['pwm'][i].tolist(), 'valid': veh_npz['valid'][i].tolist(),
                                  'select': bool(veh_npz['select'][i]), 'erpm': int(veh_npz['erpm'][i])})
                   for i, t in enumerate(veh_npz['timestamps_us'])]
        for i in np.nonzero(a['vehicle_valid'] & a['future_valid'])[0][::7]:
            t_raw, h, v, reason = drive.assemble_inputs(int(a['timestamps_us'][i]) * us, tof_buf, imu_buf, veh_buf,
                                                        150_000_000, 100_000_000)
            if reason:
                continue
            vs = drive.vehicle_state(v, None, REF, True)
            assert np.array_equal(t_raw, a['tof'][i])
            assert np.allclose(h, a['imu'][i], atol=1e-6)
            assert np.allclose(vs[[0, 1, 3]], a['vehicle'][i][[0, 1, 3]])
            checked += 1
    assert checked > 100


def test_trained_modes_from_train_split():
    _dataset_or_skip('train_test')
    assert drive._trained_modes_from_dataset('train_test', 1) == [4, 5]
    assert drive._trained_modes_from_dataset('no_such_dataset', 1) == []


def test_build_dataset_rejects_legacy_only(monkeypatch, tmp_path):
    legacy = [s['name'] for s in dataset.list_scenes() if s.get('vehicle_quality') == 'legacy']
    if not legacy:
        pytest.skip('旧ファームのシーンがありません')
    monkeypatch.setattr(config, 'DATASET_DIR', tmp_path)
    with pytest.raises(ValueError, match='旧ファーム'):
        dataset.build_dataset('legacy_only', legacy[:1])


# ---------------------------------------------------------------- 学習の残り時間
def test_training_eta_with_few_steps_per_epoch():
    """1エポックのステップ数が10未満でも、通算ステップで正しく見積もる(7+3ステップ/エポック、1ステップ0.5秒)"""
    import collections
    from jetformer.training import TrainingManager
    st = {'info': {'steps_train': 7, 'steps_val': 3}, 'epochs': 10, 'epoch': 1, 'step': 0, 'step_phase': 'train'}
    t_steps = collections.deque(maxlen=50)
    for epoch in range(1, 5):
        for phase, steps in (('train', 7), ('val', 3)):
            for k in (1, steps):                 # train.py は先頭と最後(と10ステップごと)に通知する
                st.update(epoch=epoch, step=k, step_phase=phase)
                g = TrainingManager._global_step(st)
                t_steps.append((g * 0.5, g))
    eta = TrainingManager._eta(st, t_steps)
    assert eta == pytest.approx(6 * 10 * 0.5, abs=1)          # 残り6エポック × 10ステップ × 0.5秒
