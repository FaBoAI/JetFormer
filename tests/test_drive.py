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


def run_loop(d, runner, pca, veh_data, frames=12, lookahead=1, dry_run=False, tof=None):
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
        tof_buf.append((t - 5_000_000, np.full((42, 54), 1000, np.uint16) if tof is None else tof))
        imu_buf.append((t - 2_000_000, np.zeros(6, np.float32)))
        veh_buf.append((t - 3_000_000, dict(veh_data(i) if callable(veh_data) else veh_data)))
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
    runner, pca = FakeRunner(np.tile([[1600.0, 1450.0]], (10, 1))), FakePCA(fail=2)
    assert run_loop(d, runner, pca, veh(True)) is None
    assert pca.pairs[-1] == (1600.0, 1450.0)
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
    assert drive.apply_gains((1550, 1400), REF, dict(PARAMS, steer_gain=2, throttle_gain=0.5), 1000, 2000) == (1650.0, 1450.0)
    # ブレーキ・後退側には速度のゲインを掛けない
    assert drive.apply_gains((1550, 1600), REF, dict(PARAMS, steer_gain=2, throttle_gain=2.0), 1000, 2000) == (1650.0, 1600.0)
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
        try:
            p = dataset.scene_path(name)
        except KeyError:
            continue                                 # 元のシーンが削除済み
        a = dict(np.load(root / 'aligned' / (name + '.npz')))
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
    if not checked:
        pytest.skip('データセットの元のシーンがありません')
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


def test_normalize_inputs_without_command():
    img = np.zeros((224, 224, 3), np.uint8)
    tof = np.full((42, 54), 500, np.uint16)
    imu = np.zeros((10, 6), np.float32)
    veh = np.array([1800, 1300, 1000, 5000], np.float32)
    _, _, _, with_cmd = normalize_inputs(img, tof, imu, veh)
    _, _, _, no_cmd = normalize_inputs(img, tof, imu, veh, use_command=False)
    assert with_cmd[0] == pytest.approx(0.6) and with_cmd[1] == pytest.approx(-0.4)
    assert no_cmd[0] == 0 and no_cmd[1] == 0 and no_cmd[2] == with_cmd[2]          # 回転数は残す


def test_model_info_reports_command_input(monkeypatch, tmp_path):
    runs = tmp_path / 'runs'
    for name, cfg in (('old', {}), ('nocmd', {'command_input': False})):
        (runs / name).mkdir(parents=True)
        (runs / name / drive.ENGINE_FILE).write_bytes(b'x')
        (runs / name / 'model_config.json').write_text(json.dumps(dict(cfg, num_modes=6, trained_modes=[5])))
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    assert drive.model_info('old')['command_input'] is True                       # 古い学習結果は今まで通り
    assert drive.model_info('nocmd')['command_input'] is False


def test_vehicle_sensor_rides_through_i2c_noise():
    """I2Cの通信エラーが時々起きても止まらず(その周期を飛ばすだけ)、続いたときだけ開き直す"""
    from jetformer.sensors import vehicle as vs

    def frame(seq):
        d = bytearray(vs.SNAP.pack(vs.SNAP_MAGIC, vs.SNAP_VERSION, seq & 0xFFFF, 0, 1500, 1500, 2000, 7, 0, 0, 0, 0, 0, 0))
        d[-1] = vs.crc8(bytes(d[:-1]))
        return list(d)

    class Bus(object):
        def __init__(self, pattern):
            self.pattern, self.i = pattern, 0

        def read_i2c_block_data(self, addr, reg, n):
            ok = self.pattern[self.i % len(self.pattern)]
            self.i += 1
            if not ok:
                raise OSError(121, 'Remote I/O error')
            return frame(self.i)                                  # seq を変える
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot'})
    s.bus = Bus([True, False, False, False, True, True])          # 3回続けて失敗する周期がある
    got = sum(len(s._poll_snapshot()) for _ in range(20))
    assert got > 10 and s.bus_errors > 0 and s.skipped_polls > 0
    s.bus = Bus([False])                                          # ずっと読めない → 開き直す
    with pytest.raises(RuntimeError):
        for _ in range(vs.VehicleSensor.MAX_FAIL_STREAK + 1):
            s._poll_snapshot()


def test_imu_sensor_rides_through_i2c_noise():
    from jetformer.sensors import imu as im

    class Bus(object):
        def __init__(self, fail_every):
            self.n, self.fail_every = 0, fail_every

        def write_byte_data(self, *a):
            pass

        def i2c_rdwr(self, w, r):
            self.n += 1
            if self.fail_every and self.n % self.fail_every == 0:
                raise OSError(121, 'Remote I/O error')
    s = im.ImuSensor(dict(config.IMU, rate_hz=1000))
    s.a_scale = s.g_scale = 1.0                                   # 本来は開いたときに設定される
    s.bus = Bus(3)
    got = sum(len(s._poll()) for _ in range(30))
    assert got == 20 and s.bus_errors == 10
    s.bus = Bus(1)
    with pytest.raises(RuntimeError):
        for _ in range(im.ImuSensor.MAX_FAIL_STREAK + 1):
            s._poll()


def test_loop_recovers_when_stuck_against_wall(monkeypatch, tmp_path):
    """走っていて止まった(壁に当たった)→ 自動でブレーキ・中立・後退を出し、AI_CMD に補助の種類を残す"""
    d = make_driver(monkeypatch, tmp_path)
    pred = np.tile([[1443.0, 1400.0]], (10, 1))                     # モデルは前進を出し続ける
    runner, pca = FakeRunner(pred), FakePCA()
    d.recorder.recording = True
    # 0.5秒走ってから止まる(eRPM 0)
    assert run_loop(d, runner, pca, lambda i: veh(True, erpm=-5000 if i < 15 else 0), frames=75) is None
    kinds = [row['assist'] for _, row in d.recorder.rows]
    assert 2 in kinds                                               # リカバリーした
    thr = [t for _, t in pca.pairs]
    assert max(thr) > REF[1] + 100                                  # 後退を出した
    assert d.st['stats']['recoveries'] >= 1
    rows = [row for _, row in d.recorder.rows if row['assist'] == 2]
    assert all(r['model_out'][1] == pytest.approx(1400.0, abs=60) for r in rows)   # 補助の前のモデルの値も残す


def test_loop_without_recovery_keeps_model_output(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    pred = np.tile([[1443.0, 1400.0]], (10, 1))
    runner, pca = FakeRunner(pred), FakePCA()
    d.recorder.recording = True
    monkeypatch.setitem(PARAMS, 'recovery', 0)                     # run_loop は PARAMS から設定を作る
    assert run_loop(d, runner, pca, lambda i: veh(True, erpm=-5000 if i < 15 else 0), frames=60) is None
    assert all(row['assist'] == 0 for _, row in d.recorder.rows)
    assert all(t == pytest.approx(1400.0, abs=1) for _, t in pca.pairs[2:])


def test_vehicle_sensor_counts_board_resets():
    """基板の時計が巻き戻ったら(再起動)数える。32bitの一周は数えない"""
    from jetformer.sensors import vehicle as vs

    def frame(seq, bt):
        d = bytearray(vs.SNAP.pack(vs.SNAP_MAGIC, vs.SNAP_VERSION, seq & 0xFFFF, bt, 1500, 1500, 2000, 7, 0, 0, 0, 0, 0, 0))
        d[-1] = vs.crc8(bytes(d[:-1]))
        return list(d)

    class Bus(object):
        def __init__(self, frames):
            self.frames = iter(frames)

        def read_i2c_block_data(self, addr, reg, n):
            return next(self.frames)
    s = vs.VehicleSensor({'bus': 7, 'address': 0x28, 'rate_hz': 100, 'protocol': 'snapshot'})
    s.bus = Bus([frame(10, 5_000_000), frame(11, 5_010_000), frame(2, 70_000),        # 再起動
                 frame(3, 4_294_000_000), frame(4, 20_000)])                         # 一周(数えない)
    for _ in range(5):
        s._poll_snapshot()
    assert s.board_resets == 1 and s.status()['board_resets'] == 1


def _wall_tof(dist_mm):
    """床(1.5m)の手前に、正面の壁(dist_mm)。信頼度4"""
    d = np.full((42, 54), 1500, np.uint16)
    d[6:26, 20:34] = dist_mm
    return (d | (4 << 13)).astype(np.uint16)


def _guard_driver(monkeypatch, tmp_path, mode):
    ref = tmp_path / 'floor.npy'
    np.save(ref, np.full((42, 54), 1500, np.uint16))
    monkeypatch.setattr(drive, 'FLOOR_REF_PATH', ref)
    monkeypatch.setitem(PARAMS, 'tof_guard', mode)
    monkeypatch.setitem(PARAMS, 'recovery', 0)
    d = make_driver(monkeypatch, tmp_path)
    d.recorder.recording = True
    return d


def test_tof_guard_log_only_does_not_change_output(monkeypatch, tmp_path):
    d = _guard_driver(monkeypatch, tmp_path, 1)
    runner, pca = FakeRunner(np.tile([[1443.0, 1350.0]], (10, 1))), FakePCA()
    assert run_loop(d, runner, pca, veh(True, erpm=-5000), tof=_wall_tof(500)) is None
    rows = [r for _, r in d.recorder.rows]
    assert any(r['guard'] == 2 for r in rows) and not any(r['guard_applied'] for r in rows)
    assert all(t == pytest.approx(1350.0) for _, t in pca.pairs[2:])            # 出力はモデルのまま
    assert d.st['stats']['guard_brake'] >= 1


def test_tof_guard_slows_down_when_enabled(monkeypatch, tmp_path):
    d = _guard_driver(monkeypatch, tmp_path, 2)
    runner, pca = FakeRunner(np.tile([[1443.0, 1350.0]], (10, 1))), FakePCA()
    assert run_loop(d, runner, pca, veh(True, erpm=-5000), tof=_wall_tof(500)) is None
    assert max(t for _, t in pca.pairs[2:]) > REF[1]                                 # ブレーキ側
    assert any(r['guard_applied'] for _, r in d.recorder.rows)
    # 何もなければモデルのまま
    d2 = _guard_driver(monkeypatch, tmp_path, 2)
    runner2, pca2 = FakeRunner(np.tile([[1443.0, 1350.0]], (10, 1))), FakePCA()
    assert run_loop(d2, runner2, pca2, veh(True, erpm=-5000), tof=_wall_tof(1500)) is None
    assert all(t == pytest.approx(1350.0) for _, t in pca2.pairs[2:])


def test_impact_mask_excludes_frames_around_crashes():
    from jetformer.train_data import impact_mask
    n = 300                                                   # 30fps で10秒
    t_us = (np.arange(n) * 1e6 / 30).astype(np.int64)
    imu = np.zeros((n, 10, 6), np.float32)
    imu[:, :, 2] = 9.8                                        # 重力(上下)は数えない
    imu[150, 4, 1] = -6 * 9.80665                             # 5秒目に前から6G
    m = impact_mask(t_us, imu)
    assert m[150] and m[120] and m[210]                      # 1秒前〜2秒後
    assert not m[115] and not m[215] and m.sum() == 91
    imu2 = np.zeros((n, 10, 6), np.float32)
    imu2[:, :, 2] = 3 * 9.8                                   # 上下の揺れだけなら外さない
    assert not impact_mask(t_us, imu2).any()
