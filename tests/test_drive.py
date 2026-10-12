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


def test_takeover_reason_links_to_last_takeover(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    events = []
    d.recorder.add_event = lambda kind, data: events.append((kind, data)) or True
    d.last_takeover_ns = clock.now_ns() - 2_000_000_000                # 2秒前に人が取り返した
    ev = d.takeover_reason('collision')
    assert ev['label'] == '衝突した' and 1.5 < ev['since_takeover_s'] < 3.0 and ev['recorded']
    assert events and events[0][0] == 'takeover_reason'
    d.last_takeover_ns = clock.now_ns() - 120_000_000_000              # 2分前は結びつけない
    assert d.takeover_reason('other')['takeover_us'] is None
    with pytest.raises(ValueError):
        d.takeover_reason('unknown')


# ---------------------------------------------------------------- モデルの種類(方策)・後処理・重ね合わせ
class ExtraRunner(FakeRunner):
    """追加の出力(tcp の waypoints・speed_logits の形)も返す偽のエンジン"""

    def run_numpy(self, **kw):
        out = super().run_numpy(**kw)
        out['waypoints'] = np.zeros((1, 10, 2), np.float32)
        out['speed_logits'] = np.zeros((1, 10), np.float32)
        return out


class FakeTCPPost(object):
    """スロットルを中立 − 100·v_now µs にし、ステアは actions のまま(偽の後処理)"""
    kind = 'fake_tcp'
    instances = []

    def __init__(self, pp, ref):
        self.ref, self.resets, self.seen = ref, 0, []
        FakeTCPPost.instances.append(self)

    def reset(self):
        self.resets += 1

    def step(self, outputs, v_now, params, t_ns):
        self.seen.append(sorted(outputs))
        pred = outputs['actions'] * 500 + 1500
        pred[:, 1] = self.ref[1] - 100.0
        return pred.astype(np.float32), {'mode': int(params.get('tcp_mode', 0)), 'branch': 'fake', 'v_hat': 2.5,
                                         'v_cmd': 2.0, 'v_curv': float('nan'), 'cap': 0.0}


@pytest.fixture
def fake_post(monkeypatch):
    import sys
    import types
    from jetformer.policies import post
    mod = types.ModuleType('jetformer_test_drive_post')
    mod.FakeTCPPost = FakeTCPPost
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(post.KINDS, 'fake_tcp', (mod.__name__, 'FakeTCPPost'))
    FakeTCPPost.instances = []
    return FakeTCPPost


def test_model_info_policy_fields_and_old_configs(monkeypatch, tmp_path):
    runs = tmp_path / 'runs'
    cfgs = {'old': {}, 'act': {'policy': 'act', 'policy_cfg': {'dec_layers': 2}, 'outputs': ['actions']},
            'unknown': {'policy': 'someday', 'inputs': ['image', 'mode']},
            'history': {'policy': 'act', 'inputs': ['image', 'history']}}
    for name, cfg in cfgs.items():
        (runs / name).mkdir(parents=True)
        (runs / name / drive.ENGINE_FILE).write_bytes(b'x')
        (runs / name / 'model_config.json').write_text(json.dumps(dict(cfg, num_modes=6, trained_modes=[4, 5])))
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    old = drive.model_info('old')
    assert old['policy'] == 'jetformer_v0' and old['drivable'] is True and old['why_not'] == ''
    assert old['policy_cfg'] == {'dim': 256, 'layers': 4, 'heads': 8} and old['inputs'] == list(drive.policies.BASE_INPUTS)
    assert old['outputs'] == ['actions'] and old['postprocess'] is None
    assert drive.model_info('act')['policy_title'].startswith('ACT')
    assert drive.model_info('unknown')['policy_title'] == 'someday' and drive.model_info('unknown')['drivable'] is True
    bad = drive.model_info('history')
    assert bad['drivable'] is False and 'history' in bad['why_not']
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    with pytest.raises(ValueError, match='走行できません'):
        d.start('history', 4)
    assert {m['name'] for m in drive.list_models()} == set(cfgs)          # 走れないものも一覧には出す(画面で選べなくする)


def test_loop_feeds_all_base_inputs_and_default_is_unchanged(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    pred = np.array([[1460.0 + 10 * k, 1510.0 + 5 * k] for k in range(10)], np.float32)
    runner, pca = FakeRunner(pred), FakePCA()
    d.recorder.recording = True
    monkeypatch.setitem(PARAMS, 'ensemble_j', 0)
    assert run_loop(d, runner, pca, veh(True), lookahead=3) is None
    assert set(runner.calls[0]) == {'image', 'tof', 'imu', 'state', 'mode'}
    assert runner.calls[0]['mode'].shape == (1, 6)
    assert pca.pairs[-1] == pytest.approx((pred[2, 0], pred[2, 1]))          # 後処理なし・重ね合わせなし = 今まで通り
    rows = [r for _, r in d.recorder.rows if not r['hold']]
    assert rows and all(np.array_equal(r['model_pred'], r['pred']) for r in rows)
    assert all(r['post_mode'] == -1 and np.isnan(r['v_hat']) and r['ens_j'] == 0 for r in rows)
    assert d.st['now']['post'] is None


def test_loop_runs_post_before_lookahead_and_records(monkeypatch, tmp_path, fake_post):
    d = make_driver(monkeypatch, tmp_path)
    d.st['model'] = {'postprocess': {'kind': 'fake_tcp', 'version': 1}}
    pred = np.tile([[1600.0, 1450.0]], (10, 1))
    runner, pca = ExtraRunner(pred), FakePCA()
    d.recorder.recording = True
    monkeypatch.setitem(PARAMS, 'tcp_mode', 1)
    assert run_loop(d, runner, pca, veh(True, erpm=-2000)) is None
    obj = fake_post.instances[-1]
    assert obj.seen and obj.seen[-1] == ['actions', 'speed_logits', 'waypoints']    # 追加の出力も後処理に渡る
    assert pca.pairs[-1] == pytest.approx((1600.0, REF[1] - 100.0))                 # 先読み・ゲインは後処理の後
    rows = [r for _, r in d.recorder.rows if not r['hold']]
    assert rows and all(r['post_mode'] == 1 and r['v_hat'] == 2.5 and r['v_cmd'] == 2.0 for r in rows)
    assert np.allclose(rows[-1]['model_pred'][:, 1], 1450.0) and np.allclose(rows[-1]['pred'][:, 1], REF[1] - 100.0)
    post_now = d.st['now']['post']
    assert post_now['branch'] == 'fake' and post_now['v_curv'] is None and post_now['v_now'] == pytest.approx(1.0)
    json.dumps(d.st['now'], allow_nan=False)                                       # 画面へ送れる(NaN なし)


def test_hold_resets_post(monkeypatch, tmp_path, fake_post):
    d = make_driver(monkeypatch, tmp_path)
    d.st['model'] = {'postprocess': {'kind': 'fake_tcp'}}
    runner, pca = ExtraRunner(np.tile([[1600.0, 1450.0]], (10, 1))), FakePCA()
    assert run_loop(d, runner, pca, veh(True, valid=(False, False, False))) is None     # CH3 が届かない → 中立
    assert fake_post.instances[-1].resets >= 1


def test_post_unavailable_falls_back_to_actions(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)
    d.st['model'] = {'postprocess': {'kind': 'no_such_kind'}}
    pred = np.tile([[1600.0, 1450.0]], (10, 1))
    runner, pca = ExtraRunner(pred), FakePCA()
    assert run_loop(d, runner, pca, veh(True)) is None
    assert pca.pairs[-1] == pytest.approx((1600.0, 1450.0))
    assert '操作の枝' in d.st['recording_note']
    assert d.st['post_failed'] is True


def test_new_params_are_rounded_and_old_settings_load(monkeypatch, tmp_path):
    (tmp_path / 'drive_settings.json').write_text(json.dumps({'params': {'steer_gain': 1.2}, 'run': 'x', 'mode_id': 5}))
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    assert d.params['steer_gain'] == 1.2 and d.params['tcp_mode'] == 0 and d.params['speed_cap'] == 0.0
    assert d.params['ensemble_j'] == 0 and d.params['speed_scale'] == 1.0
    p = d.set_params({'tcp_mode': 1.6, 'ensemble_j': 2.4, 'speed_cap': 9.0, 'speed_scale': 0.1, 'ensemble_m': 0.3})
    assert p['tcp_mode'] == 2 and isinstance(p['tcp_mode'], int) and p['ensemble_j'] == 2
    assert p['speed_cap'] == 5.0 and p['speed_scale'] == 0.5 and p['ensemble_m'] == pytest.approx(0.3)
    for k in ('speed_cap', 'speed_scale', 'tcp_mode', 'ensemble_j', 'ensemble_m'):
        assert k in drive.PARAM_LIMITS and k in drive.DEFAULT_PARAMS


def test_ai_cmd_npz_has_policy_fields_with_defaults(monkeypatch, tmp_path):
    """古い呼び方(新しい欄なし)の行と新しい行が混ざっても AI_CMD.npz を書け、新しい欄は既定値で埋まる"""
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    rec = Recorder(FakeHub())
    rec._warm_up = lambda: None
    log = rec.start(description='t')
    t0 = rec.log['start_ns']
    for i in range(20):
        rec._on_sample('CAM_FRONT', Sample(t0 + i * 33_000_000, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
    old_row = {'pred': np.zeros((10, 2), np.float32), 'raw': (1, 2), 'out': (1, 2), 'gains': (1, 1), 'trim': 0,
               'lookahead': 1, 'mode_id': 5, 'hold': False, 'select': True, 'latency_ms': 1.0, 'infer_ms': 1.0,
               'dry_run': False}
    new_row = dict(old_row, model_pred=np.ones((10, 2), np.float32), post_mode=1, v_hat=2.0, v_cmd=1.5, ens_j=2)
    rec.add_ai(t0 + 66_000_000, old_row)
    rec.add_ai(t0 + 99_000_000, new_row)
    rec._io.join()
    rec._finalize(0, partial=False)
    rec.stop(only_log=log)
    z = np.load(tmp_path / log / ('%s-0001' % log) / 'AI_CMD.npz')
    assert z['model_pred'].shape == (2, 10, 2) and z['model_pred'][0].sum() == 0 and z['model_pred'][1].sum() == 20
    assert z['post_mode'].tolist() == [-1, 1] and np.isnan(z['v_hat'][0]) and z['v_cmd'][1] == 1.5
    assert z['ens_j'].tolist() == [0, 2]


# ---------------------------------------------------------------- レビュー(2026-10-12)で直した点
def test_boot_does_not_carry_over_tcp_mode(monkeypatch, tmp_path):
    """tcp の出し方・目標速度の倍率は次に起動したときに持ち越さない(速度の上限は残す)"""
    (tmp_path / 'drive_settings.json').write_text(json.dumps(
        {'params': {'tcp_mode': 3, 'speed_scale': 1.4, 'speed_cap': 1.5, 'steer_gain': 1.2}, 'run': 'x', 'mode_id': 5}))
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    assert d.params['tcp_mode'] == 0 and d.params['speed_scale'] == 1.0
    assert d.params['speed_cap'] == 1.5 and d.params['steer_gain'] == 1.2


def _run_dir(runs, name, cfg):
    (runs / name).mkdir(parents=True)
    (runs / name / drive.ENGINE_FILE).write_bytes(b'engine')
    (runs / name / 'model_config.json').write_text(json.dumps(dict(cfg, num_modes=6, trained_modes=[4, 5],
                                                                   steer_center_us=1441.0, throttle_neutral_us=1496.0)))


def test_start_refuses_cap_when_postprocess_cannot_load(monkeypatch, tmp_path):
    """model_config に後処理があるのに読めないとき、速度の上限・tcp の出し方を使う走行は始めない(黙って actions だけで
    走らない)。上限 0・出し方 0 なら今までどおり操作の枝で始める"""
    runs = tmp_path / 'runs'
    _run_dir(runs, 'broken_tcp', {'policy': 'tcp', 'postprocess': {'kind': 'no_such_kind', 'version': 1}})
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    started = []
    d._run = lambda *a: started.append(a)                               # 走行のスレッドの中身は回さない
    with pytest.raises(ValueError, match='速度の上限'):
        d.start('broken_tcp', 4, params={'speed_cap': 1.0})
    with pytest.raises(ValueError, match='速度の上限'):
        d.start('broken_tcp', 4, params={'tcp_mode': 1, 'speed_cap': 0.0})
    assert not started and d.thread is None
    d.start('broken_tcp', 4, params={'speed_cap': 0.0, 'tcp_mode': 0})
    d.thread.join(2)
    assert started


def test_start_accepts_cap_when_postprocess_loads(monkeypatch, tmp_path, fake_post):
    runs = tmp_path / 'runs'
    _run_dir(runs, 'ok_tcp', {'policy': 'tcp', 'postprocess': {'kind': 'fake_tcp', 'version': 1}})
    _run_dir(runs, 'v0', {})
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    monkeypatch.setattr(drive, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    d._run = lambda *a: None
    d.start('ok_tcp', 4, params={'speed_cap': 1.0, 'tcp_mode': 1})
    d.thread.join(2)
    d.start('v0', 4, params={'speed_cap': 1.0})                         # 後処理の無いモデル: 効かないだけ(画面に出す)
    d.thread.join(2)


def test_set_params_refuses_cap_while_post_failed(monkeypatch, tmp_path):
    d = make_driver(monkeypatch, tmp_path)

    class Alive(object):
        def is_alive(self):
            return True
    d.thread = Alive()
    d.st['post_failed'] = True
    with pytest.raises(ValueError, match='後処理'):
        d.set_params({'speed_cap': 1.5})
    with pytest.raises(ValueError, match='後処理'):
        d.set_params({'tcp_mode': 1})
    assert d.set_params({'steer_gain': 1.1})['steer_gain'] == 1.1       # ほかの項目は変えられる
    d.st['post_failed'] = False
    assert d.set_params({'speed_cap': 1.5})['speed_cap'] == 1.5


def test_model_info_engine_check(monkeypatch, tmp_path):
    """<学習結果>/engine_check.json を走行タブに出す(今のエンジンの確認のときだけ)。精度の確認に落ちたエンジンは走らせない
    (2026-10-12 のレビュー。前は警告だけ。詳しくは tests/test_drive_model_select.py)"""
    runs = tmp_path / 'runs'
    _run_dir(runs, 'dino', {'policy': 'dinov2', 'engine': {'precision': 'fp16'}})
    monkeypatch.setattr(drive, 'RUNS_DIR', runs)
    assert drive.model_info('dino')['engine_check'] is None and drive.model_info('dino')['engine_precision'] == 'fp16'
    stt = (runs / 'dino' / drive.ENGINE_FILE).stat()
    chk = {'ok': False, 'accuracy_ok': False, 'speed_ok': True, 'actions_max_us': 2.854, 'gate_us': 2.5,
           'run_numpy_median_ms': 4.98, 'checked_at': '2026-10-12 18:27:24', 'note': 'n',
           'engine': {'size': stt.st_size, 'mtime_ns': stt.st_mtime_ns}}
    (runs / 'dino' / drive.ENGINE_CHECK_FILE).write_text(json.dumps(chk))
    info = drive.model_info('dino')
    assert info['engine_check']['ok'] is False and info['engine_check']['actions_max_us'] == 2.854
    assert info['drivable'] is False and '精度の確認に落ちています' in info['why_not'] and '2.85 µs > 2.5 µs' in info['why_not']
    (runs / 'dino' / drive.ENGINE_FILE).write_bytes(b'rebuilt engine')   # エンジンを作り直した → 古い確認は出さない・走れる
    assert drive.model_info('dino')['engine_check'] is None and drive.model_info('dino')['drivable'] is True


def test_closed_loop_cap_with_gain_through_apply_gains():
    """本物の制御器 + drive.apply_gains: 強い前進の操作の枝・上限 1.0 m/s で、ゲイン 0.5・2.0 でも
    ゲインの後のスロットルが thr_ff(1.0) になる(実際の較正の値。レビューの見積もり: 前はゲイン 0.5 で約 0.32 m/s)"""
    from jetformer.policies import controller as C
    offs = np.arange(-575.0, 576.0, 50.0)
    k = np.maximum.accumulate(np.clip(0.004 * offs, -1.5, 1.5)) + np.arange(len(offs)) * 1e-6
    pp = {'kind': 'tcp_controller', 'version': 1, 'speed_bins': [0.5 * i for i in range(10)],
          'steer_table': {'offset_us': offs.tolist(), 'bands_mps': [2.0, 3.0], 'curv': [k.tolist()] * 3},
          'throttle_ff': {'c0': 1490.73, 'c1': -72.073, 'c2': -13.648, 'tau_s': 0.33, 'fwd_max_us': 361.0}}
    ref = (1441.0, 1496.0)
    strong = {'actions': np.full((10, 2), -0.6, np.float32), 'waypoints': np.zeros((10, 2), np.float32),
              'speed_logits': np.zeros(10, np.float32)}
    for gain in (0.5, 1.0, 2.0):
        c = C.TCPPost(pp, ref)
        p = dict(PARAMS, throttle_gain=gain, tcp_mode=0, speed_cap=1.0)
        v = 0.0
        for i in range(450):
            out, _ = c.step(strong, v, p, i * int(1e9 / 30))
            thr = drive.apply_gains((float(out[0, 0]), float(out[0, 1])), ref, p, 1000, 2000)[1]
            fwd = ref[1] - thr
            v_ss = (fwd - (ref[1] - 1490.73)) / 72.073 if fwd > 0 else max(v - 0.1, 0.0)
            v += (max(v_ss, 0.0) - v) * (1.0 / 30) / 0.3
        assert abs(v - 1.0) < 0.1, (gain, v)
