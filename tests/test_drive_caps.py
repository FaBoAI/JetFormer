"""区間の速度上限 v1.1(safety_caps)を走行ループ(drive.py の zone_caps)につないだ部分の試験(2026-10-21)。
実機・GPU・torch は使わない(センサー・PCA9685・TensorRT・入力の正規化は偽物。時刻も決まった値で、スレッドを使わずに回す)

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python -m pytest -q tests/test_drive_caps.py

確かめること:
  - 既定は OFF・起動のたびに OFF・既存の既定値は変わらない
  - OFF のとき、変更前の drive.py(git の 30b71e8)と、同じ入力の列で出力がビット単位で同じ(上限の計算が例外になっても同じ)
  - ON のとき、出力は純粋な包み(safety_caps.zone_imu・zone_frame)を同じ入力で回した値と同じ。前進を強めない・舵は変えない
  - ON/OFF に関係なく、影の判定(cap_*)を記録に残す(AI_CMD.npz まで)。例外の走行は上限を外し、記録とメッセージに残す
  - 10/10 の実車の記録(あるときだけ): 純粋な包み・走行ループのつなぎ(drive._ZoneCaps)が G1 の再生
    (experiments/20261016_sim_calibration/g1_replay.py の prepare・shadow、v1.1)と全フレームで同じ
"""
import collections
import importlib.util
import subprocess
import time as _time
import types
from pathlib import Path

import numpy as np
import pytest

from jetformer import clock, config, drive
from jetformer import safety_caps as sc
from jetformer.sensors.base import Sample

ROOT = Path(__file__).resolve().parents[1]
BEFORE = '30b71e8'                 # 区間の速度上限をつなぐ前の drive.py
REF = (1441.0, 1496.0)             # モデル B のステアの中心・スロットルの中立
PARAMS = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'steer_trim_us': 0.0, 'lookahead': 1, 'osc_percent': 0.0,
          'esc_reverse': 1, 'recovery': 1, 'rev_us': 300.0, 'rev_s': 0.6, 'tof_guard': 1, 'speed_cap': 0.0,
          'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5}
NS = 1_000_000_000
T0 = 1_800_000_000 * NS            # 合成の時刻の始まり [ns]

# 合成の周(tests/test_safety_caps.py の synth と同じ形。左回り、R1 は右 −300 deg/s × 0.6 秒)。(秒, ヨー deg/s, 速さ m/s)
LAP = [(0.6, -300.0, 2.5), (1.0, 160.0, 2.2), (2.0, 30.0, 3.0), (1.5, 90.0, 3.2), (1.0, 40.0, 3.5), (1.0, 145.0, 2.8),
       (0.3, 0.0, 3.0)]
START = [(1.0, 0.0, 0.0), (1.0, 0.0, 2.0), (1.0, 180.0, 2.5), (0.3, 0.0, 3.0)]


def imu_stream(n_laps=2, hz=100.0):
    """[(t_ns, gz deg/s, v m/s)] 100 Hz"""
    out, k = [], 0
    for dur, g, v in START + LAP * n_laps:
        for _ in range(int(round(dur * hz))):
            out.append((T0 + int(round(k * NS / hz)), g + 0.4, v))
            k += 1
    return out


# ---------------------------------------------------------------- 偽物
class SeqRunner(object):
    """呼ばれた回ごとに決まった予測を返す TensorRT の偽物"""

    def __init__(self, preds):
        self.preds = [((np.asarray(p, np.float32) - drive.PWM_CENTER) / drive.PWM_SCALE)[None] for p in preds]
        self.n = 0

    def run_numpy(self, **kw):
        a = self.preds[self.n % len(self.preds)]
        self.n += 1
        return {'actions': a.copy()}


class FakePCA(object):
    def __init__(self):
        self.pairs, self.single = [], []

    def set_pair_us(self, sc_, s, tc, t):
        self.pairs.append((s, t))

    def set_us(self, ch, us):
        self.single.append((ch, us))

    def set_osc_scale(self, scale):
        pass


class FakeRecorder(object):
    recording = False

    def __init__(self):
        self.rows, self.events = [], []

    def add_ai(self, t_ns, data):
        self.rows.append((t_ns, data))

    def add_event(self, kind, data):
        self.events.append((kind, data))
        return True


def fake_normalize(image, tof_raw, imu, vehicle, use_command):
    """train_data.normalize_inputs の偽物(torch を読まない)。形だけ合わせる"""
    return (np.zeros((3, 4, 4), np.float32), np.zeros((1, 2, 2), np.float32), np.asarray(imu, np.float32),
            (np.asarray(vehicle, np.float32) - 1500.0) / 500.0)


def scenario(n_frames=None):
    """カメラ 30 fps の各フレームの値: 予測 [10,2]・車両の値・センサーの途切れ。戻り値 (frames, imu)"""
    imu = imu_stream(2)
    t_end = imu[-1][0]
    frames = []
    i = 0
    while True:
        cam_t = T0 + 150_000_000 + int(round(i * NS / 30))
        if cam_t > t_end - 50_000_000 or (n_frames is not None and i >= n_frames):
            break
        v = next(x[2] for x in reversed(imu) if x[0] <= cam_t)
        pred = np.tile([[1450.0, 1250.0]], (10, 1))                 # 前へ u = 3.6 m/s(上限 2.2〜3.0 より強い)
        if 60 <= i < 100 or 330 <= i < 350:
            pred[6:, 0] = 1120.0                                    # 先の予測が右いっぱい = 合図 (b)。今の舵は 1450
        if 130 <= i < 140:
            pred[:, 1] = 1650.0                                     # 後退側(補助が中立を挟む)
        if 140 <= i < 150:
            pred[:, 1] = 1380.0                                     # 弱い前進(上限より下)
        sel = not (200 <= i < 215)                                  # 人が操縦
        valid = (True, True, not (230 <= i < 233))                  # CH3 が届かない = 中立で止める
        veh = {'pwm': [1441, 1496, 2000 if sel else 1000], 'valid': list(valid), 'select': sel, 'moving': v > 0,
               'erpm': int(round(-v / sc.KV))}
        frames.append({'cam_t': cam_t, 'pred': pred, 'veh': veh, 'veh_skip': 260 <= i < 264})   # 車両の値が古い(0.1 秒超)
        i += 1
    return frames, imu


class Feeder(object):
    """cam_evt の代わり。wait() のたびに次のフレームのセンサー値をバッファへ入れ、時計を進める(スレッドなし)"""

    def __init__(self, d, frames, imu, bufs, cam_slot, box):
        self.d, self.frames, self.imu = d, frames, imu
        self.tof_buf, self.imu_buf, self.veh_buf = bufs
        self.cam_slot, self.box = cam_slot, box
        self.i, self.k = 0, 0

    def wait(self, timeout):
        if self.i >= len(self.frames):
            self.d._stop.set()
            return False
        f = self.frames[self.i]
        cam_t = f['cam_t']
        while self.k < len(self.imu) and self.imu[self.k][0] <= cam_t + 2_000_000:      # 撮像の少しあとのサンプルも届いている
            t, g, v = self.imu[self.k]
            self.imu_buf.append((t, np.array([0.0, 0.0, 9.8, 0.0, 0.0, np.deg2rad(g)], np.float32)))
            self.k += 1
        if not f['veh_skip']:
            self.veh_buf.append((cam_t - 3_000_000, dict(f['veh'])))
        self.tof_buf.append((cam_t - 5_000_000, np.full((8, 8), 1000, np.uint16)))
        self.cam_slot[0] = Sample(cam_t, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': self.i})
        self.box['ns'] = cam_t + 8_000_000
        self.box['m'] = (cam_t - T0) / 1e9 + 100.0
        self.d.heartbeat_t = self.box['m']
        self.i += 1
        return True

    def clear(self):
        pass

    def set(self):
        pass


def load_before():
    """区間の速度上限をつなぐ前の drive.py(git の BEFORE)をモジュールにする"""
    r = subprocess.run(['git', '-C', str(ROOT), 'show', BEFORE + ':jetformer/drive.py'], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip('git で変更前の drive.py を読めません: %s' % r.stderr.strip())
    spec = importlib.util.spec_from_loader('jetformer._drive_before_zone_caps', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    exec(compile(r.stdout, 'drive_before_%s.py' % BEFORE, 'exec'), mod.__dict__)
    assert not hasattr(mod, 'safety_caps')
    return mod


def run_det(mod, monkeypatch, tmp_path, params, frames, imu, dry_run=False):
    """mod(drive か変更前の drive)の _loop を決まった時刻・決まった入力の列で回す → (driver, pca, recorder)"""
    box = {'ns': T0, 'm': 100.0}
    monkeypatch.setattr(clock, 'now_ns', lambda: box['ns'])
    monkeypatch.setattr(mod, 'time', types.SimpleNamespace(monotonic=lambda: box['m'], perf_counter=_time.perf_counter,
                                                           time=lambda: 1.0, sleep=lambda s: None, strftime=_time.strftime))
    monkeypatch.setattr(mod, 'SETTINGS_PATH', tmp_path / 'drive_settings.json')
    monkeypatch.setattr(mod, 'FLOOR_REF_PATH', tmp_path / 'no_floor_ref.npy')
    rec = FakeRecorder()
    d = mod.Driver(hub=None, recorder=rec)
    d.params = dict(params)
    d.hb_token = 'tok'
    d.hb_count = 1
    d.st = {'phase': 'driving', 'mode_id': 4}
    pca = FakePCA()
    out = mod._Output(pca, config.PCA9685, REF, lambda: d.params)
    bufs = (collections.deque(maxlen=10), collections.deque(maxlen=60), collections.deque(maxlen=50))
    cam_slot = [None]
    feeder = Feeder(d, frames, imu, bufs, cam_slot, box)
    runner = SeqRunner([f['pred'] for f in frames])
    d._loop(d.st, runner, fake_normalize, pca, out, REF, 6, 0.0, cam_slot, feeder, *bufs, dry_run)
    d._stop.clear()
    return d, pca, rec


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def assert_rows_equal_before(rows_before, rows_new):
    """変更前の記録の欄(infer_ms は時間の計測なので除く)が、新しい記録でビット単位で同じ"""
    assert len(rows_before) == len(rows_new)
    for (t0, r0), (t1, r1) in zip(rows_before, rows_new):
        assert t0 == t1
        for k, v in r0.items():
            if k == 'infer_ms':
                continue
            assert same(v, r1[k]), (t0, k, v, r1[k])


# ---------------------------------------------------------------- 既定値
def test_default_off_and_reset_on_boot(monkeypatch, tmp_path):
    before = load_before()
    assert drive.DEFAULT_PARAMS['zone_caps'] == 0
    added = (('zone_caps', 'start_kick') + drive.SPEED_CTRL_SWITCHES + drive.SPEED_CTRL_VALUES
             + drive.LAUNCH_KEYS + ('rec_back_guard', 'rec_hold_keep', 'launch_no_kick'))   # 30b71e8 より後に足した項目(発進の前進・後ろの見張りは 2026-10-26。あとの 2 つは fix3)
    assert {k: v for k, v in drive.DEFAULT_PARAMS.items() if k not in added} == before.DEFAULT_PARAMS
    assert {k: v for k, v in drive.PARAM_LIMITS.items() if k not in added} == before.PARAM_LIMITS
    assert drive.PARAM_LIMITS['zone_caps'] == (0, 1)
    assert 'zone_caps' in drive.RESET_ON_BOOT and set(before.RESET_ON_BOOT) <= set(drive.RESET_ON_BOOT)
    path = tmp_path / 'drive_settings.json'
    path.write_text('{"params": {"zone_caps": 1, "steer_gain": 1.5}, "run": "x", "mode_id": 5}')
    monkeypatch.setattr(drive, 'SETTINGS_PATH', path)
    d = drive.Driver(hub=None, recorder=FakeRecorder())
    assert d.params['zone_caps'] == 0 and d.params['steer_gain'] == 1.5       # ON は次の起動に持ち越さない
    assert d.set_params({'zone_caps': 0.7})['zone_caps'] == 1
    assert d.set_params({'zone_caps': 0.2})['zone_caps'] == 0
    assert d.set_params({'zone_caps': 5})['zone_caps'] == 1
    assert d.status()['zone_caps_spec']['caps_mps'] == {'R1': 2.5, 'GM': 2.2, 'DE': 3.0, 'unknown': 3.0}


# ---------------------------------------------------------------- OFF は変更前と同じ
def test_off_is_bit_identical_to_before(monkeypatch, tmp_path):
    before = load_before()
    frames, imu = scenario()
    _, pca0, rec0 = run_det(before, monkeypatch, tmp_path, PARAMS, frames, imu)
    d1, pca1, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    assert len(pca0.pairs) > 300
    assert pca0.pairs == pca1.pairs and pca0.single == pca1.single
    assert_rows_equal_before(rec0.rows, rec1.rows)
    # 影の判定は記録に残っている(区間・上限ならこう出した値・ブレーキ)。OFF なので出力は変えていない
    rows = [r for _, r in rec1.rows]
    assert all('cap_mode' in r and r['cap_on'] is False and r['cap_applied'] is False for r in rows)
    modes = collections.Counter(r['cap_mode'] for r in rows)
    assert modes[drive.CAP_MODE_CODES['brake']] > 0 and modes[drive.CAP_MODE_CODES['cap']] > 0
    assert modes[drive.CAP_MODE_CODES['neutral']] > 0 and modes[-1] > 0
    assert any(r['cap_thr_us'] > r['out'][1] for r in rows if not r['hold'])     # 上限なら弱めていたフレームがある
    assert d1.st['zone_caps_error'] is None and d1.st['stats']['zone_applied'] == 0
    # 推論の確認(dry_run)も同じ
    _, pca0, rec0 = run_det(before, monkeypatch, tmp_path, PARAMS, frames[:120], imu, dry_run=True)
    _, pca1, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames[:120], imu, dry_run=True)
    assert pca0.pairs == pca1.pairs
    assert_rows_equal_before(rec0.rows, rec1.rows)


def test_exception_keeps_output_identical_and_is_reported(monkeypatch, tmp_path):
    """上限の計算が例外 → その走行のあいだは上限を外す(ON でも OFF と同じ出力)・記録とメッセージに残す"""
    before = load_before()
    frames, imu = scenario(260)
    _, pca0, rec0 = run_det(before, monkeypatch, tmp_path, PARAMS, frames, imu)
    real = sc.zone_frame
    calls = [0]

    def boom(*a, **kw):
        calls[0] += 1
        if calls[0] == 40:
            raise RuntimeError('試験の例外')
        return real(*a, **kw)
    monkeypatch.setattr(sc, 'zone_frame', boom)
    d, pca1, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    k = next(i for i, (_, r) in enumerate(rec1.rows) if r['cap_mode'] == -2)
    assert all(r['cap_mode'] == -2 and not r['cap_applied'] for _, r in rec1.rows[k:])
    assert_rows_equal_before(rec0.rows[k:], rec1.rows[k:])           # 例外のあとは OFF(変更前)と同じ
    assert pca1.pairs[-len(rec1.rows) + k:] == pca0.pairs[-len(rec0.rows) + k:]
    assert 'RuntimeError' in d.st['zone_caps_error'] and '上限を外しました' in d.st['recording_note']
    assert '上限を外しました' in d.st['message']
    assert [e[0] for e in rec1.events] == ['zone_caps_error']
    # 最初のフレームから例外なら、全部 OFF と同じ
    monkeypatch.setattr(sc, 'zone_frame', lambda *a, **kw: (_ for _ in ()).throw(ValueError('x')))
    _, pca2, rec2 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    assert pca2.pairs == pca0.pairs
    assert_rows_equal_before(rec0.rows, rec2.rows)


# ---------------------------------------------------------------- ON は純粋な包みと同じ
def test_on_matches_pure_wrapper_and_never_more_forward(monkeypatch, tmp_path):
    frames, imu = scenario()
    _, pca0, rec0 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    d1, pca1, rec1 = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=1), frames, imu)
    assert len(rec0.rows) == len(rec1.rows)
    # 純粋な包みを、同じ入力(車に出す直前の値 = OFF の記録の out)で回す
    st = sc.ZoneState()
    origin = imu[0][0]
    veh_seen, k = [], 0
    n_applied = 0
    by_t = {t: r for t, r in rec0.rows}
    on_by_t = {t: r for t, r in rec1.rows}
    for f in frames:
        if not f['veh_skip']:
            veh_seen.append((f['cam_t'] - 3_000_000, f['veh']))
        while k < len(imu) and imu[k][0] <= f['cam_t']:
            t, g, _ = imu[k]
            prev = [x for x in veh_seen if x[0] <= t]
            e = (prev[-1] if prev else veh_seen[0])[1]['erpm']
            gz = float(np.degrees(np.float32(np.deg2rad(g))))
            st = sc.zone_imu(st, (t - origin) / 1e9, gz, -float(e) * sc.KV)
            k += 1
        r_off, r_on = by_t.get(f['cam_t']), on_by_t.get(f['cam_t'])
        if r_off is None:
            continue
        if r_off['hold']:
            st = sc.zone_inactive(st)
            assert r_on['hold'] and r_on['out'] == r_off['out'] and r_on['cap_mode'] == -1
            continue
        veh_t, veh = veh_seen[-1]
        active = veh['select'] and (f['cam_t'] - veh_t) / 1e9 < 0.1
        thr_in, steer = r_off['out'][1], r_off['out'][0]
        st, z = sc.zone_frame(st, (f['cam_t'] - origin) / 1e9, thr_in, steer, -float(veh['erpm']) * sc.KV,
                              r_off['pred'][:, 0].tolist(), active)
        assert r_on['cap_in_us'] == thr_in and r_off['cap_thr_us'] == z.thr_us == r_on['cap_thr_us']
        assert r_on['cap_mode'] == drive.CAP_MODE_CODES[z.mode] == r_off['cap_mode']
        assert r_on['cap_zone'] == drive.ZONE_CODES[z.zone]
        assert r_on['out'][0] == steer                                   # 舵は変えない
        assert r_on['out'][1] == z.thr_us >= thr_in                      # 前進を強めない
        assert r_on['cap_applied'] == (z.thr_us != thr_in)
        n_applied += r_on['cap_applied']
    assert n_applied > 20 and d1.st['stats']['zone_applied'] == n_applied
    modes = collections.Counter(r['cap_mode'] for _, r in rec1.rows if r['cap_applied'])
    assert set(modes) == {1, 2, 3}                                       # 上限で切った・中立・ブレーキ
    assert d1.st['stats']['zone_brakes'] >= 2
    # ジャイロの位相が R1 を見つけ、2 周目の R1 の進入で(合図 (b) なしに)ブレーキを出した
    gyro_brake = [r for _, r in rec1.rows if r['cap_mode'] == 3 and r['cap_zone_gyro'] == drive.ZONE_CODES['R1_app']]
    assert gyro_brake
    # 人の操縦・車両の値が古いフレームでは層を動かさない
    for f in frames:
        r = on_by_t.get(f['cam_t'])
        if r is not None and not r['hold'] and (not f['veh']['select']):
            assert r['cap_mode'] == -1 and not r['cap_applied']
    assert d1.st['now']['zone']['on'] is True and d1.st['zone_caps_error'] is None
    assert 0 < d1.st['stats']['zone_ms'] < 5.0


# ---------------------------------------------------------------- 純粋な包みの小さな規則
def test_zone_frame_rules():
    st = sc.ZoneState()
    # 合図 (b) は AI 側のフレームだけ・1 m/s 以上
    st, z = sc.zone_frame(st, 0.0, 1250.0, 1450.0, 3.0, [1450.0] * 6 + [1120.0] * 4, False)
    assert z.mode == 'off' and z.thr_us == 1250.0 and not z.pred_signal and z.zone == 'unknown'
    st, z = sc.zone_frame(st, 0.033, 1250.0, 1450.0, 3.1, [1450.0] * 6 + [1120.0] * 4, True)
    assert z.pred_signal and z.zone == 'R1_app' and z.mode == 'brake' and z.brake_start
    assert z.thr_us == pytest.approx(1698.0)
    # 0.2 秒は保つ
    st, z = sc.zone_frame(st, 0.066, 1250.0, 1450.0, 2.0, [1450.0] * 6 + [1120.0] * 4, True)
    assert z.mode == 'brake' and not z.brake_start
    # 0.5 秒を超えて間が空いたら層の状態を戻す(ブレーキの保持もやめる)
    st, z = sc.zone_frame(st, 0.7, 1250.0, 1450.0, 2.0, [1450.0] * 6 + [1120.0] * 4, True)
    assert z.mode == 'cap' and st.cap.brake_t1 == -sc.INF
    # 中立で止めたら層の状態を戻す(位相は残す)
    st2 = sc.zone_inactive(st._replace(zone_gyro='GM'))
    assert st2.cap == sc.CapState() and st2.zone_gyro == 'GM'
    # IMU のサンプルの時刻が戻ったら捨てる
    s = sc.zone_imu(sc.ZoneState(), 1.0, 10.0, 2.0)
    assert sc.zone_imu(s, 0.99, 500.0, 2.0) is s
    # 区間のない(none)ところでは前進はそのまま
    st, z = sc.zone_frame(sc.ZoneState(zone_gyro='none'), 0.0, 1250.0, 1450.0, 3.0, [1450.0] * 10, True)
    assert z.mode == 'pass' and z.thr_us == 1250.0


def test_zone_frame_matches_cap_throttle_us_random():
    """zone_frame の上限の部分は cap_throttle_us(G1 の再生の shadow と同じ式)そのもの"""
    rng = np.random.default_rng(0)
    st = sc.ZoneState()
    cst = sc.CapState()
    for i in range(3000):
        t = i / 30.0
        zone = sc.ZONES[int(rng.integers(0, len(sc.ZONES)))]
        st = st._replace(zone_gyro=zone, turning=bool(rng.random() < 0.1))
        v = float(rng.uniform(0, 4.5))
        thr = float(rng.uniform(1100, 1700))
        steer = float(rng.uniform(1000, 2000))
        pred = rng.uniform(1100, 1800, 10).tolist()
        st, z = sc.zone_frame(st, t, thr, steer, v, pred, True)
        zone_c = sc.combine_zone(zone, st.turning, sc.pred_signal_of(pred), v=v)
        cst, thr_c, o = sc.cap_throttle_us(cst, t, zone_c, v, thr, sc.params_real_car(), steer, 1443.0)
        assert z.zone == zone_c and z.thr_us == thr_c and z.mode == o.mode and z.thr_us >= thr


# ---------------------------------------------------------------- 実車の記録で G1 の再生と同じ(記録があるときだけ)
G1_PATH = ROOT / 'experiments' / '20261016_sim_calibration' / 'g1_replay.py'
G1_TEST_LOGS = ('20261010-173445', '20261010-204605')          # ai_run_007(R1 の強い衝突)・ai_run_101


def _load_g1():
    spec = importlib.util.spec_from_file_location('g1_replay_for_test', str(G1_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('log_name', G1_TEST_LOGS)
def test_pure_wrapper_matches_g1_replay_on_real_logs(log_name):
    if not (ROOT / 'data' / 'labels' / (log_name + '.json')).exists() or not (ROOT / 'data' / 'raw' / log_name).exists():
        pytest.skip('10/10 の記録がありません: %s' % log_name)
    g1 = _load_g1()
    d = g1.prepare(log_name)
    sh = g1.shadow(d, 'v1.1', sc.params_real_car())
    L = g1.le.load_log(log_name)
    A = L['AI_CMD']
    t0 = d['info']['t0_us']
    at_raw = (A['timestamps_us'].astype(np.int64) - t0) / 1e6
    o_ = np.argsort(at_raw, kind='stable')
    pred = A['pred'][o_][:, :, 0]
    steer_us = A['out'][o_][:, 0].astype(np.float64)
    it, gz, vt, erpm = d['it'], d['gz'], d['vt'], d['erpm']
    at, sel, thr, v_at = d['at'], d['sel'], d['thr'], d['v_at']
    st = sc.ZoneState()
    k = 0
    n_diff = 0
    n_mode = collections.Counter()
    for i in range(len(at)):
        while k < len(it) and it[k] <= at[i]:
            kv = int(np.clip(np.searchsorted(vt, it[k], side='right') - 1, 0, len(vt) - 1))
            st = sc.zone_imu(st, float(it[k]), float(gz[k]), float(-erpm[kv] * g1.KV))
            k += 1
        if k == 0:
            continue                       # 最初の IMU より前のフレーム(G1 の再生は先頭のサンプルを使う)
        st, z = sc.zone_frame(st, float(at[i]), float(thr[i]), float(steer_us[i]), float(v_at[i]), pred[i].tolist(),
                              bool(sel[i]))
        zone_g1 = d['ver']['v1.1']['zone'][i]
        mode_g1 = sh['mode'][i]
        thr_g1 = float(sh['thr_out'][i])
        if z.zone != zone_g1 or z.mode != mode_g1 or z.thr_us != thr_g1 or z.brake_start != bool(sh['brake_start'][i]):
            n_diff += 1
        n_mode[z.mode] += 1
    assert n_diff == 0
    assert n_mode['off'] > 0 and n_mode['pass'] > 0 and (n_mode['cap'] + n_mode['neutral'] + n_mode['brake']) > 0


# ---------------------------------------------------------------- 記録(AI_CMD.npz)に影の判定が残る
class _Hub(object):
    def __getitem__(self, name):
        return self

    def subscribe(self, cb):
        pass

    def unsubscribe(self, cb):
        pass


def test_ai_cmd_npz_keeps_shadow_fields(monkeypatch, tmp_path):
    """走行ループの記録の行(cap_*)が AI_CMD.npz に入る。古い行(cap_* なし)は既定値(-1・nan・out のスロットル)"""
    from jetformer.recorder import Recorder
    frames, imu = scenario(240)                 # 230〜232 は CH3 が届かない = 中立で止めた行
    _, _, rec_fake = run_det(drive, monkeypatch, tmp_path, dict(PARAMS, zone_caps=0), frames, imu)
    monkeypatch.undo()
    monkeypatch.setattr(config, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(config, 'SCENE_SECONDS', 1.0)
    rec = Recorder(_Hub())
    rec._warm_up = lambda: None
    log = rec.start(description='t')
    t0 = rec.log['start_ns']
    for i in range(20):
        rec._on_sample('CAM_FRONT', Sample(t0 + i * 33_000_000, {'image': np.zeros((8, 8, 3), np.uint8), 'seq': i}))
    rows = [r for _, r in rec_fake.rows if not r['hold'] and r['cap_mode'] == 3][:1] + \
           [r for _, r in rec_fake.rows if r['hold']][:1]
    assert len(rows) == 2
    old_row = {k: v for k, v in rows[0].items() if not k.startswith('cap_')}
    for j, r in enumerate(rows + [old_row]):
        rec.add_ai(t0 + (2 + j) * 33_000_000, r)
    rec._io.join()
    rec._finalize(0, partial=False)
    rec.stop(only_log=log)
    z = np.load(tmp_path / log / ('%s-0001' % log) / 'AI_CMD.npz')
    assert z['cap_mode'].tolist() == [3, -1, -1]
    assert z['cap_zone'][0] == drive.ZONE_CODES[sc.ZONES[rows[0]['cap_zone']]] and z['cap_zone'][2] == -1
    assert z['cap_thr_us'][0] == pytest.approx(rows[0]['cap_thr_us']) and np.isnan(z['cap_thr_us'][2])
    assert z['cap_in_us'][2] == pytest.approx(old_row['out'][1])
    assert z['cap_on'].tolist() == [False, False, False] and z['cap_applied'].tolist() == [False, False, False]
    assert z['cap_limit'].dtype == np.float32 and z['cap_brake_start'].dtype == bool
    for k in ('cap_zone_gyro', 'cap_pred_sig', 'cap_rel_psi'):
        assert len(z[k]) == 3


@pytest.mark.parametrize('log_name', G1_TEST_LOGS)
def test_drive_glue_matches_g1_replay_on_real_logs(log_name):
    """走行ループのつなぎ(drive._ZoneCaps: バッファからの IMU・車両の値の取り出し・撮像時刻までの区切り・0.1 秒の鮮度)を、
    記録の時刻どおりにバッファへ入れて回すと、G1 の再生と区間・出し方・ブレーキの始まりが全フレームで同じ。
    スロットルは時刻の原点の違い(記録の始まり / 最初の IMU)で、上限を戻す途中だけ 1e-12 µs ほど違うことがある"""
    if not (ROOT / 'data' / 'labels' / (log_name + '.json')).exists() or not (ROOT / 'data' / 'raw' / log_name).exists():
        pytest.skip('10/10 の記録がありません: %s' % log_name)
    g1 = _load_g1()
    d = g1.prepare(log_name)
    sh = g1.shadow(d, 'v1.1', sc.params_real_car())
    L = g1.le.load_log(log_name)
    A, I, V = L['AI_CMD'], L['IMU'], L['VEHICLE']
    o_ = np.argsort(A['timestamps_us'].astype(np.int64), kind='stable')
    a_us = A['timestamps_us'].astype(np.int64)[o_]
    pred, out = A['pred'][o_], A['out'][o_].astype(np.float64)
    hold, dry, sel = (A[k][o_].astype(bool) for k in ('hold', 'dry_run', 'select'))
    i_us, gz = I['timestamps_us'].astype(np.int64), I['gyro'][:, 2]
    v_us, erpm = V['timestamps_us'].astype(np.int64), V['erpm']
    zc = drive._ZoneCaps()
    ib, vb = collections.deque(maxlen=60), collections.deque(maxlen=50)
    ki = kv = 0
    n = collections.Counter()
    for i in range(len(a_us)):
        while ki < len(i_us) and i_us[ki] <= a_us[i] + 2000:           # 撮像の少しあとの IMU も届いている
            ib.append((int(i_us[ki]) * 1000, np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(np.float64(gz[ki]))])))
            ki += 1
        while kv < len(v_us) and v_us[kv] <= a_us[i]:
            vb.append((int(v_us[kv]) * 1000, {'erpm': int(erpm[kv])}))
            kv += 1
        veh = next((dd for tv, dd in reversed(vb) if tv <= a_us[i] * 1000), None)
        if hold[i] or veh is None:                                       # 走行ループでは中立で止めるフレーム
            zc.hold(int(a_us[i]) * 1000, ib, vb)
            continue
        z = zc.frame(int(a_us[i]) * 1000, (out[i, 0], out[i, 1]), veh, pred[i], bool(sel[i]) and not dry[i], ib, vb)
        assert z is not None, zc.failed
        assert (z.zone, z.mode, z.brake_start) == (d['ver']['v1.1']['zone'][i], sh['mode'][i], bool(sh['brake_start'][i])), i
        assert abs(z.thr_us - float(sh['thr_out'][i])) < 1e-6, i
        n[z.mode] += 1
    assert n['off'] > 0 and n['pass'] > 0 and n['brake'] > 0
    assert np.mean(zc.ms) < 1.0
