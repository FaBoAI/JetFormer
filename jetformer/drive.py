"""AI走行: 学習結果(TensorRTエンジン)で車を操作する

カメラの新しいフレーム(30fps)ごとに:
  1. その撮像時刻までのToF・IMU(直近10サンプル)・車両の値を集める(データセットの aligned と同じ規則)
  2. normalize_inputs(学習と共通)→ TensorRT推論 → 未来10フレームの [ステアリング, スロットル] µs
  3. 「先読み」のフレームを採用 → ゲイン・トリム → PCA9685
プロポのCH3がAI側(SELECT=1、LED紫)のときだけ出力が車に届く。CH3を戻せば常に人の操縦に戻る。

モデルへの「現在の操作値」は、車に届いている操作に合わせる:
  人が操縦中 → プロポの値 / AI側 → 前のフレームの予測のうち「このフレームの時刻」に当たる値(ゲイン前)
  / 中立を出している間 → 中立
ゲイン後の値を戻すと、ゲイン>1のとき自分の出力を真似て際限なく大きくなる。また先読みで採用した値
(数フレーム先)を戻すと、毎フレーム先へ進んだことになって動きが速回しになる。どちらも学習時と違うので避ける。

安全のための動作(どれも中立を出す): センサーの値が途切れた・古い / 推論結果が異常 / AI側なのにプロポのCH3が
届いていない(送信機の電源断・圏外では基板のSELECTがAI側のまま残るため)/ PCA9685への書き込み失敗。
走行を始めたページからの問い合わせが3秒途絶えたら走行を止める。
"""

import collections
import json
import os
import threading
import time
import traceback
import uuid

import cv2
import numpy as np

from . import assist as assist_mod
from . import clock, config, modes
from .laps import LapCounter

SETTINGS_PATH = config.DATA_DIR / 'drive_settings.json'
RUNS_DIR = config.DATA_DIR / 'runs'
ENGINE_FILE = 'jetformer_fp16.engine'
IMU_STEPS = 10                     # model.IMU_STEPS(torchを読み込まずに使うため複製)
PWM_CENTER, PWM_SCALE = 1500.0, 500.0

PARAM_LIMITS = {
    'steer_gain': (0.0, 2.0),      # ステアリングのゲイン(中心からの切れ角を何倍にするか)
    'throttle_gain': (0.0, 2.0),   # 速度のゲイン(中立からのスロットル量を何倍にするか)
    'steer_trim_us': (-200.0, 200.0),
    'lookahead': (1, 10),          # 何フレーム先の予測を使うか(1 = 約33ms先)
    'osc_percent': (-10.0, 10.0),  # PCA9685の時計補正 [%]
    'esc_reverse': (0, 1),         # 1 = 後退の指令の前に中立を挟む(このESCは中立を挟まないと後退しない)
    'recovery': (0, 1),            # 1 = ぶつかって止まったら自動で少し下がってからモデルへ戻す
    'rev_us': (150.0, 500.0),      # リカバリーで下がる強さ(中立からのµs)
    'rev_s': (0.2, 1.2),           # リカバリーで後退を出し続ける時間(下がり始めてから)
    'tof_guard': (0, 3),           # ToFガード 0=OFF 1=記録だけ(動きは変えない) 2=減速 3=減速+ハンドル
}
GUARD_NAMES = {0: '', 1: '減速', 2: 'ブレーキ'}
FLOOR_REF_PATH = config.DATA_DIR / 'tof_floor_ref.npy'      # ToFガードの床の基準(自由走行から作る)
DEFAULT_PARAMS = {'steer_gain': 1.0, 'throttle_gain': 0.5, 'steer_trim_us': 0.0, 'lookahead': 1, 'osc_percent': 0.0,
                  'esc_reverse': 1, 'recovery': 1, 'rev_us': assist_mod.DEFAULT_REV_US, 'rev_s': assist_mod.DEFAULT_REV_S,
                  'tof_guard': 1}


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _trained_modes_from_dataset(dataset, stride):
    """古い学習結果用: データセットの学習用シーンに含まれていたジャンル(torchを使わずに数える)"""
    root = config.DATASET_DIR / (dataset or '')
    splits = _read_json(root / 'splits.json') if dataset else None
    if not splits:
        return []
    found = set()
    for scene in splits.get('train', []):
        try:
            a = np.load(root / 'aligned' / (scene + '.npz'))
            ok = a['future_valid'] & a['vehicle_valid']
            mode = a['mode'] if 'mode' in a else np.zeros(len(ok), np.int16)
            ok &= (mode != modes.UNSET) & ((np.arange(len(ok)) % max(int(stride or 1), 1)) == 0)
            found.update(int(m) for m in np.unique(mode[ok]))
        except (OSError, KeyError, ValueError):
            continue
    return sorted(found)


def model_info(name):
    """走行に使える学習結果の情報。エンジンが無ければ None"""
    d = RUNS_DIR / name
    if '/' in name or '..' in name or not (d / ENGINE_FILE).exists():
        return None
    cfg = _read_json(d / 'model_config.json')
    if cfg is None:
        return None
    res = _read_json(d / 'result.json') or {}
    run = _read_json(d / 'run.json') or {}
    trained = cfg.get('trained_modes')
    if trained is None:            # 古い学習結果: 学習用シーンに含まれていたジャンル(分からなければ走らせない)
        trained = _trained_modes_from_dataset(cfg.get('dataset'), cfg.get('stride'))
    trained = [m for m in trained if 0 <= m < cfg['num_modes']]
    return {
        'name': name, 'dataset': cfg.get('dataset'), 'created_at': run.get('created_at'),
        'state': res.get('state'), 'best_val_us': res.get('best_val_us'),
        'best_val_per_mode_us': res.get('best_val_per_mode_us', {}),
        'num_modes': cfg['num_modes'], 'modes': cfg.get('modes', []), 'trained_modes': trained,
        'steer_center_us': cfg.get('steer_center_us'), 'throttle_neutral_us': cfg.get('throttle_neutral_us'),
        'stride': cfg.get('stride'), 'future_steps': cfg.get('future_steps', 10),
        'command_input': cfg.get('command_input', True),
    }


def list_models():
    """新しい順(作成日時、無ければ名前)"""
    if not RUNS_DIR.exists():
        return []
    models = [m for m in (model_info(d.name) for d in RUNS_DIR.iterdir()) if m]
    return sorted(models, key=lambda m: (m['created_at'] or '', m['name']), reverse=True)


def _reference_from_dataset(dataset):
    """古い学習結果用: データセットから(ステアリング中心, スロットル中立)を求める"""
    from .train_data import JetFormerDataset
    root = config.DATASET_DIR / (dataset or '')
    if not dataset or not (root / 'splits.json').exists():
        return PWM_CENTER, PWM_CENTER
    return JetFormerDataset(str(root), 'train').action_reference()


def assemble_inputs(cam_t, tof_buf, imu_buf, veh_buf, max_age_ns, imu_max_age_ns):
    """撮像時刻 cam_t 以前の最新のToF・直近10個のIMU・最新の車両値を選ぶ(dataset._aligned_arrays と同じ規則)。

    tof_buf: [(t_ns, depth_raw)]、imu_buf: [(t_ns, [ax,ay,az m/s^2, gx,gy,gz rad/s])]、veh_buf: [(t_ns, dict)]
    戻り値 (tof_raw, imu[10,6], vehicle_dict, None) または (None, None, None, 止める理由)"""
    tof = [x for x in tof_buf if x[0] <= cam_t]
    if not tof:
        return None, None, None, 'ToFの値が来ていません'
    if cam_t - tof[-1][0] > max_age_ns:
        return None, None, None, 'ToFの値が古い(%dms)' % ((cam_t - tof[-1][0]) / 1e6)
    imu = [x for x in imu_buf if x[0] <= cam_t]
    if len(imu) < IMU_STEPS:
        return None, None, None, 'IMUの値が足りません'
    if cam_t - imu[-1][0] > imu_max_age_ns:
        return None, None, None, 'IMUの値が古い(%dms)' % ((cam_t - imu[-1][0]) / 1e6)
    veh = [x for x in veh_buf if x[0] <= cam_t]
    if not veh:
        return None, None, None, '車両(PWM)の値が来ていません'
    if cam_t - veh[-1][0] > max_age_ns:
        return None, None, None, '車両(PWM)の値が古い(%dms)' % ((cam_t - veh[-1][0]) / 1e6)
    hist = np.stack([x[1] for x in imu[-IMU_STEPS:]]).astype(np.float32)
    return tof[-1][1], hist, veh[-1][1], None


def vehicle_state(veh, prev_cmd, ref, applied):
    """モデルに渡す [steering_us, throttle_us, mode_us, erpm](学習時は人の操作値)。

    prev_cmd: 前のフレームの予測のうち、このフレームの時刻に当たる値(ゲイン前)
    applied: 前のフレームでAIの出力を出したか(中立で止めていたら False)"""
    if veh['select']:
        steer, throttle = prev_cmd if (applied and prev_cmd is not None) else ref
    elif veh['valid'][0] and veh['valid'][1]:
        steer, throttle = veh['pwm'][0], veh['pwm'][1]
    else:
        steer, throttle = ref
    return np.array([steer, throttle, veh['pwm'][2], veh['erpm']], dtype=np.float32)


def predicted_now(prev_pred, prev_cam_t, cam_t, fps):
    """前のフレームの予測 [10,2] から、撮像時刻 cam_t に当たる値を取り出す(取りこぼしたフレーム分も進める)"""
    gap = int(round((cam_t - prev_cam_t) * fps / 1e9))
    k = min(max(gap, 1), len(prev_pred)) - 1
    return float(prev_pred[k, 0]), float(prev_pred[k, 1])


def apply_gains(raw, ref, p, lo, hi):
    """採用した予測 raw=(steer, throttle) µs → 出力 µs。ゲインは中心/中立の周りに掛ける。
    速度のゲインは前進側だけ(ブレーキ・後退側はモデルのまま。ゲイン2.0でブレーキが2000µsに張り付いたため)"""
    steer = ref[0] + p['steer_trim_us'] + p['steer_gain'] * (raw[0] - ref[0])
    dt = raw[1] - ref[1]
    throttle = ref[1] + (p['throttle_gain'] * dt if dt < 0 else dt)
    return float(np.clip(steer, lo, hi)), float(np.clip(throttle, lo, hi))


class _Output(object):
    """PCA9685への出力。I2Cの一時的な失敗は読み直して続け、続けて失敗したら走行を止める"""
    MAX_FAILURES = 5          # 約150ms

    def __init__(self, pca, cfg, ref, params):
        self.pca, self.cfg, self.ref, self.params = pca, cfg, ref, params
        self.failures = 0
        self.errors = 0

    def neutral_values(self):
        return self.ref[0] + self.params()['steer_trim_us'], self.ref[1]

    def write(self, steer_us, throttle_us):
        """書けたら True。失敗したら中立を試み、続けて失敗したら RuntimeError"""
        if self.pca is None:
            return True
        try:
            self.pca.set_pair_us(self.cfg['steering_ch'], steer_us, self.cfg['throttle_ch'], throttle_us)
            self.failures = 0
            return True
        except OSError as e:
            self.errors += 1
            self.failures += 1
            self.neutral_safe(tries=2)
            if self.failures >= self.MAX_FAILURES:
                raise RuntimeError('PCA9685への書き込みが続けて失敗しました: %s' % e)
            return False

    def neutral(self):
        return self.write(*self.neutral_values())

    def neutral_safe(self, tries=6):
        """止めるとき用: スロットルを先に、チャネルごとに何度か試す。全部書けたら True"""
        if self.pca is None:
            return True
        steer, throttle = self.neutral_values()
        ok = True
        for ch, us in ((self.cfg['throttle_ch'], throttle), (self.cfg['steering_ch'], steer)):
            for _ in range(tries):
                try:
                    self.pca.set_us(ch, us)
                    break
                except OSError:
                    time.sleep(0.03)
            else:
                ok = False
        return ok


class Driver(object):
    def __init__(self, hub, recorder):
        self.hub = hub
        self.recorder = recorder
        self.lock = threading.Lock()
        self.thread = None
        self._stop = threading.Event()
        saved = _read_json(SETTINGS_PATH) or {}
        self.params = dict(DEFAULT_PARAMS)
        self.params.update({k: v for k, v in saved.get('params', {}).items() if k in DEFAULT_PARAMS})
        self.last = {'run': saved.get('run'), 'mode_id': saved.get('mode_id')}
        self.heartbeat_t = time.monotonic()
        self.hb_token = None
        self.hb_count = 0
        self.history = collections.deque(maxlen=300)
        self.st = {'phase': 'idle', 'message': '停止中', 'error': None}
        self.listeners = []        # 出来事の通知先 cb(event, **info)(音声案内など)。走行ループから呼ぶので待たせないこと

    def _emit(self, event, **info):
        for cb in list(self.listeners):
            try:
                cb(event, **info)
            except Exception:
                traceback.print_exc()

    # ------------------------------------------------------------ 操作
    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def heartbeat(self, token):
        """走行を始めたページだけが生存確認になる(ほかのページが開いていても、操作しているページが
        閉じた・通信が切れたら止まるように)"""
        if token and token == self.hb_token:
            self.heartbeat_t = time.monotonic()
            self.hb_count += 1
            return True
        return False

    def _clean_params(self, values):
        p = dict(self.params)
        for k, (lo, hi) in PARAM_LIMITS.items():
            if values.get(k) is not None:
                v = float(values[k])
                if not np.isfinite(v):
                    raise ValueError('%s が数値ではありません' % k)
                p[k] = min(max(v, lo), hi)
        p['lookahead'] = int(round(p['lookahead']))
        for k in ('esc_reverse', 'recovery'):         # 古い設定ファイル・古い画面には無い項目
            p[k] = 1 if float(p.get(k, DEFAULT_PARAMS[k])) >= 0.5 else 0
        p['tof_guard'] = int(round(float(p.get('tof_guard', DEFAULT_PARAMS['tof_guard']))))
        return p

    def _save_settings(self):
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(SETTINGS_PATH) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'params': self.params, 'run': self.last['run'], 'mode_id': self.last['mode_id']}, f,
                          ensure_ascii=False, indent=1)
            os.replace(tmp, SETTINGS_PATH)
        except OSError:
            pass

    def set_params(self, values):
        """走行中でも反映する(ゲイン・トリム・先読み・時計補正・ジャンル)。送られた項目だけ変える"""
        with self.lock:
            p = self._clean_params(values)
            if values.get('mode_id') is not None:
                mode_id = int(values['mode_id'])
                info = self.st.get('model')
                if self.running and info and mode_id not in info['trained_modes']:
                    raise ValueError('このモデルはジャンル %d を学習していません' % mode_id)
                if self.running:
                    self.st['mode_id'] = mode_id
                    if self.recorder.recording:
                        try:
                            self.recorder.set_mode(mode_id)
                        except (RuntimeError, ValueError):
                            pass
                self.last['mode_id'] = mode_id
            self.params = p
            self._save_settings()
            return dict(p)

    def start(self, run, mode_id, record=False, description='', tags=None, dry_run=False, params=None):
        """走行を始める。戻り値 (run, token)。token は生存確認(heartbeat)に使う"""
        with self.lock:
            if self.running:
                raise RuntimeError('すでに走行中です')
            info = model_info(run or '')
            if info is None:
                raise ValueError('走行に使える学習結果がありません(TensorRTエンジンが必要): %s' % run)
            mode_id = int(mode_id)
            if mode_id not in info['trained_modes']:
                raise ValueError('このモデルはジャンル %d(%s)を学習していません' % (mode_id, modes.name_of(mode_id)))
            self.params = self._clean_params(params or {})
            self.last = {'run': run, 'mode_id': mode_id}
            self._save_settings()
            self._stop.clear()
            self.history.clear()
            self.hb_token = uuid.uuid4().hex
            self.heartbeat_t = time.monotonic()
            self.hb_count = 0
            self.st = {'phase': 'starting', 'message': 'モデルとセンサーを準備中…', 'error': None, 'run': run,
                       'model': info, 'mode_id': mode_id, 'dry_run': bool(dry_run), 'record': bool(record),
                       'recording_note': '', 'ref': None, 'started': time.time(), 'stop_reason': '',
                       'stats': {}, 'now': None}
            self.thread = threading.Thread(target=self._run, args=(info, bool(record), description or '',
                                                                   list(tags or []), bool(dry_run)),
                                           name='drive', daemon=True)
            self.thread.start()
            return run, self.hb_token

    def stop(self, reason='停止ボタン'):
        if not self.running:
            raise RuntimeError('走行していません')
        self.st['stop_reason'] = reason
        self.st['message'] = '停止中…'
        self._stop.set()

    def shutdown(self, timeout=3.0):
        """サーバー終了時: 止めて、中立を出し終わるまで待つ"""
        if self.running:
            try:
                self.stop('サーバー終了')
            except RuntimeError:
                pass
            self.thread.join(timeout)

    # ------------------------------------------------------------ 走行ループ
    def _run(self, info, record, description, tags, dry_run):
        st = self.st
        pca = out = runner = listeners = None
        started_log = None
        ref = (PWM_CENTER, PWM_CENTER)
        try:
            self.hub.start()
            from .actuator import PCA9685
            if info['steer_center_us'] is not None and info['throttle_neutral_us'] is not None:
                ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
            else:
                ref = _reference_from_dataset(info['dataset'])
            st['ref'] = {'steer_center_us': ref[0], 'throttle_neutral_us': ref[1]}

            # モデルを読む前に中立を出す(前のプロセスが残した出力を、準備の数秒間も車に出し続けないように)
            pcfg = config.PCA9685
            osc_pct = self.params['osc_percent']
            try:
                pca = PCA9685(pcfg, 1.0 + osc_pct / 100.0)
            except Exception as e:
                if not dry_run:
                    raise RuntimeError('PCA9685(I2C 0x%02x)に出力できません: %s' % (pcfg['address'], e))
                st['recording_note'] = 'PCA9685に接続できないため、出力なしで推論だけ行います'
            out = _Output(pca, pcfg, ref, lambda: self.params)
            out.neutral()

            import torch
            from .train_data import normalize_inputs
            from .trt_runtime import TRTRunner
            torch.zeros(1, device='cuda')
            torch.cuda.synchronize()        # このスレッドでCUDAを初期化しておく
            runner = TRTRunner(str(RUNS_DIR / info['name'] / ENGINE_FILE))
            num_modes = info['num_modes']
            zeros = {'image': np.zeros((1, 3, 224, 224), np.float32), 'tof': np.zeros((1, 2, 42, 54), np.float32),
                     'imu': np.zeros((1, IMU_STEPS, 6), np.float32), 'state': np.zeros((1, 3), np.float32),
                     'mode': np.zeros((1, num_modes), np.float32)}
            for _ in range(3):              # 初回の推論は遅いので先に済ませる
                runner.run_numpy(**zeros)

            tof_buf = collections.deque(maxlen=10)
            imu_buf = collections.deque(maxlen=60)
            veh_buf = collections.deque(maxlen=50)
            cam_slot = [None]
            cam_evt = threading.Event()

            def on_sample(channel, s):
                if channel == 'CAM_FRONT':
                    cam_slot[0] = s
                    cam_evt.set()
                elif channel == 'TOF_FRONT':
                    tof_buf.append((s.t_ns, s.data['depth']))
                elif channel == 'IMU':
                    imu_buf.append((s.t_ns, np.concatenate([s.data['accel'], np.deg2rad(s.data['gyro'])])))
                elif channel == 'VEHICLE':
                    veh_buf.append((s.t_ns, s.data))
            listeners = on_sample
            for ch in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE'):
                self.hub[ch].subscribe(on_sample)

            if record:
                st['message'] = '収集を開始しています…'
                drive_info = {'run': info['name'], 'dataset': info['dataset'], 'dry_run': dry_run,
                              'params': dict(self.params), 'steer_center_us': ref[0], 'throttle_neutral_us': ref[1]}
                try:
                    started_log = self.recorder.start(description=description, tags=tags, mode_id=st['mode_id'],
                                                      drive=drive_info)
                except RuntimeError:
                    st['recording_note'] = 'すでに収集中のため、その収集にAIの指令も記録します'
            self._loop(st, runner, normalize_inputs, pca, out, ref, num_modes, osc_pct,
                       cam_slot, cam_evt, tof_buf, imu_buf, veh_buf, dry_run)
            st['phase'] = 'idle'
            st['message'] = '停止しました(%s)' % (st['stop_reason'] or '停止')
        except Exception as e:
            traceback.print_exc()
            st['phase'] = 'error'
            st['error'] = '%s: %s' % (type(e).__name__, e)
            st['message'] = '走行を開始できませんでした' if not st.get('stats') else '走行を止めました(エラー)'
        finally:
            # 後始末は1つずつ。どれかが失敗しても、ほかを必ず行う(中立が最優先)
            if out is not None and not out.neutral_safe():
                st['error'] = (st.get('error') or '') + ' / 停止時に中立を出せませんでした(CH3を人側へ戻してください)'
            if pca is not None:
                try:
                    pca.close()
                except Exception:
                    pass
            if listeners is not None:
                for ch in ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE'):
                    try:
                        self.hub[ch].unsubscribe(listeners)
                    except Exception:
                        pass
            if started_log:
                try:
                    st['message'] += '・収集したシーンを保存中…'
                    stopped = self.recorder.stop(only_log=started_log)     # 途中で人が始め直した収集は止めない
                    st['message'] = st['message'].replace('・収集したシーンを保存中…',
                                                          '・収集したシーンを保存しました' if stopped else '')
                except Exception:
                    traceback.print_exc()
            if st.get('announced'):
                self._emit('drive_stop', reason=st.get('stop_reason', ''), error=st['phase'] == 'error')
            if runner is not None:
                try:
                    runner = None              # GPUメモリを返す(学習に必要な空きを残す)
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    traceback.print_exc()
            st['finished'] = time.time()

    def _loop(self, st, runner, normalize_inputs, pca, out, ref, num_modes, osc_pct,
              cam_slot, cam_evt, tof_buf, imu_buf, veh_buf, dry_run):
        dcfg = config.DRIVE
        fps = config.CAMERA['fps']
        max_age = int(dcfg['max_sensor_age_ms'] * 1e6)
        imu_age = int(dcfg['imu_max_age_ms'] * 1e6)
        jpeg = [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA['jpeg_quality']]
        lo, hi = dcfg['min_us'], dcfg['max_us']
        stats = {'frames': 0, 'skipped': 0, 'holds': 0, 'fps': 0.0, 'latency_ms': 0.0, 'infer_ms': 0.0, 'i2c_errors': 0,
                 'recoveries': 0, 'esc_assists': 0, 'giveups': 0}
        helper = assist_mod.Assist(ref)
        last_kind = assist_mod.ASSIST_NONE
        guard = None                        # ToFガード(床の基準があるときだけ)
        try:
            from .tof_guard import ToFGuard
            guard = ToFGuard(np.load(FLOOR_REF_PATH), center=ref[0], neutral=ref[1])
        except (OSError, ValueError) as e:
            st['recording_note'] = (st.get('recording_note') or '') + ' ToFガードは使えません(床の基準 %s: %s)' % (
                FLOOR_REF_PATH.name, e)
        stats.update(guard_slow=0, guard_brake=0)
        last_guard = 0
        laps = LapCounter()                 # ジャイロで周回を数え、1周のタイムを測る
        st['laps'] = laps.summary()
        imu_done = [0]

        def count_laps(veh_now):
            """前のフレームから届いたIMUのサンプルを周回の計測へ"""
            if veh_now is None:
                return
            for t_ns, x in list(imu_buf):
                if t_ns <= imu_done[0]:
                    continue
                imu_done[0] = t_ns
                lap = laps.add(t_ns / 1e9, float(np.degrees(x[5])), float(veh_now['erpm']), bool(veh_now['select']),
                               float(np.hypot(x[0], x[1])) / 9.80665)
                if lap:
                    st['laps'] = laps.summary()
                    self._emit('lap', **lap)
        st['stats'] = stats
        st['phase'] = 'driving'
        st['message'] = '走行中' if not dry_run else '推論の確認中(出力は中立のまま)'
        st['announced'] = True
        self._emit('drive_start', dry_run=dry_run)
        last_select, rc_lost = None, False
        prev_pred, prev_cam_t, applied = None, None, False
        use_command = (st.get('model') or {}).get('command_input', True)   # 学習と同じく、今の操作値を入れるか
        last_seq = None
        mode_cache = {}
        times = collections.deque(maxlen=30)
        lat_hist = collections.deque(maxlen=30)
        inf_hist = collections.deque(maxlen=30)

        def hold(reason, cam=None, veh_select=None, infer_ms=np.nan):
            if veh_select is None:          # 最新の車両値から(AI側のまま止めているのか分かるように)
                veh_select = bool(veh_buf[-1][1]['select']) if veh_buf else False
            out.neutral()
            stats['holds'] += 1
            stats['i2c_errors'] = out.errors
            n_s, n_t = out.neutral_values()
            st['hold_reason'] = reason
            st['hold_since'] = st.get('hold_since') or time.time()
            st['now'] = {'raw': None, 'out': [round(n_s, 1), round(n_t, 1)], 'select': veh_select, 'hold': True, 'pred': None}
            if cam is not None:
                self.recorder.add_ai(cam.t_ns, {
                    'pred': np.full((10, 2), np.nan, np.float32), 'raw': (np.nan, np.nan), 'out': (n_s, n_t),
                    'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                    'mode_id': st['mode_id'], 'hold': True, 'select': veh_select, 'latency_ms': np.nan,
                    'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': assist_mod.ASSIST_NONE,
                    'model_out': (np.nan, np.nan), 'ttc': np.nan, 'guard': 0, 'guard_applied': False})
            helper.reset()                  # 中立で止めた後は、補助の手順を最初から

        while not self._stop.is_set():
            p = self.params
            if time.monotonic() - self.heartbeat_t > dcfg['heartbeat_s']:
                st['stop_reason'] = ('走行を始めた画面からの応答が一度もありませんでした。画面を再読み込みしてから開始してください'
                                     if self.hb_count == 0 else
                                     '走行を始めた画面との通信が %.0f 秒途絶えたため' % dcfg['heartbeat_s'])
                break
            if pca is not None and p['osc_percent'] != osc_pct:
                osc_pct = p['osc_percent']
                pca.set_osc_scale(1.0 + osc_pct / 100.0)
            if not cam_evt.wait(0.1):
                if self._stop.is_set():
                    break
                hold('カメラの映像が来ていません')
                applied = False
                continue
            cam_evt.clear()
            cam = cam_slot[0]
            seq = cam.data['seq']
            if seq == last_seq:
                continue
            if last_seq is not None and seq > last_seq + 1:
                stats['skipped'] += seq - last_seq - 1
            last_seq = seq
            if clock.now_ns() - cam.t_ns > max_age:
                hold('処理が遅れています(画像が%dms前)' % ((clock.now_ns() - cam.t_ns) / 1e6), cam)
                applied = False
                continue
            tof_raw, imu, veh, reason = assemble_inputs(cam.t_ns, list(tof_buf), list(imu_buf), list(veh_buf),
                                                        max_age, imu_age)
            if reason:
                hold(reason, cam)
                applied = False
                continue
            count_laps(veh)
            if last_select is not None and bool(veh['select']) != last_select:
                self._emit('handover', select=bool(veh['select']))       # CH3で操縦が切り替わった
            last_select = bool(veh['select'])
            if veh['select'] and not veh['valid'][2]:
                # 送信機の電源断・圏外: 基板はSELECTをAI側のまま保つので、人が取り戻せない。中立で止める
                if not rc_lost:
                    self._emit('rc_lost')
                rc_lost = True
                hold('AI側のままプロポの信号(CH3)が届いていません', cam, True)
                applied = False
                continue
            rc_lost = False
            mode_id = st['mode_id']
            prev_cmd = predicted_now(prev_pred, prev_cam_t, cam.t_ns, fps) if prev_pred is not None else None
            vehicle = vehicle_state(veh, prev_cmd, ref, applied)
            ok, enc = cv2.imencode('.jpg', cam.data['image'], jpeg)     # 学習データ(JPEG保存)と同じ画質にそろえる
            image = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else cam.data['image']
            img, tof, imu_n, state = normalize_inputs(image, tof_raw, imu, vehicle, use_command)
            if mode_id not in mode_cache:
                oh = np.zeros((1, num_modes), np.float32)
                oh[0, mode_id] = 1.0
                mode_cache[mode_id] = oh
            t0 = time.perf_counter()
            res = runner.run_numpy(image=img[None], tof=tof[None], imu=imu_n[None], state=state[None],
                                   mode=mode_cache[mode_id])
            pred = res['actions'][0] * PWM_SCALE + PWM_CENTER     # [10,2] µs
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if not np.all(np.isfinite(pred)):
                hold('推論結果が数値ではありません', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            k = min(max(p['lookahead'], 1), len(pred)) - 1
            raw = (float(pred[k, 0]), float(pred[k, 1]))
            model_us = apply_gains(raw, ref, p, lo, hi)
            # ToFガード: 衝突までの時間(TTC)が短ければ減速(・近い側から離すハンドル)。1 = 記録だけ
            g_mode, g_code, g_ttc, g_applied, base_us = int(p.get('tof_guard', 1)), 0, np.nan, False, model_us
            if guard is not None and g_mode >= 1:
                g_steer, g_thr, g_info = guard.step(model_us[0], model_us[1], tof_raw, float(veh['erpm']),
                                                    float(np.degrees(imu[-1, 5])))
                g_ttc = float(g_info['ttc'])
                g_code = {'slow': 1, 'brake': 2}.get(g_info['action'], 0)
                if g_code and g_mode >= 2 and veh['select'] and not dry_run:
                    base_us = (g_steer if g_mode >= 3 else model_us[0], g_thr)
                    g_applied = True
                if g_code and not last_guard:
                    stats['guard_brake' if g_code == 2 else 'guard_slow'] += 1
                last_guard = g_code
            # 補助: 後退の受け付け(中立を挟む)・ぶつかって止まったときの自動リカバリー
            recent = [x[1] for x in list(imu_buf)[-10:] if x[0] > cam.t_ns - 150_000_000]
            impact_g = max((float(np.hypot(a[0], a[1])) / 9.80665 for a in recent), default=0.0)
            out_us, kind = helper.step(base_us, float(veh['erpm']), bool(veh['select']) and not dry_run,
                                       time.monotonic(), esc_on=p.get('esc_reverse', 1) == 1,
                                       rec_on=p.get('recovery', 1) == 1,
                                       impact_g=impact_g, rev_us=float(p.get('rev_us', assist_mod.DEFAULT_REV_US)),
                                       rev_s=float(p.get('rev_s', assist_mod.DEFAULT_REV_S)))
            out_us = (float(np.clip(out_us[0], lo, hi)), float(np.clip(out_us[1], lo, hi)))
            if kind != last_kind:
                if kind == assist_mod.ASSIST_RECOVERY:
                    laps.note('recovery')
                    self._emit('recovery')
                elif kind == assist_mod.ASSIST_GIVEUP:
                    stats['giveups'] += 1
                    self._emit('recovery_giveup')
                last_kind = kind
            stats['recoveries'] = helper.recoveries
            if stats['frames'] % 15 == 0:
                st['laps'] = laps.summary()
            stats['esc_assists'] = helper.esc_count
            written = out.neutral() if dry_run else out.write(*out_us)
            stats['i2c_errors'] = out.errors
            if not written:
                hold('PCA9685への書き込みに失敗しました(読み直します)', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            # リカバリー中はモデルの出力を車に出していないので、次のフレームの「今の操作値」は中立にする
            prev_pred, prev_cam_t = pred, cam.t_ns
            applied = not dry_run and kind in (assist_mod.ASSIST_NONE, assist_mod.ASSIST_ESC)
            latency = (clock.now_ns() - cam.t_ns) / 1e6
            now_m = time.monotonic()
            times.append(now_m)
            lat_hist.append(latency)
            inf_hist.append(infer_ms)
            stats['frames'] += 1
            stats['fps'] = round((len(times) - 1) / max(times[-1] - times[0], 1e-6), 1) if len(times) > 1 else 0.0
            stats['latency_ms'] = round(float(np.mean(lat_hist)), 1)
            stats['latency_ms_max'] = round(float(np.max(lat_hist)), 1)
            stats['infer_ms'] = round(float(np.mean(inf_hist)), 2)
            st['hold_reason'] = None
            st['hold_since'] = None
            st['now'] = {'raw': [round(raw[0], 1), round(raw[1], 1)], 'out': [round(out_us[0], 1), round(out_us[1], 1)],
                         'select': bool(veh['select']), 'hold': False, 'erpm': int(veh['erpm']),
                         'human': [int(veh['pwm'][0]), int(veh['pwm'][1])] if veh['valid'][0] and veh['valid'][1] else None,
                         'pred': pred.round(1).tolist(), 'assist': assist_mod.ASSIST_NAMES[kind],
                         'ttc': None if not np.isfinite(g_ttc) else round(min(g_ttc, 9.9), 2),
                         'guard': GUARD_NAMES[g_code] + ('' if g_applied or not g_code else '(記録だけ)'),
                         'model_out': [round(model_us[0], 1), round(model_us[1], 1)]}
            self.history.append((now_m, raw[0], raw[1], out_us[0], out_us[1], bool(veh['select'])))
            self.recorder.add_ai(cam.t_ns, {
                'pred': pred.astype(np.float32), 'raw': raw, 'out': out_us,
                'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                'mode_id': mode_id, 'hold': False, 'select': bool(veh['select']), 'latency_ms': latency,
                'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': kind, 'model_out': model_us,
                'ttc': g_ttc, 'guard': g_code, 'guard_applied': g_applied})

    # ------------------------------------------------------------ 状態
    def status(self):
        st = dict(self.st)
        if st['phase'] in ('starting', 'driving') and not self.running:
            st['phase'] = 'error'
            st['error'] = st.get('error') or '走行スレッドが止まりました'
        st['running'] = self.running
        st['params'] = dict(self.params)
        st['last'] = dict(self.last)
        st['recording'] = self.recorder.recording
        st['elapsed_s'] = int((st.get('finished') or time.time()) - st['started']) if st.get('started') else 0
        st['heartbeat_age_s'] = round(time.monotonic() - self.heartbeat_t, 1) if self.running else None
        now = time.monotonic()
        h = list(self.history)[::3]
        st['history'] = {'t': [round(r[0] - now, 2) for r in h], 'raw_steer': [round(r[1]) for r in h],
                         'raw_throttle': [round(r[2]) for r in h], 'out_steer': [round(r[3]) for r in h],
                         'out_throttle': [round(r[4]) for r in h], 'select': [int(r[5]) for r in h]}
        st['limits'] = {k: list(v) for k, v in PARAM_LIMITS.items()}
        st['pca9685'] = {'address': config.PCA9685['address'], 'freq_hz': config.PCA9685['freq_hz'],
                         'steering_ch': config.PCA9685['steering_ch'], 'throttle_ch': config.PCA9685['throttle_ch']}
        return st
