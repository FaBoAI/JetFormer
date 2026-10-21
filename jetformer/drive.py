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

モデルの種類(方策、jetformer/policies)はどれも出力 actions [1,10,2] を持つので、先読み・ゲイン・ToFガード・補助・ラップ・
CH3 の交代・センサーの途切れで中立はそのまま効く。推論の直後・先読みの前に、方策の後処理(model_config.json の
postprocess。tcp の制御器)と予測の重ね合わせ(ensemble_j > 0)を通せる(policies.post.Stage。評価の道具と同じ手順)。
エンジンには基本の5つの入力を全部渡す(エンジンが使わない入力は読まれないだけ)。

区間の速度上限(safety_caps v1.1: R1 2.5・GM 2.2・DE 3.0 m/s。2026-10-21): パラメータ zone_caps(既定 0 = OFF、
起動のたびに OFF)。ON/OFF に関係なく、毎フレーム区間・上限・「上限ならこう出した」スロットル・ブレーキを計算して
AI の記録(AI_CMD の cap_*)に残す(影の判定)。車への出力を変えるのは zone_caps = 1 のときだけ。上限を掛けるのは
車に出す直前のスロットル(ゲイン・ToF ガード・補助のあと。上限は速さ [m/s] の決まりで、ゲインの前に掛けるとゲインで
上限の速さが変わるため。G1 の再生も記録の out = この値で判定した)。上限は前進を弱めるだけ(PWM は入力以上)。
計算が例外になったら、その走行のあいだは上限を外す(OFF と同じ出力)・記録とメッセージに残す。
v1.1 は位相が不確か(unknown: 最初の R1 の前・R1 から 16 秒超・1 秒以上止まったあと)でも 3.0 m/s で切る(4 つめの上限。
ユーザーの了承は 3 区間だけなので、画面の ON の確認で示す。ZONE_CAPS_SPEC['unknown_note'])。
走行中に OFF → ON にはできない(set_params が断る。1 走行 = 1 設定)。ON → OFF は走行中でもできる。

AI の時間(stats['ai_s']、2026-10-21): CH3 が AI 側で dry run でない時間の和。道具(label_events の dt_s)と同じ決め方で、
ベンチマークの「AI の時間が数える時間 + 1 分を過ぎたら止める」を画面で見るため。数えるだけで出力は変えない。
"""

import collections
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import assist as assist_mod
from . import clock, config, modes, policies
from . import safety_caps
from .laps import LapCounter
from .policies import post as post_mod

SETTINGS_PATH = config.DATA_DIR / 'drive_settings.json'
RUNS_DIR = config.DATA_DIR / 'runs'
ENGINE_FILE = 'jetformer_fp16.engine'
IMU_STEPS = 10                     # model.IMU_STEPS(torchを読み込まずに使うため複製)
PWM_CENTER, PWM_SCALE = 1500.0, 500.0
AI_GAP_S = 0.2                     # AI の時間: これより空いたフレームの間は 1 フレーム分だけ数える(tools/label_events.GAP_S と同じ)

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
    # 方策の後処理(tcp の制御器だけが読む)・予測の重ね合わせ(どの方策にも)
    'speed_cap': (0.0, 5.0),       # 速度の上限 [m/s](0 = なし)。tcp の制御器が前進を弱めるだけ(速くはしない)
    'speed_scale': (0.5, 1.5),     # 予測した目標速度に掛ける倍率(tcp_mode 1〜3 だけで効く)
    'tcp_mode': (0, 3),            # tcp の出し方 0=操作の枝だけ(既定・比較で評価した出力) 1=ハイブリッド 2=混ぜる 3=通過点を追う
    'ensemble_j': (0, 3),          # 予測の重ね合わせ: 何フレーム前の予測まで混ぜるか(0 = オフ)
    'ensemble_m': (0.0, 2.0),      # 予測の重ね合わせ: 古い予測を軽くする強さ(重み exp(-m·j))
    'zone_caps': (0, 1),           # 区間の速度上限 v1.1(R1 2.5・GM 2.2・DE 3.0 m/s)1 = 出力に掛ける 0 = 記録だけ(影の判定)
}
# 起動のたびに既定へ戻す項目(drive_settings.json から読まない)。区間の速度上限は ON のまま別の日に持ち越さない
RESET_ON_BOOT = ('tcp_mode', 'speed_scale', 'zone_caps')
ENGINE_CHECK_FILE = 'engine_check.json'          # エンジンの確認(精度・速さ)の結果。あれば走行タブに出す(精度の不合格は走らせない)
# 走行タブで最初に選ぶ「推奨のモデル」(方策の比較などの評価で決めたもの。tools/policy_zoo.py recommend で書く)。
# 無ければ今まで通り(前回のモデル → 一番新しいモデル)
RECOMMENDED_PATH = config.DATA_DIR / 'drive_recommended.json'
GUARD_NAMES = {0: '', 1: '減速', 2: 'ブレーキ'}
FLOOR_REF_PATH = config.DATA_DIR / 'tof_floor_ref.npy'      # ToFガードの床の基準(自由走行から作る)
DEFAULT_PARAMS = {'steer_gain': 1.0, 'throttle_gain': 0.5, 'steer_trim_us': 0.0, 'lookahead': 1, 'osc_percent': 0.0,
                  'esc_reverse': 1, 'recovery': 1, 'rev_us': assist_mod.DEFAULT_REV_US, 'rev_s': assist_mod.DEFAULT_REV_S,
                  'tof_guard': 1, 'speed_cap': 0.0, 'speed_scale': 1.0, 'tcp_mode': 0, 'ensemble_j': 0, 'ensemble_m': 0.5,
                  'zone_caps': 0}


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
    policy = policies.policy_of(cfg)            # 'policy' の無い古い学習結果は jetformer_v0
    ok, why = policies.drivable(cfg)
    check = engine_check_of(d)
    if ok and check and check.get('accuracy_ok') is False:      # 今のエンジンが精度の確認に落ちている → 走らせない
        ok, why = False, engine_gate_reason(check)
    return {
        'name': name, 'dataset': cfg.get('dataset'), 'created_at': run.get('created_at'),
        'state': res.get('state'), 'best_val_us': res.get('best_val_us'),
        'best_val_per_mode_us': res.get('best_val_per_mode_us', {}),
        'num_modes': cfg['num_modes'], 'modes': cfg.get('modes', []), 'trained_modes': trained,
        'steer_center_us': cfg.get('steer_center_us'), 'throttle_neutral_us': cfg.get('throttle_neutral_us'),
        'stride': cfg.get('stride'), 'future_steps': cfg.get('future_steps', 10),
        'command_input': cfg.get('command_input', True),
        'policy': policy, 'policy_title': policies.title_of(policy),
        'policy_cfg': cfg.get('policy_cfg') if 'policy' in cfg else dict(policies.V0_FROZEN_CFG),
        'inputs': policies.inputs_of(cfg), 'outputs': policies.outputs_of(cfg),
        'postprocess': cfg.get('postprocess'), 'drivable': ok, 'why_not': why,
        'engine_precision': (cfg.get('engine') or {}).get('precision', 'fp16'),
        'engine_check': check,
    }


def engine_check_of(run_dir):
    """<学習結果>/engine_check.json(エンジンの確認: 実フレームで TensorRT と PyTorch FP32 の差・速さ)→ 表示用の dict。
    無い・今のエンジンの確認でない(エンジンの大きさ・更新時刻が違う)なら None。
    精度の確認(accuracy_ok)に落ちたエンジンは model_info が走行できない(drivable False)にする。速さだけの不合格は警告だけ"""
    c = _read_json(Path(run_dir) / ENGINE_CHECK_FILE)
    if not isinstance(c, dict):
        return None
    try:
        stt = (Path(run_dir) / ENGINE_FILE).stat()
        eng = c.get('engine') or {}
        if int(eng.get('size', -1)) != stt.st_size or int(eng.get('mtime_ns', -1)) != stt.st_mtime_ns:
            return None
    except (OSError, TypeError, ValueError):
        return None
    keys = ('ok', 'accuracy_ok', 'speed_ok', 'actions_max_us', 'gate_us', 'run_numpy_median_ms', 'checked_at', 'note')
    return {k: c.get(k) for k in keys}


def engine_gate_reason(check):
    """精度の確認に落ちたエンジンの「走れない理由」(走行タブに出す)"""
    try:
        detail = '(TensorRT と PyTorch FP32 の差 最大 %.2f µs > %.1f µs)' % (float(check['actions_max_us']),
                                                                         float(check.get('gate_us') or 2.5))
    except (KeyError, TypeError, ValueError):
        detail = ''
    return ('エンジンが精度の確認に落ちています%s。エンジンを作り直して確認に合格するまで走行に使えません'
            '(学習タブで作り直す・tools/build_engine.py)' % detail)


def recommended():
    """推奨のモデル(data/drive_recommended.json)→ {'run', 'reason', 'source', 'decided_at'}。無い・壊れていれば None"""
    r = _read_json(RECOMMENDED_PATH)
    if not isinstance(r, dict) or not isinstance(r.get('run'), str) or not r['run']:
        return None
    return {k: r.get(k) for k in ('run', 'reason', 'source', 'decided_at')}


def save_recommended(run, reason='', source='', decided_at=None):
    """推奨のモデルを書く(走行タブが最初に選び、一覧の先頭に出す)。run が走行に使えなくても書く(画面では選べないまま)"""
    if not run or '/' in run or '..' in run:
        raise ValueError('学習結果の名前が正しくありません: %r' % run)
    rec = {'run': run, 'reason': reason or '', 'source': source or '',
           'decided_at': decided_at or time.strftime('%Y-%m-%dT%H:%M:%S')}
    RECOMMENDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(RECOMMENDED_PATH) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, RECOMMENDED_PATH)
    return rec


def list_models():
    """推奨のモデル(recommended())を先頭に、残りは新しい順(作成日時、無ければ名前)。
    推奨のモデルには recommended True と推奨の理由(recommended_reason)を付ける"""
    if not RUNS_DIR.exists():
        return []
    models = [m for m in (model_info(d.name) for d in RUNS_DIR.iterdir()) if m]
    models.sort(key=lambda m: (m['created_at'] or '', m['name']), reverse=True)
    rec = recommended()
    for m in models:
        m['recommended'] = bool(rec) and m['name'] == rec['run']
        if m['recommended']:
            m['recommended_reason'] = rec.get('reason') or ''
    return [m for m in models if m['recommended']] + [m for m in models if not m['recommended']]


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


def _num(x, nd=2):
    """表示用の数(数値でなければ None。JSON に NaN を入れない)"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def post_view(pinfo, v_now):
    """後処理の情報 → 画面の表示用(st['now']['post'])"""
    if not pinfo:
        return None
    return {'mode': int(pinfo.get('mode', -1)), 'branch': str(pinfo.get('branch', '')),
            'v_hat': _num(pinfo.get('v_hat')), 'v_cmd': _num(pinfo.get('v_cmd')), 'v_now': _num(v_now),
            'v_curv': _num(pinfo.get('v_curv')), 'cap': _num(pinfo.get('cap'))}


def uses_post(p):
    """走行のパラメータが後処理(tcp の制御器)を要るか: 速度の上限 > 0 または tcp の出し方 ≥ 1"""
    return float(p.get('speed_cap', 0) or 0) > 0 or int(p.get('tcp_mode', 0) or 0) >= 1


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


# 区間の速度上限の説明(画面・道具の表示用。値は走行ループが使う safety_caps.ZONE_CAPS_V1_1)
# v1.1 は 3 区間(R1・GM・DE)のほかに「位相が不確か(unknown)」でも上限を掛ける(4 つめ)。ユーザーの了承(2026-10-21)は
# 3 区間だけなので、画面の ON の確認でこれも示す(Fix-drive。値は変えない: G1 はこの形で判定した)
_ZC_CAP, _ZC_PHASE = safety_caps.ZONE_CAPS_V1_1.cap, safety_caps.ZONE_CAPS_V1_1.phase
ZONE_CAPS_SPEC = {'version': safety_caps.ZONE_CAPS_VERSION, 'zones': list(safety_caps.ZONES),
                  'caps_mps': {'R1': _ZC_CAP.cap_R1_app, 'GM': _ZC_CAP.cap_GM,
                               'DE': _ZC_CAP.cap_DE, 'unknown': _ZC_CAP.cap_unknown},
                  'approved_zones': ['R1', 'GM', 'DE'],      # ユーザーが了承した区間(2026-10-21)。unknown は含まない
                  'unknown_rule': {'cap_mps': _ZC_CAP.cap_unknown, 'lost_s': _ZC_PHASE.lost_s,
                                   'stop_reset_s': _ZC_PHASE.stop_reset_s},
                  'unknown_note': ('4 つめの上限: 位相が不確か(unknown)のときも %.1f m/s で切ります。走行の始めから最初の R1 を'
                                   '見つけるまで(1 周目のほぼ全部)・R1 から %g 秒を超えたとき・%g 秒以上止まったあと。'
                                   'ユーザーの了承は 3 区間(R1・GM・DE)だけです。ON の了承をもらうときに、これも含めて確かめてください'
                                   % (_ZC_CAP.cap_unknown, _ZC_PHASE.lost_s, _ZC_PHASE.stop_reset_s)
                                   if np.isfinite(_ZC_CAP.cap_unknown) else ''),
                  'brake_zones': list(_ZC_CAP.brake_zones), 'brake_us': _ZC_CAP.brake_us,
                  'modes': {'-2': '計算の失敗', '-1': '動かしていない', '0': 'そのまま', '1': '上限で切った',
                            '2': '中立にした', '3': 'ブレーキ'}}
ZONE_CODES = {z: i for i, z in enumerate(safety_caps.ZONES)}       # 記録の cap_zone(-1 = 計算していない・失敗)
CAP_MODE_CODES = dict({m: i for i, m in enumerate(safety_caps.CAP_MODES)}, off=-1)   # 記録の cap_mode(-2 = 失敗)


class _ZoneCaps(object):
    """区間の速度上限(safety_caps v1.1)の走行ループ側のつなぎ: センサーのバッファの値を数にして、純粋な包み
    (safety_caps.zone_imu・zone_frame)へ渡すだけ。状態は ZoneState 1 つ。計算が例外になったら failed に理由を残し、
    その走行のあいだは何もしない(呼ぶ側は出力を変えない = OFF と同じ)。時刻は最初のサンプルを原点にした秒"""

    def __init__(self, cfg=safety_caps.ZONE_CAPS_V1_1):
        self.cfg = cfg
        self.state = safety_caps.ZoneState()
        self.imu_done = None            # 位相へ入れた最後の IMU の時刻 [ns]
        self.origin = None
        self.failed = None              # 失敗の説明(None = 動いている)
        self.ms = collections.deque(maxlen=30)
        self.ms_max = 0.0

    def _sec(self, t_ns):
        if self.origin is None:
            self.origin = int(t_ns)
        return (int(t_ns) - self.origin) / 1e9

    @staticmethod
    def _speed_at(veh, t_ns):
        """t_ns までの最新の車両の速さ [m/s](無ければ最初の値。G1 の再生の prev_idx と同じ)"""
        if not veh:
            return 0.0
        e = veh[0][1]['erpm']
        for tv, d in reversed(veh):
            if tv <= t_ns:
                e = d['erpm']
                break
        return -float(e) * safety_caps.KV

    def _feed(self, imu_buf, veh_buf, upto_ns):
        """前回より後・upto_ns 以前(None なら全部)の IMU のサンプルを位相へ"""
        veh = list(veh_buf)
        st = self.state
        for t_ns, x in list(imu_buf):
            if self.imu_done is not None and t_ns <= self.imu_done:
                continue
            if upto_ns is not None and t_ns > upto_ns:
                break
            st = safety_caps.zone_imu(st, self._sec(t_ns), float(np.degrees(x[5])), self._speed_at(veh, t_ns), self.cfg)
            self.imu_done = t_ns
        self.state = st

    def _fail(self, e):
        if not self.failed:             # 最初の失敗を残す
            traceback.print_exc()
            self.failed = '%s: %s' % (type(e).__name__, e)

    def frame(self, cam_t_ns, out_us, veh, pred, active, imu_buf, veh_buf):
        """AI のフレーム(推論して出力を決めたフレーム)→ ZoneOut(失敗していれば None)。出力は変えない"""
        if self.failed:
            return None
        t0 = time.perf_counter()
        try:
            self._feed(imu_buf, veh_buf, cam_t_ns)
            veh_t = None
            for tv, d in reversed(list(veh_buf)):
                if d is veh:
                    veh_t = tv
                    break
            fresh = veh_t is not None and self._sec(cam_t_ns) - self._sec(veh_t) < self.cfg.fresh_s
            v = -float(veh['erpm']) * safety_caps.KV
            thr_in = float(out_us[1])
            self.state, z = safety_caps.zone_frame(self.state, self._sec(cam_t_ns), thr_in, float(out_us[0]), v,
                                                   np.asarray(pred)[:, 0].tolist(), bool(active) and fresh, self.cfg)
            if not (np.isfinite(z.thr_us) and z.thr_us >= thr_in):       # 前進を強める値は出さない(起きないはず)
                raise ValueError('上限の出力 %r が入力 %r より前進側です' % (z.thr_us, thr_in))
        except Exception as e:      # noqa: BLE001  上限が無くても走れる(OFF と同じ)
            self._fail(e)
            return None
        ms = (time.perf_counter() - t0) * 1000.0
        self.ms.append(ms)
        self.ms_max = max(self.ms_max, ms)
        return z

    def hold(self, cam_t_ns, imu_buf, veh_buf):
        """中立で止めたフレーム: 位相は進め、層の状態だけ戻す"""
        if self.failed:
            return
        try:
            self._feed(imu_buf, veh_buf, cam_t_ns)
            self.state = safety_caps.zone_inactive(self.state)
        except Exception as e:      # noqa: BLE001
            self._fail(e)

    def record(self, z, on, applied, thr_in):
        """AI の記録(AI_CMD)の cap_* の欄。z = frame() の戻り値(None = 中立で止めた / 失敗)。例外は出さない"""
        if z is not None:
            try:
                return {'cap_on': bool(on), 'cap_zone': ZONE_CODES.get(z.zone, -1),
                        'cap_zone_gyro': ZONE_CODES.get(z.zone_gyro, -1), 'cap_pred_sig': bool(z.pred_signal),
                        'cap_rel_psi': np.nan if z.rel_psi is None else float(z.rel_psi),
                        'cap_mode': CAP_MODE_CODES.get(z.mode, -1), 'cap_limit': float(z.cap_eff), 'cap_in_us': thr_in,
                        'cap_thr_us': float(z.thr_us), 'cap_brake_start': bool(z.brake_start), 'cap_applied': bool(applied)}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'cap_on': bool(on), 'cap_zone': -1, 'cap_zone_gyro': -1, 'cap_pred_sig': False, 'cap_rel_psi': np.nan,
                'cap_mode': -2 if self.failed else -1, 'cap_limit': np.nan, 'cap_in_us': thr_in, 'cap_thr_us': np.nan,
                'cap_brake_start': False, 'cap_applied': False}

    def view(self, z, on, applied):
        """画面の表示用(st['now']['zone'])。例外は出さない"""
        if z is not None:
            try:
                return {'on': bool(on), 'error': None, 'zone': str(z.zone), 'zone_gyro': str(z.zone_gyro), 'mode': str(z.mode),
                        'limit': _num(z.cap_eff), 'thr_us': _num(z.thr_us, 1), 'applied': bool(applied),
                        'brake': z.mode == 'brake'}
            except Exception as e:      # noqa: BLE001
                self._fail(e)
        return {'on': bool(on), 'error': self.failed, 'zone': None, 'zone_gyro': None, 'mode': None, 'limit': None,
                'thr_us': None, 'applied': False, 'brake': False}


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
        # tcp の出し方・目標速度の倍率は、次に起動したときに持ち越さない(試験的な出し方 2・3 が別の日・別のモデルで
        # そのまま使われないように。速度の上限は前進を弱めるだけなので残す)。レビュー 2026-10-12
        for k in RESET_ON_BOOT:
            self.params[k] = DEFAULT_PARAMS[k]
        self.last = {'run': saved.get('run'), 'mode_id': saved.get('mode_id')}
        self.heartbeat_t = time.monotonic()
        self.hb_token = None
        self.hb_count = 0
        self.history = collections.deque(maxlen=300)
        self.st = {'phase': 'idle', 'message': '停止中', 'error': None}
        self.listeners = []        # 出来事の通知先 cb(event, **info)(音声案内など)。走行ループから呼ぶので待たせないこと
        self.last_takeover_ns = None
        self.takeover_reasons = []

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
        for k in ('esc_reverse', 'recovery', 'zone_caps'):     # 古い設定ファイル・古い画面には無い項目
            p[k] = 1 if float(p.get(k, DEFAULT_PARAMS[k])) >= 0.5 else 0
        p['tof_guard'] = int(round(float(p.get('tof_guard', DEFAULT_PARAMS['tof_guard']))))
        for k in ('tcp_mode', 'ensemble_j'):           # 整数の項目(古い設定ファイルには無い)
            p[k] = int(round(float(p.get(k, DEFAULT_PARAMS[k]))))
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
            if self.running and self.st.get('post_failed') and uses_post(p) and not uses_post(self.params):
                raise ValueError('後処理を読み込めなかったため、速度の上限・TCP の出し方は使えません(操作の枝だけで走っています)')
            if self.running and p['zone_caps'] == 1 and int(self.params.get('zone_caps', 0) or 0) == 0:
                # 区間の速度上限は 1 走行 = 1 設定(画面と同じ決まりをサーバーでも守る)。OFF にするのは走行中でもできる
                raise ValueError('走行中は区間の速度上限を ON にできません。止めてから ON にしてください(1 走行 = 1 設定)')
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
            if not info.get('drivable', True):
                raise ValueError('このモデルでは走行できません: %s' % info.get('why_not'))
            mode_id = int(mode_id)
            if mode_id not in info['trained_modes']:
                raise ValueError('このモデルはジャンル %d(%s)を学習していません' % (mode_id, modes.name_of(mode_id)))
            new_params = self._clean_params(params or {})
            if info.get('postprocess') and uses_post(new_params):
                ref0 = (float(info.get('steer_center_us') or PWM_CENTER), float(info.get('throttle_neutral_us') or PWM_CENTER))
                obj, note = post_mod.make(info, ref0)
                if obj is None:
                    raise ValueError('%s。速度の上限・TCP の出し方は使えません(上限を 0、出し方を「操作の枝だけ」にすると走れます)' % note)
            self.params = new_params
            self.last = {'run': run, 'mode_id': mode_id}
            self._save_settings()
            self._stop.clear()
            self.history.clear()
            self.hb_token = uuid.uuid4().hex
            self.heartbeat_t = time.monotonic()
            self.hb_count = 0
            self.last_takeover_ns = None
            self.takeover_reasons = []
            self.st = {'phase': 'starting', 'message': 'モデルとセンサーを準備中…', 'error': None, 'run': run,
                       'model': info, 'mode_id': mode_id, 'dry_run': bool(dry_run), 'record': bool(record),
                       'recording_note': '', 'ref': None, 'started': time.time(), 'stop_reason': '',
                       'stats': {}, 'now': None}
            self.thread = threading.Thread(target=self._run, args=(info, bool(record), description or '',
                                                                   list(tags or []), bool(dry_run)),
                                           name='drive', daemon=True)
            self.thread.start()
            return run, self.hb_token

    TAKEOVER_REASONS = {'collision': '衝突した', 'stuck': '止まった', 'preemptive': '危なかった(予防)', 'other': 'その他・終了'}

    def takeover_reason(self, reason):
        """人が交代した理由を記録する(ロードマップ S3)。直前(60秒以内)の交代の時刻と結びつける"""
        if reason not in self.TAKEOVER_REASONS:
            raise ValueError('理由は %s のどれか' % ', '.join(self.TAKEOVER_REASONS))
        now = clock.now_ns()
        tk = self.last_takeover_ns if self.last_takeover_ns and now - self.last_takeover_ns < 60e9 else None
        ev = {'reason': reason, 'label': self.TAKEOVER_REASONS[reason],
              'takeover_us': int(tk // 1000) if tk else None,
              'since_takeover_s': round((now - tk) / 1e9, 2) if tk else None, 'run': self.st.get('run')}
        ev['recorded'] = self.recorder.add_event('takeover_reason', ev)
        self.takeover_reasons.append(ev)
        self.st['takeover_reasons'] = self.takeover_reasons[-20:]
        return ev

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
            # エンジンの入出力が走行の約束(基本の5つの入力・actions [1,10,2])に合うか、走る前に確かめる
            ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                        {n: list(runner.shapes[n]) for n in runner.outputs}, num_modes)
            if not ok:
                raise RuntimeError('エンジンの入出力が走行に合いません: %s' % why)
            zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}     # エンジンの形から作る
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
                pp = info.get('postprocess') or None
                drive_info = {'run': info['name'], 'dataset': info['dataset'], 'dry_run': dry_run,
                              'params': dict(self.params), 'steer_center_us': ref[0], 'throttle_neutral_us': ref[1],
                              'policy': info.get('policy', policies.DEFAULT), 'policy_cfg': info.get('policy_cfg'),
                              'postprocess': ({'kind': pp.get('kind'), 'version': pp.get('version')}
                                              if isinstance(pp, dict) else None)}
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
        # 方策の後処理(tcp の制御器)と予測の重ね合わせ。後処理の無い方策・ensemble_j = 0 なら actions のまま
        stage = post_mod.Stage(st.get('model') or {}, ref, fps=fps)
        if stage.note:
            st['recording_note'] = ((st.get('recording_note') or '') + ' ' + stage.note).strip()
        st['post_kind'] = post_mod.kind_of(st.get('model') or {}) if stage.post is not None else None
        st['post_failed'] = stage.failed
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
        zcaps = _ZoneCaps()                 # 区間の速度上限(OFF でも影の判定を記録する)
        st['zone_caps_error'] = None
        stats.update(zone_ms=0.0, zone_ms_max=0.0, zone_brakes=0, zone_applied=0)
        # AI の時間(画面の「AI の時間」。ベンチマークの「数える 240 秒 + 1 分」で止める目安。2026-10-21 Fix-all)
        stats['ai_s'] = 0.0
        ai_sum, ai_t_prev, ai_was = 0.0, None, False

        def zone_failed():
            """上限の計算が例外になった(この走行のあいだは上限を外す)ことを、1 回だけ記録とメッセージに残す"""
            if not zcaps.failed or st.get('zone_caps_error'):
                return
            st['zone_caps_error'] = zcaps.failed
            zc_on = int(self.params.get('zone_caps', 0) or 0)
            if zc_on == 1:
                note = '区間の速度上限の計算が失敗したため、この走行では上限を外しました(%s)' % zcaps.failed
            else:                   # OFF の日: 出力はもとから変えていない。止まったのは記録だけの影の判定
                note = ('区間の速度上限(OFF・記録だけ)の影の判定が計算の失敗で止まりました。車への出力は変わりません(%s)'
                        % zcaps.failed)
            st['recording_note'] = ((st.get('recording_note') or '') + ' ' + note).strip()
            st['message'] = (st.get('message') or '') + '・' + note
            try:
                self.recorder.add_event('zone_caps_error', {'error': zcaps.failed, 'run': st.get('run'),
                                                            'zone_caps': zc_on})
            except Exception:       # noqa: BLE001  記録できなくても走行は続ける
                traceback.print_exc()

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
            zcaps.hold(cam.t_ns if cam is not None else None, imu_buf, veh_buf)
            zone_failed()
            if cam is not None:
                self.recorder.add_ai(cam.t_ns, {
                    'pred': np.full((10, 2), np.nan, np.float32), 'raw': (np.nan, np.nan), 'out': (n_s, n_t),
                    'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                    'mode_id': st['mode_id'], 'hold': True, 'select': veh_select, 'latency_ms': np.nan,
                    'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': assist_mod.ASSIST_NONE,
                    'model_out': (np.nan, np.nan), 'ttc': np.nan, 'guard': 0, 'guard_applied': False,
                    'model_pred': np.full((10, 2), np.nan, np.float32), 'post_mode': -1, 'v_hat': np.nan,
                    'v_cmd': np.nan, 'ens_j': int(p.get('ensemble_j', 0)),
                    **zcaps.record(None, int(p.get('zone_caps', 0) or 0) == 1, False, n_t)})
            helper.reset()                  # 中立で止めた後は、補助の手順を最初から
            stage.reset()                   # 後処理・重ね合わせも最初から(止めていた間の予測を混ぜない)

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
            # AI の時間: 道具(tools/label_events の dt_s・ai)と同じ決め方。前のフレームが AI 側(CH3 が AI・dry run でない)なら、
            # そのフレームから今のフレームまでの時間を足す(0.2 秒を超えて空いたら 1 フレーム分)。数えるだけで出力は変えない
            if ai_t_prev is not None and ai_was:
                gap_s = (cam.t_ns - ai_t_prev) / 1e9
                ai_sum += gap_s if 0.0 <= gap_s <= AI_GAP_S else 1.0 / fps
                stats['ai_s'] = round(ai_sum, 1)
            ai_t_prev = cam.t_ns
            ai_was = bool(veh_buf) and not dry_run and bool(veh_buf[-1][1]['select'])
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
                if not veh['select']:
                    self.last_takeover_ns = cam.t_ns                        # 人が取り返した(交代の理由のボタンと結びつける)
                    st['takeovers'] = st.get('takeovers', 0) + 1
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
            # 基本の5つを必ず渡す(エンジンが使わない入力は読まれない)
            feed = {'image': img[None], 'tof': tof[None], 'imu': imu_n[None], 'state': state[None],
                    'mode': mode_cache[mode_id]}
            res = runner.run_numpy(**feed)
            model_pred = stage.model_pred(res)     # [10,2] µs(後処理・重ね合わせの前)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            if not np.all(np.isfinite(model_pred)):
                hold('推論結果が数値ではありません', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            # 後処理(tcp の制御器。無い方策はそのまま)→ 予測の重ね合わせ(ensemble_j > 0 のとき)
            v_now = post_mod.speed_of(veh['erpm'])
            try:
                pred, pinfo = stage.apply(res, model_pred, v_now, p, cam.t_ns)
            except post_mod.PostError as e:
                hold(str(e), cam, bool(veh['select']), infer_ms)          # 推論の異常と同じく中立
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
            # 区間の速度上限: 車に出す直前のスロットルに(ゲイン・ToF ガード・補助のあと)。OFF でも計算して記録する(影の判定)。
            # 出力を変えるのは zone_caps = 1 で、AI 側のフレームで、上限が前進を弱めたときだけ(計算の失敗後は外す)
            zone_on = int(p.get('zone_caps', 0) or 0) == 1
            cap_in = out_us[1]
            zout = zcaps.frame(cam.t_ns, out_us, veh, pred, bool(veh['select']) and not dry_run, imu_buf, veh_buf)
            zone_failed()
            zone_applied = False
            if zone_on and zout is not None and zout.active and zout.thr_us != out_us[1]:
                out_us = (out_us[0], float(np.clip(zout.thr_us, lo, hi)))
                zone_applied = True
            if zcaps.ms:
                stats['zone_ms'] = round(float(np.mean(zcaps.ms)), 3)
                stats['zone_ms_max'] = round(zcaps.ms_max, 3)
            written = out.neutral() if dry_run else out.write(*out_us)
            stats['i2c_errors'] = out.errors
            if not written:
                hold('PCA9685への書き込みに失敗しました(読み直します)', cam, bool(veh['select']), infer_ms)
                applied = False
                continue
            # 画面の数(出力を変えたフレーム・ブレーキの始まり)は、書けたフレームだけ数える(記録の cap_applied・
            # cap_brake_start と同じ。書けなかったフレームは中立で止めた行として記録される)
            if zone_applied:
                stats['zone_applied'] += 1
            if zout is not None and zout.brake_start:
                stats['zone_brakes'] += 1
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
                         'model_out': [round(model_us[0], 1), round(model_us[1], 1)],
                         'post': post_view(pinfo, v_now), 'ens_j': int(p.get('ensemble_j', 0)),
                         'zone': zcaps.view(zout, zone_on, zone_applied)}
            self.history.append((now_m, raw[0], raw[1], out_us[0], out_us[1], bool(veh['select'])))
            self.recorder.add_ai(cam.t_ns, {
                'pred': pred.astype(np.float32), 'raw': raw, 'out': out_us,
                'gains': (p['steer_gain'], p['throttle_gain']), 'trim': p['steer_trim_us'], 'lookahead': p['lookahead'],
                'mode_id': mode_id, 'hold': False, 'select': bool(veh['select']), 'latency_ms': latency,
                'infer_ms': infer_ms, 'dry_run': dry_run, 'assist': kind, 'model_out': model_us,
                'ttc': g_ttc, 'guard': g_code, 'guard_applied': g_applied,
                'model_pred': np.asarray(model_pred, np.float32), 'post_mode': int(pinfo.get('mode', -1)) if pinfo else -1,
                'v_hat': float(pinfo.get('v_hat', np.nan)) if pinfo else np.nan,
                'v_cmd': float(pinfo.get('v_cmd', np.nan)) if pinfo else np.nan,
                'ens_j': int(p.get('ensemble_j', 0)), **zcaps.record(zout, zone_on, zone_applied, cap_in)})

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
        st['zone_caps_spec'] = ZONE_CAPS_SPEC
        st['pca9685'] = {'address': config.PCA9685['address'], 'freq_hz': config.PCA9685['freq_hz'],
                         'steering_ch': config.PCA9685['steering_ch'], 'throttle_ch': config.PCA9685['throttle_ch']}
        return st
