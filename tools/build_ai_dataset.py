#!/usr/bin/env python3
"""AI 側のフレームも学習・評価に使えるデータセットを作る(ロードマップ S8-2)

    python tools/build_ai_dataset.py --name ai_mix_20261011
        既定: 人のきれいな走行(clean_driving_20261011 と同じシーン・同じ検証シーン)
              + AI 走行のログ全部(data/labels/ で AI のフレームがあるログ)。AI のログ 172505・204208 は丸ごと検証用
    python tools/build_ai_dataset.py --name ai_only --human-from '' --ai-logs 20261010-173445 20261010-181504
    python tools/build_ai_dataset.py --name x --label-gains recorded      # 記録時のゲインで割り戻す(= raw。モデル自身の出力)

形式は jetformer/dataset.py の build_dataset と同じ(nuScenes の表・samples・aligned/*.npz・splits.json・dataset_card.json)。
build_dataset でそのまま作ってから、aligned/*.npz の AI 側のフレームを、ラベル(tools/label_events.py が書いた
data/labels/<ログ>.npz。無ければその場で作る)で書き換え、フレームごとの配列を足す。
JetFormerDataset・tools/compare_models.py・tools/evaluate.py・hf_export はそのまま読める(足した配列は無視される)。
GPU は使わない(numpy と、build_dataset の cv2 だけ)。

AI 側のフレーム(CH3 が AI 側。label_events の ai):
  future_actions[i, k]  撮像から k+1 フレーム目(k=0..9)に車に出ていた操作 = AI_CMD の続くフレーム i+k の out を
                        「評価時のゲイン」(既定 D3: ステア 1.3・速度 1.0・トリム 0。速度のゲインは前進側だけ = 今の drive.py)で
                        割り戻した値(ゲイン前の µs)。このモデルを評価時のゲインで走らせると、ちょうど out が出る(ロードマップ 3.1 の2)。
                        out(撮像の約30ms後に出す)は次の撮像の頃まで車に出ているので、k=0 が「このフレームで決めた操作」になる。
                        人のフレームの「k+1 フレーム後のスティック」と同じ意味。フレームを取りこぼしたところは前の out が続いていたとする
                        (撮像から k+0.5 フレーム以内の最新の AI_CMD)
  future_valid          このフレーム〜9フレーム先がすべて label_valid で、同じ区間(segment_id: 操作者が変わる・マスク・0.2秒超の
                        途切れで切れる)にあり、使う out が 100ms より古くない
  vehicle[:, :2]        今の操作値 = 1フレーム前に出した out を同じゲインで割り戻した値(人のフレームではスティック。ロードマップ 3.1 の2
                        「1フレーム前に実際に出した値」)。走行中の drive.py は「前のフレームの予測の今の時刻の値」を入れているので、
                        操作値を入力するモデルではここが少し違う(結果の needed_changes を参照)
  vehicle[:, 2:]        CH3 の µs・eRPM は、AI 側でも車両データの最新の値(build_dataset は人の側のサンプルしか見ないため)
  vehicle_valid         車両データが新しく(150ms 以内)、1フレーム前(0.2秒以内)のラベルがある
  mode                  AI_CMD の mode_id(走ったときのジャンル)
人のフレーム: 値は build_dataset のまま(スティック)。future_valid に「現在〜10フレーム先の時間のフレームが、すべて人で
  label_valid」も掛ける(基板の再起動の前後 0.5秒、交代・戻しの前後 0.3秒、CH3 の揺れを外す。カメラのフレームの抜けでは切らない)。AI のログの中の人のフレームは、交代の後の後退・
  置き直しが多く、介入の検証セット(tools/make_intervention_set.py)とも重なるので、既定では future_valid=False
  (--keep-ai-log-human で残す)

aligned/*.npz に足す配列(フレームごと。ラベルの値は data/labels/<ログ>.npz の同じ時刻のフレームから):
  source         int8   0=人・1=AI・-1=ラベルなし
  label_valid    bool   このフレームの行動ラベルが使えるか(label_events の valid_action)
  action_label   [N,2]  このフレームの行動ラベル(AI: out を割り戻した値、人: スティック)。future_actions[:, 0] と同じ規則
  action_exec    [N,2]  実際に車に出た値(AI: out、人: スティック)/ action_sat [N,2] out が上下限に張り付いた
  reward         float32 r = 前進距離 − 1·衝突 − 2·失敗による交代 − 0.5·リカバリー開始 [m](ロードマップ 4.3。基板の再起動などで
                 区間の外に落ちた罰は、直前の AI の区間の最後のフレームへ移してある: label_events.carry_penalties)
  mc_return      float32 γ=0.99 のモンテカルロのリターン。区間の終わりで打ち切り(区間外は NaN)
  segment_id / segment_end / end_code / time_to_end_s   リターンの区間(ログの中で通し番号。end_code は label_events.END_CODES)
  collision / near_collision / failure_takeover / loc_code / progress_m / label_index(data/labels/<ログ>.npz の行)
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config, dataset  # noqa: E402
import label_events as le  # noqa: E402

TOOL = 'tools/build_ai_dataset.py'
STEPS = dataset.FUTURE_STEPS
FRAME_US = 1e6 / config.CAMERA['fps']
MAX_AGE_US = dataset.VEHICLE_MAX_AGE_MS * 1000      # 未来の操作に使う out・スティックの古さの上限(build_dataset と同じ)
PREV_MAX_US = int(le.GAP_S * 1e6)                    # 「1フレーム前の操作」として使える間隔
SENSORS = ('CAM_FRONT', 'TOF_FRONT', 'IMU', 'VEHICLE')
D3 = {'steer_gain': 1.3, 'throttle_gain': 1.0, 'steer_trim_us': 0.0}     # ロードマップ 0 章 D3(比べるときの既定の設定)
HUMAN_FROM = 'clean_driving_20261011'
VAL_AI_LOGS = ('20261010-172505', '20261010-204208')   # ai_run_004(ゲイン 1.5/0.9)・ai_run_010(1.2/1.4)を丸ごと検証用に

EXTRA_KEYS = {
    'source': '[N] int8 0=人・1=AI・-1=ラベルなし',
    'label_valid': '[N] このフレームの行動ラベルが使えるか(label_events の valid_action)',
    'action_label': '[N,2] このフレームの行動ラベル µs(AI: out を割り戻した値、人: スティック)',
    'action_exec': '[N,2] 実際に車に出た値 µs(AI: out、人: スティック)',
    'action_sat': '[N,2] out が上下限(1000/2000µs)に張り付いた',
    'reward': '[N] r = 前進距離 − 1·衝突 − 2·失敗による交代 − 0.5·リカバリー開始 [m](区間の外に落ちた罰は直前の AI の区間の'
              '最後のフレームへ移したもの。label_events.carry_penalties)',
    'mc_return': '[N] γ=0.99 のモンテカルロのリターン(区間の終わりで打ち切り。区間外は NaN)',
    'segment_id': '[N] リターンの区間の番号(ログの中で通し。-1 = 区間外)',
    'segment_end': '[N] 区間の最後のフレーム',
    'end_code': '[N] 区間の終わり方(1=失敗による交代 2=運用上の交代 3=戻し 4=ログの終わり 5=打ち切り)',
    'time_to_end_s': '[N] 区間の終わりまでの秒',
    'collision': '[N] 衝突(水平 2.5G 以上)のフレーム',
    'near_collision': '[N] 衝突の 1秒前〜2秒後',
    'failure_takeover': '[N] 失敗による交代の直前の AI のフレーム',
    'loc_code': '[N] 場所(label_events.LOC_NAMES: unknown/R1/GM/DE/other/HS/LH)',
    'progress_m': '[N] このフレームの間に進んだ距離 [m](eRPM の積分。後退は負)',
    'label_index': '[N] data/labels/<ログ>.npz の行(-1 = 無し)',
}


# ---------------------------------------------------------------- ラベル
def labels_dir():
    return config.DATA_DIR / 'labels'


def load_labels(log, labels_root=None, raw_dir=None):
    """data/labels/<ログ>.json・.npz を読む。無ければ label_events.label_log でその場で作る(書き出さない)。
    戻り値 (info, arrays)"""
    d = Path(labels_root or labels_dir())
    fj, fz = d / (log + '.json'), d / (log + '.npz')
    if fj.exists() and fz.exists():
        with open(fj, encoding='utf-8') as f:
            info = json.load(f)
        with np.load(fz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return info, arrays
    info, arrays = le.label_log(log, raw_dir)
    if arrays is None:
        raise ValueError('%s: %s' % (log, info.get('error', 'ラベルを作れません')))
    return info, arrays


def eval_actions(out, ref, gains=D3):
    """出した値 out [n,2] → 評価時のゲイン gains で割り戻した値(drive.apply_gains の逆。速度のゲインは前進側だけ)"""
    out = np.asarray(out, np.float64)
    n = len(out)
    g = np.tile([float(gains['steer_gain']), float(gains['throttle_gain'])], (n, 1))
    return le.invert_gains(out, g, np.full(n, float(gains.get('steer_trim_us', 0.0))), ref, forward_only=True)


def lookahead(t_us, idx, steps, frame_us=FRAME_US, max_age_us=MAX_AGE_US):
    """フレーム idx ごとに、撮像から (k+0.5) フレーム以内の最新のフレーム j[m, k](k=0..steps-1)と、
    k+1 フレーム目の時刻にその値が max_age_us より古くなく、後ろに次のフレームがあるか ok[m, k]"""
    t_us = np.asarray(t_us, np.int64)
    idx = np.asarray(idx, np.int64)
    t = t_us[idx]
    j = np.zeros((len(idx), steps), np.int64)
    ok = np.zeros((len(idx), steps), bool)
    for k in range(steps):
        q = t + int(round((k + 0.5) * frame_us))
        jk = np.searchsorted(t_us, q, side='right') - 1
        jc = np.clip(jk, 0, len(t_us) - 1)
        j[:, k] = jc
        # その値が次のフレームまで出ていたと分かること(ログの最後のフレームの先は分からない)と、古くないこと
        ok[:, k] = (jk >= 0) & (jk + 1 < len(t_us)) & (t + int(round((k + 1) * frame_us)) - t_us[jc] <= max_age_us)
    return j, ok


def horizon(t_us, label, valid, seg, steps, idx=None):
    """未来の行動 [m,steps,2] と、使えるか [m](全部 valid・同じ区間・古くない・数値)。idx を省くと全フレーム"""
    n = len(t_us)
    idx = np.arange(n) if idx is None else np.asarray(idx, np.int64)
    if not n or not len(idx):
        return np.zeros((len(idx), steps, 2), np.float32), np.zeros(len(idx), bool)
    j, ok = lookahead(t_us, idx, steps)
    seg = np.asarray(seg)
    fut = np.asarray(label, np.float64)[j]
    good = ok.all(axis=1) & np.asarray(valid, bool)[j].all(axis=1) & (seg[j] == seg[idx][:, None]).all(axis=1) \
        & (seg[idx] >= 0) & np.isfinite(fut).all(axis=(1, 2))
    return np.nan_to_num(fut).astype(np.float32), good


def human_horizon_ok(t_us, ok, steps=STEPS + 1, frame_us=FRAME_US):
    """人のフレーム用: 撮像から (steps−0.5) フレーム先までの時間にあるフレームが、すべて ok か [n]。
    人の正解(スティック)は車両データ(100Hz)から作るので、カメラのフレームの抜け・途切れでは切らない
    (スティックが古い・続かないことは build_dataset の future_valid が見ている)"""
    t_us = np.asarray(t_us, np.int64)
    ok = np.asarray(ok, bool)
    if not len(t_us):
        return np.zeros(0, bool)
    bad_cum = np.concatenate([[0], np.cumsum(~ok)])
    j_end = np.searchsorted(t_us, t_us + int(round((steps - 0.5) * frame_us)), side='right')
    return (bad_cum[j_end] - bad_cum[np.arange(len(t_us))]) == 0


def previous_command(t_us, label, max_gap_us=PREV_MAX_US):
    """1フレーム前に出した操作(ラベルの値)[n,2]・使えるか [n]・古さ [n] ms。前のフレームが人でもその値(スティック)"""
    t_us = np.asarray(t_us, np.int64)
    label = np.asarray(label, np.float64)
    n = len(t_us)
    prev = np.zeros((n, 2))
    ok = np.zeros(n, bool)
    age = np.zeros(n, np.float32)
    if n > 1:
        prev[1:] = label[:-1]
        gap = np.diff(t_us)
        ok[1:] = (gap > 0) & (gap <= max_gap_us) & np.isfinite(label[:-1]).all(axis=1)
        age[1:] = gap / 1000.0
    return np.nan_to_num(prev).astype(np.float32), ok, age


class LogLabels(object):
    """1本のログのラベルと、そこから作る行動・未来の行動(ログ全体で一度だけ計算する)"""

    def __init__(self, log, info, arrays, label_gains='eval', gains=D3, human_ignore_bits=0):
        self.log, self.info, self.a = log, info, arrays
        self.t_us = arrays['t_us'].astype(np.int64)
        self.ai = arrays['ai'].astype(bool)
        self.ai_log = bool(info.get('ai_log'))
        r = info.get('ref_us') or {}
        self.ref = (float(r.get('steer_center', 1500.0)), float(r.get('throttle_neutral', 1500.0)))
        self.label_gains = label_gains
        if label_gains == 'eval':
            ai_label = eval_actions(arrays['out'], self.ref, gains)
        elif label_gains == 'recorded':
            ai_label = arrays['action_label'].astype(np.float64)
        else:
            raise ValueError('label_gains は eval か recorded: %s' % label_gains)
        self.label = np.where(self.ai[:, None], ai_label, arrays['human_pwm'].astype(np.float64))
        self.seg = arrays['segment_id'].astype(np.int64)
        bits = arrays['mask_bits'].astype(np.int64)
        self.valid = arrays['valid_action'].astype(bool) & np.isfinite(self.label).all(axis=1)
        # 人のフレームで、無視してよいマスク(介入の検証セットでは交代の直後も使う)
        self.valid_h = np.where(self.ai, self.valid, ((bits & ~int(human_ignore_bits)) == 0)
                                & np.isfinite(self.label).all(axis=1))
        self.fut, self.fut_ok = horizon(self.t_us, self.label, self.valid, self.seg, STEPS)
        self.human_ok = human_horizon_ok(self.t_us, ~self.ai & self.valid_h, STEPS + 1)
        self.prev, self.prev_ok, self.prev_age_ms = previous_command(self.t_us, self.label)
        self.erpm = arrays['erpm'].astype(np.float64)

    def index(self, cam_t):
        """カメラの時刻 → ラベルの行(同じ時刻が無ければ -1)"""
        cam_t = np.asarray(cam_t, np.int64)
        if not len(self.t_us):
            return np.full(len(cam_t), -1, np.int64)
        k = np.clip(np.searchsorted(self.t_us, cam_t), 0, len(self.t_us) - 1)
        return np.where(self.t_us[k] == cam_t, k, -1)


def extras(L, li):
    """足す配列(ラベルの行 li、-1 はラベルなし)"""
    has = li >= 0
    k = np.clip(li, 0, None)
    a = L.a

    def pick(name, fill, dtype):
        v = a[name][k]
        shape = (len(li),) + v.shape[1:]
        return np.where(has.reshape((-1,) + (1,) * (v.ndim - 1)), v, np.full(shape, fill)).astype(dtype)
    return {
        'source': np.where(has, L.ai[k].astype(np.int8), -1).astype(np.int8),
        'label_valid': has & L.valid[k],
        'action_label': np.where(has[:, None], L.label[k], np.nan).astype(np.float32),
        'action_exec': pick('action_exec', np.nan, np.float32),
        'action_sat': pick('action_sat', False, bool),
        'reward': pick('reward', np.nan, np.float32),
        'mc_return': pick('ret', np.nan, np.float32),
        'segment_id': pick('segment_id', -1, np.int32),
        'segment_end': pick('segment_end', False, bool),
        'end_code': pick('end_code', 0, np.int8),
        'time_to_end_s': pick('time_to_end_s', np.nan, np.float32),
        'collision': pick('collision', False, bool),
        'near_collision': pick('near_collision', False, bool),
        'failure_takeover': pick('failure_takeover', False, bool),
        'loc_code': pick('loc_code', 0, np.int8),
        'progress_m': pick('progress_m', np.nan, np.float32),
        'label_index': li.astype(np.int32),
    }


def latest_ch3(veh, cam_t):
    """各カメラ時刻の最新の CH3(µs。人・AI どちらの側のサンプルも)。無ければ NaN"""
    out = np.full(len(cam_t), np.nan)
    if veh is None or not len(veh['timestamps_us']):
        return out
    ok = veh['valid'][:, 2].astype(bool)
    vt = veh['timestamps_us'][ok].astype(np.int64)
    if not len(vt):
        return out
    j = np.searchsorted(vt, cam_t, side='right') - 1
    jc = np.clip(j, 0, len(vt) - 1)
    fresh = (j >= 0) & (cam_t - vt[jc] <= MAX_AGE_US)
    return np.where(fresh, veh['pwm'][ok][jc, 2].astype(np.float64), np.nan)


def ai_cmd_modes(ai_cmd, cam_t):
    """AI_CMD の mode_id(同じ時刻のフレーム)。無ければ -1"""
    out = np.full(len(cam_t), -1, np.int16)
    if ai_cmd is None or 'mode_id' not in ai_cmd or not len(ai_cmd['timestamps_us']):
        return out
    at = ai_cmd['timestamps_us'].astype(np.int64)
    k = np.clip(np.searchsorted(at, cam_t), 0, len(at) - 1)
    return np.where(at[k] == cam_t, ai_cmd['mode_id'][k].astype(np.int16), -1).astype(np.int16)


def apply_ai_frames(a, L, li, ai_f, ch3=None, modes_ai=None):
    """aligned の dict a の AI 側のフレーム ai_f を、ラベルの行 li の値で書き換える(その場で)"""
    if not ai_f.any():
        return
    k = li[ai_f]
    erpm = L.erpm[k]
    veh = a['vehicle'][ai_f].copy()
    veh[:, 0:2] = L.prev[k]
    if ch3 is not None:
        c = ch3[ai_f]
        veh[:, 2] = np.where(np.isfinite(c), c, veh[:, 2])
    veh[:, 3] = np.nan_to_num(erpm)
    a['vehicle'][ai_f] = veh
    a['vehicle_valid'][ai_f] = L.prev_ok[k] & np.isfinite(erpm)
    if 'vehicle_age_ms' in a:
        a['vehicle_age_ms'][ai_f] = L.prev_age_ms[k]
    a['future_actions'][ai_f] = L.fut[k]
    a['future_valid'][ai_f] = L.fut_ok[k]
    if modes_ai is not None:
        m = modes_ai[ai_f]
        a['mode'][ai_f] = np.where(m >= 0, m, a['mode'][ai_f])


def augment_aligned(a, L, ai_cmd=None, veh=None, keep_ai_log_human=False, human_label_mask=True):
    """build_dataset が作った aligned の dict a を、ラベル L で書き換えて、配列を足す(その場で)。a を返す"""
    cam_t = a['timestamps_us'].astype(np.int64)
    n = len(cam_t)
    li = L.index(cam_t)
    has = li >= 0
    k = np.clip(li, 0, None)
    ai_f = has & L.ai[k]
    hu_f = has & ~L.ai[k]
    if 'vehicle' not in a:            # 車両データの無いシーン(build の前に外しているはず)
        a['vehicle'] = np.zeros((n, 4), np.float32)
        a['vehicle_valid'] = np.zeros(n, bool)
        a['vehicle_age_ms'] = np.zeros(n, np.float32)
        a['future_actions'] = np.zeros((n, STEPS, 2), np.float32)
        a['future_valid'] = np.zeros(n, bool)
        a['erpm_valid'] = np.zeros(n, bool)
    fv = a['future_valid'].copy()
    if human_label_mask:
        fv &= ~hu_f | L.human_ok[k]
    if L.ai_log and not keep_ai_log_human:
        fv &= ~hu_f
    fv &= has
    a['future_valid'] = fv
    apply_ai_frames(a, L, li, ai_f, latest_ch3(veh, cam_t), ai_cmd_modes(ai_cmd, cam_t))
    a.update(extras(L, li))
    return a


# ---------------------------------------------------------------- シーンの選び方
def scene_problem(path):
    """build_dataset に渡せないシーンの理由(使えれば None): 4つのセンサーが読めて、空でないこと"""
    for n in SENSORS:
        f = Path(path) / (n + '.npz')
        if not f.exists():
            return '%s がありません' % n
        try:
            with np.load(f, allow_pickle=False) as z:
                if 'timestamps_us' not in z.files or not len(z['timestamps_us']):
                    return '%s が空です' % n
        except Exception as e:      # 書きかけ・空のファイル(電源断)
            return '%s が読めません(%s)' % (n, type(e).__name__)
    return None


def log_scenes(log, raw_dir=None):
    root = Path(raw_dir or config.RAW_DIR) / log
    return sorted(d.name for d in root.iterdir() if d.is_dir() and (d / 'scene.json').exists())


def labeled_ai_logs(labels_root=None):
    """data/labels/ の中で AI のフレームがあるログ"""
    out = []
    for f in sorted(Path(labels_root or labels_dir()).glob('2*.json')):
        try:
            info = json.loads(f.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if info.get('ai_log') and (info.get('counts') or {}).get('ai_frames', 0) > 0:
            out.append(info['log'])
    return out


def choose_ai_scenes(log, L, keep_ai_log_human=False, raw_dir=None, min_duration=5.0):
    """AI のログから使うシーン: 読めて、使える AI のフレーム(人も残すなら使える人のフレーム)が1つ以上ある。
    戻り値 (シーン名のリスト, 外したシーン [(名前, 理由)])"""
    use, skipped = [], []
    for s in log_scenes(log, raw_dir):
        p = Path(raw_dir or config.RAW_DIR) / log / s
        why = scene_problem(p)
        if why is None:
            meta = json.loads((p / 'scene.json').read_text(encoding='utf-8'))
            if meta.get('duration_s', 0) < min_duration:
                why = '短い(%.1f 秒)' % meta.get('duration_s', 0)
        if why is None:
            with np.load(p / 'CAM_FRONT.npz', allow_pickle=False) as z:
                ct = z['timestamps_us'].astype(np.int64)
            li = L.index(ct)
            k = np.clip(li, 0, None)
            good = (li >= 0) & L.valid[k] & (L.ai[k] | keep_ai_log_human)
            if not good.any():
                why = '使える AI のフレームがありません'
        (skipped.append((s, why)) if why else use.append(s))
    return use, skipped


def human_scenes_from(name):
    """既存のデータセットのシーンと検証シーン"""
    root = config.DATASET_DIR / name
    card = json.loads((root / 'dataset_card.json').read_text(encoding='utf-8'))
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    return list(card['source_scenes']), list(splits.get('val', []))


# ---------------------------------------------------------------- 集計
def usable(a):
    """JetFormerDataset が使うフレーム(既定の include_unset=False)"""
    return a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)


def action_reference_rows(arrays):
    """JetFormerDataset.action_reference と同じ選び方の行 [steer, throttle, source]"""
    rows = []
    for a in arrays:
        m = a['vehicle_valid'] & a.get('erpm_valid', np.ones(len(a['vehicle_valid']), bool)) \
            & (np.abs(a['vehicle'][:, 3]) < 1)
        rows.append(np.column_stack([a['vehicle'][m, :2], a['source'][m]]))
    return np.concatenate(rows) if rows else np.zeros((0, 3))


def roundtrip_error(L, gains):
    """評価時のゲインで掛け直すと out に戻るか(使える AI のフレーム。張り付きも含む)。最大の差 [µs]"""
    m = L.ai & L.valid
    if not m.any() or L.label_gains != 'eval':
        return None
    n = int(m.sum())
    g = np.tile([gains['steer_gain'], gains['throttle_gain']], (n, 1))
    back = le.apply_gains_vec(L.label[m], g, np.full(n, gains.get('steer_trim_us', 0.0)), L.ref)
    return round(float(np.abs(back - L.a['out'][m]).max()), 4)


# ---------------------------------------------------------------- 作る
def build(name, human_scenes=(), human_val=(), ai_logs=(), val_ai_logs=VAL_AI_LOGS, label_gains='eval', gains=D3,
          keep_ai_log_human=False, human_label_mask=True, min_duration=5.0, overwrite=False, labels_root=None,
          extra_card=None, log=print):
    """データセットを作る。dataset_card(dict)を返す"""
    root = config.DATASET_DIR / name
    if root.exists():
        card_p = root / 'dataset_card.json'
        built_by = json.loads(card_p.read_text(encoding='utf-8')).get('built_by') if card_p.exists() else None
        if not overwrite:
            raise ValueError('同名のデータセットがあります(--overwrite で作り直す): %s' % name)
        if built_by != TOOL:
            raise ValueError('このツールで作ったデータセットではないので消しません: %s' % name)
        shutil.rmtree(root)

    labels = {}

    def get(log_name):
        if log_name not in labels:
            info, arr = load_labels(log_name, labels_root)
            labels[log_name] = LogLabels(log_name, info, arr, label_gains, gains)
        return labels[log_name]

    scenes, val, skipped = [], [], []
    for s in human_scenes:
        p = dataset.scene_path(s)
        why = scene_problem(p)
        (skipped.append((s, why)) if why else scenes.append(s))
    val += [s for s in human_val if s in scenes]
    for lg in ai_logs:
        use, sk = choose_ai_scenes(lg, get(lg), keep_ai_log_human, min_duration=min_duration)
        skipped += sk
        scenes += use
        if lg in val_ai_logs:
            val += use
    for s in scenes:                              # 人のシーンのログのラベルも読む
        get(s.rsplit('-', 1)[0])
    log('シーン %d(検証 %d)、外したシーン %d' % (len(scenes), len(val), len(skipped)))

    def progress(frac, sname):
        k = int(round(frac * len(scenes)))
        if k % 20 == 0 or k == len(scenes):
            log('  build_dataset %d/%d %s' % (k, len(scenes), sname))
    card = dataset.build_dataset(name, scenes, min_duration=min_duration, val_scenes=val, progress=progress)
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}

    counts = {sp: {'human': 0, 'ai': 0} for sp in ('train', 'val')}
    frames = {'human': 0, 'ai': 0, 'no_label': 0}
    changed_human = 0
    ref_arrays = []
    for s in card['source_scenes']:
        lg = s.rsplit('-', 1)[0]
        L = get(lg)
        p = dataset.scene_path(s)
        f = root / 'aligned' / (s + '.npz')
        with np.load(f, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        before = a['future_valid'].copy()
        ai_cmd = dataset._load(p, 'AI_CMD')
        veh = dataset._load(p, 'VEHICLE')
        augment_aligned(a, L, ai_cmd, veh, keep_ai_log_human, human_label_mask)
        np.savez_compressed(f, **a)
        src = a['source']
        frames['human'] += int((src == 0).sum())
        frames['ai'] += int((src == 1).sum())
        frames['no_label'] += int((src < 0).sum())
        if not L.ai_log:
            changed_human += int((before & ~a['future_valid'] & (src == 0)).sum())
        u = usable(a)
        sp = split_of.get(s, 'train')
        counts[sp]['human'] += int((u & (src == 0)).sum())
        counts[sp]['ai'] += int((u & (src == 1)).sum())
        if sp == 'train':
            ref_arrays.append(a)

    rows = action_reference_rows(ref_arrays)
    ref_check = {}
    for k, m in (('all', np.ones(len(rows), bool)), ('human_only', rows[:, 2] == 0)):
        r = rows[m]
        ref_check[k] = {'n': int(len(r)), 'steer_us': round(float(np.median(r[:, 0])), 1) if len(r) else None,
                        'throttle_us': round(float(np.median(r[:, 1])), 1) if len(r) else None}
    used_logs = sorted({s.rsplit('-', 1)[0] for s in card['source_scenes']})
    card.update({
        'built_by': TOOL,
        'ai_mix': {
            'label_gains': label_gains,
            'eval_gains': dict(gains) if label_gains == 'eval' else None,
            'gain_rule_note': '速度のゲインは前進側だけ(今の drive.apply_gains)。評価時のゲインで割り戻すので、記録時の規則'
                              '(古いログはブレーキ側にも掛けていた)は行動ラベルに影響しない',
            'ai_logs': list(ai_logs), 'val_ai_logs': [x for x in val_ai_logs if x in ai_logs],
            'human_val_scenes': [s for s in human_val if s in scenes],
            'keep_ai_log_human': bool(keep_ai_log_human), 'human_label_mask': bool(human_label_mask),
            'frames': frames,
            'usable_frames': counts,
            'human_frames_removed_by_label_mask': changed_human,
            'skipped_scenes': [{'scene': s, 'reason': w} for s, w in skipped],
            'labels': {lg: {'version': labels[lg].info.get('version'), 'created_at': labels[lg].info.get('created_at'),
                            'ref_us': list(labels[lg].ref),
                            'recorded_gain_rule': (labels[lg].info.get('gain_rule') or {}).get('rule'),
                            'eval_roundtrip_max_abs_us': roundtrip_error(labels[lg], gains)}
                       for lg in used_logs},
            'action_reference_check': ref_check,
            'future_rule': 'AI: k=0..9 は撮像から k+0.5 フレーム以内の最新の AI_CMD の out(続くフレーム i+k)。'
                           '現在〜9フレーム先が label_valid・同じ区間・100ms 以内。人: build_dataset のまま + '
                           '現在〜10フレーム先の時間のフレームがすべて人で label_valid',
            'vehicle_rule': 'AI: [1フレーム前の行動ラベル(割り戻し後), CH3, eRPM]。人: build_dataset のまま',
        },
    })
    card['aligned_npz'] = dict(card.get('aligned_npz', {}), **EXTRA_KEYS)
    if extra_card:
        card.update(extra_card)
    dataset._write_json(root / 'dataset_card.json', card, indent=1)
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', default='ai_mix_20261011')
    ap.add_argument('--human-from', default=HUMAN_FROM,
                    help='このデータセットと同じ人のシーン・検証シーンを入れる(空文字で入れない)')
    ap.add_argument('--human-scenes', nargs='*', default=[], help='ほかに入れる人のシーン(学習用)')
    ap.add_argument('--ai-logs', nargs='*', default=None, help='AI 走行のログ(既定: data/labels/ で AI のフレームがあるログ全部)')
    ap.add_argument('--val-ai-logs', nargs='*', default=list(VAL_AI_LOGS), help='丸ごと検証用にする AI のログ')
    ap.add_argument('--label-gains', choices=('eval', 'recorded'), default='eval')
    ap.add_argument('--eval-steer-gain', type=float, default=D3['steer_gain'])
    ap.add_argument('--eval-throttle-gain', type=float, default=D3['throttle_gain'])
    ap.add_argument('--eval-trim-us', type=float, default=D3['steer_trim_us'])
    ap.add_argument('--keep-ai-log-human', action='store_true', help='AI のログの中の人のフレームも学習に使う')
    ap.add_argument('--no-human-label-mask', action='store_true', help='人のフレームにラベルのマスクを掛けない')
    ap.add_argument('--min-duration', type=float, default=5.0)
    ap.add_argument('--overwrite', action='store_true', help='このツールで作った同名のデータセットを作り直す')
    args = ap.parse_args()
    t0 = datetime.datetime.now()
    hs, hv = (human_scenes_from(args.human_from) if args.human_from else ([], []))
    ai_logs = args.ai_logs if args.ai_logs is not None else labeled_ai_logs()
    gains = {'steer_gain': args.eval_steer_gain, 'throttle_gain': args.eval_throttle_gain,
             'steer_trim_us': args.eval_trim_us}
    card = build(args.name, hs + [s for s in args.human_scenes if s not in hs], hv, ai_logs, args.val_ai_logs,
                 args.label_gains, gains, args.keep_ai_log_human, not args.no_human_label_mask, args.min_duration,
                 args.overwrite, extra_card={'human_from': args.human_from or None})
    m = card['ai_mix']
    print(json.dumps({'name': card['name'], 'scenes': card['scenes'], 'train_scenes': card['train_scenes'],
                      'val_scenes': card['val_scenes'], 'frames': card['frames'], 'hours': card['hours'],
                      'frames_by_source': m['frames'], 'usable_frames': m['usable_frames'],
                      'human_frames_removed_by_label_mask': m['human_frames_removed_by_label_mask'],
                      'action_reference_check': m['action_reference_check'],
                      'skipped_scenes': len(m['skipped_scenes']),
                      'seconds': round((datetime.datetime.now() - t0).total_seconds(), 1)}, ensure_ascii=False, indent=1))
    print('データセット: %s' % (config.DATASET_DIR / args.name))


if __name__ == '__main__':
    main()
