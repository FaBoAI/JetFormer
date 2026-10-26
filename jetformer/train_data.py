"""学習用データローダー(エクスポートしたデータセットの aligned/*.npz と画像を読む)

    ds = JetFormerDataset('data/datasets/indoor_v1', split='train')
    image, tof, imu, state, mode, target = ds[0]      # mode は走行ジャンルのone-hot

オフライン強化学習の方策の学習用(既定では使わない。tools/rl_policies.py が作る npz を tools/train.py から渡す):
    フレームの表 = npz(scene [N] シーン名, frame [N] aligned のフレーム番号, 値の配列 [N])。load_frame_table で読む
    drop        True のフレームを学習に使わない
    keep_impact True のフレームは衝突の前後の除外(mask_impacts)から外す(AWR などで失敗の前後の AI のフレームを使う)
    weights     損失の重み(return_weight=True のとき 7番目の値として返す)
    mode_remap  学習するジャンルを付け替える(-1 = そのまま)。remap_dropout の確率で元のジャンルに戻す(分類器なし誘導の形)

ベンチマークの学習用(2026-10-21。既定では使わない。tools/train.py の --label-override-npz・--aug-* から渡す):
    label_override  フレームの表(scene, frame, future_throttle_us [M,10] µs。NaN = 元のまま)。学習の正解の
                    1〜10 フレーム先のスロットルを差し替える(区間の速度上限の出力に付け替えた正解など)
    aug             入力の拡張 {'brightness': 明るさ ±割合, 'contrast': コントラスト ±割合, 'shift_px': 上下左右のずれ
                    (モデルの入力の画素), 'tof_noise': ToF の距離(正規化)に足す雑音の標準偏差,
                    'top_brightness': 上 1/3 の明るさ ±割合(上ほど強く、なだらかに。2026-10-25 C1)}。0・None = 使わない。
                    使わないときは乱数を引かないので、同じ種なら今までと同じサンプルになる

学習 v3 用(2026-10-26 段 data。既定では使わない。tools/train.py の --action-ref-from-samples・--probe-npz から使う):
    action_reference_from_samples  作用の基準を学習に使うサンプル(index)のフレームだけから出す(action_reference は変えない)
    ProbeSet                       学習中の物差しのフレーム(直線・止まった場面・低速。probe_gates.py と同じ式)。乱数を引かない

方策の追加の正解(既定では使わない。jetformer/policies の方策のモジュールの extra_targets が作る):
    extra_targets  {シーン名: {キー: np.ndarray [シーンのフレーム数, ...]}}。渡すと各サンプルの最後に
                   {'y_<キー>': Tensor} の dict を足す(tcp の通過点・目標速度など)。渡さなければ今までと同じタプル
"""

import json
import os

import cv2
import numpy as np
import torch

from . import tof_format
from .model import ERPM_SCALE, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, PWM_CENTER, PWM_SCALE, TOF_MAX_MM

UNSET_MODE = 0

GRAVITY = 9.80665
IMPACT_G = 4.0                     # 水平の加速度(重力を除く)がこれ以上 = 衝突
IMPACT_BEFORE_S, IMPACT_AFTER_S = 1.0, 2.0


def impact_mask(timestamps_us, imu, impact_g=IMPACT_G, before_s=IMPACT_BEFORE_S, after_s=IMPACT_AFTER_S):
    """衝突の前後(before_s 前 〜 after_s 後)のフレームを True にする。AIは学習データの衝突もまねるので学習から外す
    (2026-10-10: 人の学習データに衝突が33回あり、AIも同じ場所でぶつかっていた)。
    imu: [N,10,6] 各フレームの直前10サンプル(加速度 m/s^2 が先頭3つ)。IMUは +x=右・+y=前 なので水平は x,y"""
    t = np.asarray(timestamps_us, np.float64) / 1e6
    acc = np.asarray(imu, np.float32)[:, :, :2]
    hg = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / GRAVITY
    mask = np.zeros(len(t), bool)
    for ti in t[hg >= impact_g]:
        mask |= (t >= ti - before_s) & (t <= ti + after_s)
    return mask


def normalize_inputs(image_bgr, tof_raw, imu, vehicle, use_command=True):
    """推論時(Jetson)と学習時で共通の正規化。numpy配列 → モデル入力(float32)

    tof_raw: ToFの生の値 [42,54] / vehicle: [steering_us, throttle_us, mode_us, erpm]
    use_command=False: 今の操作値(ステアリング・スロットル)を入れない(0にする)。学習時は人の操作、AI走行時は
    自分の前の予測になって条件が変わるため、カメラ・ToF・IMU・回転数だけから操作を決めさせたいときに使う"""
    img = cv2.cvtColor(image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    tof = np.stack([np.clip(tof_format.distance_mm(tof_raw) / TOF_MAX_MM, 0.0, 1.0),
                    tof_format.confidence(tof_raw).astype(np.float32) / tof_format.CONF_MAX])
    imu_n = imu.astype(np.float32).copy()
    imu_n[:, :3] /= GRAVITY
    state = np.array([(vehicle[0] - PWM_CENTER) / PWM_SCALE, (vehicle[1] - PWM_CENTER) / PWM_SCALE,
                      np.clip(vehicle[3] / ERPM_SCALE, -3.0, 3.0)], dtype=np.float32)
    if not use_command:
        state[:2] = 0.0
    return img.astype(np.float32), tof, imu_n, state


def load_frame_table(path, key):
    """フレームの表の npz(scene, frame, <key>)→ {シーン名: (フレーム番号 [M], 値 [M])}"""
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError('%s に %s がありません(あるもの: %s)' % (path, key, ', '.join(z.files)))
        scene, frame, value = z['scene'].astype(str), z['frame'].astype(np.int64), z[key]
    if not (len(scene) == len(frame) == len(value)):
        raise ValueError('%s: scene・frame・%s の長さが違います' % (path, key))
    table = {}
    for s in np.unique(scene):
        m = scene == s
        table[str(s)] = (frame[m], value[m])
    return table


def _per_frame(table, scene, n, fill, dtype):
    """フレームの表から、シーンの全フレーム分の配列を作る。表に無いフレームは fill。シーンが表に無ければエラー
    (別のデータセットの表を渡したまま学習してしまわないように)"""
    if scene not in table:
        raise KeyError('フレームの表にシーン %s がありません' % scene)
    frame, value = table[scene]
    if len(frame) and (frame.min() < 0 or frame.max() >= n):
        raise ValueError('フレームの表のフレーム番号がシーン %s の範囲(0〜%d)の外です' % (scene, n - 1))
    out = np.full(n, fill, dtype=dtype)
    out[frame] = value
    return out


AUG_KEYS = ('brightness', 'contrast', 'shift_px', 'tof_noise', 'top_brightness')
TOP_FRACTION = 1.0 / 3.0           # --aug-top-brightness が掛かる上の割合(10/25 の棚卸し: 昼と夕方の差は上 1/3 に出る)


def top_ramp(height):
    """上ほど強い重み [height, 1, 1]: 一番上の行 1.0 から、上 1/3 の境まで直線で 0 に下がる(それより下は 0)"""
    w = 1.0 - np.arange(height, dtype=np.float32) / (height * TOP_FRACTION)
    return np.clip(w, 0.0, 1.0).reshape(height, 1, 1)


def augment_image(image, brightness=0.0, contrast=0.0, shift_px=0, top_brightness=0.0):
    """カメラの画像(BGR uint8)の拡張: コントラスト ×U(1±contrast)(画像の平均の周り)・明るさ +U(±brightness)×255・
    上 1/3 の明るさ +U(±top_brightness)×255×top_ramp(一番上で全部、上 1/3 の境で 0。--aug-brightness と同じ単位)・
    上下左右に整数の画素ずらし(モデルの入力 IMAGE_SIZE の画素で ±shift_px、端は写し足す)。0 の項目は乱数を引かない
    (乱数の順: コントラスト → 明るさ → 上 1/3 → ずれ。top_brightness = 0 なら今までと同じ順・同じ値)"""
    img = image
    if contrast > 0 or brightness > 0 or top_brightness > 0:
        img = img.astype(np.float32)
        if contrast > 0:
            c = np.random.uniform(1.0 - contrast, 1.0 + contrast)
            mean = float(img.mean())
            img = (img - mean) * c + mean
        if brightness > 0:
            img = img + np.random.uniform(-brightness, brightness) * 255.0
        if top_brightness > 0:
            img = img + np.float32(np.random.uniform(-top_brightness, top_brightness) * 255.0) * top_ramp(img.shape[0])
        img = np.clip(img, 0, 255).astype(np.uint8)
    if shift_px > 0:
        dx, dy = np.random.randint(-int(shift_px), int(shift_px) + 1, size=2)
        if dx or dy:
            sx, sy = img.shape[1] / float(IMAGE_SIZE), img.shape[0] / float(IMAGE_SIZE)
            m = np.float32([[1, 0, round(dx * sx)], [0, 1, round(dy * sy)]])
            img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)
    return img


def augment_tof(tof, noise=0.0):
    """正規化した ToF [2,H,W] の距離(0 番目)に N(0, noise) を足して 0〜1 に切る。信頼度は変えない"""
    if noise <= 0:
        return tof
    tof = tof.copy()
    tof[0] = np.clip(tof[0] + np.random.normal(0.0, noise, tof[0].shape).astype(np.float32), 0.0, 1.0)
    return tof


class JetFormerDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', color_jitter=False, include_unset=False, stride=1, use_command=True,
                 mask_impacts=False, impact_g=IMPACT_G, drop=None, keep_impact=None, weights=None, mode_remap=None,
                 remap_dropout=0.0, extra_modes=None, return_weight=False, extra_targets=None,
                 label_override=None, aug=None):
        """drop・keep_impact・weights・mode_remap は load_frame_table の表(None = 使わない。既定はどれも使わない)。
        extra_modes: mode_remap の付け替え先のジャンル(data/modes.json の項目)で、データセットの表に無いもの。
        extra_targets: 方策の追加の正解 {シーン名: {キー: [フレーム数, ...]}}(None = 使わない)
        label_override: 正解のスロットルの差し替えの表(load_frame_table(..., 'future_throttle_us'))。表に無いシーンは
        そのまま。表のシーンがこの分け方に無ければエラー(検証の分け方に渡してしまわないように)
        aug: 入力の拡張の設定(AUG_KEYS。None・全部 0 = 使わない)"""
        self.root = root
        self.color_jitter = color_jitter
        self.use_command = use_command
        self.impact_masked = 0             # 衝突の前後として外したフレーム数(間引きの前)
        self.dropped = 0                   # drop で外したフレーム数(間引きの前)
        self.impact_kept = 0               # keep_impact で衝突の前後の除外から戻したフレーム数(間引きの前)
        self.remap_dropout = float(remap_dropout)
        self.return_weight = return_weight
        self.extra_keys = []                 # 追加の正解のキー(__getitem__ の最後の dict の 'y_<キー>')
        self.aug = {k: float((aug or {}).get(k, 0) or 0) for k in AUG_KEYS} if aug and any(
            float(aug.get(k, 0) or 0) > 0 for k in AUG_KEYS) else None
        self.label_overridden = 0            # 差し替えた正解の行の数(間引きの前の全フレーム)
        self.label_overridden_used = 0       # そのうち学習に使うフレームの行の数
        with open(os.path.join(root, 'dataset_card.json'), encoding='utf-8') as f:
            card = json.load(f)
        self.modes = card.get('modes', [{'id': 0, 'name': '未設定'}])
        if extra_modes:
            known = {m['id'] for m in self.modes}
            self.modes = list(self.modes) + [dict(m) for m in extra_modes if m['id'] not in known]
        self.num_modes = max(m['id'] for m in self.modes) + 1
        with open(os.path.join(root, 'splits.json')) as f:
            scenes = json.load(f)[split]
        if label_override is not None:
            stray = sorted(set(label_override) - set(scenes))
            if stray:
                raise KeyError('正解の差し替えの表のシーン %s が %s の分け方にありません' % (', '.join(stray[:3]), split))
        self.scenes = []
        self.index = []
        for s in scenes:
            a = dict(np.load(os.path.join(root, 'aligned', s + '.npz')))
            if 'vehicle' not in a or 'tof' not in a or 'imu' not in a:
                continue
            ok = a['future_valid'] & a['vehicle_valid']
            n = len(ok)
            ov_rows = None
            if label_override is not None and s in label_override:
                fr, val = label_override[s]
                val = np.asarray(val, np.float32)
                fa = np.array(a['future_actions'], np.float32, copy=True)
                if val.ndim != 2 or val.shape[1] != fa.shape[1]:
                    raise ValueError('シーン %s: 正解の差し替えの形 %s が [M,%d] ではありません' % (s, val.shape, fa.shape[1]))
                if len(fr) and (fr.min() < 0 or fr.max() >= n):
                    raise ValueError('正解の差し替えのフレーム番号がシーン %s の範囲(0〜%d)の外です' % (s, n - 1))
                sub = fa[fr, :, 1]
                m = np.isfinite(val)
                sub[m] = val[m]
                fa[fr, :, 1] = sub
                a['future_actions'] = fa
                ov_rows = np.zeros(n, np.int64)
                np.add.at(ov_rows, fr, m.sum(axis=1))
                self.label_overridden += int(ov_rows.sum())
            if 'mode' not in a:
                a['mode'] = np.zeros(len(ok), dtype=np.int16)
            if not include_unset:
                ok &= a['mode'] != UNSET_MODE          # ジャンル未設定のフレームは学習に使わない
            if drop is not None:
                d = _per_frame(drop, s, n, False, bool)
                self.dropped += int((ok & d).sum())
                ok &= ~d
            if mask_impacts:
                m = impact_mask(a['timestamps_us'], a['imu'], impact_g)
                if keep_impact is not None:
                    keep = _per_frame(keep_impact, s, n, False, bool)
                    self.impact_kept += int((ok & m & keep).sum())
                    m &= ~keep
                self.impact_masked += int((ok & m).sum())
                ok &= ~m
            if stride > 1:                              # 間引き: 30fpsの連続フレームは似ているので N枚に1枚
                ok &= (np.arange(len(ok)) % stride) == 0
            if weights is not None:
                a['_weight'] = _per_frame(weights, s, n, np.nan, np.float32)
                bad = ok & ~np.isfinite(a['_weight'])
                if bad.any():
                    raise ValueError('シーン %s: 学習に使うフレーム %d 個に重みがありません' % (s, int(bad.sum())))
            if mode_remap is not None:
                r = _per_frame(mode_remap, s, n, -1, np.int16)
                bad = ok & (r >= self.num_modes)
                if bad.any():
                    raise ValueError('シーン %s: 付け替え先のジャンル %s がジャンルの表にありません'
                                     % (s, sorted(set(r[bad].tolist()))))
                a['_remap'] = r
            if extra_targets is not None:
                if s not in extra_targets:
                    raise KeyError('%s の追加の正解がありません' % s)
                keys = sorted(extra_targets[s])
                if self.scenes and keys != self.extra_keys:
                    raise ValueError('シーン %s の追加の正解のキー %s がほかのシーン %s と違います' % (s, keys, self.extra_keys))
                self.extra_keys = keys
                for key in keys:
                    arr = np.asarray(extra_targets[s][key])
                    if len(arr) != n:
                        raise ValueError('シーン %s の追加の正解 %s の長さ %d がフレーム数 %d と違います' % (s, key, len(arr), n))
                    a['_y_' + key] = arr
            if ov_rows is not None:
                self.label_overridden_used += int(ov_rows[ok].sum())
            k = len(self.scenes)
            self.scenes.append(a)
            self.index.extend((k, int(i)) for i in np.nonzero(ok)[0])

    def __len__(self):
        return len(self.index)

    def mode_of(self, k, i, train_draw=False):
        """学習するジャンル。付け替えがあれば付け替え先(train_draw=True なら remap_dropout の確率で元のジャンル)"""
        a = self.scenes[k]
        m = int(a['mode'][i])
        r = int(a['_remap'][i]) if '_remap' in a else -1
        if r < 0 or (train_draw and self.remap_dropout > 0 and np.random.random() < self.remap_dropout):
            return m
        return r

    def __getitem__(self, n):
        k, i = self.index[n]
        a = self.scenes[k]
        image = cv2.imread(os.path.join(self.root, str(a['cam_files'][i])))
        if self.color_jitter:
            gain = np.random.uniform(0.8, 1.2, size=3)
            image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if self.aug is not None:
            image = augment_image(image, self.aug['brightness'], self.aug['contrast'], int(self.aug['shift_px']),
                                  self.aug['top_brightness'])
        img, tof, imu, state = normalize_inputs(image, a['tof'][i], a['imu'][i], a['vehicle'][i], self.use_command)
        if self.aug is not None:
            tof = augment_tof(tof, self.aug['tof_noise'])
        target = (a['future_actions'][i] - PWM_CENTER) / PWM_SCALE
        mode = np.zeros(self.num_modes, dtype=np.float32)
        mode[self.mode_of(k, i, train_draw=True) if '_remap' in a else int(a['mode'][i])] = 1.0
        out = (torch.from_numpy(img), torch.from_numpy(tof), torch.from_numpy(imu), torch.from_numpy(state),
               torch.from_numpy(mode), torch.from_numpy(target.astype(np.float32)))
        if self.return_weight:
            w = float(a['_weight'][i]) if '_weight' in a else 1.0
            out = out + (torch.tensor(w, dtype=torch.float32),)
        if self.extra_keys:                  # 方策の追加の正解(最後の dict。既定の collate がまとめる)
            out = out + ({'y_' + key: torch.as_tensor(np.array(a['_y_' + key][i])) for key in self.extra_keys},)
        return out

    def sample_weights(self):
        """index の順の重み [len](weights が無ければ 1)"""
        return np.array([float(self.scenes[k]['_weight'][i]) if '_weight' in self.scenes[k] else 1.0
                         for k, i in self.index], np.float64)

    def mode_counts(self):
        """ジャンル別のサンプル数(付け替えがあれば付け替え先で数える。remap_dropout で元に戻る分は数えない)"""
        counts = {}
        for k, i in self.index:
            m = self.mode_of(k, i)
            counts[m] = counts.get(m, 0) + 1
        return counts

    def action_reference(self):
        """(ステアリング中心, スロットル中立) [µs]: 止まっている間(eRPM=0)の人の操作値の中央値。
        走行時のゲインはこの値を中心に掛ける。止まっているフレームが少なければ PWM_CENTER"""
        rows = [a['vehicle'][a['vehicle_valid'] & a.get('erpm_valid', True) & (np.abs(a['vehicle'][:, 3]) < 1)]
                for a in self.scenes]           # 回転数が無いデータ(旧ファーム)は止まっているか分からないので除く
        rows = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
        if len(rows) < 10:
            return float(PWM_CENTER), float(PWM_CENTER)
        return round(float(np.median(rows[:, 0])), 1), round(float(np.median(rows[:, 1])), 1)

    def action_reference_from_samples(self):
        """action_reference と同じ式(止まっている間の操作値の中央値)を、学習に使うサンプル(index のフレーム。drop・
        衝突の前後・間引きのあと)だけから出す(2026-10-26 段 data、protocol_v3 案 4。tools/train.py の
        --action-ref-from-samples)。→ (ステアリング中心, スロットル中立, 数えた行の数)。行が 10 未満なら PWM_CENTER。
        今の action_reference は drop で外したフレーム(AI の記録の止まったフレームなど)も数えるので、学習に使わない
        データの止まり方で中立がずれる(dataset_v2 で 1433/1500、学習のサンプルだけなら 1441/1496)"""
        by_scene = {}
        for k, i in self.index:
            by_scene.setdefault(k, []).append(i)
        rows = []
        for k, idx in sorted(by_scene.items()):
            a = self.scenes[k]
            idx = np.asarray(idx, np.int64)
            ev = a['erpm_valid'][idx] if 'erpm_valid' in a else True
            m = a['vehicle_valid'][idx] & ev & (np.abs(a['vehicle'][idx, 3]) < 1)
            rows.append(a['vehicle'][idx][m])
        rows = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
        if len(rows) < 10:
            return float(PWM_CENTER), float(PWM_CENTER), int(len(rows))
        return round(float(np.median(rows[:, 0])), 1), round(float(np.median(rows[:, 1])), 1), int(len(rows))


# ---------------------------------------------------------------- 学習中の物差し(2026-10-26 段 data、protocol_v3 案 5)
PROBE_PARTS = ('st10', 'st23', 'stand', 'low10')
PROBE_GENRE = 5                    # C3 の机上・probe_gates.py と同じジャンル(低速走行)
PROBE_NEUTRAL_US = 1498.0          # 門の決まり(desk_lib.NEUTRAL_US)。probe_gates.py の NEUTRAL と同じ
PROBE_GO_NEUTRAL_US = 1496.0       # probe_gates.py の stand_go_frac の中立(B の中立)


class ProbeSet(object):
    """学習の途中で「直線で緩めていないか」「止まった場面・低速でブレーキ側を出していないか」を見る小さな物差しのフレーム。
    experiments/20261026_fix/diag_data/probe_gates.py の Probe と同じフレーム・同じ入力(use_command=False と同じ正規化)・
    同じ式。見るだけ: 学習の損失・最良のエポックの選び方・データには使わない。乱数を引かない(DataLoader を使わない)。

    npz(probe_gates.py make が作る probe_set.npz): <part>_ds(データセットの名前か絶対パス)・<part>_scene・<part>_frame・
    <part>_B [M,3](B の行 0〜2 のスロットル µs)。part = st10・st23・stand・low10。任意で genre(既定 5)。
    データセットの名前は datasets_root(学習の --data の親)の下で探す。

    使い方(tools/train.py の --probe-npz): P = ProbeSet(npz, datasets_root, use_command); P.batches(part) で入力の組を
    順に作り、モデルの行 0〜2 のスロットル [µs] を P.metrics({part: [M,3]}) に渡す → probe_gates.py と同じ欄の dict"""

    def __init__(self, path, datasets_root, use_command=False):
        self.path = str(path)
        self.use_command = use_command
        self.parts = {}
        with np.load(path, allow_pickle=False) as z:
            self.genre = int(z['genre']) if 'genre' in z.files else PROBE_GENRE
            for k in PROBE_PARTS:
                ds = str(z[k + '_ds'])
                root = ds if os.path.isabs(ds) else os.path.join(str(datasets_root), ds)
                scenes, frames = z[k + '_scene'].astype(str), z[k + '_frame'].astype(np.int64)
                base = z[k + '_B'].astype(np.float64)
                if not (len(scenes) == len(frames) == len(base)) or base.ndim != 2 or base.shape[1] < 3:
                    raise ValueError('%s: %s の scene・frame・B の形が合いません' % (path, k))
                self.parts[k] = {'root': root, 'items': self._items(root, scenes, frames), 'B': base}

    @staticmethod
    def _items(root, scenes, frames):
        items, cur, a = [], None, None
        for s, i in zip(scenes, frames):
            if s != cur:
                with np.load(os.path.join(root, 'aligned', s + '.npz'), allow_pickle=False) as z:
                    a = {k: z[k] for k in ('cam_files', 'tof', 'imu', 'vehicle')}
                cur = s
            i = int(i)
            if i < 0 or i >= len(a['cam_files']):
                raise ValueError('物差しのフレーム番号 %d がシーン %s の範囲の外です' % (i, s))
            items.append((os.path.join(root, str(a['cam_files'][i])), a['tof'][i].copy(), a['imu'][i].copy(),
                          a['vehicle'][i].copy()))
        return items

    def counts(self):
        return {k: len(v['items']) for k, v in self.parts.items()}

    def batches(self, part, batch=64):
        """part のフレームを並びの順に batch 個ずつ → (image, tof, imu, state) の numpy の組(乱数を引かない)"""
        items = self.parts[part]['items']
        for s in range(0, len(items), batch):
            out = ([], [], [], [])
            for path, tof, imu, veh in items[s:s + batch]:
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError('物差しの画像が読めません: %s' % path)
                for lst, x in zip(out, normalize_inputs(img, tof, imu, veh, use_command=self.use_command)):
                    lst.append(x)
            # C の並びにそろえる(probe_gates.py の DataLoader の torch.stack と同じ形。転置のままの並びだと GPU の畳み込みの
            # 計算の順が変わり、値が 0.1〜0.3 µs ずれる)
            yield tuple(np.ascontiguousarray(np.stack(x)) for x in out)

    def metrics(self, pred):
        """pred {part: [M,3] 行 0〜2 のスロットル µs(生の予測。ゲインの前)} → probe_gates.py の Probe と同じ欄・同じ丸め"""
        r = {}
        for k in ('st10', 'st23'):
            r['%s_thr_vs_b_us' % k] = round(float(np.mean(pred[k][:, 2] - self.parts[k]['B'][:, 2])), 2)
        r['stand_brake_side_frac'] = round(float(np.mean(pred['stand'][:, 0] > PROBE_NEUTRAL_US + 30)), 4)
        r['stand_go_frac'] = round(float(np.mean(pred['stand'][:, 0] < PROBE_GO_NEUTRAL_US - 30)), 4)
        r['low10_brake_frac'] = round(float(np.mean(pred['low10'][:, 0] > PROBE_NEUTRAL_US + 100)), 4)
        r['n'] = {k: int(len(v)) for k, v in pred.items()}
        return r
