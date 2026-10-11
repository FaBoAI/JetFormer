#!/usr/bin/env python3
"""オフライン強化学習 その3: 机上の評価(ゲートC)と「走行の精度」(ロードマップ 4.4・4.5、offline_rl.md 5.4)

    python tools/rl_evaluate.py heldout        # 保留データと判定の規則を固定 → data/rl/<データセット>/eval/heldout.json(上書きしない)
    python tools/rl_evaluate.py predict        # GPU: 7通りの予測(モデル B・(c)・(b)・(a) 良/元・(n) 良/元)。出来たものは飛ばす
    python tools/rl_evaluate.py crosscheck     # GPU: tools/compare_models.py の predict_scene と同じ予測になるか(1シーン)
    python tools/rl_evaluate.py evaluate       # CPU: ゲートC の各項目と走行の精度 → eval/evaluation.json
    python tools/rl_evaluate.py figures --fig-dir <dir>      # 図(日本語・大きい文字)
    python tools/rl_evaluate.py engine-check   # GPU: 走行のコード(jetformer/drive.py)と同じ手順でエンジンを読む(走らせない)

保留データ(方策を学習する前に決まっていたもの。heldout で中身と規則を書き出して固定し、結果を見てから入れ替えない):
  ・ai_mix_20261011 の val の全シーン(人のきれいな走行 11シーン = clean_driving_20261011 の val と同じ中身、
    AI のログ 172505・204208 は丸ごと)
  ・intervention_val_20261011(交代の後の人の修正。29窓。172505・204208 の3窓だけがログ単位で保留。ほかの窓は
    フレームは学習に使っていないが、同じログの AI のフレームを4本の方策が学習している)
予測は PyTorch(best.pth、FP32)。今の操作値を入れないモデルなので、教師あり = 自己入力(crosscheck で確かめる)。
R1・DE・直線の判定には、ゲートD の固定設定(ステアのゲイン 1.3・速度のゲイン 1.0・先読み 3)で走行のコードが出す値
(jetformer/drive.py の apply_gains と同じ式)を使う。

GPU を使うのは predict・crosscheck・engine-check だけ(1つずつ。ほかの GPU の処理が動いていたら始めない)。
電源が落ちても、同じコマンドをもう一度実行すれば続きから(予測は1通りずつ npz に書き、出来たものは飛ばす)。
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from jetformer import config  # noqa: E402
import rl_policies as rp  # noqa: E402

DATASET = rp.DATASET
IV_DATASET = 'intervention_val_20261011'
CLEAN_DATASET = 'clean_driving_20261011'          # モデル B の学習データ
BASE_RUN = rp.BASE_RUN
EPOCHS = 12
LOC_NAMES = ['unknown', 'R1', 'GM', 'DE', 'other', 'HS', 'LH']    # tools/label_events.py の LOC_NAMES
AI_VAL_LOGS = ('20261010-172505', '20261010-204208')

# ゲートD の固定設定(走行の出力の計算に使う)
DRIVE_SETTINGS = {'steer_gain': 1.3, 'throttle_gain': 1.0, 'lookahead': 3, 'steer_trim_us': 0.0}
NEUTRAL_US = 1498.0                # 前進の強さ(中立 − スロットル)の基準。4本の方策の throttle_neutral_us
R1_LEAD_S = 0.4                    # R1 の起点の何秒手前から(起点まで)
DE_WINDOW_S = 0.8                  # 緑マット出口から何秒まで
ENTRY_BAND_ERPM = 5500.0
STRAIGHT = {'locs': ('HS', 'DE'), 'max_abs_yaw_dps': 20.0, 'min_fwd_erpm': 2000.0}
IMPACT_G = 2.5
N_BOOT = 2000

# 判定の規則(heldout.json に書き出して固定する。数字は結果を見る前に決めた)
RULES = {
    'reference_rl': 'c',            # (a)(b)(n) の R1・DE は (c) filtered BC と比べる(ロードマップ 4.4 の 2・3)
    'reference_collapse': 'B',      # 「崩れていない」(4)は、今の基準のモデル B と比べる((c) との比較も載せる)
    'r1': '保留した R1 の通過(窓 = 起点の 0.4秒手前〜起点、窓のフレームがすべて val)で、走行の出力のスロットル(先読み 3・'
          'ゲイン 1.0)の通過ごとの平均の差(方策 − 基準、+ = 遅い)の通過単位ブートストラップ 95% 区間の下限 > 0、かつ、'
          '見積もった進入 eRPM が 5500 未満の通過の割合が基準より高い(点推定)',
    'de': '保留した DE の通過(窓 = 緑マット出口〜0.8秒後)で、走行の出力のステア(ゲイン 1.3)の差(方策 − 基準、+ = 左)の'
          '通過単位ブートストラップ 95% 区間の下限 > 0',
    'collapse': '(i) 保留の直線(場所 HS・DE、|ヨー| < 20°/s、前進 eRPM > 2000、衝突の前後を除く)で、前進の強さ'
                '(1498µs − 出力のスロットル)の低下がモデル B の 5% 以内(点推定)、(ii) 保留した人のきれいな走行'
                '(衝突の前後 2.5G を除く)の L1(1〜10フレーム先・ステアとスロットルの平均)がモデル B の 1.10倍以内(点推定)、'
                '(iii) 同じフレームで前進の偏り(3フレーム先の予測 − 人、tools/compare_models.py の fwd_bias_us)≥ −30µs',
    'labels': '学習の印(I。(n) は I_neg)が、(i) 芝などの空転(コース図の進み / 車輪の距離 < 0.6、0.5秒の窓)、'
              '(ii) 2.5G 未満の強い衝撃(水平 1.5〜2.5G、衝突の前後を除く)に、比べる集団より 10ポイント以上多く付いて'
              'いない、(iii) 人の後退(eRPM > 300)の「良い」の割合が 10% 以下。ヘアピンの内側を削る走りは位置が無いので'
              '判定しない',
    'negative_control': '陰性対照 (n)(「〜・良」のジャンルで評価)が、方策の出力の項目(2・3・4)のどれかで落ちること。'
                        '落ちなければ、ゲートC は何も見分けていない',
    'value': 'ゲートC の1(価値の AUROC)はその1(value.md)の結果をそのまま使う',
    'candidate_rl': '(a)(b) を実車の候補にするのは、1〜5 をすべて満たし、かつ陰性対照が落ちたときだけ',
    'candidate_c': '(c) filtered BC を「段階B + AI のデータ」の候補にするのは、モデル B と比べて 4 を満たし、R1 か DE の'
                   'どちらかで B より良い(区間が 0 を含まない)、かつ、どちらも B より有意に悪くないとき'
                   '(ロードマップ 4.4 ゲートD・4.5)',
    'thresholds': {'straight_drop_max': 0.05, 'l1_ratio_max': 1.10, 'fwd_bias_min_us': -30.0, 'label_excess_max': 0.10,
                   'reverse_good_max': 0.10, 'spin_ratio': 0.6, 'scrape_g': [1.5, 2.5], 'ci': 0.95, 'n_boot': N_BOOT},
    'primary_output': 'jetformer/drive.py の apply_gains と同じ式で、先読み 3・ステアのゲイン 1.3・速度のゲイン 1.0・'
                      'トリム 0 の出力(各モデルの steer_center_us・throttle_neutral_us の周り)',
}


def run_name(v):
    return rp.run_name(v, EPOCHS, DATASET)


GOOD_OF = {5: 6, 4: 7}             # 「〜・良」のジャンル(policy_sets.json の good_modes と同じ。heldout で確かめる)
# 予測する組み合わせ: 名前 → (学習結果, ジャンルの付け替え)。色・表示名は図で使う
CONFIGS = {
    'B': (BASE_RUN, None),
    'c': (run_name('c'), None),
    'b': (run_name('b'), None),
    'a_good': (run_name('a'), GOOD_OF),
    'a_base': (run_name('a'), None),
    'n_good': (run_name('n'), GOOD_OF),
    'n_base': (run_name('n'), None),
}
LABELS = {'B': 'モデル B(基準)', 'c': '(c) filtered BC', 'b': '(b) AWR', 'a_good': '(a) 印つき・良', 'a_base': '(a) 印つき・元',
          'n_good': '(n) 陰性対照・良', 'n_base': '(n) 陰性対照・元'}
COLORS = {'B': '#52514e', 'c': '#2a78d6', 'b': '#eb6834', 'a_good': '#1baf7a', 'a_base': '#eda100',
          'n_good': '#e87ba4', 'n_base': '#4a3aa7'}
JUDGED = ['a_good', 'b', 'n_good']                 # ゲートC で判定する方策((a) は走行で使う「〜・良」で)
EXTRA = ['a_base', 'n_base']                       # 参考
# レビューの後に足した対照(事後。heldout.json の規則・判定には入れない): (h) = (c) と同じ設定で、学習を人のフレーム
# だけにした微調整(「AI のデータを足した効果」と「同じ設定で 12エポック微調整した効果」を分ける)。tools/rl_review.py が作る
H_RUN = '%s-rl-h-humanft-nocmd-s%d-e%d' % (DATASET, rp.TRAIN['stride'], EPOCHS)
POSTHOC_CONFIGS = {'h': (H_RUN, None)}
LABELS['h'] = '(h) 人だけの微調整(対照)'
COLORS['h'] = '#8a8a86'
CKPTS = ('best', 'last')           # best.pth(検証が最小のエポック)・last.pth(最後のエポック)


def config_of(name):
    """予測の組み合わせ(先に決めた7通りと、事後の対照)→ (学習結果, ジャンルの付け替え)"""
    return CONFIGS[name] if name in CONFIGS else POSTHOC_CONFIGS[name]


def eval_dir(dataset=DATASET):
    return rp.rl_dir(dataset) / 'eval'


def pred_path(name, ckpt='best'):
    """予測のファイル。best.pth は pred_<名前>.npz(前からの名前のまま)、last.pth は pred_<名前>_last.npz"""
    if ckpt not in CKPTS:
        raise ValueError('ckpt は %s のどれか: %s' % (CKPTS, ckpt))
    return eval_dir() / ('pred_%s.npz' % name if ckpt == 'best' else 'pred_%s_last.npz' % name)


def load_state_dict(path):
    """best.pth(state_dict)と last.pth({'model': state_dict, 'optimizer': ...})のどちらからもモデルの重みを返す"""
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model' in sd and 'optimizer' in sd:
        sd = sd['model']
    return sd


def _sha1_file(path, limit=None):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read() if limit is None else f.read(limit))
    return h.hexdigest()


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ---------------------------------------------------------------- numpy だけの規則(テストする)

def drive_outputs(pred, ref, steer_gain=1.3, throttle_gain=1.0, lookahead=3, trim=0.0, lo=1000.0, hi=2000.0):
    """予測 [N,10,2] µs → 走行の出力 [N,2] µs(jetformer/drive.py の apply_gains と同じ式をまとめて計算)。
    ref = (ステアの中心, スロットルの中立)。速度のゲインは前進側だけに掛ける"""
    pred = np.asarray(pred, np.float64)
    k = min(max(int(lookahead), 1), pred.shape[1]) - 1
    raw = pred[:, k, :]
    steer = ref[0] + trim + steer_gain * (raw[:, 0] - ref[0])
    dt = raw[:, 1] - ref[1]
    thr = ref[1] + np.where(dt < 0, throttle_gain * dt, dt)
    return np.stack([np.clip(steer, lo, hi), np.clip(thr, lo, hi)], axis=1)


def heldout_passes(passes, log, t_us, split, lead_s=R1_LEAD_S, de_s=DE_WINDOW_S):
    """保留の通過: 窓のフレームがすべて val にある通過。log・t_us・split はフレームの表(frames.npz)の列。
    戻り値 (r1, de): [{通過の情報..., 'rows': 窓のフレームの行番号}]。窓にフレームが無い通過は入れない"""
    log, t_us, split = np.asarray(log), np.asarray(t_us, np.int64), np.asarray(split)
    by_log = {}
    for lg in np.unique(log):
        m = np.nonzero(log == lg)[0]
        by_log[str(lg)] = m[np.argsort(t_us[m], kind='stable')]
    r1, de = [], []
    for j, p in enumerate(passes):
        rows = by_log.get(p['log'])
        if rows is None:
            continue
        t = (t_us[rows] - p['t0_us']) / 1e6
        w = rows[(t >= p['ta_s'] - lead_s) & (t <= p['ta_s'])]
        if len(w) and np.all(split[w] == 'val'):
            r1.append(dict(p, pass_id=j, rows=w))
        if p.get('tg_s') is not None:
            w = rows[(t >= p['tg_s']) & (t <= p['tg_s'] + de_s)]
            if len(w) and np.all(split[w] == 'val'):
                de.append(dict(p, pass_id=j, rows=w))
    return r1, de


def straight_mask(loc_code, gz_dps, erpm, near_collision, locs=STRAIGHT['locs'],
                  max_yaw=STRAIGHT['max_abs_yaw_dps'], min_fwd=STRAIGHT['min_fwd_erpm']):
    """直線のフレーム: 場所が HS・DE、ヨーレートが小さく、前へ速く走っていて、衝突の前後でない(eRPM は前進が負)"""
    ids = [LOC_NAMES.index(x) for x in locs]
    return (np.isin(np.asarray(loc_code), ids) & (np.abs(np.asarray(gz_dps)) < max_yaw)
            & (-np.asarray(erpm) > min_fwd) & ~np.asarray(near_collision, bool))


def bootstrap_mean(d, n_boot=N_BOOT, seed=0):
    """独立な単位(通過など)の平均と、ブートストラップの 95% 区間 [平均, 下, 上]"""
    d = np.asarray(d, np.float64)
    if len(d) == 0:
        return [None, None, None]
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return [_r(d.mean(), 3), _r(np.percentile(b, 2.5), 3), _r(np.percentile(b, 97.5), 3)]


def cluster_bootstrap(values, cluster, stat, n_boot=N_BOOT, seed=0):
    """クラスタ(シーン・修正の窓)単位のブートストラップ。values = {名前: [N]}、stat(和の dict, 数) → 値
    (numpy の配列のまま計算できる式にする)。戻り値 [点推定, 2.5%, 97.5%]"""
    cluster = np.asarray(cluster)
    ids, inv = np.unique(cluster, return_inverse=True)
    sums = {k: np.bincount(inv, weights=np.asarray(v, np.float64), minlength=len(ids)) for k, v in values.items()}
    cnt = np.bincount(inv, minlength=len(ids)).astype(np.float64)
    point = stat({k: s.sum() for k, s in sums.items()}, cnt.sum())
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = np.asarray(stat({k: s[pick].sum(axis=1) for k, s in sums.items()}, cnt[pick].sum(axis=1)), np.float64)
    boot = boot[np.isfinite(boot)]
    return [_r(point, 4), _r(np.percentile(boot, 2.5), 4) if len(boot) else None,
            _r(np.percentile(boot, 97.5), 4) if len(boot) else None]


def steady_samples(thr_exec, fwd_erpm, ok, group, win=15, max_range_us=20.0):
    """直前 win フレーム(0.5秒)のスロットルの指令がほぼ一定(幅 max_range_us 以内)で、前進しているフレームの
    (窓の平均のスロットル [µs], 前進の eRPM)。group が同じ連続した行だけで窓を作る"""
    thr_exec, fwd_erpm, ok, group = (np.asarray(thr_exec, np.float64), np.asarray(fwd_erpm, np.float64),
                                     np.asarray(ok, bool), np.asarray(group))
    xs, ys = [], []
    for g in np.unique(group):
        idx = np.nonzero(group == g)[0]
        if len(idx) < win:
            continue
        u = thr_exec[idx]
        o = ok[idx]
        for e in range(win - 1, len(idx)):
            s = slice(e - win + 1, e + 1)
            if not o[s].all():
                continue
            w = u[s]
            if w.max() - w.min() <= max_range_us and fwd_erpm[idx[e]] > 0:
                xs.append(w.mean())
                ys.append(fwd_erpm[idx[e]])
    return np.array(xs), np.array(ys)


def fit_erpm_map(thr_us, fwd_erpm, bin_us=10.0, min_n=30):
    """スロットル [µs] → 定常の前進 eRPM の対応表 (x 昇順, y)。ビンごとの中央値を、µs が大きい(遅い)ほど
    小さくなるように単調にする"""
    thr_us, fwd_erpm = np.asarray(thr_us, np.float64), np.asarray(fwd_erpm, np.float64)
    b = np.floor(thr_us / bin_us).astype(int)
    xs, ys = [], []
    for k in np.unique(b):
        m = b == k
        if m.sum() >= min_n:
            xs.append(float(np.median(thr_us[m])))
            ys.append(float(np.median(fwd_erpm[m])))
    x, y = np.array(xs), np.array(ys)
    o = np.argsort(x)
    x, y = x[o], y[o]
    return x, np.minimum.accumulate(y)


def erpm_from_throttle(emap, u):
    """対応表で、スロットル [µs] から定常の前進 eRPM を引く(表の外は端の値)"""
    x, y = emap
    return np.interp(np.asarray(u, np.float64), x, y)


def entry_estimate(entry_actual, u_model, u_actual, emap):
    """見積もった進入 eRPM = 実際の進入 eRPM + 対応表での差(モデルの出力 − 実際に出した指令)。
    その場の速さ(実際の状態)を起点に、指令の違いの分だけずらす(閉ループの影響は入らない、粗い見積もり)"""
    return np.asarray(entry_actual, np.float64) + erpm_from_throttle(emap, u_model) - erpm_from_throttle(emap, u_actual)


def rolling_sum(x, group, win):
    """group が同じ連続した行の、直前 win 行(自分を含む)の和。窓が足りない行は NaN"""
    x = np.asarray(x, np.float64)
    group = np.asarray(group)
    out = np.full(len(x), np.nan)
    for g in np.unique(group):
        idx = np.nonzero(group == g)[0]
        if len(idx) < win:
            continue
        c = np.concatenate([[0.0], np.cumsum(x[idx])])
        out[idx[win - 1:]] = c[win:] - c[:-win]
    return out


def judge_r1(d_thr_ci, share_x, share_ref):
    """R1 の項目: スロットルの差の区間の下限 > 0(遅い)かつ 5500 未満の割合が上がる"""
    return bool(d_thr_ci[1] is not None and d_thr_ci[1] > 0 and share_x is not None and share_ref is not None
                and share_x > share_ref)


def judge_de(d_steer_ci):
    """DE の項目: ステアの差の区間の下限 > 0(左)"""
    return bool(d_steer_ci[1] is not None and d_steer_ci[1] > 0)


def judge_collapse(drop, l1_ratio, fwd_bias, th=RULES['thresholds']):
    """崩れていない: 直線の低下・人の L1 の悪化・前進の偏り(どれも点推定)"""
    parts = {'straight': drop is not None and drop <= th['straight_drop_max'],
             'l1': l1_ratio is not None and l1_ratio <= th['l1_ratio_max'],
             'fwd_bias': fwd_bias is not None and fwd_bias >= th['fwd_bias_min_us']}
    return all(parts.values()), parts


def judge_labels(rows, th=RULES['thresholds']):
    """報酬の悪用: rows = {名前: {'good_share', 'ref_good_share', 'n'}}。reverse は絶対の割合で見る。
    該当するフレームが無い項目は合格(印が付いていない)"""
    parts = {}
    for k, r in rows.items():
        if not r.get('n') or r.get('good_share') is None:
            parts[k] = True
        elif k == 'human_reverse':
            parts[k] = r['good_share'] <= th['reverse_good_max']
        else:
            parts[k] = r['good_share'] - r['ref_good_share'] <= th['label_excess_max']
    return all(parts.values()), parts


# ---------------------------------------------------------------- データの読み込み

def _splits(dataset):
    with open(config.DATASET_DIR / dataset / 'splits.json') as f:
        return json.load(f)


def _aligned(dataset, scene, keys=None):
    with np.load(config.DATASET_DIR / dataset / 'aligned' / (scene + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in (keys or z.files) if k in z.files}


def pred_table(dataset=DATASET, iv_dataset=IV_DATASET):
    """予測するフレームの表(順番も固定): part 0 = ai_mix の val の全シーンの vehicle_valid のフレーム、
    part 1 = intervention_val の採点するフレーム(future_valid & vehicle_valid)"""
    rows = []
    for part, ds, scenes in ((0, dataset, _splits(dataset)['val']), (1, iv_dataset, _splits(iv_dataset)['val'])):
        for s in scenes:
            a = _aligned(ds, s, ['vehicle_valid', 'future_valid'])
            ok = a['vehicle_valid'] & (a['future_valid'] if part == 1 else True)
            for i in np.nonzero(ok)[0]:
                rows.append((part, s, int(i)))
    part = np.array([r[0] for r in rows], np.int8)
    scene = np.array([r[1] for r in rows])
    frame = np.array([r[2] for r in rows], np.int32)
    key = hashlib.sha1(('|'.join('%d:%s:%d' % r for r in rows)).encode()).hexdigest()
    return {'part': part, 'scene': scene, 'frame': frame, 'key': key}


def _load_frames():
    with np.load(rp.rl_dir(DATASET) / 'frames.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'folds_json'}


def _log_of(scene):
    return scene[:15]


# ---------------------------------------------------------------- heldout(CPU。最初に1回だけ)

def cmd_heldout(args):
    out = eval_dir() / 'heldout.json'
    if out.exists():
        print('固定済み(上書きしない): %s' % out)
        return
    sp_mix, sp_clean, sp_iv = _splits(DATASET), _splits(CLEAN_DATASET), _splits(IV_DATASET)
    f = _load_frames()
    with open(rp.rl_dir(DATASET) / 'passes.json') as fh:
        passes = json.load(fh)
    pol = rp._read_json(rp.policy_dir(DATASET) / 'policy_sets.json') or {}
    good = {int(k): int(v) for k, v in (pol.get('good_modes') or {}).items()}
    if good != GOOD_OF:
        raise SystemExit('「〜・良」のジャンルの対応が policy_sets.json と違います: %s' % good)
    # 1) シーン・ログの重なり(学習に使ったものが保留に入っていないか)
    val = set(sp_mix['val'])
    train_mix, train_clean = set(sp_mix['train']), set(sp_clean['train'])
    checks = {'ai_mix_val_in_ai_mix_train': sorted(val & train_mix),
              'ai_mix_val_in_clean_train(B)': sorted(val & train_clean),
              'clean_val_equals_ai_mix_human_val': sorted(set(sp_clean['val'])) == sorted(s for s in val
                                                                                           if _log_of(s) not in AI_VAL_LOGS)}
    # 2) 修正の窓: ログ単位で保留か(4本の方策の学習のログに入っていないか)
    train_logs_mix = {_log_of(s) for s in train_mix}
    train_logs_clean = {_log_of(s) for s in train_clean}
    iv_card = json.load(open(config.DATASET_DIR / IV_DATASET / 'dataset_card.json', encoding='utf-8'))
    windows = []
    for w in iv_card['intervention']['windows']:
        if not w.get('included'):
            continue
        windows.append({'name': w['name'], 'log': w['log'], 'class': w['class'], 'loc': w['loc'], 'scored': w['scored'],
                        'log_heldout_policies': w['log'] not in train_logs_mix,
                        'log_heldout_B': w['log'] not in train_logs_clean})
    # 修正の採点フレームと同じ時刻のフレームを、方策が学習に使ったか(ai_mix の train の future_valid & vehicle_valid)
    used = set()
    for s in sp_mix['train']:
        a = _aligned(DATASET, s, ['timestamps_us', 'future_valid', 'vehicle_valid', 'mode'])
        ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
        used.update(a['timestamps_us'][ok].tolist())
    overlap = 0
    for s in sp_iv['val']:
        a = _aligned(IV_DATASET, s, ['timestamps_us', 'future_valid', 'vehicle_valid'])
        overlap += int(np.isin(a['timestamps_us'][a['future_valid'] & a['vehicle_valid']], list(used)).sum())
    checks['iv_scored_frames_used_in_policy_training'] = overlap
    # 3) 通過
    r1, de = heldout_passes(passes, f['log'], f['timestamps_us'], f['split'])
    strip = lambda ps: [{k: p[k] for k in ('pass_id', 'log', 'ta_s', 'tg_s', 'ai', 'entry_erpm', 'r1_strong', 'r1_collision',  # noqa: E731
                                           'de_collision', 'de_yaw_dps')} | {'frames': int(len(p['rows'])),
                                                                             'src': int(np.round(f['source'][p['rows']].mean()))}
                        for p in ps]
    # 4) 直線
    val_rows = f['split'] == 'val'
    st = straight_mask(f['loc_code'], f['gz'], f['erpm'], f['near_collision']) & val_rows & f['vehicle_valid']
    tab = pred_table()
    rec = {
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'note': '方策の学習(10-11 23時)の前に決まっていた ai_mix の val と intervention_val だけを使う。この表と規則は'
                '予測の前に書き出し、結果を見て入れ替えない',
        'dataset': DATASET, 'iv_dataset': IV_DATASET,
        'val_scenes': sp_mix['val'], 'human_clean_scenes': sp_clean['val'],
        'ai_val_scenes': [s for s in sp_mix['val'] if _log_of(s) in AI_VAL_LOGS],
        'iv_windows': windows,
        'checks': checks,
        'r1_passes': strip(r1), 'de_passes': strip(de),
        'counts': {'r1_passes': len(r1), 'r1_ai': sum(p['ai'] for p in r1), 'de_passes': len(de),
                   'de_ai': sum(p['ai'] for p in de), 'straight_frames': int(st.sum()),
                   'pred_frames': int(len(tab['part'])), 'pred_frames_ai_mix_val': int((tab['part'] == 0).sum()),
                   'pred_frames_iv': int((tab['part'] == 1).sum()),
                   'iv_windows_log_heldout_policies': sum(w['log_heldout_policies'] for w in windows)},
        'pred_table_key': tab['key'],
        'configs': {k: {'run': v[0], 'genre_map': v[1]} for k, v in CONFIGS.items()},
        'rules': RULES, 'drive_settings': DRIVE_SETTINGS,
        'inputs_sha1': {n: _sha1_file(p) for n, p in (
            ('frames.npz', rp.rl_dir(DATASET) / 'frames.npz'), ('passes.json', rp.rl_dir(DATASET) / 'passes.json'),
            ('advantages.npz', rp.rl_dir(DATASET) / 'advantages.npz'),
            ('ai_mix_splits', config.DATASET_DIR / DATASET / 'splits.json'),
            ('iv_splits', config.DATASET_DIR / IV_DATASET / 'splits.json'))},
        'best_pth_sha1_head': {k: _sha1_file(rp.RUNS_DIR / v[0] / 'best.pth', 1 << 20) for k, v in CONFIGS.items()},
    }
    rp._save_json(out, rec)
    print(json.dumps({k: rec[k] for k in ('checks', 'counts')}, ensure_ascii=False, indent=1))
    print('固定しました: %s' % out)


def _heldout():
    p = eval_dir() / 'heldout.json'
    if not p.exists():
        raise SystemExit('先に heldout を実行してください')
    return json.loads(p.read_text(encoding='utf-8'))


# ---------------------------------------------------------------- predict(GPU)

class _FrameSet(object):
    """予測するフレーム(画像・ToF・IMU・車両の値・ジャンル)。torch の Dataset として使う"""

    def __init__(self, tab, use_command=False):
        self.items = []
        self.use_command = use_command
        cache = {}
        for part, s, i in zip(tab['part'], tab['scene'], tab['frame']):
            ds = DATASET if part == 0 else IV_DATASET
            if (ds, s) not in cache:
                cache[(ds, s)] = _aligned(ds, s, ['cam_files', 'tof', 'imu', 'vehicle', 'mode'])
            a = cache[(ds, s)]
            self.items.append((str(config.DATASET_DIR / ds / str(a['cam_files'][i])), a['tof'][i], a['imu'][i],
                               a['vehicle'][i], int(a['mode'][i])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, n):
        import cv2
        import torch
        from jetformer.train_data import normalize_inputs
        path, tof, imu, veh, mode = self.items[n]
        img, t, im, st = normalize_inputs(cv2.imread(path), tof, imu, veh, self.use_command)
        return torch.from_numpy(img), torch.from_numpy(t), torch.from_numpy(im), torch.from_numpy(st), mode


def _genres(modes, genre_map, trained):
    g = np.array([genre_map.get(int(m), int(m)) if genre_map else int(m) for m in modes])
    sub = ~np.isin(g, trained)
    g[sub] = trained[0]                    # 学習していないジャンルは tools/compare_models.py と同じく代用
    return g, sub


def cmd_predict(args):
    held = _heldout()
    tab = pred_table()
    if tab['key'] != held['pred_table_key']:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    names = args.configs or list(CONFIGS)
    ckpt = getattr(args, 'ckpt', 'best')
    todo = [n for n in names if not pred_path(n, ckpt).exists()]
    if not todo:
        print('予測は作成済み')
        return
    rp._check_gpu_free(min_gb=2.0)
    import torch
    from torch.utils.data import DataLoader
    from jetformer.model import JetFormer, PWM_CENTER, PWM_SCALE
    loaders = {}
    for name in todo:
        run, gmap = config_of(name)
        d = rp.RUNS_DIR / run
        cfg = rp._read_json(d / 'model_config.json')
        use_cmd = bool(cfg.get('command_input', True))
        if use_cmd not in loaders:
            loaders[use_cmd] = _FrameSet(tab, use_command=use_cmd)
        fs = loaders[use_cmd]
        trained = [int(x) for x in cfg['trained_modes']]
        genres, sub = _genres([it[4] for it in fs.items], gmap, trained)
        t0 = time.time()
        model = JetFormer(num_modes=cfg['num_modes']).cuda().eval()
        model.load_state_dict(load_state_dict(d / ('%s.pth' % ckpt)))
        dl = DataLoader(fs, batch_size=args.batch, shuffle=False, num_workers=args.workers)
        preds, k = [], 0
        with torch.no_grad():
            for image, tof, imu, state, _ in dl:
                g = torch.from_numpy(genres[k:k + len(image)])
                k += len(image)
                onehot = torch.nn.functional.one_hot(g, cfg['num_modes']).float()
                p = model(image.cuda(), tof.cuda(), imu.cuda(), state.cuda(), onehot.cuda()).float().cpu().numpy()
                preds.append(p * PWM_SCALE + PWM_CENTER)
        pred = np.concatenate(preds).astype(np.float32)
        rp._save_npz(pred_path(name, ckpt), pred=pred, genre=genres.astype(np.int8), substituted=sub,
                     key=np.array(tab['key']), run=np.array(run), ckpt=np.array(ckpt))
        print('%s(%s): %d フレーム %.0f秒(ジャンルの代用 %d)' % (name, ckpt, len(pred), time.time() - t0, int(sub.sum())),
              flush=True)
        del model
        torch.cuda.empty_cache()


def load_preds(names=None, ckpt='best'):
    held = _heldout()
    out = {}
    for n in names or list(CONFIGS):
        p = pred_path(n, ckpt)
        if not p.exists():
            raise SystemExit('予測がありません: %s(predict を実行)' % p)
        with np.load(p) as z:
            if str(z['key']) != held['pred_table_key']:
                raise SystemExit('%s は別のフレームの表で作られています' % p)
            out[n] = z['pred'].astype(np.float64)
    return out


# ---------------------------------------------------------------- crosscheck(GPU。compare_models.py と同じか)

def cmd_crosscheck(args):
    out = eval_dir() / 'crosscheck.json'
    if out.exists():
        print('作成済み: %s' % out)
        return
    rp._check_gpu_free(min_gb=2.0)
    import torch
    import compare_models as cm
    tab = pred_table()
    preds = load_preds(['B', 'c'])
    scene = args.scene
    root = config.DATASET_DIR / DATASET
    a = _aligned(DATASET, scene)
    m = (tab['part'] == 0) & (tab['scene'] == scene)
    fr = tab['frame'][m]
    res = {'scene': scene, 'frames': int(m.sum())}
    for name in ('B', 'c'):
        model, cfg, trained, _ = cm.load_model(CONFIGS[name][0])
        t0 = time.time()
        tf, sf, sub = cm.predict_scene(model, cfg, trained, root, a)
        res[name] = {'max_abs_tf_vs_cached_us': _r(np.abs(tf[fr] - preds[name][m]).max(), 3),
                     'max_abs_sf_vs_cached_us': _r(np.abs(sf[fr] - preds[name][m]).max(), 3),
                     'max_abs_tf_vs_sf_us': _r(np.nanmax(np.abs(tf - sf)), 3), 'seconds': _r(time.time() - t0, 1)}
        print(name, res[name], flush=True)
        del model
        torch.cuda.empty_cache()
    rp._save_json(out, res)


# ---------------------------------------------------------------- evaluate(CPU)

def _val_arrays(tab):
    """part 0(ai_mix の val)の予測の行ごとの値: 正解・今の値・衝突の前後・場所など"""
    f = _load_frames()
    idx = {(s, int(i)): k for k, (s, i) in enumerate(zip(f['scene'], f['frame']))}
    n = len(tab['part'])
    cols = {'y': np.full((n, 10, 2), np.nan), 'cur': np.full((n, 2), np.nan), 'fv': np.zeros(n, bool),
            'impact': np.zeros(n, bool), 'mode': np.zeros(n, int), 'source': np.full(n, -1, int),
            'exec_thr': np.full(n, np.nan), 'exec_steer': np.full(n, np.nan), 'frow': np.full(n, -1, int)}
    from jetformer.train_data import impact_mask
    for part, ds in ((0, DATASET), (1, IV_DATASET)):
        for s in np.unique(tab['scene'][tab['part'] == part]):
            a = _aligned(ds, s)
            m = np.nonzero((tab['part'] == part) & (tab['scene'] == s))[0]
            i = tab['frame'][m]
            imp = impact_mask(a['timestamps_us'], a['imu'], IMPACT_G)
            cols['y'][m] = a['future_actions'][i]
            cols['cur'][m] = a['vehicle'][i, :2]
            cols['fv'][m] = a['future_valid'][i] & a['vehicle_valid'][i]
            cols['impact'][m] = imp[i]
            cols['mode'][m] = a['mode'][i]
            cols['source'][m] = a['source'][i]
            cols['exec_steer'][m] = a['action_exec'][i, 0]
            cols['exec_thr'][m] = a['action_exec'][i, 1]
            if part == 0:
                cols['frow'][m] = [idx[(s, int(x))] for x in i]
    return cols, f


def _ref_of(name):
    cfg = rp._read_json(rp.RUNS_DIR / config_of(name)[0] / 'model_config.json')
    return float(cfg['steer_center_us']), float(cfg['throttle_neutral_us'])


def _erpm_map(f):
    """スロットル → 定常の前進 eRPM の対応表(ai_mix の train のフレーム。指令は実際に車に出た値 action_exec)"""
    sp = _splits(DATASET)['train']
    thr, erpm, ok, grp = [], [], [], []
    for k, s in enumerate(sp):
        a = _aligned(DATASET, s, ['action_exec', 'vehicle', 'vehicle_valid', 'near_collision', 'label_valid'])
        thr.append(a['action_exec'][:, 1])
        erpm.append(-a['vehicle'][:, 3])
        ok.append(a['vehicle_valid'] & ~a['near_collision'] & a['label_valid'] & np.isfinite(a['action_exec'][:, 1]))
        grp.append(np.full(len(a['vehicle']), k))
    x, y = steady_samples(np.concatenate(thr), np.concatenate(erpm), np.concatenate(ok), np.concatenate(grp))
    emap = fit_erpm_map(x, y)
    return emap, {'samples': int(len(x)), 'x_us': [round(v, 1) for v in emap[0].tolist()],
                  'y_erpm': [round(v) for v in emap[1].tolist()],
                  'rule': 'ai_mix の train のフレームで、直前 0.5秒の指令(action_exec)の幅が 20µs 以内のときの '
                          '(平均の指令, 前進 eRPM)。10µs ごとの中央値(30個以上)を単調にしたもの'}


def _pass_values(ps, prow, outs, cols):
    """通過ごとの窓の平均: 各モデルの出力(スロットル・ステア)、実際に出した指令"""
    res = []
    for p in ps:
        r = prow[p['rows']]
        r = r[r >= 0]
        if not len(r):
            continue
        v = {'pass_id': p['pass_id'], 'ai': p['ai'], 'log': p['log'], 'entry_erpm': p['entry_erpm'],
             'de_yaw_dps': p.get('de_yaw_dps'), 'r1_strong': p['r1_strong'], 'de_collision': p['de_collision'],
             'frames': int(len(r)), 'exec_thr': float(np.nanmean(cols['exec_thr'][r])),
             'exec_steer': float(np.nanmean(cols['exec_steer'][r]))}
        for n, o in outs.items():
            v['thr_' + n] = float(o[r, 1].mean())
            v['steer_' + n] = float(o[r, 0].mean())
        res.append(v)
    return res


def _paired(pv, key, x, ref, sel=None):
    d = np.array([q['%s_%s' % (key, x)] - q['%s_%s' % (key, ref)] for q in pv if sel is None or sel(q)])
    return bootstrap_mean(d), len(d)


def evaluate_all(preds=None):
    """ゲートC の各項目と走行の精度。preds = {名前: 予測}(省略すると best.pth の予測。レビューでは last.pth の予測を
    入れて、チェックポイントで結論が変わるかを見る)"""
    held = _heldout()
    tab = pred_table()
    preds = load_preds() if preds is None else preds
    cols, f = _val_arrays(tab)
    names = list(CONFIGS)
    refs = {n: _ref_of(n) for n in names}
    outs = {n: drive_outputs(preds[n], refs[n], DRIVE_SETTINGS['steer_gain'], DRIVE_SETTINGS['throttle_gain'],
                             DRIVE_SETTINGS['lookahead'], DRIVE_SETTINGS['steer_trim_us']) for n in names}
    # frames.npz の行 → 予測の行
    prow = np.full(len(f['scene']), -1, int)
    m0 = tab['part'] == 0
    prow[cols['frow'][m0]] = np.nonzero(m0)[0]
    with open(rp.rl_dir(DATASET) / 'passes.json') as fh:
        passes = json.load(fh)
    r1, de = heldout_passes(passes, f['log'], f['timestamps_us'], f['split'])
    if len(r1) != held['counts']['r1_passes'] or len(de) != held['counts']['de_passes']:
        raise SystemExit('保留の通過の数が heldout.json と違います')
    res = {'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'heldout_created_at': held['created_at'],
           'configs': {n: {'run': CONFIGS[n][0], 'genre_map': CONFIGS[n][1], 'ref_us': refs[n], 'label': LABELS[n]}
                       for n in names}, 'drive_settings': DRIVE_SETTINGS, 'rules': RULES}
    # ---- ゲートC 1(価値): その1の結果
    ve = rp._read_json(rp.rl_dir(DATASET) / 'value_eval.json') or {}
    res['item1_value'] = dict(ve.get('gate_c1') or {}, source='data/rl/ai_mix_20261011/value_eval.json(value.md)')
    # ---- R1
    emap, emap_info = _erpm_map(f)
    pv = _pass_values(r1, prow, outs, cols)
    ent = np.array([q['entry_erpm'] for q in pv], np.float64)
    u_act = np.array([q['exec_thr'] for q in pv])
    fit_act = erpm_from_throttle(emap, u_act)
    emap_info['validation_heldout_r1'] = {
        'passes': len(pv), 'spearman_map_vs_entry': _r(_spearman(fit_act, ent), 3),
        'mae_erpm': _r(np.mean(np.abs(fit_act - ent)), 0), 'bias_erpm': _r(np.mean(fit_act - ent), 0),
        'note': '実際に出した指令の窓の平均を対応表に入れた値と、実際の進入 eRPM の比較(対応表の当てはまりの目安)'}
    for q in pv:
        for n in names:
            q['entry_est_' + n] = float(entry_estimate(q['entry_erpm'], q['thr_' + n], q['exec_thr'], emap))
    item2 = {'passes': len(pv), 'ai_passes': sum(q['ai'] for q in pv), 'per_config': {}, 'vs': {}}
    for n in names:
        e = np.array([q['entry_est_' + n] for q in pv])
        item2['per_config'][n] = {'thr_out_us_mean': _r(np.mean([q['thr_' + n] for q in pv]), 1),
                                  'entry_est_median': _r(np.median(e), 0),
                                  'share_lt5500': _r(np.mean(e < ENTRY_BAND_ERPM), 3),
                                  'share_lt5500_ai': _r(np.mean(e[[q['ai'] for q in pv]] < ENTRY_BAND_ERPM), 3)
                                  if any(q['ai'] for q in pv) else None}
    item2['actual'] = {'exec_thr_us_mean': _r(np.mean(u_act), 1), 'entry_erpm_median': _r(np.median(ent), 0),
                       'share_lt5500': _r(np.mean(ent < ENTRY_BAND_ERPM), 3)}
    for x, ref in [(x, 'c') for x in JUDGED + EXTRA] + [(x, 'B') for x in ['c'] + JUDGED + EXTRA]:
        ci, nn = _paired(pv, 'thr', x, ref)
        ci_ai, n_ai = _paired(pv, 'thr', x, ref, lambda q: q['ai'])
        ci_hu, n_hu = _paired(pv, 'thr', x, ref, lambda q: not q['ai'])
        lt_x = np.array([q['entry_est_' + x] < ENTRY_BAND_ERPM for q in pv], float)
        lt_r = np.array([q['entry_est_' + ref] < ENTRY_BAND_ERPM for q in pv], float)
        sh = item2['per_config']
        item2['vs']['%s-%s' % (x, ref)] = {
            'd_thr_us': ci, 'd_thr_us_ai': ci_ai, 'd_thr_us_human': ci_hu, 'n': nn, 'n_ai': n_ai, 'n_human': n_hu,
            'd_share_lt5500': bootstrap_mean(lt_x - lt_r),
            'pass': judge_r1(ci, sh[x]['share_lt5500'], sh[ref]['share_lt5500'])}
    res['item2_r1'] = item2
    res['erpm_map'] = emap_info
    res['r1_pass_values'] = [{k: (_r(v, 1) if isinstance(v, float) else v) for k, v in q.items()} for q in pv]
    # ---- DE
    pd_ = _pass_values(de, prow, outs, cols)
    item3 = {'passes': len(pd_), 'ai_passes': sum(q['ai'] for q in pd_), 'per_config': {}, 'vs': {}}
    for n in names:
        item3['per_config'][n] = {'steer_out_us_mean': _r(np.mean([q['steer_' + n] for q in pd_]), 1)}
    item3['actual'] = {'exec_steer_us_mean': _r(np.mean([q['exec_steer'] for q in pd_]), 1)}
    for x, ref in [(x, 'c') for x in JUDGED + EXTRA] + [(x, 'B') for x in ['c'] + JUDGED + EXTRA]:
        ci, nn = _paired(pd_, 'steer', x, ref)
        ci_ai, n_ai = _paired(pd_, 'steer', x, ref, lambda q: q['ai'])
        ci_hu, n_hu = _paired(pd_, 'steer', x, ref, lambda q: not q['ai'])
        item3['vs']['%s-%s' % (x, ref)] = {'d_steer_us': ci, 'd_steer_us_ai': ci_ai, 'd_steer_us_human': ci_hu,
                                           'n': nn, 'n_ai': n_ai, 'n_human': n_hu, 'pass': judge_de(ci)}
    res['item3_de'] = item3
    res['de_pass_values'] = [{k: (_r(v, 1) if isinstance(v, float) else v) for k, v in q.items()} for q in pd_]
    # ---- 崩れていない(直線・人のきれいな走行の L1・前進の偏り)
    fr = cols['frow']
    stm = np.zeros(len(fr), bool)
    stm[m0] = straight_mask(f['loc_code'][fr[m0]], f['gz'][fr[m0]], f['erpm'][fr[m0]], f['near_collision'][fr[m0]])
    scene_id = np.unique(tab['scene'], return_inverse=True)[1]
    human_scene = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    clean = human_scene & cols['fv'] & ~cols['impact']
    import compare_models as cm
    l1 = {n: np.abs(preds[n] - np.nan_to_num(cols['y'])).mean(axis=(1, 2)) for n in names}
    item4 = {'straight_frames': int(stm.sum()), 'straight_frames_ai': int((stm & (cols['source'] == 1)).sum()),
             'human_clean_frames': int(clean.sum()), 'per_config': {}, 'vs': {}}
    for n in names:
        fwd = NEUTRAL_US - outs[n][stm, 1]
        met = cm.metrics(preds[n][clean], cols['y'][clean], cols['cur'][clean], refs[n][0], refs[n][1])
        item4['per_config'][n] = {'straight_fwd_strength_us': _r(fwd.mean(), 1),
                                  'straight_thr_out_us': _r(outs[n][stm, 1].mean(), 1),
                                  'human_clean_l1_us': _r(l1[n][clean].mean(), 2),
                                  'fwd_bias_us': met['fwd_bias_us'],
                                  'fwd_bias_ci': cluster_bootstrap({'d': preds[n][clean, 2, 1] - cols['y'][clean, 2, 1]},
                                                                   scene_id[clean], lambda s, c: s['d'] / c)}
    for x, ref in [(x, 'B') for x in ['c'] + JUDGED + EXTRA] + [(x, 'c') for x in JUDGED + EXTRA]:
        fx, fr_ = NEUTRAL_US - outs[x][stm, 1], NEUTRAL_US - outs[ref][stm, 1]
        drop = cluster_bootstrap({'x': fx, 'r': fr_}, scene_id[stm], lambda s, c: 1 - s['x'] / s['r'])
        ratio = cluster_bootstrap({'x': l1[x][clean], 'r': l1[ref][clean]}, scene_id[clean], lambda s, c: s['x'] / s['r'])
        ok, parts = judge_collapse(drop[0], ratio[0], item4['per_config'][x]['fwd_bias_us'])
        item4['vs']['%s-%s' % (x, ref)] = {'straight_drop': drop, 'l1_ratio': ratio, 'pass': ok, 'parts': parts}
    res['item4_collapse'] = item4
    # ---- 報酬の悪用(学習の印の確認)
    res['item5_labels'] = label_audit(f)
    # ---- まとめ(ゲートC)
    res['verdict'] = verdict(res)
    # ---- 走行の精度(人との一致)
    res['accuracy'] = accuracy(preds, cols, tab, held, refs, scene_id, l1)
    res['candidate'] = candidate(res)
    # ---- 事後の記述(結果を見た後に足した。判定には使わない): 印のジャンルが出力をどれだけ動かすか
    post = {'note': '結果を見た後に足した記述の分析。判定(heldout.json の規則)には使わない'}
    for x, ref in (('a_good', 'a_base'), ('n_good', 'n_base')):
        post['%s-%s' % (x, ref)] = {
            'r1_d_thr_us': _paired(pv, 'thr', x, ref)[0], 'de_d_steer_us': _paired(pd_, 'steer', x, ref)[0],
            'all_val_mean_abs_d_out_us': [_r(np.abs(outs[x][m0, 0] - outs[ref][m0, 0]).mean(), 2),
                                          _r(np.abs(outs[x][m0, 1] - outs[ref][m0, 1]).mean(), 2)],
            'all_val_mean_d_out_us': [_r((outs[x][m0, 0] - outs[ref][m0, 0]).mean(), 2),
                                      _r((outs[x][m0, 1] - outs[ref][m0, 1]).mean(), 2)]}
    ai_de = [q for q in pd_ if q['ai']]
    post['de_ai_passes'] = [{'log': q['log'], 'de_yaw_dps': q['de_yaw_dps'], 'de_collision': q['de_collision'],
                             'exec_steer': _r(q['exec_steer'], 1), 'B': _r(q['steer_B'], 1), 'c': _r(q['steer_c'], 1)}
                            for q in ai_de]
    res['posthoc'] = post
    cc = rp._read_json(eval_dir() / 'crosscheck.json')
    if cc:
        res['crosscheck'] = cc
    ec = rp._read_json(eval_dir() / 'engine_check.json')
    if ec:
        res['engine_check'] = ec
    return res


def _spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 3:
        return np.nan
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def label_masks(f):
    """項目5 で使うフレームの集合(label_audit と、レビューの事後の確認 tools/rl_review.py で共通)。
    戻り値 (adv, masks)。masks: elig・spin・ref_spin・scrape・moving・reverse・has_track([N] bool)"""
    with np.load(rp.rl_dir(DATASET) / 'advantages.npz', allow_pickle=False) as z:
        adv = {k: z[k] for k in ('eligible', 'A', 'I', 'I_neg')}
    elig = adv['eligible'] & np.isfinite(adv['A']) & (adv['I'] >= 0)
    src = f['source']
    moving = -f['erpm'] > 1000
    # (i) 空転: コース図の当てはめがあるログで、0.5秒の窓のコース上の進み / 車輪の距離
    import offline_rl as orl
    ratio = np.full(len(src), np.nan)
    fwd_win = np.full(len(src), np.nan)
    has_track = np.zeros(len(src), bool)
    for lg in np.unique(f['log']):
        if not (Path(orl.TRACKS_DIR) / ('tracks_%s.npz' % lg)).exists():
            continue
        _, lab = orl._labels(lg)
        _, dprog = orl.track_arrays(lg, lab['t_us'])
        rows = np.nonzero((f['log'] == lg) & (f['label_index'] >= 0))[0]
        rows = rows[np.lexsort((f['frame'][rows], f['scene'][rows]))]
        dp = dprog[f['label_index'][rows]]
        fw = f['progress_fwd_m'][rows].astype(np.float64)
        grp = f['scene'][rows]
        s_dp = rolling_sum(dp, grp, 15)              # 窓に当てはめの無い行があれば NaN
        s_fw = rolling_sum(fw, grp, 15)
        ratio[rows] = s_dp / np.where(s_fw > 0, s_fw, np.nan)
        fwd_win[rows] = s_fw
        has_track[rows] = np.isfinite(s_dp)
    th = RULES['thresholds']
    spin = has_track & (fwd_win > 0.5) & (ratio < th['spin_ratio'])
    # (ii) 2.5G 未満の強い衝撃(水平 1.5〜2.5G)
    hg = np.zeros(len(src))
    for s in np.unique(f['scene']):
        rows = np.nonzero(f['scene'] == s)[0]
        a = _aligned(DATASET, s, ['imu'])
        acc = a['imu'][f['frame'][rows], :, :2]
        hg[rows] = np.hypot(acc[..., 0], acc[..., 1]).max(axis=1) / 9.80665
    scrape = (hg >= th['scrape_g'][0]) & (hg < th['scrape_g'][1]) & ~f['near_collision'] & moving
    reverse = (src == 0) & (f['erpm'] > 300)
    ref_spin = elig & has_track & (fwd_win > 0.5)
    return adv, {'elig': elig, 'spin': spin, 'ref_spin': ref_spin, 'scrape': scrape, 'moving': moving,
                 'reverse': reverse, 'has_track': has_track}


def label_audit(f):
    """学習に使った印(advantages.npz の I・I_neg)の確認(ロードマップ 4.4 の5)。印は学習データそのものなので、
    保留ではなく使えるフレーム全体で見る"""
    adv, mk = label_masks(f)
    elig, spin, ref_spin, scrape, moving, reverse, has_track = (mk[k] for k in (
        'elig', 'spin', 'ref_spin', 'scrape', 'moving', 'reverse', 'has_track'))
    src = f['source']
    out = {'eligible_frames': int(elig.sum()),
           'spin_frames': {'track_logs': sorted(set(f['log'][has_track].tolist())), 'n_all': int(spin.sum())},
           'labels': {}}
    for lab_name in ('I', 'I_neg'):
        I = adv[lab_name]
        rows = {}
        for name, m, ref in (('spin', spin, ref_spin), ('scrape', scrape, moving), ('human_reverse', reverse, src == 0)):
            mm, rr = elig & m, elig & ref
            rows[name] = {'n': int(mm.sum()), 'good_share': _r(np.mean(I[mm] == 1), 3) if mm.any() else None,
                          'ref_n': int(rr.sum()), 'ref_good_share': _r(np.mean(I[rr] == 1), 3) if rr.any() else None,
                          'by_loc': {LOC_NAMES[k]: int((mm & (f['loc_code'] == k)).sum()) for k in range(len(LOC_NAMES))
                                     if (mm & (f['loc_code'] == k)).any()}}
        # 参考: AI の GM(芝)の「良い」の割合と全体
        gm = elig & (src == 1) & (f['loc_code'] == LOC_NAMES.index('GM'))
        rows_ok, parts = judge_labels(rows)
        out['labels'][lab_name] = {'rows': rows, 'pass': rows_ok, 'parts': parts,
                                   'ai_gm_good_share': _r(np.mean(I[gm] == 1), 3),
                                   'ai_good_share': _r(np.mean(I[elig & (src == 1)] == 1), 3)}
    out['note'] = ('空転 = コース図の当てはめがあるログで 0.5秒の窓の(コース上の進み / 車輪の距離)< 0.6 かつ車輪の距離 > 0.5m。'
                   '比べる集団 = 同じログの車輪の距離 > 0.5m の使えるフレーム。強い衝撃 = 水平 1.5〜2.5G・衝突の前後でない・'
                   '前進 eRPM > 1000。比べる集団 = 前進 eRPM > 1000 の使えるフレーム。人の後退 = 人のフレームで eRPM > 300')
    return out


def verdict(res):
    """ゲートC の項目ごとの合否((a) は「〜・良」、(b)、(n) は「〜・良」)と、陰性対照が落ちたか"""
    v1 = bool(res['item1_value'].get('pass'))
    lab = res['item5_labels']['labels']
    table = {}
    for x in JUDGED + EXTRA:
        lab_name = 'I_neg' if x.startswith('n') else 'I'
        row = {'1_value': v1,
               '2_r1': res['item2_r1']['vs']['%s-c' % x]['pass'],
               '3_de': res['item3_de']['vs']['%s-c' % x]['pass'],
               '4_collapse': res['item4_collapse']['vs']['%s-B' % x]['pass'],
               '5_labels': lab[lab_name]['pass']}
        row['all'] = all(row.values())
        table[x] = row
    n = table['n_good']
    policy_fail = [k for k in ('2_r1', '3_de', '4_collapse') if not n[k]]
    # 見分けた項目: (n) が落ち、(a) か (b) が通った項目。(n) が落ちても (a)(b) も同じく落ちたなら、見分けていない
    # (2026-10-12 のレビューで直した。前は「(n) が 2〜4 のどれかで落ちた」だけで真にしていた)
    separating = [k for k in ('2_r1', '3_de', '4_collapse', '5_labels')
                  if not n[k] and any(table[x][k] for x in ('a_good', 'b'))]
    return {'table': table,
            'negative_control_caught_by_policy_items': policy_fail,
            'negative_control_caught_by_labels': not n['5_labels'],
            'negative_control_rule_met': bool(policy_fail),
            'separating_items': separating,
            'gate_discriminates': bool(separating),
            'note': '1 はすべての印・重みの方策で共通(価値の AUROC 0.41 で不合格)。(c) は印を使わないので 1・5 は対象外。'
                    'negative_control_rule_met = 先に決めた規則(陰性対照が方策の出力の項目 2〜4 のどれかで落ちる)。'
                    'gate_discriminates = (n) が落ち、かつ (a) か (b) が通った項目があるか(見分けたか)'}


def accuracy(preds, cols, tab, held, refs, scene_id, l1):
    """走行の精度 = 人の操作との一致(tools/compare_models.py の metrics と同じ定義)。保留した人のきれいな走行と、
    交代の後の人の修正(intervention_val)"""
    import compare_models as cm
    names = list(CONFIGS)
    m0 = tab['part'] == 0
    human = np.isin(tab['scene'], held['human_clean_scenes']) & m0
    ai_val = np.isin(tab['scene'], held['ai_val_scenes']) & m0
    iv = tab['part'] == 1
    win_of = {}
    for w in held['iv_windows']:
        win_of[w['name']] = w
    iv_heldout = iv & np.array([win_of.get(s, {}).get('log_heldout_policies', False) for s in tab['scene']])
    iv_class = np.array([win_of.get(s, {}).get('class', '') for s in tab['scene']])
    sets = {'human_clean': human & cols['fv'] & ~cols['impact'],
            'human_clean_genre4': human & cols['fv'] & ~cols['impact'] & (cols['mode'] == 4),
            'human_clean_genre5': human & cols['fv'] & ~cols['impact'] & (cols['mode'] == 5),
            'human_all': human & cols['fv'],
            'ai_val_clean_ref': ai_val & cols['fv'] & ~cols['impact'] & (cols['source'] == 1),
            'iv_all': iv & cols['fv'],
            'iv_log_heldout': iv_heldout & cols['fv'],
            'iv_train_logs': iv & ~iv_heldout & cols['fv']}
    for c in ('after_collision', 'stuck', 'preemptive'):
        sets['iv_' + c] = iv & cols['fv'] & (iv_class == c)
    out = {'sets': {k: int(v.sum()) for k, v in sets.items()},
           'note': 'MAE・向きの正解率・前進の偏りは tools/compare_models.py の metrics(教師あり = 自己入力。今の操作値を'
                   '入れないモデル)。ai_val_clean_ref は AI 自身の出力(ゲイン前)との一致で、精度ではない(参考)',
           'metrics': {}, 'diffs': {}}
    for n in names:
        out['metrics'][n] = {}
        for k, m in sets.items():
            if m.any():
                r = cm.metrics(preds[n][m], cols['y'][m], cols['cur'][m], refs[n][0], refs[n][1])
                r['l1_us'] = _r(l1[n][m].mean(), 2)
                out['metrics'][n][k] = r
    for k in ('human_clean', 'iv_all', 'iv_log_heldout', 'iv_train_logs'):
        m = sets[k]
        cl = scene_id[m] if k.startswith('human') else tab['scene'][m]
        for x, ref in [(x, 'B') for x in names if x != 'B'] + [(x, 'c') for x in JUDGED + EXTRA]:
            out['diffs']['%s-%s/%s' % (x, ref, k)] = {
                'l1': cluster_bootstrap({'d': l1[x][m] - l1[ref][m]}, cl, lambda s, c: s['d'] / c),
                'steer_mae': cluster_bootstrap({'d': np.abs(preds[x][m, :, 0] - cols['y'][m, :, 0]).mean(1)
                                                - np.abs(preds[ref][m, :, 0] - cols['y'][m, :, 0]).mean(1)},
                                               cl, lambda s, c: s['d'] / c),
                'throttle_mae': cluster_bootstrap({'d': np.abs(preds[x][m, :, 1] - cols['y'][m, :, 1]).mean(1)
                                                   - np.abs(preds[ref][m, :, 1] - cols['y'][m, :, 1]).mean(1)},
                                                  cl, lambda s, c: s['d'] / c)}
    return out


def candidate(res):
    """実車の候補の判定(heldout.json の規則)"""
    v = res['verdict']
    rl = [x for x in ('a_good', 'b') if v['table'][x]['all'] and v['negative_control_rule_met']]
    r1 = res['item2_r1']['vs']['c-B']
    de = res['item3_de']['vs']['c-B']
    col = res['item4_collapse']['vs']['c-B']
    better = (r1['d_thr_us'][1] is not None and r1['d_thr_us'][1] > 0) or (de['d_steer_us'][1] is not None
                                                                          and de['d_steer_us'][1] > 0)
    worse = (r1['d_thr_us'][2] is not None and r1['d_thr_us'][2] < 0) or (de['d_steer_us'][2] is not None
                                                                         and de['d_steer_us'][2] < 0)
    c_ok = bool(col['pass'] and better and not worse)
    return {'rl_candidates': rl, 'c_candidate': c_ok,
            'c_checks': {'collapse_vs_B': col['pass'], 'r1_or_de_better_than_B': bool(better),
                         'r1_or_de_worse_than_B': bool(worse)},
            'decision': (rl[0] if rl else ('c' if c_ok else None))}


def cmd_evaluate(args):
    res = evaluate_all()
    rp._save_json(eval_dir() / 'evaluation.json', res)
    if args.report_dir:
        rp._save_json(Path(args.report_dir) / 'evaluation.json', res)
    v = res['verdict']
    print(json.dumps({'verdict': v, 'candidate': res['candidate']}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- engine-check(GPU。走行のコードと同じ手順)

def cmd_engine_check(args):
    """jetformer/drive.py の Driver._run と同じ手順(model_info → ジャンルが trained_modes にあるか → ref →
    TRTRunner → 0 の入力で3回の空回し → JPEG を作り直した画像で normalize_inputs → apply_gains)でエンジンを読み、
    保留のフレームで、キャッシュした PyTorch の予測から計算した出力と比べる。PCA9685 には触らない(走らせない)"""
    import cv2
    rp._check_gpu_free(min_gb=2.0)
    import torch
    from jetformer import drive
    from jetformer.train_data import normalize_inputs
    from jetformer.trt_runtime import TRTRunner
    tab = pred_table()
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(np.nonzero(tab['part'] == 0)[0], size=args.samples, replace=False))
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, config.CAMERA['jpeg_quality']]
    out = rp._read_json(eval_dir() / 'engine_check.json') or {}
    for spec in args.runs:
        name, genre = spec.split(':')
        genre = int(genre)
        run = CONFIGS[name][0]
        key = '%s:%d' % (name, genre)
        info = drive.model_info(run)
        row = {'run': run, 'genre': genre, 'ui_model_info': info is not None}
        if info is None:
            out[key] = row
            continue
        row['trained_modes'] = info['trained_modes']
        row['genre_in_trained_modes'] = genre in info['trained_modes']
        row['ref_us'] = [info['steer_center_us'], info['throttle_neutral_us']]
        row['command_input'] = info['command_input']
        ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
        torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        t0 = time.time()
        runner = TRTRunner(str(drive.RUNS_DIR / run / drive.ENGINE_FILE))
        row['load_s'] = _r(time.time() - t0, 2)
        nm = info['num_modes']
        zeros = {'image': np.zeros((1, 3, 224, 224), np.float32), 'tof': np.zeros((1, 2, 42, 54), np.float32),
                 'imu': np.zeros((1, drive.IMU_STEPS, 6), np.float32), 'state': np.zeros((1, 3), np.float32),
                 'mode': np.zeros((1, nm), np.float32)}
        for _ in range(3):
            runner.run_numpy(**zeros)
        oh = np.zeros((1, nm), np.float32)
        oh[0, genre] = 1.0
        p = {'steer_gain': DRIVE_SETTINGS['steer_gain'], 'throttle_gain': DRIVE_SETTINGS['throttle_gain'],
             'steer_trim_us': DRIVE_SETTINGS['steer_trim_us'], 'lookahead': DRIVE_SETTINGS['lookahead']}
        diffs, ms, outs = [], [], []
        # 比べる相手: 同じ組み合わせのキャッシュの予測(ジャンルが同じフレームだけ)
        with np.load(eval_dir() / ('pred_%s.npz' % name)) as z:
            cpred, cgen = z['pred'], z['genre']
        for j in pick:
            s, i = str(tab['scene'][j]), int(tab['frame'][j])
            a = _aligned(DATASET, s, ['cam_files', 'tof', 'imu', 'vehicle'])
            img = cv2.imread(str(config.DATASET_DIR / DATASET / str(a['cam_files'][i])))
            ok, enc = cv2.imencode('.jpg', img, jpeg)
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
            x = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], info['command_input'])
            t1 = time.perf_counter()
            res = runner.run_numpy(image=x[0][None], tof=x[1][None], imu=x[2][None], state=x[3][None], mode=oh)
            ms.append((time.perf_counter() - t1) * 1000)
            pred = res['actions'][0] * drive.PWM_SCALE + drive.PWM_CENTER
            k = min(max(p['lookahead'], 1), len(pred)) - 1
            o = drive.apply_gains((float(pred[k, 0]), float(pred[k, 1])), ref, p, config.DRIVE['min_us'],
                                  config.DRIVE['max_us'])
            outs.append(o)
            if int(cgen[j]) == genre:
                c = drive_outputs(cpred[j:j + 1].astype(np.float64), ref, p['steer_gain'], p['throttle_gain'],
                                  p['lookahead'], p['steer_trim_us'])[0]
                diffs.append(float(np.abs(np.array(o) - c).max()))
        row.update({'samples': len(pick), 'compared': len(diffs),
                    'out_vs_cached_torch_max_us': _r(max(diffs), 2) if diffs else None,
                    'out_vs_cached_torch_median_us': _r(float(np.median(diffs)), 2) if diffs else None,
                    'infer_ms_median': _r(float(np.median(ms)), 2),
                    'out_steer_us_range': [_r(min(o[0] for o in outs), 0), _r(max(o[0] for o in outs), 0)],
                    'out_throttle_us_range': [_r(min(o[1] for o in outs), 0), _r(max(o[1] for o in outs), 0)],
                    'checked_at': datetime.datetime.now().isoformat(timespec='seconds')})
        out[key] = row
        print(key, row, flush=True)
        del runner
        torch.cuda.empty_cache()
    rp._save_json(eval_dir() / 'engine_check.json', out)


# ---------------------------------------------------------------- figures(CPU)

def cmd_figures(args):
    import figures_rl_evaluate as fig
    res = rp._read_json(eval_dir() / 'evaluation.json')
    if not res:
        raise SystemExit('先に evaluate を実行してください')
    fig.make_all(res, Path(args.fig_dir))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('heldout')
    p = sub.add_parser('predict')
    p.add_argument('--configs', nargs='*', choices=list(CONFIGS) + list(POSTHOC_CONFIGS))
    p.add_argument('--ckpt', choices=CKPTS, default='best', help='best.pth(既定)か last.pth(最後のエポック)で予測する')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=3)
    p = sub.add_parser('crosscheck')
    p.add_argument('--scene', default='20261010-202633-0003')
    p = sub.add_parser('evaluate')
    p.add_argument('--report-dir', default='')
    p = sub.add_parser('figures')
    p.add_argument('--fig-dir', required=True)
    p = sub.add_parser('engine-check')
    p.add_argument('--runs', nargs='+', default=['B:5', 'c:5', 'b:5', 'a_good:6'],
                   help='名前:ジャンル(CONFIGS の名前。走行で選ぶジャンル)')
    p.add_argument('--samples', type=int, default=32)
    args = ap.parse_args()
    {'heldout': cmd_heldout, 'predict': cmd_predict, 'crosscheck': cmd_crosscheck, 'evaluate': cmd_evaluate,
     'figures': cmd_figures, 'engine-check': cmd_engine_check}[args.cmd](args)


if __name__ == '__main__':
    main()
