#!/usr/bin/env python3
"""シミュレーターと実機を混ぜた学習(2026-10-13、案 C)の準備: 混ぜたデータセット・ドロップの表・確認・手順の固定

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= nice -n 10 python tools/sim_cotrain.py prepare   # CPU: 下の 1〜4
    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python tools/sim_cotrain.py check                # CPU: 4 だけやり直す
    cd ~/jetformer && python tools/sim_cotrain.py protocol                                   # protocol.json を 1 回だけ書く

  1. build      jetformer.dataset.build_dataset で data/datasets/simreal_mix_20261013 を作る。
                実機 = clean_driving_20261011 の 52 シーン(同じシーン)、シミュレーター = 展開した 60 シーン
                (data/raw/ds-<シミュレーターのデータセット>/)。検証 = clean_driving_20261011 の val の 11 シーンだけ
                (シミュレーターの 60 シーンはすべて学習。最良のエポックは実機だけで選ぶ = 基準 A と同じ)
  2. restore    シミュレーターの 60 シーンの aligned を、元のシミュレーターのデータセットの aligned に戻す(下の注)
  3. drop-mask  2段目(実機だけの微調整)のフレームの表 = シミュレーターのフレームを全部 drop
                (tools/train.py --drop-mask の形: scene・frame・drop)→ data/derived/simreal_mix_20261013/drop_sim_frames.npz
  4. check      データセットの表(ジャンル 4・5・8、source = mixed、ジャンル別のフレーム数)、実機のシーンの aligned が
                clean_driving_20261011 と同じか(全 52 シーン)、シミュレーターのシーンが元と同じか(全 60 シーン)、
                学習に使うフレーム(JetFormerDataset。間引き 1/3・衝突 2.5G・今の操作値なし)の数と、ドロップの表を
                付けたときに基準 A とまったく同じフレームになるか → /home/jetson/jetformer_reports/sim_cotrain_20261013/prepare_check.json

注(2026-10-13 に見つけたこと): 展開したシミュレーターのシーン(scene_import)から build_dataset で作り直すと、
  - future_actions が 1 フレームずれる: シミュレーターのカメラの間隔は 33333/33334 µs で、操作値の時刻はカメラと同じ。
    build_dataset は k フレーム先を「カメラ時刻 + int(k × 33333.3) µs 以前の最新の操作値」で引くので、間隔が 33334 µs の
    ところでは 1 フレーム前の操作値になる(1 シーン 600 フレームのうち約 400 フレームで、どこかのステップが最大 約 195 µs 違う)
  - imu が数 ms ずれる: 元の IMU の時刻はカメラ時刻 − 3,333・6,667 µs などだが、展開ではカメラ時刻 − k × 10 ms とみなす
  学習の正解が変わってしまうので、このデータセットではシミュレーターのシーンの aligned の中身(RESTORE_KEYS)を元の
  シミュレーターのデータセットの aligned で置き換える(時刻・画像のパスは作り直したもの。時刻が同じことを確かめる)。
  実機のシーンはそのまま(作り直しても clean_driving_20261011 と同じ)。置き換えの前の違いは restore_log.json に残す。

GPU は使わない(学習・評価は plan.md の手順で別に行う)。電源が落ちても、同じコマンドをもう一度実行すれば続きから
(途中で止まったデータセットの作成は消して作り直す。置き換えはシーンごとに記録し、済んだものは飛ばす)。
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from jetformer import config  # noqa: E402

REAL_DATASET = 'clean_driving_20261011'
SIM_DATASET = 'jetformer-simulator-id8-20min_20261012-150956_b55616cf'
SIM_LOG = 'ds-' + SIM_DATASET
MIX = 'simreal_mix_20261013'
SIM_MODE = 8
REAL_MODES = (4, 5)
REPORT_DIR = Path('/home/jetson/jetformer_reports/sim_cotrain_20261013')
# 置き換える aligned の中身(時刻 timestamps_us と画像のパス cam_files は作り直したものを使う)
RESTORE_KEYS = ('tof', 'tof_age_ms', 'imu', 'vehicle', 'vehicle_valid', 'vehicle_age_ms', 'future_actions', 'future_valid',
                'erpm_valid', 'mode')
COMPARE_KEYS = ('timestamps_us', 'cam_files') + RESTORE_KEYS
# 学習の条件(基準 A = 2026-10-12 の policy zoo の jetformer_v0 と同じ)
TRAIN_KW = {'stride': 3, 'use_command': False, 'mask_impacts': True, 'impact_g': 2.5}
# 実験の腕(protocol.json で固定)。A は 10/12 の学習結果をそのまま使う(新しく学習しない)
SEEDS = (0, 1)
ZOO_NAME = 'policy_zoo_20261012'
RUN_A = REAL_DATASET + '-jetformer_v0-s%d-nocmd-stride3-e30'          # 基準 A(10/12 の policy zoo)
RUN_B = 'clean_driving_20261011-nocmd-s3-20261011-024626'             # モデル B(今の走行。A の 3 本目として範囲に入れる)
# C・C+FT の名前は 2026-10-13 の指示の形(simreal_mix_20261013-cotrain-s<種>・-cotrain-ft-s<種>)。学習の前に変えた
RUN_C = MIX + '-cotrain-s%d'                                          # C: 混ぜて学習
RUN_CFT = MIX + '-cotrain-ft-s%d'                                     # C+FT: C の best.pth から実機だけで微調整
RUN_AFT = REAL_DATASET + '-jetformer_v0-s%d-ft-real-lr1e-4-e12'       # A+FT(対照): A の best.pth から同じ微調整
ARMS = ('C', 'C+FT', 'A+FT')
PRACTICAL_US = 10.0                 # 10/12 の実用上の閾値(δ)
SPREAD_MULT = 2.0                   # 「良い」に要る平均の改善 = 2 × A のばらつき


def mix_root():
    return config.DATASET_DIR / MIX


def derived_dir():
    return config.DATA_DIR / 'derived' / MIX


def drop_mask_path():
    return derived_dir() / 'drop_sim_frames.npz'


def restore_log_path():
    return derived_dir() / 'restore_log.json'


def _read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    """書き終えてから置き換える(電源断で半端なファイルを残さない)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _save_npz(path, compressed=True, **arrays):
    """npz を書き終えてから置き換える(np.savez は名前に .npz を足すので、tmp も .npz で終わる名前にする)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- numpy だけの部品(テストする)

def compare_arrays(a, b, keys):
    """2 つの aligned(dict)を keys で比べる → {キー: {'max_abs', 'mismatch', 'shape_equal', 'missing'}}。
    max_abs は数値の最大の差(bool・文字列は None)、mismatch は違う要素の数(形が違えば -1)"""
    out = {}
    for k in keys:
        if k not in a or k not in b:
            out[k] = {'max_abs': None, 'mismatch': -1, 'shape_equal': False,
                      'missing': [n for n, d in (('a', a), ('b', b)) if k not in d]}
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            out[k] = {'max_abs': None, 'mismatch': -1, 'shape_equal': False, 'missing': []}
            continue
        if x.dtype.kind in 'biuf' and y.dtype.kind in 'biuf' and x.dtype.kind != 'b':
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            out[k] = {'max_abs': float(d.max()) if d.size else 0.0, 'mismatch': int((d > 0).sum()), 'shape_equal': True,
                      'missing': []}
        else:
            out[k] = {'max_abs': None, 'mismatch': int((x.astype(str) != y.astype(str)).sum()) if x.dtype.kind in 'US'
                      else int((x != y).sum()), 'shape_equal': True, 'missing': []}
    return out


def all_equal(cmp):
    """compare_arrays の結果が全部同じか"""
    return all(v['shape_equal'] and v['mismatch'] == 0 for v in cmp.values())


def restore_from_original(rebuilt, original, keys=RESTORE_KEYS):
    """作り直した aligned(rebuilt)の keys を元の aligned(original)の値で置き換えた dict を返す。
    時刻(timestamps_us)が 1 µs でも違えば ValueError(別のシーンを混ぜないように)。keys が元に無くても ValueError"""
    t0, t1 = np.asarray(rebuilt['timestamps_us']), np.asarray(original['timestamps_us'])
    if t0.shape != t1.shape or not np.array_equal(t0.astype(np.int64), t1.astype(np.int64)):
        raise ValueError('時刻が元のシーンと違います(%d / %d フレーム)' % (len(t0), len(t1)))
    missing = [k for k in keys if k not in original]
    if missing:
        raise ValueError('元の aligned に %s がありません' % missing)
    out = dict(rebuilt)
    for k in keys:
        v = np.asarray(original[k])
        if len(v) != len(t0):
            raise ValueError('%s の長さ %d がフレーム数 %d と違います' % (k, len(v), len(t0)))
        out[k] = v.copy()
    return out


def future_shift_stats(rebuilt, original, tol_us=1.0):
    """作り直しでの future_actions の違い: {'max_abs_us', 'frames_gt_tol', 'per_step_frames_gt_tol', 'frames'}"""
    d = np.abs(np.asarray(rebuilt['future_actions'], np.float64) - np.asarray(original['future_actions'], np.float64))
    per_step = d.max(axis=2) > tol_us                     # [N, 10]
    return {'max_abs_us': float(d.max()) if d.size else 0.0, 'frames_gt_tol': int(per_step.any(axis=1).sum()),
            'per_step_frames_gt_tol': [int(x) for x in per_step.sum(axis=0)], 'frames': int(len(d))}


def drop_frame_table(scene_frames, drop_scenes):
    """フレームの表(tools/train.py --drop-mask の形)。scene_frames = {シーン名: フレーム数}(学習・検証の全シーン)、
    drop_scenes = 全部のフレームを drop にするシーン。→ {'scene' [M] str, 'frame' [M] int64, 'drop' [M] bool}。
    表に無いシーンは学習でエラーになる(train_data._per_frame)ので、drop しないシーンも全フレームを載せる"""
    drop_scenes = set(drop_scenes)
    unknown = drop_scenes - set(scene_frames)
    if unknown:
        raise ValueError('scene_frames に無いシーンを drop にしようとしています: %s' % sorted(unknown)[:5])
    names = sorted(scene_frames)
    scene = np.concatenate([np.full(int(scene_frames[s]), s, dtype='U%d' % max(len(x) for x in names)) for s in names]
                           ) if names else np.zeros(0, 'U1')
    frame = np.concatenate([np.arange(int(scene_frames[s]), dtype=np.int64) for s in names]) if names else np.zeros(0, np.int64)
    drop = np.concatenate([np.full(int(scene_frames[s]), s in drop_scenes, dtype=bool) for s in names]) if names else np.zeros(0, bool)
    return {'scene': scene, 'frame': frame, 'drop': drop}


def judge_primary(arm_vals, ref_vals, spread_mult=2.0, spread_floor=0.0, practical_us=10.0):
    """主指標(保留の実機のきれいな走行の L1、小さいほど良い)の判定(protocol.json の decision.primary)。
    arm_vals・ref_vals = 種ごとの L [µs]。ばらつき s = max(基準の種の範囲, spread_floor)、閾値 = spread_mult × s。
    良い(better): 腕の全部の種 < 基準の全部の種(max_arm < min_ref)かつ 平均の改善(ref − arm)> 閾値
    悪い(worse) : 腕の全部の種 > 基準の全部の種(min_arm > max_ref)かつ 平均の悪化(arm − ref)> 閾値
    ほかは同等(equivalent)。practical = 平均の差の大きさが practical_us 以上か(10/12 の実用上の閾値)"""
    p, r = np.asarray(arm_vals, np.float64), np.asarray(ref_vals, np.float64)
    spread = max(float(r.max() - r.min()), float(spread_floor))
    thr = spread_mult * spread
    gain = float(r.mean() - p.mean())                       # + = 腕のほうが小さい(良い)
    if p.max() < r.min() and gain > thr:
        cls = 'better'
    elif p.min() > r.max() and -gain > thr:
        cls = 'worse'
    else:
        cls = 'equivalent'
    return {'class': cls, 'gain_us': gain, 'threshold_us': thr, 'ref_spread_us': spread, 'practical': abs(gain) >= practical_us,
            'arm_mean': float(p.mean()), 'arm_min': float(p.min()), 'arm_max': float(p.max()),
            'ref_mean': float(r.mean()), 'ref_min': float(r.min()), 'ref_max': float(r.max())}


def final_verdict(best, last):
    """best.pth と last.pth の judge_primary から最終の分類(10/12 と同じ考え方):
    「良い」は best・last の両方で better のときだけ(片方だけなら同等に下げる)。「悪い」は best.pth のまま"""
    if best['class'] == 'better':
        return 'better' if last['class'] == 'better' else 'equivalent'
    return best['class']


def run_name(arm, seed):
    """腕・種 → data/runs の学習結果の名前('A' は 10/12 の学習結果、'B' はモデル B)"""
    names = {'A': RUN_A, 'C': RUN_C, 'C+FT': RUN_CFT, 'A+FT': RUN_AFT}
    if arm == 'B':
        return RUN_B
    if arm not in names or seed not in SEEDS:
        raise ValueError('知らない腕・種: %s・%s' % (arm, seed))
    return names[arm] % seed


def train_args(arm, seed, resume=False):
    """腕・種 → tools/train.py の引数のリスト(cd ~/jetformer で `nice -n 10 python -u` に続けて実行する)。
    C = 基準 A と同じ作り方(30 エポック・lr 3e-4・cosine)で混ぜたデータセット。
    C+FT・A+FT = それぞれの best.pth(同じ種)から、実機のフレームだけで lr 1e-4・12 エポック(cosine)。
    resume=True は電源断のあと(同じ引数 + --resume。train.py が前回と引数が同じかを確かめる)"""
    if arm == 'C':
        a = ['--data', 'data/datasets/' + MIX, '--epochs', '30', '--lr', '3e-4']
    elif arm == 'C+FT':
        a = ['--data', 'data/datasets/' + MIX, '--epochs', '12', '--lr', '1e-4',
             '--init', 'data/runs/%s/best.pth' % run_name('C', seed),
             '--drop-mask', 'data/derived/%s/%s' % (MIX, drop_mask_path().name)]
    elif arm == 'A+FT':
        a = ['--data', 'data/datasets/' + REAL_DATASET, '--epochs', '12', '--lr', '1e-4',
             '--init', 'data/runs/%s/best.pth' % run_name('A', seed)]
    else:
        raise ValueError('学習する腕は %s のどれか: %s' % (ARMS, arm))
    out = ['tools/train.py', '--policy', 'jetformer_v0'] + a + [
        '--out', 'data/runs/' + run_name(arm, seed), '--seed', str(seed), '--stride', '3', '--batch', '64',
        '--workers', '4', '--no-command-input', '--impact-g', '2.5']
    return out + (['--resume'] if resume else [])


def spread_of(values):
    """値の範囲(最大 − 最小)"""
    v = [float(x) for x in values]
    return max(v) - min(v)


# ---------------------------------------------------------------- シーンの一覧

def scene_lists():
    """→ (実機の 52 シーン, 実機の検証 11 シーン, シミュレーターの 60 シーン, {シミュレーターのシーン: 元のシーン名})"""
    from jetformer import dataset
    real_card = _read_json(config.DATASET_DIR / REAL_DATASET / 'dataset_card.json')
    real_split = _read_json(config.DATASET_DIR / REAL_DATASET / 'splits.json')
    if not real_card or not real_split:
        raise SystemExit('%s がありません' % REAL_DATASET)
    real = list(real_card['source_scenes'])
    real_val = list(real_split['val'])
    if sorted(real) != sorted(real_split['train'] + real_split['val']):
        raise SystemExit('%s の source_scenes と splits が合いません' % REAL_DATASET)
    log_dir = config.RAW_DIR / SIM_LOG
    if not log_dir.is_dir():
        raise SystemExit('シミュレーターのシーンがありません: %s(scene_import で展開する)' % log_dir)
    sim, origin = [], {}
    for d in sorted(log_dir.iterdir()):
        if not (d / 'scene.json').exists():
            continue
        m = dataset.load_scene_meta(d.name)
        if not dataset.is_sim_scene(m):
            raise SystemExit('%s がシミュレーターのシーンになっていません' % d.name)
        src = m.get('imported_from') or {}
        if src.get('dataset') != SIM_DATASET or not src.get('scene'):
            raise SystemExit('%s の取り込み元が %s ではありません' % (d.name, SIM_DATASET))
        sim.append(d.name)
        origin[d.name] = src['scene']
    sim_card = _read_json(config.DATASET_DIR / SIM_DATASET / 'dataset_card.json') or {}
    if len(sim) != int(sim_card.get('scenes', -1)) or sorted(origin.values()) != sorted(sim_card.get('source_scenes', [])):
        raise SystemExit('展開したシミュレーターのシーン %d 個が、元のデータセットのシーンと合いません' % len(sim))
    return real, real_val, sim, origin


# ---------------------------------------------------------------- 1. build

def step_build(real, real_val, sim):
    from jetformer import dataset
    root = mix_root()
    if (root / 'dataset_card.json').exists():
        card = _read_json(root / 'dataset_card.json')
        if sorted(card.get('source_scenes', [])) != sorted(real + sim):
            raise SystemExit('%s は別のシーンで作られています(消してから作り直してください)' % root)
        print('1. build: 作成済み(%s)' % root, flush=True)
        return card
    if root.exists():                                # 電源断などで途中で止まった作成は消して作り直す
        print('1. build: 途中で止まったデータセットを消して作り直します: %s' % root, flush=True)
        shutil.rmtree(root)
    t0 = time.time()
    last = {'t': 0.0}

    def progress(frac, name):
        if time.time() - last['t'] > 30 or frac >= 1.0:
            last['t'] = time.time()
            print('  %5.1f%% %s(%.0f秒)' % (100 * frac, name, time.time() - t0), flush=True)

    card = dataset.build_dataset(MIX, real + sim, val_scenes=real_val, progress=progress)
    print('1. build: 作成 %d シーン・%d フレーム(%.0f秒)' % (card['scenes'], card['frames'], time.time() - t0), flush=True)
    return card


# ---------------------------------------------------------------- 2. restore

def restore_log_for(log, created_at):
    """置き換えの記録 log が、今のデータセット(dataset_card の created_at)のものなら そのまま、違えば新しい記録を返す。
    電源断のあとなどでデータセットを作り直したとき、前のデータセットの「置き換え済み」の記録を使わないように"""
    if log and created_at and log.get('card_created_at') == created_at:
        return log
    return {'card_created_at': created_at, 'scenes': {}}


def step_restore(sim, origin):
    root = mix_root()
    old = _read_json(restore_log_path(), {}) or {}
    created_at = (_read_json(root / 'dataset_card.json') or {}).get('created_at')
    log = restore_log_for(old, created_at)
    if old and log is not old:
        print('2. restore: 前の記録は別に作ったデータセットのもの(%s)なので、最初から置き換えます' % old.get('card_created_at'),
              flush=True)
    scenes = log.setdefault('scenes', {})
    sim_root = config.DATASET_DIR / SIM_DATASET
    done = 0
    for s in sim:
        p = root / 'aligned' / ('%s.npz' % s)
        orig = _load_npz(sim_root / 'aligned' / ('%s.npz' % origin[s]))
        cur = _load_npz(p)
        if s in scenes and scenes[s].get('restored'):
            if not all_equal(compare_arrays(cur, orig, RESTORE_KEYS)):
                raise SystemExit('%s は置き換え済みの記録があるのに、元と違います' % s)
            continue
        if s not in scenes:                          # 置き換えの前の違い(作り直しだけのときの違い)を残す
            cmp = compare_arrays(cur, orig, ('timestamps_us',) + RESTORE_KEYS)
            scenes[s] = {'original_scene': origin[s], 'before': {k: {'max_abs': v['max_abs'], 'mismatch': v['mismatch']}
                                                                 for k, v in cmp.items()},
                         'future_shift': future_shift_stats(cur, orig), 'restored': False}
            _write_json(restore_log_path(), log)
        new = restore_from_original(cur, orig)
        _save_npz(p, **new)
        scenes[s]['restored'] = True
        _write_json(restore_log_path(), log)
        done += 1
    log.update({'dataset': MIX, 'source_dataset': SIM_DATASET, 'keys': list(RESTORE_KEYS),
                'restored_scenes': sum(1 for v in scenes.values() if v.get('restored')),
                'updated_at': datetime.datetime.now().isoformat(timespec='seconds')})
    _write_json(restore_log_path(), log)
    card_path = root / 'dataset_card.json'
    card = _read_json(card_path)
    note = {'source_dataset': SIM_DATASET, 'scenes': log['restored_scenes'], 'keys': list(RESTORE_KEYS),
            'reason': 'build_dataset で展開したシーンから作り直すと future_actions が 1 フレームずれ、imu が数 ms ずれるため、'
                      'シミュレーターのシーンの aligned を元のデータセットの値に戻した(tools/sim_cotrain.py)',
            'log': str(restore_log_path())}
    if card.get('aligned_restored') != note:
        card['aligned_restored'] = note
        _write_json(card_path, card)
    print('2. restore: シミュレーターの %d シーンを元の aligned に戻しました(今回 %d)' % (log['restored_scenes'], done),
          flush=True)
    return log


# ---------------------------------------------------------------- 3. drop mask

def mix_scene_frames():
    root = mix_root()
    split = _read_json(root / 'splits.json')
    out = {}
    for s in split['train'] + split['val']:
        with np.load(root / 'aligned' / ('%s.npz' % s), allow_pickle=False) as z:
            out[s] = int(len(z['timestamps_us']))
    return out


def step_drop_mask(sim):
    table = drop_frame_table(mix_scene_frames(), sim)
    p = drop_mask_path()
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            same = all(np.array_equal(z[k], table[k]) for k in ('scene', 'frame', 'drop'))
        if not same:
            raise SystemExit('%s が今のデータセットと合いません(消してから作り直してください)' % p)
        print('3. drop-mask: 作成済み(%s)' % p, flush=True)
    else:
        _save_npz(p, **table)
        print('3. drop-mask: %s(%d フレーム中 drop %d)' % (p, len(table['drop']), int(table['drop'].sum())), flush=True)
    return {'path': str(p), 'sha256': file_sha256(p), 'rows': int(len(table['drop'])), 'drop': int(table['drop'].sum()),
            'scenes': int(len(np.unique(table['scene']))), 'drop_scenes': int(len(np.unique(table['scene'][table['drop']])))}


# ---------------------------------------------------------------- 4. check

def _frames_of(ds):
    """JetFormerDataset の使うフレーム → 画像のパスの一覧(シーンをまたいで比べられる名前)"""
    return [str(ds.scenes[k]['cam_files'][i]) for k, i in ds.index]


def _same_file(a, b):
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    if (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino):
        return True
    return sa.st_size == sb.st_size and Path(a).read_bytes() == Path(b).read_bytes()


def step_check(real, real_val, sim, origin, drop_info=None):
    from jetformer.train_data import JetFormerDataset, load_frame_table
    root = mix_root()
    card = _read_json(root / 'dataset_card.json')
    split = _read_json(root / 'splits.json')
    real_root = config.DATASET_DIR / REAL_DATASET
    sim_root = config.DATASET_DIR / SIM_DATASET
    real_card = _read_json(real_root / 'dataset_card.json')
    sim_card = _read_json(sim_root / 'dataset_card.json')
    sim_split = _read_json(sim_root / 'splits.json')
    res = {'dataset': MIX, 'checked_at': datetime.datetime.now().isoformat(timespec='seconds'), 'ok': {}}
    ok = res['ok']

    # ---- データセットの表
    mode_ids = sorted(m['id'] for m in card['modes'])
    expect_frames = {}
    for c in (real_card, sim_card):
        for k, v in (c.get('mode_frames') or {}).items():
            if v:
                expect_frames[k] = expect_frames.get(k, 0) + int(v)
    res['card'] = {'scenes': card['scenes'], 'train_scenes': card['train_scenes'], 'val_scenes': card['val_scenes'],
                   'frames': card['frames'], 'mode_ids': mode_ids, 'num_modes': max(mode_ids) + 1,
                   'mode_frames': card['mode_frames'], 'expected_mode_frames': expect_frames, 'source': card.get('source'),
                   'tags': card.get('tags'), 'sim_scenes': card.get('sim_scenes'), 'hours': card.get('hours'),
                   'aligned_restored': bool(card.get('aligned_restored'))}
    ok['card_modes_4_5_8'] = all(m in mode_ids for m in (4, 5, 8))
    ok['card_source_mixed'] = card.get('source') == 'mixed' and 'simulator' in (card.get('tags') or [])
    ok['card_counts'] = (card['scenes'] == len(real) + len(sim) and card.get('sim_scenes') == len(sim)
                         and card['frames'] == real_card['frames'] + sim_card['frames'])
    ok['card_mode_frames'] = {k: v for k, v in card['mode_frames'].items() if v} == expect_frames
    ok['val_is_real_val'] = sorted(split['val']) == sorted(real_val) and len(split['val']) == len(real_val)
    ok['train_has_all_sim'] = set(sim) <= set(split['train']) and not (set(sim) & set(split['val']))
    ok['train_real_same_as_A'] = sorted(set(split['train']) - set(sim)) == sorted(_read_json(real_root / 'splits.json')['train'])

    # ---- 実機のシーン: clean_driving_20261011 の aligned と同じか(全シーン)
    real_cmp, real_bad, img_checked, img_bad = {}, [], 0, []
    for s in real:
        a = _load_npz(root / 'aligned' / ('%s.npz' % s))
        b = _load_npz(real_root / 'aligned' / ('%s.npz' % s))
        cmp = compare_arrays(a, b, COMPARE_KEYS)
        if not all_equal(cmp):
            real_bad.append(s)
            real_cmp[s] = cmp
        for i in sorted({0, len(a['cam_files']) // 2, len(a['cam_files']) - 1}):
            img_checked += 1
            if not _same_file(root / str(a['cam_files'][i]), real_root / str(b['cam_files'][i])):
                img_bad.append('%s:%d' % (s, i))
    res['real_scenes'] = {'compared': len(real), 'keys': list(COMPARE_KEYS), 'different': real_bad,
                          'diffs': real_cmp, 'images_checked': img_checked, 'images_different': img_bad}
    ok['real_aligned_equal'] = not real_bad
    ok['real_images_equal'] = not img_bad

    # ---- シミュレーターのシーン: 元のデータセットの aligned と同じか(全シーン。画像はパスが違うので中身・時刻で)
    sim_bad, sim_cmp, img_checked, img_bad, name_bad = [], {}, 0, [], []
    for s in sim:
        a = _load_npz(root / 'aligned' / ('%s.npz' % s))
        b = _load_npz(sim_root / 'aligned' / ('%s.npz' % origin[s]))
        cmp = compare_arrays(a, b, ('timestamps_us',) + RESTORE_KEYS)
        if not all_equal(cmp):
            sim_bad.append(s)
            sim_cmp[s] = cmp
        stem_a = [Path(str(x)).stem for x in a['cam_files']]
        stem_b = [Path(str(x)).stem for x in b['cam_files']]
        if stem_a != stem_b:
            name_bad.append(s)
        for i in sorted({0, len(a['cam_files']) // 2, len(a['cam_files']) - 1}):
            img_checked += 1
            if not _same_file(root / str(a['cam_files'][i]), sim_root / str(b['cam_files'][i])):
                img_bad.append('%s:%d' % (s, i))
    res['sim_scenes'] = {'compared': len(sim), 'keys': ['timestamps_us'] + list(RESTORE_KEYS), 'different': sim_bad,
                         'diffs': sim_cmp, 'image_names_different': name_bad, 'images_checked': img_checked,
                         'images_different': img_bad}
    ok['sim_aligned_equal_original'] = not sim_bad
    ok['sim_images_equal'] = not img_bad and not name_bad
    rlog = _read_json(restore_log_path(), {}) or {}
    shifts = [v['future_shift'] for v in (rlog.get('scenes') or {}).values()]
    if shifts:
        before_imu = [v['before']['imu']['max_abs'] for v in rlog['scenes'].values()]
        res['rebuild_without_restore'] = {
            'note': '置き換えの前(build_dataset で作り直しただけ)の、元のシミュレーターのデータセットとの違い',
            'future_actions_max_abs_us': max(x['max_abs_us'] for x in shifts),
            'future_actions_frames_gt_1us': sum(x['frames_gt_tol'] for x in shifts),
            'frames': sum(x['frames'] for x in shifts),
            'imu_max_abs': max(x for x in before_imu if x is not None)}

    # ---- シミュレーターの検証用(元の val の 12 シーン)= シミュレーターの L1 の参考に使う
    sim_val_orig = set(sim_split['val'])
    res['sim_ref_scenes'] = sorted(s for s in sim if origin[s] in sim_val_orig)
    ok['sim_ref_scenes_12'] = len(res['sim_ref_scenes']) == len(sim_val_orig)

    # ---- 学習に使うフレーム(JetFormerDataset。画像は読まない)
    kw = dict(TRAIN_KW)
    out = {}
    A_tr = JetFormerDataset(str(real_root), 'train', **kw)
    A_va = JetFormerDataset(str(real_root), 'val', **kw)
    fa_tr, fa_va = _frames_of(A_tr), _frames_of(A_va)
    out['A'] = {'train': len(A_tr), 'val': len(A_va), 'num_modes': A_tr.num_modes, 'mode_counts': A_tr.mode_counts(),
                'impact_masked': [A_tr.impact_masked, A_va.impact_masked], 'action_reference': list(A_tr.action_reference())}
    del A_tr, A_va
    C_tr = JetFormerDataset(str(root), 'train', **kw)
    C_va = JetFormerDataset(str(root), 'val', **kw)
    out['C'] = {'train': len(C_tr), 'val': len(C_va), 'num_modes': C_tr.num_modes, 'mode_counts': C_tr.mode_counts(),
                'impact_masked': [C_tr.impact_masked, C_va.impact_masked], 'action_reference': list(C_tr.action_reference())}
    fc_va = _frames_of(C_va)
    real_ref = [a['vehicle'][a['vehicle_valid'] & a.get('erpm_valid', True) & (np.abs(a['vehicle'][:, 3]) < 1)]
                for a in C_tr.scenes]
    out['C']['stopped_frames_in_train'] = int(sum(len(x) for x in real_ref))
    del C_tr, C_va
    drop = load_frame_table(str(drop_mask_path()), 'drop')
    F_tr = JetFormerDataset(str(root), 'train', drop=drop, **kw)
    F_va = JetFormerDataset(str(root), 'val', drop=drop, **kw)
    ff_tr, ff_va = _frames_of(F_tr), _frames_of(F_va)
    out['C+FT'] = {'train': len(F_tr), 'val': len(F_va), 'num_modes': F_tr.num_modes, 'mode_counts': F_tr.mode_counts(),
                   'dropped': [F_tr.dropped, F_va.dropped], 'impact_masked': [F_tr.impact_masked, F_va.impact_masked],
                   'action_reference': list(F_tr.action_reference())}
    del F_tr, F_va
    for v in out.values():
        v['mode_counts'] = {str(k): c for k, c in sorted(v['mode_counts'].items())}
    res['train_frames'] = out
    ok['C_val_same_frames_as_A'] = fc_va == fa_va
    ok['FT_train_same_frames_as_A'] = ff_tr == fa_tr          # 順番も同じ(実機のシーンが先に並ぶ)
    ok['FT_val_same_frames_as_A'] = ff_va == fa_va
    ok['FT_no_sim_frames'] = str(SIM_MODE) not in out['C+FT']['mode_counts']
    ok['C_has_sim_frames'] = out['C']['mode_counts'].get(str(SIM_MODE), 0) > 0
    ok['C_real_counts_same_as_A'] = all(out['C']['mode_counts'].get(str(m)) == out['A']['mode_counts'].get(str(m))
                                        for m in REAL_MODES)
    ok['action_reference_same_as_A'] = out['C']['action_reference'] == out['A']['action_reference']
    if drop_info is None and drop_mask_path().exists():
        drop_info = {'path': str(drop_mask_path()), 'sha256': file_sha256(drop_mask_path())}
    res['drop_mask'] = drop_info
    res['all_ok'] = all(bool(v) for v in ok.values())
    _write_json(REPORT_DIR / 'prepare_check.json', res)
    print('4. check: %s → %s' % ('すべて OK' if res['all_ok'] else 'NG: %s' % [k for k, v in ok.items() if not v],
                                REPORT_DIR / 'prepare_check.json'), flush=True)
    print(json.dumps({'card': {k: res['card'][k] for k in ('scenes', 'train_scenes', 'val_scenes', 'frames', 'mode_frames',
                                                           'source')},
                      'train_frames': {k: {kk: v[kk] for kk in ('train', 'val', 'mode_counts')} for k, v in out.items()},
                      'rebuild_without_restore': res.get('rebuild_without_restore')}, ensure_ascii=False, indent=1))
    return res


# ---------------------------------------------------------------- コマンド

def cmd_prepare(args):
    real, real_val, sim, origin = scene_lists()
    print('実機 %d シーン(検証 %d)・シミュレーター %d シーン' % (len(real), len(real_val), len(sim)), flush=True)
    step_build(real, real_val, sim)
    step_restore(sim, origin)
    info = step_drop_mask(sim)
    res = step_check(real, real_val, sim, origin, info)
    if not res['all_ok']:
        raise SystemExit(1)


def cmd_check(args):
    real, real_val, sim, origin = scene_lists()
    res = step_check(real, real_val, sim, origin)
    if not res['all_ok']:
        raise SystemExit(1)


def baseline_from_zoo(zoo):
    """10/12 の評価(data/zoo/policy_zoo_20261012/zoo_eval.json)から、基準 A(v0 の種 0・1)とモデル B の数字を抜き出す"""
    names = ['jetformer_v0_s0', 'jetformer_v0_s1', 'B']
    pick = lambda d: {n: d[n] for n in names if n in d}                       # noqa: E731
    p = zoo['primary']['L_us']
    acc = zoo['accuracy']['metrics']
    out = {
        'L_us': {ck: pick(p[ck]) for ck in ('best', 'last')},
        'per_genre': {n: {s: {k: acc[n][s][k] for k in ('n', 'l1_us', 'fwd_bias_us', 'steer_sign_acc', 'steer_mae_k3',
                                                         'throttle_mae_k3')}
                          for s in ('human_clean', 'human_clean_genre4', 'human_clean_genre5')} for n in names},
        'r1_window_mean_us': {ck: pick(zoo['r1']['window_mean_us'][ck]) for ck in ('best', 'last')},
        'de_window_mean_us': {ck: pick(zoo['de']['window_mean_us'][ck]) for ck in ('best', 'last')},
        'nc_d_mean_us': {loc: {k: v['d_mean'] for k, v in zoo[loc]['nc'].items()} for loc in ('r1', 'de')},
        'nc_bad': {loc: {k: v['bad'] for k, v in zoo[loc]['nc'].items()} for loc in ('r1', 'de')},
        'robust_dL1_us': {n: zoo['robustness']['per_run'][n]['dL1_us'] for n in names},
        'robust_per_pert_dL1_us': {n: zoo['robustness']['per_run'][n]['per_pert_dL1_us'] for n in names},
        'robust_frames': zoo['robustness']['per_run']['B']['frames'],
        'collapse': pick(zoo['collapse']['per_run']),
        'latency': pick(zoo['latency']['per_run']),
        'frames': zoo['frames'],
    }
    bl = [p['best'][n] for n in names]
    ll = [p['last'][n] for n in names]
    out['spread_us'] = {'best': spread_of(bl), 'last': spread_of(ll),
                        'best_two_seeds': spread_of(bl[:2]), 'last_two_seeds': spread_of(ll[:2])}
    out['ref_mean_us'] = {'best': float(np.mean(bl[:2])), 'last': float(np.mean(ll[:2]))}
    out['robust_ref_dL1_us'] = float(np.mean([out['robust_dL1_us'][n] for n in names[:2]]))
    out['nc_abs_max_us'] = {loc: max(abs(v) for v in out['nc_d_mean_us'][loc].values()) for loc in ('r1', 'de')}
    return out


def build_protocol(check, base, files):
    """固定する実験の手順(protocol.json の中身)。check = prepare_check.json、base = baseline_from_zoo、
    files = {名前: {'path', 'sha256'}}(固定した入力のファイル)"""
    thr = {ck: SPREAD_MULT * base['spread_us'][ck] for ck in ('best', 'last')}
    tf = check['train_frames']
    cmds = {arm: {str(s): {'train': 'cd ~/jetformer && nice -n 10 python -u ' + ' '.join(train_args(arm, s)),
                           'resume': 'cd ~/jetformer && nice -n 10 python -u ' + ' '.join(train_args(arm, s, resume=True)),
                           'run': run_name(arm, s)} for s in SEEDS} for arm in ARMS}
    return {
        'title': 'シミュレーターと実機を混ぜた学習(案 C)の実験の手順。学習の前に固定(結果を見て変えない)',
        'version': 1,
        'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'question': 'シミュレーターのデータ(ジャンル 8)を実機のデータ(ジャンル 4・5)と一緒に学習し(C)、そのあと実機の'
                    'データだけで微調整すると(C+FT)、実機の保留データの精度は基準 A(実機だけ)より上がるか',
        'model': 'TransFuser軽量型(JetFormer v0)= jetformer_v0(既定の形 dim 256・layers 4・heads 8)。モデル B と同じ構造',
        'data': {
            'mix_dataset': 'data/datasets/' + MIX,
            'real': '%s の 52 シーン(同じシーン)= 実機' % REAL_DATASET,
            'sim': '%s の 60 シーン(scene_import で展開した data/raw/%s/)= シミュレーター' % (SIM_DATASET, SIM_LOG),
            'val': '%s の val の 11 シーンだけ(実機)。シミュレーターの 60 シーンはすべて学習 → 最良のエポックは実機だけで'
                   '選ぶ(基準 A と同じ)' % REAL_DATASET,
            'card': check['card'],
            'train_frames': {k: {kk: v[kk] for kk in ('train', 'val', 'num_modes', 'mode_counts', 'action_reference')}
                             for k, v in tf.items()},
            'restore': 'シミュレーターのシーンの aligned(future_actions・imu など)は元のシミュレーターのデータセットの値に'
                       '戻した(作り直しだと future_actions が 1 フレームずれる所がある。restore_log.json)',
            'checks_ok': check['ok'],
            'sim_ref_scenes': check['sim_ref_scenes'],
        },
        'files': files,
        'arms': {
            'A': {'role': '基準(新しく学習しない)', 'runs': {str(s): run_name('A', s) for s in SEEDS},
                  'recipe': '実機だけ(clean_driving_20261011)・30 エポック・間引き 1/3・バッチ 64・lr 3e-4 cosine・AdamW wd 1e-4・'
                            '今の操作値なし・衝突 2.5G の除外・ジャンル数 6',
                  'model_B': RUN_B,
                  'predictions': '10/12 の data/zoo/%s/pred/jetformer_v0_s{0,1}{,_last}.npz・B{,_last}.npz・pert/ をそのまま使う'
                                 % ZOO_NAME},
            'C': {'role': '主の腕: 混ぜて学習', 'recipe': 'A と同じ(データだけ %s。ジャンル数 9 = 0〜8)' % MIX,
                  'commands': cmds['C']},
            'C+FT': {'role': '主の腕: C のあと実機だけで微調整',
                     'recipe': '同じ種の C の best.pth から --init。%s + ドロップの表(シミュレーターのフレームを全部外す)= '
                               '学習のフレームは A と同じ %d(順番も同じ)・検証も同じ。lr 1e-4・12 エポック・cosine・同じ種。'
                               'ジャンル数 9 のまま(model_config の trained_modes は [4, 5]。頭 8 の重みは C のまま残る)' % (MIX, tf['C+FT']['train']),
                     'commands': cmds['C+FT']},
            'A+FT': {'role': '対照(副): 微調整を足しただけの効果を分ける',
                     'recipe': '同じ種の A の best.pth から、C+FT と同じ微調整(%s・lr 1e-4・12 エポック・cosine)。'
                               'ジャンル数 6 のまま' % REAL_DATASET,
                     'commands': cmds['A+FT'],
                     'why': 'C+FT は A より 12 エポック多く学習する。C+FT が A より良くても、それがシミュレーターのおかげか、'
                            '追加の微調整のおかげかを分けるため(約 15 分)'},
        },
        'gpu_order': ['C s0', 'C s1', 'C+FT s0', 'C+FT s1', 'A+FT s0', 'A+FT s1',
                      'エンジン(各 best.pth、tools/build_engine.py、1 本ずつ)', '予測(保留 10,985 フレーム × best・last、'
                      'ずれ、シミュレーターの 12 シーン)', 'エンジンの確認(engine_load_check.py と同じ計算)'],
        'gpu_rules': 'GPU は 1 プロセスずつ(nvidia-smi・ps で確かめる。web の走行中は始めない)・バッチ ≤ 64・workers ≤ 4・nice。'
                     '電源が落ちたら同じコマンドに --resume(last.pth から)。出来たもの(result.json の state=done、予測の npz)は飛ばす',
        'evaluation': {
            'frames': '10/12 と同じ: rl_evaluate.pred_table()(key %s)の 10,985 フレームを予測し、主指標は part 0 の人の 11 シーン & '
                      'future_valid & vehicle_valid & 2.5G の衝突の前後でない %d フレーム' % (
                          base['frames']['pred_table_key'], base['frames']['human_clean']),
            'prediction': 'runtime.load_run(学習結果, best|last) → runtime.forward(PyTorch FP32・eval)。ジャンル = aligned の mode '
                          '(4・5。rl_evaluate._genres と同じ)。tools/policy_zoo.py の _Frames・_predict_frames と同じ計算',
            'note_val_selection': '主指標の 11 シーンは、A でも C でも最良のエポックを選ぶ検証のシーン(10/12 と同じ条件。D9 の甘さは'
                                  'どの腕も同じ形で入る)。last.pth の確認で、選び方による差を見る',
        },
        'primary': {
            'L': '人のきれいな走行の L1 [µs] = |予測 − 人|(10 ステップ × [ステア, スロットル] の平均)のフレーム平均(10/12 と同じ)',
            'arm_value': '腕の値 = 2 つの種の平均 L̄、[min, max]',
            'reference': 'A = v0 の種 0・1(best.pth: %.3f・%.3f µs、last.pth: %.3f・%.3f µs)。モデル B は別の行(%.3f・%.3f µs)' % (
                base['L_us']['best']['jetformer_v0_s0'], base['L_us']['best']['jetformer_v0_s1'],
                base['L_us']['last']['jetformer_v0_s0'], base['L_us']['last']['jetformer_v0_s1'],
                base['L_us']['best']['B'], base['L_us']['last']['B']),
            'spread': 'A のばらつき s_A = {v0 種 0, v0 種 1, モデル B}(同じ作り方の 3 本)の L の範囲。best %.3f µs・last %.3f µs'
                      '(種 2 本だけなら %.3f・%.3f)' % (base['spread_us']['best'], base['spread_us']['last'],
                                                   base['spread_us']['best_two_seeds'], base['spread_us']['last_two_seeds']),
            'threshold_us': thr,
            'better': '「良い」= 腕の 2 つの種の両方が A の 2 つの種の両方より小さい(max_腕 < min_A)かつ 平均の改善 L̄_A − L̄_腕 > '
                      '2 × s_A(best %.3f µs)' % thr['best'],
            'worse': '「悪い」= min_腕 > max_A かつ L̄_腕 − L̄_A > 2 × s_A',
            'equivalent': 'ほかは「同等」',
            'ckpt_check': 'last.pth の予測で同じ計算(閾値は last の 2 × s_A = %.3f µs)。best で「良い」でも last で「良い」で'
                          'なければ「同等」に下げる(10/12 と同じ)。「悪い」は best のまま' % thr['last'],
            'practical': '|L̄_A − L̄_腕| ≥ %.0f µs なら「実用上の差」(10/12 の δ)。「良い」でも 10 µs 未満なら「小さいが一貫した差」'
                         'と書く' % PRACTICAL_US,
            'code': 'tools/sim_cotrain.py の judge_primary(arm, A, spread_mult=2, spread_floor=s_A)・final_verdict(best, last)',
        },
        'secondary_comparisons': {
            'C+FT_vs_C': '同じ規則で、基準を C の種 0・1 にする(s = max(C の種の範囲, s_A))。微調整が C の上に足したもの',
            'C+FT_vs_A+FT': '同じ規則で、基準を A+FT の種 0・1 にする(s = max(A+FT の種の範囲, s_A))。'
                            'C+FT が A より「良い」とき、A+FT にも「良い」なら → シミュレーターの効果',
            'paired': '種ごとの差(C+FT 種 i − C 種 i、C+FT 種 i − A+FT 種 i)も載せる(同じ種から始めた組)',
            'own_spread': '腕の種の範囲が 2 × s_A を超えたら「ばらつきが大きい」と書く',
        },
        'secondary_metrics': {
            'per_genre': 'ジャンル 4(高速)・5(低速)別の L1・3 フレーム先のステア・スロットルの MAE・向きの正解率・前進の偏り '
                         'fwd_bias(compare_models.metrics。主指標のフレーム)',
            'speed_habit': '速さの癖が移ったか: ジャンル別の fwd_bias(種の平均)が A(種の平均)から 10 µs 以上ずれたら「速さの癖の'
                           '移り」と書く(シミュレーターは速い: スロットルの中央値 1267 µs vs 実機 1315 µs)',
            'r1_de': '10/12 と同じ(drive_outputs・ゲイン 1.3/1.0・先読み 3・各学習結果の model_config の中心と中立、R1 17 通過・DE 14 '
                     '通過、通過の選び直し 10,000 回・種 0、R = A の種の平均、「良い」= (a) 99.375%% 区間の下限 > 0 (b) 両方の種で > 0 '
                     '(c) 4 組すべて ≥ 10 µs (d) 差の差の 99.375%% 区間の下限 > 0、「悪い」= 95%% 区間の上限 < 0)。'
                     '10/12 の陰性対照で種どうしでも |d̄| が R1 %.1f µs・DE %.1f µs あり、R1 の NC-B は「悪い」になった。'
                     'それ以下の差は「ばらつきの内」と書く。判定には使わない(参考)' % (
                         base['nc_abs_max_us']['r1'], base['nc_abs_max_us']['de']),
            'robustness': '10/12 と同じ 5 つのずれ(画像 3 行下・上、明るさ ×0.8・×1.2、ToF 1 行下)、同じ %d フレーム、best.pth。'
                          'ΔL1 = ずれの L1 の平均 − ずれなしの L1。「ずれに強い」= ΔL1_腕(種の平均)≤ 0.8 × ΔL1_A(%.3f µs)かつ'
                          '両方の種で ΔL1 ≤ ΔL1_A' % (base['robust_frames'], base['robust_ref_dL1_us']),
            'collapse': '10/12 と同じ「崩れ」: 直線 542 フレームの前進の強さの低下 > 5%、fwd_bias < −30 µs、向きの正解率 < A − 0.01 の'
                        'どれか。崩れた腕は「良い」でも推奨しない',
            'sim_l1': 'シミュレーターの L1(参考): 元のシミュレーターのデータセットの val の 12 シーン(混ぜたデータセットでは学習に'
                      '入っている = 保留ではない。C・C+FT は「覚えたか・微調整で忘れたか」、A・A+FT は「実機だけの学習で'
                      'シミュレーターをどれだけ当てられるか」の目安)。フレーム = future_valid & vehicle_valid & 2.5G の衝突の前後'
                      'でない全フレーム。ジャンルの頭 8(C・C+FT)・4・5(全部の腕)で予測。ジャンルは one-hot で直接与え、'
                      'rl_evaluate._genres の代用はしない(C+FT の model_config の trained_modes は [4, 5] になるが、頭 8 の重みは'
                      ' C から引き継いでいる。A・A+FT はジャンル数 6 で頭 8 が無いので 4・5 だけ)',
            'latency': '各 best.pth の FP16 エンジン(tools/build_engine.py)を 10/12 と同じ道具(engine_load_check.py: 100 回の空回しの後 '
                       '600 回の run_numpy、実フレーム 64 個で FP32 との差)で測る。ゲート: 精度 ≤ 2.5 µs・run_numpy 中央 ≤ 6.0 ms・'
                       'p99 ≤ 8.0 ms・peak_mem ≤ 4,096 MB・NaN/Inf なし。A は 10/12 の値(中央 %.2f・%.2f ms)' % (
                           base['latency']['jetformer_v0_s0']['run_numpy_median_ms'],
                           base['latency']['jetformer_v0_s1']['run_numpy_median_ms']),
            'training': '学習の曲線(実機の検証 L1・ジャンル別)、最良のエポック、1 エポックの秒数、best − last',
        },
        'conclusion_rules': [
            '1. C+FT・C のそれぞれを主指標で A と比べ、「良い / 同等 / 悪い」と実用上の差(10 µs)を書く',
            '2. C+FT が「良い」で、C+FT vs A+FT も「良い」なら「シミュレーターを混ぜた事前学習 + 微調整で実機の精度が上がった」。'
            'C+FT vs A+FT が「同等」なら「微調整を足した効果と区別できない(シミュレーターの効果とは言えない)」',
            '3. C だけ「良い」なら「混ぜるだけで上がった」。どちらも「同等」なら「シミュレーターを混ぜても実機の保留データの精度は'
            '上がらなかった(A と同じ)」。「悪い」はそのまま書く',
            '4. 走行の推奨を変える(モデル B から替える)のは、「良い」かつ実用上の差(≥ 10 µs)かつ崩れなし・ゲート合格の腕だけ。'
            '「良い」でも 10 µs 未満なら机上の小さな改善として報告し、推奨はモデル B のまま(実車の A/B/A の候補には書く)',
            '5. 机上の結論は実車の改善を示さない(10/12 の規則 7 のまま)。この結論は TransFuser軽量型(JetFormer v0)・この学習の'
            '条件(調整なし)・このシミュレーターのデータ(60 シーン・合成の先生)に限る',
        ],
        'deviations_from_task': [
            'A+FT(対照)を足した: 追加の微調整だけの効果を分けるため。GPU が足りなければ最後に回し、無いときは結論 2 を「分けられない」と書く',
            'ばらつき s_A は種 2 本ではなく {v0 種 0, 種 1, モデル B} の 3 本の範囲(best 0.313 µs。2 本なら 0.311 µs で、ほぼ同じ)',
            'last.pth でも同じ判定をして、best だけの「良い」は「同等」に下げる(10/12 と同じ)',
            'R1・DE の区間は 10/12 と同じ 99.375% のまま(比べる数は 4 つで 98.75% でよいが、10/12 と同じ数字にそろえた。厳しめ)',
            'シミュレーターの L1 は、指示のとおり 60 シーンを全部学習に入れたので保留ではない(参考の数字)',
        ],
        'baseline_A': base,
        'gpu_time_estimate': None,       # 学習の段で 1 エポックを測ってから plan.md に書く
    }


def protocol_inputs():
    """固定する入力のファイル {名前: {'path', 'sha256'}}"""
    root = mix_root()
    paths = {'mix_dataset_card': root / 'dataset_card.json', 'mix_splits': root / 'splits.json',
             'drop_mask': drop_mask_path(), 'restore_log': restore_log_path(),
             'prepare_check': REPORT_DIR / 'prepare_check.json',
             'zoo_eval': config.DATA_DIR / 'zoo' / ZOO_NAME / 'zoo_eval.json',
             'zoo_rules': config.DATA_DIR / 'zoo' / ZOO_NAME / 'zoo_rules.json',
             'zoo_eval_protocol': config.DATA_DIR / 'zoo' / ZOO_NAME / 'eval_protocol.json',
             'real_splits': config.DATASET_DIR / REAL_DATASET / 'splits.json',
             'sim_splits': config.DATASET_DIR / SIM_DATASET / 'splits.json'}
    return {k: {'path': str(p), 'sha256': file_sha256(p)} for k, p in paths.items()}


def cmd_protocol(args):
    """protocol.json を 1 回だけ書く(学習の前に固定。あれば書かない)。準備の確認(prepare_check.json)が全部 OK のときだけ"""
    p = REPORT_DIR / 'protocol.json'
    if p.exists():
        print('固定済み: %s(書き直さない)' % p)
        return
    check = _read_json(REPORT_DIR / 'prepare_check.json')
    if not check or not check.get('all_ok'):
        raise SystemExit('準備の確認が OK になっていません(先に prepare): %s' % (REPORT_DIR / 'prepare_check.json'))
    for arm in ARMS:
        for s in SEEDS:
            if (config.DATA_DIR / 'runs' / run_name(arm, s)).exists():
                raise SystemExit('学習が始まっています(%s)。手順は学習の前に固定する' % run_name(arm, s))
    zoo = _read_json(config.DATA_DIR / 'zoo' / ZOO_NAME / 'zoo_eval.json')
    proto = build_protocol(check, baseline_from_zoo(zoo), protocol_inputs())
    _write_json(p, proto)
    print('固定しました: %s(sha256 %s)' % (p, file_sha256(p)))
    print(json.dumps({'threshold_us': proto['primary']['threshold_us'], 'spread_us': proto['baseline_A']['spread_us'],
                      'commands': {a: {s: c['train'] for s, c in v['commands'].items()} for a, v in proto['arms'].items()
                                   if 'commands' in v}}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    sub.add_parser('check')
    sub.add_parser('protocol')
    args = ap.parse_args()
    {'prepare': cmd_prepare, 'check': cmd_check, 'protocol': cmd_protocol}[args.cmd](args)


if __name__ == '__main__':
    main()
