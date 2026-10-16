#!/usr/bin/env python3
"""B2(選び直したデータで学習するモデル B の作り方)のフレームの表を作る(CPU だけ。生ログ・データセットは変えない)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \
        /home/jetson/miniconda/envs/jetracer/bin/python -u experiments/20261016_b2_curation/make_b2_tables.py

出力(data/derived/b2_curation_20261016/。書き終えてから置き換える = 電源断で半端なファイルを残さない):
  passes_b2.json      R1 の通過の表。tools/offline_rl.py の course_passes を、人の 9 ログと AI の 11 ログに掛けたもの
                      (起点 ta・緑マット出口 tg・進入 eRPM = 起点の手前 0.4 秒の最大の前進 eRPM・R1/DE の衝突)
  drop_b2.npz         --drop-mask の表(scene, frame, drop, reason)。clean_driving_20261011 の 52 シーンの全フレーム
                      reason: 0 = 残す、1 = 速い進入(≥ 6200 eRPM)の手前の窓、2 = R1 で衝突した通過
  remap_b2.npz        --mode-remap-npz の表(scene, frame, mode)。152405・152707 の全フレーム = 4(高速走行)、ほか = -1(そのまま)
  curation_card.json  規則・数(B と同じ数え方の学習サンプル数)・sha256

規則(protocol.json の curation と同じ。変えるときは deviations.md):
  1. R1 の通過は course_passes の起点 ta(1 周に 1 回の右ヘアピンの入口)。人の 9 ログの通過だけを使う
  2. 速い進入: entry_erpm ≥ 6200 の通過は、時刻 [ta − 2.0 秒, ta + 1.0 秒] のフレームを外す
  3. 衝突した通過: r1_collision(R1 の 2.5 G の衝突が [ta − 0.5, tg − 0.3](tg が無ければ ta + 3.5)にある)の通過は、
     [ta − 2.0 秒, max(ta + 1.0, 最後の R1 の衝突 + 2.0, tg − 0.3(無ければ ta + 3.5))] のフレームを外す
  4. 152405・152707 の 12 シーンのジャンルを 5 → 4 にする(学習・検証とも。学習の表の付け替えだけで、aligned は変えない)
  5. 外す表と付け替えの表は、学習と検証の両方に同じく効く(tools/train.py が両方に渡す)
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))

DATASET = 'clean_driving_20261011'
OUT_DIR = REPO / 'data' / 'derived' / 'b2_curation_20261016'
HUMAN_LOGS = ('20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707',
              '20261010-193638', '20261010-200916', '20261010-201626', '20261010-202633')
AI_LOGS = ('20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
           '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
           '20261010-204605')
RELABEL_LOGS = ('20261010-152405', '20261010-152707')
RELABEL_TO = 4
FAST_ERPM = 6200.0
BEFORE_S, AFTER_S = 2.0, 1.0
COLL_AFTER_S = 2.0
IMPACT_G = 2.5
STRIDE = 3
REASON_KEEP, REASON_FAST, REASON_COLL = 0, 1, 2


# ---------------------------------------------------------------- 規則(純粋な関数。テストする)

def collided_end(ta, tg, coll_t):
    """衝突した通過を外す窓の終わり [秒]: max(ta + 1.0, 最後の R1 の衝突 + 2.0, tg − 0.3(無ければ ta + 3.5))"""
    win_end = (tg - 0.3) if tg is not None else ta + 3.5
    last = (max(coll_t) + COLL_AFTER_S) if coll_t else -np.inf
    return float(max(ta + AFTER_S, last, win_end))


def drop_reason(t_s, passes):
    """フレームの時刻 t_s [秒、ログの t0 から] → 外す理由 [N] int8(0 残す・1 速い進入・2 衝突した通過。両方なら 2)。
    passes: そのログの通過 [{ta_s, tg_s, entry_erpm, r1_collision, r1_collision_t_s}]"""
    t = np.asarray(t_s, np.float64)
    reason = np.zeros(len(t), np.int8)
    for p in passes:
        ta = float(p['ta_s'])
        e = p.get('entry_erpm')
        if e is not None and float(e) >= FAST_ERPM:
            w = (t >= ta - BEFORE_S) & (t <= ta + AFTER_S)
            reason[w & (reason == REASON_KEEP)] = REASON_FAST
        if p.get('r1_collision'):
            end = collided_end(ta, p.get('tg_s'), p.get('r1_collision_t_s') or [])
            w = (t >= ta - BEFORE_S) & (t <= end)
            reason[w] = REASON_COLL
    return reason


def remap_value(scene):
    """シーン名 → 付け替え先のジャンル(-1 = そのまま)"""
    return RELABEL_TO if str(scene)[:15] in RELABEL_LOGS else -1


def used_samples(future_valid, vehicle_valid, mode, impact, drop=None, stride=STRIDE):
    """tools/train.py(jetformer/train_data.py の JetFormerDataset)と同じ数え方の、学習に使うフレーム [N] bool"""
    ok = np.asarray(future_valid, bool) & np.asarray(vehicle_valid, bool) & (np.asarray(mode) != 0)
    if drop is not None:
        ok &= ~np.asarray(drop, bool)
    ok &= ~np.asarray(impact, bool)
    ok &= (np.arange(len(ok)) % stride) == 0
    return ok


# ---------------------------------------------------------------- 読み書き

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def compute_passes():
    """course_passes(tools/offline_rl.py)を今のコード・ラベルで 20 ログに掛け、R1 の衝突の時刻を足す"""
    import offline_rl as orl
    from jetformer import config
    passes = orl.course_passes({'log': np.array(HUMAN_LOGS + AI_LOGS)})
    for p in passes:
        info = json.loads((config.DATA_DIR / 'labels' / (p['log'] + '.json')).read_text(encoding='utf-8'))
        end = (p['tg_s'] - 0.3) if p['tg_s'] is not None else p['ta_s'] + 3.5
        p['r1_collision_t_s'] = ([c['t_s'] for c in info['events']['collisions']
                                  if c['loc'] == 'R1' and p['ta_s'] - 0.5 <= c['t_s'] <= end]
                                 if p['r1_collision'] else [])
    return passes


def build(out_dir=OUT_DIR, passes=None):
    from jetformer import config
    from jetformer.train_data import impact_mask
    out_dir = Path(out_dir)
    passes = compute_passes() if passes is None else passes
    root = config.DATASET_DIR / DATASET
    splits = json.loads((root / 'splits.json').read_text())
    split_of = {s: k for k in ('train', 'val') for s in splits[k]}
    by_log = {}
    for p in passes:
        if p['log'] in HUMAN_LOGS:
            by_log.setdefault(p['log'], []).append(p)
    cols = {'scene': [], 'frame': [], 'drop': [], 'reason': [], 'mode': []}
    counts = {sp: {'used_B': 0, 'used_B2': 0, 'dropped_samples': 0, 'dropped_ok_frames_not_impact': 0,
                   'dropped_ok_frames_fast': 0, 'dropped_ok_frames_coll': 0,
                   'used_B_by_genre': {}, 'used_B2_by_genre_b2': {}} for sp in ('train', 'val')}
    per_scene = []
    for s in sorted(split_of):
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            a = {k: z[k] for k in ('timestamps_us', 'mode', 'future_valid', 'vehicle_valid', 'imu')}
        n = len(a['timestamps_us'])
        lg = s[:15]
        ps = by_log.get(lg, [])
        t0 = ps[0]['t0_us'] if ps else None
        reason = drop_reason((a['timestamps_us'] - t0) / 1e6, ps) if ps else np.zeros(n, np.int8)
        drop = reason > 0
        imp = impact_mask(a['timestamps_us'], a['imu'], IMPACT_G)
        uB = used_samples(a['future_valid'], a['vehicle_valid'], a['mode'], imp)
        uB2 = used_samples(a['future_valid'], a['vehicle_valid'], a['mode'], imp, drop)
        ok = a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0)
        g_old = int(np.bincount(a['mode'][ok]).argmax()) if ok.any() else 0
        g_new = remap_value(s) if remap_value(s) >= 0 else g_old
        c = counts[split_of[s]]
        c['used_B'] += int(uB.sum())
        c['used_B2'] += int(uB2.sum())
        c['dropped_samples'] += int((uB & drop).sum())
        c['dropped_ok_frames_not_impact'] += int((ok & drop & ~imp).sum())
        c['dropped_ok_frames_fast'] += int((ok & (reason == REASON_FAST)).sum())
        c['dropped_ok_frames_coll'] += int((ok & (reason == REASON_COLL)).sum())
        c['used_B_by_genre'][str(g_old)] = c['used_B_by_genre'].get(str(g_old), 0) + int(uB.sum())
        c['used_B2_by_genre_b2'][str(g_new)] = c['used_B2_by_genre_b2'].get(str(g_new), 0) + int(uB2.sum())
        per_scene.append({'scene': s, 'split': split_of[s], 'frames': n, 'genre': g_old, 'genre_b2': g_new,
                          'used_B': int(uB.sum()), 'used_B2': int(uB2.sum()),
                          'drop_frames': int(drop.sum()), 'drop_fast': int((reason == REASON_FAST).sum()),
                          'drop_coll': int((reason == REASON_COLL).sum())})
        cols['scene'] += [s] * n
        cols['frame'] += list(range(n))
        cols['drop'] += drop.tolist()
        cols['reason'] += reason.tolist()
        cols['mode'] += [remap_value(s)] * n
    scene = np.array(cols['scene'])
    frame = np.array(cols['frame'], np.int32)
    _save_json(out_dir / 'passes_b2.json', passes)
    _save_npz(out_dir / 'drop_b2.npz', scene=scene, frame=frame, drop=np.array(cols['drop'], bool),
              reason=np.array(cols['reason'], np.int8))
    _save_npz(out_dir / 'remap_b2.npz', scene=scene, frame=frame, mode=np.array(cols['mode'], np.int16))
    hp = [p for p in passes if p['log'] in HUMAN_LOGS]
    dropped_passes = [{'log': p['log'], 'ta_s': p['ta_s'], 'entry_erpm': p['entry_erpm'], 'r1_collision': p['r1_collision'],
                       'r1_strong': p['r1_strong'], 'reason': ('fast+collision' if p['r1_collision'] and p['entry_erpm'] >= FAST_ERPM
                                                               else 'collision' if p['r1_collision'] else 'fast')}
                      for p in hp if p['r1_collision'] or (p['entry_erpm'] is not None and p['entry_erpm'] >= FAST_ERPM)]
    card = {
        'name': 'b2_curation_20261016', 'dataset': DATASET,
        'rules': {'fast_erpm': FAST_ERPM, 'fast_window_s': [-BEFORE_S, AFTER_S],
                  'collided_window': '[ta − 2.0, max(ta + 1.0, 最後の R1 の衝突 + 2.0, tg − 0.3 または ta + 3.5)]',
                  'relabel_logs': list(RELABEL_LOGS), 'relabel_to': RELABEL_TO, 'impact_g': IMPACT_G, 'stride': STRIDE,
                  'applies_to': 'train と val の両方(tools/train.py が同じ表を両方に渡す)'},
        'passes': {'human': len(hp), 'ai_logs': len(passes) - len(hp), 'dropped_human_passes': len(dropped_passes),
                   'dropped_fast': sum(d['reason'] != 'collision' for d in dropped_passes),
                   'dropped_collision': sum(d['reason'] != 'fast' for d in dropped_passes)},
        'dropped_passes': dropped_passes,
        'counts': counts, 'per_scene': per_scene,
        'files': {n: {'path': str(out_dir / n), 'sha256': sha256(out_dir / n)}
                  for n in ('passes_b2.json', 'drop_b2.npz', 'remap_b2.npz')},
    }
    _save_json(out_dir / 'curation_card.json', card)
    return card


def main():
    card = build()
    print(json.dumps({k: card[k] for k in ('passes', 'counts', 'files')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
