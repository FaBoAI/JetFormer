#!/usr/bin/env python3
"""介入区間の検証セット(ロードマップ S9): AI が自分で走り込んだ状態から、人が交代して直した操作を正解にする

    python tools/make_intervention_set.py                                   # → data/datasets/intervention_val_20261011
    python tools/make_intervention_set.py --name x --classes all --kinds forward reverse idle
    python tools/compare_models.py --dataset intervention_val_20261011 --runs <モデルA> <モデルB>   # 順位を出す(GPU)

ai_run_003〜010・101(10/10 の AI 走行 8本)の交代(AI → 人)ごとに、1つの「窓」を検証用のシーンにする:
  ・前置き lead_s(既定 3秒): 交代の前の AI のフレーム。採点しない(future_valid=False)。compare_models の自己入力で、
    モデルが交代の前から自分の予測を入れて走っている状態を作るため(AI 走行と同じ)
  ・修正区間: 交代の時刻 + skip_s 〜 min(交代 + max_s(既定 5秒), 人 → AI に戻した時刻)の人のフレーム。
    正解は人のスティック(build_dataset と同じ: k+1 フレーム後のスティック)。採点するのは、
    スティックが受信中で新しく、現在〜10フレーム先の時間が人・基板の再起動の前後でなく(交代の前後 0.3秒のマスクは
    使わない)、修正の種類が kinds のもの
修正の種類(correction_kind): 0=前進(前へ走っている、または前進の指令)/ 1=後退(後ろへ動いている、または止まっていて
  ブレーキ・後退側の指令)/ 2=停止(どちらでもない)。既定は前進だけを採点する(ロードマップ 4.3: 人の修正区間のうち
  「交代後に人が前進して走っている部分だけ」が良い操作。後退・置き直しはどのモデルも予測しないので順位を乱す)
交代の分類(label_events): 既定は失敗による交代(after_collision・stuck・preemptive)だけ。運用上の交代(終了・置き直し・
  基板の再起動)は --classes all で入れる

形式は jetformer/dataset.py の aligned と同じ(dataset._aligned_arrays で作る)。splits.json は val だけ(train は空)。
窓ごとに aligned/<ログ>-tkNN.npz(NN はそのログの交代の番号)。画像は samples/CAM_FRONT/<元のシーン>/ へのハードリンク。
nuScenes の表・ToF の PNG・can_bus は作らない(hf_export には使えない)。JetFormerDataset と compare_models は読める。
AI のフレームの「今の操作値」は tools/build_ai_dataset.py と同じ(1フレーム前の out を評価時のゲイン D3 で割り戻した値)。
dataset_card.json に、窓の一覧と、モデルなしの基準の誤差(今の操作値をそのまま使う・中心/中立のまま)を書く。
GPU は使わない。
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
from jetformer import config, dataset, modes  # noqa: E402
import build_ai_dataset as bad  # noqa: E402
import label_events as le  # noqa: E402

TOOL = 'tools/make_intervention_set.py'
AI_RUN_LOGS = ('20261010-171639', '20261010-172505', '20261010-172910', '20261010-173445', '20261010-181504',
               '20261010-203924', '20261010-204208', '20261010-204605')          # ai_run_003〜010・101(005 は無い)
KINDS = ('forward', 'reverse', 'idle')
MOVING_ERPM = 300            # これより速ければ動いている(eRPM は負 = 前進)
CMD_US = 30                  # 中立からこれ以上離れていればその向きの指令
EXTRA_KEYS = dict(bad.EXTRA_KEYS, **{
    'in_correction': '[N] 修正区間の人のフレーム(採点するかは future_valid)',
    't_since_takeover_s': '[N] 交代からの秒(前置きは負)',
    'correction_kind': '[N] int8 0=前進 1=後退 2=停止(修正区間の人のフレームだけ。ほかは -1)',
    'takeover_class': '[N] int8 交代の分類(label_events.TAKEOVER_CLASSES の番号。窓で同じ値)',
})
BASE_KEYS = {
    'timestamps_us': '[N] カメラ時刻', 'cam_files': '[N] 画像の相対パス',
    'tof': '[N,42,54] uint16 直前のToF(生の値)', 'tof_age_ms': '[N]',
    'imu': '[N,%d,6] 直前のIMU accel(m/s^2) + gyro(rad/s)' % dataset.IMU_HISTORY,
    'vehicle': '[N,4] 今の steering_us, throttle_us, mode_us(CH3), erpm(AI のフレームは1フレーム前の out の割り戻し)',
    'vehicle_valid': '[N]', 'vehicle_age_ms': '[N]',
    'future_actions': '[N,%d,2] 1〜%dフレーム先の steering_us, throttle_us' % (dataset.FUTURE_STEPS, dataset.FUTURE_STEPS),
    'future_valid': '[N] 採点するフレーム(修正区間の人のフレームで、条件を満たすもの)',
    'mode': '[N] int16 走行ジャンルID', 'erpm_valid': '[N]',
}


def correction_kind(erpm, fut_throttle, neutral, moving=MOVING_ERPM, cmd=CMD_US):
    """修正の種類 0=前進 / 1=後退 / 2=停止。erpm [n](負 = 前進)、fut_throttle [n,k] µs(人のスティック)"""
    erpm = np.nan_to_num(np.asarray(erpm, np.float64))
    th = np.asarray(fut_throttle, np.float64)
    fwd_move, back_move = erpm < -moving, erpm > moving
    fwd_cmd = (th < neutral - cmd).any(axis=1)
    back_cmd = (th > neutral + cmd).any(axis=1)
    reverse = back_move | (back_cmd & ~fwd_move)
    forward = ~reverse & (fwd_move | fwd_cmd)
    return np.where(forward, 0, np.where(reverse, 1, 2)).astype(np.int8)


# ---------------------------------------------------------------- 生ログ
def load_raw(log, raw_dir=None):
    """ログの使えるシーンをつないだ配列: カメラ(時刻・画像・シーン・ジャンル)、ToF・IMU・車両(_aligned_arrays に渡せる形)、
    AI_CMD の mode_id。戻り値 (dict, 外したシーン [(名前, 理由)])"""
    root = Path(raw_dir or config.RAW_DIR) / log
    parts = {k: [] for k in ('cam_t', 'files', 'scene', 'mode', 'tof', 'imu', 'veh', 'ai')}
    skipped = []
    for s in bad.log_scenes(log, raw_dir):
        p = root / s
        why = bad.scene_problem(p)
        if why:
            skipped.append((s, why))
            continue
        meta = json.loads((p / 'scene.json').read_text(encoding='utf-8'))
        cam = dataset._load(p, 'CAM_FRONT')
        ct = cam['timestamps_us'].astype(np.int64)
        parts['cam_t'].append(ct)
        parts['files'].append(np.array([str(p / str(f)) for f in cam['files']]))
        parts['scene'].append(np.full(len(ct), s))
        parts['mode'].append(dataset._frame_modes(dataset._with_modes(meta), ct))
        tof, imu, veh = dataset._load(p, 'TOF_FRONT'), dataset._load(p, 'IMU'), dataset._load(p, 'VEHICLE')
        parts['tof'].append({'timestamps_us': tof['timestamps_us'].astype(np.int64), 'depth': tof['depth']})
        parts['imu'].append({'timestamps_us': imu['timestamps_us'].astype(np.int64), 'accel': imu['accel'],
                             'gyro': imu['gyro']})
        nv = len(veh['timestamps_us'])
        parts['veh'].append({'timestamps_us': veh['timestamps_us'].astype(np.int64), 'pwm': veh['pwm'],
                             'valid': veh['valid'], 'select': veh['select'] if 'select' in veh else np.zeros(nv, bool),
                             'erpm': veh['erpm'] if 'erpm' in veh else np.zeros(nv, np.int32)})
        ai = dataset._load(p, 'AI_CMD')
        if ai is not None and len(ai['timestamps_us']) and 'mode_id' in ai:
            parts['ai'].append({'timestamps_us': ai['timestamps_us'].astype(np.int64), 'mode_id': ai['mode_id']})
    if not parts['cam_t']:
        return None, skipped

    def cat(dicts):
        out = {k: np.concatenate([d[k] for d in dicts]) for k in dicts[0]}
        order = np.argsort(out['timestamps_us'], kind='stable')
        return {k: v[order] for k, v in out.items()}
    cam_t = np.concatenate(parts['cam_t'])
    order = np.argsort(cam_t, kind='stable')
    raw = {'cam_t': cam_t[order], 'files': np.concatenate(parts['files'])[order],
           'scene': np.concatenate(parts['scene'])[order], 'mode': np.concatenate(parts['mode'])[order],
           'tof': cat(parts['tof']), 'imu': cat(parts['imu']), 'veh': cat(parts['veh']),
           'ai_cmd': cat(parts['ai']) if parts['ai'] else None}
    return raw, skipped


# ---------------------------------------------------------------- 窓
def takeover_windows(info, lead_s, max_s, skip_s):
    """交代ごとの窓 [{'n', 't0_us', 'start_us', 'corr_us', 'end_us', 'handback_us', 'takeover'}]"""
    hb = sorted(h['t_us'] for h in info['events'].get('handbacks', []))
    out = []
    for n, tk in enumerate(info['events'].get('takeovers', [])):
        t0 = int(tk['t_us'])
        nxt = [h for h in hb if h > t0]
        end = t0 + int(max_s * 1e6)
        if nxt:
            end = min(end, nxt[0])
        out.append({'n': n + 1, 't0_us': t0, 'start_us': t0 - int(lead_s * 1e6), 'corr_us': t0 + int(skip_s * 1e6),
                    'end_us': end, 'handback_us': nxt[0] if nxt else None, 'takeover': tk})
    return out


def window_arrays(raw, L, w, kinds, neutral):
    """1つの窓の aligned の dict・元の画像のパス・元のシーン名。採点するフレームだけ future_valid=True"""
    sel = (raw['cam_t'] >= w['start_us']) & (raw['cam_t'] <= w['end_us'])
    if w['handback_us'] is not None:
        sel &= raw['cam_t'] < w['handback_us']
    ct = raw['cam_t'][sel]
    a = dataset._aligned_arrays(ct, raw['tof'], raw['imu'], raw['veh'])
    a['cam_files'] = np.array(['samples/CAM_FRONT/%s/%d.jpg' % (s, int(t)) for s, t in zip(raw['scene'][sel], ct)])
    a['mode'] = raw['mode'][sel].astype(np.int16)
    li = L.index(ct)
    has = li >= 0
    k = np.clip(li, 0, None)
    ai_f = has & L.ai[k]
    hu_f = has & ~L.ai[k]
    bad.apply_ai_frames(a, L, li, ai_f, bad.latest_ch3(raw['veh'], ct), bad.ai_cmd_modes(raw['ai_cmd'], ct))
    in_corr = hu_f & (ct >= w['corr_us'])
    kind = np.full(len(ct), -1, np.int8)
    kind[in_corr] = correction_kind(L.erpm[k][in_corr], a['future_actions'][in_corr, :, 1], neutral)
    ok = in_corr & a['future_valid'] & a['vehicle_valid'] & L.human_ok[k] \
        & np.isin(kind, [KINDS.index(x) for x in kinds])
    a['future_valid'] = ok
    a.update(bad.extras(L, li))
    a['label_valid'] = has & L.valid_h[k]
    a['in_correction'] = in_corr
    a['t_since_takeover_s'] = ((ct - w['t0_us']) / 1e6).astype(np.float32)
    a['correction_kind'] = kind
    a['takeover_class'] = np.full(len(ct), le.TAKEOVER_CLASSES.index(w['takeover']['class']), np.int8)
    return a, raw['files'][sel], sorted(set(raw['scene'][sel].tolist()))


def link_images(root, rel_files, src_files):
    for rel, src in zip(rel_files, src_files):
        dst = root / str(rel)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


# ---------------------------------------------------------------- 基準の誤差(モデルなし)
def baseline_errors(arrays, ref):
    """採点するフレームで、(1) 今の操作値をそのまま (2) 中心/中立のまま、を予測としたときの平均絶対誤差 [µs]"""
    ok = [a['future_valid'] & a['vehicle_valid'] for a in arrays]
    if not sum(int(m.sum()) for m in ok):
        return {'n': 0}
    y = np.concatenate([a['future_actions'][m] for a, m in zip(arrays, ok)]).astype(np.float64)
    cur = np.concatenate([a['vehicle'][m, :2] for a, m in zip(arrays, ok)]).astype(np.float64)
    hold = np.abs(cur[:, None, :] - y).mean(axis=0)
    const = np.abs(np.asarray(ref)[None, None, :] - y).mean(axis=0)
    return {'n': int(len(y)),
            'hold_current_steer_mae': hold[:, 0].round(1).tolist(), 'hold_current_throttle_mae': hold[:, 1].round(1).tolist(),
            'hold_current_k3': [round(float(hold[2, 0]), 1), round(float(hold[2, 1]), 1)],
            'constant_ref_k3': [round(float(const[2, 0]), 1), round(float(const[2, 1]), 1)],
            'human_steer_abs_dev_k3': round(float(np.abs(y[:, 2, 0] - ref[0]).mean()), 1),
            'human_throttle_k3_median': round(float(np.median(y[:, 2, 1])), 1)}


def dataset_baseline(name, ref):
    """比べる用: ほかのデータセットの検証シーンで同じ基準の誤差"""
    root = config.DATASET_DIR / name
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    arrays = []
    for s in splits.get('val', []):
        with np.load(root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            arrays.append({k: z[k] for k in ('future_valid', 'vehicle_valid', 'future_actions', 'vehicle')})
    return baseline_errors(arrays, ref)


# ---------------------------------------------------------------- 作る
def build(name, logs=AI_RUN_LOGS, classes=le.FAILURE_CLASSES, kinds=('forward',), lead_s=3.0, max_s=5.0, skip_s=0.0,
          gains=bad.D3, min_scored=1, overwrite=False, labels_root=None, compare_with=None, log=print):
    """検証セットを作る。dataset_card(dict)を返す"""
    if not name or any(c in name for c in '/\\.') or ' ' in name:
        raise ValueError('データセット名に使えない文字が含まれています')
    root = config.DATASET_DIR / name
    if root.exists():
        card_p = root / 'dataset_card.json'
        built_by = json.loads(card_p.read_text(encoding='utf-8')).get('built_by') if card_p.exists() else None
        if not overwrite:
            raise ValueError('同名のデータセットがあります(--overwrite で作り直す): %s' % name)
        if built_by != TOOL:
            raise ValueError('このツールで作ったデータセットではないので消しません: %s' % name)
        shutil.rmtree(root)
    (root / 'aligned').mkdir(parents=True)
    rows, names, arrays, skipped_scenes = [], [], [], []
    mode_frames = {}
    total_frames = 0
    ref = None
    for lg in logs:
        info, arr = bad.load_labels(lg, labels_root)
        L = bad.LogLabels(lg, info, arr, 'eval', gains, human_ignore_bits=le.MASK_BITS['handover'])
        ref = ref or L.ref
        raw, sk = load_raw(lg)
        skipped_scenes += [{'scene': s, 'reason': r} for s, r in sk]
        if raw is None:
            continue
        for w in takeover_windows(info, lead_s, max_s, skip_s):
            tk = w['takeover']
            a, src, scenes = window_arrays(raw, L, w, kinds, L.ref[1])
            wname = '%s-tk%02d' % (lg, w['n'])
            corr = a['in_correction']
            kc = {x: int((corr & (a['correction_kind'] == i)).sum()) for i, x in enumerate(KINDS)}
            scored = int(a['future_valid'].sum())
            fr = tk.get('frame')
            row = {'name': wname, 'log': lg, 'run': info.get('description'), 'class': tk['class'],
                   'failure': tk['failure'], 'loc': tk.get('loc'), 't_s': tk['t_s'], 'human_s': tk.get('human_s'),
                   'dt_since_collision_s': tk.get('dt_since_collision_s'), 'cause': tk.get('cause'),
                   'gains': [round(float(x), 3) for x in arr['gains'][fr]] if fr is not None else None,
                   'lookahead': int(arr['lookahead'][fr]) if fr is not None else None,
                   'frames': int(len(a['timestamps_us'])), 'lead_frames': int((a['t_since_takeover_s'] < 0).sum()),
                   'correction_frames': int(corr.sum()), 'correction_kinds': kc, 'scored': scored,
                   'scenes': scenes,
                   'included': tk['class'] in classes and scored >= min_scored}
            rows.append(row)
            if not row['included']:
                continue
            link_images(root, a['cam_files'], src)
            np.savez_compressed(root / 'aligned' / (wname + '.npz'), **a)
            names.append(wname)
            arrays.append(a)
            total_frames += len(a['timestamps_us'])
            for m, c in zip(*np.unique(a['mode'], return_counts=True)):
                mode_frames[int(m)] = mode_frames.get(int(m), 0) + int(c)
        log('%s %s: 交代 %d、窓 %d' % (lg, info.get('description'), len(info['events'].get('takeovers', [])),
                                     sum(1 for r in rows if r['log'] == lg and r['included'])))
    ref = ref or (1443.0, 1498.0)
    dataset._write_json(root / 'splits.json', {'train': [], 'val': sorted(names)}, indent=1)
    by_class = {}
    for c in le.TAKEOVER_CLASSES:
        sel = [a for a, r in zip(arrays, [r for r in rows if r['included']]) if r['class'] == c]
        if sel:
            by_class[c] = baseline_errors(sel, ref)
    inc = [r for r in rows if r['included']]
    card = {
        'name': name, 'version': 'v1.0-%s' % name, 'built_by': TOOL,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'scenes': len(names), 'train_scenes': 0, 'val_scenes': len(names), 'frames': total_frames,
        'scored_frames': sum(r['scored'] for r in inc), 'tof_frames': 0,
        'hours': round(total_frames / config.CAMERA['fps'] / 3600, 4),
        'format': 'aligned だけ(jetformer/dataset.py の aligned/*.npz と同じ配列)。nuScenes の表・ToF の PNG・can_bus は無い',
        'aligned_npz': dict(BASE_KEYS, **EXTRA_KEYS),
        'modes': modes.load(), 'mode_frames': {str(k): v for k, v in sorted(mode_frames.items())},
        'source_scenes': sorted(names), 'skipped_legacy_scenes': [],
        'intervention': {
            'logs': list(logs), 'classes': list(classes), 'kinds': list(kinds), 'lead_s': lead_s, 'max_s': max_s,
            'skip_s': skip_s, 'eval_gains': dict(gains), 'ref_us': list(ref),
            'kind_rule': '前進: eRPM < −%d か 10フレーム先までに中立−%dµs より前進側の指令(後退でなければ)。'
                         '後退: eRPM > %d か、前へ走っていないときに中立+%dµs より後退側の指令。停止: それ以外'
                         % (MOVING_ERPM, CMD_US, MOVING_ERPM, CMD_US),
            'windows': rows,
            'counts': {
                'takeovers': len(rows), 'windows': len(inc),
                'takeovers_by_class': {c: sum(1 for r in rows if r['class'] == c) for c in le.TAKEOVER_CLASSES},
                'windows_by_class': {c: sum(1 for r in inc if r['class'] == c) for c in le.TAKEOVER_CLASSES},
                'correction_frames': sum(r['correction_frames'] for r in inc),
                'correction_kinds': {x: sum(r['correction_kinds'][x] for r in inc) for x in KINDS},
                'scored_frames': sum(r['scored'] for r in inc),
            },
            'baselines': {'all': baseline_errors(arrays, ref), 'by_class': by_class},
            'skipped_scenes': skipped_scenes,
            'usage': 'python tools/compare_models.py --dataset %s --runs <モデル...>。採点は修正区間だけなので、'
                     'compare.json の "all" を見る("clean" は衝突の 1秒前〜2秒後を除くので、衝突後の交代はほとんど残らない)' % name,
        },
    }
    if compare_with:
        try:
            card['intervention']['baselines']['compare_' + compare_with] = dataset_baseline(compare_with, ref)
        except (OSError, KeyError, ValueError) as e:
            log('比べるデータセット %s を読めません: %s' % (compare_with, e))
    dataset._write_json(root / 'dataset_card.json', card, indent=1)
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', default='intervention_val_20261011')
    ap.add_argument('--logs', nargs='*', default=list(AI_RUN_LOGS))
    ap.add_argument('--classes', nargs='*', default=list(le.FAILURE_CLASSES),
                    help='交代の分類(%s)。all で全部' % ' / '.join(le.TAKEOVER_CLASSES))
    ap.add_argument('--kinds', nargs='*', default=['forward'], help='採点する修正の種類(%s)' % ' / '.join(KINDS))
    ap.add_argument('--lead-s', type=float, default=3.0, help='交代の前の AI のフレーム(採点しない)')
    ap.add_argument('--max-s', type=float, default=5.0, help='交代からこの秒まで(戻しが先ならそこまで)')
    ap.add_argument('--skip-s', type=float, default=0.0, help='交代の直後のこの秒は採点しない')
    ap.add_argument('--compare-with', default='clean_driving_20261011', help='基準の誤差を比べるデータセット')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    classes = list(le.TAKEOVER_CLASSES) if args.classes == ['all'] else args.classes
    bad_k = [k for k in args.kinds if k not in KINDS] + [c for c in classes if c not in le.TAKEOVER_CLASSES]
    if bad_k:
        ap.error('知らない値: %s' % bad_k)
    card = build(args.name, args.logs, classes, args.kinds, args.lead_s, args.max_s, args.skip_s,
                 overwrite=args.overwrite, compare_with=args.compare_with or None)
    iv = card['intervention']
    print(json.dumps({'name': card['name'], 'windows': card['val_scenes'], 'frames': card['frames'],
                      'scored_frames': card['scored_frames'], 'counts': iv['counts'], 'baselines': iv['baselines']},
                     ensure_ascii=False, indent=1))
    print('データセット: %s' % (config.DATASET_DIR / args.name))


if __name__ == '__main__':
    main()
