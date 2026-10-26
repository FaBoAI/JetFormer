#!/usr/bin/env python3
"""データセット v2 と表の確かめ(2026-10-25)。make_v2_tables.py と別のコードで数え直す。読むだけ。

numpy・OpenCV だけ(torch・GPU は使わない)。学習が使うサンプルは jetformer/train_data.JetFormerDataset の選び方を
ここで書き直して数える(future_valid & vehicle_valid & ジャンル ≠ 0 & ~drop & ~衝突の前後(水平 2.5G の 1 秒前〜2 秒後)
& フレーム番号 % 3 == 0)。書き直しが正しいかは、clean_driving_20261011 で B の train.log の数(学習 5,912・検証 1,559・
衝突の前後 3,522 / 1,083)が出るかで確かめる。

    cd ~/jetformer && /home/jetson/bin/jf-heavy --lock none --mem 3G --need-mb 1500 -- nice -n 10 \
      /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/check_v2_data.py

出力: /home/jetson/jetformer_reports/train_v2_20261025/data_checks.json(外れが 1 つでもあれば終わりの値 1)
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/jetson/jetformer')
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PROTOCOL = REP / 'protocol_v2.json'
NAME = 'dataset_v2_20261025'
DS = ROOT / 'data/datasets' / NAME
BASE = ROOT / 'data/datasets/clean_driving_20261011'
IVS = ROOT / 'data/datasets/intervention_val_20261011'
TBL = ROOT / 'data/derived' / NAME
OUT = REP / 'data_checks.json'
G = 9.80665
STRIDE = 3
IMPACT_G = 2.5
B_REF = (1441.0, 1496.0)          # protocol stage0_checks 7(B の steer_center_us・throttle_neutral_us)
EXPECT = {'v2h': 6562, 'v2m': 7757, 'bfast': 6748, 'val': 1559, 'teacher': 836, 'v2m_ai_w_share': 0.134}
TOL = 0.01


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def impact(ts_us, imu, g=IMPACT_G, before=1.0, after=2.0):
    """衝突の前後(train_data.impact_mask と同じ規則を、ここで書き直したもの)"""
    t = ts_us.astype(np.float64) / 1e6
    a = imu[:, :, :2].astype(np.float32)
    hg = np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2).max(axis=1) / G
    hit = np.sort(t[hg >= g])
    if not len(hit):
        return np.zeros(len(t), bool)
    # t の直前・直後の衝突との距離で判定(t ∈ [ti − before, ti + after])
    j = np.searchsorted(hit, t - after, side='left')        # ti >= t − after の最初
    jc = np.clip(j, 0, len(hit) - 1)
    return (j < len(hit)) & (hit[jc] <= t + before)


def table(path, key):
    with np.load(path, allow_pickle=False) as z:
        sc, fr, v = z['scene'].astype(str), z['frame'].astype(np.int64), z[key]
    out = {}
    order = np.argsort(sc, kind='stable')
    sc, fr, v = sc[order], fr[order], v[order]
    u, start = np.unique(sc, return_index=True)
    ends = list(start[1:]) + [len(sc)]
    for s, a, b in zip(u, start, ends):
        out[str(s)] = (fr[a:b], v[a:b])
    return out


def per_frame(tab, s, n, fill, dtype):
    if s not in tab:
        raise KeyError('表にシーン %s がありません' % s)
    f, v = tab[s]
    out = np.full(n, fill, dtype)
    out[f] = v
    return out


def load_scene(root, s):
    with np.load(root / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def usable(a, drop=None, impact_g=IMPACT_G):
    ok = a['future_valid'].astype(bool) & a['vehicle_valid'].astype(bool)
    ok &= a['mode'] != 0
    if drop is not None:
        ok &= ~drop
    imp = impact(a['timestamps_us'], a['imu'], impact_g)
    n_imp = int((ok & imp).sum())
    ok &= ~imp
    ok &= (np.arange(len(ok)) % STRIDE) == 0
    return ok, n_imp


def main():
    t0 = datetime.now()
    res = {'schema': 'jf-train-v2-data-checks/v1', 'created_at': t0.isoformat(timespec='seconds'),
           'script': 'experiments/20261025_train_v2/check_v2_data.py', 'dataset': NAME, 'checks': {}, 'fails': []}
    C = res['checks']

    def check(name, ok, **info):
        C[name] = dict(ok=bool(ok), **info)
        if not ok:
            res['fails'].append(name)
        print(('OK  ' if ok else 'NG  ') + name, json.dumps(info, ensure_ascii=False)[:300], flush=True)

    r = subprocess.run(['sha256sum', '-c', 'protocol_v2.sha256'], cwd=str(REP), capture_output=True, text=True)
    check('protocol_locked', r.returncode == 0)
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    sealed = list(p['seal']['logs'])
    h1 = sorted(p['holdouts']['H1']['scenes'])
    ai_logs = [x for v in p['tables']['dataset']['ai_logs'].values() for x in v]
    card = json.loads((DS / 'dataset_card.json').read_text(encoding='utf-8'))
    splits = json.loads((DS / 'splits.json').read_text(encoding='utf-8'))
    tcard = json.loads((TBL / 'tables_card.json').read_text(encoding='utf-8'))
    res['dataset_card_sha256'] = sha(DS / 'dataset_card.json')
    res['splits_sha256'] = sha(DS / 'splits.json')
    res['modes_json_sha256'] = sha(ROOT / 'data/modes.json')

    # ---- 表の sha256 が tables_card と同じ
    tfiles = {}
    for fn, info in tcard['files'].items():
        tfiles[fn] = sha(TBL / fn)
    check('tables_sha256_match_card', all(tfiles[f] == tcard['files'][f]['sha256'] for f in tfiles), sha256=tfiles)
    check('tables_card_dataset_card_same', tcard['dataset_card_sha256'] == res['dataset_card_sha256'])

    # ---- 書き直しの確かめ: clean_driving_20261011 で B の train.log の数
    bs = json.loads((BASE / 'splits.json').read_text(encoding='utf-8'))
    bcount = {}
    for sp in ('train', 'val'):
        n_use = n_imp = 0
        for s in bs[sp]:
            a = load_scene(BASE, s)
            ok, ni = usable(a)
            n_use += int(ok.sum())
            n_imp += ni
        bcount[sp] = {'samples': n_use, 'impact_masked': n_imp}
    check('recount_reproduces_B_trainlog', bcount['train'] == {'samples': 5912, 'impact_masked': 3522}
          and bcount['val'] == {'samples': 1559, 'impact_masked': 1083}, clean_driving=bcount,
          b_train_log={'train': 5912, 'val': 1559, 'impact_masked': [3522, 1083]})

    # ---- 封・H1・検証
    src = card['source_scenes']
    check('sealed_not_in_source_scenes', not [s for s in src if s.rsplit('-', 1)[0] in set(sealed)],
          sealed_logs=len(sealed))
    check('ai_logs_equal_protocol_46', card['ai_mix']['ai_logs'] == ai_logs and len(ai_logs) == 46)
    check('val_equals_H1_11', sorted(splits['val']) == h1, val=len(splits['val']))
    check('H1_not_in_train', not (set(h1) & set(splits['train'])))
    check('train_val_disjoint', not (set(splits['train']) & set(splits['val'])))
    check('genre9_in_card_modes', any(m['id'] == 9 for m in card['modes']),
          card_modes=[m['id'] for m in card['modes']])

    # ---- 学習が読むサンプルを腕ごとに数える
    drop = {k: table(TBL / ('drop_%s.npz' % k), 'drop') for k in ('v2h', 'v2m', 'bfast')}
    w_m = table(TBL / 'weights_v2m.npz', 'weight')
    w_b = table(TBL / 'weights_bfast.npz', 'weight')
    rm = table(TBL / 'remap_v2m.npz', 'mode')
    for nm, tb_ in list(drop.items()) + [('w_v2m', w_m), ('w_bf', w_b), ('remap', rm)]:
        miss = [s for s in splits['train'] + splits['val'] if s not in tb_]
        check('table_has_all_scenes_%s' % nm, not miss, missing=miss[:3])
    base_train = set(bs['train'])
    arms = {k: {'samples': 0, 'by_source': {}, 'by_mode': {}, 'impact_masked': 0, 'dropped': 0,
                'w_sum': 0.0, 'w_ai_sum': 0.0, 'w_sq': 0.0, 'by_group': {}} for k in ('v2h', 'v2m', 'bfast')}
    val_n = {'v2h': 0, 'v2m': 0, 'bfast': 0, 'none': 0}
    base_v2 = {'samples': 0}
    used_frames = {}          # (scene) -> 使うフレーム番号の和集合
    used_ts = {}
    problems = []
    g9_bad = 0
    g9_rows = 0
    teach_modes = {}
    lab_vs_exec = []
    arrays_ok = True
    key_need = ('timestamps_us', 'cam_files', 'tof', 'imu', 'vehicle', 'vehicle_valid', 'future_actions', 'future_valid',
                'mode', 'source', 'label_index', 'action_exec')
    ref_rows = []
    for sp in ('train', 'val'):
        for s in splits[sp]:
            a = load_scene(DS, s)
            n = len(a['timestamps_us'])
            for k in key_need:
                if k not in a:
                    arrays_ok = False
                    problems.append('%s: %s がありません' % (s, k))
            if a['tof'].shape[1:] != (42, 54) or a['imu'].shape[1:] != (10, 6) or a['vehicle'].shape[1:] != (4,) \
                    or a['future_actions'].shape[1:] != (10, 2):
                arrays_ok = False
                problems.append('%s: 形が違います' % s)
            if sp == 'train':        # train_data.action_reference と同じ選び方(drop に関係なく学習シーンの全部)
                m = a['vehicle_valid'].astype(bool) & a.get('erpm_valid', np.ones(n, bool)).astype(bool) \
                    & (np.abs(a['vehicle'][:, 3]) < 1)
                ref_rows.append(np.column_stack([a['vehicle'][m, :2], a['source'][m]]))
            if sp == 'val':
                ok0, _ = usable(a)
                val_n['none'] += int(ok0.sum())
            for arm in arms:
                d = per_frame(drop[arm], s, n, False, bool)
                ok, ni = usable(a, d)
                if sp == 'val':
                    val_n[arm] += int(ok.sum())
                    if ok.any():
                        used_frames.setdefault(s, set()).update(np.nonzero(ok)[0].tolist())
                    continue
                A = arms[arm]
                A['dropped'] += int((a['future_valid'] & a['vehicle_valid'] & (a['mode'] != 0) & d).sum())
                A['impact_masked'] += ni
                A['samples'] += int(ok.sum())
                srcv = a['source'][ok]
                for v_ in np.unique(srcv):
                    A['by_source'][str(int(v_))] = A['by_source'].get(str(int(v_)), 0) + int((srcv == v_).sum())
                mode = a['mode'][ok].astype(np.int64)
                if arm == 'v2m':
                    r = per_frame(rm, s, n, -1, np.int16)[ok]
                    mode = np.where(r >= 0, r, mode)
                    g9 = r == 9
                    g9_rows += int(g9.sum())
                    g9_bad += int((g9 & (srcv != 1)).sum())
                    w = per_frame(w_m, s, n, np.nan, np.float32)[ok]
                elif arm == 'bfast':
                    w = per_frame(w_b, s, n, np.nan, np.float32)[ok]
                else:
                    w = np.ones(int(ok.sum()), np.float32)
                if not np.isfinite(w).all():
                    problems.append('%s %s: 重みが無いフレームがあります' % (arm, s))
                for v_ in np.unique(mode):
                    A['by_mode'][str(int(v_))] = A['by_mode'].get(str(int(v_)), 0) + int((mode == v_).sum())
                A['w_sum'] += float(w.sum())
                A['w_sq'] += float((w.astype(np.float64) ** 2).sum())
                A['w_ai_sum'] += float(w[srcv == 1].sum())
                if s in base_train:
                    A['by_group']['base41'] = A['by_group'].get('base41', 0) + int(ok.sum())
                else:
                    hu = int((srcv == 0).sum())
                    ai = int((srcv == 1).sum())
                    A['by_group']['new_human'] = A['by_group'].get('new_human', 0) + hu
                    if ai:
                        wv = w[srcv == 1]
                        A['by_group']['ai_w1'] = A['by_group'].get('ai_w1', 0) + int((wv == 1.0).sum())
                        A['by_group']['ai_w05'] = A['by_group'].get('ai_w05', 0) + int((wv == 0.5).sum())
                        A['by_group']['ai'] = A['by_group'].get('ai', 0) + ai
                if ok.any():
                    used_frames.setdefault(s, set()).update(np.nonzero(ok)[0].tolist())
                    used_ts.setdefault(s.rsplit('-', 1)[0], []).append(a['timestamps_us'][ok].astype(np.int64))
                if arm == 'bfast' and s not in base_train:
                    for v_ in np.unique(a['mode'][ok]):
                        teach_modes[str(int(v_))] = teach_modes.get(str(int(v_)), 0) + int((a['mode'][ok] == v_).sum())
                    # 正解(k=0)と車に出た値(action_exec)の差(ゲイン 1.0 の割り戻し = 車に出た値)
                    ex = a['action_exec'][ok].astype(np.float64)
                    fa = a['future_actions'][ok][:, 0, :].astype(np.float64)
                    lab_vs_exec.append(np.abs(fa - ex))
            if sp == 'train' and s in base_train:
                ok_b, _ = usable(a)
                base_v2['samples'] += int(ok_b.sum())
    check('arrays_present_and_shapes', arrays_ok, problems=problems[:5])
    check('weights_finite_for_used_frames', not [x for x in problems if '重み' in x])
    check('base41_in_v2_equals_B_5912', base_v2['samples'] == 5912, base41_samples_in_v2=base_v2['samples'],
          note='build_ai_dataset の人のラベルのマスク(human_label_mask)で clean_driving の学習 41 シーンが減っていないか')
    for arm, key in (('v2h', 'v2h'), ('v2m', 'v2m'), ('bfast', 'bfast')):
        A = arms[arm]
        e = EXPECT[key]
        check('train_samples_%s_within_1pct' % arm, abs(A['samples'] - e) <= TOL * e,
              samples=A['samples'], expected=e, diff=A['samples'] - e, by_group=A['by_group'], by_mode=A['by_mode'])
    for arm in ('v2h', 'v2m', 'bfast'):
        check('val_samples_%s_is_1559' % arm, val_n[arm] == 1559, val=val_n[arm])
    v2m = arms['v2m']
    share_w = v2m['w_ai_sum'] / v2m['w_sum'] if v2m['w_sum'] else None
    share_u = v2m['by_source'].get('1', 0) / v2m['samples'] if v2m['samples'] else None
    check('v2m_ai_weighted_share_le_15pct', share_w is not None and share_w <= 0.15,
          weighted=round(share_w, 4), unweighted=round(share_u, 4), plan_weighted=0.134, plan_unweighted=0.154)
    bf = arms['bfast']
    share_b = bf['by_source'].get('1', 0) / bf['samples'] if bf['samples'] else None
    share_bw = bf['w_ai_sum'] / bf['w_sum'] if bf['w_sum'] else None
    check('bfast_teacher_within_1pct', abs(bf['by_source'].get('1', 0) - EXPECT['teacher']) <= TOL * EXPECT['teacher'],
          teacher_samples=bf['by_source'].get('1', 0), expected=EXPECT['teacher'],
          share_unweighted=round(share_b, 4), share_weighted=round(share_bw, 4))
    check('bfast_teacher_mode_5_only', set(teach_modes) == {'5'}, teacher_modes=teach_modes)
    check('genre9_only_ai_lap_frames', g9_bad == 0 and g9_rows > 0, genre9_samples=g9_rows, genre9_not_ai=g9_bad)
    check('v2h_has_no_ai_frames', arms['v2h']['by_source'].get('1', 0) == 0, by_source=arms['v2h']['by_source'])
    lve = np.concatenate(lab_vs_exec) if lab_vs_exec else np.zeros((0, 2))
    check('bfast_teacher_label_equals_car_output', len(lve) and float(np.median(lve)) < 1.0,
          median_abs_us=[round(float(np.median(lve[:, 0])), 3), round(float(np.median(lve[:, 1])), 3)] if len(lve) else None,
          p99_abs_us=[round(float(np.percentile(lve[:, 0], 99)), 2), round(float(np.percentile(lve[:, 1], 99)), 2)] if len(lve) else None,
          note='k=0 の正解 = このフレームで決めた out(ゲイン 1.0 の割り戻し)。action_exec はラベルの行の車に出た値')
    for arm in arms:
        A = arms[arm]
        A['weight_mean'] = round(A['w_sum'] / A['samples'], 4) if A['samples'] else None
        A['eff_samples'] = round(A['w_sum'] ** 2 / A['w_sq'], 1) if A['w_sq'] else None
    rr = np.concatenate(ref_rows) if ref_rows else np.zeros((0, 3))
    ref_all = (round(float(np.median(rr[:, 0])), 1), round(float(np.median(rr[:, 1])), 1))
    hum = rr[rr[:, 2] == 0]
    ref_h = (round(float(np.median(hum[:, 0])), 1), round(float(np.median(hum[:, 1])), 1))
    check('action_reference_within_2us_of_B', abs(ref_all[0] - B_REF[0]) <= 2 and abs(ref_all[1] - B_REF[1]) <= 2,
          v2_all=ref_all, v2_human_only=ref_h, n=int(len(rr)), B=B_REF,
          note='train_data.action_reference(学習シーンの止まっているフレーム。drop に関係なく全部)。段 0 の確かめ 7')

    # ---- 封のフレーム(画像の inode)・H1 の画像が学習に無いか
    def inodes_of_logs(logs):
        out = set()
        for lg in logs:
            for sd in sorted((ROOT / 'data/raw' / lg).iterdir()):
                cd = sd / 'CAM_FRONT'
                if cd.is_dir():
                    for f in os.scandir(cd):
                        if f.name.endswith('.jpg'):
                            out.add((f.inode(), os.stat(f.path).st_dev))
        return out
    seal_ino = inodes_of_logs(sealed)
    ds_ino = set()
    ds_ino_train = set()
    for sp in ('train', 'val'):
        for s in splits[sp]:
            cd = DS / 'samples/CAM_FRONT' / s
            for f in os.scandir(cd):
                key = (f.inode(), os.stat(f.path).st_dev)
                ds_ino.add(key)
                if sp == 'train':
                    ds_ino_train.add(key)
    check('sealed_images_not_in_dataset', not (seal_ino & ds_ino), sealed_images=len(seal_ino), dataset_images=len(ds_ino))
    h1_ino = set()
    for s in h1:
        lg = s.rsplit('-', 1)[0]
        cd = ROOT / 'data/raw' / lg / s / 'CAM_FRONT'
        for f in os.scandir(cd):
            if f.name.endswith('.jpg'):
                h1_ino.add((f.inode(), os.stat(f.path).st_dev))
    check('H1_images_not_in_train_scenes', not (h1_ino & ds_ino_train), h1_images=len(h1_ino))
    cam_bad = []
    for sp in ('train', 'val'):
        for s in splits[sp]:
            with np.load(DS / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                cf = z['cam_files'].astype(str)
            if any(lg in x for x in cf[:1] for lg in sealed):
                cam_bad.append(s)
    check('cam_files_not_from_sealed', not cam_bad)

    # ---- 介入の検証セットと、学習に使うフレームの重なり(時刻)
    ivc = json.loads((IVS / 'dataset_card.json').read_text(encoding='utf-8'))
    ov = 0
    for s in ivc['source_scenes']:
        lg = s.rsplit('-', 1)[0]
        if lg not in used_ts:
            continue
        with np.load(IVS / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            it = z['timestamps_us'].astype(np.int64)
        for kt in used_ts[lg]:
            ov += int(np.isin(kt, it).sum())
    check('interv_overlap_zero', ov == 0, overlap_frames=ov)

    # ---- 画像: 学習・検証に使うフレーム(3 つの腕の和集合)を全部読む
    n_img = n_bad = 0
    shapes = {}
    for s, frs in sorted(used_frames.items()):
        with np.load(DS / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
            cf = z['cam_files'].astype(str)
        for i in sorted(frs):
            im = cv2.imread(str(DS / cf[i]))
            n_img += 1
            if im is None:
                n_bad += 1
                continue
            shapes[str(im.shape)] = shapes.get(str(im.shape), 0) + 1
    check('images_readable_224x224x3', n_bad == 0 and set(shapes) == {'(224, 224, 3)'}, read=n_img, unreadable=n_bad,
          shapes=shapes)

    # ---- まとめ
    res['arms'] = {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in arms.items()}
    res['val_samples'] = val_n
    res['base41_samples_in_v2'] = base_v2['samples']
    res['clean_driving_recount'] = bcount
    res['card_summary'] = {k: card.get(k) for k in ('scenes', 'train_scenes', 'val_scenes', 'frames', 'hours')}
    res['card_summary'].update({k: card['ai_mix'][k] for k in ('frames', 'usable_frames', 'human_frames_removed_by_label_mask',
                                                              'action_reference_check')})
    res['card_summary']['skipped_scenes'] = len(card['ai_mix']['skipped_scenes'])
    res['seconds'] = round((datetime.now() - t0).total_seconds(), 1)
    res['ok'] = not res['fails']
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print('外れ:', res['fails'])
    sys.exit(0 if res['ok'] else 1)


if __name__ == '__main__':
    main()
