#!/usr/bin/env python3
"""解析 B の机上の試し(2026-10-25): 記録した入力(カメラ・ToF・IMU・回転数)をモデル B にもう一度入れて、
「入力の 1 つだけを変えたら、モデルのスロットルがどれだけ変わるか」を測る(開ループ。車は動かさない)。

    jf-heavy --lock big --mem 3G -- /home/jetson/miniconda/envs/jetracer/bin/python -I cond_replay.py

入れ方は走行(jetformer/drive.py)と同じ: 撮像時刻以前の最新の ToF・直近 10 個の IMU・最新の車両の eRPM、
今の操作値は入れない(B は command_input=false)、ジャンル = 記録の mode_id。FP32 の torch(走行は FP16 のエンジン)。
変えるもの:
  orig        そのまま(記録のモデルの出力と比べて、入れ方が合っているかを確かめる)
  erpm_up     入力の eRPM を ×1.15(電池が強くて速く回っているときに当たる)
  erpm_dn     入力の eRPM を ×0.87
  img_up      画像の画素を ×1.10(明るい)/ img_dn ×0.90(暗い)
スロットルは前進の量 = 1496 − 出力 [µs](先読み 3 = 予測の 3 フレーム目。ゲインの前)。
出力: /home/jetson/jetformer_reports/bench_20261024_report/cond_data/cond_replay.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

JF = Path('/home/jetson/jetformer')
sys.path.insert(0, str(JF))
import cv2  # noqa: E402
from jetformer.train_data import normalize_inputs  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.policies import post as post_mod  # noqa: E402

RAW = JF / 'data' / 'raw'
LAB = JF / 'data' / 'labels'
RUN = JF / 'data' / 'runs' / 'clean_driving_20261011-nocmd-s3-20261011-024626'
OUT = Path('/home/jetson/jetformer_reports/bench_20261024_report/cond_data')
LOGS = {9: '20261024-162033', 11: '20261024-183123', 12: '20261024-183437', 18: '20261024-185703'}
DET = ['?', 'R1', 'GM_in', 'GM_out', 'DE', 'SW', 'HS', 'LH']
VARIANTS = {'orig': (1.0, 1.0), 'erpm_up': (1.15, 1.0), 'erpm_dn': (0.87, 1.0), 'img_up': (1.0, 1.10), 'img_dn': (1.0, 0.90)}
NEUTRAL = 1496.0
STEP = 2
BATCH = 48


def cat(lg, name, keys):
    parts = {k: [] for k in keys}
    for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
        try:
            with np.load(s + '/' + name + '.npz') as z:
                for k in keys:
                    parts[k].append(z[k])
        except Exception:
            continue
    o = {k: np.concatenate(v) for k, v in parts.items()}
    idx = np.argsort(o['timestamps_us'], kind='stable')
    return {k: v[idx] for k, v in o.items()}


def main():
    model, cfg, _, _ = runtime.load_run(RUN, 'best', device='cuda')
    num_modes = cfg['num_modes']
    out = {'doc': __doc__, 'logs': {}}
    for no, lg in LOGS.items():
        with np.load(LAB / (lg + '.npz')) as z:
            lab = {k: z[k] for k in ('t_us', 'ai', 'dry_run', 'hold', 'assist', 'loc_detail', 'raw', 'erpm')}
        A = cat(lg, 'AI_CMD', ['timestamps_us', 'model_pred', 'mode_id', 'raw'])
        T = cat(lg, 'TOF_FRONT', ['timestamps_us', 'depth'])
        I = cat(lg, 'IMU', ['timestamps_us', 'accel', 'gyro'])
        V = cat(lg, 'VEHICLE', ['timestamps_us', 'erpm'])
        C = cat(lg, 'CAM_FRONT', ['timestamps_us'])
        files = []
        for s in sorted(glob.glob(str(RAW / lg / (lg + '-*')))):
            with np.load(s + '/CAM_FRONT.npz') as c:
                files += [s + '/' + f for f in c['files']]
        ct = np.concatenate([np.load(s + '/CAM_FRONT.npz')['timestamps_us'] for s in sorted(glob.glob(str(RAW / lg / (lg + '-*'))))])
        fmap = dict(zip(ct.tolist(), files))
        imu_feat = np.concatenate([I['accel'], np.deg2rad(I['gyro'])], axis=1).astype(np.float32)
        drv = lab['ai'] & ~lab['dry_run'] & ~lab['hold'] & (lab['assist'] == 0)
        sel = np.flatnonzero(drv)[::STEP]
        rec = {}
        ai_k = np.searchsorted(A['timestamps_us'], lab['t_us'][sel])
        ai_k = np.clip(ai_k, 0, len(A['timestamps_us']) - 1)
        okA = A['timestamps_us'][ai_k] == lab['t_us'][sel]
        sel, ai_k = sel[okA], ai_k[okA]
        res = {v: np.full((len(sel), 10, 2), np.nan, np.float32) for v in VARIANTS}
        for b0 in range(0, len(sel), BATCH):
            idx = sel[b0:b0 + BATCH]
            base = []
            for i in idx:
                cam_t = int(lab['t_us'][i])
                img = cv2.imread(fmap[cam_t])
                jt = np.searchsorted(T['timestamps_us'], cam_t, side='right') - 1
                ji = np.searchsorted(I['timestamps_us'], cam_t, side='right')
                jv = np.searchsorted(V['timestamps_us'], cam_t, side='right') - 1
                hist = imu_feat[max(ji - 10, 0):ji]
                if len(hist) < 10:
                    hist = np.vstack([np.repeat(hist[:1], 10 - len(hist), 0), hist])
                base.append((img, T['depth'][max(jt, 0)], hist, float(V['erpm'][max(jv, 0)])))
            for vname, (ef, gf) in VARIANTS.items():
                ims, tofs, imus, sts = [], [], [], []
                for img, tof, hist, er in base:
                    im_in = img if gf == 1.0 else np.clip(img.astype(np.float32) * gf, 0, 255).astype(np.uint8)
                    veh = np.array([1441.0, NEUTRAL, 2000.0, er * ef], np.float32)
                    im, tf, iu, st = normalize_inputs(im_in, tof, hist, veh, use_command=False)
                    ims.append(im); tofs.append(tf); imus.append(iu); sts.append(st)
                modes = np.zeros((len(idx), num_modes), np.float32)
                modes[np.arange(len(idx)), A['mode_id'][ai_k[b0:b0 + BATCH]].astype(int)] = 1.0
                r = runtime.predict(model, {'image': np.stack(ims), 'tof': np.stack(tofs), 'imu': np.stack(imus),
                                            'state': np.stack(sts), 'mode': modes})
                res[vname][b0:b0 + len(idx)] = r['actions'] * post_mod.PWM_SCALE + post_mod.PWM_CENTER
            print(no, b0, len(sel), flush=True)
        ld = lab['loc_detail'][sel].astype(int)
        rec_thr = NEUTRAL - A['model_pred'][ai_k, 2, 1]
        orig_thr = NEUTRAL - res['orig'][:, 2, 1]
        d = {'frames': int(len(sel)),
             'check_thr_mae_us': round(float(np.nanmean(np.abs(orig_thr - rec_thr))), 2),
             'check_steer_mae_us': round(float(np.nanmean(np.abs(res['orig'][:, 2, 0] - A['model_pred'][ai_k, 2, 0]))), 2),
             'check_thr_rec_p50': round(float(np.nanmedian(rec_thr)), 1), 'check_thr_orig_p50': round(float(np.nanmedian(orig_thr)), 1)}
        sec = {}
        for c, nm in enumerate(DET):
            m = ld == c
            if m.sum() < 10:
                continue
            row = {'n': int(m.sum()), 'thr_orig_p50': round(float(np.median(orig_thr[m])), 1),
                   'steer_orig_p50': round(float(np.median(res['orig'][m, 2, 0] - 1441)), 1)}
            for v in VARIANTS:
                if v == 'orig':
                    continue
                dthr = (NEUTRAL - res[v][m, 2, 1]) - orig_thr[m]
                dst = res[v][m, 2, 0] - res['orig'][m, 2, 0]
                row['d_thr_' + v] = round(float(np.mean(dthr)), 2)
                row['d_steer_' + v] = round(float(np.mean(dst)), 2)
            sec[nm] = row
        allrow = {}
        for v in VARIANTS:
            if v == 'orig':
                continue
            allrow['d_thr_' + v] = round(float(np.mean((NEUTRAL - res[v][:, 2, 1]) - orig_thr)), 2)
            allrow['d_steer_abs_' + v] = round(float(np.mean(np.abs(res[v][:, 2, 0] - res['orig'][:, 2, 0]))), 2)
        d['all'] = allrow
        d['sections'] = sec
        out['logs'][str(no)] = d
        np.savez_compressed(OUT / ('replay_%d.npz' % no), sel=sel, loc=ld, rec_thr=rec_thr,
                            **{'thr_' + v: NEUTRAL - res[v][:, 2, 1] for v in VARIANTS},
                            **{'steer_' + v: res[v][:, 2, 0] for v in VARIANTS})
        print(no, json.dumps(d, ensure_ascii=False)[:600], flush=True)
    (OUT / 'cond_replay.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
