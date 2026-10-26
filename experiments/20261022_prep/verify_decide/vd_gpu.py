#!/usr/bin/env python3
"""確認 V-decide(2026-10-23)の GPU の仕事。jf-heavy --lock big を通す。車・装置には触れない(PWM は出さない)。

選ばれた C1(bench_20261022-slow-sec-s1)のエンジンに、10/10 の AI 走行のフレームを入れ直し、出力が範囲内かを見る。
フレームの選び方は自分で書いた(ai_mix_20261011 の aligned、ログ名 11 本、source == 1 かつ vehicle_valid)。
エンジンは走行と同じ TRTRunner・policies.check_io・post.Stage で読む(走行と同じ道を通すため)。
出力は保存した desk の sanity_SLOW-SEC1.npz とも比べる。

    /home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 \
        /home/jetson/miniconda/envs/jetracer/bin/python -u vd_gpu.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
RUN = 'bench_20261022-slow-sec-s1'
OUT = Path('/home/jetson/jetformer_reports/prep_20261022/verify_decide_raw')
SAVED = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/sanity_SLOW-SEC1.npz')
LOGS10 = ('20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
          '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
          '20261010-204605')
MODE = 5


def log(*a):
    print('[%s]' % time.strftime('%H:%M:%S'), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    import cv2
    import torch
    from jetformer import config, drive, policies
    from jetformer.policies import post
    from jetformer.train_data import normalize_inputs
    from jetformer.trt_runtime import TRTRunner

    root = config.DATASET_DIR / 'ai_mix_20261011'
    splits = json.load(open(root / 'splits.json'))
    scenes = sorted({s for k in splits for s in splits[k] if s.split('-')[0] + '-' + s.split('-')[1] in LOGS10})
    frames = []
    for s in scenes:
        with np.load(root / 'aligned' / (s + '.npz')) as z:
            m = (np.asarray(z['source']) == 1) & np.asarray(z['vehicle_valid']).astype(bool)
            for i in np.flatnonzero(m):
                frames.append((s, int(i), int(z['timestamps_us'][i])))
    frames.sort(key=lambda r: (r[0][:15], r[2], r[0], r[1]))
    log('frames', len(frames), 'scenes', len(scenes))

    info = drive.model_info(RUN)
    eng = REPO / 'data' / 'runs' / RUN / drive.ENGINE_FILE
    res_out = {'run': RUN, 'engine_sha256': sha(eng), 'engine_size': eng.stat().st_size,
               'engine_mtime_ns': eng.stat().st_mtime_ns,
               'model_info': {k: info.get(k) for k in ('drivable', 'why_not', 'num_modes', 'policy', 'command_input',
                                                       'steer_center_us', 'throttle_neutral_us', 'engine_check')}}
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    runner = TRTRunner(str(eng))
    nm = int(info['num_modes'])
    ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                {n: list(runner.shapes[n]) for n in runner.outputs}, nm)
    res_out['check_io'] = [bool(ok), why]
    ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
    stage = post.Stage(info, ref, fps=config.CAMERA['fps'])
    params = dict(drive.DEFAULT_PARAMS)
    params.update({'steer_gain': 1.0, 'throttle_gain': 1.0, 'lookahead': 3, 'tof_guard': 1, 'ensemble_j': 0,
                   'speed_cap': 0.0, 'zone_caps': 0, 'tcp_mode': 0, 'speed_scale': 1.0})
    oh = np.zeros((1, nm), np.float32)
    oh[0, MODE] = 1.0
    zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
    for _ in range(3):
        runner.run_numpy(**zeros)
    uc = bool(info.get('command_input', False))
    n = len(frames)
    mp_all = np.full((n, 10, 2), np.nan, np.float32)
    pr_all = np.full((n, 10, 2), np.nan, np.float32)
    erpm = np.zeros(n)
    nonfinite_raw = 0
    post_err = 0
    ms = []
    cur, a, prev = None, None, None
    t0 = time.time()
    for k, (s, i, t_us) in enumerate(frames):
        if s != cur:
            with np.load(root / 'aligned' / (s + '.npz')) as z:
                a = {q: np.asarray(z[q]) for q in ('cam_files', 'tof', 'imu', 'vehicle')}
            cur = s
        if s != prev:
            stage.reset()
            prev = s
        img = cv2.imread(str(root / str(a['cam_files'][i])))
        erpm[k] = float(a['vehicle'][i, 3])
        im, tf, iu, sv = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], uc)
        feed = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None], 'mode': oh}
        q0 = time.perf_counter()
        r = runner.run_numpy(**feed)
        ms.append((time.perf_counter() - q0) * 1e3)
        if not all(np.isfinite(v).all() for v in r.values()):
            nonfinite_raw += 1
        mp = stage.model_pred(r)
        mp_all[k] = mp
        try:
            p, _ = stage.apply(r, mp, post.speed_of(erpm[k]), params, int(t_us) * 1000)
            pr_all[k] = p
        except post.PostError:
            post_err += 1
        if (k + 1) % 4000 == 0:
            log(k + 1, '/', n, '%.0fs' % (time.time() - t0))
    lo, hi = float(config.DRIVE['min_us']), float(config.DRIVE['max_us'])
    fin = np.isfinite(mp_all).all(axis=(1, 2)) & np.isfinite(pr_all).all(axis=(1, 2))
    v = -erpm * 0.5e-3
    low = v < 1.0
    brake = mp_all[:, 0, 1] > 1598.0
    res_out.update({
        'frames': n, 'scenes': len(scenes), 'logs': sorted({f[0][:15] for f in frames}),
        'nonfinite_raw_frames': nonfinite_raw, 'post_error_frames': post_err, 'nonfinite_frames': int((~fin).sum()),
        'out_of_500_2500_values': int(((mp_all < 500) | (mp_all > 2500)).sum() + ((pr_all < 500) | (pr_all > 2500)).sum()),
        'outside_drive_range_values': int(((pr_all < lo) | (pr_all > hi)).sum()), 'values': int(pr_all.size),
        'drive_range_us': [lo, hi],
        'steer_min_max': [float(np.nanmin(pr_all[:, :, 0])), float(np.nanmax(pr_all[:, :, 0]))],
        'thr_min_max': [float(np.nanmin(pr_all[:, :, 1])), float(np.nanmax(pr_all[:, :, 1]))],
        'pred_equals_model_pred': bool(np.array_equal(pr_all, mp_all)),
        'lowspeed_frames': int(low.sum()), 'lowspeed_brake_frames': int((low & brake).sum()),
        'any_speed_brake_frames': int(brake.sum()),
        'run_numpy_median_ms': round(float(np.median(ms)), 3), 'run_numpy_p99_ms': round(float(np.percentile(ms, 99)), 3),
        'seconds': round(time.time() - t0, 1)})
    # 保存した desk の sanity と比べる
    with np.load(SAVED) as z:
        same_rows = (len(z['scene']) == n and all(z['scene'][j] == frames[j][0] and int(z['frame'][j]) == frames[j][1]
                                                  for j in range(n)))
        res_out['saved_same_rows'] = bool(same_rows)
        if same_rows:
            d = np.abs(z['model_pred'].astype(np.float64) - mp_all.astype(np.float64))
            res_out['saved_model_pred_absdiff_max_us'] = float(d.max())
            res_out['saved_model_pred_absdiff_p99_us'] = float(np.percentile(d, 99))
            res_out['saved_erpm_same'] = bool(np.array_equal(np.asarray(z['erpm'], np.float64), erpm))
    np.savez_compressed(OUT / 'vd_sanity_SLOW-SEC1.npz', model_pred=mp_all, pred=pr_all, erpm=erpm,
                        scene=np.array([f[0] for f in frames]), frame=np.array([f[1] for f in frames], np.int32))
    json.dump(res_out, open(OUT / 'vd_gpu.json', 'w'), ensure_ascii=False, indent=1, default=str)
    log(json.dumps({k: res_out[k] for k in res_out if k not in ('model_info',)}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
