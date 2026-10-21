#!/usr/bin/env python3
"""V-models(2026-10-21・反証する側)の GPU の仕事。必ず jf-heavy --lock big を通す。車・装置には触れない(PWM を出さない)。

  - 走る 6 本(data/bench_candidates.json の B と候補)のエンジンを、drive._run と同じ順で読む
    (torch の CUDA の初期化 → TRTRunner → policies.check_io → 0 で空回し 3 回)
  - 入力は aligned から自分で作る(train_data.normalize_inputs = 走行と同じ正規化。policy_zoo・desk_gpu を使わない)
      (1) 保留 primary + 介入 interv 6,240 フレーム: 人のジャンル(保存した FP32 の予測と比べる)と、ジャンル 5(走る日)
      (2) 10/10 の AI 走行 11 本の AI のフレーム: ジャンル 5・基準の設定で Stage.apply(走行と同じ後処理)
  - PyTorch FP32(CPU)と TensorRT の差を、engine_check と違うフレーム(AI 走行の低速・R1 の手前)で見る。新しい 3 本は last も
  - 速さ: 実フレームで run_numpy を 100 回の空回しの後 600 回
出力: <out>/gpu_<id>.npz・<out>/gpu.json
"""
import gc, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path('/home/jetson/jetformer'); sys.path.insert(0, str(REPO))
OUT = Path(sys.argv[1])
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None
DS = REPO / 'data' / 'datasets'
AI_DS = 'ai_mix_20261011'

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

def main():
    import cv2
    from jetformer import drive, policies, config
    from jetformer.policies import post as post_mod
    from jetformer.train_data import normalize_inputs
    C = json.load(open(REPO / 'data' / 'bench_candidates.json'))
    cands = [(c['id'], c['run']) for c in C['candidates'] if ONLY is None or c['id'] in ONLY]
    params = dict(drive.DEFAULT_PARAMS); params.update(C['baseline']['params'])
    mode_day = int(C['baseline']['mode_id'])
    F = np.load(OUT / 'frames.npz', allow_pickle=False)
    nho = len(F['cam'])
    # AI 走行のフレーム(labels の ai_log が True の 10/10 のログ。aligned の source == 1 & vehicle_valid)
    ai_logs = sorted(json.load(open(p))['log'] for p in (REPO / 'data/labels').glob('20261010-*.json')
                     if json.load(open(p)).get('ai_log'))
    sp = json.load(open(DS / AI_DS / 'splits.json'))
    ai_rows = []
    for s in sp['train'] + sp['val']:
        if s[:15] not in ai_logs:
            continue
        with np.load(DS / AI_DS / 'aligned' / (s + '.npz')) as z:
            ok = (z['source'] == 1) & z['vehicle_valid']
            for i in np.nonzero(ok)[0]:
                ai_rows.append((s, int(i), int(z['timestamps_us'][i])))
    ai_rows.sort(key=lambda r: (r[0][:15], r[2]))
    nai = len(ai_rows)
    log('保留 %d フレーム・AI 走行 %d フレーム(%d 本)' % (nho, nai, len(ai_logs)))

    # ---- FP32(CPU)の参照に使うフレームを先に決める(AI 走行の低速 24・ばらばら 24・R1 の手前 16)
    rng = np.random.default_rng(20261021)
    ai_erpm = np.zeros(nai)
    for k, (s, i, t) in enumerate(ai_rows):
        pass
    # 低速は後で erpm を見てから選ぶので、ここでは index だけ用意する

    import torch
    torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    from jetformer.trt_runtime import TRTRunner
    E = {}
    info_all = {}
    for cid, run in cands:
        info = drive.model_info(run)
        assert info is not None and info['drivable'], (cid, info and info['why_not'])
        runner = TRTRunner(str(drive.RUNS_DIR / run / drive.ENGINE_FILE))
        ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                    {n: list(runner.shapes[n]) for n in runner.outputs}, info['num_modes'])
        zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
        for _ in range(3):
            runner.run_numpy(**zeros)
        ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
        stage = post_mod.Stage(info, ref, fps=config.CAMERA['fps'])
        nm = int(info['num_modes'])
        E[cid] = {'runner': runner, 'stage': stage, 'nm': nm, 'cmd': bool(info['command_input']), 'run': run,
                  'ho_h': np.full((nho, 10, 2), np.nan, np.float32), 'ho_5': np.full((nho, 10, 2), np.nan, np.float32),
                  'ai_model': np.full((nai, 10, 2), np.nan, np.float32), 'ai_pred': np.full((nai, 10, 2), np.nan, np.float32),
                  'ms': [], 'nonfinite': 0, 'post_err': 0}
        info_all[cid] = {'run': run, 'check_io_ok': bool(ok), 'check_io_why': why,
                         'inputs': {n: list(runner.shapes[n]) for n in runner.inputs},
                         'outputs': {n: list(runner.shapes[n]) for n in runner.outputs}, 'num_modes': nm,
                         'post_kind': post_mod.kind_of(info) if stage.post is not None else None, 'stage_failed': stage.failed}
        log(cid, 'check_io', ok, why, info_all[cid]['inputs'])

    def onehot(nm, g):
        oh = np.zeros((1, nm), np.float32); oh[0, int(g)] = 1.0
        return oh

    # ---- (1) 保留
    t0 = time.time()
    feeds_keep = {}
    keep_ho = set(np.nonzero(F['r1pre'])[0][::max(1, int(F['r1pre'].sum()) // 16)][:16].tolist())
    for k in range(nho):
        img = cv2.imread(str(F['cam'][k]))
        if img is None:
            raise SystemExit('画像が読めません: %s' % F['cam'][k])
        ds = 'ai_mix_20261011' if F['part'][k] == 0 else 'intervention_val_20261011'
        with np.load(DS / ds / 'aligned' / (str(F['scene'][k]) + '.npz')) as z:
            i = int(F['frame'][k]); tof, imu, veh = z['tof'][i], z['imu'][i], z['vehicle'][i]
        normed = {}
        for cid, e in E.items():
            uc = e['cmd']
            if uc not in normed:
                im, tf, iu, sv = normalize_inputs(img, tof, imu, veh, uc)
                normed[uc] = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None]}
            for gname, g in (('ho_h', int(F['genre'][k])), ('ho_5', mode_day)):
                if gname == 'ho_5' and not F['primary'][k]:
                    continue
                feed = dict(normed[uc]); feed['mode'] = onehot(e['nm'], g)
                a = time.perf_counter()
                res = e['runner'].run_numpy(**feed)
                e['ms'].append((time.perf_counter() - a) * 1000)
                e[gname][k] = post_mod.Stage.model_pred(res)
            if k in keep_ho:
                feeds_keep.setdefault(('ho', k, uc), dict(normed[uc], mode=onehot(e['nm'], int(F['genre'][k]))))
        if (k + 1) % 1000 == 0:
            log('保留 %d / %d(%.0f 秒)' % (k + 1, nho, time.time() - t0))
    # ---- (2) AI 走行(走行と同じ順・ジャンル 5・基準の設定)
    t0 = time.time()
    cache = {}; prev = None
    for k, (s, i, t_us) in enumerate(ai_rows):
        if s not in cache:
            cache.clear()
            with np.load(DS / AI_DS / 'aligned' / (s + '.npz')) as z:
                cache[s] = {n: z[n] for n in ('cam_files', 'tof', 'imu', 'vehicle')}
        a = cache[s]
        img = cv2.imread(str(DS / AI_DS / str(a['cam_files'][i])))
        ai_erpm[k] = float(a['vehicle'][i, 3])
        if s != prev:
            for e in E.values():
                e['stage'].reset()
            prev = s
        normed = {}
        for cid, e in E.items():
            uc = e['cmd']
            if uc not in normed:
                im, tf, iu, sv = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], uc)
                normed[uc] = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None]}
            feed = dict(normed[uc]); feed['mode'] = onehot(e['nm'], mode_day)
            b = time.perf_counter()
            res = e['runner'].run_numpy(**feed)
            e['ms'].append((time.perf_counter() - b) * 1000)
            if not all(np.isfinite(v).all() for v in res.values()):
                e['nonfinite'] += 1
            mp = post_mod.Stage.model_pred(res)
            e['ai_model'][k] = mp
            try:
                pred, _ = e['stage'].apply(res, mp, post_mod.speed_of(ai_erpm[k]), params, int(t_us) * 1000)
                e['ai_pred'][k] = pred
            except post_mod.PostError:
                e['post_err'] += 1
        if (k + 1) % 4000 == 0:
            log('AI %d / %d(%.0f 秒)' % (k + 1, nai, time.time() - t0))
    # 参照に使う AI のフレーム: 低速(< 0.3 m/s)から 24、ほかから 24(乱数は固定)
    v = -ai_erpm * 0.5e-3
    slow_i = np.nonzero(v < 0.3)[0]; fast_i = np.nonzero(v >= 0.3)[0]
    pick_ai = np.concatenate([rng.choice(slow_i, min(24, len(slow_i)), replace=False),
                              rng.choice(fast_i, 24, replace=False)])
    for k in pick_ai:
        s, i, _ = ai_rows[int(k)]
        with np.load(DS / AI_DS / 'aligned' / (s + '.npz')) as z:
            img = cv2.imread(str(DS / AI_DS / str(z['cam_files'][i])))
            tof, imu, veh = z['tof'][i], z['imu'][i], z['vehicle'][i]
        for uc in {e['cmd'] for e in E.values()}:
            im, tf, iu, sv = normalize_inputs(img, tof, imu, veh, uc)
            feeds_keep[('ai', int(k), uc)] = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None],
                                               'mode': None}
    # ---- 速さ(実フレーム 1 個・100 回空回し → 600 回)
    timing = {}
    some = next(iter(feeds_keep.values()))
    for cid, e in E.items():
        feed = {n: some[n] for n in ('image', 'tof', 'imu', 'state')}; feed['mode'] = onehot(e['nm'], mode_day)
        for _ in range(100):
            e['runner'].run_numpy(**feed)
        ts = []
        for _ in range(600):
            b = time.perf_counter(); e['runner'].run_numpy(**feed); ts.append((time.perf_counter() - b) * 1000)
        timing[cid] = {'dedicated_median_ms': round(float(np.median(ts)), 3), 'dedicated_p99_ms': round(float(np.percentile(ts, 99)), 3),
                       'loop_median_ms': round(float(np.median(e['ms'])), 3), 'loop_p99_ms': round(float(np.percentile(e['ms'], 99)), 3),
                       'loop_max_ms': round(float(np.max(e['ms'])), 3), 'loop_calls': len(e['ms'])}
        log(cid, timing[cid])
    # ---- TensorRT の出力(参照のフレーム)
    keys = sorted(feeds_keep.keys(), key=lambda x: (x[0], x[1], x[2]))
    trt_ref = {}
    for cid, e in E.items():
        rows = []
        for key in keys:
            if key[2] != e['cmd']:
                continue
            f = dict(feeds_keep[key])
            g = mode_day if key[0] == 'ai' else int(F['genre'][key[1]])
            f['mode'] = onehot(e['nm'], g)
            rows.append(post_mod.Stage.model_pred(e['runner'].run_numpy(**f)))
        trt_ref[cid] = np.array(rows)
    # エンジンを閉じる前に保存
    OUT.mkdir(parents=True, exist_ok=True)
    for cid, e in E.items():
        np.savez_compressed(OUT / ('gpu_%s.npz' % cid), ho_h=e['ho_h'], ho_5=e['ho_5'], ai_model=e['ai_model'], ai_pred=e['ai_pred'],
                            ai_erpm=ai_erpm, ai_scene=np.array([r[0] for r in ai_rows]), ai_frame=np.array([r[1] for r in ai_rows]),
                            ai_t_us=np.array([r[2] for r in ai_rows], np.int64), trt_ref=trt_ref[cid])
    summ = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'params': params, 'mode_day': mode_day, 'n_ho': nho, 'n_ai': nai,
            'ai_logs': ai_logs, 'ref_keys': [list(k) for k in keys], 'info': info_all, 'timing': timing,
            'nonfinite': {c: E[c]['nonfinite'] for c in E}, 'post_err': {c: E[c]['post_err'] for c in E}}
    json.dump(summ, open(OUT / 'gpu.json', 'w'), ensure_ascii=False, indent=1, default=str)
    for cid in list(E):
        del E[cid]['runner']
    gc.collect(); torch.cuda.empty_cache()
    # ---- PyTorch FP32(CPU)
    from jetformer.policies import runtime
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32 = {}
    for cid, run in cands:
        e = E[cid]
        for ck in ('best', 'last'):
            pth = drive.RUNS_DIR / run / ('%s.pth' % ck)
            if ck == 'last' and (not pth.exists() or not run.startswith('bench_20261021-')):
                continue
            model, cfg, spec, pcfg = runtime.load_run(drive.RUNS_DIR / run, ck, device='cpu')
            outs = []
            with torch.no_grad():
                for key in keys:
                    if key[2] != e['cmd']:
                        continue
                    f = dict(feeds_keep[key])
                    g = mode_day if key[0] == 'ai' else int(F['genre'][key[1]])
                    f['mode'] = onehot(e['nm'], g)
                    o = runtime.predict(model, f)
                    outs.append(o['actions'][0] * 500.0 + 1500.0)
            fp32['%s_%s' % (cid, ck)] = np.array(outs)
            del model; gc.collect()
            d = np.abs(fp32['%s_%s' % (cid, ck)] - trt_ref[cid])
            log(cid, ck, 'TRT − FP32 max %.3f µs p99 %.3f' % (d.max(), np.percentile(d, 99)))
    np.savez_compressed(OUT / 'gpu_fp32_ref.npz', **fp32)
    summ['fp32_vs_trt'] = {k: {'max_us': round(float(np.abs(v - trt_ref[k.split('_')[0]]).max()), 4),
                               'p99_us': round(float(np.percentile(np.abs(v - trt_ref[k.split('_')[0]]), 99)), 4),
                               'mean_us': round(float(np.abs(v - trt_ref[k.split('_')[0]]).mean()), 4),
                               'frames': int(len(v))} for k, v in fp32.items()}
    json.dump(summ, open(OUT / 'gpu.json', 'w'), ensure_ascii=False, indent=1, default=str)
    log('終わり')

if __name__ == '__main__':
    main()
