#!/usr/bin/env python3
"""段 Desk(2026-10-21)の GPU の仕事。1 つずつ別のプロセスで、必ず jf-heavy --lock big を通す。車・装置には触れない

    cd ~/jetformer
    J="/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 /home/jetson/miniconda/envs/jetracer/bin/python -u"
    $J experiments/20261021_bench_prep/desk/desk_gpu.py crosscheck   # B の best を予測し直し、10/12 の保存と同じか(≤ 0.01 µs)
    $J experiments/20261021_bench_prep/desk/desk_gpu.py predict      # SLOW・WGT・AUG の best・last の予測(出来たものは飛ばす)
    $J experiments/20261021_bench_prep/desk/desk_gpu.py sanity       # 9 本のエンジンに 10/10 の AI 走行のフレームを入れる
  (B のエンジンの確認は experiments/20261013_sim_cotrain/train/engine_load_check.py を別のプロセスで。run_desk_gpu.sh)

予測(PyTorch FP32・eval・バッチ 32・workers 3)は 10/15 realacc_eval._predict_one と同じ手順(policy_zoo._Frames・_predict_frames、
ジャンル = aligned の mode)。保存先は /home/jetson/jetformer_reports/bench_prep_20261021/desk/pred/(リポジトリの外)。
sanity は drive._run と同じ順(torch の CUDA の初期化 → TRTRunner → policies.check_io)で、エンジンを 9 本まとめて読み、
フレームを 1 回読むごとに 9 本に入れる(画像を読み直さないため)。出力は走行と同じ Stage.model_pred・Stage.apply(基準の設定)。
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tools'))
sys.path.insert(0, str(HERE))
import desk_lib as D  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
PRED = OUT / 'pred'
LOGS = HERE / 'logs'
RUNS = REPO / 'data' / 'runs'
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
B_SAVED = REPO / 'data' / 'zoo' / 'policy_zoo_20261012' / 'pred' / 'B.npz'
AI_DS = 'ai_mix_20261011'
AI_LOGS = ('20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
           '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
           '20261010-204605')
SANITY_MODE = 5
BASE_PARAMS_OVERRIDE = {'steer_gain': 1.0, 'throttle_gain': 1.0, 'lookahead': 3, 'tof_guard': 1, 'ensemble_j': 0,
                        'speed_cap': 0.0, 'zone_caps': 0, 'tcp_mode': 0, 'speed_scale': 1.0}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def pred_file(cid, ck):
    return PRED / ('%s%s.npz' % (D.RUN_OF[cid], '' if ck == 'best' else '_last'))


def pred_ok(path, ck):
    try:
        with np.load(path) as z:
            return (str(z['key']) == TABLE_KEY and str(z['ckpt']) == ck and z['pred'].shape == (10985, 10, 2)
                    and bool(np.isfinite(z['pred']).all()))
    except Exception:          # noqa: BLE001
        return False


def web_busy():
    import urllib.request
    busy = []
    for path in ('/api/drive/status', '/api/train/status'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080' + path, timeout=3) as r:
                d = json.load(r)
                if d.get('running') or d.get('recording'):
                    busy.append('管理画面 %s が running/recording' % path)
        except Exception:        # noqa: BLE001
            pass
    return busy


def check_ready():
    busy = web_busy()
    if busy:
        raise SystemExit('始めません: %s' % busy)
    import rl_policies as rp
    rp._check_gpu_free(min_gb=2.0)


def table():
    import rl_evaluate as ev
    held = ev._heldout()
    tab = ev.pred_table()
    if tab['key'] != held['pred_table_key'] or tab['key'] != TABLE_KEY:
        raise SystemExit('予測するフレームの表が heldout.json と違います')
    return tab


def predict_one(run_dir, ck, tab, batch=32, workers=3):
    """10/15 realacc_eval._predict_one と同じ"""
    import torch
    import policy_zoo as pz
    import rl_evaluate as ev
    import rl_policies as rp
    from jetformer.policies import runtime
    cfg = rp._read_json(run_dir / 'model_config.json')
    trained = np.array([int(x) for x in cfg['trained_modes']])
    model, _, spec, _ = runtime.load_run(run_dir, ck, device='cuda')
    total, trainable = runtime.count_params(model)
    fs = pz._Frames(tab, np.arange(len(tab['part'])), use_command=bool(cfg.get('command_input', True)))
    genres, sub = ev._genres([it[4] for it in fs.items], None, trained)
    res = pz._predict_frames(model, fs, genres, cfg['num_modes'], batch, workers)
    meta = dict(key=np.array(tab['key']), run=np.array(run_dir.name), ckpt=np.array(ck), policy=np.array(spec.name),
                params_total=np.array(total), params_trainable=np.array(trainable))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res, genres, sub, meta


def cmd_crosscheck(args):
    out_path = LOGS / 'crosscheck_B.json'
    tab = table()
    check_ready()
    t0 = time.time()
    res, genres, _, _ = predict_one(RUNS / D.RUN_OF['B'], 'best', tab)
    pred = res['actions'].astype(np.float64)
    with np.load(B_SAVED) as z:
        ref = z['pred'].astype(np.float64)
        same_key = str(z['key']) == tab['key']
        same_genre = bool(np.array_equal(z['genre'], genres.astype(np.int8)))
    mx = float(np.abs(pred - ref).max())
    out = {'run': D.RUN_OF['B'], 'ckpt': 'best', 'saved': str(B_SAVED), 'frames': int(len(pred)), 'same_key': same_key,
           'same_genre': same_genre, 'max_abs_us': round(mx, 5), 'limit_us': 0.01,
           'pass': bool(same_key and same_genre and mx <= 0.01), 'seconds': round(time.time() - t0, 1),
           'checked_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    write_json(out_path, out)
    log(json.dumps(out, ensure_ascii=False))
    if not out['pass']:
        raise SystemExit('crosscheck が合いません')


def cmd_predict(args):
    import rl_policies as rp
    cc = json.load(open(LOGS / 'crosscheck_B.json')) if (LOGS / 'crosscheck_B.json').exists() else {}
    if not cc.get('pass'):
        raise SystemExit('先に crosscheck を通してください')
    tab = table()
    todo = [(c, ck) for c in (args.ids or D.NEW_TRAINED) for ck in ('best', 'last') if not pred_ok(pred_file(c, ck), ck)]
    if not todo:
        log('予測は作成済み')
        return
    check_ready()
    for cid, ck in todo:
        run_dir = RUNS / D.RUN_OF[cid]
        if not (run_dir / ('%s.pth' % ck)).exists():
            log('%s: %s.pth がありません(飛ばす)' % (cid, ck))
            continue
        t0 = time.time()
        res, genres, sub, meta = predict_one(run_dir, ck, tab)
        pred = res.pop('actions')
        extra = {'out_' + k: v for k, v in res.items()}
        rp._save_npz(pred_file(cid, ck), pred=pred, genre=genres.astype(np.int8), substituted=sub, **extra, **meta)
        ok = pred_ok(pred_file(cid, ck), ck)
        log('%s(%s): %d フレーム %.0f 秒・読み直し %s' % (cid, ck, len(pred), time.time() - t0, 'OK' if ok else 'NG'))
        if not ok:
            raise SystemExit('%s の予測を読み直せません' % pred_file(cid, ck))


# ---------------------------------------------------------------- sanity(TensorRT)

def ai_frames():
    """10/10 の AI 走行 11 本の AI のフレーム(ai_mix_20261011 の source == 1 & vehicle_valid)。時刻の順"""
    from jetformer import config
    root = config.DATASET_DIR / AI_DS
    sp = json.load(open(root / 'splits.json'))
    rows = []
    for split in ('train', 'val'):
        for s in sp[split]:
            if s[:15] not in AI_LOGS:
                continue
            with np.load(root / 'aligned' / (s + '.npz')) as z:
                ok = (z['source'] == 1) & z['vehicle_valid']
                for i in np.nonzero(ok)[0]:
                    rows.append((s, int(i), int(z['timestamps_us'][i])))
    rows.sort(key=lambda r: (r[0][:15], r[2], r[0], r[1]))
    return rows


def cmd_sanity(args):
    import cv2
    import torch
    from jetformer import config, drive, policies
    from jetformer.policies import post as post_mod
    from jetformer.train_data import normalize_inputs
    from jetformer.trt_runtime import TRTRunner
    check_ready()
    ids = args.ids or D.IDS
    rows = ai_frames()
    if args.limit:
        rows = rows[:args.limit]
    log('AI のフレーム %d 個(ログ %d 本)' % (len(rows), len({r[0][:15] for r in rows})))
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    params = dict(drive.DEFAULT_PARAMS)
    params.update(BASE_PARAMS_OVERRIDE)
    lo, hi = float(config.DRIVE['min_us']), float(config.DRIVE['max_us'])
    E = {}
    for cid in ids:
        name = D.RUN_OF[cid]
        st = {'id': cid, 'run': name}
        try:
            info = drive.model_info(name)
            if info is None:
                raise RuntimeError('drive.model_info が None')
            st['drivable'] = bool(info.get('drivable'))
            st['why_not'] = info.get('why_not')
            eng = RUNS / name / drive.ENGINE_FILE
            stt = eng.stat()
            st['engine'] = {'size': stt.st_size, 'mtime_ns': stt.st_mtime_ns}
            runner = TRTRunner(str(eng))
            nm = int(info['num_modes'])
            ok, why = policies.check_io({n: list(runner.shapes[n]) for n in runner.inputs},
                                        {n: list(runner.shapes[n]) for n in runner.outputs}, nm)
            st['check_io_ok'], st['check_io_why'] = bool(ok), why
            if not ok:
                raise RuntimeError('check_io: %s' % why)
            ref = (float(info['steer_center_us']), float(info['throttle_neutral_us']))
            stage = post_mod.Stage(info, ref, fps=config.CAMERA['fps'])
            oh = np.zeros((1, nm), np.float32)
            oh[0, SANITY_MODE] = 1.0
            zeros = {n: np.zeros(runner.shapes[n], np.float32) for n in runner.inputs}
            for _ in range(3):
                runner.run_numpy(**zeros)
            E[cid] = {'runner': runner, 'stage': stage, 'mode': oh, 'cmd': bool(info.get('command_input', False)),
                      'pred': np.full((len(rows), 10, 2), np.nan, np.float32),
                      'model_pred': np.full((len(rows), 10, 2), np.nan, np.float32), 'nonfinite_out': 0,
                      'post_error': 0, 'ms': [], 'post_kind': post_mod.kind_of(info) if stage.post is not None else None,
                      'st': st}
        except Exception as e:      # noqa: BLE001
            st['error'] = '%s: %s' % (type(e).__name__, e)
            log('%s: 読めません %s' % (cid, st['error']))
            E[cid] = {'st': st}
    from jetformer import config as cfgm
    root = cfgm.DATASET_DIR / AI_DS
    cache = {}
    prev_scene = None
    t_start = time.time()
    erpm = np.zeros(len(rows))
    for k, (s, i, t_us) in enumerate(rows):
        if s not in cache:
            cache.clear()
            with np.load(root / 'aligned' / (s + '.npz')) as z:
                cache[s] = {n: z[n] for n in ('cam_files', 'tof', 'imu', 'vehicle')}
        a = cache[s]
        img = cv2.imread(str(root / str(a['cam_files'][i])))
        erpm[k] = float(a['vehicle'][i, 3])
        normed = {}
        if s != prev_scene:
            for e in E.values():
                if 'stage' in e:
                    e['stage'].reset()
            prev_scene = s
        for cid, e in E.items():
            if 'runner' not in e:
                continue
            uc = e['cmd']
            if uc not in normed:
                im, tf, iu, sv = normalize_inputs(img, a['tof'][i], a['imu'][i], a['vehicle'][i], uc)
                normed[uc] = {'image': im[None], 'tof': tf[None], 'imu': iu[None], 'state': sv[None]}
            feed = dict(normed[uc])
            feed['mode'] = e['mode']
            t0 = time.perf_counter()
            res = e['runner'].run_numpy(**feed)
            e['ms'].append((time.perf_counter() - t0) * 1000.0)
            if not all(np.isfinite(v).all() for v in res.values()):
                e['nonfinite_out'] += 1
            mp = e['stage'].model_pred(res)
            e['model_pred'][k] = mp
            try:
                pred, _ = e['stage'].apply(res, mp, post_mod.speed_of(erpm[k]), params, int(t_us) * 1000)
                e['pred'][k] = pred
            except post_mod.PostError:
                e['post_error'] += 1
        if (k + 1) % 2000 == 0:
            log('%d / %d(%.0f 秒)' % (k + 1, len(rows), time.time() - t_start))
    summary = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'frames': len(rows), 'logs': sorted({r[0][:15] for r in rows}),
               'mode_id': SANITY_MODE, 'params': {k: params[k] for k in sorted(BASE_PARAMS_OVERRIDE)}, 'range_us': [lo, hi],
               'seconds': round(time.time() - t_start, 1), 'candidates': {}}
    scene = np.array([r[0] for r in rows])
    frame = np.array([r[1] for r in rows], np.int32)
    t_arr = np.array([r[2] for r in rows], np.int64)
    for cid, e in E.items():
        st = e['st']
        if 'runner' in e:
            p, mp = e['pred'], e['model_pred']
            bad = ~np.isfinite(p).all(axis=(1, 2)) | ~np.isfinite(mp).all(axis=(1, 2))
            oor = int(((p < D.SANITY_LO) | (p > D.SANITY_HI)).sum() + ((mp < D.SANITY_LO) | (mp > D.SANITY_HI)).sum())
            clip = int(((p < lo) | (p > hi)).sum())
            st.update({'frames': len(rows), 'nonfinite_out_frames': e['nonfinite_out'], 'post_error_frames': e['post_error'],
                       'nonfinite_frames': int(bad.sum() + e['nonfinite_out'] + e['post_error']),
                       'out_of_range_values': oor, 'sanity_range_us': [D.SANITY_LO, D.SANITY_HI],
                       'outside_drive_range_values': clip, 'values': int(p.size),
                       'steer_min_us': float(np.nanmin(p[:, :, 0])), 'steer_max_us': float(np.nanmax(p[:, :, 0])),
                       'thr_min_us': float(np.nanmin(p[:, :, 1])), 'thr_max_us': float(np.nanmax(p[:, :, 1])),
                       'post_kind': e['post_kind'],
                       'run_numpy_median_ms': round(float(np.median(e['ms'])), 3),
                       'run_numpy_p99_ms': round(float(np.percentile(e['ms'], 99)), 3)})
            D_path = PRED / ('sanity_%s.npz' % cid)
            PRED.mkdir(parents=True, exist_ok=True)
            tmp = D_path.with_name(D_path.stem + '.tmp.npz')
            np.savez_compressed(tmp, pred=p, model_pred=mp, scene=scene, frame=frame, t_us=t_arr, erpm=erpm,
                                run=np.array(st['run']), mode_id=np.array(SANITY_MODE))
            os.replace(tmp, D_path)
            st['file'] = str(D_path)
        st['sanity_ok'] = D.sanity_of(st)
        summary['candidates'][cid] = st
        log('%s: sanity_ok %s %s' % (cid, st['sanity_ok'], {k: st.get(k) for k in ('nonfinite_frames', 'out_of_range_values',
                                                                                    'run_numpy_median_ms', 'error')}))
    old = json.load(open(LOGS / 'sanity.json')) if (LOGS / 'sanity.json').exists() and args.ids else None
    if old:
        old['candidates'].update(summary['candidates'])
        old['updated_at'] = summary['created_at']
        summary = old
    write_json(LOGS / 'sanity.json', summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('crosscheck', 'predict', 'sanity'))
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--limit', type=int, default=0, help='sanity: 先頭の何フレームだけ(試し)')
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    {'crosscheck': cmd_crosscheck, 'predict': cmd_predict, 'sanity': cmd_sanity}[args.cmd](args)


if __name__ == '__main__':
    main()
