#!/usr/bin/env python3
"""verify_c3(GPU): 1 本の学習を、自分の入力の作り方で予測し直して C3 の欄を数える(2026-10-26)。

desk_v2.py・eval_v2.py・tools/policy_zoo.py・verify_d3 の台本は import しない。ずれ 7 つは protocol sets.shift_tests と
deviations.md 7 の文から自分で書いた。使う道具は jetformer の本体(policies.runtime.load_run・forward、
train_data.normalize_inputs、trt_runtime.TRTRunner、policies.post.Stage)だけ。

  tab:   止まった場面 1,115 + gb_main の DE・SW・HS 1,926(recount_c3_cpu.py の sets_mine.npz)を、ジャンル 5・
         PyTorch FP32 の best で予測 → C3 の tab.npz と照らす。standstill_go_frac と(B-FAST なら)G-b を自分の予測で出す
  shift: H1 4,670 の そのまま + 7 つのずれ(aligned のジャンル)→ robust_dl1_us(そのまま = 保存した best と、自分の予測)
  trt:   10/10 の AI 走行の低速(v < 1.0 m/s)2,326 フレームを TensorRT のエンジン(ジャンル 5)に通す + H1 の低速 222 は
         shift の「そのまま」(FP32)→ brake_lowspeed_frac

書くのは c3/verify/ の下(pred_<名前>.npz・gpu_<名前>.json)だけ。

    /home/jetson/bin/jf-heavy --lock big --mem 3000M --need-mb 2000 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I .../c3/verify/repredict_c3_gpu.py <名前> [tab,shift,trt]
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
sys.path.insert(0, str(REPO))
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
DESK = REP / 'desk'
HERE = REPO / 'experiments' / '20261025_train_v2' / 'c3' / 'verify'
DATA = REPO / 'data'
DS2 = DATA / 'datasets' / 'dataset_v2_20261025'
CLEAN = DATA / 'datasets' / 'clean_driving_20261011'
AIMIX = DATA / 'datasets' / 'ai_mix_20261011'
RUNS = DATA / 'runs'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
KINDS = ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1', 'top_0.85', 'top_1.15')
BATCH = 64
GAIN = 0.90
KV = 0.5e-3


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def my_shift(kind, img, tof):
    """自分で書いたずれ。img = BGR uint8 [H,W,3](224×224)、tof = 生の ToF [42,54]"""
    H = img.shape[0]
    if kind == 'img_up3':          # 画像の中身を 3 行上へ(空いた下 3 行は一番下の行で埋める)
        idx = np.minimum(np.arange(H) + 3, H - 1)
        return img[idx], tof
    if kind == 'img_down3':        # 3 行下へ(空いた上 3 行は一番上の行で埋める)
        idx = np.maximum(np.arange(H) - 3, 0)
        return img[idx], tof
    if kind in ('bright_0.8', 'bright_1.2'):
        f = float(kind.split('_')[1])
        return np.clip(np.rint(img.astype(np.float64) * f), 0, 255).astype(np.uint8), tof
    if kind == 'tof_down1':        # ToF を 1 行下へ(一番上の行は 0)
        idx = np.arange(tof.shape[0]) - 1
        t = tof[np.maximum(idx, 0)].copy()
        t[0] = 0
        return img, t
    if kind in ('top_0.85', 'top_1.15'):   # deviations.md 7: 行 r に 1 + (f − 1)·w(r)、w は一番上 1・上 1/3 の境で 0 の直線
        f = float(kind.split('_')[1])
        r = np.arange(H, dtype=np.float64)
        w = np.maximum(0.0, 1.0 - r / (H / 3.0))
        m = 1.0 + (f - 1.0) * w
        return np.clip(np.rint(img.astype(np.float64) * m[:, None, None]), 0, 255).astype(np.uint8), tof
    raise ValueError(kind)


def car_thr(raw, neutral, gain):
    d = np.asarray(raw, np.float64) - neutral
    return np.clip(np.where(d < 0, neutral + gain * d, neutral + d), 1000.0, 2000.0)


class Model(object):
    def __init__(self, run):
        import torch
        from jetformer.policies import runtime
        from jetformer.train_data import normalize_inputs
        self.torch, self.runtime, self.norm = torch, runtime, normalize_inputs
        self.run_dir = RUNS / run
        self.cfg = json.loads((self.run_dir / 'model_config.json').read_text(encoding='utf-8'))
        self.nm = int(self.cfg['num_modes'])
        self.trained = [int(x) for x in self.cfg['trained_modes']]
        self.use_cmd = bool(self.cfg.get('command_input', True))
        self.model = runtime.load_run(self.run_dir, 'best', device='cuda')[0]

    def predict(self, items, kind=None):
        """items = [(画像のパス, tof, imu, vehicle, ジャンル)] → µs [N,10,2]"""
        import cv2
        torch = self.torch
        out = np.zeros((len(items), 10, 2), np.float64)
        with torch.no_grad():
            for b0 in range(0, len(items), BATCH):
                ch = items[b0:b0 + BATCH]
                xs = []
                for path, tof, imu, veh, g in ch:
                    im = cv2.imread(path)
                    if im is None:
                        raise SystemExit('画像が読めません: %s' % path)
                    if kind is not None:
                        im, tof = my_shift(kind, im, tof)
                    xs.append(self.norm(im, tof, imu, veh, self.use_cmd))
                g = torch.tensor([it[4] for it in ch])
                x = {'image': torch.from_numpy(np.stack([v[0] for v in xs])).cuda(),
                     'tof': torch.from_numpy(np.stack([v[1] for v in xs])).cuda(),
                     'imu': torch.from_numpy(np.stack([v[2] for v in xs])).cuda(),
                     'state': torch.from_numpy(np.stack([v[3] for v in xs])).cuda(),
                     'mode': torch.nn.functional.one_hot(g, self.nm).float().cuda()}
                a = self.runtime.forward(self.model, x)['actions'].float().cpu().numpy().astype(np.float64)
                out[b0:b0 + len(ch)] = a * 500.0 + 1500.0
        return out


def load_frames(root, pairs, genre=None, trained=None):
    """[(シーン, フレーム)] → items。genre が None なら aligned の mode(学習していないなら trained の最初)"""
    cache, items, sub = {}, [], 0
    for s, f in pairs:
        if s not in cache:
            with np.load(root / 'aligned' / (s + '.npz'), allow_pickle=False) as a:
                cache[s] = {k: a[k] for k in ('cam_files', 'tof', 'imu', 'vehicle', 'mode')}
        a = cache[s]
        f = int(f)
        g = genre
        if g is None:
            m = int(a['mode'][f])
            g = m if m in trained else trained[0]
            sub += int(g != m)
        items.append((str(root / str(a['cam_files'][f])), a['tof'][f], a['imu'][f], a['vehicle'][f], g))
    return items, sub


def main(run, parts):
    t0 = time.time()
    proto = json.load(open(REP / 'protocol_v2.json', encoding='utf-8'))
    seal = set(proto['seal']['logs'])
    out_p = HERE / ('gpu_%s.json' % run)
    out = json.load(open(out_p)) if out_p.exists() else {}
    out.update({'schema': 'jf-verify-c3-gpu/v1', 'run': run, 'script': str(Path(__file__).resolve()),
                'script_sha256': sha(Path(__file__).resolve()), 'best_pth_sha256': sha(RUNS / run / 'best.pth')})
    pred_p = HERE / ('pred_%s.npz' % run)
    P = dict(np.load(pred_p, allow_pickle=False)) if pred_p.exists() else {}
    M = Model(run)
    neu = float(M.cfg['throttle_neutral_us'])
    out['ref'] = [float(M.cfg['steer_center_us']), neu]
    log(run, '読みました', parts)

    if 'tab' in parts:
        S = np.load(HERE / 'sets_mine.npz', allow_pickle=False)
        pairs = {k: list(zip(S['%s_scene' % k].astype(str), S['%s_frame' % k])) for k in ('stand', 'gb_de', 'gb_sw', 'gb_hs')}
        for k, v in pairs.items():
            assert not any(s[:15] in seal for s, _ in v)
        allp = pairs['stand'] + pairs['gb_de'] + pairs['gb_sw'] + pairs['gb_hs']
        items, _ = load_frames(DS2, allp, genre=5)
        t1 = time.time()
        pr = M.predict(items)
        n1, n2, n3 = len(pairs['stand']), len(pairs['gb_de']), len(pairs['gb_sw'])
        P['tab_stand'] = pr[:n1]
        P['tab_gb_de'] = pr[n1:n1 + n2]
        P['tab_gb_sw'] = pr[n1 + n2:n1 + n2 + n3]
        P['tab_gb_hs'] = pr[n1 + n2 + n3:]
        # C3 の tab と照らす
        with np.load(DESK / 'sets_v2.npz', allow_pickle=False) as z:
            idx = {(s, int(f)): i for i, (s, f) in enumerate(zip(z['scene'].astype(str), z['frame']))}
        rows = np.array([idx[(s, int(f))] for s, f in allp], np.int64)
        tp = DESK / 'pred' / ('%s_tab.npz' % run)
        tab = {}
        if tp.exists():
            with np.load(tp, allow_pickle=False) as z:
                c3 = z['pred'].astype(np.float64)[rows]
                tab['c3_tab_script_sha256'] = str(z['script_sha256'])
                tab['c3_tab_best_sha256_same'] = str(z['best_sha256']) == out['best_pth_sha256']
            d = np.abs(pr - c3)
            tab['vs_c3_tab_max_abs_us'] = float(d.max())
            tab['vs_c3_tab_mean_abs_us'] = float(d.mean())
        u0 = car_thr(P['tab_stand'][:, 0, 1], neu, GAIN)
        tab['standstill_go_frac'] = float((u0 < neu - 30.0).mean())
        tab['standstill_go_frac_raw_nogain_ref'] = float((P['tab_stand'][:, 0, 1] < neu - 30.0).mean())
        if tp.exists():
            uc = car_thr(c3[:n1, 0, 1], neu, GAIN)
            tab['standstill_go_frac_from_c3_tab'] = float((uc < neu - 30.0).mean())
            tab['standstill_go_flips_vs_c3_tab'] = int(((u0 < neu - 30.0) != (uc < neu - 30.0)).sum())
        tab['frames'] = {'stand': n1, 'gb_de': n2, 'gb_sw': n3, 'gb_hs': len(pairs['gb_hs'])}
        tab['seconds'] = round(time.time() - t1, 1)
        if run.startswith('bfast_'):
            bp = HERE / ('pred_%s.npz' % B_RUN)
            if bp.exists():
                with np.load(bp, allow_pickle=False) as z:
                    PB = {k: z[k] for k in ('tab_gb_de', 'tab_gb_sw', 'tab_gb_hs')}
                nB = 1496.0
                cfgB = json.loads((RUNS / B_RUN / 'model_config.json').read_text(encoding='utf-8'))
                nB = float(cfgB['throttle_neutral_us'])
                per = {}
                allc, allb = [], []
                for k in ('gb_de', 'gb_sw', 'gb_hs'):
                    uc = car_thr(P['tab_' + k][:, 2, 1], neu, GAIN)
                    ub = car_thr(PB['tab_' + k][:, 2, 1], nB, GAIN)
                    per[k] = float(np.mean(-(uc - ub)))
                    allc.append(uc)
                    allb.append(ub)
                tab['bfast_gb_fwd_vs_b_us'] = float(np.mean(-(np.concatenate(allc) - np.concatenate(allb))))
                tab['bfast_gb_fwd_vs_b_us_per'] = per
                tab['bfast_gb_B_from'] = 'verify の B の予測(自分の予測)'
        out['tab'] = tab
        log('tab', json.dumps(tab, ensure_ascii=False))

    if 'shift' in parts or 'trt' in parts:
        with np.load(BUNDLE, allow_pickle=False) as z:
            prim = z['primary'].astype(bool)
            scene, frame = z['scene'].astype(str), z['frame'].astype(np.int64)
            y = z['y'].astype(np.float64)
            erpm_h1 = z['erpm'].astype(np.float64)
            key = str(z['key'])
        rows = np.nonzero(prim)[0]
        yy = y[rows]

    if 'shift' in parts:
        items, sub = load_frames(CLEAN, list(zip(scene[rows], frame[rows])), genre=None, trained=M.trained)
        with np.load(REP / 'eval' / 'pred' / ('%s.npz' % run), allow_pickle=False) as z:
            assert str(z['key']) == key and str(z['ckpt']) == 'best'
            saved = z['pred'].astype(np.float64)[rows]
            saved_genre = z['genre'].astype(int)[rows]
        sh = {'genre_same_as_saved': bool((np.array([it[4] for it in items]) == saved_genre).all()), 'genre_substituted': sub}
        t1 = time.time()
        P['h1_none'] = M.predict(items)
        for k in KINDS:
            P['h1_' + k] = M.predict(items, kind=k)
            log('shift', k)
        L = lambda p: float(np.abs(p - yy).mean())   # noqa: E731
        b_saved, b_mine = L(saved), L(P['h1_none'])
        per_saved = {k: L(P['h1_' + k]) - b_saved for k in KINDS}
        per_mine = {k: L(P['h1_' + k]) - b_mine for k in KINDS}
        sh['none_vs_saved_max_abs_us'] = float(np.abs(P['h1_none'] - saved).max())
        sh['none_vs_saved_mean_abs_us'] = float(np.abs(P['h1_none'] - saved).mean())
        sh['robust_dl1_us_base_saved'] = float(np.mean(list(per_saved.values())))
        sh['robust_dl1_us_base_mine'] = float(np.mean(list(per_mine.values())))
        sh['per_shift_base_saved'] = per_saved
        sh['per_shift_base_mine'] = per_mine
        spth = DESK / 'pred' / ('%s_shift.npz' % run)
        if spth.exists():
            with np.load(spth, allow_pickle=False) as z:
                sh['c3_shift_script_sha256'] = str(z['script_sha256'])
                sh['vs_c3_shift_max_abs_us'] = {k: float(np.abs(P['h1_' + k] - z['pred_' + k].astype(np.float64)).max()) for k in KINDS}
                sh['vs_c3_shift_mean_abs_us'] = {k: float(np.abs(P['h1_' + k] - z['pred_' + k].astype(np.float64)).mean()) for k in KINDS}
                sh['c3_per_shift'] = {k: L(z['pred_' + k].astype(np.float64)) - b_saved for k in KINDS}
                sh['c3_robust_dl1_us'] = float(np.mean(list(sh['c3_per_shift'].values())))
        sh['seconds'] = round(time.time() - t1, 1)
        out['shift'] = sh
        log('shift', json.dumps({k: v for k, v in sh.items() if not isinstance(v, dict)}, ensure_ascii=False))

    if 'trt' in parts:
        from jetformer.trt_runtime import TRTRunner
        from jetformer.policies import post as post_mod
        sp = json.load(open(AIMIX / 'splits.json'))
        logs1010 = set(proto['tables']['dataset']['ai_logs']['10/10'])
        fr = []
        for split in ('train', 'val'):
            for s in sp[split]:
                if s[:15] not in logs1010:
                    continue
                assert s[:15] not in seal
                with np.load(AIMIX / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
                    ok = (z['source'] == 1) & z['vehicle_valid'].astype(bool)
                    for i in np.nonzero(ok)[0]:
                        fr.append((s, int(i), int(z['timestamps_us'][i]), float(z['vehicle'][i, 3])))
        fr.sort(key=lambda r: (r[0][:15], r[2], r[0], r[1]))
        e_ai = np.array([r[3] for r in fr])
        low = np.nonzero(-e_ai * KV < 1.0)[0]
        items, _ = load_frames(AIMIX, [(fr[i][0], fr[i][1]) for i in low], genre=5)
        eng = RUNS / run / 'jetformer_fp16.engine'
        runner = TRTRunner(str(eng))
        stage = post_mod.Stage(M.cfg, (float(M.cfg['steer_center_us']), neu))
        assert stage.post is None, '後処理のある方策'
        import cv2
        oh = np.zeros((1, M.nm), np.float32)
        oh[0, 5] = 1.0
        t1 = time.time()
        tr = np.zeros((len(items), 10, 2), np.float64)
        for j, (path, tof, imu, veh, g) in enumerate(items):
            im = cv2.imread(path)
            x = M.norm(im, tof, imu, veh, M.use_cmd)
            feed = {'image': x[0][None], 'tof': x[1][None], 'imu': x[2][None], 'state': x[3][None], 'mode': oh}
            res = runner.run_numpy(**{n: feed[n] for n in runner.inputs})
            mp = stage.model_pred(res)
            pred, _ = stage.apply(res, mp, max(-float(veh[3]) * KV, 0.0), {'ensemble_j': 0}, 0)
            tr[j] = pred
        P['trt_low1010'] = tr
        k_ai = int((tr[:, 0, 1] > 1598.0).sum())
        bt = {'ai_1010': [k_ai, int(len(low))], 'engine_sha256': sha(eng), 'seconds': round(time.time() - t1, 1)}
        job = ((json.load(open(DESK / 'desk_v2.json', encoding='utf-8'))['runs'].get(run) or {}).get('trt') or {}).get('trt_job')
        if job and (DESK / 'trt' / job / ('%s.npz' % run)).exists():
            with np.load(DESK / 'trt' / job / ('%s.npz' % run), allow_pickle=False) as z:
                c3p = z['pred'].astype(np.float64)[z['part'] == 0][low]
            d = np.abs(tr - c3p)
            bt['vs_c3_trt_max_abs_us'] = float(d.max())
            bt['vs_c3_trt_mean_abs_us'] = float(d.mean())
            bt['c3_trt_job'] = job
            bt['brake_flips_vs_c3_trt'] = int(((tr[:, 0, 1] > 1598.0) != (c3p[:, 0, 1] > 1598.0)).sum())
        if 'h1_none' in P:
            v = -erpm_h1[rows] * KV
            lowh = v < 1.0
            k_h = int((P['h1_none'][lowh, 0, 1] > 1598.0).sum())
            bt['heldout_fp32_mine'] = [k_h, int(lowh.sum())]
            bt['brake_lowspeed_frac'] = (k_h + k_ai) / float(lowh.sum() + len(low))
            fp32_ai = M.predict(items)
            bt['ai_1010_fp32_ref'] = [int((fp32_ai[:, 0, 1] > 1598.0).sum()), int(len(low))]
            bt['trt_vs_fp32_max_abs_us'] = float(np.abs(fp32_ai - tr).max())
        out['trt'] = bt
        log('trt', json.dumps(bt, ensure_ascii=False))

    out['made_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out.setdefault('seconds', {})
    out['seconds'][','.join(parts)] = round(time.time() - t0, 1)
    np.savez_compressed(str(pred_p) + '.tmp.npz', **P)
    os.replace(str(pred_p) + '.tmp.npz', pred_p)
    tmp = str(out_p) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out_p)
    log('書きました', out_p, round(time.time() - t0, 1), '秒')
    return 0


if __name__ == '__main__':
    r = sys.argv[1]
    parts = sys.argv[2].split(',') if len(sys.argv) > 2 else ['tab', 'shift', 'trt']
    sys.exit(main(r, parts))
