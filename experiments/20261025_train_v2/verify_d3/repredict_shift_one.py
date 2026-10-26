#!/usr/bin/env python3
"""robust_dl1_us の 1 本の数え直し(予測から。verify_d3、2026-10-26)。

desk_v2.py・eval_v2.py・tools/policy_zoo.py を import しない。ずれ 7 つは protocol sets.shift_tests と deviations.md 7 の文から
自分で書いた。使う道具は jetformer の本体(policies.runtime.load_run・forward、train_data.normalize_inputs)だけ。

  1. H1(desk_inputs_20261022.npz の primary、4,670 フレーム)の画像・ToF・IMU・車両を clean_driving_20261011 の aligned から読む
  2. ジャンル = そのフレームの mode(学習していないジャンルなら trained_modes の最初で代わり)。shift.npz の genre と照らす
  3. そのまま + 7 つのずれで、PyTorch FP32 の best を予測(GPU)
  4. 自分の予測と、C3 の shift.npz の pred_<ずれ>・eval/pred の保存した best を照らす(差の最大・平均)
  5. 自分の予測だけで robust_dl1_us を計算(そのまま = 自分の予測、と、そのまま = 保存した best の 2 通り)

書くのは /home/jetson/jetformer_reports/train_v2_20261025/verify_d3_shift_<名前>.json だけ。

    /home/jetson/bin/jf-heavy --mem 3000M --lock big --need-mb 2000 -- nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /home/jetson/miniconda/envs/jetracer/bin/python -I .../verify_d3/repredict_shift_one.py v2_20261025-v2h-e60-s0
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
BUNDLE_P = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
DS = REPO / 'data' / 'datasets' / 'clean_driving_20261011'
RUNS = REPO / 'data' / 'runs'
KINDS = ('img_up3', 'img_down3', 'bright_0.8', 'bright_1.2', 'tof_down1', 'top_0.85', 'top_1.15')
BATCH = 64


def h(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            d.update(c)
    return d.hexdigest()


def shifted(kind, img, tof):
    """自分で書いたずれ。img = BGR uint8 [H,W,3]、tof = 生の ToF [42,54]"""
    H = img.shape[0]
    if kind == 'img_up3':                      # 中身を 3 行上へ。下の 3 行は一番下の行の写し
        out = np.empty_like(img)
        out[:H - 3] = img[3:]
        out[H - 3:] = img[H - 1]
        return out, tof
    if kind == 'img_down3':                    # 中身を 3 行下へ。上の 3 行は一番上の行の写し
        out = np.empty_like(img)
        out[3:] = img[:H - 3]
        out[:3] = img[0]
        return out, tof
    if kind.startswith('bright_'):
        f = float(kind[7:])
        return np.clip(np.rint(img.astype(np.float64) * f), 0, 255).astype(np.uint8), tof
    if kind == 'tof_down1':                    # ToF を 1 行下へ。一番上の行は 0(無効)
        t = np.zeros_like(tof)
        t[1:] = tof[:-1]
        return img, t
    if kind.startswith('top_'):                # deviations.md 7: 行 r に 1 + (f − 1)·w(r)、w = 一番上 1 → 上 1/3 の境で 0 の直線
        f = np.float32(float(kind[4:]))
        r = np.arange(H, dtype=np.float32)
        w = np.clip(np.float32(1.0) - r / np.float32(H / 3.0), 0.0, 1.0).astype(np.float32)
        m = (np.float32(1.0) + (f - np.float32(1.0)) * w).reshape(H, 1, 1)
        return np.clip(np.rint(img.astype(np.float32) * m), 0, 255).astype(np.uint8), tof
    raise ValueError(kind)


def main(run):
    import cv2
    import torch
    from jetformer.policies import runtime
    from jetformer.train_data import normalize_inputs
    t0 = time.time()
    run_dir = RUNS / run
    cfg = json.loads((run_dir / 'model_config.json').read_text(encoding='utf-8'))
    trained = [int(x) for x in cfg['trained_modes']]
    nm = int(cfg['num_modes'])
    use_cmd = bool(cfg.get('command_input', True))
    with np.load(BUNDLE_P, allow_pickle=False) as z:
        key = str(z['key'])
        scene, frame, prim = z['scene'].astype(str), z['frame'].astype(np.int64), z['primary'].astype(bool)
        y = z['y'].astype(np.float64)
    rows = np.nonzero(prim)[0]
    # 入力を読む
    cache, items = {}, []
    for i in rows:
        s = scene[i]
        if s not in cache:
            with np.load(DS / 'aligned' / (s + '.npz'), allow_pickle=False) as a:
                cache[s] = {k: a[k] for k in ('cam_files', 'tof', 'imu', 'vehicle', 'mode')}
        a = cache[s]
        f = int(frame[i])
        m = int(a['mode'][f])
        g = m if m in trained else trained[0]
        items.append((str(DS / str(a['cam_files'][f])), a['tof'][f], a['imu'][f], a['vehicle'][f], g, m))
    genres = np.array([it[4] for it in items], np.int64)
    model = runtime.load_run(run_dir, 'best', device='cuda')[0]
    preds = {k: np.zeros((len(rows), 10, 2), np.float64) for k in ('none',) + KINDS}
    with torch.no_grad():
        for b0 in range(0, len(items), BATCH):
            chunk = items[b0:b0 + BATCH]
            imgs = []
            for it in chunk:
                im = cv2.imread(it[0])
                if im is None:
                    raise SystemExit('画像が読めません: %s' % it[0])
                imgs.append(im)
            onehot = torch.nn.functional.one_hot(torch.from_numpy(genres[b0:b0 + len(chunk)]), nm).float().cuda()
            for kind in ('none',) + KINDS:
                xs = []
                for im, it in zip(imgs, chunk):
                    img2, tof2 = (im, it[1]) if kind == 'none' else shifted(kind, im, it[1])
                    xs.append(normalize_inputs(img2, tof2, it[2], it[3], use_cmd))
                x = {'image': torch.from_numpy(np.stack([v[0] for v in xs])).cuda(),
                     'tof': torch.from_numpy(np.stack([v[1] for v in xs])).cuda(),
                     'imu': torch.from_numpy(np.stack([v[2] for v in xs])).cuda(),
                     'state': torch.from_numpy(np.stack([v[3] for v in xs])).cuda(), 'mode': onehot}
                a = runtime.forward(model, x)['actions'].float().cpu().numpy().astype(np.float64)
                preds[kind][b0:b0 + len(chunk)] = a * 500.0 + 1500.0
    # 照らす
    with np.load(REP / 'eval' / 'pred' / ('%s.npz' % run), allow_pickle=False) as z:
        assert str(z['key']) == key and str(z['ckpt']) == 'best' and str(z['run']) == run
        saved = z['pred'].astype(np.float64)[rows]
    with np.load(REP / 'desk' / 'pred' / ('%s_shift.npz' % run), allow_pickle=False) as z:
        c3 = {k: z['pred_' + k].astype(np.float64) for k in KINDS}
        c3_rows = z['rows'].astype(np.int64)
        c3_genre = z['genre'].astype(np.int64)
        c3_script = str(z['script_sha256'])
        c3_best = str(z['best_sha256'])
    yy = y[rows]
    L = lambda p: float(np.abs(p - yy).mean())   # noqa: E731
    base_mine, base_saved = L(preds['none']), L(saved)
    per_mine = {k: L(preds[k]) - base_mine for k in KINDS}
    per_mine_vs_saved = {k: L(preds[k]) - base_saved for k in KINDS}
    per_c3 = {k: L(c3[k]) - base_saved for k in KINDS}
    out = {
        'schema': 'jf-train-v2-verify-d3-shift/v1', 'made_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': run,
        'script': str(Path(__file__).resolve()), 'script_sha256': h(Path(__file__).resolve()),
        'best_pth_sha256': h(run_dir / 'best.pth'), 'c3_shift_best_sha256_same': c3_best == h(run_dir / 'best.pth'),
        'c3_shift_script_sha256': c3_script,
        'rows': int(len(rows)), 'rows_same_as_c3': bool(c3_rows.shape == rows.shape and (c3_rows == rows).all()),
        'genre_same_as_c3': bool((c3_genre == genres).all()),
        'genre_substituted': int(sum(1 for it in items if it[4] != it[5])),
        'none_vs_saved_best_max_abs_us': float(np.abs(preds['none'] - saved).max()),
        'none_vs_saved_best_mean_abs_us': float(np.abs(preds['none'] - saved).mean()),
        'shift_vs_c3_max_abs_us': {k: float(np.abs(preds[k] - c3[k]).max()) for k in KINDS},
        'shift_vs_c3_mean_abs_us': {k: float(np.abs(preds[k] - c3[k]).mean()) for k in KINDS},
        'robust_dl1_us_mine': float(np.mean(list(per_mine.values()))),
        'robust_dl1_us_mine_base_saved': float(np.mean(list(per_mine_vs_saved.values()))),
        'robust_dl1_us_c3_npz': float(np.mean(list(per_c3.values()))),
        'per_shift_mine_us': per_mine, 'per_shift_c3_us': per_c3,
        'seconds': round(time.time() - t0, 1),
    }
    p = REP / ('verify_d3_shift_%s.json' % run)
    tmp = str(p) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'v2_20261025-v2h-e60-s0'))
