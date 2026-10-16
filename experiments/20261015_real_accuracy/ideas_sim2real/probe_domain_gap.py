#!/usr/bin/env python3
"""案出しの小さな確かめ 2(CPU だけ・学習なし・判定には使わない)。

シミュレーターの画像を「実機の見た目」に寄せる変換(見た目合わせ T)で、モデルにとっての違いがどれだけ縮むかを見る。

  1. 画像の量(明るさ・コントラスト・彩度・輪郭・上下の明るさ・ぼけ)を 実機 / シミュレーター / シミュレーター+T で比べる
  2. 特徴の空間での違い: 画像の背骨(ResNet-18 の最後の 512 次元を平均)で、シミュレーターと実機を
     線形の分類器が見分けられる率(半分で学び、残り半分で当てる)。背骨は 3 つ:
     ImageNet のまま / A 種 0(実機だけ)/ C-DIV 種 0(混ぜて学習)
  3. 働きでの違い: 実機だけで学んだ A 種 0 に、シミュレーターのフレームを見せたときのステアの誤差(頭 4・5)。
     T で画像を実機に寄せると、シミュレーターの先生の舵に近づくか(同じコースなので、舵の多くは道の形で決まる)
  4. 入力への頼り方: 実機の検証のフレームで、画像・ToF・IMU を 1 つずつ消したときの誤差の増え方(A 種 0・C-DIV 種 0)

T(見た目合わせ)は、別に選んだフレーム(合わせ用)から作る固定の変換:
  ぼけ(ガウス σ を輪郭の強さが実機に合うように選ぶ)→ 画像を上・中・下の 3 つの帯に分け、帯ごと・色ごとにヒストグラムを実機に合わせる(LUT)

出力: probe_domain_gap.json(このファイルの隣)。
使い方: cd ~/jetformer && CUDA_VISIBLE_DEVICES= jf-heavy --mem 1500M --lock none -- nice -n 10 python <このファイル>
"""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

JF = '/home/jetson/jetformer'
sys.path.insert(0, JF)
from jetformer.model import PWM_SCALE  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.train_data import normalize_inputs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DS = JF + '/data/datasets'
RUNS = JF + '/data/runs'
REAL = DS + '/clean_driving_20261011'
SIMS = ['sim-rl-fastest_20261014-1125', 'sim-rl-speed_20261014-1125', 'sim-rl-no_wall_20261015-0302',
        'sim-rl-recover_20261015-0302', 'sim-rl-line_20261015-0302']
RUN_A = RUNS + '/clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30'
RUN_CDIV = RUNS + '/simdiv_20261015-CDIV-s0'
N_FIT, N_EVAL = 300, 300
BANDS = 3
torch.set_num_threads(3)


def r(x, nd=3):
    return round(float(x), nd)


# ---------------------------------------------------------------- フレームの選び方(メモリを使わない読み方)
class Frames(object):
    """データセットの分けから、学習と同じ条件(future_valid & vehicle_valid & ジャンルあり & 2.5G の衝突の前後でない & 間引き 3)の
    フレームを n 個選び、そのフレームの値だけを持つ(JetFormerDataset は全シーンの配列を持つので 1.5G を越えた)"""

    def __init__(self, root, split, n, seed, stride=3):
        import json as _j
        from jetformer.train_data import impact_mask
        with open(os.path.join(root, 'splits.json')) as f:
            scenes = _j.load(f)[split]
        cand = []
        for s in scenes:
            with np.load(os.path.join(root, 'aligned', s + '.npz')) as z:
                if 'vehicle' not in z.files or 'tof' not in z.files:
                    continue
                ok = z['future_valid'] & z['vehicle_valid']
                if 'mode' in z.files:
                    ok &= z['mode'] != 0
                ok &= ~impact_mask(z['timestamps_us'], z['imu'], 2.5)
                ok &= (np.arange(len(ok)) % stride) == 0
            cand += [(s, int(i)) for i in np.nonzero(ok)[0]]
        rng = random.Random(seed)
        pick = sorted(rng.sample(cand, min(n, len(cand))))
        self.root = root
        self.rows = []
        by = {}
        for s, i in pick:
            by.setdefault(s, []).append(i)
        for s, ii in by.items():
            with np.load(os.path.join(root, 'aligned', s + '.npz')) as z:
                mode = z['mode'] if 'mode' in z.files else np.zeros(len(z['vehicle']), np.int16)
                for i in ii:
                    self.rows.append({'cam': str(z['cam_files'][i]), 'tof': z['tof'][i].copy(), 'imu': z['imu'][i].copy(),
                                      'vehicle': z['vehicle'][i].copy(), 'y': z['future_actions'][i].copy(),
                                      'mode': int(mode[i])})
        rng.shuffle(self.rows)

    def image(self, j):
        return cv2.imread(os.path.join(self.root, self.rows[j]['cam']))      # BGR 224x224


# ---------------------------------------------------------------- 画像の量
def stats(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    sob = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    h = g.shape[0]
    return [g.mean(), g.std(), hsv[..., 1].mean(), sob.mean(), lap.var(), g[:h // 3].mean(), g[h * 2 // 3:].mean()]


STAT_KEYS = ['brightness', 'contrast', 'saturation', 'edge_sobel', 'laplacian_var', 'top_brightness', 'bottom_brightness']


def med_stats(imgs):
    return {k: r(v, 1) for k, v in zip(STAT_KEYS, np.median(np.array([stats(x) for x in imgs]), axis=0))}


# ---------------------------------------------------------------- 見た目合わせ T
def band_slices(h):
    e = np.linspace(0, h, BANDS + 1).astype(int)
    return [slice(e[b], e[b + 1]) for b in range(BANDS)]


def hist_cdf(imgs, band, ch):
    h = np.zeros(256)
    for x in imgs:
        h += np.bincount(x[band, :, ch].ravel(), minlength=256)
    c = np.cumsum(h)
    return c / c[-1]


def fit_T(sim_imgs, real_imgs, sigma):
    sim_b = [blur(x, sigma) for x in sim_imgs]
    luts = []
    for band in band_slices(sim_imgs[0].shape[0]):
        row = []
        for ch in range(3):
            cs, cr = hist_cdf(sim_b, band, ch), hist_cdf(real_imgs, band, ch)
            row.append(np.clip(np.searchsorted(cr, cs, side='left'), 0, 255).astype(np.uint8))
        luts.append(row)
    return luts


def blur(x, sigma):
    return x if sigma <= 0 else cv2.GaussianBlur(x, (0, 0), sigma)


def apply_T(x, luts, sigma):
    y = blur(x, sigma).copy()
    for band, row in zip(band_slices(x.shape[0]), luts):
        for ch in range(3):
            y[band, :, ch] = row[ch][y[band, :, ch]]
    return y


def desaturate(bgr, k):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * k, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def choose_sigma(sim_imgs, real_imgs):
    target = np.median([stats(x)[3] for x in real_imgs])
    best = None
    for s in (0.0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5):
        e = np.median([stats(blur(x, s))[3] for x in sim_imgs[:120]])
        if best is None or abs(e - target) < best[1]:
            best = (s, abs(e - target), e)
    return best[0]


# ---------------------------------------------------------------- 特徴と分類
def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


@torch.no_grad()
def feats(backbone, imgs, bs=32):
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([to_tensor(v) for v in imgs[i:i + bs]])
        out.append(backbone(x).mean(dim=(2, 3)))
    return torch.cat(out).numpy().astype(np.float64)


def domain_acc(fa, fb, seed=0):
    """線形の分類器(ロジスティック回帰・L2)で a と b を見分ける率。半分で学び、残りで当てる。2 回入れ替えた平均"""
    X = np.concatenate([fa, fb])
    y = np.concatenate([np.zeros(len(fa)), np.ones(len(fb))])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    half = len(X) // 2
    accs = []
    for tr, te in ((perm[:half], perm[half:]), (perm[half:], perm[:half])):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        Xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 1e-2 * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            accs.append(float((((Xe @ w + b) > 0).float().numpy() == y[te]).mean()))
    return float(np.mean(accs))


def mean_gap(fa, fb):
    """平均の差の大きさ ÷ 実機(fb)の中のばらつき(次元ごとに標準化してから)。0 = 同じ平均"""
    sd = fb.std(0) + 1e-6
    return float(np.sqrt((((fa.mean(0) - fb.mean(0)) / sd) ** 2).mean()))


# ---------------------------------------------------------------- モデルの働き
@torch.no_grad()
def predict(model, rows, imgs, num_modes, genre, drop=None, bs=32):
    """rows(Frames.rows)と imgs(BGR、同じ順)で予測 [N,10,2](正規化の値)。drop: 'image' / 'tof' / 'imu' を消す"""
    preds, ys = [], []
    for s in range(0, len(rows), bs):
        I, T, U, S, M, Y = [], [], [], [], [], []
        for a, img in zip(rows[s:s + bs], imgs[s:s + bs]):
            im, tof, imu, st = normalize_inputs(img, a['tof'], a['imu'], a['vehicle'], use_command=False)
            m = np.zeros(num_modes, np.float32)
            m[a['mode'] if genre is None else genre] = 1.0
            I.append(im), T.append(tof), U.append(imu), S.append(st), M.append(m)
            Y.append((a['y'] - 1500.0) / PWM_SCALE)
        I, T, U, S, M = (torch.from_numpy(np.stack(v)) for v in (I, T, U, S, M))
        if drop == 'image':
            I = torch.zeros_like(I)                 # 正規化の後の 0 = ImageNet の平均の色の一様な画像
        elif drop == 'tof':
            T = torch.zeros_like(T)                 # 距離 0 = 無効
        elif drop == 'imu':
            U = torch.zeros_like(U)
        preds.append(model(I, T, U, S, M).numpy())
        ys.append(np.stack(Y))
    return np.concatenate(preds).astype(np.float64), np.concatenate(ys).astype(np.float64)


def l1_us(p, y):
    e = np.abs(p - y) * PWM_SCALE
    return {'all': r(e.mean(), 2), 'steer': r(e[:, :, 0].mean(), 2), 'throttle': r(e[:, :, 1].mean(), 2)}


def main():
    res = {'note': '案出しの参考の数字(CPU・学習なし・判定には使わない)'}
    # 実機: 合わせ用 = 学習の分け、評価用 = 検証の分け。シミュレーター: 5 モードの学習の分けから合わせ用と評価用を別に選ぶ
    real_tr = Frames(REAL, 'train', N_FIT, 1)
    real_va = Frames(REAL, 'val', N_EVAL, 2)
    per = (N_FIT + N_EVAL) // len(SIMS)
    sim_fit_rows, sim_eval_rows, sim_fit_imgs, sim_eval_imgs = [], [], [], []
    for k, name in enumerate(SIMS):
        fr = Frames(DS + '/' + name, 'train', per, 10 + k)
        h = len(fr.rows) // 2
        for j in range(len(fr.rows)):
            (sim_fit_rows if j < h else sim_eval_rows).append(fr.rows[j])
            (sim_fit_imgs if j < h else sim_eval_imgs).append(fr.image(j))
    real_fit_imgs = [real_tr.image(j) for j in range(len(real_tr.rows))]
    real_eval_imgs = [real_va.image(j) for j in range(len(real_va.rows))]
    sigma = choose_sigma(sim_fit_imgs, real_fit_imgs)
    luts = fit_T(sim_fit_imgs, real_fit_imgs, sigma)
    sim_T_imgs = [apply_T(x, luts, sigma) for x in sim_eval_imgs]
    # T2 = T のあと彩度を実機の中央値に合わせる(合わせ用のフレームで比を決める)
    sat_real = np.median([stats(x)[2] for x in real_fit_imgs])
    sat_T = np.median([stats(apply_T(x, luts, sigma))[2] for x in sim_fit_imgs[:150]])
    sat_k = float(sat_real / max(sat_T, 1e-6))
    sim_T2_imgs = [desaturate(x, sat_k) for x in sim_T_imgs]
    res['T'] = {'blur_sigma': sigma, 'bands': BANDS, 'T2_saturation_scale': r(sat_k), 'fit_frames': {'real': len(real_fit_imgs), 'sim': len(sim_fit_imgs)}}
    res['image_stats_median'] = {'real_val': med_stats(real_eval_imgs), 'sim': med_stats(sim_eval_imgs),
                                 'sim_T': med_stats(sim_T_imgs), 'sim_T2': med_stats(sim_T2_imgs)}
    np.savez_compressed(os.path.join(HERE, 'T_luts.npz'), luts=np.array(luts), sigma=sigma)
    # 見本の画像(実機 / シミュレーター / シミュレーター+T を 4 組)
    rows = [np.concatenate([real_eval_imgs[j], sim_eval_imgs[j], sim_T_imgs[j], sim_T2_imgs[j]], axis=1) for j in range(4)]
    cv2.imwrite(os.path.join(HERE, 'T_examples.jpg'), np.concatenate(rows, axis=0))
    print('T: sigma', sigma, json.dumps(res['image_stats_median'], ensure_ascii=False), flush=True)

    # 特徴の空間
    import torchvision
    inet = torchvision.models.resnet18()
    inet.load_state_dict(torch.load(os.path.expanduser('~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'),
                                    map_location='cpu', weights_only=True))
    inet_bb = torch.nn.Sequential(*list(inet.children())[:-2]).eval()
    mA, cfgA, _, _ = runtime.load_run(RUN_A, 'best')
    mC, cfgC, _, _ = runtime.load_run(RUN_CDIV, 'best')
    gap = {}
    for name, bb in (('imagenet', inet_bb), ('A_s0', mA.image_backbone), ('CDIV_s0', mC.image_backbone)):
        fr = feats(bb, real_eval_imgs)
        fs = feats(bb, sim_eval_imgs)
        ft = feats(bb, sim_T_imgs)
        ft2 = feats(bb, sim_T2_imgs)
        fr2 = feats(bb, real_fit_imgs)              # 実機どうし(学習の分け 対 検証の分け)= 下限の目安
        gap[name] = {'acc_sim_vs_real': r(domain_acc(fs, fr)), 'acc_simT_vs_real': r(domain_acc(ft, fr)), 'acc_simT2_vs_real': r(domain_acc(ft2, fr)),
                     'acc_realtrain_vs_realval': r(domain_acc(fr2, fr)),
                     'gap_sim': r(mean_gap(fs, fr)), 'gap_simT': r(mean_gap(ft, fr)), 'gap_simT2': r(mean_gap(ft2, fr)), 'gap_realtrain': r(mean_gap(fr2, fr))}
        print(name, gap[name], flush=True)
    res['feature_gap'] = gap

    # 働き: A 種 0 にシミュレーターのフレーム(元 / T)を見せたときの誤差(頭 4・5)
    func = {}
    for g in (4, 5):
        for tag, imgs in (('sim', sim_eval_imgs), ('sim_T', sim_T_imgs), ('sim_T2', sim_T2_imgs)):
            p, y = predict(mA, sim_eval_rows, imgs, cfgA['num_modes'], g)
            func['A_s0_head%d_%s' % (g, tag)] = l1_us(p, y)
        print('func', g, func, flush=True)
    # 参考: C-DIV 種 0 の頭 8(5 モードの学習の分けなので、学習に入ったフレーム)
    for tag, imgs in (('sim', sim_eval_imgs), ('sim_T', sim_T_imgs)):
        p, y = predict(mC, sim_eval_rows, imgs, cfgC['num_modes'], 8)
        func['CDIV_s0_head8_%s' % tag] = l1_us(p, y)
        if tag == 'sim':
            func['sim_label_spread_steer_us'] = r(np.abs(y[:, :, 0] - np.median(y[:, :, 0])).mean() * PWM_SCALE, 2)
    res['sim_frames_L1_us'] = func

    # 入力への頼り方(実機の検証のフレーム、ジャンルは aligned の値)
    rel = {}
    for name, m, cfg in (('A_s0', mA, cfgA), ('CDIV_s0', mC, cfgC)):
        rel[name] = {}
        for drop in (None, 'image', 'tof', 'imu'):
            p, y = predict(m, real_va.rows, real_eval_imgs, cfg['num_modes'], None, drop)
            rel[name][drop or 'none'] = l1_us(p, y)
        # 実機の画像を「シミュレーターの見た目」へ寄せた逆向きの T は作らない(ここでは実機の入力はそのまま)
        print('reliance', name, rel[name], flush=True)
    res['input_reliance_real_val_L1_us'] = rel
    res['frames'] = {'real_eval': len(real_eval_imgs), 'sim_eval': len(sim_eval_imgs)}
    with open(os.path.join(HERE, 'probe_domain_gap.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
