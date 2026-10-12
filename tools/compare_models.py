#!/usr/bin/env python3
"""学習結果どうしの精度比較(同じ検証シーンで)。レポートを data/reports/compare_<dataset>/ に作る

    python tools/compare_models.py --dataset clean_driving_20261011 --runs <旧モデル> <新モデル>

どのモデルも学習していないシーン(データセットの検証用)の各フレームで、1〜10フレーム先の [ステアリング, スロットル] を予測し、
人の操作との平均絶対誤差(MAE)を比べる。
  ・教師あり: 今の操作値に人の値を入れる(学習と同じ条件)
  ・自己入力: 今の操作値に前のフレームの予測を入れる(AI走行でCH3をAI側にしたときと同じ条件)
集計: 全フレーム / 衝突の前後を除いたフレーム(学習で外したのと同じ、水平2.5G以上の1秒前〜2秒後)/ ジャンル別 / シーンの出どころ別。
ほかに、ハンドルの向きの正解率(人が中心から100µs以上切っているフレームで、3フレーム先の予測が同じ向きか)と、
前進の強さの偏り(予測 − 人。負なら人より強く前進 = 速く走りがち)。
モデルが学習していないジャンルのシーンは、そのモデルが学習したジャンルで代用して評価する(表に印をつける)。

どのモデルの種類(方策、jetformer/policies)も同じ手順で評価する: runtime.load_run(model_config.json の policy。無ければ
jetformer_v0)→ runtime.predict(FP32)→ policies.post.Stage(走行と同じ後処理・重ね合わせ。シーンごとに時刻の順)。
後処理の設定は走行の既定(tcp_mode 0・速度の上限なし・重ね合わせなし = actions そのまま)。--post-params で変えられる
(例 '{"tcp_mode": 1, "speed_cap": 2.75}'。比較の判定には使わない参考の行)。
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import config  # noqa: E402
from jetformer.policies import post as post_mod  # noqa: E402
from jetformer.train_data import impact_mask, normalize_inputs  # noqa: E402

H = 10


def load_model(run, device='cuda'):
    """学習結果 → (モデル, model_config, 学習したジャンル, result)。どの方策も runtime.load_run で作る"""
    from jetformer.policies import runtime
    d = config.DATA_DIR / 'runs' / run
    m, cfg, _, _ = runtime.load_run(d, 'best', device=device)
    res = json.load(open(d / 'result.json', encoding='utf-8')) if (d / 'result.json').exists() else {}
    trained = cfg.get('trained_modes') or [int(k) for k in (res.get('train_mode_counts') or {})] or [5]
    return m, cfg, [int(x) for x in trained], res


def drive_params(post_params=None):
    """後処理に渡す走行のパラメータ(走行の既定 + 指定)"""
    from jetformer.drive import DEFAULT_PARAMS
    return dict(DEFAULT_PARAMS, **(post_params or {}))


def predict_scene(model, cfg, trained, root, a, post_params=None, predict=None):
    """シーンの全フレームを時刻の順に予測する → (教師あり [N,10,2] µs, 自己入力 [N,10,2] µs, ジャンルの代用 [N])。
    推論の後は走行と同じ post.Stage(後処理・重ね合わせ)を通す(教師あり・自己入力で別の Stage)。
    predict(model, inputs) → {名前: [1,...]}(既定 runtime.predict。テストで偽物を渡せる)"""
    if predict is None:
        from jetformer.policies import runtime
        predict = runtime.predict
    n = len(a['timestamps_us'])
    use_cmd = cfg.get('command_input', True)
    ref = (float(cfg.get('steer_center_us') or 1443.0), float(cfg.get('throttle_neutral_us') or 1498.0))
    params = drive_params(post_params)
    stages = {'tf': post_mod.Stage(cfg, ref), 'sf': post_mod.Stage(cfg, ref)}
    tf, sf = np.full((n, H, 2), np.nan, np.float32), np.full((n, H, 2), np.nan, np.float32)
    substituted = np.zeros(n, bool)
    prev = None
    for i in range(n):
        img = cv2.imread(str(root / str(a['cam_files'][i])))
        mode_id = int(a['mode'][i])
        if mode_id not in trained:
            mode_id, substituted[i] = trained[0], True
        mode = np.zeros((1, cfg['num_modes']), np.float32)
        mode[0, mode_id] = 1
        t_ns = int(a['timestamps_us'][i]) * 1000
        for kind in ('tf', 'sf'):
            veh = a['vehicle'][i].copy()
            if kind == 'sf' and prev is not None:
                veh[0], veh[1] = prev
            im, tof, imu, st = normalize_inputs(img, a['tof'][i], a['imu'][i], veh, use_cmd)
            res = predict(model, {'image': im[None], 'tof': tof[None], 'imu': imu[None], 'state': st[None], 'mode': mode})
            stage = stages[kind]
            mp = stage.model_pred(res)
            p, _ = stage.apply(res, mp, post_mod.speed_of(a['vehicle'][i][3]), params, t_ns)
            (tf if kind == 'tf' else sf)[i] = p
            if kind == 'sf':
                prev = (float(p[0, 0]), float(p[0, 1]))
    return tf, sf, substituted


def metrics(pred, y, cur, center, neutral):
    """pred/y: [N,10,2]。MAE(各フレーム先)、向きの正解率、前進の偏り"""
    if len(y) == 0:
        return None
    mae = np.abs(pred - y).mean(axis=0)
    dev = y[:, 2, 0] - center
    big = np.abs(dev) > 100
    sign_acc = float(np.mean(np.sign(pred[big, 2, 0] - center) == np.sign(dev[big]))) if big.any() else None
    fwd_bias = float(np.mean(pred[:, 2, 1] - y[:, 2, 1]))      # 負 = 人より強く前進
    return {'n': int(len(y)), 'steer_mae': mae[:, 0].round(1).tolist(), 'throttle_mae': mae[:, 1].round(1).tolist(),
            'steer_mae_k3': round(float(mae[2, 0]), 1), 'throttle_mae_k3': round(float(mae[2, 1]), 1),
            'steer_sign_acc': None if sign_acc is None else round(sign_acc, 3), 'fwd_bias_us': round(fwd_bias, 1),
            'hold_current_steer_k3': round(float(np.abs(cur[:, 0] - y[:, 2, 0]).mean()), 1),
            'hold_current_throttle_k3': round(float(np.abs(cur[:, 1] - y[:, 2, 1]).mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--impact-g', type=float, default=2.5)
    ap.add_argument('--stride', type=int, default=1, help='N フレームに1つ評価(速くしたいとき)')
    ap.add_argument('--post-params', default=None,
                    help='後処理に渡す走行のパラメータの JSON(既定: 走行の既定 = tcp_mode 0・上限なし・重ね合わせなし)')
    args = ap.parse_args()
    post_params = json.loads(args.post_params) if args.post_params else None
    root = config.DATASET_DIR / args.dataset
    splits = json.load(open(root / 'splits.json'))
    out_dir = config.DATA_DIR / 'reports' / ('compare_' + args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = {s: dict(np.load(root / 'aligned' / (s + '.npz'))) for s in splits['val']}
    report = {'dataset': args.dataset, 'val_scenes': splits['val'], 'impact_g': args.impact_g, 'models': {}}
    for run in args.runs:
        model, cfg, trained, res = load_model(run)
        center = float(cfg.get('steer_center_us') or 1443.0)
        neutral = float(cfg.get('throttle_neutral_us') or 1498.0)
        rows = {'tf': [], 'sf': []}
        meta = []
        for s, a in scenes.items():
            ok = a['vehicle_valid'] & a['future_valid']
            if args.stride > 1:
                ok &= (np.arange(len(ok)) % args.stride) == 0
            extra = {'post_params': post_params} if post_params else {}         # 既定は今までと同じ呼び方
            tf, sf, sub = predict_scene(model, cfg, trained, root, a, **extra)
            imp = impact_mask(a['timestamps_us'], a['imu'], args.impact_g)
            for kind, p in (('tf', tf), ('sf', sf)):
                rows[kind].append((p[ok], a['future_actions'][ok], a['vehicle'][ok][:, :2]))
            meta.append({'scene': s, 'mode': a['mode'][ok].astype(int), 'impact': imp[ok], 'sub': sub[ok],
                         'source': 'old_val' if s.startswith(('20261010-13', '20261010-14', '20261010-15')) else 'new_val'})
            print('%s %s: %d フレーム' % (run[:40], s, int(ok.sum())), flush=True)
        r = {'policy': cfg.get('policy', 'jetformer_v0'), 'post_params': post_params,
             'trained_modes': trained, 'command_input': cfg.get('command_input', True),
             'mask_impacts': cfg.get('mask_impacts', False), 'train_dataset': cfg.get('dataset'),
             'best_val_us': res.get('best_val_us'), 'train_samples': res.get('train_samples')}
        modes_all = np.concatenate([m['mode'] for m in meta])
        imp_all = np.concatenate([m['impact'] for m in meta])
        sub_all = np.concatenate([m['sub'] for m in meta])
        src_all = np.concatenate([np.full(len(m['mode']), m['source']) for m in meta])
        for kind in ('tf', 'sf'):
            P = np.concatenate([x[0] for x in rows[kind]])
            Y = np.concatenate([x[1] for x in rows[kind]])
            CUR = np.concatenate([x[2] for x in rows[kind]])
            sel = {'all': np.ones(len(Y), bool), 'clean': ~imp_all}
            for g in sorted(set(modes_all.tolist())):
                sel['clean_genre%d' % g] = ~imp_all & (modes_all == g)
            for src in ('old_val', 'new_val'):
                sel['clean_' + src] = ~imp_all & (src_all == src)
            r[kind] = {name: metrics(P[m], Y[m], CUR[m], center, neutral) for name, m in sel.items()}
            r[kind]['substituted_genre_frames'] = int(sub_all.sum())
        report['models'][run] = r
        del model
        import torch
        torch.cuda.empty_cache()
    with open(out_dir / 'compare.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    # 図: フレーム先ごとの MAE(衝突を除く、教師あり/自己入力)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    k = np.arange(1, H + 1)
    colors = ['#6b7280', '#7c3aed', '#16a34a', '#dc2626']
    for i, run in enumerate(args.runs):
        r = report['models'][run]
        lab = run.split('-s')[0]
        for ax, key, title in ((axes[0], 'steer_mae', 'ステアリング'), (axes[1], 'throttle_mae', 'スロットル')):
            ax.plot(k, r['tf']['clean'][key], color=colors[i % 4], marker='o', label=lab + ' 教師あり')
            ax.plot(k, r['sf']['clean'][key], color=colors[i % 4], marker='x', ls='--', label=lab + ' 自己入力(AI走行と同じ)')
            ax.set_title('%s の誤差(衝突の前後を除く検証フレーム)' % title)
            ax.set_xlabel('何フレーム先か(1 = 33ms)')
            ax.set_ylabel('平均絶対誤差 [µs]')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_mae.png', dpi=90)
    print(json.dumps({run: {kind: {n: (m and {kk: m[kk] for kk in ('n', 'steer_mae_k3', 'throttle_mae_k3', 'steer_sign_acc', 'fwd_bias_us')})
                                   for n, m in report['models'][run][kind].items() if n != 'substituted_genre_frames'}
                            for kind in ('tf', 'sf')} for run in args.runs}, ensure_ascii=False, indent=1))
    print('レポート: %s' % out_dir)


if __name__ == '__main__':
    main()
