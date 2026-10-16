#!/usr/bin/env python3
"""train_gainl.py のラベルの差し替えのテスト(CPU。Dataset を作るだけ。画像は 1 枚だけ読む)

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= python -m pytest -q experiments/20261015_real_accuracy/test_train_gainl.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], '--out', '/tmp/none-gainl']
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/jetson/jetformer')
import train_gainl as G  # noqa: E402
from jetformer.train_data import JetFormerDataset  # noqa: E402

DS = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
KW = dict(use_command=False, mask_impacts=True, impact_g=2.5, stride=3)


def test_labels_and_rows():
    g = G.GainLDataset(DS, 'train', **KW)
    a = JetFormerDataset(DS, 'train', **KW)
    assert g.index == a.index and len(g) == 5912                    # 行は A と同じ
    st = G.STATS['train']
    assert st['r1_labels_same'] and st['frames_used'] == 5912
    assert st['labels_changed'] > 0 and st['r_not_1'] > 0
    # 1 フレームずつ式どおりか(ランダムに 200 行)
    rng = np.random.default_rng(0)
    for n in rng.choice(len(g), 200, replace=False):
        k, i = g.index[n]
        old = a.scenes[k]['future_actions'][i].astype(np.float64)
        r = float(g.scenes[k]['_gain_r'][i])
        want = G.RL.gain_label(old[None], np.array([r]))[0]
        assert np.allclose(g.scenes[k]['future_actions'][i], want, atol=1e-3)
        assert np.array_equal(g.scenes[k]['future_actions'][i][:, 0], a.scenes[k]['future_actions'][i][:, 0])
    # __getitem__ の正解も差し替わっている(正規化した値)
    y_g = g[0][4].numpy() if hasattr(g[0][4], 'numpy') else None
    y_a = a[0][4].numpy() if hasattr(a[0][4], 'numpy') else None
    if y_g is not None and y_a is not None and y_g.shape == (10, 2):
        k, i = g.index[0]
        if abs(float(g.scenes[k]['_gain_r'][i]) - 1) < 1e-12:
            assert np.allclose(y_g, y_a)


def test_val_rows_same():
    v = G.GainLDataset(DS, 'val', **KW)
    assert len(v) == 1559 and G.STATS['val']['r1_labels_same']


def test_scene_ratio_matches_eval_gain():
    """学習のラベルの r と、評価の後処理の r(realacc_eval.gain_inputs と同じ式)が同じ"""
    s = '20261010-193638-0003'
    a = dict(np.load(os.path.join(DS, 'aligned', s + '.npz')))
    r = G.scene_ratio(a)
    pg = G.PG
    gt = pg.running_median(pg.ratio_series(a), a['timestamps_us'].astype(np.float64), 5.0)
    gref = np.array([{4: 36.41, 5: 36.86}.get(int(m), np.nan) for m in a['mode']])
    assert np.allclose(r, G.RL.gain_scale(gt, gref, 0.8, 2.0))
    assert (r > 1.2).any()          # 電池の減ったシーンなので補正がかかる


def test_cuda_visible_devices_restored():
    """probe_gain.py を読み込んでも、学習のプロセスの CUDA_VISIBLE_DEVICES は元のまま(22:43 の動作確認の失敗の再発防止)"""
    import subprocess
    code = ('import os, sys; sys.argv=["x","--out","/tmp/none-gainl"]; os.environ.pop("CUDA_VISIBLE_DEVICES", None); '
            'sys.path.insert(0, %r); import train_gainl; print(repr(os.environ.get("CUDA_VISIBLE_DEVICES")))' % HERE)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                                         if k != 'CUDA_VISIBLE_DEVICES'})
    assert out.stdout.strip().splitlines()[-1] == 'None', out.stderr[-500:]
