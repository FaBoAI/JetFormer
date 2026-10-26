#!/usr/bin/env python3
"""C1 の確かめ(2026-10-25、学習はしない・CPU): train_data.py・train.py に足した --aug-top-brightness が既定(0)のままなら、
C1 の前(commit c247daa。protocol_v2.json の defaults_untouched.sha256_now と同じ中身)の train_data.py と
同じサンプル・同じ乱数・同じバッチ・同じ損失になるか。10/21 の拡張(上 1/3 なし)も同じか。上 1/3 の拡張の大きさ。
experiments/20261021_bench_prep/train/check_default_same.py の形。

    cd ~/jetformer
    /home/jetson/bin/jf-heavy --lock big --mem 3872M --need-mb 3200 -- nice -n 10 env CUDA_VISIBLE_DEVICES= \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261025_train_v2/check_c1_default_same.py

本番のデータセット dataset_v2_20261025 の v2-H(drop_v2h.npz)の学習・検証の分け方を、段 3 と同じ設定(間引き 1/3・操作値なし・
衝突の前後 2.5G を外す・色の揺らぎあり)で読む。メモリを抑えるため、前と後のデータセットは 1 つずつ読む。
結果は experiments/20261025_train_v2/check_c1_default_same.json
"""
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

JF = '/home/jetson/jetformer'
sys.path.insert(0, JF)
sys.path.insert(0, os.path.join(JF, 'tools'))
from jetformer import policies  # noqa: E402
from jetformer import train_data as new_td  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
import train  # noqa: E402

BASE_REV = 'c247daa'
BASE_SHA256 = '8762dda4bc8c7f347bc8f706743a35f4f93e5b15a8d3e487ea39d59509d717aa'     # protocol defaults_untouched
DATA = os.path.join(JF, 'data/datasets/dataset_v2_20261025')
DROP = os.path.join(JF, 'data/derived/dataset_v2_20261025/drop_v2h.npz')
OUT = os.path.join(JF, 'experiments/20261025_train_v2/check_c1_default_same.json')
KW = dict(stride=3, use_command=False, mask_impacts=True, impact_g=2.5)
AUG_1021 = {'brightness': 0.2, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02}
AUG2 = {'brightness': 0.15, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02, 'top_brightness': 0.15}


def base_module():
    src = subprocess.run(['git', '-C', JF, 'show', '%s:jetformer/train_data.py' % BASE_REV], check=True,
                         capture_output=True).stdout
    if hashlib.sha256(src).hexdigest() != BASE_SHA256:
        raise SystemExit('C1 の前の train_data.py の sha256 が protocol と違う')
    d = tempfile.mkdtemp(prefix='c1_base_')
    path = os.path.join(d, 'train_data_c1base.py')
    with open(path, 'wb') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('jetformer.train_data_c1base', path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    spec.loader.exec_module(mod)
    return mod


def tables(td):
    rl = {'drop': td.load_frame_table(DROP, 'drop')}
    with np.load(DROP) as z:
        if 'keep_impact' in z.files:
            rl['keep_impact'] = td.load_frame_table(DROP, 'keep_impact')
    return rl


def items(ds, idx, seed):
    np.random.seed(seed)
    x = [ds[i] for i in idx]
    return x, np.random.get_state()[1].copy()


def same(xa, xb):
    return len(xa) == len(xb) and all(len(p) == len(q) and all(torch.equal(u, v) for u, v in zip(p, q))
                                      for p, q in zip(xa, xb))


def first_batch(ds):
    torch.manual_seed(0)
    np.random.seed(0)
    return next(iter(DataLoader(ds, 16, shuffle=True, num_workers=0)))


def batch_loss(model, batch):
    image, tof, imu, state, mode, target = batch[:6]
    with torch.no_grad():
        out = runtime.forward(model, dict(image=image, tof=tof, imu=imu, state=state, mode=mode), target=target, train=False)
        return float(runtime.l1_per_sample(out['actions'], target).mean())


def snapshot(td, idx_seed=0):
    """1 つの train_data で: 学習の分け方(既定・10/21 の AUG)と検証の分け方の、比べるもの一式"""
    s = {}
    tr = td.JetFormerDataset(DATA, 'train', color_jitter=True, **KW, **tables(td))
    s['n_train'], s['index'] = len(tr), list(tr.index)
    s['dropped'], s['impact_masked'] = tr.dropped, tr.impact_masked
    rng = np.random.default_rng(idx_seed)
    s['idx'] = sorted(rng.choice(len(tr), 64, replace=False).tolist())
    s['items'], s['rng'] = items(tr, s['idx'], 7)
    s['batch'] = first_batch(tr)
    s['num_modes'] = tr.num_modes
    s['action_reference'] = tr.action_reference()
    del tr
    gc.collect()
    ta = td.JetFormerDataset(DATA, 'train', color_jitter=True, aug=dict(AUG_1021), **KW, **tables(td))
    s['aug_items'], s['aug_rng'] = items(ta, s['idx'], 9)
    del ta
    gc.collect()
    va = td.JetFormerDataset(DATA, 'val', **KW, **tables(td))
    s['n_val'], s['val_index'] = len(va), list(va.index)
    vidx = list(range(0, len(va), max(1, len(va) // 32)))[:32]
    s['val_items'], s['val_rng'] = items(va, vidx, 1)
    del va
    gc.collect()
    return s


def top_effect(n=400):
    """上 1/3 の拡張(F = 0.15)で、実際の画像の上 1/3 の灰色の平均がどれだけ動くか(0〜255)。下 2/3 は動かない"""
    ds = new_td.JetFormerDataset(DATA, 'train', **KW, **tables(new_td))
    rng = np.random.default_rng(3)
    idx = rng.choice(len(ds), n, replace=False)
    np.random.seed(5)
    d_top, d_bottom, top0 = [], [], []
    for i in idx:
        k, j = ds.index[int(i)]
        img = cv2.imread(os.path.join(DATA, str(ds.scenes[k]['cam_files'][j])))
        h = img.shape[0] // 3
        out = new_td.augment_image(img, top_brightness=0.15)
        g0, g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        top0.append(float(g0[:h].mean()))
        d_top.append(float(g1[:h].mean() - g0[:h].mean()))
        d_bottom.append(float(np.abs(g1[h + 1:] - g0[h + 1:]).max()))
    d = np.array(d_top)
    res = {'samples': n, 'image_shape': list(img.shape), 'top_third_gray_median': round(float(np.median(top0)), 1),
           'top_third_mean_shift': {'min': round(float(d.min()), 1), 'p10': round(float(np.percentile(d, 10)), 1),
                                    'median': round(float(np.median(d)), 1), 'p90': round(float(np.percentile(d, 90)), 1),
                                    'max': round(float(d.max()), 1), 'abs_median': round(float(np.median(np.abs(d))), 1)},
           'bottom_two_thirds_max_change': float(max(d_bottom)),
           'top_row_max_shift_theory': round(0.15 * 255, 1),
           'top_third_mean_shift_theory': '±%.1f(一様、上 1/3 の重みの平均 %.3f)' % (
               0.15 * 255 * float(new_td.top_ramp(img.shape[0])[:h].mean()), float(new_td.top_ramp(img.shape[0])[:h].mean())),
           'dusk_vs_day_top_third': '10/24 夕方 100〜101・昼 112〜115(design_train.md 1.6)= −11〜−15'}
    del ds
    gc.collect()
    # 段 3 の AUG2 でも正解・ToF の信頼度は同じ、画像・ToF の距離は変わる(データセットは 1 つずつ読む)
    pick = [int(i) for i in idx[:32]]

    def items_of(aug):
        d = new_td.JetFormerDataset(DATA, 'train', aug=aug, **KW, **tables(new_td))
        xs = []
        for i in pick:
            np.random.seed(i)
            xs.append(d[i])
        index = list(d.index)
        del d
        gc.collect()
        return xs, index

    xb, ib = items_of(None)
    xa, ia = items_of(dict(AUG2))
    xt, it = items_of({'top_brightness': 0.15})
    res['aug2'] = {'settings': AUG2,
                   'target_same_32': all(torch.equal(x[5], y[5]) and torch.equal(x[5], z[5]) for x, y, z in zip(xb, xa, xt)),
                   'tof_conf_same_32': all(torch.equal(x[1][1], y[1][1]) for x, y in zip(xb, xa)),
                   'image_changed_of_32': sum(int(not torch.equal(x[0], y[0])) for x, y in zip(xb, xa)),
                   'tof_dist_changed_of_32': sum(int(not torch.equal(x[1][0], y[1][0])) for x, y in zip(xb, xa)),
                   'top_only_changes_only_top_third_32': all(torch.equal(x[0][:, 75:], z[0][:, 75:]) and torch.equal(x[1], z[1])
                                                             for x, z in zip(xb, xt)),
                   'top_only_image_changed_of_32': sum(int(not torch.equal(x[0][:, :75], z[0][:, :75])) for x, z in zip(xb, xt)),
                   'index_same_as_default': ia == ib == it}
    return res


def main():
    t0 = time.time()
    res = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'dataset': 'dataset_v2_20261025', 'table': 'drop_v2h.npz',
           'settings': KW, 'base': '%s(train_data.py sha256 %s…)' % (BASE_REV, BASE_SHA256[:16])}
    old = snapshot(base_module())
    new = snapshot(new_td)
    res['train_samples'] = {'base': old['n_train'], 'new': new['n_train']}
    res['val_samples'] = {'base': old['n_val'], 'new': new['n_val']}
    res['index_same'] = old['index'] == new['index'] and old['dropped'] == new['dropped'] and \
        old['impact_masked'] == new['impact_masked']
    res['items_same_64'] = same(old['items'], new['items'])
    res['rng_same_64'] = bool(np.array_equal(old['rng'], new['rng']))
    res['batch_same'] = all(torch.equal(u, v) for u, v in zip(old['batch'], new['batch']))
    torch.manual_seed(0)
    model = policies.build_policy('jetformer_v0', old['num_modes'], policies.get('jetformer_v0').resolve({}), pretrained=False)
    model.eval()
    res['loss_base'] = batch_loss(model, old['batch'])
    res['loss_new'] = batch_loss(model, new['batch'])
    res['loss_same'] = res['loss_base'] == res['loss_new']
    res['action_reference'] = {'base': old['action_reference'], 'new': new['action_reference']}
    res['aug1021_items_same_64'] = same(old['aug_items'], new['aug_items'])
    res['aug1021_rng_same_64'] = bool(np.array_equal(old['aug_rng'], new['aug_rng']))
    res['val_index_same'] = old['val_index'] == new['val_index']
    res['val_items_same_32'] = same(old['val_items'], new['val_items']) and bool(np.array_equal(old['val_rng'], new['val_rng']))
    del old, new
    gc.collect()
    res['top'] = top_effect()
    args = type('A', (), dict(train.BENCH_ARGS))()
    res['args_default_unused'] = train.bench_args_used(args) == {} and train.aug_of(args) is None
    a2 = res['top']['aug2']
    res['all_ok'] = bool(res['index_same'] and res['items_same_64'] and res['rng_same_64'] and res['batch_same']
                         and res['loss_same'] and res['action_reference']['base'] == res['action_reference']['new']
                         and res['aug1021_items_same_64'] and res['aug1021_rng_same_64']
                         and res['val_index_same'] and res['val_items_same_32'] and res['args_default_unused']
                         and res['top']['bottom_two_thirds_max_change'] == 0.0
                         and a2['target_same_32'] and a2['tof_conf_same_32'] and a2['image_changed_of_32'] >= 30
                         and a2['top_only_changes_only_top_third_32'] and a2['index_same_as_default'])
    res['duration_s'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
