#!/usr/bin/env python3
"""段 G1 の確かめ(2026-10-21、1 回だけ・学習はしない・CPU): train_data.py・train.py に足した選択肢が既定のままなら、
変える前(git の HEAD)の train_data.py と同じサンプル・同じバッチ・同じ損失になるか。新しい表(SLOW・WGT)が読めるか。

    cd ~/jetformer
    /home/jetson/bin/jf-heavy --lock big --mem 3G -- nice -n 10 env CUDA_VISIBLE_DEVICES= \
        /home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261021_bench_prep/train/check_default_same.py

本番のデータセット clean_driving_20261011 の学習の分け方を、モデル B と同じ設定(間引き 1/3・操作値なし・衝突の前後 2.5G を外す・
色の揺らぎあり)で読む。結果は experiments/20261021_bench_prep/train/check_default_same.json
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

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

DATA = os.path.join(JF, 'data/datasets/clean_driving_20261011')
DER = os.path.join(JF, 'data/derived/bench_20261021')
OUT = os.path.join(JF, 'experiments/20261021_bench_prep/train/check_default_same.json')
KW = dict(stride=3, use_command=False, mask_impacts=True, impact_g=2.5)


def head_module():
    """git の HEAD の jetformer/train_data.py を jetformer パッケージの中のモジュールとして読む"""
    src = subprocess.run(['git', '-C', JF, 'show', 'HEAD:jetformer/train_data.py'], check=True, capture_output=True).stdout
    d = tempfile.mkdtemp(prefix='g1_head_')
    path = os.path.join(d, 'train_data_head.py')
    with open(path, 'wb') as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('jetformer.train_data_head', path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'jetformer'
    spec.loader.exec_module(mod)
    return mod


def same_items(a, b, idx, seed):
    np.random.seed(seed)
    xa = [a[i] for i in idx]
    ra = np.random.get_state()[1].copy()
    np.random.seed(seed)
    xb = [b[i] for i in idx]
    rb = np.random.get_state()[1].copy()
    ok = all(len(p) == len(q) and all(torch.equal(u, v) for u, v in zip(p, q)) for p, q in zip(xa, xb))
    return ok and np.array_equal(ra, rb)


def batch_loss(model, batch):
    image, tof, imu, state, mode, target = batch[:6]
    with torch.no_grad():
        out = runtime.forward(model, dict(image=image, tof=tof, imu=imu, state=state, mode=mode), target=target, train=False)
        return float(runtime.l1_per_sample(out['actions'], target).mean())


def main():
    t0 = time.time()
    res = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'dataset': 'clean_driving_20261011', 'settings': KW}
    old_td = head_module()
    old = old_td.JetFormerDataset(DATA, 'train', color_jitter=True, **KW)
    new = new_td.JetFormerDataset(DATA, 'train', color_jitter=True, **KW)
    new_none = new_td.JetFormerDataset(DATA, 'train', color_jitter=True, label_override=None, aug=None, **KW)
    res['train_samples'] = {'head': len(old), 'new': len(new), 'new_explicit_none': len(new_none)}
    res['index_same'] = old.index == new.index == new_none.index
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(old), 64, replace=False).tolist())
    res['items_same_64'] = same_items(old, new, idx, 7) and same_items(old, new_none, idx, 7)
    # 同じ種で同じバッチ・同じ損失(DataLoader のシャッフル・色の揺らぎ。モデルは jetformer_v0 の既定の大きさ・事前学習なし)
    batches = {}
    for name, ds in (('head', old), ('new', new)):
        torch.manual_seed(0)
        np.random.seed(0)
        dl = DataLoader(ds, 16, shuffle=True, num_workers=0)
        batches[name] = next(iter(dl))
    res['batch_same'] = all(torch.equal(u, v) for u, v in zip(batches['head'], batches['new']))
    torch.manual_seed(0)
    model = policies.build_policy('jetformer_v0', old.num_modes, policies.get('jetformer_v0').resolve({}), pretrained=False)
    model.eval()
    res['loss_head'] = batch_loss(model, batches['head'])
    res['loss_new'] = batch_loss(model, batches['new'])
    res['loss_same'] = res['loss_head'] == res['loss_new']
    # 検証の分け方も同じ
    vo = old_td.JetFormerDataset(DATA, 'val', **KW)
    vn = new_td.JetFormerDataset(DATA, 'val', **KW)
    res['val_index_same'] = vo.index == vn.index
    res['val_items_same_32'] = same_items(vo, vn, list(range(0, len(vo), max(1, len(vo) // 32)))[:32], 1)
    # SLOW の表: 学習の分け方に渡せる・数が tables_card と合う・ステアは変わらない・val には渡せない
    card = json.load(open(os.path.join(DER, 'tables_card.json'), encoding='utf-8'))
    slow = new_td.load_frame_table(os.path.join(DER, 'slow_labels.npz'), 'future_throttle_us')
    ds = new_td.JetFormerDataset(DATA, 'train', label_override=slow, **KW)
    rows_used_card = sum(v['rows_used'] for v in card['stats']['changed'].values())
    res['slow'] = {'rows_all': ds.label_overridden, 'rows_used': ds.label_overridden_used,
                   'card_rows_all': card['stats']['rows_changed'], 'card_rows_used': rows_used_card,
                   'index_same_as_default': ds.index == new.index}
    st_same, thr_ge = True, True
    for a, b in zip(new.scenes, ds.scenes):
        st_same &= bool(np.array_equal(a['future_actions'][:, :, 0], b['future_actions'][:, :, 0]))
        thr_ge &= bool(np.all(b['future_actions'][:, :, 1] >= a['future_actions'][:, :, 1] - 1e-4))
    res['slow']['steer_same'] = st_same
    res['slow']['throttle_never_stronger'] = thr_ge        # PWM が大きい = 前進が弱い。前進を強めない
    try:
        new_td.JetFormerDataset(DATA, 'val', label_override=slow, **KW)
        res['slow']['val_rejected'] = False
    except KeyError:
        res['slow']['val_rejected'] = True
    # WGT の表: 学習・検証の両方に渡せる(train.py と同じ)。検証の重みは全部 1
    wgt = new_td.load_frame_table(os.path.join(DER, 'wgt_weights.npz'), 'weight')
    tw = new_td.JetFormerDataset(DATA, 'train', weights=wgt, return_weight=True, **KW)
    vw = new_td.JetFormerDataset(DATA, 'val', weights=wgt, return_weight=True, **KW)
    w = tw.sample_weights()
    res['wgt'] = {'train_samples': len(tw), 'weight_mean': round(float(w.mean()), 4), 'share_3x': round(float(np.mean(w > 1.5)), 4),
                  'effective_n': round(float(w.sum() ** 2 / (w ** 2).sum()), 1), 'val_all_one': bool(np.all(vw.sample_weights() == 1.0)),
                  'index_same_as_default': tw.index == new.index}
    # AUG: 学習の分け方だけに掛けて、正解は同じ
    aug = {'brightness': 0.2, 'contrast': 0.2, 'shift_px': 4, 'tof_noise': 0.02}
    da = new_td.JetFormerDataset(DATA, 'train', aug=aug, **KW)
    np.random.seed(0)
    xa = da[idx[0]]
    xb = new_td.JetFormerDataset(DATA, 'train', **KW)[idx[0]]
    res['aug'] = {'settings': aug, 'target_same': bool(torch.equal(xa[5], xb[5])), 'image_changed': not torch.equal(xa[0], xb[0]),
                  'tof_dist_changed': not torch.equal(xa[1][0], xb[1][0]), 'tof_conf_same': bool(torch.equal(xa[1][1], xb[1][1]))}
    res['args_default_unused'] = train.bench_args_used(type('A', (), dict(train.BENCH_ARGS))()) == {}
    res['all_ok'] = bool(res['index_same'] and res['items_same_64'] and res['batch_same'] and res['loss_same']
                         and res['val_index_same'] and res['val_items_same_32']
                         and res['slow']['rows_all'] == res['slow']['card_rows_all']
                         and res['slow']['rows_used'] == res['slow']['card_rows_used']
                         and res['slow']['index_same_as_default'] and st_same and thr_ge and res['slow']['val_rejected']
                         and res['wgt']['val_all_one'] and res['wgt']['index_same_as_default']
                         and res['aug']['target_same'] and res['aug']['image_changed'] and res['args_default_unused'])
    res['duration_s'] = round(time.time() - t0, 1)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
