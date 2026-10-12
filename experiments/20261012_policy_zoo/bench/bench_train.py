# -*- coding: utf-8 -*-
"""学習の1ステップ(バッチ64・順伝播+逆伝播+AdamW、AMP FP16)の時間とメモリを測る(データは GPU 上の乱数。読み込みの時間は含まない)

    python bench_train.py <名前> [--batch 64]
結果: results/<名前>_train.json(済んでいれば飛ばす)
tools/train.py の run_epoch と同じ: autocast(float16)・GradScaler なし・clip_grad_norm 1.0・AdamW(lr 3e-4, wd 1e-4)
"""

import os
import sys
import time

import numpy as np

from common import RESULTS, load_json, save_json, wait_gpu_free

MEM_FRACTION = 0.55          # 8GB の共有メモリ(スワップなし)で OOM による再起動を避ける: PyTorch の上限を約 4GB にする


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    name = sys.argv[1]
    batch = int(sys.argv[sys.argv.index('--batch') + 1]) if '--batch' in sys.argv else 64
    out_path = os.path.join(RESULTS, '%s_train%s.json' % (name, '' if batch == 64 else '_b%d' % batch))
    res = load_json(out_path)
    if res.get('done'):
        log('済み', name)
        return
    if not wait_gpu_free(log=log):
        raise SystemExit('GPU が空かないので止めます')
    import torch
    import protos
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)
    dev = torch.device('cuda')
    model = protos.build(name).to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    x = [t.to(dev) for t in protos.example_inputs(batch=batch)]
    x[4] = torch.nn.functional.one_hot(torch.randint(4, 6, (batch,)), 6).float().to(dev)
    target = (torch.randn(batch, 10, 2, device=dev) * 0.2)
    wp_t = torch.cumsum(torch.rand(batch, 10, 2, device=dev) * 0.1, dim=1)
    sp_t = torch.rand(batch, 1, device=dev) * 3

    def step():
        with torch.autocast('cuda', dtype=torch.float16):
            if name.startswith('act_cvae'):
                y = model(*x, actions=target)
            else:
                y = model(*x)
            y = protos.outputs_of(y)
            loss = (y['actions'].float() - target).abs().mean()
            if 'waypoints' in y:
                loss = loss + 0.5 * (y['waypoints'].float() - wp_t).abs().mean()
            if 'target_speed' in y:
                loss = loss + (y['target_speed'].float() - sp_t).abs().mean()
            if 'speed_logits' in y:          # two-hot の交差エントロピー(重み 0.1)
                p = torch.softmax(torch.randn_like(y['speed_logits'].float()), dim=-1)
                loss = loss - 0.1 * (p * torch.log_softmax(y['speed_logits'].float(), dim=-1)).sum(-1).mean()
            if getattr(model, 'last_kl', None) is not None:
                loss = loss + 10.0 * model.last_kl.float()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    def evaluate():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            model(*x)

    torch.cuda.reset_peak_memory_stats()
    t_first = time.time()
    try:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as e:
        res.update({'name': name, 'batch': batch, 'oom': True, 'error': str(e)[:300], 'done': True})
        save_json(out_path, res)
        log('メモリ不足', name)
        return
    warm_s = time.time() - t_first
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    peak_alloc = torch.cuda.max_memory_allocated() / 1e6
    peak_res = torch.cuda.max_memory_reserved() / 1e6
    model.eval()
    ev = []
    for _ in range(3):
        evaluate()
    torch.cuda.synchronize()
    for _ in range(10):
        t0 = time.perf_counter()
        evaluate()
        torch.cuda.synchronize()
        ev.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    res.update({
        'name': name, 'batch': batch, 'amp': 'fp16 autocast(GradScaler なし)',
        'train_step_ms_median': round(float(np.median(t)), 1), 'train_step_ms_p90': round(float(np.percentile(t, 90)), 1),
        'train_samples_per_s_gpu': round(batch / (np.median(t) / 1000), 1),
        'eval_batch_ms_median': round(float(np.median(ev)), 1),
        'peak_allocated_mb': round(peak_alloc, 0), 'peak_reserved_mb': round(peak_res, 0),
        'warmup_s': round(warm_s, 1),
        'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        'trainable_params_m': round(sum(p.numel() for p in params) / 1e6, 2),
        'done': True})
    save_json(out_path, res)
    log(name, '学習 %.0f ms/step(%.0f 枚/秒)・検証 %.0f ms/バッチ・ピーク %.0f MB' % (
        res['train_step_ms_median'], res['train_samples_per_s_gpu'], res['eval_batch_ms_median'], peak_alloc))


if __name__ == '__main__':
    main()
