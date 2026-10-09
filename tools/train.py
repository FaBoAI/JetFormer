#!/usr/bin/env python3
"""JetFormerの学習(クラウドGPU想定。Jetsonでも動作確認できる)

    # エクスポートしたtarを展開したディレクトリを指定
    python tools/train.py --data datasets/indoor_v1 --epochs 30 --out runs/indoor_v1
    → runs/indoor_v1/best.pth, jetformer.onnx, model_config.json(ジャンル表を含む)
      Jetsonで tools/build_engine.py runs/indoor_v1/jetformer.onnx ... に渡す
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import model as jm  # noqa: E402
from jetformer.model import IMU_STEPS, TOF_SHAPE, JetFormer  # noqa: E402
from jetformer.train_data import JetFormerDataset  # noqa: E402


def run_epoch(model, loader, device, optimizer=None, amp=True):
    """平均L1誤差と、ジャンル別の平均L1誤差 {mode_id: loss} を返す"""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    per_mode = {}
    for image, tof, imu, state, mode, target in loader:
        image, tof, imu, state, mode, target = (t.to(device, non_blocking=True)
                                                for t in (image, tof, imu, state, mode, target))
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16,
                                                           enabled=amp and device.type == 'cuda'):
            pred = model(image, tof, imu, state, mode)
            per_sample = (pred.float() - target).abs().mean(dim=(1, 2))
            loss = per_sample.mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * image.shape[0]
        n += image.shape[0]
        for m, l in zip(mode.argmax(dim=1).tolist(), per_sample.detach().tolist()):
            s, c = per_mode.get(m, (0.0, 0))
            per_mode[m] = (s + l, c + 1)
    return total / max(n, 1), {m: s / c for m, (s, c) in per_mode.items()}


def export_onnx(model, path, num_modes):
    model = model.float().eval().cpu()
    torch.backends.mha.set_fastpath_enabled(False)
    x = (torch.randn(1, 3, 224, 224), torch.rand(1, 1, *TOF_SHAPE), torch.zeros(1, IMU_STEPS, 6), torch.zeros(1, 3),
         jm.mode_onehot([1], num_modes))
    with torch.no_grad():
        torch.onnx.export(model, x, path, input_names=['image', 'tof', 'imu', 'state', 'mode'],
                          output_names=['actions'], opset_version=17, dynamo=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='データセットのディレクトリ(splits.jsonがある場所)')
    ap.add_argument('--out', default='runs/jetformer')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-pretrained', action='store_true', help='ImageNet事前学習の重みを使わない')
    ap.add_argument('--max-steps', type=int, default=0, help='動作確認用: 1エポックの最大ステップ数')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    train_ds = JetFormerDataset(args.data, 'train', color_jitter=True)
    val_ds = JetFormerDataset(args.data, 'val')
    names = {m['id']: m['name'] for m in train_ds.modes}
    print('学習 %d サンプル / 検証 %d サンプル / ジャンル数 %d / device=%s' % (
        len(train_ds), len(val_ds), train_ds.num_modes, device))
    print('ジャンル別の学習サンプル数:', {names.get(m, m): c for m, c in sorted(train_ds.mode_counts().items())})
    if args.max_steps:      # 動作確認用: ジャンルが偏らないようランダムに抜き出す
        rng = random.Random(0)
        for ds in (train_ds, val_ds):
            rng.shuffle(ds.index)
            ds.index = ds.index[:args.max_steps * args.batch]
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_modes = train_ds.num_modes
    model = JetFormer(num_modes=num_modes, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr, _ = run_epoch(model, train_dl, device, optimizer)
        va, va_modes = run_epoch(model, val_dl, device) if len(val_ds) else (tr, {})
        scheduler.step()
        mark = ''
        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
            mark = ' *'
        # L1誤差を µs に換算して表示(正規化スケール500µs)
        detail = '  '.join('%s %.0fµs' % (names.get(m, m), l * jm.PWM_SCALE) for m, l in sorted(va_modes.items()))
        print('epoch %3d  train %.0fµs  val %.0fµs  %.0fs%s  [%s]' % (
            epoch, tr * jm.PWM_SCALE, va * jm.PWM_SCALE, time.time() - t0, mark, detail))
    model.load_state_dict(torch.load(os.path.join(args.out, 'best.pth'), map_location='cpu', weights_only=True))
    export_onnx(model, os.path.join(args.out, 'jetformer.onnx'), num_modes)
    with open(os.path.join(args.out, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump({'num_modes': num_modes, 'modes': train_ds.modes, 'future_steps': jm.FUTURE_STEPS,
                   'inputs': ['image', 'tof', 'imu', 'state', 'mode'], 'pwm_center': jm.PWM_CENTER,
                   'pwm_scale': jm.PWM_SCALE, 'tof_max_mm': jm.TOF_MAX_MM, 'dataset': os.path.basename(args.data.rstrip('/'))},
                  f, ensure_ascii=False, indent=1)
    print('保存: %s/best.pth, jetformer.onnx, model_config.json' % args.out)


if __name__ == '__main__':
    main()
