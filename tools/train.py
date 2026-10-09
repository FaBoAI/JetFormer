#!/usr/bin/env python3
"""JetFormerの学習(クラウドGPU想定。Jetsonでも動作確認できる)

    # エクスポートしたtarを展開したディレクトリを指定
    python tools/train.py --data datasets/indoor_v1 --epochs 30 --out runs/indoor_v1
    → runs/indoor_v1/best.pth と jetformer.onnx(Jetsonで tools/build_engine.py に渡す)
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer.model import IMU_STEPS, TOF_SHAPE, JetFormer  # noqa: E402
from jetformer.train_data import JetFormerDataset  # noqa: E402


def run_epoch(model, loader, device, optimizer=None, amp=True):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for image, tof, imu, state, target in loader:
        image, tof, imu, state, target = (t.to(device, non_blocking=True) for t in (image, tof, imu, state, target))
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            pred = model(image, tof, imu, state)
            loss = torch.nn.functional.l1_loss(pred.float(), target)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * image.shape[0]
        n += image.shape[0]
    return total / max(n, 1)


def export_onnx(model, path):
    model = model.float().eval().cpu()
    torch.backends.mha.set_fastpath_enabled(False)
    x = (torch.randn(1, 3, 224, 224), torch.rand(1, 1, *TOF_SHAPE), torch.zeros(1, IMU_STEPS, 6), torch.zeros(1, 3))
    with torch.no_grad():
        torch.onnx.export(model, x, path, input_names=['image', 'tof', 'imu', 'state'], output_names=['actions'],
                          opset_version=17, dynamo=False)


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
    print('学習 %d サンプル / 検証 %d サンプル / device=%s' % (len(train_ds), len(val_ds), device))
    if args.max_steps:
        train_ds.index = train_ds.index[:args.max_steps * args.batch]
        val_ds.index = val_ds.index[:args.max_steps * args.batch]
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = JetFormer(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, optimizer)
        va = run_epoch(model, val_dl, device) if len(val_ds) else tr
        scheduler.step()
        mark = ''
        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
            mark = ' *'
        # L1誤差を µs に換算して表示(正規化スケール500µs)
        print('epoch %3d  train %.4f (%.0fµs)  val %.4f (%.0fµs)  %.0fs%s' % (epoch, tr, tr * 500, va, va * 500, time.time() - t0, mark))
    model.load_state_dict(torch.load(os.path.join(args.out, 'best.pth'), map_location='cpu', weights_only=True))
    export_onnx(model, os.path.join(args.out, 'jetformer.onnx'))
    print('保存: %s/best.pth, jetformer.onnx' % args.out)


if __name__ == '__main__':
    main()
