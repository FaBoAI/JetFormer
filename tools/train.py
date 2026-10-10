#!/usr/bin/env python3
"""JetFormerの学習(Jetsonの管理画面から、またはクラウドGPUで実行)

    python tools/train.py --data datasets/indoor_v1 --epochs 30 --stride 3 --out runs/indoor_v1
    → runs/indoor_v1/best.pth, jetformer.onnx, model_config.json(ジャンル表を含む), result.json
      Jetsonで tools/build_engine.py runs/indoor_v1/jetformer.onnx ... に渡す

--progress-json を付けると、進捗を「@@{JSON}」の行で出力する(管理画面が読む)。
SIGINT/SIGTERM で止めると、そこまでで最良のモデルを書き出してから終了する。
"""

import argparse
import json
import os
import random
import signal
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import model as jm  # noqa: E402
from jetformer.model import IMU_STEPS, TOF_CHANNELS, TOF_SHAPE, JetFormer  # noqa: E402
from jetformer.train_data import JetFormerDataset  # noqa: E402


class Stopped(Exception):
    pass


STOP = {'flag': False}
PROGRESS = {'enabled': False}


def emit(kind, **kw):
    if PROGRESS['enabled']:
        print('@@' + json.dumps(dict(type=kind, **kw), ensure_ascii=False), flush=True)


def run_epoch(model, loader, device, optimizer=None, amp=True, epoch=0, phase='train'):
    """平均L1誤差と、ジャンル別の平均L1誤差 {mode_id: loss} を返す"""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    per_mode = {}
    steps = len(loader)
    for k, (image, tof, imu, state, mode, target) in enumerate(loader):
        if STOP['flag']:
            raise Stopped()
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
        if k % 10 == 0 or k == steps - 1:
            emit('step', epoch=epoch, phase=phase, step=k + 1, steps=steps)
    return total / max(n, 1), {m: s / c for m, (s, c) in per_mode.items()}


def export_onnx(model, path, num_modes):
    model = model.float().eval().cpu()
    torch.backends.mha.set_fastpath_enabled(False)
    x = (torch.randn(1, 3, 224, 224), torch.rand(1, TOF_CHANNELS, *TOF_SHAPE), torch.zeros(1, IMU_STEPS, 6),
         torch.zeros(1, 3), jm.mode_onehot([1], num_modes))
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
    ap.add_argument('--stride', type=int, default=1, help='間引き: N枚に1枚を使う(1=全フレーム、3=10fps相当)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-pretrained', action='store_true', help='ImageNet事前学習の重みを使わない')
    ap.add_argument('--max-steps', type=int, default=0, help='動作確認用: 1エポックの最大ステップ数')
    ap.add_argument('--no-command-input', action='store_true',
                    help='今の操作値(ステアリング・スロットル)をモデルに入れない(AI走行時と学習時の条件をそろえる)')
    ap.add_argument('--keep-impacts', action='store_true',
                    help='衝突の前後(水平4G以上の1秒前〜2秒後)も学習に使う(既定では外す)')
    ap.add_argument('--impact-g', type=float, default=4.0,
                    help='衝突とみなす水平の加速度 [G](2.5 = 解析で衝突とした基準、4.0 = 強い衝突だけ)')
    ap.add_argument('--progress-json', action='store_true')
    args = ap.parse_args()
    PROGRESS['enabled'] = args.progress_json
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.__setitem__('flag', True))

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    use_command = not args.no_command_input
    mask_impacts = not args.keep_impacts
    train_ds = JetFormerDataset(args.data, 'train', color_jitter=True, stride=args.stride, use_command=use_command,
                                mask_impacts=mask_impacts, impact_g=args.impact_g)
    val_ds = JetFormerDataset(args.data, 'val', stride=args.stride, use_command=use_command, mask_impacts=mask_impacts,
                              impact_g=args.impact_g)
    if mask_impacts:
        print('衝突の前後として外したフレーム: 学習 %d / 検証 %d' % (train_ds.impact_masked, val_ds.impact_masked))
    names = {m['id']: m['name'] for m in train_ds.modes}
    if args.max_steps:      # 動作確認用: ジャンルが偏らないようランダムに抜き出す
        rng = random.Random(0)
        for ds in (train_ds, val_ds):
            rng.shuffle(ds.index)
            ds.index = ds.index[:args.max_steps * args.batch]
    if len(train_ds) == 0:
        emit('error', message='学習に使えるフレームがありません(操作値が無効・ジャンル未設定・AI走行中のフレームは除外されます)')
        raise SystemExit('学習に使えるフレームがありません')
    print('学習 %d サンプル / 検証 %d サンプル / 間引き 1/%d / ジャンル数 %d / device=%s' % (
        len(train_ds), len(val_ds), args.stride, train_ds.num_modes, device))
    mode_counts = {names.get(m, m): c for m, c in sorted(train_ds.mode_counts().items())}
    print('ジャンル別の学習サンプル数:', mode_counts)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    emit('start', train=len(train_ds), val=len(val_ds), epochs=args.epochs, stride=args.stride,
         steps_train=len(train_dl), steps_val=len(val_dl), modes=train_ds.modes, mode_counts=mode_counts)

    num_modes = train_ds.num_modes
    model = JetFormer(num_modes=num_modes, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best, best_epoch, best_modes = float('inf'), 0, {}
    history = []
    state = 'done'
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr, _ = run_epoch(model, train_dl, device, optimizer, epoch=epoch, phase='train')
            va, va_modes = run_epoch(model, val_dl, device, epoch=epoch, phase='val') if len(val_ds) else (tr, {})
            scheduler.step()
            improved = va < best
            if improved:
                best, best_epoch, best_modes = va, epoch, va_modes
                torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
            per_mode_us = {str(m): round(l * jm.PWM_SCALE, 1) for m, l in sorted(va_modes.items())}
            history.append({'epoch': epoch, 'train_us': round(tr * jm.PWM_SCALE, 1), 'val_us': round(va * jm.PWM_SCALE, 1),
                            'val_per_mode_us': per_mode_us, 'seconds': round(time.time() - t0, 1), 'best': improved})
            emit('epoch', **history[-1])
            detail = '  '.join('%s %.0fµs' % (names.get(m, m), l * jm.PWM_SCALE) for m, l in sorted(va_modes.items()))
            print('epoch %3d  train %.0fµs  val %.0fµs  %.0fs%s  [%s]' % (
                epoch, tr * jm.PWM_SCALE, va * jm.PWM_SCALE, time.time() - t0, ' *' if improved else '', detail), flush=True)
    except Stopped:
        state = 'stopped'
        print('停止の指示を受けました。ここまでで最良のモデルを書き出します。', flush=True)

    steer_center, throttle_neutral = train_ds.action_reference()
    train_mode_counts = {str(m): c for m, c in sorted(train_ds.mode_counts().items())}
    result = {'state': state, 'dataset': os.path.basename(args.data.rstrip('/')), 'stride': args.stride,
              'epochs_planned': args.epochs, 'epochs_done': len(history), 'batch': args.batch, 'lr': args.lr,
              'pretrained': not args.no_pretrained, 'train_samples': len(train_ds), 'val_samples': len(val_ds),
              'best_val_us': round(best * jm.PWM_SCALE, 1) if best_epoch else None, 'best_epoch': best_epoch,
              'best_val_per_mode_us': {str(m): round(l * jm.PWM_SCALE, 1) for m, l in sorted(best_modes.items())},
              'modes': train_ds.modes, 'train_mode_counts': train_mode_counts, 'command_input': use_command,
              'mask_impacts': mask_impacts, 'impact_g': args.impact_g,
              'impact_masked_frames': {'train': train_ds.impact_masked, 'val': val_ds.impact_masked},
              'steer_center_us': steer_center, 'throttle_neutral_us': throttle_neutral, 'history': history}
    if best_epoch:
        model.load_state_dict(torch.load(os.path.join(args.out, 'best.pth'), map_location='cpu', weights_only=True))
        export_onnx(model, os.path.join(args.out, 'jetformer.onnx'), num_modes)
        with open(os.path.join(args.out, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump({'num_modes': num_modes, 'modes': train_ds.modes, 'future_steps': jm.FUTURE_STEPS,
                       'inputs': ['image', 'tof', 'imu', 'state', 'mode'], 'tof_channels': TOF_CHANNELS,
                       'pwm_center': jm.PWM_CENTER, 'pwm_scale': jm.PWM_SCALE, 'tof_max_mm': jm.TOF_MAX_MM,
                       'erpm_scale': jm.ERPM_SCALE, 'dataset': result['dataset'], 'stride': args.stride,
                       # 走行時: 学習データがあるジャンルだけ選べる。ゲインはこの中心・中立の周りに掛ける
                       'trained_modes': [int(m) for m in train_mode_counts], 'train_mode_counts': train_mode_counts,
                       'steer_center_us': steer_center, 'throttle_neutral_us': throttle_neutral,
                       'command_input': use_command, 'mask_impacts': mask_impacts},
                      f, ensure_ascii=False, indent=1)
        print('保存: %s/best.pth, jetformer.onnx, model_config.json' % args.out, flush=True)
    else:
        result['state'] = 'stopped_no_model'
    result['duration_s'] = round(time.time() - t_start, 1)
    with open(os.path.join(args.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    emit('done', state=result['state'], best_val_us=result['best_val_us'], best_epoch=best_epoch,
         duration_s=result['duration_s'])


if __name__ == '__main__':
    main()
