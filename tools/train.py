#!/usr/bin/env python3
"""JetFormerの学習(Jetsonの管理画面から、またはクラウドGPUで実行)

    python tools/train.py --data datasets/indoor_v1 --epochs 30 --stride 3 --out runs/indoor_v1
    → runs/indoor_v1/best.pth, jetformer.onnx, model_config.json(ジャンル表を含む), result.json
      Jetsonで tools/build_engine.py runs/indoor_v1/jetformer.onnx ... に渡す

--progress-json を付けると、進捗を「@@{JSON}」の行で出力する(管理画面が読む)。
SIGINT/SIGTERM で止めると、そこまでで最良のモデルを書き出してから終了する。
各エポックの終わりに last.pth(モデル・最適化・学習率・履歴)を保存する。電源断などで止まったら、
同じ引数に --resume を付けて実行すると、続きのエポックから再開する(2026-10-11、電池切れで26エポック目で止まった)。

オフライン強化学習の方策(tools/rl_policies.py が使う。既定ではどれも使わず、今までと同じ学習になる):
    --init <pth>                 学習済みの重みから始める(ジャンル数が増えるときは、増えたジャンルを
                                 --init-mode-copy 6:5,7:4 の元のジャンルの埋め込み・ヘッドで初期化する)
    --seed N                     乱数の種(シャッフル・色の揺らぎ・ジャンルの付け替え)
    --drop-mask <npz>            フレームの表(drop・任意で keep_impact)。drop のフレームを外す。keep_impact は衝突の前後の除外から外す
    --sample-weights-npz <npz>   フレームの表(weight)。損失の重み(学習の平均が1になるよう割ってから掛ける)
    --mode-remap-npz <npz>       フレームの表(mode)。学習するジャンルを付け替える。--mode-remap-dropout の確率で元のジャンルに戻す
"""

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer import model as jm  # noqa: E402
from jetformer.model import FUTURE_STEPS, IMU_STEPS, TOF_CHANNELS, TOF_SHAPE, JetFormer  # noqa: E402
from jetformer.train_data import JetFormerDataset, load_frame_table  # noqa: E402


class Stopped(Exception):
    pass


STOP = {'flag': False}
PROGRESS = {'enabled': False}
# オフライン強化学習の方策用に足した引数と既定値(既定値なら使わない・記録しない)
RL_ARGS = {'init': None, 'init_mode_copy': None, 'seed': None, 'drop_mask': None, 'sample_weights_npz': None,
           'mode_remap_npz': None, 'mode_remap_dropout': 0.0}


def rl_args_used(args):
    """既定値から変えた RL_ARGS {名前: 値}"""
    return {k: getattr(args, k) for k, d in RL_ARGS.items() if getattr(args, k) != d}


def emit(kind, **kw):
    if PROGRESS['enabled']:
        print('@@' + json.dumps(dict(type=kind, **kw), ensure_ascii=False), flush=True)


def run_epoch(model, loader, device, optimizer=None, amp=True, epoch=0, phase='train', weight_scale=1.0, stats=None):
    """平均L1誤差と、ジャンル別の平均L1誤差 {mode_id: loss} を返す

    バッチに重み(7番目の値。--sample-weights-npz)があるとき: 学習の損失は mean(w/weight_scale · L1)、返す誤差は
    重み付きの平均 Σw·L1/Σw(学習・検証とも同じ定義)。stats を渡すと、重みなしの平均L1を stats['l1'] に入れる"""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    wl_sum, w_sum = 0.0, 0.0
    per_mode = {}
    steps = len(loader)
    for k, batch in enumerate(loader):
        if STOP['flag']:
            raise Stopped()
        image, tof, imu, state, mode, target = (t.to(device, non_blocking=True) for t in batch[:6])
        weight = batch[6].to(device, non_blocking=True).float() if len(batch) > 6 else None
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16,
                                                           enabled=amp and device.type == 'cuda'):
            pred = model(image, tof, imu, state, mode)
            per_sample = (pred.float() - target).abs().mean(dim=(1, 2))
            loss = per_sample.mean() if weight is None else (per_sample * weight).mean() / weight_scale
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if weight is None:
            total += loss.item() * image.shape[0]
        else:
            total += per_sample.detach().sum().item()
            wl_sum += (per_sample.detach() * weight).sum().item()
            w_sum += weight.sum().item()
        n += image.shape[0]
        for m, l in zip(mode.argmax(dim=1).tolist(), per_sample.detach().tolist()):
            s, c = per_mode.get(m, (0.0, 0))
            per_mode[m] = (s + l, c + 1)
        if k % 10 == 0 or k == steps - 1:
            emit('step', epoch=epoch, phase=phase, step=k + 1, steps=steps)
    if stats is not None:
        stats['l1'] = total / max(n, 1)
    mean = total / max(n, 1) if w_sum <= 0 else wl_sum / w_sum
    return mean, {m: s / c for m, (s, c) in per_mode.items()}


def expand_mode_state(state, model_state, copy_from=None):
    """学習済みの重み state を、ジャンル数が多いモデル(model_state)に合わせて広げる。

    ジャンルに依存するのは mode_embed(one-hot → 埋め込み)と heads(ジャンルごとに FUTURE_STEPS×2 行)だけ。
    元からあるジャンルはそのまま写し、増えたジャンルは copy_from {新ID: 元ID} の元のジャンルの値で初期化する
    (copy_from に無いものはモデルの初期値のまま)。ジャンル数が同じならそのまま返す"""
    copy_from = dict(copy_from or {})
    old_n = state['mode_embed.weight'].shape[1]
    new_n = model_state['mode_embed.weight'].shape[1]
    if old_n == new_n and not copy_from:
        return state
    if new_n < old_n:
        raise ValueError('ジャンル数が減っています(%d → %d)' % (old_n, new_n))
    for dst, src in copy_from.items():
        if not (old_n <= dst < new_n) or not (0 <= src < old_n):
            raise ValueError('--init-mode-copy %d:%d: 新しいジャンルは %d〜%d、元は 0〜%d' % (dst, src, old_n, new_n - 1, old_n - 1))
    out = dict(state)
    emb = model_state['mode_embed.weight'].clone()
    emb[:, :old_n] = state['mode_embed.weight']
    rows = FUTURE_STEPS * 2
    hw = model_state['heads.weight'].clone().view(new_n, rows, -1)
    hb = model_state['heads.bias'].clone().view(new_n, rows)
    hw[:old_n] = state['heads.weight'].view(old_n, rows, -1)
    hb[:old_n] = state['heads.bias'].view(old_n, rows)
    for dst, src in copy_from.items():
        emb[:, dst] = emb[:, src]
        hw[dst] = hw[src]
        hb[dst] = hb[src]
    out['mode_embed.weight'] = emb
    out['heads.weight'] = hw.view(new_n * rows, -1)
    out['heads.bias'] = hb.view(new_n * rows)
    return out


def parse_mode_copy(text):
    """'6:5,7:4' → {6: 5, 7: 4}"""
    out = {}
    for part in (text or '').split(','):
        if part.strip():
            dst, src = part.split(':')
            out[int(dst)] = int(src)
    return out


def args_diff(prev, now):
    """--resume のときの引数の違い {名前: (前回, 今回)}。新しい引数は既定値なら記録しないので、片方に無い = 既定値"""
    return {k: (prev.get(k), now.get(k)) for k in sorted(set(prev) | set(now)) if prev.get(k) != now.get(k)}


def file_digest(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


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
    ap.add_argument('--resume', action='store_true', help='--out の last.pth から続きを学習する(電源断のあとなど)')
    ap.add_argument('--init', default=None, help='学習済みの重み(best.pth など)から始める(--resume のときは last.pth を使う)')
    ap.add_argument('--init-mode-copy', default=None,
                    help='--init でジャンルが増えるとき、増えたジャンルを元のジャンルで初期化する(例 6:5,7:4)')
    ap.add_argument('--seed', type=int, default=None, help='乱数の種(既定: 決めない)')
    ap.add_argument('--drop-mask', default=None, help='フレームの表の npz(scene, frame, drop[, keep_impact])')
    ap.add_argument('--sample-weights-npz', default=None, help='フレームの表の npz(scene, frame, weight)。損失の重み')
    ap.add_argument('--mode-remap-npz', default=None, help='フレームの表の npz(scene, frame, mode)。学習するジャンルの付け替え(-1=そのまま)')
    ap.add_argument('--mode-remap-dropout', type=float, default=0.0,
                    help='学習中、付け替えたフレームを元のジャンルに戻す確率(検証では戻さない)')
    ap.add_argument('--progress-json', action='store_true')
    args = ap.parse_args()
    PROGRESS['enabled'] = args.progress_json
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.__setitem__('flag', True))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    use_command = not args.no_command_input
    mask_impacts = not args.keep_impacts
    rl = {}                                   # オフライン強化学習の方策の学習の設定(使ったときだけ)
    if args.drop_mask:
        with np.load(args.drop_mask) as z:
            has_keep = 'keep_impact' in z.files
        rl['drop'] = load_frame_table(args.drop_mask, 'drop')
        if has_keep:
            rl['keep_impact'] = load_frame_table(args.drop_mask, 'keep_impact')
    if args.sample_weights_npz:
        rl['weights'] = load_frame_table(args.sample_weights_npz, 'weight')
    if args.mode_remap_npz:
        rl['mode_remap'] = load_frame_table(args.mode_remap_npz, 'mode')
        from jetformer import modes as jmodes
        ids = sorted({int(v) for _, val in rl['mode_remap'].values() for v in np.unique(val) if v >= 0})
        extra = []
        for i in ids:
            m = jmodes.get(i)
            if m is None:
                raise SystemExit('--mode-remap-npz: ジャンル %d が data/modes.json にありません(先に登録してください)' % i)
            extra.append(m)
        rl['extra_modes'] = extra
    train_ds = JetFormerDataset(args.data, 'train', color_jitter=True, stride=args.stride, use_command=use_command,
                                mask_impacts=mask_impacts, impact_g=args.impact_g, remap_dropout=args.mode_remap_dropout,
                                return_weight='weights' in rl, **rl)
    val_ds = JetFormerDataset(args.data, 'val', stride=args.stride, use_command=use_command, mask_impacts=mask_impacts,
                              impact_g=args.impact_g, return_weight='weights' in rl, **rl)
    if mask_impacts:
        print('衝突の前後として外したフレーム: 学習 %d / 検証 %d' % (train_ds.impact_masked, val_ds.impact_masked))
    if rl:
        print('フレームの表: drop で外した 学習 %d / 検証 %d、衝突の前後の除外から戻した 学習 %d / 検証 %d' % (
            train_ds.dropped, val_ds.dropped, train_ds.impact_kept, val_ds.impact_kept))
    weight_scale = 1.0
    if 'weights' in rl and len(train_ds):
        w = train_ds.sample_weights()
        weight_scale = float(w.mean())
        print('損失の重み(学習): 平均 %.3f(これで割る)・中央値 %.3f・最大 %.2f・有効サンプル数 %.0f / %d' % (
            weight_scale, float(np.median(w)), float(w.max()), float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)), len(w)))
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
    start_epoch = 1
    last_path = os.path.join(args.out, 'last.pth')
    same_args = {k: getattr(args, k) for k in ('data', 'epochs', 'batch', 'lr', 'stride', 'no_command_input', 'keep_impacts',
                                               'impact_g', 'no_pretrained')}
    same_args.update(rl_args_used(args))     # 新しい引数は使ったときだけ記録する(前からの last.pth で再開できるように)
    if args.resume:
        if not os.path.exists(last_path):
            raise SystemExit('--resume: %s がありません(最初から学習してください)' % last_path)
        ck = torch.load(last_path, map_location=device, weights_only=False)
        diff = args_diff(ck['args'], same_args)
        if diff:
            raise SystemExit('--resume: 前回と引数が違います(前回, 今回): %s' % diff)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        best, best_epoch, best_modes, history = ck['best'], ck['best_epoch'], ck['best_modes'], ck['history']
        start_epoch = ck['epoch'] + 1
        print('再開: %d エポック目まで済み(最良 %d エポック目)。%d エポック目から続けます' % (
            ck['epoch'], best_epoch, start_epoch), flush=True)
        emit('resume', epoch=ck['epoch'])
    elif args.init:
        init_state = torch.load(args.init, map_location='cpu', weights_only=True)
        init_state = expand_mode_state(init_state, model.state_dict(), parse_mode_copy(args.init_mode_copy))
        model.load_state_dict(init_state)
        print('初期値: %s(ジャンル数 %d、増えたジャンルの初期化 %s)' % (
            args.init, num_modes, args.init_mode_copy or 'なし'), flush=True)
    state = 'done'
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr, _ = run_epoch(model, train_dl, device, optimizer, epoch=epoch, phase='train', weight_scale=weight_scale)
            va_stats = {}
            va, va_modes = (run_epoch(model, val_dl, device, epoch=epoch, phase='val', stats=va_stats) if len(val_ds)
                            else (tr, {}))
            scheduler.step()
            improved = va < best
            if improved:
                best, best_epoch, best_modes = va, epoch, va_modes
                torch.save(model.state_dict(), os.path.join(args.out, 'best.pth'))
            per_mode_us = {str(m): round(l * jm.PWM_SCALE, 1) for m, l in sorted(va_modes.items())}
            history.append({'epoch': epoch, 'train_us': round(tr * jm.PWM_SCALE, 1), 'val_us': round(va * jm.PWM_SCALE, 1),
                            'val_per_mode_us': per_mode_us, 'seconds': round(time.time() - t0, 1), 'best': improved})
            if 'weights' in rl and 'l1' in va_stats:    # 重み付きのときは、重みなしの検証 L1 も残す
                history[-1]['val_l1_us'] = round(va_stats['l1'] * jm.PWM_SCALE, 1)
            emit('epoch', **history[-1])
            tmp = last_path + '.tmp'                     # 途中保存(書き終えてから置き換える)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'best_modes': best_modes,
                        'history': history, 'args': same_args}, tmp)
            os.replace(tmp, last_path)
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
    if rl_args_used(args):                   # 使ったときだけ(既定の result.json は今までと同じ)
        result['rl_options'] = {
            'init': args.init, 'init_mode_copy': args.init_mode_copy, 'seed': args.seed,
            'drop_mask': args.drop_mask, 'sample_weights_npz': args.sample_weights_npz, 'mode_remap_npz': args.mode_remap_npz,
            'mode_remap_dropout': args.mode_remap_dropout, 'weight_scale': weight_scale,
            'dropped_frames': {'train': train_ds.dropped, 'val': val_ds.dropped},
            'impact_kept_frames': {'train': train_ds.impact_kept, 'val': val_ds.impact_kept},
            'sha1': {k: file_digest(getattr(args, k)) for k in ('init', 'drop_mask', 'sample_weights_npz', 'mode_remap_npz')
                     if getattr(args, k)}}
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
