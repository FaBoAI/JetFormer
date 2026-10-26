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

ベンチマークの学習(2026-10-21。既定ではどれも使わず、今までと同じ学習になる):
    --label-override-npz <npz>   フレームの表(future_throttle_us [M,10] µs、NaN = そのまま)。学習の正解の 1〜10 フレーム先の
                                 スロットルを差し替える(区間の速度上限の出力に付け替えた正解など)。検証の正解は変えない
    --aug-brightness F / --aug-contrast F / --aug-shift-px N / --aug-tof-noise F
                                 学習の入力の拡張(カメラの明るさ ±F・コントラスト ±F・上下左右に ±N 画素のずれ、
                                 ToF の距離(正規化、1.0 = 4000 mm)に標準偏差 F の雑音)。0 = 使わない。検証には掛けない
    --aug-top-brightness F       学習の入力の拡張(2026-10-25 C1): 画像の上 1/3 の明るさ ±F(--aug-brightness と同じ単位。
                                 一番上の行で全部、上 1/3 の境まで直線で 0 に下がる。夕方と昼の光の差)。0 = 使わない

学習 v3(2026-10-26 段 data、protocol_v3。既定ではどれも使わず、今までと同じ学習・同じファイルになる):
    --action-ref-from-samples    作用の基準(model_config の steer_center_us・throttle_neutral_us)を、学習に使うサンプル
                                 (drop・衝突の前後・間引きのあと)の止まったフレームだけから出す。学習そのもの(損失・重み)は変えない
    --probe-npz <npz>            学習中の物差し(直線 10/10・10/23 のスロットルの B との差・止まった場面のブレーキ側・
                                 10/10 の低速のブレーキ側。experiments/20261026_fix/diag_data/probe_gates.py と同じ式)を
                                 1 エポック目・--probe-every エポックごと・最後に流して history[n]['probe'] に入れる。
                                 学習のあとの best.pth でも流して result.json の probe_best に入れる。見るだけ(損失・最良の
                                 エポックの選び方・データには使わない)。乱数を引かない(同じ種なら物差しなしと同じ学習)
    --probe-every N              物差しを流す間隔 [エポック](既定 5)

モデルの種類(方策。jetformer/policies の登録簿。既定は jetformer_v0 = 今までの JetFormer v0 で、今までと同じ学習・同じファイル):
    --policy NAME                jetformer_v0(既定)・act・tcp・msfusion・dinov2
    --policy-cfg JSON            方策の設定(既定値から変えるものだけ。例 '{"dec_layers": 3}')
    書き出す ONNX の入出力は方策が決め、model_config.json に policy・policy_cfg・inputs(ONNX から読み直した名前)などを書く
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
from jetformer import policies  # noqa: E402
from jetformer.model import TOF_CHANNELS  # noqa: E402
from jetformer.policies import runtime  # noqa: E402
from jetformer.train_data import PROBE_PARTS, JetFormerDataset, ProbeSet, load_frame_table  # noqa: E402


class Stopped(Exception):
    pass


STOP = {'flag': False}
PROGRESS = {'enabled': False}
# オフライン強化学習の方策用に足した引数と既定値(既定値なら使わない・記録しない)
RL_ARGS = {'init': None, 'init_mode_copy': None, 'seed': None, 'drop_mask': None, 'sample_weights_npz': None,
           'mode_remap_npz': None, 'mode_remap_dropout': 0.0}
# ベンチマークの学習用に足した引数と既定値(既定値なら使わない・記録しない → 前からの last.pth で --resume できる)
BENCH_ARGS = {'label_override_npz': None, 'aug_brightness': 0.0, 'aug_contrast': 0.0, 'aug_shift_px': 0, 'aug_tof_noise': 0.0,
              'aug_top_brightness': 0.0}
# モデルの種類(方策)の引数と既定値(既定値なら記録しない → 前からの last.pth で --resume できる)
POLICY_ARGS = {'policy': policies.DEFAULT, 'policy_cfg': None}
# 学習 v3 の引数と既定値(2026-10-26。既定値なら記録しない → 前からの last.pth で --resume できる)。
# 物差し(--probe-npz・--probe-every)は見るだけなので --resume の比べに入れない(途中から付け外ししてよい)
V3_ARGS = {'action_ref_from_samples': False}
PROBE_ARGS = {'probe_npz': None, 'probe_every': 5}


def v3_args_used(args):
    """既定値から変えた V3_ARGS {名前: 値}"""
    return {k: getattr(args, k, d) for k, d in V3_ARGS.items() if getattr(args, k, d) != d}


def probe_due(epoch, epochs, every):
    """物差しを流すエポックか: 1 エポック目・every の倍数・最後"""
    return epoch == 1 or epoch == epochs or (every > 0 and epoch % every == 0)


def run_probe(model, device, probe, num_modes, batch=64):
    """物差し(ProbeSet)のフレームでモデルの行 0〜2 のスロットルを出し、probe_gates.py と同じ欄を返す。
    FP32・評価の形(model.eval・勾配なし)。終わったら学習の形に戻す。乱数の状態(random・numpy・torch・CUDA)は
    前後で保存して戻す(DataLoader を使わないので引かないが、念のため)。学習の損失・最良の選び方には使わない"""
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
              torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
    was_training = model.training
    model.eval()
    if device.type == 'cuda':           # Jetson は GPU のメモリ = プロセスのメモリ(上限 3872M に数える)。学習のキャッシュを
        torch.cuda.empty_cache()        # 返してから流し、終わったら物差しの分も返す(10/26 段 data の試しで、返さないと
    try:                                # 書き出しのところでメモリの上限に当たった。計算の値は変わらない)
        pred = {}
        with torch.no_grad():
            for part in PROBE_PARTS:
                outs = []
                for image, tof, imu, state in probe.batches(part, batch):
                    oh = torch.nn.functional.one_hot(torch.full((len(image),), probe.genre, dtype=torch.int64),
                                                     num_modes).float()
                    x = dict(image=torch.from_numpy(image).to(device), tof=torch.from_numpy(tof).to(device),
                             imu=torch.from_numpy(imu).to(device), state=torch.from_numpy(state).to(device),
                             mode=oh.to(device))
                    outs.append(runtime.forward(model, x)['actions'][:, :3, 1].float().cpu().numpy())
                pred[part] = np.concatenate(outs).astype(np.float64) * jm.PWM_SCALE + jm.PWM_CENTER
    finally:
        model.train(was_training)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return probe.metrics(pred)


def probe_line(r):
    return 'st10 %+.1fµs  st23 %+.1fµs  止まった場面のブレーキ側 %.3f  前へ %.3f  10/10 の低速のブレーキ側 %.3f' % (
        r['st10_thr_vs_b_us'], r['st23_thr_vs_b_us'], r['stand_brake_side_frac'], r['stand_go_frac'], r['low10_brake_frac'])


def rl_args_used(args):
    """既定値から変えた RL_ARGS {名前: 値}"""
    return {k: getattr(args, k) for k, d in RL_ARGS.items() if getattr(args, k) != d}


def bench_args_used(args):
    """既定値から変えた BENCH_ARGS {名前: 値}"""
    return {k: getattr(args, k, d) for k, d in BENCH_ARGS.items() if getattr(args, k, d) != d}


def aug_of(args):
    """--aug-* → JetFormerDataset の aug(全部 0 なら None = 使わない)。top_brightness は使ったときだけ入れる
    (上 1/3 を使わない拡張の result.json・表示は 10/21 と同じ)"""
    aug = {'brightness': args.aug_brightness, 'contrast': args.aug_contrast, 'shift_px': args.aug_shift_px,
           'tof_noise': args.aug_tof_noise}
    top = getattr(args, 'aug_top_brightness', 0.0)
    if top != 0.0:
        aug['top_brightness'] = top
    for k, v in aug.items():
        if v < 0:
            raise SystemExit('--aug-%s は 0 以上にしてください' % k.replace('_', '-'))
    return aug if any(v > 0 for v in aug.values()) else None


def norm_policy_cfg(text):
    """--policy-cfg の JSON → 比べるための文字列(キーの順をそろえる)。None・'{}'・空 → None"""
    if text is None or not str(text).strip():
        return None
    try:
        cfg = json.loads(text)
    except ValueError as e:
        raise SystemExit('--policy-cfg が JSON ではありません: %s' % e)
    if not isinstance(cfg, dict):
        raise SystemExit('--policy-cfg は {"キー": 値} の形で指定してください')
    return json.dumps(cfg, sort_keys=True, ensure_ascii=False) if cfg else None


def policy_args_used(args):
    """既定値から変えた POLICY_ARGS {名前: 値}(policy_cfg はキーの順をそろえた JSON)"""
    now = {'policy': getattr(args, 'policy', policies.DEFAULT), 'policy_cfg': norm_policy_cfg(getattr(args, 'policy_cfg', None))}
    return {k: v for k, v in now.items() if v != POLICY_ARGS[k]}


def check_init_compat(src_cfg, policy, cfg, spec):
    """--init の重みの学習結果(model_config.json の dict。無ければ None = 確かめない)が、今の方策・構造と同じか。
    違えば SystemExit。train_only_keys(学習だけの設定)の違いは通す"""
    if src_cfg is None:
        return
    src_spec_name = policies.policy_of(src_cfg)
    if src_spec_name != policy:
        raise SystemExit('--init: 重みのモデルの種類が %s で、今の %s と違います' % (src_spec_name, policy))
    _, src_pcfg = policies.from_config(src_cfg)
    diff = {k: (src_pcfg.get(k), cfg.get(k)) for k in sorted(set(src_pcfg) | set(cfg))
            if k not in spec.train_only_keys and src_pcfg.get(k) != cfg.get(k)}
    if diff:
        raise SystemExit('--init: 重みの構造の設定が今と違います(重み, 今): %s' % diff)


def emit(kind, **kw):
    if PROGRESS['enabled']:
        print('@@' + json.dumps(dict(type=kind, **kw), ensure_ascii=False), flush=True)


def run_epoch(model, loader, device, optimizer=None, amp=True, epoch=0, phase='train', weight_scale=1.0, stats=None):
    """平均L1誤差と、ジャンル別の平均L1誤差 {mode_id: loss} を返す

    バッチに重み(7番目の値。--sample-weights-npz)があるとき: 学習の損失は mean(w/weight_scale · L1)、返す誤差は
    重み付きの平均 Σw·L1/Σw(学習・検証とも同じ定義)。stats を渡すと、重みなしの平均L1を stats['l1'] に入れる。
    バッチの最後が dict なら方策の追加の正解(extra_targets)。

    方策が policy_loss を持つとき(tcp など): 学習の損失は policy_loss(サンプルごと)、報告する誤差はどの方策でも
    actions の L1(v0 は今までと同じ数字)。学習のとき stats に 'loss'(学習の損失の平均)・'parts'(loss_parts の平均)"""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    wl_sum, w_sum = 0.0, 0.0
    loss_sum, parts_sum = 0.0, {}
    per_mode = {}
    steps = len(loader)
    for k, batch in enumerate(loader):
        if STOP['flag']:
            raise Stopped()
        image, tof, imu, state, mode, target = (t.to(device, non_blocking=True) for t in batch[:6])
        extras = ({key: v.to(device, non_blocking=True) for key, v in batch[-1].items()}
                  if isinstance(batch[-1], dict) else None)                       # 追加の正解(最後の dict)
        weight = (batch[6].to(device, non_blocking=True).float() if len(batch) > 6 and not isinstance(batch[6], dict)
                  else None)
        inputs = dict(image=image, tof=tof, imu=imu, state=state, mode=mode)
        with torch.set_grad_enabled(train), torch.autocast(device.type, dtype=torch.float16,
                                                           enabled=amp and device.type == 'cuda'):
            out = runtime.forward(model, inputs, target=target, train=train)
            per_sample = runtime.l1_per_sample(out['actions'], target)           # 報告する誤差(どの方策でも L1)
            l1_mean = per_sample.mean()
            if train and getattr(model, 'policy_loss', None) is not None:
                per = runtime.sample_loss(model, out, target, extras)            # 学習の損失(方策の損失)
            else:
                per = per_sample
            if weight is None:
                loss = l1_mean if per is per_sample else per.mean()
            else:
                loss = (per * weight).mean() / weight_scale
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * image.shape[0]
            for key, v in runtime.loss_parts(model).items() if per is not per_sample else ():
                parts_sum[key] = parts_sum.get(key, 0.0) + v * image.shape[0]
        if weight is None:
            total += l1_mean.item() * image.shape[0]
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
        if train:
            stats['loss'] = loss_sum / max(n, 1)
            if parts_sum:
                stats['parts'] = {key: v / max(n, 1) for key, v in parts_sum.items()}
    mean = total / max(n, 1) if w_sum <= 0 else wl_sum / w_sum
    return mean, {m: s / c for m, (s, c) in per_mode.items()}


def expand_mode_state(state, model_state, copy_from=None):
    """学習済みの重み state を、ジャンル数が多いモデル(model_state)に合わせて広げる(jetformer_v0 用)。

    ジャンルに依存するのは mode_embed(one-hot → 埋め込み)と heads(ジャンルごとに FUTURE_STEPS×2 行)だけ。
    元からあるジャンルはそのまま写し、増えたジャンルは copy_from {新ID: 元ID} の元のジャンルの値で初期化する
    (copy_from に無いものはモデルの初期値のまま)。ジャンル数が同じならそのまま返す。
    ほかの方策は runtime.expand_modes に方策の genre_keys を渡す"""
    return runtime.expand_modes(state, model_state, policies.get(policies.DEFAULT).genre_keys, copy_from)


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
    """ONNX の書き出し(共通の runtime.export_onnx)。戻り値は書いた ONNX の実際の入出力"""
    return runtime.export_onnx(model, path, num_modes)


def write_json(path, obj):
    """書き終えてから置き換える(電源断で半端なファイルを残さない)"""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


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
    ap.add_argument('--label-override-npz', default=None,
                    help='フレームの表の npz(scene, frame, future_throttle_us [M,10])。学習の正解のスロットルの差し替え(NaN=そのまま)')
    ap.add_argument('--aug-brightness', type=float, default=0.0, help='学習の拡張: カメラの明るさ ±この割合(0 = 使わない)')
    ap.add_argument('--aug-contrast', type=float, default=0.0, help='学習の拡張: カメラのコントラスト ±この割合(0 = 使わない)')
    ap.add_argument('--aug-shift-px', type=int, default=0, help='学習の拡張: 画像を上下左右に ±N 画素ずらす(0 = 使わない)')
    ap.add_argument('--aug-tof-noise', type=float, default=0.0,
                    help='学習の拡張: ToF の距離(正規化、1.0 = 4000 mm)に足す雑音の標準偏差(0 = 使わない)')
    ap.add_argument('--aug-top-brightness', type=float, default=0.0,
                    help='学習の拡張: 画像の上 1/3 の明るさ ±この割合(上ほど強く、境で 0。0 = 使わない)')
    ap.add_argument('--action-ref-from-samples', action='store_true',
                    help='作用の基準(ステアの中心・スロットルの中立)を学習に使うサンプルの止まったフレームだけから出す(既定: 学習のシーン全部)')
    ap.add_argument('--probe-npz', default=None,
                    help='学習中の物差しの npz(probe_gates.py make の probe_set.npz)。1 エポック目・--probe-every ごと・最後に流す(見るだけ)')
    ap.add_argument('--probe-every', type=int, default=5, help='物差しを流す間隔 [エポック](既定 5)')
    ap.add_argument('--policy', default=policies.DEFAULT, choices=policies.names(),
                    help='モデルの種類(既定 %s = 今までの JetFormer v0)' % policies.DEFAULT)
    ap.add_argument('--policy-cfg', default=None, help='方策の設定の JSON(既定値から変えるものだけ。例 \'{"dec_layers": 3}\')')
    ap.add_argument('--progress-json', action='store_true')
    args = ap.parse_args()
    PROGRESS['enabled'] = args.progress_json
    spec = policies.get(args.policy)
    try:
        pcfg = spec.resolve(json.loads(norm_policy_cfg(args.policy_cfg) or '{}'))
    except ValueError as e:
        raise SystemExit('--policy-cfg: %s' % e)
    if spec.requires_pretrained and args.no_pretrained:
        raise SystemExit('%s は事前学習の重みが必要です(--no-pretrained では学習できません)' % spec.title)
    ok, why = spec.ready()
    if not ok:
        emit('error', message=why)
        raise SystemExit(why)
    policy_mod = spec.module_obj()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.__setitem__('flag', True))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    t_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    if getattr(policy_mod, 'prepare', None) is not None:          # 方策の準備(CPU。済んでいれば何もしない)
        policy_mod.prepare(args.data, pcfg)
    extra_targets = None
    if getattr(policy_mod, 'extra_targets', None) is not None:    # 方策の追加の正解(tcp の通過点など)
        extra_targets = policy_mod.extra_targets(args.data, pcfg)

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
    bench = {}                                # ベンチマークの学習の設定(使ったときだけ。学習の分け方だけに渡す)
    if args.label_override_npz:
        bench['label_override'] = load_frame_table(args.label_override_npz, 'future_throttle_us')
    aug = aug_of(args)
    if aug is not None:
        bench['aug'] = aug
    train_ds = JetFormerDataset(args.data, 'train', color_jitter=True, stride=args.stride, use_command=use_command,
                                mask_impacts=mask_impacts, impact_g=args.impact_g, remap_dropout=args.mode_remap_dropout,
                                return_weight='weights' in rl, extra_targets=extra_targets, **rl, **bench)
    val_ds = JetFormerDataset(args.data, 'val', stride=args.stride, use_command=use_command, mask_impacts=mask_impacts,
                              impact_g=args.impact_g, return_weight='weights' in rl, extra_targets=extra_targets, **rl)
    if mask_impacts:
        print('衝突の前後として外したフレーム: 学習 %d / 検証 %d' % (train_ds.impact_masked, val_ds.impact_masked))
    if rl:
        print('フレームの表: drop で外した 学習 %d / 検証 %d、衝突の前後の除外から戻した 学習 %d / 検証 %d' % (
            train_ds.dropped, val_ds.dropped, train_ds.impact_kept, val_ds.impact_kept))
    if 'label_override' in bench:
        print('正解の差し替え(学習だけ): 行 %d(学習に使うフレームの行 %d)' % (
            train_ds.label_overridden, train_ds.label_overridden_used))
    if aug is not None:
        print('学習の拡張:', json.dumps(aug, ensure_ascii=False))
    weight_scale = 1.0
    if 'weights' in rl and len(train_ds):
        w = train_ds.sample_weights()
        weight_scale = float(w.mean())
        print('損失の重み(学習): 平均 %.3f(これで割る)・中央値 %.3f・最大 %.2f・有効サンプル数 %.0f / %d' % (
            weight_scale, float(np.median(w)), float(w.max()), float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)), len(w)))
    names = {m['id']: m['name'] for m in train_ds.modes}
    ref_samples = None                        # --action-ref-from-samples(--max-steps で抜き出す前の学習のサンプルから)
    if args.action_ref_from_samples:
        ref_samples = train_ds.action_reference_from_samples()
        print('作用の基準(学習のサンプルの止まったフレーム %d 行から): ステアの中心 %.1f・スロットルの中立 %.1f µs'
              '(学習のシーン全部なら %.1f・%.1f)' % ((ref_samples[2], ref_samples[0], ref_samples[1]) + train_ds.action_reference()))
    probe = None
    if args.probe_npz:
        if args.probe_every < 1:
            raise SystemExit('--probe-every は 1 以上にしてください')
        probe = ProbeSet(args.probe_npz, os.path.dirname(os.path.abspath(args.data.rstrip('/'))), use_command=use_command)
        print('学習中の物差し: %s(%s。1 エポック目・%d エポックごと・最後。見るだけ)' % (
            args.probe_npz, '・'.join('%s %d' % kv for kv in probe.counts().items()), args.probe_every))
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
    model = policies.build_policy(args.policy, num_modes, pcfg, pretrained=not args.no_pretrained).to(device)
    runtime.check_forward(model, num_modes, device)       # 学習の前に出力の形を確かめる(書き出しで失敗しないように)
    n_total, n_train = runtime.count_params(model)
    if args.policy != policies.DEFAULT:
        print('モデルの種類: %s(%s)パラメータ %.1fM(学習 %.1fM)設定 %s' % (
            spec.title, args.policy, n_total / 1e6, n_train / 1e6, json.dumps(pcfg, ensure_ascii=False)), flush=True)
    amp = bool(spec.hints.get('amp', True))
    optimizer = torch.optim.AdamW(runtime.param_groups(model, args.lr), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    best, best_epoch, best_modes = float('inf'), 0, {}
    history = []
    start_epoch = 1
    last_path = os.path.join(args.out, 'last.pth')
    same_args = {k: getattr(args, k) for k in ('data', 'epochs', 'batch', 'lr', 'stride', 'no_command_input', 'keep_impacts',
                                               'impact_g', 'no_pretrained')}
    same_args.update(rl_args_used(args))     # 新しい引数は使ったときだけ記録する(前からの last.pth で再開できるように)
    same_args.update(policy_args_used(args))
    same_args.update(bench_args_used(args))
    same_args.update(v3_args_used(args))
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
        src_cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.init)), 'model_config.json')
        src_cfg = None
        if os.path.exists(src_cfg_path):
            with open(src_cfg_path, encoding='utf-8') as f:
                src_cfg = json.load(f)
        check_init_compat(src_cfg, args.policy, pcfg, spec)
        init_state = runtime.load_state(args.init)
        if spec.genre_keys:
            init_state = runtime.expand_modes(init_state, model.state_dict(), spec.genre_keys,
                                              parse_mode_copy(args.init_mode_copy))
        elif args.init_mode_copy:
            raise SystemExit('--init-mode-copy: %s はジャンルの層を宣言していないので使えません' % spec.title)
        model.load_state_dict(init_state)
        print('初期値: %s(ジャンル数 %d、増えたジャンルの初期化 %s)' % (
            args.init, num_modes, args.init_mode_copy or 'なし'), flush=True)
    state = 'done'
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr_stats = {}
            tr, _ = run_epoch(model, train_dl, device, optimizer, amp=amp, epoch=epoch, phase='train',
                              weight_scale=weight_scale, stats=tr_stats)
            va_stats = {}
            va, va_modes = (run_epoch(model, val_dl, device, amp=amp, epoch=epoch, phase='val', stats=va_stats)
                            if len(val_ds) else (tr, {}))
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
            if getattr(model, 'policy_loss', None) is not None:     # 方策の損失(v0 の history は今までと同じ)
                history[-1]['train_loss'] = round(float(tr_stats.get('loss', float('nan'))), 5)
                if tr_stats.get('parts'):
                    history[-1]['train_parts'] = {k: round(float(v), 5) for k, v in tr_stats['parts'].items()}
            if probe is not None and probe_due(epoch, args.epochs, args.probe_every):     # 学習中の物差し(見るだけ)
                tp = time.time()
                history[-1]['probe'] = run_probe(model, device, probe, num_modes)
                history[-1]['probe_s'] = round(time.time() - tp, 1)
                print('probe %3d  %s  %.0fs' % (epoch, probe_line(history[-1]['probe']), history[-1]['probe_s']), flush=True)
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

    steer_center, throttle_neutral = train_ds.action_reference() if ref_samples is None else ref_samples[:2]
    train_mode_counts = {str(m): c for m, c in sorted(train_ds.mode_counts().items())}
    result = {'state': state, 'dataset': os.path.basename(args.data.rstrip('/')), 'stride': args.stride,
              'epochs_planned': args.epochs, 'epochs_done': len(history), 'batch': args.batch, 'lr': args.lr,
              'pretrained': not args.no_pretrained, 'train_samples': len(train_ds), 'val_samples': len(val_ds),
              'best_val_us': round(best * jm.PWM_SCALE, 1) if best_epoch else None, 'best_epoch': best_epoch,
              'best_val_per_mode_us': {str(m): round(l * jm.PWM_SCALE, 1) for m, l in sorted(best_modes.items())},
              'modes': train_ds.modes, 'train_mode_counts': train_mode_counts, 'command_input': use_command,
              'mask_impacts': mask_impacts, 'impact_g': args.impact_g,
              'impact_masked_frames': {'train': train_ds.impact_masked, 'val': val_ds.impact_masked},
              'steer_center_us': steer_center, 'throttle_neutral_us': throttle_neutral, 'history': history,
              'policy': args.policy, 'policy_cfg': pcfg, 'seed': args.seed,
              'params_m': {'total': round(n_total / 1e6, 2), 'trainable': round(n_train / 1e6, 2)},
              'peak_mem_mb': (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.type == 'cuda' else None)}
    if rl_args_used(args):                   # 使ったときだけ(既定の result.json は今までと同じ)
        result['rl_options'] = {
            'init': args.init, 'init_mode_copy': args.init_mode_copy, 'seed': args.seed,
            'drop_mask': args.drop_mask, 'sample_weights_npz': args.sample_weights_npz, 'mode_remap_npz': args.mode_remap_npz,
            'mode_remap_dropout': args.mode_remap_dropout, 'weight_scale': weight_scale,
            'dropped_frames': {'train': train_ds.dropped, 'val': val_ds.dropped},
            'impact_kept_frames': {'train': train_ds.impact_kept, 'val': val_ds.impact_kept},
            'sha1': {k: file_digest(getattr(args, k)) for k in ('init', 'drop_mask', 'sample_weights_npz', 'mode_remap_npz')
                     if getattr(args, k)}}
    if bench_args_used(args):                # 使ったときだけ(既定の result.json は今までと同じ)
        result['bench_options'] = {
            'label_override_npz': args.label_override_npz, 'aug': aug,
            'label_overridden_rows': {'all': train_ds.label_overridden, 'used': train_ds.label_overridden_used},
            'sha1': ({'label_override_npz': file_digest(args.label_override_npz)} if args.label_override_npz else {})}
    if ref_samples is not None:              # 使ったときだけ(既定の result.json は今までと同じ)
        sc_all, tn_all = train_ds.action_reference()
        result['v3_options'] = {'action_ref_from_samples': True, 'action_ref_rows': ref_samples[2],
                                'action_ref_scene_all_us': [sc_all, tn_all]}
    if probe is not None:                    # 使ったときだけ。物差しは見るだけ(best は検証の L1 のまま)
        done = [h for h in history if 'probe' in h]
        result['probe_options'] = {'probe_npz': args.probe_npz, 'sha1': file_digest(args.probe_npz), 'every': args.probe_every,
                                   'genre': probe.genre, 'frames': probe.counts(), 'epochs': [h['epoch'] for h in done]}
        result['probe_history'] = [dict(epoch=h['epoch'], **{k: v for k, v in h['probe'].items() if k != 'n'}) for h in done]
        if history and 'probe' in history[-1]:        # 最後のエポック = last.pth の重み(止めたときは無いことがある)
            result['probe_last'] = dict(history[-1]['probe'], epoch=history[-1]['epoch'])
    if best_epoch:
        model.load_state_dict(torch.load(os.path.join(args.out, 'best.pth'), map_location='cpu', weights_only=True))
        if probe is not None:                # best.pth の重みで(書き出しの前。書き出しはモデルを CPU に動かす)
            result['probe_best'] = dict(run_probe(model, device, probe, num_modes), epoch=best_epoch)
            print('probe best(%d エポック目)  %s' % (best_epoch, probe_line(result['probe_best'])), flush=True)
        io = export_onnx(model, os.path.join(args.out, 'jetformer.onnx'), num_modes)
        model_config = {'num_modes': num_modes, 'modes': train_ds.modes, 'future_steps': jm.FUTURE_STEPS,
                        'inputs': io['inputs'], 'tof_channels': TOF_CHANNELS,       # 入力は書き出した ONNX から読み直した名前
                        'pwm_center': jm.PWM_CENTER, 'pwm_scale': jm.PWM_SCALE, 'tof_max_mm': jm.TOF_MAX_MM,
                        'erpm_scale': jm.ERPM_SCALE, 'dataset': result['dataset'], 'stride': args.stride,
                        # 走行時: 学習データがあるジャンルだけ選べる。ゲインはこの中心・中立の周りに掛ける
                        'trained_modes': [int(m) for m in train_mode_counts], 'train_mode_counts': train_mode_counts,
                        'steer_center_us': steer_center, 'throttle_neutral_us': throttle_neutral,
                        'command_input': use_command, 'mask_impacts': mask_impacts,
                        # モデルの種類(方策)と入出力の約束(古い学習結果に無い欄は jetformer_v0 として読む)
                        'policy': args.policy, 'policy_cfg': pcfg, 'contract': policies.CONTRACT,
                        'input_shapes': io['input_shapes'], 'outputs': io['outputs'], 'output_shapes': io['output_shapes'],
                        'engine': {'precision': spec.hints.get('engine_precision', 'fp16'), 'workspace_gb': 1.0},
                        'params_m': result['params_m']}
        if getattr(policy_mod, 'finalize', None) is not None:      # 方策の書き足し(tcp の制御器の postprocess など)
            model_config.update(policy_mod.finalize(args.out, pcfg, args.data) or {})
        write_json(os.path.join(args.out, 'model_config.json'), model_config)
        print('保存: %s/best.pth, jetformer.onnx, model_config.json' % args.out, flush=True)
    else:
        result['state'] = 'stopped_no_model'
    result['duration_s'] = round(time.time() - t_start, 1)
    write_json(os.path.join(args.out, 'result.json'), result)
    emit('done', state=result['state'], best_val_us=result['best_val_us'], best_epoch=best_epoch,
         duration_s=result['duration_s'])


if __name__ == '__main__':
    main()
