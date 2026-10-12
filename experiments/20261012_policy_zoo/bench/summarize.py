# -*- coding: utf-8 -*-
"""results/*_infer.json・*_train.json → results/summary.json と、design_bench.md に貼る表(標準出力)

    python summarize.py
"""

import glob
import json
import os

from common import RESULTS, save_json

ORDER = ['modelB_engine', 'v0', 'act', 'act_dec2', 'act_dec4', 'act_d512', 'act_cvae', 'tcp', 'tcp_dec', 'msfusion',
         'msfusion_l2', 'msfusion_l4', 'msfusion_tf', 'msfusion_rd', 'dinov2_vit', 'dinov2_p2', 'dinov2_p1', 'dinov2_r168_p2', 'dinov2_p2_ft2']
TRAIN_SAMPLES, VAL_SAMPLES, EPOCHS = 5912, 1559, 30          # モデル B と同じ条件(clean_driving_20261011・間引き 3)
V0_EPOCH_S = 34.0                                            # モデル B の実測(train.log: 1エポック 34 秒)


def load(kind):
    out = {}
    for p in glob.glob(os.path.join(RESULTS, '*_%s.json' % kind)):
        with open(p) as f:
            r = json.load(f)
        out[os.path.basename(p)[:-len('_%s.json' % kind)]] = r
    return out


def main():
    inf, tr = load('infer'), load('train')
    rows = []
    v0_gpu_epoch = None
    if 'v0' in tr and tr['v0'].get('train_step_ms_median'):
        t = tr['v0']
        v0_gpu_epoch = TRAIN_SAMPLES / 64 * t['train_step_ms_median'] / 1000 + VAL_SAMPLES / 64 * t['eval_batch_ms_median'] / 1000
    for name in ORDER:
        r, t = inf.get(name, {}), tr.get(name, {})
        if not r and not t:
            continue
        lat = r.get('latency', {})
        acc = r.get('accuracy', {})
        a = acc.get('actions') or acc.get('tokens') or {}
        row = {'name': name,
               'params_m': r.get('export', {}).get('params_m', t.get('params_m')),
               'trainable_m': r.get('export', {}).get('trainable_params_m', t.get('trainable_params_m')),
               'onnx_nodes': r.get('export', {}).get('nodes'),
               'onnx_mb': r.get('export', {}).get('onnx_mb'),
               'engine_mb': r.get('build', {}).get('engine_mb'),
               'build_s': r.get('build', {}).get('seconds'),
               'exec_med': lat.get('trt_exec', {}).get('median_ms'), 'exec_p99': lat.get('trt_exec', {}).get('p99_ms'),
               'call_med': lat.get('trt_call', {}).get('median_ms'),
               'numpy_med': lat.get('trt_run_numpy', {}).get('median_ms'),
               'numpy_p99': lat.get('trt_run_numpy', {}).get('p99_ms'),
               'torch_fp16_med': lat.get('torch_fp16', {}).get('median_ms'),
               'act_mem_mb': r.get('memory', {}).get('engine_activation_mb'),
               'load_delta_mb': r.get('memory', {}).get('runner_load_delta_mb'),
               'trt_max_us': a.get('trt_max_us'), 'trt_rel_max': a.get('trt_rel_max'),
               'cos_min': a.get('cos_min'),
               'train_ms': t.get('train_step_ms_median'), 'train_sps': t.get('train_samples_per_s_gpu'),
               'eval_ms': t.get('eval_batch_ms_median'), 'train_peak_mb': t.get('peak_allocated_mb'),
               'train_oom': t.get('oom')}
        if t.get('train_step_ms_median') and v0_gpu_epoch:
            gpu_epoch = TRAIN_SAMPLES / 64 * t['train_step_ms_median'] / 1000 + VAL_SAMPLES / 64 * t['eval_batch_ms_median'] / 1000
            # 読み込みは 4 プロセスで約 530 枚/秒(loader_rate.py)なので、どの方策も GPU が律速。
            # v0 の実測 34 秒/エポックを、GPU の計算時間の比で伸び縮みさせる
            row['gpu_epoch_s'] = round(gpu_epoch, 1)
            row['epoch_s'] = round(V0_EPOCH_S * gpu_epoch / v0_gpu_epoch, 0)
            row['run30_min'] = round(row['epoch_s'] * EPOCHS / 60, 0)
        rows.append(row)
    save_json(os.path.join(RESULTS, 'summary.json'), rows)

    def f(v, d=2):
        if v is None:
            return '—'
        return ('%.' + str(d) + 'f') % v if isinstance(v, float) else str(v)
    print('| 名前 | パラメータ(M)/学習する | ONNX ノード | エンジン MB | 作成 秒 | TRT のみ 中央/p99 ms | __call__ ms | run_numpy 中央/p99 ms |'
          ' 活性 MB | TRT 誤差 最大 µs | 学習 ms/64 | 枚/秒(GPU) | ピーク MB | 1エポック 秒 / 30エポック 分(推定) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print('| %s | %s / %s | %s | %s | %s | %s / %s | %s | %s / %s | %s | %s | %s | %s | %s | %s |' % (
            r['name'], f(r['params_m'], 1), f(r['trainable_m'], 1), f(r['onnx_nodes']), f(r['engine_mb'], 0),
            f(r['build_s'], 0), f(r['exec_med']), f(r['exec_p99']), f(r['call_med']), f(r['numpy_med']), f(r['numpy_p99']),
            f(r['act_mem_mb'], 1), f(r['trt_max_us'], 1) if r['trt_max_us'] is not None else (
                'cos ≥ %.4f' % r['cos_min'] if r['cos_min'] is not None else '—'),
            f(r['train_ms'], 0), f(r['train_sps'], 0), f(r['train_peak_mb'], 0),
            '%s / %s' % (f(r.get('epoch_s'), 0), f(r.get('run30_min'), 0)) if r.get('epoch_s') else '—'))


if __name__ == '__main__':
    main()
