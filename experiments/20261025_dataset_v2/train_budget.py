#!/usr/bin/env python3
"""設計 B(時間をかけた学習)の時間とエポックの見込み(2026-10-25)。読むだけ。

numpy だけ(torch・GPU は使わない)。data/ は変えない。出力は 1 つだけ:
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/train_budget.json

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_dataset_v2/train_budget.py

入力:
  data/runs/*/result.json(学習の記録: エポックごとの秒・学習と検証の L1)
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/design_numbers.json(設計 A の学習の候補と保留の数。参考に写す)
  学習の大きさは設計 A の design_data.md の数(v2-H 7,684・v2-M 8,879・駐車 2,866)を plan() に書いた
  /home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json(SIZE の点)
  data/raw/20261024-*/*/AI_CMD.npz(走行中の推論の時間・遅れ。10/24 の AI 走行)

数えるもの:
  1. 1 エポックの秒 = a × 学習サンプル + b × 検証サンプル + c を、実車の画像の学習の記録(方策ごと)に当てる
     (1 エポック目は CUDA の準備が入るので外し、2 エポック目からの中央値)
  2. B と同じ作り方の学習(v0・30 エポック)の、最良のエポック・最後との差・学習と検証の L1 の開き
  3. 案の学習(データの大きさ × エポック × 方策)の 1 本の時間と、日程の合計
  4. 走行中の推論の時間(AI_CMD の infer_ms)と遅れ(latency_ms)の分布
"""

import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
RUNS = ROOT / 'data' / 'runs'
RAW = ROOT / 'data' / 'raw'
OUTDIR = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
OUT = OUTDIR / 'train_budget.json'
DESIGN_A = OUTDIR / 'design_numbers.json'
SIZE_EVAL = Path('/home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json')

BASE_TRAIN, BASE_VAL = 5912, 1559           # clean_driving_20261011(間引き 3・2.5G の前後を除く)
SIM_RUN = re.compile(r'^(simdart|simrl|simdiv_20261015-(SDIV|SMOD|A4))')   # シミュレーターだけの学習(画像が違う)
B_FAMILY = ('clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
            'realacc_20261015-v0-s2', 'realacc_20261015-v0-s3', 'clean_driving_20261011-nocmd-s3-20261011-024626')
OVERHEAD_DEFAULT_S = 200.0   # 1 本ごとの準備(読み込み・1 エポック目の遅さ・ONNX の書き出し)[秒]。実測(overhead_rows)で置き換える


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def epoch_table():
    rows = []
    for p in sorted(RUNS.glob('*/result.json')):
        r = load(p)
        name = p.parent.name
        if not r or not r.get('history') or not r.get('train_samples') or SIM_RUN.match(name):
            continue
        secs = [h['seconds'] for h in r['history'][1:] if h.get('seconds')]
        if len(secs) < 3:
            continue
        allsec = [h['seconds'] for h in r['history'] if h.get('seconds')]
        dur = r.get('duration_s')
        rows.append({'run': name, 'policy': r.get('policy') or 'jetformer_v0', 'train': int(r['train_samples']),
                     'val': int(r['val_samples']), 'sec_epoch': float(np.median(secs)),
                     'epochs': len(r['history']), 'duration_s': dur,
                     # 準備の秒 = 全体 − 中央値 × エポック数(再開した学習は全体が短く出るので、負・小さすぎるものは使わない)
                     'overhead_s': (round(dur - float(np.median(secs)) * len(allsec), 1) if dur else None)})
    return rows


def fit(rows):
    x = np.array([[q['train'], q['val'], 1.0] for q in rows], np.float64)
    y = np.array([q['sec_epoch'] for q in rows], np.float64)
    if len(rows) >= 4 and len({q['train'] for q in rows}) >= 3 and len({q['val'] for q in rows}) >= 2:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    else:                                    # 検証の数が 1 種類しかない → 検証は既定の速さ(登録簿の val_rate)で固定
        coef = None
    return coef


def fit_fixed_val(rows, val_rate):
    """検証の秒を登録簿の val_rate(枚/秒)で固定して、学習の a と定数 c を当てる"""
    x = np.array([[q['train'], 1.0] for q in rows], np.float64)
    y = np.array([q['sec_epoch'] - q['val'] / val_rate for q in rows], np.float64)
    (a, c), *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ np.array([a, c])
    return {'a_s_per_sample': float(a), 'b_s_per_val_sample': 1.0 / val_rate, 'c_s': float(c),
            'train_rate_per_s': round(1.0 / a, 1), 'resid_s_max': round(float(np.abs(resid).max()), 1), 'n': len(rows)}


def b_family(rows_by_name):
    out = []
    for name in B_FAMILY:
        r = load(RUNS / name / 'result.json')
        if not r:
            continue
        h = r['history']
        val = np.array([e['val_us'] for e in h], np.float64)
        tr = np.array([e['train_us'] for e in h], np.float64)
        best = int(np.argmin(val))
        within = int(np.argmax(val <= val[-1] + 0.5)) + 1          # 最後の値 + 0.5 µs に初めて入ったエポック
        out.append({'run': name, 'best_epoch': best + 1, 'best_val_us': float(val[best]), 'last_val_us': float(val[-1]),
                    'last_minus_best_us': round(float(val[-1] - val[best]), 2), 'epoch_within_0p5_of_last': within,
                    'train_last_us': float(tr[-1]), 'gap_last_us': round(float(val[-1] - tr[-1]), 1),
                    'val_ep10': float(val[9]), 'val_ep20': float(val[19]), 'val_ep25': float(val[24])})
    return out


def sizes_from_design_a():
    d = load(DESIGN_A)
    if not d:
        return None
    hs = d.get('holdout_split', {})
    pool = [v for k, v in hs.items() if k.startswith('train_pool')]
    sealed = [v for k, v in hs.items() if k.startswith('sealed')]

    def s(group, key):
        return int(sum(v['cand'][key]['samples'] for v in group if key in v.get('cand', {})))
    over = d.get('intervention_overlap', {}).get('fwd_samples_in_windows')
    return {'pool_human_new_track': s(pool, 'a_nc_new_track'), 'pool_ai_clean_laps': s(pool, 'c_ai_clean_laps'),
            'pool_ai_clean_sub9': s(pool, 'c_ai_clean_sub9'), 'pool_takeover_3s_drive': s(pool, 'b_takeover_3s_drive'),
            'sealed_human': s(sealed, 'a_nc_new_track'), 'sealed_ai_clean_laps': s(sealed, 'c_ai_clean_laps'),
            'sealed_ai_clean_sub9': s(sealed, 'c_ai_clean_sub9'), 'sealed_near_collision': s(sealed, 'd_ai_near_collision'),
            'intervention_window_overlap': over}


def drive_timing():
    lat, inf = [], []
    for f in sorted(glob.glob(str(RAW / '20261024-*' / '*' / 'AI_CMD.npz'))):
        try:
            with np.load(f, allow_pickle=False) as z:
                if 'latency_ms' not in z.files or 'infer_ms' not in z.files:
                    continue
                sel = z['select'].astype(bool) if 'select' in z.files else np.ones(len(z['latency_ms']), bool)
                hold = z['hold'].astype(bool) if 'hold' in z.files else np.zeros(len(sel), bool)
                m = sel & ~hold
                lat.append(np.asarray(z['latency_ms'], np.float64)[m])
                inf.append(np.asarray(z['infer_ms'], np.float64)[m])
        except Exception:                       # 壊れたシーン(電源断の書きかけ)は飛ばす
            continue
    if not lat:
        return None
    lat, inf = np.concatenate(lat), np.concatenate(inf)
    lat, inf = lat[np.isfinite(lat)], inf[np.isfinite(inf)]
    q = (50, 90, 99)
    return {'frames': int(len(inf)), 'infer_ms': dict(zip(('p50', 'p90', 'p99'), np.percentile(inf, q).round(2).tolist())),
            'latency_ms': dict(zip(('p50', 'p90', 'p99'), np.percentile(lat, q).round(1).tolist())),
            'note': '10/24 の AI 走行(CH3 が AI 側・hold でない)のフレーム。infer_ms は run_numpy(エンジン + 入出力の写し)の'
                    '壁時計。エンジン単体の engine_check(B 2.34 ms)より長いのは、走行中はカメラ・ToF・記録が同時に動くため'}


def overhead_of(rows):
    v = [q['overhead_s'] for q in rows if q.get('overhead_s') is not None and q['policy'] == 'jetformer_v0'
         and q['epochs'] >= 10 and 30 <= q['overhead_s'] <= 900]
    return (round(float(np.median(v)), 1), len(v)) if v else (OVERHEAD_DEFAULT_S, 0)


def plan(models, sizes, overhead_s):
    """案の学習 1 本の時間 [時間]。models = {方策: {'a','b','c'}}"""
    # 設計 A(design_data.md の 1 章・6 章)の学習の大きさ(間引き 3)。設計 A の数が変わったらここを直す
    data = {
        'B のデータ(今)': BASE_TRAIN,
        'v2-H(人だけ)': 7684,
        'v2-M(主: 人 + AI の速い周)': 8879,
        'v2-M+P(+ 駐車(1)〜(3))': 8879 + 2866,
        '参考: 今の 2 倍': BASE_TRAIN * 2,
    }
    epochs = (30, 60, 100)
    table = []
    for pol, m in models.items():
        for dname, n in data.items():
            for e in epochs:
                sec = m['a_s_per_sample'] * n + m['b_s_per_val_sample'] * BASE_VAL + m['c_s']
                table.append({'policy': pol, 'data': dname, 'train_samples': int(n), 'epochs': e,
                              'sec_per_epoch': round(sec, 1), 'hours': round((sec * e + overhead_s) / 3600.0, 2),
                              'steps': int(n // 64 * e)})
    return data, table


def main():
    rows = epoch_table()
    by_pol = {}
    for q in rows:
        by_pol.setdefault(q['policy'], []).append(q)
    val_rate = {'jetformer_v0': 650.0, 'msfusion': 430.0, 'act': 620.0, 'tcp': 600.0, 'dinov2': 300.0}
    models = {}
    for pol, rs in by_pol.items():
        if pol in val_rate and len(rs) >= 2 and len({q['train'] for q in rs}) >= 2:
            models[pol] = fit_fixed_val(rs, val_rate[pol])
        elif pol in val_rate and rs:
            q = rs[0]
            a = (q['sec_epoch'] - q['val'] / val_rate[pol]) / q['train']
            models[pol] = {'a_s_per_sample': a, 'b_s_per_val_sample': 1.0 / val_rate[pol], 'c_s': 0.0,
                           'train_rate_per_s': round(1.0 / a, 1), 'resid_s_max': None, 'n': 1}
    # 大きくした v0(dim 384・6 層)は学習の記録が無い → 10/12 の速さの試し(bench_train)の比で見積もる
    # act_d512(35.5M)= 442 ms/64、v0 = 310 ms/64(experiments/20261012_policy_zoo/bench/results)。v0-L(約 20M)は
    # その間の 1.3 倍とみる(仮定。動かす前に bench_train で測り直す)
    if 'jetformer_v0' in models:
        v0 = models['jetformer_v0']
        models['jetformer_v0 L(dim 384・6 層、仮定 1.3 倍)'] = {
            'a_s_per_sample': v0['a_s_per_sample'] * 1.3, 'b_s_per_val_sample': v0['b_s_per_val_sample'] * 1.3,
            'c_s': v0['c_s'], 'train_rate_per_s': round(v0['train_rate_per_s'] / 1.3, 1), 'resid_s_max': None,
            'n': 0}
    sizes = sizes_from_design_a()
    overhead_s, n_over = overhead_of(rows)
    data, table = plan(models, sizes, overhead_s)
    size_eval = (load(SIZE_EVAL) or {}).get('size', {})
    out = {'schema': 'jf-dataset-v2-train-budget/v1', 'tool': 'experiments/20261025_dataset_v2/train_budget.py',
           'epoch_rows': rows, 'epoch_models': models, 'b_family': b_family({}), 'sizes_design_a': sizes,
           'plan_data': data, 'plan_table': table, 'size_eval_slopes': size_eval.get('by_metric', {}).get('primary/best'),
           'drive_timing': drive_timing(), 'overhead_per_run_s': overhead_s, 'overhead_runs': n_over}
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print('1 本ごとの準備の秒(v0 の実測の中央値):', overhead_s, '・本数', n_over)
    for pol, m in models.items():
        print(pol, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()})
    for q in table:
        if q['epochs'] in (30, 60, 100):
            print('%-40s %-34s %6d %4d  %6.1f s/ep  %5.2f h  %6d steps' % (
                q['policy'][:40], q['data'], q['train_samples'], q['epochs'], q['sec_per_epoch'], q['hours'], q['steps']))
    print('書いた:', OUT)


if __name__ == '__main__':
    sys.exit(main())
