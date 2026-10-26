#!/usr/bin/env python3
"""設計 B(時間をかけた学習)の決まりのための「種の違いの幅」(2026-10-25)。読むだけ。

numpy だけ(torch・GPU は使わない)。data/ は変えない。保存してある予測(10/12・10/15・10/22)を読み直すだけ。
出力は 1 つだけ:
  /home/jetson/jetformer_reports/dataset_v2_plan_20261025/train_seed_spread.json

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_dataset_v2/train_seed_spread.py

入力:
  /home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz
      予測の表(key 3d339512…、10,985 フレーム)の正解 y・primary(保留 4,670)・interv(介入 1,570)・r1_pre1s・zone・erpm
  data/zoo/policy_zoo_20261012/pred/*.npz・data/zoo/real_accuracy_20261015/pred/*.npz(同じ表の予測 [10985,10,2] µs)

数えるもの(どれも desk_lib と同じ定義。行 = 10 フレーム先までの何番目か):
  l1            primary の 10 行 × 2 の |予測 − 人| の平均 [µs]
  l1_zone_<区間> primary のうち zone(make_bench_tables と同じ区間の判定)ごとの L1
  l1_interv     interv の L1
  steer_bias    primary の 行 0 の 予測ステア − 人のステア の平均 [µs](+ = 左寄り。ステアは中心より大きい = 左)
  thr_bias      primary の 行 0 の 予測スロットル − 人のスロットル の平均 [µs](+ = 人より緩める)
  r1_app        primary の R1 の起点の手前 1 秒(r1_pre1s)の 行 2 のスロットルの差(desk の r1_app_thr_diff_us と同じ)
  de_thr        primary の zone == DE の 行 2 のスロットルの差(参考)
  amp           primary の予測ステアの標準偏差 ÷ 人のステアの標準偏差(10 行まとめて)
家族ごとに 値の幅(最大 − 最小)・平均を出し、「閾値 = 2 × 幅」(10/13 の judge_primary と同じ倍率)を並べる。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
OUT = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025/train_seed_spread.json')
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
ZOO = ROOT / 'data' / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = ROOT / 'data' / 'zoo' / 'real_accuracy_20261015' / 'pred'
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ROW_LOOKAHEAD = 2

# 家族: 名前 → [(id, best の npz, last の npz)]
FAMILIES = {
    'v0(B と同じ作り方)': [
        ('A_s0', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
        ('A_s1', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
        ('v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
        ('v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
        ('B', ZOO / 'B.npz', ZOO / 'B_last.npz'),
    ],
    '多段型(msfusion)': [
        ('msf_s0', ZOO / 'msfusion_s0.npz', ZOO / 'msfusion_s0_last.npz'),
        ('msf_s1', ZOO / 'msfusion_s1.npz', ZOO / 'msfusion_s1_last.npz'),
        ('msf-s2', RACC / 'realacc_20261015-msf-s2.npz', RACC / 'realacc_20261015-msf-s2_last.npz'),
        ('msf-s3', RACC / 'realacc_20261015-msf-s3.npz', RACC / 'realacc_20261015-msf-s3_last.npz'),
    ],
    '位相を回す(ROT)': [
        ('rot-s0', RACC / 'realacc_20261015-rot-s0.npz', RACC / 'realacc_20261015-rot-s0_last.npz'),
        ('rot-s1', RACC / 'realacc_20261015-rot-s1.npz', RACC / 'realacc_20261015-rot-s1_last.npz'),
    ],
    'データ半分(SIZE H)': [
        ('sizeH-s0', RACC / 'realacc_20261015-sizeH-s0.npz', RACC / 'realacc_20261015-sizeH-s0_last.npz'),
        ('sizeH-s1', RACC / 'realacc_20261015-sizeH-s1.npz', RACC / 'realacc_20261015-sizeH-s1_last.npz'),
    ],
    'データ半分・エポック 2 倍(SIZE-EQ H)': [
        ('sizeHeq-s0', RACC / 'realacc_20261015-sizeHeq-s0.npz', RACC / 'realacc_20261015-sizeHeq-s0_last.npz'),
    ],
}
# 2 本の平均(同じ種の v0 + 多段型。予測の平均で作る = ens_average と同じ)
ENS_PAIRS = [('ens_s0', 'A_s0', 'msf_s0'), ('ens_s1', 'A_s1', 'msf_s1'), ('ens_s2', 'v0-s2', 'msf-s2'),
             ('ens_s3', 'v0-s3', 'msf-s3')]
ZONES = ('R1_app', 'R1_turn', 'GM', 'DE', 'none', 'unknown')


def load_pred(path):
    with np.load(path, allow_pickle=False) as z:
        key = str(z['key']) if 'key' in z.files else None
        if key != TABLE_KEY:
            raise SystemExit('%s: 予測の表の key が違います(%s)' % (path, key))
        return np.asarray(z['pred'], np.float64)


def metrics(pred, b):
    y = np.nan_to_num(b['y'])
    prim, interv = b['primary'], b['interv']
    d = np.abs(pred - y)
    out = {'l1': float(d[prim].mean()), 'l1_steer': float(d[prim][:, :, 0].mean()),
           'l1_thr': float(d[prim][:, :, 1].mean()), 'l1_interv': float(d[interv].mean())}
    for zname in ZONES:
        m = prim & (b['zone'] == zname)
        out['l1_zone_' + zname] = float(d[m].mean()) if m.any() else None
    out['steer_bias'] = float((pred[prim][:, 0, 0] - y[prim][:, 0, 0]).mean())
    out['thr_bias'] = float((pred[prim][:, 0, 1] - y[prim][:, 0, 1]).mean())
    r1 = prim & b['r1_pre1s']
    out['r1_app'] = float((pred[r1][:, ROW_LOOKAHEAD, 1] - y[r1][:, ROW_LOOKAHEAD, 1]).mean())
    de = prim & (b['zone'] == 'DE')
    out['de_thr'] = float((pred[de][:, ROW_LOOKAHEAD, 1] - y[de][:, ROW_LOOKAHEAD, 1]).mean())
    out['amp'] = float(np.std(pred[prim][:, :, 0]) / np.std(y[prim][:, :, 0]))
    return out


def spread(rows, keys):
    res = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if len(v) >= 2:
            res[k] = {'min': round(min(v), 3), 'max': round(max(v), 3), 'mean': round(float(np.mean(v)), 3),
                      'range': round(max(v) - min(v), 3), 'thr_2x_range': round(2 * (max(v) - min(v)), 3), 'n': len(v)}
        elif v:
            res[k] = {'value': round(v[0], 3), 'n': 1}
    return res


def main():
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        b = {k: z[k] for k in ('y', 'primary', 'interv', 'r1_pre1s', 'zone', 'erpm')}
    keys = None
    per_run, preds = {}, {}
    fam_out = {}
    for fam, members in FAMILIES.items():
        rows_best, rows_last = [], []
        for rid, pb, pl in members:
            p_best = load_pred(pb)
            preds[rid] = p_best
            mb = metrics(p_best, b)
            ml = metrics(load_pred(pl), b) if pl.exists() else None
            per_run[rid] = {'family': fam, 'best': {k: (round(v, 3) if v is not None else None) for k, v in mb.items()},
                            'last': ({k: (round(v, 3) if v is not None else None) for k, v in ml.items()} if ml else None)}
            rows_best.append(mb)
            if ml:
                rows_last.append(ml)
            keys = list(mb)
        fam_out[fam] = {'members': [m[0] for m in members], 'best': spread(rows_best, keys),
                        'last': spread(rows_last, keys)}
    ens_rows = []
    for eid, a, m in ENS_PAIRS:
        mb = metrics((preds[a] + preds[m]) / 2.0, b)
        per_run[eid] = {'family': '2 本の平均(v0 + 多段型・同じ種)', 'best': {k: (round(v, 3) if v is not None else None)
                                                                     for k, v in mb.items()}, 'last': None}
        ens_rows.append(mb)
    fam_out['2 本の平均(v0 + 多段型・同じ種)'] = {'members': [e[0] for e in ENS_PAIRS], 'best': spread(ens_rows, keys)}
    # 種どうしの予測の食い違い(primary の |差| の平均)
    v0ids = ['A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B']
    prim = b['primary']
    dis = {}
    for i in range(len(v0ids)):
        for j in range(i + 1, len(v0ids)):
            dis['%s|%s' % (v0ids[i], v0ids[j])] = round(float(np.abs(preds[v0ids[i]][prim] - preds[v0ids[j]][prim]).mean()), 2)
    counts = {'primary': int(prim.sum()), 'interv': int(b['interv'].sum()), 'r1_pre1s': int((prim & b['r1_pre1s']).sum()),
              'zones': {zname: int((prim & (b['zone'] == zname)).sum()) for zname in ZONES}}
    out = {'schema': 'jf-dataset-v2-train-seed-spread/v1', 'tool': 'experiments/20261025_dataset_v2/train_seed_spread.py',
           'table_key': TABLE_KEY, 'counts': counts, 'families': fam_out, 'per_run': per_run,
           'v0_pair_disagreement_us': dis,
           'note': '閾値 = 2 × 幅(10/13 の judge_primary の spread_mult 2.0)。幅は B と同じ作り方の 5 本(v0 家族)を使う。'
                   'ENS は best の予測の平均だけ(last は作っていない)'}
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    v0 = fam_out['v0(B と同じ作り方)']['best']
    print('v0 家族(best):', {k: v0[k]['range'] for k in v0})
    print('書いた:', OUT)


if __name__ == '__main__':
    sys.exit(main())
