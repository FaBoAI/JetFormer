#!/usr/bin/env python3
"""最終の確かめ(10/15 のレポートの段): Q2 の主指標(実機の保留データの L1)を、保存した予測の npz から、評価の段の道具を使わずに計算し直す。

    python3 -I recount_real_indep.py

q2_evaluate.py・q2_recount.py・rl_evaluate.py・jetformer のライブラリは import しない(numpy だけ)。
  1. 予測の表(part・scene・frame の順)を splits.json と aligned から自分で作り、sha1 の鍵が npz の key と同じことを確かめる
     (part 0 = ai_mix_20261011 の val の全シーンの vehicle_valid、part 1 = intervention_val_20261011 の val の future_valid & vehicle_valid)
  2. 主指標のフレーム = part 0 のうち clean_driving_20261011 の val(人の 11 シーン)& future_valid & vehicle_valid & 2.5 G の衝突の
     1.0 秒前〜2.0 秒後でない。正解は clean_driving_20261011 の aligned から読む(q2_recount.py は ai_mix 側を読んだ。別の道で同じになるかを見る)
  3. L1 = 10 ステップ × [ステア, スロットル] の |予測 − 正解| の平均のフレーム平均 [µs]。C-DIV の 2 つの種(best・last)と、比べる相手の A・C
  4. 判定(protocol eval_Q2.primary の文)をここで書き直して当てはめる
  5. 漏れ: 主指標の 11 シーンが C-DIV・A・C の学習(train)に 1 つも入っていない
出力: /home/jetson/jetformer_reports/sim_diverse_train_20261015/review/final/recount_real_indep.json
"""
import hashlib
import json
import os

import numpy as np

DS = '/home/jetson/jetformer/data/datasets'
ZOO = '/home/jetson/jetformer/data/zoo'
REP = '/home/jetson/jetformer_reports/sim_diverse_train_20261015'
OUT = os.path.join(REP, 'review/final/recount_real_indep.json')
PRED = {'A_s0': 'policy_zoo_20261012/pred/jetformer_v0_s0', 'A_s1': 'policy_zoo_20261012/pred/jetformer_v0_s1',
        'C_s0': 'sim_cotrain_20261013/pred/C_s0', 'C_s1': 'sim_cotrain_20261013/pred/C_s1',
        'CDIV_s0': 'sim_diverse_train_20261015/pred/CDIV_s0', 'CDIV_s1': 'sim_diverse_train_20261015/pred/CDIV_s1'}
G = 9.80665
THR = {'A': {'best': 0.626, 'last': 1.322}, 'C': {'best': 0.954, 'last': 1.322}}   # protocol で固定した閾値


def splits(name):
    with open(os.path.join(DS, name, 'splits.json')) as f:
        return json.load(f)


def aligned(name, scene, keys):
    with np.load(os.path.join(DS, name, 'aligned', scene + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def impact(ts_us, imu, g=2.5, before=1.0, after=2.0):
    t = ts_us.astype(np.float64) / 1e6
    hg = np.hypot(imu[:, :, 0].astype(np.float32), imu[:, :, 1].astype(np.float32)).max(axis=1) / G
    m = np.zeros(len(t), bool)
    for ti in t[hg >= g]:
        m |= (t >= ti - before) & (t <= ti + after)
    return m


def table():
    rows = []
    for part, name in ((0, 'ai_mix_20261011'), (1, 'intervention_val_20261011')):
        for s in splits(name)['val']:
            a = aligned(name, s, ['vehicle_valid', 'future_valid'])
            ok = a['vehicle_valid'].astype(bool) & (a['future_valid'].astype(bool) if part == 1 else True)
            rows += [(part, s, int(i)) for i in np.nonzero(ok)[0]]
    key = hashlib.sha1('|'.join('%d:%s:%d' % r for r in rows).encode()).hexdigest()
    return rows, key


def judge(arm, ref, thr):
    arm, ref = np.asarray(arm), np.asarray(ref)
    d = arm.mean() - ref.mean()
    if arm.max() < ref.min() and -d > thr:
        return 'better', d
    if arm.min() > ref.max() and d > thr:
        return 'worse', d
    return 'equivalent', d


def main():
    rows, key = table()
    human = set(splits('clean_driving_20261011')['val'])
    idx, y = [], []
    same_labels = True
    cache = {}
    for i, (part, s, fr) in enumerate(rows):
        if part != 0 or s not in human:
            continue
        if s not in cache:
            a = aligned('clean_driving_20261011', s, ['timestamps_us', 'imu', 'future_actions', 'future_valid', 'vehicle_valid'])
            b = aligned('ai_mix_20261011', s, ['future_actions'])
            same_labels &= bool(np.array_equal(a['future_actions'], b['future_actions']))
            a['imp'] = impact(a['timestamps_us'], a['imu'])
            cache[s] = a
        a = cache[s]
        if a['future_valid'][fr] and a['vehicle_valid'][fr] and not a['imp'][fr]:
            idx.append(i)
            y.append(a['future_actions'][fr])
    idx, y = np.array(idx), np.array(y, np.float64)
    L = {'best': {}, 'last': {}}
    keys_ok = {}
    for name, p in PRED.items():
        for ck, suf in (('best', ''), ('last', '_last')):
            with np.load(os.path.join(ZOO, p + suf + '.npz'), allow_pickle=False) as z:
                keys_ok['%s_%s' % (name, ck)] = str(z['key']) == key
                pr = z['pred'][idx].astype(np.float64)
            L[ck][name] = float(np.abs(pr - y).mean(axis=(1, 2)).mean())
    ev = json.load(open(os.path.join(REP, 'eval_real.json'), encoding='utf-8'))
    diff = {ck: {n: round(L[ck][n] - ev['primary']['L_us'][ck][n], 6) for n in L[ck]} for ck in L}
    ver = {}
    for ref in ('A', 'C'):
        r = {}
        for ck in ('best', 'last'):
            cls, d = judge([L[ck]['CDIV_s0'], L[ck]['CDIV_s1']], [L[ck][ref + '_s0'], L[ck][ref + '_s1']], THR[ref][ck])
            r[ck] = {'class': cls, 'diff_us': round(d, 4), 'threshold_us': THR[ref][ck],
                     'order_margin_us': round(min(L[ck]['CDIV_s0'], L[ck]['CDIV_s1']) - max(L[ck][ref + '_s0'], L[ck][ref + '_s1']), 4)}
        r['final'] = ('better' if r['last']['class'] == 'better' else 'equivalent') if r['best']['class'] == 'better' else r['best']['class']
        ver['CDIV_vs_' + ref] = r
    # 漏れ: 主指標の 11 シーンは学習に入っていないか
    trains = {
        'C-DIV': splits('/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/data/C-DIV')['train'],
        'A(clean_driving_20261011)': splits('clean_driving_20261011')['train'],
        'C(simreal_mix_20261013)': splits('simreal_mix_20261013')['train'],
    }
    leak = {k: sorted(human & set(v)) for k, v in trains.items()}
    res = {'schema': 'sim-diverse-train-final-recount-real/v1', 'pred_table_rows': len(rows), 'pred_table_key': key,
           'npz_keys_match': keys_ok, 'all_npz_keys_match': all(keys_ok.values()), 'primary_frames': int(len(idx)),
           'human_scenes': sorted(human), 'labels_clean_equal_ai_mix': same_labels,
           'L_us': {ck: {n: round(v, 4) for n, v in L[ck].items()} for ck in L},
           'diff_vs_eval_real_us': diff, 'max_abs_diff_vs_eval_real_us': max(abs(v) for ck in diff for v in diff[ck].values()),
           'CDIV_mean_us': {ck: round((L[ck]['CDIV_s0'] + L[ck]['CDIV_s1']) / 2, 4) for ck in L},
           'A_mean_us': {ck: round((L[ck]['A_s0'] + L[ck]['A_s1']) / 2, 4) for ck in L},
           'C_mean_us': {ck: round((L[ck]['C_s0'] + L[ck]['C_s1']) / 2, 4) for ck in L},
           'verdicts': ver, 'verdicts_eval_real': {'CDIV_vs_A': ev['primary']['CDIV_vs_A']['final_class'],
                                                  'CDIV_vs_C': ev['primary']['CDIV_vs_C']['final_class']},
           'leak_human_val_in_train': leak, 'no_leak': all(len(v) == 0 for v in leak.values())}
    res['verdicts_same'] = res['verdicts']['CDIV_vs_A']['final'] == res['verdicts_eval_real']['CDIV_vs_A'] and \
        res['verdicts']['CDIV_vs_C']['final'] == res['verdicts_eval_real']['CDIV_vs_C']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(OUT + '.tmp', OUT)
    print(json.dumps({k: res[k] for k in ('pred_table_rows', 'all_npz_keys_match', 'primary_frames', 'labels_clean_equal_ai_mix', 'L_us',
                                          'max_abs_diff_vs_eval_real_us', 'CDIV_mean_us', 'A_mean_us', 'C_mean_us', 'verdicts',
                                          'verdicts_eval_real', 'verdicts_same', 'leak_human_val_in_train', 'no_leak')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
