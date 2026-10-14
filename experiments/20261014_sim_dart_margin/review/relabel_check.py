#!/usr/bin/env python3
"""確認の段(review、2026-10-14): D1 の正解の付け替えを、付け替えの道具を使わずに確かめる(読むだけ)

    python3 -I relabel_check.py [シーンの数(既定 30)]

使うもの: 取り込んだ D1 の aligned/*.npz(付け替えた後)、relabel/backup/ の NPZ(付け替える前)、
          D1 の simulation/scenes/<シーン>/frames.jsonl(ブラウザの記録。command_si = 実際に使った操作、action_info.dart.clean = 先生の乱す前の操作、noise)
確かめること:
  (0) 付け替える前の正解 = 記録の applied_pwm_us の 1〜10 先(取り込みの決まり)。PWM の式は記録から当てはめる(舵・前進それぞれ 1 次式)
  (1) 写し(backup)は future_actions・future_valid だけを持つ。future_valid は付け替えの前後で同じ
  (2) 次のフレーム t+1 が乱されていないフレーム: 最初の正解(k=0)= 付け替える前と同じ
  (3) 次のフレーム t+1 が乱されたフレーム: 最初の正解(k=0)= PWM(記録の clean[t+1])(先生が記録の状態で出した乱す前の操作)で、
      付け替える前(乱し入りの操作)とは違う
  (4) t+1〜t+10 のどれも乱されておらず、t も乱されていない(前回の行動に乱しが入らない)フレーム: 10 個の正解すべて = 付け替える前と同じ
出力: /home/jetson/jetformer_reports/sim_dart_20261014/review/relabel_check.json
"""
import glob
import json
import os
import sys

import numpy as np

NAME = 'jetformer-simulator-rl-dart-fast-10min_20261014-062216_4446c73d'
DS = f'/home/jetson/jetformer/data/datasets/{NAME}'
BK = f'/home/jetson/jetformer/experiments/20261014_sim_dart_margin/relabel/backup/{NAME}'
OUT = '/home/jetson/jetformer_reports/sim_dart_20261014/review/relabel_check.json'


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    scenes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{DS}/aligned/*.npz'))[:nmax]
    # PWM の式を記録から当てはめる
    cmd, pwm = [], []
    recs = {}
    for s in scenes:
        L = [json.loads(x) for x in open(f'{DS}/simulation/scenes/{s}/frames.jsonl', encoding='utf-8')]
        recs[s] = L
        for f in L:
            cmd.append([f['command_si']['steering'], f['command_si']['speed']])
            pwm.append(f['applied_pwm_us'])
    cmd, pwm = np.array(cmd), np.array(pwm)
    # 舵は中心から左右で傾きが違う(折れ線)。前進は 1 本の直線。それぞれ記録から当てはめる
    fit = {}
    for name, j, sel in [('steer_neg', 0, cmd[:, 0] < 0), ('steer_pos', 0, cmd[:, 0] >= 0), ('throttle', 1, np.ones(len(cmd), bool))]:
        A = np.c_[cmd[sel, j], np.ones(sel.sum())]
        coef, *_ = np.linalg.lstsq(A, pwm[sel, j], rcond=None)
        fit[name] = coef
        resid = np.abs(A @ coef - pwm[sel, j]).max()
        print(f'PWM の式 {name}: {coef[0]:.4f} × 値 + {coef[1]:.4f}(当てはめの差の最大 {resid:.2e} µs、{sel.sum()} フレーム)')

    def to_pwm(a):
        st = fit['steer_neg'] if a[0] < 0 else fit['steer_pos']
        return np.array([st[0] * a[0] + st[1], fit['throttle'][0] * a[1] + fit['throttle'][1]])

    tot = dict(scenes=0, frames=0, other_arrays_same=0, future_valid_same=0, old_equals_record=0, old_checked=0,
               next_clean_frames=0, next_clean_k0_same_as_old=0, next_noisy_frames=0, next_noisy_k0_eq_clean=0,
               next_noisy_k0_differs_from_old=0, quiet_window_frames=0, quiet_window_all_k_same=0, changed_frames=0)
    worst = dict(old_vs_record_us=0.0, k0_vs_clean_us=0.0, quiet_us=0.0, next_clean_k0_us=0.0)
    for s in scenes:
        new, old = np.load(f'{DS}/aligned/{s}.npz'), np.load(f'{BK}/{s}.npz')
        L = recs[s]
        N = len(L)
        tot['scenes'] += 1
        tot['frames'] += N
        # 写しには future_actions・future_valid・source_sha256 だけがある(ほかの配列は付け替えで触らない決まり)
        tot['other_arrays_same'] += int(set(old.files) == {'future_actions', 'future_valid', 'source_sha256'})
        tot['future_valid_same'] += int(np.array_equal(new['future_valid'], old['future_valid']))
        fa_new, fa_old = new['future_actions'].astype(np.float64), old['future_actions'].astype(np.float64)
        tot['changed_frames'] += int((np.abs(fa_new - fa_old).max(axis=(1, 2)) > 1e-3).sum())
        noisy = np.array([bool(((f.get('action_info') or {}).get('dart') or {}).get('noise')) for f in L])
        applied = np.array([f['applied_pwm_us'] for f in L], dtype=np.float64)
        clean = [((f.get('action_info') or {}).get('dart') or {}).get('clean') for f in L]
        for t in range(N - 11):
            # (0) 付け替える前 = 記録の 1〜10 先
            d0 = np.abs(fa_old[t] - applied[t + 1:t + 11]).max()
            worst['old_vs_record_us'] = max(worst['old_vs_record_us'], d0)
            tot['old_checked'] += 1
            tot['old_equals_record'] += int(d0 < 0.01)
            if not noisy[t + 1]:
                tot['next_clean_frames'] += 1
                d = np.abs(fa_new[t, 0] - fa_old[t, 0]).max()
                worst['next_clean_k0_us'] = max(worst['next_clean_k0_us'], d)
                tot['next_clean_k0_same_as_old'] += int(d < 0.01)
            else:
                tot['next_noisy_frames'] += 1
                d = np.abs(fa_new[t, 0] - to_pwm(clean[t + 1])).max()
                worst['k0_vs_clean_us'] = max(worst['k0_vs_clean_us'], d)
                tot['next_noisy_k0_eq_clean'] += int(d < 0.01)
                tot['next_noisy_k0_differs_from_old'] += int(np.abs(fa_new[t, 0] - fa_old[t, 0]).max() > 0.01)
            if not noisy[t:t + 11].any():
                tot['quiet_window_frames'] += 1
                d = np.abs(fa_new[t] - fa_old[t]).max()
                worst['quiet_us'] = max(worst['quiet_us'], d)
                tot['quiet_window_all_k_same'] += int(d < 0.01)
    res = {'dataset': NAME, 'pwm_fit': {k: list(map(float, c)) for k, c in fit.items()}, 'counts': tot,
           'worst_abs_diff_us': {k: float(v) for k, v in worst.items()},
           'note': 'float32 の NPZ なので 0.01 µs 未満を「同じ」とした。付け替えの道具(relabel/*)は使っていない'}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
