#!/usr/bin/env python3
"""TCP 型の正解(1 秒先までの通過点・目標速度)と制御器の較正を作る(CPU・numpy だけ。torch・GPU は使わない)

    nice python tools/make_tcp_labels.py data/datasets/clean_driving_20261011            # 無い・古ければ作る
    nice python tools/make_tcp_labels.py data/datasets/clean_driving_20261011 --force    # 作り直す
    nice python tools/make_tcp_labels.py data/datasets/clean_driving_20261011 --json     # 概要を JSON で

書くもの(データセットのフォルダーの中には書かない。ダウンロード・エクスポートに入るため):
    data/derived/<データセット名>/tcp_labels_v1.npz   scene・frame・waypoints [N,10,2]・v_tgt・v_now・wp_valid・version・splits_sha1
    data/derived/<データセット名>/tcp_calib_v1.json   ステア → 曲率の表(速度帯ごと)・スロットルの前向きの写像(学習のシーンだけ)

tools/train.py --policy tcp は学習の前に同じものを作る(tcp.prepare)ので、このコマンドは確かめ・作り直し用。
作り方は jetformer/policies/tcp_labels.py(spec.md §3.2.2・§3.2.3)。
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from jetformer.policies import tcp_labels  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description='TCP 型の正解(通過点・目標速度)と制御器の較正を作る')
    ap.add_argument('data', help='データセットのフォルダー(aligned/・can_bus/・splits.json)')
    ap.add_argument('--force', action='store_true', help='あっても作り直す')
    ap.add_argument('--wp-steps', type=int, default=tcp_labels.WP_STEPS, help='通過点の数(既定 10)')
    ap.add_argument('--wp-dt', type=float, default=tcp_labels.WP_DT, help='通過点の間隔 [秒](既定 0.1)')
    ap.add_argument('--json', action='store_true', help='概要を JSON で出す')
    args = ap.parse_args(argv)
    if not tcp_labels.has_can_bus(args.data):
        raise SystemExit('TCP 型には can_bus の IMU・車両の記録が要ります(%s/can_bus がありません)' % args.data)
    t0 = time.time()
    try:
        path, built = tcp_labels.ensure_labels(args.data, args.wp_steps, args.wp_dt, force=args.force)
        cpath, cbuilt = tcp_labels.ensure_calibration(args.data, force=args.force)
    except tcp_labels.LabelError as e:
        raise SystemExit(str(e))
    summary = json.loads(str(tcp_labels.load_labels(path)['summary']))
    calib = tcp_labels.load_calibration(args.data)
    st, ff = calib['steer_table'], calib['throttle_ff']
    out = {'labels': str(path), 'labels_built': built, 'calibration': str(cpath), 'calibration_built': cbuilt,
           'seconds': round(time.time() - t0, 1), 'summary': summary,
           'calibration_headline': {'lag_ms': st['lag_ms'], 'center_us': st['center_us'], 'r2_by_lag': st['r2_by_lag'],
                                    'slope_left_right_per_us': st['slope_left_right_per_us'],
                                    'full_lock_curv_by_band(right, left)': [[r[0], r[-1]] for r in st['curv']],
                                    'throttle_ff': ff}}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return out
    print('正解: %s(%s)' % (path, '作りました' if built else '済み'))
    print('  フレーム %d・通過点の有効 %d(%.1f%%)・無効の理由 %s' % (
        summary['frames'], summary['valid_frames'], 100 * summary['valid_frac'], summary['invalid_reasons']))
    for k, v in summary.items():
        if k.startswith('valid_frac_by_mode'):
            print('  ジャンル別: %s' % json.dumps(v, ensure_ascii=False))
    print('  目標速度 p10/50/90/99: %s m/s' % summary['v_tgt_p10_50_90_99(valid)'])
    print('  符号の確認: %s' % json.dumps(summary['sign_checks'], ensure_ascii=False))
    print('較正: %s(%s)' % (cpath, '作りました' if cbuilt else '済み'))
    print('  ステア → 曲率: 遅れ %d ms(R² %s)・中心 %.1f µs・小さい切れ角の傾き 左 %.5f / 右 %.5f [1/m/µs]' % (
        st['lag_ms'], st['r2_by_lag'], st['center_us'], *st['slope_left_right_per_us']))
    for b, r in zip(['v < %.0f' % st['bands_mps'][0], '%.0f ≤ v < %.0f' % tuple(st['bands_mps'][:2]),
                     '%.0f ≤ v' % st['bands_mps'][-1]], st['curv']):
        print('    %-10s m/s: いっぱいに切った曲率 右 %.2f・左 %.2f [1/m]' % (b, r[0], r[-1]))
    print('  スロットル: thr = %.1f %+.1f·v %+.1f·(v_tgt − v)/%.2f(学習の誤差 %.1f µs・前進の上限 %.0f µs・中立 %.1f µs・%d サンプル)' % (
        ff['c0'], ff['c1'], ff['c2'], ff['tau_s'], ff['train_mae_us'], ff['fwd_max_us'], ff['neutral_us'], ff['n']))
    return out


if __name__ == '__main__':
    main()
