"""レビューの確かめ(2026-10-12 夜): 検証のシーンで「正解の通過点・正解の目標速度」を TCPPost に入れたとき、
tcp_mode 1(スロットル = 目標速度 → 前向きの写像)・3(ステア = 通過点を追う)の出力が人の操作とどれだけ合うか。
CPU・numpy だけ。oracle.py と同じ手順(フレームごとに新しい TCPPost・上限なし)で、stride 1(検証の全フレーム)。

    cd ~/jetformer && nice -n 10 python tcp_check/oracle_modes.py   (報告のフォルダーから)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/home/jetson/jetformer')
from jetformer.policies import controller as C          # noqa: E402
from jetformer.policies import tcp_labels as L          # noqa: E402

ROOT = '/home/jetson/jetformer/data/datasets/clean_driving_20261011'
RUN = '/home/jetson/jetformer/data/runs/clean_driving_20261011-tcp-s0-nocmd-stride3-e30'


def main():
    cfg = json.load(open(os.path.join(RUN, 'model_config.json')))
    pp = cfg['postprocess']                               # 学習の finalize が書いた較正(学習のシーンだけ)
    center, neutral = float(cfg['steer_center_us']), float(cfg['throttle_neutral_us'])
    splits, _ = L.read_splits(ROOT)
    lab = L.labels_by_scene(ROOT, keys=('waypoints', 'v_tgt', 'v_now', 'wp_valid'))
    bins = np.array(pp['speed_bins'])
    thr_c, thr_h, st_c, st_h = [], [], [], []
    for sc in splits['val']:
        with np.load(os.path.join(ROOT, 'aligned', sc + '.npz')) as z:
            fut, fv, vv, ts = z['future_actions'], z['future_valid'], z['vehicle_valid'], z['timestamps_us']
            imp = L.impact_mask_frames(ts, z['imu'])
        d = lab[sc]
        ok = d['wp_valid'] & fv & vv & ~imp
        for i in np.nonzero(ok)[0]:
            v = float(d['v_tgt'][i])
            q = np.maximum(1 - np.abs(bins - v) / 0.5, 1e-9)
            out = {'actions': np.zeros((10, 2), np.float32), 'waypoints': d['waypoints'][i],
                   'speed_logits': np.log(q / q.sum())}
            y1, _ = C.TCPPost(pp, (center, neutral)).step(out, float(d['v_now'][i]), {'tcp_mode': 1}, int(ts[i]) * 1000)
            y3, _ = C.TCPPost(pp, (center, neutral)).step(out, float(d['v_now'][i]), {'tcp_mode': 3}, int(ts[i]) * 1000)
            thr_c.append(y1[:, 1]); thr_h.append(fut[i][:, 1])
            st_c.append(y3[:, 0]); st_h.append(fut[i][:, 0])
    tc, th = np.concatenate(thr_c), np.concatenate(thr_h)
    sc_, sh = np.concatenate(st_c), np.concatenate(st_h)
    res = {
        'frames': len(thr_c), 'center_us': center, 'neutral_us': neutral,
        'throttle_mode1': {'mae_us': round(float(np.abs(tc - th).mean()), 1),
                           'corr': round(float(np.corrcoef(tc, th)[0, 1]), 3),
                           'forward_frac_ctrl': round(float((tc < neutral).mean()), 3),
                           'forward_frac_human': round(float((th < neutral).mean()), 3),
                           'mean_ctrl_us': round(float(tc.mean()), 1), 'mean_human_us': round(float(th.mean()), 1)},
        'steer_mode3': {'mae_us': round(float(np.abs(sc_ - sh).mean()), 1),
                        'corr': round(float(np.corrcoef(sc_, sh)[0, 1]), 3)},
        'note': '検証 11 シーン・stride 1・wp_valid & future_valid & vehicle_valid & 2.5G の前後を除く。10 ステップをまとめて計算',
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oracle_modes.json'), 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
