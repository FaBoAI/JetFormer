"""Video moments: list + contact sheet (6 frames per moment)."""
import json

import cv2
import numpy as np

from load import L
from sheets import label

M = [
    ('clean_lap', '20261010-204208', 76.0, 88.4, 'いちばんきれいな区間: AIだけで約12.4秒(1周の約93%)ぶつからずに走り、DEの手前で人が予防的に交代。完全なクリーンラップは0周'),
    ('ai_only_lap', '20261010-204208', 67.9, 78.1, '唯一のAIだけの1周(10.15秒)。75.67秒にDEで右前をこすって(8.1G)そのまま走り続ける'),
    ('recovery_preempted', '20261010-204208', 16.3, 20.2, '成功したリカバリーは無し。代わりに: GM出口で衝突(16.89秒・5.1G)→止まって0.44秒(検出まであと0.06秒)で人が交代し、手で下がる'),
    ('failed_recovery_v1', '20261010-181504', 25.4, 35.4, '新しい3本では失敗・繰り返しのリカバリーは無し。比較用に以前の ai_run_008(v1): 26.3秒・29.8秒のリカバリーで後退せず(eRPM 0のまま)、33.3秒にあきらめ(中立)'),
    ('tof_early_warning', '20261010-204208', 74.3, 76.2, 'ToFガードが衝突を0.89秒前に予測(74.78秒に警告・74.94秒からブレーキ判定)→75.67秒にDEで8.1G。記録だけなので速度は落ちていない'),
    ('tof_false_alarm', '20261010-203924', 42.2, 43.8, '誤警報: バックストレートで右の壁沿いを3.7m/sで走ると、壁が進路の帯に入りTTC 0.37秒(衝突なし)'),
    ('tof_false_alarm_floor', '20261010-204605', 38.4, 41.6, '誤警報(長い): ホームストレートで右の壁・駐車枠に寄ったまま2.6秒警告(大半がブレーキ判定)が続く。衝突は1.2秒後のLH。床の基準を作り直すと1.4秒に縮む(床のずれが一因)'),
    ('hotspot_DE', '20261010-204208', 27.6, 28.9, 'ホットスポットDE: 3.6m/sでバックストレート右の仕切りに11.3G、AIはそのまま走行を続ける'),
    ('hotspot_LH', '20261010-203924', 18.4, 20.0, 'ホットスポットLH: 黒カーテンの左ヘアピンで曲がり切れずカーテンへ(7.8G、先読み10)→0.36秒で人が交代'),
    ('hotspot_HSP_new', '20261010-203924', 44.8, 47.0, '新しい場所HS駐車枠: ホームストレートを3.5m/sで右へそれて駐車枠の仕切りの端に8.4G→跳ね返り→0.98秒で交代'),
]

if __name__ == '__main__':
    rows, out = [], []
    cache = {}
    for key, log, t0, t1, what in M:
        d = cache.setdefault(log, L(log))
        i0 = int(np.searchsorted(d.ct, t0)); i1 = int(np.searchsorted(d.ct, t1))
        out.append({'id': key, 'log': log, 'run': d.name, 't_start': t0, 't_end': t1, 'what': what,
                    'scene_start': d.c_scene[min(i0, len(d.c_scene) - 1)],
                    'scene_end': d.c_scene[min(i1, len(d.c_scene) - 1)],
                    'frame_start': d.cf[min(i0, len(d.cf) - 1)].split('/raw/')[-1],
                    'frame_end': d.cf[min(i1, len(d.cf) - 1)].split('/raw/')[-1],
                    'abs_time_start': __import__('datetime').datetime.fromtimestamp((d.t0 + t0 * 1e6) / 1e6).strftime('%H:%M:%S.%f')[:-3]})
        ims = []
        for t in np.linspace(t0, t1, 6):
            k = int(np.clip(np.searchsorted(d.ct, t), 0, len(d.ct) - 1))
            a = int(np.clip(np.searchsorted(d.at, t), 0, len(d.at) - 1))
            im = cv2.imread(d.cf[k])
            tg = 'v %.1f ttc %.2f g%d' % (-d.erpm_at(t) * 5e-4, min(d.ttc[a], 9.9), d.guard[a]) if len(d.ttc) else 'v %.1f out %d/%d' % (-d.erpm_at(t) * 5e-4, d.out[a, 0], d.out[a, 1])
            im = label(im, ['%s %.2fs' % (d.name, t), tg,
                            'sel %d assist %d' % (d.sel[a], d.assist[a])] + ([key] if t == t0 else []))
            ims.append(im)
        rows.append(np.hstack(ims))
    cv2.imwrite('/home/jetson/jetformer_reports/overnight_20261011/runs/fig/video_moments.jpg', np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 75])
    json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/moments.json', 'w'), indent=1, ensure_ascii=False)
    for o in out:
        print(o)
