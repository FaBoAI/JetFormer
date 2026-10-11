"""Final video-moment list (verified 2026-10-11 after the reboot). Adds scene, first/last camera frame and wall-clock."""
import datetime
import json
import sys
import numpy as np
sys.path.insert(0, '/home/jetson/jetformer_reports/overnight_20261011/runs/src')
from load import L, KV

M = [
    ('clean_fast_lap', '20261010-204208', 67.9, 78.1,
     'クリーンラップは0周。代わりに唯一の「AIだけの1周」(LapCounter 10.15秒)。直線で最高約4 m/s。75.67秒にDEの仕切りを右前でこすり(8.1G)、そのまま走り続ける。72.8秒に基板の再起動で0.2秒中立'),
    ('clean_fast_lap_alt', '20261010-204208', 76.0, 88.5,
     'ぶつからずに走った最長のAI区間(12.4秒、約1周強)。88.43秒にDEの手前で人が予防的に交代'),
    ('successful_recovery', '20261010-204208', 16.3, 20.2,
     '成功したリカバリーは0回(リカバリーv2は一度も作動していない)。いちばん近い場面: GM出口で衝突(16.89秒・5.1G)→止まって0.47秒、検出まであと0.03秒のところで人が交代し、手で下がる'),
    ('failed_recovery', '20261010-181504', 25.4, 35.4,
     '今回の3本では失敗・繰り返しのリカバリーも0回。比較用に以前の ai_run_008(リカバリーv1): 26.3秒・29.8秒の2回とも後退せず(eRPM 0のまま)、33.3秒にあきらめ(中立)'),
    ('tof_predicted_collision', '20261010-203924', 45.0, 46.3,
     'ToFガード(記録だけ)が正面衝突を予測: 45.41秒にTTC 2.5→0.49秒で警告、45.61秒からブレーキ判定 → 45.80秒にホームストレートの駐車枠の仕切りへ3.5 m/sで8.4G(0.39秒前に警告)。記録だけなので減速していない'),
    ('tof_predicted_collision_alt', '20261010-204208', 27.4, 28.6,
     'ホットスポットDE: TTCが0.81→0.40(27.81秒、警告)→0.08秒と下がり、28.24秒に3.6 m/sで右の仕切りに11.3G(0.46秒前に警告)。AIはそのまま走り続ける'),
    ('tof_false_alarm', '20261010-203924', 42.2, 43.8,
     '誤警報: バックストレートを右の壁沿いに3.4〜3.8 m/sで走り、壁が予測コースの帯に入ってTTC 0.43〜0.45秒が0.2秒ずつ2回(衝突なし)'),
    ('tof_false_alarm_long', '20261010-204605', 38.4, 41.6,
     '長い誤警報: ホームストレートで右の壁・駐車枠に寄ったまま約2.6秒、TTC 0.15〜0.3秒(ブレーキ判定)が続く。衝突はその1.2秒後(42.48秒)の黒カーテン。床の見え方のずれ(この走行で約−23%)が一因'),
    ('hotspot_DE', '20261010-204208', 27.4, 28.9,
     'ホットスポットDE(バックストレート右の仕切り): 3.6 m/sで11.3G、跳ね返って走行は継続(今回の21回の衝突のうち11回がDE/DE後半)'),
    ('hotspot_LH', '20261010-203924', 18.4, 20.0,
     'ホットスポットLH(黒カーテンの左ヘアピン、先読み10): 曲がり切れずカーテンへ7.8G → 止まって0.23秒で人が交代'),
]
out = []
cache = {}
for mid, log, a, b, what in M:
    d = cache.get(log) or L(log)
    cache[log] = d
    ia = int(np.searchsorted(d.ct, a)); ib = int(np.clip(np.searchsorted(d.ct, b) - 1, 0, len(d.ct) - 1))
    m = (d.vt >= a) & (d.vt <= b)
    clock = datetime.datetime.fromtimestamp((d.t0 / 1e6) + a).strftime('%H:%M:%S.%f')[:-3]
    out.append({'id': mid, 'log': log, 'run': d.name, 't_start': a, 't_end': b, 'what': what,
                'scene_start': d.c_scene[ia], 'scene_end': d.c_scene[ib],
                'frame_start': d.cf[ia].replace('/home/jetson/jetformer/data/raw/', ''),
                'frame_end': d.cf[ib].replace('/home/jetson/jetformer/data/raw/', ''),
                'clock_start': clock, 'v_max_ms': round(float(np.max(-d.erpm[m]) * KV), 2)})
    print(json.dumps(out[-1], ensure_ascii=False))
json.dump(out, open('/home/jetson/jetformer_reports/overnight_20261011/runs/src/moments2.json', 'w'), indent=1, ensure_ascii=False)
