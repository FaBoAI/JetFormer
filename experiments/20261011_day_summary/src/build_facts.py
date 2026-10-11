#!/usr/bin/env python3
"""facts.json を作る: 生ログの集計(runs_table.json・distance.json)+ 各資料から確かめた事実

    python build_facts.py   # ../facts.json を書く
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'facts.json')
runs = {r['log']: r for r in json.load(open(os.path.join(HERE, 'runs_table.json')))}
dist = json.load(open(os.path.join(HERE, 'distance.json')))

AI_LOGS = [
    ('20261010-155820', 'AIトライ(Claudeから開始)'), ('20261010-160306', 'run_001'), ('20261010-163247', 'run002'),
    ('20261010-171639', 'ai_run_003'), ('20261010-172505', 'ai_run_004'), ('20261010-172910', 'ai_run_006'),
    ('20261010-173445', 'ai_run_007'), ('20261010-181504', 'ai_run_008'), ('20261010-203924', 'ai_run_009'),
    ('20261010-204208', 'ai_run_010'), ('20261010-204605', 'ai_run_101')]

AI_NOTES = {
    '20261010-155820': {
        'assist': 'なし(リカバリー機能はまだ無い)',
        'events': [
            '15:58 の修正(画面との生存確認)の直後に、Claude が API から開始した走行',
            'AI側で中立停止(HOLD)が8回。車両データが約0.52秒ずつ8回途切れた。当時は「I2Cの通信エラー」と判断し 16:10 に修正(29c2e42)',
            '記録を見直すと、途切れのたびに基板の時刻(board_t_us)と連番(seq)が0近くに戻っている。18:15 の ai_run_008 で「基板の再起動」と判断したのと同じ形(推定: 基板の再起動だった可能性が高い)',
            '交代4回のうち2回は、HOLDで止まった直後に人がCH3で取り戻したもの',
            '16:10 のドライブレポートでは 5シーン・88.5秒・AI 54.6秒・交代6回だった。シーン0001〜0002 は 17:54 に削除され、今は0003〜0005(48.4秒)だけ',
        ]},
    '20261010-160306': {
        'assist': 'なし',
        'events': [
            '16:03 の走行。この走行で AI走行ダイジェスト動画(16:03)を作成',
            'ステアゲイン1.6・先読み3。左右の切り返し 2.26回/秒(トライは0.58回/秒)、ハンドルが限界に張り付き AI時間の12%(drive_report 16:10)',
            'HOLD 4回(車両データ 約0.52秒の途切れ。基板の時刻・連番も4回リセット)',
            '42.9秒に 10.4G の正面衝突。38.3秒の 7.0G はセンサーの誤読の疑い(ダイジェスト動画の解析)',
        ]},
    '20261010-163247': {
        'assist': 'なし',
        'events': [
            'I2C修正(16:10)後の最初の走行。HOLD は1回(0.2秒)に減った',
            'AI側は26%だけ。交代8回',
            'シーン0008〜0020 は欠番(削除とみられる)。0021〜0023 は 16:39〜16:40 に壊れて(0バイト・不完全)残っている → 走行中に Jetson の電源が落ちた(電池切れ。ユーザー確認)',
        ]},
    '20261010-171639': {
        'assist': 'なし',
        'events': [
            '17:07 の電源断・再起動のあとの最初の走行。ステア1.0・速度0.8 と控えめ',
            'シーン0001・0004 が欠番(17:46 ごろ削除)。0004 が無いので途中に20秒の空白がある(カメラfps 23.6 はこの空白のため)',
            'AIだけの周 2周 10.95/11.43秒(LapCounter)。遅い',
        ]},
    '20261010-172505': {
        'assist': 'なし',
        'events': ['シーン0002 の20秒だけ残る(0001 は削除)。AI 13.7秒'],
    },
    '20261010-172910': {
        'assist': 'なし',
        'events': [
            'AI区間が1本で152秒続いた(交代は最後の1回だけ)。約16周(方位 15.94周)',
            '走行中にゲインを何度も動かし、AI側で22通りの設定が混ざった(開始時 1.8/1.2/先読み5)',
            '衝突22回(≥2.5G)。衝突マップ解析では DE(バックストレート終端の右外壁の角)が主因: DE通過16回中11回で5G以上',
            '19:03 のドライブレポート(当時の周回計測)では 16周・ベスト8.15秒・中央値9.72秒・クリーン14周。19:36 に周回の区切りを「いちばん鋭いコーナー」に変え、ベストを「衝突なしの周」で出すようにしたため、今の LapCounter では衝突なしの周は 9.75/9.82秒の2周だけ',
            'シーン0001 は 17:46:18 に削除(解析では +20.022秒ずらして元の時刻に戻した)',
        ]},
    '20261010-173445': {
        'assist': 'なし',
        'events': [
            '300秒のログ・AI 235秒・交代11回。そのうち10回は R1(黒カーテン前の左ヘアピンを抜けた直後の右ヘアピン)の衝突が発端',
            'R1 で白い仕切り板に正面から当たって止まると、AIは後退(1950〜2000µs)を出し続けたが中立を挟まないため ESC はブレーキのまま → 人が交代(「ESCの後退ルール」の発見につながった)',
            '走行中にゲインを動かし13通り。終盤 1.0/1.0 に戻してもラップは遅くならなかった(速度ゲインはモデル自身が打ち消す、params.md)',
            '衝突解析時点の report.json は17シーン・328秒と記録していたが、今は0001〜0015(300秒)だけ(AI走行が終わった284秒以降の2シーンが無い)',
        ]},
    '20261010-181504': {
        'assist': 'リカバリーv1 + 後退の受け付け(ESC補助)',
        'events': [
            'リカバリーv1 の初めての実車走行。リカバリー4回作動・後退できたのは0回・あきらめ1回(後退が中立+180µsと弱い、反動を「後退に入った」と誤認)',
            'コントローラ基板(XIAO ESP32-S3)が12回再起動(基板の時計と seq が0に戻る)。開き直しを2秒待っていたため AI が約2.4秒止まった(HOLD 7回・6.9秒)',
            '交代9回のうち3回は基板の再起動で止まった直後の交代。ほかに再起動で AI 区間が0.03秒だけ切れた所が1か所(交代に数えない)',
            'この結果から 18:33 リカバリーv2(後退 中立+300µs、ESC の規則どおり ブレーキ→止まる→中立0.15秒→後退)、18:39 再起動の待ちを 2→0.25秒 に修正',
        ]},
    '20261010-203924': {
        'assist': 'リカバリーv2 + ESC補助、ToFガード=1(記録だけ)',
        'events': [
            '夜の3本の1本目。最初の4区間は先読み10(log.json の開始値 1.65/1.35/10)。先読み10の区間は約2秒ずつで交代、衝突 31.6回/100m(先読み3では9.7)',
            'AI側が23秒しかなく、交代8回',
            'リカバリーv2 は0回作動(止まるたびに人が先に交代)',
        ]},
    '20261010-204208': {
        'assist': 'リカバリーv2 + ESC補助、ToFガード=1(記録だけ)',
        'events': [
            'ゲインを 1.2/1.4 → 1.4/1.25 に変更',
            'AIだけの周が1周 10.15秒(LapCounter。この周にも衝突1回)',
            'DE・DE2 のかすり衝突が多く、当たっても走り続けた',
            '基板の再起動2回(1回は AI 区間が0.07秒切れただけで、交代には数えない)',
        ]},
    '20261010-204605': {
        'assist': 'リカバリーv2 + ESC補助、ToFガード=1(記録だけ)',
        'events': [
            '最後のシーン(0005 不完全・0006 は0.53秒)で記録が途切れた → 20:47:46ごろ Jetson の電源断とみられる(次の起動 20:54)',
            '衝突7回 / 2.18周',
            'この3本(009/010/101)では、カメラに写る自車バンパーが9〜10行(学習データは6〜8行)、ToFの遠い床が基準より14〜23%近い → カメラ・ToFが下向きにずれた疑い(暫定)',
        ]},
}


def ai_entry(log, name):
    r = runs[log]
    lc = r['lapcounter']
    clean = sorted(x['time_s'] for x in lc['lap_times'] if x['clean'])
    ai_only = sorted(x['time_s'] for x in lc['lap_times'] if x['ai_only'] and not x['first'])
    return {
        'name': name, 'log': log, 'start': r['start_local'], 'model': r.get('model_run'),
        'genre': 5, 'params_at_start': r.get('params_at_start'),
        'gain_settings_ai_side': r['gain_settings_ai'],
        'assist': AI_NOTES[log]['assist'],
        'log_duration_s': r['duration_s'], 'ai_time_s': r['ai_time_s'], 'ai_ratio': r['ai_ratio'],
        'ai_distance_m_est': dist[log]['distance_m'],
        'ai_speed_moving_p50_ms_est': dist[log]['v_p50_moving_ms'],
        'ai_heading_laps': r['ai_heading_laps'],
        'lapcounter_laps': lc['laps'], 'lapcounter_ai_only_laps': lc['ai_only_laps'],
        'ai_only_lap_times_s': ai_only, 'ai_only_lap_median_s': lc['ai_only_median_s'],
        'clean_laps': len(clean), 'clean_lap_times_s': clean,
        'clean_lap_median_s_LapCounter': clean[len(clean) // 2] if clean else None,
        'clean_lap_best_s': clean[0] if clean else None,
        'impacts_ai': r['impacts_ai'], 'impacts_ai_ge4g': r['impacts_ai_ge4g'],
        'impacts_per_lap': r['impacts_per_ai_heading_lap'], 'impacts_per_ai_min': r['impacts_per_ai_min'],
        'takeovers': r['takeovers'], 'takeovers_per_ai_min': r['takeovers_per_ai_min'],
        'takeovers_right_after_board_reboot': r['takeovers_right_after_board_reboot'],
        'hold_events_ai': r['hold_events_ai'], 'hold_s_ai': r['hold_s_ai'],
        'board_reboots': r['board_reboots'],
        'assist_onsets': r.get('assist_onsets'), 'tof_guard_onsets_ai_raw': r.get('tof_guard_onsets_ai'),
        'latency_ms_p50_p99': r['latency_ms_p50_p99'], 'trt_infer_ms_p50': r['infer_ms_p50'], 'camera_fps': r['camera_fps'],
        'scenes_present': r['scenes_present'], 'scene_numbers_missing': r['scene_numbers_missing'],
        'scenes_broken': r['scenes_broken'],
        'events': AI_NOTES[log]['events'],
    }


HUMAN_LOGS = ['20261010-135449', '20261010-144028', '20261010-150424', '20261010-152405', '20261010-152707',
              '20261010-192141', '20261010-192654', '20261010-193130',
              '20261010-193638', '20261010-200916', '20261010-201626', '20261010-202633']


def human_entry(log):
    r = runs[log]
    lc = r['lapcounter_as_driver']
    t = [x['time_s'] for x in lc['lap_times']]
    c = sorted(x['time_s'] for x in lc['lap_times'] if not x['impacts'])
    return {'log': log, 'start': r['start_local'], 'description': r['description'], 'scene_genres': r['scene_modes'],
            'scenes': r['scenes_present'], 'recorded_s': r['recorded_s'], 'distance_m_est': dist.get(log, {}).get('distance_m'),
            'heading_laps': r['heading_laps'], 'impacts': r['impacts_all'], 'impacts_ge4g': r['impacts_all_ge4g'],
            'impacts_per_lap': r['impacts_per_heading_lap'], 'impacts_per_min': r['impacts_per_min'],
            'lapcounter_lap_times_s': t, 'no_impact_lap_times_s': c}


facts = {
    'meta': {
        'title': 'JetFormer 2026-10-10 のまとめ(ファクトベース)',
        'period': '2026-10-10 01:58(10/09の夜の着手)〜 2026-10-11 09:26(夜間学習・解析の完了)',
        'git_commits': '31本(10/10 02:30〜04:49 の夜に6本、10/10 09:09〜20:36 に21本、10/11 02:48〜09:26 に4本)',
        'generated': '2026-10-11、CPUのみ(GPU・CUDA不使用)。生ログの集計は src/runs_table.py、この JSON は src/build_facts.py',
        'definitions': {
            'AI側': 'AI_CMD.select(= VEHICLE.select、プロポCH3が1500〜2500µs)',
            'AI時間': 'select=True のフレーム間隔の合計(1間隔0.1秒で打ち切り)',
            '衝突': 'IMU の水平加速度 hypot(ax, ay) ≥ 2.5G、0.6秒以内はまとめて1回(LapCounter・夜の解析 summary.json と同じ量)。強い衝突 ≥ 4G',
            '周回(方位)': 'AI側・走行中(|eRPM|>200)の gyro z を積分(静止時の値をバイアスとして引く)÷360。左回り',
            'LapCounter': 'jetformer.laps.LapCounter(drive_report.py と同じ入力)。区切りは最初の1周でいちばん鋭いコーナー。1秒以上の記録の途切れで数え直す',
            '衝突なしの周(clean)': 'AIだけ・最初の周でない・リカバリーなし・衝突(≥2.5G)なし',
            '交代': 'AI区間の終わり(select True→False)のうち、0.5秒以内にAI側へ戻ったもの(基板の再起動による読み取りの切れ目)を除いた回数',
            '中立停止(HOLD)': 'AI側で hold=True(センサーの値が古い等で中立を出した)が始まった回数',
            '基板の再起動': 'VEHICLE.board_t_us が前のサンプルより小さくなった回数',
            '走行距離': '|eRPM|×0.5e-3 m/s(65mmタイヤ、推定±6%)の積分。空転も含む目安',
        },
        'sources': [
            '~/jetformer の git log(31コミット、10/10 02:30〜10/11 09:26)',
            '~/jetformer/docs/model_design.md、jetformer/model.py、train_data.py、drive.py、assist.py、laps.py、tools/*.py',
            '~/jetformer_reports/20261010/(朝の作業報告 slides.json)',
            '~/jetformer/data/reports/(lowspeed_20261010・lowspeed_20261010b・first_trial_20261010・drive_*・ai_runs_20261010/*.md・compare_clean_driving_20261011)',
            '~/jetformer_reports/ai_drive_digest/(16:03 ダイジェスト、006〜008 レポートの storyboard)',
            '~/jetformer_reports/overnight_20261011/(README.md、compare/model_comparison.md・compare.json、runs/summary.json、research/roadmap.md、course/course_map.json、video/*storyboard.json)',
            '~/jetformer/data/raw/*/log.json・scene.json・*.npz、data/runs/*/result.json・run.json・train.log、data/datasets/*/dataset_card.json',
            'メモリノート jetformer-project.md、このセッションの会話記録(ユーザーの発言の時刻)、last -x(起動の記録)',
        ],
    },
}

facts['model'] = {
    'name': 'JetFormer v0',
    'files': ['jetformer/model.py', 'docs/model_design.md', 'jetformer/train_data.py(normalize_inputs を学習と推論で共用)'],
    'inputs': {
        'camera': 'IMX219(sensor_mode 2 = 1920x1080@30)を V4L2 直接で取り込み、224x224 RGB・30fps。ImageNet 正規化。推論時も学習データと同じ JPEG q92 を一度通す',
        'tof': 'VL53L9(STEVAL-VL53L9)42x54 ゾーン(54x42, binning 2)、2チャネル = 距離(mm/4000、0〜1、0=無効)+ 信頼度(0〜7 → 0〜1)。エンコーダ内で有効マスクを3チャネル目に足す',
        'imu': 'ICM-20948、直前10サンプル(100Hz = 100ms)× [accel/g 3軸, gyro rad/s 3軸]',
        'state': '[(steer_us−1500)/500, (throttle_us−1500)/500, clip(eRPM/20000, ±3)]。学習時は人の操作値、AI走行時は前フレームの予測(ゲイン前)。モデルB(command_input=false)は操作値2つを0にする(eRPM は残す)',
        'genre': 'one-hot(num_modes=6: 0未設定・1〜3駐車・4高速走行・5低速走行)',
    },
    'encoders_tokens': {
        'image': 'ResNet-18(ImageNet 事前学習、fc・avgpool を除く)→ 512x7x7 → 1x1conv → 49トークン',
        'tof': '小CNN 4段(3→32→64→128→256、stride 1,2,2,2、BN+ReLU)→ 6x7 → 42トークン',
        'imu': 'MLP(60→256→256)→ 1トークン',
        'state': 'MLP(3→256→256)→ 1トークン',
        'genre': '線形(6→256、バイアスなし)→ 1トークン',
        'query': '学習可能な出力用トークン 1',
        'total_tokens': 95,
        'embeddings': '学習可能な位置埋め込み(95x256)+ モダリティ埋め込み(6種類)',
    },
    'fusion': 'Transformer Encoder 4層・d=256・8ヘッド・FFN 512・pre-LN・dropout 0.1(最終段で1回だけ融合)',
    'outputs': '出力トークン → LayerNorm → Linear+ReLU → ジャンル別ヘッド(1つの Linear で 6ジャンル×10×2 をまとめて計算し one-hot で選ぶ)→ [10, 2] = 1〜10フレーム先(33〜333ms、約0.33秒)の [ステアリング, スロットル](正規化、×500+1500 で µs)',
    'action_use': '走行時は future_actions[k](k+1フレーム先)を「先読み」k+1 として使う(既定1、当日は主に3)。ゲインはステア中心・スロットル中立(学習データで eRPM=0 のときの人の操作値の中央値: first_trial 1443/1498µs、clean_driving 1441/1496µs)を基準に掛ける',
    'parameters': {'total': 14078168, 'image_backbone_resnet18': 11176512, 'image_proj': 131328, 'tof_encoder': 388896,
                   'imu_mlp': 81408, 'state_mlp': 66816, 'genre_embed': 1536, 'transformer_fusion': 2108416,
                   'heads': 30840, 'trunk_norm_query_pos_modality': 92416,
                   'note': 'num_modes=6 で数えた(CPU で JetFormer(num_modes=6) を作って数えた)。重みファイル best.pth は 56.4MB(FP32)'},
    'speed': {
        'tensorrt_fp16_ms': {'mean': 1.9, 'p99': 2.2, 'source': 'docs/model_design.md(2026-10-10 実測、Orin Nano 8GB MAXN_SUPER、Python版 TensorRT 10.16)'},
        'pytorch_fp16_ms': {'mean': '12.5〜13.2', 'p99': 14.0},
        'trt_vs_pytorch_fp32_max_diff': 0.0007,
        'engine_size_mb': 29,
        'in_loop_trt_call_ms_p50': 5.6,
        'camera_to_pwm_latency_ms': {'p50_range': [27.6, 32.1], 'p99_range': [35.6, 43.9],
                                     'note': '撮像時刻(V4L2 のハードウェア時刻)から PCA9685 へ書き終えるまで。全AI走行の AI_CMD から。004(20秒だけ)は p50 23.1ms、101 は p99 93.8ms。夜の3本(ToFガード記録あり)は p50 30.7〜32.1ms'},
        'loop_fps': '29.2〜29.9(欠番シーンのある 003 を除く)',
        'morning_report_claim': '朝の報告: 走行ループ 30fps、撮影から出力まで平均28ms(推論5.7ms)',
    },
    'training_setup': {
        'loss': '未来10ステップ×2の L1(µs 表示は ×500)',
        'optimizer': 'AdamW lr 3e-4, weight_decay 1e-4, CosineAnnealing, batch 64, GPU では FP16 autocast',
        'augmentation': '色の揺らぎ(RGB チャネルごとに 0.8〜1.2 倍)',
        'stride': '間引き N枚に1枚(当日の本番は3 = 10fps 相当)',
        'masking': '操作値が100ms以内に無い・未来が揃わないフレームは除外。ジャンル未設定は除外。20:36 以降は衝突の前後(既定 水平4G以上の1秒前〜2秒後、clean_driving では 2.5G)を除外',
        'split': 'シーン単位(splits.json)、val_scenes で検証シーンを指定できる',
    },
    'genre_check': 'ほぼ同じ画像に「高速走行=1600µs、低速走行=1400/1450µs」と教えると、IDを変えるだけで 1600/1601µs ↔ 1399/1449µs を出し分けた(机上の検証データ、model_design.md)',
}

facts['transfuser_comparison'] = {
    'transfuser_well_known': [
        '論文: Prakash, Chitta, Geiger "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving"(CVPR 2021)と、拡張版 Chitta ほか "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving"(IEEE TPAMI)。実装 github.com/autonomousvision/transfuser',
        '入力は前方カメラ画像と LiDAR 点群(鳥瞰図 BEV のグリッドにした擬似画像)。目標地点(GPS のターゲットポイント)も使う',
        '画像と LiDAR をそれぞれ CNN(ResNet 系)で処理し、複数の解像度(4段)で両方の特徴をトークンにして Transformer の自己注意で融合し、各ブランチに戻す(マルチスケール融合)',
        '出力は自車座標の未来のウェイポイント(GRU で自己回帰的に予測)で、PID 制御器でステア・アクセル・ブレーキに変換する',
        'CARLA シミュレーターの特権エキスパートを模倣学習(ウェイポイントの L1)。TPAMI 版は深度・セマンティクス・BEV 地図・物体検出などの補助タスクも学習する',
    ],
    'same_as_transfuser': [
        'センサーごとに別のエンコーダで特徴を作り、トークンにして Transformer の自己注意で相互参照させる(幾何的な投影は使わず注意で対応づける)',
        '位置埋め込み(+ JetFormer はモダリティ埋め込み)を足してから融合',
        '模倣学習(人/エキスパートの未来の行動を L1 で当てる)、1回の推論で複数ステップ先を出す',
    ],
    'different_from_transfuser': [
        '融合は最終段の1回だけ(ResNet-18 の 7x7 と ToF の 6x7 をトークン化)。TransFuser の多段(4解像度)融合は Orin Nano で30fpsを確実にするため省いた',
        '距離センサーは LiDAR ではなく 54x42 ゾーンの ToF(VL53L9、カメラから5cm以内に近接設置)。距離+信頼度+有効マスクを小CNNで42トークンに',
        '入力トークンを追加: IMU(100ms)・現在の操作値と回転数(eRPM)・走行ジャンル',
        '出力はウェイポイントではなく、行動そのもの(10フレーム分のステア・スロットル)。自己位置が無い RC カーなので PID を使わない',
        '走行ジャンルで切り替える: ジャンルをトークンとして入力し、さらにジャンル別の出力ヘッドを持つ(条件付き模倣学習 CIL の分岐ヘッド方式)。TransFuser は目標地点で条件づける',
        '出力は学習可能な問い合わせトークン1つから(TransFuser は両ブランチの特徴をまとめて GRU へ)',
        '画像は 224x224 の1台、バックボーンは ResNet-18(TransFuser は ResNet-34 / RegNet など、より大きい)',
        '補助タスク(深度推定など)は v0 では無し(次の検討事項)',
        '学習データはシミュレーターでなく実車の人の運転(当日は約7分〜15分)。推論は TensorRT FP16 で 1.9ms',
    ],
    'uncertain': 'TransFuser の細部(各版のバックボーン、速度入力の有無、ウェイポイント数など)は版によって違う。ここでは一般に知られた構成だけを書いた。動画で細部に触れる場合は論文で確認すること',
}

facts['data_pipeline'] = {
    'vehicle': 'タミヤ TT-02 SRX(1/10 ツーリング 4WD)、65mm タイヤ、Jetson Orin Nano 8GB(MAXN_SUPER、スワップなし)、電池で駆動',
    'sensors': {
        'CAM_FRONT': 'IMX219、V4L2 直接(簡易ISP)、224x224・30fps、JPEG q92。開いた直後に約21fpsで始まる問題を 09:09/09:23 に修正',
        'TOF_FRONT': 'VL53L9(STEVAL-VL53L9)、USB CDC(ファーム公式 l9proto)、42x54 ゾーン・30Hz、生の値 u16(下位13bit=距離mm・0x1FFF=無効、上位3bit=信頼度)',
        'IMU': 'ICM-20948、I2C bus7 0x68、100Hz、±8g・±1000dps(取り付けは90度回転: +x=右、+y=前)',
        'VEHICLE': 'コントローラ基板(XIAO ESP32-S3、ファーム leracer_ctrl v2)I2C 0x28、100Hz のスナップショット(CRC-8): CH1 ステア・CH2 スロットル・CH3 モードの µs、SELECT、eRPM、ホール回数',
        'OUTPUT': 'PCA9685(I2C 0x40、60Hz)CH0=ステア、CH1=スロットル → 基板のセレクタ。プロポ CH3 が AI 側(1500〜2500µs)のときだけ車に届く。人はいつでも CH3 で取り戻せる',
        'SPEAKER': '40ピン I2S2(pin40 DOUT)→ 基板のスピーカーアンプ。Open JTalk で音声案内(収集・走行・交代・ラップ・リカバリー・学習終了など)。マイクは信号が無く既定で無効',
    },
    'collection': [
        '20秒単位のシーン(nuScenes 風の log / scene)。Web 管理画面(ポート8080、systemd jetformer-web): ライブ・収集 / シーン / ジャンル / データセット / 学習 / 走行',
        '走行ごとの収集(10:17 追加): 「n回目の走行をスタート、3、2、1、ピー」で記録開始、指定秒数(当日は20秒、202633 は10秒)で「ピー、n回目の走行終了」、指定秒数休んで指定回数くり返す。連続収集(152405/152707 など)も選べる',
        'ジャンル(02:53 追加): 0未設定・1〜3駐車・4高速走行・5低速走行。収集時に指定し、途中の切り替えも時刻つきで記録。ID は出力ヘッドの番号なので固定',
        'AI走行中の収集(評価用): AI_CMD.npz に予測10ステップ・ゲイン前後の出力・ゲイン・先読み・HOLD・遅れ・推論時間・補助・ToFガードを毎フレーム記録',
    ],
    'dataset_format': [
        'nuScenes 形式の表(v1.0-<名前>/ log・scene・sample・sample_data・calibrated_sensor・ego_pose ほか)+ samples/CAM_FRONT(jpg)・TOF_FRONT(16bit PNG 生値)+ can_bus(IMU・車両 JSON)',
        '学習用に時刻を揃えた aligned/<scene>.npz(カメラ時刻ごとに 直前のToF・直前10個のIMU・現在の操作値・1〜10フレーム先の操作値・有効フラグ・ジャンル)と splits.json・dataset_card.json',
        'AI側(SELECT)のサンプルは学習の操作値に使わない。操作値は100ms以内の有効値だけ',
        'Hugging Face へ: Dataset Viewer で見られる Parquet(画像・ToF画像・操作値・ジャンル)+ nuScenes tar + データセットカード(14:09 追加)。14:14 に akira-sasaki/jetformer-lowspeed_20261010 を非公開でアップロード(train 1,192 / validation 596 フレーム)',
    ],
    'datasets': [
        {'name': 'bench_v1', 'created': '10/10 02:38', 'scenes': '1', 'note': '夜の机上テスト(生ログは残っていない)'},
        {'name': 'v2_check', 'created': '10/10 03:20', 'scenes': '2(1/1)', 'note': 'ファーム v2 の確認'},
        {'name': 'train_test', 'created': '10/10 03:47', 'scenes': '4(3/1)', 'frames': 2114, 'note': '学習タブのテスト用(生ログ 031917・034209 は残っていない)'},
        {'name': 'lowspeed_20261010', 'created': '14:11', 'scenes': '3(2/1)', 'frames': 1788, 'source': '135449', 'note': 'HF に非公開アップロード'},
        {'name': 'lowspeed_20261010b', 'created': '14:46', 'scenes': '5(4/1)', 'frames': 2971, 'source': '135449, 144028'},
        {'name': 'lowspeed_20261010c', 'created': '15:15', 'scenes': '10(8/2)', 'frames': 5931, 'source': '135449, 144028, 150424'},
        {'name': 'first_trial_20261010', 'created': '15:35', 'scenes': '22(18/4)', 'frames': 12648, 'hours': 0.119,
         'source': '135449, 144028, 150424, 152405, 152707(すべてシーンのジャンル5=低速走行)',
         'note': '10/10 の全AI走行のモデルの学習データ。152405/152707 はログの指定は「高速走行」だがシーンは5。人の衝突を含む(衝突マップ解析で33件・強い衝突15件)'},
        {'name': 'clean_driving_20261011', 'created': '10/11 02:18', 'scenes': '52(41/11)', 'frames': 27502, 'hours': 0.26,
         'genre_frames': {'4 高速走行': 10181, '5 低速走行': 17321},
         'source': 'first_trial の22シーン + 「綺麗な走行データ」30シーン(193638 0001〜0003、200916 0001〜0010、201626 0001〜0008、202633 0001〜0009)。駐車(192141・192654・193130)は使わない',
         'note': '学習時に衝突(水平2.5G以上)の1秒前〜2秒後を除外: 学習 3,522 / 検証 1,083 フレーム。検証11シーン = first_trial の検証4 + 新しい7'},
    ],
}

facts['training_runs'] = [
    {'run': 'v2_check-s3-20261010-032020', 'when': '10/10 03:20', 'dataset': 'v2_check', 'stride': 3, 'epochs': '2/2', 'best_val_us': 20.5, 'note': '机上の動作確認'},
    {'run': 'train_test-s1-20261010-034809', 'when': '03:48', 'dataset': 'train_test', 'stride': 1, 'epochs': '8/8', 'samples': '1483/590', 'best_val_us': 1.1,
     'note': '学習タブのテスト。机上データで操作がほぼ一定のため誤差1〜2µsに意味はない(推定)。この後 03:53 にメモリ不足で本体が再起動(サーバー内のTensorRTエンジン作成が約2GB残った)→ エンジンは別プロセスに'},
    {'run': 'train_test-s2-20261010-035947', 'when': '03:59', 'dataset': 'train_test', 'stride': 2, 'epochs': '2/30(途中停止)', 'best_val_us': 5.0, 'note': '途中停止のテスト(state=stopped)'},
    {'run': 'train_test-s1-20261010-040343', 'when': '04:03', 'dataset': 'train_test', 'stride': 1, 'epochs': '4/4', 'best_val_us': 1.7},
    {'run': 'train_test-s3-20261010-044125', 'when': '04:41', 'dataset': 'train_test', 'stride': 3, 'epochs': '10/10', 'best_val_us': 2.2},
    {'run': 'train_test-s2-20261010-044445', 'when': '04:44', 'dataset': 'train_test', 'stride': 2, 'epochs': '8/8', 'best_val_us': 1.6},
    {'run': 'lowspeed_20261010-s1-20261010-141911', 'when': '14:19', 'dataset': 'lowspeed_20261010', 'scenes': '2/1', 'stride': 1, 'epochs': '30/30',
     'samples': '1171/587', 'best_val_us': 70.5, 'best_epoch': 21, 'duration_s': 257, 'command_input': True,
     'note': '最初の実車データでの学習(train 11µs まで下がるが val は 70µs 台 = 過学習)。14:25 に評価'},
    {'run': 'lowspeed_20261010b-s1-20261010-144643', 'when': '14:46', 'dataset': 'lowspeed_20261010b', 'scenes': '4/1', 'stride': 1, 'epochs': '30/30',
     'samples': '2324/587', 'best_val_us': 60.4, 'best_epoch': 22, 'duration_s': 427, 'command_input': True, 'note': '14:56 に評価(同じ検証シーン)'},
    {'run': 'lowspeed_20261010c-s1-20261010-151550', 'when': '15:15', 'dataset': 'lowspeed_20261010c', 'scenes': '8/2', 'stride': 1, 'epochs': '2 で停止',
     'note': 'result.json なし。15:16〜15:20 ごろ Jetson が2回再起動しており、その影響とみられる(推定)'},
    {'run': 'lowspeed_20261010c-s1-20261010-152040', 'when': '15:20', 'dataset': 'lowspeed_20261010c', 'stride': 1, 'epochs': '1 の途中で停止',
     'samples': '4653/1168', 'note': '理由の記録なし。直後の 15:24〜15:28 に連続収集(152405・152707)をして、15:35 に first_trial を作り直している(推定: データを足すため中止)'},
    {'run': 'first_trial_20261010-s3-20261010-153629', 'when': '15:36〜15:48', 'dataset': 'first_trial_20261010', 'scenes': '18/4', 'stride': 3, 'epochs': '30/30',
     'samples': '3371/778', 'best_val_us': 51.0, 'best_epoch': 16, 'duration_s': 586, 'engine_build_s': 67, 'command_input': True, 'mask_impacts': False,
     'genres': [5], 'steer_center_us': 1443, 'throttle_neutral_us': 1498,
     'note': '10/10 のAI走行(トライ〜101)はすべてこのモデル・ジャンル5'},
    {'run': 'clean_driving_20261011-s3-20261011-021839(モデルA)', 'when': '10/11 02:18〜02:38', 'dataset': 'clean_driving_20261011', 'scenes': '41/11', 'stride': 3,
     'epochs': '30/30', 'samples': '5912/1559', 'best_val_us': 35.7, 'best_epoch': 30, 'best_val_per_genre_us': {'4': 40.7, '5': 32.3},
     'duration_s': 1020, 'command_input': True, 'mask_impacts': '2.5G(学習3,522/検証1,083フレーム除外)', 'genres': [4, 5],
     'note': '学習誤差 13µs / 検証 36µs で過学習ぎみ'},
    {'run': 'clean_driving_20261011-nocmd-s3-20261011-024626(モデルB)', 'when': '10/11 02:46〜03:01、再開 08:2x〜08:31', 'dataset': 'clean_driving_20261011', 'scenes': '41/11',
     'stride': 3, 'epochs': '30/30', 'samples': '5912/1559', 'best_val_us': 49.0, 'best_epoch': 25, 'best_val_per_genre_us': {'4': 61.0, '5': 41.0},
     'command_input': False, 'mask_impacts': '2.5G', 'genres': [4, 5],
     'note': '電池切れで26エポック目の後(03:01ごろ)に止まり、25エポック目の重みと学習率の予定から作った途中保存で26〜30を再開(最適化の内部状態は初期化)。26〜30 は 49〜50µs で更新なし、最良は25エポック目。エンジン作成 08:33'},
]
facts['training_note'] = '検証誤差(best_val_us)はデータセット・検証シーン・衝突の除外が違うので、学習結果どうしで直接は比べられない。モデルA と B の検証誤差は、A が人の操作値を入力した条件(教師あり)の値'

facts['offline_evaluations'] = {
    'lowspeed_copycat_1425_1456': {
        'what': 'tools/evaluate.py で検証シーン 135449-0003(587 フレーム)の 1〜10フレーム先の誤差を、教師あり(今の操作値=人の値)と自己入力(今の操作値=前の予測)で測り、「今の操作を続ける」「学習データの平均」と比べた',
        'lowspeed_20261010(2シーン学習)': {'steer_mae_us_k1_k10': {'teacher': [66.5, 89.1], 'self': [68.2, 89.3], 'hold_current': [20.0, 157.4], 'train_mean': [186.3, 186.6]},
                                       'throttle_mae_us_k1_k10': {'teacher': [59.6, 62.1], 'self': [59.8, 62.2], 'hold_current': [8.1, 58.0]}},
        'lowspeed_20261010b(4シーン学習)': {'steer_mae_us_k1_k10': {'teacher': [36.1, 105.4], 'self': [64.7, 100.1], 'hold_current': [20.0, 157.4], 'train_mean': [186.7, 186.7]},
                                        'throttle_mae_us_k1_k10': {'teacher': [24.1, 54.2], 'self': [56.0, 59.4], 'hold_current': [8.1, 58.0]}},
        'finding': 'データを増やすと教師ありの近い先の誤差は下がった(ステア 66.5→36.1µs)が、自己入力では 64.7µs のままで差が開いた。近い先では「今の操作を続ける」(20µs)にも届かない。モデルが入力の「今の操作値」をまねる(コピーキャット)傾向で、AI走行では入力が自分の予測になるため効かない → 15:20 に「今の操作値を入力しない」オプション(--no-command-input)を追加',
    },
    'compare_clean_driving_20261011': {
        'when': '10/11 02:45(初版)・06:21・08:44(最終。モデルBは25エポック目の重みで同じ値)',
        'tool': 'tools/compare_models.py',
        'val_set': 'clean_driving_20261011 の検証11シーン(どのモデルも学習していない)。2フレームに1つ、衝突(2.5G)の前後を除く 2,336 フレーム(元のフレームで約4,670)。first_trial は高速走行(ジャンル4)を学習していないので、そのシーンはジャンル5で代用',
        'metric': '3フレーム先の平均絶対誤差 µs。教師あり = 今の操作値に人の値、自己入力 = 前フレームの予測(AI走行と同じ条件)',
        'results_k3_clean': {
            'first_trial(今のモデル)': {'teacher_steer': 40.7, 'teacher_throttle': 24.2, 'self_steer': 75.5, 'self_throttle': 104.3, 'self_sign_acc': 0.95, 'self_fwd_bias_us': 9.1},
            'モデルA(操作値あり)': {'teacher_steer': 35.1, 'teacher_throttle': 20.7, 'self_steer': 63.4, 'self_throttle': 128.5, 'self_sign_acc': 0.982, 'self_fwd_bias_us': -98.8},
            'モデルB(操作値なし)': {'teacher_steer': 48.7, 'teacher_throttle': 44.9, 'self_steer': 48.7, 'self_throttle': 44.9, 'self_sign_acc': 0.99, 'self_fwd_bias_us': 9.4},
            '今の操作を続ける(基準)': {'steer': 43.6, 'throttle': 15.3},
        },
        'per_genre_k3_self': {
            '低速(5)': {'first_trial': [67.8, 92.8], 'A': [61.4, 149.0], 'B': [47.7, 29.2]},
            '高速(4)': {'first_trial(代用)': [86.5, 120.9], 'A': [66.1, 99.2], 'B': [50.2, 67.4]},
        },
        'findings': [
            '衝突を外して高速走行のデータを足すと、教師ありではステア・スロットルとも約14%良くなった(first_trial → A)',
            '操作値を入力するモデルは、自分の予測を入れると誤差が倍増(A: 35→63µs、first_trial: 41→76µs)。A は自己入力でスロットルが人より約99µs前進側にずれる(R1 への入りすぎにつながる向き)',
            'モデルB は条件によらず同じ誤差で、AI走行の条件(自己入力)では最良: ステア −36%(75.5→48.7)、スロットル −57%(104.3→44.9)、ハンドルの向きの正解率 95.0→99.0%',
            'ただし自己入力の評価は B に構造上有利(B は操作値を見ないので条件の影響を受けない)。閉ループの状態のずれは測っていない。実車の first_trial は自己入力で最も悪いのにラップは人並みだった → 最終判断は実車の A/B/A',
        ],
    },
    'ai_run_analysis_reports_1807_1821': {
        'params.md': [
            '速度ゲインを上げてもほぼ速くならない: モデルが eRPM を見て生の指令を下げて打ち消す(raw ∝ gain^-1.13、R²=0.95)。ゲイン2倍で速度は中央値+9%だけ',
            'steer_gain と先読みが衝突に効くかは判断できない(走行中に設定を動かしたため。006は22通り、007は13通り)',
            '推奨: throttle_gain 1.0〜1.2、steer_gain 1.3〜1.5、先読み 2〜3。1回の走行は1設定、A-B-A、AIで4分(約24周)以上',
            'ラップ: AI 006 中央値9.5秒(14周)・007 10.1秒(13周)、人 9.8秒(24周、IQR 8.5〜12.1)(この資料の「完全周」の定義)',
        ],
        'collision_map.md': [
            '006/007 の衝突45件のうち41件(91%)が R1・DE・GM の3か所',
            'R1 は進入速度で決まる: AI は 5500 eRPM 未満で 1/21、6200 eRPM(約11km/h)以上で 6/8 が強い衝突',
            'DE: 緑マットを抜けたときの向きが人より右(AI 中央値 −4°、人 +15°)',
            'AI は学習データの人の衝突もまねている(学習データ22シーンに人の衝突33件・強い衝突15件、同じ場所)',
            'IMU の軸の訂正: +x=右、+y=前。「横からの衝突」ではなく前〜右前から当たっていた',
        ],
        'human_recovery.md': [
            'TT-02 の ESC は「前進→後退」をブレーキとして扱い、止まってから 後退側→中立(約0.12秒)→後退側 で初めて下がる(人の操作で 33/37 成功、前へ転がっている間は 0/14)',
            'AI は後退を出していたが中立を挟まなかったため動けなかった(スタック10件中8件)',
            '人の回復: ブレーキ約0.27秒 → 中立 → 後退 約1712µs を約0.28秒 → 惰性 → 壁と反対へ切って前進 約0.6秒 → AIへ返す(停止から約3.4秒)',
        ],
        'tof_predictor.md': [
            '床と背景を引いて「進路上の TTC」を出すと、TTC<0.5秒で衝突の88%を0.3秒以上前に予告(誤警報13.2回/分)。1フレーム0.45ms',
            '当たる側も ToF の左右差でほぼ分かる(AI 97%)',
        ],
    },
    'night_runs_analysis_009_010_101': {
        'source': 'overnight_20261011/runs/summary.json(02:54 初版、06:24 再検証)',
        'pooled': {'ai_time_s': 99.0, 'ai_distance_m': 219.7, 'heading_laps': 7.33, 'impacts': 21, 'impacts_per_lap': 2.86,
                   'impacts_per_lap_95ci': [1.77, 4.38], 'takeovers': 15, 'takeovers_per_ai_min': 9.1, 'clean_laps': 0,
                   'ai_only_lap_times_s': [10.62, 10.15, 9.34], 'recovery_v2_episodes': 0},
        'hotspots': {'DE': 7, 'DE2': 4, 'LH(黒カーテン)': 4, 'GM_in': 2, 'GM_out': 1, 'HS_P(駐車枠の板、新しい場所)': 2, 'R1': 1},
        'recovery_v2': '0回作動。止まった12回すべてで人の交代が先(衝突から交代まで 0.23〜1.01秒、中央値0.49秒。検出には停止0.5秒/0.8秒が必要)',
        'tof_guard_record_only': '衝突21回中20回を0.3秒以上前に警告(中央値0.46秒前)。誤警報 17.5回/分。ブレーキ判定で0.3秒以上前は4/21',
        'sensor_tilt_suspected': '写る自車バンパー: 学習データ(13:53〜15:28)6〜8行、003〜008 は2〜3.5行、009/010/101 は9〜10行。ToFの遠い床は 009/010/101 で基準より14〜23%近い(学習時・006/007 は4〜8%)→ カメラ・ToF が下向きにずれた疑い(暫定。衝突増加との因果は未確定)',
        'vs_006_007': '衝突/周は 006/007(1.24〜1.38)の約2.3倍。ただし AI 走行が各23〜51秒と短く、設定も走行中に変わったので評価には使えない(傾向だけ)',
    },
}

facts['ai_runs'] = [ai_entry(l, n) for l, n in AI_LOGS]
facts['ai_runs_deleted_or_missing'] = [
    {'log': '20261010-155018', 'name': 'first_trial_001', 'start': '15:50', 'params': '1.0/1.0/先読み1', 'scenes': 1,
     'what': '最初のAI走行。「走行を始めた画面との通信が3秒途絶えた」で停止(ユーザー 15:51「AI走行失敗しました」、15:52 停止理由を報告)。原因はブラウザにキャッシュされた古い app.js が生存確認の token を送っていなかったこと → 15:58 修正(77551f7)',
     'status': 'ログは 16:14・17:41 時点では存在したが、18:07 には削除されていた(params.md)。数値は残っていない'},
    {'log': '(ai_run_005)', 'what': 'この名前のログは見つからない(作られなかったか削除。不明)'},
]
facts['deleted_or_broken_scenes'] = [
    {'log': '20261010-155018', 'state': 'ログごと削除(first_trial_001)', 'when': '17:41〜18:07 の間'},
    {'log': '20261010-155820', 'state': 'シーン0001〜0002 削除', 'when': '17:54'},
    {'log': '20261010-163247', 'state': '0008〜0020 欠番(削除とみられる)、0021〜0023 は電源断で壊れた', 'when': '0021〜: 16:39〜16:40'},
    {'log': '20261010-171639', 'state': '0001・0004 欠番(途中に20秒の空白)', 'when': '17:46 ごろ'},
    {'log': '20261010-172505', 'state': '0001 欠番', 'when': '17:46 ごろ'},
    {'log': '20261010-172910', 'state': '0001 削除', 'when': '17:46:18'},
    {'log': '20261010-173445', 'state': '末尾2シーン(284秒以降)が無い(17:41 のレポートは17シーン・328秒)', 'when': '17:45 ごろ(フォルダの更新時刻)'},
    {'log': 'ai_run_005', 'state': 'この名前のログは見つからない', 'when': '不明'},
]
facts['ai_runs_totals'] = {
    'all_11_logs': {'ai_time_s': 748.4, 'heading_laps': 66.47, 'impacts': 104, 'impacts_per_lap': 1.56, 'takeovers': 56},
    '006_007': {'ai_time_s': 385.8, 'heading_laps': 38.59, 'impacts': 50, 'impacts_per_lap': 1.30, 'takeovers': 12},
    '009_010_101': {'ai_time_s': 99.0, 'heading_laps': 7.32, 'impacts': 21, 'impacts_per_lap': 2.87, 'takeovers': 15,
                    'note': 'summary.json は 7.33周・2.86回/周(丸めの違い)'},
    'best_clean_lap_s': {'value': 9.68, 'run': 'ai_run_007', 'note': 'LapCounter の「衝突なしの周」の最速。006 は 9.75'},
}

facts['human_reference'] = {
    'afternoon_training_data': {
        'logs': '135449・144028・150424・152405・152707(13:53〜15:28、first_trial の学習データ22シーン)',
        'recorded_s': 426.8, 'distance_m_est': 1065, 'heading_laps': 35.42, 'impacts': 35, 'impacts_ge4g': 16,
        'impacts_per_lap': 0.99, 'impacts_per_min': 4.9,
        'laps_lapcounter': {'n': 23, 'median_s': 10.07, 'best_s': 7.94, 'no_impact_n': 9, 'no_impact_median_s': 9.24},
        'laps_params_md': '完全周24周・中央値9.8秒(IQR 8.5〜12.1)・最速8.2秒(5.3/6.2秒の2周はスピンか近道の可能性で除外)',
        'note': '衝突の数は定義で変わる(params.md の x/y 別判定+跳ね除外で28件、collision_map の |a|−1g 判定で33件)',
    },
    'evening_clean_data': {
        'logs': '193638・200916・201626・202633(19:36〜20:31、説明「綺麗な走行データ」、ジャンル4が3本・5が1本)',
        'recorded_s': 508.9, 'distance_m_est': 1216, 'heading_laps': 45.99, 'impacts': 27, 'impacts_ge4g': 20,
        'impacts_per_lap': 0.59, 'impacts_per_min': 3.2,
        'laps_lapcounter': {'n': 21, 'median_s': 9.9, 'best_s': 7.3, 'no_impact_n': 13, 'no_impact_median_s': 9.64},
        'note': '20秒(202633 は10秒)ごとの走行なので、周は走行ごとに数え直した(各走行の最初の周も含めた目安)。roadmap.md は別の数え方で「衝突22回・約0.5回/周」',
    },
    'parking_data': {'logs': '192141(駐車1)・192654(駐車2)・193130(駐車3)、19:21〜19:35、21シーン 395.8秒', 'note': '学習には使っていない'},
    'per_log': [human_entry(l) for l in HUMAN_LOGS],
    'comparison': 'AI(006/007)は 1.24〜1.38回/周、夜の AI(009/010/101)は 2.86回/周。人は学習データ 0.99回/周、綺麗な走行データ 0.59回/周。ラップはAI(衝突なし 9.68〜10.18秒、AIだけの周の中央値 9.73/10.11秒)と人(約9.2〜10秒)で同程度',
}

facts['power_and_reboots'] = {
    'boot_times_last_x': ['10/10 09:17', '09:44(09:43 正常終了の後)', '10:18', '13:25', '15:16', '15:20', '16:39', '17:07', '19:47', '20:54',
                          '10/11 03:01', '(時計なしの起動 1970-01-01 表示、03:13〜04:27 の間)', '04:27', '04:29'],
    'unclean_ends': '09:43 以外はすべて正常に終了していない(crash)。10:16 はユーザーが外出のため電源を切った',
    'battery': 'Jetson は電池で動いていて、10/10〜11 の電源断(16:39・17:07・19:4x・03:01 ほか)は電池切れ(ユーザー確認 10/11 08:22「電源: Jetsonはは、電池切れ」)',
    'data_evidence': [
        '16:39〜16:40: run002 のシーン0021〜0023 が壊れている',
        '19:38:37 の 193638-0004 が0.3秒で途切れ、19:39 に 006〜008 レポート動画を書き出した後〜19:47 の間に停止',
        '20:47:46: ai_run_101 の最後のシーンが0.53秒で途切れ、20:54 に起動',
        '03:01: モデルB の学習が26エポック目の後で停止',
    ],
    'controller_board_reboots': '基板の再起動(時計と seq が0に戻る): トライ8回・run_001 4回・008 12回・010 2回。人が強く加速したときに多く、電圧低下とみられる(18:39 のコミット)。トライ・run_001 の「I2Cエラー」も同じ形だった可能性(推定)',
    'oom_0353': '10/10 03:53 にメモリ不足で本体が再起動(サーバー内の TensorRT エンジン作成で約2GBが残ったまま学習)。エンジン作成を別プロセスにして対策(朝の報告・メモリノート)',
}

facts['timeline'] = [
    ['10/10 01:58', 'ユーザー依頼: カメラ・STEVAL-VL53L9・IMU・コントローラ基板(PWM・回転数)を収集する仕組み、TransFuser のように Transformer でセンサーフュージョン、30fps 推論、20秒単位の nuScenes 風 scene 管理'],
    ['02:30', 'コミット: マルチセンサー収集基盤と Web 管理画面(330ee04)'],
    ['02:41', 'コミット: JetFormer v0 モデル・学習・TensorRT 変換(TRT FP16 約1.9ms)(70bbfb9)'],
    ['02:42', 'ユーザー依頼: 駐車(1)〜(3)・高速走行・低速走行のジャンルIDで走り方を切り替え'],
    ['02:53', 'コミット: 走行ジャンル(トークン+ジャンル別ヘッド)(58dfa75)'],
    ['03:10', 'コントローラ基板のファームを v2 に更新(ユーザー)'],
    ['03:27', 'ユーザー依頼: 学習のテストと、学習結果で走行する画面(ゲインバー、走行時の収集チェック)'],
    ['03:53', 'メモリ不足で本体が再起動 → TensorRT エンジン作成を別プロセスに'],
    ['04:49', 'コミット: AI走行タブ・学習タブ、基板 v2・ToF 公式形式(52916a7)。自動テスト17件・ブラウザ確認8件、レビュー指摘24件を修正'],
    ['04:51', '朝の作業報告動画(JetFormer_作業報告_20261010.mp4)'],
    ['08:51〜09:34', 'I2S マイク対応・カメラ 21fps 問題の修正・車のスピーカーで音声案内(be10ac1, ceff7d9, 3be3ffd, fd4f680)'],
    ['10:17', 'コミット: 走行ごとの収集(カウントダウン・終了のピー・連続収集)、Hugging Face 書き出し(05708a0)'],
    ['10:18〜13:25', 'ユーザー外出のため Jetson OFF(10:16 に連絡)'],
    ['13:53', '人の低速走行の収集開始(135449、20秒×3)'],
    ['14:05〜14:14', '収集の時刻ずれ(NTP同期前)修正、HF アップロード(lowspeed_20261010 を非公開で)'],
    ['14:19 / 14:46', '学習 lowspeed(2シーン、val 70.5µs)/ lowspeed_b(4シーン、val 60.4µs)。14:25・14:56 に評価 → 操作値をまねる傾向'],
    ['14:26', 'コミット: 評価ツール tools/evaluate.py'],
    ['14:40 / 15:04', '人の走行を追加(144028 ×2、150424 ×5)'],
    ['15:15〜15:21', 'lowspeed_c の学習を2回開始して途中で止まる(15:16・15:20 に再起動あり)'],
    ['15:20', 'コミット: 「今の操作値を入力しない」オプション(6b292e5)'],
    ['15:24〜15:28', '連続収集(152405・152707、速めの走行)'],
    ['15:35〜15:48', 'データセット first_trial(22シーン)→ 学習30エポック(val 51.0µs)→ TensorRT エンジン'],
    ['15:50', '最初のAI走行 first_trial_001 → 画面との通信途絶(3秒)で停止'],
    ['15:58', '修正(77551f7)→ Claude から AI走行トライ(155820)'],
    ['16:03', 'run_001(ステア1.6・先読み3)。この走行でダイジェスト動画を作る'],
    ['16:10', 'コミット: 「I2Cエラー」で車両データ・IMU が途切れて中立停止する問題を修正、drive_report.py(29c2e42)'],
    ['16:13', 'ユーザー依頼: AI走行のダイジェスト動画(キャラクターが横で解説、メインは走行動画とセンサー群)'],
    ['16:32', 'run002'],
    ['16:39〜16:40', 'Jetson 電源断(電池切れ)'],
    ['17:07', 'Jetson 電源断・再起動'],
    ['17:16 / 17:25', 'ai_run_003(1.0/0.8)/ ai_run_004(1.5/0.9)'],
    ['17:29', 'ai_run_006: AI 152秒連続・約16周・交代1回'],
    ['17:34', 'ai_run_007: AI 235秒・交代11回(R1 で止まる、ESC の後退ルール)'],
    ['17:46', 'AI走行ダイジェスト動画(16:03 の run_001)を書き出し。同じころ 172910-0001 などのシーンが削除される'],
    ['17:49', 'ユーザー: 「1周のタイムをどんどん縮めていきたい」'],
    ['17:54', 'コミット: 自動リカバリー v1・後退の前に中立を挟む補助(536d43a)'],
    ['17:58', 'コミット: ジャイロで周回を数えラップ表示・読み上げ(70fb365)'],
    ['18:07〜18:21', '解析レポート: params.md・collision_map.md・human_recovery.md・tof_predictor.md'],
    ['18:15', 'ai_run_008(リカバリー v1、基板の再起動12回)'],
    ['18:18', 'ユーザー: 「ぶつかった時のバックの強さが弱いので、もっと強くバック」'],
    ['18:33', 'コミット: リカバリー v2(人のやり方と ESC の規則、後退 中立+300µs)(f198431)'],
    ['18:39', 'コミット: 基板の再起動を数えて0.25秒で開き直す(f53f0e1)'],
    ['18:40', 'コミット: AI走行のベストプラクティス(889f3b8)'],
    ['18:43', 'コミット: ToF ガード(TTC 予測、既定は記録だけ)(c70de79)'],
    ['19:21〜19:35', '駐車データの収集(駐車1〜3、学習には使わない)'],
    ['19:36', 'コミット: 周回の区切りを最も鋭いコーナーに、ベストは衝突なしの周(6cbc3c9)'],
    ['19:36〜19:38', '「綺麗な走行データ」の収集開始(193638)'],
    ['19:39', 'AI走行レポート動画(006〜008)を書き出し'],
    ['19:4x', 'Jetson 電源断(19:47 起動)'],
    ['20:09〜20:31', '「綺麗な走行データ」(200916・201626・202633)'],
    ['20:36', 'コミット: 衝突の前後を学習から外す(69aaf1e)'],
    ['20:39 / 20:42 / 20:46', 'ai_run_009 / 010 / 101(リカバリー v2・ToFガード記録だけ)'],
    ['20:47', 'Jetson 電源断(ai_run_101 の記録が途切れる、20:54 起動)'],
    ['10/11 02:13', 'ユーザー依頼: AI走行の評価動画、衝突のないデータセットで30エポック学習、精度比較(02:17: 低速+高速、駐車は学習しない)'],
    ['02:18〜02:38', 'データセット clean_driving_20261011(52シーン)→ モデルA 学習(val 35.7µs)・エンジン'],
    ['02:46', 'モデルB(操作値なし)学習開始'],
    ['02:48', 'コミット: --impact-g と tools/compare_models.py(3a5c36a)'],
    ['02:53〜02:56', 'ロードマップ初版、コース図(course.html から、10.30×5.60m、基準線1周25.5m)'],
    ['03:01', '電源断(電池切れ)。モデルB は26エポック目の後で停止。この後 04:29 までに数回起動・停止'],
    ['06:06', 'ユーザー: 「jetson復活」'],
    ['06:16〜06:53', '調査(offline_rl.md・pretrain_finetune.md)とロードマップ改訂・レビュー(07時台)'],
    ['06:21', 'モデル比較(モデルBは25エポック目)'],
    ['07:12', 'AI走行評価動画(009・010・101)'],
    ['08:22', 'ユーザー: 電源断は電池切れ'],
    ['08:26', 'コミット: 学習の途中保存と再開(c4fb4e4)'],
    ['08:31〜08:33', 'モデルB を26〜30エポック再開して完了、エンジン作成'],
    ['08:44 / 08:47', '比較の最終版 / 夜間作業レポート動画'],
    ['09:25〜09:26', 'コミット: センサーの向きの点検・交代の理由ボタン・収集のメモ(652ff9a, 080a38f)'],
]

facts['uncertain_items'] = [
    'トライ・run_001 の車両データの途切れ: 当時は I2C エラーと判断したが、記録の形は基板の再起動と同じ(推定)',
    '電源断の正確な時刻: 起動記録(last -x)とデータの途切れから推定。原因が電池切れなのはユーザー確認',
    'lowspeed_c の2回の学習が止まった理由(記録なし。1回目は再起動、2回目はデータ追加のための中止と推定)',
    'ai_run_005 は記録が見つからない',
    '削除されたシーン(155018 全体、155820-0001〜0002、163247-0008〜0020、171639-0001/0004、172505-0001、172910-0001、173445 の末尾2シーン)の理由は記録なし',
    'センサーの向きのずれ(カメラ・ToF)は暫定の解析で、衝突が増えた原因かは未確定',
    '走行距離・速度は eRPM からの換算(タイヤの空転を含む、±6%)',
    '人のラップは20秒ごとの走行を周回計で数えた目安(資料ごとに数え方が違う)',
    'TransFuser の細部は版によって違う',
]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(facts, f, ensure_ascii=False, indent=1)
print('wrote', OUT)
