"""設計の段(2026-10-14、sim_dart_20261014): protocol.json を書く(結果を見る前に固定する手順)。

    python3 -I make_protocol.py <出力 protocol.json>
"""
import json
import sys

EXP = '/home/jetson/jetformer/experiments/20261014_sim_dart_margin'
REP = '/home/jetson/jetformer_reports/sim_dart_20261014'
BL_EXP = '/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap'
BL_REP = '/home/jetson/jetformer_reports/sim_rl_bestlap_20261013'
MODES = '/home/jetson/jetformer/experiments/20261013_sim_rl_modes'

GUARD_COMMON = {
    'zone': 'modes の rl/runtime/perturb.mjs の perturbAllowed(s)(R 上の s から先 2.0 m の「すき間 5 cm で動ける幅」の最小 ≥ 0.60 m、黒シートの区間・裏の直線の区間を除く)が真のときだけ始める',
    'gate_guard': '次の目標ゲートの手前 1.2 m の中では、ゲートの中心から横 |lateral| > 0.30 m なら始めない。乱しの最中に手前 1.2 m の中で |lateral| > 0.45 m になったら打ち切る(ゲートの半幅 0.65 m)',
    'order': '始める判定は「前の乱しが終わってから待つ時間が過ぎた」フレームで、zone → すき間 → ゲートの順。だめならそのフレームは始めず、次のフレームでまた判定する',
}

P = {
    'title': 'シミュレーターの JetFormer を「ずれたときの戻し方(DART)」と「余裕を持たせた線」で学習し直す(sim_dart_20261014)の手順。結果を見る前に固定(書き換えない。違えたことは deviations.md)',
    'version': 1,
    'written_at': '2026-10-14T12:50:00+09:00',
    'request': {
        'user_2026_10_14': '1. 離脱復活モードのデータで、ずれたときの戻し方を教えます。強化学習の走り手を先生にして、ずれた状態での正しい操作を教えます（DART や DAgger と呼ばれる方法です）。'
                           '2. 絶対壁にぶつからないモードや、余裕を持たせたラインのデータで学習します。にトライし、その結果を今日のレポートでまとめたいですね',
        'context': 'sim_rl_bestlap_20261013 の続き。RL のベストラップ(8.092 秒)のきれいなデータで学習した JetFormer(S-R)は、閉ループで 12 本すべて 0.9〜1.4 秒で衝突した(最初の直線で右の壁まで約 10 cm の線・ずれの戻し方を習っていない)',
        'deliverable': '今日のレポート(README.md)と約 1 分の動画(動画はメインのセッションが作る)',
    },
    'design': 'design.md(同じフォルダー。sha256 は protocol.sha256 と一緒に固定しない。design.md は説明、決まりはこの JSON)',
    'questions': {
        'Q1_DART': 'RL の最速の先生(T_fast)で、わざと線から外した状態と、その状態での先生の正しい操作(付け替えたラベル)を足すと(A1)、JetFormer の閉ループの完走は A0(S-R)より増えるか',
        'Q2_MARGIN': '余裕を持たせた線の先生(T_margin)のきれいなデータで学習すると(A2)、完走は A0 より増えるか。そのとき周回時間はどれだけ遅くなるか',
        'Q3_BOTH': '余裕の線 + DART(A3)は、余裕の線だけ(A2)より完走が増えるか',
        'Q4_DAGGER': '(時間があれば)いちばん良い腕に DAgger を 1 回(学習した JetFormer が実際に行った状態に先生のラベルを付けて足し、続きを学習)すると、完走が増えるか',
    },
    'baselines_already_measured': {
        'note': '下の数字は sim_rl_bestlap_20261013 の evaluation.json(作り直さない)。A0 の種 0 だけ、この作業で新しく足す開始 E6〜E11 を走らせる',
        'A0_S-R': {'runs': ['simrl_bestlap_20261013-SR-s0', 'simrl_bestlap_20261013-SR-s1'], 'bundles': BL_REP + '/bundles/', 'E0_E5_completed': '0/12',
                   'failure': '最初の直線(E5 はゲート 20 の直線)で 0.87〜1.35 秒に衝突。衝突の時点で RL の線から 10〜18 cm(E0 の RL の線から)',
                   'takeover': 'RL の方策に衝突の 4〜10 フレーム(0.13〜0.33 秒)前まで任せれば完走できた(eval/takeover.json)'},
        'S-T': {'E0_E5_completed': '2/12(8.830・8.840 秒)'},
        'author_onnx': {'E0_E5_completed': '2/6(9.022・9.033 秒)', 'role': '評価のしくみの陽性対照(E0 で完走・ユーザーの確認器が PASS)'},
        'RL_teacher_node': {'E0_E5': '6/6(8.085〜8.131 秒、遅延 0)'},
    },
    'teachers': {
        'T_fast': {
            'what': 'ベストラップの RL 方策(PPO main 種 1・10,010,624 ステップ)。A0 のデータを走らせたものと同じ',
            'policy_sha256': '7c43cb53cb24f7951bd4640b7bccb71f34ddbcc43599597aa75248aace6153ee',
            'browser_module_base': BL_EXP + '/web/rl-teacher-bestlap.js(sha256 14ca2875…6873)と ' + BL_REP + '/sim/dist/rl/ の features.mjs・policy_forward.mjs・policy_weights.mjs(写して使い、sha256 が同じことを確かめる)',
            'node_relabel': '同じ 3 ファイル(ブラウザと同じもの)を Node で読む',
            'speed_cap_mps': 4.5,
            'measured_now': {'E0_E11_single_lap_s': [7.780, 8.174], 'E0_lap_s': 8.093, 'first_straight_min_clearance_m': [0.098, 0.119],
                             'whole_lap_min_clearance_m': [0.062, 0.074], 'min_gate_margin_m_E0_E5': [0.036, 0.050], 'frac_frames_clearance_lt_15cm': [0.062, 0.118]},
        },
        'T_margin': {
            'what': '経路いろいろ(line)r2 の候補の方策を、一定の寄せ c0 = +0.3(左へ使える幅の 30%)・c_v = 1.0 で走らせる「余裕を持たせた線」',
            'policy': MODES + '/candidates/line_r2/policy.json',
            'policy_sha256': '1ae6be0793914094dbabfbdc325905ab8af01f12ea3bc7b2638b05c6bd0167ae',
            'params': {'line': {'family': 'const', 'c0': 0.3}, 'c_v': 1.0, 'line_profile': 'v2(方策の meta.line_profile)'},
            'runtime': MODES + '/rl/runtime/mode_runtime.mjs と同じフォルダーの全ファイル(写して使う。sha256 は design_probe/sources_sha256.txt)。createModeTeacher(world, config, variation, {mode: "line", weights, params})',
            'status_in_modes_work': 'not_published(L2 の最小すき間 3 cm の判定が、ゲート 16 から R に 79° 横を向いた開始の 1 本だけで 1.33 cm で不合格)。この作業の開始はゲート 0・±3 cm・±3°(RC 教師の約束)で、その形の開始は無い。ここで測った 12 + 10 + 30 本では衝突 0・すき間 ≥ 11.7 cm',
            'measured_now': {'E0_E11_single_lap_s': [8.609, 8.985], 'E0_lap_s': 8.924, 'first_straight_min_clearance_m': [0.436, 0.467],
                             'whole_lap_min_clearance_m': [0.1195, 0.1255], 'min_gate_margin_m_E0_E5': [0.086, 0.095], 'frac_frames_clearance_lt_15cm': [0.019, 0.048],
                             'sweep_c0_10_record_range_starts': 'c0 −0.5〜+0.4 の 10 通り × 10 本すべて完走。最初の直線の最小すき間は c0 とともに 22.5 → 46.4 cm(c0 = +0.3 が最大)、周全体の最小は 10.8〜12.3 cm(design_probe/line_sweep.json)'},
            'selection_rule': '候補(no_wall r2・line r2 の c0 −0.5〜+0.4)のうち、E0〜E11 の 12 本と記録の範囲の 10 本をすべて完走し、周全体の最小すき間 ≥ 10 cm・ゲートの横の余裕 ≥ 8 cm(最速の 6.2 cm・3.6 cm のおよそ 2 倍)を満たすものの中で、S-R がぶつかった最初の直線の最小すき間が最大のもの。同点なら 15 cm 未満のフレームの割合が小さいもの、次に速いもの',
            'selection_result': 'line c0 = +0.3(最初の直線 43.6〜46.7 cm)。no_wall はゲートの余裕 6.8 cm で条件を満たさない(しかも 0.9 秒遅い・シールドの見積もりが乱しで狂う)',
            'fallback': '記録の前の確かめ(recording.gate_checks)で T_margin がブラウザで動かない・予想と合わないときだけ、no_wall r2(bundle/no_wall、シールド v3)のきれいなデータに替える(A2 だけ。A3 は取りやめ)。deviations.md に書く',
            'r3_candidates': '使わない(この手順を固めた 13:00 の時点で r3 の確認に合格したものが無い。あとから合格しても先生を替えない)',
            'speed_cap_mps': 4.5,
        },
        'measurement_files': REP + '/design_probe/(clearance.json・starts_E0_E11.json・line_sweep.json・dart_*.json)。道具は ' + EXP + '/design/measure_teachers.mjs ほか',
    },
    'arms': {
        'common_recipe': 'JetFormer TransFuser 軽量型 jetformer_v0。A0 と同じ: tools/train.py --policy jetformer_v0 --epochs 30 --lr 3e-4 --seed <種> --stride 3 --batch 64 --no-command-input --impact-g 2.5。ImageNet から。AdamW wd 1e-4・cosine(train.py の既定)。最良のエポックは各腕の学習データの val で選ぶ(A0 と同じ決まり)。メモリの工夫(expandable_segments・THP なし・workers 1〜2)は A0 と同じ(' + BL_EXP + '/train/bestlap_train.py を写して使う)',
        'imu_phase': '間引き 3 の位相 0(A0 と同じ)。IMU の 3 つの位相の問題は全部の腕で同じなので、比べ方は崩れない。全フレーム学習・位相の回し替えはしない(学習の時間が 3 倍になり、A0 と違う作り方になる)',
        'A0': {'data': 'jetformer-simulator-rl-bestlap-20min_20261013-174515_7f3650b0(きれいな RL 20 分、60 シーン)', 'status': '学習済み(S-R 種 0・種 1)。作り直さない'},
        'A1': {'data': 'A0 の 60 シーン + D1(T_fast の DART 10 分、30 シーン、ラベルは付け替え済み)= 90 シーン・30 分', 'train_val': '72 / 18', 'seeds': [0], 'run_name': 'simdart_20261014-A1-s0'},
        'A2': {'data': 'D2(T_margin のきれいなデータ 10 分、30 シーン)', 'train_val': '24 / 6', 'seeds': [0], 'run_name': 'simdart_20261014-A2-s0'},
        'A3': {'data': 'D2 + D3(T_margin の DART 10 分、30 シーン、ラベルは付け替え済み)= 60 シーン・20 分', 'train_val': '48 / 12', 'seeds': [0], 'run_name': 'simdart_20261014-A3-s0', 'cut': '遅れたら 3 番目に削る'},
        'A4_DAgger': {'what': '一番良い腕(完走の数、同じなら周回時間の中央値)に DAgger を 1 回', 'cut': '遅れたら最初に削る(今日は時間が足りない見込み)', 'procedure': 'dagger'},
        'extra_seed': '一番良い腕の種 1(時間があれば。遅れたら 2 番目に削る)。判定は種 0 だけで行い、種 1 は「同じ傾向か」を見る参考',
        'combined_datasets': '2 つのデータセットを合わせる腕(A1・A3)は、data/datasets には新しいデータセットを作らず、' + EXP + '/train/data/<腕>/ に aligned・samples などへのシンボリックリンクと、2 つの splits.json を足した splits.json・dataset_card.json(ジャンルの表は同じ)を置いて tools/train.py --data に渡す。学習の前に、読み込んだサンプル数がそれぞれのデータの合計と同じことを確かめる',
        'data_size_confound': 'A1(30 分)> A0(20 分)、A3(20 分)> A2(10 分)。データの量も違う。今日の時間では量をそろえた対照は作れないので、結果は「DART のデータを足した効果」と書く(「同じ量のきれいなデータより良い」とは言わない)',
    },
    'dart': {
        'idea': 'DART(Laskey et al. 2017): 先生が運転している最中に、先生の操作に小さな乱しを足して車を線から外す。外れた状態の画像に、その状態で先生が出す正しい(乱していない)操作を正解として付ける。乱した操作は正解にしない',
        'noise_type': '舵のかたより(バースト)。連続の雑音(OU)はやめた: 舵 0.05 rad の OU でも最速の先生が 10 本中 1 本ぶつかり、最初の直線のずれは 2.3 cm と小さい(design_probe/dart_fastest_ou_s.json)',
        'N_fast': {'for': 'D1(T_fast)', 'steer_offset_rad': [0.06, 0.15], 'sign': '左右 等確率', 'duration_frames': [6, 15], 'gap_after_end_s': [0.6, 1.6], 'first_start_s': [0.2, 0.8],
                   'speed': '乱しの 25% で、速さの指令を × U(0.70, 0.95)(残りは先生のまま)', 'clearance_guard_m': {'start_if_dmin_ge': 0.12, 'abort_if_dmin_lt': 0.06},
                   'node_probe': '30 本で 28 本が衝突・ゲートの外しなし(2 本はぶつかった → 記録の前に乱しの種を引き直す)。乱しごとの最大のずれ: 中央値 6.5 cm・90% 点 11.8 cm、58% が 5〜25 cm。最初の直線のずれ 中央値 4.2 cm。乱したフレーム 16.7%。1 周目 8.116 秒(乱しなし 8.086)'},
        'N_margin': {'for': 'D3(T_margin)', 'steer_offset_rad': [0.10, 0.25], 'sign': '左右 等確率', 'duration_frames': [8, 20], 'gap_after_end_s': [0.5, 1.4], 'first_start_s': [0.15, 0.6],
                     'speed': '乱しの 25% で × U(0.65, 0.95)', 'clearance_guard_m': {'start_if_dmin_ge': 0.10, 'abort_if_dmin_lt': 0.05},
                     'why_larger': '余裕の線の先生は線へ強く戻すので、同じ大きさの乱しではずれが小さい(N_fast と同じ大きさで中央値 3.7 cm)。ずれの大きさを N_fast と同じくらいにそろえた',
                     'node_probe': '30 本で衝突 0、ゲートの外し 4 本(→ 種を引き直す)。乱しごとの最大のずれ: 中央値 8.0 cm・90% 点 12.0 cm、85% が 5〜25 cm。最初の直線のずれ 中央値 8.0 cm。乱したフレーム 22.5%。乱しても最小すき間 11.7 cm'},
        'guards': GUARD_COMMON,
        'target_deviation': '乱しごとの最大のずれ(同じ開始・乱しなしの先生の走りからの横ずれ、R 上の s で合わせる)の中央値 5〜10 cm・90% 点 ≤ 25 cm。記録の前の予想(Node)で確かめる',
        'determinism': '乱しは (シーンの種 seed, データセットの塩 salt, 引き直しの回 k) とフレーム番号だけで決まる(乱数は splitmix32 系。時刻を使わない)。salt: D1 = 0xD1A7、D3 = 0xD3A7。同じフレームで 2 回呼ばれたら同じ値',
        'applied_and_previous_action': '乱しの入った操作 = clamp(先生の舵 + かたより, ±0.52)・clamp(先生の速さ × 倍率, 0, 4.5)。これを sim に渡し、収集器もこれを記録する。先生の観測の「前回の行動」は実際に適用した操作(走行中の JetFormer と同じ約束)。乱していないフレームの次は先生自身の値をそのまま使う(乱しが無ければ A0 の記録と同じ計算になる)',
        'clean_proportion': 'DART のシーンの中で乱しているフレームは N_fast で約 17%、N_margin で約 22%(残りは先生がそのまま運転している = 戻しているところと線の上)。A1 全体では乱したフレームは約 6%、A3 では約 11%',
        'seeds_per_scene': '1 シーンに乱しの並び 1 つ(先生は決定的)',
        'recoverability': {
            'prefilter_node': '記録する 30 シーンを、記録と同じモジュール・元の sim-core.js で前もって走らせ、衝突・本当のゲートの外し(modes の rl/gate_miss.mjs の決まり)・有効な周が 1 周目に無い、のどれかがあれば乱しの種を引き直す(k = 1..5)。決めた k は計画に書く。5 回でだめならそのシーンは乱しの大きさ × 0.7 にする(deviations.md)',
            'browser_retry': 'ブラウザの記録で衝突・ゲートの外しが出たら、そのシーンを次の k で取り直す(1 シーン 2 回まで)。取り直した数は記録の報告に書く',
            'per_frame': 'ラベルの付け替えで、先生がそのフレームから 2 秒以内にぶつかる・ゲートを外す・すき間 2 cm 未満になるときは、そのフレームを学習から外す(relabel.validity)',
        },
        'not_used_and_why': 'modes の離脱復活(recover)の乱し(舵のパルス ±0.52·k・ブレーキ/踏み込み)と recover の方策は使わない: 確認で不合格(C1・C2 など)、r3 の下調べで今の乱しの約 54% しか立て直せない。ユーザーの言う「離脱復活モードのデータ」は、この作業では「先生を RL の走り手にし、立て直せる大きさだけ乱し、正解を先生の操作に付け替えた離脱復活のデータ」として作る',
    },
    'relabel': {
        'rule': '記録した全フレーム t に、先生のきれいな操作の列を正解として付ける。乱した操作(記録された vehicle)は正解にしない',
        'label_definition': 'future_actions[t, k](k = 0..9)= PWM(â_{t+1+k})。â は「本当の状態 s_{t+1}(= s_t に実際に適用した乱し入りの操作 ã_t を 1 フレーム進めた状態、記録の状態と同じ)から先生だけで運転したときの操作」: â_{t+1} = π(s_{t+1}), ŝ_{t+2} = f(ŝ_{t+1}, â_{t+1}), …, â_{t+10}。乱しが無ければ A0 の正解(vehicle[t+1+k])とまったく同じになる',
        'why_from_s_t_plus_1': '走行中は制御遅延 1 フレーム・先読み 1 で、画像 t から出した予測の最初の値はフレーム t+1 に使われる。フレーム t の操作は前の推論ですでに決まっている。だから正解は「すでに決まった操作 ã_t のあとの状態 s_{t+1} から、先生ならどうするか」',
        'pwm': '収集器と同じ式・同じ丸め(収集器のコードから写し、A0 の 60 シーンで差 0 µs を確かめる): 舵 1442 + (δ/0.52)·558(δ ≥ 0)/ 1442 + (δ/0.52)·442(δ < 0)、スロットル 1498 − (v/7.24)·498(PWM の中心 1442 / 1498 µs、A0 と同じ)',
        'how': 'Node で元の sim-core.js を使う。シーンの session.json(開始・物理・種)と frames.jsonl(フレームごとの適用した操作・状態)を読み、開始から記録の操作を再生して本当の状態 s_0..s_N を作る(記録の状態との差 ≤ 1e-9 m・1e-9 rad でなければそのシーンは失敗)。先生(ブラウザと同じファイル)も開始から同じ順で進める(前回の行動 = 適用した操作)。各 t で s_{t+1} から分かれ(fork)、先生だけで 60 フレーム(2 秒)走らせて、最初の 10 個を正解、60 フレーム全体を validity に使う',
        'fork': '既定の作り方: 物理は sim-core の単周の reset({initialState: {x, z, yaw, speed, steering, routeIndex}}) で s_{t+1} から始める(連続周回は initialState を受け付けないため)。設計の段の下調べ(design/fork_check.mjs): 4 つの分かれ目 × 60 フレームで連続周回と位置・向き・速さ・舵の差 0。ただし単周は周の最後のゲートで止まる(recovery_segment_complete、速さ 0)ので、経路を今の目標ゲートで回した world(アプリの reset と同じ回し方)と routeIndex 0 で分かれ、60 フレームの中で周の終わりに届かないようにする。先生は写したランタイム(' + EXP + '/web/ に写す features・mode_runtime)に内部の状態(進みの追跡 s・仮想ゲート・線の起点など)を写し取る snapshot/restore を足して分かれる。1 シーン約 36 秒(1 フレーム約 1 ms × 600 × 60)。基準の作り方(開始から t+1 まで物理と先生を全部やり直す、O(N²)、1 シーン約 3 分)と、A0 の 2 シーン・D1 の 2 シーン・D3 の 1 シーンで正解が ≤ 1e-9 で同じことを確かめてから使う。同じでなければ基準の作り方で全部やる(3 並列で 1 データセット約 30 分。deviations.md)',
        'validity': 'future_valid[t] = (元の future_valid[t]、つまり t < N − 10)かつ(先生の 2 秒の分かれた走りで衝突なし・本当のゲートの外しなし・最小すき間 ≥ 0.02 m)。vehicle_valid・mode は変えない',
        'inputs_unchanged': 'vehicle(今の操作値・eRPM)・IMU・画像・ToF は記録のまま(乱し入りの操作で動いた本当の値)。学習は --no-command-input なので、今の操作値は入力に入らない',
        'where': '取り込んだデータセットの aligned/<シーン>.npz の future_actions・future_valid を書き換える(元の配列は ' + EXP + '/relabel/backup/<データセット>/ に写す)。dataset_card.json に relabel = {schema "jf-sim-dart-relabel/v1", teacher, policy_sha256, noise, horizon 10, validity_horizon_frames 60, frames, changed_frames, invalid_frames, tool_sha256} を足す',
        'scenes_tab': 'DART のデータセット(D1・D3)はシーンのタブに展開しない(to_scenes=False)。シーンから作り直すと、正解が記録の vehicle(乱し入り)から計算し直されるため。D2 は展開してよい',
        'label_mask_safety': 'ブラウザの差し込みモジュールは info に jf を出さない(jetformer/sim/label_mask.py は action_info.jf があると乱したフレームを学習から外すため、付け替えと混ざる)。出すのは info.dart = {v, f, clean: [δ, v], noise: [dδ, 倍率] | null, ev}(アプリが action_info として残せば確かめに使う。残らなくても、付け替えは Node の再計算だけでできる)。取り込みのあと dataset_card に label_mask が無い(または masked_frames 0)ことを確かめる',
        'checks': {
            'R1_identity': 'A0 の 60 シーン(乱しなし)を同じ道具で付け替えると、学習に使う全フレームで元の future_actions との差 ≤ 1 µs、外すフレーム 0(道具が正しい証拠。D1 の前に行う)',
            'R2_replay': 'DART の各シーンで、記録の操作の再生と記録の状態の差 ≤ 1e-9。info.dart が残っていれば、再計算した先生のきれいな操作と info.dart.clean の差 ≤ 1e-9 rad・1e-9 m/s、乱しの印と記録の操作の差が一致',
            'R3_noise_frames': '乱していないフレーム t+1 では、future_actions[t, 0] が記録の vehicle[t+1] と同じ(差 ≤ 1 µs)。乱したフレームでは違う(差の平均 > 0)',
            'R4_invalid_fraction': '外したフレーム(先生が立て直せない)が各データセットで 2% 以下。超えたら記録の報告に書き、そのまま使う(しきい値で学習を止めない)',
        },
        'imu_phase': '付け替えは全フレームに行う(学習で使うのは間引き 3 の位相 0)',
    },
    'recording': {
        'size': '各データセット 30 シーン × 20 秒 = 10 分(18,000 フレーム)。A0 の記録は 7.1 フレーム/秒で 60 シーン 84 分 → 30 シーンで約 45 分。3 つで約 2 時間 15 分で、今日の時間で入るいちばん大きい量',
        'plan_source': BL_REP + '/recording/plan_rlbest60.json(ユーザーの qa/rc-collection-plan-60.json の写し)の 0〜23 番(train、baseline / ToF 雑音 25 mm が交互)と 48〜53 番(val)= 30 シーン。開始(ゲート 0・左右 ±3 cm・向き ±3°)・物理・センサー・種は A0 と同じ(D2 と A0 は同じ開始で、違うのは先生だけ)。名前だけ変える',
        'datasets': {
            'D1': {'teacher': 'T_fast + N_fast', 'collection_name': 'rl-dart-fast-10min', 'scene_names': 'dartfast10-mode8-{train,val}-NN', 'tags': ['simulator', 'rl-dart'], 'to_scenes': False, 'relabel': True},
            'D2': {'teacher': 'T_margin(乱しなし)', 'collection_name': 'rl-margin-line-10min', 'scene_names': 'margin10-mode8-{train,val}-NN', 'tags': ['simulator', 'rl-margin'], 'to_scenes': True, 'relabel': False,
                   'note': '乱しなしなので正解は記録の操作のまま(収集器が作る正解 = 先生の操作)'},
            'D3': {'teacher': 'T_margin + N_margin', 'collection_name': 'rl-margin-dart-10min', 'scene_names': 'margindart10-mode8-{train,val}-NN', 'tags': ['simulator', 'rl-margin', 'rl-dart'], 'to_scenes': False, 'relabel': True, 'cut': 'A3 を削るときは記録しない'},
        },
        'dataset_name_form': 'JetFormer の取り込み名 jetformer-simulator-<collection_name>_<時刻>_<ハッシュ>(A0 と同じ形)。新しいジャンルは作らない(ジャンル 8・num_modes 9)。data/modes.json は変えない',
        'order': 'D1 → D2 → D3',
        'how': 'ベストラップの記録と同じ方法 A: ' + BL_REP + '/sim/ のアプリの写しを ' + REP + '/sim/ に写し(ユーザーの dist とバイト単位で同じことを確かめる)、dart.html・margin.html の import map で ./rc-speed-controller.js を自分の差し込みモジュール(' + EXP + '/web/)に替える。app.js・sim-core.js・収集器・集約はユーザーのまま。browser_collect.py(写し)で「連続試験」',
        'browser': 'snap の Firefox を DISPLAY=:0(XAUTHORITY=/run/user/1000/gdm/Xauthority)、プロファイルは ' + REP + '/ff_profile、geckodriver 4449、記録のブリッジ 8811。全体を jf-heavy --mem 2500M --lock big で包む。始める前にほかの Firefox が無いこと・空きメモリ 2 GB 以上',
        'gate_checks_before_recording': [
            'Node の予想: 30 シーンを記録と同じモジュールで走らせ、乱しなしの最初の 2 シーンが A0 の記録と同じ操作(T_fast)・Node の measure と同じ周回時間(T_margin)であること、乱しの種の引き直しを決めること',
            'ブラウザの試し 2 シーン(D1 の train-00 と D2 の train-00 を短く): 差し込みが読み込まれる・info.dart が frames.jsonl に入る・jf が入らない・周回が予想と近い(1 周目の差 ≤ 0.2 秒)',
            'R1(付け替えの道具の同一性)が合格',
        ],
        'verification_after_recording': 'ベストラップの verify_rl_collection.py・check_collection_zip.py の写し(教師の id・印だけを合わせる)。60 シーンの代わりに 30 シーン。PWM の再計算の差 0 µs、記録の操作の再生 ≤ 1e-9、衝突 0・20 秒走り切り・ゲートの外し 0(DART でも)、画像がすべて違う',
        'teacher_metadata': {'schema': 'course-external-teacher-slot/v1', 'ids': {'D1': 'rl-bestlap-ppo-dart-v1', 'D2': 'rl-line-margin-c03-v1', 'D3': 'rl-line-margin-c03-dart-v1'},
                             'not_learned_policy': False, 'uses_ground_truth': True, 'label_note': 'D1・D3: 記録の操作は乱し入り。学習の正解は取り込みのあと先生の操作に付け替える'},
    },
    'eval': {
        'where': 'ベストラップの評価と同じ: ブラウザ(Firefox・DISPLAY :0 の WebGL)でアプリの写しの index.html を開き、「モデル ZIP を読み込む」で一式を読み込み、「連続試験」で走らせる。推論はユーザーの bridge.py の写しを専用の venv(onnxruntime CPU)で、ポート 8812。jf-heavy --mem 2500M --lock big',
        'settings': {'control_delay_frames': 1, 'model_lookahead': 1, 'steer_gain': 1.0, 'throttle_gain': 1.0, 'mode': 8, 'single_lap_limit_s': 15, 'physics_and_sensors': 'A0 の評価と同じ(記録と同じ)'},
        'bundle': 'ベストラップの export_bundle.py(写し)で ONNX・model_config.json・io_spec.json の ZIP。中心 1442 / 1498 µs、遅延 1・先読み 1・ゲイン 1.0・ジャンル 8',
        'starts': {
            'E0': [0, 0.0, 0.0], 'E1': [0, 0.03, 3.0], 'E2': [0, -0.03, -3.0], 'E3': [0, 0.08, -5.0], 'E4': [0, -0.08, 5.0], 'E5': [20, 0.0, 0.0],
            'E6': [0, 0.05, 2.0], 'E7': [0, -0.05, -2.0], 'E8': [0, 0.02, -4.0], 'E9': [0, -0.02, 4.0], 'E10': [10, 0.0, 0.0], 'E11': [28, 0.0, 0.0],
            'form': '[ゲート, 左ずれ m, 向き 度]。E0〜E5 はベストラップの評価と同じ。E6〜E11 は今日足した(1 つの種で 12 本にするため)。T_fast・T_margin・no_wall・line c0 はこの 12 の開始すべてで Node で完走した(design_probe/starts_E0_E11.json)'},
        'runs_per_model': '単周 12 本(E0〜E11)。参考にフライングラップ(E0 から連続周回 20 秒)1 本',
        'models': ['A1-s0', 'A2-s0', 'A3-s0(削らなければ)', 'A0-s0 の E6〜E11(E0〜E5 はベストラップの結果を使う)', '外部の参照(作者の ONNX)の E0(陽性対照。セッションの最初に 1 本)', '(あれば)一番良い腕の種 1・A4'],
        'determinism_check': 'A1-s0 の E0 を 2 回。違えば全モデルの単周を 3 回ずつの中央値(deviations.md)',
        'per_run_metrics': ['完走(下の定義)', '停止から 1 周の時間', '通過したゲート', '衝突(時刻・物・位置)', '最小すき間', '止まった時刻(衝突・時間切れ)', '衝突・終わりの時点での横ずれ(同じ開始の、その腕の先生の Node の走り / T_fast の走りから)', '0.5 秒の時点のずれ', '最初の直線を抜けたか(衝突せずに x > 2.6 m、ゲート 0 の開始だけ)'],
        'run_success': '完走 = 単周が route_complete で終わり、衝突 0・33 ゲート・黒シートの必須横断 1 回、かつ記録した操作を元の sim-core.js で再生した自前の数え直しが同じ(ベストラップの verify_eval.mjs の写し)。E0 はユーザーの qa/verify-learned-lap.mjs <結果> --model <ONNX> も PASS',
        'takeover': '完走しなかった走りは、ベストラップの takeover.mjs(写し)で、フレーム k から T_fast(と、その腕の先生)に任せて完走できる最後の k を出す(参考)',
        'verification': '全走行で、記録した操作の再生と自前の数え直しが一致・設定(モデルの sha256・遅延・先読み・ゲイン・開始)が計画どおり',
        'comparison_table': '行: A0-s0(E0〜E11)・A0-s1(E0〜E5)・S-T 種 0 / 1(E0〜E5)・作者の ONNX(E0〜E5)・A1・A2・A3・(A4・種 1)・T_fast と T_margin(Node、遅延 0)。列: E0〜E11 の結果(秒 または × 終わり方・ゲート・時刻)・完走数・完走した周の時間の中央値・止まるまでの時間の中央値・最初の直線を抜けた数',
    },
    'verdicts': {
        'note': '結果を見る前に決める。シミュレーターは決定的で、ばらつくのは開始だけ。12 本の差 1 本は「差なし」とする。Fisher の正確検定の p は参考に書くだけ',
        'c': 'c(腕) = E0〜E11 の単周 12 本のうち完走した数(種 0)',
        'Q1_DART': {'効いた': 'c(A1) ≥ 6 かつ c(A1) − c(A0-s0) ≥ 4',
                    '一部効いた': '上でなく、c(A1) − c(A0-s0) ≥ 2、または ゲート 0 の開始 10 本(E0〜E4・E6〜E9)のうち 8 本以上で最初の直線を抜けた(A0 の失敗の形は直ったが、ほかで止まる)',
                    '効かなかった': 'それ以外'},
        'Q2_MARGIN': {'効いた': 'c(A2) ≥ 6 かつ c(A2) − c(A0-s0) ≥ 4', '一部効いた': 'Q1 と同じ形(A2 で)', '効かなかった': 'それ以外',
                      'speed_cost': '完走した開始での T(A2, e) の中央値と、T_fast の Node の時間・T(A1, e)(あれば)との差'},
        'Q3_BOTH': {'上乗せあり': 'c(A3) − c(A2) ≥ 2', '差なし': '|c(A3) − c(A2)| ≤ 1', '悪くなった': 'c(A3) − c(A2) ≤ −2'},
        'Q4_DAGGER': {'上乗せあり': 'c(A4) − c(元の腕) ≥ 2', '差なし': '|差| ≤ 1', '悪くなった': '≤ −2'},
        'speed_vs_teacher': '完走した走りの T(腕, e) / T_先生(e)(その腕の先生の Node・遅延 0 の同じ開始の時間)の中央値。≤ 1.05 なら「先生に近い速さ」',
        'best_arm': '完走数が一番多い腕。同じなら完走した周の時間の中央値が小さい腕',
        'external_reference': 'E0〜E5 で、S-T(2/12)・作者の ONNX(2/6)と並べる(判定には使わない)',
    },
    'dagger': {
        'when': '削らないとき(A1〜A3 の評価が 18:30 までに終わったときだけ)',
        'idea': '学習した JetFormer が実際に行った状態(自分のずれ)に、先生の正しい操作を付けて足す',
        'states': '一番良い腕の評価の 12 本の単周それぞれについて、ブラウザの記録した操作(遅延 1 で実際に適用した操作)を、記録の連続試験で「再生の差し込みモジュール」が開始からそのまま再生する(物理は決定的なので、同じ状態をたどる)。衝突の 10 フレーム前(完走した走りは 300 フレーム目)で先生(その腕の先生)に替わり、20 秒まで先生が運転する(DART の乱しは足さない)',
        'labels': '全フレームを relabel と同じ道具で先生の操作に付け替える(再生の区間は JetFormer の操作で動いた状態、ラベルは先生)。先生が立て直せないフレームは外す',
        'training': '一番良い腕の best から、その腕のデータ + DAgger のシーン(最大 12)で 10 エポック続けて学習(lr 1e-4、ほかは同じ)。run 名 simdart_20261014-A4-s0',
        'eval': '同じ 12 本',
    },
    'time_plan_jst_2026_10_14': {
        'design': '12:20〜13:05(この段)',
        'build': '13:05〜13:50 差し込みモジュール 2 つ・乱しの共通ファイル・30 × 3 の計画と Node の予想(種の引き直し)・付け替えの道具と R1・ブラウザの試し 2 シーン',
        'record': '13:50〜14:35 D1、14:35〜15:20 D2、15:20〜16:05 D3(各 約 45 分)。記録している間に、前のデータセットの検証・取り込み・付け替え(CPU)',
        'train': '16:05〜17:05 A1(90 シーン、約 55 分)、17:05〜17:25 A2(約 20 分)、17:25〜18:05 A3(約 40 分)。各 FP16 エンジンと評価の一式も',
        'evaluate': '18:05〜19:30(陽性対照・A0-s0 の E6〜E11・A1・A2・A3 の 12 本 + フライング 1 本。1 モデル約 20〜25 分)',
        'report': '19:30〜21:00(README.md・図・evaluation.md)。動画はメインのセッション',
        'locks': 'Firefox と GPU の学習はどちらも lock big で、ほかの作業とも順番になる。待ち時間は計画に入っていないので、遅れたら下の削る順番に従う',
    },
    'cut_order_if_late': [
        '1. A4(DAgger)を削る。A1〜A3 の評価が 18:30 までに終わらなければ行わない',
        '2. 種 1(余分な種)を削る。A1〜A3 の学習が 17:15 までに終わらなければ行わない',
        '3. A3(余裕の線 + DART)を削る。D3 の記録を 15:30 までに始められない、または A3 の学習を 17:30 までに始められないとき。D3 を記録したが A3 を削ったときは D3 を取り込みだけして報告に書く',
        '4. 評価の本数を減らす: まずフライングラップ、次に E10・E11、次に E6〜E9(A3 → A2 → A1 の順に減らす)。E0〜E5 は全部の腕で残す',
    ],
    'never_cut': ['A1 と A2 の種 0 の学習と E0〜E5 の評価', 'R1〜R3 の確かめ(付け替えが正しいこと)', '陽性対照(作者の ONNX の E0)', '全走行の再生による数え直し', '今日のレポート'],
    'rules': {
        'safety': '実車・PCA9685・I2C に触れない',
        'gpu': '/home/jetson/bin/jf-heavy --mem 3G --lock big -- …(1 本ずつ)。学習が終わったらすぐに lock を放す',
        'firefox': 'jf-heavy --mem 2500M --lock big -- …、snap Firefox、DISPLAY=:0、XAUTHORITY=/run/user/1000/gdm/Xauthority、プロファイル ' + REP + '/ff_profile、geckodriver 4449、ブリッジ 8811〜8819',
        'node_cpu': 'jf-heavy --mem 1500M --lock none -- nice -n 10 …',
        'rl_training': 'jf-heavy --mem 2G --lock rl -- nice -n 15 …(この手順では RL の学習はしない)',
        'memory': '8 GB・スワップなし。Firefox・学習の前に空きメモリ 2 GB 以上を確かめる',
        'do_not_touch': ['ほかの作業のファイル(' + MODES + '・' + BL_EXP + '・' + BL_REP + '・sim_rl_modes_20261013 の報告)は読むだけ。使うものは ' + EXP + ' に写す',
                         'ユーザーのシミュレーター(/home/jetson/sim/extracted/course-simulator)・ユーザーのブリッジとモデル・jetformer-web(再起動しない)', 'jetformer/web・jetformer/sim*・data/sim*・data/modes.json', 'git commit をしない'],
        'processes': '段の終わりに自分が始めたプロセスを全部止める(次の段が続ける学習だけは除く)',
        'language': 'ユーザー向けの文・コードのコメントは日本語',
    },
    'deviation_policy': '手順と違えたら ' + REP + '/deviations.md に、時刻・何を・なぜ・結果への影響を書く。この JSON は書き換えない(sha256 は protocol.sha256)',
    'paths': {'experiment': EXP, 'report': REP, 'design_probe': REP + '/design_probe', 'bestlap_experiment': BL_EXP, 'bestlap_report': BL_REP, 'modes_experiment': MODES},
}

with open(sys.argv[1], 'w', encoding='utf-8') as f:
    json.dump(P, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('wrote', sys.argv[1])
