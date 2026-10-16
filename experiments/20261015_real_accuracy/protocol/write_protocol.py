#!/usr/bin/env python3
"""protocol.json を書く(一度だけ。書いたあと sha256 を protocol.sha256 に書いて固定する)"""
import hashlib, json, os, time
REPO = '/home/jetson/jetformer'
REP = '/home/jetson/jetformer_reports/real_accuracy_20261015'
EXP = REPO + '/experiments/20261015_real_accuracy'
OUT = REP + '/protocol.json'
if os.path.exists(OUT):
    raise SystemExit('protocol.json は書き済み(固定済み)。変更は deviations.md へ')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


base = json.load(open(REP + '/protocol_baseline_secondary.json'))
size = json.load(open(EXP + '/size/size_subsets.json'))
ENV = 'cd ~/jetformer && source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer'
COMMON = '--data data/datasets/clean_driving_20261011 --epochs 30 --stride 3 --batch 64 --lr 3e-4 --workers 4 --no-command-input --impact-g 2.5'
RUNS = '/home/jetson/jetformer/data/runs/'
HV0 = '/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u'
HMS = '/home/jetson/bin/jf-heavy --mem 3800M --lock big --need-mb 4000 -- nice -n 10 python -u'


def cmd_train(policy, name, seed, extra='', script='tools/train.py', heavy=HV0):
    return '%s %s --policy %s %s --out %s%s --seed %d%s' % (heavy, script, policy, COMMON, RUNS, name, seed, (' ' + extra) if extra else '')


files = {}
for k, p in {
    'train_rot': EXP + '/train_rot.py',
    'make_size_masks': EXP + '/size/make_size_masks.py',
    'size_subsets': EXP + '/size/size_subsets.json',
    'size_Q': EXP + '/size/size_Q.npz', 'size_H': EXP + '/size/size_H.npz', 'size_T': EXP + '/size/size_T.npz',
    'count_sets': EXP + '/protocol/count_sets.py',
    'baseline_secondary_script': EXP + '/protocol/baseline_secondary.py',
    'baseline_secondary_json': REP + '/protocol_baseline_secondary.json',
    'check_rot_size': EXP + '/protocol/check_rot_size.py',
    'write_protocol': EXP + '/protocol/write_protocol.py',
    'probe_gain_formula': EXP + '/ideas_data/probe_gain.py',
    'temporal_ensembler': REPO + '/jetformer/policies/ensemble.py',
    'judge_functions': REPO + '/tools/sim_cotrain.py',
    'policy_zoo_predictor': REPO + '/tools/policy_zoo.py',
    'rl_evaluate_tables': REPO + '/tools/rl_evaluate.py',
    'train_py': REPO + '/tools/train.py',
    'train_data_py': REPO + '/jetformer/train_data.py',
    'runtime_py': REPO + '/jetformer/policies/runtime.py',
    'real_splits': REPO + '/data/datasets/clean_driving_20261011/splits.json',
    'protocol_1013': '/home/jetson/jetformer_reports/sim_cotrain_20261013/protocol.json',
    'pred_A_s0': REPO + '/data/zoo/policy_zoo_20261012/pred/jetformer_v0_s0.npz',
    'pred_A_s1': REPO + '/data/zoo/policy_zoo_20261012/pred/jetformer_v0_s1.npz',
    'pred_B': REPO + '/data/zoo/policy_zoo_20261012/pred/B.npz',
    'pred_msf_s0': REPO + '/data/zoo/policy_zoo_20261012/pred/msfusion_s0.npz',
    'pred_msf_s1': REPO + '/data/zoo/policy_zoo_20261012/pred/msfusion_s1.npz',
    'design_md': REP + '/design.md',
}.items():
    files[k] = {'path': p, 'sha256': sha(p)}

thr = base['thresholds_2xsA']
proto = {
    'title': '実機の精度に効く方策の試し(real_accuracy_20261015)の手順。結果を見る前に固定(変更は deviations.md)',
    'version': 1,
    'written_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'request': {
        'user': '「また、実機の精度に効く方策を考えて、トライして」(2026-10-15 17:05ごろ)',
        'context': 'C-DIV はシミュレーターで 5/12 周を走れたが、保留した実機の精度は上がらなかった(C-DIV − A = +0.74 µs)',
        'deliverables': '方策の試しと正直な判定(この規則で)・報告(README.md)・約 1 分の動画の絵コンテ(描画はメインのセッション)',
        'not_the_task': 'チャットの「進捗どうですか?」はメインのセッションへの進捗の確認で、この作業の依頼ではない',
    },
    'design': 'design.md(人が読む版。この JSON と同じ内容)',
    'expectations_from_ceiling': {
        'floor_us': {'range': [22, 44], 'central_normal': 31.3, 'central_laplace': 29.5, 'D0': 44.3, 'D0_ci95': [37.5, 56.2]},
        'achieved_upper_bound_us': 46.1,
        'A_minus_floor_us': {'central': 18, 'ci95': [5.6, 29.2]},
        'battery_scene': {'scene': '20261010-193638-0003', 'frames': 573, 'share_of_error': 0.24, 'A_without_scene_us': [42.493, 42.907]},
        'statement': '余地はあるが、今夜の腕で取れるのは数 µs までの見込み。実用上の差 10 µs には届かない見込み。「良い」でも走行の推奨は変えず、A/B/A の候補にするだけ',
        'source': 'ideas_ceiling.md §1・§2、ideas_data.md §0',
    },
    'posthoc_already_seen_not_verdicts': {
        'note': '保留の 4,670 フレーム(と part 1)を見たあとの下調べ。判定に使わない。報告では「下調べ(見たあと)」の段に分けて書く',
        'ENS_v0_plus_msfusion': {'best': [46.824, 47.159], 'last': [47.359, 47.247], 'gain_best_last': [-2.31, -2.19], 'intervention': [125.6, 119.2]},
        'SEED_ENS_v0_s0_s1': {'best': 48.343, 'gain_best': -0.96},
        'TE_J2_uniform': {'best': [48.546, 48.894], 'gain_best_last': [-0.58, -0.56]},
        'TE_exp': {'gain_best_last': [-0.78, -0.77], 'intervention_worse_us': [1.0, 1.5]},
        'GAIN_W5': {'best': [44.935, 45.349], 'gain_best_last': [-4.16, -4.21], 'other_10_scenes_s0': [42.49, 42.90], 'battery_scene_s0': [96.7, 59.5]},
        'all10_mean': 46.10,
    },
    'no_unopened_holdout': {
        'checked': 'protocol/count_sets.py(予測は読まない)',
        'ai_scene_human_rows': 1661, 'ai_scene_human_future_valid': 0,
        'part1_opened_by': ['ideas_sim2real/probe_pred_cache4.py(ENS)', 'ideas_ceiling/te_check.py(TE)'],
        'consequence': '確かめは新しい種と同じ規則で行う。相手に msfusion を選んだこと(候補 4 つ)は同じフレームを見て決めた → 報告に書く',
    },
    'data': {
        'dataset': 'data/datasets/clean_driving_20261011(コピーしない)',
        'train_scenes': 41, 'val_scenes': 11, 'train_samples_stride3': 5912, 'val_samples_stride3': 1559,
        'pred_table_key': '3d339512828b17195d568f1b93ee4a28208f2e85', 'pred_table_rows': 10985,
        'primary_frames': 4670,
    },
    'files': files,
    'baseline_A': {
        'L_us': {'best': {'A_s0': 49.146, 'A_s1': 49.457, 'B': 49.144}, 'last': {'A_s0': 49.235, 'A_s1': 49.752, 'B': 49.091}},
        'spread_s_A': {'best': 0.313, 'last': 0.661},
        'threshold_us': {'best': 0.626, 'last': 1.322},
        'A_runs': ['clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30'],
        'B_run': 'clean_driving_20261011-nocmd-s3-20261011-024626',
        'A_command': 'tools/train.py --policy jetformer_v0 --data data/datasets/clean_driving_20261011 --out <run> --epochs 30 --seed {0,1} --stride 3 --batch 64 --lr 3e-4 --workers 4 --no-command-input --impact-g 2.5(train.log の 1 行目)',
        'msfusion_runs_1012': ['clean_driving_20261011-msfusion-s0-nocmd-stride3-e30', 'clean_driving_20261011-msfusion-s1-nocmd-stride3-e30'],
    },
    'primary': {
        'L': '|予測 − 人|(10 ステップ × [ステア, スロットル])のフレーム平均 [µs]。フレーム = pred_table の part 0 の人の 11 シーン & future_valid & vehicle_valid & 2.5G の衝突の前後でない 4,670',
        'prediction': 'policy_zoo._Frames・_predict_frames(PyTorch FP32・eval)、ジャンル = rl_evaluate._genres(10/13・10/15 と同じ)',
        'judge': 'tools/sim_cotrain.py judge_primary(arm_vals, ref_vals, spread_mult=2, spread_floor=s_A[ckpt]) と final_verdict(best, last)',
        'better': 'max_腕 < min_基準 かつ 平均の改善 > 2 × max(基準の範囲, s_A)。best と last の両方で better のときだけ「良い」',
        'worse': 'min_腕 > max_基準 かつ 平均の悪化 > 閾値(best のまま)',
        'practical': '|平均の差| ≥ 10 µs',
        'crosscheck': '新しい予測の道具で A_s0 の best を予測し直し、10/12 の npz と最大の差 ≤ 0.01 µs(合わなければ予測を止めて調べ、deviations.md へ)',
    },
    'secondary': {
        'thresholds_2xsA': {k: {'best': thr['best'][k]['threshold'], 'last': thr['last'][k]['threshold']}
                            for k in ('S1_no_battery_scene', 'S2_k3_steer', 'S3_intervention_part1', 'battery_scene_only', 'genre4', 'genre5', 'k3_throttle')},
        'baseline_values': base['values'],
        'frames': base['frames'],
        'S1_no_battery_scene': '電池のシーン 20261010-193638-0003 を除いた 4,097 フレーム。全部の腕で判定を並べる。GAIN では「悪い」= 害',
        'S2_k3_steer': '3 フレーム先(index 2、走行で使う値)のステアの L1。判定を並べる(参考)',
        'S3_intervention_part1': 'part 1(intervention_val の採点 1,570 フレーム)の L1。「悪い」= 拒否権(実車のカードに載せない)',
        'reference_only': ['ジャンル別 L1・fwd_bias', '崩れ(10/13 の規則: 直線 542 フレームの前進の強さの低下 > 5%・fwd_bias < −30 µs・向きの正解率 < A − 0.01)', '場所別(LH・GM・R1・HS・DE)', 'R1・DE の窓', '2 つの種の予測の食い違い(A 17.8 µs)', 'シーン別', '底からの超過 L1 − 31 µs'],
        'not_used': 'ステアの向きの正解率(0.98 で飽和)',
    },
    'arms': {
        'ENS': {
            'question': 'v0 と msfusion の 2 本の予測の平均は、A より実機の保留の精度が良いか(下調べ −2.3 µs を新しい種で確かめる)',
            'train': {
                'realacc_20261015-msf-s2': cmd_train('msfusion', 'realacc_20261015-msf-s2', 2, heavy=HMS),
                'realacc_20261015-v0-s2': cmd_train('jetformer_v0', 'realacc_20261015-v0-s2', 2),
                'realacc_20261015-msf-s3': cmd_train('msfusion', 'realacc_20261015-msf-s3', 3, heavy=HMS),
                'realacc_20261015-v0-s3': cmd_train('jetformer_v0', 'realacc_20261015-v0-s3', 3),
            },
            'pairs_fixed_before_training': {'P2': ['realacc_20261015-v0-s2', 'realacc_20261015-msf-s2'], 'P3': ['realacc_20261015-v0-s3', 'realacc_20261015-msf-s3']},
            'combine': '予測 [10,2] µs の重みなしの平均。best は best どうし、last は last どうし',
            'verdicts': {
                'ENS_vs_A': '[P2, P3] vs [A_s0, A_s1](主)',
                'SAME_vs_A': '[(A_s0+A_s1)/2, (v0-s2+v0-s3)/2] vs A(同じ形の平均の対照)',
                'ENS_vs_SAME': 'judge_primary(ENS, SAME, spread_floor=s_A)(形の違いの上乗せ)',
                'FRESH_A': '[v0-s2, v0-s3] vs A(期待は同等)',
                'FRESH_MSF': '[msf-s2, msf-s3] vs A(10/12 の再現)',
            },
            'reference_rows': ['下調べの組 (v0_s0+ms_s0)・(v0_s1+ms_s1)', '4×4 の組み合わせ全部の平均', 'ENS + GAIN', 'シーン別の改善'],
            'deploy_candidate': 'P2 の best(ゲートもこの組)',
            'gpu_minutes': 105,
        },
        'SIZE': {
            'question': '実機のきれいな走行の量を減らすと何 µs 悪くなるか → 2 倍にしたら何 µs 良くなる見込みか(測定。良し悪しの判定はしない)',
            'subsets': {k: {kk: size['subsets'][k][kk] for kk in ('scenes', 'n_scenes', 'samples_total', 'fraction_of_full', 'genre4_share')} for k in ('Q', 'H', 'T')},
            'selection': 'make_size_masks.py(乱数の種 20261015。ジャンル別にログを回して並べ、先頭から f 以上。入れ子 Q ⊂ H ⊂ T)',
            'train': {
                'realacc_20261015-sizeH-s0': cmd_train('jetformer_v0', 'realacc_20261015-sizeH-s0', 0, '--drop-mask experiments/20261015_real_accuracy/size/size_H.npz'),
                'realacc_20261015-sizeH-s1': cmd_train('jetformer_v0', 'realacc_20261015-sizeH-s1', 1, '--drop-mask experiments/20261015_real_accuracy/size/size_H.npz'),
                'realacc_20261015-sizeQ-s0': cmd_train('jetformer_v0', 'realacc_20261015-sizeQ-s0', 0, '--drop-mask experiments/20261015_real_accuracy/size/size_Q.npz'),
                'realacc_20261015-sizeQ-s1': cmd_train('jetformer_v0', 'realacc_20261015-sizeQ-s1', 1, '--drop-mask experiments/20261015_real_accuracy/size/size_Q.npz'),
            },
            'optional_train': {
                'realacc_20261015-sizeT-s0': cmd_train('jetformer_v0', 'realacc_20261015-sizeT-s0', 0, '--drop-mask experiments/20261015_real_accuracy/size/size_T.npz'),
                'realacc_20261015-sizeT-s1': cmd_train('jetformer_v0', 'realacc_20261015-sizeT-s1', 1, '--drop-mask experiments/20261015_real_accuracy/size/size_T.npz'),
            },
            'note': 'エポック数は 30 のまま(データが少ないほど勾配のステップは少ない)',
            'analysis': {
                'L': 'L(Q)・L(H)・L(全部 = A)= 2 つの種の平均(best と last、主指標と S1)。N = 学習サンプル数(1682・3046・5912)',
                's1': '(L(Q) − L(H)) / log2(N_H / N_Q)',
                's2': '(L(H) − L(全部)) / log2(N_全部 / N_H)',
                'G2_lin': 's2(平らにならないと仮定した上限)',
                'G2_pow': 's2 × clip(s2 / s1, 0, 1)(同じ割合で平らになると仮定。s1 ≤ 0 なら 0)',
                'ci': 'val 11 シーンのブートストラップ 2,000 回・乱数の種 0(シーン単位で選び直し、2 つの種の平均で計算)',
                'readings': {
                    'low_value': 'L(H) − L(全部) ≤ 0.626(best)または G2_lin < 1 µs →「2 倍でも 1 µs 未満の見込み。精度のために録画を増やす価値は低い」',
                    'high_value': 'G2_pow ≥ 2 µs →「2 倍で 2 µs 以上の見込み。きれいな周回の録画を勧める(電池の残りも記録)」',
                    'middle': 'ほか →「1〜2 µs の見込み(中くらい)」',
                    'order': 'low_value を先に調べる(両方に当たるときは low_value)',
                },
                'with_T': 'sizeT があれば L(T) も並べ、s2_T = (L(T) − L(全部)) / log2(N_全部 / N_T) を参考に出す(読み方は変えない)',
            },
            'gpu_minutes': 45,
        },
        'ROT': {
            'question': '学習の間引き(stride 3)の位相をエポックごとに回して、捨てていた 2/3 のフレームも使うと精度が上がるか',
            'code': 'experiments/20261015_real_accuracy/train_rot.py(tools/train.py を読み込み、学習の Dataset の行の選び方だけを差し替える)',
            'rule': '位相 = (エポック − 1 + 種) % 3。学習の行 = stride 1 の行のうちフレーム番号 % 3 == 位相。検証・best の選び方・学習率・エポック数・バッチは A と同じ',
            'checked_cpu': {'phase_sizes': [5912, 5903, 5879], 'phase0_equals_A': True, 'disjoint_union_equals_stride1': True, 'stride1_train': 17694, 'val_same_as_A': True},
            'smoke_test_gpu': HV0 + ' experiments/20261015_real_accuracy/train_rot.py --policy jetformer_v0 ' + COMMON.replace('--epochs 30', '--epochs 2') + ' --max-steps 2 --out ' + RUNS + 'realacc_20261015-rot-smoke --seed 0 → rot_log.jsonl に位相 0・1 が出ること。確かめたらフォルダーを消す',
            'train': {
                'realacc_20261015-rot-s0': cmd_train('jetformer_v0', 'realacc_20261015-rot-s0', 0, script='experiments/20261015_real_accuracy/train_rot.py'),
                'realacc_20261015-rot-s1': cmd_train('jetformer_v0', 'realacc_20261015-rot-s1', 1, script='experiments/20261015_real_accuracy/train_rot.py'),
            },
            'verdict': '[rot-s0, rot-s1] vs [A_s0, A_s1](主指標の規則)',
            'reference': ['同じ種どうしの差 ROT_s − A_s', '2 つの種の予測の食い違い(A 17.8 µs)'],
            'if_better': 'エンジンを作ってゲートを確かめる(約 5 分、10/13 のゲート)',
            'gpu_minutes': 47,
        },
        'ENS_GATE': {
            'what': 'v0-s2 と msf-s2 の best.pth を 1 つのモジュール(actions の平均)にまとめ、ONNX → tools/build_engine.py で FP16 のエンジン。成果物は experiments/20261015_real_accuracy/ens/(data/runs には置かない)',
            'io': '入力 image [1,3,224,224]・tof [1,2,42,54]・imu [1,10,6]・state [1,3]・mode [1,6]、出力 actions [1,10,2](v0 と同じ)',
            'gates': {'accuracy_max_us': 2.5, 'run_numpy_median_ms': 6.0, 'run_numpy_p99_ms': 8.0, 'finite': True,
                      'method': '10/12 の engine-check と同じ(実フレーム 64 個で PyTorch FP32(CPU の 2 本の平均)との差、100 回の空回しの後 600 回の run_numpy)'},
            'reference': {'v0_run_numpy_median_ms': 2.37, 'msfusion_run_numpy_median_ms': 3.19},
            'when': 'ENS のあと。ENS が「良い」でなければ、時間が残っているときだけ(参考)',
            'gpu_minutes': 15,
        },
        'Z_zero_training': {
            'GAIN': {
                'formula': 'ideas_data/probe_gain.py の ratio_series・running_median と correct の式: g = (1498 − vehicle の throttle)/(前進 eRPM/1000)、前進 eRPM > 2000 & vehicle_valid & 2.5G の衝突の前後でない。g_t = 同じシーンの過去 5 秒(今のフレームを含む)の中央値(30 フレーム未満なら補正なし)。g_ref = {4: 36.41, 5: 36.86}。予測のスロットル < 1498 のステップだけ 1498 − (1498 − p) × clip(g_t/g_ref, 0.8, 2.0)',
                'window_s': 5, 'clip': [0.8, 2.0], 'neutral_us': 1498, 'min_erpm': 2000, 'min_frames': 30, 'g_ref': {'4': 36.41, '5': 36.86},
                'reference_rows': ['窓 10 秒(後から見た値)', 'clip 0.8〜1.3(実車のカードの上限)', 'ENS + GAIN'],
                'part1': '介入の集合にも同じ式(intervention_val の aligned。ジャンルは各フレームの mode)',
                'verdict': {
                    'G1': '主指標で、組 (i) [GAIN(A_s0), GAIN(A_s1)] vs [A_s0, A_s1] と、組 (ii) [GAIN(v0-s2), GAIN(v0-s3)] vs [v0-s2, v0-s3] の両方が「良い」(閾値は主指標の s_A)',
                    'G2': 'S1(電池のシーンを除く)で、組 (i)・(ii) のどちらも「悪い」でない(閾値 best 1.134・last 1.980)',
                    'G3': 'S3(介入の集合)で、組 (i)・(ii) のどちらも「悪い」でない(閾値 best 3.458・last 4.858)',
                    'card': 'G1・G2・G3 をすべて満たせば、実車の A/B/A のカードの候補(電池が新しいとき・減ったとき)。報告には「主指標の改善はほぼ 1 シーンから」と必ず書く',
                },
            },
            'TE': {
                'formula': 'jetformer/policies/ensemble.py の TemporalEnsembler と同じ計算、ensemble_j = 2・ensemble_m = 0(一様)。表の中の同じシーンの 1・2 フレーム前の予測(無い行は飛ばして重みで割り直す)',
                'verdict': 'T1 = 主指標で組 (i)・(ii) の両方が「良い」、T3 = 介入の集合で「悪い」でない。両方を満たさなければ走行の既定(オフ)のまま',
            },
            'SEED_ENS': 'ENS の SAME の行と同じ',
            'gpu_minutes': 0,
        },
        'GAINL_optional': {
            'question': 'GAIN の補正を学習の側に入れる(効きをそろえたラベル)と、後処理だけより良いか',
            'code': 'experiments/20261015_real_accuracy/train_gainl.py(固定の後に書く。sha256 は deviations.md)',
            'label': '学習と検証の Dataset で、各フレームの r_i = clip(g_t/g_ref, 0.8, 2.0)(GAIN と同じ式、人のスロットルから、過去 5 秒)。スロットルのラベルのうち < 1498 を 1498 − (1498 − y)/r_i に。ステアは変えない',
            'smoke': 'r = 1 のフレームのラベルが元と同じ。GPU で 2 ステップ',
            'train': {
                'realacc_20261015-gainl-s0': cmd_train('jetformer_v0', 'realacc_20261015-gainl-s0', 0, script='experiments/20261015_real_accuracy/train_gainl.py'),
                'realacc_20261015-gainl-s1': cmd_train('jetformer_v0', 'realacc_20261015-gainl-s1', 1, script='experiments/20261015_real_accuracy/train_gainl.py'),
            },
            'eval': '予測に GAIN の後処理(同じ r)をかけて元のラベルと比べる',
            'verdict': '主の問い: [GAINL+後処理] vs [A+後処理](閾値 0.626・1.322)。参考: vs A',
            'gpu_minutes': 55,
        },
    },
    'prediction_and_evaluation': {
        'tool': 'experiments/20261015_real_accuracy/eval/realacc_eval.py(固定の後に書く。crosscheck・predict(GPU)・evaluate・figures(CPU)。sha256 を eval_sha256.txt に書いてから evaluate)',
        'predict_cmd': '/home/jetson/bin/jf-heavy --mem 3G --lock big --need-mb 3200 -- nice -n 10 python -u experiments/20261015_real_accuracy/eval/realacc_eval.py predict',
        'outputs': 'data/zoo/real_accuracy_20261015/pred/<学習結果>.npz・<学習結果>_last.npz(key = pred_table_key)。ずれの試験(pert)はしない',
        'reads_only': 'A・B・10/12 の msfusion の予測は data/zoo/policy_zoo_20261012/pred を読むだけ(書き換えない)',
        'skip_done': '出来た npz(読めて key が合う)は飛ばす。0 バイト・壊れた npz は作り直す',
        'evaluation_json': '/home/jetson/jetformer_reports/real_accuracy_20261015/evaluation.json と evaluation.md',
        'interim': 'ENS の 4 本の予測が出たら、ENS・FRESH・SAME・GAIN・TE を評価して書いてよい(ほかの腕の手順は固定済みなので変えない)',
    },
    'conclusion_rules': [
        '1. 腕ごとに「良い / 同等 / 悪い」と実用上の差(10 µs)を書く。下調べ(見たあと)の数字は別の段に分ける',
        '2. ENS:「良い」かつ ENS vs SAME も「良い」→「形の違う 2 本の平均で精度が上がった。同じ形の平均より上」。「良い」で ENS vs SAME が「同等」→「平均で上がった。形の違いの上乗せは言えない」。「同等」→「下調べの −2.3 µs は新しい種では確かめられなかった」',
        '3. FRESH-A が「同等」でなければ「A の 2 本は種のばらつきを小さく見ていた可能性」と書き、s_A\' = {A_s0〜s3, B} の範囲で同じ判定をやり直して感度として並べる(主の判定は変えない)',
        '4. SIZE は analysis.readings の 3 つのうちどれかを書く。録画を増やす勧めはこの読みだけで決める',
        '5. GAIN・TE は Z_zero_training の verdict の条件で書く。GAIN は「主指標の改善はほぼ 1 シーンから」を必ず添える',
        '6. 走行の推奨をモデル B から替えるのは「良い」かつ実用上の差(≥ 10 µs)かつゲート合格・崩れなし・拒否権なしの腕だけ。10 µs 未満の「良い」は「机上の小さな改善」として、推奨は B のまま、A/B/A のカードの候補にする',
        '7. どの結論も開いたループ(保留の記録)の結果で、実車の走りの改善は示さない(10/12 の規則 7)',
        '8. 多重比較: 判定する問いは ENS・ROT・GAIN・TE・GAINL(任意)。どれも 2 つの種と best・last の両方を求めるので偽陽性は小さいが、0 ではないと書く',
    ],
    'predictions_before_results': {
        'ENS': '「良い」約 65%、−1.5〜−2.3 µs。ENS vs SAME「良い」約 50%',
        'FRESH_A': '同等', 'FRESH_MSF': '同等',
        'SIZE': 'L(H) − L(全部) = +1〜+4 µs、G2_pow 0.5〜2 µs',
        'ROT': '同等(「良い」約 20%)、0〜−1 µs',
        'GAIN': 'G1 両方の組で良い(約 −4 µs)、G2・G3 害なし',
        'TE': '同等',
        'ENS_GATE': '合格、中央値 4.5〜5.5 ms',
    },
    'gpu_order': [
        '1. realacc_20261015-msf-s2', '2. realacc_20261015-v0-s2', '3. realacc_20261015-msf-s3', '4. realacc_20261015-v0-s3',
        '5. crosscheck(A_s0)→ 4 本の予測(best・last)',
        '6. realacc_20261015-sizeH-s0', '7. realacc_20261015-sizeH-s1', '8. realacc_20261015-sizeQ-s0', '9. realacc_20261015-sizeQ-s1',
        '10. ROT の動作確認 → realacc_20261015-rot-s0 → realacc_20261015-rot-s1',
        '11. 6 本の予測', '12. ENS のエンジンのゲート',
        '13.(任意)sizeT-s0・s1 + 予測', '14.(任意)GAINL の動作確認・s0・s1 + 予測',
    ],
    'cpu_parallel': '評価の道具・Z の下調べの表・介入の集合の GAIN・判定・図・報告は CPU で並行してよい(jf-heavy --mem 1500M --lock none + nice -n 10、CUDA_VISIBLE_DEVICES=)',
    'time_plan_jst': {'gpu_start': '2026-10-15 18:00ごろ', 'core_done_est': '22:30ごろ', 'optional_latest_start': '2026-10-16 03:00', 'gpu_hard_stop': '2026-10-16 06:00', 'report_by': '2026-10-16 07:00'},
    'gpu_budget_minutes': {'core_train': 197, 'predict_and_crosscheck': 45, 'gate': 15, 'optional': 95, 'total_est': 352, 'allowed': '600〜720(電源断の余裕を含む)'},
    'cut_order_if_late': ['1. GAINL', '2. sizeT', '3. ENS のゲート(ENS が「良い」でないとき)', '4. ROT(腕ごと。1 本だけの判定はしない)', '5. sizeQ', '6. sizeH'],
    'never_cut': ['ENS の 4 本と予測', 'crosscheck', 'Z の確かめ', '正直な報告(下調べと判定を分ける)'],
    'disk': {
        'free_at_freeze_gb': 6.9, 'keep_free_gb': 4.0,
        'per_run_mb': {'v0_kept': 112, 'v0_last_temp': 169, 'msfusion_kept': 152, 'msfusion_last_temp': 227, 'pred_npz': 2, 'ens_gate': 200},
        'total_est_gb': {'core': 1.4, 'with_optional': 1.9},
        'delete_last_pth': '今夜の学習結果だけ。best と last の予測を書いて読み直し(key の一致・有限)を確かめたあと、その学習結果の last.pth を消す(以後 --resume はできない)。10/12 以前の学習結果には触れない',
        'check_before_each_gpu_job': '空き < 4.5 GB → 任意の腕を削る。< 4.0 GB → 止めて deviations.md',
        'own_folders_only': ['experiments/20261015_real_accuracy/', 'data/runs/realacc_20261015-*', 'data/zoo/real_accuracy_20261015/', '/home/jetson/jetformer_reports/real_accuracy_20261015/'],
        'no_dataset_copies': True,
    },
    'memory': {'v0_family': 'jf-heavy --mem 3G --need-mb 3200', 'msfusion': 'jf-heavy --mem 3800M --need-mb 4000', 'predict': 'jf-heavy --mem 3G --need-mb 3200',
               'on_cgroup_kill': '上限を 800M 上げて --resume。deviations.md に書く'},
    'power_loss': '学習は同じコマンド + --resume(ROT は train_rot.py が rot.json を確かめる)。予測は同じコマンド(出来たものを飛ばす)。last.pth が壊れていたら最初から学習し直して deviations.md',
    'rules': [
        '実車を走らせない。PCA9685・I2C に触れない(実車の確かめは A/B/A のカードを書くだけ)',
        'GPU は 1 つずつ(jf-heavy --lock big)',
        'jetformer-web を再起動しない。git commit をしない',
        '自分の新しいフォルダーの外は消さない(読みだけのディスクの調べが並行して動いている)',
        'ユーザー向けの文章とコメントは日本語',
    ],
    'aba_card_template': {
        'when': '「良い」で拒否権の無い腕(GAIN は G1〜G3)があるときだけ書く。無ければ「実車で試す候補は無い」と書く',
        'design': 'B → 候補 → B、各 5 周、同じ電池の帯、ゲイン 1.3/1.0・先読み 3。見るもの: 周回時間・衝突・取り上げ・直線の eRPM・R1 の通過',
        'candidates': {'ENS': 'P2 のまとめたエンジン', 'ROT': '種 0 の best', 'GAIN': 'B + 補正(上限 1.3)。電池が新しいとき・15 分走って減ったときの 2 回'},
    },
    'deliverables': {
        'evaluation': 'evaluation.json・evaluation.md(全部の腕の判定・副指標・シーン別・§6 の答え合わせ)',
        'readme': 'README.md(ユーザー向け・日本語: 試したこと・効いたこと・効かなかったこと・天井との関係・次に試すこと)',
        'figures': '日本語・16 pt 以上・2160×1350: 腕ごとの主指標、データの量の曲線、天井の図、ENS のシーン別',
        'video_storyboard': 'video/storyboard.md(約 1 分・時刻や予定のセリフなし・まとめと課題で締める・音声は Gemini TTS の Achernar。描画はメインのセッション)',
        'deviations': 'deviations.md(固定の後の変更をすべて: 時刻・理由・影響)',
    },
    'deviation_policy': '固定の後に変えたことは、deviations.md に時刻・何を・なぜ・判定への影響を書く。判定の規則・閾値・組・部分集合・式は変えない(変えるなら、その腕は「固定の外」として判定から外し、参考に下げる)',
    'paths': {
        'report_dir': '/home/jetson/jetformer_reports/real_accuracy_20261015',
        'code_dir': '/home/jetson/jetformer/experiments/20261015_real_accuracy',
        'runs': '/home/jetson/jetformer/data/runs/realacc_20261015-<腕>-s<種>',
        'preds': '/home/jetson/jetformer/data/zoo/real_accuracy_20261015/pred',
        'logs': '/home/jetson/jetformer_reports/real_accuracy_20261015/logs',
    },
}
json.dump(proto, open(OUT, 'w'), ensure_ascii=False, indent=1)
h = sha(OUT)
open(REP + '/protocol.sha256', 'w').write('%s  protocol.json\n' % h)
print(h)
