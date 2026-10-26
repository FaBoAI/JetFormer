#!/usr/bin/env python3
"""protocol_v2.json を書く(2026-10-25)。学習の前に決まりを固定する。読むだけ・書くのはレポートのフォルダだけ。

    python3 -I /home/jetson/jetformer/experiments/20261025_train_v2/write_protocol_v2.py

入力: /home/jetson/jetformer_reports/train_v2_20261025/protocol_numbers.json(protocol_numbers.py が書く)
      /home/jetson/jetformer_reports/dataset_v2_plan_20261025/plan_checks.json(閾値の値。plan.md 6.3)
出力: /home/jetson/jetformer_reports/train_v2_20261025/protocol_v2.json
決まりの中身は plan.md 6 章・qualify_plan.md 1.5 / 3.7 / 3.8・critique.md をもとにした(どこから来たかは各項目の src)。
最後に自分で確かめる(学習の本数・名前・封と手本の重なり・閾値の出どころ)。合わなければ書かずに止まる。
"""
import datetime
import json
import os
import sys
from pathlib import Path

REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PLAN = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
OUT = REP / 'protocol_v2.json'
NUM = json.load(open(REP / 'protocol_numbers.json', encoding='utf-8'))
PC = json.load(open(PLAN / 'plan_checks.json', encoding='utf-8'))['seed']

B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
DS_V2 = 'data/datasets/dataset_v2_20261025'
TBL = 'data/derived/dataset_v2_20261025/'
PY = '/home/jetson/miniconda/envs/jetracer/bin/python'
PREFIX = ('cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- nice -n 10 '
          'env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 choom -n 1000 -- '
          '%s -I experiments/20261015_real_accuracy/train/nothp.py %s -u tools/train.py' % (PY, PY))
COMMON = '--stride 3 --batch 64 --workers 4 --no-command-input --impact-g 2.5'
DATA_ARGS = {
    'v2h': '--data %s --drop-mask %sdrop_v2h.npz' % (DS_V2, TBL),
    'v2m': '--data %s --drop-mask %sdrop_v2m.npz --sample-weights-npz %sweights_v2m.npz --mode-remap-npz %sremap_v2m.npz'
           % (DS_V2, TBL, TBL, TBL),
    'base': '--data data/datasets/clean_driving_20261011',
}
AUG2 = '--aug-brightness 0.15 --aug-contrast 0.2 --aug-shift-px 4 --aug-tof-noise 0.02 --aug-top-brightness 0.15'
SEALED = [r['log'] for rows in NUM['seal'].values() for r in rows]
TEACHER_LAPS = [  # inventory.json の lap_list(who=ai・clean・model B・9.0 秒未満・10/24 2 回目)。t_start [s]・周 [s]・サンプル
    ('20261024-162033', 9, 22.462, 8.252, 79), ('20261024-162033', 9, 30.714, 8.360, 81),
    ('20261024-162033', 9, 48.062, 8.398, 82), ('20261024-162033', 9, 56.460, 8.498, 82),
    ('20261024-162033', 9, 73.348, 8.681, 86), ('20261024-162033', 9, 82.030, 8.528, 83),
    ('20261024-162033', 9, 90.558, 8.902, 87), ('20261024-162033', 9, 99.459, 8.833, 85),
    ('20261024-162033', 9, 108.292, 8.960, 87), ('20261024-162305', 10, 7.454, 8.620, 84)]
V2M_HALF_LAPS = [  # 10/23 の B のきれいな 9.0〜9.5 秒の周(重み 0.5)
    ('20261023-210840', 8, 46.132, 9.209, 88), ('20261023-210840', 8, 65.019, 9.090, 88),
    ('20261023-211209', 11, 7.584, 9.350, 91), ('20261023-211547', 15, 26.951, 9.492, 92)]
TRAIN_AI_LOGS = {  # build_ai_dataset の --ai-logs(学習の回の AI 走行の記録すべて。封の 14 本は入れない)
    '10/10': ['20261010-155820', '20261010-160306', '20261010-163247', '20261010-171639', '20261010-172505',
              '20261010-172910', '20261010-173445', '20261010-181504', '20261010-203924', '20261010-204208',
              '20261010-204605'],
    '10/23': ['20261023-203740', '20261023-203753', '20261023-203843', '20261023-204403', '20261023-204755',
              '20261023-205205', '20261023-205309', '20261023-205418', '20261023-205458', '20261023-210109',
              '20261023-210840', '20261023-211020', '20261023-211118', '20261023-211209', '20261023-211308',
              '20261023-211357', '20261023-211448', '20261023-211547', '20261023-211737', '20261023-211828',
              '20261023-212202', '20261023-212242', '20261023-212538', '20261023-212657', '20261023-212802',
              '20261023-212934', '20261023-213149', '20261023-213243', '20261023-213559'],
    '10/24 2 回目': ['20261024-160133', '20261024-160536', '20261024-160654', '20261024-160847', '20261024-162033',
                    '20261024-162305'],
}


def thr_value(key, kind):
    """閾値 [µs]。plan_checks.json にあるもの(plan.md 6.3 の値)はそれを、無いもの(n = 1・3、k = 3)は protocol_numbers.json"""
    pc = PC.get(key)
    mine = NUM['thresholds'][key]
    src_pc = {'4v5': 'thr_2se_4v5', '2v5': 'thr_2se_2v5', 'pairs4': 'thr_2se_4v4'}
    if kind in src_pc and pc is not None:
        return pc[src_pc[kind]], 'plan_checks.json'
    if kind == 'pairs2' and pc is not None:
        return round(2 * pc['se_2v2'], 3), 'plan_checks.json'
    return mine['thr_' + kind], 'protocol_numbers.json'


def threshold_table():
    keys = {'l1_best': 'best_l1', 'l1_last': 'last_l1', 'interv_best': 'best_l1_interv', 'interv_last': 'last_l1_interv',
            'steer_bias_best': 'best_steer_bias', 'thr_bias_best': 'best_thr_bias'}
    out = {'vs5': {}, 'pairs': {}, 'source': {}}
    for n in (1, 2, 3, 4):
        out['vs5']['n%d' % n] = {}
        for name, key in keys.items():
            v, src = thr_value(key, '%dv5' % n)
            out['vs5']['n%d' % n][name] = v
            out['source']['vs5.n%d.%s' % (n, name)] = src
    for k in (2, 3, 4):
        out['pairs']['k%d' % k] = {}
        for name, key in keys.items():
            v, src = thr_value(key, 'pairs%d' % k)
            out['pairs']['k%d' % k][name] = v
            out['source']['pairs.k%d.%s' % (k, name)] = src
    out['five_mean'] = {name: PC[key]['mean'] for name, key in keys.items()}
    out['five_sd'] = {name: PC[key]['sd'] for name, key in keys.items()}
    return out


def v2_runs():
    runs = []

    def add(stage, name, arm, data, policy, epochs, seed, aug, lr='3e-4', note='', extra=''):
        cmd = '%s --policy %s %s %s--epochs %d --lr %s %s --seed %d --out data/runs/%s' % (
            PREFIX, policy, data, (extra + ' ') if extra else '', epochs, lr, COMMON, seed, name)
        if aug:
            cmd = cmd.replace(' --seed ', ' %s --seed ' % AUG2)
        runs.append({'stage': stage, 'name': name, 'arm': arm, 'family': 'v2', 'policy': policy, 'data': data,
                     'epochs': epochs, 'seed': seed, 'aug': 'AUG2' if aug else '色の揺らぎだけ(×0.8〜1.2、いつも入る)',
                     'lr': float(lr), 'command': cmd, 'note': note})
    for k in range(4):
        add(1, 'v2_20261025-v2h-e30-s%d' % k, 'v2h-e30', DATA_ARGS['v2h'], 'jetformer_v0', 30, k, False)
        add(1, 'v2_20261025-v2m-e30-s%d' % k, 'v2m-e30', DATA_ARGS['v2m'], 'jetformer_v0', 30, k, False)
    for k in range(4):
        add(2, 'v2_20261025-{x}-e60-s%d' % k, '{x}-e60', '{X_DATA}', 'jetformer_v0', 60, k, False,
            note='先回りで x = v2h。D1 が v2m・base なら、その x でやり直す(先回りの分は参考)')
    add(2, 'v2_20261025-{x}-e100-s0', '{x}-e100', '{X_DATA}', 'jetformer_v0', 100, 0, False,
        note='参考だけ(v3 の見込み)。選ばない。遅れたら最初に削る(任意の腕の次)')
    for k in range(4):
        add(3, 'v2_20261025-{x}-aug2-e{E}-s%d' % k, '{x}-aug2-e{E}', '{X_DATA}', 'jetformer_v0', -1, k, True,
            note='エポック = D2 の答え(既定 60)。--aug-top-brightness は C1 で足す引数')
    for k in range(2):
        add(4, 'v2_20261025-{x}-msf{A}-e{E}-s%d' % k, '{x}-msf{A}-e{E}', '{X_DATA}', 'msfusion', -1, k, False,
            note='エポック = D2、拡張 = D3 の答え({A} = "-aug2" か "")。2 本の平均の部品・単体は参考')
    for r in runs:
        if r['epochs'] == -1:
            r['epochs'] = 'D2(30 か 60。既定 60)'
            r['command'] = r['command'].replace('--epochs -1', '--epochs {E}')
        if r['stage'] == 4:
            r['aug'] = 'D3 の答え(AUG2 なら引数を足す)'
            r['command'] = r['command'].replace(' --seed ', ' {AUG_ARGS} --seed ')
    return runs


def bfast_runs():
    runs = []
    for k in range(3):
        name = 'bfast_20261025-e15-s%d' % k
        data = '--data %s --drop-mask %sdrop_bfast.npz --sample-weights-npz %sweights_bfast.npz' % (DS_V2, TBL, TBL)
        cmd = '%s --policy jetformer_v0 %s --init data/runs/%s/best.pth --epochs 15 --lr 1e-4 %s --seed %d --out data/runs/%s' % (
            PREFIX, data, B_RUN, COMMON, k, name)
        runs.append({'stage': 'B-FAST', 'name': name, 'arm': 'bfast-e15', 'family': 'B-FAST', 'policy': 'jetformer_v0',
                     'data': data, 'init': 'data/runs/%s/best.pth' % B_RUN, 'epochs': 15, 'seed': k, 'lr': 1e-4,
                     'aug': '色の揺らぎだけ(×0.8〜1.2、いつも入る)', 'command': cmd,
                     'note': 'qualify_plan 1.5 の P3。B の重みから。ジャンルの付け替えなし(手本の周は記録のジャンル 5 のまま)'})
    return runs


def main():
    thr = threshold_table()
    t = thr
    runs = v2_runs() + bfast_runs()
    gm = NUM['genre_mark']
    tim = NUM['timing']
    proto = {
        'schema': 'jf-train-v2-protocol/v1',
        'title': 'データセット v2 の 19 本と B-FAST の 3 本(計 22 本)の学習の事前の決まり。学習の前に固定し、結果を見て変えない',
        'version': 1,
        'written_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds'),
        'written_by': 'workflow のサブエージェント(決まりの固定の係)。学習・GPU・torch は使っていない。data/ は変えていない。git commit していない',
        'request': {
            'user_20261025_03': '「Best practiceに絞って、ぶつからない走行で25秒を目指しましょう」',
            'goal': '止まった状態から 3 周 25 秒・ぶつからない。次の実車は 11/3(火・祝)',
            'approved_as_relayed_by_main_session': [
                'B-FAST とデータセット v2(19 本)の学習を始める(起動は main session)',
                '10/24 の 1 回目(昼、No.1〜4)と 3 回目(夕方、No.11〜20)は封(学習に使わない確認用)',
                'AI の良い周を学習に入れてよい(10/25)',
                '新しいジャンル(速い走り)を data/modes.json に足してよい(v2-M のため)'],
            'not_the_task': '進捗の質問は main session が答える。この決まりを変える依頼ではない',
        },
        'sources': {
            'plan': str(PLAN / 'plan.md') + '(6 章・4 章・5 章・付録 B)',
            'critique': str(PLAN / 'critique.md'),
            'qualify_plan': '/home/jetson/jetformer_reports/bench_20261024_report/qualify_plan.md(1.5 P3 B-FAST・3.7・3.8・7 章)',
            'numbers': str(REP / 'protocol_numbers.json') + '(閾値の数え直し・ジャンルの片寄り・sha256・時間)',
            'doc_sha256': {Path(d['path']).name: d.get('sha256') for d in NUM['docs']},
        },
        'lock': {
            'this_file': str(OUT),
            'sha256_file': str(REP / 'protocol_v2.sha256'),
            'how_to_check': 'cd /home/jetson/jetformer_reports/train_v2_20261025 && sha256sum -c protocol_v2.sha256',
            'read_only': 'chmod a-w(protocol_v2.json・protocol_v2.md・protocol_numbers.json・protocol_v2.sha256)',
            'second_lock': {
                'file': str(REP / 'protocol_v2_lock2.json'),
                'when': '段 0 のあと・最初の v2 と B-FAST の予測の前(10/25 夜)',
                'contents': ['データセット dataset_v2_20261025 の dataset_card.json の sha256',
                             '表 7 つ(drop_v2h・drop_v2m・weights_v2m・remap_v2m・drop_bfast・weights_bfast・tables_card.json)の sha256',
                             'C3(机上のベンチ v2)・C4(キュー v2)・T1(make_v2_tables.py)・数え直しの台本の sha256',
                             '段 0 の数(下の stage0_checks の 1〜8 の答え)',
                             'data/modes.json(ジャンル 9 を足したあと)の sha256'],
                'rule': '中身は stage0_checks の決まりで出す。lock2 の sha256 を段 0 の README(ユーザーが見る)に書き、chmod a-w。'
                        'lock2 で決まりを変えてはいけない(数を埋めるだけ)',
            },
            'commit': '実行のときに了承を取って commit する(この係は commit しない)',
        },

        # ------------------------------------------------------------ 学習
        'runs_count_note': '依頼の「20 本」= v2 の 19 本 + B-FAST(1 つの腕)。B-FAST は qualify_plan 1.5・plan.md 4.2 の「種 3 本」なので、'
                           '学習のプロセスは 22 本(v2 19 + B-FAST 3)。GPU の見込みは 約 18.1 時間(v2-M が軸なら 約 23.3 時間)',
        'common_recipe': {
            'src': 'plan.md 4.1(B と同じ作り方)',
            'policy': 'TransFuser軽量型 jetformer_v0(dim 256・4 層・8 ヘッド・14.1M)。段 4 だけ TransFuser多段型 msfusion',
            'command_input': '入れない(--no-command-input)',
            'stride': '3(位相 0)。検証も 3',
            'optimizer': 'バッチ 64・AdamW(weight decay 1e-4)・勾配の上限 1.0・AMP(train.py の既定)',
            'lr_schedule': 'cosine(エポックごと・最後は 0)。温めなし(train.py の既定)',
            'impact': '--impact-g 2.5(水平 2.5G の 1 秒前〜2 秒後を外す)',
            'pretrained': 'ImageNet の ResNet-18(v2)。B-FAST は B の best.pth から(--init)',
            'best': 'H1 の検証(train.log の 1,559 サンプル)の L1 が一番小さいエポック = best.pth。11/3 に使う重みは best.pth(B と同じ)。比べは best と last の両方',
            'workers': '4(段 0 で使ったメモリの最大が 3.5 GB を超えたら 2)',
            'memory_retry': 'メモリの上限で止まった(rc 137)ときだけ、上限を 1 回だけ上げる(最大 4.5G)。deviations.md に書く',
            'command_prefix': PREFIX,
            'command_src': 'data/runs/bench_20261021-slow-s0/train.log の 2 回目の開始行(上限 3872M。1 回目の 3G は rc 137 で止まった)',
        },
        'choice_placeholders': {
            '{x}': 'D1 の答え: v2h(既定)・v2m・base(v2 を使わない = B のデータ clean_driving_20261011)',
            '{X_DATA}': DATA_ARGS,
            '{E}': 'D2 の答え: 60(既定)か 30',
            '{AUG_ARGS}': {'AUG2': AUG2, 'なし': ''},
            '{A}': 'D3 が AUG2 なら "-aug2"、なしなら ""',
            'AUG2_detail': {'色の揺らぎ(チャンネルごと)': '×0.8〜1.2(いつも入る)', '明るさ(全体)': '±0.15',
                            'コントラスト': '±0.2', '上下左右のずれ': '±4 画素', 'ToF の距離の雑音': 'σ 0.02',
                            '上 1/3 の明るさ(上ほど強く)': '±0.15(C1 で足す --aug-top-brightness)',
                            '左右の反転': '使わない(左回りのコースが別のコースになる)'},
        },
        'runs': runs,
        'arms': {
            'v2h-e30': {'stage': 1, 'seeds': [0, 1, 2, 3], 'decides': 'D1', 'train_samples': 6562},
            'v2m-e30': {'stage': 1, 'seeds': [0, 1, 2, 3], 'decides': 'D1', 'train_samples': 7757},
            '{x}-e60': {'stage': 2, 'seeds': [0, 1, 2, 3], 'decides': 'D2'},
            '{x}-e100': {'stage': 2, 'seeds': [0], 'decides': '参考だけ'},
            '{x}-aug2-e{E}': {'stage': 3, 'seeds': [0, 1, 2, 3], 'decides': 'D3'},
            '{x}-msf{A}-e{E}': {'stage': 4, 'seeds': [0, 1], 'decides': 'D4(2 本の平均の部品)。単体は参考'},
            'bfast-e15': {'stage': 'B-FAST', 'seeds': [0, 1, 2], 'decides': 'B-FAST の門 G-a〜G-f', 'train_samples': 6748},
        },
        'bfast_recipe': {
            'src': 'qualify_plan 1.5(P3)。plan.md 3.4 の案 1(封の 2 周を外す)と同じ',
            'init': 'モデル B(%s)の best.pth' % B_RUN,
            'human': 'clean_driving_20261011 の学習 41 シーン(5,912 サンプル)。B と同じ',
            'teacher_laps': [{'log': l, 'no': n, 't_start_s': t0, 'lap_s': s, 'samples': m} for l, n, t0, s, m in TEACHER_LAPS],
            'teacher_rule': '10/24 2 回目(16:01〜)の B のきれいな周で 9.0 秒未満(inventory.json の lap_list。who=ai・clean・suspect でない・model B)。'
                            'No.9 の 9 周 752 + No.10 の 1 周 84 = 836 サンプル。周の中の衝突の前後は train.py が外す',
            'not_included': ['新しい人の 650(v2 の分)', '10/23 の AI の周', '10/24 2 回目のほかの周・ほかのモデルの周', '封の 14 本の全部'],
            'labels': '車に出た値(build_ai_dataset の --label-gains eval --eval-steer-gain 1.0 --eval-throttle-gain 1.0。'
                      '既定のステア 1.3 のままにしない)。No.9 はゲイン 0.90 のあと、No.10 は 1.00 のあとの値',
            'weights': {'human': 1.0, 'teacher': 1.0,
                        'teacher_r1_window': 0.5,
                        'r1_window_rule': '手本の周のフレームで、R1 の起点 ta(tools/offline_rl.course_passes。desk_lib の r1_app と同じ)の前後 1.0 秒 [ta − 1, ta + 1) は重み 0.5',
                        'why': 'qualify_plan 1.5 の直し方 (2)「R1 の手前は人の正解を優先」。0.5 は plan.md 2.2 の R3(速めの周の重み)と同じ値。'
                               'No.9 は R1 の入口の板で 12 通過中 11 が 3 軸 2.5G 以上(最大 6.4G)なので、強く入る癖を写しにくくする'},
            'ai_share': '重みなし 836 ÷ 6,748 = 12.4%。重みつき(R1 の窓 0.5)は 12.4% 未満。R10(重みつき 15% 以下)を満たす',
            'mode': 'ジャンルの付け替えなし。手本の周は記録のジャンル 5(低速走行)のまま。11/3 もジャンル 5 で走る',
            'epochs': 15, 'lr': 1e-4, 'seeds': [0, 1, 2], 'aug': '色の揺らぎだけ',
            'gain_note': '手本の正解はゲイン 0.90 のあとの値。B と同じ表のゲインで走らせると、手本の周より少し弱くなりうる。'
                         'だから門 G-b は「同じゲインで B より前へ」を求める(同じ表で走って速くなるときだけ通す)',
            'independence': 'B-FAST の机上の結果は v2 の決まり(D1〜D5a)に使わない。v2 の結果も B-FAST の門に使わない。'
                            '両方が出会うのは 11/1 の 1 枠の選び(決めた順)だけ(plan.md 3.4)',
        },
        'tables': {
            'src': 'plan.md 2.3(T1 make_v2_tables.py が作る。名前は下のとおり。違う名前にしたら deviations.md に書く)',
            'dataset': {'name': 'dataset_v2_20261025',
                        'build': 'tools/build_ai_dataset.py --name dataset_v2_20261025 --human-from clean_driving_20261011 '
                                 '--ai-logs <下の 46 本> --val-ai-logs(空) --keep-ai-log-human --label-gains eval '
                                 '--eval-steer-gain 1.0 --eval-throttle-gain 1.0',
                        'ai_logs': TRAIN_AI_LOGS,
                        'stop_if': ['封の 14 本のどれかが source_scenes にある', '--ai-logs が省かれた',
                                    '検証(val)が H1 の 11 シーンと違う', '介入の窓(intervention_val_20261011 の 29)と学習のフレームが重なる']},
            'files': {'drop_v2h': TBL + 'drop_v2h.npz', 'drop_v2m': TBL + 'drop_v2m.npz', 'weights_v2m': TBL + 'weights_v2m.npz',
                      'remap_v2m': TBL + 'remap_v2m.npz', 'drop_bfast': TBL + 'drop_bfast.npz',
                      'weights_bfast': TBL + 'weights_bfast.npz', 'card': TBL + 'tables_card.json'},
            'v2h_keep': '10/10 人・学習 41 シーン 5,912 + 新しい人の「走りの区間」か「交代の直後 3 秒」650(plan.md 1.2 の数え方。plan_checks.py と同じマスク)',
            'v2m_keep': 'v2-H + B のきれいな周 9.0 秒未満 836(重み 1.0)+ 10/23 の 9.0〜9.5 秒 359(重み 0.5)。AI の周はジャンル 9 に付け替え',
            'v2m_half_laps': [{'log': l, 'no': n, 't_start_s': t0, 'lap_s': s, 'samples': m} for l, n, t0, s, m in V2M_HALF_LAPS],
            'genre9_entry': {'id': 9, 'name': 'AI・きれいな速い周',
                             'description': 'ゲインを焼き込んだ B のきれいな速い周(10/23・10/24 2 回目、9.5 秒未満)。学習用。収集には使わない。速さの手段ではない',
                             'src': 'plan.md 2.2 の R11・R12。data/modes.json に足すのは了承済み(main session から)。足したあとの sha256 は lock2'},
            'expected_after_build': {
                'v2h_train_samples': '6,562 ± 1%', 'v2m_train_samples': '7,757 ± 1%', 'bfast_train_samples': '6,748 ± 1%',
                'val_samples': '1,559(H1。変わらない)', 'v2m_ai_weighted_share': '13.4%(15% 以下)',
                'bfast_ai_samples': '836(± 1%)', 'genre9': 'AI の周だけ', 'sealed_in_source_scenes': 0, 'interv_overlap': 0,
                'if_off': 'どれかが外れたら学習を始めない。原因を deviations.md に書いて main session に聞く'},
        },

        # ------------------------------------------------------------ 保留と封
        'holdouts': {
            'H1': {'what': 'clean_driving_20261011 の保留 11 シーン(10/10 の人。昼 4・夜 7)', 'frames_desk': 4670,
                   'samples_train_log': 1559, 'genre_frames': NUM['h1']['genre_frames'],
                   'scenes': NUM['h1']['val_scenes'], 'scene_list_sha256': NUM['h1']['val_scene_list_sha256'],
                   'splits_json_sha256': NUM['h1']['splits_json']['sha256'],
                   'use': '主の判定。best の選び方(B と同じ)'},
            'interv': {'what': 'intervention_val_20261011(10/10 の交代の窓 29)', 'frames_desk': 1570,
                       'dataset_card_sha256': NUM['interv']['dataset_card']['sha256'],
                       'aligned_combined_sha256': NUM['interv']['aligned_combined_sha256'], 'use': '拒否権。立て直しの反応'},
            'day_night': {'day': [s for s in NUM['h1']['val_scenes'] if s[9:11] < '18'],
                          'night': [s for s in NUM['h1']['val_scenes'] if s[9:11] >= '18']},
        },
        'seal': {
            'what': '10/24 1 回目(13:46〜・昼・No.1〜4)と 3 回目(18:31〜・夕方・後輪の修理の後・No.11〜20)。記録ごと 14 本',
            'kind': '学習と選びの封。まだ見ていないデータではない(設計で 3 回目の上 1/3 の明るさ・3 回目の DE の衝突・1 回目 No.3 の止まりを見た)',
            'logs': SEALED,
            'per_log': {r['log']: {'session': s, 'labels_npz_sha256': r['labels_npz']['sha256'],
                                   'labels_json_sha256': r['labels_json']['sha256'],
                                   'raw_log_json_sha256': r['raw_log_json']['sha256'], 'n_scenes': r['n_scenes'],
                                   'scene_list_sha256': r['scene_list_sha256']}
                        for s, rows in NUM['seal'].items() for r in rows},
            'manifest_sha256': NUM['seal_manifest_sha256'],
            'manifest_format': '1 行 = "<log> <labels npz の sha256> <labels json の sha256> <raw の log.json の sha256> <シーンの一覧の sha256>"(上の順)。全体の sha256',
            'rules': ['学習に使わない(v2・B-FAST とも)。データの台本は source_scenes に封の記録があれば止まる',
                      'C3 は 2026-11-01 より前に封の記録を読もうとしたら止まる。開けるのは 11/1 の 1 回だけ(seal_opened.json を書く。あれば止まる)',
                      '開ける前に sha256 を照らす(ラベルが変わっていたら止めて main session に聞く)',
                      '開けたあとに足した腕・作り直したものは「参考」',
                      '11/3 のあと(v3)では学習に戻してよい'],
            'open_20261101': {
                'T': {'role': '拒否権', 'what': '交代 28 回の窓(make_intervention_set と同じ作り)の L1(best)',
                      'rule': 'seal_t_l1_us − 5 本の平均(封の上) > 2 × 5 本の標準偏差(封の上)→ その候補を外す'},
                'H': {'role': '参考', 'what': '人の走り・交代の直後 197 サンプルの L1'},
                'A': {'role': '参考', 'what': 'B のきれいな周 1,684 の、車に出た値との差'},
                'V': {'role': '参考', 'what': '3 回目の直線のスロットルの予測(候補 − B)'},
                'D': {'role': '参考', 'what': '3 回目の DE の衝突 25 回の手前 1 秒・DE の通過'},
                'S': {'role': '参考', 'what': '1 回目の止まった場面(上限 57 フレーム。300 未満なので門にしない)'},
                'GB_seal': {'role': '参考', 'what': 'B-FAST の G-b を 3 回目の No.11・No.12(B 0.90)でも出す(qualify_plan 1.5「最後に 1 回だけ」)'},
                'fields': ['seal_t_l1_us', 'seal_h_l1_us', 'seal_a_us', 'seal_v_thr_vs_b_us', 'seal_de_pre_thr_vs_b_us',
                           'seal_de_overshoot_vs_b_us', 'seal_standstill_go_frac', 'seal_bfast_gb_fwd_vs_b_us'],
                'predict': ['C1', 'C2', '予備(2 本の平均)', 'BF1', 'BF2', 'B', '5 本(A_s0・A_s1・v0-s2・v0-s3・B)'],
            },
        },

        # ------------------------------------------------------------ 指標
        'sign_rules': {
            '_thr_': 'PWM の差 [µs]。+ = 緩める(前進は PWM が小さい。中立は各モデルの model_config の throttle_neutral_us)',
            '_fwd_': '前進の量の差 [µs] = −(PWM の差)。+ = 強く踏む',
            'steer': 'ステア [µs]。+ = 左',
            'car_output': '「車に出る」= drive.apply_gains と同じ式(steer_trim 0・ステアのゲイン 1.0)。ref は各モデルの model_config の値',
            'gains': 'B 0.90・候補のジャンル 5 0.90・B-FAST 0.90(B と同じ表)・ジャンル 9 1.00(正解がゲインの後のため)',
            'row': '行 0 = 1 フレーム先、行 2 = 走る日の先読み 3(desk_lib の LOOKAHEAD_ROW)',
            'weights_for_fields': '予測は PyTorch FP32 の best と last(欄の名前に _best・_last。無いものは best)。推論の速さと sanity はエンジン(走行と同じ run_numpy)',
        },
        'sets': {
            'H1': '4,670 フレーム(上)', 'interv': '1,570 フレーム(上)',
            'zones': 'H1 を R1_app・R1_turn・GM・DE・none・unknown に分ける(10/22 の zones2。desk_inputs_20261022 の zone と同じ)',
            'shift_tests': 'H1 に 7 つ: 画像 3 行上・3 行下、明るさ ×0.8・×1.2、ToF 1 行下、上 1/3 ×0.85・×1.15',
            'ai_scenes': '学習の回(10/10・10/23・10/24 2 回目)の AI の衝突(水平 2.5G)の手前 1 秒 [tc − 1, tc)、場所 R1・DE・SW(ラベルの loc)。207 回。衝突の前後は学習に入らない',
            'de_pass': '学習の回の DE の通過のうち、学習に使っていない周だけ(きれいでない周・B の 9.5 秒以上・ほかのモデル)。入って 0.5 秒',
            'straight': '学習の回だけの、学習に使っていない AI のきれいな周(B の 9.5 秒以上・ほかのモデル)の直線(eRPM < −1000・|ヨー| < 40 deg/s)',
            'standstill': '学習の回の AI 側・dry run でない・|eRPM| < 400・衝突の前後でない・前に立った物が 1 m 以内に無い'
                          '(jetformer/tof_guard.ToFGuard(data/tof_floor_ref.npy).features(tof, erpm 0, gz 0, ステア中心) の x_free > 1.0 m)。上限 1,622 フレーム',
            'sanity': '10/10 の AI 走行 21,759 フレーム(ai_mix_20261011 の source == 1 & vehicle_valid)+ 10/24 2 回目の AI のフレーム',
            'gb_main': 'B-FAST の G-b・G-c: 10/23 公式の B の 3 本(No.8 20261023-210840・No.11 20261023-211209・No.15 20261023-211547)の AI 側のフレーム(dry run でない・衝突の前後でない・間引きなし)',
            'gb_ref': '参考: 10/24 2 回目の記録で B-FAST の手本の 10 周の外のフレーム(No.5〜8 の全部・No.9 と No.10 の手本の周の外)',
        },
        'fields_desk_bench_same': {
            'src': 'desk_bench(10/21・10/22)と同じ欄・同じ定義(experiments/20261021_bench_prep/desk/desk_lib.py)',
            'list': ['l1_best_us', 'l1_steer_best_us', 'l1_thr_best_us', 'l1_last_us', 'l1_interv_us', 'r1_app_thr_diff_us',
                     'nonzone_thr_diff_us', 'steer_amp_ratio', 'brake_lowspeed_frac', 'trt_median_ms', 'trt_p99_ms',
                     'engine_ok', 'sanity_ok', 'gates_pass'],
        },
        'fields_added': {
            'l1_last_us・l1_interv_last_us': 'last の重みで同じ',
            'l1_zone_<区間>_us': 'H1 の区間ごとの L1(best)',
            'l1_day_us・l1_night_us': 'H1 の昼 4 シーン・夜 7 シーンの L1(best)',
            'l1_genre4_us・l1_genre5_us': 'H1 のジャンル 4(1,922 フレーム)・5(2,748)の L1(best と last)。印「ジャンルの片寄り」に使う',
            'steer_bias_us・thr_bias_us': 'H1 の行 0 の 予測 − 人 の平均(best と last)',
            'robust_dl1_us': 'ずれの試し 7 つの L1 の増え(試し − そのまま)の平均(best)',
            'de_pre_thr_vs_b_us・r1_pre_thr_vs_b_us・sw_pre_thr_vs_b_us': 'AI の場面の手前 1 秒の、行 2 の車に出るスロットル(候補 − B)。+ = B より緩める',
            'de_overshoot_vs_b_us': 'DE の通過(入って 0.5 秒)の行 0 の予測ステア(候補 − B)の平均。+ = 左 = 右への行き過ぎが小さい(参考)',
            'straight_thr_vs_b_us': '直線の集合の、行 2 の車に出るスロットル(候補 − B)。0 に近いほどよい',
            'standstill_go_frac': '止まった場面で、行 0 の車に出るスロットルが 中立 − 30 µs より前へ出す割合',
            'g9_minus_g5_pre_thr_us': 'v2-M だけ: 同じモデルのジャンル 9(ゲイン 1.00)− ジャンル 5(0.90)の、AI の場面の R1・DE の手前 1 秒の行 2 の車に出るスロットル。'
                                      '− = ジャンル 9 が強く踏む',
            'bfast_gb_fwd_vs_b_us': 'B-FAST だけ: gb_main の DE・SW・HS(ラベルの loc_detail)のフレームの、行 2 の車に出る前進の量(候補 − B、どちらもゲイン 0.90)の平均。'
                                    'DE・SW・HS ごとの値(_de・_sw・_hs)も出す',
            'bfast_gc_fwd_vs_b_us': 'B-FAST だけ: gb_main の R1 の起点の手前 1 秒 [ta − 1, ta) のフレームの、同じ前進の量の差',
            'bfast_gb_ref_fwd_vs_b_us': 'B-FAST だけ(参考): gb_ref の DE・SW・HS で同じ',
            'bfast_gb_fwd_g100_vs_b090_us': 'B-FAST だけ(参考): B-FAST をゲイン 1.00、B を 0.90 で比べた G-b の値(qualify_plan 1.5 の「B-FAST の 1.0 が No.9 の 0.90」)',
            'bfast_de_line_vs_b_us': 'B-FAST だけ(参考): gb_main の DE の通過で、入って 0.5〜0.7 秒の行 0 のステア(候補 − B)の平均。− = 左へ切り返さない(No.9 の速い周の線 −22 µs、3 回目 +60 µs)',
            'straight_fwd_vs_b_us': '= −straight_thr_vs_b_us(前進の向き)',
            'lap_delta_est_s・lap3_est_s': '速さの見込み(参考・門にしない): lap_delta_est_s = %.5f × straight_fwd_vs_b_us [秒/周]、'
                                           'lap3_est_s = %.2f + 3 × lap_delta_est_s。' % (NUM['speed_est']['lap_delta_s_per_fwd_us'], NUM['speed_est']['no9_window3_median_s']) +
                                           '係数 = qualify_plan 1.1 の式(1 周 ≈ 14.52 − 1.50 × p90 [m/s])× 0.5e-3 m/s/eRPM × No.9 の電池の効き 35.95 eRPM/µs。'
                                           '25.75 は No.9 の 3 周の窓の中央値(走りながら・満充電・修理の前の車)。机上の順位で実車の順位は決めない',
            'ref_values': 'model_config の steer_center_us・throttle_neutral_us(B は 1441.0・1496.0)',
        },
        'stage0_checks': {
            'when': '10/25 18:00〜(段 0)。最初の v2・B-FAST の予測の前。答えは lock2 に書く',
            'list': [
                '1. B を予測し直し、10/12 の保存(data/zoo/policy_zoo_20261012/pred/B.npz・B_last.npz)と同じか(差の最大 ≤ 0.01 µs)。違えば止める',
                '2. 10/24 2 回目の AI のフレームで、B の予測(PyTorch)と記録の model_out(走行中のエンジン)の差の中央値。10 µs を超えたら止めて調べる',
                '3. 印「ジャンルの片寄り」の閾値: この protocol で固定した(下の marks。5 本の幅 × 2、保存した予測から計算済み)。段 0 で C3 が同じ値を出すかだけ確かめる(差 ≤ 0.01 µs)',
                '4. 止まった場面のフレーム数(sets.standstill)。300 以上なら門、未満なら参考。B と 2 本の平均(bench_20261021-ens-p2)の standstill_go_frac をここで測る',
                '5. データセットと表の確かめ(tables.expected_after_build)',
                '6. 下見: v2-M の軽量型・多段型・AUG2・B-FAST(--init)を 20 ステップ(--max-steps 20)。時間・メモリ(使ったメモリの最大)・読み込み・--init がジャンルの数で落ちないか',
                '7. dataset_v2 の steer_center_us・throttle_neutral_us(train_data.action_reference)。B の 1441.0・1496.0 と ±2 µs を超えて違えば、11/3 に出す前にユーザーに聞く(机上はそれぞれの model_config の値で計算)',
                '8. B の机上の推論の p99 を同じ日に測る(10/21 は 2.487 ms)。門「推論」はその日の B の 1.3 倍',
            ],
        },

        # ------------------------------------------------------------ 閾値と判定
        'thresholds_us': {
            'rule': '閾値 = 2 × 標準誤差。標準偏差 = B と同じ作り方の 5 本(A_s0・A_s1・v0-s2・v0-s3・B)の desk の H1・介入の値。'
                    '腕 n 本 対 5 本: 2·sd·√(1/n + 1/5)。同じ種 k 組の差の平均: 2·sd·√(2/k)。種どうしは独立とみなす(plan.md 6.3)',
            'values': t,
            'recount': '別のコード(protocol_numbers.py、保存した予測から)で数え直し、plan_checks.json と 0.002 µs 以内で一致(%d 項目)' % len(NUM['plan_checks_compare']),
            'five_mean_in_use': '判定で使う 5 本の平均は、C3 が同じフレームで保存した予測から計算した値(丸めない)。上の five_mean はその丸め',
            'not_worse_us': 0.5,
        },
        'judgement': {
            'vs5': {'src': 'plan.md 6.4',
                    'good': '腕の平均 − 5 本の平均 ≤ −thr_vs5(n)、かつ腕の全部の種が 5 本の平均より小さい。best と last の両方で成り立つときだけ',
                    'bad': '腕の平均 − 5 本の平均 ≥ +thr_vs5(n)、かつ全部の種が 5 本の平均より大きい。best か last のどちらかで成り立てば',
                    'same': 'ほか',
                    'metric': 'H1 の L1(l1_best_us・l1_last_us)。n = その腕の種の数(4・3・2・1)'},
            'pairs': {'src': 'plan.md 6.4', 'd': 'd_k = 腕の種 k − 基準の腕の種 k(同じ種どうし)',
                      'not_worse': 'd の平均 ≤ +0.5 µs(best と last の両方)',
                      'good': 'd の平均 ≤ −thr_pairs(k)(best と last の両方)、かつ全部の d_k < 0(best と last の両方)',
                      'bad': 'd の平均 ≥ +thr_pairs(k)、かつ全部の d_k > 0。best か last のどちらかで(D2 だけは last だけで見る)',
                      'same': 'ほか'},
            'missing_seed': '学習が止まった・削った種は数に入れない。n・k はそろった種の数で、その数の閾値を使う。同じ種の組は両方そろった種だけ',
        },
        'vetoes': {
            'src': 'plan.md 6.6',
            'interv': '腕の l1_interv(best)の平均 − 5 本の平均(128.594)> thr_vs5(n).interv_best、かつ全部の種が 5 本の平均より大きい → その腕を候補にしない',
            'bias': '腕の steer_bias(best)の平均が 5 本の平均(+0.132)± thr_vs5(n).steer_bias_best の外(n = 4 で ±3.913)→ 候補にしない。'
                    'v2-M は同じ種の v2-H との差の平均で ±%.3f(k = 4)の外' % t['pairs']['k4']['steer_bias_best'],
            'seal_T': '11/1: seal_t_l1_us − 5 本の平均(封の上)> 2 × 5 本の標準偏差(封の上)→ その候補を外す',
            'when': '段ごと(腕がそろったとき)と 11/1。B-FAST の腕(n = 3)にも同じ式で掛ける',
            'effect': '拒否権のある腕は、その段の答えに選ばない(D1 なら v2 を使わない、D2 なら 30、D3 ならなし)。最終の作り方の腕に拒否権があれば v2 の候補は無し',
        },
        'marks': {
            'src': 'plan.md 6.4 の印(学習は止めない)',
            'rise': '参考: train.log の検証の L1 の 5 エポックの移動平均が、それまでの最小より 1.0 µs 以上高い状態が 10 エポック続く',
            'too_long': '長すぎ: desk の l1_last_us − l1_best_us > 1.6 µs',
            'genre_skew': 'ジャンルの片寄り(60 と同じ種の 30、last): d4 = l1_genre4(60) − l1_genre4(30)、d5 = 同じくジャンル 5。'
                          '(d4 > %.3f かつ d5 ≤ %.3f)か(d5 > %.3f かつ d4 ≤ %.3f)で印' % (
                              gm['last_genre4']['thr_2x_range'], gm['last_genre5']['thr_2x_range'],
                              gm['last_genre5']['thr_2x_range'], gm['last_genre4']['thr_2x_range']),
            'genre_skew_threshold_us': {'genre4_last': gm['last_genre4']['thr_2x_range'], 'genre5_last': gm['last_genre5']['thr_2x_range'],
                                        'genre4_best_ref': gm['best_genre4']['thr_2x_range'], 'genre5_best_ref': gm['best_genre5']['thr_2x_range'],
                                        'rule': '5 本のジャンル別の H1 の L1 の幅(最大 − 最小)× 2(plan.md 6.2 の 3)。保存した予測から学習の前に計算した'},
        },
        'decisions': {
            'D1': {'when': '10/26 朝', 'what': 'データ',
                   'rule': ['① v2h-e30(4 本)が 5 本に対して「悪い」でなく、拒否権が無い → v2-H を軸(x = v2h)。'
                            '「悪い」か拒否権 → v2 を使わない(x = base。段 2 以降は B のデータ clean_driving_20261011)',
                            '② ①が v2-H のときだけ: v2m-e30 が同じ種の v2h-e30(k = 4)に対して全部を満たせば x = v2m。'
                            'H1 が「悪くない」・steer_bias の差の平均が ±%.3f の中・l1_interv の差の平均 ≤ +%.3f(best)・thr_bias の差の平均 ≥ −%.3f(best)・'
                            'H1 か介入で「良い」(介入の閾値 best %.3f・last %.3f)。どれかが欠けたら v2h'
                            % (t['pairs']['k4']['steer_bias_best'], t['pairs']['k4']['interv_best'], t['pairs']['k4']['thr_bias_best'],
                               t['pairs']['k4']['interv_best'], t['pairs']['k4']['interv_last']),
                            '③ v2-M にも拒否権(介入・偏り)を掛ける。あれば v2h'],
                   'after': 'x が v2h 以外なら、段 2 を x でやり直す(先回りの v2h-e60・e100 は参考)'},
            'D2': {'when': '10/26 朝', 'what': 'エポック',
                   'rule': '{x}-e60 と {x}-e30 の同じ種 4 組(last)。60 が「悪い」(d の平均 ≥ +%.3f かつ全部の d_k > 0)・'
                           '「長すぎ」の印が 2 本以上・「ジャンルの片寄り」の印が 2 組以上・60 の腕に拒否権、のどれか → 30。それ以外は 60。'
                           '100 は参考で選ばない' % t['pairs']['k4']['l1_last'],
                   'note': 'x = base のときの 30 の種 0〜3 は、B と同じ作り方の A_s0(種 0)・A_s1(1)・v0-s2(2)・v0-s3(3)(新しく学習しない)。x = v2h・v2m の 30 は段 1'},
            'D3': {'when': '10/26 夕(段 4 の前)', 'what': '拡張',
                   'rule': '{x}-aug2-e{E} と拡張なしの {x}-e{E}(同じ種 4 組)。全部を満たせば AUG2: H1 が「悪くない」・'
                           'robust_dl1_us が 4 組とも AUG2 < なし、かつ 平均(AUG2)÷ 平均(なし)≤ 0.8・'
                           'l1_day_us の差の平均 ≤ +1.0 かつ l1_night_us の差の平均 ≤ +1.0・AUG2 の腕に拒否権が無い。それ以外はなし'},
            'D4': {'when': '10/27 朝', 'what': '2 本の平均(予備)',
                   'rule': '軽量型 種 k(最終の作り方)+ 多段型 種 k の予測の平均(k = 0・1)を、同じ種の軽量型単体と比べる(best だけ。2 本の平均に last は無い)。'
                           'd の平均 ≤ −%.3f かつ両方の d_k < 0、かつ門(推論を除く)を全部通る → 予備。主の候補にしない。11/3 の 1 枠にも入れない'
                           '(走行と同じ負荷で推論を測っていないため。plan.md 6.5)。多段型単体は参考' % t['pairs']['k2']['l1_best']},
            'D5a': {'when': '10/27 朝(1 日遅れても 10/28)', 'what': 'v2 の候補の並び',
                    'rule': '最終の作り方 = D1 の x × D2 のエポック × D3 の拡張、軽量型、ジャンル 5。その腕の種を 0 → 1 → 2 → 3 の順に見て、'
                            '門(gates_v2)に全部合格した最初 = C1、次 = C2。腕に拒否権があれば C1・C2 は無し。H1 で種を選ばない。封はまだ開けない',
                    'genre9': 'x = v2m のときもジャンル 5 を先に出す。ジャンル 9 は門 g9 を通ったときだけ、時間が残ったら 1 本(plan.md 7.3)'},
            'BF': {'when': '10/26 朝〜10/27 朝(エンジンのあと G-e)', 'what': 'B-FAST の候補',
                   'rule': '種を 0 → 1 → 2 の順に見て、門(gates_bfast)に全部合格した最初 = BF1、次 = BF2。B-FAST の腕(n = そろった種の数)に拒否権があれば無し'},
            'D5b': {'when': '11/1 朝', 'what': '11/3 の新しいモデルの 1 枠', 'rule': '下の slot_20261103'},
            'order_note': '机上の順位で実車の順位は決めない(plan.md 6.4)',
        },
        'gates_v2': {
            'src': 'plan.md 6.5(候補ごと。全部に合格しないと 11/3 に出さない)',
            'engine_ok': 'engine_check.json が今のエンジンのもの・TensorRT と PyTorch の差 ≤ 2.5 µs',
            'sanity_ok': 'sets.sanity で NaN・Inf なし・500〜2500 µs・check_io',
            'inference': 'trt_p99_ms ≤ 1.3 × 同じ日に測った B の trt_p99_ms(10/21 の B は 2.487 ms → 3.23 ms)',
            'amp': 'steer_amp_ratio 0.80〜1.25',
            'l1_vs_b': '|l1_best_us − B の l1_best_us| < 10 µs',
            'brake': 'brake_lowspeed_frac ≤ 0.01(10/21 の B は 0.0082)',
            'standstill': 'standstill_go_frac ≥ B の値 − 0.10(止まった場面が 300 フレーム以上のとき。未満なら参考)',
            'human_throttle': 'thr_bias_us(best)≤ 5 本の平均(11.882)+ %.3f = %.3f µs(PWM の + = 緩める。新しい人の遅い走りでスロットルが下がっていないか)'
                              % (t['pairs']['k2']['thr_bias_best'], 11.882 + t['pairs']['k2']['thr_bias_best']),
            'straight': '|straight_thr_vs_b_us| ≤ 10 µs(10/23 の 2 本の平均の「速すぎ」と、遅すぎの両方)',
            'genre9': 'x = v2m のジャンル 9 だけ: g9_minus_g5_pre_thr_us ≥ −5 µs(ジャンル 9 が 5 より 5 µs を超えて強く踏まない)。だめならジャンル 9 を出さない(ジャンル 5 だけ)',
        },
        'gates_bfast': {
            'src': 'qualify_plan 1.5 の G-a〜G-e。G-f と安全の門は、この protocol で足した(理由は下)',
            'G-a': 'l1_best_us ≤ B の l1_best_us(49.144)+ 1.0 = 50.144 µs(H1 の 11 シーン)',
            'G-b': 'bfast_gb_fwd_vs_b_us ≥ +10 µs(gb_main の DE・SW・HS、同じゲイン 0.90)。手本の No.9 の遅い周は学習の中なので使わない。封の No.11・12 は 11/1 に参考',
            'G-c': 'bfast_gc_fwd_vs_b_us ≤ +5 µs(gb_main の R1 の手前 1 秒)',
            'G-d': 'standstill_go_frac ≥ B の値 − 0.10(止まった場面が 300 フレーム以上のとき。未満なら参考にし、11/3 の 3-1 で「止まってから 5 秒以内に出る」を確かめる)',
            'G-e': 'engine_ok(TensorRT と PyTorch の差 ≤ 2.5 µs、今のエンジン)',
            'G-f': 'straight_fwd_vs_b_us < (B をゲイン 1.00 にしたときの直線の前進の量 − 0.90 のときの量)= 0.10 × B の生の前進の量の平均(同じ直線の集合)。'
                   'B のゲインを 0.90 → 1.00 にしたのと同じ大きさを超えて直線で踏まない(10/24 2 回目のゲイン 1.0〜1.1 は満充電で SW の出口に 9 回当たり、No.10 で後輪が壊れた)',
            'safety_same_as_v2': ['sanity_ok', 'inference', 'amp', 'brake'],
            'not_applied': 'v2 の門「人の走りのスロットル」「直線 ±10 µs」は B-FAST に掛けない(G-b が DE・SW・HS で踏むことを求めるため。行き過ぎは G-f で見る)。値は参考に出す',
            'why_added': 'ユーザーの「ぶつからない」。G-f は直線の行き過ぎ、安全の門は 10/21 から全部の候補に掛けている最低限',
        },

        # ------------------------------------------------------------ 11/3 の 1 枠
        'slot_20261103': {
            'src': 'plan.md 7.1・6.4 の D5b・qualify_plan 3.8・7 章。plan.md 7.1 は「B-FAST と 1 つにまとめる。どちらを出すかは 11/1 に同じ封の確かめで決める」。その決め方をここで固定する',
            'candidates': ['BF1', 'BF2', 'C1', 'C2'],
            'not_in_slot': ['2 本の平均(予備)', '多段型単体', 'e100', '先回りでやり直した腕', 'ジャンル 9(時間が残ったときの別の 1 本)'],
            'rule': ['1. 11/1 朝に封を 1 回だけ開け、seal.open_20261101.predict の全部を予測する',
                     '2. 封の T の拒否権に落ちた候補を外す',
                     '3. 残った候補を BF1 → BF2 → C1 → C2 の順に見て、最初の 1 つを 11/3 の新しいモデルの 1 枠にする',
                     '4. 全部落ちたら枠は無し(回 3 は B と P1・P2 のまま)',
                     '5. 開けたあとに作り直し・足し直しはしない'],
            'why_bfast_first': 'B-FAST だけが周の時間の仮説を門で確かめている(G-b: 同じゲインで DE・SW・HS を B より 10 µs 以上踏む。'
                               '直線で +10 µs は 1 周 約 0.27 秒の見込み)。v2 の H1 の見込みは −0.3〜−0.6 µs で種の揺れより小さく、'
                               '25 秒への効きは机上で示せない(plan.md 0 章)。どちらも門と拒否権を通ったものだけ',
            'on_track': {
                'where': 'qualify_plan 7 章の回 3(パック C)の「新しいモデル」の枠。P1・P2 のあと',
                'settings': 'B と同じ: P1 の電圧の表のゲイン(B-FAST も同じ表)・ステア 1.00・lookahead 3・start_kick 1。P2 が使えるなら天井 ON',
                'order': '1 本目 = 表のゲイン − 0.05(安全の確かめ。強い衝突 0 で次へ)→ 2 本目 = 表どおり → B → 新しいモデル(A/B/A)。1 本 4〜6 周。各本で電圧(前・後)を書く',
                'stop': 'qualify_plan 3.7 と同じ(4G 以上 2 回・8G 以上 1 回・R1 の 3 軸 7G 以上 1 回か本の中央 4.5G 以上・同じ場所で 2 本続けて)。止まって 5 秒出ないときはその本をやめる',
                'adopt': 'qualify_plan 3.8 の P3 と同じ: 2 本続けて、3 周の窓の中央値が P1・P2 の良い方より 0.3 秒以上速く、DE と R1 の衝突/通過が増えていない。満たさなければ B のまま',
                'approval': 'data/bench_candidates.json に入れるのも、走らせるのもユーザーの了承のあと',
            },
        },

        # ------------------------------------------------------------ 進め方
        'gpu_order': {
            'src': 'plan.md 5 章。夜は次の朝に決めるのに要るものを先に、長いものを最後に',
            'order': ['段 0: 下見(20 ステップ)・物差しの固定(B・5 本・2 本の平均を C3 で予測)→ lock2',
                      '段 1: v2h-e30-s0 → v2m-e30-s0 → v2h-e30-s1 → v2m-e30-s1 → … → s3(同じ種の組が先にそろう順)',
                      'B-FAST: s0 → s1 → s2',
                      '段 2: {x}-e60-s0 → s1 → s2 → s3 → {x}-e100-s0(先回りで x = v2h)',
                      '(10/26 朝 D1・D2)',
                      '段 3: {x}-aug2-e{E}-s0 → s3(D1 が v2h 以外なら、段 2 のやり直しを先に)',
                      '(10/26 夕 D3)',
                      '段 4: {x}-msf{A}-e{E}-s0 → s1',
                      '(10/27 朝 D4・D5a・BF)'],
            'gap': '学習の間に 60 秒あける。ほかの重い仕事(別の workflow の pytest など)に順番を回す',
            'eval_hours': '机上 3 回・エンジン・確認・sanity・机上の全部・封 = 約 4.5 時間(plan.md 4.2)。B-FAST の机上・エンジン 3 本を足す',
        },
        'timing_estimate_hours': tim,
        'start_conditions': {
            'src': 'plan.md 5 章',
            'power': '夜の前に、Jetson が AC(または満充電)であることをユーザーが確かめた。確かめられない夜は、100 エポックと多段型を後ろに回す',
            'disk': '空き 5 GB 以上(10/25 朝 14 GB)。自分で作った last.pth は、予測を保存して確かめたあと消す(候補に残るものは消さない)',
            'memory': '段 0 の下見の最大で決める(common_recipe.workers・memory_retry)',
            'one_at_a_time': '学習・エンジン・机上の予測・pytest・Firefox の確かめは全部 /home/jetson/bin/jf-heavy --lock big(GPU は同時に 1 つ)',
            'deadline': '4.2 の式 × 1.2 で 10/31 06:00 を超える学習は始めない',
        },
        'resume': {
            'src': 'plan.md 5 章',
            'rule': ['train.py は毎エポックの終わりに last.pth を書く。--resume は前回と引数が同じかを確かめて続きから(失うのは最大 1 エポック)',
                     'キュー v2: 済み(result.json が done・予定のエポック・best.pth・ONNX)は飛ばす。last.pth があれば --resume。読めない last.pth は最初から',
                     '状態ファイルは tmp → fsync → 置き換え。途切れた試みは失敗に数えない。失敗は 1 回だけやり直す',
                     '電源が戻ったら、同じ 1 コマンドをもう一度動かす(自動の起動はユーザーの了承のあと)',
                     '再開は完全には同じにならない(シャッフル・拡張の乱数)。deviations.md に書く。判定はそのまま使う'],
        },
        'cut_order': {
            'src': 'plan.md 5 章 + B-FAST を入れた',
            'order': ['① 任意の腕(S1 机上・X+P・TCP型 × X)', '② {x}-e100', '③ 段 4(多段型)', '④ B-FAST の種 2',
                      '⑤ 段 3 の種 2・3', '⑥ 段 2 の種 2・3', '⑦ B-FAST の種 1'],
            'keep': '段 1 の 8 本・段 2 と段 3 の種 0・1・B-FAST の種 0・エンジン・確かめ・机上のベンチ・封の確かめ',
            'effect': '削った種は数に入れない(judgement.missing_seed)。削ったことは deviations.md に書く',
        },
        'optional_arms': {
            'src': 'plan.md 4.2(了承と時間が残ったときだけ。答えは参考。11/3 に出さない)',
            'list': ['S1 机上(過去 10 秒だけの k・v2-H・30 × 種 2)', 'X+P(駐車 1〜3 を足す・30 × 2)', 'TCP型 × X(30 × 2)'],
        },
        'interpretations': {
            'note': 'plan.md・qualify_plan であいまいだった所を、学習の前に分かっていることだけで決めた。main session が違う決め方にするなら、学習の前に deviations.md に書いて版を上げる',
            'list': [
                '学習の数: 依頼の 20 本 = v2 19 本 + B-FAST 1 腕。B-FAST は種 3 本なので学習は 22 本',
                '門「人の走りのスロットル」: plan.md 6.5 は「11.9 − 4.1 µs より下がらない」。thr_bias は + = 緩めるなので、遅くなる向きを止めるには上限(≤ 11.882 + 4.082)。上限にした',
                '門「ジャンル 9」: g9_minus_g5_pre_thr_us は PWM の差(+ = 緩める)にそろえ、「≥ −5 µs」(= ジャンル 9 が 5 µs を超えて強く踏まない)と書いた。plan.md の「+5 µs 以下」と同じ意味',
                '印「ジャンルの片寄り」の閾値は段 0 でなく、いま保存した予測から決めた(同じ決まり。結果の前)。D2 で「2 組以上」なら 30',
                '「長すぎ」の last − best は desk の H1(4,670 フレーム)で見る。「上がり」は参考だけ',
                'B-FAST の門 G-b・G-c は同じゲイン(0.90 どうし)で比べる。qualify_plan 7 章の 3-2 が B と同じ表で走るため。'
                '1.00 対 0.90 で比べると、何も覚えなかった B-FAST でも「ゲインを 0.1 上げた B」として約 +20 µs で通ってしまう(それは走行タブのゲインでできること。'
                '10/24 2 回目のゲイン 1.0〜1.1 は SW の出口に当たった)。だから 1.00 対 0.90 の値は参考だけ。'
                '見込み: 手本の正解はゲイン 0.90 のあとの値なので、同じゲインでは B より踏まない見込みが大きく、G-b に落ちやすい。落ちたら 1 枠は v2 か無し',
                'B-FAST の R1 の前後 1 秒の重みは 0.5(qualify_plan 1.5 の「下げる」。値は R3 と同じ)',
                'B-FAST に使う重みは best.pth(B と同じ。過去の学び直し 8 本(lr 1e-4・12 エポック)の best は 4〜10 エポック目で、1 エポック目に寄らない)',
                'B-FAST に G-f と安全の門を足した(gates_bfast.why_added)',
                '11/3 の 1 枠は BF1 → BF2 → C1 → C2 の順(slot_20261103.why_bfast_first)',
                '2 本の平均(予備)は 1 枠に入れない(plan.md 6.5 の推論の門。D5b の「予備」は封の上の記録だけ)',
                '止まった場面の「前の ToF が 1 m より遠い」= ToFGuard.features の x_free > 1.0 m',
                '閾値: plan_checks.json にある値はそれ(plan.md 6.3 はその丸め)。n = 1・3 と k = 3 は同じ式で protocol_numbers.json から',
            ],
        },
        'deviations': {
            'file': str(REP / 'deviations.md'),
            'rule': 'この決まりから外れたことは全部、時刻・何を・なぜ・判定への影響を書く。変えた腕は判定から外して参考に下げる(plan.md 6.7)',
            'recount': '決まりの計算は、別のコードで数え直す(10/21・10/22 の recount と同じ)。台本は /home/jetson/jetformer/experiments/20261025_train_v2/ に置く',
        },
        'files_out': {
            'desk': str(REP / 'desk' / 'desk_v2.json') + '(C3 が書く。欄の名前はこの protocol のとおり)',
            'decisions': str(REP / 'decisions.json') + '(D1〜D5b・BF の答えと、その計算に使った数)',
            'seal_opened': str(REP / 'seal_opened.json') + '(11/1 に 1 回だけ)',
        },
        'numbers_sha256': {'protocol_numbers.json': None},
        'defaults_untouched': {
            'rule': '既定値(drive.py・推奨モデル drive_recommended.json・drive_settings.json)を変えない。data/raw・data/labels を変えない',
            'sha256_now': {Path(d['path']).name: d.get('sha256') for d in NUM['defaults_now']},
        },
        'model_b_sha256': {k: v.get('sha256') for k, v in NUM['model_b_files'].items()},
        'reference_predictions_sha256': {k: ({'best': v['pred_best']['sha256'], 'last': v['pred_last']['sha256'], 'run': v['run']}
                                             if 'pred_best' in v else v.get('sha256'))
                                         for k, v in NUM['reference_predictions'].items()},
    }
    import hashlib
    proto['numbers_sha256']['protocol_numbers.json'] = hashlib.sha256(open(REP / 'protocol_numbers.json', 'rb').read()).hexdigest()

    # ---- 自分で確かめる
    names = [r['name'] for r in runs]
    assert len(runs) == 22 and len(set(names)) == 22, '学習の本数・名前'
    assert sum(1 for r in runs if r['family'] == 'v2') == 19 and sum(1 for r in runs if r['family'] == 'B-FAST') == 3
    assert len(SEALED) == 14 and len(set(SEALED)) == 14
    allai = [l for v in TRAIN_AI_LOGS.values() for l in v]
    assert len(allai) == 46 and not set(allai) & set(SEALED), '学習の AI 走行に封が入っている'
    assert not {l for l, *_ in TEACHER_LAPS} & set(SEALED) and not {l for l, *_ in V2M_HALF_LAPS} & set(SEALED)
    assert sum(m for *_, m in TEACHER_LAPS) == 836 and sum(m for *_, m in V2M_HALF_LAPS) == 359
    assert 5912 + 836 == 6748
    inv = json.load(open(PLAN / 'inventory.json', encoding='utf-8'))
    want = {'10/10': ('10/10 1 回目(13:54〜)', '10/10 2 回目(19:21〜)'), '10/23': ('10/23 1 回目(20:37〜)',),
            '10/24 2 回目': ('10/24 2 回目(16:01〜)',)}
    for k, sess in want.items():
        got = sorted(r['log'] for r in inv['records'] if r['session'] in sess and r['ai_log'] and r['usable'])
        assert got == sorted(TRAIN_AI_LOGS[k]), '学習の AI 走行が棚卸しと違う: %s' % k
    laps = []
    for r in inv['records']:
        for l in r.get('lap_list') or []:
            if l['who'] == 'ai' and l['clean'] and not l.get('suspect') and l['model'] == 'B' and r['session'] == '10/24 2 回目(16:01〜)' and l['lap_s'] < 9.0:
                laps.append((r['log'], r['ai_no'], round(l['t_start'], 3), round(l['lap_s'], 3), l['samples']))
    assert sorted(laps) == sorted(TEACHER_LAPS), '手本の周が棚卸しと違う'
    half = []
    for r in inv['records']:
        for l in r.get('lap_list') or []:
            if l['who'] == 'ai' and l['clean'] and not l.get('suspect') and l['model'] == 'B' and r['session'] == '10/23 1 回目(20:37〜)' and 9.0 <= l['lap_s'] < 9.5:
                half.append((r['log'], r['ai_no'], round(l['t_start'], 3), round(l['lap_s'], 3), l['samples']))
    assert sorted(half) == sorted(V2M_HALF_LAPS), 'v2-M の 0.5 の周が棚卸しと違う'
    assert abs(t['vs5']['n4']['l1_best'] - 0.725) < 1e-9 and abs(t['vs5']['n4']['l1_last'] - 1.031) < 1e-9
    assert abs(t['pairs']['k4']['l1_best'] - 0.765) < 1e-9 and abs(t['pairs']['k2']['l1_last'] - 1.536) < 1e-9
    assert all(v['ok'] for v in NUM['plan_checks_compare'].values())
    for r in runs:
        assert '--lock big' in r['command'] and '--mem 3872M' in r['command'] and '--need-mb 3200' in r['command']
        assert 'nothp.py' in r['command'] and 'choom -n 1000' in r['command'] and 'nice -n 10' in r['command']
    assert all('--init' in r['command'] and '--lr 1e-4' in r['command'] for r in runs if r['family'] == 'B-FAST')
    import re
    known = set(re.findall(r"add_argument\('(--[a-z0-9-]+)'", open('/home/jetson/jetformer/tools/train.py', encoding='utf-8').read()))
    used = set()
    for r in runs:
        used |= set(re.findall(r'(--[a-z][a-z0-9-]+)', r['command'].split('tools/train.py', 1)[1]))
    missing = sorted(used - known)
    assert missing == ['--aug-top-brightness'], '引数が train.py に無い: %s' % missing
    proto['train_py_args_check'] = {'used': sorted(used), 'not_yet_in_train_py': missing,
                                    'note': '--aug-top-brightness は C1(plan.md 4.5)で足す。段 3 の前にテスト(既定なら今と同じ・乱数を引かない・再開)'}
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(proto, f, ensure_ascii=False, indent=1)
        f.write('\n')
    os.replace(tmp, OUT)
    print('書いた:', OUT, '(学習 %d 本・封 %d 本)' % (len(runs), len(SEALED)))


if __name__ == '__main__':
    sys.exit(main())
