#!/usr/bin/env python3
"""protocol_v3.json を書く(2026-10-26。numpy・json だけ。学習・GPU を使わない)。
自分で確かめること(外れたら書かずに止まる):
  - protocol_v2・lock2 の sha256 が固定の値と同じ(v2 は変えない)
  - 閾値は protocol_numbers_v3.json から式で出す(手で写さない)。v2 の H1・介入の値と 0.002 µs 以内で一致
  - train.py に要る引数がある(--drop-mask・--label-override-npz・--init・--aug-*・--seed)。足す引数は「未」と書く
  - 段 0 の表(diag_data/out/ablation)の sha256・数・封の記録が 0
  - 新しい門が 5 本を全部通し、v2 の 19 本の落ち方をとらえる(自己試験)
流し方: /home/jetson/bin/jf-heavy --lock none --mem 1500M -- python3 -I write_protocol_v3.py
"""
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/jetson/jetformer')
REP = Path('/home/jetson/jetformer_reports/fix_20261026')
REP2 = Path('/home/jetson/jetformer_reports/train_v2_20261025')
HERE = Path(__file__).resolve().parent
OUT = REP / 'protocol_v3.json'
NUM = REP / 'protocol_numbers_v3.json'
ABL = REPO / 'experiments/20261026_fix/diag_data/out/ablation'
DIAG = REPO / 'experiments/20261026_fix/diag_data'
TABLES_V3 = 'data/derived/dataset_v3_20261026'
DS = 'data/datasets/dataset_v2_20261025'
B_RUN = 'clean_driving_20261011-nocmd-s3-20261011-024626'
PROTO2_SHA = 'ca2556a28584ef922411525bc8a1eac46a42764c6df3fdc56ca6a1d7aa91de3a'
LOCK2_SHA = '7dea723de889bc07b0efba2885177689ee3c4a6df11cc0f585f1c1f86f599442'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def r3(x):
    return round(float(x), 3)


def r4(x):
    return round(float(x), 4)


def main():
    if sha256(REP2 / 'protocol_v2.json') != PROTO2_SHA or sha256(REP2 / 'protocol_v2_lock2.json') != LOCK2_SHA:
        raise SystemExit('protocol_v2 か lock2 が固定の値と違う(v2 は変えない約束)')
    p2 = json.loads((REP2 / 'protocol_v2.json').read_text())
    N = json.loads(NUM.read_text())
    if N['protocol_v2_sha256'] != PROTO2_SHA:
        raise SystemExit('protocol_numbers_v3.json が別の protocol_v2 から作られた')

    # ---- 閾値(2 × 標準誤差。v2 と同じ式)
    st1 = N['part1_interv_steer']['stats']
    for key, v2key in (('h1_best', 'l1_best'), ('h1_last', 'l1_last'), ('interv_best', 'interv_best')):
        a, b = st1[key]['thresholds']['vs5_n4'], p2['thresholds_us']['values']['vs5']['n4'][v2key]
        if abs(a - b) > 0.002:
            raise SystemExit('数え直しが v2 と合わない: %s %s %s' % (key, a, b))
    f1 = N['part1_interv_steer']['five']

    def ivstat(key):
        x = np.array([f1[r][key] for r in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B')], np.float64)
        sd = float(np.std(x, ddof=1))
        th = {('vs5_n%d' % n): r3(2.0 * sd * math.sqrt(1.0 / n + 0.2)) for n in (1, 2, 3, 4)}
        th.update({('pairs_k%d' % k): r3(2.0 * sd * math.sqrt(2.0 / k)) for k in (2, 3, 4)})
        return {'mean': r4(x.mean()), 'sd': r4(sd), 'thresholds': th}

    ivs = ivstat('interv_steer_best')
    ivs_last = ivstat('interv_steer_last')
    # 平均・sd は丸めない値(5 本の 1 本ずつの値)から出す
    f5 = N['part2_desk_new_fields']['five']

    def ms(key, src):
        x = np.array([src[r][key] for r in ('A_s0', 'A_s1', 'v0-s2', 'v0-s3', 'B')], np.float64)
        return float(x.mean()), float(np.std(x, ddof=1))

    _, sd1010 = ms('straight_thr_vs_b_us_1010', f5)
    _, sd1023 = ms('straight_thr_vs_b_us_1023', f5)
    g1010 = max(10.0, 2.0 * sd1010 * math.sqrt(1.2))
    g1023 = max(10.0, 2.0 * sd1023 * math.sqrt(1.2))
    mb, sbd = ms('standstill_back_frac', f5)
    sb = {'mean5': mb, 'sd5': sbd}
    g_back = mb + 2.0 * sbd * math.sqrt(1.2)
    p5 = {k: N['part3_probe_from_tab']['runs'][v] for k, v in (
        ('A_s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30'), ('A_s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30'),
        ('v0-s2', 'realacc_20261015-v0-s2'), ('v0-s3', 'realacc_20261015-v0-s3'), ('B', B_RUN))}
    pr = {}
    for key in ('st10_thr_vs_b_us', 'st23_thr_vs_b_us', 'stand_brake_side_frac'):
        m_, s_ = ms(key, p5)
        pr[key] = {'mean5': m_, 'sd5': s_}
    gp10 = max(10.0 / 0.9, 2.0 * pr['st10_thr_vs_b_us']['sd5'] * math.sqrt(1.2))
    gp23 = max(10.0 / 0.9, 2.0 * pr['st23_thr_vs_b_us']['sd5'] * math.sqrt(1.2))
    gp_back = pr['stand_brake_side_frac']['mean5'] + 2.0 * pr['stand_brake_side_frac']['sd5'] * math.sqrt(1.2)
    fix_st23 = 2.0 * sd1023 * math.sqrt(2.0 / 4)
    fix_back = 2.0 * sb['sd5'] * math.sqrt(2.0 / 4)
    T = {'g1010': r3(g1010), 'g1023': r3(g1023), 'g_back': r4(g_back), 'gp10': r3(gp10), 'gp23': r3(gp23),
         'gp_back': r4(gp_back), 'fix_st23': r3(fix_st23), 'fix_back': r4(fix_back),
         'fut_st23': r3(3 * gp23), 'fut_back': r4(3 * gp_back)}

    # ---- 自己試験(新しい門: 5 本は全部通る・v2 の落ち方をとらえる)
    five = N['part2_desk_new_fields']['five']
    other = N['part2_desk_new_fields']['v2_and_bfast']
    probe = N['part3_probe_from_tab']['runs']

    def desk_new_ok(r):
        return {'straight_1010': abs(r['straight_thr_vs_b_us_1010']) <= T['g1010'],
                'straight_1023': abs(r['straight_thr_vs_b_us_1023']) <= T['g1023'],
                'standstill_back': r['standstill_back_frac'] <= T['g_back']}

    def probe_ok(r):
        return {'st10': abs(r['st10_thr_vs_b_us']) <= T['gp10'], 'st23': abs(r['st23_thr_vs_b_us']) <= T['gp23'],
                'stand_brake_side': r['stand_brake_side_frac'] <= T['gp_back']}

    self_test = {'five': {}, 'v2_and_bfast': {}}
    for rid, r in five.items():
        ok = desk_new_ok(r)
        if not all(ok.values()):
            raise SystemExit('新しい門が 5 本の %s を落とす: %s' % (rid, ok))
        self_test['five'][rid] = 'pass'
    run_of = {'A_s0': 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30',
              'A_s1': 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30',
              'v0-s2': 'realacc_20261015-v0-s2', 'v0-s3': 'realacc_20261015-v0-s3', 'B': B_RUN}
    for rid, run in run_of.items():
        if not all(probe_ok(probe[run]).values()):
            raise SystemExit('学習中の門が 5 本の %s を落とす' % rid)
    n_v2_fail = 0
    for run, r in other.items():
        ok = desk_new_ok(r)
        okp = probe_ok(probe[run])
        fails = [k for k, v in ok.items() if not v] + ['probe_' + k for k, v in okp.items() if not v]
        if run.startswith('bfast'):
            fails = [f for f in fails if 'straight' not in f and 'st1' not in f and 'st2' not in f]
        self_test['v2_and_bfast'][run] = fails or 'pass'
        if run.startswith('v2_') and fails:
            n_v2_fail += 1
    self_test['summary'] = '5 本は新しい門(回ごとの直線・止まった場面の後ろ側・学習中の門)を全部通る。v2 の 19 本のうち %d 本が新しい門だけでも落ちる' % n_v2_fail

    # ---- train.py の引数
    tp = (REPO / 'tools/train.py').read_text()
    have = sorted(set(re.findall(r"add_argument\('(--[a-z0-9-]+)'", tp)))
    need = ['--policy', '--data', '--drop-mask', '--label-override-npz', '--init', '--epochs', '--lr', '--stride', '--batch',
            '--workers', '--no-command-input', '--impact-g', '--seed', '--out', '--resume', '--aug-brightness',
            '--aug-contrast', '--aug-shift-px', '--aug-tof-noise', '--aug-top-brightness', '--sample-weights-npz']
    miss = [a for a in need if a not in have]
    if miss:
        raise SystemExit('train.py に無い引数: %s' % miss)
    to_add = {'--probe-npz': '学習中の物差し(案 5)。既定 OFF。5 エポックごと・1 エポック目・最後に 1 回。history[-1]["probe"] に入れる',
              '--action-ref-from-samples': '作用の基準を学習に使うサンプルだけから出す(案 4)。既定 OFF(OFF なら今と同じ値)'}
    for a in to_add:
        if a in have:
            to_add[a] += '(この時点で、もうある)'

    # ---- 段 0 の表(diag_data の ablation)
    tj = json.loads((ABL / 'tables.json').read_text())
    cnt = json.loads((ABL / 'counts.json').read_text())
    abl_files = {}
    for f in ('drop_v2h_same.npz', 'drop_v2h_noTK.npz', 'drop_v2h_noTK_noLB.npz', 'override_tk_thr_B.npz'):
        abl_files[f] = sha256(ABL / f)
        if f in tj['files'] and tj['files'][f]['sha256'] != abl_files[f]:
            raise SystemExit('%s の sha256 が tables.json と違う' % f)
    if (cnt['drop_v2h_same.npz'], cnt['drop_v2h_noTK.npz'], cnt['drop_v2h_noTK_noLB.npz']) != (6506, 6142, 6105):
        raise SystemExit('段 0 の表の学習サンプル数が diag_data と違う')
    sealed = set(p2['seal']['logs'])
    for f in ('drop_v2h_noTK_noLB.npz', 'override_tk_thr_B.npz'):
        with np.load(ABL / f, allow_pickle=False) as z:
            sc = z['scene'].astype(str)
            if f.startswith('drop'):
                sc = sc[~z['drop'].astype(bool)]
            if any(s[:15] in sealed for s in np.unique(sc)):
                raise SystemExit('%s の学習に使う行に封の記録がある' % f)

    bfd = json.loads((REP / 'diag_bfast.json').read_text())['D_label_bound']['demo_dsh']
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')
    CMD0 = ('cd ~/jetformer && /home/jetson/bin/jf-heavy --mem 3872M --lock big --need-mb 3200 -- nice -n 10 env '
            'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MALLOC_ARENA_MAX=2 choom -n 1000 -- '
            '/home/jetson/miniconda/envs/jetracer/bin/python -I experiments/20261015_real_accuracy/train/nothp.py '
            '/home/jetson/miniconda/envs/jetracer/bin/python -u tools/train.py')
    AUG2 = '--aug-brightness 0.15 --aug-contrast 0.2 --aug-shift-px 4 --aug-tof-noise 0.02 --aug-top-brightness 0.15'
    NEW = '--action-ref-from-samples --probe-npz experiments/20261026_fix/diag_data/out/probe_set.npz'
    arm_tab = {
        'v3n': {'tables': '--drop-mask %s/drop_v3n.npz' % TABLES_V3, 'n_train': 6105,
                'what': 'v2-H − 交代の直後 3 秒 364 − 低速でブレーキ側の正解 37(案 1 + 案 2 + 案 4)',
                'same_as': 'diag_data の drop_v2h_noTK_noLB.npz と同じ配列(sha256 %s)' % abl_files['drop_v2h_noTK_noLB.npz']},
        'v3d': {'tables': '--drop-mask %s/drop_v3d.npz --label-override-npz %s/override_v3d_tk_thr_B.npz' % (TABLES_V3, TABLES_V3),
                'n_train': 6469,
                'what': 'v2-H − 低速でブレーキ側の正解 37。交代の直後 364 は残し、そのスロットルの正解(行 0〜9)だけ B の予測に差し替える。ステアは人のまま(案 D + 案 2 + 案 4)',
                'same_as': '差し替えは diag_data の override_tk_thr_B.npz と同じ配列(sha256 %s)。drop は drop_v2h_same から 37 を外したもの' % abl_files['override_tk_thr_B.npz']},
        'v3b': {'tables': '--drop-mask %s/drop_v3b.npz' % TABLES_V3, 'n_train': 5875,
                'what': '比べの腕: B の 41 シーン(5,912)− 低速でブレーキ側の正解 37。新しい人を入れない(案 2 + 案 4)',
                'same_as': 'drop_v2h_same から新しい人 594 と 37 を外したもの'},
    }
    per_h = 0.768 / 6506.0
    runs = []
    for seed in range(4):
        for arm in ('v3n', 'v3d', 'v3b'):
            a = arm_tab[arm]
            name = 'v3_20261026-%s-aug2-e60-s%d' % (arm, seed)
            runs.append({'stage': 1, 'arm': arm, 'name': name, 'seed': seed, 'epochs': 60, 'n_train': a['n_train'],
                         'hours_est': round(a['n_train'] * per_h + 0.115, 2),
                         'command': '%s --policy jetformer_v0 --data %s %s --epochs 60 --lr 3e-4 --stride 3 --batch 64 '
                                    '--workers 2 --no-command-input --impact-g 2.5 %s %s --seed %d --out data/runs/%s'
                                    % (CMD0, DS, a['tables'], AUG2, NEW, seed, name)})
    bf_runs = []
    for seed in range(3):
        for kind, tab in (('bfast2', '--drop-mask %s/drop_bfast2.npz --sample-weights-npz %s/weights_bfast2.npz '
                                     '--label-override-npz %s/override_bfast2_runscale.npz' % (TABLES_V3, TABLES_V3, TABLES_V3)),
                          ('bnull', '--drop-mask %s/drop_bnull.npz' % TABLES_V3)):
            name = '%s_20261026-e15-s%d' % (kind, seed)
            bf_runs.append({'stage': 'BF2', 'arm': kind, 'name': name, 'seed': seed, 'epochs': 15, 'hours_est': 0.23,
                            'only_if': 'BF2-0 が「学習する」のときだけ',
                            'command': '%s --policy jetformer_v0 --data %s %s --init data/runs/%s/best.pth --epochs 15 '
                                       '--lr 1e-4 --stride 3 --batch 64 --workers 2 --no-command-input --impact-g 2.5 %s '
                                       '--seed %d --out data/runs/%s' % (CMD0, DS, tab, B_RUN, NEW, seed, name)})
    total_main = round(sum(r['hours_est'] for r in runs), 1)

    P = {
        'schema': 'jf-train-v3-protocol/v1',
        'title': '学習 v3(データの直し・B-FAST2)と発進の前進の事前の決まり。学習の前に固定し、結果を見て変えない',
        'version': 3,
        'written_at': now,
        'written_by': 'workflow のサブエージェント(直しの設計と事前の決まりの係)。学習・GPU・torch を使っていない(numpy だけ)。'
                      '車・基板・走行の API・jetformer-web に触れていない。data/・protocol_v2・lock2・既定値は変えていない。git commit していない',
        'request': {
            'user_20261026_10': '「それでは、今日の課題の整理と、それを修正する作業を開始してください。」',
            'goal': 'ぶつからずに、止まってから 3 周 25 秒(1 周 約 8.12 秒)。次の実車は 11/3(火・祝)',
            'not_the_task': '進捗の質問は main session が答える。この決まりを変える依頼ではない',
            'what_this_is': '課題 1(v2 の 22 本が門に不合格)・課題 2(B-FAST が踏まない)・課題 3(発進の損 0.63 秒)の直しの事前の決まり',
        },
        'relation_to_v2': {
            'protocol_v2_sha256': PROTO2_SHA,
            'lock2_sha256': LOCK2_SHA,
            'rules': [
                'protocol_v2.json・lock2 は変えない。v2 の答え(D1〜D5a・BF)はそのまま。v2 の D5b は 11/1 に v2 の決まりで書く(候補 0 → 無し)',
                'v3 は別の版。v2 から受け継ぐ: H1・介入・5 本・閾値の式・良い/同等/悪いの決まり・印(長すぎ・上がり)・封・C3 の集合(sets_v2.npz)',
                'v2 の封の決まり「11/3 のあと(v3)では学習に戻してよい」は 11/3 の後の版のこと。この v3 は 11/3 の前なので、封はそのまま守る',
                'v2 から変えたところ(v3 だけ): 介入の拒否権をステアの L1 で見る・回ごとの直線と止まった場面の後ろ側の門・学習中の門・公平な G-b・封の V の拒否権',
            ],
        },
        'sources': {k: {'path': str(v), 'sha256': sha256(v)} for k, v in (
            ('diag_data', REP / 'diag_data.md'), ('diag_bfast', REP / 'diag_bfast.md'), ('diag_bfast_json', REP / 'diag_bfast.json'),
            ('diag_launch', REP / 'diag_launch.md'), ('plan_1103_draft', REP / 'plan_1103_draft.md'),
            ('final_eval_v2', REP2 / 'final_eval.md'), ('decisions_v2', REP2 / 'decisions.json'),
            ('launch_summary', REPO / 'experiments/20261026_fix/diag_launch/out/summary.json'))},
        'lock': {
            'this_file': str(OUT),
            'sha256_file': str(REP / 'protocol_v3.sha256'),
            'how_to_check': 'cd /home/jetson/jetformer_reports/fix_20261026 && sha256sum -c protocol_v3.sha256',
            'read_only': 'chmod a-w(protocol_v3.json・protocol_v3.md・protocol_numbers_v3.json・protocol_v3.sha256)',
            'second_lock': {
                'file': str(REP / 'protocol_v3_lock2.json'),
                'when': '段 0 のあと・v3 の本番の学習(段 1)を始める前。v2 は学習のあとになった(v2 の deviations 6)ので、v3 は前に書く',
                'contents': ['data/derived/dataset_v3_20261026/ の表すべての sha256 と、train.py が読む学習サンプルの数',
                             'tools/train.py・jetformer/train_data.py・jetformer/dataset.py の sha256(段 data の変更のあと)',
                             'probe_gates.py・probe_set.npz の sha256', '机上の台本(desk_v3・eval_v3)の sha256',
                             '段 0 の答え(stage0 の 1〜9)', 'D0・BF2-0 の答え'],
                'rule': '数を埋めるだけ。lock2_v3 で決まりを変えない。sha256 を protocol_v3_lock2.sha256 に書き chmod a-w',
            },
            'commit': 'commit はユーザーの了承のあと(この係はしない)',
        },
        'background_v2': {
            'runs': 22, 'passed_all_gates': 0,
            'straight_thr_vs_b_us_v2': '+13.3〜+53.8(門 |差| ≤ 10)',
            'brake_lowspeed_frac_v2': '0.0145〜0.2115(門 ≤ 0.01。B 0.0082)',
            'h1_best_us': 'v2h-aug2-e60-s1 48.07(B 49.14・5 本の平均 49.61)',
            'interv_best_us': 'v2h-msf-aug2-e60-s1 110.54(5 本の平均 128.59)',
            'bfast_gb_fwd_vs_b_us': '−5.9〜−6.7(門 ≥ +10)',
        },
        'causes': {
            'A_data': {'what': '新しい人の「交代の直後 3 秒」364 サンプル(学習の 5.6%)。人が取り返して緩める・止める操作',
                       'numbers': ['正解: 動いているときの前進 −28.3 µs・直線 −101.0 µs・動いているのにブレーキ側 23.1%(B の 41 シーンは +183.5・+188.5・0%)',
                                   'v2-H は残差 −16〜+3 µs まで覚えた(B は −129.9)。緩める向きの引っぱりの約 9 割',
                                   '直線の緩めは 10/23 に集まる(v2-H 軽量 13 本: 10/10 +13.3・10/23 +57.4 µs。10/10 だけの 4 本 +5.5・−4.4)',
                                   '止まった場面の後ろ側(> 中立 + 30): v2-H 13 本中 11 本が 41〜75%(10/10 だけの 5 本 1.3〜5.1%)',
                                   '仮説「止まったフレームが多い」は違う(v < 0.3 m/s は 7.0% 対 6.9%)。作用の基準 1433/1500 は門に +0.4 µs だけ'],
                       'src': 'diag_data.md'},
            'B_bfast': {'what': '手本の正解がゲイン 0.90 のあとの値。0.90 で走らせると 0.90 × 0.91 = B の生の前進の 0.82 倍',
                        'numbers': ['正解 = out = model_out(差 0.0000 µs)、raw との前進の比 0.900(No.9)', '完全に覚えれば G-b −16.7 µs。観測 −5.9〜−6.7 は縮みの 33〜38%',
                                    '手本は B 自身の指令(raw と机上の B の差 中央 0.11 µs)。どの正解の直しでも、同じゲインで B より踏むことは学べない',
                                    '完全に覚えたときの上限(手本の DE・SW・HS): 今 %.2f・A %.2f・B %.2f µs(門 +10)' % (bfd['cur_bound_g090_us'], bfd['fixA_bound_g090_us'], bfd['fixB_bound_g090_us'])],
                        'src': 'diag_bfast.md'},
            'C_launch': {'what': '発進の損 0.63 秒のほとんどは LH の手前の直線の加速(モデルが前進を人の発進のように少しずつ上げる)',
                         'numbers': ['10/24 の B 9 回: 動き出すまで 0.16・直線 0.52・LH → R1 −0.03 秒', '最初の 0.5 秒の前進 102 µs・1.0 秒で 126 µs。加速の最大 5.2 m/s²(車は 7.1 m/s² まで出る)',
                                     '補助 80 µs は最初の約 0.1 秒だけ。強めても 0.06〜0.08 秒', '発進の前進 250 µs・2.5 m/s まで: 0.30 秒縮む(3.0 m/s まで 0.35)'],
                         'src': 'diag_launch.md'},
        },
        'data': {
            'dataset': DS + '(作り直さない。表だけ新しく作る)',
            'tables_dir': TABLES_V3 + '/(段 data が作る)',
            'tables': {
                'drop_v3n.npz': {'n_train': 6105, 'rule': arm_tab['v3n']['same_as']},
                'drop_v3d.npz': {'n_train': 6469, 'rule': 'drop_v2h_same から、v < 1.0 m/s で行 0 の正解 > 1598 µs の 37 サンプルを外す'},
                'override_v3d_tk_thr_B.npz': {'rows': 364, 'rule': arm_tab['v3d']['same_as']},
                'drop_v3b.npz': {'n_train': 5875, 'rule': arm_tab['v3b']['same_as']},
                'drop_bfast2.npz・weights_bfast2.npz・override_bfast2_runscale.npz・drop_bnull.npz': {'rule': 'BF2-0 が「学習する」のときだけ作る(下の bfast2)'},
            },
            'lb37': '低速でブレーキ側の正解 = B の 41 シーンの、v < 1.0 m/s で行 0 のスロットルの正解 > 1598 µs のサンプル 37(497 中)。diag_data の d3 と同じ',
            'tk364': '交代の直後 3 秒 = 新しい人のサンプルで、直前の交代(ラベルの events.takeovers)から 3 秒以内。diag_data の d1 と同じ 364',
            'checks': ['train.py の train_samples が上の n_train と同じ(±0。表は決まった配列なので)', '検証(H1)1,559 サンプル(v2 と同じ)',
                       '学習に使う行に封の記録が 0(check_seal_v2.py も OK)', '作用の基準(案 4 の引数あり)が 1441 ± 2・1496 ± 2 µs',
                       'v3d の差し替え 364 × 10 行が全部学習に使われる(train.py の表示)'],
            'if_off': '外れたら学習を始めない。main session に聞く(deviations_v3.md に書く)',
        },
        'common_recipe': {
            'src': 'v2 の最終の腕 v2h-aug2-e60 と同じ作り方(train.log の 10/25 22:24 の開始行)。変えるのはデータの表と、下の 2 つの引数だけ',
            'policy': 'TransFuser軽量型 jetformer_v0', 'epochs': 60, 'lr': 3e-4, 'batch': 64, 'workers': 2, 'stride': 3,
            'command_input': '入れない(--no-command-input)', 'impact': '--impact-g 2.5', 'aug': 'AUG2(' + AUG2 + ')',
            'best': 'H1 の検証の L1 が一番小さいエポック = best.pth。11/3 に使う重みは best.pth。比べは best と last の両方',
            'new_args': to_add,
            'new_args_rule': '2 つとも段 data が train.py に足す(既定 OFF)。OFF のとき今の学習と同じ(試験)。probe は学習の乱数を引かない(段 0 の 4)',
            'memory_retry': 'rc 137 のときだけ上限を 1 回上げる(最大 4.5G)。deviations_v3.md に書く',
            'command_prefix': CMD0,
        },
        'arms': {k: dict(v, epochs=60, seeds=[0, 1, 2, 3], aug='AUG2') for k, v in arm_tab.items()},
        'arms_why': {
            'v3n': '一番すなおな直し(原因 A を外す)。介入のステアの良さ(立て直しの正解)も消える見込み',
            'v3d': '立て直しのステアは残し、スロットルだけ B にそろえる。介入のステアの良さ(v2 の 149〜157 µs、5 本の平均 158.4)が残る見込み',
            'v3b': '新しい人のデータが要るかを見る比べの腕。AUG2(v2 の D3 で H1 −0.95・−1.15、ずれ 0.61 → 0.25)の効きだけを B のデータで見る。候補にもなる',
            'not_included': ['多段型・2 本の平均(推論の門の近く・1 枠に入れない)', 'e100・e30(v2 の D2 で 60)', 'v2-M の形・ジャンル 9(手本の正解の尺度の問題が B-FAST と同じ)',
                             '案 6・7(走りの区間の重み・AI の速い周の割り戻し)は 11/3 のあと(v4)'],
        },
        'runs': runs,
        'runs_bfast2_conditional': bf_runs,
        'stage0': {
            'when': 'この決まりのあと・段 1 の前(10/26 夕〜夜)',
            'list': [
                {'id': 1, 'what': 'sha256sum -c protocol_v2.sha256・protocol_v2_lock2.sha256・protocol_v3.sha256 と check_seal_v2.py', 'if_off': '止める'},
                {'id': 2, 'what': '段 data の表 4 つ(v3n・v3d・v3b・差し替え)の数と封(data.checks)', 'if_off': '学習を始めない'},
                {'id': 3, 'what': '案 4 の引数で作用の基準 1441 ± 2・1496 ± 2。引数なしなら今と同じ 1433・1500', 'if_off': '止めて段 data に戻す'},
                {'id': 4, 'what': 'probe の引数の ON・OFF で、同じ種の 2 エポック(--max-steps 20)の損失と best.pth が同じ。OFF・OFF で違う(GPU の揺れ)なら、ON・OFF の差が OFF・OFF の差の 2 倍以内', 'if_off': 'probe を学習の外(best・last の後から)だけにする'},
                {'id': 5, 'what': 'probe_gates.py の値を、保存した 4 本で probe_check.json と同じに出す(±0.01 µs・±0.0001)', 'if_off': '止める'},
                {'id': 6, 'what': '段 0 の短い学習(ablation_short.sh same noTK noTK_noLB tkB。15 エポック・種 0)→ D0', 'if_off': '止まったら same・noTK だけで D0'},
                {'id': 7, 'what': 'BF2-0(G-b0。diag_bfast.py の part D を流し直す。GPU なし)', 'if_off': '—'},
                {'id': 8, 'what': '机上の台本 desk_v3・eval_v3 が、保存した 5 本・v2 の 22 本で protocol_numbers_v3.json と同じ値を出す(±0.01 µs・±0.0001)', 'if_off': '止める'},
                {'id': 9, 'what': '空き容量 5 GB 以上・電源(ユーザーが AC か満充電を確かめた。10/26 07:42 に電源が落ちた)', 'if_off': '夜の学習を始めない'},
            ],
        },
        'probe': {
            'script': str(DIAG / 'probe_gates.py'), 'script_sha256': sha256(DIAG / 'probe_gates.py'),
            'set': str(DIAG / 'out/probe_set.npz'), 'set_sha256': sha256(DIAG / 'out/probe_set.npz'),
            'frames': {'st10': 441, 'st23': 581, 'stand': 372, 'low10': 582},
            'fields': {'st10_thr_vs_b_us・st23_thr_vs_b_us': '直線(10/10・10/23)の行 2 のスロットル(このモデル − B)の平均。生の予測(ゲインの前)。+ = 緩める',
                       'stand_brake_side_frac': '止まった場面(前が 1 m 以上空き)で行 0 > 1528 µs の割合',
                       'low10_brake_frac': '10/10 の AI 走行の v < 1.0 m/s で行 0 > 1598 µs の割合(門「低速のブレーキ」と同じ母集団・FP32)'},
            'when': '学習の中で 1 エポック目・5 エポックごと・最後。学習のあと best.pth と last.pth で probe_gates.py run(門に使うのはこの 2 つ)',
            'not_for': 'best のエポックの選び方・損失・データには使わない(best は H1 のまま)',
        },
        'sets': {'same_as_v2': 'protocol_v2 の sets(H1・介入・直線・止まった場面・sanity・G-b)。sets_v2.npz を作り直さない',
                 'sets_v2_sha256': sha256(REP2 / 'desk/sets_v2.npz'),
                 'why_valid_for_v3': 'v3 の学習サンプルは v2-H(v3b は B の 41 シーン)の部分集合。「学習に使っていない周」の性質はそのまま',
                 'straight_split': '直線を回で分ける: sess 0 = 10/10(882 フレーム)・sess 1 = 10/23(1,161 フレーム)'},
        'fields_added_v3': {
            'l1_interv_steer_best_us・l1_interv_steer_last_us': '介入のステアの L1(拒否権に使う)',
            'l1_interv_thr_best_us': '介入のスロットルの L1(参考。緩めるほど良くなる物差しなので、良さの主張に使わない)',
            'straight_thr_vs_b_us_1010・straight_thr_vs_b_us_1023': '直線の門を回で分けた値(行 2・車に出る値・ゲイン 0.90。+ = 緩める)',
            'standstill_back_frac': '止まった場面で、行 0 の車に出るスロットル > そのモデルの中立 + 30 µs の割合(後ろ側。StartKick が出なくなる)',
            'brake_lowspeed_moving_frac': '参考: 低速のブレーキを 0.3〜1.0 m/s だけで数えた値',
            'probe_best・probe_last・probe_history': '学習中の物差し(上)',
        },
        'thresholds_us': {
            'rule': '2 × 標準誤差(v2 と同じ)。標準偏差 = B と同じ作り方の 5 本(A_s0・A_s1・v0-s2・v0-s3・B)。腕 n 本 対 5 本: 2·sd·√(1/n + 1/5)。同じ種 k 組: 2·sd·√(2/k)。候補 1 本の門: 2·sd·√(1 + 1/5)',
            'v2_same': {'vs5': p2['thresholds_us']['values']['vs5'], 'pairs': p2['thresholds_us']['values']['pairs'],
                        'five_mean': p2['thresholds_us']['values']['five_mean'], 'not_worse_us': 0.5},
            'interv_steer': {'five_mean_best': ivs['mean'], 'five_sd_best': ivs['sd'], 'thresholds_best': ivs['thresholds'],
                             'five_mean_last': ivs_last['mean'], 'five_sd_last': ivs_last['sd'], 'thresholds_last': ivs_last['thresholds']},
            'gate_straight_1010': {'value': T['g1010'], 'rule': 'max(10, 2·sd·√1.2)。5 本の sd %.3f → %.3f' % (sd1010, 2 * sd1010 * math.sqrt(1.2))},
            'gate_straight_1023': {'value': T['g1023'], 'rule': 'max(10, 2·sd·√1.2)。5 本の sd %.3f' % sd1023},
            'gate_standstill_back': {'value': T['g_back'], 'rule': '5 本の平均 %.5f + 2·sd·√1.2(sd %.5f)。丸めない値から' % (sb['mean5'], sb['sd5'])},
            'probe_st10': {'value': T['gp10'], 'rule': 'max(10 ÷ 0.9, 2·sd·√1.2)。生の予測なので 10 µs をゲイン 0.90 で割る'},
            'probe_st23': {'value': T['gp23'], 'rule': 'max(10 ÷ 0.9, 2·sd·√1.2)。5 本の sd %.3f' % pr['st23_thr_vs_b_us']['sd5']},
            'probe_stand_brake_side': {'value': T['gp_back'], 'rule': '5 本の平均 + 2·sd·√1.2'},
            'probe_low10': {'value': 0.01, 'rule': 'v2 の門「低速のブレーキ」と同じ'},
            'fix_confirmed_straight_1023_k4': {'value': T['fix_st23'], 'rule': '同じ種 4 組の 2·sd·√(2/4)'},
            'fix_confirmed_standstill_back_k4': {'value': T['fix_back'], 'rule': '同じ種 4 組'},
            'futility': {'st23': T['fut_st23'], 'stand_brake_side': T['fut_back'], 'rule': '学習中の門の 3 倍'},
            'numbers': str(NUM),
        },
        'judgement': {'same_as_v2': 'protocol_v2 の judgement(対 5 本・同じ種どうし・種が欠けたとき)をそのまま使う', 'v2': p2['judgement']},
        'vetoes_v3': {
            'interv_steer': '腕の l1_interv_steer_best_us の平均 − %.3f > 閾値(n)(n=4 %.3f・3 %.3f・2 %.3f)、かつ全部の種が %.3f より大きい → その腕を候補にしない'
                            % (ivs['mean'], ivs['thresholds']['vs5_n4'], ivs['thresholds']['vs5_n3'], ivs['thresholds']['vs5_n2'], ivs['mean']),
            'why_steer': '介入の L1(ステア + スロットル)は、スロットルを緩めるほど良くなる(diag_data 3.3)。B 自身(129.50)も 5 本の平均(128.59)より大きく、B に近いモデルほど拒否されやすい。だから v3 はステアで見る。全体の介入の L1 は参考',
            'bias': 'v2 と同じ(steer_bias の平均が +0.132 ± 閾値(n)の外。n=4 で ±3.913)',
            'seal_T': 'v2 と同じ(11/1)',
            'seal_V': '11/1: seal_v_thr_vs_b_us(3 回目の直線。+ = 緩める)− 5 本の平均(封の上)> max(10 µs, 2 × 5 本の sd(封の上))→ その候補を外す',
            'when': '腕がそろったとき(拒否権)と 11/1(封)',
        },
        'gates_v3': {
            'src': '候補の種ごと。全部に合格しないと 11/3 に出さない',
            'v2_gates': {k: v for k, v in p2['gates_v2'].items() if k not in ('src', 'genre9')},
            'v2_gates_note': '推論の「同じ日の B」は、その日に測った B の p99 の中央値(v2 の deviations 9 と同じ読み方)。genre9 は v3 に無い',
            'straight_1010': '|straight_thr_vs_b_us_1010| ≤ %.3f µs' % T['g1010'],
            'straight_1023': '|straight_thr_vs_b_us_1023| ≤ %.3f µs' % T['g1023'],
            'standstill_back': 'standstill_back_frac ≤ %.4f' % T['g_back'],
            'probe': 'best.pth の probe で |st10| ≤ %.3f・|st23| ≤ %.3f µs・stand_brake_side ≤ %.4f・low10 ≤ 0.01' % (T['gp10'], T['gp23'], T['gp_back']),
            'why': '課題 1 の落ち方(10/23 の直線で緩める・止まった場面で後ろ側)を回ごと・場面ごとに見る。v2 の門は 10/10 と 10/23 を混ぜ、後ろ側を見ていなかった',
            'self_test': self_test,
        },
        'futility': {
            'rule': '同じ腕の種 0 と種 1 が終わり、両方とも best.pth の probe で st23 > %.1f µs か stand_brake_side > %.3f なら、その腕の種 2・3 を始めない' % (T['fut_st23'], T['fut_back']),
            'effect': 'その腕は n = 2 で判定(閾値は n2)。deviations_v3.md に書く。学習の途中では止めない(終わった種だけで決める)',
        },
        'marks': {'too_long': 'v2 と同じ(desk の last − best > 1.6 µs)', 'rise': 'v2 と同じ(参考)',
                  'loosen_at_end': '参考: best は学習中の門を通るのに last が落ちる(11/3 は best.pth を使う)'},
        'fix_confirmed': {
            'what': '「原因 A の直しが効いた」と書いてよい条件(結論の言い方の決まり。候補の選びには使わない)',
            'rule': ['v3n・v3d それぞれ、同じ種の v2h-aug2-e60(種 0〜3)と比べて: straight_thr_vs_b_us_1023 の d の平均 ≤ −%.3f かつ全部の d_k < 0 → 「10/23 の直線の緩めが減った」' % T['fix_st23'],
                     'standstill_back_frac の d の平均 ≤ −%.4f かつ全部の d_k < 0 → 「止まった場面の後ろ側が減った」' % T['fix_back'],
                     'H1 の L1 は同じ種どうしの「悪くない」(d ≤ +0.5 µs)を見る。悪くなっても候補からは外さない(門と 5 本の判定で決める)'],
        },
        'bfast2': {
            'what': 'B-FAST の作り直し(原因 B)。公平な G-b で見る。学習の前の G-b0 で止まる見込み',
            'labels_candidates': {'A': '正解をゲインの前に戻す(--label-gains recorded)。上限 %.2f µs' % bfd['fixA_bound_g090_us'],
                                  'B': '正解を走らせるゲイン 0.90 の尺度に(--eval-throttle-gain 0.90)。上限 %.2f µs' % bfd['fixB_bound_g090_us']},
            'recipe_if_trained': {'init': 'モデル B の best.pth', 'human': 'B の 41 シーン − 低速でブレーキ側の 37(5,875)', 'teacher': 'v2 と同じ 10 周 836(R1 の前後 1 秒は重み 0.5)',
                                  'labels': 'G-b0 を通った候補(A か B)を差し替えの表で入れる(データセットを作り直さない)', 'epochs': 15, 'lr': 1e-4, 'seeds': [0, 1, 2],
                                  'b_null': '同じ作り方で手本だけ抜いた学習(同じ種 0・1・2)', 'hours': '6 本 × 0.23 = 1.4 時間'},
            'BF2-0': {'when': '段 0(GPU なし)', 'rule': '手本の DE・SW・HS で 0.90 × (正解の行 2 − B の行 2) の平均(上限)≥ +10 µs、かつ 上限 × 0.35(10/23 で覚えた割合 0.33〜0.38)≥ +10 µs(= 上限 ≥ +28.6 µs)。通った候補があるときだけ学習する',
                      'expected': '今の値は A %.2f・B %.2f µs → どちらも通らない → B-FAST2 は学習しない(GPU 0)。走行タブのゲインで +10 µs(= 0.90 → 0.95)はできる' % (bfd['fixA_bound_g090_us'], bfd['fixB_bound_g090_us']),
                      'script': str(REPO / 'experiments/20261026_fix/diag_bfast/diag_bfast.py'), 'script_sha256': sha256(REPO / 'experiments/20261026_fix/diag_bfast/diag_bfast.py')},
            'gates_bfast2': {
                'G-a': p2['gates_bfast']['G-a'], 'G-b1': 'bfast_gb_fwd_vs_b_us ≥ +10 µs(v2 の G-b と同じ。0.90 どうし)',
                'G-b2': '同じ種の B-NULL と比べて ≥ +10 µs(0.90 どうし)', 'G-b3': '形: DE・SW・HS の前進 − s × B-NULL の同じ前進 ≥ +5 µs(s = 直線の前進の比 B-FAST2 ÷ B-NULL・ゲインの前)。ゲインだけの違いは 0',
                'G-c': p2['gates_bfast']['G-c'], 'G-d': p2['gates_bfast']['G-d'], 'G-e': p2['gates_bfast']['G-e'], 'G-f': p2['gates_bfast']['G-f'],
                'safety': 'sanity・推論・振れ幅・低速のブレーキ(v2 と同じ)+ standstill_back ≤ %.4f + probe の stand_brake_side・low10' % T['g_back'],
                'veto': '介入のステア(n = 3: %.3f)・偏り(v3 と同じ)' % ivs['thresholds']['vs5_n3'],
                'self_test': 'diag_bfast.md 5 章: B・B×1.056・B×1.111・B×0.91・今の B-FAST はどれも G-b1 か G-b3 で落ちる',
            },
            'next': '同じ入力で B より強い指令の手本(速い人の周・天井や自動合わせで形の違う 11/3 の周)ができたら v4 で',
        },
        'decisions': {
            'D0': {'when': '段 0 の短い学習のあと', 'what': '原因 A の確かめ(キューの順だけを決める)',
                   'rule': ['same の st23 − B が +20 µs(生)未満なら「確かめにならない」',
                            'noTK の st23 − B ≤ same の 1/3、かつ noTK の stand_brake_side ≤ 0.10 → 案 1 を確かめた。tkB で同じ式 → 案 D を確かめた',
                            'noTK_noLB の low10 ≤ 0.01 → 案 2 を確かめた'],
                   'effect': '腕・門・閾値は変えない。順: 確かめた案の腕を先に(両方なら v3d → v3n → v3b)。どちらも確かめられない・確かめにならない → v3b → v3d → v3n。種 0 の 3 本 → 種 1 の 3 本 → … の順',
                   'note': '1 本・15 エポックなので、種の揺れ(v2h-e30 の直線 +32〜+51)より大きい差だけを読む'},
            'BF2-0': {'when': '段 0', 'rule': 'bfast2.BF2-0'},
            'E1': {'when': '腕の種がそろったとき(10/27)', 'what': '腕の拒否権と 5 本の判定',
                   'rule': '拒否権(介入のステア・偏り)が無く、H1 の L1 で 5 本に「悪い」でない腕だけを残す'},
            'E2': {'when': '10/27(遅れても 10/28)', 'what': '腕の順',
                   'rule': '残った腕を「門(gates_v3)に全部合格した種の数」の多い順に並べる。同じ数なら H1 の L1(best)の腕の平均が小さい順'},
            'E3': {'when': 'E2 のあと', 'what': 'v3 の候補',
                   'rule': '1 番目の腕の種を 0 → 1 → 2 → 3 の順に見て、門に全部合格した最初 = V1、次 = V2。1 番目の腕に 1 本しかなければ、V2 = 2 番目の腕の最初の合格。H1 で種を選ばない'},
            'E4': {'when': 'B-FAST2 を学習したときだけ', 'rule': '種 0 → 1 → 2 の順に見て gates_bfast2 に全部合格した最初 = BF2-1'},
            'E5': {'when': '11/1 朝(封を開けたあと)', 'what': '11/3 の新しいモデルの 1 枠', 'rule': 'slot_20261103_v3'},
            'order_note': '机上の順位で実車の順位は決めない',
        },
        'slot_20261103_v3': {
            'rule': ['1. 11/1 朝に封を 1 回だけ開ける(v2 と一緒の 1 回。seal_opened.json を 1 つ書き、v2・v3 の両方の予測の一覧を入れる)',
                     '2. v2 の D5b を v2 の決まりで書く(候補 0 → 無し)',
                     '3. v3 の候補(BF2-1・V1・V2)に封の T・V の拒否権を掛ける',
                     '4. 残りを BF2-1 → V1 → V2 の順に見て、最初の 1 つを 11/3 の 1 枠にする',
                     '5. 全部落ちたら枠は無し(B のまま)。開けたあとに作り直さない'],
            'predict_at_open': ['B', '5 本', 'V1', 'V2', 'BF2-1(あれば)', 'v2 の C1・C2・予備・BF1・BF2(どれも無し)'],
            'on_track': dict(p2['slot_20261103']['on_track'], where='11/3 の回 3 か回 4 の前の空き(fix_plan.md 5 章)。発進の前進の比べを先にする'),
            'expectation': 'v3 の候補は B より速くなる見込みが無い(直線を B にそろえる直し)。使う決まり(3 周の窓 0.3 秒以上速い)はほぼ満たさない見込みで、走らせるのは主に次の学習の材料のため',
            'approval': 'bench_candidates.json に入れるのも走らせるのもユーザーの了承のあと',
        },
        'seal': {'same_as_v2': True, 'logs': p2['seal']['logs'], 'manifest_sha256': p2['seal']['manifest_sha256'],
                 'rules': ['学習にも選びにも使わない(v3 の表に封の記録があれば止まる)', '段 0 の前・lock2_v3 の前・11/1 に開ける前に check_seal_v2.py',
                           '開けるのは 11/1 の 1 回だけ(v2 と一緒)', '開けたあとに足した腕は参考'],
                 'fields_at_open': p2['seal']['open_20261101']['fields'] + ['seal_v_thr_vs_b_us の 5 本の平均・sd(V の拒否権)']},
        'launch_boost': {
            'what': '既定 OFF の新しい選択肢「発進の前進」(原因 C)。実装は段 launch(jetformer/assist.py・drive.py の発進の部分・tests/test_launch*.py)',
            'params': {'launch_boost': {'range': [0, 1], 'default': 0, 'note': '起動のたびに 0(RESET_ON_BOOT)。走行中は 0 → 1 にできない(1 → 0 はできる)'},
                       'launch_us': {'range': [150, 300], 'default': 250, 'note': '出す前進の下限 [µs](モデルの中立から。速さのゲインは掛けない)'},
                       'launch_mps': {'range': [2.0, 3.0], 'default': 2.5, 'note': 'この速さ(0.1 秒先の見込み)で終わり。3.5 は LH に速すぎる(9 回中 7 回)ので範囲に入れない'},
                       'launch_max_s': {'range': [0.5, 2.0], 'default': 1.2, 'note': 'これより長く出さない [秒]'}},
            'reset_on_boot': ['launch_boost', 'launch_us', 'launch_mps', 'launch_max_s'],
            'start': ['launch_boost = 1・AI 側(CH3)・dry run でない・中立で止めていない', 'AI 側にした最初のフレームで |eRPM| < 400(止まっている)',
                      '前が空いている: ToF ガードの TTC(x_free ÷ max(v, 0.3))の直近 5 フレームの中央値 ≥ 3.3 秒(= 1.0 m 以上空き)。ToF が無い・古い(0.1 秒より古い)・ガード 0 なら始めない',
                      '車両の値が 0.1 秒以内。モデルの出力が有限', 'AI の区間で 1 回だけ(CH3 を人側に戻すと次の AI 側で 1 回)。最初のフレームで条件がそろわなければ、その区間では出さない(今まで通り StartKick)'],
            'output': ['スロットル = min(モデルの値(ゲイン・ガードのあと), 中立 − launch_us)。前進だけ。中立より後ろ(ブレーキ・後退)は出さない', '止まっている間のモデルのブレーキは無視', '舵はモデルのまま(ゲイン・trim のあと)'],
            'end_any': ['① 0.1 秒先の速さの見込み(速さの天井と同じ v・a)≥ launch_mps', '② launch_max_s', '③ |ヨー| ≥ 60 deg/s', '④ 速さ ≥ 1.0 m/s で |舵 − 中央 − trim| ≥ 200 µs',
                        '⑤ 水平 2.5G 以上の衝撃', '⑥ ToF ガードが出力を変えた、または v ≥ 0.3 m/s で TTC < 0.8 秒が 2 フレーム', '⑦ CH3 を人側', '⑧ 中立で止めた(hold)・走行の停止',
                        '⑨ 車両の値が 0.1 秒より古い、または eRPM > +200(後ろへ動いた)'],
            'after_end': 'モデルの出力のまま。同じ AI の区間では発進の前進も StartKick も出さない',
            'order': 'ゲイン → ToF ガード → 発進の前進(StartKick の場所)→ リカバリー → 区間の速度上限 → 速さの天井(あとの 2 つは前進を弱めるだけ)',
            'record': ['assist = 5(ASSIST_LAUNCH「発進の前進」)', 'stats の launch_boosts', 'AI_CMD の launch_on(フレームごと)・launch_end(終わりの理由 1〜9、0 = 無し)', 'log.json の drive.params に 4 つの値'],
            'off_same': 'launch_boost = 0 のとき、PCA9685 に書く値と記録の今までの欄がビット単位で同じ(新しい欄は足してよい)',
            'tests': [
                'T1 OFF でビット単位で同じ(偽の車・偽の記録。直す前の写しを tests/fixtures に取ってから)',
                'T2 PARAM_LIMITS・既定値・範囲の外の値の切り詰め', 'T3 RESET_ON_BOOT(設定ファイルに 1 があっても起動で 0・値も既定)', 'T4 走行中に 0 → 1 にできない・1 → 0 はできる',
                'T5 始める条件(止まっている・前が空き・ToF あり)。始めない(TTC < 3.3・ToF 無し/古い・ガード 0・動いている・dry run・人側)',
                'T6 AI の区間で 1 回だけ。CH3 を戻すと次の 1 回', 'T7 出すもの: モデルがブレーキでも 中立 − U、モデルが強ければモデル、舵は変えない',
                'T8 終わりの 9 条件それぞれ。終わったら同じ区間では前進も StartKick も出さない', 'T9 発進の前進の間、スロットル < 中立(後ろ側を出さない。でたらめなモデルの出力 1 万回)',
                'T10 区間の上限・天井はあとで掛かる。ガード(2 以上)が先に効いたら終わる', 'T11 記録の欄(assist = 5・launch_on・launch_end・stats・log.json)',
                'T12 今の試験が全部通る(test_start_kick・test_assist・test_drive_caps など。jf-heavy --lock big で pytest)',
                'T13 10/24 の B の発進 9 回の記録を OFF で流して、記録の out と同じ'],
            'not_in_scope': '画面の欄は段 launch の外。11/3 に使うなら、了承のあと画面に足すか、止まっているときに /api/drive/params で入れる(ユーザーか main session)',
            'on_track_20261103': {
                'where': 'fix_plan.md 5 章の回 3。いつもの HS の場所から、同じ設定(回 2 のよい形、無ければ P1)で',
                'order': 'OFF・ON・OFF・ON・OFF・ON・OFF・ON(1〜2 周ずつ)。ON は 250 µs・2.5 m/s・1.2 秒',
                'measure': 'diag_launch/collect.py で「AI 側 → 最初の R1 の起点」の時間・失う時間・LH の入口の速さ・車輪 ÷ IMU の比',
                'stop': ['ON の最初の周で LH・R1 に当たった → ON をやめる', '車輪 ÷ IMU の比 > 1.3(空転)→ 残りの ON を launch_us 200 に(deviations_v3.md に書く)', 'ON で 2 本当たった → やめる'],
                'adopt': 'ON の中央値 ≤ OFF の中央値 − 0.15 秒・ON の最初の周の衝突(水平 2.5G 以上)0・比 ≤ 1.3・LH の入口の速さ ON − OFF ≤ +0.3 m/s(中央値)。満たせば回 4 で ON を使ってよい(ユーザーが決める)',
                'approval': 'jetformer-web の再起動・dry run(車輪を浮かせて)・コースで ON にすること、それぞれユーザーの了承',
            },
            'estimate': '0.30 秒縮む(250・2.5)。止まってから 25.0 秒に要る 1 周は 8.12 → 約 8.22 秒',
        },
        'gpu_order': {
            'order': ['段 0: probe・作用の基準の 2 ステップの確かめ → 短い学習 4 本(same・noTK・noTK_noLB・tkB)→ D0・BF2-0 → lock2_v3',
                      '段 1: D0 の順で、種 0 の 3 本 → 種 1 の 3 本 → 種 2 → 種 3(同じ種の組が先にそろう)',
                      '机上: 1 本終わるごとに予測(H1・介入・C3 の表・ずれ)・probe(best・last)・エンジン。全部終わったら E1〜E3',
                      'B-FAST2(BF2-0 が通ったときだけ)'],
            'who_starts': '段 0 の短い学習・段 1 の学習・B-FAST2 の起動は main session(この係は起動しない)',
            'gap_s': 60, 'all_gpu': '/home/jetson/bin/jf-heavy --lock big(学習・エンジン・予測・pytest・Firefox は同時に 1 つ)',
        },
        'timing_estimate_hours': {
            'per_run_rule': '1 本 = 0.768 時間 × (学習サンプル ÷ 6,506) + probe 0.04 + 準備 0.075(v2h-aug2-e60 の 4 本の実測 0.70〜0.80 時間)',
            'stage0': 0.85, 'stage1_total': total_main, 'stage1_per_run': {r['arm']: r['hours_est'] for r in runs[:3]},
            'desk_total': 2.0, 'bfast2_if_trained': 1.4,
            'total_expected': round(0.85 + total_main + 2.0, 1),
        },
        'start_conditions': {'power': '夜の前に AC(または満充電)をユーザーが確かめた。10/26 07:42 に電源が落ちた(final_eval.md 2 章)',
                             'disk': '空き 5 GB 以上(10/26 11 時 12 GB)', 'deadline': '見込み × 1.2 が 10/30 06:00 を超える学習は始めない(机上と 11/1 の封の前に 1 日)',
                             'not_running': 'jetformer-web で走行・収集が無い。jf-heavy の GPU の仕事が無い'},
        'resume': '電源断のあとは同じコマンドに --resume。完全には同じにならない(deviations_v3.md に書く。判定はそのまま)',
        'cut_order': {'order': ['① 段 0 の noTK_noLB・tkB', '② 種 3(3 本)', '③ 種 2(3 本)', '④ B-FAST2'],
                      'keep': '段 0 の same・noTK、段 1 の種 0・1 の 6 本'},
        'deviations': {'file': str(REP / 'deviations_v3.md'), 'rule': 'この決まりから外れたことは全部、時刻・何を・なぜ・判定への影響を書く。変えた腕は参考に下げる'},
        'files_out': {'desk': str(REP / 'desk_v3.json'), 'decisions': str(REP / 'decisions_v3.json'), 'lock2': str(REP / 'protocol_v3_lock2.json'),
                      'seal_opened': 'v2 の seal_opened.json と同じ 1 つ(11/1)'},
        'defaults_untouched': {
            'rule': '既定値(drive.py の DEFAULT_PARAMS の今の値・推奨モデル・drive_settings.json)を変えない。走行の制御に足すものは既定 OFF・起動で OFF。data/raw・labels を変えない',
            'default_params_now': {'start_kick': 0, 'speed_ceiling': 0, 'ceil_all_mps': 4.5, 'ceil_left_mps': 4.1, 'speed_auto': 0, 'auto_target_mps': 4.27, 'zone_caps': 0, 'throttle_gain': 0.5, 'lookahead': 1},
            'sha256_now': {k: sha256(REPO / v) for k, v in (('drive_settings.json', 'data/drive_settings.json'), ('drive_recommended.json', 'data/drive_recommended.json'),
                                                            ('bench_candidates.json', 'data/bench_candidates.json'), ('modes.json', 'data/modes.json'))},
            'note': 'bench_candidates.json は段 plan が 11/3 の日を足す(控えを取る)。ほかは変わらないこと',
        },
        'numbers_sha256': {'protocol_numbers_v3.json': sha256(NUM), 'protocol_numbers_v3.py': sha256(HERE / 'protocol_numbers_v3.py')},
        'stage0_tables_sha256': abl_files,
        'interpretations': [
            '介入の拒否権をステアにした。v2 の B-FAST の拒否権(介入 +2.13)は、ステアで見ると 156.8〜159.2(5 本の平均 158.4)で拒否権にならない',
            '回ごとの直線の門は ±10 µs を下限にした(10/10 の 2SE は 6.5 で、v2 の ±10 より厳しくしない)。10/23 は 2SE の 12.14',
            '止まった場面の後ろ側の門 0.064 は 5 本の平均 + 2SE。diag_data の「0.10 以下」は段 0 の読みだけに使う',
            '学習中の門は、best.pth の probe で見る(学習の中の値と学習のあとの値は同じ式・同じ行)。学習の途中では止めない',
            'v3 のエポック・拡張は v2 の D2(60)・D3(AUG2)の答えを受け継ぐ。v3 で決め直さない',
            '低速のブレーキの門(≤ 0.01)は v2 のまま。10/10 だけの 3 本でも 0.008〜0.208 とばらつくので、v3 も落ちうる(案 2 で正解のもとを外す)',
        ],
    }
    OUT.write_text(json.dumps(P, ensure_ascii=False, indent=1) + '\n')
    print('書いた:', OUT, sha256(OUT))
    print('閾値:', T)
    print('自己試験:', self_test['summary'])
    print('段 1 の計:', total_main, '時間', [(r['arm'], r['hours_est']) for r in runs[:3]])


if __name__ == '__main__':
    sys.exit(main())
