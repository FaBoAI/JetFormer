#!/usr/bin/env python3
"""protocol_v2 に書く数(2026-10-25)。読むだけ。numpy だけ(torch・GPU は使わない)。data/ は変えない。

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- \
        python3 -I /home/jetson/jetformer/experiments/20261025_train_v2/protocol_numbers.py

出力は 1 つだけ: /home/jetson/jetformer_reports/train_v2_20261025/protocol_numbers.json

数えるもの(どれも学習の前に分かっている数だけ。v2・B-FAST の結果は使わない):
  1. 閾値の数え直し。plan_checks.py とは別のコードで、保存した予測(10/12・10/15)から
     B と同じ作り方の 5 本(A_s0・A_s1・v0-s2・v0-s3・B)の H1 の L1・偏り・介入の L1 の標準偏差を出し、
     閾値 = 2 × 標準誤差 を「腕 n 本 対 5 本」(n = 1〜4)と「同じ種 k 組」(k = 2〜4)で並べる。
     plan_checks.json の値(4 対 5・4 組・2 組)と 0.002 µs 以内で合うかを確かめる(合わなければ止まる)。
  2. 印「ジャンルの片寄り」の閾値: H1 のジャンル 4・5 ごとの L1 の、5 本の幅(最大 − 最小)× 2(plan.md 6.2 の 3)。
     ジャンルは clean_driving_20261011 の aligned の mode(フレームごと)。
  3. sha256: 封の 14 本(ラベルの npz・json、raw の log.json、シーンの一覧)・保留(H1・介入)・5 本の予測・B・計画の文書・既定の設定。
     封の記録は中身を開かない(ファイルの sha256 とディレクトリの名前だけ)。
  4. 学習 22 本の時間の見込み(plan.md 4.2 の式)。
  5. 速さの見込みの係数(qualify_plan 1.1 の式と、No.9 の電池の効き)。
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/jetson/jetformer')
DATA = ROOT / 'data'
REP = Path('/home/jetson/jetformer_reports/train_v2_20261025')
PLAN = Path('/home/jetson/jetformer_reports/dataset_v2_plan_20261025')
BENCH = Path('/home/jetson/jetformer_reports/bench_20261024_report')
OUT = REP / 'protocol_numbers.json'
BUNDLE = Path('/home/jetson/jetformer_reports/prep_20261022/desk/pred/desk_inputs_20261022.npz')
TABLE_KEY = '3d339512828b17195d568f1b93ee4a28208f2e85'
ZOO = DATA / 'zoo' / 'policy_zoo_20261012' / 'pred'
RACC = DATA / 'zoo' / 'real_accuracy_20261015' / 'pred'
H1_DS = DATA / 'datasets' / 'clean_driving_20261011'
IV_DS = DATA / 'datasets' / 'intervention_val_20261011'

# B と同じ作り方の 5 本(train_seed_spread.py の FAMILIES['v0(B と同じ作り方)'] と同じ)
FIVE = [
    ('A_s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30', ZOO / 'jetformer_v0_s0.npz', ZOO / 'jetformer_v0_s0_last.npz'),
    ('A_s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30', ZOO / 'jetformer_v0_s1.npz', ZOO / 'jetformer_v0_s1_last.npz'),
    ('v0-s2', 'realacc_20261015-v0-s2', RACC / 'realacc_20261015-v0-s2.npz', RACC / 'realacc_20261015-v0-s2_last.npz'),
    ('v0-s3', 'realacc_20261015-v0-s3', RACC / 'realacc_20261015-v0-s3.npz', RACC / 'realacc_20261015-v0-s3_last.npz'),
    ('B', 'clean_driving_20261011-nocmd-s3-20261011-024626', ZOO / 'B.npz', ZOO / 'B_last.npz'),
]
B_RUN = DATA / 'runs' / 'clean_driving_20261011-nocmd-s3-20261011-024626'

# 封(plan.md 3.2): 10/24 1 回目(13:46〜・No.1〜4)と 3 回目(18:31〜・No.11〜20)
SEALED = {
    '10/24 1 回目(13:46〜・昼・No.1〜4)': ['20261024-134607', '20261024-134907', '20261024-135608', '20261024-135656'],
    '10/24 3 回目(18:31〜・夕方・後輪の修理の後・No.11〜20)': [
        '20261024-183123', '20261024-183437', '20261024-183656', '20261024-183826', '20261024-184043',
        '20261024-184400', '20261024-184549', '20261024-185703', '20261024-185823', '20261024-185926'],
}
# 学習に使う回の記録で、protocol が名前で指すもの(B-FAST の手本・門 G-b の記録)
NAMED_TRAIN_SESSION_LOGS = {
    'bfast_teacher(10/24 2 回目 No.9・No.10)': ['20261024-162033', '20261024-162305'],
    'gate_gb_main(10/23 公式の B No.8・11・15)': ['20261023-210840', '20261023-211209', '20261023-211547'],
}
DOCS = [
    PLAN / 'plan.md', PLAN / 'critique.md', PLAN / 'plan_checks.json', PLAN / 'train_seed_spread.json',
    PLAN / 'inventory.json', PLAN / 'design_numbers.json', PLAN / 'critique_checks.json',
    BENCH / 'qualify_plan.md', BENCH / 'tt_data.json', BENCH / 'fixes.md', BENCH / 'verify.md',
    ROOT / 'experiments' / '20261025_dataset_v2' / 'plan_checks.py',
    ROOT / 'experiments' / '20261025_dataset_v2' / 'train_seed_spread.py',
    ROOT / 'experiments' / '20261025_dataset_v2' / 'inventory.py',
]
DEFAULTS = [DATA / 'drive_settings.json', DATA / 'drive_recommended.json', DATA / 'modes.json',
            DATA / 'bench_candidates.json', ROOT / 'jetformer' / 'drive.py', ROOT / 'tools' / 'train.py',
            ROOT / 'jetformer' / 'train_data.py', ROOT / 'tools' / 'build_ai_dataset.py']

# plan.md 4.2 の時間の式
MS_PER_TRAIN_V0 = 4.96e-3
MS_PER_TRAIN_MSF = 7.81e-3
VAL_SAMPLES = 1559
MS_PER_VAL = 1.54e-3
EPOCH_FIXED_S = 1.9
PREP_S = 4.5 * 60
AUG2_FACTOR = 0.71 / 0.69          # plan.md 4.2: 60 エポックで 0.69 → AUG2 0.71 時間

# 速さの見込み(qualify_plan 1.1・README 7 章)
LAP_S_PER_MPS = -1.50              # 1 周 ≈ 14.52 − 1.50 × 周の速さ p90 [m/s](B の 60 周、相関 −0.91)
MPS_PER_ERPM = 0.5e-3              # jetformer/policies/post.py・desk_lib と同じ
ERPM_PER_US_NO9 = 35.95            # No.9 の電池の効き(eRPM ÷ 指令)
NO9_WINDOW3_MEDIAN_S = 25.75       # No.9 の 3 周の窓 9 個の中央値(走りながら)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def file_entry(p):
    p = Path(p)
    if not p.exists():
        return {'path': str(p), 'missing': True}
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}


def load_pred(path):
    with np.load(path, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('%s: 予測の表の key が違います' % path)
        return np.asarray(z['pred'], np.float64)


def h1_metrics(pred, y, prim, interv, genre):
    d = np.abs(pred - y)
    out = {'l1': float(d[prim].mean()), 'l1_interv': float(d[interv].mean()),
           'steer_bias': float((pred[prim][:, 0, 0] - y[prim][:, 0, 0]).mean()),
           'thr_bias': float((pred[prim][:, 0, 1] - y[prim][:, 0, 1]).mean())}
    for g in (4, 5):
        m = prim & (genre == g)
        out['l1_genre%d' % g] = float(d[m].mean())
    return out


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def thresholds(xs):
    s = sd(xs)
    res = {'mean': round(sum(xs) / len(xs), 3), 'sd': round(s, 3), 'range': round(max(xs) - min(xs), 3),
           'values': [round(x, 3) for x in xs]}
    for n in (1, 2, 3, 4):
        res['thr_%dv5' % n] = round(2 * s * math.sqrt(1.0 / n + 1.0 / 5), 3)
    for k in (2, 3, 4):
        res['thr_pairs%d' % k] = round(2 * s * math.sqrt(2.0 / k), 3)
    return res


def h1_genre(scene, frame, prim):
    g = np.full(len(scene), -1, np.int16)
    for s in sorted(set(scene[prim].tolist())):
        with np.load(H1_DS / 'aligned' / ('%s.npz' % s), allow_pickle=False) as a:
            mode = a['mode']
        m = scene == s
        g[m] = mode[frame[m]]
    return g


def run_hours(train_samples, epochs, msf=False, aug2=False):
    per = MS_PER_TRAIN_MSF if msf else MS_PER_TRAIN_V0
    epoch_s = train_samples * per + VAL_SAMPLES * MS_PER_VAL + EPOCH_FIXED_S
    h = (epochs * epoch_s + PREP_S) / 3600.0
    if aug2:
        h *= AUG2_FACTOR
    return round(epoch_s, 1), round(h, 3)


def main():
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z['key']) != TABLE_KEY:
            raise SystemExit('desk_inputs の key が違います')
        y = np.nan_to_num(np.asarray(z['y'], np.float64))
        prim, interv = z['primary'], z['interv']
        scene, frame = z['scene'], z['frame']
    genre = h1_genre(scene, frame, prim)
    n_g = {g: int((prim & (genre == g)).sum()) for g in (4, 5)}
    assert int(prim.sum()) == 4670 and int(interv.sum()) == 1570, '集合の大きさが計画と違う'
    assert n_g[4] + n_g[5] == 4670, 'H1 にジャンル 4・5 以外がある'
    scenes_g = {g: sorted({s for s in scene[prim & (genre == g)].tolist()}) for g in (4, 5)}

    per_run = {}
    for rid, run, pb, pl in FIVE:
        per_run[rid] = {'run': run, 'best': h1_metrics(load_pred(pb), y, prim, interv, genre),
                        'last': h1_metrics(load_pred(pl), y, prim, interv, genre)}
    thr = {}
    for which in ('best', 'last'):
        for key in ('l1', 'steer_bias', 'thr_bias', 'l1_interv', 'l1_genre4', 'l1_genre5'):
            thr['%s_%s' % (which, key)] = thresholds([per_run[r[0]][which][key] for r in FIVE])

    # plan_checks.json(別のコード)と照らす
    pc = json.load(open(PLAN / 'plan_checks.json', encoding='utf-8'))['seed']
    cmp = {}
    for which in ('best', 'last'):
        for key in ('l1', 'steer_bias', 'thr_bias', 'l1_interv'):
            a = thr['%s_%s' % (which, key)]
            b = pc['%s_%s' % (which, key)]
            pairs = {'mean': (a['mean'], b['mean']), 'sd': (a['sd'], b['sd']),
                     'thr_4v5': (a['thr_4v5'], b['thr_2se_4v5']), 'thr_pairs4': (a['thr_pairs4'], b['thr_2se_4v4']),
                     'thr_pairs2': (a['thr_pairs2'], round(2 * b['se_2v2'], 3))}
            worst = max(abs(u - v) for u, v in pairs.values())
            cmp['%s_%s' % (which, key)] = {'max_abs_diff': round(worst, 4), 'ok': worst <= 0.002}
    if not all(v['ok'] for v in cmp.values()):
        raise SystemExit('plan_checks.json と合いません: %s' % cmp)

    genre_mark = {}
    for which in ('best', 'last'):
        for g in (4, 5):
            t = thr['%s_l1_genre%d' % (which, g)]
            genre_mark['%s_genre%d' % (which, g)] = {'range_5': t['range'], 'thr_2x_range': round(2 * t['range'], 3),
                                                     'sd_5': t['sd'], 'thr_pairs4_ref': t['thr_pairs4']}

    # 封・保留・名前で指す記録の sha256
    seal = {}
    all_lines = []
    for sess, logs in SEALED.items():
        rows = []
        for log in logs:
            raw = DATA / 'raw' / log
            scenes = sorted(d for d in os.listdir(raw) if (raw / d).is_dir())
            ent = {'log': log, 'labels_npz': file_entry(DATA / 'labels' / ('%s.npz' % log)),
                   'labels_json': file_entry(DATA / 'labels' / ('%s.json' % log)),
                   'raw_log_json': file_entry(raw / 'log.json'),
                   'n_scenes': len(scenes), 'scene_list_sha256': sha256_text('\n'.join(scenes) + '\n'),
                   'scenes': scenes}
            rows.append(ent)
            all_lines.append('%s %s %s %s %s' % (log, ent['labels_npz'].get('sha256'), ent['labels_json'].get('sha256'),
                                                 ent['raw_log_json'].get('sha256'), ent['scene_list_sha256']))
        seal[sess] = rows
    seal_manifest = '\n'.join(all_lines) + '\n'
    named = {}
    for what, logs in NAMED_TRAIN_SESSION_LOGS.items():
        named[what] = [{'log': log, 'labels_npz': file_entry(DATA / 'labels' / ('%s.npz' % log)),
                        'labels_json': file_entry(DATA / 'labels' / ('%s.json' % log))} for log in logs]
    splits = json.load(open(H1_DS / 'splits.json', encoding='utf-8'))
    h1_scenes = sorted(splits['val'])
    assert h1_scenes == sorted(set(scene[prim].tolist())), 'H1 のシーンが splits.json の val と違う'
    h1 = {'dataset': str(H1_DS), 'splits_json': file_entry(H1_DS / 'splits.json'),
          'dataset_card': file_entry(H1_DS / 'dataset_card.json'), 'val_scenes': h1_scenes,
          'val_scene_list_sha256': sha256_text('\n'.join(h1_scenes) + '\n'),
          'val_aligned_sha256': {s: sha256_file(H1_DS / 'aligned' / ('%s.npz' % s)) for s in h1_scenes},
          'frames_desk': int(prim.sum()), 'samples_train_log': VAL_SAMPLES,
          'genre_frames': {str(g): n_g[g] for g in (4, 5)}, 'genre_scenes': {str(g): scenes_g[g] for g in (4, 5)}}
    iv_aligned = sorted((IV_DS / 'aligned').glob('*.npz'))
    iv = {'dataset': str(IV_DS), 'dataset_card': file_entry(IV_DS / 'dataset_card.json'),
          'aligned_files': len(iv_aligned),
          'aligned_combined_sha256': sha256_text(''.join('%s %s\n' % (p.name, sha256_file(p)) for p in iv_aligned)),
          'frames_desk': int(interv.sum())}
    refs = {'bundle': file_entry(BUNDLE)}
    for rid, run, pb, pl in FIVE:
        refs[rid] = {'run': run, 'pred_best': file_entry(pb), 'pred_last': file_entry(pl)}
    b_files = {n: file_entry(B_RUN / n) for n in ('best.pth', 'last.pth', 'model_config.json', 'jetformer.onnx',
                                                 'jetformer_fp16.engine', 'engine_check.json', 'result.json')}

    # 時間の見込み(plan.md 4.2)。v2-H 6,562・v2-M 7,757・B-FAST 5,912 + 836 = 6,748
    n_h, n_m, n_bf = 6562, 7757, 5912 + 836
    timing = {}
    for name, n, ep, msf, aug, k in [
            ('v2h-e30', n_h, 30, False, False, 4), ('v2m-e30', n_m, 30, False, False, 4),
            ('X-e60(v2-H)', n_h, 60, False, False, 4), ('X-e100(v2-H)', n_h, 100, False, False, 1),
            ('X-aug2-e60(v2-H)', n_h, 60, False, True, 4), ('X-msf-e60(v2-H)', n_h, 60, True, False, 2),
            ('X-e60(v2-M)', n_m, 60, False, False, 4), ('X-e100(v2-M)', n_m, 100, False, False, 1),
            ('X-aug2-e60(v2-M)', n_m, 60, False, True, 4), ('X-msf-e60(v2-M)', n_m, 60, True, False, 2),
            ('bfast-e15', n_bf, 15, False, False, 3)]:
        es, h = run_hours(n, ep, msf, aug)
        timing[name] = {'train_samples': n, 'epochs': ep, 'seeds': k, 'epoch_s': es, 'hours_each': h,
                        'hours_total': round(h * k, 2), 'hours_each_x1.2': round(h * 1.2, 3)}

    lap_per_us = LAP_S_PER_MPS * MPS_PER_ERPM * ERPM_PER_US_NO9
    speed = {'lap_s_per_mps_p90': LAP_S_PER_MPS, 'mps_per_erpm': MPS_PER_ERPM, 'erpm_per_us_no9': ERPM_PER_US_NO9,
             'lap_delta_s_per_fwd_us': round(lap_per_us, 5), 'no9_window3_median_s': NO9_WINDOW3_MEDIAN_S,
             'formula': 'lap_delta_est_s = %.5f × straight_fwd_vs_b_us、lap3_est_s = %.2f + 3 × lap_delta_est_s(参考)'
                        % (lap_per_us, NO9_WINDOW3_MEDIAN_S)}

    out = {'schema': 'jf-train-v2-protocol-numbers/v1',
           'tool': 'experiments/20261025_train_v2/protocol_numbers.py',
           'note': '学習の前に分かっている数だけ。v2・B-FAST の学習の結果は使っていない。封の記録は sha256 と名前だけ(中身を開いていない)',
           'thresholds': thr, 'plan_checks_compare': cmp, 'genre_mark': genre_mark, 'five_per_run': per_run,
           'h1': h1, 'interv': iv, 'reference_predictions': refs, 'model_b_files': b_files,
           'seal': seal, 'seal_manifest_sha256': sha256_text(seal_manifest), 'seal_manifest_lines': all_lines,
           'named_train_session_logs': named,
           'docs': [file_entry(p) for p in DOCS], 'defaults_now': [file_entry(p) for p in DEFAULTS],
           'timing': timing, 'speed_est': speed}
    REP.mkdir(parents=True, exist_ok=True)
    tmp = str(OUT) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUT)
    print('閾値の照らし合わせ: すべて合う(%d 項目)' % len(cmp))
    print('ジャンルの片寄り:', {k: v['thr_2x_range'] for k, v in genre_mark.items()})
    print('封の一覧の sha256:', out['seal_manifest_sha256'])
    print('書いた:', OUT)


if __name__ == '__main__':
    sys.exit(main())
