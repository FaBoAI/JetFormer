"""evaluation.json に走行の手順での確認・実車のカード・確かめた結果を足す(CPU)。本文があれば evaluation.md も組み立てる。

    python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/make_report.py            # evaluation.json(と、本文があれば evaluation.md)
    python /home/jetson/jetformer_reports/policy_zoo_20261012/eval/make_report.py --print    # 作った表(走行の手順での確認・付録 A)を表示するだけ

読むもの: eval/zoo_eval.json、eval/drive_path_check_*.json、eval/verify/verify.json、eval/tables.md(make_tables.py が作った表)、
eval/report_template.md(本文。手で書いた説明。無ければ evaluation.md は作らない)。
本文の <!--TABLES-->・<!--DRIVE_CHECK-->・<!--APPENDIX_K--> を表で置き換える(表の数字は手で写さない)。
evaluation.md と evaluation.json は一時ファイルに書いてから置き換える(途中で止まっても半端なファイルを残さない)。
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
RUNS = ['B'] + ['%s_s%d' % (p, s) for p in ('jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2') for s in (0, 1)]
SETS = [('human_clean', '人のきれいな走行(全部、4,670 フレーム)'), ('human_clean_genre4', 'ジャンル 4 高速走行(1,922 フレーム)'),
        ('human_clean_genre5', 'ジャンル 5 低速走行(2,748 フレーム)')]
CHECKS = [('model_B', 'モデル B(推奨)'), ('tcp_s0', 'tcp 種 0(机上の次点。カード 2 用)')]


def f(x, nd=1):
    return '—' if x is None else ('%.' + str(nd) + 'f') % x


def appendix_k(r):
    """付録: k1〜k10 の MAE(学習結果ごと、best.pth)"""
    m = r['accuracy']['metrics']
    out = []
    for key, title in SETS:
        for ch, chname in (('steer_mae', 'ステア'), ('throttle_mae', 'スロットル')):
            out.append('**%s — %s の MAE [µs]**(何フレーム先か。1 = 33 ms)\n' % (title, chname))
            out.append('| 学習結果 | ' + ' | '.join('k%d' % k for k in range(1, 11)) + ' |')
            out.append('|---|' + '---|' * 10)
            for n in RUNS:
                v = m[n][key][ch]
                out.append('| %s | %s |' % (n, ' | '.join(f(x) for x in v)))
            out.append('')
    return '\n'.join(out)


def drive_check_table(checks):
    """走行の手順での確認(drive_path_check.py)の表"""
    out = ['| 項目 | ' + ' | '.join(lab for _, lab in CHECKS) + ' |', '|---|' + '---|' * len(CHECKS)]

    def row(name, fn):
        out.append('| %s | %s |' % (name, ' | '.join(fn(checks[k]) for k, _ in CHECKS)))
    row('学習結果', lambda c: '`%s`' % c['run'])
    row('確かめた時刻', lambda c: c['checked_at'])
    row('方策・出力', lambda c: '%s・%s' % (c['policy'], '・'.join(c['model_info']['outputs'])))
    row('drive.model_info の drivable / check_io', lambda c: '%s / %s' % ('○' if c['model_info']['drivable'] else '×',
                                                                        '○' if c['check_io']['ok'] else '×'))
    row('エンジンの読み込み [s]・大きさ [MB]', lambda c: '%s・%s' % (f(c['load_s'], 2), f(c['engine_mb'], 1)))
    row('実フレーム 64 個の TensorRT − PyTorch FP32 の actions 最大 / p99 [µs](ゲート 2.5)',
        lambda c: '%s / %s' % (f(c['accuracy']['actions_max_us'], 2), f(c['accuracy']['actions_p99_us'], 2)))
    row('tcp の通過点 [m] / 目標速度の期待値 [m/s] の差の最大',
        lambda c: ('%s / %s' % (f(c['accuracy']['waypoints_max_m'], 4), f(c['accuracy']['speed_exp_max_mps'], 4))
                   if 'waypoints_max_m' in c['accuracy'] else '—'))
    row('run_numpy 中央 / p99 [ms]・エンジンだけ 中央 [ms]',
        lambda c: '%s / %s・%s' % (f(c['latency_ms']['run_numpy_median'], 2), f(c['latency_ms']['run_numpy_p99'], 2),
                                  f(c['latency_ms']['engine_median'], 2)))
    order = ['aba_settings', 'default', 'tcp_mode1', 'tcp_mode2', 'tcp_mode3', 'mode0_cap2', 'mode1_cap2_scale0.8', 'ensemble_j2']
    seen = {k for c in checks.values() for k in c['drive_loop']['cases']}
    cases = [k for k in order if k in seen] + sorted(seen - set(order))

    def loop_cell(c, k):
        x = c['drive_loop']['cases'].get(k)
        if not x:
            return '—'
        ok = x['error'] is None and x['holds'] == 0 and x['pca_writes'] == x['frames_fed'] and x.get('model_pred_finite')
        return '%s(%d フレーム・中立 %d・PWM %d 回・ループと直接の計算の差 %s µs)' % (
            '○' if ok else '×', x['frames_fed'], x['holds'], x['pca_writes'], f(x.get('loop_vs_direct_max_us'), 1))
    for k in cases:
        row('Driver._loop: %s' % k, lambda c, k=k: loop_cell(c, k))
    row('全体', lambda c: '**合格**' if c['ok'] else '**不合格**')
    return '\n'.join(out)


def main():
    r = json.load(open(HERE / 'zoo_eval.json', encoding='utf-8'))
    checks = {k: json.load(open(HERE / ('drive_path_check_%s.json' % k), encoding='utf-8')) for k, _ in CHECKS}
    verify = json.load(open(HERE / 'verify' / 'verify.json', encoding='utf-8'))
    made = {'<!--TABLES-->': (HERE / 'tables.md').read_text(encoding='utf-8'), '<!--DRIVE_CHECK-->': drive_check_table(checks),
            '<!--APPENDIX_K-->': appendix_k(r)}
    if '--print' in sys.argv[1:]:
        for mark in ('<!--DRIVE_CHECK-->', '<!--APPENDIX_K-->'):
            print(mark)
            print(made[mark].strip())
            print()
        return
    template = HERE / 'report_template.md'
    if template.exists():
        body = template.read_text(encoding='utf-8')
        for mark, text in made.items():
            if mark not in body:
                raise SystemExit('本文に %s がありません' % mark)
            body = body.replace(mark, text.strip())
        if '<!--' in body:
            raise SystemExit('置き換えていない印が残っています')
        tmp = REPORT / 'evaluation.md.tmp'
        tmp.write_text(body, encoding='utf-8')
        os.replace(tmp, REPORT / 'evaluation.md')
        print('書きました:', REPORT / 'evaluation.md')
        md_status = '作成済み(%s)' % datetime.datetime.now().isoformat(timespec='seconds')
    else:
        print('本文(%s)が無いので evaluation.md は作りません' % template)
        md_status = ('未作成。本文を eval/report_template.md に置いて make_report.py を実行すると、表を差し込んだ evaluation.md ができる'
                     '(評価の段の担当は .md の報告を書けないため、本文は主に渡した)')
    # evaluation.json に足す(make_tables.py の要約はそのまま)
    s = json.load(open(REPORT / 'evaluation.json', encoding='utf-8'))
    s['report_written_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    s['evaluation_md'] = md_status
    s['verification'] = {k: verify[k] for k in ('checked_at', 'zoo_eval_diffs', 'tables_identical', 'note')}
    s['drive_path_check'] = {
        k: {'run': c['run'], 'checked_at': c['checked_at'], 'ok': c['ok'], 'policy': c['policy'],
            'outputs': c['model_info']['outputs'], 'actions_max_us': c['accuracy']['actions_max_us'],
            'run_numpy_median_ms': c['latency_ms']['run_numpy_median'], 'engine_median_ms': c['latency_ms']['engine_median'],
            'loop_cases': {n: {'error': x['error'], 'frames': x['frames_fed'], 'holds': x['holds'], 'pwm_writes': x['pca_writes'],
                               'loop_vs_direct_max_us': x.get('loop_vs_direct_max_us')}
                           for n, x in c['drive_loop']['cases'].items()},
            'file': str(HERE / ('drive_path_check_%s.json' % k)),
            'note': 'PCA9685 は偽物(I2C に触れない)。ループの fps・latency・infer_ms は流す側と CPU を取り合う道具の値で、実車の値ではない'}
        for k, c in checks.items()}
    s['car_card'] = {
        'card1': {'name': '基準値の走行(A/B/A の形、B = 推奨 = モデル B)', 'model_run': r['recommended']['run'],
                  'engine': r['recommended']['engine'], 'genre': 5,
                  'fixed': {'steer_gain': 1.3, 'throttle_gain': 1.0, 'lookahead': 3, 'steer_trim_us': 0.0, 'tof_guard': 1,
                            'recovery': 1, 'esc_reverse': 1, 'ensemble_j': 0, 'speed_cap': 0.0, 'tcp_mode': 0},
                  'laps_per_block_min': 43, 'laps_per_block_target': 60, 'order': ['モデル B', 'モデル B', 'モデル B']},
        'card2': {'name': '机上の次点を試すときの形(任意・探索。判定の根拠にはならない)',
                  'model_run': r['runs']['tcp_s0'], 'engine': str(Path(r['recommended']['engine']).parents[1]
                                                                  / r['runs']['tcp_s0'] / 'jetformer_fp16.engine'),
                  'order': ['モデル B', 'tcp 種 0(tcp_mode 0・speed_cap 0)', 'モデル B'], 'genre': 5,
                  'requires': 'jetformer-web を policy zoo のコードで再起動した後(今動いているのは 11:13 の古いコード)'},
        'stop_rules': ['最初の 10 周で衝突(水平 2.5G)が 20 回を超えた', '後退の指令が 0.5 秒以上暴走した・リカバリー以外で車が後ろへ動いた',
                       '推論による中立(hold)が起きた', 'ステアが限界に張り付く・左右に振れ続ける(制御の発散)',
                       '同じ場所で 3 周続けて衝突した', '衝突の直前の速度の中央値が最初の A より 0.5 m/s 以上高い',
                       'R1 の旋回中にブレーキが出た', '車の電池が打ち切りの電圧を下回った(その電池の試行は終える)',
                       '基板の再起動・Jetson の電源断(その走行は無効、周を数え直す)'],
        'judge': 'spec §5.7: B の「1 周あたりの交代 + 衝突」が両方の A より少なく、ラップタイムが A の平均の 5% より遅くない。'
                 '各 43 周で見分けられるのは衝突/周の −40〜−50% 以上の差だけ'}
    s['figures'] = {n: str(REPORT / 'fig' / n) for n in ('eval_primary.png', 'eval_primary_zoom.png', 'eval_r1_de.png',
                                                         'eval_speed_vs_l1.png', 'eval_mae_by_step.png', 'eval_robustness.png',
                                                         'eval_tcp_modes.png')}
    s['files']['tables'] = str(HERE / 'tables.md')
    s['files']['verify'] = str(HERE / 'verify' / 'verify.json')
    tmp = REPORT / 'evaluation.json.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, REPORT / 'evaluation.json')
    print('書きました:', REPORT / 'evaluation.json')


if __name__ == '__main__':
    main()
