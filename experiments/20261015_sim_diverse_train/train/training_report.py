#!/usr/bin/env python3
"""学習の段の図と表(sim_diverse_train_20261015)

    /home/jetson/miniconda/envs/jetracer/bin/python -I ~/jetformer/experiments/20261015_sim_diverse_train/train/training_report.py

10/14 の experiments/20261014_sim_dart_margin/train/train_report.py の写しを、今日の腕に合わせて直したもの(オフライン L1 の図は外した。
オフラインの誤差・IMU の位相は protocol の offline_diagnostics で、別の段が測る)。読むだけで、学習結果は変えない。
  fig/train_curves.png       Q1 の腕のエポックごとの検証 L1(各腕の学習データ自身の val)と学習 L1。
                             破線 = 前日までの A0・A1・A2(作り直していない)、実線 = 今日の S-DIV・S-MOD(・A4)。丸 = 最良のエポック
  fig/train_curves_q2.png    Q2 の腕(C-DIV)と A・C の実機の検証 L1(同じ実機の検証 1,559 サンプル)と学習 L1
  fig/train_timeline.png     今日の学習の時刻(試行ごと。終わり方で色分け)
  train/summary.json         図と training.md の表の元の数字
  train/tables.md            training.md に貼る表
  train/runs.json            次の段(閉ループ評価・実機の保留データの評価)が使う対応表: 腕・種 → 学習結果・一式・sha256
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

REPORT = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015')
TRAIN = REPORT / 'train'
FIG = REPORT / 'fig'
RUNS = Path('/home/jetson/jetformer/data/runs')
BESTLAP_BUNDLES = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/bundles')
DART_BUNDLES = Path('/home/jetson/jetformer_reports/sim_dart_20261014/bundles')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
# 色は腕に付ける(順位ではない)。前日の A1・A2 は 10/14 の図と同じ色
COLOR = {'A0': '#8a8984', 'A1': '#2a78d6', 'A2': '#1baf7a', 'S-DIV': '#eb6834', 'S-MOD': '#4a3aa7', 'A4': '#e34948',
         'A': '#8a8984', 'C': '#2a78d6', 'C-DIV': '#eb6834'}
LABEL = {'A0': 'A0 = 最速の線のきれいなデータ 20 分(10/13)', 'A1': 'A1 = A0 + DART 10 分(10/14)',
         'A2': 'A2 = 余裕の線 10 分(10/14)', 'S-DIV': 'S-DIV = 5 モード + DART + 余裕の線 45 分(今日)',
         'S-MOD': 'S-MOD = 5 モードだけ 25 分(今日)', 'A4': 'A4 = 一番良い腕 + DAgger(今日)',
         'A': 'A = 実機だけ(10/12)', 'C': 'C = 実機 + 前の教師のシミュレーター 60 シーン(10/13)',
         'C-DIV': 'C-DIV = 実機 + 5 モードの見本 60 シーン(今日)'}
VALSET = {'A0': 'A0 の val 12 シーン', 'A1': 'A0 + D1 の val 18 シーン', 'A2': 'D2 の val 6 シーン',
          'S-DIV': '5 モード + D1 + D2 の val 27 シーン', 'S-MOD': '5 モードの val 15 シーン', 'A4': 'S-DIV と同じ val 27 シーン(DAgger の 12 シーンはすべて学習)'}
Q1_REFS = [('A0/s0', 'simrl_bestlap_20261013-SR-s0', BESTLAP_BUNDLES), ('A1/s0', 'simdart_20261014-A1-s0', DART_BUNDLES),
           ('A2/s0', 'simdart_20261014-A2-s0', DART_BUNDLES)]
Q2_REFS = [('A/s0', 'clean_driving_20261011-jetformer_v0-s0-nocmd-stride3-e30'),
           ('A/s1', 'clean_driving_20261011-jetformer_v0-s1-nocmd-stride3-e30'),
           ('C/s0', 'simreal_mix_20261013-cotrain-s0'), ('C/s1', 'simreal_mix_20261013-cotrain-s1')]
Q1_ARMS = ('S-DIV', 'S-MOD', 'A4')
TODAY_ORDER = ['S-DIV/s0', 'C-DIV/s0', 'C-DIV/s1', 'S-MOD/s0', 'A4/s0']


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'IPAGothic', 'TakaoGothic', 'VL Gothic', 'Droid Sans Fallback'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name
    return None


def style(ax, title, ylabel='', xlabel=''):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc='left')
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def attempt_kinds(rs):
    """試行の内訳: 終わった・自分の上限(cgroup)で止まった・見張りで止めた・電源断などで終わりが記録されていない"""
    out = {'done': 0, 'mem_cap': 0, 'watchdog': 0, 'no_exit_recorded': 0, 'operator': 0, 'other': 0}
    for a in rs.get('attempts', []):
        if a.get('exit') is None:
            out['no_exit_recorded'] += 1
        elif a.get('exit') == 'stopped_by_operator':
            out['operator'] += 1
        elif a.get('watchdog_killed'):
            out['watchdog'] += 1
        elif a.get('exit') == 0:
            out['done'] += 1
        elif a.get('exit') in (-9, 137):
            out['mem_cap'] += 1
        else:
            out['other'] += 1
    return out


def bundle_info(bundle_dir):
    if bundle_dir is None:
        return None
    bc = load(bundle_dir / 'bundle_check.json', {})
    ex = load(bundle_dir / 'export.json', {})
    hl = load(bundle_dir / 'http_load_check.json', {})
    files = ex.get('files') or {}
    return {'dir': str(bundle_dir), 'ok': bc.get('ok'), 'zip': ex.get('zip'), 'zip_sha256': ex.get('zip_sha256'),
            'zip_bytes': ex.get('zip_bytes'), 'onnx_vs_torch_max_us': bc.get('onnx_vs_torch_max_us'),
            'defaults': bc.get('defaults'), 'mode_input': bc.get('mode_input'),
            'onnx_sha256': (files.get('jetformer.onnx') or {}).get('sha256'),
            'model_config_sha256': (files.get('model_config.json') or {}).get('sha256'),
            'io_spec_sha256': (files.get('io_spec.json') or {}).get('sha256'),
            'http_load_ok': hl.get('ok'), 'http_load_s': ((hl.get('results') or [{}])[0]).get('load_s'),
            'http_port': hl.get('port')}


def row_of(key, run_name, rs, bundle_dir, reference):
    run = RUNS / run_name
    res = load(run / 'result.json')
    if not res or res.get('state') != 'done':
        return None
    hist = res['history']
    arm, seed = key.split('/s')
    attempts = (rs or {}).get('attempts', [])
    return {
        'key': key, 'arm': arm, 'seed': int(seed), 'run': run_name, 'dataset': res.get('dataset'),
        'train_samples': res.get('train_samples'), 'val_samples': res.get('val_samples'),
        'train_mode_counts': res.get('train_mode_counts'), 'impact_masked_frames': res.get('impact_masked_frames'),
        'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
        'last_val_us': hist[-1]['val_us'], 'last_train_us': hist[-1]['train_us'], 'epochs': len(hist),
        'val_us_epoch': [h['val_us'] for h in hist], 'train_us_epoch': [h['train_us'] for h in hist],
        'epoch_seconds': [h['seconds'] for h in hist],
        'epoch_s_median': float(np.median([h['seconds'] for h in hist])),
        'epoch_seconds_sum': round(sum(h['seconds'] for h in hist), 1),
        'attempts': len(attempts), 'attempt_kinds': attempt_kinds(rs or {}),
        'attempt_log': [{k: a.get(k) for k in ('started', 'ended', 'exit', 'wall_s', 'workers', 'resume_from_epoch',
                                                 'watchdog_killed', 'mem_cap')} for a in attempts],
        'wall_s': round(sum(a.get('wall_s') or 0 for a in attempts), 1) if attempts else None,
        'workers': (rs or {}).get('workers'), 'peak_mem_mb': res.get('peak_mem_mb'),
        'resumes': sum(1 for a in attempts if a.get('resume_from_epoch')),
        'steer_center_us_trained': res.get('steer_center_us'), 'throttle_neutral_us_trained': res.get('throttle_neutral_us'),
        'reference_only': reference, 'bundle': bundle_info(bundle_dir),
        'files': {fn: (run / fn).stat().st_size for fn in ('best.pth', 'last.pth', 'jetformer.onnx', 'model_config.json')
                  if (run / fn).exists()},
    }


def collect():
    state = load(TRAIN / 'state.json', {})
    q1, q2 = [], []
    for key, name, bdir in Q1_REFS:
        r = row_of(key, name, None, bdir / name, True)
        if r:
            q1.append(r)
    for key, name in Q2_REFS:
        r = row_of(key, name, None, None, True)
        if r:
            q2.append(r)
    runs = state.get('runs') or {}
    keys = [k for k in TODAY_ORDER if k in runs] + [k for k in runs if k not in TODAY_ORDER]
    for key in keys:
        rs = runs[key]
        arm = rs.get('arm') or key.split('/')[0]
        name = rs.get('run')
        if not name:
            continue
        r = row_of(key, name, rs, (REPORT / 'bundles' / name) if arm in Q1_ARMS else None, False)
        if r:
            (q1 if arm in Q1_ARMS else q2).append(r)
    return state, q1, q2


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', fontsize=8, color=INK2, va='center')
    ax.plot([x], [y], 'o', ms=4, color=color)


def fig_curves_q1(rows, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax, field, title in ((axes[0], 'val_us_epoch', '検証 L1(各腕の学習データ自身の val。腕ごとに val が違うので高さは比べられない)'),
                             (axes[1], 'train_us_epoch', '学習 L1(色の揺らぎあり)')):
        for r in rows:
            y = r[field]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, '--' if r['reference_only'] else '-', color=COLOR[r['arm']], lw=1.4 if r['reference_only'] else 2.2,
                    label='%s(%s)最良 %.1f µs' % (LABEL[r['arm']], VALSET.get(r['arm'], ''), r['best_val_us'])
                    if field == 'val_us_epoch' else None)
            if field == 'val_us_epoch':
                b = r['best_epoch']
                ax.plot([b], [y[b - 1]], 'o', ms=8, color=COLOR[r['arm']], mec=SURFACE, mew=2)
                if not r['reference_only']:      # 前日の腕の値は凡例に書く(線が近くて重なるため)
                    ax.annotate('%s %.1f' % (r['arm'], y[b - 1]), (b, y[b - 1]), xytext=(6, -12 if r['arm'] == 'S-MOD' else 8),
                                textcoords='offset points', fontsize=8.5, color=INK2)
        style(ax, title, 'L1 [µs](10 ステップ・舵とスロットルの平均)', 'エポック')
        ax.set_ylim(bottom=0)
        ax.set_xlim(0.5, 33)
    axes[0].legend(fontsize=7.6, frameon=False, labelcolor=INK2, loc='upper right')
    fig.suptitle('Q1 の腕の学習曲線(jetformer_v0・ImageNet から・30 エポック・間引き 3)。破線 = 前日までの腕(作り直していない)・丸 = 最良のエポック',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_curves.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_curves_q2(rows, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    for ax, field, title in ((axes[0], 'val_us_epoch', '実機の検証 L1(11 シーン・1,559 サンプル。3 つの腕で同じ val)'),
                             (axes[1], 'train_us_epoch', '学習 L1(実機 + シミュレーター。色の揺らぎあり)')):
        seen = set()
        for r in rows:
            y = r[field]
            x = np.arange(1, len(y) + 1)
            lab = None
            if field == 'val_us_epoch' and r['arm'] not in seen:
                lab = LABEL[r['arm']] + '(種 0 = 太線・種 1 = 細線)'
                seen.add(r['arm'])
            ax.plot(x, y, '--' if r['reference_only'] else '-', color=COLOR[r['arm']],
                    lw=(2.2 if r['seed'] == 0 else 1.1), label=lab)
            if field == 'val_us_epoch':
                b = r['best_epoch']
                ax.plot([b], [y[b - 1]], 'o', ms=7, color=COLOR[r['arm']], mec=SURFACE, mew=2)
        style(ax, title, 'L1 [µs](10 ステップ・舵とスロットルの平均)', 'エポック')
        ax.set_xlim(0.5, 31)
    vals = [v for r in rows for v in r['val_us_epoch']]
    if vals:
        lo = min(vals)
        axes[0].set_ylim(lo - 2, min(max(vals), lo + 25))
    axes[0].legend(fontsize=7.8, frameon=False, labelcolor=INK2, loc='upper right')
    fig.suptitle('Q2 の腕の学習曲線(参考。主指標は人のきれいな走行 4,670 フレームの L1 で、別の段が測る)。破線 = 10/12〜10/13 の腕・丸 = 最良のエポック',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_curves_q2.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _t(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')


def fig_timeline(rows, plt):
    rows = [r for r in rows if not r['reference_only'] and r['attempt_log']]
    if not rows:
        return False
    import matplotlib.dates as mdates
    kind_color = {'done': '#1baf7a', 'mem_cap': '#eda100', 'watchdog': '#e87ba4', 'cut': '#e34948', 'operator': '#2a78d6'}
    kind_label = {'done': '学習が終わった試行', 'mem_cap': '自分のメモリの上限で止まった試行(oom-kill)',
                  'watchdog': '全体の空きメモリの見張りで止めた試行', 'cut': '終わりが記録されていない試行(電源断など)',
                  'operator': '学習の段が止めた試行(workers 0 が遅いため。deviations.md)'}
    fig, ax = plt.subplots(figsize=(13, 2.2 + 0.6 * len(rows)), facecolor=SURFACE)
    used = set()
    for i, r in enumerate(rows):
        for a in r['attempt_log']:
            if not a.get('started'):
                continue
            t0 = _t(a['started'])
            if a.get('ended'):
                t1 = _t(a['ended'])
                kind = 'watchdog' if a.get('watchdog_killed') else 'done' if a.get('exit') == 0 else \
                    'mem_cap' if a.get('exit') in (-9, 137) else 'operator' if a.get('exit') == 'stopped_by_operator' else 'cut'
            else:
                t1 = t0 + datetime.timedelta(seconds=60)
                kind = 'cut'
            ax.barh(i, mdates.date2num(t1) - mdates.date2num(t0), left=mdates.date2num(t0), height=0.5,
                    color=kind_color[kind], edgecolor=SURFACE, lw=2, label=kind_label[kind] if kind not in used else None)
            used.add(kind)
            if (t1 - t0).total_seconds() > 420:     # 広い棒には workers と上限を書く
                ax.text(mdates.date2num(t0) + (mdates.date2num(t1) - mdates.date2num(t0)) / 2, i,
                        'workers %s・%s' % (a.get('workers'), a.get('mem_cap') or ''), ha='center', va='center', fontsize=7.5,
                        color=INK if kind == 'mem_cap' else '#ffffff')
        last = max(_t(a['ended']) for a in r['attempt_log'] if a.get('ended')) if any(a.get('ended') for a in r['attempt_log']) else None
        if last:
            ax.annotate('%s 種 %d  %.0f 分' % (r['arm'], r['seed'], (r['wall_s'] or 0) / 60), (mdates.date2num(last), i),
                        xytext=(6, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(['%s 種 %d' % (r['arm'], r['seed']) for r in rows])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    style(ax, '今日の学習の時刻(2026-10-15、GPU は 1 本ずつ)', '', '時刻(JST)')
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc='upper left', bbox_to_anchor=(0, -0.38), ncol=2)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + (hi - lo) * 0.12)
    fig.tight_layout()
    fig.savefig(FIG / 'train_timeline.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def fmt_attempts(r):
    if r['reference_only']:
        return '—'
    k = r['attempt_kinds']
    return '%d(%d / %d / %d / %d)' % (r['attempts'], k['mem_cap'], k['watchdog'], k['no_exit_recorded'], k['operator'])


def tables(q1, q2):
    out = ['### Q1 の腕(各腕の学習データ自身の val)', '',
           '| 腕・種 | 学習結果 | 学習 / 検証のサンプル | 最良の検証 L1 [µs] | 最良のエポック | 30 エポック目の検証 / 学習 L1 [µs] | 1 エポック [秒](中央値) | 30 エポックの計算 [分] | 学習の壁時計 [分] | 試行(上限 / 見張り / 終わり不明 / 止めた) | workers | GPU のピーク [MB] |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in q1:
        out.append('| %s 種 %d%s | `%s` | %d / %d | **%.1f** | %d | %.1f / %.1f | %.0f | %.0f | %s | %s | %s | %s |' % (
            r['arm'], r['seed'], '(参照)' if r['reference_only'] else '', r['run'], r['train_samples'], r['val_samples'],
            r['best_val_us'], r['best_epoch'], r['last_val_us'], r['last_train_us'], r['epoch_s_median'],
            r['epoch_seconds_sum'] / 60, '—' if r['wall_s'] is None else '%.0f' % (r['wall_s'] / 60), fmt_attempts(r),
            '—' if r['reference_only'] else r['workers'], r['peak_mem_mb']))
    out += ['', '### Q2 の腕(実機の検証 1,559 サンプル。3 つの腕で同じ val)', '',
            '| 腕・種 | 学習結果 | 学習 / 検証のサンプル | 最良の実機の検証 L1 [µs] | 最良のエポック | 30 エポック目の検証 / 学習 L1 [µs] | 1 エポック [秒](中央値) | 30 エポックの計算 [分] | 学習の壁時計 [分] | 試行(上限 / 見張り / 終わり不明 / 止めた) | workers | 中心(学習が出した値)[µs] |',
            '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in q2:
        out.append('| %s 種 %d%s | `%s` | %d / %d | **%.1f** | %d | %.1f / %.1f | %.0f | %.0f | %s | %s | %s | %s / %s |' % (
            r['arm'], r['seed'], '(参照)' if r['reference_only'] else '', r['run'], r['train_samples'], r['val_samples'],
            r['best_val_us'], r['best_epoch'], r['last_val_us'], r['last_train_us'], r['epoch_s_median'],
            r['epoch_seconds_sum'] / 60, '—' if r['wall_s'] is None else '%.0f' % (r['wall_s'] / 60), fmt_attempts(r),
            '—' if r['reference_only'] else r['workers'], r['steer_center_us_trained'], r['throttle_neutral_us_trained']))
    out += ['', '### シミュレーター用の一式(Q1 の腕)', '',
            '| 腕・種 | 一式の ZIP | ONNX と PyTorch の差 最大 [µs] | ブリッジの読み込み(関数) | ブリッジの読み込み(HTTP) | 読み込み [秒] | ZIP [MB] | ZIP の sha256(頭 16 文字) |',
            '|---|---|---|---|---|---|---|---|']
    for r in q1:
        b = r['bundle'] or {}
        out.append('| %s 種 %d%s | `%s` | %s | %s | %s | %s | %s | `%s` |' % (
            r['arm'], r['seed'], '(参照)' if r['reference_only'] else '', b.get('zip'), b.get('onnx_vs_torch_max_us'),
            {True: '合格', False: '不合格'}.get(b.get('ok'), '—'), {True: '合格', False: '不合格'}.get(b.get('http_load_ok'), '—'),
            b.get('http_load_s') if b.get('http_load_s') is not None else '—',
            '%.1f' % (b['zip_bytes'] / 2 ** 20) if b.get('zip_bytes') else '—', (b.get('zip_sha256') or '')[:16]))
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- training.md(1 本ごとに作り直す)
PROTO_ORDER = ['S-DIV/s0', 'C-DIV/s0', 'C-DIV/s1', 'S-MOD/s0', 'A4/s0']
ROLE = {'S-DIV/s0': 'Q1 の主の腕(削らない)', 'C-DIV/s0': 'Q2 の主の腕(削らない)', 'C-DIV/s1': 'Q2 の種 1(最後に削る)',
        'S-MOD/s0': 'Q1 の 2 つ目の腕(2 番目に削る)', 'A4/s0': 'DAgger(最初に削る。DAgger の段のデータが要る)'}
DATA_TEXT = {
    'S-DIV': '5 モードの見本 75 シーン + DART(D1)30 + 余裕の線(D2)30 = 135 シーン・45 分(学習 108 / 検証 27 シーン)',
    'S-MOD': '5 モードの見本 75 シーンだけ = 25 分(学習 60 / 検証 15 シーン)',
    'C-DIV': '実機 clean_driving_20261011 の 52 シーン(学習 41 / 検証 11)+ 5 モードの学習の 12 シーンずつ = 60 シーン・36,000 フレーム(すべて学習)',
    'A4': '一番良い Q1 の腕のデータ + DAgger の 12 シーン(10 エポック続き・lr 1e-4)'}


def train_log_progress(name):
    """学習のログの最後の epoch の行 → (エポック, 行)"""
    p = TRAIN / 'logs' / ('%s.train.log' % name)
    try:
        lines = [l for l in p.read_text(encoding='utf-8', errors='replace').splitlines() if l.startswith('epoch ')]
    except OSError:
        return None, None
    if not lines:
        return 0, None
    try:
        return int(lines[-1].split()[1]), lines[-1]
    except (ValueError, IndexError):
        return None, lines[-1]


def driver_running():
    import subprocess
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
    rows = []
    for l in out[1:]:
        parts = l.strip().split(None, 1)
        if len(parts) == 2 and parts[1].split()[0].endswith('python') and 'diverse_train.py train' in parts[1]:
            rows.append(l.strip()[:300])
    return rows


def progress_rows(state, done_keys):
    runs = state.get('runs') or {}
    drv = driver_running()
    queued = []
    for o in state.get('orders') or []:
        if any(str(o.get('pid')) == d.split()[0] for d in drv):
            queued += o.get('order') or []
    rows = []
    for key in PROTO_ORDER:
        rs = runs.get(key) or {}
        arm, seed = key.split('/s')
        name = rs.get('run') or 'simdiv_20261015-%s-s%s' % (arm.replace('-', ''), seed)
        if key in done_keys:
            st = '済み'
        elif rs.get('attempts'):
            ep, _ = train_log_progress(name)
            st = ('学習中(%s / %d エポック済み)' % (ep, 10 if arm == 'A4' else 30)) if key in queued else \
                 ('止まっている(%s エポック済み。同じコマンドで続きから)' % ep)
        elif key in queued:
            st = '順番待ち(道具の順番に入っている)'
        else:
            st = 'まだ始めていない'
        rows.append((key, ROLE[key], name, st))
    return rows, drv


def write_training_md(state, q1, q2, figs, tables_text):
    today_q1 = [r for r in q1 if not r['reference_only']]
    today_q2 = [r for r in q2 if not r['reference_only']]
    done_keys = {r['key'] for r in today_q1 + today_q2}
    ref = {r['key']: r for r in q1 + q2 if r['reference_only']}
    prog, drv = progress_rows(state, done_keys)
    starts = [a['started'] for rs in (state.get('runs') or {}).values() for a in rs.get('attempts', []) if a.get('started')]
    ends = [a['ended'] for rs in (state.get('runs') or {}).values() for a in rs.get('attempts', []) if a.get('ended')]
    L = []
    L += ['# JetFormer の学習の段 — sim_diverse_train_20261015', '',
          '- この文書は `training_report.py` が、学習が 1 本終わるたびに作り直します(最後の更新 %s)。' % datetime.datetime.now().strftime('%m/%d %H:%M'),
          '- 学習の時刻: %s 〜 %s(JST)。決まりは `protocol.json`(書き換えていない)。手順と違えたことは `deviations.md` の「学習の段」にあります。' % (
              min(starts)[11:16] if starts else '—', max(ends)[11:16] if ends else '—'),
          '- 実車・PCA9685・I2C には触れていません。実車を走らせる試験は、この作業の範囲の外です。',
          '- FP16 のエンジンは作っていません。Q1 はブリッジの onnxruntime(CPU)で ONNX を動かし、Q2 は protocol が「作らない」(no_engine)と決めているためです。', '',
          '## 0. 進み具合', '',
          '| 腕・種 | 役割 | 学習結果の名前 | 状態 |', '|---|---|---|---|']
    for key, role, name, st in prog:
        L.append('| %s | %s | `%s` | %s |' % (key.replace('/s', ' 種 '), role, name, st))
    L += ['', '- 道具: %s' % ('動いている(PID %s、`%s`)' % (drv[0].split()[0], drv[0][drv[0].find('diverse_train.py'):])
                             if drv else '動いていない'), '']
    # ---- 1. 結論
    L += ['---', '', '## 1. 結論(今の時点)', '']
    for r in today_q1:
        gap = r['last_val_us'] - r['last_train_us']
        L.append('- **%s 種 %d** は %d エポックまで学習しました。最良の検証 L1 は **%.1f µs**(%d エポック目)、最後のエポックは %.1f µs です。'
                 % (r['arm'], r['seed'], r['epochs'], r['best_val_us'], r['best_epoch'], r['last_val_us']))
        L.append('  - val は %s で、A0〜A2 の val(A0 %.1f・A1 %.1f・A2 %.1f µs)とは中身が違います。高さで「良い・悪い」は言えません。'
                 % (VALSET[r['arm']], ref['A0/s0']['best_val_us'], ref['A1/s0']['best_val_us'], ref['A2/s0']['best_val_us']))
        L.append('  - 最後のエポックの「検証 − 学習」の差は %.1f µs です(A0 %.1f・A1 %.1f・A2 %.1f µs)。'
                 % (gap, ref['A0/s0']['last_val_us'] - ref['A0/s0']['last_train_us'], ref['A1/s0']['last_val_us'] - ref['A1/s0']['last_train_us'],
                    ref['A2/s0']['last_val_us'] - ref['A2/s0']['last_train_us']))
        b = r['bundle'] or {}
        L.append('  - シミュレーター用の一式: ブリッジの関数での確かめ %s、HTTP での読み込み %s、ONNX と PyTorch の差は最大 %s µs。'
                 % ({True: '合格', False: '**不合格**'}.get(b.get('ok'), 'まだ'), {True: '合格', False: '**不合格**'}.get(b.get('http_load_ok'), 'まだ'),
                    b.get('onnx_vs_torch_max_us')))
    if today_q2:
        a = [ref[k]['best_val_us'] for k in ('A/s0', 'A/s1')]
        c = [ref[k]['best_val_us'] for k in ('C/s0', 'C/s1')]
        for r in today_q2:
            L.append('- **%s 種 %d** は %d エポックまで学習しました。実機の検証(11 シーン・1,559 サンプル)の最良の L1 は **%.1f µs**(%d エポック目)、最後は %.1f µs です。'
                     % (r['arm'], r['seed'], r['epochs'], r['best_val_us'], r['best_epoch'], r['last_val_us']))
        L.append('  - 同じ実機の検証で、A は %.1f・%.1f µs、C は %.1f・%.1f µs でした(学習の val。0.1 µs に丸めた値)。'
                 % (a[0], a[1], c[0], c[1]))
        L.append('  - ただしこれは最良のエポックを選んだのと同じ val なので、判定には使いません。判定は次の段が、人のきれいな走行 4,670 フレームの L1 で、10/13 の決まりのまま行います。')
    if not today_q1 and not today_q2:
        L.append('- まだ学習が終わった腕はありません。')
    L += ['- **この段では「走れるか」「実機の精度が上がったか」はまだ分かりません。** 完走は閉ループの評価、実機の精度は保留データの評価(次の段)で測ります。', '']
    # ---- 2. 学習のしかた
    L += ['---', '', '## 2. 学習のしかた', '',
          '- コマンドは protocol.json の作り方のとおりです(10/14 の A0〜A2、10/13 の C と同じ)。',
          '  `tools/train.py --policy jetformer_v0 --data <腕のフォルダー> --epochs 30 --lr 3e-4 --seed <種> --stride 3 --batch 64 --workers <W> --no-command-input --impact-g 2.5`',
          '  - JetFormer TransFuser軽量型(jetformer_v0)。ImageNet から学習。AdamW(wd 1e-4)と cosine。最良のエポックは各腕の学習データの val で選ぶ。',
          '  - workers は計算の結果を変えません(色の揺らしの乱数の流れだけが変わる)。',
          '- データ(準備の段で、サンプル数・ジャンル・離脱復活の外したフレーム・実機の分けが protocol と同じことを確かめた。`prepare.md`):']
    for arm in ('S-DIV', 'C-DIV', 'S-MOD', 'A4'):
        L.append('  - **%s**: %s' % (arm, DATA_TEXT[arm]))
    L += ['- 道具は `experiments/20261015_sim_diverse_train/train/diverse_train.py`(10/14 の dart_train.py の写し)。1 本ずつ、次を自動で行います。',
          '  1. 学習(jf-heavy・lock big・毎エポック last.pth。電源断のあとは同じコマンドで続きから)',
          '  2. run.json(管理画面の一覧。Q1 の腕は「シミュレーター専用(ジャンル 8)。実車では使わない」)',
          '  3. Q1 の腕だけ: シミュレーター用の一式(export_bundle.py)と、ユーザーのブリッジの関数での確かめ(check_bundle.py)',
          '  4. Q1 の腕だけ: HTTP での読み込みの確かめ(http_load_check.py、口 8825)',
          '  5. この文書と図と表(training_report.py)',
          '- **メモリと workers**(deviations.md):',
          '  - S-DIV は workers 1・上限 3G で始めると 30 秒で上限に当たり、workers 0 では 1 エポック 254〜311 秒(見込みの約 2.4 倍)でした。',
          '    読み込み(画像の読み込みと色の揺らし)が学習と同じスレッドで動き、GPU が半分ほど空いていたためです。',
          '  - そこで上限を 3400M → 3800M に上げて workers 1 で続けました(1 エポック 140〜160 秒)。読み込みのプロセスがあると、cgroup の kernel のメモリが約 900 MB 増えます。',
          '  - C-DIV・S-MOD も上限 3800M で学習します(workers は protocol のとおり C-DIV 2・S-MOD 1 から始め、上限に当たれば道具が下げる)。',
          '    C-DIV 種 0 の workers 2 は、始めの 3 分で anon 約 1.6 GB + kernel 約 1.26 GB(計 約 2.9 GB)でした。3G でも入った可能性はありますが、',
          '    S-DIV が 3400M でも途中で上限に当たったので、途中で止まって workers を下げることがないよう 3800M のままにしました(結果は変わらない)。',
          '  - 上限はこの学習のプロセスだけのものです。全体の空きメモリが 200 MB 未満で 8 秒続くと道具が自分の学習を止め(見張り)、choom 1000 で OOM のときも学習が先に止まります。', '']
    # ---- 3〜5
    L += ['---', '', '## 3. Q1 の腕の学習曲線', '']
    if 'fig/train_curves.png' in figs:
        L += ['![Q1 の学習曲線](fig/train_curves.png)', '',
              '- 破線は前日までの A0(灰)・A1(青)・A2(緑)。実線は今日の S-DIV(橙)・S-MOD(紫)。丸は最良のエポック。',
              '- 腕ごとに val の中身が違うので、左の図の高さは腕どうしで比べられません(同じ val での比べは、評価の段のオフラインの誤差で行う)。', '']
    L += [tables_text.split('### Q2')[0].strip(), '',
          '- 「試行」は学習を始めた回数です。内訳は、自分のメモリの上限で止まった回 / 全体の空きの見張りで止めた回 / 終わりが記録されていない回(電源断など)/ 学習の段が止めた回(workers を替えるため)。',
          '- エポックごとの数字は、各学習結果の `result.json` の history にあります。ログは `train/logs/<学習結果>.train.log` です。', '',
          '---', '', '## 4. Q2 の腕の学習曲線', '']
    if 'fig/train_curves_q2.png' in figs:
        L += ['![Q2 の学習曲線](fig/train_curves_q2.png)', '',
              '- 灰の破線は A(実機だけ)、青の破線は C(実機 + 前の教師のシミュレーター)。橙の実線が今日の C-DIV。太線 = 種 0、細線 = 種 1。',
              '- 左の図の val は 3 つの腕で同じ(実機の 11 シーン・1,559 サンプル)。右の学習 L1 は、C・C-DIV ではシミュレーターのフレームを含むので低く出ます。', '']
    L += ['### Q2' + tables_text.split('### Q2')[1].split('### シミュレーター用の一式')[0].rstrip(), '']
    L += ['---', '', '## 5. 時刻', '']
    if 'fig/train_timeline.png' in figs:
        L += ['![学習の時刻](fig/train_timeline.png)', '']
    L += ['| 腕・種 | 試行 | 始め | 終わり | 終わり方 | workers | 上限 | 続きから(エポック) |', '|---|---|---|---|---|---|---|---|']
    for r in today_q1 + today_q2:
        for i, a in enumerate(r['attempt_log'], 1):
            ex = a.get('exit')
            how = '済み' if ex == 0 else '上限で止まった(oom-kill)' if ex in (-9, 137) else '学習の段が止めた' if ex == 'stopped_by_operator' \
                else '見張りが止めた' if a.get('watchdog_killed') else '記録なし' if ex is None else str(ex)
            L.append('| %s 種 %d | %d | %s | %s | %s | %s | %s | %s |' % (
                r['arm'], r['seed'], i, (a.get('started') or '')[11:19], (a.get('ended') or '—')[11:19], how, a.get('workers'),
                a.get('mem_cap') or '—', a.get('resume_from_epoch') or '—'))
    L += ['']
    # ---- 6. 一式
    L += ['---', '', '## 6. シミュレーター用の一式と、ブリッジでの確かめ', '',
          '### シミュレーター用の一式' + tables_text.split('### シミュレーター用の一式')[1].rstrip(), '',
          '- 作り方は 10/14 と同じです(export_bundle.py の写し。変えたのはパスと作業名だけ)。',
          '  - ONNX は学習結果のものをそのまま写す(sha256 が同じことを確かめる)。',
          '  - model_config.json: 中心 1442 / 1498 µs、制御の遅れ 1 フレーム、ジャンル 8・ゲイン 1.0・先読み 1、作者の参照モデルと同じ物理・センサーの欄。',
          '    学習が出した中心(シミュレーターでは止まっているフレームが走り出しだけなので、約 1399 / 1188.5 µs)は simulator_bundle に記録だけする。',
          '  - io_spec.json は直した model_config から作る。',
          '- **ブリッジの関数での確かめ**(check_bundle.py、シミュレーターの Python): アプリの「モデル ZIP を読み込む」と同じ順に save_bundle → activate_model を呼び、',
          '  ジャンル 8・学習したジャンル [8]・one-hot の長さ 9・中心・遅れ・ゲイン・先読みと、ONNX(onnxruntime CPU)と PyTorch の出力の差(val の 16 フレーム)を確かめる。',
          '- **HTTP での確かめ**(http_load_check.py): ブリッジの写しをこの作業の口 8825 で立て、POST /api/models/bundle → POST /api/models/activate(ジャンル 8)→ GET /api/status。',
          '  - ブリッジの写し(`sim/bridge.py`・`sim/jetformer_policy.py`)はユーザーのものとバイト単位で同じ。jetformer-web のユーザーのブリッジには触っていません。',
          '- アプリの写し `sim/`: 10/14 の写しから記録(runtime/datasets 2.5 GB)を除いて写しました(75 ファイル・9.6 MB)。',
          '  dist の app.js・sim-core.js・index.html・rc-speed-controller.js と bridge.py・jetformer_policy.py が、ユーザーのシミュレーターと 10/14 の写しの両方と同じ(`train/sim_copy_check.json`)。', '']
    # ---- 7. 引き継ぎ
    L += ['---', '', '## 7. 次の段への引き継ぎ', '',
          '- 対応表: `train/runs.json`(腕・種 → 学習結果・一式の ZIP・sha256)。A0〜A2・A・C は前の作業の学習結果(作り直していない)。',
          '- 評価する一式: ' + ('、'.join('`bundles/%s.zip`' % r['run'] for r in today_q1) if today_q1 else 'まだ無い'),
          '- 学習の途中で lock big(Firefox・GPU の予測)を使いたいときは、`train/PAUSE` を置きます。道具は今の学習を終えてから次を始めずに待ちます。',
          '  使い終わったら `train/PAUSE` を消すと、次の学習が始まります(jf-heavy の lock big を待っているだけでも、今の学習が終われば先に取れます)。',
          '- 電源断のあとは、`cd ~/jetformer && nohup python -u experiments/20261015_sim_diverse_train/train/diverse_train.py train <腕/種 ...> --mem-cap 3800M >> train/logs/driver.log &` で続きから。',
          '  済んだ学習結果は飛ばします。', '']
    # ---- 8. 限界
    L += ['---', '', '## 8. 限界・注意', '',
          '- **学習の val の L1 は、閉ループの成績でも実機の主指標でもありません。** S-DIV の val には線・速さの違う 5 モードが入るので、ひとつの頭では「平均」しかまねできない分の誤差が残ります。',
          '- **種は Q1 の腕で 0 だけ**です(protocol のとおり)。',
          '- **IMU の位相**: 間引き 3 の学習は、シミュレーターの IMU の 3 つの位相のうち位相 0 しか見ていません(A0〜A2 と同じ)。位相ごとの誤差は評価の段のオフラインの誤差で測ります。',
          '- 途中で workers を替えた腕は、色の揺らしの乱数の流れと、続きからのエポックの並べ方が、止めずに学習した場合と少し違います(電源断のときと同じ扱い)。', '']
    # ---- 9. ファイル
    L += ['---', '', '## 9. ファイル', '',
          '- 報告: `training.md`(この文書)、`train/tables.md`・`train/summary.json`・`train/runs.json`・`train/state.json`・`train/sim_copy_check.json`',
          '- 図: ' + '・'.join('`%s`' % f for f in figs),
          '- ログ: `train/logs/`(driver.log・<学習結果>.train.log・.bundle.log・.http_load.log・training_report.log)',
          '- 一式: `bundles/<学習結果>/`(export.json・bundle_check.json・http_load_check.json)と `bundles/<学習結果>.zip`',
          '- 学習結果: ' + '・'.join('`/home/jetson/jetformer/data/runs/%s`' % r['run'] for r in today_q1 + today_q2),
          '- 道具: `/home/jetson/jetformer/experiments/20261015_sim_diverse_train/train/`(diverse_train.py・training_report.py・export_bundle.py・check_bundle.py・http_load_check.py・nothp.py)', '']
    tmp = REPORT / 'training.md.tmp'
    tmp.write_text('\n'.join(L), encoding='utf-8')
    tmp.replace(REPORT / 'training.md')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    state, q1, q2 = collect()
    figs = []
    if q1:
        fig_curves_q1(q1, plt)
        figs.append('fig/train_curves.png')
    if q2:
        fig_curves_q2(q2, plt)
        figs.append('fig/train_curves_q2.png')
    if fig_timeline(q1 + q2, plt):
        figs.append('fig/train_timeline.png')
    summary = {'generated_at': datetime.datetime.now().isoformat(timespec='seconds'), 'q1': q1, 'q2': q2,
               'state_updated': state.get('updated'), 'figures': figs}
    tmp = TRAIN / 'summary.json.tmp'
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(TRAIN / 'summary.json')
    tables_text = tables(q1, q2)
    (TRAIN / 'tables.md').write_text(tables_text, encoding='utf-8')
    write_training_md(state, q1, q2, figs, tables_text)
    runs = {'schema': 'sim-diverse-train-runs/v1',
            'note': '次の段(閉ループ評価・実機の保留データの評価)が使う対応表。A0〜A2・A・C は前の作業の学習結果(作り直していない)',
            'runs': {}}
    for r in q1 + q2:
        b = r['bundle'] or {}
        runs['runs'][r['key']] = {
            'arm': r['arm'], 'seed': r['seed'], 'run': r['run'], 'run_dir': str(RUNS / r['run']), 'dataset': r['dataset'],
            'best_val_us': r['best_val_us'], 'best_epoch': r['best_epoch'], 'reference_only': r['reference_only'],
            'question': 'Q1' if r in q1 else 'Q2',
            'bundle_dir': b.get('dir'), 'bundle_zip': b.get('zip'), 'bundle_zip_sha256': b.get('zip_sha256'),
            'onnx_sha256': b.get('onnx_sha256'), 'bundle_model_config_sha256': b.get('model_config_sha256'),
            'bundle_io_spec_sha256': b.get('io_spec_sha256'), 'bundle_ok': b.get('ok'), 'http_load_ok': b.get('http_load_ok'),
            'mode_id': 8 if r in q1 else None, 'num_modes': 9}
    tmp = TRAIN / 'runs.json.tmp'
    tmp.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.replace(TRAIN / 'runs.json')
    print(json.dumps([{k: r[k] for k in ('key', 'best_val_us', 'best_epoch', 'last_val_us', 'epoch_s_median')} for r in q1 + q2],
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
