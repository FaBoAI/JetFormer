#!/usr/bin/env python3
"""シミュレーター + 実機の混合学習(案 C)の学習の段のまとめ(CPU だけ。torch を読まない)

    cd ~/jetformer && python /home/jetson/jetformer_reports/sim_cotrain_20261013/train/cotrain_summary.py

各学習結果の result.json・state.json・エンジンの確認から、train/summary.json と図(fig/)を書く:
    fig/train_val_curves.png   実機の検証 L1 のエポックごとの曲線: C(混ぜる)と A(10/12、実機だけ)、種 0・1
    fig/train_val_by_genre.png ジャンル 4(高速)・5(低速)別の検証 L1: C と A
    fig/train_ft_curves.png    微調整 12 エポック: C+FT と A+FT(点線 = 元の best.pth の検証 L1)
    fig/train_loss_train.png   学習 L1(C は シミュレーターのフレームを含む平均)と検証 L1
出来ていない学習結果は飛ばす(途中でも実行できる)。
"""

import json
import os
import statistics
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
FIG = REPORT / 'fig'
RUNS = REPO / 'data' / 'runs'
ENGINE_CHECK_DIR = HERE / 'engine_check'
ARMS = {
    'A': 'clean_driving_20261011-jetformer_v0-s%d-nocmd-stride3-e30',
    'C': 'simreal_mix_20261013-cotrain-s%d',
    'C+FT': 'simreal_mix_20261013-cotrain-ft-s%d',
    'A+FT': 'clean_driving_20261011-jetformer_v0-s%d-ft-real-lr1e-4-e12',
}
RUN_B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
SEEDS = (0, 1)
# 色(dataviz の既定の並び: 1 青・2 橙・3 アクア・4 黄)。腕ごとに固定(順位で塗らない)
COLOR = {'A': '#2a78d6', 'C': '#eb6834', 'C+FT': '#1baf7a', 'A+FT': '#eda100', 'B': '#52514e'}
LABEL = {'A': 'A(実機だけ・10/12)', 'C': 'C(シミュレーター+実機)', 'C+FT': 'C+FT(C → 実機で微調整)',
         'A+FT': 'A+FT(A → 実機で微調整・対照)', 'B': 'モデル B'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d9d8d4', '#fcfcfb'


def load(path, default=None):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def collect():
    state = load(HERE / 'state.json', {})
    out = {}
    for arm, pat in ARMS.items():
        for s in SEEDS:
            name = pat % s
            res = load(RUNS / name / 'result.json')
            if not res or res.get('state') != 'done':
                continue
            hist = res['history']
            secs = [h['seconds'] for h in hist]
            rs = (state.get('runs') or {}).get('%s/s%d' % (arm, s), {})
            chk = load(ENGINE_CHECK_DIR / ('%s.json' % name)) if arm != 'A' else None
            out['%s/s%d' % (arm, s)] = {
                'run': name, 'arm': arm, 'seed': s, 'epochs': len(hist),
                'best_val_us': res['best_val_us'], 'best_epoch': res['best_epoch'],
                'best_val_per_mode_us': res.get('best_val_per_mode_us'),
                'last_val_us': hist[-1]['val_us'], 'last_val_per_mode_us': hist[-1].get('val_per_mode_us'),
                'best_minus_last_us': round(res['best_val_us'] - hist[-1]['val_us'], 1),
                'train_last_us': hist[-1]['train_us'],
                'train_samples': res['train_samples'], 'val_samples': res['val_samples'],
                'train_mode_counts': res.get('train_mode_counts'), 'num_modes': len(res.get('modes') or []) and
                max(m['id'] for m in res['modes']) + 1,
                'steer_center_us': res.get('steer_center_us'), 'throttle_neutral_us': res.get('throttle_neutral_us'),
                'epoch1_s': secs[0], 'epoch2_s': secs[1] if len(secs) > 1 else None,
                'epoch_s_median': statistics.median(secs), 'epochs_s_sum': round(sum(secs), 1),
                'duration_s': res.get('duration_s'), 'peak_mem_mb': res.get('peak_mem_mb'),
                'rl_options': res.get('rl_options'),
                'train_wall_s': rs.get('train_wall_s'), 'attempts': rs.get('attempts'),
                'resumes': rs.get('resumes'), 'engine_wall_s': (rs.get('engine') or {}).get('wall_s'),
                'check_wall_s': (rs.get('check') or {}).get('wall_s'),
                'engine_check': chk and {
                    'all_gates': chk.get('all_gates'), 'drivable': (chk.get('model_info') or {}).get('drivable'),
                    'trained_modes': (chk.get('model_info') or {}).get('trained_modes'),
                    'num_modes': (chk.get('model_info') or {}).get('num_modes'),
                    'check_io': (chk.get('check_io') or {}).get('ok'),
                    'actions_max_us': (chk.get('accuracy') or {}).get('actions_max_us'),
                    'actions_p99_us': (chk.get('accuracy') or {}).get('actions_p99_us'),
                    'finite': (chk.get('accuracy') or {}).get('finite'),
                    'run_numpy_median_ms': (chk.get('latency_ms') or {}).get('run_numpy_median'),
                    'run_numpy_p99_ms': (chk.get('latency_ms') or {}).get('run_numpy_p99'),
                    'engine_median_ms': (chk.get('latency_ms') or {}).get('engine_median'),
                    'engine_mb': chk.get('engine_mb'), 'load_s': chk.get('load_s'),
                    'frames': {k: (chk.get('frames') or {}).get(k) for k in ('dataset', 'val_frames', 'modes')}},
                'history': hist,
            }
    b = load(RUNS / RUN_B / 'result.json')
    if b:
        out['B'] = {'run': RUN_B, 'best_val_us': b['best_val_us'], 'best_epoch': b['best_epoch'],
                    'last_val_us': b['history'][-1]['val_us'], 'history': b['history']}
    return state, out


def setup_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'lines.linewidth': 2})
    return plt


def _curve(ax, runs, arm, key=lambda h: h['val_us'], x0=0, label_last=True):
    for s in SEEDS:
        r = runs.get('%s/s%d' % (arm, s))
        if not r:
            continue
        xs = [h['epoch'] + x0 for h in r['history']]
        ys = [key(h) for h in r['history']]
        ax.plot(xs, ys, color=COLOR[arm], ls='-' if s == 0 else '--', lw=2,
                label='%s 種 %d' % (LABEL[arm], s))
        be = r['best_epoch']
        ax.plot([be + x0], [key(r['history'][be - 1])], 'o', ms=8, color=COLOR[arm], mec=SURF, mew=2, zorder=5)


def fig_val(plt, runs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={'width_ratios': [1.15, 1]})
    for ax, ylim, title in ((axes[0], None, '実機の検証 L1(全体)'), (axes[1], (47, 56), '同じ図の拡大(47〜56 µs)')):
        for arm in ('A', 'C'):
            _curve(ax, runs, arm)
        if 'B' in runs:
            hb = runs['B']['history']
            ax.plot([h['epoch'] for h in hb], [h['val_us'] for h in hb], color=COLOR['B'], lw=1.2, ls=':',
                    label='モデル B(参考)')
        ax.set_xlabel('エポック')
        ax.set_ylabel('検証 L1 [µs](実機の 11 シーン・1,559 サンプル)')
        ax.set_title(title, color=INK)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(44, 68)
    axes[0].legend(loc='upper right', frameon=False)
    fig.suptitle('TransFuser軽量型(JetFormer v0): シミュレーターを混ぜた学習 C と実機だけの A の検証の曲線(● = best.pth のエポック)',
                 fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'train_val_curves.png', dpi=110)
    plt.close(fig)


def fig_genre(plt, runs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharex=True)
    for ax, g, name in ((axes[0], '4', 'ジャンル 4(高速走行)'), (axes[1], '5', 'ジャンル 5(低速走行)')):
        for arm in ('A', 'C'):
            _curve(ax, runs, arm, key=lambda h, g=g: h['val_per_mode_us'].get(g, float('nan')))
        ax.set_title('%s の検証 L1' % name, color=INK)
        ax.set_xlabel('エポック')
        ax.set_ylabel('検証 L1 [µs]')
        ax.set_ylim(39 if g == '5' else 58, 50 if g == '5' else 72)
    axes[0].legend(loc='upper right', frameon=False)
    fig.suptitle('ジャンル別の検証 L1(実機): C と A(縦軸は拡大。序盤のエポックは枠の外)', fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'train_val_by_genre.png', dpi=110)
    plt.close(fig)


def fig_ft(plt, runs):
    have = [a for a in ('C+FT', 'A+FT') if any('%s/s%d' % (a, s) in runs for s in SEEDS)]
    if not have:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, s in zip(axes, SEEDS):
        for arm, src in (('C+FT', 'C'), ('A+FT', 'A')):
            r = runs.get('%s/s%d' % (arm, s))
            if not r:
                continue
            xs = [h['epoch'] for h in r['history']]
            ax.plot(xs, [h['val_us'] for h in r['history']], color=COLOR[arm], lw=2, label=LABEL[arm])
            ax.plot([r['best_epoch']], [r['best_val_us']], 'o', ms=8, color=COLOR[arm], mec=SURF, mew=2, zorder=5)
            init = runs.get('%s/s%d' % (src, s))
            if init:
                ax.axhline(init['best_val_us'], color=COLOR[src], lw=1.4, ls=':',
                           label='元の %s 種 %d の best.pth(%.1f µs)' % (src, s, init['best_val_us']))
        ax.set_title('種 %d: 微調整(実機だけ・lr 1e-4・12 エポック)' % s, color=INK)
        ax.set_xlabel('微調整のエポック')
        ax.set_ylim(46, 54)
        ax.legend(loc='upper right', frameon=False)
    axes[0].set_ylabel('検証 L1 [µs](実機の 11 シーン)')
    fig.suptitle('微調整の曲線: C+FT と対照の A+FT(● = best.pth、点線 = 微調整の前)', fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / 'train_ft_curves.png', dpi=110)
    plt.close(fig)
    return True


def fig_train_loss(plt, runs):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for arm in ('A', 'C'):
        _curve(ax, runs, arm, key=lambda h: h['train_us'])
    ax.set_yscale('log')
    ax.set_xlabel('エポック')
    ax.set_ylabel('学習 L1 [µs](対数)')
    ax.set_title('学習 L1(C はシミュレーターのフレーム 2/3 を含む平均。色の揺らぎあり)', color=INK)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / 'train_loss_train.png', dpi=110)
    plt.close(fig)


def _f(v, fmt='%.1f'):
    return '—' if v is None else fmt % v


def tables_md(state, runs):
    """training.md に貼る表(train/tables.md)"""
    rows = []
    rows.append('| 腕・種 | 学習結果(data/runs/) | 学習 / 検証 サンプル | ジャンル数 | best の検証 L1 [µs](エポック) | '
                '高速 / 低速 [µs] | 最後 [µs] | best − last | 学習 L1(最後) |')
    rows.append('|---|---|---|---|---|---|---|---|---|')
    for key in ['A/s0', 'A/s1', 'C/s0', 'C/s1', 'C+FT/s0', 'C+FT/s1', 'A+FT/s0', 'A+FT/s1']:
        r = runs.get(key)
        if not r:
            continue
        pm = r.get('best_val_per_mode_us') or {}
        rows.append('| %s | `%s` | %s / %s | %s | %s(%d) | %s / %s | %s | %s | %s |' % (
            key.replace('/s', ' 種 '), r['run'], '{:,}'.format(r['train_samples']), '{:,}'.format(r['val_samples']),
            r['num_modes'], _f(r['best_val_us']), r['best_epoch'], _f(pm.get('4')), _f(pm.get('5')),
            _f(r['last_val_us']), _f(r['best_minus_last_us'], '%+.1f'), _f(r['train_last_us'])))
    if 'B' in runs:
        b = runs['B']
        rows.append('| モデル B(参考) | `%s` | 5,912 / 1,559 | 6 | %s(%d) | — | %s | — | — |' % (
            b['run'], _f(b['best_val_us']), b['best_epoch'], _f(b['last_val_us'])))
    t = ['## 表 1: 学習の結果(学習中の検証。主指標ではない)', ''] + rows + ['']
    rows = ['| 腕・種 | エポック | 1 エポック目 [秒] | 2 エポック目 [秒] | 中央 [秒] | エポックの合計 [分] | 学習の実時間 [分] | '
            '再開 | エンジン [秒] | 確認 [秒] | ピークのメモリ [MB] |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for key in ['C/s0', 'C/s1', 'C+FT/s0', 'C+FT/s1', 'A+FT/s0', 'A+FT/s1']:
        r = runs.get(key)
        if not r:
            continue
        rows.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            key.replace('/s', ' 種 '), r['epochs'], _f(r['epoch1_s']), _f(r['epoch2_s']), _f(r['epoch_s_median']),
            _f(r['epochs_s_sum'] / 60), _f((r['train_wall_s'] or 0) / 60), r.get('resumes') or 0,
            _f(r['engine_wall_s'], '%.0f'), _f(r['check_wall_s'], '%.0f'), _f(r['peak_mem_mb'], '%.0f')))
    t += ['## 表 2: 実時間', ''] + rows + ['']
    rows = ['| 腕・種 | drivable・check_io | ジャンル数・走行で選べるジャンル | actions の差 最大 / p99 [µs] | 精度 ≤ 2.5 µs | '
            'run_numpy 中央 / p99 [ms] | エンジンだけ 中央 [ms] | エンジン [MB] | 全ゲート |', '|---|---|---|---|---|---|---|---|---|']
    for key in ['C/s0', 'C/s1', 'C+FT/s0', 'C+FT/s1', 'A+FT/s0', 'A+FT/s1']:
        r = runs.get(key)
        c = r and r.get('engine_check')
        if not c:
            continue
        rows.append('| %s | %s・%s | %s・%s | %s / %s | %s | %s / %s | %s | %s | %s |' % (
            key.replace('/s', ' 種 '), '合格' if c['drivable'] else '不合格', '合格' if c['check_io'] else '不合格',
            c['num_modes'], c['trained_modes'], _f(c['actions_max_us'], '%.2f'), _f(c['actions_p99_us'], '%.2f'),
            '合格' if (c['actions_max_us'] is not None and c['actions_max_us'] <= 2.5) else '不合格',
            _f(c['run_numpy_median_ms'], '%.2f'), _f(c['run_numpy_p99_ms'], '%.2f'), _f(c['engine_median_ms'], '%.2f'),
            _f(c['engine_mb']), '合格' if c['all_gates'] else '不合格'))
    t += ['## 表 3: エンジンの確認', ''] + rows + ['']
    (HERE / 'tables.md').write_text('\n'.join(t), encoding='utf-8')


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    state, runs = collect()
    plt = setup_plt()
    figs = []
    if any(k.startswith('C/') for k in runs):
        fig_val(plt, runs)
        fig_genre(plt, runs)
        fig_train_loss(plt, runs)
        figs += ['train_val_curves.png', 'train_val_by_genre.png', 'train_loss_train.png']
    if fig_ft(plt, runs):
        figs.append('train_ft_curves.png')
    slim = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in runs.items()}
    for k, v in runs.items():          # 曲線は val・ジャンル別だけ残す
        slim[k]['curve'] = [{'epoch': h['epoch'], 'train_us': h['train_us'], 'val_us': h['val_us'],
                             'val_per_mode_us': h.get('val_per_mode_us'), 'seconds': h['seconds']} for h in v['history']]
    write_json(HERE / 'summary.json', {'runs': slim, 'figures': figs, 'state_updated': state.get('updated'),
                                       'protocol_sha256': state.get('protocol_sha256'), 'all_done': state.get('all_done')})
    tables_md(state, runs)
    for k, v in slim.items():
        print('%-8s %-45s best %5.1f(e%2s) last %5.1f  epoch1 %s s  median %s s  wall %s s' % (
            k, v['run'], v['best_val_us'], v.get('best_epoch'), v['last_val_us'], v.get('epoch1_s'),
            v.get('epoch_s_median'), v.get('train_wall_s')))
    print('図:', figs)


if __name__ == '__main__':
    main()
