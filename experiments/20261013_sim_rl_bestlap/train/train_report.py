#!/usr/bin/env python3
"""学習の段の図と表(2026-10-14)

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I ~/jetformer/experiments/20261013_sim_rl_bestlap/train/train_report.py

  fig/train_curves.png       エポックごとの検証 L1(各データ自身の val)と学習 L1。S-R = 青、S-T = 橙、種 0 = 実線・種 1 = 破線。
                             最良のエポックに丸
  fig/train_offline_l1.png   オフライン L1 のステップごと(1〜10 フレーム先)。行 = RL の val・教師の val、列 = 舵・スロットル
  fig/train_frame_phase.png  各データ自身の val の L1 を、フレーム番号を 3 で割った余りごとに(IMU の位相)
  train/summary.json         図と train.md の表の元の数字(state.json・result.json・engine_check・bundle_check・offline_l1 から)
  train/tables.md            train.md に貼る表
  train/runs.json            次の段(閉ループ評価)が使う対応表: 腕・種 → 学習結果・一式のフォルダー・ZIP・sha256
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORT = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013')
TRAIN = REPORT / 'train'
FIG = REPORT / 'fig'
RUNS = Path('/home/jetson/jetformer/data/runs')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ARM_COLOR = {'S-R': '#2a78d6', 'S-T': '#eb6834'}
ARM_LABEL = {'S-R': 'S-R(RL のデータ)', 'S-T': 'S-T(教師のデータ)'}
SEED_STYLE = {0: '-', 1: '--'}


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
    """試行の内訳: 終わった・自分の上限(cgroup)で止まった・見張りで止めた・連続実行を始め直した(学習の前)"""
    out = {'done': 0, 'mem_cap': 0, 'watchdog': 0, 'restarted_before_start': 0, 'other': 0}
    for a in rs.get('attempts_before_memory_fix', []) + rs.get('attempts', []):
        if a.get('exit') is None:
            out['restarted_before_start'] += 1
        elif a.get('watchdog_killed'):
            out['watchdog'] += 1
        elif a.get('exit') == 0:
            out['done'] += 1
        elif a.get('exit') in (-9, 137):
            out['mem_cap'] += 1
        else:
            out['other'] += 1
    return out


def collect():
    state = load(TRAIN / 'state.json', {})
    rows = []
    for key in state.get('order', []):
        rs = state.get('runs', {}).get(key)
        if not rs:
            continue
        run = RUNS / rs['run']
        res = load(run / 'result.json')
        if not res or res.get('state') != 'done':
            continue
        ec = load(TRAIN / 'engine_check' / ('%s.json' % rs['run']), {})
        bc = load(REPORT / 'bundles' / rs['run'] / 'bundle_check.json', {})
        ex = load(REPORT / 'bundles' / rs['run'] / 'export.json', {})
        hist = res['history']
        rows.append({
            'key': key, 'arm': rs['arm'], 'seed': rs['seed'], 'run': rs['run'], 'dataset': res['dataset'],
            'train_samples': res['train_samples'], 'val_samples': res['val_samples'],
            'best_val_us': res['best_val_us'], 'best_epoch': res['best_epoch'],
            'best_val_per_mode_us': res.get('best_val_per_mode_us'),
            'last_val_us': hist[-1]['val_us'], 'last_train_us': hist[-1]['train_us'],
            'min_train_us': min(h['train_us'] for h in hist),
            'val_us_epoch': [h['val_us'] for h in hist], 'train_us_epoch': [h['train_us'] for h in hist],
            'epoch_s_median': float(np.median([h['seconds'] for h in hist])),
            'duration_s': res.get('duration_s'), 'train_wall_s': rs.get('train_wall_s'),
            'attempts': len(rs.get('attempts', [])) + len(rs.get('attempts_before_memory_fix', [])),
            'attempt_kinds': attempt_kinds(rs), 'epoch_seconds_sum': round(sum(h['seconds'] for h in hist), 1),
            'workers': rs.get('workers'), 'peak_mem_mb': res.get('peak_mem_mb'),
            'steer_center_us_trained': res.get('steer_center_us'), 'throttle_neutral_us_trained': res.get('throttle_neutral_us'),
            'engine': {k: (ec.get('accuracy') or {}).get(k) for k in ('actions_max_us', 'actions_p99_us', 'actions_mean_us', 'finite')}
            | {k: (ec.get('latency_ms') or {}).get(k) for k in ('run_numpy_median', 'run_numpy_p99', 'engine_median')}
            | {'all_gates': ec.get('all_gates'), 'engine_mb': ec.get('engine_mb'), 'check_io': (ec.get('check_io') or {}).get('ok')},
            'bundle': {'ok': bc.get('ok'), 'zip': ex.get('zip'), 'zip_sha256': ex.get('zip_sha256'), 'zip_bytes': ex.get('zip_bytes'),
                       'onnx_vs_torch_max_us': bc.get('onnx_vs_torch_max_us'), 'gates': bc.get('gates'),
                       'activated': bc.get('activated'), 'defaults': bc.get('defaults'),
                       'mode_input': bc.get('mode_input'), 'onnx_sha256': (ex.get('files') or {}).get('jetformer.onnx', {}).get('sha256'),
                       'model_config_sha256': (ex.get('files') or {}).get('model_config.json', {}).get('sha256'),
                       'io_spec_sha256': (ex.get('files') or {}).get('io_spec.json', {}).get('sha256')},
        })
    return state, rows


def fig_curves(rows, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=SURFACE)
    for ax, field, title in ((axes[0], 'val_us_epoch', '検証 L1(各データ自身の val 12 シーン)'),
                             (axes[1], 'train_us_epoch', '学習 L1(色の揺らぎあり)')):
        for r in rows:
            y = r[field]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, SEED_STYLE[r['seed']], color=ARM_COLOR[r['arm']], lw=2,
                    label='%s 種 %d' % (ARM_LABEL[r['arm']], r['seed']))
            if field == 'val_us_epoch':
                b = r['best_epoch']
                ax.plot([b], [y[b - 1]], 'o', ms=8, color=ARM_COLOR[r['arm']], mec=SURFACE, mew=2)
                ax.annotate('種 %d: %.1f' % (r['seed'], y[b - 1]), (b, y[b - 1]),
                            xytext=(-62 if r['seed'] == 0 else -62, 8 if r['seed'] == 0 else -14),
                            textcoords='offset points', fontsize=8, color=INK2)
        style(ax, title, 'L1 [µs](10 ステップ・舵とスロットルの平均)', 'エポック')
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK2, loc='upper right')
    fig.suptitle('JetFormer の学習曲線(jetformer_v0・ImageNet から・30 エポック・間引き 3)。丸 = 最良のエポック',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_curves.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_offline(off, rows, plt):
    if not off:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), facecolor=SURFACE, sharex=True)
    x = np.arange(1, 11)
    seeds = {r['key']: r for r in rows}
    for i, (set_key, set_title) in enumerate((('RL_val', 'RL のデータの val'), ('teacher_val', '教師のデータの val'))):
        entry = off['sets'][set_key]
        for j, (field, what) in enumerate((('steer_per_step_us', '舵'), ('throttle_per_step_us', 'スロットル'))):
            ax = axes[i, j]
            for key, m in entry['models'].items():
                r = seeds.get(key)
                if r is None:
                    continue
                ax.plot(x, m[field], SEED_STYLE[r['seed']], color=ARM_COLOR[r['arm']], lw=2, marker='o', ms=4,
                        label='%s 種 %d' % (ARM_LABEL[r['arm']], r['seed']))
            style(ax, '%s・%s(%d フレーム)' % (set_title, what, entry['frames']), 'L1 [µs]', '何フレーム先か(1 = シミュレーターの走行で使う値)')
            ax.set_xticks(x)
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK2, loc='center left')
    fig.suptitle('オフライン L1(全フレーム・間引きなし・FP32)。データごと・ステップごとの誤差', color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_offline_l1.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def fig_phase(off, rows, plt):
    """各データ自身の val の L1 を、フレーム番号を 3 で割った余りごとに(学習と学習中の検証は余り 0 だけを使う)"""
    if not off:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), facecolor=SURFACE, sharey=False)
    for ax, arm, set_key, title in ((axes[0], 'S-R', 'RL_val', 'S-R を RL のデータの val で'),
                                    (axes[1], 'S-T', 'teacher_val', 'S-T を教師のデータの val で')):
        x = np.arange(3)
        w = 0.36
        for j, r in enumerate([r for r in rows if r['arm'] == arm]):
            v = off['sets'][set_key]['models'][r['key']]['both_mean10_by_frame_mod3_us']
            bars = ax.bar(x + (j - 0.5) * w, v, w - 0.04, color=ARM_COLOR[arm], alpha=1.0 if r['seed'] == 0 else 0.55,
                          edgecolor=SURFACE, linewidth=2, label='種 %d' % r['seed'])
            for bx, bv in zip(x + (j - 0.5) * w, v):
                ax.text(bx, bv, '%.1f' % bv, ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(['余り 0\n(学習・学習中の検証に使った位相)', '余り 1\n(IMU が 3.3 ms ずれる)', '余り 2\n(IMU が 6.7 ms ずれる)'])
        style(ax, title, 'L1 [µs](10 ステップ・舵とスロットルの平均)')
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc='upper left')
        ax.set_ylim(0, max(ax.get_ylim()[1], 1) * 1.12)
    fig.suptitle('フレーム番号を 3 で割った余りごとの検証 L1(全フレーム・FP32)。間引き 3 の学習は余り 0 の位相しか見ていない',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_frame_phase.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def tables(rows, off):
    out = ['### 学習(各データ自身の val)', '',
           '| 腕・種 | 学習結果 | 学習 / 検証のサンプル | 最良の検証 L1 [µs] | 最良のエポック | 30 エポック目の検証 / 学習 L1 [µs] | 1 エポック [秒](中央値) | 30 エポックの計算 [分] | 試行(上限 / 見張り / 始め直し) | workers |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        k = r['attempt_kinds']
        out.append('| %s 種 %d | `%s` | %d / %d | **%.1f** | %d | %.1f / %.1f | %.0f | %.0f | %d(%d / %d / %d) | %d |' % (
            r['arm'], r['seed'], r['run'], r['train_samples'], r['val_samples'], r['best_val_us'], r['best_epoch'],
            r['last_val_us'], r['last_train_us'], r['epoch_s_median'], r['epoch_seconds_sum'] / 60, r['attempts'],
            k['mem_cap'], k['watchdog'], k['restarted_before_start'], r['workers']))
    if off:
        out += ['', '### オフライン L1(全フレーム・FP32。1 ステップ目 = 走行で使う値、10 = 10 ステップの平均)', '',
                '| 腕・種 | RL の val 舵 1 / 10 | RL の val スロットル 1 / 10 | RL の val 平均(余り 0 / 1 / 2) | 教師の val 舵 1 / 10 | 教師の val スロットル 1 / 10 | 教師の val 平均(余り 0 / 1 / 2) |',
                '|---|---|---|---|---|---|---|']
        keys = [r['key'] for r in rows]
        for k in keys:
            a = off['sets']['RL_val']['models'].get(k)
            b = off['sets']['teacher_val']['models'].get(k)
            if not a or not b:
                continue
            arm, seed = k.split('/s')
            ra, rb = a.get('both_mean10_by_frame_mod3_us') or [0, 0, 0], b.get('both_mean10_by_frame_mod3_us') or [0, 0, 0]
            out.append('| %s 種 %s | %.1f / %.1f | %.1f / %.1f | %.1f(%.1f / %.1f / %.1f) | %.1f / %.1f | %.1f / %.1f | %.1f(%.1f / %.1f / %.1f) |' % (
                arm, seed, a['steer_step1_us'], a['steer_mean10_us'], a['throttle_step1_us'], a['throttle_mean10_us'],
                a['both_mean10_us'], ra[0], ra[1], ra[2],
                b['steer_step1_us'], b['steer_mean10_us'], b['throttle_step1_us'], b['throttle_mean10_us'],
                b['both_mean10_us'], rb[0], rb[1], rb[2]))
    out += ['', '### エンジン(FP16)とシミュレーター用の一式', '',
            '| 腕・種 | エンジン: FP32 との差 最大 / 平均 [µs] | run_numpy 中央値 / p99 [ms] | 確認 | 一式: ONNX と PyTorch の差 最大 [µs] | ブリッジの読み込み | ZIP [MB] |',
            '|---|---|---|---|---|---|---|']
    for r in rows:
        e, b = r['engine'], r['bundle']
        out.append('| %s 種 %d | %s / %s | %s / %s | %s | %s | %s | %s |' % (
            r['arm'], r['seed'], e.get('actions_max_us'), e.get('actions_mean_us'), e.get('run_numpy_median'), e.get('run_numpy_p99'),
            '合格' if e.get('all_gates') else '不合格', b.get('onnx_vs_torch_max_us'), '合格' if b.get('ok') else '不合格',
            '%.1f' % (b['zip_bytes'] / 2 ** 20) if b.get('zip_bytes') else '—'))
    return '\n'.join(out) + '\n'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    state, rows = collect()
    off = load(TRAIN / 'offline_l1.json')
    if rows:
        fig_curves(rows, plt)
    has_off = fig_offline(off, rows, plt)
    if has_off:
        fig_phase(off, rows, plt)
    summary = {'rows': rows, 'offline_l1': off, 'state_updated': state.get('updated'), 'all_done': state.get('all_done'),
               'figures': ['fig/train_curves.png'] + (['fig/train_offline_l1.png', 'fig/train_frame_phase.png'] if has_off else [])}
    (TRAIN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    (TRAIN / 'tables.md').write_text(tables(rows, off), encoding='utf-8')
    runs = {'schema': 'sim-rl-bestlap-train-runs/v1', 'note': 'protocol.json の run 名 simrl-bestlap-<腕>-s<種> の代わりに、この名前を使った',
            'runs': {r['key']: {'arm': r['arm'], 'seed': r['seed'], 'run': r['run'], 'run_dir': str(RUNS / r['run']),
                                'dataset': r['dataset'], 'best_val_us': r['best_val_us'], 'best_epoch': r['best_epoch'],
                                'engine': str(RUNS / r['run'] / 'jetformer_fp16.engine'),
                                'bundle_dir': str(REPORT / 'bundles' / r['run']), 'bundle_zip': r['bundle']['zip'],
                                'bundle_zip_sha256': r['bundle']['zip_sha256'], 'onnx_sha256': r['bundle']['onnx_sha256'],
                                'bundle_model_config_sha256': r['bundle']['model_config_sha256'],
                                'bundle_io_spec_sha256': r['bundle']['io_spec_sha256'], 'bundle_ok': r['bundle']['ok'],
                                'mode_id': 8, 'num_modes': 9} for r in rows}}
    (TRAIN / 'runs.json').write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('key', 'best_val_us', 'best_epoch', 'last_val_us')} for r in rows], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
