#!/usr/bin/env python3
"""学習の段の図と表(sim_dart_20261014)

    /home/jetson/miniconda/envs/jetracer/bin/python -I ~/jetformer/experiments/20261014_sim_dart_margin/train/train_report.py

ベストラップの train_report.py の写しを、この作業の腕(A0 = S-R 種 0 の参照・A1・A2)に合わせて直したもの。
  fig/train_curves.png       エポックごとの検証 L1(各腕の学習データ自身の val)と学習 L1。A0 = 灰・A1 = 青・A2 = 緑。丸 = 最良のエポック
  fig/train_offline_l1.png   オフライン L1(offline_l1.json があれば): 3 つの val と、D1 の val の「付け替えで正解が変わったフレーム」
  train/summary.json         図と train.md の表の元の数字
  train/tables.md            train.md に貼る表
  train/runs.json            次の段(閉ループ評価)が使う対応表: 腕・種 → 学習結果・一式のフォルダー・ZIP・sha256
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPORT = Path('/home/jetson/jetformer_reports/sim_dart_20261014')
TRAIN = REPORT / 'train'
FIG = REPORT / 'fig'
RUNS = Path('/home/jetson/jetformer/data/runs')
BESTLAP_BUNDLES = Path('/home/jetson/jetformer_reports/sim_rl_bestlap_20261013/bundles')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ARM_COLOR = {'A0': '#8a8984', 'A1': '#2a78d6', 'A2': '#1baf7a'}
ARM_LABEL = {'A0': 'A0 = S-R(最速の線のきれいなデータ 20 分)', 'A1': 'A1 = A0 + DART 10 分(ずれたときの戻し方)',
             'A2': 'A2 = 余裕を持たせた線 10 分'}
ARM_DATA = {'A0': 'A0 の val 12 シーン', 'A1': 'A0 + D1 の val 18 シーン', 'A2': 'D2 の val 6 シーン'}
A0_REF = ('A0/s0', 'simrl_bestlap_20261013-SR-s0')


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
    out = {'done': 0, 'mem_cap': 0, 'watchdog': 0, 'no_exit_recorded': 0, 'other': 0}
    for a in rs.get('attempts', []):
        if a.get('exit') is None:
            out['no_exit_recorded'] += 1
        elif a.get('watchdog_killed'):
            out['watchdog'] += 1
        elif a.get('exit') == 0:
            out['done'] += 1
        elif a.get('exit') in (-9, 137):
            out['mem_cap'] += 1
        else:
            out['other'] += 1
    return out


def row_of(key, run_name, rs, bundle_dir):
    run = RUNS / run_name
    res = load(run / 'result.json')
    if not res or res.get('state') != 'done':
        return None
    bc = load(bundle_dir / 'bundle_check.json', {})
    ex = load(bundle_dir / 'export.json', {})
    hl = load(bundle_dir / 'http_load_check.json', {})
    hist = res['history']
    arm, seed = key.split('/s')
    return {
        'key': key, 'arm': arm, 'seed': int(seed), 'run': run_name, 'dataset': res['dataset'],
        'train_samples': res['train_samples'], 'val_samples': res['val_samples'],
        'best_val_us': res['best_val_us'], 'best_epoch': res['best_epoch'],
        'last_val_us': hist[-1]['val_us'], 'last_train_us': hist[-1]['train_us'],
        'val_us_epoch': [h['val_us'] for h in hist], 'train_us_epoch': [h['train_us'] for h in hist],
        'epoch_s_median': float(np.median([h['seconds'] for h in hist])),
        'epoch_seconds_sum': round(sum(h['seconds'] for h in hist), 1),
        'attempts': len((rs or {}).get('attempts', [])), 'attempt_kinds': attempt_kinds(rs or {}),
        'workers': (rs or {}).get('workers'), 'peak_mem_mb': res.get('peak_mem_mb'),
        'resumes': sum(1 for a in (rs or {}).get('attempts', []) if a.get('resume_from_epoch')),
        'steer_center_us_trained': res.get('steer_center_us'), 'throttle_neutral_us_trained': res.get('throttle_neutral_us'),
        'reference_only': rs is None,
        'bundle': {'dir': str(bundle_dir), 'ok': bc.get('ok'), 'zip': ex.get('zip'), 'zip_sha256': ex.get('zip_sha256'),
                   'zip_bytes': ex.get('zip_bytes'), 'onnx_vs_torch_max_us': bc.get('onnx_vs_torch_max_us'),
                   'gates': bc.get('gates'), 'defaults': bc.get('defaults'), 'mode_input': bc.get('mode_input'),
                   'onnx_sha256': (ex.get('files') or {}).get('jetformer.onnx', {}).get('sha256'),
                   'model_config_sha256': (ex.get('files') or {}).get('model_config.json', {}).get('sha256'),
                   'io_spec_sha256': (ex.get('files') or {}).get('io_spec.json', {}).get('sha256'),
                   'http_load_ok': hl.get('ok'), 'http_load_s': ((hl.get('results') or [{}])[0]).get('load_s')},
    }


def collect():
    state = load(TRAIN / 'state.json', {})
    rows = []
    r0 = row_of(A0_REF[0], A0_REF[1], None, BESTLAP_BUNDLES / A0_REF[1])
    if r0:
        rows.append(r0)
    for key in state.get('order', []):
        rs = state.get('runs', {}).get(key)
        if not rs:
            continue
        r = row_of(key, rs['run'], rs, REPORT / 'bundles' / rs['run'])
        if r:
            rows.append(r)
    return state, rows


def fig_curves(rows, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), facecolor=SURFACE)
    for ax, field, title in ((axes[0], 'val_us_epoch', '検証 L1(各腕の学習データ自身の val。腕ごとに val が違う)'),
                             (axes[1], 'train_us_epoch', '学習 L1(色の揺らぎあり)')):
        for r in rows:
            y = r[field]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, '--' if r['reference_only'] else '-', color=ARM_COLOR[r['arm']], lw=2,
                    label='%s(%s)' % (ARM_LABEL[r['arm']], ARM_DATA[r['arm']]))
            if field == 'val_us_epoch':
                b = r['best_epoch']
                ax.plot([b], [y[b - 1]], 'o', ms=8, color=ARM_COLOR[r['arm']], mec=SURFACE, mew=2)
                off_xy = {'A0': (-10, 8), 'A1': (-8, 9), 'A2': (-8, -15)}[r['arm']]
                ax.annotate('%.1f' % y[b - 1], (b, y[b - 1]), xytext=off_xy, textcoords='offset points',
                            fontsize=8.5, color=ARM_COLOR[r['arm']], ha='right')
        style(ax, title, 'L1 [µs](10 ステップ・舵とスロットルの平均)', 'エポック')
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7.8, frameon=False, labelcolor=INK2, loc='upper right')
    fig.suptitle('JetFormer の学習曲線(jetformer_v0・ImageNet から・30 エポック・間引き 3)。破線 = A0(10/13 に学習済み)・丸 = 最良のエポック',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_curves.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_offline(off, rows, plt):
    if not off:
        return False
    keys = [r['key'] for r in rows if r['key'] in off['models']]
    panels = [('A0_val', 'all', '最速の線(A0)の val'), ('D2_val', 'all', '余裕の線(D2)の val'),
              ('D1_val', 'unchanged', 'DART(D1)の val: 正解が変わらないフレーム'),
              ('D1_val', 'changed', 'DART(D1)の val: 付け替えで正解が変わったフレーム\n(線から外れた状態・乱しの最中)')]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.4), facecolor=SURFACE, sharey=True)
    w = 0.8 / max(len(keys), 1)
    for ax, (sk, sub, title) in zip(axes, panels):
        entry = off['sets'].get(sk)
        if not entry:
            ax.set_visible(False)
            continue
        for j, k in enumerate(keys):
            m = (entry['models'].get(k) or {}).get(sub)
            if not m:
                continue
            arm = k.split('/')[0]
            vals = [m['steer_step1_us'], m['throttle_step1_us']]
            xs = np.arange(2) + (j - (len(keys) - 1) / 2) * w
            ax.bar(xs, vals, w * 0.92, color=ARM_COLOR[arm], edgecolor=SURFACE, lw=1.5, label=ARM_LABEL[arm].split(' = ')[0])
            for bx, bv in zip(xs, vals):
                ax.text(bx, bv, '%.0f' % bv, ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(['舵', 'スロットル'])
        n = ((entry['models'].get(keys[0]) or {}).get(sub) or {}).get('frames') if keys else None
        style(ax, '%s\n(%s フレーム)' % (title, n), 'L1 [µs](1 ステップ目 = 走行で使う値)' if ax is axes[0] else '')
    axes[0].legend(fontsize=8.5, frameon=False, labelcolor=INK2, loc='upper left')
    fig.suptitle('オフライン L1(val の全フレーム・間引きなし・FP32。正解との差の平均、小さいほど先生に近い)。参考で、合否には使わない',
                 color=INK, fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(FIG / 'train_offline_l1.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def tables(rows, off):
    out = ['### 学習(各腕の学習データ自身の val)', '',
           '| 腕・種 | 学習結果 | 学習 / 検証のサンプル | 最良の検証 L1 [µs] | 最良のエポック | 30 エポック目の検証 / 学習 L1 [µs] | 1 エポック [秒](中央値) | 30 エポックの計算 [分] | 試行(上限 / 見張り / 終わり不明) | workers |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        k = r['attempt_kinds']
        out.append('| %s 種 %d%s | `%s` | %d / %d | **%.1f** | %d | %.1f / %.1f | %.0f | %.0f | %s | %s |' % (
            r['arm'], r['seed'], '(参照・10/13)' if r['reference_only'] else '', r['run'], r['train_samples'], r['val_samples'],
            r['best_val_us'], r['best_epoch'], r['last_val_us'], r['last_train_us'], r['epoch_s_median'],
            r['epoch_seconds_sum'] / 60,
            '—' if r['reference_only'] else '%d(%d / %d / %d)' % (r['attempts'], k['mem_cap'], k['watchdog'], k['no_exit_recorded']),
            '—' if r['reference_only'] else r['workers']))
    if off:
        out += ['', '### オフライン L1(val の全フレーム・FP32。1 ステップ目 = 走行で使う値 / 10 ステップの平均、単位 µs)', '',
                '| 腕・種 | A0 の val 舵 1 / 10 | A0 の val スロットル 1 / 10 | D2 の val 舵 1 / 10 | D2 の val スロットル 1 / 10 | D1 の val(変わらない)舵 1 | D1 の val(変わった)舵 1 / スロットル 1 | 変わったフレーム: 付け替える前の正解との舵 1 | 変わったフレーム: 先生の正解に近い割合 |',
                '|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            k = r['key']
            a = (off['sets'].get('A0_val') or {}).get('models', {}).get(k)
            b = (off['sets'].get('D2_val') or {}).get('models', {}).get(k)
            c = (off['sets'].get('D1_val') or {}).get('models', {}).get(k)
            if not a or not b or not c:
                continue
            out.append('| %s 種 %d | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f / %.1f | %.1f | %.1f / %.1f | %.1f | %.0f%% |' % (
                r['arm'], r['seed'], a['all']['steer_step1_us'], a['all']['steer_mean10_us'],
                a['all']['throttle_step1_us'], a['all']['throttle_mean10_us'],
                b['all']['steer_step1_us'], b['all']['steer_mean10_us'], b['all']['throttle_step1_us'], b['all']['throttle_mean10_us'],
                c['unchanged']['steer_step1_us'], c['changed']['steer_step1_us'], c['changed']['throttle_step1_us'],
                c['changed_vs_old_label']['steer_step1_us'], 100 * c['changed_steer_step1_closer_to_teacher_frac']))
        d1 = off['sets'].get('D1_val') or {}
        if d1:
            out += ['', 'D1 の val: %d フレーム中、付け替えで 1 ステップ目の正解が 1 µs より大きく変わったのは %d フレーム(変わった大きさの平均 舵 %.1f µs)。' % (
                d1['frames'], d1.get('changed_frames', 0), d1.get('label_change_step1_steer_us_mean_on_changed', float('nan')))]
    out += ['', '### シミュレーター用の一式', '',
            '| 腕・種 | 一式の ZIP | ONNX と PyTorch の差 最大 [µs] | ブリッジの読み込み(関数) | ブリッジの読み込み(HTTP) | ZIP [MB] | ZIP の sha256 |',
            '|---|---|---|---|---|---|---|']
    for r in rows:
        b = r['bundle']
        out.append('| %s 種 %d | `%s` | %s | %s | %s | %s | `%s` |' % (
            r['arm'], r['seed'], b.get('zip'), b.get('onnx_vs_torch_max_us'), '合格' if b.get('ok') else '不合格',
            {True: '合格', False: '不合格'}.get(b.get('http_load_ok'), '—'),
            '%.1f' % (b['zip_bytes'] / 2 ** 20) if b.get('zip_bytes') else '—', (b.get('zip_sha256') or '')[:16]))
    return '\n'.join(out) + '\n'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(exist_ok=True)
    state, rows = collect()
    off = load(TRAIN / 'offline_l1.json')
    figs = []
    if rows:
        fig_curves(rows, plt)
        figs.append('fig/train_curves.png')
    if fig_offline(off, rows, plt):
        figs.append('fig/train_offline_l1.png')
    summary = {'rows': rows, 'offline_l1': off, 'state_updated': state.get('updated'), 'all_done': state.get('all_done'),
               'figures': figs}
    (TRAIN / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    (TRAIN / 'tables.md').write_text(tables(rows, off), encoding='utf-8')
    runs = {'schema': 'sim-dart-train-runs/v1',
            'note': '閉ループ評価の段が使う対応表。A0/s0 はベストラップの S-R 種 0(一式もベストラップのもの。作り直さない)',
            'runs': {r['key']: {'arm': r['arm'], 'seed': r['seed'], 'run': r['run'], 'run_dir': str(RUNS / r['run']),
                                'dataset': r['dataset'], 'best_val_us': r['best_val_us'], 'best_epoch': r['best_epoch'],
                                'bundle_dir': r['bundle']['dir'], 'bundle_zip': r['bundle']['zip'],
                                'bundle_zip_sha256': r['bundle']['zip_sha256'], 'onnx_sha256': r['bundle']['onnx_sha256'],
                                'bundle_model_config_sha256': r['bundle']['model_config_sha256'],
                                'bundle_io_spec_sha256': r['bundle']['io_spec_sha256'], 'bundle_ok': r['bundle']['ok'],
                                'http_load_ok': r['bundle']['http_load_ok'], 'reference_only': r['reference_only'],
                                'mode_id': 8, 'num_modes': 9} for r in rows}}
    (TRAIN / 'runs.json').write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('key', 'best_val_us', 'best_epoch', 'last_val_us')} for r in rows], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
