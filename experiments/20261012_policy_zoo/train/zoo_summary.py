#!/usr/bin/env python3
"""方策の比較(段階4)の学習のまとめ(CPU だけ。何度でもやり直せる)

    cd ~/jetformer && python /home/jetson/jetformer_reports/policy_zoo_20261012/train/zoo_summary.py

読むもの: plan.json・state.json(zoo_train.py)、各学習結果の result.json・model_config.json・run.json、
engine_check/<名前>.json(engine_load_check.py)、モデル B の result.json。
書くもの: summary.json(このフォルダー)、図 ../fig/train_*.png(日本語・大きい文字)。

ここの「検証 L1」は train.py が best.pth を選ぶのに使った検証(clean の val 11 シーン・stride 3・1,559 サンプル)の値で、
spec §5.2 の主指標(stride 1・4,670 フレーム、評価の段で計算する)ではない。
"""

import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path('/home/jetson/jetformer')
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
FIG = REPORT / 'fig'
RUNS = REPO / 'data' / 'runs'
MODEL_B = 'clean_driving_20261011-nocmd-s3-20261011-024626'
ORDER = ['jetformer_v0', 'act', 'tcp', 'msfusion', 'dinov2']
TITLE = {'jetformer_v0': 'JetFormer v0', 'act': 'ACT 型', 'tcp': 'TCP 型', 'msfusion': '多段融合', 'dinov2': 'DINOv2'}
# 分類の色(固定の順。方策 = 色、種 = 線の形)。モデル B は灰色の基準線
COLOR = {'jetformer_v0': '#2a78d6', 'act': '#eb6834', 'tcp': '#1baf7a', 'msfusion': '#eda100', 'dinov2': '#e87ba4'}
TEXT, MUTED, GRID, REF = '#0b0b0b', '#52514e', '#d9d8d4', '#8a8984'
sys.path.insert(0, str(REPO))


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


def run_row(key, name, st):
    d = RUNS / name
    res = load(d / 'result.json') or {}
    cfg = load(d / 'model_config.json') or {}
    hist = res.get('history') or []
    chk = load(HERE / 'engine_check' / ('%s.json' % name))
    row = {'key': key, 'run': name, 'policy': res.get('policy'), 'seed': res.get('seed'), 'state': res.get('state'),
           'epochs_done': res.get('epochs_done'), 'best_val_us': res.get('best_val_us'), 'best_epoch': res.get('best_epoch'),
           'last_val_us': hist[-1]['val_us'] if hist else None, 'min_val_us': min((h['val_us'] for h in hist), default=None),
           'last_train_us': hist[-1]['train_us'] if hist else None,
           'best_val_per_mode_us': res.get('best_val_per_mode_us'), 'params_m': res.get('params_m'),
           'peak_mem_mb': res.get('peak_mem_mb'),
           'epoch_s_median': round(statistics.median(h['seconds'] for h in hist), 1) if hist else None,
           'epoch_s_first': hist[0]['seconds'] if hist else None,
           'train_epochs_s_sum': round(sum(h['seconds'] for h in hist), 1) if hist else None,
           'files': sorted(f for f in ('best.pth', 'last.pth', 'model_config.json', 'result.json', 'run.json', 'jetformer.onnx',
                                       'jetformer_fp16.engine', 'train.log', 'engine.log') if (d / f).exists()),
           'onnx_mb': round((d / 'jetformer.onnx').stat().st_size / 2 ** 20, 1) if (d / 'jetformer.onnx').exists() else None,
           'engine_mb': (round((d / 'jetformer_fp16.engine').stat().st_size / 2 ** 20, 1)
                         if (d / 'jetformer_fp16.engine').exists() else None),
           'model_config_policy': cfg.get('policy'), 'policy_cfg': cfg.get('policy_cfg'),
           'engine_precision': (cfg.get('engine') or {}).get('precision'),
           'postprocess_kind': (cfg.get('postprocess') or {}).get('kind') if cfg.get('postprocess') else None}
    if st:
        row['attempts'] = st.get('attempts')
        row['resumed'] = [a['resume_from_epoch'] for a in st.get('attempts', []) if a.get('resume_from_epoch')]
        row['train_wall_s'] = round(sum(a.get('wall_s') or 0 for a in st.get('attempts', [])), 1)
        row['engine_wall_s'] = (st.get('engine') or {}).get('wall_s')
        row['code'] = sorted({a.get('code') for a in st.get('attempts', [])} | ({st['engine'].get('code')}
                                                                                if st.get('engine') else set()))
    if hist and 'train_parts' in hist[-1]:
        row['last_train_parts'] = hist[-1]['train_parts']
        row['last_train_loss'] = hist[-1].get('train_loss')
    if chk:
        row['engine_check'] = {'all_gates': chk.get('all_gates'), 'drivable': chk['model_info'].get('drivable'),
                               'check_io': chk['check_io']['ok'], 'accuracy': chk.get('accuracy'),
                               'latency_ms': chk.get('latency_ms'), 'postprocess_kind': chk['model_info'].get('postprocess_kind')}
    try:
        from jetformer import drive
        info = drive.model_info(name)
        row['drive_list'] = {'listed': info is not None, 'drivable': (info or {}).get('drivable'),
                             'policy_title': (info or {}).get('policy_title')}
    except Exception as e:      # noqa: BLE001
        row['drive_list'] = {'error': repr(e)}
    return row


def summarize():
    plan = load(HERE / 'plan.json') or {}
    state = load(HERE / 'state.json') or {}
    rows = []
    for key, name in (plan.get('runs') or {}).items():
        if (RUNS / name / 'result.json').exists():
            rows.append(run_row(key, name, (state.get('runs') or {}).get(key)))
    by_pol = {}
    for p in ORDER:
        vals = [r['best_val_us'] for r in rows if r['policy'] == p and r['state'] == 'done' and r['best_val_us'] is not None]
        lasts = [r['last_val_us'] for r in rows if r['policy'] == p and r['state'] == 'done']
        if vals:
            by_pol[p] = {'n': len(vals), 'best_val_mean_us': round(sum(vals) / len(vals), 2), 'best_val_min_us': min(vals),
                         'best_val_max_us': max(vals), 'last_val_mean_us': round(sum(lasts) / len(lasts), 2)}
    b = load(RUNS / MODEL_B / 'result.json') or {}
    summary = {'plan': {k: plan.get(k) for k in ('epochs', 'projected_min', 'budget_min', 'limit_min', 'cut', 'per_run_min')},
               'probe': state.get('probe'), 'runs': rows, 'by_policy': by_pol,
               'model_b': {'run': MODEL_B, 'best_val_us': b.get('best_val_us'), 'best_epoch': b.get('best_epoch'),
                           'last_val_us': (b.get('history') or [{}])[-1].get('val_us')},
               'codes': sorted({c for r in rows for c in (r.get('code') or []) if c}),
               'note': '検証 L1 = train.py の検証(clean val 11 シーン・stride 3・1,559 サンプル)。spec §5.2 の主指標ではない'}
    write_json(HERE / 'summary.json', summary)
    return summary


# ---------------------------------------------------------------- 図
def _setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'legend.fontsize': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': TEXT,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': TEXT, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                         'savefig.facecolor': '#fcfcfb', 'lines.linewidth': 2.0})
    return plt


def _save(plt, fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('図:', FIG / name)


def _hist(name):
    return (load(RUNS / name / 'result.json') or {}).get('history') or []


def figures(summary):
    plt = _setup()
    plan = load(HERE / 'plan.json') or {}
    runs = plan.get('runs') or {}
    b_hist = _hist(MODEL_B)
    # 1. 方策ごとの小さな図: 検証 L1(太)と学習 L1(細)、種 0 = 実線・種 1 = 破線、モデル B の検証 = 灰色
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.6), sharey=True)
    for ax, p in zip(axes, ORDER):
        for seed, ls in ((0, '-'), (1, '--')):
            h = _hist(runs.get('%s/s%d' % (p, seed), ''))
            if not h:
                continue
            ep = [x['epoch'] for x in h]
            ax.plot(ep, [x['val_us'] for x in h], ls, color=COLOR[p], lw=2.4, label='検証 種%d' % seed)
            ax.plot(ep, [x['train_us'] for x in h], ls, color=COLOR[p], lw=1.2, alpha=0.55, label='学習 種%d' % seed)
            best = min(h, key=lambda x: x['val_us'])
            ax.plot([best['epoch']], [best['val_us']], 'o', ms=8, color=COLOR[p], mec='#fcfcfb', mew=2)
        if b_hist:
            ax.plot([x['epoch'] for x in b_hist], [x['val_us'] for x in b_hist], ':', color=REF, lw=2, label='モデル B 検証')
        ax.set_title(TITLE[p])
        ax.set_xlabel('エポック')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_ylim(10, 90)
        over = [(seed, x['epoch'], x['val_us']) for seed in (0, 1) for x in _hist(runs.get('%s/s%d' % (p, seed), ''))
                if x['val_us'] > 90]
        if over:                                        # 縦の範囲の外の値(切れている)を文字で残す
            lines = ['種%d: ' % sd + '・'.join('e%d %.0f' % (e, v) for s2, e, v in over if s2 == sd)
                     for sd in (0, 1) if any(o[0] == sd for o in over)]
            ax.text(0.97, 0.03, '上に切れた検証 [µs]\n' + '\n'.join(lines), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=11, color=MUTED)
    axes[0].set_ylabel('L1 [µs](検証 = 太線)')
    axes[-1].legend(loc='upper right', frameon=False)
    fig.suptitle('学習曲線(clean_driving_20261011・stride 3・今の操作値なし・2.5G 除外)  ● = best.pth のエポック', y=1.02)
    _save(plt, fig, 'train_curves_by_policy.png')

    # 2. 検証 L1 の重ね描き(種の平均、範囲を帯で)・縦を拡大
    fig, ax = plt.subplots(figsize=(13, 7))
    ends = []
    for p in ORDER:
        hs = [_hist(runs.get('%s/s%d' % (p, s), '')) for s in (0, 1)]
        hs = [h for h in hs if h]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        ep = [hs[0][i]['epoch'] for i in range(n)]
        vals = [[h[i]['val_us'] for h in hs] for i in range(n)]
        mean = [sum(v) / len(v) for v in vals]
        ax.fill_between(ep, [min(v) for v in vals], [max(v) for v in vals], color=COLOR[p], alpha=0.15, lw=0)
        ax.plot(ep, mean, color=COLOR[p], lw=2.2, label='%s(種 %d 本の平均・帯 = 範囲)' % (TITLE[p], len(hs)))
        ends.append((mean[-1], p, ep[-1]))
    if b_hist:
        ax.plot([x['epoch'] for x in b_hist], [x['val_us'] for x in b_hist], ':', color=REF, lw=2.2,
                label='モデル B(旧コード・種なし・26 で再開)')
    ax.set_ylim(44, 70)
    gap = 0.9                                           # 右端の直接の名前: 下から順に 0.9 µs 以上あけて重ならないように
    prev = None
    for y, p, x in sorted(ends):
        ty = y if prev is None else max(y, prev + gap)
        prev = ty
        ax.annotate('%s %.1f' % (TITLE[p], y), (x, y), xytext=(x + 0.8, ty), color=COLOR[p], fontsize=13, va='center',
                    arrowprops=dict(arrowstyle='-', color=COLOR[p], lw=0.8))
    ax.set_xlim(0, 36)
    ax.set_xlabel('エポック')
    ax.set_ylabel('検証 L1 [µs](stride 3・1,559 サンプル)')
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('検証 L1 の推移(方策ごと、種の平均と範囲)')
    _save(plt, fig, 'train_val_overlay.png')

    # 3. best.pth の検証 L1: 種ごとの点と範囲(モデル B は灰色の線)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, p in enumerate(ORDER):
        rs = [r for r in summary['runs'] if r['policy'] == p and r['best_val_us'] is not None]
        if not rs:
            continue
        ys = [r['best_val_us'] for r in rs]
        ax.plot([min(ys), max(ys)], [i, i], color=COLOR[p], lw=2)
        for r in rs:
            ax.plot([r['best_val_us']], [i], 'o' if r['seed'] == 0 else 's', ms=10, color=COLOR[p], mec='#fcfcfb', mew=2)
            ax.annotate('種%d %.1f' % (r['seed'], r['best_val_us']), (r['best_val_us'], i), xytext=(0, 12 if r['seed'] == 0 else -22),
                        textcoords='offset points', ha='center', fontsize=12, color=TEXT)
    if summary['model_b'].get('best_val_us'):
        ax.axvline(summary['model_b']['best_val_us'], color=REF, ls=':', lw=2)
        ax.annotate('モデル B %.1f' % summary['model_b']['best_val_us'], (summary['model_b']['best_val_us'], -0.45),
                    xytext=(6, 0), textcoords='offset points', color=MUTED, fontsize=12, va='center')
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([TITLE[p] for p in ORDER])
    ax.set_ylim(len(ORDER) - 0.5, -0.8)             # 上下を逆に(上が v0)。上の余白にモデル B の名前
    ax.set_xlabel('best.pth の検証 L1 [µs](stride 3。主指標ではない)')
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_title('best.pth の検証 L1(● 種 0・■ 種 1)')
    _save(plt, fig, 'train_best_val.png')

    # 4. 方策の損失の内訳(tcp・act。参考)
    parts_runs = [(k, n) for k, n in runs.items() if any('train_parts' in x for x in _hist(n))]
    if parts_runs:
        keys = sorted({kk for _, n in parts_runs for x in _hist(n) for kk in (x.get('train_parts') or {})})
        fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5), squeeze=False)
        for ax, kk in zip(axes[0], keys):
            for k, n in sorted(parts_runs):
                h = [x for x in _hist(n) if kk in (x.get('train_parts') or {})]
                if not h:
                    continue
                p = k.split('/')[0]
                ax.plot([x['epoch'] for x in h], [x['train_parts'][kk] for x in h], '-' if k.endswith('s0') else '--',
                        color=COLOR.get(p, MUTED), label='%s %s' % (TITLE.get(p, p), k.split('/')[1]))
            ax.set_title(kk)
            ax.set_xlabel('エポック')
            ax.grid(axis='y', color=GRID, lw=0.8)
        axes[0][0].legend(frameon=False)
        fig.suptitle('方策の学習の損失の内訳(学習の平均。参考)', y=1.03)
        _save(plt, fig, 'train_loss_parts.png')


def main():
    s = summarize()
    figures(s)
    print(json.dumps(s['by_policy'], ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
