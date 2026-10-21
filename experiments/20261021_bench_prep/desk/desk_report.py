#!/usr/bin/env python3
"""desk_bench.json → 図(fig/)と desk_bench.md(CPU・numpy・matplotlib だけ。torch を読まない)

    /home/jetson/bin/jf-heavy --lock none --mem 1500M -- /home/jetson/miniconda/envs/jetracer/bin/python -I \\
        /home/jetson/jetformer/experiments/20261021_bench_prep/desk/desk_report.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_lib as D  # noqa: E402

OUT = Path('/home/jetson/jetformer_reports/bench_prep_20261021/desk')
FIG = OUT / 'fig'
NOTES = HERE / 'desk_notes.md'          # 手で書いた「見えたこと」(あれば md の 3 章に入れる)

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#d6d5d0'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#8a8984'


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def name(r):
    return '%s %s' % (r['id'], r['label'])


def hbar(ax, rows, key, title, unit, fmt='%.1f', lines=(), err=None, xlim=None):
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        v = r.get(key) if not callable(key) else key(r)
        col = ORANGE if r['id'] == 'B' else BLUE
        if v is None:
            ax.text(0.01, y, '値なし', transform=ax.get_yaxis_transform(), va='center', color=INK2, fontsize=13)
            continue
        ax.barh(y, v, height=0.62, color=col, edgecolor=SURFACE, linewidth=2)
        xt = v
        if err is not None:
            e = err(r)
            if e and e[0] is not None:
                ax.plot([e[0], e[1]], [y, y], color=INK, lw=1.5)
                ax.plot([e[0], e[0]], [y - 0.15, y + 0.15], color=INK, lw=1.5)
                ax.plot([e[1], e[1]], [y - 0.15, y + 0.15], color=INK, lw=1.5)
                xt = max(v, e[1])
        ax.text(xt, y, '  ' + fmt % v, va='center', ha='left', color=INK, fontsize=13)
    for x, lab, ls in lines:
        ax.axvline(x, color=GRAY, lw=1.5, ls=ls)
        ax.text(x, -0.75, lab, ha='center', va='top', color=INK2, fontsize=12)
    ax.set_ylim(-1.3, len(rows) - 0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([name(r) for r in rows])
    ax.set_title(title, loc='left', color=INK)
    ax.set_xlabel(unit)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlim:
        ax.set_xlim(*xlim)


def figures(doc):
    plt = setup()
    FIG.mkdir(parents=True, exist_ok=True)
    rows = doc['candidates']
    b = next(r for r in rows if r['id'] == 'B')
    files = []
    # 1. 保留の L1(best)と門の線
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    lb = b['l1_best_us']
    hbar(ax[0], rows, 'l1_best_us', '保留の人の走行 4,670 フレームの L1(best)', 'µs(小さいほど人に近い)',
         lines=((lb, 'B', '-'), (lb + 10, '門 B+10', '--')), xlim=(0, max(lb + 22, 75)))
    ls = b['l1_steer_best_us']
    hbar(ax[1], rows, 'l1_steer_best_us', 'ステアだけの L1(best。SLOW の門 A1)', 'µs',
         lines=((ls, 'B', '-'), (ls + 10, 'A1 B+10', '--')), xlim=(0, max(ls + 22, 75)))
    p = FIG / 'f1_l1_heldout.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p)
    # 2. 介入の集合と R1 の手前
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    hbar(ax[0], rows, 'l1_interv_us', '介入の集合 1,570 フレームの L1(desk_key の 1 番目)', 'µs(小さいほど良い)',
         lines=((b['l1_interv_us'], 'B', '-'),), xlim=(0, max(r['l1_interv_us'] or 0 for r in rows) * 1.15))
    vals = [r.get('r1_app_thr_diff_us') for r in rows if r.get('r1_app_thr_diff_us') is not None]
    errs = [x for r in rows for x in ((r.get('ref') or {}).get('r1_app_thr_diff_ci90_us') or []) if x is not None]
    lo = min(vals + errs + [-25.0]) - 6
    hi = max(vals + errs + [5.0]) + 14
    hbar(ax[1], rows, 'r1_app_thr_diff_us', 'R1 の手前 1 秒: 予測 − 人のスロットル(正 = 緩める)', 'µs(線: 起点 12 のかたまりの 90% 区間)',
         lines=((0, '人', '-'), (-10, 'low', ':'), (-20, 'high', '--')),
         err=lambda r: (r.get('ref') or {}).get('r1_app_thr_diff_ci90_us'), xlim=(lo, hi))
    p = FIG / 'f2_interv_r1.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p)
    # 3. 振れ幅と推論の時間
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    hbar(ax[0], rows, 'steer_amp_ratio', '予測ステアの振れ幅 ÷ 人の振れ幅', '比(1.0 = 人と同じ)', fmt='%.3f',
         lines=((0.8, '0.80', '--'), (1.0, '1.0', '-'), (1.25, '1.25', '--')), xlim=(0.6, 1.4))
    hbar(ax[1], rows, 'trt_p99_ms', 'TensorRT の推論の時間(99% 点)', 'ms(門 ≤ 10)', fmt='%.2f',
         lines=((8.0, 'low ≤ 8', ':'), (10.0, '門 10', '--')), xlim=(0, 11.5))
    p = FIG / 'f3_amp_trt.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p)
    return files


# ---------------------------------------------------------------- md

def f(x, nd=1):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return '合格' if x else '**不合格**'
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return '—'
    return ('%.' + str(nd) + 'f') % v


def sgn(x, nd=1):
    if x is None:
        return '—'
    return ('%+.' + str(nd) + 'f') % float(x)


def md(doc, files):
    rows = doc['candidates']
    b = next(r for r in rows if r['id'] == 'B')
    L = []
    w = L.append
    w('# 机上のベンチマーク(段 Desk、2026-10-21)')
    w('')
    w('- 実車のベンチ real_bench_20261023 の候補 9 本(B と候補 8 本)の机上の指標です。欄の名前は protocol.json の desk_fields と A1 のとおりです。')
    w('- 機械が読む版: `desk_bench.json`(このフォルダ)。作った台本: `/home/jetson/jetformer/experiments/20261021_bench_prep/desk/`。')
    w('- protocol.json の sha256: `%s`(protocol.sha256 と%s)' % (doc['protocol_sha256'], '一致' if doc['protocol_sha256_ok'] else '**不一致**'))
    w('- 作成: %s' % doc['created_at'])
    w('- 車は走らせていません。装置・jetformer-web には触れていません。GPU は jf-heavy --lock big で 1 つずつ使いました。')
    w('')
    w('## 1. 表')
    w('')
    w('### 1.1 門に使う欄')
    w('')
    w('| id | 表示名 | 家族 | l1_best_us | l1_steer_best_us | steer_amp_ratio | trt_p99_ms | engine_ok | sanity_ok | brake_lowspeed_frac | gates_pass | gate_fail_reasons |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        gp = '(掛けない)' if r['id'] == 'B' else f(r.get('gates_pass'))
        gr = '—' if r['id'] == 'B' else (', '.join(r.get('gate_fail_reasons') or []) or 'なし')
        w('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['id'], r['label'], r.get('family') or '—', f(r.get('l1_best_us')), f(r.get('l1_steer_best_us')),
            f(r.get('steer_amp_ratio'), 3), f(r.get('trt_p99_ms'), 2), f(r.get('engine_ok')), f(r.get('sanity_ok')),
            f(r.get('brake_lowspeed_frac'), 4), gp, gr))
    w('')
    w('- 門(protocol gates): engine_ok・sanity_ok・trt_p99_ms ≤ 10・0.80 ≤ steer_amp_ratio ≤ 1.25・l1_best_us − B < 10.0。')
    w('  - 家族 label_r1(SLOW)だけ A1: 5 番目を l1_steer_best_us − B < 10.0 に置き換え、brake_lowspeed_frac ≤ 0.01 を足す。')
    w('- B の値: l1_best_us %s・l1_steer_best_us %s(門の 5 番目の基準)。' % (f(b['l1_best_us'], 3), f(b['l1_steer_best_us'], 3)))
    w('')
    w('### 1.2 選び方と読み方に使う欄')
    w('')
    w('| id | l1_interv_us | r1_app_thr_diff_us(90% 区間) | B との差(90% 区間) | l1_last_us | l1_thr_best_us | trt_median_ms |')
    w('|---|---|---|---|---|---|---|')
    for r in rows:
        ref = r.get('ref') or {}
        ci = ref.get('r1_app_thr_diff_ci90_us') or [None, None]
        vb = ref.get('r1_app_thr_vs_B_us')
        cib = ref.get('r1_app_thr_vs_B_ci90_us') or [None, None]
        w('| %s | %s | %s(%s〜%s) | %s | %s | %s | %s |' % (
            r['id'], f(r.get('l1_interv_us')), sgn(r.get('r1_app_thr_diff_us')), sgn(ci[0]), sgn(ci[1]),
            '—' if r['id'] == 'B' else '%s(%s〜%s)' % (sgn(vb), sgn(cib[0]), sgn(cib[1])),
            f(r.get('l1_last_us')), f(r.get('l1_thr_best_us')), f(r.get('trt_median_ms'), 2)))
    w('')
    w('### 1.3 sanity(10/10 の AI 走行のフレームを TensorRT に)と低速のブレーキ')
    w('')
    w('| id | フレーム | NaN・Inf のフレーム | 500〜2500 µs の外 | 1000〜2000 µs の外(走行が切る。参考) | ステアの幅 µs | スロットルの幅 µs | 低速ブレーキ 保留 | 低速ブレーキ AI 走行 |')
    w('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        s = r.get('sanity') or {}
        bl = r.get('brake_lowspeed') or {}
        h, a = bl.get('heldout') or [None, None], bl.get('ai_1010') or [None, None]
        w('| %s | %s | %s | %s | %s(%s) | %s〜%s | %s〜%s | %s / %s | %s / %s |' % (
            r['id'], s.get('frames', '—'), s.get('nonfinite_frames', '—'), s.get('out_of_range_values', '—'),
            s.get('outside_drive_range_values', '—'),
            ('%.2f%%' % (100 * s['outside_drive_range_share'])) if s.get('outside_drive_range_share') is not None else '—',
            f(s.get('steer_min_us'), 0), f(s.get('steer_max_us'), 0), f(s.get('thr_min_us'), 0), f(s.get('thr_max_us'), 0),
            h[0], h[1], a[0], a[1]))
    w('')
    sp = doc.get('seed_spread_ref') or {}
    if sp:
        w('### 1.4 参考: 同じ作り方で種だけ違う学習結果の幅(門・選び方には使わない)')
        w('')
        w('| 作り方 | 本数 | l1_best_us の幅 | l1_interv_us の幅 | r1_app_thr_diff_us の幅 | steer_amp_ratio の幅 | 中身 |')
        w('|---|---|---|---|---|---|---|')
        for fam, v in sp.items():
            runs = v['runs']
            w('| %s | %d | %s | %s | %s | %s | %s |' % (
                fam, len(runs), f(v['range']['l1_best_us'], 2), f(v['range']['l1_interv_us'], 1),
                f(v['range']['r1_app_thr_diff_us'], 1), f(v['range']['steer_amp_ratio'], 3),
                '・'.join('%s %s/%s/%s' % (k, f(r['l1_best_us']), f(r['l1_interv_us']), sgn(r['r1_app_thr_diff_us']))
                         for k, r in runs.items())))
        w('')
        w('- 中身は「名前 l1_best/l1_interv/r1_app」。保存した 10/12・10/15 の予測から、同じ式で数えました。')
        w('')
    w('- 図: ' + '・'.join('`fig/%s`' % p.name for p in files))
    w('')
    w('## 2. 定義(結果を見る前に決めたもの)')
    w('')
    c = doc['counts']
    w('- 予測は PyTorch FP32(best・last の重み)。10/12・10/13・10/15 に保存した予測はそのまま読み、新しい 3 本(SLOW・WGT・AUG)はこの段で同じ手順で作りました。')
    w('  - 手順が同じことの確かめ: B の best を予測し直すと、10/12 の保存と最大 0.0 µs の差(crosscheck_B.json)。')
    w('  - B の L1 は 10/15 の値と一致(best %s・last %s・介入 %s µs)。' % tuple(
        f(doc['check_realacc_B'][k][1], 3) for k in ('l1_best_us', 'l1_last_us', 'l1_interv_us')))
    w('  - ENS は v0-s2 と msf-s2 の予測の平均です(ens_average と同じ。S4 で、まとめた best.pth と 0.00009 µs 以内)。ENS の last は 2 本の last の平均です。')
    w('- 集合')
    for k, v in doc['definitions']['sets'].items():
        w('  - %s: %s' % (k, v))
    for k in ('l1', 'r1_app_thr_diff_us', 'steer_amp_ratio', 'brake_lowspeed_frac', 'engine_ok', 'sanity_ok', 'trt'):
        w('- %s: %s' % (k, doc['definitions'][k]))
    w('- r1_app の 90%% 区間: R1 の起点(%d 個)のかたまりを復元抽出(2,000 回・種 0・百分位)。B との差は同じフレームの差を同じ方法で。' % c['r1_anchors_used'])
    w('- sanity の幅 500〜2500 µs: 12:33 の 200 フレームの試しで、ステアが 1000 µs を少し下回る値(908〜1005 µs)がほぼ全部に出ました。人の正解も 983 µs まであります。走行は 1000〜2000 µs に切るので、壊れたモデルを見つける幅として 500〜2500 µs にしました。1000〜2000 µs の外の数は参考に出します。**本番の全フレームを見る前に決めました。**')
    w('')
    w('## 3. 読み方')
    w('')
    if NOTES.exists():
        L.extend(NOTES.read_text(encoding='utf-8').rstrip().split('\n'))
    else:
        w('- (見えたことはまだ書いていません)')
    w('')
    w('## 4. 数え直し')
    w('')
    w('- 予測・正解・集合は `pred/desk_inputs.npz`、10/10 の AI 走行の TensorRT の出力は `pred/sanity_<id>.npz` に保存しました。')
    w('- `recount.py`(desk_lib・desk_eval を読まない別のコード)で数え直せます。結果は `experiments/20261021_bench_prep/desk/logs/recount.json`。')
    (OUT / 'desk_bench.md').write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    doc = json.load(open(OUT / 'desk_bench.json', encoding='utf-8'))
    files = figures(doc)
    md(doc, files)
    print('書きました:', OUT / 'desk_bench.md', [str(p) for p in files])


if __name__ == '__main__':
    main()
