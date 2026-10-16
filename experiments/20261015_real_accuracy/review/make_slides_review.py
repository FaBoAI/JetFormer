#!/usr/bin/env python3
"""見直しの段(review)で作り直した、約 1 分の動画の図 6 枚(2160 × 1350)。実機のカメラの画像(保留の 11 シーン)と、
前回のシミュレーターの実際の画面を図に入れる(CPU だけ)。

    cd ~/jetformer && CUDA_VISIBLE_DEVICES= /home/jetson/bin/jf-heavy --mem 1500M --lock none -- nice -n 10 \\
        /home/jetson/miniconda/envs/jetracer/bin/python experiments/20261015_real_accuracy/review/make_slides_review.py

  r1_problem  前回の課題: シミュレーターでは完走できたが、実機の精度は上がらなかった(シミュレーターの画面・実機のカメラ)
  r2_plan     今日: 3 つの観点で考えて 5 つ試した(保留の 11 シーンの実機のカメラ)
  r3_battery  誤差の大きいシーン = 電池が減ったシーン(そのシーンの実機のカメラ・スロットルと回転数・誤差の時間変化)
  r4_result   結果: 平均(ENS)と電池の補正(GAIN)が「良い」、ただし際どい・ほぼ 1 シーン
  r5_size     データの量の曲線(録画を増やすと良くなりそう)
  r6_floor    精度の床(人の揺れ)と今の位置
数は review/recount.json(保存した予測から自分のコードで数え直した値 = evaluation.json と一致)と、これまでの報告の値。
"""
import json
import runpy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jetson/jetformer')
CLEAN = REPO / 'data' / 'datasets' / 'clean_driving_20261011'
REP = Path('/home/jetson/jetformer_reports/real_accuracy_20261015')
OUT = REP / 'video' / 'fig'
SIM_LAP = Path('/home/jetson/jetformer_reports/sim_diverse_train_20261015/screens/lap_SDIV_E2')
RC = json.loads((REP / 'review' / 'recount.json').read_text(encoding='utf-8'))
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, GREEN, PURPLE, RED, GOLD, GRAY = '#2a78d6', '#eb6834', '#1bab7a', '#6a4fc0', '#c0392b', '#c98500', '#8a8985'
W_IN, H_IN, DPI = 14.4, 9.0, 150
BATTERY = '20261010-193638-0003'


def jp_font():
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            return name


def slide(title, sub):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=SURFACE)
    fig.text(0.035, 0.935, title, fontsize=30, fontweight='bold', color=INK, va='center')
    fig.text(0.037, 0.875, sub, fontsize=15, color=INK2, va='center')
    return fig


def box(fig, x, y, w, h, edge, face, lines):
    ax = fig.add_axes([x, y, w, h])
    ax.axis('off')
    asp = (w * W_IN) / (h * H_IN)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle='round,pad=0.0,rounding_size=0.03', transform=ax.transAxes,
                                facecolor=face, edgecolor=edge, linewidth=3, mutation_aspect=asp))
    n = len(lines)
    for i, (t, size, bold, col) in enumerate(lines):
        ax.text(0.04, 1 - (i + 0.75) / (n + 0.4), t, fontsize=size, fontweight='bold' if bold else 'normal', color=col,
                transform=ax.transAxes, va='center')
    return ax


def banner(fig, text, y=0.035):
    ax = fig.add_axes([0.03, y, 0.94, 0.075])
    ax.axis('off')
    asp = (0.94 * W_IN) / (0.075 * H_IN)
    ax.add_patch(FancyBboxPatch((0.005, 0.05), 0.99, 0.9, boxstyle='round,pad=0.0,rounding_size=0.012', transform=ax.transAxes,
                                facecolor='#f0f4fa', edgecolor=INK, linewidth=3.5, mutation_aspect=asp))
    ax.text(0.02, 0.5, text, fontsize=21, fontweight='bold', color=INK, transform=ax.transAxes, va='center')


def show(fig, im, rect, label=None, color=None, size=14):
    ax = fig.add_axes(rect)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color or GRID)
        sp.set_linewidth(3 if color else 2)
    if label:
        ax.set_title(label, fontsize=size, color=color or INK2, fontweight='bold', pad=6)
    return ax


def view3d(path):
    return Image.open(path).convert('RGB').crop((34, 116, 1038, 540))


def scene_arrays(s):
    with np.load(CLEAN / 'aligned' / (s + '.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in ('timestamps_us', 'vehicle', 'cam_files', 'mode')}


def real_frame(s, t_s=None, frac=None):
    a = scene_arrays(s)
    t = (a['timestamps_us'] - a['timestamps_us'][0]) / 1e6
    i = int(np.searchsorted(t, t_s)) if t_s is not None else int(frac * (len(t) - 1))
    i = min(i, len(t) - 1)
    return Image.open(CLEAN / str(a['cam_files'][i])).convert('RGB'), a, i, t[i]


def style_ax(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#bdbbb5')
    ax.spines['bottom'].set_color('#bdbbb5')
    ax.tick_params(colors=INK2, labelsize=13)
    ax.grid(color=GRID, lw=1, zorder=0)


# ---------------------------------------------------------------- r1

PAST = [('今の作り方 A(10/12)', 49.30), ('ほかの 4 つの形(10/12)', None), ('シミュ混合 C(10/13)', 49.73),
        ('いろいろな走り C-DIV(10/15)', 50.04)]


def r1_problem():
    fig = slide('前回の課題: シミュレーターでは完走、でも実機の精度は同じ',
                '10/15 の 1 本目: いろいろな走りのデータで学習 → シミュレーターの 1 周は 12 回中 5 回完走。実機の保留データの誤差は変わらず')
    show(fig, view3d(SIM_LAP / 'eval_SDIV-s0-E2-single_f123_window.png'), [0.03, 0.45, 0.43, 0.36],
         'シミュレーター(前回の S-DIV・完走した 1 周)', color=BLUE, size=15)
    show(fig, Image.open(SIM_LAP / 'eval_SDIV-s0-E2-single_f123_camera.png').convert('RGB'), [0.03, 0.14, 0.2, 0.27],
         'シミュレーターの車載カメラ', size=13)
    im, _, _, _ = real_frame('20261010-202633-0007', frac=0.4)
    show(fig, im, [0.25, 0.14, 0.2, 0.27], '実機のカメラ(保留のシーン)', color=ORANGE, size=13)
    ax = fig.add_axes([0.62, 0.20, 0.35, 0.58])
    rows = [('今の作り方 A', 49.30, BLUE), ('多段の融合', 49.30, GRAY), ('TCP 型', 49.73, GRAY), ('ACT 型', 50.38, GRAY),
            ('DINOv2 型', 50.40, GRAY), ('シミュ混合 C', 49.73, GRAY), ('C-DIV(いろいろな走り)', 50.04, ORANGE)]
    for k, (n, v, c) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.scatter([v], [y], s=260, color=c, zorder=3)
        ax.text(v + 0.08, y, '%.2f' % v, va='center', fontsize=15, color=INK)
    ax.axvspan(49.30 - 0.626, 49.30 + 0.626, color=BLUE, alpha=0.10, lw=0, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _, _ in rows][::-1], fontsize=14)
    ax.set_xlim(48.4, 51.0)
    ax.set_xlabel('実機の保留データの誤差 [µs](小さいほど良い)', fontsize=14, color=INK2)
    style_ax(ax)
    ax.set_title('実機の誤差は、どれも約 49 µs(帯 = 同等の幅)', fontsize=16, color=INK, loc='left', fontweight='bold', pad=8)
    banner(fig, '今日の 2 本目: 実機の精度に効く方策を考えて、試す(判定の決まりは学習の前に固定)')
    fig.savefig(OUT / 'r1_problem.png', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- r2

def r2_plan():
    fig = slide('今日: 3 つの観点で考え、5 つを試した',
                '確かめる場所: 保留した人のきれいな走行 11 シーン・4,670 フレーム(下の写真は、その実機のカメラ)。車は走らせていない')
    val = sorted(RC['per_scene_best'])
    for i, s in enumerate(val):
        im, _, _, _ = real_frame(s, frac=0.4)
        x = 0.03 + i * 0.0855
        show(fig, im, [x, 0.62, 0.08, 0.2], s[9:15] + '-' + s[-1], color=RED if s == BATTERY else None, size=11)
    fig.text(0.03, 0.585, '赤 = 電池が減っていたシーン(誤差の 24% がこの 1 シーン)', fontsize=13, color=RED, fontweight='bold')
    cols = [
        ('① 誤差の中身から', BLUE, '#eef4fc', ['平均(ENS): 形の違う 2 本の予測の平均',
                                              '電池の補正(GAIN): 減った効きを補う',
                                              '時間の平均(TE): 前の予測も混ぜる']),
        ('② シミュレーターとの差から', PURPLE, '#f3f1fa', ['画像の差は色でなく中身(床・壁)',
                                                     'シミュの役目は平均の相手だけ',
                                                     '→ 実機のデータの使い方を見直す']),
        ('③ 精度の床から', GREEN, '#ecf7f2', ['位相を回す(ROT): 捨てた 2/3 も使う',
                                            'データの量(SIZE): 減らして伸びを測る',
                                            '床(人の揺れ)は約 31 µs と見積もり']),
    ]
    for i, (t, col, face, items) in enumerate(cols):
        box(fig, 0.03 + i * 0.318, 0.14, 0.305, 0.40, col, face,
            [(t, 18, True, col)] + [(x, 14.5, False, INK) for x in items])
    banner(fig, '判定: 2 つの種の両方が A より小さく、差が閾値(0.6 µs・1.3 µs)より大きいときだけ「良い」')
    fig.savefig(OUT / 'r2_plan.png', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- r3

def r3_battery():
    R = runpy.run_path(str(HERE / 'recount.py'), run_name='recount_lib')
    part, scene, frame, Y, primary = R['part'], R['scene'], R['frame'], R['Y'], R['primary']
    m = (part == 0) & (scene == BATTERY)
    a = scene_arrays(BATTERY)
    t_all = (a['timestamps_us'] - a['timestamps_us'][0]) / 1e6
    fr = frame[m]
    t = t_all[fr]
    ok = primary[m]
    errA = (np.abs(R['V']['best']['A_s0'] - Y).mean(axis=(1, 2)) + np.abs(R['V']['best']['A_s1'] - Y).mean(axis=(1, 2)))[m] / 2
    errG = (np.abs(R['V']['best']['GAIN(A_s0)'] - Y).mean(axis=(1, 2))
            + np.abs(R['V']['best']['GAIN(A_s1)'] - Y).mean(axis=(1, 2)))[m] / 2
    sb = RC['per_scene_best'][BATTERY]
    fig = slide('誤差の大きいシーンは、電池が減っていた',
                '保留のシーン 20261010-193638-0003(高速・フレームの 12%)。人はスロットルを強く踏んでいるのに、回転数(速さ)が落ちていく')
    for i, ts in enumerate((2.0, 9.5, 18.0)):
        im, aa, k, tt = real_frame(BATTERY, t_s=ts)
        thr, erpm = aa['vehicle'][k, 1], -aa['vehicle'][k, 3]
        show(fig, im, [0.03 + i * 0.148, 0.565, 0.138, 0.221], '%.1f 秒' % tt, color=RED if ts > 15 else None, size=14)
        fig.text(0.03 + i * 0.148 + 0.069, 0.55, 'スロットル %d\n回転数 %d' % (round(thr), round(erpm)), fontsize=12.5,
                 color=INK2, ha='center', va='top')
    ax = fig.add_axes([0.53, 0.53, 0.44, 0.27])
    tf = t_all
    sm = lambda x: np.convolve(x, np.ones(15) / 15, mode='same')  # noqa: E731
    ax.plot(tf, sm(1498 - a['vehicle'][:, 1]), color=ORANGE, lw=2.5, label='踏み込み(1498 − スロットル)[µs]')
    ax.plot(tf, sm(-a['vehicle'][:, 3]) / 10, color=BLUE, lw=2.5, label='回転数 ÷ 10 [eRPM]')
    ax.set_xlim(0, tf[-1])
    ax.set_ylim(0, 800)
    style_ax(ax)
    ax.legend(fontsize=12, loc='upper left', frameon=False, ncol=2)
    ax.set_title('人の踏み込みと回転数(1/2 秒の平均)', fontsize=15, color=INK, loc='left', fontweight='bold')
    ax2 = fig.add_axes([0.53, 0.17, 0.44, 0.27])
    ax2.scatter(t[ok], errA[ok], s=6, color=GRAY, label='今の作り方 A', zorder=2)
    ax2.scatter(t[ok], errG[ok], s=6, color=GREEN, label='A + 電池の補正', zorder=3)
    ax2.set_xlim(0, tf[-1])
    ax2.set_ylim(0, 320)
    ax2.set_xlabel('シーンの中の時間 [秒]', fontsize=13, color=INK2)
    style_ax(ax2)
    ax2.legend(fontsize=12, loc='upper left', frameon=False, markerscale=4, ncol=2)
    ax2.set_title('フレームごとの誤差 [µs]', fontsize=15, color=INK, loc='left', fontweight='bold')
    box(fig, 0.03, 0.14, 0.45, 0.30, RED, '#fbefed', [
        ('このシーンだけで誤差の 24%', 18, True, INK),
        ('A(種 0)の誤差 %.1f µs(ほかの 10 シーンは %.1f µs)' % (sb['A_s0'], RC['per_variant_all']['best']['A_s0']['S1_no_battery_scene']), 14, False, INK),
        ('電池の補正で %.1f → %.1f µs' % (sb['A_s0'], sb['GAIN(A_s0)']), 14, False, INK),
        ('モデルは今の操作を見ないので、電池の減りを知らない', 14, False, INK)])
    banner(fig, '電池の補正: 直前 5 秒の「踏み込み ÷ 回転数」から、前進のスロットルを強める')
    fig.savefig(OUT / 'r3_battery.png', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- r4

def r4_result():
    q = RC['questions']
    pv = RC['per_variant_primary']['best']
    fig = slide('結果: 平均と電池の補正が「良い」。ただし小さく、際どい',
                '保留データ 4,670 フレームの誤差(best の checkpoint・点 = 2 つの種)。帯 = A の平均 ± 閾値 0.626 µs')
    rows = [('今の作り方 A(種 0・1)', ['A_s0', 'A_s1'], BLUE, None),
            ('平均 ENS(v0 + 多段、新しい種)', ['P2', 'P3'], ORANGE, 'ENS_vs_A'),
            ('電池の補正 GAIN(A に)', ['GAIN(A_s0)', 'GAIN(A_s1)'], GREEN, 'GAIN_i'),
            ('同じ形 2 本の平均', ['SAME_A', 'SAME_new'], GRAY, 'SAME_vs_A'),
            ('時間の平均 TE', ['TE(A_s0)', 'TE(A_s1)'], GRAY, 'TE_i'),
            ('位相を回す ROT', ['rot-s0', 'rot-s1'], PURPLE, 'ROT_vs_A'),
            ('v0 の新しい種 2・3', ['v0-s2', 'v0-s3'], GRAY, 'FRESH_A')]
    ja = {'better': '良い', 'equivalent': '同等', 'worse': '悪い'}
    ax = fig.add_axes([0.25, 0.17, 0.40, 0.66])
    for k, (n, vs, c, qid) in enumerate(rows):
        y = len(rows) - 1 - k
        xs = [pv[v] for v in vs]
        ax.scatter(xs, [y] * 2, s=240, color=c, zorder=3)
        if qid:
            g = q[qid]['primary']['best']['gain_us']
            ax.text(max(xs) + 0.18, y, '%s %+.1f' % (ja[q[qid]['primary']['final']], -g), va='center', fontsize=15,
                    color=INK, fontweight='bold' if q[qid]['primary']['final'] == 'better' else 'normal')
    a_mean = (pv['A_s0'] + pv['A_s1']) / 2
    ax.axvspan(a_mean - 0.626, a_mean + 0.626, color=BLUE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(a_mean, color=BLUE, ls='--', lw=1.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _, _, _ in rows][::-1], fontsize=14)
    ax.set_xlim(44.0, 52.5)
    ax.set_xlabel('誤差 [µs](小さいほど良い)・数字 = A との差', fontsize=13, color=INK2)
    style_ax(ax)
    s = RC['gain_battery_share']['GAIN_i']
    box(fig, 0.70, 0.52, 0.27, 0.31, ORANGE, '#fdf1ea', [
        ('平均 ENS: 際どい', 17, True, INK),
        ('best −2.6・last −2.3 µs', 14, False, INK),
        ('v0 の新しい種が A より 0.9 悪い', 14, False, INK),
        ('4 本のばらつきで測ると「同等」', 14, False, INK)])
    box(fig, 0.70, 0.17, 0.27, 0.31, GREEN, '#ecf7f2', [
        ('電池の補正: ほぼ 1 シーン', 17, True, INK),
        ('−4.2 µs のうち %d%% が' % round(100 * s['share']), 14, False, INK),
        ('電池の減ったシーンから', 14, False, INK),
        ('ほかの 10 シーンは +0.4 µs', 14, False, INK)])
    banner(fig, 'どれも実用上の差 10 µs 未満。走行のおすすめはモデル B のまま(実車の試しはカードに)')
    fig.savefig(OUT / 'r4_result.png', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- r5

def r5_size():
    pv = RC['per_variant_primary']['best']
    sz = RC['size']['primary/best']
    N = {'Q': 1682, 'H': 3046, 'T': 4682, 'full': 5912}
    fig = slide('データの量: 減らすと誤差が増えた → 増やせば、まだ下がりそう',
                '学習のシーンを 29%・52%・79% に減らして、同じ作り方で学習(種 0・1)。誤差は保留データ 4,670 フレーム')
    ax = fig.add_axes([0.07, 0.17, 0.50, 0.64])
    xs, ys = [], []
    for k, names in (('Q', ['sizeQ-s0', 'sizeQ-s1']), ('H', ['sizeH-s0', 'sizeH-s1']), ('T', ['sizeT-s0', 'sizeT-s1']),
                     ('full', ['A_s0', 'A_s1'])):
        v = [pv[n] for n in names]
        ax.scatter([N[k]] * 2, v, s=110, color=BLUE, alpha=0.55, zorder=3)
        xs.append(N[k])
        ys.append(np.mean(v))
    ax.plot(xs, ys, color=BLUE, lw=3, zorder=2, marker='o', ms=12)
    for x, y, lab in zip(xs, ys, ('29%', '52%', '79%', '全部')):
        ax.text(x, y + 0.9, '%s\n%.1f' % (lab, y), ha='center', va='bottom', fontsize=14, color=INK)
    x2 = N['full'] * 2
    ax.plot([N['full'], x2], [ys[-1], ys[-1] - sz['s2']], color=BLUE, lw=2, ls=':', zorder=2)
    ax.plot([N['full'], x2], [ys[-1], ys[-1] - sz['G2_pow']], color=BLUE, lw=2, ls=':', zorder=2)
    ax.fill_between([N['full'], x2], [ys[-1], ys[-1] - sz['s2']], [ys[-1], ys[-1] - sz['G2_pow']], color=BLUE, alpha=0.12, lw=0)
    ax.text(x2, ys[-1] - sz['s2'] - 0.4, '2 倍の見込み\n%.1f〜%.1f' % (ys[-1] - sz['s2'], ys[-1] - sz['G2_pow']),
            ha='center', va='top', fontsize=14, color=BLUE, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1682, 3046, 5912, 11824])
    ax.set_xticklabels(['1,682', '3,046', '5,912', '2 倍'])
    ax.set_xlim(1400, 15000)
    ax.set_ylim(42, 63)
    ax.set_xlabel('学習のサンプル数(間引き 3)', fontsize=13, color=INK2)
    ax.set_ylabel('誤差 [µs]', fontsize=13, color=INK2)
    style_ax(ax)
    eq = RC['size_eq_seed0']['best']
    box(fig, 0.62, 0.50, 0.35, 0.31, BLUE, '#eef4fc', [
        ('2 倍で 2〜4 µs の見込み', 18, True, INK),
        ('平らになる形で 2.1 µs・まっすぐなら 3.9 µs', 14, False, INK),
        ('選び直しの幅は 0.5〜5.0 µs(読みは際どい)', 14, False, INK),
        ('学習の回数をそろえても、効果の大半が残った', 14, False, INK)])
    box(fig, 0.62, 0.17, 0.35, 0.29, PURPLE, '#f3f1fa', [
        ('位相を回す(同じ周のフレーム)は効かず', 16, True, INK),
        ('効くのは「新しい周」の数', 14, False, INK),
        ('(回数をそろえた確かめは種 0 だけの参考:', 13, False, INK2),
        (' 52%% で %.1f → %.1f µs)' % (eq['LH30'], eq['LH59']), 13, False, INK2)])
    banner(fig, 'きれいな周回の録画を増やすのが、次に効きそうな方策(電池の残りも記録する)')
    fig.savefig(OUT / 'r5_size.png', facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- r6

def r6_floor():
    pv = RC['per_variant_primary']['best']
    fig = slide('精度の床(人の揺れ)まで、まだ余地がある',
                '床 = 同じ場所・同じ状態で、人が別の周に出す操作の差から見積もった、下げられない誤差(コース上の位置を当てはめて推定)')
    ax = fig.add_axes([0.30, 0.20, 0.64, 0.60])
    rows = [('今の作り方 A', (pv['A_s0'] + pv['A_s1']) / 2, BLUE),
            ('今ある 10 本の平均(参考)', 46.1, GRAY),
            ('平均 ENS(新しい種)', (pv['P2'] + pv['P3']) / 2, ORANGE),
            ('平均 + 電池の補正(参考)', (pv['GAIN(P2)'] + pv['GAIN(P3)']) / 2, GREEN)]
    for k, (n, v, c) in enumerate(rows):
        y = len(rows) - k
        ax.barh(y, v, color=c, height=0.55, zorder=3)
        ax.text(v + 0.6, y, '%.1f' % v, va='center', fontsize=17, color=INK, fontweight='bold')
    ax.axvspan(22, 44, color=GREEN, alpha=0.10, lw=0, zorder=0)
    ax.axvline(31, color=GREEN, lw=3, ls='--', zorder=4)
    ax.barh(0, 31, color='#cfeadf', height=0.55, zorder=2)
    ax.text(31 + 0.6, 0, '約 31(幅 22〜44)', va='center', fontsize=17, color=GREEN, fontweight='bold')
    ax.set_yticks(range(len(rows) + 1))
    ax.set_yticklabels(['床の見積もり(人の揺れ)'] + [n for n, _, _ in rows][::-1], fontsize=15)
    ax.set_xlim(0, 60)
    ax.set_xlabel('誤差 [µs](小さいほど良い)', fontsize=14, color=INK2)
    style_ax(ax)
    banner(fig, '49 µs は床ではない。ただ下がったのは数 µs で、実用上の差 10 µs にはまだ届かない')
    fig.savefig(OUT / 'r6_floor.png', facecolor=SURFACE)
    plt.close(fig)


def main():
    jp_font()
    OUT.mkdir(parents=True, exist_ok=True)
    r1_problem()
    r2_plan()
    r3_battery()
    r4_result()
    r5_size()
    r6_floor()
    for p in sorted(OUT.glob('r*.png')):
        print(p, Image.open(p).size)


if __name__ == '__main__':
    main()
