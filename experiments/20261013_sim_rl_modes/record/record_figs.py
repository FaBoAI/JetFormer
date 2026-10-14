#!/usr/bin/env python3
"""記録の段の図(2026-10-14、sim_rl_modes)。record/variation.mjs の --figdata と --out の JSON から描く。

    python -I record/record_figs.py --figdata fig.json --variation variation_raw.json --outdir <reports>/fig

  record_trajectories.png   モードごとの軌跡の重ね描き(コースの柵・柱・開始の点。離脱復活は乱した操作の区間を濃く)
  record_speed_hist.png     速さの分布(開始から 3 m 以降のフレーム)
  record_clearance_hist.png すき間の分布(フレームごと。シーンの最小は下の印)
  record_ey_by_s.png        参照ループ上の位置 s ごとの横ずれ e_y(線のばらつき = VM3 の元)
  record_variation_summary.png  VM3・VM7 の教師との比(モードごと・全モード合わせ)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 色(dataviz の参照パレットの順。教師は中立の灰色)
COLORS = {'teacher': '#7a7974', 'fastest': '#2a78d6', 'no_wall': '#eb6834', 'recover': '#1baf7a', 'speed': '#c98500'}
NAMES = {'teacher': '前の教師(RC加速教師・60 シーンの先頭 15)', 'fastest': '最速', 'no_wall': '絶対壁にぶつからない', 'recover': '離脱復活',
         'speed': '速さいろいろ'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def jp_font():
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'Noto Serif CJK JP'):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = name
            break
    matplotlib.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                                'axes.titlecolor': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110})


def draw_course(ax, course):
    for a, b, kind in course['segments']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#9a9993', lw=1.2, solid_capstyle='round', zorder=1)
    for x, z, r, kind in course['disks']:
        ax.add_patch(plt.Circle((x, z), max(r, .03), color='#9a9993', zorder=1))
    b = course['bounds']
    ax.set_xlim(b['minX'] - .1, b['maxX'] + .1)
    ax.set_ylim(b['maxZ'] + .1, b['minZ'] - .1)       # 上から見た図(z は下向き)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def order(fig):
    return [k for k in ('teacher', 'fastest', 'no_wall', 'recover', 'speed') if k in fig['groups']]


def trajectories(fig, var, out):
    keys = order(fig)
    n = len(keys)
    cols = 3 if n > 4 else n
    rows = int(np.ceil(n / cols))
    f, axs = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.9 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, k in zip(axs, keys):
        draw_course(ax, fig['course'])
        sc = fig['groups'][k][:15] if k == 'teacher' else fig['groups'][k]
        for s in sc:
            x, z = np.array(s['x']), np.array(s['z'])
            ax.plot(x, z, color=COLORS[k], lw=.8, alpha=.55, zorder=2)
            if k == 'recover':
                l = np.array(s['l'])
                xm, zm = np.where(l == 0, x, np.nan), np.where(l == 0, z, np.nan)
                ax.plot(xm, zm, color=INK, lw=1.6, alpha=.9, zorder=3)
            ax.plot(x[0], z[0], 'o', ms=4.5, mfc=COLORS[k], mec='white', mew=1, zorder=4)
        g = (var.get('modes') or {}).get(k) or ((var.get('teacher') or {}).get('first15') if k == 'teacher' else None) or {}
        vm3 = (g.get('VM3_m') or {}).get('mean_abs_dev')
        sub = f'{len(sc)} シーン・VM3 {vm3 * 100:.2f} cm' if vm3 is not None else f'{len(sc)} シーン'
        ax.set_title(f'{NAMES[k]}\n{sub}', fontsize=10, loc='left')
    for ax in axs[n:]:
        ax.axis('off')
    if 'recover' in keys:
        f.text(.01, .01, '離脱復活の濃い線 = わざと乱した操作の区間(学習から外す)。丸 = 開始の位置。', fontsize=8.5, color=INK2)
    f.suptitle('記録した走りの軌跡(上から見た図。灰色の線と点はコースの柵と柱)', x=.01, ha='left', fontsize=12, color=INK)
    f.tight_layout(rect=(0, .02, 1, .96))
    f.savefig(out, dpi=130)
    plt.close(f)


def small_multiple_hist(fig, var, out, field, bins, xlabel, title, after3m=True, rug=None, xlim=None, overflow=False):
    keys = order(fig)
    f, axs = plt.subplots(len(keys), 1, figsize=(8.2, 1.55 * len(keys) + .8), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        sc = fig['groups'][k][:15] if k == 'teacher' else fig['groups'][k]
        vals = []
        for s in sc:
            v, ss = np.array(s[field]), np.array(s['s'])
            vals.append(v[ss - s['s0'] >= 3] if after3m else v)
        vals = np.concatenate(vals)
        med = float(np.median(vals))
        if overflow:
            over = float(np.mean(vals > bins[-1]))
            ax.hist(vals[vals <= bins[-1]], bins=bins, color=COLORS[k], edgecolor='white', linewidth=.6, weights=np.full(int((vals <= bins[-1]).sum()), 1 / len(vals)))
        else:
            ax.hist(np.clip(vals, bins[0], bins[-1]), bins=bins, color=COLORS[k], edgecolor='white', linewidth=.6, density=True)
        if not overflow or med <= bins[-1]:
            ax.axvline(med, color=INK, lw=1)
        if not overflow or med <= bins[-1]:
            ax.text(med, ax.get_ylim()[1] * .92, f' 中央値 {med:.2f}', fontsize=8.5, color=INK, va='top')
        if rug:
            mins = rug(k, sc)
            ax.plot(mins, np.full(len(mins), -.06 * ax.get_ylim()[1]), '|', color=INK, ms=9, mew=1.2, clip_on=False)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        lab = NAMES[k].split('(')[0] + (f'\n{bins[-1]:.1f} m 超 {over * 100:.0f}%' if overflow else '')
        ax.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=9.5, color=INK)
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.set_axisbelow(True)
    if xlim:
        axs[-1].set_xlim(*xlim)
    axs[-1].set_xlabel(xlabel)
    f.suptitle(title, x=.01, ha='left', fontsize=11.5, color=INK)
    f.tight_layout(rect=(0, 0, 1, .95))
    f.savefig(out, dpi=130)
    plt.close(f)


def ey_by_s(fig, var, out):
    keys = order(fig)
    L = fig['loop_length']
    f, axs = plt.subplots(len(keys), 1, figsize=(9.5, 1.5 * len(keys) + .8), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        sc = fig['groups'][k][:15] if k == 'teacher' else fig['groups'][k]
        for s in sc:
            ss, ey = np.array(s['s']), np.array(s['ey'])
            keep = ss - s['s0'] >= 3
            sm = np.mod(ss, L)
            sm = np.where(keep, sm, np.nan)
            jump = np.r_[False, np.diff(sm) < -L / 2]
            sm[jump] = np.nan
            ax.plot(sm, ey * 100, color=COLORS[k], lw=.7, alpha=.6)
        ax.axhline(0, color=INK2, lw=.6)
        ax.set_ylabel(NAMES[k].split('(')[0], rotation=0, ha='right', va='center', fontsize=9.5, color=INK)
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)
    axs[-1].set_xlabel('参照ループ上の位置 s [m](1 周 %.2f m。各シーンの開始から 3 m 以降)' % L)
    f.suptitle('位置ごとの横ずれ e_y [cm](参照ループの左が正)— 線が広がるほど、通る経路がいろいろ', x=.01, ha='left', fontsize=11.5, color=INK)
    f.tight_layout(rect=(0, 0, 1, .95))
    f.savefig(out, dpi=130)
    plt.close(f)


def summary(var, out):
    j = var.get('judged') or {}
    ref = j.get('reference') or {}
    keys = [k for k in ('fastest', 'no_wall', 'recover', 'speed') if k in (var.get('modes') or {})]
    labels = [NAMES[k] for k in keys] + ['全モード合わせ']
    t3, t7 = ref.get('teacher_VM3_m'), ref.get('teacher_VM7_cells_15')
    vm3 = [var['modes'][k]['VM3_m']['mean_abs_dev'] / t3 for k in keys] + [var['pooled']['VM3_m']['mean_abs_dev'] / t3]
    vm7 = [var['modes'][k]['VM7']['cells'] / t7 for k in keys] + [var['pooled']['VM7_equal_frames']['cells'] / t7]
    cols = [COLORS[k] for k in keys] + ['#4a3aa7']
    f, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, vals, ttl, goal in ((axs[0], vm3, 'VM3 線のばらつき(教師 = 1)', 10), (axs[1], vm7, 'VM7 状態の網羅(教師 = 1)', 3)):
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, color=cols, height=.62, edgecolor='white', linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f' {v:.1f} 倍', va='center', fontsize=9, color=INK,
                    bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
        ax.axvline(1, color=INK2, lw=1)
        ax.axvline(goal, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_title(f'{ttl}\n点線 = 全モード合わせの目標 {goal} 倍', loc='left', fontsize=10.5)
        ax.grid(axis='x', color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(max(vals), goal) * 1.18)
    f.suptitle('前の教師のデータ(RC加速教師 60 シーン)と比べたばらつき', x=.01, ha='left', fontsize=11.5, color=INK)
    f.tight_layout(rect=(0, 0, 1, .93))
    f.savefig(out, dpi=130)
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figdata', required=True)
    ap.add_argument('--variation', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--replay', action='append', default=[], help='key=replay_check の JSON(シーンの正確な最小すき間)')
    a = ap.parse_args()
    jp_font()
    fig = json.loads(Path(a.figdata).read_text())
    var = json.loads(Path(a.variation).read_text())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    exact = {}
    for x in a.replay:
        k, p = x.split('=', 1)
        exact[k] = {s['scene']: s['min_clearance_m'] for s in json.loads(Path(p).read_text())['scenes']}

    def rug(k, sc):
        return [exact.get(k, {}).get(s['scene'], min(s['d'])) for s in sc]
    trajectories(fig, var, out / 'record_trajectories.png')
    small_multiple_hist(fig, var, out / 'record_speed_hist.png', 'v', np.arange(0, 4.6, .1), '速さ [m/s]',
                        '速さの分布(開始から 3 m 以降のフレーム。面積 = 1)', xlim=(0, 4.6))
    small_multiple_hist(fig, var, out / 'record_clearance_hist.png', 'd', np.arange(0, .505, .01), 'すき間 [m](車体の長方形と柵・柱のいちばん近い距離)',
                        'すき間の分布(全フレーム。0.5 m までを描き、より広いフレームの割合は右に。下の縦線 = シーンごとの最小)', after3m=False, rug=rug,
                        xlim=(0, .5), overflow=True)
    ey_by_s(fig, var, out / 'record_ey_by_s.png')
    if var.get('judged', {}).get('reference') and var.get('pooled'):
        summary(var, out / 'record_variation_summary.png')
    print('書いた', sorted(p.name for p in out.glob('record_*.png')))


if __name__ == '__main__':
    main()
