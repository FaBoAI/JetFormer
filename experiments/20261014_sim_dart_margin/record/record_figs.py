#!/usr/bin/env python3
"""記録の段の図(2026-10-14、sim_dart_20261014)。入力は record/deviation_stats.mjs の出力(recording/D*/deviation.json)と
記録の前の Node の予想(record_prep/prediction_D*.json)。データは変えない。

    source ~/miniconda/etc/profile.d/conda.sh && conda activate jetracer
    python -I record_figs.py

  fig/record_dev_hist_<D>.png  乱しごとの最大のずれ(2 つの基準と Node の予想)と、フレームごとのずれ(乱しなしの記録 = 床)
  fig/record_traj_<D>.png      例のシーン: 符号つきのずれ・舵(記録した乱し入りの操作 / 付け替えた先生の正解)・上から見た軌跡
  fig/record_clearance.png     すき間の累積分布(A0・D1・D2)
  fig/record_lines.png         最速の先生の線と余裕の線(上から、最初の直線の拡大)
  fig/record_recovery_<D>.png  乱しの始まりにそろえた平均: 横ずれと、舵(記録 / 付け替えた正解)の、乱しが無かったときの先生の舵からの差
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

R = Path(os.environ.get('DART_REPORT', '/home/jetson/jetformer_reports/sim_dart_20261014'))   # 試しのときだけ変える
FIG = R / 'fig'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'
NOISE_SHADE = '#f4e9c4'
LABEL = {'D1': 'D1 最速の先生 + 乱し', 'D2': 'D2 余裕の線(乱しなし)', 'D3': 'D3 余裕の線 + 乱し', 'A0': 'A0 最速の先生(乱しなし)'}


def jp_font():
    import matplotlib
    from matplotlib import font_manager
    for name in ('Noto Sans CJK JP', 'Noto Sans JP', 'IPAexGothic', 'Noto Serif CJK JP', 'Droid Sans Fallback'):
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


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def shade_noise(ax, t, noisy):
    noisy = np.asarray(noisy, bool)
    f = 0
    while f < len(noisy):
        if noisy[f]:
            g = f
            while g < len(noisy) and noisy[g]:
                g += 1
            ax.axvspan(t[f], t[min(g, len(t) - 1)], color=NOISE_SHADE, lw=0, zorder=0)
            f = g
        else:
            f += 1


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jp_font()
    FIG.mkdir(parents=True, exist_ok=True)
    dev = {d: load(R / 'recording' / d / 'deviation.json') for d in ('D1', 'D2', 'D3')}
    dev['A0'] = load(R / 'recording' / 'A0_floor' / 'deviation.json')
    pred = {d: load(R / 'record_prep' / f'prediction_{d}.json') for d in ('D1', 'D3')}
    made = []

    # 1) ずれの分布(データセットごとに左: 乱しごとの最大のずれ、右: フレームごとのずれ)
    noisy_sets = [d for d in ('D1', 'D3') if dev[d]]
    for d in noisy_sets:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), facecolor=SURFACE)
        ax = axes[0]
        bins = np.arange(0, 32, 1.0)
        ev = 100 * np.array(dev[d]['hist']['ev_max_m'])
        evf = 100 * np.array(dev[d]['hist'].get('fork_ev_max_m') or [])
        ax.axvspan(5, 25, color=GRID, alpha=.45, lw=0, zorder=0)
        ax.hist(ev, bins=bins, histtype='step', lw=2, color=BLUE,
                label=f"乱しなしの走り(開始から)との差 = 手順の決まり(中央値 {np.median(ev):.1f}・90% 点 {np.percentile(ev, 90):.1f} cm)")
        if len(evf):
            ax.hist(evf, bins=bins, histtype='step', lw=2, color=ORANGE,
                    label=f"乱しの始まりから先生だけで走った分かれとの差(中央値 {np.median(evf):.1f}・90% 点 {np.percentile(evf, 90):.1f} cm)")
        p = pred.get(d)
        if p:
            ax.axvline(100 * p['summary']['ev_max_all_p50'], color=INK2, ls='--', lw=1.2,
                       label=f"記録の前の Node の予想の中央値 {100 * p['summary']['ev_max_all_p50']:.1f} cm")
        style(ax, f"{LABEL[d]}: 乱しごとの最大のずれ({len(ev)} 回、始まり〜終わり + 1.5 秒)", '乱しの回数', 'ずれ [cm](R 上の同じ場所で比べた横ずれ)')
        ax.text(24.7, ax.get_ylim()[1] * .5, '目標の範囲 5〜25 cm', color=INK2, fontsize=8, ha='right')
        ax.legend(fontsize=7.3, frameon=False, loc='upper right')
        ax = axes[1]
        bins2 = np.arange(0, 30.5, .5)
        for dd, c in ((d, BLUE), ('D2', AQUA), ('A0', GRAY)):
            if not dev.get(dd):
                continue
            fd = 100 * np.array([x for x in dev[dd]['hist']['frame_dev_m'] if x is not None])
            ax.hist(fd, bins=bins2, histtype='step', lw=1.8 if dd == d else 1.3, color=c, density=True,
                    label=f"{LABEL[dd]}(中央値 {np.median(fd):.1f}・90% 点 {np.percentile(fd, 90):.1f}・最大 {fd.max():.1f} cm)")
        ax.set_yscale('log')
        style(ax, 'フレームごとのずれ(手順の決まり、全フレーム)', '割合(密度、対数)', 'ずれ [cm]')
        ax.legend(fontsize=7.3, frameon=False, loc='upper right', title='乱しなしの記録(D2・A0)= ブラウザと Node の閉ループの差の床', title_fontsize=7.3)
        fig.tight_layout()
        out = FIG / f'record_dev_hist_{d}.png'
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        made.append(str(out))

    # 2) 例の軌跡
    for d in noisy_sets:
        exs = [s for s in dev[d]['scenes'] if s.get('example')]
        if not exs:
            continue
        s = max(exs, key=lambda q: q['events'] and max(e['max_dev_m'] for e in q['events']) or 0)
        ex = s['example']
        t = np.array(ex['t'])
        T = min(len(t), 300)   # 最初の 10 秒
        fig = plt.figure(figsize=(12.5, 12.5), facecolor=SURFACE)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.45])
        ax = fig.add_subplot(gs[0, 0])
        sd = np.array([np.nan if v is None else 100 * v for v in ex['signed_dev']])
        sf = np.array([np.nan if v is None else 100 * v for v in ex.get('fork_signed_dev') or [None] * len(sd)])
        shade_noise(ax, t[:T], ex['noisy'][:T])
        ax.plot(t[:T], sd[:T], color=GRAY, lw=1.2, label='乱しなしの走り(開始から)との差')
        ax.plot(t[:T], sf[:T], color=INK, lw=1.8, label='乱しの始まりからの先生だけの分かれとの差(乱しの窓 + 1.5 秒)')
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax.axhline(0, color=INK2, lw=.8)
        style(ax, f"{s['name']}(最初の 10 秒): 横ずれ(左 +)。黄色の帯 = 乱しを入れたフレーム", 'ずれ [cm]')
        ax = fig.add_subplot(gs[1, 0], sharex=fig.axes[0])
        shade_noise(ax, t[:T], ex['noisy'][:T])
        app = np.array(ex['applied_steer'])
        ax.plot(t[:T], app[:T], color=ORANGE, lw=1.6, label='記録した舵(乱し入り = 実際に動かした操作)')
        if ex.get('label_steer_next'):
            lab = np.array([np.nan if v is None else v for v in ex['label_steer_next']] + [np.nan])
            # labels[t][0] は フレーム t+1 の舵。時刻 t+1 に置く
            tl = np.concatenate([t[1:], [t[-1] + 1 / 30]])
            ax.plot(tl[:T], lab[:T], color=BLUE, lw=1.6, label='付け替えた正解(先生がその状態で出す舵)')
        ax.axhline(0, color=INK2, lw=.8)
        style(ax, '舵: 乱しの間、正解は乱しを打ち消す向き。乱しのあとは線へ戻す舵', '舵 [rad]', '時刻 [秒]')
        ax.legend(fontsize=8, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))
        ax = fig.add_subplot(gs[2, 0])
        xz = np.array(ex['xz'])
        cxz = np.array(ex['clean_xz'])
        ax.plot(cxz[:, 0], cxz[:, 1], color=GRAY, lw=1.2, label='乱しなしの先生(Node、同じ開始)')
        ax.plot(xz[:, 0], xz[:, 1], color=INK, lw=1.0, alpha=.8, label='記録(ブラウザ、乱しあり)')
        nz = np.asarray(ex['noisy'], bool)
        ax.scatter(xz[nz, 0], xz[nz, 1], s=6, color=ORANGE, zorder=3, label='乱しを入れたフレーム')
        ax.set_aspect('equal')
        style(ax, '上から見た軌跡(20 秒、約 2.4 周)', 'z [m]', 'x [m]')
        ax.legend(fontsize=7.5, frameon=False, loc='best')
        fig.tight_layout()
        out = FIG / f'record_traj_{d}.png'
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        made.append(str(out))

    # 3) 乱しの始まりにそろえた平均(戻し方)
    for d in noisy_sets:
        ea = dev[d].get('event_aligned') or []
        if not ea:
            continue
        k = np.array([q['k'] for q in ea]) / 30
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor=SURFACE, sharex=True)
        ax = axes[0]
        nzf = np.array([q['noisy_frac'] for q in ea])
        ax.fill_between(k, 0, 1, where=nzf > .5, color=NOISE_SHADE, transform=ax.get_xaxis_transform(), lw=0, zorder=0)
        ax.fill_between(k, 100 * np.array([q['dev_p25'] for q in ea]), 100 * np.array([q['dev_p75'] for q in ea]), color=GRID, lw=0,
                        label='25〜75% 点')
        ax.plot(k, 100 * np.array([q['dev_mean'] for q in ea]), color=INK, lw=2, label='平均')
        ax.axhline(0, color=INK2, lw=.8)
        style(ax, f"{LABEL[d]}: 乱しの始まりにそろえた横ずれ({ea[0]['n']} 回)", 'ずれ [cm](+ = 乱しが押した向き)',
              '乱しの始まりからの時間 [秒](黄色 = 半分以上の乱しがまだ続いている)')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        ax = axes[1]
        ax.fill_between(k, 0, 1, where=nzf > .5, color=NOISE_SHADE, transform=ax.get_xaxis_transform(), lw=0, zorder=0)
        ax.plot(k, [q['applied_minus_fork_mean'] for q in ea], color=ORANGE, lw=2, label='記録した舵(乱し入り)')
        lab = [np.nan if q['label_minus_fork_mean'] is None else q['label_minus_fork_mean'] for q in ea]
        ax.plot(k, lab, color=BLUE, lw=2, label='付け替えた正解(先生の舵)')
        ax.axhline(0, color=INK2, lw=.8)
        style(ax, '舵(分かれ = 乱しが無かったときの先生の舵との差、平均)', '舵の差 [rad](+ = 乱しの向き)', '乱しの始まりからの時間 [秒]')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        fig.tight_layout()
        out = FIG / f'record_recovery_{d}.png'
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        made.append(str(out))
    # 4) すき間(先生の観測の d_min: 車体から壁・物までの最小の距離)の分布
    sets = [(dd, c) for dd, c in (('A0', GRAY), ('D1', BLUE), ('D2', AQUA), ('D3', ORANGE)) if dev.get(dd) and dev[dd]['hist'].get('frame_dmin_m')]
    if sets:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.4), facecolor=SURFACE)
        for dd, c in sets:
            x = np.sort(100 * np.array(dev[dd]['hist']['frame_dmin_m']))
            y = np.arange(1, len(x) + 1) / len(x)
            sm = dev[dd]['summary']
            ax.plot(x, y, color=c, lw=2, label=f"{LABEL[dd]}(10 cm 未満 {100 * sm['frac_frames_dmin_lt_10cm']:.1f}%・"
                                              f"最初の直線の最小 中央値 {100 * (sm.get('first_straight_min_dmin_p50_m') or 0):.1f} cm)")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, .35)
        style(ax, 'すき間(車体から壁・物まで)の累積分布: 下の端(近いフレーム)', 'その値以下のフレームの割合', 'すき間 [cm]')
        ax.legend(fontsize=7.5, frameon=False, loc='upper left')
        fig.tight_layout()
        out = FIG / 'record_clearance.png'
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        made.append(str(out))
    # 5) 線の比べ(上から): 最速の先生(乱しなし、Node)と余裕の線(D2 の記録)。同じ開始(train-00)
    world_p = Path('/home/jetson/jetformer/experiments/20261013_sim_rl_bestlap/sim/world-v3.json')   # 読むだけ(ベストラップの写し)
    ex1 = next((s for s in (dev.get('D1') or {}).get('scenes', []) if s.get('example') and s['name'].endswith('train-00')), None)
    ex2 = next((s for s in (dev.get('D2') or {}).get('scenes', []) if s.get('example') and s['name'].endswith('train-00')), None)
    if ex1 and ex2 and world_p.exists():
        world = json.loads(world_p.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), facecolor=SURFACE, gridspec_kw={'width_ratios': [1.45, 1]})
        for ax, zoom in zip(axes, (False, True)):
            for seg in world['segments']:
                ax.plot([seg['a'][0], seg['b'][0]], [seg['a'][1], seg['b'][1]], color='#9b4a3c', lw=1.4, alpha=.85)
            for dsk in world['disks']:
                ax.add_patch(plt.Circle((dsk['x'], dsk['z']), max(dsk.get('radius', .02), .02), color='#6f6e69', alpha=.55, lw=0))
            a = np.array(ex1['example']['clean_xz'])
            b = np.array(ex2['example']['xz'])
            n = 300 if not zoom else 60
            ax.plot(a[:n, 0], a[:n, 1], color=BLUE, lw=2, label='最速の先生(A0 の線、Node・乱しなし)')
            ax.plot(b[:n, 0], b[:n, 1], color=AQUA, lw=2, label='余裕の線の先生(D2 の記録、ブラウザ)')
            ax.set_aspect('equal')
            if zoom:
                ax.set_xlim(-1.2, 2.9)
                ax.set_ylim(1.3, 2.95)
                style(ax, '最初の直線(S-R がぶつかった所)、最初の 2 秒', 'z [m]', 'x [m]')
            else:
                style(ax, '上から見た線(最初の 10 秒、train-00 の開始)。茶 = 壁(レール)、灰 = 柱', 'z [m]', 'x [m]')
                ax.legend(fontsize=8, frameon=False, loc='center')
        fig.tight_layout()
        out = FIG / 'record_lines.png'
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        made.append(str(out))
    print(json.dumps({'made': made}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
